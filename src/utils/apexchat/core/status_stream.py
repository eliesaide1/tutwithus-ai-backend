"""
Per-request SSE status streaming — user-friendly edition.

Architecture
────────────
Replaces the old ``log_stream.py`` debug-log approach with a curated set of
human-readable status events.  Instead of forwarding raw structlog records,
callers explicitly emit *named phases* through a thin helper API.  Each phase
maps to a pre-written user-facing message defined in ``STATUS_MESSAGES`` below.

All message copy lives in one place (``STATUS_MESSAGES``) so non-engineers can
update the wording without touching any logic.

SSE event shape
───────────────
Every event sent to the client is a JSON object with at minimum:

    {
        "type":     "status" | "connected" | "done" | "error",
        "phase":    "<phase_key>",          # present on type=status
        "message":  "<user-facing text>",   # present on type=status / error
        "timestamp": "<ISO-8601 UTC>"
    }

Optional fields (only when the value is known at emit time):
    "tool":       the tool being used (e.g. "dashboard", "nms")
    "confidence": routing confidence ("high" / "medium" / "low")

Usage
─────
1.  At the start of a streaming request call ``attach_status_stream()`` to get
    a (token, queue) pair.
2.  Anywhere in the workflow call ``emit_status(phase, **extras)`` — no import
    of the queue is required; it is resolved from the context-var.
3.  The SSE generator (``status_stream_generator``) drains the queue and yields
    ``data: <json>\\n\\n`` chunks.
4.  Call ``signal_stream_done(queue)`` when the workflow finishes.
5.  Call ``detach_status_stream(token)`` in the ``finally`` block.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import AsyncGenerator

import structlog

logger = structlog.get_logger(__name__)

# ── Queue context-var ─────────────────────────────────────────────────────────

_status_queue_var: ContextVar[asyncio.Queue | None] = ContextVar(
    "_status_queue_var", default=None
)

_MAX_QUEUE_SIZE = 128


# ── Status message catalogue ──────────────────────────────────────────────────
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  EDIT THIS SECTION to change what users see during processing.              │
# │                                                                             │
# │  Keys are internal phase identifiers — never shown to users.               │
# │  Values are the strings that appear in the chat UI.                        │
# │                                                                             │
# │  You can use {tool} as a placeholder; it is substituted at emit time       │
# │  when the caller supplies tool= to emit_status().                          │
# └─────────────────────────────────────────────────────────────────────────────┘

STATUS_MESSAGES: dict[str, str] = {
    # ── Workflow lifecycle ────────────────────────────────────────────────────
    "request_received":       "Got your request, let me think about it…",
    "session_resolved":       "Session ready.",
    "routing_start":          "Understanding your request…",
    "routing_complete":       "I know exactly what to do — working on it now.",
    "routing_fallback":       "Hmm, let me try a different approach…",
    "source_injection":       "Loading context from your source data…",
    "tool_start":             "Working on it…",
    "tool_complete":          "Done! Putting together your answer.",
    "finalizing":             "Almost there — finalizing your response…",
    "memory_saving":          "Saving this conversation to your history.",
    "workflow_complete":      "All done!",

    # ── Per-tool status messages ───────────────────────────────────────────────
    # General / fallback
    "tool_general":           "Thinking about your question…",

    # Dashboard
    "tool_dashboard":         "Building your dashboard…",
    "tool_dashboard_sql":     "Running the data queries…",
    "tool_dashboard_render":  "Rendering the charts…",


    # Navigation
    "tool_navigation":        "Preparing your navigation…",

  

    # Web search
    "tool_web_search":        "Searching the web for the latest information…",

    # RAG
    "tool_rag":               "Searching your documents…",
    "tool_rag_ingesting":     "Indexing your document — this may take a moment…",
    "tool_rag_retrieving":    "Retrieving relevant passages…",

    # Memory recall
    "tool_memory":            "Recalling what you've shared with me…",

    # Report generation
    "tool_report_generation":         "Generating your report…",
    "tool_report_generation_format":  "Formatting and exporting the document…",

    # Source analysis
    "tool_source_analysis":   "Analyzing the selected data sources…",

    # ── Error states ──────────────────────────────────────────────────────────
    "error_routing":          "I had trouble understanding the request — trying my best.",
    "error_tool":             "Something went wrong while processing. I'll still try to respond.",
    "error_timeout":          "The request is taking longer than expected. Hang tight…",
}

# ── Public helpers ─────────────────────────────────────────────────────────────

def attach_status_stream() -> tuple[object, asyncio.Queue]:
    """
    Attach a fresh status queue to the current async context.

    Returns
    -------
    token
        The ContextVar token — pass to ``detach_status_stream`` when done.
    queue
        The asyncio.Queue to pass to ``status_stream_generator``.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
    token = _status_queue_var.set(queue)
    return token, queue


def detach_status_stream(token: object) -> None:
    """Remove the status queue from the current context."""
    _status_queue_var.reset(token)  # type: ignore[arg-type]


async def signal_stream_done(queue: asyncio.Queue) -> None:
    """Enqueue the sentinel so the SSE generator closes the stream."""
    await queue.put(None)


def emit_status(phase: str, **extras) -> None:
    """
    Enqueue a user-facing status event for the current request (non-blocking).

    Parameters
    ----------
    phase:
        One of the keys defined in ``STATUS_MESSAGES``.  If the key is not
        found, the phase string itself is sent as a fallback message so
        nothing is silently swallowed.
    **extras:
        Optional metadata forwarded to the client as additional JSON fields.
        Typical keys: ``tool``, ``confidence``.  The ``{tool}`` placeholder
        in the message copy is substituted from ``extras.get("tool")``.
    """
    queue = _status_queue_var.get()
    if queue is None:
        return  # Not streaming — no-op

    raw_message = STATUS_MESSAGES.get(phase, phase)

    # Substitute {tool} placeholder if caller provided a tool name
    tool_name = extras.get("tool", "")
    if "{tool}" in raw_message and tool_name:
        raw_message = raw_message.replace("{tool}", tool_name)

    payload: dict = {
        "type": "status",
        "phase": phase,
        "message": raw_message,
        "timestamp": _now(),
    }
    payload.update(extras)

    try:
        queue.put_nowait(payload)
    except asyncio.QueueFull:
        pass  # Never block the workflow thread


async def emit_status_async(phase: str, **extras) -> None:
    """
    Async variant of ``emit_status`` — awaits if the queue is full.

    Prefer ``emit_status`` (sync) in hot paths; use this when you can afford
    to await and want guaranteed delivery even under burst load.
    """
    queue = _status_queue_var.get()
    if queue is None:
        return

    raw_message = STATUS_MESSAGES.get(phase, phase)
    tool_name = extras.get("tool", "")
    if "{tool}" in raw_message and tool_name:
        raw_message = raw_message.replace("{tool}", tool_name)

    payload: dict = {
        "type": "status",
        "phase": phase,
        "message": raw_message,
        "timestamp": _now(),
    }
    payload.update(extras)

    try:
        await asyncio.wait_for(queue.put(payload), timeout=0.1)
    except (asyncio.QueueFull, asyncio.TimeoutError):
        pass


async def status_stream_generator(
    queue: asyncio.Queue,
    timeout: float = 120.0,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted status events from *queue*.

    Yields
    ------
    str
        ``data: <json>\\n\\n`` chunks.  The final chunk is always
        ``data: {"type": "done", ...}\\n\\n``.

    Parameters
    ----------
    queue:
        The queue returned by ``attach_status_stream``.
    timeout:
        Max seconds to wait for the next event.  Guards against stalled tasks.
    """
    # Initial handshake so the client knows the connection is alive
    yield _sse_event({"type": "connected", "message": "Connected.", "timestamp": _now()})

    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            yield _sse_event({
                "type": "error",
                "phase": "error_timeout",
                "message": STATUS_MESSAGES["error_timeout"],
                "timestamp": _now(),
            })
            break

        if event is None:
            # Sentinel — stream is finished
            yield _sse_event({"type": "done", "timestamp": _now()})
            break

        yield _sse_event(event)


# ── Backward-compatibility shim ───────────────────────────────────────────────
# Callers that previously imported from log_stream can import these aliases
# while migration to the new API is in progress.

def attach_log_stream() -> tuple[object, asyncio.Queue]:
    """Alias for ``attach_status_stream`` — kept for backward compatibility."""
    return attach_status_stream()


def detach_log_stream(token: object) -> None:
    """Alias for ``detach_status_stream`` — kept for backward compatibility."""
    detach_status_stream(token)


async def log_stream_generator(
    queue: asyncio.Queue,
    timeout: float = 120.0,
) -> AsyncGenerator[str, None]:
    """Alias for ``status_stream_generator`` — kept for backward compatibility."""
    async for chunk in status_stream_generator(queue, timeout=timeout):
        yield chunk


def stream_log_processor(
    _logger: logging.Logger,
    _method: str,
    event_dict: dict,
) -> dict:
    """
    Structlog processor shim — intentionally a no-op in the new design.

    Raw log records are no longer forwarded to the SSE stream.  Status
    events are emitted explicitly via ``emit_status()`` instead.

    Kept so the structlog processor chain in ``logging.py`` does not need
    to be changed immediately.
    """
    return event_dict

# ── Private helpers ───────────────────────────────────────────────────────────

def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()