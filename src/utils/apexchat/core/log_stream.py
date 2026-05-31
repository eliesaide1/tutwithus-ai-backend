"""
Per-request SSE log streaming.

Architecture
───────────
A lightweight context-var based mechanism that lets any structlog log call
emit a copy of the log event to an asyncio.Queue tied to the current HTTP
request.  The SSE endpoint drains that queue and pushes events to the client
in real time.

Usage
─────
1.  At the start of a streaming request call ``attach_log_stream()`` to get
    a (token, queue) pair.  The token must be reset when the request ends.
2.  The SSE generator calls ``log_stream_generator(queue)`` — it yields
    ``data: <json>\\n\\n`` chunks.
3.  The structlog processor ``stream_log_processor`` checks the context-var;
    if a queue is attached it enqueues a copy of the event dict (non-blocking).

This is intentionally zero-dependency beyond the stdlib and structlog — no
Redis, no external message broker.
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

# ── Context variable ──────────────────────────────────────────────────────────
# Holds the asyncio.Queue for the current request, or None when not streaming.
_log_queue_var: ContextVar[asyncio.Queue | None] = ContextVar(
    "_log_queue_var", default=None
)

# Maximum number of log events buffered per request before we start dropping.
_MAX_QUEUE_SIZE = 256


# ── Public helpers ────────────────────────────────────────────────────────────

def attach_log_stream() -> tuple[object, asyncio.Queue]:
    """
    Attach a fresh log queue to the current async context.

    Returns
    -------
    token
        The ContextVar token — call ``_log_queue_var.reset(token)`` when done.
    queue
        The asyncio.Queue to pass to ``log_stream_generator``.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE_SIZE)
    token = _log_queue_var.set(queue)
    return token, queue


def detach_log_stream(token: object) -> None:
    """Remove the log queue from the current context."""
    _log_queue_var.reset(token)  # type: ignore[arg-type]


async def signal_stream_done(queue: asyncio.Queue) -> None:
    """Enqueue a sentinel so the SSE generator knows the workflow is finished."""
    await queue.put(None)


async def log_stream_generator(
    queue: asyncio.Queue,
    timeout: float = 120.0,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted log events from *queue*.

    Yields
    ------
    str
        ``data: <json>\\n\\n`` chunks, followed by a final
        ``data: {"type": "done"}\\n\\n`` sentinel.

    Parameters
    ----------
    queue:
        The queue returned by ``attach_log_stream``.
    timeout:
        Maximum seconds to wait for the next event before giving up and
        closing the stream.  Acts as a safety valve against stalled requests.
    """
    # Send an initial ping so the client knows the connection is alive.
    yield _sse_event({"type": "connected", "timestamp": _now()})

    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            yield _sse_event(
                {"type": "error", "message": "Stream timed out", "timestamp": _now()}
            )
            break

        if event is None:
            # Sentinel — workflow finished.
            yield _sse_event({"type": "done", "timestamp": _now()})
            break

        yield _sse_event(event)


# ── Structlog processor ───────────────────────────────────────────────────────

def stream_log_processor(
    _logger: logging.Logger,
    _method: str,
    event_dict: dict,
) -> dict:
    """
    Structlog processor: if a log queue is attached to the current context,
    enqueue a copy of the event dict for SSE delivery (non-blocking put_nowait).

    This processor must be added **before** the final renderer in the
    structlog processor chain so that it sees the fully-populated event dict.
    """
    queue = _log_queue_var.get()
    if queue is not None:
        payload = {
            "type": "log",
            "level": event_dict.get("level", "info"),
            "event": event_dict.get("event", ""),
            "timestamp": event_dict.get("timestamp", _now()),
        }
        # Carry a small set of useful diagnostic fields when present.
        for key in ("session_id", "request_id", "tool", "node", "error"):
            if key in event_dict:
                payload[key] = event_dict[key]

        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            # Drop silently — never block the application thread.
            pass

    return event_dict


# ── Private helpers ───────────────────────────────────────────────────────────

def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
