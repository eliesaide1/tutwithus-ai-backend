"""
AI Assistant - Production-Ready FastAPI Application with LangGraph Workflow

Also serves as the development server entry point.
For production, use the Dockerfile CMD instead.

Usage:
    python Apexchat_api.py
    python Apexchat_api.py --reload   # auto-reload on file changes

Endpoints:
  POST /api/v1/chat            - Main chat endpoint (JSON or SSE depending on body.stream)
  GET  /api/v1/health          - Liveness probe
  GET  /api/v1/readiness       - Readiness probe
  GET  /api/v1/sessions/{id}   - Session info (dev only)
  POST /api/v1/payload_store   - Store a response payload with LLM-generated metadata

Streaming behaviour
───────────────────
When the client sends ``"stream": true`` the endpoint returns an SSE response
(Content-Type: text/event-stream).  Each event is a JSON object:

  {"type": "connected",  "timestamp": "..."}                    – handshake
  {"type": "log",        "level": "info", "event": "…", …}     – live log lines
  {"type": "response",   "data": {<ChatResponse JSON>}}         – final answer
  {"type": "error",      "message": "…", "request_id": "…"}    – fatal error
  {"type": "done",       "timestamp": "..."}                    – stream closed

When ``"stream": false`` (default) the endpoint behaves as a normal POST that
returns a ``ChatResponse`` JSON object.  This object always contains a
user-friendly ``response`` string; on workflow errors that string is an apology
and the structured ``error_detail`` field carries diagnostic information.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated, AsyncGenerator, Optional

# ── sys.path fix — ensures `src` modules are importable when running directly ─
# Must appear before any project imports so all imports below resolve correctly.
_FILE_DIR  = os.path.dirname(os.path.abspath(__file__))          # .../src/api
_SRC_DIR   = os.path.abspath(os.path.join(_FILE_DIR, ".."))     # .../src
_ROOT_DIR  = os.path.abspath(os.path.join(_SRC_DIR, ".."))      # .../Apexchatv1
_UTILS_DIR = os.path.abspath(os.path.join(_SRC_DIR, "utils"))   # .../src/utils

for _p in (_ROOT_DIR, _SRC_DIR, _UTILS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_extra = os.pathsep.join([_ROOT_DIR, _SRC_DIR, _UTILS_DIR])
_existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = f"{_extra}{os.pathsep}{_existing}" if _existing else _extra

import structlog
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from utils.config import *

from utils.apexchat.core.status_stream import (
    attach_status_stream,
    detach_status_stream,
    status_stream_generator,
    signal_stream_done,
)
from utils.apexchat.core.memory import get_memory_system, init_memory_system
from utils.apexchat.core.memory import ApexchatMemorySystem
from utils.apexchat.core.mongodb import _extract_user_id_str
from utils.apexchat.core.sessions import SessionStore, get_session_store
from utils.apexchat.core.workflow import run_workflow
from utils.apexchat.middleware.request_id import RequestIDMiddleware
from utils.apexchat.schemas.models import (
    ChatRequest,
    ChatResponse,
    DataVizRunRequest,
    DataVizRunResponse,
    ErrorResponse,
    HealthStatus,
    PayloadStoreRequest,
    PayloadStoreResponse,
    ReadinessStatus,
    RoutingConfidence,
    RoutingInfo,
    ToolType,
    WorkflowState,
)

from utils.Mongodb_tools import MONGODB_TOOLS
from utils.apexchat.core.llm import get_general_tool_client
from utils.apexchat.core.memory import get_memory_system, init_memory_system
from utils.apexchat.core.logging import setup_logging

from utils.config import *

# Global memory system instance (initialised in lifespan().)
memory_manager: ApexchatMemorySystem | None = None

setup_logging()
logger = structlog.get_logger(__name__)

# ── User-facing apology messages (no stack traces) ────────────────────────────

_APOLOGY_WORKFLOW = (
    "I'm sorry, something went wrong while processing your request. "
    "Our team has been notified. Please try again in a moment."
)
_APOLOGY_TOOL = (
    "I'm sorry, I ran into an issue completing that action. "
    "Please try again or rephrase your question."
)
_APOLOGY_STREAM = (
    "I'm sorry, an error occurred while streaming the response. "
    "Please try again."
)


# ── Application lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    global memory_manager
    logger.info(
        "Starting AI Assistant",
        version=APP_VERSION,
        environment=ENVIRONMENT,
    )

    # ── DB-backed services initialisation ─────────────────────────────────────
    # Both the memory system and the payload store service share the same
    # MONGODB_TOOLS instance and LLM client, so we initialise them together.
    # The app remains fully runnable without a DB — both services degrade
    # gracefully and return 503 when not configured.
    # if MONGO_DB_HOST and MONGO_DB_PASSWORD:
    try:

        mongodb_tools = MONGODB_TOOLS()
        llm_client = get_general_tool_client()

        init_memory_system(mongodb_tools=mongodb_tools, llm_client=llm_client)
        memory_manager = get_memory_system()
        logger.info("Memory system initialised successfully")

    except Exception as exc:
        # Non-fatal: the app runs without these services if the DB is
        # unavailable. Affected endpoints will return HTTP 503.
        logger.warning(
            "DB-backed service initialisation failed — memory and payload store unavailable",
            error=str(exc),
        )
    # else:
    #     logger.info("DB-backed services skipped — MONGO_DB_HOST or MONGO_DB_PASSWORD not set")

    yield

    logger.info("Shutting down AI Assistant")


# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter()
SessionStoreDep = Annotated[SessionStore, Depends(get_session_store)]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_initial_state(
    body: ChatRequest,
    session_id: str,
    state: WorkflowState,
    request_id: str,
    user_id: str,
) -> WorkflowState:
    state.request_id = request_id
    state.user_message = body.message
    state.file_base64 = body.file_base64
    state.object_id = body.object_id
    extracted_user_id = _extract_user_id_str(user_id)
    state.user_id = extracted_user_id if extracted_user_id else user_id
    state.nodes_visited = []
    state.error = None
    state.error_node = None

    # Sticky flag: only set when the frontend signals an upload succeeded.
    # Never assign body.material_uploaded directly — that would clear the flag
    # on every subsequent turn and rewind ConfirmationStep back to AWAITING_MATERIAL.
    if body.material_uploaded:
        state.booking_state.material_uploaded = True

    # ── Clear per-turn execution metadata ────────────────────────────────────
    # NOTE: We intentionally do NOT clear tool-specific result fields here
    # (kg_result, nms_results, branch_list_results, etc.) because downstream
    # tools depend on them.  For example, pandas_analytics reads state.kg_result
    # to build DataFrames for analysis on the *next* turn.
    #
    # Stale data is stripped from the API *response* instead — see
    # _build_chat_response(), which only includes the current tool's output.
    state.tool_response = ""
    state.final_response = ""
    state.routing_decision = None
    state.tool_execution_time_ms = 0.0
    state.total_execution_time_ms = 0.0

    return state


def _build_chat_response(
    final_state: WorkflowState,
    session_id: str,
    request_id: str,
    elapsed_ms: float,
) -> ChatResponse:
    routing_decision = final_state.routing_decision

    user_response = final_state.final_response
    error_detail = None

    if final_state.error:
        if not user_response or user_response == final_state.error:
            user_response = _APOLOGY_TOOL
        error_detail = {
            "error_code": "tool_execution_error",
            "error_node": final_state.error_node or "unknown",
            "message": final_state.error,
        }

    # ── Only include the current tool's result in the response ───────────────
    # The state object accumulates results across turns for cross-tool deps
    # (e.g. pandas_analytics reads state.kg_result).  But the API response
    # should only carry the output of the tool that ran THIS turn, so the
    # frontend doesn't re-render stale widgets from previous turns.
    current_tool = routing_decision.tool if routing_decision else ToolType.GENERAL

    return ChatResponse(
        session_id=session_id,
        request_id=request_id,
        response=user_response,
        routing=RoutingInfo(
            tool_used=current_tool,
            confidence=routing_decision.confidence if routing_decision else RoutingConfidence.LOW,
            reasoning=routing_decision.reasoning if routing_decision else "N/A",
        ),
        execution_time_ms=round(elapsed_ms, 2),
        error_detail=error_detail,
        dashboard_config=None,
        html_dashboard=None,
        nav_data=final_state.nav_data if current_tool == ToolType.NAVIGATION else None,
        web_search_results=final_state.web_search_results if current_tool == ToolType.WEB_SEARCH else None,
        report_generation_results=final_state.report_generation_results if current_tool == ToolType.REPORT_GENERATION else None,
        injected_source_ids=[s["id"] for s in final_state.injected_sources]
        if final_state.injected_sources
        else [],    
        booking_contract=final_state.booking_contract if current_tool == ToolType.BOOKING else None,
        booking_step=final_state.booking_state.step,
        want_to_nav=final_state.want_to_nav if current_tool == ToolType.BOOKING else None,
        rescheduling_contract=final_state.rescheduling_contract if current_tool == ToolType.RESCHEDULING else None,
        data_viz_results=final_state.data_viz_results if current_tool == ToolType.DATA_VIZ else None,
    )


def _parse_source_ids(sources: str | None) -> list[int]:
    """
    Parse a comma-separated string of integer IDs into a de-duplicated list.
    
    Returns an empty list if *sources* is ``None``, blank, or contains no
    valid integers.  Invalid tokens are skipped with a warning so a single
    malformed ID never blocks the whole request.
    """
    if not sources or not sources.strip():
        return []

    ids: list[int] = []
    for token in sources.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            ids.append(int(token))
        except ValueError:
            logger.warning("Ignoring non-integer source ID", token=token)

    # Preserve order, remove duplicates
    seen: set[int] = set()
    unique: list[int] = []
    for id_ in ids:
        if id_ not in seen:
            seen.add(id_)
            unique.append(id_)

    return unique


def _stamp_source_ids(state: WorkflowState, sources: str | None) -> WorkflowState:
    """
    Parse the ``sources`` string from the request and store the resulting IDs
    on ``state.pending_source_ids``.

    This always overwrites both source fields — even when ``sources`` is null
    or empty — so stale IDs from a previous turn stored in the session never
    bleed into the current turn's routing decision.
    """
    ids = _parse_source_ids(sources)

    # Always reset, whether ids is populated or empty.
    # This is the critical invariant: routing is decided by the *current*
    # request only, never by what a previous session turn left behind.
    state.pending_source_ids = ids
    state.injected_sources = None

    if ids:
        logger.info(
            "Source IDs stamped onto state",
            source_ids=ids,
            session_id=state.session_id,
        )
    else:
        logger.debug(
            "No sources in request — source fields cleared",
            session_id=state.session_id,
        )

    return state



async def _resolve_session(
    body: ChatRequest,
    session_store: SessionStore,
    request_id: str,
) -> tuple[str, WorkflowState]:
    session_id = (body.session_id or "").strip() or None
    state: WorkflowState | None = None

    if session_id:
        state = await session_store.get(session_id)
        if state is None:
            logger.info("Session not found, creating new", provided_session_id=session_id)

    if state is None:
        session_id = session_id or str(uuid.uuid4())
        state = WorkflowState(session_id=session_id)
        logger.info("New session created", session_id=session_id)

    return session_id, state


# ── Chat Endpoint ─────────────────────────────────────────────────────────────

@router.post(
    "/chat",
    summary="Send a message to the AI assistant",
    description=(
        "Processes a user message through the LangGraph workflow. "
        "Maintains conversation context within a session. "
        "Set ``stream=true`` to receive a live SSE log stream instead of "
        "a plain JSON response."
    ),
    responses={
        200: {
            "description": (
                "JSON ChatResponse (stream=false) or SSE stream (stream=true). "
                "Workflow errors are surfaced as a user-friendly apology in the "
                "`response` field — HTTP status is still 200."
            ),
        },
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Chat"],
)
async def chat(
    request: Request,
    body: ChatRequest,
    session_store: SessionStoreDep,
    x_user_id: str = Header(default="", alias="X-User-ID"),
):
    """
    Main chat endpoint.

    Routes to either the standard JSON handler or the SSE streaming handler
    depending on ``body.stream``.
    """
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    user_id = (body.user_id or "").strip() or x_user_id.strip()
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "missing_user_id",
                "message": "user_id is required. Please provide it in the request body.",
                "request_id": request_id,
            },
        )

    if body.stream:
        return await _handle_streaming(body, session_store, user_id, request_id)
    return await _handle_standard(body, session_store, user_id, request_id)


# ── Standard (non-streaming) handler ─────────────────────────────────────────

async def _handle_standard(
    body: ChatRequest,
    session_store: SessionStore,
    user_id: str,
    request_id: str,
) -> ChatResponse:
    session_id, state = await _resolve_session(body, session_store, request_id)
    state = _build_initial_state(body, session_id, state, request_id, user_id)
    state = _stamp_source_ids(state, body.sources)

    logger.info(
        "Processing chat request",
        session_id=session_id,
        request_id=request_id,
        message_preview=body.message[:60],
        history_length=len(state.conversation_history),
    )

    start_time = time.perf_counter()

    try:
        final_state = await run_workflow(state)
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "Workflow execution failed",
            session_id=session_id,
            request_id=request_id,
            error=str(exc),
            exc_info=True,
        )
        return ChatResponse(
            session_id=session_id,
            request_id=request_id,
            response=_APOLOGY_WORKFLOW,
            routing=RoutingInfo(
                tool_used=ToolType.GENERAL,
                confidence=RoutingConfidence.LOW,
                reasoning="Workflow failed before routing could complete",
            ),
            execution_time_ms=round(elapsed_ms, 2),
            error_detail={
                "error_code": "workflow_execution_error",
                "error_node": "run_workflow",
                "message": str(exc),
            },
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    await session_store.set(session_id, final_state)

    chat_response = _build_chat_response(final_state, session_id, request_id, elapsed_ms)

    logger.info(
        "Chat request completed",
        session_id=session_id,
        request_id=request_id,
        tool_used=chat_response.routing.tool_used.value,
        elapsed_ms=round(elapsed_ms, 2),
        has_error=chat_response.error_detail is not None,
    )

    return chat_response


# ── SSE streaming handler ─────────────────────────────────────────────────────

async def _handle_streaming(
    body: ChatRequest,
    session_store: SessionStore,
    user_id: str,
    request_id: str,
) -> StreamingResponse:
    return StreamingResponse(
        _stream_workflow(body, session_store, user_id, request_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "X-Request-ID": request_id,
        },
    )


async def _stream_workflow(
    body: ChatRequest,
    session_store: SessionStore,
    user_id: str,
    request_id: str,
) -> AsyncGenerator[str, None]:
    token, queue = attach_status_stream()

    try:
        session_id, state = await _resolve_session(body, session_store, request_id)
        state = _build_initial_state(body, session_id, state, request_id, user_id)
        state = _stamp_source_ids(state, body.sources)

        logger.info(
            "Processing streaming chat request",
            session_id=session_id,
            request_id=request_id,
            message_preview=body.message[:60],
            history_length=len(state.conversation_history),
        )

        start_time = time.perf_counter()
        result_holder: list[WorkflowState | Exception] = []

        async def _run() -> None:
            try:
                result = await run_workflow(state)
                result_holder.append(result)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Streaming workflow execution failed",
                    session_id=session_id,
                    request_id=request_id,
                    error=str(exc),
                    exc_info=True,
                )
                result_holder.append(exc)
            finally:
                await signal_stream_done(queue)

        workflow_task = asyncio.create_task(_run())

        async for sse_chunk in status_stream_generator(queue):
            yield sse_chunk

        await workflow_task

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if not result_holder:
            yield _sse_json({"type": "error", "message": _APOLOGY_STREAM, "request_id": request_id})
            return

        outcome = result_holder[0]

        if isinstance(outcome, Exception):
            error_response = ChatResponse(
                session_id=session_id,
                request_id=request_id,
                response=_APOLOGY_WORKFLOW,
                routing=RoutingInfo(
                    tool_used=ToolType.GENERAL,
                    confidence=RoutingConfidence.LOW,
                    reasoning="Workflow failed",
                ),
                execution_time_ms=round(elapsed_ms, 2),
                error_detail={
                    "error_code": "workflow_execution_error",
                    "error_node": "run_workflow",
                    "message": str(outcome),
                },
            )
            yield _sse_json({"type": "response", "data": error_response.model_dump(mode="json")})
        else:
            final_state: WorkflowState = outcome
            await session_store.set(session_id, final_state)
            chat_response = _build_chat_response(final_state, session_id, request_id, elapsed_ms)
            logger.info(
                "Streaming chat request completed",
                session_id=session_id,
                request_id=request_id,
                tool_used=chat_response.routing.tool_used.value,
                elapsed_ms=round(elapsed_ms, 2),
                has_error=chat_response.error_detail is not None,
            )
            yield _sse_json({"type": "response", "data": chat_response.model_dump(mode="json")})

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "SSE stream generator failed unexpectedly",
            request_id=request_id,
            error=str(exc),
            exc_info=True,
        )
        yield _sse_json({"type": "error", "message": _APOLOGY_STREAM, "request_id": request_id})
    finally:
        detach_status_stream(token)


def _sse_json(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── Health Probes ─────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthStatus,
    summary="Liveness probe",
    tags=["Observability"],
)
async def health() -> HealthStatus:
    """Kubernetes liveness probe. Returns 200 if the service is running."""
    return HealthStatus(
        status="ok",
        version=APP_VERSION,
        environment=ENVIRONMENT,
    )


@router.get(
    "/readiness",
    response_model=ReadinessStatus,
    summary="Readiness probe",
    tags=["Observability"],
)
async def readiness(session_store: SessionStoreDep) -> ReadinessStatus:
    """Kubernetes readiness probe. Verifies all dependencies are available."""
    checks = {}

    try:
        _ = session_store.session_count
        checks["session_store"] = True
    except Exception:
        checks["session_store"] = False

    ready = all(checks.values())
    return ReadinessStatus(ready=ready, checks=checks)

# ── Application factory ───────────────────────────────────────────────────────

def create_application() -> FastAPI:
    """
    Application factory following 12-factor app principles.

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title=APP_NAME,
        description="Production-ready AI Assistant with LangGraph workflow orchestration",
        version=APP_VERSION,
        docs_url="/docs" if ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if ENVIRONMENT != "production" else None,
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ───────────────────────────────────────────────────────────────
    app.include_router(router, prefix=API_PREFIX)

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", str(uuid.uuid4())),
            },
        )

    # ── Request timing middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        return response

    return app


app = create_application()


# ── Development server entry point ────────────────────────────────────────────

def _run_server():
    """Start the uvicorn development server."""

    print(f"Starting AI Assistant on v{APEX_ENGINES_HOST}:{APEX_CHAT_PORT}")
    print(f"Docs available at: http://localhost:{APEX_CHAT_PORT}/docs")

    uvicorn.run(
        "Apexchat_api:app",
        host=APEX_ENGINES_HOST,
        port=APEX_CHAT_PORT,
        reload= True,
        workers= 100,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    _run_server()