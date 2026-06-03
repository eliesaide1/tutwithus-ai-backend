"""
LangGraph workflow orchestration.

Implements a state-machine workflow:

  [START] → [orchestrate] → [execute_tool] → [finalize] → [END]

The orchestrator LLM analyzes the request and produces a RoutingDecision
using structured output. The appropriate tool then executes, and the result
is finalized into the workflow state.

example tool workflow:
User query
    └─► Orchestrator LLM (routes to geo_report)
            └─► GeoReportTool.execute()
                    └─► GeoReportEngine.run()
                            ├─► LLM extracts branch_name from query
                            ├─► POST https://...vdfs_geo_report  {branch_name, table_id: null, region: null}
                            └─► Returns geo_data → state.geo_report_results → ChatResponse.geo_report_results
"""

import json
import time
from functools import lru_cache

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
import asyncio

import re
from utils.config import *
from utils.apexchat.core.llm import get_orchestrator_client
from utils.apexchat.schemas.models import (
    BookingStep,
    ConversationMessage,
    MessageRole,
    ReschedulingStep,
    RoutingDecision,
    RoutingConfidence,
    ToolType,
    WorkflowState,
)


class _TopicChangeDecision(BaseModel):
    """Structured output for the topic-change classifier."""
    topic_changed: bool = Field(
        description=(
            "True ONLY if the user's message is clearly unrelated to the active "
            "booking/rescheduling flow (e.g. greetings, unrelated questions, "
            "small talk, knowledge questions, navigation requests). "
            "False if the user is still answering, correcting, confirming, "
            "cancelling, or otherwise engaging with the flow."
        )
    )
    reasoning: str = Field(default="", description="Brief justification.")
from utils.apexchat.tools import get_tool
from utils.apexchat.tools.booking import machine as booking_machine
from utils.apexchat.core.status_stream import emit_status
from utils.apexchat.core.memory import get_memory_system

logger = structlog.get_logger(__name__)


# ── Orchestrator System Prompt ────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM_PROMPT = """You are an intelligent request router for an AI assistant system.
Your sole job is to analyze incoming user messages and route them to the most appropriate tool.

Available tools:
- general: Handles everyday conversational interactions including greetings ("hi", "hello"),
  introductions ("my name is X"), small talk ("how are you?", "what's up?"),
  and any general questions that don't require a specialized tool.
- navigation: Handles ALL screen/module navigation requests including:
  * Opening screens ("open the transactions screen", "go to dashboard", "take me to reports")
  * Navigation intent ("navigate to", "show me", "I want to see", "bring up the X screen")
  * Module access ("open risk module", "go to the client list", "show the payments page")
  Route here whenever the user wants to move to or open a specific screen, page, or module.
- web_search: Handles general knowledge questions and real-time information lookups including:
  * Current events and news ("what happened today in Lebanon?", "latest tech news")
  * Factual questions ("what is the population of Beirut?", "who is the president of France?")
  * Prices and market data ("what is the price of gold today?", "Bitcoin price")
  * How-to questions ("how to reset my password?", "how does photosynthesis work?")
  * Any question that requires up-to-date or external information not available in internal tools
  Route here whenever the user asks a general knowledge question, wants current information,
  or asks about anything that is NOT handled by the other specialized tools above.
- rag: Handles ALL questions about the tutoring platform AND document Q&A including:
  * Platform knowledge — ANY question about the platform itself:
    - Tutors / teachers / instructors ("who are the tutors?", "tell me about the teachers", "what tutors do you have?")
      (NOTE: this is ONLY for questions ASKING ABOUT existing tutors. If the user wants to FIND, GET, or be
       matched with a tutor/teacher for a learner — e.g. "find my son an Arabic teacher", "I need a math tutor",
       "I'm looking for a chemistry teacher" — that is a booking intent and MUST route to `booking`, NOT rag.)
    - Subjects / courses ("what subjects do you offer?", "do you have math tutoring?", "available courses")
    - Grade levels ("what levels do you support?", "do you teach high school?")
    - How booking works ("how do I book a session?", "what is the booking process?", "how does it work?")
    - Credits and pricing ("how do credits work?", "what are the bundles?", "how much does it cost?")
    - General info about the website ("tell me about the platform", "what is this website?", "what can I do here?")
  * Document upload: ingest a PDF ("upload this document", "add this file to my documents")
  * Document Q&A: ask questions about uploaded PDFs ("what does the document say about X?", "summarize the PDF")
  Route here for ANY question about the tutoring platform or its services, and for document upload/Q&A.
  Do NOT route here for external/internet information — use web_search for that.
- memory: Handles recall facts such as:
    * Personal information ("what is my name?", "where am I located?", "what do I like?")
    * Preferences ("what are my preferences?", "what do I like?", "what have I told you about me?")
    route here whenever the user asks for information about themselves that they have previously shared with the assistant, or asks for their preferences or any personal facts they have mentioned in past conversations.
- booking: Handles ALL tutoring session booking requests including:
  * Expressing intent to book ("I want to book a session", "book a tutor", "schedule a lesson")
  * Wanting to FIND, GET, or be matched with a tutor/teacher for a learner — these are booking intents,
    NOT knowledge questions ("find me a tutor", "I need a teacher for my son", "find my son a good Arabic teacher",
    "I'm looking for a math teacher", "we need a chemistry tutor", "help me find an English teacher")
  * Any message that occurs WITHIN an active booking conversation (choosing a level, subject,
    teacher, time slot, confirming a booking, etc.)
  * Wallet inquiries related to booking ("I recharged my wallet", "I topped up")
  Route here whenever the user wants to book or schedule a tutoring session,
  or when the conversation history shows an ongoing booking flow.
- data_viz: ADMIN-ONLY natural-language data visualization over the MongoDB database.
  Translates analytical questions into MongoDB aggregations, executes them, and
  returns a labeled chart or multi-chart dashboard (bar / line / pie / scatter / area / table).
  Route here when the
  user asks for ad-hoc analytics, distributions, counts, sums, averages, time-series,
  rankings, dashboards, or any "show me a chart of..." style question that requires
  querying the live database. Examples:
  * "show me a bar chart of bookings per subject"
  * "plot revenue by month for the last year"
  * "how many active users do we have per level?"
  * "visualise the top 10 tutors by session count"
  * "give me a pie chart of users by role"
  Do NOT confuse with `dashboard` (which builds saved multi-chart dashboards via SQL)
  — data_viz is for one-off, ad-hoc MongoDB analytics that can be answered with
  one or more labeled charts.
- rescheduling: Handles ALL tutoring session rescheduling requests for any user (student or admin).
  Only reschedules the date and time of an existing session — nothing else.
  Route here when:
  * A user expresses intent to reschedule ("reschedule my session", "I want to change the time
    of my booking", "move my session to another date", "reschedule session <ID>")
  * A user provides a session ID along with a new date and/or new time in the same message
    (e.g. "reschedule session 64f1a2b3... to Friday 7 March from 10:00 to 11:00")
  * Any message that occurs WITHIN an active rescheduling conversation (providing or correcting
    the session ID, new date, new start/end time, or confirming/cancelling the reschedule)
  Do NOT route here for brand-new bookings — use the booking tool for those.


Instructions:
- Analyze the user's message carefully
- Select the most appropriate tool
- Provide your confidence level (high/medium/low)
- Give brief reasoning for your decision
- Only set requires_clarification=true if the message is completely ambiguous

Default to the "general" tool for any conversational, unclear, or broad requests."""

# ── Flow Lock Helper ─────────────────────────────────────────────────────────

_BOOKING_TERMINAL = {BookingStep.IDLE, BookingStep.COMPLETED}
_RESCHEDULING_TERMINAL = {ReschedulingStep.IDLE, ReschedulingStep.COMPLETED}

# Phrases that signal the user wants to START a booking while another flow is active
_BOOKING_ESCAPE_PHRASES = (
    "book a", "book new", "book another", "new booking", "make a booking",
    "i want to book", "i'd like to book", "i would like to book",
    "can i book", "want to book", "start a booking", "start booking",
    "schedule a new session", "schedule a new class", "schedule a new lesson",
    "book a session", "book a class", "book a lesson",
)

# Phrases that signal the user wants to reschedule while booking flow is active
_RESCHEDULING_ESCAPE_PHRASES = (
    "reschedule", "change my session", "move my session",
    "change my appointment", "change the booking",
)

# ── Deterministic booking-intent guard ────────────────────────────────────────
# gpt-4o-mini frequently misroutes "find/get/need a tutor for my son" to `rag`
# because the sentence mentions "teacher"/"tutor". But an ACTION verb
# (find/get/need/looking for/book/hire) applied to a tutor/teacher is an
# unambiguous NEW booking intent, so we force the booking tool and skip the LLM.
#
# We deliberately do NOT match interrogative/how-to questions ("who are the
# tutors?", "how do I book?", "tell me about the teachers") — those are genuine
# platform-knowledge questions and must keep flowing to the LLM router (→ rag).
_BOOKING_INTENT_RE = re.compile(
    r"\b(find|get|match|need|hire|arrange|book|looking for|want)\b"
    r"(?:(?!\bhow\b)[^.?!]){0,60}?"
    r"\b(tutor|teacher|instructor)\b",
    re.IGNORECASE,
)
_RAG_QUESTION_PREFIX_RE = re.compile(
    r"^\s*(who|what|which|where|when|tell me|do you|does|is there|are there|"
    r"how\s+(do|does|can|much|many))\b",
    re.IGNORECASE,
)


def _looks_like_new_booking_intent(msg: str) -> bool:
    """True for explicit 'find/get/need a tutor' phrasing, False for knowledge questions."""
    if not msg or not msg.strip():
        return False
    if _RAG_QUESTION_PREFIX_RE.search(msg):
        return False
    return bool(_BOOKING_INTENT_RE.search(msg))


def _cancel_active_flow(state: WorkflowState, locked_tool: ToolType) -> None:
    """Silently reset whichever multi-turn flow is currently active."""
    if locked_tool == ToolType.RESCHEDULING:
        rs = state.rescheduling_state
        rs.step = ReschedulingStep.IDLE
        rs.course_id = None
        rs.date = None
        rs.time_from = None
        rs.cached_slots = []
    elif locked_tool == ToolType.BOOKING:
        booking_machine.reset(state.booking_state)


# Prompt for the lightweight topic-change classifier. We err on the side of
# staying in-flow: only flag a topic change when the message is clearly off
# the booking/rescheduling subject.
_TOPIC_CHANGE_SYSTEM_PROMPT = """You decide whether a user has changed the subject mid-flow.

The user is currently inside a multi-turn {flow_name} conversation with an AI assistant. \
At each turn, the assistant either asks the user for one of the {flow_name} fields, or \
shows a summary asking for confirmation.

Set topic_changed = TRUE only when the user's message is CLEARLY unrelated to \
{flow_name} — for example:
  - Greetings or small talk ("hi", "how are you", "thanks")
  - Unrelated knowledge / web questions ("what's the weather", "who is X")
  - Requests to open a different screen / module
  - Questions about the platform itself (pricing, tutors list, how it works)
  - Personal-memory questions ("what is my name?")
  - Any topic that has nothing to do with {flow_name}

Set topic_changed = FALSE when the user is still engaging with the {flow_name} flow:
  - Providing or correcting a value (level, subject, teacher, date, time, session ID, etc.)
  - Confirming or cancelling the flow ("yes", "no", "cancel", "confirm", "ok")
  - Asking a clarifying question about the current step (e.g. "what subjects are available?")
  - Picking a slot number, choosing from a list, or rephrasing a previous answer
  - Empty / unclear short replies that may still belong to the flow

When in doubt, prefer FALSE — staying in-flow is the safe default."""


async def _user_changed_subject(state: WorkflowState, locked_tool: ToolType) -> bool:
    """
    LLM-backed classifier: returns True when the user's current message is
    clearly off-topic from the active booking/rescheduling flow.

    Failures default to False — we never want a transient LLM error to drop
    a user out of a half-finished flow.
    """
    msg = (state.user_message or "").strip()
    if not msg:
        return False

    flow_name = "booking" if locked_tool == ToolType.BOOKING else "rescheduling"
    system_prompt = _TOPIC_CHANGE_SYSTEM_PROMPT.format(flow_name=flow_name)

    try:
        orchestrator = get_orchestrator_client()
        structured = orchestrator.with_structured_output(_TopicChangeDecision)
        messages = [SystemMessage(content=system_prompt)]
        # Light context: include the assistant's last turn so the classifier
        # understands what was just asked of the user.
        for m in state.conversation_history[-2:]:
            if m.role == MessageRole.ASSISTANT:
                messages.append(SystemMessage(content=f"[Assistant just said] {m.content}"))
        messages.append(HumanMessage(content=msg))
        decision: _TopicChangeDecision = await structured.ainvoke(messages)
        return bool(decision.topic_changed)
    except Exception as exc:
        logger.warning(
            "Topic-change classifier failed — staying in flow",
            session_id=state.session_id,
            locked_tool=locked_tool.value,
            error=str(exc),
        )
        return False


async def _try_escape_flow(state: WorkflowState, locked_tool: ToolType) -> bool:
    """
    Returns True if the user's message clearly targets a *different* tool than
    the one currently locked, and silently cancels the active flow so the
    orchestrator can re-route the same message to the correct tool.

    Two escape signals are checked, in order:
      1. Hard-coded phrase match for the opposite multi-turn flow (cheap, no LLM).
      2. LLM topic-change classifier — handles the general "user changed the
         subject" case (greetings, unrelated questions, navigation, etc.).
    """
    msg = state.user_message.lower()

    if locked_tool == ToolType.RESCHEDULING:
        if any(phrase in msg for phrase in _BOOKING_ESCAPE_PHRASES):
            _cancel_active_flow(state, locked_tool)
            logger.info(
                "Flow escape: rescheduling cancelled — booking intent detected",
                session_id=state.session_id,
            )
            return True

    if locked_tool == ToolType.BOOKING:
        if any(phrase in msg for phrase in _RESCHEDULING_ESCAPE_PHRASES):
            _cancel_active_flow(state, locked_tool)
            logger.info(
                "Flow escape: booking cancelled — rescheduling intent detected",
                session_id=state.session_id,
            )
            return True

    if await _user_changed_subject(state, locked_tool):
        _cancel_active_flow(state, locked_tool)
        logger.info(
            "Flow escape: subject change detected — active flow cancelled",
            session_id=state.session_id,
            locked_tool=locked_tool.value,
        )
        return True

    return False


def _get_active_flow_tool(state: WorkflowState) -> ToolType | None:
    """
    Returns the tool that owns an in-progress multi-turn flow, or None.

    When a flow is active the orchestrator LLM is bypassed entirely so that
    mid-flow messages (slot picks, confirmations, corrections, etc.) can never
    be misrouted to the opposite tool.  The flow lock is released automatically
    once the tool resets its step to IDLE or COMPLETED (e.g. on cancellation or
    successful completion).
    """
    if state.booking_state.step not in _BOOKING_TERMINAL:
        return ToolType.BOOKING

    if state.rescheduling_state.step not in _RESCHEDULING_TERMINAL:
        return ToolType.RESCHEDULING

    return None


# ── Workflow Node Functions ───────────────────────────────────────────────────

async def orchestrate_node(state: WorkflowState) -> WorkflowState:
    """
    Orchestrator node: analyzes the request and decides which tool to use.

    Uses structured output to guarantee a valid RoutingDecision is returned.
    Falls back to GENERAL tool on any error.
    """
    node_name = "orchestrate"
    state.nodes_visited.append(node_name)
    start_time = time.perf_counter()

    # ── Flow lock: skip LLM routing when a multi-turn flow is active ─────────
    locked_tool = _get_active_flow_tool(state)
    if locked_tool and not await _try_escape_flow(state, locked_tool):
        state.routing_decision = RoutingDecision(
            tool=locked_tool,
            confidence=RoutingConfidence.HIGH,
            reasoning=f"Flow lock: active {locked_tool.value} flow in progress — orchestrator bypassed.",
            requires_clarification=False,
        )
        logger.info(
            "Flow lock active — orchestrator bypassed",
            session_id=state.session_id,
            locked_tool=locked_tool.value,
        )
        return state

    # ── Deterministic booking-intent guard ───────────────────────────────────
    # Force the booking tool for explicit "find/get/need a tutor" phrasing so the
    # LLM router can't misroute it to `rag` just because it mentions "teacher".
    if _looks_like_new_booking_intent(state.user_message):
        state.routing_decision = RoutingDecision(
            tool=ToolType.BOOKING,
            confidence=RoutingConfidence.HIGH,
            reasoning="Deterministic guard: explicit find/get/need-a-tutor booking intent.",
            requires_clarification=False,
        )
        emit_status("routing_complete", tool=ToolType.BOOKING.value, confidence="high")
        logger.info(
            "Booking-intent guard matched — routing to booking",
            session_id=state.session_id,
            message_preview=state.user_message[:80],
        )
        return state

    emit_status("routing_start")
    logger.info(
        "Orchestrating request",
        session_id=state.session_id,
        request_id=state.request_id,
        message_preview=state.user_message[:80],
    )

    orchestrator = get_orchestrator_client()
    structured_llm = orchestrator.with_structured_output(RoutingDecision)

    # Build context-aware messages
    messages = [SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT)]
    for msg in state.conversation_history[-5:]:
        if msg.role == MessageRole.USER:
            messages.append(HumanMessage(content=f"[Previous] {msg.content}"))
    messages.append(HumanMessage(content=state.user_message))

    last_error: Exception | None = None
    last_content: str = ""

    for attempt in range(3):
        try:
            decision: RoutingDecision = await structured_llm.ainvoke(messages)
            state.routing_decision = decision

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            emit_status("routing_complete", tool=decision.tool.value, confidence=decision.confidence.value)
            logger.info(
                "Routing decision made",
                session_id=state.session_id,
                tool=decision.tool.value,
                confidence=decision.confidence.value,
                reasoning=decision.reasoning,
                elapsed_ms=round(elapsed_ms, 2),
            )
            return state

        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            # Capture any partial content from the exception context for fallback parsing
            raw = getattr(e, "input_value", "") or ""
            if raw:
                last_content = raw

            is_rate_limit = "rate_limit" in err_str or "429" in err_str or "eof" in err_str or "json_invalid" in err_str
            if is_rate_limit and attempt < 2:
                wait = 2 ** attempt  # 1s, 2s
                logger.warning(
                    "Orchestration rate-limit/truncated response, retrying",
                    session_id=state.session_id,
                    attempt=attempt + 1,
                    wait_seconds=wait,
                    error=str(e),
                )
                await asyncio.sleep(wait)
                continue

            # Non-retryable error — break immediately
            break

    # ── Fallback: try to salvage tool from partial JSON content ──────────────
    _TOOL_RE = re.compile(r'"tool"\s*:\s*"([^"]+)"')
    _CONF_RE = re.compile(r'"confidence"\s*:\s*"([^"]+)"')

    salvaged_tool: ToolType | None = None
    salvaged_conf: str = "low"

    if last_content:
        if m := _TOOL_RE.search(last_content):
            try:
                salvaged_tool = ToolType(m.group(1))
            except ValueError:
                pass
        if m := _CONF_RE.search(last_content):
            salvaged_conf = m.group(1)

    if salvaged_tool:
        logger.warning(
            "Orchestration failed but salvaged tool from partial response",
            session_id=state.session_id,
            salvaged_tool=salvaged_tool.value,
            error=str(last_error),
        )
        emit_status("routing_fallback", tool=salvaged_tool.value)
        try:
            conf = RoutingConfidence(salvaged_conf)
        except ValueError:
            conf = RoutingConfidence.LOW
        state.routing_decision = RoutingDecision(
            tool=salvaged_tool,
            confidence=conf,
            reasoning="Salvaged from partial rate-limited response",
            requires_clarification=False,
        )
    else:
        logger.error(
            "Orchestration failed, falling back to general tool",
            session_id=state.session_id,
            error=str(last_error),
            error_type=type(last_error).__name__ if last_error else "unknown",
        )
        emit_status("error_routing")
        state.routing_decision = RoutingDecision(
            tool=ToolType.GENERAL,
            confidence=RoutingConfidence.LOW,
            reasoning="Fallback due to orchestration error",
            requires_clarification=False,
        )

    return state

async def execute_tool_node(state: WorkflowState) -> WorkflowState:
    """
    Tool execution node: runs the selected tool and captures its response.
    """
    node_name = "execute_tool"
    state.nodes_visited.append(node_name)
    start_time = time.perf_counter()

    tool_type = state.routing_decision.tool
    # Emit a tool-specific status event
    tool_phase = f"tool_{tool_type.value}"
    emit_status(tool_phase, tool=tool_type.value)
    logger.info(
        "Executing tool",
        session_id=state.session_id,
        tool=tool_type.value,
    )

    try:
        tool = get_tool(tool_type)
        response = await tool.execute(state)
        state.tool_response = response
        state.tool_execution_time_ms = (time.perf_counter() - start_time) * 1000

        emit_status("tool_complete", tool=tool_type.value)
        logger.info(
            "Tool execution complete",
            session_id=state.session_id,
            tool=tool_type.value,
            response_length=len(response),
            elapsed_ms=round(state.tool_execution_time_ms, 2),
        )

    except Exception as e:
        logger.error(
            "Tool execution failed",
            session_id=state.session_id,
            tool=tool_type.value,
            error=str(e),
            exc_info=True,
        )
        emit_status("error_tool", tool=tool_type.value)
        state.error = str(e)
        state.error_node = node_name
        state.tool_response = (
            "I'm sorry, I encountered an issue processing your request. "
            "Please try again in a moment."
        )

    return state


async def finalize_node(state: WorkflowState) -> WorkflowState:
    """
    Finalization node: updates conversation history and prepares final response.
    """
    node_name = "finalize"
    state.nodes_visited.append(node_name)

    emit_status("finalizing")
    # Set the final response
    state.final_response = state.tool_response

    # Update conversation history
    state.conversation_history.append(
        ConversationMessage(
            role=MessageRole.USER,
            content=state.user_message,
        )
    )
    state.conversation_history.append(
        ConversationMessage(
            role=MessageRole.ASSISTANT,
            content=state.final_response,
            metadata={
                "tool": state.routing_decision.tool.value if state.routing_decision else "unknown",
                "routing_confidence": state.routing_decision.confidence.value
                if state.routing_decision
                else "unknown",
            },
        )
    )

    # Trim history if needed
    max_history = MAX_CONVERSATION_HISTORY
    if len(state.conversation_history) > max_history:
        state.conversation_history = state.conversation_history[-max_history:]

    logger.info(
        "Workflow finalized",
        session_id=state.session_id,
        nodes_visited=state.nodes_visited,
        history_length=len(state.conversation_history),
        has_error=state.error is not None,
    )

    return state


async def persist_memory_node(state: WorkflowState) -> WorkflowState:
    """
    Memory persistence node: saves the completed turn to the database.
    Runs as a background task so it does NOT block the user's response.
    """
    node_name = "persist_memory"
    state.nodes_visited.append(node_name)

    if not state.user_id:
        logger.warning("persist_memory_node skipped — user_id is empty", session_id=state.session_id)
        return state

    emit_status("memory_saving")
    
    try:
        memory = get_memory_system()
    except RuntimeError:
        logger.info("Memory system not initialised — skipping persistence", session_id=state.session_id)
        return state
    
    try:
        route_type = state.routing_decision.tool.value if state.routing_decision else None

        # FIRE AND FORGET: Dispatch to background task instead of awaiting
        asyncio.create_task(
            memory.persist_conversation_turn(
                student_id=state.user_id,
                user_query=state.user_message,
                assistant_response=state.final_response,
                route_type=route_type,
                session_id=state.session_id,
            )
        )
        
        logger.info(
            "Memory persistence dispatched to background",
            session_id=state.session_id,
            student_id=state.user_id,
        )

    except Exception as exc:
        logger.error(
            "persist_memory_node failed to dispatch",
            session_id=state.session_id,
            error=str(exc),
            exc_info=True,
        )

    return state


async def source_route_node(state: WorkflowState) -> WorkflowState:
    """
    Source routing node: fetches the requested payloads from the DB and
    injects them into the user message before execution reaches the tool.

    This node is only reached when ``state.pending_source_ids`` is non-empty
    (guaranteed by the conditional edge at START).  It:

    1. Fetches payloads for every ID in ``pending_source_ids``.
    2. Formats them as XML-delimited context blocks and prepends them to
       ``state.user_message`` so the LLM sees them as grounded context.
    3. Stores the raw fetched list on ``state.injected_sources`` for the tool.
    4. Sets ``routing_decision = SOURCE_ANALYSIS`` unconditionally — the
       presence of source IDs is sufficient to determine the right tool.
    """
    node_name = "source_route"
    state.nodes_visited.append(node_name)

    source_ids = state.pending_source_ids

    logger.info(
        "Source route: fetching payloads and bypassing orchestrator",
        session_id=state.session_id,
        request_id=state.request_id,
        source_ids=source_ids,
    )

    # ── Fetch payloads ────────────────────────────────────────────────────────
    fetched: list[dict] = []
    try:
        from utils.apexchat.core.payload_store import get_payload_store_service
        service = get_payload_store_service()
        fetched = await service.fetch_sources(source_ids)
    except Exception as exc:
        logger.error(
            "source_route_node: payload fetch failed — proceeding without context",
            session_id=state.session_id,
            source_ids=source_ids,
            error=str(exc),
            exc_info=True,
        )

    # ── Build XML context block and augment user_message ─────────────────────
    if fetched:
        parts: list[str] = []
        for src in fetched:
            payload_str = json.dumps(src["payload"], ensure_ascii=False, indent=2)
            parts.append(f'<source id="{src["id"]}">\n{payload_str}\n</source>')

        context_block = "\n".join(parts)
        state.user_message = (
            f"<injected_sources>\n{context_block}\n</injected_sources>\n\n"
            f"{state.user_message}"
        )
        state.injected_sources = fetched
        logger.info(
            "Source payloads injected into user message",
            session_id=state.session_id,
            source_ids=source_ids,
            sources_found=len(fetched),
        )
    else:
        logger.warning(
            "No payloads found for source IDs — SourceAnalysisTool will answer without context",
            session_id=state.session_id,
            source_ids=source_ids,
        )

    # ── Set routing decision ──────────────────────────────────────────────────
    state.routing_decision = RoutingDecision(
        tool=ToolType.SOURCE_ANALYSIS,
        confidence=RoutingConfidence.HIGH,
        reasoning=(
            f"Request includes {len(source_ids)} source ID(s) {source_ids} — "
            f"routed directly to SourceAnalysisTool."
        ),
        requires_clarification=False,
    )

    return state


def _route_from_start(state) -> str:
    """
    Conditional edge function called at START.

    Checks ``pending_source_ids`` — set by the API layer from the raw request
    string before any DB call, so it is always a reliable signal.

    Handles both Pydantic WorkflowState objects and plain dicts because
    LangGraph's behaviour depends on the version and how ``ainvoke`` is called.

    Returns
    -------
    ``"source_route"``  — when source IDs are present
    ``"orchestrate"``   — for all ordinary turns
    """
    # Handle both WorkflowState (Pydantic) and dict (LangGraph serialised form)
    if isinstance(state, dict):
        pending = state.get("pending_source_ids") or []
    else:
        pending = getattr(state, "pending_source_ids", None) or []

    if pending:
        return "source_route"

    return "orchestrate"


# ── Graph Construction ────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    """
    Construct the LangGraph workflow.

    Graph topology:
        START → orchestrate → execute_tool → finalize → END

    Adding a conditional branch (e.g., for a new tool type) is straightforward:
        graph.add_conditional_edges(
            "orchestrate",
            route_by_tool_type,
            {"general": "execute_general", "search": "execute_search"}
        )
    """
    # Use Pydantic model for LangGraph state
    graph = StateGraph(WorkflowState)

    # Register nodes
    graph.add_node("orchestrate", orchestrate_node)
    graph.add_node("source_route", source_route_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("persist_memory", persist_memory_node)

    # ── Edges ─────────────────────────────────────────────────────────────────
    # START branches at the very first step:
    #   • sources present  → source_route (sets routing decision, skips LLM orchestrator)
    #   • no sources       → orchestrate  (normal LLM routing)
    # Both paths then converge at execute_tool.
    #
    #   START ──[conditional]──► source_route ──┐
    #                        └──► orchestrate   ─┴──► execute_tool ──► finalize ──► persist_memory ──► END
    graph.add_conditional_edges(
        START,
        _route_from_start,
        {
            "source_route": "source_route",
            "orchestrate": "orchestrate",
        },
    )
    graph.add_edge("source_route", "execute_tool")
    graph.add_edge("orchestrate", "execute_tool")
    graph.add_edge("execute_tool", "finalize")
    graph.add_edge("finalize", "persist_memory")
    graph.add_edge("persist_memory", END)

    return graph


@lru_cache(maxsize=1)
def get_compiled_workflow():
    """
    Build and compile the workflow graph (singleton, cached).

    Returns:
        Compiled LangGraph workflow ready for execution
    """
    workflow = build_workflow()
    compiled = workflow.compile()
    logger.info("LangGraph workflow compiled successfully")
    return compiled


async def run_workflow(state: WorkflowState) -> WorkflowState:
    """
    Execute the full workflow for a given state.

    Args:
        state: Initial workflow state with user_message populated

    Returns:
        Final workflow state with response and updated history
    """
    workflow = get_compiled_workflow()
    start_time = time.perf_counter()

    # LangGraph works with dicts; convert to/from Pydantic
    state_dict = state.model_dump()
    result_dict = await workflow.ainvoke(state_dict)

    # Reconstruct state from result
    final_state = WorkflowState(**result_dict)
    final_state.total_execution_time_ms = (time.perf_counter() - start_time) * 1000

    return final_state
