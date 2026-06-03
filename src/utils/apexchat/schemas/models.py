"""
Pydantic v2 schemas for structured outputs, API requests/responses,
and internal data models.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ── Routing Enums ─────────────────────────────────────────────────────────────

class ToolType(str, Enum):
    """
    Enumeration of available tools in the workflow.

    Adding a new tool:
    1. Add its identifier here
    2. Implement the tool in app/tools/
    3. Register it in app/tools/__init__.py TOOL_REGISTRY
    4. Update the orchestrator system prompt in workflow.py
    """
    GENERAL = "general"
    NAVIGATION = "navigation"
    WEB_SEARCH = "web_search"
    RAG = "rag"
    MEMORY = "memory"
    REPORT_GENERATION = "report_generation"
    DASHBOARD = "dashboard"
    SOURCE_ANALYSIS = "source_analysis"
    BOOKING = "booking"
    RESCHEDULING = "rescheduling"
    DATA_VIZ = "data_viz"


class RoutingConfidence(str, Enum):
    """Confidence level of the routing decision."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── Orchestrator Structured Output ────────────────────────────────────────────
class RoutingDecision(BaseModel):
    """
    Structured output from the orchestrator LLM.
    Used with .with_structured_output() for reliable parsing.
    """
    tool: ToolType = Field(
        description="The tool to route this request to"
    )
    confidence: RoutingConfidence = Field(
        description="Confidence level of this routing decision"
    )
    reasoning: str = Field(
        description="Brief explanation of why this tool was selected",
        max_length=500,
    )
    requires_clarification: bool = Field(
        default=False,
        description="Whether the request needs clarification before processing"
    )
    clarification_prompt: str | None = Field(
        default=None,
        description="If clarification needed, what to ask the user",
    )




# ── Booking State Models ──────────────────────────────────────────────────────────

class BookingStep(str, Enum):
    """
    States of the multi-turn booking conversation.

    Order is meaningful — `BOOKING_STEP_ORDER` in tools/booking/machine.py
    defines the only legal forward progression.
    """
    IDLE = "idle"
    WALLET_NEEDED = "wallet_needed"
    AWAITING_LEVEL = "awaiting_level"
    AWAITING_SUBJECT = "awaiting_subject"
    AWAITING_CURRICULUM = "awaiting_curriculum"
    AWAITING_TEACHER = "awaiting_teacher"
    AWAITING_SLOT = "awaiting_slot"
    AWAITING_DESCRIPTION = "awaiting_description"
    AWAITING_INVITEES = "awaiting_invitees"
    AWAITING_MATERIAL = "awaiting_material"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"


class BookingInvitee(BaseModel):
    email: str
    firstName: str
    lastName: str


class BookingTransitionEvent(BaseModel):
    """One audit-log entry recording a state-machine transition."""
    ts: datetime = Field(default_factory=datetime.utcnow)
    from_step: BookingStep
    to_step: BookingStep
    trigger: str
    detail: dict[str, Any] = Field(default_factory=dict)


class BookingState(BaseModel):
    """
    Persistent booking-flow state, stored inside WorkflowState across turns.

    All forward progression goes through tools/booking/machine.py — never
    mutate `step` directly outside that module.
    """
    step: BookingStep = BookingStep.IDLE

    wallet_ok: bool = False

    level_id: str | None = None
    level_name: str | None = None
    requires_curriculum: bool = True

    subject_id: str | None = None
    subject_name: str | None = None

    curriculum_code: str | None = None

    tutor_id: str | None = None
    tutor_name: str | None = None

    date: str | None = None
    time_from: str | None = None
    time_to: str | None = None

    description: str | None = None

    invitees: list[BookingInvitee] = Field(default_factory=list)

    # Flipped to True when the frontend sends `material_uploaded: true` on the
    # chat request after its file upload succeeds. Sticky once set — the
    # booking flow only reads it from then on.
    material_uploaded: bool = False

    # Soft slot reservation: ConfirmationStep re-validates against tutor
    # availability before emitting the contract regardless of this value.
    slot_lock_expires_at: datetime | None = None

    cached_levels: list[dict] = Field(default_factory=list)
    cached_subjects: list[dict] = Field(default_factory=list)
    cached_curricula: list[str] = Field(default_factory=list)
    cached_tutors: list[dict] = Field(default_factory=list)
    cached_slots: list[dict] = Field(default_factory=list)

    last_step_completed: BookingStep | None = None
    step_history: list[BookingTransitionEvent] = Field(default_factory=list)

    # Transient: hints extracted from a multi-field user message (e.g. "secondary,
    # English, French curriculum"). Each upcoming step pops its own key, validates
    # against cached options, and either applies it or stops to present options.
    # Cleared on cancel / completion via `machine.reset()`.
    pending_hints: dict[str, Any] = Field(default_factory=dict)

    # Transient: when the user named a time up-front ("find an English teacher at
    # 1pm"), the teacher list is filtered to tutors with a slot at that time.
    # `tutor_time_filter` holds the time (HH:mm UTC) the list is filtered to;
    # `tutor_time_missed` holds a requested time that matched NO tutor (so we
    # fall back to the full list and tell the user). Both cleared by reset().
    tutor_time_filter: str | None = None
    tutor_time_missed: str | None = None


# ── Rescheduling State Models ─────────────────────────────────────────────────

class ReschedulingStep(str, Enum):
    """States of the admin rescheduling conversation."""
    IDLE = "idle"
    COLLECTING = "collecting"
    AWAITING_SLOT_SELECTION = "awaiting_slot_selection"  # Slots fetched, user picking one
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"


class ReschedulingState(BaseModel):
    """
    Persistent rescheduling-flow state, stored inside WorkflowState across turns.
    Available to both students and admins.
    """
    step: ReschedulingStep = ReschedulingStep.IDLE

    # All three fields are required to build the contract
    course_id: str | None = None   # MongoDB ObjectId of the session to reschedule
    date: str | None = None        # YYYY-MM-DD
    time_from: str | None = None   # HH:mm  (24-hour)

    # Populated when the user provides only the session ID so we can present
    # the tutor's available slots for selection instead of asking for a free
    # date/time.  Cleared on cancel / completion.
    cached_slots: list[dict] = Field(default_factory=list)

    # Session IDs that have already been rescheduled in this conversation.
    # Used to block a second reschedule of the same session while still allowing
    # the user to reschedule a different one. Persists across cancels/restarts.
    rescheduled_session_ids: list[str] = Field(default_factory=list)


class ReschedulingStepResult(BaseModel):
    """
    Structured output produced by the rescheduling LLM on each conversational turn.
    """
    response_text: str = Field(
        description=(
            "A brief, friendly message for the current step. "
            "If fields are still missing ask for the next missing one only. "
            "If all three fields (session ID, date, start time) are collected, "
            "show a clear summary and ask the user to type 'yes' to confirm or 'no' to cancel."
        )
    )
    course_id: str | None = Field(
        default=None,
        description=(
            "The session/course ID the user wants to reschedule "
            "(extract exactly as the user wrote it)."
        ),
    )
    date: str | None = Field(
        default=None,
        description=(
            "New session date converted to YYYY-MM-DD format. "
            "Parse natural-language dates such as 'Friday 7 March 2026' → '2026-03-07'."
        ),
    )
    time_from: str | None = Field(
        default=None,
        description=(
            "New start time in HH:mm 24-hour format. "
            "Convert '10:00 am' → '10:00', '2:30 pm' → '14:30'."
        ),
    )
    confirmed: bool = Field(
        default=False,
        description="True when the user explicitly says yes / confirm.",
    )
    cancel: bool = Field(
        default=False,
        description="True when the user explicitly says no / cancel.",
    )



# ── Conversation State ────────────────────────────────────────────────────────
class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationMessage(BaseModel):
    """A single message in the conversation history."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowState(BaseModel):
    """
    LangGraph workflow state - passed between nodes and persisted.

    This is the single source of truth for a conversation's execution state.
    """
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    request_id: str = Field(default_factory=lambda: str(uuid4()))

    # User identity — set by the API layer on every request
    user_id: str = ""

    # Current request
    user_message: str = ""
    file_base64: str | None = None

    # Screen context sent by the front-end (used to validate KG page access)
    object_id: str | None = None

    # Raw list of payload IDs parsed from the client ``sources`` field.
    # Set by the API layer immediately — before any DB call — so it is always
    # present as a reliable routing signal even if the fetch later fails.
    pending_source_ids: list[int] = Field(default_factory=list)

    # Payloads fetched from Apexchat_payload_store once source_route_node runs.
    # Each item: {"id": <int>, "payload": <dict>}
    injected_sources: list[dict[str, Any]] | None = None
    
    # Conversation history (maintained across turns)
    conversation_history: list[ConversationMessage] = Field(default_factory=list)

    # Routing state
    routing_decision: RoutingDecision | None = None

    # Tool execution
    tool_response: str = ""
    tool_execution_time_ms: float = 0.0

    # Final output
    final_response: str = ""

    # Error tracking
    error: str | None = None
    error_node: str | None = None

    # Execution metadata
    nodes_visited: list[str] = Field(default_factory=list)
    total_execution_time_ms: float = 0.0

    # ── Dashboard Tool state (persisted across multi-turn flows) ──────────────
    # Keyed by user_id (session_id used as user_id in this project)
    dashboard_sessions: dict[str, Any] = Field(default_factory=dict)

    # Last completed dashboard JSON config (for the API response)
    dashboard_config: dict[str, Any] | None = None

    # Raw HTML returned from the dashboard renderer API
    html_dashboard: str | None = None
    
    # ── Navigation Tool state ─────────────────────────────────────────────────
    # Set when a navigation request is successfully resolved.
    # Schema: {"screen_name": str, "object_id": str}
    # The front-end uses object_id to trigger the actual screen transition.
    nav_data: dict[str, Any] | None = None

    web_search_results: dict[str, Any] | None = None
   
    # RAG tool state
    rag_sessions: dict[str, Any] = Field(default_factory=dict)
    rag_results: dict[str, Any] | None = None

    # ── Report Generation Tool state ──────────────────────────────────────────
    # Populated by ReportGenerationTool on every invocation.
    # Schema on success:  {"success": True,  "excel_file": ..., "pdf_file": ..., ...}
    # Schema on failure:  {"success": False, "error": "...", ...}
    report_generation_results: dict[str, Any] | None = None

    # ── Booking Tool state (persisted across multi-turn booking flow) ───────────────
    user_id: str | None = None
    booking_state: BookingState = Field(default_factory=BookingState)

    # Populated when the user confirms a booking (emitted to the API response)
    booking_contract: dict[str, Any] | None = None
    want_to_nav: str | None = None

    # ── Rescheduling Tool state (persisted across multi-turn rescheduling flow) ─
    rescheduling_state: ReschedulingState = Field(default_factory=ReschedulingState)

    # Populated when the admin confirms a reschedule (emitted to the API response)
    rescheduling_contract: dict[str, Any] | None = None

    # ── Data Viz Tool state ──────────────────────────────────────────────────
    # Populated by DataVizTool with chart specs, pipelines, and result metadata.
    data_viz_results: dict[str, Any] | None = None

    class Config:
        arbitrary_types_allowed = True


# ── Dashboard Schemas ─────────────────────────────────────────────────────────

class DashboardChartSchema(BaseModel):
    """Structured output schema for a single chart inside a parsed dashboard request."""
    purpose: str = Field(description="Detailed, self-contained description of what data the chart should show.")
    title: str = Field(default="", description="Short human-readable chart title (2-5 words).")
    description: str = Field(default="", description="1-2 sentence description of what the chart displays.")
    chart_type: str = Field(default="bar", description="One of: bar, line, scatter, pie, table, area, donut, heatmap.")
    screen: str = Field(default="", description="Standardised system screen/module name, or empty string if none mentioned.")
    sql_query: str | None = Field(default=None, description="User-supplied raw SQL query. Leave null when absent.")


class DashboardRequestSchema(BaseModel):
    """Structured output schema for the full parsed dashboard creation request."""

    class DashboardMeta(BaseModel):
        title: str = Field(default="", description="Dashboard title.")
        description: str = Field(default="", description="Short dashboard description.")
        theme: str = Field(default="light", description="'light' or 'dark'.")

    dashboard: DashboardMeta = Field(default_factory=DashboardMeta)
    charts: list[DashboardChartSchema] = Field(default_factory=list)


# ── API Request/Response Schemas ──────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Incoming chat request from the client."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The user's message",
        examples=["hi", "how are you?", "my name is Alice"],
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session ID for conversation continuity. "
                    "If not provided, a new session is created.",
    )
    user_id: str | None = Field(
        default=None,
        description="Required user identifier. Conversation is blocked if not provided.",
    )


    model_config = {
        "json_schema_extra": {
            "examples": [
                {"message": "Hello, how are you?"},
                {"message": "My name is Alice", "session_id": "abc-123"},
                {
                    "message": "analyze this regulatory document",
                    "file_base64": "JVBERi0xLjQK...",
                    "object_id": "ScreenBuilder_0380010105_SB",
                },
            ]
        }
    }
    
    file_base64: str | None = Field(
        default=None,
        description="Optional base64-encoded PDF file for tools that require document input (e.g. RegAI, RAG).",
    )
    

    object_id: str | None = Field(
        default=None,
        description="Screen/context identifier sent by the front-end. "
                    "Used to validate that the user is on the correct page "
                    "for the requested operation (e.g. 67986 for wallet KG, "
                    "68199 for transactions/branch KG).",
    )

    stream: bool = Field(
        default=False,
        description=(
            "When true the server opens an SSE connection and streams structured "
            "log events in real time while the workflow executes. The final "
            "ChatResponse is delivered as the last SSE event with type='response'. "
            "When false (default) the endpoint behaves as a normal JSON POST."
        ),
    )

    sources: str | None = Field(
        default=None,
        description=(
            "Comma-separated row IDs from ``Apexchat_payload_store`` to inject as "
            "context into the LLM for this request.  Example: ``\"132,136\"``. "
            "When provided, the corresponding payloads are fetched and prepended "
            "to the user message before the workflow executes."
        ),
        examples=["132,136", "7"],
    )

    material_uploaded: bool | None = Field(
        default=None,
        description=(
            "Set to true on the next chat call after the frontend's file upload "
            "succeeds while ``booking_step == 'awaiting_material'``. Sticky on "
            "the session — only needs to be sent once; subsequent turns can "
            "omit it. Has no effect outside the booking flow."
        ),
    )


class RoutingInfo(BaseModel):
    """Routing decision details exposed in API response."""
    tool_used: ToolType
    confidence: RoutingConfidence
    reasoning: str


class ChatResponse(BaseModel):
    """Response returned to the client."""
    session_id: str
    request_id: str
    response: str
    routing: RoutingInfo
    execution_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # When a recoverable error occurs the workflow still returns a user-friendly
    # apology in `response`.  `error_detail` carries structured diagnostic info
    # for callers that want to handle it programmatically (logging, retry logic).
    # It is None on successful requests.
    error_detail: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Populated only when the workflow encountered a recoverable error. "
            "Contains 'error_code', 'error_node', and 'message' keys. "
            "The `response` field always contains the user-facing apology text."
        ),
    )
    # Optional dashboard output — only present when the dashboard tool runs
    dashboard_config: dict[str, Any] | None = None
    html_dashboard: str | None = None
    
    # Optional Navigation output — only present when the navigation tool runs
    # Schema: {"screen_name": str, "object_id": str}
    nav_data: dict[str, Any] | None = None
    web_search_results: dict[str, Any] | None = None
    # Optional Report Generation output — only present when the report_generation tool runs
    report_generation_results: dict[str, Any] | None = None
    # IDs of Apexchat_payload_store rows that were fetched and injected as context
    # for this request.  Empty list when no sources were provided.
    injected_source_ids: list[int] = Field(default_factory=list)
    booking_contract: dict[str, Any] | None = None
    booking_step: BookingStep | None = None
    want_to_nav: str | None = None
    rescheduling_contract: dict[str, Any] | None = None
    data_viz_results: dict[str, Any] | None = None


class HealthStatus(BaseModel):
    """Health check response."""
    status: str
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReadinessStatus(BaseModel):
    """Readiness probe response."""
    ready: bool
    checks: dict[str, bool]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str
    message: str
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    
    
# ── Memory Schemas ───────────────────────────────────────────────────────────

class DetectedFact(BaseModel):
    """A fact detected in a user message."""
    fact_text: str = Field(description="The extracted fact text")
    fact_type: str = Field(description="Type of fact: profile, preference, event, goal, habit")
    confidence: float = Field(description="Confidence score (0-1) of the fact detection")


class DetectedFactList(BaseModel):
    """List of detected facts from a message."""
    facts: list[DetectedFact] = Field(default_factory=list)


class ReformulatedFact(BaseModel):
    """Reformulated fact for standardized storage."""
    canonical_text: str = Field(description="Standardized fact text starting with 'User'")


class ContradictionCheck(BaseModel):
    """Result of checking for contradictions between facts."""
    action: str = Field(description="Action to take: UPDATE, IGNORE, or APPEND")
    reason: str = Field(description="Explanation of the relationship between facts")
    
    
    
    
class NormalizedQuery(BaseModel):
    """Normalized query for semantic search."""
    normalized: str = Field(description="Statement-form query optimized for fact retrieval")


class ContextSummary(BaseModel):
    """Summary of recent conversation context."""
    summary: str = Field(description="1-2 sentence summary of the conversation")


class MemoryAnswer(BaseModel):
    """Grounded answer from memory retrieval."""
    answer: str = Field(description="Natural-language answer based on stored facts")


class KnowledgeExplanation(BaseModel):
    """Personal explanation of what the system knows about the user."""
    explanation: str = Field(description="Warm, conversational summary of user knowledge")

# ── Payload Store Schemas ─────────────────────────────────────────────────────

class MetadataEnrichment(BaseModel):
    """
    Structured output from the LLM that generates a human-readable title and
    description for a stored response payload.
    Used with ``.with_structured_output()`` so the model always returns both fields.
    """
    data_title: str = Field(
        description=(
            "Short descriptive title summarising what the payload represents. "
            "Must not exceed 255 characters. "
            "Do not use generic words like 'data', 'json', or 'payload'."
        ),
        max_length=255,
    )
    data_description: str = Field(
        description=(
            "Brief description explaining what the payload contains and its purpose. "
            "Must not exceed 255 characters. "
            "Do not use generic words like 'data', 'json', or 'payload'."
        ),
        max_length=255,
    )


class PayloadStoreRequest(BaseModel):
    """Request body for the /payload_store endpoint."""
    user_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the user whose payload is being stored.",
    )
    session_id: str = Field(
        ...,
        min_length=1,
        description="Session that produced the payload.",
    )
    payloads: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of arbitrary JSON objects to persist.",
    )


class PayloadStoreResponse(BaseModel):
    """Response returned by the /payload_store endpoint on success."""
    status: str = "success"
    session_id: str
    timestamp: str


# ── Data Viz Run Schemas ──────────────────────────────────────────────────────

class DataVizRunRequest(BaseModel):
    """
    Payload for POST /api/v1/data_viz/run.

    The frontend hits this endpoint on its refresh interval to execute a
    previously-generated pipeline and get fresh rows.  Admin-only.
    """
    user_id: str = Field(
        ...,
        min_length=1,
        description=(
            "Caller identifier — may be a raw user id OR a JWT.  "
            "The same admin gate as the chat tool applies."
        ),
    )
    collection: str = Field(
        ...,
        min_length=1,
        description="MongoDB collection name (must match a known collection).",
    )
    pipeline: List[Dict[str, Any]] = Field(
        ...,
        description="MongoDB aggregation pipeline as a list of stage objects.",
    )
    limit: Optional[int] = Field(
        default=5000,
        ge=1,
        le=50000,
        description="Hard upper bound on returned rows (safety cap).",
    )
    enrichment: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=(
            "Optional ID-to-label resolution hints carried over from the "
            "data_viz tool's chart spec.  Each entry is "
            "``{column, lookup_collection, label_field}``.  After the pipeline "
            "executes, columns whose value still looks like an ObjectId are "
            "swapped in-place for the referenced collection's label field."
        ),
    )


class DataVizRunResponse(BaseModel):
    """Response from POST /api/v1/data_viz/run."""
    success: bool
    collection: str
    row_count: int
    data: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
