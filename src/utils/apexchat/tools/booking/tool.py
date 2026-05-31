"""
BookingTool — public entry point for the booking flow.

Pure dispatcher: it looks up the handler for the current state, runs it once,
and (rarely) re-enters the loop when the handler advanced cleanly without
needing further user input.

All actual logic — state transitions, validation, presentation, contract
emission — lives in the dependency modules.
"""

import time

import structlog

from utils.apexchat.core.llm import LLMClient, get_general_tool_client
from utils.apexchat.core.status_stream import emit_status
from utils.apexchat.schemas.models import BookingState, BookingStep, WorkflowState
from utils.apexchat.tools.general import BaseTool

from . import services
from .errors import BookingError, InvalidTransitionError
from .machine import advance, is_terminal, reset, rewind
from .matching import is_cancel
from .multi_extractor import (
    HINT_CURRICULUM,
    HINT_DESCRIPTION,
    HINT_INVITEES,
    HINT_LEVEL,
    HINT_SLOT,
    HINT_SUBJECT,
    HINT_TEACHER,
    extract_multi_step,
    intent_to_hints,
    looks_multi_step,
)
from .presenters import curriculum_label
from .steps import HANDLERS, StepResult, _present

logger = structlog.get_logger(__name__)

# Hard guard against runaway auto-advance loops (each iteration MUST consume
# state strictly, but a bug could otherwise loop forever).
_MAX_DISPATCH_ITERATIONS = 12

# Steps where multi-step pre-extraction makes sense. Wallet, material upload,
# and confirmation expect a single deterministic answer — we don't burn an LLM
# call to look for extra fields there.
_MULTI_HINT_STEPS: frozenset[BookingStep] = frozenset({
    BookingStep.AWAITING_LEVEL,
    BookingStep.AWAITING_SUBJECT,
    BookingStep.AWAITING_CURRICULUM,
    BookingStep.AWAITING_TEACHER,
    BookingStep.AWAITING_SLOT,
    BookingStep.AWAITING_DESCRIPTION,
    BookingStep.AWAITING_INVITEES,
})

# Order in which we summarise captured fields back to the user. Mirrors the
# canonical step order so prefix lines feel natural ("Got: <level>, <subject>…").
_CAPTURE_ORDER: tuple[tuple[str, BookingStep], ...] = (
    (HINT_LEVEL, BookingStep.AWAITING_LEVEL),
    (HINT_SUBJECT, BookingStep.AWAITING_SUBJECT),
    (HINT_CURRICULUM, BookingStep.AWAITING_CURRICULUM),
    (HINT_TEACHER, BookingStep.AWAITING_TEACHER),
    (HINT_SLOT, BookingStep.AWAITING_SLOT),
    (HINT_DESCRIPTION, BookingStep.AWAITING_DESCRIPTION),
    (HINT_INVITEES, BookingStep.AWAITING_INVITEES),
)


def _captured_summary(
    bs: BookingState,
    applied_hints: list[str],
) -> str:
    """
    Build a one-line acknowledgement listing every multi-step field captured
    in this turn. Empty string when no hints were applied — single-step turns
    use the per-step prompt verbatim and don't need a recap.
    """
    if not applied_hints:
        return ""

    parts: list[str] = []
    for hint_key, _ in _CAPTURE_ORDER:
        if hint_key not in applied_hints:
            continue
        label = _hint_label(hint_key, bs)
        if label:
            parts.append(label)
    if not parts:
        return ""
    return f"Got it — {', '.join(parts)}."


def _hint_label(hint_key: str, bs: BookingState) -> str:
    if hint_key == HINT_LEVEL and bs.level_name:
        return f"level: {bs.level_name}"
    if hint_key == HINT_SUBJECT and bs.subject_name:
        return f"subject: {bs.subject_name}"
    if hint_key == HINT_CURRICULUM and bs.curriculum_code:
        return f"curriculum: {curriculum_label(bs.curriculum_code) or bs.curriculum_code}"
    if hint_key == HINT_TEACHER and bs.tutor_name:
        return f"teacher: {bs.tutor_name}"
    if hint_key == HINT_SLOT and bs.date and bs.time_from:
        return f"slot: {bs.date} {bs.time_from}"
    if hint_key == HINT_DESCRIPTION and bs.description:
        return f"focus: {bs.description}"
    if hint_key == HINT_INVITEES:
        if bs.invitees:
            return f"invitees: {len(bs.invitees)}"
        return "invitees: none"
    return ""


class BookingTool(BaseTool):
    """
    Multi-turn booking assistant.

    Step 1 wallet check → 2 level → 3 subject → 4 curriculum (skippable) →
    5 teacher → 6 slot → 7 description (optional) → 8 invitees (optional) →
    9 material upload (required, set out-of-band) → 10 confirmation.
    """

    @property
    def name(self) -> str:
        return "booking"

    @property
    def description(self) -> str:
        return (
            "Guides students through booking a tutoring session via a strict, "
            "auditable state machine: wallet → level → subject → curriculum → "
            "teacher → slot → description → invitees → material → confirmation."
        )

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or get_general_tool_client()

    async def execute(self, state: WorkflowState) -> str:
        emit_status("tool_booking")
        start = time.perf_counter()
        bs = state.booking_state

        if not state.user_id or 'guest' in state.user_id:
            return (
                "I'd love to help you book a session, but it looks like you're not "
                "signed in. Please log in or sign up first to book a session."
            )

        try:
            response = await self._dispatch(state, bs)
        except BookingError as exc:
            logger.warning(
                "BookingError surfaced to user",
                session_id=state.session_id,
                code=exc.code,
                detail=exc.detail,
            )
            response = exc.user_message
        except Exception as exc:
            logger.error(
                "BookingTool unexpected error",
                session_id=state.session_id,
                error=str(exc),
                exc_info=True,
            )
            response = (
                "Something went wrong while processing your booking. "
                "Let's start over — just say 'book a session' to begin again."
            )
            reset(bs)

        logger.info(
            "BookingTool turn complete",
            session_id=state.session_id,
            step=bs.step.value,
            elapsed_ms=round((time.perf_counter() - start) * 1000, 1),
        )
        return response

    # ── Dispatcher ────────────────────────────────────────────────────────────

    async def _dispatch(self, state: WorkflowState, bs: BookingState) -> str:
        if bs.step == BookingStep.IDLE:
            await self._enter_flow(state, bs)
            if bs.step == BookingStep.WALLET_NEEDED:
                return _present(bs) or "Let's start your booking."
            # Otherwise we're at AWAITING_LEVEL — fall through to the dispatch
            # loop so any multi-step hints in the user's first message can be
            # applied in the same turn.

        if bs.step == BookingStep.COMPLETED:
            msg = (state.user_message or "").strip().lower()
            wants_new = any(
                kw in msg
                for kw in (
                    "book another", "another session", "new session",
                    "schedule another", "book a session", "schedule a lesson",
                    "book a lesson", "book lesson", "book new",
                )
            )
            if wants_new:
                reset(bs)
                await self._enter_flow(state, bs)
                if bs.step == BookingStep.WALLET_NEEDED:
                    return _present(bs) or "Let's start your next booking."
                # else: fall through into the dispatch loop with the wants_new
                # message — multi-extractor may pick up additional hints
                # ("book another with English math").
            else:
                return (
                    "Your previous booking is already confirmed. "
                    "Just say 'book another session' whenever you're ready."
                )

        # ── Multi-step pre-extraction ────────────────────────────────────────
        # If the user packed several fields into one message, hoist them into
        # `bs.pending_hints` once. Each step in the loop below pops its own key
        # before falling back to per-step extraction. We do this BEFORE the
        # cancel check inside the loop because cancel is detected directly on
        # the raw message and we want the original wording available.
        message = state.user_message or ""
        if (
            message.strip()
            and not is_cancel(message)
            and bs.step in _MULTI_HINT_STEPS
            and not bs.pending_hints
            and looks_multi_step(message)
        ):
            try:
                multi_intent = await extract_multi_step(message, bs, self._llm)
            except Exception as exc:
                logger.warning(
                    "multi_step_extraction_failed",
                    session_id=state.session_id,
                    error=str(exc),
                )
                multi_intent = None

            if multi_intent is not None:
                if multi_intent.cancel:
                    reset(bs)
                    return (
                        "Booking cancelled — no worries. "
                        "Say 'book a session' to start again."
                    )
                if multi_intent.rewind_to:
                    try:
                        rewind(bs, multi_intent.rewind_to, trigger="user_rewind_multi")
                    except InvalidTransitionError as exc:
                        return exc.user_message
                hints = intent_to_hints(multi_intent)
                if hints:
                    bs.pending_hints = hints
                    # The multi-extractor consumed the user's message; clearing
                    # it stops per-step LLM fallbacks from re-processing the
                    # same text and over-matching downstream steps.
                    state.user_message = ""
                    message = ""

        # ── Step dispatch loop ───────────────────────────────────────────────
        responses: list[str] = []
        applied_hints: list[str] = []

        for _ in range(_MAX_DISPATCH_ITERATIONS):
            if is_cancel(state.user_message or "") and bs.step != BookingStep.IDLE:
                reset(bs)
                return "Booking cancelled — no worries. Say 'book a session' to start again."

            handler = HANDLERS.get(bs.step)
            if handler is None:
                return _present(bs) or "Something went wrong — let's try again."

            result: StepResult = await handler.process(state, bs, self._llm)

            applied = result.extras.get("hint_applied") if result.extras else None
            if applied:
                applied_hints.append(applied)

            if result.response:
                responses.append(result.response)

            if not result.advanced:
                break
            if is_terminal(bs.step):
                break

            # The advancing handler already consumed the user's message; stop the
            # next iteration from also extracting from it. We keep the loop only
            # to render the next step's `present()` text in the same turn.
            state.user_message = ""

        # Drop any leftover hints — they belonged to steps we didn't reach
        # because an earlier hint failed validation. The user will be asked
        # for the next field through the normal flow on the following turn.
        if bs.pending_hints:
            bs.pending_hints.clear()

        prefix = _captured_summary(bs, applied_hints)
        body = "\n\n".join(r for r in responses if r)
        if prefix and body:
            return f"{prefix}\n\n{body}"
        return prefix or body

    async def _enter_flow(self, state: WorkflowState, bs: BookingState) -> None:
        """Run wallet check then transition to AWAITING_LEVEL or WALLET_NEEDED."""
        funded = await services.is_wallet_funded(state.user_id or "")
        if not funded:
            advance(bs, BookingStep.WALLET_NEEDED, trigger="wallet_check_failed")
            return
        bs.wallet_ok = True
        advance(bs, BookingStep.AWAITING_LEVEL, trigger="wallet_check_passed")
        from .steps import LevelStep
        await LevelStep().ensure_data(state, bs)
