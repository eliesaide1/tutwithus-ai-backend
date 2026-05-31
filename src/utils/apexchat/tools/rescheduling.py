"""
Rescheduling Tool — lets any authenticated user (student or admin) reschedule
an existing tutoring session.

Conversation flow — two paths
──────────────────────────────
PATH A — session ID only (new assisted flow)
  1. User provides just the session ID (no date/time yet).
  2. Tool fetches the tutor's available time slots from the DB and shows them
     as a numbered list so the user can pick one.
  3. User selects a slot number *or* types a free-form date/time.
  4. Tool populates date + time_from and shows the confirmation summary.
  5. On "yes" → writes rescheduling_contract; on "no" → cancels.

PATH B — partial or complete info (original collecting flow)
  1. User expresses intent to reschedule and may provide some or all of the
     three required pieces of information in their first message.
  2. Tool extracts whatever is present and asks for each missing field, one
     at a time, until all three are collected.
  3. Once all three fields are filled the tool shows a confirmation summary.
  4. On "yes" → writes rescheduling_contract; on "no" → cancels.

The two paths merge at the AWAITING_CONFIRMATION step — everything from
confirmation onward is identical.

Server-side validations (enforced by the backend that receives the contract):
  • The requesting user must own the session          (403 if not)
  • The session must still be > 24 hours away         (400 if too close)
  • The new datetime must not be in the past          (400)
  • The session must not already be completed         (400)
  • The tutor must be free at the requested slot      (409 if conflicting)
  • Timezones are resolved from user/tutor profiles — not passed here.

Contract shape written to state.rescheduling_contract
──────────────────────────────────────────────────────
{
    "courseId":  "<24-char hex MongoDB ObjectId>",
    "date":      "YYYY-MM-DD",
    "timeFrom":  "HH:mm",
}
"""

import re
import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from utils.apexchat.core.llm import get_general_tool_client
from utils.apexchat.core.mongodb import fetch_session_available_slots
from utils.apexchat.core.status_stream import emit_status
from utils.apexchat.schemas.models import (
    ReschedulingState,
    ReschedulingStep,
    ReschedulingStepResult,
    WorkflowState,
)
from utils.apexchat.tools.general import BaseTool

logger = structlog.get_logger(__name__)

# Maximum number of slots to display to the user
_MAX_SLOTS_SHOWN = 10


# ── Format validators ─────────────────────────────────────────────────────────

_DATE_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$")
_TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
_OBJECT_ID_RE = re.compile(r"^[0-9a-fA-F]{24}$")


def _is_valid_date(value: str | None) -> bool:
    return bool(value and _DATE_RE.fullmatch(value))


def _is_valid_time(value: str | None) -> bool:
    return bool(value and _TIME_RE.fullmatch(value))


def _is_valid_course_id(value: str | None) -> bool:
    """Accept 24-char hex MongoDB ObjectId strings only."""
    return bool(value and _OBJECT_ID_RE.fullmatch(value))


# ── LLM system prompt ─────────────────────────────────────────────────────────

def _build_extraction_prompt(
    course_id: str,
    date: str,
    time_from: str,
    missing_fields: str,
    available_slots: list[dict] | None = None,
) -> str:
    from datetime import datetime, timezone
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    slots_section = ""
    if available_slots:
        lines = ["\nAvailable time slots for this session (for reference):"]
        for i, slot in enumerate(available_slots[:_MAX_SLOTS_SHOWN], 1):
            lines.append(
                f"  {i}. {slot['date']}  {slot['timeFrom']} – {slot['timeTo']}"
            )
        lines.append(
            "If the user refers to one of these slots by number or description, "
            "extract the corresponding date and timeFrom values from the list above."
        )
        slots_section = "\n".join(lines)

    return f"""You are an assistant helping a user reschedule a tutoring session.

Today's date (UTC): {today_utc}. Use this to resolve relative date phrases like
"tomorrow", "next Monday", "this Friday" into YYYY-MM-DD format.

Your job:
1. Extract rescheduling details from the user's message.
2. Convert natural-language dates and times to the required formats:
   - Date  → YYYY-MM-DD   (e.g. "Friday 7 March 2026"  → "2026-03-07")
   - Time  → HH:mm 24-h   (e.g. "10:00 am" → "10:00", "2:30 pm" → "14:30")
3. Extract the session/course ID — it must be a 24-character hex string
   (MongoDB ObjectId). Copy it exactly as the user wrote it; do NOT modify,
   shorten, or guess it.
4. Merge extracted values with what is already collected (shown below).
5. Write a short, professional response_text:
   - If required fields are still missing → acknowledge what was captured and
     ask for the NEXT missing field only (one at a time).
   - If all three fields are now filled → show a clear summary and ask the
     user to type "yes" to confirm or "no" to cancel.
   - If the user says "yes" / "confirm" → set confirmed=true.
   - If the user says "no" / "cancel"   → set cancel=true.

CRITICAL RULES:
- Never invent, assume, or fill in any field value — only set it if the user
  explicitly stated it in their current message.
- Output dates ONLY in YYYY-MM-DD format. Output times ONLY in HH:mm 24-hour format.
- If you cannot parse a date or time into the correct format, leave the field null.
- Keep response_text to 2–4 sentences maximum.
{slots_section}
Already collected:
- Session ID : {course_id}
- New date   : {date}
- Start time : {time_from}

Missing required fields: {missing_fields}
"""


# ── Tool ──────────────────────────────────────────────────────────────────────

class ReschedulingTool(BaseTool):
    """
    Multi-turn rescheduling assistant.

    Supports two paths:
      • Assisted  — user provides only the session ID; the tool fetches the
                    tutor's available slots and lets the user pick one.
      • Direct    — user provides some or all fields up front; the tool
                    collects any missing ones and asks for confirmation.

    Both paths emit an identical rescheduling_contract once confirmed.
    (End time is calculated server-side from the session's original duration.)
    """

    @property
    def name(self) -> str:
        return "rescheduling"

    @property
    def description(self) -> str:
        return (
            "Helps any user reschedule an existing tutoring session by "
            "collecting the session ID, new date, and new start/end times."
        )

    # ── Entry point ───────────────────────────────────────────────────────────

    async def execute(self, state: WorkflowState) -> str:
        emit_status("tool_rescheduling")

        if not state.user_id or 'guest' in state.user_id:
            return (
                "I'd love to help you reschedule a session, but it looks like you're not "
                "signed in. Please log in or sign up first to reschedule a session."
            )

        rs = state.rescheduling_state

        # A previous reschedule finished on a prior turn — start a fresh flow
        # so the user can reschedule a *different* session. Clear any leftover
        # contract from the previous turn so it isn't re-emitted in the API
        # response for this new request.
        if rs.step == ReschedulingStep.COMPLETED:
            self._reset(rs)
            state.rescheduling_contract = None

        if rs.step in (ReschedulingStep.IDLE, ReschedulingStep.COLLECTING):
            rs.step = ReschedulingStep.COLLECTING
            return await self._handle_collecting(state)

        if rs.step == ReschedulingStep.AWAITING_SLOT_SELECTION:
            return await self._handle_slot_selection(state)

        if rs.step == ReschedulingStep.AWAITING_CONFIRMATION:
            return await self._handle_confirmation(state)

        return "Something went wrong with the rescheduling flow. Please try again."

    # ── Step handlers ─────────────────────────────────────────────────────────

    async def _handle_collecting(self, state: WorkflowState) -> str:
        """Use the LLM to extract fields and advance the conversation."""
        rs = state.rescheduling_state
        missing = self._missing_fields(rs)

        system_prompt = _build_extraction_prompt(
            course_id=rs.course_id or "not provided",
            date=rs.date or "not provided",
            time_from=rs.time_from or "not provided",
            missing_fields=", ".join(missing) if missing else "none — ready to confirm",
            available_slots=rs.cached_slots or None,
        )

        llm = get_general_tool_client()
        structured_llm = llm.with_structured_output(ReschedulingStepResult)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.user_message),
        ]

        try:
            result: ReschedulingStepResult = await structured_llm.ainvoke(messages)
        except Exception as exc:
            logger.error("Rescheduling LLM extraction failed", error=str(exc))
            return (
                "I had trouble understanding that. "
                "Could you please provide the rescheduling details again?"
            )

        # Handle explicit cancel before merging
        if result.cancel:
            return self._cancel(rs)

        # Merge any newly extracted fields
        self._merge(rs, result)

        # ── Duplicate-reschedule guard ────────────────────────────────────────
        # If the user supplied a session ID that has already been rescheduled
        # earlier in this conversation, refuse politely and reset the flow
        # without producing a new contract (and without re-emitting the old
        # one). The user may then provide a different session ID.
        if rs.course_id and rs.course_id in rs.rescheduled_session_ids:
            blocked_id = rs.course_id
            self._reset(rs)
            state.rescheduling_contract = None
            return (
                f"Session **{blocked_id}** has already been rescheduled. "
                "You can't reschedule the same session twice. "
                "If you'd like to reschedule a different session, please share its session ID."
            )

        # ── Still need the session ID before we can do anything else ──────────
        if not rs.course_id:
            return result.response_text

        # ── Fetch tutor's available slots once we know the session ───────────
        if not rs.cached_slots:
            try:
                slots = await fetch_session_available_slots(rs.course_id)
            except Exception as exc:
                logger.warning(
                    "Rescheduling: slot fetch failed",
                    error=str(exc),
                )
                slots = []

            if not slots:
                # No bookable slots → the user can't pick anything; abort the
                # flow rather than letting them invent a free-form time we
                # cannot validate.
                self._reset(rs)
                state.rescheduling_contract = None
                return (
                    "I couldn't find any available time slots for this session right now. "
                    "Please try again later or contact support."
                )

            rs.cached_slots = slots

        # ── Validate any user-provided date+time against the available slots ──
        # The user can only reschedule into an existing slot — free-form picks
        # are rejected and we re-prompt with the list.
        if rs.date and rs.time_from:
            if self._slot_is_available(rs):
                rs.step = ReschedulingStep.AWAITING_CONFIRMATION
                return self._confirmation_prompt(rs)

            requested_date = rs.date
            requested_time = rs.time_from
            rs.date = None
            rs.time_from = None
            rs.step = ReschedulingStep.AWAITING_SLOT_SELECTION
            return (
                f"Sorry, **{requested_date} {requested_time}** isn't one of the tutor's "
                "available slots. Please pick one of the options below:\n\n"
                + self._slots_prompt(rs)
            )

        # Either date or time is missing — show the slot list so the user
        # picks an exact, validated option (no free-form input allowed).
        rs.step = ReschedulingStep.AWAITING_SLOT_SELECTION
        return self._slots_prompt(rs)

    async def _handle_slot_selection(self, state: WorkflowState) -> str:
        """
        User is choosing from the displayed list of available slots.

        Fast-path: a plain integer (1..N) directly selects the matching cached
        slot — no LLM call.

        Fallback: any other text is parsed by the LLM in _handle_collecting,
        which then validates the extracted date/time against cached_slots and
        either confirms or re-shows the list. Free-form picks that don't match
        an available slot are rejected.
        """
        rs = state.rescheduling_state
        msg = state.user_message.strip()

        _CANCEL_KW = {
            "no", "cancel", "abort", "stop", "discard", "nope", "nevermind",
            "never mind", "don't", "do not",
        }
        if any(kw in msg.lower() for kw in _CANCEL_KW):
            return self._cancel(rs)

        shown = min(_MAX_SLOTS_SHOWN, len(rs.cached_slots))

        # Fast-path: plain integer slot selection
        try:
            choice = int(msg)
        except ValueError:
            choice = None

        if choice is not None:
            if 1 <= choice <= shown:
                selected = rs.cached_slots[choice - 1]
                rs.date = selected["date"]
                rs.time_from = selected["timeFrom"]
                rs.step = ReschedulingStep.AWAITING_CONFIRMATION
                return self._confirmation_prompt(rs)
            return (
                f"Please choose a number between 1 and {shown} from the list above."
            )

        # Fallback: LLM-based extraction (user described a slot in natural
        # language). _handle_collecting validates the result against
        # cached_slots and re-prompts if the pick isn't on the list.
        rs.step = ReschedulingStep.COLLECTING
        return await self._handle_collecting(state)

    async def _handle_confirmation(self, state: WorkflowState) -> str:
        """
        Accept a yes/no from the user and finalise or cancel.

        Strategy:
        1. Fast-path keyword scan for unambiguous confirmations / cancellations.
        2. If the message is ambiguous (e.g. the user is correcting a value),
           pass it back through the collecting LLM so the updated field is
           extracted and the confirmation summary is re-shown.
        """
        rs = state.rescheduling_state
        msg = state.user_message.strip().lower()

        _CONFIRM_KW = {
            "yes", "confirm", "confirmed", "ok", "okay", "sure", "proceed",
            "correct", "go ahead", "looks good", "let's go", "do it",
            "yep", "yup", "absolutely", "that's right", "right", "perfect",
        }
        _CANCEL_KW = {
            "no", "cancel", "abort", "stop", "discard", "nope", "nevermind",
            "never mind", "don't", "do not",
        }

        if any(kw in msg for kw in _CONFIRM_KW):
            rs.step = ReschedulingStep.COMPLETED
            if rs.course_id and rs.course_id not in rs.rescheduled_session_ids:
                rs.rescheduled_session_ids.append(rs.course_id)
            state.rescheduling_contract = self._build_contract(rs)
            logger.info(
                "Rescheduling contract created",
                user_id=state.user_id,
                course_id=rs.course_id,
                date=rs.date,
                time_from=rs.time_from,
            )
            return (
                "Rescheduling in progress! Your session will be updated shortly "
                "if all details are valid."
            )

        if any(kw in msg for kw in _CANCEL_KW):
            return self._cancel(rs)

        # ── Ambiguous: user may be correcting a value — re-collect then re-confirm ──
        rs.step = ReschedulingStep.COLLECTING
        return await self._handle_collecting(state)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _missing_fields(self, rs: ReschedulingState) -> list[str]:
        """Return human-readable names of required fields not yet filled."""
        missing = []
        if not rs.course_id:
            missing.append("session ID")
        if not rs.date:
            missing.append("new date (YYYY-MM-DD)")
        if not rs.time_from:
            missing.append("new start time (HH:mm)")
        return missing

    def _merge(self, rs: ReschedulingState, result: ReschedulingStepResult) -> None:
        """
        Overwrite state fields with non-None values the LLM extracted.
        Rejects values that fail strict format validation so malformed LLM
        output never silently pollutes the state.
        """
        if result.course_id:
            if _is_valid_course_id(result.course_id):
                rs.course_id = result.course_id
            else:
                logger.warning(
                    "Rescheduling: course_id rejected — not a valid 24-char hex ObjectId",
                    raw=result.course_id,
                )
        if result.date:
            if _is_valid_date(result.date):
                rs.date = result.date
            else:
                logger.warning(
                    "Rescheduling: date rejected — invalid YYYY-MM-DD format",
                    raw=result.date,
                )
        if result.time_from:
            if _is_valid_time(result.time_from):
                rs.time_from = result.time_from
            else:
                logger.warning(
                    "Rescheduling: time_from rejected — invalid HH:mm format",
                    raw=result.time_from,
                )

    def _slot_is_available(self, rs: ReschedulingState) -> bool:
        """
        Return True iff (rs.date, rs.time_from) matches one of the tutor's
        cached available slots. The reschedule must land on an existing slot —
        free-form date/times are not allowed.
        """
        if not (rs.date and rs.time_from and rs.cached_slots):
            return False
        return any(
            slot.get("date") == rs.date and slot.get("timeFrom") == rs.time_from
            for slot in rs.cached_slots
        )

    def _reset(self, rs: ReschedulingState) -> None:
        """Clear the in-flight reschedule fields without touching history."""
        rs.step = ReschedulingStep.IDLE
        rs.course_id = None
        rs.date = None
        rs.time_from = None
        rs.cached_slots = []

    def _slots_prompt(self, rs: ReschedulingState) -> str:
        """Format the tutor's available slots as a numbered list for the user."""
        slots = rs.cached_slots[:_MAX_SLOTS_SHOWN]
        lines = ["Here are the available time slots for this session:\n"]
        for i, slot in enumerate(slots, 1):
            lines.append(
                f"**{i}.** {slot['date']}  {slot['timeFrom']} – {slot['timeTo']}"
            )
        lines.append(
            "\nType the **slot number** to select it, "
            "or enter a specific **date and time** if you prefer a different one."
        )
        return "\n".join(lines)

    def _confirmation_prompt(self, rs: ReschedulingState) -> str:
        """Return a formatted summary asking the user to confirm or cancel."""
        return (
            "Please review the rescheduling details:\n\n"
            f"- **Session ID:** {rs.course_id}\n"
            f"- **New Date:** {rs.date}\n"
            f"- **New Start Time:** {rs.time_from}\n\n"
            "Type **yes** to confirm the reschedule, or **no** to cancel."
        )

    def _build_contract(self, rs: ReschedulingState) -> dict:
        """Build the rescheduling_contract payload for the API response."""
        return {
            "courseId": rs.course_id,
            "date": rs.date,
            "timeFrom": rs.time_from,
        }

    def _cancel(self, rs: ReschedulingState) -> str:
        """Reset all collected fields and return a cancellation message."""
        rs.step = ReschedulingStep.IDLE
        rs.course_id = None
        rs.date = None
        rs.time_from = None
        rs.cached_slots = []
        return "Rescheduling cancelled. No changes have been made to your session."
