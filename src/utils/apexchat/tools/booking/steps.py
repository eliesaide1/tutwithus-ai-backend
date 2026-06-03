"""
Step handlers — one per state in the booking flow.

Every handler implements the same `process(state, bs, llm)` contract:

    1. ensure_data       — load any cached lists this step needs
    2. deterministic     — try cheap exact / numeric / keyword matching
    3. llm_fallback      — single structured-output call only when needed
    4. validate + apply  — check candidate against cached options, mutate state
    5. transition        — call machine.advance / machine.rewind

Steps NEVER mutate `bs.step` directly — only `machine` does that.

Every handler returns a `StepResult` that the dispatcher uses to decide whether
to immediately re-enter the loop (e.g. user picked a level + already named the
subject in the same message) or to send the response to the user and wait.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from utils.apexchat.core.llm import LLMClient
from utils.apexchat.schemas.models import (
    BookingInvitee,
    BookingState,
    BookingStep,
    WorkflowState,
)

from . import services
from .errors import (
    BookingError,
    BusinessRuleViolation,
    InvalidTransitionError,
    SlotConflict,
    StepDataError,
)
from .extractor import StepIntent, extract
from .machine import (
    advance,
    invalidate_dependents,
    next_step,
    reset,
    rewind,
)
from .matching import (
    is_cancel,
    is_confirm,
    is_skip,
    match_curriculum,
    match_level,
    match_slot,
    match_subject,
    match_tutor,
)
from .multi_extractor import (
    HINT_CURRICULUM,
    HINT_DESCRIPTION,
    HINT_INVITEES,
    HINT_LEVEL,
    HINT_SLOT,
    HINT_SUBJECT,
    HINT_TEACHER,
    SKIP,
    extract_multi_step,
    intent_to_hints,
)
from .presenters import (
    curriculum_label,
    format_booking_summary,
    format_completion,
    format_curricula,
    format_levels,
    format_slots,
    format_subjects,
    format_tutors,
    _level_display_name,
    _subject_display_name,
    _tutor_display_name,
)
from .validators import (
    is_valid_email,
    slot_lead_time_ok,
)
from .contract import build_contract

logger = structlog.get_logger(__name__)


# ── Result envelope ───────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """
    Outcome of one step handler invocation.

    advanced=True signals the dispatcher MAY immediately re-enter the loop on
    the new step (only safe when consuming nothing more from the user message).
    """
    response: str
    advanced: bool = False
    consume_message: bool = True
    extras: dict[str, Any] = field(default_factory=dict)


# ── Helpers shared across steps ───────────────────────────────────────────────

def _cancel(bs: BookingState) -> StepResult:
    reset(bs)
    return StepResult(
        response="Booking cancelled — no worries. Let me know whenever you'd like to start fresh.",
        advanced=False,
    )


async def _try_rewind(
    state: WorkflowState,
    bs: BookingState,
    intent: StepIntent,
    llm: LLMClient,
) -> StepResult | None:
    """
    Apply a user-requested rewind. When the same message also carries a new
    value for the rewound-to step (e.g. "go back to subject and pick physics"),
    pull it through the multi-extractor and stage it as a pending hint so the
    dispatch loop applies it on the next iteration and continues forward.
    Returns None when the intent isn't a rewind.
    """
    if intent.action != "rewind" or not intent.rewind_to:
        return None
    try:
        rewind(bs, intent.rewind_to, trigger="user_rewind")
    except InvalidTransitionError as exc:
        return StepResult(response=exc.user_message)

    # The same message may name a new value for the rewound step (or even for
    # later ones the user is touching at once). Run the multi-extractor on
    # the original message; any hints it returns get staged for the dispatch
    # loop, which will pop them step-by-step starting from the rewound step.
    message = state.user_message or ""
    if message.strip():
        try:
            multi = await extract_multi_step(message, bs, llm)
        except Exception as exc:
            logger.warning("multi_extract_after_rewind_failed", error=str(exc))
            multi = None
        if multi is not None:
            hints = intent_to_hints(multi)
            if hints:
                bs.pending_hints = hints
                # The message has been consumed by the multi-extractor; keep
                # per-step LLM fallbacks from re-reading it on later iterations.
                state.user_message = ""
                # advanced=True so the dispatcher loops back into the rewound
                # step's handler and picks up the staged hint.
                return StepResult(response="", advanced=True)

    return StepResult(
        response=f"Got it — let's go back. {_present(bs)}",
        advanced=False,
    )


def _present(bs: BookingState) -> str:
    handler = HANDLERS.get(bs.step)
    if not handler:
        return ""
    return handler.present(bs)


# ── Base ──────────────────────────────────────────────────────────────────────

class StepHandler:
    step: BookingStep
    requires_llm_fallback: bool = True

    # Key in bs.pending_hints that this step is responsible for. None when the
    # step does not participate in multi-step extraction (wallet, material,
    # confirmation).
    hint_key: str | None = None

    async def ensure_data(self, state: WorkflowState, bs: BookingState) -> None:
        """Load cached options for this step. Raise StepDataError on hard failure."""
        return None

    def deterministic(self, message: str, bs: BookingState) -> Any:
        """Try cheap exact matching. Return a typed candidate or None."""
        return None

    def apply_candidate(self, candidate: Any, bs: BookingState) -> str | None:
        """Mutate bs with the candidate. Return None on success or an error message."""
        return None

    def present(self, bs: BookingState) -> str:
        return ""

    def _candidate_from_hint(self, hint_value: Any, bs: BookingState) -> Any:
        """
        Validate a value pulled from `bs.pending_hints` against this step's
        cached options. Default: treat the hint as a free-text message and
        run the same deterministic matcher. Steps with structured hints
        (slot, description, invitees) override this.
        """
        if isinstance(hint_value, str):
            return self.deterministic(hint_value, bs)
        return None

    async def _try_apply_hint(
        self,
        state: WorkflowState,
        bs: BookingState,
    ) -> StepResult | None:
        """
        If `bs.pending_hints` carries this step's value, validate and apply it.

        Returns a StepResult when a hint was processed (success OR rejection).
        Returns None when no hint exists, so the caller falls back to its
        regular per-step flow.
        """
        if not self.hint_key or self.hint_key not in bs.pending_hints:
            return None

        hint_value = bs.pending_hints.pop(self.hint_key)

        try:
            await self.ensure_data(state, bs)
        except StepDataError as exc:
            return StepResult(response=exc.user_message)

        candidate = self._candidate_from_hint(hint_value, bs)
        if candidate is None:
            # Hint provided but didn't resolve — drop the rest of the chain so
            # we don't apply downstream hints that depend on this one (e.g. a
            # subject hint depends on the level being set first).
            bs.pending_hints.clear()
            return StepResult(
                response=self._hint_rejection(hint_value, bs),
                advanced=False,
                extras={"hint_rejected": self.hint_key},
            )

        err = self.apply_candidate(candidate, bs)
        if err:
            bs.pending_hints.clear()
            return StepResult(response=err, advanced=False)

        return await self._after_apply(
            state, bs, extras={"hint_applied": self.hint_key}
        )

    def _hint_rejection(self, hint_value: Any, bs: BookingState) -> str:
        """Message shown when a multi-step hint cannot be resolved."""
        return (
            f"I couldn't match \"{hint_value}\" to one of the available options. "
            f"{self.present(bs)}"
        )

    async def process(
        self,
        state: WorkflowState,
        bs: BookingState,
        llm: LLMClient,
    ) -> StepResult:
        message = state.user_message or ""

        if is_cancel(message):
            return _cancel(bs)

        # Multi-step hint takes priority over re-reading the user_message,
        # which the dispatcher already cleared once the multi-extractor
        # consumed it for this turn.
        hint_result = await self._try_apply_hint(state, bs)
        if hint_result is not None:
            return hint_result

        try:
            await self.ensure_data(state, bs)
        except StepDataError as exc:
            return StepResult(response=exc.user_message)

        candidate = self.deterministic(message, bs)
        intent: StepIntent | None = None

        if candidate is None and self.requires_llm_fallback and message.strip():
            intent = await extract(self.step, message, bs, llm)
            rewind_result = await _try_rewind(state, bs, intent, llm)
            if rewind_result:
                return rewind_result
            if intent.action == "cancel":
                return _cancel(bs)
            candidate = self._candidate_from_intent(intent, bs)

        if candidate is None:
            return StepResult(response=self.present(bs), advanced=False)

        err = self.apply_candidate(candidate, bs)
        if err:
            return StepResult(response=err, advanced=False)

        return await self._after_apply(state, bs)

    def _candidate_from_intent(self, intent: StepIntent, bs: BookingState) -> Any:
        return None

    async def _after_apply(
        self,
        state: WorkflowState,
        bs: BookingState,
        *,
        extras: dict[str, Any] | None = None,
    ) -> StepResult:
        target = next_step(bs.step, requires_curriculum=bs.requires_curriculum)
        try:
            advance(bs, target, trigger="step_complete")
        except InvalidTransitionError as exc:
            return StepResult(response=exc.user_message)

        next_handler = HANDLERS.get(bs.step)
        if next_handler:
            try:
                await next_handler.ensure_data(state, bs)
            except StepDataError as exc:
                return StepResult(response=exc.user_message)
        # No response here — the dispatcher loop will iterate and let the next
        # step's handler render its own present() text. Returning the next
        # step's present() here AND advanced=True caused the same prompt to be
        # appended twice ("Which subject…\n\nWhich subject…").
        return StepResult(response="", advanced=True, extras=extras or {})


# ── Wallet ────────────────────────────────────────────────────────────────────

class WalletStep(StepHandler):
    step = BookingStep.WALLET_NEEDED
    requires_llm_fallback = False

    async def process(
        self,
        state: WorkflowState,
        bs: BookingState,
        llm: LLMClient,
    ) -> StepResult:
        if is_cancel(state.user_message or ""):
            return _cancel(bs)

        funded = await services.is_wallet_funded(state.user_id or "")
        if not funded:
            return StepResult(
                response=(
                    "Your wallet still appears empty. Please recharge it and let me know "
                    "when you're ready — I'll pick up right where we left off."
                ),
            )

        bs.wallet_ok = True
        try:
            advance(bs, BookingStep.AWAITING_LEVEL, trigger="wallet_funded")
        except InvalidTransitionError as exc:
            return StepResult(response=exc.user_message)

        await LevelStep().ensure_data(state, bs)
        return StepResult(
            response=f"Great — your wallet is funded! 🎉\n\n{LevelStep().present(bs)}",
        )

    def present(self, bs: BookingState) -> str:
        return (
            "Your wallet doesn't have any hours left. "
            "Please recharge it and let me know when you're ready — "
            "I'll resume from here."
        )


# ── Level ─────────────────────────────────────────────────────────────────────

class LevelStep(StepHandler):
    step = BookingStep.AWAITING_LEVEL
    hint_key = HINT_LEVEL

    async def ensure_data(self, state: WorkflowState, bs: BookingState) -> None:
        if not bs.cached_levels:
            bs.cached_levels = await services.list_levels()
        if not bs.cached_levels:
            raise StepDataError(
                "No academic levels are available right now. Please check back soon.",
                code="no_levels",
            )

    def deterministic(self, message: str, bs: BookingState) -> dict | None:
        return match_level(message, bs.cached_levels)

    def _candidate_from_intent(self, intent: StepIntent, bs: BookingState) -> dict | None:
        if intent.action != "select" or not intent.selected_index:
            return None
        idx = intent.selected_index
        if 1 <= idx <= len(bs.cached_levels):
            return bs.cached_levels[idx - 1]
        return None

    def apply_candidate(self, candidate: dict, bs: BookingState) -> str | None:
        bs.level_id = candidate["_id"]
        bs.level_name = _level_display_name(candidate)
        bs.requires_curriculum = bool(candidate.get("requiresCurriculum", True))
        invalidate_dependents(bs, BookingStep.AWAITING_LEVEL)
        return None

    def present(self, bs: BookingState) -> str:
        return f"Which academic level do you need?\n\n{format_levels(bs.cached_levels)}"


# ── Subject ───────────────────────────────────────────────────────────────────

class SubjectStep(StepHandler):
    step = BookingStep.AWAITING_SUBJECT
    hint_key = HINT_SUBJECT

    async def ensure_data(self, state: WorkflowState, bs: BookingState) -> None:
        if not bs.level_id:
            raise StepDataError(
                "Let's pick an academic level first.",
                code="missing_level",
            )
        if not bs.cached_subjects:
            bs.cached_subjects = await services.list_subjects(bs.level_id)
        if not bs.cached_subjects:
            rewind(bs, BookingStep.AWAITING_LEVEL, trigger="no_subjects_for_level")
            raise StepDataError(
                f"No subjects available for {bs.level_name}. Let's pick a different level.\n\n"
                f"{format_levels(bs.cached_levels)}",
                code="no_subjects",
            )

    def deterministic(self, message: str, bs: BookingState) -> dict | None:
        return match_subject(message, bs.cached_subjects)

    def _candidate_from_intent(self, intent: StepIntent, bs: BookingState) -> dict | None:
        if intent.action != "select" or not intent.selected_index:
            return None
        idx = intent.selected_index
        if 1 <= idx <= len(bs.cached_subjects):
            return bs.cached_subjects[idx - 1]
        return None

    def apply_candidate(self, candidate: dict, bs: BookingState) -> str | None:
        bs.subject_id = candidate["_id"]
        bs.subject_name = _subject_display_name(candidate)
        invalidate_dependents(bs, BookingStep.AWAITING_SUBJECT)
        return None

    def present(self, bs: BookingState) -> str:
        return f"Which subject would you like to study?\n\n{format_subjects(bs.cached_subjects)}"


# ── Curriculum ────────────────────────────────────────────────────────────────

class CurriculumStep(StepHandler):
    step = BookingStep.AWAITING_CURRICULUM
    hint_key = HINT_CURRICULUM

    async def ensure_data(self, state: WorkflowState, bs: BookingState) -> None:
        if not (bs.level_id and bs.subject_id):
            raise StepDataError("Let's finish picking the level and subject first.", code="missing_prereqs")

        if not bs.cached_curricula:
            bs.cached_curricula = await services.list_curricula(bs.level_id, bs.subject_id)

        # If the level claimed it requires a curriculum but none exist, treat it
        # as not-required and skip — the next-step logic in machine.advance handles this.
        if not bs.cached_curricula:
            bs.requires_curriculum = False

    async def process(self, state, bs, llm):
        await self.ensure_data(state, bs)
        if not bs.requires_curriculum:
            # The level didn't require a curriculum — drop any curriculum hint
            # the user supplied so it doesn't linger or leak into a later flow.
            bs.pending_hints.pop(HINT_CURRICULUM, None)
            try:
                advance(bs, BookingStep.AWAITING_TEACHER, trigger="no_curriculum_required")
            except InvalidTransitionError as exc:
                return StepResult(response=exc.user_message)
            try:
                await TeacherStep().ensure_data(state, bs)
            except StepDataError as exc:
                return StepResult(response=exc.user_message)
            # Empty response — the dispatcher loop will render TeacherStep.present()
            # itself on the next iteration. Returning it here too caused duplicates.
            return StepResult(response="", advanced=True)
        return await super().process(state, bs, llm)

    def deterministic(self, message: str, bs: BookingState) -> str | None:
        return match_curriculum(message, bs.cached_curricula)

    def _candidate_from_intent(self, intent: StepIntent, bs: BookingState) -> str | None:
        if intent.action != "select":
            return None
        if intent.selected_curriculum and intent.selected_curriculum in bs.cached_curricula:
            return intent.selected_curriculum
        if intent.selected_index and 1 <= intent.selected_index <= len(bs.cached_curricula):
            return bs.cached_curricula[intent.selected_index - 1]
        return None

    def apply_candidate(self, candidate: str, bs: BookingState) -> str | None:
        bs.curriculum_code = candidate
        invalidate_dependents(bs, BookingStep.AWAITING_CURRICULUM)
        return None

    def present(self, bs: BookingState) -> str:
        return f"Which curriculum are you following?\n\n{format_curricula(bs.cached_curricula)}"


# ── Teacher ───────────────────────────────────────────────────────────────────

class TeacherStep(StepHandler):
    step = BookingStep.AWAITING_TEACHER
    hint_key = HINT_TEACHER

    async def ensure_data(self, state: WorkflowState, bs: BookingState) -> None:
        if not (bs.level_id and bs.subject_id):
            raise StepDataError("Let's finish picking the level and subject first.", code="missing_prereqs")
        if not bs.cached_tutors:
            bs.cached_tutors = await services.list_tutors(
                bs.level_id, bs.subject_id, bs.curriculum_code
            )
        if not bs.cached_tutors:
            if bs.requires_curriculum and bs.curriculum_code:
                rewind(bs, BookingStep.AWAITING_CURRICULUM, trigger="no_tutors_for_curriculum")
                raise StepDataError(
                    "No teachers are available for that curriculum. Please pick a different one.\n\n"
                    f"{format_curricula(bs.cached_curricula)}",
                    code="no_tutors",
                )
            failed_subject_id = bs.subject_id
            rewind(bs, BookingStep.AWAITING_SUBJECT, trigger="no_tutors_for_subject")
            # Don't offer the subject that just dead-ended — re-listing it (e.g.
            # "Robotics" with no available tutor) would only loop the user back
            # to the same "no teachers" message.
            if failed_subject_id:
                bs.cached_subjects = [
                    s for s in bs.cached_subjects if s.get("_id") != failed_subject_id
                ]
            if not bs.cached_subjects:
                # Every subject at this level is unbookable — go back to levels.
                rewind(bs, BookingStep.AWAITING_LEVEL, trigger="no_bookable_subjects")
                raise StepDataError(
                    "There are no teachers available for that level right now. "
                    f"Let's pick a different level.\n\n{format_levels(bs.cached_levels)}",
                    code="no_tutors",
                )
            raise StepDataError(
                "No teachers are available for that subject. Please pick a different one.\n\n"
                f"{format_subjects(bs.cached_subjects)}",
                code="no_tutors",
            )

    def deterministic(self, message: str, bs: BookingState) -> dict | None:
        return match_tutor(message, bs.cached_tutors)

    def _candidate_from_intent(self, intent: StepIntent, bs: BookingState) -> dict | None:
        if intent.action != "select" or not intent.selected_index:
            return None
        idx = intent.selected_index
        if 1 <= idx <= len(bs.cached_tutors):
            return bs.cached_tutors[idx - 1]
        return None

    def apply_candidate(self, candidate: dict, bs: BookingState) -> str | None:
        bs.tutor_id = candidate["_id"]
        bs.tutor_name = _tutor_display_name(candidate)
        invalidate_dependents(bs, BookingStep.AWAITING_TEACHER)
        return None

    def present(self, bs: BookingState) -> str:
        return f"Who would you like to learn with?\n\n{format_tutors(bs.cached_tutors)}"


# ── Slot ──────────────────────────────────────────────────────────────────────

# Soft slot reservation TTL — the actual atomic re-check still happens at
# confirmation time via services.slot_still_available().
_SLOT_LOCK_MINUTES = 5


class SlotStep(StepHandler):
    step = BookingStep.AWAITING_SLOT
    hint_key = HINT_SLOT

    def _candidate_from_hint(self, hint_value: Any, bs: BookingState) -> dict | None:
        if not isinstance(hint_value, dict):
            return None
        idx = hint_value.get("index")
        if isinstance(idx, int) and 1 <= idx <= len(bs.cached_slots):
            return bs.cached_slots[idx - 1]
        date = hint_value.get("date")
        time_from = hint_value.get("time")
        if date and time_from:
            for s in bs.cached_slots:
                if s["date"] == date and s["timeFrom"] == time_from:
                    return s
        if date:
            same_date = [s for s in bs.cached_slots if s["date"] == date]
            if len(same_date) == 1:
                return same_date[0]
        return None

    async def ensure_data(self, state: WorkflowState, bs: BookingState) -> None:
        if not (bs.level_id and bs.subject_id and bs.tutor_id):
            raise StepDataError("Let's pick the teacher first.", code="missing_prereqs")
        if not bs.cached_slots:
            bs.cached_slots = await services.list_tutor_slots(
                bs.level_id, bs.subject_id, bs.curriculum_code, bs.tutor_id
            )
        if not bs.cached_slots:
            rewind(bs, BookingStep.AWAITING_TEACHER, trigger="no_slots_for_tutor")
            raise StepDataError(
                f"{bs.tutor_name} has no upcoming slots. Please pick a different teacher.\n\n"
                f"{format_tutors(bs.cached_tutors)}",
                code="no_slots",
            )

    def deterministic(self, message: str, bs: BookingState) -> dict | None:
        return match_slot(message, bs.cached_slots)

    def _candidate_from_intent(self, intent: StepIntent, bs: BookingState) -> dict | None:
        if intent.action != "select":
            return None
        if intent.selected_index and 1 <= intent.selected_index <= len(bs.cached_slots):
            return bs.cached_slots[intent.selected_index - 1]
        if intent.selected_date and intent.selected_time_from:
            for s in bs.cached_slots:
                if s["date"] == intent.selected_date and s["timeFrom"] == intent.selected_time_from:
                    return s
        return None

    def apply_candidate(self, candidate: dict, bs: BookingState) -> str | None:
        from datetime import timedelta
        if not slot_lead_time_ok(candidate["date"], candidate["timeFrom"]):
            return (
                f"That slot ({candidate['date']} {candidate['timeFrom']} UTC) is too close "
                f"to the current time — sessions must start at least 48 hours from now. "
                f"Please pick another slot."
            )
        bs.date = candidate["date"]
        bs.time_from = candidate["timeFrom"]
        bs.time_to = candidate["timeTo"]
        bs.slot_lock_expires_at = datetime.utcnow() + timedelta(minutes=_SLOT_LOCK_MINUTES)
        return None

    def present(self, bs: BookingState) -> str:
        return f"Which time slot works best? (all times UTC)\n\n{format_slots(bs.cached_slots)}"


# ── Description (optional) ────────────────────────────────────────────────────

class DescriptionStep(StepHandler):
    step = BookingStep.AWAITING_DESCRIPTION
    hint_key = HINT_DESCRIPTION

    def _candidate_from_hint(self, hint_value: Any, bs: BookingState) -> str | None:
        if hint_value == SKIP:
            return _AUTOGEN_FOCUS(bs)
        if isinstance(hint_value, str) and hint_value.strip():
            return hint_value.strip()
        return None

    def deterministic(self, message: str, bs: BookingState) -> str | None:
        msg = (message or "").strip()
        if not msg:
            return None
        if is_skip(msg):
            return _AUTOGEN_FOCUS(bs)
        # Free-form text that's longer than a yes/no — accept as the focus.
        # Short answers fall through to the LLM extractor for safer parsing.
        if len(msg.split()) >= 3:
            return msg
        return None

    def _candidate_from_intent(self, intent: StepIntent, bs: BookingState) -> str | None:
        if intent.action == "skip":
            return _AUTOGEN_FOCUS(bs)
        if intent.action == "select" and intent.free_text:
            return intent.free_text.strip() or None
        return None

    def apply_candidate(self, candidate: str, bs: BookingState) -> str | None:
        bs.description = candidate
        return None

    def present(self, bs: BookingState) -> str:
        return (
            f"What would you like to focus on for this {bs.subject_name} session? "
            f"You can also reply 'skip' for general revision."
        )


def _AUTOGEN_FOCUS(bs: BookingState) -> str:
    return f"General {bs.subject_name} revision"


# ── Invitees (optional) ───────────────────────────────────────────────────────

class InviteesStep(StepHandler):
    step = BookingStep.AWAITING_INVITEES
    hint_key = HINT_INVITEES

    def _candidate_from_hint(
        self, hint_value: Any, bs: BookingState
    ) -> list[BookingInvitee] | None:
        if hint_value == SKIP:
            return []
        if not isinstance(hint_value, list):
            return None
        valid: list[BookingInvitee] = []
        for inv in hint_value[:6]:
            if not isinstance(inv, dict):
                continue
            email = (inv.get("email") or "").strip()
            first = (inv.get("firstName") or "").strip()
            last = (inv.get("lastName") or "").strip()
            if email and first and last and is_valid_email(email):
                valid.append(BookingInvitee(email=email, firstName=first, lastName=last))
        return valid or None

    def deterministic(self, message: str, bs: BookingState) -> list[BookingInvitee] | None:
        msg = (message or "").strip()
        if not msg:
            return None
        if is_skip(msg) or _is_explicit_decline(msg):
            return []
        return None

    def _candidate_from_intent(
        self, intent: StepIntent, bs: BookingState
    ) -> list[BookingInvitee] | None:
        if intent.action == "skip":
            return []
        if intent.action != "select" or not intent.invitees:
            return None
        valid: list[BookingInvitee] = []
        for inv in intent.invitees[:6]:
            if inv.email and inv.firstName and inv.lastName and is_valid_email(inv.email):
                valid.append(
                    BookingInvitee(
                        email=inv.email.strip(),
                        firstName=inv.firstName.strip(),
                        lastName=inv.lastName.strip(),
                    )
                )
        return valid or None

    def apply_candidate(
        self, candidate: list[BookingInvitee], bs: BookingState
    ) -> str | None:
        bs.invitees = candidate
        return None

    def present(self, bs: BookingState) -> str:
        return (
            "Would you like to invite anyone to this session (parent, guardian, friend)? "
            "Up to 6 invitees, each needs a first name, last name, and email. "
            "Reply 'no' to skip."
        )


def _is_explicit_decline(msg: str) -> bool:
    decline_phrases = {
        "no thanks", "no thank you", "no one", "noone", "nobody",
        "just me", "only me", "skip invitees",
    }
    m = msg.lower().strip()
    return any(p in m for p in decline_phrases)


# ── Material upload (required) ────────────────────────────────────────────────

class MaterialStep(StepHandler):
    """
    The material is uploaded out-of-band by the frontend. We only check the
    flag that the frontend's upload endpoint flips on `bs.material_uploaded`.
    """
    step = BookingStep.AWAITING_MATERIAL

    async def process(self, state, bs, llm):
        message = state.user_message or ""

        if is_cancel(message):
            return _cancel(bs)

        if message.strip() and not bs.material_uploaded:
            intent = await extract(self.step, message, bs, llm)
            rewind_result = await _try_rewind(state, bs, intent, llm)
            if rewind_result:
                return rewind_result

        if not bs.material_uploaded:
            return StepResult(response=self.present(bs))

        try:
            advance(bs, BookingStep.AWAITING_CONFIRMATION, trigger="material_uploaded")
        except InvalidTransitionError as exc:
            return StepResult(response=exc.user_message)

        # Status prefix only. The dispatcher loop renders ConfirmationStep.present()
        # on the next iteration; including it here would duplicate the summary.
        return StepResult(response="Materials received. ✅", advanced=True)

    def present(self, bs: BookingState) -> str:
        return (
            "Almost done — please upload your session materials using the upload button "
            "(PDF, DOC, or DOCX). I'll continue as soon as the upload is complete."
        )


# ── Confirmation ──────────────────────────────────────────────────────────────

class ConfirmationStep(StepHandler):
    step = BookingStep.AWAITING_CONFIRMATION

    async def process(self, state, bs, llm):
        message = state.user_message or ""

        if is_cancel(message):
            return _cancel(bs)

        # When reached via auto-advance (dispatcher cleared user_message after
        # the previous step consumed it), just render the summary — no LLM call.
        if not message.strip():
            return StepResult(response=self.present(bs))

        if is_confirm(message):
            return await self._finalize(state, bs)

        intent = await extract(self.step, message, bs, llm)

        rewind_result = await _try_rewind(state, bs, intent, llm)
        if rewind_result:
            return rewind_result

        if intent.action == "cancel":
            return _cancel(bs)
        if intent.action == "confirm":
            return await self._finalize(state, bs)

        return StepResult(response=self.present(bs))

    async def _finalize(self, state: WorkflowState, bs: BookingState) -> StepResult:
        if not bs.material_uploaded:
            try:
                rewind(bs, BookingStep.AWAITING_MATERIAL, trigger="material_missing_at_confirm")
            except InvalidTransitionError as exc:
                return StepResult(response=exc.user_message)
            return StepResult(response=MaterialStep().present(bs))

        if not slot_lead_time_ok(bs.date or "", bs.time_from or ""):
            try:
                rewind(bs, BookingStep.AWAITING_SLOT, trigger="lead_time_violation")
            except InvalidTransitionError as exc:
                return StepResult(response=exc.user_message)
            await SlotStep().ensure_data(state, bs)
            return StepResult(
                response=(
                    "That slot is now within 48 hours of starting and can no longer be booked. "
                    f"Please pick another.\n\n{format_slots(bs.cached_slots)}"
                )
            )

        try:
            still_available = await services.slot_still_available(
                bs.level_id or "",
                bs.subject_id or "",
                bs.curriculum_code,
                bs.tutor_id or "",
                bs.date or "",
                bs.time_from or "",
                bs.time_to or "",
            )
        except Exception as exc:
            logger.error("Slot re-validation failed", error=str(exc), exc_info=True)
            still_available = True

        if not still_available:
            try:
                rewind(bs, BookingStep.AWAITING_SLOT, trigger="slot_taken_at_confirm")
            except InvalidTransitionError as exc:
                return StepResult(response=exc.user_message)
            bs.cached_slots = []
            await SlotStep().ensure_data(state, bs)
            return StepResult(
                response=(
                    f"Sorry — that slot was just taken. Please choose another.\n\n"
                    f"{format_slots(bs.cached_slots)}"
                )
            )

        contract = build_contract(bs)
        state.booking_contract = contract
        state.want_to_nav = "session"

        try:
            advance(bs, BookingStep.COMPLETED, trigger="confirmed")
        except InvalidTransitionError as exc:
            return StepResult(response=exc.user_message)

        completion_msg = format_completion(bs)
        # Reset transient state so the next booking starts clean while preserving
        # the contract / nav already written to WorkflowState.
        reset(bs)
        return StepResult(response=completion_msg)

    def present(self, bs: BookingState) -> str:
        return format_booking_summary(bs)


# ── Registry ──────────────────────────────────────────────────────────────────

HANDLERS: dict[BookingStep, StepHandler] = {
    BookingStep.WALLET_NEEDED: WalletStep(),
    BookingStep.AWAITING_LEVEL: LevelStep(),
    BookingStep.AWAITING_SUBJECT: SubjectStep(),
    BookingStep.AWAITING_CURRICULUM: CurriculumStep(),
    BookingStep.AWAITING_TEACHER: TeacherStep(),
    BookingStep.AWAITING_SLOT: SlotStep(),
    BookingStep.AWAITING_DESCRIPTION: DescriptionStep(),
    BookingStep.AWAITING_INVITEES: InviteesStep(),
    BookingStep.AWAITING_MATERIAL: MaterialStep(),
    BookingStep.AWAITING_CONFIRMATION: ConfirmationStep(),
}
