"""
Single LLM extractor used as a fallback by every step.

Deterministic matchers (matching.py) cover ~90% of turns. When the user's
message is too free-form for those (e.g. "the British one for next Friday
afternoon", invitee names + emails, custom session focus), this module is
called to extract a structured StepIntent.

The same `StepIntent` schema is used across steps; each step picks only the
fields it needs. A focused prompt per step keeps the LLM output narrow.
"""

from datetime import datetime, timezone
from typing import Literal

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from utils.apexchat.core.llm import LLMClient
from utils.apexchat.schemas.models import BookingState, BookingStep

from .matching import CURRICULUM_LABELS
from .presenters import (
    format_curricula,
    format_levels,
    format_slots,
    format_subjects,
    format_tutors,
)

logger = structlog.get_logger(__name__)


class InviteePayload(BaseModel):
    email: str = ""
    firstName: str = ""
    lastName: str = ""


class StepIntent(BaseModel):
    """
    LLM extraction result. All fields are optional — each step uses only the
    subset relevant to its prompt.
    """

    action: Literal["select", "rewind", "cancel", "confirm", "skip", "none"] = Field(
        default="none",
        description=(
            "What the user is trying to do. 'select' = picking an option for the "
            "current step. 'rewind' = wants to change a previous step. 'cancel' "
            "= abort booking. 'confirm' = says yes to the summary. 'skip' = "
            "explicitly declines an optional step. 'none' = unclear / ambiguous."
        ),
    )

    selected_index: int | None = Field(
        default=None,
        description="1-based index into the option list shown for this step, when applicable.",
    )

    selected_curriculum: str | None = Field(
        default=None,
        description="One of: US, UK, FR, NC. Only used by the curriculum step.",
    )

    selected_date: str | None = Field(
        default=None, description="YYYY-MM-DD. Only used by the slot step."
    )
    selected_time_from: str | None = Field(
        default=None, description="HH:mm 24h UTC. Only used by the slot step."
    )

    free_text: str | None = Field(
        default=None,
        description="User-provided session focus / description. Only used by the description step.",
    )

    invitees: list[InviteePayload] | None = Field(
        default=None,
        description=(
            "Up to 6 invitees with full email + firstName + lastName. "
            "Only used by the invitees step."
        ),
    )

    rewind_to: BookingStep | None = Field(
        default=None,
        description=(
            "When action='rewind', the step to go back to. Must be one of: "
            "awaiting_level, awaiting_subject, awaiting_curriculum, "
            "awaiting_teacher, awaiting_slot, awaiting_description, awaiting_invitees."
        ),
    )


# ── Prompt templates ──────────────────────────────────────────────────────────

_BASE = """You are an extraction assistant for a tutoring-session booking flow.

Output ONLY structured JSON matching the StepIntent schema. Never invent values
the user did not provide. Set action='none' if you cannot determine intent.

Today's date (UTC): {today}.

Current step: {step}
Already collected: {summary}
"""

_REWIND_HINT = """
If the user wants to change an earlier choice, set action='rewind' and rewind_to
to the appropriate step value. Allowed rewind targets:
  awaiting_level, awaiting_subject, awaiting_curriculum, awaiting_teacher,
  awaiting_slot, awaiting_description, awaiting_invitees.

If the user wants to abort, set action='cancel'.
"""


def _summary(bs: BookingState) -> str:
    parts = []
    if bs.level_name:
        parts.append(f"level={bs.level_name}")
    if bs.subject_name:
        parts.append(f"subject={bs.subject_name}")
    if bs.curriculum_code:
        parts.append(f"curriculum={bs.curriculum_code}")
    if bs.tutor_name:
        parts.append(f"teacher={bs.tutor_name}")
    if bs.date and bs.time_from:
        parts.append(f"slot={bs.date} {bs.time_from}-{bs.time_to}")
    if bs.description:
        parts.append(f"focus={bs.description}")
    return ", ".join(parts) if parts else "nothing yet"


def _today() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _build_prompt(step: BookingStep, bs: BookingState) -> str:
    base = _BASE.format(today=_today(), step=step.value, summary=_summary(bs))

    if step == BookingStep.AWAITING_LEVEL:
        return base + f"""
Pick a level. The numbered list shown to the user:
{format_levels(bs.cached_levels)}

Set action='select' and selected_index when the user clearly picks one.
{_REWIND_HINT}"""

    if step == BookingStep.AWAITING_SUBJECT:
        return base + f"""
Pick a subject. The numbered list shown to the user:
{format_subjects(bs.cached_subjects)}

Set action='select' and selected_index when the user clearly picks one.
{_REWIND_HINT}"""

    if step == BookingStep.AWAITING_CURRICULUM:
        labels = "\n".join(
            f"  - {c}: {CURRICULUM_LABELS.get(c, c)}" for c in bs.cached_curricula
        )
        return base + f"""
Pick a curriculum code from: {', '.join(bs.cached_curricula)}.
{labels}

Set action='select' and selected_curriculum to one of those codes.
{_REWIND_HINT}"""

    if step == BookingStep.AWAITING_TEACHER:
        return base + f"""
Pick a teacher. The numbered list shown to the user:
{format_tutors(bs.cached_tutors)}

Set action='select' and selected_index when the user clearly picks one.
{_REWIND_HINT}"""

    if step == BookingStep.AWAITING_SLOT:
        return base + f"""
Pick a time slot. The numbered list shown to the user:
{format_slots(bs.cached_slots)}

You may set:
  - selected_index = the slot number, OR
  - selected_date and selected_time_from = exact values matching one slot.

Do NOT guess if the user is vague ("any time", "asap"). Set action='none' instead.
{_REWIND_HINT}"""

    if step == BookingStep.AWAITING_DESCRIPTION:
        return base + """
The user is being asked what to focus on for the session.

  - If they provide actual focus content (a topic, chapter, exam, skill):
    set action='select' and free_text to a cleaned-up version.
  - If they say skip / no preference / "anything":
    set action='skip'.
  - Otherwise set action='none'.
""" + _REWIND_HINT

    if step == BookingStep.AWAITING_INVITEES:
        return base + """
The user is being asked whether to invite anyone (max 6, optional).

  - If they decline: set action='skip'.
  - If they provide one or more invitees with full email + firstName + lastName:
    set action='select' and invitees to the parsed list.
  - If they want invitees but haven't given details yet: set action='none'.

Never invent emails or names.
""" + _REWIND_HINT

    if step == BookingStep.AWAITING_MATERIAL:
        return base + """
The user is being asked to upload session materials via the upload button.
Materials arrive out-of-band — there is nothing to extract from a typed message
beyond rewind / cancel. Default to action='none'.
""" + _REWIND_HINT

    if step == BookingStep.AWAITING_CONFIRMATION:
        return base + """
The user is being shown a booking summary and asked to confirm.

  - yes / confirm => action='confirm'.
  - no / cancel   => action='cancel'.
  - "change X"    => action='rewind' with rewind_to set.
  - Otherwise     => action='none'.
"""

    return base + _REWIND_HINT


async def extract(
    step: BookingStep,
    user_message: str,
    bs: BookingState,
    llm: LLMClient,
) -> StepIntent:
    """
    Run the LLM with the step's focused prompt and return the structured intent.
    Falls back to action='none' on any error so callers can re-prompt the user.
    """
    structured = llm.with_structured_output(StepIntent)
    messages = [
        SystemMessage(content=_build_prompt(step, bs)),
        HumanMessage(content=user_message),
    ]
    try:
        return await structured.ainvoke(messages)
    except Exception as exc:
        logger.warning(
            "Booking LLM extraction failed",
            step=step.value,
            error=str(exc),
        )
        return StepIntent(action="none")
