"""
Single-pass extractor for messages that name multiple booking fields at once.

When the user says "I want secondary, English, French curriculum" the per-step
extractor (extractor.py) only reads the field for the current step and the
remaining values are lost. This module is invoked once per turn, returns every
field the user mentioned, and stores them on `bs.pending_hints` so each
subsequent step can pop and apply its own value as the dispatcher walks the
canonical step order.

The matching itself stays in matching.py — this module only extracts user-typed
references; each step still validates the hint against its cached options and
stops the chain if the hint cannot be resolved.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from utils.apexchat.core.llm import LLMClient
from utils.apexchat.schemas.models import BookingState, BookingStep

from .matching import CURRICULUM_LABELS

logger = structlog.get_logger(__name__)


# ── Hint keys (stable strings used as keys in `bs.pending_hints`) ─────────────
# Each step handler in steps.py reads the key matching its step.
HINT_LEVEL = "level"
HINT_SUBJECT = "subject"
HINT_CURRICULUM = "curriculum"
HINT_TEACHER = "teacher"
HINT_SLOT = "slot"
HINT_DESCRIPTION = "description"
HINT_INVITEES = "invitees"

# Sentinel placed in hints[key] when the user explicitly skips an optional step.
SKIP = "__skip__"


class MultiInvitee(BaseModel):
    email: str = ""
    firstName: str = ""
    lastName: str = ""


class MultiStepIntent(BaseModel):
    """
    Every field detected in a single user message. All fields optional — only
    the ones the user actually mentioned are filled.
    """

    level_query: str | None = Field(
        default=None,
        description=(
            "User's reference to an academic level, copied verbatim "
            "(e.g. 'secondary', 'grade 9', 'university prep'). Null if not mentioned."
        ),
    )
    subject_query: str | None = Field(
        default=None,
        description="User's reference to a subject (e.g. 'English', 'Math'). Null if not mentioned.",
    )
    curriculum_query: str | None = Field(
        default=None,
        description=(
            "User's reference to a curriculum (e.g. 'French', 'British', 'UK', 'NC'). "
            "May be a code or a label. Null if not mentioned."
        ),
    )
    teacher_query: str | None = Field(
        default=None,
        description=(
            "The PROPER NAME of a specific teacher/tutor the user named "
            "(e.g. 'Bob', 'Mr. Smith', 'Sarah Khoury'). Null unless an actual "
            "person's name is given. Do NOT fill this from generic role phrases "
            "like 'a good arabic teacher', 'a math tutor', or 'any teacher' — the "
            "subject in those belongs in subject_query and teacher_query stays null."
        ),
    )

    slot_date: str | None = Field(
        default=None,
        description="YYYY-MM-DD format. Only when the user clearly named a date.",
    )
    slot_time: str | None = Field(
        default=None,
        description="HH:mm 24h UTC. Only when the user clearly named a start time.",
    )
    slot_index: int | None = Field(
        default=None,
        description="1-based index when the user said 'first slot', 'option 2', etc.",
    )

    description: str | None = Field(
        default=None,
        description="Free-form session focus content (a topic, chapter, exam, skill).",
    )
    skip_description: bool = Field(
        default=False,
        description="True when user explicitly declines to provide a focus ('skip', 'no preference').",
    )

    invitees: list[MultiInvitee] | None = Field(
        default=None,
        description="Up to 6 invitees with full email + firstName + lastName.",
    )
    skip_invitees: bool = Field(
        default=False,
        description="True when user declines to invite anyone ('no thanks', 'just me').",
    )

    cancel: bool = Field(
        default=False,
        description="True when user explicitly aborts the booking ('cancel', 'never mind').",
    )
    rewind_to: BookingStep | None = Field(
        default=None,
        description=(
            "When the user wants to revisit an earlier step. One of: "
            "awaiting_level, awaiting_subject, awaiting_curriculum, "
            "awaiting_teacher, awaiting_slot, awaiting_description, awaiting_invitees."
        ),
    )


# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT = """You are a booking-flow extractor. The user is booking a tutoring
session and may name several fields in a single message — extract EVERY field
they mention at once.

Output ONLY structured JSON matching the MultiStepIntent schema. Never invent
values the user did not explicitly provide. Leave a field null when it is not
clearly named.

Today's date (UTC): {today}.

Already collected so far: {summary}.

Available curriculum codes and labels:
{curriculum_labels}

Rules:
  - Copy the user's wording verbatim into *_query fields. Each step will match
    those against its own option list — do not normalise or pick an index.
  - For slot_date / slot_time: only fill them if the user wrote an explicit date
    or HH:mm time. Vague references ("any morning") stay null.
  - teacher_query: ONLY a specific person's name. Never a generic phrase such as
    "a good arabic teacher" or "a math tutor" — the subject in those goes to
    subject_query and teacher_query stays null.
  - skip_description / skip_invitees default false; flip them only on explicit
    decline.
  - cancel = true only on a clear abort ("cancel", "never mind", "stop").
  - rewind_to: only when the user wants to change a *previous* choice.
"""


def _summary(bs: BookingState) -> str:
    parts: list[str] = []
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


def _curriculum_labels() -> str:
    return "\n".join(f"  - {code}: {label}" for code, label in CURRICULUM_LABELS.items())


# ── Triggering heuristic ──────────────────────────────────────────────────────

# Cheap word-level signals that the message names more than one booking field.
# When none fire we skip the LLM call and let per-step extractors handle the
# turn — saves latency on simple inputs like "1", "yes", or "Math".
_MULTI_KEYWORDS = (
    "level", "subject", "curriculum", "teacher", "tutor", "slot", "session",
    "english", "french", "british", "american", "lebanese", "national",
    "primary", "secondary", "elementary", "intermediate", "university",
    "math", "physics", "chemistry", "biology", "history", "arabic",
)


# Role words that, on their own, signal a booking request that names a field
# (e.g. "find me a robotics teacher", "I want a chemistry tutor"). The subject
# can be anything in the catalogue, so we don't rely on a fixed subject list to
# decide whether to run the extractor — any of these is enough.
_ROLE_WORDS = ("teacher", "tutor", "course", "class", "lesson", "learn", "study")


def looks_multi_step(message: str) -> bool:
    """
    True when the message is worth running through the multi-extractor.

    Fires on at least 4 words AND any of: a comma, a conjunction, a booking
    role word (teacher/tutor/course/…), or two recognised subject keywords.
    The role-word check lets "find me a robotics teacher" trigger extraction
    even though "robotics" isn't in the static subject keyword list.
    """
    msg = (message or "").strip().lower()
    if len(msg.split()) < 4:
        return False

    if "," in msg:
        return True
    if " and " in msg or " with " in msg or " plus " in msg:
        return True

    if any(w in msg for w in _ROLE_WORDS):
        return True

    matched = sum(1 for kw in _MULTI_KEYWORDS if kw in msg)
    return matched >= 2


# ── Public API ────────────────────────────────────────────────────────────────

async def extract_multi_step(
    message: str,
    bs: BookingState,
    llm: LLMClient,
) -> MultiStepIntent:
    """
    Run the LLM with a prompt covering every booking field and return the
    structured intent. Returns an empty MultiStepIntent on any failure so
    callers can safely fall back to per-step extraction.
    """
    structured = llm.with_structured_output(MultiStepIntent)
    system = _PROMPT.format(
        today=_today(),
        summary=_summary(bs),
        curriculum_labels=_curriculum_labels(),
    )
    messages = [SystemMessage(content=system), HumanMessage(content=message)]
    try:
        return await structured.ainvoke(messages)
    except Exception as exc:
        logger.warning("Multi-step extraction failed", error=str(exc))
        return MultiStepIntent()


def intent_to_hints(intent: MultiStepIntent) -> dict[str, Any]:
    """
    Convert a MultiStepIntent into the `bs.pending_hints` dict that step
    handlers consume. Only keys the user actually filled appear in the result.
    """
    hints: dict[str, Any] = {}

    if intent.level_query:
        hints[HINT_LEVEL] = intent.level_query.strip()
    if intent.subject_query:
        hints[HINT_SUBJECT] = intent.subject_query.strip()
    if intent.curriculum_query:
        hints[HINT_CURRICULUM] = intent.curriculum_query.strip()
    if intent.teacher_query:
        hints[HINT_TEACHER] = intent.teacher_query.strip()

    slot_payload: dict[str, Any] = {}
    if intent.slot_date:
        slot_payload["date"] = intent.slot_date
    if intent.slot_time:
        slot_payload["time"] = intent.slot_time
    if intent.slot_index:
        slot_payload["index"] = intent.slot_index
    if slot_payload:
        hints[HINT_SLOT] = slot_payload

    if intent.description:
        hints[HINT_DESCRIPTION] = intent.description.strip()
    elif intent.skip_description:
        hints[HINT_DESCRIPTION] = SKIP

    if intent.invitees:
        hints[HINT_INVITEES] = [inv.model_dump() for inv in intent.invitees]
    elif intent.skip_invitees:
        hints[HINT_INVITEES] = SKIP

    return hints
