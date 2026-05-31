"""
Booking state machine.

Single source of truth for `BookingState.step` mutations. No other module
in the booking package writes `bs.step` directly — everything goes through
`advance()` or `rewind()` here so transitions are validated and logged.

Forward order (the "happy path"):
    IDLE
      → WALLET_NEEDED  (only when wallet is empty)
      → AWAITING_LEVEL
      → AWAITING_SUBJECT
      → AWAITING_CURRICULUM   (skipped when level.requiresCurriculum is False)
      → AWAITING_TEACHER
      → AWAITING_SLOT
      → AWAITING_DESCRIPTION
      → AWAITING_INVITEES
      → AWAITING_MATERIAL
      → AWAITING_CONFIRMATION
      → COMPLETED

Rewind: any earlier step in the sequence is reachable. All collected fields
strictly downstream of the rewind target are wiped, plus their cached lists.
"""

from datetime import datetime
from typing import Any

import structlog

from utils.apexchat.schemas.models import (
    BookingState,
    BookingStep,
    BookingTransitionEvent,
)

from .errors import InvalidTransitionError

logger = structlog.get_logger(__name__)

# Linear order. Index in this list defines which steps are upstream/downstream.
BOOKING_STEP_ORDER: list[BookingStep] = [
    BookingStep.IDLE,
    BookingStep.WALLET_NEEDED,
    BookingStep.AWAITING_LEVEL,
    BookingStep.AWAITING_SUBJECT,
    BookingStep.AWAITING_CURRICULUM,
    BookingStep.AWAITING_TEACHER,
    BookingStep.AWAITING_SLOT,
    BookingStep.AWAITING_DESCRIPTION,
    BookingStep.AWAITING_INVITEES,
    BookingStep.AWAITING_MATERIAL,
    BookingStep.AWAITING_CONFIRMATION,
    BookingStep.COMPLETED,
]

_STEP_INDEX: dict[BookingStep, int] = {s: i for i, s in enumerate(BOOKING_STEP_ORDER)}

# Steps the user can rewind to from a later step. (WALLET_NEEDED and the two
# terminal steps are not user-rewindable destinations.)
REWINDABLE_STEPS: frozenset[BookingStep] = frozenset({
    BookingStep.AWAITING_LEVEL,
    BookingStep.AWAITING_SUBJECT,
    BookingStep.AWAITING_CURRICULUM,
    BookingStep.AWAITING_TEACHER,
    BookingStep.AWAITING_SLOT,
    BookingStep.AWAITING_DESCRIPTION,
    BookingStep.AWAITING_INVITEES,
    BookingStep.AWAITING_MATERIAL,
})

# Fields that belong to each step. When the user rewinds to step X, every
# field listed for steps strictly after X is cleared.
_STEP_FIELDS: dict[BookingStep, list[str]] = {
    BookingStep.AWAITING_LEVEL: ["level_id", "level_name", "requires_curriculum"],
    BookingStep.AWAITING_SUBJECT: ["subject_id", "subject_name"],
    BookingStep.AWAITING_CURRICULUM: ["curriculum_code"],
    BookingStep.AWAITING_TEACHER: ["tutor_id", "tutor_name"],
    BookingStep.AWAITING_SLOT: ["date", "time_from", "time_to", "slot_lock_expires_at"],
    BookingStep.AWAITING_DESCRIPTION: ["description"],
    BookingStep.AWAITING_INVITEES: ["invitees"],
    BookingStep.AWAITING_MATERIAL: ["material_uploaded"],
}

# Cached option lists tied to each step. Cleared alongside their step's fields.
_STEP_CACHES: dict[BookingStep, list[str]] = {
    BookingStep.AWAITING_SUBJECT: ["cached_subjects"],
    BookingStep.AWAITING_CURRICULUM: ["cached_curricula"],
    BookingStep.AWAITING_TEACHER: ["cached_tutors"],
    BookingStep.AWAITING_SLOT: ["cached_slots"],
}

# Per-step empty-value defaults used when wiping.
_FIELD_DEFAULTS: dict[str, Any] = {
    "level_id": None,
    "level_name": None,
    "requires_curriculum": True,
    "subject_id": None,
    "subject_name": None,
    "curriculum_code": None,
    "tutor_id": None,
    "tutor_name": None,
    "date": None,
    "time_from": None,
    "time_to": None,
    "description": None,
    "invitees": [],
    "material_uploaded": False,
    "slot_lock_expires_at": None,
}


def step_index(step: BookingStep) -> int:
    return _STEP_INDEX[step]


def is_terminal(step: BookingStep) -> bool:
    """IDLE and COMPLETED release the workflow flow-lock."""
    return step in (BookingStep.IDLE, BookingStep.COMPLETED)


def next_step(current: BookingStep, *, requires_curriculum: bool) -> BookingStep:
    """
    Return the next step in the canonical order, applying the only legal
    explicit skip: AWAITING_SUBJECT → AWAITING_TEACHER when no curriculum
    is required.
    """
    if current == BookingStep.AWAITING_SUBJECT and not requires_curriculum:
        return BookingStep.AWAITING_TEACHER

    idx = _STEP_INDEX[current]
    if idx + 1 >= len(BOOKING_STEP_ORDER):
        return current
    return BOOKING_STEP_ORDER[idx + 1]


def advance(
    bs: BookingState,
    target: BookingStep,
    *,
    trigger: str,
    detail: dict | None = None,
) -> None:
    """
    Forward-only transition. Verifies `target` is the canonical next step
    given current state. Raises InvalidTransitionError if a different step
    is requested (no jumping ahead).
    """
    expected = next_step(bs.step, requires_curriculum=bs.requires_curriculum)
    if target not in (expected, BookingStep.WALLET_NEEDED, BookingStep.COMPLETED):
        # Allow IDLE → AWAITING_LEVEL even when WALLET_NEEDED is in between.
        idle_to_level = bs.step == BookingStep.IDLE and target == BookingStep.AWAITING_LEVEL
        if not idle_to_level:
            raise InvalidTransitionError(bs.step, target)

    skipped_curriculum = (
        bs.step == BookingStep.AWAITING_SUBJECT
        and target == BookingStep.AWAITING_TEACHER
        and not bs.requires_curriculum
    )

    bs.last_step_completed = bs.step if bs.step != BookingStep.IDLE else bs.last_step_completed
    _record(bs, bs.step, target, trigger=trigger, detail=detail or {})
    bs.step = target

    if skipped_curriculum:
        _record(
            bs,
            BookingStep.AWAITING_CURRICULUM,
            BookingStep.AWAITING_CURRICULUM,
            trigger="skip_curriculum",
            detail={"reason": "level.requiresCurriculum=false"},
        )


def rewind(
    bs: BookingState,
    target: BookingStep,
    *,
    trigger: str,
    detail: dict | None = None,
) -> None:
    """
    Move to an earlier step and wipe every field downstream of the target.
    Raises InvalidTransitionError if the target is not strictly earlier.
    """
    if target not in REWINDABLE_STEPS:
        raise InvalidTransitionError(bs.step, target)

    target_idx = _STEP_INDEX[target]
    current_idx = _STEP_INDEX[bs.step]
    if target_idx >= current_idx:
        raise InvalidTransitionError(bs.step, target)

    invalidate_dependents(bs, target)
    # Rewinding to a step means the user wants to redo it — wipe its own
    # collected fields too (date/time, description, invitees, ...). Cached
    # option lists for the target step are kept so its presenter can render
    # without an extra fetch.
    for f in _STEP_FIELDS.get(target, []):
        setattr(bs, f, _FIELD_DEFAULTS.get(f))
    _record(bs, bs.step, target, trigger=trigger, detail=detail or {"rewind": True})
    bs.step = target


def invalidate_dependents(bs: BookingState, target: BookingStep) -> None:
    """
    Wipe every field and cache belonging to a step strictly after `target`.
    Public so steps that change a foundational selection (e.g. teacher) can
    purge stale caches without doing a full rewind.
    """
    target_idx = _STEP_INDEX[target]
    for step, fields in _STEP_FIELDS.items():
        if _STEP_INDEX[step] > target_idx:
            for f in fields:
                setattr(bs, f, _FIELD_DEFAULTS.get(f))
    for step, caches in _STEP_CACHES.items():
        if _STEP_INDEX[step] > target_idx:
            for c in caches:
                setattr(bs, c, [])


def reset(bs: BookingState) -> None:
    """Wipe the whole booking state back to a clean IDLE — used on cancel/restart."""
    fresh = BookingState()
    for field in fresh.model_fields:
        setattr(bs, field, getattr(fresh, field))


def _record(
    bs: BookingState,
    from_step: BookingStep,
    to_step: BookingStep,
    *,
    trigger: str,
    detail: dict,
) -> None:
    event = BookingTransitionEvent(
        ts=datetime.utcnow(),
        from_step=from_step,
        to_step=to_step,
        trigger=trigger,
        detail=detail,
    )
    bs.step_history.append(event)
    logger.info(
        "Booking transition",
        from_step=from_step.value,
        to_step=to_step.value,
        trigger=trigger,
        **{k: v for k, v in detail.items() if isinstance(v, (str, int, float, bool))},
    )
