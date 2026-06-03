"""
Thin wrappers over the MongoDB layer.

Centralises every external read so step handlers stay pure and so tests can
monkey-patch a single module instead of reaching into the DB layer.
"""

from typing import Any

from utils.apexchat.core.mongodb import (
    check_wallet_funded,
    extract_future_slots,
    fetch_all_subjects,
    fetch_curricula,
    fetch_levels,
    fetch_subjects,
    fetch_tutors,
)

from .matching import match_subject


async def is_wallet_funded(user_id: str) -> bool:
    return await check_wallet_funded(user_id)


# Levels intentionally hidden from the AI booking menu. They still exist
# platform-wide — they're just not offered as options in the booking flow.
_HIDDEN_BOOKING_LEVELS = {
    "extracurricular courses",
    "standardized tests",
    "business management",
}


def _level_en_name(level: dict[str, Any]) -> str:
    name = level.get("name")
    if isinstance(name, dict):
        return (name.get("en") or "").strip()
    return (name or "").strip()


async def list_levels() -> list[dict[str, Any]]:
    levels = await fetch_levels()
    return [
        lv for lv in levels
        if _level_en_name(lv).lower() not in _HIDDEN_BOOKING_LEVELS
    ]


async def resolve_subject_first(
    subject_query: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Resolve a user-named subject to the level(s) that offer it.

    Matches `subject_query` against every active subject, then returns the
    matched subject doc together with the level docs that offer it. Levels are
    drawn from the FULL active-level list (including ones hidden from the menu),
    so explicitly naming e.g. "robotics" can still reach its Extracurricular
    level. Returns (None, []) when the subject can't be matched.
    """
    subjects = await fetch_all_subjects()
    match = match_subject(subject_query, subjects)
    if not match:
        return None, []
    level_ids = set(match.get("levelIds", []))
    levels = [lv for lv in await fetch_levels() if str(lv["_id"]) in level_ids]
    return match, levels


async def list_subjects(level_id: str) -> list[dict[str, Any]]:
    return await fetch_subjects(level_id)


async def list_curricula(level_id: str, subject_id: str) -> list[str]:
    """
    Return only curricula that actually have a BOOKABLE teacher for this
    level+subject — i.e. at least one tutor with an active future slot.

    `fetch_curricula` lists every curriculum any matching tutor *teaches*, even
    when none of those tutors have availability. Offering those traps the user
    in a loop: they pick a curriculum, the teacher step finds no one, and bounces
    them back to the same dead-end list. Filtering here means every offered
    curriculum leads to a real teacher; when none qualify we return [] so the
    curriculum step skips the question and shows available teachers directly.
    """
    curricula = await fetch_curricula(level_id, subject_id)
    if not curricula:
        return []

    bookable: list[str] = []
    for code in curricula:
        if await fetch_tutors(level_id, subject_id, code):
            bookable.append(code)
    return bookable


async def list_tutors(
    level_id: str,
    subject_id: str,
    curriculum: str | None,
    *,
    at_time: str | None = None,
    at_date: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return tutors for the given level/subject(/curriculum).

    When `at_time` (HH:mm, UTC) and/or `at_date` (YYYY-MM-DD) are supplied — e.g.
    the user said "find an English teacher at 1pm" — the list is narrowed to
    tutors who have at least one ACTIVE future slot starting at that time/date.
    With no filter, every available tutor is returned (original behaviour).
    """
    tutors = await fetch_tutors(level_id, subject_id, curriculum)
    if not at_time and not at_date:
        return tutors

    filtered: list[dict[str, Any]] = []
    for t in tutors:
        slots = extract_future_slots(t)
        if any(
            (at_time is None or s["timeFrom"] == at_time)
            and (at_date is None or s["date"] == at_date)
            for s in slots
        ):
            filtered.append(t)
    return filtered


async def list_tutor_slots(
    level_id: str,
    subject_id: str,
    curriculum: str | None,
    tutor_id: str,
) -> list[dict[str, Any]]:
    """
    Return the named tutor's future ACTIVE slots.

    Re-fetches the tutor doc rather than relying on a stale cached entry — used
    both when first showing the slot list AND when re-validating the chosen slot
    at confirmation time.
    """
    tutors = await fetch_tutors(level_id, subject_id, curriculum)
    tutor_doc = next((t for t in tutors if t["_id"] == tutor_id), None)
    if not tutor_doc:
        return []
    return extract_future_slots(tutor_doc)


async def slot_still_available(
    level_id: str,
    subject_id: str,
    curriculum: str | None,
    tutor_id: str,
    date: str,
    time_from: str,
    time_to: str,
) -> bool:
    """Re-check a single slot's availability immediately before contract emission."""
    slots = await list_tutor_slots(level_id, subject_id, curriculum, tutor_id)
    return any(
        s["date"] == date and s["timeFrom"] == time_from and s["timeTo"] == time_to
        for s in slots
    )
