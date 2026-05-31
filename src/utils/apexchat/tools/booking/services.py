"""
Thin wrappers over the MongoDB layer.

Centralises every external read so step handlers stay pure and so tests can
monkey-patch a single module instead of reaching into the DB layer.
"""

from typing import Any

from utils.apexchat.core.mongodb import (
    check_wallet_funded,
    extract_future_slots,
    fetch_curricula,
    fetch_levels,
    fetch_subjects,
    fetch_tutors,
)


async def is_wallet_funded(user_id: str) -> bool:
    return await check_wallet_funded(user_id)


async def list_levels() -> list[dict[str, Any]]:
    return await fetch_levels()


async def list_subjects(level_id: str) -> list[dict[str, Any]]:
    return await fetch_subjects(level_id)


async def list_curricula(level_id: str, subject_id: str) -> list[str]:
    return await fetch_curricula(level_id, subject_id)


async def list_tutors(level_id: str, subject_id: str, curriculum: str | None) -> list[dict[str, Any]]:
    return await fetch_tutors(level_id, subject_id, curriculum)


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
