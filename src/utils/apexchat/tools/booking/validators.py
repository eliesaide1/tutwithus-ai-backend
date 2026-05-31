"""Pure validation helpers — no I/O, no LLM, no state mutation."""

import re
from datetime import datetime, timezone

# Minimum lead time before a booked session may start (hours).
MIN_LEAD_HOURS = 48

DATE_RE = re.compile(r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$")
TIME_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
OBJECT_ID_RE = re.compile(r"^[0-9a-fA-F]{24}$")


def is_valid_date(value: str | None) -> bool:
    return bool(value and DATE_RE.fullmatch(value))


def is_valid_time(value: str | None) -> bool:
    return bool(value and TIME_RE.fullmatch(value))


def is_valid_email(value: str | None) -> bool:
    return bool(value and EMAIL_RE.fullmatch(value))


def is_valid_object_id(value: str | None) -> bool:
    return bool(value and OBJECT_ID_RE.fullmatch(value))


def is_valid_curriculum_code(value: str | None) -> bool:
    return value in {"US", "UK", "FR", "NC"}


def slot_lead_time_ok(date: str, time_from: str, *, min_hours: int = MIN_LEAD_HOURS) -> bool:
    """True iff (date, time_from) is at least `min_hours` from now (UTC)."""
    if not (is_valid_date(date) and is_valid_time(time_from)):
        return False
    try:
        slot_dt = datetime.strptime(f"{date} {time_from}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    delta = slot_dt - datetime.now(tz=timezone.utc)
    return delta.total_seconds() >= min_hours * 3600
