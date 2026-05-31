"""User-facing message formatters. Pure functions, no I/O."""

from collections import defaultdict
from datetime import datetime

from utils.apexchat.schemas.models import BookingState

from .matching import (
    CURRICULUM_LABELS,
    _level_display_name,
    _subject_display_name,
    _tutor_display_name,
)


def curriculum_label(code: str | None) -> str:
    if not code:
        return ""
    return CURRICULUM_LABELS.get(code, code)


def format_levels(levels: list[dict]) -> str:
    if not levels:
        return ""
    lines = []
    for i, lv in enumerate(levels, 1):
        name = _level_display_name(lv)
        lines.append(f"  **{i}.** {name}")
    return "\n".join(lines)


def format_subjects(subjects: list[dict]) -> str:
    if not subjects:
        return ""
    lines = []
    for i, s in enumerate(subjects, 1):
        lines.append(f"  **{i}.** {_subject_display_name(s)}")
    return "\n".join(lines)


def format_curricula(codes: list[str]) -> str:
    if not codes:
        return ""
    return "\n".join(
        f"  **{i}.** {curriculum_label(c)} ({c})" for i, c in enumerate(codes, 1)
    )


def format_tutors(tutors: list[dict]) -> str:
    if not tutors:
        return ""
    lines = []
    for i, t in enumerate(tutors, 1):
        name = _tutor_display_name(t)
        bio = (t.get("bio") or "").strip()
        bio_str = f" — {bio[:120]}" if bio else ""
        degree = t.get("degree") or ""
        degree_str = f" [{degree}]" if degree else ""
        lines.append(f"  **{i}.** {name}{degree_str}{bio_str}")
    return "\n".join(lines)


def format_slots(slots: list[dict]) -> str:
    if not slots:
        return ""
    by_date: dict[str, list] = defaultdict(list)
    ordering: list[tuple[str, dict]] = []
    for s in slots:
        ordering.append((s["date"], s))
    by_date_keys = sorted({d for d, _ in ordering})

    lines: list[str] = []
    idx = 1
    for date_str in by_date_keys:
        try:
            label = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A, %B %d, %Y").replace(" 0", " ")
        except ValueError:
            label = date_str
        lines.append(f"  **{label}**")
        for s in [s for d, s in ordering if d == date_str]:
            lines.append(f"    **{idx}.** {s['timeFrom']} – {s['timeTo']} UTC")
            idx += 1
    return "\n".join(lines)


def format_booking_summary(bs: BookingState) -> str:
    invitees_line = "None"
    if bs.invitees:
        invitees_line = ", ".join(
            f"{inv.firstName} {inv.lastName} ({inv.email})" for inv in bs.invitees
        )
    return (
        "Here's your booking summary:\n\n"
        f"- **Level:** {bs.level_name}\n"
        f"- **Subject:** {bs.subject_name}\n"
        f"- **Curriculum:** {curriculum_label(bs.curriculum_code) or '—'}\n"
        f"- **Teacher:** {bs.tutor_name}\n"
        f"- **Date:** {bs.date}\n"
        f"- **Time:** {bs.time_from} – {bs.time_to} UTC\n"
        f"- **Focus:** {bs.description or '—'}\n"
        f"- **Invitees:** {invitees_line}\n"
        f"- **Materials uploaded:** {'yes' if bs.material_uploaded else 'no'}\n\n"
        "Type **yes** to confirm or **no** to cancel."
    )


def format_completion(bs: BookingState) -> str:
    invitees_line = "None"
    if bs.invitees:
        invitees_line = ", ".join(
            f"{inv.firstName} {inv.lastName} ({inv.email})" for inv in bs.invitees
        )
    focus = bs.description or f"{bs.level_name} {bs.subject_name} session"
    return (
        "You're all set! Your session is booked.\n\n"
        "Booking details:\n"
        f"- Subject: {bs.subject_name}\n"
        f"- Level: {bs.level_name}\n"
        f"- Curriculum: {curriculum_label(bs.curriculum_code) or '—'}\n"
        f"- Teacher: {bs.tutor_name}\n"
        f"- Date: {bs.date}\n"
        f"- Time: {bs.time_from} – {bs.time_to} UTC\n"
        f"- Focus: {focus}\n"
        f"- Invitees: {invitees_line}"
    )
