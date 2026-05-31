"""
Deterministic matchers — try cheap, exact rules before any LLM call.

Every step's `deterministic_match` method calls into here. Returns a typed
result the step can apply directly, or None if the input is ambiguous and
the LLM extractor should be used.
"""

import re
from difflib import SequenceMatcher

CURRICULUM_LABELS: dict[str, str] = {
    "US": "American (US) Curriculum",
    "UK": "British Curriculum",
    "FR": "French Curriculum",
    "NC": "Lebanese / National Curriculum",
}

CONFIRM_KW = frozenset({
    "yes", "y", "confirm", "confirmed", "ok", "okay", "sure", "proceed",
    "correct", "go ahead", "looks good", "let's go", "do it", "book it",
    "yep", "yup", "absolutely", "perfect", "that's right",
})

CANCEL_KW = frozenset({
    "cancel", "abort", "stop", "discard", "nevermind", "never mind",
    "forget it", "don't book", "do not book",
})

SKIP_KW = frozenset({
    "skip", "no preference", "no pref", "doesn't matter", "doesnt matter",
    "anything", "you decide", "no thanks", "no thank you",
    "no", "nope", "none", "nothing", "n/a", "na",
})

ORDINAL_WORDS = {
    "first": 1, "1st": 1, "one": 1,
    "second": 2, "2nd": 2, "two": 2,
    "third": 3, "3rd": 3, "three": 3,
    "fourth": 4, "4th": 4, "four": 4,
    "fifth": 5, "5th": 5, "five": 5,
    "sixth": 6, "6th": 6, "six": 6,
    "seventh": 7, "7th": 7, "seven": 7,
    "eighth": 8, "8th": 8, "eight": 8,
    "ninth": 9, "9th": 9, "nine": 9,
    "tenth": 10, "10th": 10, "ten": 10,
}


def normalize(text: str | None) -> str:
    return (text or "").strip().lower()


def is_cancel(text: str) -> bool:
    t = normalize(text)
    return any(re.search(rf"\b{re.escape(kw)}\b", t) for kw in CANCEL_KW)


def is_confirm(text: str) -> bool:
    t = normalize(text)
    return any(re.search(rf"\b{re.escape(kw)}\b", t) for kw in CONFIRM_KW)


def is_skip(text: str) -> bool:
    t = normalize(text)
    return any(re.search(rf"\b{re.escape(kw)}\b", t) for kw in SKIP_KW)


def parse_index(text: str, list_size: int) -> int | None:
    """
    Return a 1-based index when the message is a clean numeric / ordinal pick.
    Returns None for anything ambiguous (multiple numbers, embedded in prose, etc.).
    """
    t = normalize(text)
    if not t:
        return None

    digits = re.findall(r"\d+", t)
    if len(digits) == 1:
        try:
            n = int(digits[0])
            if 1 <= n <= list_size:
                return n
        except ValueError:
            pass

    for word, n in ORDINAL_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", t) and 1 <= n <= list_size:
            return n

    return None


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _entry_names(entry: dict, *fields: str) -> list[str]:
    """Pull a list of candidate display names from a dict entry."""
    names: list[str] = []
    for field in fields:
        val = entry.get(field)
        if isinstance(val, dict):
            for v in val.values():
                if isinstance(v, str) and v.strip():
                    names.append(v.strip())
        elif isinstance(val, str) and val.strip():
            names.append(val.strip())
    return names


def _level_display_name(entry: dict) -> str:
    name = entry.get("name", {})
    if isinstance(name, dict):
        return name.get("en") or name.get("ar") or "Unknown"
    return name or "Unknown"


def _subject_display_name(entry: dict) -> str:
    return _level_display_name(entry)


def _tutor_display_name(entry: dict) -> str:
    return f"{entry.get('firstName', '').strip()} {entry.get('lastName', '').strip()}".strip()


def match_level(text: str, levels: list[dict]) -> dict | None:
    """Match by index → exact name → fuzzy name. Returns the entry or None."""
    idx = parse_index(text, len(levels))
    if idx is not None:
        return levels[idx - 1]
    return _fuzzy_pick(text, levels, lambda lv: _entry_names(lv, "name"))


def match_subject(text: str, subjects: list[dict]) -> dict | None:
    idx = parse_index(text, len(subjects))
    if idx is not None:
        return subjects[idx - 1]
    return _fuzzy_pick(text, subjects, lambda s: _entry_names(s, "name"))


def match_curriculum(text: str, codes: list[str]) -> str | None:
    t = normalize(text)
    if not t or not codes:
        return None

    idx = parse_index(text, len(codes))
    if idx is not None:
        return codes[idx - 1]

    upper = t.upper()
    for code in codes:
        if upper == code or re.search(rf"\b{code}\b", upper):
            return code

    best, best_score = None, 0.0
    for code in codes:
        label = CURRICULUM_LABELS.get(code, code)
        score = _similarity(t, label.lower())
        if t in label.lower() or label.lower() in t:
            return code
        if score > best_score:
            best, best_score = code, score
    return best if best_score >= 0.6 else None


def match_tutor(text: str, tutors: list[dict]) -> dict | None:
    idx = parse_index(text, len(tutors))
    if idx is not None:
        return tutors[idx - 1]
    return _fuzzy_pick(text, tutors, lambda t: [_tutor_display_name(t)], threshold=0.5)


def match_slot(text: str, slots: list[dict]) -> dict | None:
    """
    Pick a slot deterministically.

    - Numeric pick → use it.
    - Exact YYYY-MM-DD with optional HH:mm → unambiguous match.
    Anything else (vague phrases, multiple matches) returns None.
    """
    idx = parse_index(text, len(slots))
    if idx is not None:
        return slots[idx - 1]

    t = normalize(text)
    date_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t)
    time_match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", t)

    if date_match:
        date = date_match.group(1)
        same_date = [s for s in slots if s["date"] == date]
        if not same_date:
            return None
        if time_match:
            hh = int(time_match.group(1))
            mm = time_match.group(2)
            time_from = f"{hh:02d}:{mm}"
            exact = [s for s in same_date if s["timeFrom"] == time_from]
            if len(exact) == 1:
                return exact[0]
            return None
        if len(same_date) == 1:
            return same_date[0]

    return None


def _fuzzy_pick(
    text: str,
    entries: list[dict],
    name_extractor,
    threshold: float = 0.6,
) -> dict | None:
    t = normalize(text)
    if not t or not entries:
        return None

    # 1) substring containment is unambiguous when only one entry contains it.
    contains = []
    for entry in entries:
        for name in name_extractor(entry):
            n = name.lower()
            if t in n or n in t:
                contains.append(entry)
                break
    if len(contains) == 1:
        return contains[0]

    # 2) best fuzzy score above threshold.
    best, best_score = None, 0.0
    for entry in entries:
        for name in name_extractor(entry):
            score = _similarity(t, name)
            if score > best_score:
                best, best_score = entry, score
    return best if best_score >= threshold else None
