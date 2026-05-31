"""
Schema introspection / training data builder for the MongoDB-adapted Vanna engine.

Vanna AI's training model has three kinds of context:
    1. DDL — schema descriptions
    2. Documentation — free-text business notes
    3. Question / Query pairs — few-shot examples

For MongoDB we infer "DDL" by sampling a handful of documents per collection
and recording the field paths and inferred types.  In addition, this module
attempts to detect *reference fields* (foreign-key-like links between
collections) and *label fields* (the human-readable column for IDs).  This
metadata is fed to the LLM so it can produce ``$lookup`` stages that swap
opaque ObjectIds for human-readable names in chart axes.

Invoked once at engine init and cached on the engine instance.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import structlog
from bson import ObjectId
from pymongo.database import Database

logger = structlog.get_logger(__name__)


# Collections ignored for training — internal app state, not analytical data.
_DEFAULT_IGNORED = {
    "counters",
    "sessions",
    "messages",
    "user_facts",
    "fact_changes",
    "apexchat_payload_store",
    "document_store",
}

# Field names that look like a human-readable label.  Order matters — we pick
# the first one that exists in a collection as the "default label" for that
# collection when other collections reference it via _id / *_id.  Both
# snake_case and camelCase variants are listed because Mongo deployments mix
# both (Mongoose defaults are camelCase).
_LABEL_FIELD_CANDIDATES: tuple[str, ...] = (
    "name",
    "fullName", "full_name", "fullname",
    "displayName", "display_name",
    "title",
    "label",
    "firstName", "first_name",
    "username", "user_name",
    "subject", "subjectName", "subject_name",
    "courseName", "course_name", "courseTitle", "course_title",
    "email",
    "code",
)

# Locale keys to try when a label field turns out to be a multilingual dict.
# We prefer English for the default rendering, but the LLM can override.
_LOCALE_PRIORITY: tuple[str, ...] = ("en", "ar", "fr", "es", "default", "value")

# Suffix patterns we treat as foreign-key style references.
_FK_SUFFIX_RE = re.compile(r"(.+?)(?:_id|_ids|Id|Ids|Ref|_ref|_uuid)$")


def _infer_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, ObjectId):
        return "objectId"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        if value:
            return f"array<{_infer_type(value[0])}>"
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _flatten_keys(doc: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Walk a document and emit {dotted.path: inferred_type}."""
    out: dict[str, str] = {}
    for k, v in doc.items():
        path = f"{prefix}.{k}" if prefix else k
        out[path] = _infer_type(v)
        if isinstance(v, dict):
            out.update(_flatten_keys(v, path))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out.update(_flatten_keys(v[0], f"{path}[]"))
    return out


def _pick_label_field(fields: dict[str, str]) -> str | None:
    """
    Choose the best human-readable label field for a collection.

    Handles three real-world cases:
      • Plain string label (`name: "Algebra"`) — returned as-is.
      • Multilingual dict (`name: {en: "Algebra", ar: "..."}`) — returns the
        dotted path to the preferred locale, e.g. ``"name.en"``.  Detected by
        the presence of a dotted child of the candidate (e.g. ``name.en``)
        whose own type is "string".
      • No `name` field but a person collection with `firstName` /
        `firstname` — returned so the LLM knows to use it (and ideally
        concatenate with `lastName`, which the system prompt instructs it on).
    """
    paths = set(fields.keys())

    for candidate in _LABEL_FIELD_CANDIDATES:
        if candidate in paths:
            typ = fields[candidate]
            if typ == "object":
                # Multilingual dict — pick the best locale child if present
                for locale in _LOCALE_PRIORITY:
                    dotted = f"{candidate}.{locale}"
                    if dotted in paths and fields[dotted] == "string":
                        return dotted
                # No known locale — fall through to next candidate
                continue
            if typ in {"string", "int", "float"}:
                return candidate

    # Fall back to a string field that is NOT an id-looking suffix
    for path, typ in fields.items():
        if typ != "string":
            continue
        leaf = path.split(".")[-1]
        if leaf == "_id" or _FK_SUFFIX_RE.match(leaf):
            continue
        return path
    return None


def _candidate_collections_for_fk(
    field_path: str,
    collection_names: list[str],
) -> list[str]:
    """
    Given a field name like ``user_id`` or ``tutorIds[]``, guess which
    collections it might reference.  Returns a list of plausible names ranked
    best-guess first.  A heuristic, not authoritative.
    """
    leaf = field_path.split(".")[-1].rstrip("[]")
    m = _FK_SUFFIX_RE.match(leaf)
    if not m:
        return []
    base = m.group(1).lower()

    candidates: list[str] = []

    # Exact matches first
    for col in collection_names:
        if col.lower() == base:
            candidates.append(col)
        elif col.lower() == base + "s":
            candidates.append(col)
        elif col.lower().rstrip("s") == base:
            candidates.append(col)

    # Substring fallback — base must appear as a whole token in the collection
    if not candidates:
        for col in collection_names:
            tokens = re.split(r"[_\s-]+", col.lower())
            if base in tokens:
                candidates.append(col)

    # Common aliases — booking systems often reference users via tutor_id /
    # user_id / instructor_id but the actual collection is just `users`.
    if not candidates:
        person_aliases = {
            "tutor", "student", "enrolledstudent", "enrolled",
            "instructor", "teacher", "learner",
            "owner", "creator", "author", "admin", "user",
            "host", "participant", "assignee", "assignedto",
            "createdby", "updatedby", "approvedby", "rejectedby",
            "referrer", "referredby", "invitedby",
        }
        if base in person_aliases:
            for col in ("users", "um_user"):
                if col in collection_names:
                    candidates.append(col)
        # Strip a leading verb-prefix and retry (enrolled<X>, assigned<X>)
        for prefix in ("enrolled", "assigned", "selected", "chosen", "active"):
            if base.startswith(prefix) and len(base) > len(prefix):
                rest = base[len(prefix):]
                for col in collection_names:
                    if col.lower() == rest or col.lower() == rest + "s":
                        candidates.append(col)
                        break
                if rest in person_aliases:
                    for col in ("users", "um_user"):
                        if col in collection_names and col not in candidates:
                            candidates.append(col)

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


def _detect_references(
    schema: dict[str, dict[str, str]],
) -> dict[str, dict[str, dict[str, str]]]:
    """
    Build ``{collection: {field_path: {"collection": ref_col, "label_field": ...}}}``.
    """
    collection_names = list(schema.keys())
    refs: dict[str, dict[str, dict[str, str]]] = {}

    for col, fields in schema.items():
        col_refs: dict[str, dict[str, str]] = {}
        for path, typ in fields.items():
            leaf = path.split(".")[-1].rstrip("[]")
            if leaf == "_id":
                # Self-id is not a reference
                continue
            if not _FK_SUFFIX_RE.match(leaf):
                continue
            # Only treat strings, objectIds, and arrays of those as references
            base_type = typ.replace("array<", "").rstrip(">")
            if base_type not in {"string", "objectId"}:
                continue

            candidates = _candidate_collections_for_fk(path, collection_names)
            if not candidates:
                continue
            ref_col = candidates[0]
            label = _pick_label_field(schema.get(ref_col, {})) or "_id"
            col_refs[path] = {"collection": ref_col, "label_field": label}
        if col_refs:
            refs[col] = col_refs

    return refs


def _detect_label_fields(
    schema: dict[str, dict[str, str]],
) -> dict[str, str]:
    """Pick a default label field per collection (used by ID enrichment)."""
    out: dict[str, str] = {}
    for col, fields in schema.items():
        label = _pick_label_field(fields)
        if label:
            out[col] = label
    return out


def introspect_schema(
    db: Database,
    sample_size: int = 8,
    ignored: set[str] | None = None,
) -> dict[str, Any]:
    """
    Sample each collection and return a structured schema bundle:

    {
        "fields":       {col: {field_path: type}},
        "references":   {col: {field_path: {"collection": ref, "label_field": label}}},
        "label_fields": {col: label_field},
        "row_counts":   {col: int},
    }

    The bundle is what the LLM is shown and what the engine uses for
    enrichment / lookup hints.
    """
    ignored_set = (ignored or set()) | _DEFAULT_IGNORED
    fields: dict[str, dict[str, str]] = {}
    counts: dict[str, int] = {}

    for col_name in db.list_collection_names():
        if col_name in ignored_set or col_name.startswith("system."):
            continue

        merged: Counter[str] = Counter()
        type_obs: dict[str, Counter[str]] = {}
        try:
            for doc in db[col_name].find({}, limit=sample_size):
                flat = _flatten_keys(doc)
                for path, typ in flat.items():
                    type_obs.setdefault(path, Counter())[typ] += 1
                    merged[path] += 1
            try:
                counts[col_name] = db[col_name].estimated_document_count()
            except Exception:
                counts[col_name] = 0
        except Exception as exc:
            logger.warning(
                "Schema introspection failed for collection",
                collection=col_name,
                error=str(exc),
            )
            continue

        if not merged:
            continue

        # Pick the most common type for each path
        col_fields: dict[str, str] = {}
        for path, type_counter in type_obs.items():
            col_fields[path] = type_counter.most_common(1)[0][0]
        fields[col_name] = col_fields

    references = _detect_references(fields)
    label_fields = _detect_label_fields(fields)

    return {
        "fields": fields,
        "references": references,
        "label_fields": label_fields,
        "row_counts": counts,
    }


def schema_to_prompt_text(schema: dict[str, Any]) -> str:
    """
    Render the introspected schema into a compact text block for the LLM,
    including reference hints so the model can emit ``$lookup`` stages.
    """
    fields = schema.get("fields", {}) if isinstance(schema, dict) else {}
    references = schema.get("references", {}) if isinstance(schema, dict) else {}
    counts = schema.get("row_counts", {}) if isinstance(schema, dict) else {}
    labels = schema.get("label_fields", {}) if isinstance(schema, dict) else {}

    if not fields:
        return "(no collections discovered)"

    lines: list[str] = []
    for col, col_fields in fields.items():
        approx = counts.get(col, 0)
        size_hint = f" (~{approx:,} docs)" if approx else ""
        label = labels.get(col)
        label_hint = f"  // human-readable label: `{label}`" if label else ""
        lines.append(f"Collection `{col}`{size_hint}:{label_hint}")
        for path, typ in col_fields.items():
            ref = references.get(col, {}).get(path)
            ref_hint = (
                f"   → references `{ref['collection']}._id`, label=`{ref['label_field']}`"
                if ref
                else ""
            )
            lines.append(f"  - {path}: {typ}{ref_hint}")
        lines.append("")

    if references:
        lines.append("REFERENCE GRAPH (use $lookup to resolve IDs to names):")
        for col, col_refs in references.items():
            for path, info in col_refs.items():
                lines.append(
                    f"  {col}.{path}  →  {info['collection']}._id "
                    f"(display `{info['collection']}.{info['label_field']}`)"
                )
        lines.append("")

    return "\n".join(lines).rstrip()


# ── Seed training examples ───────────────────────────────────────────────────
#
# Hand-picked canonical question/pipeline pairs that teach the LLM the
# "house style" against the actual production schema:
#   • CamelCase field names (`tutorId`, `subjectId`, `createdAt`).
#   • Multilingual `name` dicts → project ``$ifNull: ["$x.name.en", "$x.name.ar", "Unknown"]``.
#   • User display names → `$concat` of `firstName` + `lastName`.
#   • Always `$lookup` foreign-key IDs and `preserveNullAndEmptyArrays: true`
#     on the `$unwind` so soft-deleted refs don't drop rows.
#   • Always cap output with `$limit`.

SEED_EXAMPLES: list[dict[str, Any]] = [
    {
        "question": "Show me the number of users per role",
        "collection": "users",
        "pipeline": [
            {"$group": {"_id": "$role", "count": {"$sum": 1}}},
            {"$project": {"_id": 0, "role": "$_id", "count": 1}},
            {"$sort": {"count": -1}},
            {"$limit": 50},
        ],
    },
    {
        "question": "Top 10 tutors by number of courses",
        "collection": "courses",
        "pipeline": [
            {"$group": {"_id": "$tutorId", "courses": {"$sum": 1}}},
            {
                "$lookup": {
                    "from": "users",
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "tutor",
                }
            },
            {"$unwind": {"path": "$tutor", "preserveNullAndEmptyArrays": True}},
            {
                "$project": {
                    "_id": 0,
                    "tutor": {
                        "$trim": {
                            "input": {
                                "$concat": [
                                    {"$ifNull": ["$tutor.firstName", ""]},
                                    " ",
                                    {"$ifNull": ["$tutor.lastName", ""]},
                                ]
                            }
                        }
                    },
                    "courses": 1,
                }
            },
            {"$sort": {"courses": -1}},
            {"$limit": 10},
        ],
    },
    {
        "question": "Courses per month for the last year",
        "collection": "courses",
        "pipeline": [
            {"$match": {"date": {"$ne": None}}},
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": "%Y-%m", "date": "$date"}
                    },
                    "courses": {"$sum": 1},
                }
            },
            {"$project": {"_id": 0, "month": "$_id", "courses": 1}},
            {"$sort": {"month": 1}},
            {"$limit": 24},
        ],
    },
    {
        "question": "Distribution of courses by subject",
        "collection": "courses",
        "pipeline": [
            {"$group": {"_id": "$subjectId", "count": {"$sum": 1}}},
            {
                "$lookup": {
                    "from": "subjects",
                    "localField": "_id",
                    "foreignField": "_id",
                    "as": "subject",
                }
            },
            {"$unwind": {"path": "$subject", "preserveNullAndEmptyArrays": True}},
            {
                "$project": {
                    "_id": 0,
                    "subject": {
                        "$ifNull": [
                            "$subject.name.en",
                            {"$ifNull": ["$subject.name.ar", "Unknown"]},
                        ]
                    },
                    "count": 1,
                }
            },
            {"$sort": {"count": -1}},
            {"$limit": 30},
        ],
    },
    {
        "question": "Revenue per month from successful transactions",
        "collection": "transactions",
        "pipeline": [
            {"$match": {"status": "paid"}},
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": "%Y-%m", "date": "$createdAt"}
                    },
                    "revenue": {"$sum": "$amount"},
                    "orders": {"$sum": 1},
                }
            },
            {"$project": {"_id": 0, "month": "$_id", "revenue": 1, "orders": 1}},
            {"$sort": {"month": 1}},
            {"$limit": 24},
        ],
    },
    {
        "question": "Active vs cancelled courses",
        "collection": "courses",
        "pipeline": [
            {
                "$group": {
                    "_id": {
                        "$cond": [{"$eq": ["$isCanceled", True]}, "Cancelled", "Active"]
                    },
                    "count": {"$sum": 1},
                }
            },
            {"$project": {"_id": 0, "status": "$_id", "count": 1}},
            {"$sort": {"count": -1}},
            {"$limit": 10},
        ],
    },
    {
        "question": "Top subjects by hours taught",
        "collection": "courses",
        "pipeline": [
            {"$group": {"_id": "$subjectId",
                        "hours": {"$sum": {"$divide": ["$duration", 60]}}}},
            {"$lookup": {"from": "subjects", "localField": "_id",
                         "foreignField": "_id", "as": "subject"}},
            {"$unwind": {"path": "$subject", "preserveNullAndEmptyArrays": True}},
            {"$project": {
                "_id": 0,
                "subject": {"$ifNull": ["$subject.name.en",
                                          {"$ifNull": ["$subject.name.ar", "Unknown"]}]},
                "hours": {"$round": ["$hours", 1]}
            }},
            {"$sort": {"hours": -1}},
            {"$limit": 15},
        ],
    },
]


# ── Schema documentation block (free-text notes for the LLM) ─────────────────
#
# Surfaces facts that pure introspection misses: business meaning, which
# collection is the booking/session of record, multilingual conventions, etc.
SCHEMA_DOCUMENTATION: str = """\
Business glossary (read carefully before generating any pipeline):

• `users` — every account, role-discriminated.  Display name is
  `firstName + " " + lastName` (always concat both, do NOT plot only one).
  Roles include "admin", "tutor", "student".  Filter by `role` to scope a
  query to one population.  Arabic names live in `firstNameAr` / `lastNameAr`.

• `courses` — the booking record (a scheduled tutor-student session).  The
  user-visible "bookings" / "sessions" / "lessons" all map to this
  collection.  Key fields: `tutorId` (→ users), `enrolledStudentId` (→ users),
  `subjectId` (→ subjects), `levelId` (→ levels), `date` (the session date),
  `duration` (minutes), `amountPaid`, `isCanceled`, `rating`.

• `meetings` — the *Zoom session* attached to a course (NOT the booking).
  Use this only for questions about Zoom-account utilisation, attendance, or
  meeting duration vs. scheduled duration.  Linked from `courses.meetingId`.

• `subjects`, `levels`, `bundles`, `credits` — reference data.  All have a
  multilingual `name` field of shape `{en: "...", ar: "..."}`.  ALWAYS
  project via `$ifNull: ["$x.name.en", "$x.name.ar", "Unknown"]` — never
  project the raw `name` object (it renders as "[object Object]").

• `transactions` — payments.  `userId` (→ users), `amount`, `status`
  ("paid" / "pending" / "failed"), `createdAt`.  Use `status: "paid"` when
  the user asks about revenue.

• `wallets` — credit balances.  `userId` (→ users), `contents` (array of
  credit allocations), `usedCodes` (promo codes redeemed).

• `promocodes` — discount codes.  `bundleIds` / `creditIds` (arrays of
  references to bundles / credits collections).

Time fields are camelCase: `createdAt`, `updatedAt`, `startTime`, `expiryDate`,
`lastLoginAt`.  The course-session date is `courses.date`.  When the user
asks "per month / per week / per day" without specifying a date column,
prefer `createdAt` for transactions and `date` for courses.
"""
