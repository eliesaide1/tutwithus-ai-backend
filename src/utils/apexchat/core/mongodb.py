"""
MongoDB async client (motor) and booking-flow query helpers.

All ObjectId values returned from the DB are cast to str before being
returned to callers, so no BSON types leak into Pydantic models.

When settings.USE_DEMO_DATA is True, every public function delegates
to app.core.demo_data instead of hitting MongoDB — no connection required.
"""

from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import structlog
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from utils.config import *

logger = structlog.get_logger(__name__)


# ── Client / DB singleton ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_client() -> AsyncIOMotorClient:
    return AsyncIOMotorClient(MONGODB_URI)


def get_db() -> AsyncIOMotorDatabase:
    return _get_client()[MONGODB_DB_NAME]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _str_id(doc: dict) -> dict:
    """Convert the _id field of a document from ObjectId to str in-place."""
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


def _today_str() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _parse_iso_to_hhmm(iso_str: str) -> str:
    """Convert an ISO-8601 UTC datetime string to HH:mm format."""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.strftime("%H:%M")
    except Exception:
        return iso_str[:5]  # best-effort fallback


def _safe_object_id(object_id: str):
    """Convert a 24-hex string or JWT-like value to ObjectId, or return None."""
    if not isinstance(object_id, str) or object_id == "":
        return None

    if ObjectId.is_valid(object_id):
        return ObjectId(object_id)

    # If user_id is a JWT token, attempt to extract uid/sub from payload
    if object_id.count(".") == 2:
        try:
            import base64
            import json

            payload_b64 = object_id.split(".")[1]
            padding = '=' * (-len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64 + padding).decode('utf-8'))

            for key in ["uid", "user_id", "sub", "id"]:
                candidate = payload.get(key)
                if isinstance(candidate, str) and ObjectId.is_valid(candidate):
                    return ObjectId(candidate)
        except Exception:
            pass

    return None


def _extract_user_id_str(user_id: str) -> str | None:
    """
    Extract the raw 24-hex user ID string from either a plain ObjectId string
    or a JWT token. Returns the plain string (not ObjectId) for DB queries
    where userId is stored as a string field.
    """
    obj_id = _safe_object_id(user_id)
    if obj_id is None:
        return None
    return str(obj_id)


# ── Wallet ────────────────────────────────────────────────────────────────────

async def check_wallet_funded(user_id: str) -> bool:
    """Return True if the student's wallet has at least one credit with amount > 0.

    Uses an aggregation pipeline that matches the wallet by userId and projects
    a `walletRecharged` boolean — true when any entry in `contents` has amount > 0.
    """
    
    user_id_obj = _safe_object_id(user_id)
    if user_id_obj is None:
        logger.warning("Invalid user_id for wallet check", user_id=user_id)
        return False

    db = get_db()
    user_id_str = str(user_id_obj)
    project_stage = {
        "$project": {
            "userId": 1,
            "contents": 1,
            "walletRecharged": {
                "$gt": [
                    {
                        "$size": {
                            "$ifNull": [
                                {
                                    "$filter": {
                                        "input": {
                                            "$cond": [
                                                {"$isArray": "$contents"},
                                                "$contents",
                                                {"$objectToArray": {"$ifNull": ["$contents", {}]}},
                                            ]
                                        },
                                        "as": "entry",
                                        "cond": {
                                            "$gt": [
                                                {
                                                    "$ifNull": [
                                                        "$$entry.amount",
                                                        "$$entry.v.amount",
                                                    ]
                                                },
                                                0,
                                            ]
                                        },
                                    }
                                },
                                [],
                            ]
                        }
                    },
                    0,
                ]
            },
        }
    }
    pipeline = [
        {"$match": {"$or": [{"userId": user_id_obj}, {"userId": user_id_str}]}},
        project_stage,
    ]
    cursor = db.wallets.aggregate(pipeline)
    doc = await cursor.to_list(length=1)
    if not doc:
        logger.warning(
            "No wallet found for student",
            user_id=user_id,
            user_id_obj=user_id_str,
            db_name=db.name,
        )
        return False
    logger.warning(
        "Wallet check result",
        user_id=user_id,
        wallet_recharged=bool(doc[0].get("walletRecharged", False)),
    )
    return bool(doc[0].get("walletRecharged", False))


async def fetch_wallet_summary(user_id: str) -> list[dict[str, Any]]:
    """Return a list of active credits with hours remaining.

    NOTE: Same userId/contents caveats as check_wallet_funded above.
    The aggregation pipeline $unwind does not work on a map field, so we
    fetch the wallet document and join credits manually.
    """
    if USE_DEMO_DATA:
        return [{"creditName": "Demo Credit", "hoursRemaining": 5, "nbHours": 10}]

    user_id_obj = _safe_object_id(user_id)
    if user_id_obj is None:
        logger.warning("Invalid user_id for wallet summary", user_id=user_id)
        return []

    db = get_db()
    wallet = await db.wallets.find_one({"userId": user_id_obj})
    if not wallet:
        return []

    # contents is a map {"0": {amount, creditId, _id}, "1": {...}, ...}
    contents = wallet.get("contents", {})
    results = []

    for entry in contents.values():
        amount = entry.get("amount", 0)
        if amount <= 0:
            continue

        credit_id = entry.get("creditId")
        if not credit_id:
            continue

        credit = await db.credits.find_one({"_id": ObjectId(credit_id)})
        if credit:
            results.append({
                "creditId": str(credit_id),
                "hoursRemaining": amount,
                "creditName": credit.get("name"),
                "nbHours": credit.get("nbHours"),
                "price": credit.get("price"),
            })

    return results


# ── Levels ────────────────────────────────────────────────────────────────────

async def fetch_levels() -> list[dict[str, Any]]:
    """Return all active levels sorted by display order."""
    if USE_DEMO_DATA:
        from app.core.demo_data import demo_fetch_levels
        return await demo_fetch_levels()

    db = get_db()
    cursor = db.levels.find(
        {"isActive": True},
        {"_id": 1, "name": 1, "order": 1, "requiresCurriculum": 1},
    ).sort("order", 1)
    levels = []
    async for doc in cursor:
        _str_id(doc)
        levels.append(doc)
    return levels


# ── Subjects ──────────────────────────────────────────────────────────────────

async def fetch_subjects(level_id: str) -> list[dict[str, Any]]:
    """Return active subjects for the given level."""
    if USE_DEMO_DATA:
        from app.core.demo_data import demo_fetch_subjects
        return await demo_fetch_subjects(level_id)

    level_obj_id = _safe_object_id(level_id)
    if level_obj_id is None:
        logger.warning("Invalid level_id for fetch_subjects", level_id=level_id)
        return []

    db = get_db()
    cursor = db.subjects.find(
        {"isActive": True, "levelIds": level_obj_id},
        {"_id": 1, "name": 1, "iconUrl": 1},
    )
    subjects = []
    async for doc in cursor:
        _str_id(doc)
        subjects.append(doc)
    return subjects


# ── Curricula ─────────────────────────────────────────────────────────────────

async def fetch_curricula(level_id: str, subject_id: str) -> list[str]:
    """Return distinct curriculum codes taught by tutors for this level+subject."""
    if USE_DEMO_DATA:
        from app.core.demo_data import demo_fetch_curricula
        return await demo_fetch_curricula(level_id, subject_id)

    db = get_db()

    subject_obj_id = _safe_object_id(subject_id)
    level_obj_id = _safe_object_id(level_id)
    if subject_obj_id is None or level_obj_id is None:
        logger.warning(
            "Invalid IDs for fetch_curricula",
            subject_id=subject_id,
            level_id=level_id,
        )
        return []

    codes: list[str] = await db.users.distinct(
        "curriculums",
        {
            "role": "tutor",
            "isBlocked": False,
            "subjects.subjectId": subject_obj_id,
            "subjects.levelIds": level_obj_id,
        },
    )
    return codes


# ── Tutors ────────────────────────────────────────────────────────────────────

async def fetch_tutors(
    level_id: str,
    subject_id: str,
    curriculum: str | None,
) -> list[dict[str, Any]]:
    """
    Return tutors filtered by level + subject + (optionally) curriculum.

    Only tutors with at least one ACTIVE future availability slot are returned.
    """
    if USE_DEMO_DATA:
        from app.core.demo_data import demo_fetch_tutors
        return await demo_fetch_tutors(level_id, subject_id, curriculum)

    subject_obj_id = _safe_object_id(subject_id)
    level_obj_id = _safe_object_id(level_id)
    if subject_obj_id is None or level_obj_id is None:
        logger.warning(
            "Invalid IDs for fetch_tutors",
            subject_id=subject_id,
            level_id=level_id,
        )
        return []

    query: dict[str, Any] = {
        "role": "tutor",
        "isBlocked": False,
        "subjects.subjectId": subject_obj_id,
        "subjects.levelIds": level_obj_id,
    }
    if curriculum:
        query["curriculums"] = curriculum

    db = get_db()
    today = _today_str()
    tutors: list[dict] = []

    async for doc in db.users.find(
        query,
        {
            "_id": 1,
            "firstName": 1,
            "lastName": 1,
            "bio": 1,
            "degree": 1,
            "timezone": 1,
            "availableTimes": 1,
        },
    ):
        # Only include tutors with at least one active future slot
        has_slot = any(
            a.get("status") == "ACTIVE" and a.get("date", "") >= today
            for a in doc.get("availableTimes", [])
        )
        if has_slot:
            _str_id(doc)
            tutors.append(doc)

    return tutors


# ── Availability ──────────────────────────────────────────────────────────────

def extract_future_slots(tutor_doc: dict) -> list[dict[str, Any]]:
    """
    Parse a tutor's availableTimes and return future ACTIVE slots as plain dicts.

    Each returned item:
        {"date": "YYYY-MM-DD", "timeFrom": "HH:mm", "timeTo": "HH:mm"}
    """
    today = _today_str()
    slots: list[dict[str, Any]] = []

    for availability in tutor_doc.get("availableTimes", []):
        if availability.get("status") != "ACTIVE":
            continue
        date_str = availability.get("date", "")
        if date_str < today:
            continue
        for slot in availability.get("timeSlots", []):
            start = slot.get("startHour", "")
            end = slot.get("endHour", "")
            if start and end:
                slots.append(
                    {
                        "date": date_str,
                        "timeFrom": _parse_iso_to_hhmm(start),
                        "timeTo": _parse_iso_to_hhmm(end),
                    }
                )

    # Sort by date then start time
    slots.sort(key=lambda s: (s["date"], s["timeFrom"]))
    return slots


# ── Rescheduling helpers ──────────────────────────────────────────────────────

async def fetch_session_available_slots(course_id: str) -> list[dict[str, Any]]:
    """
    Given a session/course ObjectId, look up the session's tutor and return
    their future ACTIVE available time slots.

    Look-up chain:
        courses._id  →  courses.tutorId  →  users._id  →  availableTimes

    Returns: [{"date": "YYYY-MM-DD", "timeFrom": "HH:mm", "timeTo": "HH:mm"}, ...]
    Returns [] if the session is not found, has no tutor, or the tutor has no
    upcoming slots.
    """
    
    session_id_obj = _safe_object_id(course_id)
    if session_id_obj is None:
        logger.warning("fetch_session_available_slots: invalid course_id", course_id=course_id)
        return []

    db = get_db()

    # Step 1: fetch the session to find its tutorId
    session = await db.courses.find_one(
        {"_id": session_id_obj},
        {"tutorId": 1},
    )
    if not session:
        logger.warning("fetch_session_available_slots: session not found", course_id=course_id)
        return []

    raw_tutor_id = session.get("tutorId")
    if raw_tutor_id is None:
        logger.warning("fetch_session_available_slots: session has no tutorId", course_id=course_id)
        return []

    # tutorId may already be an ObjectId (BSON) or a plain hex string
    tutor_id_obj = raw_tutor_id if isinstance(raw_tutor_id, ObjectId) else _safe_object_id(str(raw_tutor_id))
    if tutor_id_obj is None:
        logger.warning(
            "fetch_session_available_slots: cannot resolve tutorId",
            raw_tutor_id=str(raw_tutor_id),
        )
        return []

    # Step 2: fetch the tutor's availability
    tutor = await db.users.find_one(
        {"_id": tutor_id_obj},
        {"availableTimes": 1},
    )
    if not tutor:
        logger.warning(
            "fetch_session_available_slots: tutor not found",
            tutor_id=str(tutor_id_obj),
        )
        return []

    return extract_future_slots(tutor)


# ── Knowledge-base helpers (used by RAGTool) ─────────────────────────────────

async def fetch_all_tutors_for_knowledge() -> list[dict[str, Any]]:
    """Return all active tutors with profile fields for the RAG knowledge base."""
    db = get_db()
    tutors: list[dict] = []
    async for doc in db.users.find(
        {"role": "tutor", "isBlocked": False},
        {"_id": 1, "firstName": 1, "lastName": 1, "bio": 1, "degree": 1,
         "subjects": 1, "curriculums": 1},
    ):
        _str_id(doc)
        tutors.append(doc)
    return tutors


async def fetch_all_subjects_for_knowledge() -> list[dict[str, Any]]:
    """Return all active subjects for the RAG knowledge base."""
    db = get_db()
    subjects: list[dict] = []
    async for doc in db.subjects.find({"isActive": True}, {"_id": 1, "name": 1}):
        _str_id(doc)
        subjects.append(doc)
    return subjects


async def fetch_all_subjects() -> list[dict[str, Any]]:
    """Return every active subject with its level associations, across ALL levels.

    Used by the booking flow's subject-first routing: when the user names a
    subject before choosing a level, we look up which level(s) offer it. Both
    `_id` and every entry in `levelIds` are returned as plain strings.
    """
    db = get_db()
    out: list[dict[str, Any]] = []
    async for doc in db.subjects.find(
        {"isActive": True},
        {"_id": 1, "name": 1, "levelIds": 1},
    ):
        _str_id(doc)
        doc["levelIds"] = [str(x) for x in doc.get("levelIds", [])]
        out.append(doc)
    return out