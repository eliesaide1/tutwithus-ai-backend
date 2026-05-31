"""
Database models and helpers for MongoDB.

Replaces the previous SQLAlchemy ORM layer. Collections mirror the old
PostgreSQL tables; document shapes are documented in comments below.
"""
from pymongo import MongoClient
from contextlib import contextmanager
from utils.config import MONGODB_URI, MONGO_DB_NAME

# ── MongoDB client singleton ─────────────────────────────────────────────────

_client = MongoClient(MONGODB_URI)
_db = _client[MONGO_DB_NAME]


def get_mongo_db():
    """Return the default pymongo Database handle."""
    return _db


@contextmanager
def get_db_session():
    """
    Context-manager kept for API compatibility.

    MongoDB does not require explicit commit/rollback for single-document
    writes.  This yields the database handle and is a no-op on exit.
    """
    yield _db


# ── Sequence emulation ───────────────────────────────────────────────────────

def get_next_sequence_value(db, sequence_name: str) -> int:
    """
    Atomically increment and return a counter stored in the `counters`
    collection.  Mirrors PostgreSQL ``nextval(sequence_name)``.
    """
    result = db.counters.find_one_and_update(
        {"_id": sequence_name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=True,   # return the *updated* document
    )
    return result["seq"]


# ── Document schemas (for reference — not enforced at driver level) ──────────
#
# Collection: anldba_reports
# {
#     "report_id": int,
#     "report_name": str,
#     "report_description": str,
#     "creation_date": datetime,
#     "created_by": int,
#     "pdf_id": int
# }
#
# Collection: anldba_report_details
# {
#     "pdf_id": int,
#     "report_executive": str,
#     "report_summary": str,
#     "report_pdf": str,
#     "report_structure": str,
#     "creation_date": datetime,
#     "type": str,
#     "report_name": str
# }
#
# Collection: anldba_report_sheets
# {
#     "sheet_id": int,
#     "report_id": int,
#     "sheet_name": str,
#     "sheet_headers": str,
#     "metadata": str,
#     "footer": str,
#     "code_header": str,
#     "description_header": str,
#     "is_arabic": bool
# }
#
# Collection: anldba_report_elements
# {
#     "element_id": int,
#     "report_id": int,
#     "sheet_id": int,
#     "element_description": str,
#     "element_number": str,
#     "formula": str
# }
#
# Collection: anldba_action_details
# {
#     "action_id": int,
#     "pdf_id": int,
#     "action_definition": str,
#     "recommendation": str,
#     "user_id": str,
#     "entity": str,
#     "priority": str,
#     "obligation": str,
#     "execution_date": datetime
# }
#
# Collection: anldba_report_text_extraction
# {
#     "report_id": int,
#     "report_name": str,
#     "full_text": str,
#     "summary_text": str,
#     "creation_date": datetime,
#     "created_by": int,
#     "embedding": list[float],        # 1024-dim vector
#     "summary_embedding": list[float], # 1024-dim vector
#     "pdf_base64": bytes,
#     "issued_country": int,
#     "source": str,
#     "circular_number": str,
#     "execution_date": datetime,
#     "issued_date": datetime,
#     "keywords": str,
#     "scope": str,
#     "topic": str,
#     "affected_institutions": str,
#     "article_type": str,
#     "regulation_type": str
# }
#
# Collection: anldba_report_keywords
# {
#     "keywords": str     # also the unique key
# }
