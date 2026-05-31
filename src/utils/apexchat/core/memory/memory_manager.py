"""
Memory Manager — async persistence layer for the Apexchat memory system.

All database calls are blocking (pymongo) and are pushed to a thread-pool
worker via asyncio.to_thread, keeping the FastAPI / LangGraph event loop free.

MongoDB version: replaces the former psycopg2 + PostgreSQL implementation.
Vector similarity search is performed in Python (cosine similarity via numpy)
since local MongoDB does not ship with Atlas Search vector indexes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from utils.config import *
from utils.apexchat.core.memory.embedding_wrapper import TextEmbedder

logger = structlog.get_logger(__name__)

VALID_FACT_TYPES = frozenset({"profile", "preference", "event", "goal", "habit"})
VALID_CHANGE_TYPES = frozenset({"created", "updated", "contradicted", "merged"})


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _next_id(db, counter_name: str) -> int:
    """Atomically increment and return an auto-increment counter."""
    result = db.counters.find_one_and_update(
        {"_id": counter_name},
        {"$inc": {"seq": 1}},
        upsert=True,
        return_document=True,
    )
    return result["seq"]


class MemoryManager:
    """
    Central async memory management system for Apexchat (MongoDB).

    Owns all database operations for facts, sessions, and messages.
    """

    def __init__(
        self,
        mongodb_tools,
        embedding_model: TextEmbedder,
        llm_client=None,
    ) -> None:
        self.mongodb_tools = mongodb_tools
        self.embedding_model = embedding_model
        self._llm_client = llm_client
        logger.info("MemoryManager initialised")

    # ── Embedding helper ──────────────────────────────────────────────────────

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate a normalised embedding vector asynchronously."""
        try:
            result = await self.embedding_model.aencode(text, normalize_embeddings=True)
            return np.array(result, dtype=np.float32)
        except Exception as exc:
            logger.error("embed_text failed", error=str(exc), exc_info=True)
            return np.zeros(MEMORY_EMBEDDING_DIMENSIONS, dtype=np.float32)

    # ── Session management ────────────────────────────────────────────────────

    async def get_or_create_session(
        self,
        student_id: str,
        session_id: Optional[str] = None,
        reuse_inactive_timeout: int | None = None,
    ) -> Tuple[int, str]:
        """Return an active (session_db_id, session_id), creating one if needed."""
        timeout = reuse_inactive_timeout or MEMORY_SESSION_INACTIVE_TIMEOUT_MINUTES
        return await asyncio.to_thread(
            self._get_or_create_session_sync, student_id, session_id, timeout
        )

    def _get_or_create_session_sync(
        self,
        student_id: str,
        session_id: Optional[str],
        timeout: int,
    ) -> Tuple[int, str]:
        try:
            db = self.mongodb_tools.get_db_connection()
            col = db.sessions

            if session_id:
                row = col.find_one({"session_id": session_id})

                if row:
                    db_id = row["id"]
                    found_sid = row["session_id"]
                    last_activity = row.get("last_activity_at")
                    is_active = row.get("is_active", False)

                    elapsed = timedelta(0)
                    if last_activity:
                        if last_activity.tzinfo is None:
                            elapsed = datetime.utcnow() - last_activity
                        else:
                            elapsed = datetime.now(timezone.utc) - last_activity

                    if is_active and elapsed > timedelta(minutes=timeout):
                        logger.info(
                            "Session timed out — resetting",
                            session_id=found_sid,
                            inactive_minutes=round(elapsed.total_seconds() / 60, 1),
                        )
                        col.update_one(
                            {"id": db_id},
                            {"$set": {
                                "is_active": True,
                                "started_at": datetime.now(timezone.utc),
                                "last_activity_at": datetime.now(timezone.utc),
                                "ended_at": None,
                                "message_count": 0,
                                "session_title": None,
                                "session_summary": None,
                            }},
                        )
                        logger.info("Timed-out session reset and reused", session_id=found_sid)
                        return db_id, found_sid

                    elif is_active:
                        col.update_one(
                            {"id": db_id},
                            {"$set": {"last_activity_at": datetime.now(timezone.utc)}},
                        )
                        logger.debug("Reusing active session", session_id=found_sid)
                        return db_id, found_sid

                    else:
                        col.update_one(
                            {"id": db_id},
                            {"$set": {
                                "is_active": True,
                                "last_activity_at": datetime.now(timezone.utc),
                            }},
                        )
                        logger.info("Reactivated inactive session", session_id=found_sid)
                        return db_id, found_sid

            else:
                row = col.find_one(
                    {"student_id": student_id, "is_active": True},
                    sort=[("last_activity_at", -1)],
                )
                if row:
                    old_sid = row["session_id"]
                    logger.info(
                        "Closing previous session (empty session_id in request)",
                        session_id=old_sid,
                    )
                    self._close_session_sync(old_sid)

            # Create new session
            new_sid = session_id if session_id else f"session_{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            new_db_id = _next_id(db, "session_id")

            col.insert_one({
                "id": new_db_id,
                "session_id": new_sid,
                "student_id": student_id,
                "started_at": datetime.now(timezone.utc),
                "last_activity_at": datetime.now(timezone.utc),
                "ended_at": None,
                "message_count": 0,
                "is_active": True,
                "session_title": None,
                "session_summary": None,
            })

            logger.info("New session created", session_id=new_sid, student_id=student_id)
            return new_db_id, new_sid

        except Exception as exc:
            logger.error("_get_or_create_session_sync failed", error=str(exc), exc_info=True)
            raise

    async def close_session_with_summary(self, session_id: str) -> None:
        """Mark a session as ended and generate a title/summary via the LLM."""
        try:
            history = await self.get_full_session_history(session_id)
            messages = history.get("messages", [])

            if not messages:
                logger.info("No messages in session — closing without summary", session_id=session_id)
                await self.end_session(session_id)
                return

            if self._llm_client:
                conversation_text = "\n".join(
                    f"{'User' if m['message_type'] == 'user_query' else 'Assistant'}: {m['content']}"
                    for m in messages
                )

                from langchain_core.messages import HumanMessage
                from utils.apexchat.schemas.models import ConversationSummary

                structured_llm = self._llm_client.with_structured_output(ConversationSummary)
                prompt = (
                    f"Analyze this conversation and provide a concise title (max 10 words) "
                    f"and a brief summary (max 30 words).\n\n{conversation_text}"
                )

                try:
                    result: ConversationSummary = await structured_llm.ainvoke(
                        [HumanMessage(content=prompt)]
                    )
                    await asyncio.to_thread(
                        self._update_session_summary_sync,
                        session_id,
                        result.title,
                        result.summary,
                    )
                    logger.info(
                        "Session closed with summary",
                        session_id=session_id,
                        title=result.title,
                    )
                    return
                except Exception as exc:
                    logger.warning(
                        "LLM summary failed — closing without summary",
                        session_id=session_id,
                        error=str(exc),
                    )

            await self.end_session(session_id)

        except Exception as exc:
            logger.error("close_session_with_summary failed", session_id=session_id, error=str(exc))
            await self.end_session(session_id)

    def _close_session_sync(self, session_id: str) -> None:
        try:
            db = self.mongodb_tools.get_db_connection()
            db.sessions.update_one(
                {"session_id": session_id},
                {"$set": {
                    "ended_at": datetime.now(timezone.utc),
                    "is_active": False,
                }},
            )
        except Exception as exc:
            logger.error("_close_session_sync failed", session_id=session_id, error=str(exc))

    def _update_session_summary_sync(self, session_id: str, title: str, summary: str) -> None:
        db = self.mongodb_tools.get_db_connection()
        db.sessions.update_one(
            {"session_id": session_id},
            {"$set": {
                "ended_at": datetime.now(timezone.utc),
                "is_active": False,
                "session_title": title,
                "session_summary": summary,
            }},
        )

    async def update_session_activity(self, session_db_id: int) -> None:
        await asyncio.to_thread(self._update_session_activity_sync, session_db_id)

    def _update_session_activity_sync(self, session_db_id: int) -> None:
        try:
            db = self.mongodb_tools.get_db_connection()
            db.sessions.update_one(
                {"id": session_db_id},
                {
                    "$set": {"last_activity_at": datetime.now(timezone.utc)},
                    "$inc": {"message_count": 1},
                },
            )
        except Exception as exc:
            logger.error("_update_session_activity_sync failed", error=str(exc))

    async def end_session(self, session_id: str) -> None:
        await asyncio.to_thread(self._close_session_sync, session_id)
        logger.info("Session ended", session_id=session_id)

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_session_info_sync, session_id)

    def _get_session_info_sync(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            db = self.mongodb_tools.get_db_connection()
            row = db.sessions.find_one({"session_id": session_id})
            if not row:
                return None
            return {
                "session_db_id": row["id"],
                "session_id": row["session_id"],
                "student_id": row["student_id"],
                "started_at": row.get("started_at"),
                "last_activity_at": row.get("last_activity_at"),
                "message_count": row.get("message_count", 0),
                "is_active": row.get("is_active", False),
            }
        except Exception as exc:
            logger.error("_get_session_info_sync failed", session_id=session_id, error=str(exc))
            return None

    async def get_latest_session_for_user(self, student_id: str) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_latest_session_sync, student_id)

    def _get_latest_session_sync(self, student_id: str) -> Optional[Dict[str, Any]]:
        try:
            db = self.mongodb_tools.get_db_connection()
            row = db.sessions.find_one(
                {"student_id": student_id},
                sort=[("last_activity_at", -1)],
            )
            if not row:
                return None
            return {
                "session_db_id": row["id"],
                "session_id": row["session_id"],
                "started_at": row.get("started_at"),
                "last_activity_at": row.get("last_activity_at"),
                "is_active": row.get("is_active", False),
            }
        except Exception as exc:
            logger.error("_get_latest_session_sync failed", student_id=student_id, error=str(exc))
            return None

    async def get_recent_sessions(self, student_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_recent_sessions_sync, student_id, limit)

    def _get_recent_sessions_sync(self, student_id: str, limit: int) -> List[Dict[str, Any]]:
        try:
            db = self.mongodb_tools.get_db_connection()
            rows = list(
                db.sessions.find({"student_id": student_id})
                .sort("last_activity_at", -1)
                .limit(limit)
            )
            return [
                {
                    "session_db_id": r["id"],
                    "session_id": r["session_id"],
                    "started_at": r["started_at"].isoformat() if r.get("started_at") else None,
                    "last_activity_at": r["last_activity_at"].isoformat() if r.get("last_activity_at") else None,
                    "ended_at": r["ended_at"].isoformat() if r.get("ended_at") else None,
                    "message_count": r.get("message_count", 0),
                    "is_active": r.get("is_active", False),
                    "session_title": r.get("session_title"),
                    "session_summary": r.get("session_summary"),
                }
                for r in rows
            ]
        except Exception as exc:
            logger.error("_get_recent_sessions_sync failed", student_id=student_id, error=str(exc))
            return []

    # ── Message management ────────────────────────────────────────────────────

    async def store_message(
        self,
        session_db_id: int,
        student_id: str,
        message_type: str,
        content: str,
        route_type: Optional[str] = None,
    ) -> int:
        if message_type not in {"user_query", "assistant_response"}:
            raise ValueError(f"Invalid message_type: {message_type!r}")

        embedding = await self.embed_text(content)
        message_id = await asyncio.to_thread(
            self._store_message_sync,
            session_db_id, student_id, message_type, content, route_type, embedding,
        )
        await self.update_session_activity(session_db_id)
        return message_id

    def _store_message_sync(
        self,
        session_db_id: int,
        student_id: str,
        message_type: str,
        content: str,
        route_type: Optional[str],
        embedding: np.ndarray,
    ) -> int:
        try:
            db = self.mongodb_tools.get_db_connection()
            msg_id = _next_id(db, "message_id")
            db.messages.insert_one({
                "id": msg_id,
                "session_id": session_db_id,
                "student_id": student_id,
                "message_type": message_type,
                "content": content,
                "route_type": route_type,
                "embedding": embedding.tolist(),
                "created_at": datetime.now(timezone.utc),
            })
            logger.debug("Message stored", message_id=msg_id, message_type=message_type)
            return msg_id
        except Exception as exc:
            logger.error("_store_message_sync failed", error=str(exc), exc_info=True)
            raise

    async def get_recent_messages(
        self,
        student_id: str,
        limit: int = 10,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_recent_messages_sync, student_id, limit, session_id)

    def _get_recent_messages_sync(
        self, student_id: str, limit: int, session_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        try:
            db = self.mongodb_tools.get_db_connection()

            if session_id:
                # Look up the session's internal id
                session_doc = db.sessions.find_one({"session_id": session_id})
                if not session_doc:
                    return []
                query = {"student_id": student_id, "session_id": session_doc["id"]}
            else:
                query = {"student_id": student_id}

            rows = list(
                db.messages.find(query, {"embedding": 0})
                .sort("created_at", -1)
                .limit(limit)
            )
            messages = [
                {
                    "id": r["id"],
                    "message_type": r["message_type"],
                    "content": r["content"],
                    "route_type": r.get("route_type"),
                    "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                }
                for r in rows
            ]
            return list(reversed(messages))
        except Exception as exc:
            logger.error("_get_recent_messages_sync failed", student_id=student_id, error=str(exc))
            return []

    async def search_messages_by_embedding(
        self,
        student_id: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        time_window_days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            self._search_messages_by_embedding_sync,
            student_id, query_embedding, top_k, time_window_days,
        )

    def _search_messages_by_embedding_sync(
        self,
        student_id: str,
        query_embedding: np.ndarray,
        top_k: int,
        time_window_days: Optional[int],
    ) -> List[Dict[str, Any]]:
        try:
            db = self.mongodb_tools.get_db_connection()
            query: dict = {"student_id": student_id}
            if time_window_days:
                cutoff = datetime.now(timezone.utc) - timedelta(days=time_window_days)
                query["created_at"] = {"$gte": cutoff}

            docs = list(db.messages.find(query))

            # Compute cosine similarity in Python
            scored = []
            for doc in docs:
                emb = doc.get("embedding")
                if emb is None:
                    continue
                sim = _cosine_similarity(query_embedding, np.array(emb, dtype=np.float32))
                scored.append((sim, doc))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:top_k]

            return [
                {
                    "id": doc["id"],
                    "message_type": doc["message_type"],
                    "content": doc["content"],
                    "route_type": doc.get("route_type"),
                    "created_at": doc["created_at"].isoformat() if doc.get("created_at") else None,
                    "similarity": sim,
                }
                for sim, doc in top
            ]
        except Exception as exc:
            logger.error("_search_messages_by_embedding_sync failed", error=str(exc))
            return []

    async def get_full_session_history(self, session_id: str) -> Dict[str, Any]:
        return await asyncio.to_thread(self._get_full_session_history_sync, session_id)

    def _get_full_session_history_sync(self, session_id: str) -> Dict[str, Any]:
        try:
            db = self.mongodb_tools.get_db_connection()
            s = db.sessions.find_one({"session_id": session_id})
            if not s:
                logger.warning("Session not found", session_id=session_id)
                return {"session": None, "messages": []}

            msg_rows = list(
                db.messages.find({"session_id": s["id"]}, {"embedding": 0})
                .sort("created_at", 1)
            )
            return {
                "session": {
                    "session_db_id": s["id"],
                    "session_id": s["session_id"],
                    "student_id": s["student_id"],
                    "started_at": s["started_at"].isoformat() if s.get("started_at") else None,
                    "last_activity_at": s["last_activity_at"].isoformat() if s.get("last_activity_at") else None,
                    "ended_at": s["ended_at"].isoformat() if s.get("ended_at") else None,
                    "message_count": s.get("message_count", 0),
                    "is_active": s.get("is_active", False),
                },
                "messages": [
                    {
                        "id": r["id"],
                        "message_type": r["message_type"],
                        "content": r["content"],
                        "route_type": r.get("route_type"),
                        "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                    }
                    for r in msg_rows
                ],
            }
        except Exception as exc:
            logger.error("_get_full_session_history_sync failed", session_id=session_id, error=str(exc), exc_info=True)
            raise

    # ── Fact management ───────────────────────────────────────────────────────

    async def store_fact(
        self,
        student_id: str,
        fact_type: str,
        fact_text: str,
        fact_value: Optional[str] = None,
        session: Optional[str] = None,
        source_message_id: Optional[int] = None,
        change_type: str = "created",
        change_reason: Optional[str] = None,
        old_fact_id: Optional[int] = None,
    ) -> int:
        if fact_type not in VALID_FACT_TYPES:
            raise ValueError(f"Invalid fact_type {fact_type!r}. Must be one of {VALID_FACT_TYPES}.")
        if change_type not in VALID_CHANGE_TYPES:
            raise ValueError(f"Invalid change_type {change_type!r}. Must be one of {VALID_CHANGE_TYPES}.")

        embedding = await self.embed_text(fact_text)
        new_fact_id = await asyncio.to_thread(
            self._store_fact_sync,
            student_id, fact_type, fact_text, fact_value, session,
            source_message_id, change_type, change_reason, old_fact_id, embedding,
        )
        logger.info(
            "Fact stored",
            new_fact_id=new_fact_id,
            change_type=change_type,
            fact_preview=fact_text[:60],
            student_id=student_id,
        )
        return new_fact_id

    def _store_fact_sync(
        self,
        student_id: str,
        fact_type: str,
        fact_text: str,
        fact_value: Optional[str],
        session: Optional[str],
        source_message_id: Optional[int],
        change_type: str,
        change_reason: Optional[str],
        old_fact_id: Optional[int],
        embedding: np.ndarray,
    ) -> int:
        db = self.mongodb_tools.get_db_connection()
        now = datetime.now(timezone.utc)

        if old_fact_id:
            db.user_facts.update_one(
                {"id": old_fact_id},
                {"$set": {"is_active": False, "updated_at": now}},
            )

        new_fact_id = _next_id(db, "user_fact_id")
        db.user_facts.insert_one({
            "id": new_fact_id,
            "student_id": student_id,
            "session": session,
            "fact_type": fact_type,
            "fact_text": fact_text,
            "fact_value": fact_value,
            "source_message_id": source_message_id,
            "is_active": True,
            "embedding": embedding.tolist(),
            "superseded_by": None,
            "created_at": now,
            "updated_at": now,
        })

        if old_fact_id:
            db.user_facts.update_one(
                {"id": old_fact_id},
                {"$set": {"superseded_by": new_fact_id}},
            )

        db.fact_changes.insert_one({
            "id": _next_id(db, "fact_change_id"),
            "student_id": student_id,
            "old_fact_id": old_fact_id,
            "new_fact_id": new_fact_id,
            "change_type": change_type,
            "change_reason": change_reason,
            "triggered_by_message_id": source_message_id,
            "changed_at": now,
        })

        return new_fact_id

    async def get_active_facts(
        self,
        student_id: str,
        fact_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_active_facts_sync, student_id, fact_type, limit)

    def _get_active_facts_sync(
        self, student_id: str, fact_type: Optional[str], limit: Optional[int]
    ) -> List[Dict[str, Any]]:
        try:
            db = self.mongodb_tools.get_db_connection()
            query: dict = {"student_id": student_id, "is_active": True}
            if fact_type:
                query["fact_type"] = fact_type

            cursor = db.user_facts.find(query, {"embedding": 0}).sort("created_at", -1)
            if limit:
                cursor = cursor.limit(limit)

            return [
                {
                    "id": r["id"],
                    "fact_type": r["fact_type"],
                    "fact_text": r["fact_text"],
                    "fact_value": r.get("fact_value"),
                    "session": r.get("session"),
                    "created_at": r["created_at"].isoformat() if r.get("created_at") else None,
                    "updated_at": r["updated_at"].isoformat() if r.get("updated_at") else None,
                    "source_message_id": r.get("source_message_id"),
                }
                for r in cursor
            ]
        except Exception as exc:
            logger.error("_get_active_facts_sync failed", student_id=student_id, error=str(exc))
            return []

    async def search_facts_by_embedding(
        self,
        student_id: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        fact_type: Optional[str] = None,
        only_active: bool = True,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(
            self._search_facts_by_embedding_sync,
            student_id, query_embedding, top_k, fact_type, only_active,
        )

    def _search_facts_by_embedding_sync(
        self,
        student_id: str,
        query_embedding: np.ndarray,
        top_k: int,
        fact_type: Optional[str],
        only_active: bool,
    ) -> List[Dict[str, Any]]:
        try:
            db = self.mongodb_tools.get_db_connection()
            query: dict = {"student_id": student_id}
            if only_active:
                query["is_active"] = True
            if fact_type:
                query["fact_type"] = fact_type

            docs = list(db.user_facts.find(query))

            scored = []
            for doc in docs:
                emb = doc.get("embedding")
                if emb is None:
                    continue
                sim = _cosine_similarity(query_embedding, np.array(emb, dtype=np.float32))
                scored.append((sim, doc))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:top_k]

            return [
                {
                    "id": doc["id"],
                    "fact_type": doc["fact_type"],
                    "fact_text": doc["fact_text"],
                    "fact_value": doc.get("fact_value"),
                    "session": doc.get("session"),
                    "created_at": doc["created_at"].isoformat() if doc.get("created_at") else None,
                    "updated_at": doc["updated_at"].isoformat() if doc.get("updated_at") else None,
                    "is_active": doc.get("is_active"),
                    "superseded_by": doc.get("superseded_by"),
                    "similarity": sim,
                }
                for sim, doc in top
            ]
        except Exception as exc:
            logger.error("_search_facts_by_embedding_sync failed", error=str(exc))
            return []

    async def get_fact_history(self, fact_id: int) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_fact_history_sync, fact_id)

    def _get_fact_history_sync(self, fact_id: int) -> List[Dict[str, Any]]:
        try:
            db = self.mongodb_tools.get_db_connection()
            history: list[dict] = []
            current_id: Optional[int] = fact_id
            visited: set[int] = set()

            while current_id and current_id not in visited:
                visited.add(current_id)
                row = db.user_facts.find_one({"id": current_id}, {"embedding": 0})
                if not row:
                    break
                history.append({
                    "id": row["id"],
                    "fact_type": row["fact_type"],
                    "fact_text": row["fact_text"],
                    "fact_value": row.get("fact_value"),
                    "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
                    "updated_at": row["updated_at"].isoformat() if row.get("updated_at") else None,
                    "is_active": row.get("is_active"),
                    "superseded_by": row.get("superseded_by"),
                })
                prev = db.user_facts.find_one({"superseded_by": current_id}, {"id": 1})
                current_id = prev["id"] if prev else None

            return list(reversed(history))
        except Exception as exc:
            logger.error("_get_fact_history_sync failed", fact_id=fact_id, error=str(exc))
            return []

    async def get_fact_changes(
        self,
        student_id: str,
        limit: int = 20,
        change_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_fact_changes_sync, student_id, limit, change_type)

    def _get_fact_changes_sync(
        self, student_id: str, limit: int, change_type: Optional[str]
    ) -> List[Dict[str, Any]]:
        try:
            db = self.mongodb_tools.get_db_connection()
            query: dict = {"student_id": student_id}
            if change_type:
                query["change_type"] = change_type

            fc_docs = list(
                db.fact_changes.find(query)
                .sort("changed_at", -1)
                .limit(limit)
            )

            results = []
            for fc in fc_docs:
                old_fact = db.user_facts.find_one({"id": fc.get("old_fact_id")}, {"fact_text": 1}) if fc.get("old_fact_id") else None
                new_fact = db.user_facts.find_one({"id": fc["new_fact_id"]}, {"fact_text": 1})

                results.append({
                    "id": fc["id"],
                    "old_fact_id": fc.get("old_fact_id"),
                    "new_fact_id": fc["new_fact_id"],
                    "change_type": fc["change_type"],
                    "change_reason": fc.get("change_reason"),
                    "changed_at": fc["changed_at"].isoformat() if fc.get("changed_at") else None,
                    "old_fact_text": old_fact["fact_text"] if old_fact else None,
                    "new_fact_text": new_fact["fact_text"] if new_fact else None,
                })

            return results
        except Exception as exc:
            logger.error("_get_fact_changes_sync failed", student_id=student_id, error=str(exc))
            return []
