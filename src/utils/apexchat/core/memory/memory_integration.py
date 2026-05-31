"""
VChat Memory System — async top-level integration facade.

Wires together MemoryManager, FactProcessor, and MemoryRetrieval into a single
entry point used by the workflow layer. All public methods are async.

MongoDB version: replaces the former psycopg2 + PostgreSQL implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
import asyncio

from utils.apexchat.core.memory.embedding_wrapper import get_embedder
from utils.apexchat.core.memory.fact_processor import FactProcessor
from utils.apexchat.core.memory.memory_manager import MemoryManager
from utils.apexchat.core.memory.memory_retrieval import MemoryRetrieval
from utils.Mongodb_tools import MONGODB_TOOLS


logger = structlog.get_logger(__name__)
mongodb_tools = MONGODB_TOOLS()

class ApexchatMemorySystem:
    """
    Unified async memory facade for the VChat assistant.

    Args:
        mongodb_tools: MongoDB connection manager (MONGODB_TOOLS instance).
        llm_client: LLMClient instance — used by FactProcessor and
            MemoryRetrieval for with_structured_output() calls, and by
            MemoryManager for session summarisation.
    """

    def __init__(self, mongodb_tools, llm_client) -> None:
        try:
            embedder = get_embedder()
            self.llm = llm_client
            self.memory_manager = MemoryManager(
                mongodb_tools=mongodb_tools,
                embedding_model=embedder,
                llm_client=llm_client,
            )
            self.fact_processor = FactProcessor(
                memory_manager=self.memory_manager,
                llm_client=llm_client,
            )
            self.memory_retrieval = MemoryRetrieval(
                memory_manager=self.memory_manager,
                llm_client=llm_client,
            )
            logger.info("ApexchatMemorySystem initialised")
        except Exception as exc:
            logger.error("ApexchatMemorySystem init failed", error=str(exc), exc_info=True)
            raise

    # ── full session history ───────────────────────────────────────────────────

    async def get_full_session_history(self, session_id: str) -> Dict[str, Any]:
        """
        Get full history for a session including all messages.
        Delegates to MemoryManager which handles MongoDB queries.
        """
        try:
            result = await self.memory_manager.get_full_session_history(session_id)
            msg_count = len(result.get("messages", []))
            logger.debug(f"Retrieved {msg_count} messages for session {session_id}")
            return result
        except Exception as e:
            logger.error(f"Error getting full session history: {e}", exc_info=True)
            raise

    # ── closing session with summary ──────────────────────────────────────────

    async def close_session_with_summary(self, session_id: str):
        """
        Close a session, generate title/summary using LLM, and update DB.
        Delegates to MemoryManager which handles MongoDB queries.
        """
        try:
            await self.memory_manager.close_session_with_summary(session_id)
        except Exception as e:
            logger.error(f"Error in close_session_with_summary: {e}")
            await self.end_session(session_id)

    # ── Persistence ───────────────────────────────────────────────────────────

    async def persist_conversation_turn(
        self,
        student_id: str,
        user_query: str,
        assistant_response: str,
        route_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Persist session metadata, store messages, and extract facts concurrently.
        """
        try:
            logger.info("Persisting conversation turn", student_id=student_id)

            sid = session_id if (session_id and session_id.strip()) else None
            session_db_id, resolved_sid = await self.memory_manager.get_or_create_session(
                student_id=student_id,
                session_id=sid,
            )

            user_msg_task = self.memory_manager.store_message(
                session_db_id=session_db_id,
                student_id=student_id,
                message_type="user_query",
                content=user_query,
                route_type=None
            )

            assistant_msg_task = self.memory_manager.store_message(
                session_db_id=session_db_id,
                student_id=student_id,
                message_type="assistant_response",
                content=assistant_response,
                route_type=route_type
            )

            facts_task = self.fact_processor.process_and_store_facts(
                student_id=student_id,
                message=user_query,
                session=resolved_sid,
            )

            results = await asyncio.gather(
                user_msg_task,
                assistant_msg_task,
                facts_task,
                return_exceptions=True
            )

            user_msg_id = results[0] if not isinstance(results[0], Exception) else None
            assistant_msg_id = results[1] if not isinstance(results[1], Exception) else None

            stored_facts = results[2] if not isinstance(results[2], Exception) else []
            if isinstance(results[2], Exception):
                logger.warning("Fact extraction failed", error=str(results[2]))

            if stored_facts:
                logger.info("Facts stored", count=len(stored_facts))

            return {
                "success": True,
                "session_id": resolved_sid,
                "session_db_id": session_db_id,
                "user_message_id": user_msg_id,
                "assistant_message_id": assistant_msg_id,
                "facts_stored": len(stored_facts),
                "facts": stored_facts,
            }

        except Exception as exc:
            logger.error(
                "persist_conversation_turn failed",
                student_id=student_id,
                error=str(exc),
                exc_info=True,
            )
            return {"success": False, "error": str(exc), "facts_stored": 0}

    # ── Retrieval ─────────────────────────────────────────────────────────────

    async def retrieve_conversation_context(
        self,
        student_id: str,
        current_query: str,
        session_id: Optional[str] = None,
        include_facts: bool = True,
        include_messages: bool = True,
        max_facts: int | None = None,
        max_messages: int | None = None,
    ) -> Dict[str, Any]:
        """Gather relevant context before generating a response."""
        context: Dict[str, Any] = {
            "student_id": student_id,
            "query": current_query,
            "facts": [],
            "messages": [],
            "summary": "",
        }
        try:
            if include_facts:
                context["facts"] = await self.memory_retrieval.retrieve_relevant_facts(
                    student_id=student_id,
                    query=current_query,
                    top_k=max_facts,
                )
            if include_messages:
                recent = await self.memory_retrieval.retrieve_recent_context(
                    student_id=student_id,
                    session_id=session_id,
                    message_limit=max_messages,
                )
                context["messages"] = recent.get("messages", [])
                context["summary"] = recent.get("summary", "")

            logger.info(
                "Context retrieved",
                facts_count=len(context["facts"]),
                messages_count=len(context["messages"]),
            )
        except Exception as exc:
            logger.error("retrieve_conversation_context failed", student_id=student_id, error=str(exc), exc_info=True)
            context["error"] = str(exc)

        return context

    async def answer_memory_query(self, student_id: str, query: str) -> str:
        try:
            return await self.memory_retrieval.answer_question_from_memory(
                student_id=student_id, question=query
            )
        except Exception as exc:
            logger.error("answer_memory_query failed", student_id=student_id, error=str(exc))
            return "I'm having trouble accessing my memory right now."

    # ── Fact management ───────────────────────────────────────────────────────

    async def get_user_facts(
        self, student_id: str, fact_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        try:
            return await self.memory_manager.get_active_facts(
                student_id=student_id, fact_type=fact_type
            )
        except Exception as exc:
            logger.error("get_user_facts failed", student_id=student_id, error=str(exc))
            return []

    async def get_fact_history(self, fact_id: int) -> List[Dict[str, Any]]:
        try:
            return await self.memory_manager.get_fact_history(fact_id)
        except Exception as exc:
            logger.error("get_fact_history failed", fact_id=fact_id, error=str(exc))
            return []

    # ── User profile ──────────────────────────────────────────────────────────

    async def get_user_profile(self, student_id: str) -> Dict[str, Any]:
        try:
            return await self.memory_retrieval.get_user_profile(student_id)
        except Exception as exc:
            logger.error("get_user_profile failed", student_id=student_id, error=str(exc))
            return {"student_id": student_id, "error": str(exc)}

    async def explain_what_i_know(self, student_id: str) -> str:
        try:
            return await self.memory_retrieval.explain_what_i_know(student_id)
        except Exception as exc:
            logger.error("explain_what_i_know failed", student_id=student_id, error=str(exc))
            return "I'm having trouble organising what I know about you right now."

    # ── Session management ────────────────────────────────────────────────────

    async def end_session(self, session_id: str) -> None:
        try:
            await self.memory_manager.end_session(session_id)
        except Exception as exc:
            logger.error("end_session failed", session_id=session_id, error=str(exc))

    async def get_recent_sessions(
        self, student_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        try:
            return await self.memory_manager.get_recent_sessions(
                student_id=student_id, limit=limit
            )
        except Exception as exc:
            logger.error("get_recent_sessions failed", student_id=student_id, error=str(exc))
            return []

    # ── Analytics ─────────────────────────────────────────────────────────────

    async def get_memory_stats(self, student_id: str) -> Dict[str, Any]:
        try:
            facts = await self.memory_manager.get_active_facts(student_id=student_id)
            changes = await self.memory_manager.get_fact_changes(student_id=student_id, limit=100)

            facts_by_type: Dict[str, int] = {}
            for fact in facts:
                facts_by_type[fact["fact_type"]] = facts_by_type.get(fact["fact_type"], 0) + 1

            changes_by_type: Dict[str, int] = {}
            for change in changes:
                changes_by_type[change["change_type"]] = (
                    changes_by_type.get(change["change_type"], 0) + 1
                )

            created_dates = [f["created_at"] for f in facts if f.get("created_at")]
            return {
                "total_facts": len(facts),
                "facts_by_type": facts_by_type,
                "total_changes": len(changes),
                "changes_by_type": changes_by_type,
                "oldest_fact": min(created_dates) if created_dates else None,
                "newest_fact": max(created_dates) if created_dates else None,
            }
        except Exception as exc:
            logger.error("get_memory_stats failed", student_id=student_id, error=str(exc))
            return {"error": str(exc)}


# ── Module-level singleton ─────────────────────────────────────────────────────

_memory_system: ApexchatMemorySystem | None = None


def init_memory_system(mongodb_tools, llm_client) -> ApexchatMemorySystem:
    """
    Initialise the global ApexchatMemorySystem singleton.
    Call once in the FastAPI lifespan handler after all dependencies are ready.
    """
    global _memory_system
    _memory_system = ApexchatMemorySystem(mongodb_tools=mongodb_tools, llm_client=llm_client)
    logger.info("Global memory system initialised")
    return _memory_system


def get_memory_system() -> ApexchatMemorySystem:
    """Return the global ApexchatMemorySystem singleton."""
    if _memory_system is None:
        raise RuntimeError(
            "Memory system not initialised. "
            "Call init_memory_system() during application startup."
        )
    return _memory_system
