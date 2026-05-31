"""
Memory Recall Tool — answers questions about stored user facts and history.

Responsibilities
----------------
1. Validate that a user_id is present in state.
2. Resolve the global ApexchatMemorySystem singleton.
3. Delegate to memory_system.answer_memory_query() for semantic recall.
4. Handle all failure modes gracefully and log every path.

Architecture
------------
This tool is intentionally thin: all retrieval logic lives in
Apexchat.core.memory (MemoryRetrieval → MemoryManager → PostgreSQL + FAISS).
The tool is only responsible for extracting the right inputs from WorkflowState
and returning a clean string — exactly the same contract as every other BaseTool.

Integration checklist (already done by the project):
- ToolType.MEMORY added to Apexchat/schemas/models.py
- MemoryRecallTool registered in Apexchat/tools/__init__.py TOOL_REGISTRY
- "memory" routing description present in workflow.py orchestrator prompt
"""

from __future__ import annotations

import time

import structlog

from utils.apexchat.schemas.models import WorkflowState
from utils.apexchat.tools.general import BaseTool
from utils.apexchat.core.status_stream import emit_status

logger = structlog.get_logger(__name__)

# Sentinel returned when the memory system is unavailable so the caller can
# distinguish a "system down" response from a legitimate "nothing stored" one.
_MEMORY_UNAVAILABLE = (
    "I'm having trouble accessing my memory right now. Please try again later."
)
_NO_user_id = "I need your user ID to look up stored information."


class MemoryRecallTool(BaseTool):
    """
    Handles all queries that ask the assistant to recall stored user information.

    Examples of routed queries
    --------------------------
    - "What is my name?"
    - "Where am I located?"
    - "What are my preferences?"
    - "What do you know about me?"
    - "Have I told you my job title?"

    The tool resolves the ApexchatMemorySystem singleton at *call time* (not at
    construction time) so it is safe to instantiate before the memory system
    has been initialised during application startup.
    """

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return (
            "Answers questions about stored user facts and preferences by "
            "performing semantic search over the user's memory store."
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_memory_system():
        """
        Return the global ApexchatMemorySystem, or None if it is not yet ready.

        Using a late import + try/except means the tool degrades gracefully
        instead of crashing the whole workflow when memory is unavailable.
        """
        try:
            from apexchat.core.memory import get_memory_system  # noqa: PLC0415
            return get_memory_system()
        except RuntimeError:
            # init_memory_system() has not been called yet
            logger.warning("MemoryRecallTool — memory system not initialised")
            return None
        except Exception as exc:  # pragma: no cover
            logger.error("MemoryRecallTool — unexpected error resolving memory system", error=str(exc))
            return None

    # ── BaseTool contract ─────────────────────────────────────────────────────

    async def execute(self, state: WorkflowState) -> str:
        """
        Recall stored user information relevant to state.user_message.

        Args:
            state: Current workflow state.  The following fields are used:
                - state.user_message  — the natural-language question
                - state.user_id       — used to scope the memory lookup
                - state.session_id    — used only for structured logging

        Returns:
            A natural-language string answer drawn from stored memory, or an
            informative fallback message when memory is unavailable / empty.
        """
        emit_status("tool_memory")
        start_time = time.perf_counter()

        # ── Guard: user identity ──────────────────────────────────────────────
        user_id = (state.user_id or "").strip()
        if not user_id:
            logger.warning(
                "MemoryRecallTool — user_id missing, cannot recall",
                session_id=state.session_id,
            )
            return _NO_user_id

        # ── Guard: memory system availability ────────────────────────────────
        memory_system = self._resolve_memory_system()
        if memory_system is None:
            logger.error(
                "MemoryRecallTool — memory system unavailable",
                session_id=state.session_id,
                user_id=user_id,
            )
            return _MEMORY_UNAVAILABLE

        # ── Core recall ───────────────────────────────────────────────────────
        query = state.user_message
        logger.info(
            "MemoryRecallTool executing",
            session_id=state.session_id,
            user_id=user_id,
            query_preview=query[:80],
        )

        try:
            answer = await memory_system.answer_memory_query(
                student_id=user_id,
                query=query,
            )
        except Exception as exc:
            logger.error(
                "MemoryRecallTool — answer_memory_query raised",
                session_id=state.session_id,
                user_id=user_id,
                error=str(exc),
                exc_info=True,
            )
            return _MEMORY_UNAVAILABLE

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "MemoryRecallTool completed",
            session_id=state.session_id,
            user_id=user_id,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return answer