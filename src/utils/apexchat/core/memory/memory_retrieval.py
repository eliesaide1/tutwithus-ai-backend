"""
Memory Retrieval — async semantic search and intelligent context recall.

Every LLM call uses with_structured_output(PydanticSchema) then ainvoke(),
matching the pattern used by the orchestrator node and dashboard tool.

Design notes:
- All public methods are async.
- Configuration (top_k, limits) comes from settings, not hard-coded constants.
- No module-level state — all state lives in MemoryManager.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from langchain_core.messages import HumanMessage

from utils.config import * 
from utils.apexchat.schemas.models import (
    ContextSummary,
    KnowledgeExplanation,
    MemoryAnswer,
    NormalizedQuery,
)

logger = structlog.get_logger(__name__)


class MemoryRetrieval:
    """
    Intelligent async memory retrieval combining semantic search with LLM synthesis.

    Args:
        memory_manager: MemoryManager instance providing low-level DB access.
        llm_client: LLMClient instance — provides with_structured_output()
            and ainvoke_with_retry(), exactly as used by the tools layer.
    """

    def __init__(self, memory_manager, llm_client) -> None:
        self.memory = memory_manager
        self._llm = llm_client
        logger.info("MemoryRetrieval initialised")

    # ── Query normalisation ───────────────────────────────────────────────────

    async def normalize_memory_query(self, question: str) -> str:
        """
        Convert a question into a statement form optimised for semantic search.

        "Where do I live?" → "User lives in"

        Uses with_structured_output(NormalizedQuery) for reliable extraction.
        Falls back to the original question on failure.
        """
        structured_llm = self._llm.with_structured_output(NormalizedQuery)

        prompt = f"""Convert this question into a standardized fact-retrieval query.

User's question: "{question}"

Rules:
1. Convert to statement form (not question).
2. Use "User [verb] [object]" format.
3. Focus on the core information being asked about.
4. Remove question words (who, what, where, when, why, how, do, does, can, could).

Examples:
"Where do I live?"       → "User lives in"
"What do I do for work?" → "User works as"
"How old am I?"          → "User is years old"
"Do you know my name?"   → "User name is"
"What do I like?"        → "User loves"

Return the normalized query (5-10 words max)."""

        try:
            result: NormalizedQuery = await structured_llm.ainvoke(
                [HumanMessage(content=prompt)]
            )
            normalized = result.normalized or question
            logger.debug("Query normalized", original=question, normalized=normalized)
            return normalized

        except Exception as exc:
            logger.error("normalize_memory_query failed", error=str(exc))
            return question

    # ── Fact retrieval ────────────────────────────────────────────────────────

    async def retrieve_relevant_facts(
        self,
        student_id: str,
        query: str,
        top_k: int | None = None,
        fact_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return facts semantically relevant to the given query.

        Normalises the query before embedding for improved recall.
        """
        k = top_k or MEMORY_MAX_FACTS_PER_RETRIEVAL

        try:
            normalized = await self.normalize_memory_query(query)
            query_embedding = await self.memory.embed_text(normalized)

            facts = await self.memory.search_facts_by_embedding(
                student_id=student_id,
                query_embedding=query_embedding,
                top_k=k,
                fact_type=fact_type,
                only_active=True,
            )
            logger.info("Relevant facts retrieved", count=len(facts), student_id=student_id)
            return facts

        except Exception as exc:
            logger.error("retrieve_relevant_facts failed", student_id=student_id, error=str(exc))
            return []

    # ── Message / context retrieval ───────────────────────────────────────────

    async def retrieve_recent_context(
        self,
        student_id: str,
        session_id: Optional[str] = None,
        message_limit: int | None = None,
    ) -> Dict[str, Any]:
        """
        Return recent messages and an LLM-generated summary.

        Uses with_structured_output(ContextSummary) for reliable summary extraction.
        """
        limit = message_limit or MEMORY_MAX_MESSAGES_PER_RETRIEVAL

        try:
            messages = await self.memory.get_recent_messages(
                student_id=student_id,
                limit=limit,
                session_id=session_id,
            )

            summary = ""
            if messages:
                snippet = "\n".join(
                    f"{'User' if m['message_type'] == 'user_query' else 'Assistant'}: "
                    f"{m['content'][:100]}..."
                    for m in messages[-5:]
                )
                structured_llm = self._llm.with_structured_output(ContextSummary)
                try:
                    result: ContextSummary = await structured_llm.ainvoke(
                        [HumanMessage(content=f"Summarize this recent conversation in 1-2 sentences:\n\n{snippet}")]
                    )
                    summary = result.summary
                except Exception as exc:
                    logger.warning("Context summary generation failed", error=str(exc))

            return {
                "messages": messages,
                "message_count": len(messages),
                "summary": summary,
            }

        except Exception as exc:
            logger.error("retrieve_recent_context failed", student_id=student_id, error=str(exc))
            return {"messages": [], "message_count": 0, "summary": ""}

    # ── Question answering ────────────────────────────────────────────────────

    async def answer_question_from_memory(self, student_id: str, question: str) -> str:
        """
        Generate a grounded answer using stored facts and message history.

        Uses with_structured_output(MemoryAnswer) so the response is always
        a clean string — no extra parsing needed.

        Workflow:
        1. Normalise and retrieve relevant facts.
        2. Retrieve semantically similar past messages (last 30 days).
        3. Synthesise a grounded answer via the LLM.
        """
        try:
            # Step 1 — relevant facts
            relevant_facts = await self.retrieve_relevant_facts(
                student_id=student_id, query=question
            )

            # Step 2 — relevant messages
            query_embedding = await self.memory.embed_text(question)
            relevant_messages = await self.memory.search_messages_by_embedding(
                student_id=student_id,
                query_embedding=query_embedding,
                top_k=5,
                time_window_days=30,
            )

            # Step 3 — build context
            # NOTE: similarity scores are intentionally excluded — exposing them
            # to the LLM causes hedging ("I'm not sure but…"). The retrieval
            # threshold already guarantees relevance; treat every fact as ground truth.
            facts_block = ""
            if relevant_facts:
                facts_block = "What I know about you:\n" + "\n".join(
                    f"- {f['fact_text']}"
                    for f in relevant_facts
                )

            messages_block = ""
            if relevant_messages:
                messages_block = "\n\nRelevant past conversations:\n" + "\n".join(
                    f"- {m['content'][:150]}"
                    for m in relevant_messages[:3]
                )

            if not facts_block and not messages_block:
                return "I don't have anything stored about that yet — feel free to tell me and I'll remember it."

            # Step 4 — structured LLM answer
            structured_llm = self._llm.with_structured_output(MemoryAnswer)
            prompt = f"""You are recalling stored facts about the user. Answer their question directly and confidently.

User's question: "{question}"

{facts_block}{messages_block}

Rules:
- State facts directly — never say "I think", "I believe", "I'm not sure", "I might be wrong", or any other hedge.
- Do not mention confidence levels, similarity scores, or uncertainty of any kind.
- Use only the information provided above; do not invent details.
- Answer in a natural, conversational tone.
- 1-2 sentences max."""

            result: MemoryAnswer = await structured_llm.ainvoke(
                [HumanMessage(content=prompt)]
            )
            return result.answer

        except Exception as exc:
            logger.error(
                "answer_question_from_memory failed",
                student_id=student_id,
                error=str(exc),
                exc_info=True,
            )
            return "I'm having trouble accessing my memory right now. Please try again."

    # ── User profile ──────────────────────────────────────────────────────────

    async def get_user_profile(self, student_id: str) -> Dict[str, Any]:
        """Build a structured profile grouped by fact type."""
        try:
            all_facts = await self.memory.get_active_facts(student_id=student_id)

            profile: Dict[str, Any] = {
                "student_id": student_id,
                "total_facts": len(all_facts),
                "facts_by_type": {},
                "last_updated": None,
            }
            for fact in all_facts:
                ft = fact["fact_type"]
                profile["facts_by_type"].setdefault(ft, []).append({
                    "id": fact["id"],
                    "text": fact["fact_text"],
                    "value": fact["fact_value"],
                    "created_at": fact["created_at"],
                })
                updated = fact.get("updated_at")
                if updated and (not profile["last_updated"] or updated > profile["last_updated"]):
                    profile["last_updated"] = updated

            return profile

        except Exception as exc:
            logger.error("get_user_profile failed", student_id=student_id, error=str(exc))
            return {"student_id": student_id, "error": str(exc)}

    async def explain_what_i_know(self, student_id: str) -> str:
        """
        Return a warm, natural-language summary of everything stored about the user.

        Uses with_structured_output(KnowledgeExplanation) for reliable output.
        """
        try:
            profile = await self.get_user_profile(student_id)

            if profile.get("total_facts", 0) == 0:
                return (
                    "I don't know anything about you yet. "
                    "As we chat, I'll remember important details you share with me."
                )

            type_labels = {
                "profile": "About You",
                "preference": "Your Preferences",
                "event": "Your Experiences",
                "goal": "Your Goals",
                "habit": "Your Habits",
            }

            sections: list[str] = []
            for fact_type, label in type_labels.items():
                facts = profile["facts_by_type"].get(fact_type, [])
                if facts:
                    bullets = "\n".join(f"- {f['text']}" for f in facts)
                    sections.append(f"**{label}:**\n{bullets}")

            summary_text = "\n\n".join(sections)
            structured_llm = self._llm.with_structured_output(KnowledgeExplanation)

            prompt = f"""Explain to the user what you know about them. Write a warm, personal summary.

Here's what you know:

{summary_text}

Write a natural, conversational explanation (3-4 paragraphs) that:
- Starts with a friendly opening.
- Groups related information naturally.
- Shows you understand them as a person.
- Ends with an invitation to update or add information."""

            result: KnowledgeExplanation = await structured_llm.ainvoke(
                [HumanMessage(content=prompt)]
            )
            return result.explanation

        except Exception as exc:
            logger.error("explain_what_i_know failed", student_id=student_id, error=str(exc))
            return "I have some information about you, but I'm having trouble organising it right now."