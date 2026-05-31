"""
Fact Processor — async LLM-driven extraction and deduplication of user facts.

Every LLM call uses the project's LLMClient pattern:
  - llm_client.with_structured_output(PydanticSchema) for structured outputs
  - await structured_llm.ainvoke([HumanMessage(content=prompt)])

This matches exactly how the orchestrator node and dashboard tool call the LLM,
eliminating the fragile JSON string parsing that was used before.

Design notes:
- All methods are async, consistent with the rest of the codebase.
- Confidence thresholding uses MEMORY_FACT_CONFIDENCE_THRESHOLD.
- No module-level state — all state lives in the injected MemoryManager.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from langchain_core.messages import HumanMessage

from utils.config import *
from utils.apexchat.schemas.models import (
    ContradictionCheck,
    DetectedFact,
    DetectedFactList,
    ReformulatedFact,
)

logger = structlog.get_logger(__name__)


class FactProcessor:
    """
    Detects, reformulates, and stores facts extracted from user messages.

    Args:
        memory_manager: MemoryManager instance for fact persistence.
        llm_client: LLMClient instance (the same one used by tools — provides
            with_structured_output() and ainvoke_with_retry()).
    """

    def __init__(self, memory_manager, llm_client) -> None:
        self.memory = memory_manager
        self._llm = llm_client
        logger.info("FactProcessor initialised")

    # ── Fact detection ────────────────────────────────────────────────────────

    async def detect_facts_in_message(self, message: str) -> List[DetectedFact]:
        """
        Use the LLM with structured output to detect facts in the user's message.

        Returns only facts whose confidence meets MEMORY_FACT_CONFIDENCE_THRESHOLD.
        """
        structured_llm = self._llm.with_structured_output(DetectedFactList)

        prompt = f"""You are an intelligent Memory Archivist. Analyze the user's message to extract facts.

User Message: "{message}"

Priority order:

PRIORITY 1: EXPLICIT COMMANDS (Override strict filtering)
If the user explicitly asks to "save", "remember", "note", or "don't forget" information:
   - EXTRACT IT even if trivial or about a 3rd party.
   - Strip the command part ("Remember that X" → fact_text = "X").
   - TASKS ("Remember to buy milk") → do NOT extract.
   - FACTS ("Remember that I drink almond milk") → EXTRACT.

PRIORITY 2: IMPLICIT FACTS (Strict filtering — no explicit command)
   - Subject must be the USER ("I", "me").
   - Information must have long-term biographical or preference value.
   - Ignore 3rd-party gossip unless it describes the user's relationship.


critical instructions:
- Do not save names or wallet addresses of 3rd parties mentioned in passing ("create a for x name" → ignore "x", "create a kg with wallet address y" → ignore "y").
- Only consider as fact if the user mentions the ownership or relationship to the 3rd party ("My friend Alice is a lawyer" → fact_text = "User's friend Alice is a lawyer", "I have a friend who is a lawyer" → ignore, "my wallet address is x" → fact_text = "User's wallet address is x").


FACT TYPES: profile, preference, event, goal, habit

Return a facts list. Return an empty list if nothing qualifies."""

        try:
            result: DetectedFactList = await structured_llm.ainvoke(
                [HumanMessage(content=prompt)]
            )
            threshold = MEMORY_FACT_CONFIDENCE_THRESHOLD
            valid = [f for f in result.facts if f.confidence >= threshold]
            if valid:
                logger.info("Facts detected in message", count=len(valid))
            return valid

        except Exception as exc:
            logger.error("detect_facts_in_message failed", error=str(exc), exc_info=True)
            return []

    # ── Fact reformulation ────────────────────────────────────────────────────

    async def reformulate_fact(self, fact_text: str, fact_type: str) -> str:
        """
        Standardise a raw fact for storage using structured output.

        Strips command phrasing and normalises to "User [verb] [object]" form
        for consistent semantic matching.

        Returns the original fact_text on failure.
        """
        structured_llm = self._llm.with_structured_output(ReformulatedFact)

        prompt = f"""Standardize this fact for database storage.

Input: "{fact_text}"
Type: {fact_type}

Rules:
1. Remove command phrasing:
   "Remember that I like apples" → "User likes apples"
   "Save the fact that my code is 1234" → "User's code is 1234"
2. Start with "User".
3. Be concise and specific."""

        try:
            result: ReformulatedFact = await structured_llm.ainvoke(
                [HumanMessage(content=prompt)]
            )
            return result.canonical_text or fact_text

        except Exception as exc:
            logger.error("reformulate_fact failed", error=str(exc))
            return fact_text

    # ── Contradiction / update detection ─────────────────────────────────────

    async def check_for_contradictions(
        self,
        student_id: str,
        new_fact_text: str,
        new_fact_type: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Determine whether the new fact updates an existing one.

        Uses semantic search to find candidates, then asks the LLM to classify
        each relationship using structured output (ContradictionCheck).

        Returns a dict with "old_fact", "is_update", "reason" on UPDATE,
        or None when no update is detected.
        """
        new_embedding = await self.memory.embed_text(new_fact_text)
        similar_facts = await self.memory.search_facts_by_embedding(
            student_id=student_id,
            query_embedding=new_embedding,
            top_k=3,
            fact_type=new_fact_type,
            only_active=True,
        )

        threshold = MEMORY_SEMANTIC_SIMILARITY_THRESHOLD
        structured_llm = self._llm.with_structured_output(ContradictionCheck)

        for existing in similar_facts:
            if existing["similarity"] < threshold:
                continue

            prompt = f"""Compare these two facts about the same user.

Old: "{existing['fact_text']}"
New: "{new_fact_text}"

Classify the relationship:
- UPDATE: New fact replaces old (e.g. "I moved to NY" replaces "I live in LA").
- IGNORE: Both facts say the same thing.
- APPEND: Both are independently true (e.g. "I like red" and "I like blue")."""

            try:
                result: ContradictionCheck = await structured_llm.ainvoke(
                    [HumanMessage(content=prompt)]
                )
                if result.action == "UPDATE":
                    logger.info(
                        "Fact update detected",
                        old_fact_id=existing["id"],
                        reason=result.reason,
                    )
                    return {
                        "old_fact": existing,
                        "is_update": True,
                        "reason": result.reason,
                    }
            except Exception as exc:
                logger.warning(
                    "Contradiction check failed for candidate",
                    old_fact_id=existing["id"],
                    error=str(exc),
                )
                continue

        return None

    # ── Public pipeline ───────────────────────────────────────────────────────

    async def process_and_store_facts(
        self,
        student_id: str,
        message: str,
        session: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Full async pipeline: detect → reformulate → deduplicate → persist.

        Args:
            student_id: User identifier.
            message: Raw user message to analyse.
            session: Optional session identifier to tag stored facts.

        Returns:
            List of {"status": "created"|"updated", "text": <canonical text>}.
        """
        detected = await self.detect_facts_in_message(message)
        if not detected:
            return []

        stored: List[Dict[str, Any]] = []

        for fact in detected:
            canonical_text = await self.reformulate_fact(fact.fact_text, fact.fact_type)
            contradiction = await self.check_for_contradictions(
                student_id, canonical_text, fact.fact_type
            )

            if contradiction:
                logger.info("Updating existing fact", reason=contradiction["reason"])
                await self.memory.store_fact(
                    student_id=student_id,
                    fact_type=fact.fact_type,
                    fact_text=canonical_text,
                    session=session,
                    change_type="updated",
                    change_reason=contradiction["reason"],
                    old_fact_id=contradiction["old_fact"]["id"],
                )
                stored.append({"status": "updated", "text": canonical_text})
            else:
                logger.info("Storing new fact", fact_preview=canonical_text[:60])
                await self.memory.store_fact(
                    student_id=student_id,
                    fact_type=fact.fact_type,
                    fact_text=canonical_text,
                    session=session,
                    change_type="created",
                )
                stored.append({"status": "created", "text": canonical_text})

        return stored
