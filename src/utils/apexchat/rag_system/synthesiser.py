"""
RAG Answer Synthesiser — generates grounded answers from retrieved chunks.

Design
------
* Takes a list of RetrievedChunk objects and a user query.
* Builds a structured prompt that presents context clearly, source-attributed.
* Applies relevance filtering: chunks below min_score_threshold are dropped.
* Deduplicates near-identical chunks before synthesis (Jaccard similarity).
* Returns a structured SynthesisResult with the answer, citations, and confidence.

This module is pure logic — no I/O, no FAISS, no HTTP.  The RAGTool wires
it to the LLM client.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from apexchat.rag_system.document_store import RetrievedChunk


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """A single source reference included in the answer."""
    filename: str
    page: int          # 0-based
    chunk_idx: int
    score: float
    excerpt: str       # first 200 chars of the chunk


@dataclass
class SynthesisResult:
    """Output from the answer synthesiser."""
    answer: str
    citations: list[Citation]
    confidence: str    # "high" | "medium" | "low" | "none"
    num_chunks_used: int
    query: str


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM = """\
You are a knowledgeable assistant for an online tutoring platform.
You answer questions using the source excerpts provided below.

Source types you may receive:
- "platform_knowledge_base": Live data from the platform's database — tutors, subjects, grade levels, booking process, credits. This is authoritative and always up-to-date.
- Any other filename: A document uploaded by the user (PDF or similar).

Rules:
1. For questions about the platform (tutors, subjects, levels, booking, pricing, credits) — base your answer on "platform_knowledge_base" excerpts.
2. For questions about uploaded documents — base your answer on those document excerpts.
3. If both source types are relevant, incorporate information from both.
4. Do not include inline citations, bracketed references, or page markers in the answer.
   For example, do not add text like [platform_knowledge_base,p1] or [Doc: file.pdf, p.2].
5. If the excerpts do not contain enough information to answer, say so clearly.
6. Be concise but complete. Use markdown formatting (bullet points, headers) when it aids clarity.
7. Never fabricate information or fill gaps with assumptions.
"""

_SYNTHESIS_USER_TEMPLATE = """\
## User Question
{query}

## Source Excerpts
{context_block}

## Instructions
Answer the question using the excerpts above. \
Prioritise "platform_knowledge_base" for platform-related questions and uploaded documents for document-specific questions. \
If the answer cannot be determined from the excerpts, state that clearly and explain what information is missing.
"""


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _jaccard_similarity(a: str, b: str) -> float:
    """Token-level Jaccard similarity — fast dedup heuristic."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def deduplicate_chunks(
    chunks: list["RetrievedChunk"],
    similarity_threshold: float = 0.85,
) -> list["RetrievedChunk"]:
    """
    Remove near-duplicate chunks using Jaccard similarity.

    Keeps the highest-scoring chunk when duplicates are detected.
    Runs in O(n²) which is acceptable for typical top-k values (≤20).
    """
    if not chunks:
        return []

    # Sort by score descending so we keep the best version of duplicates
    sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
    kept: list["RetrievedChunk"] = []

    for candidate in sorted_chunks:
        is_duplicate = any(
            _jaccard_similarity(candidate.text, kept_chunk.text) >= similarity_threshold
            for kept_chunk in kept
        )
        if not is_duplicate:
            kept.append(candidate)

    return kept


def filter_by_score(
    chunks: list["RetrievedChunk"],
    min_score: float,
) -> list["RetrievedChunk"]:
    """Drop chunks whose cosine similarity falls below *min_score*."""
    return [c for c in chunks if c.score >= min_score]


def build_context_block(chunks: list["RetrievedChunk"]) -> str:
    """
    Format chunks into a numbered context block for the LLM prompt.

    Each excerpt is prefixed with its source so the model can cite it.
    """
    lines: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] Source: {chunk.filename} | Page {chunk.page + 1} | Score: {chunk.score:.3f}"
        lines.append(header)
        lines.append(chunk.text.strip())
        lines.append("")  # blank line between chunks
    return "\n".join(lines)


def build_synthesis_prompt(query: str, chunks: list["RetrievedChunk"]) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for the synthesis LLM call.
    """
    context_block = build_context_block(chunks)
    user_prompt = _SYNTHESIS_USER_TEMPLATE.format(
        query=query,
        context_block=context_block,
    )
    return _SYNTHESIS_SYSTEM, user_prompt


def extract_citations(chunks: list["RetrievedChunk"]) -> list[Citation]:
    """Build Citation objects from the chunks used for synthesis."""
    return [
        Citation(
            filename=c.filename,
            page=c.page,
            chunk_idx=c.chunk_idx,
            score=round(c.score, 4),
            excerpt=c.text[:200],
        )
        for c in chunks
    ]


_CITATION_STRIP_RE = re.compile(
    r"\s*\[(?:platform_knowledge_base|Doc:[^\]]+)\][\s\n]*"
)


def strip_rag_citations(answer: str) -> str:
    """Remove inline RAG citation markers from a generated answer."""
    cleaned = _CITATION_STRIP_RE.sub(" ", answer)
    return re.sub(r"[ \t\n]{2,}", " ", cleaned).strip()


def assess_confidence(chunks: list["RetrievedChunk"], answer: str) -> str:
    """
    Heuristic confidence assessment based on chunk scores and answer content.

    Returns "high" | "medium" | "low" | "none".
    """
    if not chunks:
        return "none"

    no_answer_phrases = [
        "cannot be determined",
        "not mentioned",
        "not found",
        "no information",
        "insufficient",
        "do not contain",
        "unable to answer",
    ]
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in no_answer_phrases):
        return "low"

    top_score = chunks[0].score if chunks else 0.0
    avg_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0.0

    if top_score >= 0.75 and avg_score >= 0.60:
        return "high"
    if top_score >= 0.55:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main Synthesiser class
# ---------------------------------------------------------------------------

class AnswerSynthesiser:
    """
    Orchestrates retrieval-result post-processing and answer generation.

    The synthesiser is stateless — it only holds configuration defaults.
    Pass an LLM client at call time so it can be swapped in tests.
    """

    def __init__(
        self,
        min_score_threshold: float = 0.30,
        max_chunks_for_synthesis: int = 8,
        dedup_similarity_threshold: float = 0.85,
    ) -> None:
        self.min_score_threshold = min_score_threshold
        self.max_chunks = max_chunks_for_synthesis
        self.dedup_threshold = dedup_similarity_threshold

    def prepare_chunks(
        self,
        raw_chunks: list["RetrievedChunk"],
    ) -> list["RetrievedChunk"]:
        """
        Filter, deduplicate, and trim chunks before synthesis.

        Returns up to self.max_chunks relevant, unique chunks sorted by score.
        """
        filtered = filter_by_score(raw_chunks, self.min_score_threshold)
        deduped = deduplicate_chunks(filtered, self.dedup_threshold)
        # Re-sort after dedup (order may change)
        deduped.sort(key=lambda c: c.score, reverse=True)
        return deduped[: self.max_chunks]

    async def synthesise(
        self,
        query: str,
        raw_chunks: list["RetrievedChunk"],
        llm_client,           # LLMClient instance with ainvoke_with_retry
    ) -> SynthesisResult:
        """
        Generate a grounded answer for *query* from *raw_chunks*.

        Args:
            query:       The user's original question.
            raw_chunks:  Unfiltered chunks from the vector search.
            llm_client:  An LLMClient instance (from Apexchat.core.llm).

        Returns:
            SynthesisResult with the answer, citations, and confidence.
        """
        # Post-process retrieved chunks
        chunks = self.prepare_chunks(raw_chunks)

        if not chunks:
            return SynthesisResult(
                answer=(
                    "I couldn't find relevant information in the uploaded documents "
                    "to answer your question. Please make sure you have uploaded the "
                    "relevant document(s) and try rephrasing your query."
                ),
                citations=[],
                confidence="none",
                num_chunks_used=0,
                query=query,
            )

        # Build prompt
        system_prompt, user_prompt = build_synthesis_prompt(query, chunks)

        # LLM call
        from langchain_core.messages import HumanMessage, SystemMessage  # noqa: PLC0415
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = await llm_client.ainvoke_with_retry(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        answer = strip_rag_citations(answer)

        citations = extract_citations(chunks)
        confidence = assess_confidence(chunks, answer)

        return SynthesisResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            num_chunks_used=len(chunks),
            query=query,
        )
