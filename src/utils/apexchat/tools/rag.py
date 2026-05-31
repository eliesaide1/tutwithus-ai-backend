# """
# RAG Tool — Retrieval-Augmented Generation using the local rag_system engine.

# Responsibilities
# ----------------
# 1. Classify intent: document upload (PDF ingestion) vs. document search (Q&A).
# 2. For uploads: decode PDF → ingest into the local FAISS-backed DocumentStore.
# 3. For searches: embed query → retrieve top-k chunks → synthesise a grounded answer.
# 4. Persist results in state.rag_results for the API layer.

# Architecture change
# -------------------
# Previously this tool POSTed to an external RAG API
# (https://172.31.13.135:8050/rag-api/...).  It now uses the self-contained
# Apexchat.rag_system package:

#   External API (removed)              Local engine (new)
#   ─────────────────────────           ────────────────────────────────────
#   POST /rag-api/ingest       →        DocumentStore.ingest()
#   POST /rag-api/query        →        DocumentStore.query()
#                                       + AnswerSynthesiser.synthesise()

# Benefits
# --------
# * Zero network latency — all ops happen in-process.
# * No dependency on a running external service.
# * FAISS index is persisted to disk per user so documents survive server restarts.
# * Full control over chunking, embedding, and answer generation quality.
# * Easier to test, monitor, and debug.
# """

# from __future__ import annotations

# import re
# import time
# from typing import Any

# import structlog
# from langchain_core.messages import HumanMessage, SystemMessage
# from pydantic import BaseModel, Field

# # config vars imported via: from utils.config import *
# from utils.Apexchat.core.llm import LLMClient, get_rag_tool_client
# from utils.Apexchat.core.memory.embedding_wrapper import get_embedder
# from utils.Apexchat.rag_system import AnswerSynthesiser, get_document_store
# from utils.Apexchat.schemas.models import WorkflowState
# from utils.Apexchat.tools.general import BaseTool

# from utils.config import *
# logger = structlog.get_logger(__name__)


# # ── Structured Output Schema ─────────────────────────────────────────────────

# class RAGIntentSchema(BaseModel):
#     """Structured output: classify RAG intent and extract metadata."""

#     intent: str = Field(
#         description=(
#             "The user's intent. One of: 'upload' (user wants to upload/ingest "
#             "a PDF document) or 'search' (user wants to ask a question about "
#             "previously uploaded documents)."
#         ),
#     )
#     filename: str = Field(
#         default="uploaded_document.pdf",
#         description=(
#             "The filename of the document being uploaded. "
#             "Extract from the user message if mentioned, otherwise use default."
#         ),
#     )
#     search_query: str = Field(
#         default="",
#         description=(
#             "The refined search query for document Q&A. "
#             "Only populated when intent is 'search'."
#         ),
#     )


# # ── Prompts ───────────────────────────────────────────────────────────────────

# _INTENT_PROMPT = """\
# Analyze the user's message and determine their RAG intent.

# User message: "{query}"

# Context:
# - Has file data attached: {has_file}
# - Previously uploaded documents in session: {has_docs}

# Rules:
# - If the user has file data attached OR explicitly mentions uploading/ingesting a document → intent = "upload"
# - If the user asks a question about documents, wants information from docs, or references content → intent = "search"
# - Extract the filename if the user mentions one (e.g., "upload report.pdf" → filename = "report.pdf")
# - For search intent, refine the query to be clear and specific for document retrieval

# Classify the intent now.\
# """


# # ── RAGTool ───────────────────────────────────────────────────────────────────

# class RAGTool(BaseTool):
#     """
#     Handles document upload (PDF ingestion) and document Q&A via local RAG.

#     Flow:
#         execute()
#             → _classify_intent()         LLM: upload vs search
#             → _handle_upload()           Ingest PDF into local DocumentStore
#             OR
#             → _handle_search()           Query DocumentStore → synthesise answer
#             → persist results in state.rag_results
#     """

#     def __init__(self, llm_client: LLMClient | None = None) -> None:
#         self._llm = llm_client or get_rag_tool_client()
#         self._embedder = get_embedder()
#         self._synthesiser = AnswerSynthesiser(
#             min_score_threshold=RAG_MIN_SCORE,
#             max_chunks_for_synthesis=RAG_MAX_CHUNKS,
#             dedup_similarity_threshold=0.85,
#         )

#     # ── BaseTool interface ─────────────────────────────────────────────────────

#     @property
#     def name(self) -> str:
#         return "rag"

#     @property
#     def description(self) -> str:
#         return (
#             "Handles document upload (PDF ingestion) and retrieval-augmented "
#             "generation Q&A over previously uploaded documents. "
#             "Uses a local vector store — no external API required."
#         )

#     async def execute(self, state: WorkflowState) -> str:
#         start_time = time.perf_counter()
#         query = state.user_message.strip()

#         logger.info(
#             "RAGTool executing",
#             session_id=state.session_id,
#             message_preview=query[:80],
#         )

#         # Get or create per-session state
#         session = self._get_session(state)

#         # Resolve file data (if any)
#         file_base64 = self._resolve_file_base64(state)
#         has_file = bool(file_base64)
#         has_docs = bool(session.get("uploaded_docs"))

#         # ── Step 1: Classify intent ────────────────────────────────────────────
#         intent_result = await self._classify_intent(query, has_file, has_docs)
#         intent = intent_result.intent if intent_result else ("upload" if has_file else "search")

#         logger.info(
#             "RAG intent classified",
#             intent=intent,
#             has_file=has_file,
#             has_docs=has_docs,
#             session_id=state.session_id,
#         )

#         # ── Step 2: Execute based on intent ───────────────────────────────────
#         if intent == "upload":
#             result = await self._handle_upload(
#                 state, session, file_base64,
#                 intent_result.filename if intent_result else "uploaded_document.pdf",
#             )
#         else:
#             search_query = (
#                 intent_result.search_query if intent_result and intent_result.search_query
#                 else query
#             )
#             result = await self._handle_search(state, session, search_query)

#         # Persist session
#         self._save_session(state, session)

#         elapsed_ms = (time.perf_counter() - start_time) * 1000
#         logger.info(
#             "RAGTool completed",
#             session_id=state.session_id,
#             intent=intent,
#             elapsed_ms=round(elapsed_ms, 2),
#             success="error" not in (state.rag_results or {}),
#         )

#         return result

#     # ── Intent classification ──────────────────────────────────────────────────

#     async def _classify_intent(
#         self, query: str, has_file: bool, has_docs: bool,
#     ) -> RAGIntentSchema | None:
#         """Use structured LLM output to classify upload vs search intent."""
#         try:
#             structured_llm = self._llm.with_structured_output(RAGIntentSchema)
#             prompt = _INTENT_PROMPT.format(
#                 query=query,
#                 has_file=has_file,
#                 has_docs=has_docs,
#             )
#             return await structured_llm.ainvoke(prompt)
#         except Exception as e:
#             logger.error("RAG intent classification failed", error=str(e), exc_info=True)
#             return None

#     # ── Upload handler ─────────────────────────────────────────────────────────

#     async def _handle_upload(
#         self,
#         state: WorkflowState,
#         session: dict[str, Any],
#         file_base64: str | None,
#         filename: str,
#     ) -> str:
#         """Ingest a PDF document into the local FAISS-backed DocumentStore."""
#         if not file_base64:
#             err = (
#                 "I couldn't find any PDF data to upload. "
#                 "Please attach the PDF file with your request."
#             )
#             state.rag_results = {"error": err}
#             return err

#         user_id = state.session_id

#         try:
#             store = await get_document_store()
#             ingest_result = await store.ingest(
#                 file_base64=file_base64,
#                 filename=filename,
#                 user_id=user_id,
#                 embedder=self._embedder,
#             )
#         except ValueError as e:
#             err = str(e)
#             logger.warning("RAG ingest validation error", error=err, session_id=state.session_id)
#             state.rag_results = {"error": err}
#             return (
#                 f"Failed to upload the document.\n\n"
#                 f"**Reason:** {err}\n\n"
#                 "Please check the file and try again."
#             )
#         except Exception as e:
#             err = f"Unexpected error during document ingestion: {e}"
#             logger.error("RAG ingest unexpected error", error=str(e), exc_info=True)
#             state.rag_results = {"error": err}
#             return (
#                 f"Failed to upload the document due to an unexpected error.\n\n"
#                 f"**Details:** {e}\n\n"
#                 "Please try again or contact support if the issue persists."
#             )

#         # Track uploaded doc in session
#         uploaded_docs = session.setdefault("uploaded_docs", [])
#         uploaded_docs.append({
#             "filename": ingest_result.filename,
#             "doc_id": ingest_result.doc_id,
#             "num_pages": ingest_result.num_pages,
#             "num_chunks": ingest_result.num_chunks,
#             "user_id": user_id,
#         })

#         success_msg = (
#             f"✅ **'{ingest_result.filename}'** uploaded and indexed successfully.\n\n"
#             f"- **Pages processed:** {ingest_result.num_pages}\n"
#             f"- **Text chunks indexed:** {ingest_result.num_chunks}\n"
#             f"- **Indexing time:** {ingest_result.elapsed_ms:.0f} ms\n\n"
#             "You can now ask questions about this document."
#         )

#         state.rag_results = {
#             "answer": success_msg,
#             "doc_id": ingest_result.doc_id,
#             "filename": ingest_result.filename,
#             "num_pages": ingest_result.num_pages,
#             "num_chunks": ingest_result.num_chunks,
#         }
#         return success_msg

#     # ── Search handler ─────────────────────────────────────────────────────────

#     async def _handle_search(
#         self,
#         state: WorkflowState,
#         session: dict[str, Any],
#         query: str,
#     ) -> str:
#         """Query the local DocumentStore and synthesise a grounded answer."""
#         user_id = state.session_id

#         try:
#             store = await get_document_store()

#             has_docs = await store.has_documents(user_id)
#             if not has_docs:
#                 no_docs_msg = (
#                     "No documents have been uploaded yet for this session. "
#                     "Please upload a PDF document first, then ask your question."
#                 )
#                 state.rag_results = {"error": no_docs_msg}
#                 return no_docs_msg

#             top_k = RAG_TOP_K
#             raw_chunks = await store.query(
#                 query=query,
#                 user_id=user_id,
#                 embedder=self._embedder,
#                 top_k=top_k,
#             )

#         except Exception as e:
#             err = f"Error retrieving documents: {e}"
#             logger.error("RAG query failed", error=str(e), exc_info=True)
#             state.rag_results = {"error": err}
#             return (
#                 f"I couldn't retrieve an answer from the documents.\n\n"
#                 f"**Reason:** {err}\n\n"
#                 "Please try again."
#             )

#         # Synthesise answer using the LLM
#         try:
#             synthesis = await self._synthesiser.synthesise(
#                 query=query,
#                 raw_chunks=raw_chunks,
#                 llm_client=self._llm,
#             )
#         except Exception as e:
#             err = f"Error generating answer: {e}"
#             logger.error("RAG synthesis failed", error=str(e), exc_info=True)
#             state.rag_results = {"error": err}
#             return (
#                 f"I retrieved relevant excerpts but failed to generate an answer.\n\n"
#                 f"**Reason:** {err}"
#             )

#         # Build sources list for the API response
#         sources = [
#             {
#                 "filename": c.filename,
#                 "page": c.page + 1,
#                 "score": round(c.score, 4),
#                 "excerpt": c.excerpt,
#             }
#             for c in synthesis.citations
#         ]

#         state.rag_results = {
#             "answer": synthesis.answer,
#             "sources": sources,
#             "query": query,
#             "confidence": synthesis.confidence,
#             "num_chunks_used": synthesis.num_chunks_used,
#         }

#         answer = synthesis.answer
#         if sources and synthesis.confidence != "none":
#             source_lines = "\n".join(
#                 f"- **{s['filename']}**, page {s['page']} (relevance: {s['score']:.2f})"
#                 for s in sources
#             )
#             answer += f"\n\n---\n**Sources:**\n{source_lines}"

#         return answer

#     # ── File resolution ────────────────────────────────────────────────────────

#     @staticmethod
#     def _resolve_file_base64(state: WorkflowState) -> str | None:
#         """
#         Extract base64-encoded PDF data from the request.

#         Resolution order:
#         1. state.file_base64  (set by the API layer from the request body)
#         2. Inline base64 data embedded in the user message
#         """
#         if getattr(state, "file_base64", None):
#             return state.file_base64

#         match = re.search(
#             r"(data:application/pdf;base64,)?([A-Za-z0-9+/=\s]{100,})",
#             state.user_message,
#         )
#         if match:
#             return match.group(2).strip()

#         return None

#     # ── Session helpers ────────────────────────────────────────────────────────

#     @staticmethod
#     def _get_session(state: WorkflowState) -> dict[str, Any]:
#         sid = state.session_id
#         if sid not in state.rag_sessions:
#             state.rag_sessions[sid] = {"uploaded_docs": []}
#         return state.rag_sessions[sid]

#     @staticmethod
#     def _save_session(state: WorkflowState, session: dict[str, Any]) -> None:
#         state.rag_sessions[state.session_id] = session

"""
RAG Tool — Retrieval-Augmented Generation using the local rag_system engine.

Responsibilities
----------------
1. Classify intent: document upload (PDF ingestion) vs. document search (Q&A).
2. For uploads: decode PDF → ingest into the local FAISS-backed DocumentStore.
3. For searches: embed query → retrieve top-k chunks → synthesise a grounded answer.
4. Persist results in state.rag_results for the API layer.

Architecture change
-------------------
Previously this tool POSTed to an external RAG API
(https://172.31.13.135:8050/rag-api/...).  It now uses the self-contained
Apexchat.rag_system package:

  External API (removed)              Local engine (new)
  ─────────────────────────           ────────────────────────────────────
  POST /rag-api/ingest       →        DocumentStore.ingest()
  POST /rag-api/query        →        DocumentStore.query()
                                      + AnswerSynthesiser.synthesise()

Benefits
--------
* Zero network latency — all ops happen in-process.
* No dependency on a running external service.
* FAISS index is persisted to disk per user so documents survive server restarts.
* Full control over chunking, embedding, and answer generation quality.
* Easier to test, monitor, and debug.
"""

from __future__ import annotations

import re
import time
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# config vars imported via: from utils.config import *
from utils.apexchat.core.llm import LLMClient, get_rag_tool_client
from utils.apexchat.core.memory.embedding_wrapper import get_embedder
from utils.apexchat.core.mongodb import (
    fetch_levels,
    fetch_all_tutors_for_knowledge,
    fetch_all_subjects_for_knowledge,
)
from utils.apexchat.core.status_stream import emit_status
from utils.apexchat.rag_system import AnswerSynthesiser, get_document_store
from utils.apexchat.rag_system.document_store import RetrievedChunk
from utils.apexchat.schemas.models import WorkflowState
from utils.apexchat.tools.general import BaseTool

from utils.config import *
logger = structlog.get_logger(__name__)


# ── Structured Output Schema ─────────────────────────────────────────────────

class RAGIntentSchema(BaseModel):
    """Structured output: classify RAG intent and extract metadata."""

    intent: str = Field(

    )
    filename: str = Field(
        default="uploaded_document.pdf",
        description=(
            "The filename of the document being uploaded. "
            "Extract from the user message if mentioned, otherwise use default."
        ),
    )
    search_query: str = Field(
        default="",
        description=(
            "The refined search query for document Q&A. "
            "Populated when intent is 'search' or 'upload_and_search'."
        ),
    )


# ── Prompts ───────────────────────────────────────────────────────────────────

_INTENT_PROMPT = """\
Analyze the user's message and determine their RAG intent.

User message: "{query}"

Context:
- Has file data attached: {has_file}
- Previously uploaded documents in session: {has_docs}

Rules:
- If the user has file data attached AND also asks a question or requests information about the document → intent = "upload_and_search"
- If the user has file data attached but does NOT ask any question (only mentions uploading/ingesting) → intent = "upload"
- If the user asks a question about documents without attaching file data → intent = "search"
- Extract the filename if the user mentions one (e.g., "upload report.pdf" → filename = "report.pdf")
- For "upload_and_search" or "search" intent, extract and refine the question into a clear search_query for document retrieval

Examples of "upload_and_search":
- "Here is my document, what are the key findings?" (file attached)
- "Upload this PDF and summarize it" (file attached)
- "Analyze this document and tell me about the revenue" (file attached)
- "What does this document say about compliance?" (file attached)

Examples of "upload" only:
- "Upload this document" (file attached, no question)
- "Ingest this PDF" (file attached, no question)
- "Add this file to my documents" (file attached, no question)

Classify the intent now.\
"""


# ── RAGTool ───────────────────────────────────────────────────────────────────

class RAGTool(BaseTool):
    """
    Handles document upload (PDF ingestion) and document Q&A via local RAG.

    Flow:
        execute()
            → _classify_intent()         LLM: upload vs search
            → _handle_upload()           Ingest PDF into local DocumentStore
            OR
            → _handle_search()           Query DocumentStore → synthesise answer
            → persist results in state.rag_results
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or get_rag_tool_client()
        self._embedder = get_embedder()
        self._synthesiser = AnswerSynthesiser(
            min_score_threshold=RAG_MIN_SCORE,
            max_chunks_for_synthesis=RAG_MAX_CHUNKS,
            dedup_similarity_threshold=0.85,
        )

    # ── BaseTool interface ─────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "rag"

    @property
    def description(self) -> str:
        return (
            "Handles document upload (PDF ingestion) and retrieval-augmented "
            "generation Q&A over previously uploaded documents. "
            "Uses a local vector store — no external API required."
        )

    async def execute(self, state: WorkflowState) -> str:
        emit_status("tool_rag")
        start_time = time.perf_counter()
        query = state.user_message.strip()

        logger.info(
            "RAGTool executing",
            session_id=state.session_id,
            message_preview=query[:80],
        )

        # Get or create per-session state
        session = self._get_session(state)

        # Resolve file data (if any)
        file_base64 = self._resolve_file_base64(state)
        has_file = bool(file_base64)
        has_docs = bool(session.get("uploaded_docs"))

        # ── Step 1: Classify intent ────────────────────────────────────────────
        intent_result = await self._classify_intent(query, has_file, has_docs)
        intent = intent_result.intent if intent_result else (
            "upload_and_search" if has_file and len(query.split()) > 3
            else "upload" if has_file
            else "search"
        )

        logger.info(
            "RAG intent classified",
            intent=intent,
            has_file=has_file,
            has_docs=has_docs,
            session_id=state.session_id,
        )

        # ── Step 2: Execute based on intent ───────────────────────────────────
        if intent == "upload":
            result = await self._handle_upload(
                state, session, file_base64,
                intent_result.filename if intent_result else "uploaded_document.pdf",
            )
        elif intent == "upload_and_search":
            # Combined flow: ingest the document first, then answer the question
            upload_result = await self._handle_upload(
                state, session, file_base64,
                intent_result.filename if intent_result else "uploaded_document.pdf",
            )
            # Only proceed to search if upload succeeded (no error in rag_results)
            if state.rag_results and "error" not in state.rag_results:
                search_query = (
                    intent_result.search_query if intent_result and intent_result.search_query
                    else query
                )
                search_result = await self._handle_search(state, session, search_query)
                # Merge: keep upload metadata in rag_results but replace the answer
                # with the search answer so the user gets a single combined response
                if state.rag_results and "error" not in state.rag_results:
                    state.rag_results["upload_summary"] = upload_result
                result = search_result
            else:
                result = upload_result
        else:
            search_query = (
                intent_result.search_query if intent_result and intent_result.search_query
                else query
            )
            result = await self._handle_search(state, session, search_query)

        # Persist session
        self._save_session(state, session)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "RAGTool completed",
            session_id=state.session_id,
            intent=intent,
            elapsed_ms=round(elapsed_ms, 2),
            success="error" not in (state.rag_results or {}),
        )

        return result

    # ── Intent classification ──────────────────────────────────────────────────

    async def _classify_intent(
        self, query: str, has_file: bool, has_docs: bool,
    ) -> RAGIntentSchema | None:
        """Use structured LLM output to classify upload vs search intent."""
        try:
            structured_llm = self._llm.with_structured_output(RAGIntentSchema)
            prompt = _INTENT_PROMPT.format(
                query=query,
                has_file=has_file,
                has_docs=has_docs,
            )
            return await structured_llm.ainvoke(prompt)
        except Exception as e:
            logger.error("RAG intent classification failed", error=str(e), exc_info=True)
            return None

    # ── Upload handler ─────────────────────────────────────────────────────────

    async def _handle_upload(
        self,
        state: WorkflowState,
        session: dict[str, Any],
        file_base64: str | None,
        filename: str,
    ) -> str:
        """Ingest a PDF document into the local FAISS-backed DocumentStore."""
        if not file_base64:
            err = (
                "I couldn't find any PDF data to upload. "
                "Please attach the PDF file with your request."
            )
            state.rag_results = {"error": err}
            return err

        user_id = state.session_id

        try:
            store = await get_document_store()
            emit_status("tool_rag_ingesting")
            ingest_result = await store.ingest(
                file_base64=file_base64,
                filename=filename,
                user_id=user_id,
                embedder=self._embedder,
            )
        except ValueError as e:
            err = str(e)
            logger.warning("RAG ingest validation error", error=err, session_id=state.session_id)
            state.rag_results = {"error": err}
            return (
                f"Failed to upload the document.\n\n"
                f"**Reason:** {err}\n\n"
                "Please check the file and try again."
            )
        except Exception as e:
            err = f"Unexpected error during document ingestion: {e}"
            logger.error("RAG ingest unexpected error", error=str(e), exc_info=True)
            state.rag_results = {"error": err}
            return (
                f"Failed to upload the document due to an unexpected error.\n\n"
                f"**Details:** {e}\n\n"
                "Please try again or contact support if the issue persists."
            )

        # Track uploaded doc in session
        uploaded_docs = session.setdefault("uploaded_docs", [])
        uploaded_docs.append({
            "filename": ingest_result.filename,
            "doc_id": ingest_result.doc_id,
            "num_pages": ingest_result.num_pages,
            "num_chunks": ingest_result.num_chunks,
            "user_id": user_id,
        })

        success_msg = (
            f"✅ **'{ingest_result.filename}'** uploaded and indexed successfully.\n\n"
            f"- **Pages processed:** {ingest_result.num_pages}\n"
            f"- **Text chunks indexed:** {ingest_result.num_chunks}\n"
            f"- **Indexing time:** {ingest_result.elapsed_ms:.0f} ms\n\n"
            "You can now ask questions about this document."
        )

        state.rag_results = {
            "answer": success_msg,
            "doc_id": ingest_result.doc_id,
            "filename": ingest_result.filename,
            "num_pages": ingest_result.num_pages,
            "num_chunks": ingest_result.num_chunks,
        }
        return success_msg

    # ── Search handler ─────────────────────────────────────────────────────────

    async def _handle_search(
        self,
        state: WorkflowState,
        session: dict[str, Any],
        query: str,
    ) -> str:
        """
        Answer a question using:
          1. Live platform knowledge fetched from MongoDB (always included).
          2. User-uploaded document chunks (included when documents exist).

        Both source types are merged and sent to the synthesiser together.
        """
        emit_status("tool_rag")
        user_id = state.session_id

        # ── 1. Always fetch live platform knowledge from DB ────────────────────
        emit_status("tool_rag_retrieving")
        db_chunks = await self._build_db_knowledge_chunks()

        # ── 2. Also query uploaded documents when the user has any ────────────
        doc_chunks: list[RetrievedChunk] = []
        try:
            store = await get_document_store()
            if await store.has_documents(user_id):
                doc_chunks = await store.query(
                    query=query,
                    user_id=user_id,
                    embedder=self._embedder,
                    top_k=RAG_TOP_K,
                )
                logger.info(
                    "RAG: user doc chunks retrieved",
                    session_id=state.session_id,
                    count=len(doc_chunks),
                )
        except Exception as e:
            logger.warning(
                "RAG: document store query failed, continuing with DB knowledge only",
                error=str(e),
                session_id=state.session_id,
            )

        # ── 3. Combine and synthesise ─────────────────────────────────────────
        all_chunks = db_chunks + doc_chunks

        try:
            synthesis = await self._synthesiser.synthesise(
                query=query,
                raw_chunks=all_chunks,
                llm_client=self._llm,
            )
        except Exception as e:
            err = f"Error generating answer: {e}"
            logger.error("RAG synthesis failed", error=str(e), exc_info=True)
            state.rag_results = {"error": err}
            return (
                f"I retrieved relevant information but failed to generate an answer.\n\n"
                f"**Reason:** {err}"
            )

        # Build sources list — exclude platform_knowledge_base from user-facing citations
        sources = [
            {
                "filename": c.filename,
                "page": c.page + 1,
                "score": round(c.score, 4),
                "excerpt": c.excerpt,
            }
            for c in synthesis.citations
            if c.filename != "platform_knowledge_base"
        ]

        state.rag_results = {
            "answer": synthesis.answer,
            "sources": sources,
            "query": query,
            "confidence": synthesis.confidence,
            "num_chunks_used": synthesis.num_chunks_used,
            "used_db_knowledge": True,
            "used_uploaded_docs": bool(doc_chunks),
        }

        answer = synthesis.answer
        if sources and synthesis.confidence != "none":
            source_lines = "\n".join(
                f"- **{s['filename']}**, page {s['page']} (relevance: {s['score']:.2f})"
                for s in sources
            )
            answer += f"\n\n---\n**Sources:**\n{source_lines}"

        return answer

    # ── DB knowledge builder ───────────────────────────────────────────────────

    async def _build_db_knowledge_chunks(self) -> list[RetrievedChunk]:
        """
        Fetch live tutoring platform data from MongoDB and return as synthetic
        RetrievedChunk objects so the synthesiser can treat them as grounded context.

        Four chunks are produced:
          0 — static platform overview + booking process description
          1 — available grade levels (from DB)
          2 — available subjects (from DB)
          3 — available tutors with brief profiles (from DB)
        """
        chunks: list[RetrievedChunk] = []

        # ── Chunk 0: Static platform overview (always present) ─────────────────
        overview_text = (
            "TUTORING PLATFORM OVERVIEW\n"
            "===========================\n"
            "This is an online tutoring platform that connects students with qualified "
            "tutors for personalised one-on-one or small-group tutoring sessions.\n\n"
            "HOW TO BOOK A SESSION:\n"
            "1. Select your grade level (e.g. Elementary, Middle School, High School)\n"
            "2. Choose a subject (e.g. Math, Physics, English, Biology)\n"
            "3. Select a curriculum/program if required (American, British, French, Lebanese/National)\n"
            "4. Pick a tutor from the available list\n"
            "5. Choose an available date and time slot\n"
            "6. Optionally add a session focus/description\n"
            "7. Optionally invite up to 6 guests by email\n"
            "8. Confirm the booking\n\n"
            "CREDITS & WALLET:\n"
            "- Sessions are paid using credits purchased through bundles\n"
            "- Credits are stored in your wallet; you must have credits available to book\n"
            "- Purchase credits from the Bundles page (Bundles > Credits tab)\n\n"
            "RESCHEDULING:\n"
            "- Existing sessions can be rescheduled to a new date and time\n"
            "- Provide the session ID along with the new date and start time"
        )
        chunks.append(RetrievedChunk(
            text=overview_text,
            filename="platform_knowledge_base",
            doc_id="platform_db",
            page=0,
            chunk_idx=0,
            score=0.95,
        ))

        # ── Chunk 1: Levels ────────────────────────────────────────────────────
        try:
            levels = await fetch_levels()
            if levels:
                levels_text = "AVAILABLE GRADE LEVELS:\n" + "\n".join(
                    f"- {lv['name']}" for lv in levels
                )
                chunks.append(RetrievedChunk(
                    text=levels_text,
                    filename="platform_knowledge_base",
                    doc_id="platform_db",
                    page=0,
                    chunk_idx=1,
                    score=0.95,
                ))
        except Exception as e:
            logger.warning("RAG: failed to fetch levels for knowledge base", error=str(e))

        # ── Chunk 2: Subjects ──────────────────────────────────────────────────
        try:
            subjects = await fetch_all_subjects_for_knowledge()
            if subjects:
                subjects_text = "AVAILABLE SUBJECTS:\n" + "\n".join(
                    f"- {s['name']}" for s in subjects
                )
                chunks.append(RetrievedChunk(
                    text=subjects_text,
                    filename="platform_knowledge_base",
                    doc_id="platform_db",
                    page=0,
                    chunk_idx=2,
                    score=0.95,
                ))
        except Exception as e:
            logger.warning("RAG: failed to fetch subjects for knowledge base", error=str(e))

        # ── Chunk 3: Tutors ────────────────────────────────────────────────────
        try:
            tutors = await fetch_all_tutors_for_knowledge()
            if tutors:
                tutor_lines: list[str] = []
                for t in tutors:
                    name = f"{t.get('firstName', '')} {t.get('lastName', '')}".strip()
                    degree = t.get("degree") or ""
                    bio = (t.get("bio") or "")[:150]
                    curricula = ", ".join(t.get("curriculums") or [])
                    line = f"- {name}"
                    if degree:
                        line += f" | Degree: {degree}"
                    if curricula:
                        line += f" | Curricula: {curricula}"
                    if bio:
                        line += f" | {bio}"
                    tutor_lines.append(line)

                tutors_text = f"AVAILABLE TUTORS ({len(tutors)} total):\n" + "\n".join(tutor_lines)
                chunks.append(RetrievedChunk(
                    text=tutors_text,
                    filename="platform_knowledge_base",
                    doc_id="platform_db",
                    page=0,
                    chunk_idx=3,
                    score=0.95,
                ))
        except Exception as e:
            logger.warning("RAG: failed to fetch tutors for knowledge base", error=str(e))

        return chunks

    # ── File resolution ────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_file_base64(state: WorkflowState) -> str | None:
        """
        Extract base64-encoded PDF data from the request.

        Resolution order:
        1. state.file_base64  (set by the API layer from the request body)
        2. Inline base64 data embedded in the user message
        """
        if getattr(state, "file_base64", None):
            return state.file_base64

        match = re.search(
            r"(data:application/pdf;base64,)?([A-Za-z0-9+/=\s]{100,})",
            state.user_message,
        )
        if match:
            return match.group(2).strip()

        return None

    # ── Session helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _get_session(state: WorkflowState) -> dict[str, Any]:
        sid = state.session_id
        if sid not in state.rag_sessions:
            state.rag_sessions[sid] = {"uploaded_docs": []}
        return state.rag_sessions[sid]

    @staticmethod
    def _save_session(state: WorkflowState, session: dict[str, Any]) -> None:
        state.rag_sessions[state.session_id] = session