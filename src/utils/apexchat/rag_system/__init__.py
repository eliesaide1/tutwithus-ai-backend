"""
Apexchat.rag_system — production-grade local RAG engine.

Public API
----------
DocumentStore   — manages per-user FAISS indices + PDF ingestion
AnswerSynthesiser — post-processes retrieved chunks, generates grounded answers
get_document_store() — returns the global DocumentStore singleton

Quick start
-----------
    from Apexchat.rag_system import get_document_store, AnswerSynthesiser

    store = await get_document_store()
    result = await store.ingest(file_base64, filename, user_id, embedder)
    chunks = await store.query(query, user_id, embedder)

    synthesiser = AnswerSynthesiser()
    synthesis = await synthesiser.synthesise(query, chunks, llm_client)
    print(synthesis.answer)
"""

from utils.apexchat.rag_system.document_store import (
    DocumentStore,
    IngestResult,
    QueryResult,
    RetrievedChunk,
    get_document_store,
)
from utils.apexchat.rag_system.synthesiser import AnswerSynthesiser, SynthesisResult, Citation

__all__ = [
    "DocumentStore",
    "IngestResult",
    "QueryResult",
    "RetrievedChunk",
    "get_document_store",
    "AnswerSynthesiser",
    "SynthesisResult",
    "Citation",
]
