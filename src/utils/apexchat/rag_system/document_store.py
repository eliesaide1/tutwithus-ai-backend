"""
RAG Document Store — MongoDB vector store for PDF ingestion and retrieval.

Architecture
------------
* PDF parsing   : pypdf (pure-Python, zero native deps)
* Chunking      : Recursive character splitting with overlap, respects sentence boundaries
* Embeddings    : Reuses the project's existing TextEmbedder (OpenRouter / OpenAI)
* Storage       : MongoDB (embeddings stored as arrays, similarity computed in Python)
* Concurrency   : All heavy CPU/IO/DB work is pushed to asyncio.to_thread()

MongoDB collection: ``document_store``
Document shape:
{
    "doc_id": str,
    "chunk_index": int,
    "content": str,
    "metadata": { "filename": str, "page": int, "user_id": str, "char_start": int },
    "embedding": list[float]   # e.g. 384-dim vector
}
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog
from utils.Mongodb_tools import MONGODB_TOOLS
from utils.config import MONGODB_URI, RAG_EMBEDDING_DIM, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, RAG_TOP_K

logger = structlog.get_logger(__name__)
_mongodb_tools = MONGODB_TOOLS()

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _require_pypdf():
    try:
        from pypdf import PdfReader
        return PdfReader
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDF parsing. "
            "Install it with: pip install pypdf"
        ) from exc


def _cosine_similarity(a, b) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """Single text chunk with provenance metadata."""
    text: str
    doc_id: str
    filename: str
    page: int
    chunk_idx: int
    char_start: int
    user_id: str


@dataclass
class RetrievedChunk:
    """Chunk returned by a similarity search, augmented with score."""
    text: str
    filename: str
    doc_id: str
    page: int
    chunk_idx: int
    score: float


@dataclass
class IngestResult:
    """Summary returned after a successful ingest."""
    doc_id: str
    filename: str
    num_pages: int
    num_chunks: int
    elapsed_ms: float
    message: str


@dataclass
class QueryResult:
    """Summary returned after a similarity search."""
    answer: str
    sources: list[RetrievedChunk]
    query: str
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

class TextChunker:
    """
    Recursive character splitter with configurable chunk size and overlap.
    """

    _SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[tuple[str, int]]:
        if not text.strip():
            return []
        chunks: list[tuple[str, int]] = []
        self._split_recursive(text, 0, chunks)
        return chunks

    def _split_recursive(self, text: str, base_offset: int, out: list[tuple[str, int]]) -> None:
        if len(text) <= self.chunk_size:
            stripped = text.strip()
            if stripped:
                out.append((stripped, base_offset))
            return

        for sep in self._SEPARATORS:
            if sep and sep in text:
                parts = text.split(sep)
                current = ""
                current_offset = base_offset
                for i, part in enumerate(parts):
                    candidate = current + (sep if current else "") + part
                    if len(candidate) > self.chunk_size and current:
                        stripped = current.strip()
                        if stripped:
                            out.append((stripped, current_offset))
                        overlap_start = max(0, len(current) - self.chunk_overlap)
                        current = current[overlap_start:] + (sep if current else "") + part
                        current_offset = base_offset + text.index(part) if part in text else current_offset
                    else:
                        current = candidate
                if current.strip():
                    out.append((current.strip(), current_offset))
                return

        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                out.append((chunk, base_offset + start))
            start = end - self.chunk_overlap


# ---------------------------------------------------------------------------
# MongoDB Document Store
# ---------------------------------------------------------------------------

class DocumentStore:
    """
    MongoDB vector store for RAG.
    Embeddings are stored as arrays; similarity is computed in Python.
    """

    def __init__(
        self,
        db_dsn: str,
        embedding_dim: int = 384,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 5,
    ) -> None:
        self._embedding_dim = embedding_dim
        self._chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._top_k = top_k
        logger.info("DocumentStore (MongoDB) initialised", embedding_dim=embedding_dim)

    @staticmethod
    def _parse_pdf_sync(pdf_bytes: bytes) -> list[tuple[str, int]]:
        PdfReader = _require_pypdf()
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            pages = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((text, page_num))
            return pages
        except Exception as exc:
            logger.error("PDF parsing failed", error=str(exc))
            return []

    async def ingest(
        self,
        file_base64: str,
        filename: str,
        user_id: str,
        embedder,
    ) -> IngestResult:
        t0 = time.perf_counter()
        doc_id = str(uuid.uuid4())

        # 1. Decode
        try:
            b64_clean = file_base64.split(",")[-1].strip()
            pdf_bytes = base64.b64decode(b64_clean)
        except Exception as exc:
            raise ValueError(f"Invalid base64 PDF data: {exc}") from exc

        # 2. Parse PDF
        pages = await asyncio.to_thread(self._parse_pdf_sync, pdf_bytes)
        if not pages:
            raise ValueError("Could not extract any text from the PDF.")

        # 3. Chunk
        chunks: list[Chunk] = []
        chunk_idx = 0
        for page_text, page_num in pages:
            for chunk_text, char_start in self._chunker.split(page_text):
                chunks.append(Chunk(
                    text=chunk_text, doc_id=doc_id, filename=filename,
                    page=page_num, chunk_idx=chunk_idx, char_start=char_start, user_id=user_id
                ))
                chunk_idx += 1

        if not chunks:
            raise ValueError("PDF parsed successfully but produced no text chunks.")

        # 4. Embed
        texts = [c.text for c in chunks]
        embeddings = await embedder.aencode(texts, normalize_embeddings=True)
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()
        embeddings = [e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings]

        # 5. Build documents for MongoDB
        mongo_docs = []
        for i, chunk in enumerate(chunks):
            mongo_docs.append({
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_idx,
                "content": chunk.text,
                "metadata": {
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "user_id": chunk.user_id,
                    "char_start": chunk.char_start,
                },
                "embedding": embeddings[i],
            })

        # 6. Batch insert
        def _insert_sync():
            db = _mongodb_tools.get_db_connection()
            db.document_store.insert_many(mongo_docs)

        await asyncio.to_thread(_insert_sync)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Document ingested to MongoDB", user_id=user_id, doc_id=doc_id, num_chunks=len(chunks))

        return IngestResult(
            doc_id=doc_id, filename=filename, num_pages=len(pages),
            num_chunks=len(chunks), elapsed_ms=round(elapsed_ms, 2),
            message=f"Successfully ingested '{filename}' into MongoDB."
        )

    async def query(
        self,
        query: str,
        user_id: str,
        embedder,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        t0 = time.perf_counter()
        k = top_k or self._top_k

        # 1. Embed query
        query_vec = await embedder.aencode(query, normalize_embeddings=True)
        if isinstance(query_vec, np.ndarray):
            query_vec = query_vec.flatten().tolist()

        # 2. Fetch all chunks for this student, compute similarity in Python
        def _query_sync():
            db = _mongodb_tools.get_db_connection()
            docs = list(db.document_store.find(
                {"metadata.user_id": user_id},
            ))

            scored = []
            for doc in docs:
                emb = doc.get("embedding")
                if emb is None:
                    continue
                sim = _cosine_similarity(query_vec, emb)
                scored.append((sim, doc))

            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[:k]

        top_results = await asyncio.to_thread(_query_sync)

        # 3. Map results
        results = []
        for score, doc in top_results:
            metadata = doc.get("metadata", {})
            results.append(RetrievedChunk(
                text=doc["content"],
                filename=metadata.get("filename", "unknown"),
                doc_id=doc["doc_id"],
                page=metadata.get("page", 0),
                chunk_idx=doc["chunk_index"],
                score=float(score)
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("MongoDB Query completed", user_id=user_id, results_found=len(results), elapsed_ms=elapsed_ms)
        return results

    async def has_documents(self, user_id: str) -> bool:
        def _check_sync():
            db = _mongodb_tools.get_db_connection()
            return db.document_store.find_one({"metadata.user_id": user_id}) is not None
        return await asyncio.to_thread(_check_sync)

    async def get_manifest(self, user_id: str) -> list[dict[str, Any]]:
        def _run():
            db = _mongodb_tools.get_db_connection()
            pipeline = [
                {"$match": {"metadata.user_id": user_id}},
                {"$group": {
                    "_id": {"doc_id": "$doc_id", "filename": "$metadata.filename"},
                    "num_chunks": {"$sum": 1},
                }},
            ]
            return list(db.document_store.aggregate(pipeline))

        rows = await asyncio.to_thread(_run)
        return [
            {
                "doc_id": r["_id"]["doc_id"],
                "filename": r["_id"]["filename"],
                "num_chunks": r["num_chunks"],
            }
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Module-level singleton (lazy-initialised)
# ---------------------------------------------------------------------------

_store: DocumentStore | None = None
_store_lock = asyncio.Lock()

async def get_document_store() -> DocumentStore:
    """Return the global DocumentStore singleton (MongoDB backed)."""
    global _store
    if _store is not None:
        return _store

    async with _store_lock:
        if _store is not None:
            return _store

        db_dsn = MONGODB_URI
        embedding_dim = RAG_EMBEDDING_DIM
        chunk_size = RAG_CHUNK_SIZE
        chunk_overlap = RAG_CHUNK_OVERLAP
        top_k = RAG_TOP_K

        _store = DocumentStore(
            db_dsn=db_dsn,
            embedding_dim=embedding_dim,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
        )

    return _store
