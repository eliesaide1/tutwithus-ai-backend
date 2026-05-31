"""
Embedding Provider — async-compatible wrapper for generating text embeddings.

Uses asyncio.to_thread to push the blocking OpenAI SDK call off the event loop,
keeping the FastAPI / LangGraph async runtime unblocked.

Design notes:
- Configuration comes exclusively from Apexchat.core.config.settings.
- Logging uses structlog, consistent with the rest of the codebase.
- get_embedder() returns a cached singleton so weights are loaded once.
- The public interface mirrors SentenceTransformer.encode() for compatibility
  with existing MemoryManager.embed_text() call sites.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Union

import numpy as np
import structlog
from langchain_openai import OpenAIEmbeddings

from utils.config import *

logger = structlog.get_logger(__name__)


class TextEmbedder:
    """
    Async-compatible embedding wrapper backed by OpenRouter / OpenAI.

    Args:
        target_dimensions: Output vector size. Must match the embedding
            dimension in the database (default: settings.MEMORY_EMBEDDING_DIMENSIONS).
    """

    def __init__(self, target_dimensions: int | None = None) -> None:
        self.embedding_dim: int = target_dimensions or MEMORY_EMBEDDING_DIMENSIONS
        self._provider: OpenAIEmbeddings | None = self._initialize_provider()

    # ── Provider initialisation ───────────────────────────────────────────────

    def _initialize_provider(self) -> OpenAIEmbeddings | None:
        """
        Build the best available embedding provider.

        Priority:
          1. OpenRouter (project settings — no extra secrets needed)
          2. Direct OpenAI (reads OPENAI_API_KEY from environment as fallback)
        """
        if OPEN_ROUTER_Apexchat_API_KEY:
            try:
                provider = OpenAIEmbeddings(
                    model=MEMORY_EMBEDDING_MODEL,
                    dimensions=self.embedding_dim,
                    openai_api_key=OPEN_ROUTER_Apexchat_API_KEY,
                    openai_api_base=OPENROUTER_BASE_URL,
                )
                logger.info(
                    "Embedding provider initialised",
                    provider="openrouter",
                    model=MEMORY_EMBEDDING_MODEL,
                    dimensions=self.embedding_dim,
                )
                return provider
            except Exception as exc:
                logger.warning(
                    "OpenRouter embedding init failed — falling back to direct OpenAI",
                    error=str(exc),
                )

        try:
            provider = OpenAIEmbeddings(
                model=MEMORY_EMBEDDING_MODEL,
                dimensions=self.embedding_dim,
            )
            logger.info(
                "Embedding provider initialised",
                provider="openai_direct",
                model=MEMORY_EMBEDDING_MODEL,
                dimensions=self.embedding_dim,
            )
            return provider
        except Exception as exc:
            logger.error(
                "All embedding providers failed — semantic search disabled",
                error=str(exc),
            )

        return None

    # ── Sync encode (used internally and in blocking contexts) ────────────────

    def encode(
        self,
        texts: Union[str, list[str]],
        normalize_embeddings: bool = True,
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """
        Generate embedding(s) synchronously.

        Prefer aencode() inside async call paths to avoid blocking the event loop.
        """
        is_single = isinstance(texts, str)
        text_list: list[str] = [texts] if is_single else texts

        if not self._provider:
            logger.error("Embedding provider offline — returning zero vectors")
            zero = np.zeros(self.embedding_dim, dtype=np.float32)
            return zero if is_single else [zero for _ in text_list]

        try:
            raw: list[list[float]] = self._provider.embed_documents(text_list)
            result: list[np.ndarray] = []
            for r in raw:
                vec = np.array(r, dtype=np.float32)
                if normalize_embeddings:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                result.append(vec)
            return result[0] if is_single else result

        except Exception as exc:
            logger.error(
                "Embedding generation failed — returning zero vectors",
                error=str(exc),
                exc_info=True,
            )
            zero = np.zeros(self.embedding_dim, dtype=np.float32)
            return zero if is_single else [zero for _ in text_list]

    # ── Async encode (preferred inside async call paths) ──────────────────────

    async def aencode(
        self,
        texts: Union[str, list[str]],
        normalize_embeddings: bool = True,
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """
        Generate embedding(s) asynchronously.

        Pushes the blocking SDK call to a thread-pool worker so the event loop
        stays unblocked — matches the pattern used by httpx.AsyncClient and
        asyncio.gather throughout the dashboard tool.
        """
        return await asyncio.to_thread(self.encode, texts, normalize_embeddings)

    @property
    def is_available(self) -> bool:
        """True when a live embedding provider is configured."""
        return self._provider is not None


# ── Module-level singleton ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedder:
    """
    Return the cached global TextEmbedder instance.

    Use this instead of constructing new instances — embedding model
    loading is expensive and should happen once at startup.
    """
    return TextEmbedder()


# Backward-compatibility alias
SentenceTransformer = TextEmbedder
