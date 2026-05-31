"""
In-memory session store for conversation state.

Maintains conversation history across multiple requests within a session.
In production, replace with Redis for horizontal scalability:

    pip install redis
    # Update SessionStore.get/set to use aioredis
"""

import asyncio
import time
from collections import OrderedDict

import structlog

from utils.config import *
from utils.apexchat.schemas.models import WorkflowState

logger = structlog.get_logger(__name__)


class SessionStore:
    """
    Thread-safe in-memory session store with TTL eviction.

    For horizontal scaling, swap the internal dict for a Redis client.
    The interface (get/set/delete) remains identical.

    Note: This implementation is suitable for single-instance deployments.
    For multi-instance deployments, use Redis or another distributed cache.
    """

    def __init__(
        self,
        max_sessions: int = 10_000,
        ttl_seconds: int = None,
    ):
        self._store: OrderedDict[str, tuple[WorkflowState, float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._max_sessions = max_sessions
        self._ttl = ttl_seconds or SESSION_TTL_SECONDS

    async def get(self, session_id: str) -> WorkflowState | None:
        """
        Retrieve session state by ID.

        Args:
            session_id: Unique session identifier

        Returns:
            WorkflowState if found and not expired, None otherwise
        """
        async with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return None

            state, created_at = entry
            if time.time() - created_at > self._ttl:
                del self._store[session_id]
                logger.debug("Session expired", session_id=session_id)
                return None

            # Move to end (LRU order)
            self._store.move_to_end(session_id)
            return state

    async def set(self, session_id: str, state: WorkflowState) -> None:
        """
        Store or update session state.

        Args:
            session_id: Unique session identifier
            state: WorkflowState to persist
        """
        async with self._lock:
            # Evict oldest entries if at capacity
            while len(self._store) >= self._max_sessions:
                oldest_key, _ = self._store.popitem(last=False)
                logger.warning(
                    "Session evicted (capacity)",
                    evicted_session_id=oldest_key,
                )

            self._store[session_id] = (state, time.time())
            self._store.move_to_end(session_id)

    async def delete(self, session_id: str) -> bool:
        """
        Delete a session.

        Returns:
            True if session existed and was deleted, False otherwise
        """
        async with self._lock:
            if session_id in self._store:
                del self._store[session_id]
                return True
            return False

    async def cleanup_expired(self) -> int:
        """
        Remove all expired sessions.

        Returns:
            Number of sessions removed
        """
        now = time.time()
        async with self._lock:
            expired = [
                sid
                for sid, (_, created_at) in self._store.items()
                if now - created_at > self._ttl
            ]
            for sid in expired:
                del self._store[sid]

        if expired:
            logger.info("Cleaned up expired sessions", count=len(expired))
        return len(expired)

    @property
    def session_count(self) -> int:
        return len(self._store)


# Module-level singleton
_session_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    """Get the global session store instance (dependency injection)."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
        logger.info("Session store initialized")
    return _session_store
