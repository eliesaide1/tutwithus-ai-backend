"""
LLM client factory with retry logic, structured output support,
and OpenRouter integration via LangChain.
"""

import asyncio
from functools import lru_cache
from typing import Any, Type, TypeVar

import structlog
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from utils.config import *

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """
    Wrapper around LangChain ChatOpenAI configured for OpenRouter.

    Provides:
    - Retry logic with exponential backoff
    - Structured output generation
    - Request/response logging
    - Timeout management
    """

    def __init__(
        self,
        model: str,
        temperature: float = None,
        max_tokens: int = None,
    ):
        self.model = model
        self.temperature = temperature or LLM_TEMPERATURE
        self.max_tokens = max_tokens or LLM_MAX_TOKENS
        self._client = self._build_client()

    def _build_client(self) -> ChatOpenAI:
        """Build the underlying LangChain ChatOpenAI client."""
        return ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=OPEN_ROUTER_Apexchat_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
            default_headers=openrouter_headers,
            request_timeout=LLM_REQUEST_TIMEOUT,
        )

    def with_structured_output(self, schema: Type[T]) -> Any:
        """
        Return a runnable that produces structured output conforming to schema.

        Args:
            schema: Pydantic model class defining the expected output structure

        Returns:
            LangChain runnable with structured output
        """
        return self._client.with_structured_output(schema, method="json_schema")

    @property
    def base_client(self) -> ChatOpenAI:
        """Access the underlying LangChain client directly."""
        return self._client

    async def ainvoke_with_retry(self, messages: list, **kwargs) -> Any:
        """
        Invoke the LLM with automatic retry on transient failures.

        Args:
            messages: List of LangChain message objects
            **kwargs: Additional arguments passed to ainvoke

        Returns:
            LLM response

        Raises:
            Exception: If all retry attempts are exhausted
        """
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(
                multiplier=RETRY_BACKOFF_MULTIPLIER,
                min=RETRY_DELAY,
                max=RETRY_DELAY * 10,
            ),
            retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
            reraise=True,
        ):
            with attempt:
                try:
                    logger.debug(
                        "LLM request",
                        model=self.model,
                        attempt=attempt.retry_state.attempt_number,
                        message_count=len(messages),
                    )
                    response = await self._client.ainvoke(messages, **kwargs)
                  
                    return response
                except Exception as e:
                    logger.warning(
                        "LLM request failed",
                        model=self.model,
                        attempt=attempt.retry_state.attempt_number,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise


@lru_cache(maxsize=4)
def get_orchestrator_client() -> LLMClient:
    """
    Get cached orchestrator LLM client.

    The orchestrator uses lower temperature for more deterministic routing.
    Uses higher max_tokens to accommodate structured output generation.
    """
    return LLMClient(
        model=ORCHESTRATOR_MODEL,
        temperature=0.1,  # Deterministic routing decisions
        max_tokens=2048,  # Higher limit for structured JSON output
    )


@lru_cache(maxsize=4)
def get_general_tool_client() -> LLMClient:
    """Get cached general tool LLM client."""
    return LLMClient(
        model=GENERAL_TOOL_MODEL,
        temperature=LLM_TEMPERATURE,
    )


@lru_cache(maxsize=4)
def get_dashboard_tool_client() -> LLMClient:
    """
    Get cached dashboard tool LLM client.

    Uses lower temperature than general for deterministic SQL / JSON generation.
    """
    return LLMClient(
        model=GENERAL_TOOL_MODEL,
        temperature=0.2,
    )
    

@lru_cache(maxsize=4)
def get_navigation_tool_client() -> LLMClient:
    """
    Get cached navigation tool LLM client.

    Uses low temperature (0.1) — screen name extraction is a classification
    task, not a creative one. Determinism matters more than variety here.
    """
    return LLMClient(
        model=NAVIGATION_TOOL_MODEL,
        temperature=0.1,
    )


@lru_cache(maxsize=4)
def get_web_search_tool_client() -> LLMClient:
    """
    Get cached web search tool LLM client.
 
    Uses moderate temperature (0.3) — synthesis needs some creativity
    to produce coherent answers, but not too much to hallucinate.
    """
    return LLMClient(
        model=WEB_SEARCH_TOOL_MODEL,
        temperature=0.3,
    )


@lru_cache(maxsize=4)
def get_rag_tool_client() -> LLMClient:
    """
    Get cached RAG tool LLM client.

    Uses low temperature (0.1) — intent classification (upload vs search)
    is a deterministic task.
    """
    return LLMClient(
        model=RAG_TOOL_MODEL,
        temperature=0.1,
    )
    
@lru_cache(maxsize=4)
def get_memory_tool_client() -> LLMClient:
    """
    Get cached memory tool LLM client.

    Uses low temperature (0.1) — memory retrieval is a deterministic task.
    """
    return LLMClient(
        model=MEMORY_TOOL_MODEL,
        temperature=0.1,
    )
    


