"""
General Tool - handles basic user interactions, greetings, and small talk.

This is the initial tool in the extensible tool registry. New tools follow
the same interface pattern defined by BaseTool.
"""

import time
from abc import ABC, abstractmethod

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from utils.apexchat.core.llm import LLMClient, get_general_tool_client
from utils.apexchat.schemas.models import ConversationMessage, MessageRole, WorkflowState
from utils.config import *
from utils.apexchat.core.status_stream import emit_status
logger = structlog.get_logger(__name__)


# ── Base Tool Interface ───────────────────────────────────────────────────────

class BaseTool(ABC):
    """
    Abstract base class for all workflow tools.

    To add a new tool:
    1. Create a new file in app/tools/
    2. Subclass BaseTool
    3. Implement execute() method
    4. Register in tool_registry (app/tools/__init__.py)
    5. Add ToolType enum entry in schemas/models.py
    6. Update orchestrator system prompt
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier matching ToolType enum."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for documentation."""
        ...

    @abstractmethod
    async def execute(self, state: WorkflowState) -> str:
        """
        Execute the tool and return a response string.

        Args:
            state: Current workflow state including conversation history

        Returns:
            Tool's response to the user
        """
        ...


# ── General Tool ──────────────────────────────────────────────────────────────

class GeneralTool(BaseTool):
    """
    Handles everyday conversational interactions:
    - Greetings ("hi", "hello", "hey")
    - Self-introductions ("my name is X")
    - Small talk ("how are you", "what's up")
    - General questions without a specialized tool

    Uses GPT-4o-mini for cost-efficient, high-quality responses.
    """

    SYSTEM_PROMPT = """your name is 'Lumi' and you are a friendly, helpful AI assistant. Your role is to handle 
everyday conversational interactions warmly and naturally.

Guidelines:
- Respond in a friendly, conversational tone
- Keep responses concise but complete (2-4 sentences typically)
- Remember context from earlier in the conversation
- If the user introduces themselves, use their name in subsequent responses
- Be genuine and personable without being sycophantic
- If asked about capabilities, explain you can handle general conversation 
  and that specialized tools are available for specific tasks"""

    def __init__(self, llm_client: LLMClient | None = None):
        self._llm_client = llm_client or get_general_tool_client()

    @property
    def name(self) -> str:
        return "general"

    @property
    def description(self) -> str:
        return (
            "Handles basic user interactions including greetings, "
            "introductions, small talk, and general conversation"
        )

    def _build_messages(self, state: WorkflowState) -> list:
        """
        Build the message list for the LLM, including conversation history.

        Args:
            state: Current workflow state

        Returns:
            List of LangChain message objects
        """
        messages = [SystemMessage(content=self.SYSTEM_PROMPT)]

        # Inject recent conversation history (bounded by config)
        
        recent_history = state.conversation_history[
            -MAX_CONVERSATION_HISTORY:
        ]

        for msg in recent_history:
            if msg.role == MessageRole.USER:
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == MessageRole.ASSISTANT:
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=msg.content))

        # Add current user message
        messages.append(HumanMessage(content=state.user_message))
        return messages

    async def execute(self, state: WorkflowState) -> str:
        """
        Generate a conversational response to the user's message.

        Args:
            state: Current workflow state

        Returns:
            Assistant's response string
        """
        emit_status("tool_general")
        start_time = time.perf_counter()
        messages = self._build_messages(state)

        logger.info(
            "GeneralTool executing",
            session_id=state.session_id,
            message_preview=state.user_message[:50],
            history_length=len(state.conversation_history),
        )

        response = await self._llm_client.ainvoke_with_retry(messages)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "GeneralTool completed",
            session_id=state.session_id,
            elapsed_ms=round(elapsed_ms, 2),
        )

        return response.content
