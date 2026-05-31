"""
Tool registry - central registry for all available workflow tools.

To register a new tool:
1. Import it here
2. Add an instance to the TOOL_REGISTRY dict using its ToolType key
"""

from utils.apexchat.schemas.models import ToolType
from utils.apexchat.tools.general import BaseTool, GeneralTool
from utils.apexchat.tools.navigation import NavigationTool
from utils.apexchat.tools.web_search import WebSearchTool
from utils.apexchat.tools.rag import RAGTool
from utils.apexchat.tools.memory_recall import MemoryRecallTool
from utils.apexchat.tools.booking import BookingTool
from utils.apexchat.tools.rescheduling import ReschedulingTool
from utils.apexchat.tools.data_viz.tool import DataVizTool

# Registry maps ToolType -> tool instance
# This is the single place to register new tools
TOOL_REGISTRY: dict[ToolType, BaseTool] = {
    ToolType.GENERAL: GeneralTool(),
    ToolType.NAVIGATION: NavigationTool(),
    ToolType.WEB_SEARCH: WebSearchTool(),
    ToolType.RAG: RAGTool(),
    ToolType.MEMORY: MemoryRecallTool(),
    ToolType.BOOKING: BookingTool(),
    ToolType.RESCHEDULING: ReschedulingTool(),
    ToolType.DATA_VIZ: DataVizTool(),
}


def get_tool(tool_type: ToolType) -> BaseTool:
    """
    Retrieve a tool from the registry by type.

    Args:
        tool_type: The ToolType enum value

    Returns:
        The corresponding tool instance

    Raises:
        KeyError: If the tool type is not registered
    """
    if tool_type not in TOOL_REGISTRY:
        raise KeyError(
            f"Tool '{tool_type}' is not registered. "
            f"Available tools: {list(TOOL_REGISTRY.keys())}"
        )
    return TOOL_REGISTRY[tool_type]


__all__ = ["TOOL_REGISTRY", "get_tool", "BaseTool"]
