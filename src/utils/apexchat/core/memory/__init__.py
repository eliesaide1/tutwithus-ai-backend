"""
Apexchat.core.memory — Apexchat persistent memory subsystem.

Public API:

    from Apexchat.core.memory import (
        ApexchatMemorySystem,
        get_memory_system,
        init_memory_system,
        MemoryManager,
        FactProcessor,
        MemoryRetrieval,
        TextEmbedder,
        get_embedder,
    )
"""

from utils.apexchat.core.memory.embedding_wrapper import TextEmbedder, get_embedder
from utils.apexchat.core.memory.fact_processor import FactProcessor
from utils.apexchat.core.memory.memory_integration import (
    ApexchatMemorySystem,
    get_memory_system,
    init_memory_system,
)
from utils.apexchat.core.memory.memory_manager import MemoryManager
from utils.apexchat.core.memory.memory_retrieval import MemoryRetrieval

__all__ = [
    "ApexchatMemorySystem",
    "get_memory_system",
    "init_memory_system",
    "MemoryManager",
    "FactProcessor",
    "MemoryRetrieval",
    "TextEmbedder",
    "get_embedder",
]
