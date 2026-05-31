"""Environment configuration loader for ApexAI Engines"""
import os
from pathlib import Path
from typing_extensions import Literal
from dotenv import load_dotenv

# Load .env file  root directory
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Base Configuration
BASE_URL = os.getenv("BASE_URL")
BASE_HTTPS_URL = os.getenv("BASE_HTTPS_URL")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL")
OPENROUTER_URL = os.getenv("OPENROUTER_URL")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

# OpenRouter API Keys
OPEN_ROUTER_Apexchat_API_KEY = os.getenv("OPEN_ROUTER_Apexchat_API_KEY")

# APEXAI-ENGINES Configuration
APEX_ENGINES_HOST = os.getenv("APEX_ENGINES_HOST")

# MongoDB Extended Configuration
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
MONGO_DB_USER = os.getenv("MONGO_DB_USER")
MONGO_DB_PASSWORD = os.getenv("MONGO_DB_PASSWORD")
MONGO_DB_HOST = os.getenv("MONGO_DB_HOST")
MONGO_DB_PORT = os.getenv("MONGO_DB_PORT")
MONGO_ADB_USER = os.getenv("MONGO_ADB_USER")
MONGO_ADB_PASSWORD = os.getenv("MONGO_ADB_PASSWORD")
MONGO_DB_DRIVER = os.getenv("MONGO_DB_DRIVER")

# Service Ports
APEX_CHAT_PORT = int(os.getenv("APEX_CHAT_PORT")) if os.getenv("APEX_CHAT_PORT") is not None else None

# ── Apexchat Application ───────────────────────────────────────────────────────────
APP_NAME = os.getenv("APP_NAME")
APP_VERSION = os.getenv("APP_VERSION")
ENVIRONMENT = os.getenv("ENVIRONMENT")
DEBUG = os.getenv("DEBUG").lower() == "true"
API_PREFIX = os.getenv("API_PREFIX")

# ── OpenRouter LLM ────────────────────────────────────────────────────────
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL")
GENERAL_TOOL_MODEL = os.getenv("GENERAL_TOOL_MODEL")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS"))
LLM_REQUEST_TIMEOUT = int(os.getenv("LLM_REQUEST_TIMEOUT"))

# ── Computed properties ───────────────────────────────────────────────────
is_production: bool = (ENVIRONMENT == "production")

openrouter_headers: dict = {
    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER"),
    "X-Title": APP_NAME,
}


# ── Memory System ─────────────────────────────────────────────────────────
MEMORY_EMBEDDING_MODEL = os.getenv("MEMORY_EMBEDDING_MODEL")
MEMORY_EMBEDDING_DIMENSIONS = int(os.getenv("MEMORY_EMBEDDING_DIMENSIONS"))
MEMORY_SESSION_INACTIVE_TIMEOUT_MINUTES = int(os.getenv("MEMORY_SESSION_INACTIVE_TIMEOUT_MINUTES"))
MEMORY_MAX_FACTS_PER_RETRIEVAL = int(os.getenv("MEMORY_MAX_FACTS_PER_RETRIEVAL"))
MEMORY_MAX_MESSAGES_PER_RETRIEVAL = int(os.getenv("MEMORY_MAX_MESSAGES_PER_RETRIEVAL"))
MEMORY_FACT_CONFIDENCE_THRESHOLD = float(os.getenv("MEMORY_FACT_CONFIDENCE_THRESHOLD"))
MEMORY_SEMANTIC_SIMILARITY_THRESHOLD = float(os.getenv("MEMORY_SEMANTIC_SIMILARITY_THRESHOLD"))

# ── Retry Configuration ───────────────────────────────────────────────────
MAX_RETRIES = int(os.getenv("MAX_RETRIES"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY"))
RETRY_BACKOFF_MULTIPLIER = float(os.getenv("RETRY_BACKOFF_MULTIPLIER"))

# ── Web Search Tool ───────────────────────────────────────────────────────
WEB_SEARCH_TOOL_MODEL = os.getenv("WEB_SEARCH_TOOL_MODEL")
WEB_SEARCH_TIMEOUT_SECONDS = int(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS"))
WEB_SEARCH_RATE_LIMIT_SECONDS = float(os.getenv("WEB_SEARCH_RATE_LIMIT_SECONDS"))

# ── report_generation Tool ─────────────────────────────────────────────────
REPORT_GENERATION_API_URL = os.getenv("NMS_GENERATE_SQL_URL")
REPORT_GENERATION_TOOL_MODEL = os.getenv("REPORT_GENERATION_TOOL_MODEL")

# ── RAG Tool (local rag_system engine — no external API) ──────────────────
RAG_TOOL_MODEL = os.getenv("RAG_TOOL_MODEL")
RAG_STORAGE_PATH = os.getenv("RAG_STORAGE_PATH")
RAG_EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE"))
RAG_MAX_CHUNKS = int(os.getenv("RAG_MAX_CHUNKS"))
MEMORY_TOOL_MODEL = os.getenv("MEMORY_TOOL_MODEL")


# ── Navigation Tool ───────────────────────────────────────────────────────
NAVIGATION_TOOL_MODEL = os.getenv("NAVIGATION_TOOL_MODEL")
SCREEN_CACHE_TTL_SECONDS = int(os.getenv("SCREEN_CACHE_TTL_SECONDS"))

# ── Logging ───────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL")
LOG_FORMAT = os.getenv("LOG_FORMAT")

# ── Conversation State ────────────────────────────────────────────────────
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS"))

# ── MongoDB ───────────────────────────────────────────────────────────────
MONGODB_URI = os.getenv("MONGO_DB_URI")
MONGODB_DB_NAME = os.getenv("MONGO_DB_NAME")
USE_DEMO_DATA =False

mongo_dsn = MONGODB_URI 

# ── Security ──────────────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS").split(",") if os.getenv("CORS_ORIGINS") else []
API_KEY_HEADER = os.getenv("API_KEY_HEADER")
API_KEYS = os.getenv("API_KEYS").split(",") if os.getenv("API_KEYS") else []
