"""
Structured logging configuration using structlog.
Outputs JSON in production, colored console output in development.

The ``stream_log_processor`` is injected just before the final renderer so
that every log call also fans out to any per-request SSE queue that is
currently attached (see Apexchat/core/log_stream.py).
"""

import logging
import sys

import structlog

from utils.config import *


def setup_logging() -> None:
    """
    Configure structlog for the application.

    Uses JSON output for production (machine-readable) and
    pretty console output for development (human-readable).

    The SSE streaming processor is always registered so that streaming
    requests can receive live logs without any extra configuration.
    """
    # Import here to avoid a circular import at module load time.
    from utils.apexchat.core.log_stream import stream_log_processor  # noqa: PLC0415

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _add_app_context,
        # ── SSE fan-out ───────────────────────────────────────────────────────
        # Must come AFTER all enrichment processors so the event dict is fully
        # populated before we copy it into the per-request queue.
        stream_log_processor,
    ]

    if LOG_FORMAT == "json" or is_production:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging for third-party libraries
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, LOG_LEVEL),
    )

    # Reduce noise from verbose third-party loggers
    for logger_name in ["httpx", "httpcore", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _add_app_context(
    logger: logging.Logger, method: str, event_dict: dict
) -> dict:
    """Add application-level context to every log entry."""
    event_dict["app"] = APP_NAME
    event_dict["version"] = APP_VERSION
    event_dict["environment"] = ENVIRONMENT
    return event_dict


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a named logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)
