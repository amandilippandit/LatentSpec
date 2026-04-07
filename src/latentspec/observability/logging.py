"""Structured-logging configuration via `structlog`.

Outputs JSON in production (LATENTSPEC_LOG_FORMAT=json) and a readable
console renderer in dev. The standard-library `logging` module's events are
piped through structlog so third-party libraries (Anthropic SDK, Z3, sklearn)
match the same shape.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging(*, level: str | int = "INFO") -> None:
    log_format = os.environ.get("LATENTSPEC_LOG_FORMAT", "console").lower()
    json_mode = log_format == "json"

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    pre_chain = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
    ]

    processors: list = [
        *pre_chain,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_mode:
        processors.append(structlog.processors.JSONRenderer())
        formatter_renderer = structlog.processors.JSONRenderer()
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()))
        formatter_renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=pre_chain,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            formatter_renderer,
        ],
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
