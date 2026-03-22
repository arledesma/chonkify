"""Structured logging helpers for chonkify."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Render log records as JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        structured = getattr(record, "structured", None)
        if isinstance(structured, dict):
            payload.update(structured)
        return json.dumps(payload, sort_keys=True)


def configure_logging(*, verbose: bool = False, json_logs: bool = False) -> logging.Logger:
    """Configure and return the package logger."""

    logger = logging.getLogger("chonkify")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    handler = logging.StreamHandler(sys.stderr)
    if json_logs:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured event."""

    logger.info(event, extra={"structured": {"event": event, **fields}})
