"""Centralized logging configuration.

Import `get_logger(__name__)` from application modules instead of
calling `logging.getLogger` directly, so log formatting stays
consistent across the project.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def configure_logging() -> None:
    """Configure the root logger once per process.

    Reads LOG_LEVEL directly from the environment rather than going
    through `app.core.config.Settings`, so that modules needing only
    logging (e.g. the tools layer) don't pull in a pydantic dependency
    just to log a message.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Avoid duplicate handlers if this is somehow called twice.
    root_logger.handlers = [handler]

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring logging on first use."""
    configure_logging()
    return logging.getLogger(name)
