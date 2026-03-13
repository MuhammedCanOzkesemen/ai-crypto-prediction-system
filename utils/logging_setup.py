"""
Application logging setup for the cryptocurrency forecasting platform.

Provides a reusable logger factory and consistent format.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

# -----------------------------------------------------------------------------
# Format and level
# -----------------------------------------------------------------------------

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = logging.INFO


def get_logger(
    name: str,
    level: int | str = DEFAULT_LEVEL,
    format_string: str = DEFAULT_FORMAT,
    stream: Any = sys.stdout,
) -> logging.Logger:
    """
    Return a configured logger for the given module/component name.

    Parameters
    ----------
    name : str
        Logger name (typically __name__ of the calling module).
    level : int or str
        Logging level (e.g. logging.INFO or "INFO").
    format_string : str
        Log message format.
    stream : file-like
        Output stream for the handler.

    Returns
    -------
    logging.Logger
        Configured logger. If the name already has handlers, the existing
        logger is returned without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    if isinstance(level, str):
        level = getattr(logging, level.upper(), DEFAULT_LEVEL)
    logger.setLevel(level)
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    formatter = logging.Formatter(format_string, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def configure_root_logger(
    level: int | str = DEFAULT_LEVEL,
    format_string: str = DEFAULT_FORMAT,
) -> None:
    """
    Configure the root logger (e.g. at application startup).
    Call once from the main entry point or API app.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), DEFAULT_LEVEL)
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt=DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )
