"""Centralized logging for OpenJury using Rich.

Provides colored, structured console output with proper log levels.
All modules should use ``from openjury._logging import logger`` instead of ``print()``.

Log levels:
    - ``logger.debug()``   — Verbose details (hidden by default)
    - ``logger.info()``    — Normal operation messages
    - ``logger.warning()`` — Something unexpected but recoverable
    - ``logger.error()``   — Something failed

Environment variable ``OPENJURY_LOG_LEVEL`` controls verbosity:
    - DEBUG, INFO (default), WARNING, ERROR
"""

from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# ── Custom theme ─────────────────────────────────────────────────
_theme = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "model": "bold magenta",
        "dataset": "bold blue",
        "path": "dim",
        "number": "bold yellow",
    }
)

console = Console(theme=_theme)


def _setup_logger() -> logging.Logger:
    """Create and configure the openjury logger with Rich handler."""
    log_level = os.environ.get("OPENJURY_LOG_LEVEL", "INFO").upper()

    handler = RichHandler(
        console=console,
        show_path=False,
        show_time=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        markup=True,
    )
    handler.setLevel(getattr(logging, log_level, logging.INFO))

    _logger = logging.getLogger("openjury")
    _logger.setLevel(getattr(logging, log_level, logging.INFO))
    _logger.addHandler(handler)
    _logger.propagate = False

    return _logger


logger = _setup_logger()
