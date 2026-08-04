"""Logging setup — thin re-export of :mod:`irisflow.logging.setup`."""

from __future__ import annotations

from irisflow.logging.setup import (
    bind,
    bind_frame,
    bound,
    clear_context,
    configure_logging,
    get_logger,
)

__all__ = [
    "bind",
    "bind_frame",
    "bound",
    "clear_context",
    "configure_logging",
    "get_logger",
]
