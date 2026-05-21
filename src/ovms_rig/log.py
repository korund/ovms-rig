"""Logging setup driven by runtime.log_level from the declaration.

Adds a TRACE level (below DEBUG) so the thin client mirrors OVMS levels.
"""

from __future__ import annotations

import logging

TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def _trace(self: logging.Logger, msg: str, *args, **kwargs) -> None:
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)


logging.Logger.trace = _trace  # type: ignore[attr-defined]


_LEVELS = {
    "TRACE": TRACE,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


_OWN_HANDLERS: list[logging.Handler] = []


def configure(level_name: str) -> None:
    """Install a single stderr handler at the given level.

    Always rebinds the handler to the current sys.stderr so repeat calls (eg.
    inside test runners that capture stderr) take effect. Only removes
    handlers this module installed previously -- foreign handlers (eg. pytest
    caplog's LogCaptureHandler) are left in place.
    """
    import sys

    level = _LEVELS[level_name]
    root = logging.getLogger()
    for h in list(_OWN_HANDLERS):
        if h in root.handlers:
            root.removeHandler(h)
        _OWN_HANDLERS.remove(h)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler.setLevel(level)
    root.addHandler(handler)
    _OWN_HANDLERS.append(handler)
    root.setLevel(level)
