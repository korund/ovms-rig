"""Signal forwarding and graceful shutdown for the ovms child process.

On POSIX the rig installs SIGTERM and SIGINT handlers that forward the
signal to the child and then wait.  After GRACEFUL_TIMEOUT_S seconds the
child is sent SIGKILL.

On win32 Windows does not support POSIX signals in the same way.  The rig
calls proc.send_signal(CTRL_BREAK_EVENT) on SIGBREAK and falls back to
proc.terminate() on SIGINT (Ctrl-C is handled by the console naturally via
the shared console group).  After GRACEFUL_TIMEOUT_S seconds proc.kill()
is used.
"""

from __future__ import annotations

import logging
import signal
import subprocess
import sys

from ovms_rig.proc import terminate_tree

GRACEFUL_TIMEOUT_S = 30

logger = logging.getLogger(__name__)


def install(proc: subprocess.Popen) -> None:  # type: ignore[type-arg]
    """Install OS-appropriate signal handlers that forward to *proc*."""
    if sys.platform == "win32":
        _install_win32(proc)
    else:
        _install_posix(proc)


# ---------------------------------------------------------------------------
# POSIX
# ---------------------------------------------------------------------------

def _install_posix(proc: subprocess.Popen) -> None:  # type: ignore[type-arg]
    def _forward(signum: int, _frame: object) -> None:
        logger.debug("received signal %d, forwarding to ovms (pid=%d)", signum, proc.pid)
        try:
            proc.send_signal(signum)
        except ProcessLookupError:
            pass  # child already gone
        _wait_or_kill(proc)

    signal.signal(signal.SIGTERM, _forward)
    signal.signal(signal.SIGINT, _forward)


# ---------------------------------------------------------------------------
# win32
# ---------------------------------------------------------------------------

def _install_win32(proc: subprocess.Popen) -> None:  # type: ignore[type-arg]
    # SIGBREAK maps to CTRL_BREAK_EVENT on Windows and can be forwarded to
    # a child in the same process group.
    def _on_sigbreak(signum: int, _frame: object) -> None:
        logger.debug("received SIGBREAK, sending CTRL_BREAK_EVENT to ovms (pid=%d)", proc.pid)
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        except (ProcessLookupError, OSError):
            pass
        _wait_or_kill(proc)

    def _on_sigint(signum: int, _frame: object) -> None:
        logger.debug("received SIGINT, terminating ovms (pid=%d)", proc.pid)
        try:
            proc.terminate()
        except (ProcessLookupError, OSError):
            pass
        _wait_or_kill(proc)

    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _on_sigbreak)  # type: ignore[attr-defined]
    signal.signal(signal.SIGINT, _on_sigint)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _wait_or_kill(proc: subprocess.Popen) -> None:  # type: ignore[type-arg]
    """Wait up to GRACEFUL_TIMEOUT_S then force-kill."""
    terminate_tree(proc, graceful_timeout=GRACEFUL_TIMEOUT_S)
