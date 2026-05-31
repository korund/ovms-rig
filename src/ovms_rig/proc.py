"""Cross-platform process spawning and termination primitives.

This module provides low-level helpers for spawning killable process groups
and terminating entire process trees (parent + all children), used across
probes and stages.

Platform differences:
- win32: CREATE_NEW_PROCESS_GROUP flag at spawn; taskkill /T for tree kill
- POSIX: start_new_session=True at spawn; killpg for tree kill

Usage:
    # Spawn a process that can be killed as a group
    kwargs = spawn_kwargs()
    proc = subprocess.Popen(cmd, **kwargs)

    # Gracefully shut down with a timeout, then force-kill if needed
    terminate_tree(proc, graceful_timeout=5.0)
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


def spawn_kwargs() -> dict[str, Any]:
    """Return subprocess.Popen kwargs that establish a killable process group.

    On win32: sets CREATE_NEW_PROCESS_GROUP so taskkill /T can reach children.
    On POSIX: sets start_new_session=True so killpg can kill the session.

    Returns:
        dict of kwargs suitable for subprocess.Popen(**kwargs)
    """
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return kwargs


def terminate_tree(proc: subprocess.Popen, graceful_timeout: float = 0.0) -> None:  # type: ignore[type-arg]
    """Terminate process and all its children.

    On win32: uses taskkill /F /T to kill the process tree.
    On POSIX: uses killpg to kill the entire process group.

    If graceful_timeout > 0, waits that long for the process to exit before
    issuing the kill. Handles already-dead processes gracefully.

    Args:
        proc: subprocess.Popen instance to terminate
        graceful_timeout: seconds to wait for graceful shutdown before force-kill (0 = immediate)
    """
    # Check if already dead
    if proc.poll() is not None:
        return

    pid = proc.pid
    try:
        # If graceful_timeout is set, try a soft termination first
        if graceful_timeout > 0:
            try:
                proc.wait(timeout=graceful_timeout)
                logger.debug("[proc] process tree (PID %d) exited gracefully", pid)
                return
            except subprocess.TimeoutExpired:
                logger.debug(
                    "[proc] process tree (PID %d) did not exit within %.1fs, force-killing",
                    pid, graceful_timeout
                )

        # Force-kill the process tree
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=5,
                check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

        # Wait for process to actually exit
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            logger.warning("[proc] process tree (PID %d) did not exit after kill signal", pid)

        logger.debug("[proc] killed process tree (PID %d)", pid)
    except Exception as e:
        logger.warning("[proc] failed to kill process tree (PID %d): %s", pid, e)
