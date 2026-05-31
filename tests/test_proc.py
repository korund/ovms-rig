"""Integration tests for process spawning and termination.

Tests verify that spawn_kwargs establishes a killable process group and
terminate_tree can kill an entire process tree (parent + all children)
on both win32 and POSIX platforms.

Each test spawns an actual child process that itself spawns a grandchild,
then calls terminate_tree and verifies the entire tree is dead.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import textwrap
import time

import pytest

from ovms_rig.proc import spawn_kwargs, terminate_tree


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _spawn_python_child(grandchild_sleep_s: int = 30) -> subprocess.Popen:
    """Spawn a Python subprocess that spawns a grandchild process.

    The child process runs a simple script that:
    1. Spawns a grandchild process (sleep command)
    2. Waits indefinitely, giving the grandchild time to become a zombie/reaper

    Args:
        grandchild_sleep_s: How long the grandchild should sleep (default 30s)

    Returns:
        Popen instance of the child process
    """
    # Build script with proper indentation for python -c execution
    script = textwrap.dedent(f"""
import subprocess
import sys
import time

if sys.platform == "win32":
    proc = subprocess.Popen(
        ["powershell", "-Command", "Start-Sleep -Seconds {grandchild_sleep_s}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
else:
    proc = subprocess.Popen(
        ["sleep", "{grandchild_sleep_s}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

print(f"grandchild_pid={{proc.pid}}", flush=True)
sys.stdout.flush()

while True:
    time.sleep(0.1)
""").strip()

    kwargs = spawn_kwargs()
    kwargs["stdout"] = subprocess.PIPE
    kwargs["stderr"] = subprocess.DEVNULL
    kwargs["text"] = True

    proc = subprocess.Popen([sys.executable, "-c", script], **kwargs)
    return proc


def _read_grandchild_pid(child_proc: subprocess.Popen, timeout_s: float = 2.0) -> int | None:
    """Read the grandchild PID from child process stdout.

    The child process prints "grandchild_pid=<PID>" to stdout on startup.
    We read this line to get the grandchild process ID.

    Args:
        child_proc: Popen instance of the child process
        timeout_s: Max seconds to wait for the PID line

    Returns:
        The grandchild PID, or None if timeout or EOF
    """
    start = time.time()
    while time.time() - start < timeout_s:
        line = child_proc.stdout.readline()
        if not line:
            return None
        line = line.strip()
        if line.startswith("grandchild_pid="):
            try:
                return int(line.split("=")[1])
            except (ValueError, IndexError):
                return None
        time.sleep(0.01)
    return None


def _is_process_alive(pid: int) -> bool:
    """Check if a process with the given PID is alive.

    Args:
        pid: Process ID to check

    Returns:
        True if the process exists, False otherwise
    """
    if sys.platform == "win32":
        # On Windows, use tasklist to check process existence
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            return str(pid) in result.stdout
        except (subprocess.TimeoutExpired, Exception):
            return False
    else:
        # On POSIX, use kill -0 (signals 0 to check existence without sending signal)
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


# ---------------------------------------------------------------------------
# Tests: spawn_kwargs establishes a process group
# ---------------------------------------------------------------------------

def test_spawn_kwargs_returns_dict():
    """spawn_kwargs returns a non-empty dict of Popen kwargs."""
    kwargs = spawn_kwargs()
    assert isinstance(kwargs, dict)
    assert len(kwargs) > 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
def test_spawn_kwargs_win32_sets_creation_flags():
    """On win32, spawn_kwargs sets CREATE_NEW_PROCESS_GROUP."""
    kwargs = spawn_kwargs()
    assert "creationflags" in kwargs
    assert kwargs["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-specific test")
def test_spawn_kwargs_posix_sets_start_new_session():
    """On POSIX, spawn_kwargs sets start_new_session=True."""
    kwargs = spawn_kwargs()
    assert "start_new_session" in kwargs
    assert kwargs["start_new_session"] is True


# ---------------------------------------------------------------------------
# Tests: terminate_tree kills an entire process tree
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test (uses taskkill)")
def test_terminate_tree_win32_kills_process_tree():
    """On win32, terminate_tree kills the entire process tree with taskkill."""
    child_proc = _spawn_python_child(grandchild_sleep_s=30)

    # Extract grandchild PID
    grandchild_pid = _read_grandchild_pid(child_proc, timeout_s=3.0)
    assert grandchild_pid is not None, "Failed to read grandchild PID"

    # Verify both child and grandchild are alive before kill
    assert child_proc.poll() is None, "Child should be alive"
    assert _is_process_alive(grandchild_pid), f"Grandchild {grandchild_pid} should be alive"

    # Kill the tree
    terminate_tree(child_proc, graceful_timeout=0.0)

    # Verify both are dead
    time.sleep(0.5)  # Give the OS time to reap processes
    assert child_proc.poll() is not None, "Child should be dead"
    assert not _is_process_alive(grandchild_pid), f"Grandchild {grandchild_pid} should be dead"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-specific test (uses killpg)")
def test_terminate_tree_posix_kills_process_tree():
    """On POSIX, terminate_tree kills the entire process group with killpg."""
    child_proc = _spawn_python_child(grandchild_sleep_s=30)

    # Extract grandchild PID
    grandchild_pid = _read_grandchild_pid(child_proc, timeout_s=3.0)
    assert grandchild_pid is not None, "Failed to read grandchild PID"

    # Verify both child and grandchild are alive before kill
    assert child_proc.poll() is None, "Child should be alive"
    assert _is_process_alive(grandchild_pid), f"Grandchild {grandchild_pid} should be alive"

    # Kill the tree
    terminate_tree(child_proc, graceful_timeout=0.0)

    # Verify both are dead
    time.sleep(0.5)  # Give the OS time to reap processes
    assert child_proc.poll() is not None, "Child should be dead"
    assert not _is_process_alive(grandchild_pid), f"Grandchild {grandchild_pid} should be dead"


def test_terminate_tree_graceful_timeout_waits():
    """terminate_tree with graceful_timeout waits before force-killing."""
    # Spawn a short-lived child (will exit quickly)
    script = "import time; time.sleep(0.1)"
    kwargs = spawn_kwargs()
    kwargs["stdout"] = subprocess.DEVNULL
    kwargs["stderr"] = subprocess.DEVNULL
    proc = subprocess.Popen([sys.executable, "-c", script], **kwargs)

    # Call terminate_tree with a long timeout; child should exit gracefully
    start = time.time()
    terminate_tree(proc, graceful_timeout=5.0)
    elapsed = time.time() - start

    # Should have exited in < 1s (graceful exit), not waited full timeout
    assert elapsed < 1.0, f"Expected graceful exit in <1s, took {elapsed:.2f}s"
    assert proc.poll() is not None, "Child should be dead"


def test_terminate_tree_idempotent_on_already_dead():
    """terminate_tree is idempotent: can be called on already-dead process."""
    script = "import time; time.sleep(0.1)"
    kwargs = spawn_kwargs()
    kwargs["stdout"] = subprocess.DEVNULL
    kwargs["stderr"] = subprocess.DEVNULL
    proc = subprocess.Popen([sys.executable, "-c", script], **kwargs)

    # Wait for it to exit naturally
    proc.wait()
    assert proc.poll() is not None, "Child should be dead"

    # Call terminate_tree on already-dead process; should not raise
    terminate_tree(proc, graceful_timeout=0.0)
    assert proc.poll() is not None, "Child should still be dead"


def test_terminate_tree_zero_timeout_force_kills():
    """terminate_tree with graceful_timeout=0 force-kills immediately."""
    child_proc = _spawn_python_child(grandchild_sleep_s=30)

    # Extract grandchild PID
    grandchild_pid = _read_grandchild_pid(child_proc, timeout_s=3.0)
    assert grandchild_pid is not None, "Failed to read grandchild PID"

    # Verify both are alive
    assert child_proc.poll() is None, "Child should be alive"
    assert _is_process_alive(grandchild_pid), f"Grandchild {grandchild_pid} should be alive"

    # Kill with zero timeout (no graceful wait)
    start = time.time()
    terminate_tree(child_proc, graceful_timeout=0.0)
    elapsed = time.time() - start

    # Should be killed quickly (not waiting the full 30s sleep)
    assert elapsed < 5.0, f"Expected fast kill, took {elapsed:.2f}s"

    # Both should be dead
    time.sleep(0.5)
    assert child_proc.poll() is not None, "Child should be dead"
    assert not _is_process_alive(grandchild_pid), f"Grandchild {grandchild_pid} should be dead"
