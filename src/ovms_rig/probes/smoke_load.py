"""Smoke-load validation: probe OVMS with generated config to catch parse errors.

Launches OVMS in minimal probe mode and parses output for fail-markers
(libprotobuf ERROR, LOADING_PRECONDITION_FAILED, mediapipe parse failures)
and success-markers (MediapipeGraphDefinition initializing graph nodes).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

from ovms_rig.config import Declaration
from ovms_rig.env import build_env
from ovms_rig.command import build as build_command
from ovms_rig.probes import ovms_binary
from ovms_rig.report import CheckResult

logger = logging.getLogger(__name__)

NAME = "smoke-load"


class ProbeFailure(Exception):
    """Probe failed with diagnostic log tail."""

    def __init__(self, msg: str, log_tail: list[str]):
        super().__init__(msg)
        self.log_tail = log_tail


def check(decl: Declaration) -> CheckResult:
    """Validate generated config.json by probing OVMS.

    Args:
        decl: Declaration with config paths and OVMS binary location.

    Returns:
        CheckResult with status, summary, details (fail_markers), hint.
    """
    ovms_cfg = decl.ovms
    local = decl.local

    # Determine active profile and graph count.
    active_models: set[str] = set()
    for profile_name, profile in ovms_cfg.profiles.items():
        if profile.active:
            active_models = set(profile.models)
            break

    num_graphs = len(active_models)

    # No active profile: nothing to validate.
    if num_graphs == 0:
        return CheckResult(
            name=NAME,
            status="ok",
            summary="no active profile to validate",
        )

    # Resolve OVMS binary.
    binary, _src = ovms_binary.resolve(decl.cli_override, local)
    if binary is None or not binary.is_file():
        return CheckResult(
            name=NAME,
            status="warn",
            summary="ovms binary not resolved; smoke-load skipped",
            hint="run `ovms-rig status` to diagnose",
        )

    store = local.models.repository_path
    config_json = store / "config.json"

    # Skip if config.json doesn't exist (likely fresh rig, no activation done yet).
    if not config_json.is_file():
        return CheckResult(
            name=NAME,
            status="ok",
            summary="config.json not yet generated; smoke-load skipped",
            hint="run `ovms-rig activate {profile_name}` to generate config and run smoke-load",
        )

    # Run the probe.
    fail_markers = []
    try:
        fail_markers, recent_lines = _probe_ovms(binary, config_json, num_graphs, ovms_cfg.runtime, local.runtime)
    except ProbeFailure as e:
        details = {"log_tail": e.log_tail}
        return CheckResult(
            name=NAME,
            status="error",
            summary=str(e),
            details=details,
        )
    except Exception as e:
        return CheckResult(
            name=NAME,
            status="error",
            summary=f"smoke-load failed: {e}",
        )

    if fail_markers:
        return CheckResult(
            name=NAME,
            status="error",
            summary="OVMS rejected config",
            details={"fail_markers": fail_markers, "log_tail": recent_lines},
        )

    return CheckResult(
        name=NAME,
        status="ok",
        summary=f"OVMS accepted config: {num_graphs} graph(s) validated",
    )


def _probe_ovms(
    binary: Path,
    config_json: Path,
    num_graphs: int,
    runtime,
    local_runtime,
) -> tuple[list[str], list[str]]:
    """Launch OVMS probe and collect fail-markers.

    Args:
        binary: Path to ovms executable.
        config_json: Path to config.json to probe.
        num_graphs: Expected number of graphs to initialize.
        runtime: Parsed Runtime section of ovms.yaml.
        local_runtime: Local runtime config (cache_dir, etc).

    Returns:
        Tuple of (fail_marker_lines, recent_log_lines).

    Raises:
        ProbeFailure: On timeout or graph initialization failure, with log tail.
    """
    cmd = build_command(binary, config_json, runtime, local_runtime, [], log_level_override="DEBUG")
    env = build_env(binary.parent)

    log_file = tempfile.NamedTemporaryFile(suffix=".log", delete=False, mode="w")
    log_path = log_file.name
    log_file.close()

    cmd.extend(["--log_path", log_path])
    logger.debug("[smoke-load] command: %s", " ".join(cmd))

    fail_lines: list[str] = []
    recent_lines: deque[str] = deque(maxlen=50)
    initialized_count = 0
    proc: subprocess.Popen | None = None

    try:
        kwargs: dict = {"env": env}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )

        start_time = time.time()
        timeout_sec = 30

        with open(log_path, "r") as f:
            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout_sec:
                    raise ProbeFailure(
                        f"smoke-load timed out after {timeout_sec}s "
                        f"(expected {num_graphs} graph(s), got {initialized_count})",
                        list(recent_lines),
                    )

                line = f.readline()

                if not line:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.1)
                    continue

                line = line.rstrip("\n\r")
                logger.debug("[smoke-load] %s", line)
                recent_lines.append(line)

                # Check fail markers.
                if _is_fail_line(line):
                    fail_lines.append(line)
                    logger.warning("[smoke-load] fail marker: %s", line)

                # Check success marker.
                if not fail_lines and "MediapipeGraphDefinition initializing graph nodes" in line:
                    initialized_count += 1
                    logger.debug("[smoke-load] graph %d/%d initialized", initialized_count, num_graphs)

                # Stop if we hit fail or success.
                if fail_lines or initialized_count >= num_graphs:
                    break

        if fail_lines:
            return fail_lines, list(recent_lines)

        if initialized_count < num_graphs:
            raise ProbeFailure(
                f"expected {num_graphs} graph(s), saw {initialized_count} initializing",
                list(recent_lines),
            )

        logger.info("[smoke-load] validation passed: all %d graph(s) initialized", num_graphs)
        return [], list(recent_lines)

    finally:
        if proc is not None and proc.poll() is None:
            _kill_process_tree(proc)

        try:
            os.unlink(log_path)
        except (OSError, FileNotFoundError):
            pass


def _is_fail_line(line: str) -> bool:
    """Check if a log line matches any fail-marker pattern."""
    if "[libprotobuf ERROR" in line:
        return True
    if "state changed to:" in line and "LOADING_PRECONDITION_FAILED" in line:
        return True
    if "Trying to parse mediapipe graph definition:" in line and "failed" in line:
        return True
    return False


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill process and all its children."""
    if proc.poll() is not None:
        return

    pid = proc.pid
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=5,
                check=False,
            )
        else:
            import os
            import signal

            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass

        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass

        logger.debug("[smoke-load] killed process tree (PID %d)", pid)
    except Exception as e:
        logger.warning("[smoke-load] failed to kill process tree (PID %d): %s", pid, e)
