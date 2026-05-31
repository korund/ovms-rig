"""Smoke-load validation: probe OVMS with generated config to catch parse errors.

Launches OVMS in minimal probe mode and parses output for fail-markers
(libprotobuf ERROR, LOADING_PRECONDITION_FAILED, mediapipe parse failures)
and success-markers (MediapipeGraphDefinition initializing graph nodes).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from collections import deque
from pathlib import Path

from ovms_rig.config import Declaration
from ovms_rig.env import build_env
from ovms_rig.command import build as build_command
from ovms_rig.proc import spawn_kwargs, terminate_tree
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

    # Determine active profile and split models by type: mediapipe (LLM,
    # task-based) signal load via a graph-init marker; plain (model_config_list)
    # models signal load via a model status-change to AVAILABLE.
    active_models: set[str] = set()
    for profile_name, profile in ovms_cfg.profiles.items():
        if profile.active:
            active_models = set(profile.models)
            break

    expected_mediapipe = 0
    expected_plain = 0
    for name in active_models:
        entry = ovms_cfg.models.get(name)
        if entry is None:
            continue
        if ovms_cfg.repository[entry.source].task is None:
            expected_plain += 1
        else:
            expected_mediapipe += 1
    num_expected = expected_mediapipe + expected_plain

    # No active profile / no models: nothing to validate.
    if num_expected == 0:
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
        fail_markers, recent_lines = _probe_ovms(
            binary, config_json, expected_mediapipe, expected_plain,
            ovms_cfg.runtime, local.runtime,
        )
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
        summary=f"OVMS accepted config: {num_expected} model(s) validated",
    )


def _probe_ovms(
    binary: Path,
    config_json: Path,
    expected_mediapipe: int,
    expected_plain: int,
    runtime,
    local_runtime,
) -> tuple[list[str], list[str]]:
    """Launch OVMS probe and collect fail-markers.

    Args:
        binary: Path to ovms executable.
        config_json: Path to config.json to probe.
        expected_mediapipe: number of LLM graphs expected to initialize.
        expected_plain: number of plain models expected to reach AVAILABLE.
        runtime: Parsed Runtime section of ovms.yaml.
        local_runtime: Local runtime config (cache_dir, etc).

    Returns:
        Tuple of (fail_marker_lines, recent_log_lines).

    Raises:
        ProbeFailure: On timeout or load failure, with log tail.
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
    mediapipe_count = 0
    plain_count = 0
    proc: subprocess.Popen | None = None

    try:
        kwargs = spawn_kwargs()
        kwargs["env"] = env
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL

        proc = subprocess.Popen(cmd, **kwargs)

        start_time = time.time()
        timeout_sec = 30

        with open(log_path, "r") as f:
            while True:
                elapsed = time.time() - start_time
                if elapsed > timeout_sec:
                    raise ProbeFailure(
                        f"smoke-load timed out after {timeout_sec}s "
                        f"(mediapipe {mediapipe_count}/{expected_mediapipe}, "
                        f"plain {plain_count}/{expected_plain})",
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

                # Check success markers, both chosen to fire EARLY (config
                # accepted, load begun) -- smoke must not wait for the full model
                # to compile into memory. Mediapipe (LLM) graphs log a graph-init
                # line; plain models transition to "LOADING" with error_code OK
                # (verified against a real plain pp-doclayout-m load: LOADING
                # precedes the heavy compile that ends in AVAILABLE). Parse /
                # precondition failures surface as fail-markers before this point.
                if not fail_lines:
                    if "MediapipeGraphDefinition initializing graph nodes" in line:
                        mediapipe_count += 1
                    elif '"state": "LOADING"' in line and '"error_code": "OK"' in line:
                        plain_count += 1

                # Stop if we hit fail or both expectations are met.
                if fail_lines or (
                    mediapipe_count >= expected_mediapipe
                    and plain_count >= expected_plain
                ):
                    break

        if fail_lines:
            return fail_lines, list(recent_lines)

        if mediapipe_count < expected_mediapipe or plain_count < expected_plain:
            raise ProbeFailure(
                f"expected mediapipe {expected_mediapipe}, plain {expected_plain}; "
                f"saw mediapipe {mediapipe_count}, plain {plain_count}",
                list(recent_lines),
            )

        logger.info(
            "[smoke-load] validation passed: mediapipe %d, plain %d",
            mediapipe_count, expected_plain,
        )
        return [], list(recent_lines)

    finally:
        if proc is not None and proc.poll() is None:
            terminate_tree(proc, graceful_timeout=0.0)

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
