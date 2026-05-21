"""Start stage: precheck then run ovms in the foreground.

Precheck calls the status stage internally. Hard errors from status
(binary not found, config invalid, port in use) fail immediately. Soft
checks (missing models, live config mismatch) emit warnings and continue.

After precheck the stage builds the ovms command, inherits the process
environment from env.build_env(), forks ovms via subprocess.Popen, and
stays alive as the parent -- forwarding SIGTERM/SIGINT to ovms and
exiting with the same return code.
"""

from __future__ import annotations

from ovms_rig.stages.start.launch import run

__all__ = ["run"]
