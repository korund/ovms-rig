"""Start stage: run blocking probes, then exec ovms in the foreground.

Blocking probes (declaration, ovms binary, models, port) must pass before launch.
Diagnostic probes are not run here -- `rig status` shows the full picture.
The process inherits env from env.build_env(), forks ovms via subprocess.Popen, and
stays alive forwarding SIGTERM/SIGINT.
"""

from __future__ import annotations

from ovms_rig.stages.start.launch import run

__all__ = ["run"]
