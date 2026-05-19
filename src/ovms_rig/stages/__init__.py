"""Pipeline stages. Each module exposes a run(ctx: dict) -> int callable."""

from ovms_rig.stages import apply, fetch, start, status

__all__ = ["apply", "fetch", "start", "status"]
