"""Pipeline stages. Each module exposes a run(ctx: dict) -> int callable."""

from ovms_rig.stages import activation, fetch, remove, start, status

__all__ = ["activation", "fetch", "remove", "start", "status"]
