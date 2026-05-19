"""Entry point and subcommand dispatcher for the ovms-rig loader."""

from __future__ import annotations

import sys

import click

from ovms_rig.stages import apply, fetch, start, status

LOG_LEVELS = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"]


@click.group()
@click.option(
    "--config",
    "config_path",
    default="config/ovms.yaml",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Path to the OVMS declaration.",
)
@click.option(
    "--local",
    "local_path",
    default="config/local.yaml",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Path to per-host overrides. Missing file is fine.",
)
@click.option(
    "--ovms-path",
    "ovms_path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Override path to the ovms binary. Wins over local.yaml and PATH.",
)
@click.option(
    "--log-level",
    "log_level",
    default=None,
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    help="Override runtime log level from ovms.yaml.",
)
@click.pass_context
def main(
    ctx: click.Context,
    config_path: str,
    local_path: str,
    ovms_path: str | None,
    log_level: str | None,
) -> None:
    """Declarative loader for OpenVINO Model Server."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["local_path"] = local_path
    ctx.obj["ovms_path"] = ovms_path
    ctx.obj["log_level"] = log_level


@main.command("status")
@click.pass_context
def cmd_status(ctx: click.Context) -> None:
    """Report current state of the rig vs declaration. Read-only, no side effects."""
    sys.exit(status.run(ctx.obj))


@main.command(
    "fetch",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def cmd_fetch(ctx: click.Context) -> None:
    """Pull missing models. Extra args are forwarded to `ovms --pull` verbatim."""
    ctx.obj["extras"] = list(ctx.args)
    sys.exit(fetch.run(ctx.obj))


@main.command("apply")
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help="Write proposed graph.pbtxt and config.json to build/ without touching live files.",
)
@click.pass_context
def cmd_apply(ctx: click.Context, dry_run: bool) -> None:
    """Back up live files into .backup/<timestamp>/ and apply the declaration to them."""
    ctx.obj["dry_run"] = dry_run
    sys.exit(apply.run(ctx.obj))


@main.command("start")
@click.pass_context
def cmd_start(ctx: click.Context) -> None:
    """Run preflight, then exec ovms in the foreground."""
    sys.exit(start.run(ctx.obj))


@main.command("preflight")
@click.pass_context
def cmd_preflight(ctx: click.Context) -> None:
    """Sugar: status -> fetch -> apply. Bring the rig to the declared state."""
    ctx.obj.setdefault("dry_run", False)
    for stage in (status, fetch, apply):
        rc = stage.run(ctx.obj)
        if rc != 0:
            sys.exit(rc)
    sys.exit(0)


if __name__ == "__main__":
    main()
