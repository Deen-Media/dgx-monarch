"""Argument parser for the lightweight :command:`dgxm` entry point."""
from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping

from . import json_report
from .update_types import DEFAULT_TARGET

Command = Callable[[argparse.Namespace], int]


def build_parser(commands: Mapping[str, Command]) -> argparse.ArgumentParser:
    """Build the public CLI while command implementations stay in leaf modules."""
    parser = argparse.ArgumentParser(
        prog="dgxm",
        description="dgx-monarch cluster CLI. To bring up a cluster, install the pack, run setup, "
        "then doctor.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="cluster.toml path (default: the first that exists of $DGXM_CLUSTER_TOML if set, "
        "./cluster.toml and ~/.config/dgx-monarch/cluster.toml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "version", help="print the dgx-monarch version and its torchmonarch pin"
    ).set_defaults(func=commands["version"])
    p_doctor = json_report.subcommand(
        sub, "doctor", "run preflight checks on the driver and every host"
    )
    p_doctor.add_argument(
        "--repair",
        action="store_true",
        help="set cluster.toml to mode 0600 (the only allowlisted repair), then run the checks",
    )
    p_doctor.add_argument(
        "--yes", action="store_true", help="apply the repair without asking"
    )
    p_doctor.add_argument(
        "--receipt", default=None, help="absolute path for the sanitized repair receipt"
    )
    p_doctor.set_defaults(func=commands["doctor"])
    sub.add_parser("init", help="write cluster.toml interactively (legacy; use setup)").set_defaults(
        func=commands["init"]
    )

    p_setup = sub.add_parser(
        "setup", help="probe the hosts you name and print the plan; --apply applies it"
    )
    p_setup.add_argument(
        "--host",
        action="append",
        default=[],
        help="a host as NAME,FABRIC_IP[,GPUS[,SSH_USER[,COMFY_DIR]]] (repeatable; omit to be asked)",
    )
    p_setup.add_argument("--client-ip", default=None, help="the driver's fabric IP, as a literal address")
    p_setup.add_argument(
        "--profile", choices=("safe", "balanced", "advanced"), default="balanced"
    )
    p_setup.add_argument("--fabric-profile", default=None)
    p_setup.add_argument("--python-bin", default="python3")
    p_setup.add_argument("--ssh-key", default="")
    p_setup.add_argument("--comfy-dir", default="")
    p_setup.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="a path relative to ComfyUI whose bytes must match on every host (repeatable)",
    )
    p_setup.add_argument(
        "--output",
        dest="setup_config_path",
        default=None,
        help="where to write cluster.toml (default: --config, else ~/.config/dgx-monarch/cluster.toml)",
    )
    p_setup.add_argument("--apply", action="store_true", help="apply the plan (asks first unless --yes)")
    p_setup.add_argument("--yes", action="store_true", help="with --apply, skip the confirmation prompt")
    p_setup.add_argument(
        "--acknowledge-trusted-fabric",
        action="store_true",
        help="confirm the dedicated fabric is source-restricted to trusted peers",
    )
    p_setup.add_argument(
        "--install-service",
        action="store_true",
        help="install the worker service as a systemd user unit, but do not enable or start it",
    )
    p_setup.add_argument(
        "--start-worker-service",
        action="store_true",
        help="allow setup to enable and start the worker service; needs --install-service, "
        "and --verify needs this flag",
    )
    p_setup.add_argument(
        "--privileged-process-inspection",
        action="store_true",
        help="use the root process-inspection helper an administrator already started (docs/INSTALL.md)",
    )
    p_setup.add_argument(
        "--verify",
        action="store_true",
        help="after setup starts the workers, run doctor, then the attach, NCCL and source smoke",
    )
    p_setup.add_argument("--receipt", default=None, help="absolute path for the sanitized receipt")
    p_setup.add_argument(
        "--json", action="store_true", help="print the sanitized plan and result as JSON (an apply needs --yes)"
    )
    p_setup.set_defaults(func=commands["setup"])

    p_up = sub.add_parser(
        "up", help="sync the package, then start or restart the worker service on every host"
    )
    p_up.add_argument(
        "--no-sync", action="store_true", help="skip rsyncing the package to hosts"
    )
    p_up.set_defaults(func=commands["up"])

    sub.add_parser(
        "down", help="stop the worker service and its actors on every host"
    ).set_defaults(func=commands["down"])

    p_reap = sub.add_parser(
        "reap", help="kill actor processes that outlived their client, on every host and this box"
    )
    p_reap.add_argument("--dry-run", action="store_true", help="print the plan, signal nothing")
    p_reap.add_argument(
        "--grace",
        type=float,
        default=0.0,
        help="only processes orphaned this many seconds, or alive that long "
        "when nothing recorded a dwell clock for them",
    )
    p_reap.set_defaults(func=commands["reap"])

    p_top = sub.add_parser("top", help="live cluster dashboard (UMA pool, rails, renders, gates)")
    p_top.add_argument(
        "--host",
        default=None,
        help="driver ComfyUI host:port (default: a local driver, preferring one with a live mesh; "
        "else 127.0.0.1:8188)",
    )
    p_top.add_argument(
        "--interval", type=float, default=1.0, help="seconds between polls, 0.1 to 3600 (default: 1)"
    )
    p_top.add_argument("--replay", default=None, help="play back a recorded .jsonl file instead of live data")
    p_top.add_argument("--record", default=None, help="also append every tick to this .jsonl file")
    p_top.add_argument("--theme", default="spark", choices=["spark", "ember", "mono"])
    p_top.set_defaults(func=commands["top"])

    p_gate = sub.add_parser(
        "gate", help="identity-gate the last prompt you ran; your workflow is not changed"
    )
    p_gate.add_argument(
        "--host",
        default=None,
        help="driver ComfyUI host:port (default: a local driver, preferring one with a live mesh; "
        "else 127.0.0.1:8188)",
    )
    p_gate.add_argument(
        "--report-dir",
        default=os.path.expanduser("~/ComfyUI/output"),
        help="ComfyUI output directory that holds the gate ledger and reports",
    )
    p_gate.add_argument("--timeout", type=int, default=1800)
    gate_action = p_gate.add_mutually_exclusive_group()
    gate_action.add_argument(
        "--last",
        dest="gate_action",
        action="store_const",
        const="last",
        help="gate the most recent prompt (default)",
    )
    gate_action.add_argument(
        "--list",
        dest="gate_action",
        action="store_const",
        const="list",
        help="print the newest recorded verdict for each gate key and exit",
    )
    gate_action.add_argument(
        "--repair",
        dest="gate_action",
        action="store_const",
        const="repair",
        help="list open RETESTING rows that no terminal row superseded",
    )
    p_gate.add_argument(
        "--apply",
        action="store_true",
        help="with --repair, append the missing terminal INCONCLUSIVE rows",
    )
    p_gate.set_defaults(func=commands["gate"], gate_action="last")

    p_restart = sub.add_parser(
        "restart", help="run down, then up; nothing starts if the stop fails"
    )
    p_restart.add_argument("--no-sync", action="store_true")
    p_restart.set_defaults(func=commands["restart"])

    json_report.subcommand(
        sub, "status", "show each host's worker service and listener, and the attached mesh"
    ).set_defaults(func=commands["status"])
    sub.add_parser(
        "install-service",
        help="install, enable and start the worker service as a systemd user unit on every host",
    ).set_defaults(func=commands["install_service"])
    p_update = sub.add_parser(
        "update",
        help="pull, reinstall and restart worker services, then run doctor; "
        "--verify stages and proves an exact release",
    )
    p_update.add_argument(
        "--host",
        default=None,
        help="ComfyUI host:port to check first; the update refuses while a driver runs there "
        "(local ports are checked too), and with --verify an explicit --host always refuses",
    )
    p_update.add_argument(
        "--verify",
        action="store_true",
        help="stage an exact release on every host and verify it before declaring success",
    )
    p_update.add_argument(
        "--target", default=DEFAULT_TARGET, help="exact commit or ref for --verify (default: origin default branch)"
    )
    p_update.add_argument(
        "--yes", action="store_true", help="with --verify, activate the staged release without asking"
    )
    p_update.add_argument("--receipt", default=None, help="absolute path for the sanitized --verify receipt")
    p_update.set_defaults(func=commands["update"])
    sub.add_parser(
        "uninstall",
        help="stop and remove the worker services and units, and offer to delete cluster.toml; "
        "models and venvs stay",
    ).set_defaults(func=commands["uninstall"])
    return parser
