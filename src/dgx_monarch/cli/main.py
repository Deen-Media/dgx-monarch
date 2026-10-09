"""The `dgxm` command handlers and their narrow user-facing error handling.

Every stage is idempotent (DESIGN.md §11 rule 5). Bring-up is one command per
stage: install, then setup, then doctor.
"""
from __future__ import annotations

import sys
from pathlib import Path

from .. import TORCHMONARCH_PIN, __version__
from ..config import (
    ClusterConfig,
    ClusterConfigError,
    find_config_path,
    load_cluster_config,
)
from ..tui.data import poll_interval_is_valid
from . import (
    actor_sweep,
    arguments,
    comfy_ports,
    doctor_repair,
    legacy_init,
    legacy_update,
    lifecycle,
    mesh_health_row,
    pincheck,
    probe_certainty,
)
from . import doctor as doctor_mod
from .config_removal import remove_exact_config
from .setup_config_io import read_snapshot
from .setup_config_lock import SetupConfigLock, SetupConfigLockUnavailable
from .update_types import DEFAULT_TARGET


def _load_config(args) -> ClusterConfig | None:
    try:
        path = find_config_path(getattr(args, "config", None))
        if path is None:
            return None
        return load_cluster_config(path)
    except ClusterConfigError as exc:
        # A found-but-malformed cluster.toml (unknown fabric_profile, hostname
        # bind, bad TOML) must print its actionable message, not a raw traceback.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _require_config(args):
    config = _load_config(args)
    if config is None:
        print(
            "no cluster.toml found (searched $DGXM_CLUSTER_TOML if set, ./cluster.toml and "
            "~/.config/dgx-monarch/cluster.toml). "
            "Run `dgxm setup` (or the legacy `dgxm init`), or pass --config.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return config


def cmd_version(_args) -> int:
    print(f"dgx-monarch {__version__} (torchmonarch pin {TORCHMONARCH_PIN})")
    return 0


def cmd_doctor(args) -> int:
    config = _load_config(args)
    if getattr(args, "repair", False):
        if args.json:
            print("error: --repair cannot be combined with --json", file=sys.stderr)
            return 2
        if config is None or not config.source:
            print("error: --repair needs a loaded cluster.toml", file=sys.stderr)
            return 2
        repaired, receipt = doctor_repair.run_config_permission_repair(
            config.source,
            assume_yes=bool(getattr(args, "yes", False)),
            receipt_path=getattr(args, "receipt", None),
            output=sys.stdout,
        )
        if receipt is not None:
            print(f"repair receipt: {receipt}")
        if not repaired:
            return 1
    elif getattr(args, "yes", False) or getattr(args, "receipt", None):
        print("error: --yes and --receipt require --repair", file=sys.stderr)
        return 2
    return doctor_mod.doctor_exit(config, as_json=args.json)


def cmd_init(args) -> int:
    return legacy_init.run(args)


def cmd_setup(args) -> int:
    from .setup_cli import run

    return run(args)


def cmd_up(args) -> int:
    config = _require_config(args)
    # Run the start refusals that need no host before the pin probe, which reaches every host.
    if not lifecycle.start_preflight(config):
        return 1
    pincheck.warn_on_pin_mismatch(config, TORCHMONARCH_PIN)
    return 0 if lifecycle.up(config, sync=not args.no_sync) else 1


def cmd_down(args) -> int:
    return 0 if lifecycle.down(_require_config(args)) else 1


def cmd_restart(args) -> int:
    config = _require_config(args)
    if not lifecycle.start_preflight(config):
        return 1
    pincheck.warn_on_pin_mismatch(config, TORCHMONARCH_PIN)
    return 0 if lifecycle.restart(config, sync=not args.no_sync) else 1


def cmd_reap(args) -> int:
    """Sweep actor processes that outlived their client.

    Needs no cluster.toml: a local-mode fleet spawns its actors from the
    ComfyUI process itself, so it has no hosts for `down` or `up` to iterate
    and this is its only sweep.
    """
    ok = actor_sweep.reap_command(
        _load_config(args), dry_run=args.dry_run, grace_s=args.grace)
    return 0 if ok else 1


def cmd_status(args) -> int:
    rows, mesh_block = mesh_health_row.status_report(
        _require_config(args), as_json=args.json
    )
    bad = [r for r in rows if r["loop"] == "stopped" or r["port"] == "closed"]
    if bad or mesh_health_row.doctor_verdict(mesh_block)[0] == "fail":
        return 1
    # Exit 0 requires an observed, healthy mesh. Missing ComfyUI or unreadable
    # loop/port state gets a separate unknown code, not a stopped or failed verdict.
    blind = [r for r in rows if "unknown" in (r["loop"], r["port"])]
    return (probe_certainty.UNKNOWN_EXIT
            if blind or mesh_health_row.unobserved(mesh_block) else 0)


def _cmd_gate_impl(args) -> int:
    """Gate the last-executed prompt without touching the user's workflow:
    pull it from /history, swap the sampler for the Identity Gate node in the
    API copy, queue, and report the verdict."""
    import json
    import time
    import urllib.request
    import uuid

    action = getattr(
        args,
        "gate_action",
        "list" if getattr(args, "list", False) else "last",
    )
    if action == "list":  # disk-only: no driver, no probe
        from ..gate_audit import trust_rows
        from ..gate_ledger import GateLedger

        entries = trust_rows(GateLedger(args.report_dir).entries())
        latest = {e.get("key"): e for e in entries}
        if not latest:
            print(f"no gate verdicts recorded yet ({args.report_dir})")
            return 0
        for e in latest.values():
            print(f"{e.get('verdict', '?'):12s} {e.get('model', '?'):45s} "
                  f"loras={e.get('loras', '?')} comfy={e.get('comfy', '?')} {e.get('time', '')}")
        return 0

    if getattr(args, "apply", False) and action != "repair":
        print("error: --apply needs --repair", file=sys.stderr)
        return 2
    if action == "repair":  # disk-only: no driver, no probe
        from . import gate_repair

        return gate_repair.repair(
            args.report_dir, apply=getattr(args, "apply", False))

    args.host = _resolve_driver_host(args.host)
    base = f"http://{args.host}"
    hist = json.load(urllib.request.urlopen(f"{base}/history?max_items=1", timeout=15))
    if not hist:
        print("nothing in ComfyUI history; render your workflow once, then re-run")
        return 2
    _, entry = next(iter(hist.items()))
    prompt = entry["prompt"][2]
    samplers = {"DGXMonarchKSampler", "DGXMonarchKSamplerAdvanced", "DGXMonarchKSamplerPipeline"}
    target = next((nid for nid, n in prompt.items()
                   if isinstance(n, dict) and n.get("class_type") in samplers), None)
    if target is None:
        print("the last prompt has no DGX Monarch sampler node to gate")
        return 2
    i = prompt[target]["inputs"]
    seed_input = i.get("noise_seed", i.get("seed", 42))
    run_id = uuid.uuid4().hex
    prompt[target] = {"class_type": "DGXMonarchIdentityGate", "inputs": {
        "model": i["model"], "positive": i["positive"], "negative": i["negative"],
        "latent_image": i["latent_image"],
        "noise_seed": _prompt_input(seed_input, 42, int),
        "steps": 2, "cfg": _prompt_input(i.get("cfg", 1.0), 1.0, float),
        "sampler_name": i.get("sampler_name", "euler"),
        "scheduler": i.get("scheduler", "simple"),
        "run_id": run_id,
    }}
    req = urllib.request.Request(f"{base}/prompt", data=json.dumps({"prompt": prompt}).encode(),
                                 headers={"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req, timeout=60))["prompt_id"]
    print(f"gating the last prompt (id {pid}) ...")
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        h = json.load(urllib.request.urlopen(f"{base}/history/{pid}", timeout=15))
        if h:
            ok = h[pid].get("status", {}).get("status_str") == "success"
            break
        time.sleep(3)
    else:
        print("gate timed out; check the driver log, or raise --timeout")
        return 2
    report = _find_gate_report(args.report_dir, run_id)
    if not ok or report is None:
        print("gate run failed; check the driver log")
        return 2
    verdict = report.get("verdict")
    print(json.dumps(report, indent=1))
    print(f"VERDICT: {verdict}")
    return {"PASS": 0, "FAIL": 1}.get(verdict if isinstance(verdict, str) else "", 3)


def cmd_gate(args) -> int:
    """Run `dgxm gate`; a driver or network failure, or a JSON, key, type or value
    error in a response, prints one error and returns 2."""
    import json
    import urllib.error

    if args.timeout <= 0:
        print("error: --timeout must be greater than zero", file=sys.stderr)
        return 2
    try:
        return _cmd_gate_impl(args)
    except (OSError, urllib.error.URLError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"gate could not reach ComfyUI, or could not read its response or the gate ledger: {exc}",
              file=sys.stderr)
        return 2


def _find_gate_report(report_dir: str, run_id: str) -> dict | None:
    """Find only the report produced by this CLI invocation.

    A shared JSONL file can contain concurrent node runs and old verdicts, so
    accepting its last row can report success for the wrong prompt.
    """
    import json

    try:
        lines = (Path(report_dir) / "dgxm_gate_reports.jsonl").read_text().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            report = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(report, dict) and report.get("run_id") == run_id:
            return report
    return None


def _prompt_input(value, default, cast):
    """Preserve a Comfy API link; coerce only literal widget values."""
    if isinstance(value, list) and len(value) == 2:
        return value
    return cast(default if value is None else value)


def _resolve_driver_host(host: str | None) -> str:
    """Return an explicit host as given; otherwise ask the shared schema-confirming local-driver probe."""
    if host:
        return host
    return comfy_ports.resolve_driver_host(host, doctor_mod._find_comfy_ports())


def _driver_probe(host: str | None = None) -> tuple[str | None, bool]:
    """The driver that has this node pack loaded, and True when a driver confirmed or every candidate answered.

    An explicit host is checked first, but never suppresses local discovery:
    otherwise ``dgxm update --host <stopped-instance>`` could miss a different
    live driver that will retain the old modules. The second value keeps a
    probe that never answered apart from one that answered "no driver", which
    is the difference between an idle box and an unknown one.
    """
    return comfy_ports.driver_probe(host, doctor_mod._find_comfy_ports())


def cmd_top(args) -> int:
    """Live cluster dashboard (read-only; HTTP + sysfs, never a mesh client)."""
    if not poll_interval_is_valid(args.interval):
        print("error: --interval must be between 0.1 and 3600 seconds", file=sys.stderr)
        return 2
    try:
        from ..tui.app import run

        # Textual loads only when run() builds the app, so a missing [tui] extra can
        # raise from this call as well as from the import above.
        run(driver=_resolve_driver_host(args.host), interval=args.interval,
            replay=args.replay, record=args.record, theme=args.theme,
            config_path=getattr(args, "config", None))
    except ImportError as exc:
        print(f"dgxm top needs the TUI extra ({exc}). Install it with the Python that runs ComfyUI: "
              "<python> -m pip install -e '<ComfyUI dir>/custom_nodes/dgx-monarch[tui]' (docs/INSTALL.md)")
        print("note: dgxm update installs with --no-deps, which keeps torch and NCCL in place "
              "and never adds this extra; install it once on each driver box")
        return 2
    except (OSError, ValueError) as exc:
        print(f"dgxm top could not start: {exc}", file=sys.stderr)
        return 2
    return 0


def cmd_install_service(args) -> int:
    return 0 if lifecycle.install_systemd(_require_config(args)) else 1


def cmd_update(args) -> int:
    if getattr(args, "verify", False):
        from .update_command import run_verified_update

        config = _require_config(args)
        repo = Path(__file__).resolve().parents[3]
        return run_verified_update(
            config,
            repo=repo,
            target_ref=args.target,
            assume_yes=bool(args.yes),
            driver_host=args.host,
            receipt_path=args.receipt,
        )
    if args.target != DEFAULT_TARGET or args.yes or args.receipt:
        print("error: --target, --yes, and --receipt require --verify", file=sys.stderr)
        return 2
    return legacy_update.run(args, load_config=_load_config, driver_probe=_driver_probe)


def cmd_uninstall(args) -> int:
    config = _load_config(args)
    acted = config is not None and bool(config.hosts)
    ok = True
    if config is not None and config.hosts and config.source:
        source = Path(config.source).expanduser().absolute()
        try:
            with SetupConfigLock(source):
                current = load_cluster_config(source)
                if current != config:
                    print("uninstall refused: the cluster config changed after it was loaded; nothing was removed",
                          file=sys.stderr)
                    return 1
                snapshot = read_snapshot(source)
                ok = _uninstall_cluster(config, source, snapshot)
        except (ClusterConfigError, OSError, SetupConfigLockUnavailable):
            print("uninstall refused: config serialization is unavailable; another config operation "
                  "may hold the lock, or the config could not be read again", file=sys.stderr)
            return 1
    elif config is not None and config.hosts:
        ok = _uninstall_cluster(config, None, None)
    if not acted:
        print("no cluster config was found; no worker services or units were changed.")
    elif ok:
        print("worker services and units removed. The node pack, venvs and models were left alone;")
        print("remove the custom_nodes/dgx-monarch checkout to finish uninstalling.")
    else:
        print(
            "uninstall incomplete: a worker service, unit or the config file could not be removed; "
            "review the errors above and retry.",
            file=sys.stderr,
        )
    return 0 if ok else 1


def _uninstall_cluster(config, source, snapshot) -> bool:
    """Refuse a foreign unit first, then remove services and the bound config.

    The ownership refusal has to come before `down`, or the worker is already
    stopped and the generation record already cleared when it fires.
    """
    if not lifecycle.units_are_removable(config):
        return False
    if lifecycle.down(config) is not True:
        return False
    ok = lifecycle.uninstall_systemd(config)
    if ok and source is not None and snapshot is not None:
        if input(f"remove {source}? [y/N] ").strip().lower() == "y" and not remove_exact_config(snapshot):
            print("could not remove the config: it changed after it was read, or its removal failed "
                  "or could not be verified", file=sys.stderr)
            return False
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = arguments.build_parser({
        "version": cmd_version,
        "doctor": cmd_doctor,
        "init": cmd_init,
        "setup": cmd_setup,
        "up": cmd_up,
        "down": cmd_down,
        "reap": cmd_reap,
        "top": cmd_top,
        "gate": cmd_gate,
        "restart": cmd_restart,
        "status": cmd_status,
        "install_service": cmd_install_service,
        "update": cmd_update,
        "uninstall": cmd_uninstall,
    })
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
