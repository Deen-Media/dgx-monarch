"""Typed config, guided setup, launcher and owned uninstall adapters."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ..cli import launcher, lifecycle
from ..cli.config_removal import remove_exact_config
from ..cli.setup_command import build_setup_plan, run_setup
from ..cli.setup_config_io import apply_config, private_roundtrip, read_snapshot, rollback_config
from ..cli.setup_config_lock import SetupConfigLock
from ..cli.setup_config_transaction import ConfigTransaction
from ..cli.setup_models import CandidateHost, SetupRequest
from ..config import load_cluster_config, render_cluster_toml
from .contracts import StudioError, digest


def absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise StudioError("operation_path", "Management paths must be absolute.")
    return path.absolute()


def snapshot(path: Path) -> dict[str, Any]:
    value = read_snapshot(path)
    return {"digest": value.digest, "mode": value.mode, "identity": list(value.identity) if value.identity else None,
            "existed": value.existed}


def hosts(values: list[dict[str, Any]]) -> tuple[CandidateHost, ...]:
    allowed = {"name", "fabric_ip", "gpus", "ssh_user", "comfy_dir"}
    if any(set(value) - allowed for value in values):
        raise StudioError("operation_fields", "Unsupported host parameter.")
    try:
        return tuple(CandidateHost(**value) for value in values)
    except (TypeError, ValueError) as exc:
        raise StudioError("operation_fields", "Each explicit host requires a name, fabric_ip and positive GPU count.") from exc


def setup_request(path: Path, parameters: dict[str, Any]) -> SetupRequest:
    values = {**{key: value for key, value in parameters.items() if key not in {"output", "receipt"}}, "hosts": hosts(parameters["hosts"]), "artifacts": tuple(parameters["artifacts"]),
              "fabric_profile": parameters["fabric_profile"] or None}
    if parameters["verify"] and not (parameters["install_service"] and parameters["start_workers"]):
        raise StudioError("operation_fields", "Setup verification requires explicit paired install and start authority.")
    return SetupRequest(config_path=path, receipt_path=absolute(parameters["receipt"]) if parameters.get("receipt") else None, **values)


def setup_view(plan: Any) -> dict[str, Any]:
    original = plan.snapshot
    return {**plan.as_dict(), "config_text": plan.config_text, "diff": plan.diff,
            "snapshot": {"digest": original.digest, "mode": original.mode,
                         "identity": list(original.identity) if original.identity else None, "existed": original.existed},
            "request_binding": plan.request_binding}


def file_identity(path: Path) -> dict[str, Any]:
    """Bind an existing launcher target or destination without executing it."""
    if not path.exists() and not path.is_symlink():
        return {"exists": False}
    import os
    import stat

    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > 1024 * 1024:
            raise StudioError("operation_path", "Launcher files must be bounded regular files.")
        data = os.read(descriptor, 1024 * 1024 + 1)
        after = os.fstat(descriptor)
        def identity(item: Any) -> list[int]:
            return [item.st_dev, item.st_ino, item.st_mode, item.st_size, item.st_mtime_ns, item.st_ctime_ns]
        if identity(before) != identity(after) or identity(after) != identity(path.stat()):
            raise StudioError("operation_changed", "Launcher file changed during review.", 409)
        return {"exists": True, "resolved": str(path.resolve()), "identity": identity(after), "digest": digest(data.hex())}
    finally:
        os.close(descriptor)


def preview(action: str, path: Path, parameters: dict[str, Any]) -> dict[str, Any]:
    if action == "setup":
        return setup_view(build_setup_plan(setup_request(path, parameters)))
    if action == "init":
        selected = hosts(parameters["hosts"])
        if any(host.ssh_user or host.comfy_dir for host in selected):
            raise StudioError("operation_fields", "Use guided setup for per-host SSH users or ComfyUI directories.")
        text = render_cluster_toml([(h.name, h.fabric_ip, h.gpus) for h in selected], parameters["client_ip"],
                                   parameters["fabric_profile"] or "generic-roce", selected[0].fabric_ip,
                                   ssh_key=parameters["ssh_key"], python_bin=parameters["python_bin"],
                                   transport_security=parameters["transport_security"])
        private_roundtrip(text)
        return {"config_text": text, "snapshot": snapshot(path), "config_digest": digest(text),
                "scope": "Write configuration only; no Worker service is installed or started."}
    if action == "launcher":
        executable, destination = absolute(parameters["executable"]), absolute(parameters["destination"])
        status = launcher.install(str(executable), str(path), str(destination), apply=False)
        return {"status": status, "target": file_identity(executable), "destination": file_identity(destination)}
    config = load_cluster_config(path)
    removable = lifecycle.units_are_removable(config)
    return {"removable": removable is True, "hosts": len(config.hosts), "snapshot": snapshot(path),
            "scope": "Remove owned Worker services and units; preserve node pack, Python environments and models."}


def execute(action: str, path: Path, parameters: dict[str, Any], reviewed: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if action == "setup":
        request = setup_request(path, parameters)
        plan = build_setup_plan(request)
        if setup_view(plan) != reviewed:
            raise StudioError("operation_changed", "Setup probes or the exact config plan changed; review again.", 409)
        outcome = run_setup(replace(request, apply=True, assume_yes=True), plan=plan)
        return ("succeeded" if outcome.status == "succeeded" else "failed" if outcome.status == "failed" else "unknown",
                {"source": "guided_setup", **outcome.as_dict()})
    if action == "launcher":
        if preview(action, path, parameters) != reviewed:
            raise StudioError("operation_changed", "Launcher identity changed after review.", 409)
        value = launcher.install(parameters["executable"], str(path), parameters["destination"], apply=True)
        return "succeeded", {"source": "owned_launcher", "result": value}
    with SetupConfigLock(path):
        if snapshot(path) != reviewed["snapshot"]:
            raise StudioError("config_changed", "Configuration identity changed after review.", 409)
        if action == "init":
            transaction = ConfigTransaction(path, reviewed["config_text"], read_snapshot(path), apply_config)
            try:
                mutation = transaction.publish()
            except BaseException:
                recovered = transaction.recover_mutation()
                if recovered is not None:
                    rollback_config(recovered)
                raise
            if not mutation.durable:
                restored = rollback_config(mutation)
                return "failed" if restored else "unknown", {"source": "config_transaction", "rollback_confirmed": restored}
            return "succeeded", {"source": "config_transaction", "config_digest": mutation.new_digest,
                                 "changed": mutation.changed, "durable": True}
        config = load_cluster_config(path)
        if not reviewed["removable"] or not lifecycle.units_are_removable(config):
            raise StudioError("uninstall_ownership", "Worker unit ownership cannot be established.", 409)
        original = read_snapshot(path)
        if lifecycle.down(config) is not True:
            return "unknown", {"source": "owned_uninstall", "step": "worker_stop", "config_retained": True}
        if lifecycle.uninstall_systemd(config) is not True:
            return "unknown", {"source": "owned_uninstall", "step": "unit_removal", "config_retained": True}
        removed = remove_exact_config(original) if parameters["remove_config"] else False
        return ("unknown" if parameters["remove_config"] and not removed else "succeeded",
                {"source": "owned_uninstall", "services_removed": True, "config_removed": removed})
