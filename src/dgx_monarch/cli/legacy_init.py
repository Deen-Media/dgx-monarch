"""Compatibility implementation for the original interactive ``dgxm init``."""
from __future__ import annotations

import os
from pathlib import Path

from ..config import (
    FABRIC_PROFILES,
    is_literal_ip,
    load_cluster_config,
    render_cluster_toml,
)
from ..constants import DEFAULT_NCCL_MASTER_PORT, DEFAULT_WORKER_PORT
from . import lifecycle
from .setup_config_io import apply_config, read_snapshot, rollback_config
from .setup_config_lock import SetupConfigLock, SetupConfigLockUnavailable
from .setup_config_transaction import ConfigTransaction


def run(args: object) -> int:
    """Generate the legacy config; new installations should use ``setup``."""
    print("dgx-monarch cluster init. Enter hosts in rank order; the first host is the NCCL master.")
    hosts: list[tuple[str, str, int]] = []
    while True:
        name = input(f"host {len(hosts)} ssh name/IP (empty to finish): ").strip()
        if not name:
            if hosts:
                break
            print("enter at least one host")
            continue
        while True:
            fabric_ip = input(f"  fabric IP for {name} (worker bind + NCCL): ").strip() or name
            if is_literal_ip(fabric_ip):
                break
            print(
                f"  {fabric_ip!r} is not a literal IP; worker bind addresses must be "
                "fabric IPs, never hostnames (docs/CLUSTER.md)"
            )
        while True:
            gpus_raw = input("  gpus [1]: ").strip() or "1"
            try:
                gpus = int(gpus_raw)
                if gpus <= 0:
                    raise ValueError
                break
            except ValueError:
                print(f"  {gpus_raw!r} is not a positive GPU count (e.g. 1)")
        hosts.append((name, fabric_ip, gpus))

    profiles = ", ".join(FABRIC_PROFILES)
    while True:
        profile = input(f"fabric profile [{profiles}] (default generic-roce): ").strip() or "generic-roce"
        if profile in FABRIC_PROFILES:
            break
        print(f"  unknown profile {profile!r}; choose one of: {profiles}")
    while True:
        client_ip = input("driver fabric IP (client_bind; empty = first host's IP): ").strip() or hosts[0][1]
        if is_literal_ip(client_ip):
            break
        print(f"  {client_ip!r} is not a literal IP (the client advertises this address)")
    python_bin = input("worker python (venv with torchmonarch+comfy deps) [python3]: ").strip() or "python3"
    ssh_key = input("ssh key path (empty = default agent/keys): ").strip()
    print("worker transports have no peer authentication: a client that reaches the fabric can "
          "execute actor code (SECURITY.md).")
    acknowledgement = input(
        "after source-restricting all traffic on the dedicated fabric to trusted "
        "driver/cluster peers, type trusted_fabric: "
    ).strip()
    if acknowledgement != "trusted_fabric":
        print("aborted: network isolation was not acknowledged")
        return 2

    text = render_cluster_toml(
        hosts=hosts,
        client_ip=client_ip,
        fabric_profile=profile,
        master_addr=hosts[0][1],
        master_port=DEFAULT_NCCL_MASTER_PORT,
        ssh_key=ssh_key,
        worker_port=DEFAULT_WORKER_PORT,
        python_bin=python_bin,
        transport_security=acknowledgement,
    )
    config_arg = getattr(args, "config", None)
    selected = Path(config_arg).expanduser() if config_arg else Path.home() / ".config" / "dgx-monarch" / "cluster.toml"
    dest = Path(os.path.abspath(os.fspath(selected)))
    try:
        snapshot = read_snapshot(dest)
    except OSError:
        print("aborted: the config destination could not be inspected safely")
        return 1
    if snapshot.existed and input(f"{dest} exists. Overwrite? [y/N] ").strip().lower() != "y":
        print("aborted")
        return 1
    transaction = ConfigTransaction(dest, text, snapshot, apply_config)
    try:
        with SetupConfigLock(dest):
            try:
                mutation = transaction.publish()
            except BaseException:
                recovered = transaction.recover_mutation()
                if recovered is not None:
                    rollback_config(recovered)
                raise
            if not mutation.durable:
                if not rollback_config(mutation):
                    print("aborted: config durability was not confirmed and rollback is uncertain")
                else:
                    print("aborted: config durability was not confirmed")
                return 1
            config = load_cluster_config(dest)
            print(f"wrote {dest}")

            lifecycle_ok = True
            installed_service = False
            if input("install and start the worker services as systemd user units now? [y/N] ").strip().lower() == "y":
                lifecycle_ok = lifecycle.install_systemd(config)
                installed_service = lifecycle_ok
            elif input(
                "start the worker service now (through its systemd unit if one exists, else nohup)? [Y/n] "
            ).strip().lower() in ("", "y"):
                lifecycle_ok = lifecycle.up(config) is True
            if not lifecycle_ok:
                print("worker setup failed; fix the error above, then run `dgxm install-service` or `dgxm up`")
                return 1
    except SetupConfigLockUnavailable:
        print("aborted: another config operation is active")
        return 1
    except OSError:
        print("aborted: the config changed or could not be published safely")
        return 1
    if installed_service:
        print("next: dgxm doctor && dgxm status (the systemd worker units are already active)")
    else:
        print("next: dgxm up && dgxm doctor && dgxm status")
    return 0
