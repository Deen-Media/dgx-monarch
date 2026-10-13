"""Small configuration readers that do not import Monarch or Torch."""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config_schema import ClusterConfigError, validate_worker_args
from .monarch_policy import MonarchPolicy, parse_policy


def local_settings(path: Path | None) -> tuple[dict[str, object], MonarchPolicy]:
    """Read local actor knobs once; malformed supplied tables fail visibly."""
    if path is None:
        return {}, MonarchPolicy()
    try:
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ClusterConfigError(f"{path}: invalid TOML: {exc}") from exc
    except OSError:
        return {}, MonarchPolicy()
    args = validate_worker_args(raw.get("worker_args", {}), context=f"{path}: worker_args")
    return dict(args), parse_policy(raw.get("monarch", {}))


# Fields that cannot change math, residency or transport. Every other field is
# hashed, so a new config field invalidates trust until someone classifies it.
UNTRUSTED_FIELDS = frozenset({"auto_heal", "ssh_key", "source"})
UNTRUSTED_HOST_FIELDS = frozenset({"ssh_user"})
UNTRUSTED_MONARCH_FIELDS = frozenset({"startup_timeout_s", "diagnostics"})
_UNMATCHED_DRIVER_WARNED = False


def trust_config_fingerprint(config, fallback: str) -> str:
    """Hash the cluster settings that can change math, residency or transport.

    The Gate capability context binds this value (DESIGN.md section 5.2). Out
    of it: lifecycle-only keys (auto_heal, ssh_key, hosts[].ssh_user), the
    readiness budget (monarch.startup_timeout_s), export settings
    (monarch.diagnostics) and the config path, which the context binds
    separately as config_source. Editing those keeps persisted PASS rows.
    """
    from .config import ClusterConfig

    if not isinstance(config, ClusterConfig):
        return fallback
    full = asdict(config)
    payload = {name: value for name, value in full.items() if name not in UNTRUSTED_FIELDS}
    payload["hosts"] = [{name: value for name, value in host.items() if name not in UNTRUSTED_HOST_FIELDS}
                        for host in full["hosts"]]
    payload["monarch"] = {name: value for name, value in full["monarch"].items()
                          if name not in UNTRUSTED_MONARCH_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def driver_host(config) -> str:
    """The [[hosts]] name whose worker address shares client_bind's IP, or "".

    The production driver and the RDMA qualifier both resolve the driver's
    per-host RDMA target through this name. A driver with no matching entry
    (no client_bind, or a driver box without a Worker) resolves as "" and uses
    the global [monarch.rdma] target; one WARNING per process says so. Several
    matching entries must resolve to the same target.
    """
    global _UNMATCHED_DRIVER_WARNED
    import ipaddress

    from .config import tcp_endpoint

    def address(url: Any) -> Any:
        try:
            return ipaddress.ip_address(tcp_endpoint(url)[0])
        except ValueError:
            return None

    hosts = tuple(getattr(config, "hosts", ()) or ())
    if not hosts:
        return ""
    bind = address(getattr(config, "client_bind", ""))
    matches = [host.name for host in hosts
               if bind is not None and address(getattr(host, "address", "")) == bind]
    if not matches:
        if not _UNMATCHED_DRIVER_WARNED:
            _UNMATCHED_DRIVER_WARNED = True
            from .log import get_logger

            get_logger(__name__).warning(
                "cluster.client_bind matches no [[hosts]] address, so the driver uses the global "
                "[monarch.rdma] target rather than a host_targets entry")
        return ""
    policy = getattr(config, "monarch", MonarchPolicy())
    if len({policy.rdma.resolved(name).target for name in matches}) > 1:
        raise ClusterConfigError("cluster.client_bind matches several [[hosts]] entries whose "
                                 "[monarch.rdma] host_targets differ; the driver's NIC is ambiguous")
    return matches[0]
