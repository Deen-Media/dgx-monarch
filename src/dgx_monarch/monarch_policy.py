"""Validated process policy for the pinned Monarch runtime.

Environment variables have highest precedence in Monarch 0.7. Install the
resolved policy before importing the runtime, then verify its native readback.
Diagnostic export settings are deliberately outside the transport identity.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import sys
import threading
from dataclasses import asdict, dataclass, field, replace
from typing import Any
from urllib.parse import urlsplit

from .config_schema import ClusterConfigError

_DEVICE = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
POLICY_ENV = "DGXM_MONARCH_POLICY"
_LOCK = threading.RLock()
_ACTIVE: dict[str, Any] | None = None
# Fixed release defaults outside the supported tuning surface. Include them
# in installation/readback so ambient native knobs cannot bypass this policy.
_FIXED_RDMA = {
    "rdma_disable_ibverbs": False, "rdma_host_odp": False,
    "rdma_max_chunk_size_mb": 64, "rdma_tcp_fallback_parallelism": 1,
    "rdma_qps_per_cq": 64, "rdma_cq_poller_per_device": True,
    "rdma_runtime_worker_threads": 4,
}


@dataclass(frozen=True)
class RdmaPolicy:
    target: str = ""
    host_targets: tuple[tuple[str, str], ...] = ()
    peer_device_affinity: str = "match_name"
    max_nics_per_buffer: int = 1
    qps_per_peer: int = 1
    min_stripe_size_kb: int = 512
    application_chunks: int = 1
    allow_tcp_fallback: bool = False
    qp_init_timeout_s: int = 30

    def resolved(self, host: str = "") -> RdmaPolicy:
        return replace(self, target=dict(self.host_targets).get(host, self.target), host_targets=())


@dataclass(frozen=True)
class DiagnosticsPolicy:
    metrics: bool = False
    otlp_endpoint: str = ""
    otlp_logs: bool = False


@dataclass(frozen=True)
class MonarchPolicy:
    startup_timeout_s: int = 180
    rdma: RdmaPolicy = field(default_factory=RdmaPolicy)
    diagnostics: DiagnosticsPolicy = field(default_factory=DiagnosticsPolicy)


def _table(value: object, keys: set[str], context: str) -> dict:
    if not isinstance(value, dict):
        raise ClusterConfigError(f"{context} must be a table")
    unknown = set(value) - keys
    if unknown:
        raise ClusterConfigError(f"{context}: unsupported keys {sorted(map(str, unknown))}")
    return value


def _integer(value: object, low: int, high: int, context: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ClusterConfigError(f"{context} must be an integer in {low}..{high}")
    return value


def _target(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise ClusterConfigError(f"{context} must be a target string")
    if value == "":
        return value
    kind, _, name = value.partition(":")
    if (kind == "nic" and _DEVICE.fullmatch(name)) or (
        kind in {"cpu", "gpu"} and name.isascii() and name.isdecimal() and int(name) <= 255
    ):
        return value
    raise ClusterConfigError(f"{context} must be nic:<device>, cpu:<numa>, gpu:<ordinal>, or empty")


def _host_target(value: object) -> str:
    """A per-host override names that host's NIC; empty, cpu and gpu are refused here."""
    target = _target(value, "monarch.rdma.host_targets")
    if not target.startswith("nic:"):
        raise ClusterConfigError("monarch.rdma.host_targets values must be nic:<device>; "
                                 "omit a host to use [monarch.rdma] target")
    return target


def _affinity(value: object) -> str:
    if value in ("any", "match_name"):
        return str(value)
    if isinstance(value, str) and value.startswith("groups:") and len(value) <= 2048:
        names = [name for group in value[7:].split("|") for name in group.split(",")]
        if all(_DEVICE.fullmatch(name) for name in names) and len(set(names)) == len(names):
            return value
    raise ClusterConfigError("monarch.rdma.peer_device_affinity must be any, match_name, or disjoint groups:<devices>")


def parse_policy(raw: object, *, host_names: tuple[str, ...] | None = None) -> MonarchPolicy:
    values = _table(raw, {"startup_timeout_s", "rdma", "diagnostics"}, "monarch")
    rdma = _table(values.get("rdma", {}), set(RdmaPolicy.__dataclass_fields__), "monarch.rdma")
    defaults = RdmaPolicy()
    targets = rdma.get("host_targets", {})
    if not isinstance(targets, dict) or len(targets) > 64:
        raise ClusterConfigError("monarch.rdma.host_targets must be a table of at most 64 hosts")
    if any(not isinstance(name, str) or not name or len(name) > 255 for name in targets):
        raise ClusterConfigError("monarch.rdma.host_targets has an invalid host name")
    if host_names is not None and set(targets) - set(host_names):
        raise ClusterConfigError("monarch.rdma.host_targets names an unconfigured host")
    limits = {"max_nics_per_buffer": (1, 8), "qps_per_peer": (1, 8),
              "min_stripe_size_kb": (4, 65536), "application_chunks": (1, 8),
              "qp_init_timeout_s": (1, 300)}
    numbers = {key: _integer(rdma.get(key, getattr(defaults, key)), low, high,
                            f"monarch.rdma.{key}") for key, (low, high) in limits.items()}
    # 0.7 pairs NICs one-to-one; one device cannot appear in two pairs.
    if math.prod(numbers[key] for key in ("max_nics_per_buffer", "qps_per_peer", "application_chunks")) > 32:
        raise ClusterConfigError("Monarch RDMA NICs * QPs * application chunks must not exceed 32")
    fallback = rdma.get("allow_tcp_fallback", False)
    if type(fallback) is not bool:
        raise ClusterConfigError("monarch.rdma.allow_tcp_fallback must be a boolean")
    transport = RdmaPolicy(
        target=_target(rdma.get("target", ""), "monarch.rdma.target"),
        host_targets=tuple(sorted((name, _host_target(value)) for name, value in targets.items())),
        peer_device_affinity=_affinity(rdma.get("peer_device_affinity", defaults.peer_device_affinity)),
        allow_tcp_fallback=fallback, **numbers,
    )
    diag = _table(values.get("diagnostics", {}), {"metrics", "otlp_endpoint", "otlp_logs"}, "monarch.diagnostics")
    metrics, endpoint = diag.get("metrics", False), diag.get("otlp_endpoint", "")
    if type(metrics) is not bool:
        raise ClusterConfigError("monarch.diagnostics.metrics must be a boolean")
    if not isinstance(endpoint, str) or len(endpoint) > 2048:
        raise ClusterConfigError("monarch.diagnostics.otlp_endpoint must be a URL")
    logs = diag.get("otlp_logs", False)
    if type(logs) is not bool:
        raise ClusterConfigError("monarch.diagnostics.otlp_logs must be a boolean")
    if endpoint and not logs:
        raise ClusterConfigError("Monarch 0.7 OTLP also exports upstream logs; set monarch.diagnostics.otlp_logs=true explicitly")
    if endpoint:
        try:
            url = urlsplit(endpoint)
            valid = (url.scheme in {"http", "https"} and url.hostname and not url.username
                     and not url.password and not url.query and not url.fragment and url.port != 0)
        except ValueError:
            valid = False
        if not valid or any(ord(char) < 33 for char in endpoint):
            raise ClusterConfigError("monarch.diagnostics.otlp_endpoint must be an HTTP(S) URL without credentials/query/fragment")
    return MonarchPolicy(
        startup_timeout_s=_integer(values.get("startup_timeout_s", 180), 30, 3600, "monarch.startup_timeout_s"),
        rdma=transport, diagnostics=DiagnosticsPolicy(metrics, endpoint, logs),
    )


def policy_dict(policy: MonarchPolicy) -> dict[str, Any]:
    payload = asdict(policy)
    payload["rdma"]["host_targets"] = dict(policy.rdma.host_targets)
    return payload


def transport_identity(policy: MonarchPolicy) -> str:
    """Hash the RDMA block only; the readiness budget and export settings are not transport."""
    payload = policy_dict(policy)["rdma"]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def admit_native_targets(policy: MonarchPolicy, participants: tuple[str, ...], *,
                         unnamed: str = "the driver") -> None:
    """Refuse native return unless every participant resolves to a named NIC.

    ``participants`` holds configured host names; "" stands for a process with
    no host entry, which uses the global target. Driver admission passes every
    host before transport commit; each process also checks itself.
    """
    for name in dict.fromkeys(participants):
        target = policy.rdma.resolved(name).target
        if not target.startswith("nic:"):
            who = f"host {name!r}" if name else unnamed
            raise ClusterConfigError(
                f"native RDMA requires an explicit selected target for every participant; {who} "
                f"resolves to {target or 'upstream selection'}. Set [monarch.rdma] target, or that "
                "host's host_targets entry, to nic:<device>")


def encode_policy(policy: MonarchPolicy, host: str = "") -> str:
    payload = policy_dict(replace(policy, rdma=policy.rdma.resolved(host)))
    return base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).decode()


def policy_from_token(token: str) -> MonarchPolicy:
    if not isinstance(token, str) or len(token) > 16384:
        raise ClusterConfigError("invalid Monarch policy token")
    try:
        payload = json.loads(base64.b64decode(token, altchars=b"-_", validate=True))
    except (ValueError, UnicodeError) as exc:
        raise ClusterConfigError("invalid Monarch policy token") from exc
    return parse_policy(payload)


def policy_environment(policy: MonarchPolicy, host: str = "") -> dict[str, str]:
    rdma = policy.rdma.resolved(host)
    fields = asdict(rdma)
    fields.pop("host_targets")
    chunks = fields.pop("application_chunks")
    fields["ibverbs_target"] = fields.pop("target")
    fields["qp_init_timeout"] = f"{fields.pop('qp_init_timeout_s')}s"
    result = {f"MONARCH_RDMA_{key.upper()}": str(value).lower() if isinstance(value, bool) else str(value)
              for key, value in fields.items()}
    result.update({f"MONARCH_{key.upper()}": str(value).lower() if isinstance(value, bool) else str(value)
                   for key, value in _FIXED_RDMA.items()})
    result.update({POLICY_ENV: encode_policy(policy, host), "DGXM_RDMA_QP_SPLIT": str(chunks)})
    if policy.diagnostics.otlp_endpoint:
        if not policy.diagnostics.otlp_logs:
            raise ClusterConfigError("OTLP export also sends upstream logs; set otlp_logs=true explicitly")
        result["OTEL_EXPORTER_OTLP_ENDPOINT"] = policy.diagnostics.otlp_endpoint
    return result


def apply_policy(policy: MonarchPolicy, host: str = "", *, role: str = "driver",
                 native_enabled: bool = False) -> None:
    """Install once per process; a transport change requires a fresh process.

    These variables outrank Runtime/File/ClientOverride in the pinned runtime.
    Conflicting ambient settings are refused instead of silently ignored.
    """
    global _ACTIVE
    resolved = replace(policy, rdma=policy.rdma.resolved(host))
    parse_policy(policy_dict(resolved))  # programmatic constructors obey the same contract
    if native_enabled:  # whatever max_nics_per_buffer says: no unqualified auto-selection
        admit_native_targets(policy, (host,), unnamed=f"this {role} process")
    identity = transport_identity(resolved)
    environment = policy_environment(resolved)
    with _LOCK:
        if _ACTIVE is not None and _ACTIVE["transport_sha256"] != identity:
            raise RuntimeError("Monarch transport policy changed; restart this process before attaching")
        if (os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
                and not policy.diagnostics.otlp_endpoint):
            raise ClusterConfigError("ambient OTLP endpoint conflicts with disabled diagnostic export")
        if any(key.startswith("MONARCH_RDMA_") and key not in environment for key in os.environ):
            raise ClusterConfigError("unmanaged MONARCH_RDMA_ environment setting; use the typed policy")
        for key, value in environment.items():
            if key == POLICY_ENV:
                continue
            if key in os.environ and os.environ[key] != value:
                raise ClusterConfigError(f"{key} conflicts with the validated Monarch policy")
        # The exporter is initialized at native import. configure_policy must
        # see the original environment to detect an already imported runtime.
        os.environ.update({key: value for key, value in environment.items()
                           if key != "OTEL_EXPORTER_OTLP_ENDPOINT"})
        activated = bool(_ACTIVE and _ACTIVE.get("native_activated"))
        _ACTIVE = {"schema": 1, "role": role, "transport_sha256": identity,
                   "policy": policy_dict(resolved), "native_readback": "pending",
                   "native_activated": activated}


def _native_settings(policy: MonarchPolicy) -> dict[str, Any]:
    expected = asdict(policy.rdma)
    expected.pop("host_targets")
    expected.pop("application_chunks")
    expected["ibverbs_target"] = expected.pop("target")
    expected["qp_init_timeout"] = f"{expected.pop('qp_init_timeout_s')}s"
    return {**_FIXED_RDMA, **{f"rdma_{key}": value for key, value in expected.items()}}


def application_chunks() -> int | None:
    """The admitted chunk count, never a mutable environment override."""
    with _LOCK:
        return None if _ACTIVE is None else int(_ACTIVE["policy"]["rdma"]["application_chunks"])


def activate_native_policy() -> dict[str, Any]:
    """Set the Runtime layer before this application's first RDMA use.

    Comfy discovery may import Monarch before it knows the Init node's config.
    Its native Env layer is captured at import, so changing os.environ alone
    cannot configure that driver. An earlier conflicting Env value still wins
    over Runtime and the mandatory readback refuses it. Child interpreters also
    receive the same policy before startup from the launch command.
    """
    from monarch.config import configure

    with _LOCK:
        if _ACTIVE is None:
            raise RuntimeError("Monarch process policy has not been installed")
        if not _ACTIVE.get("native_activated"):
            public_rdma = sys.modules.get("monarch._src.rdma.rdma")
            if getattr(public_rdma, "_rdma_manager_init_cache", None):
                raise RuntimeError("Monarch RDMA was initialized before policy admission; restart this process")
        configure(**_native_settings(parse_policy(_ACTIVE["policy"])))
        result = verify_native_policy()
        _ACTIVE["native_activated"] = True
        result["native_activated"] = True
        return result


def verify_native_policy() -> dict[str, Any]:
    """Read the effective native config; a mismatch cannot authorize a transfer."""
    from monarch.config import get_global_config

    with _LOCK:
        if _ACTIVE is None:
            raise RuntimeError("Monarch process policy has not been installed")
        policy = parse_policy(_ACTIVE["policy"])
        actual = get_global_config()
        for key, value in _native_settings(policy).items():
            if actual.get(key) != value:
                raise RuntimeError(f"Monarch effective {key} differs from validated policy")
        _ACTIVE["native_readback"] = "verified"
        _ACTIVE["readback_scope"] = "process_configuration"
        return snapshot()


def snapshot() -> dict[str, Any]:
    with _LOCK:
        if _ACTIVE is None:
            return {"schema": 1, "available": False}
        result = {key: value for key, value in _ACTIVE.items() if key != "policy"}
        result["available"] = True
        result["rdma"] = dict(_ACTIVE["policy"]["rdma"])
        result["rdma"].pop("host_targets", None)
        result["metrics_requested"] = _ACTIVE["policy"]["diagnostics"]["metrics"]
        return result
