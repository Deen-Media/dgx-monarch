"""Typed validation primitives shared by config loading and actor endpoints."""
from __future__ import annotations

import ipaddress
import math
import re
import socket
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Literal, TypedDict, cast
from urllib.parse import urlsplit


class ClusterConfigError(ValueError):
    """cluster.toml is missing, malformed, or violates a DESIGN invariant."""


MAX_CLUSTER_HOSTS = 64
MAX_GPUS_PER_HOST = 64
MAX_WORLD_SIZE = 256

# Every reader admits [studio]; only Studio's own readers look inside it
# (validate_studio_table), so a typo there never stops a render.
_ROOT_KEYS = frozenset({"cluster", "fabric", "hosts", "worker_args", "monarch", "studio"})
_CLUSTER_KEYS = frozenset({
    "auto_heal",
    "client_bind",
    "comfy_dir",
    "fabric_profile",
    "nccl_master_addr",
    "nccl_master_port",
    "python",
    "rdma_latent_return",
    "rdma_min_bytes",
    "ssh_key",
    "transport_security",
})
_HOST_KEYS = frozenset({"address", "comfy_dir", "gpus", "name", "ssh_user"})


def _reject_unknown_keys(
    values: dict, allowed: frozenset[str], *, context: str,
) -> None:
    """Fail before a typo can silently select a default or the wrong table."""
    unknown = sorted(str(key) for key in values if key not in allowed)
    if unknown:
        raise ClusterConfigError(
            f"{context} contains unsupported key(s): {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(sorted(allowed))}")


def validate_root_keys(values: dict, path: Path) -> None:
    _reject_unknown_keys(
        values, _ROOT_KEYS, context=f"{path}: cluster.toml top level")


def validate_cluster_keys(values: dict, path: Path) -> None:
    _reject_unknown_keys(values, _CLUSTER_KEYS, context=f"{path}: [cluster]")


def validate_fabric_profile_name(value: object, host_count: int, path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ClusterConfigError(
            f"{path}: cluster.fabric_profile must be a non-empty string")
    if host_count > 1 and value == "single-node":
        raise ClusterConfigError(
            f"{path}: fabric_profile 'single-node' is invalid for {host_count} hosts; "
            "select a network fabric profile explicitly")
    return value


def validate_host_keys(values: dict, path: Path, index: int) -> None:
    _reject_unknown_keys(
        values, _HOST_KEYS, context=f"{path}: [[hosts]] entry {index}")


class WorkerArgs(TypedDict, total=False):
    """The complete worker policy accepted from config; keep it small."""

    reserve_vram_gb: float
    uma_reserve_gb: float
    disable_pinned_memory: bool
    disable_async_offload: bool
    disable_smart_memory: bool
    disable_custom_nodes: bool
    mmap_fallback: bool
    lora_low_rss: bool
    slab_weights: bool | Literal["auto"]
    comfy_managed: bool
    load_profile: bool
    compile_dit: bool
    # Off switch and first-step trace of the nvfp4 shared activation scale. Set
    # the switch here, not by exporting DGXM_SHARED_ACT_SCALE by hand;
    # adapters/quant_activation_scale.py says why.
    shared_act_scale: bool
    log_act_scale: bool
    swap_verify: int
    fsdp_prefetch_depth: int
    safetensors_backend: Literal["pread", "mmap"]
    # The kernel for comfy-kitchen's INT8 ConvRot GEMM inside the actors;
    # absent means "triton". The driver reads DGXM_INT8_GEMM instead
    # (kernels/int8_gemm.py).
    int8_gemm: Literal["triton", "cutlass"]
    # Worker-assisted VAE decode (vae_decode_route.py); absent means "off".
    # The driver alone reads it: DRIVER_ONLY_KEYS keeps it out of the actors'
    # policy and the Gate's contexts, since a decode changes no DiT math.
    worker_vae_decode: Literal["off", "auto"]


_BOOL_KEYS = frozenset({
    "disable_pinned_memory", "disable_async_offload", "disable_smart_memory",
    "disable_custom_nodes", "mmap_fallback", "lora_low_rss", "load_profile",
    "compile_dit", "comfy_managed", "shared_act_scale", "log_act_scale",
})
_FLOAT_LIMITS = {"reserve_vram_gb": (0.0, 1024.0), "uma_reserve_gb": (0.0, 1024.0)}
_KEYS = frozenset({*_BOOL_KEYS, *_FLOAT_LIMITS, "slab_weights", "swap_verify",
                   "safetensors_backend", "fsdp_prefetch_depth", "int8_gemm",
                   "worker_vae_decode"})
# Keys only the driver reads. They never reach an actor, so they stay out of
# the published worker policy, its key and every Gate capability context
# (actor_worker_args), and out of the trust fingerprint (monarch_policy_io).
DRIVER_ONLY_KEYS = frozenset({"worker_vae_decode"})


def actor_worker_args(values: Mapping[str, object] | None) -> dict[str, object]:
    """The config policy an actor takes: every key but the driver-only ones."""
    return {key: value for key, value in (values or {}).items() if key not in DRIVER_ONLY_KEYS}


def validate_worker_args(values: object, *, context: str = "worker_args",
                         reject_unknown: bool = True) -> WorkerArgs:
    """Validate policy without truthiness coercion (``bool("off")`` is true)."""
    if not isinstance(values, dict):
        raise ClusterConfigError(
            f"{context} must be a table/mapping (got {type(values).__name__})")
    unknown = sorted(str(key) for key in values if key not in _KEYS)
    if unknown and reject_unknown:
        raise ClusterConfigError(
            f"{context} contains unsupported key(s): {', '.join(unknown)}. "
            f"Allowed keys: {', '.join(sorted(_KEYS))}")
    # With reject_unknown=False (the actor-side call in actor/comfy_bridge.py)
    # keys outside _KEYS pass through unchecked. One changes behavior: the Init
    # node's family_override (family_select.WORKER_ARG_KEY), which the Init node
    # checks and adapters._forced_adapter refuses when it names no adapter.
    out: dict[str, object] = dict(values) if not reject_unknown else {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ClusterConfigError(f"{context} keys must be strings (got {key!r})")
        field = f"{context}.{key}"
        if key not in _KEYS:
            continue
        if key in _BOOL_KEYS:
            if not isinstance(value, bool):
                raise ClusterConfigError(f"{field} must be a boolean true/false, not {value!r}")
            out[key] = value
        elif key in _FLOAT_LIMITS:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ClusterConfigError(f"{field} must be a finite number (got {value!r})")
            try:
                number = float(value)
            except OverflowError as exc:
                raise ClusterConfigError(f"{field} is too large (got {value!r})") from exc
            low, high = _FLOAT_LIMITS[key]
            if not math.isfinite(number) or not low <= number <= high:
                raise ClusterConfigError(
                    f"{field} must be finite and in {low:g}..{high:g} (got {value!r})")
            out[key] = number
        elif key == "slab_weights":
            if not isinstance(value, bool) and value != "auto":
                raise ClusterConfigError(
                    f'{field} must be true, false, or "auto" (got {value!r})')
            out[key] = value
        elif key == "swap_verify":
            if isinstance(value, bool) or not isinstance(value, int) or not -1 <= value <= 64:
                raise ClusterConfigError(f"{field} must be an integer in -1..64 (got {value!r})")
            out[key] = value
        elif key == "fsdp_prefetch_depth":
            # adapters/fsdp.py apply_fsdp_capacity_mode explains each depth.
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 8:
                raise ClusterConfigError(f"{field} must be an integer in 1..8 (got {value!r})")
            out[key] = value
        elif key == "safetensors_backend":
            if value not in ("pread", "mmap"):
                raise ClusterConfigError(f'{field} must be "pread" or "mmap" (got {value!r})')
            out[key] = value
        elif key == "int8_gemm":
            if value not in ("triton", "cutlass"):
                raise ClusterConfigError(f'{field} must be "triton" or "cutlass" (got {value!r})')
            out[key] = value
        elif key == "worker_vae_decode":
            if value not in ("off", "auto"):
                raise ClusterConfigError(f'{field} must be "off" or "auto" (got {value!r})')
            out[key] = value
    # Cross-key rule, so a cluster.toml cannot set a combination the Init node
    # refuses. comfy-managed residency forces both levers off inside the
    # worker; a config that also asked for either would be silently overruled.
    # `is True` and not truthiness: "auto" is a resolution policy, not a
    # residency request, and apply_policy forces it to False inside the worker.
    # The Init node, too, refuses only an explicit `on`.
    if out.get("comfy_managed") is True:
        if out.get("slab_weights") is True:
            raise ClusterConfigError(
                f"{context}: comfy_managed and slab_weights are two different residencies "
                "for the same weights; set slab_weights to false or \"auto\" (or omit it)")
        if out.get("lora_low_rss") is True:
            raise ClusterConfigError(
                f"{context}: comfy_managed forces lora_low_rss off; remove lora_low_rss "
                "or set it to false")
    return cast(WorkerArgs, out)


_FABRIC_KEY = re.compile(r"(?:NCCL|GLOO|UCX)_[A-Z0-9_]+\Z")
_DANGEROUS_FABRIC_TOKENS = frozenset({"FILE", "PLUGIN"})


def _dangerous_fabric_key(key: str) -> bool:
    """Match loader/path fields as underscore-delimited variable-name tokens."""
    tokens = key.split("_")
    if any(token in _DANGEROUS_FABRIC_TOKENS for token in tokens):
        return True
    return any(
        left == "MODULE" and right == "DIR"
        for left, right in pairwise(tokens)
    )


def validate_fabric_env(values: object, profile: str, path: Path | None = None) -> dict[str, str]:
    """Allow network-library tuning, never process-loader/path injection."""
    location = f"{path}: [fabric.{profile}]" if path is not None else profile
    if not isinstance(values, dict):
        raise ClusterConfigError(f"{location} must be a mapping")
    out: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not _FABRIC_KEY.fullmatch(key):
            raise ClusterConfigError(
                f"{location} key {key!r} is not an allowed "
                "NCCL_/GLOO_/UCX_ tuning variable")
        if key == "NCCL_PROTO":
            raise ClusterConfigError(
                f"{location} key 'NCCL_PROTO' must remain unset; protocol "
                "selection is not an operator fabric-profile knob")
        if _dangerous_fabric_key(key):
            raise ClusterConfigError(
                f"{location} key {key!r} may load code or write an "
                "arbitrary path and is not permitted")
        if not isinstance(value, (str, int, float, bool)):
            raise ClusterConfigError(f"{location} {key} must be a scalar value")
        rendered = "1" if value is True else "0" if value is False else str(value)
        if len(rendered) > 4096 or any(ord(ch) < 32 or ord(ch) == 127 for ch in rendered):
            raise ClusterConfigError(
                f"{location} {key} contains control characters or is longer than 4096 characters")
        out[key] = rendered
    return out


def validate_fabric_tables(values: object, path: Path) -> dict[str, dict[str, str]]:
    """Validate selected and dormant profiles before either can reach actors."""
    if not isinstance(values, dict):
        raise ClusterConfigError(
            f"{path}: `fabric` must contain [fabric.<name>] tables")
    out: dict[str, dict[str, str]] = {}
    for name, profile_values in values.items():
        if not isinstance(name, str) or not name:
            raise ClusterConfigError(
                f"{path}: fabric profile names must be non-empty strings")
        if not isinstance(profile_values, dict):
            raise ClusterConfigError(
                f"{path}: fabric entry {name!r} must be a [fabric.<name>] table")
        # A dormant typo must fail while the file is read, not remain latent
        # until an operator selects a different profile later.
        out[name] = validate_fabric_env(profile_values, name, path)
    return out


def validate_shell_token(value: object, field: str, path: Path) -> str:
    """Validate an ssh destination/user token before it reaches ssh or rsync."""
    if (not isinstance(value, str) or not value or value.startswith("-")
            or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in value)):
        raise ClusterConfigError(
            f"{path}: {field} must be a non-empty string, must not start with '-', "
            "and must not contain whitespace/control characters")
    return value


def validate_ssh_host(value: str, field: str, path: Path) -> str:
    """Reject SSH/rsync destination delimiters while retaining literal IPv6."""
    value = validate_shell_token(value, field, path)
    if "@" in value:
        raise ClusterConfigError(
            f"{path}: {field} must not contain '@', which rewrites the SSH login target")
    if ":" in value:
        try:
            literal = ipaddress.ip_address(value)
        except ValueError as exc:
            raise ClusterConfigError(
                f"{path}: {field} must not contain ':' unless it is a literal IPv6 address"
            ) from exc
        if literal.version != 6:
            raise ClusterConfigError(
                f"{path}: {field} must not contain ':' unless it is a literal IPv6 address")
    return value


def validate_ssh_user(value: str, field: str, path: Path) -> str:
    """Reject delimiters that can change the SSH or rsync destination parse."""
    value = validate_shell_token(value, field, path)
    if "@" in value or ":" in value:
        raise ClusterConfigError(
            f"{path}: {field} must not contain '@' or ':' destination delimiters")
    return value


def validate_plain_string(value: object, field: str, path: Path, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        qualifier = "a non-empty string" if nonempty else "a string"
        raise ClusterConfigError(f"{path}: {field} must be {qualifier}")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ClusterConfigError(f"{path}: {field} must not contain control characters")
    return value


# [studio]: read only by dgxm studio and the companion's driver hint, never by
# load_cluster_config, so ClusterConfig and the Gate trust fingerprint stay as they are.
_STUDIO_KEYS = frozenset({"driver", "port", "driver_service", "driver_command"})
DEFAULT_STUDIO_DRIVER = "http://127.0.0.1:8188"
DEFAULT_STUDIO_PORT = 8220
# A systemd user service by exact name; used with fullmatch.
DRIVER_UNIT_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]*\.service")
_ORIGIN_LIMIT = 2048
_COMMAND_LIMIT = 512
_DRIVER_SHAPE = ("studio.driver must be an http or https origin with a host and no user, path, query, "
                 "fragment or zone")
_DRIVER_LISTEN = ("studio.driver names a listen address; use the address a client connects to, "
                  "such as http://127.0.0.1:8191")
_DRIVER_LOCAL = ("studio.driver must name this computer (127.0.0.1, [::1] or localhost); "
                 "for a driver on another computer pass dgxm studio --driver")


class StudioTable(TypedDict):
    """The [studio] table with its defaults filled in."""

    driver: str | None
    port: int
    driver_service: str | None
    driver_command: str | None


def canonical_driver_origin(origin: str) -> str:
    """The form a Studio state directory binds: lowercase host, bracketed IPv6, default port dropped."""
    parsed = urlsplit(origin)
    host = (parsed.hostname or "").lower()
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    authority = host if port is None or (parsed.scheme, port) in {("http", 80), ("https", 443)} else f"{host}:{port}"
    return f"{parsed.scheme}://{authority}"


def _unspecified(host: str) -> bool:
    """Whether a host is a listen-everywhere address in any spelling, 0 and 0x0 included."""
    try:
        return ipaddress.ip_address(host).is_unspecified
    except ValueError:
        pass
    if not re.fullmatch(r"[0-9A-Fa-fXx.]+", host):
        return False
    try:
        # inet_aton takes the short forms a connect also takes, such as 0 and 0.0.
        return socket.inet_aton(host) == bytes(4)
    except OSError:
        return False


def _this_computer(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if address.version == 4:
        return address in ipaddress.ip_network("127.0.0.0/8")
    return address == ipaddress.IPv6Address("::1")


def validate_driver_origin(value: str, *, local_only: bool) -> str:
    """Return a Studio driver origin unchanged when it is valid and canonical.

    The rules of the [studio] driver key: at most 2048 printable characters,
    http or https, a host, no user, password, path other than an empty one or
    "/", query or fragment, a port urlsplit can parse, no "%" anywhere (no IPv6
    zone), no listen-everywhere address, and the value must equal
    canonical_driver_origin(value), so "http://127.0.0.1:8191/" is refused with
    the slashless form named. With local_only the host must also be a
    127.0.0.0/8 literal, ::1 or localhost; a typed --driver skips only that rule.
    Raises ValueError whose text is the refusal without a path prefix.
    """
    if not isinstance(value, str) or not value:
        raise ValueError("studio.driver must be a non-empty string")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError("studio.driver must not contain control characters")
    if len(value) > _ORIGIN_LIMIT or "%" in value or not all(33 <= ord(ch) <= 126 for ch in value):
        raise ValueError(_DRIVER_SHAPE)
    try:
        parsed = urlsplit(value)
        _ = parsed.port  # urlsplit checks the port only when it is read
    except ValueError as exc:
        raise ValueError(_DRIVER_SHAPE) from exc
    host = parsed.hostname or ""
    if (parsed.scheme not in {"http", "https"} or not host or "@" in parsed.netloc
            or parsed.path not in {"", "/"} or parsed.query or parsed.fragment):
        raise ValueError(_DRIVER_SHAPE)
    if _unspecified(host):
        raise ValueError(_DRIVER_LISTEN)
    if local_only and not _this_computer(host):
        raise ValueError(_DRIVER_LOCAL)
    canonical_origin = canonical_driver_origin(value)
    if value != canonical_origin:
        raise ValueError(f"studio.driver must be written as '{canonical_origin}'")
    return value


def validate_studio_table(values: object, path: Path) -> StudioTable:
    """Check the [studio] table and fill its defaults; ClusterConfigError names the file."""
    if not isinstance(values, dict):
        raise ClusterConfigError(f"{path}: [studio] must be a table")
    _reject_unknown_keys(values, _STUDIO_KEYS, context=f"{path}: [studio]")
    table = StudioTable(driver=None, port=DEFAULT_STUDIO_PORT, driver_service=None, driver_command=None)
    if "driver" in values:
        driver = validate_plain_string(values["driver"], "studio.driver", path, nonempty=True)
        try:
            table["driver"] = validate_driver_origin(driver, local_only=True)
        except ValueError as exc:
            raise ClusterConfigError(f"{path}: {exc}") from exc
    if "port" in values:
        port = values["port"]
        if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
            raise ClusterConfigError(f"{path}: studio.port must be an integer from 1024 to 65535")
        table["port"] = port
    if "driver_service" in values:
        unit = values["driver_service"]
        if not isinstance(unit, str) or not DRIVER_UNIT_PATTERN.fullmatch(unit):
            raise ClusterConfigError(
                f"{path}: studio.driver_service must name a systemd user service, such as comfyui.service")
        if "worker" in unit.lower():
            raise ClusterConfigError(f"{path}: studio.driver_service must not name a Worker unit")
        if unit == "dgxm-studio.service":
            raise ClusterConfigError(f"{path}: studio.driver_service must not name dgxm-studio.service")
        table["driver_service"] = unit
    if "driver_command" in values:
        command = values["driver_command"]
        if (not isinstance(command, str) or not 1 <= len(command) <= _COMMAND_LIMIT
                or not all(32 <= ord(ch) <= 126 for ch in command)):
            raise ClusterConfigError(
                f"{path}: studio.driver_command must be 1 to 512 printable ASCII characters")
        table["driver_command"] = command
    return table
