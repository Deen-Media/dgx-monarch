"""Load ``cluster.toml`` and resolve fabric profiles (DESIGN.md §6.1).

Configuration covers hosts, bind addresses, transport, and NCCL environment.
Search order is an explicit path, ``DGXM_CLUSTER_TOML``, ``./cluster.toml``,
then ``~/.config/dgx-monarch/cluster.toml``. Hardware settings live in named
profiles rather than runtime branches.
"""
from __future__ import annotations

import ipaddress
import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from . import config_schema as _schema
from .constants import DEFAULT_NCCL_MASTER_PORT, DEFAULT_RDMA_MIN_BYTES, DEFAULT_WORKER_PORT

ClusterConfigError = _schema.ClusterConfigError

def is_literal_ip(value: str) -> bool:
    if not isinstance(value, str):
        return False
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


def unmapped_bind_ip(value: str) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, bool]:
    """Return the address to classify and whether it was IPv4-mapped.

    CPython versions disagree on loopback, unspecified, and multicast properties
    of IPv4-mapped IPv6 literals. Classify the unwrapped IPv4 address so mapped
    and plain spellings receive the same bind and advertise checks. Raise
    ``ValueError`` for a nonliteral address.
    """
    ip = ipaddress.ip_address(value)
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is None:
        return ip, False
    return mapped, True


def format_tcp_address(host: str, port: int) -> str:
    if not isinstance(host, str):
        raise ValueError(f"TCP host must be a string literal IP address (got {type(host).__name__})")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError(f"TCP port must be an integer in 0..65535 (got {port!r})")
    ip = ipaddress.ip_address(host)
    rendered = f"[{ip.compressed}]" if ip.version == 6 else ip.compressed
    return f"tcp://{rendered}:{port}"


def tcp_endpoint(url: str) -> tuple[str, int]:
    """Return the literal host and port from a validated tcp address."""
    if not isinstance(url, str):
        raise ValueError(f"TCP endpoint must be a string (got {type(url).__name__})")
    parts = urlsplit(url)
    if (parts.scheme.lower() != "tcp" or parts.username is not None
            or parts.password is not None or parts.path or parts.query or parts.fragment
            or parts.hostname is None or not is_literal_ip(parts.hostname)
            or parts.port is None):
        raise ValueError(f"not a complete tcp endpoint: {url!r}")
    return parts.hostname, parts.port


def _validate_bind(
    url: object,
    what: str,
    path: Path,
    *,
    allow_zero_port: bool,
) -> str:
    """Validate a literal-IP worker or client bind.

    Hostname advertisement can resolve to loopback and make workers dial
    themselves. Parse-time validation reports that configuration error before
    the attach timeout (docs/TROUBLESHOOTING.md #1).
    """
    if not isinstance(url, str):
        raise ClusterConfigError(
            f"{path}: {what} must be a string tcp:// URL (got {type(url).__name__})")
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise ClusterConfigError(f"{path}: {what} is not a valid tcp:// URL ({url!r}): {exc}") from exc
    if parts.scheme.lower() != "tcp":
        raise ClusterConfigError(f"{path}: {what} must be a tcp:// URL (got {url!r})")
    if parts.username is not None or parts.password is not None:
        raise ClusterConfigError(f"{path}: {what} must not contain user-info ({url!r})")
    if parts.path or parts.query or parts.fragment:
        raise ClusterConfigError(
            f"{path}: {what} must contain only a host and port, with no path/query/fragment "
            f"(got {url!r})")
    try:
        port = parts.port
    except ValueError as exc:
        raise ClusterConfigError(f"{path}: {what} has an invalid port ({url!r}): {exc}") from exc
    if parts.hostname is None or not is_literal_ip(parts.hostname):
        raise ClusterConfigError(
            f"{path}: {what} host must be an explicit fabric IP, not a hostname (got {url!r}). "
            "A hostname bind can advertise loopback through /etc/hosts (docs/TROUBLESHOOTING.md #1); "
            "use a literal IP, such as tcp://192.0.2.11:0 for client_bind or tcp://192.0.2.12:26600 for a host."
        )
    bind_ip, mapped = unmapped_bind_ip(parts.hostname)
    if mapped:
        raise ClusterConfigError(
            f"{path}: {what} is the IPv4-mapped spelling of {bind_ip.compressed} "
            f"(got {url!r}); write the IPv4 address itself, which is what every "
            "bind and advertise check reads")
    if bind_ip.is_unspecified or bind_ip.is_multicast:
        raise ClusterConfigError(
            f"{path}: {what} must be a unicast interface address, not "
            f"{bind_ip.compressed!r}")
    if port is None:
        raise ClusterConfigError(f"{path}: {what} must include a port (got {url!r}); use tcp://<ip>:<port>")
    if port == 0 and not allow_zero_port:
        raise ClusterConfigError(f"{path}: {what} worker port must be nonzero (got {url!r})")
    return format_tcp_address(parts.hostname, port)


# Fabric profiles apply NCCL/GLOO/UCX settings inside actor processes only.
# docs/VALIDATION.md records the evidence behind each empirical setting.
FABRIC_PROFILES: dict[str, dict[str, str]] = {
    # 2x DGX Spark (GB10) back-to-back over ConnectX-7 200G RoCE, both rails.
    "dgx-spark-pair": {
        "NCCL_SOCKET_IFNAME": "enp1s0f0np0",
        "GLOO_SOCKET_IFNAME": "enp1s0f0np0",
        "NCCL_IB_HCA": "rocep1s0f0,roceP2p1s0f0",
        "UCX_NET_DEVICES": "enp1s0f0np0",
        "NCCL_IB_GID_INDEX": "3",
        # GB10 has no GPUDirect RDMA (cuMemGdrSupport 0), so NCCL stages
        # through host memory; SYS keeps it from trying GDR paths.
        "NCCL_NET_GDR_LEVEL": "SYS",
        "NCCL_BUFFSIZE": "16777216",
        "NCCL_DEBUG": "WARN",
        # NCCL_PROTO must stay unset: PROTO=LL is an FSDP-killer (VALIDATION.md).
    },
    # Generic RoCE template: users fill in their interface/HCA names.
    "generic-roce": {
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_DEBUG": "WARN",
    },
    # Generic InfiniBand: NCCL autodetects the fabric, so nothing else is set.
    "generic-ib": {
        "NCCL_DEBUG": "WARN",
    },
    "single-node": {},
}


def detect_roce_rails(sys_root: str = "/sys") -> list[tuple[str, str]]:
    """(netdev, rdma_device) pairs for RDMA-capable ports that are UP.

    The mapping comes from sysfs: every rdma device exposes its paired netdev
    under /sys/class/infiniband/<dev>/device/net/. Sorted case-insensitively
    by rdma device so the primary rail is stable across boxes.
    """
    rails: list[tuple[str, str]] = []
    ib_root = os.path.join(sys_root, "class", "infiniband")
    if not os.path.isdir(ib_root):
        return rails
    for rdma_dev in os.listdir(ib_root):
        net_dir = os.path.join(ib_root, rdma_dev, "device", "net")
        if not os.path.isdir(net_dir):
            continue
        for netdev in os.listdir(net_dir):
            try:
                with open(os.path.join(sys_root, "class", "net", netdev, "operstate")) as f:
                    oper = f.read().strip()
            except OSError:
                continue
            if oper == "up":
                rails.append((netdev, rdma_dev))
    rails.sort(key=lambda pair: pair[1].lower())
    return rails


def fixup_fabric_ifaces(env: dict[str, str], sys_root: str = "/sys") -> tuple[dict[str, str], str]:
    """Resolve a profile that names an absent or down interface.

    An active configured interface passes through unchanged. Otherwise sysfs
    supplies the active RDMA rails used to rewrite socket and HCA variables.
    The note is empty only when the configured interface is up or unset.
    """
    want = env.get("NCCL_SOCKET_IFNAME", "")
    if not want:
        return env, ""
    try:
        with open(os.path.join(sys_root, "class", "net", want, "operstate")) as f:
            if f.read().strip() == "up":
                return env, ""
        state = "not up"
    except OSError:
        state = "absent"
    rails = detect_roce_rails(sys_root)
    if not rails:
        return env, (
            f"fabric profile interface {want!r} is {state} on this host and no UP RDMA rails "
            "were detected. NCCL will likely fail; check cabling and the fabric profile"
        )
    healed = dict(env)
    primary = rails[0][0]
    for key in ("NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "UCX_NET_DEVICES"):
        if key in healed:
            healed[key] = primary
    if "NCCL_IB_HCA" in healed:
        healed["NCCL_IB_HCA"] = ",".join(rdma for _, rdma in rails)
    return healed, (
        f"fabric profile interface {want!r} is {state} on this host; the cable may be in the other "
        f"QSFP port set. Rails that are up, {rails}, replace the configured socket, UCX and HCA names"
    )


@dataclass(frozen=True)
class HostConfig:
    name: str                 # ssh-reachable name or IP; hosts[0]'s also tells whether rank 0 shares the driver box
    address: str              # monarch worker bind address, e.g. tcp://192.0.2.12:26600
    gpus: int = 1
    ssh_user: str = ""        # empty = current user
    comfy_dir: str = ""       # empty = same as driver


@dataclass(frozen=True)
class ClusterConfig:
    hosts: tuple[HostConfig, ...] = ()
    auto_heal: bool = True    # restart every Worker service before attach if one is down, and after a failed attach
    client_bind: str = ""     # tcp://<fabric-ip>:0; a cluster attach refuses without it (docs/TROUBLESHOOTING.md #1)
    fabric_profile: str = "single-node"
    fabric_env: dict[str, str] = field(default_factory=dict)  # overrides/extends the profile
    nccl_master_addr: str = ""   # default: the first host's worker bind IP
    nccl_master_port: int = DEFAULT_NCCL_MASTER_PORT
    comfy_dir: str = ""          # driver-side ComfyUI root (auto-detected when empty)
    # DESIGN §5.6 one-sided RDMA return, size-gated below. Off by default:
    # Monarch can pair a transfer across rails with no route.
    rdma_latent_return: bool = False
    rdma_min_bytes: int = DEFAULT_RDMA_MIN_BYTES  # §5.6: latents smaller than this return via messaging
    ssh_key: str = ""            # key used by dgxm lifecycle commands
    python_bin: str = "python3"  # interpreter that has torchmonarch + comfy deps, on every host
    # The attach API offers no peer authentication, so a networked cluster must
    # declare its fabric isolated (SECURITY.md).
    transport_security: str = ""
    # [worker_args] table: comfy knobs applied inside actor procs (reserve_vram_gb,
    # disable_pinned_memory, disable_smart_memory; actor/comfy_bridge._apply_worker_args).
    # Init-node widgets win on a clash.
    worker_args: dict[str, object] = field(default_factory=dict)
    source: str = ""             # path this config was loaded from ("" = local-only default)

    @property
    def world_size(self) -> int:
        return sum(h.gpus for h in self.hosts) if self.hosts else 0

    def resolved_fabric_env(self) -> dict[str, str]:
        profile = dict(FABRIC_PROFILES.get(self.fabric_profile, {}))
        profile.update(self.fabric_env)
        return profile

    def resolved_master_addr(self) -> str:
        if self.nccl_master_addr:
            return self.nccl_master_addr
        if self.hosts:
            return tcp_endpoint(self.hosts[0].address)[0]
        return "127.0.0.1"


def default_config_paths() -> list[Path]:
    paths = []
    env = os.environ.get("DGXM_CLUSTER_TOML")
    if env:
        paths.append(Path(env))
    paths.append(Path.cwd() / "cluster.toml")
    paths.append(Path.home() / ".config" / "dgx-monarch" / "cluster.toml")
    return paths


def find_config_path(explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            # An explicit typo must not select local mode and skip checks.
            raise ClusterConfigError(
                f"cluster config {explicit!r} does not exist (or is not a file)")
        return p
    for p in default_config_paths():
        if p.is_file():
            return p
    return None


def _host_from_table(h, path: Path, index: int) -> HostConfig:
    if not isinstance(h, dict):
        raise ClusterConfigError(
            f"{path}: hosts must be [[hosts]] array-of-tables entries (got {type(h).__name__}); "
            "check for a bare [hosts] table"
        )
    _schema.validate_host_keys(h, path, index)
    name = h.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ClusterConfigError(f"{path}: a [[hosts]] entry needs a `name` string that is not blank")
    _schema.validate_ssh_host(name, "host name", path)
    address = h.get("address", "")
    if not address:
        # DESIGN §5.1: worker bind addresses are explicit fabric IPs, never
        # derived from a hostname, which can resolve to loopback
        # (docs/TROUBLESHOOTING.md #1). Only a literal IP name may stand in for a missing address.
        if is_literal_ip(name):
            address = format_tcp_address(name, DEFAULT_WORKER_PORT)
        else:
            raise ClusterConfigError(
                f"{path}: host {name!r} has no `address`. Worker services must bind an explicit "
                f'fabric IP (address = "tcp://<fabric-ip>:{DEFAULT_WORKER_PORT}"); deriving it '
                "from the ssh name can bind loopback through /etc/hosts (docs/CLUSTER.md)."
            )
    address = _validate_bind(
        address, f"host {name!r} address", path, allow_zero_port=False)
    gpus = h.get("gpus", 1)
    if (isinstance(gpus, bool) or not isinstance(gpus, int)
            or not 1 <= gpus <= _schema.MAX_GPUS_PER_HOST):
        raise ClusterConfigError(
            f"{path}: host {name!r} gpus must be a positive integer no greater than "
            f"{_schema.MAX_GPUS_PER_HOST} (got {gpus!r})")
    ssh_user = _schema.validate_plain_string(h.get("ssh_user", ""), f"host {name!r} ssh_user", path)
    if ssh_user:
        _schema.validate_ssh_user(ssh_user, f"host {name!r} ssh_user", path)
    comfy_dir = _schema.validate_plain_string(h.get("comfy_dir", ""), f"host {name!r} comfy_dir", path)
    return HostConfig(
        name=name,
        address=address,
        gpus=gpus,
        ssh_user=ssh_user,
        comfy_dir=comfy_dir,
    )


def load_cluster_config(path: str | Path) -> ClusterConfig:
    path = Path(path).expanduser()
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ClusterConfigError(f"{path}: invalid TOML: {exc}") from exc

    _schema.validate_root_keys(raw, path)
    cluster = raw.get("cluster", {})
    if not isinstance(cluster, dict):
        raise ClusterConfigError(f"{path}: `cluster` must be a [cluster] table")
    _schema.validate_cluster_keys(cluster, path)
    hosts_raw = raw.get("hosts", [])
    if not isinstance(hosts_raw, list):
        raise ClusterConfigError(
            f"{path}: `hosts` must be [[hosts]] array-of-tables entries, one per host")
    if len(hosts_raw) > _schema.MAX_CLUSTER_HOSTS:
        raise ClusterConfigError(
            f"{path}: cluster has {len(hosts_raw)} hosts; maximum is {_schema.MAX_CLUSTER_HOSTS}")
    hosts = tuple(
        _host_from_table(host, path, index)
        for index, host in enumerate(hosts_raw, 1)
    )
    if not hosts:
        raise ClusterConfigError(
            f"{path}: no [[hosts]] entries. A cluster config needs at least one host; "
            "for single-box multi-GPU, switch the Init node to local mode instead."
        )
    gpu_counts = {host.gpus for host in hosts}
    if len(gpu_counts) != 1:
        raise ClusterConfigError(
            f"{path}: heterogeneous gpus-per-host is not supported; configured counts are "
            f"{sorted(gpu_counts)}")
    world_size = sum(host.gpus for host in hosts)
    if world_size > _schema.MAX_WORLD_SIZE:
        raise ClusterConfigError(
            f"{path}: cluster world size {world_size} exceeds the supported maximum "
            f"of {_schema.MAX_WORLD_SIZE}")

    profile_name = _schema.validate_fabric_profile_name(
        cluster.get("fabric_profile", "single-node"), len(hosts), path)
    fabric_env: dict[str, str] = {}
    validated_fabric_tables = _schema.validate_fabric_tables(
        raw.get("fabric", {}), path)
    if profile_name in validated_fabric_tables:
        fabric_env = validated_fabric_tables[profile_name]
    elif profile_name not in FABRIC_PROFILES:
        raise ClusterConfigError(
            f"{path}: unknown fabric_profile {profile_name!r} with no [fabric.{profile_name}] "
            f"table. NCCL would run on the fabric with no profile settings. Known profiles: "
            f"{', '.join(FABRIC_PROFILES)}."
        )

    client_bind = cluster.get("client_bind", "")
    if client_bind:
        client_bind = _validate_bind(
            client_bind, "cluster.client_bind", path, allow_zero_port=True)
    elif not isinstance(client_bind, str):
        raise ClusterConfigError(f"{path}: cluster.client_bind must be a string tcp:// URL")

    transport_security = cluster.get("transport_security", "")
    if transport_security not in ("", "trusted_fabric"):
        raise ClusterConfigError(
            f"{path}: cluster.transport_security must be \"trusted_fabric\" or empty (got "
            f"{transport_security!r}); the attach API dgx-monarch calls offers no "
            "peer authentication (SECURITY.md)")

    master_addr = cluster.get("nccl_master_addr", "")
    if not isinstance(master_addr, str):
        raise ClusterConfigError(f"{path}: cluster.nccl_master_addr must be a literal IP string")
    if master_addr:
        if not is_literal_ip(master_addr):
            raise ClusterConfigError(
                f"{path}: cluster.nccl_master_addr must be a literal IPv4/IPv6 address "
                f"(got {master_addr!r})")
        master_ip, mapped = unmapped_bind_ip(master_addr)
        if mapped:
            raise ClusterConfigError(
                f"{path}: cluster.nccl_master_addr is the IPv4-mapped spelling "
                f"of {master_ip.compressed}; write the IPv4 address itself")
        if master_ip.is_unspecified or master_ip.is_multicast:
            raise ClusterConfigError(
                f"{path}: cluster.nccl_master_addr must be a unicast interface address "
                f"(got {master_addr!r})")
        master_addr = master_ip.compressed

    master_port = cluster.get("nccl_master_port", DEFAULT_NCCL_MASTER_PORT)
    if isinstance(master_port, bool) or not isinstance(master_port, int) or not 1 <= master_port <= 65535:
        raise ClusterConfigError(
            f"{path}: cluster.nccl_master_port must be an integer in 1..65535 "
            f"(got {master_port!r})")

    rdma_min_bytes = cluster.get("rdma_min_bytes", DEFAULT_RDMA_MIN_BYTES)
    if (isinstance(rdma_min_bytes, bool) or not isinstance(rdma_min_bytes, int)
            or rdma_min_bytes < 0):
        raise ClusterConfigError(
            f"{path}: cluster.rdma_min_bytes must be a non-negative integer "
            f"(got {rdma_min_bytes!r})")

    for key in ("auto_heal", "rdma_latent_return"):
        value = cluster.get(key, True)
        if not isinstance(value, bool):
            raise ClusterConfigError(f"{path}: cluster.{key} must be a boolean (got {value!r})")
    for key, default in (("comfy_dir", ""), ("ssh_key", ""), ("python", "python3")):
        _schema.validate_plain_string(
            cluster.get(key, default), f"cluster.{key}", path, nonempty=(key == "python"))
    worker_args = raw.get("worker_args", {})
    worker_args = _schema.validate_worker_args(worker_args, context=f"{path}: worker_args")

    return ClusterConfig(
        hosts=hosts,
        client_bind=client_bind,
        auto_heal=cluster.get("auto_heal", True),
        fabric_profile=profile_name,
        fabric_env=fabric_env,
        nccl_master_addr=master_addr,
        nccl_master_port=master_port,
        comfy_dir=cluster.get("comfy_dir", ""),
        rdma_latent_return=cluster.get("rdma_latent_return", False),
        rdma_min_bytes=rdma_min_bytes,
        ssh_key=cluster.get("ssh_key", ""),
        python_bin=cluster.get("python", "python3"),
        transport_security=transport_security,
        worker_args=dict(worker_args),
        source=str(path),
    )


def local_config(config_path: Path | None = None) -> ClusterConfig:
    """Zero-config local mesh: single host, local GPUs (DESIGN.md §6.1 path 1).

    Cluster facts stay out, because a local mesh has none. `[worker_args]` is
    not a cluster fact: it is the operator's comfy knobs for every actor proc,
    and a local worker is an actor proc. `source` stays empty so the mesh cache
    key and the doctor keep reading this as the local path.

    `config_path` is the already-resolved config, so an operator who named one
    gets that file's table rather than whichever the search order finds. It
    falls back to the search for callers holding no path.
    """
    if config_path is None:
        config_path = find_config_path()
    return ClusterConfig(
        fabric_profile="single-node",
        worker_args=_worker_args_only(config_path),
        source="",
    )


def _worker_args_only(path: Path | None) -> dict[str, object]:
    """Read just `[worker_args]` out of a cluster config, ignoring the rest.

    A malformed file, or a malformed table inside it, raises the same typed
    error cluster mode raises for it: a silent skip would drop the operator's
    comfy knobs from local mode with no error.
    """
    if path is None:
        return {}
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ClusterConfigError(f"{path}: invalid TOML: {exc}") from exc
    except OSError:
        # Racing removal or an unreadable file leaves the local path zero-config,
        # which is where it started.
        return {}
    if not isinstance(raw, dict):
        return {}
    worker_args = _schema.validate_worker_args(
        raw.get("worker_args", {}), context=f"{path}: worker_args")
    return dict(worker_args)


def _toml_string(value: str, field: str) -> str:
    """Render a user-provided TOML basic string without hand-built quoting."""
    if not isinstance(value, str) or "\x00" in value:
        raise ClusterConfigError(f"{field} must be a string without NUL bytes")
    # JSON's quoted-string grammar is valid TOML basic-string grammar for the
    # characters the callers accept, and it escapes quotes, backslashes and
    # control characters in interactive paths and SSH aliases.
    return json.dumps(value, ensure_ascii=False)


CLUSTER_TOML_TEMPLATE = """\
# dgx-monarch cluster config. See docs/CLUSTER.md for the full reference.

[cluster]
# Client bind address on the fabric IP. REQUIRED for multi-host: the default
# transport advertises whatever the hostname resolves to, which is loopback on
# many setups, and the attach dies with MESH_ATTACH_CONFIG_TIMEOUT.
client_bind = {client_bind}
fabric_profile = {fabric_profile}
transport_security = {transport_security}  # after source-restricting all dedicated-fabric traffic
nccl_master_addr = {master_addr}
nccl_master_port = {master_port}
python = {python_bin}
# rdma_latent_return = false  # off by default; see docs/DESIGN.md §5.6
# rdma_min_bytes = 8388608    # below this a latent returns via messaging (default 8 MiB)
# comfy_dir = "/home/user/ComfyUI"
ssh_key = {ssh_key}

{host_blocks}
# Named fabric profiles may be overridden/extended per cluster:
# [fabric.{fabric_profile_name}]
# NCCL_SOCKET_IFNAME = "eth0"

# comfy behavior knobs applied inside every actor proc (Init widgets win):
# [worker_args]
# disable_pinned_memory = true
# reserve_vram_gb = 8.0
"""

HOST_BLOCK_TEMPLATE = """\
[[hosts]]
name = {name}
address = {address}
gpus = {gpus}

"""


def render_cluster_toml(
    hosts: list[tuple[str, str, int]],  # (ssh name, fabric ip, gpus)
    client_ip: str,
    fabric_profile: str,
    master_addr: str,
    master_port: int = DEFAULT_NCCL_MASTER_PORT,
    ssh_key: str = "",
    worker_port: int = DEFAULT_WORKER_PORT,
    python_bin: str = "python3",
    transport_security: str = "",
) -> str:
    if isinstance(worker_port, bool) or not isinstance(worker_port, int) or not 1 <= worker_port <= 65535:
        raise ClusterConfigError(f"worker_port must be an integer in 1..65535 (got {worker_port!r})")
    if isinstance(master_port, bool) or not isinstance(master_port, int) or not 1 <= master_port <= 65535:
        raise ClusterConfigError(f"master_port must be an integer in 1..65535 (got {master_port!r})")
    if transport_security not in ("", "trusted_fabric"):
        raise ClusterConfigError(
            "transport_security must be empty, or 'trusted_fabric' once the fabric is isolated"
        )
    _schema.validate_fabric_profile_name(fabric_profile, len(hosts), Path("cluster.toml"))
    if fabric_profile not in FABRIC_PROFILES:
        raise ClusterConfigError(
            f"fabric_profile must be one of {', '.join(FABRIC_PROFILES)} "
            f"(got {fabric_profile!r})"
        )
    _schema.validate_plain_string(python_bin, "python", Path("cluster.toml"), nonempty=True)
    _schema.validate_plain_string(ssh_key, "ssh_key", Path("cluster.toml"))

    def literal_ip(
        value: str, field: str, *, reject_mapped: bool = False,
    ) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
        if not isinstance(value, str):
            raise ClusterConfigError(
                f"{field} must be a valid literal IP address string (got {value!r})"
            )
        try:
            ip = ipaddress.ip_address(value)
        except ValueError as exc:
            raise ClusterConfigError(
                f"{field} must be a valid literal IP address (got {value!r}): {exc}"
            ) from exc
        unwrapped, mapped = unmapped_bind_ip(value)
        if reject_mapped and mapped:
            raise ClusterConfigError(
                f"{field} must not use the IPv4-mapped spelling of "
                f"{unwrapped.compressed}; write the IPv4 address itself")
        if unwrapped.is_unspecified or unwrapped.is_multicast:
            raise ClusterConfigError(
                f"{field} must be a unicast interface address (got {value!r})")
        return ip

    client = literal_ip(client_ip, "client_ip")
    master = literal_ip(master_addr, "master_addr", reject_mapped=True)

    def host_block(name: str, address: str, gpus: int) -> str:
        if not isinstance(name, str) or not name:
            raise ClusterConfigError("host name must be a non-empty string")
        _schema.validate_ssh_host(name, "host name", Path("cluster.toml"))
        if (isinstance(gpus, bool) or not isinstance(gpus, int)
                or not 1 <= gpus <= _schema.MAX_GPUS_PER_HOST):
            raise ClusterConfigError(
                f"host {name!r} gpus must be a positive integer no greater than "
                f"{_schema.MAX_GPUS_PER_HOST} (got {gpus!r})")
        ip = literal_ip(address, f"host {name!r} address")
        address_text = format_tcp_address(ip.compressed, worker_port)
        return HOST_BLOCK_TEMPLATE.format(
            name=_toml_string(name, "host name"),
            address=_toml_string(address_text, "host address"),
            gpus=gpus,
        )

    if not hosts:
        raise ClusterConfigError("hosts must contain at least one entry")
    if len(hosts) > _schema.MAX_CLUSTER_HOSTS:
        raise ClusterConfigError(f"cluster has too many hosts (maximum {_schema.MAX_CLUSTER_HOSTS})")
    host_blocks = "".join(
        host_block(name, address, gpus) for name, address, gpus in hosts
    )
    gpu_counts = {gpus for _name, _address, gpus in hosts}
    if len(gpu_counts) != 1:
        raise ClusterConfigError("heterogeneous gpus-per-host is not supported")
    if sum(gpus for _name, _address, gpus in hosts) > _schema.MAX_WORLD_SIZE:
        raise ClusterConfigError(f"cluster world size exceeds maximum {_schema.MAX_WORLD_SIZE}")

    return CLUSTER_TOML_TEMPLATE.format(
        client_bind=_toml_string(format_tcp_address(client.compressed, 0), "client_bind"),
        fabric_profile=_toml_string(fabric_profile, "fabric_profile"),
        fabric_profile_name=fabric_profile,
        master_addr=_toml_string(master.compressed, "master_addr"),
        master_port=master_port,
        ssh_key=_toml_string(ssh_key, "ssh_key"),
        host_blocks=host_blocks,
        python_bin=_toml_string(python_bin, "python"),
        transport_security=_toml_string(transport_security, "transport_security"),
    )
