"""dgxm doctor: preflight checks of the driver, this box and every configured host, one row per check."""
from __future__ import annotations

import ipaddress
import os
import shutil
import socket

from .. import TORCHMONARCH_PIN
from ..config import ClusterConfig, format_tcp_address, tcp_endpoint
from . import (
    doctor_nccl,
    doctor_permissions,
    doctor_probes,
    json_report,
    listener_generation,
    mesh_health_row,
    probe_certainty,
    sol_attn_row,
)
from .advertised_address import hostname_row, unresolved_row
from .comfy_ports import find_comfy_ports as _find_comfy_ports
from .doctor_nccl import normalize_proto as _normalize_nccl_proto
from .doctor_nccl import uses_ll as _uses_nccl_ll
from .hwmon import board_hotspot_verdict
from .lifecycle import _pybin_shell, _tcp_probe, run_on_host
from .probe_certainty import probe_row as _row
from .worker_health import passive_worker_health, worker_service_row

_OK, _WARN, _FAIL = probe_certainty.OK, probe_certainty.WARN, probe_certainty.FAIL


def _env_tokens(line: str) -> dict[str, str]:
    """Parse the remote probe's whitespace-separated ``key=value`` fields."""
    return {
        token.split("=", 1)[0]: token.split("=", 1)[1]
        for token in line.split()
        if "=" in token
    }


def _transport_security_row(config: ClusterConfig | None) -> dict:
    if config is None or not config.hosts:
        return _row(_OK, "worker transport security", "local mode; no Worker service listener")
    if config.transport_security != "trusted_fabric":
        return _row(
            _FAIL,
            "worker transport security",
            "torchmonarch worker transports have no peer authentication at the "
            "API dgx-monarch calls. Source-restrict all traffic on the whole "
            "dedicated fabric interface to the driver and cluster peers; a rule "
            "covering port 26600 alone isolates nothing, because spawned actors "
            "allocate secondary dynamic ports. Then set "
            '`cluster.transport_security = "trusted_fabric"` (SECURITY.md).',
        )
    public = []
    for host in config.hosts:
        address, _ = tcp_endpoint(host.address)
        if ipaddress.ip_address(address).is_global:
            public.append(address)
    if public:
        return _row(
            _WARN,
            "worker transport security",
            f"trusted_fabric acknowledged, but globally routable worker IPs are configured: "
            f"{', '.join(public)}. Verify the whole fabric interface is source-restricted to "
            "the driver and cluster peers, not one listener port (SECURITY.md).",
        )
    return _row(
        _OK,
        "worker transport security",
        "trusted_fabric acknowledged; the whole dedicated fabric interface must stay "
        "source-restricted to the driver and cluster peers, since spawned actors allocate "
        "secondary dynamic ports (SECURITY.md)",
    )


def _parse_version(v: str):
    """Return a PEP 440 version key, or fall back to a numeric tuple.

    The fallback preserves numeric ordering, such as 1.48.0 after 1.45.20.
    """
    try:
        from packaging.version import parse as parse_pep440

        return parse_pep440(v)
    except Exception:
        return tuple(int(x) for x in str(v).split(".") if x.isdigit())


def classify_frontend_skew(system: dict, resolve_latest=None) -> tuple[str, str]:
    """Frontend and backend compatibility verdict from a /system_stats payload.

    A served frontend newer than the backend blank-screens the ComfyUI canvas
    (vue-i18n compile error, no menu; docs/TROUBLESHOOTING.md #19) while every
    backend API the pack uses keeps answering 200, so no other row catches it.
    The served version comes from argv's `--front-end-version`. An `@latest`
    spec resolves through `resolve_latest(owner_repo)` to the newest downloaded
    copy under `web_custom_versions`. Without the flag, it comes from the
    installed `comfyui-frontend-package`. Returns (status, detail); a non-OK
    detail says what to do."""
    required = (system or {}).get("required_frontend_version")
    argv = [str(a) for a in (system or {}).get("argv") or []]
    served = None
    source = "comfyui-frontend-package"
    moving_target = False
    if "--front-end-version" in argv:
        try:
            spec = argv[argv.index("--front-end-version") + 1]
        except IndexError:
            spec = ""
        owner_repo, _, tag = spec.rpartition("@")
        source = spec
        if tag == "latest":
            moving_target = True
            served = resolve_latest(owner_repo or spec) if resolve_latest else None
        elif tag:
            served = tag
    else:
        for pkg in (system or {}).get("comfy_package_versions") or []:
            if pkg.get("name") == "comfyui-frontend-package":
                served = pkg.get("installed")
                break
    if not required or not served:
        return (_WARN,
                f"could not determine {'required' if not required else 'served'} "
                f"frontend version (served source: {source}); if the canvas is "
                "blank, suspect frontend/backend skew (docs/TROUBLESHOOTING.md #19)")
    try:
        newer = _parse_version(served) > _parse_version(required)
        older = _parse_version(served) < _parse_version(required)
    except Exception:
        if served == required:
            newer = older = False
        else:
            return (_WARN,
                    f"serving frontend {served}, backend requires {required}, and "
                    "the versions cannot be compared on this host; if the canvas "
                    "is blank, treat this as skew (docs/TROUBLESHOOTING.md #19)")
    latest_note = (" Note: --front-end-version ...@latest re-downloads the newest "
                   "frontend at every launch, so the next frontend release can put it "
                   "ahead of the backend; pin an explicit version." if moving_target else "")
    if newer:
        return (_FAIL,
                f"serving frontend {served}, but this ComfyUI expects {required}. A "
                "frontend newer than the backend leaves the canvas blank (no menu, "
                "vue-i18n error). Fix it one of two ways: pin the launcher to "
                f"`--front-end-version Comfy-Org/ComfyUI_frontend@{required}`, or "
                "update ComfyUI (`git pull` in the checkout) so the backend targets "
                f"the newer frontend. Then relaunch.{latest_note}")
    if older:
        return (_WARN,
                f"serving frontend {served}, older than the backend's required "
                f"{required}; UI features may be missing. Fix: `pip install "
                f"comfyui-frontend-package=={required}` in ComfyUI's venv (or pin the "
                f"launcher flag to @{required}), then relaunch.{latest_note}")
    if moving_target:
        return (_WARN,
                f"frontend {served} matches the required {required} today, but the "
                f"launcher uses @latest.{latest_note}")
    return (_OK, f"served frontend {served} matches backend requirement ({source})")


def _resolve_latest_frontend(comfy_dir: str):
    """Resolver for `@latest` specs: ComfyUI downloads each version into
    web_custom_versions/<owner>_<repo>/<version>/. The newest-modified
    version directory is taken as the one the running instance serves."""

    def resolve(owner_repo: str):
        root = os.path.join(comfy_dir, "web_custom_versions",
                            owner_repo.replace("/", "_"))
        try:
            dirs = [(os.path.getmtime(os.path.join(root, d)), d)
                    for d in os.listdir(root)
                    if os.path.isdir(os.path.join(root, d))]
        except OSError:
            return None
        return max(dirs)[1] if dirs else None

    return resolve


def _frontend_skew_row() -> dict:
    """Probe every local ComfyUI instance for frontend skew, so one healthy
    instance cannot mask a skewed one."""
    import json
    import urllib.request

    verdicts: list[tuple[str, str, str]] = []   # (status, host:port, detail)
    for port, pid in _find_comfy_ports().items():
        candidate = f"127.0.0.1:{port}"
        try:
            with urllib.request.urlopen(
                    f"http://{candidate}/system_stats", timeout=2) as r:
                system = (json.load(r) or {}).get("system") or {}
        except Exception:
            continue
        if not system:
            continue
        argv = [str(a) for a in system.get("argv") or []]
        comfy_dir = ""
        if argv:
            argv0 = argv[0]
            if not os.path.isabs(argv0) and pid is not None:
                # `python main.py` from the checkout gives a relative argv[0]:
                # resolve it against that process's cwd to find its actual
                # frontend version and avoid misclassifying version skew.
                try:
                    import psutil

                    argv0 = os.path.join(psutil.Process(pid).cwd(), argv0)
                except Exception:
                    argv0 = ""
            if argv0:
                comfy_dir = os.path.dirname(os.path.abspath(argv0))
        status, detail = classify_frontend_skew(
            system, resolve_latest=_resolve_latest_frontend(comfy_dir))
        verdicts.append((status, candidate, detail))
    if not verdicts:
        return _row(_OK, "frontend/backend skew",
                    "no running ComfyUI found; start the driver and re-run "
                    "doctor to check the served frontend against the backend")
    rank = {_FAIL: 2, _WARN: 1, _OK: 0}
    worst = max(verdicts, key=lambda v: rank.get(v[0], 0))
    others = [f"{c}: {st}" for st, c, _ in verdicts if c != worst[1]]
    detail = f"{worst[1]}: {worst[2]}"
    if others:
        detail += f" (also probed {', '.join(others)})"
    return _row(worst[0], "frontend/backend skew", detail)


def _mesh_health_row() -> dict:
    """The driver's own mesh state, read from the driver over HTTP.

    No host probe can see it; `cli/mesh_health_row.py` says why.
    """
    kind, detail = mesh_health_row.doctor_verdict(mesh_health_row.fetch_mesh_block())
    return _row({"fail": _FAIL, "warn": _WARN}.get(kind, _OK), "mesh health", detail)


def _failures(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["status"] == _FAIL]


def doctor_exit(config: ClusterConfig | None, as_json: bool = False) -> int:
    """Run every check once for the exit code its rows justify.

    Both modes read one set of rows; `--json` only redirects the prose. A run
    with no FAIL row exits `UNKNOWN_EXIT` when a critical probe did not answer.
    """
    rows = (json_report.quietly(lambda: _doctor_rows(config)) if as_json
            else _doctor_rows(config))
    failures = _failures(rows)
    if as_json:
        json_report.emit(json_report.doctor_payload(rows, len(failures)))
    return probe_certainty.exit_code(rows, len(failures))


def run_doctor(config: ClusterConfig | None, as_json: bool = False) -> bool:
    """Return True when no check failed and every critical probe answered.

    Commands that change cluster state must refuse to proceed on missing critical
    observations.
    """
    return doctor_exit(config, as_json=as_json) == 0


def _doctor_rows(config: ClusterConfig | None) -> list[dict]:
    """Run checks in report order and print each result as it arrives."""
    rows: list[dict] = []
    permissions = doctor_permissions.config_permissions_row(config)
    rows.append(_row(permissions["status"], permissions["name"], permissions["detail"]))

    print("driver checks:")
    # The loopback-advertisement trap (docs/TROUBLESHOOTING.md #1), judged
    # against the loaded config: a client_bind on the fabric IP answers it.
    try:
        this_host = socket.gethostname()
        ok, text = hostname_row(this_host, socket.gethostbyname(this_host), config)
    except OSError as exc:
        ok, text = unresolved_row(config, exc)
    rows.append(_row(_OK if ok else _WARN, "hostname resolution", text))

    rows.append(_frontend_skew_row())

    try:
        from importlib.metadata import version

        installed = version("torchmonarch")
        status = _OK if installed == TORCHMONARCH_PIN else _FAIL
        rows.append(_row(status, "torchmonarch",
                         f"installed {installed}, pin {TORCHMONARCH_PIN}"
                         + ("" if installed == TORCHMONARCH_PIN else "; `dgxm update` (not --verify) installs the "
                            "checkout's pin. Change the pin only with a CHANGELOG entry and a test battery run")))
    except Exception as exc:
        rows.append(_row(_FAIL, "torchmonarch", f"not importable: {exc!r}"))

    # A `pip install` in a venv other than ComfyUI's pulls PyPI's CPU-only torch
    # (aarch64 has no CUDA wheels there), and the first render fails far from
    # the cause.
    try:
        import torch

        if torch.cuda.is_available():
            rows.append(_row(_OK, "torch CUDA", f"{torch.__version__}, {torch.cuda.device_count()} device(s)"))
        else:
            rows.append(_row(_FAIL, "torch CUDA",
                             f"torch {torch.__version__} has no CUDA in this interpreter. If it is not "
                             "the venv that runs ComfyUI, run the install with ComfyUI's Python (docs/INSTALL.md)"))
    except Exception as exc:
        rows.append(_row(_FAIL, "torch CUDA", f"torch not importable: {exc!r}"))

    # These rows check NCCL_PROTO in the driver env (inherited by local-mode procs) and the
    # fabric profile (applied inside cluster actor procs); the per-host loop below checks the
    # worker-host shell env. Loading cluster.toml refuses the key in any [fabric.*] table and
    # no shipped profile sets it, so only a ClusterConfig built in code reaches the fabric rows.
    proto = os.environ.get("NCCL_PROTO", "")
    normalized_proto = _normalize_nccl_proto(proto)
    if _uses_nccl_ll(proto):
        rows.append(_row(_FAIL, "NCCL_PROTO",
                         f"{normalized_proto} includes LL, the FSDP-killer; "
                         "unset it (docs/VALIDATION.md)"))
    else:
        rows.append(_row(_OK, "NCCL_PROTO", normalized_proto or "unset (correct default)"))
    if config is not None:
        fabric_proto = config.resolved_fabric_env().get("NCCL_PROTO", "")
        normalized_fabric_proto = _normalize_nccl_proto(fabric_proto)
        if _uses_nccl_ll(fabric_proto):
            rows.append(_row(_FAIL, "fabric NCCL_PROTO",
                             f"[fabric.{config.fabric_profile}] sets NCCL_PROTO={normalized_fabric_proto}, "
                             "which includes LL, the FSDP-killer. Worker setup refuses any "
                             "NCCL_PROTO in a fabric profile; remove it (docs/VALIDATION.md)"))
        elif fabric_proto:
            rows.append(_row(_WARN, "fabric NCCL_PROTO",
                             f"set to {normalized_fabric_proto!r} in the fabric profile, which worker setup refuses"))

    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                meminfo[k] = int(v.strip().split()[0])
        avail_gib = meminfo.get("MemAvailable", 0) / 2**20
        total_gib = meminfo.get("MemTotal", 1) / 2**20
        cached_gib = meminfo.get("Cached", 0) / 2**20
        if avail_gib < total_gib * 0.35 and cached_gib > total_gib * 0.3:
            rows.append(_row(_WARN, "memory headroom",
                             f"{avail_gib:.0f}/{total_gib:.0f} GiB available; page cache holds {cached_gib:.0f} GiB. "
                             "On unified memory ComfyUI plans loads against free memory, so drop the "
                             "page cache before a large load: `sync && echo 3 | sudo tee /proc/sys/vm/drop_caches`"))
        else:
            rows.append(_row(_OK, "memory headroom", f"{avail_gib:.0f}/{total_gib:.0f} GiB available"))
    except Exception as exc:
        rows.append(_row(_WARN, "memory headroom", repr(exc)))

    ib_dir = "/sys/class/infiniband"
    if os.path.isdir(ib_dir):
        rows.append(_row(_OK, "rdma rails", ", ".join(sorted(os.listdir(ib_dir))) or "none"))
    else:
        rows.append(_row(_OK, "rdma rails", "no ibverbs devices (TCP fallback applies)"))
    if config is not None:
        from ..config import fixup_fabric_ifaces

        _, rail_note = fixup_fabric_ifaces(config.resolved_fabric_env())
        if rail_note:
            rows.append(_row(_WARN, "fabric ifaces",
                             rail_note + "; each worker repeats this check at setup. To pin the interface "
                             f"instead, name one that is up on every host in [fabric.{config.fabric_profile}]"))
        elif config.resolved_fabric_env().get("NCCL_SOCKET_IFNAME"):
            rows.append(_row(_OK, "fabric ifaces",
                             f"{config.resolved_fabric_env()['NCCL_SOCKET_IFNAME']} is up on this host"))

    rows.append(_transport_security_row(config))

    # These rows must run in local mode too, so they stay above the early return
    # below: a bad safetensors build, a stale Sage ABI or a missing Python.h
    # otherwise shows only at the first render.
    print("local health checks:")
    rows.extend(_spark_health_rows(config))

    if config is None or not config.hosts:
        print("cluster checks: skipped (no cluster.toml; local mode needs none)")
        print(f"doctor: {len(rows)} checks, {len(_failures(rows))} failures")
        return rows

    print(f"cluster checks ({config.source}):")
    if not config.client_bind:
        rows.append(_row(_FAIL, "client_bind", "missing; attach refuses without it, since the driver may advertise "
                         "loopback, which ends in MESH_ATTACH_CONFIG_TIMEOUT. Set it to tcp://<driver fabric IP>:0"))
    else:
        rows.append(_row(_OK, "client_bind", config.client_bind))

    if shutil.which("rsync") is None:
        rows.append(_row(_FAIL, "rsync", "not installed; `dgxm up` cannot push the package (sync is on by default)"))

    # NCCL master port occupancy outside a render (docs/TROUBLESHOOTING.md #3):
    # a listener here is usually a crashed client's lingering TCPStore.
    master_addr = config.resolved_master_addr()
    occupancy = _tcp_probe(format_tcp_address(master_addr, config.nccl_master_port), timeout=1.0)
    if occupancy:
        rows.append(_row(_WARN, "nccl master port",
                         f"{master_addr}:{config.nccl_master_port} is accepting connections outside a "
                         "render, likely a stale TCPStore; `dgxm restart` clears it"))
    elif occupancy is None:
        # The connect timed out. Free and unreachable look the same from here,
        # so the row says it read nothing rather than claim the port is free.
        rows.append(_row(_WARN, "nccl master port",
                         f"unobserved: {master_addr}:{config.nccl_master_port} did not answer "
                         "in 1s; whether a stale TCPStore holds it is unknown"))
    else:
        rows.append(_row(_OK, "nccl master port", f"{master_addr}:{config.nccl_master_port} free"))

    # Driver-side failure state can block the next render even when all
    # Worker service checks pass.
    rows.append(_mesh_health_row())

    driver_torch = ""
    try:
        import torch

        driver_torch = torch.__version__
    except Exception:
        pass

    remote_torch: dict[str, str] = {}
    sol_install: dict[str, dict[str, str]] = {}
    nccl_libs: dict[str, dict[str, str]] = {}
    for host in config.hosts:
        health = passive_worker_health(config, host, runner=run_on_host)
        rows.append(_row(*worker_service_row(host.name, host.address, health)))

        # One heredoc carries every remote probe. Add a probe as a new print
        # line: each consumer below selects its own line by prefix.
        comfy_dir = host.comfy_dir or config.comfy_dir or "~/ComfyUI"
        comfy_dir_literal = repr(comfy_dir)
        wanted_iface_literal = repr(
            config.resolved_fabric_env().get("NCCL_SOCKET_IFNAME", ""))
        script = f"""
{_pybin_shell(config.python_bin)}
"$PYBIN" - <<'EOF'
import glob, importlib.metadata as md, os
try:
    tm = md.version("torchmonarch")
except Exception:
    tm = "MISSING"
try:
    import torch
    tv = torch.__version__
except Exception:
    tv = "MISSING"
proto = ",".join(p.strip().upper() for p in os.environ.get("NCCL_PROTO", "").split(",") if p.strip())
print(f"torchmonarch={{tm}} torch={{tv}} comfy={{os.path.isdir(os.path.expanduser({comfy_dir_literal}))}} nccl_proto={{proto or 'unset'}}")
want = {wanted_iface_literal}
try:
    iface = open(f"/sys/class/net/{{want}}/operstate").read().strip() if want else "unset"
except OSError:
    iface = "absent"
active = 0
for state_path in glob.glob("/sys/class/infiniband/*/ports/*/state"):
    try:
        active += int("ACTIVE" in open(state_path).read())
    except OSError:
        pass
print(f"fabric_iface={{iface}} rdma_active={{active}}")
{sol_attn_row.PROBE_SOURCE}
{doctor_nccl.PROBE_SOURCE}
{doctor_probes.SWAP_PROBE_SOURCE}
{listener_generation.probe_source(host.address)}
EOF
"""
        sol_install[host.name] = {}  # Unknown until this host's probe answers.
        try:
            result = run_on_host(config, host, script, timeout=60)
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "no output").strip().splitlines()[-1]
                rows.append(_row(
                    _FAIL, f"{host.name} env",
                    f"remote check exited {result.returncode}: {detail}"))
                # Report unknown listener generation even when the host check fails.
                rows.append(_row(*listener_generation.unobserved_row(
                    host.name, f"remote check exited {result.returncode}")))
                continue
            lines = result.stdout.strip().splitlines()
            line = next((line for line in lines if line.startswith("torchmonarch=")), "no output")
            env_fields = _env_tokens(line)
            remote_monarch = env_fields.get("torchmonarch")
            bad = (
                remote_monarch in (None, "MISSING")
                or env_fields.get("torch") in (None, "MISSING")
                or env_fields.get("comfy") != "True"
            )
            version_mismatch = remote_monarch != TORCHMONARCH_PIN
            status = _FAIL if bad or version_mismatch else _OK
            detail = line + (
                "; version skew vs driver pin" if version_mismatch and not bad else "")
            rows.append(_row(status, f"{host.name} env", detail))
            fabric_line = next((line for line in lines if line.startswith("fabric_iface=")), "")
            fields = dict(token.split("=", 1) for token in fabric_line.split() if "=" in token)
            iface_state = fields.get("fabric_iface", "unknown")
            try:
                active_rails = int(fields.get("rdma_active", "0"))
            except ValueError:
                active_rails = 0
            rdma_expected = config.fabric_profile != "single-node"
            if iface_state in ("absent", "down", "unknown"):
                fabric_status = _WARN if active_rails else (_FAIL if rdma_expected else _WARN)
                fabric_detail = (
                    f"configured socket interface is {iface_state}; {active_rails} ACTIVE RDMA rail(s) "
                    + ("let the worker rewrite its fabric env at setup" if active_rails else "detected")
                )
                rows.append(_row(fabric_status, f"{host.name} fabric", fabric_detail))
            elif rdma_expected and active_rails == 0:
                rows.append(_row(_FAIL, f"{host.name} fabric",
                                 f"interface state {iface_state}; no ACTIVE RDMA rails"))
            else:
                rows.append(_row(_OK, f"{host.name} fabric",
                                 f"interface {iface_state}; {active_rails} ACTIVE RDMA rail(s)"))
            if env_fields.get("torch") not in (None, "MISSING"):
                remote_torch[host.name] = env_fields["torch"]
            sol_line = next((ln for ln in lines if ln.startswith(sol_attn_row.LINE_PREFIX)), "")
            sol_install[host.name] = _env_tokens(sol_line) if sol_line else {}
            nccl_libs[host.name] = doctor_nccl.fields(lines)
            swap_line = next((ln for ln in lines if ln.startswith(doctor_probes.SWAP_LINE_PREFIX)), "")
            rows.append(_row(*doctor_probes.remote_swap_row(host.name, swap_line)))
            rows.append(_row(*listener_generation.doctor_row(
                host.name, listener_generation.select(lines))))
            remote_proto = env_fields.get("nccl_proto", "")
            if _uses_nccl_ll(remote_proto):
                rows.append(_row(_FAIL, f"{host.name} NCCL_PROTO",
                                 f"{remote_proto} includes LL in the worker host shell env. The worker "
                                 "loop drops NCCL_PROTO before it starts actors, but other NCCL jobs started from "
                                 "that shell inherit it, and LL is the FSDP-killer: unset it (docs/VALIDATION.md)"))
        except Exception as exc:
            blind, reason = probe_certainty.blind_row(_FAIL, exc)
            rows.append(_row(blind, f"{host.name} env", f"ssh failed: {exc!r}",
                             reason, critical=True))
            rows.append(_row(*listener_generation.unobserved_row(
                host.name, f"ssh failed: {type(exc).__name__}")))

    # torch versions must match across hosts and the driver (docs/DESIGN.md
    # sections 6.1 and 6.3).
    versions = {v for v in remote_torch.values() if v and v != "MISSING"}
    if driver_torch:
        versions.add(driver_torch)
    if len(versions) > 1:
        detail = ", ".join(
            f"{name}={ver}" for name, ver in [("driver", driver_torch), *sorted(remote_torch.items())] if ver)
        rows.append(_row(_FAIL, "torch version match", f"skew across hosts: {detail}"))
    elif remote_torch:
        rows.append(_row(_OK, "torch version match", next(iter(versions), "unknown")))

    rows.append(_row(*doctor_nccl.doctor_row(nccl_libs, [host.name for host in config.hosts])))

    # Sol-attn requires sequence parallelism. The early return above excludes
    # configless, single-rank runs from this mesh-wide installation check.
    rows.append(_row(*sol_attn_row.doctor_row(sol_install)))

    print(f"doctor: {len(rows)} checks, {len(_failures(rows))} failures")
    return rows


def _spark_health_rows(config: ClusterConfig | None) -> list[dict]:
    """Rows this box can check on itself. docs/TROUBLESHOOTING.md, "The Spark failure catalog",
    describes the GB10 failure modes among them."""
    import glob
    import subprocess
    import sys

    rows: list[dict] = []
    multi_host = bool(config and len(config.hosts) > 1)
    rdma_expected = bool(multi_host and config and config.fabric_profile != "single-node")

    # Unexpected shutdowns. The Spark power-delivery fault hard-resets the box
    # and logs nothing; what it leaves is a reboot record with no shutdown
    # record before it.
    try:
        out = subprocess.run(["last", "-Fxn30", "shutdown", "reboot"],
                             capture_output=True, text=True, timeout=10).stdout
        events = [line.split()[0] for line in out.splitlines() if line.strip()
                  and line.split()[0] in ("reboot", "shutdown")]
        # walk newest-first: every reboot should be preceded (below it) by a shutdown
        crashes = 0
        for i, ev in enumerate(events):
            if ev == "reboot" and (i + 1 >= len(events) or events[i + 1] != "shutdown"):
                crashes += 1
        if crashes > 1:  # the oldest reboot in the window often lacks its pair
            rows.append(_row(_WARN, "unexpected shutdowns",
                             f"{crashes - 1} reboot(s) without a clean shutdown in the last 30 boot records. "
                             "On a Spark this matches the power-delivery fault, a hard power-off that logs nothing. "
                             "Doctor runs no load test, so it cannot tell if the GPU stayed stuck at a low clock; "
                             "a stuck GPU needs a cold power cycle, not a reboot (TROUBLESHOOTING, PD-stuck state)"))
        else:
            rows.append(_row(_OK, "unexpected shutdowns", "none in recent boot records"))
    except Exception:
        rows.append(_row(_WARN, "unexpected shutdowns",
                         "unobserved: the `last` boot-record probe did not complete, so the "
                         "crash history is unknown. Read it by hand: `last -Fx shutdown reboot`"))

    # Check safe_open's backend parameter: 0.8.0rc0 passes a simple version
    # check but lacks it, causing TypeError on model load.
    try:
        import inspect

        import safetensors

        has_backend = "backend" in inspect.signature(safetensors.safe_open).parameters
        if has_backend:
            rows.append(_row(_OK, "safetensors pread",
                             f"{safetensors.__version__}: safe_open has the backend param"))
        else:
            rows.append(_row(_FAIL, "safetensors pread",
                             f"{safetensors.__version__} has no safe_open backend param "
                             "(older than 0.8.0 or a pre-release), so loads stay on mmap or crash. Install "
                             "the final release: pip install 'safetensors>=0.8.0' (not an rc)"))
    except Exception as exc:
        rows.append(_row(_WARN, "safetensors pread", f"probe failed: {exc!r}"))

    # SageAttention import state and, on an idle GPU, a bounded kernel probe
    # that catches a stale-ABI or shadowed build (docs/TROUBLESHOOTING.md #89).
    rows.append(_row(*doctor_probes.sage_kernel_row(sys.executable)))

    # Sage-to-torch fallbacks counted from the driver log: whether sage ran,
    # not only whether it was selected.
    rows.append(_row(*doctor_probes.sage_fallback_row()))

    # Unified memory is the box's only memory, so an active swap device pages
    # weights under a near-limit overrun instead of a clean OOM kill
    # (docs/INSTALL.md). The row warns and never fails (`swap_row`).
    rows.append(_row(*doctor_probes.swap_row()))

    # RDMA rail link state, multi-host only. A rail that negotiated down stays
    # down until a reboot (a CX7 firmware link-training bug); no software fix
    # exists, so do not spend time on driver restarts.
    if rdma_expected:
        try:
            active = []
            down_phys_up = []
            for state_path in glob.glob("/sys/class/infiniband/*/ports/*/state"):
                dev = state_path.split("/")[4]
                state = open(state_path).read().strip()
                phys = open(state_path.replace("/state", "/phys_state")).read().strip()
                if "ACTIVE" in state:
                    active.append(dev)
                elif "LinkUp" in phys:
                    down_phys_up.append(dev)
            if len(active) >= 2:
                rows.append(_row(_OK, "RDMA rails", f"{len(active)} ACTIVE ({', '.join(sorted(active))})"))
            elif active:
                rows.append(_row(_WARN, "RDMA rails",
                                 f"only {active} ACTIVE; a cabled rail is down. A CX7 rail that "
                                 "failed link training stays down until the affected box reboots; "
                                 "no software fix exists"))
            elif glob.glob("/sys/class/infiniband/*"):
                rows.append(_row(_FAIL, "RDMA rails",
                                 "no ACTIVE rails; check the QSFP cables. If they are seated and "
                                 "phys_state reads LinkUp, reboot the box (CX7 link-training bug)"))
            else:
                rows.append(_row(_FAIL, "RDMA rails",
                                 "no RDMA devices visible; if lspci shows the PCI bridge but no "
                                 "Mellanox NIC, the CX7 failed PCIe retraining after an update, and "
                                 "only a full power cycle (not a reboot) recovers it"))
        except Exception as exc:
            rows.append(_row(_WARN, "RDMA rails", f"probe failed: {exc!r}"))

    # NCCL_P2P_DISABLE breaks cross-Spark NCCL over ConnectX; require it unset
    # for multi-host operation.
    p2p = os.environ.get("NCCL_P2P_DISABLE")
    if p2p and multi_host:
        rows.append(_row(_FAIL, "NCCL_P2P_DISABLE",
                         f"set to {p2p!r}; unset it before any cross-Spark NCCL work"))
    elif p2p:
        rows.append(_row(_WARN, "NCCL_P2P_DISABLE",
                         f"set to {p2p!r}; fine on one box, but unset it before pairing Sparks"))
    else:
        rows.append(_row(_OK, "NCCL_P2P_DISABLE", "unset (correct)"))

    # Report the power-load probe as not run. Synthetic matmul could contend
    # with or OOM a render and cannot validate cluster setup.
    try:
        import torch

        if torch.cuda.is_available():
            rows.append(_row(
                _WARN,
                "GPU clock under load",
                "synthetic CUDA load NOT RUN: doctor stays passive and gives no PD-state verdict",
            ))
        else:
            rows.append(_row(
                _OK,
                "GPU clock under load",
                "skipped: no CUDA device on this host",
            ))
    except Exception:
        rows.append(_row(_WARN, "GPU clock under load",
                         "unobserved: the torch CUDA probe did not complete, so this row "
                         "cannot say whether the box has a usable CUDA device"))

    # Missing python3.x-dev headers make Sage/Triton JIT compilation fail;
    # ComfyUI can then fall back to slower torch attention without stopping.
    try:
        import sysconfig

        header = os.path.join(sysconfig.get_paths()["include"], "Python.h")
        if os.path.exists(header):
            rows.append(_row(_OK, "python dev headers", "Python.h present (triton JIT can compile)"))
        else:
            rows.append(_row(_FAIL, "python dev headers",
                             "Python.h missing; triton and sage kernels fail to build on every call, "
                             "and ComfyUI logs an error and runs torch attention instead (up to ~20x "
                             "slower). Install the python3.x-dev package for this interpreter's version"))
    except Exception as exc:
        rows.append(_row(_WARN, "python dev headers", f"probe unavailable ({exc!r})"))

    # The Spark fan curve tracks CPU load, so the community-identified 'temp6'
    # board sensor can sit at 90 C+ without a ramp. `hwmon.py` reads the
    # kernel's hwmon tree, so the check needs no package.
    try:
        hazard, detail = board_hotspot_verdict()
        rows.append(_row(_WARN if hazard else _OK, "board hotspots", detail))
    except Exception as exc:
        rows.append(_row(_WARN, "board hotspots", f"probe unavailable ({exc!r})"))

    # On unified memory every other GPU process shares the render pool. The
    # row only lists them.
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout
        procs = [line.strip() for line in out.splitlines() if line.strip()]
        if procs:
            rows.append(_row(_OK, "co-resident GPU procs",
                             f"{len(procs)}: " + "; ".join(p[:60] for p in procs[:4])))
        else:
            rows.append(_row(_OK, "co-resident GPU procs", "none"))
    except Exception:
        rows.append(_row(_WARN, "co-resident GPU procs",
                         "unobserved: the `nvidia-smi` compute-app query did not complete, so "
                         "who else shares the render pool is unknown"))

    # A worker actor that outlived its client holds its retained pool until it
    # exits: render pool nobody can reclaim on unified memory
    # (docs/TROUBLESHOOTING.md #51). A process counts from 4 GiB resident
    # (`actor_reaper.FAT_ANON_GIB`). WARN, not FAIL, because a sweep clears it.
    try:
        from . import actor_reaper

        report = actor_reaper.orphan_report()
        if report.orphans:
            named = "; ".join(f"pid {p.pid} {p.rss_gib:.1f} GiB resident ({p.gib:.1f} pooled) "
                              f"{p.signal_text}, {p.dwell_text}" for p in report.orphans[:4])
            rows.append(_row(
                _WARN, "orphaned actor procs",
                f"{len(report.orphans)} monarch actor process(es) hold "
                f"{sum(p.rss_gib for p in report.orphans):.1f} GiB with no live owner ({named}). "
                "That memory returns to the OS only when the process exits. This row sees only "
                "the box doctor runs on. Fix: `dgxm reap` sweeps every host in cluster.toml and "
                "this box; `dgxm restart` sweeps every configured host. A process that survives "
                "a sweep is stuck in the kernel: capture its pid and the worker journal "
                "(TROUBLESHOOTING #51)"))
        else:
            rows.append(_row(_OK, "orphaned actor procs",
                             f"none ({report.tracked} tracked, {report.attached} attached)"))
    except Exception:
        rows.append(_row(_WARN, "orphaned actor procs",
                         "unobserved: the orphan scan did not complete, so any pool held by an "
                         "actor with no live owner is unknown. Run `dgxm reap` to sweep any that exist"))

    return rows
