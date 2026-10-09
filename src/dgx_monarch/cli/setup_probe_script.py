"""Read-only probe script that guided setup runs on each host."""

from __future__ import annotations

import base64
import json
import shlex
from collections.abc import Sequence

MARKER = "DGXM_SETUP_PROBE="


def build_probe_script(
    python_bin: str,
    comfy_dir: str,
    artifacts: Sequence[str],
    *,
    worker_address: str = "",
) -> str:
    """Return a script that lifecycle.run_on_host pipes into ``bash -s``; every input is quoted or base64-encoded."""
    encoded = base64.b64encode(json.dumps(list(artifacts)).encode("utf-8")).decode("ascii")
    unavailable = json.dumps(_unavailable_payload(len(artifacts)), separators=(",", ":"))
    return f'''set -u
PYBIN={shlex.quote(python_bin)}
case "$PYBIN" in "~/"*) PYBIN="$HOME/${{PYBIN#\\~/}}";; "~") PYBIN="$HOME";; esac
COMFY={shlex.quote(comfy_dir)}
case "$COMFY" in "~/"*) COMFY="$HOME/${{COMFY#\\~/}}";; "~") COMFY="$HOME";; esac
if [ ! -x "$PYBIN" ] && ! command -v -- "$PYBIN" >/dev/null 2>&1; then
  printf '%s\n' '{MARKER}{unavailable}'
  exit 0
fi
export DGXM_SETUP_COMFY="$COMFY"
export DGXM_SETUP_ARTIFACTS={encoded}
export DGXM_SETUP_WORKER_ADDRESS={shlex.quote(worker_address)}
export PYTHONDONTWRITEBYTECODE=1
exec "$PYBIN" -I -B - <<'DGXM_PY'
import base64, hashlib, importlib.metadata, ipaddress, json, os, pathlib, shutil, subprocess, sys

os.environ["GIT_OPTIONAL_LOCKS"] = "0"

def run(argv, cwd=None, env=None):
    try:
        return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None

def read(path):
    try:
        return pathlib.Path(path).read_text(errors="replace").strip()
    except OSError:
        return ""

def endpoint(value):
    try:
        if value.startswith("tcp://"):
            value = value[len("tcp://"):]
        if value.startswith("["):
            host, port = value[1:].split("]:", 1)
        else:
            host, port = value.rsplit(":", 1)
        return ipaddress.ip_address(host.split("%", 1)[0]), int(port)
    except (AttributeError, ValueError):
        return None

def exact_worker_process(address):
    if not address:
        return None
    unknown = False
    try:
        entries = tuple(pathlib.Path("/proc").iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != os.getuid():
                continue
            argv = (entry / "cmdline").read_bytes().split(b"\\0")
        except FileNotFoundError:
            continue
        except OSError:
            unknown = True
            continue
        values = [item.decode("utf-8", errors="replace") for item in argv if item]
        module = any(values[index:index + 2] == ["-m", "dgx_monarch.cli.worker_loop"] for index in range(len(values) - 1))
        target = any(values[index:index + 2] == ["--address", address] for index in range(len(values) - 1))
        if module and target:
            return True
    return None if unknown else False

def exact_listener(address):
    expected = endpoint(address)
    if expected is None:
        return None
    ip, port = expected
    packed = (b"".join(ip.packed[index:index + 4][::-1] for index in range(0, 16, 4))
              if ip.version == 6 else ip.packed[::-1])
    suffix = ":" + format(port, "04X")
    tables = [("/proc/net/tcp6" if ip.version == 6 else "/proc/net/tcp",
               {{packed.hex().upper() + suffix, "0" * len(packed.hex()) + suffix}})]
    if ip.version == 4:
        mapped = ipaddress.IPv6Address("::ffff:" + str(ip)).packed
        mapped = b"".join(mapped[index:index + 4][::-1] for index in range(0, 16, 4))
        tables.append(("/proc/net/tcp6", {{mapped.hex().upper() + suffix, "0" * 32 + suffix}}))
    unknown = False
    for table, endpoints in tables:
        try:
            rows = pathlib.Path(table).read_text().splitlines()[1:]
        except OSError:
            unknown = True
            continue
        if any(len(row.split()) > 3 and row.split()[1] in endpoints and row.split()[3] == "0A" for row in rows):
            return True
    return None if unknown else False

payload = json.loads({json.dumps(json.dumps(_unavailable_payload(len(artifacts)), separators=(",", ":")))})
payload["python_available"] = True
payload["python_version"] = ".".join(map(str, sys.version_info[:3]))
try:
    import torch
    payload["torch_version"] = str(torch.__version__)
    payload["cuda_available"] = bool(torch.cuda.is_available())
    if payload["cuda_available"]:
        count = int(torch.cuda.device_count())
        payload["gpu_count"] = count
        props = [torch.cuda.get_device_properties(index) for index in range(count)]
        payload["gpu_models"] = [str(prop.name) for prop in props]
        integrated = [getattr(prop, "is_integrated", None) for prop in props]
        payload["integrated"] = bool(integrated[0]) if integrated and None not in integrated and len(set(integrated)) == 1 else None
except Exception:
    pass
try:
    payload["torchmonarch_version"] = importlib.metadata.version("torchmonarch")
except importlib.metadata.PackageNotFoundError:
    pass

root = pathlib.Path(os.environ["DGXM_SETUP_COMFY"]).expanduser()
payload["comfy_exists"] = root.is_dir()
payload["comfy_runtime_marker"] = payload["comfy_exists"] and (root / "comfy" / "sd.py").is_file()
if payload["comfy_exists"]:
    git_env = {{key: value for key, value in os.environ.items() if not key.startswith("GIT_")}}
    git_env.update({{"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull, "GIT_OPTIONAL_LOCKS": "0"}})
    git = [shutil.which("git") or "", "-c", "core.fsmonitor=false"]
    top = run([*git, "rev-parse", "--show-toplevel"], cwd=root, env=git_env) if git[0] else None
    payload["comfy_git"] = bool(top and top.returncode == 0 and pathlib.Path(top.stdout.strip()).resolve() == root.resolve())
    if payload["comfy_git"]:
        status = run([*git, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"], cwd=root, env=git_env)
        assume = run([*git, "ls-files", "-v", "-z"], cwd=root, env=git_env)
        typed = run([*git, "ls-files", "-t", "-z"], cwd=root, env=git_env)
        head = run([*git, "rev-parse", "HEAD"], cwd=root, env=git_env)
        git_rows = (status, assume, typed)
        if all(row is not None and row.returncode == 0 for row in git_rows):
            hidden = any(line[:1].islower() for line in assume.stdout.split("\\0")) or any(line.startswith("S ") for line in typed.stdout.split("\\0"))
            payload["comfy_dirty"] = bool(status.stdout.strip()) or hidden
            payload["comfy_only_missing_examples"] = False
            records = status.stdout.split("\\0")
            missing = [record[3:] for record in records[:-1]]
            examples = {{"input/example.png", "output/_output_images_will_be_put_here"}}
            if not hidden and records[-1] == "" and missing and len(set(missing)) == len(missing) and all(record.startswith(" D ") for record in records[:-1]) and set(missing) <= examples:
                staged = run([*git, "diff", "--cached", "--name-only", "-z", "HEAD", "--"], cwd=root, env=git_env)
                tracked = run([*git, "ls-tree", "-z", "HEAD", "--", *sorted(missing)], cwd=root, env=git_env)
                expected = {{}}
                if tracked is not None and tracked.returncode == 0:
                    for entry in tracked.stdout.split("\\0")[:-1]:
                        meta, name = entry.split("\t", 1)
                        mode, kind, oid = meta.split()
                        if mode in {{"100644", "100755"}} and kind == "blob":
                            expected[name] = oid
                absent = all(not os.path.lexists(root / name) and (root / name).parent.is_dir() and not (root / name).parent.is_symlink() for name in missing)
                payload["comfy_only_missing_examples"] = bool(staged is not None and staged.returncode == 0 and not staged.stdout and set(expected) == set(missing) and absent)
        value = head.stdout.strip().lower() if head and head.returncode == 0 else ""
        payload["comfy_commit"] = value if len(value) == 40 and all(c in "0123456789abcdef" for c in value) else None

ib = pathlib.Path("/sys/class/infiniband")
layers = []
interfaces = 0
if ib.is_dir():
    for device in sorted(ib.iterdir(), key=lambda item: item.name.lower()):
        interfaces += 1
        ports = device / "ports"
        if ports.is_dir():
            for port in sorted(ports.iterdir(), key=lambda item: item.name):
                if not read(port / "state").startswith("4:"):
                    continue
                layer = read(port / "link_layer")
                layers.append(layer if layer in {{"Ethernet", "InfiniBand"}} else "Unknown")
payload["fabric_interface_count"] = interfaces
payload["link_layers"] = sorted(set(layers))
payload["rsync_available"] = shutil.which("rsync") is not None
systemd = run(["systemctl", "--user", "show-environment"])
payload["systemd_user_available"] = bool(systemd and systemd.returncode == 0)
unit = pathlib.Path.home() / ".config/systemd/user/dgxm-worker.service"
shown = run(["systemctl", "--user", "show", "dgxm-worker.service",
             "--property=LoadState", "--property=ActiveState", "--no-pager"])
loaded = None
if shown is not None and shown.returncode == 0:
    fields = dict(line.partition("=")[::2] for line in shown.stdout.splitlines() if "=" in line)
    load_state = fields.get("LoadState")
    active_state = fields.get("ActiveState")
    loaded = False if load_state == "not-found" else True if load_state == "loaded" else None
    if active_state == "inactive":
        payload["systemd_service_active"] = False
    elif active_state in {"active", "activating", "reloading", "deactivating", "failed"}:
        payload["systemd_service_active"] = True
unit_present = os.path.lexists(unit)
payload["service_installed"] = True if unit_present or loaded is True else False if loaded is False else None
address = os.environ["DGXM_SETUP_WORKER_ADDRESS"]
payload["worker_process_active"] = exact_worker_process(address)
payload["worker_listener_active"] = exact_listener(address)
activity = [payload["systemd_service_active"], payload["worker_process_active"], payload["worker_listener_active"]]
payload["service_active"] = True if True in activity else (False if all(item is False for item in activity) else None)
linger = run(["loginctl", "show-user", str(os.geteuid()), "--property=Linger", "--value"])
if linger is not None and linger.returncode == 0 and linger.stdout.strip() in {"yes", "no"}:
    payload["linger_enabled"] = linger.stdout.strip() == "yes"

requested = json.loads(base64.b64decode(os.environ["DGXM_SETUP_ARTIFACTS"]).decode("utf-8"))
payload["artifacts"] = []
try:
    resolved_root = root.resolve(strict=True)
except OSError:
    resolved_root = None
for ordinal, relative in enumerate(requested, 1):
    row = {{"ordinal": ordinal, "exists": False, "regular": False, "size": None, "sha256": None}}
    try:
        candidate = (root / relative).resolve(strict=True)
        inside = False
        if resolved_root is not None:
            try:
                candidate.relative_to(resolved_root)
                inside = True
            except ValueError:
                pass
        row["exists"] = inside and candidate.exists()
        row["regular"] = inside and candidate.is_file()
        if row["regular"]:
            digest = hashlib.sha256()
            with candidate.open("rb") as handle:
                before = os.fstat(handle.fileno())
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                after = os.fstat(handle.fileno())
            final = candidate.stat()
            identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            if identity(before) != identity(after) or identity(after) != identity(final):
                raise OSError("artifact changed")
            row["size"] = before.st_size
            row["sha256"] = digest.hexdigest()
    except OSError:
        pass
    payload["artifacts"].append(row)
print("{MARKER}" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
DGXM_PY
'''


def _unavailable_payload(artifact_count: int) -> dict[str, object]:
    return {
        "schema": 1,
        "python_available": False,
        "python_version": None,
        "torch_version": None,
        "cuda_available": None,
        "gpu_count": None,
        "integrated": None,
        "gpu_models": [],
        "torchmonarch_version": None,
        "comfy_exists": None,
        "comfy_runtime_marker": None,
        "comfy_git": None,
        "comfy_dirty": None,
        "comfy_only_missing_examples": None,
        "comfy_commit": None,
        "fabric_interface_count": None,
        "link_layers": [],
        "rsync_available": None,
        "systemd_user_available": None,
        "linger_enabled": None,
        "service_installed": None,
        "systemd_service_active": None,
        "worker_process_active": None,
        "worker_listener_active": None,
        "service_active": None,
        "artifacts": [
            {"ordinal": ordinal, "exists": False, "regular": False, "size": None, "sha256": None}
            for ordinal in range(1, artifact_count + 1)
        ],
    }
