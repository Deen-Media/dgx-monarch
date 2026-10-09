"""Keep the update controller independent of the checkout it updates."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..config import ClusterConfig, load_cluster_config
from ..runtime_provenance import dgx_source_manifest_sha256
from .operator_receipt import receipt_destination

# Verify the inventory before importing its package initializer.
_LOADER = r'''
import hashlib,json,os,pathlib,stat,sys
root=pathlib.Path(sys.argv[1])
def regular(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        st=os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.geteuid():
            raise RuntimeError("unsafe bootstrap file")
        with os.fdopen(fd,"rb",closefd=False) as stream: return stream.read()
    finally: os.close(fd)
st=root.lstat()
if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.geteuid() or stat.S_IMODE(st.st_mode)!=0o700:
    raise RuntimeError("unsafe bootstrap directory")
data=regular(root/"manifest.json")
if hashlib.sha256(data).hexdigest()!=sys.argv[2]: raise RuntimeError("bootstrap manifest changed")
manifest=json.loads(data)
actual={}
for current,dirs,files in os.walk(root/"dgx_monarch",followlinks=False):
    for name in dirs:
        if (pathlib.Path(current)/name).is_symlink(): raise RuntimeError("bootstrap symlink")
    for name in files:
        path=pathlib.Path(current)/name
        actual[path.relative_to(root).as_posix()]=hashlib.sha256(regular(path)).hexdigest()
if actual!=manifest["files"]: raise RuntimeError("bootstrap source changed")
sys.dont_write_bytecode=True
sys.pycache_prefix="/proc/self/fd/2147483647"
sys.path.insert(0,str(root))
from dgx_monarch.cli.update_bootstrap import _child
raise SystemExit(_child(root,manifest))
'''


def _regular_bytes(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("bootstrap input is not a regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            value = stream.read()
        after = os.fstat(fd)
        current = path.lstat()
        identities = [(s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) for s in (before, after, current)]
        if identities[0] != identities[1] or identities[1] != identities[2]:
            raise ValueError("bootstrap input changed while reading")
        return value
    finally:
        os.close(fd)


def _private_root(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("bootstrap root must be absolute")
    for item in reversed((path, *path.parents)):
        if item.is_symlink():
            raise ValueError("bootstrap root contains a symlink")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("bootstrap root must be a private directory owned by this user")


def _write_private(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _snapshot(source: Path, root: Path, request: dict[str, Any]) -> tuple[Path, str]:
    _private_root(root)
    before = dgx_source_manifest_sha256(source)
    snapshot = Path(tempfile.mkdtemp(prefix="controller-", dir=root))
    files: dict[str, str] = {}
    try:
        for current, directories, names in os.walk(source, followlinks=False):
            directories[:] = sorted(name for name in directories if name != "__pycache__")
            for name in directories:
                if (Path(current) / name).is_symlink():
                    raise ValueError("controller source contains a symlink")
            for name in sorted(names):
                source_file = Path(current) / name
                if source_file.is_symlink():
                    raise ValueError("controller source contains a symlink")
                if not name.endswith(".py"):
                    if name.endswith(".pyc"):
                        raise ValueError("controller source contains a direct bytecode file")
                    continue
                payload = _regular_bytes(source_file)
                relative = Path("dgx_monarch") / source_file.relative_to(source)
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                _write_private(target, payload)
                files[relative.as_posix()] = hashlib.sha256(payload).hexdigest()
        if before != dgx_source_manifest_sha256(source) or before != dgx_source_manifest_sha256(snapshot / "dgx_monarch"):
            raise ValueError("controller source changed during snapshot")
        manifest = {"version": 1, "source_manifest": before, "files": files, "request": request}
        payload = json.dumps(manifest, sort_keys=True).encode()
        _write_private(snapshot / "manifest.json", payload)
        return snapshot, hashlib.sha256(payload).hexdigest()
    except BaseException:
        shutil.rmtree(snapshot)
        raise


def _child(root: Path, manifest: dict[str, Any]) -> int:
    from .update_command import SystemUpdateOps
    from .update_entrypoint import run_serialized_update
    from .update_transaction import UpdateRequest

    def interrupt(_signum: int, _frame: object) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    request = manifest["request"]
    config_path = Path(request["config"])
    if hashlib.sha256(_regular_bytes(config_path)).hexdigest() != request["config_sha256"]:
        raise ValueError("cluster configuration changed before update")
    config = load_cluster_config(config_path)
    if hashlib.sha256(_regular_bytes(config_path)).hexdigest() != request["config_sha256"]:
        raise ValueError("cluster configuration changed while loading")
    repo = Path(request["repo"])
    repo_stat = repo.stat()
    if [repo_stat.st_dev, repo_stat.st_ino] != request["repo_identity"]:
        raise ValueError("original repository identity changed")
    if repo.resolve(strict=True) != repo:
        raise ValueError("original repository path changed")
    if dgx_source_manifest_sha256(repo / "src" / "dgx_monarch") != manifest["source_manifest"]:
        raise ValueError("original checkout source changed before update")
    result = run_serialized_update(
        repo=repo, request=UpdateRequest(request["target_ref"], request["assume_yes"]),
        ops=SystemUpdateOps(repo, config, driver_host=request["driver_host"]),
        receipt_path=request["receipt_path"],
    )
    _write_private(root / "settled.json", json.dumps({"exit_code": result}).encode())
    return result


def _wait_child(process: subprocess.Popen[bytes], snapshot: Path, *, timeout: float = 120) -> int:
    def interrupt(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    prior_term = signal.signal(signal.SIGTERM, interrupt)
    prior_int = signal.getsignal(signal.SIGINT)
    try:
        try:
            return process.wait()
        except (KeyboardInterrupt, SystemExit):
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt, SystemExit):
                print(f"update settlement is still pending; controller PID {process.pid}; recovery: {snapshot}", file=sys.stderr)
            print(f"update controller retained for recovery: {snapshot}", file=sys.stderr)
            raise
    finally:
        signal.signal(signal.SIGTERM, prior_term)
        signal.signal(signal.SIGINT, prior_int)


def launch_verified_update(
    config: ClusterConfig,
    *,
    repo: Path,
    target_ref: str,
    assume_yes: bool,
    driver_host: str | None,
    receipt_path: str | os.PathLike[str] | None,
) -> int:
    snapshot: Path | None = None
    try:
        destination = str(receipt_destination(receipt_path)) if receipt_path is not None else None
        original_repo = repo.expanduser().resolve(strict=True)
        config_path = Path(config.source).expanduser().resolve(strict=True)
        if not config.source or not original_repo.is_dir():
            raise ValueError("verified update requires a repository and saved configuration")
        config_bytes = _regular_bytes(config_path)
        loaded = load_cluster_config(config_path)
        if replace(config, source=str(config_path)) != replace(loaded, source=str(config_path)):
            raise ValueError("loaded configuration differs from its saved file")
        if config_bytes != _regular_bytes(config_path):
            raise ValueError("configuration changed while preparing update")
        source = Path(__file__).resolve().parents[1]
        if dgx_source_manifest_sha256(source) != dgx_source_manifest_sha256(original_repo / "src" / "dgx_monarch"):
            raise ValueError("loaded controller does not match the original checkout")
        repo_stat = original_repo.stat()
        request = {
            "repo_identity": [repo_stat.st_dev, repo_stat.st_ino],
            "repo": str(original_repo), "config": str(config_path),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "target_ref": target_ref, "assume_yes": assume_yes,
            "driver_host": driver_host, "receipt_path": destination,
        }
        root = Path.home() / ".local" / "state" / "dgx-monarch" / "update-controllers"
        snapshot, digest = _snapshot(source, root, request)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", _LOADER, str(snapshot), digest],
            cwd=snapshot, env=env, start_new_session=True,
        )
        try:
            _write_private(snapshot / "process.json", json.dumps({
                "pid": process.pid, "executable": sys.executable, "manifest_sha256": digest,
                "repo": str(original_repo),
            }).encode())
        finally:
            # Publication failure must not abandon a running transaction.
            result = _wait_child(process, snapshot)
        if result == 0 and json.loads(_regular_bytes(snapshot / "settled.json")) == {"exit_code": 0}:
            shutil.rmtree(snapshot)
        else:
            print(f"update controller retained for recovery: {snapshot}", file=sys.stderr)
        return result
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"verified update bootstrap refused: {exc}", file=sys.stderr)
        if snapshot is not None:
            print(f"update controller retained for recovery: {snapshot}", file=sys.stderr)
        return 2
