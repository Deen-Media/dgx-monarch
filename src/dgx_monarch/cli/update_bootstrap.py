"""Keep the update controller independent of the checkout it updates."""
from __future__ import annotations

import hashlib
import json
import os
import re
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



def _recovery_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
         "-c", "core.untrackedCache=false", "-C", str(repo), *args],
        capture_output=True, text=True, timeout=30,
        env={"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "GIT_NO_REPLACE_OBJECTS": "1",
             "LANG": "C", "LC_ALL": "C", "PATH": os.defpath},
    )
    if result.returncode:
        raise ValueError("recovery checkout Git inspection failed")
    return result.stdout.strip()


def _verify_loaded_controller(source: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name != "dgx_monarch" and not name.startswith("dgx_monarch."):
            continue
        path = getattr(module, "__file__", None)
        if not isinstance(path, str) or not Path(path).resolve(strict=True).is_relative_to(source):
            raise ValueError("loaded recovery controller module came from another source")


def _recovery_binding(repo: Path, controller: Path, target: str, source: Path) -> dict:
    from .update_inspect import inspect_checkout

    if re.fullmatch(r"[0-9a-f]{40}", target) is None:
        raise ValueError("recovery target must be a full 40-character commit")
    controller = controller.expanduser().resolve(strict=True)
    if controller == repo or source != controller / "src/dgx_monarch":
        raise ValueError("recovery requires the loaded controller's separate checkout")
    _verify_loaded_controller(source)
    for checkout in (repo, controller):
        if _recovery_git(checkout, "status", "--porcelain=v1", "--untracked-files=all"):
            raise ValueError("recovery checkouts must be clean")
        if any(line[:1].islower() or line.startswith("S ") for line in _recovery_git(checkout, "ls-files", "-v").splitlines()):
            raise ValueError("recovery checkout hides tracked changes")
    head = _recovery_git(repo, "rev-parse", "HEAD")
    if _recovery_git(controller, "rev-parse", "HEAD") != target:
        raise ValueError("recovery controller HEAD differs from the exact target")
    origin = _recovery_git(repo, "remote", "get-url", "origin")
    if origin != _recovery_git(controller, "remote", "get-url", "origin"):
        raise ValueError("recovery controller origin differs from the original checkout")
    _recovery_git(controller, "merge-base", "--is-ancestor", head, target)
    original_metadata, _ = inspect_checkout(repo, head)
    controller_metadata, _ = inspect_checkout(controller, target)
    if controller_metadata.source_manifest != dgx_source_manifest_sha256(source):
        raise ValueError("loaded recovery controller differs from its reviewed checkout")
    return {"original_head": head, "original_source_manifest": original_metadata.source_manifest,
            "origin_sha256": hashlib.sha256(origin.encode()).hexdigest(),
            "controller_commit": target, "controller_source_manifest": controller_metadata.source_manifest}


def _check_recovery_original(repo: Path, request: dict, source_manifest: str) -> None:
    recovery = request["recovery"]
    info = repo.stat()
    if repo.resolve(strict=True) != repo or [info.st_dev, info.st_ino] != request["repo_identity"]:
        raise ValueError("original repository identity changed during recovery")
    if request["target_ref"] != recovery["controller_commit"] or source_manifest != recovery["controller_source_manifest"]:
        raise ValueError("recovery controller or target binding changed")
    if _recovery_git(repo, "rev-parse", "HEAD") != recovery["original_head"]:
        raise ValueError("original repository commit changed during recovery")
    if dgx_source_manifest_sha256(repo / "src/dgx_monarch") != recovery["original_source_manifest"]:
        raise ValueError("original repository source changed during recovery")
    origin = _recovery_git(repo, "remote", "get-url", "origin")
    if hashlib.sha256(origin.encode()).hexdigest() != recovery["origin_sha256"]:
        raise ValueError("original repository origin changed during recovery")
    if hashlib.sha256(_regular_bytes(Path(request["config"]))).hexdigest() != request["config_sha256"]:
        raise ValueError("cluster configuration changed during recovery")


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
    ops: SystemUpdateOps
    if "recovery" in request:
        _check_recovery_original(repo, request, manifest["source_manifest"])

        class RecoveryOps(SystemUpdateOps):
            def resolve_target(self, target_ref: str) -> str:
                # run_serialized_update already holds the original checkout lock.
                _check_recovery_original(repo, request, manifest["source_manifest"])
                return super().resolve_target(target_ref)

        ops = RecoveryOps(repo, config, driver_host=request["driver_host"])
    else:
        if dgx_source_manifest_sha256(repo / "src" / "dgx_monarch") != manifest["source_manifest"]:
            raise ValueError("original checkout source changed before update")
        ops = SystemUpdateOps(repo, config, driver_host=request["driver_host"])
    result = run_serialized_update(
        repo=repo, request=UpdateRequest(request["target_ref"], request["assume_yes"]),
        ops=ops,
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
    controller_repository: Path | None = None,
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
        recovery = None
        if controller_repository is not None:
            recovery = _recovery_binding(original_repo, controller_repository, target_ref, source)
        elif dgx_source_manifest_sha256(source) != dgx_source_manifest_sha256(original_repo / "src" / "dgx_monarch"):
            raise ValueError("loaded controller does not match the original checkout")
        repo_stat = original_repo.stat()
        request = {
            "repo_identity": [repo_stat.st_dev, repo_stat.st_ino],
            "repo": str(original_repo), "config": str(config_path),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "target_ref": target_ref, "assume_yes": assume_yes,
            "driver_host": driver_host, "receipt_path": destination,
        }
        if recovery is not None:
            request["recovery"] = recovery
            print(f"Original checkout: {original_repo}")
            print(f"Current commit: {recovery['original_head']}")
            print(f"Reviewed recovery controller and target: {target_ref}")
            print("The normal verified-update checks and confirmation still apply.")
        root = Path.home() / ".local" / "state" / "dgx-monarch" / "update-controllers"
        snapshot, digest = _snapshot(source, root, request)
        if recovery is not None and controller_repository is not None and _recovery_binding(original_repo, controller_repository, target_ref, source) != recovery:
            raise ValueError("recovery identity changed while copying the controller")
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
