"""Readiness checks, config rendering, private validation, atomic publication and rollback for guided setup."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .. import TORCHMONARCH_PIN
from ..config import FABRIC_PROFILES, ClusterConfig, load_cluster_config, render_cluster_toml
from ..constants import DEFAULT_NCCL_MASTER_PORT, DEFAULT_WORKER_PORT
from ..operator_profiles import ProfileResolution
from .setup_config_fd import close_preserving, read_bounded, unlink_preserving, write_all
from .setup_config_parent import prepare_config_parent, validate_real_parent
from .setup_config_types import ConfigMutation as ConfigMutation
from .setup_config_types import ConfigSnapshot as ConfigSnapshot
from .setup_probe import ArtifactComparison, HostProbe
from .setup_verification import SetupVerification as SetupVerification
from .setup_verification import SetupVerificationProgress as SetupVerificationProgress
from .setup_verification import run_setup_smoke as run_setup_smoke
from .setup_verification import smoke_succeeded as smoke_succeeded
from .setup_verification import verify_setup as verify_setup

if TYPE_CHECKING:
    from .setup_config_transaction import ConfigTransaction

_MAX_CONFIG_BYTES = 1_000_000
_GB10_MODEL_FINGERPRINT = hashlib.sha256(b'["NVIDIA GB10"]').hexdigest()
_MIN_PYTHON = (3, 11)


def confirm_setup(prompt: str) -> bool:
    return input(prompt).strip().lower() == "y"


def publish_receipt(
    writer: Callable[..., Path], receipt: Mapping[str, object], path: Path | None, *, required: bool
) -> bool:
    if path is None and not required:
        return False
    try:
        writer(receipt, path)
        return True
    except Exception:
        return False


def request_binding(payload: Mapping[str, object]) -> str:
    """Hash the non-consent request fields a reviewed plan must match, without disclosing them."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(b"dgxm-setup-request-v1\0" + encoded.encode("ascii")).hexdigest()


def setup_readiness(
    *,
    probes: Sequence[HostProbe],
    expected_gpus: Sequence[int],
    artifacts: Sequence[ArtifactComparison],
    fabric_profile: str,
    install_service: bool,
    start_workers: bool,
    verify: bool,
    transport_security: str,
) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    if verify and not start_workers:
        blockers.append("verify_requires_worker_start")
    for expected, probe in zip(expected_gpus, probes, strict=True):
        prefix = f"host_{probe.ordinal}"
        if not probe.reachable:
            blockers.append(f"{prefix}_{probe.failure or 'unreachable'}")
            continue
        if not probe.python_available:
            blockers.append(f"{prefix}_python_unavailable")
            continue
        if not _supported_python(probe.python_version):
            blockers.append(f"{prefix}_python_version_unsupported")
        if probe.torch_version is None or probe.cuda_available is not True:
            blockers.append(f"{prefix}_cuda_torch_unavailable")
        if probe.gpu_count != expected:
            blockers.append(f"{prefix}_gpu_count_mismatch")
        if probe.torchmonarch_version != TORCHMONARCH_PIN:
            blockers.append(f"{prefix}_torchmonarch_pin_mismatch")
        if probe.comfy_exists is not True:
            blockers.append(f"{prefix}_comfy_missing")
        elif probe.comfy_runtime_marker is not True:
            blockers.append(f"{prefix}_comfy_runtime_marker_missing")
        elif probe.comfy_dirty is True and probe.comfy_only_missing_examples is not True:
            blockers.append(f"{prefix}_comfy_dirty")
        elif probe.comfy_git is not True:
            blockers.append(f"{prefix}_comfy_not_git")
        elif probe.comfy_dirty is None or probe.comfy_commit is None:
            blockers.append(f"{prefix}_comfy_git_state_unknown")
        elif probe.comfy_dirty is True:
            warnings.append(f"{prefix}_comfy_missing_examples_preserved")
        if (install_service or start_workers) and probe.rsync_available is not True:
            blockers.append(f"{prefix}_rsync_unavailable")
        if install_service:
            if probe.systemd_user_available is not True:
                blockers.append(f"{prefix}_systemd_user_unavailable")
            if probe.linger_enabled is not True:
                blockers.append(f"{prefix}_linger_disabled")
        if (install_service or start_workers) and probe.service_installed is not False:
            blockers.append(f"{prefix}_service_not_owned")
        if (install_service or start_workers) and probe.service_active is not False:
            blockers.append(f"{prefix}_worker_state_not_owned")
    for field_name in ("python_version", "torch_version", "torchmonarch_version"):
        values = {getattr(probe, field_name) for probe in probes if probe.reachable and probe.python_available}
        if len(values) > 1:
            blockers.append(f"cohort_{field_name}_mismatch")
    commits = {probe.comfy_commit for probe in probes if probe.comfy_git is True}
    if len(commits) > 1:
        blockers.append("cohort_comfy_commit_mismatch")
    blockers.extend(f"artifact_{item.ordinal}_{item.state}" for item in artifacts if item.state != "match")
    if transport_security != "trusted_fabric":
        blockers.append("trusted_fabric_not_acknowledged")
    if len(probes) > 1 and fabric_profile == "single-node":
        blockers.append("multi_host_single_node_fabric")
    return sorted(set(blockers)), sorted(set(warnings))


def _supported_python(version: str | None) -> bool:
    parts = version.split(".") if version else []
    numeric = len(parts) >= 2 and all(part.isascii() and part.isdigit() for part in parts[:2])
    return numeric and (int(parts[0]), int(parts[1])) >= _MIN_PYTHON


def recommend_fabric(probes: Sequence[HostProbe]) -> str | None:
    """Return single-node for one host, else a profile only if all hosts' active RDMA ports share one link layer."""
    if len(probes) == 1:
        return "single-node"
    if not probes or any(not probe.reachable or not probe.link_layers for probe in probes):
        return None
    layers = set().union(*(set(probe.link_layers) for probe in probes))
    if (
        len(probes) == 2
        and layers == {"Ethernet"}
        and all(probe.integrated is True and probe.gpu_count == 1 for probe in probes)
        and all(probe.model_fingerprint == _GB10_MODEL_FINGERPRINT for probe in probes)
    ):
        return "dgx-spark-pair"
    if layers == {"InfiniBand"}:
        return "generic-ib"
    if layers == {"Ethernet"}:
        return "generic-roce"
    return None


def render_setup_config(
    hosts: Sequence[tuple[str, str, int, str, str]],
    *,
    client_ip: str,
    fabric_profile: str,
    resolution: ProfileResolution,
    python_bin: str,
    ssh_key: str,
    comfy_dir: str,
    transport_security: str,
) -> str:
    """Render profile output using only keys accepted by the strict schema."""
    if fabric_profile not in FABRIC_PROFILES:
        raise ValueError(f"unknown fabric profile {fabric_profile!r}")
    base = render_cluster_toml(
        hosts=[(name, address, gpus) for name, address, gpus, _user, _comfy in hosts],
        client_ip=client_ip,
        fabric_profile=fabric_profile,
        master_addr=hosts[0][1],
        master_port=DEFAULT_NCCL_MASTER_PORT,
        ssh_key=ssh_key,
        worker_port=DEFAULT_WORKER_PORT,
        python_bin=python_bin,
        transport_security=transport_security,
    )
    output: list[str] = []
    host_index = -1
    for line in base.splitlines():
        output.append(line)
        if line == "[cluster]":
            output.extend(
                f"{key} = {'true' if value else 'false'}" for key, value in sorted(resolution.cluster_set.items())
            )
            if comfy_dir:
                output.append(f"comfy_dir = {json.dumps(comfy_dir)}")
        elif line == "[[hosts]]":
            host_index += 1
        elif line.startswith("gpus = ") and host_index >= 0:
            _name, _address, _gpus, ssh_user, host_comfy = hosts[host_index]
            if ssh_user:
                output.append(f"ssh_user = {json.dumps(ssh_user)}")
            if host_comfy:
                output.append(f"comfy_dir = {json.dumps(host_comfy)}")
    if resolution.worker_args_set:
        output.extend(["", "[worker_args]"])
        output.extend(
            f"{key} = {'true' if value else 'false'}" for key, value in sorted(resolution.worker_args_set.items())
        )
    return "\n".join(output).rstrip() + "\n"


def decode_snapshot(snapshot: ConfigSnapshot) -> str:
    """Decode exactly what the operator will review; refuse lossy diffs."""
    if not snapshot.existed:
        return ""
    try:
        return snapshot.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OSError("existing cluster config is not valid UTF-8") from exc


def private_roundtrip(
    text: str,
    *,
    loader: Callable[[str | Path], ClusterConfig] = load_cluster_config,
    temp_dir: str | os.PathLike[str] | None = None,
) -> ClusterConfig:
    """Validate a candidate through the config loader using a temporary file removed on every exit."""
    payload = text.encode("utf-8")
    if len(payload) > _MAX_CONFIG_BYTES:
        raise ValueError("cluster config candidate exceeds the setup size limit")
    fd, name = tempfile.mkstemp(prefix=".dgxm-setup-", suffix=".toml", dir=temp_dir)
    path = Path(name)
    try:
        os.fchmod(fd, 0o600)
        write_all(fd, payload)
        os.fsync(fd)
        close_preserving(fd)
        fd = -1
        result = loader(path)
    except BaseException as primary:
        if fd >= 0:
            close_preserving(fd, primary)
        unlink_preserving(path, primary)
        raise
    unlink_preserving(path)
    return result


def read_snapshot(path: str | os.PathLike[str]) -> ConfigSnapshot:
    target = Path(path).expanduser().absolute()
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return ConfigSnapshot(target, False, b"", None, None, None)
    if target.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise OSError("cluster config target must be a regular file, not a symlink")
    if metadata.st_size > _MAX_CONFIG_BYTES:
        raise OSError("existing cluster config exceeds the setup size limit")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(target, flags)
    try:
        opened = os.fstat(fd)
        if opened.st_dev != metadata.st_dev or opened.st_ino != metadata.st_ino:
            raise OSError("cluster config changed while it was opened")
        if hasattr(os, "geteuid") and opened.st_uid != os.geteuid():
            raise PermissionError("cluster config must be owned by the current user")
        content = read_bounded(fd, _MAX_CONFIG_BYTES + 1)
        after = os.fstat(fd)
        close_preserving(fd)
        fd = -1
    except BaseException as primary:
        if fd >= 0:
            close_preserving(fd, primary)
        raise
    if len(content) > _MAX_CONFIG_BYTES:
        raise OSError("existing cluster config exceeds the setup size limit")
    final = target.lstat()
    if not (_stat_identity(metadata) == _stat_identity(opened) == _stat_identity(after) == _stat_identity(final)):
        raise OSError("cluster config changed while it was read")
    return ConfigSnapshot(
        target,
        True,
        content,
        hashlib.sha256(content).hexdigest(),
        stat.S_IMODE(final.st_mode),
        (final.st_dev, final.st_ino),
    )


def apply_config(
    path: str | os.PathLike[str],
    text: str,
    *,
    expected: ConfigSnapshot,
    _transaction: ConfigTransaction | None = None,
) -> ConfigMutation:
    """Publish exact UTF-8 bytes at mode 0600, first backing up the file it replaces under a name never overwritten."""
    target = Path(path).expanduser().absolute()
    if target != expected.path:
        raise ValueError("setup plan and apply target differ")
    prepare_config_parent(target.parent)
    current = read_snapshot(target)
    if (current.existed, current.digest, current.mode, current.identity) != (
        expected.existed,
        expected.digest,
        expected.mode,
        expected.identity,
    ):
        raise OSError("cluster config changed after the setup plan was built")
    payload = text.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if current.existed and current.content == payload and current.mode == 0o600:
        mutation = ConfigMutation(target, False, current, digest, None, current.identity, True)
        if _transaction is not None:
            _transaction.note_mutation(mutation)
        return mutation
    backup = _backup(current) if current.existed else None
    installed_identity, durable = _atomic_replace(
        target,
        payload,
        0o600,
        expected=current,
        transaction=_transaction,
        backup_path=backup,
    )
    mutation = ConfigMutation(target, True, current, digest, backup, installed_identity, durable)
    if _transaction is not None:
        _transaction.note_mutation(mutation)
    return mutation


def rollback_config(mutation: ConfigMutation) -> bool:
    """Undo only this mutation's exact replacement; never overwrite a later writer's file."""
    if not mutation.changed:
        return True
    try:
        current = read_snapshot(mutation.path)
        if (
            not current.existed
            or current.digest != mutation.new_digest
            or current.identity != mutation.installed_identity
        ):
            return False
        if mutation.prior.existed:
            prior_mode = mutation.prior.mode if mutation.prior.mode is not None else 0o600
            _identity, durable = _atomic_replace(
                mutation.path,
                mutation.prior.content,
                prior_mode,
                expected=current,
            )
            return durable
        return _unlink_expected(mutation.path, current)
    except OSError:
        return False


def _backup(snapshot: ConfigSnapshot) -> Path:
    if snapshot.digest is None:
        raise ValueError("cannot back up a cluster config without a digest")
    backup = snapshot.path.with_name(f".{snapshot.path.name}.setup-backup-{snapshot.digest[:12]}")
    try:
        _write_exclusive(backup, snapshot.content, 0o600)
        _fsync_directory(backup.parent)
    except FileExistsError:
        existing = read_snapshot(backup)
        if not existing.existed or existing.content != snapshot.content:
            raise OSError("cluster config backup name is already occupied") from None
        if existing.mode != 0o600:
            raise OSError("existing cluster config backup is not private") from None
    return backup


def _atomic_replace(
    path: Path,
    payload: bytes,
    mode: int,
    *,
    expected: ConfigSnapshot,
    transaction: ConfigTransaction | None = None,
    backup_path: Path | None = None,
) -> tuple[tuple[int, int], bool]:
    validate_real_parent(path.parent)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}")
    identity: tuple[int, int] | None = None
    try:
        identity = _write_exclusive(temporary, payload, mode)
        if transaction is not None:
            transaction.note_prepared(identity, backup_path)
        if expected.existed:
            if not _same_snapshot(read_snapshot(path), expected):
                raise OSError("cluster config changed immediately before publication")
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                raise OSError("cluster config appeared immediately before publication") from None
            temporary.unlink()
        published = read_snapshot(path)
        digest = hashlib.sha256(payload).hexdigest()
        if published.identity != identity or published.digest != digest or published.mode != mode:
            raise OSError("cluster config publication readback was not exact")
        try:
            _fsync_directory(path.parent)
        except OSError:
            durable = False
        else:
            durable = True
        if transaction is not None:
            transaction.note_durable(durable)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        return identity, durable
    except BaseException:
        if identity is not None:
            try:
                _restore_interrupted_publish(path, payload, identity, expected)
            except BaseException:
                pass
        try:
            temporary.unlink()
        except BaseException:
            pass
        raise


def _restore_interrupted_publish(
    path: Path,
    payload: bytes,
    identity: tuple[int, int],
    prior: ConfigSnapshot,
) -> None:
    """Undo a failed or interrupted publish only if the target still holds that publish's inode and bytes."""
    current = read_snapshot(path)
    if current.identity != identity or current.digest != hashlib.sha256(payload).hexdigest():
        return
    if prior.existed:
        prior_mode = prior.mode if prior.mode is not None else 0o600
        _atomic_replace(path, prior.content, prior_mode, expected=current)
    else:
        _unlink_expected(path, current)


def _same_snapshot(current: ConfigSnapshot, expected: ConfigSnapshot) -> bool:
    return current == expected


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _unlink_expected(path: Path, expected: ConfigSnapshot) -> bool:
    if not _same_snapshot(read_snapshot(path), expected):
        return False
    path.unlink()
    _fsync_directory(path.parent)
    return True


def _write_exclusive(path: Path, payload: bytes, mode: int) -> tuple[int, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, mode)
    identity: tuple[int, int] | None = None
    try:
        metadata = os.fstat(fd)
        identity = metadata.st_dev, metadata.st_ino
        os.fchmod(fd, mode)
        write_all(fd, payload)
        os.fsync(fd)
        close_preserving(fd)
        fd = -1
    except BaseException as primary:
        if identity is None:
            try:
                metadata = os.fstat(fd)
                identity = metadata.st_dev, metadata.st_ino
            except BaseException:
                pass
        if fd >= 0:
            close_preserving(fd, primary)
        try:
            current = path.lstat()
            if identity is not None and (current.st_dev, current.st_ino) == identity:
                unlink_preserving(path, primary)
        except BaseException:
            pass
        raise
    if identity is None:
        raise OSError("setup config temporary identity was unavailable")
    return identity


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    except BaseException as primary:
        close_preserving(fd, primary)
        raise
    close_preserving(fd)
