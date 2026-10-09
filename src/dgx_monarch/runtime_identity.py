"""Path-redacted machine, package, and artifact runtime identities."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
from pathlib import Path


def _stat_tuple(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def machine_identity_sha256() -> str:
    machine_id = Path("/etc/machine-id").read_text(encoding="utf-8").strip()
    if not machine_id:
        raise RuntimeError("runtime provenance machine identity is unavailable")
    return hashlib.sha256(
        b"dgx-monarch-wan-machine-v1\0" + machine_id.encode()
    ).hexdigest()


def runtime_instance_sha256() -> str:
    """Return a stable, path-redacted identity for this exact process lifetime."""
    try:
        tail = Path("/proc/self/stat").read_text(encoding="utf-8").rpartition(") ")[2]
        start_ticks = tail.split()[19]
    except (OSError, IndexError) as exc:
        raise RuntimeError("runtime provenance process identity is unavailable") from exc
    payload = f"{machine_identity_sha256()}:{os.getpid()}:{start_ticks}".encode()
    return hashlib.sha256(b"dgx-monarch-runtime-instance-v1\0" + payload).hexdigest()


def runtime_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    from . import __version__

    # Secondary workers import the atomically synced managed source copy. A
    # worker may have no dgx-monarch dist-info, or stale metadata from an older
    # local install, so the version comes from the loaded package, which the
    # source manifest and origin provenance already bind.
    result = {
        "python": platform.python_version(),
        "dgx-monarch": __version__,
    }
    for name in ("torch", "torchmonarch", "xfuser", "yunchang"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "missing"
    return result


def resolved_artifact_identities(manifest: object) -> list[dict[str, object]]:
    import folder_paths

    if not isinstance(manifest, list) or len(manifest) > 64:
        raise ValueError("runtime provenance artifact manifest is invalid")
    allowed_kinds = {"diffusion_models", "loras", "text_encoders"}
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in manifest:
        if not isinstance(item, dict) or set(item) != {"id", "kind", "file"}:
            raise ValueError("runtime provenance artifact row is invalid")
        item_id, kind, filename = item["id"], item["kind"], item["file"]
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in seen
            or kind not in allowed_kinds
            or not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ValueError("runtime provenance artifact binding is invalid")
        resolved = folder_paths.get_full_path(kind, filename)
        if resolved is None:
            raise RuntimeError("runtime provenance artifact did not resolve")
        path = Path(resolved).resolve(strict=True)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or _stat_tuple(before) != _stat_tuple(after)
            or _stat_tuple(after) != _stat_tuple(path_after)
        ):
            raise RuntimeError("runtime provenance artifact identity is unstable")
        payload = json.dumps(
            [item_id, *_stat_tuple(after)], separators=(",", ":")
        ).encode()
        rows.append(
            {
                "id": item_id,
                "identity_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        seen.add(item_id)
    return sorted(rows, key=lambda row: str(row["id"]))
