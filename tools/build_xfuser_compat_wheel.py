#!/usr/bin/env python3
"""Build the local xFuser 0.7.0+dgxm.npuimport1 compatibility wheel.

The tool reads the public xFuser 0.7.0 wheel (from PyPI, or --wheel) and checks
its SHA-256 and RECORD. Then it makes the ring package import its NPU backend
only on request, so CUDA installs need no NPU-only yunchang helpers, sets the
new version and rewrites RECORD; no other file content changes. The upstream
wheel is not vendored, and the tool uses only the standard library.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

OFFICIAL_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/16/a5/3c155a618f485e69033d5da528e671ede9d2e0ca25500b5003ca1bfdc8d4/"
    "xfuser-0.7.0-py3-none-any.whl"
)
OFFICIAL_SHA256 = "873a76402e7423d0382375886acf7ceabe6ba10183e8249b529f02251d13c0cd"
OFFICIAL_VERSION = "0.7.0"
VERSION = "0.7.0+dgxm.npuimport1"
RING_INIT = "xfuser/core/long_ctx_attention/ring/__init__.py"
VERSION_FILE = "xfuser/__version__.py"

PATCHED_RING = b'''from .ring_flash_attn import (
    xdit_ring_flash_attn_func,
    xdit_sana_ring_flash_attn_func,
)

__all__ = [
    "xdit_ring_flash_attn_func",
    "xdit_sana_ring_flash_attn_func",
    "xdit_ring_npu_flash_attn_func",
]


def __getattr__(name):
    """Load the optional NPU ring backend only for an NPU request.

    CUDA installs use the normal ring backend and must not require optional
    NPU-only yunchang helpers merely to construct LongContextAttention.
    """
    if name != "xdit_ring_npu_flash_attn_func":
        raise AttributeError(name)
    try:
        from .ring_npu_flash_attn import xdit_ring_npu_flash_attn_func
    except ImportError as exc:
        raise ImportError(
            "xFuser NPU ring attention requires a yunchang release that "
            "exports the NPU ring helpers"
        ) from exc
    return xdit_ring_npu_flash_attn_func
'''
PATCHED_VERSION = f"# private local build\n__version__ = version = '{VERSION}'\n".encode()


class WheelValidationError(ValueError):
    """The supplied wheel is not the pinned upstream xFuser release."""


def sha256(data: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(data).hexdigest()


def record_hash(data: bytes) -> str:
    """Return the wheel RECORD representation of a SHA-256 digest."""
    encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
    return f"sha256={encoded}"


def _safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and "\\" not in name and not path.is_absolute() and ".." not in path.parts


def _validate_record(files: dict[str, bytes], record_name: str) -> None:
    try:
        rows = list(csv.reader(io.TextIOWrapper(io.BytesIO(files[record_name]), newline="")))
    except KeyError as exc:
        raise WheelValidationError(f"wheel does not contain {record_name}") from exc

    expected = set(files)
    expected.remove(record_name)
    seen: set[str] = set()
    for row in rows:
        if len(row) != 3 or not row[0]:
            raise WheelValidationError("wheel RECORD contains an invalid row")
        name, digest, size = row
        if name == record_name:
            if digest or size:
                raise WheelValidationError("wheel RECORD gives itself a hash or size")
            continue
        if name in seen or name not in expected:
            raise WheelValidationError("wheel RECORD lists a member twice or lists a file the wheel lacks")
        if digest != record_hash(files[name]) or size != str(len(files[name])):
            raise WheelValidationError(f"wheel RECORD does not match {name}")
        seen.add(name)
    if seen != expected:
        raise WheelValidationError("wheel RECORD omits one or more members")


def read_verified_wheel(path: Path) -> dict[str, tuple[zipfile.ZipInfo, bytes]]:
    """Return the members of the pinned upstream wheel at *path*; raise WheelValidationError otherwise."""
    payload = path.read_bytes()
    if sha256(payload) != OFFICIAL_SHA256:
        raise WheelValidationError("official xFuser wheel SHA-256 mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or any(not _safe_member_name(name) for name in names):
                raise WheelValidationError("wheel has duplicate or unsafe member names")
            if any(info.is_dir() for info in infos):
                raise WheelValidationError("wheel contains directory entries")
            members = {info.filename: (info, archive.read(info)) for info in infos}
    except zipfile.BadZipFile as exc:
        raise WheelValidationError("official xFuser wheel is not a valid ZIP archive") from exc

    dist_info = f"xfuser-{OFFICIAL_VERSION}.dist-info/"
    record_name = f"{dist_info}RECORD"
    files = {name: data for name, (_, data) in members.items()}
    _validate_record(files, record_name)
    required = {RING_INIT, VERSION_FILE, f"{dist_info}METADATA", record_name}
    missing = required.difference(members)
    if missing:
        raise WheelValidationError(f"wheel is missing required member: {sorted(missing)[0]}")
    metadata = files[f"{dist_info}METADATA"]
    if metadata.count(b"\nVersion: 0.7.0\n") != 1:
        raise WheelValidationError("wheel metadata does not declare xFuser 0.7.0 exactly once")
    return members


def _renamed_member(name: str) -> str:
    old_prefix = f"xfuser-{OFFICIAL_VERSION}.dist-info/"
    new_prefix = f"xfuser-{VERSION}.dist-info/"
    return f"{new_prefix}{name.removeprefix(old_prefix)}" if name.startswith(old_prefix) else name


def patched_members(source: dict[str, tuple[zipfile.ZipInfo, bytes]]) -> dict[str, tuple[zipfile.ZipInfo, bytes]]:
    """Apply the lazy NPU import, the new version and dist-info name, and a fresh RECORD."""
    members = {_renamed_member(name): (copy.copy(info), data) for name, (info, data) in source.items()}
    new_prefix = f"xfuser-{VERSION}.dist-info/"
    new_record = f"{new_prefix}RECORD"
    members[RING_INIT] = (members[RING_INIT][0], PATCHED_RING)
    members[VERSION_FILE] = (members[VERSION_FILE][0], PATCHED_VERSION)
    metadata_name = f"{new_prefix}METADATA"
    metadata_info, metadata = members[metadata_name]
    members[metadata_name] = (
        metadata_info,
        metadata.replace(b"\nVersion: 0.7.0\n", f"\nVersion: {VERSION}\n".encode(), 1),
    )
    record_info = members.pop(new_record)[0]
    rows = [[name, record_hash(data), str(len(data))] for name, (_, data) in sorted(members.items())]
    rows.append([new_record, "", ""])
    output = io.StringIO()
    csv.writer(output, lineterminator="\r\n").writerows(rows)
    members[new_record] = (record_info, output.getvalue().encode())
    return members


def _write_wheel(path: Path, members: dict[str, tuple[zipfile.ZipInfo, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, (source_info, data) in sorted(members.items()):
            info = copy.copy(source_info)
            info.filename = name
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, data)


def write_output(output_dir: Path, members: dict[str, tuple[zipfile.ZipInfo, bytes]]) -> Path:
    """Atomically create the wheel without replacing an existing output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"xfuser-{VERSION}-py3-none-any.whl"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing wheel: {output}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".xfuser-wheel-", suffix=".tmp", dir=output_dir)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_wheel(temporary, members)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def download_official_wheel(output_dir: Path, *, opener=urllib.request.urlopen) -> Path:
    """Download the pinned PyPI wheel to a temporary file in *output_dir* and verify it; the caller deletes it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".xfuser-input-", suffix=".whl", dir=output_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination, opener(OFFICIAL_WHEEL_URL) as response:
            while chunk := response.read(1024 * 1024):
                destination.write(chunk)
        read_verified_wheel(temporary)
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build(wheel: Path | None, output_dir: Path, *, opener=urllib.request.urlopen) -> Path:
    """Verify the upstream input and create the local compatibility wheel."""
    temporary_input: Path | None = None
    try:
        if wheel is None:
            temporary_input = download_official_wheel(output_dir, opener=opener)
            wheel = temporary_input
        return write_output(output_dir, patched_members(read_verified_wheel(wheel)))
    finally:
        if temporary_input is not None:
            temporary_input.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path,
                        help="local copy of the xFuser 0.7.0 wheel for an offline build; the tool checks its SHA-256")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for the new local wheel")
    args = parser.parse_args()
    try:
        output = build(args.wheel, args.output_dir)
    except (OSError, WheelValidationError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(output)


if __name__ == "__main__":
    main()
