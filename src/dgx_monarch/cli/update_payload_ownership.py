"""Separate verified distribution payloads in shared namespace directories."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import re
import stat
from collections.abc import Mapping
from email.parser import BytesParser
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path, PurePosixPath


class PayloadOwnershipError(ValueError):
    """Shared payload ownership could not be established."""


def _read(path: Path) -> bytes:
    for parent in path.parents:
        if parent.is_symlink():
            raise PayloadOwnershipError("distribution metadata has a symlink ancestor")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise PayloadOwnershipError("distribution metadata is not regular")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read()
        after = os.fstat(fd)
        current = path.lstat()
        identities = [(s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns) for s in (before, after, current)]
        if identities[0] != identities[1] or identities[1] != identities[2]:
            raise PayloadOwnershipError("distribution payload changed while reading")
        return data
    finally:
        os.close(fd)


def _safe_path(value: str) -> bool:
    return bool(
        value and not value.startswith("/") and "\\" not in value and ":" not in value
        and all(ord(c) >= 32 and ord(c) != 127 for c in value)
        and all(part not in ("", ".", "..") for part in value.split("/"))
    )


def _records(payload: bytes) -> tuple[dict[str, tuple[str, str]], bool]:
    records: dict[str, tuple[str, str]] = {}
    invalid = False
    for row in csv.reader(io.StringIO(payload.decode("utf-8"))):
        if len(row) != 3:
            invalid = True
            continue
        path, digest, size = row
        if not _safe_path(path):
            # Installer-created console scripts live outside site-packages.
            if not re.fullmatch(r"(?:\.\./)+bin/[^/\\:]+", path):
                invalid = True
            continue
        if path in records:
            invalid = True
        records[path] = digest, size
    return records, invalid


def _verified_record(data: bytes, record: tuple[str, str]) -> bool:
    digest, size = record
    expected = "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
    return digest == expected and size == str(len(data))


def _module_path(path: str) -> tuple[str, bool] | None:
    for suffix in sorted([".py", *EXTENSION_SUFFIXES], key=len, reverse=True):
        if path.endswith(suffix):
            module = path[:-len(suffix)]
            package = module.endswith("/__init__")
            return (module.removesuffix("/__init__"), package)
    return None


def _namespace_path(path: str, expected: Mapping[str, str]) -> bool:
    parts = PurePosixPath(path).parts
    ancestors = {"/".join(parts[:count]) for count in range(1, len(parts))}
    modules = {item for name in expected if (item := _module_path(name)) is not None}
    # A regular package owns its subtree. A module cannot gain a competing
    # foreign package, extension, or directory at its import name.
    if any(module in ancestors for module, _package in modules):
        return False
    foreign = _module_path(path)
    if foreign is not None:
        module, _package = foreign
        expected_dirs = {parent.as_posix() for name in expected for parent in PurePosixPath(name).parents}
        if module in expected_dirs or any(module == own for own, _kind in modules):
            return False
    return len(parts) > 1


def foreign_namespace_files(
    site: Path, expected: Mapping[str, str], found: Mapping[str, str],
) -> set[str]:
    """Exclude only independently recorded, intact foreign namespace files.

    Installed records establish local ownership, not publisher authenticity.
    The pinned wheel remains authoritative for every expected payload path.
    """
    owners: dict[str, list[str]] = {}
    identities: dict[str, int] = {}
    candidates: list[tuple[Path, str, bytes, dict[str, tuple[str, str]], bool]] = []
    for info in site.iterdir():
        if not info.name.lower().endswith(".dist-info"):
            continue
        if info.is_symlink() or not info.is_dir():
            raise PayloadOwnershipError("distribution metadata directory is unsafe")
        metadata = _read(info / "METADATA")
        message = BytesParser().parsebytes(metadata, headersonly=True)
        names, versions = message.get_all("Name", []), message.get_all("Version", [])
        if len(names) != 1 or len(versions) != 1 or not names[0] or not versions[0]:
            raise PayloadOwnershipError("distribution identity is ambiguous")
        name = re.sub(r"[-_.]+", "-", names[0]).lower()
        identities[name] = identities.get(name, 0) + 1
        if name == "torchmonarch":
            continue
        record_path = info / "RECORD"
        if not record_path.exists():
            continue
        records, invalid = _records(_read(record_path))
        relevant = records.keys() & (expected.keys() | found.keys())
        if relevant:
            candidates.append((info, name, metadata, records, invalid))
    if identities.get("torchmonarch", 0) > 1:
        raise PayloadOwnershipError("pinned distribution identity is ambiguous")
    excluded: set[str] = set()
    for info, name, metadata, records, invalid in candidates:
        if invalid or identities[name] != 1:
            raise PayloadOwnershipError("foreign distribution inventory is ambiguous")
        metadata_record = records.get(f"{info.name}/METADATA")
        if metadata_record is None or not _verified_record(metadata, metadata_record):
            raise PayloadOwnershipError("foreign distribution metadata does not match its record")
        if records.keys() & expected.keys():
            raise PayloadOwnershipError("foreign distribution claims pinned payload")
        for path in records.keys() & found.keys():
            owners.setdefault(path, []).append(name)
            if len(owners[path]) != 1 or not _namespace_path(path, expected):
                raise PayloadOwnershipError("foreign payload overlaps a protected package")
            data = _read(site / path)
            if not _verified_record(data, records[path]) or hashlib.sha256(data).hexdigest() != found[path]:
                raise PayloadOwnershipError("foreign payload does not match its record")
            excluded.add(path)
    return excluded
