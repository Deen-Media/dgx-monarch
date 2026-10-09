"""Separate verified distribution payloads in shared namespace directories."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import re
import stat
import sys
from collections.abc import Mapping
from email.parser import BytesParser
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path, PurePosixPath

# Official color-matcher 0.6.0 wheel: data installed both inside site-packages
# and through the wheel data scheme. External copies are never opened here.
_COLOR_MATCHER_DATA: dict[str, tuple[str, str]] = {
    "scotland_house.png": ("sha256=7nKDHIIwtWkoW1JU2yNz6I4L7650rLYCLUKgyOka19E", "298652"),
    "scotland_pitie.png": ("sha256=5aor6EphGdWqn9_rS76CTXuP6juqccIXcyXcg0O8YZM", "304007"),
    "scotland_plain.png": ("sha256=RZuXuvsUifnqLxw3aYilcbN8tW1G6gilrZGpjLAswXc", "280371"),
}


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


def _records(
    payload: bytes, *, color_matcher: bool = False,
) -> tuple[dict[str, tuple[str, str]], bool]:
    """Parse owned site files; recognize ancillary external spellings lexically.

    External entries are never resolved, read, or returned as owned payload.
    The color-matcher data exception also requires verified in-site copies.
    """
    records: dict[str, tuple[str, str]] = {}
    invalid = False
    seen: set[str] = set()
    external = {
        "../../../tests/data/" + name: record for name, record in _COLOR_MATCHER_DATA.items()
    } if color_matcher else {}
    if color_matcher and sys.implementation.cache_tag:
        external["../../../bin/__pycache__/cli." + sys.implementation.cache_tag + ".pyc"] = ("", "")
    for row in csv.reader(io.StringIO(payload.decode("utf-8"))):
        if len(row) != 3:
            invalid = True
            continue
        path, digest, size = row
        if path in seen:
            invalid = True
        seen.add(path)
        if not _safe_path(path):
            if path in external and (digest, size) == external[path]:
                continue
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


def _color_matcher_assets(site: Path, records: Mapping[str, tuple[str, str]]) -> None:
    """Bind ignored external data records to verified copies inside site-packages."""
    for name, expected_record in _COLOR_MATCHER_DATA.items():
        path = "tests/data/" + name
        if records.get(path) != expected_record or not _verified_record(_read(site / path), expected_record):
            raise PayloadOwnershipError("color-matcher test data differs from its wheel")


def _empty_color_test_initializer(
    path: str, data: bytes, expected: Mapping[str, str],
) -> bool:
    """Allow an empty initializer only in the pinned wheel's bundled test tree.

    This converts that shared namespace into a regular package. It grants no
    exception for production packages or a pinned test initializer/module.
    """
    return (
        path == "tests/__init__.py" and data == b""
        and any(name.startswith("tests/") for name in expected)
        and not any(
            name in {"tests.pyc", "tests/__init__.pyc"}
            or ((module := _module_path(name)) is not None and module[0] == "tests")
            for name in expected
        )
    )


def foreign_namespace_files(
    site: Path, expected: Mapping[str, str], found: Mapping[str, str],
) -> set[str]:
    """Exclude only independently recorded, intact foreign namespace files.

    Installed records establish local ownership, not publisher authenticity.
    The pinned wheel remains authoritative for every expected payload path.
    """
    owners: dict[str, list[str]] = {}
    identities: dict[str, int] = {}
    candidates: list[tuple[Path, str, bytes, dict[str, tuple[str, str]], bool, bool]] = []
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
        color_matcher = name == "color-matcher" and versions[0] == "0.6.0"
        records, invalid = _records(_read(record_path), color_matcher=color_matcher)
        relevant = records.keys() & (expected.keys() | found.keys())
        if relevant:
            candidates.append((info, name, metadata, records, invalid, color_matcher))
    if identities.get("torchmonarch", 0) > 1:
        raise PayloadOwnershipError("pinned distribution identity is ambiguous")
    excluded: set[str] = set()
    for info, name, metadata, records, invalid, color_matcher in candidates:
        if invalid or identities[name] != 1:
            raise PayloadOwnershipError("foreign distribution inventory is ambiguous")
        metadata_record = records.get(f"{info.name}/METADATA")
        if metadata_record is None or not _verified_record(metadata, metadata_record):
            raise PayloadOwnershipError("foreign distribution metadata does not match its record")
        if records.keys() & expected.keys():
            raise PayloadOwnershipError("foreign distribution claims pinned payload")
        if color_matcher:
            _color_matcher_assets(site, records)
        for path in records.keys() & found.keys():
            owners.setdefault(path, []).append(name)
            data = _read(site / path)
            if len(owners[path]) != 1 or not (
                _namespace_path(path, expected)
                or (color_matcher and _empty_color_test_initializer(path, data, expected))
            ):
                raise PayloadOwnershipError("foreign payload overlaps a protected package")
            if not _verified_record(data, records[path]) or hashlib.sha256(data).hexdigest() != found[path]:
                raise PayloadOwnershipError("foreign payload does not match its record")
            excluded.add(path)
    return excluded
