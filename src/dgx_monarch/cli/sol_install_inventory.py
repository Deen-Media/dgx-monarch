"""Canonical installed distributions, with one proven editable-source exception."""
from __future__ import annotations

import importlib.metadata as md
import json
import re
import stat
import sysconfig
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote, urlsplit


class DistributionInventoryError(ValueError):
    """A bounded metadata certainty refusal, never arbitrary metadata text."""


def _name(distribution: md.Distribution) -> str:
    name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", name):
        raise DistributionInventoryError("invalid distribution inventory name")
    return name


def _metadata_path(distribution: md.Distribution) -> Path | None:
    # stdlib exposes no public metadata-directory identity API. Limit this
    # private attribute to its concrete filesystem PathDistribution provider;
    # zip/custom providers cannot establish the editable exception.
    if not isinstance(distribution, md.PathDistribution):
        return None
    path = getattr(distribution, "_path", None)
    if not isinstance(path, Path) or path.is_symlink():
        return None
    path = path.resolve(strict=True)
    return path if path.is_dir() else None


def _editable_pair(name: str, candidates: list[md.Distribution], sites: set[Path]) -> md.Distribution:
    if len(candidates) != 2 or len({item.version for item in candidates}) != 1:
        raise DistributionInventoryError(f"duplicate distribution {name} is not a proven editable pair")
    paths = [_metadata_path(item) for item in candidates]
    authoritative = [index for index, path in enumerate(paths)
                     if path is not None and path.name.endswith(".dist-info") and path.parent in sites]
    if len(authoritative) != 1:
        raise DistributionInventoryError(f"duplicate distribution {name} lacks one installed authority")
    index = authoritative[0]
    installed_path, source_path = paths[index], paths[1 - index]
    if (installed_path is None or source_path is None or not source_path.name.endswith(".egg-info")
            or re.sub(r"[-_.]+", "-", source_path.name[:-9]).lower() != name):
        raise DistributionInventoryError(f"duplicate distribution {name} lacks a matching source companion")
    direct = installed_path / "direct_url.json"
    try:
        info = direct.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
            raise ValueError("unusable metadata")
        value = json.loads(direct.read_text())
        url = urlsplit(value["url"])
        if (value.get("dir_info", {}).get("editable") is not True or url.scheme != "file"
                or url.netloc not in ("", "localhost") or url.query or url.fragment):
            raise ValueError("unproven editable origin")
        named_root = Path(unquote(url.path, errors="strict"))
        if not named_root.is_absolute():
            raise ValueError("unproven project root")
        root = named_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("unproven project root")
        # Support the conventional flat/src setuptools companion only. Merely
        # being somewhere beneath the editable repository is insufficient.
        if source_path.parent not in {root, root / "src"}:
            raise ValueError("unproven source companion location")
        if named_root.resolve(strict=True) != root:
            raise ValueError("editable project root changed")
    except Exception:
        raise DistributionInventoryError(f"duplicate distribution {name} editable origin unproven") from None
    return candidates[index]


def distribution_inventory(distributions: Iterable[md.Distribution] | None = None,
                           *, site_roots: Iterable[Path] | None = None) -> dict[str, md.Distribution]:
    """Deduplicate identical metadata inodes, then prove any editable pair.

    Different versions, multiple installed dist-info directories, or an
    unproven source egg-info are always refused. The installed dist-info owns
    package requirements and RECORD inventory for an accepted editable pair.
    """
    sites = {path.resolve(strict=True) for path in (site_roots if site_roots is not None else
             (Path(sysconfig.get_path("purelib")), Path(sysconfig.get_path("platlib"))))}
    groups: dict[str, list[md.Distribution]] = {}
    physical: dict[tuple[int, int], tuple[str, str]] = {}
    for distribution in md.distributions() if distributions is None else distributions:
        name = _name(distribution)
        path = _metadata_path(distribution)
        if path is not None:
            info = path.stat()
            key = (info.st_dev, info.st_ino)
            identity = (name, distribution.version)
            if key in physical:
                if physical[key] != identity:
                    raise DistributionInventoryError("distribution metadata changed during inventory")
                continue
            physical[key] = identity
        groups.setdefault(name, []).append(distribution)
    return {name: rows[0] if len(rows) == 1 else _editable_pair(name, rows, sites)
            for name, rows in groups.items()}
