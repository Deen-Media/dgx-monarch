"""Stable local source identity bound into setup plans and deployments."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..runtime_provenance import dgx_source_manifest_sha256


@dataclass(frozen=True)
class SetupSource:
    root: Path = field(repr=False)
    manifest: str

    def __post_init__(self) -> None:
        if not isinstance(self.root, Path) or not self.root.is_absolute():
            raise ValueError("setup source root must be an absolute Path")
        if not isinstance(self.manifest, str) or re.fullmatch(r"[0-9a-f]{64}", self.manifest) is None:
            raise ValueError("setup source manifest must be a SHA-256 digest")


def read_setup_source() -> SetupSource:
    root = Path(__file__).resolve(strict=True).parents[1]
    return SetupSource(root, dgx_source_manifest_sha256(root))
