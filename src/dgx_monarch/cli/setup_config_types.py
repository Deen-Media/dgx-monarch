"""Validated snapshot and mutation records for cluster config publication."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


def _identity(value: object) -> bool:
    return isinstance(value, tuple) and len(value) == 2 and all(type(item) is int and item >= 0 for item in value)


@dataclass(frozen=True)
class ConfigSnapshot:
    path: Path
    existed: bool
    content: bytes
    digest: str | None
    mode: int | None
    identity: tuple[int, int] | None

    def __post_init__(self) -> None:
        digest = hashlib.sha256(self.content).hexdigest() if type(self.content) is bytes else None
        common = (
            isinstance(self.path, Path)
            and self.path.is_absolute()
            and type(self.existed) is bool
            and type(self.content) is bytes
        )
        present = self.digest == digest and type(self.mode) is int and _identity(self.identity)
        absent = self.content == b"" and self.digest is None and self.mode is None and self.identity is None
        if not common or (present if self.existed else absent) is not True:
            raise ValueError("invalid setup config snapshot")


@dataclass(frozen=True)
class ConfigMutation:
    path: Path
    changed: bool
    prior: ConfigSnapshot
    new_digest: str
    backup_path: Path | None
    installed_identity: tuple[int, int] | None
    durable: bool

    @property
    def rollback_available(self) -> bool:
        return self.changed

    def __post_init__(self) -> None:
        digest = isinstance(self.new_digest, str) and len(self.new_digest) == 64
        digest = digest and all(char in "0123456789abcdef" for char in self.new_digest)
        backup = self.backup_path is None or (isinstance(self.backup_path, Path) and self.backup_path.is_absolute())
        identity = self.installed_identity is None or _identity(self.installed_identity)
        no_change = (
            self.prior.existed
            and self.prior.mode == 0o600
            and self.installed_identity == self.prior.identity
            and self.new_digest == self.prior.digest
            and self.backup_path is None
            and self.durable
        )
        if (
            not isinstance(self.path, Path)
            or not self.path.is_absolute()
            or not isinstance(self.prior, ConfigSnapshot)
            or self.path != self.prior.path
            or type(self.changed) is not bool
            or type(self.durable) is not bool
            or not digest
            or not backup
            or not identity
            or (self.changed and self.installed_identity is None)
            or (self.changed and self.prior.existed and self.backup_path is None)
            or (not self.changed and not no_change)
        ):
            raise ValueError("invalid setup config mutation")
