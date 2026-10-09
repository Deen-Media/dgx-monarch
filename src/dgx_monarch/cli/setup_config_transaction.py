"""Track one config publish so an interrupted caller can still prove, and roll back, its own write."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from .setup_config_types import ConfigMutation, ConfigSnapshot

ConfigPublisher = Callable[..., ConfigMutation]


class ConfigTransaction:
    """Track publication across interruptions using the prepared file's inode.

    The publisher must call ``note_prepared`` before moving the temporary file.
    Otherwise ``recover_mutation`` cannot identify a write interrupted after the
    move.
    """

    def __init__(
        self,
        path: Path,
        text: str,
        expected: ConfigSnapshot,
        publisher: ConfigPublisher,
    ) -> None:
        target = path.expanduser().absolute()
        if target != expected.path or not isinstance(text, str) or not callable(publisher):
            raise ValueError("invalid setup config transaction")
        self.path = target
        self.text = text
        self.expected = expected
        self.publisher = publisher
        self.new_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        self._prepared_identity: tuple[int, int] | None = None
        self._backup_path: Path | None = None
        self._durable = False
        self._mutation: ConfigMutation | None = None

    def note_prepared(self, identity: tuple[int, int], backup_path: Path | None) -> None:
        if (
            not isinstance(identity, tuple)
            or len(identity) != 2
            or any(type(value) is not int or value < 0 for value in identity)
            or (self.expected.existed and backup_path is None)
        ):
            raise ValueError("invalid prepared setup config identity")
        self._prepared_identity = identity
        self._backup_path = backup_path

    def note_durable(self, durable: bool) -> None:
        if type(durable) is not bool:
            raise ValueError("invalid setup config durability")
        self._durable = durable

    def note_mutation(self, mutation: ConfigMutation) -> None:
        if (
            not isinstance(mutation, ConfigMutation)
            or mutation.path != self.path
            or mutation.prior != self.expected
            or mutation.new_digest != self.new_digest
        ):
            raise ValueError("setup config publisher returned an unrelated mutation")
        self._mutation = mutation

    def publish(self) -> ConfigMutation:
        mutation = self.publisher(
            self.path,
            self.text,
            expected=self.expected,
            _transaction=self,
        )
        self.note_mutation(mutation)
        return mutation

    def recover_mutation(self) -> ConfigMutation | None:
        """Return the recorded mutation, or reconstruct it from the prepared inode.

        Recovery requires the new bytes at mode 0600. Without a prepared inode, only a
        no-op can be proven: the reviewed file already contained those bytes at 0600
        and remains unchanged.
        """
        if self._mutation is not None:
            return self._mutation
        if self._prepared_identity is None:
            if (
                self.expected.existed
                and self.expected.digest == self.new_digest
                and self.expected.mode == 0o600
                and self.unchanged()
            ):
                return ConfigMutation(
                    self.path,
                    False,
                    self.expected,
                    self.new_digest,
                    None,
                    self.expected.identity,
                    True,
                )
            return None
        from .setup_config_io import read_snapshot

        try:
            current = read_snapshot(self.path)
        except OSError:
            return None
        if (
            not current.existed
            or current.identity != self._prepared_identity
            or current.digest != self.new_digest
            or current.mode != 0o600
        ):
            return None
        return ConfigMutation(
            self.path,
            True,
            self.expected,
            self.new_digest,
            self._backup_path,
            self._prepared_identity,
            self._durable,
        )

    def unchanged(self) -> bool:
        """Return true only when the exact reviewed snapshot still occupies the path."""
        from .setup_config_io import read_snapshot

        try:
            current = read_snapshot(self.path)
        except OSError:
            return False
        return (
            current.existed == self.expected.existed
            and current.content == self.expected.content
            and current.digest == self.expected.digest
            and current.mode == self.expected.mode
            and current.identity == self.expected.identity
        )
