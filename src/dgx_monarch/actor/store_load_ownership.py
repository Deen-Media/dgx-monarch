"""Durable provisional ownership for one ModelStore fresh-load transaction."""
from __future__ import annotations

from typing import Any

from ..transfer_utils import raise_with_distinct_cause, reconcile_error
from . import slab_lifetime


def _discard_identity(handoff: list[Any], owned: Any) -> None:
    for index, candidate in enumerate(handoff):
        if candidate is owned:
            handoff.pop(index)
            return


class FreshLoadOwnership:
    """Process-root pin/slab handoffs until cleanup or slot adoption."""

    def __init__(self) -> None:
        self.pin_handoff: list[Any] = []
        self.slab_handoff: list[Any] = []
        self._adoption_store: Any = None
        self._adoption_attr: str | None = None
        self._adoption_model: Any = None
        self._adoption_ready = False
        # This call has no resource-bearing predecessor: if its return/store is
        # interrupted, the rooted transaction is still empty and recoverable.
        self._poisoned_before = slab_lifetime.prepublish_resource(self)

    def prepare_adoption(self, store: Any, slot_attr: str, stored: Any) -> None:
        """Record the exact candidate before its potentially interrupted store."""
        self._adoption_model = stored
        self._adoption_attr = slot_attr
        self._adoption_store = store
        self._adoption_ready = True

    def _is_published(self) -> bool:
        return bool(
            self._adoption_ready
            and self._adoption_store is not None
            and self._adoption_attr is not None
            and getattr(self._adoption_store, self._adoption_attr)
            is self._adoption_model
        )

    def _confirm_adopted_children(self) -> None:
        for owned in tuple(self.slab_handoff):
            confirm = getattr(owned, "confirm_handoff", None)
            if confirm is not None:
                confirm()

    def close(self) -> None:
        """Close every exact child after the caller's confirmed global unload."""
        if self._is_published():
            # The exact slot is now the durable owner. A recovery interruption
            # must never let provisional cleanup close underneath that resident.
            self._confirm_adopted_children()
            self.pin_handoff.clear()
            self.slab_handoff.clear()
            return
        failure: BaseException | None = None
        failure_cause: BaseException | None = None
        for handoff in (self.slab_handoff, self.pin_handoff):
            for owned in tuple(handoff):
                try:
                    owned.close()
                    _discard_identity(handoff, owned)
                except BaseException as exc:
                    failure, failure_cause = reconcile_error(
                        failure,
                        exc,
                        "another fresh-load child cleanup also failed",
                    )
        if failure is not None:
            raise_with_distinct_cause(failure, failure_cause)

    def cleanup(self, error: BaseException) -> bool:
        """Clean a failed load while preserving this prepublished root."""
        cleaned = slab_lifetime.cleanup_failed_load(
            None,
            error,
            "model-store failed load",
            resources=(self,),
        )
        if cleaned:
            slab_lifetime.confirm_prepublished_resource(
                self, self._poisoned_before)
        return cleaned

    def disarm_and_confirm(self) -> None:
        """Transfer both children to an identity-confirmed StoredModel slot."""
        if not self._is_published():
            raise RuntimeError("fresh-load ownership has no exact published slot")
        self._confirm_adopted_children()
        self.pin_handoff.clear()
        self.slab_handoff.clear()
        slab_lifetime.confirm_prepublished_resource(
            self, self._poisoned_before)
