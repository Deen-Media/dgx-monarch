"""Owned resources and identity for one resident model-store slot."""
from __future__ import annotations

import time
import weakref
from dataclasses import dataclass, field
from typing import Any

from ..transfer_utils import raise_with_distinct_cause, reconcile_error
from .store_detect import LivePrecisionEvidence


@dataclass
class StoredModel:
    base_key: tuple
    request_key: tuple
    base_patcher: Any
    active_patcher: Any
    quant_kind: str
    precision_evidence: LivePrecisionEvidence
    family: str
    artifact_identity: dict
    lora_sig: tuple = ()
    loaded_at: float = field(default_factory=time.time)
    unbake: Any = None
    slab: Any = None
    slab_auto_retry: bool = False
    fsdp_checkpoint_pin: Any = None
    residency_rung: str = ""
    # The low-RSS bake merged the stack into these weights and freed ComfyUI's
    # backup, so this base cannot be re-cloned for a different stack.
    base_baked: bool = False
    # The stack was baked on the device onto the loaded base (the default's in-place bake, or
    # the classic bake's fallback, actor/in_place_fallback.py), not by ComfyUI's own load.
    baked_in_place: bool = False
    # The pair's backup agreement freed ComfyUI's backup on this slot (actor/backup_agreement.py): a later
    # render on it has no backup to be short of, so the room check asks this flag, not the backup.
    freed_for_sampler: bool = False

    def __post_init__(self) -> None:
        # The model points back at its slot, weakly, so the one-GPU partial-load route
        # (actor/lora_partial_load.py) can mark the slot baked when it bakes the model whole.
        # A deep copy of the model would share the reference and point at this slot, and a
        # model that carries it cannot be pickled; no path in the pack copies or pickles one.
        model = getattr(self.base_patcher, "model", None)
        if model is not None:
            try:
                model._dgxm_stored = weakref.ref(self)
            except (AttributeError, TypeError):
                pass

    def close(self) -> None:
        """Close every owned resource; operational cancellation wins."""
        failure: BaseException | None = None
        failure_cause: BaseException | None = None
        for label, resource in (
            ("pinned checkpoint close also failed", self.fsdp_checkpoint_pin),
            ("weight slab close also failed", self.slab),
        ):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as exc:
                failure, failure_cause = reconcile_error(failure, exc, label)
        if failure is not None:
            raise_with_distinct_cause(failure, failure_cause)
