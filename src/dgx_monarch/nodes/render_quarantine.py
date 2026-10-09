"""The persisted identity-gate quarantine: carrying a known FAIL forward.

A ceremony that FAILed wrote its verdict to the gate ledger. A later driver or
worker session has no memory of that render, so before setup this module reads
the verdict back and turns off every guarded residency lever the FAIL impugns.

Two callers reach it: the normal submit path in ``nodes/render_submit`` and the
Fleet dispatch in ``nodes/fleet``. The auto-gate orchestration in
``nodes/auto_gate`` reaches ``_apply_persisted_quarantine`` through its runtime
module, which is why ``nodes/common`` re-exports that one name.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import residency_mode
from ..log import get_logger
from ..mesh import MeshHandle
from ..topology import Topology

if TYPE_CHECKING:
    from .common import ModelSpec

log = get_logger(__name__)


_QUARANTINE_LEVERS = residency_mode.QUARANTINE_LEVERS


def _apply_persisted_quarantine(model: ModelSpec, entry: Any) -> list[str]:
    # Per-lever metadata explains the finding; it is not a positive grant for
    # the other risky path. A FAIL changes the capability context, so only a
    # separate exact PASS for that post-quarantine context can re-enable one.
    reported = entry.get("quarantine_levers") if isinstance(entry, dict) else None
    if not residency_mode.quarantine_metadata_exact(reported):
        log.warning(
            "persisted identity-gate FAIL reports %r; expanding quarantine "
            "to every guarded residency path", reported,
        )
    # Return the keys written, not the whole table: comfy_managed is written
    # only when the graph already carries it (a quarantine must never add that
    # key to a classic render's capability context), so naming it in the
    # caller's log would claim a lever this graph never had.
    values = residency_mode.quarantine_values(model.mesh.worker_args)
    model.mesh.worker_args.update(values)
    return list(values)


def _enforce_persisted_quarantine(
    model: ModelSpec,
    resolved_handle: MeshHandle | None = None,
    *,
    resolved_topology: Topology | None = None,
    resolved_attention: str | None = None,
) -> None:
    """Carry a known FAIL into a fresh driver/worker session before setup."""
    from .common import _AUTO_GATE_ACTIVE

    if getattr(_AUTO_GATE_ACTIVE, "on", False):
        return  # an explicit ceremony must be able to re-test the lazy path
    try:
        from ..gate_ledger import GateLedger, comfy_commit
        from .gate import (
            _combo_of,
            _effective_worker_args,
            _ledger_dir,
            gate_capability_context,
        )

        # submit_render may have replaced a defunct cached graph handle. Bind
        # the ledger lookup to the exact live handle that this request will
        # set up and dispatch, never the stale ModelSpec reference.
        handle = (resolved_handle if resolved_handle is not None
                  else getattr(model.mesh, "handle", None))
        effective = (_effective_worker_args(model, handle) if handle is not None
                     else dict(model.mesh.worker_args))
        from .gate_fsdp import topology_requires_fsdp_proof
        if topology_requires_fsdp_proof(resolved_topology, model.loras):
            # FSDP has no stock-residency fallback within the same topology.
            # Its exact PASS/denial is enforced by normal-render authorization.
            return
        lora_risk = bool(model.loras) and effective.get("lora_low_rss") is not False
        slab_risk = effective.get("slab_weights") is not False or residency_mode.requested(effective)
        if not lora_risk and not slab_risk:
            return
        key, artifacts = _combo_of(model)
        ledger = GateLedger(_ledger_dir())
        context = (gate_capability_context(
            model,
            handle,
            resolved_topology=resolved_topology,
            resolved_attention=resolved_attention,
        ) if handle is not None else
                   {"worker_args": dict(model.mesh.worker_args)})
        # One ledger read keeps the verdict and its diagnostic cause coherent;
        # a concurrent writer between separate reads could otherwise pair this
        # FAIL with a different outcome's metadata.
        state, entry = ledger.lookup_with_entry(key, artifacts, comfy_commit(), context)
    except Exception as exc:
        # A prior auto-gate lookup may already have observed sticky FAIL.  A
        # transient/corrupt second read cannot be treated as permission to
        # render risky residency, so disable every guarded path before setup.
        model.mesh.worker_args.update(residency_mode.quarantine_values(model.mesh.worker_args))
        log.error(
            "could not check persisted identity-gate quarantine (%r); "
            "forcing every guarded residency path off", exc)
        return
    if state != "fail":
        return
    # submit_render's ensure_setup sees the changed worker-args key and either
    # starts fresh workers in stock mode or pushes the change to live workers
    # before dispatch. Mutate locally first so a failed RPC cannot leave a
    # workflow believing it is quarantined when it is not.
    levers = _apply_persisted_quarantine(model, entry)
    log.error(
        "identity gate previously FAILED for %s; enforcing persisted quarantine "
        "(%s=off) before render", model.unet_name, "+".join(levers),
    )
    try:
        from ..telemetry import emit

        emit("quarantine", model=model.unet_name, origin="persisted_fail")
    except Exception:
        pass
