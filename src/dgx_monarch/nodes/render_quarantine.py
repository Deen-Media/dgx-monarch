"""The persisted identity-gate quarantine: carrying a known FAIL forward.

A ceremony that FAILed wrote its verdict to the gate ledger, and in this
process to the process verdict cache. A later render has no memory of that
ceremony, so before setup this module reads the verdict back and returns a
per-dispatch clone of the model with every guarded residency lever the FAIL
impugns turned off.

The quarantine belongs to one checkpoint combination in one capability
context (docs/DESIGN.md, invariant 2). Every checkpoint wired to one Init node
shares that node's policy dict, so a FAIL never writes ``lora_low_rss`` or
``slab_weights`` into it: another checkpoint on the same Init keeps the levers
the graph asked for and runs its own first-use ceremony. Two writes still land
on the shared dict. ``comfy_managed`` is a bootstrap policy that a live worker
cannot change for one dispatch, so a FAIL turns it off for the whole Init
node. An unreadable ledger turns every guarded lever off on the graph, because
nothing then says which combination is safe.

Two callers dispatch the clone: the normal submit path in
``nodes/render_submit`` and the Fleet call in ``nodes/fleet``. ``nodes/common``
re-exports ``_apply_persisted_quarantine`` for tests and older callers.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import residency_mode
from ..log import get_logger
from ..mesh import MeshHandle
from ..topology import Topology
from .gate_identity import model_with_worker_overrides
from .gate_quarantine_scope import combination_label

if TYPE_CHECKING:
    from .common import ModelSpec

log = get_logger(__name__)


_QUARANTINE_LEVERS = residency_mode.QUARANTINE_LEVERS


def _quarantined_dispatch(model: ModelSpec, handle: Any = None) -> tuple[Any, list[str]]:
    """Clone ``model`` for one dispatch with every guarded lever off."""
    # Name the keys this dispatch turns off, not the whole table: comfy_managed
    # is written only when the graph already carries it (a quarantine must
    # never add that key to a classic render's capability context), so naming
    # it in the caller's log would claim a lever this graph never had.
    values = residency_mode.quarantine_values(model.mesh.worker_args)
    if residency_mode.WORKER_ARG in values:
        # A live worker refuses a comfy_managed change, and turning it off for
        # one checkpoint only would need a mesh reset at every switch between
        # checkpoints. So this lever alone goes off for the whole Init node.
        model.mesh.worker_args[residency_mode.WORKER_ARG] = False
    return model_with_worker_overrides(model, values, handle=handle), list(values)


def _apply_persisted_quarantine(
    model: ModelSpec, entry: Any, handle: Any = None,
) -> tuple[Any, list[str]]:
    """Return a per-dispatch clone with a ledger FAIL's levers off, and their names."""
    # Per-lever metadata explains the finding; it is not a positive grant for
    # the other risky path. A FAIL changes the capability context, so only a
    # separate exact PASS for that post-quarantine context can re-enable one.
    reported = entry.get("quarantine_levers") if isinstance(entry, dict) else None
    if not residency_mode.quarantine_metadata_exact(reported):
        log.warning(
            "persisted identity-gate FAIL reports %r; expanding quarantine "
            "to every guarded residency path", reported,
        )
    return _quarantined_dispatch(model, handle)


def _enforce_persisted_quarantine(
    model: ModelSpec,
    resolved_handle: MeshHandle | None = None,
    *,
    resolved_topology: Topology | None = None,
    resolved_attention: str | None = None,
) -> Any | None:
    """Carry a known FAIL into this dispatch before setup.

    Return the per-dispatch clone to render with, or None when this
    combination carries no FAIL in this capability context and the caller
    keeps its model. The graph's own lora_low_rss and slab_weights stay as
    the graph asked, so other checkpoints on the same Init node keep them.
    """
    from .common import _AUTO_GATE_ACTIVE

    if getattr(_AUTO_GATE_ACTIVE, "on", False):
        return None  # an explicit ceremony must be able to re-test the lazy path
    try:
        from ..gate_ledger import GateLedger, comfy_commit, gate_verdict_token
        from .common import _process_gate_verdict
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
            return None
        lora_risk = bool(model.loras) and effective.get("lora_low_rss") is not False
        slab_risk = effective.get("slab_weights") is not False or residency_mode.requested(effective)
        if not lora_risk and not slab_risk:
            return None
        key, artifacts = _combo_of(model)
        ledger = GateLedger(_ledger_dir())
        commit = comfy_commit()
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
        state, entry = ledger.lookup_with_entry(key, artifacts, commit, context)
        # A FAIL measured in this session is also held in the process verdict
        # cache, so a ledger write that failed cannot lift it. The ledger API
        # also takes a bare digest string for the artifact set.
        digest = artifacts if isinstance(artifacts, str) else artifacts.current
        session_fail = state != "fail" and _process_gate_verdict(
            gate_verdict_token(key, digest, commit, context)) == "FAIL"
    except Exception as exc:
        # A prior auto-gate lookup may already have observed sticky FAIL.  A
        # transient/corrupt second read cannot be treated as permission to
        # render risky residency, and nothing says which combination is safe,
        # so disable every guarded path on the graph before setup.
        model.mesh.worker_args.update(residency_mode.quarantine_values(model.mesh.worker_args))
        log.error(
            "could not check persisted identity-gate quarantine (%r); "
            "forcing every guarded residency path off", exc)
        return None
    if state != "fail" and not session_fail:
        return None
    # The caller's setup sees the clone's worker args and either starts fresh
    # workers in stock mode or pushes the change to live workers before
    # dispatch. A failed RPC cannot leave this dispatch believing it is
    # quarantined when it is not: the policy it sends is the clone's.
    if session_fail:
        quarantined, levers = _quarantined_dispatch(model, resolved_handle)
    else:
        quarantined, levers = _apply_persisted_quarantine(model, entry, resolved_handle)
    # comfy_managed is the one lever that leaves the shared dict too, so the
    # line names the residency pair as per-combination and it as Init-wide.
    pair = [lever for lever in levers if lever != residency_mode.WORKER_ARG]
    log.error(
        "identity gate %s for %s; this dispatch renders with %s off for that "
        "combination in this capability context, and other checkpoints on "
        "this Init node keep theirs%s",
        "FAILED earlier this session" if session_fail else "previously FAILED",
        combination_label(model), "+".join(pair),
        ". comfy_managed is now off for the whole Init node, because a live "
        "worker cannot change it for one dispatch"
        if residency_mode.WORKER_ARG in levers else "",
    )
    try:
        from ..telemetry import emit

        emit("quarantine", model=model.unet_name,
             origin="session_fail" if session_fail else "persisted_fail")
    except Exception:
        pass
    return quarantined
