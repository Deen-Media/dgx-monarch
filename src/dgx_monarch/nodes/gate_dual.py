"""The first-use identity gate for a dual-model render (Ideogram 4).

A dual-model render runs two checkpoints: the conditional model in the first
slot and the unconditional model in the second. Its proof is the dual render
itself, run as every ceremony runs: both slots from a clean load, a lazy-swap
cycle on each slot that carries a LoRA stack, the render again, and with slab
residency on, one more render with slab off. A PASS binds both checkpoints and
both stacks under the dual row identity (``gate_dual_identity``), and only a
dual PASS can grant a dual render its residency.

Three placements get no dual proof, because the proof cannot run there:

- A cfg-parallel topology (explicit cfg2, dm-cfg2): each rank holds one of the
  two models, and a rank cannot swap-test the model it does not hold. The
  render runs on stock residency, as it did before.
- An FSDP topology: the clean-reload proof reloads one model slot. Under
  ``auto_gate=first_use`` the render refuses, typed.
- Comfy-managed residency: no proof covers two models under it, so the stock
  stamp meets the Worker's comfy-managed refusal.

The conditional slot's swap report stands for both slots' residency: one
worker policy and one family govern both, and the slab-off leg compares the
whole dual render.
"""
from __future__ import annotations

import copy
from typing import Any

from .. import mesh_setup, residency_mode
from ..fsdp_proof_scope import topology_resolves_fsdp
from ..log import get_logger
from ..transfer_utils import safe_call
from .gate_inconclusive import CYCLE_KEY

log = get_logger(__name__)

# Why each placement gets no dual proof. Read-only after import.
CFG_PARALLEL_REASON = ("each rank of a cfg-parallel topology holds one of the two models, and a "
                       "rank cannot swap-test the model it does not hold")
FSDP_REASON = "the FSDP clean-reload proof reloads one model slot, not both"
COMFY_MANAGED_REASON = ("the first-use gate proves a dual-model render only under classic "
                        "residency; no comfy-managed proof covers two models")


def _cfg_degree(topology: Any) -> int:
    if topology is None:
        return 1
    value = topology.get("cfg") if isinstance(topology, dict) else getattr(topology, "cfg", 1)
    return int(value or 1)


def skip_reason(resolved_topology: Any, effective_worker_args: dict | None) -> str | None:
    """Why the first-use gate cannot prove a dual-model render here, or None when it can."""
    if residency_mode.requested(effective_worker_args):
        return COMFY_MANAGED_REASON
    if topology_resolves_fsdp(resolved_topology):
        return FSDP_REASON
    if _cfg_degree(resolved_topology) > 1:
        return CFG_PARALLEL_REASON
    return None


def stock_by_rule(request: dict, topology: Any) -> bool:
    """Whether this dispatch is a dual-model render a cfg-parallel topology keeps on stock."""
    return request.get("uncond_model") is not None and _cfg_degree(topology) > 1


def forced_stock_state(request: dict, topology: Any) -> str:
    """The state a stock dispatch notes (``refusal_lead``): why it could not keep slab."""
    from .refusal_lead import DUAL_MODEL, DUAL_UNPROVEN

    if request.get("uncond_model") is None:
        return "unknown"
    return DUAL_MODEL if stock_by_rule(request, topology) else DUAL_UNPROVEN


def refuse_unprovable(model: Any, latent: dict, cfg_value: float, handle: Any) -> None:
    """Raise before any ceremony side effect when the gate cannot prove this dual render."""
    from .common import _resolve_topology_for_latent
    from .gate import _effective_worker_args

    topology = None
    if hasattr(model.mesh, "topology_preset"):  # lightweight rigs carry no topology
        topology, _sage, _why = _resolve_topology_for_latent(
            model, latent, cfg_value, int(getattr(handle, "world", 1)))
    reason = skip_reason(topology, _effective_worker_args(model, handle))
    if reason is not None:
        raise ValueError(f"the identity gate cannot prove this dual-model render: {reason}")


def ceremony_stacks(model_request: dict, request: dict) -> tuple[list[dict], list[dict]]:
    """Every LoRA entry the proof covers, and the conditional slot's own stack."""
    cond = [dict(entry) for entry in model_request.get("loras") or []]
    uncond = request.get("uncond_model") or {}
    return [*cond, *(dict(entry) for entry in uncond.get("loras") or [])], cond


def fold_uncond_cycle(handle: Any, cond_rows: list, request: dict, binding: dict,
                      setup_token: Any) -> list:
    """Add the unconditional slot's swap cycle to each rank's report when it carries a stack.

    A slot with no stack has nothing to swap, so it neither proves nor spoils
    the fold; a slot with a stack must hot-swap both ways on every rank.
    """
    uncond = request.get("uncond_model")
    stack = [dict(entry) for entry in (uncond or {}).get("loras") or []]
    if uncond is None or not stack:
        return cond_rows
    uncond_rows = handle.call_all(
        "gate_swap_cycle", uncond["unet_name"], dict(uncond.get("options") or {}), stack,
        binding["artifact_sets"][1], timeout_s=1800, slot="uncond",
        **mesh_setup.token_kwargs(setup_token))
    by_rank = {row.get("rank"): row for row in uncond_rows or () if isinstance(row, dict)}
    return [_fold(row, by_rank.get(row.get("rank"))) if isinstance(row, dict) else row
            for row in cond_rows]


def _fold(cond: dict, uncond: dict | None) -> dict:
    if uncond is None or uncond.get("setup_generation") != cond.get("setup_generation"):
        return {**cond, "conclusive": False, CYCLE_KEY: False,
                "reason": "no current unconditional-slot swap report from this rank"}
    cond_settled = bool(cond.get("conclusive")) or bool(cond.get(CYCLE_KEY))
    swapped = bool(uncond.get("conclusive")) and cond_settled
    row = {**cond, "conclusive": swapped, CYCLE_KEY: False,
           "transitions": {"cond": cond.get("transitions"), "uncond": uncond.get("transitions")}}
    row.pop("reason", None)
    if not swapped:
        reasons = [str(text) for text in (
            None if cond.get(CYCLE_KEY) else cond.get("reason"), uncond.get("reason")) if text]
        row["reason"] = "; ".join(reasons) or "the unconditional slot's swaps fell back to reload"
    return row


def authorize_dual(model: Any, handle: Any, uncond: dict, *, requested: dict,
                   model_request: dict, effective: dict, gate_active: bool, session_verdict,
                   resolved_topology: Any, resolved_attention: str):
    """Authorize one dual-model dispatch: a dual PASS grant, stock, or the FSDP refusal."""
    from ..mesh_safety import fleet_policy_is_risky
    from .gate_fsdp import FsdpGateProofError
    from .gate_identity import NormalRenderAuthorization

    stock = NormalRenderAuthorization(
        {**requested, "slab_weights": False, "lora_low_rss": False}, model_request, None, "stock")
    first_use = getattr(model.mesh, "auto_gate", "off") == "first_use"
    if topology_resolves_fsdp(resolved_topology):
        if first_use and not gate_active:
            # A literal opening: the sweep reads untagged refusals from raise sites.
            raise FsdpGateProofError(
                "automatic FSDP execution is unavailable for a dual-model request: "
                f"{FSDP_REASON}, so the first-use gate cannot prove both checkpoints. Use a "
                "topology without FSDP, or set auto_gate=off to run it under FSDP with slab "
                "and low-RSS residency off")
        return stock
    risky = any(fleet_policy_is_risky(spec, effective) for spec in (model_request, uncond))
    reason = skip_reason(resolved_topology, effective)
    if not first_use or reason is not None:
        if reason is not None and risky:
            safe_call(log.warning, "dual-model render: %s, so the identity gate cannot prove "
                      "it; this dispatch runs on stock residency", reason)
        return stock
    if gate_active:
        return NormalRenderAuthorization(requested, model_request, None, "gate_internal")
    if not risky:
        return NormalRenderAuthorization(requested, model_request, None, "stock")
    from ..worker_runtime import WorkerRuntimeMismatch

    try:
        granted = _dual_grant(model, handle, uncond, requested, model_request, effective,
                              session_verdict, resolved_topology, resolved_attention)
    except WorkerRuntimeMismatch:
        raise  # two torch builds hold no Gate authority (worker_runtime.py)
    except Exception as exc:
        safe_call(log.error, "dual-model Gate authorization failed (%r); forcing stock "
                  "residency", exc)
        return stock
    return stock if granted is None else granted


def _dual_grant(model, handle, uncond, requested, model_request, effective, session_verdict,
                resolved_topology, resolved_attention):
    """The ``required`` authorization a dual PASS earns, or None after logging why not."""
    from .. import __version__, gate_ledger
    from . import gate_identity, gate_inconclusive
    from .gate import _ledger_dir

    captured, key, identity, artifacts, binding = gate_identity.capture(model, uncond)
    if captured != model_request:
        raise RuntimeError("dual-model request changed during authorization")
    context = gate_identity.gate_capability_context(
        model, handle, effective_worker_args=effective,
        resolved_topology=resolved_topology, resolved_attention=resolved_attention)
    if context["resolved_topology"] is None:
        raise RuntimeError("dual-model authorization has no concrete topology")
    commit = identity["comfy"]
    token = gate_ledger.gate_verdict_token(key, artifacts.current, commit, context)
    lookup = gate_ledger.GateLedger(_ledger_dir()).lookup_with_integrity(
        key, artifacts, commit, context)
    process_verdict = session_verdict(token)
    denied = process_verdict in ("FAIL", "INCONCLUSIVE", "ERROR")
    entry = lookup.entry
    if not denied and (lookup.state == "pass" or (
            lookup.session_pass_safe and (entry is None or entry.get("verdict") == "PASS")
            and lookup.state in ("unknown", "stale") and process_verdict == "PASS")):
        return gate_identity.NormalRenderAuthorization(requested, model_request, {
            "capability": "normal_render_residency",
            "gate_protocol": gate_ledger.GATE_PROTOCOL_VERSION,
            "dgx_monarch": __version__,
            "comfy": commit,
            "artifact_digest": artifacts.current,
            "gate_token": list(token),
            "combo_key": key,
            "model_request": copy.deepcopy(model_request),
            "uncond_model_request": copy.deepcopy(uncond),
            "artifact_sets": copy.deepcopy(binding["artifact_sets"]),
            "capability_context": copy.deepcopy(context),
        }, "required")
    if gate_inconclusive.dispatch_grants_skip(lookup, token, process_verdict):
        log.info("dual-model render: the identity gate found nothing to gate for this "
                 "combination, so it keeps that verdict; this dispatch runs on stock residency")
        return None
    log.warning("dual-model render has no exact Gate PASS for the current %s state; forcing "
                "stock residency for this dispatch", lookup.state)
    if requested.get("slab_weights") is True:
        from .refusal_lead import DUAL_UNPROVEN, note_forced_stock

        note_forced_stock(DUAL_UNPROVEN)  # an Identity Gate node's PASS would not grant it
    return None
