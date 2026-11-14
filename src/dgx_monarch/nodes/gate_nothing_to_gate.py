"""Record a no-material first-use verdict from Worker evidence, with no proof render.

A first-use ceremony on a graph with no LoRA stack, on a family whose ``auto``
slab residency resolves to stock, can only end INCONCLUSIVE with nothing to
compare (``gate_inconclusive.classify``). It reaches that verdict after an
unload, a fresh load and two proof renders at the user's own shape, and the
two renders feed nothing but the report's latent difference: on a pair, an
H3 still guide at 896x1152 and 362 frames spent 585.8 s on them before the
render began (2026-11-13). The verdict grants nothing either way: the render
that follows dispatches on stock residency.

So on the automatic path, when every term of :func:`nothing_to_gate` holds,
:func:`record_if_provable` runs the ceremony without its renders. The capacity
stop prices the request first, as in the ceremony. Then, under the
proof-cohort attestor before and after, every rank unloads, the fleet agrees
and loads this checkpoint, and the Workers run the empty-stack swap cycle,
whose stamps go through the ceremony's own ``slab_proof.establish`` and
``gate_inconclusive.classify``. Only a no-material classification writes, and
it writes the exercised context's row alone: no Fleet or explicit-on slab
sibling is revoked on an inference. The row is Worker evidence, as a
ceremony's is; it and the report say no proof render ran
(``gate_inconclusive.BASIS_KEY``), and later queues read it as they read a
ceremony's (``gate_inconclusive.row_grants_skip``).

The driver's predicate only chooses this path. Anything it cannot read, and
any Worker answer other than no material (a vouched family, slab active, a
rank missing), returns None, and today's full ceremony runs and meets the same
fault in its own words. Nothing is written before that decision, and a return
after the load unloads first, so the ceremony's capacity stop prices the
Workers as it would have. The explicit
Identity Gate node never comes here, because its output is the proof's latent.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from ..log import get_logger
from ..transfer_utils import safe_call
from . import gate_inconclusive

log = get_logger(__name__)

ORIGIN = "auto_first_use"


class _NoRender(RuntimeError):
    """``slab_proof.establish`` asked for a render; this path renders nothing."""


def _no_render(*_args: Any, **_kwargs: Any) -> dict:
    raise _NoRender("slab residency is expected for the family the Workers report")


def _declined(term: str) -> None:
    safe_call(log.info, "identity gate: the full ceremony runs, because the driver cannot "
              "rule out material to compare (%s)", term)


def _family(model: Any) -> tuple[str | None, bool]:
    """(family, forced) as a Worker resolves ``auto`` slab for this checkpoint.

    A forced family wins (``family_select.override_for_model``, the value the
    Worker's store carries as ``family_override``). Otherwise the header
    decides, as ``loader_preflight._effective_slab`` reads it for pricing. An
    unreadable or unmatched header answers (None, False).
    """
    from .. import family_select

    forced = family_select.override_for_model(model)
    if forced:
        return str(forced), True
    try:
        import folder_paths

        from ..adapters.detect import sniff_checkpoint

        path = folder_paths.get_full_path("diffusion_models", model.unet_name)
        if path is None:
            return None, False
        family = family_select.effective_family(sniff_checkpoint(path)[0], None)
    except Exception:
        return None, False
    return (None if family in (None, "", "unknown") else str(family)), False


def _rescue_consent_live(model: Any) -> bool:
    """Whether a capacity-rescue consent could make a Worker slab-load this checkpoint.

    Such a consent loads an unvouched family on slab when stock cannot fit
    (``actor/capacity_quote.py``, ``RUNG_CONSENTED_RESCUE``). ``render_grant``
    reads a failed lookup as no grant, and the dispatch's consent wire reads the
    same call (``consent_rescue.render_consent_wire``); a consent projected at
    the loader turns ``slab_weights`` on, which the slab term refuses. A raise
    here reads as live.
    """
    try:
        from .consent_rescue import render_grant

        return render_grant(model) is not None
    except Exception:
        return True


def nothing_to_gate(model: Any, request: Mapping[str, Any], effective: Mapping[str, Any],
                    resolved_topology: Any) -> str | None:
    """The driver's reason a ceremony could find nothing to compare, or None.

    Each term reads what the Worker's own decision reads; None runs today's
    ceremony and logs the term that declined:

    - one model: the request names no ``uncond_model`` (a dual proof covers both slots);
    - no LoRA stack: ``model.loras`` and the model request's ``loras`` are empty
      (``DGXMonarchLoraLoader`` is the only route a stack reaches a DGXM_MODEL,
      and ``actor/gate_cycle.py`` stamps no material on exactly this);
    - not comfy-managed (``residency_mode.requested``), where the rung itself
      is the lever under proof;
    - no FSDP layout (``fsdp_proof_scope``), which needs its clean-reload proof;
    - slab off, or ``auto`` on a known family outside ``SLAB_VOUCHED_FAMILIES``
      (``actor/store_residency.py`` resolves ``auto`` that way); slab on, an
      unknown spelling, or an unread family runs the ceremony;
    - no live capacity-rescue consent.
    """
    term, family_clause = _first_declining_term(model, request, effective, resolved_topology)
    if term is not None:
        _declined(term)
        return None
    return f"no LoRA stack, one model, {family_clause}"


def _first_declining_term(model: Any, request: Mapping[str, Any], effective: Mapping[str, Any],
                          resolved_topology: Any) -> tuple[str | None, str]:
    """(the first term that declines, or None; the slab clause of the reason)."""
    from .. import residency_mode
    from ..capacity_fit import SLAB_VOUCHED_FAMILIES
    from ..fsdp_proof_scope import topology_requires_fsdp_proof, topology_resolves_fsdp

    if request.get("uncond_model") is not None:
        return "a second model", ""
    request_dict = model.request_dict() if hasattr(model, "request_dict") else {}
    if getattr(model, "loras", None) or request_dict.get("loras"):
        return "a LoRA stack", ""
    if residency_mode.requested(effective):
        return "comfy-managed residency", ""
    if resolved_topology is None:
        return "no resolved layout", ""
    if (topology_resolves_fsdp(resolved_topology)
            or topology_requires_fsdp_proof(resolved_topology, [])):
        return "an FSDP layout", ""
    slab = effective.get("slab_weights", "auto")
    if slab is False:
        family_clause = "slab residency is off"
    elif slab == "auto":
        family, forced = _family(model)
        if family is None:
            return "the checkpoint header names no family the driver knows", ""
        if family in SLAB_VOUCHED_FAMILIES:
            return f"family {family} takes slab residency under slab_weights=auto", ""
        family_clause = (f"the forced family {family}" if forced else f"family {family}") + (
            " takes no slab residency under slab_weights=auto")
    else:
        return f"slab_weights={slab!r}", ""
    if _rescue_consent_live(model):
        return "a live capacity-rescue consent", ""
    return None, family_clause


def _resolved(runtime: Mapping[str, Any], ceremony_model: Any, latent: dict,
              cfg_value: float, handle: Any) -> tuple[Any, str]:
    """The topology and attention the ceremony would bind, read as it reads them
    (``gate_ceremony.gather_ceremony_evidence``, including its test-rig branch)."""
    from .common import _resolve_topology_for_latent, resolve_sample_attention

    if hasattr(ceremony_model.mesh, "topology_preset"):
        topology, sage, _reason = _resolve_topology_for_latent(
            ceremony_model, latent, cfg_value,
            int(getattr(handle, "world", getattr(ceremony_model.mesh, "world", 1))))
    else:  # lightweight unit-test rigs lack the public MeshSpec fields
        from ..topology import Topology

        world = int(getattr(handle, "world", 1))
        topology, sage = Topology(world=world, dp=world), False
    return topology, resolve_sample_attention(
        getattr(ceremony_model.mesh, "attention", "unknown"), sage)


def _fall_through_unloaded(handle: Any, term: str) -> None:
    """Leave the Workers empty before the full ceremony runs.

    That ceremony's capacity stop charges the DiT as a load still to come
    (``gate_capacity_stop.dit_already_loaded`` reads only the render site's
    memos), so a model this path loaded would be counted twice and could stop
    a request today's ceremony runs. Its first step after the stop is the same
    unload, so this costs nothing.
    """
    _declined(term)
    try:
        handle.call_all("unload", timeout_s=600)
    except Exception as exc:
        safe_call(log.warning, "identity gate: the unload before the full ceremony "
                  "failed (%r); the ceremony unloads again before its first render", exc)


def _checkpoint_path(model: Any) -> str | None:
    import folder_paths

    return folder_paths.get_full_path("diffusion_models", model.unet_name)


def _free_driver_for_load(ceremony_model: Any, frozen_request: dict) -> None:
    """Free the driver's models first where the Workers would not fit this load.

    Every rank has just unloaded, so the load is still to come whatever the
    render site's memos say: a model an earlier render left warm passed the
    user's render site as placed. ``pending_load_relief`` refuses nothing and
    never raises; the agreement after it decides the load as before.
    """
    from . import pending_load_relief

    try:
        family, _forced = _family(ceremony_model)
        path = _checkpoint_path(ceremony_model)
    except Exception:
        return
    if family is not None and path is not None:
        pending_load_relief.relieve(ceremony_model, frozen_request, family=family, path=path,
                                    weights_loaded=False)


def _worker_evidence(runtime: Mapping[str, Any], ceremony_model: Any, model_request: dict,
                     identity: dict, binding: dict, frozen_request: dict, frozen_latent: dict,
                     cfg_value: float, handle: Any, effective: dict, placed: dict) -> Any:
    """The ceremony's Worker evidence without its renders, attested on every rank.

    Unload first, as the ceremony does, so the cycle's family is this
    checkpoint's and not a model an earlier graph left resident. ``placed``
    turns true once the load is asked for, so a fall-through after it unloads.
    """
    from .. import capacity_agreement
    from .gate_provenance import ProofCohortAttestor
    from .slab_proof import establish

    token_kwargs = runtime["mesh_setup"].token_kwargs
    unet_name = model_request["unet_name"]
    options = dict(model_request.get("options") or {})
    # Its own attestor: a fall-through leaves the ceremony's untouched.
    attest = ProofCohortAttestor()
    token = runtime["render_setup_token"](ceremony_model, frozen_latent, cfg_value, handle)
    attest("pre", handle, token)
    handle.call_all("unload", timeout_s=600)
    _free_driver_for_load(ceremony_model, frozen_request)
    capacity_agreement.agree_load(handle, unet_name, options, [], "cond")
    placed["load"] = True
    handle.call_all("load_model", unet_name, options, [], timeout_s=1800, **token_kwargs(token))
    raw_cycle = handle.call_all("gate_swap_cycle", unet_name, options, [], identity,
                                timeout_s=1800, **token_kwargs(token))
    proof = establish(ceremony_model, model_request, frozen_request, binding, identity,
                      frozen_latent, cfg_value, 0, handle, effective, {}, raw_cycle, _no_render)
    attest("post", handle, token)
    # The Workers log nothing for a cycle, so the driver names each rank's stamp.
    safe_call(log.info, "identity gate: swap-cycle stamps, no proof render: %s", "; ".join(
        f"rank {row.get('rank')} on {row.get('host')}: family {row.get('family')}, "
        f"no_material {bool(row.get('no_material'))}, slab_active {bool(row.get('slab_active'))}, "
        f"setup generation {row.get('setup_generation')}" for row in proof.cycle))
    return proof


def record_if_provable(*, runtime: Mapping[str, Any], model: Any, request: dict, latent: dict,
                       cfg_value: float, origin: str, run_id: str, handle: Any,
                       fsdp_scope_state: Mapping[str, bool] | None = None) -> dict | None:
    """Write the no-material verdict from Worker evidence, or return None to run the ceremony.

    ``runtime`` is the public :mod:`nodes.gate` namespace, as for the ceremony.
    Only the capacity stop's own exception leaves this function; every other
    doubt returns None before anything is written.
    """
    if origin != ORIGIN or (fsdp_scope_state or {}).get("required"):
        return None
    t0 = time.perf_counter()
    try:
        from .gate_identity import capture
        from .gate_proof_request import report_fields

        # The predicate and its readers only read, so they run on the caller's
        # request and latent; a graph they decline pays no clone here, and the
        # ceremony freezes its own copy as before.
        effective = runtime["_effective_worker_args"](model, handle)
        ceremony_model = runtime["_model_with_worker_overrides"](model, effective, handle=handle)
        topology, attention = _resolved(runtime, ceremony_model, latent, cfg_value, handle)
        basis = nothing_to_gate(ceremony_model, request, effective, topology)
        if basis is None:
            return None
        frozen_request, frozen_latent = runtime["freeze_transaction"]((request, latent))
        model_request, key, identity, artifacts, binding = capture(ceremony_model, None)
        commit = identity["comfy"]
        capability_context = runtime["gate_capability_context"](
            ceremony_model, handle, effective_worker_args=effective,
            resolved_topology=topology, resolved_attention=attention)
        gate_token = runtime["gate_verdict_token"](
            key, artifacts.current, commit, capability_context)
        sampling = report_fields(frozen_request)
        ledger = runtime["GateLedger"](runtime["_ledger_dir"]())
    except Exception as exc:
        _declined(f"the driver could not read the graph: {exc!r}")
        return None
    stop = runtime["gate_capacity_stop"]
    try:  # the ceremony's first step: a stop writes its one row and raises
        stop.stop_unless_request_fits(
            ceremony_model, frozen_request, frozen_latent,
            (ledger, key, artifacts, commit, capability_context, model_request["unet_name"]),
            origin=origin, run_id=run_id, loras=0, worker_args=effective)
    except stop.CeremonyCapacityStop:
        raise
    except Exception as exc:
        _declined(f"the capacity check raised {exc!r}")
        return None
    placed = {"load": False}
    try:
        proof = _worker_evidence(runtime, ceremony_model, model_request, identity, binding,
                                 frozen_request, frozen_latent, cfg_value, handle, effective,
                                 placed)
    except Exception as exc:
        term = f"the Workers' evidence: {exc!r}"
        if placed["load"]:
            _fall_through_unloaded(handle, term)
        else:
            _declined(term)
        return None
    from .. import residency_mode

    # The ceremony's own reading of a leverless proof (gate_ceremony), so the
    # report's reasons match a ceremony's word for word.
    _conclusive, reasons = residency_mode.leverless_ceremony_verdict(effective, proof.reasons)
    result: dict[str, Any] = {
        "verdict": "INCONCLUSIVE",
        "origin": origin,
        "run_id": run_id,
        "model": model_request["unet_name"],
        "loras": 0,
        "swap_transitions": [row.get("transitions") for row in proof.cycle],
        "proof_kind": "none",
        "inconclusive_reasons": reasons,
        "latents_identical": None,
        "max_abs_latent_diff": None,
        **sampling,
        "wall_s": round(time.perf_counter() - t0, 1),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cross_mode": None,
        gate_inconclusive.BASIS_KEY: gate_inconclusive.BASIS_NO_RENDER,
        "proof_renders": 0,
        "_gate_token": gate_token,
        "_gate_tokens": [gate_token],
        "latent": None,
    }
    kind = gate_inconclusive.classify(result, proof, effective, fsdp_scope=False)
    if kind != gate_inconclusive.KIND_NO_MATERIAL:
        _fall_through_unloaded(handle, "the Workers' swap cycle did not stamp nothing to "
                               "compare: " + ("; ".join(reasons) or "no reason given"))
        return None
    result[gate_inconclusive.RESULT_KEY] = kind
    from ..telemetry import emit

    emit("gate", verdict="INCONCLUSIVE", origin=origin, run_id=run_id,
         model=model.unet_name, loras=0, max_diff=None)
    gate_inconclusive.log_nothing_to_gate(origin, result["model"], basis)
    try:
        from .gate_verdict import append_report_row, record_verdict_rows

        detail = {"model": result["model"], "loras": 0, "origin": origin, "run_id": run_id,
                  "max_abs_latent_diff": None, "quarantine_levers": [],
                  gate_inconclusive.BASIS_KEY: gate_inconclusive.BASIS_NO_RENDER}
        record_verdict_rows(runtime, ledger, key, artifacts, commit, "INCONCLUSIVE", detail,
                            result, capability_context, None, [])
        append_report_row(runtime, result)
    except Exception as exc:
        safe_call(runtime["log"].warning, "could not persist identity-gate verdict: %r", exc)
    return result
