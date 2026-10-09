"""Evidence gathering for one identity-gate ceremony.

Runs the stock-lineage render, the swap or FSDP reload cycle, the swap-lineage
render and the cross-residency reference leg under one auto-gate guard, then
returns every value the verdict half reads as one frozen record.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..refusal import RefusalClass
from ..transfer_utils import raise_with_distinct_cause, reconcile_error, safe_call

if TYPE_CHECKING:  # annotation only; the raise sites resolve it through runtime
    from .gate_fsdp import FsdpGateProofError


@dataclass(frozen=True)
class CeremonyEvidence:
    """What the ceremony body binds and the verdict half goes on to read.

    ``fsdp_scope_state`` is the dict the body used, which is not the caller's
    argument when that argument was None.
    """

    t0: float
    frozen_request: dict
    transaction_versions: Any
    original_worker_args: dict
    ceremony_model: Any
    model_request: dict
    key: str
    artifacts: Any
    stack: list[dict]
    commit: Any
    lazy_swap_applicable: bool
    resolved_topology: Any
    resolved_attention: str
    fsdp_proof_scope: bool
    fsdp_scope_state: dict[str, bool]
    capability_context: Any
    fleet_capability_context: Any
    ledger: Any
    a: dict
    b: dict
    cycle: list
    conclusive: bool
    reasons: list
    slab_proof: Any
    fsdp_reload_proof: bool
    cross_mode: dict | None
    cross_reference_latent: dict | None


def gather_ceremony_evidence(
    *,
    runtime: Mapping[str, Any],
    model: Any,
    request: dict,
    latent: dict,
    cfg_value: float,
    steps_hint: int,
    origin: str,
    run_id: str,
    handle: Any,
    provenance_attestor,
    fsdp_scope_state: dict[str, bool] | None = None,
) -> CeremonyEvidence:
    """Run every ceremony leg and return the evidence the verdict is read from.

    ``runtime`` is the public :mod:`nodes.gate` namespace. Seams are read from it
    at call time, so a monkeypatch there reaches each ``runtime[...]`` call.
    """
    import time as _time

    t0 = _time.perf_counter()
    # The ceremony's own reference renders must never trigger another
    # first-use ceremony. Preserve an outer auto-gate guard when this helper is
    # called from the automatic path.
    from .common import _AUTO_GATE_ACTIVE, _restore_auto_gate_active
    from .gate_identity import bind_request, capture
    from .slab_proof import establish

    gate_was_active = getattr(_AUTO_GATE_ACTIVE, "on", False)
    cross_mode: dict | None = None
    cross_reference_latent: dict | None = None
    ceremony_error: BaseException | None = None
    # Everything a capacity row needs, bound as soon as the ledger exists. The
    # abort arm runs for failures raised before that point too, where none of
    # these names is bound yet.
    capacity_row: tuple[Any, ...] | None = None
    # Records the attempt to append a terminal row, not a confirmed append:
    # both writers swallow a failed append, and a re-attempt from the abort arm
    # would fail the same way while duplicating the row on the retry path.
    terminal_row_written = False
    # Whether the retest transaction was opened. A terminal row supersedes a
    # RETESTING row; with no such row it supersedes whatever the ledger last
    # said, which is the inherited PASS the ceremony has not yet revoked.
    retest_opened = False
    retest_tokens: list = []
    if fsdp_scope_state is None:
        fsdp_scope_state = {}
    fsdp_proof_scope = bool(
        fsdp_scope_state and fsdp_scope_state.get("required")
    )
    try:
        _AUTO_GATE_ACTIVE.on = True
        # Freeze the full mathematical transaction once.  Every render leg
        # receives its own deep copy so a sampler or test double that mutates
        # nested request/latent state cannot change what a later leg proves.
        frozen_request, frozen_latent = runtime["freeze_transaction"]((request, latent))
        transaction_versions = runtime["transaction_tensor_versions"](
            (frozen_request, frozen_latent))

        def transaction_render(render_model, render_request, render_latent, **render_kwargs):
            runtime["require_transaction_unchanged"](transaction_versions)
            rendered = runtime["run_render"](
                render_model, render_request, render_latent, **render_kwargs)
            runtime["require_transaction_unchanged"](transaction_versions)
            return rendered
        original_worker_args = runtime["_effective_worker_args"](model, handle)
        # Every ceremony leg uses this pinned handle and immutable worker-policy
        # snapshot. The mutable Comfy graph object may be quarantined by another
        # request, but it cannot silently replace the risky lineage this gate
        # captured and then earn PASS for different stock-residency renders.
        ceremony_model = runtime["_model_with_worker_overrides"](
            model, original_worker_args, handle=handle)
        model_request, key, ceremony_identity, artifacts, artifact_binding = capture(
            ceremony_model, frozen_request.get("uncond_model"))
        stack = [dict(entry) for entry in model_request.get("loras") or []]
        # Every render, swap, and ledger row shares this transaction snapshot.
        commit = ceremony_identity["comfy"]
        lazy_swap_applicable = (
            bool(stack) and original_worker_args.get("lora_low_rss") is not False)
        from .common import _resolve_topology_for_latent, resolve_sample_attention

        if hasattr(ceremony_model.mesh, "topology_preset"):
            resolved_topology, resolved_sage, _resolved_reason = (
                _resolve_topology_for_latent(
                ceremony_model,
                frozen_latent,
                cfg_value,
                int(getattr(handle, "world", getattr(ceremony_model.mesh, "world", 1))),
            ))
        else:  # lightweight unit-test rigs lack the public MeshSpec fields
            from ..topology import Topology

            resolved_world = int(getattr(handle, "world", 1))
            resolved_topology = Topology(world=resolved_world, dp=resolved_world)
            resolved_sage = False
        resolved_attention = resolve_sample_attention(
            getattr(ceremony_model.mesh, "attention", "unknown"), resolved_sage)
        resolved_fsdp = bool(
            resolved_topology.get("fsdp")
            if isinstance(resolved_topology, dict)
            else getattr(resolved_topology, "fsdp", False)
        )
        fsdp_proof_scope = resolved_fsdp and not stack
        if fsdp_scope_state is not None:
            fsdp_scope_state["required"] = fsdp_proof_scope
        # Snapshot both grants from the same complete effective policy before
        # the ceremony can temporarily change residency or quarantine a lever.
        # The fleet context leaves out the distributed math topology: this
        # ceremony proves the same per-rank storage path Fleet uses.
        capability_context = runtime["gate_capability_context"](
            ceremony_model,
            handle,
            effective_worker_args=original_worker_args,
            resolved_topology=resolved_topology,
            resolved_attention=resolved_attention,
        )
        fleet_capability_context = runtime["fleet_residency_capability_context"](
            ceremony_model, handle, effective_worker_args=original_worker_args)
        # Re-testing is itself a trust transition. Revoke every capability
        # that could inherit an older PASS in one durable row before the first
        # unload/render side effect. A crash or failed final append then leaves
        # a denial, never the superseded PASS. Potential explicit-slab siblings
        # are guarded even when this run cannot prove equivalence; only a
        # proof-qualified final PASS may re-enable them.
        potential_sibling_contexts = runtime["equivalent_slab_mode_contexts"](
            ceremony_model,
            handle,
            None,
            "INCONCLUSIVE",
            None,
            original_worker_args,
            resolved_topology,
            resolved_attention,
        )
        retest_contexts = (
            [capability_context]
            if fsdp_proof_scope
            else [
                capability_context,
                fleet_capability_context,
                *potential_sibling_contexts,
            ]
        )
        ledger = runtime["GateLedger"](runtime["_ledger_dir"]())
        capacity_row = (
            ledger, key, artifacts, commit, capability_context,
            model_request["unet_name"])
        # The retest row and the process publication below are the ceremony's
        # first durable writes. Price the clean reload here, and raise rather
        # than return, so a stop leaves the residency, the fleet and the process
        # verdict maps as it found them. If this moves after either write, a
        # stop leaves a sticky denial for a proof that never ran.
        if fsdp_proof_scope:
            price = runtime["fsdp_reload_price"].price_fsdp_clean_reload(
                handle, model_request)
            if price.applies and not price.fits:
                sentence = runtime["fsdp_reload_price"].record_capacity_stop(
                    *capacity_row, origin=origin, run_id=run_id, price=price)
                terminal_row_written = True
                # No unload, no render, no reload cycle: there is no ceremony
                # residency to clean up, so the abort arm's cleanup does nothing.
                fsdp_scope_state["cleanup_attempted"] = True
                fsdp_scope_state["cleanup_confirmed"] = True
                raise runtime["FsdpGateProofError"](
                    sentence, verdict="INCONCLUSIVE")
        ledger.begin_retest_required(
            key,
            artifacts,
            commit,
            retest_contexts,
            {
                "model": model_request["unet_name"],
                "loras": len(stack),
                "origin": origin,
                "run_id": run_id,
                "phase": "ceremony-preflight",
            },
        )
        retest_opened = True
        retest_tokens = [
            runtime["gate_verdict_token"](key, artifacts.current, commit, context)
            for context in retest_contexts
        ]
        runtime["_publish_process_gate_verdicts"](retest_tokens, "INCONCLUSIVE")
        cycle_token = runtime["render_setup_token"](
            ceremony_model, frozen_latent, cfg_value, handle)
        provenance_attestor("pre", handle, cycle_token)
        # B must come from a clean disk load. Without this unload, auto-gating
        # runs after the user's first render and can use an already-corrupted
        # lazy-swap resident as its "reference", allowing identically wrong
        # B/A renders to pass.
        runtime["require_confirmed_fsdp_unload"](
            handle.call_all("unload", timeout_s=600), handle, fsdp_proof_scope,
            phase="baseline")
        b = transaction_render(
            ceremony_model,
            bind_request(runtime["copy_transaction"](frozen_request), artifact_binding),
            runtime["copy_transaction"](frozen_latent),
                       cfg_value=cfg_value, steps_hint=steps_hint)
        fsdp_reload_proof = False
        if fsdp_proof_scope:
            from .gate_fsdp import establish as establish_fsdp
            from .slab_proof import SlabProof

            # Reprice before reload: the baseline render may have loaded the driver's
            # text encoder since preflight. This is an in-flight stop, so cleanup
            # retires ceremony residency and records the new measurements. See
            # docs/VALIDATION.md, the 2026-08-26 flux2 first-use proof.
            reprice = runtime["fsdp_reload_price"].price_fsdp_clean_reload(
                handle, model_request)
            if reprice.applies and not reprice.fits:
                sentence = runtime["fsdp_reload_price"].record_capacity_stop(
                    *capacity_row, origin=origin, run_id=run_id, price=reprice,
                    stage="reprice")
                terminal_row_written = True
                raise runtime["FsdpGateProofError"](
                    sentence, verdict="INCONCLUSIVE")
            raw_cycle = handle.call_all(
                "gate_fsdp_reload_cycle",
                model_request["unet_name"],
                dict(model_request.get("options") or {}),
                stack,
                ceremony_identity,
                timeout_s=1800,
                **runtime["mesh_setup"].token_kwargs(cycle_token),
            )
            fsdp_proof = establish_fsdp(raw_cycle, handle)
            cycle = fsdp_proof.cycle
            conclusive = fsdp_proof.conclusive
            reasons = fsdp_proof.reasons
            fsdp_reload_proof = conclusive
            slab_proof = SlabProof(
                b=b,
                cycle=[],
                complete=False,
                conclusive=False,
                reasons=[],
                expected=False,
                active=False,
                family=(cycle[0].get("family") if cycle else None),
            )
            if not conclusive:
                raise runtime["FsdpGateProofError"](
                    "FSDP clean-reload proof is incomplete: "
                    + "; ".join(reasons),
                    verdict="INCONCLUSIVE",
                )
        else:
            raw_cycle = handle.call_all(
                "gate_swap_cycle", model_request["unet_name"],
                dict(model_request.get("options") or {}), stack, ceremony_identity,
                timeout_s=1800, **runtime["mesh_setup"].token_kwargs(cycle_token))
            slab_proof = establish(
                ceremony_model, model_request, frozen_request,
                artifact_binding, ceremony_identity, frozen_latent,
                cfg_value, steps_hint, handle, original_worker_args, b, raw_cycle,
                transaction_render)
            b, cycle = slab_proof.b, slab_proof.cycle
            conclusive, reasons = slab_proof.conclusive, slab_proof.reasons
        # A no-LoRA, non-slab ceremony has no ordinary risky lineage to prove
        # except under comfy-managed residency, where the rung itself is
        # the lever under proof and the a/b repeat plus the in-path per-rank
        # identity are its claim (residency_mode.leverless_ceremony_verdict).
        if not stack and not slab_proof.expected and not fsdp_reload_proof:
            from .. import residency_mode

            conclusive, reasons = residency_mode.leverless_ceremony_verdict(
                original_worker_args, reasons)
        a = transaction_render(
            ceremony_model,
            bind_request(runtime["copy_transaction"](frozen_request), artifact_binding),
            runtime["copy_transaction"](frozen_latent),
                       cfg_value=cfg_value, steps_hint=steps_hint)
        # The worker-reported residency decides whether a stock cudaMalloc
        # reference is required; the helper keeps this ceremony's immutable
        # transaction and pinned handle.
        if fsdp_proof_scope:
            cross_mode, cross_reference_latent = None, None
        else:
            cross_mode, cross_reference_latent = runtime["run_cross_residency_reference"](
                runtime=runtime,
                ceremony_model=ceremony_model,
                handle=handle,
                original_worker_args=original_worker_args,
                slab_proof=slab_proof,
                frozen_request=frozen_request,
                frozen_latent=frozen_latent,
                artifact_binding=artifact_binding,
                cfg_value=cfg_value,
                steps_hint=steps_hint,
                stock_latent=b,
                transaction_render=transaction_render,
                bind_request=bind_request,
            )
        if runtime["_effective_worker_args"](model, handle) != original_worker_args:
            raise RuntimeError(
                "identity-gate worker policy changed during the ceremony; "
                "discarding the verdict")
        provenance_attestor("post", handle, cycle_token)
    except BaseException as exc:
        fsdp_refusal: FsdpGateProofError | None = None
        settled_tag = None
        retracted = False
        if fsdp_proof_scope and isinstance(exc, Exception):
            settled_tag = runtime["settled_refusal_tag"](exc)
            if isinstance(exc, runtime["FsdpGateProofError"]):
                fsdp_refusal = exc
            elif (capacity_row is not None
                    and runtime["is_stock_load_capacity_error"](exc)):
                # This arm takes a capacity refusal from any in-flight load:
                # the proof render, the reload cycle, or anything else that
                # reaches a worker load. It runs before the settled arms and
                # converts as the cross-residency leg does, so a tagged class C
                # worker refusal is recorded with cross_mode and the capacity
                # detail that a settled passthrough would drop.
                fsdp_refusal = runtime["FsdpGateProofError"](
                    runtime["fsdp_reload_price"].record_capacity_stop(
                        *capacity_row, origin=origin, run_id=run_id, exc=exc),
                    verdict="INCONCLUSIVE",
                )
                terminal_row_written = True
            elif (settled_tag is not None
                    and settled_tag.refusal_class is RefusalClass.KNOWN_WRONG
                    and settled_tag.waivable):
                # The one settled refusal that may not pass through unchanged;
                # gate_abort.ceremony_waiver_boundary_error says why.
                fsdp_refusal = runtime["ceremony_waiver_boundary_error"](
                    settled_tag, exc)
            elif settled_tag is not None:
                # Already answered. Leaving fsdp_refusal unset lets the bare
                # raise below re-raise it with its class and remedy intact.
                pass
            else:
                fsdp_refusal = runtime["aborted_proof_error"](exc)
        ceremony_error = fsdp_refusal if fsdp_refusal is not None else exc
        if fsdp_proof_scope and isinstance(exc, Exception):
            # FSDP only: capacity_row is bound before the price runs, and an
            # ordinary residency ceremony opens two or more retest contexts
            # that one terminal row cannot supersede; those orphans belong to
            # `dgxm gate --repair`. A BaseException that is not an Exception (a
            # ComfyUI interrupt, KeyboardInterrupt) skips this block, writes no
            # row and keeps its denial, as a killed ceremony does
            # (docs/TROUBLESHOOTING.md #35). A retest that never opened counts
            # as row-already-written: an abort before the transaction opened
            # leaves the ledger alone, and only the preflight capacity stop
            # writes in that window.
            retracted = runtime["close_aborted_proof"](
                runtime, capacity_row=capacity_row, origin=origin,
                run_id=run_id, tag=settled_tag, exc=exc,
                ceremony_error=ceremony_error, retest_tokens=retest_tokens,
                row_written=terminal_row_written or not retest_opened)
        if fsdp_proof_scope and handle is not None:
            runtime["cleanup_aborted_fsdp_proof"](
                handle,
                ceremony_error,
                logger=runtime["log"],
                state=fsdp_scope_state,
            )
        # An identity drift (or any other aborted ceremony) proves nothing.
        # Residency/swap proofs force stock before surfacing the error. FSDP
        # has no same-topology stock fallback, so its typed refusal propagates.
        if fsdp_proof_scope:
            runtime["log_aborted_proof"](
                runtime["log"], settled_tag, retracted, exc)
        else:
            safe_call(runtime["log"].error, "identity gate aborted (%r); forcing stock residency", exc)
        from ..adapters.sol_attention import is_sol_waiver_refusal

        # A ceremony that aborted on the sol-attn accuracy waiver proves
        # nothing about residency either, and quarantining slab and lazy-swap
        # would blame them for a kernel the operator chose. Bounded like the
        # class-U rescue exemption in quarantine_unproven_paths: one named
        # guard, read from the refusal tag, not from the message text.
        sol_waiver_abort = is_sol_waiver_refusal(exc)
        if sol_waiver_abort:
            safe_call(runtime["log"].error,
                      "identity gate aborted on the sol-attn accuracy waiver "
                      "(%r); residency levers unchanged", exc)
        if handle is not None and not fsdp_proof_scope and not sol_waiver_abort:
            try:
                runtime["_force_stock_quarantine"](model, handle, cause=exc)
            except BaseException as cleanup_exc:
                safe_call(runtime["log"].error,
                          "identity-gate stock quarantine failed while preserving "
                          "%r: %r", ceremony_error, cleanup_exc)
                winner, cause = reconcile_error(
                    ceremony_error, cleanup_exc, "identity-gate stock fallback failed")
                ceremony_error = winner
                if winner is not exc:
                    raise_with_distinct_cause(winner, cause)
        if fsdp_refusal is not None and fsdp_refusal is not exc:
            raise fsdp_refusal from exc
        raise
    finally:
        active_cleanup_error = _restore_auto_gate_active(gate_was_active)
        if active_cleanup_error is not None:
            if ceremony_error is not None:
                winner, cause = reconcile_error(
                    ceremony_error, active_cleanup_error,
                    "identity-gate active-context restoration was interrupted")
                if winner is active_cleanup_error:
                    raise_with_distinct_cause(winner, cause)
            else:
                raise active_cleanup_error
    return CeremonyEvidence(
        t0=t0,
        frozen_request=frozen_request,
        transaction_versions=transaction_versions,
        original_worker_args=original_worker_args,
        ceremony_model=ceremony_model,
        model_request=model_request,
        key=key,
        artifacts=artifacts,
        stack=stack,
        commit=commit,
        lazy_swap_applicable=lazy_swap_applicable,
        resolved_topology=resolved_topology,
        resolved_attention=resolved_attention,
        fsdp_proof_scope=fsdp_proof_scope,
        fsdp_scope_state=fsdp_scope_state,
        capability_context=capability_context,
        fleet_capability_context=fleet_capability_context,
        ledger=ledger,
        a=a,
        b=b,
        cycle=cycle,
        conclusive=conclusive,
        reasons=reasons,
        slab_proof=slab_proof,
        fsdp_reload_proof=fsdp_reload_proof,
        cross_mode=cross_mode,
        cross_reference_latent=cross_reference_latent,
    )
