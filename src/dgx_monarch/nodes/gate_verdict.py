"""Verdict derivation and publication for one identity-gate ceremony.

Reads the record the ceremony body produced, decides PASS/FAIL/INCONCLUSIVE,
pushes the levers a FAIL implicates off for that combination, and writes the
process tokens, the ledger rows and the gate report row.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

from ..transfer_utils import safe_call
from .gate_ceremony import CeremonyEvidence
from .gate_proof_request import report_fields

# What a rank's swap cycle reports with no LoRA stack (actor/gate_cycle.py).
NO_STACK_CYCLE_REASON = "no lora stack: lazy swap is not applicable"


def publish_ceremony_verdict(
    *,
    runtime: Mapping[str, Any],
    evidence: CeremonyEvidence,
    model: Any,
    handle: Any,
    origin: str,
    run_id: str,
) -> dict:
    """Derive the verdict from the gathered evidence and publish it.

    ``runtime`` is the public :mod:`nodes.gate` namespace, read at call time so
    a monkeypatch of that module reaches every seam.
    """
    import time as _time

    t0 = evidence.t0
    frozen_request = evidence.frozen_request
    transaction_versions = evidence.transaction_versions
    original_worker_args = evidence.original_worker_args
    ceremony_model = evidence.ceremony_model
    model_request = evidence.model_request
    key = evidence.key
    artifacts = evidence.artifacts
    stack = evidence.stack
    commit = evidence.commit
    lazy_swap_applicable = evidence.lazy_swap_applicable
    resolved_topology = evidence.resolved_topology
    resolved_attention = evidence.resolved_attention
    fsdp_proof_scope = evidence.fsdp_proof_scope
    fsdp_scope_state = evidence.fsdp_scope_state
    capability_context = evidence.capability_context
    fleet_capability_context = evidence.fleet_capability_context
    ledger = evidence.ledger
    a = evidence.a
    b = evidence.b
    cycle = evidence.cycle
    conclusive = evidence.conclusive
    reasons = evidence.reasons
    slab_proof = evidence.slab_proof
    fsdp_reload_proof = evidence.fsdp_reload_proof
    cross_mode = evidence.cross_mode
    cross_reference_latent = evidence.cross_reference_latent

    a.pop("_dgxm_denoised", None)
    b.pop("_dgxm_denoised", None)
    if cross_reference_latent is not None:
        cross_reference_latent.pop("_dgxm_denoised", None)
    # A renderer can retain the shared frozen tensor and mutate it through a
    # version-bypassing view after transaction_render's post-call check. Recheck
    # immediately before deriving anything that can publish a Gate grant.
    runtime["require_transaction_unchanged"](transaction_versions)
    identical, max_diff = runtime["compare_latents"](a["samples"], b["samples"])
    verdict = "INCONCLUSIVE" if not conclusive else ("PASS" if identical else "FAIL")
    if cross_mode is not None:
        # A proven slab-vs-stock divergence outranks both a PASS and an
        # INCONCLUSIVE swap cycle: the cross leg compares two stock-lineage
        # renders and is valid regardless of whether the swaps took the lazy
        # path. An errored cross leg leaves slab exactness unproven rather
        # than disproven, so a would-be PASS degrades to INCONCLUSIVE.
        if cross_mode["verdict"] == "FAIL":
            verdict = "FAIL"
        elif cross_mode["verdict"] == "PASS" and not lazy_swap_applicable:
            # A no-swap cross PASS still needs its same-residency repeat to agree.
            verdict = "PASS" if identical else "INCONCLUSIVE"
            reasons = ([] if identical else [*reasons,
                "same-residency repeat diverged despite a passing cross-residency comparison"])
        elif cross_mode["verdict"] in ("ERROR", "CAPACITY") and (
                verdict == "PASS" or (verdict == "INCONCLUSIVE" and not lazy_swap_applicable)):
            # An unavailable or incomplete stock reference can only ever be
            # INCONCLUSIVE. CAPACITY explains why; it grants nothing. With no
            # lazy swap to prove, the cross leg is the whole proof, so its
            # failure is the reason, not the swap cycle's "not applicable".
            verdict = "INCONCLUSIVE"
            if not slab_proof.error:
                if not lazy_swap_applicable:
                    reasons = [reason for reason in reasons
                               if reason != NO_STACK_CYCLE_REASON]
                reasons = [*reasons,
                           "the stock reference cannot load here (capacity refusal), "
                           "so the slab render has nothing to be compared with"
                           if cross_mode["verdict"] == "CAPACITY"
                           else "cross-residency reference leg failed"]

    result = {
        "verdict": verdict,
        "origin": origin,
        "run_id": run_id,
        "model": model_request["unet_name"],
        "loras": len(stack),
        "swap_transitions": [r.get("transitions") for r in cycle],
        "proof_kind": (
            runtime["FSDP_PROOF_KIND"]
            if fsdp_reload_proof
            else "slab_cross_residency"
            if cross_mode is not None and not lazy_swap_applicable
            else "lora_low_rss_swap"
            if lazy_swap_applicable
            else "none"
        ),
        "inconclusive_reasons": reasons,
        "latents_identical": identical,
        "max_abs_latent_diff": max_diff,
        **report_fields(frozen_request),
        "wall_s": round(_time.perf_counter() - t0, 1),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cross_mode": cross_mode,
        "_gate_token": runtime["gate_verdict_token"](
            key, artifacts.current, commit, capability_context),
        # A slab-vs-stock FAIL must return the proven stock-residency leg, not
        # the slab output the ceremony just rejected.
        "latent": (cross_reference_latent
                   if cross_mode is not None and cross_mode.get("verdict") == "FAIL"
                   and cross_reference_latent is not None else b),
    }
    if fsdp_proof_scope and verdict == "FAIL":
        result["fsdp_cleanup_confirmed"] = runtime["cleanup_aborted_fsdp_proof"](
            handle, runtime["FsdpGateProofError"]("terminal FSDP Gate FAIL", verdict="FAIL"),
            logger=runtime["log"], state=fsdp_scope_state)
    from ..telemetry import emit

    emit("gate", verdict=verdict, origin=origin, run_id=run_id,
         model=model.unet_name, loras=len(stack), max_diff=max_diff)
    quarantine_levers: list[str] = []
    if verdict == "PASS":
        if fsdp_reload_proof:
            runtime["log"].info(
                "identity gate PASS (%s): independently loaded FSDP shards render "
                "bit-identically (%s)",
                origin,
                model.unet_name,
            )
        else:
            runtime["log"].info("identity gate PASS (%s): its proof renders are bit-identical "
                     "(%s, %d loras)", origin, model.unet_name, len(stack))
    elif verdict == "INCONCLUSIVE":
        runtime["_gate_identity"].record_inconclusive(
            result, slab_proof, original_worker_args, fsdp_scope=fsdp_proof_scope)
    elif fsdp_reload_proof:
        # The FSDP proof runs with no LoRA stack, and FSDP turns slab residency
        # off, so neither risky lever was in play. A deterministic-reload
        # mismatch denies this exact FSDP capability but must not be mislabeled
        # as a LoRA/slab finding or issue a vacuous quarantine RPC.
        runtime["log"].error(
            "identity gate FAIL (%s): independently loaded FSDP shards diverge "
            "(max latent diff %s) for %s; no FSDP capability was granted.",
            origin,
            max_diff,
            model.unet_name,
        )
    else:
        # Two independent findings pick the lever(s): a cross-mode FAIL proves
        # slab residency unsafe; an in-mode divergence (conclusive swaps whose
        # lineages differ) proves the lazy path unsafe. Either, or both, can
        # hold, including when the swap cycle was INCONCLUSIVE and only the
        # cross leg failed. The quarantine belongs to this combination in this
        # capability context, so it never touches the graph's policy dict,
        # which other checkpoints on the same Init node share. The push below
        # takes its policy from a per-dispatch clone; each later dispatch of
        # this combination re-applies it from the process verdict cache and
        # the ledger (nodes/render_quarantine).
        cross_failed = cross_mode is not None and cross_mode.get("verdict") == "FAIL"
        in_mode_diverged = conclusive and not identical
        quarantined_args: dict[str, bool] = {}
        levers = quarantine_levers
        if cross_failed:
            quarantined_args["slab_weights"] = False
            levers.append("slab_weights")
            runtime["log"].error("identity gate FAIL (%s): slab-residency render diverges from "
                      "stock residency (max latent diff %s) for %s.",
                      origin, cross_mode.get("max_abs_latent_diff") if cross_mode
                      else None, model.unet_name)
        if in_mode_diverged or not cross_failed:
            quarantined_args["lora_low_rss"] = False
            levers.append("lora_low_rss")
            runtime["log"].error("identity gate FAIL (%s): its proof renders diverge from each other "
                      "(max latent diff %s) for %s.",
                      origin, max_diff, model.unet_name)
        from .. import residency_mode

        # The next dispatch of this combination turns comfy_managed off for
        # the whole Init node (nodes/render_quarantine), so say so here too.
        managed = residency_mode.WORKER_ARG in (getattr(model.mesh, "worker_args", None) or {})
        runtime["log"].error("QUARANTINING: %s switched off for %s in this capability context; other "
                  "checkpoints on this Init node keep their lora_low_rss and slab_weights.%s The gate "
                  "ledger keeps the FAIL for later sessions. Renders of this combination use comfy's stock "
                  "path where one exists; a live comfy_managed worker refuses them until a mesh reset. Report "
                  "this with your comfy commit and dgxm status.", " + ".join(levers),
                  f"{model.unet_name} (combination {key})",  # a dual proof's own key
                  " The next dispatch of this combination turns comfy_managed off for the whole Init node."
                  if managed else "")
        quarantined = runtime["_model_with_worker_overrides"](model, quarantined_args, handle=handle)
        try:
            runtime["_apply_worker_policy"](
                handle, runtime["_effective_worker_args"](quarantined, handle), timeout_s=600)
        except Exception as exc:
            safe_call(runtime["log"].error, "quarantine push failed (%r); the next dispatch of this "
                      "combination sends %s=off again before it renders", exc, " + ".join(levers))
        emit("quarantine",
             model=model.unet_name,
             lever=runtime["_gate_identity"].revoke_consents_after_gate_fail(model, levers))
    sibling_contexts = (
        []
        if fsdp_proof_scope
        else runtime["equivalent_slab_mode_contexts"](
            ceremony_model,
            handle,
            slab_proof,
            verdict,
            cross_mode,
            original_worker_args,
            resolved_topology,
            resolved_attention,
        )
    )
    verdict_contexts = (
        [capability_context]
        if fsdp_proof_scope
        else [capability_context, fleet_capability_context, *sibling_contexts]
    )
    process_tokens = [
        runtime["gate_verdict_token"](key, artifacts.current, commit, process_context)
        for process_context in verdict_contexts
    ]
    result["_gate_tokens"] = process_tokens
    if origin != "auto_first_use":
        # Explicit ceremonies publish their newest normal/Fleet verdict as one
        # process-local transaction before ledger I/O. Auto publication waits
        # for auto_gate.maybe_auto_gate's post-ceremony token-drift check.
        runtime["_publish_process_gate_verdicts"](process_tokens, verdict, result)
    try:
        detail = {"model": model_request["unet_name"], "loras": len(stack), "origin": origin,
                  "run_id": run_id, "max_abs_latent_diff": max_diff,
                  "quarantine_levers": quarantine_levers,
                  **({"cross_mode": cross_mode["verdict"]} if cross_mode else {}),
                  **({"capacity_detail": cross_mode.get("capacity_detail")}
                     if cross_mode and cross_mode["verdict"] == "CAPACITY" else {})}
        record_verdict_rows(
            runtime, ledger, key, artifacts, commit, verdict, detail, result,
            capability_context,
            None if fsdp_proof_scope else fleet_capability_context, sibling_contexts)
        runtime["record_ceremony_certificate"](
            ledger, key, artifacts, commit, cross_mode, capability_context, detail)
        if verdict == "INCONCLUSIVE" and not fsdp_proof_scope:
            # After the rows: the consent read must see this verdict, not the
            # RETESTING row it supersedes.
            runtime["_gate_inconclusive"].settle_stock_reference_unavailable(
                model, result, detail)
        append_report_row(runtime, result)
    except Exception as exc:
        safe_call(runtime["log"].warning, "could not persist identity-gate verdict: %r", exc)
    return result


def record_verdict_rows(runtime: Mapping[str, Any], ledger: Any, key: str, artifacts: Any,
                        commit: Any, verdict: str, detail: dict, result: Mapping[str, Any],
                        capability_context: Any, fleet_capability_context: Any | None,
                        sibling_contexts: list) -> None:
    """The ledger rows one verdict writes: the context it exercised, the Fleet
    context (None under an FSDP proof scope, which has none), and every
    explicit-on slab sibling. Shared with nodes/gate_nothing_to_gate.py, which
    passes neither, so its no-render verdict writes the exercised row this way
    and revokes nothing it did not exercise."""
    classified = runtime["_gate_inconclusive"]
    ledger.record(key, artifacts, commit, verdict,
                  classified.ledger_detail(detail, result), capability_context)
    revocation = classified.revocation_detail(detail, result)
    if fleet_capability_context is not None:
        ledger.record(key, artifacts, commit, verdict, revocation, fleet_capability_context)
    for sibling in sibling_contexts:
        ledger.record(key, artifacts, commit, verdict,
                      {**revocation, "stamped": "slab-mode-equivalence"}, sibling)


def append_report_row(runtime: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    """Append one gate report row: the result without its latent or private keys."""
    report_row = {k: v for k, v in result.items() if k != "latent" and not k.startswith("_")}
    with open(os.path.join(runtime["_ledger_dir"](), "dgxm_gate_reports.jsonl"), "a") as f:
        f.write(json.dumps(report_row) + "\n")
