"""Verdict derivation and publication for one identity-gate ceremony.

Reads the record the ceremony body produced, decides PASS/FAIL/INCONCLUSIVE,
quarantines the levers a FAIL implicates, and writes the process tokens, the
ledger rows and the gate report row.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from typing import Any

from ..transfer_utils import safe_call
from .gate_ceremony import CeremonyEvidence


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
        elif cross_mode["verdict"] in ("ERROR", "CAPACITY") and verdict == "PASS":
            # An unavailable or incomplete stock reference can only ever be
            # INCONCLUSIVE. CAPACITY explains why; it grants nothing.
            verdict = "INCONCLUSIVE"
            if not slab_proof.error:
                reasons = [*reasons,
                           "stock reference cannot load here (capacity refusal)"
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
        "seed": frozen_request.get("noise_seed"),
        "steps": frozen_request.get("steps"),
        "cfg": frozen_request.get("cfg"),
        "sampler": frozen_request.get("sampler_name"),
        "scheduler": frozen_request.get("scheduler"),
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
        # cross leg failed. The local spec mutates before the RPC so an
        # unreachable worker cannot leave the session configured for a path
        # this exact artifact/commit has proven unsafe.
        cross_failed = cross_mode is not None and cross_mode.get("verdict") == "FAIL"
        in_mode_diverged = conclusive and not identical
        quarantined_args = dict(model.mesh.worker_args)
        levers = quarantine_levers
        if cross_failed:
            quarantined_args["slab_weights"] = False
            model.mesh.worker_args["slab_weights"] = False
            levers.append("slab_weights")
            runtime["log"].error("identity gate FAIL (%s): slab-residency render diverges from "
                      "stock residency (max latent diff %s) for %s.",
                      origin, cross_mode.get("max_abs_latent_diff") if cross_mode
                      else None, model.unet_name)
        if in_mode_diverged or not cross_failed:
            quarantined_args["lora_low_rss"] = False
            model.mesh.worker_args["lora_low_rss"] = False
            levers.append("lora_low_rss")
            runtime["log"].error("identity gate FAIL (%s): its proof renders diverge from each other "
                      "(max latent diff %s) for %s.",
                      origin, max_diff, model.unet_name)
        runtime["log"].error("QUARANTINING: %s switched off for this graph; the gate ledger keeps the FAIL for later "
                  "sessions. Renders use comfy's stock path where one exists; a live comfy_managed worker refuses "
                  "them until a mesh reset. Report this with your comfy commit and dgxm status.", " + ".join(levers))
        try:
            runtime["_apply_worker_policy"](
                handle, runtime["_effective_worker_args"](model, handle), timeout_s=600)
        except Exception as exc:
            safe_call(runtime["log"].error, "quarantine push failed (%r); set %s=off on the Init "
                      "node manually", exc, " + ".join(levers))
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
                  **({"cross_mode": cross_mode["verdict"]} if cross_mode else {})}
        ledger.record(key, artifacts, commit, verdict,
                      runtime["_gate_inconclusive"].ledger_detail(detail, result), capability_context)
        if not fsdp_proof_scope:
            ledger.record(
                key, artifacts, commit, verdict, detail, fleet_capability_context)
        for sibling in sibling_contexts:
            ledger.record(key, artifacts, commit, verdict,
                          {**detail, "stamped": "slab-mode-equivalence"}, sibling)
        runtime["record_ceremony_certificate"](
            ledger, key, artifacts, commit, cross_mode, capability_context, detail)
        report_row = {k: v for k, v in result.items()
                      if k != "latent" and not k.startswith("_")}
        with open(os.path.join(runtime["_ledger_dir"](), "dgxm_gate_reports.jsonl"), "a") as f:
            f.write(json.dumps(report_row) + "\n")
    except Exception as exc:
        safe_call(runtime["log"].warning, "could not persist identity-gate verdict: %r", exc)
    return result
