"""Cross-residency reference leg for the identity gate.

The stock reference loads the checkpoint the way stock residency does, with
the host copy it is placed from, so it can need far more room than the slab
legs before it. The driver's own models count against that price on the head,
the Flux 2 text encoder alone 33 GiB, and ComfyUI reloads every one of them on
demand; the ceremony's conditioning is already encoded. So when the reference
leg meets a capacity refusal, typed or a Worker's own refusal wrapped as
remote text, the driver frees what it can get back
(``driver_unload.check_with_unload``: ComfyUI's models on this host, then the
actors' allocator caches) and tries the leg once more. Only the last answer
counts. Measured 2026-11-04: Flux 2 fp8mixed on a fresh pair priced 74.3 GiB
against 63.0 GiB on the head with the encoder loaded, a box that read 97.5
GiB before the graph ran.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .. import driver_unload, first_render
from ..fsdp_reload_price import capacity_classification
from ..gate_audit import fold_rank_certificates, parse_measured

# Re-exported for nodes/gate.py: the verdict half's persist block reads the
# certificate recorder from that namespace.
from ..gate_audit import record_ceremony_certificate as record_ceremony_certificate
from ..transfer_utils import failure_summary, safe_call
from . import pending_load_relief
from .latent_identity import compare_latents


def run_cross_residency_reference(
    *,
    runtime: Mapping[str, Any],
    ceremony_model: Any,
    handle: Any,
    original_worker_args: dict,
    slab_proof: Any,
    frozen_request: dict,
    frozen_latent: dict,
    artifact_binding: dict,
    cfg_value: float,
    steps_hint: int,
    stock_latent: dict,
    transaction_render: Callable[..., dict],
    bind_request: Callable[[dict, dict], dict],
) -> tuple[dict | None, dict | None]:
    """Render under stock residency and compare with the in-mode stock leg."""
    if slab_proof.error:
        return {"verdict": "ERROR", "detail": slab_proof.error}, None
    if not slab_proof.expected:
        return None, None

    # Only now is the leg certain to run, so only now is its notice sent: it
    # names this proof render for the panel, the ticker and `dgxm top`'s run
    # block, which would otherwise read it as one more "render".
    first_render.cross_residency_check()
    cross_reference_latent = None
    try:
        with runtime["_temporary_worker_policy"](
            handle, original_worker_args, {"slab_weights": False},
        ) as applied:
            # Every worker must leave slab residency, or this would be a
            # vacuous slab-vs-slab comparison.
            still_slab = [row for row in applied if row and row.get("slab_weights")]
            if still_slab:
                raise RuntimeError(
                    f"{len(still_slab)} worker(s) did not leave slab residency "
                    "for the cross-mode reference")
            handle.call_all("unload", timeout_s=600)
            pending_load_relief.note_unloaded()  # the stock reference's load is still to come
            stock_model = runtime["_model_with_worker_overrides"](
                ceremony_model, {"slab_weights": False}, handle=handle)
            rendered: list[dict] = []

            def stock_reference() -> None:
                # A capacity refusal loads nothing on any rank, so the leg can
                # run again from the same frozen transaction.
                rendered[:] = [transaction_render(
                    stock_model,
                    bind_request(
                        runtime["copy_transaction"](frozen_request),
                        artifact_binding,
                    ),
                    runtime["copy_transaction"](frozen_latent),
                    cfg_value=cfg_value,
                    steps_hint=steps_hint,
                )]

            # On one Spark, and after an admitted quote on two, the Worker's own
            # ladder refuses inside the actor and the refusal arrives as remote
            # text, so the leg counts it by the classifier the except below uses.
            driver_unload.check_with_unload(
                [stock_reference], handle=handle,
                subject="the identity gate's stock reference render",
                is_refusal=lambda exc: (driver_unload.is_capacity_refusal(exc)
                                        or runtime["is_stock_load_capacity_error"](exc)))
            candidate = rendered[0]
            cross_reference_latent = candidate
        samples = candidate["samples"]
    except Exception as exc:
        if runtime["is_artifact_binding_error"](exc):
            raise
        if runtime["is_stock_load_capacity_error"](exc):
            detail = failure_summary(exc)[:200]
            try:
                measured = parse_measured(exc)
            except BaseException:
                measured = None
            safe_call(
                runtime["log"].warning,
                "cross-residency reference cannot load under stock residency "
                "(%s), so this hardware cannot prove slab exactness", detail)
            # The slab leg still ran. Carry its per-rank byte-verify
            # certificates and the refusal's measured numbers so the ceremony
            # can record what was proven (these bytes are the checkpoint's)
            # beside what was not (that the render matches stock). Both fold to
            # None on incomplete evidence, and then no certificate row is written.
            # ``capacity_detail`` names the refusal's class and guard in the
            # FSDP capacity stop's bounded vocabulary, never its text: the
            # consent reader keeps slab only behind a typed class C refusal.
            return {"verdict": "CAPACITY", "detail": detail,
                    "capacity_detail": capacity_classification(exc),
                    "certificate": fold_rank_certificates(
                        getattr(slab_proof, "cycle", None), getattr(handle, "world", None)),
                    "measured": measured}, None
        try:
            memory_exhausted = runtime["is_memory_exhaustion"](exc)
        except BaseException:
            memory_exhausted = False
        hint = (" (memory ran out during this leg; see docs/TROUBLESHOOTING.md)"
                if memory_exhausted else "")
        detail = failure_summary(exc)[:200]
        safe_call(
            runtime["log"].warning,
            "cross-residency reference leg failed (%s); slab exactness stays "
            "unproven%s", detail, hint)
        return {"verdict": "ERROR", "detail": detail}, None

    same, max_diff = compare_latents(samples, stock_latent["samples"])
    return {
        "verdict": "PASS" if same else "FAIL",
        "max_abs_latent_diff": max_diff,
    }, cross_reference_latent
