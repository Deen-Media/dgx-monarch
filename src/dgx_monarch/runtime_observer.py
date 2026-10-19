"""Small safe hooks at existing render evidence boundaries.

These observations never authorize sampling, teardown, ACK or retry. Worker
reply validation and resource retirement remain with their original owners.
"""
from __future__ import annotations

import uuid
from typing import Any

from . import execution_events as events


def prepare(request: dict, fleet_index: int | None = None) -> events.Scope | None:
    scope = events.capture(request.get("_dgxm_render_id", ""), fleet_index)
    if scope is not None:
        request["_dgxm_observe_runtime"] = True
    return scope


def dispatch(request: dict, handle: Any, topo: Any, scope: events.Scope | None) -> events.Scope | None:
    if scope is None:
        return None
    try:
        topology = {key: getattr(topo, key) for key in ("world", "dp", "ulysses", "ring", "cfg", "fsdp")}
        reports = getattr(handle, "worker_capabilities", [])
        identities = [{key: row[key] for key in ("rank", "host", "world", "topology", "sol_attn", "source_manifest_sha256", "torch_version", "comfy_commit") if key in row}
                      for row in reports if isinstance(row, dict)]
        events.emit(scope, "dispatch_submitted", effective_topology=topology, attention=request.get("sage_kernel"), attention_source="driver_selection",
                    setup_generation=getattr(handle, "setup_generation", None),
                    config_fingerprint=getattr(handle, "config_fingerprint", None),
                    model_fingerprints=request.get("_dgxm_artifact_sets"), worker_runtime=identities,
                    residency_mode=request.get("_dgxm_normal_residency_mode"),
                    admission="driver_guards_and_dispatch_returned", worker_acceptance="not_yet_observed")
    except Exception:
        pass
    return scope


def submission_failed(guard: Any, error: BaseException) -> None:
    if not events.journal.active():
        return
    scope = events.capture(getattr(guard, "render_id", None) or uuid.uuid4().hex)
    events.emit(scope, "submission_failed", error_type=type(error).__name__,
                dispatch_started=bool(getattr(guard, "dispatch_started", False)), outcome="not_established")


def _workers(results: list[dict]) -> list[dict]:
    return [{key: row[key] for key in ("rank", "dp_rank", "host", "sample_s", "gpu_load_s", "transition",
                                       "latent_stats", "runtime_identity") if key in row}
            for row in results if isinstance(row, dict)]


def finish(scope: events.Scope | None, callback: Any, *args: Any, **kwargs: Any) -> Any:
    with events.activate(scope):
        result = callback(*args, **kwargs)
    # Callback has validated cross-rank signatures and materialized outputs.
    # A received result is not a native ownership ACK.
    if scope is not None:
        events.emit(scope, "worker_results", workers=_workers(args[0]), authority="validated_render_results")
        events.emit(scope, "result_validated", outcome="validated_output", ownership="retirement_pending")
    return result


def completed(scope: events.Scope | None) -> None:
    events.emit(scope, "render_completed", outcome="validated_output_and_retired_lease")


def failed(scope: events.Scope | None, error: BaseException, phase: str, refused: bool) -> None:
    events.emit(scope, "render_failed", phase=phase, error_type=type(error).__name__,
                outcome="refused" if refused else "unknown", worker_refusal_verified=refused)


def fleet_read(item: Any, callback: Any, *args: Any) -> Any:
    with events.activate(getattr(item, "_execution_scope", None)):
        return callback(*args)


def fleet_result(item: Any, result: dict) -> None:
    scope = getattr(item, "_execution_scope", None)
    if scope is not None:
        item._execution_output_validated = True
        events.emit(scope, "worker_results", workers=_workers([result]), authority="validated_fleet_output")
        events.emit(scope, "result_validated", outcome="validated_output", ownership="retirement_pending")


def fleet_failed(item: Any, error: BaseException) -> None:
    events.emit(getattr(item, "_execution_scope", None), "fleet_item_failed", error_type=type(error).__name__,
                outcome="refused" if item.refused else "unknown", worker_refusal_verified=bool(item.refused))


def fleet_settled(item: Any) -> None:
    if (getattr(item, "_execution_output_validated", False) and not getattr(item, "_execution_completed", False)
            and item.consumed and item.retired and item.audit_retired and item.progress_closed):
        item._execution_completed = True
        events.emit(getattr(item, "_execution_scope", None), "fleet_item_completed", outcome="validated_output_and_retired_lease")


def worker_identity(request: dict, worker: Any, result: dict) -> None:
    """Attach request-bound facts only for an explicit observer dispatch."""
    if request.get("_dgxm_observe_runtime") is not True:
        return
    try:
        from .runtime_identity import runtime_versions
        from .runtime_provenance import cached_dgx_source_manifest_sha256

        result["runtime_identity"] = {
            "source": "worker_result", "render_id": request.get("_dgxm_render_id"),
            "rank": worker.rank, "setup_generation": worker._setup_generation,
            "topology": dict(worker.topology), "versions": runtime_versions(),
            "selected_attention": getattr(getattr(worker, "_attn", None), "kernel", None),
            "effective_attention": getattr(getattr(worker, "_attn", None), "effective_kernel", None),
            "source_manifest_sha256": cached_dgx_source_manifest_sha256(),
        }
    except Exception:
        result["runtime_identity"] = {"source": "worker_result", "state": "unavailable"}
