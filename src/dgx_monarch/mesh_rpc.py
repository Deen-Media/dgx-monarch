"""Driver-side RPC dispatch, ambiguity, and artifact-preflight helpers.

:class:`MeshHandle` in ``mesh.py`` keeps the public, monkeypatchable method
seams. This module implements worker RPC dispatch and request artifact
verification, and never imports the handle class, which would make an import
cycle.
"""
from __future__ import annotations

from typing import Any

from . import mesh_safety, mesh_setup
from .log import get_logger
from .transfer_utils import (
    raise_with_distinct_cause,
    reconcile_error,
    safe_note,
)

log = get_logger(__name__)


def call_all(
    handle: Any,
    endpoint_name: str,
    *args: Any,
    timeout_s: float = 900.0,
    setup_token: mesh_setup.SetupToken | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Dispatch an all-rank endpoint under any required session capability."""
    if endpoint_name in mesh_setup.SESSION_SCOPED_ENDPOINTS:
        from .mesh_session import mutation_render_session

        with mutation_render_session(handle):
            return handle._call_all_bound(
                endpoint_name,
                *args,
                timeout_s=timeout_s,
                setup_token=setup_token,
                **kwargs,
            )
    return handle._call_all_bound(
        endpoint_name,
        *args,
        timeout_s=timeout_s,
        setup_token=setup_token,
        **kwargs,
    )


def call_all_bound(
    handle: Any,
    endpoint_name: str,
    *args: Any,
    timeout_s: float = 900.0,
    setup_token: mesh_setup.SetupToken | None = None,
    **kwargs: Any,
) -> list[Any]:
    """Dispatch after any required session capability is active."""
    send_started = False
    mutation_pending = False
    future = None

    def send() -> Any:
        nonlocal mutation_pending, send_started
        # From this publication forward a raised/return-lost dispatch may
        # already be running remotely.
        if endpoint_name in mesh_setup.AMBIGUOUS_MUTATION_ENDPOINTS:
            # Setup-independent cleanup endpoints stay callable without a
            # READY setup, but must never overwrite a prior ambiguous-mutation
            # latch. This runs under handle.lock for every session-scoped
            # mutation, so refusal and publication share one linearization
            # point.
            mesh_setup.require_not_dirty(
                handle, f"dispatch {endpoint_name}")
            # Arm compensation before publishing the pessimistic latch.  An
            # async exception can land after ``__setattr__`` commits the state
            # but before the next instruction; the outer handler must already
            # know that it owns this pre-dispatch latch so it can retire it.
            mutation_pending = True
            handle.setup_cleanup_state = mesh_setup.cleanup_in_progress(
                int(getattr(handle, "setup_generation", 0)),
                f"{endpoint_name} RPC completion", float(timeout_s))
        send_started = True
        return getattr(handle.workers, endpoint_name).call(*args, **kwargs)

    try:
        future = mesh_setup.dispatch_endpoint(
            handle, endpoint_name, send, setup_token=setup_token)
        value_mesh = handle._await_or_evict(future, timeout_s)
    except BaseException as exc:
        # A worker-raised consent offer becomes a panel card here, on the one
        # driver frame that holds every load, gate-leg and sigma refusal.
        # Local import: only this exception branch reaches the observer.
        from .consent_observe import observe_refusal

        try:
            observe_refusal(exc)
        except BaseException as observe_exc:
            safe_note(exc, "worker refusal observation failed", observe_exc)
        ambiguous = (
            endpoint_name in mesh_setup.AMBIGUOUS_MUTATION_ENDPOINTS
            and send_started
            and (
                future is None
                or isinstance(exc, TimeoutError)
                or not isinstance(exc, Exception)
            )
        )
        if ambiguous:
            # Retry the latch once if it raises; ``exc`` stays the error
            # raised. Latching is idempotent and never clears an existing
            # DIRTY state, so the retry is safe.
            for attempt in range(2):
                try:
                    handle._latch_ambiguous_mutation(
                        endpoint_name, timeout_s, exc)
                except BaseException as latch_exc:
                    safe_note(
                        exc, f"ambiguous RPC DIRTY latch attempt {attempt + 1} failed",
                        latch_exc)
                else:
                    break
        elif mutation_pending:
            # Before ``send_started`` even a BaseException cannot have reached
            # the remote endpoint, so clear the latch armed before publication.
            # After a send, an ordinary failure is confirmed and clears it; a
            # timeout or an interruption is ambiguous and stays DIRTY (above).
            for attempt in range(2):
                try:
                    _clear_confirmed_mutation(handle, endpoint_name)
                except BaseException as clear_exc:
                    safe_note(
                        exc, f"confirmed RPC latch clear attempt {attempt + 1} failed",
                        clear_exc)
                else:
                    break
        raise
    if mutation_pending:
        first_clear_error: BaseException | None = None
        clear_cause: BaseException | None = None
        clear_succeeded = False
        for _attempt in range(2):
            try:
                _clear_confirmed_mutation(handle, endpoint_name)
            except BaseException as clear_exc:
                if first_clear_error is None:
                    first_clear_error = clear_exc
                else:
                    first_clear_error, clear_cause = reconcile_error(
                        first_clear_error,
                        clear_exc,
                        "confirmed RPC latch clear retry failed",
                    )
            else:
                clear_succeeded = True
                break
        if (
            first_clear_error is not None
            and not isinstance(first_clear_error, Exception)
        ):
            # Cancellation outranks an earlier ordinary clear failure once
            # the retry has made the completed RPC's latch terminal.
            raise_with_distinct_cause(first_clear_error, clear_cause)
        if not clear_succeeded and first_clear_error is not None:
            # Both attempts failed: raise the first ordinary failure.
            raise_with_distinct_cause(first_clear_error, clear_cause)
    values = [value for _point, value in value_mesh.items()]
    if endpoint_name in ("load_model", "load_uncond_model"):
        from .actor.model_store import request_artifact_identity

        unet_name = args[0] if args else kwargs["unet_name"]
        lora_stack = args[2] if len(args) > 2 else kwargs.get("lora_stack")
        mesh_safety.assert_artifact_parity(
            values, [request_artifact_identity(unet_name, lora_stack)])
    elif endpoint_name == "artifact_identity":
        mesh_safety.assert_artifact_parity(values)
    return values


def latch_ambiguous_mutation(
    handle: Any,
    endpoint_name: str,
    timeout_s: float,
    exc: BaseException,
    *,
    logger: Any = log,
) -> None:
    """Fail closed when a mutation may still be executing remotely."""
    with handle.lock:
        state = handle.setup_cleanup_state
        if (state is None or (
                state.outcome is mesh_setup.SetupCleanupOutcome.IN_PROGRESS
                and state.phase == f"{endpoint_name} RPC completion")):
            handle.setup_cleanup_state = mesh_setup.cleanup_failure(
                int(getattr(handle, "setup_generation", 0)),
                f"{endpoint_name} RPC completion",
                float(timeout_s),
                exc,
            )
    try:
        logger.error(
            "%s RPC completion is ambiguous (%r); the mesh is DIRTY and must "
            "be recycled before any more render or residency work",
            endpoint_name,
            exc,
        )
    except BaseException as log_exc:
        # The state publication is the safety boundary; observability must not
        # replace the original RPC failure after DIRTY is durable.
        safe_note(exc, "ambiguous RPC logging also failed", log_exc)


def _clear_confirmed_mutation(handle: Any, endpoint_name: str) -> None:
    """Clear only this RPC's pessimistic latch after a confirmed outcome."""
    with handle.lock:
        state = handle.setup_cleanup_state
        if (state is not None
                and state.outcome is mesh_setup.SetupCleanupOutcome.IN_PROGRESS
                and state.phase == f"{endpoint_name} RPC completion"):
            handle.setup_cleanup_state = None


def verify_request_artifacts(
    handle: Any,
    request: dict,
    worker_index: int | None = None,
) -> None:
    """Verify request artifacts against the selected READY setup generation."""
    # Cache check before any disk I/O: a marker hit means this exact request
    # already proved parity in this setup generation. Worker sample() still
    # derives fresh local identity and refuses drift after marker publication.
    residency_grant = request.get("_dgxm_fleet_residency_grant")
    normal_grant = request.get("_dgxm_normal_residency_grant")
    artifact_binding = request.get("_dgxm_artifact_binding")
    marker = request.get("_dgxm_artifact_preflight")
    if (residency_grant is None and normal_grant is None
            and artifact_binding is None and marker == {
        "generation": handle.setup_generation,
        "worker_index": worker_index,
    }):
        return
    specs = [request["model"]]
    if request.get("uncond_model"):
        specs.append(request["uncond_model"])
    from .actor.model_store import request_artifact_identity

    expected = [
        request_artifact_identity(spec["unet_name"], spec.get("loras"))
        for spec in specs
    ]
    mesh_safety.assert_request_artifact_binding(request, expected)
    active_worker_args = dict(getattr(handle, "active_worker_args", {}) or {})
    mesh_safety.assert_normal_render_residency_mode(
        request, active_worker_args)
    if request.get("_dgxm_fleet_job") or residency_grant is not None:
        expected_context = {
            "capability": mesh_safety.FLEET_RESIDENCY_CAPABILITY,
            "rank_world": 1,
            "worker_args": active_worker_args,
            **mesh_safety.physical_capability_context(handle),
        }
        # Authorization and parity are separate obligations. Bind the current
        # bytes/combo/policy to the grant, then prove selected-worker parity.
        mesh_safety.assert_fleet_residency_grant(
            request,
            expected,
            effective_worker_args=active_worker_args,
            expected_context=expected_context,
            rank_world=1,
        )
    if normal_grant is not None:
        policy = request.get("_dgxm_normal_policy")
        if not isinstance(policy, dict):
            raise RuntimeError("normal-render request has no pinned policy snapshot")
        setup_key = getattr(handle, "setup_key", None)
        if (not isinstance(setup_key, tuple) or len(setup_key) < 2
                or setup_key[-2] != policy.get("attention")
                or setup_key[-1] != policy.get("sync_ulysses")):
            raise RuntimeError(
                "normal-render Gate policy no longer matches the READY setup")
        physical = mesh_safety.physical_capability_context(handle)
        active_topology = getattr(handle, "topology", None)
        driver_topology = {
            name: getattr(active_topology, name)
            for name in ("ulysses", "ring", "cfg", "dp", "fsdp")
        }
        expected_context = {
            "worker_args": active_worker_args,
            "mesh_mode": physical["mesh_mode"],
            "config_source": physical["config_source"],
            "config_fingerprint": physical["config_fingerprint"],
            "world": physical["physical_world"],
            "hosts": physical["hosts"],
            "gpus_per_host": physical["gpus_per_host"],
            "topology_preset": policy.get("topology_preset"),
            "attention": policy.get("attention"),
            "sync_ulysses": policy.get("sync_ulysses"),
            "resolved_topology": driver_topology,
            "resolved_attention": request.get("sage_kernel"),
        }
        from .fsdp_proof_scope import topology_requires_fsdp_proof

        if topology_requires_fsdp_proof(
            driver_topology,
            request["model"].get("loras"),
        ):
            from .fsdp_proof_scope import FSDP_PROOF_SCOPE

            expected_context["proof_scope"] = FSDP_PROOF_SCOPE
        mesh_safety.assert_normal_render_residency_grant(
            request,
            expected,
            effective_worker_args=active_worker_args,
            expected_context=expected_context,
            rank_world=int(getattr(handle, "world", 0) or 0),
            setup_generation=int(getattr(handle, "setup_generation", -1)),
            setup_key=tuple(setup_key),
            worker_args_key=tuple(getattr(handle, "worker_args_key", ()) or ()),
            worker_topology=driver_topology,
        )
    if worker_index is None:
        # Collective render: every participant proves parity before collectives.
        results = handle.call_all("artifact_identity", specs, timeout_s=120)
    else:
        # Fleet WORLD=1 jobs verify only the actor that executes this request.
        coords = {"gpus": worker_index % handle.gpus_per_host}
        if "hosts" in list(handle.workers.extent.labels):
            coords["hosts"] = worker_index // handle.gpus_per_host
        worker = handle.workers.slice(**coords)
        result = handle._await_or_evict(
            worker.artifact_identity.call_one(specs), timeout_s=120)
        results = [result]
    mesh_safety.assert_artifact_parity(results, expected)
    request["_dgxm_artifact_sets"] = expected
    request["_dgxm_artifact_preflight"] = {
        "generation": handle.setup_generation,
        "worker_index": worker_index,
    }
