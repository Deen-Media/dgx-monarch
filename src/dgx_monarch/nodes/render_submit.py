"""The eager half of a render: assemble the request, then dispatch it.

One submission from the caller's model and latent to a published
``PendingRender``: adoption-context claim, packed-model binding, topology
resolution, the persisted quarantine, normal-render authorization, setup, the
wire request, progress, and the sample dispatch itself, all under the one
interruption-safe ownership guard in ``nodes/submit_guard``.

Every name this path needs from ``nodes/common`` is imported per call, not at
module scope. That keeps the module graph acyclic, and it keeps one seam per
name: the binder already defaults its liveness check to ``common.ensure_live``,
so resolving the direct calls anywhere else would let one render see two
different functions.
"""
from __future__ import annotations

import os
import uuid
from functools import partial
from typing import TYPE_CHECKING, cast

from .. import mesh_setup
from ..adoption_evidence import (
    CONTEXT_REQUEST_KEY as _ADOPTION_CONTEXT_REQUEST_KEY,
)
from ..adoption_evidence import (
    claim_active_context as _claim_adoption_context,
)
from ..adoption_evidence import (
    consume_context_claim as _consume_adoption_context_claim,
)
from ..adoption_evidence import (
    context_sha256 as _adoption_context_sha256,
)
from ..adoption_evidence import (
    render_id_sha256 as _adoption_render_id_sha256,
)
from ..adoption_evidence import (
    request_sha256 as _adoption_request_sha256,
)
from ..log import get_logger
from ..mesh import MeshHandle, mark_defunct_on_supervision_failure
from ..progress import ProgressReceiver
from ..transfer import pack_latent
from . import consent_waiver, render_preflight
from .gate_identity import (
    authorize_normal_render,
    model_for_request,
    model_with_worker_overrides,
)
from .pending import PendingRender, PendingRenderHandoff
from .render_quarantine import _enforce_persisted_quarantine
from .render_result import _finish_render
from .render_session import RetiredRenderHandleError, claim_render_session
from .render_validation import validate_render_topology
from .submit_guard import SubmitRenderGuard as _SubmitRenderGuard
from .submit_guard import run_guarded_submit

if TYPE_CHECKING:
    from .common import ModelSpec

log = get_logger(__name__)

_SAMPLE_TIMEOUT_S = float(os.environ.get("DGXM_SAMPLE_TIMEOUT", "3600"))
_latent_without_topology_metadata = render_preflight.latent_without_topology_metadata
_UNCLAIMED_ADOPTION_CONTEXT = object()


def submit_render(model: ModelSpec, request: dict, latent: dict, cfg_value: float | None,
                  steps_hint: int, seq: int = 0, depth: int = 1, progress_on_step=None,
                  handoff: PendingRenderHandoff | None = None, *,
                  _claimed_adoption_context: object = (
                      _UNCLAIMED_ADOPTION_CONTEXT)) -> PendingRender:
    """Eager half of a render under one interruption-safe ownership guard."""
    # Claims are one-shot: consume once, up front, so a healed retry reuses the
    # same evidence. A second claim of the active scope or a second consume of
    # a caller's claim raises ResidentAdoptionEvidenceError.
    adoption_context = _consume_adoption_context_claim(
        _claim_adoption_context()
        if _claimed_adoption_context is _UNCLAIMED_ADOPTION_CONTEXT
        else _claimed_adoption_context)

    def attempt() -> PendingRender:
        guard = _SubmitRenderGuard()

        def body() -> PendingRender:
            return _submit_render_guarded(
                model, request, latent, cfg_value, steps_hint, seq, depth,
                progress_on_step, handoff, guard,
                adoption_context=adoption_context)

        def on_failure(exc: BaseException) -> None:
            if (guard.dispatch_started and guard.handle is not None
                    and isinstance(exc, Exception)):
                mark_defunct_on_supervision_failure(guard.handle, exc)

        return run_guarded_submit(guard, body, on_failure)

    try:
        return attempt()
    except mesh_setup.LifecycleBusyError as exc:
        if not _heal_abandoned_wedge_once(model, exc):
            raise
        return attempt()


def _heal_abandoned_wedge_once(model: ModelSpec, exc: BaseException) -> bool:
    """Reset and retry once when abandoned leases are the only dispatch blocker.

    An untagged worker crash abandons its lease, blocking later dispatches with
    ``LifecycleBusyError``. Recycle permits abandoned leases but requires no live
    sample or pending RDMA read. Confirm the ProcMesh stop, then retry on the new
    fleet. Other busy states and unconfirmed stops still raise; active work is
    never interrupted (docs/TROUBLESHOOTING.md #87).
    """
    from ..rdma_read_job import pending_job_count_for_handle

    spec = getattr(model, "mesh", None)
    handle = getattr(spec, "handle", None)
    if not isinstance(handle, MeshHandle):
        return False
    try:
        abandoned = sum(
            int(count) for count in handle.abandoned_sample_leases.values())
        active = sum(int(count) for count in handle.sample_leases.values())
        pending_reads = pending_job_count_for_handle(handle)
    except Exception:
        return False
    if not abandoned or active or pending_reads:
        return False
    log.info(
        "healing the abandoned-lease wedge: a crashed render left %d abandoned "
        "sample lease(s), so dispatch raises %s; recycling the attached mesh "
        "once, then retrying this submit on the new fleet "
        "(docs/TROUBLESHOOTING.md #87)", abandoned, type(exc).__name__)
    outcome = handle.recycle_detailed()
    if not outcome.ok:
        log.warning(
            "abandoned-lease heal could not reset the mesh (%s: %s); the "
            "original refusal stands", outcome.status.value, outcome.detail)
        return False
    # As on every process-stop path, clear the driver's residency
    # memos must not credit weights the respawned fleet will not hold.
    from . import recycle_drain

    recycle_drain.drop_residency_memos()
    return True


def _submit_render_guarded(
    model: ModelSpec,
    request: dict,
    latent: dict,
    cfg_value: float | None,
    steps_hint: int,
    seq: int,
    depth: int,
    progress_on_step,
    handoff: PendingRenderHandoff | None,
    guard: _SubmitRenderGuard,
    *,
    adoption_context: dict | None = None,
) -> PendingRender:
    """Build and publish a PendingRender while ``guard`` owns every handoff."""
    from .common import (
        _AUTO_GATE_ACTIVE,
        _bind_packed_render_model,
        _process_gate_verdict,
        _resolve_topology_for_latent,
        ensure_live,
        resolve_sample_attention,
    )

    model, bound_handle = _bind_packed_render_model(
        model, latent, cfg_value)
    ref_summary = request.get("_dgxm_krea2_ref_preflight")
    for internal_key in (
        "_dgxm_normal_residency_grant",
        "_dgxm_normal_policy",
        "_dgxm_normal_residency_mode",
        "_dgxm_artifact_sets",
        "_dgxm_artifact_preflight",
        "_dgxm_krea2_ref_preflight",
        _ADOPTION_CONTEXT_REQUEST_KEY,
    ):
        request.pop(internal_key, None)
    model = model_for_request(model, request)
    spec = model.mesh
    handle = (bound_handle if bound_handle is not None
              else ensure_live(spec.handle))
    guard.handle = handle
    if isinstance(handle, MeshHandle) and handoff is None:
        raise ValueError(
            "real render submission requires a caller-owned "
            "PendingRenderHandoff before any session or sample side effect")

    try:
        session, owns_session = claim_render_session(handle, guard.candidate)
    except RetiredRenderHandleError:
        # Supervision may retire the handle after the first ensure_live()
        # result but before the session claim linearizes.  A direct submit's
        # still-unbound candidate can safely resolve exactly once more.  An
        # inherited pipeline session cannot span handles and its second claim
        # therefore fails closed with ConcurrentRenderSessionError.
        refreshed_model, refreshed_handle = _bind_packed_render_model(
            model, latent, cfg_value)
        if refreshed_handle is not None:
            model = refreshed_model
            spec = model.mesh
            handle = refreshed_handle
        else:
            handle = ensure_live(handle)
        guard.handle = handle
        session, owns_session = claim_render_session(handle, guard.candidate)
    session_close = session.close if owns_session else None
    with session.activate():
        world = int(getattr(handle, "world", getattr(spec, "world", 1)))
        topo, sage, reason = _resolve_topology_for_latent(
            model, latent, cfg_value, world)
        render_preflight.preflight_krea2_reference_latents(model, ref_summary, topo)
        render_preflight.preflight_minimax_h3_topology(model, topo)
        sample_attention = resolve_sample_attention(spec.attention, sage)
        render_preflight.preflight_sol_attention(
            model, topo, sample_attention, handle)
        _enforce_persisted_quarantine(
            model,
            handle,
            resolved_topology=topo,
            resolved_attention=sample_attention,
        )
        authorization = authorize_normal_render(
            model, handle, uncond_model_request=request.get("uncond_model"),
            gate_active=bool(getattr(_AUTO_GATE_ACTIVE, "on", False)),
            session_verdict=_process_gate_verdict,
            resolved_topology=topo,
            resolved_attention=sample_attention,
        )

        # A dual-model request is stamped stock whatever any ceremony did,
        # and auto_gate off does not change that, so a blocked ceremony is
        # not what refuses it. The generic refusal, which names the second
        # model, answers it.
        if request.get("uncond_model") is None:
            consent_waiver.assert_ceremony_not_blocked(
                model, handle, authorization)
        model = model_with_worker_overrides(
            model, authorization.worker_args, handle=handle)
        spec = model.mesh
        validate_render_topology(model, topo, latent["samples"])
        log.info("render topology: %s (%s)", topo.describe(), reason)
        setup_token = mesh_setup.ensure_request_setup(
            handle, topo, spec.attention, spec.sync_ulysses, spec.worker_args)
        expected_adoption_context_sha256 = None
        expected_adoption_setup_generation = None
        if adoption_context is not None:
            expected_adoption_context_sha256 = _adoption_context_sha256(
                adoption_context)
            expected_adoption_setup_generation = getattr(
                setup_token, "generation", None)
            if (
                type(expected_adoption_setup_generation) is not int
                or expected_adoption_setup_generation < 0
            ):
                raise RuntimeError(
                    "resident-adoption setup generation is unavailable")
        request["model"] = authorization.model_request
        request["latent"] = pack_latent(_latent_without_topology_metadata(latent))
        request["sage_kernel"] = sample_attention
        request["sync_ulysses"] = spec.sync_ulysses
        request["render_seq"] = int(seq)
        request["pipeline_depth"] = int(depth)
        request["_dgxm_render_id"] = render_id = uuid.uuid4().hex
        # Publish the retirement authority before the stamp can take an audit
        # slot (SubmitRenderGuard.__init__ says why). A healed retry mints a
        # fresh id and never returns to this one.
        guard.render_id = render_id
        consent_waiver.stamp_request(request, model, topo, world, render_id)
        request["_dgxm_normal_residency_mode"] = authorization.residency_mode
        if adoption_context is not None:
            request[_ADOPTION_CONTEXT_REQUEST_KEY] = adoption_context
        if authorization.residency_grant is not None:
            setup_token = cast(mesh_setup.SetupToken, setup_token)
            grant = dict(authorization.residency_grant)
            grant.update(
                setup_generation=setup_token.generation,
                setup_key=setup_token.key,
                worker_args_key=setup_token.worker_args_key,
                worker_topology={
                    "ulysses": topo.ulysses,
                    "ring": topo.ring,
                    "cfg": topo.cfg,
                    "dp": topo.dp,
                    "fsdp": topo.fsdp,
                },
            )
            request["_dgxm_normal_residency_grant"] = grant
            request["_dgxm_normal_policy"] = {
                "topology_preset": spec.topology_preset,
                "attention": spec.attention,
                "sync_ulysses": spec.sync_ulysses,
            }

    timeout = max(_SAMPLE_TIMEOUT_S, steps_hint * 120.0)
    from ..telemetry import render_progress

    # Publish the cleanup intent before either helper can start background
    # work and then lose its return value to an asynchronous exception.
    guard.telemetry_started = True
    render_progress.start(
        steps_hint, {"model": model.unet_name}, token=guard.telemetry_token)
    progress = ProgressReceiver(
        steps_hint, on_step=progress_on_step,
        on_cancel=lambda: handle.cancel_sample(render_id, wait=False))
    guard.progress = progress
    progress.__enter__()

    with session.activate():
        future = mesh_setup.prepare_sample(handle, setup_token)
        guard.future = future
        if future is not None:
            session.track(future)
        guard.dispatch_started = True
        submitted = mesh_setup.submit_sample(
            handle, request, progress.port, setup_token, future)
        if future is None:
            future = submitted
            guard.future = future
            session.track(future)
        elif submitted is not future:
            raise RuntimeError("sample dispatch replaced its prepared authority")

    finish_render = _finish_render
    if adoption_context is not None:
        if (
            expected_adoption_context_sha256 is None
            or expected_adoption_setup_generation is None
        ):
            raise RuntimeError(
                "resident-adoption driver expectations are unavailable")
        finish_render = partial(
            _finish_render,
            expected_adoption_context_sha256=expected_adoption_context_sha256,
            expected_adoption_request_sha256=_adoption_request_sha256(
                request,
                expected_context_sha256=expected_adoption_context_sha256,
            ),
            expected_adoption_render_id_sha256=(
                _adoption_render_id_sha256(render_id)
            ),
            expected_adoption_setup_generation=(
                expected_adoption_setup_generation
            ),
        )
    pending = PendingRender(
        handle, future, progress, topo, latent, timeout, render_id, finish_render,
        session_close=session_close, telemetry_token=guard.telemetry_token)
    guard.pending = pending
    # Finish the caller-owned session transition before publishing the
    # PendingRender. A direct session starts draining here; its tracked future
    # holds the shared owner through the PendingRender's raw read and terminal
    # lease retirement. Under an inherited Pipeline or Gate session the unbound
    # candidate's close is a no-op. A successful handoff then leaves no cleanup
    # whose lost return could strand the owner or an unreachable PendingRender.
    guard.candidate.close()
    guard._candidate_cleaned = True
    if handoff is not None:
        handoff.publish(pending)
    guard.transferred = True
    return pending
