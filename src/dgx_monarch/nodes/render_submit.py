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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from typing import TYPE_CHECKING, cast

from .. import mesh_setup, restart_reattach, runtime_observer
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
from ..delivery_faults import WorkerDeliveryError
from ..log import get_logger
from ..mesh import MeshHandle, mark_defunct_on_supervision_failure
from ..progress import ProgressReceiver
from ..transfer import pack_latent
from . import consent_waiver, model_patch_preflight, render_preflight
from .abandoned_heal import HealOutcome, heal_abandoned_wedge_once
from .gate_identity import (
    authorize_normal_render,
    model_for_request,
    model_with_worker_overrides,
)
from .pending import PendingRender, PendingRenderHandoff
from .render_quarantine import _enforce_persisted_quarantine
from .render_result import _finish_render
from .render_session import (
    RetiredRenderHandleError,
    active_render_session,
    claim_render_session,
)
from .render_validation import validate_render_topology
from .resolved_topology import telemetry_facts
from .submit_guard import SubmitRenderGuard as _SubmitRenderGuard
from .submit_guard import run_guarded_submit

if TYPE_CHECKING:
    from ..topology import Topology
    from .common import MeshSpec, ModelSpec

log = get_logger(__name__)

_SAMPLE_TIMEOUT_S = float(os.environ.get("DGXM_SAMPLE_TIMEOUT", "3600"))
_latent_without_topology_metadata = render_preflight.latent_without_topology_metadata
_UNCLAIMED_ADOPTION_CONTEXT = object()
# Render lifetime: set only while a RenderPipeline submit runs, and reset when
# that submit returns. It collects the pipeline's earlier results, whose sample
# leases would otherwise refuse this dispatch's setup change.
_COLLECT_BEFORE_SETUP_CHANGE: ContextVar[Callable[[], None] | None] = ContextVar(
    "dgxm_collect_before_setup_change", default=None)
# Render lifetime: what the finish record says about a re-attach inside this
# render (restart_reattach.py). Set only around the one retry after it.
_REATTACH_FACTS: ContextVar[dict | None] = ContextVar("dgxm_reattach_facts", default=None)
# read-only after import: the same two facts for every other render.
_NO_REATTACH = {"reattach": None, "reattach_s": None}


def _reattach_facts() -> dict:
    """The finish record's re-attach facts for this render (a ContextVar read)."""
    return _REATTACH_FACTS.get() or _NO_REATTACH


@contextmanager
def collect_before_setup_change(collect: Callable[[], None]) -> Iterator[None]:
    """Let the submits inside this block collect the caller's earlier results
    when their dispatch needs a different setup from the live one."""
    token = _COLLECT_BEFORE_SETUP_CHANGE.set(collect)
    try:
        yield
    finally:
        _COLLECT_BEFORE_SETUP_CHANGE.reset(token)


def _setup_changes(handle: object, topo: Topology, spec: MeshSpec) -> bool:
    """Whether ``ensure_setup`` for this dispatch changes the live setup.

    The same two comparisons ``MeshHandle.ensure_setup`` makes: the process
    group shape, then the worker settings with the cluster base layer merged
    in. Either change refuses while any sample lease is live. A key that a
    failed policy push cleared reads as a change, as it does there.
    """
    if not isinstance(handle, MeshHandle):
        return False
    shape = mesh_setup.nccl_setup_key(topo, spec.attention, spec.sync_ulysses)
    wanted = handle._worker_args_key(handle.effective_worker_args(spec.worker_args))
    with handle.lock:
        return handle.setup_key != shape or handle.worker_args_key != wanted


def _collect_earlier_results_before_setup_change(
    handle: object, topo: Topology, spec: MeshSpec,
) -> None:
    """Collect a pipeline's in-flight results before this dispatch changes setup.

    A per-combination quarantine gives a failed checkpoint stock settings
    while another checkpoint on the same Init node keeps the risky ones, so
    two dispatches in one pipeline can need different worker settings. The
    earlier render's result lease would make the change refuse, so the
    pipeline collects it first, the way it does before a ceremony.
    """
    collect = _COLLECT_BEFORE_SETUP_CHANGE.get()
    if collect is not None and _setup_changes(handle, topo, spec):
        collect()


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

    guards: list[_SubmitRenderGuard] = []
    # A Pipeline or Gate session is pinned to the mesh it opened on, so only a
    # submit that owns its session may replace that mesh mid-render.
    inherited = active_render_session() is not None

    def attempt() -> PendingRender:
        guard = _SubmitRenderGuard()
        guards.append(guard)

        def body() -> PendingRender:
            return _submit_render_guarded(
                model, request, latent, cfg_value, steps_hint, seq, depth,
                progress_on_step, handoff, guard,
                adoption_context=adoption_context)

        def on_failure(exc: BaseException) -> None:
            runtime_observer.submission_failed(guard, exc)
            if (guard.dispatch_started and guard.handle is not None
                    and isinstance(exc, Exception)):
                mark_defunct_on_supervision_failure(guard.handle, exc)

        return run_guarded_submit(guard, body, on_failure)

    try:
        return attempt()
    except mesh_setup.LifecycleBusyError as exc:
        # Heal the fleet that refused, guard.handle. After a recycle the Init
        # node's cached handle stays retired with its abandoned count, while
        # ensure_live dispatches on a newer fleet. Every busy refusal here
        # follows the guard's publication of that fleet, so ``refused`` is
        # None only under test doubles. The retry starts from ``model`` again.
        refused = guards[-1].handle
        if not _heal_abandoned_wedge_once(
                model if refused is None
                else render_preflight._model_with_bound_handle(model, refused),
                exc):
            raise
        return attempt()
    except WorkerDeliveryError as exc:
        if inherited:
            raise
        return _attempt_after_restart(
            attempt, guards[-1].handle, exc, model, request, latent)


def _attempt_after_restart(
        attempt: Callable[[], PendingRender], refused: object,
        exc: WorkerDeliveryError, model: ModelSpec, request: dict,
        latent: dict) -> PendingRender:
    """Attach again inside this render when every Worker loop restarted, then retry once.

    ``restart_reattach.heal`` retires the stale fleet and attaches by the new
    ids, or raises the refusal that names its remedy. The retry's ``ensure_live``
    then finds the new mesh. Its finish record carries the re-attach and its wall.
    """
    outcome = restart_reattach.heal(refused, exc)
    # As on every process-stop path: the new fleet holds none of the old weights.
    from . import recycle_drain

    recycle_drain.drop_residency_memos()
    # The caller's memory check ran before the restart showed, so it may have
    # credited the old fleet's weights as resident. Price again on the new mesh
    # before the retry loads anything: weights and activations that each fit
    # can still overrun together, and a refusal here is the one a first
    # attempt without the credit raises. The check reads only the latent and
    # the request's positive conditioning, which the first attempt left alone.
    if getattr(model, "mesh", None) is not None and outcome.handle is not None:
        model = render_preflight._model_with_bound_handle(model, outcome.handle)
    render_preflight.activation_footprint_preflight_for_request(model, request, latent)
    token = _REATTACH_FACTS.set(
        {"reattach": restart_reattach.EVENT, "reattach_s": round(outcome.wall_s, 2)})
    try:
        return attempt()
    finally:
        _REATTACH_FACTS.reset(token)


def _heal_abandoned_wedge_once(model: ModelSpec, exc: BaseException) -> bool:
    """Reset and retry once when abandoned leases are the only dispatch blocker.

    ``model`` comes bound to the fleet whose dispatch refused. The heal itself
    lives in ``nodes/abandoned_heal``, which Fleet shares.
    Other busy states and unconfirmed stops still raise; active work is never
    interrupted (docs/TROUBLESHOOTING.md #87).
    """
    handle = getattr(getattr(model, "mesh", None), "handle", None)
    return heal_abandoned_wedge_once(
        handle, blocker=f"dispatch raises {type(exc).__name__}",
        then="retrying this submit on the new fleet") is HealOutcome.HEALED


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
        resolved = _resolve_topology_for_latent(
            model, latent, cfg_value, world)
        topo, sage, reason = resolved
        render_preflight.preflight_krea2_reference_latents(model, ref_summary, topo)
        model_patch_preflight.preflight_moss_image_reference_images(model, ref_summary, topo)
        model_patch_preflight.preflight_moss_image_attention_bias(model, request, topo)
        render_preflight.preflight_minimax_h3_topology(model, topo)
        model_patch_preflight.preflight_model_patches(model, topo, request)
        sample_attention = resolve_sample_attention(spec.attention, sage)
        render_preflight.preflight_sol_attention(
            model, topo, sample_attention, handle)
        # A known FAIL quarantines this combination's dispatch only: the clone
        # carries its levers off, and the graph's shared policy dict keeps what
        # other checkpoints on the same Init node asked for.
        model = _enforce_persisted_quarantine(
            model,
            handle,
            resolved_topology=topo,
            resolved_attention=sample_attention,
        ) or model
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
        topology_label = topo.describe()
        log.info("render topology: %s (%s)", topology_label, reason)
        _collect_earlier_results_before_setup_change(handle, topo, spec)
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

    # What the finished render's summary records beside its wall time: the
    # topology it ran, the auto facts behind it and the kernel it used.
    progress_meta = {
        "model": model.unet_name, "topology": topology_label,
        "attention": sample_attention, **telemetry_facts(resolved),
        **_reattach_facts()}
    # Publish the cleanup intent before either helper can start background
    # work and then lose its return value to an asynchronous exception.
    guard.telemetry_started = True
    render_progress.start(
        steps_hint, progress_meta, token=guard.telemetry_token)
    progress = ProgressReceiver(
        steps_hint, on_step=progress_on_step, step_token=guard.telemetry_token,
        on_cancel=lambda: handle.cancel_sample(render_id, wait=False))
    guard.progress = progress
    progress.__enter__()

    execution_scope = runtime_observer.prepare(request)
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

    execution_scope = runtime_observer.dispatch(request, handle, topo, execution_scope)
    # The collector notes the sampler wall under this render's token, so a
    # pipelined render in flight beside it cannot publish the wall as its own.
    finish_render = partial(_finish_render, telemetry_token=guard.telemetry_token)
    if adoption_context is not None:
        if (
            expected_adoption_context_sha256 is None
            or expected_adoption_setup_generation is None
        ):
            raise RuntimeError(
                "resident-adoption driver expectations are unavailable")
        finish_render = partial(
            finish_render,
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
    finish_render = partial(runtime_observer.finish, execution_scope, finish_render)
    pending = PendingRender(
        handle, future, progress, topo, latent, timeout, render_id, finish_render,
        session_close=session_close, telemetry_token=guard.telemetry_token,
        execution_scope=execution_scope)
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
