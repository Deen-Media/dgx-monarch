"""Fleet throughput: independent world-1 renders spread across every GPU.

Jobs come from prompt lines encoded by CLIP, CONDITIONING on ``positive``, or
both. Negatives broadcast when single or match the job count. Job i renders
with seed ``noise_seed + i``; results return in prompt order.
"""
from __future__ import annotations

import copy
import uuid

from .. import mesh_setup, sampling_contract, telemetry_fleet
from ..constants import MODEL_TYPE, NODE_CATEGORY
from ..log import get_logger
from ..mesh import ensure_live, mark_defunct_on_supervision_failure
from ..progress import ProgressReceiver
from ..topology import Topology
from ..transfer import pack_latent, read_latent_result
from ..transfer_utils import (
    prefer_error,
    raise_with_distinct_cause,
    reconcile_error,
    safe_note,
)
from . import consent_waiver
from .common import ModelSpec, conditioning_for_wire
from .fleet_cleanup import (
    _cancel_fleet_pending,
    _close_fleet_progress,
    _drain_fleet_collections,
    _FleetPending,
    _retire_fleet_audit,
    _retire_fleet_future,
)
from .fleet_policy import (
    _fleet_worker_policy as _fleet_worker_policy,
)
from .fleet_policy import (
    _FleetAuthorization as _FleetAuthorization,
)
from .render_preflight import (
    latent_without_topology_metadata as _latent_without_topology_metadata,
)
from .render_preflight import (
    preflight_sol_sequence_parallel as _preflight_sol_sequence_parallel,
)
from .render_quarantine import _enforce_persisted_quarantine
from .render_session import RenderSession
from .samplers import _aggregate_render_outputs, _sampler_names, _scheduler_names

log = get_logger(__name__)
MAX_FLEET_JOBS = 1024


def _close_session_preserving_primary(
    session: RenderSession, primary: BaseException | None
) -> None:
    """Close Fleet ownership; cancellation outranks an ordinary render failure."""
    cleanup_error: BaseException | None = None
    for _attempt in range(2):
        try:
            session.close()
        except BaseException as cleanup_exc:
            cleanup_error = prefer_error(
                cleanup_error, cleanup_exc, "fleet session close retry failed")
        else:
            break
    if cleanup_error is None:
        return
    if primary is None:
        raise cleanup_error
    winner, cause = reconcile_error(
        primary, cleanup_error, "fleet session close failed")
    if winner is not primary:
        raise_with_distinct_cause(winner, cause)


def build_jobs(prompts_text: str, positive_list, negative_list, base_seed: int):
    """Assemble one job dict (positive, text, negative, seed) per job.

    Conditioning-list jobs come first, then prompt lines, which carry their raw
    ``text`` and a None ``positive`` until the caller encodes them.
    """
    jobs = []
    for cond in positive_list or []:
        jobs.append({"positive": cond, "text": None})
        if len(jobs) > MAX_FLEET_JOBS:
            raise ValueError(
                f"fleet: more than {MAX_FLEET_JOBS} jobs exceeds the per-prompt limit")
    for line in (prompts_text or "").splitlines():
        line = line.strip()
        if line:
            jobs.append({"positive": None, "text": line})
            if len(jobs) > MAX_FLEET_JOBS:
                raise ValueError(
                    f"fleet: more than {MAX_FLEET_JOBS} jobs exceeds the per-prompt limit")
    if len(jobs) > MAX_FLEET_JOBS:
        raise ValueError(
            f"fleet: {len(jobs)} jobs exceeds the per-prompt limit of {MAX_FLEET_JOBS}")
    negatives = list(negative_list or [])
    if len(negatives) > 1 and len(negatives) != len(jobs):
        raise ValueError(
            f"negative list has {len(negatives)} entries but there are {len(jobs)} "
            "jobs; provide one negative (broadcast) or exactly one per job")
    for i, job in enumerate(jobs):
        job["negative"] = negatives[i] if len(negatives) > 1 else (
            negatives[0] if negatives else None)
        job["seed"] = int(base_seed) + i
    return jobs


def _collect_fleet_wave(handle, pending: list, results: list, timeout_s: float,
                        job_count: int, *, job_outputs: list | None = None,
                        latent_template: dict | None = None,
                        source_latent: dict | None = None) -> None:
    """Drain every submitted job and close every progress receiver.

    A failure is raised only after the rest of the wave has been collected and
    settled, so a failed job cannot strand futures or listening sockets.
    """
    # Normalize ownership records before any collection side effect. A retry
    # then sees exact per-resource state and never recollects a terminal future.
    for index, item in enumerate(list(pending)):
        if not isinstance(item, _FleetPending):
            job_index, future, progress, *metadata = item
            pending[index] = _FleetPending(
                job_index,
                future,
                progress,
                metadata[0] if metadata else None,
            )

    work_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    surface_error: BaseException | None = None

    def record(current, candidate, label):
        nonlocal surface_error
        surface_error = prefer_error(surface_error, candidate, label)
        return prefer_error(current, candidate, label)

    def prepare(item, result):
        if result.get("latent") is None:
            raise RuntimeError(
                f"fleet job {item.job_index} returned no latent ({result})")
        if isinstance(item.future, mesh_setup.SetupBoundFuture):
            results[item.job_index] = read_latent_result(result["latent"], item.future)
        else:
            results[item.job_index] = read_latent_result(result["latent"])
        item.consumed = True
        if job_outputs is not None:
            output = {**(latent_template or {}), "samples": results[item.job_index]}
            extra = result.get("latent_extra")
            if extra is not None and not isinstance(extra, dict):
                raise RuntimeError("fleet worker latent_extra must be a mapping")
            if isinstance(extra, dict) and "batch_index" in extra:
                output["batch_index"] = extra["batch_index"]
            stamps = consent_waiver.prepare_result_stamps(
                [result], source_latent, strict=True)
        else:
            output, stamps = None, None
        return result, output, stamps

    def publish(item, prepared):
        result, output, stamps = prepared
        if job_outputs is not None:
            job_outputs[item.job_index] = consent_waiver.publish_result_stamps(
                output, stamps)
        # Recorded with the job's output: a rejected reply leaves its telemetry row open.
        telemetry_fleet.fleet_jobs.record_job(item.job_index, item.rank, result)
        log.info(
            "fleet job %d/%d done on %s rank %s in %.1fs", item.job_index + 1,
            job_count, result.get("host"), result.get("rank"),
            result.get("sample_s", -1))

    def classify(item, exc):
        from .pending import typed_worker_refusal

        try:
            item.refused = typed_worker_refusal(exc)
        except BaseException as refusal_exc:
            safe_note(exc, "fleet typed-refusal inspection failed", refusal_exc)

    def settle(item):
        nonlocal cleanup_error

        for completed_attr, callback, label in (
            ("audit_retired", lambda: _retire_fleet_audit(item.render_id),
             "fleet render audit retirement failed"),
            ("retired", lambda: _retire_fleet_future(
                item.future, consumed=item.consumed or item.refused),
             "fleet lease retirement failed"),
            ("progress_closed", lambda: _close_fleet_progress(item.progress),
             "fleet progress cleanup failed"),
        ):
            if getattr(item, completed_attr):
                continue
            stage_error: BaseException | None = None
            for _attempt in range(2):
                try:
                    candidate = callback()
                    if candidate is None:
                        setattr(item, completed_attr, True)
                except BaseException as helper_exc:
                    candidate = helper_exc
                else:
                    if candidate is None:
                        break
                stage_error = prefer_error(stage_error, candidate, f"{label} retry failed")
            if stage_error is not None and (
                not getattr(item, completed_attr)
                or not isinstance(stage_error, Exception)
            ):
                cleanup_error = record(cleanup_error, stage_error, label)

    for _item, exc, label in _drain_fleet_collections(
        handle, list(pending), timeout_s, prepare, publish, classify, settle
    ):
        work_error = record(work_error, exc, label)
    for item in list(pending):
        settle(item)

    pending[:] = [
        item for item in pending
        if not (item.retired and item.audit_retired and item.progress_closed)
    ]
    first_error = work_error
    if surface_error is not None and not isinstance(surface_error, Exception):
        first_error = surface_error
    elif cleanup_error is not None:
        first_error = prefer_error(
            first_error, cleanup_error, "fleet wave cleanup also failed")
    if first_error is not None:
        if isinstance(first_error, Exception):
            try:
                mark_defunct_on_supervision_failure(handle, first_error)
            except BaseException as supervision_exc:
                safe_note(
                    first_error, "fleet supervision publication failed",
                    supervision_exc)
        raise first_error


class DGXMonarchFleetKSampler:
    """One independent render per GPU: prompts in, ordered latent batch out."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "clip": ("CLIP",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "prompts": ("STRING", {"multiline": True, "default": "",
                            "tooltip": "One prompt per line, each its own render. Jobs go to the "
                            "GPUs in turn, one per GPU per wave. Leave it empty if you wire a "
                            "conditioning list into `positive` instead."}),
                "noise_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff,
                               "tooltip": "Job i renders with seed noise_seed+i."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_sampler_names(),),
                "scheduler": (_scheduler_names(),),
            },
            "optional": {
                "positive": ("CONDITIONING", {"tooltip": "Optional per-job conditioning "
                             "list from a node that outputs a list. With `prompts` also set, "
                             "these jobs come first, then the prompt lines."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latents (prompt order)",)
    FUNCTION = "fleet"
    CATEGORY = NODE_CATEGORY

    def fleet(self, model, clip, negative, latent_image, prompts, noise_seed,
              steps, cfg, sampler_name, scheduler, positive=None, unique_id=None):
        # INPUT_IS_LIST: every arg arrives as a list; widgets are 1-element.
        spec: ModelSpec = model[0]
        clip0 = clip[0]
        # Private topology hints are input-only. Fleet bypasses common's
        # normal request/output path, so scrub once here and use this clean
        # LATENT for every job plus the returned batch.
        source_latent = dict(latent_image[0])
        consent_waiver.validate_inherited_stamps(source_latent)
        latent = _latent_without_topology_metadata(source_latent)
        prompts_text = prompts[0]
        base_seed, steps_n = int(noise_seed[0]), int(steps[0])
        cfg_v = float(cfg[0])
        sampler, sched = sampler_name[0], scheduler[0]

        jobs = build_jobs(prompts_text, positive, negative, base_seed)
        if not jobs:
            raise ValueError("fleet: no jobs; give prompt lines or a conditioning list")
        for job in jobs:
            if job["negative"] is None:
                raise ValueError("fleet: a negative conditioning input is required")
            if job["positive"] is None:
                tokens = clip0.tokenize(job["text"])
                job["positive"] = clip0.encode_from_tokens_scheduled(tokens)

        mesh = spec.mesh
        handle = ensure_live(mesh.handle)
        session = RenderSession()
        primary: BaseException | None = None
        try:
            # Keep ownership across every wave, including zero-lease gaps.
            session.bind(handle)
            with session.activate():
                _enforce_persisted_quarantine(spec, handle)
                result = self._fleet_bound(
                    spec, latent, jobs, steps_n, cfg_v, sampler, sched, handle,
                    session, source_latent=source_latent)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            cleanup_error: BaseException | None = None
            for _attempt in range(2):
                try:
                    _close_session_preserving_primary(session, primary)
                except BaseException as cleanup_exc:
                    cleanup_error = prefer_error(
                        cleanup_error, cleanup_exc,
                        "fleet session cleanup helper retry failed")
                else:
                    break
            if cleanup_error is not None:
                if primary is None:
                    raise cleanup_error
                winner, cause = reconcile_error(
                    primary, cleanup_error, "fleet session cleanup failed")
                if winner is not primary:
                    raise_with_distinct_cause(winner, cause)
        return result

    def _fleet_bound(self, spec: ModelSpec, latent: dict, jobs: list,
                     steps_n: int, cfg_v: float, sampler: str, sched: str,
                     handle, session: RenderSession, *,
                     source_latent: dict | None = None):
        """Run every fleet wave while one logical render session is bound."""
        mesh = spec.mesh
        # Fleet jobs run world-1 each, a degree Sol-Attn can never host, so the
        # selection refuses here at the driver instead of inside worker setup.
        _preflight_sol_sequence_parallel(Topology(world=1), mesh.attention)
        authorization = _fleet_worker_policy(spec, handle)
        setup_token = mesh_setup.ensure_request_setup(
            handle, Topology(world=1), mesh.attention, mesh.sync_ulysses,
            authorization.worker_args, fleet=True)
        world = handle.world
        telemetry_fleet.open_wave_for(jobs, handle)
        log.info("fleet: %d jobs across %d GPUs (world-1 each, stock single-GPU math)",
                 len(jobs), world)

        try:
            from comfy.utils import ProgressBar

            bar = ProgressBar(steps_n * len(jobs))
        except Exception:
            bar = None
        done_steps = {}

        def on_step(job_idx):
            def _cb(msg):
                done_steps[job_idx] = msg.get("step", 0)
                if bar is not None:
                    bar.update_absolute(sum(done_steps.values()), steps_n * len(jobs))
            return _cb

        results: list = [None] * len(jobs)
        job_outputs: list = [None] * len(jobs)
        pending: list = []
        for i, job in enumerate(jobs):
            try:
                progress = None
                future = None
                entry = None
                published = False
                render_id = None
                # Build the request inside the same try as progress setup and
                # dispatch: a custom tensor wrapper or the grant deepcopy may
                # raise BaseException after an earlier job was submitted, and
                # that job's wave must still be drained.
                request = {
                    "kind": "ksampler_advanced",
                    "model": authorization.model_request,
                    "latent": pack_latent(latent),
                    "positive": conditioning_for_wire(job["positive"]),
                    "negative": conditioning_for_wire(job["negative"]),
                    "noise_seed": job["seed"],
                    "steps": steps_n, "cfg": cfg_v,
                    "sampler_name": sampler, "scheduler": sched, "denoise": 1.0,
                    "advanced": {"add_noise": True, "start_at_step": 0,
                                 "end_at_step": None,
                                 "return_with_leftover_noise": False},
                    "sage_kernel": mesh.attention,
                    "sync_ulysses": mesh.sync_ulysses,
                    "render_seq": i, "pipeline_depth": 1,
                    "_dgxm_fleet_job": True,
                    "_dgxm_render_id": uuid.uuid4().hex,
                }
                if authorization.residency_grant is not None:
                    request["_dgxm_fleet_residency_grant"] = copy.deepcopy(
                        authorization.residency_grant)
                render_id = request["_dgxm_render_id"]
                consent_waiver.stamp_request(request, spec, Topology(world=1), 1, render_id)
                progress = ProgressReceiver(
                    steps_n, on_step=on_step(i),
                    on_cancel=lambda rid=render_id: handle.cancel_sample(rid, wait=False))
                progress.__enter__()
                future = mesh_setup.prepare_sample(handle, setup_token)
                submit_kwargs = mesh_setup.token_kwargs(setup_token)
                if future is not None:
                    session.track(future)
                    submit_kwargs["authority"] = future
                submitted = handle.submit_sample_to(
                    i % world, request, progress_port=progress.port,
                    **submit_kwargs)
                if future is None:
                    future = submitted
                    session.track(future)
                elif submitted is not future:
                    raise RuntimeError(
                        "fleet sample dispatch replaced its prepared authority")
                entry = _FleetPending(i, future, progress, render_id, i % world)
                pending.append(entry)
                published = True
                telemetry_fleet.fleet_jobs.start_job(i, entry.rank)
                # Keep at most `world` jobs in flight.
                if len(pending) == world or i == len(jobs) - 1:
                    _collect_fleet_wave(
                        handle, pending, results,
                        max(900.0, steps_n * 120.0), len(jobs),
                        job_outputs=job_outputs, latent_template=sampling_contract.latent_without_size_tags(latent),
                        source_latent=source_latent,
                    )
            except BaseException as exc:
                cleanup_error: BaseException | None = None
                interrupted_entry = locals().get("entry")
                was_published = bool(locals().get("published")) or (
                    interrupted_entry is not None
                    and any(item is interrupted_entry for item in pending)
                )
                interrupted_future = locals().get("future")
                interrupted_progress = locals().get("progress")
                interrupted_render_id = locals().get("render_id")
                for callback, label in (
                    (lambda rid=interrupted_render_id:
                     _cancel_fleet_pending(handle, pending, extra_render_id=rid),
                     "fleet cancellation before abort drain failed"),
                    (lambda future=interrupted_future, published=was_published:
                     _retire_fleet_future(future, consumed=False)
                     if not published and future is not None else None,
                     "fleet lease cleanup after submission failed"),
                    (lambda progress=interrupted_progress, published=was_published:
                     _close_fleet_progress(progress)
                     if not published and progress is not None else None,
                     "fleet progress cleanup after submission failed"),
                    (lambda: _collect_fleet_wave(
                            handle, pending, results,
                            max(900.0, steps_n * 120.0), len(jobs), job_outputs=job_outputs,
                            latent_template=sampling_contract.latent_without_size_tags(latent),
                            source_latent=source_latent) if pending else None,
                     "fleet drain after submission failed"),
                    (lambda render_id=locals().get("render_id"):
                     _retire_fleet_audit(render_id),
                     "fleet submission audit retirement failed"),
                ):
                    stage_error: BaseException | None = None
                    for _attempt in range(2):
                        try:
                            candidate = callback()
                            if candidate is None:
                                break
                        except BaseException as helper_exc:
                            candidate = helper_exc
                        stage_error = prefer_error(stage_error, candidate, f"{label} retry failed")
                    if stage_error is not None:
                        cleanup_error = prefer_error(
                            cleanup_error, stage_error, label)
                if isinstance(exc, Exception):
                    try:
                        mark_defunct_on_supervision_failure(handle, exc)
                    except BaseException as supervision_exc:
                        safe_note(
                            exc, "fleet submission supervision publication failed",
                            supervision_exc)
                if cleanup_error is not None:
                    winner, cause = reconcile_error(
                        exc, cleanup_error, "fleet submission cleanup failed")
                    if winner is not exc:
                        raise_with_distinct_cause(winner, cause)
                raise

        return (_aggregate_render_outputs(job_outputs, "Fleet results"),)
