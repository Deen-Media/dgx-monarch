"""Shared node-side plumbing: the MeshSpec / ModelSpec graph objects, topology
resolution and the render entry point.

The render dispatch lives elsewhere: ``nodes/render_submit`` assembles and
dispatches one submission, ``nodes/render_result`` collects it, and
``nodes/render_quarantine`` carries a persisted identity-gate FAIL forward.
This module holds what every node imports, plus thin wrappers onto
``nodes/auto_gate``; the maps those wrappers publish into live in
``nodes/gate_process_state``.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from .. import family_select, latent_scale
from ..adapters.detect import (
    CheckpointSniffError,
    sniff_checkpoint_for_topology,
    sniff_nearest_signatures,
)
from ..adoption_evidence import (
    claim_active_context as _claim_adoption_context,
)
from ..adoption_evidence import (
    resident_adoption_evidence_context as resident_adoption_evidence_context,
)
from ..adoption_evidence import (
    suspend_active_context as _suspend_adoption_context,
)
from ..log import get_logger
from ..mesh import MeshHandle, ensure_live
from ..model_sampling import ModelSamplingSpec, freeze_model_sampling, normalize_model_sampling
from ..qwen_image21_cache import normalize_qwen_image21_cache
from ..sampling_contract import latent_size_tags
from ..topology import Topology, choose_auto_topology, topology_from_preset
from ..transfer import pack_conditioning
from ..transfer_utils import (
    prefer_error,
    raise_with_distinct_cause,
    reconcile_error,
    safe_call,
    safe_note,
)
from . import auto_gate as _auto_gate_impl
from . import consent_waiver, gate_inconclusive, gate_process_state, render_preflight
from .gate_identity import model_for_request
from .gate_identity import quarantine_unproven_paths as _quarantine_unproven_paths
from .gate_quarantine_scope import (
    restore_requested_levers as _restore_requested_levers,
)
from .pending import PendingRenderHandoff
from .render_quarantine import (
    _apply_persisted_quarantine as _apply_persisted_quarantine,
)
from .render_submit import submit_render

log = get_logger(__name__)

LATENT_DOWNSCALE_METADATA_KEY = render_preflight.LATENT_DOWNSCALE_METADATA_KEY


@dataclass(frozen=True)
class MeshSpec:
    """Output of DGXMonarchInit: the mesh + the requested parallelism policy."""

    handle: MeshHandle
    topology_preset: str          # "auto" or a PRESETS key
    attention: str
    sync_ulysses: bool
    worker_args: dict = field(default_factory=dict)
    pipeline_depth: int = 1        # cross-render prefetch depth (F2); 1 = off
    auto_gate: str = "first_use"   # "first_use" | "off" (Init widget)

    @property
    def world(self) -> int:
        return self.handle.world


@dataclass(frozen=True)
class ModelSpec:
    """Output of the loader nodes: what to have resident, not the weights."""

    mesh: MeshSpec
    unet_name: str
    options: dict = field(default_factory=dict)
    loras: tuple = ()
    slot: str = "cond"
    model_sampling: ModelSamplingSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_sampling", freeze_model_sampling(self.model_sampling))

    def with_lora(self, name: str, strength: float) -> ModelSpec:
        return replace(self, loras=(*self.loras, {"name": name, "strength": float(strength)}))

    def with_model_sampling(self, value: object | None) -> ModelSpec:
        return replace(self, model_sampling=freeze_model_sampling(value))

    def with_qwen_image21_cache(self, value: object) -> ModelSpec:
        """Return a model handle carrying an explicit Qwen 2.1 cache policy.

        This is a DGXM_MODEL transform.  Wiring Comfy's stock MODEL cache node
        into a remote handle would be a type error and would configure only the
        driver, not either worker.
        """
        cache = normalize_qwen_image21_cache(value)
        if cache is None:
            raise ValueError("qwen_image21_cache must be explicit on a cache node")
        options = dict(self.options)
        options["qwen_image21_cache"] = cache
        return replace(self, options=options)

    def request_dict(self) -> dict:
        options = dict(self.options)
        cache = normalize_qwen_image21_cache(options.get("qwen_image21_cache"))
        if cache is not None:
            options["qwen_image21_cache"] = cache
        return {
            "unet_name": self.unet_name,
            "options": options,
            "loras": [dict(entry) for entry in self.loras],
            "model_sampling": normalize_model_sampling(self.model_sampling),
        }


# A header sniff cannot tie checkpoint variant, graph shape, LoRA, quantization
# or topology scope to a named evidence record, and no runtime family has
# evidence across all of them, so every multi-rank auto family gets the
# once-per-process advisory; docs/MODELS.md rows stay the authority. An
# exemption needs a family-wide promotion record and a matching change to
# tests/test_docs_status_sync.py.
_HW_VALIDATED_FAMILIES = frozenset()  # type: frozenset[str]
_impl_warned: set[str] = set()  # proc lifetime, never evicted: one warning per key
_impl_warned_lock = threading.Lock()


def _first_impl_warn(family: str) -> bool:
    """True exactly once per key: locked test-and-add, so overlapping
    renders (pipeline depth > 1, fleet) cannot race the membership check."""
    with _impl_warned_lock:
        if family in _impl_warned:
            return False
        _impl_warned.add(family)
        return True


def _warn_nearest_signature(path: str) -> None:
    """Name the row an unmatched header came closest to.

    It runs only when no row matched, so it never misnames a supported
    checkpoint, and it tells the operator which family_adapter value to try. A
    near miss identifies nothing, so the text suggests rather than asserts.
    """
    try:
        nearest = sniff_nearest_signatures(path)
    except (CheckpointSniffError, OSError):
        return
    if not nearest:
        return
    best = nearest[0]
    log.warning(
        "no signature matched this checkpoint header; the closest is %s (%d of %d "
        "key paths present, missing %s). If that is the architecture, name it in "
        "the Init node's family_adapter widget; auto has no table row or preflight "
        "for it.", best.family, best.present, best.total, ", ".join(best.missing))


def resolve_topology(model: ModelSpec, latent_samples, cfg_value: float | None,
                     world: int | None = None, *, latent_downscale: int | None = None,
                     size_tags: tuple = (None, None)) -> tuple[Topology, bool, str]:
    """Resolve the mesh's topology preset for this render.

    Returns (topology, sage, reason). Explicit presets pass through; "auto"
    consults the benchmark-seeded table using the checkpoint header sniff.
    ``latent_downscale`` (private benchmark metadata) overrides the ratio
    comfy's latent format declares for the checkpoint. ``size_tags`` are the
    latent's own, so auto reads the grid the workers will sample.
    """
    scale_override = (
        None if latent_downscale is None
        else render_preflight.validate_latent_downscale(latent_downscale)
    )
    spec = model.mesh
    world = spec.world if world is None else int(world)
    if spec.topology_preset != "auto":
        topo = topology_from_preset(spec.topology_preset, world)
        return topo, False, f"preset: {spec.topology_preset}"

    import folder_paths

    path = folder_paths.get_full_path("diffusion_models", model.unet_name)
    if path is None:
        raise FileNotFoundError(
            f"model {model.unet_name!r} not found on the driver; auto topology needs the "
            "checkpoint header. Use an explicit topology preset or add the file."
        )
    sniffed, quant = sniff_checkpoint_for_topology(path)
    forced = family_select.override_for_model(model)
    family = family_select.effective_family(sniffed, forced)
    if forced:
        # A forced family must not emit the header-selected advisory. Warn once
        # regardless of promotion status and log the header result for diagnosis.
        if _first_impl_warn(f"forced:{forced}"):
            log.warning(
                "family_adapter forced this render to %r; the checkpoint header "
                "reads %r. Topology and the family preflights follow the forced "
                "value, and this run carries none of the %s row's hardware "
                "evidence (docs/MODELS.md).", forced, sniffed, forced)
    elif world > 1 and family not in _HW_VALIDATED_FAMILIES:
        if _first_impl_warn(family):
            log.warning(
                "runtime family %r was selected from a checkpoint header, which cannot "
                "bind the graph, artifact, LoRA, quantization, and topology to a named "
                "hardware-evidence scope. Check the exact row and Notes in docs/MODELS.md "
                "before treating this cluster render as hardware-validated.", family)
    # Not inside the cluster advisory: the nearest-row hint concerns an
    # unmatched header, not hardware evidence, so single-GPU auto renders get
    # it too. Keyed by path, so each unmatched file gets its own hint.
    if not forced and family == "unknown" and _first_impl_warn(f"unknown-family:{path}"):
        _warn_nearest_signature(path)
    weight_dtype = model.options.get("weight_dtype", "default")
    # Keep fp32 in this set (the sniff reports it as itself): the loader casts
    # every full-precision storage kind to fp8, so the fp8 row describes the
    # render.
    if quant in {"bf16", "fp16", "fp32"} and weight_dtype.startswith("fp8"):
        quant = "fp8"

    # Per-frame spatial megapixels of the grid the workers will sample, at the
    # ratio comfy's latent format declares for this checkpoint (the legacy
    # 8x/RGB guess when comfy names none). latent_scale holds the one
    # definition, shared with the sweep matrix.
    shape = latent_scale.rendered_latent_shape(path, latent_samples, *size_tags)
    scale = scale_override
    if scale is None:
        scale = latent_scale.spatial_downscale_for(path, latent_samples)
    megapixels = latent_scale.latent_megapixels(shape, scale)

    # fits_resident stays at its True default: auto never selects FSDP. The
    # driver prices its own host through the loader preflight; nothing on the
    # driver prices a remote worker's pool, which needs worker-side facts.
    # Capacity runs use the explicit *+fsdp presets (docs/MODELS.md).
    decision = choose_auto_topology(
        family, quant, megapixels, world, cfg_value=cfg_value,
        batch_size=shape[0])
    return decision.topology, decision.sage, decision.reason


def _resolve_topology_for_latent(
    model: ModelSpec,
    latent: dict,
    cfg_value: float | None,
    world: int | None = None,
) -> tuple[Topology, bool, str]:
    """Resolve one sampler latent: its size tags and private benchmark scale metadata."""
    extra: dict[str, Any] = {}  # only what the latent carries; untagged calls keep their shape
    if LATENT_DOWNSCALE_METADATA_KEY in latent:
        extra["latent_downscale"] = render_preflight.validate_latent_downscale(
            latent[LATENT_DOWNSCALE_METADATA_KEY])
    if latent_size_tags(latent) != (None, None):
        extra["size_tags"] = latent_size_tags(latent)
    return resolve_topology(model, latent["samples"], cfg_value, world, **extra)


def resolve_sample_attention(declared: object, sage: bool) -> str:
    """The kernel string one render actually runs, for every caller that needs it.

    Three sites resolve this: the render submission, the auto-gate trigger, and
    the ceremony's evidence. They must agree exactly, because the value enters
    the gate capability context and is compared for equality against the
    dispatched request at two worker preflights; one site canonicalizing
    differently would make every residency and consent context read stale.

    A sol-attn kernel beats an auto-table sage suggestion; any other kernel
    gives way to SAGE_AUTO. The table gives sage as a resolution bracket,
    which cannot predict a content-dependent sparse kernel, so a row must
    never replace a sol-attn kernel the operator named.
    """
    from ..adapters.sol_attention import is_sol_kernel

    if is_sol_kernel(declared):
        return str(declared)
    return "SAGE_AUTO" if sage else str(declared)


def run_render(model: ModelSpec, request: dict, latent: dict, cfg_value: float | None,
               steps_hint: int) -> dict:
    """Sequential render: eager submit, immediate collect, no overlap. The
    pipelined path shares ``submit_render``; this must stay its depth-1 form."""
    consent_waiver.validate_inherited_stamps(latent)
    # Family-scoped capacity refusal (docs/TROUBLESHOOTING.md #47) before any
    # other side effect; no-op for every family but the registered ones.
    render_preflight.activation_footprint_preflight_for_request(model, request, latent)
    # Claim the opt-in one-render authority before packed preflight, automatic
    # Gate orchestration, model binding, or mesh lifecycle can have a side
    # effect. Copied ContextVars share the locked scope, so only one caller can
    # cross this boundary.
    adoption_context_claim = _claim_adoption_context()

    # Packed DP is a public typed refusal, so surface it before first-use Gate
    # orchestration can run a ceremony or mutate residency policy.
    model, _bound_handle = _bind_packed_render_model(
        model, latent, cfg_value)

    # Prove the path before dispatching the user's full render. A failed
    # ceremony sets the policy to stock first, so no pixels from the disproved
    # slab/lazy-swap path can leave this call. The proof context belongs to the
    # user render, not to nested first-use Gate legs before it. An earlier
    # combination's aborted ceremony proved nothing about this one, so this
    # combination starts from the levers the graph asked for.
    _restore_requested_levers(model)
    with _suspend_adoption_context():
        gate_result = _maybe_auto_gate(
            model, request, latent, cfg_value, steps_hint)
    if gate_result not in gate_inconclusive.NO_QUARANTINE_VERDICTS:
        _quarantine_unproven_paths(model)
    handoff = PendingRenderHandoff()
    try:
        pending = submit_render(
            model_for_request(model, request), request, latent, cfg_value, steps_hint,
            handoff=handoff,
            _claimed_adoption_context=adoption_context_claim,
        )
        result = pending.result()
        render_preflight.note_successful_render(getattr(model, "unet_name", None))
        return result
    except BaseException as primary:
        cleanup_error: BaseException | None = None
        for attempt in range(2):
            try:
                handoff.abort()
            except BaseException as cleanup_exc:
                cleanup_error = prefer_error(
                    cleanup_error,
                    cleanup_exc,
                    f"render handoff cleanup attempt {attempt + 1} failed",
                )
            else:
                break
        interrupted_pending = locals().get("pending")
        if interrupted_pending is not None:
            try:
                interrupted_pending.abandon()
            except BaseException as exc:
                cleanup_error = prefer_error(
                    cleanup_error,
                    exc,
                    "render abandon after terminal result failed",
                )
                safe_call(
                    log.warning, "render abandon after terminal result failed: %r", exc)
        if cleanup_error is not None:
            winner, cause = reconcile_error(
                primary, cleanup_error, "render cleanup also failed")
            if winner is not primary:
                raise_with_distinct_cause(winner, cause)
        raise
    finally:
        interrupted_pending = locals().get("pending")
        if interrupted_pending is not None:
            try:
                handoff.clear(interrupted_pending)
            except BaseException as cleanup_exc:
                active = sys.exception()
                if active is None:
                    raise
                safe_note(active, "render handoff clear failed", cleanup_exc)


# Forward these names to ``nodes/gate_process_state`` at read time. Its
# bindings are republished, so ordinary imports would retain stale objects.
_GATE_PROCESS_STATE_NAMES = frozenset({
    "_AUTO_GATE_ACTIVE",
    "_AUTO_GATE_CONDITION",
    "_AUTO_GATE_LOCK",
    "_AUTO_GATE_RUNNING",
    "_AUTO_GATE_SESSION",
    "_AUTO_GATE_SESSION_LIMIT",
    "_AUTO_GATE_WAIT_S",
    "_PROCESS_GATE_DENIALS",
})

if TYPE_CHECKING:  # the forward answers at runtime; the types come from the home
    from .gate_process_state import (  # noqa: F401
        _AUTO_GATE_ACTIVE,
        _AUTO_GATE_CONDITION,
        _AUTO_GATE_LOCK,
        _AUTO_GATE_RUNNING,
        _AUTO_GATE_SESSION,
        _AUTO_GATE_SESSION_LIMIT,
        _AUTO_GATE_WAIT_S,
        _PROCESS_GATE_DENIALS,
    )


def __getattr__(name: str) -> Any:
    """Read one Gate process-state binding from its home, at the moment asked."""
    if name in _GATE_PROCESS_STATE_NAMES:
        return getattr(gate_process_state, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _record_process_gate_verdicts(
    tokens: list[tuple[str, ...]] | tuple[tuple[str, ...], ...], verdict: str,
    ceremony: dict | None = None) -> None:
    _auto_gate_impl.record_process_gate_verdicts(tokens, verdict, ceremony)


def _retract_process_gate_verdicts(
    tokens: list[tuple[str, ...]] | tuple[tuple[str, ...], ...]) -> None:
    _auto_gate_impl.retract_process_gate_verdicts(tokens)


def _process_gate_verdict(token: tuple[str, ...]) -> str | None:
    return _auto_gate_impl.process_gate_verdict(token)


def _process_gate_verdict_locked(token: tuple[str, ...]) -> str | None:
    return _auto_gate_impl.process_gate_verdict_locked(token)


def _restore_auto_gate_active(previous: bool) -> BaseException | None:
    return _auto_gate_impl.restore_auto_gate_active(previous)


def _release_auto_gate_claim(token: tuple[str, ...]) -> BaseException | None:
    return _auto_gate_impl.release_auto_gate_claim(token)


def _auto_gate_context(
    model: ModelSpec,
    request_kind: str,
    latent: dict | None = None,
    cfg_value: float | None = None,
) -> tuple[str, tuple[str, ...]] | None:
    return _auto_gate_impl.auto_gate_context(
        model, request_kind, latent, cfg_value)


def auto_gate_required(
    model: ModelSpec,
    request_kind: str = "ksampler",
    latent: dict | None = None,
    cfg_value: float | None = None,
) -> bool:
    if latent is not None:
        model, _bound_handle = _bind_packed_render_model(
            model, latent, cfg_value)
    return _auto_gate_impl.auto_gate_required(
        model, request_kind, latent, cfg_value)


def _maybe_auto_gate(model: ModelSpec, request: dict, latent: dict,
                     cfg_value: float | None, steps_hint: int) -> str | None:
    model, _bound_handle = _bind_packed_render_model(
        model, latent, cfg_value)
    return _auto_gate_impl.maybe_auto_gate(
        model, request, latent, cfg_value, steps_hint)


def _bind_packed_render_model(
    model, latent: dict, cfg_value: float | None, *, ensure_live_fn=None
):
    return render_preflight.bind_packed_render_model(
        model, latent, cfg_value,
        resolve_topology=_resolve_topology_for_latent,
        ensure_live_fn=ensure_live if ensure_live_fn is None else ensure_live_fn,
    )


from .pipeline import RenderPipeline  # noqa: E402,F401 - nodes/samplers imports RenderPipeline from here


def conditioning_for_wire(conds) -> Any:
    if conds is None:
        return None
    return pack_conditioning(conds)
