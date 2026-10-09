"""Render validations that must finish before mesh lifecycle side effects."""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any

from .. import (
    accuracy_waiver,
    driver_footprint,
    family_select,
    h3_activation,
    latent_scale,
    mesh_safety,
    residency_mode,
)
from ..adapters.detect import CheckpointSniffError, sniff_checkpoint
from ..refusal import RefusalClass, refusal
from ..sampling_contract import latent_size_tags
from .latent_outputs import is_direct_nested_tensor
from .render_validation import validate_render_topology

LATENT_DOWNSCALE_METADATA_KEY = "_dgxm_latent_downscale"
RESIDENT_ADOPTION_EVIDENCE_KEY = "_dgxm_adoption_evidence"


def validate_latent_downscale(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("latent_downscale must be an integer")
    if value <= 0:
        raise ValueError("latent_downscale must be a positive integer")
    return value


def latent_without_topology_metadata(latent: dict) -> dict:
    """Remove private input metadata before building the next result.

    Waiver provenance belongs to the render that produced the input.
    ``consent_waiver.stamp_result`` carries it forward as inherited instead of
    letting a clean chained render claim or overwrite that waiver.
    """
    clean = dict(latent)
    clean.pop(LATENT_DOWNSCALE_METADATA_KEY, None)
    clean.pop(RESIDENT_ADOPTION_EVIDENCE_KEY, None)
    clean.pop(accuracy_waiver.STAMPED_RESULT_KEY, None)
    return clean


def preflight_packed_render_topology(
    model: Any,
    latent: dict,
    cfg_value: float | None,
    resolve_topology: Any,
    world: int | None = None,
) -> None:
    """Surface packed-DP refusal before Gate or session side effects."""
    samples = latent.get("samples")
    if (not is_direct_nested_tensor(samples, "input latent samples")
            or not hasattr(model.mesh, "topology_preset")):
        return  # a mesh stand-in without topology_preset cannot resolve a topology to validate
    if world is None:
        handle = getattr(model.mesh, "handle", None)
        world = getattr(model.mesh, "world", getattr(handle, "world", None))
    topo, _sage, _reason = resolve_topology(model, latent, cfg_value, world)
    validate_render_topology(model, topo, samples)


def _model_with_bound_handle(model: Any, handle: Any) -> Any:
    """Clone only the graph handle while preserving its shared policy dict."""
    if getattr(getattr(model, "mesh", None), "handle", None) is handle:
        return model
    if (hasattr(model, "__dataclass_fields__")
            and hasattr(model.mesh, "__dataclass_fields__")):
        return replace(model, mesh=replace(model.mesh, handle=handle))
    cloned = copy.copy(model)
    mesh = copy.copy(model.mesh)
    object.__setattr__(mesh, "handle", handle)
    object.__setattr__(cloned, "mesh", mesh)
    return cloned


def bind_packed_render_model(
    model: Any,
    latent: dict,
    cfg_value: float | None,
    *,
    resolve_topology: Any,
    ensure_live_fn: Any,
) -> tuple[Any, Any | None]:
    """Bind packed validation to the exact mesh resolution that it guards."""
    samples = latent.get("samples")
    mesh = getattr(model, "mesh", None)

    # Project consent before the packed-latent guard so every render path and
    # rank sees the same residency policy. Later quarantine enforcement may
    # overwrite it in the fail-closed direction. Residency belongs to the
    # checkpoint, not to the latent container type.
    from . import consent_projection

    consent_projection.project_for_render(model)

    if (not is_direct_nested_tensor(samples, "input latent samples")
            or mesh is None or not hasattr(mesh, "topology_preset")):
        return model, None

    handle = mesh.handle
    # Cluster config is mutable, so its cached world is advisory. Local GPU
    # count is fixed across healing and can retain the zero-lifecycle fast path.
    if not getattr(getattr(handle, "config", None), "hosts", ()):
        preflight_packed_render_topology(
            model, latent, cfg_value, resolve_topology)

    def validate_current_world(_config: Any, world: int) -> None:
        preflight_packed_render_topology(
            model, latent, cfg_value, resolve_topology, world=int(world))

    handle = ensure_live_fn(handle, mesh_preflight=validate_current_world)
    return _model_with_bound_handle(model, handle), handle


def krea2_ref_preflight_summary(positive, negative=None) -> tuple[bool, str | None]:
    """Combine positive+negative raw CONDITIONING lists into one
    (has_reference_latents, resolved_method) summary for
    preflight_krea2_reference_latents. Call before conditioning_for_wire
    packs either list."""
    from ..adapters.krea2 import krea2_reference_latents_summary

    has_p, method_p = krea2_reference_latents_summary(positive)
    has_n, method_n = krea2_reference_latents_summary(negative)
    return has_p or has_n, method_p if method_p is not None else method_n


def preflight_krea2_reference_latents(
    model: Any, ref_summary: tuple[bool, str | None] | None, topo: Any
) -> None:
    """Refuse unsupported Krea2 reference latents before worker dispatch.

    The shared adapter predicate keeps this driver mirror aligned with the
    worker guard. Sniff failures leave that guard as the backstop. Model option
    patches remain worker-only because no DGX node serializes them.
    """
    if ref_summary is None:
        return
    has_refs, method = ref_summary
    from ..adapters.krea2 import (
        KREA2_REF_LATENTS_REFUSAL_MESSAGE,
        krea2_ref_latents_would_reject,
    )

    if not krea2_ref_latents_would_reject(has_refs, method):
        return
    if topo.sequence_parallel <= 1 and topo.cfg <= 1:
        return
    try:
        import folder_paths

        from ..adapters.detect import sniff_checkpoint

        path = folder_paths.get_full_path("diffusion_models", model.unet_name)
        if path is None:
            return
        family = family_select.effective_family(
            sniff_checkpoint(path)[0], family_select.override_for_model(model))
    except Exception:
        return  # fail open: the mid-forward guard is the backstop
    if family != "krea2":
        return
    from ..adapters.base import UnsupportedModelError

    raise UnsupportedModelError(KREA2_REF_LATENTS_REFUSAL_MESSAGE)


def preflight_minimax_h3_topology(model: Any, topo: Any) -> None:
    """Refuse H3 cfg/dp topology before mesh setup or worker RPC.

    H3 conditionings remain separate batch-1 calls, so cfg parallel has no
    batch to split. Packed data parallel is refused earlier; this predicate is
    its backstop. Sniff failures leave the earlier and worker-side guards in
    place. ``sniff_checkpoint`` is memoized by path metadata.
    """
    from ..adapters.minimax_h3 import (
        MINIMAX_H3_BATCH_CAPPED_MESSAGE,
        minimax_h3_topology_would_reject,
    )

    if not minimax_h3_topology_would_reject(
            int(getattr(topo, "cfg", 1)), int(getattr(topo, "dp", 1))):
        return
    try:
        import folder_paths

        from ..adapters.detect import sniff_checkpoint

        path = folder_paths.get_full_path("diffusion_models", model.unet_name)
        if path is None:
            return
        family = family_select.effective_family(
            sniff_checkpoint(path)[0], family_select.override_for_model(model))
    except Exception:
        return  # fail open: the packed-DP guard and the stock batch cap remain
    if family != "minimax_h3":
        return
    from ..adapters.base import UnsupportedModelError

    raise UnsupportedModelError(MINIMAX_H3_BATCH_CAPPED_MESSAGE)


def _driver_footprint_preflight_for_request(
    model: Any, request: dict, family: str, path: str,
) -> None:
    """Check a pending driver-host DiT load (docs/TROUBLESHOOTING.md #52).

    Packed conditioning proves the driver stack is already in ``MemAvailable``,
    so this site reports but does not charge it. An explicit topology counts
    the weights as resident (the loader site priced its eager load), and a
    dtype cast skips the check. A cleared estimate is memoized (see
    ``_LAST_DRIVER_CHARGED_UNET``); only a complete over-budget one refuses.
    """
    try:
        profile = driver_footprint.DRIVER_STACK_FAMILIES.get(family)
        if (profile is None or driver_footprint.preflight_disabled()
                or not mesh_safety.gpu_is_integrated()
                or driver_footprint.dtype_cast_requested(
                    getattr(model, "options", None))):
            return
        avail = mesh_safety.mem_available_bytes()
        if avail is None:
            return  # off Linux: host MemAvailable bounds nothing here
        unet_name = getattr(model, "unet_name", "") or family
        mesh = getattr(model, "mesh", None)
        handle = getattr(mesh, "handle", None)
        worker_args = dict(getattr(mesh, "worker_args", None) or {})
        if handle is not None:
            # Use the canonical resolver so config reserve and residency agree
            # with the worker-side check.
            from .gate_identity import resolve_effective_worker_args
            worker_args = resolve_effective_worker_args(
                model, handle, requested_worker_args=worker_args)
        weights_resident = (
            str(getattr(mesh, "topology_preset", "auto") or "auto") != "auto"
            or unet_name in (_LAST_RENDERED_UNET, _LAST_SUBMITTED_UNET,
                             _LAST_DRIVER_CHARGED_UNET)
            or driver_footprint.weights_owned_by_activation_preflight(family))
        pixels, blocks, frames = driver_footprint.ref_pixel_volume_from_latents(
            request, profile)
        estimate = driver_footprint.estimate_driver_footprint(
            profile=profile,
            mem_available=avail,
            weight_bytes=driver_footprint.file_size_bytes(path),
            weights_resident=weights_resident,
            co_resident=driver_footprint.rank0_co_resident(mesh),
            slab_weights=residency_mode.pricing_value(worker_args),
            ref_pixels=pixels, ref_blocks=blocks, ref_frames=frames,
            reserve=driver_footprint.reserve_bytes(worker_args))
        driver_footprint.driver_footprint_preflight(
            estimate, unet_name=unet_name, profile=profile)
        if estimate.weight_bytes:
            note_driver_charged_render(unet_name)
    except driver_footprint.DriverFootprintCapacityError:
        raise
    except Exception:
        return  # fail open: a broken estimator must never block a render


def activation_footprint_preflight_for_request(
    model: Any, request: dict, latent: dict,
) -> None:
    """Run family-scoped capacity checks before worker RPC.

    Shared family/path resolution feeds the render-memory price and the
    driver-load, H3 render-fit and reference/pose activation checks
    (docs/TROUBLESHOOTING.md #52, #56 and #47). A missing shape, an unsupported
    file type, or an unreadable header with no forced family fails open, so
    unrelated render paths keep their behavior.
    """
    unet_name = getattr(model, "unet_name", None)
    if not isinstance(unet_name, str):
        return
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if getattr(samples, "shape", None) is None:
        return
    try:
        import folder_paths
    except ImportError:
        return
    path = folder_paths.get_full_path("diffusion_models", unet_name)
    if path is None:
        return  # resolve_topology's own FileNotFoundError covers this later
    if not path.lower().endswith((".safetensors", ".sft")):
        return  # no safetensors header to sniff; conservative no-op
    forced_family = family_select.override_for_model(model)
    try:
        family = family_select.effective_family(
            sniff_checkpoint(path)[0], forced_family)
    except CheckpointSniffError:
        if forced_family is None:
            return  # unreadable/implausible header; degrade rather than crash
        # A forced family stands on its own: an unreadable header is the case
        # the override exists to cover, so the preflights still run.
        family = forced_family
    if forced_family is not None:
        family_select.assert_override_admits_checkpoint(forced_family, path)
    family_select.warn_auto_admits_checkpoint(family, path)
    # Run family-disjoint driver/H3 checks before the activation registry
    # return. Sample memo credit first so this render cannot credit its own
    # newly cleared pending load.
    memo_credit = unet_name in (_LAST_RENDERED_UNET, _LAST_SUBMITTED_UNET,
                                _LAST_DRIVER_CHARGED_UNET)
    from ..render_memory_price import preflight_render_memory

    # The grid the workers will sample, size tags applied, computed once for
    # every estimate below (the latent's own shape when comfy names no config).
    video_shape = latent_scale.rendered_latent_shape(
        path, samples, *latent_size_tags(latent))
    preflight_render_memory(path, unet_name, video_shape)
    _driver_footprint_preflight_for_request(model, request, family, path)
    h3_activation.preflight_h3_activation_for_request(
        model, request, latent, family=family, path=path,
        unet_name=unet_name, memo_credit=memo_credit, latent_shape=video_shape)
    if family not in mesh_safety.REF_POSE_TOKEN_FAMILIES:
        return
    # Scan every positive cond entry's extras dict, not only the first:
    # conditioning_set_values_with_timestep_range (the pose path in comfy's
    # SCAIL node) can land its values on a later entry, and reference_latents
    # append. Malformed (None/tensor) extras slots degrade to shape-only
    # estimation, never abort the render. The negative mirrors the positive per
    # the node, and cond/uncond run as separate model calls, so scanning the
    # positive bounds the peak.
    positive = request.get("positive") if isinstance(request, dict) else None
    ref_latents: list = []
    pose_latent = None
    for entry in positive if isinstance(positive, (list, tuple)) else ():
        extra = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else None
        if not isinstance(extra, dict):
            continue
        ref_latents.extend(extra.get("reference_latents") or ())
        if pose_latent is None:
            pose_latent = extra.get("pose_video_latent")
    ref_shapes = [
        shape for t in ref_latents
        if (shape := getattr(t, "shape", None)) is not None
    ]
    mesh_safety.activation_footprint_preflight(
        family, path, unet_name,
        video_shape=video_shape,
        ref_shapes=ref_shapes,
        pose_shape=getattr(pose_latent, "shape", None) if pose_latent is not None else None,
        sp_degree=1,  # ignores sequence-parallel sharding, which only shrinks
                      # the true footprint: never a false pass, at worst an
                      # over-eager refusal.
        weights_resident=(unet_name in (_LAST_RENDERED_UNET, _LAST_SUBMITTED_UNET)),
    )


# Session lifetime, as are the two memos below; nodes/recycle_drain drains all
# three with the loader-site memos. A stale value in any of them can omit a
# charge but never causes a false refusal. A successful render proves its
# checkpoint is already in ``MemAvailable``.
_LAST_RENDERED_UNET: str | None = None

# Session lifetime. Stamped when preflight clears a pending load, before either
# submission path: a render can fail after loading but before the submitted or
# success memo is set, and this keeps a retry from charging that resident file.
_LAST_DRIVER_CHARGED_UNET: str | None = None

# Session lifetime. Do not charge a checkpoint already loading in a pipeline.
_LAST_SUBMITTED_UNET: str | None = None


def note_submitted_render(unet_name: object) -> None:
    """Record the checkpoint a pipelined submission dispatched (pipeline.py)."""
    global _LAST_SUBMITTED_UNET
    _LAST_SUBMITTED_UNET = unet_name if isinstance(unet_name, str) else None


def note_successful_render(unet_name: object) -> None:
    """Record the checkpoint a successful render left resident (common.py, pipeline.py)."""
    global _LAST_RENDERED_UNET
    _LAST_RENDERED_UNET = unet_name if isinstance(unet_name, str) else None


def note_driver_charged_render(unet_name: object) -> None:
    """Record the checkpoint whose pending load the driver preflight just
    cleared, so a retry after a failed render is not charged for weights the
    worker still holds."""
    global _LAST_DRIVER_CHARGED_UNET
    _LAST_DRIVER_CHARGED_UNET = unet_name if isinstance(unet_name, str) else None


def preflight_sol_sequence_parallel(topo: Any, resolved_attention: object) -> None:
    """Refuse a sol-attn selection on a topology that keeps stock attention.

    Its own function because the eager-setup sites reach a worker before any
    render does: worker setup builds the sol implementation, which refuses
    sp 1 at construction, and a raise there unwinds through the setup rollback
    instead of telling the operator what to change.
    """
    from ..adapters.base import UnsupportedModelError
    from ..adapters.sol_attention import is_sol_kernel

    if not is_sol_kernel(resolved_attention):
        return
    sp_degree = int(getattr(topo, "ulysses", 1)) * int(getattr(topo, "ring", 1))
    if sp_degree >= 2:
        return
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        "the sol-attn kernel is sequence-parallel only and this render "
        f"resolves to a sequence-parallel degree of {sp_degree}, where the "
        "model keeps ComfyUI's own attention. Selecting it here would "
        "record a kernel that never ran. Run this workflow on a uly* "
        "topology, or use a SAGE_* kernel or TORCH_FLASH instead.",
    ))


def preflight_sol_attention(model: Any, topo: Any, resolved_attention: str,
                            handle: Any = None) -> None:
    """Refuse a sol-attn selection this topology, mesh or family cannot host.

    All three are known before dispatch, so refuse there: no lease is taken,
    no rank enters a collective, and the operator sees the reason instead of a
    worker error.

    Topology is the driver's own fact and is checked hard, as is a worker that
    reported no sol-attn kernel. The family is sniffed: a family outside the
    lever's scope refuses here; a failed sniff falls through to the worker's
    scope binding, because the worker's adapter comes from the loaded model and
    is the authority.
    """
    from ..adapters.base import UnsupportedModelError
    from ..adapters.sol_attention import (
        SOL_ATTN_FAMILIES,
        assert_family_supported,
        is_sol_kernel,
    )
    from ..adapters.sol_attention_guards import SOL_INSTALL_SPEC

    if not is_sol_kernel(resolved_attention):
        return
    preflight_sol_sequence_parallel(topo, resolved_attention)
    missing = [
        str(report.get("host") or f"rank {report.get('rank')}")
        for report in (getattr(handle, "worker_capabilities", None) or ())
        if isinstance(report, dict) and report.get("sol_attn") is False
    ]
    if missing:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "the sol-attn kernel is missing on "
            f"{', '.join(sorted(set(missing)))}, and every box in the mesh "
            "needs it: a rank that refuses inside a collective abandons the "
            f"render and strands its peers. Install it there with "
            f"`{SOL_INSTALL_SPEC}`, or use a SAGE_* kernel or TORCH_FLASH "
            "instead.",
        ))
    try:
        import folder_paths

        from ..adapters.detect import sniff_checkpoint

        path = folder_paths.get_full_path("diffusion_models", model.unet_name)
        if path is None:
            return
        family = family_select.effective_family(
            sniff_checkpoint(path)[0], family_select.override_for_model(model))
    except Exception:
        return  # fail open: the worker's own adapter is the authority
    if family not in SOL_ATTN_FAMILIES:
        assert_family_supported(family)
