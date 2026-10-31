"""The packed rows a render samples, and the sample deadline they set.

The driver counts rows for two kinds of family: MiniMax H3 through
``h3_rows``, and the reference and pose families of
``mesh_safety.REF_POSE_TOKEN_FAMILIES`` through the token count their
activation preflight prices (``render_preflight``, docs/TROUBLESHOOTING.md
#47), which reads its reference and pose shapes through ``ref_pose_shapes``
here. ``sample_deadline_s`` turns the count into the seconds
``render_submit`` hands to ``PendingRender`` (``sample_deadline``,
docs/TROUBLESHOOTING.md #139).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .. import family_select, h3_rows, latent_scale, mesh_safety, sample_deadline
from ..adapters.detect import CheckpointSniffError, sniff_checkpoint
from ..h3_activation import H3_FAMILY
from ..log import get_logger
from ..sampling_contract import latent_size_tags

if TYPE_CHECKING:
    from ..topology import Topology

log = get_logger(__name__)


def ref_pose_shapes(request: dict) -> tuple[list, Any]:
    """The reference latent shapes and the pose latent shape in every positive
    entry's extras: the SCAIL pose path can land on a later entry, and
    reference_latents append. Malformed extras degrade to shape-only
    estimation. The negative mirrors the positive and runs as its own model
    call, so the positive bounds the peak."""
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
    return ref_shapes, (getattr(pose_latent, "shape", None)
                        if pose_latent is not None else None)


def packed_rows_for_request(model: Any, request: dict, latent: dict) -> tuple[int, str] | None:
    """The packed rows this render samples, and the family that counted them.

    The family and the sampled grid resolve as in
    ``render_preflight.activation_footprint_preflight_for_request``, without
    its warnings. Every other family, and any input this cannot read, returns
    None, so the deadline keeps the rule it had before rows were counted.
    Never raises: a broken count must not block a render.
    """
    try:
        unet_name = getattr(model, "unet_name", None)
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if not isinstance(unet_name, str) or getattr(samples, "shape", None) is None:
            return None
        import folder_paths

        path = folder_paths.get_full_path("diffusion_models", unet_name)
        if path is None or not path.lower().endswith((".safetensors", ".sft")):
            return None
        forced_family = family_select.override_for_model(model)
        try:
            sniffed = sniff_checkpoint(path)[0]
        except CheckpointSniffError:
            if forced_family is None:
                return None
            sniffed = forced_family
        family = family_select.effective_family(sniffed, forced_family)
        if family != H3_FAMILY and family not in mesh_safety.REF_POSE_TOKEN_FAMILIES:
            return None
        video_shape = latent_scale.rendered_latent_shape(
            path, samples, *latent_size_tags(latent))
        if family == H3_FAMILY:
            rows = h3_rows.rows_from_request(request, latent, video_shape=video_shape)
            total = rows.total if rows.resolved else 0
        else:
            ref_shapes, pose_shape = ref_pose_shapes(request)
            total = sum(mesh_safety.ref_pose_token_counts(
                video_shape, ref_shapes, pose_shape))
        return (total, family) if total > 0 else None
    except Exception as exc:
        log.debug("packed row count unavailable for the sample deadline: %r", exc)
        return None


def sample_deadline_s(model: Any, request: dict, latent: dict,
                      topo: Topology, steps_hint: int) -> float:
    """The seconds this render's collect waits, logged once at INFO.

    The rows divide by the sequence-parallel degree, the only axis that splits
    a sequence: cfg and dp give each rank whole sequences. For MiniMax H3 this
    is the world size its activation preflight divides by, since H3 refuses
    cfg and dp.
    """
    counted = packed_rows_for_request(model, request, latent)
    rows, source = counted if counted is not None else (None, None)
    try:
        ranks = max(int(getattr(topo, "sequence_parallel", 1)), 1)
    except (TypeError, ValueError):
        ranks = 1
    deadline = sample_deadline.sample_deadline(steps_hint, rows, ranks, source=source)
    log.info("%s", sample_deadline.describe(deadline))
    return deadline.seconds
