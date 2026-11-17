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

import math
from typing import TYPE_CHECKING, Any

from .. import family_select, h3_rows, latent_scale, mesh_safety, sample_deadline
from ..adapters.detect import CheckpointSniffError, sniff_checkpoint
from ..h3_activation import H3_FAMILY
from ..h3_context_windows import normalize_context_windows, windowed_rows
from ..log import get_logger
from ..sampling_contract import latent_size_tags

if TYPE_CHECKING:
    from ..topology import Topology

log = get_logger(__name__)


def _percent(value: Any) -> float:
    """A window bound as a float; anything but an int or a float (a bool, a
    string, None) raises TypeError, and the window is then read as throughout."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"window bound {value!r} is not a number")
    return float(value)


def _window(extra: dict) -> tuple[float, float]:
    """The share of the schedule a conditioning entry runs in, as ComfyUI reads
    it (``comfy/samplers.calculate_start_end_timesteps``, the clip pair applied
    in its order; ``get_area_and_mult`` skips an entry only outside it, so both
    ends count), held to 0 to 1, where ``percent_to_sigma`` saturates. A bound
    that is not a finite number never skips the entry in ComfyUI (a NaN sigma
    compares false), so such a window, and an unreadable one, is throughout."""
    try:
        start = _percent(extra.get("start_percent", 0.0))
        end = _percent(extra.get("end_percent", 1.0))
        if "clip_start_percent" in extra:
            start = max(_percent(extra["clip_start_percent"]), start)
            end = min(_percent(extra["clip_end_percent"]), end)
    except (KeyError, TypeError):
        return 0.0, 1.0
    if not (math.isfinite(start) and math.isfinite(end)):
        return 0.0, 1.0
    return min(max(start, 0.0), 1.0), min(max(end, 0.0), 1.0)


def _pose_rows(pose: Any) -> int:
    shape = getattr(pose, "shape", None) if pose is not None else None
    return mesh_safety.ref_pose_token_counts((1, 0, 0, 0, 0), (), shape)[1] if shape is not None else -1


def ref_pose_shapes(request: dict) -> tuple[list, Any]:
    """The reference latent shapes and the pose latent shape of the positive
    entries that can run in one step, a Sampler Custom guider's included
    (``h3_rows.positive_entries``): the SCAIL pose path can land on a later
    entry, and reference_latents append. Entries whose timestep windows do not
    overlap never run together, so the price is the largest set whose windows
    share a point (every set ComfyUI can run at once starts at an entry's own
    start), with its references summed and its largest pose. Entries with no
    window run throughout, as all of them did here before. Malformed extras
    degrade to shape-only estimation."""
    entries = []
    for entry in h3_rows.positive_entries(request):
        extra = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else None
        if not isinstance(extra, dict):
            continue
        refs = [shape for t in extra.get("reference_latents") or ()
                if (shape := getattr(t, "shape", None)) is not None]
        pose = extra.get("pose_video_latent")
        entries.append((_window(extra), refs, pose))
    best: tuple[int, list, Any] = (-1, [], None)
    for at, _end in (window for window, _refs, _pose in entries):
        active = [(refs, pose) for (start, end), refs, pose in entries if start <= at <= end]
        refs = [shape for shapes, _pose in active for shape in shapes]
        # The largest pose of the set, so the count only grows with the set and
        # covers every entry's own call; the first in entry order may be smaller.
        pose = max((pose for _shapes, pose in active), key=_pose_rows, default=None)
        pose_shape = getattr(pose, "shape", None) if pose is not None else None
        rows = sum(mesh_safety.ref_pose_token_counts((1, 0, 0, 0, 0), refs, pose_shape))
        if rows > best[0]:
            best = (rows, refs, pose_shape)
    return best[1], best[2]


def packed_rows_for_request(model: Any, request: dict, latent: dict) -> tuple[int, str] | None:
    """The packed rows this render samples, and the family that counted them.

    The family and the sampled grid resolve as in
    ``render_preflight.activation_footprint_preflight_for_request``, without
    its warnings. Every other family, and any input this cannot read, returns
    None, so the deadline keeps the rule it had before rows were counted.
    Under MiniMax H3 context windows a step runs every window, each carrying
    the text, keyframe and reference rows again, so the count is their sum
    (``h3_context_windows.windowed_rows``). Never raises: a broken count must
    not block a render.
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
            windows = normalize_context_windows(getattr(model, "context_windows", None))
            if windows is not None and rows.resolved:
                total = windowed_rows(latent_t=rows.latent_t, frame_rows=rows.frame_rows,
                                      audio_t=rows.audio // 2,
                                      fixed_rows=rows.text + rows.cond + rows.ref, wire=windows)
        else:
            ref_shapes, pose_shape = ref_pose_shapes(request)
            total = sum(mesh_safety.ref_pose_token_counts(
                video_shape, ref_shapes, pose_shape))
        return (total, family) if total > 0 else None
    except Exception as exc:
        log.debug("packed row count unavailable for the sample deadline: %r", exc)
        return None


def h3_calls_per_step(request: dict, latent: dict, cfg_value: float | None) -> int:
    """The model calls one MiniMax H3 step makes, as the request states them.

    H3 conditionings are separate batch-1 calls (cfg and dp are refused for
    it), so each positive entry is a call, and so is each negative where the
    sampler runs the uncond pass (CFG other than 1, or a sampler that keeps it
    at 1): ``h3_ring_pad.model_calls``, which reads the same entries for the
    ring pad check. An entry that check leaves out, one ComfyUI may skip at
    some steps or call on part of the latent, is not counted, so the count
    never exceeds what every step runs. At least one.

    A two-checkpoint guider runs its negative on its second model, which the
    ring pad check leaves out because it reads this model's layout alone. The
    step waits for that call too, so it is counted here as a cfg guider's
    negative is, the image-only call included where no negative is wired.
    """
    from .h3_ring_pad import model_calls

    guider = request.get("guider") if isinstance(request, dict) else None
    if isinstance(guider, dict) and guider.get("kind") == "dual_model":
        request = {**request, "guider": {**guider, "kind": "cfg",
                                         "negative": guider.get("negative") or [[None, {}]]}}
    try:
        return max(len(model_calls(request, latent, cfg_value)), 1)
    except Exception as exc:
        log.debug("model call count unavailable for the sample deadline: %r", exc)
        return 1


def sample_deadline_s(model: Any, request: dict, latent: dict,
                      topo: Topology, steps_hint: int, cfg_value: float | None = None) -> float:
    """The seconds this render's collect waits, logged once at INFO.

    The rows divide by the sequence-parallel degree, the only axis that splits
    a sequence: cfg and dp give each rank whole sequences. For MiniMax H3 this
    is the world size its activation preflight divides by, since H3 refuses
    cfg and dp, and each model call of a step gets its own allowance. The
    reference and pose families batch their cond and uncond, and keep one.
    """
    counted = packed_rows_for_request(model, request, latent)
    rows, source = counted if counted is not None else (None, None)
    try:
        ranks = max(int(getattr(topo, "sequence_parallel", 1)), 1)
    except (TypeError, ValueError):
        ranks = 1
    calls = h3_calls_per_step(request, latent, cfg_value) if source == H3_FAMILY else 1
    deadline = sample_deadline.sample_deadline(steps_hint, rows, ranks, source=source, calls=calls)
    log.info("%s", sample_deadline.describe(deadline))
    return deadline.seconds
