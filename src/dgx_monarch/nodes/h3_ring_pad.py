"""Refuse a padded ring for MiniMax H3 and MiniMax-H3-Image on the driver (issue #188).

Both adapters drop divisibility pad rows exactly on pure Ulysses only, and
refuse a padded ring or hybrid layout inside the first model call
(adapters/minimax_h3.py, adapters/minimax_h3_image.py). By then the Workers
have set the layout up and, under ``auto``, loaded the model, and a first-use
gate with something to compare has run its proof render into the same refusal
(docs/TROUBLESHOOTING.md #49). The packed length is known here already: each
conditioning entry is one model call, and its total is the text, guide,
reference, audio and video rows stock's ``PackedLayout`` packs
(``h3_rows.exact_call_total``). So when the resolved layout has a ring degree,
this module counts every call that will run and refuses one that pads, before
the gate, setup and the load, with the Worker's own tag: class K behind the
same waiver for H3, class P for MiniMax-H3-Image. It runs at the first-use
gate's entry (``auto_gate.maybe_auto_gate``), which every sequential render
enters, and at a pipelined render's submit (``pipeline.RenderPipeline``).

It only refuses. The auto decision never reads the conditioning, because
several sites resolve a render's topology and must agree (nodes/common.py
``resolve_topology``). Whatever it cannot read goes on to the Worker's guard:
MiniMax H3 Context Windows, whose windows each pack their own length with the
guides stock crops to them; an unreadable conditioning entry; an entry ComfyUI
may skip at some steps or call on part of the latent (a step window, a default
entry, an area, or a mask set to its bounds); an unreadable latent; a
checkpoint whose family it cannot name.

The same two sites also refuse at once the one ring layout an adapter refuses
whatever the sequence: Wan SCAIL's ring attention above a sequence-parallel
world of two (``refuse_scail_ring``), which ``auto`` reached at four Sparks
by folding SCAIL's uly2 row into uly2+ring2.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .. import family_select, h3_rows, latent_scale
from ..adapters.detect import CheckpointSniffError, sniff_checkpoint
from ..h3_activation import H3_FAMILY
from ..h3_context_windows import normalize_context_windows
from ..log import get_logger
from ..refusal import RefusalClass, refusal
from ..sampling_contract import latent_size_tags
from ..topology import PRESETS
from . import consent_waiver
from .render_preflight import CFG_KEEPS_UNCOND_METADATA_KEY

log = get_logger(__name__)

H3_IMAGE_FAMILY = "minimax_h3_image"
# The one attention the adapters shard, in every shipped H3 and H3-Image file:
# 56 query and 56 key-value heads of 128 (comfy/ldm/minimax/model.py).
H3_HEADS = 56
TROUBLESHOOTING = 49
# The Worker's guard id (adapters/minimax_h3.H3_RING_PAD_GUARD), spelled here
# so the refusal ledger reads it; tests/test_h3_ring_pad_preflight.py holds
# the two equal.
GUARD = "ring_pad:minimax_h3"
SCAIL_FAMILY = "wan_scail"
# SCAIL and SCAIL-2 ship as Wan 14B DiTs: 5120 wide, 40 heads of 128.
SCAIL_HEADS = 40
FAMILIES = (H3_FAMILY, H3_IMAGE_FAMILY, SCAIL_FAMILY)


def _family_and_path(model: Any) -> tuple[str, str] | None:
    """The checkpoint's family and path when it is one of ``FAMILIES``, else None.

    Read as ``render_preflight.preflight_minimax_h3_topology`` reads it, and
    failing open the same way: no path, no ComfyUI, or an unreadable header
    with no forced family leaves the Worker's guard in place.
    """
    unet_name = getattr(model, "unet_name", None)
    if not isinstance(unet_name, str) or not unet_name:
        return None
    try:
        import folder_paths

        path = folder_paths.get_full_path("diffusion_models", unet_name)
        if path is None:
            return None
        forced = family_select.override_for_model(model)
        try:
            sniffed = sniff_checkpoint(path)[0]
        except CheckpointSniffError:
            if forced is None:
                return None
            sniffed = forced
        family = family_select.effective_family(sniffed, forced)
    except Exception:
        return None
    return (family, str(path)) if family in FAMILIES else None


def _entries(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _uncond_runs(source: Mapping[str, Any], cfg_value: Any, latent: Mapping[str, Any]) -> bool:
    """Whether the sampler makes the negative's model call on this model.

    ComfyUI skips the unconditional pass at exactly CFG 1.0 unless the sampler
    keeps it (``math.isclose``, comfy/samplers.py), and the sampler nodes tag
    the latent when it does. A basic guider has no negative, and the
    dual-model guider runs it on its second model.
    """
    if source.get("kind") in ("basic", "dual_model"):
        return False
    if latent.get(CFG_KEEPS_UNCOND_METADATA_KEY):
        return True
    try:
        return cfg_value is not None and not math.isclose(float(cfg_value), 1.0)
    except (TypeError, ValueError):
        return False


# A step window on an entry, as ComfyUI keys it: Conditioning Set Timestep
# Range sets ``start_percent`` and ``end_percent``, a CLIP hook schedule the
# ``clip_`` pair (comfy/sd.py). Only a start at or under 0 and an end at or
# over 1 mean the whole schedule (``percent_to_sigma`` reads them as its ends).
_WINDOW_STARTS = ("start_percent", "clip_start_percent")
_WINDOW_ENDS = ("end_percent", "clip_end_percent")
# The sigmas ComfyUI converts a window into at sampling, on a copy; present on
# an entry here only if something else set them, and no value is read as whole.
_WINDOW_SIGMAS = ("timestep_start", "timestep_end")


def _called_whole(entry: Any) -> bool:
    """Whether ComfyUI calls the model on this entry at every step, on the whole latent.

    ``calc_cond_batch`` (comfy/samplers.py) skips an entry outside its step
    window, so a window short of the whole schedule may never run on a render
    that samples part of a schedule, or on custom sigmas. A ``default`` entry
    runs only where the others leave the latent uncovered, and an ``area``, or a
    mask set to its bounds, narrows the latent the call packs. The count cannot
    say whether such an entry runs, or what it packs, so it leaves it to the
    Worker's guard, as every other doubt here. A mask alone, ``strength``,
    ``hooks`` and ``control`` change neither.
    """
    extra = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else None
    if not isinstance(extra, Mapping):
        return True  # exact_call_total reads it as unreadable and counts nothing
    if "default" in extra or "area" in extra or any(key in extra for key in _WINDOW_SIGMAS):
        return False
    if extra.get("set_area_to_bounds") and "mask" in extra:
        return False
    try:
        starts = [float(extra[key]) for key in _WINDOW_STARTS if key in extra]
        ends = [float(extra[key]) for key in _WINDOW_ENDS if key in extra]
    except (TypeError, ValueError):
        return False
    return all(value <= 0.0 for value in starts) and all(value >= 1.0 for value in ends)


def model_calls(request: Mapping[str, Any], latent: Mapping[str, Any],
                cfg_value: Any) -> list[tuple[str, Any]]:
    """Each conditioning entry that becomes a model call at every step, with its side.

    KSampler requests carry the pair; Sampler Custom carries it in the guider.
    An entry ComfyUI may skip at some steps, or call on part of the latent, is
    left out (``_called_whole``): one that pads beside entries that divide
    would refuse a render the Worker's guard runs.
    """
    guider = request.get("guider")
    source = guider if isinstance(guider, Mapping) else request
    calls = [("positive", entry) for entry in _entries(source.get("positive"))]
    if _uncond_runs(source, cfg_value, latent):
        calls += [("negative", entry) for entry in _entries(source.get("negative"))]
    return [(side, entry) for side, entry in calls if _called_whole(entry)]


def fitting_presets(world: int, heads: int = H3_HEADS) -> list[str]:
    """The presets that run this world on pure Ulysses with whole heads per rank.

    Ring and cfg at one, the degrees multiplying to the world so data
    parallelism stays at one (H3 refuses it), and a Ulysses degree that divides
    ``heads``. No FSDP preset: a ring render never had FSDP (ring with FSDP
    refuses), and H3 under FSDP has no reference comparison. Empty at worlds
    3, 5, 6 and 7.
    """
    names = []
    for name, degrees in PRESETS.items():
        ulysses = int(degrees.get("ulysses", 1))
        if (ulysses > 1 and ulysses == world and heads % ulysses == 0
                and int(degrees.get("ring", 1)) == 1 and int(degrees.get("cfg", 1)) == 1
                and int(degrees.get("dp", 1)) == 1 and not degrees.get("fsdp")):
            names.append(name)
    return names


def _chosen(topo: Any, world: int, model: Any) -> str:
    """How this render came to the layout: ``auto``'s choice or a typed preset."""
    preset = str(getattr(getattr(model, "mesh", None), "topology_preset", "auto"))
    layout = topo.describe()
    return (f"`auto` resolved {layout} at world {world}" if preset == "auto"
            else f"topology {layout} runs at world {world}")


# What a named preset rests on where no run of it is on record: uly2 has the
# project's own runs, and H3's uly4 the community reports its fold rests on.
UNRUN = " (exact by construction; no run of it is on record)"


def _fix(world: int, heads: int, why: str, no_preset: str, run: tuple[str, ...] = ("uly2",)) -> str:
    presets = fitting_presets(world, heads)
    if presets:
        named = " or ".join(name + ("" if name in run else UNRUN) for name in presets)
        return f"Set the Init node's topology to {named}, {why}"
    return (f"No Ulysses preset fills world {world} for this model{no_preset}. "
            "Render on two or four Sparks (uly2 or uly4)")


def _message(family: str, padded: list[tuple[str, int, int]], topo: Any, world: int,
             degree: int, model: Any) -> str:
    side, total, pads = padded[0]
    chosen = _chosen(topo, world, model)
    more = (f" {len(padded) - 1} more call(s) in this render pad as well."
            if len(padded) > 1 else "")
    fix = _fix(world, H3_HEADS, "which drops the pad rows exactly",
               ", and 'single' derives data parallelism, which it refuses",
               ("uly2", "uly4") if family == H3_FAMILY else ("uly2",))
    # H3's Worker refusal cites the left-edge corruption the same mechanism
    # caused on 2026-07-10; MiniMax-H3-Image's states the mechanism alone.
    seen = " (the left-edge corruption measured on 2026-07-10)" if family == H3_FAMILY else ""
    return (
        f"{family}: {chosen}, and the {side} prompt's packed sequence there is {total} "
        f"rows, which needs {pads} divisibility pad row(s) to split {degree} ways.{more} "
        "The exact pad exclusion is Ulysses-only. The model's attention is maskless "
        "and fully bidirectional, so ring or hybrid attention would attend the "
        f"synthetic rows{seen}. The "
        "driver counted the rows before the first-use check, setup and sampling. "
        f"{fix}, or edit the prompt or canvas until each packed "
        f"total divides {degree}, "
    )


# The last way out every refusal here names; a module constant, so the
# refusal ledger reads the alternative a class P message must carry.
SINGLE_GPU = "or take a true single-GPU reference (mode=local, gpus_per_host=1)."


def preflight(model: Any, request: Mapping[str, Any], latent: Mapping[str, Any],
              topo: Any, world: int, cfg_value: Any, *,
              found: tuple[str, str] | None = None) -> None:
    """Refuse a resolved ring layout on which a model call of this render pads.

    ``topo`` is the layout this render resolved and ``world`` the mesh's size.
    A live H3 ring-pad waiver for this combination and layout lets the render
    through, as the Worker's guard would (``guard_waived_for``).
    """
    if int(getattr(topo, "ring", 1) or 1) <= 1 or not isinstance(latent, Mapping):
        return
    found = found or _family_and_path(model)
    if found is None or found[0] not in (H3_FAMILY, H3_IMAGE_FAMILY):
        return
    family, path = found
    image = family == H3_IMAGE_FAMILY
    if not image:
        try:
            if normalize_context_windows(getattr(model, "context_windows", None)) is not None:
                return
        except ValueError:
            return  # a malformed value is the Worker's refusal to make
    samples = latent.get("samples")
    try:
        video_shape = latent_scale.rendered_latent_shape(path, samples, *latent_size_tags(latent))
    except Exception:
        return
    geometry = h3_rows.exact_geometry(latent, video_shape=video_shape, image=image)
    if geometry is None:
        return
    degree = int(getattr(topo, "ulysses", 1) or 1) * int(topo.ring)
    padded = []
    for side, entry in model_calls(request, latent, cfg_value):
        total = h3_rows.exact_call_total(entry, geometry, image=image)
        if total is not None and total % degree:
            padded.append((side, total, -total % degree))
    if not padded:
        return
    from ..adapters.base import UnsupportedModelError

    head = _message(family, padded, topo, world, degree, model)
    if image:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS, head + SINGLE_GPU, troubleshooting=TROUBLESHOOTING))
    from .. import accuracy_waiver

    if guard_waived_for(model, topo, world):
        log.info("minimax_h3: %d call(s) pad on %s, and a live ring-pad waiver covers "
                 "this combination; the Workers apply it", len(padded), topo.describe())
        return
    from ..consent_observe import observe_refusal

    try:
        raise UnsupportedModelError(refusal(
            RefusalClass.KNOWN_WRONG,
            head + SINGLE_GPU,
            guard=GUARD,
            waivable=True,
            panel_action=accuracy_waiver.panel_action(GUARD),
            troubleshooting=TROUBLESHOOTING,
        ) + accuracy_waiver.card_tail_for(
            GUARD, unet_name=str(getattr(model, "unet_name", "")),
            options=getattr(model, "options", None),
            lora_stack=list(getattr(model, "loras", ()) or ()),
            topology=consent_waiver.topology_wire(topo), world=world, family_hint=H3_FAMILY))
    except UnsupportedModelError as exc:
        # The Worker's refusal reaches the panel through the RPC layer's
        # observer; this one never left the driver, so it raises its card here.
        observe_refusal(exc, getattr(getattr(model, "mesh", None), "worker_args", None))
        raise


def guard_waived_for(model: Any, topo: Any, world: object) -> bool:
    """Whether ``consent_waiver.stamp_request`` would carry a live grant for the
    H3 ring-pad guard on this dispatch.

    The same lookup, without stamping, auditing or logging a use. False where
    ``stamp_request`` would stamp nothing: a running first-use ceremony, a
    store fault, or a checkpoint the driver cannot find.
    """
    from .. import accuracy_waiver
    from .consent_rescue import combo_identity

    kind = accuracy_waiver.KNOWN_WRONG_GUARDS.get(GUARD)
    if kind is None or consent_waiver._gate_ceremony_active():
        return False
    try:
        import folder_paths

        unet_name = getattr(model, "unet_name", None)
        if not isinstance(unet_name, str) or not unet_name:
            return False
        path = folder_paths.get_full_path("diffusion_models", unet_name)
        if not path:
            return False
        combo, _artifacts = combo_identity(unet_name, getattr(model, "options", None) or {},
                                           str(path), getattr(model, "loras", ()) or ())
        return kind in consent_waiver.live_grants(
            path=str(path), combo_key=combo, topo=topo, world=world)
    except Exception as exc:  # A store fault grants no waiver, as in stamp_request.
        log.warning("accuracy-waiver lookup for guard %s failed (%r)", GUARD, exc)
        return False


def preflight_for_render(model: Any, request: Mapping[str, Any], latent: Mapping[str, Any],
                         cfg_value: Any, *, resolve_topology: Any = None) -> None:
    """The same check before a render's first-use gate and dispatch.

    Resolves the topology as the render will (``resolve_topology`` defaults to
    ``common._resolve_topology_for_latent``). A resolution that raises is left
    for the render's own resolution to raise, so this adds no new failure.
    """
    found = _family_and_path(model)
    if found is None:
        return
    if resolve_topology is None:
        from .common import _resolve_topology_for_latent as resolve_topology
    mesh = getattr(model, "mesh", None)
    world = getattr(getattr(mesh, "handle", None), "world", getattr(mesh, "world", None))
    try:
        topo, _sage, _reason = resolve_topology(
            model, latent, cfg_value, None if world is None else int(world))
    except Exception:
        return
    if found[0] == SCAIL_FAMILY:
        refuse_scail_ring(model, topo, int(topo.world))
        return
    preflight(model, request, latent, topo, int(topo.world), cfg_value, found=found)


def refuse_scail_ring(model: Any, topo: Any, world: int) -> None:
    """Refuse at once a ring layout Wan SCAIL's adapter refuses at injection.

    ``wan_variants.assert_scail_ring_degree`` refuses ring attention at a
    sequence-parallel world above two, whatever the sequence, inside the
    Worker's injection: after setup and, under ``auto``, the load. At four
    Sparks ``auto`` folds SCAIL's uly2 row into uly2+ring2, so every such render
    met it there (docs/TROUBLESHOOTING.md #136). The predicate reads the layout
    alone, so the driver asks it first and names the Ulysses preset that fits.
    """
    from ..adapters.base import UnsupportedModelError
    from ..adapters.wan_variants import SCAIL_TROUBLESHOOTING, assert_scail_ring_degree

    ring = int(getattr(topo, "ring", 1) or 1)
    try:
        assert_scail_ring_degree(ring, int(getattr(topo, "ulysses", 1) or 1) * ring)
    except UnsupportedModelError:
        fix = _fix(world, SCAIL_HEADS, "which keeps SCAIL on Ulysses attention",
                   ", and ring refuses above a sequence-parallel world of two")
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"wan_scail: {_chosen(topo, world, model)}, and SCAIL refuses ring attention "
            "at a sequence-parallel world above two: community four-Spark users "
            "reported SCAIL-2 renders on ring4 and on this fold outside the 0.10 "
            "one-step limit against one GPU, and no record here covers ring beyond "
            "world two. The driver refused before the first-use check, setup and the "
            f"load. {fix}, " + SINGLE_GPU,
            troubleshooting=SCAIL_TROUBLESHOOTING,
        )) from None
