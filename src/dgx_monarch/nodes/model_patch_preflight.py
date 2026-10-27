"""Driver preflights for the per-render model patches (Skip Layer Guidance, the SD3
shift, LoRA) and for MOSS-Image reference images and text attention bias.

All five run once per render in nodes/render_submit.py, after the topology
resolves and before any Worker call, so a refusal costs no dispatch. Each fails
open where it cannot read its facts: the Worker belt (actor/guidance_render.py)
still guards Skip Layer Guidance, the shift guard refuses only a config that
says multiplier 1.0 outright, the model store refuses a LoRA on a refused
quantized file on every Worker (lora_quant_guard.py), and the MOSS-Image
sharded forward refuses reference images on every rank and the bias on every
rank that holds it.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .. import family_select, lora_quant_guard
from ..refusal import RefusalClass, refusal
from ..skip_layer_guidance import (
    normalize_skip_layer_guidance,
    skip_layer_guidance_active,
    skip_layer_guidance_refusal_reason,
)


def preflight_model_patches(model: Any, topo: Any, request: Mapping[str, Any]) -> None:
    """The one driver entry: the skip check, then the shift and LoRA checks on every model."""
    preflight_skip_layer_guidance(model, topo, request)
    preflight_shift_node(model)
    uncond = request.get("uncond_model")
    if uncond is not None:
        preflight_shift_node(uncond)
    preflight_lora_quant(model, uncond)


def honouring_families() -> frozenset[str]:
    """Families whose sharded forward is shown to honor ComfyUI's block replacement."""
    from ..adapters import ADAPTERS

    return frozenset(a.family for a in ADAPTERS if getattr(a, "sp_honours_block_replace", False))


def _sniffed_checkpoint(unet_name: Any, override: str | None) -> tuple[str, str] | None:
    """(family, quant kind) from a checkpoint's header, or None when it cannot be read."""
    try:
        import folder_paths

        from ..adapters.detect import sniff_checkpoint

        path = folder_paths.get_full_path("diffusion_models", unet_name)
        if path is None:
            return None
        family, quant_kind = sniff_checkpoint(path)
        return family_select.effective_family(family, override), quant_kind
    except Exception:
        return None


def _sniffed_family(model: Any) -> str | None:
    """The checkpoint family from its header, or None when it cannot be read."""
    try:
        sniffed = _sniffed_checkpoint(model.unet_name, family_select.override_for_model(model))
    except Exception:
        return None
    return None if sniffed is None else sniffed[0]


def preflight_lora_quant(model: Any, uncond: Any = None) -> None:
    """Refuse a LoRA on a quantized file whose baked LoRA renders wrong (#129).

    Checks the model and the unconditional model of a two-checkpoint guider,
    a ModelSpec or a model request dict alike. A header is read only for a
    model that carries a LoRA, and an unreadable one passes: every Worker's
    model store refuses the same pair before any LoRA build or bake.
    """
    for target in (model, uncond):
        if isinstance(target, Mapping):
            unet_name, loras = target.get("unet_name"), target.get("loras")
        else:
            unet_name, loras = getattr(target, "unet_name", None), getattr(target, "loras", None)
        if not loras or not unet_name:
            continue
        sniffed = _sniffed_checkpoint(unet_name, family_select.override_for_model(model))
        if sniffed is not None:
            lora_quant_guard.assert_lora_quant_supported(*sniffed, list(loras))


def renders_on_one_gpu(topo: Any) -> bool:
    """Whether the resolved topology is one GPU: world 1 and no degree above one."""
    degrees = [int(getattr(topo, name, 1)) for name in ("world", "ulysses", "ring", "cfg", "dp")]
    return max(degrees) == 1 and not getattr(topo, "fsdp", False)


def preflight_moss_image_reference_images(model: Any, ref_summary: Any, topo: Any) -> None:
    """Refuse MOSS-Image reference images on every topology but one GPU (#130).

    ``ref_summary`` is the sampler's (has references, method) pair, read from
    the raw conditioning before the wire packs it. Stock MOSS-Image uses the
    references whatever the method, so only their presence counts. The header
    is read only for a render that carries references off one GPU, and an
    unreadable one passes: on every topology but one GPU each Worker's cond
    bind (adapters/moss_image.py) refuses alike before stock runs a block over
    the references, and the sharded forward refuses again.
    """
    if not ref_summary or not ref_summary[0] or renders_on_one_gpu(topo):
        return
    if _sniffed_family(model) != "moss_image":
        return
    from ..adapters.base import UnsupportedModelError
    from ..adapters.moss_image_seams import REFERENCE_IMAGES_REFUSAL

    raise UnsupportedModelError(REFERENCE_IMAGES_REFUSAL)


def carries_context_attention_bias(request: Mapping[str, Any]) -> bool:
    """Whether a prompt of the render carries an effective MOSS-Image text attention bias.

    Reads both prompts where the sampler puts them: the request's own pair,
    or the guider's on SamplerCustom. ``pack_conditioning`` keeps each
    ``[tensor, extras]`` entry and only moves its tensors to the CPU. The
    Worker's sharded forward decides on the same predicate. A malformed
    entry passes, and so does a bias the predicate cannot read.
    """
    from ..adapters.moss_image_seams import context_attention_bias_is_effective

    guider = request.get("guider")
    prompts = [request.get("positive"), request.get("negative")]
    if isinstance(guider, Mapping):
        prompts += [guider.get("positive"), guider.get("negative")]
    for prompt in prompts:
        for entry in prompt if isinstance(prompt, list | tuple) else ():
            extras = entry[1] if isinstance(entry, list | tuple) and len(entry) > 1 else None
            if not isinstance(extras, Mapping):
                continue
            try:
                if context_attention_bias_is_effective(extras.get("context_attention_bias")):
                    return True
            except Exception:
                continue
    return False


def preflight_moss_image_attention_bias(model: Any, request: Mapping[str, Any], topo: Any) -> None:
    """Refuse a MOSS-Image text attention bias under sequence parallelism (#135).

    Only the sharded forward drops the bias, so only a Ulysses or ring degree
    above one refuses: single, cfg2 and dp2 run stock's forward with it. Both
    prompts count, so on uly2+cfg2 a bias on one of them refuses the whole
    render here, where the forward would refuse in one cfg group while the
    other waits at the cfg gather. The header is read only for a render that
    carries an effective bias under sequence parallelism, and an unreadable
    one passes: the sharded forward then refuses on every rank that holds
    the bias.
    """
    sp = int(getattr(topo, "ulysses", 1)) * int(getattr(topo, "ring", 1))
    if sp <= 1 or not isinstance(request, Mapping) or not carries_context_attention_bias(request):
        return
    if _sniffed_family(model) != "moss_image":
        return
    from ..adapters.base import UnsupportedModelError
    from ..adapters.moss_image_seams import CONTEXT_ATTENTION_BIAS_REFUSAL

    raise UnsupportedModelError(CONTEXT_ATTENTION_BIAS_REFUSAL)


def preflight_skip_layer_guidance(model: Any, topo: Any, request: Mapping[str, Any]) -> None:
    """Refuse an active skip the resolved topology cannot carry (#126).

    The two-checkpoint arm needs no header. The family arm reads it, and the
    cfg arm reads it to name only the topologies that keep the skip on this
    family. An unreadable header disables the family arm alone and leaves the
    cfg text in its broad form: the Worker belt reads the bound adapter instead.
    """
    uncond = request.get("uncond_model")
    uncond_skip = uncond.get("skip_layer_guidance") if isinstance(uncond, Mapping) else None
    if not (skip_layer_guidance_active(getattr(model, "skip_layer_guidance", None))
            or skip_layer_guidance_active(uncond_skip)):
        return
    cfg = int(getattr(topo, "cfg", 1))
    sp = int(getattr(topo, "ulysses", 1)) * int(getattr(topo, "ring", 1))
    dual_model = uncond is not None
    family, honours = "this family", True
    if (sp > 1 or cfg > 1) and not dual_model:
        sniffed = _sniffed_family(model)
        if sniffed is not None:
            family, honours = sniffed, sniffed in honouring_families()
    reason = skip_layer_guidance_refusal_reason(
        cfg=cfg, sp=sp, family=family, family_honours=honours, dual_model=dual_model)
    if reason is None:
        return
    from ..adapters.base import UnsupportedModelError

    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"{reason} To keep this topology and guider instead, clear both layer lists on "
        "the node; the render then runs without the skip.",
        troubleshooting=126))


def refuse_shift_after_skip(model: Any) -> None:
    """Refuse the SD3 shift node wired after a skip whose window the shift would move (#126).

    Stock ComfyUI turns the skip's step window into sigmas when its node runs, from
    the sampling the model has at that point. Every Worker applies the shift first
    and the skip second, so a model that reaches the shift node already carrying a
    skip would get its window from another sampling than stock's. A window of 0 to
    1 is safe: ComfyUI's percent_to_sigma returns both ends before it reads the
    sampling.
    """
    spec = getattr(model, "skip_layer_guidance", None)
    if not skip_layer_guidance_active(spec):
        return
    wire = normalize_skip_layer_guidance(spec)
    if wire is None or (wire["start_percent"] <= 0.0 and wire["end_percent"] >= 1.0):
        return
    from ..adapters.base import UnsupportedModelError

    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        "Model Sampling SD3 (DGX Monarch) follows Skip Layer Guidance (DGX Monarch) in this "
        f"graph, and the skip's step window runs from {wire['start_percent']:g} to "
        f"{wire['end_percent']:g}. Stock ComfyUI turns that window into sigmas when the skip "
        "node runs, from the sampling the model has at that point, while every Worker applies "
        "the shift before the skip, so the skip would fall on other steps than stock's. Put "
        "the shift node before the skip node instead; a window of 0 to 1 is not affected.",
        troubleshooting=126))


def _shift_target(model: Any) -> tuple[Any, Any]:
    """(checkpoint name, shift spec) from a ModelSpec or a model request dict."""
    if isinstance(model, Mapping):
        return model.get("unet_name"), model.get("model_sampling")
    return getattr(model, "unet_name", None), getattr(model, "model_sampling", None)


def preflight_shift_node(model: Any) -> None:
    """Refuse the SD3 shift node on a checkpoint whose config samples at multiplier 1.0 (#128).

    Only an explicit 1.0 refuses. A config with no multiplier key, no config, or
    a detection that raises (some fp8 headers do) fails open, so no family that
    renders today changes outcome unless its own config says 1.0.
    """
    unet_name, sampling = _shift_target(model)
    if sampling is None or not unet_name:
        return
    try:
        import folder_paths

        from ..render_memory_price import detect_comfy_model_config

        path = folder_paths.get_full_path("diffusion_models", unet_name)
        if path is None:
            return
        config = detect_comfy_model_config(path)
        multiplier = (getattr(config, "sampling_settings", None) or {}).get("multiplier")
        config_name = type(config).__name__
    except Exception:
        return
    if isinstance(multiplier, bool) or not isinstance(multiplier, int | float) or multiplier != 1.0:
        return
    from ..adapters.base import UnsupportedModelError

    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        "Model Sampling SD3 (DGX Monarch) applies ComfyUI's ModelSamplingSD3 at "
        f"multiplier 1000, and ComfyUI's {config_name} config for {unet_name} samples at "
        "multiplier 1.0, so this render would be noise (ComfyUI issue 17119 reports it "
        "for MOSS-Image). Remove the node and render with the checkpoint's own sampling "
        "instead; for MOSS-Image that already applies the vendor's shift of 5.",
        troubleshooting=128))
