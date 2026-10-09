"""Actor-side sampling that mirrors stock ComfyUI sampler semantics exactly.

The KSampler paths implement stock `common_ksampler` behavior; the custom
path builds stock guiders (comfy.samplers.CFGGuider and subclasses) from a plain
guider SPEC dict and wraps, never forks, the stock SAMPLER/SIGMAS objects the
graph handed the driver (DESIGN.md §5.7).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..log import get_logger
from ..sampling_contract import custom_schedule_steps
from ..transfer_utils import failure_summary, safe_call
from . import latent_outputs
from .sampling_validation import require_known_sampler

log = get_logger(__name__)


class RenderCancelledError(RuntimeError):
    """Raised synchronously on every rank at a denoise step boundary."""


def _normalize_sample_output(value: Any, path: str = "sampler output") -> Any:
    """Return an independent, detached, contiguous CPU latent tree."""
    return latent_outputs.normalize_tensor_tree(value, path)


def _custom_denoised_output(guider, samples: Any, x0: Any) -> Any:
    """Apply stock x0 processing and strictly restore packed modalities."""
    return latent_outputs.custom_denoised_output(
        guider, samples, x0, _normalize_sample_output
    )


def _cond_tensor(entry):
    """The (B, tokens, dim) cross-attention tensor of one conditioning entry,
    or None for entries without one (e.g. the [[None, {}]] image-only uncond)."""
    if not isinstance(entry, (list, tuple)) or not entry:
        return None
    candidate = entry[0]
    return candidate if torch.is_tensor(candidate) and candidate.ndim == 3 else None


def equalize_cond_lengths(adapter, positive, negative, latent_samples):
    """Give cond and uncond equal token counts so ComfyUI batches them.

    cfg-parallel splits the batched cond+uncond model call across the cfg
    group, and ComfyUI only builds that batched call when both conditionings
    share a cross-attention shape. A blank or much shorter negative would
    otherwise disable cfg-parallel. Families opt in through
    `Adapter.cfg_cond_padding`:

      "pad": append zero rows, and extend a shipped text keep-mask with zeros
                    to match; where none ships, the family's patched forward
                    re-derives one (krea2: inject_cfg_pad_forward).
      "pad+mask": append zero rows and attach an additive attention bias to
                    the conditioning dict; the stock forward applies it over
                    the joint [text, image] keys. Chroma instead trims both
                    back off in its own cfg-pad forward before stock sees
                    them (inject_cfg_pad_forward); Flux 1.x and LongCat hand
                    the bias straight to stock.
      "pad+text-mask": append zero rows and attach a text-only additive
                    attention bias shaped (B, text). Hunyuan expands that bias
                    over its joint stream inside the stock forward.
      "none": return the inputs untouched.

    Only pure-cfg topologies pad: under sp > 1 the USP kernels cannot apply
    the pad mask. The caller (actor/sample_protocol.py) gates on sp == 1 and
    says what uly*+cfg* topologies take instead. Every call records
    nvfp4's cfg fold fact (`cfg_dispatch.record_cfg_pair_lengths`): the original
    lengths, before any pad, when the family's `cfg_pad_restores_stock_call` is
    true, and otherwise an empty list, which reads as a fold.
    """
    from ..adapters import cfg_dispatch
    rule = adapter.cfg_cond_padding
    entries = [*positive, *negative]
    lengths = [t.shape[1] for t in map(_cond_tensor, entries) if t is not None]
    cfg_dispatch.record_cfg_pair_lengths(
        lengths if getattr(adapter, "cfg_pad_restores_stock_call", False) else [])
    if rule == "none" or not lengths:
        return positive, negative
    longest = max(lengths)
    if min(lengths) == longest:
        return positive, negative
    if rule == "pad+text-mask":
        extended = []
        for entry in entries:
            extras = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            if extras.get("conditioning_byt5small") is not None:
                extended.append("ByT5")
            if extras.get("clip_vision_output") is not None:
                extended.append("CLIP vision")
        if extended:
            from ..adapters.base import UnsupportedModelError

            kinds = "/".join(sorted(set(extended)))
            raise UnsupportedModelError(
                f"hunyuan cfg-parallel cannot equalize asymmetric Qwen prompt lengths "
                f"when {kinds} conditioning is present: stock Hunyuan appends those tokens "
                "after consuming the Qwen text mask, so a padded mask would have the wrong "
                "width. Use equal-length positive and negative prompts, or a non-cfg topology."
            )

    image_keys = grid_h = grid_w = 0
    if rule in ("pad+mask", "pad+text-mask"):
        for entry in entries:
            if len(entry) > 1 and isinstance(entry[1], dict) and entry[1].get("attention_mask") is not None:
                log.warning(
                    "cfg cond equalization skipped: a conditioning already ships its own "
                    "attention_mask (regional prompt?); cfg-parallel may not engage")
                return positive, negative
        if rule == "pad+mask":
            if int(latent_samples.shape[1]) == 3:  # pixel space has no latent grid
                from ..adapters import cfg_parallel

                cfg_parallel.refuse_pixel_space_pad_mask(adapter.family)
            # Token grid as stock chroma computes it (patch 2); joint-key bias.
            grid_h = (int(latent_samples.shape[-2]) + 1) // 2
            grid_w = (int(latent_samples.shape[-1]) + 1) // 2
            image_keys = grid_h * grid_w
    if rule == "pad+mask":
        # Align the joint (text + image) key extent to a multiple of 8 with
        # masked-out extra columns: same math, and an odd extent drives cudnn's
        # f16 flash-fprop SDPA into a misaligned-address fault.
        # 63x63-style grids make the image side odd, so align the sum.
        longest = -(-(longest + image_keys) // 8) * 8 - image_keys
    elif rule == "pad+text-mask":
        # Hunyuan builds its own joint mask from (B, text), so align the text extent.
        longest = -(-longest // 8) * 8

    def stretch(cond_list):
        out = []
        for entry in cond_list:
            tensor = _cond_tensor(entry)
            extras = dict(entry[1]) if len(entry) > 1 and isinstance(entry[1], dict) else {}
            rows = tensor.shape[1] if tensor is not None else longest
            if tensor is not None and rows < longest:
                tensor = F.pad(tensor, (0, 0, 0, longest - rows))  # zero rows on dim 1
            if rule == "pad" and tensor is not None and rows < longest:
                # A conditioning that already ships a text keep-mask needs the
                # mask stretched too, or the pair still will not batch: comfy
                # checks every model_cond key (samplers.py cond_equal_size) and
                # a plain CONDRegular demands exact shape equality (conds.py).
                # Lens's extra_conds publishes cross-attention and mask as CONDRegular,
                # so without this ComfyUI will not batch its cfg2 pair and cfg-parallel
                # refuses the batch-one call. Zero is the drop value:
                # 1 marks a real token (sd1_clip.py process_tokens).
                mask = extras.get("attention_mask")
                if torch.is_tensor(mask) and mask.ndim == 2 and mask.shape[-1] == rows:
                    extras["attention_mask"] = F.pad(mask, (0, longest - rows))
                elif mask is not None:
                    log.warning(
                        "cfg cond equalization padded the text rows but left an "
                        "attention_mask alone: it does not span the %d text tokens it "
                        "accompanies, so cfg-parallel may still refuse a batch of one",
                        rows)
            elif rule == "pad+mask" and tensor is not None:
                # Additive float bias over the joint keys: appended text rows get
                # the dtype minimum, all else 0; float, comfy casts to model dtype.
                key_pos = torch.arange(longest + image_keys, device=tensor.device)
                appended = (key_pos >= rows) & (key_pos < longest)
                extras["attention_mask"] = (
                    appended.to(tensor.dtype) * torch.finfo(tensor.dtype).min).reshape(1, 1, -1)
                extras["attention_mask_img_shape"] = (grid_h, grid_w)
            elif rule == "pad+text-mask" and tensor is not None:
                # Hunyuan's txt_mask contract is (B, text), not Chroma's joint
                # bias. Keep it floating/additive: stock Hunyuan uses it as-is in
                # TokenRefiner and its own joint key mask; model_base retains an
                # all-zero mask (concatenable conds).
                key_pos = torch.arange(longest, device=tensor.device)
                appended = key_pos >= rows
                bias = appended.to(tensor.dtype) * torch.finfo(tensor.dtype).min
                extras["attention_mask"] = bias.unsqueeze(0).expand(tensor.shape[0], -1)
            out.append([tensor if tensor is not None else entry[0], extras])
        return out

    log.info("cfg cond equalization: text lengths %s -> %d (%s)", sorted(set(lengths)), longest, rule)
    return stretch(positive), stretch(negative)


def build_guider(model_patcher, spec: dict, uncond_patcher=None):
    """Materialize a guider SPEC into a stock comfy guider object.

    Spec kinds:
      {"kind": "basic", "positive": conds}
      {"kind": "cfg", "positive": conds, "negative": conds, "cfg": float}
      {"kind": "dual_model", "positive": conds, "negative": conds|None, "cfg": float}
    """
    import comfy.samplers

    kind = spec.get("kind")
    if kind == "basic":
        class _BasicGuider(comfy.samplers.CFGGuider):
            def set_conds(self, positive):
                self.inner_set_conds({"positive": positive})

        guider = _BasicGuider(model_patcher)
        guider.set_conds(spec["positive"])
        return guider

    if kind == "cfg":
        guider = comfy.samplers.CFGGuider(model_patcher)
        guider.set_conds(spec["positive"], spec["negative"])
        guider.set_cfg(float(spec["cfg"]))
        return guider

    if kind == "dual_model":
        from .dual_model_cfg import build_dual_model_guider

        return build_dual_model_guider(model_patcher, spec, uncond_patcher)

    raise ValueError(f"unknown guider spec kind {kind!r}")


def _dp_info() -> tuple[int, int]:
    from xfuser.core.distributed import get_data_parallel_rank, get_data_parallel_world_size

    return get_data_parallel_rank(), get_data_parallel_world_size()


def is_result_leader() -> bool:
    """Leader-only result transfer: sp rank 0 and cfg rank 0 of each dp group."""
    from xfuser.core.distributed import (
        get_classifier_free_guidance_rank,
        get_sequence_parallel_rank,
    )

    return get_sequence_parallel_rank() == 0 and get_classifier_free_guidance_rank() == 0


def split_batch_for_dp(latent_samples: torch.Tensor) -> torch.Tensor:
    """data-parallel: each dp group renders its slice of the latent batch.

    Noise is generated for the full batch from the request seed and sliced
    identically, so the math matches a single-GPU batch render exactly.
    """
    dp_rank, dp_world = _dp_info()
    if dp_world <= 1:
        return latent_samples
    from ..transfer import is_nested_tensor

    if is_nested_tensor(latent_samples):
        raise RuntimeError(
            "dp over packed multi-modality latents (LTX AV) is not supported; "
            "use a uly or ring topology for AV renders."
        )
    batch = latent_samples.shape[0]
    if batch % dp_world != 0:
        raise RuntimeError(
            f"dp{dp_world} needs latent batch size divisible by {dp_world} (got {batch}). "
            "Raise the batch to a multiple of the dp degree, or lower the degree."
        )
    return torch.chunk(latent_samples, dp_world, dim=0)[dp_rank]


def split_mask_for_dp(noise_mask: torch.Tensor | None, latent_batch: int) -> torch.Tensor | None:
    """dp slice for a noise mask, honoring stock broadcast semantics.

    Stock comfy broadcasts the mask to the latent batch inside prepare_mask
    (comfy.utils.repeat_to_batch_size), so a single mask over a batch-N latent
    is a normal workflow. Broadcast the same way before slicing.
    """
    if noise_mask is None:
        return None
    _, dp_world = _dp_info()
    if dp_world <= 1:
        return noise_mask
    if noise_mask.shape[0] != latent_batch:
        import comfy.utils

        noise_mask = comfy.utils.repeat_to_batch_size(noise_mask, latent_batch)
    return split_batch_for_dp(noise_mask)


def _slice_tensor_for_dp(value: torch.Tensor, latent_batch: int, path: str) -> torch.Tensor:
    """Slice one batch-leading conditioning tensor for this DP rank.

    Stock ``CONDRegular.process_cond`` cycles or truncates any leading batch
    with ``repeat_to_batch_size``. Materialize that global batch before taking
    this rank's slice; repeating only after the split would restart the cycle at
    row zero on every DP rank and misalign prompts with latents.
    """
    dp_rank, dp_world = _dp_info()
    if dp_world <= 1 or value.ndim == 0:
        return value
    if latent_batch % dp_world:
        raise RuntimeError(
            f"dp{dp_world} needs latent batch size divisible by {dp_world} (got {latent_batch})"
        )
    leading = int(value.shape[0])
    if leading == 1:
        return value
    if leading == 0:
        raise RuntimeError(
            f"dp{dp_world} cannot align empty conditioning tensor {path}"
        )
    if leading != latent_batch:
        import comfy.utils

        value = comfy.utils.repeat_to_batch_size(value, latent_batch)
    local_batch = latent_batch // dp_world
    return value.narrow(0, dp_rank * local_batch, local_batch)


def _slice_control_for_dp(
    control, latent_batch: int, path: str, exempt_keys: frozenset[str] = frozenset()
):
    """Clone a Comfy ControlBase-like chain and DP-slice its input tensors."""
    if not (
        hasattr(control, "copy")
        and hasattr(control, "cond_hint_original")
        and hasattr(control, "previous_controlnet")
    ):
        raise RuntimeError(
            f"DP conditioning contains an opaque control object at {path}; DP cannot split its "
            "batch inputs per rank. Use a non-DP topology for this control implementation."
        )
    try:
        cloned = control.copy()
    except Exception as exc:
        raise RuntimeError(
            f"DP could not clone the control object at {path}; use a non-DP topology"
        ) from exc

    cloned.cond_hint_original = _slice_dp_value(
        control.cond_hint_original, latent_batch, f"{path}.cond_hint_original", exempt_keys
    )
    if hasattr(control, "extra_concat_orig"):
        cloned.extra_concat_orig = _slice_dp_value(
            control.extra_concat_orig, latent_batch, f"{path}.extra_concat_orig", exempt_keys
        )
    if hasattr(control, "extra_args"):
        cloned.extra_args = _slice_dp_value(
            control.extra_args, latent_batch, f"{path}.extra_args", exempt_keys
        )
    previous = control.previous_controlnet
    cloned.previous_controlnet = (
        _slice_control_for_dp(previous, latent_batch, f"{path}.previous_controlnet", exempt_keys)
        if previous is not None else None
    )
    # Never carry a cache produced for the full batch into a local DP render.
    if hasattr(cloned, "cond_hint"):
        cloned.cond_hint = None
    if hasattr(cloned, "extra_concat"):
        cloned.extra_concat = None
    return cloned


def _slice_dp_value(value, latent_batch: int, path: str, exempt_keys: frozenset[str] = frozenset()):
    if torch.is_tensor(value):
        return _slice_tensor_for_dp(value, latent_batch, path)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key in exempt_keys:
                # A raw shared sequence (e.g. anima's t5xxl_ids/t5xxl_weights):
                # stays whole and untouched, identical on every dp rank.
                out[key] = item
            elif key == "control" and item is not None and not isinstance(item, (dict, list, tuple)):
                out[key] = _slice_control_for_dp(item, latent_batch, item_path, exempt_keys)
            else:
                out[key] = _slice_dp_value(item, latent_batch, item_path, exempt_keys)
        return out
    if isinstance(value, list):
        return [
            _slice_dp_value(item, latent_batch, f"{path}[{i}]", exempt_keys)
            for i, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _slice_dp_value(item, latent_batch, f"{path}[{i}]", exempt_keys)
            for i, item in enumerate(value)
        )
    return value


def split_conditioning_for_dp(
    conds, latent_batch: int, name: str = "conditioning", exempt_keys: frozenset[str] = frozenset()
):
    """Return conditioning aligned with this rank's latent slice."""
    _, dp_world = _dp_info()
    if dp_world <= 1 or conds is None:
        return conds
    return _slice_dp_value(conds, latent_batch, name, exempt_keys)


def split_guider_for_dp(
    spec: dict, latent_batch: int, exempt_keys: frozenset[str] = frozenset()
) -> dict:
    """DP-slice the conditioning carried by a custom guider spec."""
    _, dp_world = _dp_info()
    if dp_world <= 1:
        return spec
    out = dict(spec)
    for key in ("positive", "negative"):
        if out.get(key) is not None:
            out[key] = split_conditioning_for_dp(out[key], latent_batch, f"guider.{key}", exempt_keys)
    return out


def make_progress_callback(progress_port, total_steps: int, cancel_event=None):
    """Stream leader progress and observe driver cancellation on every rank."""
    if progress_port is None and cancel_event is None:
        return None, lambda: None

    def callback(step, x0, x, total):
        if cancel_event is not None and cancel_event.is_set():
            raise RenderCancelledError("distributed render cancelled by the ComfyUI driver")
        if progress_port is not None:
            try:
                progress_port.send({"step": int(step) + 1, "total": int(total or total_steps)})
            except Exception as exc:
                safe_call(log.debug, "progress channel failure: %s", failure_summary(exc))

    def finish():
        try:
            if progress_port is not None:
                progress_port.send({"done": True})
        except Exception as exc:
            safe_call(log.debug, "progress finish signal failed: %s", failure_summary(exc))

    return callback, finish


def model_sampling_render_clone(model_patcher: Any, value: object | None) -> Any:
    """Return a per-render patcher with an explicit sampling-object patch.

    Comfy patcher clones share the underlying model and object-patch backup. A
    shifted render can therefore leave that shared model holding its temporary
    sampling object until another patcher is loaded. Always patch a fresh clone
    with the resident patcher's stock object, even when this request asks for no
    override, so an unshifted render cannot inherit the prior render's shift.
    """
    from ..model_sampling import normalize_model_sampling

    model_sampling = normalize_model_sampling(value)
    stock_sampling = model_patcher.get_model_object("model_sampling")
    render_patcher = model_patcher.clone()
    render_patcher.add_object_patch("model_sampling", stock_sampling)
    if model_sampling is None:
        return render_patcher

    from comfy_extras.nodes_model_advanced import ModelSamplingSD3

    shifted, = ModelSamplingSD3().patch(render_patcher, model_sampling["shift"])
    return shifted


def run_ksampler(
    model_patcher, request: dict, progress_port=None, cancel_event=None,
    dp_cond_exempt_keys: frozenset[str] = frozenset(),
) -> tuple:
    """The KSampler / KSamplerAdvanced execution path (stock semantics).

    Returns (samples, out_dict_or_None): samples on every rank (the identity
    gate compares their stats), the latent dict only on the result leader.
    """
    import comfy.sample as comfy_sample

    require_known_sampler(request["sampler_name"], request["scheduler"])
    latent = request["latent"]
    # Stock order: fix once (not idempotent) with the size tags; noise reads the full batch.
    full_latent = latent_outputs.fixed_empty_latent(model_patcher, latent)
    latent_batch = int(full_latent.shape[0])
    positive = split_conditioning_for_dp(
        request["positive"], latent_batch, "positive", dp_cond_exempt_keys)
    negative = split_conditioning_for_dp(
        request["negative"], latent_batch, "negative", dp_cond_exempt_keys)
    latent_image = split_batch_for_dp(full_latent)

    advanced = request.get("advanced") or {}
    disable_noise = bool(advanced.get("add_noise") is False)
    seed = int(request["noise_seed"])

    if disable_noise:
        noise = latent_outputs.zeros_like_latent(latent_image, "KSampler latent")
    else:
        noise = comfy_sample.prepare_noise(full_latent, seed, latent.get("batch_index"))
        noise = split_batch_for_dp(noise)

    noise_mask = split_mask_for_dp(latent.get("noise_mask"), int(latent["samples"].shape[0]))

    leader = is_result_leader()
    callback, finish = make_progress_callback(
        progress_port if leader else None, int(request["steps"]), cancel_event)

    start_step = advanced.get("start_at_step")
    last_step = advanced.get("end_at_step")
    force_full_denoise = bool(advanced) and not advanced.get("return_with_leftover_noise", False)

    try:
        samples = comfy_sample.sample(
            model_patcher,
            noise,
            int(request["steps"]),
            float(request["cfg"]),
            request["sampler_name"],
            request["scheduler"],
            positive,
            negative,
            latent_image,
            denoise=float(request.get("denoise", 1.0)),
            disable_noise=disable_noise,
            start_step=start_step,
            last_step=last_step,
            force_full_denoise=force_full_denoise,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=True,
            seed=seed,
        )
    finally:
        finish()

    latent_outputs.validate_tensor_tree(
        samples, "KSampler output", require_finite=True
    )
    if not leader:
        return samples, None
    out = dict(latent)
    out.pop("noise_mask", None)
    out["samples"] = _normalize_sample_output(samples)
    return samples, out


def run_custom(model_patcher, request: dict, uncond_patcher=None, progress_port=None,
               cancel_event=None, dp_cond_exempt_keys: frozenset[str] = frozenset()) -> tuple:
    """SamplerCustom path: guider spec + stock SAMPLER object + SIGMAS tensor."""
    import comfy.sample as comfy_sample
    import comfy.samplers  # noqa: F401  (sampler objects unpickle against this module)

    latent = request["latent"]
    full_latent = latent_outputs.fixed_empty_latent(model_patcher, latent)  # once, as in run_ksampler
    latent_batch = int(full_latent.shape[0])
    latent_image = split_batch_for_dp(full_latent)
    sigmas = request["sigmas"]
    schedule_steps = custom_schedule_steps(sigmas)
    if schedule_steps == 0:
        samples, denoised = latent_outputs.zero_step_outputs(
            latent_image, _normalize_sample_output
        )
        if not is_result_leader():
            return samples, None
        out = dict(latent)
        out.pop("noise_mask", None)
        out["samples"] = _normalize_sample_output(samples)
        out["denoised"] = denoised
        return samples, out

    seed = int(request["noise_seed"])
    noise_object = request.get("noise_object")
    if noise_object is not None:
        # The stock NOISE object generates against the fixed latent with its size tags, as stock
        # does. Its seed gives every rank the same full-batch noise; each then takes its dp slice.
        fixed = dict(latent)
        fixed["samples"] = full_latent
        noise = split_batch_for_dp(noise_object.generate_noise(fixed))
    elif request.get("add_noise", True):
        noise = comfy_sample.prepare_noise(full_latent, seed, latent.get("batch_index"))
        noise = split_batch_for_dp(noise)
    else:
        noise = latent_outputs.zeros_like_latent(latent_image, "custom sampler latent")

    noise_mask = split_mask_for_dp(latent.get("noise_mask"), int(latent["samples"].shape[0]))

    guider_spec = split_guider_for_dp(request["guider"], latent_batch, dp_cond_exempt_keys)
    guider = build_guider(model_patcher, guider_spec, uncond_patcher)
    sampler = request["sampler_object"]

    total_steps = schedule_steps
    leader = is_result_leader()
    callback, finish = make_progress_callback(
        progress_port if leader else None, total_steps, cancel_event)

    x0_holder: dict[str, Any] = {}

    def wrapped_callback(step, x0, x, total):
        x0_holder["x0"] = x0
        if callback is not None:
            callback(step, x0, x, total)

    try:
        samples = guider.sample(
            noise, latent_image, sampler, sigmas,
            denoise_mask=noise_mask, callback=wrapped_callback,
            disable_pbar=True, seed=seed,
        )
    finally:
        finish()

    latent_outputs.validate_tensor_tree(
        samples, "custom sampler output", require_finite=True
    )
    if not leader:
        return samples, None
    out = dict(latent)
    out.pop("noise_mask", None)
    out["samples"] = _normalize_sample_output(samples)
    if "x0" in x0_holder:
        try:
            out["denoised"] = _custom_denoised_output(
                guider, samples, x0_holder["x0"]
            )
        except Exception as exc:
            if latent_outputs.is_direct_nested_tensor(samples):
                raise RuntimeError(
                    "custom sampler could not reconstruct packed denoised output"
                ) from exc
            safe_call(log.debug, "denoised preview extraction failed: %s", failure_summary(exc))
    elif latent_outputs.is_direct_nested_tensor(samples):
        raise RuntimeError(
            "custom sampler returned packed samples without a denoised x0 callback"
        )
    return samples, out
