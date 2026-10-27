#!/usr/bin/env python3
"""Behavioral contracts for the ComfyUI interfaces used by dgx-monarch.

Import checks miss behavior changes behind stable signatures (docs/DESIGN.md
5.10). These fixtures exercise installed ComfyUI types and functions, assert
their results, and call the consuming production helper where one exists.
Each failure names the affected interface.

The seams, in run order:

  1. sampler callback         which x0/x shape a callback gets per latent kind
  2. packed latent layout     pack_latents/unpack_latents, and the width the
                              worker recomputes on its own
  3. latent format            process_latent_in/process_latent_out types
  4. empty latent channels    fix_empty_latent_channels reshaping rules
  5. noise                    seed determinism across ranks, NOISE objects
  6. conditioning wire        the keys a cond dict must survive the wire with
  7. guider and sampler       the objects the actors build and unpickle
  8. load path                prefix strip, state-dict conversion, assign
  9. batch cycling            cycle-or-truncate, the rule the dp splitter mirrors
 10. keyed wrapper            the keyed DIFFUSION_MODEL wrapper cfg-parallel
                              installs, and the executor it hands a family
 11. cfg split                which comfy seam each family's cfg-parallel
                              split rides, and the declaration held to it
 12. cond dispatch            the central CALC_COND_BATCH seam the per-cond
                              dispatch rides, and the output list it returns
 13. prepare sampling guard   the keyed PREPARE_SAMPLING wrapper the cross-rank
                              partial-load guard and the dual-model hand-off
                              ride, what the load call it fires after is built
                              from, and the flag it reads the verdict off
 14. minimax h3 audio carry   stock outer forward unscales the carried audio
                              before the rebound inner forward and converts its
                              raw velocity exactly once on return
 15. ltx rope payload         legacy frequency pair and current rotation-matrix
                              pair shape, split marker, and token axis
 16. ideogram4 rope payload   current frequency tuple conversion to the
                              rotation-matrix form the adapter shards
 17. ltx metadata config      the checkpoint __metadata__ fields detection
                              merges into dit_config, and the keyframe probe
 18. ltx stg passthrough      the flagged self-attention short circuit, and the
                              adapter replacement that has to reproduce it
 19. ltx keyframe embedding   the keyframe marker mask, and the pre-shard point
                              the marker is applied at
 20. ltx guide attenuation    which guide settings build a self-attention bias,
                              the predicate the adapter reads before biasing
 21. ltx guide bias layout    the two rectangles that bias carries, and the
                              query split comfy applies them with
 22. minimax h3 guide layout  the segment kinds an anchored guide packs, and
                              the guide rows the row accounting charges for it
 23. minimax h3 guide inputs  the pixel and audio input names the loader-side
                              wall prices an anchored guide's encode by
 24. minimax h3 latent mask   the two latent noise-mask parameters the packed
                              forward takes, and the refusal a set mask meets
 25. minimax h3 pdd head      the sampler sigma, schedule and flow shifts a
                              stacked PDD output head blends its row blocks by,
                              and the places the rebound forward reads them from
 26. nvfp4 activation scale   every quantization format classified in the pack's
                              activation-scale table, the whole-tensor fallback
                              nvfp4 alone takes, the per-module scales the
                              quantized Linear reads, and the fused INT8 SiTU
                              input that never calls the module while nvfp4
                              falls back to it
 27. chroma text stream      the row count chroma's text encoder hands the
                              model, the floor it pads to, and the one route
                              an attention mask can reach its forward by
 28. cfg combine             the order a folded cond/uncond pair goes in, and
                              the combine the benchmark matrix scorer copies
                              when it reduces such a pair
 29. loader model folders    the models subfolder each stock loader resolves
                              its widget against, which is the dest column of
                              the per-template artifact manifest, and the one
                              VAE dropdown entry that names no file
 30. fsdp shard size          the two numbers comfy sizes a load by, the cached
                              patcher size and the per-module byte sum, and the
                              rule that decides whether the block loop offloads
 31. resident ledger         what a build-time full load writes into comfy's
                              weight ledger, the early return a later load
                              takes off it, and the pending patch stack that
                              does not take it
 32. offload pin              that clone() carries a pinned offload device to
                              every hot-swap patcher, that a patch-uuid
                              mismatch and a detach both move the model to
                              exactly it, and that a matching uuid moves nothing
 33. outside forward modules  which supported families reach a diffusion-model
                              method from extra_conds, and the modules that
                              method runs, which FSDP has to replicate, or the
                              method an input refused off one GPU alone reaches
 34. flux family text         the widths flux2 and longcat hand the model, and
                              that no text row either of them reads is masked
 35. cogvideox layout         the key the CogVideoX config is detected from, the
                              1.5 image-to-video shape it reads, and the
                              diffusers spelling comfy answers nothing for
 36. media load inputs        the input each stock media loader names its file
                              on, that the options come from the input folder,
                              and that a name the folder does not hold is
                              refused rather than loaded
 37. waiver history           current sampler dispatch IDs survive Comfy output
                              parsing and stay in the owning prompt's history
 38. lumina attention container  NextDiT._forward's ming-image infix admits
                              both trees, and once AttentionTensorContainer
                              exists, usp_options' override still receives
                              plain tensors through comfy's real wrap_attn
 39. attention keyword handoff  a pack op installed on a cosmos Attention
                              the way Anima installs its attn_op takes the
                              q/k/v containers and preferred_attention comfy
                              hands it and matches stock output, a real
                              comfy CrossAttention hands an
                              optimized_attention_override exactly func, q, k,
                              v, heads, attn_precision, transformer_options and
                              _inside_attn_wrapper, the Qwen Image 2.1
                              attention and Wan-Animate 2 attention never run
                              a kernel a block pins, which stock runs, and
                              neither does the stock cross-attention and
                              refiner attention the sequence-parallel
                              forwards leave in place; the pin detector reads
                              a real checkpoint's and a real LoRA's pins, and
                              the probe tool's key patterns find each
                              family's pinnable attention, MOSS-Image's
                              block and text pins included
 40. cfg++ uncond pass        every sampler the auto table's cfg++ rule names
                              sets disable_cfg1_optimization before its first
                              model call, the CFG 1.0 shortcut still reads
                              that flag, and a plain sampler does not set it
 41. flux3 conds branch       stock FLUX.3 caches the early text and reference
                              states from extra_conds only under inference mode,
                              with no hooks, a cross_attn and no flow map; the
                              forward takes both input sets alike, and the bound
                              extra_conds takes the cache off inference mode and
                              leaves no inference tensor in its output
 42. moss_image conds         CONDRegular text that folds only at equal length,
                              no constant, no mask, no cond branch, rope.bands
                              popped, flow at multiplier 1.0 and shift 5.0, the
                              Flux2 latent, and the tokenizer's 10 and 23 rows;
                              since 60aade0bb, the reference keys the edit node
                              sets, the latents extra_conds hands the forward
                              outside inference mode, and the keys and values
                              it caches inside it; since 9fe7ef764, the text
                              bias the node sets, published as CONDRegular
 43. moss_image block surface  the two fused-norm calls a block makes, directly
                              or through modulate, which equals the norm with
                              no reference, the rotary table, the [image, text]
                              join, the block keys, the replace call, the
                              malloc graph and the attention call shape the
                              sharded forward re-runs; since 9fe7ef764, the
                              text bias as a key mask, where all zeros change
                              no bit
 44. skip layer guidance simple  the stock node's widgets, and the two
                              single-cond calls with an identity skip it makes
                              inside its window, one call outside or at CFG 1
 45. lora shape-changing set  the shape test the LoRA shape guard runs, and the
                              control LoRA that widens video_embedder
 46. dynamic prompt asymmetry  CLIP Text Encode's text declares dynamicPrompts and
                              Text (Multiline)'s value does not, which the
                              MOSS-Image layout graph's JSON prompt relies on
 47. weight-only linear dispatch  the formats that quantize no activation and
                              the layouts they load as, the 2-D reshape the
                              quantized Linear skips for them, and outside
                              inference mode the INT8 GEMM through mm for a
                              3-D activation whose strides chain, the
                              dequantizing expand for one whose strides do
                              not, and mm for both behind the pack's wrapper

Runs in the comfy-canary workflow against comfy master, and as a pytest
wherever comfy is importable (COMFY_DIR overrides the location).
"""

from __future__ import annotations

import inspect
import math
import os
import pickle
import sys
from collections.abc import Callable
from pathlib import Path


def _ensure_comfy() -> None:
    comfy_dir = os.path.abspath(os.environ.get("COMFY_DIR", "../ComfyUI"))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    sys.argv = ["comfy-seam-contracts", "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()


def _ensure_repo() -> None:
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    src = os.path.join(root, "src")
    if os.path.isdir(src) and src not in sys.path:
        sys.path.insert(0, src)
    # The sweep harness is not installed; the media and waiver history seams
    # import it straight out of the checkout.
    root = os.path.abspath(root)
    if root not in sys.path:
        sys.path.insert(0, root)


# One conditioning-free carrier reused by the seams that need a model object.
# Comfy's own BaseModel methods are bound onto it, so the arithmetic under test
# is comfy's, not a re-implementation: constructing a real BaseModel needs a
# checkpoint and a full model_config, which no CPU canary can supply.
def _latent_model(latent_format):
    import comfy.model_base
    import torch

    class LatentModel(torch.nn.Module):
        process_latent_in = comfy.model_base.BaseModel.process_latent_in
        process_latent_out = comfy.model_base.BaseModel.process_latent_out

        def __init__(self) -> None:
            super().__init__()
            self.latent_format = latent_format

    return LatentModel()


def _patcher(model):
    import torch
    from comfy.model_patcher import ModelPatcher

    cpu = torch.device("cpu")
    return ModelPatcher(model, cpu, cpu)


def _av_modalities():
    """A two-modality latent in the shape a packed audio/video render carries."""
    import torch

    video = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    audio = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2) * 0.5
    return video, audio


def _fake_guider(model):
    """The two attributes `custom_denoised_output` reads off a guider."""
    from types import SimpleNamespace

    return SimpleNamespace(model_patcher=SimpleNamespace(model=model))


def _assert_sampler_callback_seam() -> None:
    """What CFGGuider.sample hands a callback, per latent kind.

    Comfy 49a7422 (#15196) started re-nesting the callback x0 for every
    multi-modality latent, so the only x0 shape a packed render still got was
    the one shape the worker refused (2026-08-05, docs/TROUBLESHOOTING.md #64).
    The transaction below stands a stub `outer_sample` in for the model, which
    leaves comfy's real packing, callback wrapping and output re-nesting as the
    only code under test, then feeds what the callback received to the
    production rebuild.
    """
    import comfy.latent_formats
    import comfy.nested_tensor
    import comfy.samplers
    import torch

    from dgx_monarch.actor.latent_outputs import (
        custom_denoised_output,
        normalize_tensor_tree,
    )
    from dgx_monarch.sampling_contract import custom_schedule_steps

    inner: dict[str, object] = {}
    seen: dict[str, object] = {}

    class StubOuterGuider(comfy.samplers.CFGGuider):
        """A guider whose model is replaced, and whose comfy half is real."""

        def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
                         callback=None, disable_pbar=False, seed=None, latent_shapes=None):
            inner["noise"] = noise
            inner["latent"] = latent_image
            inner["latent_shapes"] = latent_shapes
            if callback is not None:
                callback(0, latent_image + 1, latent_image, 1)
            return latent_image

    def record(step, x0, x, total):
        seen["x0"] = x0
        seen["x"] = x

    latent_format = comfy.latent_formats.SD3()
    guider = StubOuterGuider(_patcher(_latent_model(latent_format)))
    guider.inner_set_conds({})

    video, audio = _av_modalities()
    nested = comfy.nested_tensor.NestedTensor([video, audio])
    nested_noise = comfy.nested_tensor.NestedTensor(
        [torch.zeros_like(video), torch.zeros_like(audio)]
    )
    sigmas = torch.tensor([1.0, 0.0])

    samples = guider.sample(nested_noise, nested, object(), sigmas, callback=record)

    packed_width = sum(math.prod(part.shape[1:]) for part in (video, audio))
    assert type(inner["noise"]) is torch.Tensor, (
        "sampler callback seam: comfy stopped packing nested noise into one flat tensor"
    )
    assert tuple(inner["latent"].shape) == (1, 1, packed_width), (
        "sampler callback seam: the flat pack the sampler runs on changed shape; "
        f"got {tuple(inner['latent'].shape)}, expected {(1, 1, packed_width)}"
    )
    assert [tuple(shape) for shape in inner["latent_shapes"]] == [
        tuple(video.shape), tuple(audio.shape)
    ], "sampler callback seam: latent_shapes no longer carries the modality order"

    for slot in ("x0", "x"):
        value = seen[slot]
        assert type(value) is comfy.nested_tensor.NestedTensor, (
            f"sampler callback seam: comfy handed a packed render {slot} as "
            f"{type(value).__name__}; the worker rebuilds the denoised output from it"
        )
        assert [tuple(part.shape) for part in value.unbind()] == [
            tuple(video.shape), tuple(audio.shape)
        ], f"sampler callback seam: callback {slot} lost the sampled modality shapes"

    assert type(samples) is comfy.nested_tensor.NestedTensor, (
        "sampler callback seam: CFGGuider.sample stopped re-nesting its packed output"
    )

    # The production rebuild, fed the x0 comfy handed the callback.
    denoised = custom_denoised_output(guider, samples, seen["x0"], normalize_tensor_tree)
    expected = [
        (part + 1) / latent_format.scale_factor + latent_format.shift_factor
        for part in (video, audio)
    ]
    assert [tuple(part.shape) for part in denoised.unbind()] == [
        tuple(video.shape), tuple(audio.shape)
    ], "sampler callback seam: the rebuilt denoised output lost a modality shape"
    assert all(
        torch.equal(rebuilt, want)
        for rebuilt, want in zip(denoised.unbind(), expected, strict=True)
    ), "sampler callback seam: process_latent_out arithmetic changed under the rebuild"

    # Single-modality latents keep the flat callback contract, which is the
    # other branch of the same production function.
    seen.clear()
    flat = video.clone()
    flat_samples = guider.sample(
        torch.zeros_like(flat), flat, object(), sigmas, callback=record
    )
    assert type(seen["x0"]) is torch.Tensor, (
        "sampler callback seam: comfy started wrapping single-modality callback x0"
    )
    assert type(flat_samples) is torch.Tensor, (
        "sampler callback seam: comfy started nesting single-modality sampler output"
    )

    # Where the two zero-step guards meet: comfy short-circuits an empty sigma
    # vector, and the worker refuses that vector, so nothing can fall between
    # them. A one-element schedule is the real zero-step render.
    empty_sigmas = torch.zeros(0)
    assert guider.sample(torch.zeros_like(flat), flat, object(), empty_sigmas) is flat, (
        "sampler callback seam: comfy stopped returning the input latent for an "
        "empty sigma vector"
    )
    try:
        custom_schedule_steps(empty_sigmas)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "sampler callback seam: the worker accepted an empty sigma vector"
        )
    assert custom_schedule_steps(torch.tensor([1.0])) == 0


def _assert_packed_latent_layout_seam() -> None:
    """The flat pack layout the worker recomputes without asking comfy.

    `latent_outputs._denoised_from_packed_x0` derives the packed width from the
    sampled modality shapes and rejects anything else, so comfy's packing
    arithmetic is a contract, not an implementation detail.
    """
    import comfy.latent_formats
    import comfy.nested_tensor
    import comfy.utils
    import torch

    from dgx_monarch.actor.latent_outputs import (
        custom_denoised_output,
        normalize_tensor_tree,
    )

    video, audio = _av_modalities()
    packed, latent_shapes = comfy.utils.pack_latents([video, audio])
    width = sum(math.prod(part.shape[1:]) for part in (video, audio))
    assert tuple(packed.shape) == (1, 1, width), (
        f"packed latent layout seam: pack_latents returned {tuple(packed.shape)}, "
        f"expected {(1, 1, width)}"
    )
    assert [tuple(shape) for shape in latent_shapes] == [
        tuple(video.shape), tuple(audio.shape)
    ], "packed latent layout seam: pack_latents reordered or reshaped the modalities"
    assert torch.equal(
        packed, torch.cat([video.reshape(1, 1, -1), audio.reshape(1, 1, -1)], dim=-1)
    ), "packed latent layout seam: modalities are no longer concatenated in order"

    unpacked = comfy.utils.unpack_latents(packed, latent_shapes)
    assert type(unpacked) is list, (
        "packed latent layout seam: unpack_latents stopped returning a plain list"
    )
    assert all(
        torch.equal(part, original)
        for part, original in zip(unpacked, (video, audio), strict=True)
    ), "packed latent layout seam: the pack/unpack round trip is no longer exact"

    # One shape means "already flat", so the combined tensor comes back
    # unreshaped. The worker's shape check fails closed on that instead of
    # passing on a mangled latent.
    single = comfy.utils.unpack_latents(packed, latent_shapes[:1])
    assert len(single) == 1 and single[0] is packed, (
        "packed latent layout seam: single-shape unpack_latents changed behavior"
    )

    # The flat branch of the production rebuild, fed the pack comfy produced.
    latent_format = comfy.latent_formats.SD3()
    guider = _fake_guider(_latent_model(latent_format))
    nested = comfy.nested_tensor.NestedTensor([video, audio])
    denoised = custom_denoised_output(guider, nested, packed, normalize_tensor_tree)
    expected = [
        part / latent_format.scale_factor + latent_format.shift_factor
        for part in (video, audio)
    ]
    assert all(
        torch.equal(rebuilt, want)
        for rebuilt, want in zip(denoised.unbind(), expected, strict=True)
    ), "packed latent layout seam: the flat-x0 rebuild no longer matches the pack"


def _assert_latent_format_seam() -> None:
    """process_latent_in / process_latent_out input and output types."""
    import comfy.latent_formats
    import comfy.nested_tensor
    import torch

    latent_format = comfy.latent_formats.SD3()
    model = _latent_model(latent_format)
    flat = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)

    encoded = model.process_latent_in(flat)
    assert type(encoded) is torch.Tensor and encoded.shape == flat.shape, (
        "latent format seam: process_latent_in changed its output type or shape"
    )
    assert torch.equal(
        encoded, (flat - latent_format.shift_factor) * latent_format.scale_factor
    ), "latent format seam: process_latent_in arithmetic changed"
    decoded = model.process_latent_out(encoded)
    assert torch.allclose(decoded, flat, atol=1e-5), (
        "latent format seam: process_latent_out is no longer the inverse of process_latent_in"
    )

    # The packed families hand the same methods a NestedTensor. Structure,
    # dtype and per-modality independence are the contract the worker's
    # shape/dtype match then enforces.
    av_format = comfy.latent_formats.MiniMaxH3AV()
    av_model = _latent_model(av_format)
    video, audio = _av_modalities()
    nested = comfy.nested_tensor.NestedTensor([video, audio])
    out = av_model.process_latent_out(nested)
    assert type(out) is comfy.nested_tensor.NestedTensor and out is not nested, (
        "latent format seam: process_latent_out no longer returns a fresh NestedTensor"
    )
    parts = out.unbind()
    assert [tuple(part.shape) for part in parts] == [
        tuple(video.shape), tuple(audio.shape)
    ], "latent format seam: process_latent_out changed a packed modality shape"
    assert [part.dtype for part in parts] == [video.dtype, audio.dtype], (
        "latent format seam: process_latent_out changed a packed modality dtype"
    )
    assert all(type(part) is torch.Tensor for part in parts), (
        "latent format seam: packed modalities came back as Tensor subclasses"
    )


def _assert_empty_latent_channel_seam() -> None:
    """fix_empty_latent_channels: the reshape both sampler paths run first.

    The worker reads the batch and the modality shapes off the result, so a
    change in when comfy resizes or unsqueezes moves every downstream split.

    Comfy 0696f61 gave LatentFormat a `fix_empty_latent` hook and let H3
    override it, so on H3 an empty plain latent leaves this call already
    packed as an audio-video pair. Each tree is pinned to its own result
    rather than to whichever one runs, so a third rule breaks the seam here.
    """
    import comfy.latent_formats
    import comfy.nested_tensor
    import comfy.sample
    import torch

    image = _patcher(_latent_model(comfy.latent_formats.SD3()))
    empty = torch.zeros(1, 4, 8, 8)
    fixed = comfy.sample.fix_empty_latent_channels(image, empty)
    assert tuple(fixed.shape) == (1, 16, 8, 8), (
        "empty latent channels seam: an all-zero latent is no longer widened to the "
        f"format's channel count; got {tuple(fixed.shape)}"
    )
    populated = torch.ones(1, 4, 8, 8)
    assert comfy.sample.fix_empty_latent_channels(image, populated) is populated, (
        "empty latent channels seam: a populated latent is no longer passed through"
    )

    video = _patcher(_latent_model(comfy.latent_formats.MiniMaxH3AV()))
    volume = comfy.sample.fix_empty_latent_channels(video, torch.zeros(1, 4, 8, 8))
    if hasattr(comfy.latent_formats.LatentFormat, "fix_empty_latent"):
        assert volume.is_nested, (
            "empty latent channels seam: the format hook no longer packs an empty H3 "
            f"latent into an audio-video pair; got {type(volume).__name__}"
        )
        assert [tuple(part.shape) for part in volume.unbind()] == [
            (1, 24, 1, 8, 8), (1, 32, 2, 2)
        ], (
            "empty latent channels seam: the packed pair an empty H3 latent expands to "
            f"changed; got {[tuple(part.shape) for part in volume.unbind()]}"
        )
    else:
        assert tuple(volume.shape) == (1, 32, 1, 8, 8), (
            "empty latent channels seam: a 3-dimension format no longer unsqueezes a "
            f"4-dimension latent; got {tuple(volume.shape)}"
        )

    modalities = comfy.nested_tensor.NestedTensor(list(_av_modalities()))
    assert comfy.sample.fix_empty_latent_channels(video, modalities) is modalities, (
        "empty latent channels seam: packed latents are no longer left alone"
    )


def _assert_noise_seam() -> None:
    """Seed determinism, batch_index, and the NOISE objects that ride the wire.

    Every rank generates its own noise from the same seed, so stock's
    generator order is the cross-rank identity guarantee.
    """
    import comfy.nested_tensor
    import comfy.sample
    import torch
    from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise

    from dgx_monarch.actor.latent_outputs import zeros_like_latent

    latent = torch.zeros(2, 4, 8, 8)
    first = comfy.sample.prepare_noise(latent, 7)
    assert torch.equal(first, comfy.sample.prepare_noise(latent, 7)), (
        "noise seam: prepare_noise is no longer deterministic for one seed; every "
        "rank would sample different noise"
    )
    assert not torch.equal(first, comfy.sample.prepare_noise(latent, 8)), (
        "noise seam: prepare_noise ignores the seed"
    )
    assert tuple(first.shape) == tuple(latent.shape) and first.dtype == latent.dtype

    indexed = comfy.sample.prepare_noise(latent, 7, [0, 0])
    assert torch.equal(indexed[0], indexed[1]), (
        "noise seam: batch_index no longer selects one noise row per index"
    )

    packed = comfy.nested_tensor.NestedTensor(list(_av_modalities()))
    nested_noise = comfy.sample.prepare_noise(packed, 7)
    assert type(nested_noise) is comfy.nested_tensor.NestedTensor, (
        "noise seam: prepare_noise stopped returning a NestedTensor for packed latents"
    )
    assert [tuple(part.shape) for part in nested_noise.unbind()] == [
        tuple(part.shape) for part in packed.unbind()
    ], "noise seam: packed noise lost a modality shape"

    assert torch.equal(
        Noise_RandomNoise(7).generate_noise({"samples": latent}), first
    ), (
        "noise seam: the stock NOISE object and prepare_noise disagree; the object "
        "rides to the workers and must reproduce the driver's schedule"
    )
    assert torch.equal(
        Noise_RandomNoise(7).generate_noise({"samples": latent, "batch_index": [0, 0]}),
        indexed,
    ), "noise seam: the stock NOISE object stopped honoring batch_index"

    empty = Noise_EmptyNoise().generate_noise({"samples": latent})
    assert torch.equal(empty, zeros_like_latent(latent, "noise seam latent")), (
        "noise seam: the worker's disabled-noise latent no longer matches stock's"
    )


def _assert_conditioning_wire_seam() -> None:
    """The cond keys the driver must move to the workers intact.

    `transfer.pack_conditioning` rebuilds a cond tree by structure with every
    tensor moved to CPU, so what comfy later reads off that tree is the real
    contract: an entry is `[tensor|None, dict]`, comfy binds the tensor to
    `cross_attn`, and every other key in the dict (pooled_output, the family
    extras) survives untouched.
    """
    import comfy.sampler_helpers
    import comfy.samplers
    import torch

    from dgx_monarch.transfer import pack_conditioning

    cross_attn = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    pooled = torch.ones(1, 4)
    # LTX image conditioning puts three more shapes on this wire. keyframe_idxs
    # is a plain tensor, generated_keyframes a dict of ints, and
    # guide_attention_entries a list of dicts mixing a tensor, a None and a
    # tuple: one entry per LTXV Add Guide, with pixel_mask None whenever the
    # guide carries no spatial mask. A packer that grew a key allowlist, or that
    # stopped recursing into lists, tuples or None, would drop a guide's
    # position and silently render the wrong thing.
    guide_entries = [
        {"pre_filter_count": 880, "strength": 1.0,
         "pixel_mask": None, "latent_shape": (1, 22, 40)},
        {"pre_filter_count": 880, "strength": 1.0,
         "pixel_mask": torch.ones(1, 1, 1, 2, 2), "latent_shape": (1, 22, 40)},
    ]
    extras = {
        "minimax_refs": [torch.zeros(1, 2)],
        "some_scalar": 3,
        "keyframe_idxs": torch.zeros(1, 3, 4, 2),
        "guide_attention_entries": guide_entries,
        "generated_keyframes": {"tokens_per_frame": 880, "first_latent_frame": 0,
                                "num_keyframes": 1},
    }
    conds = [[cross_attn, {"pooled_output": pooled, **extras}]]

    wire = pack_conditioning(conds)
    assert type(wire) is list and type(wire[0]) is list and len(wire[0]) == 2, (
        "conditioning wire seam: the packed cond lost the [tensor, dict] entry shape"
    )
    assert set(wire[0][1]) == {"pooled_output", *extras}, (
        "conditioning wire seam: packing dropped or renamed a cond key"
    )
    wired_entries = wire[0][1]["guide_attention_entries"]
    assert type(wired_entries) is list and len(wired_entries) == len(guide_entries), (
        "conditioning wire seam: packing changed the guide entry list"
    )
    assert wired_entries[0]["pixel_mask"] is None, (
        "conditioning wire seam: packing no longer carries an absent guide mask"
    )
    assert torch.equal(wired_entries[1]["pixel_mask"], guide_entries[1]["pixel_mask"]), (
        "conditioning wire seam: packing changed a guide mask's values"
    )
    assert wired_entries[0]["latent_shape"] == (1, 22, 40), (
        "conditioning wire seam: packing changed a guide's latent shape tuple"
    )
    assert wire[0][1]["generated_keyframes"] == extras["generated_keyframes"], (
        "conditioning wire seam: packing changed the generated keyframe record"
    )
    assert torch.equal(wire[0][0], cross_attn) and torch.equal(
        wire[0][1]["pooled_output"], pooled
    ), "conditioning wire seam: packing changed a cond tensor's values"

    converted = comfy.sampler_helpers.convert_cond(wire)
    assert converted[0]["cross_attn"] is wire[0][0], (
        "conditioning wire seam: comfy no longer binds the entry tensor to cross_attn"
    )
    for key in ("pooled_output", "minimax_refs", "some_scalar",
                "keyframe_idxs", "guide_attention_entries", "generated_keyframes"):
        assert key in converted[0], (
            f"conditioning wire seam: comfy's cond conversion dropped {key!r}"
        )
    for key in ("model_conds", "uuid"):
        assert key in converted[0], (
            f"conditioning wire seam: comfy's cond conversion stopped adding {key!r}"
        )

    guider = comfy.samplers.CFGGuider(_patcher(torch.nn.Linear(1, 1)))
    guider.set_conds(wire, wire)
    guider.set_cfg(4.0)
    assert set(guider.original_conds) == {"positive", "negative"}, (
        "conditioning wire seam: CFGGuider.set_conds changed its slot names"
    )
    assert guider.original_conds["positive"][0]["cross_attn"] is wire[0][0]
    assert guider.cfg == 4.0

    # An image-only uncond entry carries no tensor at all, which is the shape
    # the dual-model guider builds when no negative is wired.
    empty = comfy.sampler_helpers.convert_cond([[None, {}]])
    assert "cross_attn" not in empty[0], (
        "conditioning wire seam: comfy now invents a cross_attn for a text-free cond"
    )


def _assert_guider_and_sampler_seam() -> None:
    """The guider and SAMPLER objects the actors build, unpickle, and call."""
    import inspect

    import comfy.samplers
    import torch
    from comfy.model_sampling import ModelSamplingDiscreteFlow
    from comfy_extras.nodes_custom_sampler import Guider_DualModel

    from dgx_monarch.actor.sampling import build_guider
    from dgx_monarch.actor.sampling_validation import require_known_sampler
    from dgx_monarch.sampling_contract import custom_schedule_steps

    patcher = _patcher(torch.nn.Linear(1, 1))
    uncond_patcher = _patcher(torch.nn.Linear(1, 1))
    conds = [[torch.zeros(1, 2, 3), {"pooled_output": torch.ones(1, 4)}]]

    basic = build_guider(patcher, {"kind": "basic", "positive": conds})
    assert isinstance(basic, comfy.samplers.CFGGuider), (
        "guider seam: the basic guider is no longer a CFGGuider subclass"
    )
    assert set(basic.original_conds) == {"positive"}

    cfg = build_guider(
        patcher, {"kind": "cfg", "positive": conds, "negative": conds, "cfg": 3.5}
    )
    assert type(cfg) is comfy.samplers.CFGGuider and cfg.cfg == 3.5
    assert set(cfg.original_conds) == {"positive", "negative"}

    dual = build_guider(
        patcher,
        {"kind": "dual_model", "positive": conds, "negative": None, "cfg": 2.0},
        uncond_patcher,
    )
    # Stage a registration for each stock prepare call so load_models_gpu
    # loads both checkpoints and DISABLE_SMART_MEMORY evicts neither. The cond
    # prepare keeps uncond resident for its forward; the uncond prepare keeps
    # cond resident across warm renders. Never hold both registrations at once:
    # patcher.clone() has no cycle guard, as the probe below demonstrates.
    assert isinstance(dual, Guider_DualModel) and dual.cfg == 2.0, (
        "guider seam: the dual-model guider changed type or lost its cfg slot"
    )
    assert type(dual).__mro__[1] is Guider_DualModel, (
        "guider seam: the both-resident guider must subclass the stock dual "
        "guider directly so every stock semantic carries"
    )
    assert dual.uncond_model_patcher is uncond_patcher, (
        "guider seam: the dual guider lost its uncond patcher slot"
    )
    assert callable(getattr(patcher, "set_additional_models", None)) and callable(
        getattr(patcher, "remove_additional_models", None)
    ), (
        "guider seam: ModelPatcher lost the additional-models API the "
        "both-resident bracket rides (issues #301, #314)"
    )
    assert "cross_attn" not in dual.original_conds["negative"][0], (
        "guider seam: the text-free uncond entry the dual-model path builds when no "
        "negative is wired gained a cross_attn"
    )

    # Where the hand-off rides, read off stock's own source. Stock prepares
    # the uncond model in its own prepare_sampling call, driven by the uncond
    # patcher's own model options, and only then hands the cond prepare to
    # CFGGuider.outer_sample. The bracket installs its keyed wrapper on that
    # dict, so the wrapper fires between the two prepares and on no other
    # load. A drift in any clause below is silent: the wrapper stops firing,
    # the cond prepare loses the uncond, and the uncond forward reads CPU
    # weights.
    dual_source = inspect.getsource(Guider_DualModel.outer_sample)
    assert "self.uncond_model_patcher.model_options" in dual_source, (
        "guider seam: the uncond prepare no longer reads the uncond patcher's own "
        "model options, so the dual-model hand-off never fires and the cond prepare "
        "evicts the uncond again (issues #301, #314)"
    )
    assert dual_source.index("self.uncond_model_patcher.model_options") < dual_source.index(
        "super().outer_sample"
    ), (
        "guider seam: the uncond prepare no longer runs before the cond prepare, so "
        "the hand-off no longer sits between the two (issues #301, #314)"
    )
    assert "math.isclose(self.cfg, 1.0)" in dual_source, (
        "guider seam: the uncond prepare lost its cfg 1.0 skip, which is the "
        "predicate the bracket reads to pick its one-way shape (issue #314)"
    )

    # The dm-cfg2 split guider (actor/dual_model_cfg.py) mirrors stock's own
    # dual-model predict_noise: one calc_cond_batch forward per model, then a
    # cfg_function combine. Pin that shape so a comfy change to the seam the
    # override copies fails in CI, not silently in a render.
    predict_source = inspect.getsource(Guider_DualModel.predict_noise)
    assert predict_source.count("calc_cond_batch") == 2, (
        "guider seam: stock Guider_DualModel.predict_noise no longer runs one "
        "calc_cond_batch forward per model; the dm-cfg2 split guider mirrors that "
        "shape (one local forward per rank), so verify its override still holds"
    )
    assert "cfg_function" in predict_source, (
        "guider seam: stock dual-model predict_noise stopped combining through "
        "cfg_function; the dm-cfg2 split guider combines the gathered predictions "
        "the same way and must be re-checked"
    )
    cfg_guider_sig = list(
        inspect.signature(comfy.samplers.CFGGuider.predict_noise).parameters)
    assert cfg_guider_sig == ["self", "x", "timestep", "model_options", "seed"], (
        "guider seam: CFGGuider.predict_noise changed its signature; the dm-cfg2 "
        "split guider overrides exactly (x, timestep, model_options, seed)"
    )
    combine_sig = list(inspect.signature(comfy.samplers.cfg_function).parameters)
    assert combine_sig[:6] == [
        "model", "cond_pred", "uncond_pred", "cond_scale", "x", "timestep"], (
        "guider seam: comfy.samplers.cfg_function changed its leading positional "
        "arguments; the dm-cfg2 split guider calls it (model, cond_pred, "
        "uncond_pred, cfg, x, timestep)"
    )

    # The hand-off reads the staged registration off the patcher with this
    # call, so it closes over nothing.
    assert callable(getattr(patcher, "get_additional_models_with_key", None)), (
        "guider seam: ModelPatcher lost the per-key read the hand-off uses to find "
        "what this render staged (issue #314)"
    )

    # The bracket's outer finally drops both keys whatever happened, so
    # removing a key that was never set has to stay free.
    patcher.remove_additional_models("dgxm_never_registered")

    # Why the bracket stages its two registrations instead of holding both:
    # clone() copies each additional model by cloning it and guards no cycle,
    # so a cond and an uncond registered on each other turn any clone taken
    # during a render into unbounded recursion. Unwinding that leaves one
    # half-built patcher for the collector, so this probe prints an "Exception
    # ignored in ModelPatcher.__del__" line: it is the defect being pinned,
    # not drift.
    first, second = _patcher(torch.nn.Linear(1, 1)), _patcher(torch.nn.Linear(1, 1))
    first.set_additional_models("dgxm_cycle_probe", [second])
    second.set_additional_models("dgxm_cycle_probe", [first])
    try:
        first.clone()
    except RecursionError:
        pass
    else:
        raise AssertionError(
            "guider seam: ModelPatcher.clone survived an additional-models cycle; "
            "the dual-model bracket stages its two registrations to keep one from "
            "ever existing, and that staging may no longer be load-bearing"
        )

    # A SAMPLER object crosses an actor RPC by value. Pickling must keep the
    # sampler function itself, not a look-alike.
    sampler = comfy.samplers.sampler_object("euler")
    restored = pickle.loads(pickle.dumps(sampler))  # noqa: S301  trusted local object
    assert type(restored) is type(sampler), (
        "guider seam: the stock SAMPLER object no longer survives a pickle round trip"
    )
    assert restored.sampler_function is sampler.sampler_function, (
        "guider seam: an unpickled SAMPLER carries a different sampler function"
    )

    # The registry the worker refuses unknown names against.
    require_known_sampler("euler", "normal")
    try:
        require_known_sampler("dgxm_no_such_sampler", "normal")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "guider seam: an unregistered sampler name was accepted; stock comfy "
            "would silently render euler instead"
        )

    from types import SimpleNamespace

    model_sampling = ModelSamplingDiscreteFlow(SimpleNamespace(sampling_settings={}))
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, "normal", 5)
    assert sigmas.ndim == 1 and sigmas.numel() == 6, (
        "guider seam: calculate_sigmas no longer returns steps+1 sigmas; "
        f"got {tuple(sigmas.shape)} for 5 steps"
    )
    assert custom_schedule_steps(sigmas) == 5, (
        "guider seam: the worker's step count and comfy's schedule length disagree"
    )
    assert float(sigmas[-1]) == 0.0 and bool((sigmas[:-1] > sigmas[1:]).all()), (
        "guider seam: the sigma schedule is no longer strictly decreasing to zero"
    )


def _assert_load_path_seam() -> None:
    """Prefix strip, state-dict conversion, and the assign the slab needs.

    `comfy_bridge.slab_load` records slab regions under the key comfy will use
    after it strips `unet_prefix`, and the zero-copy load depends on
    `assign=True` adopting the supplied storage rather than copying into it.
    Neither is provable from a signature.
    """
    import comfy.model_base
    import comfy.supported_models_base
    import torch

    def holder(model_config, module):
        return type(
            "LoadHolder", (), {"model_config": model_config, "diffusion_model": module}
        )()

    stock_config = comfy.supported_models_base.BASE({})
    state_dict = {"a": torch.zeros(1)}
    assert stock_config.process_unet_state_dict(state_dict) is state_dict, (
        "load path seam: the default state-dict conversion is no longer a pass-through"
    )

    adopted = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    linear = torch.nn.Linear(3, 2, bias=False)
    supplied = {"pfx.weight": adopted, "unrelated": torch.zeros(1)}
    comfy.model_base.BaseModel.load_model_weights(
        holder(stock_config, linear), supplied, unet_prefix="pfx.", assign=True
    )
    assert set(supplied) == {"unrelated"}, (
        "load path seam: load_model_weights no longer POPS prefixed keys out of the "
        "state dict it was handed"
    )
    assert linear.weight.data_ptr() == adopted.data_ptr(), (
        "load path seam: assign=True stopped adopting the supplied storage; the "
        "zero-copy slab load would silently become a full copy"
    )

    copied = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    copy_target = torch.nn.Linear(3, 2, bias=False)
    comfy.model_base.BaseModel.load_model_weights(
        holder(stock_config, copy_target), {"pfx.weight": copied},
        unet_prefix="pfx.", assign=False,
    )
    assert copy_target.weight.data_ptr() != copied.data_ptr(), (
        "load path seam: assign=False stopped copying, so the assign flag no longer "
        "decides residency"
    )
    assert torch.equal(copy_target.weight.data, copied)

    # The conversion hook runs between the prefix strip and the module load.
    # That ordering is why the slab records stripped keys.
    class RenamingConfig(comfy.supported_models_base.BASE):
        def process_unet_state_dict(self, state_dict):
            return {
                ("weight" if key == "renamed" else key): value
                for key, value in state_dict.items()
            }

    renamed = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    renamed_target = torch.nn.Linear(3, 2, bias=False)
    comfy.model_base.BaseModel.load_model_weights(
        holder(RenamingConfig({}), renamed_target), {"pfx.renamed": renamed},
        unet_prefix="pfx.", assign=True,
    )
    assert renamed_target.weight.data_ptr() == renamed.data_ptr(), (
        "load path seam: process_unet_state_dict no longer runs between the prefix "
        "strip and load_state_dict"
    )


def _assert_batch_cycling_seam() -> None:
    """Cycle-or-truncate, the rule the dp splitter reproduces by hand.

    A cond or a mask whose leading batch does not match the latent batch is
    cycled by `comfy.utils.repeat_to_batch_size`, reached from both sides the
    pack touches: `CONDRegular.process_cond` and `reshape_mask`. The dp splitter
    materializes that global cycled batch before taking its own slice, because
    cycling after the split restarts at row zero on every rank. A change to the
    cycle order therefore hands each rank rows stock never paired with its
    latents, with no error anywhere.
    """
    import comfy.conds
    import comfy.sampler_helpers
    import comfy.utils
    import torch

    from dgx_monarch.actor import sampling

    rows = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1)
    cycled = comfy.utils.repeat_to_batch_size(rows, 4)
    assert torch.equal(cycled, rows[[0, 1, 2, 0]]), (
        "batch cycling seam: repeat_to_batch_size no longer cycles rows from the top; "
        "every dp rank would hold conditioning stock never paired with its latents"
    )
    assert torch.equal(comfy.utils.repeat_to_batch_size(rows, 2), rows[:2]), (
        "batch cycling seam: an over-long batch is no longer truncated from the front"
    )
    assert comfy.utils.repeat_to_batch_size(rows, 3) is rows, (
        "batch cycling seam: an exact-length batch is no longer passed through untouched"
    )

    conded = comfy.conds.CONDRegular(rows).process_cond(batch_size=4)
    assert torch.equal(conded.cond, cycled), (
        "batch cycling seam: CONDRegular.process_cond stopped cycling a cond to the batch"
    )

    mask = comfy.sampler_helpers.prepare_mask(
        torch.stack([torch.zeros(8, 8), torch.ones(8, 8)]),
        (4, 16, 8, 8),
        torch.device("cpu"),
    )
    assert tuple(mask.shape) == (4, 16, 8, 8), (
        "batch cycling seam: prepare_mask no longer broadcasts a short mask to the "
        f"latent batch and channel count; got {tuple(mask.shape)}"
    )
    assert (
        torch.equal(mask[0], mask[2])
        and torch.equal(mask[1], mask[3])
        and not torch.equal(mask[0], mask[1])
    ), "batch cycling seam: a short noise mask is no longer cycled row by row over the batch"

    # The production splitters, driven at dp2 against the same cycled batch.
    conds = [[rows, {"pooled_output": rows.reshape(3, 1)}]]
    original_dp_info = sampling._dp_info
    try:
        sampling._dp_info = lambda: (0, 2)
        rank0 = sampling.split_conditioning_for_dp(conds, 4)
        sampling._dp_info = lambda: (1, 2)
        rank1 = sampling.split_conditioning_for_dp(conds, 4)
        rank1_mask = sampling.split_mask_for_dp(rows, 4)
    finally:
        sampling._dp_info = original_dp_info

    assert torch.equal(torch.cat([rank0[0][0], rank1[0][0]]), cycled), (
        "batch cycling seam: the dp ranks together no longer cover comfy's cycled batch"
    )
    assert torch.equal(rank1[0][0], cycled[2:]), (
        "batch cycling seam: a dp rank restarted the cycle at row zero instead of taking "
        "its slice of the global cycled batch; prompts and latents would misalign silently"
    )
    assert torch.equal(
        rank1[0][1]["pooled_output"],
        comfy.utils.repeat_to_batch_size(conds[0][1]["pooled_output"], 4)[2:],
    ), "batch cycling seam: a cond dict entry no longer follows the same cycle as its tensor"
    assert torch.equal(rank1_mask, cycled[2:]), (
        "batch cycling seam: the dp noise-mask slice no longer matches comfy's broadcast"
    )


def _assert_keyed_wrapper_seam() -> None:
    """The keyed DIFFUSION_MODEL wrapper the cfg-parallel dispatch rides.

    `actor/worker` installs one wrapper under the pack's key, and
    `adapters/cfg_parallel` binds the call by reading `executor.original`. If
    comfy stopped routing keyed wrappers into the all-wrappers view a family
    forward builds its executor from, every rank would compute the full batch
    and still return correct pixels, which no identity gate can see.
    """
    import importlib
    import inspect

    import comfy.patcher_extension as pe

    from dgx_monarch.constants import CFG_WRAPPER_KEY

    seen: list[object] = []

    def wrapper(executor, value):
        seen.append(executor.original)
        return executor(value) + 1

    def stock_forward(value):
        return value * 2

    def installed(options):
        return pe.get_wrappers_with_key(
            pe.WrappersMP.DIFFUSION_MODEL, CFG_WRAPPER_KEY, options, is_model_options=True
        )

    model_options: dict[str, object] = {}
    assert installed(model_options) == [], (
        "keyed wrapper seam: an empty model_options already reports a keyed wrapper"
    )
    pe.add_wrapper_with_key(
        pe.WrappersMP.DIFFUSION_MODEL, CFG_WRAPPER_KEY, wrapper,
        model_options, is_model_options=True,
    )
    assert installed(model_options) == [wrapper], (
        "keyed wrapper seam: a wrapper added under the pack's key does not read back under it"
    )

    transformer_options = model_options["transformer_options"]
    routed = pe.get_all_wrappers(pe.WrappersMP.DIFFUSION_MODEL, transformer_options)
    assert routed == [wrapper], (
        "keyed wrapper seam: a keyed wrapper is no longer visible to the all-wrappers view "
        "a family forward builds its executor from; cfg-parallel would go silently inert"
    )

    executor = pe.WrapperExecutor.new_class_executor(stock_forward, object(), routed)
    assert executor.execute(3) == 7, (
        "keyed wrapper seam: the wrapper no longer surrounds the call comfy routes through it"
    )
    assert seen == [stock_forward], (
        "keyed wrapper seam: executor.original no longer exposes the stock forward whose "
        "signature the cfg slice rule binds its arguments against"
    )

    # A second add appends rather than replacing, which is why the install site
    # reads the key before writing it.
    pe.add_wrapper_with_key(
        pe.WrappersMP.DIFFUSION_MODEL, CFG_WRAPPER_KEY, wrapper,
        model_options, is_model_options=True,
    )
    assert len(installed(model_options)) == 2, (
        "keyed wrapper seam: a second add under one key no longer appends, so the install "
        "site's read-before-write guard is holding something else"
    )

    # The forwards that route the wrapper. A real model needs a checkpoint, so
    # the routing itself is read off the live source.
    for module_name, class_name in (
        ("comfy.ldm.flux.model", "Flux"),
        ("comfy.ldm.minimax.model", "MiniMaxH3Model"),
    ):
        source = inspect.getsource(
            getattr(importlib.import_module(module_name), class_name).forward
        )
        assert "WrappersMP.DIFFUSION_MODEL" in source and "WrapperExecutor" in source, (
            f"keyed wrapper seam: {module_name}.{class_name}.forward no longer builds a "
            "wrapper executor over the DIFFUSION_MODEL wrappers"
        )


def _assert_cfg_split_seam() -> None:
    """Where each family's cfg-parallel split runs.

    Comfy applies a DIFFUSION_MODEL wrapper nowhere central: the family's own
    forward builds the executor over it, and three families build none, so a
    split installed there is inert and every rank renders the whole batch
    (2026-09-04). APPLY_MODEL is the seam comfy itself applies for every
    family, one call further out. Each adapter declares which one it rides;
    this walks the installed comfy and holds the declaration to it in both
    directions, so a family added on the wrong seam fails here instead of
    rendering twice on hardware.
    """
    import importlib
    import inspect
    import re

    import comfy.model_base

    from dgx_monarch.adapters import ADAPTERS
    from dgx_monarch.adapters.cfg_parallel import cfg_split_seam_type
    from dgx_monarch.comfy_rebound_sites import ADAPTER_BIND_SITES

    outer = inspect.getsource(comfy.model_base.BaseModel.apply_model)
    assert "WrappersMP.APPLY_MODEL" in outer and "WrapperExecutor" in outer, (
        "cfg split seam: BaseModel.apply_model no longer builds a wrapper executor "
        "over the APPLY_MODEL wrappers, so the seam three families ride is gone"
    )

    unet_model = re.compile(r"unet_model\s*=\s*([\w.]+)")

    def diffusion_model_paths(name: str) -> list[str]:
        """The comfy DiT class a model_base type constructs, off its own source."""
        klass = getattr(comfy.model_base, name, None)
        assert klass is not None, (
            f"cfg split seam: adapters name model_base.{name}, which comfy no longer has"
        )
        for entry in klass.__mro__:
            init = entry.__dict__.get("__init__")
            if init is None:
                continue
            found = [path for path in unet_model.findall(inspect.getsource(init))
                     if "." in path]
            if found:
                return found
        raise AssertionError(
            f"cfg split seam: no __init__ in model_base.{name}'s chain names a unet_model, "
            "so the class whose forward decides the seam cannot be read"
        )

    def routes_diffusion_model(path: str) -> bool:
        """Whether this DiT's forward reaches a DIFFUSION_MODEL executor.

        A subclass that overrides forward and delegates with ``super().forward``
        still reaches its parent's executor (Anima over MiniTrainDIT, LTXAV over
        LTXV); one that overrides without delegating shadows it.
        """
        module, _, name = path.rpartition(".")
        klass = getattr(importlib.import_module(module), name)
        for entry in klass.__mro__:
            forward = entry.__dict__.get("forward")
            if forward is None:
                continue
            source = inspect.getsource(forward)
            if "WrappersMP.DIFFUSION_MODEL" in source and "WrapperExecutor" in source:
                return True
            if "super().forward(" not in source:
                return False
        return False

    declared, checked = {}, 0
    for adapter in ADAPTERS:
        seam = getattr(adapter, "cfg_split_seam", "diffusion_model")
        declared[adapter.family] = seam
        names = tuple(adapter.model_base_classes) + tuple(
            getattr(adapter, "exact_model_base_classes", ()))
        for path in sorted({p for name in names for p in diffusion_model_paths(name)}):
            reached = "diffusion_model" if routes_diffusion_model(path) else "apply_model"
            assert reached == seam, (
                f"cfg split seam: {adapter.family} declares cfg_split_seam={seam!r} but "
                f"{path}.forward reaches the {reached!r} seam; a split on the wrong seam "
                "installs, never runs, and both ranks render the whole batch"
            )
            checked += 1
    assert checked >= len(ADAPTERS), (
        f"cfg split seam: only {checked} of {len(ADAPTERS)} families resolved a DiT class"
    )

    # A rebound instance forward replaces the class forward, so an adapter that
    # binds `forward` on a family whose stock forward carries the executor would
    # take the split out with it. The three that bind it ride APPLY_MODEL.
    shadowed = sorted({
        family for site in ADAPTER_BIND_SITES
        if site.receiver == "diffusion_model" and site.method == "forward"
        for family in site.families
        if declared.get(family, "diffusion_model") != "apply_model"
    })
    assert not shadowed, (
        f"cfg split seam: {shadowed} rebind diffusion_model.forward while declaring the "
        "diffusion_model seam, so the bound forward drops the executor the split rides"
    )

    types = {cfg_split_seam_type(adapter) for adapter in ADAPTERS}
    assert types == {"diffusion_model", "apply_model"}, (
        f"cfg split seam: the shipped families resolve to wrapper types {sorted(types)}; "
        "one of the two seams is no longer exercised by any family"
    )


def _assert_cond_dispatch_seam() -> None:
    """The central seam the per-cond dispatch rides, and what it returns.

    Comfy applies this seam itself for every family, in
    ``_calc_cond_batch_outer``, so unlike the two model-call seams no adapter
    declares it. The dispatch hands each cfg rank a contiguous group of the
    cond list and masks the rest to ``None``, so three stock properties have
    to hold: the executor wraps ``_calc_cond_batch`` over the CALC_COND_BATCH
    wrappers, the call takes the five arguments the wrapper binds by name, and
    the inner function allocates one output per cond index before it looks at
    that cond, which is what makes a masked index come back as exact zeros
    rather than as a short list.
    """
    import inspect

    import comfy.patcher_extension
    import comfy.samplers

    assert comfy.patcher_extension.WrappersMP.CALC_COND_BATCH == "calc_cond_batch", (
        "cond dispatch seam: WrappersMP.CALC_COND_BATCH is no longer the "
        "'calc_cond_batch' wrapper type the dispatch installs under"
    )

    outer = inspect.getsource(comfy.samplers._calc_cond_batch_outer)
    for token in ("WrapperExecutor.new_executor", "_calc_cond_batch",
                  "WrappersMP.CALC_COND_BATCH", "is_model_options=True"):
        assert token in outer, (
            f"cond dispatch seam: _calc_cond_batch_outer no longer names {token!r}, "
            "so the seam the per-cond dispatch installs on is gone or has moved"
        )

    for name in ("calc_cond_batch", "_calc_cond_batch"):
        parameters = list(inspect.signature(
            getattr(comfy.samplers, name)).parameters)
        assert parameters == ["model", "conds", "x_in", "timestep", "model_options"], (
            f"cond dispatch seam: comfy.samplers.{name} now takes {parameters}; the "
            "dispatch binds the call by name and reads 'conds' and 'x_in' off it"
        )

    inner = inspect.getsource(comfy.samplers._calc_cond_batch)
    allocate = inner.index("out_conds.append(torch.zeros_like(x_in))")
    assert allocate < inner.index("if cond is not None:"), (
        "cond dispatch seam: _calc_cond_batch no longer allocates every cond's output "
        "before testing that cond, so masking a cond index would shorten the returned "
        "list instead of zeroing one entry"
    )
    assert "cond_or_uncond.append(o[1])" in inner, (
        "cond dispatch seam: _calc_cond_batch no longer labels a batched chunk with its "
        "own cond index, so masking the list would renumber the labels the model sees"
    )

    _assert_cond_fold_rule(inner)


def _assert_cond_fold_rule(inner: str) -> None:
    """Comfy's own fold rule, which the dispatch decides its branch on.

    The dispatch takes the slice path when comfy can batch this step's
    conditionings into one model call and runs one cond per rank when it
    cannot, so the rule it mirrors has to keep being comfy's: the batching loop
    groups by hook group and then takes every cond against the first with
    ``can_concat_cond``; ``cond_equal_size`` compares the keys and then asks
    each cond object's own ``can_concat``; and every class in ``comfy.conds``
    answers that method, which is where the shape rule, the cross-attention
    repeat rule and the constant value rule live. The one input the loop reads
    that is not replicated across ranks is free memory, which decides how many
    concatenable conds share a call and never whether they can concat, so the
    contract below pins where it sits.
    """
    import inspect

    import comfy.conds
    import comfy.samplers

    from dgx_monarch.adapters.cfg_dispatch import conds_fold

    for token in ("hooked_to_run", "can_concat_cond(to_run[x][0], first[0])",
                  "get_free_memory"):
        assert token in inner, (
            f"cond dispatch seam: _calc_cond_batch no longer names {token!r}, so the "
            "fold rule the dispatch mirrors, or the free-memory reading it deliberately "
            "does not mirror, has moved"
        )

    equal_size = inspect.getsource(comfy.samplers.cond_equal_size)
    for token in ("c1.keys() != c2.keys()", "c1[k].can_concat(c2[k])"):
        assert token in equal_size, (
            f"cond dispatch seam: cond_equal_size no longer names {token!r}, so the "
            "fold rule is no longer 'same keys, then each cond's own can_concat'"
        )
    concat_cond = inspect.getsource(comfy.samplers.can_concat_cond)
    for token in ("c1.input_x.shape != c2.input_x.shape", "objects_concatable(c1.control",
                  "objects_concatable(c1.patches", "cond_equal_size(c1.conditioning"):
        assert token in concat_cond, (
            f"cond dispatch seam: can_concat_cond no longer names {token!r}, so the "
            "outer comparison the dispatch mirrors has changed"
        )

    classes = [value for name, value in vars(comfy.conds).items()
               if name.startswith("COND") and isinstance(value, type)]
    assert classes, "cond dispatch seam: comfy.conds publishes no COND classes"
    for klass in classes:
        assert callable(getattr(klass, "can_concat", None)), (
            f"cond dispatch seam: comfy.conds.{klass.__name__} exposes no can_concat, "
            "so the dispatch cannot ask that class its own fold rule"
        )
        assert callable(getattr(klass, "process_cond", None)), (
            f"cond dispatch seam: comfy.conds.{klass.__name__} exposes no process_cond, "
            "which is what repeats a cond to the latent batch before the comparison"
        )

    # The transaction: real comfy conds through the pack's rule, with comfy's
    # own can_concat_cond as the answer key on the same pair.
    import torch

    x_in = torch.zeros(1, 4, 8, 8)
    for left, right in ((77, 77), (77, 154), (11, 77)):
        conds = [[{"uuid": index, "model_conds": {"c_crossattn": comfy.conds.CONDCrossAttn(
            torch.zeros(1, length, 16))}}] for index, length in enumerate((left, right))]
        pair = [comfy.samplers.get_area_and_mult(entry[0], x_in, torch.zeros(1))
                for entry in conds]
        want = comfy.samplers.can_concat_cond(pair[0], pair[1])
        assert conds_fold(conds, x_in) is want, (
            f"cond dispatch seam: comfy folds cross-attention conds of {left} and "
            f"{right} tokens into one call = {want}, and the pack's conds_fold "
            "disagrees, so the dispatch would take the wrong branch"
        )

    constants = [[{"model_conds": {"num_tokens": comfy.conds.CONDConstant(value)}}]
                 for value in (7, 11)]
    assert conds_fold(constants, x_in, "num_tokens") is False, (
        "cond dispatch seam: two CONDConstants of different value no longer read as "
        "unfoldable, so a declaring family would take the slice and refuse class P"
    )

    # Each dispatching family's declaration held to the comfy source that
    # publishes it: the constant is the whole rank-replicated fact the fold
    # decision reads, so a renamed key must fail here and not on hardware.
    import comfy.model_base

    from dgx_monarch.adapters import ADAPTERS

    checked = 0
    for adapter in ADAPTERS:
        key = getattr(adapter, "cfg_batch_constant", None)
        if key is None:
            continue
        for name in adapter.model_base_classes:
            klass = getattr(comfy.model_base, name, None)
            assert klass is not None, (
                f"cond dispatch seam: {adapter.family} names model_base.{name}, "
                "which comfy no longer has"
            )
            source = "".join(
                inspect.getsource(entry.__dict__["extra_conds"])
                for entry in klass.__mro__ if "extra_conds" in entry.__dict__)
            assert f"out['{key}'] = comfy.conds.CONDConstant" in source, (
                f"cond dispatch seam: {adapter.family} declares cfg_batch_constant="
                f"{key!r}, but model_base.{name}'s extra_conds chain publishes no such "
                "CONDConstant, so the fold decision reads a key comfy never sets"
            )
            checked += 1
    assert checked, (
        "cond dispatch seam: no shipped family declares a cfg_batch_constant, so the "
        "per-cond dispatch is installed nowhere and this contract proves nothing"
    )


def _assert_prepare_sampling_guard_seam() -> None:
    """The PREPARE_SAMPLING seam the cross-rank partial-load guard rides.

    The guard has to read ComfyUI's own load verdict after comfy has decided
    how much of the model it can hold and before the first forward, because
    that verdict is what diverges between two boxes of a pair. Three things
    have to keep holding: the wrapper reads back under the pack's key, the
    all-wrappers view `prepare_sampling` builds its executor from carries it,
    and `prepare_sampling` still routes through that executor at all. If any
    of them stops, the guard goes inert and a divergent render completes with
    no error, which no identity gate on one rank can see.

    The same seam carries the dual-model bracket's hand-off, so what that
    load call is built from is pinned here too: the nested additional models
    of the patcher being prepared, gathered behind comfy's own cycle guard.
    """
    import inspect

    import comfy.patcher_extension as pe
    import comfy.sampler_helpers

    from dgx_monarch.actor.partial_load_guard import WRAPPER_KEY

    def installed(options):
        return pe.get_wrappers_with_key(
            pe.WrappersMP.PREPARE_SAMPLING, WRAPPER_KEY, options, is_model_options=True
        )

    seen: list[object] = []

    def wrapper(executor, *args, **kwargs):
        seen.append(args)
        return executor(*args, **kwargs)

    model_options: dict[str, object] = {}
    assert installed(model_options) == [], (
        "prepare-sampling seam: an empty model_options already reports a keyed wrapper"
    )
    pe.add_wrapper_with_key(
        pe.WrappersMP.PREPARE_SAMPLING, WRAPPER_KEY, wrapper,
        model_options, is_model_options=True,
    )
    assert installed(model_options) == [wrapper], (
        "prepare-sampling seam: a wrapper added under the pack's key does not read back"
    )
    routed = pe.get_all_wrappers(
        pe.WrappersMP.PREPARE_SAMPLING, model_options, is_model_options=True)
    assert routed == [wrapper], (
        "prepare-sampling seam: a keyed wrapper is no longer visible to the all-wrappers "
        "view prepare_sampling builds its executor from; the partial-load guard would go "
        "silently inert"
    )

    def stock(model, noise_shape, conds, model_options=None,
              force_full_load=False, force_offload=False):
        return (model, conds, [])

    executor = pe.WrapperExecutor.new_executor(stock, routed)
    result = executor.execute(object(), (1, 4, 8, 8), {}, model_options=model_options)
    assert isinstance(result, tuple) and len(result) == 3, (
        "prepare-sampling seam: the wrapped call no longer returns "
        "(real_model, conds, models); the guard reads the model off result[0]"
    )
    assert len(seen) == 1, (
        "prepare-sampling seam: the wrapper no longer surrounds the call comfy routes "
        "through it"
    )

    source = inspect.getsource(comfy.sampler_helpers.prepare_sampling)
    assert "WrappersMP.PREPARE_SAMPLING" in source and "WrapperExecutor" in source, (
        "prepare-sampling seam: comfy.sampler_helpers.prepare_sampling no longer builds a "
        "wrapper executor over the PREPARE_SAMPLING wrappers"
    )
    inner = inspect.getsource(comfy.sampler_helpers._prepare_sampling)
    assert "load_models_gpu" in inner, (
        "prepare-sampling seam: the wrapped body no longer calls load_models_gpu, so the "
        "wrapper no longer fires after comfy has decided this rank's residency"
    )
    assert "get_nested_additional_models" in inner, (
        "prepare-sampling seam: the wrapped body no longer gathers the patcher's nested "
        "additional models into the load call, so the dual-model bracket can no longer "
        "put both checkpoints in one prepare and each prepare would evict the other's "
        "model again (issues #301, #314)"
    )
    gathered = inspect.getsource(
        __import__("comfy.model_patcher", fromlist=["ModelPatcher"]
                   ).ModelPatcher.get_nested_additional_models)
    assert "cache_set" in gathered, (
        "prepare-sampling seam: the nested gather dropped its own cycle guard"
    )

    # The flag the guard reads the verdict off. Comfy sets it beside the
    # "loaded partially" log line the journals show.
    patcher_source = inspect.getsource(
        __import__("comfy.model_patcher", fromlist=["ModelPatcher"]).ModelPatcher)
    assert "self.model.model_lowvram = True" in patcher_source, (
        "prepare-sampling seam: ModelPatcher no longer records a partial load in "
        "model.model_lowvram, which is the guard's whole input"
    )


def _assert_minimax_h3_audio_carry_seam() -> None:
    """The stock outer H3 forward around dgx-monarch's rebound `_forward`.

    Comfy bdcb886a moved the dual-schedule conversion out of `_forward` and
    into `MiniMaxH3Model.forward`: the sampler now carries audio scaled onto
    the video schedule, the outer forward restores the stream's own latent for
    wrappers/the network, and it converts the raw audio velocity back exactly
    once afterward. The adapter binds only `_forward`, so this outer shell is
    production behavior, not a helper the adapter may copy.

    A tiny checkpoint-free model drives the real Comfy outer method around the
    production adapter. Identity shard/gather stand in for world 1 only to keep
    this CPU transaction out of xFuser; packing, schedule mapping, adapter
    return semantics, wrapper execution, and the outer carry conversion are
    all the real paths.
    """
    import comfy.patcher_extension
    import torch
    from comfy.ldm.minimax import model as comfy_h3

    from dgx_monarch.adapters import minimax_h3 as adapter_module
    from dgx_monarch.adapters.base import InjectionContext
    from dgx_monarch.adapters.minimax_h3 import MiniMaxH3Adapter

    seen: dict[str, torch.Tensor] = {}

    class ProjectRows(torch.nn.Module):
        def __init__(self, stream: str):
            super().__init__()
            self.stream = stream

        def forward(self, rows):
            seen[self.stream] = rows.detach().clone()
            return torch.zeros(rows.shape[0], 2, dtype=rows.dtype, device=rows.device)

    class FixedHeads(torch.nn.Module):
        def forward(self, _hidden, _t_emb, video_seg, audio_seg,
                    _sigma, _sample_sigmas, _shifts):
            video_rows = video_seg[1] - video_seg[0]
            audio_rows = audio_seg[1] - audio_seg[0]
            return (
                torch.full((video_rows, 4), 2.0, dtype=torch.float32),
                torch.full((audio_rows, 1), 5.0, dtype=torch.float32),
            )

    class TinyH3:
        forward = comfy_h3.MiniMaxH3Model.forward

        def __init__(self):
            self.blocks = []
            self.patch_size = (1, 2, 2)
            self.latents_dim = 1
            self.hidden_size = 2
            self.sigma_shift_video = 12.0
            self.sigma_shift_audio = 3.0
            self.use_adaln_curves = False
            self.time_embedder = lambda values: values[:, None].repeat(1, 2)
            self.video_patch_proj = ProjectRows("video")
            self.audio_patch_proj = ProjectRows("audio")
            self.token_refiner = object()
            self.final_layer = FixedHeads()

        @staticmethod
        def rope_freqs(position_ids, _device):
            return torch.zeros(
                position_ids.shape[0], 2, dtype=torch.float32, device=position_ids.device
            )

        @staticmethod
        def _cond_video_rows(_payload, _device):
            return None

        @staticmethod
        def _cond_audio_rows(_payload, _device):
            return None

    model = TinyH3()
    MiniMaxH3Adapter().inject_usp(
        model, InjectionContext(topology_sp=1, usp_attention=object())
    )

    original = (
        adapter_module.sp_world,
        adapter_module.shard_seq,
        adapter_module.sp_gather,
        adapter_module.padded_row_indices,
    )
    adapter_module.sp_world = lambda: 1
    adapter_module.shard_seq = lambda tensor, dim=1: (tensor, tensor.shape[dim])
    adapter_module.sp_gather = lambda tensor, _orig, dim=1: tensor
    adapter_module.padded_row_indices = lambda _segments: []
    try:
        video = torch.ones(1, 1, 1, 2, 2, dtype=torch.float32)
        carried_audio = torch.full((1, 1, 2, 1), 10.0, dtype=torch.float32)
        context = torch.zeros(1, 1, 2, dtype=torch.float32)
        timestep = torch.tensor([500.0])
        layout = comfy_h3.PackedLayout(1, 1, 2, 2, 1)
        payload = {"seed": 0, "layout": layout, "audio_scale": 4.0}

        output = model.forward(
            [video, carried_audio], timestep, context, {}, minimax_payload=payload
        )
        sigma_a = comfy_h3.time_shift_sigma(
            torch.tensor(0.5), model.sigma_shift_video, model.sigma_shift_audio
        )
        assert torch.allclose(sigma_a, torch.tensor(0.2)), (
            "minimax h3 audio carry seam: the video-to-audio schedule mapping moved"
        )
        assert torch.equal(carried_audio, torch.full_like(carried_audio, 10.0)), (
            "minimax h3 audio carry seam: the stock outer forward mutated the sampler's "
            "carried audio input in place"
        )
        assert torch.allclose(seen["audio"], torch.full((2, 1), 4.0)), (
            "minimax h3 audio carry seam: the rebound inner forward did not receive audio "
            "unscaled from the sampler's carried video schedule"
        )
        assert torch.equal(output[0], torch.full_like(video, -2.0)), (
            "minimax h3 audio carry seam: the outer carry conversion changed the video "
            "stream returned by the rebound adapter"
        )
        assert torch.allclose(output[1], torch.full_like(carried_audio, -20.0)), (
            "minimax h3 audio carry seam: the outer forward did not unscale the carried "
            "audio and convert the adapter's raw velocity exactly once"
        )

        unscaled = model.forward(
            [video, carried_audio],
            timestep,
            context,
            {},
            minimax_payload={"seed": 0, "layout": layout, "audio_scale": 1.0},
        )
        assert torch.equal(unscaled[1], torch.full_like(carried_audio, -5.0)), (
            "minimax h3 audio carry seam: audio_scale=1 no longer leaves the adapter's raw "
            "audio velocity untouched"
        )

        class MixedLegacyOuter(TinyH3):
            def forward(
                self, x, timestep, context, transformer_options=None,
                minimax_payload=None, **kwargs,
            ):
                """Stands in for the outer forward from before comfy bdcb886a, which
                only passed its call through the wrappers.
                """
                options = transformer_options or {}
                executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                    self._forward,
                    self,
                    comfy.patcher_extension.get_all_wrappers(
                        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, options
                    ),
                )
                return executor.execute(
                    x, timestep, context, options,
                    minimax_payload=minimax_payload,
                    **kwargs,
                )

        mixed = MixedLegacyOuter()
        MiniMaxH3Adapter().inject_usp(
            mixed, InjectionContext(topology_sp=1, usp_attention=object())
        )
        try:
            mixed.forward(
                [video, carried_audio], timestep, context, {}, minimax_payload=payload
            )
        except adapter_module.UnsupportedModelError as exc:
            assert "partial or mixed ComfyUI update" in str(exc), (
                "minimax h3 audio carry seam: mixed-tree refusal lost its actionable "
                "compatibility diagnosis"
            )
        else:
            raise AssertionError(
                "minimax h3 audio carry seam: a current payload marker around the legacy "
                "outer forward reached the rebound distributed forward"
            )
    finally:
        (
            adapter_module.sp_world,
            adapter_module.shard_seq,
            adapter_module.sp_gather,
            adapter_module.padded_row_indices,
        ) = original


def _assert_ideogram4_rope_payload_seam() -> None:
    """Hold the ideogram4 rope contract adapters/pixeldit.py implements.

    The adapter mirrors comfy's own forward: llama precompute_freqs_cis on the
    (sharded) positions, then ideogram4's _split_half_rope_matrix, and the
    comfy-kitchen rms_rope_split_half kernel downstream takes the stacked
    rotation matrix (a Tensor), never the (cos, sin, neg_sin) tuple. Both are
    pure functions, so the installed comfy is the transaction.
    """
    import torch
    from comfy.ldm.ideogram4.model import _split_half_rope_matrix
    from comfy.text_encoders.llama import precompute_freqs_cis

    head_dim, tokens = 8, 6
    positions = torch.arange(tokens, dtype=torch.float32).repeat(3, 1)
    payload = precompute_freqs_cis(
        head_dim, positions, 10000.0, rope_dims=[2, 1, 1],
        interleaved_mrope=True, device=torch.device("cpu"))
    assert isinstance(payload, tuple) and len(payload) == 3, (
        "ideogram4 rope seam: precompute_freqs_cis no longer returns the "
        "(cos, sin, neg_sin) tuple the adapter feeds the matrix conversion"
    )
    matrix = _split_half_rope_matrix(payload)
    assert torch.is_tensor(matrix), (
        "ideogram4 rope seam: _split_half_rope_matrix no longer returns the "
        "Tensor rms_rope_split_half requires"
    )
    assert matrix.shape[-2:] == (2, 2), (
        f"ideogram4 rope seam: rotation tail moved: {tuple(matrix.shape)}"
    )
    assert matrix.shape[1] == tokens or matrix.shape[0] == tokens, (
        f"ideogram4 rope seam: token axis vanished: {tuple(matrix.shape)}"
    )


def _assert_ltx_rope_payload_seam() -> None:
    """The ltx adapter shards the rope payload comfy builds; hold its shape.

    comfy 7c59a078 (#15056, 2026-07-24) rebuilt the payload as
    (rotation_matrix, split_mode) with the matrix laid out
    (B, T, heads, head_dim/2, 2, 2), sequence on dim 1. This is the exact
    contract adapters/ltx.py _shard_pe implements. freqs_cis_matrix is a pure
    function, so the installed comfy's own builder is the transaction.
    """
    import torch
    from comfy.ldm.lightricks.model import freqs_cis_matrix

    batch, tokens, heads, half_hd = 2, 6, 3, 12
    freqs = torch.arange(batch * tokens * half_hd, dtype=torch.float32).reshape(
        batch, tokens, half_hd)
    payload = freqs_cis_matrix(freqs, 0, True, heads, torch.float32)
    assert isinstance(payload, tuple) and len(payload) == 2, (
        "ltx rope seam: freqs_cis_matrix no longer returns "
        "(rotation_matrix, split_mode); adapters/ltx.py _shard_pe implements "
        "that exact payload"
    )
    rotation_matrix, split_mode = payload
    assert split_mode is True, (
        "ltx rope seam: the split_mode marker no longer rides slot 1"
    )
    expected = (batch, tokens, heads, half_hd // heads, 2, 2)
    assert tuple(rotation_matrix.shape) == expected, (
        f"ltx rope seam: rotation matrix moved: {tuple(rotation_matrix.shape)} "
        f"!= {expected}; the sequence axis the adapter shards sits on dim 1"
    )
    assert rotation_matrix[:, 1:].shape[1] == tokens - 1, (
        "ltx rope seam: slicing the token axis (dim 1) no longer selects tokens"
    )


def _assert_ltx_metadata_config_seam() -> None:
    """LTX 2.5 geometry arrives in the file header, not in the weight names.

    Detection reads a handful of shapes and then merges
    `__metadata__["config"]["transformer"]` wholesale into dit_config. Comfy's
    own constructor defaults disagree with the shipped checkpoints (connector
    depth 2 against 8, av cross timestep scale 1.0 against 1000.0), so a load
    path that drops the header builds a structurally different model and still
    accepts most of the weights. The loader half of that contract is pinned in
    tests/test_pread_gate.py and tests/test_slab.py; this is the comfy half.
    """
    import json

    import torch
    from comfy.model_detection import detect_unet_config

    prefix = "model.diffusion_model."
    state_dict = {
        f"{prefix}adaln_single.emb.timestep_embedder.linear_1.bias": torch.zeros(8),
        f"{prefix}audio_adaln_single.linear.weight": torch.zeros(8, 8),
        f"{prefix}transformer_blocks.0.attn2.to_k.weight": torch.zeros(4096, 4096),
        f"{prefix}transformer_blocks.1.attn1.to_q.weight": torch.zeros(8, 8),
        f"{prefix}keyframes_abs_pos_embedding": torch.zeros(1, 4096),
    }
    carried = {
        "connector_num_layers": 8,
        "connector_num_attention_heads": 32,
        "av_ca_timestep_scale_multiplier": 1000.0,
        "cross_attention_adaln": True,
        "apply_gated_attention": True,
    }
    metadata = {"config": json.dumps({"transformer": carried})}

    config = detect_unet_config(state_dict, prefix, metadata)
    assert config["image_model"] == "ltxav", (
        "ltx metadata seam: the audio_adaln_single key no longer routes an AV "
        "checkpoint to ltxav, so 2.5 would bind the video-only class"
    )
    assert config["attention_head_dim"] == 128 and config["cross_attention_dim"] == 4096, (
        "ltx metadata seam: the attn2 k-projection no longer yields the head "
        f"dim and cross dim: {config['attention_head_dim']}, "
        f"{config['cross_attention_dim']}"
    )
    missing = {key: value for key, value in carried.items() if config.get(key) != value}
    assert not missing, (
        f"ltx metadata seam: the header config no longer reaches dit_config: {missing}"
    )

    bare = detect_unet_config(state_dict, prefix)
    assert not any(key in bare for key in carried), (
        "ltx metadata seam: dit_config now carries the header fields without "
        "the header, so this transaction no longer proves the header is the "
        "only carrier"
    )

    del state_dict[f"{prefix}keyframes_abs_pos_embedding"]
    without = detect_unet_config(state_dict, prefix, metadata)
    assert without["use_keyframes_abs_pos_embedding"] is False, (
        "ltx metadata seam: the keyframe marker probe no longer overrides the "
        "header, so a header claiming the parameter would build it absent"
    )


def _assert_ltx_stg_passthrough_seam() -> None:
    """Spatio-temporal guidance degrades a flagged self-attention to its values.

    comfy 57ce8e1a short circuits `CrossAttention.forward` to `out = v` ahead of
    the norms and rope when `transformer_options["stg_skip_self_attn"]` is set,
    then still applies the per-head gate. adapters/ltx.py replaces that forward
    wholesale, so it reads the same flag by the same name: a rename upstream
    would leave the adapter running real attention and scoring the guidance
    against a prediction identical to the unperturbed one.
    """
    import comfy.ops
    import torch
    from comfy.ldm.lightricks.model import CrossAttention
    from comfy.ldm.modules.attention import optimized_attention

    from dgx_monarch.adapters.ltx import _make_usp_cross_attention

    heads, dim_head = 2, 4
    attention = CrossAttention(
        query_dim=heads * dim_head, heads=heads, dim_head=dim_head,
        apply_gated_attention=True, dtype=torch.float32,
        device=torch.device("cpu"), operations=comfy.ops.disable_weight_init,
    )
    # disable_weight_init leaves parameters uninitialized; equality assertions
    # over NaN would pass or fail for the wrong reason.
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.copy_(torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32))
    x = torch.randn(1, 3, heads * dim_head, generator=generator)
    context = torch.randn(1, 5, heads * dim_head, generator=generator)
    flag = {"stg_skip_self_attn": True}

    flagged = attention.forward(x, transformer_options=flag)
    plain = attention.forward(x, transformer_options={})
    assert not torch.allclose(flagged, plain), (
        "ltx stg seam: the stg_skip_self_attn flag no longer changes a "
        "self-attention output, so the guidance node perturbs nothing"
    )
    assert torch.equal(
        attention.forward(x, context=context, transformer_options=flag),
        attention.forward(x, context=context, transformer_options={}),
    ), (
        "ltx stg seam: the flag now also short circuits cross-attention, which "
        "the adapter's context test does not reproduce"
    )

    # No rope: the stock forward prefers a fused rotate over the adapter's two
    # calls, and that difference is not what this seam is about.
    replacement = _make_usp_cross_attention(optimized_attention)
    assert torch.equal(replacement(attention, x, transformer_options=flag), flagged), (
        "ltx stg seam: the adapter replacement no longer reproduces the stock "
        "short circuit; under sequence parallelism it would return real "
        "attention where comfy returns the value projection"
    )
    assert torch.allclose(
        replacement(attention, x, transformer_options={}), plain, atol=1e-6), (
        "ltx stg seam: the adapter replacement diverges from stock on the "
        "unflagged path, so the flagged agreement above proves nothing"
    )

    with torch.no_grad():
        attention.to_q.weight.zero_()
        attention.to_q.bias.zero_()
        attention.to_k.weight.zero_()
        attention.to_k.bias.zero_()
    assert torch.equal(attention.forward(x, transformer_options=flag), flagged), (
        "ltx stg seam: the flagged output now depends on the query or key "
        "projection, so it is no longer the per-token value passthrough a "
        "sharded rank can compute from its own rows"
    )


def _assert_minimax_h3_guide_layout_seam() -> None:
    """What an anchored H3 guide puts into the packed layout.

    comfy e01fb4c56 replaced the first/last-only keyframe anchor with an
    arbitrary-frame one, and gave a guide's soundtrack its own segment kind.
    Two things downstream depend on it: the adapter vets every segment kind
    before assembly, so an unlisted kind refuses a render rather than sharding
    it; and h3_rows prices the packed total for the activation preflight, where
    an overcharge refuses a render that fits. Both are checked here against
    layouts comfy builds, not against a re-implementation.
    """
    import torch
    from comfy.ldm.minimax.model import PackedLayout

    from dgx_monarch import h3_rows
    from dgx_monarch.adapters.minimax_h3 import H3_STREAM_OF
    from dgx_monarch.adapters.minimax_h3_packing import H3_SEG_MODALITY

    text_len, latent_t, lat_h, lat_w, audio_t = 3, 2, 48, 84, 5
    grid = h3_rows.frame_rows(lat_h, lat_w)

    def guide(frame, video_frames=None, audio_frames=None):
        block = {"resolved_frame_index": frame}
        if video_frames is not None:
            block["latent"] = torch.zeros(1, 24, video_frames, lat_h, lat_w)
        if audio_frames is not None:
            block["audio_latent"] = torch.zeros(1, 32, 2, audio_frames)
        return block

    cases = (
        ("a still", [guide(0, video_frames=1)], grid),
        ("a clip", [guide(1, video_frames=3)], 3 * grid),
        ("audio alone", [guide(2, audio_frames=4)], 2 * 4),
        ("both streams", [guide(1, video_frames=1, audio_frames=2)], grid + 2 * 2),
        ("two chained", [guide(0, video_frames=1),
                         guide(3, video_frames=2, audio_frames=1)],
         3 * grid + 2),
    )
    for name, keyframes, want in cases:
        layout = PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t,
                              keyframes=keyframes)
        kinds = {kind for _, _, kind in layout.segments}
        unvetted = kinds - set(H3_SEG_MODALITY)
        assert not unvetted, (
            f"minimax h3 guide layout seam: {name} packs segment kind(s) "
            f"{sorted(unvetted)} the adapter has never vetted for sharding"
        )
        sourced = kinds - {"text"} - set(H3_STREAM_OF)
        assert not sourced, (
            f"minimax h3 guide layout seam: {name} packs segment kind(s) "
            f"{sorted(sourced)} with no replicated row source"
        )
        charged = sum(stop - start for start, stop, kind in layout.segments
                      if kind in ("cond", "cond_audio"))
        assert charged == want, (
            f"minimax h3 guide layout seam: {name} now packs {charged} guide "
            f"rows where the shipped row accounting expects {want}"
        )
        priced = h3_rows.rows_from_request(
            {"positive": [[torch.zeros(1, text_len, 8),
                           {"minimax_keyframes": keyframes}]]},
            {"samples": _H3Nested(torch.zeros(1, 24, latent_t, lat_h, lat_w),
                                  torch.zeros(1, 32, 2, audio_t))},
        )
        assert priced.cond == charged, (
            f"minimax h3 guide layout seam: {name} prices {priced.cond} guide "
            f"rows against the {charged} comfy packs"
        )


def _assert_minimax_h3_guide_inputs_seam() -> None:
    """The names the loader-side wall prices an anchored guide by.

    That wall reads comfy's hidden API-format prompt and charges a reference
    encode by input name. Add Guide's `image` and `audio` match none of the
    reference prefixes, so before the wall named them a guide clip was admitted
    at a canvas whose encode needs more than 94 GiB (1344x768, 2026-08-14). The
    price map is keyed on names this node owns, so this seam holds the node's
    own schema against it: a renamed input, or a new pixel or audio input,
    breaks the canary instead of a host.
    """
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide

    from dgx_monarch import h3_activation
    from dgx_monarch.nodes import loader_graph

    schema = MiniMaxH3AddGuide.define_schema()
    assert schema.node_id in loader_graph._GUIDE_NODE_CLASSES, (
        f"minimax h3 guide inputs seam: the guide node is {schema.node_id!r}, "
        f"which the loader-side wall does not price "
        f"({loader_graph._GUIDE_NODE_CLASSES})"
    )
    priced = {"IMAGE": loader_graph._GUIDE_IMAGE_KEYS,
              "AUDIO": loader_graph._GUIDE_AUDIO_KEYS}
    for section in MiniMaxH3AddGuide.INPUT_TYPES().values():
        if not isinstance(section, dict):
            continue
        for name, spec in section.items():
            io_type = spec[0] if isinstance(spec, tuple) else spec
            if io_type in priced:
                assert name in priced[io_type], (
                    f"minimax h3 guide inputs seam: {io_type} input {name!r} "
                    f"is not in the wall's priced vocabulary "
                    f"{priced[io_type]}, so a graph wiring it encodes bytes "
                    f"nothing charges"
                )

    # The loader widget the length proof reads. Comfy's LoadImage carries its
    # filename here; a rename does not endanger a host (an unnamed file prices
    # the worst case) but it silently retires the proof, so it is pinned.
    import nodes as comfy_nodes

    load_image = comfy_nodes.NODE_CLASS_MAPPINGS["LoadImage"]
    file_widgets = {
        name for section in load_image.INPUT_TYPES().values()
        if isinstance(section, dict) for name, spec in section.items()
        if isinstance(spec, tuple) and isinstance(spec[0], (list, tuple))
    }
    assert set(loader_graph._GUIDE_FILE_KEYS) & file_widgets, (
        f"minimax h3 guide inputs seam: LoadImage names its file on "
        f"{sorted(file_widgets)}, none of which the wall reads "
        f"({loader_graph._GUIDE_FILE_KEYS}), so no guide length can be proved"
    )

    # And the wall charges through those names, on the geometry the
    # guide is resized to rather than the one it arrives at.
    width, height = 1344, 768
    prompt = {
        "1": {"class_type": "EmptyMiniMaxH3LatentAV",
              "inputs": {"width": width, "height": height, "length": 124}},
        "2": {"class_type": schema.node_id,
              "inputs": {"positive": ["3", 0], "latent": ["1", 0],
                         "vae": ["4", 0], "image": ["5", 0], "frame_idx": 62}},
    }
    terms = loader_graph.driver_stack_terms(
        prompt, family=h3_activation.H3_FAMILY, sp_degree=2)
    want = h3_activation.guide_encode_bytes(width * height)
    assert terms.guide_refs == 1 and terms.guide_bytes == want > 0, (
        f"minimax h3 guide inputs seam: the wall charges {terms.guide_bytes} "
        f"bytes for a {width}x{height} guide where the measured probe prices "
        f"{want}"
    )


class _H3Nested:
    """comfy's NestedTensor duck for the row-accounting call above."""

    def __init__(self, video, audio):
        self.tensors = [video, audio]
        self.shape = video.shape


def _assert_ltx_keyframe_embedding_seam() -> None:
    """The 2.5 keyframe marker is added before the seam the adapter shards.

    `keyframes_abs_pos_embedding` is the only new DiT weight in LTX 2.5. It is
    added to every token whose temporal start is 0, minus the trailing guide
    tokens, inside `_process_input` and ahead of `_process_transformer_blocks`.
    adapters/ltx.py wraps the block loop, so the marker costs the workers no
    math only while it stays on that side of the boundary.
    """
    import inspect

    import torch
    from comfy.ldm.lightricks.av_model import LTXAVModel
    from comfy.ldm.lightricks.model import LTXVModel

    class Keyframed:
        keyframes_abs_pos_mask = LTXVModel.keyframes_abs_pos_mask
        apply_keyframes_abs_pos_embedding = LTXVModel.apply_keyframes_abs_pos_embedding

    inner_dim, tokens = 4, 5
    model = Keyframed()
    model.keyframes_abs_pos_embedding = torch.full((1, inner_dim), 5.0)
    pixel_coords = torch.zeros(1, 3, tokens)
    pixel_coords[0, 0] = torch.tensor([0.0, 0.0, 8.0, 0.0, 16.0])

    mask = model.keyframes_abs_pos_mask(
        pixel_coords, orig_shape=None, grid_mask=None,
        num_guide_tokens=0, generated_keyframes=None)
    assert mask.tolist() == [[True, True, False, True, False]], (
        f"ltx keyframe seam: the marker no longer selects the tokens whose "
        f"temporal start is 0: {mask.tolist()}"
    )
    guarded = model.keyframes_abs_pos_mask(
        pixel_coords, orig_shape=None, grid_mask=None,
        num_guide_tokens=2, generated_keyframes=None)
    assert guarded.tolist() == [[True, True, False, False, False]], (
        f"ltx keyframe seam: trailing guide tokens are no longer excluded from "
        f"the marker: {guarded.tolist()}"
    )
    marked = model.apply_keyframes_abs_pos_embedding(
        torch.zeros(1, tokens, inner_dim), pixel_coords, orig_shape=None,
        grid_mask=None, num_guide_tokens=0, generated_keyframes=None)
    assert marked[0, :, 0].tolist() == [5.0, 5.0, 0.0, 5.0, 0.0], (
        f"ltx keyframe seam: the marker is no longer added per masked token: "
        f"{marked[0, :, 0].tolist()}"
    )

    assert "self.apply_keyframes_abs_pos_embedding(" in inspect.getsource(
        LTXVModel._process_input), (
        "ltx keyframe seam: the marker left _process_input; the adapter shards "
        "downstream of it and would apply it to a partial sequence"
    )
    assert "super()._process_input(" in inspect.getsource(LTXAVModel._process_input), (
        "ltx keyframe seam: the AV input path no longer runs the video-side "
        "_process_input, so the marker may reach the sharded stream unapplied"
    )
    late = [
        f"{owner.__name__}.{method}"
        for owner, method in (
            (LTXVModel, "_process_transformer_blocks"),
            (LTXAVModel, "_process_transformer_blocks"),
        )
        if "apply_keyframes_abs_pos_embedding" in inspect.getsource(
            getattr(owner, method))
    ]
    assert not late, (
        f"ltx keyframe seam: the marker is now applied inside the block loop "
        f"the adapter binds, on sharded tokens: {late}"
    )


def _assert_ltx_guide_attenuation_seam() -> None:
    """Which guide settings make comfy build a self-attention bias.

    LTXV Add Guide appends guide frames and records one entry per guide. comfy
    turns those entries into an additive attention bias only when a guide asks
    to be attenuated: strength other than 1.0, or a spatial pixel mask.
    adapters/ltx.py re-expresses exactly that bias under ulysses and refuses it
    under ring. Both halves of the predicate are comfy's, so a change here
    silently turns a plain render into a biased one, or a biased one into a
    sharded render that drops a bias it should have applied.
    """
    import torch
    from comfy.ldm.lightricks.model import LTXVModel

    build = LTXVModel._build_guide_self_attention_mask
    x = torch.zeros(1, 8, 4)

    def entries(strength, pixel_mask=None):
        return [{"pre_filter_count": 2, "strength": strength,
                 "pixel_mask": pixel_mask, "latent_shape": (1, 1, 2),
                 "surviving_count": 2}]

    plain = {"num_guide_tokens": 2, "resolved_guide_entries": entries(1.0)}
    assert build(None, x, {}, plain) is None, (
        "ltx guide seam: a strength 1.0 guide with no spatial mask now builds "
        "an attention bias, so the adapter shards a render comfy biases"
    )
    for name, args in (
        ("no guide tokens", {"num_guide_tokens": 0}),
        ("no resolved entries", {"num_guide_tokens": 2, "resolved_guide_entries": []}),
    ):
        assert build(None, x, {}, args) is None, (
            f"ltx guide seam: {name} now builds an attention bias"
        )

    attenuated = {"num_guide_tokens": 2, "resolved_guide_entries": entries(0.7)}
    assert build(None, x, {}, attenuated) is not None, (
        "ltx guide seam: a guide strength other than 1.0 no longer builds the "
        "attention bias the adapter refuses on; the refusal would now let a "
        "silently unattenuated render through"
    )

    masked = {"num_guide_tokens": 2,
              "resolved_guide_entries": entries(1.0, torch.full((1, 1, 1, 2, 2), 0.5))}
    assert build(LTXVModel, x, {}, masked) is not None, (
        "ltx guide seam: a spatial guide mask no longer builds an attention bias"
    )
    # A mask that asks for nothing is short circuited back to None even though
    # it trips the needs_mask test. The adapter refuses on the built mask rather
    # than on the settings, so comfy's own no-op cases stay renderable.
    no_op = {"num_guide_tokens": 2,
             "resolved_guide_entries": entries(1.0, torch.ones(1, 1, 1, 2, 2))}
    assert build(LTXVModel, x, {}, no_op) is None, (
        "ltx guide seam: an all-ones guide mask at strength 1.0 now builds a "
        "bias, so the adapter would refuse a render comfy leaves unbiased"
    )

    # The AV model takes the same builder with x as a [video, audio] pair, and
    # sizes the bias off the video stream.
    assert build(None, [x, torch.zeros(1, 3, 2)], {}, attenuated) is not None, (
        "ltx guide seam: the AV two-stream input no longer reaches the builder"
    )



def _assert_ltx_guide_bias_layout_seam() -> None:
    """The bias layout the sharded LTX path re-expresses, both halves.

    adapters/ltx.py reads GuideAttentionMask's four slots and rebuilds comfy's
    own query split for the sharded run, so it depends on two upstream facts.
    First the rectangles: guides at the tail, weights in log space, both shaped
    (1, 1, ., T) so they broadcast over whatever heads a rank holds. Second the
    split: comfy attends the noisy queries, then the tracked queries, then
    anything after them, each against the full keys. A change to either turns
    the sharded render into a differently-weighted picture with no error.
    """
    import inspect

    import torch
    from comfy.ldm.lightricks.model import (
        GuideAttentionMask,
        _attention_with_guide_mask,
    )

    total, guide_start, tracked = 10, 7, 3
    weights = torch.tensor([0.5, 1.0, 0.0], dtype=torch.float32)
    mask = GuideAttentionMask(total, guide_start, tracked, weights)

    assert (mask.guide_start, mask.tracked_count) == (guide_start, tracked), (
        "ltx guide bias seam: the mask no longer reports where the guides start"
    )
    assert mask.noisy_mask.shape == (1, 1, 1, total), (
        f"ltx guide bias seam: noisy rectangle is {tuple(mask.noisy_mask.shape)}, "
        "not (1, 1, 1, T); the adapter broadcasts it over a rank's heads"
    )
    assert mask.tracked_mask.shape == (1, 1, tracked, total), (
        f"ltx guide bias seam: tracked rectangle is "
        f"{tuple(mask.tracked_mask.shape)}, not (1, 1, tracked, T)"
    )
    finfo = torch.finfo(weights.dtype)
    assert torch.allclose(
        mask.noisy_mask[0, 0, 0, guide_start:guide_start + tracked],
        torch.tensor([math.log(0.5), 0.0, finfo.min]),
    ), "ltx guide bias seam: the weights are no longer log-space additive"
    assert not mask.noisy_mask[0, 0, 0, :guide_start].any(), (
        "ltx guide bias seam: noisy tokens now carry a bias on noisy keys"
    )
    assert not mask.tracked_mask[0, 0, :, guide_start:].any(), (
        "ltx guide bias seam: guide tokens now carry a bias on guide keys"
    )

    source = inspect.getsource(_attention_with_guide_mask)
    for fragment, note in (
        ("q[:, :guide_start, :]", "the noisy query group"),
        ("q[:, guide_start:tracked_end, :]", "the tracked guide query group"),
        ("q[:, tracked_end:, :]", "the unbiased trailing query group"),
        ("mask=guide_mask.noisy_mask", "the noisy group's own sub-mask"),
        ("mask=guide_mask.tracked_mask", "the tracked group's own sub-mask"),
        ("low_precision_attention=False", "the SDPA fallback the adapter matches"),
    ):
        assert fragment in source, (
            f"ltx guide bias seam: {note} is gone from _attention_with_guide_mask; "
            "the sharded re-expression copies this exact split"
        )


def _assert_minimax_h3_latent_mask_seam() -> None:
    """The latent noise-mask parameters comfy ff6c8a8 added to H3's forward.

    Upstream answers a mask by running masked rows at their own timestep, which
    turns a modulation row into a per-token index tensor. The rebound
    `_forward` carries one row per packed segment, so the adapter refuses a set
    mask instead of dropping it. This pins the parameter list to the two shapes
    the rebound contract admits, holds the outer and inner forwards to the same
    shape, and drives the real stock outer forward to prove a set mask reaches
    the rebound inner on either tree while an unset one still renders.
    """
    import torch
    from comfy.ldm.minimax import model as comfy_h3

    from dgx_monarch.adapters import minimax_h3 as adapter_module
    from dgx_monarch.adapters.base import InjectionContext
    from dgx_monarch.adapters.minimax_h3 import MiniMaxH3Adapter

    def explicit(method):
        return tuple(
            name
            for name, parameter in inspect.signature(method).parameters.items()
            if parameter.kind is not inspect.Parameter.VAR_KEYWORD
        )

    base = ("self", "x", "timestep", "context", "transformer_options", "minimax_payload")
    masked = (*base, "denoise_mask", "audio_denoise_mask")
    inner = explicit(comfy_h3.MiniMaxH3Model._forward)
    assert inner in (base, masked), (
        f"minimax h3 latent mask seam: MiniMaxH3Model._forward now takes {inner}, "
        "which is neither admitted shape; the adapter re-expresses this list"
    )
    assert explicit(comfy_h3.MiniMaxH3Model.forward) == inner, (
        "minimax h3 latent mask seam: the outer forward no longer declares the same "
        "parameters as the inner one it hands the wrapper executor"
    )

    class ProjectRows(torch.nn.Module):
        def forward(self, rows):
            return torch.zeros(rows.shape[0], 2, dtype=rows.dtype, device=rows.device)

    class FixedHeads(torch.nn.Module):
        def forward(self, _hidden, _t_emb, video_seg, audio_seg,
                    _sigma, _sample_sigmas, _shifts):
            return (
                torch.zeros(video_seg[1] - video_seg[0], 4, dtype=torch.float32),
                torch.zeros(audio_seg[1] - audio_seg[0], 1, dtype=torch.float32),
            )

    class TinyMaskedH3:
        forward = comfy_h3.MiniMaxH3Model.forward

        def __init__(self):
            self.blocks = []
            self.patch_size = (1, 2, 2)
            self.latents_dim = 1
            self.hidden_size = 2
            self.sigma_shift_video = 12.0
            self.sigma_shift_audio = 3.0
            self.use_adaln_curves = False
            self.time_embedder = lambda values: values[:, None].repeat(1, 2)
            self.video_patch_proj = ProjectRows()
            self.audio_patch_proj = ProjectRows()
            self.token_refiner = object()
            self.final_layer = FixedHeads()

        @staticmethod
        def rope_freqs(position_ids, _device):
            return torch.zeros(
                position_ids.shape[0], 2, dtype=torch.float32, device=position_ids.device
            )

        @staticmethod
        def _cond_video_rows(_payload, _device):
            return None

        @staticmethod
        def _cond_audio_rows(_payload, _device):
            return None

    model = TinyMaskedH3()
    MiniMaxH3Adapter().inject_usp(
        model, InjectionContext(topology_sp=1, usp_attention=object())
    )
    original = (
        adapter_module.sp_world,
        adapter_module.shard_seq,
        adapter_module.sp_gather,
        adapter_module.padded_row_indices,
    )
    adapter_module.sp_world = lambda: 1
    adapter_module.shard_seq = lambda tensor, dim=1: (tensor, tensor.shape[dim])
    adapter_module.sp_gather = lambda tensor, _orig, dim=1: tensor
    adapter_module.padded_row_indices = lambda _segments: []
    try:
        video = torch.ones(1, 1, 1, 2, 2, dtype=torch.float32)
        audio = torch.full((1, 1, 2, 1), 10.0, dtype=torch.float32)
        context = torch.zeros(1, 1, 2, dtype=torch.float32)
        timestep = torch.tensor([500.0])
        payload = {
            "seed": 0,
            "layout": comfy_h3.PackedLayout(1, 1, 2, 2, 1),
            "audio_scale": 1.0,
        }
        arguments = ([video, audio], timestep, context, {})
        unmasked = model.forward(*arguments, minimax_payload=payload)
        assert len(unmasked) == 2, (
            "minimax h3 latent mask seam: an unset mask no longer renders through the "
            "rebound forward"
        )
        for name in ("denoise_mask", "audio_denoise_mask"):
            try:
                model.forward(
                    *arguments,
                    minimax_payload=payload,
                    **{name: torch.zeros(1, 1, 1, 2, 2, dtype=torch.float32)},
                )
            except adapter_module.UnsupportedModelError as exc:
                assert name in str(exc) and "mode=local" in str(exc), (
                    f"minimax h3 latent mask seam: the {name} refusal lost its name or "
                    "its single-GPU remedy"
                )
            else:
                raise AssertionError(
                    f"minimax h3 latent mask seam: a set {name} reached the sharded "
                    "modulation plan instead of being refused"
                )
    finally:
        (
            adapter_module.sp_world,
            adapter_module.shard_seq,
            adapter_module.sp_gather,
            adapter_module.padded_row_indices,
        ) = original


def _assert_minimax_h3_pdd_head_seam() -> None:
    """The sampler schedule comfy 2504e68d made the H3 output head require.

    A PDD LoRA stacks row blocks into `video_out` and `audio_out`, and the head
    blends the blocks the step it is taking spans, so `FinalLayer.forward` now
    takes the current sigma, the sampler's whole sigma table and the two flow
    shifts. The rebound `_forward` calls that head itself, so it has to hand
    down the same three values from the same places: the clamped video sigma it
    already computes, the table off the stock `transformer_options`, and the
    pair of shift overrides.

    A checkpoint-free head runs on a stock bank and on a stacked one. Both legs
    hold the production adapter against the stock inner forward on the same
    weights, and a hook on the head reads back the three values it was handed,
    so a wrong sigma fails here even where the block span it picks would not
    move. Identity shard/gather stand in for world 1 only; packing, schedule
    mapping and the head itself are the real paths.
    """
    import comfy.ops
    import torch
    from comfy.ldm.minimax import model as comfy_h3

    from dgx_monarch.adapters import minimax_h3 as adapter_module
    from dgx_monarch.adapters.base import InjectionContext
    from dgx_monarch.adapters.minimax_h3 import MiniMaxH3Adapter

    class ProjectRows(torch.nn.Module):
        def forward(self, rows):
            return torch.zeros(rows.shape[0], 2, dtype=rows.dtype, device=rows.device)

    def build_head(blocks):
        """A real FinalLayer whose head bank holds `blocks` row blocks."""
        head = comfy_h3.FinalLayer(
            2, 2, 4, 1, 1e-6, apply_silu=True, adaln_dtype=torch.float32,
            dtype=torch.float32, device="cpu",
            operations=comfy.ops.disable_weight_init)
        if blocks > 1:
            for linear, width in ((head.video_out, 4), (head.audio_out, 1)):
                linear.weight = torch.nn.Parameter(
                    torch.empty(width * blocks, 2, dtype=torch.float32))
                linear.bias = torch.nn.Parameter(
                    torch.empty(width * blocks, dtype=torch.float32))
        # disable_weight_init leaves parameters uninitialized, so fill every one
        # from one seeded generator: both legs must read identical weights.
        generator = torch.Generator().manual_seed(7)
        for parameter in head.parameters():
            with torch.no_grad():
                parameter.copy_(torch.empty(parameter.shape).normal_(
                    0.0, 0.2, generator=generator))
        head.register_forward_pre_hook(lambda _module, args: handed.append(args))
        return head

    def build_model(head):
        class TinyPddH3:
            forward = comfy_h3.MiniMaxH3Model.forward
            _forward = comfy_h3.MiniMaxH3Model._forward
            # ComfyUI 2d6b7328 moved the embed and pack out of the stock _forward
            # into this helper; bind it where it exists so the stand-in runs both.
            if hasattr(comfy_h3.MiniMaxH3Model, "_embed_and_pack"):
                _embed_and_pack = comfy_h3.MiniMaxH3Model._embed_and_pack

            def __init__(self):
                self.blocks = []
                self.patch_size = (1, 2, 2)
                self.latents_dim = 1
                self.hidden_size = 2
                self.sigma_shift_video = 12.0
                self.sigma_shift_audio = 3.0
                self.use_adaln_curves = False
                self.time_embedder = lambda values: values[:, None].repeat(1, 2)
                self.video_patch_proj = ProjectRows()
                self.audio_patch_proj = ProjectRows()
                self.token_refiner = object()
                self.final_layer = head

            @staticmethod
            def rope_freqs(position_ids, _device):
                return torch.zeros(
                    position_ids.shape[0], 2, dtype=torch.float32,
                    device=position_ids.device)

            @staticmethod
            def _cond_video_rows(_payload, _device):
                return None

            @staticmethod
            def _cond_audio_rows(_payload, _device):
                return None

        return TinyPddH3()

    original = (
        adapter_module.sp_world,
        adapter_module.shard_seq,
        adapter_module.sp_gather,
        adapter_module.padded_row_indices,
    )
    stock_pdd_head = comfy_h3._pdd_head
    adapter_module.sp_world = lambda: 1
    adapter_module.shard_seq = lambda tensor, dim=1: (tensor, tensor.shape[dim])
    adapter_module.sp_gather = lambda tensor, _orig, dim=1: tensor
    adapter_module.padded_row_indices = lambda _segments: []
    blended: list[tuple[int, int, int, float]] = []
    handed: list[tuple[object, ...]] = []

    def recording_pdd_head(head, hidden, n, start, stop, flow_shift):
        blended.append((n, start, stop, flow_shift))
        return stock_pdd_head(head, hidden, n, start, stop, flow_shift)

    try:
        video = torch.ones(1, 1, 1, 2, 2, dtype=torch.float32)
        audio = torch.full((1, 1, 2, 1), 10.0, dtype=torch.float32)
        context = torch.zeros(1, 1, 2, dtype=torch.float32)
        timestep = torch.tensor([800.0])
        # Sigma 0.5 is the fixed point of 1 - sigma, where model time and the
        # sampler sigma read alike; 0.8 tells them apart. The step it names
        # still clamps onto the second block of a two-block bank.
        table = torch.tensor([1.0, 0.8, 0.4, 0.0], dtype=torch.float32)

        def render(model, options):
            return model.forward(
                [video, audio], timestep, context, options,
                minimax_payload={
                    "seed": 0,
                    "layout": comfy_h3.PackedLayout(1, 1, 2, 2, 1),
                    "audio_scale": 1.0,
                },
            )

        def check_handed(shifts):
            """The three values the head read, as the rebound forward handed them.

            The block span a stacked head picks saturates over most of the
            schedule, so the span alone cannot tell the video sigma from the
            audio one or from a constant. Read the arguments instead.
            """
            assert len(handed) == 1, (
                f"minimax h3 pdd head seam: the head ran {len(handed)} time(s) in one "
                "step, so the rebound forward no longer calls it exactly once"
            )
            sigma, schedule, read_shifts = handed[0][4:]
            assert isinstance(sigma, torch.Tensor) and torch.equal(sigma, torch.tensor(0.8)), (
                f"minimax h3 pdd head seam: the head read {sigma!r} as this step's sigma "
                "rather than the clamped video sigma the timestep carries"
            )
            assert schedule is table, (
                f"minimax h3 pdd head seam: the head read {schedule!r} as the sigma "
                "schedule rather than the sampler's own table"
            )
            assert read_shifts == shifts, (
                f"minimax h3 pdd head seam: the head read shifts {read_shifts!r} rather "
                f"than {shifts!r} in video, audio order"
            )

        for blocks in (1, 2):
            head = build_head(blocks)
            rebound = build_model(head)
            MiniMaxH3Adapter().inject_usp(
                rebound, InjectionContext(topology_sp=1, usp_attention=object()))
            blended.clear()
            handed.clear()
            comfy_h3._pdd_head = recording_pdd_head
            try:
                adapted = render(rebound, {"sample_sigmas": table})
            finally:
                comfy_h3._pdd_head = stock_pdd_head
            check_handed((12.0, 3.0))
            stock = render(build_model(head), {"sample_sigmas": table})
            assert torch.equal(adapted[0], stock[0]) and torch.equal(adapted[1], stock[1]), (
                f"minimax h3 pdd head seam: with {blocks} head row block(s) the rebound "
                "forward no longer reproduces the stock inner forward on the same weights"
            )
            expected = [] if blocks == 1 else [(2, 1, 2, 12.0), (2, 1, 2, 3.0)]
            assert blended == expected, (
                f"minimax h3 pdd head seam: {blocks} head row block(s) blended {blended}, "
                f"expected {expected}; the sigma or the schedule the adapter hands the "
                "head no longer locates the step it is taking"
            )

        # The head reads the two flow shifts per stream, and the overrides the
        # sigma-shift node writes have to survive the rebound call.
        stacked = build_model(build_head(2))
        MiniMaxH3Adapter().inject_usp(
            stacked, InjectionContext(topology_sp=1, usp_attention=object()))
        blended.clear()
        handed.clear()
        comfy_h3._pdd_head = recording_pdd_head
        try:
            render(stacked, {"sample_sigmas": table,
                             "minimax_h3_sigma_shift_video": 8.0,
                             "minimax_h3_sigma_shift_audio": 5.0})
        finally:
            comfy_h3._pdd_head = stock_pdd_head
        check_handed((8.0, 5.0))
        assert [row[3] for row in blended] == [8.0, 5.0], (
            f"minimax h3 pdd head seam: the head blended {blended}, so the per-stream "
            "sigma-shift overrides no longer reach it through the rebound forward"
        )

        # The table belongs to the sampler. A rebound forward that substituted
        # one of its own would hide this raise instead of reporting it.
        try:
            render(stacked, {})
        except ValueError as exc:
            assert "sigma schedule" in str(exc), (
                "minimax h3 pdd head seam: a stacked head without the sampler schedule "
                f"raised {exc!r} rather than naming the missing schedule"
            )
        else:
            raise AssertionError(
                "minimax h3 pdd head seam: a stacked head ran without the sampler "
                "schedule, so the rebound forward is not passing the sampler's own table"
            )
    finally:
        comfy_h3._pdd_head = stock_pdd_head
        (
            adapter_module.sp_world,
            adapter_module.shard_seq,
            adapter_module.sp_gather,
            adapter_module.padded_row_indices,
        ) = original


def _assert_nvfp4_activation_scale_seam() -> None:
    """The scale an nvfp4 activation gets when the checkpoint ships none.

    The shared-scale hook (adapters/quant_activation_scale) depends on three
    facts: every quantization format comfy offers is classified in the pack's
    table, nvfp4 alone takes a whole-tensor statistic, and the quantized
    Linear reads the per-module `input_scale` the hook writes and the
    `pre_quant_scale` the installer declines on. A silent change to any one of
    them turns a sharded render's numbers wrong while every pixel still
    arrives, which no identity gate can see. The last check pins the route
    MOSS-Image's ffn_down takes through ``comfy.ops.linear_input_act``: an
    INT8 ConvRot weight takes comfy-kitchen's fused path and never calls the
    module, and an nvfp4 weight falls back to the module, where the shared
    scale is applied (adapters/moss_image.py ``sp_quant_refusal_reason``).
    """
    import inspect

    import comfy.ops
    import torch
    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class

    from dgx_monarch.adapters.quant_activation_scale import (
        ACTIVATION_SCALE_RULES,
        NVFP4_SCALE_DIVISOR,
        SHARD_DEPENDENT_LAYOUTS,
    )

    unclassified = sorted(
        config["comfy_tensor_layout"] for config in QUANT_ALGOS.values()
        if config["comfy_tensor_layout"] not in ACTIVATION_SCALE_RULES)
    assert not unclassified, (
        "nvfp4 activation scale seam: comfy offers quantization layout(s) the "
        f"pack's activation-scale table does not classify: {unclassified}")
    shard_dependent = sorted(
        config["comfy_tensor_layout"] for config in QUANT_ALGOS.values()
        if config["comfy_tensor_layout"] in SHARD_DEPENDENT_LAYOUTS)
    assert shard_dependent == ["TensorCoreNVFP4Layout"], (
        "nvfp4 activation scale seam: the set of shard-dependent layouts moved, "
        f"got {shard_dependent}")

    layout = get_layout_class("TensorCoreNVFP4Layout")
    assert layout.get_padded_shape((17, 17)) == (32, 32), (
        "nvfp4 activation scale seam: the layout no longer aligns to 16, which "
        "is the block width benchmark/chroma_numerics_probe.py emulates")
    base = torch.randn(32, 32, dtype=torch.bfloat16)
    far = base.clone()
    far[31, 31] = 40.0  # outside row 0's first 16-element block
    _qdata, near_params = layout.quantize(base)
    _qdata, far_params = layout.quantize(far)
    assert not torch.equal(near_params.scale, far_params.scale), (
        "nvfp4 activation scale seam: the no-scale fallback no longer reads a "
        "WHOLE-tensor statistic, so a per-rank shard would no longer move it")
    expected = (base.abs().amax() / NVFP4_SCALE_DIVISOR).to(torch.float32)
    assert torch.equal(near_params.scale, expected), (
        "nvfp4 activation scale seam: the fallback scale is no longer "
        "amax(abs(x)) divided in the activation dtype by the pack's constant")
    scaled = layout.quantize(base, scale=expected)[1].scale
    assert torch.equal(scaled, expected), (
        "nvfp4 activation scale seam: the scale keyword no longer reaches the "
        "layout, so an injected input_scale would be ignored")
    assert torch.equal(
        QuantizedTensor.from_float(base, "TensorCoreNVFP4Layout").params.scale,
        near_params.scale), (
        "nvfp4 activation scale seam: from_float and the layout no longer agree")

    source = inspect.getsource(comfy.ops.mixed_precision_ops)
    for name in ("input_scale", "pre_quant_scale", "_full_precision_mm",
                 "comfy_force_cast_weights", "weight_function"):
        assert f"'{name}'" in source or f"self.{name}" in source, (
            "nvfp4 activation scale seam: the quantized Linear no longer reads "
            f"{name}, which the shared-scale wrapper's gate reproduces")

    # The expert bank shares the layout and takes no scale, so the installer
    # declines it rather than paying a collective for an attribute nothing
    # reads. `num_experts` is the only thing that tells the two apart: comfy
    # declares both as plain torch.nn.Module subclasses, so isinstance cannot.
    assert "self.num_experts = num_experts" in source, (
        "nvfp4 activation scale seam: MoEExperts no longer sets num_experts, "
        "which is how the installer tells an expert bank from a Linear")
    assert "class Linear(torch.nn.Module" in source, (
        "nvfp4 activation scale seam: the quantized Linear is no longer a plain "
        "Module, so the installer's discriminator may now have a better one")
    experts = source[source.index("class MoEExperts"):]
    call = "QuantizedTensor.from_float(input, self.layout_type)"
    assert call in experts and f"{call[:-1]}, scale=" not in experts, (
        "nvfp4 activation scale seam: MoEExperts now passes a scale to "
        "from_float, so the installer could cover it instead of declining it")

    def routed(layout_name):
        module = comfy.ops.disable_weight_init.Linear(256, 64, bias=False, dtype=torch.bfloat16)
        module.weight = torch.nn.Parameter(QuantizedTensor.from_float(
            torch.randn(64, 256, dtype=torch.bfloat16), layout_name), requires_grad=False)
        calls: list = []

        def forward(rows, *args, **kwargs):
            calls.append(tuple(rows.shape))
            return torch.zeros(*rows.shape[:-1], 64, dtype=torch.bfloat16)

        module.forward = forward
        out = comfy.ops.linear_input_act(module, torch.randn(3, 512, dtype=torch.bfloat16), "situ")
        return calls, out

    calls, out = routed("TensorWiseINT8Layout")
    assert not calls and tuple(out.shape) == (3, 64), (
        "nvfp4 activation scale seam: an INT8 ConvRot linear_input_act now calls the module "
        f"({calls}), so MOSS-Image's ffn_down no longer takes the fused path")
    calls, _out = routed("TensorCoreNVFP4Layout")
    assert calls == [(3, 256)], (
        "nvfp4 activation scale seam: an nvfp4 linear_input_act no longer falls back to the "
        f"module ({calls}), so the shared-scale wrapper would be bypassed")


def _assert_chroma_text_stream_seam() -> None:
    """What chroma's text encoder hands the model, and what it never attaches.

    Comfy's encoder does not hand chroma the short negative of an unequal
    prompt pair padded up to the positive under an attention mask, the way
    the driver's own cfg equalizer pads and masks (checked 2026-09-06). The
    PixArt T5 path chroma loads pads to a floor of one row and builds its
    encoder with masks off, so every text row the model reads is a real token
    on a single-GPU render and on a sharded one alike, and two prompts of
    different length stay two streams of different length. Flux's own T5 pads
    every prompt to 256 rows, which is why chroma is the flux-family member
    whose two conds keep different shapes and take two model calls. In stock
    comfy the only route a mask has into chroma's forward is regional
    conditioning, which the sharded forward refuses before any collective.
    """
    import inspect

    import comfy.ldm.chroma.model
    import comfy.model_base
    import comfy.text_encoders.flux
    import comfy.text_encoders.pixart_t5
    import comfy.text_encoders.sd3_clip

    tokenizer = comfy.text_encoders.pixart_t5.PixArtTokenizer()
    stream = getattr(tokenizer, tokenizer.clip)

    def rows(text: str) -> int:
        chunks = tokenizer.tokenize_with_weights(text)[tokenizer.clip_name]
        assert len(chunks) == 1, (
            "chroma text stream seam: the PixArt T5 tokenizer now splits a "
            f"prompt into {len(chunks)} chunks, so one chunk's length is no "
            "longer the row count the model reads")
        return len(chunks[0])

    assert (stream.min_length, stream.pad_to_max_length) == (1, False), (
        "chroma text stream seam: the PixArt T5 tokenizer now pads to "
        f"min_length {stream.min_length} with pad_to_max_length "
        f"{stream.pad_to_max_length}; a floor above one row would equalize "
        "prompt lengths the sharded forward reads at their own lengths today")
    assert rows("") == 1, (
        "chroma text stream seam: the blank negative no longer reaches chroma "
        f"as one row but as {rows('')}, so the canonical blank case and the "
        "pad rule written for it have both moved")
    negative = ("low quality, blurry, watermark, text artifacts, oversaturated "
                "colors, deformed anatomy, noisy background")
    raw = len(stream.tokenizer(negative)["input_ids"])
    assert rows(negative) == raw, (
        f"chroma text stream seam: a {raw}-token prompt now reaches the model "
        f"as {rows(negative)} rows, so comfy appends rows of its own and the "
        "extra ones carry no mask that would tell the model to ignore them")
    assert rows(negative) != rows(negative + " and a tiger in natural light"), (
        "chroma text stream seam: two prompts of different length now reach "
        "the model at one row count, so the family no longer takes two model "
        "calls for an unequal pair")

    flux_stream = comfy.text_encoders.flux.FluxTokenizer().t5xxl
    assert flux_stream.min_length == 256, (
        "chroma text stream seam: flux's own T5 floor moved from 256 to "
        f"{flux_stream.min_length}; that floor is the contrast that makes "
        "chroma the flux-family member whose two conds keep different shapes")

    default = inspect.signature(
        comfy.text_encoders.sd3_clip.T5XXLModel.__init__
    ).parameters["attention_mask"].default
    assert default is False, (
        "chroma text stream seam: the T5XXL encoder now defaults "
        f"attention_mask to {default!r}, so the chroma path may attach an "
        "encoder mask the sharded attention would have to honor")
    assert "attention_mask" not in inspect.getsource(
            comfy.text_encoders.pixart_t5), (
        "chroma text stream seam: the PixArt T5 path now names attention_mask, "
        "so it may turn the encoder mask on where it was off")
    assert "attention_mask_img_shape" in inspect.getsource(
            comfy.model_base.Flux.extra_conds), (
        "chroma text stream seam: the flux extra_conds no longer gates the "
        "attention mask on a regional conditioning shape, so a mask may reach "
        "chroma's forward from somewhere else")
    assert 'attn_mask=kwargs.get("attention_mask", None)' in inspect.getsource(
            comfy.ldm.chroma.model.Chroma._forward), (
        "chroma text stream seam: chroma's forward no longer takes its mask "
        "from the attention_mask cond alone, so the sharded refusal may now "
        "miss a mask the stock path applies")


def _assert_cfg_combine_seam() -> None:
    """The order of a folded cond/uncond pair, and how comfy combines the two.

    The benchmark matrix scorer reduces a pair a leg hands back the way the
    sampler reduces it (benchmark/gates.py, scored_latent), so it depends on
    two facts: the pair goes in cond first and uncond second, and the combine is
    uncond + (cond - uncond) * cfg. A drift in either turns a matrix number
    quietly wrong while every cell still reports one.
    """
    import inspect
    import os
    import sys

    import comfy.samplers
    import torch

    benchmark = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmark")
    if benchmark not in sys.path:
        sys.path.insert(0, benchmark)
    from gates import scored_latent

    source = inspect.getsource(comfy.samplers.sampling_function)
    assert "conds = [cond, uncond_]" in source, (
        "cfg combine seam: sampling_function no longer orders the pair cond "
        "first, so the scorer's first rows may be the uncond leg")
    assert "cfg_function(model, out[0], out[1], cond_scale" in source, (
        "cfg combine seam: the combine no longer takes the batched outputs in "
        "the order the conds went in")

    cond, uncond = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    combined = comfy.samplers.cfg_function(
        None, cond, uncond, 3.5, torch.zeros_like(cond), torch.zeros(1),
        model_options={})
    assert torch.equal(combined, uncond + (cond - uncond) * 3.5), (
        "cfg combine seam: cfg_function no longer combines the pair as "
        "uncond + (cond - uncond) * cfg")
    scored, form = scored_latent(torch.cat([cond, uncond]), 1, 3.5)
    assert torch.equal(scored, combined), (
        "cfg combine seam: the matrix scorer's own combine no longer matches "
        "comfy's, so a scored cfg pair is not the latent the sampler kept")
    assert "cfg 3.5" in form, (
        "cfg combine seam: the scorer stopped naming the cfg it combined at, "
        "which is the only record a report row carries of what it scored")


def _assert_loader_model_folder_seam() -> None:
    """Hold the models subfolder each stock loader reads its widget against.

    example_workflows/artifacts.toml writes one dest per artifact and
    tools/check_artifacts.py reads it back, so that column is a claim about
    comfy. A renamed folder kind, or a loader that starts resolving against a
    different one, would stage every file of that kind in the wrong place with
    nothing else going red. VAELoader's pixel_space entry is the other half of
    the claim: it is a dropdown name with no file behind it, which is why the
    manifest leaves it out.
    """
    import folder_paths
    import nodes
    from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader

    for kind in ("diffusion_models", "loras", "text_encoders", "clip_vision",
                 "vae", "latent_upscale_models"):
        assert kind in folder_paths.folder_names_and_paths, (
            "loader model folders seam: comfy no longer registers the models "
            f"folder {kind!r} the manifest writes as a dest"
        )
    reads = (
        (nodes.CLIPLoader.load_clip, "text_encoders"),
        (nodes.DualCLIPLoader.load_clip, "text_encoders"),
        (nodes.CLIPVisionLoader.load_clip, "clip_vision"),
        (nodes.VAELoader.load_vae, "vae"),
        (LatentUpscaleModelLoader.execute, "latent_upscale_models"),
    )
    for method, kind in reads:
        assert f'"{kind}"' in inspect.getsource(method), (
            f"loader model folders seam: {method.__qualname__} no longer "
            f"resolves its filename against {kind!r}, so the manifest dest "
            "for every artifact that loader names is wrong"
        )
    assert "pixel_space" in nodes.VAELoader.vae_list(nodes.VAELoader), (
        "loader model folders seam: VAELoader stopped offering the synthetic "
        "pixel_space entry, so the templates that pick it now name a file the "
        "manifest has no row for"
    )


def _assert_fsdp_shard_size_seam() -> None:
    """The two numbers comfy sizes a load by, and which one the block loop reads.

    A sharded model holds 1/world of its weights on each rank, and comfy has to
    be told twice or it moves the wrong amount. ``ModelPatcher.model_size()``
    returns the cached ``size`` the FSDP wrap writes, and ``partially_load``
    compares that against the memory it was offered to pick the full path. The
    block loop inside ``load`` then charges every module whatever
    ``comfy.model_management.module_size`` returns, resolved at call time, and
    a DTensor answers ``nbytes`` for the whole logical tensor. With only the
    first number set, a rank that held 30.9 GiB was charged 61.7 GiB and
    offloaded most of what it already had (2026-09-07).
    """
    import comfy.model_management
    import comfy.model_patcher
    import comfy.ops
    import torch

    class _Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = comfy.ops.manual_cast.Linear(256, 256, bias=False)
            self.second = comfy.ops.manual_cast.Linear(256, 256, bias=False)

    class _Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.diffusion_model = _Inner()

        def model_dtype(self):
            return torch.float32

    cpu = torch.device("cpu")

    def patcher():
        return comfy.model_patcher.ModelPatcher(
            _Base(), load_device=cpu, offload_device=cpu)

    # 1. module_size is the state dict's own byte sum. That is why a shard-aware
    #    replacement can be one function and change no other model's answer.
    whole = patcher()
    entries = whole.model.state_dict()
    assert entries and comfy.model_management.module_size(whole.model) == sum(
        tensor.nbytes for tensor in entries.values()), (
        "fsdp shard size seam: module_size is no longer the state dict's nbytes sum")

    # 2. model_size() answers with the cached size whenever one is set, which is
    #    the attribute the wrap writes this rank's local bytes into.
    assert whole.model_size() > 0, "fsdp shard size seam: model_size measured nothing"
    whole.size = 4096
    assert whole.model_size() == 4096, (
        "fsdp shard size seam: model_size stopped honoring the cached size, so the "
        "wrap has nowhere to declare a shard")

    # 3. The rule partially_load decides by, run for real: memory over the model
    #    size takes the full path, and the block loop then offloads nothing.
    full = patcher()
    full.partially_load(cpu, extra_memory=full.model_size() * 2)
    assert full.model.model_lowvram is False, (
        "fsdp shard size seam: memory over the model size no longer forces a full load")

    # 4. And the block loop charges module_size per module, resolved at call
    #    time. Halving what it returns is what a shard needs: the same offered
    #    memory goes from a partial load to a whole one.
    one_module = comfy.model_management.module_size(
        patcher().model.diffusion_model.first)
    offered = int(one_module * 1.2)
    partial = patcher()
    partial.partially_load(cpu, extra_memory=offered)
    assert partial.model.model_lowvram is True, (
        "fsdp shard size seam: comfy no longer offloads when the offered memory is "
        "under the modules' charged bytes")

    stock = comfy.model_management.module_size
    comfy.model_management.module_size = lambda module: stock(module) // 2
    try:
        halved = patcher()
        halved.partially_load(cpu, extra_memory=offered)
    finally:
        comfy.model_management.module_size = stock
    assert halved.model.model_lowvram is False, (
        "fsdp shard size seam: the block loop no longer reads "
        "comfy.model_management.module_size at call time, so a sharded model cannot "
        "be charged its local bytes")


def _assert_offload_pin_seam() -> None:
    """Why an FSDP base keeps its offload device on the compute device.

    A LoRA hot-swap hands ComfyUI a new clone whose patch uuid differs from the
    one the model's weights were loaded under. ``partially_load`` then unpatches
    the whole model to the patcher's ``offload_device`` before loading it again,
    and on unified memory that move is a full host copy of every weight that
    frees nothing (2026-09-28). ``adapters/fsdp.py`` pins the base's offload
    device to its load device, as the slab loader does, and relies on two
    things: ``clone()`` carries the pinned device to every hot-swap patcher, and
    the uuid mismatch moves the model to exactly that device.
    """
    import torch
    from comfy.model_patcher import ModelPatcher

    cpu = torch.device("cpu")
    model = torch.nn.Linear(8, 8, bias=False)
    model.device = cpu
    base = ModelPatcher(model, cpu, cpu, model.weight.numel() * model.weight.element_size())
    base.partially_load(cpu, 1e32)
    pinned = torch.device("cpu", 0)
    base.offload_device = pinned
    swap = base.clone()
    assert swap.offload_device == pinned, (
        "offload pin seam: clone() no longer carries the base's offload_device, "
        "so a hot-swap patcher would offload FSDP shards to the default device")
    swap.add_patches({"weight": (torch.zeros(8, 8),)}, 1.0)
    swap.patches.clear()
    assert swap.patches_uuid != model.current_weight_patches_uuid, (
        "offload pin seam: add_patches no longer gives a clone a new patch uuid; "
        "re-read issue #471, since the reload this pin prevents may be gone")
    moved = []
    to = model.to
    model.to = lambda *args, **kwargs: moved.append(args[0] if args else kwargs.get("device")) or to(*args, **kwargs)
    try:
        swap.partially_load(cpu, 1e32)
    finally:
        model.to = to
    assert moved and moved[0] == pinned, (
        "offload pin seam: a patch-uuid mismatch no longer unpatches the model to "
        f"the patcher's offload_device (moves seen: {moved}); the pin in "
        "adapters/fsdp.py may be guarding a path comfy no longer takes")
    # A stock hot-swap names the new patcher's uuid on the model instead
    # (actor/store_bake.py _adopt_patch_uuid): a match moves nothing.
    again = base.clone()
    again.add_patches({"weight": (torch.zeros(8, 8),)}, 1.0)
    again.patches.clear()
    model.current_weight_patches_uuid = again.patches_uuid
    moved.clear()
    model.to = lambda *args, **kwargs: moved.append(args[0] if args else kwargs.get("device")) or to(*args, **kwargs)
    try:
        again.partially_load(cpu, 1e32)
    finally:
        model.to = to
    assert not moved, (
        "offload pin seam: a matching patch uuid still moved the model "
        f"({moved}); the stock hot-swap's uuid adoption no longer avoids comfy's reload")
    # A drop detaches through the same unpatch, to that clone's offload device
    # (actor/resident_ledger.py unload_without_offload).
    model.to = lambda *args, **kwargs: moved.append(args[0] if args else kwargs.get("device")) or to(*args, **kwargs)
    try:
        again.detach(unpatch_all=True)
    finally:
        model.to = to
    assert moved == [pinned], (
        f"offload pin seam: detach no longer moves the model to the clone's offload_device ({moved}); "
        "a discarded stock resident may be copied to the host again")


def _assert_resident_ledger_seam() -> None:
    """The ledger a slab or an FSDP shard writes so comfy stops asking twice.

    ComfyUI sizes a sample-time load off two fields of the model object,
    ``model_loaded_weight_memory`` and ``model.device``, and a slab build or a
    shard build fills neither: the weights reach the compute device without
    comfy's own load ever running. Comfy then asks for a whole model of free
    memory for bytes the box already holds, cannot get it, and partially loads
    (2026-09-09). ``actor/resident_ledger.py`` answers by running comfy's own
    full load once, at build time, through the call comfy's manager makes
    under ``force_full_load``.

    Four things have to keep holding. An unpatched key costs no copy, so the
    build-time load moves nothing. That load writes the loaded weight memory,
    ``model_lowvram``, the device and the patch uuid. A second ``partially_load``
    on the loaded model takes the early return and never reaches ``load``
    again, which is what makes the sample-time ask cheap. And a patcher that
    still holds patches does not take that early return, which is why the
    ledger scopes a pending stack out instead of declaring it.
    """
    import comfy.model_management as mm
    import torch
    from comfy.model_patcher import ModelPatcher

    from dgx_monarch.actor import resident_ledger

    device = torch.device("cpu")
    model = torch.nn.Linear(8, 8, bias=False)
    # BaseModel carries this field; a bare module does not, and
    # LoadedModel.model_memory_required reads it through current_loaded_device.
    model.device = device
    size = model.weight.numel() * model.weight.element_size()
    patcher = ModelPatcher(model, device, device, size)

    assert patcher.model.model_loaded_weight_memory == 0, (
        "resident ledger seam: a fresh ModelPatcher no longer starts with an "
        "empty weight ledger, so the zero this change fills is not the zero "
        "comfy reads"
    )
    weight = patcher.patch_weight_to_device("weight")
    assert weight is model.weight, (
        "resident ledger seam: patch_weight_to_device no longer returns the live "
        "weight untouched for a key with no patch, so the build-time full load "
        "would copy every weight it walks"
    )

    assert resident_ledger.declare(patcher, "canary") is None, (
        "resident ledger seam: the pack declined to declare a plain resident patcher"
    )
    assert patcher.model.model_loaded_weight_memory == patcher.model_size(), (
        "resident ledger seam: comfy's full load no longer records the model's "
        "own size as the loaded weight memory"
    )
    assert patcher.model.model_lowvram is False, (
        "resident ledger seam: comfy's full load no longer clears model_lowvram, "
        "which is both the early return's gate and the partial-load guard's input"
    )
    assert patcher.model.device == device, (
        "resident ledger seam: comfy's load no longer moves the model's own device "
        "field to the load device, so its manager keeps pricing the whole model"
    )
    assert patcher.model.current_weight_patches_uuid == patcher.patches_uuid, (
        "resident ledger seam: comfy's load no longer stamps the patch uuid it "
        "loaded, so the next load would unpatch and reload these weights"
    )
    assert getattr(model, "comfy_patched_weights", False) is True, (
        "resident ledger seam: comfy's load no longer marks a loaded module, so "
        "its own partial unload could no longer find these weights"
    )

    assert patcher.clone().patches_uuid == patcher.patches_uuid, (
        "resident ledger seam: a clone no longer carries its parent's patch uuid, "
        "so the clone the sampler takes before sampling would unpatch and reload "
        "the weights the declaration just placed"
    )

    loaded = mm.LoadedModel(patcher)
    assert loaded.model_memory_required(device) == 0, (
        "resident ledger seam: comfy still prices a fully loaded model above zero, "
        "so declaring the residency would not lower the sample-time ask"
    )

    def refuse(*args, **kwargs):
        raise AssertionError(
            "resident ledger seam: partially_load reached load() on a model comfy "
            "already reports as fully loaded, so a declared slab or shard would be "
            "walked again at sample time"
        )

    patcher.load = refuse  # type: ignore[method-assign]
    assert patcher.partially_load(device, 1 << 20) == 0, (
        "resident ledger seam: partially_load no longer returns zero for a model "
        "it already holds"
    )

    reloaded: list[tuple] = []
    patcher.load = lambda *a, **k: reloaded.append(a)  # type: ignore[method-assign]
    patcher.add_patches({"weight": (torch.zeros_like(model.weight),)}, 1.0)
    patcher.partially_load(device, 1 << 20)
    assert reloaded, (
        "resident ledger seam: a patcher holding pending weight patches now takes "
        "partially_load's already-loaded early return, so declaring a residency "
        "with an unbaked LoRA stack on it would silently drop the stack"
    )

def _calls_on_self(source: str, receiver: str | None) -> set:
    """Attribute names called on ``self`` (or on ``self.<receiver>``)."""
    import ast
    import textwrap

    called = set()
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        value = node.func.value
        if receiver is None:
            if isinstance(value, ast.Name) and value.id == "self":
                called.add(node.func.attr)
        elif (isinstance(value, ast.Attribute) and value.attr == receiver
                and isinstance(value.value, ast.Name) and value.value.id == "self"):
            called.add(node.func.attr)
    return called


def _unet_class_of(model_class) -> object:
    """The one ``unet_model=`` class a comfy ``model_base`` class builds."""
    import ast
    import importlib
    import textwrap

    source = inspect.getsource(model_class.__init__)
    dotted = [
        ast.unparse(node.value)
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.keyword) and node.arg == "unet_model"
    ]
    assert len(dotted) == 1, (
        f"outside forward modules seam: {model_class.__name__} no longer builds "
        f"its diffusion model from one unet_model class, got {dotted}"
    )
    module_path, _, member = dotted[0].rpartition(".")
    return getattr(importlib.import_module(module_path), member)


def _assigns_on_self(model_class, attr: str) -> bool:
    """Whether any class in the model's own chain assigns ``self.<attr>``."""
    import ast
    import textwrap

    for klass in model_class.__mro__:
        if klass.__module__.split(".")[0] != "comfy":
            continue
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(klass)))):
            if not isinstance(node, ast.Attribute) or node.attr != attr:
                continue
            if isinstance(node.ctx, ast.Store) and isinstance(
                    node.value, ast.Name) and node.value.id == "self":
                return True
    return False


def _self_module_paths(source: str) -> set:
    """Dotted module paths a method reads off ``self``.

    ``self.<attr>`` gives ``attr``; a constant subscript,
    ``self.<attr>["<key>"]``, names the child ``attr.key``; a variable
    subscript, ``self.<attr>[mode]``, names the whole container. Methods and
    plain attributes come out too; the caller keeps the paths that resolve
    to a module holding parameters.
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(source))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    paths = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            continue
        parent = parents.get(node)
        if (isinstance(parent, ast.Subscript) and parent.value is node
                and isinstance(parent.slice, ast.Constant) and isinstance(parent.slice.value, str)):
            paths.add(f"{node.attr}.{parent.slice.value}")
        else:
            paths.add(node.attr)
    return paths


def _covers(declared: str, touched: str) -> bool:
    """Whether ``declared`` is a dotted-segment prefix of ``touched``."""
    head = tuple(declared.split("."))
    return tuple(touched.split("."))[:len(head)] == head


def _flux3_outside_forward_call(model) -> None:
    """What Flux3.extra_conds runs: preprocess_conds with text and references."""
    import torch

    generator = torch.Generator().manual_seed(33)
    context = torch.randn(1, 80, 20480, generator=generator) * 0.1
    refs = [torch.randn(1, 128, 2, 3, generator=generator)]
    with torch.no_grad():
        model.preprocess_conds(context, refs)


# Class-scoped census rows (adapters/fsdp_islands.py) and the tiny real
# diffusion model plus outside-forward call each is checked against.
_CLASS_SCOPED_PROBES: dict = {
    "Flux3": (lambda: _tiny_flux3(), _flux3_outside_forward_call),
}


def _assert_class_scoped_outside_forward_row(class_name, paths, methods, flat) -> None:
    """One class-scoped row against its method source and a live call.

    The reader collects the module paths the extra_conds methods read off
    ``self`` and keeps those that own parameters on a tiny real model; the
    live call puts a forward pre-hook on every descendant that owns
    parameters and records the dotted paths that ran. Each set must be
    covered by the row, and every row path must cover something in each,
    under the segment prefix rule. The flat census never holds these names.
    """
    import torch

    from dgx_monarch.adapters.fsdp_islands import resolve_module_path

    assert class_name in _CLASS_SCOPED_PROBES, (
        f"outside forward modules seam: no tiny real model probes the {class_name} row")
    build, call = _CLASS_SCOPED_PROBES[class_name]
    model = build()
    read = set()
    for method_name in methods:
        method = getattr(type(model), method_name, None)
        assert method is not None, (
            f"outside forward modules seam: {type(model).__name__} has no {method_name}")
        for path in _self_module_paths(inspect.getsource(method)):
            module = resolve_module_path(model, path)
            if isinstance(module, torch.nn.Module) and any(True for _ in module.parameters()):
                read.add(path)
    ran: set = set()
    names = {module: name for name, module in model.named_modules()}
    handles = [
        module.register_forward_pre_hook(lambda m, _args: ran.add(names[m]))
        for module in names
        if any(True for _ in module.parameters(recurse=False))
    ]
    try:
        call(model)
    finally:
        for handle in handles:
            handle.remove()
    for label, touched in (("reads", read), ("runs", ran)):
        uncovered = sorted(t for t in touched if not any(_covers(p, t) for p in paths))
        assert not uncovered, (
            f"outside forward modules seam: {class_name}.extra_conds {label} "
            f"{uncovered} outside its forward and the class-scoped row misses them")
        stale = sorted(p for p in paths if not any(_covers(p, t) for t in touched))
        assert not stale, (
            f"outside forward modules seam: the {class_name} row names {stale}, which "
            f"extra_conds no longer {label.rstrip('s')}s outside the forward")
    assert not set(paths) & set(flat), (
        f"outside forward modules seam: class-scoped {class_name} paths entered the "
        "flat census, which replicates them for every family")


def _assert_outside_forward_module_seam() -> None:
    """Hold the census of modules a family runs outside its denoise forward.

    comfy runs ``BaseModel.extra_conds`` once per sampling, ahead of the
    denoise loop, so a diffusion-model module it reaches there runs with no
    FSDP2 pre-forward hook: sharded, its weight is a DTensor meeting a plain
    text state and every rank dies on the first addmm (2026-09-09).
    ``adapters/fsdp_islands.py`` names those modules per family and the wrap
    replicates them instead. Both directions are held here, because either
    drift is silent: a family that starts reaching a module the table misses
    crashes the next FSDP render of it, and a row that names a module comfy
    stopped running there replicates bytes for nothing. The exclusion table
    beside it is held the same way: a name is not unique to the family that
    declares it, and a family that owns one on its own forward path must not
    have it replicated.
    """
    import comfy.model_base

    from dgx_monarch.adapters import ADAPTERS
    from dgx_monarch.adapters.detect import model_base_touchpoints
    from dgx_monarch.adapters.fsdp_islands import (
        AUXILIARY_MODULE_ATTRS,
        CLASS_OUTSIDE_FORWARD_MODULE_PATHS,
        FORWARD_PATH_NAME_OWNERS,
        OUTSIDE_FORWARD_MODULE_ATTRS,
        REFUSED_OUTSIDE_FORWARD_CALLS,
        outside_forward_module_attrs,
    )

    supported = model_base_touchpoints(ADAPTERS)
    reaching: dict = {}
    for name in supported:
        model_class = getattr(comfy.model_base, name, None)
        if model_class is None:
            continue
        for klass in model_class.__mro__:
            extra_conds = klass.__dict__.get("extra_conds")
            if extra_conds is None:
                continue
            called = _calls_on_self(inspect.getsource(extra_conds), "diffusion_model")
            if called:
                reaching.setdefault(klass.__name__, set()).update(called)
    # A refused row names methods an input reaches that the driver refuses off
    # one GPU; a tree whose model lacks every such method (before ComfyUI
    # 60aade0bb for MOSSImage) reaches none of them, and the row is inert there.
    live_refused: set = set()
    for class_name, methods in REFUSED_OUTSIDE_FORWARD_CALLS.items():
        assert class_name in supported, (
            f"outside forward modules seam: the refused row {class_name} is no longer a "
            "supported comfy model_base class")
        assert class_name not in OUTSIDE_FORWARD_MODULE_ATTRS and (
            class_name not in CLASS_OUTSIDE_FORWARD_MODULE_PATHS), (
            f"outside forward modules seam: {class_name} is both refused and replicated")
        unet_class = _unet_class_of(getattr(comfy.model_base, class_name))
        present = {method for method in methods if hasattr(unet_class, method)}
        assert reaching.get(class_name, set()) == present, (
            f"outside forward modules seam: {class_name}.extra_conds now calls "
            f"{sorted(reaching.get(class_name, ()))} on the diffusion model, and the refused "
            f"row names {sorted(present)} of the methods this tree has")
        if present:
            live_refused.add(class_name)
    tabled = (set(OUTSIDE_FORWARD_MODULE_ATTRS) | set(CLASS_OUTSIDE_FORWARD_MODULE_PATHS)
              | live_refused)
    assert set(reaching) == tabled, (
        "outside forward modules seam: the supported families whose extra_conds "
        f"calls a diffusion-model method are {sorted(reaching)}, and the tables "
        f"name {sorted(tabled)}"
    )
    for class_name, attrs in OUTSIDE_FORWARD_MODULE_ATTRS.items():
        unet_class = _unet_class_of(getattr(comfy.model_base, class_name))
        touched: set = set()
        for method_name in reaching[class_name]:
            method = getattr(unet_class, method_name, None)
            assert method is not None, (
                f"outside forward modules seam: {unet_class.__name__} has no "
                f"{method_name}, which {class_name}.extra_conds calls"
            )
            touched |= _calls_on_self(inspect.getsource(method), None)
        assert touched == set(attrs), (
            f"outside forward modules seam: {class_name} runs {sorted(touched)} "
            f"outside its forward and the table names {sorted(attrs)}"
        )
        assert set(attrs) <= set(AUXILIARY_MODULE_ATTRS), (
            f"outside forward modules seam: the census tuple the wrap reads "
            f"misses {sorted(set(attrs) - set(AUXILIARY_MODULE_ATTRS))}"
        )
    for class_name, paths in CLASS_OUTSIDE_FORWARD_MODULE_PATHS.items():
        _assert_class_scoped_outside_forward_row(
            class_name, paths, reaching[class_name], AUXILIARY_MODULE_ATTRS)
    for class_name, owned in FORWARD_PATH_NAME_OWNERS.items():
        model_class = getattr(comfy.model_base, class_name, None)
        assert class_name in supported and model_class is not None, (
            f"outside forward modules seam: {class_name} is excluded from the "
            "census but is no longer a supported comfy model_base class"
        )
        assert class_name not in reaching, (
            f"outside forward modules seam: {class_name} now reaches "
            f"{sorted(reaching.get(class_name, ()))} from extra_conds, so it "
            "belongs in the census table rather than the exclusion table"
        )
        unet_class = _unet_class_of(model_class)
        for attr in owned:
            assert attr in AUXILIARY_MODULE_ATTRS, (
                f"outside forward modules seam: {class_name} is excluded from "
                f"{attr}, which no family declares any more"
            )
            assert _assigns_on_self(unet_class, attr), (
                f"outside forward modules seam: {unet_class.__name__} no longer "
                f"builds {attr}, so the exclusion for {class_name} is stale"
            )
        excluded = set(AUXILIARY_MODULE_ATTRS) - set(
            outside_forward_module_attrs(model_class.__new__(model_class)))
        assert excluded == set(owned), (
            f"outside forward modules seam: the wrap subtracts {sorted(excluded)} "
            f"for {class_name} and the table names {sorted(owned)}"
        )


def _assert_flux_family_text_stream_seam() -> None:
    """What flux2 and longcat hand the model, and what neither of them masks.

    The flux-family divisibility pad rows are excluded from attention for all
    four members since 2026-09-09, and the row counts that decides come from
    comfy rather than from this pack. Flux 1.x is covered by the chroma text
    stream seam, which holds its T5 floor of 256 and the encoder mask that
    stays off. The other two are held here.

    Flux2 front-pads every text cond to 512 rows in model_base and attaches no
    mask, so the stock forward attends the pad rows and a sharded one has to
    attend the same rows. LongCat pads and truncates every prompt to one width
    in its tokenizer and trims the chat template off both ends after encoding,
    which is why two prompts of different length still take one shape on this
    family. Both encoders do return an attention mask of their own, and neither
    turns it into a cond: `Flux.extra_conds` only forwards a mask that came
    from regional conditioning, which the sharded forward refuses outright. So
    no text row these families read is masked out, and the exclusion has only
    the divisibility pads to drop. That is the answer the chroma text stream
    seam holds for chroma (2026-09-06), on a different encoder.
    """
    import inspect

    import comfy.ldm.flux.model
    import comfy.model_base
    import comfy.text_encoders.longcat_image
    import comfy.text_encoders.qwen_image

    flux2_source = inspect.getsource(comfy.model_base.Flux2.extra_conds)
    assert "target_text_len = 512" in flux2_source, (
        "flux family text seam: flux2 no longer front-pads its text cond to "
        "512 rows in extra_conds, so the width the sharded forward shards is "
        "no longer fixed and the pad legs written for it name the wrong shape")
    assert "attention_mask" not in flux2_source, (
        "flux family text seam: flux2 extra_conds now names attention_mask, "
        "so it may hand the sharded forward a mask over its own front pad")
    assert "attention_mask" not in inspect.getsource(
            comfy.model_base.LongCatImage.extra_conds), (
        "flux family text seam: longcat extra_conds now names attention_mask, "
        "so a masked text row may reach the sharded forward")

    base = comfy.text_encoders.longcat_image.LongCatImageBaseTokenizer
    assert base(None).max_length == 512, (
        "flux family text seam: the longcat tokenizer's own pad width moved "
        "off 512, which is the number the family's shard math is written on")
    tokenizer = comfy.text_encoders.longcat_image.LongCatImageTokenizer()
    widths = {
        len(tokenizer.tokenize_with_weights(text)["qwen25_7b"][0])
        for text in ("", "a lighthouse on a rocky coast", "x " * 400)
    }
    assert len(widths) == 1, (
        "flux family text seam: longcat now hands the model different row "
        f"counts for different prompts ({sorted(widths)}), so its two conds no "
        "longer share one shape and the pad rule written for it has moved")
    trim = inspect.getsource(
        comfy.text_encoders.longcat_image.LongCatImageTEModel.encode_token_weights)
    assert "out = out[:, template_end:]" in trim and "out[:, :-suffix_len]" in trim, (
        "flux family text seam: longcat no longer trims the chat template off "
        "both ends after encoding, so the width the model reads is no longer "
        "the padded prompt alone")

    qwen = inspect.signature(
        comfy.text_encoders.qwen_image.Qwen25_7BVLIModel.__init__
    ).parameters["attention_mask"].default
    assert qwen is True, (
        "flux family text seam: the longcat encoder now defaults its "
        f"attention mask to {qwen!r}; the pad rows it masks reach neither a "
        "one-GPU forward nor a sharded one today, and that is why the sharded "
        "exclusion drops divisibility pads alone")
    assert 'attn_mask=kwargs.get("attention_mask", None)' in inspect.getsource(
            comfy.ldm.flux.model.Flux._forward), (
        "flux family text seam: the flux forward no longer takes its mask from "
        "the attention_mask cond alone, so the sharded refusal may now miss a "
        "mask the stock path applies")


def _assert_cogvideox_layout_seam() -> None:
    """Comfy builds CogVideoX from the flattened layout and from nothing else.

    The header sniff names `cogvideo` for both spellings a CogVideoX file ships
    under, and the diffusers spelling is a naming answer only: no loader can
    build that file. Detection keys on the flattened
    `blocks.0.norm1.linear.weight`. `load_diffusion_model_state_dict` tries
    three branches in turn, detection, the mmdit conversion and the legacy
    diffusers unet path, and all three answer nothing for the same weights
    under diffusers names. If comfy ever converts them, this seam goes red and
    docs/TROUBLESHOOTING.md #102 has to change.
    """
    import torch
    from comfy.model_detection import (
        convert_diffusers_mmdit,
        detect_unet_config,
        model_config_from_diffusers_unet,
        model_config_from_unet_config,
    )

    flattened = {
        "blocks.0.norm1.linear.weight": torch.zeros(18432, 512),
        "blocks.0.attn_out.weight": torch.zeros(3072, 3072),
        "patch_embed.proj.weight": torch.zeros(3072, 256),
        "patch_embed.text_proj.weight": torch.zeros(3072, 4096),
        "time_embedding_linear_1.weight": torch.zeros(512, 3072),
        "ofs_embedding_linear_1.weight": torch.zeros(512, 512),
        "proj_out.weight": torch.zeros(128, 3072),
    }
    config = detect_unet_config(flattened, "")
    assert config is not None and config["image_model"] == "cogvideox", (
        "cogvideox layout seam: the flattened block norm no longer routes a "
        f"CogVideoX checkpoint to the cogvideox config: {config}")
    assert (config["in_channels"], config["patch_size_t"], config["ofs_embed_dim"]) \
        == (32, 2, 512), (
        "cogvideox layout seam: the 1.5 image-to-video shape no longer reads "
        f"off the patch embed and the OFS embedding: {config}")
    model_config = model_config_from_unet_config(config, flattened)
    assert type(model_config).__name__ == "CogVideoX_I2V", (
        "cogvideox layout seam: 32 input channels no longer select the "
        f"image-to-video surface: {type(model_config).__name__}")

    renames = {
        "blocks.0.norm1.linear.weight": "transformer_blocks.0.norm1.linear.weight",
        "blocks.0.attn_out.weight": "transformer_blocks.0.attn1.to_q.weight",
        "time_embedding_linear_1.weight": "time_embedding.linear_1.weight",
        "ofs_embedding_linear_1.weight": "ofs_embedding.linear_1.weight",
    }
    diffusers = {renames.get(key, key): value for key, value in flattened.items()}
    assert detect_unet_config(diffusers, "") is None, (
        "cogvideox layout seam: detection now answers for the diffusers "
        "spelling, so the sniff row for it names a file a loader can build")
    assert convert_diffusers_mmdit(diffusers, "") is None, (
        "cogvideox layout seam: there is now a diffusers conversion for this "
        "family, so the operator remedy of converting the file by hand is stale")
    assert model_config_from_diffusers_unet(diffusers) is None, (
        "cogvideox layout seam: the last branch of the diffusion model load "
        "now answers for the diffusers spelling, so a loader can reach a "
        "config for that file after all")


def _assert_media_load_inputs_seam() -> None:
    """The input each stock media loader names its file on, and the refusal.

    The sweep matrix prunes a template whose load nodes name media the input
    folder does not hold, because comfy rejects such a graph before anything
    loads and every cell derived from it could only reject (2026-09-09). The
    pruning depends on two contracts: the input name each loader carries the
    file on, and that a name outside the folder is refused rather than loaded.
    Both are read here off the installed comfy, and the matrix's own map is
    held to them.
    """
    import folder_paths
    import nodes
    from benchmark.sweep.matrix import MEDIA_INPUTS, missing_media
    from comfy_extras.nodes_audio import LoadAudio
    from comfy_extras.nodes_video import LoadVideo

    input_dir = folder_paths.get_input_directory()
    present = sorted(
        name for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name)))
    assert present, (
        "media load seam: the comfy input folder is empty, so this seam proves "
        "nothing about which names a loader accepts")

    for name in ("LoadImage", "LoadImageMask"):
        node = nodes.NODE_CLASS_MAPPINGS[name]
        required = node.INPUT_TYPES()["required"]
        assert MEDIA_INPUTS[name] in required, (
            f"media load seam: {name} no longer names its file on "
            f"{MEDIA_INPUTS[name]!r}, so the matrix reads the wrong input and "
            "prunes nothing")
        options = required[MEDIA_INPUTS[name]][0]
        assert isinstance(options, list) and set(options) <= set(present), (
            f"media load seam: {name} no longer builds its options from the "
            "input folder, so absence there no longer predicts a rejection")

    for node, expected in ((LoadVideo, "file"), (LoadAudio, "audio")):
        inputs = node.define_schema().inputs
        names = [getattr(entry, "id", None) or getattr(entry, "name", None)
                 for entry in inputs]
        assert MEDIA_INPUTS[node.__name__] == expected and expected in names, (
            f"media load seam: {node.__name__} names its file on {names}, not "
            f"{expected!r}, so the matrix reads the wrong input")

    # The refusal itself, through comfy's own validator: a name the folder does
    # not hold is an error string, and one it holds is accepted.
    absent = "dgxm_seam_absent_media.png"
    assert not folder_paths.exists_annotated_filepath(absent), (
        "media load seam: this fixture name must not exist for the check below")
    assert isinstance(nodes.LoadImage.VALIDATE_INPUTS(absent), str), (
        "media load seam: LoadImage no longer refuses a file the input folder "
        "does not hold, so a graph naming user media would load instead")
    assert nodes.LoadImage.VALIDATE_INPUTS(present[0]) is True, (
        "media load seam: LoadImage refuses a file the input folder does hold, "
        "so the matrix would prune a template that runs")

    # The consumer, on the same folder: a present name claims nothing, an
    # absent one is named.
    graph = {"1": {"class_type": "LoadImage", "inputs": {"image": present[0]}},
             "2": {"class_type": "LoadVideo", "inputs": {"file": absent}}}
    assert missing_media(graph, Path(input_dir)) == [absent], (
        "media load seam: the matrix no longer names exactly the file the "
        "input folder is missing")


def _assert_waiver_history_seam() -> None:
    """A waived sampler output keeps its latent and joins only its own history.

    The model call is replaced with a tiny latent result. The real production
    sampler, Comfy result parser and prompt-history queue, and sweep history
    reader carry the metadata across the boundary being checked.
    """
    from types import SimpleNamespace
    from unittest.mock import patch

    import execution
    import torch
    from benchmark.sweep.driver import waiver_run_ids

    from dgx_monarch.accuracy_waiver import STAMPED_RESULT_KEY
    from dgx_monarch.nodes import samplers

    samples = torch.zeros((1, 4, 2, 2))
    node = samplers.DGXMonarchKSampler()
    queue = execution.PromptQueue(SimpleNamespace(queue_updated=lambda: None))
    variants = (
        ("normal", [], []),
        ("waived", [{"run_id": "current-dispatch", "guard": "ring_pad"}], ["current-dispatch"]),
        ("inherited", [{"run_id": "ancestor-dispatch", "guard": "ring_pad", "inherited": True}], []),
        ("foreign", [{"run_id": "foreign-dispatch", "guard": "ring_pad"}], ["foreign-dispatch"]),
    )
    for index, (prompt_id, stamps, expected_ids) in enumerate(variants):
        latent = {"samples": samples}
        if stamps:
            latent[STAMPED_RESULT_KEY] = stamps
        with patch.object(samplers, "run_render", return_value=latent):
            returned = node.sample(None, 0, 1, 1.0, "euler", "simple", [], [], {"samples": samples})
        outputs, ui, expanded = execution.get_output_from_returns([returned], node)
        assert not expanded, "waiver history: a sampler unexpectedly expanded a graph"
        assert outputs[0][0]["samples"] is samples, "waiver history: sampler latent values changed"
        assert isinstance(returned, dict) == bool(expected_ids), (
            "waiver history: unwaived or inherited-only output changed its tuple contract")
        queue.put((index, prompt_id, {}, {}, []))
        _item, task_id = queue.get(timeout=0)
        queue.task_done(task_id, {"outputs": {"sampler": ui}},
                        execution.PromptQueue.ExecutionStatus("success", True, []))
        history = queue.get_history(prompt_id)
        assert waiver_run_ids(history[prompt_id]) == expected_ids, (
            "waiver history: current dispatch IDs did not survive the owning prompt history")
    assert waiver_run_ids(queue.get_history("waived")["waived"]) == ["current-dispatch"], (
        "waiver history: another prompt's dispatch polluted the original history")


def _assert_lumina_attention_container_seam() -> None:
    """NextDiT._forward's ming-image infix, and comfy's real container wrap.

    Comfy 3b4c0b0e (ming-image support) inserted direct_context and
    ref_frames before transformer_options; comfy_rebound_signatures admits
    both shapes through optional_infix (optional_trailing cannot express an
    insertion that is not trailing). NextDiTPixelSpace never gained the infix
    and keeps its plain contract. Comfy 1568e6cf wraps lumina's q, k and v in
    AttentionTensorContainer. Where that class exists, this also drives one
    real q/k/v transaction through comfy's own ``optimized_attention`` and
    ``wrap_attn`` to prove the ``usp_options`` override still receives plain
    tensors rather than wrapped containers: a container that got through would
    fall through to unsharded math under USP with no error. Where the class is
    absent, only the signature pin runs.
    """
    from comfy.ldm.lumina import model as comfy_lumina

    from dgx_monarch.adapters.base import usp_options

    def explicit(method):
        return tuple(
            name
            for name, parameter in inspect.signature(method).parameters.items()
            if parameter.kind is not inspect.Parameter.VAR_KEYWORD
        )

    base = ("self", "x", "timesteps", "context", "num_tokens", "attention_mask",
            "ref_latents", "ref_contexts", "siglip_feats", "transformer_options")
    infixed = ("self", "x", "timesteps", "context", "num_tokens", "attention_mask",
               "ref_latents", "ref_contexts", "siglip_feats", "direct_context",
               "ref_frames", "transformer_options")
    inner = explicit(comfy_lumina.NextDiT._forward)
    assert inner in (base, infixed), (
        f"lumina attention container seam: NextDiT._forward now takes {inner}, "
        "which is neither admitted shape; comfy_rebound_signatures needs a new row"
    )
    assert explicit(comfy_lumina.NextDiTPixelSpace._forward) == base, (
        "lumina attention container seam: NextDiTPixelSpace._forward gained "
        "ming-image params comfy_rebound_signatures does not admit for it"
    )

    import comfy.ldm.modules.attention as attention_module

    container_cls = getattr(attention_module, "AttentionTensorContainer", None)
    if container_cls is None:
        return  # this comfy predates the container seam

    import torch

    q = torch.randn(1, 3, 8)
    k = torch.randn(1, 3, 8)
    v = torch.randn(1, 3, 8)
    received = {}

    def fake_usp_attention(*args, **kwargs):
        received["args"] = args
        return torch.zeros(1, 3, 8)

    options = usp_options({}, fake_usp_attention)
    out = attention_module.optimized_attention(
        container_cls(q), container_cls(k), container_cls(v), 2,
        transformer_options=options,
    )
    assert torch.equal(out, torch.zeros(1, 3, 8)), (
        "lumina attention container seam: usp_options' override did not run "
        "through comfy's real AttentionTensorContainer wrap_attn"
    )
    got_q, got_k, got_v, got_heads = received["args"]
    assert isinstance(got_q, torch.Tensor) and torch.equal(got_q, q), (
        "lumina attention container seam: the sharded dispatcher received a "
        "container instead of a plain tensor -- comfy would silently fall "
        "through to unsharded math under USP"
    )
    assert torch.equal(got_k, k) and torch.equal(got_v, v) and got_heads == 2, (
        "lumina attention container seam: k/v/heads did not survive the unwrap intact"
    )



def _seeded(module, seed: int):
    """Fill a comfy module built with disable_weight_init, which leaves weights empty."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return module


def _assert_attention_keyword_handoff_seam() -> None:
    """What comfy hands the attention callables the pack installs.

    Comfy c194dd00c lets a model file pin an attention kernel per block, so
    models pass ``preferred_attention=self.comfy_attention``; comfy 8d534945e
    boxes q/k/v in AttentionTensorContainer. Cosmos ``Attention.
    compute_attention`` calls ``self.attn_op`` directly with both, and Anima
    replaces that op on every self-attention, so the replacement must take the
    containers and accept the keyword. On 2026-10-13 it did neither, and an
    Anima uly2 render died with a TypeError inside the worker's sample.

    Half one builds a real cosmos Attention, installs
    ``anima._make_usp_attn_op`` exactly as ``AnimaAdapter.inject_usp`` does,
    and drives ``forward`` through the real dispatcher an ordinary kernel
    gets, with comfy's own attention standing in for xFuser's, so the output
    must equal the stock op's bit for bit. That dispatcher drops a keyword it
    does not know, so the op must refuse one with a typed class P refusal
    instead: a keyword comfy starts passing fails here by name, and an op
    that lets one through fails the refusal check. Half two drives a real comfy CrossAttention, a call site that passes
    containers, ``attn_precision`` and ``preferred_attention``, through
    ``usp_options``' override, and pins the exact arguments ``wrap_attn``
    hands an override: a keyword that starts reaching the overrides fails here
    before it reaches a ring kernel's surface check on hardware.

    Half three covers the two families whose sequence-parallel forwards call
    comfy's native attention themselves, Qwen Image 2.1 and Wan-Animate 2. It
    pins each attention block to a stand-in kernel through a real
    ComfyAttention and checks that stock runs it, then that the injected
    forward on one rank never does and matches the unpinned stock forward bit
    for bit, with Qwen's cache-miss prefix fill included. A pin ran there
    under a TORCH_FLASH stamp, and the pack asked for no Sol-Attn waiver
    (issue #30). Half four (``_assert_stock_attention_sites_drop_pins``)
    holds the same contract for the stock attention the other forwards leave
    in place. Half five (``_assert_attention_pins_detected``) holds the
    detector the render record and the waiver read where no sequence
    parallelism drops a pin. Half six (``_assert_moss_image_pins_dropped``)
    holds MOSS-Image's block and text-block pins and its probe patterns.
    """
    import comfy.ldm.modules.attention as attention_module
    import comfy.ops
    import torch
    from comfy.ldm.cosmos import predict2

    from dgx_monarch.adapters.anima import _make_usp_attn_op
    from dgx_monarch.adapters.base import (
        UnsupportedModelError,
        _make_usp_attention_callable,
        usp_options,
    )
    from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag

    container_cls = getattr(attention_module, "AttentionTensorContainer", None)
    stock_takes_keyword = "preferred_attention" in inspect.signature(
        predict2.torch_attention_op).parameters
    operations = comfy.ops.disable_weight_init
    heads, head_dim = 2, 8
    x = torch.randn(1, 6, heads * head_dim, generator=torch.Generator().manual_seed(1))

    cosmos = _seeded(predict2.Attention(heads * head_dim, None, n_heads=heads,
                                        head_dim=head_dim, operations=operations), 0)
    stock = cosmos(x)

    kernel_calls = []

    def stub_kernel(_unused, q_l, k_l, v_l):
        # xFuser's place in the real dispatcher: (B, L, H, D) in and out.
        kernel_calls.append(1)
        b, s, h, d = q_l.shape
        out = attention_module.optimized_attention(
            q_l.transpose(1, 2), k_l.transpose(1, 2), v_l.transpose(1, 2), h,
            skip_reshape=True)
        return out.reshape(b, s, h, d)

    real_dispatcher = _make_usp_attention_callable(stub_kernel)
    dispatched = {}

    def recording_dispatcher(q, k, v, heads, skip_reshape=False, **kwargs):
        dispatched.update(q=q, k=k, v=v, heads=heads, skip_reshape=skip_reshape,
                          kwargs=dict(kwargs))
        return real_dispatcher(q, k, v, heads, skip_reshape=skip_reshape, **kwargs)

    handed = {}
    pack_op = _make_usp_attn_op(recording_dispatcher)

    def recording_op(q, k, v, **kwargs):
        handed.update(q=q, kwargs=dict(kwargs))
        return pack_op(q, k, v, **kwargs)

    cosmos.attn_op = recording_op  # the assignment AnimaAdapter.inject_usp makes
    try:
        pack = cosmos(x)
    except UnsupportedModelError as exc:
        raise AssertionError(
            "attention keyword handoff: comfy now hands attn_op "
            f"{sorted(handed.get('kwargs', {}))}, and Anima's op refuses what it "
            "does not apply; decide whether a sharded kernel can apply it, then "
            "adopt it") from exc

    assert stock_takes_keyword == ("preferred_attention" in handed["kwargs"]), (
        "attention keyword handoff: cosmos compute_attention and the stock "
        "torch_attention_op disagree on preferred_attention, so the keyword "
        "contract Anima's op mirrors has moved"
    )
    if stock_takes_keyword:
        assert handed["kwargs"]["preferred_attention"] is cosmos.comfy_attention, (
            "attention keyword handoff: compute_attention no longer passes the "
            "block's own ComfyAttention as preferred_attention"
        )
    if container_cls is not None:
        assert isinstance(handed["q"], container_cls), (
            "attention keyword handoff: cosmos forward stopped boxing q/k/v in "
            "AttentionTensorContainer before attn_op"
        )
    assert all(isinstance(dispatched[name], torch.Tensor) for name in "qkv"), (
        "attention keyword handoff: Anima's op passed a container to the "
        "sharded dispatcher instead of taking it"
    )
    assert dispatched["heads"] == heads and dispatched["skip_reshape"] is True, (
        "attention keyword handoff: Anima's op lost the (B, H, S, D) layout"
    )
    assert dispatched["kwargs"] == {}, (
        f"attention keyword handoff: comfy now hands attn_op {sorted(dispatched['kwargs'])} "
        "beyond transformer_options and preferred_attention; decide whether a "
        "sharded kernel can apply it"
    )
    assert torch.equal(stock, pack), (
        "attention keyword handoff: Anima's op under comfy's real cosmos "
        "Attention no longer matches the stock torch_attention_op"
    )

    def op_with_a_new_keyword(q, k, v, **kwargs):
        # Stands in for a keyword a later comfy adds to compute_attention.
        return pack_op(q, k, v, **kwargs, future_option=1)

    cosmos.attn_op = op_with_a_new_keyword
    kernel_calls.clear()
    refused = ""
    try:
        cosmos(x)
    except UnsupportedModelError as exc:
        refused = str(exc)
    tag = parse_leading_refusal_tag(refused)
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS, (
        "attention keyword handoff: Anima's op let a keyword it does not apply "
        "reach the dispatcher, which drops it for an ordinary kernel, instead "
        "of refusing with a typed class P refusal"
    )
    assert "'future_option'" in refused and not kernel_calls, (
        "attention keyword handoff: the refusal must name the keyword and land "
        "before any attention kernel runs"
    )

    cross = _seeded(attention_module.CrossAttention(heads * head_dim, heads=heads,
                                                   dim_head=head_dim,
                                                   attn_precision=torch.float32,
                                                   operations=operations), 2)
    stock = cross(x)

    received = {}

    def fake_sharded(q, k, v, heads, **kwargs):
        received.update(q=q, kwargs=dict(kwargs))
        return attention_module.optimized_attention(
            q, k, v, heads, attn_precision=kwargs.get("attn_precision"))

    options = usp_options({}, fake_sharded)
    override = options["optimized_attention_override"]

    def recording_override(func, *args, **kwargs):
        received.update(func=func, args=args, override_kwargs=dict(kwargs))
        return override(func, *args, **kwargs)

    options["optimized_attention_override"] = recording_override
    pack = cross(x, transformer_options=options)

    args = received["args"]
    assert callable(received["func"]) and len(args) == 4 and args[3] == heads, (
        "attention keyword handoff: wrap_attn no longer hands an override "
        "(func, q, k, v, heads) positionally"
    )
    assert all(isinstance(t, torch.Tensor) for t in args[:3]), (
        "attention keyword handoff: wrap_attn handed an override containers "
        "instead of taking them"
    )
    assert set(received["override_kwargs"]) == {
        "attn_precision", "transformer_options", "_inside_attn_wrapper"}, (
        "attention keyword handoff: wrap_attn now hands an override "
        f"{sorted(received['override_kwargs'])}; the pack's overrides forward "
        "unknown keywords to the dispatcher, whose ring checks refuse them"
    )
    assert received["override_kwargs"]["attn_precision"] is torch.float32, (
        "attention keyword handoff: attn_precision did not reach the override"
    )
    assert received["kwargs"] == {"attn_precision": torch.float32, "drop_rows": None}, (
        "attention keyword handoff: usp_options' override handed the dispatcher "
        f"{sorted(received['kwargs'])} rather than attn_precision and drop_rows"
    )
    assert torch.equal(stock, pack), (
        "attention keyword handoff: CrossAttention through usp_options' override "
        "no longer matches the stock call"
    )

    # Half three: Qwen Image 2.1 and Wan-Animate 2 call comfy's native
    # attention themselves (issue #30). A block pinned to a kernel must not run
    # it there, so the output equals an unpinned stock forward.
    import types

    from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
    from comfy.ldm.wan.model_animate2 import WanAnimate2Model

    from dgx_monarch.adapters import base as adapter_base
    from dgx_monarch.adapters import qwen_image21 as qwen21_adapter
    from dgx_monarch.adapters import wan_animate2 as animate2_adapter
    from dgx_monarch.adapters.base import InjectionContext

    pinned_runs = []

    def pinned_kernel(q, k, v, heads, *_args, skip_reshape=False,
                      skip_output_reshape=False, **_kwargs):
        # Stands in for comfy-kitchen Sol, which needs a GPU, and is told apart
        # from real attention by its zero output.
        pinned_runs.append(1)
        if isinstance(q, attention_module.AttentionTensorContainer):
            q, _k, _v = q.take(), k.take(), v.take()
        if skip_reshape and not skip_output_reshape:
            b, h, rows, d = q.shape
            return torch.zeros(b, rows, h * d, dtype=q.dtype)
        return torch.zeros_like(q)

    def pin(module):
        assert isinstance(module, attention_module.ComfyAttention), (
            "attention keyword handoff: a native-attention block no longer "
            "carries a ComfyAttention, so the per-block preference has moved"
        )
        module.config = {"attention": "comfy_kitchen_sol", "tau": 1.0}
        module.function = attention_module.wrap_attn(pinned_kernel)

    flash = types.SimpleNamespace(effective_kernel="TORCH_FLASH")

    generator = torch.Generator().manual_seed(45)
    qwen = QwenImage21Transformer2DModel(
        in_channels=64, out_channels=64, num_layers=1, attention_head_dim=64,
        num_attention_heads=1, context_in_dim=4, mlp_ratio=1,
        axes_dims_rope=(16, 16, 32), operations=operations)
    with torch.no_grad():
        for parameter in qwen.parameters():
            parameter.normal_(mean=0.0, std=0.02, generator=generator)
    q_x = torch.randn(2, 64, 1, 5, generator=generator)
    q_t = torch.tensor([0.25, 0.75])
    q_context = torch.randn(2, 3, 4, generator=generator)
    q_refs = [torch.randn(2, 64, 1, 2, generator=generator)]
    unpinned = qwen._forward(q_x, q_t, q_context, ref_latents=q_refs, image_slots=[1])
    for block in qwen.transformer_blocks:
        pin(block.attn.comfy_attention)
    pinned = qwen._forward(q_x, q_t, q_context, ref_latents=q_refs, image_slots=[1])
    assert pinned_runs and not torch.equal(pinned, unpinned), (
        "attention keyword handoff: stock Qwen Image 2.1 no longer runs a block's "
        "pinned kernel, so dropping the preference may no longer be needed"
    )

    saved_qwen = (adapter_base.sp_world, adapter_base.sp_rank, qwen21_adapter.sp_gather)
    adapter_base.sp_world = lambda: 1
    adapter_base.sp_rank = lambda: 0
    qwen21_adapter.sp_gather = lambda tensor, orig, dim=1: tensor.narrow(dim, 0, orig)
    try:
        qwen21_adapter.QwenImage21Adapter().inject_usp(
            qwen, InjectionContext(topology_sp=1, usp_attention=flash))
        pinned_runs.clear()
        uncached = qwen._forward(q_x, q_t, q_context, ref_latents=q_refs, image_slots=[1])
        # A cache miss fills the prefix through comfy's own block_causal_attention.
        qwen.current_patcher = types.SimpleNamespace(get_free_memory=lambda _device: 1 << 60)
        qwen.reset_prefix_cache(True)
        filled = qwen._forward(
            q_x, q_t, q_context, ref_latents=q_refs, image_slots=[1],
            transformer_options={"qwen_image21_cache": {"device": "gpu", "dtype": "default"}})
    finally:
        adapter_base.sp_world, adapter_base.sp_rank, qwen21_adapter.sp_gather = saved_qwen
    assert not pinned_runs, (
        "attention keyword handoff: the Qwen Image 2.1 sequence-parallel forward "
        "ran a block's pinned kernel under the widget's TORCH_FLASH stamp"
    )
    assert torch.equal(uncached, unpinned) and torch.equal(filled, unpinned), (
        "attention keyword handoff: the Qwen Image 2.1 sequence-parallel forward "
        "with a pinned block no longer matches the unpinned stock forward"
    )

    generator = torch.Generator().manual_seed(91)
    wan = WanAnimate2Model(
        patch_size=(1, 2, 2), text_len=4, in_dim=36, dim=8, ffn_dim=16,
        freq_dim=2, text_dim=4, out_dim=1, num_heads=1, num_layers=1,
        device="cpu", dtype=torch.float32, operations=operations).eval()
    with torch.no_grad():
        for parameter in wan.parameters():
            parameter.uniform_(-0.1, 0.1, generator=generator)
    w_x = torch.randn(1, 36, 3, 2, 4, generator=generator)
    w_pose = torch.randn(1, 16, 2, 2, 4, generator=generator)
    w_context = torch.randn(1, 4, 4, generator=generator)
    w_t = torch.tensor([0.7])
    with torch.no_grad():
        unpinned = wan._forward(w_x, w_t, w_context, pose_latents=w_pose)
        # Self-attention through the pack's native call, cross-attention
        # through the override the forward hands the stock module.
        for block in wan.blocks:
            pin(block.self_attn.comfy_attention)
            pin(block.cross_attn.comfy_attention)
        pinned_runs.clear()
        pinned = wan._forward(w_x, w_t, w_context, pose_latents=w_pose)
    assert pinned_runs and not torch.equal(pinned, unpinned), (
        "attention keyword handoff: stock Wan-Animate 2 attention no longer "
        "runs a block's pinned kernel, so dropping the preference may no longer be needed"
    )

    adapter_cls = animate2_adapter.WanAnimate2Adapter
    saved_wan = (animate2_adapter.sp_world, animate2_adapter.sp_rank,
                 adapter_cls.__dict__["_gather_spatial"])
    animate2_adapter.sp_world = lambda: 1
    animate2_adapter.sp_rank = lambda: 0
    adapter_cls._gather_spatial = staticmethod(lambda tensor, _frames, _pixels: tensor)
    try:
        adapter_cls().inject_usp(wan, InjectionContext(topology_sp=1, usp_attention=flash))
        pinned_runs.clear()
        with torch.no_grad():
            injected = wan._forward(w_x, w_t, w_context, pose_latents=w_pose)
    finally:
        animate2_adapter.sp_world, animate2_adapter.sp_rank = saved_wan[:2]
        adapter_cls._gather_spatial = saved_wan[2]
    assert not pinned_runs, (
        "attention keyword handoff: the Wan-Animate 2 sequence-parallel forward "
        "ran a block's pinned kernel under the widget's TORCH_FLASH stamp"
    )
    assert torch.equal(injected, unpinned), (
        "attention keyword handoff: the Wan-Animate 2 sequence-parallel forward "
        "with pinned attention no longer matches the unpinned stock forward"
    )

    _assert_stock_attention_sites_drop_pins(pin, pinned_runs, operations)
    _assert_attention_pins_detected(operations)
    _assert_moss_image_pins_dropped(pin, pinned_runs)


class _OneSpRank:
    """Stand in for xFuser's process groups as one sequence-parallel rank.

    The injected forwards shard, gather and read the Ulysses and ring degrees
    through ``xfuser.core.distributed``. On one rank a shard is the whole
    sequence and a gather is the identity, so the forward reaches comfy's
    modules exactly as stock does. The previous modules come back on exit.
    """

    _NAMES = ("xfuser", "xfuser.core", "xfuser.core.distributed")

    def __enter__(self):
        import types

        self.saved = {name: sys.modules.get(name) for name in self._NAMES}
        group = types.SimpleNamespace(all_gather=lambda tensor, dim: tensor)
        distributed = types.ModuleType("xfuser.core.distributed")
        distributed.get_sequence_parallel_world_size = lambda: 1
        distributed.get_sequence_parallel_rank = lambda: 0
        distributed.get_ulysses_parallel_world_size = lambda: 1
        distributed.get_ring_parallel_world_size = lambda: 1
        distributed.get_sp_group = lambda: group
        sys.modules["xfuser"] = types.ModuleType("xfuser")
        sys.modules["xfuser.core"] = types.ModuleType("xfuser.core")
        sys.modules["xfuser.core.distributed"] = distributed
        return self

    def __exit__(self, *_exc):
        for name, module in self.saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _assert_stock_attention_sites_drop_pins(pin, pinned_runs, operations) -> None:
    """Half four of the attention keyword handoff seam (issue #30).

    Under sequence parallelism the pack leaves some of comfy's attention in
    place: cross-attention against the replicated text, the refiners every
    rank runs whole, and the text refiners comfy runs from ``extra_conds``.
    The forwards hand those modules an override that ``wrap_attn`` checks
    before the block's preference (adapters/stock_attention.py). Tiny real
    models of Wan 2.1 (t2v and i2v), LTX, Kandinsky 5, Anima's cosmos DiT,
    Z-Image and Krea2 (pure Ulysses and ring) pin every such site: stock must
    run each pin, and the injected forward on one rank must run none and
    match the unpinned stock forward bit for bit. The sites the forward's
    options copy cannot reach get the same check through the adapter's own
    binding: WanDancer's music encoder, MiniMax H3's token refiner, the LTX
    audio-video text connector, and FLUX.3's early text and reference blocks,
    which comfy runs from ``preprocess_conds`` with no options; a tiny real
    FLUX.3 pins one of each beside an image block and a single block. Comfy's
    own attention takes the dispatcher's place.
    """
    import types

    import comfy.ldm.modules.attention as attention_module
    import torch
    from comfy.ldm.cosmos.predict2 import MiniTrainDIT
    from comfy.ldm.flux.layers import EmbedND
    from comfy.ldm.kandinsky5.model import Kandinsky5
    from comfy.ldm.krea2.model import SingleStreamDiT
    from comfy.ldm.lightricks.embeddings_connector import Embeddings1DConnector
    from comfy.ldm.lightricks.model import LTXVModel
    from comfy.ldm.lumina.model import NextDiT
    from comfy.ldm.minimax.model import TokenRefiner
    from comfy.ldm.wan.model import WanModel
    from comfy.ldm.wan.model_wandancer import MusicEncoderLayer

    from dgx_monarch.adapters.anima import AnimaAdapter
    from dgx_monarch.adapters.base import InjectionContext
    from dgx_monarch.adapters.flux3 import Flux3Adapter
    from dgx_monarch.adapters.kandinsky5 import Kandinsky5Adapter
    from dgx_monarch.adapters.krea2 import Krea2Adapter
    from dgx_monarch.adapters.ltx import LTXAdapter
    from dgx_monarch.adapters.minimax_h3 import MiniMaxH3Adapter
    from dgx_monarch.adapters.wan_family import WanAdapter
    from dgx_monarch.adapters.wan_variants import WanDancerAdapter
    from dgx_monarch.adapters.zimage import ZImageAdapter

    def one_rank_attention(q, k, v, heads, *args, drop_rows=None, kv_drop_rows=None,
                           sequence_order=None, query_bias_groups=None, **kwargs):
        # The sharded dispatcher's place on one rank: no pad rows, no reorder.
        assert not drop_rows and not kv_drop_rows and query_bias_groups is None
        return attention_module.optimized_attention(q, k, v, heads, *args, **kwargs)

    def check(name, model, sites, run, adapter, pure_ulysses=False):
        with torch.no_grad():
            unpinned = run()
            for module in sites:
                pin(module.comfy_attention)
            pinned_runs.clear()
            pinned = run()
            assert len(pinned_runs) >= len(sites) and not torch.equal(pinned, unpinned), (
                f"attention keyword handoff: stock {name} no longer runs a pinned "
                "kernel at every site, so the site list has moved")
            with _OneSpRank():
                adapter.inject_usp(model, InjectionContext(
                    topology_sp=1, usp_attention=one_rank_attention,
                    pure_ulysses=pure_ulysses))
                pinned_runs.clear()
                injected = run()
        assert not pinned_runs, (
            f"attention keyword handoff: the {name} sequence-parallel forward ran "
            "a pinned kernel in the stock attention it leaves in place")
        assert torch.equal(injected, unpinned), (
            f"attention keyword handoff: the {name} sequence-parallel forward with "
            "pinned stock attention no longer matches the unpinned stock forward")

    generator = torch.Generator().manual_seed(93)
    t = torch.tensor([0.6])

    def randn(*shape):
        return torch.randn(*shape, generator=generator)

    for kind in ("t2v", "i2v"):
        wan = _seeded(WanModel(
            model_type=kind, patch_size=(1, 2, 2), text_len=4, in_dim=4, dim=16,
            ffn_dim=32, freq_dim=2, text_dim=4, out_dim=4, num_heads=2, num_layers=2,
            device="cpu", dtype=torch.float32, operations=operations).eval(), 3)
        x, context = randn(1, 4, 2, 2, 4), randn(1, 4, 4)
        clip = randn(1, 3, 1280) if kind == "i2v" else None
        check(f"Wan 2.1 {kind}", wan, [block.cross_attn for block in wan.blocks],
              lambda wan=wan, x=x, context=context, clip=clip: wan._forward(
                  x, t, context, clip_fea=clip), WanAdapter())

    ltxv = _seeded(LTXVModel(
        in_channels=4, cross_attention_dim=16, attention_head_dim=8,
        num_attention_heads=2, caption_channels=8, num_layers=2, device="cpu",
        dtype=torch.float32, operations=operations).eval(), 11)
    x, context = randn(1, 4, 2, 2, 2), randn(1, 5, 8)
    check("LTX", ltxv, [block.attn2 for block in ltxv.transformer_blocks],
          lambda: ltxv._forward(x, t, context, None), LTXAdapter())

    kandinsky = _seeded(Kandinsky5(
        in_visual_dim=4, out_visual_dim=4, in_text_dim=8, in_text_dim2=8, time_dim=8,
        model_dim=16, ff_dim=32, visual_embed_dim=16, patch_size=(1, 2, 2),
        num_text_blocks=2, num_visual_blocks=2, axes_dims=(4, 2, 2), device="cpu",
        dtype=torch.float32, operations=operations).eval(), 5)
    x, context, pooled = randn(1, 4, 1, 4, 4), randn(1, 3, 8), randn(1, 8)
    check("Kandinsky 5", kandinsky,
          [block.self_attention for block in kandinsky.text_transformer_blocks]
          + [block.cross_attention for block in kandinsky.visual_transformer_blocks],
          lambda: kandinsky._forward(x, t, context, pooled), Kandinsky5Adapter())

    # Anima's own class adds only the llm_adapter, which runs before this
    # forward and holds no ComfyAttention; the adapter binds the DiT's forward.
    cosmos = _seeded(MiniTrainDIT(
        max_img_h=8, max_img_w=8, max_frames=1, in_channels=4, out_channels=4,
        patch_spatial=2, patch_temporal=1, concat_padding_mask=False,
        model_channels=24, num_blocks=2, num_heads=2, crossattn_emb_channels=8,
        pos_emb_cls="rope3d", use_adaln_lora=True, adaln_lora_dim=4, device="cpu",
        dtype=torch.float32, operations=operations).eval(), 6)
    x, context = randn(1, 4, 1, 8, 8), randn(1, 5, 8)
    check("Anima", cosmos, [block.cross_attn for block in cosmos.blocks],
          lambda: cosmos._forward(x, t, context), AnimaAdapter())

    zimage = _seeded(NextDiT(
        patch_size=2, in_channels=4, dim=24, n_layers=2, n_refiner_layers=1,
        n_heads=2, n_kv_heads=2, multiple_of=8, ffn_dim_multiplier=1.0, qk_norm=True,
        cap_feat_dim=8, axes_dims=(4, 4, 4), axes_lens=(64, 16, 16),
        time_scale=1000.0, pad_tokens_multiple=32, device="cpu", dtype=torch.float32,
        operations=operations).eval(), 8)
    x, context = randn(1, 4, 8, 8), randn(1, 5, 8)
    check("Z-Image", zimage,
          [layer.attention for layer in [*zimage.context_refiner, *zimage.noise_refiner]],
          lambda: zimage._forward(x, t, context, 5), ZImageAdapter())

    for pure in (True, False):
        krea = _seeded(SingleStreamDiT(
            features=32, tdim=8, txtdim=8, heads=2, kvheads=2, multiplier=1, layers=2,
            patch=1, channels=4, txtlayers=3, txtheads=2, txtkvheads=2, device="cpu",
            dtype=torch.float32, operations=operations).eval(), 9)
        x, context = randn(1, 4, 4, 4), randn(1, 5, 24)
        fusion = krea.txtfusion
        # Under ring or hybrid the refiner blocks carry the sharded override.
        blocks = [*fusion.layerwise_blocks, *(fusion.refiner_blocks if pure else ())]
        check(f"Krea2 ({'pure Ulysses' if pure else 'ring'})", krea,
              [block.attn for block in blocks],
              lambda krea=krea, x=x, context=context: krea._forward(x, t, context),
              Krea2Adapter(), pure_ulysses=pure)

    # FLUX.3: the early text and reference blocks run from preprocess_conds,
    # which hands them no options, so the adapter binds their attention; the
    # image and single blocks take the forward's sharded override.
    flux3 = _tiny_flux3(seed=12)
    x, context = randn(1, 128, 3, 4), randn(1, 80, 20480) * 0.1
    refs = [randn(1, 128, 2, 3)]
    check("FLUX.3", flux3,
          [flux3.txt_mode_blocks[0], flux3.content_mode_blocks["image_cond"][0],
           flux3.content_mode_blocks["image"][0], flux3.single_blocks[1]],
          lambda: flux3._forward(x, t, context, refs), Flux3Adapter(), pure_ulysses=True)

    # Sites the forward's options copy cannot reach. Each host carries only the
    # stock modules the adapter binds, and the adapter's own inject_usp binds them.
    music = [_seeded(MusicEncoderLayer(dim=16, num_heads=2, ffn_dim=32, device="cpu",
                                       dtype=torch.float32, operations=operations), 21 + i)
             for i in range(2)]
    ids = torch.arange(5, dtype=torch.float32).reshape(1, -1, 1)
    freqs = EmbedND(dim=8, theta=10000.0, axes_dim=[8])(ids).movedim(1, 2)
    feature = randn(1, 5, 16)

    def run_music():
        out = feature
        for layer in music:
            out = layer(out, freqs)
        return out

    check("WanDancer music encoder", types.SimpleNamespace(
        blocks=[], music_encoder=music,
        music_injector=types.SimpleNamespace(injector=[])),
        [layer.self_attn for layer in music], run_music, WanDancerAdapter())

    refiner = _seeded(TokenRefiner(2, 16, 2, 8, 32, 1e-6, 1e-6, 1e-6, device="cpu",
                                   dtype=torch.float32, operations=operations), 31)
    text = randn(5, 16)
    check("MiniMax H3 token refiner", types.SimpleNamespace(blocks=[], token_refiner=refiner),
          [block.attn for block in refiner.blocks], lambda: refiner(text),
          MiniMaxH3Adapter())

    class _ConnectorHost:
        # What LTXAdapter's audio-video branch reads; the block loop is not run.
        transformer_blocks = ()
        audio_inner_dim = 16

        def _process_transformer_blocks(self, *_args, **_kwargs):
            raise AssertionError("not run")

    host = _ConnectorHost()
    for name, seed in (("video_embeddings_connector", 41), ("audio_embeddings_connector", 42)):
        setattr(host, name, _seeded(Embeddings1DConnector(
            attention_head_dim=8, num_attention_heads=2, num_layers=2,
            num_learnable_registers=None, split_rope=True, double_precision_rope=True,
            device="cpu", dtype=torch.float32, operations=operations), seed))
    hidden = randn(1, 5, 16)
    connectors = (host.video_embeddings_connector, host.audio_embeddings_connector)
    check("LTX audio-video text connector", host,
          [block.attn1 for connector in connectors for block in connector.transformer_1d_blocks],
          lambda: torch.cat([connector(hidden)[0] for connector in connectors], dim=-1),
          LTXAdapter())


def _probe_tool():
    """tools/make_pin_probe_lora.py, loaded from the checkout (tools is no package)."""
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "tools" / "make_pin_probe_lora.py"
    spec = importlib.util.spec_from_file_location("make_pin_probe_lora", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_attention_pins_detected(operations) -> None:
    """Half five of the attention keyword handoff seam (issue #30).

    Without sequence parallelism every stock call runs a block's pin, so the
    pack reads the pins to record them and to ask for the Sol-Attn waiver
    (dgx_monarch/attention_pins.py). It reads ``ComfyAttention.config`` and
    ``function``, names a bound kernel by its function's ``__name__`` and
    ``tau`` keyword, and finds the module by its attribute name,
    ``comfy_attention``; each is pinned here. A tiny real Wan 2.1 model loads
    a state dict carrying two config tensors encoded by ComfyUI's own
    ``_save_to_state_dict``, and the detector must report both. A partial
    ``load_state_dict``, the shape of the bf16 and fp16 FSDP shard build,
    clears them, and the detector must say so. A LoRA built by tools/make_pin_probe_lora.py then goes
    through ComfyUI's own ``load_lora_for_models``: the pin must sit in the
    patcher's object patches, which the detector reads, until ``patch_model``
    puts it in place. Last, the probe tool's key patterns must find exactly the
    pinnable attention modules of tiny real Krea2, Anima (cosmos DiT),
    Kandinsky 5, Wan 2.1, Qwen Image 2.1 and FLUX.3 models (split and fused
    projections), and FLUX.3's early pattern exactly its ten early blocks,
    whose real config tensors the detector reads where the int8 files put
    them, on the block itself.
    """
    import re
    import types

    import comfy.ldm.modules.attention as attention_module
    import comfy.sd
    import torch
    from comfy.ldm.cosmos.predict2 import MiniTrainDIT
    from comfy.ldm.kandinsky5.model import Kandinsky5
    from comfy.ldm.krea2.model import SingleStreamDiT
    from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
    from comfy.ldm.wan.model import WanModel
    from comfy.model_patcher import ModelPatcher

    from dgx_monarch import attention_pins

    comfy_attention = attention_module.ComfyAttention()
    assert (getattr(comfy_attention, attention_pins.CONFIG_ATTR, "absent") is None
            and getattr(comfy_attention, attention_pins.FUNCTION_ATTR, "absent") is None), (
        "attention keyword handoff: ComfyAttention no longer keeps config and function, "
        "so the pin detector reads nothing")
    for name, kernel in attention_pins.KERNEL_FUNCTIONS.items():
        bound = getattr(attention_module, name, None)
        assert bound is not None and bound.__name__ == name, (
            f"attention keyword handoff: ComfyUI no longer binds {kernel} as {name}")
    tau = inspect.signature(attention_module.attention_comfy_kitchen_sol).parameters.get("tau")
    assert tau is not None and tau.kind is inspect.Parameter.KEYWORD_ONLY, (
        "attention keyword handoff: the Sol binding no longer takes tau by keyword")
    methods = set(re.findall(r'method == "([a-z0-9_]+)"',
                             inspect.getsource(attention_module.ComfyAttention._load_from_state_dict)))
    assert methods == set(attention_pins.KNOWN_KERNELS), (
        f"attention keyword handoff: ComfyAttention now binds {sorted(methods)}; classify "
        "each kernel in attention_pins (exact or approximate) before this passes")

    def encoded(config):
        # ComfyUI's own encoding: the config ComfyAttention saves.
        module = attention_module.ComfyAttention()
        module.config = config
        return module.state_dict()["config"]

    def wan():
        return WanModel(
            model_type="t2v", patch_size=(1, 2, 2), text_len=4, in_dim=4, dim=16,
            ffn_dim=32, freq_dim=2, text_dim=4, out_dim=4, num_heads=2, num_layers=2,
            device="cpu", dtype=torch.float32, operations=operations).eval()

    class _Holder(torch.nn.Module):
        """The BaseModel shape the patcher and comfy's LoRA key map read."""

        def __init__(self, diffusion_model):
            super().__init__()
            self.diffusion_model = diffusion_model
            self.model_config = types.SimpleNamespace(unet_config={})

    state = wan().state_dict()
    state["blocks.0.self_attn.comfy_attention.config"] = encoded(
        {"attention": "comfy_kitchen_sol", "tau": 0.7})
    state["blocks.1.cross_attn.comfy_attention.config"] = encoded(
        {"attention": "comfy_kitchen_int8"})
    dit = wan()
    dit.load_state_dict(state, strict=False)  # BaseModel.load_model_weights' call
    cpu = torch.device("cpu")
    patcher = ModelPatcher(_Holder(dit), cpu, cpu)
    found = attention_pins.scan_loaded(patcher)
    assert {path: (pin.kernel, pin.tau) for path, pin in found.items()} == {
        "diffusion_model.blocks.0.self_attn.comfy_attention": ("comfy_kitchen_sol", 0.7),
        "diffusion_model.blocks.1.cross_attn.comfy_attention": ("comfy_kitchen_int8", None),
    }, f"attention keyword handoff: the detector read {found} from a pinned checkpoint"
    for path, pin in found.items():
        module = attention_pins._resolve(patcher.model, path)
        assert pin.bound is (module.function is not None), (
            "attention keyword handoff: the detector misread whether ComfyUI bound the kernel")

    # A LoRA's pin through ComfyUI's own patcher, with the probe tool's keys.
    tool = _probe_tool()
    paths = tool.attention_paths("wan21", list(state))
    assert paths == ["blocks.0.self_attn", "blocks.1.self_attn"], paths
    lora = tool.build_lora(paths, tool.pin_config("comfy_kitchen_int8", None))
    patched, _clip = comfy.sd.load_lora_for_models(patcher, None, lora, 1.0, 0)
    staged = sorted(key for key in patched.object_patches if key.endswith(".comfy_attention"))
    assert staged == [f"diffusion_model.{path}.comfy_attention" for path in paths], (
        f"attention keyword handoff: a LoRA .config key no longer stages an object patch "
        f"(staged {staged})")
    summary = attention_pins.summarize([patched], sequence_parallel=False)
    assert summary is not None and summary.kernels == {"comfy_kitchen_int8": 3}, (
        f"attention keyword handoff: the detector missed a LoRA's pin ({summary})")
    live = dit.blocks[1].self_attn.comfy_attention
    assert live.config is None, "a staged LoRA pin is live before patch_model"
    patched.patch_model(load_weights=False)
    assert dit.blocks[1].self_attn.comfy_attention.config == {"attention": "comfy_kitchen_int8"}, (
        "attention keyword handoff: patch_model no longer puts a LoRA's pin in place")
    patched.unpatch_model(unpatch_weights=False)
    assert dit.blocks[1].self_attn.comfy_attention is live

    # A partial load clears every pin, as the bf16 and fp16 FSDP shard build does.
    dit.load_state_dict({"blocks.0.self_attn.q.weight": state["blocks.0.self_attn.q.weight"]},
                        strict=False)
    attention_pins.note_injected(patcher)
    cleared = attention_pins.summarize([patcher], sequence_parallel=False)
    assert cleared is not None and cleared.dropped == {attention_pins.DROPPED_FSDP: 2}, (
        "attention keyword handoff: a partial load_state_dict no longer clears a block's "
        "pin, so the FSDP record may be wrong")

    # The probe tool's key patterns against real classes.
    families = {
        "wan21": wan(),
        "krea2": SingleStreamDiT(
            features=32, tdim=8, txtdim=8, heads=2, kvheads=2, multiplier=1, layers=2,
            patch=1, channels=4, txtlayers=3, txtheads=2, txtkvheads=2, device="cpu",
            dtype=torch.float32, operations=operations),
        "anima": MiniTrainDIT(
            max_img_h=8, max_img_w=8, max_frames=1, in_channels=4, out_channels=4,
            patch_spatial=2, patch_temporal=1, concat_padding_mask=False,
            model_channels=24, num_blocks=2, num_heads=2, crossattn_emb_channels=8,
            pos_emb_cls="rope3d", use_adaln_lora=True, adaln_lora_dim=4, device="cpu",
            dtype=torch.float32, operations=operations),
        "kandinsky5": Kandinsky5(
            in_visual_dim=4, out_visual_dim=4, in_text_dim=8, in_text_dim2=8, time_dim=8,
            model_dim=16, ff_dim=32, visual_embed_dim=16, patch_size=(1, 2, 2),
            num_text_blocks=2, num_visual_blocks=2, axes_dims=(4, 2, 2), device="cpu",
            dtype=torch.float32, operations=operations),
        "qwen_image21": QwenImage21Transformer2DModel(
            in_channels=64, out_channels=64, num_layers=2, attention_head_dim=64,
            num_attention_heads=1, context_in_dim=4, mlp_ratio=1,
            axes_dims_rope=(16, 16, 32), operations=operations),
        "flux3": _tiny_flux3(),
        "flux3_fused": _tiny_flux3(fused=True),
    }
    for family, model in families.items():
        keys = list(model.state_dict())
        if family == "anima":
            keys = ["net." + key for key in keys]  # the prefix Anima files carry
        owners = sorted(name.rpartition(".")[0] for name, module in model.named_modules()
                        if isinstance(module, attention_module.ComfyAttention))
        pattern = family.removesuffix("_fused")
        matched = tool.attention_paths(pattern, keys, cross=True)
        assert sorted(matched) == owners, (
            f"attention keyword handoff: the probe tool's {family} pattern finds {matched}, "
            f"and the model's pinnable attention is {owners}")

    # FLUX.3's early pattern: the ten blocks comfy runs from preprocess_conds,
    # the stock sites a pin probe exercises. Each block owns its attention, so
    # the config sits at <block>.comfy_attention.config, as in the int8 files.
    flux3 = families["flux3"]
    early = sorted(name for name, _module in flux3.named_modules()
                   if (name.count(".") == 1 and name.startswith("txt_mode_blocks."))
                   or (name.count(".") == 2 and name.startswith("content_mode_blocks.image_cond.")))
    for model in (flux3, families["flux3_fused"]):
        assert tool.attention_paths("flux3_early", list(model.state_dict())) == early, (
            "attention keyword handoff: the probe tool's flux3_early pattern no longer "
            "finds exactly the early text and reference blocks")
    flux3_state = flux3.state_dict()
    pinned_paths = (early[0], early[-1], "content_mode_blocks.image.0", "single_blocks.1")
    for path in pinned_paths:
        flux3_state[f"{path}.comfy_attention.config"] = encoded({"attention": "comfy_kitchen_int8"})
    flux3_live = _tiny_flux3()
    flux3_live.load_state_dict(flux3_state, strict=False)
    found = attention_pins.scan_loaded(ModelPatcher(_Holder(flux3_live), cpu, cpu))
    assert {path: pin.kernel for path, pin in found.items()} == {
        f"diffusion_model.{path}.comfy_attention": "comfy_kitchen_int8" for path in pinned_paths
    }, f"attention keyword handoff: the detector read {found} from a pinned FLUX.3 checkpoint"


def _assert_moss_image_pins_dropped(pin, pinned_runs) -> None:
    """Half six of the attention keyword handoff seam: MOSS-Image's per-block pins.

    The INT8 files pin every block's attention through a real ComfyAttention.
    Stock runs each pin, block and text block alike; the sharded forward runs
    none: the blocks attend through its override and the two text blocks, run
    whole on every rank, get ComfyUI's default backend, and the output equals
    the unpinned stock forward bit for bit. The probe tool's ``moss_image``
    pattern must find exactly the pinnable attention of a tiny real model,
    fused and split, and ``moss_image_text`` only its two text blocks.
    """
    import comfy.ldm.modules.attention as attention_module
    import torch

    from dgx_monarch.adapters.base import InjectionContext
    from dgx_monarch.adapters.moss_image import MOSSImageAdapter

    def one_rank_attention(q, k, v, heads, *args, drop_rows=None, **kwargs):
        # The sharded dispatcher's place on one rank: no pad rows.
        assert not drop_rows, "attention keyword handoff: a one-rank MOSS-Image call named pad rows"
        return attention_module.optimized_attention(q, k, v, heads, *args, **kwargs)

    model = _tiny_moss(seed=39)
    x, timestep, context = _moss_inputs(seed=39)
    with torch.no_grad():
        unpinned = model._forward(x, timestep, context)
        for module in (model.layers[0], model.layers[2], model.text_embedder.blocks[1]):
            pin(module.comfy_attention)
        pinned_runs.clear()
        pinned = model._forward(x, timestep, context)
        assert len(pinned_runs) >= 3 and not torch.equal(pinned, unpinned), (
            "attention keyword handoff: stock MOSS-Image no longer runs a block's pinned kernel")
        with _OneSpRank():
            MOSSImageAdapter().inject_usp(model, InjectionContext(
                topology_sp=1, usp_attention=one_rank_attention, pure_ulysses=True))
            pinned_runs.clear()
            injected = model._forward(x, timestep, context)
    assert not pinned_runs, (
        "attention keyword handoff: the MOSS-Image sequence-parallel forward ran a pinned kernel")
    assert torch.equal(injected, unpinned), (
        "attention keyword handoff: the MOSS-Image sequence-parallel forward with pinned blocks no "
        "longer matches the unpinned stock forward")

    tool = _probe_tool()
    for fused in (True, False):
        tiny = _tiny_moss(fused=fused)
        keys = list(tiny.state_dict())
        owners = sorted(name.rpartition(".")[0] for name, module in tiny.named_modules()
                        if isinstance(module, attention_module.ComfyAttention))
        matched = tool.attention_paths("moss_image", keys, cross=True)
        assert sorted(matched) == owners, (
            f"attention keyword handoff: the probe tool's moss_image pattern finds {matched}, and "
            f"the {'fused' if fused else 'split'} model's pinnable attention is {owners}")
        text = [owner for owner in owners if owner.startswith("text_embedder.")]
        assert tool.attention_paths("moss_image_text", keys) == text and len(text) == 2, (
            "attention keyword handoff: the probe tool's moss_image_text pattern no longer finds "
            "exactly the two text blocks")


def _assert_cfg_pp_uncond_pass_seam() -> None:
    """Which samplers keep the uncond pass at CFG 1.0, and the flag that says so.

    Auto may take a cfg-parallel row at CFG 1.0 only under a sampler that
    keeps the uncond pass (topology.sampler_keeps_uncond_at_cfg1, a cfg_pp
    suffix). comfy keeps that pass only when the sampler sets
    disable_cfg1_optimization. A named sampler that stopped setting it would
    hand cfg-parallel a batch of one at CFG 1.0, and every such render would
    refuse; a sampler that sets it without the suffix only costs speed.
    """
    import inspect
    import types

    import comfy.k_diffusion.sampling as k_sampling
    import comfy.samplers
    import torch

    from dgx_monarch.topology import sampler_keeps_uncond_at_cfg1

    class FirstCall(Exception):
        pass

    def flag_at_first_call(name: str) -> bool:
        seen: dict[str, bool] = {}

        def model(x, sigma, **extra_args):
            options = extra_args.get("model_options", {})
            seen["flag"] = bool(options.get("disable_cfg1_optimization", False))
            raise FirstCall

        model_sampling = types.SimpleNamespace(noise_scale=1.0)
        model.inner_model = types.SimpleNamespace(  # type: ignore[attr-defined]
            model_patcher=types.SimpleNamespace(get_model_object=lambda _name: model_sampling))
        sampler = getattr(k_sampling, f"sample_{name}")
        try:
            sampler(model, torch.zeros(1, 4, 8, 8), torch.tensor([1.0, 0.5, 0.0]),
                    extra_args={"model_options": {}, "seed": 0}, disable=True)
        except FirstCall:
            return seen["flag"]
        raise AssertionError(f"cfg++ seam: {name} returned without calling the model")

    shortcut = inspect.getsource(comfy.samplers.sampling_function)
    assert "disable_cfg1_optimization" in shortcut, (
        "cfg++ seam: sampling_function's CFG 1.0 shortcut no longer reads "
        "disable_cfg1_optimization, so the flag no longer decides the uncond pass")
    names = [name for name in comfy.samplers.KSampler.SAMPLERS
             if sampler_keeps_uncond_at_cfg1(name)]
    assert names == [name for name in comfy.samplers.KSampler.SAMPLERS
                     if name.endswith("cfg_pp")] and names, (
        "cfg++ seam: the rule no longer names exactly the cfg_pp samplers")
    dropped = [name for name in names if not flag_at_first_call(name)]
    assert not dropped, (
        f"cfg++ seam: {dropped} no longer set disable_cfg1_optimization, so auto "
        "would take a cfg row at CFG 1.0 for a sampler that drops the uncond pass")
    assert not flag_at_first_call("euler"), (
        "cfg++ seam: plain euler now sets disable_cfg1_optimization, so the rule "
        "under-reads it; widen sampler_keeps_uncond_at_cfg1")


def _tiny_flux3(*, flow_map: bool = False, fused: bool = False, seed: int = 0):
    """A real comfy.ldm.flux3.model.Flux3 at toy width, CPU float32, seeded.

    Hidden 256 over two heads keeps the head dimension at 128, which the
    four 32-wide rotary axes need; one block per early and image stack and
    two single blocks.
    """
    import comfy.ops
    import torch
    from comfy.ldm.flux3.model import Flux3

    generator = torch.Generator().manual_seed(seed)
    model = Flux3(hidden_size=256, num_heads=2, depth=1, depth_single_blocks=2,
                  fused_qkv=fused, flow_map=flow_map, dtype=torch.float32, device="cpu",
                  operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return model


def _tiny_flux3_base(*, flow_map: bool = False):
    """A real model_base.Flux3 built by ComfyUI's own diffusion-model load.

    The state dict carries the vendor files' ``*_norm.scale`` names, which
    the Flux loader renames, so the load is the one a real file takes.
    """
    import comfy.sd

    state = {key.replace("_norm.weight", "_norm.scale"): value
             for key, value in _tiny_flux3(flow_map=flow_map).state_dict().items()}
    return comfy.sd.load_diffusion_model_state_dict(state).model


def _assert_flux3_conds_branch_seam() -> None:
    """Which conditioning branch FLUX.3 takes, and the bind that fixes it.

    ``Flux3.extra_conds`` runs the early text and reference blocks once per
    conditioning and hands ``_forward`` their states, but only when inference
    mode is on, no hooks are set, ``cross_attn`` is present and the model is
    no flow map; otherwise ``_forward`` runs them every call. A Worker samples
    on a thread where inference mode is off, so the adapter's
    ``prepare_base_model`` runs stock ``extra_conds`` under inference mode
    and clones every inference tensor out (adapters/flux3.py). Pinned here on
    a tiny real model_base.Flux3: (a) the four conditions of stock's cache,
    (b) the cached keys and their cond classes, (c) ``_forward`` taking both
    input sets to equal outputs at batch 1, (d) the bound call from a thread
    outside inference mode equal to stock's under inference mode, and (e) no
    inference tensor in the bound output, base and flow-map model alike.
    """
    from concurrent.futures import ThreadPoolExecutor

    import comfy.conds
    import torch

    from dgx_monarch.adapters.flux3 import Flux3Adapter

    generator = torch.Generator().manual_seed(41)
    context = torch.randn(1, 80, 20480, generator=generator) * 0.1
    refs = [torch.randn(1, 128, 3, 4, generator=generator)]
    cpu = torch.device("cpu")
    base = _tiny_flux3_base()
    flow = _tiny_flux3_base(flow_map=True)
    assert base.diffusion_model.flow_map is False and flow.diffusion_model.flow_map is True, (
        "flux3 conds branch: the step-distilled checkpoint no longer builds a flow map")

    def conds(model, inference=True, **extra):
        kwargs = {"cross_attn": context, "reference_latents": refs, "device": cpu, **extra}
        if inference:
            with torch.inference_mode():
                return model.extra_conds(**kwargs)
        return model.extra_conds(**kwargs)

    cached = {"txt_states", "ref_states", "ref_shapes"}
    assert set(conds(base)) >= cached and "c_crossattn" not in conds(base), (
        "flux3 conds branch: stock no longer caches the early states under inference mode")
    for label, out in (("outside inference mode", conds(base, inference=False)),
                       ("with hooks", conds(base, hooks=object())),
                       ("on the flow map", conds(flow))):
        assert not cached & set(out) and {"c_crossattn", "ref_latents"} <= set(out), (
            f"flux3 conds branch: stock now caches {label}, so the in-forward branch moved")
    no_text = conds(base, cross_attn=None)
    assert not cached & set(no_text), (
        "flux3 conds branch: stock caches with no cross_attn")
    stock = conds(base)
    assert isinstance(stock["txt_states"], comfy.conds.CONDRegular) and isinstance(
        stock["ref_states"], comfy.conds.CONDRegular) and type(
        stock["ref_shapes"]) is comfy.conds.CONDConstant, (
        "flux3 conds branch: the cached states changed cond class")
    assert stock["ref_shapes"].cond == [tuple(ref.shape[-2:]) for ref in refs], (
        "flux3 conds branch: ref_shapes no longer lists each reference grid")

    x = torch.randn(1, 128, 4, 4, generator=generator)
    timestep = torch.tensor([0.5])
    dm = base.diffusion_model
    with torch.no_grad():
        given = dm._forward(x, timestep, transformer_options={},
                            txt_states=stock["txt_states"].cond,
                            ref_states=stock["ref_states"].cond,
                            ref_shapes=stock["ref_shapes"].cond)
        computed = dm._forward(x, timestep, context, refs, transformer_options={})
    assert torch.equal(given, computed), (
        "flux3 conds branch: _forward no longer gives equal outputs from cached and "
        "computed early states at batch 1")

    def walk(value):
        if isinstance(value, list):
            for item in value:
                yield from walk(item)
        elif torch.is_tensor(value):
            yield value

    for model in (base, flow):
        Flux3Adapter().prepare_base_model(model)
        with ThreadPoolExecutor(1) as pool:
            assert not pool.submit(torch.is_inference_mode_enabled).result(), (
                "flux3 conds branch: a fresh executor thread now inherits inference mode")
            bound = pool.submit(conds, model, False).result()
        reference = conds(model) if model is base else None
        if reference is not None:
            assert set(bound) == set(reference) and all(
                torch.equal(bound[key].cond, reference[key].cond)
                for key in ("txt_states", "ref_states")), (
                "flux3 conds branch: the bound extra_conds off inference mode differs "
                "from stock's under it")
        else:
            assert {"c_crossattn", "ref_latents"} <= set(bound), (
                "flux3 conds branch: the bind moved the flow map off stock's in-forward branch")
        leaked = [key for key, cond in bound.items()
                  for tensor in walk(getattr(cond, "cond", None)) if tensor.is_inference()]
        assert not leaked, (
            f"flux3 conds branch: the bound output still holds inference tensors in {leaked}")


# MOSS-Image's vendor negative tags (the USERGUIDE's recommended list), which
# the shipped CFG 3 templates encode.
_MOSS_NEGATIVE = "nsfw, worst quality, bad anatomy, watermark, jpeg artifact"


def _tiny_moss(*, fused: bool = True, heads: int = 2, text_dim: int = 64, seed: int = 0):
    """A real comfy.ldm.moss_image.model.MOSSImage at toy width, CPU float32, seeded.

    128 a head keeps the four 32-wide rotary axes; three blocks put a skipped
    block in the middle.
    """
    import comfy.ldm.moss_image.model as moss
    import comfy.ops
    import torch

    generator = torch.Generator().manual_seed(seed)
    model = moss.MOSSImage(dim=128 * heads, ffn_dim=512, num_heads=heads, num_layers=3,
                           text_dim=text_dim, latent_channels=128, fused_qkv=fused,
                           operations=comfy.ops.disable_weight_init, dtype=torch.float32,
                           device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return model


def _tiny_moss_patcher():
    """A real model_base.MOSSImage in its ModelPatcher, by ComfyUI's own load.

    Detection keeps the default 4096-wide text input, so the state dict carries
    it, and the vendor files' ``rope.bands``, which the load pops.
    """
    import comfy.sd
    import torch

    state = dict(_tiny_moss(text_dim=4096).state_dict())
    state["rope.bands"] = torch.arange(16, dtype=torch.float32)
    return comfy.sd.load_diffusion_model_state_dict(state)


def _moss_inputs(grid=(3, 3), text=6, seed=0):
    import torch

    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(1, 128, *grid, generator=generator), torch.tensor([0.6]),
            torch.randn(1, text, 64, generator=generator))


def _moss_slg_step(patcher, sigma: float, *, start: float = 0.0, end: float = 1.0,
                   uncond: bool = True):
    """One step of ComfyUI's own SkipLayerGuidanceDiTSimple function, its calls captured.

    Returns the patched patcher, the conds it was handed, each
    ``calc_cond_batch`` call as ``(conds, model_options, outputs)`` and what the
    function returned. The stand-in ``calc_cond_batch`` gives every output its
    own value, so a returned output can be told apart.
    """
    import comfy.samplers
    import torch
    from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple

    patched = SkipLayerGuidanceDiTSimple.execute(
        patcher, start, end, double_layers="", single_layers="1")[0]
    function = patched.model_options["sampler_calc_cond_batch_function"]
    calls: list = []
    real = comfy.samplers.calc_cond_batch

    def capture(model, conds, x_in, timestep, model_options):
        outputs = [torch.full_like(x_in, float(10 * len(calls) + index)) for index in range(len(conds))]
        calls.append((conds, model_options, outputs))
        return outputs

    conds = [[{"cond": "positive"}], [{"cond": "negative"}] if uncond else None]
    comfy.samplers.calc_cond_batch = capture
    try:
        result = function({"conds": conds, "input": torch.zeros(1, 128, 2, 2),
                           "sigma": torch.tensor([sigma]), "model": patcher.model,
                           "model_options": patched.model_options})
    finally:
        comfy.samplers.calc_cond_batch = real
    return patched, conds, calls, result


def _assert_moss_image_conds_seam() -> None:
    """What MOSS-Image's conditioning publishes, which decides its cfg2 path.

    ``model_base.MOSSImage.extra_conds`` publishes ``c_crossattn`` as
    CONDRegular, which concatenates only at equal shapes, and no CONDConstant,
    so a prompt pair of different lengths runs as two calls that cfg-parallel
    dispatches one per rank, and an equal pair folds into one call. Every image
    row's RoPE position depends on the prompt length, so no driver pad is exact
    (``cfg_cond_padding = "none"``). With no reference ``extra_conds`` runs
    nothing of the diffusion model; a plain model's ``concat_cond`` gives no
    control input; the load pops ``rope.bands``; the
    sampling is flow at multiplier 1.0 and shift 5.0 (so the SD3 shift node,
    multiplier 1000, would be noise); the latent is Flux2 at 128 channels and
    ratio 16; no attention mask reaches the model; and ComfyUI's MOSS tokenizer
    gives an empty negative 10 tokens and the vendor tags 23, the two parities
    the hardware legs pick prompts by.
    """
    import comfy.conds
    import comfy.latent_formats
    import comfy.model_base
    import comfy.text_encoders.moss_image
    import torch

    patcher = _tiny_moss_patcher()
    base = patcher.model
    assert type(base) is comfy.model_base.MOSSImage, (
        f"moss_image conds seam: the vendor layout now loads as {type(base).__name__}")
    assert not any("rope.bands" in key for key in base.state_dict()), (
        "moss_image conds seam: the load keeps rope.bands, an fp32 tensor the live model never held")
    kept = base.model_config.process_unet_state_dict(
        {"rope.bands": torch.zeros(16), "layers.0.norm1.weight": torch.zeros(4)})
    assert set(kept) == {"layers.0.norm1.weight"}, (
        f"moss_image conds seam: process_unet_state_dict now keeps {sorted(kept)}")

    cpu = torch.device("cpu")
    noise = torch.zeros(1, 128, 4, 4)
    generator = torch.Generator().manual_seed(42)
    touched: list[str] = []
    dm = base.diffusion_model
    def touch(module, args):
        touched.append(type(module).__name__)

    hooks = [module.register_forward_pre_hook(touch) for module in dm.modules()]

    def refuse_rope(*_args, **_kwargs):
        touched.append("rope")
        raise AssertionError("extra_conds built a rotary table")

    dm.rope = refuse_rope
    try:
        def conds(length, **extra):
            context = torch.randn(1, length, 4096, generator=generator)
            return base.extra_conds(cross_attn=context, noise=noise, device=cpu, **extra)

        short = conds(10, attention_mask=torch.ones(1, 10))
        long, twin = conds(23), conds(10)
        concat = base.concat_cond(noise=noise, device=cpu)
    finally:
        for hook in hooks:
            hook.remove()
        del dm.rope
    assert not touched, f"moss_image conds seam: extra_conds now runs {sorted(set(touched))}"
    assert set(short) == {"c_crossattn"}, (
        f"moss_image conds seam: extra_conds now publishes {sorted(short)}, so a mask or a "
        "constant may reach the model")
    assert type(short["c_crossattn"]) is comfy.conds.CONDRegular, (
        f"moss_image conds seam: c_crossattn is {type(short['c_crossattn']).__name__}, not CONDRegular")
    assert not short["c_crossattn"].can_concat(long["c_crossattn"]), (
        "moss_image conds seam: two prompts of different lengths now concatenate into one call")
    assert short["c_crossattn"].can_concat(twin["c_crossattn"]), (
        "moss_image conds seam: two prompts of equal length no longer concatenate")
    assert concat is None, "moss_image conds seam: a plain model's concat_cond now returns an input"

    assert base.model_type == comfy.model_base.ModelType.FLOW, (
        f"moss_image conds seam: the model type is {base.model_type}, not flow")
    sampling = base.model_sampling
    assert (float(sampling.multiplier), float(sampling.shift)) == (1.0, 5.0), (
        f"moss_image conds seam: sampling is multiplier {sampling.multiplier}, shift {sampling.shift}")
    latent = base.latent_format
    assert type(latent) is comfy.latent_formats.Flux2 and latent.latent_channels == 128 and (
        latent.spacial_downscale_ratio == 16), (
        f"moss_image conds seam: the latent format is {type(latent).__name__}, "
        f"{latent.latent_channels} channels, ratio {latent.spacial_downscale_ratio}")

    tokenizer = comfy.text_encoders.moss_image.MOSSImageTokenizer()
    lengths = {text: [len(section) for sections in tokenizer.tokenize_with_weights(text).values()
                      for section in sections]
               for text in ("", _MOSS_NEGATIVE)}
    assert lengths == {"": [10], _MOSS_NEGATIVE: [23]}, (
        f"moss_image conds seam: the MOSS tokenizer now gives {lengths} tokens for the empty "
        "negative and the vendor tags; the hardware legs pick N's parity by these counts")
    if hasattr(type(dm), "preprocess_refs"):
        _assert_moss_image_reference_conds(base)


# Stock's reference methods, and whether one-GPU ComfyUI caches the keys and
# values for it in extra_conds (inference mode, no hooks).
_MOSS_REFERENCE_METHODS = {None: True, "index": False, "index_timestep_zero": False,
                           "index_timestep_zero_isolated": True,
                           "index_timestep_zero_text_start": True}


def _assert_moss_image_reference_conds(base) -> None:
    """What MOSS-Image reference images publish since ComfyUI 60aade0bb (#130).

    Text Encode MOSS-Image Edit sets ``reference_latents`` and
    ``reference_latents_method`` with conditioning_set_values, the two keys the
    driver's summary reads before the wire packs a cond. Outside inference
    mode extra_conds hands the latents to the forward as a CONDList whatever
    the method, and runs nothing of the model; inside it, as one-GPU ComfyUI
    runs, the isolated methods cache every block's keys and values once
    through preprocess_refs. Either way the forward takes them as its own
    three reference parameters, which the sharded forward refuses before any
    collective. A Worker samples on a thread outside inference mode, so the
    adapter's ``prepare_base_model`` runs stock ``extra_conds`` under it
    (adapters/moss_image.py): from a fresh thread the bound call must return
    stock's in-mode output for every method, with no inference tensor. Since
    9fe7ef764 the node also sets ``context_attention_bias`` from its
    ``image_strength``, and extra_conds publishes it as a CONDRegular for the
    forward's parameter of that name, which the sharded forward refuses
    unless it is all zeros (#135).
    """
    import inspect
    from concurrent.futures import ThreadPoolExecutor

    import comfy.conds
    import comfy.model_base
    import torch
    from comfy_extras import nodes_moss_image

    edit = nodes_moss_image.TextEncodeMOSSImageEdit
    assert edit.define_schema().node_id == "TextEncodeMOSSImageEdit", (
        "moss_image conds seam: the edit node's id moved, and the #130 refusal names it")
    source = inspect.getsource(edit.execute)
    assert '"reference_latents"' in source and '"reference_latents_method"' in source, (
        "moss_image conds seam: the edit node no longer sets reference_latents and "
        "reference_latents_method, the keys the driver preflight reads")
    # 9fe7ef764's text bias: the #135 refusal names the node and this widget.
    widgets = {item.id: item for item in edit.define_schema().inputs}
    assert '"context_attention_bias"' in source and getattr(
        widgets.get("image_strength"), "default", None) == 1.0, (
        "moss_image conds seam: the edit node no longer sets context_attention_bias from an "
        "image_strength that defaults to 1.0, as the #135 refusal says")
    assert "torch.is_inference_mode_enabled()" in inspect.getsource(
        comfy.model_base.MOSSImage.extra_conds), (
        "moss_image conds seam: extra_conds no longer gates its reference cache on inference "
        "mode, so a Worker may now run preprocess_refs outside the forward")
    dm = base.diffusion_model
    names = tuple(inspect.signature(type(dm)._forward).parameters)
    assert names[5:] == ("ref_latents", "ref_latents_method", "ref_kv", "context_attention_bias",
                         "kwargs"), (
        f"moss_image conds seam: the forward's reference and bias parameters are now {names}")

    cpu = torch.device("cpu")
    generator = torch.Generator().manual_seed(7)
    context = torch.randn(1, 10, 4096, generator=generator)
    refs = [torch.randn(1, 128, 2, 2, generator=generator)]
    noise = torch.zeros(1, 128, 4, 4)
    bias = torch.zeros(1, 10)
    published = base.extra_conds(cross_attn=context, noise=noise, device=cpu,
                                 context_attention_bias=bias).get("context_attention_bias")
    assert type(published) is comfy.conds.CONDRegular and published.cond is bias, (
        f"moss_image conds seam: extra_conds publishes the text bias as {published}, "
        "not a CONDRegular holding it")
    ran: list = []
    stock = dm.preprocess_refs
    dm.preprocess_refs = lambda latents, *args, **kwargs: ran.append(len(latents)) or stock(
        latents, *args, **kwargs)
    try:
        def publish(method):
            extra = {} if method is None else {"reference_latents_method": method}
            return base.extra_conds(cross_attn=context, noise=noise, device=cpu,
                                    reference_latents=refs, **extra)

        for method in _MOSS_REFERENCE_METHODS:
            out = publish(method)
            assert type(out.get("ref_latents")) is comfy.conds.CONDList and "ref_kv" not in out, (
                f"moss_image conds seam: outside inference mode, method {method} now publishes "
                f"{sorted(out)}")
            assert ("ref_latents_method" in out) is (method is not None), (
                f"moss_image conds seam: method {method} publishes {sorted(out)}")
        assert ran == [], "moss_image conds seam: extra_conds ran preprocess_refs outside inference mode"
        with torch.inference_mode():
            cached = {method: "ref_kv" in publish(method) for method in _MOSS_REFERENCE_METHODS}
    finally:
        del dm.preprocess_refs
    assert cached == _MOSS_REFERENCE_METHODS, (
        f"moss_image conds seam: inside inference mode the methods that cache are now {cached}")
    assert ran == [1] * sum(_MOSS_REFERENCE_METHODS.values()), (
        f"moss_image conds seam: preprocess_refs ran {ran} inside inference mode")

    from dgx_monarch.adapters.moss_image import MOSSImageAdapter

    def tensors(value):
        if isinstance(value, list):
            for item in value:
                yield from tensors(item)
        elif torch.is_tensor(value):
            yield value

    stock_conds = {}
    with torch.inference_mode():
        for method in _MOSS_REFERENCE_METHODS:
            stock_conds[method] = publish(method)
    MOSSImageAdapter().prepare_base_model(base)
    with ThreadPoolExecutor(1) as pool:
        assert not pool.submit(torch.is_inference_mode_enabled).result(), (
            "moss_image conds seam: a fresh executor thread now inherits inference mode")
        bound = {method: pool.submit(publish, method).result() for method in _MOSS_REFERENCE_METHODS}
    for method, out in bound.items():
        stock_out = stock_conds[method]
        assert set(out) == set(stock_out) and ("ref_kv" in out) is _MOSS_REFERENCE_METHODS[method], (
            f"moss_image conds seam: the bound extra_conds publishes {sorted(out)} for method "
            f"{method}, stock in inference mode {sorted(stock_out)}")
        for key, cond in out.items():
            got, want = list(tensors(cond.cond)), list(tensors(stock_out[key].cond))
            assert len(got) == len(want) and all(
                torch.equal(a, b) for a, b in zip(got, want, strict=True)), (
                f"moss_image conds seam: the bound {key} for method {method} differs from stock's")
            assert not any(tensor.is_inference() for tensor in got), (
                f"moss_image conds seam: the bound {key} for method {method} holds an inference tensor")


def _assert_moss_image_block_surface_seam() -> None:
    """What the MOSS-Image sharded forward re-runs of stock's block and forward.

    adapters/moss_image.py keeps stock's prelude, cuts the joint stream into
    one window per rank and runs stock's own ``Block.forward`` there, with the
    block's two fused-norm calls launched at stock's row count through the
    global it reaches the norm by (adapters/moss_image_seams.py):
    ``modulated_rms_norm`` before ComfyUI 60aade0bb, ``modulate`` since, which
    with no reference returns the norm's bits with ``mod[i + 1]`` as scale and
    ``mod[i]`` as shift, as ``residual`` returns ``gated_residual``'s. Pinned on a
    tiny real MOSSImage: ``Block.forward`` reads one known form and makes exactly
    two norm calls over its whole stream with ``[B, 1, D]`` modulation; the
    text embedder calls its blocks positionally, with a None mask fourth since
    ComfyUI 9fe7ef764; ``_forward`` builds one table for all N rows with a
    known ``rope`` form, hands the text embedder its last T rows, joins
    ``[image, text]``, sets ``total_blocks`` and ``block_type`` before the
    prefetch queue and ``block_index`` before the replace lookup, calls a
    ``("single_block", i)`` patch with five keys and ``original_block``, hands
    each block no reference (none, or None twice on the later form) and, since
    9fe7ef764, ``mask=None``, opens and closes the malloc graph around the
    loop, and runs the head norm on the image rows; with no reference, the later rope
    form gets an empty shape list and offset 0; the table is ``[1, N, 1, 64, 2, 2]``,
    contiguous, image rows first; master's ``rope`` reads ``rope_options``; an
    identity skip equals dropping the block; ``attention()`` hands
    ``optimized_attention`` (q, k, v, heads) plus ``transformer_options`` and
    ``preferred_attention``, and a None ``mask`` since 9fe7ef764; no
    ``patches`` hook is read; ComfyUI's own skip passes the identity test the
    vetting runs on this interpreter. On that tree the text bias is the key
    mask ``_assert_moss_text_bias_is_a_key_mask`` pins.
    """
    import copy

    import comfy.ldm.moss_image.model as moss
    import comfy.model_prefetch as prefetch
    import torch

    from dgx_monarch.adapters.flux3_adaln_parity import loads_global
    from dgx_monarch.adapters.moss_image_seams import (
        block_norm_route,
        is_identity_block_patch,
        rope_call,
        vet_block_replacements,
    )

    route = block_norm_route(moss.Block.forward)
    assert route is not None, (
        "moss_image block surface: Block.forward reads neither modulated_rms_norm nor the pair "
        "modulate and residual alone")
    assert loads_global(moss.Block.forward, "attention"), (
        "moss_image block surface: Block.forward no longer reads the module global attention")
    model = _tiny_moss(seed=43)
    rope_call(model)    # refuses class P on a fourth rope form
    if route == "modulate":
        _assert_moss_modulate_is_the_norm(moss, model)
    # ComfyUI 9fe7ef764 added the text bias; every call it touches then names a mask.
    takes_bias = "context_attention_bias" in inspect.signature(moss.MOSSImage._forward).parameters
    x, timestep, context = _moss_inputs()
    hw, text = x.shape[2] * x.shape[3], context.shape[1]
    rows = hw + text
    events: list = []
    seen: dict = {"norms": [], "attention": [], "text_blocks": [], "ropes": [], "replace": []}
    saved = {name: getattr(moss, name) for name in ("modulated_rms_norm", "optimized_attention")}
    saved_prefetch = {name: getattr(prefetch, name) for name in (
        "make_prefetch_queue", "prefetch_queue_pop", "malloc_graph_begin", "malloc_graph_end")}

    def norm(x, norm_module, scale, shift):
        seen["norms"].append((tuple(x.shape), tuple(scale.shape), tuple(shift.shape)))
        return saved["modulated_rms_norm"](x, norm_module, scale, shift)

    def attention(*args, **kwargs):
        seen["attention"].append((len(args), sorted(kwargs), kwargs.get("mask") is None))
        return saved["optimized_attention"](*args, **kwargs)

    def make_prefetch_queue(queue, device, options):
        events.append(("queue", len(queue), options.get("total_blocks"), options.get("block_type")))
        return saved_prefetch["make_prefetch_queue"](queue, device, options)

    def prefetch_queue_pop(queue, device, module, *args, **kwargs):
        events.append(("pop", kwargs.get("malloc_scope")))
        return saved_prefetch["prefetch_queue_pop"](queue, device, module, *args, **kwargs)

    def graph(label, function):
        def record(*args, **kwargs):
            events.append((label,))
            return function(*args, **kwargs)
        return record

    stock_rope = type(model).rope

    def rope(*args):
        table = stock_rope(model, *args)
        seen["ropes"].append((len(args), table, args[5:]))
        return table

    def replace(args, extra):
        seen["replace"].append((sorted(args), sorted(extra), args["transformer_options"]["block_index"]))
        return extra["original_block"](args)

    captured: dict = {}

    def keep(name, value):
        # A hook that returns a value replaces the call's input or output, so this returns None.
        captured.setdefault(name, value.detach().clone())

    def text_block(module, args, kwargs):
        seen["text_blocks"].append((len(args), sorted(kwargs), args[3:]))

    block_args: list = []
    hooks = [
        model.layers[0].register_forward_pre_hook(
            lambda module, args, kwargs: block_args.append(
                (args[4:], sorted(kwargs), kwargs.get("mask") is None)), with_kwargs=True),
        model.video_embedder.register_forward_hook(lambda module, args, out: keep("image", out)),
        model.text_embedder.register_forward_pre_hook(lambda module, args: keep("text_pe", args[1])),
        model.text_embedder.register_forward_hook(lambda module, args, out: keep("text", out)),
        model.layers[0].register_forward_pre_hook(lambda module, args: keep("joint", args[0])),
    ] + [block.register_forward_pre_hook(text_block, with_kwargs=True)
         for block in model.text_embedder.blocks]
    moss.modulated_rms_norm, moss.optimized_attention = norm, attention
    prefetch.make_prefetch_queue, prefetch.prefetch_queue_pop = make_prefetch_queue, prefetch_queue_pop
    prefetch.malloc_graph_begin = graph("graph begin", saved_prefetch["malloc_graph_begin"])
    prefetch.malloc_graph_end = graph("graph end", saved_prefetch["malloc_graph_end"])
    model.rope = rope

    def raising_hook(*_args, **_kwargs):
        raise AssertionError("a patches hook ran")

    options = {"patches_replace": {"dit": {("single_block", 1): replace}},
               "patches": {hook: [raising_hook] for hook in
                           ("attn1_patch", "attn1_output_patch", "attn2_patch", "middle_patch")}}
    try:
        with torch.no_grad():
            out = model(x, timestep, context, transformer_options=options)
    finally:
        for hook in hooks:
            hook.remove()
        moss.modulated_rms_norm, moss.optimized_attention = saved["modulated_rms_norm"], saved[
            "optimized_attention"]
        for name, function in saved_prefetch.items():
            setattr(prefetch, name, function)
        del model.rope

    blocks = len(model.layers)
    width = model.video_embedder.out_features
    assert seen["norms"] == [((1, rows, width), (1, 1, width), (1, 1, width))] * (2 * blocks) + [
        ((1, hw, width), (1, 1, width), (1, 1, width))], (
        f"moss_image block surface: the fused-norm calls are now {seen['norms']}")
    text_call = (4, [], (None,)) if takes_bias else (3, [], ())
    assert seen["text_blocks"] == [text_call] * len(model.text_embedder.blocks), (
        "moss_image block surface: TextEmbedder no longer calls its blocks positionally with "
        f"(x, pe, transformer_options{', None' if takes_bias else ''}): {seen['text_blocks']}")
    assert len(seen["ropes"]) == 1 and seen["ropes"][0][0] in (4, 5, 7), (
        f"moss_image block surface: _forward built {len(seen['ropes'])} rotary tables")
    assert seen["ropes"][0][2] in ((), ([], 0)), (
        f"moss_image block surface: with no reference stock's rope now gets {seen['ropes'][0][2]}")
    plain_calls = [[((None, None), ["mask"], True)]] if takes_bias else [
        [((), [], True)], [((None, None), [], True)]]
    assert block_args in plain_calls, (
        f"moss_image block surface: with no reference or bias a block now gets {block_args}")
    table = seen["ropes"][0][1]
    assert tuple(table.shape) == (1, rows, 1, 64, 2, 2) and table.is_contiguous() and (
        table.dtype == torch.float32), (
        f"moss_image block surface: the table is {tuple(table.shape)} {table.dtype}")
    assert torch.equal(captured["text_pe"], table[:, hw:]), (
        "moss_image block surface: the text embedder no longer gets the table's last T rows")
    joint = captured["joint"]
    assert torch.equal(joint[:, :hw], captured["image"]) and torch.equal(joint[:, hw:], captured["text"]), (
        "moss_image block surface: the joint stream is no longer [image, text]")
    identity = torch.eye(2).expand(64, 2, 2)
    assert torch.equal(table[0, :, 0, :16], identity[:16].expand(rows, 16, 2, 2)), (
        "moss_image block surface: rotary pairs 0 to 15 are no longer the identity axis")
    assert torch.equal(table[0, hw:, 0, 32:], identity[32:].expand(text, 32, 2, 2)), (
        "moss_image block surface: text rows no longer sit at h = w = 0, so the image rows may not come first")
    assert torch.equal(table[0, hw, 0, 16:32], identity[16:32]), (
        "moss_image block surface: the first text row is no longer at f = 0")
    assert all(torch.equal(table[0, row, 0, 16:32], table[0, 0, 0, 16:32]) for row in range(hw)) and (
        not torch.equal(table[0, 0, 0, 16:32], identity[16:32])), (
        "moss_image block surface: image rows no longer share one f position past the text")
    queue = [event for event in events if event[0] == "queue" and event[1] == blocks]
    assert queue == [("queue", blocks, blocks, "single")], (
        f"moss_image block surface: total_blocks and block_type are not set before the queue: {events}")
    begin, end = events.index(("graph begin",)), events.index(("graph end",))
    assert events.index(queue[0]) < begin < end and events[begin + 1:end] == [("pop", "block")] * (
        blocks + 1), f"moss_image block surface: the malloc graph no longer wraps the block loop: {events}"
    assert seen["replace"] == [(["img", "mod", "pe", "transformer_options", "vec"],
                                ["original_block"], 1)], (
        f"moss_image block surface: a single_block patch now gets {seen['replace']}")
    keywords = (["mask"] if takes_bias else []) + ["preferred_attention", "transformer_options"]
    assert seen["attention"] == [(4, keywords, True)] * (
        blocks + len(model.text_embedder.blocks)), (
        f"moss_image block surface: attention() now hands optimized_attention {seen['attention']}")
    assert out.shape == x.shape, f"moss_image block surface: the output is {tuple(out.shape)}"

    if seen["ropes"][0][0] >= 5:
        moved = stock_rope(model, 3, 3, text, torch.device("cpu"), {"rope_options": {"shift_t": 1.0}})
        assert not torch.equal(moved, table), (
            "moss_image block surface: master's rope no longer reads rope_options")
    if takes_bias:
        _assert_moss_text_bias_is_a_key_mask(moss, model, x, timestep, context)

    _patched, _conds, calls, _result = _moss_slg_step(_tiny_moss_patcher(), 0.5)
    skip = calls[1][1]["transformer_options"]["patches_replace"]["dit"][("single_block", 1)]
    assert is_identity_block_patch(skip), (
        "moss_image block surface: ComfyUI's Skip Layer Guidance skip no longer passes the identity "
        "test on this interpreter, so every sharded skip render would refuse")
    skipped = {"patches_replace": {"dit": {("single_block", 1): skip}}}
    assert vet_block_replacements(skipped, blocks) == {1: skip}, (
        "moss_image block surface: the vetting no longer admits ComfyUI's skip")
    dropped = copy.deepcopy(model)
    del dropped.layers[1]
    with torch.no_grad():
        with_skip = model(x, timestep, context, transformer_options=skipped)
        without = dropped(x, timestep, context, transformer_options={})
    assert torch.equal(with_skip, without), (
        "moss_image block surface: an identity skip on block 1 no longer equals dropping block 1")


def _assert_moss_modulate_is_the_norm(moss, model) -> None:
    """Since 60aade0bb: with no reference, modulate and residual are the two older calls.

    The sharded block launches stock's own ``modulate`` at stock's rows, so
    the norm it wraps must still be row-local and read as a module global;
    with a reference split, the rows past it take the other modulation, which
    is why the sharded forward refuses references.
    """
    import torch

    from dgx_monarch.adapters.flux3_adaln_parity import loads_global

    assert loads_global(moss.modulate, "modulated_rms_norm"), (
        "moss_image block surface: modulate no longer reads the module global modulated_rms_norm")
    generator = torch.Generator().manual_seed(44)
    width = model.video_embedder.out_features
    x = torch.randn(2, 7, width, generator=generator)
    y = torch.randn(2, 7, width, generator=generator)
    mod = tuple(torch.randn(2, 1, width, generator=generator) * 0.1 for _ in range(6))
    norm = model.layers[0].norm1
    with torch.no_grad():
        for index in (0, 3):
            assert torch.equal(moss.modulate(x, norm, mod, index, None),
                               moss.modulated_rms_norm(x, norm, mod[index + 1], mod[index])), (
                f"moss_image block surface: modulate({index}) with no reference is no longer the norm")
        for index in (2, 5):
            assert torch.equal(moss.residual(x.clone(), y, mod, index, None),
                               moss.gated_residual(x.clone(), y, mod[index])), (
                f"moss_image block surface: residual({index}) with no reference is no longer "
                "gated_residual")
        split = moss.modulate(x, norm, mod, 0, (4, mod[::-1]))
        assert torch.equal(split[:, :4], moss.modulated_rms_norm(x[:, :4], norm, mod[1], mod[0])) and (
            not torch.equal(split[:, 4:], moss.modulated_rms_norm(x[:, 4:], norm, mod[1], mod[0]))), (
            "moss_image block surface: a reference split no longer modulates the rows past it apart")


def _assert_moss_text_bias_is_a_key_mask(moss, model, x, timestep, context) -> None:
    """Since 9fe7ef764: the text bias is an additive mask on the text keys alone.

    Stock hands each text block ``bias[:, None, None]`` and each block the bias
    padded with zeros over the image rows, ``[B, 1, 1, N]``, and ``attention()``
    hands either to ``optimized_attention`` as ``mask``, so every query's logits
    for the text keys move and the image keys' do not. The sharded kernel takes
    no such mask, so the sharded forward refuses a bias (#135); an all-zero one
    returns stock's unbiased bits, so the sharded forward drops it instead.
    """
    import torch

    hw, text = x.shape[2] * x.shape[3], context.shape[1]
    bias = torch.zeros(1, text)
    bias[:, 1:3] = -1.5
    masks: dict = {"text": [], "blocks": [], "attention": []}
    stock_attention = moss.optimized_attention

    def attention(*args, **kwargs):
        masks["attention"].append(kwargs.get("mask"))
        return stock_attention(*args, **kwargs)

    hooks = [block.register_forward_pre_hook(lambda module, args: masks["text"].append(args[3]))
             for block in model.text_embedder.blocks] + [
        block.register_forward_pre_hook(
            lambda module, args, kwargs: masks["blocks"].append(kwargs.get("mask")), with_kwargs=True)
        for block in model.layers]
    moss.optimized_attention = attention
    try:
        with torch.no_grad():
            biased = model(x, timestep, context, transformer_options={}, context_attention_bias=bias)
    finally:
        for hook in hooks:
            hook.remove()
        moss.optimized_attention = stock_attention
    text_mask = bias[:, None, None]
    joint_mask = torch.cat((torch.zeros(1, hw), bias), dim=1)[:, None, None]
    texts, blocks = len(model.text_embedder.blocks), len(model.layers)
    assert len(masks["text"]) == texts and all(
        torch.is_tensor(mask) and torch.equal(mask, text_mask) for mask in masks["text"]), (
        f"moss_image block surface: the text blocks now get masks {masks['text']}, not bias[:, None, None]")
    assert len(masks["blocks"]) == blocks and all(
        torch.is_tensor(mask) and torch.equal(mask, joint_mask) for mask in masks["blocks"]), (
        "moss_image block surface: the blocks no longer get the bias padded with zeros over the "
        f"image rows: {masks['blocks']}")
    expected = [text_mask] * texts + [joint_mask] * blocks
    assert len(masks["attention"]) == len(expected) and all(
        torch.is_tensor(mask) and torch.equal(mask, want)
        for mask, want in zip(masks["attention"], expected, strict=True)), (
        "moss_image block surface: attention() no longer hands optimized_attention the bias as mask")
    with torch.no_grad():
        plain = model(x, timestep, context, transformer_options={})
        zero = model(x, timestep, context, transformer_options={},
                     context_attention_bias=torch.zeros(1, text))
    assert torch.equal(zero, plain), (
        "moss_image block surface: an all-zero text bias no longer returns the unbiased bits")
    assert not torch.equal(biased, plain), (
        "moss_image block surface: a text bias no longer moves the output")


def _assert_skip_layer_guidance_simple_seam() -> None:
    """ComfyUI's SkipLayerGuidanceDiTSimple, which the DGX Monarch carrier applies as is.

    The Worker runs the node's own ``execute`` on the render clone
    (actor/guidance_render.py), and the carrier node copies its widgets, so the
    widget names, order, defaults and ranges are a contract. Digit-free layer
    strings leave the model untouched; inside the window the installed
    function makes exactly two ``calc_cond_batch`` calls, ``[cond, None]`` with
    the caller's options and ``[None, uncond]`` with an identity
    ``("single_block", k)`` replacement in a copied options dict, and returns
    their outputs unchanged; outside the window and with no uncond (CFG 1) it
    makes one ordinary call. Two single-cond calls are why cfg2 refuses an
    active skip, and the copied dict is what every sequence-parallel rank skips
    alike.
    """
    from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple

    from dgx_monarch.adapters.moss_image_seams import is_identity_block_patch

    inputs = {item.id: item for item in SkipLayerGuidanceDiTSimple.define_schema().inputs}
    assert list(inputs) == ["model", "double_layers", "single_layers", "start_percent", "end_percent"], (
        f"skip layer guidance seam: the node's inputs are now {list(inputs)}")
    assert (inputs["double_layers"].default, inputs["single_layers"].default) == ("7, 8, 9", "7, 8, 9"), (
        "skip layer guidance seam: the layer defaults moved")
    for name, default in (("start_percent", 0.0), ("end_percent", 1.0)):
        item = inputs[name]
        assert (item.default, item.min, item.max, item.step) == (default, 0.0, 1.0, 0.001), (
            f"skip layer guidance seam: {name} is now {item.default} in [{item.min}, {item.max}] "
            f"by {item.step}")

    patcher = _tiny_moss_patcher()
    for layers in ("", ", ;"):
        same = SkipLayerGuidanceDiTSimple.execute(patcher, 0.0, 1.0, double_layers=layers,
                                                  single_layers=layers)[0]
        assert same is patcher, f"skip layer guidance seam: layers {layers!r} no longer leave the model as is"
    patched, conds, calls, result = _moss_slg_step(patcher, 0.5)
    assert patched is not patcher and "sampler_calc_cond_batch_function" not in patcher.model_options, (
        "skip layer guidance seam: execute no longer patches a clone")
    assert [call[0] for call in calls] == [[conds[0], None], [None, conds[1]]], (
        f"skip layer guidance seam: inside the window the calls are {[call[0] for call in calls]}")
    assert calls[0][1] is patched.model_options, (
        "skip layer guidance seam: the cond call no longer gets the caller's options")
    skip_options = calls[1][1]
    dit = skip_options["transformer_options"]["patches_replace"]["dit"]
    assert skip_options is not patched.model_options and set(dit) == {("single_block", 1)} and (
        is_identity_block_patch(dit[("single_block", 1)])), (
        "skip layer guidance seam: the uncond call no longer carries one identity skip in a copied dict")
    assert "patches_replace" not in patched.model_options.get("transformer_options", {}), (
        "skip layer guidance seam: the skip leaked into the caller's options")
    assert result[0] is calls[0][2][0] and result[1] is calls[1][2][1], (
        "skip layer guidance seam: the function no longer returns the two outputs unchanged")
    # end 0.5 maps through shift 5 to sigma 0.8333, so sigma 0.5 sits outside.
    patched, conds, calls, result = _moss_slg_step(patcher, 0.5, end=0.5)
    assert [call[0] for call in calls] == [conds] and calls[0][1] is patched.model_options, (
        "skip layer guidance seam: outside the window the step is no longer one ordinary call")
    _patched, conds, calls, _result = _moss_slg_step(patcher, 0.5, uncond=False)
    assert [call[0] for call in calls] == [conds], (
        "skip layer guidance seam: with no uncond (CFG 1) the step is no longer one ordinary call")


def _assert_lora_shape_changing_set_seam() -> None:
    """The shape test the LoRA shape guard runs, on ComfyUI's own patch list.

    actor/lora_shape_guard.py refuses class P, before any bake, a patch whose
    result has another shape than the live weight, with ComfyUI's own
    predicate ``comfy.lora.calculate_shape(patches, weight, key)``, the one the
    model patcher uses. Pinned here: a "set" patch of another shape reports the
    new shape, while diff, LoRA and offset patches keep the weight's; and on
    master ``convert_lora`` routes a MOSS-Image color and depth control state
    dict (``video_embedder.weight`` beside ``layers.0.proj.q.lora_A.weight``)
    to "set" patches that widen ``video_embedder`` from 128 to 384 inputs.
    """
    import comfy.lora
    import comfy.lora_convert
    import comfy.utils
    import torch

    patcher = _tiny_moss_patcher()
    key = "diffusion_model.video_embedder.weight"
    weight = comfy.utils.get_attr(patcher.model, key)
    width = weight.shape[0]
    assert tuple(weight.shape) == (width, 128), f"lora shape-changing set seam: video_embedder is {weight.shape}"
    generator = torch.Generator().manual_seed(45)
    control = {"video_embedder.weight": torch.randn(width, 384, generator=generator),
               "video_embedder.bias": torch.randn(width, generator=generator),
               "layers.0.proj.q.lora_A.weight": torch.randn(4, width, generator=generator),
               "layers.0.proj.q.lora_B.weight": torch.randn(width, 4, generator=generator)}
    converted = comfy.lora_convert.convert_lora(dict(control))
    assert {"diffusion_model.video_embedder.set_weight",
            "diffusion_model.video_embedder.bias.set_weight"} <= set(converted), (
        f"lora shape-changing set seam: convert_lora no longer routes the control LoRA: {sorted(converted)}")
    key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
    patches = comfy.lora.load_lora(converted, key_map)
    assert patches[key][0] == "set", f"lora shape-changing set seam: the control weight loads as {patches[key][0]!r}"
    active = patcher.clone()
    active.add_patches(patches, 1.0)
    assert tuple(comfy.lora.calculate_shape(active.patches[key], weight, key)) == (width, 384), (
        "lora shape-changing set seam: calculate_shape no longer reports a set patch's new shape")
    qkv = "diffusion_model.layers.0.proj.qkv.weight"
    qkv_weight = comfy.utils.get_attr(patcher.model, qkv)
    assert comfy.lora.calculate_shape(active.patches[qkv], qkv_weight, qkv) == qkv_weight.shape, (
        "lora shape-changing set seam: an offset LoRA patch onto proj.qkv now changes its shape")
    rank4 = {"diffusion_model.layers.0.proj.o.lora_up.weight": torch.randn(width, 4, generator=generator),
             "diffusion_model.layers.0.proj.o.lora_down.weight": torch.randn(4, width, generator=generator),
             "diffusion_model.layers.0.proj.o.alpha": torch.tensor(4.0)}
    plain = patcher.clone()
    plain.add_patches(comfy.lora.load_lora(rank4, key_map), 1.0)
    plain.add_patches({key: ("diff", (torch.zeros_like(weight),))}, 1.0)
    o_key = "diffusion_model.layers.0.proj.o.weight"
    o_weight = comfy.utils.get_attr(patcher.model, o_key)
    assert comfy.lora.calculate_shape(plain.patches[o_key], o_weight, o_key) == o_weight.shape, (
        "lora shape-changing set seam: a plain LoRA patch now changes the weight's shape")
    assert comfy.lora.calculate_shape(plain.patches[key], weight, key) == weight.shape, (
        "lora shape-changing set seam: a same-shape diff patch now changes the weight's shape")


def _assert_dynamic_prompt_asymmetry_seam() -> None:
    """CLIP Text Encode's text runs the browser's dynamic prompt pass; Text (Multiline)'s value does not.

    The MOSS-Image layout template (tools/gen_templates.py) types its JSON layout
    into a Text (Multiline) node and wires it to CLIP Text Encode's ``text``,
    because the frontend rewrites ``{a|b}`` braces on Queue in a widget that
    declares ``dynamicPrompts``. A connected input runs no such pass, so the JSON
    reaches the encoder byte for byte only while the primitive's widget declares
    none. Pinned here from both schemas as the running ComfyUI publishes them.
    """
    import comfy_extras.nodes_primitive as primitives
    import nodes
    from comfy_api.internal import _ComfyNodeInternal

    text = nodes.CLIPTextEncode.INPUT_TYPES()["required"]["text"]
    assert text[0] == "STRING" and text[1].get("dynamicPrompts") is True, (
        f"dynamic prompt asymmetry seam: CLIPTextEncode.text is now {text!r}")
    infos = [cls.GET_NODE_INFO_V1() for cls in vars(primitives).values()
             if isinstance(cls, type) and issubclass(cls, _ComfyNodeInternal)
             and cls.__module__ == primitives.__name__]
    info = next((row for row in infos if row.get("name") == "PrimitiveStringMultiline"), None)
    assert info is not None, (
        "dynamic prompt asymmetry seam: comfy_extras/nodes_primitive.py no longer defines PrimitiveStringMultiline")
    value = info["input"]["required"]["value"]
    assert value[0] == "STRING" and value[1].get("multiline") is True, (
        f"dynamic prompt asymmetry seam: PrimitiveStringMultiline.value is now {value!r}")
    assert not value[1].get("dynamicPrompts"), (
        "dynamic prompt asymmetry seam: PrimitiveStringMultiline.value now declares dynamicPrompts, so the "
        "browser would rewrite the MOSS-Image layout graph's JSON braces on Queue")


def _assert_weight_only_linear_dispatch_seam() -> None:
    """The route ComfyUI's weight-only quantized Linear takes into its kernel.

    ComfyUI quantizes no activation for a format whose ``quantize_input`` is
    False, and its mixed-precision Linear then hands ``F.linear`` the
    activation as it arrives. Outside inference mode, as on a Worker's
    sampling thread, ``aten.linear`` decomposes: a 3-D activation whose
    leading strides chain folds into ``mm``, which reaches comfy-kitchen's
    INT8 GEMM, and any other one expands the weight, which no layout handles,
    so the weight is dequantized and the linear runs unquantized. The pack
    guards both ends (adapters/int8_linear_layout.py hands these layouts 2-D
    rows, and adapters/flux3_adaln_parity.py ``at_stock_rows`` returns fresh
    strides). Pinned on a real INT8 ConvRot Linear: the format table the
    wrapper copies, the reshape the Linear skips, and ``mm``, never
    ``expand``, for the rows ``at_stock_rows`` keeps and for the prefix slice
    the old batch-1 layout handed over once it is behind the wrapper, each
    equal to the inference-mode result, stock's own, bit for bit. Whether
    that slice unwrapped still takes ``expand`` depends on the torch build,
    so the seam only reports it.
    """
    import json

    import comfy.ops
    import torch
    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor

    from dgx_monarch.adapters import int8_linear_layout
    from dgx_monarch.adapters.flux3_adaln_parity import at_stock_rows

    weight_only = {name: config["comfy_tensor_layout"] for name, config in QUANT_ALGOS.items()
                   if config.get("quantize_input", True) is False}
    assert weight_only == dict(int8_linear_layout.WEIGHT_ONLY_FORMATS), (
        "weight-only linear dispatch seam: ComfyUI's formats that quantize no activation are "
        f"now {weight_only}, and adapters/int8_linear_layout.py copies "
        f"{dict(int8_linear_layout.WEIGHT_ONLY_FORMATS)}")
    source = inspect.getsource(comfy.ops.mixed_precision_ops)
    for phrase in ('.get("quantize_input", True)', "if _use_quantized and quantize_input:",
                   "input.reshape(-1, input_shape[-1])",
                   "weight_only_quant = _use_quantized and not quantize_input"):
        assert phrase in source, (
            "weight-only linear dispatch seam: the quantized Linear no longer reads "
            f"{phrase!r}, so its 2-D reshape may now cover the weight-only formats")

    generator = torch.Generator().manual_seed(47)
    width, out_features, rows, stock_rows = 256, 64, 5, 9
    packed = QuantizedTensor.from_float(
        (torch.randn(out_features, width, generator=generator) * 0.02).to(torch.bfloat16),
        "TensorWiseINT8Layout", is_weight=True, per_channel=True, convrot=True,
        convrot_groupsize=256)
    conf = torch.tensor(list(json.dumps(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode()),
        dtype=torch.uint8)

    def build():
        linear = comfy.ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16).Linear(
            width, out_features, bias=False)
        linear.load_state_dict({"weight": packed._qdata.clone(),
                                "weight_scale": packed._params.scale.clone(),
                                "comfy_quant": conf.clone()}, strict=False)
        assert linear.quant_format == "int8_tensorwise" and linear.weight._params.convrot, (
            "weight-only linear dispatch seam: an INT8 ConvRot state dict no longer loads "
            "as a ConvRot QuantizedTensor")
        return linear

    seen: list[str] = []
    own = "__torch_dispatch__" in QuantizedTensor.__dict__
    stock_dispatch = inspect.getattr_static(QuantizedTensor, "__torch_dispatch__")

    def spy(cls, func, types, args=(), kwargs=None):
        seen.append(func.overloadpacket.__name__)
        return stock_dispatch.__func__(cls, func, types, args, kwargs)

    def route(linear, x):
        seen.clear()
        return linear(x), list(seen)

    QuantizedTensor.__torch_dispatch__ = classmethod(spy)
    try:
        plain, wrapped = build(), build()
        assert int8_linear_layout.install(wrapped) == 1, (
            "weight-only linear dispatch seam: the pack's wrapper no longer takes a loaded "
            "INT8 ConvRot Linear")
        piece = torch.randn(1, rows, width, generator=generator).to(torch.bfloat16)
        with torch.inference_mode():
            stock, _path = route(plain, piece.clone())
        with torch.no_grad():
            kept = at_stock_rows(lambda call: call * 1, piece, stock_rows)
            out, path = route(plain, kept)
            assert "mm" in path and "expand" not in path and torch.equal(out, stock), (
                f"weight-only linear dispatch seam: the rows at_stock_rows keeps now dispatch {path}")
            call = torch.zeros(1, stock_rows, width, dtype=torch.bfloat16)
            call[:, :rows] = piece
            stale = call[:, :rows].contiguous()
            _out, unwrapped = route(plain, stale)
            out, path = route(wrapped, stale)
            assert "mm" in path and "expand" not in path and torch.equal(out, stock), (
                f"weight-only linear dispatch seam: behind the wrapper the slice dispatches {path}")
            if "expand" not in unwrapped:
                print("weight-only linear dispatch seam: this torch already folds the unwrapped "
                      f"prefix slice ({unwrapped}); the wrapper has no expand route to close here")
    finally:
        if own:
            QuantizedTensor.__torch_dispatch__ = stock_dispatch
        else:
            del QuantizedTensor.__torch_dispatch__


SEAMS: tuple[tuple[str, Callable[[], None]], ...] = (
    ("sampler callback", _assert_sampler_callback_seam),
    ("packed latent layout", _assert_packed_latent_layout_seam),
    ("latent format", _assert_latent_format_seam),
    ("empty latent channels", _assert_empty_latent_channel_seam),
    ("noise", _assert_noise_seam),
    ("conditioning wire", _assert_conditioning_wire_seam),
    ("guider and sampler", _assert_guider_and_sampler_seam),
    ("load path", _assert_load_path_seam),
    ("batch cycling", _assert_batch_cycling_seam),
    ("keyed wrapper", _assert_keyed_wrapper_seam),
    ("cfg split", _assert_cfg_split_seam),
    ("cond dispatch", _assert_cond_dispatch_seam),
    ("prepare sampling guard", _assert_prepare_sampling_guard_seam),
    ("minimax h3 audio carry", _assert_minimax_h3_audio_carry_seam),
    ("ltx rope payload", _assert_ltx_rope_payload_seam),
    ("ideogram4 rope payload", _assert_ideogram4_rope_payload_seam),
    ("ltx metadata config", _assert_ltx_metadata_config_seam),
    ("ltx stg passthrough", _assert_ltx_stg_passthrough_seam),
    ("ltx keyframe embedding", _assert_ltx_keyframe_embedding_seam),
    ("ltx guide attenuation", _assert_ltx_guide_attenuation_seam),
    ("ltx guide bias layout", _assert_ltx_guide_bias_layout_seam),
    ("minimax h3 guide layout", _assert_minimax_h3_guide_layout_seam),
    ("minimax h3 guide inputs", _assert_minimax_h3_guide_inputs_seam),
    ("minimax h3 latent mask", _assert_minimax_h3_latent_mask_seam),
    ("minimax h3 pdd head", _assert_minimax_h3_pdd_head_seam),
    ("nvfp4 activation scale", _assert_nvfp4_activation_scale_seam),
    ("chroma text stream", _assert_chroma_text_stream_seam),
    ("cfg combine", _assert_cfg_combine_seam),
    ("loader model folders", _assert_loader_model_folder_seam),
    ("fsdp shard size", _assert_fsdp_shard_size_seam),
    ("resident ledger", _assert_resident_ledger_seam),
    ("offload pin", _assert_offload_pin_seam),
    ("outside forward modules", _assert_outside_forward_module_seam),
    ("flux family text", _assert_flux_family_text_stream_seam),
    ("cogvideox layout", _assert_cogvideox_layout_seam),
    ("media load inputs", _assert_media_load_inputs_seam),
    ("waiver history", _assert_waiver_history_seam),
    ("lumina attention container", _assert_lumina_attention_container_seam),
    ("attention keyword handoff", _assert_attention_keyword_handoff_seam),
    ("cfg++ uncond pass", _assert_cfg_pp_uncond_pass_seam),
    ("flux3 conds branch", _assert_flux3_conds_branch_seam),
    ("moss_image conds", _assert_moss_image_conds_seam),
    ("moss_image block surface", _assert_moss_image_block_surface_seam),
    ("skip layer guidance simple", _assert_skip_layer_guidance_simple_seam),
    ("lora shape-changing set", _assert_lora_shape_changing_set_seam),
    ("dynamic prompt asymmetry", _assert_dynamic_prompt_asymmetry_seam),
    ("weight-only linear dispatch", _assert_weight_only_linear_dispatch_seam),
)


def main() -> None:
    _ensure_comfy()
    _ensure_repo()
    if not SEAMS:
        raise SystemExit("comfy seam contract suite is vacuous")
    for name, check in SEAMS:
        try:
            check()
        except Exception as exc:
            raise RuntimeError(f"comfy seam contract broken: {name}") from exc
        print(f"seam ok: {name}")
    print(f"comfy seam behavioral contracts green: {len(SEAMS)} seams")


if __name__ == "__main__":
    main()
