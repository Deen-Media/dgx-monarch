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
                              nvfp4 alone takes, and the per-module scales the
                              quantized Linear reads
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
                              method runs, which FSDP has to replicate
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
    arrives, which no identity gate can see.
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
        FORWARD_PATH_NAME_OWNERS,
        OUTSIDE_FORWARD_MODULE_ATTRS,
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
    assert set(reaching) == set(OUTSIDE_FORWARD_MODULE_ATTRS), (
        "outside forward modules seam: the supported families whose extra_conds "
        f"calls a diffusion-model method are {sorted(reaching)}, and the table "
        f"names {sorted(OUTSIDE_FORWARD_MODULE_ATTRS)}"
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
