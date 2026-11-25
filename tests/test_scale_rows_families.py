"""The shared NVFP4 activation scale counts only real rows, family by family.

Without a static input scale an NVFP4 Linear quantizes against
``amax(abs(x)) / (448 * 6)``, and under sequence parallelism the shared scale
reduces each rank's amax over the group (adapters/quant_activation_scale.py).
A rank's input also holds the rows ``shard_seq`` added so a stream divides the
shard; after a block's norm and modulation they are not zero, and one of them
could set the scale the whole group quantizes against
(adapters/scale_rows.py).

Each test here runs one family's comfy model at toy size with every Linear
wrapped as NVFP4 under the shared scale (``ScaleRecorder``): once whole, as
one GPU runs it, and once through the family's own two-rank harness with the
divisibility pad rows filled with NaN (``nan_pads``). Every rank must reduce
every Linear call to the amax one GPU took over the whole tensor: a pad row
would read NaN, and a real row left out would read low. The families with an
exclusion test file of their own (CogVideoX, the Wan family, Kandinsky5,
Anima) carry this test there.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from pad_rows_comfy_helpers import ScaleRecorder, assert_scales_match, nan_pads, real_comfy

WORLD = 2


def _whole(monkeypatch, model, call) -> list:
    """One GPU's calls: every Linear wrapped, the identity as the group reduce."""
    recorder = ScaleRecorder(monkeypatch)
    assert ScaleRecorder.install(model) > 0
    with torch.no_grad():
        call(model)
    return recorder, recorder.take()[0]


# Krea2: odd text and odd image streams, so both segments of the joint stream pad.
@pytest.mark.parametrize("text_rows,latent_hw", [(5, (5, 6)), (5, (4, 8)), (4, (6, 6))])
def test_krea2(monkeypatch, text_rows, latent_hw):
    import test_krea2_comfy_exactness as k2
    from dgx_monarch.adapters.krea2 import Krea2Adapter

    with real_comfy("scale-krea2", "comfy.ldm.krea2.model") as modules:
        args = k2._inputs(text_rows, latent_hw, 1)
        recorder, expected = _whole(
            monkeypatch, k2._toy(modules),
            lambda m: type(m)._forward(m, *args, transformer_options={}))
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, Krea2Adapter)
        k2._run(monkeypatch, modules, k2._toy(modules), args, {}, pure_ulysses=True)
        assert_scales_match(expected, recorder.take(), WORLD)


# Ernie-Image: one joint [text, image] stream; odd text, odd image, or both.
@pytest.mark.parametrize("grid,text_rows", [((3, 4), 5), ((3, 3), 4), ((3, 3), 5)])
def test_ernie(monkeypatch, grid, text_rows):
    import test_ernie_stock_equivalence as er
    from dgx_monarch.adapters.ernie import ErnieAdapter

    with real_comfy("scale-ernie", "comfy.ldm.ernie.model") as modules:
        inputs = er._inputs(grid, text_rows, 1)
        recorder, expected = _whole(
            monkeypatch, er._tiny_ernie(modules),
            lambda m: m(*inputs, transformer_options={}))
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, ErnieAdapter)
        er._sharded(monkeypatch, er._tiny_ernie(modules), inputs, {},
                    modules[1].optimized_attention)
        assert_scales_match(expected, recorder.take(), WORLD)


# FLUX.3: the image stream alone in the double blocks, then the joint stream
# with reference images and text in the single blocks.
@pytest.mark.parametrize("case", ["odd-grid", "one-ref", "two-refs"])
def test_flux3(monkeypatch, case):
    import test_flux3_comfy_exactness as f3
    from dgx_monarch.adapters.flux3 import Flux3Adapter

    with real_comfy("scale-flux3", "comfy.ldm.flux3.model") as (ops, flux3_model):
        comfy = SimpleNamespace(ops=ops, model=flux3_model)
        grid, refs, text_rows, batch = f3.CASES[case]
        inputs = f3._inputs(grid, refs, text_rows, batch)
        recorder, expected = _whole(
            monkeypatch, f3._tiny(comfy), lambda m: m(*inputs, transformer_options={}))
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, Flux3Adapter)
        f3._sharded(monkeypatch, f3._tiny(comfy), inputs, {}, flux3_model.optimized_attention)
        assert_scales_match(expected, recorder.take(), WORLD)


# Lens: separate image and text streams in its double blocks.
@pytest.mark.parametrize("grid,text_rows", [((3, 3), 5), ((2, 4), 5), ((3, 3), 4)])
def test_lens(monkeypatch, grid, text_rows):
    import test_lens_comfy_exactness as ln
    from dgx_monarch.adapters.lens import LensAdapter

    with real_comfy("scale-lens", "comfy.ldm.lens.model") as (ops, lens_model):
        x, timestep, context, mask = inputs = ln._inputs(grid, text_rows, 1, "none")
        stock = ln._build(ops, lens_model)
        recorder, expected = _whole(
            monkeypatch, ln._build(ops, lens_model, stock.state_dict()),
            lambda m: m._forward(x, timestep, context, attention_mask=mask))
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, LensAdapter)
        ln._sharded(monkeypatch, ops, lens_model, stock, inputs, pure_ulysses=True)
        assert_scales_match(expected, recorder.take(), WORLD)


# Lens with a uniform trailing text mask: pure Ulysses keeps the masked rows,
# as stock does, so their amax is stock's. Ring and hybrid trim them, so with
# NVFP4 Linears under the shared scale every rank refuses, class P, before any
# Linear runs.
@pytest.mark.parametrize("grid,text_rows", [((3, 4), 6), ((2, 4), 8), ((3, 3), 7)])
def test_lens_trailing_text_mask(monkeypatch, grid, text_rows):
    import test_lens_comfy_exactness as ln
    from dgx_monarch.adapters import base
    from dgx_monarch.adapters.lens import LensAdapter
    from dgx_monarch.refusal import RefusalClass, parse_refusal_tag

    with real_comfy("scale-lens", "comfy.ldm.lens.model") as (ops, lens_model):
        x, timestep, context, mask = inputs = ln._inputs(grid, text_rows, 1, "trailing")
        stock = ln._build(ops, lens_model)
        recorder, expected = _whole(
            monkeypatch, ln._build(ops, lens_model, stock.state_dict()),
            lambda m: m._forward(x, timestep, context, attention_mask=mask))
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, LensAdapter)
        ln._sharded(monkeypatch, ops, lens_model, stock, inputs, pure_ulysses=True)
        assert_scales_match(expected, recorder.take(), WORLD)

        models = [ln._build(ops, lens_model, stock.state_dict()) for _ in range(WORLD)]
        for model in models:
            LensAdapter().inject_usp(model, base.InjectionContext(
                topology_sp=WORLD, usp_attention=base.make_usp_attention("TORCH_FLASH"),
                pure_ulysses=False))
        for rank, model in enumerate(models):  # it refuses before any collective
            with pytest.raises(base.UnsupportedModelError) as caught, torch.no_grad():
                model._forward(x, timestep, context, attention_mask=mask)
            assert parse_refusal_tag(str(caught.value)).refusal_class is RefusalClass.PHYSICS
            assert "pads trailing tokens" in str(caught.value), rank
        assert not any(recorder.take().values())  # no Linear ran, so none reduced


# HunyuanVideo 1.5: a text and an image stream in the double blocks, the joint
# stream in the single blocks; image and video, odd text and odd image.
@pytest.mark.parametrize("case", ["image-odd-text-odd-image", "image-odd-image",
                                  "video-vision-odd-text"])
def test_hunyuan(monkeypatch, case):
    import test_hunyuan_stock_equivalence as hy
    from dgx_monarch.adapters.hunyuan import HunyuanAdapter

    spec = hy.CASES[case]
    with real_comfy("scale-hunyuan", "comfy.ldm.hunyuan_video.model") as modules:
        x, timesteps, context, kwargs = hy._inputs(spec, 1)
        build = {"video": spec.get("video", False), "guidance": spec.get("guidance", False),
                 "singles": spec.get("singles", 2)}
        hy._MODELS.clear()
        recorder, expected = _whole(
            monkeypatch, hy._build(modules, **build),
            lambda m: m._forward(x, timesteps, context, transformer_options={}, **kwargs))
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, HunyuanAdapter)
        hy._MODELS.clear()
        models = [hy._build(modules, **build) for _ in range(WORLD)]
        hy._sharded(monkeypatch, models, x, timesteps, context, kwargs)
        hy._MODELS.clear()
        assert_scales_match(expected, recorder.take(), WORLD)


# Boogu: four sharded loops (noise refiner, reference refiner, double stream,
# single stream), each on its own stream.
@pytest.mark.parametrize("text_rows,grid,refs", [(5, (2, 3), ()), (5, (3, 3), (1, 2)),
                                                 (4, (3, 3), (2,))])
def test_boogu(monkeypatch, text_rows, grid, refs):
    import test_boogu_comfy_exactness as bo
    from dgx_monarch.adapters.boogu import BooguAdapter

    with real_comfy("scale-boogu", "comfy.ldm.boogu.model",
                    "comfy.ldm.modules.attention") as comfy:
        x, timesteps, context, ref_latents = bo._inputs(text_rows, grid, refs, 1)
        torch.manual_seed(77)
        recorder, expected = _whole(
            monkeypatch, bo._model(comfy),
            lambda m: m.forward(x, timesteps, context, text_rows, ref_latents=ref_latents,
                                transformer_options={}))
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, BooguAdapter)
        torch.manual_seed(77)   # bo._run builds its stock model, then the ranks from it
        bo._run(monkeypatch, comfy, text_rows, grid, refs, 1)
        assert_scales_match(expected, recorder.take(), WORLD)


def _sdpa_kernel(q, k, v, heads):
    """Torch's attention on comfy's (B, L, H*D) layout."""
    def split(t):
        return t.reshape(t.shape[0], t.shape[1], heads, -1).transpose(1, 2)

    out = torch.nn.functional.scaled_dot_product_attention(split(q), split(k), split(v))
    return out.transpose(1, 2).reshape(q.shape[0], q.shape[1], -1)


def _generic_ranks(monkeypatch, adapter, model, kernel, call, *, ctx_attention=None,
                   patch=()):
    """Two rank threads on the shared stub fabric (pad_rows_comfy_helpers), each
    model bound by ``adapter`` and wrapped by the recorder ``on_inject`` set."""
    from dgx_monarch.adapters import base
    from dgx_monarch.adapters.base import InjectionContext
    from pad_rows_comfy_helpers import install, replicas, run_ranks

    harness = install(monkeypatch, kernel)
    for module, name, value in patch:
        monkeypatch.setattr(module, name, value)
    attention = ctx_attention or base.make_usp_attention("TORCH_FLASH")
    models = replicas(model)
    for rank_model in models:
        adapter.inject_usp(rank_model, InjectionContext(
            topology_sp=WORLD, usp_attention=attention, pure_ulysses=True))
    return run_ranks(harness, models, call)


# Qwen-Image 2.1: the replicated [text, references] prefix and this rank's
# target rows in one block call; an odd target pads.
@pytest.mark.parametrize("lat,refs", [((3, 5), 0), ((4, 5), 1), ((3, 3), 1)])
def test_qwen_image21(monkeypatch, lat, refs):
    from types import SimpleNamespace as NS

    from dgx_monarch.adapters.qwen_image21 import QwenImage21Adapter

    with real_comfy("scale-q21", "comfy.ldm.qwen_image21.model") as (ops, q21):
        torch.manual_seed(31)
        model = q21.QwenImage21Transformer2DModel(
            in_channels=64, out_channels=64, num_layers=2, attention_head_dim=6,
            num_attention_heads=2, context_in_dim=4, mlp_ratio=1, axes_dims_rope=(2, 2, 2),
            operations=ops.disable_weight_init)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(mean=0.0, std=0.05)
        model.reset_prefix_cache(False)
        generator = torch.Generator().manual_seed(9)
        x = torch.randn(1, 64, *lat, generator=generator)
        context = torch.randn(1, 5, 4, generator=generator)
        ref_latents = [torch.randn(1, 64, 2, 2, generator=generator) for _ in range(refs)]
        slots = [2] * refs
        t = torch.tensor([0.4])

        def call(m):
            return m._forward(x, t, context, ref_latents=ref_latents, image_slots=slots,
                              transformer_options={})

        stock = q21.QwenImage21Transformer2DModel(
            in_channels=64, out_channels=64, num_layers=2, attention_head_dim=6,
            num_attention_heads=2, context_in_dim=4, mlp_ratio=1, axes_dims_rope=(2, 2, 2),
            operations=ops.disable_weight_init)
        stock.load_state_dict(model.state_dict())
        stock.reset_prefix_cache(False)
        recorder, expected = _whole(monkeypatch, stock, call)
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, QwenImage21Adapter)
        _generic_ranks(monkeypatch, QwenImage21Adapter(), model, q21.optimized_attention, call,
                       ctx_attention=NS(effective_kernel="TORCH_FLASH"))
        assert_scales_match(expected, recorder.take(), WORLD)


# MiniMax H3 on the image graph: one packed (rows, features) stream sharded on
# dim 0, video and audio heads after the gather; an odd latent pads.
@pytest.mark.parametrize("lat,ref_count", [((5, 7), 0), ((6, 4), 1), ((4, 6), 2)])
def test_minimax_h3(monkeypatch, lat, ref_count):
    import test_p_h3_image_comfy_exactness as ph
    from dgx_monarch.adapters import minimax_h3
    from dgx_monarch.adapters.minimax_h3 import MiniMaxH3Adapter

    with real_comfy("scale-h3", "comfy.ldm.minimax.model", "comfy.model_base") as (
            ops, h3, model_base):
        comfy = SimpleNamespace(h3=h3, model_base=model_base, ops=ops)
        generator = torch.Generator().manual_seed(3)
        video = torch.randn(1, 24, 1, *lat, generator=generator)
        audio = torch.randn(1, 32, 2, 2, generator=generator)
        context = torch.randn(1, 9, ph.TEXT_DIM, generator=generator)
        timestep = torch.tensor([810.083])
        refs = ph._refs(ref_count, lat)

        def call(network):
            options = {"minimax_h3_sigma_shift_video": 1.0, "minimax_h3_sigma_shift_audio": 1.0,
                       "sample_sigmas": torch.tensor([1.0, 0.810083, 0.1, 0.0]),
                       "sigmas": timestep / 1000.0}
            payload = ph._payload(comfy, context.shape[1], lat, refs, 1.0)
            with torch.inference_mode():
                return network.forward([video.clone(), audio.clone()], timestep,
                                       context.clone(), options, minimax_payload=payload)

        recorder, expected = _whole(monkeypatch, ph._tiny(comfy), call)
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, MiniMaxH3Adapter)
        _generic_ranks(monkeypatch, MiniMaxH3Adapter(), ph._tiny(comfy), h3.optimized_attention,
                       call, patch=((minimax_h3, "sp_world", lambda: WORLD),))
        assert_scales_match(expected, recorder.take(), WORLD)


# MiniMax-H3-Image: the same packed stream with no audio, one latent frame.
@pytest.mark.parametrize("case", ["odd-one-pad", "three-references",
                                  "text-across-the-shard-line-with-tags"])
def test_minimax_h3_image(monkeypatch, case):
    import test_minimax_h3_image_comfy as hi
    from dgx_monarch.adapters import minimax_h3_image
    from dgx_monarch.adapters.minimax_h3_image import MiniMaxH3ImageAdapter

    with real_comfy("scale-h3i", "comfy.ldm.minimax.model", "comfy.ldm.minimax.vae",
                    "comfy.model_base") as (ops, h3, vae, model_base):
        if not hasattr(h3, "MiniMaxH3ImageModel"):
            pytest.skip("this ComfyUI predates MiniMax-H3-Image")
        comfy = SimpleNamespace(h3=h3, model_base=model_base, ops=ops, vae=vae)
        lat, text, run = hi._case(case, 0.5)
        x, context = hi._inputs(lat, text=text)

        def call(network):
            return hi._run(comfy, network, x, context, **run)[0]

        recorder, expected = _whole(monkeypatch, hi._tiny(comfy), call)
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, MiniMaxH3ImageAdapter)
        _generic_ranks(monkeypatch, MiniMaxH3ImageAdapter(), hi._tiny(comfy),
                       h3.optimized_attention, call,
                       patch=((minimax_h3_image, "sp_world", lambda: WORLD),))
        assert_scales_match(expected, recorder.take(), WORLD)


def _flux_patches():
    import test_flux_pad_exclusion as fpe
    from dgx_monarch.adapters import chroma_text, flux_family

    return ((flux_family, "sp_rank", fpe._rank), (chroma_text, "sp_rank", fpe._rank),
            (chroma_text, "sp_world", lambda: WORLD))


# Flux 1 and Flux 2: image and text streams in the double blocks, the joint
# stream in the single blocks; odd image (3 x 3 tokens) and odd text (7 rows),
# whose joint stream (16) divides, and 6 or 8 text rows, whose joint stream pads.
@pytest.mark.parametrize("text_rows", [7, 6, 8])
@pytest.mark.parametrize("family", ["flux2", "flux"])
def test_flux(monkeypatch, family, text_rows):
    import test_flux_views_comfy_exactness as fv
    from dgx_monarch.adapters.flux_family import Flux2Adapter, FluxAdapter

    with real_comfy("scale-flux", "comfy.ldm.flux.model", "comfy.ldm.flux.math",
                    "comfy.sd") as (ops, flux_model, flux_math, sd):
        comfy = SimpleNamespace(ops=ops, model=flux_model, sd=sd)
        generator = torch.Generator().manual_seed(4)
        if family == "flux2":
            build, adapter = (lambda: fv._base(comfy).diffusion_model), Flux2Adapter
            x = torch.randn(1, fv.CHANNELS, 3, 3, generator=generator)
            context = torch.randn(1, text_rows, fv.TEXT_DIM, generator=generator)
            kwargs: dict = {}
        else:
            build, adapter = (lambda: fv._base_flux1(comfy, "flux")[0].diffusion_model), FluxAdapter
            x = torch.randn(1, 16, 6, 6, generator=generator)
            context = torch.randn(1, text_rows, 64, generator=generator)
            kwargs = {"y": torch.randn(1, 32, generator=generator),
                      "guidance": torch.tensor([3.5])}
        t = torch.tensor([0.6])

        def call(m):
            return m(x, t, context, transformer_options={}, **kwargs)

        recorder, expected = _whole(monkeypatch, build(), call)
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, adapter)
        _generic_ranks(monkeypatch, adapter(), build(), flux_math.optimized_attention, call,
                       patch=_flux_patches())
        assert_scales_match(expected, recorder.take(), WORLD)


# Chroma: the Flux block layout with an approximator in place of the time MLP.
@pytest.mark.parametrize("hw,text_rows", [((6, 6), 7), ((6, 8), 5), ((4, 6), 4)])
def test_chroma(monkeypatch, hw, text_rows):
    import test_chroma_cfg_pad_forward as cc
    from dgx_monarch.adapters.flux_family import ChromaAdapter

    with real_comfy("scale-chroma", "comfy.ldm.chroma.model", "comfy.ldm.flux.math") as (
            ops, chroma_model, flux_math):
        generator = torch.Generator().manual_seed(6)
        x = torch.randn(1, cc.RAW_CHANNELS, *hw, generator=generator)
        context = torch.randn(1, text_rows, cc.CONTEXT_DIM, generator=generator)
        t, guidance = torch.tensor([0.5]), torch.tensor([0.3])

        def call(m):
            return m._forward(x, t, context, guidance, None, {})

        recorder, expected = _whole(monkeypatch, cc._tiny_chroma((ops, chroma_model)), call)
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, ChromaAdapter)
        _generic_ranks(monkeypatch, ChromaAdapter(), cc._tiny_chroma((ops, chroma_model)),
                       flux_math.optimized_attention, call, patch=_flux_patches())
        assert_scales_match(expected, recorder.take(), WORLD)


# Qwen-Image: separate text and image streams in its dual-stream blocks; odd
# text, odd image (3 x 3 tokens), or both.
@pytest.mark.parametrize("hw,text_rows", [((6, 6), 7), ((6, 8), 5), ((6, 6), 4)])
def test_qwen_image(monkeypatch, hw, text_rows):
    import test_flux_pad_exclusion as fpe
    from dgx_monarch.adapters import chroma_text, qwen_image
    from dgx_monarch.adapters.qwen_image import QwenImageAdapter

    with real_comfy("scale-qwen", "comfy.ldm.qwen_image.model") as (ops, qwen_model):
        def build():
            torch.manual_seed(12)
            model = qwen_model.QwenImageTransformer2DModel(
                patch_size=2, in_channels=64, out_channels=16, num_layers=2,
                attention_head_dim=16, num_attention_heads=2, joint_attention_dim=12,
                pooled_projection_dim=8, axes_dims_rope=(4, 6, 6), dtype=torch.float32,
                device="cpu", operations=ops.disable_weight_init)
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    centre = 1.0 if "norm" in name else 0.0
                    parameter.copy_(torch.randn_like(parameter) * 0.1 + centre)
            return model

        generator = torch.Generator().manual_seed(8)
        x = torch.randn(1, 16, 1, *hw, generator=generator)
        context = torch.randn(1, text_rows, 12, generator=generator)
        t = torch.tensor([0.5])

        def call(m):
            return m._forward(x, t, context, transformer_options={})

        recorder, expected = _whole(monkeypatch, build(), call)
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, QwenImageAdapter)
        _generic_ranks(monkeypatch, QwenImageAdapter(), build(), _sdpa_kernel,
                       call, patch=((qwen_image, "sp_rank", fpe._rank),
                                    (chroma_text, "sp_rank", fpe._rank),
                                    (chroma_text, "sp_world", lambda: WORLD)))
        assert_scales_match(expected, recorder.take(), WORLD)


# LTX 2.3 (LTXAV): the video and the audio stream each pad at their own tail,
# and the modality cross attentions read the other stream's rows.
@pytest.mark.parametrize("case", [(13, 0, 1, False), (13, 6, 1, False), (12, 7, 2, True)])
def test_ltx(monkeypatch, case):
    import test_ltx_comfy_exactness as lx
    from dgx_monarch.adapters.ltx import LTXAdapter

    with real_comfy("scale-ltx", "comfy.ldm.lightricks.model", "comfy.ldm.lightricks.av_model",
                    "comfy.quant_ops") as modules:
        ops, _model_module, av_module, _quant_ops = modules

        def make_call():
            x, context, timestep, pe = lx._inputs(av_module, *case)
            return (x, context, None, timestep, pe), {}

        def whole(m):
            args, kwargs = make_call()
            return type(m)._process_transformer_blocks(
                m, *args, transformer_options={"optimized_attention_override": lx._Attention(ops)},
                self_attention_mask=None, **kwargs)

        recorder, expected = _whole(monkeypatch, lx._toy(modules), whole)
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, LTXAdapter)
        lx._compare(monkeypatch, modules, lx._toy(modules), make_call)
        assert_scales_match(expected, recorder.take(), WORLD)


# Ideogram 4: one packed [text, image] stream, odd and even text and grids.
@pytest.mark.parametrize("text_rows,grid", [(5, (2, 4)), (4, (3, 3)), (5, (3, 3))])
def test_ideogram4(monkeypatch, text_rows, grid):
    import test_ideogram4_stock_equivalence as ig
    from dgx_monarch.adapters.pixeldit import Ideogram4Adapter

    with real_comfy("scale-ideogram4", "comfy.ldm.ideogram4.model") as (ops, ideogram):
        x, timesteps, context = ig._inputs(text_rows, grid, 1)
        recorder, expected = _whole(
            monkeypatch, ig._model(ops, ideogram),
            lambda m: m._forward(x, timesteps, context, transformer_options={}))
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, Ideogram4Adapter)
        ig._sharded(monkeypatch, ig._model(ops, ideogram), x, timesteps, context)
        assert_scales_match(expected, recorder.take(), WORLD)


# Ideogram 4 with a padded text mask: stock runs the masked text tokens as a
# segment of their own, and their values enter each Linear's statistic, so the
# pack's segment call must give them stock's values (a tail mask, a mask
# mid-prompt, on a packed total that pads and one that does not).
@pytest.mark.parametrize("mask,grid", [([1, 1, 1, 0, 0], (2, 4)), ([1, 0, 1, 1, 0], (3, 3)),
                                       ([1, 0, 1, 1, 1], (2, 4))])
def test_ideogram4_masked_text(monkeypatch, mask, grid):
    import test_ideogram4_stock_equivalence as ig
    from dgx_monarch.adapters.pixeldit import Ideogram4Adapter

    with real_comfy("scale-ideogram4", "comfy.ldm.ideogram4.model") as (ops, ideogram):
        x, timesteps, context = ig._inputs(len(mask), grid, 1)
        attention_mask = torch.tensor([mask])
        recorder, expected = _whole(
            monkeypatch, ig._model(ops, ideogram),
            lambda m: ig._stock_masked(m, x, timesteps, context, attention_mask))
        nan_pads(monkeypatch)
        recorder.on_inject(monkeypatch, Ideogram4Adapter)
        model = ig._model(ops, ideogram)
        _outputs, kernel, _drops = ig._sharded(monkeypatch, model, x, timesteps, context,
                                               attention_mask=attention_mask)
        assert_scales_match(expected, recorder.take(), WORLD)
        # Each block: the real rows' call, then the masked rows' segment.
        rows = grid[0] * grid[1] + len(mask)
        masked = len(mask) - sum(mask)
        for rank in range(WORLD):
            keys = [call[3].shape[1] for call in kernel.calls if call[0] == rank]
            assert keys == [rows - masked, masked] * len(model.layers), rank
