"""Each stamped family's own forward hands the stamp to its cfg-pad forward.

The pad stamps each entry's row count as a model cond (adapters/cfg_pad_rows.py)
and ComfyUI's cond path carries it to ``model.forward`` as a keyword. From there
each family's stock ``forward`` passes its keywords through its wrapper
executor, the cfg split among the wrappers, to ``_forward``, which the adapter
binds. A family whose ``forward`` dropped an unknown keyword would leave the
pad in the negative's attention with nothing to say so, and the tests that call
``_forward`` directly would not see it. So each family's own model at toy size
runs a shorter negative, padded and stamped, through ComfyUI's cond path
(``convert_cond``, ``encode_model_conds``, ``get_area_and_mult``,
``can_concat_cond``, ``cond_cat``), its real ``forward`` and the cfg split on
two ranks, and each rank is held to stock's forward on its conditioning alone,
bit for bit. Krea2 has the same run in tests/test_cfg2_zeroed_negative.py.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from dgx_monarch.actor.sampling import equalize_cond_lengths
from dgx_monarch.adapters import cfg_pad_rows
from dgx_monarch.adapters import cfg_parallel as cfg_parallel_module
from dgx_monarch.adapters.kandinsky5 import Kandinsky5Adapter
from dgx_monarch.adapters.lens import LensAdapter
from dgx_monarch.adapters.mage_flow import MageFlowAdapter
from dgx_monarch.adapters.qwen_image import QwenImageAdapter
from dgx_monarch.adapters.zimage import ZImageAdapter

KEY = cfg_pad_rows.TEXT_ROWS
PROMPT_ROWS, NEGATIVE_ROWS = 10, 5
LDM = ("comfy.ldm.qwen_image.model", "comfy.ldm.mage_flow.model", "comfy.ldm.lumina.model",
       "comfy.ldm.kandinsky5.model", "comfy.ldm.lens.model")


@pytest.fixture(scope="module")
def comfy():
    from real_comfy_helpers import real_comfy

    with real_comfy("cfg-pad-stamp-delivery", *LDM, "comfy.samplers", "comfy.sampler_helpers",
                    "comfy.conds", "comfy.patcher_extension") as modules:
        ops, qwen, mage, lumina, k5, lens, samplers, helpers, conds, pe = modules
        yield {"ops": ops, "qwen": qwen, "mage": mage, "lumina": lumina, "k5": k5, "lens": lens,
               "samplers": samplers, "helpers": helpers, "conds": conds, "pe": pe}


def _fill(model, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            centre = 1.0 if "norm" in name else 0.0
            parameter.copy_(0.1 * torch.randn(parameter.shape, generator=generator) + centre)
    return model


def _qwen(c):
    return _fill(c["qwen"].QwenImageTransformer2DModel(
        patch_size=2, in_channels=64, out_channels=16, num_layers=2, attention_head_dim=16,
        num_attention_heads=2, joint_attention_dim=12, pooled_projection_dim=8,
        axes_dims_rope=(4, 6, 6), dtype=torch.float32, device="cpu",
        operations=c["ops"].disable_weight_init), 21)


def _mage(c):
    return _fill(c["mage"].MageFlowTransformer2DModel(
        in_channels=4, out_channels=4, num_layers=1, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=12, axes_dims_rope=(2, 2, 4),
        dtype=torch.float32, device=torch.device("cpu"),
        operations=c["ops"].disable_weight_init), 22)


def _zimage(c):
    # Z-Image's 256-wide modulation needs dim >= 256; the toy keeps Lumina's
    # own, which leaves the text path and the forward chain as they are.
    return _fill(c["lumina"].NextDiT(
        patch_size=2, in_channels=4, dim=32, n_layers=1, n_refiner_layers=1, n_heads=2,
        n_kv_heads=2, multiple_of=16, ffn_dim_multiplier=1.0, qk_norm=True, cap_feat_dim=12,
        axes_dims=(4, 6, 6), axes_lens=(64, 32, 32), dtype=torch.float32, device="cpu",
        operations=c["ops"].disable_weight_init), 23)


def _kandinsky5(c):
    return _fill(c["k5"].Kandinsky5(
        in_visual_dim=4, out_visual_dim=4, in_text_dim=12, in_text_dim2=10, time_dim=16,
        model_dim=32, ff_dim=64, visual_embed_dim=16, patch_size=(1, 2, 2), num_text_blocks=1,
        num_visual_blocks=2, axes_dims=(4, 6, 6), dtype=torch.float32, device="cpu",
        operations=c["ops"].disable_weight_init), 24)


def _lens(c):
    return _fill(c["lens"].LensTransformer2DModel(
        patch_size=2, in_channels=8, out_channels=2, num_layers=2, attention_head_dim=8,
        num_attention_heads=2, enc_hidden_dim=12, axes_dims_rope=(2, 2, 4),
        multi_layer_encoder_feature=True, selected_layer_index=(0, 1), dtype=torch.float32,
        device="cpu", operations=c["ops"].disable_weight_init), 25)


def _plain(conds, params):
    return {"c_crossattn": conds.CONDRegular(params["cross_attn"])}


def _zimage_conds(conds, params):
    # Lumina2.extra_conds with no attention mask: the caption length rides as a
    # constant, read on the padded text, so the padded pair concatenates.
    cross_attn = params["cross_attn"]
    return {"c_crossattn": conds.CONDRegular(cross_attn),
            "num_tokens": conds.CONDConstant(cross_attn.shape[1])}


def _kandinsky5_conds(conds, params):
    return {"c_crossattn": conds.CONDRegular(params["cross_attn"]),
            "y": conds.CONDRegular(params["pooled_output"])}


@dataclass(frozen=True)
class Family:
    name: str
    build: Callable[[dict], Any]
    adapter: type
    x_shape: tuple[int, ...]
    text_dim: int
    publish: Callable[[Any, dict], dict] = _plain
    pooled_dim: int = 0

    def stock(self, model, x, t, context, pooled):
        """Stock's class forward on one conditioning alone."""
        forward = type(model)._forward
        if self.name == "zimage":
            return forward(model, x, t, context, context.shape[1], transformer_options={})
        if self.name == "kandinsky5":
            return forward(model, x, t, context, pooled, transformer_options={})
        return forward(model, x, t, context, transformer_options={})


FAMILIES = [
    Family("qwen_image", _qwen, QwenImageAdapter, (1, 16, 1, 4, 6), 12),
    Family("mage_flow", _mage, MageFlowAdapter, (1, 4, 4, 6), 12),
    Family("zimage", _zimage, ZImageAdapter, (1, 4, 4, 6), 12, _zimage_conds),
    Family("kandinsky5", _kandinsky5, Kandinsky5Adapter, (1, 4, 2, 4, 6), 12,
           _kandinsky5_conds, pooled_dim=10),
    Family("lens", _lens, LensAdapter, (1, 8, 4, 6), 24),
]


def _model_calls(c, family, positive, negative, x_in, timestep):
    """The calls ComfyUI makes for the pair: one batched call when the two
    concatenate, else one each. Each is (x, t, context, extra keywords)."""
    samplers, helpers, conds = c["samplers"], c["helpers"], c["conds"]

    def extra_conds(**params):
        return family.publish(conds, params)

    items = []
    for side, kind in ((positive, "positive"), (negative, "negative")):
        encoded = samplers.encode_model_conds(
            extra_conds, helpers.convert_cond(side), x_in, "cpu", kind)
        items.append(samplers.get_area_and_mult(encoded[0], x_in, timestep))
    groups = [items] if samplers.can_concat_cond(items[0], items[1]) else [[i] for i in items]
    calls = []
    for group in groups:
        kwargs = samplers.cond_cat([item.conditioning for item in group])
        context = kwargs.pop("c_crossattn")
        repeat = (len(group),) + (1,) * (x_in.ndim - 1)
        calls.append((x_in.repeat(*repeat), timestep.repeat(len(group)), context, kwargs))
    return calls


def _split_on_two_ranks(monkeypatch, c, adapter, model, call):
    """The batched call through the model's own forward and the cfg split, rank 0
    then rank 1; each rank's slice as it would enter the all-gather."""
    pe = c["pe"]
    x, t, context, kwargs = call
    model_options: dict = {}
    cfg_parallel_module.install_cfg_parallel_wrapper(adapter, model_options)
    outputs = []
    for rank in (0, 1):
        monkeypatch.setattr(cfg_parallel_module, "cfg_world", lambda: 2)
        monkeypatch.setattr(cfg_parallel_module, "cfg_rank", lambda rank=rank: rank)
        sent: list = []
        monkeypatch.setattr(cfg_parallel_module, "_all_gather_cfg",
                            lambda local, sent=sent: sent.append(local) or torch.cat([local] * 2))
        options = {**model_options["transformer_options"],
                   "cond_or_uncond": [0, 1], "uuids": ["cond", "uncond"]}
        assert pe.get_all_wrappers(pe.WrappersMP.DIFFUSION_MODEL, options)
        with torch.no_grad():
            model.forward(x, t, context, transformer_options=options, **kwargs)
        outputs.append(sent[0])
    return outputs


@pytest.mark.parametrize("family", FAMILIES, ids=[f.name for f in FAMILIES])
@pytest.mark.parametrize("negative", ["zeroed", "real"])
def test_a_padded_negative_reaches_stock_through_the_familys_own_forward(
        comfy, monkeypatch, family, negative):
    generator = torch.Generator().manual_seed(len(family.name))
    prompt = torch.randn(1, PROMPT_ROWS, family.text_dim, generator=generator)
    short = (torch.zeros(1, NEGATIVE_ROWS, family.text_dim) if negative == "zeroed"
             else torch.randn(1, NEGATIVE_ROWS, family.text_dim, generator=generator))
    pooled = torch.randn(1, family.pooled_dim, generator=generator) if family.pooled_dim else None
    x = torch.randn(*family.x_shape, generator=generator)
    t = torch.tensor([0.4])
    extras = {"pooled_output": pooled} if pooled is not None else {}

    reference_model = family.build(comfy)
    with torch.no_grad():
        reference = [family.stock(reference_model, x, t, cond, pooled) for cond in (prompt, short)]
    adapter = family.adapter()
    positive, padded = equalize_cond_lengths(
        adapter, [[prompt, dict(extras)]], [[short, dict(extras)]], x)
    assert padded[0][0].shape[1] == PROMPT_ROWS
    assert padded[0][1]["model_conds"][KEY].cond.tolist() == [NEGATIVE_ROWS]
    calls = _model_calls(comfy, family, positive, padded, x, t)
    # ComfyUI batches the padded pair, and the stamp rides the batched call.
    assert len(calls) == 1 and calls[0][2].shape[0] == 2
    assert calls[0][3][KEY].tolist() == [PROMPT_ROWS, NEGATIVE_ROWS]

    model = family.build(comfy)
    adapter.inject_cfg_pad_forward(model)
    ranks = _split_on_two_ranks(monkeypatch, comfy, adapter, model, calls[0])
    for rank, (local, expected) in enumerate(zip(ranks, reference, strict=True)):
        assert torch.equal(local, expected), f"{family.name} rank {rank}"
    # The same call with its count lost on the way refuses on both ranks before
    # any collective, since this render was stamped.
    unstamped = (calls[0][0], calls[0][1], calls[0][2],
                 {k: v for k, v in calls[0][3].items() if k != KEY})
    plain = family.build(comfy)
    adapter.inject_cfg_pad_forward(plain)
    assert _refusals_on_two_ranks(monkeypatch, adapter, plain, unstamped) == [True, True]
    # The count is what makes the cut: with no stamped render noted, the
    # negative's rank attends its five pad rows.
    cfg_pad_rows.record_render(None)
    pad_attended = _split_on_two_ranks(monkeypatch, comfy, adapter, plain, unstamped)[1]
    assert not torch.equal(pad_attended, reference[1]), family.name


def _refusals_on_two_ranks(monkeypatch, adapter, model, call):
    """Each rank's call through the cfg split: whether it refused, class P, with
    no all-gather issued."""
    from dgx_monarch.adapters.base import UnsupportedModelError
    from dgx_monarch.refusal import RefusalClass, parse_refusal_tag

    x, t, context, kwargs = call
    model_options: dict = {}
    cfg_parallel_module.install_cfg_parallel_wrapper(adapter, model_options)
    refused = []
    for rank in (0, 1):
        monkeypatch.setattr(cfg_parallel_module, "cfg_world", lambda: 2)
        monkeypatch.setattr(cfg_parallel_module, "cfg_rank", lambda rank=rank: rank)
        sent: list = []
        monkeypatch.setattr(cfg_parallel_module, "_all_gather_cfg",
                            lambda local, sent=sent: sent.append(local) or torch.cat([local] * 2))
        options = {**model_options["transformer_options"],
                   "cond_or_uncond": [0, 1], "uuids": ["cond", "uncond"]}
        try:
            with torch.no_grad():
                model.forward(x, t, context, transformer_options=options, **kwargs)
            refused.append(False)
        except UnsupportedModelError as exc:
            assert parse_refusal_tag(str(exc)).refusal_class is RefusalClass.PHYSICS
            assert adapter.family in str(exc) and "no count" in str(exc)
            refused.append(not sent)
    return refused
