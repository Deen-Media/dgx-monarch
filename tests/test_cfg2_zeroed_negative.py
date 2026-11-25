"""cfg2 with a zeroed negative: each rank makes stock's own call for its conditioning.

A ConditioningZeroOut negative is all zero rows of real conditioning. The pair
legs of 2026-11-24 rendered two forms of it on cfg2 against one GPU:

* the zeroed negative of an empty prompt, shorter than the prompt. The Worker
  pads it so ComfyUI can batch the pair, and before the stamp the forward could
  not tell its zero rows from the pad's, so it ran all of them. Stock never
  batches the two (``CONDRegular`` needs equal shapes) and calls the negative
  at its own rows;
* the zeroed positive, as long as the prompt. Nothing is padded and ComfyUI
  batches the pair, so the cfg split hands each rank one row of that call.

Each case runs through ComfyUI's own cond path (``convert_cond``,
``encode_model_conds``, ``get_area_and_mult``, ``can_concat_cond``,
``cond_cat``) and the cfg split on two ranks, and each rank's output is held to
stock's forward on its conditioning alone, bit for bit. Where this CPU's GEMMs
round a batch-2 call as they round its two batch-1 calls (Qwen-Image 2.1's toy
here) the ranks are held to stock's batched call too. Krea2's toy rounds the two
batch sizes apart on this CPU for any conditioning, so there the batched call
is checked for the property the split relies on: a row of stock's batched call
does not depend on the other row.
"""
from __future__ import annotations

import pytest
import torch

from dgx_monarch.actor.sampling import equalize_cond_lengths
from dgx_monarch.adapters import cfg_pad_rows
from dgx_monarch.adapters import cfg_parallel as cfg_parallel_module
from dgx_monarch.adapters.krea2 import Krea2Adapter
from dgx_monarch.adapters.qwen_image21 import QwenImage21Adapter

KEY = cfg_pad_rows.TEXT_ROWS


@pytest.fixture(scope="module")
def comfy_modules():
    from real_comfy_helpers import real_comfy

    with real_comfy("cfg2-zeroed", "comfy.ldm.krea2.model", "comfy.ldm.qwen_image21.model",
                    "comfy.samplers", "comfy.sampler_helpers", "comfy.conds",
                    "comfy.patcher_extension") as modules:
        yield modules


def _model_calls(comfy, positive, negative, x_in, timestep):
    """The model calls ComfyUI makes for the pair: one batched call when the two
    concatenate, else one call each. Each is (x, t, context, extra keywords)."""
    _ops, _krea2, _q21, samplers, helpers, conds, _pe = comfy

    def extra_conds(**params):   # what both families' extra_conds publish for the text
        return {"c_crossattn": conds.CONDRegular(params["cross_attn"])}

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
        rows = len(group) * x_in.shape[0]
        calls.append((x_in.repeat(len(group), 1, 1, 1), timestep.repeat(rows), context, kwargs))
    return calls


def _split_on_two_ranks(monkeypatch, comfy, adapter, model, call):
    """The batched call through the cfg split, rank 0 then rank 1; each rank's slice."""
    pe = comfy[-1]
    x, t, context, kwargs = call
    model_options: dict = {}
    cfg_parallel_module.install_cfg_parallel_wrapper(adapter, model_options)
    assert pe.get_all_wrappers(pe.WrappersMP.DIFFUSION_MODEL, model_options["transformer_options"])
    outputs = []
    for rank in (0, 1):
        monkeypatch.setattr(cfg_parallel_module, "cfg_world", lambda: 2)
        monkeypatch.setattr(cfg_parallel_module, "cfg_rank", lambda rank=rank: rank)
        sent: list = []
        monkeypatch.setattr(cfg_parallel_module, "_all_gather_cfg",
                            lambda local, sent=sent: sent.append(local) or torch.cat([local] * 2))
        options = {**model_options["transformer_options"],
                   "cond_or_uncond": [0, 1], "uuids": ["cond", "uncond"]}
        with torch.no_grad():
            model.forward(x, t, context, transformer_options=options, **kwargs)
        outputs.append(sent[0])
    return outputs


def _krea2(comfy):
    from test_krea2_comfy_exactness import _toy

    return _toy(comfy[:2])


def _krea2_inputs(rows):
    from test_krea2_comfy_exactness import CHANNELS, TXT_DIM, TXT_LAYERS

    generator = torch.Generator().manual_seed(rows)
    prompt = torch.randn(1, 10, TXT_LAYERS * TXT_DIM, generator=generator)
    zeroed = torch.zeros(1, rows, TXT_LAYERS * TXT_DIM)
    x = torch.randn(1, CHANNELS, 4, 6, generator=generator)
    return prompt, zeroed, x, torch.tensor([0.4])


@pytest.mark.parametrize("negative_rows", [5, 10], ids=["zeroed-empty-prompt", "zeroed-positive"])
def test_krea2_cfg2_runs_a_zeroed_negative_as_stock_runs_it(comfy_modules, monkeypatch,
                                                             negative_rows):
    prompt, zeroed, x, t = _krea2_inputs(negative_rows)
    stock = _krea2(comfy_modules)
    with torch.no_grad():
        reference = [type(stock)._forward(stock, x, t, c, transformer_options={})
                     for c in (prompt, zeroed)]
    adapter = Krea2Adapter()
    positive, negative = equalize_cond_lengths(adapter, [[prompt, {}]], [[zeroed, {}]], x)
    # The shorter pair is padded and stamped; the equal pair is left alone.
    stamped = negative_rows < prompt.shape[1]
    assert (KEY in (negative[0][1].get("model_conds") or {})) is stamped
    calls = _model_calls(comfy_modules, positive, negative, x, t)
    assert len(calls) == 1 and calls[0][2].shape[0] == 2   # ComfyUI batches the pair
    assert (KEY in calls[0][3]) is stamped
    model = _krea2(comfy_modules)
    adapter.inject_cfg_pad_forward(model)
    ranks = _split_on_two_ranks(monkeypatch, comfy_modules, adapter, model, calls[0])
    for rank, (local, expected) in enumerate(zip(ranks, reference, strict=True)):
        assert torch.equal(local, expected), f"rank {rank}"


def test_a_stamped_render_whose_count_is_lost_refuses_on_both_ranks(comfy_modules, monkeypatch):
    """A real 6-row negative padded to 10, its count dropped between ComfyUI's
    cond_cat and the cfg split: every rank refuses, class P, before the
    all-gather, where the forward would otherwise attend four pad rows."""
    from dgx_monarch.adapters.base import UnsupportedModelError

    generator = torch.Generator().manual_seed(6)
    from test_krea2_comfy_exactness import CHANNELS, TXT_DIM, TXT_LAYERS

    prompt = torch.randn(1, 10, TXT_LAYERS * TXT_DIM, generator=generator)
    negative = torch.randn(1, 6, TXT_LAYERS * TXT_DIM, generator=generator)
    x = torch.randn(1, CHANNELS, 4, 6, generator=generator)
    t = torch.tensor([0.4])
    adapter = Krea2Adapter()
    positive, padded = equalize_cond_lengths(adapter, [[prompt, {}]], [[negative, {}]], x)
    calls = _model_calls(comfy_modules, positive, padded, x, t)
    assert len(calls) == 1 and calls[0][3].pop(KEY).tolist() == [10, 6]
    model = _krea2(comfy_modules)
    adapter.inject_cfg_pad_forward(model)
    pe = comfy_modules[-1]
    model_options: dict = {}
    cfg_parallel_module.install_cfg_parallel_wrapper(adapter, model_options)
    for rank in (0, 1):
        monkeypatch.setattr(cfg_parallel_module, "cfg_world", lambda: 2)
        monkeypatch.setattr(cfg_parallel_module, "cfg_rank", lambda rank=rank: rank)
        sent: list = []
        monkeypatch.setattr(cfg_parallel_module, "_all_gather_cfg",
                            lambda local, sent=sent: sent.append(local) or torch.cat([local] * 2))
        options = {**model_options["transformer_options"],
                   "cond_or_uncond": [0, 1], "uuids": ["cond", "uncond"]}
        assert pe.get_all_wrappers(pe.WrappersMP.DIFFUSION_MODEL, options)
        x_in, t_in, context, kwargs = calls[0]
        with pytest.raises(UnsupportedModelError, match="krea2: this render's cfg pair"):
            model.forward(x_in, t_in, context, transformer_options=options, **kwargs)
        assert sent == [], f"rank {rank} reached the all-gather"
    # An unstamped render's call (an equal pair, nothing padded) is untouched.
    cfg_pad_rows.record_render(None)
    with torch.no_grad():
        model.forward(x_in, t_in, context, transformer_options=options, **kwargs)


def test_krea2_batched_row_does_not_depend_on_the_other_row(comfy_modules):
    """The zeroed row of stock's batched call is the same whatever the other row
    holds: stock's forward derives nothing across the batch, so a rank holding
    one row alone is handed nothing the batched call shared."""
    prompt, zeroed, x, t = _krea2_inputs(10)
    model = _krea2(comfy_modules)
    partners = (prompt, prompt * -3.0, zeroed)
    with torch.no_grad():
        rows = [type(model)._forward(model, x.repeat(2, 1, 1, 1), t.repeat(2),
                                     torch.cat([partner, zeroed]), transformer_options={})[1:]
                for partner in partners]
    assert all(torch.equal(rows[0], row) for row in rows[1:])


def _qwen21(comfy):
    _ops, _krea2, q21 = comfy[:3]
    torch.manual_seed(31)
    model = q21.QwenImage21Transformer2DModel(
        in_channels=64, out_channels=64, num_layers=2, attention_head_dim=6,
        num_attention_heads=2, context_in_dim=4, mlp_ratio=1, axes_dims_rope=(2, 2, 2),
        operations=comfy[0].disable_weight_init)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.05)
    model.reset_prefix_cache(False)
    return model


@pytest.mark.parametrize("negative", ["zeroed-positive", "real"])
def test_qwen_image21_cfg2_slice_path_is_stocks_separate_calls(comfy_modules, monkeypatch,
                                                                negative):
    """Qwen-Image 2.1 pads nothing (cfg_cond_padding none): an equal pair takes
    the slice path, where each rank runs one entry alone. Its row is therefore
    stock's own call on that conditioning alone, bit for bit, and agrees with
    stock's batched call only to float rounding: with mkldnn off this CPU rounds
    the batch-2 call apart from the two batch-1 calls, and the ranks still equal
    the batch-1 calls."""
    generator = torch.Generator().manual_seed(7)
    prompt = torch.randn(1, 9, 4, generator=generator)
    other = torch.zeros_like(prompt) if negative == "zeroed-positive" else torch.randn(
        1, 9, 4, generator=generator)
    x = torch.randn(1, 64, 4, 6, generator=generator)
    t = torch.tensor([0.4])
    adapter = QwenImage21Adapter()
    positive, negative_cond = equalize_cond_lengths(adapter, [[prompt, {}]], [[other, {}]], x)
    calls = _model_calls(comfy_modules, positive, negative_cond, x, t)
    assert len(calls) == 1 and KEY not in calls[0][3]
    model = _qwen21(comfy_modules)
    with torch.no_grad():
        batched = model._forward(*calls[0][:3], transformer_options={})
        alone = [model._forward(x, t, c, transformer_options={}) for c in (prompt, other)]
    # Stock's batch-2 call and its two batch-1 calls are the same math but may
    # round apart, by CPU, BLAS and thread count; that is why an equal pair leaves
    # cfg2 where a bytes-equal route exists. Only float rounding separates them.
    assert all(torch.allclose(batched[i:i + 1], alone[i], rtol=1e-4, atol=1e-5)
               for i in range(2))
    ranks = _split_on_two_ranks(monkeypatch, comfy_modules, adapter, model, calls[0])
    for rank, local in enumerate(ranks):
        assert torch.equal(local, alone[rank]), f"rank {rank}"
