"""The cfg2 pad is trimmed by each conditioning's own row count, never guessed.

On a pure cfg topology the Worker pads the shorter conditioning of a pair with
zero rows so ComfyUI batches the two (actor/sampling.equalize_cond_lengths),
and each trimming family's cfg-pad forward cuts those rows off again so every
rank makes stock's own call. A zero row cannot say whether the pad added it: a
zeroed negative (ConditioningZeroOut) is nothing but zero rows, all of them
real conditioning, and a prompt may end in zero rows of its own. So the pad
stamps every entry of the pair with its count from before the pad, the model
receives that count as a forward keyword, and the forward trims to it.

The case that went wrong: a 10-row prompt and a 6-row zeroed negative. The
negative reached the model at 10 rows where stock calls it at 6; four more zero
rows became real keys after the text embedding's bias.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

from dgx_monarch.actor.sampling import equalize_cond_lengths
from dgx_monarch.adapters import cfg_pad_rows
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.cfg_parallel import _take_slice
from dgx_monarch.adapters.kandinsky5 import Kandinsky5Adapter
from dgx_monarch.adapters.krea2 import Krea2Adapter
from dgx_monarch.adapters.lens import LensAdapter
from dgx_monarch.adapters.mage_flow import MageFlowAdapter
from dgx_monarch.adapters.qwen_image import QwenImageAdapter
from dgx_monarch.adapters.zimage import ZImageAdapter
from dgx_monarch.refusal import RefusalClass, parse_refusal_tag

KEY = cfg_pad_rows.TEXT_ROWS


class _CondRegular:
    """ComfyUI's CONDRegular (comfy/conds.py) as far as these tests need it."""

    def __init__(self, cond):
        self.cond = cond

    def process_cond(self, batch_size, **kwargs):
        repeats = -(-batch_size // self.cond.shape[0])
        return _CondRegular(self.cond.repeat(repeats, *[1] * (self.cond.ndim - 1))[:batch_size])

    def can_concat(self, other):
        return self.cond.shape == other.cond.shape

    def concat(self, others):
        return torch.cat([self.cond, *(other.cond for other in others)])


def stub_comfy_conds(monkeypatch) -> None:
    """The one comfy module the stamp imports; comfy is not importable in this suite."""
    comfy = types.ModuleType("comfy")
    conds = types.ModuleType("comfy.conds")
    conds.CONDRegular = _CondRegular
    comfy.conds = conds
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.conds", conds)


@pytest.fixture
def comfy_conds(monkeypatch):
    stub_comfy_conds(monkeypatch)


def _rows_of(entry):
    stamp = (entry[1].get("model_conds") or {}).get(KEY)
    return None if stamp is None else stamp.cond.tolist()


TRIMMING = [Krea2Adapter, QwenImageAdapter, MageFlowAdapter, Kandinsky5Adapter, ZImageAdapter,
            LensAdapter]


@pytest.mark.parametrize("adapter_cls", TRIMMING)
def test_the_pad_stamps_every_entry_with_its_own_row_count(comfy_conds, adapter_cls):
    prompt = [[torch.randn(1, 10, 16), {"pooled_output": torch.ones(1, 4)}]]
    zeroed = [[torch.zeros(1, 6, 16), {}]]
    pos, neg = equalize_cond_lengths(adapter_cls(), prompt, zeroed, torch.zeros(1, 16, 8, 8))
    assert pos[0][0].shape[1] == neg[0][0].shape[1] == 10
    assert _rows_of(pos[0]) == [10] and _rows_of(neg[0]) == [6]
    # The rest of the conditioning is untouched, and the input lists are not mutated.
    assert torch.equal(pos[0][1]["pooled_output"], torch.ones(1, 4))
    assert "model_conds" not in prompt[0][1] and "model_conds" not in zeroed[0][1]


def test_an_equal_pair_is_neither_padded_nor_stamped(comfy_conds):
    pos, neg = equalize_cond_lengths(Krea2Adapter(), [[torch.randn(1, 6, 16), {}]],
                                     [[torch.zeros(1, 6, 16), {}]], torch.zeros(1, 16, 8, 8))
    assert _rows_of(pos[0]) is None and _rows_of(neg[0]) is None


def test_a_family_that_keeps_its_pad_gets_no_stamp(comfy_conds):
    class _Keeper:
        cfg_cond_padding = "pad"
        cfg_pad_restores_stock_call = False

    _pos, neg = equalize_cond_lengths(_Keeper(), [[torch.randn(1, 9, 4), {}]],
                                      [[torch.randn(1, 4, 4), {}]], torch.zeros(1, 4, 8, 8))
    assert neg[0][0].shape[1] == 9 and _rows_of(neg[0]) is None


def test_the_stamps_batch_and_split_with_the_pair(comfy_conds):
    """ComfyUI repeats each stamp to the latent batch and concatenates the pair
    like the text; the cfg split then hands each rank its own rows' counts."""
    pos, neg = equalize_cond_lengths(Krea2Adapter(), [[torch.randn(1, 10, 16), {}]],
                                     [[torch.zeros(1, 6, 16), {}]], torch.zeros(2, 16, 8, 8))
    first, second = (entry[1]["model_conds"][KEY].process_cond(batch_size=2) for entry in (pos[0], neg[0]))
    assert first.can_concat(second)
    batched = first.concat([second])
    assert batched.tolist() == [10, 10, 6, 6]
    assert _take_slice(batched, 4, 2, 0).tolist() == [10, 10]
    assert _take_slice(batched, 4, 2, 1).tolist() == [6, 6]


class _Recorder:
    """Stands in for the stock `_forward` a cfg-pad forward hands its call to."""

    def __init__(self):
        self.calls = []

    def _forward(self, x, timesteps, context, *args, attention_mask=None, **kwargs):
        num_tokens = args[0] if args else kwargs.get("num_tokens")
        self.calls.append({"context": context, "attention_mask": attention_mask,
                           "num_tokens": num_tokens, "kwargs": dict(kwargs)})
        return x


def _bound(adapter_cls):
    model = type(f"_Fake{adapter_cls.__name__}", (_Recorder,), {})()
    adapter_cls().inject_cfg_pad_forward(model)
    return model


DELEGATING = [QwenImageAdapter, MageFlowAdapter, ZImageAdapter, LensAdapter]


def _call(model, adapter_cls, context, rows):
    extra = {} if rows is None else {KEY: torch.tensor(rows)}
    x = torch.zeros(context.shape[0], 1, 2, 2)
    if adapter_cls is ZImageAdapter:
        return model._forward(x, torch.zeros(context.shape[0]), context, context.shape[1], **extra)
    return model._forward(x, torch.zeros(context.shape[0]), context, **extra)


@pytest.mark.parametrize("adapter_cls", DELEGATING)
def test_a_padded_zeroed_negative_reaches_stock_at_its_own_rows(adapter_cls):
    model = _bound(adapter_cls)
    zeroed = torch.zeros(1, 6, 16)
    padded = torch.cat([zeroed, torch.zeros(1, 4, 16)], dim=1)
    _call(model, adapter_cls, padded, [6])
    call = model.calls[-1]
    assert torch.equal(call["context"], zeroed)
    assert call["attention_mask"] is None
    assert KEY not in call["kwargs"]
    if adapter_cls is ZImageAdapter:
        assert call["num_tokens"] == 6


@pytest.mark.parametrize("adapter_cls", [QwenImageAdapter, MageFlowAdapter, LensAdapter,
                                         ZImageAdapter])
def test_a_shipped_text_mask_is_cut_with_the_text(comfy_conds, adapter_cls):
    """Lens's encoder ships a keep-mask, and so can a Qwen-Image encoder. The pad
    extends it with zeros so the pair batches; the forward cuts text and mask
    back to the entry's own rows, which is the call stock makes for it. Z-Image's
    Lumina forward ignores the mask, and its cut keeps the four alike."""
    prompt_mask = torch.ones(1, 10, dtype=torch.long)
    negative_mask = torch.tensor([[1, 1, 1, 0, 1, 1]])   # a masked token mid-prompt
    negative = torch.randn(1, 6, 16)
    _pos, neg = equalize_cond_lengths(
        adapter_cls(), [[torch.randn(1, 10, 16), {"attention_mask": prompt_mask}]],
        [[negative, {"attention_mask": negative_mask}]], torch.zeros(1, 16, 8, 8))
    assert neg[0][1]["attention_mask"].shape == (1, 10) and _rows_of(neg[0]) == [6]
    model = _bound(adapter_cls)
    num_tokens = (10,) if adapter_cls is ZImageAdapter else ()
    model._forward(torch.zeros(1, 1, 2, 2), torch.zeros(1), neg[0][0], *num_tokens,
                   attention_mask=neg[0][1]["attention_mask"],
                   **{KEY: neg[0][1]["model_conds"][KEY].cond})
    call = model.calls[-1]
    assert torch.equal(call["context"], negative)
    assert torch.equal(call["attention_mask"], negative_mask)


@pytest.mark.parametrize("adapter_cls", DELEGATING)
def test_an_unstamped_call_is_stock_even_when_it_ends_in_zero_rows(adapter_cls):
    """No stamp means no pad: a prompt whose last rows are zero keeps them."""
    model = _bound(adapter_cls)
    prompt = torch.ones(1, 8, 16)
    prompt[:, 6:] = 0.0
    _call(model, adapter_cls, prompt, None)
    assert torch.equal(model.calls[-1]["context"], prompt)


@pytest.mark.parametrize("adapter_cls", [QwenImageAdapter, MageFlowAdapter, LensAdapter])
def test_a_ragged_batch_keeps_its_residual_rows_masked(adapter_cls):
    model = _bound(adapter_cls)
    context = torch.randn(2, 10, 16)
    _call(model, adapter_cls, context, [7, 5])
    call = model.calls[-1]
    assert torch.equal(call["context"], context[:, :7])
    mask = call["attention_mask"]
    if adapter_cls is LensAdapter:
        assert mask.dtype == torch.bool
        assert mask.tolist() == [[True] * 7, [True] * 5 + [False] * 2]
    else:
        assert mask.shape == (2, 7)
        assert bool((mask[0] == 0).all()) and bool((mask[1, :5] == 0).all())
        assert bool((mask[1, 5:] == torch.finfo(context.dtype).min).all())


def test_zimage_refuses_a_ragged_batch_it_cannot_mask():
    model = _bound(ZImageAdapter)
    with pytest.raises(UnsupportedModelError, match="maskless"):
        _call(model, ZImageAdapter, torch.randn(2, 10, 16), [7, 5])


def test_kandinsky5_trims_by_the_stamp(monkeypatch):
    """Kandinsky5's cfg-pad forward rebuilds text RoPE for the trimmed rows."""
    seen = {}

    class _K5:
        patch_size = (1, 1, 1)

        def rope_encode_3d(self, *args, **kwargs):
            return None

        def rope_encode_1d(self, rows, **kwargs):
            seen["text_rope_rows"] = rows

        def forward_orig(self, x, timestep, context, y, freqs, freqs_text, **kwargs):
            seen["context"], seen["kwargs"] = context, kwargs
            return x

    common_dit = types.ModuleType("comfy.ldm.common_dit")
    common_dit.pad_to_patch_size = lambda x, patch: x
    for name, module in (("comfy", types.ModuleType("comfy")),
                         ("comfy.ldm", types.ModuleType("comfy.ldm")),
                         ("comfy.ldm.common_dit", common_dit)):
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["comfy"].ldm = sys.modules["comfy.ldm"]
    sys.modules["comfy.ldm"].common_dit = common_dit
    model = _K5()
    Kandinsky5Adapter().inject_cfg_pad_forward(model)
    zeroed = torch.zeros(1, 6, 16)
    padded = torch.cat([zeroed, torch.zeros(1, 4, 16)], dim=1)
    model._forward(torch.zeros(1, 1, 1, 2, 2), torch.zeros(1), padded, torch.zeros(1, 4),
                   **{KEY: torch.tensor([6])})
    assert torch.equal(seen["context"], zeroed) and seen["text_rope_rows"] == 6
    assert KEY not in seen["kwargs"]
    with pytest.raises(UnsupportedModelError, match="maskless"):
        model._forward(torch.zeros(2, 1, 1, 2, 2), torch.zeros(2), torch.randn(2, 10, 16),
                       torch.zeros(2, 4), **{KEY: torch.tensor([7, 5])})


def test_the_stamp_rides_comfys_own_cond_path_to_the_forward(comfy_krea2, monkeypatch):
    """Real ComfyUI, each step it takes between the pad and the model call:
    convert_cond keeps the stamped model_conds, encode_model_conds merges the
    family's extra_conds over it, get_area_and_mult and cond_cat repeat and
    batch it with the text, and BaseModel._apply_model hands it to the
    diffusion model as an int64 keyword."""
    import comfy.conds
    import comfy.model_base
    import comfy.sampler_helpers
    import comfy.samplers

    pos, neg = equalize_cond_lengths(Krea2Adapter(), [[torch.randn(1, 10, 16), {}]],
                                     [[torch.zeros(1, 6, 16), {}]], torch.zeros(2, 16, 8, 8))
    x_in = torch.zeros(2, 16, 8, 8)

    def extra_conds(**params):   # what a family's extra_conds publishes for the text
        return {"c_crossattn": comfy.conds.CONDRegular(params["cross_attn"])}

    batched = []
    for side in (pos, neg):
        conds = comfy.samplers.encode_model_conds(
            extra_conds, comfy.sampler_helpers.convert_cond(side), x_in, "cpu", "positive")
        assert set(conds[0]["model_conds"]) == {"c_crossattn", KEY}
        batched.append(comfy.samplers.get_area_and_mult(conds[0], x_in, torch.tensor([1.0])))
    assert comfy.samplers.can_concat_cond(batched[0], batched[1])
    model_kwargs = comfy.samplers.cond_cat([item.conditioning for item in batched])
    assert model_kwargs[KEY].tolist() == [10, 10, 6, 6]

    seen = {}

    class _Sampling:
        calculate_input = staticmethod(lambda sigma, x: x)
        timestep = staticmethod(lambda t: t)
        calculate_denoised = staticmethod(lambda sigma, out, x: out)

    class _Model:
        model_sampling = _Sampling()
        current_patcher = None

        def get_dtype_inference(self):
            return torch.float16

        def process_timestep(self, t, **kwargs):
            return t

        def diffusion_model(self, x, t, context=None, control=None, transformer_options=None,
                            **kwargs):
            seen.update(kwargs)
            return x

    comfy.model_base.BaseModel._apply_model(
        _Model(), torch.zeros(4, 16, 8, 8), torch.ones(4), **model_kwargs)
    assert seen[KEY].dtype == torch.long and seen[KEY].tolist() == [10, 10, 6, 6]


@pytest.fixture(scope="module")
def comfy_krea2():
    from real_comfy_helpers import real_comfy

    with real_comfy("cfg-pad-rows", "comfy.ldm.krea2.model") as modules:
        yield modules


@pytest.mark.parametrize("negative_rows", [6, 9])
def test_krea2_runs_a_padded_zeroed_negative_as_stock_runs_it(comfy_krea2, negative_rows):
    """Comfy's own Krea2 model at toy size: the cfg-pad forward on the padded,
    stamped negative returns stock's output for the negative's own rows,
    bit for bit."""
    from test_krea2_comfy_exactness import CHANNELS, TXT_DIM, TXT_LAYERS, _toy

    model = _toy(comfy_krea2)
    generator = torch.Generator().manual_seed(negative_rows)
    x = torch.randn(1, CHANNELS, 4, 6, generator=generator)
    timesteps = torch.tensor([0.4])
    zeroed = torch.zeros(1, negative_rows, TXT_LAYERS * TXT_DIM)
    padded = torch.cat([zeroed, torch.zeros(1, 10 - negative_rows, TXT_LAYERS * TXT_DIM)], dim=1)
    with torch.no_grad():
        stock = type(model)._forward(model, x, timesteps, zeroed, transformer_options={})
        Krea2Adapter().inject_cfg_pad_forward(model)
        trimmed = model._forward(x, timesteps, padded, transformer_options={},
                                 **{KEY: torch.tensor([negative_rows])})
        unstamped = model._forward(x, timesteps, padded, transformer_options={})
    assert torch.equal(trimmed, stock)
    # Without the count the call keeps all ten rows, which is not stock's call.
    assert not torch.equal(unstamped, stock)


@pytest.fixture(scope="module")
def comfy_lens():
    from real_comfy_helpers import real_comfy

    with real_comfy("cfg-pad-rows-lens", "comfy.ldm.lens.model") as modules:
        yield modules


@pytest.mark.parametrize("negative", ["zeroed", "masked"])
def test_lens_runs_a_padded_negative_as_stock_runs_it(comfy_lens, negative):
    """Comfy's own Lens model at toy size: the pad stretches the negative and
    its keep-mask to the prompt's rows and stamps its count; the cfg-pad forward
    returns stock's output for the negative alone, bit for bit."""
    from test_lens_comfy_exactness import _build

    ops, lens_model = comfy_lens
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(1, 8, 2, 4, generator=generator)
    timestep = torch.tensor([0.3])
    prompt = torch.randn(1, 10, 24, generator=generator)
    if negative == "zeroed":
        neg_text, neg_extras = torch.zeros(1, 6, 24), {}
    else:
        neg_text = torch.randn(1, 6, 24, generator=generator)
        neg_extras = {"attention_mask": torch.tensor([[1, 1, 1, 0, 1, 1]])}
    _pos, neg = equalize_cond_lengths(
        LensAdapter(), [[prompt, {"attention_mask": torch.ones(1, 10, dtype=torch.long)}]],
        [[neg_text, neg_extras]], x)
    assert neg[0][0].shape[1] == 10 and _rows_of(neg[0]) == [6]
    model = _build(ops, lens_model)
    with torch.no_grad():
        stock = model._forward(x, timestep, neg_text,
                               attention_mask=neg_extras.get("attention_mask"))
        LensAdapter().inject_cfg_pad_forward(model)
        cut = model._forward(x, timestep, neg[0][0],
                             attention_mask=neg[0][1].get("attention_mask"),
                             **{KEY: neg[0][1]["model_conds"][KEY].cond})
    assert torch.equal(cut, stock)


def test_every_stamped_family_binds_a_forward_that_reads_the_stamp():
    """The stamping set is explicit: every registered family the pad stamps
    (cfg_pad_rows.stamped, a pad it cuts back off, never per-cond dispatch)
    binds a cfg-pad forward carrying the reads_rows mark, and a family the pad
    does not stamp is never handed one."""
    from dgx_monarch.adapters import ADAPTERS
    from dgx_monarch.adapters.cfg_parallel import cfg_dispatches_per_cond

    stamping = set()
    for adapter in ADAPTERS:
        if not cfg_pad_rows.stamped(adapter) or cfg_dispatches_per_cond(adapter):
            continue
        stamping.add(adapter.family)
        model = type(f"_Fake{adapter.family}", (_Recorder,), {})()
        adapter.inject_cfg_pad_forward(model)
        cfg_pad_rows.assert_read(model, adapter.family)
    assert stamping == {"krea2", "qwen_image", "mage_flow", "zimage", "kandinsky5", "lens"}


def test_a_stamp_meeting_a_forward_that_ignores_it_refuses(monkeypatch):
    """The cfg split checks every stamped call: a stamp bound for a forward that
    does not read it refuses typed, naming the family, before any slice or
    collective; a reading forward passes."""
    from dgx_monarch.adapters import cfg_parallel

    monkeypatch.setattr(cfg_parallel, "cfg_world", lambda: 1)
    wrapper = cfg_parallel.make_cfg_parallel_wrapper(Krea2Adapter())

    class _Executor:
        def __init__(self, model):
            self.class_obj, self.original = model, model._forward

        def __call__(self, *args, **kwargs):
            return "ran"

    stock = _Recorder()
    with pytest.raises(UnsupportedModelError) as caught:
        wrapper(_Executor(stock), torch.zeros(2, 1), **{KEY: torch.tensor([6, 6])})
    tag = parse_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert "krea2" in str(caught.value)
    assert wrapper(_Executor(_bound(Krea2Adapter)), torch.zeros(2, 1),
                   **{KEY: torch.tensor([6, 6])}) == "ran"
    assert wrapper(_Executor(stock), torch.zeros(2, 1)) == "ran"   # no stamp, no check


def test_a_stamped_render_refuses_a_call_with_text_and_no_count(monkeypatch, comfy_conds):
    """The pad notes the render it stamped; then a call with text and no count
    lost it on the way and refuses, class P, naming the family. A call with no
    text tensor passes, and so does every call of an unstamped render."""
    from dgx_monarch.adapters import cfg_parallel

    monkeypatch.setattr(cfg_parallel, "cfg_world", lambda: 1)
    wrapper = cfg_parallel.make_cfg_parallel_wrapper(Krea2Adapter())

    class _Executor:
        def __init__(self, model):
            self.class_obj, self.original = model, model._forward

        def __call__(self, *args, **kwargs):
            return "ran"

    model = _bound(Krea2Adapter)
    text = torch.randn(2, 10, 16)
    assert wrapper(_Executor(model), torch.zeros(2, 1), torch.zeros(2), text) == "ran"
    equalize_cond_lengths(Krea2Adapter(), [[torch.randn(1, 10, 16), {}]],
                          [[torch.randn(1, 6, 16), {}]], torch.zeros(1, 1, 2, 2))
    with pytest.raises(UnsupportedModelError) as caught:
        wrapper(_Executor(model), torch.zeros(2, 1), torch.zeros(2), text)
    tag = parse_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert "krea2" in str(caught.value) and "no count" in str(caught.value)
    assert wrapper(_Executor(model), torch.zeros(2, 1), torch.zeros(2), None) == "ran"
    assert wrapper(_Executor(model), torch.zeros(2, 1), torch.zeros(2), text,
                   **{KEY: torch.tensor([10, 6])}) == "ran"
    # An equal pair is never padded, so it notes nothing; a render clears the note.
    cfg_pad_rows.record_render(None)
    equalize_cond_lengths(Krea2Adapter(), [[torch.randn(1, 6, 16), {}]],
                          [[torch.randn(1, 6, 16), {}]], torch.zeros(1, 1, 2, 2))
    assert wrapper(_Executor(model), torch.zeros(2, 1), torch.zeros(2), text) == "ran"


@pytest.mark.parametrize("counts", [[6, 6], [6, 4]], ids=["uniform", "ragged"])
def test_the_cut_is_a_dense_tensor_as_stocks_own_text_is(counts):
    """A strided (B, rows, D) view rounds apart from stock's dense text in a
    GEMM over several rows (Kandinsky5 at batch 2), so the cut is a copy."""
    context = torch.randn(2, 10, 16)
    cut, _ragged = cfg_pad_rows.trim(context, torch.tensor(counts), "kandinsky5")
    assert cut.shape == (2, 6, 16) and cut.is_contiguous()
    assert torch.equal(cut, context[:, :6])


def test_every_render_starts_with_no_stamp_noted():
    """The note is per render: run_sample clears it beside the fold fact, before
    the pad can set it, so a later render that pads nothing (a basic guider, an
    equal pair) never meets a note an earlier render left."""
    import inspect

    from dgx_monarch.actor import sample_protocol

    body = inspect.getsource(sample_protocol)
    cleared = body.index("cfg_pad_rows.record_render(None)")
    assert body.index("cfg_dispatch.record_cfg_pair_lengths([])") < cleared
    assert cleared < body.index("equalize_cond_lengths(")


def test_the_pad_leaves_the_requests_conditioning_as_it_was(comfy_conds):
    """The stamp and the pad live in copies: the lists and dicts the request
    carried, which a cached graph output can share, come back untouched."""
    prompt = [[torch.randn(1, 10, 16), {"model_conds": {"other": object()}}]]
    zeroed = [[torch.zeros(1, 6, 16), {}]]
    before = (prompt[0][0].clone(), dict(prompt[0][1]), dict(prompt[0][1]["model_conds"]))
    equalize_cond_lengths(Krea2Adapter(), prompt, zeroed, torch.zeros(1, 16, 8, 8))
    assert torch.equal(prompt[0][0], before[0]) and zeroed[0][0].shape[1] == 6
    assert prompt[0][1] == before[1] and prompt[0][1]["model_conds"] == before[2]
    assert zeroed[0][1] == {}


def test_a_stamp_that_does_not_match_the_batch_refuses():
    """A count per batch row, or a typed dgx-monarch fault: never a guess."""
    model = _bound(QwenImageAdapter)
    with pytest.raises(UnsupportedModelError) as caught:
        _call(model, QwenImageAdapter, torch.randn(2, 10, 16), [7])
    tag = parse_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    with pytest.raises(UnsupportedModelError):
        _call(model, QwenImageAdapter, torch.randn(1, 10, 16), [11])
