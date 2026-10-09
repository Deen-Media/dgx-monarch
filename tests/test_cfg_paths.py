"""CPU coverage for cfg-parallel: driver cond equalization and the fold fact it
records, krea2's trailing-pad trim, the wrapper's batch slicing on both split
seams, the per-cond dispatch, and the nvfp4 shared scale on the two cfg paths."""
import pytest
import torch

from dgx_monarch.actor.sampling import equalize_cond_lengths
from dgx_monarch.adapters import cfg_dispatch
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.cfg_parallel import (
    _leading_batch,
    _take_slice,
    make_cfg_parallel_wrapper,
)
from dgx_monarch.adapters.krea2 import _trim_trailing_pad
from dgx_monarch.adapters.pixeldit import Ideogram4Adapter


@pytest.fixture(autouse=True)
def _cfg_pair_folds_starts_true():
    """equalize_cond_lengths records a fold fact as a side effect, so every
    test here starts and ends at the default, whatever an earlier one recorded."""
    cfg_dispatch.record_cfg_pair_lengths([])
    yield
    cfg_dispatch.record_cfg_pair_lengths([])


class _FakeAdapter:
    def __init__(self, rule, restores_stock_call=False):
        self.cfg_cond_padding = rule
        self.cfg_pad_restores_stock_call = restores_stock_call


def test_ideogram4_typed_refuses_cfg_parallel_before_sampling():
    with pytest.raises(UnsupportedModelError, match="separate calls") as caught:
        make_cfg_parallel_wrapper(Ideogram4Adapter())
    message = str(caught.value)
    assert "topology 'auto'" in message
    assert "explicit ring topology" in message
    assert "mode=local" in message and "gpus_per_host=1" in message
    assert "Use ring2" not in message and "topology 'single'" not in message


def _batch_one_refusal(monkeypatch, family: str) -> str:
    """Drive one family's cfg-parallel wrapper with a single-row model call."""
    import dgx_monarch.adapters.cfg_parallel as cfg_parallel
    from dgx_monarch.adapters import ADAPTERS

    monkeypatch.setattr(cfg_parallel, "cfg_world", lambda: 2)
    monkeypatch.setattr(cfg_parallel, "cfg_rank", lambda: 0)
    adapter = next(item for item in ADAPTERS if item.family == family)

    class Executor:
        def __init__(self):
            self.original = self.forward

        def forward(self, x, transformer_options=None):
            raise AssertionError("signature only")

        def __call__(self, x, transformer_options=None):
            raise AssertionError("the wrapper must refuse before it calls comfy")

    wrapper = make_cfg_parallel_wrapper(adapter)
    with pytest.raises(UnsupportedModelError) as caught:
        wrapper(Executor(), torch.zeros(1, 4), {})
    return str(caught.value)


def test_a_single_conditioning_refuses_the_same_way_for_krea2_and_flux(monkeypatch):
    """The batch-1 card is a fact about the call, not about the family.

    Measured 2026-09-02: krea2 at cfg2 with CFG 1.0 rendered and flux1 at cfg2
    with CFG 1.0 refused, because krea2's template asks for a cfg++ sampler
    that keeps the unconditional pass. Hand krea2 a batch of one and it must
    refuse exactly as flux1 does, and the card must name the cfg++ sampler as
    the other way to earn a batch of two.
    """
    krea2 = _batch_one_refusal(monkeypatch, "krea2")
    flux = _batch_one_refusal(monkeypatch, "flux")
    assert krea2 == flux
    assert "batch=1, which does not split into 2 equal slices" in krea2
    assert "cfg_pp" in krea2
    assert "[dgxm:P" in krea2


def _cond(tokens, dim=8, batch=1, fill=1.0):
    return [[torch.full((batch, tokens, dim), fill), {}]]


def test_equalize_noop_when_lengths_match():
    pos, neg = _cond(7), _cond(7)
    out_pos, out_neg = equalize_cond_lengths(_FakeAdapter("pad"), pos, neg, torch.zeros(1, 4, 8, 8))
    assert out_pos is pos and out_neg is neg


def test_equalize_pads_shorter_side_with_zero_rows():
    pos, neg = _cond(9), _cond(4)
    out_pos, out_neg = equalize_cond_lengths(_FakeAdapter("pad"), pos, neg, torch.zeros(1, 4, 8, 8))
    assert out_pos[0][0].shape[1] == 9
    assert out_neg[0][0].shape[1] == 9
    assert float(out_neg[0][0][:, 4:].abs().sum()) == 0.0
    assert float(out_neg[0][0][:, :4].abs().sum()) > 0.0


def _lens_cond(tokens, layers=4, hidden=6, batch=1, fill=1.0):
    """One Lens conditioning entry, shaped as the shipped stack builds it.

    Derived from ComfyUI 3216c62e. The GPT-OSS encoder stacks its selected
    layers and flattens them, returning features of shape (B, S, L*H) beside a
    long keep-mask of shape (B, S) whose entries are 1 for a real token and 0
    for padding: see ``LensGptOssClipModel.encode_token_weights`` and its
    ``_gather_tokens`` helper in comfy/text_encoders/gpt_oss.py (the mask is
    built with ``mask[i, :len(x)] = 1``, then both are trimmed by the 97-token
    template offset). ``num_layers_stacked`` sits in the same extras dict. On
    the real checkpoint L is 4 and H is 2880, so the feature width is 11520;
    the width is irrelevant to concatability, so this uses small numbers.

    ``S`` is whatever the prompt tokenized to: the Lens tokenizer does not pad
    to a fixed length, so a long positive and a short negative reach the
    sampler with different token counts.
    """
    return [[
        torch.full((batch, tokens, layers * hidden), fill),
        {
            "pooled_output": None,
            "attention_mask": torch.ones(batch, tokens, dtype=torch.long),
            "num_layers_stacked": layers,
        },
    ]]


def _comfy_conds_would_batch(entry_a, entry_b, keys=("c_crossattn", "attention_mask")):
    """Mirror ComfyUI's cond-batching rule over the keys Lens publishes.

    ComfyUI batches cond with uncond only when ``cond_equal_size`` holds
    (comfy/samplers.py: identical key sets, and ``can_concat`` true for every
    key), and Lens publishes both of its keys as a plain ``CONDRegular``
    (comfy/model_base.py, class ``Lens.extra_conds``), whose ``can_concat``
    demands exact shape equality (comfy/conds.py). Lens gets no help from
    ``CONDCrossAttn``'s lowest-common-multiple padding, which is the fallback
    that saves families publishing their cross-attention under that class.

    comfy is not importable in this suite, so this mirrors the rule rather than
    calling it. The shapes below are what ``model_conds`` would hold.
    """
    def conds_of(entry):
        extras = entry[1]
        out = {"c_crossattn": tuple(entry[0].shape)}
        mask = extras.get("attention_mask")
        if mask is not None:
            out["attention_mask"] = tuple(mask.shape)
        return out

    a, b = conds_of(entry_a), conds_of(entry_b)
    return a.keys() == b.keys() and all(a[k] == b[k] for k in keys if k in a) and a == b


def test_equalize_pad_extends_a_shipped_keep_mask_so_lens_batches():
    """Padding the text rows alone does not make the Lens pair batch.

    The shipped dgx-monarch-lens-t2i template has a long positive and a short
    "blurry, low quality" negative, and Lens conditioning carries a keep-mask
    beside the features. Equalizing only the cross-attention tensor leaves the
    two masks at their own token counts, ``cond_equal_size`` rejects the pair
    on that second key, and cfg-parallel gets the batch of one it refuses.
    """
    pos, neg = _lens_cond(9), _lens_cond(4)
    out_pos, out_neg = equalize_cond_lengths(
        _FakeAdapter("pad"), pos, neg, torch.zeros(1, 4, 8, 8))

    # Both keys agree in shape, which CONDRegular.can_concat requires.
    assert out_pos[0][0].shape == (1, 9, 24)
    assert out_neg[0][0].shape == (1, 9, 24)
    assert tuple(out_pos[0][1]["attention_mask"].shape) == (1, 9)
    assert tuple(out_neg[0][1]["attention_mask"].shape) == (1, 9)
    assert _comfy_conds_would_batch(out_pos[0], out_neg[0])

    # The extension is a keep-mask: the four real tokens keep 1, the five pad
    # rows get 0, so the stock forward drops exactly the rows the driver added.
    neg_mask = out_neg[0][1]["attention_mask"]
    assert neg_mask.dtype == torch.long
    assert neg_mask[0, :4].tolist() == [1, 1, 1, 1]
    assert neg_mask[0, 4:].tolist() == [0, 0, 0, 0, 0]
    # The longer side is already at full width and is not rewritten.
    assert out_pos[0][1]["attention_mask"].tolist() == [[1] * 9]
    # Padded rows are zero rows, and the real rows are untouched.
    assert float(out_neg[0][0][:, 4:].abs().sum()) == 0.0
    assert float(out_neg[0][0][:, :4].abs().sum()) > 0.0
    # Every other extra survives, including the layer count the features encode.
    assert out_neg[0][1]["num_layers_stacked"] == 4
    assert out_pos[0][1]["num_layers_stacked"] == 4
    assert out_neg[0][1]["pooled_output"] is None


def test_equalize_pad_without_a_shipped_mask_is_unchanged():
    """Families that ship no mask keep the plain zero-row behaviour.

    krea2 declares the "pad" rule and ships no mask: its text encoder drops an
    all-ones mask, and its cfg-pad forward reads the pad off the zero rows.
    flux2's encoder does ship one, which this rule extends and Flux.extra_conds
    ignores; flux2 needs no key mask, since ComfyUI front-pads its text to 512
    tokens. Nothing may be invented where no mask was shipped.
    """
    pos, neg = _cond(9), _cond(4)
    out_pos, out_neg = equalize_cond_lengths(
        _FakeAdapter("pad"), pos, neg, torch.zeros(1, 4, 8, 8))
    assert out_neg[0][0].shape[1] == 9
    assert "attention_mask" not in out_neg[0][1]
    assert "attention_mask" not in out_pos[0][1]


def test_equalize_pad_leaves_a_mask_of_unrecognized_width_alone():
    """A mask that does not span its own text rows is not guessed at.

    The extension is only correct for a (B, tokens) keep-mask covering exactly
    the rows being padded. Anything else keeps its value, and the operator gets
    a warning instead of a silently wrong mask.
    """
    pos = _lens_cond(9)
    neg = _lens_cond(4)
    neg[0][1]["attention_mask"] = torch.ones(1, 1, 4)  # joint/bias shape, not (B, tokens)
    out_pos, out_neg = equalize_cond_lengths(
        _FakeAdapter("pad"), pos, neg, torch.zeros(1, 4, 8, 8))
    assert out_neg[0][0].shape[1] == 9                       # rows still padded
    assert tuple(out_neg[0][1]["attention_mask"].shape) == (1, 1, 4)
    assert not _comfy_conds_would_batch(out_pos[0], out_neg[0])


def test_equalize_pad_mask_covers_joint_keys():
    latent = torch.zeros(1, 4, 15, 16)  # odd latent height: ceil grid
    pos, neg = _cond(9), _cond(4)
    out_pos, out_neg = equalize_cond_lengths(_FakeAdapter("pad+mask"), pos, neg, latent)
    grid = ((15 + 1) // 2) * ((16 + 1) // 2)  # stock chroma token grid
    # Text pads to 16, not 9: equalize_cond_lengths (actor/sampling.py) rounds
    # the joint key extent, 9 + 64 = 73, up to 80, a multiple of 8, because an
    # odd extent drives cudnn's f16 flash-fprop SDPA into a misaligned-address
    # fault (2026-08-21).
    bias = out_neg[0][0 + 1]["attention_mask"]
    assert list(bias.shape) == [1, 1, 16 + grid]
    # Appended text rows masked, real text + image keys visible.
    assert float(bias[0, 0, :4].abs().sum()) == 0.0
    assert bool((bias[0, 0, 4:16] < -1e30).all())
    assert float(bias[0, 0, 16:].abs().sum()) == 0.0
    assert out_neg[0][1]["attention_mask_img_shape"] == (8, 8)
    # The longer side also pads to the aligned length and masks exactly its
    # own appended rows.
    pos_bias = out_pos[0][1]["attention_mask"]
    assert out_pos[0][0].shape[1] == 16
    assert float(pos_bias[0, 0, :9].abs().sum()) == 0.0
    assert bool((pos_bias[0, 0, 9:16] < -1e30).all())
    assert float(pos_bias[0, 0, 16:].abs().sum()) == 0.0


def test_equalize_pad_mask_aligns_joint_extent_on_odd_grids():
    # 63x63 grid (odd image side, e.g. 1008px): text-only alignment would leave
    # the joint key extent odd; the sum must land on a multiple of 8.
    latent = torch.zeros(1, 4, 126, 126)
    pos, neg = _cond(9), _cond(4)
    out_pos, _ = equalize_cond_lengths(_FakeAdapter("pad+mask"), pos, neg, latent)
    grid = 63 * 63
    padded = out_pos[0][0].shape[1]
    assert padded >= 9
    assert (padded + grid) % 8 == 0
    assert list(out_pos[0][1]["attention_mask"].shape) == [1, 1, padded + grid]


def test_equalize_hunyuan_mask_is_text_only_and_additive():
    pos, neg = _cond(9, batch=2), _cond(4, batch=2)
    out_pos, out_neg = equalize_cond_lengths(
        _FakeAdapter("pad+text-mask"), pos, neg, torch.zeros(2, 4, 8, 8)
    )
    pos_mask = out_pos[0][1]["attention_mask"]
    neg_mask = out_neg[0][1]["attention_mask"]
    assert list(pos_mask.shape) == [2, 16]  # aligned to a multiple of 8
    assert list(neg_mask.shape) == [2, 16]
    assert float(pos_mask[:, :9].abs().sum()) == 0.0
    assert bool((pos_mask[:, 9:] < -1e30).all())
    assert float(neg_mask[:, :4].abs().sum()) == 0.0
    assert bool((neg_mask[:, 4:] < -1e30).all())
    assert "attention_mask_img_shape" not in out_neg[0][1]


@pytest.mark.parametrize("extra", ["conditioning_byt5small", "clip_vision_output"])
def test_equalize_hunyuan_rejects_extended_text_with_asymmetric_qwen(extra):
    pos, neg = _cond(9), _cond(4)
    neg[0][1][extra] = object()
    with pytest.raises(UnsupportedModelError, match="wrong width"):
        equalize_cond_lengths(
            _FakeAdapter("pad+text-mask"), pos, neg, torch.zeros(1, 4, 8, 8)
        )


def test_equalize_respects_existing_mask():
    pos = [[torch.ones(1, 9, 8), {"attention_mask": torch.zeros(1, 1, 9)}]]
    neg = _cond(4)
    out_pos, out_neg = equalize_cond_lengths(_FakeAdapter("pad+mask"), pos, neg, torch.zeros(1, 4, 8, 8))
    assert out_pos is pos and out_neg is neg


def test_equalize_none_rule_is_identity():
    pos, neg = _cond(9), _cond(4)
    assert equalize_cond_lengths(_FakeAdapter("none"), pos, neg, None) == (pos, neg)


def test_equalize_pad_mask_leaves_image_only_none_entry_untouched():
    # [[None, {}]] is the image-only uncond that `_cond_tensor`'s docstring
    # names: dual-model guidance's separate uncond model carries no
    # cross-attention tensor, so it passes through unchanged, never padded or
    # given an attention_mask, while a sibling entry with a real, shorter
    # tensor still equalizes against the longer positive.
    pos = _cond(9)
    neg = [*_cond(4), [None, {}]]
    out_pos, out_neg = equalize_cond_lengths(
        _FakeAdapter("pad+mask"), pos, neg, torch.zeros(1, 4, 8, 8))
    assert out_pos[0][0].shape[1] == 16             # aligned length
    assert out_neg[0][0].shape[1] == 16             # real entry padded 4 -> 16
    assert "attention_mask" in out_neg[0][1]
    assert out_neg[1] == [None, {}]                  # None entry: untouched, no mask key


def test_cross_attn_pair_folds_equal_lengths():
    assert cfg_dispatch.cross_attn_pair_folds(44, 44) is True


def test_cross_attn_pair_folds_within_the_lcm_cap():
    # lcm(2, 8) = 8, repeat 8 // 2 = 4: at comfy's own cap, still folds.
    assert cfg_dispatch.cross_attn_pair_folds(2, 8) is True


def test_cross_attn_pair_folds_past_the_lcm_cap():
    # lcm(2, 11) = 22, repeat 22 // 2 = 11 > 4: does not fold.
    assert cfg_dispatch.cross_attn_pair_folds(2, 11) is False


def test_cross_attn_pair_folds_the_measured_uneven_pair_does_not_fold():
    # The shipped chroma pair (docs/VALIDATION.md, 2026-10-01): lcm 700, repeat 25 > 4.
    assert cfg_dispatch.cross_attn_pair_folds(100, 28) is False


def test_cross_attn_pair_folds_zero_length_never_folds():
    assert cfg_dispatch.cross_attn_pair_folds(0, 9) is False


def test_record_cfg_pair_lengths_empty_or_single_folds():
    cfg_dispatch.record_cfg_pair_lengths([])
    assert cfg_dispatch.cfg_pair_folds() is True
    cfg_dispatch.record_cfg_pair_lengths([7])
    assert cfg_dispatch.cfg_pair_folds() is True


def test_record_cfg_pair_lengths_checks_every_length_against_the_first():
    cfg_dispatch.record_cfg_pair_lengths([9, 9, 4])
    assert cfg_dispatch.cfg_pair_folds() is False
    cfg_dispatch.record_cfg_pair_lengths([9, 9, 9])
    assert cfg_dispatch.cfg_pair_folds() is True


def test_equalize_records_the_fold_fact_from_the_original_lengths():
    """The decision reads the lengths before any pad, not the padded shapes
    the family rule produces: a pad that makes 100/28 read as 100/100 must
    not make this read True."""
    trimmer = _FakeAdapter("pad+mask", restores_stock_call=True)
    equalize_cond_lengths(trimmer, _cond(100), _cond(28), torch.zeros(1, 4, 8, 8))
    assert cfg_dispatch.cfg_pair_folds() is False
    equalize_cond_lengths(trimmer, _cond(44), _cond(44), torch.zeros(1, 4, 8, 8))
    assert cfg_dispatch.cfg_pair_folds() is True


@pytest.mark.parametrize("rule", ["none", "pad", "pad+mask"])
def test_a_family_that_keeps_the_pad_keeps_the_shared_scale(rule):
    """Only a family whose cfg-pad forward trims back to stock's own call
    reads the original pair's fold. One that keeps the pad (Flux, LongCat,
    Lens) runs every cfg rank on the padded batch, which no single-GPU call
    makes either, so its fold fact stays true and it keeps the cfg reducer."""
    latent = None if rule == "none" else torch.zeros(1, 4, 8, 8)
    equalize_cond_lengths(_FakeAdapter(rule), _cond(100), _cond(28), latent)
    assert cfg_dispatch.cfg_pair_folds() is True


def test_the_measured_trimming_families_declare_it():
    from dgx_monarch.adapters.boogu import BooguAdapter
    from dgx_monarch.adapters.flux_family import ChromaAdapter, Flux2Adapter, FluxAdapter, LongCatAdapter
    from dgx_monarch.adapters.kandinsky5 import Kandinsky5Adapter
    from dgx_monarch.adapters.krea2 import Krea2Adapter
    from dgx_monarch.adapters.lens import LensAdapter
    from dgx_monarch.adapters.qwen_image import QwenImageAdapter
    from dgx_monarch.adapters.zimage import ZImageAdapter

    for cls in (ChromaAdapter, Krea2Adapter, ZImageAdapter, QwenImageAdapter,
                BooguAdapter, Kandinsky5Adapter):
        assert cls.cfg_pad_restores_stock_call is True, cls.__name__
        assert hasattr(cls, "inject_cfg_pad_forward"), cls.__name__
    for cls in (FluxAdapter, Flux2Adapter, LongCatAdapter, LensAdapter):
        assert cls.cfg_pad_restores_stock_call is False, cls.__name__


def test_both_ranks_derive_the_fold_fact_from_the_shipped_conds_alone(monkeypatch):
    """Rank replicated: the fact comes only from lengths every rank's request
    carries, never from this rank's own identity."""
    import dgx_monarch.adapters.base as base

    def _boom(*_a, **_k):
        raise AssertionError("the fold fact must not read a rank accessor")

    monkeypatch.setattr(base, "cfg_rank", _boom, raising=False)
    monkeypatch.setattr(base, "sp_rank", _boom, raising=False)
    monkeypatch.setattr(base, "cfg_world", _boom, raising=False)

    pos, neg = _cond(100), _cond(28)
    seen = set()
    for _rank in (0, 1):
        equalize_cond_lengths(_FakeAdapter("pad+mask", restores_stock_call=True),
                              pos, neg, torch.zeros(1, 4, 8, 8))
        seen.add(cfg_dispatch.cfg_pair_folds())
    assert seen == {False}


# The forward trims the driver's trailing zero-pad instead of masking it, so the
# common (uniform / single-entry) case runs mask-free on the flash fast path;
# only a ragged batch keeps an additive bias over the residual per-entry pad.
def test_trim_none_without_padding():
    ctx = torch.ones(2, 5, 3, 4)
    out, mask = _trim_trailing_pad(ctx)
    assert mask is None
    assert out.shape[1] == 5
    assert torch.equal(out, ctx)


def test_trim_uniform_trailing_run_is_maskless():
    ctx = torch.ones(1, 6, 3, 4)
    ctx[:, 4:] = 0.0  # driver-appended rows
    out, mask = _trim_trailing_pad(ctx)
    assert mask is None                     # trimmed away: flash path, no mask
    assert out.shape[1] == 4                # trimmed to the real length
    assert torch.equal(out, ctx[:, :4])


def test_trim_conditioning_zero_out_untouched():
    # An all-zero conditioning is a real uncond, not padding: never trimmed.
    ctx = torch.zeros(1, 6, 3, 4)
    out, mask = _trim_trailing_pad(ctx)
    assert mask is None
    assert out.shape[1] == 6


def test_trim_ragged_batch_masks_residual_after_trim():
    # Different real lengths: trim to the longest (5), mask the shorter entry's
    # residual pad within the trimmed span.
    ctx = torch.ones(2, 8, 3, 4)
    ctx[0, 5:] = 0.0  # entry 0 real length 5 (the longest)
    ctx[1, 3:] = 0.0  # entry 1 real length 3
    out, mask = _trim_trailing_pad(ctx)
    assert out.shape[1] == 5                           # trimmed to the longest real
    assert mask is not None                            # ragged: residual masked
    assert list(mask.shape) == [2, 1, 5]
    assert float(mask[0, 0, :5].abs().sum()) == 0.0    # entry 0: full within trim
    assert bool((mask[1, 0, 3:5] < 0).all())           # entry 1: 3..4 masked
    assert float(mask[1, 0, :3].abs().sum()) == 0.0


def test_trim_interior_zero_rows_are_not_padding():
    ctx = torch.ones(1, 6, 3, 4)
    ctx[:, 2] = 0.0  # interior zero row: content follows it
    out, mask = _trim_trailing_pad(ctx)
    assert mask is None
    assert out.shape[1] == 6                # not trimmed (content after the zero)


def test_leading_batch_finds_tensors_in_containers():
    assert _leading_batch(torch.zeros(4, 3)) == 4
    assert _leading_batch({"c_crossattn": [torch.zeros(6, 2)]}) == 6
    assert _leading_batch(None) is None
    assert _leading_batch(torch.tensor(1.0)) is None


def test_take_slice_handles_batch_multiples():
    # cfg2 with latent batch 3: call batch 6, each rank takes 3 rows.
    x = torch.arange(6).view(6, 1).float()
    r0 = _take_slice(x, batch=6, world=2, rank=0)
    r1 = _take_slice(x, batch=6, world=2, rank=1)
    assert torch.equal(torch.cat((r0, r1)), x)
    assert r0.shape[0] == 3


def test_take_slice_leaves_broadcast_and_bookkeeping():
    broadcast = torch.zeros(1, 5)
    assert _take_slice(broadcast, batch=4, world=2, rank=1) is broadcast
    control = {"output": [torch.arange(4).view(4, 1), None], "note": "keep"}
    sliced = _take_slice(control, batch=4, world=2, rank=1)
    assert torch.equal(sliced["output"][0], torch.tensor([[2], [3]]))
    assert sliced["output"][1] is None
    assert sliced["note"] == "keep"


def _patcher_extension():
    """A stand-in for the two comfy pieces the split runs through.

    The battery carries no ComfyUI, so the wrapper store and the executor are
    rebuilt here to comfy's contract. That contract is pinned against the real
    thing by the "keyed wrapper" and "cfg split" seams in
    tests/canary/comfy_seam_contracts.py, which run where comfy is importable.
    """
    import types

    module = types.ModuleType("comfy.patcher_extension")

    class WrappersMP:
        DIFFUSION_MODEL = "diffusion_model"
        APPLY_MODEL = "apply_model"
        CALC_COND_BATCH = "calc_cond_batch"

    class WrapperExecutor:
        def __init__(self, original, class_obj, wrappers, idx):
            self.original, self.class_obj = original, class_obj
            self.wrappers, self.idx = list(wrappers), idx
            self.is_last = idx == len(wrappers)

        def __call__(self, *args, **kwargs):
            return WrapperExecutor(
                self.original, self.class_obj, self.wrappers, self.idx + 1
            ).execute(*args, **kwargs)

        def execute(self, *args, **kwargs):
            if self.is_last:
                return self.original(*args, **kwargs)
            return self.wrappers[self.idx](self, *args, **kwargs)

        @classmethod
        def new_class_executor(cls, original, class_obj, wrappers, idx=0):
            return cls(original, class_obj, wrappers, idx)

        @classmethod
        def new_executor(cls, original, wrappers, idx=0):
            return cls(original, None, wrappers, idx)

    def _options(store, is_model_options):
        return store.setdefault("transformer_options", {}) if is_model_options else store

    def add_wrapper_with_key(kind, key, wrapper, store, is_model_options=False):
        wrappers = _options(store, is_model_options).setdefault("wrappers", {})
        wrappers.setdefault(kind, {}).setdefault(key, []).append(wrapper)

    def get_wrappers_with_key(kind, key, store, is_model_options=False):
        options = store.get("transformer_options", {}) if is_model_options else store
        return list(options.get("wrappers", {}).get(kind, {}).get(key, []))

    def get_all_wrappers(kind, options, is_model_options=False):
        if is_model_options:
            options = options.get("transformer_options", {})
        found = []
        for entry in options.get("wrappers", {}).get(kind, {}).values():
            found.extend(entry)
        return found

    module.WrappersMP = WrappersMP
    module.WrapperExecutor = WrapperExecutor
    module.add_wrapper_with_key = add_wrapper_with_key
    module.get_wrappers_with_key = get_wrappers_with_key
    module.get_all_wrappers = get_all_wrappers
    return module


def _install_patcher_extension(monkeypatch):
    """Register the stand-in under both names ``import comfy.x as y`` reads."""
    import sys
    import types

    extension = _patcher_extension()
    package = sys.modules.get("comfy") or types.ModuleType("comfy")
    monkeypatch.setattr(package, "patcher_extension", extension, raising=False)
    monkeypatch.setitem(sys.modules, "comfy", package)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", extension)
    return extension


class _ApplyModelSeamAdapter:
    """One family whose stock forward routes no DIFFUSION_MODEL executor."""

    family = "toy"
    cfg_cond_padding = "none"
    cfg_parallel_supported = True
    cfg_split_seam = "apply_model"
    cfg_batch_constant = None


class _ToyModel:
    """A BaseModel stand-in: apply_model wraps _apply_model, as stock does."""

    def __init__(self, model_options):
        self.model_options = model_options
        self.seen = []

    # Both signatures, mutable defaults included, are comfy's own: the slice
    # rule binds the call against ``_apply_model``'s and names every argument
    # off it, so a paraphrase would test a different contract.
    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                    transformer_options={}, **kwargs):  # noqa: B006
        import comfy.patcher_extension as pe

        return pe.WrapperExecutor.new_class_executor(
            self._apply_model, self,
            pe.get_all_wrappers(pe.WrappersMP.APPLY_MODEL, transformer_options),
        ).execute(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None,
                     transformer_options={}, **kwargs):  # noqa: B006
        self.seen.append({"x": x, "t": t, "c_crossattn": c_crossattn,
                          "transformer_options": transformer_options, **kwargs})
        # Row-wise, so a rank's slice of the output is its slice of the input.
        return x * 10.0 + t.reshape(-1, 1)


def _drive_apply_model_seam(monkeypatch, rank, gathered):
    """Install the split on the apply_model seam and run one batch-2 call."""
    import dgx_monarch.adapters.cfg_parallel as cfg_parallel

    _install_patcher_extension(monkeypatch)
    monkeypatch.setattr(cfg_parallel, "cfg_world", lambda: 2)
    monkeypatch.setattr(cfg_parallel, "cfg_rank", lambda: rank)

    model_options: dict = {}
    wrapper_type = cfg_parallel.install_cfg_parallel_wrapper(
        _ApplyModelSeamAdapter(), model_options)
    assert wrapper_type == "apply_model"

    model = _ToyModel(model_options)
    transformer_options = {
        **model_options["transformer_options"],
        "cond_or_uncond": [0, 1], "uuids": ["cond", "uncond"],
        "sigmas": torch.tensor([3.0, 3.0]),
    }
    monkeypatch.setattr(
        cfg_parallel, "_all_gather_cfg",
        lambda local: gathered.append(local) or torch.cat([local] * 2, dim=0))
    out = model.apply_model(
        torch.arange(8.0).reshape(2, 4), torch.tensor([1.0, 2.0]),
        None, torch.arange(12.0).reshape(2, 2, 3), None, transformer_options,
        num_tokens=7, attention_mask=torch.ones(2, 5))
    return model, out


def test_apply_model_seam_hands_each_rank_its_own_slice(monkeypatch):
    """The split reaches the model call on a family with no DIFFUSION_MODEL
    executor: each rank runs half the rows and no argument arrives full."""
    gathered: list = []
    model, out = _drive_apply_model_seam(monkeypatch, 1, gathered)
    assert len(model.seen) == 1
    call = model.seen[0]
    # Rank 1 of 2 takes the second row of every batch-leading tensor.
    assert torch.equal(call["x"], torch.arange(8.0).reshape(2, 4)[1:2])
    assert torch.equal(call["t"], torch.tensor([2.0]))
    assert torch.equal(call["c_crossattn"], torch.arange(12.0).reshape(2, 2, 3)[1:2])
    assert torch.equal(call["attention_mask"], torch.ones(1, 5))
    # A CONDConstant is one value for the whole call and never a row.
    assert call["num_tokens"] == 7
    # The condition labels follow the rows this rank ran.
    assert call["transformer_options"]["cond_or_uncond"] == [1]
    assert call["transformer_options"]["uuids"] == ["uncond"]
    # sigmas is execution metadata; it stays whole.
    assert torch.equal(call["transformer_options"]["sigmas"], torch.tensor([3.0, 3.0]))
    assert len(gathered) == 1 and gathered[0].shape[0] == 1
    assert out.shape[0] == 2


def test_apply_model_seam_ranks_cover_the_batch_between_them(monkeypatch):
    """Rank 0 and rank 1 run disjoint rows whose union is the whole call."""
    rows = []
    for rank in (0, 1):
        model, _ = _drive_apply_model_seam(monkeypatch, rank, [])
        rows.append(model.seen[0]["x"])
    assert torch.equal(torch.cat(rows, dim=0), torch.arange(8.0).reshape(2, 4))


def test_install_is_idempotent_across_a_reload(monkeypatch):
    """add_wrapper_with_key appends, so a second install would slice twice."""
    import dgx_monarch.adapters.cfg_parallel as cfg_parallel

    _install_patcher_extension(monkeypatch)
    from dgx_monarch.constants import CFG_WRAPPER_KEY

    adapter, model_options = _ApplyModelSeamAdapter(), {}
    cfg_parallel.install_cfg_parallel_wrapper(adapter, model_options)
    cfg_parallel.install_cfg_parallel_wrapper(adapter, model_options)
    installed = model_options["transformer_options"]["wrappers"]["apply_model"]
    assert len(installed[CFG_WRAPPER_KEY]) == 1


def test_only_the_unrouted_families_declare_the_apply_model_seam():
    """The declaration alone picks the seam at run time, so pin the shipped set.

    The canary holds each declaration against ComfyUI's own source; this holds
    the set itself, so a family cannot move seams unnoticed.
    """
    from dgx_monarch.adapters import ADAPTERS

    by_seam: dict[str, set] = {}
    for adapter in ADAPTERS:
        by_seam.setdefault(adapter.cfg_split_seam, set()).add(adapter.family)
    assert by_seam["apply_model"] == {"boogu", "ernie", "omnigen2"}
    assert "flux" in by_seam["diffusion_model"] and "krea2" in by_seam["diffusion_model"]


def test_an_unknown_seam_declaration_is_refused(monkeypatch):
    import dgx_monarch.adapters.cfg_parallel as cfg_parallel

    _install_patcher_extension(monkeypatch)

    class _Bogus:
        family = "bogus"
        cfg_split_seam = "sampler"

    with pytest.raises(RuntimeError, match="cfg_split_seam"):
        cfg_parallel.cfg_split_seam_type(_Bogus())


def test_batch_one_reads_the_same_card_on_every_family(monkeypatch):
    """One card for every family, the constant families included.

    Unequal prompts on a family that publishes a per-cond CONDConstant dispatch
    one cond per rank one seam further out, so a batch of one that still
    arrives here is the sampler case, which every family answers the same way.
    The card never names CONDConstant.
    """
    boogu = _batch_one_refusal(monkeypatch, "boogu")
    assert boogu == _batch_one_refusal(monkeypatch, "flux")
    assert boogu == _batch_one_refusal(monkeypatch, "omnigen2")
    assert boogu == _batch_one_refusal(monkeypatch, "ernie")
    assert "[dgxm:P" in boogu and "cfg_pp" in boogu
    assert "CONDConstant" not in boogu


def test_only_the_constant_families_are_certain_to_dispatch():
    """Every family installs the dispatch and comfy's concat rule decides per
    render. A declared constant makes the dispatch certain, which the driver's
    pad skip and the sweep's prediction both read, so pin the shipped set."""
    from dgx_monarch.adapters import ADAPTERS
    from dgx_monarch.adapters.cfg_parallel import cfg_dispatches_per_cond

    by_family = {adapter.family: adapter for adapter in ADAPTERS}
    certain = {adapter.family for adapter in ADAPTERS
               if cfg_dispatches_per_cond(adapter)}
    assert certain == {"boogu", "omnigen2"}
    # ernie pads nothing and declares nothing, so only the cond shapes answer.
    assert by_family["ernie"].cfg_cond_padding == "none"
    assert not cfg_dispatches_per_cond(by_family["ernie"])
    # boogu keeps its "pad" rule; no pad can make its constant fold.
    assert by_family["boogu"].cfg_cond_padding == "pad"


class _ConstantFamilyAdapter:
    """A family whose two conditionings can never fold into one call."""

    family = "toy"
    cfg_cond_padding = "none"
    cfg_parallel_supported = True
    cfg_split_seam = "apply_model"
    cfg_batch_constant = "num_tokens"


class _Constant:
    """comfy.conds.CONDConstant, reduced to the value the fold rule reads."""

    def __init__(self, value):
        self.cond = value


class _ToyCond:
    """A cond object carrying comfy's two methods and its own concat rule.

    The dispatch calls ``process_cond`` and then ``can_concat`` and never looks
    at the tensor itself, so whatever this class answers is what the fold
    decision answers. ``verdict`` lets one test hold a pair of different shapes
    that concatenate anyway, which is comfy's cross-attention repeat rule.
    """

    def __init__(self, tensor, verdict=None):
        self.cond = tensor
        self.verdict = verdict
        self.batches: list[int] = []

    def process_cond(self, batch_size, area=None, **kwargs):
        self.batches.append(batch_size)
        data = self.cond
        if data.shape[0] != batch_size:
            data = data.expand(batch_size, *data.shape[1:])
        return _ToyCond(data, self.verdict)

    def can_concat(self, other):
        if self.verdict is not None:
            return self.verdict
        return self.cond.shape == other.cond.shape


def _cond_slot(tokens: int, weight: float) -> list:
    """One cond slot of the sampler's list: a list of conditioning dicts."""
    return [{"model_conds": {"num_tokens": _Constant(tokens)}, "weight": weight}]


def _shape_slot(length: int, weight: float, verdict=None) -> list:
    """One cond slot whose only conditioning is a cross-attention stream."""
    return [{"model_conds": {"c_crossattn": _ToyCond(torch.ones(1, length, 2), verdict)},
             "weight": weight}]


class _ToyCondBatch:
    """comfy's ``_calc_cond_batch``, reduced to what the dispatch relies on.

    Two properties are stock and load-bearing, and the "cond dispatch" seam in
    tests/canary/comfy_seam_contracts.py holds both against the real thing: one
    output is allocated per cond index before that cond is looked at, so a
    masked index comes back as exact zeros; and a cond's work lands at its own
    index, so masking never renumbers anything.
    """

    def __init__(self):
        self.seen: list[tuple[int, int]] = []

    @staticmethod
    def _tokens(item) -> int:
        """The one number this stand-in computes from, whichever cond ships it."""
        model_conds = item["model_conds"]
        if "num_tokens" in model_conds:
            return int(model_conds["num_tokens"].cond)
        return int(model_conds["c_crossattn"].cond.shape[1])

    def __call__(self, model, conds, x_in, timestep, model_options):
        out = [torch.zeros_like(x_in) for _ in conds]
        for index, entry in enumerate(conds):
            if entry is None:
                continue
            for item in entry:
                tokens = self._tokens(item)
                self.seen.append((index, tokens))
                out[index] = out[index] + x_in * item["weight"] + float(tokens)
        return out


def _dispatch_once(monkeypatch, conds, world, rank, gather=None, adapter=None,
                   inner=None):
    """Run one rank's cond batch. Returns (outputs, local stack, model calls)."""
    import dgx_monarch.adapters.cfg_dispatch as cfg_dispatch
    import dgx_monarch.adapters.cfg_parallel as cfg_parallel

    pe = _install_patcher_extension(monkeypatch)
    monkeypatch.setattr(cfg_dispatch, "cfg_world", lambda: world)
    monkeypatch.setattr(cfg_dispatch, "cfg_rank", lambda: rank)

    sent: dict = {}

    def _gather(local):
        sent["local"] = local
        return torch.cat([local] * world, dim=0) if gather is None else gather(local)

    monkeypatch.setattr(cfg_parallel, "_all_gather_cfg", _gather)

    model_options: dict = {}
    seam = cfg_dispatch.install_cond_dispatch_wrapper(
        adapter or _ConstantFamilyAdapter(), model_options)
    assert seam == "calc_cond_batch"

    inner = _ToyCondBatch() if inner is None else inner
    executor = pe.WrapperExecutor.new_executor(
        inner, pe.get_all_wrappers(
            pe.WrappersMP.CALC_COND_BATCH, model_options, is_model_options=True))
    out = executor.execute(
        object(), conds, torch.arange(6.0).reshape(2, 3), torch.tensor([1.0]),
        model_options)
    return out, sent.get("local"), inner.seen


def _reference(conds) -> list:
    """The list a single GPU returns for the same conds, by the same inner."""
    return _ToyCondBatch()(
        object(), conds, torch.arange(6.0).reshape(2, 3), torch.tensor([1.0]), {})


def _gathered(monkeypatch, conds, world, adapter=None):
    """Every rank's dispatch, with the gather completed from real peer data."""
    stacks, calls = [], []
    for rank in range(world):
        _out, local, seen = _dispatch_once(
            monkeypatch, conds, world, rank, adapter=adapter)
        stacks.append(local)
        calls.append(seen)
    full = torch.cat(stacks, dim=0)
    outputs = [_dispatch_once(monkeypatch, conds, world, rank,
                              gather=lambda local: full, adapter=adapter)[0]
               for rank in range(world)]
    return outputs, stacks, calls


def test_two_conds_that_never_fold_run_one_per_rank(monkeypatch):
    """Each rank runs only its own conditioning, and the gathered list is the
    single-GPU list entry for entry, in the sampler's original order."""
    conds = [_cond_slot(7, 2.0), _cond_slot(11, 5.0)]
    outputs, stacks, calls = _gathered(monkeypatch, conds, 2)
    # The model work split: one cond per rank, and its own index kept.
    assert calls == [[(0, 7)], [(1, 11)]]
    # Each rank's contribution is exactly what a single GPU computes for it.
    reference = _reference(conds)
    assert torch.equal(stacks[0][0], reference[0])
    assert torch.equal(stacks[1][0], reference[1])
    for out in outputs:
        assert len(out) == 2
        assert torch.equal(out[0], reference[0])
        assert torch.equal(out[1], reference[1])


def test_three_conds_on_two_ranks_keep_the_order(monkeypatch):
    """Contiguous groups, so the gathered slots land back in cond order and the
    odd count leaves one padded slot rather than a ragged collective."""
    conds = [_cond_slot(7, 2.0), _cond_slot(11, 5.0), _cond_slot(13, 9.0)]
    outputs, stacks, calls = _gathered(monkeypatch, conds, 2)
    assert calls == [[(0, 7), (1, 11)], [(2, 13)]]
    # Both ranks send two slots; rank 1's second is the empty contribution.
    assert stacks[0].shape[0] == 2 and stacks[1].shape[0] == 2
    assert torch.equal(stacks[1][1], torch.zeros_like(stacks[1][1]))
    reference = _reference(conds)
    for out in outputs:
        assert len(out) == 3
        assert all(torch.equal(item, want) for item, want in zip(out, reference, strict=True))


def test_a_rank_with_nothing_to_do_joins_the_gather(monkeypatch):
    """Two conds across three ranks: the third runs no model call and still
    sends its slot, so no rank waits on a collective the others skipped."""
    conds = [_cond_slot(7, 2.0), _cond_slot(11, 5.0)]
    outputs, stacks, calls = _gathered(monkeypatch, conds, 3)
    assert calls == [[(0, 7)], [(1, 11)], []]
    assert stacks[2].shape[0] == 1
    assert torch.equal(stacks[2][0], torch.zeros_like(stacks[2][0]))
    reference = _reference(conds)
    for out in outputs:
        assert len(out) == 2
        assert all(torch.equal(item, want) for item, want in zip(out, reference, strict=True))


def test_one_conditioning_takes_no_wrapper_action(monkeypatch):
    """At CFG 1.0 comfy passes [cond, None]. There is nothing to dispatch, so
    the call goes through untouched and no collective is entered."""
    conds = [_cond_slot(7, 2.0), None]
    out, local, calls = _dispatch_once(monkeypatch, conds, 2, 1)
    assert local is None
    assert calls == [(0, 7)]
    assert all(torch.equal(item, want) for item, want in zip(out, _reference(conds), strict=True))


def test_conds_that_fold_stay_on_the_slice_path(monkeypatch):
    """Equal constants mean comfy can batch them, and the slice is the faster
    path there: one call of two rows against two calls of one."""
    conds = [_cond_slot(9, 2.0), _cond_slot(9, 5.0)]
    out, local, calls = _dispatch_once(monkeypatch, conds, 2, 0)
    assert local is None
    assert calls == [(0, 9), (1, 9)]
    assert all(torch.equal(item, want) for item, want in zip(out, _reference(conds), strict=True))


def test_both_ranks_decide_the_same_path_from_the_same_conds(monkeypatch):
    """Rank agreement: the branch is taken off the cond list and cfg_world(),
    both replicated, before any collective, so no rank gathers alone."""
    import dgx_monarch.adapters.cfg_dispatch as cfg_dispatch

    for conds, expected in (([_cond_slot(9, 2.0), _cond_slot(9, 5.0)], "slice"),
                            ([_cond_slot(7, 2.0), _cond_slot(11, 5.0)], "dispatch")):
        paths = set()
        for rank in range(2):
            _dispatch_once(monkeypatch, conds, 2, rank)
            paths.add(cfg_dispatch.cfg_path_taken())
        assert paths == {expected}


def test_the_slice_steps_aside_inside_a_dispatched_call(monkeypatch):
    """A rank running one cond owns every row of that call, so the batch guard
    must not see a batch of one and refuse it."""
    import dgx_monarch.adapters.cfg_dispatch as cfg_dispatch
    import dgx_monarch.adapters.cfg_parallel as cfg_parallel

    monkeypatch.setattr(cfg_parallel, "cfg_world", lambda: 2)
    monkeypatch.setattr(cfg_parallel, "cfg_rank", lambda: 0)
    ran = []

    class Executor:
        def __init__(self):
            self.original = self.forward

        def forward(self, x, transformer_options=None):
            raise AssertionError("signature only")

        def __call__(self, x, transformer_options=None):
            ran.append(x)
            return x

    wrapper = cfg_parallel.make_cfg_parallel_wrapper(_ConstantFamilyAdapter())
    with pytest.raises(UnsupportedModelError):
        wrapper(Executor(), torch.zeros(1, 4), {})
    with cfg_dispatch._dispatch_scope():
        assert torch.equal(wrapper(Executor(), torch.zeros(1, 4), {}), torch.zeros(1, 4))
    assert len(ran) == 1
    # The flag is restored, so the next render's slice guard is armed again.
    assert not cfg_dispatch.dispatch_in_progress()


def test_cond_groups_are_contiguous_and_cover_the_list():
    from dgx_monarch.adapters.cfg_dispatch import cond_group_bounds

    assert [cond_group_bounds(2, 2, rank) for rank in range(2)] == [(0, 1), (1, 2)]
    assert [cond_group_bounds(3, 2, rank) for rank in range(2)] == [(0, 2), (2, 3)]
    assert [cond_group_bounds(2, 3, rank) for rank in range(3)] == [(0, 1), (1, 2), (2, 2)]
    # Every index lands in exactly one group, whatever the remainder.
    for count in range(1, 8):
        for world in range(1, 5):
            covered = [index for rank in range(world)
                       for index in range(*cond_group_bounds(count, world, rank))]
            assert covered == list(range(count))


def test_a_family_without_a_constant_installs_the_dispatch_too(monkeypatch):
    """The install has no declaration gate. Ernie declares nothing and its two
    prompts still do not concatenate, so the wrapper has to be there for
    comfy's own rule to answer at the seam."""
    import dgx_monarch.adapters.cfg_dispatch as cfg_dispatch

    _install_patcher_extension(monkeypatch)
    from dgx_monarch.constants import CFG_DISPATCH_WRAPPER_KEY

    model_options: dict = {}
    assert cfg_dispatch.install_cond_dispatch_wrapper(
        _ApplyModelSeamAdapter(), model_options) == "calc_cond_batch"
    installed = model_options["transformer_options"]["wrappers"]["calc_cond_batch"]
    assert len(installed[CFG_DISPATCH_WRAPPER_KEY]) == 1


def test_conds_of_different_shape_dispatch_without_any_declaration(monkeypatch):
    """The ernie case (2026-09-05): two prompts of different token counts, a
    family that pads nothing and publishes no constant. The cond objects' own
    can_concat says no, so each rank runs one conditioning instead of the slice
    wrapper refusing class P on a model call of batch 1."""
    conds = [_shape_slot(7, 2.0), _shape_slot(11, 5.0)]
    outputs, stacks, calls = _gathered(
        monkeypatch, conds, 2, adapter=_ApplyModelSeamAdapter())
    assert calls == [[(0, 7)], [(1, 11)]]
    reference = _reference(conds)
    assert torch.equal(stacks[0][0], reference[0])
    for out in outputs:
        assert all(torch.equal(item, want)
                   for item, want in zip(out, reference, strict=True))


def test_conds_of_one_shape_fold_without_any_declaration(monkeypatch):
    """Equal shapes concatenate, so the batched call exists and the slice keeps
    it. This is the path every family was measured on and it must not move."""
    conds = [_shape_slot(9, 2.0), _shape_slot(9, 5.0)]
    out, local, calls = _dispatch_once(
        monkeypatch, conds, 2, 0, adapter=_ApplyModelSeamAdapter())
    assert local is None
    assert calls == [(0, 9), (1, 9)]
    assert all(torch.equal(item, want)
               for item, want in zip(out, _reference(conds), strict=True))


def test_the_cond_objects_own_verdict_is_the_fold_decision(monkeypatch):
    """Comfy's cross-attention conds concatenate two different lengths by
    repeat when the lowest common multiple is close enough. The rule here never
    compares shapes itself, so a pair that says it concatenates folds."""
    conds = [_shape_slot(10, 2.0, verdict=True), _shape_slot(20, 5.0, verdict=True)]
    _out, local, calls = _dispatch_once(
        monkeypatch, conds, 2, 0, adapter=_ApplyModelSeamAdapter())
    assert local is None and calls == [(0, 10), (1, 20)]


def test_the_fold_rule_processes_each_cond_to_the_latent_batch(monkeypatch):
    """Comfy compares the processed conds, and process_cond is what repeats a
    cond to the latent batch, so a cond shipped at batch 1 against a latent of
    two rows must be compared at two rows."""
    from dgx_monarch.adapters.cfg_dispatch import conds_fold

    conds = [_shape_slot(9, 2.0), _shape_slot(9, 5.0)]
    assert conds_fold(conds, torch.zeros(2, 3))
    for entry in conds:
        assert entry[0]["model_conds"]["c_crossattn"].batches == [2]


def test_a_cond_this_rule_does_not_model_keeps_the_slice():
    """An area, a mask or a hook group changes what comfy compares or what a
    rank alone would patch, so such a cond takes the proved path rather than a
    guess. The declared constant is read first and is not held back by it."""
    from dgx_monarch.adapters.cfg_dispatch import conds_fold

    x_in = torch.zeros(1, 3)
    for key, value in (("area", [8, 8, 0, 0]), ("mask", torch.ones(1, 8, 8)),
                       ("hooks", object()), ("default", True)):
        conds = [_shape_slot(7, 2.0), _shape_slot(11, 5.0)]
        conds[1][0][key] = value
        assert conds_fold(conds, x_in), key
        # The same pair with a differing constant still dispatches.
        conds[0][0]["model_conds"]["num_tokens"] = _Constant(7)
        conds[1][0]["model_conds"]["num_tokens"] = _Constant(11)
        assert not conds_fold(conds, x_in, "num_tokens"), key


def test_a_hooks_key_holding_none_does_not_block_the_dispatch():
    """Comfy reads hooks by value and leaves the key present holding None, so a
    presence test there would silently keep every render on the slice."""
    from dgx_monarch.adapters.cfg_dispatch import conds_fold

    conds = [_shape_slot(7, 2.0), _shape_slot(11, 5.0)]
    for entry in conds:
        entry[0]["hooks"] = None
    assert not conds_fold(conds, torch.zeros(1, 3))


def test_two_conds_on_different_controls_keep_the_slice():
    """Comfy compares control by identity. One shared object concatenates; two
    objects, or one against none, are a case with no measurement behind it."""
    from dgx_monarch.adapters.cfg_dispatch import conds_fold

    x_in, shared = torch.zeros(1, 3), object()
    conds = [_shape_slot(7, 2.0), _shape_slot(11, 5.0)]
    conds[0][0]["control"] = shared
    assert conds_fold(conds, x_in)
    conds[1][0]["control"] = shared
    assert not conds_fold(conds, x_in)


def test_dispatch_install_is_idempotent_across_a_reload(monkeypatch):
    import dgx_monarch.adapters.cfg_dispatch as cfg_dispatch

    _install_patcher_extension(monkeypatch)
    from dgx_monarch.constants import CFG_DISPATCH_WRAPPER_KEY

    adapter, model_options = _ConstantFamilyAdapter(), {}
    cfg_dispatch.install_cond_dispatch_wrapper(adapter, model_options)
    cfg_dispatch.install_cond_dispatch_wrapper(adapter, model_options)
    installed = model_options["transformer_options"]["wrappers"]["calc_cond_batch"]
    assert len(installed[CFG_DISPATCH_WRAPPER_KEY]) == 1


class _ScaledLinear(torch.nn.Module):
    """comfy's quantized nvfp4 Linear, reduced to what the wrapper binds on.

    Only the layout name and a forward, because the shared-scale install reads
    the layout and comfy's own gate reads attributes this module lacks.
    tests/test_nvfp4_shard_scale.py holds the quantizing shape of this fake.
    """

    layout_type = "TensorCoreNVFP4Layout"

    def forward(self, value):
        return value


class _ScaledCondBatch(_ToyCondBatch):
    """The inner cond batch, with one wrapped nvfp4 linear inside each call.

    One model call per conditioning present, which is what comfy runs on the
    dispatch path and an over-count on the slice path. It decides how many
    times a reducer runs, never which reducers run, which is what the two
    tests below read.
    """

    def __init__(self, module):
        super().__init__()
        self.module = module

    def __call__(self, model, conds, x_in, timestep, model_options):
        out = super().__call__(model, conds, x_in, timestep, model_options)
        for entry in conds:
            if entry is not None:
                self.module(x_in)
        return out


def _scaled_tree(topology, reducers):
    """A one-linear model with the shared activation scale installed on it."""
    from dgx_monarch.adapters import quant_activation_scale as qas

    tree = torch.nn.Module()
    tree.proj = _ScaledLinear()
    coverage = qas.install(tree, topology, family="chroma", reducers=reducers)
    return tree, coverage


def _naming_reducers():
    applied: list[str] = []

    def named(name):
        def reduce(tensor):
            applied.append(name)
            return tensor

        return reduce

    return {"sp": named("sp"), "cfg": named("cfg")}, applied


def test_a_dispatched_cond_reduces_its_amax_over_no_cfg_group(monkeypatch):
    """The two cfg paths want different reducers.

    On the slice path one batched call is split between the ranks, so the max
    over the cfg group is the amax one GPU takes over the whole call. On the
    dispatch path each rank runs a whole conditioning of its own, which is the
    call one GPU makes for it, so merging the two amax is a scale that GPU
    never computes.
    """
    reducers, applied = _naming_reducers()
    tree, coverage = _scaled_tree({"cfg": 2, "ulysses": 2}, reducers)
    assert coverage.reducers == ("sp", "cfg")

    # Two conds of different token counts: they never fold, so this rank runs
    # its own and the wrapper enters the dispatch scope around it.
    _dispatch_once(monkeypatch, [_cond_slot(7, 2.0), _cond_slot(11, 5.0)], 2, 0,
                   inner=_ScaledCondBatch(tree.proj))
    assert applied == ["sp"]

    # The same conds at one token count fold into one batched call, which the
    # cfg group slices, so the reducer the plan carries is right again.
    applied.clear()
    _dispatch_once(monkeypatch, [_cond_slot(7, 2.0), _cond_slot(7, 5.0)], 2, 0,
                   inner=_ScaledCondBatch(tree.proj))
    assert applied == ["sp", "cfg", "sp", "cfg"]
    assert coverage.reducers == ("sp", "cfg")  # the record the class-K bar reads


def test_three_conds_on_two_ranks_leave_the_cfg_group_nothing_to_wait_on(
        monkeypatch):
    """The wedge, not only the wrong number.

    Three conds on two ranks give rank 0 two model calls and rank 1 one. A cfg
    all-reduce inside the call would be issued twice on one rank and once on
    the other, and the group would hang on the third with no peer to answer it.
    """
    counts = []
    for rank in (0, 1):
        reducers, applied = _naming_reducers()
        tree, _coverage = _scaled_tree({"cfg": 2}, reducers)
        _dispatch_once(
            monkeypatch,
            [_cond_slot(7, 2.0), _cond_slot(11, 5.0), _cond_slot(13, 3.0)],
            2, rank, inner=_ScaledCondBatch(tree.proj))
        counts.append(applied.count("cfg"))
    assert counts == [0, 0]
