"""LensAdapter CPU coverage. Nothing imports comfy at module scope; the mask and
id helpers run on plain tensors.

Pins:
  * `_usp_text_trim` reads an all-ones mask as no mask and trims a uniform
    trailing pad to the real length (the forward trims only on ring and hybrid;
    lens.py's module docstring says why);
  * a ragged batch or a hole inside the kept text raises a typed refusal, never
    a silent drop, through `_usp_text_trim` and the bound USP forward;
  * `_local_pe_ids` shards each stream's ids on its own, then joins them image
    first, and the forward's pad-row indices follow that order;
  * the "pad" cfg forward's text-length bool keep-mask (Lens polarity, not the
    additive Chroma joint shape) and its idempotency;
  * the detect signature (Lens keys also match qwen_image's, so detect lists
    lens first).
"""
import sys
import types

import pytest
import torch

from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.lens import LensAdapter, _local_pe_ids, _usp_text_trim


def test_declared_contract():
    # "pad", not "pad+mask": stock Lens takes a text-length bool keep-mask,
    # passed through unchanged (no upscale_dit_mask), and joins it after the
    # image keys, so the shared Chroma-shaped joint pad mask does not fit.
    # inject_cfg_pad_forward derives a keep-mask only for a ragged maskless
    # batch; it trims a uniform pad instead.
    assert LensAdapter.family == "lens"
    assert LensAdapter.cfg_cond_padding == "pad"
    assert LensAdapter.model_base_classes == ("Lens",)


def test_all_ones_mask_keeps_full_length():
    # An all-ones mask is no mask in real arithmetic (stock's additive key bias
    # is all zero), so every text token stays: trim == seq.
    mask = torch.ones(1, 7)
    assert _usp_text_trim(mask, 7) == 7


def test_uniform_trailing_pad_trims_to_real_length():
    # Both entries hold 4 real rows of 7, so the trim is 4.
    mask = torch.ones(2, 7)
    mask[:, 4:] = 0.0
    assert _usp_text_trim(mask, 7) == 4


def test_fully_masked_text_raises_typed():
    # A fully-zero keep-mask means "attend nothing": a real mask, not a
    # trailing pad, which always keeps at least one real token. The sharded
    # forward refuses it rather than attend all text. A fully-zero context in
    # the cfg path is real input instead (test_cfg_pad_zeroed_negative_attends_all_rows).
    with pytest.raises(UnsupportedModelError, match="masked position inside"):
        _usp_text_trim(torch.zeros(1, 5), 5)


def test_ragged_batch_raises_typed():
    # Entry 0 has 2 real rows and entry 1 has 3. After the trim to 3, entry 0's
    # kept region is still masked, so the sharded forward refuses it.
    mask = torch.ones(2, 4)
    mask[0, 2:] = 0.0
    mask[1, 3:] = 0.0
    with pytest.raises(UnsupportedModelError, match="uniform trailing pad"):
        _usp_text_trim(mask, 4)


def test_mid_sequence_hole_raises_typed():
    # A masked position inside the kept text is not a trailing pad: it survives
    # the trim and must be refused rather than attended.
    mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    with pytest.raises(UnsupportedModelError, match="masked position inside"):
        _usp_text_trim(mask, 4)


def test_local_pe_ids_img_first(monkeypatch):
    # sp=1 (identity shard): ids come out [img, txt], as the block joins
    # cat([img, txt]) and _lens_position_ids puts image rows first.
    import dgx_monarch.adapters.lens as lens_mod

    monkeypatch.setattr(lens_mod, "shard_seq", lambda t, dim=1: (t, t.shape[dim]))
    img_ids = torch.zeros(1, 4, 3)   # 4 image tokens, value 0
    txt_ids = torch.ones(1, 3, 3)    # 3 text tokens, value 1
    out = _local_pe_ids(img_ids, txt_ids)
    assert out.shape[1] == 7
    assert torch.equal(out[:, :4], img_ids)
    assert torch.equal(out[:, 4:], txt_ids)


def test_local_pe_ids_shards_each_stream_separately(monkeypatch):
    # world=2, rank 1: each stream is sharded on its own axis, then joined
    # [img, txt]. Chunking the joined ids instead would hand rank 1 only the
    # text ids 10..13.
    import dgx_monarch.adapters.lens as lens_mod

    def fake_shard(t, dim=1):
        return torch.chunk(t, 2, dim=dim)[1], t.shape[dim]  # rank 1 of 2

    monkeypatch.setattr(lens_mod, "shard_seq", fake_shard)
    img_ids = torch.arange(4).reshape(1, 4, 1).repeat(1, 1, 3).float()      # 0,1,2,3
    txt_ids = torch.arange(10, 14).reshape(1, 4, 1).repeat(1, 1, 3).float()  # 10,11,12,13
    out = _local_pe_ids(img_ids, txt_ids)
    assert torch.equal(out[:, :2, 0], torch.tensor([[2.0, 3.0]]))     # img rank-1 half
    assert torch.equal(out[:, 2:, 0], torch.tensor([[12.0, 13.0]]))   # txt rank-1 half


class _Sentinel(Exception):
    """Raised by a stub placed just past the mask gate to prove control reached it."""


def _usp_fake():
    # multi_layer_encoder_feature=False keeps the pre-mask setup trivial: the
    # single-tensor text path with identity img_in/txt_norm/txt_in reaches the
    # mask block, and time_text_embed is the first op past it.
    fake = types.SimpleNamespace()
    fake.multi_layer_encoder_feature = False
    fake.transformer_blocks = []  # inject_usp's log line reads len()
    fake.img_in = lambda t: t
    fake.txt_norm = lambda t: t
    fake.txt_in = lambda t: t

    def boom(*a, **k):
        raise _Sentinel

    fake.time_text_embed = boom
    return fake


def _inject_usp(fake):
    LensAdapter().inject_usp(fake, InjectionContext(topology_sp=2, usp_attention=object()))


def test_usp_binds_forward():
    fake = _usp_fake()
    _inject_usp(fake)
    assert callable(fake._forward)


def test_usp_all_ones_mask_passes_the_gate():
    # An all-ones mask is not refused: control reaches time_text_embed.
    fake = _usp_fake()
    _inject_usp(fake)
    with pytest.raises(_Sentinel):
        fake._forward(torch.zeros(1, 4, 2, 2), torch.zeros(1), torch.zeros(1, 3, 8),
                      attention_mask=torch.ones(1, 3))


def test_usp_no_mask_passes_the_gate():
    fake = _usp_fake()
    _inject_usp(fake)
    with pytest.raises(_Sentinel):
        fake._forward(torch.zeros(1, 4, 2, 2), torch.zeros(1), torch.zeros(1, 3, 8))


def test_usp_ragged_mask_raises_typed():
    # A ragged mask is refused at the gate, never dropped with a warning.
    fake = _usp_fake()
    _inject_usp(fake)
    mask = torch.ones(2, 3)
    mask[0, 2:] = 0.0  # entry 0 shorter than entry 1
    with pytest.raises(UnsupportedModelError, match="uniform trailing pad"):
        fake._forward(torch.zeros(2, 4, 2, 2), torch.zeros(2), torch.zeros(2, 3, 8),
                      attention_mask=mask)


@pytest.fixture
def lens_position_ids_stub(monkeypatch):
    """Stub comfy.ldm.lens.model._lens_position_ids, the one comfy import inside
    usp_forward. Shape only: the capture below raises before any block reads
    the ids, so only the image/text split (img_len = height*width) matters.
    """

    def fake_lens_position_ids(frame, height, width, text_seq_len, scale_rope=True, device=None):
        return torch.zeros(height * width + text_seq_len, 3)

    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    lens_pkg = types.ModuleType("comfy.ldm.lens")
    model_mod = types.ModuleType("comfy.ldm.lens.model")
    model_mod._lens_position_ids = fake_lens_position_ids
    comfy.ldm = ldm
    ldm.lens = lens_pkg
    lens_pkg.model = model_mod
    for name, mod in [("comfy", comfy), ("comfy.ldm", ldm),
                      ("comfy.ldm.lens", lens_pkg), ("comfy.ldm.lens.model", model_mod)]:
        monkeypatch.setitem(sys.modules, name, mod)


def _usp_fake_full():
    # Stubs time_text_embed and pos_embed so the forward reaches the
    # shard_seq/padded_row_indices/usp_options wiring.
    fake = _usp_fake()
    fake.time_text_embed = lambda timestep, hidden_states: torch.zeros(1)
    fake.pos_embed = lambda ids: ids
    return fake


class _DropRowsCapture(Exception):
    """Raised by the usp_options stub with its drop_rows argument, so a test
    reads the value without running the block loop or xFuser collectives."""

    def __init__(self, drop_rows):
        self.drop_rows = drop_rows


@pytest.fixture
def capture_drop_rows(monkeypatch):
    import dgx_monarch.adapters.lens as lens_mod

    def fake_usp_options(transformer_options, usp_attention, drop_rows=None,
                         sequence_order=None):
        raise _DropRowsCapture(drop_rows)

    monkeypatch.setattr(lens_mod, "usp_options", fake_usp_options)


def test_usp_drop_rows_img_first_odd_image_even_text(
        sp, lens_position_ids_stub, capture_drop_rows):
    # world=2: img_len=3 (odd -> local 2, pad 1), text_seq_len=4 (even -> local
    # 2, no pad). Gathered layout is [img_r0(2), txt_r0(2), img_r1(2), txt_r1(2)]
    # = 8 rows; img's pad row is rank1's local slot 1 -> gathered row
    # 1*4 + 0 + 1 = 5. Text first, krea2's order, would give row 7, so this
    # pins the image-first order.
    sp(2, 0)
    fake = _usp_fake_full()
    _inject_usp(fake)
    with pytest.raises(_DropRowsCapture) as exc:
        fake._forward(torch.zeros(1, 2, 1, 3), torch.zeros(1), torch.zeros(1, 4, 8))
    assert exc.value.drop_rows == [5]


def test_usp_drop_rows_even_image_odd_text(sp, lens_position_ids_stub, capture_drop_rows):
    # world=2: img_len=2 (even -> local 1, no pad), text_seq_len=3 (odd ->
    # local 2, pad 1). A forward that discards the text stream's original
    # length (`_` for `txt_orig_len`) names no drop row for a padded text
    # stream. Gathered layout [img_r0(1), txt_r0(2), img_r1(1), txt_r1(2)] = 6
    # rows; the text pad row is rank1's local slot 1 -> gathered row
    # 1*3 + 1 + 1 = 5.
    sp(2, 0)
    fake = _usp_fake_full()
    _inject_usp(fake)
    with pytest.raises(_DropRowsCapture) as exc:
        fake._forward(torch.zeros(1, 2, 1, 2), torch.zeros(1), torch.zeros(1, 3, 8))
    assert exc.value.drop_rows == [5]


def test_usp_drop_rows_empty_when_both_streams_divide_world(
        sp, lens_position_ids_stub, capture_drop_rows):
    # world=2, img_len=2 and text_seq_len=2 both divide the SP world:
    # padded_row_indices returns [], and the dispatcher keeps its maskless path.
    sp(2, 0)
    fake = _usp_fake_full()
    _inject_usp(fake)
    with pytest.raises(_DropRowsCapture) as exc:
        fake._forward(torch.zeros(1, 2, 1, 2), torch.zeros(1), torch.zeros(1, 2, 8))
    assert exc.value.drop_rows == []


class _FakeLensCfg:
    """Fake whose class defines _forward, so `type(self)._forward` (the stock
    seam the cfg-pad forward calls) resolves to this recording stub."""

    def __init__(self):
        self.recorded = []

    def _forward(self, x, timestep, context, attention_mask=None,
                 transformer_options=None, control=None, **kwargs):
        self.recorded.append({"context": context, "attention_mask": attention_mask})
        return context


def _cfg_call(model, context, attention_mask=None):
    bs = context.shape[0]
    model._forward(torch.zeros(bs, 4, 2, 2), torch.zeros(bs), context,
                   attention_mask=attention_mask)
    return model.recorded[-1]


def test_cfg_pad_trims_uniform_pad_maskfree():
    model = _FakeLensCfg()
    LensAdapter().inject_cfg_pad_forward(model)
    ctx = torch.ones(1, 6, 8)
    ctx[:, 4:] = 0.0  # driver-appended rows
    rec = _cfg_call(model, ctx)
    assert rec["context"].shape[1] == 4       # trailing pad trimmed
    assert rec["attention_mask"] is None      # uniform pad, so no mask


def test_cfg_pad_ragged_keeps_bool_keep_mask():
    model = _FakeLensCfg()
    LensAdapter().inject_cfg_pad_forward(model)
    ctx = torch.ones(2, 6, 8)
    ctx[0, 4:] = 0.0
    ctx[1, 5:] = 0.0  # different real lengths in one call
    rec = _cfg_call(model, ctx)
    assert rec["context"].shape[1] == 5       # trimmed to the longest real length
    mask = rec["attention_mask"]
    # A (B, trim) text-length bool keep-mask (True = attend): Lens polarity,
    # not the additive (1, 1, txt+img) Chroma joint shape.
    assert mask is not None and mask.dtype == torch.bool and mask.shape == (2, 5)
    assert int((~mask[0]).sum()) == 1 and int((~mask[1]).sum()) == 0
    assert bool(mask[0, 4].item()) is False   # entry 0's pad row masked off


def test_cfg_pad_unpadded_is_stock_passthrough():
    model = _FakeLensCfg()
    LensAdapter().inject_cfg_pad_forward(model)
    rec = _cfg_call(model, torch.ones(1, 5, 8))
    assert rec["context"].shape[1] == 5 and rec["attention_mask"] is None


def test_cfg_pad_zeroed_negative_attends_all_rows():
    # A fully-zero ConditioningZeroOut negative is real input, not pad: no trim
    # and no mask, so stock attends every row, as on one GPU.
    model = _FakeLensCfg()
    LensAdapter().inject_cfg_pad_forward(model)
    rec = _cfg_call(model, torch.zeros(1, 5, 8))
    assert rec["context"].shape[1] == 5 and rec["attention_mask"] is None


def test_cfg_pad_real_mask_passes_through_untouched():
    model = _FakeLensCfg()
    LensAdapter().inject_cfg_pad_forward(model)
    real = torch.ones(1, 5, dtype=torch.bool)
    rec = _cfg_call(model, torch.ones(1, 5, 8), attention_mask=real)
    assert rec["attention_mask"] is real  # honored by stock, never re-derived


def test_cfg_pad_bind_is_idempotent():
    # A second install must leave `type(self)._forward` on the stock class
    # method; the wrapper there would trim twice or recurse.
    model = _FakeLensCfg()
    adapter = LensAdapter()
    adapter.inject_cfg_pad_forward(model)
    adapter.inject_cfg_pad_forward(model)
    ctx = torch.ones(1, 6, 8)
    ctx[:, 4:] = 0.0
    rec = _cfg_call(model, ctx)
    assert rec["context"].shape[1] == 4 and rec["attention_mask"] is None


# Control overlap: stock Lens adds control to the image stream at global rows
# [0, add_len) (the ControlNet branch of comfy's LensTransformer2DModel._forward),
# so span_start is always 0 and image and text stay separate. That math is the
# shared flux_family._control_span_overlap, pinned once in
# test_flux_adapter.py::test_sharded_adds_reconstruct_stock_global_add, whose
# span_start=0 rows cover this family's whole case.


# matches() is tested in tests/test_adapter_matches.py: exact-type accept
# (comfy.model_base has one Lens class and no subclass of it), a typed refusal
# for an unvetted subclass, and a decline for a foreign model.


# A copy of the lens row of detect._SIGNATURES, built on comfy's own Lens
# fingerprint in comfy/model_detection.py (attn.norm_added_q.weight and
# img_mlp.w1.weight).
_LENS_SIG = ("transformer_blocks.", "attn.norm_added_q.", "img_mlp.w1.")
_QWEN_SIG = ("transformer_blocks.", "txt_norm.", "img_in.")

_LENS_KEYS = [
    "model.diffusion_model.img_in.weight",
    "model.diffusion_model.txt_in.weight",
    "model.diffusion_model.txt_norm.0.weight",  # multi-layer ModuleList
    "model.diffusion_model.time_text_embed.timestep_embedder.linear_1.weight",
    "model.diffusion_model.transformer_blocks.0.attn.img_qkv.weight",
    "model.diffusion_model.transformer_blocks.0.attn.txt_qkv.weight",
    "model.diffusion_model.transformer_blocks.0.attn.norm_added_q.weight",
    "model.diffusion_model.transformer_blocks.0.attn.to_add_out.weight",
    "model.diffusion_model.transformer_blocks.0.img_mod.1.weight",
    "model.diffusion_model.transformer_blocks.0.img_mlp.w1.weight",  # SwiGLU GateMLP
    "model.diffusion_model.norm_out.linear.weight",
    "model.diffusion_model.proj_out.weight",
]

# Qwen-Image keys: a FeedForward img_mlp (net.*, no w1) and separate
# to_q/add_q_proj (no fused img_qkv).
_QWEN_KEYS = [
    "model.diffusion_model.txt_norm.weight",
    "model.diffusion_model.img_in.weight",
    "model.diffusion_model.txt_in.weight",
    "model.diffusion_model.transformer_blocks.0.img_mod.1.weight",
    "model.diffusion_model.transformer_blocks.0.attn.to_q.weight",
    "model.diffusion_model.transformer_blocks.0.attn.add_q_proj.weight",
    "model.diffusion_model.transformer_blocks.0.attn.norm_added_q.weight",
    "model.diffusion_model.transformer_blocks.0.img_mlp.net.0.proj.weight",
    "model.diffusion_model.norm_out.linear.weight",
    "model.diffusion_model.proj_out.weight",
]


def _sig_matches(keys, sig):
    from dgx_monarch.adapters.detect import _has_segment, _strip_prefix

    stripped = [_strip_prefix(k) for k in keys]
    return all(_has_segment(stripped, needle) for needle in sig)


def test_lens_signature_matches_lens_excludes_qwen():
    # Qwen-Image has no img_mlp.w1. (SwiGLU), so that needle excludes it.
    # attn.norm_added_q. marks a dual-stream block.
    assert _sig_matches(_LENS_KEYS, _LENS_SIG)
    assert not _sig_matches(_QWEN_KEYS, _LENS_SIG)


def test_lens_keys_collide_with_qwen_signature_so_lens_must_precede():
    # Lens also carries transformer_blocks., txt_norm. and img_in., so it
    # matches the qwen_image signature; detect._SIGNATURES must list lens first.
    assert _sig_matches(_LENS_KEYS, _QWEN_SIG)
    assert not _sig_matches(_QWEN_KEYS, _LENS_SIG)  # ...and qwen never sniffs as lens


def test_lens_keys_do_not_match_other_existing_families():
    # Outside qwen_image, which detect resolves by order, the lens keys match
    # no other family's signature.
    from dgx_monarch.adapters.detect import _SIGNATURES

    for family, needles in _SIGNATURES:
        if family in ("qwen_image", "lens"):
            continue  # qwen collision is handled by ordering (test above)
        assert not _sig_matches(_LENS_KEYS, needles), f"lens keys wrongly match {family}"
