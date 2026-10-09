"""CPU tests for BooguAdapter using a fake model and patched collectives.

Coverage includes four-stage sharding, joint RoPE alignment, double-to-single
re-sharding, output negation, and typed token-count and batch guards. CFG
padding tests cover caption-length detection, uniform trimming, per-row RoPE
offsets, and invalid masks. Dispatch and detection tests distinguish Boogu
from its Omnigen2 parent and require the correct registry order.

ComfyUI is not imported at module scope; matches() uses a fake model_base.
"""
import sys
import types

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters import boogu as bg
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.boogu import (
    BooguAdapter,
    _assert_uniform_seq_lengths,
    _content_caption_lengths,
    _keep_mask_caption_lengths,
    _split_joint_rope,
    _text_pad_key_bias,
)
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from fake_model_base_helpers import install_fake_model_base


def _ids(start, length):
    # Distinct per-position values, so alignment can be compared.
    return torch.arange(start, start + length).view(1, length, 1).float()


def test_declared_contract():
    # "pad", not "pad+mask": Boogu inherits Omnigen2.extra_conds, which sums a
    # conditioning-dict attention_mask into the num_tokens CONDConstant as a 0/1
    # token mask. A pad+mask additive bias there would give a wrong num_tokens.
    # The cfg-pad forward handles the pad itself.
    assert BooguAdapter.family == "boogu"
    assert BooguAdapter.cfg_cond_padding == "pad"
    assert BooguAdapter.model_base_classes == ("Boogu",)
    assert BooguAdapter.exact_model_base == "Boogu"


def test_split_joint_rope_splits_caption_and_image():
    # Full joint rope is [caption(l_instruct) ; ref+noise]; the split feeds the
    # double stage its instruct rope and img rope separately.
    full = _ids(0, 9)
    cap, img = _split_joint_rope(full, 3)
    assert torch.equal(cap, full[:, :3])
    assert torch.equal(img, full[:, 3:])
    assert cap.shape[1] == 3 and img.shape[1] == 6


def test_uniform_seq_lengths_passes_and_ragged_rejects():
    _assert_uniform_seq_lengths([12])
    _assert_uniform_seq_lengths([12, 12])
    with pytest.raises(UnsupportedModelError, match="non-uniform per-entry sequence lengths"):
        _assert_uniform_seq_lengths([12, 15])


def test_stage3_joint_rope_is_two_local_shards_not_a_full_chunk(sp):
    # The double block attends [instruct_local ; img_local], so its joint rope is
    # the instruct rope shard then the img rope shard (docs/ADAPTERS.md, "RoPE
    # alignment under sharding"). A chunk of the embedded full rope gives a rank
    # the wrong stream boundaries even when both streams divide the SP degree.
    l_instruct, img_len = 4, 4
    full = _ids(0, l_instruct + img_len)
    instruct_rope, img_rope = _split_joint_rope(full, l_instruct)
    for rank in (0, 1):
        sp(2, rank)
        instr_local, _ = base.shard_seq(instruct_rope, dim=1)
        img_local, _ = base.shard_seq(img_rope, dim=1)
        joint_local = torch.cat([instr_local, img_local], dim=1)
        full_chunk, _ = base.shard_seq(full, dim=1)
        # The joint rope is the two independent shards concatenated ...
        assert torch.equal(joint_local, torch.cat([instr_local, img_local], dim=1))
        # ... and differs from a chunk of the embedded full rope.
        assert not torch.equal(joint_local, full_chunk)


def test_stage3_img_rope_shard_matches_img_token_shard(sp):
    # The block's img self-attention uses the img-only rope; it must shard
    # identically to the img (ref+noise) token stream so RoPE stays aligned.
    combined_img = _ids(100, 6)         # ref+noise tokens
    img_rope = _ids(500, 6)
    for rank in (0, 1):
        sp(2, rank)
        img_tok_local, tok_orig = base.shard_seq(combined_img, dim=1)
        img_rope_local, rope_orig = base.shard_seq(img_rope, dim=1)
        assert tok_orig == rope_orig == 6
        assert img_tok_local.shape[1] == img_rope_local.shape[1]


def test_double_to_single_reshard_round_trips(sp):
    # Stage 3 keeps instruct and img as separate shards; the re-shard gathers
    # both, concatenates [text, combined_img], and re-shards the joined stream
    # for stage 4. Every step must round-trip to the full joined sequence.
    instruct = _ids(0, 4)
    combined_img = _ids(50, 6)          # ref+noise
    joined_expected = torch.cat([instruct, combined_img], dim=1)

    instr_shards, img_shards, orig_i, orig_j = [], [], 0, 0
    for rank in (0, 1):
        sp(2, rank)
        il, orig_i = base.shard_seq(instruct, dim=1)
        jl, orig_j = base.shard_seq(combined_img, dim=1)
        instr_shards.append(il)
        img_shards.append(jl)
    instruct_full = torch.cat(instr_shards, dim=1).narrow(1, 0, orig_i)
    img_full = torch.cat(img_shards, dim=1).narrow(1, 0, orig_j)
    joined = torch.cat([instruct_full, img_full], dim=1)
    assert torch.equal(joined, joined_expected)

    # Re-shard the joined stream (stage 4) and gather it back.
    reshards, orig = [], 0
    for rank in (0, 1):
        sp(2, rank)
        local, orig = base.shard_seq(joined, dim=1)
        reshards.append(local)
    regathered = torch.cat(reshards, dim=1).narrow(1, 0, orig)
    assert torch.equal(regathered, joined_expected)


def test_boogu_28_7_gqa_expands_kv_before_uly2(monkeypatch):
    calls = []

    class FakeLongContextAttention:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, _joint, q, k, v):
            calls.append((q, k, v))
            return q

    class FakeAttnType:
        @classmethod
        def __class_getitem__(cls, name):
            return name

    long_ctx = types.ModuleType("xfuser.core.long_ctx_attention")
    long_ctx.xFuserLongContextAttention = FakeLongContextAttention
    kernels = types.ModuleType("yunchang.kernels")
    kernels.AttnType = FakeAttnType
    monkeypatch.setitem(sys.modules, "xfuser.core.long_ctx_attention", long_ctx)
    monkeypatch.setitem(sys.modules, "yunchang.kernels", kernels)

    attn = base.make_usp_attention("TORCH_FLASH")
    override = base.usp_options({}, attn)["optimized_attention_override"]
    q = torch.randn(1, 28, 6, 2)
    k = torch.arange(7.0).view(1, 7, 1, 1).expand(1, 7, 6, 2)
    v = k + 100
    out = override(
        None, q, k, v, 28,
        skip_reshape=True, enable_gqa=True,
    )

    q_l, k_l, v_l = calls[0]
    assert q_l.shape == k_l.shape == v_l.shape == (1, 6, 28, 2)
    assert torch.equal(k_l[0, 0, :, 0], torch.arange(7.0).repeat_interleave(4))
    assert k_l.shape[2] % 2 == 0  # uly2 sees 28-head MHA, never raw 7-head KV
    assert out.shape == (1, 6, 56)

    calls.clear()
    out_flat = override(
        None,
        q.transpose(1, 2).reshape(1, 6, 56),
        k.transpose(1, 2).reshape(1, 6, 14),
        v.transpose(1, 2).reshape(1, 6, 14),
        28,
        enable_gqa=True,
    )
    assert calls[0][1].shape == (1, 6, 28, 2)
    assert out_flat.shape == (1, 6, 56)


def test_gqa_head_mismatch_requires_explicit_flag(monkeypatch):
    class FakeLongContextAttention:
        def __init__(self, **kwargs):
            pass

        def __call__(self, _joint, q, k, v):
            return q

    class FakeAttnType:
        @classmethod
        def __class_getitem__(cls, name):
            return name

    long_ctx = types.ModuleType("xfuser.core.long_ctx_attention")
    long_ctx.xFuserLongContextAttention = FakeLongContextAttention
    kernels = types.ModuleType("yunchang.kernels")
    kernels.AttnType = FakeAttnType
    monkeypatch.setitem(sys.modules, "xfuser.core.long_ctx_attention", long_ctx)
    monkeypatch.setitem(sys.modules, "yunchang.kernels", kernels)

    attn = base.make_usp_attention("TORCH_FLASH")
    with pytest.raises(UnsupportedModelError, match="enable_gqa"):
        attn(
            torch.randn(1, 28, 6, 2),
            torch.randn(1, 7, 6, 2),
            torch.randn(1, 7, 6, 2),
            28,
            skip_reshape=True,
        )


class _RecBlock:
    """Records every call; returns the shape the stage's caller expects."""

    def __init__(self, kind, sink):
        self.kind, self.sink = kind, sink

    def __call__(self, *args, **kwargs):
        self.sink.append((self.kind, args, kwargs))
        if self.kind == "double":       # returns (img, instruct)
            return args[0], args[1]
        return args[0]                  # context / noise / ref / single: pass the stream


class _FakeBoogu:
    """Minimal BooguTransformer2DModel: identity embed/refine, controlled rope
    and flatten, recording blocks. hidden dim 4 == patch(2)*patch(2)*out(1) so the
    final rearrange resolves."""

    def __init__(self, sink, with_ref):
        self.patch_size = 2
        self.with_ref = with_ref
        self.sink = sink
        self.image_index_embedding = torch.zeros(5, 4)
        self.context_refiner = [_RecBlock("context", sink)]
        self.noise_refiner = [_RecBlock("noise", sink)]
        self.ref_image_refiner = [_RecBlock("ref", sink)]
        self.double_stream_layers = [_RecBlock("double", sink)]
        self.single_stream_layers = [_RecBlock("single", sink)]

    # per-token identity embedders
    def x_embedder(self, t):
        return t

    def ref_image_patch_embedder(self, t):
        return t

    def time_caption_embed(self, timestep, context, dtype):
        return torch.tensor([[7.0]]), context      # (temb marker, embedded caption == input)

    def flat_and_pad_to_seq(self, hidden_states, ref):
        noise = torch.arange(4 * 4).view(1, 4, 4).float()   # 4 noise tokens, dim 4
        if self.with_ref:
            ref_flat = torch.arange(2 * 4).view(1, 2, 4).float()  # 2 ref tokens
            l_ref, ref_sizes = [[2]], [[(4, 4)]]
        else:
            ref_flat, l_ref, ref_sizes = None, [[0]], [None]
        return (noise, ref_flat, None, None, l_ref, [4], ref_sizes, [(8, 8)])

    def rope_embedder(self, bs, enc_len, cap_lens, l_ref, l_img, ref_sizes, img_sizes, device):
        ref_len = 2 if self.with_ref else 0
        full = _ids(0, 3 + ref_len + 4)            # [cap(3) ; ref ; noise]
        return (_ids(0, 3), _ids(3, ref_len), _ids(3 + ref_len, 4), full, [3], [3 + ref_len + 4])

    def norm_out(self, hidden_states, temb):
        return hidden_states


@pytest.fixture
def comfy_stubs(monkeypatch):
    """Stub the two comfy leaves the forward imports (pad_to_patch_size,
    cast_to), plus identity shard/gather so the sp=1 structure is observable."""
    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    common_dit = types.ModuleType("comfy.ldm.common_dit")
    common_dit.pad_to_patch_size = lambda x, patch: x
    mm = types.ModuleType("comfy.model_management")
    mm.cast_to = lambda t, dtype=None, device=None: t
    comfy.ldm = ldm
    ldm.common_dit = common_dit
    comfy.model_management = mm
    for name, mod in [("comfy", comfy), ("comfy.ldm", ldm),
                      ("comfy.ldm.common_dit", common_dit),
                      ("comfy.model_management", mm)]:
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(bg, "shard_seq", lambda t, dim=1: (t, t.shape[dim]))
    monkeypatch.setattr(bg, "sp_gather", lambda t, orig, dim=1: t)


def _run_forward(with_ref):
    sink = []
    model = _FakeBoogu(sink, with_ref)
    BooguAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))
    ref = [torch.zeros(1, 16, 1, 8, 8)] if with_ref else None
    out = model.forward(torch.zeros(1, 1, 4, 4), torch.zeros(1), torch.ones(1, 3, 4),
                        3, ref_latents=ref, transformer_options={})
    return sink, out


@pytest.mark.parametrize("with_ref", [False, True])
def test_four_stage_wiring_and_negation(comfy_stubs, with_ref):
    sink, out = _run_forward(with_ref)
    kinds = [k for k, _a, _kw in sink]

    # Stage order: context (local), noise, [ref], double, single.
    if with_ref:
        assert kinds == ["context", "noise", "ref", "double", "single"]
    else:
        assert kinds == ["context", "noise", "double", "single"]

    rec = {k: (a, kw) for k, a, kw in sink}

    # Stage 1 stays local: plain transformer_options, no USP override.
    ctx_args, ctx_kw = rec["context"]
    assert "optimized_attention_override" not in ctx_kw["transformer_options"]
    assert ctx_args[0].shape[1] == 3                      # full caption, unsharded
    assert ctx_args[1] is None                            # text mask (none here)

    # Stages 2-4 route through the USP override.
    for stage in (["noise", "ref"] if with_ref else ["noise"]) + ["double", "single"]:
        assert "optimized_attention_override" in rec[stage][1]["transformer_options"]

    # Stage 3 double block: img and instruct as separate streams, the joint and
    # img ropes, and both masks None.
    d_args, d_kw = rec["double"]
    img_in, instruct_in, joint_rope, img_rope = d_args[0], d_args[1], d_args[2], d_args[3]
    ref_len = 2 if with_ref else 0
    assert instruct_in.shape[1] == 3                       # instruct stream
    assert img_in.shape[1] == ref_len + 4                  # [ref ; noise] image stream
    assert joint_rope.shape[1] == 3 + ref_len + 4          # [instruct_rope ; img_rope]
    assert img_rope.shape[1] == ref_len + 4                # img-only rope
    assert d_kw["joint_attention_mask"] is None and d_kw["img_attention_mask"] is None

    # Stage 4 single block: the re-concatenated [text, ref, img] stream and the full rope.
    s_args, _s_kw = rec["single"]
    assert s_args[0].shape[1] == 3 + ref_len + 4
    assert s_args[1] is None                               # maskless
    assert s_args[2].shape[1] == 3 + ref_len + 4           # full joint rope

    # Output is the negated noise tail, as stock BooguTransformer2DModel.forward
    # returns it; with identity blocks that is the flattened noise the fake made.
    from einops import rearrange
    noise = torch.arange(4 * 4).view(1, 4, 4).float()
    expected = rearrange(noise, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=2, w=2, p1=2, p2=2)
    assert torch.equal(out, -expected)


def test_ref_index_embedding_added_pre_shard(comfy_stubs):
    # The in-place image_index_embedding add runs before the shard, on the full
    # ref stream: a non-zero marker must reach the ref refiner.
    sink = []
    model = _FakeBoogu(sink, with_ref=True)
    model.image_index_embedding = torch.full((5, 4), 3.0)
    BooguAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))
    model.forward(torch.zeros(1, 1, 4, 4), torch.zeros(1), torch.ones(1, 3, 4),
                  3, ref_latents=[torch.zeros(1, 16, 1, 8, 8)], transformer_options={})
    ref_args = next(a for k, a, _kw in sink if k == "ref")
    base_ref = torch.arange(2 * 4).view(1, 2, 4).float()
    assert torch.equal(ref_args[0], base_ref + 3.0)        # index embedding applied


def _inject(model):
    BooguAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))


def test_usp_binds_forward(comfy_stubs):
    model = _FakeBoogu([], with_ref=False)
    _inject(model)
    assert callable(model.forward) and "forward" in model.__dict__


def test_num_tokens_mismatch_rejects_typed(comfy_stubs):
    # num_tokens (4) != caption length (3): the image rope would be mis-sliced.
    model = _FakeBoogu([], with_ref=False)
    _inject(model)
    with pytest.raises(UnsupportedModelError, match="num_tokens"):
        model.forward(torch.zeros(1, 1, 4, 4), torch.zeros(1), torch.ones(1, 3, 4),
                      4, transformer_options={})


def test_ragged_batch_rejects_typed(comfy_stubs):
    # rope_embedder reports non-uniform per-entry seq_lengths: a typed reject.
    model = _FakeBoogu([], with_ref=False)
    model.rope_embedder = lambda *a, **k: (_ids(0, 3), _ids(3, 0), _ids(3, 4),
                                           _ids(0, 7), [3], [7, 9])
    _inject(model)
    with pytest.raises(UnsupportedModelError, match="non-uniform per-entry sequence lengths"):
        model.forward(torch.zeros(1, 1, 4, 4), torch.zeros(1), torch.ones(1, 3, 4),
                      3, transformer_options={})


def test_caption_lengths_from_content_uniform_ragged_zeroout():
    # With no mask, a row's real length runs to its last non-zero token.
    uni = torch.ones(1, 6, 4)
    uni[:, 4:] = 0.0
    assert _content_caption_lengths(uni) == [4]

    rag = torch.ones(2, 6, 4)
    rag[0, 4:] = 0.0
    rag[1, 5:] = 0.0
    assert _content_caption_lengths(rag) == [4, 5]

    # ConditioningZeroOut is real input, so it keeps its full length.
    assert _content_caption_lengths(torch.zeros(1, 5, 4)) == [5]


def test_caption_lengths_from_keep_mask():
    mask = torch.zeros(2, 6)
    mask[0, :4] = 1.0
    mask[1, :5] = 1.0
    assert _keep_mask_caption_lengths(mask, 6) == [4, 5]
    assert _keep_mask_caption_lengths(torch.ones(1, 5), 5) == [5]
    # A (B, 1, tokens) mask is the same keep-mask with a broadcast axis.
    assert _keep_mask_caption_lengths(torch.ones(1, 1, 5), 5) == [5]


def test_keep_mask_rejects_gapped_and_foreign_conventions():
    gapped = torch.ones(1, 5)
    gapped[0, 1] = 0.0                                  # a hole inside the caption
    with pytest.raises(UnsupportedModelError, match="not a prefix"):
        _keep_mask_caption_lengths(gapped, 5)

    leading = torch.ones(1, 5)
    leading[0, :2] = 0.0                                # left padding
    with pytest.raises(UnsupportedModelError, match="not a prefix"):
        _keep_mask_caption_lengths(leading, 5)

    additive = torch.zeros(1, 5)
    additive[0, 3:] = torch.finfo(torch.float32).min    # chroma-style bias, not a keep mask
    with pytest.raises(UnsupportedModelError, match="not the keep-mask"):
        _keep_mask_caption_lengths(additive, 5)

    with pytest.raises(UnsupportedModelError, match="not the keep-mask"):
        _keep_mask_caption_lengths(torch.ones(1, 4), 5)  # mask narrower than the caption


def test_remaining_cfg_refusals_are_typed_class_p():
    # Both are class P: no guard, no waiver, and each names the topologies that
    # still run the workflow (DESIGN.md section 5.9).
    for build in (lambda: _keep_mask_caption_lengths(torch.ones(1, 4), 5),
                  lambda: _keep_mask_caption_lengths(
                      torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]]), 5)):
        with pytest.raises(UnsupportedModelError) as excinfo:
            build()
        tag = parse_leading_refusal_tag(str(excinfo.value))
        assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
        assert tag.guard is None and not tag.waivable
        assert "'single'" in str(excinfo.value) and "'uly2'" in str(excinfo.value)


def test_text_pad_key_bias_drops_exactly_the_pad_columns():
    bias = _text_pad_key_bias([2, 4], 4, 3, torch.float32, torch.device("cpu"))
    assert bias.shape == (2, 1, 1, 7)                    # (B, 1, 1, text + image)
    drop = torch.finfo(torch.float32).min
    assert torch.equal(bias[0, 0, 0], torch.tensor([0.0, 0.0, drop, drop, 0, 0, 0]))
    assert torch.equal(bias[1, 0, 0], torch.zeros(7))    # row 1 fills the width
    # Image keys are never dropped: every row attends the whole image stream.
    assert torch.equal(bias[:, 0, 0, 4:], torch.zeros(2, 3))


class _FakeBooguCfg:
    """Fake whose class defines forward, so `type(self).forward` (the stock seam
    the cfg-pad forward delegates to) resolves to this recording stub."""

    def __init__(self):
        self.recorded = []

    def forward(self, x, timesteps, context, num_tokens, ref_latents=None,
                attention_mask=None, transformer_options=None, **kwargs):
        self.recorded.append({"context": context, "num_tokens": num_tokens,
                              "attention_mask": attention_mask})
        return context


def _cfg_call(model, context, attention_mask=None):
    model.forward(torch.zeros(context.shape[0], 1, 4, 4), torch.zeros(context.shape[0]),
                  context, context.shape[1], attention_mask=attention_mask)
    return model.recorded[-1]


def test_cfg_pad_trims_uniform_and_fixes_num_tokens():
    model = _FakeBooguCfg()
    BooguAdapter().inject_cfg_pad_forward(model)
    ctx = torch.ones(1, 6, 4)
    ctx[:, 4:] = 0.0                                        # driver-appended pad
    rec = _cfg_call(model, ctx)
    assert rec["context"].shape[1] == 4                    # trimmed to real length
    assert rec["num_tokens"] == 4                          # num_tokens re-derived
    assert rec["attention_mask"] is None


def test_cfg_pad_unpadded_is_stock_passthrough():
    model = _FakeBooguCfg()
    BooguAdapter().inject_cfg_pad_forward(model)
    rec = _cfg_call(model, torch.ones(1, 5, 4))
    assert rec["context"].shape[1] == 5 and rec["num_tokens"] == 5


def test_cfg_pad_zeroout_delegates_full_length():
    model = _FakeBooguCfg()
    BooguAdapter().inject_cfg_pad_forward(model)
    rec = _cfg_call(model, torch.zeros(1, 5, 4))           # ConditioningZeroOut negative
    assert rec["context"].shape[1] == 5 and rec["num_tokens"] == 5
    assert rec["attention_mask"] is None


def test_cfg_pad_uniform_keep_mask_trims_to_the_stock_forward():
    # A padded caption whose rows share one real length is the reachable case:
    # ComfyUI only batches conds whose num_tokens agree, and num_tokens is that
    # length. Trimming the pad away makes the stock forward the reference.
    model = _FakeBooguCfg()
    BooguAdapter().inject_cfg_pad_forward(model)
    mask = torch.zeros(2, 6)
    mask[:, :4] = 1.0
    rec = _cfg_call(model, torch.ones(2, 6, 4), attention_mask=mask)
    assert rec["context"].shape[1] == 4 and rec["num_tokens"] == 4
    assert rec["attention_mask"] is None


def test_cfg_pad_bind_is_idempotent():
    # A repeat install must not make `type(self).forward` the wrapper (which
    # would double-trim or recurse): it stays the stock class method.
    model = _FakeBooguCfg()
    adapter = BooguAdapter()
    adapter.inject_cfg_pad_forward(model)
    adapter.inject_cfg_pad_forward(model)
    ctx = torch.ones(1, 6, 4)
    ctx[:, 4:] = 0.0
    rec = _cfg_call(model, ctx)
    assert rec["context"].shape[1] == 4 and rec["num_tokens"] == 4


def _reference_position_ids(cap_len, img_h, img_w):
    """The position ids one unpadded single-GPU render builds for one prompt.

    Transcribed from the rope embedder ComfyUI ships for this family
    (comfy/ldm/omnigen/omnigen2.py, OmniGen2RotaryPosEmbed.forward): the
    caption numbers 0..cap_len-1 on all three axes, then, with no reference
    image, every image token carries the caption length on axis 0 and its own
    row/col on axes 1 and 2.
    """
    caption = torch.arange(cap_len).view(cap_len, 1).repeat(1, 3)
    rows = torch.arange(img_h).view(img_h, 1).repeat(1, img_w).flatten()
    cols = torch.arange(img_w).view(1, img_w).repeat(img_h, 1).flatten()
    image = torch.stack([torch.full_like(rows, cap_len), rows, cols], dim=1)
    return caption.float(), image.float()


class _RopeRefBoogu:
    """Fake boogu whose rope_embedder follows the shipped per-row rule.

    It returns position ids in place of frequencies so a test can compare the
    numbers the offset is built from. Everything else is the identity,
    so the assembled joint rope is exactly what the forward hands the blocks.
    """

    def __init__(self, img_h=2, img_w=2):
        self.patch_size = 2
        self.img_h, self.img_w = img_h, img_w
        self.img_len = img_h * img_w
        self.seen = []
        self.context_refiner = [_RecBlock("context", self.seen)]
        self.double_stream_layers = [_RecBlock("double", self.seen)]
        self.single_stream_layers = [_RecBlock("single", self.seen)]

    def time_caption_embed(self, timestep, context, dtype):
        return torch.zeros(context.shape[0], 1), context

    def flat_and_pad_to_seq(self, hidden_states, ref):
        batch = hidden_states.shape[0]
        noise = torch.zeros(batch, self.img_len, 4)
        return (noise, None, None, None, [[0]] * batch, [self.img_len] * batch,
                [None] * batch, [(self.img_h * 2, self.img_w * 2)] * batch)

    def img_patch_embed_and_refine(self, hidden_states, ref, *args, **kwargs):
        return hidden_states

    def rope_embedder(self, batch, encoder_seq_len, cap_lens, l_ref, l_img,
                      ref_sizes, img_sizes, device):
        self.seen.append(("rope", (batch, encoder_seq_len, list(cap_lens)), {}))
        caption = torch.zeros(batch, encoder_seq_len, 3)
        image = torch.zeros(batch, self.img_len, 3)
        for row, cap_len in enumerate(cap_lens):
            cap_ids, img_ids = _reference_position_ids(cap_len, self.img_h, self.img_w)
            caption[row, :cap_len] = cap_ids
            image[row] = img_ids
        return (caption, torch.zeros(batch, 0, 3), image, None, list(cap_lens), None)

    def norm_out(self, hidden_states, temb):
        return hidden_states


def _first_call(seen, kind):
    """The (args, kwargs) of the first recorded call of one stage."""
    return next((args, kwargs) for name, args, kwargs in seen if name == kind)


def _ragged_cfg_call(comfy_stubs_applied, cap_lens, width):
    model = _RopeRefBoogu()
    BooguAdapter().inject_cfg_pad_forward(model)
    batch = len(cap_lens)
    context = torch.zeros(batch, width, 4)
    mask = torch.zeros(batch, width)
    for row, cap_len in enumerate(cap_lens):
        context[row, :cap_len] = 1.0
        mask[row, :cap_len] = 1.0
    model.forward(torch.zeros(batch, 1, 4, 4), torch.zeros(batch), context, width,
                  attention_mask=mask, transformer_options={})
    return model


def test_per_row_offset_matches_the_single_gpu_ids_for_every_row(comfy_stubs):
    # In an unequal batch, each row's image position ids are the ones that
    # row's own unpadded single-GPU render would build.
    cap_lens = [3, 5]
    model = _ragged_cfg_call(comfy_stubs, cap_lens, width=5)

    asked = [args for kind, args, _kw in model.seen if kind == "rope"]
    assert asked == [(2, 5, [3, 5])]          # per-row lengths, not one num_tokens

    double_args, double_kwargs = _first_call(model.seen, "double")
    rope = double_args[2]
    assert rope.shape == (2, 5 + model.img_len, 3)

    for row, cap_len in enumerate(cap_lens):
        cap_ids, img_ids = _reference_position_ids(cap_len, model.img_h, model.img_w)
        assert torch.equal(rope[row, :cap_len], cap_ids)          # caption numbering
        assert torch.equal(rope[row, 5:], img_ids)                # image offset by cap_len
        assert torch.equal(rope[row, cap_len:5], torch.zeros(5 - cap_len, 3))

    # The shorter row's image tokens carry 3 on axis 0, the longer row's carry 5:
    # one padded num_tokens could not have expressed both.
    assert rope[0, 5, 0].item() == 3 and rope[1, 5, 0].item() == 5
    assert double_kwargs["joint_attention_mask"] is not None


def test_ragged_batch_masks_the_pad_rows_out_of_every_text_key(comfy_stubs):
    model = _ragged_cfg_call(comfy_stubs, [3, 5], width=5)
    drop = torch.finfo(torch.float32).min

    text_bias = _first_call(model.seen, "context")[0][1]
    assert text_bias.shape == (2, 1, 1, 5)                  # text keys only
    assert torch.equal(text_bias[0, 0, 0], torch.tensor([0.0, 0.0, 0.0, drop, drop]))
    assert torch.equal(text_bias[1, 0, 0], torch.zeros(5))

    joint = _first_call(model.seen, "double")[1]
    single = _first_call(model.seen, "single")[0]
    for bias in (joint["joint_attention_mask"], single[1]):
        assert bias.shape == (2, 1, 1, 5 + model.img_len)
        assert torch.equal(bias[0, 0, 0, 3:5], torch.tensor([drop, drop]))
        assert torch.equal(bias[0, 0, 0, 5:], torch.zeros(model.img_len))
    assert joint["img_attention_mask"] is None               # image self-attn is maskless


def test_ragged_batch_still_negates_and_keeps_the_image_tail(comfy_stubs):
    model = _ragged_cfg_call(comfy_stubs, [3, 5], width=5)
    # The single stage sees [text(5) ; image], and the head slices the image tail.
    single = _first_call(model.seen, "single")[0]
    assert single[0].shape[1] == 5 + model.img_len


@pytest.fixture
def fake_model_base(monkeypatch):
    """Fake comfy.model_base in which Boogu subclasses Omnigen2, as upstream does."""
    return install_fake_model_base(monkeypatch, {"Omnigen2": None, "Boogu": "Omnigen2"})


# tests/test_adapter_matches.py tables the exact-type accept, the typed reject
# on an unvetted Boogu subclass and the generic foreign-model decline. The two
# cases below pin the real Boogu/Omnigen2 subclass relationship.
def test_matches_ignores_omnigen2_parent(fake_model_base):
    # A plain Omnigen2 is the parent of Boogu: BooguAdapter's isinstance-Boogu
    # walk never matches it, so it declines without raising (the sibling
    # Omnigen2Adapter claims it).
    assert BooguAdapter().matches(fake_model_base.Omnigen2()) is False


def test_registry_order_omnigen2_raises_on_boogu(fake_model_base):
    # The sibling Omnigen2Adapter raises on a Boogu instance (Boogu passes its
    # isinstance-Omnigen2 walk but fails exact-type), so adapters/__init__.py
    # must list BooguAdapter before Omnigen2Adapter, or Boogu never reaches its
    # own adapter.
    from dgx_monarch.adapters.omnigen2 import Omnigen2Adapter

    with pytest.raises(UnsupportedModelError, match="omnigen2 variant Boogu"):
        Omnigen2Adapter().matches(fake_model_base.Boogu())


_BOOGU_KEYS = [
    "model.diffusion_model.time_caption_embed.timestep_embedder.linear_1.bias",
    "model.diffusion_model.x_embedder.weight",
    "model.diffusion_model.ref_image_patch_embedder.weight",
    "model.diffusion_model.noise_refiner.0.attn.to_q.weight",
    "model.diffusion_model.ref_image_refiner.0.attn.to_q.weight",
    "model.diffusion_model.context_refiner.0.attn.to_q.weight",
    "model.diffusion_model.double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
    "model.diffusion_model.double_stream_layers.0.img_self_attn.to_q.weight",
    "model.diffusion_model.single_stream_layers.0.attn.to_q.weight",
    "model.diffusion_model.norm_out.linear_1.weight",
]
_OMNIGEN2_KEYS = [
    "model.diffusion_model.time_caption_embed.timestep_embedder.linear_1.bias",
    "model.diffusion_model.x_embedder.weight",
    "model.diffusion_model.noise_refiner.0.attn.to_q.weight",
    "model.diffusion_model.ref_image_refiner.0.attn.to_q.weight",
    "model.diffusion_model.context_refiner.0.attn.to_q.weight",
    "model.diffusion_model.layers.0.attn.to_q.weight",
    "model.diffusion_model.norm_out.linear_1.weight",
]

# A copy of the boogu row in detect._SIGNATURES, the dual-stream stage keys
# unique to Boogu. Change both together.
_BOOGU_SIGNATURE = ("double_stream_layers.", "img_instruct_attn.")


def test_proposed_boogu_signature_matches_boogu_excludes_omnigen2():
    from dgx_monarch.adapters.detect import _has_segment, _strip_prefix

    boogu = [_strip_prefix(k) for k in _BOOGU_KEYS]
    og2 = [_strip_prefix(k) for k in _OMNIGEN2_KEYS]
    assert all(_has_segment(boogu, n) for n in _BOOGU_SIGNATURE)        # matches boogu
    assert not all(_has_segment(og2, n) for n in _BOOGU_SIGNATURE)      # excludes omnigen2
    # The omnigen2 row's `layers.` needle never matches boogu's
    # double_stream_layers. or single_stream_layers. keys.
    assert not _has_segment(boogu, "layers.")


def test_current_detect_never_mislabels_boogu():
    # The boogu key set never detects as another family.
    from dgx_monarch.adapters.detect import detect_family_from_keys

    assert detect_family_from_keys(_BOOGU_KEYS) in ("unknown", "boogu")
