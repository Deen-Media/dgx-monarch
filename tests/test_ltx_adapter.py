"""LTXAdapter on CPU: registry dispatch, the pe shard, the LTXV and LTXAV
injection surface, the STG passthrough, timestep shard bookkeeping, the
attention-override rule, guide bias under Ulysses and divisibility pads. No
distributed init and no hardware; a full forward needs real xfuser and comfy.

LTX declares two comfy model_base classes, LTXV (video only) and LTXAV (audio
and video), and neither subclasses the other, so its matches() loops over a
tuple of exact types, as Kandinsky5 and Hunyuan do, instead of FluxAdapter's
single exact_model_base attribute. Registry dispatch runs through the real
package registry (dgx_monarch.adapters.ADAPTERS and get_adapter) over a fake
comfy.model_base holding only LTXV, LTXAV and LTXVExotic. That is safe for
every other adapter: each one's exact-type raise comes after its own
isinstance check (base.Adapter.matches), which returns False, never raises,
when its class name is absent from the fake module.
"""
import sys
import types

import pytest
import torch

from dgx_monarch.adapters import ADAPTERS, attention_patches, base, get_adapter, ltx
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.ltx import (
    _LTX_SUPPORTED_EXACT,
    LTXAdapter,
    _make_usp_cross_attention,
    _shard_pe,
)
from fake_model_base_helpers import install_fake_model_base


@pytest.fixture
def fake_model_base(monkeypatch):
    """A fake comfy.model_base holding only LTXV, LTXAV and an unvetted LTXV
    subclass; the module docstring says why the whole ADAPTERS registry can
    run on it."""
    return install_fake_model_base(
        monkeypatch, {"LTXV": None, "LTXAV": None, "LTXVExotic": "LTXV"}
    )


def test_registry_dispatches_ltxv_to_ltx_adapter(fake_model_base):
    assert type(get_adapter(fake_model_base.LTXV())) is LTXAdapter


def test_registry_dispatches_ltxav_to_ltx_adapter(fake_model_base):
    assert type(get_adapter(fake_model_base.LTXAV())) is LTXAdapter


def test_registry_surfaces_typed_reject_for_unvetted_subclass(fake_model_base):
    # get_adapter must not swallow LTXAdapter's typed reject into its own
    # generic "no adapter" error. The specific reason should surface.
    with pytest.raises(UnsupportedModelError, match="ltx variant"):
        get_adapter(fake_model_base.LTXVExotic())


def test_ltx_adapter_registered_exactly_once():
    assert sum(isinstance(a, LTXAdapter) for a in ADAPTERS) == 1


# The matches() trio (exact-type accept for LTXV and LTXAV, typed reject on
# an unvetted subclass, foreign-model decline) lives in
# tests/test_adapter_matches.py.
def test_supported_exact_matches_declared_model_base_classes():
    assert set(_LTX_SUPPORTED_EXACT) == set(LTXAdapter.model_base_classes)


def test_shard_pe_rotation_matrix_form_shards_dim1(sp):
    # comfy #15056 payload: (B, T, heads, head_dim/2, 2, 2), sequence on dim 1.
    matrix = torch.arange(2 * 4 * 3 * 5 * 2 * 2).reshape(2, 4, 3, 5, 2, 2).float()
    sp(2, 1)
    matrix_local, mode = _shard_pe((matrix, True))
    assert mode is True
    expected, _ = base.shard_seq(matrix, dim=1)
    assert torch.equal(matrix_local, expected)
    assert matrix_local.shape[-2:] == (2, 2)


def test_shard_pe_unknown_arity_refuses_typed(sp):
    # A future upstream re-shape must refuse typed, never die mid-forward: on
    # 2026-08-06 an untyped death here abandoned the sample and wedged the
    # fleet until a recycle.
    sp(2, 0)
    with pytest.raises(UnsupportedModelError, match="rope payload"):
        _shard_pe((torch.zeros(2, 4, 5),))


def test_shard_pe_split_layout_shards_sequence_dim(sp):
    # The (B, H, T, d) split layout shards on dim 2, the sequence axis.
    cos = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float()
    sin = -cos.clone()
    sp(2, 1)
    cos_local, sin_local, mode = _shard_pe((cos, sin, "split"))
    assert mode == "split"
    expected, _ = base.shard_seq(cos, dim=2)
    assert torch.equal(cos_local, expected)
    assert torch.equal(sin_local, -expected)


def test_shard_pe_interleaved_layout_shards_dim1(sp):
    # The (B, T, d) interleaved layout shards on dim 1.
    cos = torch.arange(2 * 4 * 5).reshape(2, 4, 5).float()
    sin = -cos.clone()
    sp(2, 0)
    cos_local, sin_local, mode = _shard_pe((cos, sin, "interleaved"))
    assert mode == "interleaved"
    expected, _ = base.shard_seq(cos, dim=1)
    assert torch.equal(cos_local, expected)
    assert torch.equal(sin_local, -expected)


def test_shard_pe_preserves_split_mode_marker(sp):
    # split_mode passes through unchanged; only cos and sin shard.
    cos = torch.zeros(1, 1, 4, 2)
    sp(2, 0)
    for mode in ("split", "interleaved", "whatever-marker"):
        _, _, out_mode = _shard_pe((cos, cos, mode))
        assert out_mode == mode


_STOCK = object()


class _Mod:
    def __init__(self):
        self.forward = _STOCK


class _LTXVBlock:
    def __init__(self):
        self.attn1 = _Mod()
        self.attn2 = _Mod()  # Text cross-attention stays stock (not USP-routed).


class _FakeLTXV:
    # The adapter fetches the stock block loop from the class
    # (`type(diffusion_model)._process_transformer_blocks`), so the sentinel
    # must live at class scope.
    _process_transformer_blocks = _STOCK

    def __init__(self, n=2):
        self.transformer_blocks = [_LTXVBlock() for _ in range(n)]


class _LTXAVBlock(_LTXVBlock):
    def __init__(self):
        super().__init__()
        self.audio_attn1 = _Mod()
        self.audio_attn2 = _Mod()  # Audio text cross-attention stays stock.
        self.audio_to_video_attn = _Mod()
        self.video_to_audio_attn = _Mod()


class _FakeLTXAV:
    _process_transformer_blocks = _STOCK

    def __init__(self, n=2):
        self.transformer_blocks = [_LTXAVBlock() for _ in range(n)]
        self.audio_inner_dim = 32  # the hasattr() marker inject_usp dispatches on


def test_inject_usp_dispatches_ltxv_self_attention_only():
    model = _FakeLTXV()
    LTXAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))
    for block in model.transformer_blocks:
        assert block.attn1.forward is not _STOCK  # USP-routed self-attention
        assert block.attn2.forward is _STOCK       # text cross-attention: local q, full text k/v
    assert model._process_transformer_blocks is not _STOCK


def test_inject_usp_dispatches_ltxav_on_audio_inner_dim():
    model = _FakeLTXAV()
    LTXAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))
    for block in model.transformer_blocks:
        assert block.attn1.forward is not _STOCK
        assert block.attn2.forward is _STOCK                # text cross (video side) stays stock
        assert block.audio_attn1.forward is not _STOCK
        assert block.audio_attn2.forward is _STOCK           # text cross (audio side) stays stock
        assert block.audio_to_video_attn.forward is not _STOCK
        assert block.video_to_audio_attn.forward is not _STOCK
    assert model._process_transformer_blocks is not _STOCK


class _FakeCrossAttention:
    """The CrossAttention surface the bound forward touches. The norms record,
    so a test can pin that the STG branch exits ahead of them."""

    def __init__(self, dim=8, heads=2, gated=True):
        torch.manual_seed(11)
        self.heads = heads
        self.dim_head = dim // heads
        self.normed = []
        self.to_q = torch.nn.Linear(dim, dim)
        self.to_k = torch.nn.Linear(dim, dim)
        self.to_v = torch.nn.Linear(dim, dim)
        self.to_out = torch.nn.Linear(dim, dim)
        self.to_gate_logits = torch.nn.Linear(dim, heads) if gated else None

    def q_norm(self, tensor):
        self.normed.append("q")
        return tensor

    def k_norm(self, tensor):
        self.normed.append("k")
        return tensor


def _stub_rope(monkeypatch, apply_rotary_emb):
    """Stand in for comfy's rope helper at the import the forward makes."""
    model = types.ModuleType("comfy.ldm.lightricks.model")
    model.apply_rotary_emb = apply_rotary_emb
    lightricks = types.ModuleType("comfy.ldm.lightricks")
    lightricks.model = model
    ldm = types.ModuleType("comfy.ldm")
    ldm.lightricks = lightricks
    comfy_mod = types.ModuleType("comfy")
    comfy_mod.ldm = ldm
    for name, module in (("comfy", comfy_mod), ("comfy.ldm", ldm),
                         ("comfy.ldm.lightricks", lightricks),
                         ("comfy.ldm.lightricks.model", model)):
        monkeypatch.setitem(sys.modules, name, module)


def _explode(*args, **kwargs):
    raise AssertionError("the STG passthrough reached attention or rope")


def _gated_tail(module, out, x):
    """The per-head gate and projection both branches share."""
    b, t_len, _ = out.shape
    gates = 2.0 * torch.sigmoid(module.to_gate_logits(x))
    gated = (out.view(b, t_len, module.heads, module.dim_head)
             * gates.unsqueeze(-1)).view(b, t_len, -1)
    return module.to_out(gated)


def test_stg_flag_returns_the_gated_value_projection_with_no_attention(monkeypatch):
    # STG (LTX 2.5 spatio-temporal guidance) perturbs one cond pass by
    # degrading flagged self-attention to out = v.
    # A rank that ran real attention instead would return the unperturbed
    # prediction, and the node would guide against an identical tensor.
    _stub_rope(monkeypatch, _explode)
    module = _FakeCrossAttention()
    forward = _make_usp_cross_attention(_explode)
    x = torch.randn(1, 6, 8)

    out = forward(module, x, pe=(torch.zeros(1, 6, 2, 2, 2, 2), True),
                  transformer_options={"stg_skip_self_attn": True})

    # No collective: the value projection is per token, so the local shard is
    # the answer and the USP callable is never reached.
    assert torch.equal(out, _gated_tail(module, module.to_v(x), x))
    assert module.normed == []


def test_stg_passthrough_without_the_per_head_gate_is_the_bare_projection(monkeypatch):
    _stub_rope(monkeypatch, _explode)
    module = _FakeCrossAttention(gated=False)
    forward = _make_usp_cross_attention(_explode)
    x = torch.randn(1, 6, 8)

    out = forward(module, x, transformer_options={"stg_skip_self_attn": True})

    assert torch.equal(out, module.to_out(module.to_v(x)))


def test_stg_flag_leaves_context_bearing_attention_on_the_usp_path(monkeypatch):
    # The per-block flag reaches every attention in the block; only the two
    # self-attentions short circuit, so the audio-video cross attentions must
    # still take the USP path.
    _stub_rope(monkeypatch, lambda tensor, pe: tensor)
    module = _FakeCrossAttention()
    seen = []

    def attn(q, k, v, heads, query_bias_groups=None, drop_rows=None,
             kv_drop_rows=None):
        seen.append((heads, query_bias_groups, drop_rows, kv_drop_rows))
        return v

    forward = _make_usp_cross_attention(attn)
    x = torch.randn(1, 6, 8)
    forward(module, x, context=torch.randn(1, 6, 8), pe=(torch.zeros(1, 6, 2, 2, 2, 2), True),
            transformer_options={"stg_skip_self_attn": True})

    assert seen == [(module.heads, None, None, None)]
    assert module.normed == ["q", "k"]


def test_without_the_stg_flag_self_attention_keeps_norms_rope_and_usp(monkeypatch):
    roped = []

    def apply_rotary_emb(tensor, pe):
        roped.append(pe)
        return tensor

    _stub_rope(monkeypatch, apply_rotary_emb)
    module = _FakeCrossAttention()
    seen = []

    def attn(q, k, v, heads, query_bias_groups=None, drop_rows=None,
             kv_drop_rows=None):
        seen.append((heads, query_bias_groups, drop_rows, kv_drop_rows))
        return v

    forward = _make_usp_cross_attention(attn)
    x = torch.randn(1, 6, 8)

    out = forward(module, x, pe=(torch.zeros(1, 6, 2, 2, 2, 2), True), transformer_options={})

    assert seen == [(module.heads, None, None, None)]
    assert module.normed == ["q", "k"]
    assert len(roped) == 2
    assert torch.equal(out, _gated_tail(module, module.to_v(x), x))


def test_a_raw_mask_at_the_patched_attention_still_refuses(monkeypatch):
    """This backstop one level under the block guard must stay.

    Guide attenuation arrives as query groups, so anything else in the mask
    channel is a mask class this adapter has never read, and the sharded kernel
    would drop it and raise nothing.
    """
    _stub_rope(monkeypatch, lambda tensor, pe: tensor)
    forward = _make_usp_cross_attention(_explode)

    with pytest.raises(UnsupportedModelError) as raised:
        forward(_FakeCrossAttention(), torch.randn(1, 6, 8),
                mask=torch.zeros(1, 1, 6, 6),
                pe=(torch.zeros(1, 6, 2, 2, 2, 2), True), transformer_options={})

    message = str(raised.value)
    assert message.startswith("[dgxm:P]")
    assert "single" in message


class _FakeCompressedTimestep:
    """comfy's CompressedTimestep, copied so the CPU test can build the exact
    shapes `_prepare_timestep` produces (comfy/ldm/lightricks/av_model.py).
    The fallback arm is the one that matters: a stream that is neither per-token
    nor a whole number of frames keeps patches_per_frame 1 over its rows."""

    def __init__(self, tensor, patches_per_frame, per_frame=False):
        self.batch_size, n, self.feature_dim = tensor.shape
        if per_frame:
            self.patches_per_frame = patches_per_frame
            self.num_frames = n
            self.data = tensor
        elif (patches_per_frame is not None and n >= patches_per_frame
                and n % patches_per_frame == 0):
            self.patches_per_frame = patches_per_frame
            self.num_frames = n // patches_per_frame
            self.data = tensor.view(
                self.batch_size, self.num_frames, patches_per_frame, self.feature_dim
            )[:, :, 0, :].contiguous()
        else:
            self.patches_per_frame = 1
            self.num_frames = n
            self.data = tensor

    def expand(self):
        if self.patches_per_frame == 1:
            return self.data
        return self.data.unsqueeze(2).expand(
            self.batch_size, self.num_frames, self.patches_per_frame, self.feature_dim
        ).reshape(self.batch_size, -1, self.feature_dim)


class _FakeGuideAttentionMask:
    """comfy's GuideAttentionMask, copied (comfy/ldm/lightricks/model.py).

    The adapter reads all four slots, so the copy is the executable statement of
    what it depends on: guides at [guide_start, guide_start + tracked_count),
    weights already in log space, and both rectangles head-independent.
    """

    __slots__ = ("guide_start", "noisy_mask", "tracked_count", "tracked_mask")

    def __init__(self, total_tokens, guide_start, tracked_count, tracked_weights):
        dtype = tracked_weights.dtype
        finfo = torch.finfo(dtype)
        positive = tracked_weights > 0
        log_w = torch.full_like(tracked_weights, finfo.min)
        log_w[positive] = torch.log(tracked_weights[positive].clamp(min=finfo.tiny))
        self.guide_start = guide_start
        self.tracked_count = tracked_count
        self.noisy_mask = torch.zeros((1, 1, 1, total_tokens), dtype=dtype)
        self.noisy_mask[:, :, :, guide_start:guide_start + tracked_count] = \
            log_w.view(1, 1, 1, -1)
        self.tracked_mask = torch.zeros((1, 1, tracked_count, total_tokens), dtype=dtype)
        self.tracked_mask[:, :, :, :guide_start] = log_w.view(1, 1, -1, 1)


def _stub_av_model(monkeypatch):
    """Stand in for the comfy imports the sharded forward makes."""
    av_model = types.ModuleType("comfy.ldm.lightricks.av_model")
    av_model.CompressedTimestep = _FakeCompressedTimestep
    model = types.ModuleType("comfy.ldm.lightricks.model")
    model.GuideAttentionMask = _FakeGuideAttentionMask
    model.apply_rotary_emb = lambda tensor, pe: tensor
    lightricks = types.ModuleType("comfy.ldm.lightricks")
    lightricks.av_model = av_model
    lightricks.model = model
    ldm = types.ModuleType("comfy.ldm")
    ldm.lightricks = lightricks
    comfy_mod = types.ModuleType("comfy")
    comfy_mod.ldm = ldm
    for name, module in (("comfy", comfy_mod), ("comfy.ldm", ldm),
                         ("comfy.ldm.lightricks", lightricks),
                         ("comfy.ldm.lightricks.model", model),
                         ("comfy.ldm.lightricks.av_model", av_model)):
        monkeypatch.setitem(sys.modules, name, module)


def _stub_ring(monkeypatch, degree, world=2):
    """Report the parallel degrees without bringing a process group up.

    Keep ``world`` equal to the ``sp`` fixture's degree, so a reader that asks
    xfuser for the sequence-parallel size agrees with ``base.sp_world``, which
    the fixture patches.
    """
    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.get_ring_parallel_world_size = lambda: degree
    distributed.get_sequence_parallel_world_size = lambda: world
    core = types.ModuleType("xfuser.core")
    core.distributed = distributed
    xfuser = types.ModuleType("xfuser")
    xfuser.core = core
    for name, module in (("xfuser", xfuser), ("xfuser.core", core),
                         ("xfuser.core.distributed", distributed)):
        monkeypatch.setitem(sys.modules, name, module)


class _RecordingLTXAV(_FakeLTXAV):
    """_FakeLTXAV with a stock block loop that records what it is handed."""

    seen: dict = {}

    def _process_transformer_blocks(self, x, context, attention_mask, timestep, pe,
                                    transformer_options=None, self_attention_mask=None,
                                    **kwargs):
        type(self).seen = {"x": x, "timestep": timestep, "pe": pe,
                           "self_attention_mask": self_attention_mask,
                           "transformer_options": transformer_options}
        return [x[0], x[1]]


_V_TOKENS, _A_TOKENS, _FEAT = 8, 4, 6


def _drive_ltxav(monkeypatch, sp, rank, v_ts, a_ts, cross_ts, *,
                 guide=None, video_tokens=_V_TOKENS, audio_tokens=_A_TOKENS):
    """Inject, then run one block loop at `rank` and return the stock arguments."""
    _stub_av_model(monkeypatch)
    monkeypatch.setattr(ltx, "sp_gather", lambda t, orig_len, dim=1: t)
    sp(2, rank)
    model = _RecordingLTXAV()
    LTXAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))

    def rope(tokens):
        return (torch.zeros(1, tokens, 1, 1, 2, 2), True)

    model._process_transformer_blocks(
        [torch.randn(1, video_tokens, _FEAT), torch.randn(1, audio_tokens, _FEAT)],
        [None, None], None,
        [v_ts, a_ts, cross_ts, torch.zeros(1, 1, _FEAT), torch.zeros(1, 1, _FEAT)],
        [(rope(video_tokens), rope(video_tokens)),
         (rope(audio_tokens), rope(audio_tokens))],
        transformer_options={}, self_attention_mask=guide,
    )
    return _RecordingLTXAV.seen["timestep"]


def _values(ts):
    return ts.data if isinstance(ts, _FakeCompressedTimestep) else ts


@pytest.mark.parametrize("rank", [0, 1])
def test_ltxav_broadcast_video_timestep_survives_the_shard(monkeypatch, sp, rank):
    # A scalar sigma (t2v: no denoise mask) makes comfy build the video
    # timestep, the a2v scale/shift and the a2v gate as one broadcast row that
    # every token reads. Sharding that row pads it up to the SP degree, so rank 1
    # would enter the block loop with an all-zero AdaLN embedding: the video-only
    # step at the sp=2 token boundary measured on 2026-08-12.
    row = torch.arange(1.0, _FEAT + 1).reshape(1, 1, _FEAT)
    v_ts = _FakeCompressedTimestep(row.clone(), _V_TOKENS // 2)
    cross_in = [torch.full((1, 1, _FEAT), 7.0),
                _FakeCompressedTimestep(row.clone() * 2, _V_TOKENS // 2),
                _FakeCompressedTimestep(row.clone() * 3, _V_TOKENS // 2),
                torch.full((1, 1, _FEAT), 9.0)]
    assert v_ts.patches_per_frame == 1 and v_ts.num_frames == 1

    out = _drive_ltxav(monkeypatch, sp, rank, v_ts, torch.full((1, 1, _FEAT), 5.0), cross_in)

    assert torch.equal(_values(out[0]), row)
    for got, expected in zip(out[2], cross_in, strict=True):
        assert torch.equal(_values(got), _values(expected))
        assert _values(got).abs().sum() > 0


@pytest.mark.parametrize("rank", [0, 1])
def test_ltxav_per_token_video_timestep_still_shards(monkeypatch, sp, rank):
    # The i2v/denoise-mask form: one row per token, compressed per frame. Every
    # rank must keep its own contiguous half.
    # Frame-constant per-token rows: what comfy compresses losslessly (it keeps
    # the first patch of each frame), so token i carries the value of frame i//2.
    per_token = (torch.arange(_V_TOKENS, dtype=torch.float32) // 2).reshape(1, _V_TOKENS, 1).repeat(1, 1, _FEAT)
    v_ts = _FakeCompressedTimestep(per_token.clone(), 2)
    a_per_token = torch.arange(_A_TOKENS, dtype=torch.float32).reshape(1, _A_TOKENS, 1).repeat(1, 1, _FEAT)
    assert v_ts.patches_per_frame == 2 and v_ts.num_frames == _V_TOKENS // 2

    out = _drive_ltxav(monkeypatch, sp, rank, v_ts, a_per_token.clone(), [])

    half = _V_TOKENS // 2
    assert torch.equal(_values(out[0]), per_token[:, rank * half:(rank + 1) * half, :])
    a_half = _A_TOKENS // 2
    assert torch.equal(out[1], a_per_token[:, rank * a_half:(rank + 1) * a_half, :])


@pytest.mark.parametrize("fake", [_FakeLTXV, _FakeLTXAV])
def test_foreign_attention_override_refuses_before_any_sharding(sp, fake):
    # Comfy exposes the same option key to graphs. Whichever writer loses, it
    # loses without a word, so the refusal lands before the first collective
    # and on input every rank sees identically.
    sp(2, 0)
    model = fake()
    LTXAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))
    options = {"optimized_attention_override": lambda *args, **kwargs: None}

    with pytest.raises(UnsupportedModelError) as excinfo:
        model._process_transformer_blocks(None, None, None, None, None,
                                          transformer_options=options)

    assert str(excinfo.value).startswith("[dgxm:P]")
    assert "optimized_attention_override" in str(excinfo.value)


def test_our_own_attention_override_is_not_a_foreign_writer():
    # The rule keys on the mark usp_options stamps, so a family that threads
    # its own override never refuses on it.
    options = base.usp_options({}, lambda *args, **kwargs: None)

    attention_patches.assert_no_foreign_attention_override(options, "ltx")
    attention_patches.assert_no_foreign_attention_override({}, "ltx")


def test_adapter_contract_declared():
    adapter = LTXAdapter()
    assert adapter.family == "ltx"
    assert adapter.model_base_classes == ("LTXV", "LTXAV")
    assert adapter.cfg_cond_padding == "none"


def _guide_mask(total_tokens, guide_tokens, strength):
    """comfy's mask for one guide of ``guide_tokens`` rows at the tail."""
    weights = torch.full((guide_tokens,), strength)
    return _FakeGuideAttentionMask(total_tokens, total_tokens - guide_tokens,
                                   guide_tokens, weights)


def test_a_strength_one_guide_never_reaches_the_bias_path(monkeypatch, sp):
    """comfy returns None at strength 1.0, so nothing about that render moves.

    The mask channel stays empty and no bias reaches the block loop, which keeps
    the 2026-08-12 guide legs, rendered at strength 1.0, valid (docs/VALIDATION.md,
    "Image conditioning: image to video and first and last frame").
    """
    _drive_ltxav(monkeypatch, sp, 0, torch.zeros(1, _V_TOKENS, _FEAT),
                 torch.zeros(1, _A_TOKENS, _FEAT), None, guide=None)

    assert _RecordingLTXAV.seen["self_attention_mask"] is None


@pytest.mark.parametrize("rank", [0, 1])
def test_an_attenuated_guide_becomes_query_groups_over_the_full_sequence(
        monkeypatch, sp, rank):
    """The bias spans the whole token axis on every rank, not the local chunk.

    After the Ulysses head scatter a rank holds all 8 video tokens for half the
    heads, so the group boundaries are absolute positions in that sequence and
    come out identical on both ranks.
    """
    _stub_ring(monkeypatch, 1)
    guide = _guide_mask(_V_TOKENS, 2, 0.7)

    _drive_ltxav(monkeypatch, sp, rank, torch.zeros(1, _V_TOKENS, _FEAT),
                 torch.zeros(1, _A_TOKENS, _FEAT), None, guide=guide)

    groups = _RecordingLTXAV.seen["self_attention_mask"].groups
    assert [(start, end) for start, end, _ in groups] == [(0, 6), (6, 8)]
    assert torch.equal(groups[0][2], guide.noisy_mask)
    assert torch.equal(groups[1][2], guide.tracked_mask)


def test_a_padded_video_stream_biases_its_own_rows_and_no_synthetic_one(
        monkeypatch, sp):
    """9 tokens over 2 ranks pads to 10, and the groups still end at 9.

    The pad row leaves the kernel before any group is applied, so the partition
    covers the rows comfy built the mask over and nothing else. Building it over
    10 instead would hand the pad key a zero bias column, which is no bias at
    all: every biased query would read a synthetic key at full weight.
    """
    _stub_ring(monkeypatch, 1)
    guide = _guide_mask(9, 2, 0.7)

    _drive_ltxav(monkeypatch, sp, 0, torch.zeros(1, 9, _FEAT),
                 torch.zeros(1, _A_TOKENS, _FEAT), None, guide=guide,
                 video_tokens=9)

    groups = _RecordingLTXAV.seen["self_attention_mask"].groups
    assert [(start, end) for start, end, _ in groups] == [(0, 7), (7, 9)]


@pytest.mark.parametrize("ring", [2, 4])
def test_ring_still_refuses_an_attenuated_guide_naming_ulysses(monkeypatch, sp, ring):
    """No ring rank holds the key axis the bias spans, so ring refuses.

    The refusal names what does work, and lands before the first shard so both
    ranks refuse on the same request and the lease retires consumed.
    """
    _stub_ring(monkeypatch, ring)

    with pytest.raises(UnsupportedModelError) as raised:
        _drive_ltxav(monkeypatch, sp, 0, torch.zeros(1, _V_TOKENS, _FEAT),
                     torch.zeros(1, _A_TOKENS, _FEAT), None,
                     guide=_guide_mask(_V_TOKENS, 2, 0.7))

    message = str(raised.value)
    assert message.startswith("[dgxm:P]")
    assert "uly" in message and "single" in message


def test_an_unrecognized_mask_object_refuses_rather_than_being_duck_typed(
        monkeypatch, sp):
    """A future comfy mask would carry different semantics under those names."""
    _stub_ring(monkeypatch, 1)

    with pytest.raises(UnsupportedModelError) as raised:
        _drive_ltxav(monkeypatch, sp, 0, torch.zeros(1, _V_TOKENS, _FEAT),
                     torch.zeros(1, _A_TOKENS, _FEAT), None,
                     guide=torch.zeros(1, 1, _V_TOKENS, _V_TOKENS))

    assert str(raised.value).startswith("[dgxm:P]")


def test_the_video_only_model_carries_the_same_guide_bias(monkeypatch, sp):
    """LTXV (no audio stream) shares the machinery, over its single stream."""
    _stub_av_model(monkeypatch)
    _stub_ring(monkeypatch, 1)
    monkeypatch.setattr(ltx, "sp_gather", lambda t, orig_len, dim=1: t)
    sp(2, 0)
    seen = {}

    class _RecordingLTXV(_FakeLTXV):
        def _process_transformer_blocks(self, x, context, attention_mask, timestep,
                                        pe, transformer_options=None,
                                        self_attention_mask=None, **kwargs):
            seen["mask"] = self_attention_mask
            return x

    model = _RecordingLTXV()
    LTXAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))
    model._process_transformer_blocks(
        torch.randn(1, _V_TOKENS, _FEAT), None, None, None,
        (torch.zeros(1, _V_TOKENS, 1, 1, 2, 2), True), transformer_options={},
        self_attention_mask=_guide_mask(_V_TOKENS, 2, 0.5))

    assert [(start, end) for start, end, _ in seen["mask"].groups] == [(0, 6), (6, 8)]


# Divisibility pads. Both streams shard, neither divides on its own, and the
# two cross attentions read one stream's queries against the other's keys.

@pytest.mark.parametrize("rank", [0, 1])
def test_both_ltxav_streams_publish_their_own_pad_rows(monkeypatch, sp, rank):
    """9 video rows and 5 audio rows over 2 ranks: each stream pads by one.

    The gathered coordinate of a pad is its own stream's tail, so the two
    differ, and both ranks compute the same pair.
    """
    _stub_ring(monkeypatch, 1)
    _drive_ltxav(monkeypatch, sp, rank, torch.zeros(1, 9, _FEAT),
                 torch.zeros(1, 5, _FEAT), None, video_tokens=9, audio_tokens=5)

    options = _RecordingLTXAV.seen["transformer_options"]
    assert options[ltx._PAD_ROWS_KEY] == {"video": [9], "audio": [5]}


def test_even_ltxav_streams_publish_no_pad_rows(monkeypatch, sp):
    """The no-pad guarantee: nothing to drop means the path does not move."""
    _stub_ring(monkeypatch, 1)
    _drive_ltxav(monkeypatch, sp, 0, torch.zeros(1, _V_TOKENS, _FEAT),
                 torch.zeros(1, _A_TOKENS, _FEAT), None)

    options = _RecordingLTXAV.seen["transformer_options"]
    assert options[ltx._PAD_ROWS_KEY] == {"video": [], "audio": []}


@pytest.mark.parametrize("queries,keys,expected", [
    ("video", None, ([9], None)),
    ("audio", None, ([5], None)),
    ("video", "audio", ([9], [5])),
    ("audio", "video", ([5], [9])),
])
def test_each_role_drops_the_rows_of_the_streams_it_actually_reads(
        monkeypatch, queries, keys, expected):
    """Each attention drops the pad rows of the streams it reads.

    audio_to_video takes video queries against audio keys (comfy av_model.py
    BasicAVTransformerBlock.forward), so handing it one row set would drop a
    real audio key at a video coordinate and leave the synthetic one in the
    kernel. A self-attention sends no key set at all, which keeps the joint
    families' call shape unchanged.
    """
    _stub_rope(monkeypatch, lambda tensor, pe: tensor)
    module = _FakeCrossAttention()
    seen = []

    def attn(q, k, v, heads, query_bias_groups=None, drop_rows=None,
             kv_drop_rows=None):
        seen.append((drop_rows, kv_drop_rows))
        return v

    forward = _make_usp_cross_attention(attn, queries, keys)
    x = torch.randn(1, 6, 8)
    forward(module, x, context=torch.randn(1, 6, 8),
            pe=(torch.zeros(1, 6, 2, 2, 2, 2), True),
            transformer_options={
                ltx._PAD_ROWS_KEY: {"video": [9], "audio": [5]}})

    assert seen == [expected]


@pytest.mark.parametrize("ring", [2, 4])
def test_a_padded_ltx_stream_refuses_on_ring_before_any_collective(
        monkeypatch, sp, ring):
    """Ring has no full-sequence point, so the pads cannot be dropped there.

    Rendering them anyway is the left-edge class measured on 2026-07-10
    (docs/TROUBLESHOOTING.md #21), so it refuses with the shared class-K
    wording and its waiver card.
    """
    _stub_ring(monkeypatch, ring)

    with pytest.raises(UnsupportedModelError, match="ulysses-only"):
        _drive_ltxav(monkeypatch, sp, 0, torch.zeros(1, 9, _FEAT),
                     torch.zeros(1, _A_TOKENS, _FEAT), None, video_tokens=9)


@pytest.mark.parametrize("ring", [2, 4])
def test_an_even_ltx_stream_still_renders_on_ring(monkeypatch, sp, ring):
    """Ring still renders an even stream: the guard covers pads only."""
    _stub_ring(monkeypatch, ring)

    _drive_ltxav(monkeypatch, sp, 0, torch.zeros(1, _V_TOKENS, _FEAT),
                 torch.zeros(1, _A_TOKENS, _FEAT), None)

    assert _RecordingLTXAV.seen["transformer_options"][ltx._PAD_ROWS_KEY] == {
        "video": [], "audio": []}


@pytest.mark.parametrize("queries,keys,expected", [
    ("video", "audio", ([9], [])),
    ("audio", "video", (None, [9])),
    ("video", None, ([9], None)),
    ("audio", None, (None, None)),
])
def test_a_cross_attention_names_its_key_stream_even_when_it_did_not_pad(
        monkeypatch, queries, keys, expected):
    """`None` on the key side means "the keys drop where the queries do", the
    self-attention contract, so only a self-attention sends it.

    A cross attention whose key stream did not pad sends the empty list. Sending
    None there addressed video rows on a shorter audio axis, and on hardware the
    CUDA scatter answered with a device-side assert that killed the worker
    (2026-08-13).
    """
    _stub_rope(monkeypatch, lambda tensor, pe: tensor)
    module = _FakeCrossAttention()
    seen = []

    def attn(q, k, v, heads, query_bias_groups=None, drop_rows=None,
             kv_drop_rows=None):
        seen.append((drop_rows, kv_drop_rows))
        return v

    forward = _make_usp_cross_attention(attn, queries, keys)
    x = torch.randn(1, 6, 8)
    forward(module, x, context=torch.randn(1, 6, 8),
            pe=(torch.zeros(1, 6, 2, 2, 2, 2), True),
            transformer_options={
                ltx._PAD_ROWS_KEY: {"video": [9], "audio": []}})

    assert seen == [expected]
