"""Anima adapter layout invariants (CPU, no distributed init, no comfy import
at module scope).

Anima's denoise forward is the stock cosmos MiniTrainDIT: a 5D (B, T, H, W, D)
grid whose block loop flattens `b (t h w) d` for maskless self-attention. The
port shards the H axis. These tests pin the layout that shard depends on:

  * the 3D RoPE table shards on the same H axis with the same token pad; a
    differing pad (or an unpadded chunk on the flat L axis) rotates later
    ranks' tokens by the wrong position, for T=1 and for the T>1 interleave;
  * the pad unit is one whole H-row (W tokens) and the gather trims it, bit-exact;
  * the attn_op wrapper hands USP the (B, H, S, D) layout and returns (B, S, H*D);
  * the exact-type allowlist keeps the sibling Cosmos-Predict2 model out;
  * the llm_adapter, which runs in extra_conds before the sharded forward, is
    never in the bound set (self-attn is; cross-attn stays stock).
"""
import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters.anima import AnimaAdapter, _make_usp_attn_op, _shard_rope_h
from dgx_monarch.adapters.base import InjectionContext
from fake_model_base_helpers import install_fake_model_base


@pytest.mark.parametrize("T,H,W", [(1, 6, 4), (2, 6, 4), (1, 5, 3), (3, 4, 2)])
def test_rope_shards_identically_to_h_tokens(sp, T, H, W):
    # Encode every token with its global (t h w) flatten index; the RoPE row for
    # that token carries the same index. If the H-shard of x and the H-shard of
    # the RoPE table stay aligned, the two flattened local slices are equal on
    # every rank for T=1 (contiguous) and T>1 (interleaved) alike.
    idx = torch.arange(T * H * W).float()
    x = idx.reshape(1, T, H, W, 1)          # x[0, t, h, w, 0] == flatten index
    rope = idx.reshape(T * H * W, 1)         # rope row i == index i (rest dim = 1)
    for rank in (0, 1):
        sp(2, rank)
        x_local, _ = base.shard_seq(x, dim=2)          # (1, T, H_local, W, 1)
        rope_local = _shard_rope_h(rope, T, H, W)       # (T*H_local*W, 1)
        # The block flattens x_local as `b (t h w) d`; row-major reshape matches.
        x_flat = x_local.reshape(-1)
        # Same pad, chunk and order, so the RoPE rows match the tokens exactly;
        # zero-pad slots are zero in both, so they compare equal too.
        assert rope_local.reshape(-1).shape == x_flat.shape
        assert torch.equal(rope_local.reshape(-1), x_flat)


def test_naive_unpadded_rope_chunk_misaligns_the_later_rank(sp):
    # Chunking the unpadded flat table hands the later rank a table shorter
    # than its token shard (H=5 pads one row). _shard_rope_h pads, then
    # chunks, which keeps them equal.
    T, H, W = 1, 5, 3
    idx = torch.arange(T * H * W).float()
    x = idx.reshape(1, T, H, W, 1)
    rope = idx.reshape(T * H * W, 1)
    sp(2, 1)
    x_local, _ = base.shard_seq(x, dim=2)
    naive = torch.chunk(rope, 2, dim=0)[1]
    assert naive.shape[0] != x_local.reshape(-1).shape[0]
    assert torch.equal(_shard_rope_h(rope, T, H, W).reshape(-1), x_local.reshape(-1))


def test_pad_unit_is_one_h_row_of_w_tokens_and_gather_trims(sp):
    # H=3 at world 2 pads to 4: one extra H-row of W tokens.
    T, H, W, D = 1, 3, 4, 2
    x = torch.randn(1, T, H, W, D)
    chunks = []
    for rank in (0, 1):
        sp(2, rank)
        local, h_orig = base.shard_seq(x, dim=2)
        chunks.append(local)
    assert h_orig == H
    assert chunks[0].shape[2] == 2

    padded = torch.cat(chunks, dim=2)
    assert torch.equal(padded[:, :, H:], torch.zeros(1, T, 1, W, D))
    gathered = padded.narrow(2, 0, h_orig)
    assert gathered.shape[2] == H
    assert torch.equal(gathered, x)


def test_t1_shard_is_a_contiguous_token_slice(sp):
    # For T=1 the `(t h w)` flatten is `(h w)`, so an H-shard is a contiguous
    # run of the flattened sequence.
    T, H, W = 1, 4, 3
    idx = torch.arange(T * H * W).float()
    x = idx.reshape(1, T, H, W, 1)
    sp(2, 0)
    x_local, _ = base.shard_seq(x, dim=2)
    flat = x_local.reshape(-1)
    assert torch.equal(flat, torch.arange(0, (H // 2) * W).float())  # rows 0..1 -> tokens 0..5


def test_usp_attn_op_hands_usp_the_bhsd_layout():
    captured = {}

    def fake_attn(q, k, v, heads, skip_reshape=False, **kw):
        captured["shape"] = tuple(q.shape)
        captured["heads"] = heads
        captured["skip_reshape"] = skip_reshape
        b, h, s, d = q.shape
        return torch.zeros(b, s, h * d)

    op = _make_usp_attn_op(fake_attn)
    B, S, H, D = 1, 12, 4, 8
    q = torch.randn(B, S, H, D)             # cosmos attn_op sees (B, S, H, D)
    out = op(q, q, q, transformer_options={"anything": 1})
    assert captured["shape"] == (B, H, S, D)   # USP wants (B, H, S, D)
    assert captured["heads"] == H
    assert captured["skip_reshape"] is True
    assert out.shape == (B, S, H * D)


@pytest.fixture
def fake_model_base(monkeypatch):
    """CosmosPredict2 is Anima's sibling under a shared BaseModel root,
    separate and unsupported here."""
    return install_fake_model_base(
        monkeypatch, {"BaseModel": None, "Anima": "BaseModel", "CosmosPredict2": "BaseModel"}
    )


def test_dp_cond_exempt_keys_covers_raw_t5_sequences():
    # Anima's t5xxl_ids/t5xxl_weights are raw, un-batched sequence tensors in
    # the conditioning extras dict; the generic dp cond slicer must leave them
    # whole on every rank.
    assert AnimaAdapter().dp_cond_exempt_keys == frozenset({"t5xxl_ids", "t5xxl_weights"})


# tests/test_adapter_matches.py tables the exact-type accept and the typed
# reject of an unvetted Anima subclass. CosmosPredict2 is a third case: a
# real BaseModel sibling of Anima, not a subclass, so matches() returns False
# and the registry falls through to its typed 'no adapter' error.
def test_matches_excludes_sibling_cosmos_predict2(fake_model_base):
    assert AnimaAdapter().matches(fake_model_base.CosmosPredict2()) is False


_STOCK_OP = object()
_STOCK_FWD = object()


class _FakeAttn:
    def __init__(self):
        self.attn_op = _STOCK_OP


class _FakeBlock:
    def __init__(self):
        self.self_attn = _FakeAttn()
        self.cross_attn = _FakeAttn()


class _FakeLLMAdapter:
    """The 6-block T5 cross-attention adapter is structurally attn-bearing, but
    it runs in extra_conds and must never be rebound."""

    def __init__(self):
        self.blocks = [_FakeBlock() for _ in range(6)]


class _FakeAnima:
    def __init__(self, n_blocks=4):
        self.blocks = [_FakeBlock() for _ in range(n_blocks)]
        self.llm_adapter = _FakeLLMAdapter()
        self._forward = _STOCK_FWD


def test_inject_binds_self_attn_only_and_forward():
    model = _FakeAnima()
    AnimaAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))

    for block in model.blocks:
        assert block.self_attn.attn_op is not _STOCK_OP   # self-attn -> USP
        assert block.cross_attn.attn_op is _STOCK_OP        # cross-attn untouched
    # Every block gets the same closure.
    assert all(b.self_attn.attn_op is model.blocks[0].self_attn.attn_op for b in model.blocks)
    assert model._forward is not _STOCK_FWD                # distributed forward bound


def test_llm_adapter_is_never_in_the_bound_set():
    # The llm_adapter and every attention module it owns stay stock. It never
    # enters the sharded diffusion forward, so binding it would corrupt (or,
    # under FSDP, deadlock) the extra_conds path.
    model = _FakeAnima()
    AnimaAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))

    for block in model.llm_adapter.blocks:
        assert block.self_attn.attn_op is _STOCK_OP
        assert block.cross_attn.attn_op is _STOCK_OP


def test_state_dict_keys_never_mis_detect_as_another_family():
    # Real Anima checkpoint keys (cosmos MiniTrainDIT blocks plus the
    # Anima-only llm_adapter) must not match another family's signature. Wan
    # ("text_embedding." plus "self_attn.") must not claim them: Anima has
    # self_attn but no text_embedding.
    from dgx_monarch.adapters.detect import _has_segment, _strip_prefix, detect_family_from_keys

    anima_keys = [
        "model.diffusion_model.x_embedder.proj.1.weight",
        "model.diffusion_model.t_embedder.1.linear_1.weight",
        "model.diffusion_model.blocks.0.self_attn.q_proj.weight",
        "model.diffusion_model.blocks.0.cross_attn.q_proj.weight",
        "model.diffusion_model.blocks.0.mlp.layer1.weight",
        "model.diffusion_model.blocks.0.mlp.layer2.weight",
        "model.diffusion_model.final_layer.linear.weight",
        "model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight",
        "model.diffusion_model.llm_adapter.embed.weight",
    ]
    assert detect_family_from_keys(anima_keys) in ("unknown", "anima")

    # The anima row's segments ("llm_adapter." plus "mlp.layer1.") match Anima.
    stripped = [_strip_prefix(k) for k in anima_keys]
    assert all(_has_segment(stripped, n) for n in ("llm_adapter.", "mlp.layer1."))
    # They miss a plain Cosmos-Predict2 checkpoint (same cosmos blocks, no
    # llm_adapter): llm_adapter separates the two families.
    cosmos_predict2 = [k for k in stripped if "llm_adapter." not in k]
    assert not all(_has_segment(cosmos_predict2, n) for n in ("llm_adapter.", "mlp.layer1."))
