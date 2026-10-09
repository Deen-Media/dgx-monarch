"""CPU coverage for the comfy PixelDiT and PiD adapter (no comfy import, no
distributed init). The pixel stage folds every per-pixel tensor to (B*L, ...),
so s_cond, the per-pixel tokens, each pixel block's pos_comp and PiD's
per-token LQ features must all take the same L (patch) shard. These tests pin
that bookkeeping, the exact-gather attention order, the typed lq_latent reject
and the detect signature.
"""
import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters import pixeldit_comfy as pc
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.pixeldit_comfy import PixelDiTAdapter


@pytest.fixture
def sp(monkeypatch):
    """Patch the xfuser rank accessors shard_seq consults."""

    def set_rank(world, rank):
        monkeypatch.setattr(base, "sp_world", lambda: world)
        monkeypatch.setattr(base, "sp_rank", lambda: rank)
        monkeypatch.setattr(pc, "sp_world", lambda: world)
        monkeypatch.setattr(pc, "sp_rank", lambda: rank)

    return set_rank


def test_shard_bl_selects_contiguous_patch_chunk_and_aligns_with_patch_stream(sp):
    # The pixel stage folds per-pixel tensors to (B*L, P2, ph). Sharding that on
    # the L axis must select the same contiguous patch indices as the patch
    # stream s (B, L, hidden); otherwise s_cond[p] and the pixel tokens of
    # patch p would come from different patches on a rank.
    batch, length, p2, ph, hidden = 2, 6, 4, 3, 5
    # Row b*L + l of patch_id holds its patch index l, and so does s[b, l].
    patch_id = torch.arange(length).view(1, length, 1, 1).expand(batch, length, p2, ph).reshape(batch * length, p2, ph).float()
    s = torch.arange(length).view(1, length, 1).expand(batch, length, hidden).float()
    for rank in (0, 1):
        sp(2, rank)
        x_local, x_orig = pc._shard_bl(patch_id, batch)
        s_local, s_orig = base.shard_seq(s, dim=1)
        assert x_orig == length and s_orig == length
        x_patches = x_local.reshape(batch, -1, p2, ph)[0, :, 0, 0]  # (L_local,)
        s_patches = s_local[0, :, 0]                                # (L_local,)
        assert torch.equal(x_patches, s_patches)


def test_shard_bl_gather_bl_roundtrip_reconstructs_full_L(sp, monkeypatch):
    # Odd L=5 pads to 6 at world 2; the gather must trim the pad and restore
    # the stock (B*L, ...) layout exactly.
    batch, length, p2, ph = 2, 5, 4, 3
    x = torch.arange(batch * length * p2 * ph).float().reshape(batch * length, p2, ph)

    shards = {}
    for rank in (0, 1):
        sp(2, rank)
        local, orig = pc._shard_bl(x, batch)
        assert orig == length
        shards[rank] = local

    def fake_gather(t, orig_len, dim=1):
        parts = [shards[r].reshape(batch, -1, p2, ph) for r in (0, 1)]
        return torch.cat(parts, dim=dim).narrow(dim, 0, orig_len)

    monkeypatch.setattr(pc, "sp_gather", fake_gather)
    full = pc._gather_bl(shards[0], batch, length)
    assert full.shape == (batch * length, p2, ph)
    assert torch.equal(full, x)


def test_gather_bl_world1_is_identity(sp, monkeypatch):
    # At world 1, shard_seq is identity and the gather is a pad-trim no-op; the
    # (B*L,...) <-> (B, L_local, ...) reshape round-trip must preserve order.
    batch, length, p2, ph = 3, 4, 2, 2
    x = torch.arange(batch * length * p2 * ph).float().reshape(batch * length, p2, ph)
    sp(1, 0)
    local, orig = pc._shard_bl(x, batch)
    assert orig == length and local.shape == x.shape
    monkeypatch.setattr(pc, "sp_gather", lambda t, orig_len, dim=1: t.narrow(dim, 0, orig_len))
    assert torch.equal(pc._gather_bl(local, batch, orig), x)


def test_pixel_pos_comp_shard_aligns_with_pixel_token_shard(sp):
    # The distributed PiTBlock shards its internally-built pos_comp (L, ...) on
    # dim=0; that shard must cover the same patch indices as x_comp, which comes
    # from the _shard_bl'd per-pixel tokens. Both go through shard_seq on the L
    # axis, so the indices coincide.
    batch, length, p2, ph, dim = 1, 6, 4, 3, 5
    patch_id = torch.arange(length).view(1, length, 1, 1).expand(batch, length, p2, ph).reshape(batch * length, p2, ph).float()
    pos_comp = torch.arange(length).view(length, 1).expand(length, dim).float()  # row l encodes patch l
    for rank in (0, 1):
        sp(2, rank)
        x_local, _ = pc._shard_bl(patch_id, batch)
        pos_local, _ = base.shard_seq(pos_comp, dim=0)
        x_patches = x_local.reshape(batch, -1, p2, ph)[0, :, 0, 0]  # (L_local,)
        assert torch.equal(x_patches, pos_local[:, 0])


def test_lq_features_take_the_same_L_shard_as_the_patch_stream(sp):
    # lq_features[k] are (B, L, hidden) on the patch grid. They must shard
    # token-for-token with the patch stream so the elementwise sigma gate lines
    # up (comfy pid.py `SigmaAwareGate` gates each token).
    batch, length, hidden = 2, 6, 4
    s = torch.arange(batch * length * hidden).float().reshape(batch, length, hidden)
    lq_a = s.clone()
    lq_b = (s + 100.0).clone()
    for rank in (0, 1):
        sp(2, rank)
        s_local, _ = base.shard_seq(s, dim=1)
        lq_local = pc._shard_lq_features([lq_a, lq_b])
        assert len(lq_local) == 2
        assert torch.equal(lq_local[0], s_local)
        assert torch.equal(lq_local[1], s_local + 100)


def test_lq_features_pad_and_trim_survive_odd_L(sp):
    # Features pad exactly like the patch stream, keeping the per-token gate
    # aligned until the common gather trims the synthetic row.
    batch, length, hidden = 1, 5, 3
    f = torch.randn(batch, length, hidden)
    padded, orig = base.pad_seq_to_multiple(f, 2, dim=1)
    assert orig == length and padded.shape[1] == 6
    assert float(padded[:, length:].abs().sum()) == 0.0
    sp(2, 1)
    (lq_local,) = pc._shard_lq_features([f])
    assert torch.equal(lq_local, padded.chunk(2, dim=1)[1])


def test_exact_gather_restores_stock_stream_order_and_local_rows(sp, monkeypatch):
    # Each rank presents [text_local, image_local]. The collective produces
    # rank-major order, while stock sees [all text, all image]. Include one
    # padded row in each stream to pin both reordering and trim/zero restore.
    segments = ((3, 2), (5, 3))
    rank_width = 5
    rank_major = torch.tensor(
        [10, 11, 20, 21, 22, 12, 99, 23, 24, 98], dtype=torch.float32
    ).view(1, 1, 10, 1)
    expected_global = torch.tensor(
        [10, 11, 12, 20, 21, 22, 23, 24], dtype=torch.float32
    ).view(1, 1, 8, 1)
    monkeypatch.setattr(pc, "_gather_rank_major", lambda _tensor: rank_major)

    seen = []

    def stock(q, k, v, heads, **kwargs):
        seen.extend((q.clone(), k.clone(), v.clone()))
        assert heads == 1
        assert kwargs["skip_reshape"] is True
        assert kwargs["skip_output_reshape"] is True
        return q

    for rank, expected in (
        (0, [10, 11, 20, 21, 22]),
        (1, [12, 0, 23, 24, 0]),
    ):
        sp(2, rank)
        local = rank_major.narrow(2, rank * rank_width, rank_width)
        output = pc._gather_stock_attention(
            stock, local, local, local, 1,
            segments=segments, skip_reshape=True, skip_output_reshape=True,
        )
        assert output.flatten().tolist() == expected

    assert all(torch.equal(t, expected_global) for t in seen)

    sp(2, 1)
    local = rank_major.narrow(2, rank_width, rank_width)
    output = pc._gather_stock_attention(
        stock, local, local, local, 1,
        segments=segments, skip_reshape=True, skip_output_reshape=False,
    )
    assert output.shape == (1, rank_width, 1)
    assert output.flatten().tolist() == [12, 0, 23, 24, 0]


def test_stock_gather_options_uses_original_attention_without_recursing(monkeypatch):
    called = []
    monkeypatch.setattr(pc, "_gather_stock_attention",
                        lambda stock, *args, **kwargs: called.append(stock) or "ok")
    original = object()
    options = pc._stock_gather_options({"kept": 1}, ((4, 2),))
    assert options["kept"] == 1
    assert options["optimized_attention_override"](
        original, object(), object(), object(), 1,
        transformer_options=options, _inside_attn_wrapper=True,
    ) == "ok"
    assert called == [original]


class _FakePiD:
    """Minimal stand-in: hasattr(lq_proj) marks it a PiD; empty block lists let
    inject_usp bind without a real model."""

    def __init__(self):
        self.lq_proj = object()
        self.patch_blocks = []
        self.pixel_blocks = []
        self._forward = object()


def test_missing_lq_latent_raises_typed_before_any_comfy_import():
    model = _FakePiD()
    PixelDiTAdapter().inject_usp(
        model, InjectionContext(topology_sp=2, usp_attention=object())
    )
    assert model._forward is not object
    with pytest.raises(UnsupportedModelError, match="lq_latent"):
        # Dummy x/timesteps: the guard fires first, so comfy is never imported.
        model._forward(torch.zeros(1, 3, 32, 32), torch.zeros(1), lq_latent=None)


def test_pixel_blocks_are_bound_for_distributed_execution():
    class _Blk:
        def __init__(self):
            self.forward = object()

    class _FakeT2I:
        def __init__(self):
            self.patch_blocks = [object(), object()]
            self.pixel_blocks = [_Blk(), _Blk()]
            self._forward = object()

    model = _FakeT2I()
    stock = [b.forward for b in model.pixel_blocks]
    PixelDiTAdapter().inject_usp(
        model, InjectionContext(topology_sp=2, usp_attention=object())
    )
    for blk, was in zip(model.pixel_blocks, stock, strict=True):
        assert blk.forward is not was  # rebound to the L-sharded PiTBlock forward
    assert model._forward is not object


def test_validated_sp_forward_has_no_accuracy_waiver_gate():
    model = _FakePiD()
    PixelDiTAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))
    for _render in range(2):
        with pytest.raises(UnsupportedModelError, match="PiD requires lq_latent") as exc:
            model._forward(torch.zeros(1, 3, 32, 32), torch.zeros(1))
        assert "sp_unvalidated" not in str(exc.value)


# The matches() trio (both exact types accepted, typed reject on an unknown
# PixelDiT subclass, foreign-model decline) is tabled generically in
# tests/test_adapter_matches.py.


def test_detect_signature_proposal_never_clashes_with_ideogram4():
    # Comfy PixelDiT T2I and PiD checkpoint keys resolve to "pixeldit_comfy"
    # through its detect._SIGNATURES row ("unknown" is still accepted below),
    # and must never match ideogram4's `llm_cond_proj.` needle, nor the reverse.
    from dgx_monarch.adapters.detect import detect_family_from_keys

    pixeldit_t2i_keys = [
        "model.diffusion_model.pixel_embedder.proj.weight",
        "model.diffusion_model.s_embedder.proj.weight",
        "model.diffusion_model.y_embedder.proj.weight",
        "model.diffusion_model.y_pos_embedding",
        "model.diffusion_model.patch_blocks.0.attn.qkv_x.weight",
        "model.diffusion_model.pixel_blocks.0.compress_to_attn.weight",
        "model.diffusion_model.final_layer.linear.weight",
    ]
    pid_keys = [
        *pixeldit_t2i_keys,
        "model.diffusion_model.lq_proj.latent_proj.0.weight",
        "model.diffusion_model.lq_proj.gate_modules.0.content_proj.weight",
    ]
    for keys in (pixeldit_t2i_keys, pid_keys):
        assert detect_family_from_keys(keys) in ("unknown", "pixeldit_comfy")
        assert detect_family_from_keys(keys) != "ideogram4"

    ideogram4_keys = [
        "model.diffusion_model.llm_cond_proj.weight",
        "model.diffusion_model.llm_cond_norm.weight",
        "model.diffusion_model.layers.0.attention.wqkv.weight",
    ]
    assert "pixel_embedder." not in " ".join(ideogram4_keys)
    assert detect_family_from_keys(ideogram4_keys) == "ideogram4"
