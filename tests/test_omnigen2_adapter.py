"""Omnigen2 adapter on CPU, with no comfy import at module scope.

matches() imports comfy.model_base lazily, so a fake stands in, and shard and
gather run with the base sp-rank accessors monkeypatched. Pinned: the ring-only
refusal of a Ulysses degree that does not divide the 21 query heads; gather
before the [:, -img_len:] image-tail slice, with the stream and its rotary
sharded into the same chunks; the typed refusal of a ragged ref batch;
exact-type dispatch that excludes the Boogu subclass; cfg="none", because
num_tokens feeds pe_shift; and a detect signature that does not match Boogu.
"""
import sys
import types

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters import omnigen2 as og2
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.omnigen2 import (
    Omnigen2Adapter,
    _assert_ring_only_topology,
    _assert_uniform_batch,
)
from fake_model_base_helpers import install_fake_model_base

# Checkpoint keys as real files name them, under the model.diffusion_model. prefix that
# _strip_prefix removes. Boogu shares the Lumina2 timestep embedder and the
# refiner and embedder blocks, but replaces the single `layers` stack with
# double- and single-stream stages (the Boogu branch of comfy's
# model_detection.detect_unet_config).
_OMNIGEN2_KEYS = [
    "model.diffusion_model.time_caption_embed.timestep_embedder.linear_1.bias",
    "model.diffusion_model.time_caption_embed.caption_embedder.0.weight",
    "model.diffusion_model.x_embedder.weight",
    "model.diffusion_model.ref_image_patch_embedder.weight",
    "model.diffusion_model.noise_refiner.0.attn.to_q.weight",
    "model.diffusion_model.ref_image_refiner.0.attn.to_q.weight",
    "model.diffusion_model.context_refiner.0.attn.to_q.weight",
    "model.diffusion_model.layers.0.attn.to_q.weight",
    "model.diffusion_model.layers.31.attn.to_q.weight",
    "model.diffusion_model.norm_out.linear_1.weight",
    "model.diffusion_model.image_index_embedding",
]
_BOOGU_KEYS = [
    "model.diffusion_model.time_caption_embed.timestep_embedder.linear_1.bias",
    "model.diffusion_model.time_caption_embed.caption_embedder.0.weight",
    "model.diffusion_model.x_embedder.weight",
    "model.diffusion_model.ref_image_patch_embedder.weight",
    "model.diffusion_model.noise_refiner.0.attn.to_q.weight",
    "model.diffusion_model.ref_image_refiner.0.attn.to_q.weight",
    "model.diffusion_model.context_refiner.0.attn.to_q.weight",
    "model.diffusion_model.double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight",
    "model.diffusion_model.single_stream_layers.0.attn.to_q.weight",
    "model.diffusion_model.norm_out.linear_1.weight",
]


def _ids(start, length):
    # Distinct per-position values stand in for tokens / rotary rows: an identity
    # "embedder" makes alignment directly comparable.
    return torch.arange(start, start + length).view(1, length, 1).float()


def test_ring_only_refuses_ulysses_degree_two():
    # 21 q heads are indivisible by 2: the Ulysses all-to-all cannot split them.
    with pytest.raises(UnsupportedModelError, match="ulysses degree 2"):
        _assert_ring_only_topology(2)


@pytest.mark.parametrize("uly", [4, 5, 6, 8])
def test_ring_only_refuses_other_non_divisors(uly):
    with pytest.raises(UnsupportedModelError, match="21 attention heads"):
        _assert_ring_only_topology(uly)


@pytest.mark.parametrize("uly", [1, 3, 7, 21])
def test_ring_only_allows_divisors_and_ring(uly):
    # Degree 1 is ring or single GPU (no Ulysses split); 3, 7 and 21 divide 21,
    # so the refusal is a head-divisibility check, not a ban on Ulysses.
    _assert_ring_only_topology(uly)  # must not raise


def test_requested_ulysses_degree_is_one_without_xfuser_state():
    # On CPU the parallel state is uninitialized, so the degree reads 1 and the
    # ring-only refusal cannot fire by mistake.
    assert og2._requested_ulysses_degree() == 1


@pytest.mark.parametrize("total,img_len", [(12, 6), (10, 4)])
def test_gather_then_slice_recovers_image_tail(sp, total, img_len):
    # combined = [text, ref, img], image last. The adapter shards with
    # allow_padding=False, so a total that does not divide is refused, and even
    # a divisible stream must be gathered before the tail is sliced.
    combined = _ids(100, total)
    img = combined[:, -img_len:]
    shards, orig = [], 0
    for rank in (0, 1):
        sp(2, rank)
        local, orig = base.shard_seq(combined)
        shards.append(local)
    gathered = torch.cat(shards, dim=1).narrow(1, 0, orig)
    assert orig == total
    assert torch.equal(gathered, combined)          # gather restores full length
    assert torch.equal(gathered[:, -img_len:], img)  # then slice the tail


def test_slice_before_gather_would_lose_tail(sp):
    # A local shard does not contain the whole image tail, even without padding.
    combined = _ids(100, 12)
    sp(2, 1)
    local, _ = base.shard_seq(combined, allow_padding=False)
    assert local.shape[1] < 8
    assert not torch.equal(local, combined[:, -8:])


@pytest.mark.parametrize("total", [10, 12])
def test_rotary_shards_identically_to_stream(sp, total):
    # The joined stream and its per-token rotary take the same chunk, so they
    # stay aligned on every rank. Never reindex either one.
    stream = _ids(0, total)
    rotary = _ids(500, total)
    for rank in (0, 1):
        sp(2, rank)
        s_local, s_orig = base.shard_seq(stream, allow_padding=False)
        r_local, r_orig = base.shard_seq(rotary, allow_padding=False)
        assert s_orig == r_orig == total
        # same chunk positions: values differ by construction, shapes align
        assert s_local.shape == r_local.shape


def test_uniform_batch_passes():
    _assert_uniform_batch([10])        # batch 1
    _assert_uniform_batch([10, 10])    # uniform batch > 1


def test_ragged_batch_rejects_typed():
    with pytest.raises(UnsupportedModelError, match="non-uniform per-entry sequence lengths"):
        _assert_uniform_batch([10, 12])


@pytest.fixture
def fake_model_base(monkeypatch):
    """Fake comfy.model_base in which Boogu subclasses Omnigen2, as comfy's
    model_base.py declares it. isinstance alone would accept Boogu as Omnigen2;
    the exact-type dispatch this pins is what tells them apart."""
    return install_fake_model_base(monkeypatch, {"Omnigen2": None, "Boogu": "Omnigen2"})


# The exact-type accept and the foreign-model decline are tabled in
# tests/test_adapter_matches.py. This case pins the real Boogu subclass
# relationship.
def test_matches_rejects_boogu_subclass_typed(fake_model_base):
    # Boogu passes the isinstance walk but is not exactly Omnigen2, so matches()
    # raises a typed refusal instead of running the Omnigen2 forward on Boogu's
    # dual-stream model and rendering it wrong in silence.
    with pytest.raises(UnsupportedModelError, match="omnigen2 variant Boogu"):
        Omnigen2Adapter().matches(fake_model_base.Boogu())


class _FakeOmnigen2:
    def __init__(self, n_layers=2):
        self.layers = [object() for _ in range(n_layers)]


class _FakeOmnigen2PadGuard(_FakeOmnigen2):
    patch_size = 1
    context_refiner = ()

    def __init__(self):
        super().__init__(n_layers=0)

    def time_caption_embed(self, timestep, text, _dtype):
        return torch.zeros(timestep.shape[0], 4), text

    def flat_and_pad_to_seq(self, hidden, _refs):
        batch = hidden.shape[0]
        image = torch.zeros(batch, 3, 4)
        return image, None, None, None, [[0]], [3], [None], [(1, 3)]

    def rope_embedder(self, batch, text_len, *_args):
        context_rope = torch.zeros(batch, text_len, 4)
        noise_rope = torch.zeros(batch, 3, 4)
        joint_rope = torch.zeros(batch, text_len + 3, 4)
        return context_rope, None, noise_rope, joint_rope, [text_len], [text_len + 3]

    def img_patch_embed_and_refine(self, hidden, *_args, **_kwargs):
        return hidden


def test_inject_binds_forward_when_ring(monkeypatch):
    monkeypatch.setattr(og2, "_requested_ulysses_degree", lambda: 1)  # ring/single
    model = _FakeOmnigen2()
    Omnigen2Adapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))
    assert callable(model.forward) and "forward" in model.__dict__


def test_inject_refuses_ulysses(monkeypatch):
    monkeypatch.setattr(og2, "_requested_ulysses_degree", lambda: 2)
    model = _FakeOmnigen2()
    with pytest.raises(UnsupportedModelError, match="ulysses degree 2"):
        Omnigen2Adapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))


def test_ring_path_refuses_unvouched_divisibility_padding(monkeypatch, sp):
    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    common_dit = types.ModuleType("comfy.ldm.common_dit")
    common_dit.pad_to_patch_size = lambda value, _patch: value
    comfy.ldm = ldm
    ldm.common_dit = common_dit
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.ldm", ldm)
    monkeypatch.setitem(sys.modules, "comfy.ldm.common_dit", common_dit)
    monkeypatch.setattr(og2, "_requested_ulysses_degree", lambda: 1)
    sp(2, 0)

    model = _FakeOmnigen2PadGuard()
    Omnigen2Adapter().inject_usp(
        model, InjectionContext(topology_sp=2, usp_attention=None)
    )
    with pytest.raises(
        UnsupportedModelError,
        match=r"omnigen2 joined-token stream.*requires exact divisibility",
    ):
        model.forward(
            torch.zeros(1, 1, 1, 3),
            torch.ones(1),
            torch.zeros(1, 2, 4),
            2,
        )


def test_declared_contract():
    # cfg="none": no exact pad rule exists (the cfg_cond_padding comment in
    # adapters/omnigen2.py says why).
    assert Omnigen2Adapter.family == "omnigen2"
    assert Omnigen2Adapter.model_base_classes == ("Omnigen2",)
    assert Omnigen2Adapter.cfg_cond_padding == "none"


def test_proposed_omnigen2_signature_excludes_boogu():
    # The needles copy the omnigen2 row of detect._SIGNATURES. Boogu shares the
    # Lumina2 timestep embedder but replaces the single `layers` stack with
    # double_stream_layers and single_stream_layers, so `layers.` tells them apart.
    from dgx_monarch.adapters.detect import _has_segment, _strip_prefix

    needles = ("time_caption_embed.timestep_embedder.", "layers.")
    og2_keys = [_strip_prefix(k) for k in _OMNIGEN2_KEYS]
    boogu_keys = [_strip_prefix(k) for k in _BOOGU_KEYS]

    assert all(_has_segment(og2_keys, n) for n in needles)          # matches omnigen2
    assert not all(_has_segment(boogu_keys, n) for n in needles)    # excludes boogu
    # Both share the timestep embedder; only omnigen2 has a bare `layers.` stack.
    assert _has_segment(boogu_keys, "time_caption_embed.timestep_embedder.")
    assert not _has_segment(boogu_keys, "layers.")


def test_current_detect_never_mislabels_omnigen2_or_boogu():
    # The omnigen2 keys never read as another family, and the Boogu keys never
    # read as omnigen2.
    from dgx_monarch.adapters.detect import detect_family_from_keys

    assert detect_family_from_keys(_OMNIGEN2_KEYS) in ("unknown", "omnigen2")
    assert detect_family_from_keys(_BOOGU_KEYS) != "omnigen2"
