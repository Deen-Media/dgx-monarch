"""Ernie-Image sharding bookkeeping (CPU, no comfy import, no distributed init).

The joint sequence is [image, text] with the image first and the image
temporal RoPE index equal to the text length Tmax. These tests pin the
pure-logic invariants the adapter relies on: full-length id construction,
identical token and rope sharding (never reindex), image-first head slicing,
and the typed attention-mask rejects (which fire before any comfy import,
so plain fakes exercise the real bound forwards).
"""
import types

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.ernie import ErnieAdapter, _joint_rope_ids


def _ids(start, length):
    # Distinct per-position values stand in for tokens/rope rows: an identity
    # "embedder" makes alignment directly comparable.
    return torch.arange(start, start + length).view(1, length, 1).float()


def test_joint_rope_ids_image_first_at_tmax():
    ids = _joint_rope_ids(1, 2, 3, 4, None, torch.device("cpu"))
    assert ids.shape == (1, 2 * 3 + 4, 3)

    img, txt = ids[0, :6], ids[0, 6:]
    # Image temporal index == Tmax; text occupies temporal 0..Tmax-1.
    assert torch.all(img[:, 0] == 4.0)
    assert torch.equal(txt[:, 0], torch.arange(4.0))
    assert torch.all(txt[:, 1:] == 0.0)
    # Row-major (h, w) spatial grid.
    assert torch.equal(img[:, 1], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
    assert torch.equal(img[:, 2], torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]))


def test_joint_rope_ids_honor_rope_options():
    opts = {"scale_y": 2.0, "scale_x": 3.0, "shift_t": 5.0, "shift_y": 1.0, "shift_x": 0.5}
    ids = _joint_rope_ids(1, 2, 2, 3, opts, torch.device("cpu"))
    img = ids[0, :4]
    assert torch.all(img[:, 0] == 3.0 + 5.0)  # Tmax + shift_t
    # h: linspace(shift_y, (hp-1)*scale_y + shift_y, hp); w likewise.
    assert torch.equal(img[:, 1], torch.tensor([1.0, 1.0, 3.0, 3.0]))
    assert torch.equal(img[:, 2], torch.tensor([0.5, 3.5, 0.5, 3.5]))


def test_image_index_depends_on_full_text_length():
    # The image temporal index tracks the full text length: ids rebuilt from a
    # local text shard (tmax=2 instead of 3) put the image at the wrong index.
    full = _joint_rope_ids(1, 1, 2, 3, None, torch.device("cpu"))
    local_wrong = _joint_rope_ids(1, 1, 2, 2, None, torch.device("cpu"))
    assert torch.all(full[0, :2, 0] == 3.0)
    assert not torch.equal(local_wrong[0, :2, 0], full[0, :2, 0])


@pytest.mark.parametrize("tmax", [3, 4])  # odd and even text lengths
def test_rope_shards_identically_to_tokens(sp, tmax):
    # Tokens and rope rows are both full-length joint [image, text] tensors;
    # the same pad+chunk keeps them aligned on every rank, whatever the parity
    # of the text length (never reindex).
    img, txt = _ids(200, 6), _ids(100, tmax)
    joint_tokens = torch.cat((img, txt), dim=1)
    joint_rope = torch.cat((img, txt), dim=1)
    for rank in (0, 1):
        sp(2, rank)
        tok_local, _ = base.shard_seq(joint_tokens)
        rope_local, _ = base.shard_seq(joint_rope)
        assert torch.equal(tok_local, rope_local)


def test_gather_then_head_slice_recovers_image_tokens(sp):
    # n_img + tmax = 9 does not divide sp=2: shard_seq pads to 10, the
    # gather trims back to 9, and the head slice recovers image-first rows.
    img, txt = _ids(200, 6), _ids(100, 3)
    joint = torch.cat((img, txt), dim=1)
    shards, orig = [], 0
    for rank in (0, 1):
        sp(2, rank)
        local, orig = base.shard_seq(joint)
        shards.append(local)
    gathered = torch.cat(shards, dim=1).narrow(1, 0, orig)
    assert torch.equal(gathered, joint)
    assert torch.equal(gathered[:, :6], img)


class _FakeAttention:
    pass


class _FakeErnie:
    def __init__(self, n_layers):
        self.layers = [types.SimpleNamespace(self_attention=_FakeAttention()) for _ in range(n_layers)]


def _injected(n_layers=2):
    model = _FakeErnie(n_layers)
    ErnieAdapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=None))
    return model


def test_forward_rejects_real_attention_mask():
    model = _injected()
    with pytest.raises(UnsupportedModelError, match="attention mask"):
        model.forward(
            torch.zeros(1, 4, 2, 2), torch.zeros(1), torch.zeros(1, 3, 8),
            attention_mask=torch.ones(1, 3),
        )


def test_attention_bind_rejects_real_mask():
    model = _injected(n_layers=3)
    for layer in model.layers:
        with pytest.raises(UnsupportedModelError, match="attention mask"):
            layer.self_attention.forward(torch.zeros(1, 4, 8), attention_mask=torch.ones(1, 4))


def test_inject_binds_forward_and_every_layer():
    model = _injected(n_layers=3)
    assert "forward" in model.__dict__
    for layer in model.layers:
        assert "forward" in layer.self_attention.__dict__


# The matches() trio (exact-type accept, typed reject on an unvetted
# subclass, foreign-model decline) is in the table in tests/test_adapter_matches.py.


def test_adapter_contract_declared():
    adapter = ErnieAdapter()
    assert adapter.family == "ernie"
    assert adapter.model_base_classes == ("ErnieImage",)
    # A driver pad would be inexact: the image RoPE index is the text length
    # Tmax, and the maskless blocks would attend the pad rows.
    assert adapter.cfg_cond_padding == "none"
