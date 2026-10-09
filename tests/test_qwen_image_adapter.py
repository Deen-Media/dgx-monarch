"""QwenImageAdapter CPU coverage. Nothing imports comfy: the bound forwards run
on SimpleNamespace and fake-class models.

Pins the typed rejects of a real mask and of Edit-2511 (index_timestep_zero)
references through the real bound USP forward, and plain `index` references
passing that gate; the "pad" cfg forward's text-length (B, trim) additive mask
(Qwen's convention, not Chroma's joint shape) and its idempotency; the declared
contract; and a detect-signature collision guard.
"""
import types

import pytest
import torch

from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.qwen_image import QwenImageAdapter


def test_declared_contract():
    # The comment above cfg_cond_padding in adapters/qwen_image.py says why Qwen
    # uses "pad".
    assert QwenImageAdapter.family == "qwen_image"
    assert QwenImageAdapter.cfg_cond_padding == "pad"
    assert QwenImageAdapter.model_base_classes == ("QwenImage",)


class _Sentinel(Exception):
    """Raised by a stub placed just past a gate to prove control reached it."""


def _usp_fake(default_ref_method="index"):
    fake = types.SimpleNamespace()
    fake.patch_size = 2
    fake.default_ref_method = default_ref_method
    fake.transformer_blocks = []  # inject_usp's log line reads len()

    def process_img(x, index=0, h_offset=0, w_offset=0):
        # (hidden_states, img_ids, orig_shape). The pre-shard reject tests only
        # need .shape[1]; process_img ignores the input tensor.
        return torch.zeros(1, 4, 8), torch.zeros(1, 4, 3), (1, 16, 1, 8, 8)

    fake.process_img = process_img
    return fake


def _inject_usp(fake):
    QwenImageAdapter().inject_usp(fake, InjectionContext(topology_sp=2, usp_attention=object()))


def test_usp_binds_forward():
    fake = _usp_fake()
    _inject_usp(fake)
    assert callable(fake._forward)


def test_usp_real_mask_raises_typed():
    # comfy's Qwen `Attention.forward` adds a mask over the joint keys, which the
    # Ulysses all-to-all cannot carry. Only the sharded forward rejects.
    fake = _usp_fake()
    _inject_usp(fake)
    with pytest.raises(UnsupportedModelError, match="mask"):
        fake._forward(torch.zeros(1, 16, 1, 8, 8), torch.zeros(1), torch.zeros(1, 8, 3584),
                      attention_mask=torch.zeros(1, 8))


def test_usp_edit_2511_timestep_zero_raises_typed():
    # index_timestep_zero splits per-block modulation at a global token index
    # (comfy's `_forward` hands `timestep_zero_index = num_embeds` to the block's
    # `_modulate` and `_apply_gate`); reject it before it renders wrong on shards.
    fake = _usp_fake("index_timestep_zero")
    _inject_usp(fake)
    with pytest.raises(UnsupportedModelError, match="index_timestep_zero"):
        fake._forward(torch.zeros(1, 16, 1, 8, 8), torch.zeros(1), torch.zeros(1, 8, 3584),
                      ref_latents=[torch.zeros(1, 16, 1, 8, 8)])


def test_usp_plain_index_ref_editing_passes_the_2511_gate():
    # Standard `index` references have no timestep-zero split, so they must
    # reach the sharded body. img_in is the first model call past the gate.
    fake = _usp_fake("index")

    def boom(*a, **k):
        raise _Sentinel

    fake.img_in = boom
    _inject_usp(fake)
    with pytest.raises(_Sentinel):  # reaching img_in means the 2511 gate passed
        fake._forward(torch.zeros(1, 16, 1, 8, 8), torch.zeros(1), torch.zeros(1, 8, 3584),
                      ref_latents=[torch.zeros(1, 16, 1, 8, 8)])


class _FakeQwenCfg:
    """Fake whose class defines _forward, so `type(self)._forward` (the stock
    seam the cfg-pad forward calls) resolves to this recording stub."""

    def __init__(self):
        self.recorded = []

    def _forward(self, x, timesteps, context, attention_mask=None, ref_latents=None,
                 additional_t_cond=None, transformer_options=None, control=None, **kwargs):
        self.recorded.append({"context": context, "attention_mask": attention_mask})
        return context


def _cfg_call(model, context, attention_mask=None):
    bs = context.shape[0]
    model._forward(torch.zeros(bs, 16, 1, 8, 8), torch.zeros(bs), context,
                   attention_mask=attention_mask)
    return model.recorded[-1]


def test_cfg_pad_trims_uniform_pad_maskfree():
    model = _FakeQwenCfg()
    QwenImageAdapter().inject_cfg_pad_forward(model)
    ctx = torch.ones(1, 6, 8)
    ctx[:, 4:] = 0.0  # driver-appended rows
    rec = _cfg_call(model, ctx)
    assert rec["context"].shape[1] == 4      # trailing pad trimmed
    assert rec["attention_mask"] is None     # a uniform pad needs no mask (flash path)


def test_cfg_pad_ragged_keeps_2d_text_mask():
    model = _FakeQwenCfg()
    QwenImageAdapter().inject_cfg_pad_forward(model)
    ctx = torch.ones(2, 6, 8)
    ctx[0, 4:] = 0.0
    ctx[1, 5:] = 0.0  # different real lengths in one call
    rec = _cfg_call(model, ctx)
    assert rec["context"].shape[1] == 5      # trimmed to the longest real length
    mask = rec["attention_mask"]
    # A (B, trim) text-length additive mask, not Chroma's (1, 1, txt+img) joint shape.
    assert mask is not None and mask.shape == (2, 5)
    assert int((mask[0] < 0).sum()) == 1 and int((mask[1] < 0).sum()) == 0


def test_cfg_pad_unpadded_is_stock_passthrough():
    model = _FakeQwenCfg()
    QwenImageAdapter().inject_cfg_pad_forward(model)
    rec = _cfg_call(model, torch.ones(1, 5, 8))
    assert rec["context"].shape[1] == 5 and rec["attention_mask"] is None


def test_cfg_pad_real_mask_passes_through_untouched():
    model = _FakeQwenCfg()
    QwenImageAdapter().inject_cfg_pad_forward(model)
    real = torch.zeros(1, 5)
    rec = _cfg_call(model, torch.ones(1, 5, 8), attention_mask=real)
    assert rec["attention_mask"] is real  # honored by stock, never re-derived


def test_cfg_pad_bind_is_idempotent():
    # A second install must not make `type(self)._forward` the wrapper, which
    # would double-trim or recurse: it stays the stock class method.
    model = _FakeQwenCfg()
    adapter = QwenImageAdapter()
    adapter.inject_cfg_pad_forward(model)
    adapter.inject_cfg_pad_forward(model)
    ctx = torch.ones(1, 6, 8)
    ctx[:, 4:] = 0.0
    rec = _cfg_call(model, ctx)
    assert rec["context"].shape[1] == 4 and rec["attention_mask"] is None


# Control overlap: comfy's Qwen `_forward` adds control to the image stream at
# global rows [0, add_len). The streams stay separate, so span_start is always 0,
# with no leading text block to offset past, unlike flux single blocks. The
# span_start=0 rows of test_flux_adapter.py::test_sharded_adds_reconstruct_stock_global_add
# pin that shared math (flux_shared._control_span_overlap) for this family.


# The matches() trio (exact-type accept, typed reject on an unvetted subclass,
# foreign-model decline) lives in tests/test_adapter_matches.py. comfy loads the
# Edit, 2509 and 2511 checkpoints as plain QwenImage; its subclasses MageFlow and
# QwenImage21 fail the exact-type check here and have adapters of their own.


def test_qwen_keys_never_mis_detect_as_another_family():
    # Qwen's row in detect._SIGNATURES is ("transformer_blocks.", "txt_norm.",
    # "img_in."). No earlier row may match these keys; the nearest are Lens, which
    # also needs img_mlp.w1., and qwen_image21, which also needs modulation.1.
    from dgx_monarch.adapters.detect import detect_family_from_keys

    keys = [
        "model.diffusion_model.txt_norm.weight",
        "model.diffusion_model.img_in.weight",
        "model.diffusion_model.txt_in.weight",
        "model.diffusion_model.time_text_embed.timestep_embedder.linear_1.weight",
        "model.diffusion_model.transformer_blocks.0.img_mod.1.weight",
        "model.diffusion_model.transformer_blocks.0.attn.to_q.weight",
        "model.diffusion_model.transformer_blocks.0.attn.add_q_proj.weight",
        "model.diffusion_model.transformer_blocks.0.attn.norm_added_q.weight",
        "model.diffusion_model.norm_out.linear.weight",
        "model.diffusion_model.proj_out.weight",
    ]
    assert detect_family_from_keys(keys) in ("unknown", "qwen_image")
