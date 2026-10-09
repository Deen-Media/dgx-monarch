"""CPU contracts for the distinct Mage-Flow adapter (no Comfy import at module scope)."""
from __future__ import annotations

import types

import pytest
import torch

from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.mage_flow import MageFlowAdapter


def test_declared_contract():
    assert MageFlowAdapter.family == "mage_flow"
    assert MageFlowAdapter.model_base_classes == ("MageFlow",)
    assert MageFlowAdapter.cfg_cond_padding == "pad"
    assert MageFlowAdapter.attention_head_dim == 128


def test_usp_rejects_stock_text_bias_before_sharding():
    model = types.SimpleNamespace(transformer_blocks=[])
    MageFlowAdapter().inject_usp(model, InjectionContext(2, object()))
    with pytest.raises(UnsupportedModelError, match="attention mask"):
        model._forward(torch.zeros(1, 128, 1, 1), torch.zeros(1), torch.zeros(1, 2, 8),
                       attention_mask=torch.tensor([[0.0, -1.0]]))


def test_usp_all_valid_boolean_mask_is_the_stock_noop(monkeypatch):
    import dgx_monarch.adapters.mage_flow as mage

    monkeypatch.setattr(mage, "_mask_is_noop", lambda mask: True)
    model = types.SimpleNamespace(transformer_blocks=[])
    MageFlowAdapter().inject_usp(model, InjectionContext(2, object()))
    # Passing the mask gate reaches process_img instead of a typed rejection.
    model.process_img = lambda x, index=0: (_ for _ in ()).throw(RuntimeError("past-mask-gate"))
    with pytest.raises(RuntimeError, match="past-mask-gate"):
        model._forward(torch.zeros(1, 128, 1, 1), torch.zeros(1), torch.zeros(1, 2, 8),
                       attention_mask=torch.ones(1, 2, dtype=torch.bool))


def test_usp_preserves_mage_reference_ids_and_unrotated_text(monkeypatch):
    import dgx_monarch.adapters.mage_flow as mage

    monkeypatch.setattr(mage, "shard_seq", lambda value, dim: (value, value.shape[dim]))
    monkeypatch.setattr(mage, "sp_gather", lambda value, length, dim: value)
    monkeypatch.setattr(mage, "sp_rank", lambda: 0)
    monkeypatch.setattr(mage, "padded_row_indices", lambda segments: [])
    seen = []

    class Model:
        transformer_blocks = []
        out_channels = 128

        def process_img(self, x, index=0):
            # Use index in every image id so the test detects accidental Qwen
            # offsets or lost reference identity before local RoPE embedding.
            b, _c, h, w = x.shape
            rows = x.movedim(1, -1).reshape(b, h * w, 128)
            ids = torch.zeros(b, h * w, 3)
            ids[..., 0] = index
            ids[..., 1] = torch.tensor([-(h - h // 2) + row // w for row in range(h * w)])
            return rows, ids, (h, w)

        img_in = staticmethod(lambda value: value)
        txt_norm = staticmethod(lambda value: value)
        txt_in = staticmethod(lambda value: value)
        time_text_embed = staticmethod(lambda timestep, hidden: torch.zeros(hidden.shape[0], 1))
        norm_out = staticmethod(lambda value, temb: value)
        proj_out = staticmethod(lambda value: value)

        @staticmethod
        def pe_embedder(ids):
            seen.append(ids.clone())
            return ids

    model = Model()
    MageFlowAdapter().inject_usp(model, InjectionContext(2, object()))
    target = torch.zeros(1, 128, 3, 1)
    reference = torch.zeros(1, 128, 1, 1)
    result = model._forward(target, torch.zeros(1), torch.ones(1, 2, 8), ref_latents=[reference])
    assert result.shape == target.shape
    ids = seen[-1]
    assert torch.equal(ids[:, :2], torch.zeros(1, 2, 3))  # text RoPE is identity
    assert set(ids[0, 2:, 0].tolist()) == {0.0, 1.0}      # target then reference frame
    assert ids[0, 2, 1].item() == -2.0                   # Mage's odd-height centring


def test_pure_ulysses_uses_native_joint_order_with_reference_and_pad_rows(monkeypatch):
    """Mage passes the exact local text/image pair order to full-axis USP."""
    import dgx_monarch.adapters.base as base
    import dgx_monarch.adapters.mage_flow as mage
    from dgx_monarch.adapters.usp_sequence_order import RankMajorJointOrder

    seen = {}
    monkeypatch.setattr(base, "sp_world", lambda: 2)

    def shard(value, dim=1):
        original = value.shape[dim]
        # Rank-local lengths after per-stream zero padding: text 3 rows to 2,
        # target plus reference image 5 rows to 3.
        return value.narrow(dim, 0, (original + 1) // 2), original

    def options(_options, _attention, drop_rows=None, sequence_order=None):
        seen["drop_rows"] = drop_rows
        seen["sequence_order"] = sequence_order
        return {}

    monkeypatch.setattr(mage, "shard_seq", shard)
    monkeypatch.setattr(mage, "sp_gather", lambda value, length, dim: torch.cat(
        (value, value), dim=dim).narrow(dim, 0, length))
    monkeypatch.setattr(mage, "sp_rank", lambda: 0)
    monkeypatch.setattr(mage, "usp_options", options)

    class Model:
        transformer_blocks = []
        out_channels = 128

        def process_img(self, x, index=0):
            b, _c, h, w = x.shape
            rows = x.movedim(1, -1).reshape(b, h * w, 128)
            ids = torch.zeros(b, h * w, 3)
            ids[..., 0] = index
            return rows, ids, (h, w)

        img_in = staticmethod(lambda value: value)
        txt_norm = staticmethod(lambda value: value)
        txt_in = staticmethod(lambda value: value)
        time_text_embed = staticmethod(lambda timestep, hidden: torch.zeros(hidden.shape[0], 1))
        pe_embedder = staticmethod(lambda ids: ids)
        norm_out = staticmethod(lambda value, temb: value)
        proj_out = staticmethod(lambda value: value)

    model = Model()
    MageFlowAdapter().inject_usp(model, InjectionContext(2, object(), pure_ulysses=True))
    target = torch.zeros(1, 128, 3, 1)
    reference = torch.zeros(1, 128, 2, 1)
    result = model._forward(target, torch.zeros(1), torch.ones(1, 3, 128),
                            ref_latents=[reference])
    assert result.shape == target.shape
    order = seen["sequence_order"]
    assert isinstance(order, RankMajorJointOrder)
    assert (order.text_rows, order.image_rows) == (2, 3)
    # drop_rows arrive in rank-major coordinates: rank 1's text row 1 and image
    # row 2. The descriptor maps them to stock's [all text, all image] ones.
    assert seen["drop_rows"] == [6, 9]
    rank_major = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1, 1)
    canonical, _k, _v, drops, _kv_drops, permutation = order.canonicalize(
        rank_major, rank_major, rank_major, seen["drop_rows"], None, None)
    assert torch.equal(canonical[:, :, 0, 0], torch.tensor(
        [[0, 1, 5, 6, 2, 3, 4, 7, 8, 9]], dtype=torch.float32))
    assert drops == [3, 9]
    assert torch.equal(order.restore(canonical, permutation), rank_major)


class _FakeMageCfg:
    def __init__(self):
        self.calls = []

    def _forward(self, x, timestep, context, attention_mask=None, ref_latents=None,
                 transformer_options=None, control=None, **kwargs):
        self.calls.append((context, attention_mask))
        return context


def test_cfg_padding_restores_stock_text_input():
    model = _FakeMageCfg()
    MageFlowAdapter().inject_cfg_pad_forward(model)
    context = torch.ones(1, 6, 8)
    context[:, 4:] = 0
    model._forward(torch.zeros(1, 128, 1, 1), torch.zeros(1), context)
    trimmed, mask = model.calls[-1]
    assert trimmed.shape[1] == 4
    assert mask is None


def test_cfg_padding_keeps_a_real_mask():
    model = _FakeMageCfg()
    MageFlowAdapter().inject_cfg_pad_forward(model)
    mask = torch.zeros(1, 4)
    model._forward(torch.zeros(1, 128, 1, 1), torch.zeros(1), torch.ones(1, 4, 8), attention_mask=mask)
    assert model.calls[-1][1] is mask


def test_nvfp4_scale_scope_excludes_only_mage_stream_padding_rows():
    from dgx_monarch.adapters import mage_nvfp4_scale as scale

    text = torch.tensor([[[1.0], [999.0]]])
    image = torch.tensor([[[2.0], [3.0], [888.0]]])

    def block(**_kwargs):
        text_real = scale.real_input("transformer_blocks.0.attn.add_q_proj", text)
        image_real = scale.real_input("transformer_blocks.0.attn.to_q", image)
        # img/text output role mapping is explicit; unrelated replicated input
        # keeps its full value even when token lengths happen to coincide.
        other = scale.real_input("time_text_embed.linear_1", text)
        return text_real, image_real, other

    text_real, image_real, other = scale.call(
        block, text_original=3, text_local=2, image_original=5, image_local=3, rank=1,
    )
    assert torch.equal(text_real, torch.tensor([[[1.0]]]))
    assert torch.equal(image_real, torch.tensor([[[2.0], [3.0]]]))
    assert other is text


def test_nvfp4_scale_scope_empty_real_span_contributes_no_padded_rows():
    from dgx_monarch.adapters import mage_nvfp4_scale as scale

    padded = torch.tensor([[[1000.0], [1001.0]]])
    result = scale.call(
        lambda **_kwargs: scale.real_input("transformer_blocks.0.txt_mlp.net.0.proj", padded),
        text_original=2, text_local=2, image_original=2, image_local=2, rank=1,
    )
    assert result.shape[1] == 0


def test_nvfp4_empty_real_span_uses_zero_before_shared_max_reduction():
    from dgx_monarch.adapters import quant_activation_scale as qas

    seen = []
    plan = qas.SharedScalePlan((("sp", lambda value: seen.append(value.clone()) or torch.tensor([7.0])),))
    empty = torch.empty((1, 0, 4), dtype=torch.bfloat16)
    scale = qas.shared_activation_scale(empty, plan, 2.0)
    assert torch.equal(seen[0], torch.zeros(1, dtype=torch.float32))
    assert torch.equal(scale, torch.tensor(3.5, dtype=torch.bfloat16))


def test_nvfp4_row_scope_restores_after_block_error():
    from dgx_monarch.adapters import mage_nvfp4_scale as scale

    with pytest.raises(RuntimeError, match="boom"):
        scale.call(lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
                   text_original=1, text_local=2, image_original=1, image_local=2, rank=0)
    value = torch.ones((1, 2, 1))
    assert scale.real_input("transformer_blocks.0.txt_mlp.net.0.proj", value) is value


class _MixedDenseLinear(torch.nn.Module):
    """comfy's mixed-precision Linear holding an unquantized BF16 weight.

    It subclasses torch.nn.Module, not torch.nn.Linear, and leaves
    `layout_type` unset, which is how an NVFP4 native-view file loads
    `to_add_out`.
    """

    def __init__(self, source: torch.nn.Linear):
        super().__init__()
        self.weight = torch.nn.Parameter(source.weight.detach().clone(), requires_grad=False)
        self.bias = torch.nn.Parameter(source.bias.detach().clone(), requires_grad=False)

    def forward(self, value):
        return torch.nn.functional.linear(value, self.weight, self.bias)


class _NVFP4TextProjectionBlock(torch.nn.Module):
    """Small native-shape stand-in for Mage's Qwen-derived double block."""

    def __init__(self, *, nvfp4=True, output_quantized=False, mixed_precision=False):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.to_add_out = torch.nn.Linear(3, 2, dtype=torch.bfloat16)
        if mixed_precision:
            self.attn.to_add_out = _MixedDenseLinear(self.attn.to_add_out)
        if output_quantized:
            self.attn.to_add_out.layout_type = "TensorCoreNVFP4Layout"
        if nvfp4:
            # The NVFP4 text projections are not selected for full rows.
            for name in ("add_q_proj", "add_k_proj", "add_v_proj"):
                quantized = torch.nn.Identity()
                quantized.layout_type = "TensorCoreNVFP4Layout"
                setattr(self.attn, name, quantized)
        self.seen = []
        self.raise_after_projection = False

    def forward(self, *, encoder_hidden_states, **_kwargs):
        self.seen.append(self.attn.to_add_out(encoder_hidden_states))
        if self.raise_after_projection:
            raise RuntimeError("boom")
        return self.seen[-1]


def _project(monkeypatch, block, value, *, text_original, image_original, rank=0,
             pure_ulysses=True, gather=None):
    """The forward's nesting: full-row projection scope around the NVFP4 row scope."""
    from dgx_monarch.adapters import chroma_text, mage_flow
    from dgx_monarch.adapters import mage_nvfp4_scale as scale

    if gather is not None:
        monkeypatch.setattr(mage_flow, "sp_gather", gather)
    monkeypatch.setattr(chroma_text, "sp_rank", lambda: rank)
    monkeypatch.setattr(chroma_text, "sp_world", lambda: 2)
    with chroma_text.full_row_text_projections(
            block, text_original, image_original, pure_ulysses=pure_ulysses,
            select=mage_flow._text_modules, gather=mage_flow._shared_gather()):
        return scale.call(
            block, text_original=text_original, text_local=value.shape[1],
            image_original=image_original, image_local=-(-image_original // 2),
            rank=rank, encoder_hidden_states=value)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_nvfp4_dense_text_projection_uses_global_native_view_and_zero_pads(
    monkeypatch, mixed_precision,
):
    """The BF16 projection sees stock's B>=2 [text, image] view shape.

    `to_add_out` in an NVFP4 block sees stock's joint view at batch 2, stride
    (36, 3, 1), with stock's values; its output is re-sharded with a zero pad
    row and its instance forward is restored.
    """
    torch.manual_seed(901)
    block = _NVFP4TextProjectionBlock(mixed_precision=mixed_precision)
    source = torch.randn(2, 5, 3, dtype=torch.bfloat16)
    padded = torch.cat((source, torch.zeros(2, 1, 3, dtype=torch.bfloat16)), dim=1)
    chunks = torch.chunk(padded, 2, dim=1)
    original = block.attn.to_add_out.forward
    seen = []

    def record(value):
        seen.append((value.shape, value.stride(), value.is_contiguous(), value.clone()))
        return original(value)

    block.attn.to_add_out.forward = record
    result = _project(
        monkeypatch, block, chunks[1], text_original=5, image_original=7, rank=1,
        gather=lambda value, length, dim=1: torch.cat(chunks, dim=dim).narrow(dim, 0, length))
    stock_input = torch.empty_strided(
        (2, 5, 3), ((5 + 7) * 3, 3, 1), dtype=torch.bfloat16,
    )
    stock_input.copy_(source)
    expected_full = original(stock_input)
    expected = torch.zeros(2, 3, 2, dtype=torch.bfloat16)
    expected[:, :2].copy_(expected_full[:, 3:5])
    assert torch.equal(result, expected)
    assert [entry[:3] for entry in seen] == [((2, 5, 3), (36, 3, 1), False)]
    assert torch.equal(seen[0][3], stock_input)
    assert vars(block.attn.to_add_out)["forward"] is record


@pytest.mark.parametrize("pure_ulysses,output_quantized,mixed_precision", [
    (False, False, False),
    (False, False, True),
    (True, True, False),
    (True, True, True),
])
def test_text_projection_leaves_ring_and_quantized_outputs_untouched(
    monkeypatch, pure_ulysses, output_quantized, mixed_precision,
):
    block = _NVFP4TextProjectionBlock(output_quantized=output_quantized,
                                      mixed_precision=mixed_precision)
    original = block.attn.to_add_out.forward
    value = torch.zeros(1, 2, 3, dtype=torch.bfloat16)

    def gather(*_args, **_kwargs):
        raise AssertionError("an unselected projection gathered")

    _project(monkeypatch, block, value, text_original=4, image_original=6,
             pure_ulysses=pure_ulysses, gather=gather)
    assert block.attn.to_add_out.forward == original
    assert "forward" not in vars(block.attn.to_add_out)
    assert block.seen[-1].shape == (1, 2, 2)


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_a_dense_output_projection_takes_full_rows_without_nvfp4(monkeypatch, mixed_precision):
    """BF16, FP16 and plain FP8 blocks, and an MXFP8 file's BF16 `to_add_out`."""
    block = _NVFP4TextProjectionBlock(nvfp4=False, mixed_precision=mixed_precision)
    rows = []
    original = block.attn.to_add_out.forward

    def record(value):
        rows.append(value.shape[1])
        return original(value)

    block.attn.to_add_out.forward = record
    value = torch.zeros(1, 2, 3, dtype=torch.bfloat16)
    _project(monkeypatch, block, value, text_original=4, image_original=6,
             gather=lambda value, length, dim=1: torch.cat((value, value), dim=dim))
    assert rows == [4]
    assert vars(block.attn.to_add_out)["forward"] is record


def test_text_projection_restores_instance_forward_after_error(monkeypatch):
    block = _NVFP4TextProjectionBlock()
    original = block.attn.to_add_out.forward
    block.raise_after_projection = True

    with pytest.raises(RuntimeError, match="boom"):
        _project(monkeypatch, block, torch.zeros(1, 2, 3, dtype=torch.bfloat16),
                 text_original=2, image_original=4, gather=lambda value, length, dim=1: value)
    assert block.attn.to_add_out.forward == original
    assert "forward" not in vars(block.attn.to_add_out)
