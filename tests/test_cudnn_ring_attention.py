from __future__ import annotations

import sys
import types

import pytest
import torch

from dgx_monarch.adapters import cudnn_ring_attention as subject
from dgx_monarch.adapters.base import UnsupportedModelError


def _fake_inputs(dtype=torch.float16):
    return (torch.zeros(1, 2, 1, 4, dtype=dtype),) * 3


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scale", [None, 0.5, 0.0, -0.5])
def test_native_cudnn_arguments_layout_and_true_lse(monkeypatch, dtype, scale):
    q, k, v = _fake_inputs(dtype)
    seen = {}
    monkeypatch.setattr(subject, "_validate_qkv", lambda *_: None)
    def native(qh, kh, vh, bias, compute_lse, dropout, causal, debug, *, scale):
        seen.update(q=qh, k=kh, v=vh, bias=bias, compute_lse=compute_lse, dropout=dropout,
                    causal=causal, debug=debug, scale=scale)
        return qh + 1, torch.tensor([[[0.0, 1.1]]], dtype=torch.float32), None, None, 0, 0, None, None, None
    monkeypatch.setattr(subject, "_aten_cudnn_attention_op", lambda: native)
    out, lse = subject.TorchCudnnRingProcessor()(q, k, v, softmax_scale=scale)
    assert out.shape == q.shape and out.dtype == q.dtype and torch.equal(lse, torch.tensor([[[0.0, 1.1]]]))
    assert seen["q"].shape == (1, 1, 2, 4)
    assert (seen["bias"], seen["compute_lse"], seen["dropout"], seen["causal"], seen["debug"], seen["scale"]) == (None, True, 0.0, False, False, scale)


def test_zero_lse_is_valid_native_metadata(monkeypatch):
    q, k, v = _fake_inputs()
    monkeypatch.setattr(subject, "_validate_qkv", lambda *_: None)
    monkeypatch.setattr(subject, "_aten_cudnn_attention_op", lambda: lambda qh, *args, **kw: (
        qh, torch.zeros(1, 1, 2, dtype=torch.float32)))
    _out, lse = subject.TorchCudnnRingProcessor()(q, k, v)
    assert torch.count_nonzero(lse) == 0


def test_lse_with_a_trailing_unit_dim_reads_as_b_h_l(monkeypatch):
    """torch 2.12's cuDNN attention returns the LSE as (B,H,L,1) on GB10."""
    q, k, v = _fake_inputs()
    monkeypatch.setattr(subject, "_validate_qkv", lambda *_: None)
    monkeypatch.setattr(subject, "_aten_cudnn_attention_op", lambda: lambda qh, *args, **kw: (
        qh, torch.tensor([[[[0.0], [1.1]]]], dtype=torch.float32)))
    _out, lse = subject.TorchCudnnRingProcessor()(q, k, v)
    assert lse.shape == (1, 1, 2) and torch.equal(lse, torch.tensor([[[0.0, 1.1]]]))


@pytest.mark.parametrize("out,lse", [
    (torch.zeros(1, 1, 2, 4, dtype=torch.float16), torch.zeros(1, 1, 3, dtype=torch.float32)),
    (torch.zeros(1, 1, 2, 4, dtype=torch.float32), torch.zeros(1, 1, 2, dtype=torch.float32)),
    (torch.zeros(1, 1, 2, 4, dtype=torch.float16), torch.zeros(1, 1, 2, dtype=torch.float16)),
    (torch.zeros(1, 1, 2, 4, dtype=torch.float16), torch.zeros(1, 1, 2, 2, dtype=torch.float32)),
])
def test_malformed_native_contract_is_rejected(monkeypatch, out, lse):
    q, k, v = _fake_inputs()
    monkeypatch.setattr(subject, "_validate_qkv", lambda *_: None)
    monkeypatch.setattr(subject, "_aten_cudnn_attention_op", lambda: lambda *_args, **_kwargs: (out, lse))
    with pytest.raises(UnsupportedModelError):
        subject.TorchCudnnRingProcessor()(q, k, v)


def test_qkv_validation_rejects_incompatible_kv_lengths_before_device_use():
    q = torch.empty(1, 2, 3, 4, dtype=torch.float16)
    k = torch.empty(1, 5, 3, 4, dtype=torch.float16)
    v = torch.empty(1, 4, 3, 4, dtype=torch.float16)
    with pytest.raises(UnsupportedModelError, match="K/V length"):
        subject._validate_qkv(q, k, v)


class _DenseProcessor:
    def __init__(self, *, bad_lse=False, bad_out=False):
        self.bad_lse, self.bad_out = bad_lse, bad_out

    def __call__(self, q, k, v, *, softmax_scale, **_kwargs):
        qh, kh, vh = (item.transpose(1, 2).float() for item in (q, k, v))
        scores = qh @ kh.transpose(-1, -2) * softmax_scale
        lse = torch.logsumexp(scores, dim=-1)
        out = (torch.softmax(scores, dim=-1) @ vh).transpose(1, 2).to(q.dtype)
        return (out + 1 if self.bad_out else out, torch.zeros_like(lse) if self.bad_lse else lse)


def test_probe_accepts_independent_dense_qkv_reference_on_cpu():
    subject._probe_native(_DenseProcessor(), device=torch.device("cpu"))


@pytest.mark.parametrize("processor", [_DenseProcessor(bad_lse=True), _DenseProcessor(bad_out=True)])
def test_probe_rejects_wrong_native_metadata_or_output(processor):
    with pytest.raises(RuntimeError, match="incorrect"):
        subject._probe_native(processor, device=torch.device("cpu"))


def test_probe_rejects_small_lse_error_with_correct_output():
    def inaccurate(q, k, v, **kwargs):
        out, lse = _DenseProcessor()(q, k, v, **kwargs)
        return out, lse + 0.01

    with pytest.raises(RuntimeError, match="incorrect native LSE"):
        subject._probe_native(inaccurate, device=torch.device("cpu"))


def _fake_factory_modules(monkeypatch, *, selector_returns_processor=True, drop_processor=False):
    class AttnType:
        pass
    class Xfuser:
        last = None
        def __init__(self, *, use_sync, attn_type, attn_processor):
            self.use_sync, self.attn_type, self.attn_processor = use_sync, attn_type, attn_processor
            if drop_processor:
                self.attn_processor = object()
            self.ring_attn_fn = self.default_ring
            type(self).last = self
        def default_ring(self, *args):
            return args[1]
        def __call__(self, attn, q, k, v):
            self.called_shapes = (q.shape, k.shape, v.shape)
            return self.ring_attn_fn(attn, q, k, v)
    kernels = types.ModuleType("yunchang.kernels")
    kernels.AttnType = AttnType
    kernels.select_flash_attn_impl = lambda _sentinel, **kwargs: (
        kwargs["attn_processor"] if selector_returns_processor else object())
    long_ctx = types.ModuleType("xfuser.core.long_ctx_attention")
    long_ctx.xFuserLongContextAttention = Xfuser
    monkeypatch.setitem(sys.modules, "yunchang.kernels", kernels)
    monkeypatch.setitem(sys.modules, "xfuser.core.long_ctx_attention", long_ctx)
    return Xfuser


def test_factory_retains_exact_custom_processor_and_default_ring(monkeypatch):
    xfuser = _fake_factory_modules(monkeypatch)
    monkeypatch.setattr(subject, "_probe_native", lambda processor: None)
    callable_ = subject.make_cudnn_ring_usp_attention(sync_ulysses=False)
    assert callable(callable_)
    assert xfuser.last.use_sync is False
    assert isinstance(xfuser.last.attn_type, subject._CudnnRingSentinel)
    assert isinstance(xfuser.last.attn_processor, subject.TorchCudnnRingProcessor)
    assert xfuser.last.ring_attn_fn.__func__ is xfuser.default_ring


def test_factory_callable_preserves_comfy_layout_and_default_ring_path(monkeypatch):
    xfuser = _fake_factory_modules(monkeypatch)
    monkeypatch.setattr(subject, "_probe_native", lambda processor: None)
    # This test covers the binding/layout seam; native CUDA input validation
    # and numerical execution have separate tests and hardware requirements.
    monkeypatch.setattr(subject, "_validate_qkv", lambda *args: None)
    attention = subject.make_cudnn_ring_usp_attention()
    q = torch.arange(24, dtype=torch.float16).reshape(1, 3, 8)
    k = torch.zeros((1, 5, 8), dtype=torch.float16)
    assert torch.equal(attention(q, k, k, heads=2), q)
    assert xfuser.last.called_shapes == ((1, 3, 2, 4), (1, 5, 2, 4), (1, 5, 2, 4))
    assert xfuser.last.ring_attn_fn.__func__ is xfuser.default_ring


@pytest.mark.parametrize("extra", [
    {"deterministic": True}, {"window_size": (1, 1)}, {"softcap": 1.0},
    {"alibi_slopes": torch.ones(1)}, {"return_softmax": True},
    {"softmax_scale": 0.5}, {"dropout_p": 0.1}, {"unknown": True},
])
def test_extra_comfy_options_refuse_before_ring_call(monkeypatch, extra):
    xfuser = _fake_factory_modules(monkeypatch)
    monkeypatch.setattr(subject, "_probe_native", lambda processor: None)
    attention = subject.make_cudnn_ring_usp_attention()
    q = torch.zeros((1, 3, 8), dtype=torch.float16)
    with pytest.raises(UnsupportedModelError, match="extra Comfy"):
        attention(q, q, q, heads=2, **extra)
    assert not hasattr(xfuser.last, "called_shapes")


def test_factory_rejects_xfuser_that_drops_processor_before_probe(monkeypatch):
    _fake_factory_modules(monkeypatch, drop_processor=True)
    monkeypatch.setattr(subject, "_probe_native", lambda processor: pytest.fail("probe ran"))
    with pytest.raises(RuntimeError, match="xFuser did not retain"):
        subject.make_cudnn_ring_usp_attention()


def test_factory_probe_failure_prevents_callable_publication(monkeypatch):
    _fake_factory_modules(monkeypatch)
    monkeypatch.setattr(subject, "_probe_native", lambda _processor: (_ for _ in ()).throw(RuntimeError("probe")))
    with pytest.raises(RuntimeError, match="probe"):
        subject.make_cudnn_ring_usp_attention()


def test_factory_rejects_selector_that_drops_processor(monkeypatch):
    _fake_factory_modules(monkeypatch, selector_returns_processor=False)
    monkeypatch.setattr(subject, "_probe_native", lambda processor: None)
    with pytest.raises(RuntimeError, match="yunchang"):
        subject.make_cudnn_ring_usp_attention()


def test_processor_real_lse_distinguishes_unequal_ring_blocks(monkeypatch):
    """Replacing the processor LSE with zeros breaks the true ring merge."""
    q = torch.tensor([[[[1.0]]]], dtype=torch.float16)
    first_k, first_v = torch.tensor([[[[0.0]]]], dtype=torch.float16), torch.tensor([[[[0.0]]]], dtype=torch.float16)
    second_k, second_v = torch.tensor([[[[1.0986123]]]], dtype=torch.float16), torch.tensor([[[[1.0]]]], dtype=torch.float16)
    monkeypatch.setattr(subject, "_validate_qkv", lambda *_: None)
    def native(qh, kh, vh, *_args, **_kwargs):
        scores = (qh.float() * kh.float()).sum(-1)
        lse = scores
        return vh, lse, None, None, 0, 0, None, None, None
    monkeypatch.setattr(subject, "_aten_cudnn_attention_op", lambda: native)
    first_out, first_lse = subject.TorchCudnnRingProcessor()(q, first_k, first_v)
    second_out, second_lse = subject.TorchCudnnRingProcessor()(q, second_k, second_v)
    def merge(out, lse, block, block_lse):
        return out - torch.sigmoid(block_lse - lse) * (out - block)
    merged = merge(first_out.float(), first_lse.transpose(1, 2).unsqueeze(-1), second_out.float(), second_lse.transpose(1, 2).unsqueeze(-1))
    zeroed = merge(first_out.float(), torch.zeros_like(merged), second_out.float(), torch.zeros_like(merged))
    expected = torch.softmax(torch.tensor([0.0, 1.0986123]), dim=0)[1]
    assert merged.item() == pytest.approx(expected.item(), rel=1e-3)
    assert zeroed.item() == pytest.approx(0.5)


@pytest.mark.parametrize("kwargs", [{"causal": True}, {"window_size": (1, 1)}, {"alibi_slopes": torch.ones(1)}, {"softcap": 1.0}, {"return_softmax": True}, {"deterministic": True}, {"unknown": 1}])
def test_unsupported_surface_is_rejected_before_native(monkeypatch, kwargs):
    monkeypatch.setattr(subject, "_aten_cudnn_attention_op", lambda: pytest.fail("native op called"))
    with pytest.raises(UnsupportedModelError):
        subject.TorchCudnnRingProcessor()(*_fake_inputs(), **kwargs)
