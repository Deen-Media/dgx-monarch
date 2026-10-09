"""Prove usp_options' override still engages under comfy's AttentionTensorContainer seam.

Comfy commit 8d534945 (2026-09-27, "Add comfy_attention and
AttentionTensorContainer to a few models") made several families' attention
call sites wrap q/k/v in ``AttentionTensorContainer`` and pass
``preferred_attention=`` to ``optimized_attention``/``optimized_attention_masked``.
The real dispatcher, ``comfy.ldm.modules.attention.wrap_attn``
(attention.py:200-238 at comfy master 7fbcfa8b), unwraps those containers to
plain tensors before calling ``transformer_options["optimized_attention_override"]``,
unless the override declares a ``container_function`` attribute.
``usp_options`` in ``adapters/base.py`` builds a plain nested function with no
such attribute, so every family that threads ``optimized_attention_override``
keeps working unmodified across the seam. A family that rebinds an attention
submodule never reaches comfy's wrapper, so the seam does not touch it.

The stub reproduces that dispatcher and imports no comfy, so it runs in the
CPU suite; the "lumina attention container" seam in
tests/canary/comfy_seam_contracts.py drives comfy's real wrap_attn. The stub
catches a regression in ``usp_options``' own override shape, not a change in
comfy's wrapper. That regression is silent: an override that gained a
``container_function`` attribute, or stopped being called, would hand wrapped
containers (or nothing) to the sharded attention path with no error, and the
output would be wrong under a topology that looked like it ran USP.
"""
from __future__ import annotations

import torch

from dgx_monarch.adapters.base import USP_ATTENTION_OVERRIDE_ATTR, usp_options


class _AttentionTensorContainer:
    """Reproduction of comfy master's container (attention.py:181-197)."""

    __slots__ = ("tensor",)

    def __init__(self, tensor):
        self.tensor = tensor

    def peek(self):
        if self.tensor is None:
            raise RuntimeError("attention tensor container has already been consumed")
        return self.tensor

    def take(self):
        tensor = self.peek()
        self.tensor = None
        return tensor


def _wrap_attn(func):
    """Reproduction of comfy master's wrap_attn (attention.py:200-238)."""

    def wrapper(*args, **kwargs):
        preferred_attention = kwargs.pop("preferred_attention", None)
        containers = None
        if len(args) >= 3 and isinstance(args[0], _AttentionTensorContainer):
            containers = args[:3]

        remove_key = False
        try:
            if "_inside_attn_wrapper" not in kwargs:
                transformer_options = kwargs.get("transformer_options", None)
                remove_key = True
                kwargs["_inside_attn_wrapper"] = True
                if transformer_options is not None:
                    override = transformer_options.get("optimized_attention_override")
                    if override is not None:
                        if containers is not None:
                            if hasattr(override, "container_function"):
                                return override.container_function(*args, **kwargs)
                            args = tuple(c.take() for c in containers) + args[3:]
                        return override(func, *args, **kwargs)
                if preferred_attention is not None and preferred_attention.function is not None:
                    return preferred_attention.function(*args, **kwargs)

            if containers is not None:
                if wrapper.container_function is not None:
                    return wrapper.container_function(*args, **kwargs)
                args = tuple(c.take() for c in containers) + args[3:]
            return func(*args, **kwargs)
        finally:
            if remove_key:
                del kwargs["_inside_attn_wrapper"]

    wrapper.container_function = None
    return wrapper


class _FakePreferredAttention:
    def __init__(self):
        self.calls = 0

    def function(self, *args, **kwargs):
        self.calls += 1
        return "preferred-attention-ran"


def _fake_stock_attention(q, k, v, heads, **kwargs):
    raise AssertionError("stock attention ran: the override did not engage")


@_wrap_attn
def _stub_optimized_attention(q, k, v, heads, **kwargs):
    return _fake_stock_attention(q, k, v, heads, **kwargs)


def _qkv_containers(shape=(1, 4, 8)):
    q = torch.randn(*shape)
    k = torch.randn(*shape)
    v = torch.randn(*shape)
    return q, k, v, (
        _AttentionTensorContainer(q),
        _AttentionTensorContainer(k),
        _AttentionTensorContainer(v),
    )


def test_usp_override_has_no_container_function_marker():
    """The override stays a plain function: one with a container_function
    attribute would get still-wrapped containers from comfy's wrapper."""
    override = usp_options({}, lambda *a, **k: None)["optimized_attention_override"]
    assert not hasattr(override, "container_function")
    assert getattr(override, USP_ATTENTION_OVERRIDE_ATTR, False) is True


def test_wrapped_containers_reach_the_usp_dispatcher_as_plain_tensors():
    """The positive case: comfy hands wrap_attn three containers, the override
    is installed, and the sharded dispatcher receives real tensors, not
    containers, with the stock path and preferred_attention never touched."""
    q, k, v, containers = _qkv_containers()
    received = {}

    def fake_usp_attention(*args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return torch.zeros(1, 4, 8)

    preferred = _FakePreferredAttention()
    options = usp_options({}, fake_usp_attention, drop_rows=(1, 2))

    out = _stub_optimized_attention(
        *containers, 4, mask=None, transformer_options=options,
        preferred_attention=preferred,
    )

    assert torch.equal(out, torch.zeros(1, 4, 8))
    assert preferred.calls == 0
    got_q, got_k, got_v, got_heads = received["args"]
    assert isinstance(got_q, torch.Tensor) and torch.equal(got_q, q)
    assert isinstance(got_k, torch.Tensor) and torch.equal(got_k, k)
    assert isinstance(got_v, torch.Tensor) and torch.equal(got_v, v)
    assert got_heads == 4
    assert received["kwargs"]["drop_rows"] == (1, 2)
    assert received["kwargs"]["mask"] is None
    assert "_inside_attn_wrapper" not in received["kwargs"]
    assert "transformer_options" not in received["kwargs"]
    # Every container was consumed by the unwrap, exactly once.
    for container in containers:
        assert container.tensor is None


def test_negative_control_stock_runs_without_an_override():
    """Proves this harness can detect a fall-through: with no override
    installed, wrap_attn's own container-unwrap still runs the stock path."""
    _, _, _, containers = _qkv_containers()
    try:
        _stub_optimized_attention(*containers, 4, transformer_options={})
    except AssertionError as exc:
        assert "stock attention ran" in str(exc)
    else:
        raise AssertionError("expected the stock path to run and raise")


def test_negative_control_a_container_function_marked_override_bypasses_unwrap():
    """If usp_options gained a container_function attribute, comfy would hand
    it still-wrapped containers instead of tensors; this control proves the
    harness sees that case."""
    _, _, _, containers = _qkv_containers()
    seen = {}

    # comfy calls override.container_function(*args, **kwargs) with the original
    # positional args (still-wrapped q/k/v and heads) and no func prepended:
    # its opt-in for a backend that wants the containers.
    def marked_override(q, k, v, heads, **kwargs):
        seen["qkv"] = (q, k, v)
        return "ran"

    marked_override.container_function = marked_override
    result = _stub_optimized_attention(
        *containers, 4, transformer_options={"optimized_attention_override": marked_override},
    )
    assert result == "ran"
    assert all(isinstance(a, _AttentionTensorContainer) for a in seen["qkv"])


def test_usp_attention_callable_accepts_the_unwrapped_shapes_containers_would_carry():
    """End-to-end: feed the raw tensors wrap_attn's unwrap hands the override
    through the real dgx-monarch dispatcher builder, both calling conventions
    comfy's optimized_attention takes (3-D packed, and 4-D skip_reshape)."""
    from dgx_monarch.adapters.base import _make_usp_attention_callable

    calls = []

    def fake_usp_attn(_group, q_l, k_l, v_l):
        calls.append((q_l.shape, k_l.shape, v_l.shape))
        return torch.zeros_like(q_l)

    dispatch = _make_usp_attention_callable(fake_usp_attn)

    _, _, _, containers3 = _qkv_containers(shape=(1, 4, 8))
    out3 = _stub_optimized_attention(
        *containers3, 2, transformer_options=usp_options({}, dispatch),
    )
    assert out3.shape == (1, 4, 8)
    assert calls, "the sharded dispatcher never ran"

    calls.clear()
    q4 = torch.randn(1, 2, 4, 4)
    k4 = torch.randn(1, 2, 4, 4)
    v4 = torch.randn(1, 2, 4, 4)
    containers4 = (
        _AttentionTensorContainer(q4),
        _AttentionTensorContainer(k4),
        _AttentionTensorContainer(v4),
    )
    out4 = _stub_optimized_attention(
        *containers4, 2, skip_reshape=True, skip_output_reshape=True,
        transformer_options=usp_options({}, dispatch),
    )
    assert out4.shape == (1, 2, 4, 4)
    assert calls
