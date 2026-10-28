"""Ideogram4 refuses a Ulysses degree that does not divide its 18 heads (#137).

A four-Spark owner's explicit uly4 render on 1.3.0 loaded both model slots on
all four ranks and crashed in yunchang's all-to-all with a shape error, since
18 heads do not reshape to 4 by 4 (issue #101). The adapter now runs the head
check FLUX.3 and MOSS-Image already run, at injection and before any forward
binds, and the sweep matrix labels the same cells refuse:P.

CPU-only: the model is a namespace that carries the head count where comfy's
Ideogram4 transformer keeps it, and the Ulysses degree is monkeypatched.
"""
from __future__ import annotations

import types

import pytest

from dgx_monarch.adapters import pixeldit
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.flux3_packing import refuse_indivisible_heads
from dgx_monarch.adapters.pixeldit import (
    IDEOGRAM4_DIVISORS,
    IDEOGRAM4_HEADS,
    Ideogram4Adapter,
    ideogram4_heads,
)
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag


def _model(heads: int | None = IDEOGRAM4_HEADS):
    attention = types.SimpleNamespace(num_heads=heads) if heads is not None else types.SimpleNamespace()
    return types.SimpleNamespace(layers=[types.SimpleNamespace(attention=attention)])


def test_the_head_count_is_read_from_the_first_layer():
    assert ideogram4_heads(_model()) == 18
    assert ideogram4_heads(_model(None)) is None
    assert ideogram4_heads(types.SimpleNamespace()) is None
    assert ideogram4_heads(types.SimpleNamespace(layers=[])) is None


@pytest.mark.parametrize("degree, ok", [(1, True), (2, True), (3, True), (6, True), (9, True),
                                        (4, False), (5, False), (8, False)])
def test_only_degrees_that_divide_18_pass(degree, ok):
    if ok:
        refuse_indivisible_heads(IDEOGRAM4_HEADS, degree, family="ideogram4",
                                 divisors=IDEOGRAM4_DIVISORS)
        return
    with pytest.raises(UnsupportedModelError) as caught:
        refuse_indivisible_heads(IDEOGRAM4_HEADS, degree, family="ideogram4",
                                 divisors=IDEOGRAM4_DIVISORS)
    text = str(caught.value)
    assert parse_leading_refusal_tag(text).refusal_class is RefusalClass.PHYSICS
    assert text.startswith("[dgxm:P] ideogram4: this checkpoint has 18 attention heads")
    assert f"degree {degree}" in text and IDEOGRAM4_DIVISORS in text


def test_injection_refuses_uly4_before_any_forward_binds(monkeypatch):
    monkeypatch.setattr(pixeldit, "ulysses_degree", lambda: 4)
    model = _model()
    with pytest.raises(UnsupportedModelError, match="18 attention heads"):
        Ideogram4Adapter().inject_usp(model, InjectionContext(topology_sp=4, usp_attention=None))
    assert not hasattr(model, "forward")
    assert not hasattr(model, "forward_orig")


def test_injection_passes_the_guard_at_a_dividing_degree(monkeypatch):
    """At degree 2 the guard passes and injection continues to the binding step."""
    monkeypatch.setattr(pixeldit, "ulysses_degree", lambda: 2)
    seen: list[tuple[int, int]] = []

    def record(heads, degree, *, family, divisors):
        seen.append((heads, degree))

    monkeypatch.setattr(pixeldit, "refuse_indivisible_heads", record)
    model = _model()
    try:
        Ideogram4Adapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=None))
    except (AttributeError, TypeError):
        pass  # the namespace model has no blocks to bind; the guard ran first
    assert seen == [(18, 2)]


def test_a_model_without_a_readable_head_count_skips_the_guard(monkeypatch):
    monkeypatch.setattr(pixeldit, "ulysses_degree", lambda: 4)
    called = []
    monkeypatch.setattr(pixeldit, "refuse_indivisible_heads",
                        lambda *a, **k: called.append(a))
    try:
        Ideogram4Adapter().inject_usp(types.SimpleNamespace(),
                                      InjectionContext(topology_sp=4, usp_attention=None))
    except (AttributeError, TypeError):
        pass
    assert called == []
