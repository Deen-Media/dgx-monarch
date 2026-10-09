"""Krea2 rebound-forward call contract against stock SingleStreamDiT.

Both krea2 injections replace `SingleStreamDiT._forward`, and stock `forward`
hands the wrapper executor every argument by position. A rewritten forward
whose positional order drifts from stock is therefore a TypeError on the first
render, not an import error, so an import check cannot catch it.

Comfy 14843 (2026-07-18) inserted `ref_latents` between `attention_mask` and
`transformer_options`; every distributed krea2 render raised
"usp_forward() takes from 4 to 6 positional arguments but 7 were given" until
the adapter matched. The expected order is read from the comfy_surface
manifest so the manifest stays the single source of truth: the canary fails
when the manifest goes stale against comfy master, and these tests fail when
the adapter goes stale against the manifest.

CPU-only, no comfy import: the guards run before the adapter's first comfy
import, and the signature checks are pure `inspect` work.
"""

from __future__ import annotations

import inspect

import pytest

from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.krea2 import Krea2Adapter
from dgx_monarch.comfy_surface import TOUCHPOINTS


def _manifest_positional(path: str) -> tuple[str, ...]:
    for touchpoint in TOUCHPOINTS:
        if touchpoint.path == path:
            return touchpoint.positional
    raise AssertionError(f"{path} is not inventoried in comfy_surface.TOUCHPOINTS")


class _FakeKrea2:
    """Minimal bind target: `bind` only needs an object that accepts setattr,
    and injection only reads `blocks` for its log line."""

    def __init__(self, blocks: int = 4, default_ref_method=None) -> None:
        self.blocks = [object()] * blocks
        self.default_ref_method = default_ref_method


def _injected(kind: str, **model_kwargs):
    model = _FakeKrea2(**model_kwargs)
    adapter = Krea2Adapter()
    if kind == "usp":
        adapter.inject_usp(model, InjectionContext(topology_sp=2, usp_attention=lambda *a, **k: None))
    else:
        adapter.inject_cfg_pad_forward(model)
    return model._forward


@pytest.mark.parametrize("kind", ["usp", "cfg_pad"])
def test_rebound_forward_matches_stock_positional_order(kind):
    """The bound forward accepts stock's positional sequence, in stock's order."""
    expected = _manifest_positional("comfy.ldm.krea2.model.SingleStreamDiT._forward")
    assert expected[0] == "self"
    stock_order = expected[1:]  # `self` is bound away on the injected method

    signature = inspect.signature(_injected(kind))
    actual = tuple(
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )[: len(stock_order)]
    assert actual == stock_order, (
        f"{kind} forward positional order {actual} != stock {stock_order}; comfy dispatches "
        "these positionally, so any drift is a TypeError on the first render"
    )

    # The exact call stock's WrapperExecutor makes, argument-for-argument.
    signature.bind(*([object()] * len(stock_order)))


@pytest.mark.parametrize("kind", ["usp", "cfg_pad"])
def test_plain_txt2img_call_is_accepted(kind):
    """ref_latents=None positional (every non-reference render) reaches the body.

    The body then needs comfy and real tensors, so an ImportError, an
    AttributeError, or a TypeError that is not about positional arguments
    proves the guards passed. A positional-argument TypeError is the
    regression this pins.
    """
    forward = _injected(kind)
    with pytest.raises((ImportError, AttributeError, TypeError)) as excinfo:
        forward(object(), object(), object(), None, None, {})
    if isinstance(excinfo.value, TypeError):
        assert "positional argument" not in str(excinfo.value), (
            f"{kind} forward still rejects stock's positional call shape: {excinfo.value}"
        )


@pytest.mark.parametrize("kind", ["usp", "cfg_pad"])
def test_reference_latents_are_rejected_not_ignored(kind):
    """Ref latents concatenate global tokens; silently dropping them would
    render a plain txt2img image for an edit workflow."""
    forward = _injected(kind, default_ref_method="index_timestep_zero")
    with pytest.raises(UnsupportedModelError, match="reference latents"):
        forward(object(), object(), object(), None, [object()], {})


@pytest.mark.parametrize("kind", ["usp", "cfg_pad"])
def test_reference_latents_without_method_are_not_rejected(kind):
    """Stock only activates the ref path when a method is resolved, so an empty
    list or a method-less model must not trip the guard."""
    forward = _injected(kind, default_ref_method=None)
    with pytest.raises((ImportError, AttributeError, TypeError)):
        forward(object(), object(), object(), None, [object()], {})

    forward = _injected(kind, default_ref_method="index_timestep_zero")
    with pytest.raises((ImportError, AttributeError, TypeError)):
        forward(object(), object(), object(), None, [], {})


@pytest.mark.parametrize("kind", ["usp", "cfg_pad"])
def test_post_input_patch_is_rejected(kind):
    """A post_input patch rewrites the global img/txt streams and position ids
    before the block loop; neither distributed forward applies it."""
    forward = _injected(kind)
    options = {"patches": {"post_input": [lambda args: args]}}
    with pytest.raises(UnsupportedModelError, match="post_input"):
        forward(object(), object(), object(), None, None, options)


@pytest.mark.parametrize("hook", ["attn1_patch", "attn1_output_patch"])
def test_attn1_patches_rejected_on_sharded_path(hook):
    """Under USP the patch would receive one rank's shard of q/k/v."""
    forward = _injected("usp")
    options = {"patches": {hook: [lambda *a, **k: {}]}}
    with pytest.raises(UnsupportedModelError, match=hook):
        forward(object(), object(), object(), None, None, options)


@pytest.mark.parametrize("hook", ["attn1_patch", "attn1_output_patch"])
def test_attn1_patches_allowed_on_cfg_pad_path(hook):
    """The cfg-pad forward holds the complete stream and publishes block_index,
    so stock's patch contract holds and the patch is not rejected."""
    forward = _injected("cfg_pad")
    options = {"patches": {hook: [lambda *a, **k: {}]}}
    with pytest.raises((ImportError, AttributeError, TypeError)):
        forward(object(), object(), object(), None, None, options)
