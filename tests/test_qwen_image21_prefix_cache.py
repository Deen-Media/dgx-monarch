"""The writable Qwen Image 2.1 prefix cache of ComfyUI 503d785f0 (#134).

ComfyUI 503d785f0 (2026-10-22) appended ``prefix_cache`` to
``QwenImage21Transformer2DModel._forward``. Under dynamic VRAM, for a cache
device of ``auto``, the model base admits a writable K/V cache per sampling
run and hands it over there, and stock's ``auto`` stops reaching
``select_prefix_cache``. The sharded forward names the parameter and refuses
that case on every rank alike before any collective. ``off``, ``gpu``,
``cpu`` and a static patcher take stock's own branch on both trees.

The first half runs on fakes, CPU only. The second half binds ComfyUI's real
class in a spawned child (COMFY_DIR or COMFYUI_DIR) and runs against the
installed tree and master alike: the touchpoint admits each tree's forward,
the bound forward names each tree's parameters, and the refusal fires before
the sequence is built and before any exchange.
"""
from __future__ import annotations

import ast
import inspect
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch

from dgx_monarch.adapters import qwen_image21 as q21
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.qwen_image21 import (
    WRITABLE_PREFIX_CACHE_REFUSAL,
    QwenImage21Adapter,
    refuse_writable_prefix_cache,
    stock_forward_takes_prefix_cache,
)
from dgx_monarch.comfy_rebound_signatures import REBOUND_METHOD_CONTRACTS
from dgx_monarch.comfy_rebound_validation import rebound_signature_incompatibility
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag

PATH = "comfy.ldm.qwen_image21.model.QwenImage21Transformer2DModel._forward"
CONTRACT = next(contract for contract in REBOUND_METHOD_CONTRACTS if contract.path == PATH)
_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
AUTO = {"device": "auto", "dtype": "default"}
GPU = {"device": "gpu", "dtype": "default"}
CPU = {"device": "cpu", "dtype": "int8"}
OFF = {"device": "off", "dtype": "default"}
NO_IS_DYNAMIC = "no is_dynamic"


def _explicit(function) -> list[str]:
    return [name for name, parameter in inspect.signature(function).parameters.items()
            if parameter.kind not in _VARIADIC]


def _patcher(dynamic):
    if dynamic is None:
        return None
    if dynamic == NO_IS_DYNAMIC:
        return types.SimpleNamespace()
    return types.SimpleNamespace(is_dynamic=lambda: dynamic)


# (prefix_cache, cache options, writable seam, patcher is_dynamic, refuses).
RULES = {
    "auto-dynamic-on-503d785f0": (None, AUTO, True, True, True),
    "auto-admitted-cache": (object(), AUTO, True, True, True),
    "auto-no-patcher": (None, AUTO, True, None, True),
    "auto-patcher-without-is-dynamic": (None, AUTO, True, NO_IS_DYNAMIC, True),
    "auto-static-patcher": (None, AUTO, True, False, False),
    "auto-before-503d785f0": (None, AUTO, False, True, False),
    "gpu-dynamic": (None, GPU, True, True, False),
    "cpu-dynamic": (None, CPU, True, True, False),
    "off-dynamic": (None, OFF, True, True, False),
    "absent-policy-reads-off": (None, None, True, True, False),
    "a-stray-cache-on-gpu": (object(), GPU, True, True, True),
    "a-stray-cache-before-503d785f0": (object(), OFF, False, False, True),
}


@pytest.mark.parametrize("case", sorted(RULES))
def test_the_rule_reads_only_facts_every_rank_shares(case):
    prefix_cache, options, seam, dynamic, refuses = RULES[case]
    model = types.SimpleNamespace(current_patcher=_patcher(dynamic))
    if not refuses:
        refuse_writable_prefix_cache(model, prefix_cache, options, seam)
        return
    with pytest.raises(UnsupportedModelError) as caught:
        refuse_writable_prefix_cache(model, prefix_cache, options, seam)
    assert str(caught.value) == WRITABLE_PREFIX_CACHE_REFUSAL


def test_the_refusal_is_class_p_and_names_its_way_out():
    tag = parse_leading_refusal_tag(WRITABLE_PREFIX_CACHE_REFUSAL)
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert "docs/TROUBLESHOOTING.md #134" in WRITABLE_PREFIX_CACHE_REFUSAL
    for remedy in ("'gpu'", "'cpu'", "'off'", "topology 'single'", "gpus_per_host=1"):
        assert remedy in WRITABLE_PREFIX_CACHE_REFUSAL


def _forward_without(self, x, timesteps, context, ref_latents=None, image_slots=None,
                     transformer_options={}, **kwargs):  # noqa: B006
    raise AssertionError("the class forward is never called once bound")


def _forward_with(self, x, timesteps, context, ref_latents=None, image_slots=None,
                  transformer_options={}, prefix_cache=None, **kwargs):  # noqa: B006
    raise AssertionError("the class forward is never called once bound")


def test_the_seam_reads_the_class_forward():
    assert stock_forward_takes_prefix_cache(type("Master", (), {"_forward": _forward_with}))
    assert not stock_forward_takes_prefix_cache(type("Before", (), {"_forward": _forward_without}))
    assert not stock_forward_takes_prefix_cache(object)


def _bound(seam: bool, dynamic, monkeypatch):
    """A fake model whose class forward has the given form, bound by the adapter."""
    cls = type("Fake", (), {"_forward": _forward_with if seam else _forward_without})
    model = cls()
    built: list = []
    model.transformer_blocks = []
    model.reset_prefix_cache = lambda enabled: None
    model.current_patcher = _patcher(dynamic)

    def build_sequence(*_args):
        built.append(1)
        raise AssertionError("sequence built")

    model.build_sequence = build_sequence
    monkeypatch.setattr(q21, "sp_gather", lambda *_a, **_k: pytest.fail("no collective may run"))
    monkeypatch.setattr(q21, "agree_prefix_cache",
                        lambda *_a, **_k: pytest.fail("no collective may run"))
    QwenImage21Adapter().inject_usp(model, InjectionContext(
        topology_sp=2, usp_attention=types.SimpleNamespace(effective_kernel="TORCH_FLASH")))
    return model, built


def _call(model, options, prefix_cache=None):
    return model._forward(torch.empty(1, 64, 1, 1), torch.zeros(1), torch.empty(1, 1, 1),
                          transformer_options={"qwen_image21_cache": options},
                          prefix_cache=prefix_cache)


@pytest.mark.parametrize("seam", [True, False], ids=["503d785f0", "a071826fd"])
def test_the_bound_forward_names_prefix_cache_and_the_contract_admits_it(monkeypatch, seam):
    model, _built = _bound(seam, True, monkeypatch)
    usp_forward = model._forward.__func__
    assert _explicit(usp_forward) == [*CONTRACT.positional_or_keyword, *CONTRACT.optional_trailing]
    assert rebound_signature_incompatibility(inspect.signature(usp_forward), CONTRACT) is None
    stock = _forward_with if seam else _forward_without
    assert rebound_signature_incompatibility(inspect.signature(stock), CONTRACT) is None


def test_auto_on_the_writable_tree_refuses_before_the_sequence_or_any_collective(monkeypatch):
    model, built = _bound(True, True, monkeypatch)
    with pytest.raises(UnsupportedModelError) as caught:
        _call(model, AUTO)
    assert str(caught.value) == WRITABLE_PREFIX_CACHE_REFUSAL
    assert built == []


@pytest.mark.parametrize(("seam", "dynamic", "options"), [
    (True, True, GPU), (True, True, CPU), (True, True, OFF), (True, False, AUTO),
    (False, True, AUTO),
], ids=["gpu", "cpu", "off", "auto-static-patcher", "auto-before-503d785f0"])
def test_stocks_own_branch_passes_the_guard(monkeypatch, seam, dynamic, options):
    model, built = _bound(seam, dynamic, monkeypatch)
    with pytest.raises(AssertionError, match="sequence built"):
        _call(model, options)
    assert built == [1]


@pytest.mark.parametrize("seam", [True, False], ids=["503d785f0", "a071826fd"])
def test_a_cache_handed_over_refuses_whatever_the_device(monkeypatch, seam):
    model, built = _bound(seam, False, monkeypatch)
    with pytest.raises(UnsupportedModelError) as caught:
        _call(model, GPU, prefix_cache={torch.device("cpu"): object()})
    assert str(caught.value) == WRITABLE_PREFIX_CACHE_REFUSAL
    assert built == []


# Real ComfyUI, in a spawned child: comfy's argument parsing is process-global.

def _comfy_dir() -> Path | None:
    raw = os.environ.get("COMFY_DIR") or os.environ.get("COMFYUI_DIR")
    if not raw:
        return None
    path = Path(raw)
    return path if (path / "comfy" / "options.py").is_file() else None


# (patcher is_dynamic, cache options, prefix_cache handed over).
REAL_CASES = {
    "auto-dynamic": (True, AUTO, False),
    "auto-static": (False, AUTO, False),
    "gpu-dynamic": (True, GPU, False),
    "off-dynamic": (True, OFF, False),
    "stray-cache": (False, GPU, True),
}


def _real_comfy_child(_rank, comfy_dir, result_file):
    sys.path.insert(0, comfy_dir)
    import comfy.options

    sys.argv = ["pytest-qwen-image21-prefix-cache", "--cpu"]
    comfy.options.args_parsing = True
    import comfy.conds
    import comfy.ops
    from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel as stock_class

    import dgx_monarch.adapters.base as base

    base.sp_world = lambda: 2
    base.sp_rank = lambda: 0

    class Reached(Exception):
        pass

    def collective(*_args, **_kwargs):
        raise Reached

    q21.sp_gather = collective
    q21.agree_prefix_cache = collective
    stock_signature = inspect.signature(stock_class._forward)
    seam = stock_forward_takes_prefix_cache(stock_class)
    torch.manual_seed(7)
    model = stock_class(
        in_channels=64, out_channels=64, num_layers=1, attention_head_dim=6,
        num_attention_heads=1, context_in_dim=4, mlp_ratio=1, axes_dims_rope=(2, 2, 2),
        operations=comfy.ops.disable_weight_init,
    )
    QwenImage21Adapter().inject_usp(model, InjectionContext(
        topology_sp=2, usp_attention=types.SimpleNamespace(effective_kernel="TORCH_FLASH")))
    model.reset_prefix_cache(True)
    bound = _explicit(model._forward.__func__)
    stock = _explicit(stock_class._forward)
    built: list = []
    build_sequence = model.build_sequence
    model.build_sequence = lambda *args: built.append(1) or build_sequence(*args)
    x, t, context = torch.randn(1, 64, 1, 3), torch.tensor([0.5]), torch.randn(1, 3, 4)
    refs, slots = [torch.randn(1, 64, 1, 2)], [1]
    outcomes = {}
    for name, (dynamic, options, handed) in REAL_CASES.items():
        model.current_patcher = types.SimpleNamespace(
            is_dynamic=lambda dynamic=dynamic: dynamic, get_free_memory=lambda _device: 1 << 60)
        built.clear()
        try:
            with torch.no_grad():
                model._forward(x, t, context, ref_latents=refs, image_slots=slots,
                               transformer_options={"qwen_image21_cache": options},
                               prefix_cache={x.device: object()} if handed else None)
            outcomes[name] = "returned"
        except UnsupportedModelError as exc:
            exact = str(exc) == WRITABLE_PREFIX_CACHE_REFUSAL
            outcomes[name] = "refused" if exact and not built else f"refused late: {exc}"
        except Reached:
            outcomes[name] = "collective"
    Path(result_file).write_text(repr((
        rebound_signature_incompatibility(stock_signature, CONTRACT),
        seam, hasattr(comfy.conds, "CONDKVCache"),
        bound == (stock if seam else [*stock, "prefix_cache"]),
        outcomes,
    )))


def test_real_comfy_admits_the_tree_and_refuses_auto_only_where_stock_moved():
    """On the writable tree auto under dynamic VRAM refuses before the sequence;
    everything else reaches the first exchange, as before 503d785f0."""
    comfy_dir = _comfy_dir()
    if comfy_dir is None:
        pytest.skip("real Comfy canary requires COMFY_DIR or COMFYUI_DIR")
    writable_tree = "class CONDKVCache" in (comfy_dir / "comfy" / "conds.py").read_text()
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as directory:
        result_file = str(Path(directory) / "result")
        mp.spawn(_real_comfy_child, args=(str(comfy_dir), result_file), nprocs=1, join=True)
        incompatibility, seam, kv_tree, names_match, outcomes = ast.literal_eval(
            Path(result_file).read_text())
    assert incompatibility is None
    assert seam is kv_tree is writable_tree
    assert names_match
    assert outcomes == {
        "auto-dynamic": "refused" if writable_tree else "collective",
        "auto-static": "collective",
        "gpu-dynamic": "collective",
        "off-dynamic": "collective",
        "stray-cache": "refused",
    }
