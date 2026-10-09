"""Check the driver's render-memory estimate and dispatch decision.

Decision tests inject the bound without importing ComfyUI. A canary compares
the production formula and its constants with ComfyUI's BaseModel.memory_required
on a shape-only instance.
"""
from __future__ import annotations

import importlib
import json
import os
import struct
import sys
from types import ModuleType, SimpleNamespace

import pytest

from dgx_monarch import mesh_safety
from dgx_monarch import render_memory_price as rmp
from dgx_monarch.mesh_safety import StockLoadCapacityError
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from dgx_monarch.render_memory_price import RenderMemoryPriceError
from module_location_helpers import from_checkout

_CANARY_FACTOR = 2.8  # a plausible memory_usage_factor; the input, not the mirror


class _DetectedConfig:
    """Stands in for comfy's detection. The price reads this one field."""

    def __init__(self, factor: float) -> None:
        self.memory_usage_factor = factor


def _comfy_module(name: str, required: bool) -> ModuleType:
    """Import a comfy module, hard-failing when COMFY_DIR names a checkout."""
    if required:
        return importlib.import_module(name)
    return pytest.importorskip(name)


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


def _from_checkout(module: object, comfy_dir: str) -> bool:
    """Does this module's file live under the checkout? Catches the names a
    real comfy import adds under no comfy prefix: nodes, node_helpers,
    execution, latent_preview, server and the rest."""
    return from_checkout(module, comfy_dir)


@pytest.fixture
def isolated_comfy_modules():
    """Give the formula canary an empty comfy namespace and take it back.

    A stub `comfy` another file leaves behind makes `import comfy.options` fail
    with "'comfy' is not a package", and a real tree imported here would stay
    loaded for every later stub-based test. The snapshot taken here is
    restored exactly: what was present comes back, and everything the canary
    imported, by comfy name or by checkout path, goes.
    """
    comfy_dir = os.path.abspath(os.environ.get("COMFY_DIR") or "../ComfyUI")
    preserved = {name: module for name, module in sys.modules.items() if _is_comfy_module(name)}
    for name in preserved:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        # Classify first, then pop: a namespace package's path re-resolves
        # through its parent in sys.modules while it is being read.
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or _from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)


def _shape_only_checkpoint(path: str) -> str:
    """Write a safetensors file whose header is the only part worth reading."""
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}
    ).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        handle.write(b"\0" * 16)
    return path


def _arm(monkeypatch, *, bound, avail, floors=9 * 2**30):
    monkeypatch.setattr(rmp, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(rmp, "env_enabled", lambda _name: False)
    monkeypatch.setattr(rmp, "comfy_render_memory_bound",
                        lambda _path, _shape: (bound, {"config": "Fake",
                                                       "memory_usage_factor": 2.8}))
    monkeypatch.setattr(rmp, "mem_available_bytes", lambda: avail)
    monkeypatch.setattr(rmp, "_floors_bytes", lambda: floors)
    return {"path": "/fake.safetensors", "unet_name": "fake.safetensors",
            "latent_shape": (1, 3, 64, 64)}


def test_an_over_box_estimate_refuses_typed_before_dispatch(monkeypatch):
    kwargs = _arm(monkeypatch, bound=137 * 2**30, avail=93 * 2**30)
    with pytest.raises(RenderMemoryPriceError) as raised:
        rmp.preflight_render_memory(**kwargs)
    text = str(raised.value)
    assert "ComfyUI's own sample-" in text and "#339" in text
    assert "docs/TROUBLESHOOTING.md #79" in text
    # Never offer a single-box render as the remedy: on 2026-09-02 one kept
    # allocating until it had to be killed to keep the host alive.
    assert "mode=local" not in text
    tag = parse_leading_refusal_tag(text)
    assert tag is not None
    assert tag.refusal_class is RefusalClass.CAPACITY
    assert tag.guard == "activation_footprint_preflight"
    assert tag.waivable is False


def test_a_world_one_render_is_priced_through_the_real_caller(
        monkeypatch, tmp_path):
    """The world-1 case, pinned where it can regress.

    The leaf is world blind, so only the caller could put a world clause back.
    This runs the driver wrapper on a model whose mesh reports world 1 and
    requires the refusal. The evidence is a world-1 render on 2026-09-02: it
    loaded on one box, reported zero usable bytes and had to be killed.
    """
    from dgx_monarch.nodes import render_preflight

    checkpoint = _shape_only_checkpoint(str(tmp_path / "big.safetensors"))
    folder_paths = ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: checkpoint
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint",
                        lambda _path: ("zimage", "bf16"))
    _arm(monkeypatch, bound=137 * 2**30, avail=93 * 2**30)

    class _Latent:
        shape = (1, 16, 1, 128, 128)

    model = SimpleNamespace(
        unet_name="big.safetensors",
        mesh=SimpleNamespace(world=1, worker_args={}))
    with pytest.raises(RenderMemoryPriceError) as raised:
        render_preflight.activation_footprint_preflight_for_request(
            model, {}, {"samples": _Latent()})
    tag = parse_leading_refusal_tag(str(raised.value))
    assert tag is not None
    assert tag.refusal_class is RefusalClass.CAPACITY
    assert tag.guard == "activation_footprint_preflight"
    assert tag.waivable is False


def test_the_price_raises_its_own_type_not_the_stock_load_family(monkeypatch):
    """A refusal inside a ceremony leg must not read as a load that failed.

    ``nodes/gate_cross_mode.run_cross_residency_reference`` records "cannot
    LOAD under stock residency" for that family, and
    ``nodes/gate_ceremony.gather_ceremony_evidence`` converts it to an aborted
    proof. The weights load here; the estimate is what does not fit.
    """
    kwargs = _arm(monkeypatch, bound=137 * 2**30, avail=93 * 2**30)
    with pytest.raises(RenderMemoryPriceError) as raised:
        rmp.preflight_render_memory(**kwargs)
    exc = raised.value
    assert not isinstance(exc, StockLoadCapacityError)
    assert mesh_safety.is_stock_load_capacity_error(exc) is False


def test_comfy_reserve_is_counted_once_at_the_24gib_boundary(monkeypatch):
    reserve = 24 * 2**30
    mm = SimpleNamespace(
        extra_reserved_memory=lambda: reserve,
        minimum_inference_memory=lambda: int(0.8 * 2**30) + reserve,
    )
    comfy = ModuleType("comfy")
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    assert rmp._floors_bytes() == int(0.8 * 2**30) + reserve
    kwargs = _arm(monkeypatch, bound=int(0.2 * 2**30), avail=25 * 2**30,
                  floors=rmp._floors_bytes())
    rmp.preflight_render_memory(**kwargs)
    monkeypatch.setattr(rmp, "mem_available_bytes", lambda: 25 * 2**30 - 2)
    with pytest.raises(RenderMemoryPriceError):
        rmp.preflight_render_memory(**kwargs)


def test_a_fitting_estimate_passes(monkeypatch):
    rmp.preflight_render_memory(**_arm(monkeypatch, bound=5 * 2**30, avail=93 * 2**30))


def test_the_disabled_env_passes_without_estimating(monkeypatch):
    """The documented env var is the only escape: with it set, even an over-box
    estimate passes."""
    kwargs = _arm(monkeypatch, bound=137 * 2**30, avail=1 * 2**30)
    monkeypatch.setattr(rmp, "env_enabled", lambda _name: True)
    rmp.preflight_render_memory(**kwargs)


def test_a_broken_estimator_fails_open(monkeypatch):
    kwargs = _arm(monkeypatch, bound=137 * 2**30, avail=1 * 2**30)
    monkeypatch.setattr(rmp, "comfy_render_memory_bound",
                        lambda _p, _s: (_ for _ in ()).throw(RuntimeError("boom")))
    rmp.preflight_render_memory(**kwargs)


def test_the_bound_mirrors_comfys_own_formula(monkeypatch, tmp_path, isolated_comfy_modules):
    """Canary: the production bound equals comfy's flash-path formula."""
    named = os.environ.get("COMFY_DIR")
    comfy_dir = os.path.abspath(named or "../ComfyUI")
    if not os.path.isdir(comfy_dir):
        if named is not None:
            # A named COMFY_DIR is the canary job's own contract. Skipping there
            # reports the step green without ever comparing anything.
            pytest.fail(f"COMFY_DIR names no ComfyUI checkout: {comfy_dir}")
        pytest.skip("no ComfyUI checkout (the comfy-canary workflow covers it)")
    required = named is not None
    monkeypatch.syspath_prepend(comfy_dir)
    # model_base reaches cli_args through model_management.  Make Comfy choose
    # its CPU-safe path before its first import; this remains a formula canary,
    # not an accelerator or xFuser import test.
    monkeypatch.setattr(sys, "argv", ["pytest-render-memory-price", "--cpu"])
    options = _comfy_module("comfy.options", required)
    options.enable_args_parsing()
    comfy_mb = _comfy_module("comfy.model_base", required)
    comfy_mm = _comfy_module("comfy.model_management", required)
    # Hold the upstream API contract that made the 24-GiB hardware refusal
    # wrong: minimum_inference_memory includes the configured reserve once.
    with monkeypatch.context() as reserve_scope:
        reserve_scope.setattr(comfy_mm, "EXTRA_RESERVED_VRAM", 0)
        baseline_floor = comfy_mm.minimum_inference_memory()
        reserve_scope.setattr(comfy_mm, "EXTRA_RESERVED_VRAM", 24 * 2**30)
        assert comfy_mm.minimum_inference_memory() == baseline_floor + 24 * 2**30
        assert rmp._floors_bytes() == int(baseline_floor + 24 * 2**30)
    mm = _comfy_module("comfy.model_management", required)
    detection = _comfy_module("comfy.model_detection", required)
    import torch

    inst = object.__new__(comfy_mb.BaseModel)
    # Bypass nn.Module.__setattr__: this shape-only instance never runs
    # BaseModel.__init__ and builds no diffusion model.
    object.__setattr__(inst, "memory_usage_factor", _CANARY_FACTOR)
    object.__setattr__(inst, "memory_usage_factor_conds", ())
    object.__setattr__(inst, "memory_usage_shape_process", {})
    object.__setattr__(inst, "get_dtype_inference", lambda: torch.bfloat16)
    # Hosted CPU runners usually report neither capability. The canary checks
    # the flash/xformers formula the production bound uses, so force that
    # branch instead of letting host probing skip it.
    monkeypatch.setattr(mm, "xformers_enabled", lambda: False)
    monkeypatch.setattr(mm, "pytorch_attention_flash_attention", lambda: True)
    shape = [1, 3, 64, 64]  # the single batch of minimum_memory_required
    theirs = inst.memory_required(shape)

    # The mirrored arithmetic, read from the production constants. A hand-typed
    # copy of 2 or of one megabyte passes this line through any edit to them.
    area = shape[0] * shape[2] * shape[3]
    ours = area * rmp._MIN_DTYPE_BYTES * 0.01 * _CANARY_FACTOR * rmp._MB
    assert ours == pytest.approx(theirs, rel=1e-6)

    # And the leaf itself, from a real header. Comfy's detection is the one
    # part a header-only file cannot drive, so pin that name (a rename fails
    # here) and let the header parse, the area walk and the bound run for real.
    monkeypatch.setattr(detection, "model_config_from_unet",
                        lambda *_args, **_kwargs: _DetectedConfig(_CANARY_FACTOR))
    found = rmp.comfy_render_memory_bound(
        _shape_only_checkpoint(str(tmp_path / "price.safetensors")), shape)
    assert found is not None, (
        "comfy_render_memory_bound returned None. It fails open by design, so "
        "an upstream rename retires the whole preflight in silence")
    bound, detail = found
    assert bound == pytest.approx(theirs, rel=1e-6)
    assert detail["dtype_bytes"] == rmp._MIN_DTYPE_BYTES
    assert detail["memory_usage_factor"] == _CANARY_FACTOR
