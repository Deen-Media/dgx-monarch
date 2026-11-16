"""Every unload of ComfyUI's models forgets the driver models a loader charged.

The loader's memo of charged driver models (``loader_graph._CHARGED_STACK_ARTIFACTS``)
lets a second loader node credit a stack an earlier one charged. Once ComfyUI's
models are unloaded, nothing it held is loaded, so no earlier charge stands for
them. Review of the 1.5.0 promotion, finding c1#3: only the loader's own unload
emptied the memo. The render-site unload (``driver_unload._after_refusal``, which
the render-site checks, the pending-load relief, the driver spend price and the
gate's stock reference reach) and a Clear VRAM node at ``hard`` left it standing,
so the next loader with nothing held charged the whole stack at 0.00 GiB.

CPU only, over the rig in tests/test_graph_load_price.py and the model list of
tests/test_requeue_price.py, whose ``unload_all_models`` empties it.
"""
from __future__ import annotations

import pytest

from dgx_monarch import driver_footprint, driver_unload, mesh_safety, telemetry
from dgx_monarch.nodes import loader_preflight, ops
from dgx_monarch.refusal import RefusalClass, refusal
from test_graph_load_price import (  # noqa: F401  # the rig is an autouse fixture
    _GIB,
    DIT,
    TE,
    _Mesh,
    _rig,
    _switch_id,
    _template,
)
from test_requeue_price import _comfy_holds


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    """No render running in this driver, no settle wait, no unload switch."""
    monkeypatch.setattr(telemetry, "render_progress", telemetry.RenderProgress())
    monkeypatch.setattr(driver_unload, "_SETTLE_S", 0.0)
    monkeypatch.delenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, raising=False)


def _graph() -> dict:
    graph = _template()
    graph[_switch_id(graph)]["inputs"]["switch"] = False
    return graph


def _first_queue(monkeypatch) -> _Mesh:
    """The loader admits the switch-off graph and charges its encoder and VAE; the encode then
    loads the encoder, which ComfyUI holds."""
    mesh = _Mesh()
    assert loader_preflight.preflight_loader_footprint(mesh, DIT, {}, _graph()) is None
    return mesh


def _refusing() -> None:
    raise mesh_safety.StockLoadCapacityError(refusal(
        RefusalClass.CAPACITY, "probe preflight: short.", guard="activation_footprint_preflight",
        waivable=False))


def _second_queue_refuses(mesh: _Mesh, monkeypatch) -> str:
    """The next queue runs the loader again (an Init widget changed) with nothing held, on a box
    with room for the resident DiT's floor and not for the encoder and the VAE beside it."""
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 15 * _GIB)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, DIT, {}, _graph())
    return str(raised.value)


def test_a_render_site_unload_forgets_what_the_loader_charged(monkeypatch):
    mesh = _first_queue(monkeypatch)
    calls = _comfy_holds(monkeypatch, TE)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as refused:
        driver_unload.check_with_unload([_refusing], subject="this render")
    assert calls.unload == 1 and "Before refusing, the driver unloaded the 1 model" in str(refused.value)
    text = _second_queue_refuses(mesh, monkeypatch)
    assert "need 20.0 GiB on this box and 15.0 GiB is available" in text
    assert "an earlier loader node charged this stack" not in text
    assert "text encoder 9.8 GiB (p-h3-image_text_encoder_bf16.safetensors" in text


def test_a_clear_vram_node_at_hard_forgets_what_the_loader_charged(monkeypatch):
    mesh = _first_queue(monkeypatch)
    calls = _comfy_holds(monkeypatch, TE)
    assert ops._free_driver_models("hard")["unloaded_models"] is True
    assert calls.unload == 1
    text = _second_queue_refuses(mesh, monkeypatch)
    assert "text encoder 9.8 GiB (p-h3-image_text_encoder_bf16.safetensors" in text


def test_a_clear_vram_node_at_hard_with_nothing_held_keeps_the_charge(monkeypatch):
    """The node unloads through the driver's own unload, so it keeps the rule the render-site
    unload keeps: where no model left ComfyUI, a charge still to load stands."""
    from dgx_monarch.nodes import loader_graph

    _first_queue(monkeypatch)
    calls = _comfy_holds(monkeypatch)
    assert ops._free_driver_models("hard")["unloaded_models"] is True
    assert calls.unload == 1 and TE in loader_graph._CHARGED_STACK_ARTIFACTS


def test_an_unload_that_frees_nothing_keeps_the_charge(monkeypatch):
    """ComfyUI held nothing to unload, so the encoder the loader charged is still to load in this
    prompt, and the next loader node credits it as before."""
    from dgx_monarch.nodes import loader_graph

    _first_queue(monkeypatch)
    calls = _comfy_holds(monkeypatch)
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        driver_unload.check_with_unload([_refusing], subject="this render")
    assert calls.unload == 0 and TE in loader_graph._CHARGED_STACK_ARTIFACTS
