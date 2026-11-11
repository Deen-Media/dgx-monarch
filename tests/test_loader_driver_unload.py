"""The loader frees ComfyUI's own models before it refuses (nodes/loader_relief.py).

The render-site checks have done this since issue #56: when a capacity check
would refuse and ComfyUI holds a model on the driver host, the driver unloads
them through ComfyUI's own model management, reads MemAvailable again and
prices again. The loader preflight did not, so a session that had rendered one
graph met the loader's refusal for the next with the first graph's encoders
still counted as spent.

At the loader the unload comes second, after the swap credit
(tests/test_loader_swap_credit.py), and these tests hold its walls: it runs
once, only when the price would otherwise refuse, never while a render is
running in this driver, and never past the kill switch; afterwards the whole
load is priced again from a fresh reading; and a refusal that still stands
says what was unloaded and what was kept.
"""
from __future__ import annotations

import logging

import pytest

import test_capacity_old_model as cards
import test_driver_unload as unloads
from dgx_monarch import capacity_agreement, consent_pending, driver_footprint, driver_unload, mesh_safety, telemetry
from dgx_monarch.log import get_logger
from dgx_monarch.nodes import loader_graph, loader_preflight
from test_capacity_old_model import NEW, _answer, _loader_rig, _stock
from test_driver_unload import TE_BYTES, VAE_BYTES, H3TextEncoder, H3VideoVAE, _comfy
from test_loader_swap_credit import _held

# The old-model rig (stub folder_paths, clean cards) and the unload rig (fresh
# progress tracker, no settle wait, no kill switch, the box's reading).
_rig, _isolated, box = cards._rig, unloads._isolated, unloads.box

_GIB = 2 ** 30
OLD = "longcat_image_bf16.safetensors"


@pytest.fixture
def journal():
    """The relief module's log records; the pack's loggers do not propagate."""
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    logger = get_logger("dgx_monarch.nodes.loader_relief")
    logger.addHandler(handler)
    yield records
    logger.removeHandler(handler)


def _rigged(monkeypatch, tmp_path, box, *, held_gib: float | None, avail_gib: float = 40.0):
    """The loader rig (a 20 GiB file: 47 GiB at the load window) on a box whose reading moves."""
    mesh = _loader_rig(monkeypatch, tmp_path, held_gib=held_gib or 1.0)
    row = (_held(OLD, held_gib)
           if held_gib else _stock("fits", size_gib=1.0, avail_gib=200.0, resident=None))
    asked: list = []
    monkeypatch.setattr(capacity_agreement, "collect", lambda *_a, **_k:
                        asked.append(1) or [_answer(0, row, box="this box")])
    box.mem = int(avail_gib * _GIB)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: box.mem)
    return mesh, asked


def _cards() -> list[str]:
    return [card["kind"] for card in consent_pending.pending_cards()]


def test_the_load_proceeds_when_the_unload_makes_room(monkeypatch, tmp_path, box, journal):
    """Nothing to drop in the slot, 7 GiB short, and ComfyUI holds 22.1 GiB of another graph's models."""
    mesh, _asked = _rigged(monkeypatch, tmp_path, box, held_gib=None)
    calls = _comfy(monkeypatch, box)
    assert loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond") is None
    assert calls.unload == 1 and calls.mm.current_loaded_models == [] and _cards() == []
    lines = [record.getMessage() for record in journal]
    assert len(lines) == 1 and "fits after unloading the 2 models (H3TextEncoder, H3VideoVAE)" in lines[0]
    assert "from 40.0 GiB to 62.1 GiB" in lines[0]


def test_the_credit_comes_first_and_spares_the_unload(monkeypatch, tmp_path, box, journal):
    mesh, _asked = _rigged(monkeypatch, tmp_path, box, held_gib=12.0)
    calls = _comfy(monkeypatch, box)
    assert loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond") is None
    assert calls.unload == 0, "the drop alone made room: ComfyUI keeps its models"
    assert len(calls.mm.current_loaded_models) == 2
    assert ["drops longcat_image_bf16" in record.getMessage() for record in journal] == [True]


def test_a_load_that_fits_unloads_nothing(monkeypatch, tmp_path, box):
    mesh, asked = _rigged(monkeypatch, tmp_path, box, held_gib=None, avail_gib=60.0)
    calls = _comfy(monkeypatch, box)
    assert loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond") is None
    assert calls.unload == 0 and asked == []


def test_the_credit_and_the_unload_add_up(monkeypatch, tmp_path, box, journal):
    """47 GiB needed: 40 read, 4 to be dropped, and a 5.4 GiB VAE ComfyUI can reload."""
    mesh, _asked = _rigged(monkeypatch, tmp_path, box, held_gib=4.0)
    calls = _comfy(monkeypatch, box, models=((H3VideoVAE, VAE_BYTES),))
    assert loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond") is None
    assert calls.unload == 1 and _cards() == []
    lines = [record.getMessage() for record in journal]
    assert any("fits after unloading the 1 model (H3VideoVAE)" in line for line in lines)
    assert any("drops longcat_image_bf16, which frees at least 4.0 GiB; MemAvailable read 45.4 GiB" in line
               for line in lines), "the credit is counted on the reading after the unload"


def test_a_load_still_short_refuses_once_and_says_what_was_counted_and_unloaded(
        monkeypatch, tmp_path, box):
    """A true shortage: 1 GiB to drop and a 1 GiB model to unload against 7 GiB missing."""
    mesh, asked = _rigged(monkeypatch, tmp_path, box, held_gib=1.0)
    calls = _comfy(monkeypatch, box, models=((H3VideoVAE, 1 * _GIB),))
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond")
    assert calls.unload == 1, "one unload a load, never a second"
    assert len(asked) == 2, "the Workers are asked again on the reading after the unload"
    text = str(raised.value)
    assert "needs 47.0 GiB on this box and 41.0 GiB is available, 6.0 GiB short." in text
    assert ("The Worker on this box drops longcat_image_bf16 before this load, which frees at "
            "least 1.0 GiB, and qwen_image_2512_bf16 is still 5.0 GiB short after that.") in text
    assert ("Before refusing, the driver unloaded the 1 model (H3VideoVAE) ComfyUI held on this "
            "host: MemAvailable read 40.0 GiB before the unload and 41.0 GiB after it, and the "
            "figures above are the reading after it.") in text


def test_a_running_render_keeps_its_models_and_the_refusal_says_so(monkeypatch, tmp_path, box):
    mesh, _asked = _rigged(monkeypatch, tmp_path, box, held_gib=None)
    calls = _comfy(monkeypatch, box)
    telemetry.render_progress.start(10, token=object())
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond")
    assert calls.unload == 0 and len(calls.mm.current_loaded_models) == 2
    assert ("The driver did not unload the 2 models ComfyUI holds on this host before refusing, "
            "because a render is still running in this driver; queue again once that work "
            "ends.") in str(raised.value)


def test_the_kill_switch_keeps_the_first_reading(monkeypatch, tmp_path, box):
    mesh, _asked = _rigged(monkeypatch, tmp_path, box, held_gib=None)
    calls = _comfy(monkeypatch, box)
    monkeypatch.setenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, "1")
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond")
    assert calls.unload == 0 and "unload" not in str(raised.value)


def test_a_loader_that_names_no_slot_still_unloads(monkeypatch, tmp_path, box):
    """The unload needs no slot; only the credit does."""
    mesh, _asked = _rigged(monkeypatch, tmp_path, box, held_gib=None)
    calls = _comfy(monkeypatch, box)
    assert loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None) is None
    assert calls.unload == 1


def test_an_unload_forgets_the_driver_models_an_earlier_loader_charged(monkeypatch, tmp_path, box):
    """Nothing ComfyUI held is loaded any more, so no earlier charge may stand in for it."""
    mesh, _asked = _rigged(monkeypatch, tmp_path, box, held_gib=None)
    _comfy(monkeypatch, box)
    loader_graph.remember(loader_graph._CHARGED_STACK_ARTIFACTS, "some_encoder.safetensors")
    assert loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond") is None
    assert "some_encoder.safetensors" not in loader_graph._CHARGED_STACK_ARTIFACTS


def test_an_unload_that_frees_nothing_is_not_priced_again(monkeypatch, tmp_path, box):
    """ComfyUI kept every model: one pass, the first reading, no sentence about an unload."""
    mesh, asked = _rigged(monkeypatch, tmp_path, box, held_gib=None)
    calls = _comfy(monkeypatch, box, kept=(H3TextEncoder, H3VideoVAE))
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond")
    assert calls.unload == 1 and len(asked) == 1 and "unloaded" not in str(raised.value)
    assert TE_BYTES and "40.0 GiB is available" in str(raised.value)
