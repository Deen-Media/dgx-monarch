"""The driver unloads ComfyUI's own models before a render-site capacity refusal.

CPU only. ComfyUI's model management is a stub in ``sys.modules`` whose
``unload_all_models`` empties its model list and raises the fake MemAvailable
reading by what those models held, so every verdict below is the one the
re-read decides. Sources: issue #93 item 2, and the back-to-back MiniMax H3
refusal it quotes (35.6 GiB needed per rank, 33.3 GiB usable, 38.3 GiB
MemAvailable minus a 5.0 GiB reserve) on a two-Spark pair, where ComfyUI's text
encoder and VAE sat inside the driver host's reading.
"""
from __future__ import annotations

import ast
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import (
    driver_footprint,
    driver_unload,
    h3_activation,
    h3_rows,
    mesh_safety,
    render_memory_price,
    telemetry,
)
from dgx_monarch.log import get_logger
from dgx_monarch.nodes import render_preflight
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag, refusal
from dgx_monarch.render_memory_price import RenderMemoryPriceError

SRC = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"

_GIB = 2 ** 30
GUARD = "activation_footprint_preflight"

# What ComfyUI held on the driver host after an H3 render: the text encoder
# and the video VAE, at the sizes tests/test_h3_activation_preflight.py uses.
TE_BYTES = int(16.7 * _GIB)
VAE_BYTES = int(5.4 * _GIB)


class H3TextEncoder:
    """Stands in for the class name ComfyUI logs for a loaded text encoder."""


class H3VideoVAE:
    """Stands in for the class name ComfyUI logs for a loaded VAE."""


class _Entry:
    """comfy's LoadedModel: ``model`` dereferences a weak reference to the
    patcher, so a model that is gone reads as None."""

    def __init__(self, inner: object | None, held: int) -> None:
        self._patcher = None if inner is None else SimpleNamespace(model=inner)
        self.held = held

    @property
    def model(self):
        return self._patcher


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """A fresh progress tracker, no settle wait, no kill switch, and a UMA box."""
    monkeypatch.setattr(telemetry, "render_progress", telemetry.RenderProgress())
    monkeypatch.setattr(driver_unload, "_SETTLE_S", 0.0)
    monkeypatch.delenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, raising=False)
    monkeypatch.delenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, raising=False)
    monkeypatch.delenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, raising=False)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    # The render memory price binds the probe by name; patched too, so no
    # test here asks torch about a device.
    monkeypatch.setattr(render_memory_price, "gpu_is_integrated", lambda: True)


@pytest.fixture
def box(monkeypatch):
    """The driver host's MemAvailable, read through the one shared probe."""
    state = SimpleNamespace(mem=0)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: state.mem)
    return state


def _comfy(monkeypatch, box, *, models=((H3TextEncoder, TE_BYTES), (H3VideoVAE, VAE_BYTES)),
           kept=(), error: Exception | None = None) -> SimpleNamespace:
    """Install the model-management stub. ``kept`` names classes it refuses
    to unload; ``error`` makes the unload raise."""
    mm = types.ModuleType("comfy.model_management")
    mm.current_loaded_models = [
        _Entry(cls() if cls is not None else None, held) for cls, held in models]
    calls = SimpleNamespace(unload=0, soft=0, mm=mm)

    def unload_all_models():
        calls.unload += 1
        if error is not None:
            raise error
        stays = [entry for entry in mm.current_loaded_models
                 if entry.model is not None and type(entry.model.model) in kept]
        box.mem += sum(entry.held for entry in mm.current_loaded_models
                       if entry not in stays and entry.model is not None)
        mm.current_loaded_models[:] = stays

    def soft_empty_cache():
        calls.soft += 1

    mm.unload_all_models = unload_all_models
    mm.soft_empty_cache = soft_empty_cache
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    return calls


def _priced(box, need: int, *, seen: list | None = None, label: str = "probe",
            error: type[Exception] = mesh_safety.StockLoadCapacityError):
    """A check that refuses below ``need`` bytes, in the shape the real ones
    raise: tagged class C, budget first."""
    def check():
        if seen is not None:
            seen.append(label)
        if box.mem < need:
            raise error(refusal(
                RefusalClass.CAPACITY,
                f"{label} preflight: {need / _GIB:.1f} GiB needed, "
                f"{box.mem / _GIB:.1f} GiB MemAvailable.",
                guard=GUARD, waivable=False))
    return check


@pytest.fixture
def journal():
    """This module's log records. The loggers do not propagate, so the
    handler goes on the module logger directly."""
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    logger = get_logger("dgx_monarch.driver_unload")
    logger.addHandler(handler)
    yield records
    logger.removeHandler(handler)


# 1. The decision, over stand-in checks.

def test_a_refusal_with_nothing_to_unload_is_unchanged(monkeypatch, box):
    box.mem = 20 * _GIB
    calls = _comfy(monkeypatch, box, models=())
    check = _priced(box, 30 * _GIB)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as first:
        check()
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload([check])
    assert str(excinfo.value) == str(first.value)
    assert calls.unload == 0 and calls.soft == 0
    assert box.mem == 20 * _GIB


def test_a_refusal_off_a_comfy_process_is_unchanged(monkeypatch, box):
    """No ComfyUI in the process: nothing is held, so nothing changes."""
    box.mem = 20 * _GIB
    monkeypatch.setitem(sys.modules, "comfy", None)
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload([_priced(box, 30 * _GIB)])
    assert "unload" not in str(excinfo.value)


def test_entries_whose_model_is_gone_are_not_unloadable(monkeypatch, box):
    box.mem = 20 * _GIB
    calls = _comfy(monkeypatch, box, models=((None, TE_BYTES),))
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        driver_unload.check_with_unload([_priced(box, 30 * _GIB)])
    assert calls.unload == 0


def test_the_render_proceeds_when_the_unload_makes_room(monkeypatch, box, journal):
    """The issue's case: 2.3 GiB short with the text encoder and VAE held."""
    box.mem = int(38.3 * _GIB)
    calls = _comfy(monkeypatch, box)
    driver_unload.check_with_unload(
        [_priced(box, int(40.6 * _GIB))], subject="h3.safetensors")
    assert calls.unload == 1 and calls.soft == 1
    assert calls.mm.current_loaded_models == []
    lines = [record.getMessage() for record in journal]
    assert len(lines) == 1, lines
    assert journal[0].levelno == logging.INFO
    for token in ("h3.safetensors", "2 models (H3TextEncoder, H3VideoVAE)",
                  "from 38.3 GiB to 60.4 GiB", "a gain of 22.1 GiB"):
        assert token in lines[0], lines[0]


def test_a_render_still_short_refuses_with_the_reading_after_the_unload(monkeypatch, box):
    box.mem = 10 * _GIB
    calls = _comfy(monkeypatch, box)
    seen: list = []
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload([_priced(box, 60 * _GIB, seen=seen)])
    message = str(excinfo.value)
    assert calls.unload == 1
    assert seen == ["probe", "probe"]
    # The budget at the front is the second reading, and the tag still leads.
    assert "60.0 GiB needed, 32.1 GiB MemAvailable." in message[:200]
    tag = parse_leading_refusal_tag(message)
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY
    assert tag.guard == GUARD and tag.waivable is False
    assert message.endswith(
        "Before refusing, the driver unloaded the 2 models (H3TextEncoder, "
        "H3VideoVAE) ComfyUI held on this host: MemAvailable read 10.0 GiB "
        "before the unload and 32.1 GiB after it, and the figures above are "
        "the reading after it."), message


def test_the_unload_runs_once_and_never_repeats_an_earlier_check(monkeypatch, box):
    """The checks before the refusing one passed on a lower reading, so the
    run restarts at the refusing one, and a refusal after the unload is final."""
    box.mem = 30 * _GIB
    calls = _comfy(monkeypatch, box)
    seen: list = []
    steps = [_priced(box, 20 * _GIB, seen=seen, label="first"),
             _priced(box, 40 * _GIB, seen=seen, label="second"),
             _priced(box, 90 * _GIB, seen=seen, label="third")]
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload(steps)
    assert seen == ["first", "second", "second", "third"]
    assert calls.unload == 1
    assert str(excinfo.value).startswith(
        "[dgxm:C guard=activation_footprint_preflight waivable=0] third preflight")


def test_a_render_memory_refusal_gets_the_same_unload(monkeypatch, box):
    """The render memory refusal is not a stock-load error on purpose; it
    reads the same host and gets the same chance."""
    box.mem = 20 * _GIB
    calls = _comfy(monkeypatch, box)
    driver_unload.check_with_unload([_priced(box, 30 * _GIB, error=RenderMemoryPriceError)])
    assert calls.unload == 1
    box.mem = 0
    calls = _comfy(monkeypatch, box)
    with pytest.raises(RenderMemoryPriceError) as excinfo:
        driver_unload.check_with_unload([_priced(box, 90 * _GIB, error=RenderMemoryPriceError)])
    assert type(excinfo.value) is RenderMemoryPriceError
    assert "Before refusing, the driver unloaded" in str(excinfo.value)


def test_a_driver_footprint_refusal_keeps_its_type(monkeypatch, box):
    box.mem = 0
    _comfy(monkeypatch, box)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as excinfo:
        driver_unload.check_with_unload([_priced(
            box, 90 * _GIB, error=driver_footprint.DriverFootprintCapacityError)])
    assert type(excinfo.value) is driver_footprint.DriverFootprintCapacityError
    assert mesh_safety.is_stock_load_capacity_error(excinfo.value)


def test_a_model_comfy_keeps_is_not_named(monkeypatch, box):
    box.mem = 0
    _comfy(monkeypatch, box, kept=(H3VideoVAE,))
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload([_priced(box, 90 * _GIB)])
    assert "the 1 model (H3TextEncoder) ComfyUI held" in str(excinfo.value)


def test_an_unload_that_frees_no_model_refuses_on_the_first_reading(monkeypatch, box):
    box.mem = 20 * _GIB
    _comfy(monkeypatch, box, kept=(H3TextEncoder, H3VideoVAE))
    seen: list = []
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload([_priced(box, 30 * _GIB, seen=seen)])
    assert seen == ["probe"]
    assert "unload" not in str(excinfo.value)


def test_an_unload_that_fails_still_refuses(monkeypatch, box):
    """A failed unload never turns into a pass."""
    box.mem = 20 * _GIB
    calls = _comfy(monkeypatch, box, error=RuntimeError("comfy broke"))
    seen: list = []
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload([_priced(box, 30 * _GIB, seen=seen)])
    assert calls.unload == 1 and seen == ["probe"]
    message = str(excinfo.value)
    assert "20.0 GiB MemAvailable." in message[:200]
    assert message.endswith(
        "The driver tried to unload the 2 models (H3TextEncoder, H3VideoVAE) "
        "ComfyUI holds on this host before refusing, and the unload failed "
        "(RuntimeError), so the figures above are the reading before it.")


def test_the_kill_switch_keeps_the_first_refusal(monkeypatch, box):
    box.mem = 20 * _GIB
    calls = _comfy(monkeypatch, box)
    monkeypatch.setenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, "1")
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload([_priced(box, 30 * _GIB)])
    assert calls.unload == 0
    assert "unload" not in str(excinfo.value)


@pytest.mark.parametrize("value", ["0", "false", "off", ""])
def test_an_off_spelling_leaves_the_unload_on(monkeypatch, box, value):
    box.mem = 20 * _GIB
    calls = _comfy(monkeypatch, box)
    monkeypatch.setenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, value)
    driver_unload.check_with_unload([_priced(box, 30 * _GIB)])
    assert calls.unload == 1


def test_any_other_error_passes_through_without_an_unload(monkeypatch, box):
    calls = _comfy(monkeypatch, box)

    def broken():
        raise ValueError("not a capacity answer")

    with pytest.raises(ValueError, match="not a capacity answer"):
        driver_unload.check_with_unload([broken])
    assert calls.unload == 0


# 2. A running render is never unloaded.

def _leased(**fields) -> SimpleNamespace:
    return SimpleNamespace(sample_leases={}, abandoned_sample_leases={}, **fields)


@pytest.mark.parametrize(("busy", "reason"), [
    ("progress", "a render is still running in this driver"),
    ("lease", "a render result is still held on the mesh"),
    ("abandoned", "an abandoned render may still be running on the Workers"),
    ("dispatch", "a call to the Workers is still in flight"),
    ("unreadable", "the driver could not read whether a render is running"),
])
def test_a_running_render_is_never_unloaded(monkeypatch, box, busy, reason):
    box.mem = 20 * _GIB
    calls = _comfy(monkeypatch, box)
    handle = _leased()
    if busy == "progress":
        telemetry.render_progress.start(10, token=object())
    elif busy == "lease":
        handle.sample_leases = {3: 1}
    elif busy == "abandoned":
        handle.abandoned_sample_leases = {2: 1}
    elif busy == "dispatch":
        from dgx_monarch import mesh_setup_state
        handle.setup_cleanup_state = SimpleNamespace(
            outcome=mesh_setup_state.SetupCleanupOutcome.IN_PROGRESS)
    else:
        handle.sample_leases = {3: "unreadable"}
    seen: list = []
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload([_priced(box, 30 * _GIB, seen=seen)], handle=handle)
    assert calls.unload == 0 and calls.soft == 0
    assert seen == ["probe"]
    assert len(calls.mm.current_loaded_models) == 2
    message = str(excinfo.value)
    assert "20.0 GiB MemAvailable." in message[:200]
    assert message.endswith(
        "The driver did not unload the 2 models (H3TextEncoder, H3VideoVAE) "
        f"ComfyUI holds on this host before refusing, because {reason}; queue "
        "again once that work ends."), message


def test_a_finished_render_frees_the_tracker_for_the_next_unload(monkeypatch, box):
    box.mem = 20 * _GIB
    calls = _comfy(monkeypatch, box)
    token = telemetry.render_progress.start(10, token=object())
    telemetry.render_progress.finish(token)
    driver_unload.check_with_unload([_priced(box, 30 * _GIB)], handle=_leased())
    assert calls.unload == 1




# 3. The actors' caches go last.
#
# The 2026-10-29 second request of a 15 second H3 session: the driver unloaded
# its two VAEs (MemAvailable 51.5 to 56.5 GiB) and refused 8.7 GiB short while
# the actor on the head still held the first render's 17.6 GiB of activations.

class _Mesh:
    """A live mesh whose setup completed. ``clear_vram`` at ``soft`` hands the
    box back what the actor on this host kept in its allocator cache."""

    def __init__(self, box, *, cache: int = 0, world: int = 2,
                 error: Exception | None = None, events: list | None = None,
                 **fields) -> None:
        self.box, self.cache, self.world, self.error = box, cache, world, error
        self.events = [] if events is None else events
        self.calls: list = []
        self.setup_key = ("setup",)
        self.defunct = False
        self.teardown_complete = False
        self.setup_cleanup_state = None
        self.sample_leases: dict = {}
        self.abandoned_sample_leases: dict = {}
        for name, value in fields.items():
            setattr(self, name, value)

    def call_all(self, endpoint, *args, timeout_s):
        self.calls.append((endpoint, args, timeout_s))
        self.events.append("trim")
        if self.error is not None:
            raise self.error
        self.box.mem += self.cache
        self.cache = 0
        return [{"rank": rank, "models": {}} for rank in range(self.world)]


def _in_order(calls: SimpleNamespace, events: list) -> None:
    """Record the stub's unload in ``events``, beside the mesh's trim."""
    inner = calls.mm.unload_all_models

    def unload_all_models():
        events.append("unload")
        inner()

    calls.mm.unload_all_models = unload_all_models


TRIM = ("clear_vram", ("soft",), driver_unload._TRIM_TIMEOUT_S)


def test_a_check_that_passes_frees_nothing(monkeypatch, box):
    box.mem = 90 * _GIB
    calls = _comfy(monkeypatch, box)
    mesh = _Mesh(box, cache=int(17.6 * _GIB))
    driver_unload.check_with_unload([_priced(box, 60 * _GIB)], handle=mesh)
    assert calls.unload == 0 and mesh.calls == []


def test_the_trim_never_runs_when_the_unload_makes_room(monkeypatch, box):
    box.mem = 30 * _GIB
    calls = _comfy(monkeypatch, box)
    mesh = _Mesh(box, cache=int(17.6 * _GIB))
    driver_unload.check_with_unload([_priced(box, 50 * _GIB)], handle=mesh)
    assert calls.unload == 1 and mesh.calls == []


def test_the_actors_release_their_caches_when_the_unload_falls_short(
        monkeypatch, box, journal):
    """Driver unload first, the actor trim second, and the check once after each."""
    box.mem = 30 * _GIB
    events: list = []
    calls = _comfy(monkeypatch, box)
    _in_order(calls, events)
    mesh = _Mesh(box, cache=int(17.6 * _GIB), events=events)
    seen: list = []
    driver_unload.check_with_unload(
        [_priced(box, 60 * _GIB, seen=seen)], handle=mesh, subject="h3.safetensors")
    assert events == ["unload", "trim"]
    assert seen == ["probe", "probe", "probe"]
    assert mesh.calls == [TRIM]
    lines = [record.getMessage() for record in journal]
    assert len(lines) == 1, lines
    assert journal[0].levelno == logging.INFO
    for token in ("actor trim: h3.safetensors fits after the driver unloaded the 2 "
                  "models (H3TextEncoder, H3VideoVAE) ComfyUI held on this host "
                  "(MemAvailable 30.0 GiB to 52.1 GiB)",
                  "and 2 actors released their allocator caches",
                  "from 52.1 GiB to 69.7 GiB across the trim, a gain of 17.6 GiB"):
        assert token in lines[0], lines[0]


@pytest.mark.parametrize("error", [
    mesh_safety.StockLoadCapacityError, RenderMemoryPriceError,
    driver_footprint.DriverFootprintCapacityError,
])
def test_a_refusal_after_the_trim_names_the_unload_and_the_trim(
        monkeypatch, box, journal, error):
    box.mem = 10 * _GIB
    _comfy(monkeypatch, box)
    mesh = _Mesh(box, cache=4 * _GIB)
    with pytest.raises(error) as excinfo:
        driver_unload.check_with_unload([_priced(box, 60 * _GIB, error=error)],
                                        handle=mesh, subject="h3.safetensors")
    assert type(excinfo.value) is error
    message = str(excinfo.value)
    assert mesh.calls == [TRIM]
    assert "60.0 GiB needed, 36.1 GiB MemAvailable." in message[:200]
    tag = parse_leading_refusal_tag(message)
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY
    assert tag.guard == GUARD and tag.waivable is False
    assert message.endswith(
        "Before refusing, the driver unloaded the 2 models (H3TextEncoder, "
        "H3VideoVAE) ComfyUI held on this host: MemAvailable read 10.0 GiB "
        "before the unload and 32.1 GiB after it. It then asked the 2 actors on "
        "the mesh to release their allocator caches: MemAvailable read 32.1 GiB "
        "before the trim and 36.1 GiB after it, and the figures above are the "
        "reading after it."), message
    lines = [record.getMessage() for record in journal]
    assert len(lines) == 1 and journal[0].levelno == logging.INFO
    assert "actor trim: h3.safetensors still does not fit after" in lines[0]


def test_the_trim_runs_alone_when_comfy_holds_nothing(monkeypatch, box, journal):
    box.mem = 50 * _GIB
    calls = _comfy(monkeypatch, box, models=())
    mesh = _Mesh(box, cache=int(17.6 * _GIB))
    driver_unload.check_with_unload(
        [_priced(box, 60 * _GIB)], handle=mesh, subject="h3.safetensors")
    assert calls.unload == 0 and mesh.calls == [TRIM]
    assert [record.getMessage() for record in journal] == [
        "actor trim: h3.safetensors fits after 2 actors released their allocator "
        "caches; MemAvailable went from 50.0 GiB to 67.6 GiB across the trim, a "
        "gain of 17.6 GiB"]


def test_a_trim_alone_that_falls_short_names_only_the_trim(monkeypatch, box):
    box.mem = 10 * _GIB
    _comfy(monkeypatch, box, models=())
    mesh = _Mesh(box, cache=4 * _GIB, world=1)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload([_priced(box, 60 * _GIB)], handle=mesh)
    message = str(excinfo.value)
    assert "60.0 GiB needed, 14.0 GiB MemAvailable." in message[:200]
    assert message.endswith(
        "Before refusing, the driver asked the 1 actor on the mesh to release "
        "their allocator caches: MemAvailable read 10.0 GiB before the trim and "
        "14.0 GiB after it, and the figures above are the reading after it."), message


def test_a_trim_after_comfy_kept_every_model_names_only_the_trim(monkeypatch, box):
    box.mem = 10 * _GIB
    calls = _comfy(monkeypatch, box, kept=(H3TextEncoder, H3VideoVAE))
    mesh = _Mesh(box, cache=4 * _GIB)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload([_priced(box, 60 * _GIB)], handle=mesh)
    assert calls.unload == 1 and mesh.calls == [TRIM]
    assert "the driver unloaded" not in str(excinfo.value)
    assert str(excinfo.value).endswith(
        "MemAvailable read 10.0 GiB before the trim and 14.0 GiB after it, and "
        "the figures above are the reading after it.")


def test_the_trim_runs_once_from_the_step_that_still_refuses(monkeypatch, box):
    box.mem = 30 * _GIB
    _comfy(monkeypatch, box)
    mesh = _Mesh(box, cache=int(17.6 * _GIB))
    seen: list = []
    steps = [_priced(box, 20 * _GIB, seen=seen, label="first"),
             _priced(box, 50 * _GIB, seen=seen, label="second"),
             _priced(box, 65 * _GIB, seen=seen, label="third")]
    driver_unload.check_with_unload(steps, handle=mesh)
    assert seen == ["first", "second", "second", "third", "third"]
    assert mesh.calls == [TRIM]
    seen.clear()
    box.mem = 30 * _GIB
    _comfy(monkeypatch, box)
    mesh = _Mesh(box, cache=int(17.6 * _GIB))
    steps[2] = _priced(box, 90 * _GIB, seen=seen, label="third")
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload(steps, handle=mesh)
    assert seen == ["first", "second", "second", "third", "third"]
    assert mesh.calls == [TRIM]
    assert str(excinfo.value).startswith(
        "[dgxm:C guard=activation_footprint_preflight waivable=0] third preflight")


@pytest.mark.parametrize("state", ["none", "not_set_up", "defunct", "dirty", "torn_down"])
def test_no_live_mesh_means_no_trim_and_no_error(monkeypatch, box, state):
    """A trim needs a mesh whose actors ran setup; anything else is a no-op,
    and the refusal carries the unload's sentence alone."""
    box.mem = 10 * _GIB
    _comfy(monkeypatch, box)
    mesh = _Mesh(box, cache=int(17.6 * _GIB))
    if state == "not_set_up":
        mesh.setup_key = None
    elif state == "defunct":
        mesh.defunct = True
    elif state == "dirty":
        mesh.setup_cleanup_state = SimpleNamespace(outcome="dirty")
    elif state == "torn_down":
        mesh.teardown_complete = True
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload(
            [_priced(box, 60 * _GIB)], handle=None if state == "none" else mesh)
    assert mesh.calls == []
    assert str(excinfo.value).endswith(
        "MemAvailable read 10.0 GiB before the unload and 32.1 GiB after it, and "
        "the figures above are the reading after it.")


@pytest.mark.parametrize("held", [True, False])
def test_a_failed_trim_is_logged_and_never_masks_the_refusal(
        monkeypatch, box, journal, held):
    box.mem = 10 * _GIB
    _comfy(monkeypatch, box, **({} if held else {"models": ()}))
    mesh = _Mesh(box, cache=int(17.6 * _GIB), error=RuntimeError("rank 1 went away"))
    seen: list = []
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload(
            [_priced(box, 60 * _GIB, seen=seen)], handle=mesh, subject="h3.safetensors")
    assert type(excinfo.value) is mesh_safety.StockLoadCapacityError
    assert mesh.calls == [TRIM]
    assert seen == (["probe", "probe"] if held else ["probe"])
    assert [record.levelno for record in journal] == [logging.WARNING]
    warning = journal[0].getMessage()
    assert "actor trim failed before refusing h3.safetensors" in warning
    assert "rank 1 went away" in warning
    message = str(excinfo.value)
    tag = parse_leading_refusal_tag(message)
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY
    if held:
        assert "60.0 GiB needed, 32.1 GiB MemAvailable." in message[:200]
        assert message.endswith(
            "and the figures above are the reading after it. It also asked the "
            "actors on the mesh to release their allocator caches, and that "
            "request failed (RuntimeError)."), message
    else:
        assert "60.0 GiB needed, 10.0 GiB MemAvailable." in message[:200]
        assert message.endswith(
            "Before refusing, the driver asked the actors on the mesh to release "
            "their allocator caches, and that request failed (RuntimeError), so "
            "the figures above are the reading before it."), message


@pytest.mark.parametrize("held", [True, False])
@pytest.mark.parametrize("busy", ["progress", "lease"])
def test_a_running_render_is_never_trimmed(monkeypatch, box, held, busy):
    box.mem = 20 * _GIB
    calls = _comfy(monkeypatch, box, **({} if held else {"models": ()}))
    mesh = _Mesh(box, cache=40 * _GIB)
    if busy == "progress":
        telemetry.render_progress.start(10, token=object())
    else:
        mesh.sample_leases = {3: 1}
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload([_priced(box, 30 * _GIB)], handle=mesh)
    assert mesh.calls == [] and calls.unload == 0
    message = str(excinfo.value)
    assert "trim" not in message and "actors" not in message
    assert ("because" in message) is held


def test_the_kill_switch_turns_the_trim_off_too(monkeypatch, box):
    box.mem = 20 * _GIB
    calls = _comfy(monkeypatch, box)
    mesh = _Mesh(box, cache=40 * _GIB)
    monkeypatch.setenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, "1")
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        driver_unload.check_with_unload([_priced(box, 30 * _GIB)], handle=mesh)
    assert calls.unload == 0 and mesh.calls == []
    assert "unload" not in str(excinfo.value) and "trim" not in str(excinfo.value)


def test_the_rescue_reaches_a_worker_only_to_release_its_cache():
    """The Workers' residents live in other processes and only a Worker RPC
    could reach them. The rescue's one Worker call is clear_vram at soft,
    which frees the allocator cache and unloads no model; nothing here names
    another endpoint, a per-actor call or a recycle, and nothing imports the
    mesh, an actor, monarch or torch."""
    tree = ast.parse((SRC / "driver_unload.py").read_text())
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "call_all"]
    assert len(calls) == 1
    endpoint, level = calls[0].args[:2]
    assert isinstance(endpoint, ast.Name) and endpoint.id == "TRIM_ENDPOINT"
    assert isinstance(level, ast.Name) and level.id == "TRIM_LEVEL"
    assert (driver_unload.TRIM_ENDPOINT, driver_unload.TRIM_LEVEL) == ("clear_vram", "soft")
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for name in ("call_one", "workers", "recycle_detailed", "capacity_quote", "slice",
                 "unload_all", "apply_worker_args"):
        assert name not in attributes, name
    strings = {node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    for name in ("hard", "unload", "recycle", "load_model", "teardown_group"):
        assert name not in strings, name
    imported = {alias.name for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    for name in ("mesh", "actor", "monarch", "torch"):
        assert not any(item.split(".")[0] == name for item in imported), imported


# 4. The shipped render path: the MiniMax H3 activation preflight.

class _Fake:
    """A tensor stand-in: ``shape`` is all h3_rows reads from it."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


class _Nested:
    """comfy's NestedTensor duck: video first, audio second."""

    def __init__(self, video: tuple[int, ...], audio: tuple[int, ...]):
        self.tensors = [_Fake(video), _Fake(audio)]
        self.shape = video


UNET = "minimax_h3_ref2va_int8_pruned.safetensors"
DIT_BYTES = 20_970_379_616          # the pruned int8 artifact, 19.53 GiB
CANVAS = (1344, 768, 124)           # Config B, 37,730 packed rows


def _h3_latent() -> dict:
    width, height, length = CANVAS
    frames = h3_rows.align_frame_count(length)
    video = (1, 24, h3_rows.latent_t_for_frames(frames), height // 16, width // 16)
    return {"samples": _Nested(video, (1, 32, 2, h3_rows.audio_t_for_frames(frames)))}


def _h3_request() -> dict:
    return {"positive": [[_Fake((1, 20, 4096)), {}]]}


def _h3_rig(monkeypatch, tmp_path, *, preset: str) -> SimpleNamespace:
    checkpoint = tmp_path / UNET
    checkpoint.write_bytes(b"x")
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: str(checkpoint)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint",
                        lambda _path: ("minimax_h3", "int8"))
    monkeypatch.setattr(driver_footprint, "file_size_bytes", lambda _path: DIT_BYTES)
    for memo in ("_LAST_RENDERED_UNET", "_LAST_SUBMITTED_UNET", "_LAST_DRIVER_CHARGED_UNET"):
        monkeypatch.setattr(render_preflight, memo, None)
    rpc: list = []
    handle = _leased(owns_hosts=False, world=2,
                     effective_worker_args=lambda requested=None: dict(requested or {}),
                     call_all=lambda *a, **k: rpc.append(a))
    mesh = SimpleNamespace(topology_preset=preset, worker_args={}, world=2, handle=handle)
    return SimpleNamespace(
        model=SimpleNamespace(unet_name=UNET, mesh=mesh, options={}), rpc=rpc)


def _estimates(monkeypatch) -> list:
    seen: list = []
    real = h3_activation.estimate_h3_activation

    def capture(**kwargs):
        estimate = real(**kwargs)
        seen.append(estimate)
        return estimate

    monkeypatch.setattr(h3_activation, "estimate_h3_activation", capture)
    return seen


def _h3_need(*, weights: int) -> int:
    """MemAvailable at which this render's H3 estimate just fits at world 2."""
    rows = h3_rows.rows_from_request(_h3_request(), _h3_latent())
    estimate = h3_activation.estimate_h3_activation(
        rows=rows, sp_degree=2, weight_bytes=weights, weights_resident=not weights,
        mem_available=0, reserve=driver_footprint.reserve_bytes(None))
    return estimate.projected + estimate.reserve


def test_back_to_back_h3_renders_proceed_after_the_unload(monkeypatch, tmp_path, box, journal):
    """An explicit uly2 preset, the weights resident on both ranks, and the
    driver host 2.3 GiB short while ComfyUI holds its text encoder and VAE."""
    rig = _h3_rig(monkeypatch, tmp_path, preset="uly2")
    need = _h3_need(weights=0)
    box.mem = need - int(2.3 * _GIB)
    calls = _comfy(monkeypatch, box)
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        h3_activation.h3_activation_preflight(
            h3_activation.estimate_h3_activation(
                rows=h3_rows.rows_from_request(_h3_request(), _h3_latent()),
                sp_degree=2, weight_bytes=0, weights_resident=True,
                mem_available=box.mem, reserve=driver_footprint.reserve_bytes(None)),
            unet_name=UNET)
    render_preflight.activation_footprint_preflight_for_request(
        rig.model, _h3_request(), _h3_latent())
    assert calls.unload == 1
    assert rig.rpc == []
    lines = [record.getMessage() for record in journal]
    assert len(lines) == 1 and UNET in lines[0] and "a gain of 22.1 GiB" in lines[0]


def test_an_h3_render_still_short_refuses_with_the_reading_after_the_unload(
        monkeypatch, tmp_path, box):
    rig = _h3_rig(monkeypatch, tmp_path, preset="uly2")
    box.mem = 2 * _GIB
    # Only the VAE is left to unload: the text encoder's 16.7 GiB alone would
    # make room for this render at the 2026-10-31 activation price.
    calls = _comfy(monkeypatch, box, models=((H3VideoVAE, VAE_BYTES),))
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        render_preflight.activation_footprint_preflight_for_request(
            rig.model, _h3_request(), _h3_latent())
    message = str(excinfo.value)
    assert calls.unload == 1 and rig.rpc == []
    assert message.startswith("[dgxm:C guard=activation_footprint_preflight waivable=0] "
                              "minimax_h3 activation preflight: 15.9 GiB needed per rank, "
                              "2.4 GiB usable (7.4 GiB MemAvailable minus a 5.0 GiB reserve")
    assert message.endswith("MemAvailable read 2.0 GiB before the unload and 7.4 GiB "
                            "after it, and the figures above are the reading after it.")


def test_the_h3_re_check_keeps_the_memo_sampled_before_the_driver_check(
        monkeypatch, tmp_path, box):
    """Under `auto` the driver footprint check clears the pending load and
    stamps its memo. The H3 check that refused runs again after the unload
    with the memo sampled first, so it still charges the weights: a re-check
    that re-read the memo would credit this render for its own charge."""
    rig = _h3_rig(monkeypatch, tmp_path, preset="auto")
    need = _h3_need(weights=DIT_BYTES)
    box.mem = need - int(2.3 * _GIB)
    calls = _comfy(monkeypatch, box)
    driver_checks: list = []
    real = driver_footprint.estimate_driver_footprint
    monkeypatch.setattr(driver_footprint, "estimate_driver_footprint",
                        lambda **kw: driver_checks.append(kw) or real(**kw))
    estimates = _estimates(monkeypatch)
    render_preflight.activation_footprint_preflight_for_request(
        rig.model, _h3_request(), _h3_latent())
    assert calls.unload == 1
    assert len(driver_checks) == 1
    assert render_preflight._LAST_DRIVER_CHARGED_UNET == UNET
    assert [estimate.weight_bytes for estimate in estimates] == [DIT_BYTES, DIT_BYTES]
    assert not estimates[0].fits and estimates[1].fits


def test_an_h3_render_behind_a_running_one_is_refused_without_an_unload(
        monkeypatch, tmp_path, box):
    """A pipelined submission behind a render in flight: the driver keeps
    ComfyUI's models and the refusal says why."""
    rig = _h3_rig(monkeypatch, tmp_path, preset="uly2")
    box.mem = 2 * _GIB
    calls = _comfy(monkeypatch, box)
    telemetry.render_progress.start(10, token=object())
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        render_preflight.activation_footprint_preflight_for_request(
            rig.model, _h3_request(), _h3_latent())
    assert calls.unload == 0
    message = str(excinfo.value)
    assert "(2.0 GiB MemAvailable minus" in message[:200]
    assert "because a render is still running in this driver" in message


def test_a_second_h3_render_proceeds_after_the_actors_release_their_caches(
        monkeypatch, tmp_path, box, journal):
    """The second request of a session at the render site: the VAE unload
    falls short and the actor's retained activations make up the rest. Sized
    from the live constants, so the case holds whatever the H3 fit says."""
    rig = _h3_rig(monkeypatch, tmp_path, preset="uly2")
    events: list = []
    mesh = _Mesh(box, cache=5 * _GIB, events=events, owns_hosts=False,
                 effective_worker_args=lambda requested=None: dict(requested or {}))
    rig.model.mesh.handle = mesh
    box.mem = _h3_need(weights=0) - 6 * _GIB
    calls = _comfy(monkeypatch, box, models=((H3VideoVAE, 2 * _GIB),))
    _in_order(calls, events)
    render_preflight.activation_footprint_preflight_for_request(
        rig.model, _h3_request(), _h3_latent())
    assert events == ["unload", "trim"]
    assert mesh.calls == [TRIM]
    assert rig.rpc == []
    lines = [record.getMessage() for record in journal]
    assert len(lines) == 1 and lines[0].startswith(f"actor trim: {UNET} fits after")


def test_the_refusal_entries_the_unload_names_exist():
    """The module docstring and the H3 refusal point at #56; the docs must
    carry the step under that entry and under #47 and #52."""
    troubleshooting = (SRC.parents[1] / "docs" / "TROUBLESHOOTING.md").read_text()
    for heading in ("## 47. ", "## 52. ", "## 56. "):
        start = troubleshooting.index(heading)
        end = troubleshooting.index("\n## ", start + 1)
        assert driver_unload.DRIVER_UNLOAD_DISABLE_ENV in troubleshooting[start:end], heading
