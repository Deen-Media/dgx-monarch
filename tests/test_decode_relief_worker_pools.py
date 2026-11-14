"""The decode relief's ask to the Workers: their idle device pools, from the fleet attached now.

``vae_decode_relief.release_worker_pools`` makes one call of the trim the
driver unload already makes, to the live fleet that serves this driver's
renders. The fleet is found from a weak note left at submit, and never used as
noted: a reset retires that handle and attaches another (the lesson of #179).
CPU only; the fleet is a plain object and the trim is recorded, not sent.
"""
from __future__ import annotations

import gc
import sys
from types import SimpleNamespace

import pytest

from dgx_monarch import driver_unload, mesh, vae_decode_relief
from dgx_monarch.nodes import recycle_drain

GIB = 2 ** 30


class _Fleet:
    """What ``driver_unload.trimmable`` reads on a mesh handle."""

    def __init__(self, name, **state):
        self.name = name
        self.setup_key = state.get("setup_key", ("uly2", 2))
        self.defunct = state.get("defunct", False)
        self.teardown_complete = state.get("teardown_complete", False)
        self.setup_cleanup_state = state.get("setup_cleanup_state")

    def call_all(self, *args, **kwargs):
        raise AssertionError("the tests record the trim; nothing is sent")


@pytest.fixture(autouse=True)
def _no_note_outlives_a_test():
    """The note is process state: a fake fleet left in it would reach the relief's other tests."""
    vae_decode_relief.note_serving(None)
    yield
    vae_decode_relief.note_serving(None)


@pytest.fixture
def trims(monkeypatch):
    asked: list = []

    def trim(fleet, level):
        assert level == driver_unload.POOL_LEVEL == "pool"  # never soft: that drops the Worker's decode VAE
        asked.append(fleet)
        return driver_unload.Trimmed(2, 7 * 2 ** 30, 13 * 2 ** 30)

    monkeypatch.setattr(driver_unload, "trim_actors", trim)
    return asked


def test_the_note_never_keeps_a_fleet_alive(trims):
    fleet = _Fleet("live")
    vae_decode_relief.note_serving(fleet)
    assert vae_decode_relief.serving_now() is fleet
    del fleet
    gc.collect()
    assert vae_decode_relief.serving_now() is None
    assert vae_decode_relief.release_worker_pools() == (vae_decode_relief.NO_FLEET, False) and trims == []


def test_a_reset_fleet_is_never_called_and_the_one_attached_in_its_place_is(trims, monkeypatch):
    retired = _Fleet("retired", defunct=True, teardown_complete=True)
    fresh = _Fleet("fresh")
    vae_decode_relief.note_serving(retired)
    monkeypatch.setattr(recycle_drain, "attached_fleet", lambda handle: fresh if handle is retired else None)
    assert vae_decode_relief.serving_now() is fresh
    clause, released = vae_decode_relief.release_worker_pools()
    assert trims == [fresh] and released is True
    assert "2 Worker actor(s) then released their idle device pools" in clause
    assert "MemAvailable read 7.0 GiB before that and 13.0 GiB after it" in clause
    # No fleet attached in its place: nobody is called.
    trims.clear()
    monkeypatch.setattr(recycle_drain, "attached_fleet", lambda handle: None)
    assert vae_decode_relief.serving_now() is None and vae_decode_relief.release_worker_pools() == (
        vae_decode_relief.RESET_FLEET, False)
    assert trims == []


@pytest.mark.parametrize("state", [{"setup_key": None}, {"setup_cleanup_state": "DIRTY"}],
                         ids=["not set up", "cleanup failed"])
def test_a_fleet_that_is_not_ready_gets_no_call(trims, state):
    fleet = _Fleet("unready", **state)
    vae_decode_relief.note_serving(fleet)
    assert vae_decode_relief.serving_now() is None
    assert vae_decode_relief.release_worker_pools() == (vae_decode_relief.UNREADY_FLEET, False)
    assert trims == []


def test_a_claim_that_fails_is_said_and_the_decode_goes_on(monkeypatch):
    fleet = _Fleet("busy")
    vae_decode_relief.note_serving(fleet)

    def owned(handle, level):
        raise TimeoutError("another render session owns this mesh")

    monkeypatch.setattr(driver_unload, "trim_actors", owned)
    clause, released = vae_decode_relief.release_worker_pools()
    assert "the Workers were not asked for their idle device pools" in clause
    assert "TimeoutError: another render session owns this mesh" in clause and released is False


# The path the hardware takes: the note a submitted render leaves (``nodes/render_submit.py``), the fleet held only
# by the mesh registry, the real lookup and the real trim, and the decode through the relief's own wrapper. No
# fleet is handed to the relief. On 2026-11-13 (two held pair renders) a text encoder free met the decode's price
# on ComfyUI's reading, the relief never asked, and its line carried no Worker clause at all.

class _Attached:
    """A mesh handle as the registry holds it after a render: set up, idle, two actors."""

    def __init__(self, source="cluster.toml", **state):
        self.config = SimpleNamespace(source=source, hosts=("first", "second"))
        self.setup_key = ("uly2", 2)
        self.defunct = state.get("defunct", False)
        self.teardown_complete = state.get("teardown_complete", False)
        self.setup_cleanup_state = None
        self.calls: list = []

    def call_all(self, endpoint, *args, timeout_s):
        self.calls.append((endpoint, args))
        return [{"rank": 0}, {"rank": 1}]


def _decode_after_a_render(monkeypatch, caplog, noted):
    from test_vae_decode_relief import LATENT, _available, _loaded, _model_management, _TextEncoder, _Vae, _wrapped

    monkeypatch.setitem(sys.modules, "comfy.memory_management", SimpleNamespace(aimdo_enabled=True))
    monkeypatch.setitem(vae_decode_relief._SAMPLING, "depth", 0)
    monkeypatch.delenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, raising=False)
    monkeypatch.setattr(vae_decode_relief, "_SETTLE_S", 0.0)
    vae_decode_relief.note_serving(noted)  # the note render_submit makes beside the guard's handle
    del noted
    gc.collect()
    # As on the head: priced at 7.2 GiB with its reserve against 5.2 free; the text encoder free meets it.
    mm = _model_management([int(5.2 * GIB), int(7.3 * GIB)], held=[_loaded(_TextEncoder())],
                           inference=GIB, reserve=2 * GIB)
    _available(monkeypatch, int(12.8 * GIB), int(14.9 * GIB), int(14.9 * GIB), int(21.3 * GIB))
    wrapped, calls, answer = _wrapped(monkeypatch, mm)
    with caplog.at_level("INFO"):
        assert wrapped(_Vae(price=int(5.2 * GIB)), LATENT) is answer
    assert len(calls) == 1
    return next(record.getMessage() for record in caplog.records if "decode relief:" in record.getMessage())


def test_a_decode_after_a_render_asks_the_fleet_that_served_it(monkeypatch, caplog):
    monkeypatch.setitem(mesh._MESHES, ("cluster.toml",), _Attached())  # the registry holds it, not the test
    line = _decode_after_a_render(monkeypatch, caplog, mesh._MESHES[("cluster.toml",)])
    fleet = mesh._MESHES[("cluster.toml",)]
    assert fleet.calls == [("clear_vram", (driver_unload.POOL_LEVEL,))]
    assert ("2 Worker actor(s) then released their idle device pools: MemAvailable read 14.9 GiB before that "
            "and 21.3 GiB after it") in line
    assert "MemAvailable read 12.8 GiB before that and 14.9 GiB after it" in line
    assert line.endswith("the decode starts with 7.3 GiB free")


def test_a_reset_since_the_render_asks_the_fleet_attached_in_its_place(monkeypatch, caplog):
    retired, fresh = _Attached(defunct=True, teardown_complete=True), _Attached()
    monkeypatch.setitem(mesh._MESHES, ("cluster.toml",), fresh)
    keep = retired  # a cached MeshSpec still holds the retired handle
    line = _decode_after_a_render(monkeypatch, caplog, retired)
    assert keep.calls == [] and fresh.calls == [("clear_vram", (driver_unload.POOL_LEVEL,))]
    assert "2 Worker actor(s) then released their idle device pools" in line


def test_with_no_fleet_attached_the_line_says_so(monkeypatch, caplog):
    line = _decode_after_a_render(monkeypatch, caplog, None)
    assert vae_decode_relief.NO_FLEET in line and line.endswith("the decode starts with 7.3 GiB free")


def test_a_fleet_a_pipelined_render_owns_is_not_asked_and_the_line_says_why(monkeypatch, caplog):
    fleet = _Attached()
    monkeypatch.setitem(mesh._MESHES, ("cluster.toml",), fleet)

    def owned(handle, level):
        raise TimeoutError("another render session is active on this mesh handle")

    monkeypatch.setattr(driver_unload, "trim_actors", owned)
    line = _decode_after_a_render(monkeypatch, caplog, fleet)
    assert "the Workers were not asked for their idle device pools (TimeoutError: another render session" in line
