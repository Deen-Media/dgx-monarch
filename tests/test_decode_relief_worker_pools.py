"""The decode relief's ask to the Workers: their idle device pools, from the fleet attached now.

``vae_decode_relief.release_worker_pools`` makes one call of the trim the
driver unload already makes, to the live fleet that serves this driver's
renders. The fleet is found from a weak note left at submit, and never used as
noted: a reset retires that handle and attaches another (the lesson of #179).
CPU only; the fleet is a plain object and the trim is recorded, not sent.
"""
from __future__ import annotations

import gc

import pytest

from dgx_monarch import driver_unload, vae_decode_relief
from dgx_monarch.nodes import recycle_drain


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
    assert vae_decode_relief.release_worker_pools() == ("", False) and trims == []


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
    assert vae_decode_relief.serving_now() is None and vae_decode_relief.release_worker_pools() == ("", False)
    assert trims == []


@pytest.mark.parametrize("state", [{"setup_key": None}, {"setup_cleanup_state": "DIRTY"}],
                         ids=["not set up", "cleanup failed"])
def test_a_fleet_that_is_not_ready_gets_no_call(trims, state):
    fleet = _Fleet("unready", **state)
    vae_decode_relief.note_serving(fleet)
    assert vae_decode_relief.serving_now() is None and vae_decode_relief.release_worker_pools() == ("", False)
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
