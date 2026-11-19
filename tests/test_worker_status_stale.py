"""A status read never waits on the GPU thread.

Torch reads its allocated and reserved figures under its allocator lock while
the reader keeps the interpreter lock. A comfy-managed forward held the first
and wanted the second, and a status read that met it stopped the Worker for
good (docs/TROUBLESHOOTING.md #62). Status now reads the device only while the
GPU lock is free, and answers with its last figures, marked stale, otherwise.
"""
from __future__ import annotations

import ast
import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import fsdp_reload_price
from dgx_monarch.actor import worker_status
from dgx_monarch.runtime_timing import MeasuredAsyncLock

SRC = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"
FIGURES = {"free_gib": 60.0, "total_gib": 121.6, "allocated_gib": 15.5, "reserved_gib": 18.1}


def _never_read() -> dict[str, float]:
    raise AssertionError("the device was read while the GPU lock was held")


class _Clock:
    def __init__(self, at: float) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at


def _held(worker, **kwargs):
    """The reply a status call gets while a GPU job holds the Worker's lock."""
    async def run():
        async with worker._gpu_lock:
            return worker_status.device_figures(worker, **kwargs)
    return asyncio.run(run())


def _worker():
    return SimpleNamespace(_gpu_lock=MeasuredAsyncLock())


def test_a_free_gpu_lock_gives_fresh_figures_with_no_mark():
    worker = _worker()

    reply = worker_status.device_figures(worker, read=lambda: dict(FIGURES))

    assert reply == FIGURES
    assert "stale" not in reply and "age_s" not in reply


def test_a_held_gpu_lock_gives_the_last_figures_marked_stale_with_their_age():
    worker = _worker()
    clock = _Clock(100.0)
    worker_status.device_figures(worker, read=lambda: dict(FIGURES), now=clock)

    clock.at = 142.34
    reply = _held(worker, read=_never_read, now=clock)

    assert reply == {**FIGURES, "stale": True, "age_s": 42.3}


def test_a_held_gpu_lock_before_any_read_gives_no_figures_and_reads_nothing():
    assert _held(_worker(), read=_never_read) is None


def test_the_figures_come_back_fresh_once_the_job_ends():
    worker = _worker()
    worker_status.device_figures(worker, read=lambda: dict(FIGURES))
    assert _held(worker, read=_never_read)["stale"] is True

    later = {**FIGURES, "allocated_gib": 0.0, "reserved_gib": 0.0}
    assert worker_status.device_figures(worker, read=lambda: dict(later)) == later


def test_a_reply_cannot_change_the_figures_kept_for_the_next_one():
    worker = _worker()
    first = worker_status.device_figures(worker, read=lambda: dict(FIGURES))
    first["reserved_gib"] = -1.0

    assert _held(worker, read=_never_read)["reserved_gib"] == FIGURES["reserved_gib"]


def test_the_status_reply_says_why_it_has_no_figures(monkeypatch):
    """The whole reply still comes: a busy Worker is not a silent one."""
    monkeypatch.setattr(worker_status, "source_manifest_sha256", lambda: "source")
    monkeypatch.setattr(worker_status, "_read_device_figures", _never_read)
    mm = types.ModuleType("comfy.model_management")
    mm.MAX_PINNED_MEMORY, mm.NUM_STREAMS, mm.EXTRA_RESERVED_VRAM = -1, 0, 0
    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    worker = SimpleNamespace(
        rank=0, world=1, topology={}, _attn=None, _setup_key=object(),
        store=SimpleNamespace(snapshot=lambda: {}, current=None),
        _memory_detail=lambda: {"ok": True}, _gpu_lock=MeasuredAsyncLock())

    async def run():
        async with worker._gpu_lock:
            return worker_status.status_impl(worker)

    status = asyncio.run(run())

    assert status["vram"] is None
    assert "GPU thread is busy" in status["note"]
    assert "vram_error" not in status
    assert status["rank"] == 0 and status["models"] == {}


def test_only_the_guarded_reader_touches_the_device_in_this_module():
    """A new read of the allocator beside the guard brings the wait back."""
    tree = ast.parse((SRC / "actor" / "worker_status.py").read_text())
    readers = {"mem_get_info", "memory_allocated", "memory_reserved", "memory_stats"}
    found = {}
    for func in (n for n in tree.body if isinstance(n, ast.FunctionDef)):
        names = {n.attr for n in ast.walk(func) if isinstance(n, ast.Attribute)}
        if names & readers:
            found[func.name] = sorted(names & readers)

    assert found == {
        "_read_device_figures": ["mem_get_info", "memory_allocated", "memory_reserved"]}


@pytest.mark.parametrize("vram, credited", [
    ({"reserved_gib": 18.1}, True),
    ({"reserved_gib": 18.1, "stale": True, "age_s": 42.3}, False),
    ({"reserved_gib": 18.1, "stale": False}, True),
])
def test_a_stale_reserved_figure_credits_nothing_to_an_fsdp_reload(vram, credited):
    """An old figure may name a pool the rank has since given back."""
    share = 12 * 2**30

    credit = fsdp_reload_price._rank_credit_bytes({"vram": vram}, share)

    assert (credit > 0) is credited
