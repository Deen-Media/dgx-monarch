"""A Worker makes room for its CUDA context, and refuses in words if it still cannot make one.

On 2026-11-12 a Worker that started with 1.5 GiB of free memory (45 GiB
available, the rest page cache and a test hold) could not make its CUDA
context: its setup ended in a raw ``CUDA error: out of memory`` from
``cuDevicePrimaryCtxRetain``. Two more starts failed, with 4.61 and 8.85 GiB
free; one worked with 8.87 and every other with 10.6 or more. ``actor/context_room.py`` writes to memory before the first CUDA call
so that the kernel gives page cache back, and turns what is left of the
failure into a class C refusal.

Every test here fakes the memory readings, the driver library and the
allocation: none reads this host's memory or touches a GPU.
"""
from __future__ import annotations

import ast
import logging
import sys
import types
from pathlib import Path

import pytest

import test_worker_box as boxes
from dgx_monarch import box_names, residency_mode, worker_box
from dgx_monarch.actor import context_room, worker_env
from dgx_monarch.log import get_logger
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag, read_rank

SRC = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"
_GIB = 2 ** 30
OOM = ("CUDA error: out of memory\nSearch for `cudaErrorMemoryAllocation' ... Returning 2 "
       "(CUDA_ERROR_OUT_OF_MEMORY) from cuDevicePrimaryCtxRetain")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(context_room.DISABLE_ENV, raising=False)
    monkeypatch.setattr(context_room, "_LAST", context_room.Outcome())


@pytest.fixture
def journal():
    """This module's log records; the pack's loggers do not propagate."""
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    logger = get_logger("dgx_monarch.actor.context_room")
    logger.addHandler(handler)
    yield records
    logger.removeHandler(handler)


class _Box:
    """A box whose page cache gives way to what a process writes, as the kernel's does."""

    def __init__(self, free_gib: float, available_gib: float, *, fail_at: int = 0) -> None:
        self.free, self.available = int(free_gib * _GIB), int(available_gib * _GIB)
        self.fail_at = fail_at
        self.taken: list[int] = []
        self.live = 0
        self.peak_live = 0
        self.reads = 0
        self.asked = 0
        self.ticks = 0.0

    def read(self):
        self.reads += 1
        return self.free, self.available

    def shares(self):
        self.asked += 1
        return True

    def take(self, nbytes: int):
        if self.fail_at and len(self.taken) + 1 == self.fail_at:
            raise MemoryError("no more")
        self.taken.append(nbytes)
        self.live += nbytes
        self.peak_live = max(self.peak_live, self.live)
        # The first pages written are the free ones; the rest come from the cache.
        self.free = max(0, self.free - nbytes)
        self.available -= nbytes
        box = self

        class _Held:
            def close(self):
                box.live -= nbytes
                box.free += nbytes
                box.available += nbytes

        return _Held()

    def clock(self):
        self.ticks += 0.5
        return self.ticks

    def run(self, **over):
        return context_room.make_room(read=self.read, shares=over.get("shares", self.shares),
                                      take=self.take, clock=self.clock,
                                      preload=over.get("preload", lambda: None))


# 1. An ordinary start pays nothing.

def test_a_roomy_box_reads_once_and_does_nothing_else(journal):
    box = _Box(free_gib=43.3, available_gib=98.0)
    outcome = box.run()
    assert outcome.state == "roomy"
    assert box.reads == 1 and box.asked == 0 and box.taken == []
    assert journal == []
    assert context_room.status_block() == {"state": "roomy"}


def test_free_memory_at_the_mark_is_roomy():
    box = _Box(free_gib=16.0, available_gib=60.0)
    assert box.run().state == "roomy" and box.taken == []


# 2. Short of free memory with plenty available: the 2026-11-12 states.

@pytest.mark.parametrize(("free", "available"), [(1.5, 44.7), (4.6, 47.4), (8.8, 50.8)])
def test_the_worker_writes_up_to_the_mark_holds_it_and_releases_it_together(
        journal, free, available):
    box = _Box(free_gib=free, available_gib=available)
    outcome = box.run()
    assert outcome.state == "made_room" and outcome.why == ""
    assert box.taken == [_GIB] * 16, "steps of 1 GiB up to the 16 GiB mark"
    assert box.peak_live == 16 * _GIB, "every step is held until the last one is written"
    assert box.live == 0, "all of it is released"
    assert outcome.free_after == box.free and box.free >= 16 * _GIB
    (record,) = journal
    assert record.levelno == logging.INFO
    line = record.getMessage()
    assert line.startswith(f"free memory was {free} GiB with {available} GiB available, so the "
                           "Worker made room for its GPU context: 16.0 GiB touched and released in ")
    assert line.endswith(f"; {box.free / _GIB:.1f} GiB is free now")
    block = context_room.status_block()
    assert block["state"] == "made_room" and block["touched_gib"] == 16.0
    assert block["free_before_gib"] == free and block["free_after_release_gib"] == 16.0
    assert block["context"] == "left to the bootstrap"
    assert block["free_at_context_gib"] is None, "nobody has read it at the context yet"


def test_torch_is_imported_before_any_memory_is_freed_and_only_when_room_is_made():
    """A file read after the release fills free memory with page cache again."""
    box = _Box(free_gib=1.6, available_gib=45.1)
    events: list = []
    inner = box.take

    def take(nbytes):
        events.append("take")
        return inner(nbytes)

    box.take = take
    box.run(preload=lambda: events.append("preload"))
    assert events[0] == "preload" and events.count("preload") == 1 and events.count("take") == 16
    roomy = _Box(free_gib=40.0, available_gib=90.0)
    roomy.run(preload=lambda: pytest.fail("a roomy start imports nothing"))
    short = _Box(free_gib=1.6, available_gib=16.0)
    short.run(preload=lambda: pytest.fail("no room is made, so nothing is imported"))


def test_the_amount_is_the_mark_and_not_the_mark_less_what_is_free():
    """Writing 11.4 GiB on a box with 4.6 GiB free would leave about 11.4 free, not the
    16 GiB the mark asks for."""
    box = _Box(free_gib=4.6, available_gib=47.4)
    box.run()
    assert sum(box.taken) == context_room.MARK_BYTES


def test_exactly_the_floor_left_is_allowed():
    """21 GiB available is the mark and the floor: all sixteen steps, the last leaving 5."""
    box = _Box(free_gib=0.5, available_gib=21.0)
    assert box.run().state == "made_room" and box.taken == [_GIB] * 16


def test_a_reading_a_hair_under_the_need_never_prints_as_the_need(journal):
    box = _Box(free_gib=1.6, available_gib=20.999)
    assert box.run().why == "20.9 GiB available is under the 21.0 GiB it needs to do that safely"
    assert journal[0].getMessage().startswith("free memory is 1.6 GiB with 20.9 GiB available, ")


# 3. It never takes the box under its floor.

def test_it_stops_before_a_step_that_would_leave_under_the_floor(journal):
    box = _Box(free_gib=1.0, available_gib=21.0)
    box.available = int(9.5 * _GIB)          # the reading falls after the guard passed

    def read():
        box.reads += 1
        return (box.free, int(21.0 * _GIB)) if box.reads == 1 else (box.free, box.available)

    outcome = context_room.make_room(read=read, shares=box.shares, take=box.take, clock=box.clock,
                                     preload=lambda: None)
    assert outcome.state == "made_room"
    assert box.taken == [_GIB] * 4, "9.5 GiB available allows four steps above a 5 GiB floor"
    assert box.live == 0
    assert "one more step would have left under 5.0 GiB available" in outcome.why
    assert "one more step would have left under 5.0 GiB available" in journal[0].getMessage()


def test_too_little_available_makes_no_room_and_asks_no_driver_library(journal):
    box = _Box(free_gib=1.6, available_gib=20.9)
    outcome = box.run()
    assert outcome.state == "skipped" and box.taken == [] and box.asked == 0
    assert outcome.why == "20.9 GiB available is under the 21.0 GiB it needs to do that safely"
    assert journal[0].getMessage() == (
        "free memory is 1.6 GiB with 20.9 GiB available, and the Worker made no room for its "
        "GPU context: 20.9 GiB available is under the 21.0 GiB it needs to do that safely")


# 4. Only on a GPU that shares the host's memory, and never past its switch.

@pytest.mark.parametrize(("answer", "why"), [
    (False, "this GPU has memory of its own"),
    (None, "the CUDA driver library could not say whether this GPU shares the host's memory"),
])
def test_another_kind_of_gpu_or_an_unknown_one_makes_no_room(journal, answer, why):
    box = _Box(free_gib=1.6, available_gib=45.1)
    outcome = box.run(shares=lambda: answer)
    assert outcome.state == "skipped" and outcome.why == why and box.taken == []
    assert journal[0].getMessage().endswith(why)


def test_the_switch_turns_the_step_off(monkeypatch, journal):
    monkeypatch.setenv(context_room.DISABLE_ENV, "1")
    box = _Box(free_gib=1.6, available_gib=45.1)
    outcome = box.run()
    assert outcome.state == "off" and box.reads == 0 and box.taken == [] and journal == []
    assert context_room.DISABLE_ENV in outcome.sentence()


def test_a_host_whose_memory_cannot_be_read_makes_no_room(journal):
    outcome = context_room.make_room(read=lambda: None, shares=lambda: True,
                                     take=lambda n: pytest.fail("no reading, no step"))
    assert outcome.state == "not_run" and journal == []


# 5. A fault in the step leaves the start as it was.

def test_a_fault_mid_step_releases_everything_and_never_raises(journal):
    box = _Box(free_gib=1.6, available_gib=45.1, fail_at=4)
    outcome = box.run()
    assert outcome.state == "failed" and "MemoryError" in outcome.why
    assert box.taken == [_GIB] * 3 and box.live == 0, "what was held is released"
    assert [record.levelno for record in journal] == [logging.WARNING]
    assert "goes on to make the context as before" in journal[0].getMessage()


def test_a_fault_in_the_reading_never_raises(journal):
    def read():
        raise OSError("gone")

    outcome = context_room.make_room(read=read, shares=lambda: True, take=lambda n: None)
    assert outcome.state == "failed" and journal[0].levelno == logging.WARNING


# 6. The readings and the allocation themselves.

def test_meminfo_reads_free_and_available(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text("MemTotal:       127532032 kB\nMemFree:         1720320 kB\n"
                    "MemAvailable:   47290368 kB\nCached:         44564480 kB\n")
    assert context_room.meminfo(str(path)) == (1720320 * 1024, 47290368 * 1024)
    path.write_text("MemTotal: 1 kB\nMemFree: 5 kB\n")
    assert context_room.meminfo(str(path)) is None
    assert context_room.meminfo(str(tmp_path / "absent")) is None


def test_taken_memory_is_written_on_every_page_and_never_locked():
    held = context_room._take(9 * 2 ** 20 + 4096)
    try:
        assert len(held) == 9 * 2 ** 20 + 4096
        assert all(held[offset] == 1 for offset in range(0, len(held), 4096))
        assert held[len(held) - 1] == 1
    finally:
        held.close()
    names = {node.attr for node in ast.walk(ast.parse((SRC / "actor/context_room.py").read_text()))
             if isinstance(node, ast.Attribute)}
    assert not {"mlock", "mlockall", "MAP_LOCKED", "MAP_POPULATE"} & names


class _Cuda:
    """Stands in for the CUDA driver library's three device calls."""

    def __init__(self, *, init=0, get=0, attribute=0, integrated=1):
        self.results = {"init": init, "get": get, "attribute": attribute}
        self.integrated = integrated
        self.calls: list = []

    def cuInit(self, flags):
        self.calls.append(("cuInit", flags))
        return self.results["init"]

    def cuDeviceGet(self, device, ordinal):
        self.calls.append(("cuDeviceGet", ordinal))
        return self.results["get"]

    def cuDeviceGetAttribute(self, value, attribute, device):
        self.calls.append(("cuDeviceGetAttribute", attribute))
        value._obj.value = self.integrated
        return self.results["attribute"]


@pytest.mark.parametrize(("library", "answer"), [
    (_Cuda(integrated=1), True),
    (_Cuda(integrated=0), False),
    (_Cuda(init=999), None),
    (_Cuda(get=101), None),
    (_Cuda(attribute=1), None),
])
def test_the_gpu_kind_is_asked_of_the_driver_library_without_a_context(monkeypatch, library, answer):
    monkeypatch.setattr(context_room.ctypes, "CDLL", lambda name: library)
    assert context_room.device_shares_host_memory() is answer
    names = [call[0] for call in library.calls]
    assert names == ["cuInit", "cuDeviceGet", "cuDeviceGetAttribute"][:len(names)]
    assert not any("Ctx" in name for name in names), "no call that makes a context"
    if len(library.calls) == 3:
        assert library.calls[1] == ("cuDeviceGet", 0)
        assert library.calls[2] == ("cuDeviceGetAttribute", 18)  # CU_DEVICE_ATTRIBUTE_INTEGRATED


def test_a_missing_driver_library_is_an_unknown_gpu(monkeypatch):
    def absent(name):
        raise OSError("libcuda.so.1: cannot open shared object file")

    monkeypatch.setattr(context_room.ctypes, "CDLL", absent)
    assert context_room.device_shares_host_memory() is None


def test_the_module_imports_torch_and_comfy_only_by_name_at_the_moment_it_needs_them():
    """Nothing here asks torch about a device before the Worker makes the context: the one
    CUDA call is the context's own, ``mem_get_info``, the call ComfyUI's import makes first."""
    source = (SRC / "actor/context_room.py").read_text()
    tree = ast.parse(source)
    cuda = [ast.unparse(node) for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
            and node.value.attr == "cuda"]
    assert cuda == ["importlib.import_module('torch').cuda.mem_get_info"]
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    imported |= {(node.module or "").split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom)}
    assert not {"torch", "comfy", "psutil"} & imported


# 7. The bootstrap: room first, then the context, then a refusal in words.

def _bridge(monkeypatch, events: list, error: BaseException | None = None):
    from dgx_monarch.actor import comfy_bridge

    def ensure(*args, **kwargs):
        events.append(("ensure_comfy", args, kwargs))
        if error is not None:
            raise error

    monkeypatch.setattr(comfy_bridge, "ensure_comfy", ensure)
    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    monkeypatch.setattr(context_room, "make_room",
                        lambda: events.append("make_room") or context_room.Outcome("roomy"))


def test_setup_calls_the_bootstrap_that_makes_room_first(monkeypatch):
    assert worker_env.ensure_comfy is context_room.ensure_comfy
    events: list = []
    _bridge(monkeypatch, events)
    context_room.ensure_comfy("/comfy", {"x": 1}, gpus_per_host=1)
    assert events == ["make_room", ("ensure_comfy", ("/comfy", {"x": 1}), {"gpus_per_host": 1})]


def test_a_process_that_has_its_context_makes_no_room(monkeypatch):
    events: list = []
    _bridge(monkeypatch, events)
    monkeypatch.setitem(sys.modules, "comfy.model_management", types.ModuleType("mm"))
    context_room.ensure_comfy("/comfy", rank=(0, 2))
    assert events == [("ensure_comfy", ("/comfy", None), {})]


# 8. Room, then the context at once.

MADE = context_room.Outcome("made_room", int(1.6 * _GIB), int(45.1 * _GIB),
                            16 * _GIB, 2.0, int(15.8 * _GIB))


_MAKE_CONTEXT = context_room.make_context


def _made_room(monkeypatch, events: list, create, error: BaseException | None = None):
    """The bootstrap after a step that made room, with ``create`` as the context's call."""
    _bridge(monkeypatch, events, error=error)
    monkeypatch.setattr(context_room, "make_room", lambda: events.append("make_room") or MADE)
    monkeypatch.setattr(context_room, "make_context",
                        lambda outcome, rank=None: _MAKE_CONTEXT(outcome, rank, create=create))


def test_the_worker_makes_the_context_straight_after_the_release(monkeypatch):
    events: list = []
    _made_room(monkeypatch, events, lambda: events.append("context"))
    context_room.ensure_comfy("/comfy", {}, gpus_per_host=1, rank=(1, 2))
    assert events == ["make_room", "context", ("ensure_comfy", ("/comfy", {}), {"gpus_per_host": 1})]
    block = context_room.status_block()
    assert block["context"] == "made at once" and block["free_at_context_gib"] == 15.8


def test_comfy_managed_residency_leaves_the_context_to_the_bootstrap(monkeypatch):
    """Stage A of that bring-up runs in the bootstrap and stays ahead of the context."""
    events: list = []
    _made_room(monkeypatch, events, lambda: pytest.fail("no context ahead of stage A"))
    managed = {residency_mode.WORKER_ARG: True}
    assert residency_mode.requested(managed)
    context_room.ensure_comfy("/comfy", managed)
    assert events == ["make_room", ("ensure_comfy", ("/comfy", managed), {})]


@pytest.mark.parametrize("state", ["roomy", "skipped", "off", "not_run", "failed"])
def test_a_start_that_made_no_room_leaves_the_context_to_the_bootstrap(monkeypatch, state):
    """The early context follows the room, so the off switch gives back the old start."""
    events: list = []
    _bridge(monkeypatch, events)
    monkeypatch.setattr(context_room, "make_room",
                        lambda: events.append("make_room") or context_room.Outcome(state))
    monkeypatch.setattr(context_room, "make_context",
                        lambda *_a, **_k: pytest.fail("no early context where no room was made"))
    context_room.ensure_comfy("/comfy", {})
    assert events == ["make_room", ("ensure_comfy", ("/comfy", {}), {})]


def test_the_bootstrap_reads_free_memory_just_ahead_of_a_context_left_to_it(monkeypatch, journal):
    """Memory freed can be taken again before the bootstrap's import asks the driver."""
    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    monkeypatch.setattr(context_room, "_LAST", MADE)
    context_room.at_bootstrap_context(lambda: (int(3.2 * _GIB), int(44.0 * _GIB)))
    (record,) = journal
    assert record.levelno == logging.INFO
    assert record.getMessage() == "3.2 GiB is free as the bootstrap makes the Worker's GPU context"
    block = context_room.status_block()
    assert block["free_after_release_gib"] == 15.8 and block["free_at_context_gib"] == 3.2
    assert block["context"] == "left to the bootstrap"


@pytest.mark.parametrize("last", [
    context_room.Outcome(), context_room.Outcome("roomy", 43 * _GIB, 98 * _GIB),
    context_room.Outcome("skipped", why="this GPU has memory of its own"),
    context_room.Outcome("made_room", _GIB, 45 * _GIB, 16 * _GIB, 2.0, 16 * _GIB,
                         context_made=True),
])
def test_the_bootstrap_reads_nothing_where_no_context_was_left_to_it(monkeypatch, journal, last):
    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    monkeypatch.setattr(context_room, "_LAST", last)
    context_room.at_bootstrap_context(lambda: pytest.fail("no reading is wanted here"))
    assert journal == [] and context_room._LAST is last


def test_the_bootstraps_reading_never_raises(monkeypatch, journal):
    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    monkeypatch.setattr(context_room, "_LAST", MADE)

    def broken():
        raise OSError("no /proc here")

    context_room.at_bootstrap_context(broken)
    assert [r.levelno for r in journal] == [logging.WARNING] and context_room._LAST is MADE
    context_room.at_bootstrap_context(lambda: None)
    assert context_room._LAST is MADE


def test_a_refusal_from_the_bootstrap_carries_what_was_free_when_it_asked(monkeypatch):
    from dgx_monarch.actor import comfy_bridge

    def bootstrap(*args, **kwargs):
        context_room.at_bootstrap_context(lambda: (int(3.2 * _GIB), int(44.0 * _GIB)))
        raise RuntimeError(OOM)

    def made():
        monkeypatch.setattr(context_room, "_LAST", MADE)
        return MADE

    monkeypatch.setattr(comfy_bridge, "ensure_comfy", bootstrap)
    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    monkeypatch.setattr(context_room, "make_room", made)
    monkeypatch.setattr(context_room, "meminfo", lambda: (int(3.0 * _GIB), int(44.0 * _GIB)))
    with pytest.raises(context_room.ContextMemoryError) as raised:
        context_room.ensure_comfy("/comfy", {residency_mode.WORKER_ARG: True})
    assert ("released it, after which 15.8 GiB was free; 3.2 GiB was free when the context was "
            "asked for, and the context still could not be made") in str(raised.value)


def test_stage_a_reads_it_as_its_last_act_and_the_import_follows_at_once():
    """Nothing may sit between the reading and the import that makes the context."""
    dynamic = ast.parse((SRC / "actor" / "comfy_dynamic.py").read_text())
    stage_a = next(n for n in ast.walk(dynamic)
                   if isinstance(n, ast.FunctionDef) and n.name == "stage_a")
    assert ast.unparse(stage_a.body[-2]) == "_context_reading()"
    assert ast.unparse(stage_a.body[-1]) == "return True"
    (off,) = (n for n in stage_a.body if isinstance(n, ast.If) and ast.unparse(n.test) == "not want")
    assert [ast.unparse(n) for n in off.body] == ["_context_reading()", "return False"]
    reading = next(n for n in ast.walk(dynamic)
                   if isinstance(n, ast.FunctionDef) and n.name == "_context_reading")
    assert ast.unparse(reading.body[-1]) == "context_room.at_bootstrap_context()"
    bridge = ast.parse((SRC / "actor" / "comfy_bridge.py").read_text())
    ensure = next(n for n in ast.walk(bridge)
                  if isinstance(n, ast.FunctionDef) and n.name == "ensure_comfy")
    at = next(i for i, n in enumerate(ensure.body) if "stage_a(" in ast.unparse(n))
    assert ast.unparse(ensure.body[at + 1]) == "import comfy.model_management"


def test_a_context_that_cannot_be_made_after_the_release_refuses_at_once(monkeypatch):
    events: list = []
    failure = RuntimeError(OOM)

    def create():
        raise failure

    _made_room(monkeypatch, events, create)
    monkeypatch.setattr(context_room, "meminfo", lambda: (int(15.8 * _GIB), int(45.0 * _GIB)))
    with pytest.raises(context_room.ContextMemoryError) as raised:
        context_room.ensure_comfy("/comfy", {}, rank=(1, 2))
    assert raised.value.__cause__ is failure
    assert events == ["make_room"], "one try at the context: the bootstrap does not run"
    text, rank = read_rank(str(raised.value))
    assert rank == 1, "the refusal carries its rank, so the driver can name the box"
    assert "it wrote to 16.0 GiB of memory and released it, after which 15.8 GiB was free" in text


def test_the_driver_names_the_box_of_a_worker_that_could_not_start(monkeypatch):
    def create():
        raise RuntimeError(OOM)

    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    monkeypatch.setattr(context_room, "meminfo", lambda: (int(4.6 * _GIB), int(47.4 * _GIB)))
    with pytest.raises(context_room.ContextMemoryError) as raised:
        context_room.make_context(MADE, (1, 2), create=create)
    wrapper = boxes._remote(raised.value)
    worker_box.name(wrapper, boxes._pair(monkeypatch))
    inner = str(wrapper.exception)
    assert inner.startswith("[dgxm:C guard=gpu_context_memory waivable=0] On the sibling, this "
                            "Worker could not start on its GPU"), inner
    assert read_rank(inner)[1] is None


def test_a_single_rank_marks_nothing(monkeypatch):
    def create():
        raise RuntimeError(OOM)

    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    monkeypatch.setattr(context_room, "meminfo", lambda: (int(4.6 * _GIB), int(47.4 * _GIB)))
    with pytest.raises(context_room.ContextMemoryError) as raised:
        context_room.make_context(MADE, (0, 1), create=create)
    assert read_rank(str(raised.value))[1] is None


def test_another_fault_in_the_early_context_leaves_it_to_the_bootstrap(monkeypatch, journal):
    events: list = []

    def create():
        raise RuntimeError("CUDA error: no CUDA-capable device is detected")

    _made_room(monkeypatch, events, create)
    context_room.ensure_comfy("/comfy", {})
    assert events == ["make_room", ("ensure_comfy", ("/comfy", {}), {})]
    assert [record.levelno for record in journal] == [logging.WARNING]
    assert "the bootstrap makes it as before" in journal[0].getMessage()


def test_an_out_of_memory_in_the_bootstrap_after_the_worker_made_the_context_is_not_this_refusal(
        monkeypatch):
    events: list = []
    failure = RuntimeError(OOM)
    _made_room(monkeypatch, events, lambda: events.append("context"), error=failure)
    with pytest.raises(RuntimeError) as raised:
        context_room.ensure_comfy("/comfy", {})
    assert raised.value is failure


def test_setup_hands_the_bootstrap_its_rank_and_the_driver_names_setup_errors():
    setup = next(node for node in ast.walk(ast.parse((SRC / "actor/worker_env.py").read_text()))
                 if isinstance(node, ast.FunctionDef) and node.name == "setup_impl")
    (call,) = [node for node in ast.walk(setup) if isinstance(node, ast.Call)
               and getattr(node.func, "id", None) == "ensure_comfy"]
    assert ast.unparse(next(kw.value for kw in call.keywords if kw.arg == "rank")) == "(rank, world)"
    transition = (SRC / "mesh_setup_transition.py").read_text()
    handler = transition[transition.index("except BaseException as setup_exc:"):]
    assert handler.index("worker_box.name(setup_exc, handle,") < handler.index("setup rollback")


@pytest.mark.usefixtures("isolated_environ")
def test_a_rank_whose_first_cuda_call_fails_refuses_before_it_joins_the_group(monkeypatch):
    """The Worker's own setup ends in the typed refusal and never reaches the rendezvous.

    The rank that cannot make its context answers at once; it is its peer that
    then waits for it (the driver's rollback, ``mesh_setup_transition``).
    """
    from dgx_monarch.actor import comfy_bridge

    def bootstrap(*_args, **_kwargs):
        raise RuntimeError(OOM)

    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    monkeypatch.setattr(comfy_bridge, "ensure_comfy", bootstrap)
    monkeypatch.setattr(context_room, "make_room", lambda: context_room.Outcome(
        "skipped", int(8.85 * _GIB), int(20.0 * _GIB), why="too little was available"))
    monkeypatch.setattr(context_room, "meminfo", lambda: (int(8.85 * _GIB), int(20.0 * _GIB)))
    monkeypatch.setattr("dgx_monarch.config.fixup_fabric_ifaces", lambda value: (value, ""))
    monkeypatch.setattr(worker_env.rendezvous, "generation_store",
                        lambda *a, **k: pytest.fail("the group setup must not begin"))
    worker = types.SimpleNamespace(_setup_key=None, _setup_cleanup_failed=False,
                                   rank=None, world=None, topology={})
    env = {"setup_generation": 1, "world": 2, "rank": 0, "gpus_per_host": 1,
           "local_gpu_index": 0, "master_addr": "192.0.2.1", "master_port": 29777,
           "topology": {"ulysses": 2, "ring": 1, "cfg": 1, "dp": 1, "fsdp": False},
           "fabric_env": {}, "comfy_dir": "/comfy", "worker_args": {}}
    with pytest.raises(context_room.ContextMemoryError) as raised:
        worker_env.setup_impl(worker, env)
    text, rank = read_rank(str(raised.value))
    assert rank == 0 and worker._setup_key is None and worker._setup_cleanup_failed is True
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY
    assert "8.8 GiB of memory is free on this Worker's box and 20.0 GiB is available" in text
    assert "The Worker made no room first: too little was available." in text


def test_setup_names_the_box_of_the_rank_whose_answer_it_awaited(monkeypatch):
    """Setup goes out a rank at a time, so the driver knows whose error it holds.

    A Fleet actor is rank 0 of its own world and marks nothing; the driver
    names its box all the same.
    """
    from topology_transition_helpers import _acks, _Future, _handle, _Workers

    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    monkeypatch.setattr(context_room, "meminfo", lambda: (int(4.6 * _GIB), int(47.4 * _GIB)))
    with pytest.raises(context_room.ContextMemoryError) as raised:
        context_room.refuse_unmade_context(RuntimeError(OOM), MADE, (0, 1))
    assert read_rank(str(raised.value))[1] is None
    wrapper = boxes._remote(raised.value, "setup")
    workers = _Workers([_Future(_acks()), _Future(_acks())],
                       setup_futures=[_Future({"rank": 0}), _Future(error=wrapper)])
    handle, _old, new = _handle(workers)
    with pytest.raises(type(wrapper)) as failed:
        handle.ensure_setup(new, "TORCH_FLASH")
    words = box_names.on_box(box_names.fleet_labels(handle, 2)[1])
    assert words.startswith("On GPU 1 on "), words
    inner = str(failed.value.exception)
    assert inner.startswith("[dgxm:C guard=gpu_context_memory waivable=0] " + words
                            + "this Worker could not start on its GPU"), inner
    assert len(workers.teardown_group.calls) == 2, "the rollback still runs"


def test_on_a_pair_a_peer_s_refusal_behind_rank_0_s_answer_is_not_read(monkeypatch):
    """Today's limit (DESIGN, the pair's wait): the driver raises the answer it awaits.

    When the sibling cannot make its context, the head waits at its store, and the
    driver, awaiting the head, raises the head's answer; the sibling's refusal is
    never read. The first-failure item turns this test around.
    """
    from topology_transition_helpers import _acks, _Future, _handle, _Workers

    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    monkeypatch.setattr(context_room, "meminfo", lambda: (int(4.6 * _GIB), int(47.4 * _GIB)))
    with pytest.raises(context_room.ContextMemoryError) as raised:
        context_room.refuse_unmade_context(RuntimeError(OOM), MADE, (1, 2))
    timeout = boxes._remote(RuntimeError(
        "Timed out after 600 seconds waiting for clients. 1/2 clients joined."), "setup")
    workers = _Workers([_Future(_acks()), _Future(_acks())],
                       setup_futures=[_Future(error=timeout),
                                      _Future(error=boxes._remote(raised.value, "setup"))])
    handle, _old, new = _handle(workers)
    with pytest.raises(type(timeout)) as failed:
        handle.ensure_setup(new, "TORCH_FLASH")
    assert failed.value is timeout
    assert "could not start on its GPU" not in str(failed.value.exception)


def test_a_fault_in_the_naming_never_stops_the_setup_rollback(monkeypatch):
    """ensure_setup words the error before its rollback; the words must never cost the rollback."""
    from topology_transition_helpers import _acks, _Future, _handle, _Workers

    def broken(*_args, **_kwargs):
        raise RuntimeError("the labels could not be read")

    monkeypatch.setattr(worker_box, "_name", broken)
    wrapper = boxes._remote(RuntimeError("a Worker's own setup error"), "setup")
    workers = _Workers([_Future(_acks()), _Future(_acks())],
                       setup_futures=[_Future({"rank": 0}), _Future(error=wrapper)])
    handle, _old, new = _handle(workers)
    with pytest.raises(type(wrapper)) as failed:
        handle.ensure_setup(new, "TORCH_FLASH")
    assert failed.value is wrapper
    assert len(workers.teardown_group.calls) == 2, "the rollback still runs"


def test_a_driver_side_setup_error_is_left_as_it_is():
    from topology_transition_helpers import _acks, _Future, _handle, _Workers

    workers = _Workers([_Future(_acks()), _Future(_acks())],
                       setup_futures=[_Future({"rank": 0}), RuntimeError("dispatch failed")])
    handle, _old, new = _handle(workers)
    with pytest.raises(RuntimeError) as failed:
        handle.ensure_setup(new, "TORCH_FLASH")
    assert str(failed.value) == "dispatch failed"


def test_a_context_that_still_cannot_be_made_is_refused_in_words(monkeypatch):
    events: list = []
    failure = RuntimeError(OOM)
    _bridge(monkeypatch, events, error=failure)
    made = context_room.Outcome("made_room", int(1.6 * _GIB), int(45.1 * _GIB),
                                16 * _GIB, 2.0, int(15.8 * _GIB))
    monkeypatch.setattr(context_room, "make_room", lambda: made)
    monkeypatch.setattr(context_room, "meminfo", lambda: (int(15.8 * _GIB), int(45.0 * _GIB)))
    # Comfy-managed residency: the Worker makes no context of its own, so the
    # bootstrap's import is the try that fails.
    with pytest.raises(context_room.ContextMemoryError) as raised:
        context_room.ensure_comfy("/comfy", {residency_mode.WORKER_ARG: True})
    assert raised.value.__cause__ is failure
    text = str(raised.value)
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY
    assert tag.guard == context_room.GUARD and not tag.waivable
    assert text == (
        "[dgxm:C guard=gpu_context_memory waivable=0] this Worker could not start on its GPU: "
        "the NVIDIA driver could not get "
        "memory for a new CUDA context. 15.8 GiB of memory is free on this Worker's box and 45.0 "
        "GiB is available: the driver needs free memory, and the rest of what is available is page "
        "cache, files the system has read and still keeps. Before that the Worker made room: "
        "it wrote to 16.0 GiB of memory and released it, after which 15.8 GiB was free, and "
        "the context still could not be made, so free memory may not be all the driver "
        "lacks. Nothing was loaded. Close a program that holds "
        "memory, or wait for a large download or file copy to end, then click Reset attached "
        "mesh and queue again. See docs/TROUBLESHOOTING.md #12.")
    assert "Traceback" not in text and "cuDevicePrimaryCtxRetain" not in text


def _refused(monkeypatch, outcome, reading) -> str:
    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    monkeypatch.setattr(context_room, "meminfo", lambda: reading)
    with pytest.raises(context_room.ContextMemoryError) as raised:
        context_room.refuse_unmade_context(RuntimeError(OOM), outcome)
    return str(raised.value)


@pytest.mark.parametrize(("outcome", "says"), [
    (context_room.Outcome("off", why=f"the step is off ({context_room.DISABLE_ENV}=1 on the Worker)"),
     f"The Worker made no room first: the step is off ({context_room.DISABLE_ENV}=1 on the Worker)."),
    (context_room.Outcome("skipped", why="this GPU has memory of its own"),
     "The Worker made no room first: this GPU has memory of its own."),
    (context_room.Outcome("roomy", 17 * _GIB, 40 * _GIB), "The Worker made no room first."),
])
def test_the_refusal_says_what_the_worker_did_first(monkeypatch, outcome, says):
    text = _refused(monkeypatch, outcome, (int(1.6 * _GIB), int(45.1 * _GIB)))
    assert "1.6 GiB of memory is free on this Worker's box and 45.1 GiB is available" in text
    assert says in text


def test_the_refusal_stands_where_memory_cannot_be_read(monkeypatch):
    text = _refused(monkeypatch, context_room.Outcome(), None)
    assert "The memory of this Worker's box could not be read." in text
    assert text.startswith("[dgxm:C guard=gpu_context_memory waivable=0] ")


def test_the_switch_reaches_the_actor_through_the_worker_loops_filter():
    """The worker loop deletes every name outside its list before actors inherit it."""
    from dgx_monarch.cli import worker_process_env

    environment = {"PATH": "/usr/bin", context_room.DISABLE_ENV: "1"}
    assert worker_process_env.filter_worker_environment(environment) == 0
    assert environment[context_room.DISABLE_ENV] == "1"


def test_the_guard_is_in_the_vocabulary_and_offers_no_card():
    from dgx_monarch.refusal import GUARDS, WAIVABLE_GUARDS

    spec = GUARDS[context_room.GUARD]
    assert spec.refusal_class is RefusalClass.CAPACITY and not spec.waivable_now
    assert context_room.GUARD not in WAIVABLE_GUARDS


@pytest.mark.parametrize("error", [
    FileNotFoundError("ComfyUI not found at '/comfy' on this host."),
    RuntimeError("CUDA error: no CUDA-capable device is detected"),
    ImportError("No module named 'comfy'"),
    MemoryError("out of memory"),
])
def test_any_other_bootstrap_error_passes_unchanged(monkeypatch, error):
    events: list = []
    _bridge(monkeypatch, events, error=error)
    with pytest.raises(type(error)) as raised:
        context_room.ensure_comfy("/comfy")
    assert raised.value is error


def test_an_out_of_memory_after_the_context_exists_is_not_this_refusal(monkeypatch):
    """Once comfy.model_management has imported, the context was made: a later CUDA
    out-of-memory in the bootstrap is another failure and keeps its own words."""
    from dgx_monarch.actor import comfy_bridge

    failure = RuntimeError(OOM)

    def ensure(*args, **kwargs):
        sys.modules["comfy.model_management"] = types.ModuleType("mm")
        raise failure

    monkeypatch.delitem(sys.modules, "comfy.model_management", raising=False)
    monkeypatch.setattr(comfy_bridge, "ensure_comfy", ensure)
    monkeypatch.setattr(context_room, "make_room", lambda: context_room.Outcome("roomy"))
    try:
        with pytest.raises(RuntimeError) as raised:
            context_room.ensure_comfy("/comfy")
        assert raised.value is failure
    finally:
        sys.modules.pop("comfy.model_management", None)


def test_the_worker_status_carries_the_step():
    tree = ast.parse((SRC / "actor/worker_status.py").read_text())
    keys = {node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert "context_room" in keys
    assert context_room.status_block() == {"state": "not_run"}
