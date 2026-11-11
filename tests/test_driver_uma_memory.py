"""driver_uma_memory.py: ComfyUI in the driver reads free memory from MemAvailable.

On unified memory ``torch.cuda.mem_get_info`` follows MemFree, so ComfyUI's
batch and tile choices swing with the page cache. The driver's correction
keeps comfy-aimdo's floor, leaves every other device to ComfyUI's own
function, and never stacks on itself or on an actor's patch. The last tests
run real ComfyUI on CPU (skipped where it is not importable).
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

from dgx_monarch import driver_uma_memory as uma
from module_location_helpers import from_checkout

GIB = 2 ** 30
MIB = 2 ** 20
SPARK_TOTAL = 124542 * MIB  # what comfy-aimdo logs for GB10


class Device:
    def __init__(self, kind, index=0):
        self.type, self.index = kind, index

    def __str__(self):
        return f"{self.type}:{self.index}"


INTEGRATED, DISCRETE, CPU = Device("cuda", 0), Device("cuda", 1), Device("cpu")


def _stock(device_free, cached_free=3 * GIB):
    """ComfyUI's reading: the device figure (which follows MemFree) plus the allocator's cached-free bytes."""
    calls = []

    def get_free_memory(dev=None, torch_free_too=False):
        calls.append((dev, torch_free_too))
        total = device_free + cached_free
        return (total, cached_free) if torch_free_too else total

    get_free_memory.calls = calls
    return get_free_memory


def _total_of(device):
    return SPARK_TOTAL if device is INTEGRATED else None


def _reading(stock, available):
    return uma.corrected_reading(stock, _total_of, lambda: INTEGRATED, lambda: available)


def test_the_floor_is_comfy_aimdos_rule_for_an_integrated_device():
    # The driver log of 2026-11-11: "comfy-aimdo integrated Linux GPU RAM headroom: 7783 MB".
    assert uma.floor_bytes(SPARK_TOTAL) // MIB == 7783
    assert uma.floor_bytes(16 * GIB) == 2 * GIB   # a sixteenth would be 1 GiB: the 2 GiB minimum
    assert uma.floor_bytes(256 * GIB) == 8 * GIB  # a sixteenth would be 16 GiB: the 8 GiB maximum
    assert uma.device_free_bytes(1 * GIB, SPARK_TOTAL) == 0  # never negative


def test_a_full_page_cache_no_longer_shrinks_the_reading():
    """MemFree at 5 GiB with 66 GiB available: stock reads 8, the driver reads 66 less the floor."""
    stock = _stock(device_free=5 * GIB)
    reading = _reading(stock, available=66 * GIB)
    floor = uma.floor_bytes(SPARK_TOTAL)
    assert stock(INTEGRATED) == 8 * GIB
    assert reading(INTEGRATED) == 66 * GIB - floor + 3 * GIB
    assert reading() == reading(INTEGRATED)  # the default device
    assert reading(INTEGRATED, True) == (66 * GIB - floor + 3 * GIB, 3 * GIB)
    assert reading(INTEGRATED, torch_free_too=True)[1] == 3 * GIB  # the allocator term is stock's


def test_on_an_empty_box_the_reading_is_stocks_less_the_floor():
    """The oracle: where MemFree equals MemAvailable the two differ by the floor and nothing else."""
    for free in (20 * GIB, 66 * GIB, 110 * GIB):
        stock = _stock(device_free=free)
        reading = _reading(stock, available=free)
        assert reading(INTEGRATED) == stock(INTEGRATED) - uma.floor_bytes(SPARK_TOTAL)


@pytest.mark.parametrize("device", [DISCRETE, CPU], ids=["a discrete GPU", "the CPU"])
def test_any_other_device_gets_comfyuis_own_reading(device):
    stock = _stock(device_free=5 * GIB)
    reading = _reading(stock, available=66 * GIB)
    assert reading(device) == stock(device) == 8 * GIB
    assert reading(device, True) == (8 * GIB, 3 * GIB)
    assert (device, True) in stock.calls and (device, False) in stock.calls


def test_vae_batches_stop_following_memfree():
    """ComfyUI's own arithmetic (sd.py, VAE.encode): batch = free / estimate, at least 1."""
    need = int(3.45 * GIB)  # one 1024 view, BF16, by ComfyUI's estimate
    splits = {"stock": set(), "driver": set()}
    for memfree in (4, 7, 12, 23, 66):  # MemFree over a session; MemAvailable stays near 66
        stock = _stock(device_free=memfree * GIB, cached_free=0)
        splits["stock"].add(max(1, int(stock(INTEGRATED) / need)))
        splits["driver"].add(max(1, int(_reading(stock, available=66 * GIB)(INTEGRATED) / need)))
    assert splits["stock"] == {1, 2, 3, 6, 19}
    assert splits["driver"] == {16}  # six views are one batch every time


def _fake_comfy(monkeypatch, *, integrated=True, kind="cuda", probe_error=None, platform="linux"):
    stock = _stock(device_free=5 * GIB)
    device = Device(kind)

    def properties(_device):
        if probe_error is not None:
            raise probe_error
        return SimpleNamespace(is_integrated=1 if integrated else 0, total_memory=SPARK_TOTAL)

    mm = SimpleNamespace(get_free_memory=stock, get_torch_device=lambda: device)
    torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=properties))
    monkeypatch.setattr(uma, "_comfy", lambda: (mm, torch))
    monkeypatch.setattr(uma, "host_available_bytes", lambda: 66 * GIB)
    monkeypatch.setattr(uma.sys, "platform", platform)
    monkeypatch.delenv(uma.DISABLE_ENV, raising=False)
    return mm, stock, device


def test_install_on_an_integrated_gpu_and_only_once(monkeypatch, caplog):
    mm, stock, device = _fake_comfy(monkeypatch)
    with caplog.at_level("INFO"):
        assert uma.install_driver() == "installed"
    installed = mm.get_free_memory
    assert installed is not stock and installed.__wrapped__ is stock
    assert "MemAvailable less a 7.6 GiB floor" in caplog.text and uma.DISABLE_ENV in caplog.text
    assert uma.install_driver() == "already" and mm.get_free_memory is installed  # never stacked
    assert getattr(installed, uma.DRIVER_MARK) is True
    assert installed(device) == 66 * GIB - uma.floor_bytes(SPARK_TOTAL) + 3 * GIB
    assert installed() == installed(device)


@pytest.mark.parametrize("case, expected", [
    ("discrete", "not_integrated"), ("cpu", "not_integrated"), ("off", "off"),
    ("not linux", "not_linux"), ("probe fails", "probe_failed"), ("actor patch in place", "already"),
    ("no comfy", "no_comfy")])
def test_install_leaves_comfyuis_reading_alone(monkeypatch, case, expected):
    kwargs = {"discrete": {"integrated": False}, "cpu": {"kind": "cpu"},
              "not linux": {"platform": "win32"},
              "probe fails": {"probe_error": RuntimeError("no CUDA device")}}.get(case, {})
    mm, stock, _device = _fake_comfy(monkeypatch, **kwargs)
    if case == "off":
        monkeypatch.setenv(uma.DISABLE_ENV, "1")
    if case == "actor patch in place":
        setattr(stock, uma.ACTOR_MARK, True)
    if case == "no comfy":
        def missing():
            raise ImportError("No module named 'comfy'")
        monkeypatch.setattr(uma, "_comfy", missing)
    assert uma.install_driver() == expected
    assert mm.get_free_memory is stock


class _RecordComfyImports:
    """A meta-path finder that records every attempt to import ComfyUI and finds nothing."""

    def __init__(self):
        self.seen = []

    def find_spec(self, name, path=None, target=None):
        if name == "comfy" or name.startswith("comfy."):
            self.seen.append(name)
        return None


def _no_comfy(monkeypatch):
    for name in [n for n in sys.modules if n == "comfy" or n.startswith("comfy.")]:
        monkeypatch.delitem(sys.modules, name)
    finder = _RecordComfyImports()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    return finder


def test_install_imports_nothing_of_comfyui_itself(monkeypatch):
    """A process that has not imported ComfyUI's model management is left to do so in its own time.

    The finder records an attempt and finds nothing; it must record, not raise,
    because the install swallows an import error.
    """
    monkeypatch.delenv(uma.DISABLE_ENV, raising=False)
    monkeypatch.setattr(uma.sys, "platform", "linux")
    finder = _no_comfy(monkeypatch)
    assert uma.install_driver() == "no_comfy"
    assert uma.host_available_bytes() > 0  # psutil's figure, with no ComfyUI reader to ask
    assert finder.seen == []


def test_the_available_figure_is_comfyuis_own_reader_where_the_process_has_it(monkeypatch):
    """ComfyUI's reader knows a cgroup limit; psutil does not. The process's own reader answers."""
    reader = SimpleNamespace(virtual_memory_available=lambda: 12345)
    monkeypatch.setitem(sys.modules, "comfy.system_memory", reader)
    assert uma.host_available_bytes() == 12345


def test_an_argument_or_answer_the_wrapper_does_not_know_goes_to_comfyuis_own_reading():
    """Upstream drift: a keyword its callers pass, or a longer answer, must not break a render."""
    def newer(dev=None, torch_free_too=False, for_dynamic=False):
        return (8 * GIB, 3 * GIB, 0) if torch_free_too else 8 * GIB

    reading = uma.corrected_reading(newer, _total_of, lambda: INTEGRATED, lambda: 66 * GIB)
    assert reading(INTEGRATED, for_dynamic=True) == 8 * GIB
    assert reading(INTEGRATED, True) == (8 * GIB, 3 * GIB, 0)  # upstream's shape, for upstream's callers
    assert reading(INTEGRATED) == 8 * GIB  # and upstream's reading from then on


def test_a_failed_correction_stays_failed():
    """The WARNING says ComfyUI's reading answers from here on; a transient failure must not flip back."""
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 2:
            raise OSError("transient")
        return 66 * GIB

    reading = uma.corrected_reading(_stock(device_free=5 * GIB), _total_of, lambda: INTEGRATED, flaky)
    assert reading(INTEGRATED) == 66 * GIB - uma.floor_bytes(SPARK_TOTAL) + 3 * GIB
    assert [reading(INTEGRATED) for _ in range(3)] == [8 * GIB] * 3


def test_the_entry_point_installs_both_driver_hooks():
    """The reading is on by default: the pack's entry point calls its install at node import."""
    import ast
    import pathlib

    tree = ast.parse((pathlib.Path(__file__).resolve().parents[1] / "__init__.py").read_text())
    top_level_calls = {f"{stmt.value.func.value.id}.{stmt.value.func.attr}" for stmt in tree.body
                       if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                       and isinstance(stmt.value.func, ast.Attribute)
                       and isinstance(stmt.value.func.value, ast.Name)}
    assert {"driver_uma_memory.install_driver", "vae_item_batches.install_driver"} <= top_level_calls


@pytest.mark.parametrize("broken", ["available", "total_of", "default_device"])
def test_a_failure_of_the_correction_answers_with_comfyuis_own_reading(broken, caplog):
    """The function runs at every batch choice of the user's graph: it must never raise."""
    def fails(*_args):
        raise OSError("/proc/meminfo is gone")

    stock = _stock(device_free=5 * GIB)
    reading = uma.corrected_reading(
        stock, fails if broken == "total_of" else _total_of,
        fails if broken == "default_device" else (lambda: INTEGRATED),
        fails if broken == "available" else (lambda: 66 * GIB))
    device = None if broken == "default_device" else INTEGRATED
    with caplog.at_level("WARNING"):
        assert reading(device) == stock(device) == 8 * GIB
        assert reading(device, True) == (8 * GIB, 3 * GIB)
    assert caplog.text.count("the unified-memory reading failed") == 1  # said once, not at every call


def test_an_actors_patch_replaces_the_drivers_reading(monkeypatch):
    """The actor's own patch (actor/comfy_bridge.py), run over a driver's reading.

    An actor process can import this pack's entry point (a Worker that loads
    custom nodes) or inherit a driver's module. Its patch looks for its own
    marker only, so it installs its reading whatever it finds, and a Worker's
    loads keep the reading its identity records were taken with: psutil's
    available memory over the actors on the host, no floor.
    """
    from dgx_monarch.actor import comfy_bridge

    device = Device("cuda")
    driver_reading = _reading(_stock(device_free=5 * GIB), available=66 * GIB)
    assert getattr(driver_reading, uma.DRIVER_MARK) is True
    mm = SimpleNamespace(get_free_memory=driver_reading, get_torch_device=lambda: device)
    comfy = SimpleNamespace(model_management=mm)
    torch = SimpleNamespace(cuda=SimpleNamespace(
        get_device_properties=lambda _device: SimpleNamespace(is_integrated=1),
        memory_stats=lambda _device: {"reserved_bytes.all.current": 5 * GIB, "active_bytes.all.current": 2 * GIB}))
    psutil = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=66 * GIB))
    for name, module in (("comfy", comfy), ("comfy.model_management", mm), ("torch", torch), ("psutil", psutil)):
        monkeypatch.setitem(sys.modules, name, module)

    comfy_bridge._patch_uma_free_memory(2)

    actor_reading = mm.get_free_memory
    assert actor_reading is not driver_reading
    assert getattr(actor_reading, uma.ACTOR_MARK) is True and not getattr(actor_reading, uma.DRIVER_MARK, False)
    assert actor_reading(device) == 66 * GIB // 2 + 3 * GIB  # over two actors, no floor
    assert actor_reading(device, True) == (66 * GIB // 2 + 3 * GIB, 3 * GIB)
    # And the other order: a driver install that meets an actor's reading leaves it.
    monkeypatch.setattr(uma, "_comfy", lambda: (mm, torch))
    monkeypatch.setattr(uma.sys, "platform", "linux")
    monkeypatch.delenv(uma.DISABLE_ENV, raising=False)
    assert uma.install_driver() == "already" and mm.get_free_memory is actor_reading


@pytest.fixture(scope="module")
def comfy_cpu():
    """Real ComfyUI with --cpu, restored afterwards."""
    def is_comfy(name):
        return name == "folder_paths" or name == "comfy" or name.startswith(("comfy.", "comfy_"))

    preserved = {name: module for name, module in sys.modules.items() if is_comfy(name)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before, argv_before = list(sys.path), sys.argv
    comfy_dir = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    try:
        sys.argv = ["pytest-driver-uma-memory", "--cpu"]
        try:
            options = pytest.importorskip("comfy.options", reason="no ComfyUI on sys.path")
            options.enable_args_parsing()
            mm = pytest.importorskip("comfy.model_management")
        finally:
            sys.argv = argv_before
        yield mm
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if is_comfy(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def test_real_comfyui_on_cpu_is_left_alone(comfy_cpu, monkeypatch):
    monkeypatch.delenv(uma.DISABLE_ENV, raising=False)
    before = comfy_cpu.get_free_memory
    assert uma.install_driver() == "not_integrated"
    assert comfy_cpu.get_free_memory is before


def test_real_comfyuis_vae_takes_its_batch_from_the_installed_reading(comfy_cpu, monkeypatch):
    """ComfyUI looks the function up at each call, so its VAE follows the corrected reading.

    The CPU stands in for the integrated device: a reading that fits one image
    a batch (MemFree after the page cache fills) splits six views six ways;
    the corrected one, with 66 GiB available, takes them in one batch.
    """
    import comfy.sd
    import torch

    torch.set_num_threads(2)
    vae = comfy.sd.VAE(sd={"decoder.conv_in.weight": torch.zeros(512, 4, 3, 3)})
    pixels = torch.rand(6, 64, 64, 3, generator=torch.Generator().manual_seed(6))
    need = vae.memory_used_encode((6, 3, 64, 64), vae.vae_dtype)
    sizes = []
    encode = vae.first_stage_model.encode
    monkeypatch.setattr(vae.first_stage_model, "encode",
                        lambda x, *args, **kwargs: (sizes.append(int(x.shape[0])), encode(x, *args, **kwargs))[1])

    def follows_memfree(dev=None, torch_free_too=False):
        free = int(need * 1.5)
        return (free, 0) if torch_free_too else free

    monkeypatch.setattr(comfy_cpu, "get_free_memory", follows_memfree)
    vae.encode(pixels)
    assert sizes == [1] * 6
    sizes.clear()
    monkeypatch.setattr(comfy_cpu, "get_free_memory", uma.corrected_reading(
        follows_memfree, lambda device: SPARK_TOTAL, comfy_cpu.get_torch_device, lambda: 66 * GIB))
    vae.encode(pixels)
    assert sizes == [6]


def test_real_comfyuis_reading_is_the_shape_the_wrapper_replaces(comfy_cpu):
    """A local pin on the term the wrapper replaces, until the forward-contracts canary lands."""
    import inspect

    assert list(inspect.signature(comfy_cpu.get_free_memory).parameters) == ["dev", "torch_free_too"]
    source = inspect.getsource(comfy_cpu.get_free_memory)
    assert "mem_free_cuda, _ = torch.cuda.mem_get_info(dev)" in source
    assert "mem_free_total = mem_free_cuda + mem_free_torch" in source
    assert "return (mem_free_total, mem_free_torch)" in source


def test_real_comfyuis_reading_keeps_its_signature_and_its_allocator_term(comfy_cpu):
    """What the wrapper relies on: (dev, torch_free_too) and a (total, cached) pair."""
    import torch

    total, cached = comfy_cpu.get_free_memory(torch.device("cpu"), True)
    assert isinstance(total, int | float) and total == cached  # the CPU device: one figure
    wrapped = uma.corrected_reading(comfy_cpu.get_free_memory, lambda device: None,
                                    comfy_cpu.get_torch_device, lambda: 0)
    assert wrapped(torch.device("cpu"), True)[0] > 0  # passed through, not zeroed by the fake available
    assert uma.host_available_bytes() > 0  # ComfyUI's own MemAvailable reader, or psutil
