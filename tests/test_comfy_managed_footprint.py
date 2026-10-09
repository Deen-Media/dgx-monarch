"""The pre-load capacity wall the comfy-managed rung charges its artifact
against: the host floor, the pinned-staging factor and the refusal text."""
from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace

import pytest

from comfy_managed_helpers import (  # noqa: F401  # autouse fixture import.
    GIB,
    _isolated_process_state,
    _tag,
)
from dgx_monarch import capacity_floor, mesh_safety, residency_mode
from dgx_monarch.actor import comfy_dynamic, store_residency
from dgx_monarch.refusal import RefusalClass


@pytest.fixture
def artifact(tmp_path):
    """A real file, because the wall reads its size off disk."""
    path = tmp_path / "m.safetensors"
    path.write_bytes(b"\0" * 4096)
    return path


def _check(*, rung, path, options=None, preflight=None, pinned=False):
    """Run the wall and report what it handed the classic preflight.

    ``pinned`` is always stated so no case depends on whether a ComfyUI happens
    to be importable in the test environment.
    """
    seen: list[tuple] = []

    def _record(load_path, unet_name, model_options):
        seen.append((load_path, unet_name, model_options))

    decision = store_residency.ResidencyDecision(
        False,
        store_residency.RUNG_COMFY_MANAGED if rung else store_residency.RUNG_STOCK_FITS,
        "reason", False)
    store_residency.preload_capacity_check(
        decision, str(path), "m.safetensors", options if options is not None else {},
        preflight=preflight or _record, pinned_staging=pinned)
    return seen


@pytest.fixture
def _integrated(monkeypatch):
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)


def _avail(monkeypatch, value):
    """Both probes resolve through the mesh_safety module at call time, so
    patching the module attribute reaches the wall."""
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: value)


def test_a_classic_rung_reprices_without_the_bare_file_wall(_integrated, monkeypatch, artifact):
    _avail(monkeypatch, 90 * GIB)
    assert _check(rung=False, path=artifact) == []


def test_the_rung_does_not_call_the_stock_wall(_integrated, monkeypatch, artifact):
    """The stock message offers slab_weights=on as the thing that might still
    fit, and this rung has forced that off. A wall that names an unreachable
    remedy is the bare refusal class C forbids."""
    _avail(monkeypatch, 90 * GIB)
    assert _check(rung=True, path=artifact) == []


def test_the_rung_charges_the_artifact_at_one_x_and_no_arena(
        _integrated, monkeypatch, tmp_path):
    """The rung's wall charges the whole artifact, not only a floor.

    DynamicVRAM places weights lazily, but that does not put the resident cost
    below the file: in the one stock ComfyUI measurement on this hardware
    (2026-08-05), a 61.7 GiB artifact drove a working set of 86.3 GiB and a
    MemAvailable floor of 28.4 GiB (docs/VALIDATION.md, comfy-managed residency
    acceptance; summarized in docs/TROUBLESHOOTING.md #62, "The capacity wall
    it keeps"). What the rung removes is the 0.85x legacy load arena, the only
    term dropped here.
    """
    big = tmp_path / "big.safetensors"
    with big.open("wb") as handle:      # sparse: 100 GiB of size, no disk
        handle.truncate(100 * GIB)
    _avail(monkeypatch, 60 * GIB)
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        _check(rung=True, path=big)
    # The arena is not charged: 1.0x plus the floor is the whole sum, where a
    # legacy load is charged 1.85x.
    _avail(monkeypatch, 100 * GIB + capacity_floor.ABSOLUTE_HOST_FLOOR_BYTES)
    assert _check(rung=True, path=big) == []


def test_the_rung_refuses_a_host_that_is_already_below_the_floor(
        _integrated, monkeypatch, artifact):
    floor = capacity_floor.ABSOLUTE_HOST_FLOOR_BYTES
    _avail(monkeypatch, floor // 2)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        _check(rung=True, path=artifact)
    text = str(excinfo.value)
    tag = _tag(text)
    assert tag is not None
    assert tag.refusal_class is RefusalClass.CAPACITY
    assert tag.guard == "stock_load_preflight"
    assert tag.waivable is False
    assert "docs/TROUBLESHOOTING.md #53" in text
    assert "nothing was quarantined" in text
    assert "comfy-managed" in text


def test_the_floor_is_the_one_the_stock_and_slab_prices_charge():
    """Every residency charges one floor with one derivation, since what the
    rest of the box needs does not depend on which residency placed the weights.
    While this wall charged 4 GiB and the stock and slab prices the derived
    5 GiB, one residency admitted a load the other refused on the same box
    (2026-09-04)."""
    from dgx_monarch import capacity_fit

    assert capacity_floor.ABSOLUTE_HOST_FLOOR_BYTES == 5 * GIB
    assert capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES is (
        capacity_floor.ABSOLUTE_HOST_FLOOR_BYTES)
    assert not hasattr(residency_mode, "COMFY_MANAGED_FLOOR_BYTES")
    # And the price is the artifact, its staging factor, and that floor.
    assert capacity_fit.managed_required_bytes(10 * GIB, False) == (
        10 * GIB + capacity_floor.ABSOLUTE_HOST_FLOOR_BYTES)
    assert capacity_fit.managed_required_bytes(10 * GIB, True) == (
        int(10 * GIB * residency_mode.PINNED_STAGING_FACTOR)
        + capacity_floor.ABSOLUTE_HOST_FLOOR_BYTES)


def test_the_floor_passes_when_the_host_has_room(_integrated, monkeypatch, artifact):
    _avail(monkeypatch, capacity_floor.ABSOLUTE_HOST_FLOOR_BYTES * 3)
    assert _check(rung=True, path=artifact) == []


def test_the_wall_skips_on_a_discrete_gpu(monkeypatch, artifact):
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: False)
    _avail(monkeypatch, 1)
    assert _check(rung=True, path=artifact) == []


def test_the_wall_skips_when_meminfo_is_unreadable(_integrated, monkeypatch, artifact):
    _avail(monkeypatch, None)
    assert _check(rung=True, path=artifact) == []


def test_the_wall_skips_under_a_dtype_cast(_integrated, monkeypatch, artifact):
    """Same skip semantics as stock_load_fit: a cast has a resident size the
    file does not predict, so this can never falsely refuse."""
    _avail(monkeypatch, 1)
    assert _check(rung=True, path=artifact, options={"dtype": "fp8_e4m3fn"}) == []


def test_pinned_staging_doubles_the_charge(_integrated, monkeypatch, tmp_path):
    """The staging copy is real memory and the wall has to charge it.

    A pinned host buffer is sized at twice the model and filled with a second
    copy of every weight. On unified memory that copy comes out of the same
    budget as the first, so the wall charges 2.3x the artifact instead of 1x.
    """
    big = tmp_path / "big.safetensors"
    with big.open("wb") as handle:          # sparse: 40 GiB of size, no disk
        handle.truncate(40 * GIB)
    room = 40 * GIB + capacity_floor.ABSOLUTE_HOST_FLOOR_BYTES
    _avail(monkeypatch, room)
    assert _check(rung=True, path=big, pinned=False) == []
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        _check(rung=True, path=big, pinned=True)
    _avail(monkeypatch, int(40 * GIB * store_residency.PINNED_STAGING_FACTOR)
           + capacity_floor.ABSOLUTE_HOST_FLOOR_BYTES)
    assert _check(rung=True, path=big, pinned=True) == []


def test_the_pinned_factor_is_the_measured_one():
    """A recalibration is an evidence change. Measured 2026-08-12 on a worker
    with nothing else resident: a 39.13 GiB BF16 checkpoint took MemAvailable
    from 115.14 GiB to 25.45 GiB, which is 2.29x the artifact."""
    assert store_residency.PINNED_STAGING_FACTOR == 2.3
    measured = (115.14 - 25.45) / 39.13
    assert measured <= store_residency.PINNED_STAGING_FACTOR
    assert round(measured, 2) == 2.29


def test_the_wall_decides_the_leg_that_killed_a_worker(
        _integrated, monkeypatch, tmp_path):
    """Adversarial reconstruction of the flux2 leg of 2026-08-12.

    A 60.02 GiB BF16 checkpoint was admitted against 77.27 GiB of MemAvailable
    by a wall that charged one copy plus the floor. The worker then staged it
    through a pinned host buffer and died in the NVIDIA driver's allocator.
    Under that pinned staging the wall refuses; under zero-copy staging the
    same leg fits. Fix such a refusal by changing the staging, not by raising
    the threshold.
    """
    checkpoint = tmp_path / "flux2-dev.safetensors"
    with checkpoint.open("wb") as handle:
        handle.truncate(int(60.02 * GIB))
    _avail(monkeypatch, int(77.27 * GIB))

    # The wall that admitted it: one copy plus a 4 GiB floor, whatever the staging.
    assert 60.02 + 4 <= 77.27

    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        _check(rung=True, path=checkpoint, pinned=True)
    text = str(excinfo.value)
    assert "60.0 GiB" in text and "77.3 GiB" in text
    assert "2.30x" in text
    assert "pinned" in text
    tag = _tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY

    assert _check(rung=True, path=checkpoint, pinned=False) == []


def test_the_refusal_names_the_shortfall_and_the_staging_it_priced(
        _integrated, monkeypatch, tmp_path):
    """Class C refusals carry their numbers, and the reader has to be able to
    tell which of the two chargeable staging modes produced them."""
    big = tmp_path / "big.safetensors"
    with big.open("wb") as handle:
        handle.truncate(50 * GIB)
    _avail(monkeypatch, 20 * GIB)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        _check(rung=True, path=big, pinned=False)
    zero_copy = str(excinfo.value)
    assert "35.0 GiB short" in zero_copy      # 50 + 5 - 20
    assert "page out of the checkpoint mapping" in zero_copy
    assert "pinned host buffer" not in zero_copy
    # The card says what the wall adds and that the floor is the shared one,
    # not this residency's own.
    assert "5.0 GiB host floor every residency on this box is priced over" in zero_copy
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        _check(rung=True, path=big, pinned=True)
    pinned = str(excinfo.value)
    assert "100.0 GiB short" in pinned        # 50 * 2.3 + 5 - 20
    assert "pinned host buffer" in pinned


def test_pinned_staging_is_read_off_comfys_own_budget(monkeypatch):
    """The wall prices what ComfyUI will do, not what a worker argument asked
    for: ``pinned_hostbuf_size`` multiplies by two exactly while
    ``MAX_PINNED_MEMORY`` is positive, so that is the number to read."""
    module = SimpleNamespace(MAX_PINNED_MEMORY=-1)
    monkeypatch.setitem(sys.modules, "comfy", SimpleNamespace(model_management=module))
    monkeypatch.setitem(sys.modules, "comfy.model_management", module)
    assert comfy_dynamic.pinned_staging_active() is False
    module.MAX_PINNED_MEMORY = 112090 * 1024 ** 2
    assert comfy_dynamic.pinned_staging_active() is True


def test_pinned_staging_reads_false_without_a_comfy(monkeypatch):
    """No ComfyUI, no claim: the floor still applies and the wall stays the
    more permissive of the two, which is the safe direction for a probe that
    could not run."""
    monkeypatch.setitem(sys.modules, "comfy", None)
    monkeypatch.setitem(sys.modules, "comfy.model_management", None)
    assert comfy_dynamic.pinned_staging_active() is False


def test_the_capacity_refusal_is_recognized_as_capacity(
        _integrated, monkeypatch, artifact):
    """The one refusal on this path that a ceremony should auto-skip.

    A leg that hits an already-full box records CAPACITY and moves on, the
    opposite of the rung's class P yields, which is why this arm reuses the
    existing worker-side stock-residency guard rather than inventing one.
    """
    _avail(monkeypatch, GIB)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        _check(rung=True, path=artifact)
    assert mesh_safety.is_stock_load_capacity_error(excinfo.value) is True


def test_the_model_store_routes_its_stock_arm_through_the_wall():
    """One line in the store's fresh-load path, and the module-global preflight
    name stays the one tests monkeypatch."""
    from dgx_monarch.actor import model_store, store_load

    assert "preload_capacity_check" in inspect.getsource(store_load)
    assert "stock_load_preflight" in inspect.getsource(model_store)
