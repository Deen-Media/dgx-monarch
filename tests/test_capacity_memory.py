"""The shmem finding: this kernel does not credit the weight slab to MemAvailable.

Every price in the tree reads MemAvailable and none subtracts Shmem. That holds
only while the kernel keeps shmem off the file LRU, so these tests measure it:
against a pinned GB10 /proc/meminfo, against the 544 recorded rows in
tests/fixtures, and against the arithmetic a naive correction would use.

The corpus is the three followups memory samplers of 2026-08-12 and 2026-09-02,
the only records that carry MemTotal, MemFree, MemAvailable, Cached, Shmem and
AnonPages together, with host names stripped and timestamps made relative
(docs/VALIDATION.md, 2026-09-03 shmem record). The sweep's per-cell sampler
records two fields per host block and cannot test any of this; it is pinned
here only as the field-poor shape the probe must abstain on.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from dgx_monarch import telemetry
from dgx_monarch.capacity_memory import PROC_MEMINFO, read_meminfo, shmem_credited

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
MODULE = REPO / "src" / "dgx_monarch" / "capacity_memory.py"
GIB = 1 << 30

# Shmem above this marks a recorded row that holds the 60 GiB weight slab. Below
# it a row holds no slab (Shmem near zero), a slab still filling or draining, or
# a steady plateau of a few GiB; the naive correction fails on the placed slab.
SLAB_RESIDENT_GIB = 30.0


def _corpus() -> list[dict]:
    text = (FIXTURES / "meminfo_credit_corpus.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _sweep_rows() -> list[dict]:
    text = (FIXTURES / "meminfo_sweep_two_field.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _synthetic(**overrides: float) -> dict[str, float]:
    """A meminfo shaped like this kernel's, in bytes, before overrides."""
    base = {
        "MemTotal": 121.63 * GIB,
        "MemFree": 37.0 * GIB,
        "MemAvailable": 84.0 * GIB,
        "Cached": 50.4 * GIB,
        "Buffers": 0.9 * GIB,
        "Shmem": 6.2 * GIB,
        "AnonPages": 11.6 * GIB,
        "Active(anon)": 15.4 * GIB,
        "Inactive(anon)": 2.4 * GIB,
        "Active(file)": 0.8 * GIB,
        "Inactive(file)": 44.5 * GIB,
    }
    base.update(overrides)
    return base


def test_shmem_is_not_credited_on_this_kernel():
    """The finding, against a pinned reading from a GB10 box."""
    meminfo = read_meminfo(str(FIXTURES / "meminfo_gb10.txt"))
    assert meminfo is not None
    credit = shmem_credited(meminfo)
    assert credit.complete is True
    assert credit.credited is False
    assert meminfo["Shmem"] > 0, "a zero-shmem reading proves nothing about the LRU"

    # The mechanism too: shmem sits on the anon LRU, so the anon counters carry
    # it over AnonPages and the file counters do not.
    anon_excess = credit.anon_lru - meminfo["AnonPages"]
    assert abs(anon_excess - meminfo["Shmem"]) < 0.02 * meminfo["Shmem"]
    file_pages = meminfo["Cached"] + meminfo["Buffers"] - meminfo["Shmem"]
    assert abs(credit.file_lru - file_pages) < 0.02 * file_pages


def test_probe_fires_when_shmem_is_on_the_file_lru():
    """A kernel with shmem on the file LRU, where this module would be wrong, answers True."""
    moved = _synthetic()
    moved["Active(file)"] += moved["Shmem"]
    moved["Active(anon)"] -= moved["Shmem"]
    credit = shmem_credited(moved)
    assert credit.credited is True
    assert credit.complete is True


def test_probe_is_false_at_zero_shmem():
    """At zero Shmem the comparison would flip on rounding alone, so it answers False."""
    idle = _synthetic(Shmem=0.0)
    idle["Inactive(file)"] = idle["Cached"] + idle["Buffers"] - idle["Active(file)"]
    assert shmem_credited(idle).credited is False


def test_probe_is_none_on_a_field_poor_read():
    """The sweep's two-field shape: no LRU counters, so no answer."""
    rows = _sweep_rows()
    assert rows, "the recorded sweep shape is the point of the fixture"
    for row in rows:
        assert set(row) == {"box", "t", "MemAvailable", "AnonPages"}
        credit = shmem_credited(row)
        assert credit.credited is None
        assert credit.complete is False
        assert credit.file_lru is None and credit.anon_lru is None


def test_unreadable_meminfo_returns_none(tmp_path):
    """No reading is not zero, the same rule the capacity_fit and mesh_safety
    probes follow when MemAvailable is unreadable: they make no claim."""
    assert read_meminfo(str(tmp_path / "absent")) is None
    assert read_meminfo(str(tmp_path)) is None
    assert shmem_credited(None).credited is None


def test_read_meminfo_returns_sized_rows_in_bytes():
    meminfo = read_meminfo(str(FIXTURES / "meminfo_gb10.txt"))
    assert meminfo is not None
    assert meminfo["MemTotal"] == 127533828 * 1024
    assert meminfo["Shmem"] == 6491924 * 1024
    # Unitless rows are page counts, not sizes, so they stay out of a mapping
    # whose every value is bytes.
    assert "HugePages_Total" not in meminfo
    assert "Hugepagesize" in meminfo


@pytest.mark.skipif(not Path(PROC_MEMINFO).exists(), reason="no procfs")
def test_the_live_reading_carries_the_probe_fields():
    meminfo = read_meminfo()
    assert meminfo is not None
    assert meminfo["MemAvailable"] > 0
    shmem_credited(meminfo)  # never raises on a real reading


def test_recorded_rows_reproduce_the_kernel_credit():
    """MemAvailable minus MemFree is the file cache with Shmem taken out.

    Measured over the 544 rows: max deviation 1.19 GiB, median 0.71, which is
    the kernel's watermark and reserve haircut. A kernel that credited shmem
    would show tens of GiB here, so rows recorded on such a kernel fail this test.
    """
    worst = 0.0
    for row in _corpus():
        deviation = abs(row["MemAvailable"] - row["MemFree"] - (row["Cached"] - row["Shmem"]))
        worst = max(worst, deviation)
    assert worst < 2.0, f"the kernel's cache credit moved: worst deviation {worst:.2f} GiB"


def test_naive_shmem_subtraction_is_what_this_module_refuses_to_do():
    """Never subtract Shmem from MemAvailable; the recorded rows show why.

    `MemAvailable - Shmem` on a box holding a placed slab is negative on 97 of
    the 110 recorded rows, worst -51.91 GiB. A price built on it refuses every
    load, of any size, on a box that already holds one.
    """
    resident = [row for row in _corpus() if row["Shmem"] > SLAB_RESIDENT_GIB]
    assert len(resident) == 110
    deducted = sorted(row["MemAvailable"] - row["Shmem"] for row in resident)
    assert sum(1 for value in deducted if value < 0) == 97
    assert deducted[0] == pytest.approx(-51.91, abs=0.01)
    # What the runtime reads on those rows is positive throughout, and one of
    # them was rendering at the time.
    assert all(row["MemAvailable"] > 0 for row in resident)


def test_availability_can_sit_below_free():
    """si_mem_available() starts from free pages minus the reserve, so
    MemAvailable below MemFree is correct, not a bad reading. An invariant
    raising these rows to MemFree would invent memory."""
    below = [row for row in _corpus() if row["MemAvailable"] < row["MemFree"]]
    assert len(below) == 7
    worst = min(row["MemAvailable"] - row["MemFree"] for row in below)
    assert worst == pytest.approx(-0.37, abs=0.01)
    for row in below:
        assert shmem_credited(row).credited is None  # no LRU fields, no claim


def test_telemetry_reads_the_sampler_fields():
    """Append-only: MemAvailable stays, and every appended field this kernel
    offers reaches the row a recorded sample is built from."""
    raw = Path(PROC_MEMINFO).read_text() if Path(PROC_MEMINFO).exists() else ""
    mem = telemetry._meminfo()
    assert "MemAvailable" in mem
    for field in ("Dirty", "Writeback", "Active(anon)", "Inactive(anon)",
                  "Active(file)", "Inactive(file)"):
        if f"{field}:" in raw:
            assert field in mem, f"{field} is readable here and did not reach the row"


def test_telemetry_survives_a_partial_meminfo(monkeypatch):
    """A status row that raises is the silent-rank path, so the probe may cost
    only its own key: None when the reading lacks the LRU fields or is missing,
    and absent when the read raises."""
    # Stub the GPU and pool readings (each can shell out to nvidia-smi); this
    # test reads the memory plane only.
    monkeypatch.setattr(telemetry, "_gpu_stats", dict)
    monkeypatch.setattr(telemetry, "_gpu_proc_gib", lambda: 0.0)
    monkeypatch.setattr(telemetry, "_retained_pool_gib", lambda: 0.0)
    monkeypatch.setattr(telemetry, "read_meminfo",
                        lambda *args, **kwargs: {"MemAvailable": 84 * GIB})
    assert telemetry.host_stats()["shmem_credited"] is None

    monkeypatch.setattr(telemetry, "read_meminfo", lambda *args, **kwargs: None)
    assert telemetry.host_stats()["shmem_credited"] is None

    def _explode(*args, **kwargs):
        raise RuntimeError("procfs went away")

    monkeypatch.setattr(telemetry, "read_meminfo", _explode)
    stats = telemetry.host_stats()
    assert "shmem_credited" not in stats
    assert stats["host"] and "mem_gib" in stats


def test_capacity_memory_is_a_pure_leaf():
    """capacity_memory imports only the standard library and the package
    logger: it measures the kernel, and every price stays in capacity_fit and
    mesh_safety."""
    tree = ast.parse(MODULE.read_text())
    imported: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))
    local = [name for name in imported if name.startswith(".")]
    assert local == [".log"], f"the leaf grew a package import: {local}"
    for name in imported:
        if name.startswith("."):
            continue
        root = name.split(".")[0]
        assert root in sys.stdlib_module_names, f"{name} is not standard library"
