"""Test the memory sampler and capacity-test allocator with bounded fixtures.

Production sampling uses SSH and capacity tests hold tens of GiB. These unit
tests use text parsing and injected memory readings; allocation fixtures use
MiB-sized blocks.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmark.sweep import memhog, memsample  # noqa: E402

# Three rows of a sampler file, the shape the runner writes: one timestamp, one
# leg label, and one dict of GiB floats per box.
ROWS = [
    {"t": 1.0, "leg": "m1-leg3",
     "head": {"MemAvailable": 70.5, "Shmem": 1.0, "AnonPages": 20.0,
              "Active(file)": 3.0, "Inactive(file)": 9.0},
     "second-box": {"MemAvailable": 110.0, "Shmem": 0.3, "AnonPages": 2.0,
                    "Active(file)": 1.0, "Inactive(file)": 4.0}},
    {"t": 2.0, "leg": "m1-leg3",
     "head": {"MemAvailable": 11.25, "Shmem": 63.5, "AnonPages": 21.0,
              "Active(file)": 2.0, "Inactive(file)": 1.5},
     "second-box": {"MemAvailable": 109.5, "Shmem": 0.3, "AnonPages": 2.0,
                    "Active(file)": 1.0, "Inactive(file)": 4.5}},
    {"t": 3.0, "leg": "m1-leg3",
     "head": {"MemAvailable": 12.0, "Shmem": 63.5, "AnonPages": 20.5,
              "Active(file)": 2.5, "Inactive(file)": 2.0},
     "second-box": {"MemAvailable": 110.25, "Shmem": 0.3, "AnonPages": 2.0,
                    "Active(file)": 1.25, "Inactive(file)": 4.0}},
]


def _lines(rows: list[dict]) -> list[str]:
    return [json.dumps(row) + "\n" for row in rows]


def test_the_summary_reads_the_four_leg_figures_on_every_box():
    stats = memsample.summarize(_lines(ROWS))
    assert stats["samples"] == 3 and stats["legs"] == ["m1-leg3"]
    head = stats["boxes"]["head"]
    # The low water inside the fill and the settled figure are the pair the
    # threshold legs are read on, so both survive into the summary.
    assert head["MemAvailable"] == {"min": 11.25, "max": 70.5, "last": 12.0}
    assert head["Shmem"] == {"min": 1.0, "max": 63.5, "last": 63.5}
    assert head["AnonPages"] == {"min": 20.0, "max": 21.0, "last": 20.5}
    # File pages are the sum of the two file LRU lists; no meminfo line carries
    # it, and the page cache question is stated in it.
    assert head["file"] == {"min": 3.5, "max": 12.0, "last": 4.5}
    assert stats["boxes"]["second-box"]["file"] == {"min": 5.0, "max": 5.5, "last": 5.25}


def test_the_summary_still_reads_a_file_written_before_the_fields_landed():
    old = [{"t": 1.0, "head": {"MemAvailable": 86.5, "AnonPages": 22.2}},
           {"t": 2.0, "head": {"MemAvailable": 80.0, "AnonPages": 30.1}}]
    stats = memsample.summarize(_lines(old))
    head = stats["boxes"]["head"]
    assert stats["legs"] == [] and head["MemAvailable"]["min"] == 80.0
    assert "Shmem" not in head and "file" not in head
    text = memsample.render_summary("old.jsonl", stats)
    assert "n/a, no such field in this file" in text
    assert "min     80.00" in text and "max     86.50" in text and "last     80.00" in text


def test_a_box_that_answered_nothing_is_no_row_in_the_summary():
    rows = [{"t": 1.0, "leg": "m4", "head": {"MemAvailable": 70.0}, "second-box": {}}]
    stats = memsample.summarize(_lines(rows))
    assert list(stats["boxes"]) == ["head"]


def test_the_host_list_takes_the_second_box_and_anything_named_by_hand():
    # The empty string is the local box, which is what the runner's host list
    # means by it, and a name given twice is still one box.
    assert memsample.hosts_for("second-box", []) == ["", "second-box"]
    assert memsample.hosts_for("", ["third-box"]) == ["", "third-box"]
    assert memsample.hosts_for("second-box", ["second-box"]) == ["", "second-box"]
    assert memsample.hosts_for("", []) == [""]


def test_the_summary_mode_reads_a_file_off_disk(tmp_path):
    path = tmp_path / "m1-leg3.jsonl"
    path.write_text("".join(_lines(ROWS)))
    text = memsample.summary_for(path)
    assert "samples 3, legs m1-leg3" in text and "min     11.25" in text


def _tiny_hog(baseline_gib: float, target_gib: float) -> memhog.Hog:
    """A hog against a fake box whose MemAvailable falls by whatever it holds.

    Every size is a few thousandths of a GiB: a 1 MiB block, an 8 MiB hold, a
    2 MiB band, where the real hog moves 64 MiB blocks inside a 0.5 GiB band.
    The arithmetic and the page touching are the same.
    """
    holder: dict = {}
    hog = memhog.Hog(target_gib, lambda: baseline_gib - holder["hog"].held_gib,
                     tolerance_gib=0.002, floor_gib=0.001,
                     block_gib=0.001, step_cap_gib=0.004)
    holder["hog"] = hog
    return hog


def _settle(hog: memhog.Hog, limit: int = 50) -> str:
    for _ in range(limit):
        verb, _available = hog.step()
        if verb in ("held", "starved"):
            return verb
    return "never settled"


def test_the_hog_holds_the_target_and_gives_it_all_back():
    hog = _tiny_hog(0.02, 0.012)
    assert _settle(hog) == "held"
    # 8 MiB to take a 20 MiB box down to 12, within the 2 MiB band.
    assert 0.006 <= hog.held_gib <= 0.01
    assert abs(hog.read_available() - 0.012) <= hog.tolerance
    assert hog.blocks and len(hog.blocks[0]) == int(0.001 * memhog.GIB)
    hog.release()
    assert hog.held_gib == 0.0 and hog.read_available() == 0.02


def test_the_hog_gives_memory_back_when_the_box_needs_it():
    hog = _tiny_hog(0.02, 0.012)
    assert _settle(hog) == "held"
    grabbed = hog.held_gib

    def squeezed() -> float:  # something else on the box took 6 MiB
        return 0.014 - hog.held_gib

    hog.read_available = squeezed  # the hog is the one that gives way
    assert _settle(hog) == "held"
    assert hog.held_gib < grabbed


def test_a_box_already_under_the_target_is_said_once_and_not_spun_on():
    hog = memhog.Hog(0.012, lambda: 0.004, tolerance_gib=0.002, floor_gib=0.001,
                     block_gib=0.001, step_cap_gib=0.004)
    assert hog.step() == ("starved", 0.004)
    assert hog.held_gib == 0.0
    stop, said = threading.Event(), []
    memhog.hold(hog, stop, report_s=60.0, poll_s=0.0,
                log=lambda line: (said.append(line), stop.set()))
    assert len(said) == 1 and "(starved)" in said[0]


def test_the_hold_loop_says_what_it_holds_and_stops_when_told():
    hog = _tiny_hog(0.02, 0.012)
    stop, said = threading.Event(), []

    def log(line: str) -> None:
        said.append(line)
        if len(said) >= 2:
            stop.set()

    memhog.hold(hog, stop, report_s=0.0, poll_s=0.0, log=log)
    assert len(said) == 2 and said[-1].startswith("holding ")
    assert "against a 0.01 GiB target" in said[-1]


def test_a_target_under_the_floor_is_refused_before_a_page_is_taken():
    with pytest.raises(ValueError, match=r"under the 8\.00 GiB floor"):
        memhog.Hog(4.0)
    with pytest.raises(SystemExit) as refused:
        memhog.main(["--target-gib", "4"])
    assert refused.value.code == 2


def test_the_hog_reads_the_same_meminfo_field_the_sampler_does(monkeypatch, tmp_path):
    fake = tmp_path / "meminfo"
    fake.write_text("MemAvailable:   2097152 kB\nSwapCached:     1048576 kB\n")
    monkeypatch.setattr(memhog, "MEMINFO_PATH", fake)
    assert memhog.mem_available_gib() == 2.0
