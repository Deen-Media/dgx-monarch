"""doctor's board-hotspot reader: the kernel's hwmon tree, not a package.

Every test that reads temperatures builds a fake sysfs tree under tmp_path or
stubs the `sensors` output, so none reads this machine's /sys and all run on
an x86 CI runner.
"""
from __future__ import annotations

import os
import subprocess

import pytest

from dgx_monarch.cli import hwmon


def _chip(root, node, name, temps):
    """One fake hwmon chip. temps: {index: (millicelsius_text, label|None)}."""
    chip_dir = root / node
    chip_dir.mkdir(parents=True, exist_ok=True)
    (chip_dir / "name").write_text(f"{name}\n")
    for index, (milli, label) in temps.items():
        (chip_dir / f"temp{index}_input").write_text(f"{milli}\n")
        if label is not None:
            (chip_dir / f"temp{index}_label").write_text(f"{label}\n")
    return chip_dir


def _read(root):
    """The readings only; the tests about unreadable inputs read the whole scan."""
    return hwmon.read_hwmon_temps(str(root)).readings


def _names(readings):
    return [reading.name for reading in readings]


def test_reader_keeps_every_unlabeled_sensor_and_names_it_from_its_own_file(tmp_path):
    _chip(tmp_path, "hwmon0", "acpitz",
          {index: (f"{40000 + index * 1000}", None) for index in range(1, 8)})
    readings = _read(tmp_path)
    assert len(readings) == 7
    by_name = {reading.name: reading.celsius for reading in readings}
    assert "hwmon0/acpitz/temp6" in by_name
    assert by_name["hwmon0/acpitz/temp6"] == pytest.approx(46.0)


def test_reader_uses_the_sibling_label_when_the_chip_publishes_one(tmp_path):
    _chip(tmp_path, "hwmon1", "nvme",
          {1: ("43850", "Composite"), 2: ("43850", "Sensor 1")})
    readings = _read(tmp_path)
    assert _names(readings) == ["hwmon1/nvme/Composite", "hwmon1/nvme/Sensor 1"]
    assert readings[0].celsius == pytest.approx(43.85)


def test_reader_keeps_four_identically_named_chips_apart(tmp_path):
    for node in ("hwmon2", "hwmon3", "hwmon4"):
        _chip(tmp_path, node, "mlx5", {1: ("51000", "asic")})
    _chip(tmp_path, "hwmon5", "mlx5", {1: ("51000", "asic"), 2: ("40000", "Module1")})
    readings = _read(tmp_path)
    assert len(readings) == 5
    assert len(set(_names(readings))) == 5
    assert "hwmon2/mlx5/asic" in _names(readings)
    assert "hwmon5/mlx5/Module1" in _names(readings)


def test_reader_orders_tempn_and_hwmonn_by_number_not_by_text(tmp_path):
    _chip(tmp_path, "hwmon9", "acpitz",
          {1: ("40000", None), 9: ("41000", None), 10: ("42000", None)})
    _chip(tmp_path, "hwmon10", "nvme", {1: ("41000", "Composite")})
    assert _names(_read(tmp_path)) == [
        "hwmon9/acpitz/temp1", "hwmon9/acpitz/temp9", "hwmon9/acpitz/temp10",
        "hwmon10/nvme/Composite"]


def test_reader_reads_a_negative_temperature(tmp_path):
    _chip(tmp_path, "hwmon0", "nvme", {1: ("-273150", "Sensor 1")})
    assert _read(tmp_path)[0].celsius == pytest.approx(-273.15)


def test_reader_counts_a_broken_input_instead_of_shrinking_the_total(tmp_path):
    chip_dir = _chip(tmp_path, "hwmon0", "acpitz", {3: ("47000", None)})
    (chip_dir / "temp1_input").write_text("")
    (chip_dir / "temp2_input").write_text("not-a-number\n")
    (chip_dir / "temp3_max").write_text("105000\n")
    (chip_dir / "fan1_input").write_text("2400\n")
    scan = hwmon.read_hwmon_temps(str(tmp_path))
    assert _names(scan.readings) == ["hwmon0/acpitz/temp3"]
    assert scan.readings[0].celsius == pytest.approx(47.0)
    assert scan.unreadable == 2


def test_reader_counts_a_chip_directory_it_could_not_list(tmp_path, monkeypatch):
    _chip(tmp_path, "hwmon0", "acpitz", {1: ("48000", None)})
    locked = _chip(tmp_path, "hwmon1", "nvme", {1: ("99000", "Composite")})
    real_listdir = os.listdir

    def guarded(path):
        if str(path) == str(locked):
            raise PermissionError(13, "Permission denied")
        return real_listdir(path)

    monkeypatch.setattr(os, "listdir", guarded)
    scan = hwmon.read_hwmon_temps(str(tmp_path))
    assert _names(scan.readings) == ["hwmon0/acpitz/temp1"]
    assert scan.unreadable == 1


def test_reader_returns_nothing_when_the_tree_is_absent_or_empty(tmp_path):
    assert hwmon.read_hwmon_temps(str(tmp_path / "nope")) == hwmon.TempScan([], 0)
    assert hwmon.read_hwmon_temps(str(tmp_path)) == hwmon.TempScan([], 0)


def test_reader_ignores_a_non_hwmon_entry_in_the_class_directory(tmp_path):
    _chip(tmp_path, "notahwmon", "impostor", {1: ("99000", None)})
    assert _read(tmp_path) == []


def test_reader_never_shells_out(tmp_path, monkeypatch):
    def raiser(*_args, **_kwargs):
        raise AssertionError("the sysfs reader must not spawn a subprocess")

    monkeypatch.setattr(subprocess, "run", raiser)
    _chip(tmp_path, "hwmon0", "acpitz", {1: ("48800", None)})
    assert _names(_read(tmp_path)) == ["hwmon0/acpitz/temp1"]


_SENSORS_SAMPLE = """nvme-pci-0100
Adapter: PCI adapter
Composite:
  temp1_input: 45.850
  temp1_max: 84.850
  temp1_min: -273.150
  temp1_crit: 89.850
"""


def test_sensors_fallback_parses_a_label_that_does_not_start_with_temp(monkeypatch):
    monkeypatch.setattr(hwmon, "_sensors_output", lambda: _SENSORS_SAMPLE)
    readings = hwmon.read_sensors_temps()
    assert _names(readings) == ["nvme-pci-0100/Composite"]
    assert readings[0].celsius == pytest.approx(45.85)


def test_sensors_fallback_keeps_a_negative_value_and_ignores_non_input_fields(
        monkeypatch):
    monkeypatch.setattr(hwmon, "_sensors_output", lambda: _SENSORS_SAMPLE + (
        "cold-isa-0000\n"
        "Adapter: ISA adapter\n"
        "outside:\n"
        "  temp2_input: -5.000\n"
        "  temp2_crit: 100.000\n"
    ))
    readings = hwmon.read_sensors_temps()
    assert _names(readings) == ["nvme-pci-0100/Composite", "cold-isa-0000/outside"]
    assert readings[1].celsius == pytest.approx(-5.0)


def test_verdict_uses_sensors_only_when_hwmon_answered_nothing(tmp_path, monkeypatch):
    calls = []

    def recorder():
        calls.append(1)
        return _SENSORS_SAMPLE

    monkeypatch.setattr(hwmon, "_sensors_output", recorder)
    _chip(tmp_path, "hwmon0", "acpitz", {1: ("48800", None)})
    hazard, detail = hwmon.board_hotspot_verdict(str(tmp_path))
    assert calls == []
    assert "sysfs hwmon" in detail and hazard is False

    empty = tmp_path / "empty"
    empty.mkdir()
    hazard, detail = hwmon.board_hotspot_verdict(str(empty))
    assert calls == [1]
    assert "`sensors`" in detail and hazard is False


def test_verdict_is_ok_and_names_the_hottest_sensors_under_the_limit(tmp_path):
    _chip(tmp_path, "hwmon0", "acpitz",
          {1: ("40000", None), 2: ("50000", None), 3: ("60000", None)})
    hazard, detail = hwmon.board_hotspot_verdict(str(tmp_path))
    assert hazard is False
    assert detail.startswith("3 sensors via sysfs hwmon")
    assert "hottest hwmon0/acpitz/temp3 60.0 C" in detail
    assert "hazard 90 C" in detail


def test_verdict_warns_at_the_threshold_not_only_above_it(tmp_path):
    _chip(tmp_path, "hwmon0", "acpitz", {6: ("90000", None)})
    hazard, detail = hwmon.board_hotspot_verdict(str(tmp_path))
    assert hazard is True
    assert "hwmon0/acpitz/temp6 90.0 C" in detail
    assert "1 of 1 sensor at or above 90 C" in detail


def test_verdict_names_every_hot_sensor_hottest_first(tmp_path):
    _chip(tmp_path, "hwmon0", "acpitz",
          {1: ("95000", None), 2: ("92000", None), 3: ("91000", None),
           4: ("45000", None)})
    hazard, detail = hwmon.board_hotspot_verdict(str(tmp_path))
    assert hazard is True
    assert "3 of 4 sensors at or above 90 C" in detail
    order = [detail.index(f"hwmon0/acpitz/temp{index}") for index in (1, 2, 3)]
    assert order == sorted(order)


def test_verdict_carries_the_unreadable_count_into_both_wordings(tmp_path):
    chip_dir = _chip(tmp_path, "hwmon0", "acpitz",
                     {1: ("48000", None), 2: ("50000", None)})
    (chip_dir / "temp3_input").write_text("")
    hazard, detail = hwmon.board_hotspot_verdict(str(tmp_path))
    assert hazard is False
    assert detail.startswith("2 sensors via sysfs hwmon (1 unreadable), hottest ")
    (chip_dir / "temp4_input").write_text("94000\n")
    hazard, detail = hwmon.board_hotspot_verdict(str(tmp_path))
    assert hazard is True
    assert "1 of 3 sensors at or above 90 C (1 unreadable): " in detail


def test_verdict_says_nothing_about_drops_when_every_sensor_read(tmp_path):
    _chip(tmp_path, "hwmon0", "acpitz", {1: ("48000", None)})
    _, detail = hwmon.board_hotspot_verdict(str(tmp_path))
    assert "unreadable" not in detail
    assert detail.startswith("1 sensor via sysfs hwmon, hottest ")


def test_verdict_counts_the_unreadable_when_the_box_shows_no_temperature(
        tmp_path, monkeypatch):
    chip_dir = _chip(tmp_path, "hwmon0", "acpitz", {})
    (chip_dir / "temp1_input").write_text("")
    monkeypatch.setattr(hwmon, "_sensors_output", lambda: "")
    hazard, detail = hwmon.board_hotspot_verdict(str(tmp_path))
    assert hazard is False
    assert "nothing readable under" in detail and "(1 unreadable)" in detail


def test_verdict_keeps_a_fractional_hazard_threshold(tmp_path):
    _chip(tmp_path, "hwmon0", "acpitz", {1: ("45000", None)})
    hazard, detail = hwmon.board_hotspot_verdict(str(tmp_path), 44.5)
    assert hazard is True
    assert "at or above 44.5 C" in detail
    hazard, detail = hwmon.board_hotspot_verdict(str(tmp_path), 45.5)
    assert hazard is False
    assert "hazard 45.5 C" in detail


def test_verdict_for_a_box_with_no_sensors_is_information_not_a_fault(
        tmp_path, monkeypatch):
    monkeypatch.setattr(hwmon, "_sensors_output", lambda: "")
    hazard, detail = hwmon.board_hotspot_verdict(str(tmp_path))
    assert hazard is False
    assert "no temperature sensors on this box" in detail
    assert "probe unavailable" not in detail
    assert "sensors-detect" not in detail


def test_hazard_threshold_is_the_documented_90_c():
    assert hwmon.HOTSPOT_C == 90.0
