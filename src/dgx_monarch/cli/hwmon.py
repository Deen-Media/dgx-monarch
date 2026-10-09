"""Read board temperatures from sysfs, with optional lm-sensors fallback.

Doctor's ``board hotspots`` row checks sensors that can exceed 90 C while the
Spark's CPU-driven fan curve stays low (docs/TROUBLESHOOTING.md). Read
``/sys/class/hwmon`` first so the check works without extra packages. Use
``sensors`` only when sysfs yields no temperatures.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

HWMON_ROOT = "/sys/class/hwmon"
HOTSPOT_C = 90.0
_SENSORS_VALUE = re.compile(r"^\s+temp(\d+)_input:\s*(-?\d+(?:\.\d+)?)\s*$")


@dataclass(frozen=True)
class TempReading:
    """One temperature sensor: a display name and degrees Celsius."""

    name: str
    celsius: float


@dataclass(frozen=True)
class TempScan:
    """Temperature readings and the number of unreadable entries.

    Count failed or invalid ``temp*_input`` reads and unlistable chip directories
    so a lost sensor remains visible in the report.
    """

    readings: list[TempReading]
    unreadable: int


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _numeric_key(name: str, prefix: str) -> tuple[int, str]:
    """Order tempN and hwmonN by their number, so temp10 lands after temp9."""
    digits = name[len(prefix):].split("_", 1)[0]
    return (int(digits) if digits.isdigit() else -1, name)


def _chip_temps(chip_dir: str, node: str) -> TempScan:
    """Read every temp*_input in one hwmon chip directory.

    Keep unlabeled sensors using their file path, such as ``hwmon0/acpitz/temp6``.
    The node prefix distinguishes identically named chips. Count an unlistable chip
    as one unreadable entry because its individual sensors cannot be counted.
    """
    chip = _read_text(os.path.join(chip_dir, "name")) or "unknown"
    try:
        entries = os.listdir(chip_dir)
    except OSError:
        return TempScan([], 1)
    inputs = [name for name in entries
              if name.startswith("temp") and name.endswith("_input")]
    readings: list[TempReading] = []
    unreadable = 0
    for entry in sorted(inputs, key=lambda name: _numeric_key(name, "temp")):
        raw = _read_text(os.path.join(chip_dir, entry))
        try:
            milli = float(raw)
        except ValueError:
            unreadable += 1
            continue
        stem = entry[: -len("_input")]
        label = _read_text(os.path.join(chip_dir, f"{stem}_label"))
        readings.append(TempReading(f"{node}/{chip}/{label or stem}", milli / 1000.0))
    return TempScan(readings, unreadable)


def read_hwmon_temps(root: str = HWMON_ROOT) -> TempScan:
    """Every temperature sysfs exposes, with no external binary."""
    try:
        nodes = os.listdir(root)
    except OSError:
        return TempScan([], 0)
    readings: list[TempReading] = []
    unreadable = 0
    for node in sorted(nodes, key=lambda name: _numeric_key(name, "hwmon")):
        chip_dir = os.path.join(root, node)
        if node.startswith("hwmon") and os.path.isdir(chip_dir):
            scan = _chip_temps(chip_dir, node)
            readings.extend(scan.readings)
            unreadable += scan.unreadable
    return TempScan(readings, unreadable)


def _sensors_output() -> str:
    """Return ``sensors -u`` output, or "" if lm-sensors is absent.

    The machine format avoids degree symbols and leading plus signs; LC_ALL=C
    fixes the decimal separator.
    """
    try:
        result = subprocess.run(
            ["sensors", "-u"], capture_output=True, text=True, timeout=5,
            check=False, env={**os.environ, "LC_ALL": "C"})
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout


def read_sensors_temps() -> list[TempReading]:
    """Second source, reached only when sysfs yielded nothing.

    libsensors reads the same hwmon tree, so this rarely finds a sensor sysfs
    missed; it covers a kernel or container layout the sysfs reader does not
    expect. Labels come from the feature heading, so `Composite:` and `asic:`
    are kept, not dropped.
    """
    chip = "sensors"
    feature = ""
    readings: list[TempReading] = []
    for line in _sensors_output().splitlines():
        if not line[:1].isspace():
            stripped = line.strip()
            if not stripped or stripped.startswith("Adapter:"):
                continue
            if stripped.endswith(":"):
                feature = stripped[:-1]
            else:
                chip, feature = stripped, ""
            continue
        match = _SENSORS_VALUE.match(line)
        if match:
            name = feature or f"temp{match.group(1)}"
            readings.append(TempReading(f"{chip}/{name}", float(match.group(2))))
    return readings


def _named(readings: list[TempReading], limit: int) -> str:
    hottest = sorted(readings, key=lambda reading: -reading.celsius)[:limit]
    return ", ".join(f"{r.name} {r.celsius:.1f} C" for r in hottest)


def _degrees(value: float) -> str:
    """Format temperatures without unnecessary trailing decimal zeros."""
    return f"{value:g}"


def _count(total: int) -> str:
    return "1 sensor" if total == 1 else f"{total} sensors"


def _shortfall(unreadable: int) -> str:
    return f" ({unreadable} unreadable)" if unreadable else ""


def board_hotspot_verdict(
    root: str = HWMON_ROOT, hazard_c: float = HOTSPOT_C
) -> tuple[bool, str]:
    """Return (hazard, detail) for doctor's board-hotspots row.

    Set hazard when a sensor reaches hazard_c. Missing temperatures or read errors
    appear in the detail without a warning: an empty mlx5 transceiver bay, for
    example, can fail every temperature read without a thermal fault.
    """
    scan = read_hwmon_temps(root)
    readings, unreadable = scan.readings, scan.unreadable
    source = "sysfs hwmon"
    if not readings:
        readings = read_sensors_temps()
        source = "`sensors` (no hwmon temps)"
    if not readings:
        return False, (
            f"no temperature sensors on this box: nothing readable under {root}"
            f"{_shortfall(unreadable)} and `sensors` added none. This row cannot "
            "warn here, and a board sensor can reach 90 C or more while the fan "
            "curve, which tracks CPU load, stays flat (docs/TROUBLESHOOTING.md #59)"
        )
    hot = [reading for reading in readings if reading.celsius >= hazard_c]
    if hot:
        return True, (
            f"{len(hot)} of {_count(len(readings))} at or above "
            f"{_degrees(hazard_c)} C{_shortfall(unreadable)}: {_named(hot, 4)}. "
            "The Spark fan curve tracks CPU load and can miss a board "
            "sensor; check airflow and case clearance "
            "(docs/TROUBLESHOOTING.md #59)"
        )
    return False, (
        f"{_count(len(readings))} via {source}{_shortfall(unreadable)}, hottest "
        f"{_named(readings, 3)}; hazard {_degrees(hazard_c)} C"
    )
