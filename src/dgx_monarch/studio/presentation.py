"""Versioned display semantics; these thresholds never authorize execution."""
from __future__ import annotations

import math
from typing import Any, TypeGuard

PRESENTATION_VERSION = 1
MEMORY_COLORS = {
    "gpu": "#22d3ee", "slab": "#34d399", "pool": "#fb923c",
    "anonymous": "#c084fc", "other": "#64748b", "cache": "#facc15",
}
MEMORY_ANSI256 = dict(zip(MEMORY_COLORS, (45, 79, 208, 141, 244, 220), strict=True))
# Six different standard slots, so a 16-color terminal keeps every segment
# apart unless its own palette paints two slots alike. Pool takes the slot Rich
# picks for its orange; cache keeps yellow.
MEMORY_ANSI = dict(zip(MEMORY_COLORS, ("cyan", "green", "bright_red", "magenta", "white", "yellow"), strict=True))
STATUS_COLORS = {
    **dict.fromkeys(("ready", "idle", "rendering", "busy"), "#4ade80"),
    **dict.fromkeys(("degraded", "legacy-active"), "#facc15"),
    **dict.fromkeys(("blocked", "unknown"), "#f87171"),
}
# The status token each Studio job state displays with. A job on its way to the
# driver or running there reads as rendering; a cancel, asked for or done,
# reads as degraded; a failure, or an outcome the companion cannot confirm,
# reads as blocked.
JOB_STATUS = {
    **dict.fromkeys(("submitting", "accepted", "queued", "running"), "rendering"),
    **dict.fromkeys(("cancel_requested", "cancelled"), "degraded"),
    "succeeded": "ready",
    **dict.fromkeys(("failed", "unknown"), "blocked"),
}
UNITS: dict[str, dict[str, Any]] = {
    "GiB": {"precision": 1, "divisor": 1073741824, "source_unit": "bytes"},
    "GB/s": {"precision": 2, "divisor": 1000000000, "source_unit": "bytes/s"},
    "%": {"precision": 0, "divisor": 1, "source_unit": "%"},
    "W": {"precision": 1, "divisor": 1, "source_unit": "W"},
    "°C": {"precision": 0, "divisor": 1, "source_unit": "°C"},
    "s": {"precision": 2, "divisor": 1, "source_unit": "s"},
    "MHz": {"precision": 0, "divisor": 1, "source_unit": "MHz"},
    "GHz": {"precision": 2, "divisor": 1000, "source_unit": "MHz"},
    "count": {"precision": 0, "divisor": 1, "source_unit": "count"},
}
# Elapsed and total job time as minutes and seconds, a separate rule from the
# "s" unit above: whole seconds, rounded down, and minutes with no cap, so 61
# minutes and 1 second reads 61:01. The largest value is the largest integer
# both Python and JavaScript hold exactly; beyond it, as for a missing,
# negative or non-finite value, the time is unknown.
DURATION_CLOCK: dict[str, Any] = {"format": "m:ss", "source_unit": "s", "rounding": "floor",
                                  "minutes": "uncapped", "maximum_s": 2**53 - 1, "missing": "unknown"}
AVAILABLE_MEMORY_THRESHOLD_GIB = 20.0
# A Worker reports slab_gib only while a slab holds its model, so a missing
# value is a known zero unless the report could be hiding one.
SLAB_RULE = {
    "zero": ["no Worker reports the host (a driver-only box, or dgxm top reading local stats while the driver is down)",
             "the Worker reports weight_residency other than slab", "the Worker reports no resident model"],
    "unknown": ["the Worker sent memory_error", "the Worker sent no memory report",
                "the Worker sent a slab_gib that is not a finite number",
                "the Worker reports slab residency, or a resident model with no weight_residency, and no slab_gib"],
}


def _finite(value: object) -> TypeGuard[int | float]:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def format_metric(value: object, unit: str, *, source_units: bool = False) -> str:
    """Format a display-unit value, optionally converting its documented source unit."""
    spec = UNITS[unit]
    if not _finite(value):
        return "unknown"
    number = float(value) / (float(spec["divisor"]) if source_units else 1)
    suffix = "" if unit == "count" else f" {unit}"
    return f"{number:.{spec['precision']}f}{suffix}"


def format_duration_clock(value: object) -> str:
    """Format a duration in seconds under DURATION_CLOCK, such as 1:13 or 61:01."""
    if not _finite(value) or value < 0 or value > DURATION_CLOCK["maximum_s"]:
        return str(DURATION_CLOCK["missing"])
    whole = int(value)
    return f"{whole // 60}:{whole % 60:02d}"


def available_memory_status(value_gib: object) -> str:
    """Preserve the existing strict >20 GiB display boundary, independently of guards."""
    if not _finite(value_gib):
        return "unknown"
    return "ready" if float(value_gib) > AVAILABLE_MEMORY_THRESHOLD_GIB else "degraded"


def slab_reading(meta: object) -> float | None:
    """Slab GiB a host's memory bar may draw under SLAB_RULE, or None when unknown.

    ``meta`` is the Worker report for the host, or None when no Worker reports it.
    """
    if meta is None:
        return 0.0
    if not isinstance(meta, dict) or meta.get("memory_error") is not None:
        return None
    memory = meta.get("memory")
    if not isinstance(memory, dict):
        return None
    if "slab_gib" in memory:
        value = memory["slab_gib"]
        return max(0.0, float(value)) if _finite(value) else None
    residency = memory.get("weight_residency")
    if residency is None:
        # Every Worker that holds a model also reports its weight_residency.
        return None if "patcher_class" in memory or "model" in memory else 0.0
    return 0.0 if isinstance(residency, str) and residency != "slab" else None


def freshness(sample_time: object, now: float, *, interval_s: float = 1.0) -> str:
    """Use source timestamps; missing/future samples cannot be presented as live."""
    if not _finite(sample_time) or not _finite(now) or not _finite(interval_s) or interval_s <= 0:
        return "unknown"
    age = now - float(sample_time)
    if age < 0:
        return "unknown"
    return "fresh" if age <= max(3 * interval_s, 3.0) else "stale"


def presentation_contract() -> dict[str, Any]:
    """Serializable contract shared by Python and generated browser assets."""
    metric_fields = [
        ("host.memory.total", "Total memory", "mem_gib.MemTotal", "GiB"),
        ("host.memory.available", "Available memory", "mem_gib.MemAvailable", "GiB"),
        ("host.memory.gpu", "GPU memory", "gpu_proc_gib", "GiB"),
        ("host.memory.slab", "Slab", "worker.memory.slab_gib", "GiB"),
        ("host.memory.pool", "Retained pool", "pool_gib", "GiB"),
        ("host.memory.anonymous", "Anonymous", "mem_gib.AnonPages", "GiB"),
        ("host.memory.cache", "Reclaimable cache", "derived: min(MemAvailable, max(0, MemAvailable - MemFree))", "GiB"),
        ("host.gpu.utilization", "GPU utilization", "gpu.util_pct", "%"),
        ("host.gpu.clock", "GPU clock", "gpu.clock_mhz", "MHz"),
        ("host.gpu.power", "GPU power", "gpu.power_w", "W"),
        ("host.gpu.temperature", "GPU temperature", "gpu.temp_c", "°C"),
        ("host.rail.rx", "Rail receive", "rails.*.rx_bytes", "GB/s"),
        ("host.rail.tx", "Rail transmit", "rails.*.tx_bytes", "GB/s"),
        ("render.step", "Step", "render.step", "count"),
        ("render.steps", "Total steps", "render.steps", "count"),
        ("render.step_duration", "Step duration", "render.sec_per_step", "s"),
        ("render.last.duration", "Last render duration", "render.last.wall_s", "s"),
    ]
    return {
        "schema_version": PRESENTATION_VERSION,
        "memory": {
            key: {"hex": color, "ansi256": MEMORY_ANSI256[key], "ansi": MEMORY_ANSI[key], "label": {
                "gpu": "GPU", "slab": "Slab", "pool": "Retained pool", "anonymous": "Anonymous",
                "other": "Other", "cache": "Reclaimable cache",
            }[key]} for key, color in MEMORY_COLORS.items()
        },
        "free_memory_color": None,
        "memory_order": list(MEMORY_COLORS),
        "memory_overlap": {
            "stack_raw_counters": False,
            "reason": "Retained pools overlap AnonPages and slabs overlap Shmem; GPU accounting shares the physical UMA total. Apply the bounded partition before stacking.",
            "partition_source": "web/js/dgx_monarch_segments.js",
        },
        "status": {
            key: {"hex": color, "ansi": {"#4ade80": "green", "#facc15": "yellow", "#f87171": "red"}[color],
                  "icon": {"#4ade80": "✓", "#facc15": "!", "#f87171": "?"}[color], "label": key}
            for key, color in STATUS_COLORS.items()
        },
        "job_states": dict(JOB_STATUS),
        "theme": {"background": "#111318", "text": "#f3f5f7", "brand": "#87f900", "focus": "#22d3ee",
                  "selection": "#334155", "warning": "#facc15"},
        "units": UNITS,
        "duration_clock": dict(DURATION_CLOCK),
        "missing": "unknown",
        "thresholds": {"available_memory_gib": {"operator": ">", "value": AVAILABLE_MEMORY_THRESHOLD_GIB,
                        "above": "ready", "otherwise": "degraded", "authority": "presentation_only"}},
        "freshness": {"default_interval_s": 1, "stale_after_intervals": 3, "minimum_stale_s": 3,
                      "retain_last_sample": True, "future_timestamp": "unknown",
                      "source_sampling": "host route cached 1 s; retained pool cached 10 s; preserve source timestamps"},
        "metrics": [{"id": identity, "name": name, "source": source, "unit": unit,
                     "aggregation": "per_host_per_rail_rate" if ".rail." in identity else "none",
                     "missing": "unknown", "threshold": "available_memory_gib" if identity == "host.memory.available" else None,
                     "safety_authority": "backend_policy",
                     **({"missing": "slab_rule", "missing_rule": SLAB_RULE} if identity == "host.memory.slab" else {})}
                    for identity, name, source, unit in metric_fields],
        "render_fields": ["topology", "auto_row", "megapixels", "quant", "attention", "sec_per_step", "last_step_t"],
        "rail_rate": {"directions": ["rx", "tx"], "identity": ["host", "rail"],
                      "counter_reset": "unknown", "nonpositive_interval": "unknown", "divisor": 1000000000},
    }
