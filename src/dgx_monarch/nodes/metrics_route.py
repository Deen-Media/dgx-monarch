"""Bounded Prometheus text projection for the telemetry route."""
from __future__ import annotations

import math
from collections.abc import Callable
from itertools import islice
from typing import cast

from ..runtime_diagnostics import BUCKET_BOUNDS_S, MAX_SERIES, valid_labels

_MAX_METRIC_WORKERS = 256
_MAX_METRIC_RAILS_PER_WORKER = 64


def _count(value: object) -> int | None:
    return (value if isinstance(value, int) and not isinstance(value, bool)
            and 0 <= value <= (1 << 53) - 1 else None)


def _phase_metrics(metric: Callable, block: object, *, scope: str,
                   worker_index: int | None = None, stale: bool = False) -> None:
    """Project a fixed schema without trusting remote labels or bucket names."""
    labels: dict = {"scope": scope}
    if worker_index is not None:
        labels["worker_index"] = worker_index
    if not isinstance(block, dict):
        # Source without the block (a partial deploy) reads 0, never absent.
        metric("diagnostics_available", False, **labels)
        return
    available = block.get("available") is True and not stale
    metric("diagnostics_available", available, **labels)
    if not available or block.get("schema") != 1:
        return
    rows = block.get("phases")
    if not isinstance(rows, list):
        return
    seen: set[tuple[str, str, str]] = set()
    for row in rows[:MAX_SERIES]:
        if not isinstance(row, dict):
            continue
        key = (row.get("phase"), row.get("outcome"), row.get("return_mode"))
        if not valid_labels(*key) or key in seen:
            continue
        seen.add(cast(tuple[str, str, str], key))
        count = _count(row.get("count"))
        buckets = row.get("buckets")
        if (count is None or not isinstance(buckets, list)
                or len(buckets) != len(BUCKET_BOUNDS_S) + 1
                or any(_count(value) is None for value in buckets)
                or sum(buckets) != count):
            continue
        dimensions = {**labels, "phase": key[0], "outcome": key[1], "return_mode": key[2]}
        metric("phase_duration_seconds_count", count, **dimensions)
        metric("phase_duration_seconds_sum", row.get("sum_s"), **dimensions)
        cumulative = 0
        for bound, amount in zip((*BUCKET_BOUNDS_S, "+Inf"), buckets, strict=True):
            cumulative += amount
            metric("phase_duration_seconds_bucket", cumulative, le=bound, **dimensions)


def _prometheus_label(value: object) -> str:
    """Prometheus text-format escaping for an untrusted label value."""
    return (str(value).replace("\\", "\\\\").replace("\n", "\\n")
            .replace("\r", "\\n").replace('"', '\\"'))


def _return_metrics(metric: Callable, block: object, *, scope: str,
                    worker_index: int | None = None, stale: bool = False) -> None:
    labels: dict = {"scope": scope}
    if worker_index is not None:
        labels["worker_index"] = worker_index
    if not isinstance(block, dict):
        metric("return_ownership_available", False, **labels)
        return
    available = block.get("schema") == 1 and block.get("available") is True and not stale
    metric("return_ownership_available", available, **labels)
    if not available:
        return
    for field, name in (("live_handoffs", "live_handoffs"), ("tombstones", "tombstones"),
                        ("native_poison_owners", "poison_owners"),
                        ("pending_read_jobs", "pending_read_jobs")):
        metric(f"return_{name}", _count(block.get(field)), **labels)
    valid = {("rdma", "native"), ("message", "configured_message"),
             ("message", "registry_capacity"), ("message", "nested_tensor"),
             ("message", "below_threshold"), ("message", "pack_fallback")}
    rows = block.get("return_counts")
    if not isinstance(rows, list):
        return
    seen = set()
    for row in rows[:len(valid)]:
        if not isinstance(row, dict):
            continue
        kind, reason = row.get("kind"), row.get("reason")
        if not isinstance(kind, str) or not isinstance(reason, str):
            continue
        key = (kind, reason)
        if key not in valid or key in seen:
            continue
        seen.add(key)
        metric("return_descriptor_total", _count(row.get("count")), kind=kind, reason=reason, **labels)


def _metrics_text(telemetry: dict) -> str:
    lines = []

    def metric(name, value, **labels):
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, (int, float)):
            return
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            return
        if not finite:
            return
        rendered = ",".join(
            f'{key}="{_prometheus_label(item)}"'
            for key, item in labels.items()
        )
        lines.append(
            f"dgxm_{name}{{{rendered}}} {value}"
            if rendered else f"dgxm_{name} {value}"
        )

    render_value = telemetry.get("render", {})
    render = render_value if isinstance(render_value, dict) else {}
    metric("render_active", int(bool(render.get("active"))))
    metric("render_step", render.get("step"))
    stale = bool(telemetry.get("telemetry_error"))
    metric("telemetry_stale", stale)
    mesh = telemetry.get("mesh")
    if isinstance(mesh, dict) and not stale:
        state = mesh.get("state")
        if state in ("ok", "busy", "dirty", "poisoned", "idle", "unknown"):
            metric("mesh_state", 1, state=state)
            if state != "unknown":
                for field in ("active_leases", "abandoned_samples"):
                    metric(f"mesh_{field}", _count(mesh.get(field)))
                for field in ("pending", "poisoned"):
                    if isinstance(mesh.get(field), bool):
                        metric(f"mesh_{field}", mesh[field])
    _phase_metrics(metric, telemetry.get("diagnostics"), scope="driver", stale=stale)
    _return_metrics(metric, telemetry.get("return_ownership"), scope="driver", stale=stale)
    workers = telemetry.get("workers", [])
    if not isinstance(workers, list):
        workers = []
    poll = telemetry.get("status_poll")
    poll_stale = "status_poll" in telemetry and (
        not isinstance(poll, dict) or poll.get("fresh") is not True)
    for ordinal, worker in enumerate(workers[:_MAX_METRIC_WORKERS]):
        if not isinstance(worker, dict):
            continue
        worker_poll = worker.get("status_poll")
        worker_stale = (stale or poll_stale or bool(worker.get("telemetry_error"))
                        or "status_error" in worker or ("status_poll" in worker and (
                            not isinstance(worker_poll, dict) or worker_poll.get("fresh") is not True)))
        _phase_metrics(metric, worker.get("diagnostics"), scope="worker", worker_index=ordinal,
                       stale=worker_stale)
        _return_metrics(metric, worker.get("return_ownership"), scope="worker", worker_index=ordinal,
                        stale=worker_stale)
        host_stats = worker.get("host")
        if not isinstance(host_stats, dict):
            continue
        host = host_stats.get("host")
        gpu_value = host_stats.get("gpu")
        gpu = gpu_value if isinstance(gpu_value, dict) else {}
        mem_value = host_stats.get("mem_gib")
        mem = mem_value if isinstance(mem_value, dict) else {}
        metric("gpu_util_pct", gpu.get("util_pct"), host=host)
        metric("gpu_power_w", gpu.get("power_w"), host=host)
        metric("mem_available_gib", mem.get("MemAvailable"), host=host)
        rails_value = host_stats.get("rails")
        rails = rails_value if isinstance(rails_value, dict) else {}
        for device, counters in islice(
            rails.items(), _MAX_METRIC_RAILS_PER_WORKER,
        ):
            if not isinstance(counters, dict):
                continue
            metric("rail_rx_bytes_total", counters.get("rx_bytes"),
                   host=host, rail=device)
            metric("rail_tx_bytes_total", counters.get("tx_bytes"),
                   host=host, rail=device)
        memory_value = worker.get("memory")
        memory = memory_value if isinstance(memory_value, dict) else {}
        metric("swap_verify_failures", memory.get("swap_verify_failures"), host=host)
        metric("unbake_file_backed_gib", memory.get("unbake_file_backed_gib"), host=host)
    return "\n".join(lines) + "\n"
