"""Bounded Prometheus text projection for the telemetry route."""
from __future__ import annotations

import math
from itertools import islice

_MAX_METRIC_WORKERS = 256
_MAX_METRIC_RAILS_PER_WORKER = 64


def _prometheus_label(value: object) -> str:
    """Prometheus text-format escaping for an untrusted label value."""
    return (str(value).replace("\\", "\\\\").replace("\n", "\\n")
            .replace("\r", "\\n").replace('"', '\\"'))


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
    workers = telemetry.get("workers", [])
    if not isinstance(workers, list):
        workers = []
    for worker in workers[:_MAX_METRIC_WORKERS]:
        if not isinstance(worker, dict):
            continue
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
