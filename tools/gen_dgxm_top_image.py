#!/usr/bin/env python3
"""Regenerate docs/media/dgxm-top.svg: dgxm top replaying a synthetic sample.

The sample holds made-up readings for two boxes. It is not a benchmark result
and names no real host, address, client or LoRA. Rendering needs the [tui]
extra (Textual and Rich). The README's dgxm-top.png is a raster copy of this
SVG at the same size.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TARGET = ROOT / "docs/media/dgxm-top.svg"
SIZE = (120, 46)
TITLE = "DGX Monarch | dgxm top"
START = 1_767_268_800.0  # 2026-01-01 12:00:00 UTC; the clock shows UTC.
TICKS = 64


def _host(name: str, *, avail: float, gpu: float, slab: float, rx: float, tx: float, util: int) -> dict:
    return {"source": "worker",
            "stats": {"host": name, "t": 0.0,
                      "mem_gib": {"MemTotal": 121.6, "MemAvailable": avail, "MemFree": avail - 14.0,
                                  "AnonPages": 14.0, "Shmem": slab + 0.4, "Cached": 15.0 + slab, "Buffers": 0.2},
                      "gpu_proc_gib": gpu, "pool_gib": 4.0,
                      "gpu": {"util_pct": util, "clock_mhz": 2100, "power_w": 61.5, "temp_c": 58}},
            "rails": {"mlx5_0": {"rx_gbs": rx, "tx_gbs": tx}}}


def sample_ticks() -> list[dict]:
    """A recorded-shape replay: one idle sample per second after a short render."""
    from dgx_monarch.operator_activity import telemetry_presentation
    from dgx_monarch.tui.diagnostics_view import diagnostics_view

    lifecycle = {"worker_service": {"state": "ready", "detail": "2 workers online"},
                 "attached_mesh": {"state": "ready", "detail": "2 GPUs connected"},
                 "render_session": {"state": "idle", "detail": "Ready for next render"}}
    readiness = {"schema_version": 1, "overall": "ready", "lifecycle": lifecycle,
                 "presentation": telemetry_presentation("ready", ["ready", "ready", "idle"])}
    diagnostics = diagnostics_view({"diagnostics": {"schema": 1, "available": True, "recent": [
        {"phase": "attach", "outcome": "ok", "return_mode": "none", "duration_s": 3.1},
        {"phase": "readiness", "outcome": "ok", "return_mode": "none", "duration_s": 0.4},
        {"phase": "collect", "outcome": "ok", "return_mode": "message", "duration_s": 0.02},
    ]}})
    last = {"model": "flux1-dev", "wall_s": 35.2, "steps": 28, "ceremony": False, "topology": "uly2",
            "auto_row": 11, "sec_per_step": 1.18}
    workers = {
        "node-0001": {"vram": None, "memory_error": None, "nccl": True, "models": None, "memory": {
            "weight_residency": "slab", "slab_gib": 22.0, "lora_mode": "low_rss",
            "unbake_file_backed_gib": 1.2, "swap_verify_failures": 0}},
        "node-0002": {"vram": None, "memory_error": None, "nccl": True, "models": None, "memory": {
            "weight_residency": "cudaMalloc", "lora_mode": "hot_swap", "swap_verify_failures": 0}},
    }
    first, ended = 20, 55  # the render's first and last sample
    last = {**last, "ended_at": START + ended + 0.2}
    events = {2: [{"t": START + 2, "kind": "notice", "phase": "deferred", "note": "NCCL bring-up on first render"}],
              6: [{"t": START + 6, "kind": "notice", "phase": "deferred", "note": "cold load on first render"},
                  {"t": START + 6.5, "kind": "load", "state": "resident"}],
              18: [{"t": START + 18, "kind": "swap", "loras": 1}],
              ended + 1: [{"t": START + ended + 1, "kind": "gate", "verdict": "PASS"}],
              TICKS - 1: [{"t": START + TICKS - 1, "kind": "verify", "fails": 0}]}
    ticks = []
    for index in range(TICKS):
        # Sequence-parallel traffic pulses once per denoise step while the
        # render runs; the rails sit near idle before and after it.
        busy = first <= index <= ended
        wave = 13.0 + 9.0 * math.sin(index * 1.9) if busy else 0.05
        hosts = {"node-0001": _host("node-0001", avail=59.6, gpu=4.0, slab=22.0, rx=wave, tx=wave * 0.96,
                                     util=96 if busy else 0),
                 "node-0002": _host("node-0002", avail=61.6, gpu=24.0, slab=0.0, rx=wave * 0.97, tx=wave,
                                     util=95 if busy else 0)}
        render = ({"active": True, "started": START + first, "model": last["model"], "steps": last["steps"],
                   "step": min(last["steps"], (index - first) * 28 // (ended - first) + 1)} if busy else
                  {"active": False, "last": last} if index > ended else {"active": False})
        ticks.append({"t": START + index, "driver_up": True, "loops": [], "comfy": "sample",
                      "readiness": readiness, "diagnostics": diagnostics, "render": render,
                      "ledger": {"combinations": {"PASS": 3}}, "hosts": hosts, "workers": workers,
                      "events": events.get(index, [])})
    return ticks


async def _render(recording: Path) -> str:
    from dgx_monarch.tui.app import DgxmTopApp

    app = DgxmTopApp(replay=str(recording), theme="spark")
    async with app.run_test(size=SIZE) as pilot:
        app.motion = False
        await pilot.pause(1.5)  # the header wings finish appearing within a second
        app.redraw()
        await pilot.pause()
        svg = app.export_screenshot(title=TITLE)
    # Rich stamps a credit comment into every export; the image needs none.
    return "\n".join(line for line in svg.splitlines() if not line.strip().startswith("<!--")) + "\n"


def render() -> str:
    os.environ["TZ"] = "UTC"
    time.tzset()
    with tempfile.TemporaryDirectory() as directory:
        recording = Path(directory) / "sample.jsonl"
        recording.write_text("".join(json.dumps(tick) + "\n" for tick in sample_ticks()))
        return asyncio.run(_render(recording))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=TARGET)
    args = parser.parse_args()
    args.output.write_text(render(), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
