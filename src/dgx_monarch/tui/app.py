"""Read-only Textual dashboard for ``dgxm top``.

Poll the driver's /dgxm/telemetry route over HTTP, or read local host statistics
when the driver is unavailable. Never create a Monarch client: a second client
can disrupt the attached mesh (docs/DESIGN.md section 5.8).

Space pauses and arrow keys navigate the samples retained by RingStore. The
rail graph's event row marks swaps, gates, loads, quarantine and audit failures.
``--record`` appends polled samples, ``s`` saves the retained samples, and
``--replay`` reads either format.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, cast

from rich.text import Text

from .braille import EVENT_STYLES, bar, braille_rows, segmented_bar
from .brand import header_layout, measured_activity, supports_unicode
from .data import Poller, RingStore, event_time
from .readiness_view import readiness_text
from .run_story import RunStory, StoryTrack, story_spans
from .view_helpers import (
    _append_record,
    _loop_indicator,
    _loop_label,
    _marker_lane,
    _redraw_panels,
    _series,
    gate_chip_style,
)

# The six memory-bar segments (model is the gpu segment) take their colors from
# the theme, never one of the sixteen standard ANSI names: a terminal can remap
# those slots until two look alike. "spark" is the browser sidebar's six
# swatches in hex, so a box reads the same in both places. "mono" alternates
# light and dark greys so neighbors stand apart. Status keys are exempt: "spark"
# keeps ANSI green, yellow and red, so the terminal's own red still means danger.
THEMES = {
    "spark": {"accent": "bright_cyan", "model": "#22d3ee", "slab": "#34d399",
              "pool": "#fb923c", "anon": "#c084fc", "other": "#64748b",
              "cache": "#facc15", "good": "green", "warn": "yellow", "bad": "red",
              "bolt": "bright_yellow"},
    "ember": {"accent": "orange1", "model": "dark_orange", "slab": "light_salmon1",
              "pool": "hot_pink3", "anon": "red3", "other": "grey50",
              "cache": "gold1", "good": "green3", "warn": "orange1", "bad": "red1",
              "bolt": "orange1"},
    "mono": {"accent": "white", "model": "grey100", "slab": "grey54",
             "pool": "grey85", "anon": "grey42", "other": "grey70",
             "cache": "grey30", "good": "white", "warn": "grey85", "bad": "white",
             "bolt": "white"},
}

class DgxmTopApp:
    """Built when called, so tests and CI can import this module without textual."""

    def __new__(cls, *args, **kwargs):
        from textual.app import App
        from textual.binding import Binding
        from textual.widgets import Static

        poller_args = kwargs

        class _Panel(Static):
            def on_mount(self):
                self.top = self.app

        class _App(App):
            TITLE = "DGX Monarch | dgxm top"
            CSS = """
            Screen { layout: vertical; }
            .panel { border: round $primary 30%; padding: 0 1; height: auto; }
            #ticker { height: 6; }
            """
            BINDINGS = [
                Binding("q", "quit", "quit"),
                Binding("space", "pause", "pause/scrub"),
                Binding("left", "back", "◀ scrub"),
                Binding("right", "fwd", "scrub ▶"),
                Binding("a", "anim", "animations"),
                Binding("t", "theme", "theme"),
                Binding("s", "snapshot", "save recording"),
                Binding("l", "ledger", "ledger"),
                Binding("w", "workers", "workers"),
            ]

            def __init__(self):
                super().__init__()
                self.replay = poller_args.get("replay")
                self.record = poller_args.get("record")
                self.interval = float(poller_args.get("interval", 1.0))
                self.ring = (RingStore.load(self.replay) if self.replay
                             else RingStore(interval=self.interval))
                self.poller = None if self.replay else Poller(
                    driver=poller_args.get("driver", "127.0.0.1:8191"),
                    config_path=poller_args.get("config_path"))
                self.paused = bool(self.replay)
                # Update run phases live or reconstruct them through the paused
                # cursor. Replay starts paused.
                self.story = StoryTrack()
                self.cursor = -1          # ring index (negative, -1 == newest)
                self.animate = True
                self.theme_name = poller_args.get("theme", "spark")
                self.frame = 0
                self._polling = False
                self._view_generation = 0
                self._last_poll_error: str | None = None

            @property
            def pal(self):
                return THEMES[self.theme_name]

            def current(self) -> dict | None:
                return self.ring.at(self.cursor if self.paused else -1)

            def view(self) -> RunStory:
                """Return run phases through the paused cursor, or the live view when running."""
                return self.story.view(list(self.ring.ticks),
                                       self.cursor if self.paused else None)

            def compose(self):
                yield _Panel(id="header", classes="panel")
                yield _Panel(id="hosts", classes="panel")
                yield _Panel(id="rails", classes="panel")
                yield _Panel(id="render", classes="panel")
                yield _Panel(id="story", classes="panel")
                yield _Panel(id="ticker", classes="panel")

            async def on_mount(self):
                if not self.replay:
                    self.set_interval(self.interval, self.refresh_data)
                self.set_interval(0.25, self.redraw)
                await self.refresh_data()

            async def refresh_data(self):
                if self.paused or self.poller is None or self._polling:
                    return
                self._polling = True
                view_generation = self._view_generation
                try:
                    try:
                        # Keep blocking HTTP/local-stat reads off Textual's event loop.
                        tick = await asyncio.to_thread(self.poller.tick)
                        # A pause/scrub owns its fixed ring view. Discard a poll
                        # that began before any navigation, even if the user has
                        # already scrubbed back to LIVE while it was in flight.
                        if self.paused or view_generation != self._view_generation:
                            return
                        tick["events"] = self.ring.dedupe_events(tick["events"])
                        self._last_poll_error = None
                    except Exception as exc:
                        if self.paused or view_generation != self._view_generation:
                            return
                        # A malformed field after the fetch still counts as one
                        # poll: push and record an UNKNOWN tick in its place, so
                        # --record keeps the cycle.
                        repr_exc = repr(exc)
                        if repr_exc != self._last_poll_error:
                            self._last_poll_error = repr_exc
                            self.notify(f"poll failed: {exc}", title="poll error",
                                        severity="error", timeout=10)
                        tick = self.poller.unknown_tick(type(exc).__name__)

                    self.ring.push(tick)
                    if self.record:
                        try:
                            await asyncio.to_thread(_append_record, self.record, tick)
                        except OSError as exc:
                            self.notify(
                                f"recording stopped: {exc}", title="recording error",
                                severity="error", timeout=10,
                            )
                            self.record = None
                    # Feed the run story after recording, so a run-story bug
                    # cannot leave a row missing from an otherwise valid capture.
                    self.story.observe(tick)
                finally:
                    self._polling = False

            def redraw(self):
                self.frame += 1
                tick = self.current()
                _redraw_panels(
                    self, _Panel, tick,
                    ("header", "hosts", "rails", "render", "story", "ticker"),
                )

            def draw_header(self, tick):
                pal = self.pal
                text = Text(" DGX MONARCH ", style="bold")
                if tick is None:
                    text.append("  waiting for first sample…", style="dim")
                    return self.brand_header(text, tick)
                mode = ("REPLAY" if self.replay else
                        "PAUSED" if self.paused else "LIVE")
                mode_style = {"LIVE": pal["good"], "PAUSED": pal["warn"],
                              "REPLAY": pal["accent"]}[mode]
                text.append(f"  {mode} ", style=f"bold {mode_style}")
                if self.paused and self.ring.ticks:
                    text.append(f"[{self.cursor + len(self.ring.ticks) + 1}"
                                f"/{len(self.ring.ticks)}] ", style="dim")
                text.append(time.strftime(" %H:%M:%S ", time.localtime(tick["t"])), style="dim")
                text.append(f" ComfyUI telemetry {tick.get('comfy') or '?'} ", style="dim")
                text.append(" driver ", style="dim")
                text.append("●", style=pal["good"] if tick["driver_up"] else pal["bad"])
                for loop in tick.get("loops", []):
                    text.append(f"  {_loop_label(loop['addr'])} ", style="dim")
                    symbol, style = _loop_indicator(loop.get("up"), pal)
                    text.append(symbol, style=style)
                text.append_text(readiness_text(tick, pal))
                return self.brand_header(text, tick)

            def brand_header(self, text, tick):
                return header_layout(
                    text, width=self.size.width, frame=self.frame, animate=self.animate,
                    paused=self.paused, replay=bool(self.replay),
                    active=measured_activity(tick, now=time.time(), interval=self.interval),
                    mono=self.theme_name == "mono", unicode=supports_unicode(),
                )

            def draw_hosts(self, tick):
                pal = self.pal
                if not tick or not tick.get("hosts"):
                    return Text("no host telemetry yet", style="dim")
                out = Text()
                width = max(24, self.size.width - 46)
                for i, (host, entry) in enumerate(sorted(tick["hosts"].items())):
                    stats = entry["stats"]
                    mem = stats.get("mem_gib", {})
                    # Five used categories sum to MemTotal - MemAvailable.
                    # Reclaimable cache and the uncolored free remainder split
                    # MemAvailable, so colored segments + background = MemTotal.
                    total = mem.get("MemTotal") or 121.6
                    avail = mem.get("MemAvailable") or 0
                    anon = mem.get("AnonPages") or 0
                    cache = (mem.get("Cached") or 0) + (mem.get("Buffers") or 0)
                    wmeta = tick.get("workers", {}).get(host) or {}
                    # GB10: CUDA allocations are driver-owned pages outside
                    # AnonPages; the nvidia-smi per-process sum is models+CUDA.
                    # Capping each later segment by the used memory left absorbs
                    # any CUDA mapping AnonPages also counts.
                    used = max(0.0, total - avail)
                    gpu_mem = min(stats.get("gpu_proc_gib") or 0, used)
                    # The zero-copy weight slab keeps model bytes in Shmem
                    # (memfd), which neither the nvidia-smi per-process sum nor
                    # AnonPages counts; without this segment a slab-resident
                    # model shows as "other". Worker telemetry gives its size,
                    # capped by Shmem and by the used memory left.
                    wmem = wmeta.get("memory") or {}
                    slab = min(wmem.get("slab_gib") or 0,
                               mem.get("Shmem") or 0,
                               max(0.0, used - gpu_mem))
                    # The retained allocator pool is part of AnonPages and
                    # outlives a model unload.
                    pool = min(stats.get("pool_gib") or 0, anon,
                               used - gpu_mem - slab)
                    anon_seg = min(anon - pool, used - gpu_mem - slab - pool)
                    other = used - gpu_mem - slab - pool - anon_seg
                    memfree = mem.get("MemFree")
                    reclaim = (max(0.0, avail - memfree) if memfree is not None
                               else min(avail, max(0.0, cache - (mem.get("Shmem") or 0))))
                    if i:
                        out.append("\n")
                    out.append(f"{host[-4:]:>5s} ", style=f"bold {pal['accent']}")
                    out.append("[")
                    for span, style_key in segmented_bar(
                            [(gpu_mem, pal["model"]), (slab, pal["slab"]),
                             (pool, pal["pool"]), (anon_seg, pal["anon"]),
                             (other, pal["other"]), (reclaim, pal["cache"])], total, width):
                        out.append(span, style=style_key)
                    out.append("] ")
                    out.append(f"used {used:5.1f}/{total:.0f}G  avail {avail:5.1f}G",
                               style=pal["good"] if avail > 20 else pal["bad"])
                    gpu = stats.get("gpu", {})
                    if gpu:
                        out.append(f"   GPU {gpu.get('util_pct', 0):3d}% "
                                   f"{(gpu.get('clock_mhz') or 0) / 1000:.2f}GHz "
                                   f"{gpu.get('power_w', 0):5.1f}W {gpu.get('temp_c', 0):2d}°C",
                                   style="dim")
                    mm = wmem
                    if mm:
                        out.append("\n      ")
                        out.append(f"{mm.get('lora_mode', '?')} ", style=pal["accent"])
                        fb = mm.get("unbake_file_backed_gib")
                        if fb is not None:
                            out.append(f"unbake {fb}G file-backed  ", style="dim")
                        fails = mm.get("swap_verify_failures", 0)
                        out.append(f"verify✓ fails {fails}",
                                   style=pal["good"] if not fails else f"bold {pal['bad']}")
                    out.append("\n      ", style="dim")
                    out.append("█", style=pal["model"])
                    out.append(f" gpu {gpu_mem:.1f}G (models+CUDA)  ", style="dim")
                    if slab:
                        out.append("█", style=pal["slab"])
                        out.append(f" slab {slab:.1f}G (model, zero-copy)  ", style="dim")
                    out.append("█", style=pal["pool"])
                    out.append(f" pool {pool:.1f}G (retained)  ", style="dim")
                    out.append("█", style=pal["anon"])
                    out.append(f" anon {anon_seg:.1f}G  ", style="dim")
                    out.append("█", style=pal["other"])
                    out.append(f" other {other:.1f}G  ", style="dim")
                    out.append("█", style=pal["cache"])
                    out.append(f" cache {reclaim:.1f}G reclaimable  ", style="dim")
                    out.append(f"free {max(0.0, avail - reclaim):.1f}G", style="dim")
                return out

            def draw_rails(self, tick):
                pal = self.pal
                if not tick:
                    return Text("")
                out = Text()
                width = max(20, (self.size.width - 30) // 2)
                hosts = sorted(tick.get("hosts", {}).items())
                shown = False
                for host, entry in hosts[:1]:  # rails are symmetric: the first host by name is enough
                    for rail, rate in sorted((entry.get("rails") or {}).items()):
                        gbs = rate["rx_gbs"] + rate["tx_gbs"]
                        series = _series(
                            self.ring, self.cursor if self.paused else -1,
                            lambda t, h=host, r=rail: (
                                (t["hosts"].get(h, {}).get("rails") or {}).get(r, {})
                                .get("rx_gbs", 0)
                                + (t["hosts"].get(h, {}).get("rails") or {}).get(r, {})
                                .get("tx_gbs", 0)),
                            width * 2)
                        rows = braille_rows(series, width, 1, vmax=25.0)
                        out.append(f"{rail:>4s} ", style=f"bold {pal['accent']}")
                        out.append(rows[0] if rows else "", style=pal["accent"])
                        out.append(f" {gbs:5.2f} GB/s   ", style="dim")
                        shown = True
                    if shown:
                        out.append("\n     ")
                        out.append_text(_marker_lane(
                            self.ring, self.cursor if self.paused else -1, width))
                return out if shown else Text("no RDMA rails visible", style="dim")

            def draw_render(self, tick):
                pal = self.pal
                if not tick:
                    return Text("")
                render = tick.get("render") or {}
                out = Text()
                counts = (tick.get("ledger") or {}).get("combinations") or {}
                if render.get("active"):
                    step = render.get("step") or 0
                    steps = render.get("steps") or 1
                    # Distinguish proof, cross-check, and requested renders.
                    try:
                        label = self.view().live_label(render)
                    except Exception:
                        label = "render"
                    out.append(f" {label} ", style=f"bold {pal['accent']} reverse")
                    out.append(f" {render.get('model', '')} ", style="bold")
                    out.append(f" step {step}/{steps} ")
                    out.append(bar(step / max(1, steps), max(10, self.size.width - 60)),
                               style=pal["accent"])
                    sps = render.get("sec_per_step")
                    if sps:
                        eta = max(0, (steps - step)) * sps
                        out.append(f" {sps:.2f}s/it eta {eta:4.0f}s", style="dim")
                else:
                    out.append(" idle ", style="dim reverse")
                    last = render.get("last_wall_s")
                    if last:
                        out.append(f"  last render {last}s", style="dim")
                if counts:
                    # Color each verdict's count, not the whole row; gate_chip_style says why.
                    out.append("   gates:")
                    for verdict, count in sorted(counts.items()):
                        out.append(f" {verdict}:{count}", style=gate_chip_style(pal, verdict))
                return out

            def draw_story(self, tick):
                # Keep the latest run's phase timings and total until another run
                # starts. When paused, show timings through the cursor's sample.
                pal = self.pal
                styles = {"head": f"bold {pal['accent']} reverse", "cost": pal["accent"],
                          "good": f"bold {pal['good']}", "warn": pal["warn"], "dim": "dim"}
                out = Text()
                if self.story.last_error:
                    return out.append(
                        f"run story unavailable ({self.story.last_error})",
                        style=styles["warn"],
                    )
                for text, key in story_spans(self.view().story()):
                    out.append(text, style=styles.get(key, ""))
                return out

            def draw_ticker(self, tick):
                pal = self.pal
                if not tick:
                    return Text("")
                events: list = []
                idx = self.cursor if self.paused else -1
                for t in list(self.ring.ticks)[: len(self.ring.ticks) + idx + 1][-240:]:
                    events.extend(t.get("events", []))
                out = Text()
                for ev in events[-4:]:
                    ts = time.strftime("%H:%M:%S", time.localtime(event_time(ev)))
                    kind = ev.get("kind", "?")
                    body = "  ".join(f"{k}={v}" for k, v in ev.items()
                                     if k not in ("t", "kind"))
                    out.append(f"{ts} ", style="dim")
                    out.append(f"{kind:<10s}", style=EVENT_STYLES.get(kind, pal["accent"]))
                    out.append(f" {body}\n", style="")
                return out or Text("no events yet; render something", style="dim")

            def action_pause(self):
                if self.replay:
                    return
                self._view_generation += 1
                self.paused = not self.paused
                self.cursor = -1

            def action_back(self):
                # The left arrow works from LIVE too: the first press pauses
                # and steps back five ticks.
                self._view_generation += 1
                self.paused = True
                self.cursor = max(-len(self.ring.ticks), self.cursor - 5)

            def action_fwd(self):
                if not (self.paused or self.replay):
                    return
                self._view_generation += 1
                self.cursor = self.cursor + 5
                if self.cursor >= -1 and not self.replay:
                    self.cursor = -1
                    self.paused = False  # past the newest tick: back to LIVE
                self.cursor = min(-1, self.cursor)

            def action_anim(self):
                self.animate = not self.animate

            def action_theme(self):
                names = list(THEMES)
                self.theme_name = names[(names.index(self.theme_name) + 1) % len(names)]

            def action_snapshot(self):
                path = time.strftime("dgxm-top-%Y%m%d-%H%M%S.jsonl")
                n = self.ring.save(path)
                self.notify(f"saved {n} ticks to {path}")

            def action_ledger(self):
                tick = self.current() or {}
                rows = (tick.get("ledger") or {}).get("latest") or []
                body = "\n".join(
                    f"{e.get('verdict', '?'):12s} {e.get('model', '?')} "
                    f"loras={e.get('loras', '?')} comfy={e.get('comfy', '?')} {e.get('time', '')}"
                    for e in rows) or "no gate verdicts recorded"
                self.notify(body, title="gate ledger", timeout=10)

            def action_workers(self):
                import json as _json

                tick = self.current() or {}
                body = _json.dumps(tick.get("workers") or {}, indent=1, default=str)[:1500]
                self.notify(body or "no worker detail", title="workers", timeout=10)

        return _App()


def run(**kwargs) -> None:
    cast(Any, DgxmTopApp(**kwargs)).run()
