"""Sample /proc/meminfo for manually driven capacity tests.

Write the sweep runner's fields and JSON-lines format at its sampling cadence,
with a test label on each row. This lets the same reader summarize automated
sweep cells and tests where an operator holds memory and loads a checkpoint.

    python -m benchmark.sweep.memsample --config sweep.toml --leg m1-leg3 \
        --out legs/m1-leg3.jsonl
    python -m benchmark.sweep.memsample --summary legs/m1-leg3.jsonl

Stop on SIGINT, SIGTERM, or after --duration-s.
"""
from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from collections.abc import Iterable
from pathlib import Path

from .matrix import load_config
from .run import Sampler

# The four figures a leg is read at. File pages are the sum of the two file LRU
# lists, which no single meminfo line carries and which is the figure the page
# cache question is stated in.
FILE_PAGES = ("Active(file)", "Inactive(file)")
SUMMARY_FIELDS = ("MemAvailable", "Shmem", "AnonPages", "file")
NOT_A_BOX = ("t", "leg")  # every other key on a row names a box


def hosts_for(sibling: str, extra: Iterable[str]) -> list[str]:
    """The local box, the config's second box, and anything named by hand.

    The empty string is the local box, which is what the runner's own host list
    means by it and what ``_on`` runs without ssh.
    """
    hosts = [""]
    for host in [sibling, *extra]:
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def summarize(lines: Iterable[str]) -> dict:
    """Min, max and last of the four leg figures, per box.

    A row written before 2026-09-03 carries only MemAvailable and AnonPages, so
    a field a file never held is left out rather than reported as zero.
    """
    boxes: dict[str, dict[str, dict[str, float]]] = {}
    legs: list[str] = []
    samples = 0
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        samples += 1
        if row.get("leg") and row["leg"] not in legs:
            legs.append(row["leg"])
        for name, values in row.items():
            if name in NOT_A_BOX or not isinstance(values, dict) or not values:
                continue
            held = boxes.setdefault(name, {})
            for field in SUMMARY_FIELDS:
                if field == "file":
                    if not all(key in values for key in FILE_PAGES):
                        continue
                    value = round(sum(values[key] for key in FILE_PAGES), 3)
                elif field in values:
                    value = values[field]
                else:
                    continue
                seen = held.setdefault(field, {"min": value, "max": value, "last": value})
                seen["min"] = min(seen["min"], value)
                seen["max"] = max(seen["max"], value)
                seen["last"] = value
    return {"samples": samples, "legs": legs, "boxes": boxes}


def render_summary(name: str, stats: dict) -> str:
    """The table an operator reads a leg from, one line per box and figure."""
    lines = [f"# meminfo summary: {name}",
             f"samples {stats['samples']}, legs {', '.join(stats['legs']) or 'unlabelled'}", ""]
    for box, held in stats["boxes"].items():
        lines.append(box)
        for field in SUMMARY_FIELDS:
            seen = held.get(field)
            if seen is None:
                lines.append(f"  {field:<14} n/a, no such field in this file")
                continue
            lines.append(f"  {field:<14} min {seen['min']:>9.2f}  max {seen['max']:>9.2f}  "
                         f"last {seen['last']:>9.2f}")
    return "\n".join(lines) + "\n"


def summary_for(path: Path) -> str:
    with open(path) as handle:
        return render_summary(path.name, summarize(handle))


def sample(hosts: list[str], path: Path, leg: str, duration_s: float) -> dict:
    """Sample until a signal or the duration, and answer with the summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: stop.set())
    sampler = Sampler(hosts, path, label=leg)
    sampler.start()
    print(f"sampling {len(hosts)} box(es) at 1 Hz into {path}, leg "
          f"{leg or 'unlabelled'}; stop with SIGINT or SIGTERM", flush=True)
    deadline = time.monotonic() + duration_s if duration_s else 0.0
    while not stop.wait(0.5):
        if deadline and time.monotonic() >= deadline:
            break
    sampler.close()
    return sampler.summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.sweep.memsample",
        description="Sample every meminfo field the capacity legs read, at 1 Hz, on every box.")
    parser.add_argument("--summary", help="read a sampler file and print min, max and last")
    parser.add_argument("--config", help="sweep TOML; its sibling box joins the host list")
    parser.add_argument("--host", action="append", default=[],
                        help="another ssh host to sample, repeatable")
    parser.add_argument("--out", help="JSONL file to append rows to")
    parser.add_argument("--leg", default="", help="label written on every row")
    parser.add_argument("--duration-s", type=float, default=0.0,
                        help="stop after this many seconds; 0 samples until a signal")
    args = parser.parse_args(argv)
    if args.summary:
        print(summary_for(Path(args.summary).expanduser()), end="")
        return 0
    if not args.out:
        parser.error("pass --out with the file to write, or --summary with a file to read")
    sibling = load_config(Path(args.config))["sibling"] if args.config else ""
    hosts = hosts_for(sibling, args.host)
    summary = sample(hosts, Path(args.out).expanduser(), args.leg, args.duration_s)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
