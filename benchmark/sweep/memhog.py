"""Hold local MemAvailable near a target during a capacity test.

Allocate anonymous pages and touch each page to make it resident. Read
/proc/meminfo after every adjustment so the controller accounts for other
processes and the kernel's current measurement.

    python -m benchmark.sweep.memhog --target-gib 70

SIGINT or SIGTERM releases the allocations and exits 0. Targets below
MIN_TARGET_GIB are refused to leave enough memory for SSH and manual shutdown.
"""
from __future__ import annotations

import argparse
import signal
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .run import parse_meminfo

GIB = 2**30
PAGE = 4096  # one store per page is what makes a mapping resident
MIN_TARGET_GIB = 8.0
TOLERANCE_GIB = 0.5
# One eighth of the tolerance, so growing and shrinking are both finer than the
# band the hog is asked to land in.
BLOCK_GIB = 0.0625
# Per step, so a signal lands within one allocation rather than behind a 60 GiB
# one, and so the reading the next step acts on is fresh.
STEP_CAP_GIB = 4.0
REPORT_S = 60.0
MEMINFO_PATH = Path("/proc/meminfo")


def mem_available_gib() -> float:
    """This box's MemAvailable in GiB, read the way the sampler reads it."""
    return parse_meminfo(MEMINFO_PATH.read_text()).get("MemAvailable", 0.0)


class Hog:
    """Anonymous pages held against a MemAvailable target."""

    def __init__(self, target_gib: float,
                 read_available: Callable[[], float] = mem_available_gib, *,
                 tolerance_gib: float = TOLERANCE_GIB, floor_gib: float = MIN_TARGET_GIB,
                 block_gib: float = BLOCK_GIB, step_cap_gib: float = STEP_CAP_GIB) -> None:
        if target_gib < floor_gib:
            raise ValueError(
                f"a target of {target_gib:.2f} GiB is under the {floor_gib:.2f} GiB floor: "
                "a box held that low stops answering ssh, so the operator cannot end the leg by hand")
        self.target, self.read_available = target_gib, read_available
        self.tolerance, self.block_gib, self.step_cap = tolerance_gib, block_gib, step_cap_gib
        self.blocks: list[bytearray] = []

    @property
    def held_gib(self) -> float:
        return round(len(self.blocks) * self.block_gib, 3)

    def _grab(self, count: int) -> None:
        nbytes = int(self.block_gib * GIB)
        pages = -(-nbytes // PAGE)
        for _ in range(count):
            block = bytearray(nbytes)
            block[::PAGE] = b"\x01" * pages  # one C-level store per page
            self.blocks.append(block)

    def _drop(self, count: int) -> None:
        del self.blocks[len(self.blocks) - count:]

    def step(self) -> tuple[str, float]:
        """One adjustment toward the target, and the reading it acted on.

        The verbs are grew, shrank, held, and starved: the box is already under
        the target with nothing of ours to give back, which is a fact about the
        rest of the box and not something the hog can fix.
        """
        available = self.read_available()
        gap = available - self.target
        if abs(gap) <= self.tolerance:
            return "held", available
        if gap < 0 and not self.blocks:
            return "starved", available
        blocks = min(int(abs(gap) / self.block_gib), int(self.step_cap / self.block_gib))
        if not blocks:
            return "held", available
        if gap > 0:
            self._grab(blocks)
            return "grew", available
        self._drop(min(blocks, len(self.blocks)))
        return "shrank", available

    def release(self) -> None:
        self.blocks.clear()


def hold(hog: Hog, stop: threading.Event, report_s: float = REPORT_S, poll_s: float = 1.0,
         log: Callable[[str], None] = print) -> None:
    """Adjust memory until stopped, reporting held memory once a minute.

    Read again immediately after an adjustment. Wait after an unchanged step to
    avoid busy polling when the host is already below its target.
    """
    said, last = "", 0.0
    while not stop.is_set():
        verb, available = hog.step()
        now = time.monotonic()
        if now - last >= report_s or (verb == "starved" and said != verb):
            log(f"holding {hog.held_gib:.2f} GiB, MemAvailable {available:.2f} GiB "
                f"against a {hog.target:.2f} GiB target ({verb})")
            last, said = now, verb
        if verb in ("held", "starved"):
            stop.wait(poll_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.sweep.memhog",
        description="Hold this box's MemAvailable at a target while a capacity leg runs.")
    parser.add_argument("--target-gib", type=float, required=True,
                        help=f"MemAvailable to hold, in GiB; under {MIN_TARGET_GIB:.0f} is refused")
    parser.add_argument("--tolerance-gib", type=float, default=TOLERANCE_GIB,
                        help="how far from the target, in GiB, still counts as held")
    args = parser.parse_args(argv)
    try:
        hog = Hog(args.target_gib, tolerance_gib=args.tolerance_gib)
    except ValueError as exc:
        parser.error(str(exc))
    stop = threading.Event()
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda *_: stop.set())
    print(f"holding MemAvailable at {args.target_gib:.2f} GiB, within "
          f"{args.tolerance_gib:.2f} GiB; stop with SIGINT or SIGTERM", flush=True)
    try:
        hold(hog, stop)
    finally:
        hog.release()
    print(f"released, MemAvailable {mem_available_gib():.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
