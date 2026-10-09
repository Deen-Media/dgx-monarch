"""Capture setup receipt timing before planning probes resolve the config digest."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, repr=False)
class PrestartedReceipt:
    """materialize() replays the start time and tick as the receipt builder's first clock and timer readings."""

    started_at: datetime
    started_tick: float
    clock: Callable[[], datetime]
    timer: Callable[[], float]

    @classmethod
    def start(
        cls,
        *,
        clock: Callable[[], datetime] | None = None,
        timer: Callable[[], float] | None = None,
    ) -> PrestartedReceipt:
        selected_clock = clock or (lambda: datetime.now(UTC))
        selected_timer = timer or time.monotonic
        return cls(selected_clock(), selected_timer(), selected_clock, selected_timer)

    def materialize(
        self,
        factory: Callable[..., Any],
        profile: str,
        config_digest: str,
        source_digest: str,
    ) -> Any:
        first_clock = first_timer = True

        def replay_clock() -> datetime:
            nonlocal first_clock
            if first_clock:
                first_clock = False
                return self.started_at
            return self.clock()

        def replay_timer() -> float:
            nonlocal first_timer
            if first_timer:
                first_timer = False
                return self.started_tick
            return self.timer()

        return factory(
            "setup",
            profile=profile,
            source_hashes={"cluster": config_digest, "source": source_digest},
            clock=replay_clock,
            timer=replay_timer,
        )
