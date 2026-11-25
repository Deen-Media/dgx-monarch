"""The one-step amax trace of the shared NVFP4 activation scale (log_act_scale).

The caller passes its own logger, so the line keeps its module's name.
"""
from __future__ import annotations

import logging
from typing import Any

import torch


@torch.compiler.disable
def trace_line(log: logging.Logger, path: str, record: list[torch.Tensor], scale: torch.Tensor,
               plan: Any) -> None:
    """One line per wrapped Linear per rank, for the first step only.

    The only host sync of the shared scale, and it runs under the knob alone.
    """
    local, reduced = record
    log.info("shared activation scale: %s local_amax=%.6g reduced_amax=%.6g "
             "scale=%.6g reducers=%s", path, float(local), float(reduced),
             float(scale), ",".join(plan.names) or "none")
