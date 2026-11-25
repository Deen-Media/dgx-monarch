"""Slab residency needs unified memory: the one posture rule that holds off it.

``comfy_bridge._uma_memory_defaults`` sets a Worker's unified-memory posture,
and calls ``without_slab`` on every path that leaves that posture (a discrete
GPU, or a probe that could not tell).
"""
from __future__ import annotations

from ..log import get_logger
from ..transfer_utils import safe_call

log = get_logger(__name__)


def without_slab(worker_args: dict, why: str) -> dict:
    """Keep slab residency off where the GPU's memory is not the host's.

    A slab wraps the arena's host address as a device tensor, which is true
    only on unified memory. Any request for it, an explicit ``on`` or a
    configured ``"auto"``, turns off with a warning, and the weights load stock.
    """
    if not worker_args.get("slab_weights"):
        return worker_args
    safe_call(log.warning, "slab_weights needs unified memory and %s; weights stay on cudaMalloc", why)
    return {**worker_args, "slab_weights": False}
