"""Topology-bound adapter attention context construction."""
from __future__ import annotations

from typing import Any

from ..adapters import InjectionContext


def injection_context(topology: dict[str, Any], sequence_parallel: int,
                      attention: object) -> InjectionContext:
    """Enable pure-Ulysses adapter behavior only when Ring is absent.

    This includes joint-order restoration, full-row text paths, pad-row removal,
    and stock fused kernels. CFG is orthogonal to the sequence group, so
    Ulysses+CFG remains eligible; Ring and hybrid topologies do not.
    """
    ulysses = int(topology.get("ulysses", 1))
    ring = int(topology.get("ring", 1))
    return InjectionContext(
        topology_sp=sequence_parallel, usp_attention=attention,
        pure_ulysses=(sequence_parallel > 1 and ulysses > 1 and ring == 1),
    )
