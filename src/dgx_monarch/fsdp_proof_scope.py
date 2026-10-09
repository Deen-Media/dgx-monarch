"""The FSDP clean-reload proof scope and the topology question that selects it.

The node-side Gate path and the driver's residency check both ask this of a
resolved topology, so the name and the predicate sit below either caller.
"""
from __future__ import annotations

FSDP_PROOF_SCOPE = "fsdp_clean_reload_v1"


def topology_resolves_fsdp(resolved_topology) -> bool:
    """Whether this concrete topology shards the weights, LoRA or not.

    FSDP has no stock residency to fall back to, with or without a LoRA stack,
    so a render without a PASS is refused typed rather than dispatched with
    the residency levers forced off.
    """
    return bool(
        resolved_topology.get("fsdp")
        if isinstance(resolved_topology, dict)
        else getattr(resolved_topology, "fsdp", False)
    )


def topology_requires_fsdp_proof(
    resolved_topology,
    loras,
) -> bool:
    """Whether this concrete no-LoRA topology needs the FSDP proof scope."""
    return topology_resolves_fsdp(resolved_topology) and not bool(loras)
