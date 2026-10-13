"""What auto read for one render, carried beside the resolved topology.

``resolve_topology`` returns ``(topology, sage, reason)``. Its auto path
returns a ``ResolvedTopology``, which unpacks to the same three values and
also carries ``facts``: the table row, the quant kind and the latent
megapixels auto matched on. The render's progress record keeps them, so a
reader of /dgxm/telemetry can see where the table's choice is slow. Nothing
decides on them. A preset reads no checkpoint header, so it carries no facts,
and neither does a test double that returns a plain tuple.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..topology import Topology


@dataclass(frozen=True)
class TopologyFacts:
    row: int | None  # None when no table row applied (world size 1)
    quant: str
    megapixels: float


class ResolvedTopology(tuple[Topology, bool, str]):
    """``(topology, sage, reason)`` that also carries the auto facts."""

    facts: TopologyFacts | None

    def __new__(cls, topology: Topology, sage: bool, reason: str,
                facts: TopologyFacts | None = None) -> ResolvedTopology:
        resolved = super().__new__(cls, (topology, sage, reason))
        resolved.facts = facts
        return resolved

    def __reduce__(self) -> tuple[Any, ...]:
        # tuple's own reduce would rebuild through __new__ with one argument.
        return (type(self), (*self, self.facts))


def telemetry_facts(resolved: object) -> dict[str, Any]:
    """The progress record's auto facts, each None when ``resolved`` has none."""
    facts = getattr(resolved, "facts", None)
    if not isinstance(facts, TopologyFacts):
        return {"auto_row": None, "megapixels": None, "quant": None}
    return {
        "auto_row": facts.row,
        "megapixels": round(facts.megapixels, 3),
        "quant": facts.quant,
    }
