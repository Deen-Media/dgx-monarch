"""What auto read for one render, carried beside the resolved topology.

``resolve_topology`` returns ``(topology, sage, reason)``. Its auto path
returns a ``ResolvedTopology``, which unpacks to the same three values and
also carries ``facts``: the table row, the family, quant kind and latent
megapixels auto matched on, the CFG and cfg++ flag it read, the cfg evidence
row that admitted cfg2 or the cfg row skipped for missing evidence, and the
batch. The render's progress record keeps them, so a reader of
/dgxm/telemetry can audit the table against measured times. Nothing decides
on them. A preset reads no checkpoint header, so it carries no facts,
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
    family: str | None = None  # row 99 serves every family without a row
    cfg: float | None = None
    keeps_uncond: bool | None = None  # the cfg++ tag the latent carried
    evidence_row: int | None = None  # the record that admitted cfg-parallel
    skipped_cfg_row: int | None = None  # a cfg row passed over for its evidence
    batch: int | None = None


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


_AUDIT_FIELDS = ("family", "cfg", "keeps_uncond", "evidence_row", "skipped_cfg_row", "batch")


def auto_facts(decision: object, *, world: int, quant: str, megapixels: float,
               family: str, cfg: float | None, keeps_uncond: bool,
               batch: int) -> TopologyFacts:
    """What auto read and chose for one render, from its AutoDecision.

    World 1 consults no table row, whatever the decision's rule field holds,
    so the row facts stay None there. A test double without the fields reads
    as no row and no cfg-parallel.
    """
    from ..topology_cfg_evidence import cfg_evidence_for

    table = world > 1
    rule = getattr(decision, "rule", None) if table else None
    cfg_parallel = int(getattr(getattr(decision, "topology", None), "cfg", 1) or 1) > 1
    evidence = cfg_evidence_for(rule) if cfg_parallel else None
    skipped = getattr(decision, "skipped_cfg", None) if table else None
    return TopologyFacts(
        row=getattr(rule, "row", None), quant=quant, megapixels=megapixels,
        family=family, cfg=None if cfg is None else float(cfg),
        keeps_uncond=bool(keeps_uncond), evidence_row=getattr(evidence, "row", None),
        skipped_cfg_row=getattr(skipped, "row", None), batch=int(batch))


def telemetry_facts(resolved: object) -> dict[str, Any]:
    """The progress record's auto facts, each None when ``resolved`` has none."""
    facts = getattr(resolved, "facts", None)
    if not isinstance(facts, TopologyFacts):
        return {"auto_row": None, "megapixels": None, "quant": None,
                **dict.fromkeys(_AUDIT_FIELDS)}
    return {
        "auto_row": facts.row,
        "megapixels": round(facts.megapixels, 3),
        "quant": facts.quant,
        **{name: getattr(facts, name) for name in _AUDIT_FIELDS},
    }
