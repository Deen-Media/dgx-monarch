"""Measured eligibility for the auto table's cfg-parallel rows.

Auto picks cfg2 only when one record covers that exact table row: row number,
family, checkpoint quantization, resolution band, world, sage off, an elapsed-time
improvement and fidelity. Missing or inconclusive evidence never grants cfg2; auto
then skips the row and falls back to Ulysses instead of refusing an otherwise
supported render.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CfgEvidence:
    """One scoped result that may authorize one cfg-parallel auto row."""

    row: int
    family: str
    quants: tuple[str, ...]
    min_mp: float
    max_mp: float
    min_world: int
    speedup: str
    fidelity: str
    faster: bool
    exact: bool
    in_band: bool
    basis: str

    def permits(self, row: object, quant_kind: str) -> bool:
        """Whether this result covers the candidate row and artifact."""
        return (
            getattr(row, "row", None) == self.row
            and getattr(row, "family", None) == self.family
            and (not self.quants or quant_kind in self.quants)
            and getattr(row, "mp_min", None) == self.min_mp
            and getattr(row, "mp_max", None) == self.max_mp
            and getattr(row, "world_min", None) == self.min_world
            and getattr(row, "topology", None) == {"cfg": 2}
            and getattr(row, "sage", None) is False
            and self.faster
            and self.exact
            and self.in_band
        )

    def clause(self) -> str:
        return (f"cfg2 evidence: {self.speedup}, {self.fidelity}, inside this row's "
                f"band; {self.basis}")


# Keyed by immutable table-row number rather than family. One family may hold
# more than one quant or resolution row, and a record grants only the quants it
# measured.
CFG2_EVIDENCE: dict[int, CfgEvidence] = {
    20: CfgEvidence(
        row=20,
        family="chroma",
        quants=("bf16", "fp8", "int8"),
        min_mp=0.0,
        max_mp=1.2,
        min_world=2,
        speedup="1.84x to 2.01x faster than one GPU and 1.29x to 1.34x faster than uly2",
        fidelity="bit-identical to one GPU on the shipped prompt pair",
        faster=True,
        exact=True,
        in_band=True,
        basis="the 2026-10-01 probes and warm walls at 1.05 MP after the fix that trims chroma's "
              "cfg2 text pad, on bf16, fp8_scaled, fp8mixed, mxfp8 and int8_convrot; uly2 reads 0.129 "
              "on mxfp8; nvfp4 shares the fp8 kind and stays class K (an equal-length pair reads 0.148 "
              "on cfg2); under a waiver the shipped uneven pair reads 0.0 on cfg2 against 0.297 on uly2",
    ),
    28: CfgEvidence(
        row=28,
        family="longcat",
        quants=("bf16",),
        min_mp=0.0,
        max_mp=1.2,
        min_world=2,
        speedup="2.00x faster",
        fidelity="step-nrms 0.006",
        faster=True,
        exact=True,
        in_band=True,
        basis="measured 2026-09-06 at 1.05 MP",
    ),
}


def cfg_evidence_for(row: object) -> CfgEvidence | None:
    """The evidence tied to this exact cfg row, if any."""
    row_id = getattr(row, "row", None)
    return CFG2_EVIDENCE.get(row_id) if isinstance(row_id, int) else None


def cfg_row_is_eligible(row: object, quant_kind: str) -> bool:
    """Fail-closed cfg admission shared by auto selection and tests."""
    record = cfg_evidence_for(row)
    return record is not None and record.permits(row, quant_kind)
