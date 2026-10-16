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
    # Checkpoint variants (checkpoint_variant) that read the record's family
    # and were measured to fail its bar; the record never grants them.
    excludes: tuple[str, ...] = ()

    def permits(self, row: object, quant_kind: str, variant: str | None = None) -> bool:
        """Whether this result covers the candidate row and artifact."""
        return (
            (variant is None or variant not in self.excludes)
            and getattr(row, "row", None) == self.row
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


_BIT_IDENTICAL = ("bit-identical to one GPU on the shipped prompt pair, full schedule and "
                  "one-step probe")

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
    # Campaign K, 2026-10-14. One GPU means batch row 0 of a dp2 batch-2 run on
    # the same build; walls are run_matrix warm sampler walls with no text
    # encode or VAE decode, so they do not compare with row 20's API walls.
    15: CfgEvidence(
        row=15,
        family="krea2",
        quants=("fp8", "bf16"),
        min_mp=1.2,
        max_mp=2.36,
        min_world=2,
        speedup="1.99x to 2.02x faster than one GPU; on FP8 1.21x to 1.29x faster than uly2 "
                "with SAGE_AUTO (1.25x to 1.30x faster than uly2 on TORCH_FLASH) and on BF16 "
                "1.23x to 1.28x faster than uly2",
        fidelity=_BIT_IDENTICAL,
        faster=True,
        exact=True,
        in_band=True,
        basis="Campaign K 2026-10-14, dgx-monarch d549f10d, ComfyUI e8665ca8, head capped at "
              "2100 MHz, run_matrix warm sampler walls on the template prompt pair at CFG 1 under "
              "euler_cfg_pp: the Turbo fp8_scaled file at 1152, 1216, 1280, 1408 and 1536 square "
              "plus a CFG 4.5 euler control at 1536, the RAW BF16 file at 1152, 1216, 1280 and "
              "1536 square and the Turbo BF16 file at 1536; mxfp8 and nvfp4 files share the fp8 "
              "kind and were not measured (September RAW nvfp4 cfg2 read 1-step NRMS 0.0113); "
              "RAW INT8 ConvRot read bit-identical at the same four sizes but 0.97x, 0.98x, "
              "1.03x and 1.08x against uly2, so this record does not grant it. Kroma BF16 "
              "(2026-10-16, dgx-monarch 70fb8997, ComfyUI 29e28e2b, CFG 1.5 under euler, "
              "1152 and 1536 square) was bit-identical and 1.24x to 1.31x faster than uly2 on "
              "an empty text negative, but its shipped ConditioningZeroOut negative has the "
              "positive's shape, one GPU folds the pair into one batch-2 call, and cfg2 read "
              "1-step NRMS 0.0087 to 0.0108 with full renders diverging, so this record "
              "excludes it",
        excludes=("kroma",),
    ),
    19: CfgEvidence(
        row=19,
        family="chroma",
        quants=("bf16", "fp8", "int8"),
        min_mp=1.2,
        max_mp=1.64,
        min_world=2,
        speedup="1.97x to 2.01x faster than one GPU, 1.30x to 1.31x faster than uly2 on BF16, "
                "1.26x to 1.30x faster than uly2 with SAGE_AUTO on FP8 and 1.17x to 1.33x faster "
                "than uly2 on INT8",
        fidelity=_BIT_IDENTICAL,
        faster=True,
        exact=True,
        in_band=True,
        basis="Campaign K 2026-10-14, dgx-monarch d549f10d, ComfyUI e8665ca8, head capped at "
              "2100 MHz, run_matrix warm sampler walls on Chroma1-HD BF16, FP8-mixed and INT8 "
              "ConvRot at 1152, 1216 and 1280 square on the template prompt pair; fp8_scaled and "
              "mxfp8 files were not measured in this band, nvfp4 shares the fp8 kind and stays "
              "class K, and Radiance shares the bf16 kind and was not measured in this band",
    ),
    68: CfgEvidence(
        row=68,
        family="qwen_image",
        quants=("fp8",),
        min_mp=1.2,
        max_mp=1.77,
        min_world=2,
        speedup="2.05x to 2.10x faster than one GPU and 1.58x to 1.66x faster than uly2",
        fidelity=_BIT_IDENTICAL,
        faster=True,
        exact=True,
        in_band=True,
        basis="Campaign K 2026-10-14, dgx-monarch d549f10d, ComfyUI e8665ca8, head capped at "
              "2100 MHz, run_matrix warm sampler walls on the base 2512 fp8_e4m3fn file at "
              "1152, 1216, 1280 and 1328 square on the template prompt pair; Edit-2511 and "
              "other fp8-kind files were not measured",
    ),
    79: CfgEvidence(
        row=79,
        family="boogu",
        quants=("fp8",),
        min_mp=0.0,
        max_mp=1.2,
        min_world=2,
        speedup="1.98x faster than one GPU and 1.40x faster than uly2",
        fidelity=_BIT_IDENTICAL,
        faster=True,
        exact=True,
        in_band=True,
        basis="Campaign K 2026-10-14, dgx-monarch d549f10d, ComfyUI e8665ca8, head capped at "
              "2100 MHz, run_matrix warm sampler walls on the fp8_scaled file at 1024 square; "
              "the template's unequal prompt pair takes the per-cond dispatch path, and an "
              "equal pair, which folds onto the slice path, was not measured",
    ),
    47: CfgEvidence(
        row=47,
        family="anima",
        quants=("bf16",),
        min_mp=0.0,
        max_mp=1.2,
        min_world=2,
        speedup="1.96x faster than one GPU and 1.27x faster than uly2",
        fidelity=_BIT_IDENTICAL,
        faster=True,
        exact=True,
        in_band=True,
        basis="Campaign K 2026-10-14, dgx-monarch 65d14333, ComfyUI e8665ca8, head capped at "
              "2100 MHz, run_matrix warm sampler walls on anima-base-v1.0 at 1024 square; the "
              "template's unequal prompt pair takes the per-cond dispatch path, and an equal "
              "pair, which folds onto the slice path, was not measured",
    ),
}


def cfg_evidence_for(row: object) -> CfgEvidence | None:
    """The evidence tied to this exact cfg row, if any."""
    row_id = getattr(row, "row", None)
    return CFG2_EVIDENCE.get(row_id) if isinstance(row_id, int) else None


def cfg_row_is_eligible(row: object, quant_kind: str, variant: str | None = None) -> bool:
    """Fail-closed cfg admission shared by auto selection and tests."""
    record = cfg_evidence_for(row)
    return record is not None and record.permits(row, quant_kind, variant)


# Kroma (a Krea2 fine-tune) ships as a merge, and its header records the merge
# inputs under these metadata names; Krea2 RAW and Turbo files carry none.
KROMA_MERGE_METADATA_KEYS = frozenset({"base", "copy_prefixes", "finetune", "target"})


def checkpoint_variant(family: str, metadata_keys: tuple[str, ...]) -> str | None:
    """Name a checkpoint a cfg record may treat apart from its family.

    Only the header decides: a renamed Kroma file is still Kroma, and a Kroma
    file re-saved without its merge metadata reads as plain Krea2.
    """
    if family == "krea2" and KROMA_MERGE_METADATA_KEYS <= set(metadata_keys):
        return "kroma"
    return None


def sniffed_checkpoint_variant(path: str, family: str) -> str | None:
    """``checkpoint_variant`` of a file on disk; an unreadable header names none."""
    from .adapters.detect import CheckpointSniffError, sniff_metadata_keys

    try:
        return checkpoint_variant(family, sniff_metadata_keys(path))
    except (CheckpointSniffError, OSError):
        return None
