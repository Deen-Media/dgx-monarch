"""Measured eligibility for the auto table's cfg-parallel rows.

Auto picks cfg2 only when one record covers that exact table row: row number,
family, checkpoint quantization, resolution band, world, sage off, an elapsed-time
improvement and fidelity, and, for a record measured on unequal prompt pairs
only, a pair whose text rows differ. Missing or inconclusive evidence never
grants cfg2; auto then skips the row and falls back to Ulysses instead of
refusing an otherwise supported render.
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
    # "unequal": the record was measured exact only on prompt pairs whose text
    # rows differ, which one GPU runs as two calls. A pair of equal rows folds
    # into one batch-2 call there, which cfg2's two batch-1 calls do not
    # reproduce, so the record grants only a render the sampler tagged
    # (cfg_pair.text_rows_differ). "any" grants every pair.
    pairs: str = "any"
    # The kinds the "unequal" scope covers, as ``pair_kind`` names them; empty
    # covers every kind. An equal pair leaves cfg2 only where the row it then
    # lands on reads bytes equal to one GPU, so a kind whose fallback is an
    # approximate route keeps every pair on cfg2 (row 15: Krea2 FP8 lands on
    # row 11's SAGE route; a block-scaled file, or a full-precision file cast
    # to fp8 at load, on any fp8 row), wherever the cfg2 row runs
    # (topology.cfg_row_strands).
    pair_quants: tuple[str, ...] = ()

    def scopes_pairs(self, pair_kind: str) -> bool:
        """Whether this record grants only a differing pair for this kind (``pair_kind``)."""
        return self.pairs == "unequal" and (not self.pair_quants
                                             or pair_kind in self.pair_quants)

    def permits(self, row: object, quant_kind: str, variant: str | None = None,
                pair_differs: bool = False, pair_kind: str | None = None) -> bool:
        """Whether this result covers the candidate row, artifact and prompt pair.

        ``pair_kind`` is the finer kind the pair scope reads (``pair_kind``),
        the quant kind when the caller has no file to read.
        """
        return (
            (variant is None or variant not in self.excludes)
            and (pair_differs or not self.scopes_pairs(pair_kind or quant_kind))
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
_BIT_IDENTICAL_UNEQUAL = ("bit-identical to one GPU on a prompt pair whose text rows differ, "
                          "full schedule and one-step probe")
_EQUAL_PAIR_READING = ("on 2026-11-07 (dgx-monarch a76e967, ComfyUI 8a98ecf58, head capped, SM "
                       "plateau 2093 MHz) that pair read bit-identical again, while two pairs "
                       "of equal text rows, which fold onto the slice path, did not")

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
        fidelity="bit-identical to one GPU on the shipped prompt pair, whose text rows differ",
        faster=True,
        exact=True,
        in_band=True,
        basis="the 2026-10-01 probes and warm walls at 1.05 MP after the fix that trims chroma's "
              "cfg2 text pad, on bf16, fp8_scaled, fp8mixed, mxfp8 and int8_convrot; uly2 reads 0.129 "
              "on mxfp8; nvfp4 shares the fp8 kind and stays class K (an equal-length pair reads 0.148 "
              "on cfg2); under a waiver the shipped uneven pair reads 0.0 on cfg2 against 0.297 on uly2; "
              "a BF16 pair of equal text rows, which one GPU folds into one call, read NRMS 0.034 on "
              "cfg2, so since 2026-11-07 this record grants BF16, FP8 storage and INT8 only a pair "
              "whose text rows differ; an MXFP8, NVFP4 or cast file has no bytes-equal reading on "
              "the fallback, so since 2026-11-24 its equal pair keeps cfg2",
        pairs="unequal",
        pair_quants=("bf16", "fp8", "int8"),
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
              "excludes it. The template pair's text rows differ. On 2026-11-24 (dgx-monarch "
              "c5d9084, RAW BF16, 1280 square, CFG 3.5, one step) pairs of equal rows read "
              "1-step NRMS 0.0104 (a ConditioningZeroOut of the positive) and 0.039 to 0.040 "
              "(real negatives) on cfg2, equal to one GPU's two separate calls. An equal "
              "pair leaves cfg2 only where its fallback reads bytes equal to one GPU: on BF16 "
              "that is row 12 (uly2 on the Init kernel, bytes equal on the pair that day), so "
              "since then this record grants BF16 only a pair whose text rows differ; on FP8 "
              "it would be row 11's SAGE route, 1-step NRMS 0.030 to 0.058 in Campaign K "
              "against row 0 of a same-build dp2 render, the one-GPU reference, further than "
              "cfg2, so FP8 keeps every pair",
        excludes=("kroma",),
        pairs="unequal",
        pair_quants=("bf16",),
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
        fidelity=_BIT_IDENTICAL_UNEQUAL,
        faster=True,
        exact=True,
        in_band=True,
        basis="Campaign K 2026-10-14, dgx-monarch d549f10d, ComfyUI e8665ca8, head capped at "
              "2100 MHz, run_matrix warm sampler walls on Chroma1-HD BF16, FP8-mixed and INT8 "
              "ConvRot at 1152, 1216 and 1280 square on the template prompt pair, whose text rows "
              "differ; fp8_scaled and mxfp8 files were not measured in this band, nvfp4 shares the "
              "fp8 kind and stays class K, and Radiance shares the bf16 kind and was not measured in "
              "this band; a BF16 pair of equal text rows read NRMS 0.034 on cfg2 at 1024, so since "
              "2026-11-07 this record grants only a pair whose text rows differ. Since 2026-11-24 "
              "that holds only where the equal pair's fallback reads bytes equal to one GPU: BF16 "
              "falls to row 22, uly2 on the Init kernel, which read Chroma1-HD BF16 exact against "
              "a same-build one-GPU render at 1024 and 1536 on 2026-10-02, a pair of equal rows "
              "included, and INT8 to row 22 too, read bit-identical in Campaign K against row 0 "
              "of a same-build dp2 render, the one-GPU reference; FP8 falls to row 21's SAGE "
              "route, 1-step NRMS 0.057 to 0.078 against that reference, further than cfg2's "
              "equal pair, so an FP8 file, block-scaled or cast ones included, keeps every pair",
        pairs="unequal",
        pair_quants=("bf16", "int8"),
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
              "other fp8-kind files were not measured. No pair of equal text rows was "
              "measured here, and one GPU folds such a pair into one batch-2 call where "
              "cfg2 makes two batch-1 calls, which read apart on Chroma, Boogu, Krea2 and "
              "Qwen-Image 2.1 (2026-11-07 and 2026-11-24); an equal pair's fallback, row 71 "
              "(uly2 on the Init kernel), read bit-identical to one GPU on 2026-10-06, so "
              "since 2026-11-24 this record grants an FP8 file only a pair whose text rows "
              "differ. A block-scaled file (MXFP8 or NVFP4) reads fp8 too, but no sharded "
              "render of one is bytes equal, so its equal pair keeps cfg2, as does that of "
              "a full-precision file cast to fp8 at load, which no record measured",
        pairs="unequal",
        pair_quants=("fp8",),
    ),
    79: CfgEvidence(
        row=79,
        family="boogu",
        quants=("fp8",),
        min_mp=0.0,
        max_mp=1.2,
        min_world=2,
        speedup="1.98x faster than one GPU and 1.40x faster than uly2",
        fidelity=_BIT_IDENTICAL_UNEQUAL,
        faster=True,
        exact=True,
        in_band=True,
        basis="Campaign K 2026-10-14, dgx-monarch d549f10d, ComfyUI e8665ca8, head capped at "
              "2100 MHz, run_matrix warm sampler walls on the fp8_scaled file at 1024 square "
              "on the template's prompt pair, whose text rows differ, so it takes the "
              f"per-cond dispatch path; {_EQUAL_PAIR_READING}: full-schedule NRMS 0.279 and "
              "0.077, one-step 0.014 and 0.025; an MXFP8, NVFP4 or cast file has no bytes-equal "
              "reading on the fallback, so since 2026-11-24 its equal pair keeps cfg2",
        pairs="unequal",
        pair_quants=("fp8",),
    ),
}


def cfg_evidence_for(row: object) -> CfgEvidence | None:
    """The evidence tied to this exact cfg row, if any."""
    row_id = getattr(row, "row", None)
    return CFG2_EVIDENCE.get(row_id) if isinstance(row_id, int) else None


def cfg_row_is_eligible(row: object, quant_kind: str, variant: str | None = None,
                        pair_differs: bool = False, pair_kind: str | None = None) -> bool:
    """Fail-closed cfg admission shared by auto selection and tests."""
    record = cfg_evidence_for(row)
    return record is not None and record.permits(row, quant_kind, variant, pair_differs,
                                                 pair_kind)


# What ``pair_kind`` names an fp8 header whose file is block scaled.
BLOCK_SCALED_FP8 = "fp8_block_scaled"
# What ``pair_kind`` names a full-precision file the loader casts to fp8.
CAST_FP8 = "fp8_cast"


def pair_kind(path: str | None, quant_kind: str) -> str:
    """The kind a pair scope reads: the quant kind, with two fp8 kinds named apart.

    A scope moves an equal pair to the row a record measured bytes equal to one
    GPU, and the fp8 rows were measured on files that store FP8 weights, with
    or without a per-tensor scale. The driver's ``fp8`` kind also covers MXFP8
    and NVFP4 files, whose block scales are F8 tensors too, and their sharded
    renders are not bytes equal (Krea2 NVFP4 uly2 reads 0.025 to 0.034). The
    file's layer records answer first where they name one kind; otherwise a
    block-scaled file holds U8 or F4 weights or scales (the records themselves
    are U8 tensors and are not counted). Such a file reads
    ``BLOCK_SCALED_FP8``, which no scope names, so its equal pair keeps cfg2.
    A full-precision file the loader casts to fp8 holds no F8 tensor at all, and
    no record measured one, so it reads ``CAST_FP8``, which no scope names
    either: its equal pair keeps cfg2 until a leg reads it. The caller has just
    read this header for the quant kind, so one that cannot be read here (the
    file changed in between) adds nothing and leaves the quant kind, as the
    sweep matrix predicts it.
    """
    if quant_kind != "fp8" or path is None:
        return quant_kind
    from .adapters import layer_records
    from .adapters.detect import CheckpointSniffError, read_safetensors_keys

    named = layer_records.name_fp8_header_kind(path)
    if named == "fp8":
        return quant_kind
    if named in ("mxfp8", "nvfp4"):
        return BLOCK_SCALED_FP8
    try:
        _keys, header = read_safetensors_keys(path)
    except (CheckpointSniffError, OSError):
        return quant_kind
    tensors = header.get("__tensors__", {})
    dtypes = {str(info.get("dtype", "")) for name, info in tensors.items()
              if isinstance(info, dict)
              and not (name == "comfy_quant" or name.endswith(".comfy_quant"))}
    if dtypes & {"U8", "F4"} or any(dt.startswith("F8_E8M0") for dt in dtypes):
        return BLOCK_SCALED_FP8
    if not any(dt.startswith("F8_") for dt in dtypes):
        return CAST_FP8
    return quant_kind


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


def loaded_quant_kind(quant_kind: str, weight_dtype: object) -> str:
    """The kind auto reads for a header's ``quant_kind`` under the loader's ``weight_dtype``.

    fp32 stays in the set (the sniff reports it as itself): the loader casts
    every full-precision storage kind to fp8, so the fp8 row describes it.
    """
    if quant_kind in {"bf16", "fp16", "fp32"} and str(weight_dtype).startswith("fp8"):
        return "fp8"
    return quant_kind


def file_tags(path: str, family: str, quant_kind: str, pair_differs: bool) -> dict[str, str]:
    """What the checkpoint file adds to auto's cfg evidence, as ``choose_auto_topology`` tags.

    The ``variant`` a record may exclude, read under a forced family too (naming
    krea2 must not lift a Kroma exclusion), and the ``pair_kind`` a pair scope
    reads, which only an equal or unread pair meets. Each rides only when it
    adds something, so an untagged call keeps its shape.
    """
    tags: dict[str, str] = {}
    variant = sniffed_checkpoint_variant(path, family)
    if variant:
        tags["variant"] = variant
    if not pair_differs:
        kind = pair_kind(path, quant_kind)
        if kind != quant_kind:
            tags["pair_kind"] = kind
    return tags


def sniffed_checkpoint_variant(path: str, family: str) -> str | None:
    """``checkpoint_variant`` of a file on disk; an unreadable header names none."""
    from .adapters.detect import CheckpointSniffError, sniff_metadata_keys

    try:
        return checkpoint_variant(family, sniff_metadata_keys(path))
    except (CheckpointSniffError, OSError):
        return None
