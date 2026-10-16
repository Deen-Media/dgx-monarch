"""Topology model and benchmark-backed auto-mode table (DESIGN.md §5.4).

Rules are ordered data (topology_table) keyed by model family, quantization,
megapixels, and world size. Each decision logs its matching row and rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .log import get_logger
from .refusal import RefusalClass, refusal
from .topology_cfg_evidence import cfg_evidence_for, cfg_row_is_eligible
from .topology_table import _FALLBACK_RULE, AUTO_TABLE, AutoRule

log = get_logger(__name__)


@dataclass(frozen=True)
class Topology:
    ulysses: int = 1
    ring: int = 1
    cfg: int = 1
    dp: int = 1
    fsdp: bool = False
    world: int = 1

    @property
    def sequence_parallel(self) -> int:
        return self.ulysses * self.ring

    @property
    def model_parallel(self) -> int:
        return self.sequence_parallel * self.cfg

    def validate(self) -> None:
        for name, v in (("ulysses", self.ulysses), ("ring", self.ring), ("cfg", self.cfg),
                        ("dp", self.dp), ("world", self.world)):
            if v < 1:
                raise ValueError(f"topology degree {name}={v} must be >= 1")
        if self.world % self.model_parallel != 0:
            raise ValueError(
                f"world size {self.world} is not divisible by ulysses*ring*cfg = "
                f"{self.ulysses}*{self.ring}*{self.cfg} = {self.model_parallel}"
            )
        if self.fsdp and self.ring > 1:
            raise ValueError(refusal(
                RefusalClass.KNOWN_WRONG,
                "ring sequence parallelism cannot combine with FSDP: a model "
                "reload changes ring attention output (measured 2026-08-27 on "
                "flux1-dev bf16: resident ring2 render-unload-render read NRMS "
                "0.004436 while uly2 stayed exact under the same sequence; issue #340), so the "
                "clean-reload gate can never certify the combination and every "
                "first-use ceremony on it fails. There is no waiver for this refusal. "
                "Use uly2+fsdp, which read bit-identical to resident on 19 image rows "
                "of the 2026-08-26 FSDP census, or resident ring2 without FSDP."))
        # Validate xFuser's exact world product before its opaque assertion.
        derived_dp = self.world // self.model_parallel
        if self.dp != derived_dp:
            raise ValueError(
                f"world size {self.world} != dp*ulysses*ring*cfg = "
                f"{self.dp}*{self.ulysses}*{self.ring}*{self.cfg}; set dp={derived_dp} "
                "(with_derived_dp) or adjust the degrees"
            )

    def with_derived_dp(self) -> Topology:
        derived = self.world // self.model_parallel
        return replace(self, dp=max(derived, 1))

    def describe(self) -> str:
        parts = []
        if self.ulysses > 1:
            parts.append(f"uly{self.ulysses}")
        if self.ring > 1:
            parts.append(f"ring{self.ring}")
        if self.cfg > 1:
            parts.append(f"cfg{self.cfg}")
        if self.dp > 1:
            parts.append(f"dp{self.dp}")
        if self.fsdp:
            parts.append("fsdp")
        return "+".join(parts) if parts else "single"


# Node widget presets; "auto" resolves through AUTO_TABLE.
PRESETS: dict[str, dict[str, int | bool]] = {
    "single": {},
    "uly2": {"ulysses": 2},
    "uly4": {"ulysses": 4},
    "uly8": {"ulysses": 8},
    "ring2": {"ring": 2},
    "ring4": {"ring": 4},
    "cfg2": {"cfg": 2},
    "dp2": {"dp": 2},
    "uly2+cfg2": {"ulysses": 2, "cfg": 2},
    "uly2+fsdp": {"ulysses": 2, "fsdp": True},
    "ring2+fsdp": {"ring": 2, "fsdp": True},
    # Capacity presets that keep the cheaper parallelism at 1 MP and below. The
    # NCCL launch-order guard counts cfg as model parallelism
    # (actor/worker_env.nccl_launch_order_needed).
    "cfg2+fsdp": {"cfg": 2, "fsdp": True},
    "dp2+fsdp": {"dp": 2, "fsdp": True},
}


def declares_fsdp(preset: str) -> bool:
    """Whether a named preset shards the weights, without resolving a world.

    Three sites ask it: the comfy-managed policy refusal and the shard-build
    price in nodes/loader_preflight, and the LoRA lever rule in
    nodes/consent_rescue. "auto" answers False even though
    ``choose_auto_topology`` can add the flag to a row that does not fit
    resident: a preset is what an operator declared, and a caller that has not
    resolved the auto table must keep the answer that charges more, never the
    one that charges less.
    """
    return preset != "auto" and bool(PRESETS.get(preset, {}).get("fsdp"))


def topology_from_preset(preset: str, world: int) -> Topology:
    if preset == "auto":
        raise ValueError("resolve 'auto' via choose_auto_topology()")
    if preset not in PRESETS:
        raise ValueError(f"unknown topology preset {preset!r}; choices: auto, {', '.join(PRESETS)}")
    topo = Topology(world=world, **PRESETS[preset])  # type: ignore[arg-type]
    topo = topo.with_derived_dp()
    topo.validate()
    return topo


# Family -> {target Ulysses degree: every sharded query and KV head count that
# degree must divide}. Empty: the evidence validates the degree-2 rows but no
# fold of data-parallel ranks into a larger Ulysses degree. Add one entry per
# family and degree a campaign validates.
VALIDATED_ULYSSES_FOLDS: dict[str, dict[int, tuple[int, ...]]] = {}


def _presets_tiling(world: int) -> list[str]:
    """Preset names whose degrees multiply to ``world``, so dp stays at one.

    The operator's escape from an auto refusal: every one of these runs the
    whole world on model parallelism, whatever the batch size is.
    """
    names = []
    for name, degrees in PRESETS.items():
        product = 1
        for key in ("ulysses", "ring", "cfg"):
            product *= int(degrees.get(key, 1))
        if product == world:
            names.append(name)
    return names


@dataclass(frozen=True)
class AutoDecision:
    topology: Topology
    sage: bool
    reason: str
    rule: AutoRule = field(repr=False, default=_FALLBACK_RULE)
    # The cfg row that matched the render's shape but lacked evidence for its
    # quant or band, so the walk went on to a later row; None when none did.
    skipped_cfg: AutoRule | None = field(repr=False, default=None)


def sampler_keeps_uncond_at_cfg1(name: object) -> bool:
    """Whether this sampler keeps the unconditional pass at CFG 1.0.

    ComfyUI drops that pass at CFG 1.0 unless ``disable_cfg1_optimization`` is
    set (comfy/samplers.py). In ComfyUI e8665ca8 every sampler that sets it
    is a cfg++ sampler named with a ``cfg_pp`` suffix
    (comfy/k_diffusion/sampling.py). The seam canary checks that the shortcut
    reads the flag and that every suffixed sampler sets it; the other way round
    it probes only plain euler. A sampler that set the flag without the suffix
    would cost speed, not fidelity: auto would skip a cfg row it could take.
    Kept, cond and uncond fold into one call of batch 2, which a cfg topology
    splits; adapters/cfg_parallel.py states the same rule in its batch-1
    refusal. An unknown name answers False, so auto keeps skipping cfg rows at
    CFG 1.0 when it cannot tell.
    """
    return isinstance(name, str) and name.endswith("cfg_pp")


def choose_auto_topology(
    family: str,
    quant_kind: str,
    megapixels: float,
    world: int,
    cfg_value: float | None = None,
    fits_resident: bool = True,
    batch_size: int = 1,
    keeps_uncond: bool = False,
    variant: str | None = None,
) -> AutoDecision:
    """Resolve ``auto`` and log the selected row and rationale.

    Auto skips a CFG-parallel row for a CFG-1 render unless ``keeps_uncond``
    says the sampler keeps the unconditional pass there
    (``sampler_keeps_uncond_at_cfg1``), and skips any cfg row with no passing
    evidence (topology_cfg_evidence), including a row whose record excludes the
    checkpoint ``variant`` (``checkpoint_variant``). ``fits_resident=False``
    adds FSDP for capacity. Spare ranks use data parallel only when the batch
    divides evenly; otherwise a pure sequence-parallel row folds them in, and
    a cfg row raises UnsupportedModelError.
    """
    if world <= 1:
        topo = Topology(world=max(world, 1))
        return AutoDecision(topo, False, "auto: single; world size 1", _FALLBACK_RULE)

    chosen: AutoRule | None = None
    skipped_cfg: AutoRule | None = None
    for rule in AUTO_TABLE:
        if rule.family != family:
            continue
        if rule.quants and quant_kind not in rule.quants:
            continue
        if not (rule.mp_min <= megapixels < rule.mp_max):
            continue
        if world < rule.world_min:
            continue
        if (rule.topology.get("cfg", 1) != 1 and cfg_value is not None
                and abs(cfg_value - 1.0) < 1e-6 and not keeps_uncond):
            continue  # cfg-parallel has no uncond pass to parallelize at cfg 1.0
        if rule.topology.get("cfg", 1) != 1 and not cfg_row_is_eligible(rule, quant_kind, variant):
            skipped_cfg = rule
            continue
        chosen = rule
        break
    note = None
    if chosen is None:
        chosen = _FALLBACK_RULE
        # Detect whether CFG-1 alone excluded an otherwise matching row. A
        # sampler that keeps the uncond pass excluded none that way.
        cfg_skipped = not keeps_uncond and any(
            rule.family == family
            and (not rule.quants or quant_kind in rule.quants)
            and rule.mp_min <= megapixels < rule.mp_max
            and world >= rule.world_min
            and rule.topology.get("cfg", 1) != 1
            for rule in AUTO_TABLE
        )
        if skipped_cfg is not None:
            record = cfg_evidence_for(skipped_cfg)
            if record is None:
                detail = "has no exact-row evidence record"
            elif variant is not None and variant in record.excludes:
                detail = f"was measured on this {variant} checkpoint and does not grant it"
            elif record.quants and quant_kind not in record.quants:
                detail = f"has no passing {quant_kind} result"
            else:
                detail = "has no passing in-band speed-and-fidelity result"
            note = (f"cfg2 table row {skipped_cfg.row} {detail}; uly2 is the "
                    "conservative supported fallback")
        elif cfg_skipped:
            note = f"{family} at cfg 1.0 skips cfg-parallel rows; uly2 is the USP default"
        elif any(rule.family == family for rule in AUTO_TABLE):
            # Known family with no matching quant, resolution, or world row.
            note = (f"no {family} table row matches this quant/megapixels/world shape; "
                    f"uly2 is the conservative default")
        elif family == "unknown":
            # No signature matched, so every family-keyed preflight fails its
            # equality test and stays disarmed. The note says so rather than
            # leave it to the fallback row's generic text.
            note = ("no signature matched this checkpoint header, so no table row "
                    "and no family preflight applies; uly2 is the conservative "
                    "default. Name the architecture in the Init node's "
                    "family_adapter widget to restore both (docs/MODELS.md)")
        else:
            # Any other family with no AUTO_TABLE row lands here, such as
            # mage_flow or qwen_image21, and so does any unregistered name. The
            # note names the family: the "unknown" note says no family preflight
            # applies, but a registered family still arms its preflights.
            note = (f"family {family!r} has no auto table row, so the generic "
                    "fallback applies; uly2 is the conservative default")
    if note is None:
        note = chosen.note
        excluded = cfg_evidence_for(skipped_cfg) if skipped_cfg is not None else None
        if excluded is not None and variant in excluded.excludes:
            note = (f"{note}; cfg2 table row {excluded.row} was measured on this "
                    f"{variant} checkpoint and does not grant it")

    topo_kwargs = dict(chosen.topology)
    if not fits_resident:
        topo_kwargs["fsdp"] = True
    topo = Topology(world=world, **topo_kwargs)  # type: ignore[arg-type]

    if world % topo.model_parallel != 0:
        # Odd worlds cannot tile degree-2 rows. Pure Ring avoids head constraints
        # and keeps data parallel at one for batch-1 renders.
        topo = Topology(world=world, ring=world, fsdp=topo.fsdp)
        note = f"{note}; world {world} does not tile the row's degrees; pure ring{world}"

    topo = topo.with_derived_dp()
    if topo.dp > 1 and batch_size % topo.dp != 0:
        # Fold unusable DP ranks into sequence parallelism. Prefer Ulysses when
        # every head count tiles the validated target because Ring cannot exclude
        # divisibility pads; otherwise Ring is the head-count-free fallback.
        target_uly = topo.ulysses * topo.dp
        heads = VALIDATED_ULYSSES_FOLDS.get(family, {}).get(target_uly, ())
        # Grow only a pure-Ulysses row. Folding a CFG row would create an
        # unvalidated Ulysses-plus-CFG composite.
        if (topo.ring == 1 and topo.cfg == 1 and heads
                and all(h % target_uly == 0 for h in heads)):
            topo = replace(topo, ulysses=target_uly, dp=1)
            note = (f"{note}; batch {batch_size} cannot split dp; folded into "
                    f"ulysses{target_uly} (heads tile; keeps exact pad exclusion)")
        elif topo.cfg > 1:
            # A ring fold fails the same way a Ulysses fold does: either builds
            # a cfg-plus-sequence composite no row names, so auto refuses rather
            # than resolve a shape nothing measured. World 2 never reaches
            # this: a cfg2 row leaves no spare rank there.
            from .adapters.base import UnsupportedModelError

            choices = ", ".join(_presets_tiling(world))
            clause = (
                f"pick an explicit preset that tiles world {world}: {choices}"
                if choices else
                f"no shipped preset tiles world {world} exactly, so render this "
                "at a world size one of them tiles")
            # DESIGN 5.9: a waiver is an expert choice about a typed topology,
            # so this auto refusal is class K with no guard.
            owned = (
                "That list names what tiles the world, not what auto would "
                "grant: a typed preset is the operator's choice, and one of "
                "them can be the composite auto just declined to derive. "
                if choices else "")
            raise UnsupportedModelError(refusal(
                RefusalClass.KNOWN_WRONG,
                f"auto matched table row {chosen.row} for {family}, which is "
                f"cfg{topo.cfg}, and batch {batch_size} cannot split the "
                f"{topo.dp} data-parallel rank(s) world {world} leaves over. "
                "Folding them into ulysses or ring would build a "
                "cfg-plus-sequence topology no auto table row names and no "
                f"fidelity matrix has measured. Either {clause}, or send a "
                f"batch size that is a multiple of {topo.dp}. {owned}There is "
                "no waiver for this refusal: auto resolves measured rows only.",
            ))
        else:
            topo = replace(topo, ring=topo.ring * topo.dp, dp=1)
            note = f"{note}; batch {batch_size} cannot split dp; folded into ring{topo.ring}"
    topo.validate()

    reason = (
        f"auto: {topo.describe()}; {family} {quant_kind or 'bf16'} at {megapixels:.2f} MP, "
        f"world {world}, table row {chosen.row} ({note})"
    )
    if topo.cfg > 1:
        if keeps_uncond and cfg_value is not None and abs(cfg_value - 1.0) < 1e-6:
            reason = f"{reason}; the cfg++ sampler keeps the uncond pass at cfg 1.0"
        # The row picked cfg-parallel, so the line names the evidence record
        # that admitted it and that record's scope.
        record = cfg_evidence_for(chosen)
        if record is not None:
            reason = f"{reason}; {record.clause()}"
    log.info(reason)
    return AutoDecision(topo, chosen.sage, reason, chosen, skipped_cfg)
