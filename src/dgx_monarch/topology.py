"""Topology model and benchmark-backed auto-mode table (DESIGN.md §5.4).

Rules are ordered data keyed by model family, quantization, megapixels, and
world size. Each decision logs its matching row and rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .log import get_logger
from .refusal import RefusalClass, refusal
from .topology_cfg_evidence import cfg_evidence_for, cfg_row_is_eligible

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


@dataclass(frozen=True)
class AutoRule:
    """One first-match row of the auto-mode decision table."""
    row: int
    family: str
    quants: tuple[str, ...]
    mp_min: float
    mp_max: float
    world_min: int
    topology: dict[str, int | bool]
    sage: bool
    note: str


_INF = float("inf")

# Family -> {target Ulysses degree: every sharded query and KV head count that
# degree must divide}. Empty: the evidence validates the degree-2 rows but no
# fold of data-parallel ranks into a larger Ulysses degree. Add one entry per
# family and degree a campaign validates.
VALIDATED_ULYSSES_FOLDS: dict[str, dict[int, tuple[int, ...]]] = {}

# Seeded from dated two-Spark measurements in docs/VALIDATION.md. Each row's
# ``note`` records its evidence or pending benchmark status; docs/MODELS.md owns
# per-family operator guidance. A cfg-parallel row also needs an exact-row
# evidence record that admits its quantization (topology_cfg_evidence).
AUTO_TABLE: tuple[AutoRule, ...] = (
    # --- krea2 (wan-family DiT, image) -----------------------------------
    AutoRule(11, "krea2", ("fp8",), 1.2, _INF, 2, {"ulysses": 2}, True,
             "USP wins >= 1.5 MP; sage speeds up fp8 USP at high res (about 5-8% at 1536 in the "
             "2026-06 campaign, fp8 only; int8 was never benchmarked with sage); 2026-10-06 Turbo "
             "fp8 uly2 on TORCH_FLASH bit-identical to dp2 and one GPU at 1536 "
             "(was 1-step NRMS 0.035 that day before the fix); the sage route "
             "this row picks had no reading"),
    AutoRule(12, "krea2", (), 1.2, _INF, 2, {"ulysses": 2}, False,
             "USP wins >= 1.5 MP; sage is slower on bf16; 2026-10-06 RAW BF16 "
             "and INT8 uly2 bit-identical to dp2 and one GPU (BF16 at 1536 and "
             "1448, INT8 at 1536; was 1-step NRMS 0.010 BF16 and 0.016 INT8 that "
             "day before the fix)"),
    # --- chroma (flux-family DiT) -----------------------------------------
    AutoRule(20, "chroma", (), 0.0, 1.2, 2, {"cfg": 2}, False,
             "cfg2 below 1.2 MP; on the shipped graph at 1024 square on 2026-10-01 it read "
             "bit-identical to one GPU and ran faster than uly2 on bf16, fp8 and int8"),
    AutoRule(21, "chroma", ("fp8",), 1.2, _INF, 2, {"ulysses": 2}, True,
             "uly2 >= 1.5 MP; sage on for fp8 USP at high res"),
    AutoRule(22, "chroma", (), 1.2, _INF, 2, {"ulysses": 2}, False,
             "uly2 >= 1.5 MP; sage off on bf16"),
    # --- flux family (12B double/single DiT; rows seeded from chroma 20-22,
    # same architecture; scoped hardware evidence recorded per row) ----------
    AutoRule(24, "flux", ("fp8",), 1.2, _INF, 2, {"ulysses": 2}, True,
             "seeded from chroma row 21; sage on for fp8+USP; this FP8 row is "
             "outside the 2026-08-11 standard Flux 1 Dev checkpoint scope"),
    AutoRule(25, "flux", (), 1.2, _INF, 2, {"ulysses": 2}, False,
             "seeded from chroma row 22; 2026-10-06 standard Flux 1 Dev uly2 "
             "bit-identical to dp2 and to one GPU (was 1-step NRMS 0.008 on "
             "2026-08-11); Schnell remains unclaimed"),
    AutoRule(27, "flux2", (), 1.2, _INF, 2, {"ulysses": 2}, False,
             "48 heads tile uly2; about 32B, so use a *+fsdp preset when it does not fit "
             "resident; 2026-10-06 fp8mixed uly2 bit-identical to dp2 at 1024 and 1040"),
    AutoRule(28, "longcat", (), 0.0, 1.2, 2, {"cfg": 2}, False,
             "seeded from chroma row 20; 2026-08-11 LongCat BF16 cfg2 matched "
             "DP2 at 1-step NRMS 0.009, and the 2026-09 sweep passed it at 2.00x "
             "the one-GPU speed at 1.05 MP, inside this row's band"),
    AutoRule(29, "longcat", (), 1.2, _INF, 2, {"ulysses": 2}, False,
             "seeded from chroma row 22; 2026-10-06 LongCat BF16 uly2 "
             "bit-identical to dp2 and to one GPU (was 1-step NRMS 0.097 on "
             "2026-08-11)"),
    # --- qwen_image (dual-stream 60-block MMDiT, ~20B; Chroma recipe) ------
    AutoRule(71, "qwen_image", ("fp8",), 1.2, _INF, 2, {"ulysses": 2}, False,
             "sage off since 2026-10-05: on the shipped 1328x1328 base 2512 FP8 "
             "graph, auto with SAGE_AUTO read 1-step NRMS 0.162 against one GPU, "
             "over the 0.10 floor (cell f43477b46788), while uly2 read 0.082 on "
             "both TORCH_FLASH and TORCH_CUDNN (c2b9258ffce7, b3bc8bf2ea6b); "
             "recorded 2026-09-14 to 2026-09-23, ComfyUI 3216c62e, head capped at 2100 MHz; "
             "2026-10-06 base 2512 FP8 uly2 bit-identical to dp2 and one GPU "
             "at 1328 and 1024 (was 1-step NRMS 0.048 that day before the fix)"),
    AutoRule(72, "qwen_image", (), 1.2, _INF, 2, {"ulysses": 2}, False,
             "seeded from chroma row 22; about 20B, so use a *+fsdp preset when it "
             "does not fit resident; non-FP8 and Edit-2511 scopes remain unclaimed"),
    # --- lens (~4B dual-stream MMDiT; real CFG; bf16-only) -----------------
    AutoRule(74, "lens", ("fp8",), 1.2, _INF, 2, {"ulysses": 2}, False,
             "24 heads of d64 tile both uly2 and ring2; sage off since 2026-10-05: on "
             "the shipped 1344x1344 MXFP8 graph uly2 read 1-step NRMS 0.026 "
             "on TORCH_FLASH (9d51070285bf, warm 20.0 s), 0.033 on TORCH_CUDNN "
             "(3b2b53c02706, 20.1 s) and 0.037 on SAGE_AUTO (911b65bc6ef2, "
             "20.3 s), so sage bought no speed; head capped at 2100 MHz"),
    AutoRule(75, "lens", (), 1.2, _INF, 2, {"ulysses": 2}, False,
             "uly2 >= 1.2 MP; bf16 only (fp16 NaNs); after the 2026-07-28 USP pad-row "
             "fix, that day's hardware re-gate matched DP2 at NRMS 0.060-0.061; "
             "2026-10-06 bf16 uly2 bit-identical to one GPU on the template"),
    # --- zimage (NextDiT single-stream: latent + pixel-space, ~6B) ---------
    AutoRule(77, "zimage", (), 1.2, _INF, 2, {"ulysses": 2}, False,
             "uly2 >= 1.2 MP; 30 heads tile uly2 but not uly4; the 2026-08-11 "
             "latent comparison matched DP2 at 1-step NRMS 0.000"),
    # --- boogu (dual-stream image DiT, ~10B, 4 attention stages) -----------
    AutoRule(78, "boogu", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "28 heads tile uly2, USP default; about 20GB bf16 resident, *+fsdp for headroom; "
             "cfg2 is an explicit preset, and since 2026-09-05 it takes any prompt pair: "
             "unequal num_tokens dispatch one cond per rank instead of folding; "
             "2026-10-06 fp8 uly2 bit-identical to dp2 (was 1-step NRMS 0.012)"),
    # --- pixeldit_comfy (two-stage pixel-space DiT: PixelDiT T2I + PiD) ----
    AutoRule(31, "pixeldit_comfy", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "world-2 exact-gather attention matched dp2 at NRMS 0.000 over "
             "the full PixelDiT 1024 and PiD 4096 schedules (2026-08-11); "
             "never cfg2, which read 1-step NRMS 0.206 against the 0.10 floor on 2026-08-21"),
    # --- ideogram4 (PixelDiT, head_dim 256) -------------------------------
    AutoRule(30, "ideogram4", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "2026-10-06 fp8 conditional model uly2 at CFG 1 bit-identical to "
             "dp2 and one GPU at 1024 and 1040 (was 1-step NRMS 0.088 at 1040 "
             "that day before the fix), and the shipped dual-model graph read bit-identical "
             "to one GPU on bf16, fp8 and int8; the 2026-08-11 fp8 hardware comparison "
             "matched DP2 at 1-step NRMS 0.000 at 1024 and covers the tested fp8 pair only; "
             "on 2026-08-21 the int8-convrot, mxfp8 and bf16 pairs passed and nvfp4 failed; "
             "never cfg2 (dual-model CFG runs a separate uncond model)"),
    # --- ltx (video DiT) ---------------------------------------------------
    AutoRule(40, "ltx", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "video seq len: USP always; use a *+fsdp preset when the model does not fit resident"),
    # --- wan (video DiT, incl. Wan 2.2 MoE) --------------------------------
    AutoRule(50, "wan", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "video seq len: USP always; use *+fsdp presets for capacity (14B on UMA)"),
    # --- wan variants (SCAIL/SCAIL2 animation, WanDancer audio; 14B) -------
    AutoRule(51, "wan_scail", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "SCAIL/SCAIL-2 14B animation DiT; USP always (video seq len); *+fsdp for capacity"),
    AutoRule(52, "wan_dancer", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "WanDancer 14B audio DiT; USP always; *+fsdp for capacity"),
    AutoRule(54, "wan_animate2", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "Wan-Animate2 14B driving-video DiT; spatial-within-frame exact gather; *+fsdp for capacity"),
    # --- minimax_h3 (packed AV DiT: video + audio in one token sequence) ---
    # One full-range row: no measurement splits H3 by megapixels. Auto reads H3
    # at its per-frame size, comfy's 16x ratio times the NestedTensor's video
    # grid (latent_scale; docs/VALIDATION.md "Auto megapixels follow the model's
    # latent ratio", 2026-10-05).
    AutoRule(53, "minimax_h3", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "packed video+audio single sequence: USP always (56 heads tile "
             "uly2). No cfg2 row: stock cannot concatenate two H3 conditionings "
             "into one batched call, and the DiT caps batch at 1; cfg2 and dp "
             "both typed-refuse on the driver before dispatch. Ring only when "
             "the packed total needs no divisibility pad: pad-row exclusion is "
             "ulysses-only, so the odd-world fallback and the world-4 fold "
             "refuse a padded sequence instead of rendering it"),
    # --- omnigen2 (Lumina2-style single-stack; 21 q / 7 kv heads) ----------
    AutoRule(80, "omnigen2", (), 0.0, _INF, 2, {"ring": 2}, False,
             "21 q and 7 kv heads do not divide by 2, so ulysses cannot split them; ring is "
             "the only SP topology. Refiners local; cfg2 stays an explicit preset, since 2026-09-05 "
             "on the per-cond dispatch rather than on a batch the num_tokens constant blocks"),
    # --- anima (cosmos MiniTrainDIT ~2B image DiT, H-axis shard) -----------
    AutoRule(48, "anima", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "about 2B cosmos-MiniTrainDIT, uly2 default (16 heads d128); no cfg2 row "
             "seeded and cfg_cond_padding none: comfy's 512 pad lands inside the "
             "forward, after the batching decision, so an unequal prompt pair "
             "splits the model call at any length (2026-09-02)"),
    # --- hunyuan (flux double/single DiT: image 2.1 + video 1.5/SR, one family)
    AutoRule(46, "hunyuan", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "USP default for both the image (2.1) and video (1.5/SR) checkpoints "
             "of this family; per-frame megapixels cannot separate them, so no "
             "cfg2 row. Image cfg2 is an explicit preset (cfg_cond_padding pad+mask); "
             "even head counts tile uly2; uly2 HW-vouched for Image 2.1, refiner, "
             "and Video 1.5 on 2026-07-15; split-custom SR stays explicit-topology; "
             "2026-10-06 Image 2.1 FP8 and Video 1.5 FP16 uly2 bit-identical to "
             "dp2 and one GPU (was 1-step NRMS 0.014 and 0.056 that day before "
             "the fix)"),
    # --- cogvideo (joint-attention video DiT) ------------------------------
    AutoRule(42, "cogvideo", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "video seq len: USP always; 30 (2B) and 48 (5B) heads both tile uly2"),
    # --- kandinsky5 (Wan-pattern video+image DiT) --------------------------
    AutoRule(44, "kandinsky5", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "Wan-pattern DiT; uly2 HW-vouched for Image and Video Lite on "
             "2026-07-13; Video Pro uses explicit uly2+fsdp for capacity; "
             "image-model cfg2 crossover remains unmeasured"),
    # --- ernie (single-stream image DiT) -----------------------------------
    AutoRule(60, "ernie", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "uly2 default (32 heads / head_dim 128); cfg2 with shape-equal prompts "
             "proved exact on HW (1-step NRMS 0.000, 2026-07-13 matrix); "
             "2026-10-06 BF16 uly2 bit-identical to dp2 and one GPU at 1024 and "
             "1040 (was 1-step NRMS 0.016 at 1040 that day before the fix)"),
)

_FALLBACK_RULE = AutoRule(99, "*", (), 0.0, _INF, 2, {"ulysses": 2}, False,
                          "unknown family: uly2 is the safe default on 2+ ranks")


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


def choose_auto_topology(
    family: str,
    quant_kind: str,
    megapixels: float,
    world: int,
    cfg_value: float | None = None,
    fits_resident: bool = True,
    batch_size: int = 1,
) -> AutoDecision:
    """Resolve ``auto`` and log the selected row and rationale.

    Auto skips a CFG-parallel row for a CFG-1 render, and skips any cfg row
    with no passing evidence (topology_cfg_evidence). ``fits_resident=False``
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
        if rule.topology.get("cfg", 1) != 1 and cfg_value is not None and abs(cfg_value - 1.0) < 1e-6:
            continue  # cfg-parallel has no uncond pass to parallelize at cfg 1.0
        if rule.topology.get("cfg", 1) != 1 and not cfg_row_is_eligible(rule, quant_kind):
            skipped_cfg = rule
            continue
        chosen = rule
        break
    note = None
    if chosen is None:
        chosen = _FALLBACK_RULE
        # Detect whether CFG-1 alone excluded an otherwise matching row.
        cfg_skipped = any(
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
        f"auto: {topo.describe()}; {family} {quant_kind or 'bf16'} at {megapixels:.1f} MP, "
        f"world {world}, table row {chosen.row} ({note})"
    )
    if topo.cfg > 1:
        # The row picked cfg-parallel, so the line names the evidence record
        # that admitted it and that record's scope.
        record = cfg_evidence_for(chosen)
        if record is not None:
            reason = f"{reason}; {record.clause()}"
    log.info(reason)
    return AutoDecision(topo, chosen.sage, reason, chosen)
