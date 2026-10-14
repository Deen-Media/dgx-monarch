"""The auto-mode decision table (DESIGN.md section 5.4), as ordered data.

``topology.choose_auto_topology`` walks ``AUTO_TABLE`` first match and owns
the decision; this module holds only the rows and their shape. A cfg-parallel
row also needs an exact-row evidence record (``topology_cfg_evidence``).
"""
from __future__ import annotations

from dataclasses import dataclass


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

# Seeded from dated two-Spark measurements in docs/VALIDATION.md. Each row's
# ``note`` records its evidence or pending benchmark status; docs/MODELS.md owns
# per-family operator guidance. A cfg-parallel row also needs an exact-row
# evidence record that admits its quantization (topology_cfg_evidence).
AUTO_TABLE: tuple[AutoRule, ...] = (
    # --- krea2 (wan-family DiT, image) -----------------------------------
    AutoRule(15, "krea2", (), 1.2, 2.36, 2, {"cfg": 2}, False,
             "cfg2 from 1.2 to 2.36 MP on FP8: Campaign K (2026-10-14, ComfyUI e8665ca8, "
             "head capped at 2100 MHz) read Turbo fp8_scaled bit-identical to one GPU and "
             "1.21x to 1.29x faster than uly2 with SAGE_AUTO at 1152 to 1536 square, at "
             "CFG 1 under euler_cfg_pp and at CFG 4.5 under euler; RAW BF16 and INT8 were "
             "measured at 1536 only and take row 12"),
    AutoRule(11, "krea2", ("fp8",), 1.2, _INF, 2, {"ulysses": 2}, True,
             "USP wins >= 1.5 MP; sage speeds up fp8 USP at high res (about 5-8% at 1536 in the "
             "2026-06 campaign, fp8 only; int8 was never benchmarked with sage); 2026-10-06 Turbo "
             "fp8 uly2 on TORCH_FLASH bit-identical to dp2 and one GPU at 1536 "
             "(was 1-step NRMS 0.035 that day before the fix); Campaign K (2026-10-14, "
             "head capped at 2100 MHz) read this sage route on Turbo fp8 at 1152 to 1536 "
             "square at 1-step NRMS 0.030 to 0.058 against one GPU and 1 to 4 percent "
             "faster than TORCH_FLASH uly2; row 15 takes cfg2 below 2.36 MP"),
    AutoRule(12, "krea2", (), 1.2, _INF, 2, {"ulysses": 2}, False,
             "USP wins >= 1.5 MP; sage is slower on bf16; 2026-10-06 RAW BF16 "
             "and INT8 uly2 bit-identical to dp2 and one GPU (BF16 at 1536 and "
             "1448, INT8 at 1536; was 1-step NRMS 0.010 BF16 and 0.016 INT8 that "
             "day before the fix)"),
    # --- chroma (flux-family DiT) -----------------------------------------
    AutoRule(20, "chroma", (), 0.0, 1.2, 2, {"cfg": 2}, False,
             "cfg2 below 1.2 MP; on the shipped graph at 1024 square on 2026-10-01 it read "
             "bit-identical to one GPU and ran faster than uly2 on bf16, fp8 and int8"),
    AutoRule(19, "chroma", (), 1.2, 1.64, 2, {"cfg": 2}, False,
             "cfg2 from 1.2 to 1.64 MP on BF16 and FP8: Campaign K (2026-10-14, ComfyUI "
             "e8665ca8, head capped at 2100 MHz) read Chroma1-HD BF16 and FP8-mixed "
             "bit-identical to one GPU at 1152, 1216 and 1280 square, 1.30x to 1.31x "
             "faster than uly2 on BF16 and 1.26x to 1.30x faster than uly2 with SAGE_AUTO "
             "on FP8; INT8 was not measured in this band and takes row 22"),
    AutoRule(21, "chroma", ("fp8",), 1.2, _INF, 2, {"ulysses": 2}, True,
             "uly2 >= 1.5 MP; sage on for fp8 USP at high res; Campaign K (2026-10-14, "
             "head capped at 2100 MHz) read this sage route on FP8-mixed at 1152 to 1280 "
             "square at 1-step NRMS 0.057 to 0.078 against one GPU and 3 to 5 percent "
             "faster than TORCH_FLASH uly2; row 19 takes cfg2 below 1.64 MP"),
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
    AutoRule(68, "qwen_image", (), 1.2, 1.77, 2, {"cfg": 2}, False,
             "cfg2 from 1.2 to 1.77 MP on FP8: Campaign K (2026-10-14, ComfyUI e8665ca8, "
             "head capped at 2100 MHz) read base 2512 FP8 bit-identical to one GPU and "
             "1.58x to 1.66x faster than uly2 at 1152, 1216, 1280 and 1328 square; "
             "Edit-2511 was not measured"),
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
    AutoRule(79, "boogu", (), 0.0, 1.2, 2, {"cfg": 2}, False,
             "cfg2 below 1.2 MP on FP8: Campaign K (2026-10-14, ComfyUI e8665ca8, head "
             "capped at 2100 MHz) read the shipped unequal prompt pair, on the per-cond "
             "dispatch path, bit-identical to one GPU and 1.40x faster than uly2 at 1024 "
             "square"),
    AutoRule(78, "boogu", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "28 heads tile uly2, USP default; about 20GB bf16 resident, *+fsdp for headroom; "
             "row 79 takes cfg2 below 1.2 MP on FP8; elsewhere cfg2 is an explicit preset, "
             "and since 2026-09-05 it takes any prompt pair: "
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
    AutoRule(47, "anima", (), 0.0, 1.2, 2, {"cfg": 2}, False,
             "cfg2 below 1.2 MP on BF16: Campaign K (2026-10-14, dgx-monarch 65d14333, "
             "ComfyUI e8665ca8, head capped at 2100 MHz) read the shipped unequal prompt "
             "pair, on the per-cond dispatch path, bit-identical to one GPU and 1.27x "
             "faster than uly2 at 1024 square"),
    AutoRule(48, "anima", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "about 2B cosmos-MiniTrainDIT, uly2 default (16 heads d128); row 47 takes "
             "cfg2 below 1.2 MP on BF16; cfg_cond_padding none: comfy's 512 pad lands "
             "inside the forward, after the batching decision, so an unequal prompt pair "
             "splits the model call at any length and cfg2 takes the per-cond dispatch "
             "(2026-09-02)"),
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
             "Campaign K (2026-10-14) read explicit image-model cfg2 at 1024 square "
             "bit-identical to one GPU and 1.22x faster than uly2, but Image, Video Lite "
             "and Video Pro share this family and kind and a video frame is under 1.2 MP, "
             "so there is no cfg2 row"),
    # --- ernie (single-stream image DiT) -----------------------------------
    AutoRule(60, "ernie", (), 0.0, _INF, 2, {"ulysses": 2}, False,
             "uly2 default (32 heads / head_dim 128); cfg2 with shape-equal prompts "
             "proved exact on HW (1-step NRMS 0.000, 2026-07-13 matrix); "
             "2026-10-06 BF16 uly2 bit-identical to dp2 and one GPU at 1024 and "
             "1040 (was 1-step NRMS 0.016 at 1040 that day before the fix)"),
)

_FALLBACK_RULE = AutoRule(99, "*", (), 0.0, _INF, 2, {"ulysses": 2}, False,
                          "unknown family: uly2 is the safe default on 2+ ranks")
