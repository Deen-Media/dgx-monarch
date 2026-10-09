"""Evidence-backed MiniMax H3 activation constants.

Calibration data stays separate from guard logic so a recalibration is
reviewable on its own. This module has no project or third-party imports.
"""
from __future__ import annotations

# Checkpoint architecture.
H3_HIDDEN_DIM = 5376
_H3_BF16_BYTES = 2

# Calibration posture. Config B is 1344x768x124, or 37,730 packed rows:
#
# * One-rank BF16 must refuse after two kernel kills on 2026-08-05.
# * World-2 uly2 BF16 must pass; measured completion was 727.6 s.
# * One-rank int8-convrot must pass; the pruned checkpoint completed at
#   50.74 s/step (2026-08-03).
#
# See docs/VALIDATION.md for the hardware records. Let S be sharded bytes per
# per-rank row, R replicated bytes per total row, U usable bytes (MemAvailable
# minus the driver reserve, 4 GiB at this calibration on 2026-08-05), and W the
# weight charge. The 114.7 GiB idle and the 28.4 GiB post-render floor below
# come from one stock single-process reference run of the 61.7 GiB bf16
# artifact; docs/VALIDATION.md records both (2026-08-05).
#
# RESIDENT: DiT 61.7 + text encoder 16.7 + VAEs 5.8 = 84.2 GiB.
# MemAvailable is about 32.4 GiB at dispatch: that run's 28.4 GiB post-render
# floor at Config A plus about 4 GiB of activations and runtime. U=28.4.
#       refuse at N=1:  37,730*(S + R) > U            ->  S >   734,495
#       pass   at N=2:  18,865*S + 37,730*R <= U      ->  S <= 1,468,990
#
# PENDING (`auto`): resident text encoder 16.7 + VAEs 5.8 GiB gives about
# 92.2 GiB available from that run's 114.7 GiB idle, U=88.2 GiB, and
# W=66,280,487,368 B. That run's own reconciliation puts the encoder at
# 14.6 GiB; the tighter 16.7 GiB figure, from another run, is used instead.
# Recomputing with 94.3 GiB gives (739,376, 1,468,991], centre 1,042,178; the
# shipped value is 0.99x it.
#       refuse at N=1:  W + 37,730*(S + R) > U        ->  S >   679,612
#       pass   at N=2:  W + 18,865*S + 37,730*R <= U  ->  S <= 1,359,225
#
# SINGLE-RANK INT8: resident, U=56.5 GiB, W=0. U charges the 31.70 GiB
# non-pruned int8-convrot file, but the completion ran the 19.53 GiB pruned one
# (docs/VALIDATION.md, H3 promotion Leg 1). The larger charge only lowers this
# ceiling, and the PENDING ceiling sits below it.
#       pass   at N=1:  37,730*(S + R) <= U           ->  S <= 1,534,181
#
# Intersection: (734,495, 1,359,225], geometric centre 999,172. The selected
# S=96*5376*2=1,032,192 B (0.98 MiB/row), 1.41x above the floor and 1.32x below
# the ceiling. The reserve is 5 GiB since 2026-09-04; at the same readings
# that moves the intersection to (706,036, 1,302,308], which still holds S.
#
# VAEs 5.8 is the two files' size in decimal GB. They hold 5,813,063,304 B,
# 5.41 GiB (docs/VALIDATION.md, staged artifact sizes), so each U that subtracts
# them is 0.39 GiB low. With the exact bytes the PENDING ceiling is 1,381,204
# (1,324,287 at the 5 GiB reserve). The intersection's floor comes from
# RESIDENT, whose U is the measured 28.4 GiB with no VAE term, so S holds
# either way.
#
# Sanity bounds: an autograd-free block can concurrently hold x, h, two MLP
# projections, and the gate: 11*hidden*2=118,272 B/row. The selected value is
# 8.7x that floor. The gap covers fp32 patch staging, attention workspaces,
# retained allocator memory, and block prefetch (645.5 M parameters,
# 1.20 GiB BF16 per block).
#
# The kill left 24.9 GiB unattributed at 37,730 rows: 708,618 B/row combined,
# or 634,890 B/row sharded. The selected value is 1.63x that lower bound. Using
# the measured slope directly would estimate 24.9 GiB against the resident
# 28.4 GiB budget and incorrectly pass the failing case. The generic
# mesh_safety factor of 8 is also below H3's anatomy floor.
#
# This slope has no fixed intercept. Prefetch, allocator, and workspace costs
# are amortized into it. That run's working set left about 4 GiB for
# activations and runtime at Config A's 356 rows, where this formula estimates
# 0.37 GiB. It is therefore a lower bound below roughly 3,900 rows; above that,
# the 1.63x slope margin covers the missing floor. All calibration requirements
# above use 37,730 rows.
_H3_BLOCK_FACTOR = 96
H3_SHARDED_ROW_BYTES = _H3_BLOCK_FACTOR * H3_HIDDEN_DIM * _H3_BF16_BYTES

# Replicated buffers are charged per total packed row because SP does not divide
# them: fp32 patchify/projections, full-stream assembly, gather output, and
# final-layer fp32 slices. Per-row terms are 10,752 assembly + 10,752 gather +
# 21,504 final slices + 32,256 patch staging/BF16 copy = 75,264 B. The last term
# is 5376*(4+2), matching 765 MiB fp32 plus 383 MiB BF16 over 37,296 video rows.
#
# Round down to 72 KiB. Safety comes from overcharging final slices across all
# rows and charging a gather buffer even at world 1; those exceed the 1,536 B
# reduction. Charging every total row biases toward refusal; the OOM it guards
# cannot be caught. Config B charges 2.59 GiB versus about 2.64 GiB measured.
H3_REPLICATED_ROW_BYTES = 73_728

# Guide encode, measured 2026-08-14 on one DGX Spark (121.6 GiB, 113 GiB idle).
# The probe method is docs/VALIDATION.md, "The guide encode, and what it excludes
# (2026-08-14, one DGX Spark)".
#
# The values below are the drop during the encode, with the 4.85 GiB VAE
# already resident, so they do not restate the VAE charge.
#
#   canvas      frames   cost        result
#   1344x768    1        13.6 GiB    completed, latent [1,24,1,48,84]
#   448x256     5        16.2 GiB    completed, latent [1,24,2,16,28]
#   672x384     5        66.9 GiB    completed, latent [1,24,2,24,42]
#   1344x768    5        >94 GiB     never completed, watchdog killed it
#   448x256     22       33.4 GiB    completed, latent [1,24,7,16,28]
#   448x256     39       47.5 GiB    completed, latent [1,24,12,16,28]
#
# Cost is superlinear in canvas area: 2.25x the pixels from 448x256 to 672x384
# costs 4.1x the memory. Three points and one censored bound do not support a
# curve, so pricing rounds an area up to the next measured point and charges
# what that point measured, rounded up to whole GiB. Every charge is therefore
# a number a run produced, and above the largest entry the price saturates
# rather than print an extrapolation as evidence. The top entry is a lower
# bound: that encode never finished. So is the saturation, on a host that can
# encode past it. Recalibrate both where they complete.
#
# The frame dimension was swept at one canvas, 448x256, the only geometry where
# a long clip completes on this hardware: 16.2, 33.4, 47.5 GiB at 5, 22 and 39
# frames. That is about a gigabyte per frame, so length matters as much as a
# resolution step and cannot be left at the five-frame floor. Where that sweep
# applies, an undeclared clip is priced at the longest measured length, since
# an unread link can carry any of them. Longer clips than 39 frames, and the
# frame dimension at larger canvases, stay unmeasured and therefore
# under-counted, the same direction every unread reference block takes; at
# those canvases the five-frame charge already refuses this hardware.
_GIB = 2 ** 30
H3_GUIDE_CLIP_ENCODE_BYTES: tuple[tuple[int, int], ...] = (
    (448 * 256, 17 * _GIB),
    (672 * 384, 67 * _GIB),
    (1344 * 768, 94 * _GIB),
)

# One still at the largest canvas measured. A still is a cheap degenerate path:
# one latent frame, no temporal window. Only one geometry was measured, so a
# larger canvas holds this per-pixel rate by assumption, which convexity makes
# a lower bound rather than a guess.
H3_GUIDE_STILL_ENCODE_BYTES = 14 * _GIB
H3_GUIDE_STILL_MEASURED_PIXELS = 1344 * 768

# Measured at 448x256 on 2026-08-14, encode alone, VAE already resident. Used
# only at or below that area, where the sweep was run.
H3_GUIDE_CLIP_LENGTH_BYTES: tuple[tuple[int, int], ...] = (
    (5, 17 * _GIB),
    (22, 34 * _GIB),
    (39, 48 * _GIB),
)
H3_GUIDE_LENGTH_MEASURED_PIXELS = 448 * 256

# Under five frames Add Guide anchors the first image alone
# (comfy_extras/nodes_minimax_h3.py, MiniMaxH3AddGuide.execute), so a declared
# batch of one to four costs one still.
H3_GUIDE_CLIP_FLOOR_FRAMES = 5
