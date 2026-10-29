"""Evidence-backed MiniMax H3 activation constants.

Calibration data stays separate from guard logic so a recalibration is
reviewable on its own. This module has no project or third-party imports.
"""
from __future__ import annotations

# Checkpoint architecture.
H3_HIDDEN_DIM = 5376
_H3_BF16_BYTES = 2

# Per rank the guard charges F + S*rows_per_rank + R*rows_total: a fixed floor
# F, S sharded bytes per per-rank row and R replicated bytes per total row
# (below). docs/VALIDATION.md, "H3 capacity", holds the hardware records.
#
# HISTORY. The 2026-08-05 fit had no floor. Config B is 1344x768x124, or
# 37,730 packed rows. Its one-rank BF16 render was killed twice on 2026-08-05,
# its world-2 uly2 BF16 render completed in 727.6 s, and the pruned INT8
# checkpoint completed it on one rank at 50.74 s/step (2026-08-03). At a 4 GiB
# reserve those bounded S to (734,495, 1,359,225], and S shipped at
# 96*5376*2 = 1,032,192 B, near that window's geometric centre. The kill left
# 24.9 GiB unattributed at 37,730 rows; the slope sat 1.63x above that reading
# and carried every fixed cost. The killed run still had the legacy load
# path's retained arena, since removed.
#
# The 2026-10-29 render broke that fit: 1344x768x362 on uly2 with the pruned
# INT8 ConvRot DiT resident, 109,463 rows (107,856 video, 1,206 audio, 401
# text), 54,732 per rank. Each actor's GPU use beyond its 19.53 GiB of weights
# peaked at 17.6 GiB, and the actor-only host's whole use at the peak was
# 46.4 GiB with the weights, about 6 GiB of page cache growth and 8.8 GiB of
# RSS. The old slope charged 60.1 GiB per rank there, about 3.4x the measured
# reservation, and refused a second render (51.5 GiB usable) and the gate's
# slab path (56.3 GiB usable). Both refusals were false.
#
# REQUIREMENTS, per rank. U is MemAvailable minus the reserve and W the weight
# charge. The reserve has been 5 GiB since 2026-09-04
# (driver_footprint.reserve_bytes); the 2026-08-05 fit used 4 GiB, and its
# readings are in brackets. The 114.7 GiB idle and the 28.4 GiB post-render
# floor come from one stock single-process reference run of the 61.7 GiB bf16
# artifact (docs/VALIDATION.md, 2026-08-05).
#
#   1. Config B, one rank, BF16: killed, must refuse.
#      RESIDENT: DiT 61.7 + text encoder 16.7 + VAEs 5.8 = 84.2 GiB, leaving
#      about 32.4 GiB at dispatch (the 28.4 GiB floor plus about 4 GiB of
#      activations and runtime at Config A): U = 27.4 [28.4].
#      PENDING (`auto`): only the encoder and the VAEs are in, 92.2 GiB of the
#      114.7 GiB idle: U = 87.2 [88.2], W = 66,280,487,368 B. That run's own
#      reconciliation puts the encoder at 14.6 GiB; the tighter 16.7 GiB
#      figure, from another run, is used instead.
#   2. Config B, world 2 uly2, BF16, 18,865 rows per rank: completed, must
#      pass in both postures.
#   3. Config B, one rank, INT8 under an explicit preset: completed, must pass.
#      U = 55.5 [56.5], W = 0. U charges the 31.70 GiB non-pruned file, but the
#      completion ran the 19.53 GiB pruned one (docs/VALIDATION.md, H3
#      promotion Leg 1); the larger charge only lowers this ceiling.
#   4. The 2026-10-29 render, weights resident: completed, must pass at
#      U = 51.5 (the second render: 56.5 GiB MemAvailable minus 5) and at
#      U = 56.3 (the gate's slab path), and must charge more than the measured
#      17.6 GiB.
#
# FIT. Three requirements bind: point 1 resident, point 2 pending and point 4
# at 51.5 GiB. Their budgets carry whole-GiB doubt (92.2 GiB is derived, the
# encoder reads 14.6 or 16.7 GiB, the VAEs 5.41 or 5.8), so the fit maximises
# the smallest margin in GiB. It takes each requirement at the reserve that
# makes it harder, point 1 at the bracketed 4 GiB budgets and every pass at
# 5 GiB, so the verdicts hold at either reserve. The optimum is F = 7.70 GiB
# and S = 631,756 B, every binding margin 4.09 GiB. Rounded to F = 7.5 GiB and
# S = 59*5376*2 = 634,368 B (0.60 MiB/row), the margins in GiB are:
#
#   point                            estimate  budget        margin
#   1 killed, resident                  32.38  27.4 [28.4]   refuses by 4.98 [3.98]
#   1 killed, pending, W 61.73          94.11  87.2 [88.2]   refuses by 6.91 [5.91]
#   2 uly2 BF16, resident               21.24  27.4 [28.4]   passes by 6.16 [7.16]
#   2 uly2 BF16, pending, W 61.73       82.96  87.2 [88.2]   passes by 4.24 [5.24]
#   3 one-rank INT8, preset             32.38  55.5 [56.5]   passes by 23.12 [24.12]
#   4 362 frames, second render         47.35  51.5          passes by 4.15
#   4 362 frames, gate slab path        47.35  56.3          passes by 8.95
#
# tests/test_h3_activation_preflight.py replays every row and recomputes the
# optimum. Against the two measured lower bounds, the estimate is 1.30x the
# 24.9 GiB the kill left unattributed and 2.69x the 17.6 GiB reservation at
# point 4 (1.75x the host-level 27 GiB that counts RSS and page cache). The
# refusal margin at point 1 fell from 10.5 to 4.0 GiB at the 28.4 GiB budget.
# The kill's reading carried the removed arena, so it overstates what that
# shape needs today, yet an estimate equal to it would pass that shape at
# either reserve, so the estimate must stay above it.
#
# F is fitted, not measured. Config A ran in about 4 GiB of activations and
# runtime, and the sibling's actor held 0.9 GiB beyond its weights between the
# 2026-10-29 load and the first step, so a small render is charged a few GiB
# more than it uses. That is the safe direction.
#
# ONE SPARK. The 362-frame cell at the loader is 109,062 rows (107,856 video,
# 1,206 audio) on one rank, with the pruned DiT pending (19.53 GiB), 92.8 GiB
# MemAvailable and the 5 GiB reserve. This fit charges 79.4 GiB of activations
# (7.5 floor, 64.4 sharded, 7.5 replicated) and 99.0 GiB in all against
# 87.8 GiB usable: REFUSED, 11.2 GiB short. The old slope charged 112.3 GiB of
# activations there, 44.1 GiB short. The 243-frame rung, 73,386 rows, fits at
# the same reading with 12.4 GiB to spare. A measured one-Spark run decides
# the 362-frame cell, not these constants.
#
# KNIFE EDGE. Constants that admit that cell exist, but none keep this fit's
# margins: every F and S that admit it leave point 1's refusal, point 2's
# pending pass or the cell itself with 2.5 GiB or less. Balancing the three
# gives F = 12.45 GiB and S = 451,181 B (about 42*5376*2), 2.5 GiB each, with a
# floor that charges small renders 5 GiB more. From S = 48*5376*2 up, any pair
# that admits the cell refuses point 1 by under 1 GiB, and from 49 up no pair
# that meets the four points admits it.
#
# VAEs 5.8 is the two files' size in decimal GB. They hold 5,813,063,304 B,
# 5.41 GiB (docs/VALIDATION.md, staged artifact sizes), so each pending U is
# 0.39 GiB low: with exact bytes point 2's pending pass gains 0.39 GiB and
# point 1's pending refusal loses as much. Point 1's resident budget, which
# binds, has no VAE term.
#
# Sanity bounds: an autograd-free block can concurrently hold x, h, two MLP
# projections, and the gate: 11*hidden*2 = 118,272 B/row. S is 5.4x that. The
# gap covers fp32 patch staging, attention workspaces, retained allocator
# memory and block prefetch (645.5 M parameters, 1.20 GiB BF16 per block). The
# generic mesh_safety factor of 8 is below H3's anatomy floor.
H3_ACTIVATION_FLOOR_BYTES = 15 * 2 ** 29          # 7.5 GiB
_H3_BLOCK_FACTOR = 59
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
# The refit keeps R: one measured shape cannot split it from S, and the
# 2026-10-29 reservation has room for its 7.5 GiB at 109,463 rows.
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
