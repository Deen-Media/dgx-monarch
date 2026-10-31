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
# 96*5376*2 = 1,032,192 B. The kill left 24.9 GiB unattributed at 37,730 rows;
# the slope sat 1.63x above that reading and carried every fixed cost.
#
# The 2026-10-29 refit added a 7.5 GiB floor and cut S to 59*5376*2 =
# 634,368 B, the rounded optimum of four points: the kill as a must-refuse
# point, the world-2 and INT8 completions, and the 2026-10-29 pair render at
# 362 frames, which the old slope had charged 60.1 GiB per rank against the
# 17.6 GiB it held. That fit still refused the 362-frame cell on one Spark,
# 11.2 GiB short.
#
# On 2026-10-31 the killed shape rendered. On dev e4f23be the head ran
# 1344x768x124 as one rank beside the driver, 38,111 packed rows (the template
# prompt's 401 text rows), with the 61.73 GiB BF16 DiT slab-resident under a
# capacity-rescue consent and both driver preflights off for the measurement:
# 20 steps in 1320 s, the actor holding 5.5 GiB beyond the slab through every
# step. The 2026-08-05 kill carried the legacy load path's retained arena,
# since removed, so its 24.9 GiB is no longer a lower bound, and point 1 turned
# from a refusal into a pass. On 2026-10-30 one rank on the sibling alone had
# rendered the 362-frame cell, 109,463 rows, holding 17.0 GiB beyond its
# 19.53 GiB of weights; it became point 5.
#
# REQUIREMENTS, per rank. U is MemAvailable minus the reserve and W the weight
# charge. The reserve has been 5 GiB since 2026-09-04
# (driver_footprint.reserve_bytes); the 2026-08-05 fit used 4 GiB, and the
# 4 GiB budgets are in brackets. Every requirement is now a pass or a measured
# lower bound, so each pass takes the 5 GiB budget, the harder one.
#
#   A. Config A, 512x320x5, one rank: the estimate must exceed the about
#      4 GiB of activations and runtime the 2026-08-05 stock single-process
#      reference run of the bf16 artifact used there, above its 28.4 GiB
#      post-render floor (docs/VALIDATION.md, 2026-08-05). That run's prompt
#      length is not on record, so the bound counts its 336 video and audio
#      rows only, the stricter count; Config A's committed 356 rows add
#      0.004 GiB.
#   1. The 2026-10-31 head render, one rank, BF16, 38,111 rows: completed, must
#      pass in both postures and charge more than the 5.5 GiB it held.
#      RESIDENT: 29.70 GiB MemAvailable with the slab mapped and the actor at
#      0.3 GiB, the second before its pool grew: U = 24.70 [25.70].
#      PENDING: 92.34 GiB with the text encoder and VAEs in and the DiT not yet
#      loaded: U = 87.34 [88.34], W = 66,280,487,368 B.
#   2. Config B, world 2 uly2, BF16, 18,865 rows per rank: completed, must
#      pass in both postures. Its budgets come from the same reference run:
#      114.7 GiB idle and the 28.4 GiB floor. RESIDENT: about 32.4 GiB at
#      dispatch (that floor plus the Config A use above): U = 27.4 [28.4].
#      PENDING: 92.2 GiB with the encoder (16.7 GiB from another run; that run
#      reads 14.6) and the VAEs in: U = 87.2 [88.2], W as point 1.
#   3. Config B, one rank, INT8 under an explicit preset: completed, must pass.
#      U = 55.5 [56.5], W = 0. U charges the 31.70 GiB non-pruned file, but the
#      completion ran the 19.53 GiB pruned one (docs/VALIDATION.md, H3
#      promotion Leg 1); the larger charge only lowers this ceiling.
#   4. The 2026-10-29 pair render, 1344x768x362 on uly2, 109,463 rows (107,856
#      video, 1,206 audio, 401 text), 54,732 per rank, weights resident:
#      completed, must pass at U = 51.5 (the second render) and at U = 56.3
#      (the gate's slab path), and charge more than the 17.6 GiB it held.
#   5. The 2026-10-30 one-rank render on the sibling alone, the same 109,463
#      rows, weights resident: completed, must pass at U = 85.29 [86.29], from
#      the 94,681,052 kB MemAvailable read the second before the actor's pool
#      grew, and charge more than the 17.0 GiB it held.
#
# FIT. The budgets carry whole-GiB doubt (92.2 GiB is derived, the encoder
# reads 14.6 or 16.7 GiB, the VAEs 5.41 or 5.8), so the fit maximises the
# smallest margin in GiB, with F >= 0 as a bound. The five points alone would
# put F at -4.03 GiB, a floor that charges a small render less than nothing;
# Config A's reading holds it up. The optimum is F = 10.87 GiB and
# S = 120,545 B, with Config A, point 1 resident and point 4's lower bound
# tied at 6.93 GiB. Rounded as before, F to a half GiB is 11.0 GiB. S to the
# nearest block factor would be 11*5376*2 = 118,272 B, the anatomy floor
# below, which S must exceed, so S = 12*5376*2 = 129,024 B (0.12 MiB/row).
# The margins in GiB are:
#
#   point                            estimate  budget          margin
#   A Config A, used about 4.0          11.06   4.0            above by 7.06
#   1 head BF16, resident               18.20  24.70 [25.70]   passes by 6.51 [7.51]
#   1 head BF16, pending, W 61.73       79.92  87.34 [88.34]   passes by 7.42 [8.42]
#   1 head BF16, held 5.5               18.20   5.50           above by 12.70
#   2 uly2 BF16, resident               15.86  27.4 [28.4]     passes by 11.54 [12.54]
#   2 uly2 BF16, pending, W 61.73       77.59  87.2 [88.2]     passes by 9.61 [10.61]
#   3 one-rank INT8, preset             18.12  55.5 [56.5]     passes by 37.38 [38.38]
#   4 362 frames, second render         25.09  51.5            passes by 26.41
#   4 362 frames, gate slab path        25.09  56.3            passes by 31.21
#   4 362 frames, held 17.6             25.09  17.6            above by 7.49
#   5 one rank 362, resident            31.67  85.29 [86.29]   passes by 53.63 [54.63]
#   5 one rank 362, held 17.0           31.67  17.0            above by 14.67
#
# Point 1 resident binds at 6.51 GiB, Config A follows at 7.06 and point 4's
# lower bound at 7.49. tests/test_h3_activation_preflight.py replays every
# row, recomputes the optimum and checks the rounding.
#
# THE MEASURED POINTS. Six reservations are on record (docs/VALIDATION.md,
# "Activation across clip lengths"): on the pair 7.3, 12.5 and 17.6 GiB at
# 18,865, 36,895 and 54,732 rows per rank; one rank 11.6 and 17.0 GiB at
# 73,790 and 109,463 rows, and 5.5 GiB for the BF16 head render at 38,111.
# The charge clears each one, by 1.43x at the pair's 362 frames and up to
# 3.31x. They follow the whole packed total, not a rank's own rows: one rank
# holding every row held no more than a pair rank holding half of them. A
# least-squares line through the five INT8 ones puts about 165 KB on each row
# of the total and about nothing on a rank's own rows, over a 1.7 GiB floor,
# so a larger R would follow them more closely. No requirement forces that,
# so R stays; the floor carries the rest, and at world 4 this fit still
# charges 21.8 GiB a rank for 109,463 rows against the 18.1 the line predicts.
# No render above world 2 is on record.
#
# F is fitted to the requirements, not measured: Config A used about 4 GiB, so
# a small render is charged about 7 GiB more than it uses. That is the safe
# direction.
#
# ONE SPARK. The 362-frame cell at the loader is 109,062 rows (107,856 video,
# 1,206 audio) on one rank, with the pruned DiT pending (19.53 GiB), 92.8 GiB
# MemAvailable and the 5 GiB reserve. This fit charges 31.6 GiB of activations
# (11.0 floor, 13.1 sharded, 7.5 replicated) and 51.1 GiB in all against
# 87.8 GiB usable: ADMITTED, with 36.7 GiB to spare. The 2026-10-29 fit refused
# it 11.2 GiB short and the 2026-08-05 slope 44.1. The 243-frame rung, 73,386
# rows, fits at the same reading with 43.4 GiB to spare. Nothing was tuned for
# the cell: held to the same footing as the requirements, it leaves the
# optimum where it is. One rank on the sibling alone rendered it; on the head
# beside the driver it has no run yet (docs/VALIDATION.md names the check).
#
# KNIFE EDGE. The 2026-10-29 fit could admit that cell only by giving up margin
# on the kill or on point 2. This fit admits it by 36.7 GiB at no cost, and no
# requirement sits closer than 6.5 GiB. The edge is now Config A: its reading
# is an about-4 GiB figure from one reference run, and it alone holds the floor
# up, so a measured small render on today's stack is the next point to take.
#
# VAEs 5.8 is the two files' size in decimal GB. They hold 5,813,063,304 B,
# 5.41 GiB (docs/VALIDATION.md, staged artifact sizes), so point 2's pending U
# is 0.39 GiB low; with exact bytes it gains 0.39 GiB.
#
# Sanity bounds: an autograd-free block can concurrently hold x, h, two MLP
# projections, and the gate: 11*hidden*2 = 118,272 B/row. S is 1.09x that, the
# least block factor above it. Fp32 patch staging, attention workspaces,
# retained allocator memory and block prefetch (645.5 M parameters, 1.20 GiB
# BF16 per block) sit in the floor and in R. The generic mesh_safety factor of
# 8 is below H3's anatomy floor.
H3_ACTIVATION_FLOOR_BYTES = 11 * 2 ** 30          # 11.0 GiB
_H3_BLOCK_FACTOR = 12
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
# Both refits keep R (see THE MEASURED POINTS).
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
