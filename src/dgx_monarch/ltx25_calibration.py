"""Evidence-backed LTX dual-stream activation constants.

Calibration data stays separate from guard logic so a recalibration is
reviewable on its own. This module has no project or third-party imports.
"""
from __future__ import annotations

# Checkpoint architecture, read from the 2.5 DiT's ``__metadata__[config]``:
# 48 blocks, video 32 heads x 128 = 4096, audio 32 heads x 64 = 2048, ff 16384
# video / 8192 audio. LTX 2.3's 22B config is identical apart from ``ff_bias``,
# so these constants price the whole ``ltx`` family, not 2.5 alone.
LTX_VIDEO_DIM = 4096
LTX_AUDIO_DIM = 2048
_LTX_BF16_BYTES = 2

# CALIBRATION, 2026-08-12. The series and its instrument are
# docs/VALIDATION.md, "LTX activation calibration (2026-08-12, two DGX Sparks)".
#
#   sibling idle                                          115.17 GiB
#   stock load spike                                       74.63
#   post-load settled                                      88.37  ->  26.80 spent
#   after M1, 32,640 video rows (16,320/rank)              85.06  ->  30.11 spent
#   after M2, 57,600 video rows (28,800/rank)              81.68  ->  33.49 spent
#
# The staging arena is returned before the render starts (74.63 -> 88.37), and
# the allocator then keeps its render high-water, so each post-render step is an
# unmasked peak. Two readings result.
#
# 1. THE SLOPE. M2 loaded nothing, so its step over M1's is pure activation:
#        12,480*S_v + 24,960*R_v = 3.38 GiB   ->   S_v + 2*R_v = 290,797 B
#    The two absolute points bracket it from below, 218 KB and 249 KB
#    per per-rank row, as an intercept-free slope over a positive floor must.
#
# 2. THE PER-RANK FLOOR. 26.80 spent minus the 20.03 GiB file leaves 6.77 GiB
#    of CUDA context, torch, NCCL buffers and worker runtime. Only about
#    0.85 GiB of it exists before the load, so at the loader node the rest is
#    unspent and nothing else charges it.
#
# Splitting the slope needs the adapter's anatomy, because one world-2 series
# constrains only the sum. Replicated bytes are charged per total row because
# sequence parallelism does not divide them: adapters/ltx.py shards inside the
# block loop only, so the full-sequence x, the sp_gather buffer, the stock
# patchify_proj and _process_output, and the fp64 rope build all run undivided.
# That is about seven bf16 copies of a 4096-wide row, 57,344 B, and half that
# on the 2048-wide audio stream. The measured sum then fixes the sharded term
# at 290,797 - 114,688 = 176,109 B, or 21.5 bf16 copies of a per-rank row.
#
# SHIPPED: block factor 27, which is 221,184 B and 1.26x the fitted sharded
# term. Against the measured per-rank non-weight footprint (floor plus
# activations) the whole charge is 1.18x at M1 and 1.17x at M2. Never borrow
# the H3 slope here: the H3 block factor at LTX's width, 96 x 4096 x 2 =
# 786,432 B a row, is 2.7x the measured cost and would refuse shapes this
# hardware completes.
#
# VERDICTS this calibration is pinned to (tests/test_ltx25_activation_preflight):
#   * every measured pass fits, at its own recorded MemAvailable: Config A and
#     Config B on int8 and on bf16, M1, M2, and single-rank Config B;
#   * native 4K at 121 frames, 130,560 video rows, fits at world 2 on either
#     text-encoder tier, and completed on hardware;
#   * native 4K at 361 frames, 375,360 video rows, refuses at world 2 on both
#     text-encoder tiers and in both graph execution orders.
# The refusal threshold is text-encoder dependent. Before the encoder loads,
# the int8 encoder is charged about 11.7 GiB less than bf16 (10.1 GiB less file
# at loader_graph's 1.15 resident ratio); with it resident, the 4K rows the
# test pins read about 14 GiB more MemAvailable with int8 (94.4 against
# 80.5 GiB). A shape near the line can flip on that alone.
_LTX_BLOCK_FACTOR = 27
LTX_SHARDED_VIDEO_ROW_BYTES = _LTX_BLOCK_FACTOR * LTX_VIDEO_DIM * _LTX_BF16_BYTES
# Derived from the dimension ratio, not fitted: 121 frames at 24 fps is 126
# audio rows against 130,560 video rows at 4K, too few to measure separately.
LTX_SHARDED_AUDIO_ROW_BYTES = _LTX_BLOCK_FACTOR * LTX_AUDIO_DIM * _LTX_BF16_BYTES

LTX_REPLICATED_VIDEO_ROW_BYTES = 57_344      # 56 KiB
LTX_REPLICATED_AUDIO_ROW_BYTES = 28_672      # 28 KiB

# 6.75 GiB, just under the 6.77 measured, so the margin lives in the slope
# rather than in a term that does not grow with the request. A warm re-render
# is charged this again while its previous floor is already inside the reading;
# that double charge is deliberate slack and stays far below the weight term it
# replaces on the warm path.
LTX_RANK_FLOOR_BYTES = 7_247_757_312
