"""The absolute host floor every load price charges, and why it is 5 GiB.

One constant and its derivation. Operators meet this floor in refusal cards,
so the derivation below stays whole; ``capacity_fit`` re-exports the constant.
"""
from __future__ import annotations

# The room this host must still hold after the load. Every load price (stock,
# slab, FSDP shard build, comfy-managed, and the driver footprint's default
# reserve) charges at least this much as an absolute term, not a share of the
# file, because what it pays for is ComfyUI's own decision to keep a model
# whole, and that decision does not scale with the checkpoint. Derived in four
# steps; the legs are in docs/VALIDATION.md, the 2026-09-03 partial-load
# threshold record:
#
# 1. ComfyUI's rule. ``load_models_gpu`` in comfy/model_management.py keeps the
#    weights whole only while the free memory left after the load stays at or
#    above ``minimum_memory_required``, which is
#    max(minimum_inference_memory(), memory_required + extra_reserved_memory()),
#    that is max(0.8 GiB + R, A + R) for R the launcher's reserve-vram figure
#    and A the sampler's activation estimate for this latent. Below it,
#    ``lowvram_model_memory`` comes out under the model size and comfy loads the
#    weights partially. This rig launches its drivers at reserve-vram 2, so the
#    static half of that term is 2.8 GiB and the rest of it is A.
#
# 2. What A is worth here. Boundary of 2026-09-03, flux2 bf16 60.02 GiB: the
#    batch-1 512 px legs stayed whole and rendered with 12.5 to 13.2 GiB
#    available after the load (the holds at 66, 70 and 90 GiB and one
#    unconstrained leg), and the dp2 batch-2 leg went partial at 11.0 GiB and
#    refused class C at partial_load_divergence. So A is worth about 2 GiB at
#    batch 1 and roughly doubles with the batch, and the earlier 4 GiB floor
#    covered the 2.8 GiB static half and almost none of A.
#
# 3. The upper bound: nothing that provably ran may be refused. The hold-66
#    leg was priced at 64.0 GiB against 65.4 GiB available and rendered.
#    That log figure carries one decimal, so the reading it rounds from is at
#    least 65.35, and the bound is 65.35 minus the 60.02 file: about 5.33 GiB.
#    5 GiB is the largest round figure under it, with 0.33 GiB to spare, and a
#    leg that pins the band tighter is what a raise would need.
#
# 4. On the stock side that rule has nothing to protect. On 2026-09-03 the
#    sweep corpus held 220 residency rows across 2529 cells, every one of them
#    residency=slab rung=vouched_auto, so no successful stock load of any
#    checkpoint is on file for this floor to refuse, and the price claims no
#    such record. Its one supporting datum is the 2026-09-02 incident the stock
#    multiplier in ``capacity_fit`` also cites: the rank that placed the
#    checkpoint read about 113 GiB at the price against the 111.0 GiB the
#    multiplier alone charged (1.85x that day), a margin near 2 GiB, then filled
#    100.77 GiB of anonymous pages and stopped answering, and the cell's
#    recorded outcome is a crash. The floor refuses that admission; it is one
#    crash, not a band. The other box of that run is the one recorded as
#    rung=stock_fits at 108.75 GiB available, under the 111.0 GiB price either
#    way; the record cannot say whether that rung was a price or a skipped
#    probe, and that box never placed the checkpoint in anonymous memory at all.
#
# This floor does not refuse the load that found the fault. Rank 0 of that dp2
# leg was admitted with 11.5 GiB of headroom over its price, so refusing it
# needs a floor above 15.5 GiB, which would refuse the holds at 66 and 70 GiB
# and the unconstrained leg, all of which rendered. A depends on the latent,
# and this price runs at the loader, before any latent exists, so the term
# meant to close the remaining gap is the driver-stack projection at the loader
# site (``nodes/loader_graph``), not a larger constant here. What the floor does
# buy is the marginal band: an admission that would leave the box between 4 and
# 5 GiB has room for neither comfy's static reserve nor a batch-1 activation
# set, and it is refused before the load rather than after.
#
# Raising this number needs a leg that refuses nothing on the 2026-09-03 hold
# record; lowering it needs a leg that renders below the band.
ABSOLUTE_HOST_FLOOR_BYTES = 5 * (1 << 30)
