"""Dependency-light residency vocabulary shared by drivers and workers.

Comfy-managed placement is explicit-only and refuses unsupported combinations.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

MODE_STOCK = "stock"
MODE_SLAB = "slab"
MODE_COMFY_MANAGED = "comfy_managed"
RESIDENCY_MODES = frozenset({MODE_STOCK, MODE_SLAB, MODE_COMFY_MANAGED})

# A pricing token, never a reported residency: an FSDP rank reports stock, so
# it stays out of RESIDENCY_MODES. The driver footprint estimator uses it to
# name the window it charged.
MODE_FSDP_SHARD = "fsdp_shard"

# Omit a false key because worker args bind gate capability contexts.
WORKER_ARG = "comfy_managed"
# Published only after successful worker activation.
ENV_ACTIVE = "DGXM_COMFY_MANAGED"
TROUBLESHOOTING = 62

# Default slab reserve when ``uma_reserve_gb`` is absent. The shared
# ``capacity_floor`` exceeds this default, so this value alone cannot lower
# a price. ``capacity_fit.slab_floor_bytes`` caps an operator reserve at the
# retained stock arena: slab file plus floor never exceeds the stock price
# of 2.1 times file size plus the absolute floor.
UMA_HOST_FLOOR_BYTES = 4 * 1024 ** 3

# What a managed load costs when ComfyUI stages through a pinned host buffer,
# as a multiple of the checkpoint. ``pinned_hostbuf_size`` sizes that buffer at
# twice min(model, MAX_PINNED_MEMORY), and on unified memory the copy it fills
# lands in the same pool the device already computes from, so the second copy is
# charged against the same budget as the first. Measured 2026-08-12 on a worker
# with nothing else resident, 39.13 GiB BF16 checkpoint: MemAvailable fell from
# 115.14 to 25.45 GiB, or 2.29x the artifact, of which anonymous pages took
# 40.95 GiB and device allocations took 42.80 GiB. Rounded up to the nearest
# tenth from that one clean measurement. The same arithmetic at 60.02 GiB needs
# 137 GiB, which is why a 60.02 GiB managed load admitted against 77.27 GiB
# available on 2026-08-12 died in the NVIDIA driver's allocator rather than
# rendering.
PINNED_STAGING_FACTOR = 2.3

# Quarantine may change this key only when its gate context already contains it.
QUARANTINE_LEVERS = ("lora_low_rss", "slab_weights", WORKER_ARG)

# FAIL can attribute only modes compared by the ceremony.
REPORTABLE_QUARANTINE_LEVERS = ("lora_low_rss", "slab_weights")


# One home for the FSDP refusal text, so the operator meets one sentence for one
# combination: the driver raises it at the loader node, above the footprint
# card, and the worker raises it at the load funnel.
COMFY_MANAGED_FSDP_REFUSAL = (
    "comfy-managed residency is on for this worker and FSDP is active for this "
    "render. FSDP reshards weights into DTensors after the load while comfy's "
    "DynamicVRAM pages the same weights independently on every rank, and the "
    "two have never run together on this hardware. Nothing was loaded and "
    "nothing was quarantined. What works instead: for FSDP topologies, turn "
    "the Init node's comfy_managed widget off and restart the Worker service "
    "(`dgxm restart`), so FSDP's own sharding provides the capacity; or pick "
    "a topology without FSDP."
)


# The other home for a comfy-managed sentence, raised by the driver alone. The
# generic stock-mode refusal in mesh_residency.py offers a first-use ceremony as
# its remedy. For a combination whose proof render meets a class-K guard that
# remedy can never succeed, because a ceremony renders without accuracy waivers
# on purpose, so this sentence names the guard and the settings that do work.
# ``{guard}`` is the ledger guard the ceremony refused on, and
# ``{headless}`` its variable, named here because a class-P refusal gets no
# panel sentence of its own and the card it points at expires after 30 minutes.
COMFY_MANAGED_KNOWN_WRONG_REFUSAL = (
    "comfy-managed residency is on for this worker and the first-use identity "
    "ceremony for this combination cannot prove it. The ceremony's own proof "
    "render refuses on the class-K guard {guard}, and a ceremony renders "
    "without accuracy waivers on purpose, so granting that waiver does not "
    "clear the guard there and the ceremony can never reach PASS for this "
    "combination. Unlike slab_weights and lora_low_rss, there is no safe "
    "residency to fall back to: ComfyUI's DynamicVRAM is a bootstrap policy "
    "and cannot be turned off for one render. Nothing was loaded and nothing "
    "was quarantined. What works instead: grant the {guard} waiver from the "
    "panel, or headless {headless}, then set the Init node's auto_gate widget "
    "to off for this graph. That skips the proof and dispatches on the policy "
    "the graph asked for, and every render is stamped rendered-under-waiver "
    "(docs/TROUBLESHOOTING.md #67 says what auto_gate=off costs). Or turn the "
    "Init node's comfy_managed widget off and restart the Worker service "
    "(`dgxm restart`)."
)


class ComfyManagedResidencyError(RuntimeError):
    """Class-P refusal, never a skippable stock-capacity error."""


def requested(worker_args: Mapping[str, Any] | None) -> bool:
    """Return true only for an explicit boolean enablement."""
    if not worker_args:
        return False
    return worker_args.get(WORKER_ARG) is True


def leverless_ceremony_verdict(
    worker_args: Mapping[str, Any] | None,
    reasons: list[str],
) -> tuple[bool, list[str]]:
    """Allow a leverless verdict only when the rung itself was exercised.

    This proves repeat/per-rank identity for Comfy-managed residency, not
    equivalence with another residency mode.
    """
    if requested(worker_args):
        return True, sorted({
            *reasons,
            "comfy-managed residency: the a/b repeat and the in-path per-rank "
            "identity are the rung's own proof; no swap or slab lineage "
            "exists under it",
        })
    return False, sorted({
        *reasons,
        "no LoRA/slab lineage and no complete FSDP clean-reload proof",
    })


def active() -> bool:
    """Return the activation latch published after successful worker setup."""
    return os.environ.get(ENV_ACTIVE) == "1"


def pricing_value(worker_args: Mapping[str, Any] | None) -> bool | str | None:
    """Return the residency value used by all driver footprint estimators."""
    if requested(worker_args):
        return MODE_COMFY_MANAGED
    if not worker_args:
        return None
    value = worker_args.get("slab_weights")
    return value if isinstance(value, (bool, str)) else None


def charges_legacy_arena(slab_weights: bool | str | None) -> bool:
    """Charge the legacy load arena only for exact stock residency."""
    return slab_weights is False


def arena_omitted_note(slab_weights: bool | str | None) -> str:
    """The note text for the branch where no legacy load arena is charged."""
    if slab_weights == MODE_COMFY_MANAGED:
        return ("omitted, the worker runs comfy-managed residency: comfy places and "
                "pages the weights itself, and stock comfy with DynamicVRAM showed "
                "a retained pool of 0 GiB on 2026-07-08")
    if slab_weights == MODE_FSDP_SHARD:
        return ("omitted, the FSDP streaming shard build assigns file-backed "
                "rows into shards and never makes the full copy a legacy load "
                "retains")
    if slab_weights is True:
        return "omitted, slab residency has no legacy load arena"
    return "omitted, slab_weights is auto or unset"


def quarantine_metadata_exact(reported: Any) -> bool:
    """Require both attributable levers and no unknown quarantine lever."""
    if not isinstance(reported, list):
        return False
    if not all(isinstance(lever, str) for lever in reported):
        return False
    values = set(reported)
    return (values.issuperset(REPORTABLE_QUARANTINE_LEVERS)
            and values.issubset(QUARANTINE_LEVERS))


def quarantine_values(worker_args: Mapping[str, Any] | None) -> dict[str, bool]:
    """Return FAIL policy without changing the capability-context shape."""
    values = {"lora_low_rss": False, "slab_weights": False}
    if worker_args is not None and WORKER_ARG in worker_args:
        values[WORKER_ARG] = False
    return values
