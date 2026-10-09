"""Host-memory capacity prices for preflight and residency selection.

Two probes read one set of measurements through ``mesh_safety``:
``stock_load_fit`` prices a cudaMalloc load at the file plus the host copy it
is placed from, ``slab_load_fit`` prices slab residency at the file plus a capped
floor, both of them over ``ABSOLUTE_HOST_FLOOR_BYTES``, and a skipped probe
claims nothing either way. The FSDP shard-build and comfy-managed prices live
here too, both over that same floor. ``adapters/fsdp`` re-exports the first, so
worker preflights and identity-check reloads use the same estimate;
``driver_footprint`` keeps its own, lower shard-build factor.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from . import mesh_safety, residency_mode
from .capacity_floor import ABSOLUTE_HOST_FLOOR_BYTES as ABSOLUTE_HOST_FLOOR_BYTES
from .capacity_lora_bake import lora_bake_bytes as _lora_bake_bytes
from .safetensors_header import (
    SafetensorsHeaderError,
    UnsupportedSafetensorsDtypeError,
    read_safetensors_header,
)

# Both probes stay in ``gate_audit_vocab.MEASURED_PROBES`` so their measurements
# can support a capacity-certified ledger row.
PROBE_STOCK_LOAD_FIT = "worker_stock_load"
PROBE_SLAB_LOAD_FIT = "worker_slab_load"

# Families with identity-gate evidence for zero-copy slab residency. The
# driver's loader preflight (``nodes/loader_preflight``) and the worker's
# residency ladder and capacity quote consult this set before pricing a stock
# load for ``auto``; ``actor/model_store`` re-exports it. Vouching is per
# family; flux2's evidence is fp8mixed only. bf16 flux2 is in this set and
# slab-loads on any rank with a warm memo; what its size excludes is the
# vouching ceremony, whose stock comparison leg cannot fit beside the driver
# text encoder. FSDP is not the alternative: ``worker_env.slab_mode_effective``
# is False on a sharded topology, so an FSDP rank takes no slab at all.
SLAB_VOUCHED_FAMILIES = frozenset({"krea2", "flux2"})

_SKIP_DTYPE_CAST = "a loader dtype cast changes the resident size"
_SKIP_DISCRETE = "the CUDA device is not integrated (host memory does not bound capacity)"
_SKIP_NO_MEMINFO = "kernel MemAvailable is unreadable here"
_SKIP_SLAB_DTYPE_CAST = (
    "a loader dtype cast adds a second shared-memory arena, the annex, whose "
    "size this probe does not measure"
)
_SKIP_SLAB_UNSUPPORTED_DTYPE = (
    "the slab loader cannot wrap a dtype this checkpoint stores, so this load "
    "falls back to the stock loader and takes the stock price"
)
_SKIP_NOT_SAFETENSORS = "the zero-copy slab loader only wraps safetensors files"


def gib(value: int | float) -> float:
    """One rounding, so every block and every card print the same figure."""
    return round(value / (1 << 30), 1)


@dataclass(frozen=True, slots=True)
class StockFit:
    """Whether a stock cudaMalloc load of this checkpoint fits, and by how much."""

    fits: bool
    applies: bool          # False when the probe was skipped; see skipped_reason
    size_bytes: int
    avail_bytes: int | None
    skipped_reason: str = ""

    @property
    def size_gib(self) -> float:
        return round(self.size_bytes / (1 << 30), 1)

    @property
    def avail_gib(self) -> float | None:
        return None if self.avail_bytes is None else round(self.avail_bytes / (1 << 30), 1)

    @property
    def required_bytes(self) -> int:
        """The load-path transient the verdict compares against."""
        if not self.applies:
            return int(self.size_bytes)
        return stock_required_bytes(self.size_bytes)

    @property
    def required_gib(self) -> float:
        return round(self.required_bytes / (1 << 30), 1)

    @property
    def headroom_gib(self) -> float | None:
        if self.avail_bytes is None:
            return None
        return round((self.avail_bytes - self.required_bytes) / (1 << 30), 1)

    def measured(self) -> dict[str, object]:
        """The consent card's and the v9 ledger's block: bytes for the schema, GiB to read."""
        return {
            "probe": PROBE_STOCK_LOAD_FIT,
            "checkpoint_bytes": int(self.size_bytes),
            "mem_available_bytes": (None if self.avail_bytes is None
                                    else int(self.avail_bytes)),
            "required_bytes": int(self.required_bytes),
            "headroom_bytes": (0 if self.avail_bytes is None
                               else int(self.avail_bytes - self.required_bytes)),
            "weights_gib": self.size_gib,
            "required_gib": self.required_gib,
            "mem_available_gib": self.avail_gib,
            "headroom_gib": self.headroom_gib,
            "fits": bool(self.fits),
            "applies": bool(self.applies),
        }


@dataclass(frozen=True, slots=True)
class SlabFit:
    """Slab residency's verdict: ``StockFit``'s fields plus the floor charged."""

    fits: bool
    applies: bool          # False when the probe was skipped; see skipped_reason
    size_bytes: int
    avail_bytes: int | None
    floor_bytes: int = 0
    skipped_reason: str = ""
    # The LoRA bake's stray transient (capacity_lora_bake.py), appended last so
    # every existing positional construction keeps its meaning.
    lora_bytes: int = 0

    @property
    def size_gib(self) -> float:
        return round(self.size_bytes / (1 << 30), 1)

    @property
    def avail_gib(self) -> float | None:
        return None if self.avail_bytes is None else round(self.avail_bytes / (1 << 30), 1)

    @property
    def floor_gib(self) -> float:
        return round(self.floor_bytes / (1 << 30), 1)

    @property
    def lora_gib(self) -> float:
        return round(self.lora_bytes / (1 << 30), 1)

    @property
    def required_bytes(self) -> int:
        """The file, the LoRA bake's stray transient, and the rest-of-box floor."""
        if not self.applies:
            return int(self.size_bytes)
        return int(self.size_bytes) + int(self.floor_bytes) + int(self.lora_bytes)

    @property
    def required_gib(self) -> float:
        return round(self.required_bytes / (1 << 30), 1)

    @property
    def headroom_gib(self) -> float | None:
        if self.avail_bytes is None:
            return None
        return round((self.avail_bytes - self.required_bytes) / (1 << 30), 1)

    def measured(self) -> dict[str, object]:
        """``StockFit.measured``'s keys plus two for the card; the ledger row shape is unchanged."""
        return {
            "probe": PROBE_SLAB_LOAD_FIT,
            "checkpoint_bytes": int(self.size_bytes),
            "mem_available_bytes": (None if self.avail_bytes is None
                                    else int(self.avail_bytes)),
            "required_bytes": int(self.required_bytes),
            "headroom_bytes": (0 if self.avail_bytes is None
                               else int(self.avail_bytes - self.required_bytes)),
            "floor_bytes": int(self.floor_bytes),   # for the card; the ledger drops it
            "lora_bytes": int(self.lora_bytes),     # for the card; the ledger drops it
            "weights_gib": self.size_gib,
            "required_gib": self.required_gib,
            "mem_available_gib": self.avail_gib,
            "headroom_gib": self.headroom_gib,
            "fits": bool(self.fits),
            "applies": bool(self.applies),
        }


# The retained load arena a stock cudaMalloc load leaves behind, as a share of
# the file. Measured: a legacy load retained 20.7 GiB for a 24.5 GiB model, or
# 0.845, rounded up here so the charge is never under the measurement
# (docs/VALIDATION.md, the retained allocator pool entry).
# ``slab_floor_bytes`` caps the slab's rest-of-box floor at this share.
LEGACY_ARENA_RATIO = 0.85

# The stock cudaMalloc load path transiently costs more host memory than the
# file while comfy materializes and places weights: the loaded state sits in
# anonymous host memory while ComfyUI builds the device copy beside it, and the
# host side frees only once every weight has moved. A stock load places every
# weight at load time (``store_load`` through ``resident_ledger.declare``), so
# the price is that full-placement peak. ``driver_footprint`` imports this ratio
# rather than keeping a second copy, so the driver and the worker cannot charge
# two prices for one placement.
#
# Measured 2026-10-01 on the 33.0 GiB flux2 fp8mixed file: the host copy took
# 34.7 GiB and ComfyUI sized the device copy at 33.8 GiB, 2.08x the
# file between them, and a full load that started at 74.5 GiB available
# troughed at 10.4. ``STOCK_PLACEMENT_RATIO`` is the share above the file,
# rounded up so the charge is never under the measurement. The earlier bf16
# readings sat lower because ComfyUI was offloading part of the model under
# pressure: 1.27x before an OS kill at 69.23 GiB available and 1.83x at 107.41
# (2026-08-12, 60.02 GiB), and 1.694x with 10.58 GiB left at 107.46
# (2026-09-02). A partial load holds the host copy until a later pass completes
# it, so it was never a cheaper placement, only a slower one. The retained
# arena (``LEGACY_ARENA_RATIO``) is what a finished legacy load leaves behind.
STOCK_PLACEMENT_RATIO = 1.1
STOCK_LOAD_TRANSIENT_FACTOR = 1.0 + STOCK_PLACEMENT_RATIO


def stock_required_bytes(size_bytes: int) -> int:
    """The file, the host copy it is placed from, and the absolute host floor."""
    return int(size_bytes * STOCK_LOAD_TRANSIENT_FACTOR) + ABSOLUTE_HOST_FLOOR_BYTES


# What kills an FSDP load is anonymous commit, not warm page cache. The build
# wraps every parameter on meta and moves only this rank's rows from the mmap
# view (adapters/fsdp_shard_build.py), with no second state-dict copy and no
# full-size device copy per block, so a rank commits its shards (1/world) plus
# one chunk in flight.
#
# Measured on the flux2 bf16 60 GiB file in legs 3 and 4 of 2026-08-26:
# a single build's anonymous high-water was 32.8 GiB, 0.547x the 60 GiB file,
# which is the 0.5 shards plus a 0.047 transient (the largest block, 1.83 GiB,
# is 0.03x; the rest is allocator slack). The per-world factor is 1/world plus
# a flat 0.08, which sits above that 0.047 measured transient with margin and
# below what the box can hold: for flux2 at world 2, 0.58x = 34.8 GiB, and the
# ceremony's clean reload had 42 GiB free on the head at the price (leg 4). A
# 0.25 fraction priced 45 GiB and refused that reload. When the world size is
# unknown, or the checkpoint is not bf16 or fp16 (a quantized direct wrap or an
# unproven kind), the 1.2x full-copy bound applies. Replicated modules are not
# priced one by one; their bytes fall inside the flat fraction
# (docs/VALIDATION.md, 2026-09-09).
FSDP_MATERIALIZE_FACTOR = 1.2
FSDP_STREAM_TRANSIENT_FRACTION = 0.08
FSDP_STREAMING_CHECKPOINT_KINDS = frozenset({"bf16", "fp16"})


def fsdp_materialize_factor(
    world: int | None, checkpoint_kind: str | None = None,
) -> float:
    """Return the streaming fraction or conservative direct-wrap bound."""
    if checkpoint_kind not in FSDP_STREAMING_CHECKPOINT_KINDS:
        return FSDP_MATERIALIZE_FACTOR
    if world is None or int(world) < 1:
        return FSDP_MATERIALIZE_FACTOR
    return 1.0 / int(world) + FSDP_STREAM_TRANSIENT_FRACTION


def fsdp_required_bytes(
    size_bytes: int, world: int | None = None, checkpoint_kind: str | None = None,
) -> int:
    """Return the FSDP load transient plus the shared host floor."""
    return (int(size_bytes * fsdp_materialize_factor(world, checkpoint_kind))
            + ABSOLUTE_HOST_FLOOR_BYTES)


# The guard the comfy-managed and FSDP shard-build walls raise under; neither runs a probe here.
PROBE_PREFLIGHT_WALL = "stock_load_preflight"


def preflight_wall_measured(size: int, avail: int, required: int,
                            fits: bool) -> dict[str, object]:
    """Format managed and FSDP capacity measurements for the ledger and UI.

    These estimates do not use ``StockFit``. Include the measurement block so a
    priced result remains distinguishable from an unpriced one, and include GiB
    values for the fleet card's display and sorting.
    """
    return {"probe": PROBE_PREFLIGHT_WALL, "checkpoint_bytes": int(size),
            "mem_available_bytes": int(avail), "required_bytes": int(required),
            "headroom_bytes": int(avail - required), "fits": fits, "applies": True,
            "weights_gib": gib(size), "required_gib": gib(required),
            "mem_available_gib": gib(avail), "headroom_gib": gib(avail - required)}


def managed_required_bytes(size: int, pinned_staging: bool) -> int:
    """The fourth price, beside stock, slab and the shard build: one resident
    copy, a second under pinned staging, and the shared absolute host floor."""
    factor = residency_mode.PINNED_STAGING_FACTOR if pinned_staging else 1.0
    return int(size * factor) + ABSOLUTE_HOST_FLOOR_BYTES


def stock_load_fit(path: str, model_options: dict) -> StockFit:
    """Measure whether stock residency can hold ``path`` on this host."""
    if model_options.get("dtype") is not None:
        return StockFit(True, False, 0, None, _SKIP_DTYPE_CAST)
    if not mesh_safety.gpu_is_integrated():
        return StockFit(True, False, 0, None, _SKIP_DISCRETE)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    avail = mesh_safety.mem_available_bytes()
    if avail is None:
        return StockFit(True, False, size, None, _SKIP_NO_MEMINFO)
    fit = StockFit(True, True, size, avail)
    return StockFit(fit.required_bytes <= avail, True, size, avail)


def slab_floor_bytes(size: int, reserve_bytes: int | None) -> int:
    """The rest-of-box floor a slab load charges; the pre-bake backstop charges the same one."""
    return max(ABSOLUTE_HOST_FLOOR_BYTES, min(max(residency_mode.UMA_HOST_FLOOR_BYTES, int(reserve_bytes or 0)),
                                              int(LEGACY_ARENA_RATIO * size)))


def slab_load_fit(path: str, model_options: dict, *, reserve_bytes: int | None = None,
                  lora_stack: list[dict] | None = None,
                  resolve_lora_path: Callable[[str], str] | None = None,
                  lora_credit_bytes: int = 0) -> SlabFit:
    """Measure whether zero-copy slab residency can hold ``path`` on this host.

    One times the file, measured: a fill took Shmem 0.05 to 60.07 GiB for a
    60.02 GiB file and MemAvailable never overshot the settled figure. The size
    is ``getsize``, not the header's aligned total, so both prices read one
    measurement. On top sits the floor the rest of the box needs: the operator's
    reserve, capped by the retained stock arena and never under
    ``ABSOLUTE_HOST_FLOOR_BYTES``, which the stock price also carries, and,
    where ``lora_stack`` and ``resolve_lora_path`` are both given, the bake's
    own stray transient (``capacity_lora_bake.lora_bake_bytes``). Both callers
    that price a slab load pass the same two, so neither prices a bake the
    other misses.
    """
    if model_options.get("dtype") is not None:
        return SlabFit(True, False, 0, None, 0, _SKIP_SLAB_DTYPE_CAST)
    if not mesh_safety.gpu_is_integrated():
        return SlabFit(True, False, 0, None, 0, _SKIP_DISCRETE)
    if not path.lower().endswith((".safetensors", ".sft")):
        return SlabFit(True, False, 0, None, 0, _SKIP_NOT_SAFETENSORS)
    size = os.path.getsize(path) if os.path.exists(path) else 0
    header = None
    if os.path.exists(path):
        try:
            # The same read WeightSlab.__init__ makes before it sizes its arena.
            header = read_safetensors_header(path)
        except UnsupportedSafetensorsDtypeError:
            return SlabFit(True, False, size, None, 0, _SKIP_SLAB_UNSUPPORTED_DTYPE)
        except SafetensorsHeaderError:
            # A container this parser rejects outright is not a capability miss.
            # The slab load fails on its own terms; the price still holds.
            pass
    avail = mesh_safety.mem_available_bytes()
    if avail is None:
        return SlabFit(True, False, size, None, 0, _SKIP_NO_MEMINFO)
    floor = slab_floor_bytes(size, reserve_bytes)
    lora_bytes = max(0, (
        _lora_bake_bytes(path, header.tensors, lora_stack, resolve_lora_path)
        if lora_stack and resolve_lora_path is not None and header is not None
        else 0) - max(int(lora_credit_bytes), 0))
    return SlabFit(size + floor + lora_bytes <= avail, True, size, avail, floor,
                   lora_bytes=lora_bytes)
