"""GPUWorker status, memory introspection, and latent signatures."""
from __future__ import annotations

import hashlib
import os
import socket
from typing import Any

from .. import residency_mode
from ..log import get_logger
from ..runtime_provenance import (
    cached_dgx_source_manifest_sha256 as source_manifest_sha256,
)
from ..runtime_provenance import runtime_provenance_snapshot as _runtime_snapshot
from ..transfer_utils import failure_summary

log = get_logger(__name__)


_SIGNATURE_CHUNK_ELEMENTS = 1 << 18
_PROJECTION_LANES = (
    (1_103_515_245, 12_345, 2_147_483_647),
    (1_664_525, 1_013_904_223, 2_147_483_629),
)


def runtime_provenance_snapshot(
    artifact_manifest: object = None,
    *,
    custom_nodes_disabled_override: bool | None = None,
) -> dict[str, object]:
    """Bind the canonical snapshot to this process's bootstrap policy."""
    from .comfy_bridge import _CUSTOM_NODES_DISABLED

    custom_nodes_disabled = (
        _CUSTOM_NODES_DISABLED is True
        if custom_nodes_disabled_override is None
        else custom_nodes_disabled_override is True
    )
    return _runtime_snapshot(
        artifact_manifest,
        custom_nodes_disabled=custom_nodes_disabled,
    )


def _add_projection_chunk(values, start: int, totals, counts, buckets: int) -> None:
    import torch

    idx = torch.arange(start, start + values.numel(), dtype=torch.int64)
    for lane, (multiplier, increment, prime) in enumerate(_PROJECTION_LANES):
        hashed = (idx * multiplier + increment).remainder(prime)
        bucket = hashed.remainder(buckets)
        weights = hashed.to(torch.float64).mul_(2.0 / prime).sub_(1.0)
        offset = lane * buckets
        totals[offset:offset + buckets].scatter_add_(0, bucket, values * weights)
        counts[offset:offset + buckets].add_(
            torch.bincount(bucket, minlength=buckets).to(torch.float64))


def _finish_projection(totals, counts) -> list[float]:
    normalized = totals / counts.sqrt().clamp_min_(1.0)
    return [round(float(value), 9) for value in normalized.tolist()]


def _latent_signature(t) -> dict:
    """Full-content identity/statistics with bounded host scratch memory.

    Fixed-size chunks feed every byte to SHA-256 and every value to the moments
    and projection while keeping peak scratch to a few MiB.
    """
    import torch  # Keep torch out of module scope to preserve bootstrap order.

    flat = t.detach().contiguous().reshape(-1)
    n = flat.numel()
    digest = hashlib.sha256()
    totals = torch.zeros(64, dtype=torch.float64)
    counts = torch.zeros(64, dtype=torch.float64)
    seen = 0
    mean = 0.0
    m2 = 0.0
    for start in range(0, n, _SIGNATURE_CHUNK_ELEMENTS):
        cpu = flat[start:start + _SIGNATURE_CHUNK_ELEMENTS].to("cpu").contiguous()
        digest.update(memoryview(cpu.view(torch.uint8).reshape(-1).numpy()))
        values = cpu.to(torch.float64)
        chunk_n = values.numel()
        if chunk_n:
            chunk_mean = float(values.mean())
            chunk_m2 = float((values - chunk_mean).square().sum())
            delta = chunk_mean - mean
            combined = seen + chunk_n
            mean += delta * chunk_n / combined
            m2 += chunk_m2 + delta * delta * seen * chunk_n / combined
            seen = combined
        _add_projection_chunk(values, start, totals, counts, 32)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype).removeprefix("torch."),
        "numel": n,
        "mean": round(mean, 6) if n else 0.0,
        "std": round((m2 / (n - 1)) ** 0.5, 6) if n > 1 else 0.0,
        "sha256": digest.hexdigest(),
        "projection": _finish_projection(totals, counts),
    }


def status_impl(worker) -> dict:
    # Read the event sequence and tail atomically so they describe one snapshot.
    from ..telemetry import event_snapshot

    event_seq, events = event_snapshot(32)
    info: dict[str, Any] = {
        "host": socket.gethostname(),
        "rank": worker.rank,
        "world": worker.world,
        "topology": dict(worker.topology),
        "models": worker.store.snapshot(),
        "nccl": False,
        # A failed setup/teardown rollback is a persistent dirty-state latch.
        # Publish only its boolean verdict: the exception detail stays in the
        # worker log, while telemetry can still refuse to call the fleet ready.
        "setup_cleanup_failed": bool(
            getattr(worker, "_setup_cleanup_failed", False)),
        "event_seq": event_seq,
        "events": events,
    }
    attention = getattr(worker, "_attn", None)
    attention_attestation = getattr(attention, "attestation", lambda: None)()
    if attention_attestation is not None:
        info["attention_attestation"] = attention_attestation
    try:
        info["source_manifest_sha256"] = source_manifest_sha256()
    except OSError:
        info["source_manifest_sha256"] = "unavailable"
    if worker._setup_key is None:
        # CUDA access here would initialize the default device before setup can
        # apply CUDA_VISIBLE_DEVICES.
        info["vram"] = None
        info["note"] = "not set up yet (VRAM stats available after the Init node runs)"
        return info
    try:
        import torch
        import torch.distributed as dist

        info["nccl"] = bool(dist.is_initialized())
        free, total = torch.cuda.mem_get_info()
        info["vram"] = {
            "free_gib": round(free / 2**30, 1),
            "total_gib": round(total / 2**30, 1),
            "allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 1),
            "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 1),
        }
        # Report live ComfyUI memory posture so status can detect drift from the
        # validated unified-memory baseline.
        import comfy.model_management as _mm
        _pin = getattr(_mm, "MAX_PINNED_MEMORY", None)
        info["mem_config"] = {
            "num_streams": int(getattr(_mm, "NUM_STREAMS", -1)),
            "pinned_disabled": _pin is not None and _pin < 0,
            "reserve_vram_gib": round(int(getattr(_mm, "EXTRA_RESERVED_VRAM", 0)) / 2**30, 1),
        }
    except Exception as exc:
        info["vram_error"] = failure_summary(exc)
    try:
        from ..telemetry import host_stats

        info["host"] = host_stats()
    except Exception as exc:
        info["host_error"] = failure_summary(exc)
    # Isolate unified-memory detail so its failure cannot hide VRAM telemetry.
    try:
        info["memory"] = worker._memory_detail()
    except Exception as exc:
        info["memory_error"] = failure_summary(exc)
    return info


def provenance_baseline_impl(
    worker, setup_generation: int, artifact_manifest: object = None
) -> dict:
    """Capture a strict setup-bound source and event watermark.

    The endpoint runs behind the GPU lock and single-thread executor. Generation
    checks reject delayed calls after setup replacement. Rank, world, and source
    identify the participant without exposing its checkout path.
    """
    if (isinstance(setup_generation, bool)
            or not isinstance(setup_generation, int)
            or setup_generation < 1):
        raise ValueError("provenance baseline requires a positive setup generation")
    actual_generation = getattr(worker, "_setup_generation", None)
    if worker._setup_key is None or actual_generation is None:
        raise RuntimeError("provenance baseline requires a READY worker setup")
    if setup_generation != actual_generation:
        raise RuntimeError(
            "provenance baseline setup generation is stale "
            f"(expected {setup_generation}, actual {actual_generation})"
        )
    rank, world = worker.rank, worker.world
    if (isinstance(rank, bool) or not isinstance(rank, int)
            or isinstance(world, bool) or not isinstance(world, int)
            or not 0 <= rank < world):
        raise RuntimeError("provenance baseline requires published rank/world metadata")

    from ..telemetry import event_sequence

    if artifact_manifest is None:
        return {
            "rank": rank,
            "world": world,
            "topology": dict(worker.topology),
            "setup_generation": actual_generation,
            "source_manifest_sha256": source_manifest_sha256(),
            "event_seq": event_sequence(),
        }

    from .comfy_bridge import _CUSTOM_NODES_DISABLED

    if _CUSTOM_NODES_DISABLED is not True:
        raise RuntimeError(
            "strict provenance requires custom nodes disabled at bootstrap"
        )
    stored = getattr(worker, "_setup_provenance", None)
    if (
        not isinstance(stored, tuple)
        or len(stored) != 2
        or stored[0] != setup_generation
        or not isinstance(stored[1], dict)
    ):
        raise RuntimeError("provenance baseline lacks its READY setup snapshot")
    runtime_provenance = runtime_provenance_snapshot(artifact_manifest)
    dgx_provenance = runtime_provenance.get("dgx_monarch")
    if not isinstance(dgx_provenance, dict):
        raise RuntimeError("runtime provenance lacks dgx_monarch source facts")
    return {
        "rank": rank,
        "world": world,
        "topology": dict(worker.topology),
        "setup_generation": actual_generation,
        "source_manifest_sha256": dgx_provenance["source_manifest_sha256"],
        "setup_provenance": stored[1],
        "post_provenance": runtime_provenance,
        # Capture last so earlier GPU-queue work precedes this watermark.
        "event_seq": event_sequence(),
    }


def memory_detail(worker) -> dict:
    import comfy.memory_management as cmm
    import comfy.utils
    import safetensors

    from . import comfy_dynamic, store_residency

    d: dict = {
        "safetensors_version": safetensors.__version__,
        "safetensors_backend": "pread" if getattr(safetensors.safe_open, "_dgxm_pread", False) else "mmap",
        # ComfyUI's own global confirms the worker's bootstrap state.
        "aimdo_enabled": bool(getattr(cmm, "aimdo_enabled", False)),
        "disable_mmap": bool(getattr(comfy.utils, "DISABLE_MMAP", False)),
        **comfy_dynamic.snapshot(),
    }
    # Private dirty memory is the non-reclaimable unified-pool footprint.
    try:
        with open(f"/proc/{os.getpid()}/smaps_rollup") as f:
            roll = {}
            for line in f:
                k, _, rest = line.partition(":")
                if k in ("Rss", "Pss", "Private_Dirty"):
                    roll[k] = round(int(rest.split()[0]) / 2**20, 2)
        d["proc_rss_gib"] = roll.get("Rss")
        d["proc_pss_gib"] = roll.get("Pss")
        d["proc_private_dirty_gib"] = roll.get("Private_Dirty")
    except OSError:
        pass
    stored = worker.store.current
    if stored is not None:
        patcher = stored.active_patcher
        d["patcher_class"] = type(patcher).__name__
        d["model"] = stored.base_key[0] if stored.base_key else None
        d["quant"] = stored.quant_kind
        d["lora_count"] = len(stored.lora_sig)  # count only, never names
        # Report actual backing, not requested policy.
        d["lora_mode"] = (
            "low_rss" if stored.unbake is not None
            else "hot_swap" if stored.lora_sig
            else "n/a"
        )
        # Auto discovery and unsupported-dtype fallback may remain stock while
        # slab policy stays enabled.
        d["weight_residency"] = (
            residency_mode.MODE_COMFY_MANAGED
            if getattr(stored, "residency_rung", "") == store_residency.RUNG_COMFY_MANAGED
            else "slab" if stored.slab is not None else "cudaMalloc")
        if stored.slab is not None:
            d.update(stored.slab.telemetry())
        vals = list(getattr(patcher, "backup", {}).values())  # snapshot: status is lock-free
        backup_bytes = sum(
            w.numel() * w.element_size()
            for bk in vals if (w := getattr(bk, "weight", None)) is not None and hasattr(w, "numel")
        )
        d["lora_backup_keys"] = len(vals)
        d["lora_backup_gib"] = round(backup_bytes / 2**30, 2)
        rec = getattr(stored, "unbake", None)
        if rec is not None:
            # Separate file-backed restore data from resident fallback snapshots.
            d["unbake_file_backed_keys"] = len(rec.mapped) + len(rec.quant)
            d["unbake_file_backed_gib"] = round(rec.mapped_bytes / 2**30, 2)
            d["unbake_resident_keys"] = rec.resident_count
            d["unbake_resident_gib"] = round(rec.resident_bytes / 2**30, 2)
        d["swap_verify_keys"] = worker.store.swap_verify
        d["swap_verify_failures"] = worker.store.verify_failures
        d["uma_reserve_gb"] = float(getattr(worker, "uma_reserve_gb", 0.0) or 0.0)
        mlw = getattr(patcher.model, "model_loaded_weight_memory", None)
        d["model_loaded_weight_gib"] = round(mlw / 2**30, 2) if mlw else None
        d["full_load"] = not bool(getattr(patcher.model, "model_lowvram", False))
    return d


def check_uma_reserve(worker) -> None:
    """Warn when a load or swap breaches reserved unified-memory headroom."""
    reserve = float(getattr(worker, "uma_reserve_gb", 0.0) or 0.0)
    if reserve <= 0:
        return
    try:
        avail = None
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                avail = int(line.split()[1]) / 2**20
                break
        if avail is not None and avail < reserve:
            log.warning(
                "UMA reserve breached: %.1f GiB available, under the %.1f GiB reserved "
                "for co-resident processes such as an LLM server; those processes now risk "
                "an OOM kill. Use a smaller quant, lora_low_rss=on, or a lower "
                "resolution.", avail, reserve)
            from ..telemetry import emit

            emit("uma_reserve_breach", available_gib=round(avail, 1),
                 reserve_gib=reserve)
    except Exception:
        pass
