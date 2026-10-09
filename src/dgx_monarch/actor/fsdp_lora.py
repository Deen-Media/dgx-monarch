"""LoRA on FSDP-sharded weights: the full-then-slice bake.

Under FSDP every parameter is a DTensor whose local tensor is this rank's
dim-0 chunk, a view of the buffer FSDP2 all-gathers from
(``torch/distributed/fsdp/_fully_shard/_fsdp_param.py`` ``_init_sharded_param``).
Comfy's LoRA merge wants a whole weight: ``calculate_weight`` runs on the full
tensor and ``stochastic_rounding`` draws a key-seeded stream over the full
shape, so the only way to reproduce the resident bake exactly is to compute
the full patched weight and slice it afterwards. Each rank therefore:

1. reads the pristine full weight from the checkpoint file with one pread (a
   fresh CPU tensor, freed after the key),
2. runs comfy's own math on it (``comfy.lora.calculate_weight`` in the LoRA
   compute dtype on ``load_device``, then the key-seeded rounding to the
   storage dtype, as ``store_bake.bake_key`` does),
3. takes ``torch.chunk(out, world, dim=0)[rank]``, the chunking FSDP2 used at
   wrap time, and copies it into the local shard.

No collective runs, every rank computes the same full tensor, and the rounding
stream is the resident one. Un-baking never needs a backup: a key that leaves
the stack is restored from the file, and because a dim-0 chunk of a row-major
tensor is one contiguous byte range, that restore is a single pread of this
rank's rows. Comfy's load and backup machinery never touches a sharded model,
so ``lora_low_rss`` is a launch requirement, not a preference.

The ambient verify runs comfy's own ``patch_weight_to_device(return_weight=True)``
against the pristine full weight by parking it on the module for the call and
restoring the identical DTensor parameter object afterwards, so FSDP's
bookkeeping (which holds that object) sees nothing change.

LoRA on quantized shards is not admitted: a quantized key refuses before any
write.
"""
from __future__ import annotations

import math
import os
from typing import Any

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from ..transfer_utils import failure_summary, safe_call
from .unbake import UnbakeError, UnbakeRecord, _is_quantized, live_tensor
from .unbake_file import (
    _ST_TO_TORCH,
    _candidate_file_keys,
    _FileTensor,
    _fingerprint_windows,
    read_safetensors_header,
)

log = get_logger(__name__)


def shard_rows(rows: int, world: int, rank: int) -> tuple[int, int]:
    """[start, stop) rows this rank holds, with torch.chunk's dim-0 semantics."""
    if rows <= 0 or world <= 0:
        return 0, 0
    chunk = math.ceil(rows / world)
    start = min(rank * chunk, rows)
    stop = min(start + chunk, rows)
    return start, stop


def local_shard(param: torch.Tensor) -> torch.Tensor:
    """The rank's unpadded local chunk of a sharded parameter."""
    local = getattr(param, "_local_tensor", None)
    if local is None:
        raise UnbakeError("parameter is not FSDP-sharded (no local tensor)")
    return local


def mesh_rank_world() -> tuple[int, int]:
    import torch.distributed as dist

    if not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def write_shard(param: torch.Tensor, full: torch.Tensor, rank: int, world: int) -> None:
    """Copy this rank's dim-0 chunk of ``full`` into the sharded parameter."""
    local = local_shard(param)
    start, stop = shard_rows(int(full.shape[0]) if full.ndim else 0, world, rank)
    chunk = full[start:stop]
    if tuple(chunk.shape) != tuple(local.shape):
        raise UnbakeError(
            f"shard geometry drift: rank {rank}/{world} expects local shape "
            f"{tuple(local.shape)} but the full tensor chunks to {tuple(chunk.shape)}"
        )
    with torch.no_grad():
        local.copy_(chunk.to(device=local.device, dtype=local.dtype))


def _file_tensor(header: dict[str, _FileTensor], key: str, param: torch.Tensor) -> _FileTensor:
    for cand in _candidate_file_keys(key):
        ft = header.get(cand)
        if ft is None:
            continue
        if ft.dtype != param.dtype or ft.shape != tuple(param.shape):
            # This is the in-bake backstop for an fp32 island cast live, which the
            # header check (adapters.detect.fsdp_lora_admission_property) cannot
            # predict. Every rank loads the same file and casts it the same way, so
            # the class P tag is rank symmetric (docs/DESIGN.md section 5.4).
            raise UnbakeError(refusal(
                RefusalClass.PHYSICS,
                f"checkpoint tensor {cand} is {ft.dtype}/{ft.shape} but the live "
                f"parameter is {param.dtype}/{tuple(param.shape)}; a LoRA under FSDP "
                "needs the file's own dtype (no live cast). Use a resident topology "
                "for this file, or a bf16 artifact with no fp32 islands."
            ))
        return ft
    raise UnbakeError(f"no checkpoint tensor for key {key}")


def _torch_dtype(ft: _FileTensor) -> torch.dtype:
    dtype = _ST_TO_TORCH.get(ft.dtype) if isinstance(ft.dtype, str) else ft.dtype
    if not isinstance(dtype, torch.dtype):
        raise UnbakeError(f"unsupported checkpoint dtype {ft.dtype!r}")
    return dtype


def read_full(fd: int, ft: _FileTensor) -> torch.Tensor:
    """The whole pristine tensor, as a fresh CPU tensor."""
    raw = os.pread(fd, ft.nbytes, ft.start)
    if len(raw) != ft.nbytes:
        raise UnbakeError("short read of a checkpoint tensor")
    dtype = _torch_dtype(ft)
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(ft.shape)


def read_rows(fd: int, ft: _FileTensor, start: int, stop: int) -> torch.Tensor:
    """Rows [start, stop) of a row-major tensor: one contiguous pread."""
    rows = ft.shape[0] if ft.shape else 1
    row_bytes = ft.nbytes // max(rows, 1)
    length = max(stop - start, 0) * row_bytes
    dtype = _torch_dtype(ft)
    tail = tuple(ft.shape[1:])
    if length == 0:
        return torch.empty((0, *tail), dtype=dtype)
    raw = os.pread(fd, length, ft.start + start * row_bytes)
    if len(raw) != length:
        raise UnbakeError("short read of a checkpoint shard")
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape((stop - start, *tail))


def _shard_bytes_match(fd: int, ft: _FileTensor, param: torch.Tensor, rank: int, world: int) -> bool:
    """Sampled byte equality between this rank's rows on disk and its shard."""
    local = local_shard(param)
    rows = ft.shape[0] if ft.shape else 1
    start, stop = shard_rows(rows, world, rank)
    row_bytes = ft.nbytes // max(rows, 1)
    nbytes = (stop - start) * row_bytes
    if nbytes != local.numel() * local.element_size():
        return False
    if nbytes == 0:
        return True
    flat = local.detach().reshape(-1).contiguous().view(torch.uint8)
    base = ft.start + start * row_bytes
    try:
        for offset, length in _fingerprint_windows(nbytes):
            ours = bytes(flat[offset:offset + length].cpu().numpy().tobytes())
            if os.pread(fd, length, base + offset) != ours:
                return False
    except (OSError, RuntimeError):
        return False
    return True


def capture_record(model: Any, keys, base_path: str, rank: int, world: int) -> UnbakeRecord:
    """An UnbakeRecord whose mapped keys are byte-verified against this rank's rows."""
    header, _metadata, parsed = read_safetensors_header(
        base_path, want_metadata=True, want_identity=True)
    record = UnbakeRecord(
        path=base_path, file_size=parsed.file_size, file_mtime_ns=parsed.file_mtime_ns,
        file_dev=parsed.file_dev, file_ino=parsed.file_ino, file_ctime_ns=parsed.file_ctime_ns)
    with open(base_path, "rb") as f:
        fd = f.fileno()
        for key in keys:
            param = live_tensor(model, key)
            if _is_quantized(param) or _is_quantized(getattr(param, "_local_tensor", None)):
                # This is the backstop for a quantized checkpoint the header preflight
                # (adapters.detect.fsdp_lora_admission_property) misses. Every rank sees
                # the same wrapper on the same key, so the class P tag is rank symmetric
                # (docs/DESIGN.md section 5.4).
                raise UnbakeError(refusal(
                    RefusalClass.PHYSICS,
                    f"quantized key {key} under FSDP: LoRA on quantized shards is "
                    "not admitted. Use a resident topology for this file, or a "
                    "bf16, fp16, or plain-dtype fp8 checkpoint under FSDP."))
            ft = _file_tensor(header, key, param)
            if not _shard_bytes_match(fd, ft, param, rank, world):
                raise UnbakeError(
                    f"live shard of {key} does not match the checkpoint bytes on rank "
                    f"{rank}; the model is not pristine, reload contract applies")
            record.mapped[key] = ft
    return record


def bake_key(active, key: str, pristine_full: torch.Tensor, rank: int, world: int) -> None:
    """comfy's merge on the pristine full weight, sliced into this rank's shard."""
    import comfy.float
    import comfy.lora
    import comfy.model_management as mm
    import comfy.model_patcher as comfy_mp
    import comfy.utils

    patches = active.patches[key]
    weight, set_func, convert_func = comfy_mp.get_key_weight(active.model, key)
    if set_func is not None or convert_func is not None:
        # As in capture_record's quantized-key check, the class P tag is rank
        # symmetric: a quantized setter on this key never varies by rank or run.
        raise UnbakeError(refusal(
            RefusalClass.PHYSICS,
            f"key {key} carries a quantized setter; LoRA on quantized shards is "
            "not admitted. Use a resident topology for this file, or a bf16, "
            "fp16, or plain-dtype fp8 checkpoint under FSDP."))
    bake_device = getattr(active, "load_device", None) or local_shard(weight).device
    temp = mm.cast_to_device(
        pristine_full, bake_device, mm.lora_compute_dtype(bake_device), copy=True)
    out = comfy.lora.calculate_weight(patches, temp, key)
    out = comfy.float.stochastic_rounding(
        out, weight.dtype, seed=comfy.utils.string_to_seed(key))
    write_shard(weight, out, rank, world)
    del temp, out


def verify_key(active, key: str, pristine_full: torch.Tensor, rank: int, world: int) -> None:
    """comfy's own reference bake, sliced, must bit-match the written shard."""
    import comfy.utils

    module_key, _, attr = key.rpartition(".")
    module = comfy.utils.get_attr(active.model, module_key) if module_key else active.model
    sharded = getattr(module, attr)
    if not isinstance(sharded, torch.nn.Parameter):
        raise UnbakeError(f"key {key} is not a parameter")
    device = getattr(active, "load_device", None) or local_shard(sharded).device
    reference_param = torch.nn.Parameter(
        pristine_full.to(device=device), requires_grad=False)
    # Park the pristine full weight on the module for comfy's reference
    # computation; the identical DTensor Parameter object goes back afterwards.
    module._parameters[attr] = reference_param
    try:
        ref = active.patch_weight_to_device(key, device_to=device, return_weight=True)
    finally:
        module._parameters[attr] = sharded
    if _is_quantized(ref):
        raise UnbakeError(f"comfy's reference for {key} is quantized; LoRA on quantized shards is not admitted")
    start, stop = shard_rows(int(ref.shape[0]) if ref.ndim else 0, world, rank)
    ours = local_shard(sharded)
    ref_chunk = ref[start:stop].to(device=ours.device)
    if ref.dtype != ours.dtype or not torch.equal(ref_chunk, ours):
        raise UnbakeError(
            f"ambient bake verify MISMATCH on {key} (FSDP shard): this bake does not "
            "bit-match comfy's own computation, which usually means a ComfyUI update "
            "changed bake semantics. The worker discards the model. A reload bakes "
            "through this same shard path, because comfy's stock merge cannot patch a "
            "sharded weight. Use a resident topology for this file to load through "
            "comfy's stock path instead, and report the mismatch with your ComfyUI commit.")
    del ref, reference_param


def _touched_keys(*records: UnbakeRecord | None) -> set[str]:
    out: set[str] = set()
    for record in records:
        if record is not None:
            out.update(record.mapped)
    return out


def apply_stack(store, active, base_path: str, previous: UnbakeRecord | None) -> UnbakeRecord:
    """Restore keys that left the stack, bake the keys in it, verify a sample.

    Returns the record for the keys now baked. Raises ``UnbakeError`` on any
    inconsistency, after which the caller discards the slot and reloads.
    """
    import random

    rank, world = mesh_rank_world()
    patch_keys = list(active.patches)
    _header, _metadata, parsed = read_safetensors_header(
        base_path, want_metadata=True, want_identity=True)
    if previous is not None and not previous.stat_ok():
        raise UnbakeError(f"checkpoint {base_path} changed since capture")
    # Keys the new stack does not touch must go back to pristine first.
    leaving = sorted(_touched_keys(previous) - set(patch_keys))
    novel = [k for k in patch_keys if previous is None or k not in previous.mapped]
    fresh = capture_record(active.model, novel, base_path, rank, world) if novel else None
    record = UnbakeRecord(
        path=base_path, file_size=parsed.file_size, file_mtime_ns=parsed.file_mtime_ns,
        file_dev=parsed.file_dev, file_ino=parsed.file_ino, file_ctime_ns=parsed.file_ctime_ns)
    for key in patch_keys:
        for source in (previous, fresh):
            if source is not None and key in source.mapped:
                record.mapped[key] = source.mapped[key]
                break
        else:
            raise UnbakeError(f"patched key {key} has no checkpoint binding")
    n_verify = (len(patch_keys) if store.swap_verify < 0
                else min(store.swap_verify, len(patch_keys)))
    verify_keys = set(random.sample(patch_keys, n_verify)) if n_verify else set()
    with open(base_path, "rb") as f:
        fd = f.fileno()
        for key in leaving:
            ft = previous.mapped[key]  # type: ignore[union-attr]
            param = live_tensor(active.model, key)
            rows = ft.shape[0] if ft.shape else 1
            start, stop = shard_rows(rows, world, rank)
            chunk = read_rows(fd, ft, start, stop)
            local = local_shard(param)
            with torch.no_grad():
                local.copy_(chunk.to(device=local.device, dtype=local.dtype))
        for key in patch_keys:
            ft = record.mapped[key]
            pristine = read_full(fd, ft)
            bake_key(active, key, pristine, rank, world)
            if key in verify_keys:
                verify_key(active, key, pristine, rank, world)
            del pristine
    if verify_keys:
        log.info("ambient verify (FSDP shards): %d/%d keys bit-matched comfy's own bake (%s)",
                 len(verify_keys), len(patch_keys), ", ".join(sorted(verify_keys)))
    active.patches.clear()
    active.backup.clear()
    active.backup_buffers.clear()
    log.info("LoRA on FSDP shards: baked %d key(s), restored %d, rank %d/%d",
             len(patch_keys), len(leaving), rank, world)
    return record


def bake_stack(store, active, base_path: str) -> UnbakeRecord | None:
    """First bake of a stack onto a freshly sharded base (replaces merge-and-free)."""
    try:
        return apply_stack(store, active, base_path, None)
    except UnbakeError:
        raise
    except Exception as exc:
        safe_call(log.warning, "LoRA on FSDP shards: bake failed (%s)", failure_summary(exc))
        raise UnbakeError(f"FSDP LoRA bake failed: {failure_summary(exc)}") from exc


def lazy_swap(store, stored, active, unet_name: str) -> UnbakeRecord | None:
    """store_bake.lazy_swap for FSDP shards: restore leavers, bake the new stack."""
    from . import model_store as _ms

    base_path = _ms.resolve_model_path("diffusion_models", unet_name)
    record = apply_stack(store, active, base_path, stored.unbake)
    return record if record.mapped else None
