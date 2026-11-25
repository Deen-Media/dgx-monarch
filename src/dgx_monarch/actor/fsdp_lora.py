"""LoRA on FSDP-sharded weights: the full-then-slice bake.

Under FSDP every sharded parameter is a DTensor whose local tensor is this rank's
dim-0 chunk, a view of the buffer FSDP2 all-gathers from
(``torch/distributed/fsdp/_fully_shard/_fsdp_param.py`` ``_init_sharded_param``).
Comfy's LoRA merge wants a whole weight: ``calculate_weight`` runs on the full
tensor and ``stochastic_rounding`` draws a key-seeded stream over the full
shape, so the only way to reproduce the resident bake exactly is to compute
the full patched weight and slice it afterwards. Each rank therefore:

1. reads the pristine full weight from the checkpoint file with one pread (a
   fresh CPU tensor, freed after the key),
2. runs the bake ``store_bake.bake_key`` runs on it, on ``load_device``:
   stock's cast-time math where stock has one weight for the key
   (``actor/lora_cast_time.py``), else comfy's classic merge in the LoRA
   compute dtype with the key-seeded rounding to the storage dtype,
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

A census module, one a family runs outside its diffusion forward
(``adapters/fsdp_islands.auxiliary_module_parameters``), is not sharded: the
wrap passes its parameters to ``fully_shard`` as ``ignored_params`` and they
stay plain tensors. A LoRA key on one bakes as a world-1 shard: the same
pristine read, the same comfy merge and key-seeded rounding, written whole on
every rank, which is the resident bake on each rank. The replicated set comes
from the call the wrap makes, once per stack, so the two cannot drift. Any
other plain tensor still raises.

LoRA on quantized shards is not admitted: a quantized key refuses before any
write.

A key ComfyUI renames as it loads the file (FLUX stores a q/k norm weight as
``<...>_norm.scale``; the model and a LoRA call it ``<...>_norm.weight``) is
read under the file's name, taken from ComfyUI's own rename
(``actor/load_renames.py``) and byte-verified like any other key. A key the
file holds under no name refuses class P before any write.
"""
from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any, NoReturn

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from ..transfer_utils import failure_summary, safe_call
from . import lora_cast_time
from .fsdp_lora_file import (
    _bound_to,
    _BoundCheckpoint,
    _file_tensor,
    _files_identical,
    _identity,
    _shard_bytes_match,
    _whole,
    local_shard,
    read_full,
    read_rows,
    shard_geometry,
    shard_rows,
)
from .load_renames import load_time_renames
from .unbake import UnbakeError, UnbakeRecord, UnbakeRefused, _is_quantized, live_tensor
from .unbake_file import _FileTensor, read_safetensors_header

__all__ = ["apply_stack", "bake_key", "bake_stack", "capture_record", "lazy_swap", "local_shard",
           "mesh_rank_world", "read_full", "read_rows", "replicated_parameter_ids", "shard_geometry",
           "shard_rows", "verify_key", "write_shard"]

log = get_logger(__name__)


def replicated_parameter_ids(model: Any) -> frozenset[int]:
    """Ids of the parameters the FSDP wrap left whole, for this model.

    ``model`` is the comfy ``model_base`` the patcher holds; the wrap reads
    the same ``replicated_parameters(diffusion_model, model)``
    (``adapters/fsdp.py``): the census modules, the fp32 islands and every
    zero-dimensional parameter. Ids, because a tensor's ``==`` is elementwise.
    """
    from ..adapters.fsdp_islands import replicated_parameters

    diffusion_model = getattr(model, "diffusion_model", None)
    if diffusion_model is None:
        return frozenset()
    return frozenset(id(p) for p in replicated_parameters(diffusion_model, model))


def mesh_rank_world() -> tuple[int, int]:
    import torch.distributed as dist

    if not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def write_shard(
    param: torch.Tensor, full: torch.Tensor, rank: int, world: int,
    replicated: frozenset[int] = frozenset(),
) -> None:
    """Copy this rank's dim-0 chunk of ``full`` into the sharded parameter,
    or, for a census-replicated parameter, the whole tensor."""
    local = local_shard(param, replicated)
    rank, world = shard_geometry(param, replicated, rank, world)
    start, stop = shard_rows(int(full.shape[0]) if full.ndim else 0, world, rank)
    # A whole parameter takes the whole tensor; a zero-dimensional one has no rows to slice.
    chunk = full if world == 1 else full[start:stop]
    if tuple(chunk.shape) != tuple(local.shape):
        raise UnbakeError(
            f"shard geometry drift: rank {rank}/{world} expects local shape "
            f"{tuple(local.shape)} but the full tensor chunks to {tuple(chunk.shape)}"
        )
    with torch.no_grad():
        local.copy_(chunk.to(device=local.device, dtype=local.dtype))


def _refuse_quantized(key: str) -> NoReturn:
    # This is the backstop for a quantized checkpoint the header preflight
    # (adapters.detect.fsdp_lora_admission_property) misses. Every rank sees
    # the same wrapper on the same key, so the class P tag is rank symmetric
    # (docs/DESIGN.md section 5.4).
    raise UnbakeRefused(refusal(
        RefusalClass.PHYSICS,
        f"quantized key {key} under FSDP: LoRA on quantized shards is "
        "not admitted. Use a resident topology for this file, or a "
        "bf16, fp16, or plain-dtype fp8 checkpoint under FSDP."))


def capture_record(
    model: Any, keys, base_path: str, rank: int, world: int,
    replicated: frozenset[int] = frozenset(),
    checkpoint: _BoundCheckpoint | None = None,
) -> UnbakeRecord:
    """An UnbakeRecord whose mapped keys are byte-verified against this rank's rows.

    ``checkpoint`` is the descriptor ``apply_stack`` opened and bound to the
    header it read; without one the capture binds its own.
    """
    if checkpoint is None:
        with _bound_checkpoint(base_path) as own:
            return capture_record(model, keys, base_path, rank, world, replicated, own)
    header, parsed, fd = checkpoint.header, checkpoint.parsed, checkpoint.fd
    record = UnbakeRecord(
        path=base_path, file_size=parsed.file_size, file_mtime_ns=parsed.file_mtime_ns,
        file_dev=parsed.file_dev, file_ino=parsed.file_ino, file_ctime_ns=parsed.file_ctime_ns)
    # A key ComfyUI renames as it loads the file is found under the file's name;
    # the byte check below still has to pass on it.
    renames = load_time_renames(model, header)
    bound: list[tuple[str, torch.Tensor, _FileTensor]] = []
    for key in keys:
        param = live_tensor(model, key)
        if _is_quantized(param) or _is_quantized(getattr(param, "_local_tensor", None)):
            _refuse_quantized(key)
        bound.append((key, param, _file_tensor(header, key, param, renames,
                                               identical=lambda a, b: _files_identical(fd, a, b))))
    # Every refusal above reads the file and the model's structure, the same on
    # every rank. The rows differ by rank, so they are checked only once every
    # key is bound: a rank whose rows fail cannot raise ahead of a refusal its
    # peer meets on a later key.
    for key, param, ft in bound:
        if not _shard_bytes_match(fd, ft, param, rank, world, replicated):
            raise UnbakeError(
                f"live shard of {key} does not match the checkpoint bytes on rank "
                f"{rank}; the model is not pristine, reload contract applies")
        record.mapped[key] = ft
    return record


@contextlib.contextmanager
def _bound_checkpoint(base_path: str, expected: Any = None) -> Iterator[_BoundCheckpoint]:
    """Read the header, open the file once, and bind every read to that descriptor.

    The descriptor and the path must both name the inode, size and times the
    header read reported, before the first read and again after the last, as
    ``unbake.restore_pristine`` checks its own reads; an atomic replacement or
    an in-place rewrite in between refuses instead of baking another file's
    bytes at the old offsets. ``expected``, when given, is the identity of the
    file the model was loaded from (its FSDP checkpoint pin). On the way out
    the checkpoint's clean pages are dropped from the page cache: the bake
    reads each patched key whole, and on unified memory those pages sit beside
    the shards until the kernel reclaims them.
    """
    header, _metadata, parsed = read_safetensors_header(
        base_path, want_metadata=True, want_identity=True)
    want = _identity(parsed)
    fd = os.open(base_path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        if not _bound_to(fd, base_path, want):
            raise UnbakeError(
                f"checkpoint {base_path} changed after its header was read; "
                "reload contract applies")
        if expected is not None and _identity(expected) != want:
            raise UnbakeError(
                f"checkpoint {base_path} changed since this model was loaded from it (its "
                "device, inode, size, modification time or change time differ); reload "
                "contract applies")
        yield _BoundCheckpoint(fd, header, parsed)
        if not _bound_to(fd, base_path, want):
            raise UnbakeError(
                f"checkpoint {base_path} changed while LoRA bake reads were in progress; "
                "reload contract applies")
    finally:
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass
        os.close(fd)


def bake_key(
    active, key: str, pristine_full: torch.Tensor, rank: int, world: int,
    replicated: frozenset[int] = frozenset(),
) -> None:
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
    bake_device = getattr(active, "load_device", None) or local_shard(weight, replicated).device
    # The resident weight is the file tensor cast to the live dtype, as a
    # restore writes it; both bakes start there, and so does verify_key.
    pristine = pristine_full.to(dtype=local_shard(weight, replicated).dtype)
    # Stock's cast-time weight where it has one (#145), else the classic bake.
    out = lora_cast_time.cast_time_weight(active, key, bake_device, source=pristine)
    if out is None:
        temp = mm.cast_to_device(
            pristine, bake_device, mm.lora_compute_dtype(bake_device), copy=True)
        out = comfy.lora.calculate_weight(patches, temp, key)
        out = comfy.float.stochastic_rounding(
            out, weight.dtype, seed=comfy.utils.string_to_seed(key))
        del temp
    write_shard(weight, out, rank, world, replicated)
    del out


def verify_key(
    active, key: str, pristine_full: torch.Tensor, rank: int, world: int,
    replicated: frozenset[int] = frozenset(),
) -> None:
    """comfy's own reference bake, sliced, must bit-match the written shard."""
    import comfy.utils

    module_key, _, attr = key.rpartition(".")
    module = comfy.utils.get_attr(active.model, module_key) if module_key else active.model
    sharded = getattr(module, attr)
    if not isinstance(sharded, torch.nn.Parameter):
        raise UnbakeError(f"key {key} is not a parameter")
    device = getattr(active, "load_device", None) or local_shard(sharded, replicated).device
    reference_param = torch.nn.Parameter(pristine_full.to(
        device=device, dtype=local_shard(sharded, replicated).dtype), requires_grad=False)
    # Park the pristine full weight on the module for comfy's reference
    # computation; the identical DTensor Parameter object goes back afterwards.
    module._parameters[attr] = reference_param
    try:
        ref = active.patch_weight_to_device(key, device_to=device, return_weight=True)
    finally:
        module._parameters[attr] = sharded
    if _is_quantized(ref):
        raise UnbakeError(f"comfy's reference for {key} is quantized; LoRA on quantized shards is not admitted")
    rank, world = shard_geometry(sharded, replicated, rank, world)
    start, stop = shard_rows(int(ref.shape[0]) if ref.ndim else 0, world, rank)
    ours = local_shard(sharded, replicated)
    ref_chunk = (ref if world == 1 else ref[start:stop]).to(device=ours.device)
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


def _refuse_quantized_setters(active, patch_keys) -> None:
    """``bake_key``'s quantized-setter refusal, raised for every key before the first write."""
    if not patch_keys:
        return
    import comfy.model_patcher as comfy_mp

    for key in patch_keys:
        _weight, set_func, convert_func = comfy_mp.get_key_weight(active.model, key)
        if set_func is not None or convert_func is not None:
            _refuse_quantized(key)


def apply_stack(store, active, base_path: str, previous: UnbakeRecord | None,
                expected: Any = None) -> UnbakeRecord:
    """Restore keys that left the stack, bake the keys in it, verify a sample.

    Every check that can refuse the stack runs before the first write, so a
    refusal (``UnbakeRefused``) leaves the resident and its stack as they
    were. Every read goes through one descriptor bound to the header
    (``_bound_checkpoint``); ``expected`` is the identity of the file the model
    was loaded from, when the store holds one. Returns the record for the
    keys now baked. Raises ``UnbakeError`` on any other inconsistency, after
    which the caller discards the slot and reloads.
    """
    import random

    rank, world = mesh_rank_world()
    replicated = replicated_parameter_ids(active.model)
    patch_keys = list(active.patches)
    if previous is not None and not previous.stat_ok():
        raise UnbakeError(f"checkpoint {base_path} changed since capture")
    with _bound_checkpoint(base_path, expected) as checkpoint:
        parsed, fd = checkpoint.parsed, checkpoint.fd
        # Keys the new stack does not touch must go back to pristine first.
        leaving = sorted(_touched_keys(previous) - set(patch_keys))
        novel = [k for k in patch_keys if previous is None or k not in previous.mapped]
        fresh = (capture_record(active.model, novel, base_path, rank, world, replicated, checkpoint)
                 if novel else None)
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
        _refuse_quantized_setters(active, patch_keys)
        n_verify = (len(patch_keys) if store.swap_verify < 0
                    else min(store.swap_verify, len(patch_keys)))
        verify_keys = set(random.sample(patch_keys, n_verify)) if n_verify else set()
        # The first write is below this line.
        for key in leaving:
            ft = previous.mapped[key]  # type: ignore[union-attr]
            param = live_tensor(active.model, key)
            key_rank, key_world = shard_geometry(param, replicated, rank, world)
            if key_world == 1:
                chunk = read_full(fd, ft)  # a whole parameter, a zero-dimensional one included
            else:
                start, stop = shard_rows(ft.shape[0] if ft.shape else 1, key_world, key_rank)
                chunk = read_rows(fd, ft, start, stop)
            local = local_shard(param, replicated)
            with torch.no_grad():
                local.copy_(chunk.to(device=local.device, dtype=local.dtype))
        whole = 0
        for key in patch_keys:
            ft = record.mapped[key]
            pristine = read_full(fd, ft)
            whole += _whole(live_tensor(active.model, key), replicated)
            bake_key(active, key, pristine, rank, world, replicated)
            if key in verify_keys:
                verify_key(active, key, pristine, rank, world, replicated)
            del pristine
    if verify_keys:
        log.info("ambient verify (FSDP shards): %d/%d keys bit-matched comfy's own bake (%s)",
                 len(verify_keys), len(patch_keys), ", ".join(sorted(verify_keys)))
    active.patches.clear()
    active.backup.clear()
    active.backup_buffers.clear()
    log.info("LoRA on FSDP shards: baked %d key(s) (%d whole on a replicated census "
             "module), restored %d, rank %d/%d",
             len(patch_keys), whole, len(leaving), rank, world)
    return record


def _pinned_identity(pin: Any) -> Any:
    """The identity of the file an FSDP load pinned, or None."""
    return getattr(pin, "file_identity", None) if pin is not None else None


def bake_stack(store, active, base_path: str, pin: Any = None) -> UnbakeRecord | None:
    """First bake of a stack onto a freshly sharded base (replaces merge-and-free).

    ``pin`` is the load's checkpoint pin: the bake reads only the file the
    model was built from.
    """
    try:
        return apply_stack(store, active, base_path, None, _pinned_identity(pin))
    except UnbakeError:
        raise
    except Exception as exc:
        safe_call(log.warning, "LoRA on FSDP shards: bake failed (%s)", failure_summary(exc))
        raise UnbakeError(f"FSDP LoRA bake failed: {failure_summary(exc)}") from exc


def lazy_swap(store, stored, active, unet_name: str) -> UnbakeRecord | None:
    """store_bake.lazy_swap for FSDP shards: restore leavers, bake the new stack."""
    from . import model_store as _ms

    base_path = _ms.resolve_model_path("diffusion_models", unet_name)
    record = apply_stack(store, active, base_path, stored.unbake,
                         _pinned_identity(getattr(stored, "fsdp_checkpoint_pin", None)))
    return record if record.mapped else None
