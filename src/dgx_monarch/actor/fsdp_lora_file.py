"""The checkpoint side of the LoRA bake on FSDP shards (``actor/fsdp_lora.py``).

A rank's rows of a sharded weight, the file tensor a patched key binds to, the
reads that bring pristine bytes back, the byte checks that prove a shard
pristine, and the identity one open descriptor must keep for the whole bake.
``actor.fsdp_lora`` re-exports every public name here.
"""
from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from ..refusal import RefusalClass, refusal
from .load_renames import renamed_file_key
from .unbake import UnbakeError, UnbakeRefused
from .unbake_file import _ST_TO_TORCH, _candidate_file_keys, _FileTensor, _fingerprint_windows


def shard_rows(rows: int, world: int, rank: int) -> tuple[int, int]:
    """[start, stop) rows this rank holds, with torch.chunk's dim-0 semantics."""
    if rows <= 0 or world <= 0:
        return 0, 0
    chunk = math.ceil(rows / world)
    start = min(rank * chunk, rows)
    stop = min(start + chunk, rows)
    return start, stop


def _whole(param: torch.Tensor, replicated: frozenset[int]) -> bool:
    return getattr(param, "_local_tensor", None) is None and id(param) in replicated


def local_shard(param: torch.Tensor, replicated: frozenset[int] = frozenset()) -> torch.Tensor:
    """The rank's unpadded local chunk of a sharded parameter.

    A census-replicated parameter is its own whole local view; any other
    parameter with no local tensor raises.
    """
    local = getattr(param, "_local_tensor", None)
    if local is not None:
        return local
    if id(param) in replicated:
        return param
    raise UnbakeError("parameter is not FSDP-sharded (no local tensor)")


def shard_geometry(
    param: torch.Tensor, replicated: frozenset[int], rank: int, world: int,
) -> tuple[int, int]:
    """(rank, world) for this parameter's rows: a replicated one is a world-1 shard."""
    return (0, 1) if _whole(param, replicated) else (rank, world)


def _file_tensor(
    header: dict[str, _FileTensor], key: str, param: torch.Tensor,
    renames: dict[str, str] | None = None,
    identical: Callable[[_FileTensor, _FileTensor], bool] | None = None,
) -> _FileTensor:
    """The checkpoint tensor a patcher key restores from, chosen from the file alone.

    Every rank holds the same file and makes the same choice; nothing here
    reads a rank's rows, which differ by rank. ``renames`` is
    ``load_time_renames`` for this model and file, made by ComfyUI's own load
    function run over the file's names. Where it puts a renamed file tensor
    under the key (FLUX's ``<...>_norm.scale``, held as ``<...>_norm.weight``),
    that tensor is the one the model loaded, whatever else the file holds
    under the key's own name. Otherwise the candidates are the key's namespace
    spellings: one of the live dtype and shape is the one; several that
    ``identical`` finds equal in every byte serve alike; several that differ
    leave the file silent on which one ComfyUI loaded, and every rank refuses
    the same way before any weight is written. The caller then checks this
    rank's rows against the bound tensor.
    """
    renamed = renamed_file_key(renames, key) if renames else None
    names = [renamed] if renamed is not None and renamed in header else _candidate_file_keys(key)
    fitting: list[tuple[str, _FileTensor]] = []
    misfit: tuple[str, _FileTensor] | None = None
    for cand in names:
        ft = header.get(cand)
        if ft is None:
            continue
        if ft.dtype != param.dtype or ft.shape != tuple(param.shape):
            misfit = misfit or (cand, ft)
            continue
        fitting.append((cand, ft))
    if len(fitting) > 1 and (identical is None
                             or not all(identical(fitting[0][1], ft) for _, ft in fitting[1:])):
        # Every rank reads the same header and the same bytes, so the class P
        # tag is rank symmetric, and capture_record raises it before any shard
        # is written (docs/DESIGN.md section 5.4).
        raise UnbakeRefused(refusal(
            RefusalClass.PHYSICS,
            f"the checkpoint holds {key} under {len(fitting)} names "
            f"({', '.join(name for name, _ in fitting)}) whose tensors differ, and the file does "
            "not say which one ComfyUI loaded. A LoRA under FSDP bakes from the file and "
            "restores from it, so it needs to know. Use a resident topology for this stack "
            "(the same preset without +fsdp), or a checkpoint that holds each weight once."))
    if fitting:
        return fitting[0][1]
    if misfit is not None:
        cand, ft = misfit
        # This is the in-bake backstop for an fp32 island cast live, which the
        # header check (adapters.detect.fsdp_lora_admission_property) cannot
        # predict. Every rank loads the same file and casts it the same way, so
        # the class P tag is rank symmetric (docs/DESIGN.md section 5.4).
        raise UnbakeRefused(refusal(
            RefusalClass.PHYSICS,
            f"checkpoint tensor {cand} is {ft.dtype}/{ft.shape} but the live "
            f"parameter is {param.dtype}/{tuple(param.shape)}; a LoRA under FSDP "
            "needs the file's own dtype (no live cast). Use a resident topology "
            "for this file, or a bf16 artifact with no fp32 islands."
        ))
    # No tensor under the key's own name, a namespace spelling of it, or a name
    # ComfyUI renames to it at load (a weight ComfyUI builds from part of a
    # tensor lands here too). Under FSDP the file is the only pristine copy
    # (no backup exists for a sharded weight), so the bake cannot go on. Every
    # rank reads the same header for the same stack, so the class P tag is rank
    # symmetric, and capture_record raises it before any shard is written.
    raise UnbakeRefused(refusal(
        RefusalClass.PHYSICS,
        f"a LoRA in this stack patches {key}, and the checkpoint file holds no "
        "tensor under that name, and none that ComfyUI renames to it at load. A "
        "LoRA under FSDP bakes from the file and restores from it, so it needs one. "
        "Use a resident topology for this stack (the same preset without "
        "+fsdp, which keeps its own copy of the weight), or drop the LoRA that "
        "patches this key."))


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


def _shard_bytes_match(
    fd: int, ft: _FileTensor, param: torch.Tensor, rank: int, world: int,
    replicated: frozenset[int] = frozenset(),
) -> bool:
    """Sampled byte equality between this rank's rows on disk and its shard."""
    local = local_shard(param, replicated)
    rank, world = shard_geometry(param, replicated, rank, world)
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


_COMPARE_CHUNK = 64 << 20


def _files_identical(fd: int, a: _FileTensor, b: _FileTensor) -> bool:
    """Whether two checkpoint tensors hold the same bytes, read in bounded chunks."""
    if a.nbytes != b.nbytes:
        return False
    done = 0
    while done < a.nbytes:
        n = min(_COMPARE_CHUNK, a.nbytes - done)
        if os.pread(fd, n, a.start + done) != os.pread(fd, n, b.start + done):
            return False
        done += n
    return True


_IDENTITY_FIELDS = ("file_dev", "file_ino", "file_size", "file_mtime_ns", "file_ctime_ns")


def _identity(source: Any) -> tuple:
    return tuple(getattr(source, name) for name in _IDENTITY_FIELDS)


def _stat_identity(st: os.stat_result) -> tuple:
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _bound_to(fd: int, path: str, want: tuple) -> bool:
    try:
        return _stat_identity(os.fstat(fd)) == want and _stat_identity(os.stat(path)) == want
    except OSError:
        return False


@dataclass
class _BoundCheckpoint:
    """One open descriptor of the checkpoint, proven to be the file whose header was read."""

    fd: int
    header: dict[str, _FileTensor]
    parsed: Any
