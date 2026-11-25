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
from .load_renames import with_renamed
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
    matches: Callable[[_FileTensor], bool] | None = None,
    matches_exactly: Callable[[_FileTensor], bool] | None = None,
    identical: Callable[[_FileTensor, _FileTensor], bool] | None = None,
) -> _FileTensor:
    """The checkpoint tensor a patcher key restores from.

    ``renames`` is ``load_time_renames`` for this model and file: a key ComfyUI
    renames at load (FLUX's ``<...>_norm.scale``, held as ``<...>_norm.weight``)
    sits under none of the namespace spellings, and is tried last.

    ``matches`` is the byte check against the live rows. A file may hold a
    tensor under the key's own name and another under the name ComfyUI loaded
    it from; the first candidate of the live dtype and shape whose bytes match
    is the one, as in the resident capture (``unbake._map_plain``). When more
    than one candidate has the live dtype and shape, ``matches_exactly``
    compares each with every byte of this rank's rows, not the sampled
    windows: two such tensors can agree in the windows and differ elsewhere,
    and the bake reads the bound one whole. Where this rank's rows equal two
    candidates that ``identical`` finds different elsewhere, the rows cannot
    choose, and the bake refuses rather than guess. With no match the first
    candidate of the right dtype and shape is returned, and the caller's own
    check then says the model is not pristine.
    """
    fitting: list[_FileTensor] = []
    misfit: tuple[str, _FileTensor] | None = None
    for cand in with_renamed(_candidate_file_keys(key), renames, key):
        ft = header.get(cand)
        if ft is None:
            continue
        if ft.dtype != param.dtype or ft.shape != tuple(param.shape):
            misfit = misfit or (cand, ft)
            continue
        fitting.append(ft)
    if len(fitting) > 1 and matches_exactly is not None:
        exact = [ft for ft in fitting if matches_exactly(ft)]
        if len(exact) > 1 and identical is not None and not all(identical(exact[0], ft) for ft in exact[1:]):
            # This rank's rows equal two tensors that differ elsewhere. The bake
            # reads the bound tensor whole, and a patch that mixes rows would
            # then depend on a choice these rows cannot make; another rank's
            # rows may settle it, but nothing here can know which.
            raise UnbakeError(
                f"the checkpoint holds {key} under two names that agree on this rank's rows "
                "and differ elsewhere, so the bake cannot tell which one ComfyUI loaded; "
                "reload contract applies")
        if exact:
            return exact[0]
        return fitting[0]
    for ft in fitting:
        if matches is None or matches(ft):
            return ft
    if fitting:
        return fitting[0]
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


def _shard_rows_equal(
    fd: int, ft: _FileTensor, param: torch.Tensor, rank: int, world: int,
    replicated: frozenset[int] = frozenset(),
) -> bool:
    """Every byte of this rank's rows on disk equals its shard (the sampled check's exact form)."""
    local = local_shard(param, replicated)
    rank, world = shard_geometry(param, replicated, rank, world)
    rows = ft.shape[0] if ft.shape else 1
    start, stop = shard_rows(rows, world, rank)
    nbytes = (stop - start) * (ft.nbytes // max(rows, 1))
    if nbytes != local.numel() * local.element_size():
        return False
    try:
        ours = local.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes()
        theirs = os.pread(fd, nbytes, ft.start + start * (ft.nbytes // max(rows, 1)))
    except (OSError, RuntimeError):
        return False
    return theirs == ours


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
