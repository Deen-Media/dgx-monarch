"""Zero-copy shared-memory weights adopted through Comfy's assign=True path.

Lazy un-bake and replacement casts reuse a primary slab and annex; close only
after model release. GB10 prerequisites and measured results: docs/VALIDATION.md.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from ..transfer_utils import (
    failure_summary,
    raise_with_distinct_cause,
    reconcile_error,
)
from . import slab_lifetime
from .slab_arena import _ALIGN, _Arena, _discard_identity, _OwnedDescriptor
from .slab_certificate import CertificateBuilder, SlabByteCertificate
from .slab_wrap import wrap_u8 as _wrap_u8
from .unbake import drop_file_cache, live_tensor, read_safetensors_header

log = logging.getLogger("dgx-monarch")

_READ_CHUNK = 256 << 20
_ANNEX_CACHE_DRAIN_BYTES = 256 << 20

@dataclass
class _Region:
    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]


@dataclass
class _AnnexCacheDrain:
    """Trim only completed, retired CUDA cast sources in bounded batches."""

    _bytes: int = 0

    @staticmethod
    def settle_copy(source: torch.Tensor) -> bool:
        if not source.is_cuda or not torch.cuda.is_available():
            return False
        # Complete the raw-pointer copy before retiring its Parameter storage.
        torch.cuda.synchronize()
        return True

    def retire(self, nbytes: int) -> None:
        self._bytes += nbytes
        if self._bytes >= _ANNEX_CACHE_DRAIN_BYTES:
            self.flush()

    def flush(self) -> None:
        if not self._bytes:
            return
        # Trim failures propagate through normal failed-load ownership cleanup.
        torch.cuda.empty_cache()
        self._bytes = 0


# Observe open slabs by file identity; never share them between store slots,
# whose in-place LoRA bakes could corrupt each other. Per-instance tokens
# make close-interrupted counts repairable over the process lifetime.
_OPEN_SLABS: dict[tuple, int] = {}  # Process lifetime.
_OPEN_SLAB_OWNERS: dict[tuple, set[object]] = {}  # Process lifetime.
_OPEN_SLABS_LOCK = threading.RLock()


class WeightSlab:
    """A checkpoint's weights, resident in shared memory, computed on in place.

    Lifetime contract: the ModelStore owns the slab and closes it only after
    the model that references it has been freed (a wrapped tensor holds a raw
    pointer, not a reference). One slab per store slot, never shared (see
    _OPEN_SLABS)."""

    def __init__(self, path: str, *, handoff: list[WeightSlab] | None = None):
        self.path = path
        self.arena: _Arena = None  # type: ignore[assignment]  # partial-init sentinel
        self.annex: _Arena | None = None
        self._arena_handoff: list[_Arena] = []
        self._annex_handoff: list[_Arena] = []
        self._handoff = handoff
        self._provisional_poisoned_before: bool | None = None
        self._close_guard: Any = None
        self._registration_token = object()
        self._registered = False
        self.certificate: SlabByteCertificate | None = None
        self.regions: dict[str, _Region] = {}
        # comfy may strip a checkpoint prefix before load_model_weights; the
        # load hook records the stripped key -> region mapping by data_ptr so
        # reabsorb() can find a stray's original region under its module name.
        self.key_map: dict[str, _Region] = {}
        self.cast_keys: list[str] = []      # filled by the load hook
        self.read_seconds = 0.0
        self.stray_bytes = 0                # quant strays reabsorb cannot take
        try:
            # Retain this exact object at process level before any file, mapping, or descriptor.
            self._provisional_poisoned_before = slab_lifetime.prepublish_resource(self)
            if handoff is not None:
                handoff.append(self)
            self.headers, self.metadata, identity = read_safetensors_header(
                path, want_metadata=True, want_identity=True)
            self.file_size, self.file_dev, self.file_ino = (
                identity.file_size, identity.file_dev, identity.file_ino)
            self.file_mtime_ns = identity.file_mtime_ns
            self.file_ctime_ns = identity.file_ctime_ns
            self.header_len = identity.header_len
            self._open_key = (path, self.file_dev, self.file_ino, self.file_size,
                              self.file_mtime_ns, self.file_ctime_ns)
            aligned_total = sum(
                (ft.nbytes + _ALIGN - 1) // _ALIGN * _ALIGN + _ALIGN
                for ft in self.headers.values())
            _Arena("dgxm-weight-slab", aligned_total, handoff=self._arena_handoff)
            self.arena = self._arena_handoff[-1]
            self.arena.confirm_handoff()
            _discard_identity(self._arena_handoff, self.arena)
            self._read_all()
            # Publish intent before a counter update can be interrupted.
            self._registered = True
            with _OPEN_SLABS_LOCK:
                owners = _OPEN_SLAB_OWNERS.setdefault(self._open_key, set())
                owners.add(self._registration_token)
                count = len(owners)
                _OPEN_SLABS[self._open_key] = count
            if count > 1:
                log.warning(
                    "second live slab for the same checkpoint (%d total): slabs are "
                    "per slot because bakes write in place, so this slab holds another copy of about %.1f GiB",
                    count,
                    sum(ft.nbytes for ft in self.headers.values()) / (1 << 30))
        except BaseException as primary:
            try:
                slab_lifetime.close_slab_or_retain_after_explicit_unload(
                    self, "partial weight slab")
            except BaseException as cleanup_error:
                strongest, cause = reconcile_error(
                    primary,
                    cleanup_error,
                    "partial weight slab cleanup also failed",
                )
                raise_with_distinct_cause(strongest, cause)
            raise

    def confirm_handoff(self) -> None:
        poisoned_before = self._provisional_poisoned_before
        if poisoned_before is None:
            return
        slab_lifetime.confirm_prepublished_resource(self, poisoned_before)
        self._provisional_poisoned_before = None

    def _read_all(self) -> None:
        source = _OwnedDescriptor.open(self.path, "slab checkpoint source descriptor")
        src = source._fd
        t0 = time.monotonic()
        try:
            arena_mm = self.arena.mm
            if arena_mm is None:
                raise RuntimeError("slab arena mapping is unavailable")
            if not self._matches_stat(os.fstat(src)) or not self._matches_stat(os.stat(self.path)):
                raise RuntimeError("checkpoint changed between header validation and slab read")
            # Byte-verify each region the moment its own read completes: it is
            # still pristine (no cast, no bake, no reabsorb), its pages are the
            # ones just read, and the pinned identity is enforced on both sides
            # of this loop, so no new time-of-check window opens.
            builder = CertificateBuilder(
                path=self.path, file_identity=self.file_identity,
                header_len=self.header_len, tensor_count=len(self.headers),
                total_bytes=sum(ft.nbytes for ft in self.headers.values()))
            for name, ft in self.headers.items():
                off = self.arena.place(ft.nbytes)
                self.regions[name] = _Region(off, ft.nbytes, ft.dtype, ft.shape)
                done = 0
                while done < ft.nbytes:
                    n = min(_READ_CHUNK, ft.nbytes - done)
                    got = os.preadv(
                        src, [memoryview(arena_mm)[off + done:off + done + n]],
                        ft.start + done)
                    if got != n:
                        raise RuntimeError(
                            f"short read for {name}: {got} != {n} at {done}")
                    done += n
                builder.verify_region(src, name, ft, arena_mm, off)
            if not self._matches_stat(os.fstat(src)) or not self._matches_stat(os.stat(self.path)):
                raise RuntimeError("checkpoint changed while slab bytes were being read")
            self.certificate = builder.issue(src)
        except BaseException as primary:
            # Do not replace the read failure with a teardown failure.
            try:
                slab_lifetime.close_or_retain_after_explicit_unload(
                    source, "slab checkpoint source descriptor"
                )
            except BaseException as cleanup_error:
                strongest, cause = reconcile_error(
                    primary,
                    cleanup_error,
                    "checkpoint source cleanup also failed",
                )
                raise_with_distinct_cause(strongest, cause)
            raise
        else:
            slab_lifetime.close_or_retain_after_explicit_unload(
                source, "slab checkpoint source descriptor"
            )
        finally:
            pending = sys.exception()
            try:
                drop_file_cache(self.path)
            except BaseException as cache_error:
                if pending is None:
                    raise
                strongest, cause = reconcile_error(
                    pending,
                    cache_error,
                    "checkpoint page-cache eviction also failed",
                )
                raise_with_distinct_cause(strongest, cause)
        self.read_seconds = time.monotonic() - t0

    def _tensor(self, region: _Region, *, cuda: bool) -> torch.Tensor:
        flat = _wrap_u8(self.arena.base + region.offset, region.nbytes,
                        cuda=cuda)
        return flat.view(region.dtype).reshape(region.shape)

    def state_dict(self) -> dict[str, torch.Tensor]:
        cuda = torch.cuda.is_available()
        return {name: self._tensor(region, cuda=cuda and _slab_tensor_uses_cuda(name))
                for name, region in self.regions.items()}

    def contains(self, ptr: int) -> bool:
        return self.arena.contains(ptr) or (
            self.annex is not None and self.annex.contains(ptr))

    def region_by_ptr(self, ptr: int) -> _Region | None:
        if not self.arena.contains(ptr):
            return None
        off = ptr - self.arena.base
        for region in self.regions.values():
            if region.offset <= off < region.offset + region.nbytes:
                return region
        return None

    @property
    def file_identity(self) -> str:
        """store_family.file_identity() spelling, for the pinned inode.

        Derived from the identity the header validation pinned rather than a
        fresh stat: the certificate must name the file it verified.
        """
        return (f"{self.file_dev}:{self.file_ino}:{self.file_size}:"
                f"{self.file_mtime_ns}:{self.file_ctime_ns}")

    def _matches_stat(self, st: os.stat_result) -> bool:
        return (
            st.st_dev == self.file_dev
            and st.st_ino == self.file_ino
            and st.st_size == self.file_size
            and st.st_mtime_ns == self.file_mtime_ns
            and st.st_ctime_ns == self.file_ctime_ns
        )

    def stat_ok(self) -> bool:
        try:
            st = os.stat(self.path)
        except OSError:
            return False
        return self._matches_stat(st)

    @property
    def total_gib(self) -> float:
        total = self.arena.capacity + (self.annex.capacity if self.annex else 0)
        return round(total / (1 << 30), 2)

    def _region_for_stray(self, name: str, tensor: torch.Tensor) -> _Region | None:
        region = self.key_map.get(name) or self.regions.get(name)
        if (region is not None
                and region.nbytes == tensor.numel() * tensor.element_size()
                and region.dtype == tensor.dtype):
            return region
        return None

    def _reabsorb_quant(self, module: Any, name: str, prefix: str) -> int:
        """Migrate one strayed QuantizedTensor weight's payload into the slab.

        comfy's bake replaces the whole wrapper: `set_weight` requantizes and
        assigns a new Parameter in cudaMalloc. The wrapper is cloned, never
        re-derived: the payload bytes (`_qdata`) are copied into the layer's
        original slab region and a new wrapper is built from the same layout
        name and the same params object (scale tensors, a few KB, stay where
        they are). Never rebuild a baked wrapper from file metadata: baked
        bytes need not match the file's serialization transforms (int8-convrot
        diverged; see docs/VALIDATION.md).

        Returns migrated bytes; 0 leaves the key in cudaMalloc (still
        correct, but heavier)."""
        module_key, _, param_name = name.rpartition(".")
        if param_name != "weight" or not module_key:
            return 0
        live = live_tensor(module, name)
        # the wrapper attrs may sit on the Parameter or on its .data,
        # depending on how the subclass propagates through Parameter()
        wrapper: Any = None
        for cand in (live, getattr(live, "data", None)):
            if cand is not None and getattr(cand, "_qdata", None) is not None:
                wrapper = cand
                break
        if wrapper is None:
            return 0
        qdata = wrapper._qdata
        layout_name = getattr(wrapper, "_layout_cls", None)
        params = getattr(wrapper, "_params", None)
        if not isinstance(layout_name, str) or params is None:
            return 0
        fkey = f"{prefix}{module_key}.{param_name}"
        region = self.key_map.get(fkey) or self.regions.get(fkey)
        nbytes = qdata.numel() * qdata.element_size()
        if region is None or region.nbytes != nbytes:
            return 0  # padded storage or layout drift: leave it be

        dest = _wrap_u8(self.arena.base + region.offset, nbytes,
                        cuda=qdata.device.type == "cuda")
        slab_qdata = dest.view(qdata.dtype).reshape(qdata.shape)
        slab_qdata.copy_(qdata.detach())
        op: Any = live_tensor(module, module_key)
        op.weight = torch.nn.Parameter(
            type(wrapper)(slab_qdata, layout_name, params), requires_grad=False)
        return nbytes

    def reabsorb(self, module: torch.nn.Module, prefix: str = "") -> dict:
        """Migrate strayed Parameters back into slab memory.

        comfy's first-load bake path (patch_weight_to_device calls
        set_attr_param) replaces Parameters with freshly allocated tensors;
        the load hook's dtype pre-cast does the same for F32-stored keys.
        For each named parameter whose storage is outside the slab: copy it
        into its original region (same dtype+size for the baked case) or into
        the annex (the cast case), and swap the Parameter in place. A strayed
        QuantizedTensor weight gets its payload copied into the slab under a
        cloned wrapper (_reabsorb_quant); any that cannot move stay in
        cudaMalloc, counted in `quant_stray_gib`.
        """
        strays: list[tuple[str, torch.nn.Parameter]] = []
        quant_strays: list[str] = []
        annex_need = 0
        self.stray_bytes = 0                   # recomputed per pass
        quant_moved_bytes = 0
        for name, param in module.named_parameters():
            t = param.data
            if type(t) is not torch.Tensor:      # QuantizedTensor subclass etc.
                quant_strays.append(name)
                continue
            if self.contains(t.data_ptr()):
                continue
            strays.append((name, param))
            if self._region_for_stray(prefix + name, t) is None:
                annex_need += t.numel() * t.element_size() + _ALIGN

        for name in quant_strays:
            try:
                migrated = self._reabsorb_quant(module, name, prefix)
                if not migrated:
                    t = live_tensor(module, name)
                    self.stray_bytes += t.numel() * t.element_size()
            except Exception as exc:
                # _reabsorb_quant mutates the module only as its last action,
                # so the model is intact here; the stray stays in cudaMalloc.
                try:
                    log.warning(
                        "quant reabsorb failed for %s (%s); the tensor stays in cudaMalloc",
                        name,
                        failure_summary(exc),
                    )
                except BaseException:
                    pass
                continue
            quant_moved_bytes += migrated

        if annex_need and self.annex is None:
            _Arena("dgxm-weight-slab-annex", annex_need,
                   handoff=self._annex_handoff)
            self.annex = self._annex_handoff[-1]
            self.annex.confirm_handoff()
            _discard_identity(self._annex_handoff, self.annex)
        # A re-run against a partially consumed annex may not fit every
        # region-less stray; those stay in cudaMalloc (counted and logged), but
        # strays whose original region can take them back are reabsorbed regardless.
        moved = 0
        moved_bytes = 0
        skipped = 0
        annex_drain = _AnnexCacheDrain()
        for name, param in strays:
            t = param.data
            nbytes = t.numel() * t.element_size()
            region = self._region_for_stray(prefix + name, t)
            if region is not None:
                target_arena, off = self.arena, region.offset
            else:
                annex = self.annex
                if annex is None or annex.cursor + nbytes > annex.capacity:
                    skipped += 1
                    continue
                target_arena, off = annex, annex.place(nbytes)
            flat = _wrap_u8(target_arena.base + off, nbytes,
                            cuda=t.is_cuda)
            dest = flat.view(t.dtype).reshape(t.shape)
            dest.copy_(t)
            annex_cuda_source = (
                target_arena is self.annex and annex_drain.settle_copy(t))
            param.data = dest
            moved += 1
            moved_bytes += nbytes
            del t, dest, flat
            if annex_cuda_source:
                # Only annex growth needs this trim; primary bakes are unchanged.
                annex_drain.retire(nbytes)
        annex_drain.flush()
        if moved and torch.cuda.is_available():
            torch.cuda.synchronize()
        if skipped:
            log.warning("slab annex too small: %d strays stay in cudaMalloc", skipped)
        return {"reabsorbed": moved,
                "reabsorbed_gib": round(moved_bytes / (1 << 30), 2),
                "quant_reabsorbed_gib": round(quant_moved_bytes / (1 << 30), 2),
                "skipped": skipped,
                "quant_stray_gib": round(self.stray_bytes / (1 << 30), 2)}

    def close(self) -> None:
        """Unmap and free; call only after the referencing model is gone."""
        if self._close_guard is not None:
            self._close_guard.close()
        failure: BaseException | None = None
        failure_cause: BaseException | None = None
        arenas: list[_Arena] = []
        for arena in (self.arena, self.annex, *self._arena_handoff,
                      *self._annex_handoff):
            if arena is None or any(owned is arena for owned in arenas):
                continue
            arenas.append(arena)
            try:
                arena.close()
            except BaseException as exc:
                failure, failure_cause = reconcile_error(
                    failure,
                    exc,
                    "another slab arena close also failed",
                )
                try:
                    log.error(
                        "slab arena close failed: %s",
                        failure_summary(exc),
                    )
                except BaseException:
                    pass
        if failure is not None:
            raise_with_distinct_cause(failure, failure_cause)
        self._arena_handoff.clear()
        self._annex_handoff.clear()
        if self._registered:
            with _OPEN_SLABS_LOCK:
                owners = _OPEN_SLAB_OWNERS.get(self._open_key)
                if owners is not None:
                    owners.discard(self._registration_token)
                    count = len(owners)
                    if count:
                        _OPEN_SLABS[self._open_key] = count
                    else:
                        _OPEN_SLABS.pop(self._open_key, None)
                        _OPEN_SLAB_OWNERS.pop(self._open_key, None)
                self._registered = False
        _discard_identity(self._handoff, self)
        self.confirm_handoff()

    def telemetry(self) -> dict[str, Any]:
        return {
            "slab_gib": self.total_gib,
            "slab_read_s": round(self.read_seconds, 1),
            "slab_cast_keys": len(self.cast_keys),
            "slab_quant_stray_gib": round(self.stray_bytes / (1 << 30), 2),
            "slab_cert": self.certificate.public() if self.certificate else None,
        }


def _slab_tensor_uses_cuda(name: str) -> bool:
    """Keep Comfy's U8 quant JSON markers CPU-readable for Tensor.numpy()."""
    return name != "comfy_quant" and not name.endswith(".comfy_quant")
