"""Build FSDP shards on meta, then move only each rank's rows to the device.

FSDP2 normally moves full parameters before chunking them. Replace parameters
with meta tensors first so ``fully_shard`` uses deferred initialization without
copying full blocks. Slice local rows from the original host tensors, move
them to the device, and install DTensors with ``from_local``. This needs no
collective because every rank reads the same checkpoint. FSDP2's documented
meta-init path repads uneven shards and rebinds gather buffers at lazy init.
The build high-water is the local shards plus one chunk.

Do not use ``distribute_tensor``: it moves the full tensor to the mesh device
before chunking. Copy mmap-backed rows through a pinned bounce buffer. Direct
device copies can pin source pages with write intent, break copy-on-write,
and retain anonymous copies while any tensor holds the mapping. See
DESIGN.md §5.4 and VALIDATION.md.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from ..log import get_logger

log = get_logger(__name__)

# One pinned bounce buffer per build, reused row range by row range.
BOUNCE_BYTES = 256 << 20


def shard_device() -> torch.device:
    """Where this rank's shards live: the current cuda device, else cpu."""
    try:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
    except Exception:  # a torch that answers is_available without a driver (CI stubs)
        pass
    return torch.device("cpu")


class DeviceCopier:
    """Copy host tensors to a device without pinning the checkpoint's pages.

    One pinned bounce buffer per build (``close`` frees it): a memcpy in, the
    buffer out to the device, so the driver never sees a file-backed source.
    ``bounced`` counts the bytes that took that path; off cuda, or with the
    pin refused, the copy is a clone, which reads the mapping the same way.
    """

    def __init__(self, bounce_bytes: int = BOUNCE_BYTES) -> None:
        self.bounced = 0
        self._buffer: torch.Tensor | None = None
        if torch.cuda.is_available():
            try:
                self._buffer = torch.empty(bounce_bytes, dtype=torch.uint8, pin_memory=True)
            except Exception as exc:
                log.debug("pinned bounce buffer unavailable, copying rows by clone (%r)", exc)

    def close(self) -> None:
        self._buffer = None

    def copy(self, rows: torch.Tensor, device: torch.device) -> torch.Tensor:
        """``rows`` on ``device``: a fresh tensor unless it is already there."""
        if rows.is_meta or rows.device.type != "cpu":
            return rows if rows.device == device else rows.to(device=device)
        if device.type == "cpu":
            return rows.clone()
        if self._buffer is None or rows.numel() == 0:
            return rows.clone().to(device=device)
        source = rows.contiguous()
        out = torch.empty(source.shape, dtype=source.dtype, device=device)
        flat_in = source.view(-1).view(torch.uint8)
        flat_out = out.view(-1).view(torch.uint8)
        step = self._buffer.numel()
        for offset in range(0, flat_in.numel(), step):
            count = min(step, flat_in.numel() - offset)
            self._buffer[:count].copy_(flat_in[offset:offset + count])
            flat_out[offset:offset + count].copy_(self._buffer[:count])
        self.bounced += flat_in.numel()
        return out


def _stage_wrapped(param: Any, copier: DeviceCopier, device: torch.device) -> Any:
    """A quantized wrapper rebuilt on ``device``, every tensor field bounced."""
    params = param._params
    fields = {field.name: getattr(params, field.name) for field in dataclasses.fields(params)}
    for name, value in fields.items():
        if isinstance(value, torch.Tensor):
            fields[name] = copier.copy(value, device)
    return param._with(copier.copy(param._qdata, device), type(params)(**fields))


def stage_on_device(module: torch.nn.Module, copier: DeviceCopier,
                    device: torch.device | None = None,
                    ignored: set | None = None) -> int:
    """Move ``module``'s host parameters and buffers to the device, bounced.

    Run right before each ``fully_shard``, where FSDP2 would otherwise move
    the module itself with ``.to(device)``. It moves exactly what FSDP2
    moves, every buffer and every parameter outside ``ignored``, and leaves
    meta parameters to ``build_shards``. Returns the bytes moved; the device
    transient is the one block the direct wrap paid.
    """
    device = shard_device() if device is None else device
    ignored = ignored or set()
    moved = 0
    for owner in module.modules():
        for name, param in list(owner._parameters.items()):
            if param is None or param.is_meta or param.device.type != "cpu" or param in ignored:
                continue
            if hasattr(param, "_qdata") and hasattr(param, "_with"):
                # A wrapper carries its bytes as Python attributes, which
                # ``.data`` would leave behind, so it is rebuilt whole; an
                # ignored parameter never gets here (skipped above).
                owner._parameters[name] = torch.nn.Parameter(
                    _stage_wrapped(param, copier, device), requires_grad=False)
            else:
                # Through .data: the wrap's ignored set holds these by identity.
                param.data = copier.copy(param.data, device)
            moved += param.numel() * param.element_size()
        for buffer in list(owner._buffers.values()):
            if buffer is None or buffer.is_meta or buffer.device.type != "cpu":
                continue
            # Through .data, the way FSDP2 moves a buffer, so two modules that
            # share one are not forked into a device copy each.
            buffer.data = copier.copy(buffer, device)
            moved += buffer.numel() * buffer.element_size()
    return moved


def shard_rows(rows: int, world: int, rank: int) -> tuple[int, int]:
    """The one shard-geometry contract; canonical in actor/fsdp_lora.py."""
    from ..actor.fsdp_lora import shard_rows as _canonical_shard_rows

    return _canonical_shard_rows(rows, world, rank)


def _is_dtensor(tensor: Any) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover - torch without distributed
        return False
    return isinstance(tensor, DTensor)


def _set_parameter(root: torch.nn.Module, name: str, param: torch.nn.Parameter) -> None:
    owner_name, _, attr = name.rpartition(".")
    owner = root.get_submodule(owner_name) if owner_name else root
    owner._parameters[attr] = param


def _shard_capacity_refusal(name: str, exc: BaseException):
    """Report a device allocation failure during shard construction as class C.

    Available memory can fall after preflight, including when another process
    loads a text encoder into the shared pool.
    """
    from ..mesh_safety import StockLoadCapacityError
    from ..refusal import RefusalClass, refusal

    return StockLoadCapacityError(refusal(
        RefusalClass.CAPACITY,
        f"the FSDP shard build ran out of unified memory materializing {name}: "
        "something took host memory after the launch was priced (another model, "
        "the driver's text encoder, or a concurrent render). Use Clear VRAM, or "
        "re-queue when the host is quieter.",
        guard="stock_load_preflight", waivable=False))


def build_shards(diffusion_model: torch.nn.Module, wrap: Callable[[], None],
                 ignored: set | None = None, copier: DeviceCopier | None = None) -> int:
    """Run ``wrap`` (the fully_shard calls) on meta parameters, then fill shards.

    Returns the number of parameters materialized from host rows. Parameters in
    ``ignored``, zero-dimensional ones, and any parameter the wrap left as a
    plain tensor (test doubles that stub ``fully_shard``) keep their original
    data untouched. The caller puts zero-dimensional parameters in ``ignored``
    too, so the ``ndim == 0`` test below is a second guard.
    """
    ignored = ignored or set()
    copier = DeviceCopier() if copier is None else copier
    originals: dict[str, torch.Tensor] = {}
    for name, param in list(diffusion_model.named_parameters()):
        if param in ignored or param.is_meta or param.ndim == 0:
            continue
        originals[name] = param.data
        # A Parameter cannot take meta data through .data (set_data refuses a
        # different tensor type), so the owning module gets a meta Parameter.
        _set_parameter(diffusion_model, name, torch.nn.Parameter(
            torch.empty(param.shape, dtype=param.dtype, device="meta"),
            requires_grad=False))
    try:
        wrap()
    except BaseException:
        for name, full in originals.items():
            _set_parameter(diffusion_model, name,
                           torch.nn.Parameter(full, requires_grad=False))
        raise
    sharded: dict[str, torch.nn.Parameter] = {}
    materialized = 0
    for name, full in originals.items():
        param = diffusion_model.get_parameter(name)
        if not _is_dtensor(param):
            # Not wrapped (a stubbed fully_shard): leave it exactly as loaded.
            _set_parameter(diffusion_model, name,
                           torch.nn.Parameter(full, requires_grad=False))
            continue
        from torch.distributed.tensor import DTensor

        wrapped: Any = param
        mesh = wrapped.device_mesh
        placements = wrapped.placements
        shard_dim = getattr(placements[0], "dim", 0) if placements else 0
        if shard_dim != 0:
            raise RuntimeError(
                f"FSDP shard build: {name} is sharded on dim {shard_dim}; only dim 0 "
                "is supported")
        start, stop = shard_rows(int(full.shape[0]), mesh.size(), mesh.get_local_rank())
        device = torch.device(mesh.device_type)
        if device.type == "cuda":
            device = torch.device("cuda", torch.cuda.current_device())
        try:
            local = copier.copy(full[start:stop], device)
        except torch.cuda.OutOfMemoryError as exc:
            raise _shard_capacity_refusal(name, exc) from exc
        except RuntimeError as exc:
            if "NV_ERR_NO_MEMORY" not in str(exc) and "out of memory" not in str(exc).lower():
                raise
            raise _shard_capacity_refusal(name, exc) from exc
        dtensor = DTensor.from_local(
            local, mesh, placements, run_check=False,
            shape=torch.Size(full.shape), stride=tuple(full.stride()))
        sharded[name] = torch.nn.Parameter(dtensor, requires_grad=False)
        materialized += 1
    if sharded:
        diffusion_model.load_state_dict(sharded, strict=False, assign=True)
    return materialized


def drop_sharded_weights_in_place(patchers: Any, origin: str = "") -> tuple[int, int]:
    """Release discarded DTensor weights before ComfyUI can offload them.

    ComfyUI normally moves weights to the offload device during detach, creating
    an unnecessary host copy of shards about to be freed. Replace them with
    empty CPU placeholders first. Pass only patchers from the dropping slot;
    surviving slots still need their weights.

    Restore each wrapped module's original class too: FSDP2's ``_apply`` expects
    DTensor ``_local_tensor`` attributes that plain placeholders lack.

    On failure, log and return (0, 0). The caller then uses ComfyUI's slower
    staging path rather than failing the drop.
    """
    try:
        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.tensor import DTensor

        seen: set[int] = set()
        dropped = 0
        freed = 0
        for patcher in patchers:
            model = getattr(patcher, "model", None)
            if not isinstance(model, torch.nn.Module) or id(model) in seen:
                continue
            seen.add(id(model))
            for module in model.modules():
                for attr, param in list(module._parameters.items()):
                    if param is None or not isinstance(param.data, DTensor):
                        continue
                    local = param.data.to_local()
                    freed += local.element_size() * local.nelement()
                    module._parameters[attr] = torch.nn.Parameter(
                        torch.empty(0, device="cpu", dtype=param.dtype),
                        requires_grad=False)
                    dropped += 1
            if dropped:
                for module in model.modules():
                    if isinstance(module, FSDPModule):
                        restored: Any = next(
                            cls for cls in type(module).__mro__
                            if cls is not FSDPModule
                            and not issubclass(cls, FSDPModule))
                        module.__class__ = restored
        if dropped:
            log.info(
                "%s: dropped %d sharded parameters in place (%.1f GiB freed "
                "before the global unload)", origin or "shard drop", dropped,
                freed / (1 << 30))
        return dropped, freed
    except Exception as exc:
        log.warning("%s: in-place shard drop failed; the unload will stage "
                    "weights instead: %r", origin or "shard drop", exc)
        return 0, 0
