"""Worker side of Worker-assisted VAE decode (DESIGN.md section 5.6).

``vae_decode`` decodes one unit the driver planned: a latent tile of
ComfyUI's ``tiled_scale_multidim`` loop, or one MiniMax H3 temporal chunk. It
runs on the GPU thread under the actor's lock, so it never overlaps a render,
and it never raises: a unit this rank cannot decode exactly comes back as a
decline with its reason, and the driver decodes it instead.

What the actor may hold: one VAE, the last one a decode asked for, built from
this host's own copy of the file the driver's VAE came from (same name, same
artifact signature) with the driver's dtype. It is placed on the GPU directly,
outside ComfyUI's model manager, so loading it can never evict the resident
DiT. Its load is priced against MemAvailable before any byte is read, and each
unit's working set before it runs, both with the absolute host floor; a price
that does not fit is a decline. ``status()`` reports it, and ``clear_vram`` (at
either level), ``unload`` and a teardown that releases models drop it, so the
driver's own free-before-refusing step reclaims it like any other reloadable
cache.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import socket
import time
from dataclasses import dataclass
from typing import Any

from monarch.actor import concurrent_endpoint

from ..log import get_logger
from ..transfer_utils import failure_summary
from ..vae_decode_env import Decline, adopted, describe, mismatches, qualified
from ..vae_decode_plan import pretouched_cpu

log = get_logger(__name__)

SCHEMA = 1
KINDS = ("multidim", "h3_chunk")
_FOLDERS = ("vae", "vae_approx")
_GIB = float(1 << 30)


@dataclass
class ResidentVae:
    """The one VAE an actor holds for decode, and what it was built from."""

    key: tuple
    vae: Any
    label: str
    bytes: int
    dtype: str
    env: dict
    load_s: float


@dataclass(frozen=True)
class UnitRequest:
    """One validated request off the wire."""

    kind: str
    unit: int
    input: Any
    folder: str
    name: str
    signature: str
    metadata: dict | None
    dtype: Any
    output_dtype: Any
    env: dict


class VaeDecodeMixin:
    """Actor endpoint surface for Worker-assisted VAE decode."""

    _gpu_lock: Any
    _on_gpu: Any

    @concurrent_endpoint
    async def vae_decode(self, request: dict) -> dict:
        """Decode one planned unit, or decline it; never raises (vae_decode_route.py)."""
        async with self._gpu_lock:
            return await self._on_gpu(run, self, request)


def _dtype(text: object) -> Any:
    import torch

    name = str(text).rsplit(".", 1)[-1]
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype) or not value.is_floating_point:
        raise Decline(f"{text!r} names no floating dtype")
    return value


def _validated(request: object) -> UnitRequest:
    import torch

    if not isinstance(request, dict) or request.get("schema") != SCHEMA:
        raise Decline("the request is not a schema-1 VAE decode unit; update the driver and the Workers together")
    kind = request.get("kind")
    vae = request.get("vae")
    tensor = request.get("input")
    env = request.get("env")
    if kind not in KINDS or not isinstance(vae, dict) or not isinstance(env, dict):
        raise Decline("the request names no known unit kind, VAE or environment")
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
        raise Decline("the unit's latent must arrive as a CPU tensor")
    folder, name = vae.get("folder"), vae.get("name")
    if folder not in _FOLDERS or not isinstance(name, str) or not name or os.path.isabs(name) or ".." in name.split("/"):
        raise Decline("the request's VAE name is not a plain name under the vae folders")
    metadata = vae.get("metadata")
    output_dtype = request.get("output_dtype")
    return UnitRequest(
        kind=str(kind), unit=int(request.get("unit", -1)), input=tensor, folder=str(folder), name=name,
        signature=str(vae.get("signature", "")), metadata=metadata if isinstance(metadata, dict) else None,
        dtype=_dtype(vae.get("dtype")), output_dtype=None if output_dtype is None else _dtype(output_dtype),
        env=env)


def _available_bytes() -> int:
    from ..capacity_memory import read_meminfo

    meminfo = read_meminfo()
    available = (meminfo or {}).get("MemAvailable")
    if not isinstance(available, int):
        raise Decline("MemAvailable cannot be read on this host, so no VAE load is priced")
    return available


def _require_room(worker: Any, need: int, what: str) -> None:
    """Refuse a step that would leave less than the host floor (or the operator's reserve)."""
    from ..capacity_floor import ABSOLUTE_HOST_FLOOR_BYTES

    try:
        reserve = int(float(getattr(worker, "uma_reserve_gb", 0.0) or 0.0) * _GIB)
    except (TypeError, ValueError):
        reserve = 0
    keep = max(ABSOLUTE_HOST_FLOOR_BYTES, reserve)
    available = _available_bytes()
    if need + keep > available:
        raise Decline(f"not enough memory to {what}: {need / _GIB:.1f} GiB plus a {keep / _GIB:.1f} GiB "
                      f"floor against {available / _GIB:.1f} GiB available")


def _metadata_digest(metadata: dict | None) -> str:
    payload = json.dumps(metadata or {}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load(worker: Any, spec: UnitRequest) -> ResidentVae:
    """Build this host's copy of the driver's VAE and place it on the GPU."""
    from ..capacity_fit import STOCK_LOAD_TRANSIENT_FACTOR
    from ..gate_artifacts import artifact_signature
    from .comfy_bridge import resolve_model_path

    try:
        path = resolve_model_path(spec.folder, spec.name)
    except FileNotFoundError as exc:
        raise Decline(str(exc)) from exc
    if artifact_signature(path) != spec.signature:
        raise Decline(f"this host's {spec.folder}/{spec.name} is not the driver's file (artifact signature differs)")
    # A stock load holds the file and its host copy together before placement.
    _require_room(worker, int(os.path.getsize(path) * STOCK_LOAD_TRANSIENT_FACTOR), "load the VAE")
    started = time.perf_counter()
    import comfy.sd
    import comfy.utils

    state = comfy.utils.load_torch_file(path)
    vae = comfy.sd.VAE(sd=state, metadata=spec.metadata, dtype=spec.dtype)
    del state
    vae.throw_exception_if_invalid()
    built = qualified(type(vae.first_stage_model))
    if built != spec.env.get("vae_class"):
        raise Decline(f"this host built {built} where the driver has {spec.env.get('vae_class')}")
    size = int(vae.model_size())
    _require_room(worker, size, "place the VAE")
    # Placed directly rather than through load_models_gpu: ComfyUI's manager
    # frees memory by unloading other models, and the DiT must stay resident.
    vae.patcher.patch_model(device_to=vae.device, lowvram_model_memory=0, load_weights=True)
    return ResidentVae(
        key=(spec.folder, spec.name, spec.signature, str(spec.dtype), _metadata_digest(spec.metadata)),
        vae=vae, label=f"{spec.folder}/{spec.name}", bytes=size, dtype=str(spec.dtype),
        env=describe(vae), load_s=time.perf_counter() - started)


def _resident(worker: Any, spec: UnitRequest) -> tuple[ResidentVae, bool]:
    key = (spec.folder, spec.name, spec.signature, str(spec.dtype), _metadata_digest(spec.metadata))
    held: ResidentVae | None = getattr(worker, "_vae_resident", None)
    if held is not None and held.key == key:
        return held, False
    drop_resident(worker)
    worker._vae_resident = _load(worker, spec)
    log.info("VAE decode: loaded %s (%s, %.1f GiB) in %.1f s for Worker-assisted decode",
             worker._vae_resident.label, worker._vae_resident.dtype,
             worker._vae_resident.bytes / _GIB, worker._vae_resident.load_s)
    return worker._vae_resident, True


def _decode(worker: Any, held: ResidentVae, spec: UnitRequest) -> Any:
    import torch

    vae = held.vae
    fsm = vae.first_stage_model
    _require_room(worker, int(vae.memory_used_decode(tuple(spec.input.shape), vae.vae_dtype)), "decode the unit")
    latent = spec.input.to(vae.device)
    import comfy.model_management as mm

    device_context = getattr(mm, "cuda_device_context", None)
    with torch.inference_mode(), adopted(fsm, spec.env):
        if device_context is not None:
            with device_context(vae.device):
                out = _run_unit(fsm, spec, latent)
        else:
            out = _run_unit(fsm, spec, latent)
    return pretouched_cpu(out)


def _run_unit(fsm: Any, spec: UnitRequest, latent: Any) -> Any:
    if spec.kind == "multidim":
        # The stock per-tile function: decode(a.to(vae_dtype).to(device)).to(dtype=output_dtype);
        # the driver already applied the first cast.
        out = fsm.decode(latent)
        return out if spec.output_dtype is None else out.to(dtype=spec.output_dtype)
    return fsm._adaptive_decode(latent)


def run(worker: Any, request: object) -> dict:
    """The endpoint body: decode one unit or say why not. Never raises."""
    started = time.perf_counter()
    reply: dict[str, Any] = {"schema": SCHEMA, "host": _host(), "rank": getattr(worker, "rank", None),
                             "unit": request.get("unit") if isinstance(request, dict) else None}
    try:
        if getattr(worker, "_setup_key", None) is None:
            raise Decline("this rank has not completed setup")
        if (getattr(worker, "_active_worker_args", None) or {}).get("comfy_managed"):
            raise Decline("comfy_managed residency is on in this actor")
        spec = _validated(request)
        held, loaded = _resident(worker, spec)
        problems = mismatches(spec.env, held.env)
        if problems:
            raise Decline("; ".join(problems))
        reply.update(ok=True, output=_decode(worker, held, spec), loaded=loaded,
                     load_s=round(held.load_s, 3) if loaded else None)
    except Decline as decline:
        reply.update(ok=False, reason=str(decline))
    except Exception as exc:
        _empty_cache()
        reply.update(ok=False, reason=f"the decode failed: {failure_summary(exc)}")
    reply["seconds"] = round(time.perf_counter() - started, 3)
    return reply


def _host() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "this host"


def _empty_cache() -> None:
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
    except Exception:
        pass


def drop_resident(worker: Any) -> bool:
    """Release the decode VAE, if any; ``clear_vram``, ``unload`` and a releasing teardown call this."""
    held: ResidentVae | None = getattr(worker, "_vae_resident", None)
    worker._vae_resident = None
    if held is None:
        return False
    log.info("VAE decode: released %s (%.1f GiB)", held.label, held.bytes / _GIB)
    del held
    _empty_cache()
    return True


def status_block(worker: Any) -> dict:
    """The resident decode VAE as ``status()`` reports it, for capacity readers."""
    held: ResidentVae | None = getattr(worker, "_vae_resident", None)
    if held is None:
        return {"resident": None, "bytes": 0}
    return {"resident": held.label, "bytes": held.bytes, "dtype": held.dtype}
