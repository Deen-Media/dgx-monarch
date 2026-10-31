"""Driver side of the Worker-assisted VAE decode RPC (DESIGN.md section 5.6).

Each unit is one ``vae_decode`` call on one rank, dispatched through
``mesh_setup.dispatch_endpoint`` under the decode's render session and the
READY setup token, and awaited through ``MeshHandle._await_or_evict``, so a
dead actor evicts the fleet as on every other path and a slow one only times
out. The endpoint is setup-bound and outside the ambiguous-mutation set: a
decode that times out may still finish on its rank, but it changes nothing a
later render reads except the rank's resident decode VAE, which ``status()``
reports and ``clear_vram`` drops.

What crosses the wire, per unit: the unit's latent as a CPU tensor in the
VAE's dtype (the cast the stock decode function makes first), the VAE's folder,
name, artifact signature, header metadata and dtype, the output dtype, and the
driver's decode environment (vae_decode_env.describe). The reply carries the
decoded unit as a CPU tensor in the dtype the stock function returns: the
output dtype for a tile, the VAE dtype for an H3 chunk canvas.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from . import mesh_setup

ENDPOINT = "vae_decode"
SCHEMA = 1
# The first unit a rank takes may include building its VAE from disk.
FIRST_UNIT_TIMEOUT_S = 600.0
UNIT_TIMEOUT_S = 300.0
# Decoded replies a rank finished wait on the driver until the stock loop
# reaches them, beside the decode's own output buffer. This caps their bytes,
# reserved at claim (vae_decode_schedule.UnitBoard): about twelve MiniMax H3
# chunk canvases at 1344x768 in fp16 or bf16, against the ten a 362-frame
# clip leaves on a remote rank when both ranks decode at the same pace.
REPLY_BUDGET_BYTES = 2 << 30


def vae_source(vae: Any) -> dict[str, Any] | str:
    """The file a Worker loads to hold the driver's VAE, or why there is none.

    ComfyUI's VAELoader records ``(comfy.sd.load_vae_patcher, (path, metadata,
    device))`` on the VAE's patcher as its reload factory. A VAE from a
    checkpoint or built in memory has no such record and stays on the driver.
    """
    init = getattr(getattr(vae, "patcher", None), "cached_patcher_init", None)
    if not isinstance(init, tuple) or len(init) != 2 or not isinstance(init[1], tuple) or not init[1]:
        return "the VAE did not come from a VAE loader file, so no Worker can load the same bytes"
    factory, args = init
    if getattr(factory, "__name__", "") != "load_vae_patcher":
        return "the VAE did not come from a VAE loader file, so no Worker can load the same bytes"
    path = os.path.realpath(str(args[0]))
    metadata = args[1] if len(args) > 1 and isinstance(args[1], dict) else None
    import folder_paths

    for folder in ("vae", "vae_approx"):
        for root in folder_paths.get_folder_paths(folder):
            relative = os.path.relpath(path, os.path.realpath(root))
            if not relative.startswith(os.pardir) and not os.path.isabs(relative):
                from .gate_artifacts import artifact_signature

                signature = artifact_signature(path)
                if signature in ("unreadable", "unstable"):
                    return f"the VAE file's identity is {signature} on this host"
                return {"folder": folder, "name": relative.replace(os.sep, "/"),
                        "signature": signature, "metadata": metadata,
                        "dtype": str(getattr(vae, "vae_dtype", ""))}
    return "the VAE file is outside ComfyUI's vae folders, so the Workers cannot name it"


def _driver_gpu_index() -> int:
    """The physical GPU this ComfyUI process decodes on."""
    try:
        import comfy.model_management as mm

        index = int(getattr(mm.get_torch_device(), "index", 0) or 0)
    except Exception:
        index = 0
    visible = [entry.strip() for entry in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if entry.strip()]
    if index < len(visible) and visible[index].isdigit():
        return int(visible[index])
    return index


def remote_ranks(handle: Any) -> list[int]:
    """Every rank whose GPU is not this driver's own.

    The driver decodes its own share, so a rank that shares its GPU (the
    driver host's Worker) takes none: it would only compete for the same GPU
    and add a transfer. Hosts follow cluster.toml order, as setup numbers them.
    """
    from .monarch_policy_io import driver_host

    world = int(getattr(handle, "world", 0) or 0)
    per_host = max(1, int(getattr(handle, "gpus_per_host", 1) or 1))
    config = getattr(handle, "config", None)
    hosts = [getattr(host, "name", "") for host in (getattr(config, "hosts", ()) or ())]
    try:
        here = driver_host(config) if hosts else ""
    except Exception:
        here = ""
    gpu = _driver_gpu_index()
    remote = []
    for rank in range(world):
        host_index, gpu_index = divmod(rank, per_host)
        on_driver_host = (not hosts) or (host_index < len(hosts) and hosts[host_index] == here and bool(here))
        if not (on_driver_host and gpu_index == gpu):
            remote.append(rank)
    return remote


def _worker(handle: Any, rank: int) -> Any:
    per_host = max(1, int(handle.gpus_per_host))
    coords = {"gpus": rank % per_host}
    if "hosts" in list(handle.workers.extent.labels):
        coords["hosts"] = rank // per_host
    return handle.workers.slice(**coords)


def request_base(kind: str, source: dict[str, Any], env: dict[str, Any],
                 output_dtype: Any) -> dict[str, Any]:
    """Every field of a unit request but the unit index and its latent."""
    return {"schema": SCHEMA, "kind": kind, "vae": dict(source), "env": dict(env),
            "output_dtype": None if output_dtype is None else str(output_dtype)}


def sender(handle: Any, token: Any, base: dict[str, Any], inputs: list[Any]) -> Callable[[int, int], Any]:
    """``send(rank, unit)``: dispatch one unit without waiting for it."""

    def send(rank: int, unit: int) -> Any:
        worker = _worker(handle, rank)
        request = {**base, "unit": unit, "input": inputs[unit]}
        return mesh_setup.dispatch_endpoint(
            handle, ENDPOINT, lambda: getattr(worker, ENDPOINT).call_one(request), setup_token=token)

    return send


def waiter(handle: Any) -> Callable[[Any, float], Any]:
    """``wait(future, timeout_s)``: the reply, with a dead actor evicting the fleet."""

    def wait(future: Any, timeout_s: float) -> Any:
        return handle._await_or_evict(future, timeout_s)

    return wait


def acceptor(expected_dtype: Any, expected_shape: Callable[[int], tuple[int | None, ...]]
             ) -> Callable[[Any], str | None]:
    """Check a reply before the loop may use it: an ``ok`` CPU tensor of the unit's dtype and shape.

    ``expected_shape(unit)`` gives every dimension the plan knows; None is a
    dimension only the model knows (a tile's decoded size).
    """
    import torch

    def accept(reply: Any) -> str | None:
        if not isinstance(reply, dict) or reply.get("schema") != SCHEMA:
            return "the rank answered with no schema-1 reply; update the driver and the Workers together"
        where = f"rank {reply.get('rank')} on {reply.get('host')}"
        if not reply.get("ok"):
            return f"{where} declined: {reply.get('reason') or 'no reason given'}"
        output = reply.get("output")
        if not isinstance(output, torch.Tensor) or output.device.type != "cpu":
            return f"{where} returned no CPU tensor"
        if expected_dtype is not None and output.dtype != expected_dtype:
            return f"{where} returned {output.dtype}, the stock decode returns {expected_dtype}"
        unit = reply.get("unit")
        if not isinstance(unit, int):
            return f"{where} returned no unit index"
        shape = expected_shape(unit)
        got = tuple(int(n) for n in output.shape)
        if len(got) != len(shape) or any(want is not None and want != have for want, have in zip(shape, got, strict=True)):
            return f"{where} returned shape {got} for unit {unit}, expected {shape}"
        return None

    return accept
