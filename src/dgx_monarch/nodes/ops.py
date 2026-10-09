"""Ops nodes: cluster status and VRAM clearing (DESIGN.md section 5.7)."""
from __future__ import annotations

import json

from ..constants import MESH_TYPE, NODE_CATEGORY
from ..mesh import ensure_live
from .common import MeshSpec


class DGXMonarchStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mesh": (MESH_TYPE,)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "status"
    CATEGORY = NODE_CATEGORY
    OUTPUT_NODE = True

    def status(self, mesh: MeshSpec):
        results = ensure_live(mesh.handle).call_all("status", timeout_s=60)
        payload: dict[str, object] = {"workers": results}
        try:
            from ..gate_audit import trust_rows
            from ..gate_ledger import GateLedger
            from .gate import _ledger_dir

            entries = trust_rows(GateLedger(_ledger_dir()).entries())
            latest: dict[object, dict] = {}
            for e in entries:
                latest[e.get("key")] = e
            counts: dict[str, int] = {}
            for e in latest.values():
                counts[e.get("verdict", "?")] = counts.get(e.get("verdict", "?"), 0) + 1
            payload["identity_gates"] = {"combinations": counts,
                                         "ledger": "output/dgxm_gate_ledger.jsonl"}
        except Exception:
            pass
        text = json.dumps(payload, indent=2, default=str)
        return {"ui": {"text": [text]}, "result": (text,)}

    @classmethod
    def IS_CHANGED(cls, mesh):
        return float("nan")  # always re-poll


def _free_driver_models(level: str) -> dict:
    """Unload models resident in the ComfyUI driver process.

    Mesh ``clear_vram`` reaches only worker DiTs, not the driver's text encoder or
    VAE. On UMA, offloading to host RAM does not free the physical pool; report
    ``pool_freed_gib`` rather than assuming an unload returned all memory.
    ``soft`` clears allocator caches and ``hard`` also unloads models, which reload
    on demand. See docs/VALIDATION.md for retained-memory measurements.
    """
    import socket

    import comfy.model_management as mm
    import psutil

    before = psutil.virtual_memory().available
    unloaded = False
    if level == "hard":
        mm.unload_all_models()
        unloaded = True
    mm.soft_empty_cache()
    after = psutil.virtual_memory().available
    return {
        "host": socket.gethostname(),
        "level": level,
        "unloaded_models": unloaded,
        "pool_freed_gib": round((after - before) / 2**30, 2),
        "pool_available_gib": round(after / 2**30, 2),
    }


class DGXMonarchClearVRAM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": (MESH_TYPE,),
                "level": (["soft", "hard", "recycle"], {"default": "soft",
                          "tooltip": "soft frees allocator caches. hard also unloads every "
                          "resident model. recycle resets the Attached mesh: the worker processes "
                          "stop, which frees their models and returns the retained allocator pool "
                          "(the orange 'pool' gauge segment, about one model's size per worker) to "
                          "the OS for a co-resident LLM; no other level returns it. An active "
                          "render or result lease blocks recycle and leaves the workers running; "
                          "retry when it ends. The next render pays a fresh bring-up (about 45-60 s)."}),
            },
            "optional": {
                "include_driver": ("BOOLEAN", {"default": True,
                    "tooltip": "Also free this ComfyUI process's own models (text encoder and VAE), not "
                    "only the worker DiTs. hard and recycle unload them; soft only releases caches. On "
                    "a discrete GPU this frees their VRAM. On unified memory comfy offloads them to "
                    "process RAM instead of dropping them, so read pool_freed_gib in the status "
                    "rather than assume the whole footprint is gone. "
                    "They reload on the next render."}),
                "samples": ("LATENT", {"tooltip": "Optional pass-through. When connected, the node "
                            "returns this latent only after cleanup succeeds, so a VAE decode connected "
                            "to it runs after that cleanup."}),
            },
        }

    RETURN_TYPES = ("STRING", "LATENT")
    RETURN_NAMES = ("status", "samples")
    FUNCTION = "clear"
    CATEGORY = NODE_CATEGORY
    OUTPUT_NODE = True

    def clear(self, mesh: MeshSpec, level: str = "soft", include_driver: bool = True,
              samples=None):
        from ..telemetry import emit

        handle = ensure_live(mesh.handle)
        if level == "recycle":
            # The typed path checks active sessions and sample-result leases
            # under the lifecycle lock before any destructive action. There is
            # no pre-stop clear_vram RPC: proc exit returns the model and arena
            # together, while a busy refusal leaves active work untouched.
            outcome = handle.recycle_detailed()
            results: dict[str, object] = {"recycle": outcome.as_dict()}
            if outcome.ok:
                # The same drain the /dgxm/recycle route runs: this node stops
                # the procs through the same call, so without it the driver's
                # residency memos would credit weights the next fleet lacks.
                from . import recycle_drain

                recycle_drain.drop_residency_memos()
                results["recycled"] = outcome.detail
            else:
                results["recycled"] = (
                    f"RECYCLE {outcome.status.value.upper()}: {outcome.detail}")
                if samples is not None:
                    raise RuntimeError(
                        "cannot forward samples: Attached-mesh reset did not complete "
                        f"({outcome.status.value}: {outcome.detail})")
        else:
            results = {"workers": handle.call_all("clear_vram", level, timeout_s=300)}
        if include_driver and (level != "recycle" or outcome.ok):
            results["driver"] = _free_driver_models("hard" if level == "recycle" else level)
        elif include_driver:
            results["driver"] = (
                "not freed because the Attached-mesh reset did not complete; active or "
                "unknown lifecycle state remains untouched")
        text = json.dumps(results, indent=2, default=str)
        emit("clear_vram", level=level)
        return {"ui": {"text": [text]}, "result": (text, samples)}

    @classmethod
    def IS_CHANGED(cls, mesh, level, include_driver=True, samples=None):
        return float("nan")
