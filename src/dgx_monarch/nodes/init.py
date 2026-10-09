"""DGXMonarchInit: cluster attach + topology policy. Outputs MESH."""
from __future__ import annotations

from .. import family_select, first_render
from ..adapters import SELECTABLE_FAMILIES
from ..config import find_config_path
from ..constants import MESH_TYPE, NODE_CATEGORY
from ..mesh import config_fingerprint, get_mesh
from ..topology import PRESETS, topology_from_preset
from .common import MeshSpec

# Widget vocabulary, read-only after import.
_ATTENTION_KERNELS = ["TORCH_FLASH", "TORCH_CUDNN", "TORCH_EFFICIENT", "FA", "FA3", "SAGE_AUTO", "SAGE_FP8", "SAGE_FP16", "SOL_ATTN_TAU0.6", "SOL_ATTN_TAU0.7", "SOL_ATTN_TAU1.0"]
# Derive selectable families from the adapter registry. Read-only after import.
_FAMILY_CHOICES = [family_select.AUTO, *SELECTABLE_FAMILIES]


class DGXMonarchInit:
    """Attach workers and select an explicit or per-render automatic topology."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "topology": (["auto", *PRESETS.keys()], {"default": "auto"}),
                "mode": (["auto", "local", "cluster"], {"default": "auto",
                         "tooltip": "auto uses the cluster when it finds a cluster.toml, else the "
                         "local GPUs. cluster refuses without a cluster.toml."}),
            },
            # Append-only contract: ComfyUI persists widget values by position.
            # Add new widgets at the end and append the matching INIT_ORDERS row
            # in web/js/dgx_monarch_widgets.js.
            "optional": {
                "attention": (_ATTENTION_KERNELS, {"default": "TORCH_FLASH", "advanced": True}),
                "config_path": ("STRING", {"default": "", "advanced": True,
                                "tooltip": "Path to a cluster.toml. Empty searches $DGXM_CLUSTER_TOML, "
                                "./cluster.toml, then ~/.config/dgx-monarch/cluster.toml."}),
                "gpus_per_host": ("INT", {"default": 0, "min": 0, "max": 64, "advanced": True,
                                  "tooltip": "Local mode only. 0 uses every visible GPU."}),
                "reserve_vram_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.5,
                                    "advanced": True,
                                    "tooltip": "VRAM in GiB that ComfyUI keeps in reserve in each worker. "
                                    "0 keeps the default: cluster.toml [worker_args] if set, else 8 GiB "
                                    "on unified memory (docs/VALIDATION.md) and ComfyUI's own on a "
                                    "discrete GPU."}),
                "sync_ulysses": ("BOOLEAN", {"default": True, "advanced": True}),
                "compile_dit": ("BOOLEAN", {"default": False, "advanced": True,
                                "tooltip": "torch.compile the DiT blocks (max-autotune fp8, no cudagraphs). "
                                "Measured 21% faster on fp8-scaled Krea2 at cfg2 on GB10 (2026-07-04); "
                                "the first render adds a one-time autotune warmup. Chroma, Flux, Flux2 "
                                "and LongCat have no `blocks` list, so they stay eager. Opt in and "
                                "validate per model."}),
                "pipeline_depth": ("INT", {"default": 1, "min": 1, "max": 8, "advanced": True,
                                   "tooltip": "How many renders the KSampler Pipeline node keeps in "
                                   "flight. 1, the default, runs them one after another. 2 or more "
                                   "overlaps the next render's transfers with the current render's "
                                   "compute; that node's description says when this helps."}),
                "mmap_fallback": ("BOOLEAN", {"default": False, "advanced": True,
                                 "tooltip": "Diagnostic. Off (default): on unified memory the worker reads "
                                 "safetensors with pread, a direct file read that avoids the mmap "
                                 "page-cache spike at load (FSDP loads keep mmap); output is "
                                 "byte-identical. On: use the stock mmap loader instead, for an A/B "
                                 "comparison."}),
                "lora_low_rss": (["auto", "on", "off"], {"default": "auto", "advanced": True,
                                 "tooltip": "LoRA memory mode. auto (default): on for unified-memory GPUs (DGX "
                                 "Spark), off for discrete ones. on: bake at full render speed, then replace "
                                 "comfy's original-weight backup with byte-verified references into the checkpoint "
                                 "file. That frees about the LoRA-covered share of the model (24 GiB on Krea2 bf16, "
                                 "2026-07-07) for a co-resident LLM; output is bit-identical. A stack change "
                                 "un-bakes in place from the checkpoint, exact on every quant (fp8, int8, mxfp8 "
                                 "and nvfp4 included), in about 10-23 s (2026-07-07 and 2026-07-08); a checkpoint "
                                 "it cannot capture this way reloads in full on a stack change. off: stock comfy, "
                                 "with sub-second strength changes from a resident backup that costs up to one "
                                 "more model in the pool. A LoRA stack under FSDP needs it on."}),
                "slab_weights": (["auto", "on", "off"], {"default": "auto", "advanced": True,
                                 "tooltip": "Zero-copy weight residency (experimental). on: the DiT's "
                                 "weights live in a shared-memory slab the GPU reads directly "
                                 "(GB10-class unified memory), with no cudaMalloc copy, no CPU staging "
                                 "at load and no retained per-process arena. On Krea2 bf16 (24.5 GiB, "
                                 "2026-07-08) the raw pread took 8.5 s and full load readiness 20.1 s; "
                                 "unloading returns the memory to the OS at once. Output is "
                                 "bit-identical; gate it. Needs lora_low_rss; ignored under FSDP, and "
                                 "under compile_dit except on Chroma, Flux, Flux2 and LongCat. auto "
                                 "(default): on unified-memory GPUs, slab for gate-vouched families "
                                 "(currently krea2 and flux2). A checkpoint's first load stays stock "
                                 "while its family is learned, every later load uses the slab; other "
                                 "families and discrete GPUs stay stock. off: stock cudaMalloc weights."}),
                "auto_gate": (["first_use", "off"], {"default": "first_use", "advanced": True,
                              "tooltip": "first_use (default): before the first render of a model, "
                              "residency and LoRA combination the gate ledger cannot vouch for, run one "
                              "automatic identity gate (2-4 short proof renders, once per combination); "
                              "a FAIL or ERROR turns the unproved paths off before the full render. "
                              "off: skip that proof and own the accuracy risk. off withholds no feature: "
                              "LoRA renders, lora_low_rss, slab_weights and the capacity rescue all run "
                              "as configured, on the levers first_use would have proved. What off "
                              "costs: it records no PASS, so Fleet keeps forcing slab_weights and "
                              "lora_low_rss off for an unproved combination until you run `dgxm gate` "
                              "or the Identity Gate node. A ceremony that already FAILED still "
                              "quarantines both levers, because that verdict is measured, not "
                              "missing. See docs/TROUBLESHOOTING.md #67."}),
                "load_profile": ("BOOLEAN", {"default": False, "advanced": True,
                                 "tooltip": "Diagnostic. Profile each load_models_gpu call in "
                                 "the workers with cProfile and, for a call slower than 2 s, write "
                                 "the top 30 entries by cumulative time to "
                                 "/tmp/dgxm-load-profile.<pid>.<t>.<random>.txt. "
                                 "Leave off for normal use."}),
                "swap_verify": ("INT", {"default": 2, "min": -1, "max": 64, "advanced": True,
                                "tooltip": "Keys per lazy swap to bit-check against comfy's own "
                                "bake math (ambient verify). 0 turns it off; -1 checks every key "
                                "and adds about 1 s per key. A mismatch drops the slot for a clean "
                                "stock reload."}),
                "uma_reserve_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.5,
                                   "advanced": True,
                                   "tooltip": "Unified-memory co-residency guard, in GiB: warn when "
                                   "a model load or swap leaves less MemAvailable than this for an "
                                   "LLM server or other tenants of the pool, and refuse a driver-side "
                                   "weight load that would eat into it (docs/TROUBLESHOOTING.md #52). "
                                   "0 turns the warning off; the 5 GiB refusal floor still applies."}),
                "comfy_managed": (["off", "on"], {"default": "off", "advanced": True,
                                  "tooltip": "Experimental, opt-in, single-model renders only. "
                                  "Brings ComfyUI's own DynamicVRAM (comfy-aimdo) up inside the "
                                  "workers, as stock ComfyUI's main.py does, and loads the DiT "
                                  "through it: comfy places and pages the weights itself, so a "
                                  "load behaves like stock ComfyUI on this box, not like a "
                                  "distributed worker. It has no `auto` and is never a default. "
                                  "It forces lora_low_rss and slab_weights off, and it refuses, "
                                  "rather than runs degraded, anything it has not been proven against: "
                                  "any LoRA stack, FSDP, dual-model renders, and the RDMA latent "
                                  "return. Changing it needs an Attached mesh reset. "
                                  "See docs/TROUBLESHOOTING.md #62."}),
                "family_adapter": (_FAMILY_CHOICES, {"default": family_select.AUTO, "advanced": True,
                                   "tooltip": "Which model family's adapter handles this "
                                   "checkpoint. auto (default): detection picks it. Name a family "
                                   "for a finetune whose header detection reads wrong or not at "
                                   "all: that family's topology row, capacity preflights and "
                                   "adapter apply, even when ComfyUI built a derived subclass that "
                                   "detection refuses. A named family that does not fit the model "
                                   "ComfyUI built gets a typed refusal before any kernel runs, "
                                   "never a wrong render. It cannot make ComfyUI load a checkpoint "
                                   "ComfyUI cannot detect, and it does not correct a misread "
                                   "quantization. A forced run is gated as its own combination and "
                                   "never inherits a family's slab vouching: for zero-copy "
                                   "residency on a finetune, set slab_weights=on yourself. "
                                   "See docs/MODELS.md."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (MESH_TYPE,)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "init"
    CATEGORY = NODE_CATEGORY

    @classmethod
    def IS_CHANGED(cls, topology, mode, config_path="", **kwargs):
        """Invalidate Comfy's cached MESH output when cluster.toml changes."""
        try:
            resolved = find_config_path(config_path or None)
            use_cluster = mode == "cluster" or (mode == "auto" and resolved is not None)
            if not use_cluster or resolved is None:
                return "local"
            return f"{resolved}:{config_fingerprint(resolved)}"
        except Exception as exc:
            # Preserve the actual Init error while ensuring a repaired config
            # is not pinned behind the previous cached token.
            return f"config-error:{type(exc).__name__}:{exc}"

    def init(self, topology, mode, attention="TORCH_FLASH", config_path="", gpus_per_host=0,
             reserve_vram_gb=0.0, sync_ulysses=True, compile_dit=False, pipeline_depth=1,
             mmap_fallback=False, lora_low_rss="auto", slab_weights="auto",
             auto_gate="first_use", load_profile=False, swap_verify=2,
             uma_reserve_gb=0.0, comfy_managed="off", family_adapter=family_select.AUTO,
             unique_id=None):
        return self._init_with_bootstrap_policy(
            topology=topology,
            mode=mode,
            attention=attention,
            config_path=config_path,
            gpus_per_host=gpus_per_host,
            reserve_vram_gb=reserve_vram_gb,
            sync_ulysses=sync_ulysses,
            compile_dit=compile_dit,
            pipeline_depth=pipeline_depth,
            mmap_fallback=mmap_fallback,
            lora_low_rss=lora_low_rss,
            slab_weights=slab_weights,
            auto_gate=auto_gate,
            load_profile=load_profile,
            swap_verify=swap_verify,
            uma_reserve_gb=uma_reserve_gb,
            comfy_managed=comfy_managed,
            family_adapter=family_adapter,
            disable_custom_nodes=False,
            unique_id=unique_id,
        )

    def _init_with_bootstrap_policy(
        self, topology, mode, attention="TORCH_FLASH", config_path="", gpus_per_host=0,
        reserve_vram_gb=0.0, sync_ulysses=True, compile_dit=False, pipeline_depth=1,
        mmap_fallback=False, lora_low_rss="auto", slab_weights="auto",
        auto_gate="first_use", load_profile=False, swap_verify=2,
        uma_reserve_gb=0.0, comfy_managed="off", family_adapter=family_select.AUTO,
        disable_custom_nodes=False, unique_id=None,
    ):
        """Initialize with an explicit custom-node bootstrap policy."""
        handle = get_mesh(config_path=config_path, mode=mode, gpus_per_host=gpus_per_host)

        worker_args = {"mmap_fallback": bool(mmap_fallback)}
        # Omit automatic policy so workers resolve it from hardware. Keep
        # boolean spellings compatible with saved workflows.
        explicit = {"on": True, "true": True, "off": False, "false": False}.get(
            str(lora_low_rss).strip().lower())
        if explicit is not None:
            worker_args["lora_low_rss"] = explicit
        slab_explicit = {"on": True, "true": True, "off": False, "false": False}.get(
            str(slab_weights).strip().lower())
        if slab_explicit is not None:
            worker_args["slab_weights"] = slab_explicit
        # Resolve this after both residency levers because it overrides them.
        # Emit the key only when enabled; worker args are part of gate contexts,
        # so an unconditional key would invalidate existing PASS rows.
        if str(comfy_managed).strip().lower() in ("on", "true"):
            if slab_explicit is True:
                raise ValueError(
                    "comfy_managed=on and slab_weights=on are two different residencies "
                    "for the same weights. Set slab_weights to auto or off, or turn "
                    "comfy_managed off.")
            if explicit is True:
                raise ValueError(
                    "comfy_managed=on forces lora_low_rss off (docs/TROUBLESHOOTING.md #62); "
                    "set lora_low_rss to auto or off, or turn comfy_managed off.")
            worker_args["comfy_managed"] = True
            worker_args["slab_weights"] = False
            worker_args["lora_low_rss"] = False
        # Same rule as the residency levers above, and for the same reason.
        # Absent means auto, which is what every saved workflow and every
        # recorded row already means.
        forced_family = family_select.override_from_worker_args(
            {family_select.WORKER_ARG_KEY: family_adapter})
        if forced_family is not None:
            if forced_family not in SELECTABLE_FAMILIES:
                raise ValueError(
                    f"family_adapter={forced_family!r} names no dgx-monarch adapter. "
                    f"Choose one of: {', '.join(SELECTABLE_FAMILIES)}, or 'auto'.")
            worker_args[family_select.WORKER_ARG_KEY] = forced_family
        if load_profile:
            worker_args["load_profile"] = True
        worker_args["swap_verify"] = int(swap_verify)
        if uma_reserve_gb > 0:
            worker_args["uma_reserve_gb"] = float(uma_reserve_gb)
        if reserve_vram_gb > 0:
            worker_args["reserve_vram_gb"] = reserve_vram_gb
        if compile_dit:
            worker_args["compile_dit"] = True
        if disable_custom_nodes:
            worker_args["disable_custom_nodes"] = True

        spec = MeshSpec(
            handle=handle,
            topology_preset=topology,
            attention=attention,
            sync_ulysses=bool(sync_ulysses),
            worker_args=worker_args,
            pipeline_depth=int(pipeline_depth),
            auto_gate=str(auto_gate),
        )

        if topology != "auto":
            # Explicit presets initialize groups for eager loading. Automatic
            # topology waits for the first render's family and resolution.
            topo = topology_from_preset(topology, handle.world)
            from . import render_preflight
            from .render_session import mutation_render_session

            render_preflight.preflight_sol_sequence_parallel(topo, attention)
            with mutation_render_session(handle):
                handle.ensure_setup(
                    topo, attention, bool(sync_ulysses), worker_args)
        else:
            first_render.nccl_deferred()

        return (spec,)
