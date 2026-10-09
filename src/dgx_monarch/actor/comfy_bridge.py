"""Bootstrap ComfyUI inside a Monarch worker proc.

comfy.cli_args parses sys.argv at import time, so the argv guard must run
before the first `import comfy.*` anywhere in the proc. All comfy imports in
the actor package go through ensure_comfy() first.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time
from typing import Any, cast

from ..log import get_logger
from ..transfer_utils import (
    failure_summary,
    raise_with_distinct_cause,
    reconcile_error,
    safe_call,
)
from . import comfy_dynamic, slab_lifetime
from .comfy_custom_nodes import (
    load_custom_node_modules as _load_custom_node_modules,
)
from .comfy_custom_nodes import (
    preload_failure_hint as _preload_failure_hint,  # noqa: F401 - compatibility
)
from .pread_backend import _disable_pread_backend, _enable_pread_backend
from .slab_arena import _finish_prepublished, _SlabHookRestore

log = get_logger(__name__)

_BOOTSTRAPPED: str | None = None  # proc lifetime: ensure_comfy sets it once and a later disagreement raises
_CUSTOM_NODES_DISABLED: bool | None = None  # proc lifetime: same latch, same one-way rule
_MEMORY_BASELINE_MODULE = None  # proc lifetime: the module identity the baseline below was taken under
_MEMORY_BASELINE: dict[str, object] = {}  # proc lifetime: comfy's untouched knobs, retaken on a module swap


def _patch_uma_free_memory(uma_pool_shares: int = 1) -> None:
    """Report GB10's shared host/device memory pool to ComfyUI.

    ``torch.cuda.mem_get_info`` under-reports available memory on GB10, causing
    ComfyUI to offload models and stall at "0 MB usable". This can recur during
    every ``prepare_sampling`` call, so a one-time full load is insufficient.
    Use psutil's host-available memory divided by the number of GPUWorker actors
    on this host, preventing concurrent actors from each budgeting the full pool.
    The patch is idempotent and applies only to integrated CUDA devices.
    See docs/TROUBLESHOOTING.md #12 for the original measurements.
    """
    import comfy.model_management as mm

    if getattr(mm.get_free_memory, "_dgxm_uma", False):
        return
    import psutil
    import torch

    try:
        if not getattr(torch.cuda.get_device_properties(mm.get_torch_device()), "is_integrated", 0):
            return  # discrete GPU: real VRAM, cuda.mem_get_info is correct
    except Exception as exc:
        safe_call(
            log.warning,
            "UMA free-memory patch skipped (%s); estimate loader may mis-size",
            failure_summary(exc),
        )
        return

    shares = max(1, int(uma_pool_shares))

    def _uma_free_memory(dev=None, torch_free_too=False):
        if dev is None:
            dev = mm.get_torch_device()
        host_available = psutil.virtual_memory().available // shares
        if hasattr(dev, "type") and dev.type in ("cpu", "mps"):
            return (host_available, host_available) if torch_free_too else host_available
        stats = torch.cuda.memory_stats(dev)
        mem_free_torch = stats["reserved_bytes.all.current"] - stats["active_bytes.all.current"]
        mem_free_total = host_available + mem_free_torch
        return (mem_free_total, mem_free_torch) if torch_free_too else mem_free_total

    cast(Any, _uma_free_memory)._dgxm_uma = True
    mm.get_free_memory = _uma_free_memory
    log.info(
        "patched get_free_memory for GB10 unified memory (psutil host-available/%d)",
        shares,
    )


def _uma_memory_defaults(worker_args: dict, integrated: bool | None = None) -> dict:
    """Default the loop-hosted workers to the validated single-Spark memory posture.

    The workers hold the DiT, so on unified memory they default to the driver's
    validated baseline (--disable-pinned-memory --disable-async-offload
    --reserve-vram), behind the is_integrated gate _patch_uma_free_memory uses;
    an explicit worker_arg still wins. Discrete GPUs keep comfy defaults (pinned host RAM costs no
    VRAM there, and async offload overlaps weight copies with compute). On a fully loaded
    render comfy gates these knobs off, so they do not shrink the resident set:
    they give baseline parity and partial-load insurance, not a smaller render
    working set.
    """
    # This function is also the actor-side trust boundary.  A caller can reach
    # setup/apply_worker_args without loading cluster.toml, so validate again
    # here instead of relying on driver-side parsing.  Never coerce strings:
    # bool("off") is True.
    from ..config_schema import validate_worker_args

    worker_args = comfy_dynamic.apply_policy(validate_worker_args(
        worker_args, context="worker_args", reject_unknown=False))
    if integrated is None:
        import comfy.model_management as mm
        import torch

        try:
            integrated = bool(torch.cuda.get_device_properties(mm.get_torch_device()).is_integrated)
        except Exception as exc:
            safe_call(
                log.warning,
                "UMA posture probe failed (%s); this worker keeps stock comfy "
                "defaults and skips the unified-memory posture",
                failure_summary(exc),
            )
            return worker_args
    if not integrated:
        return worker_args
    out = dict(worker_args)
    out.setdefault("disable_pinned_memory", True)
    out.setdefault("disable_async_offload", True)
    out.setdefault("disable_smart_memory", True)
    if not out.get("reserve_vram_gb"):
        out["reserve_vram_gb"] = 8.0
    # pread is the default UMA load backend: a direct file read that avoids the
    # safetensors mmap page-cache spike (the mmap sits in the same physical pool
    # as the materialized weights). Byte-identical output. mmap_fallback opts a
    # render back to the stock mmap loader for A/B.
    if not out.get("mmap_fallback"):
        out.setdefault("safetensors_backend", "pread")
    # Low-RSS LoRA is the UMA default to avoid a model-sized backup in the shared
    # pool. Lazy un-bake preserves exact stack changes (docs/VALIDATION.md).
    # Explicit Init settings win. Discrete GPUs keep stock backups in host RAM,
    # where baked hot-swap is faster and does not consume VRAM.
    out.setdefault("lora_low_rss", True)
    # slab_weights `auto` resolves per family at load time: slab for families
    # in capacity_fit.SLAB_VOUCHED_FAMILIES, each carrying its own passed
    # identity ceremony (docs/VALIDATION.md carries the dated results), and
    # stock cudaMalloc for everything else. Growing that set is an evidence
    # change, not a code change here. A checkpoint's family is knowable
    # only after a load, so the first load of a new file stays stock and
    # memoizes the detected family by file identity; every later load of the
    # same bytes slab-loads. Explicit on/off from the Init widget wins. It
    # requires low_rss: baked mode keeps comfy's full original-weight backup
    # on offload_device, which slab mode pins to the GPU. That backup would
    # cost +1x model of cudaMalloc next to the slab.
    out.setdefault("slab_weights", "auto")
    if out.get("slab_weights") and not out.get("lora_low_rss"):
        log.warning("slab_weights requires lora_low_rss; weights stay on cudaMalloc")
        out["slab_weights"] = False
    return out


@contextlib.contextmanager
def slab_load(path: str, *, handoff: list[Any] | None = None):
    """Make Comfy adopt one slab; caller closes it only after freeing the model."""
    import comfy.model_base
    import comfy.utils
    import torch

    from .slab import WeightSlab

    owner: list[WeightSlab] = handoff if handoff is not None else []
    owner_mark = len(owner)
    try:
        WeightSlab(path, handoff=owner)
        weight_slab = owner[-1]
    except BaseException as primary:
        construction_cleanup_error: BaseException | None = None
        if len(owner) > owner_mark:
            try:
                slab_lifetime.close_slab_or_retain_after_explicit_unload(
                    owner[owner_mark], "slab construction handoff")
            except BaseException as caught:
                construction_cleanup_error = caught
        if construction_cleanup_error is not None:
            strongest, cause = reconcile_error(
                primary,
                construction_cleanup_error,
                "slab construction handoff cleanup also failed",
            )
            raise_with_distinct_cause(strongest, cause)
        raise
    orig_ltf = comfy.utils.load_torch_file
    orig_lmw = comfy.model_base.BaseModel.load_model_weights
    tensors_exposed = False

    def _slab_load_torch_file(ckpt, safe_load=False, device=None,
                              return_metadata=False):
        nonlocal tensors_exposed
        if ckpt != path:
            return orig_ltf(ckpt, safe_load=safe_load, device=device,
                            return_metadata=return_metadata)
        tensors_exposed = True
        sd = weight_slab.state_dict()
        md = weight_slab.metadata or None
        return (sd, md) if return_metadata else sd

    def _assign_load_model_weights(self, sd, unet_prefix="", assign=False):
        # Record stripped key -> region so reabsorb can restore module names.
        for key, tensor in sd.items():
            if isinstance(tensor, torch.Tensor):
                region = weight_slab.region_by_ptr(tensor.data_ptr())
                if region is not None:
                    module_key = (key[len(unet_prefix):]
                                  if unet_prefix and key.startswith(unet_prefix)
                                  else key)
                    weight_slab.key_map[module_key] = region
        want: dict[str, torch.dtype] = {}
        for name, p in self.diffusion_model.named_parameters():
            want[unet_prefix + name] = p.dtype
        for name, b in self.diffusion_model.named_buffers():
            want[unet_prefix + name] = b.dtype
        for key in list(sd.keys()):
            tensor = sd[key]
            want_dtype = want.get(key)
            if (want_dtype is not None and isinstance(tensor, torch.Tensor)
                    and tensor.dtype != want_dtype
                    and tensor.dtype.is_floating_point
                    and want_dtype.is_floating_point):
                sd[key] = tensor.to(want_dtype)
                weight_slab.cast_keys.append(key)
        return orig_lmw(self, sd, unet_prefix=unet_prefix, assign=True)

    restore = _SlabHookRestore(
        comfy.utils, comfy.model_base.BaseModel, orig_ltf, orig_lmw, weight_slab)
    try:
        # Keep the slab's provisional root until the hook owner is itself
        # bound and prepublished; there must be no unowned instruction gap.
        restore.bind()
        restore.prepublish()
        weight_slab.confirm_handoff()
        try:
            comfy.utils.load_torch_file = _slab_load_torch_file
            comfy.model_base.BaseModel.load_model_weights = _assign_load_model_weights
            yield weight_slab
        finally:
            _finish_prepublished(restore, sys.exception())
    except BaseException as load_exc:
        # Retain raw-pointer tensors until a confirmed global unload clears frames.
        load_cleanup_error: BaseException | None = None
        if tensors_exposed:
            try:
                slab_lifetime.retain(weight_slab, load_exc)
            except BaseException as caught:
                load_cleanup_error = caught
        else:
            try:
                slab_lifetime.close_slab_or_retain_after_explicit_unload(
                    weight_slab, "pre-exposure slab load")
            except BaseException as caught:
                load_cleanup_error = caught
        if load_cleanup_error is not None:
            strongest, cause = reconcile_error(
                load_exc,
                load_cleanup_error,
                "slab load cleanup also failed",
            )
            raise_with_distinct_cause(strongest, cause)
        raise


def load_diffusion_model_slab(path: str, model_options: dict, loader,
                              unet_name: str | None = None, *,
                              handoff: list[Any] | None = None):
    """Load through the slab, or stock-load a dtype the slab cannot wrap."""
    from ..capacity_fit import stock_load_fit
    from ..mesh_safety import StockLoadCapacityError
    from ..refusal import RefusalClass, refusal
    from ..safetensors_header import UnsupportedSafetensorsDtypeError

    owner = handoff if handoff is not None else []
    owner_mark = len(owner)
    slab = None
    try:
        with slab_load(path, handoff=owner) as slab:
            base = loader(path, model_options=model_options)
    except BaseException as load_exc:
        owned_slab = (
            slab
            if slab is not None
            else owner[owner_mark]
            if len(owner) > owner_mark
            else None
        )
        if owned_slab is not None or not isinstance(
                load_exc, UnsupportedSafetensorsDtypeError):
            base = None
            slab_lifetime.cleanup_failed_load(
                owned_slab, load_exc, "slab model load")
            raise
        # A valid unsupported dtype is a capability miss; corruption is not.
        safe_call(
            log.warning,
            "slab setup cannot represent a checkpoint dtype for %s (%s); "
            "falling back to stock non-slab loading",
            os.path.basename(path),
            failure_summary(load_exc),
        )
        # The fallback is a real stock load, so it takes the ladder's price: the file plus the host copy it is
        # placed from. A bare file-size wall would admit the whole 1.0x to 2.1x band.
        fit = stock_load_fit(path, model_options)
        if fit.applies and not fit.fits:
            raise StockLoadCapacityError(refusal(
                RefusalClass.CAPACITY, f"stock residency cannot load {unet_name or os.path.basename(path)}: the load needs "
                f"{fit.required_gib} GiB (a {fit.size_gib} GiB file, the host copy it is placed from and the "
                f"host floor) and only {fit.avail_gib} GiB of unified memory is available. Slab residency cannot "
                "help: the slab loader cannot represent this checkpoint's dtype, so it fell back to a stock load, "
                "and no capacity consent can change that. What would fit: more free unified memory on this host, "
                "or a pruned or quantized artifact.", guard="stock_load_preflight", waivable=False)) from load_exc
        return loader(path, model_options=model_options), None, True

    # Pin offload to GPU so detach cannot copy/strand the slab in a CPU transient.
    try:
        import comfy.model_management as mm

        base.offload_device = mm.get_torch_device()
    except BaseException as pin_exc:
        base = None
        slab_lifetime.cleanup_failed_load(slab, pin_exc, "slab offload-device pin")
        raise
    return base, slab, False


def ensure_comfy(
    comfy_dir: str,
    worker_args: dict | None = None,
    *,
    gpus_per_host: int = 1,
) -> None:
    """Idempotent per-proc comfy bootstrap. Safe to call from every endpoint."""
    global _BOOTSTRAPPED, _CUSTOM_NODES_DISABLED
    custom_nodes_disabled = bool((worker_args or {}).get("disable_custom_nodes"))
    if _BOOTSTRAPPED is not None:
        if _BOOTSTRAPPED != comfy_dir:
            raise RuntimeError(
                f"this worker proc already bootstrapped ComfyUI from {_BOOTSTRAPPED} and "
                f"cannot bootstrap again from {comfy_dir}. Reset the attached mesh, or restart "
                "the Worker service (`dgxm restart`).")
        if _CUSTOM_NODES_DISABLED is not custom_nodes_disabled:
            raise RuntimeError(
                "this worker proc already bootstrapped with a different custom-node "
                "policy (disable_custom_nodes). Reset the attached mesh, or restart the "
                "Worker service (`dgxm restart`), so a fresh process takes the new policy")
        _apply_worker_args(_uma_memory_defaults(worker_args or {}))
        return

    if not os.path.isdir(comfy_dir) or not os.path.isfile(os.path.join(comfy_dir, "comfy", "sd.py")):
        raise FileNotFoundError(
            f"ComfyUI not found at {comfy_dir!r} on this host. Every worker host needs a ComfyUI "
            "checkout at the same path (or set hosts[].comfy_dir in cluster.toml)."
        )

    # comfy.cli_args must not see monarch/bootstrap argv.
    sys.argv = ["dgx-monarch-worker"]
    if comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)

    os.environ.setdefault("XDIT_LOGGING_LEVEL", "WARN")

    # Stage A of comfy's own DynamicVRAM bring-up: aimdo.so into the global
    # namespace. Off unless the operator asked; raises rather than falling back.
    dynamic = comfy_dynamic.stage_a(worker_args or {})

    import comfy.model_management  # noqa: F401  (import order: after argv guard)

    _patch_uma_free_memory(gpus_per_host)
    _instrument_gpu_load()

    _BOOTSTRAPPED = comfy_dir
    _CUSTOM_NODES_DISABLED = custom_nodes_disabled
    effective = _uma_memory_defaults(worker_args or {})
    _apply_worker_args(effective)
    # Stage B: hooks, device, patcher rebind. After the reserve is resolved and
    # before custom nodes load, mirroring stock ComfyUI main.py's order.
    comfy_dynamic.stage_b(dynamic, reserve_vram_gb=effective.get("reserve_vram_gb"))
    if not custom_nodes_disabled:
        _load_custom_node_modules()
    log.info("comfy bootstrapped from %s", comfy_dir)


_GPU_LOAD_S: list = [0.0]  # proc lifetime, one slot: accumulates until gpu_load_seconds_reset zeroes it


def _record_gpu_load_profile(profiler: Any, t0: float) -> None:
    took = time.perf_counter() - t0
    _GPU_LOAD_S[0] += took
    if profiler is None:
        return
    profiler.disable()
    if took < 2.0:
        return
    import io
    import pstats
    import tempfile

    buf = io.StringIO()
    pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(30)
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"dgxm-load-profile.{os.getpid()}.{int(t0)}.",
            suffix=".txt",
            delete=False,
        ) as output_file:
            output_path = output_file.name
            output_file.write(f"load_models_gpu took {took:.1f}s\n")
            output_file.write(buf.getvalue())
        log.info("gpu-load profile (%.1fs) written to %s", took, output_path)
    except OSError:
        pass


def _instrument_gpu_load() -> None:
    """Accumulate time in ComfyUI's ``load_models_gpu``; install only once.

    Samples return ``gpu_load_s`` so load stalls can be distinguished from slow
    denoising. ``nodes/render_result.py`` displays values above one second.
    """
    import comfy.model_management as mm

    if getattr(mm.load_models_gpu, "_dgxm_timed", False):
        return
    inner = mm.load_models_gpu

    def timed(*a, **k):
        t0 = time.perf_counter()
        profiler = None
        if os.environ.get("DGXM_LOAD_PROFILE"):
            import cProfile

            profiler = cProfile.Profile()
            profiler.enable()
        primary_error: BaseException | None = None
        try:
            return inner(*a, **k)
        except BaseException as caught:
            primary_error = caught
            raise
        finally:
            try:
                _record_gpu_load_profile(profiler, t0)
            except BaseException as finalize_error:
                if primary_error is None:
                    if not isinstance(finalize_error, Exception):
                        raise
                else:
                    strongest, cause = reconcile_error(
                        primary_error,
                        finalize_error,
                        "gpu-load profiling finalization also failed",
                    )
                    if strongest is not primary_error:
                        raise_with_distinct_cause(strongest, cause)

    cast(Any, timed)._dgxm_timed = True
    mm.load_models_gpu = timed


def gpu_load_seconds_reset() -> float:
    """Return the accumulated load_models_gpu time and reset the counter."""
    v = _GPU_LOAD_S[0]
    _GPU_LOAD_S[0] = 0.0
    return v


def _apply_worker_args(worker_args: dict) -> None:
    """Apply the curated subset of comfy CLI behavior that matters on workers.

    Keys: reserve_vram_gb (float), disable_pinned_memory (bool), disable_async_offload
    (bool), disable_smart_memory (bool), safetensors_backend ("pread"|unset). The UMA
    guidance behind the defaults is in docs/VALIDATION.md.
    """
    import comfy.model_management as mm

    global _MEMORY_BASELINE_MODULE, _MEMORY_BASELINE
    if _MEMORY_BASELINE_MODULE is not mm:
        _MEMORY_BASELINE_MODULE = mm
        _MEMORY_BASELINE = {
            "EXTRA_RESERVED_VRAM": getattr(mm, "EXTRA_RESERVED_VRAM", 0),
            "NUM_STREAMS": getattr(mm, "NUM_STREAMS", 0),
            "DISABLE_SMART_MEMORY": getattr(mm, "DISABLE_SMART_MEMORY", False),
            "MAX_PINNED_MEMORY": getattr(mm, "MAX_PINNED_MEMORY", 0),
        }

    reserve = worker_args.get("reserve_vram_gb")
    if reserve is not None:
        mm.EXTRA_RESERVED_VRAM = int(float(reserve) * 1024 * 1024 * 1024)
    else:
        mm.EXTRA_RESERVED_VRAM = _MEMORY_BASELINE["EXTRA_RESERVED_VRAM"]

    if worker_args.get("safetensors_backend") == "pread":
        _enable_pread_backend()
    else:
        _disable_pread_backend()  # revert a worker that already patched pread

    # UMA: async offload streams and cast buffers are pure overhead in one
    # pool. Assign both branches so removing a worker arg restores the baseline.
    mm.NUM_STREAMS = (
        0 if worker_args.get("disable_async_offload")
        else _MEMORY_BASELINE["NUM_STREAMS"])
    mm.DISABLE_SMART_MEMORY = (
        True if worker_args.get("disable_smart_memory")
        else _MEMORY_BASELINE["DISABLE_SMART_MEMORY"])
    # comfy's pin_memory returns False the moment MAX_PINNED_MEMORY <= 0.
    mm.MAX_PINNED_MEMORY = (
        -1 if worker_args.get("disable_pinned_memory")
        else _MEMORY_BASELINE["MAX_PINNED_MEMORY"])
    # comfy's import-time "pinned N / M streams" lines are both overwritten
    # above and nothing else records the result; log the final UMA posture.
    log.info("UMA posture: streams=%s pinned=%s smart_memory=%s reserve=%.1fGiB",
             mm.NUM_STREAMS, mm.MAX_PINNED_MEMORY,
             not mm.DISABLE_SMART_MEMORY, mm.EXTRA_RESERVED_VRAM / 2**30)


def resolve_model_path(kind: str, name: str) -> str:
    """Resolve a model NAME (as shown in loader widgets) to this host's path.

    Model files are referenced by name so each host resolves against its own
    models directory; the requirement is the same relative layout on all hosts.
    """
    import folder_paths

    path = folder_paths.get_full_path(kind, name)
    if path is None:
        import socket

        raise FileNotFoundError(
            f"model {name!r} (kind {kind!r}) not found on host {socket.gethostname()}. "
            "Cluster renders need the model present on every worker host under the same name."
        )
    return path
