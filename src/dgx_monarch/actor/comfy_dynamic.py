"""Bootstrap ComfyUI DynamicVRAM inside a Monarch worker.

DynamicVRAM patches libcuda and rebinds ComfyUI's model patcher process-wide.
Those changes cannot be safely reversed in a live process, so managed and
classic residency cannot mix within one worker. Policy changes require an
Attached-mesh reset. ComfyUI and torch imports stay local so drivers can import
this module without a ComfyUI checkout.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from .. import residency_mode
from ..log import get_logger
from ..refusal import RefusalClass, refusal
from ..residency_mode import ComfyManagedResidencyError
from ..transfer_utils import failure_summary, failure_text, safe_call

log = get_logger(__name__)

# Process lifetime: latches prevent later policy applications from contradicting
# this worker's bootstrap state. Only recycling resets them.
_REQUESTED: bool | None = None
_ACTIVE: bool = False  # Process lifetime: what this worker brought up.
# Process lifetime: ``ensure_comfy`` marks bootstrap complete before stage B.
# Record its outcome so a later early return cannot turn failure into classic
# residency under a managed-residency capability context.
_STAGE_B_DONE: bool | None = None  # None until attempted; never retried.

_SUBMODULES = ("host_buffer", "vram_buffer", "model_vbar", "model_mmap")

_BRINGUP_TAIL = (
    " Nothing was loaded and nothing was quarantined. What works instead: turn "
    "the Init node's comfy_managed widget off (or remove comfy_managed from "
    "cluster.toml) and reset the attached mesh, which gives this box its "
    "ordinary stock or slab residency back."
)


def active() -> bool:
    """Whether this process brought comfy's DynamicVRAM up."""
    return residency_mode.active()


def apply_policy(worker_args: Mapping[str, Any]) -> dict[str, Any]:
    """Enforce the process policy and its owned settings.

    Every worker-argument application reaches this function, including hot
    reapplication and discrete hosts. Managed residency disables low-RSS LoRA
    swaps and slab weights because DynamicVRAM owns patching and placement.

    Leave pinned memory alone here, so the host's own posture governs it. On
    unified memory that posture turns pinning off, as the ComfyUI processes on
    this class of box run, and that makes DynamicVRAM zero-copy:
    ``pinned_hostbuf_size`` returns 0 for a disabled budget, the staging
    HostBuffer is built with no capacity, and weights page straight out of the
    checkpoint mapping. Pinning on sizes that buffer at twice the model and
    materializes a second, unswappable copy of every weight in the same physical
    pool the device computes from. ``residency_mode.PINNED_STAGING_FACTOR`` holds
    the measurement and the capacity charge that copy earns when an operator
    asks for it anyway.
    """
    values = dict(worker_args)
    want = residency_mode.requested(values)
    _latch(want)
    _assert_bring_up_not_failed()
    if not want:
        return values
    values["lora_low_rss"] = False
    values[residency_mode.WORKER_ARG] = True
    values["slab_weights"] = False
    return values


def _assert_bring_up_not_failed() -> None:
    """Refuse reuse after stage B fails.

    A live process cannot safely retry the patcher and driver hooks. Refusing
    here prevents classic residency from running under a managed capability
    context; recovery requires recycling the Attached mesh.
    """
    if _REQUESTED is not True or _STAGE_B_DONE is not False:
        return
    raise ComfyManagedResidencyError(refusal(
        RefusalClass.PHYSICS,
        "comfy-managed residency was requested and this worker process already tried "
        "to bring ComfyUI's DynamicVRAM up and failed; the earlier refusal in this "
        "worker's log says why. The process cannot try again: comfy is already "
        "imported and its patcher was never rebound, so a render now would run "
        "classic residency in a worker whose policy still claims comfy-managed "
        "residency, and an identity-gate PASS from it would bind to a capability "
        "context the worker never honored. Nothing was loaded and nothing was quarantined. "
        "What works instead: reset the attached mesh (the panel's Reset attached mesh "
        "button), restart the worker service with dgxm restart only if needed, and "
        "either fix the cause in that first refusal or "
        "turn the Init node's comfy_managed widget off.",
        troubleshooting=residency_mode.TROUBLESHOOTING))


def _latch(want: bool) -> None:
    """A comfy-managed proc cannot become a classic one, or the reverse."""
    global _REQUESTED
    if _REQUESTED is not None and _REQUESTED is not want:
        raise ComfyManagedResidencyError(refusal(
            RefusalClass.PHYSICS,
            "comfy_managed is a bootstrap policy and this worker process already "
            f"started with it {'on' if _REQUESTED else 'off'}. The policy applies only "
            "when a process first bootstraps ComfyUI, and ComfyUI's DynamicVRAM installs "
            "inline instruction patches into the CUDA driver library and rebinds comfy's "
            "model patcher process-wide, which a live process cannot undo. Nothing was "
            "loaded and nothing was quarantined. What works instead: reset the attached "
            "mesh (the panel's Reset attached mesh button), restart the worker service "
            "with dgxm restart only if needed, and queue the render again.",
            troubleshooting=residency_mode.TROUBLESHOOTING))
    _REQUESTED = want


def stage_a(worker_args: Mapping[str, Any] | None) -> bool:
    """Load aimdo.so into the global namespace, before comfy is imported.

    ``control.init()`` loads the library and ctypes signatures without touching
    CUDA. It need not precede torch; the order that matters is a live CUDA
    context before ``stage_b`` calls ``init_devices``. Failures here must occur
    before ComfyUI bootstrap so managed residency cannot silently fall back.
    """
    # Clear inherited state so only this process's successful bootstrap can
    # claim managed residency.
    os.environ.pop(residency_mode.ENV_ACTIVE, None)
    want = residency_mode.requested(worker_args)
    _latch(want)
    if not want:
        return False
    try:
        import comfy_aimdo.control as ctl
    except ImportError as exc:
        raise ComfyManagedResidencyError(refusal(
            RefusalClass.PHYSICS,
            "comfy-managed residency was requested but comfy-aimdo is not installed "
            f"in this worker's Python environment ({failure_text(exc)})." + _BRINGUP_TAIL,
            troubleshooting=residency_mode.TROUBLESHOOTING)) from exc
    # Protocol compatibility: 0.4.13 accepts ``nvml_pressure``, 0.4.10 accepts
    # ``simple_vram_headroom``, and 0.4.9 accepts no keywords. Stage B reapplies
    # the resolved headroom through the idempotent initializer.
    try:
        loaded = ctl.init(nvml_pressure=True)
    except TypeError:
        try:
            loaded = ctl.init(simple_vram_headroom=None)
        except TypeError:
            loaded = ctl.init()
    if not loaded:
        raise ComfyManagedResidencyError(refusal(
            RefusalClass.PHYSICS,
            "comfy-managed residency was requested but comfy_aimdo.control.init() "
            "could not load aimdo.so on this host." + _BRINGUP_TAIL,
            troubleshooting=residency_mode.TROUBLESHOOTING))
    _rewire_submodule_lib(ctl)
    return True


def _rewire_submodule_lib(ctl: Any) -> None:
    """Re-point comfy_aimdo's submodules at the initialized library handle.

    Submodules capture ``control.lib`` during import, often before library
    initialization, and otherwise retain ``None``. Rewire each captured handle;
    comfy-aimdo 0.4.13 still requires this.
    """
    handle = getattr(ctl, "lib", None)
    if handle is None:
        log.warning("comfy_aimdo.control has no initialized library handle to re-wire")
        return
    sub: Any
    for name in _SUBMODULES:
        try:
            sub = __import__(f"comfy_aimdo.{name}", fromlist=[name])
            if getattr(sub, "lib", None) is None:
                sub.lib = handle
        except Exception as exc:  # best effort, per submodule
            safe_call(
                log.warning,
                "comfy_aimdo.%s could not be re-wired (%s)",
                name,
                failure_summary(exc),
            )


def stage_b(enabled: bool, *, reserve_vram_gb: float | None = None) -> bool:
    """Install the hooks, claim the device, and rebind comfy's patcher.

    This stage requires libcuda and a live CUDA context, so it runs after
    ComfyUI bootstrap.
    """
    global _ACTIVE, _STAGE_B_DONE
    if not enabled:
        os.environ.pop(residency_mode.ENV_ACTIVE, None)
        return False

    # Arm before any fallible work so every failure remains visible on re-entry.
    _STAGE_B_DONE = False

    import comfy.memory_management as cmm
    import comfy.model_management as mm
    import comfy.model_patcher
    import comfy_aimdo.control as ctl
    import torch

    # Workers reset argv before ComfyUI parses it, so its command-line residency
    # flags remain at defaults. Any future worker-side posture must extend this
    # gate rather than combine with DynamicVRAM implicitly.
    if not mm.is_nvidia() or mm.is_wsl() or mm.torch_version_numeric < (2, 8):
        raise ComfyManagedResidencyError(refusal(
            RefusalClass.PHYSICS,
            "comfy-managed residency was requested but this host does not meet "
            "ComfyUI's own DynamicVRAM requirements (NVIDIA, not WSL, torch 2.8 or "
            "newer)." + _BRINGUP_TAIL,
            troubleshooting=residency_mode.TROUBLESHOOTING))

    # Apply the same reserve to ComfyUI and aimdo. The per-device headroom below
    # is a separate setting and stays zero to avoid charging the reserve twice.
    headroom = None if not reserve_vram_gb else int(float(reserve_vram_gb) * 1024 ** 3)
    # Keep aimdo's native log default. Its bridge emits through Python logging,
    # where the worker level already filters output.
    try:
        ctl.init(simple_vram_headroom=headroom, nvml_pressure=True)
    except TypeError:
        try:
            ctl.init(simple_vram_headroom=headroom)
        except TypeError:
            ctl.init()

    # ``init_devices`` resolves libcuda symbols against a live CUDA context.
    # CUDA_VISIBLE_DEVICES already narrows this process to one worker GPU.
    torch.cuda.init()
    torch.cuda.set_device(mm.get_torch_device())
    index = torch.cuda.current_device()
    try:
        ok = ctl.init_devices([(index, 0)])
    except TypeError:  # 0.4.9 protocol
        ok = ctl.init_devices([index])
    if not ok:
        raise ComfyManagedResidencyError(refusal(
            RefusalClass.PHYSICS,
            "comfy-managed residency was requested but "
            "comfy_aimdo.control.init_devices() reported no working install for "
            f"device {index}." + _BRINGUP_TAIL,
            troubleshooting=residency_mode.TROUBLESHOOTING))

    # Publish both ComfyUI globals only after device initialization succeeds.
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    cmm.aimdo_enabled = True
    _ACTIVE = True
    _STAGE_B_DONE = True
    os.environ[residency_mode.ENV_ACTIVE] = "1"
    log.info("comfy-managed residency ACTIVE: comfy-aimdo %s, device %d, "
             "simple_vram_headroom %s, patcher ModelPatcherDynamic",
             _aimdo_version(), index,
             "unset" if headroom is None else f"{headroom / 2 ** 30:.1f} GiB")
    return True


def pinned_staging_active() -> bool:
    """Whether DynamicVRAM will stage weights through a pinned host buffer.

    ComfyUI's own budget is the authority: ``pinned_hostbuf_size`` returns
    ``min(model, MAX_PINNED_MEMORY) * 2``, and zero once the budget is not
    positive, so this reads the same number the staging decision reads rather
    than re-deriving it from worker arguments. Only the sign counts: a positive
    budget smaller than the model still stages, and the capacity wall prices
    the worst case. An unimportable or unreadable ComfyUI answers False, which
    prices the load as zero-copy; the capacity wall's floor still applies, and
    a wrong guess here can only make the wall more permissive on a host that
    already refused to be measured.
    """
    try:
        import comfy.model_management as mm
    except Exception:
        return False
    budget = getattr(mm, "MAX_PINNED_MEMORY", 0)
    try:
        return float(budget) > 0
    except (TypeError, ValueError):
        return False


def assert_rdma_compatible() -> None:
    """Refuse native RDMA latent return in a ComfyUI-managed worker.

    DynamicVRAM and native latent return both register host memory. Their
    combined operation is unsupported, so the constructor refuses it even
    though native latent return defaults off.
    """
    if not residency_mode.active():
        return
    raise ComfyManagedResidencyError(refusal(
        RefusalClass.PHYSICS,
        "comfy-managed residency and the RDMA latent return cannot run in the same "
        "worker. ComfyUI's DynamicVRAM registers host memory with the CUDA driver "
        "through inline patches into libcuda, and the RDMA return registers its own "
        "host buffers for the multi-NIC scatter; that collision has never been "
        "measured. Nothing was loaded and nothing was quarantined. What works "
        "instead: leave rdma_latent_return off, its default since 2026-07-29, or "
        "turn the Init node's comfy_managed widget off and reset the attached mesh.",
        troubleshooting=residency_mode.TROUBLESHOOTING))


def _aimdo_version() -> str:
    try:
        from importlib.metadata import version

        return str(version("comfy-aimdo"))
    except Exception:  # Status must remain available if metadata lookup fails.
        return "unknown"


def snapshot() -> dict[str, Any]:
    """Worker status fields for this rung."""
    return {
        "comfy_managed": bool(_ACTIVE),
        "comfy_aimdo": _aimdo_version() if _ACTIVE else None,
    }
