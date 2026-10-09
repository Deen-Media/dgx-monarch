"""Lazy safetensors pread-backend shim for Comfy worker processes."""
from __future__ import annotations

from typing import Any, cast

from ..log import get_logger

# The logger keeps comfy_bridge's name so a worker journal shows one load-path
# category across the bootstrap and the pread shim.
log = get_logger("dgx_monarch.actor.comfy_bridge")


def _enable_pread_backend() -> None:
    """Default safetensors reads to the pread backend (positional pread(2), no
    mmap) process-wide. comfy's load_torch_file resolves ``safetensors.safe_open``
    at call time, so defaulting the package attribute intercepts every DiT and
    LoRA read with no comfy edit, except under comfy-managed residency: with
    ``comfy.memory_management.aimdo_enabled`` set, load_torch_file reads through
    comfy-aimdo's file mapping (``comfy.utils.load_safetensors``) and never
    calls ``safe_open``. Idempotent; needs safetensors >= 0.8.0 (the ``backend``
    param); older versions stay on mmap."""
    import safetensors

    if getattr(safetensors.safe_open, "_dgxm_pread", False):
        return
    # Feature-detect, never version-parse: 0.8.0rc0 satisfies >=0.8.0-rc.0
    # (how diffusers' specifier pulls it in) and splits to (0, 8), yet lacks
    # the backend param, so a version gate passes and the first load dies with
    # TypeError. The class signature is authoritative.
    import inspect
    import re

    try:
        params = inspect.signature(safetensors.safe_open).parameters
        has_backend = "backend" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        # introspection unavailable: trust only clean X.Y.Z final releases
        ver = tuple(int(p) for p in safetensors.__version__.split(".")[:2] if p.isdigit())
        has_backend = (bool(re.fullmatch(r"[0-9]+(\.[0-9]+)*", safetensors.__version__))
                       and ver >= (0, 8))
    if not has_backend:
        log.warning("safetensors %s has no pread backend (need final >= 0.8.0; "
                    "pre-releases like 0.8.0rc0 lack it); staying on mmap",
                    safetensors.__version__)
        return
    _orig = safetensors.safe_open

    def _pread_safe_open(*args, **kwargs):
        kwargs.setdefault("backend", "pread")  # comfy never sets it
        return _orig(*args, **kwargs)

    shim = cast(Any, _pread_safe_open)
    shim._dgxm_pread = True
    shim._dgxm_orig = _orig
    cast(Any, safetensors).safe_open = shim
    log.info("safetensors load backend: pread (no mmap)")


def _disable_pread_backend() -> None:
    """Revert to the stock (mmap) safetensors loader for the mmap_fallback A/B
    switch. Takes effect on the next model load (a resident model is not
    reloaded); idempotent."""
    import safetensors

    shim = safetensors.safe_open
    if getattr(shim, "_dgxm_pread", False):
        cast(Any, safetensors).safe_open = cast(Any, shim)._dgxm_orig
        log.info("safetensors load backend: mmap (fallback)")


def mmap_load_window():
    """Use stock mmap reads for one FSDP model load, restoring the backend on exit.

    pread materializes a checkpoint-sized anonymous state dict alongside the
    model copy. mmap keeps the state dict file-backed and reclaimable, avoiding
    that second anonymous copy in unified memory. Non-FSDP loads retain pread,
    whose page-cache behavior was better in those measurements.
    """
    import contextlib

    @contextlib.contextmanager
    def _window():
        import safetensors

        shim = safetensors.safe_open
        if not getattr(shim, "_dgxm_pread", False):
            yield
            return
        cast(Any, safetensors).safe_open = cast(Any, shim)._dgxm_orig
        log.info("FSDP load: safetensors mmap window "
                 "(the state dict stays file-backed)")
        try:
            yield
        finally:
            cast(Any, safetensors).safe_open = shim

    return _window()
