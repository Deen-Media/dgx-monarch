"""Backend selection for the installed Sol-Attn package.

Two jobs, both about what runs rather than what was requested: import the
optional package or refuse naming the exact install, and put GB10 on the fast
dispatch path upstream never selects for it.
"""
from __future__ import annotations

import threading
from typing import Final

import torch

from ..log import get_logger
from . import sol_attention_guards as guards

log = get_logger(__name__)

_CUTE_LOCK = threading.Lock()
_CUTE_STATE: dict[str, str] = {}

# GB10 reports compute capability (12, 1). Upstream's CuTe table matches
# capability exactly and carries no (12, 1) key, so a Spark always lands on the
# portable Triton reference.
GB10_CAPABILITY: Final = (12, 1)
SM120_CUTE_BACKEND: Final = "cute_sm120"

# What to install to get off the portable path, named where an operator reading
# a triton backend line needs it. The DSL is the CuTe compiler. Without it the
# package falls back to Triton and says nothing, so the install line is named
# here rather than in the absence refusal, which names what the kernel needs to
# run at all. apache-tvm-ffi is already required there and appears here only
# because this check is written once for both.
CUTE_PATH_INSTALL: Final = {
    "cutlass": "pip install nvidia-cutlass-dsl==4.7.0",
    "tvm_ffi": "pip install apache-tvm-ffi",
}


def enable_cute_on_gb10() -> str:
    """Add GB10 to CuTe dispatch only when all prerequisites are available.

    NEVER SPOOF THE REPORTED CAPABILITY: (12, 0) targets sm_120a,
    whose cubins cannot execute on sm_121. Adding the native key lets the
    Cutlass DSL compile for sm_121a.

    ``_CUTE_BACKENDS`` is private upstream state. Guard every access, retain the
    package's default selection on failure, and log the outcome so a Triton run
    cannot be mistaken for CuTe performance evidence.
    """
    import importlib.util

    with _CUTE_LOCK:
        if "reason" in _CUTE_STATE:
            return _CUTE_STATE["reason"]
        reason = _apply_cute_dispatch(importlib.util.find_spec)
        _CUTE_STATE["reason"] = reason
        log.info("sol-attn CuTe dispatch on this device: %s", reason)
        return reason


def _apply_cute_dispatch(find_spec) -> str:
    """One attempt at the dispatch entry; returns what happened, never raises."""
    try:
        capability = torch.cuda.get_device_capability()
    except Exception as exc:  # A device that cannot be queried takes triton.
        return f"not applied: device capability unreadable ({type(exc).__name__})"
    if tuple(capability) != GB10_CAPABILITY:
        return (f"not applied: device capability {tuple(capability)} is not "
                f"{GB10_CAPABILITY}, so upstream dispatch already decides")
    for module in ("cutlass", "tvm_ffi"):
        try:
            present = find_spec(module) is not None
        except Exception:
            present = False
        if not present:
            return (f"not applied: {module} is not importable, so the CuTe "
                    "path cannot run and upstream's Triton selection stands. "
                    f"Install it with `{CUTE_PATH_INSTALL[module]}`")
    try:
        import sol_attn.interface as interface

        table = interface._CUTE_BACKENDS
        if not isinstance(table, dict):
            return "not applied: upstream dispatch table is not a dict"
        table.setdefault(GB10_CAPABILITY, SM120_CUTE_BACKEND)
    except Exception as exc:
        return f"not applied: {type(exc).__name__} reading upstream dispatch"
    return f"applied: {GB10_CAPABILITY} -> {SM120_CUTE_BACKEND}"


def load_sol_attn():
    """Import the installed package, or refuse naming the exact install."""
    import importlib.util

    for module, label in (("sol_attn", "the sol-attn package"),
                          ("tvm_ffi", "apache-tvm-ffi")):
        try:
            present = importlib.util.find_spec(module) is not None
        except Exception:
            present = False
        if not present:
            guards.refuse_missing_package(label)
    try:
        from sol_attn import get_sol_attn_backend, sol_attn
    except ImportError:
        guards.refuse_missing_package("the sol-attn package")
    return sol_attn, get_sol_attn_backend


def sol_attn_available() -> bool:
    """Whether this box could run the kernel at all. Never raises."""
    import importlib.util

    try:
        return all(importlib.util.find_spec(m) is not None
                   for m in ("sol_attn", "tvm_ffi"))
    except Exception:
        return False


