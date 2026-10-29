"""The ``int8_gemm`` option: which kernel runs comfy-kitchen's INT8 ConvRot GEMM.

comfy-kitchen sends every INT8 linear on sm_80 and newer through one CUTLASS
call. ``int8_gemm_triton`` (vendored, MIT) runs that call as a Triton kernel
tuned for GB10 and leaves on CUTLASS the fused-residual calls, rows under 128
and calls with K not a multiple of 16 or a strided tensor. docs/VALIDATION.md,
"INT8 GEMM kernel comparison", records the two kernels giving identical output,
so the Triton kernel is on by default.

Who reads what:

- An actor reads ``[worker_args] int8_gemm`` ("triton" or "cutlass").
  ``mesh_runtime.bootstrap_proc`` registers the import hook in every actor
  process, and setup sets the mode before ``ensure_comfy`` imports ComfyUI,
  which imports comfy-kitchen. A later apply_worker_args switches the live
  process in place: the two kernels give the same bits, so no resident model
  has to reload.
- The driver, the ComfyUI process, reads ``DGXM_INT8_GEMM`` ("triton" or
  "cutlass") once, at node import. ComfyUI has imported comfy-kitchen by then,
  so the backend is wrapped in place; the import hook covers a process that
  imports it later.
- Both default to "triton", and neither reads the other's setting.

Rails, none of which raises into a render: a comfy-kitchen release outside
``VALIDATED_KITCHEN``, no Python.h for this interpreter, Triton failing to
load, or a value that names neither mode keeps stock CUTLASS for the process
and logs one WARNING naming the reason. A kernel that fails to compile or run
at a call hands that call and every later one to CUTLASS, with one WARNING.
The mode in effect logs one INFO line naming the comfy-kitchen version and the
mode, so the driver log and the Worker journal say which kernel runs. Every
line starts with ``INT8 GEMM:``.
"""
from __future__ import annotations

import os
import sys
import sysconfig
import threading
from collections.abc import Mapping
from types import ModuleType
from typing import Any, cast

from ..log import get_logger
from . import int8_gemm_triton as vendored

ENV = "DGXM_INT8_GEMM"
WORKER_ARG = "int8_gemm"
TRITON = "triton"
CUTLASS = "cutlass"
MODES = (TRITON, CUTLASS)
DEFAULT = TRITON
VALIDATED_KITCHEN = vendored.VALIDATED_KITCHEN
PREFIX = "INT8 GEMM:"
DEFAULT_SOURCE = "the default"
WORKER_ARG_SOURCE = f"[worker_args] {WORKER_ARG}"

log = get_logger(__name__)
_LOCK = threading.RLock()


class _State:
    """This process's requested mode, where it came from, and what was logged."""

    def __init__(self) -> None:
        self.mode = DEFAULT
        self.source = DEFAULT_SOURCE
        # The requested value named neither mode; its WARNING stands for the
        # INFO line, and the process runs stock CUTLASS.
        self.invalid = False
        # Why the kernel cannot run in this process, once known. Never cleared:
        # the comfy-kitchen release, the interpreter and a failed build do not
        # change under a live process.
        self.blocked: str | None = None
        self.finder: vendored.Finder | None = None
        # The mode the last INFO line named, and every WARNING already logged.
        self.announced: str | None = None
        self.warned: set[str] = set()


_STATE = _State()


def _warn(reason: str, outcome: str = "stock CUTLASS for this process") -> None:
    """Log one WARNING line per distinct reason and process."""
    if reason not in _STATE.warned:
        _STATE.warned.add(reason)
        log.warning("%s %s; %s", PREFIX, reason, outcome)


def _announce(mode: str) -> None:
    """Log the INFO line for the mode in effect, once per change of mode."""
    if _STATE.invalid or _STATE.announced == mode:
        return
    _STATE.announced = mode
    version = vendored.kitchen_version() or "unknown"
    if mode == TRITON:
        log.info("%s Triton kernel on for comfy-kitchen %s INT8 linears (int8_gemm=triton, from %s)",
                 PREFIX, version, _STATE.source)
    else:
        log.info("%s stock comfy-kitchen %s CUTLASS kernel (int8_gemm=cutlass, from %s)",
                 PREFIX, version, _STATE.source)


def parse(value: object) -> str | None:
    """The mode a setting names, or None when it names neither. None means the default."""
    if value is None:
        return DEFAULT
    if isinstance(value, str) and value.strip().lower() in MODES:
        return value.strip().lower()
    return None


def python_headers() -> tuple[bool, str]:
    """Whether Triton's launcher build can include Python.h, and the directory it names.

    Triton compiles a small C launcher for each kernel signature against this
    interpreter's headers. It passes the include directory of the default
    sysconfig scheme (posix_prefix in place of Debian's posix_local), and the
    C compiler also searches CPATH and C_INCLUDE_PATH. Worker services filter
    both variables out of the actor environment, so on a Worker only the
    interpreter's own include directory counts.
    """
    scheme = sysconfig.get_default_scheme()
    if scheme == "posix_local":
        scheme = "posix_prefix"
    include = sysconfig.get_paths(scheme=scheme)["include"]
    extra = [entry for name in ("CPATH", "C_INCLUDE_PATH")
             for entry in os.environ.get(name, "").split(os.pathsep) if entry]
    return any(os.path.isfile(os.path.join(path, "Python.h")) for path in (include, *extra)), include


def _blocked() -> str | None:
    """Why the Triton kernel cannot run in this process, or None when it can."""
    if _STATE.blocked is None:
        version = vendored.kitchen_version()
        if version not in VALIDATED_KITCHEN:
            _STATE.blocked = (f"comfy-kitchen {version or 'without version metadata'} is not in the "
                              f"validated set ({', '.join(VALIDATED_KITCHEN)})")
        else:
            try:
                found, include = python_headers()
            except Exception as exc:
                found, include = False, f"an unreadable include directory: {exc!r:.200}"
            if not found:
                _STATE.blocked = (f"Python.h is missing for this interpreter (Triton builds its "
                                  f"launcher against {include})")
    return _STATE.blocked


def _backend_proxy() -> vendored.CProxy | None:
    current = getattr(sys.modules.get(vendored.TARGET), "_C", None)
    return current if isinstance(current, vendored.CProxy) else None


def _kernel_on() -> bool:
    proxy = _backend_proxy()
    return proxy is not None and not getattr(proxy, "_broken", False)


def active_mode() -> str:
    """The kernel comfy-kitchen's INT8 GEMM uses in this process now."""
    try:
        return TRITON if _kernel_on() else CUTLASS
    except Exception:
        return CUTLASS


def _on_kernel_failure(exc: BaseException) -> None:
    """The proxy's callback when the kernel fails at a call: CUTLASS from then on."""
    with _LOCK:
        reason = f"the Triton kernel failed to compile or run ({exc!r:.200})"
        _STATE.blocked = reason
        backend = sys.modules.get(vendored.TARGET)
        proxy = _backend_proxy()
        if backend is not None and proxy is not None:
            cast(Any, backend)._C = proxy._real
        _warn(reason, "stock CUTLASS for the rest of this process")


def _apply_backend(backend: ModuleType) -> None:
    """Wrap or unwrap comfy-kitchen's CUDA extension to match the mode."""
    current = getattr(backend, "_C", None)
    real = current._real if isinstance(current, vendored.CProxy) else current
    if _STATE.mode != TRITON:
        if current is not real:
            cast(Any, backend)._C = real
        _announce(CUTLASS)
        return
    if isinstance(current, vendored.CProxy):
        return
    if real is None or not hasattr(real, "cutlass_int8_dequant"):
        if _STATE.announced != "absent":
            _STATE.announced = "absent"
            log.info("%s comfy-kitchen %s has no CUDA extension in this process, so there is no "
                     "CUTLASS INT8 GEMM to replace", PREFIX, vendored.kitchen_version() or "unknown")
        return
    reason = _blocked()
    if reason is not None:
        _warn(reason)
        return
    try:
        gemm = vendored.build_triton_gemm()
    except Exception as exc:
        _STATE.blocked = reason = f"Triton could not be loaded ({exc!r:.200})"
        _warn(reason)
        return
    cast(Any, backend)._C = vendored.CProxy(real, gemm, _on_kernel_failure)
    _announce(TRITON)


def _apply_top(top: ModuleType) -> None:
    """Keep comfy_kitchen.int8_linear's compile-safe wrapper only while the kernel is on."""
    current = getattr(top, "int8_linear", None)
    if current is None:
        return
    wrapped = bool(getattr(current, "_ck_compile_safe", False))
    if _kernel_on():
        if not wrapped:
            try:
                vendored.compile_safe_int8_linear(top)
            except Exception as exc:
                _warn(f"the compile-safe int8_linear wrapper was not applied ({exc!r:.200})",
                      "the Triton kernel stays on")
    elif wrapped and getattr(current, "__wrapped__", None) is not None:
        cast(Any, top).int8_linear = current.__wrapped__


def _fail_safe(exc: BaseException) -> None:
    """Best effort back to stock after an unexpected error here, with one WARNING."""
    try:
        backend = sys.modules.get(vendored.TARGET)
        proxy = _backend_proxy()
        if backend is not None and proxy is not None:
            cast(Any, backend)._C = proxy._real
        _warn(f"the option could not be applied ({exc!r:.200})")
    except Exception:
        pass


def _on_exec(name: str, module: ModuleType) -> None:
    """The import hook's callback, right after comfy-kitchen or its CUDA backend executes."""
    try:
        with _LOCK:
            if name == vendored.TARGET:
                _apply_backend(module)
            elif name == vendored.TOP:
                _apply_top(module)
    except Exception as exc:
        _fail_safe(exc)


def _ensure_finder() -> None:
    if _STATE.finder is None:
        _STATE.finder = vendored.Finder(_on_exec)
    if _STATE.finder not in sys.meta_path:
        sys.meta_path.insert(0, _STATE.finder)


def configure(value: object, source: str = DEFAULT_SOURCE) -> str:
    """Set this process's mode and apply it wherever comfy-kitchen is already loaded.

    ``value`` is the setting as read, None for unset. The import hook applies
    the mode to a comfy-kitchen imported later. Returns the kernel in effect
    now ("triton" or "cutlass"). Never raises.
    """
    try:
        with _LOCK:
            mode = parse(value)
            _STATE.invalid = mode is None
            if mode is None:
                _warn(f"{source} is {value!r}, which names neither \"triton\" nor \"cutlass\"")
                mode = CUTLASS
            _STATE.mode, _STATE.source = mode, source
            _ensure_finder()
            backend = sys.modules.get(vendored.TARGET)
            if backend is not None:
                _apply_backend(backend)
            top = sys.modules.get(vendored.TOP)
            if top is not None:
                _apply_top(top)
    except Exception as exc:
        _fail_safe(exc)
    return active_mode()


def install_actor() -> None:
    """Register the import hook in a fresh actor process; setup sets the mode later.

    Runs from ``mesh_runtime.bootstrap_proc``. It imports only the standard
    library and never raises.
    """
    try:
        with _LOCK:
            _ensure_finder()
    except Exception as exc:
        _fail_safe(exc)


def configure_from_worker_args(worker_args: Mapping[str, object] | None) -> str:
    """An actor's mode from ``[worker_args] int8_gemm``; an absent key means the default."""
    try:
        if worker_args and WORKER_ARG in worker_args:
            return configure(worker_args[WORKER_ARG], WORKER_ARG_SOURCE)
    except Exception as exc:
        _fail_safe(exc)
        return active_mode()
    return configure(None, DEFAULT_SOURCE)


def install_driver(environ: Mapping[str, str] | None = None) -> str:
    """The ComfyUI process's mode from ``DGXM_INT8_GEMM``, applied at node import."""
    try:
        raw = (os.environ if environ is None else environ).get(ENV)
    except Exception:
        raw = None
    if raw is None or not raw.strip():
        return configure(None, DEFAULT_SOURCE)
    return configure(raw, ENV)
