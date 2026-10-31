"""The two FP8 compute paths, and the actors' ``[worker_args] fp8_matrix_mult``.

ComfyUI picks a model's operations class once, when it builds the model
(``comfy.ops.pick_operations``):

- A checkpoint that carries comfy-kitchen quantization metadata takes
  ``mixed_precision_ops``, and comfy-kitchen's own GEMMs run its quantized
  linears whatever any flag says; a layer whose record sets
  ``full_precision_matrix_mult`` dequantizes instead.
- Any other checkpoint takes ``fp8_ops`` when the device has FP8 compute and
  either the loader passed ``fp8_optimizations`` (the ``fp8_e4m3fn_fast``
  weight dtype) or the process runs with ``--fast fp8_matrix_mult``; the two
  meet in one ``or``. ``fp8_ops`` runs a linear whose weight is stored as
  ``float8_e4m3fn`` through ``torch._scaled_mm``, after casting its input to
  FP8 with a unit scale; a weight in any other dtype keeps the cast path.
- Otherwise a model stored as FP8 takes ``manual_cast``, which dequantizes
  every FP8 weight to the compute dtype for every matmul, and a full-precision
  model takes ``disable_weight_init``.

ComfyUI's flags never reach the actors: an actor imports ComfyUI with an empty
argument list, so ``--fast fp8_matrix_mult`` on the driver changes only the
driver's own loads. ``[worker_args] fp8_matrix_mult`` is the actors' switch,
applied to every DiT build through the loader option that ComfyUI reads beside
the flag:

- ``auto`` (the default): pass what the loader's weight_dtype asks for, as
  stock ComfyUI does without the flag.
- ``on``: pass ``fp8_optimizations`` on every build, which is what the flag
  does to the driver's loads.
- ``off``: never pass it, so a loader's ``fp8_e4m3fn_fast`` builds as
  ``fp8_e4m3fn`` and FP8 weights take the dequantizing path.

The class binds when the model is built, so a changed setting drops the
actor's residents (``actor/worker_env.apply_worker_args_impl``). Every build
in the driver and in each actor logs one INFO line starting ``FP8 compute:``
that names the class ComfyUI picked, whether ``fp8_optimizations`` was set and
whether the flag was, so the driver log and the Worker journals tell the paths
apart. docs/MODELS.md, "FP8 compute paths", is the operator's page.

Standard library only. It reads ComfyUI's modules where the process already
imported them and never imports them itself, and nothing here raises into a
load.
"""
from __future__ import annotations

import functools
import inspect
import sys
import threading
from collections.abc import Callable, Mapping
from typing import Any, cast

from .log import get_logger

WORKER_ARG = "fp8_matrix_mult"
AUTO = "auto"
ON = "on"
OFF = "off"
MODES = (AUTO, ON, OFF)
DEFAULT = AUTO
PREFIX = "FP8 compute:"
# The comfy model option the loader's fp8_e4m3fn_fast sets (comfy.sd reads it
# into model_config.optimizations["fp8"], and model_base hands that to
# pick_operations).
OPTION = "fp8_optimizations"
# The PerformanceFeature value of ComfyUI's --fast fp8_matrix_mult.
FLAG = "fp8_matrix_mult"
DEFAULT_SOURCE = "the default"
WORKER_ARG_SOURCE = f"[worker_args] {WORKER_ARG}"
# Marks the wrapper so a second install leaves it alone.
_MARK = "_dgxm_fp8_logged"

# The operations classes comfy.ops defines, by the qualname prefix its class
# objects carry; mixed_precision_ops builds a class per call inside a function.
_OPERATIONS = ("mixed_precision_ops", "fp8_ops", "cublas_ops", "manual_cast", "disable_weight_init")
# What each class does to the matmuls of the model it builds.
_MATMULS = {
    "mixed_precision_ops": "comfy-kitchen's kernels run the quantized linears whatever the flags say",
    "fp8_ops": "linears stored as float8_e4m3fn run torch._scaled_mm on FP8 tensor cores",
    "cublas_ops": "FP16 linears run the cublas_ops kernels",
    "manual_cast": "weights are cast to the compute dtype for every matmul",
    "disable_weight_init": "weights compute in their stored dtype",
}

log = get_logger(__name__)
_LOCK = threading.Lock()


class _State:
    """This process's setting, where it came from, and its role in the log line."""

    def __init__(self) -> None:
        self.mode = DEFAULT
        self.source = DEFAULT_SOURCE
        # "the driver" or "this actor": the first install sets it, and an
        # actor's configure sets it again.
        self.role = ""


_STATE = _State()


def parse(value: object) -> str | None:
    """The mode a setting names, None for a value that names none. Unset is the default."""
    if value is None:
        return DEFAULT
    if isinstance(value, str) and value.strip().lower() in MODES:
        return value.strip().lower()
    return None


def setting(worker_args: Mapping[str, object] | None) -> tuple[str, str]:
    """(mode, source) as ``[worker_args]`` states it; an absent key is the default.

    A value that names no mode reads as the default: config_schema refuses it
    at the driver and again at the actor's trust boundary, so none arrives.
    """
    if worker_args and WORKER_ARG in worker_args:
        mode = parse(worker_args[WORKER_ARG])
        if mode is not None:
            return mode, WORKER_ARG_SOURCE
    return DEFAULT, DEFAULT_SOURCE


def current_mode() -> str:
    return _STATE.mode


def configure_from_worker_args(worker_args: Mapping[str, object] | None) -> bool:
    """Set this actor's mode from validated worker args; True when the mode changed."""
    with _LOCK:
        mode, source = setting(worker_args)
        changed = mode != _STATE.mode
        _STATE.mode, _STATE.source = mode, source
        # Only actors call this, so it names the role even where the pack's
        # entrypoint ran first and claimed the driver's.
        _STATE.role = "this actor"
    if changed:
        log.info("%s this actor builds with %s=%s from %s; the next build of each model "
                 "takes it", PREFIX, WORKER_ARG, mode, source)
    return changed


def load_options(model_options: Mapping[str, Any]) -> dict[str, Any]:
    """The comfy model options an actor's DiT build passes under this process's mode.

    ``on`` adds ``fp8_optimizations``, ``off`` removes it, ``auto`` keeps the
    loader's own choice. A checkpoint with comfy-kitchen metadata takes
    ``mixed_precision_ops`` either way, so only plain-weight files move.
    """
    out = dict(model_options)
    if _STATE.mode == ON:
        out[OPTION] = True
    elif _STATE.mode == OFF:
        out.pop(OPTION, None)
    return out


def operations_name(operations: object) -> str:
    """The comfy.ops class name a pick returned, or its qualified name when unknown.

    ``mixed_precision_ops`` returns a class it defines inside itself, so the
    name is the first qualname part comfy.ops defines.
    """
    qualname = str(getattr(operations, "__qualname__", "") or type(operations).__name__)
    return next((part for part in qualname.split(".") if part in _OPERATIONS), qualname)


def _dtype_name(dtype: object) -> str:
    return "unset" if dtype is None else str(dtype).removeprefix("torch.")


def _flag_on() -> bool | None:
    """Whether this process runs with --fast fp8_matrix_mult; None when unreadable."""
    module = sys.modules.get("comfy.cli_args")
    fast = getattr(getattr(module, "args", None), "fast", None)
    if fast is None:
        return None
    return any(getattr(feature, "value", feature) == FLAG for feature in fast)


def _fp8_compute(load_device: object) -> bool | None:
    """comfy's own answer to whether this device computes in FP8; None when unreadable."""
    module = sys.modules.get("comfy.model_management")
    probe = getattr(module, "supports_fp8_compute", None)
    if probe is None:
        return None
    try:
        return bool(probe(load_device))
    except Exception:
        return None


def describe(bound: Mapping[str, Any], operations: object) -> str:
    """The INFO line for one pick, from the call's bound arguments and its result."""
    name = operations_name(operations)
    quantized = bool(getattr(bound.get("model_config"), "quant_config", None))
    flag, fp8 = _flag_on(), _fp8_compute(bound.get("load_device"))
    weights = ("comfy-kitchen quantized weights" if quantized
               else f"weights stored as {_dtype_name(bound.get('weight_dtype'))}")
    facts = [f"fp8_optimizations={bool(bound.get('fp8_optimizations'))}",
             "--fast fp8_matrix_mult " + {True: "on", False: "off", None: "unread"}[flag],
             "FP8 compute " + {True: "supported", False: "unsupported", None: "unread"}[fp8]]
    if bound.get("disable_fast_fp8"):
        facts.append("this load opts out of fp8_ops")
    if _STATE.role == "this actor":
        facts.append(f"{WORKER_ARG}={_STATE.mode} from {_STATE.source}")
    effect = _MATMULS.get(name, "an operations class this release does not name")
    if name == "fp8_ops" and "float8_e4m3fn" not in str(bound.get("weight_dtype")):
        effect = "fp8_ops runs torch._scaled_mm only for float8_e4m3fn weights, so these keep the cast path"
    elif name == "manual_cast" and "float8" in str(bound.get("weight_dtype")):
        effect = "FP8 weights are dequantized to the compute dtype for every matmul"
    return (f"{PREFIX} {_STATE.role or 'this process'} picked {name} for {weights}, compute "
            f"dtype {_dtype_name(bound.get('compute_dtype'))} ({', '.join(facts)}); {effect}")


def _wrap(original: Callable[..., Any]) -> Callable[..., Any]:
    """Log one line per pick and return comfy's own result unchanged."""
    try:
        signature: inspect.Signature | None = inspect.signature(original)
    except (TypeError, ValueError):
        signature = None

    @functools.wraps(original)
    def pick_operations(*args: Any, **kwargs: Any) -> Any:
        operations = original(*args, **kwargs)
        try:
            if signature is not None:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                log.info("%s", describe(bound.arguments, operations))
        except Exception:
            pass
        return operations

    setattr(pick_operations, _MARK, True)
    return pick_operations


def install(role: str) -> bool:
    """Wrap ``comfy.ops.pick_operations`` once in this process; True when the wrapper is in place.

    ``role`` names the process in the line ("the driver" or "this actor") when
    nothing named it yet. It wraps only a ``comfy.ops`` this process already
    imported, and imports nothing itself: importing ComfyUI early would select
    its device before the caller does. An actor installs before each build,
    after ``comfy.sd`` has imported ``comfy.ops``. Never raises.
    """
    try:
        with _LOCK:
            if not _STATE.role:
                _STATE.role = role
            ops = sys.modules.get("comfy.ops")
            current = getattr(ops, "pick_operations", None)
            if current is None:
                return False
            if not getattr(current, _MARK, False):
                cast(Any, ops).pick_operations = _wrap(current)
            return True
    except Exception:
        return False


def install_driver() -> bool:
    """The ComfyUI process's hook, at node import; ComfyUI has imported comfy.ops by then."""
    return install("the driver")
