"""What must match between the driver and a Worker for a split decode to be exact.

A Worker's tile is only as good as the process that decoded it. ComfyUI's
flags never reach the actors (they start with a bare argv), so a driver run
with ``--bf16-vae`` or ``--use-sage-attention`` decodes differently from a
default actor. Three rules follow, applied by the Worker before each unit:

* Compared, never adopted: the ComfyUI commit, torch and CUDA versions, the
  comfy-kitchen release and the VAE class. A difference means a different
  decode function, so that rank declines and the driver decodes its units.
* Adopted for the decode: the VAE dtype (the Worker builds its VAE with the
  driver's), the output dtype, the attention function each VAE module calls
  and the torch backend flags that change matmul and convolution numerics.
  The Worker sets them under its GPU lock and puts its own back after.
* An attention kernel the Worker cannot run (one ComfyUI registers only when
  its package imports, such as sage) is a decline, not a substitution.
"""
from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

COMPARED = ("comfy", "torch", "cuda", "kitchen", "vae_class")
_ATTENTION_NAME = "optimized_attention"
_GLOBAL = "global:"
_ATTRIBUTE = "attr:" + _ATTENTION_NAME


class Decline(RuntimeError):
    """This rank cannot decode the unit exactly; the driver decodes it."""


def qualified(obj: Any) -> str:
    return f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', type(obj).__qualname__)}"


def _dist_version(*names: str) -> str:
    from importlib import metadata

    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return "absent"


def _registry() -> dict[str, Any]:
    try:
        attention = importlib.import_module("comfy.ldm.modules.attention")
    except Exception:
        return {}
    return dict(getattr(attention, "REGISTERED_ATTENTION_FUNCTIONS", {}) or {})


def _binding(fn: Any, registry: dict[str, Any]) -> dict[str, Any]:
    key = next((name for name, value in registry.items() if value is fn), None)
    return {"name": qualified(fn), "registry": key}


def attention_bindings(fsm: Any) -> dict[str, dict[str, Any]]:
    """The attention function each module of the VAE calls, by where it is bound.

    A module-level ``optimized_attention`` (MiniMax H3's decoder imports
    ComfyUI's global choice by name) and an instance attribute of that name
    (the Wan VAE picks one at construction) are the two seams in ComfyUI's
    VAEs; both are recorded so a Worker can bind the driver's choice.
    """
    registry = _registry()
    out: dict[str, dict[str, Any]] = {}
    owners: set[str] = set()
    attributes: dict[str, Any] = {}
    for module in fsm.modules():
        owner = type(module).__module__
        if owner not in owners:
            owners.add(owner)
            fn = getattr(sys.modules.get(owner), _ATTENTION_NAME, None)
            if callable(fn):
                out[_GLOBAL + owner] = _binding(fn, registry)
        fn = module.__dict__.get(_ATTENTION_NAME)
        if callable(fn):
            attributes[qualified(fn)] = fn
    if len(attributes) == 1:
        out[_ATTRIBUTE] = _binding(next(iter(attributes.values())), registry)
    elif attributes:
        out[_ATTRIBUTE] = {"name": ",".join(sorted(attributes)), "registry": None}
    return out


def backend_flags() -> dict[str, Any]:
    """Torch settings that change matmul or convolution results, by name."""
    import torch

    matmul = torch.backends.cuda.matmul
    flags: dict[str, Any] = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn.allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn.benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn.deterministic": bool(torch.backends.cudnn.deterministic),
    }
    for name in ("allow_tf32", "allow_fp16_reduced_precision_reduction",
                 "allow_bf16_reduced_precision_reduction", "allow_fp16_accumulation"):
        if hasattr(matmul, name):
            flags[f"cuda.matmul.{name}"] = getattr(matmul, name)
    return flags


def describe(vae: Any) -> dict[str, Any]:
    """This process's decode environment for ``vae`` (a ``comfy.sd.VAE``)."""
    import torch

    from .gate_ledger import comfy_commit

    fsm = vae.first_stage_model
    return {
        "comfy": comfy_commit(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "kitchen": _dist_version("comfy-kitchen", "comfy_kitchen"),
        "vae_class": qualified(type(fsm)),
        "attention": attention_bindings(fsm),
        "flags": backend_flags(),
    }


def mismatches(driver: dict[str, Any], here: dict[str, Any]) -> list[str]:
    """The compared fields that differ, as ``field: here != driver`` text."""
    return [f"{name} {here.get(name)!r} differs from the driver's {driver.get(name)!r}"
            for name in COMPARED if here.get(name) != driver.get(name)]


def _resolve(binding: dict[str, Any], registry: dict[str, Any]) -> Any:
    key = binding.get("registry")
    if key is not None:
        fn = registry.get(key)
        if fn is None:
            raise Decline(f"attention {key!r} is not available on this host")
        return fn
    name = str(binding.get("name", ""))
    parts = name.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        try:
            for attribute in parts[cut:]:
                obj = getattr(obj, attribute)
        except AttributeError:
            break
        if callable(obj):
            return obj
    raise Decline(f"attention {name!r} cannot be found on this host")


_PRECISION = "float32_matmul_precision"


def _set_flag(name: str, value: Any) -> None:
    import torch

    if name == _PRECISION:
        torch.set_float32_matmul_precision(value)
    elif name.startswith("cudnn."):
        setattr(torch.backends.cudnn, name.split(".", 1)[1], value)
    elif name.startswith("cuda.matmul."):
        setattr(torch.backends.cuda.matmul, name.rsplit(".", 1)[1], value)


def _apply_flags(wanted: dict[str, Any]) -> None:
    """Set ``wanted``, the float32 matmul precision last.

    ``cuda.matmul.allow_tf32`` is a legacy alias of that precision. On torch
    2.12 and later, writing the alias after the precision, even the value it
    already reads, mixes the legacy and new state and the precision getter
    raises. Written last, the precision sets every backend at once and the
    alias reads the value that precision implies.
    """
    for name, value in wanted.items():
        if name != _PRECISION:
            _set_flag(name, value)
    if _PRECISION in wanted:
        _set_flag(_PRECISION, wanted[_PRECISION])


def _unread_flags(wanted: dict[str, Any]) -> list[str]:
    """The flags of ``wanted`` that do not read back, every one when torch cannot say."""
    try:
        now = backend_flags()
    except Exception:
        return sorted(wanted)
    return [name for name, value in wanted.items() if now.get(name) != value]


@contextmanager
def adopted(fsm: Any, driver: dict[str, Any]) -> Iterator[None]:
    """Decode under the driver's attention functions and backend flags, then restore.

    Callers hold the actor's GPU lock, so no render runs under these settings.
    """
    registry = _registry()
    plans: list[tuple[Any, str, Any]] = []
    for key, binding in dict(driver.get("attention") or {}).items():
        if key.startswith(_GLOBAL):
            module = sys.modules.get(key[len(_GLOBAL):])
            if module is None or not hasattr(module, _ATTENTION_NAME):
                raise Decline(f"this host's {key[len(_GLOBAL):]} has no {_ATTENTION_NAME}")
            plans.append((module, _ATTENTION_NAME, _resolve(binding, registry)))
        elif key == _ATTRIBUTE:
            fn = _resolve(binding, registry)
            plans.extend((module, _ATTENTION_NAME, fn) for module in fsm.modules()
                         if _ATTENTION_NAME in module.__dict__)
    here = backend_flags()
    flags = {name: value for name, value in dict(driver.get("flags") or {}).items()
             if name in here and here[name] != value}
    saved = {name: here[name] for name in flags}
    undo: list[tuple[Any, str, Any]] = []
    try:
        _apply_flags(flags)
        unread = _unread_flags(flags)
        if unread:
            raise Decline(f"this host does not hold the driver's torch flags {', '.join(unread)}")
        for target, attribute, value in plans:
            undo.append((target, attribute, getattr(target, attribute)))
            setattr(target, attribute, value)
        yield
    finally:
        for target, attribute, old in reversed(undo):
            setattr(target, attribute, old)
        _apply_flags(saved)
