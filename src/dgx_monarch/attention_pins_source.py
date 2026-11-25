"""Whose attention pins a render runs: the checkpoint's own, as its header set them, or not.

In a world of one GPU a pin from the checkpoint's own header runs as stock ComfyUI runs it,
so it asks nothing (docs/DESIGN.md, class K; docs/ADAPTERS.md). Every other pin
keeps the class-K Sol-Attn waiver. "Its own" is read strictly:

* any LoRA object patch on a ``comfy_attention`` path makes the render's pins a
  LoRA's, one that restates the header pin unchanged included;
* each module the forward will run at a pinned path must carry the kernel and
  tau the load scan cached. That is the module ComfyUI puts back from the
  shared ``object_patches_backup`` before the forward (``ModelPatcher.unpatch_model``
  inside ``partially_load``) when it holds one, and the live module otherwise.
  A module that differs, or a path that no longer resolves, counts as a LoRA
  pin: a stale LoRA module left live never renders quietly.

Cost: one attribute chain per pinned path and per backed-up pin path, about 50
on the largest pinned file; no module walk. Standard library only.
"""
from __future__ import annotations

from typing import Any, Final

# In a world of one GPU only a pin the checkpoint's header did not set reaches
# the class-K raise; ``assert_waived`` adds this to its message there.
ALONE_SOURCE: Final = ("In a world of one GPU the checkpoint's own pins run as stock runs them and "
                       "ask nothing; this stack's LoRA changes the checkpoint's pins, or a live "
                       "attention module differs from the checkpoint's. ")


def checkpoint_only(patcher: Any) -> bool:
    """Whether every pin ``patcher`` renders is its checkpoint's own, unchanged."""
    from . import attention_pins as pins

    model = getattr(patcher, "model", None)
    if model is None:
        return False

    def on_pin_path(path: Any) -> bool:
        return isinstance(path, str) and path.rpartition(".")[2] == pins.PIN_MODULE

    if any(on_pin_path(path) for path in (getattr(patcher, "object_patches", None) or {})):
        return False
    header = pins._checkpoint_pins(patcher)
    backup = {path: module for path, module in
              (getattr(patcher, "object_patches_backup", None) or {}).items() if on_pin_path(path)}
    for path in set(header) | set(backup):
        module = backup[path] if path in backup else pins._resolve(model, path)
        if module is None:
            return False
        want = header.get(path)
        if want is not None and want.cleared:
            # Unreachable in a world of one GPU, which has no FSDP; a cleared pin
            # runs nothing, so a caller off one GPU reads it as the checkpoint's.
            continue
        found = pins.block_pin(module)
        if (found is None) != (want is None):
            return False
        if found is not None and want is not None and (found.kernel, found.tau) != (want.kernel, want.tau):
            return False
    return True
