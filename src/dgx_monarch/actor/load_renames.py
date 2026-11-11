"""The file key of a tensor ComfyUI renames as it loads a checkpoint.

A ComfyUI model config may rename a checkpoint's keys on the way into the
model (``supported_models.*.process_unet_state_dict``). FLUX files store a
q/k norm weight as ``<...>_norm.scale`` and the model holds it as
``<...>_norm.weight``; Chroma, Hunyuan3D and HunyuanVideo rename more. A LoRA
names the model's key, so a bake that restores a key from the file has to find
the file's name for it. ``actor/unbake_file._candidate_file_keys`` knows the
namespaces a key may sit under, not these renames.

The inverse comes from ComfyUI's own function, run over the header's names
with each name standing in for its tensor, so no rename list lives here and a
ComfyUI that adds one is followed. A name is never trusted on its own: the
caller still checks the file tensor's dtype, shape and bytes against the live
parameter (``actor/fsdp_lora.capture_record``).

Standard library only, and no ComfyUI import: the function arrives on the
loaded model's own config.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# The namespaces a diffusion model's tensors sit under in a checkpoint file,
# longest first, as unbake_file._candidate_file_bindings spells them.
FILE_NAMESPACES = ("model.diffusion_model.", "diffusion_model.", "")
PATCHER_NAMESPACE = "diffusion_model."


def load_time_renames(model: Any, header: Iterable[str]) -> dict[str, str]:
    """Model key to file key, for each tensor ComfyUI renames as it loads this file.

    Keys are the model's names without the patcher's ``diffusion_model.``
    prefix; values are file keys present in ``header``. Only renamed tensors
    appear. A model with no config, a config with no such function, or one
    that reads tensor values and so fails on a name, yields ``{}``: the caller
    then finds no tensor and refuses.
    """
    process = getattr(getattr(model, "model_config", None), "process_unet_state_dict", None)
    if not callable(process):
        return {}
    # The file's own key order, so every rank builds the same map from the same file.
    ordered = [name for name in header if isinstance(name, str)]
    present = set(ordered)
    renames: dict[str, str] = {}
    for namespace in FILE_NAMESPACES:
        names = {name[len(namespace):]: name for name in ordered if name.startswith(namespace)}
        if not names:
            continue
        try:
            loaded = process(dict(names))
        except Exception:  # a config that reads tensor values: no renames known
            continue
        if not isinstance(loaded, dict):
            continue
        for model_key, file_key in loaded.items():
            if (isinstance(model_key, str) and isinstance(file_key, str) and file_key in present
                    and model_key != file_key[len(namespace):]):
                renames.setdefault(model_key, file_key)
    return renames


def renamed_file_key(renames: dict[str, str], key: str) -> str | None:
    """The file key ``load_time_renames`` holds for a patcher key, or ``None``."""
    bare = key[len(PATCHER_NAMESPACE):] if key.startswith(PATCHER_NAMESPACE) else key
    return renames.get(bare)


def with_renamed(candidates: Iterable[str], renames: dict[str, str] | None, key: str) -> list[str]:
    """``candidates`` (the namespace spellings of ``key``) with its renamed file key last."""
    out = list(candidates)
    renamed = renamed_file_key(renames, key) if renames else None
    if renamed is not None and renamed not in out:
        out.append(renamed)
    return out


__all__ = ["FILE_NAMESPACES", "PATCHER_NAMESPACE", "load_time_renames", "renamed_file_key",
           "with_renamed"]
