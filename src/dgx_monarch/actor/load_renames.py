"""The file key of a tensor ComfyUI renames as it loads a checkpoint.

A ComfyUI model config may rename a checkpoint's keys on the way into the
model (``supported_models.*.process_unet_state_dict``). FLUX files store a
q/k norm weight as ``<...>_norm.scale`` and the model holds it as
``<...>_norm.weight``; Chroma, Hunyuan3D and HunyuanVideo rename more. A LoRA
names the model's key, so a bake that restores a key from the file has to find
the file's name for it. ``actor/unbake_file._candidate_file_keys`` knows the
namespaces a key may sit under, not these renames.

The inverse comes from ComfyUI's own function, run over the header's names
with a stand-in for each tensor, so no rename list lives here and a ComfyUI
that adds one is followed. The stand-in carries the file key and the header's
shape. Some configs also build weights as they rename (PixelDiT splits a
modulation weight in two and strips ``net.`` from every other key): what a
config makes by reading a stand-in names no file key, and the keys it only
moves still map. A name is never trusted on its own: the caller still checks
the file tensor's dtype, shape and bytes against the live parameter
(``actor/fsdp_lora.capture_record``).

Standard library only, and no ComfyUI import: the function arrives on the
loaded model's own config.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# The namespaces a diffusion model's tensors sit under in a checkpoint file,
# longest first, as unbake_file._candidate_file_bindings spells them.
FILE_NAMESPACES = ("model.diffusion_model.", "diffusion_model.", "")
PATCHER_NAMESPACE = "diffusion_model."


class _Built:
    """What a config makes by reading a stand-in (a slice, a view, a cast).

    It is no tensor of the file, so the weight it becomes maps to no file key.
    Any read on a stand-in, or on what such a read gave, yields it. Arithmetic
    on it raises, and ``load_time_renames`` then knows no rename under that
    namespace.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return _BUILT

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _BUILT

    def __getitem__(self, index: Any) -> Any:
        return _BUILT


_BUILT = _Built()


class _Named(_Built):
    """Stands in for one file tensor: its file key and, where the header has one, its shape."""

    __slots__ = ("file_key", "shape")

    def __init__(self, file_key: str, entry: Any) -> None:
        self.file_key = file_key
        shape = getattr(entry, "shape", None)
        if isinstance(shape, (tuple, list)) and all(isinstance(n, int) for n in shape):
            self.shape = tuple(shape)


def _own_copy(config: Any) -> Any:
    """A shallow copy of a model config, so one that keeps state as it runs
    (ComfyUI's SAM3 stashes keys on itself) leaves the model's own alone.

    Not ``copy.copy``: ComfyUI's config base answers any missing attribute,
    ``__setstate__`` included, with ``None`` and a logged warning.
    """
    try:
        cls: Any = type(config)
        clone = cls.__new__(cls)
        clone.__dict__.update(vars(config))
        return clone
    except Exception:
        return config


def load_time_renames(model: Any, header: Iterable[str]) -> dict[str, str]:
    """Model key to file key, for each tensor ComfyUI renames as it loads this file.

    Keys are the model's names without the patcher's ``diffusion_model.``
    prefix; values are file keys present in ``header``. Only tensors ComfyUI
    moves whole under another name appear: a weight it builds from part of a
    tensor has no file key. A model with no config, a config with no such
    function, or one whose function fails on the stand-ins, yields ``{}``: the
    caller then finds no tensor and refuses.
    """
    process = getattr(_own_copy(getattr(model, "model_config", None)), "process_unet_state_dict", None)
    if not callable(process):
        return {}
    entries = header if isinstance(header, Mapping) else {}
    # The file's own key order, so every rank builds the same map from the same file.
    ordered = [name for name in header if isinstance(name, str)]
    present = set(ordered)
    renames: dict[str, str] = {}
    for namespace in FILE_NAMESPACES:
        names = {name[len(namespace):]: _Named(name, entries.get(name))
                 for name in ordered if name.startswith(namespace)}
        if not names:
            continue
        try:
            loaded = process(names)
        except Exception:  # a config that computes with tensor values: no renames known
            continue
        if not isinstance(loaded, dict):
            continue
        for model_key, moved in loaded.items():
            if not (isinstance(model_key, str) and isinstance(moved, _Named)):
                continue
            file_key = moved.file_key
            if file_key in present and model_key != file_key[len(namespace):]:
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
