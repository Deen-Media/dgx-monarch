"""Smaller files on disk that would fit in place of a model the loader priced.

``graph_load_price`` asks for them when a load does not fit and names the ones
that fit. A DiT file is named only when this graph would run it: the remedy is
never one the graph would refuse for another reason (docs/DESIGN.md section
5.9). A driver model's file is priced by pricing the graph that loads it, so
it takes the same credits as the graph it replaces. No ComfyUI import at
module scope.
"""
from __future__ import annotations

import functools
import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from .. import driver_footprint
from .graph_live import live_nodes
from .loader_graph import DriverStackTerms

_GIB = 2 ** 30
DIT_FOLDERS = ("diffusion_models", "unet")
_TE_FOLDERS = ("text_encoders", "clip", "clip_vision")
_VAE_FOLDERS = ("vae",)
_MODEL_FOLDERS = ("checkpoints", "diffusion_models", "unet")
# The streaming shard build's kinds (adapters/fsdp.py): only these halve the
# DiT on a pair, so only these name the pair's preset.
_SHARDED_KINDS = frozenset({"bf16", "fp16"})
# A DiT twin is named only when it is a file the repository's manifest lists
# (a graph here names it), of a kind every topology here runs, and when none of
# the class P or K refusals the driver can read from its header and the graph
# would meet it: the LoRA route, the FSDP launch, the upstream gate, a pinned
# approximate attention kernel off sequence parallelism. Read-only after import.
_MANIFEST = Path(__file__).resolve().parents[3] / "example_workflows" / "artifacts.toml"
_TWIN_KINDS = frozenset({"bf16", "fp16", "fp8", "int8", "w4a8"})
_PIN_SUFFIX = ".comfy_attention.config"
_SEQUENCE_PRESETS = ("uly", "ring")
# A precision or quantization token at the end of a file stem: the files that
# differ only here are one model in another precision, as MODELS lists them.
_PRECISION = re.compile(
    r"[-_.](?:bf16|fp16|fp32|f16|f32|fp8(?:[-_](?:e4m3fn|e5m2|scaled|mixed))*|"
    r"fp8mixed|int8(?:[-_]convrot)?|int4|w4a8|w4a16|w8a8|nvfp4|mxfp8|"
    r"q\d(?:_k)?(?:_[msl])?)$", re.IGNORECASE)
_MAX_TWINS = 3
_MAX_MODELS = 2

def _gib(value: float) -> str:
    return f"{value / _GIB:.1f}"


def _join(items: Iterable[str], word: str = "and") -> str:
    items = list(items)
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


def _stem(name: str) -> str | None:
    match = _PRECISION.search(os.path.splitext(os.path.basename(name))[0])
    return os.path.splitext(os.path.basename(name))[0][:match.start()] if match else None


def twins_on_disk(folders: tuple[str, ...], name: str, below: int) -> list[tuple[str, int]]:
    """Files on disk here that hold the same model in another precision, under ``below`` bytes.

    Same folder and subfolder, same extension, and the same stem once a
    precision token is taken off the end. Largest first, the closest copy.
    """
    stem = _stem(name)
    if not stem:
        return []
    import folder_paths

    where, ext = os.path.dirname(name), os.path.splitext(name)[1].lower()
    found: dict[str, int] = {}
    for folder in folders:
        try:
            listed = folder_paths.get_filename_list(folder)
        except Exception:  # an unknown folder lists nothing
            continue
        for other in listed:
            if (other == name or other in found or os.path.dirname(other) != where
                    or os.path.splitext(other)[1].lower() != ext or _stem(other) != stem):
                continue
            size = driver_footprint.file_size_bytes(folder_paths.get_full_path(folder, other))
            if 0 < size < below:
                found[other] = size
    return sorted(found.items(), key=lambda item: -item[1])


def _same_model(path: str, family: str) -> str | None:
    """The twin's quant kind when its header reads the same family, else None."""
    from ..adapters.detect import sniff_checkpoint

    try:
        sniffed, kind = sniff_checkpoint(path)
    except Exception:  # a header that cannot be read is never named
        return None
    return kind if sniffed == family else None


@functools.lru_cache(maxsize=1)
def shipped_files() -> frozenset[tuple[str, str]]:
    """The (folder, file) rows of the repository's manifest, or none when it is absent."""
    try:
        import tomllib

        rows = tomllib.loads(_MANIFEST.read_text()).get("artifact", [])
    except Exception:  # no manifest beside this checkout: no DiT twin is named
        return frozenset()
    return frozenset((str(row.get("dest")), str(row.get("file")))
                     for row in rows if isinstance(row, Mapping))


def _pins_attention(path: str) -> bool:
    """Whether the file pins an attention kernel (attention_pins.py), from its header."""
    from ..safetensors_header import read_safetensors_header

    try:
        return any(name.endswith(_PIN_SUFFIX) for name in read_safetensors_header(path).tensors)
    except Exception:  # an unreadable header could pin anything
        return True


def _loras_pin(loras: tuple[str, ...]) -> bool:
    """Whether a LoRA of the graph carries an attention pin, from its header; unreadable reads as yes."""
    import folder_paths

    for name in loras:
        try:
            path = folder_paths.get_full_path("loras", name)
            if not path or _pins_attention(str(path)):
                return True
        except Exception:  # a LoRA that cannot be read could pin anything
            return True
    return False


def twin_admitted(path: str, kind: str | None, *, family: str, loras: tuple[str, ...],
                  mesh: Any) -> bool:
    """Whether this graph would run ``path`` in place of its DiT, as far as the
    driver can tell before any load. Each test is a refusal that file would
    otherwise meet: class P for the LoRA route (TROUBLESHOOTING #129), the FSDP
    launch and the upstream gate, class K for a pinned kernel off sequence
    parallelism (#117), except the twin's own pins in a world of one GPU, which
    run as stock runs them while no LoRA of the graph pins."""
    from .. import lora_quant_guard, upstream_gate
    from ..adapters.fsdp import FSDP_ADMITTED_CHECKPOINT_KINDS
    from ..topology import declares_fsdp

    preset = str(getattr(mesh, "topology_preset", "auto") or "auto")
    if (("diffusion_models", os.path.basename(path)) not in shipped_files()
            or kind not in _TWIN_KINDS or lora_quant_guard.lora_refused(family, kind, loras)
            or (declares_fsdp(preset) and kind not in FSDP_ADMITTED_CHECKPOINT_KINDS)):
        return False
    try:
        upstream_gate.raise_refusal(path)
    except Exception:
        return False
    world = getattr(mesh, "world", None)  # an unknown world is not a world of one GPU
    sequence = preset.startswith(_SEQUENCE_PRESETS) and isinstance(world, int) and world >= 2
    alone = isinstance(world, int) and world < 2
    return sequence or not _pins_attention(path) or (alone and not _loras_pin(loras))


def dit_twins(shape: Any, fits: Callable[[Any], bool], unet_name: str, family: str,
               *, loras: tuple[str, ...], mesh: Any) -> list[str]:
    import folder_paths

    named = []
    for twin, size in twins_on_disk(DIT_FOLDERS, unet_name, shape.weight_bytes):
        path = str(folder_paths.get_full_path("diffusion_models", twin))
        kind = _same_model(path, family)
        world = shape.fsdp_world if kind in _SHARDED_KINDS else 0
        if (kind is not None and twin_admitted(path, kind, family=family, loras=loras, mesh=mesh)
                and fits(replace(shape, weight_bytes=size, fsdp_world=world))):
            named.append(f"{twin} ({_gib(size)} GiB)")
        if len(named) == _MAX_TWINS:
            break
    return named


def _driver_inputs(prompt: Mapping) -> list[tuple[str, tuple[str, ...]]]:
    """Each driver model a live loader names, by its full relative name, with its folders."""
    from .loader_graph import _DRIVER_MODEL_KEYS, _TE_NAME_KEYS, _VAE_NAME_KEYS, _WORKER_LOADER_PREFIX

    live = live_nodes(prompt)
    found: dict[str, tuple[str, ...]] = {}
    for key, node in prompt.items():
        inputs = node.get("inputs") if isinstance(node, Mapping) else None
        if str(key) not in live or not isinstance(inputs, Mapping):
            continue
        on_driver = not str(node.get("class_type", "")).startswith(_WORKER_LOADER_PREFIX)
        for name, value in inputs.items():
            if not isinstance(name, str) or not isinstance(value, str) or not value:
                continue
            folders = (_TE_FOLDERS if name.startswith(_TE_NAME_KEYS) else _VAE_FOLDERS
                       if name.startswith(_VAE_NAME_KEYS) else _MODEL_FOLDERS
                       if name in _DRIVER_MODEL_KEYS and on_driver else ())
            if folders:
                found.setdefault(value, folders)
    return list(found.items())


def _swapped(prompt: Mapping, old: str, new: str) -> dict:
    """The graph with every loader input naming ``old`` naming ``new`` instead."""
    return {key: ({**node, "inputs": {name: new if value == old else value
                                      for name, value in node["inputs"].items()}}
                  if isinstance(node, Mapping) and isinstance(node.get("inputs"), Mapping)
                  else node) for key, node in prompt.items()}


def model_twins(shape: Any, fits: Callable[[Any], bool], terms: DriverStackTerms,
                 load_of: Callable[[DriverStackTerms], int], prompt: object,
                 stack_for: Callable[[object], DriverStackTerms]) -> list[str]:
    """Twins of the driver models, each priced by pricing the graph that loads it.

    The swapped graph goes through the same credits as this one, so a held
    model's credit is never counted twice, and a file in a subfolder is looked
    for beside itself. Files are named as the loader widget shows them, by
    their path relative to the model folder.
    """
    from .loader_graph import _artifact_bytes

    if not isinstance(prompt, Mapping) or not terms.models_bytes:
        return []
    charged, clauses = set(terms.models), []
    for name, folders in _driver_inputs(prompt):
        if os.path.basename(name) not in charged:
            continue
        named = []
        for twin, size in twins_on_disk(folders, name, _artifact_bytes(folders, name)):
            other = stack_for(_swapped(prompt, name, twin))
            load = load_of(other) - load_of(terms)      # what the load window carries
            if fits(replace(shape, stack_bytes=shape.stack_bytes + other.total - terms.total,
                            load_stack_bytes=shape.load_stack_bytes + load)):
                named.append(f"{twin} ({_gib(size)} GiB)")
            if len(named) == _MAX_TWINS:
                break
        if named:
            clauses.append(f"{_join(named, 'or')} in place of {name}")
        if len(clauses) == _MAX_MODELS:
            break
    return clauses
