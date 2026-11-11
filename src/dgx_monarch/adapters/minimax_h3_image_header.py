"""MiniMax-H3-Image: what a safetensors header alone proves about it.

ComfyUI 0.45.0 (upstream pull request 17655) builds a MiniMax H3 file from two
key paths, ``video_patch_proj.weight`` and ``condition_proj.weight``, and names
it the image model when ``audio_patch_proj.weight`` is absent
(``comfy/model_detection.py``, ``image_model`` "minimax_h3_image", built as
``model_base.MiniMaxH3Image``). ComfyUI 0.44.0 detects no such file. The
detector's ``_SIGNATURES`` row carries the same two paths after H3's row, whose
audio path takes every file that has one first.
``tests/test_minimax_h3_image_comfy_detection.py`` holds both against ComfyUI's
own detection in the canary workflow.

Three more facts come from the header, none needing the model:

* The F16 adaLN linears. Every published DiT keeps each block's and the final
  layer's adaLN projection on the curve basis in F16 (51 tensors, 0.19 percent
  of the BF16 file) beside a BF16 core. They are islands, not an F16 core, so the
  file reads ``bf16``, and the FSDP precision proof counts their bytes as island
  bytes under the ceiling F32 islands already meet. Any other F16 tensor keeps
  the reading ``fp16``.
* The sigma grid. ``MiniMaxH3ImageModel`` takes ``pdd_sigmas`` from the file's
  ``config`` metadata (``transformer.pdd_sigmas``) and has no default for it, so
  stock ComfyUI fails to build a file without it. The upstream gate refuses such
  a file first (``upstream_gate.py``).
* The family itself, which this release carries no model forward for: the
  upstream gate refuses every file of the family before any load.

Its own imports are standard library only, as ``upstream_gate.py``'s are; importing
it still loads the ``dgx_monarch.adapters`` package, whose registry imports torch.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, Final

FAMILY: Final = "minimax_h3_image"
# ComfyUI's own pair; H3's row (video_patch_proj. plus audio_patch_proj.) comes first.
SIGNATURE: Final = ("video_patch_proj.", "condition_proj.")
# The adaLN projections on the curve basis, the only F16 tensors a published file holds.
_ADALN_ISLAND: Final = re.compile(r"(?:^|\.)(?:blocks\.\d+|final_layer)\.adaln_proj\.linear\.weight$")


def f16_islands(keys: Iterable[str], tensors: Mapping[str, Any]) -> frozenset[str]:
    """The F16 tensors that are this family's adaLN islands, or none.

    Non-empty only when the keys read as MiniMax-H3-Image and every F16 tensor in
    the header is an adaLN projection weight (``blocks.N.adaln_proj.linear.weight``
    or ``final_layer.adaln_proj.linear.weight``). One F16 tensor anywhere else, or
    another family, answers none, so the header keeps its ``fp16`` reading.
    """
    from .detect import detect_family_from_keys

    if detect_family_from_keys(list(keys)) != FAMILY:
        return frozenset()
    f16 = [name for name, info in tensors.items()
           if isinstance(info, Mapping) and info.get("dtype") == "F16"]
    if not f16 or not all(_ADALN_ISLAND.search(name) for name in f16):
        return frozenset()
    return frozenset(f16)


def grid_missing(metadata: Mapping[str, Any] | None) -> bool:
    """Whether the header's ``config`` metadata lacks ``transformer.pdd_sigmas``.

    ComfyUI reads the grid with ``json.loads(metadata["config"])["transformer"]``
    and passes it to the model, which takes no default. A grid that is not a
    non-empty list of numbers, or a config that is not JSON, counts as missing.
    The length against ``pdd_steps`` is not checked here; the family's adapter
    checks it when it retires the family refusal.
    """
    raw = (metadata or {}).get("config")
    if not isinstance(raw, str):
        return True
    try:
        config = json.loads(raw)
    except ValueError:
        return True
    transformer = config.get("transformer") if isinstance(config, dict) else None
    grid = transformer.get("pdd_sigmas") if isinstance(transformer, dict) else None
    return not (isinstance(grid, list) and grid
                and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        for value in grid))
