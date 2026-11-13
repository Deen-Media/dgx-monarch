"""What a driver node's model input costs the driver's box, as the loader charges it.

A checkpoint or diffusion model a stock loader names loads whole on the
driver, so its file is the charge. LTX's audio-video text encoder loader
(``LTXAVTextEncoderLoader``) names an LTX checkpoint for its text projection
alone. In ComfyUI's code:

- ``comfy.sd.load_clip`` loads each file with ``comfy.utils.load_torch_file``
  and hands the state dicts to ``CLIP``, which calls ``load_sd`` on each and
  keeps none (comfy/sd.py, ``load_clip`` and ``CLIP.__init__``);
- with DynamicVRAM on, ``load_torch_file`` maps a safetensors file
  (``load_safetensors``, a ``comfy_aimdo`` mapping), so a tensor is resident
  only once something reads it (comfy/utils.py);
- ``LTXAVTEModel.load_sd`` keeps the checkpoint's ``text_embedding_projection.``
  tensors, and its two embeddings connectors only in compat mode: where the
  audio connector has no third block and its first block's query bias is 3840
  wide (comfy/text_encoders/lt.py). Nothing else of the checkpoint is read.

So with DynamicVRAM on, the charge is the bytes of the kept tensors, read from
the file's header (``ltx_text_bytes``), cached by path, size and time. Without
DynamicVRAM the safetensors library decides what a tensor holds, which
ComfyUI's code does not show, and an unreadable header shows nothing: both keep
the whole file, the price as it was.
"""
from __future__ import annotations

import os
import sys
import threading

LTX_TEXT_LOADER = "LTXAVTextEncoderLoader"
_PROJECTION = "text_embedding_projection."
_CONNECTORS = ("model.diffusion_model.video_embeddings_connector.",
               "model.diffusion_model.audio_embeddings_connector.")
_AUDIO_BLOCK = "model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.{}.attn1.to_q.bias"
_COMPAT_WIDTH = 3840
_LIMIT = 64
_LOCK = threading.Lock()
# Session lifetime, capped at _LIMIT, oldest first: the kept bytes per
# (path, size, mtime). A changed file reads again; a lost entry reads the
# header again.
_KEPT: dict[tuple[str, int, int], int] = {}


def ltx_text_bytes(path: str) -> int:
    """The bytes ``LTXAVTEModel.load_sd`` keeps of this LTX checkpoint, from its header; raises where unread."""
    from ..safetensors_header import read_safetensors_header

    stat = os.stat(path)
    key = (path, stat.st_size, stat.st_mtime_ns)
    if key in _KEPT:
        return _KEPT[key]
    tensors = read_safetensors_header(path).tensors
    first = tensors.get(_AUDIO_BLOCK.format(0))
    compat = _AUDIO_BLOCK.format(2) not in tensors and first is not None and first.shape[:1] == (_COMPAT_WIDTH,)
    kept = (_PROJECTION, *_CONNECTORS) if compat else (_PROJECTION,)
    figure = sum(tensor.end - tensor.begin for name, tensor in tensors.items() if name.startswith(kept))
    with _LOCK:
        _KEPT[key] = figure
        while len(_KEPT) > _LIMIT:
            _KEPT.pop(next(iter(_KEPT)), None)
    return figure


def driver_model_bytes(class_type: str, key: str, value: str) -> int:
    """The loader's charge for one model input of a driver node."""
    from .loader_graph import _DRIVER_MODEL_KEYS, _artifact_bytes

    folders = _DRIVER_MODEL_KEYS[key]
    memory = sys.modules.get("comfy.memory_management")
    if class_type != LTX_TEXT_LOADER or key != "ckpt_name" or not getattr(memory, "aimdo_enabled", False):
        return _artifact_bytes(folders, value)
    try:
        import folder_paths

        path = next(found for found in (folder_paths.get_full_path(folder, value) for folder in folders) if found)
        return ltx_text_bytes(str(path))
    except Exception:
        return _artifact_bytes(folders, value)
