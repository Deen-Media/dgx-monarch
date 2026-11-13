"""What a driver node's model input costs the driver's box, as the loader charges it.

A checkpoint or diffusion model a stock loader names loads whole on the
driver, so its file is the charge. LTX's audio-video text encoder loader
(``LTXAVTextEncoderLoader``) names an LTX checkpoint for its text projection
alone: ComfyUI's ``load_clip`` keeps the tensors under ``LTX_TEXT_PREFIXES``
(comfy/text_encoders/lt.py, ``LTXAVTEModel.load_sd``) and drops the rest.
ComfyUI maps a safetensors file (``comfy.utils.load_torch_file``: the
DynamicVRAM loader, or ``safe_open`` without it), so the rest of the checkpoint
never becomes resident, and the charge is the kept tensors' bytes, read from the
file's header. With ``--disable-mmap`` and DynamicVRAM off, ``load_torch_file``
copies every tensor first, so the whole file is the charge. An unreadable
header also charges the whole file, the price as it was.
"""
from __future__ import annotations

import sys

LTX_TEXT_LOADER = "LTXAVTextEncoderLoader"
LTX_TEXT_PREFIXES = ("text_embedding_projection.", "model.diffusion_model.video_embeddings_connector.",
                     "model.diffusion_model.audio_embeddings_connector.")


def _reads_whole_file() -> bool:
    """Whether ComfyUI copies every tensor of a safetensors file as it loads it."""
    memory = sys.modules.get("comfy.memory_management")
    utils = sys.modules.get("comfy.utils")
    return not getattr(memory, "aimdo_enabled", False) and bool(getattr(utils, "DISABLE_MMAP", False))


def driver_model_bytes(class_type: str, key: str, value: str) -> int:
    """The loader's charge for one model input of a driver node."""
    from .loader_graph import _DRIVER_MODEL_KEYS, _artifact_bytes

    folders = _DRIVER_MODEL_KEYS[key]
    if class_type != LTX_TEXT_LOADER or key != "ckpt_name" or _reads_whole_file():
        return _artifact_bytes(folders, value)
    try:
        import folder_paths

        from ..safetensors_header import read_safetensors_header

        path = next(found for found in (folder_paths.get_full_path(folder, value) for folder in folders) if found)
        tensors = read_safetensors_header(path).tensors
        return sum(tensor.end - tensor.begin for name, tensor in tensors.items()
                   if name.startswith(LTX_TEXT_PREFIXES))
    except Exception:
        return _artifact_bytes(folders, value)
