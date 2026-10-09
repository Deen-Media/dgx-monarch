"""The lazy xFuser ring surface a real long-context constructor imports."""
from __future__ import annotations

import importlib


def load_gpu_surface(import_module=importlib.import_module) -> None:
    """Import the public runtime path without creating a group or model."""
    import_module("xfuser.core.distributed")
    ring = import_module("xfuser.core.long_ctx_attention.ring")
    if not callable(getattr(ring, "xdit_ring_flash_attn_func", None)):
        raise RuntimeError("xFuser ring backend does not export xdit_ring_flash_attn_func")
