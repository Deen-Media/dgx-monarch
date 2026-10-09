"""The LoRA bake's header-only price: mapping, compute-dtype charge, fallback.

``KREA2_MAPPED_KEYS`` is the exact (key, dtype, shape) list
``krea2_turbo_lora_rank_64_bf16.safetensors`` maps onto
``krea2_raw_bf16.safetensors`` (271 of 271 targets, read off both real files'
headers on the head box, 2026-09-28; see docs/VALIDATION.md "The LoRA bake's
unpriced slab stray"). The real files are not in CI, so this module writes
sparse synthetic files: a safetensors header the real parser accepts,
truncated to the declared size, with no tensor bytes written or read.
"""
import json
import os
import struct

from dgx_monarch import capacity_lora_bake
from dgx_monarch.safetensors_header import DTYPE_BITS, read_safetensors_header

# (name, dtype, shape): the checkpoint's own keys, with no "diffusion_model." prefix.
KREA2_MAPPED_KEYS = [
    ('blocks.0.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.0.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.0.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.0.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.0.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.0.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.0.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.0.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.1.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.1.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.1.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.1.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.1.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.1.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.1.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.1.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.10.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.10.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.10.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.10.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.10.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.10.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.10.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.10.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.11.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.11.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.11.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.11.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.11.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.11.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.11.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.11.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.12.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.12.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.12.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.12.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.12.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.12.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.12.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.12.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.13.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.13.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.13.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.13.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.13.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.13.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.13.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.13.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.14.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.14.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.14.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.14.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.14.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.14.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.14.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.14.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.15.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.15.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.15.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.15.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.15.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.15.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.15.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.15.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.16.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.16.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.16.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.16.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.16.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.16.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.16.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.16.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.17.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.17.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.17.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.17.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.17.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.17.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.17.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.17.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.18.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.18.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.18.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.18.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.18.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.18.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.18.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.18.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.19.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.19.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.19.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.19.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.19.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.19.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.19.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.19.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.2.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.2.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.2.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.2.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.2.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.2.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.2.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.2.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.20.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.20.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.20.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.20.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.20.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.20.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.20.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.20.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.21.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.21.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.21.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.21.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.21.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.21.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.21.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.21.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.22.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.22.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.22.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.22.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.22.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.22.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.22.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.22.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.23.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.23.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.23.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.23.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.23.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.23.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.23.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.23.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.24.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.24.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.24.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.24.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.24.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.24.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.24.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.24.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.25.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.25.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.25.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.25.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.25.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.25.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.25.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.25.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.26.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.26.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.26.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.26.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.26.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.26.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.26.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.26.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.27.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.27.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.27.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.27.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.27.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.27.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.27.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.27.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.3.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.3.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.3.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.3.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.3.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.3.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.3.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.3.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.4.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.4.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.4.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.4.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.4.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.4.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.4.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.4.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.5.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.5.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.5.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.5.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.5.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.5.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.5.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.5.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.6.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.6.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.6.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.6.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.6.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.6.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.6.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.6.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.7.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.7.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.7.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.7.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.7.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.7.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.7.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.7.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.8.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.8.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.8.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.8.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.8.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.8.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.8.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.8.mlp.up.weight', 'BF16', [16384, 6144]),
    ('blocks.9.attn.gate.weight', 'BF16', [6144, 6144]),
    ('blocks.9.attn.wk.weight', 'BF16', [1536, 6144]),
    ('blocks.9.attn.wo.weight', 'BF16', [6144, 6144]),
    ('blocks.9.attn.wq.weight', 'BF16', [6144, 6144]),
    ('blocks.9.attn.wv.weight', 'BF16', [1536, 6144]),
    ('blocks.9.mlp.down.weight', 'BF16', [6144, 16384]),
    ('blocks.9.mlp.gate.weight', 'BF16', [16384, 6144]),
    ('blocks.9.mlp.up.weight', 'BF16', [16384, 6144]),
    ('first.bias', 'F32', [6144]),
    ('first.weight', 'F32', [6144, 64]),
    ('last.linear.bias', 'F32', [64]),
    ('last.linear.weight', 'F32', [64, 6144]),
    ('tmlp.0.bias', 'F32', [6144]),
    ('tmlp.0.weight', 'F32', [6144, 256]),
    ('tmlp.2.bias', 'F32', [6144]),
    ('tmlp.2.weight', 'F32', [6144, 6144]),
    ('tproj.1.bias', 'F32', [36864]),
    ('tproj.1.weight', 'F32', [36864, 6144]),
    ('txtfusion.layerwise_blocks.0.attn.gate.weight', 'BF16', [2560, 2560]),
    ('txtfusion.layerwise_blocks.0.attn.wk.weight', 'BF16', [2560, 2560]),
    ('txtfusion.layerwise_blocks.0.attn.wo.weight', 'BF16', [2560, 2560]),
    ('txtfusion.layerwise_blocks.0.attn.wq.weight', 'BF16', [2560, 2560]),
    ('txtfusion.layerwise_blocks.0.attn.wv.weight', 'BF16', [2560, 2560]),
    ('txtfusion.layerwise_blocks.0.mlp.down.weight', 'BF16', [2560, 6912]),
    ('txtfusion.layerwise_blocks.0.mlp.gate.weight', 'BF16', [6912, 2560]),
    ('txtfusion.layerwise_blocks.0.mlp.up.weight', 'BF16', [6912, 2560]),
    ('txtfusion.layerwise_blocks.1.attn.gate.weight', 'BF16', [2560, 2560]),
    ('txtfusion.layerwise_blocks.1.attn.wk.weight', 'BF16', [2560, 2560]),
    ('txtfusion.layerwise_blocks.1.attn.wo.weight', 'BF16', [2560, 2560]),
    ('txtfusion.layerwise_blocks.1.attn.wq.weight', 'BF16', [2560, 2560]),
    ('txtfusion.layerwise_blocks.1.attn.wv.weight', 'BF16', [2560, 2560]),
    ('txtfusion.layerwise_blocks.1.mlp.down.weight', 'BF16', [2560, 6912]),
    ('txtfusion.layerwise_blocks.1.mlp.gate.weight', 'BF16', [6912, 2560]),
    ('txtfusion.layerwise_blocks.1.mlp.up.weight', 'BF16', [6912, 2560]),
    ('txtfusion.projector.weight', 'F32', [1, 12]),
    ('txtfusion.refiner_blocks.0.attn.gate.weight', 'BF16', [2560, 2560]),
    ('txtfusion.refiner_blocks.0.attn.wk.weight', 'BF16', [2560, 2560]),
    ('txtfusion.refiner_blocks.0.attn.wo.weight', 'BF16', [2560, 2560]),
    ('txtfusion.refiner_blocks.0.attn.wq.weight', 'BF16', [2560, 2560]),
    ('txtfusion.refiner_blocks.0.attn.wv.weight', 'BF16', [2560, 2560]),
    ('txtfusion.refiner_blocks.0.mlp.down.weight', 'BF16', [2560, 6912]),
    ('txtfusion.refiner_blocks.0.mlp.gate.weight', 'BF16', [6912, 2560]),
    ('txtfusion.refiner_blocks.0.mlp.up.weight', 'BF16', [6912, 2560]),
    ('txtfusion.refiner_blocks.1.attn.gate.weight', 'BF16', [2560, 2560]),
    ('txtfusion.refiner_blocks.1.attn.wk.weight', 'BF16', [2560, 2560]),
    ('txtfusion.refiner_blocks.1.attn.wo.weight', 'BF16', [2560, 2560]),
    ('txtfusion.refiner_blocks.1.attn.wq.weight', 'BF16', [2560, 2560]),
    ('txtfusion.refiner_blocks.1.attn.wv.weight', 'BF16', [2560, 2560]),
    ('txtfusion.refiner_blocks.1.mlp.down.weight', 'BF16', [2560, 6912]),
    ('txtfusion.refiner_blocks.1.mlp.gate.weight', 'BF16', [6912, 2560]),
    ('txtfusion.refiner_blocks.1.mlp.up.weight', 'BF16', [6912, 2560]),
    ('txtmlp.1.bias', 'F32', [6144]),
    ('txtmlp.1.weight', 'F32', [6144, 2560]),
    ('txtmlp.3.bias', 'F32', [6144]),
    ('txtmlp.3.weight', 'F32', [6144, 6144]),
]


def _write_safetensors(path, tensors: dict, metadata: dict | None = None) -> None:
    """A header-valid safetensors file with no tensor bytes written.

    ``tensors`` maps a key to ``(dtype, shape)``. Offsets are contiguous; the
    file is truncated (sparse) to the declared data size, which is all
    ``read_safetensors_header`` inspects.
    """
    header: dict = {}
    cursor = 0
    for key, (dtype, shape) in tensors.items():
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * DTYPE_BITS[dtype] // 8
        header[key] = {"dtype": dtype, "shape": list(shape),
                       "data_offsets": [cursor, cursor + nbytes]}
        cursor += nbytes
    if metadata:
        header["__metadata__"] = metadata
    body = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
    os.truncate(path, 8 + len(body) + cursor)


def _lora_entries_for(keys) -> dict:
    """The LoRA's own tensors for a set of base keys, ``diffusion_model.``
    prefixed, one matrix pair per ``.weight`` target and one delta per
    ``.bias`` target. Sizes are trivial: only the mapped stem matters."""
    entries: dict = {}
    for key, _dtype, _shape in keys:
        stem = f"diffusion_model.{key}"
        if key.endswith(".bias"):
            entries[stem[: -len(".bias")] + ".diff_b"] = ("BF16", [1])
        else:
            base = stem[: -len(".weight")]
            entries[f"{base}.lora_down.weight"] = ("BF16", [4, 4])
            entries[f"{base}.lora_up.weight"] = ("BF16", [4, 4])
    return entries


def _resolver(paths: dict):
    return lambda name: paths[name]


def test_the_krea2_header_pair_prices_23_88_gib(tmp_path):
    ckpt_path = tmp_path / "krea2_raw_bf16.safetensors"
    lora_path = tmp_path / "krea2_turbo_lora_rank_64_bf16.safetensors"
    _write_safetensors(ckpt_path, {k: (d, s) for k, d, s in KREA2_MAPPED_KEYS})
    _write_safetensors(lora_path, _lora_entries_for(KREA2_MAPPED_KEYS))

    tensors = read_safetensors_header(str(ckpt_path)).tensors
    stack = [{"name": "krea2_turbo_lora_rank_64_bf16.safetensors", "strength": 0.8}]
    priced = capacity_lora_bake.lora_bake_bytes(
        str(ckpt_path), tensors, stack, _resolver({stack[0]["name"]: str(lora_path)}))

    assert round(priced / (1 << 30), 2) == 23.88
    # As stored the keys total 24.47 GiB (15 F32 keys at 4 bytes). Every plain
    # key prices at two bytes instead, which matches the worker journal's
    # "reabsorbed 430 strays (23.88 GiB)" for this pair.


def test_an_empty_stack_changes_nothing(tmp_path):
    ckpt_path = tmp_path / "m.safetensors"
    _write_safetensors(ckpt_path, {"w.weight": ("BF16", [4, 4])})
    tensors = read_safetensors_header(str(ckpt_path)).tensors

    assert capacity_lora_bake.lora_bake_bytes(str(ckpt_path), tensors, None, _resolver({})) == 0
    assert capacity_lora_bake.lora_bake_bytes(str(ckpt_path), tensors, [], _resolver({})) == 0


def test_text_encoder_only_keys_add_nothing(tmp_path):
    ckpt_path = tmp_path / "m.safetensors"
    lora_path = tmp_path / "te_only.safetensors"
    _write_safetensors(ckpt_path, {"blocks.0.attn.wq.weight": ("BF16", [64, 64])})
    _write_safetensors(lora_path, {
        "lora_te1_text_model.encoder.layers.0.mlp.fc1.lora_down.weight": ("BF16", [4, 4]),
        "lora_te1_text_model.encoder.layers.0.mlp.fc1.lora_up.weight": ("BF16", [4, 4]),
    })
    tensors = read_safetensors_header(str(ckpt_path)).tensors
    stack = [{"name": "te_only.safetensors", "strength": 1.0}]

    priced = capacity_lora_bake.lora_bake_bytes(
        str(ckpt_path), tensors, stack, _resolver({stack[0]["name"]: str(lora_path)}))
    assert priced == 0


def test_an_unmapped_target_prices_the_whole_checkpoint(tmp_path):
    """A target this module cannot map never under-charges: it falls back to
    the checkpoint's own file size rather than the (smaller) mapped subset."""
    ckpt_path = tmp_path / "m.safetensors"
    lora_path = tmp_path / "fused.safetensors"
    _write_safetensors(ckpt_path, {
        "blocks.0.attn.wq.weight": ("BF16", [64, 64]),      # 8192 bytes
        "blocks.0.attn.qkv.weight": ("BF16", [192, 64]),    # fused; no to_q key
    })
    _write_safetensors(lora_path, {
        # Maps cleanly onto wq.
        "diffusion_model.blocks.0.attn.wq.lora_down.weight": ("BF16", [4, 4]),
        "diffusion_model.blocks.0.attn.wq.lora_up.weight": ("BF16", [4, 4]),
        # Diffusers-named, fused: no single base key answers it.
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": ("BF16", [4, 4]),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": ("BF16", [4, 4]),
    })
    tensors = read_safetensors_header(str(ckpt_path)).tensors
    stack = [{"name": "fused.safetensors", "strength": 1.0}]

    priced = capacity_lora_bake.lora_bake_bytes(
        str(ckpt_path), tensors, stack, _resolver({stack[0]["name"]: str(lora_path)}))
    file_size = os.path.getsize(ckpt_path)

    assert priced == file_size
    mapped_only = 64 * 64 * 2   # what a partial (wq-only) count would have priced
    assert priced > mapped_only, "the fallback must not under-charge the mapped subset"


def test_a_quantized_key_prices_at_its_stored_width_not_upcast(tmp_path):
    """comfy never upcasts a quantized stray: fp8 prices at one byte per
    element, not the two-byte compute-dtype charge a plain key gets."""
    ckpt_path = tmp_path / "m.safetensors"
    lora_path = tmp_path / "l.safetensors"
    _write_safetensors(ckpt_path, {"blocks.0.mlp.up.weight": ("F8_E4M3", [128, 128])})
    _write_safetensors(lora_path, {
        "diffusion_model.blocks.0.mlp.up.lora_down.weight": ("BF16", [4, 4]),
        "diffusion_model.blocks.0.mlp.up.lora_up.weight": ("BF16", [4, 4]),
    })
    tensors = read_safetensors_header(str(ckpt_path)).tensors
    stack = [{"name": "l.safetensors", "strength": 1.0}]

    priced = capacity_lora_bake.lora_bake_bytes(
        str(ckpt_path), tensors, stack, _resolver({stack[0]["name"]: str(lora_path)}))
    assert priced == 128 * 128 * 1   # stored width, not the 2-byte compute charge


def test_a_resolution_failure_prices_the_whole_checkpoint(tmp_path):
    """An unreadable or unresolvable LoRA is unknown, not free."""
    ckpt_path = tmp_path / "m.safetensors"
    _write_safetensors(ckpt_path, {"blocks.0.attn.wq.weight": ("BF16", [64, 64])})
    tensors = read_safetensors_header(str(ckpt_path)).tensors
    stack = [{"name": "missing.safetensors", "strength": 1.0}]

    def _raise(_name):
        raise FileNotFoundError("no such lora")

    priced = capacity_lora_bake.lora_bake_bytes(str(ckpt_path), tensors, stack, _raise)
    assert priced == os.path.getsize(ckpt_path)


def test_an_unrecognized_lora_container_prices_the_whole_checkpoint(tmp_path):
    """A file with no key this module's suffix table recognizes is unknown,
    not an empty, free stack."""
    ckpt_path = tmp_path / "m.safetensors"
    lora_path = tmp_path / "odd.safetensors"
    _write_safetensors(ckpt_path, {"blocks.0.attn.wq.weight": ("BF16", [64, 64])})
    _write_safetensors(lora_path, {"some.unrecognized.key": ("BF16", [4, 4])})
    tensors = read_safetensors_header(str(ckpt_path)).tensors
    stack = [{"name": "odd.safetensors", "strength": 1.0}]

    priced = capacity_lora_bake.lora_bake_bytes(
        str(ckpt_path), tensors, stack, _resolver({stack[0]["name"]: str(lora_path)}))
    assert priced == os.path.getsize(ckpt_path)


def test_a_stack_of_two_loras_dedupes_a_shared_target(tmp_path):
    """One bake, one stray per base key, even where two LoRAs both patch it."""
    ckpt_path = tmp_path / "m.safetensors"
    lora_a = tmp_path / "a.safetensors"
    lora_b = tmp_path / "b.safetensors"
    _write_safetensors(ckpt_path, {"blocks.0.attn.wq.weight": ("BF16", [64, 64])})
    for path in (lora_a, lora_b):
        _write_safetensors(path, {
            "diffusion_model.blocks.0.attn.wq.lora_down.weight": ("BF16", [4, 4]),
            "diffusion_model.blocks.0.attn.wq.lora_up.weight": ("BF16", [4, 4]),
        })
    tensors = read_safetensors_header(str(ckpt_path)).tensors
    stack = [{"name": "a.safetensors", "strength": 1.0},
             {"name": "b.safetensors", "strength": 0.5}]

    priced = capacity_lora_bake.lora_bake_bytes(
        str(ckpt_path), tensors, stack,
        _resolver({"a.safetensors": str(lora_a), "b.safetensors": str(lora_b)}))
    assert priced == 64 * 64 * 2   # once, not twice


def test_a_kohya_prefix_maps_onto_a_bare_checkpoint_key(tmp_path):
    """Chroma's Hyper LoRA: ``lora_unet_double_blocks_0_txt_attn_qkv`` over
    the checkpoint's own bare ``double_blocks.0.txt_attn.qkv.weight``."""
    ckpt_path = tmp_path / "chroma.safetensors"
    lora_path = tmp_path / "hyper.safetensors"
    _write_safetensors(ckpt_path, {"double_blocks.0.txt_attn.qkv.weight": ("BF16", [128, 128])})
    _write_safetensors(lora_path, {
        "lora_unet_double_blocks_0_txt_attn_qkv.lora_down.weight": ("BF16", [16, 128]),
        "lora_unet_double_blocks_0_txt_attn_qkv.lora_up.weight": ("BF16", [128, 16]),
        "lora_unet_double_blocks_0_txt_attn_qkv.alpha": ("BF16", []),
    })
    tensors = read_safetensors_header(str(ckpt_path)).tensors
    stack = [{"name": "hyper.safetensors", "strength": 1.0}]

    priced = capacity_lora_bake.lora_bake_bytes(
        str(ckpt_path), tensors, stack, _resolver({stack[0]["name"]: str(lora_path)}))
    assert priced == 128 * 128 * 2


def test_a_doubled_checkpoint_prefix_maps_onto_a_bare_lora_stem(tmp_path):
    """A ComfyUI-saved export's ``model.diffusion_model.`` prefix (Qwen,
    LTX) has nothing on the LoRA side to strip it against; the checkpoint's
    own key must be indexed under its stripped form too."""
    ckpt_path = tmp_path / "qwen.safetensors"
    lora_path = tmp_path / "lightning.safetensors"
    _write_safetensors(ckpt_path, {
        "model.diffusion_model.transformer_blocks.0.attn.to_k.weight": ("BF16", [64, 64]),
    })
    _write_safetensors(lora_path, {
        "transformer_blocks.0.attn.to_k.lora_down.weight": ("BF16", [8, 64]),
        "transformer_blocks.0.attn.to_k.lora_up.weight": ("BF16", [64, 8]),
    })
    tensors = read_safetensors_header(str(ckpt_path)).tensors
    stack = [{"name": "lightning.safetensors", "strength": 1.0}]

    priced = capacity_lora_bake.lora_bake_bytes(
        str(ckpt_path), tensors, stack, _resolver({stack[0]["name"]: str(lora_path)}))
    assert priced == 64 * 64 * 2


def test_separate_qkv_targets_map_onto_one_fused_key_priced_once(tmp_path):
    """z-image: LoRA ``attention.to_q/to_k/to_v`` over one fused
    ``attention.qkv.weight``; three targets, one stray, priced once."""
    ckpt_path = tmp_path / "zimage.safetensors"
    lora_path = tmp_path / "tarot.safetensors"
    _write_safetensors(ckpt_path, {
        "layers.0.attention.qkv.weight": ("BF16", [192, 64]),   # 3x fused
        "layers.0.attention.out.weight": ("BF16", [64, 64]),
    })
    _write_safetensors(lora_path, {
        "diffusion_model.layers.0.attention.to_q.lora_A.weight": ("BF16", [8, 64]),
        "diffusion_model.layers.0.attention.to_q.lora_B.weight": ("BF16", [64, 8]),
        "diffusion_model.layers.0.attention.to_k.lora_A.weight": ("BF16", [8, 64]),
        "diffusion_model.layers.0.attention.to_k.lora_B.weight": ("BF16", [64, 8]),
        "diffusion_model.layers.0.attention.to_v.lora_A.weight": ("BF16", [8, 64]),
        "diffusion_model.layers.0.attention.to_v.lora_B.weight": ("BF16", [64, 8]),
        "diffusion_model.layers.0.attention.to_out.0.lora_A.weight": ("BF16", [8, 64]),
        "diffusion_model.layers.0.attention.to_out.0.lora_B.weight": ("BF16", [64, 8]),
    })
    tensors = read_safetensors_header(str(ckpt_path)).tensors
    stack = [{"name": "tarot.safetensors", "strength": 1.0}]

    priced = capacity_lora_bake.lora_bake_bytes(
        str(ckpt_path), tensors, stack, _resolver({stack[0]["name"]: str(lora_path)}))
    # The fused qkv (192*64*2) once, plus the separate out (64*64*2).
    assert priced == 192 * 64 * 2 + 64 * 64 * 2


def test_a_projection_alias_with_no_fused_sibling_still_falls_back(tmp_path):
    """The ``to_q`` alias maps only onto a same-path fused ``qkv`` sibling, never
    an invented key: a diffusers ``to_q`` with no ``qkv`` sibling still misses."""
    ckpt_path = tmp_path / "m.safetensors"
    lora_path = tmp_path / "l.safetensors"
    _write_safetensors(ckpt_path, {"blocks.0.attn.proj.weight": ("BF16", [64, 64])})
    _write_safetensors(lora_path, {
        "transformer_blocks.0.attn.to_q.lora_A.weight": ("BF16", [8, 64]),
        "transformer_blocks.0.attn.to_q.lora_B.weight": ("BF16", [64, 8]),
    })
    tensors = read_safetensors_header(str(ckpt_path)).tensors
    stack = [{"name": "l.safetensors", "strength": 1.0}]

    priced = capacity_lora_bake.lora_bake_bytes(
        str(ckpt_path), tensors, stack, _resolver({stack[0]["name"]: str(lora_path)}))
    assert priced == os.path.getsize(ckpt_path)


def test_flux2_diffusers_to_comfy_fusion_is_not_modeled(tmp_path):
    """flux2's diffusers ``to_q``/``to_k``/``to_v`` fuse into a differently
    named weight (``img_attn.qkv``, not a same-path ``qkv`` sibling); this
    module must not guess across that rename, only across a literal fused
    sibling at the same parent path."""
    ckpt_path = tmp_path / "flux2.safetensors"
    lora_path = tmp_path / "turbo.safetensors"
    _write_safetensors(ckpt_path, {
        "double_blocks.0.img_attn.qkv.weight": ("BF16", [192, 64]),
    })
    _write_safetensors(lora_path, {
        "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": ("BF16", [8, 64]),
        "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": ("BF16", [64, 8]),
    })
    tensors = read_safetensors_header(str(ckpt_path)).tensors
    stack = [{"name": "turbo.safetensors", "strength": 1.0}]

    priced = capacity_lora_bake.lora_bake_bytes(
        str(ckpt_path), tensors, stack, _resolver({stack[0]["name"]: str(lora_path)}))
    assert priced == os.path.getsize(ckpt_path)


# A swap over a stock resident of the same checkpoint: under slab auto the store
# adopts it, so the bake term gets one file of credit (docs/VALIDATION.md,
# 2026-09-29 comfy-bump validation).
_KEY = repr(("krea2_raw_bf16.safetensors", (), (("krea2_turbo_lora_rank_64_bf16.safetensors", 0.8),), "bf16"))
_HELD_STOCK = {"residency": "stock", "request_key": repr(("krea2_raw_bf16.safetensors", (), (), "bf16"))}
_SWAP = {"would_load": "swap"}


def test_a_stock_swap_of_the_same_checkpoint_credits_one_file():
    assert capacity_lora_bake.swap_credit_bytes(_HELD_STOCK, _KEY, 24 << 30, _SWAP) == 24 << 30


def test_the_swap_credit_is_narrow():
    slab = dict(_HELD_STOCK, residency="slab")
    other = dict(_HELD_STOCK, request_key=repr(("flux2-dev.safetensors", (), (), "bf16")))
    requant = dict(_HELD_STOCK, request_key=repr(("krea2_raw_bf16.safetensors", (), (), "fp8")))
    for held, context in ((slab, _SWAP), (other, _SWAP), (requant, _SWAP), (None, _SWAP),
                          (_HELD_STOCK, {"would_load": "fresh"}), (_HELD_STOCK, None),
                          (dict(_HELD_STOCK, request_key="not a repr"), _SWAP)):
        assert capacity_lora_bake.swap_credit_bytes(held, _KEY, 24 << 30, context) == 0, (held, context)
    # An explicit slab_weights=on reloads the slot into slab rather than adopting the stock copy.
    assert capacity_lora_bake.swap_credit_bytes(_HELD_STOCK, _KEY, 24 << 30, _SWAP, True) == 0


def test_slab_load_fit_nets_the_credit_off_the_bake_term_only(tmp_path, monkeypatch):
    from dgx_monarch import capacity_fit, mesh_safety

    ckpt_path = tmp_path / "krea2_raw_bf16.safetensors"
    lora_path = tmp_path / "krea2_turbo_lora_rank_64_bf16.safetensors"
    _write_safetensors(ckpt_path, {k: (d, s) for k, d, s in KREA2_MAPPED_KEYS})
    _write_safetensors(lora_path, _lora_entries_for(KREA2_MAPPED_KEYS))
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 200 << 30)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    stack = [{"name": lora_path.name, "strength": 0.8}]
    resolve = _resolver({lora_path.name: str(lora_path)})

    full = capacity_fit.slab_load_fit(str(ckpt_path), {}, lora_stack=stack, resolve_lora_path=resolve)
    part = capacity_fit.slab_load_fit(str(ckpt_path), {}, lora_stack=stack, resolve_lora_path=resolve,
                                      lora_credit_bytes=4 << 30)
    over = capacity_fit.slab_load_fit(str(ckpt_path), {}, lora_stack=stack, resolve_lora_path=resolve,
                                      lora_credit_bytes=100 << 30)

    assert full.lora_bytes > 4 << 30
    assert part.lora_bytes == full.lora_bytes - (4 << 30)
    assert over.lora_bytes == 0
    assert (part.size_bytes, part.floor_bytes) == (full.size_bytes, full.floor_bytes)
