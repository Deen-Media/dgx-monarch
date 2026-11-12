"""dgx-monarch ComfyUI entrypoint.

ComfyUI imports this entrypoint from custom_nodes/. Add src to sys.path and
export the node mappings from the installable dgx_monarch package.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dgx_monarch import demo_inputs, driver_uma_memory, fp8_compute, vae_decode_relief, vae_encode_item  # noqa: E402
from dgx_monarch.kernels import int8_gemm  # noqa: E402
from dgx_monarch.mesh import install_fault_hook  # noqa: E402
from dgx_monarch.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

# The INT8 ConvRot GEMM kernel in this ComfyUI process (docs/MODELS.md):
# DGXM_INT8_GEMM selects "triton" (the default) or "cutlass". ComfyUI imported
# comfy-kitchen before this pack, so its CUDA backend is wrapped in place, and
# the import hook covers a process that imports it later. It never raises; a
# fallback to stock CUTLASS logs one WARNING.
int8_gemm.install_driver()

# One `FP8 compute:` line per model ComfyUI builds in this process, naming the
# operations class it picked (docs/MODELS.md, FP8 compute paths). It only logs.
fp8_compute.install_driver()

# On a GPU that shares the host's memory (a DGX Spark), ComfyUI's free-memory
# reading follows MemFree and its VAE batch choices swing with the
# page cache. This process then reads MemAvailable less a floor
# (src/dgx_monarch/driver_uma_memory.py). A discrete GPU is left alone, it
# never raises, and DGXM_DISABLE_DRIVER_UMA_MEMORY=1 turns it off.
driver_uma_memory.install_driver()

# On the same GPUs, a VAE encode of several images runs one image a call, so
# its result does not depend on free memory (src/dgx_monarch/vae_encode_item.py).
# Decode is not touched. It never raises, it steps aside if ComfyUI's call
# changes, and DGXM_DISABLE_VAE_ENCODE_ITEM=1 turns it off.
vae_encode_item.install_driver()

# On the same GPUs, before a VAE decode that ComfyUI prices above its free
# memory, this process has ComfyUI free what the decode needs from the models
# it holds idle, then runs ComfyUI's decode unchanged
# (src/dgx_monarch/vae_decode_relief.py). It never raises, it steps aside if a
# call it makes fails, and DGXM_DISABLE_VAE_DECODE_RELIEF=1 turns it off.
vae_decode_relief.install_driver()

# Copy the templates' demo images, clips and music into ComfyUI's input folder
# where it lacks them, never replacing a file (src/dgx_monarch/demo_inputs.py).
demo_inputs.place_demo_inputs()

# Broadcast actor and host fault details to logs and ComfyUI websocket toasts
# (DESIGN.md §5.1); web/js/dgx_monarch.js displays the toast. Installs now only
# if Monarch is already loaded; otherwise prepare_driver installs the hook right
# after the driver loads Monarch, once its policy and OTLP endpoint are applied.
install_fault_hook()

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
