"""dgx-monarch ComfyUI entrypoint.

ComfyUI imports this entrypoint from custom_nodes/. Add src to sys.path and
export the node mappings from the installable dgx_monarch package.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dgx_monarch.kernels import int8_gemm  # noqa: E402
from dgx_monarch.mesh import install_fault_hook  # noqa: E402
from dgx_monarch.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

# The INT8 ConvRot GEMM kernel in this ComfyUI process (docs/MODELS.md):
# DGXM_INT8_GEMM selects "triton" (the default) or "cutlass". ComfyUI imported
# comfy-kitchen before this pack, so its CUDA backend is wrapped in place, and
# the import hook covers a process that imports it later. It never raises; a
# fallback to stock CUTLASS logs one WARNING.
int8_gemm.install_driver()

# Broadcast actor and host fault details to logs and ComfyUI websocket toasts
# (DESIGN.md §5.1); web/js/dgx_monarch.js displays the toast. Installs now only
# if Monarch is already loaded; otherwise prepare_driver installs the hook right
# after the driver loads Monarch, once its policy and OTLP endpoint are applied.
install_fault_hook()

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
