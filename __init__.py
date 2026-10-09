"""dgx-monarch ComfyUI entrypoint.

ComfyUI imports this entrypoint from custom_nodes/. Add src to sys.path and
export the node mappings from the installable dgx_monarch package.
"""
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from dgx_monarch.mesh import install_fault_hook  # noqa: E402
from dgx_monarch.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

# Broadcast actor and host fault details to logs and ComfyUI websocket toasts
# (DESIGN.md §5.1); web/js/dgx_monarch.js displays the toast.
install_fault_hook()

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
