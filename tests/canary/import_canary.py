#!/usr/bin/env python3
"""Import smoke test for the dgx-monarch worker env. Run on each box:

  PYTHONPATH= ~/monarch-env/bin/python ~/projects/dgx-monarch/tests/canary/import_canary.py

The default path exercises the GPUWorker import surface: ComfyUI loaders and
samplers, xFuser distributed state, Monarch actors, and dgx-monarch. Set
``DGXM_IMPORT_CPU=1`` in CPU-only CI. That mode keeps every other import but
checks only that xFuser is installed: under a CPU-only torch build, importing
xFuser raises NotImplementedError from ``xfuser.envs.get_device_version``.
"""
import os
import socket
import sys
from importlib.util import find_spec

_REPO_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
for p in (os.path.expanduser(os.environ.get("COMFYUI_DIR", "~/ComfyUI")), _REPO_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

_CPU_ONLY = os.environ.get("DGXM_IMPORT_CPU") == "1"

if _CPU_ONLY:
    sys.argv = ["import_test", "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
else:
    sys.argv = ["import_test"]  # comfy.cli_args parses argv at import time

import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import monarch.actor  # noqa: F401, E402
from comfy.patcher_extension import WrappersMP  # noqa: F401, E402

if _CPU_ONLY:
    assert find_spec("xfuser") is not None, "xfuser package is not installed"
    xfuser_status = "xfuser installed (accelerator import deferred on CPU)"
else:
    from xfuser_import_surface import load_gpu_surface

    load_gpu_surface()
    xfuser_status = "xfuser distributed and lazy ring import OK"

from dgx_monarch import TORCHMONARCH_PIN, __version__  # noqa: E402
from dgx_monarch.actor import GPUWorker  # noqa: F401, E402
from dgx_monarch.adapters import ADAPTERS  # noqa: E402
from dgx_monarch.topology import AUTO_TABLE  # noqa: E402

assert ADAPTERS, "adapter registry is empty"
adapter_types = [type(adapter).__name__ for adapter in ADAPTERS]
assert len(adapter_types) == len(set(adapter_types)), (
    f"duplicate adapter class registrations: {adapter_types}"
)
assert all(adapter.family for adapter in ADAPTERS), "every adapter needs a non-empty family label"
assert AUTO_TABLE, "auto table is empty"

print(
    f"import test ALL OK on {socket.gethostname()} "
    f"(dgx-monarch {__version__}, torchmonarch pin {TORCHMONARCH_PIN}; "
    f"{xfuser_status})"
)
