"""Import real ComfyUI modules for a CPU test, then put sys.modules back.

The comfy-canary job runs the tests that use it against comfy master and the
newest release (``COMFYUI_DIR`` names the clone); elsewhere they skip when no
ComfyUI is importable.
"""
from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Iterator

import pytest

from module_location_helpers import from_checkout

COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@contextlib.contextmanager
def real_comfy(label: str, *module_names: str) -> Iterator[tuple]:
    """Import real comfy modules on the CPU path, then put sys.modules back.

    Yields ``comfy.ops`` followed by each named module. Skips where ComfyUI is
    not importable; the comfy-canary job runs these files against comfy master.
    """
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before, argv_before = list(sys.path), sys.argv
    if os.path.isdir(COMFY_DIR) and COMFY_DIR not in sys.path:
        sys.path.insert(0, COMFY_DIR)
    try:
        sys.argv = [f"pytest-{label}", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            modules = tuple(pytest.importorskip(name) for name in module_names)
        finally:
            sys.argv = argv_before
        yield (ops, *modules)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, COMFY_DIR)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before
