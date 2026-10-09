"""Runs the slab-hook canary wherever comfy is importable (the comfy-canary
CI job, or a local ComfyUI checkout at COMFY_DIR, default ../ComfyUI).
Elsewhere it skips; the daily comfy-canary workflow enforces it."""
import os
import sys

import pytest

from module_location_helpers import from_checkout as module_from_checkout


def test_slab_hooks_against_comfy():
    def is_comfy_module(name):
        return name == "folder_paths" or name == "comfy" or name.startswith(("comfy.", "comfy_"))

    preserved_modules = {
        name: module for name, module in sys.modules.items() if is_comfy_module(name)
    }
    for name in preserved_modules:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    comfy_dir = os.path.abspath(os.environ.get("COMFY_DIR", "../ComfyUI"))

    def from_checkout(module):
        # A real comfy import also adds nodes, node_helpers, execution,
        # latent_preview, server and more under no comfy prefix; every module
        # loaded from the checkout is removed with it.
        return module_from_checkout(module, comfy_dir)
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        # comfy.utils reaches comfy.cli_args during import, so select the CPU
        # device before pytest probes that module.
        sys.argv = ["pytest-slab-hook", "--cpu"]
        options = pytest.importorskip(
            "comfy.options", reason="no ComfyUI on sys.path (canary job covers it)")
        options.enable_args_parsing()
        pytest.importorskip("comfy.utils")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "canary"))
        from slab_hook_canary import main

        main()
    finally:
        sys.argv = original_argv
        # Classify first, then pop: a namespace package's path re-resolves
        # through its parent in sys.modules while it is being read.
        gone = [name for name, module in list(sys.modules.items())
                if is_comfy_module(name) or from_checkout(module)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved_modules)
        sys.path[:] = original_path
