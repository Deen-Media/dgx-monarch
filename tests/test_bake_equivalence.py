"""Runs the bake-equivalence canary wherever comfy is importable: the daily
comfy-canary workflow, or locally with COMFY_DIR (default ../ComfyUI).
Elsewhere it skips."""
import os
import sys

import pytest

from module_location_helpers import from_checkout as module_from_checkout


def test_bake_equivalence_against_comfy():
    def is_comfy_module(name):
        return name == "folder_paths" or name == "comfy" or name.startswith(("comfy.", "comfy_"))

    preserved_modules = {
        name: module for name, module in sys.modules.items() if is_comfy_module(name)
    }
    # Import against one coherent real Comfy tree, never a partial stub an
    # earlier unit test installed.
    for name in preserved_modules:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    comfy_dir = os.path.abspath(os.environ.get("COMFY_DIR", "../ComfyUI"))

    def from_checkout(module):
        # A real comfy import also adds nodes, node_helpers, execution,
        # latent_preview, server and more under no comfy prefix; what the
        # checkout provided leaves with it.
        return module_from_checkout(module, comfy_dir)
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        # comfy.lora imports comfy.cli_args transitively. Arm Comfy's parser
        # and present --cpu before that first import: the canary's main does
        # the same, too late for the importorskip calls below.
        sys.argv = ["pytest-bake-equivalence", "--cpu"]
        options = pytest.importorskip(
            "comfy.options", reason="no ComfyUI on sys.path (canary job covers it)")
        options.enable_args_parsing()
        pytest.importorskip("comfy.lora")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "canary"))
        from bake_equivalence_canary import main

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
