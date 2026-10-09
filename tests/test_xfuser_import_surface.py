"""The GPU import canary imports xFuser's distributed package, then its ring module, and lets a
ring import error propagate (docs/TROUBLESHOOTING.md #104)."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SURFACE = Path(__file__).parent / "canary" / "xfuser_import_surface.py"
spec = importlib.util.spec_from_file_location("xfuser_import_surface", SURFACE)
assert spec is not None and spec.loader is not None
surface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(surface)


def test_lazy_ring_import_is_exercised_without_group_or_model_setup():
    seen = []

    def import_module(name):
        seen.append(name)
        if name == "xfuser.core.long_ctx_attention.ring":
            return SimpleNamespace(xdit_ring_flash_attn_func=lambda: None)
        return SimpleNamespace()

    surface.load_gpu_surface(import_module)

    assert seen == ["xfuser.core.distributed", "xfuser.core.long_ctx_attention.ring"]


def test_lazy_ring_import_failure_propagates_from_the_gpu_canary():
    def import_module(name):
        if name == "xfuser.core.long_ctx_attention.ring":
            raise ImportError("cannot import update_npu_out")
        return SimpleNamespace()

    with pytest.raises(ImportError, match="update_npu_out"):
        surface.load_gpu_surface(import_module)
