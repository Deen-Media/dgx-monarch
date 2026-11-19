"""Stand-in ComfyUI checkouts whose prefix function has the release rule or master's.

Made-up minimal functions, not ComfyUI's code. The gate reads which rule a
checkout holds from that function's text, or asks the function itself once a
process has loaded it (dgx_monarch.stock_prefix_drift), so each stand-in is
also a module that answers.
"""
from __future__ import annotations

import importlib.util
import py_compile
import types
from pathlib import Path

from dgx_monarch.stock_prefix_drift import SENSENOVA_KEYS

_COUNT = "    return 'net.' if sum(key.startswith('net.') for key in state_dict) > 5 else 'model.'\n"
MASTER = f"def unet_prefix_from_state_dict(state_dict):\n{_COUNT}"
RELEASE = ("def unet_prefix_from_state_dict(state_dict):\n"
           f"    if {SENSENOVA_KEYS[0]!r} in state_dict and {SENSENOVA_KEYS[1]!r} in state_dict:\n"
           "        return ''\n" + _COUNT)


def write_checkout(root: Path, text: str) -> Path:
    """``root`` as a checkout whose ``comfy/model_detection.py`` is ``text``; rewrites it."""
    (root / "comfy").mkdir(parents=True, exist_ok=True)
    (root / "comfy" / "model_detection.py").write_text(text)
    return root


def compiled_only(root: Path, text: str) -> Path:
    """``root`` as a checkout that ships ``comfy/model_detection.pyc`` and no source."""
    source = write_checkout(root, text) / "comfy" / "model_detection.py"
    py_compile.compile(str(source), cfile=str(source.with_suffix(".pyc")), doraise=True)
    source.unlink()
    return root


def loaded(root: Path) -> types.ModuleType:
    """The module ``root``'s source makes, as a process that imported it holds it.

    It is not put in ``sys.modules``: a test sets it there through monkeypatch.
    """
    spec = importlib.util.spec_from_file_location(
        "comfy.model_detection", root / "comfy" / "model_detection.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package(*roots: Path) -> types.ModuleType:
    """A stand-in for the imported ``comfy`` namespace package over these checkouts, in order."""
    module = types.ModuleType("comfy")
    module.__path__ = [str(root / "comfy") for root in roots]
    return module
