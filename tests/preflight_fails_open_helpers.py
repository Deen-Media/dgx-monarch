"""The fail-open contract of every driver preflight built on the `try: ...
path = folder_paths.get_full_path(...); if path is None: return; family =
...sniff_checkpoint(path)...; except Exception: return` shape in
nodes/render_preflight.py (preflight_krea2_reference_latents,
preflight_minimax_h3_topology, and any later preflight of that shape).

It pins two cases: sniff_checkpoint raising must not propagate, and a
checkpoint path that resolves to None must return before sniff_checkpoint's
family check.
"""
from __future__ import annotations

import sys
import types
from collections.abc import Callable
from typing import Any

from dgx_monarch.adapters import detect as detect_mod


def install_fake_folder_paths(monkeypatch, path: str | None) -> None:
    """Install a fake folder_paths module whose get_full_path returns path."""
    fp = types.ModuleType("folder_paths")
    fp.get_full_path = lambda _kind, _name: path
    monkeypatch.setitem(sys.modules, "folder_paths", fp)


def assert_sniff_checkpoint_raising_fails_open(
    monkeypatch,
    *,
    preflight: Callable[..., None],
    model: Any,
    refusing_args: tuple,
    default_path: str,
) -> None:
    """sniff_checkpoint raising must leave the preflight silent (no raise),
    even though `refusing_args` would otherwise refuse once family resolves."""
    install_fake_folder_paths(monkeypatch, default_path)

    def _boom(path):
        raise OSError("corrupt header")

    monkeypatch.setattr(detect_mod, "sniff_checkpoint", _boom)
    preflight(model, *refusing_args)


def assert_missing_checkpoint_file_fails_open(
    monkeypatch,
    *,
    preflight: Callable[..., None],
    model: Any,
    refusing_args: tuple,
    family: str,
) -> None:
    """get_full_path returning None must leave the preflight silent (no
    raise), even though sniff_checkpoint, if reached, would resolve the one
    family `refusing_args` would otherwise refuse."""
    install_fake_folder_paths(monkeypatch, None)
    monkeypatch.setattr(detect_mod, "sniff_checkpoint", lambda path: (family, "bf16"))
    preflight(model, *refusing_args)
