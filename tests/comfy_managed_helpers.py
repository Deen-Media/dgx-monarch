"""Shared rig for the comfy-managed residency suites: process-state isolation,
the ladder call, a recording store and the source-reading helpers."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import residency_mode
from dgx_monarch.actor import comfy_dynamic, store_residency
from dgx_monarch.capacity_fit import StockFit
from dgx_monarch.refusal import parse_refusal_tag

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "dgx_monarch"

GIB = 1 << 30
FITS = StockFit(True, True, 20 * GIB, 90 * GIB)
DOES_NOT_FIT = StockFit(False, True, 62 * GIB, 47 * GIB)
STACK = [{"name": "a.safetensors", "strength": 1.0}]


@pytest.fixture(autouse=True)
def _isolated_process_state(monkeypatch):
    """The rung is process state. Never let one test's latch reach another."""
    monkeypatch.delenv(residency_mode.ENV_ACTIVE, raising=False)
    monkeypatch.setattr(comfy_dynamic, "_REQUESTED", None, raising=False)
    monkeypatch.setattr(comfy_dynamic, "_ACTIVE", False, raising=False)
    monkeypatch.setattr(comfy_dynamic, "_STAGE_B_DONE", None, raising=False)
    yield


@pytest.fixture
def rung_on(monkeypatch):
    """This process claims a successful bring-up, the way stage_b does."""
    monkeypatch.setenv(residency_mode.ENV_ACTIVE, "1")


def _resolve(**overrides):
    kwargs = {
        "path": "/models/h3_fl2va_bf16.safetensors",
        "unet_name": "h3_fl2va_bf16.safetensors",
        "model_options": {},
        "slab_weights": "auto",
        "slab_capable_path": True,
        "lora_low_rss": True,
        "fsdp_launch": False,
        "blocked_reason": "",
        "authoritative_slab_retry": False,
        "memoized_family": lambda _path: None,
        "vouched_families": frozenset({"krea2"}),
        "rescue_consent": None,
        "fit_probe": lambda _path, _options: FITS,
        "file_identity": lambda _path: "1:2:3:4:5",
        "compile_dit": lambda: False,
    }
    kwargs.update(overrides)
    return store_residency.resolve(**kwargs)


def _worker(*, topology=None, store=None):
    return SimpleNamespace(
        topology=topology or {}, world=1,
        store=store if store is not None else _RecordingStore())


class _RecordingStore:
    """Stands in for ModelStore: records that a load was attempted."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def ensure(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return ("patcher", "load")


def _tag(exc_or_text):
    return parse_refusal_tag(str(exc_or_text))


def _tree(relative: str) -> ast.Module:
    return ast.parse((SRC / relative).read_text())


def _func(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node  # type: ignore[return-value]
    raise AssertionError(f"{name} not found")
