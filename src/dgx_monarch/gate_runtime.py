"""The torch build a Gate proof ran under, and the notice that names a torch change.

Every capability context carries ``runtime``, the driver's torch and CUDA
build, so a PASS proven under one torch never vouches for another: the
context string differs and the lookup reads stale. A row written before this
field existed has none, so it reads stale too, and the reader never parses it
to decide trust. Only the first-use notice below parses old rows, to say why
the gate runs again.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

RUNTIME_FIELD = "runtime"


def runtime_build_identity() -> dict[str, str]:
    """The torch and CUDA build this driver imported.

    Local actors start from the driver's interpreter, and doctor's
    ``torch version match`` row holds every Worker host to the driver's torch
    (DESIGN.md section 6.3), so this names the build every rank of a proof ran.
    The NCCL library is left out: torch reports only its build-time NCCL, which
    the torch version already fixes, and the library an actor loads is unknown
    to the driver before dispatch; doctor's ``NCCL library`` row compares it.
    """
    import torch

    version = getattr(torch, "version", None)
    return {"torch": str(getattr(torch, "__version__", "unknown")),
            "cuda": str(getattr(version, "cuda", None) or "none")}


def _without_runtime(context: Any) -> Any:
    if not isinstance(context, dict):
        return None
    return {name: value for name, value in context.items() if name != RUNTIME_FIELD}


def _label(runtime: Any) -> str:
    if not isinstance(runtime, dict):
        return "no recorded torch build"
    return f"torch {runtime.get('torch', 'unknown')} with CUDA {runtime.get('cuda', 'unknown')}"


def torch_change(ledger_dir: str | Callable[[], str], token: tuple[str, ...]) -> str | None:
    """Say why a stale combination proves again when only the torch build moved.

    ``ledger_dir`` is the directory or a function that returns it, called
    inside the guard so a lookup failure cannot fail the ceremony. ``token``
    is the tuple ``gate_ledger.gate_verdict_token`` builds. Only a
    PASS row equal to it in key, artifact digest, ComfyUI commit, package
    source, protocol, package version and every context field but ``runtime``
    qualifies, so the notice never blames torch for a change it did not cause.
    An unknown ComfyUI commit, an unreadable row or ledger, and any other case
    return None, and the notice keeps its general list of causes. The caller
    asks only when the lookup reads stale.
    """
    try:
        key, digest, commit, wanted, source, protocol, version = token
        if commit == "unknown":  # the lookup reads stale for that alone
            return None
        current = json.loads(wanted)
        directory = ledger_dir() if callable(ledger_dir) else ledger_dir
        from .gate_ledger import GateLedger

        for entry in reversed(GateLedger(directory).entries()):
            if (entry.get("verdict") != "PASS" or entry.get("key") != key
                    or entry.get("artifacts") != digest or entry.get("comfy") != commit
                    or entry.get("dgx_source") != source or entry.get("dgx_monarch") != version
                    or str(entry.get("gate_protocol")) != protocol):
                continue
            try:
                recorded = json.loads(entry.get("capability_context") or "null")
            except ValueError:
                continue
            if (_without_runtime(recorded) == _without_runtime(current)
                    and recorded.get(RUNTIME_FIELD) != current.get(RUNTIME_FIELD)):
                return (f"proven under {_label(recorded.get(RUNTIME_FIELD))}, not under "
                        f"this driver's {_label(current.get(RUNTIME_FIELD))}")
    except Exception:
        return None
    return None
