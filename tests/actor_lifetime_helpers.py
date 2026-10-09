"""Recording thread factory shared by actor-lifetime tests.

``_Recorder`` captures ``threading.Thread`` arguments without starting work.
"""
from __future__ import annotations

from types import SimpleNamespace


class _Recorder:
    """Record thread construction and start calls without running a thread."""

    def __init__(self) -> None:
        self.created: list[dict] = []

    def __call__(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(
            start=lambda: self.created[-1].setdefault("started", True),
            is_alive=lambda: True,
            **{k: v for k, v in kwargs.items() if k in {"name", "daemon"}},
        )
