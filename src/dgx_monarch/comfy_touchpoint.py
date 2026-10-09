"""The Touchpoint primitive shared by the ComfyUI compatibility manifests.

A leaf module so `comfy_surface.py` (the host manifest and its assertions) and
`comfy_forward_contracts.py` (the model-family forward contracts) can both
build touchpoints without an import cycle. `comfy_surface`, the public entry
point, re-exports it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Touchpoint:
    """One required attribute and the exact call shape dgx-monarch uses."""

    module: str
    attribute: str
    positional: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()
    callable: bool = False

    @property
    def path(self) -> str:
        return f"{self.module}.{self.attribute}"


def _call(
    module: str, attribute: str, *parameters: str, positional: tuple[str, ...] = ()
) -> Touchpoint:
    """Declare positional arity plus keyword names used by production calls."""
    return Touchpoint(module, attribute, positional, parameters, callable=True)


def _attr(module: str, attribute: str) -> Touchpoint:
    return Touchpoint(module, attribute)
