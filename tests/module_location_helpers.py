"""Read module locations without triggering lazy imports.

Real-ComfyUI tests use these helpers to snapshot and restore sys.modules.
Read a module's namespace directly: ``getattr`` can invoke a lazy import hook
(such as transformers') and fail on an unrelated optional dependency.
"""
from __future__ import annotations

import os


def module_own(module: object, attribute: str) -> object:
    """A module's own attribute, or None; never triggers a module-level __getattr__."""
    try:
        return vars(module).get(attribute)
    except TypeError:
        return None


def module_location(module: object) -> str | None:
    """The file, or first package directory, a module came from; None for a stub."""
    location = module_own(module, "__file__")
    if not location:
        paths = module_own(module, "__path__")
        if paths is None:
            return None
        try:
            location = next(iter(paths), None)
        except (KeyError, TypeError):
            # A namespace path re-resolves through its parent in sys.modules
            # and raises once the parent is gone.
            return None
    return location if isinstance(location, str) else None


def from_checkout(module: object, comfy_dir: str) -> bool:
    """Does this module's file live under the ComfyUI checkout?

    A real comfy import also adds nodes, node_helpers, execution,
    latent_preview, server and more under no comfy prefix, and what the
    checkout provided has to leave with it.
    """
    location = module_location(module)
    return location is not None and os.path.abspath(location).startswith(comfy_dir + os.sep)
