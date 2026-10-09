"""Fake comfy.model_base class trees for adapter dispatch tests."""
from __future__ import annotations

import sys
import types


def install_fake_model_base(monkeypatch, bases):
    """Build and install a fake comfy.model_base with the given class tree.

    `bases` maps each class name to its parent's name in the same mapping, or
    to None for a root that subclasses object. A parent must come before its
    children in dict order. A parent lets a test reproduce a real comfy
    subclass relationship, or build an unlisted subclass for matches() to
    accept or reject. Returns the fake comfy.model_base module, with
    every class set as an attribute by name.
    """
    mb = types.ModuleType("comfy.model_base")
    built: dict[str, type] = {}
    for name, parent in bases.items():
        cls = type(name, (built[parent] if parent else object,), {})
        built[name] = cls
        setattr(mb, name, cls)
    comfy_mod = types.ModuleType("comfy")
    comfy_mod.model_base = mb
    monkeypatch.setitem(sys.modules, "comfy", comfy_mod)
    monkeypatch.setitem(sys.modules, "comfy.model_base", mb)
    return mb
