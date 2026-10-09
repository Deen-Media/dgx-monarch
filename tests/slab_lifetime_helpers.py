"""The slab-lifetime isolation reset that store and slab suites call, from a
fixture or mid-test before a second load, to clear the retained-failed-load
bookkeeping in dgx_monarch.actor.slab_lifetime. Each global returns to its
value at import.
"""
from __future__ import annotations


def reset_slab_lifetime(monkeypatch):
    """Zero the retained-failed-load bookkeeping slab_lifetime owns."""
    from dgx_monarch.actor import slab_lifetime

    monkeypatch.setattr(slab_lifetime, "_RETAINED_FAILED_LOAD_SLABS", [])
    monkeypatch.setattr(slab_lifetime, "_RETAINED_FAILED_LOAD_RESOURCES", [])
    monkeypatch.setattr(slab_lifetime, "_FAILED_LOAD_CLEANUP_POISONED", False)
    monkeypatch.setattr(slab_lifetime, "_PROVISIONAL_CLEAR_WHEN_EMPTY", False)
