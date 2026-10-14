"""Identity-gate ceremony windows and claims for the render-progress tracker.

Windows are counted per thread, as first_render keeps its ceremony
thread-local: a Fleet or pipeline render on another thread is no proof leg.
The count is a depth, so an inner close never ends an outer window. The
outermost close leaves a one-shot claim for the render that starts next on
that thread, and a claim expires so a refused render cannot label a later
one a first render.

Stdlib only and stateless: each function returns the tracker's next state
dict, and the tracker calls them under its lock.
"""
from __future__ import annotations


def window_opened(state: dict, ident: int) -> dict:
    """One more open window on the calling thread."""
    windows = dict(state.get("_ceremony_windows", {}))
    windows[ident] = windows.get(ident, 0) + 1
    return {**state, "_ceremony_windows": windows}


def window_closed(state: dict, ident: int, now: float, claim_s: float) -> dict:
    """One window closed; the outermost close leaves the claim and drops expired ones."""
    windows = dict(state.get("_ceremony_windows", {}))
    depth = int(windows.get(ident, 0))
    next_state = {**state}
    if depth > 1:
        windows[ident] = depth - 1
    else:
        windows.pop(ident, None)
        if depth == 1:
            claims = {thread: at for thread, at in state.get("_ceremony_claim", {}).items()
                      if now - at < claim_s}
            claims[ident] = now
            next_state["_ceremony_claim"] = claims
    next_state["_ceremony_windows"] = windows
    return next_state


def forgotten(state: dict) -> dict:
    """Every window and claim on every thread dropped (tests and reset)."""
    return {**state, "_ceremony_windows": {}, "_ceremony_claim": {}}
