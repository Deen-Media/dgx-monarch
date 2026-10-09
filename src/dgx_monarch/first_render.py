"""Publish first-render costs to the log, telemetry, and browser.

A new model stack may pay for NCCL setup, cold loading, and two to four identity
proof renders before its requested render. The measured reference workflow took
about 250 seconds cold and about 45 warm (docs/TROUBLESHOOTING.md #61).

Headless processes keep log and telemetry output but skip the optional websocket
broadcast. Every publication is best effort and cannot fail a render.
``once_key`` deduplicates costs paid once per driver process.
"""
from __future__ import annotations

import threading

from .log import get_logger

log = get_logger(__name__)

EVENT_NAME = "dgx-monarch.notice"
# Toast title for first-render costs; other callers pass their own `summary`.
DEFAULT_SUMMARY = "DGX Monarch: first render"
# Process lifetime, keyed by `once_key`, never evicted: each cost is paid once
# per driver process; used keys remain recorded until `reset`.
_ANNOUNCED: set[str] = set()
_LOCK = threading.Lock()
# Ceremony lifetime, per thread: a Fleet or pipeline render on another thread
# cannot count toward this thread's proof legs, and the tracker keys its mirror
# by the same thread. A flag suffices because the auto-gate re-entry guard makes
# nested windows unreachable. Call `gate_started` only inside the try whose
# finally calls `gate_finished`: if the window opens earlier, an exception
# strands the flag and later renders on that thread count as proof renders.
_CEREMONY = threading.local()


def _claim(once_key: str | None) -> bool:
    """Spend one dedup key. True when this caller owns the announcement."""
    if once_key is None:
        return True
    with _LOCK:
        if once_key in _ANNOUNCED:
            return False
        _ANNOUNCED.add(once_key)
        return True


def _tell_tracker(window: str) -> None:
    """Mirror this thread's ceremony window onto the render tracker.

    ``open`` starts it, ``close`` leaves a one-shot claim for the render the
    gate let through, and ``forget`` drops both. Telemetry decides nothing.
    """
    try:
        from .telemetry import render_progress

        if window == "open":
            render_progress.note_ceremony()
        elif window == "close":
            render_progress.close_ceremony()
        else:
            render_progress.forget_ceremony()
    except Exception:
        pass  # telemetry must never fail the render


def reset() -> None:
    """Forget every dedup key and close any open ceremony (tests only)."""
    with _LOCK:
        _ANNOUNCED.clear()
    _CEREMONY.active = False
    _CEREMONY.proofs = 0
    _tell_tracker("forget")


def notice(phase: str, message: str, note: str, *,
           once_key: str | None = None, toast: bool = True,
           severity: str = "info", sticky: bool = False,
           summary: str | None = None) -> bool:
    """Publish one phase notice without raising into the render.

    ``message`` feeds the log and toast; compact ``note`` feeds telemetry.
    ``severity`` styles the toast, ``sticky`` keeps a long phase visible, and
    ``summary`` overrides its title. Returns False only for an already-used
    ``once_key``.
    """
    if not _claim(once_key):
        return False
    # The log is the contract with headless verification, so it stays outside
    # the try blocks that swallow a browser or ring failure.
    log.info("%s", message)
    try:
        from .telemetry import emit

        emit("notice", phase=phase, note=note)
    except Exception:
        pass  # telemetry must never fail the render
    try:
        from server import PromptServer

        PromptServer.instance.send_sync(
            EVENT_NAME,
            {"phase": phase, "message": message, "note": note, "toast": toast,
             "severity": severity, "sticky": sticky,
             "summary": summary or DEFAULT_SUMMARY},
        )
    except Exception:
        pass  # headless or non-ComfyUI driver
    return True


def nccl_deferred() -> None:
    """Say what an `auto` topology defers onto the first render."""
    notice(
        "deferred",
        "topology auto: NCCL bring-up is deferred to the first render, which "
        "pays that one-time cost before it samples",
        "NCCL bring-up on first render",
        once_key="deferred:nccl",
    )


def load_deferred(endpoint: str, reason: str = "auto topology") -> None:
    """Say what a deferred loader node costs the first render.

    ``reason`` names why the load deferred, so the journal line names the right
    topology: ``auto topology`` on the auto path, ``explicit cfg2 dm-cfg2`` when
    the dm-cfg2 per-rank load defers under the explicit cfg2 preset.
    """
    notice(
        "deferred",
        f"{endpoint} is deferred to the first render ({reason}), which pays "
        "a one-time cold weight load from disk",
        f"cold {endpoint} on first render",
        once_key=f"deferred:{endpoint}",
    )


# Read-only after import.
_GATE_REASON = {
    "unknown": "not yet gated",
    "inconclusive": "still inconclusive",
}
_GATE_STALE = ("stale or not vouched (a model or LoRA file, the dgx-monarch release or "
               "source, or the residency, worker, attention, cluster or topology "
               "settings changed; the ComfyUI commit changed or cannot be read; or "
               "the gate ledger holds a damaged line or a non-PASS row for this "
               "combination)")


def gate_started(state: str) -> None:
    """Open the identity-check window and announce its additional render steps."""
    _CEREMONY.active = True
    _CEREMONY.proofs = 0
    _tell_tracker("open")
    notice(
        "gate",
        "auto-gate: this model+stack combination is "
        f"{_GATE_REASON.get(state, _GATE_STALE)}, so the identity gate runs "
        "once now: 2-4 short proof renders (3 or 4 under slab residency), then "
        "your render. Later renders of this combination skip the gate. To turn "
        "the gate off, set auto_gate=off on the Init node",
        "identity gate running (2-4 proof renders)",
        sticky=True,
    )


def note_proof_render() -> int | None:
    """Count one ceremony proof render; None when no ceremony is open."""
    if not getattr(_CEREMONY, "active", False):
        return None
    index = int(getattr(_CEREMONY, "proofs", 0)) + 1
    _CEREMONY.proofs = index
    notice(
        "gate_proof",
        f"auto-gate: identity gate proof render {index} of 2-4; your own "
        "render has not started yet",
        f"proof render {index}",
        toast=False,
    )
    return index


def cross_residency_check() -> None:
    """Name the ceremony leg that renders the same seed under stock residency.

    That leg also counts in ``note_proof_render``; its own phase tells it apart
    from the candidate proof legs.
    """
    notice(
        "gate_cross",
        "auto-gate: cross-residency check. The same seed renders again under "
        "stock residency so the gate can compare the two latents bit for bit; "
        "your own render has not started yet",
        "cross-residency check",
        toast=False,
    )


_GATE_DONE_VOUCHED = ("Your render starts now and is the only cost left; the "
                      "next render of this combination skips the gate and the "
                      "cold load")
# Only PASS authorizes optimized residency. The caller chooses quarantine,
# stock fallback, or refusal for every other outcome.
_GATE_DONE_DENIED = ("The optimized residency paths stay off for this "
                     "combination; the driver log says whether the render "
                     "falls back to stock residency or is refused, and "
                     "docs/TROUBLESHOOTING.md #61 explains a verdict other "
                     "than PASS")


def gate_finished(result: dict | None) -> None:
    """Close the identity-check window and report elapsed time without raising."""
    proofs = int(getattr(_CEREMONY, "proofs", 0))
    _CEREMONY.active = False
    _CEREMONY.proofs = 0
    _tell_tracker("close")
    if not isinstance(result, dict):
        # With no verdict, the caller decides stock fallback or refusal. The
        # notice names both and leaves failure detail in the driver log.
        notice(
            "gate_done",
            "auto-gate: the identity gate did not finish. The optimized "
            "residency paths stay off for this combination. The driver log has "
            "the error and says whether your render continues on stock "
            "residency or is refused",
            "gate did not finish, optimized residency off",
            severity="warn",
        )
        return
    verdict = str(result.get("verdict") or "?")
    wall = result.get("wall_s")
    spent = (f" in {float(wall):.1f}s"
             if isinstance(wall, int | float) and not isinstance(wall, bool)
             else "")
    counted = f" over {proofs} proof renders" if proofs else ""
    passed = verdict == "PASS"
    notice(
        "gate_done",
        f"auto-gate: identity gate {verdict}{spent}{counted}. "
        + (_GATE_DONE_VOUCHED if passed else _GATE_DONE_DENIED),
        (f"gate {verdict}{spent}, render starting" if passed
         else f"gate {verdict}{spent}, optimized residency off"),
        severity="info" if passed else "warn",
    )
