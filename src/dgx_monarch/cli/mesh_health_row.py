"""Report the driver's mesh health in ``dgxm status`` and ``dgxm doctor``.

Attach readiness lives in the driver, so read each local driver's telemetry
through HTTP. The payload supplies its own explanation to keep diagnostics
consistent with the mesh layer.

Never attach a mesh for this check (docs/DESIGN.md section 5.8). An attach
reply timeout can leave Worker services unable to accept later attaches
(docs/TROUBLESHOOTING.md #2).
"""
from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping

from ..config import ClusterConfig
from ..mesh_evidence import mesh_block_dirty, mesh_block_unobserved
from . import comfy_ports, json_report, lifecycle

_TIMEOUT_S = 1.5
_ENTRY = "docs/TROUBLESHOOTING.md #65"
# Which responder to believe when several answer, ranked on `state` alone. An
# unreadable driver outranks one with no known state, and both outrank a
# healthy state: one instance reporting health is no observation of another.
# Ranks are total; `max` keeps the first tied candidate. A block dirty only
# by the non-state fields mesh_evidence weighs does not outrank a clean one.
_UNANSWERED = 5
_BLIND = 4
_PREFERENCE = {"poisoned": 7, "dirty": 6, "busy": 3, "ok": 2, "idle": 1}
# Distinguish an unreadable driver from an older driver that serves no mesh
# block: connection failures and missing telemetry need different remedies.
_UNREADABLE = {"dgxm_probe_unreadable": True}


def fetch_mesh_block(
    candidates: tuple[str, ...] | None = None,
    timeout_s: float = _TIMEOUT_S,
) -> Mapping | None:
    """The driver's mesh block; `None` when nothing listened on any candidate.

    `dgxm` runs on the driver box, so the driver is a local ComfyUI. Every
    one of them is asked, by discovery and not by convention: a box can run
    several at once, a custom launcher takes any port, and a bare responder on
    8188 must never mask the instance holding the dirty fleet. A driver that
    listened but could not be read comes back as `_UNREADABLE`, unless a dirty
    or poisoned block from another candidate outranks it.

    An empty mapping means a driver answered but reported no mesh view, which
    is what an older driver on this box looks like. Callers must not read that
    as health.
    """
    probes = comfy_ports.driver_candidates() if candidates is None else candidates
    seen: list[tuple[int, Mapping]] = []
    for candidate in probes:
        try:
            with urllib.request.urlopen(
                    f"http://{candidate}/dgxm/telemetry", timeout=timeout_s) as response:
                payload = json.load(response)
        except Exception as exc:
            if _probe_refused(exc):
                continue  # nothing listening: a normal state, not a view
            seen.append((_UNANSWERED, _UNREADABLE))  # a driver this CLI could not read
            continue
        block = payload.get("mesh") if isinstance(payload, Mapping) else None
        block = block if isinstance(block, Mapping) else {}
        seen.append((_PREFERENCE.get(_state(block), _BLIND), block))
    if not seen:
        return None
    return max(seen, key=lambda item: item[0])[1]


def _probe_refused(error: BaseException) -> bool:
    """Return whether a connection was refused, establishing no listener there.

    Discovery includes default ports even without a known process. Other failures
    mean unreadable state and must remain visible when ranking driver responses.
    """
    import urllib.error

    if isinstance(error, urllib.error.HTTPError):
        return False
    return isinstance(error, ConnectionRefusedError) or isinstance(
        getattr(error, "reason", None), ConnectionRefusedError)


def _unreadable(block: Mapping | None) -> bool:
    """Whether this view is `_UNREADABLE`, which no driver serves."""
    return block is not None and block.get("dgxm_probe_unreadable") is True


def _state(block: Mapping) -> str:
    return str(block.get("state") or "")


def _leases(block: Mapping) -> str:
    try:
        count = int(block.get("active_leases") or 0)
    except (TypeError, ValueError):
        count = 0
    return f"{count} active lease" if count == 1 else f"{count} active leases"


def _reason(block: Mapping) -> str:
    return str(block.get("reason") or "the driver refuses to attach to it")


def _remedy(block: Mapping) -> str:
    return str(
        block.get("remedy")
        or "reset the attached mesh; restart the worker service only if teardown is unconfirmed"
    )


def _busy(block: Mapping) -> str:
    return str(block.get("busy_phase") or "a worker operation")


def status_line(block: Mapping | None) -> str:
    """The text after `mesh: ` in `dgxm status`."""
    if block is None:
        return "no driver reachable (no mesh view)"
    if _unreadable(block):
        return "unreadable (a local driver gave no readable answer on its telemetry route)"
    if not block:
        return "no mesh view (this driver serves no mesh block)"
    state = _state(block)
    if state == "ok":
        return f"ok (cached, {_leases(block)})"
    if state == "idle":
        return "idle (no attached mesh; the next render session creates one)"
    if state == "busy":
        return f"busy ({_busy(block)} in flight); wait for it, nothing to fix"
    if state == "dirty":
        return f"DIRTY ({_reason(block)}); {_remedy(block)}. See {_ENTRY}"
    if state == "poisoned":
        # Report poisoned driver state separately: resetting the attached
        # mesh cannot clear a fault retained by the driver process.
        return f"POISONED ({_reason(block)}); {_remedy(block)}. See {_ENTRY}"
    if state == "unknown":
        return "unknown (the driver could not read its own mesh state)"
    return f"unrecognized (this dgxm does not know the state {state!r}; upgrade dgxm)"


def state_name(block: Mapping | None) -> str:
    """`status_line`'s subject as a token, for callers that do not read prose.

    The driver cannot name three cases for itself, so they get names here:
    nothing listened, a driver could not be read, and one served no mesh block.
    """
    if block is None:
        return "no-driver"
    if _unreadable(block):
        return "driver-unreadable"
    if not block:
        return "no-mesh-block"
    return _state(block) or "unknown"


def doctor_verdict(block: Mapping | None) -> tuple[str, str]:
    """Return ``(kind, detail)`` with kind ``ok``, ``warn`` or ``fail``.

    A dirty mesh is a definite failure: the driver records that the next attach
    will raise. Report the recovery command. A busy fleet passes because recycling
    during its load or gate cycle would interrupt active work.
    """
    if block is None:
        return "ok", ("no driver reachable; start ComfyUI and re-run doctor "
                      "for the mesh view")
    if _unreadable(block):
        return "warn", ("a local driver gave no readable answer on its telemetry "
                        "route, so this row is blind; check that ComfyUI is responsive "
                        f"before trusting the rows above. See {_ENTRY}")
    if not block:
        return "warn", ("this driver serves no mesh block; on an older driver "
                        "this row is blind, so upgrade it before trusting the "
                        "rows above")
    # Checked before the state names: a block can be dirty by fields `state`
    # does not carry, and mesh_evidence decides which.
    if mesh_block_dirty(block):
        return "fail", (f"{_reason(block)}. Every attach fails until this "
                        f"clears, whatever the Worker service rows say. "
                        f"{_remedy(block)}. See {_ENTRY}")
    state = _state(block)
    if state == "ok":
        return "ok", f"attached mesh ready, {_leases(block)}"
    if state == "idle":
        return "ok", "no attached mesh; the next render session creates one"
    if state == "busy":
        return "ok", (f"{_busy(block)} in flight; an attached mesh its own driver is "
                      "working on is not a dirty one")
    if state == "unknown":
        return "warn", ("the driver could not read its own mesh state, so "
                        f"this row is blind; re-run doctor, and see {_ENTRY} "
                        "if it stays unreadable")
    # A driver newer than this CLI: the fix is to upgrade dgxm, not the driver.
    return "warn", (f"this dgxm does not know the mesh state {state!r}, so "
                    "this row is blind; upgrade dgxm on this box")


def unobserved(block: Mapping | None) -> bool:
    """Whether `dgxm status` learned nothing about the attached mesh."""
    return mesh_block_unobserved(block)


def status_report(
    config: ClusterConfig, as_json: bool = False
) -> tuple[list[dict], Mapping | None]:
    """`dgxm status`: the per-host loop and port rows, then the mesh row.

    Composed here rather than inside `lifecycle.status()` because the rows
    above come from per-host probes and this one reads the local drivers.
    The host rows come back to the caller in both modes: the exit code is
    theirs to decide, and `--json` must not move it.
    """
    if not as_json:
        rows = lifecycle.status(config)
        block = fetch_mesh_block()
        print(f"  mesh: {status_line(block)}")
        return rows, block
    rows = json_report.quietly(lambda: lifecycle.status(config))
    block = fetch_mesh_block()
    json_report.emit(json_report.status_payload(
        rows, state_name(block), status_line(block), mesh_block=block))
    return rows, block


def status_with_mesh(config: ClusterConfig, as_json: bool = False) -> list[dict]:
    """Compatibility view returning the per-host rows only."""
    return status_report(config, as_json=as_json)[0]
