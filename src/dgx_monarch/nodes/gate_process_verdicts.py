"""Process-local identity-gate verdicts: one publication and its reads.

``nodes/auto_gate`` re-exports every name here, and ``nodes.common`` reaches
them through that module. The maps live in ``nodes/gate_process_state`` and
are reached through that module object, so the denial map's copy-on-write
publish stays visible to every later reader. ``gate_inconclusive`` resolves
against ``nodes.common`` per call: a module-top import would cycle, and a
test's patch on ``nodes.common`` must reach this module.
"""
from __future__ import annotations

from . import gate_process_state as _state
from .gate_token_lifetimes import trim_session


def record_process_gate_verdicts(
    tokens: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
    verdict: str, ceremony: dict | None = None,
) -> None:
    """Atomically publish one ceremony across normal/Fleet capabilities."""
    from .common import gate_inconclusive

    unique_tokens = tuple(dict.fromkeys(tokens))
    with _state._AUTO_GATE_LOCK:
        gate_inconclusive.remember_kind(unique_tokens, ceremony)
        denied = gate_inconclusive.session_denial_tokens(unique_tokens, ceremony)
        denials = dict(_state._PROCESS_GATE_DENIALS)
        if verdict in ("FAIL", "INCONCLUSIVE", "ERROR"):
            denials.update(dict.fromkeys(denied, verdict))
        else:
            for token in unique_tokens:
                denials.pop(token, None)
        # One copy-on-write store: an async interruption exposes all old or all
        # new revocations, never normal updated while Fleet remains stale.
        _state._PROCESS_GATE_DENIALS = denials
        for token in unique_tokens:
            if token in denied:
                _state._AUTO_GATE_SESSION[token] = verdict
            else:  # a sibling: its retest-guard denial lifts, its PASS is revoked
                denials.pop(token, None)
                _state._AUTO_GATE_SESSION.pop(token, None)
        trim_session(_state)  # the one capped store here (gate_token_lifetimes)


def process_gate_verdict(token: tuple[str, ...]) -> str | None:
    with _state._AUTO_GATE_LOCK:
        return process_gate_verdict_locked(token)


def process_gate_verdict_locked(token: tuple[str, ...]) -> str | None:
    """Read one verdict while the caller owns ``_AUTO_GATE_LOCK``."""
    return (
        _state._PROCESS_GATE_DENIALS.get(token)
        or _state._AUTO_GATE_SESSION.get(token)
    )
