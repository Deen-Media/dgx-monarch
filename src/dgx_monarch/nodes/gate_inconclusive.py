"""Classify and enforce INCONCLUSIVE identity-gate verdicts.

No-material skips only a later empty ceremony for the exact exercised context;
it grants no risky residency. A ceremony a measured class K guard stopped
grants nothing either and takes no lever. Every other incomplete or prevented
proof stays unproven and quarantines both residency levers for its own
combination, within the exemptions of ``consent_rescue.unproven_quarantine_levers``.
Classification is structured, token-bound evidence, never inferred from logs.
"""
from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from typing import Any

from .. import residency_mode
from ..gate_ledger import GateLedgerLookup
from ..log import get_logger

log = get_logger(__name__)

# Structured classification shared by reports and ledger detail.
RESULT_KEY = "inconclusive_kind"
KIND_NO_MATERIAL = "no_material"
KIND_UNPROVEN = "unproven"

# Rank-report marker consumed by cycle_had_nothing_to_swap.
CYCLE_KEY = "no_material"

# Dispatcher-only classification; persisted verdicts remain INCONCLUSIVE.
NO_MATERIAL_VERDICT = "INCONCLUSIVE_NO_MATERIAL"

# The other dispatcher-only verdict: a ceremony a measured class K guard
# stopped. Unproven like every abort, and it quarantines nothing.
KNOWN_WRONG_ABORT_VERDICT = "ERROR_KNOWN_WRONG_ABORT"

# The dispatch verdicts that take no lever off the graph, read by both render
# paths so one vocabulary decides it. ``None`` (no ceremony called for) and the
# last two disproved nothing about residency; PASS proved the path;
# DUAL_MODEL_STOCK ran no ceremony and renders stock.
NO_QUARANTINE_VERDICTS = frozenset({
    None, "PASS", "DUAL_MODEL_STOCK", NO_MATERIAL_VERDICT,
    KNOWN_WRONG_ABORT_VERDICT,
})

# First-use state, distinct from ledger verdicts to prevent trust comparisons.
GRANTED_STATE = "granted_no_material"

_LOCK = threading.Lock()
_NO_MATERIAL_TOKENS: dict[tuple[str, ...], bool] = {}  # Session lifetime; never evicted, matching denials.
# Session lifetime: class-K abort classifications, published and cleared
# with denial-map entries. Capping this store separately could turn a
# remembered abort into INCONCLUSIVE and quarantine unrelated residency.
_KNOWN_WRONG_ABORT_TOKENS: dict[tuple[str, ...], bool] = {}


def cycle_had_nothing_to_swap(slab_proof: Any) -> bool:
    """Require validated current all-rank no-swap evidence and one family."""
    cycle = list(getattr(slab_proof, "cycle", ()) or ())
    if (
        getattr(slab_proof, "complete", False) is not True
        or not cycle
        or getattr(slab_proof, "family", None) is None
    ):
        return False
    return all(
        isinstance(row, dict) and bool(row.get(CYCLE_KEY)) for row in cycle
    )


def classify(
    result: dict,
    slab_proof: Any,
    worker_args: dict | None,
    *,
    fsdp_scope: bool,
) -> str:
    """Classify INCONCLUSIVE from structural proof facts only."""
    if (
        fsdp_scope
        or bool(result.get("loras"))
        or result.get("cross_mode") is not None
        or getattr(slab_proof, "error", None)
        or getattr(slab_proof, "expected", False)
        or getattr(slab_proof, "active", False)
        # A Comfy-managed repeat is proof material, so INCONCLUSIVE is unproven.
        or residency_mode.requested(worker_args)
        or not cycle_had_nothing_to_swap(slab_proof)
    ):
        return KIND_UNPROVEN
    return KIND_NO_MATERIAL


def record_inconclusive(
    result: dict,
    slab_proof: Any,
    worker_args: dict | None,
    *,
    fsdp_scope: bool,
) -> str:
    """Stamp the structural kind and report its severity."""
    kind = classify(result, slab_proof, worker_args, fsdp_scope=fsdp_scope)
    result[RESULT_KEY] = kind
    origin, model = result.get("origin"), result.get("model")
    if kind == KIND_NO_MATERIAL:
        log.info("identity gate: nothing to gate on this graph (%s, %s): no LoRA stack "
                 "and slab residency is not engaged, so there is nothing to compare; "
                 "INCONCLUSIVE recorded, residency levers unchanged", origin, model)
        return kind
    reasons = "; ".join(result.get("inconclusive_reasons") or ())
    log.warning("identity gate INCONCLUSIVE (%s, %s): %s; the swaps did not take "
                "the lazy path, so there is nothing to compare",
                origin, model, reasons or "see swap_transitions")
    return kind


def is_no_material(result: Mapping[str, Any]) -> bool:
    """Return whether a finished result carries the no-material kind."""
    return result.get(RESULT_KEY) == KIND_NO_MATERIAL


def finished_no_material(ceremony: Mapping[str, Any]) -> bool:
    """Return whether a finished ceremony has the no-material outcome."""
    return str(ceremony.get("verdict")) == "INCONCLUSIVE" and is_no_material(ceremony)


def dispatch_verdict(result: Mapping[str, Any]) -> str:
    """Return the dispatch verdict for a newly completed ceremony."""
    if finished_no_material(result):
        return NO_MATERIAL_VERDICT
    return str(result.get("verdict"))


def known_wrong_abort_verdict(cause: BaseException) -> str | None:
    """The abort verdict a measured class K guard earns, or None.

    Dispatcher-only, like the no-material verdict above, and it grants no
    residency: authorization still finds no PASS and stamps the render stock.
    What it carries is that nothing was disproved about residency here, so the
    render path must not take a lever off the graph.
    """
    from .gate_quarantine_scope import measured_abort_guard

    if measured_abort_guard(cause) is None:
        return None
    return KNOWN_WRONG_ABORT_VERDICT


def settle_known_wrong_abort(
    token: tuple[str, ...] | None, cause: BaseException
) -> str | None:
    """Remember a class-K abort without changing its residency denial.

    The proof already cached INCONCLUSIVE for each tested context. Preserve that
    denial so ``authorize_normal_render`` continues to select stock on retries.
    Also record the accuracy classification so later reads do not interpret the
    same abort as a reason to quarantine the graph's residency settings.
    """
    verdict = known_wrong_abort_verdict(cause)
    if verdict is not None and token:
        remember_known_wrong_abort([tuple(token)])
    return verdict


def remember_known_wrong_abort(tokens: Iterable[tuple[str, ...]]) -> None:
    """Mark the tokens whose ceremony a measured class K guard stopped."""
    with _LOCK:
        for token in tokens:
            key = tuple(token)
            _KNOWN_WRONG_ABORT_TOKENS[key] = True
            # An abort is not a finished no-material ceremony. The two
            # classifications answer the same question, so one token never
            # carries both.
            _NO_MATERIAL_TOKENS.pop(key, None)


def forget_known_wrong_aborts() -> None:
    """Forget every marked abort (driver teardown, tests)."""
    with _LOCK:
        _KNOWN_WRONG_ABORT_TOKENS.clear()


def _exempt_token(ceremony: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Return only the directly exercised token eligible for the skip."""
    if not finished_no_material(ceremony):
        return None
    return tuple(ceremony.get("_gate_token") or ()) or None


def remember_kind(
    tokens: Iterable[tuple[str, ...]], ceremony: Mapping[str, Any] | None
) -> None:
    """Refresh classifications within the verdict publication lock.

    Any result other than finished no-material clears the token, preventing a
    fresh denial from pairing with stale skip evidence.
    """
    exempt = None if ceremony is None else _exempt_token(ceremony)
    with _LOCK:
        for token in tokens:
            key = tuple(token)
            if key == exempt:
                _NO_MATERIAL_TOKENS[key] = True
            else:
                _NO_MATERIAL_TOKENS.pop(key, None)
            # A published verdict supersedes an earlier abort's class: this
            # token has a fresh answer behind it now.
            _KNOWN_WRONG_ABORT_TOKENS.pop(key, None)


def session_denial_tokens(
    tokens: Iterable[tuple[str, ...]], ceremony: Mapping[str, Any] | None
) -> tuple[tuple[str, ...], ...]:
    """Return the process-cache tokens that this result must deny.

    A no-material result tested only its own context. Its Fleet and explicit-slab
    siblings retain ledger retest requirements, but their provisional process
    denials are removed so their first render can run the required proof. Other
    results deny or clear every token they published.
    """
    keys = tuple(tuple(token) for token in tokens)
    exempt = None if ceremony is None else _exempt_token(ceremony)
    if exempt is None:
        return keys
    return tuple(key for key in keys if key == exempt)


def cached_verdict(token: tuple[str, ...], verdict: str | None) -> str | None:
    """Apply remembered classification to a cached session verdict."""
    if verdict != "INCONCLUSIVE":
        return verdict
    key = tuple(token)
    with _LOCK:
        remembered = _NO_MATERIAL_TOKENS.get(key, False)
        known_wrong = _KNOWN_WRONG_ABORT_TOKENS.get(key, False)
    if remembered:
        return NO_MATERIAL_VERDICT
    return KNOWN_WRONG_ABORT_VERDICT if known_wrong else verdict


def ledger_detail(detail: dict, ceremony: Mapping[str, Any]) -> dict:
    """Classify only the exercised row; sibling rows are revocations."""
    if not finished_no_material(ceremony):
        return detail
    return {**detail, RESULT_KEY: KIND_NO_MATERIAL}


def row_grants_skip(lookup: GateLedgerLookup) -> bool:
    """Require an exact, safe, structurally classified INCONCLUSIVE row.

    Legacy rows, RETESTING, and newer ledger damage grant no skip.
    """
    entry = lookup.entry
    return (
        lookup.state == "inconclusive"
        and lookup.session_pass_safe
        and isinstance(entry, dict)
        and entry.get("verdict") == "INCONCLUSIVE"
        and entry.get(RESULT_KEY) == KIND_NO_MATERIAL
    )


def granted_state(lookup: GateLedgerLookup, fsdp_proof_required: bool) -> str:
    """Return first-use state; FSDP always requires exact PASS."""
    if fsdp_proof_required or not row_grants_skip(lookup):
        return lookup.state
    return GRANTED_STATE


def granted_verdict(state: str) -> str | None:
    """Return the dispatch verdict when no ceremony ran."""
    return NO_MATERIAL_VERDICT if state == GRANTED_STATE else None


def dispatch_grants_skip(
    lookup: GateLedgerLookup, token: tuple[str, ...], process_verdict: str | None
) -> bool:
    """Require durable skip evidence without a later process denial."""
    return row_grants_skip(lookup) and cached_verdict(token, process_verdict) in (
        None, NO_MATERIAL_VERDICT)
