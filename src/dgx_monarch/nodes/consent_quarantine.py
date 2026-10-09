"""Check gate quarantine before applying capacity-rescue consent.

``gate_ledger.lookup_with_integrity`` decides render eligibility; this module
decides whether consent may load the same bytes into slab residency.
Context-scoped records expire for both readers when the gate protocol or
package version changes. Older unscoped records retain their quarantine.

A FAIL copied to an equivalent slab context applies only when the original
cross-residency comparison failed.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..log import get_logger

log = get_logger(__name__)


_QUARANTINE_REASON = (
    "an identity gate previously FAILED for these exact artifacts and quarantined slab "
    "residency for them, so no consent can load them slab-resident. A "
    "capability-scoped FAIL clears when you re-run the Identity Gate for this combination and it passes; "
    "an unscoped one predates those rows and no later run lifts it, so use a different artifact"
)
_RETESTING_REASON = (
    "the identity gate is RETESTING these exact artifacts, so its durable denial stands "
    "until a terminal verdict is recorded; consent cannot enable slab residency"
)
_INCONCLUSIVE_REASON = (
    "the identity gate was INCONCLUSIVE for these exact artifacts; only a later explicit "
    "PASS can clear an earlier quarantine, so consent cannot enable slab residency"
)
_CONTEXTLESS_TERMINAL_REASON = (
    "the newest matching identity-gate PASS has no capability context and is diagnostic-"
    "only, so it cannot clear quarantine or enable consent; run an exact contextual gate"
)
# Six things read as unknown, and each asks the operator for a different
# action, so each has its own sentence: a shared one would send an operator to
# check a sound ledger when the real problem is a row bound to another build.
# Every one fails closed.
_UNKNOWN_QUERY_REASON = (
    "the consent request did not name a complete artifact identity, so no identity-gate "
    "row could be looked up and consent cannot enable slab residency; re-open the card "
    "from the refusal that raised it"
)
_UNKNOWN_LEDGER_REASON = (
    "the identity-gate ledger could not be read on this host, so quarantine state is "
    "unknown and consent cannot enable slab residency; check the ComfyUI output "
    "directory and the gate ledger file"
)
_UNKNOWN_DAMAGE_REASON = (
    "the identity-gate ledger is damaged above the newest row matching these exact "
    "artifacts, so a torn line could have carried anything and no verdict for them can "
    "be read unambiguously; re-run the Identity Gate for this combination"
)
# The damage sentence above is only true where a torn line sits above a row
# this read matched. A lookup that passes no capability context reads every
# scope at once, and a lookup that matched no row has no newest row to be above,
# so both would state a falsehood a scoped read of the same ledger contradicts.
# The state is unchanged: a torn line could still have carried a FAIL in a scope
# with no surviving rows, and clearing is per scope.
_UNKNOWN_UNSCOPED_DAMAGE_REASON = (
    "the identity-gate ledger carries a damaged record and this read cannot tie it to a "
    "capability-scoped row for these exact artifacts, so a torn line could still have "
    "carried a verdict for them and none can be read unambiguously; re-run the Identity "
    "Gate for this combination, which writes the scoped row this read wants"
)
_UNKNOWN_UNBOUND_REASON = (
    "the newest identity-gate verdict for these exact artifacts is not authoritative "
    "for this gate protocol, package version or source build, so it cannot clear "
    "quarantine or enable consent; re-run the Identity Gate on this build"
)
_UNKNOWN_VERDICT_REASON = (
    "the newest identity-gate row for these exact artifacts carries a verdict this "
    "reader does not recognize, so consent cannot enable slab residency; re-run the "
    "Identity Gate for this combination"
)


@dataclass(frozen=True, slots=True)
class QuarantineQuery:
    """Hashable artifact/context identity for one consent decision."""

    combo_key: str
    current: str
    legacy: str
    legacy_complete: bool
    context: str | None
    valid: bool = True


@dataclass(frozen=True, slots=True)
class QuarantineDecision:
    """One fail-closed consent projection result."""

    state: str
    reason: str = ""


_CLEAR = QuarantineDecision("clear")
_FAILED = QuarantineDecision("fail", _QUARANTINE_REASON)
_RETESTING = QuarantineDecision("denied", _RETESTING_REASON)
_INCONCLUSIVE = QuarantineDecision("denied", _INCONCLUSIVE_REASON)
_DIAGNOSTIC_PASS = QuarantineDecision("denied", _CONTEXTLESS_TERMINAL_REASON)
_UNKNOWN_QUERY = QuarantineDecision("unknown", _UNKNOWN_QUERY_REASON)
_UNKNOWN_LEDGER = QuarantineDecision("unknown", _UNKNOWN_LEDGER_REASON)
_UNKNOWN_DAMAGE = QuarantineDecision("unknown", _UNKNOWN_DAMAGE_REASON)
_UNKNOWN_UNSCOPED_DAMAGE = QuarantineDecision(
    "unknown", _UNKNOWN_UNSCOPED_DAMAGE_REASON)
_UNKNOWN_UNBOUND = QuarantineDecision("unknown", _UNKNOWN_UNBOUND_REASON)
_UNKNOWN_VERDICT = QuarantineDecision("unknown", _UNKNOWN_VERDICT_REASON)


def quarantine_query(combo_key_value: object, artifacts: Any,
                     capability_context: object = None) -> QuarantineQuery:
    """Normalize one consent/gate join without losing the legacy digest alias."""
    try:
        from .. import consent_store

        if capability_context is None:
            context = None
        elif isinstance(capability_context, str):
            context = capability_context
        elif isinstance(capability_context, dict):
            context = consent_store.canonical_context(capability_context)
        else:
            raise TypeError("capability context is not canonicalizable")
        current = str(getattr(artifacts, "current", artifacts) or "")
        legacy = str(getattr(artifacts, "legacy", current) or "")
        complete = bool(getattr(artifacts, "legacy_complete", True))
        key = combo_key_value if isinstance(combo_key_value, str) else ""
        return QuarantineQuery(key, current, legacy, complete, context,
                               bool(key and current and legacy))
    except (TypeError, ValueError, RecursionError):
        return QuarantineQuery("", "", "", False, None, False)


def _scope_rows(query: QuarantineQuery, rows: Sequence[tuple[int, dict[str, Any]]],
                ) -> dict[str | None, list[tuple[int, dict[str, Any]]]]:
    scoped: dict[str | None, list[tuple[int, dict[str, Any]]]] = {}
    for line, entry in rows:
        verdict = entry.get("verdict")
        values = (entry.get("blocked_contexts", ()) if verdict == "RETESTING"
                  else (entry.get("capability_context"),))
        for value in values:
            scope = value if isinstance(value, str) else None
            if query.context is not None and scope not in (None, query.context):
                continue
            scoped.setdefault(scope, []).append((line, entry))
    return scoped


# The gate writes this on a row it copies onto the explicit-slab sibling of
# the context its ceremony ran in.
_SLAB_EQUIVALENCE_STAMP = "slab-mode-equivalence"


def _fail_still_speaks(entry: dict[str, Any], scope: str | None,
                       protocol: int, version: str) -> bool:
    """Whether one FAIL row still stands for the capability it names.

    Two kinds do not.

    A sibling row stamped by slab-mode equivalence whose ceremony never
    diverged on the cross-residency leg records nothing about slab residency:
    that leg is the only part of a ceremony slab residency runs. The writer
    (``gate_identity.equivalent_slab_mode_contexts``) applies the same rule;
    reading applies it too, for the rows written before the writer did.

    A context-scoped row binds to the release that wrote it, the same binding
    the gate's own scoped lookup requires, so a protocol or package move
    retires it for both readers at once. An unscoped row predates capability
    scoping and carries no binding to compare, so it stays.
    """
    if (entry.get("stamped") == _SLAB_EQUIVALENCE_STAMP
            and entry.get("cross_mode") != "FAIL"):
        return False
    if scope is None:
        return True
    return (entry.get("gate_protocol") == protocol
            and entry.get("dgx_monarch") == version)


def _scope_decision(rows: Sequence[tuple[int, dict[str, Any]]], damage: int,
                    scope: str | None, source: str | None, protocol: int,
                    version: str) -> QuarantineDecision:
    # Drop the FAILs `_fail_still_speaks` rejects before anything reads this scope, so
    # a retired row cannot block through the newest-row rules below either.
    rows = [(line, entry) for line, entry in rows
            if entry.get("verdict") != "FAIL"
            or _fail_still_speaks(entry, scope, protocol, version)]
    if not rows:
        # No matching row, so nothing here is above one.
        return _UNKNOWN_UNSCOPED_DAMAGE if damage else _CLEAR
    line, latest = rows[-1]
    if damage > line:
        return _UNKNOWN_DAMAGE
    fail_is_live = False
    for _line, entry in rows:
        if entry.get("verdict") == "FAIL":
            fail_is_live = True
        elif (entry.get("verdict") == "PASS" and scope is not None
              and source is not None and entry.get("gate_protocol") == protocol
              and entry.get("dgx_monarch") == version
              and entry.get("dgx_source") == source):
            fail_is_live = False
    if fail_is_live:
        return _FAILED
    verdict = latest.get("verdict")
    if scope is None and verdict == "PASS":
        return _DIAGNOSTIC_PASS
    authoritative = (scope is not None and source is not None
                     and latest.get("gate_protocol") == protocol
                     and latest.get("dgx_monarch") == version
                     and latest.get("dgx_source") == source)
    if verdict in ("PASS", "INCONCLUSIVE", "RETESTING") and not authoritative:
        return _UNKNOWN_UNBOUND
    if verdict == "RETESTING":
        return _RETESTING
    if verdict == "INCONCLUSIVE":
        return _INCONCLUSIVE
    if verdict != "PASS":
        return _UNKNOWN_VERDICT
    return _CLEAR


def quarantine_decisions(
    queries: Sequence[QuarantineQuery]) -> dict[QuarantineQuery, QuarantineDecision]:
    """Resolve context-scoped quarantine decisions from one integrity scan."""
    wanted = set(queries)
    decisions = {query: _UNKNOWN_QUERY for query in wanted if not query.valid}
    valid = {query for query in wanted if query.valid}
    if not valid:
        return decisions
    try:
        import folder_paths

        from .. import __version__, runtime_provenance
        from ..gate_ledger import GATE_PROTOCOL_VERSION, GateLedger

        entries, damage = GateLedger(
            folder_paths.get_output_directory()).entries_with_integrity()
        try:
            source = runtime_provenance.cached_dgx_source_manifest_sha256()
        except Exception as exc:
            source = None
            log.warning("identity-gate source identity could not be read (%r)", exc)
    except Exception as exc:
        log.warning("consent quarantine lookup failed closed (%r)", exc)
        decisions.update(dict.fromkeys(valid, _UNKNOWN_LEDGER))
        return decisions
    index: dict[tuple[str, str], set[QuarantineQuery]] = {}
    for query in valid:
        index.setdefault((query.combo_key, query.current), set()).add(query)
        index.setdefault((query.combo_key, query.legacy), set()).add(query)
    matched: dict[QuarantineQuery, list[tuple[int, dict[str, Any]]]] = {
        query: [] for query in valid}
    for line, entry in entries:
        entry_key, entry_artifacts = entry.get("key"), entry.get("artifacts")
        if not isinstance(entry_key, str) or not isinstance(entry_artifacts, str):
            continue
        for query in index.get((entry_key, entry_artifacts), ()):
            if (entry.get("artifacts") == query.legacy and not query.legacy_complete
                    and entry.get("verdict") != "FAIL"):
                continue
            matched[query].append((line, entry))
    for query, rows in matched.items():
        scopes = _scope_rows(query, rows)
        if query.context is not None:
            exact = _scope_decision(
                scopes.get(query.context, ()), damage, query.context, source,
                GATE_PROTOCOL_VERSION, __version__)
            global_rows = scopes.get(None, ())
            global_decision = _scope_decision(
                global_rows, damage, None, source, GATE_PROTOCOL_VERSION, __version__)
            decisions[query] = (global_decision
                                if global_rows and global_decision.state == "fail" else exact)
            continue
        scope_decisions = [
            _scope_decision(
                scope_rows, damage, scope, source, GATE_PROTOCOL_VERSION, __version__)
            for scope, scope_rows in scopes.items()
        ]
        # One cause has to be surfaced when several scopes read unknown at once.
        # Damage outranks the rest because it explains every other unknown and
        # is the one an operator can act on; otherwise the first unknown in
        # scope order wins, and scope order is ledger order. A scope that read
        # damage above its own newest row is picked before the unscoped
        # sentence, or an accurate cause is overwritten with a vaguer one.
        unknown = (next((item for item in scope_decisions
                         if item == _UNKNOWN_DAMAGE), None)
                   or (_UNKNOWN_UNSCOPED_DAMAGE if damage else None)
                   or next((item for item in scope_decisions
                            if item.state == "unknown"), None))
        decisions[query] = (next((item for item in scope_decisions
                                  if item.state == "fail"), None)
                            or next((item for item in scope_decisions
                                     if item.state == "denied"), None)
                            or unknown or _CLEAR)
    return decisions


def quarantine_decision(combo_key_value: object, artifacts: Any,
                        capability_context: object = None) -> QuarantineDecision:
    query = quarantine_query(combo_key_value, artifacts, capability_context)
    return quarantine_decisions((query,))[query]


def quarantine_decision_for_load(
    combo_key_value: object, artifacts: Any, capability_context: object = None,
) -> QuarantineDecision:
    """Return the fail-closed decision for one prospective slab-resident load.

    Two reads, in this order. The first is contextless, and every answer but
    ``unknown`` settles there: a current FAIL quarantines those bytes
    whatever scope it ran in, which keeps a quarantine outranking every consent
    (``docs/DESIGN.md`` section 5.9, invariant 2). Only an ``unknown`` is asked
    again under the load's own capability context, because it may describe a
    capability this load does not use; the second ledger scan runs on that path
    alone. The caller gets the typed decision: unreadable ledger evidence
    blocks admission, but it is not measured wrongness.
    """
    decision = quarantine_decision(combo_key_value, artifacts)
    if decision.state != "unknown" or capability_context is None:
        return decision
    return quarantine_decision(combo_key_value, artifacts, capability_context)


def quarantine_reason(combo_key_value: str, artifacts: Any,
                      capability_context: object = None) -> str:
    """Return the exact denial reason; unreadable evidence fails closed."""
    return quarantine_decision(
        combo_key_value, artifacts, capability_context).reason
