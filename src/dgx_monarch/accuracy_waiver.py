"""Class-K accuracy waivers and their per-render evidence.

These waivers cover guards with measured divergence, not merely unproven math.
Refusal remains the default. A waiver must be stamped by the driver on the exact
dispatch; class K is never eligible for automatic rescue. Workers do not read
the consent store, and version skew fails closed to refusal. Each used waiver
is stamped on the result and recorded as permanent audit evidence.

This shared leaf imports only the standard library and consent vocabulary.
"""
from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import consent_descriptor
from .consent_kinds import KIND_SPECS
from .gate_audit_vocab import CONSENT_SOURCES, KIND_GUARD_PREFIXES, check_guard
from .log import get_logger
from .refusal import GUARDS, PanelAction, RefusalClass

log = get_logger(__name__)

# The driver stamps this private key after re-resolving consent for each
# dispatch, preventing stale grants from surviving resubmission.
REQUEST_KEY = "_dgxm_accuracy_waivers"
# Per-rank result key for waivers that fired.
RESULT_KEY = "waived"
# Render metadata key published by the driver.
STAMPED_RESULT_KEY = "_dgxm_rendered_under_waiver"

def known_wrong_guards(guards: Mapping[str, Any]) -> dict[str, str]:
    """Map each wired, waivable known-wrong guard to its consent kind.

    Deriving this from the shared guard vocabulary prevents disagreement between
    refusal validation and waiver execution.
    """
    return {name: spec.consent_kind for name, spec in guards.items()
            if spec.refusal_class is RefusalClass.KNOWN_WRONG and spec.waivable_now}


KNOWN_WRONG_GUARDS: dict[str, str] = known_wrong_guards(GUARDS)
KNOWN_WRONG_KINDS: tuple[str, ...] = tuple(sorted(set(KNOWN_WRONG_GUARDS.values())))


def _assert_never_auto() -> None:
    """Class K is never auto-eligible. Checked at import, not by convention."""
    for kind in KNOWN_WRONG_KINDS:
        spec = KIND_SPECS.get(kind)
        if spec is None:
            raise ValueError(f"guard vocabulary names unknown consent kind {kind!r}")
        if spec.auto_eligible:
            raise ValueError(
                f"{kind} is class K and must never be auto-eligible: there is no "
                "accuracy analog of the standing capacity toggle")


_assert_never_auto()


def stamp_text(kind: str) -> str | None:
    """Return the shared rendered-under-waiver text for a class-K kind."""
    spec = KIND_SPECS.get(kind)
    if spec is None or spec.refusal_class != "K":
        return None
    return f"rendered-under-waiver: {spec.kind} ({spec.measured})"


def env_tokens(raw: object) -> set[str]:
    """The comma-separated names one class-K environment value carries."""
    return {piece.strip() for piece in str(raw).split(",") if piece.strip()}


def env_spellings(kind: str) -> frozenset[str]:
    """The two values the shared variable accepts for one wired class-K kind."""
    spec = KIND_SPECS[kind]
    return frozenset({spec.default_guard, spec.kind})


# Derived, not listed: a third wired class-K kind arms this without an edit.
ENV_ACCEPTED: frozenset[str] = frozenset().union(
    *(env_spellings(kind) for kind in KNOWN_WRONG_KINDS))


def env_grants_kind(spec: Any, raw: object) -> bool:
    """Whether the value names this kind. A scoped ledger guard id does not."""
    return bool(env_tokens(raw) & {spec.default_guard, spec.kind})


def _accepted_phrase() -> str:
    return "; ".join(f"{KIND_SPECS[kind].default_guard!r} or {kind!r}"
                     for kind in KNOWN_WRONG_KINDS)


def _did_you_mean(unread: list[str]) -> str:
    """Name the unscoped spelling behind a guard-prefixed token, if there is one.

    Read the prefixes off the wired kinds, the same source ``ENV_ACCEPTED``
    uses. The audit vocabulary also carries retired prefixes, and suggesting
    one of those would offer another value this build cannot read.
    """
    prefixes = {KIND_SPECS[kind].default_guard for kind in KNOWN_WRONG_KINDS}
    hits = sorted({name.split(":", 1)[0] for name in unread
                   if name.split(":", 1)[0] in prefixes})
    return f" Did you mean {', '.join(repr(hit) for hit in hits)}?" if hits else ""


# One warning per unreadable set of names, not one per kind and not one per
# dispatch.
_ENV_UNREADABLE: set[str] = set()


def warn_unreadable_env(spec: Any, raw: str) -> None:
    """Warn once about the parts of a class-K value this build cannot read.

    One variable carries every wired class-K kind, so a value naming one kind
    is not unreadable to the others: only tokens outside the whole vocabulary
    are. It takes an unscoped guard or a kind, never a truthy flag and never a
    family-scoped ledger guard.
    """
    unread = sorted(env_tokens(raw) - ENV_ACCEPTED)
    key = ",".join(unread)
    if not unread or key in _ENV_UNREADABLE:
        return
    _ENV_UNREADABLE.add(key)
    log.warning("%s: %s names nothing this build can waive, so the guard still "
                "refuses. It takes %s; it does not take the scoped guard ids "
                "from the ledger's waiver rows.%s",
                spec.env_var, ", ".join(repr(name) for name in unread),
                _accepted_phrase(), _did_you_mean(unread))


def panel_action(guard: str) -> PanelAction:
    """Return the panel action and explicit headless value for a guard.

    Class K requires an unscoped guard name rather than a truthy flag. The
    registry supplies the shared button label.
    """
    kind = KNOWN_WRONG_GUARDS[guard]
    spec = KIND_SPECS[kind]
    return PanelAction(spec.primary_label, spec.env_var,
                       env_value=KIND_GUARD_PREFIXES[kind])


# Match ``Topology.describe`` without importing topology and its adapter graph
# into this shared leaf. A test keeps both spellings aligned.
_DEGREE_LABELS: tuple[tuple[str, str], ...] = (
    ("ulysses", "uly"), ("ring", "ring"), ("cfg", "cfg"), ("dp", "dp"))


def topology_label(topology: Mapping[str, Any] | None) -> str:
    """Return the operator-facing topology label used to scope a waiver.

    Driver and worker derive it from the same topology mapping, keeping cards,
    memos, and audit rows aligned.
    """
    values = dict(topology or {})

    def degree(name: str) -> int:
        try:
            return int(values.get(name, 1) or 1)
        except (TypeError, ValueError):
            return 1

    parts = [f"{label}{degree(name)}" for name, label in _DEGREE_LABELS
             if degree(name) > 1]
    if values.get("fsdp"):
        parts.append("fsdp")
    return "+".join(parts) or "single"


def waiver_context(*, combo_key: str, topology: Mapping[str, Any] | None,
                   world: object) -> dict[str, str]:
    """Return the combination, topology, and world scoped by a class-K waiver.

    This is narrower than the trust context because measured divergence is a
    property of these fields and should not be invalidated by unrelated context.
    """
    size = world if isinstance(world, int) and not isinstance(world, bool) else 0
    return {"combo_key": str(combo_key), "topology": topology_label(topology),
            "world": str(size)}


@dataclass(frozen=True, slots=True)
class BoundModel:
    """What the guards need to describe the checkpoint they are refusing."""

    unet_name: str
    file_identity: str
    memo_context: dict[str, str]
    topology: dict[str, Any]
    world: int


@dataclass(slots=True)
class _Scope:
    """One thread's active waivers and recorded uses."""

    run_id: str = ""
    grants: dict[str, dict[str, str]] = field(default_factory=dict)
    bound: BoundModel | None = None
    fired: dict[str, dict[str, Any]] = field(default_factory=dict)


_LOCAL = threading.local()


def _scope() -> _Scope:
    scope = getattr(_LOCAL, "scope", None)
    if scope is None:
        scope = _Scope()
        _LOCAL.scope = scope
    return scope


def bind_model(unet_name: str, options: Mapping[str, Any] | None,
               lora_stack: Sequence[Any] | None, topology: Mapping[str, Any] | None,
               world: object, *, dispatch: bool = False) -> None:
    """Bind the conditional checkpoint to the next guard without raising.

    Dual-model requests bind only the conditional slot. A load outside the
    authorized dispatch clears thread-local grants so reused worker threads
    cannot apply a previous render's consent.
    """
    scope = _scope()
    if not dispatch and (scope.grants or scope.fired):
        log.info("a load outside the authorized dispatch disarms %d accuracy waiver(s) "
                 "on this worker thread", len(scope.grants))
        scope.grants = {}
        scope.fired = {}
        scope.run_id = ""
    try:
        from .gate_ledger import combo_key

        node_options = dict(options or {})
        names = [str(entry.get("name", "")) for entry in (lora_stack or [])
                 if isinstance(entry, Mapping)]
        combo = combo_key(str(unet_name), node_options, names)
        scope.bound = BoundModel(
            unet_name=str(unet_name),
            file_identity=_file_identity(str(unet_name)),
            memo_context=waiver_context(
                combo_key=combo, topology=topology, world=world),
            topology=dict(topology or {}),
            world=int(world) if isinstance(world, int) else 0)
    except Exception as exc:  # a card is best effort; the refusal is not
        log.debug("accuracy-waiver model context not bound (%r)", exc)


def _file_identity(unet_name: str) -> str:
    """Return checkpoint identity or an unreadable sentinel without raising.

    Use ``consent_store`` to avoid importing the worker runtime into this shared
    leaf. A test keeps the identity spellings aligned.
    """
    try:
        import folder_paths

        from .consent_store import file_identity

        path = folder_paths.get_full_path("diffusion_models", unet_name)
        return file_identity(str(path)) if path else "unreadable"
    except Exception:
        return "unreadable"


def _context_fingerprint(context: Mapping[str, str]) -> str:
    """Return a stable short identifier for a narrow waiver context."""
    import hashlib

    from .consent_store import canonical_context

    payload = canonical_context(dict(context))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def activate(request: Mapping[str, Any] | None) -> int:
    """Install this dispatch's thread-local grants and return their count.

    Each sample clears the previous scope first, so authorization cannot outlive
    its dispatch. Malformed stamps install nothing and leave guards refusing.
    """
    scope = _scope()
    scope.grants = {}
    scope.fired = {}
    scope.run_id = ""
    if not isinstance(request, Mapping):
        return 0
    scope.run_id = str(request.get("_dgxm_render_id", "") or "")
    payload = request.get(REQUEST_KEY)
    if not isinstance(payload, Mapping):
        return 0
    from . import consent_pending

    for kind, wire in payload.items():
        grant = consent_pending.grant_from_wire(wire)
        if (grant is None or grant.kind != kind or kind not in KNOWN_WRONG_KINDS
                or grant.consent_source not in CONSENT_SOURCES):
            # Unknown, malformed, non-accuracy, or unsupported-channel stamps
            # grant nothing, preventing an unaudited render.
            log.warning("an accuracy-waiver stamp for %r was not readable and was "
                        "ignored; the guard it names still refuses", kind)
            continue
        scope.grants[kind] = {"id": grant.id, "kind": grant.kind,
                              "consent_source": grant.consent_source}
    return len(scope.grants)


def granted(guard: str) -> dict[str, str] | None:
    """The live grant covering this guard on this dispatch, or None."""
    kind = KNOWN_WRONG_GUARDS.get(guard)
    if kind is None:
        return None
    return _scope().grants.get(kind)


def waived(guard: str) -> bool:
    """Return whether the guard may proceed and record its first use.

    Records are keyed by guard so repeated attention calls log once per render.
    """
    grant = granted(guard)
    if grant is None:
        return False
    scope = _scope()
    if guard in scope.fired:
        return True
    kind = grant["kind"]
    stamp = stamp_text(kind) or f"rendered-under-waiver: {kind}"
    bound = scope.bound
    scope.fired[guard] = {
        "guard": guard,
        "kind": kind,
        "consent_id": grant["id"],
        "consent_source": grant.get("consent_source", "unknown"),
        "run_id": scope.run_id,
        "stamp": stamp,
        "unet_name": "" if bound is None else bound.unet_name,
    }
    log.error("KNOWN-WRONG MATH PROCEEDING UNDER WAIVER: guard %s waived for %s by "
              "consent %s (%s); this output is %s",
              guard, (bound.unet_name if bound is not None else "this render"),
              grant["id"], grant.get("consent_source", "unknown"), stamp)
    return True


def stamps() -> list[dict[str, Any]]:
    """Return the waivers used by this dispatch for inclusion in its result."""
    return [dict(entry) for entry in _scope().fired.values()]


def clear() -> None:
    """Drop this thread's authorization (teardown, tests)."""
    _LOCAL.scope = _Scope()


def card_tail(guard: str, *, family_hint: str | None = None) -> str:
    """Return the bounded descriptor appended after the refusal text.

    An unwaivable guard, missing binding, or encoding error returns an empty tail
    and leaves the refusal in force.
    """
    try:
        descriptor = _descriptor(guard, family_hint=family_hint)
        return "" if descriptor is None else "\n" + consent_descriptor.encode(descriptor)
    except Exception as exc:  # an un-carded refusal is still a refusal
        log.warning("accuracy-waiver card descriptor not encoded for %s (%r)", guard, exc)
        return ""


def _descriptor(guard: str, *, family_hint: str | None
                ) -> consent_descriptor.ConsentDescriptor | None:
    kind = KNOWN_WRONG_GUARDS.get(guard)
    bound = _scope().bound
    if kind is None or bound is None:
        return None
    spec = KIND_SPECS[kind]
    if not spec.wired:
        return None
    scoped = check_guard(kind, guard) or guard
    fingerprint = _context_fingerprint(bound.memo_context)
    return consent_descriptor.ConsentDescriptor(
        version=consent_descriptor.CONSENT_DESCRIPTOR_VERSION,
        kind=kind,
        consent_id=consent_descriptor.consent_id(kind, bound.file_identity, fingerprint),
        unet_name=bound.unet_name,
        file_identity=bound.file_identity or "unreadable",
        context_fingerprint=fingerprint,
        family_hint=family_hint,
        # Preserve the scoped guard so the audit row names the affected family.
        measured={"probe": scoped, "measured": str(spec.measured or ""),
                  "topology": bound.memo_context.get("topology", ""),
                  "world": bound.world},
        memo_context=dict(bound.memo_context),
        human_reason=f"{spec.risk} Measured: {spec.measured}.",
        evidence=spec.evidence or "",
        panel_action=spec.primary_label,
        env_fallback=spec.env_var,
    )
