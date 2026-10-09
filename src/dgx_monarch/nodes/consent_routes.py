"""Driver endpoints for consent cards and standing rescue.

``GET /dgxm/consents`` returns pending and active consent state.
``POST /dgxm/consent`` accepts, dismisses, revokes, or changes auto-rescue.
POST uses the same CSRF controls as recycle; ComfyUI authentication remains a
separate boundary. GET exposes only basenames and opaque keys and performs no
mesh RPC or durable writes. It rechecks quarantine before projecting live rows.

Acceptance writes the permanent audit row before the memo. If either required
write fails, no bypass is authorized.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

from .. import consent_pending, consent_store
from ..log import get_logger
from . import routes

log = get_logger(__name__)

ACTION = "consent"
MAX_BODY_BYTES = 8192
# Bound ledger writes without refusing normal panel clicks.
RATE_LIMIT_N = 10
RATE_LIMIT_WINDOW_S = 10.0
# Keep bounded filesystem work off the event loop.
_READ_TIMEOUT_S = 5.0
_ACTION_TIMEOUT_S = 20.0
# The ledger refuses a longer reason (gate_audit_vocab.check_text).
REASON_MAX_CHARS = 400

# Process lifetime: one token bucket limits writes to the shared ledger.
_RATE: dict[str, float] = {"tokens": float(RATE_LIMIT_N), "t": 0.0}
_RATE_LOCK = threading.Lock()


def record_waiver(
    *,
    action: str,
    kind: str,
    target_guard: str | None,
    consent_id: str,
    consent_source: str,
    reason: str,
    combo_key: str,
    artifacts: str,
    memo_context: dict[str, Any],
    capability_context: dict[str, Any] | None,
    unet_name: str,
    file_identity: str,
    loras: int,
    revoked_by: str | None = None,
    stamp: str | None = None,
    run_id: str | None = None,
) -> str | None:
    """Append a permanent WAIVER row and return its key on success.

    The ledger's builder validates row structure and derives the waiver class.
    Audit namespaces and discriminated contexts keep waiver rows from granting
    or displacing gate authority.
    """
    try:
        import folder_paths

        from .. import gate_audit
        from ..gate_audit_vocab import WaiverNotAuditedError, waiver_key
        from ..gate_ledger import GateLedger, comfy_commit
    except ImportError as exc:
        log.error("consent waiver not recorded (%r); the consent was not saved", exc)
        return None
    fields: dict[str, Any] = {
        "combo_key": combo_key,
        "waiver_kind": kind,
        "target_guard": target_guard,
        "consent_id": consent_id,
        "consent_source": consent_source,
        "memo_context": memo_context,
        "capability_context": capability_context,
        "file_identity": file_identity,
        "unet_name": unet_name,
        "loras": loras,
        "reason": reason,
    }
    if revoked_by is not None:
        fields["revoked_by"] = revoked_by
    if stamp is not None:
        fields["stamp"] = stamp
    if run_id is not None:
        # A use row binds the waiver stamp to the render that produced it.
        fields["run_id"] = run_id
    try:
        ledger = GateLedger(folder_paths.get_output_directory())
        commit = comfy_commit()
        if action == "grant":
            # A grant is refused unless its audit row is durable.
            gate_audit.record_waiver_grant(ledger, artifacts, commit, **fields)
        elif not gate_audit.record_waiver(ledger, artifacts, commit, action=action, **fields):
            return None
    except (OSError, TypeError, ValueError, WaiverNotAuditedError) as exc:
        # A persistence or schema error returns None, which refuses a grant.
        log.error("consent waiver row rejected (%r)", exc)
        return None
    return waiver_key(kind, combo_key)


def _rate_limit_ok(now: float | None = None) -> bool:
    """Monotonic token bucket, process-wide, POSTs only."""
    stamp = time.monotonic() if now is None else now
    with _RATE_LOCK:
        last = _RATE["t"] or stamp
        refill = (stamp - last) * (RATE_LIMIT_N / RATE_LIMIT_WINDOW_S)
        _RATE["tokens"] = min(float(RATE_LIMIT_N), _RATE["tokens"] + max(0.0, refill))
        _RATE["t"] = stamp
        if _RATE["tokens"] < 1.0:
            return False
        _RATE["tokens"] -= 1.0
        return True


def _capacity_query(kind: object, combo_key: object, artifacts: object,
                    artifacts_legacy: object, legacy_complete: object,
                    capability_context: object) -> Any:
    spec = consent_pending.KIND_SPECS.get(str(kind or ""))
    if spec is None or spec.refusal_class != "C":
        return None
    from ..gate_ledger import ArtifactSetSignature
    from .consent_rescue import quarantine_query
    if (not isinstance(artifacts, str) or not artifacts
            or not isinstance(artifacts_legacy, str) or not artifacts_legacy
            or not isinstance(legacy_complete, bool)):
        return quarantine_query("", "")
    signature = ArtifactSetSignature(artifacts, artifacts_legacy, legacy_complete)
    return quarantine_query(combo_key, signature, capability_context)


def _live_rows(memo: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Project rows against one quarantine snapshot without mutating state."""
    pending: list[tuple[dict[str, Any], Any]] = []
    queries: set[Any] = set()
    for card in consent_pending.pending_cards():
        entry = consent_pending.peek_pending(str(card["key"]), str(card["id"]))
        if entry is None:
            continue
        desc = entry.descriptor
        query = _capacity_query(
            desc.kind, desc.combo_key, desc.artifacts, desc.artifacts_legacy,
            desc.artifacts_legacy_complete, desc.capability_context)
        if query is not None:
            queries.add(query)
        pending.append((card, query))

    listed_active = consent_store.list_active(memo)
    active: list[tuple[dict[str, Any], Any]] = []
    for row in listed_active:
        record = consent_store.ConsentRecord.from_json(memo["consents"].get(row["key"]))
        if record is None:
            continue
        query = _capacity_query(
            record.kind, record.combo_key, record.artifacts, record.artifacts_legacy,
            record.artifacts_legacy_complete, record.capability_context)
        if query is not None:
            queries.add(query)
        active.append((row, query))

    from .consent_rescue import quarantine_decisions
    decisions = quarantine_decisions(tuple(queries))
    pending_rows = [card for card, query in pending
                    if query is None or decisions[query].state == "clear"]
    active_rows: list[dict[str, Any]] = []
    for row, query in active:
        if query is not None and decisions[query].state != "clear":
            continue
        spec = consent_pending.KIND_SPECS.get(str(row.get("kind", "")))
        row["style"] = spec.style if spec is not None else "capacity"
        active_rows.append(row)
    return pending_rows, active_rows


def consent_state() -> dict[str, Any]:
    """Return cards, active consents, toggle state, and the kind table."""
    memo = consent_store.read()
    pending, active = _live_rows(memo)
    return {
        "t": time.time(),
        "schema": consent_store.SCHEMA,
        "auto_rescue": memo["auto_rescue"] is True,
        "auto_rescue_env": consent_pending.auto_rescue_env(),
        "pending": pending,
        "active": active,
        "kinds": {
            spec.kind: {
                "class": spec.refusal_class,
                "style": spec.style,
                "auto_eligible": spec.auto_eligible,
                "wired": spec.wired,
            }
            for spec in consent_pending.KIND_SPECS.values()
        },
    }


def _reason_for(spec: consent_pending.KindSpec, numbers: str | None) -> str:
    """Return the card's reason, dropping numbers before truncating meaning."""
    if numbers:
        candidate = f"{spec.risk} Measured: {numbers}."
        if len(candidate) <= REASON_MAX_CHARS:
            return candidate
    return spec.risk[:REASON_MAX_CHARS]


def _stamp_for(spec: consent_pending.KindSpec) -> str | None:
    """Return the shared class-K waiver stamp used by rows and render metadata."""
    from .. import accuracy_waiver

    return accuracy_waiver.stamp_text(spec.kind)


def _consent_payload(record: consent_store.ConsentRecord, key: str) -> dict[str, Any]:
    return {
        "id": record.id,
        "key": key,
        "kind": record.kind,
        "class": record.waiver_class,
        "artifact": record.artifact,
        "granted_at": record.granted_at,
        "consent_source": record.consent_source,
    }


def _accept(key: str, pending_id: str) -> tuple[int, dict[str, Any]]:
    entry = consent_pending.peek_pending(key, pending_id)
    if entry is None:
        # Require a live server-issued question to prevent preauthorization.
        return 409, {"ok": False,
                     "detail": "no pending consent for that request; queue the render again"}
    desc = entry.descriptor
    spec = consent_pending.KIND_SPECS[desc.kind]
    query = _capacity_query(
        desc.kind, desc.combo_key, desc.artifacts, desc.artifacts_legacy,
        desc.artifacts_legacy_complete, desc.capability_context)
    if query is not None:
        from .consent_rescue import quarantine_decisions

        decision = quarantine_decisions((query,))[query]
        if decision.state != "clear":
            if decision.state == "fail":
                consent_pending.take_pending(key, pending_id)
                revoke_memo_key(key)
            return 409, {"ok": False, "detail": decision.reason}
    existing = consent_store.lookup(desc.kind, desc.path, desc.memo_context)
    if existing is not None:
        consent_pending.take_pending(key, pending_id)
        return 200, {"ok": True, "already": True, "consent": _consent_payload(existing, key)}

    consent_id = entry.consent_id
    reason = _reason_for(spec, desc.numbers)
    try:
        identity = consent_store.file_identity(desc.path)
        # Refuse if artifact identity changed after the card was minted. A new
        # file requires a new authorization and matching audit identity.
        if consent_store.memo_key(desc.kind, desc.path, dict(desc.memo_context)) != key:
            consent_pending.take_pending(key, pending_id)
            return 409, {"ok": False,
                         "detail": ("the checkpoint changed since this question was asked, "
                                    "so the consent was not saved; queue the render again")}
    except OSError as exc:
        return 503, {"ok": False, "detail": f"consent could not be saved: {exc}"}
    ledger_key = record_waiver(
        action="grant",
        kind=desc.kind,
        target_guard=desc.target_guard,
        consent_id=consent_id,
        consent_source="panel",
        reason=reason,
        combo_key=desc.combo_key,
        artifacts=desc.artifacts,
        memo_context=dict(desc.memo_context),
        capability_context=desc.capability_context,
        unet_name=desc.unet_name or desc.artifact,
        file_identity=identity,
        loras=desc.loras,
        stamp=_stamp_for(spec),
    )
    if ledger_key is None:
        return 503, {"ok": False,
                     "detail": ("the waiver could not be recorded in the identity-gate ledger, "
                                "so the consent was not saved; check that the ComfyUI output "
                                "directory is writable, then click again")}
    record = consent_store.ConsentRecord(
        id=consent_id,
        kind=desc.kind,
        target_guard=desc.target_guard,
        waiver_class=spec.refusal_class,
        artifact=desc.artifact,
        path=desc.path,
        file_identity=identity,
        memo_context=dict(desc.memo_context),
        granted_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        granted_epoch=time.time(),
        consent_source="panel",
        reason=reason,
        ledger_key=ledger_key,
        combo_key=desc.combo_key,
        artifacts=desc.artifacts,
        artifacts_legacy=desc.artifacts_legacy,
        artifacts_legacy_complete=desc.artifacts_legacy_complete,
        capability_context=desc.capability_context,
        unet_name=desc.unet_name or desc.artifact,
        loras=desc.loras,
    )
    try:
        consent_store.grant(record, key)
    except consent_store.ConsentStoreError as exc:
        # The durable row records an attempted grant. Either the memo write
        # failed or the memo holds it unsynced, so keep the card open for retry.
        return 503, {"ok": False, "detail": f"consent could not be saved: {exc}"}
    consent_pending.take_pending(key, pending_id)
    return 200, {"ok": True, "consent": _consent_payload(record, key),
                 "ledger": {"recorded": True, "key": ledger_key}}


def _revoke(key: str, revoked_by: str = "user") -> tuple[int, dict[str, Any]]:
    memo = consent_store.read()["consents"].get(key)
    record = consent_store.ConsentRecord.from_json(memo)
    try:
        removed = consent_store.revoke(key)
    except consent_store.ConsentStoreError as exc:
        return 503, {"ok": False, "detail": f"consent could not be revoked: {exc}"}
    if removed and record is not None:
        # Revoke the memo first. Audit failure must not leave consent active.
        spec = consent_pending.KIND_SPECS.get(record.kind)
        record_waiver(
            action="revoke",
            kind=record.kind,
            target_guard=record.target_guard or None,
            consent_id=record.id,
            consent_source=record.consent_source,
            reason=record.reason,
            combo_key=record.combo_key or "",
            artifacts=record.artifacts or "",
            memo_context=dict(record.memo_context),
            capability_context=record.capability_context,
            unet_name=record.unet_name or record.artifact,
            file_identity=record.file_identity,
            loras=record.loras,
            revoked_by=revoked_by,
            stamp=_stamp_for(spec) if spec is not None else None,
        )
    return 200, {"ok": True, "already": not removed}


def revoke_memo_key(key: str) -> bool:
    """Revoke one memo by key with a gate_fail audit row. False when absent."""
    if not consent_store.read()["consents"].get(key):
        return False
    status, _payload = _revoke(key, revoked_by="gate_fail")
    return status == 200


def revoke_for_gate_fail(unet_name: object) -> int:
    """Revoke every rescue consent invalidated by an identity-gate FAIL.

    Match checkpoint name across narrow memo contexts. Store failure does not
    weaken the quarantine already applied by the caller.
    """
    if not isinstance(unet_name, str) or not unet_name:
        return 0
    revoked = 0
    for row in consent_store.list_active():
        record = consent_store.ConsentRecord.from_json(
            consent_store.read()["consents"].get(row["key"]))
        if record is None or record.waiver_class != "C":
            continue
        if record.unet_name != unet_name and record.artifact != unet_name:
            continue
        status, _payload = _revoke(row["key"], revoked_by="gate_fail")
        revoked += 1 if status == 200 else 0
    return revoked


STANDING_KIND = "auto-rescue-standing"
STANDING_COMBO = "standing"
STANDING_REASON = ("Capacity rescue is allowed without asking: when stock residency cannot fit a "
                   "checkpoint, it is loaded slab-resident instead. Every rescue is still byte-verified "
                   "against the checkpoint, and the gate ledger records each combination it rescues.")


def _auto_rescue(value: object) -> tuple[int, dict[str, Any]]:
    if not isinstance(value, bool):
        return 400, {"ok": False, "detail": "auto_rescue needs a boolean value"}
    # Audit the standing decision separately from each later rescue event.
    standing = record_waiver(
        action="grant" if value else "revoke",
        kind=STANDING_KIND,
        target_guard=None,
        consent_id=consent_pending.new_consent_id(),
        consent_source="panel",
        reason=STANDING_REASON,
        combo_key=STANDING_COMBO,
        artifacts="0" * 64,
        memo_context={"scope": "standing"},
        capability_context=None,
        unet_name="(every checkpoint)",
        file_identity="(standing authorization, no checkpoint)",
        loras=0,
        revoked_by="user" if not value else None,
    )
    if standing is None:
        return 503, {"ok": False,
                     "detail": ("the standing authorization could not be recorded in the identity-gate ledger, so "
                                "the toggle did not change; check that the ComfyUI output directory is writable")}
    try:
        consent_store.set_auto_rescue(value)
    except consent_store.ConsentStoreError as exc:
        return 503, {"ok": False, "detail": f"the toggle could not be saved: {exc}"}
    # Enabling clears open eligible cards; later rescues still write per-load
    # rows. Class K is never eligible. Disabling revokes nothing.
    cleared = consent_pending.clear_auto_eligible() if value else 0
    log.info("auto-rescue on capacity set to %s from the panel (%d card(s) cleared)",
             value, cleared)
    return 200, {"ok": True, "auto_rescue": value, "cleared": cleared}


def handle_action(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """One POST, resolved synchronously off the event loop."""
    action = body.get("action")
    if action == "auto_rescue":
        return _auto_rescue(body.get("value"))
    key = body.get("key")
    pending_id = body.get("id")
    if not isinstance(key, str) or not key:
        return 400, {"ok": False, "detail": "consent action needs a key"}
    if action == "revoke":
        return _revoke(key)
    if not isinstance(pending_id, str) or not pending_id:
        return 400, {"ok": False, "detail": "consent action needs the pending id"}
    if action == "accept":
        return _accept(key, pending_id)
    if action == "dismiss":
        # Dismissal removes only the invitation and writes no consent.
        dismissed = consent_pending.take_pending(key, pending_id) is not None
        return 200, {"ok": True, "dismissed": dismissed}
    return 400, {"ok": False, "detail": f"unknown consent action {action!r}"}


def register(app: Any, web: Any) -> None:
    """Attach the consent routes to the same PromptServer route table."""

    @app.routes.get("/dgxm/consents")
    async def dgxm_consents(_request):
        try:
            payload = await routes._await_bounded(consent_state, _READ_TIMEOUT_S)
        except TimeoutError:
            return web.json_response(
                {"t": time.time(), "pending": [], "active": [],
                 "consent_error": "TimeoutError"},
                status=503)
        return web.json_response(json.loads(json.dumps(payload, default=str)))

    @app.routes.post("/dgxm/consent")
    async def dgxm_consent(request):
        allowed, detail = routes._action_request_allowed(
            request.headers, request.scheme, ACTION)
        if not allowed:
            raise web.HTTPForbidden(text=detail)
        if not _rate_limit_ok():
            return web.json_response(
                {"ok": False, "detail": "too many consent actions; try again in a moment"},
                status=429)
        length = request.content_length
        if length is not None and length > MAX_BODY_BYTES:
            return web.json_response(
                {"ok": False, "detail": "consent request body too large"}, status=413)
        raw = await request.content.read(MAX_BODY_BYTES + 1)
        if len(raw) > MAX_BODY_BYTES:
            return web.json_response(
                {"ok": False, "detail": "consent request body too large"}, status=413)
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            return web.json_response(
                {"ok": False, "detail": "consent request body is not JSON"}, status=400)
        if not isinstance(body, dict):
            return web.json_response(
                {"ok": False, "detail": "consent request body is not an object"}, status=400)
        try:
            status, payload = await routes._await_bounded(
                lambda: handle_action(body), _ACTION_TIMEOUT_S)
        except TimeoutError:
            return web.json_response(
                {"ok": False, "detail": "consent action timed out server-side; see driver log"},
                status=503)
        return web.json_response(payload, status=status)

    log.info("consent routes registered: /dgxm/consents /dgxm/consent")
