"""Lift a failed cluster bring-up's transport poison once its spawned processes are proven gone.

When a Worker dies while the driver brings a fleet up, the bring-up's own
rollback stop often cannot confirm (Monarch's controller ends a stop of a lost
host without confirming it), and get_mesh poisons the session: no second fleet
may start beside processes whose stop is unknown. The transport itself is not
what that poison protects. Under the pinned Monarch a driver attaches to the
Worker loops again and spawns a fresh fleet in the same session after the
loops restarted (restart_reattach), and after a stop that timed out once the
liveness proof released it (mesh_liveness); nothing in a failed spawn closes
the process transport, which ``enable_transport`` binds once. What is unknown
is whether the processes the bring-up spawned still run.

So the next render, and Reset attached mesh, ask every configured host the
liveness question the stop-timeout latch uses (mesh_liveness.probe_fleet): no
live actor, no live ledger row and no unreadable process. Only a fleet-wide
gone lifts the poison and retires the bring-up's cleanup claims; alive,
unknown or an unreachable host keep it, and the refusal says which. Only the
poisons a failed bring-up records qualify, only for a cluster fleet, and only
while every unresolved claim leaves nothing but its spawned processes unknown
(``_claims_to_prove``): an attach whose result handoff was interrupted
or a spawn callback whose removal is unconfirmed is a state no proof settles.
A local fleet's poison is left as it was.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from . import mesh_creation, mesh_factory
from .log import get_logger

log = get_logger("dgx_monarch.mesh")

# The poisons get_mesh and the creation finalizer record for a failed
# bring-up, as against an attach or transport failure (mesh_attach).
_BRING_UP_POISONS = ("actor spawn failed", "mesh creation failed")
# The last proof and the claims it was taken for, reused for those claims
# within the liveness probe's own interval; empty once a proof lifted them.
_LAST_PROOF: list[Any] = []


def provable(poison: object) -> bool:
    """Whether this poison is a failed bring-up's, which a liveness proof may lift."""
    return isinstance(poison, str) and poison.startswith(_BRING_UP_POISONS)


def _hosts_text(evidence: Any) -> str:
    return ",".join(f"{address}={answer}" for address, answer in getattr(evidence, "per_host", ()))


def _claims_to_prove() -> list[tuple[object, mesh_factory.FleetHandoff]] | None:
    """The unresolved creation claims, when the processes they spawned are their only unknown.

    None when any claim leaves more than that unknown: an attach whose result
    handoff was interrupted, a spawn callback whose removal is unconfirmed, a
    claim with no handoff, or a spawn with no hosts to ask (a local fleet).
    Those are states no liveness proof settles. Reads the creation claims under
    their own lock (mesh_factory owns them).
    """
    with mesh_factory._CREATION_CLEANUP_LOCK:
        pending = [(claim, handoff) for claim, handoff
                   in mesh_factory._CREATION_CLEANUP_CLAIMS.items()
                   if mesh_factory._cleanup_pending(handoff)]
    claims: list[tuple[object, mesh_factory.FleetHandoff]] = []
    for claim, handoff in pending:
        if (not isinstance(handoff, mesh_factory.FleetHandoff) or not handoff.failure_latched
                or not getattr(handoff.config, "hosts", ()) or handoff.callback_cleanup_unconfirmed
                or (handoff.attach_in_progress and not handoff.transport_failure_safe)):
            return None
        claims.append((claim, handoff))
    # One probe answers for every claim only when they all spawned on the same hosts.
    if len({_addresses(handoff.config) for _claim, handoff in claims}) > 1:
        return None
    return claims or None


def _addresses(config: Any) -> tuple[str, ...]:
    return tuple(str(getattr(host, "address", "")) for host in getattr(config, "hosts", ()))


def _retire_claims(claims: list[tuple[object, mesh_factory.FleetHandoff]]) -> bool:
    """Retire these claims once a proof showed their processes gone; False if any changed."""
    with mesh_factory._CREATION_CLEANUP_LOCK:
        held = mesh_factory._CREATION_CLEANUP_CLAIMS
        if any(held.get(claim) is not handoff for claim, handoff in claims):
            return False
        for claim, handoff in claims:
            handoff.cleanup_confirmed = True
            handoff.spawn_in_progress = False
            held.pop(claim, None)
        return True


def release(runtime: dict[str, Any], *, runner: Any = None) -> bool:
    """Lift the poison when every host shows the failed bring-up's processes gone."""
    from . import driver_exit, mesh_liveness

    poison = runtime["_TRANSPORT_POISON"]
    if not provable(poison):
        return False
    with runtime["_MESH_LOCK"]:
        if runtime["_MESH_CREATING"]:
            return False
    claims = _claims_to_prove()
    if claims is None:
        return False
    # A proof answers only for the claims it was taken for: a later bring-up's
    # processes are newer than any earlier answer.
    keys = frozenset(claim for claim, _handoff in claims)
    evidence = (mesh_liveness._still_answers(_LAST_PROOF[1])
                if _LAST_PROOF and _LAST_PROOF[0] == keys else None)
    if evidence is None:
        config = claims[0][1].config
        evidence = mesh_liveness.probe_fleet(SimpleNamespace(config=config), runner=runner)
        if driver_exit.exiting():
            return False
        _LAST_PROOF[:] = [keys, evidence]
    if evidence.answer != mesh_liveness.GONE:
        log.warning("a failed bring-up's processes are not proven gone (answer=%s hosts=%s); "
                    "this session still refuses a new fleet", evidence.answer, _hosts_text(evidence))
        return False
    with runtime["_MESH_LOCK"]:
        if (runtime["_TRANSPORT_POISON"] is not poison or runtime["_MESH_CREATING"]
                or not _retire_claims(claims)):
            return False
        runtime["_TRANSPORT_POISON"] = None
        _LAST_PROOF.clear()
    # The Monarch state a liveness release leaves (a stop it never confirmed), so
    # the next attach takes the released fleet's rules (mesh_attach).
    from . import mesh_teardown

    mesh_teardown.note_liveness_release(addresses=list(_addresses(claims[0][1].config)))
    log.warning("a failed bring-up's poison lifted on liveness proof: the processes it spawned "
                "are gone on every host (hosts=%s); the next attach builds a fresh fleet",
                _hosts_text(evidence))
    return True


def settle(runtime: dict[str, Any]) -> None:
    """At every attach: reap abandoned creations, then lift a proven failed bring-up's poison."""
    mesh_creation.reap_abandoned_attempts(runtime)
    if runtime["_TRANSPORT_POISON"] is not None:
        release(runtime)


def liftable(poison: object) -> bool:
    """Whether a liveness proof can lift this poison: a failed cluster bring-up's, nothing else open."""
    return provable(poison) and _claims_to_prove() is not None


def refusal(poison: str) -> str:
    """get_mesh's refusal while the session's transport is poisoned, with the remedy that works."""
    if not liftable(poison):
        return ("this ComfyUI session's Monarch transport is not reusable after a partial "
                f"bring-up failure ({poison}); restart ComfyUI before retrying")
    last = f"; the last check read {_hosts_text(_LAST_PROOF[1])}" if _LAST_PROOF else ""
    return (f"a failed bring-up left processes the driver cannot yet prove gone ({poison}){last}. "
            "It checks every host again at the next render and at Reset attached mesh, and "
            "builds a fresh fleet once they are gone; restart ComfyUI if a host keeps them")
