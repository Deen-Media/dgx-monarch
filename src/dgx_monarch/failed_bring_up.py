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
is whether the processes the bring-up spawned still run. A bring-up whose
rollback stop completed without error, or that started no process, leaves
nothing unknown and records no poison (mesh_factory.left_nothing_running).

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

A spawn that may still be queued needs more. Monarch's spawn_procs does not
block, so a bring-up that failed before Monarch reported every rank's proc
started (FleetHandoff.spawn_settled) may have left a spawn waiting in a Worker
loop that is frozen or stalled rather than dead: no actor exists yet, and the
spawn lands when the loop resumes. No actor now proves nothing about it. So
for such a bring-up every host must also show that the loop it attached to no
longer runs, since a queued spawn dies with its loop: the attach keeps each
dialed loop's listener generation (mesh_attach.read_dial), and the scan lists
the generation of every live Worker loop (actor_reaper.liveness). Where that
loop still runs, or the host cannot list its loops, the poison stays and the
refusal names the host with the remedy for its reason (_HOLD_WORDS): a Worker
service restart ends a spawn its loop still holds, and the next check lifts. A
published fleet needs no such rule: its handle exists only after every rank's
proc started.
"""
from __future__ import annotations

import re
import threading
from types import SimpleNamespace
from typing import Any

from . import mesh_creation, mesh_factory
from .log import get_logger

log = get_logger("dgx_monarch.mesh")

# The poisons get_mesh and the creation finalizer record for a failed
# bring-up, as against an attach or transport failure (mesh_attach).
_BRING_UP_POISONS = ("actor spawn failed", "mesh creation failed")
_GENERATION = re.compile(r"[0-9a-f]{12}")  # a listener generation (cli/listener_generation)
# The last proof as one immutable triple: the claims it was taken for, its
# evidence, and the hosts a spawn that may be queued held it on. Every reader
# takes it in one read, so keys and evidence always come from the same call; it
# is replaced, and cleared only while it is still the triple the clearing call
# stored, under _PROOF_LOCK.
_LAST_PROOF: tuple[frozenset[object], Any, tuple[tuple[str, str], ...]] | None = None
_PROOF_LOCK = threading.Lock()


# A spawn that failed before it called Monarch, outside the factory's own
# rollback: a spawn host that cannot be isolated (another torchmonarch than the
# pin) or a Worker module that will not import. Nothing started, and a retry in
# this process meets the same fault, so the refusal names it, not the generic poison.
# It stays restart-only though its text begins like a liftable bring-up poison
# (provable): the outer except in get_mesh confirms and pops this attempt's
# settled creation claim, so _claims_to_prove finds nothing left to prove and
# liftable() is False.
_BEFORE_ANY_PROCESS = "actor spawn failed before any process started"


def spawn_failure_poison(handoff: mesh_factory.FleetHandoff, primary: BaseException) -> str:
    """The poison get_mesh records for a spawn that failed outside the factory's rollback."""
    if handoff.proc_spawn_called:
        return "actor spawn failed; cleanup state unavailable"
    return f"{_BEFORE_ANY_PROCESS} ({mesh_factory.safe_failure_evidence(primary)})"


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


def _queued_spawn(handoff: mesh_factory.FleetHandoff) -> bool:
    """Whether this bring-up issued a spawn Monarch never reported settled, so it may still be queued."""
    return handoff.proc_spawn_called and not handoff.spawn_settled


def _loops_holding(claims: list[tuple[object, mesh_factory.FleetHandoff]],
                   evidence: Any) -> tuple[tuple[str, str], ...]:
    """The hosts where a spawn that may still be queued could land, each with its reason.

    Such a host passes only when its scan listed every live Worker loop and none
    is the loop the bring-up dialed there. Otherwise it holds: ``running`` (that
    loop still runs), ``unlisted`` (the host answered without a complete list:
    an older package, a malformed list, a live loop the scan could not name) or
    ``unrecorded`` (the attach kept no generation for it).
    """
    listed = dict(getattr(evidence, "loops", ()) or ())
    held: dict[str, str] = {}
    for _claim, handoff in claims:
        for address in _addresses(handoff.config) if _queued_spawn(handoff) else ():
            dialed = handoff.loop_gens.get(address, "")
            if not _GENERATION.fullmatch(dialed):
                held.setdefault(address, "unrecorded")
            elif listed.get(address) is None:
                held.setdefault(address, "unlisted")
            elif dialed in listed[address]:
                held.setdefault(address, "running")
    return tuple(sorted(held.items()))


# Per hold reason: what holds, and the remedy a test shows lifts it.
_HOLD_WORDS = {
    "running": ("the Worker loop it attached to still runs on {hosts}",
                "restart the Worker service on {hosts} (dgxm restart), which ends the queued spawn "
                "with its loop"),
    # Two causes: a package too old to list loops, and a live loop the scan could
    # not name. dgxm up syncs the package and restarts every Worker service, so it
    # clears both; a Worker restart on that host clears only the second.
    "unlisted": ("{hosts} could not list its Worker loops",
                 "run dgxm up, which syncs the current package and restarts the Worker service "
                 "on every host (a Worker restart on {hosts} alone, dgxm restart, clears only a "
                 "live loop the check could not name)"),
    "unrecorded": ("the attach recorded no loop for {hosts}", "restart ComfyUI"),
}


def _store(proof: tuple[frozenset[object], Any, tuple[tuple[str, str], ...]] | None,
           *, only_if: object = None) -> None:
    """Replace the last proof, or with ``only_if`` clear it only while it is still that triple."""
    global _LAST_PROOF
    with _PROOF_LOCK:
        if only_if is None or _LAST_PROOF is only_if:
            _LAST_PROOF = proof


def release(runtime: dict[str, Any], *, runner: Any = None, fresh: bool = False) -> bool:
    """Lift the poison when every host shows the failed bring-up's processes gone.

    ``fresh`` asks every host again whatever the last answer was: the Reset
    passes it after stopping an attached fleet, whose own processes made any
    earlier answer alive.
    """
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
    # processes are newer than any earlier answer. One read of the triple.
    keys = frozenset(claim for claim, _handoff in claims)
    last = _LAST_PROOF
    seen_keys, seen_evidence, seen_held = last if last is not None else (None, None, ())
    evidence = mesh_liveness._still_answers(seen_evidence) if seen_keys == keys and not fresh else None
    if evidence is not None and seen_held and evidence.age_s > mesh_liveness.PROBE_MIN_INTERVAL_S:
        evidence = None  # a loop hold is asked again, so a Worker restart there counts at once
    if evidence is None:
        config = claims[0][1].config
        evidence = mesh_liveness.probe_fleet(SimpleNamespace(config=config), runner=runner)
        if driver_exit.exiting():
            return False
    held = _loops_holding(claims, evidence) if evidence.answer == mesh_liveness.GONE else ()
    proof = (keys, evidence, held)
    _store(proof)
    if evidence.answer != mesh_liveness.GONE:
        log.warning("a failed bring-up's processes are not proven gone (answer=%s hosts=%s); "
                    "this session still refuses a new fleet", evidence.answer, _hosts_text(evidence))
        return False
    if held:
        log.warning("a failed bring-up's spawn may still be queued in a Worker loop (held=%s); "
                    "this session still refuses a new fleet",
                    ",".join(f"{address}={reason}" for address, reason in held))
        return False
    with runtime["_MESH_LOCK"]:
        if (runtime["_TRANSPORT_POISON"] is not poison or runtime["_MESH_CREATING"]
                or not _retire_claims(claims)):
            return False
        runtime["_TRANSPORT_POISON"] = None
    _store(None, only_if=proof)
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


def held_hosts() -> tuple[tuple[str, str], ...]:
    """The (host, reason) pairs the last check of the current claims held on for a queued spawn."""
    last, claims = _LAST_PROOF, _claims_to_prove()
    if last is None or claims is None or last[0] != frozenset(claim for claim, _handoff in claims):
        return ()
    return last[2]


def hold_parts() -> list[tuple[str, str]]:
    """What holds and the remedy that lifts it, one pair per reason the last check held on."""
    held, parts = held_hosts(), []
    for reason, (what, remedy) in _HOLD_WORDS.items():
        hosts = ", ".join(address for address, why in held if why == reason)
        if hosts:
            parts.append((what.format(hosts=hosts), remedy.format(hosts=hosts)))
    return parts


def fleet_attached(runtime: dict[str, Any] | None) -> bool:
    """Whether a fleet not yet stopped is cached: its processes keep every host busy for the proof.

    The liveness proof counts every actor process on a host, so while such a
    fleet runs no host can show the failed bring-up's processes gone. Only a
    Reset, which stops it first (nodes/recycle_drain), lets the proof lift.
    """
    return runtime is not None and any(
        not getattr(handle, "teardown_complete", False)
        for handle in list(dict(runtime.get("_MESHES") or {}).values()))


def refusal(poison: str, runtime: dict[str, Any] | None = None) -> str:
    """get_mesh's refusal while the session's transport is poisoned, with the remedy that works."""
    if poison.startswith(_BEFORE_ANY_PROCESS):
        return (f"the last bring-up could not start the Worker fleet: {poison}. Nothing of it "
                "runs, but a retry in this ComfyUI process meets the same fault: fix its cause (a "
                "torchmonarch other than the pin, or a dgx-monarch install that will not import; "
                "docs/INSTALL.md), then restart ComfyUI")
    if not liftable(poison):
        return ("this ComfyUI session's Monarch transport is not reusable after a partial "
                f"bring-up failure ({poison}); restart ComfyUI before retrying")
    parts = hold_parts()
    if parts:
        return (f"a failed bring-up may still have a spawn queued in a Worker loop ({poison}): "
                + "; ".join(f"{what}, so {remedy}" for what, remedy in parts)
                + ". The next render and Reset attached mesh check again and build a fresh fleet "
                "once no such loop can hold it")
    last = _LAST_PROOF
    tail = f"; the last check read {_hosts_text(last[1])}" if last is not None else ""
    if fleet_attached(runtime):
        return (f"a failed bring-up left processes the driver cannot yet prove gone ({poison}){tail}. "
                "The attached Worker fleet's own processes keep every host busy, so a render "
                "cannot lift it: press Reset attached mesh, which resets that fleet and then "
                "checks every host; restart ComfyUI if a host keeps them")
    return (f"a failed bring-up left processes the driver cannot yet prove gone ({poison}){tail}. "
            "It checks every host again at the next render and at Reset attached mesh, and "
            "builds a fresh fleet once they are gone; restart ComfyUI if a host keeps them")
