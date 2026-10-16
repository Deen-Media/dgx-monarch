"""Delivery failures the fault hook saw, so the capacity price wait can end early.

After the Worker loops restart under an attached mesh, a message to one of the
old actors does not come back to the call that sent it. Monarch reports the
lost message to the driver's fault hook as a supervision event (`undeliverable
message ... delivery failure: ttl expired for dgxm_worker<...>@...tcp://<ip>:<port>`),
and the call's own future never resolves. The capacity agreement used to wait
its whole 30 s deadline for such a future (docs/TROUBLESHOOTING.md #2, #93).

The fault-marker site in ``mesh_safety`` records here every fault that hits a
delivery marker, absorbed or not, and ``await_rank`` reads the record between
short waits. A failure ends the wait at once only when it names both a fleet
host's address and the actor mesh the wait prices on: Monarch writes that mesh
as ``dgxm_worker<uid>``, and ``ActorMesh._name`` reads ``dgxm_worker-<uid>``.
A late failure from a mesh that Recycle, Reset, the supervision reconcile or a
re-attach retired names the same address and another uid, so it never counts.

The first message a driver sends to a restarted loop rides the session the old
process accepted and is lost with no report at all (worker_proc_id.py). When
the price request is that first message, nothing comes back, so ``await_rank``
asks the rank once more after ``REASK_S``: the second request reaches the new
process, which reports it undeliverable at once. The quote mutates nothing, so
a second request to a rank that is only slow costs one more read.

Leaf module: the standard library only.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

# The name mesh_factory spawns the fleet's actor mesh under. Monarch writes a
# destination actor as ``dgxm_worker<uid>``, so a failure for any other actor,
# such as a retired mesh's controller polling its old host agent, never counts.
WORKER_MESH = "dgxm_worker"
# How long reading the mesh's name may take. It is local state of a created
# mesh (about a millisecond on CPU, after its processes died as well).
NAME_READ_S = 5.0
# The markers mesh_safety's fault-marker site matches (tests pin the two equal).
MARKERS = ("undeliverable", "delivery failure:", "ttl expired for", "channel closed")
# One poll of a rank's reply. A failure recorded during the wait ends it
# within one step.
POLL_S = 0.5
# When a rank has not answered after this long, its request goes out once more.
REASK_S = 2.0
_KEEP = 64


class WorkerDeliveryError(RuntimeError):
    """A delivery failure to a fleet actor ended the capacity price wait early.

    Class C on the agreement's guard (restart_reattach.py raises it). Like a
    silent rank it is a liveness fault, never a capacity verdict, so it stays
    outside the stock-load capacity family. It lives in this leaf so the render
    path can catch it without importing the actor package at node registration.
    """


# One actor mesh as Monarch names it in a fault: ``dgxm_worker<uid>``.
_MESH = re.compile(re.escape(WORKER_MESH) + r"<([^>\s]+)>")
_LOCK = threading.Lock()
# Process lifetime, bounded: the latest recorded failures, by monotonic stamp.
_SEEN: deque[tuple[float, str]] = deque(maxlen=_KEEP)


def mesh_uid(name: Any) -> str | None:
    """The uid in ``dgxm_worker-<uid>``, the name ``ActorMesh._name`` reads; else None."""
    prefix = f"{WORKER_MESH}-"
    if not isinstance(name, str) or not name.startswith(prefix):
        return None
    uid = name[len(prefix):]
    return uid if uid and re.fullmatch(r"[^<>\s]+", uid) else None


def record(text: str, *, now: float | None = None) -> bool:
    """Keep one reported fault that names a fleet actor; return whether it was kept."""
    if f"{WORKER_MESH}<" not in text or not any(marker in text for marker in MARKERS):
        return False
    stamp = time.monotonic() if now is None else now
    with _LOCK:
        _SEEN.append((stamp, text[:4000]))
    return True


def names(text: str, address: str) -> bool:
    """Whether ``text`` names ``address`` exactly: port 2660 never matches 26600."""
    return bool(address) and re.search(re.escape(address) + r"(?![0-9])", text) is not None


def failed_since(since: float, addresses: Iterable[str], mesh: str | None) -> list[str]:
    """Each address a failure recorded at or after ``since`` names, in the given order.

    Only a failure that names ``mesh`` (a uid, ``mesh_uid``) counts; with no
    mesh, none does.
    """
    if not mesh:
        return []
    with _LOCK:
        recent = [text for stamp, text in _SEEN
                  if stamp >= since and mesh in _MESH.findall(text)]
    return [address for address in dict.fromkeys(addresses)
            if any(names(text, address) for text in recent)]


def fleet_addresses(handle: Any) -> list[str]:
    """A cluster fleet's Worker loop addresses in host order; empty for a local fleet."""
    hosts = getattr(getattr(handle, "config", None), "hosts", None) or ()
    return [str(getattr(host, "address", "") or "") for host in hosts]


def clear() -> None:
    """Forget every record (tests)."""
    with _LOCK:
        _SEEN.clear()


def await_rank(await_step: Callable[[Any, float], Any], future: Any, left: float, *,
               watched: Iterable[str], mesh: str | None, since: float,
               on_failure: Callable[[list[str]], Any],
               reask: Callable[[], Any] | None = None) -> Any:
    """Await one rank's reply for at most ``left`` seconds, watching for delivery failures.

    With no address to watch (a local fleet) or no mesh uid to match (its name
    could not be read), this is the one wait it always was. Otherwise the wait
    runs in ``POLL_S`` steps; after a step that times out, a failure recorded
    since ``since`` that names ``mesh`` and a watched address goes to
    ``on_failure``, which raises. ``reask`` runs once, after ``REASK_S``.
    Raises ``TimeoutError`` when ``left`` is spent.
    """
    watched = [address for address in watched if address]
    if not watched or not mesh:
        return await_step(future, left)
    waited = 0.0
    while True:
        step = min(POLL_S, left - waited)
        try:
            return await_step(future, step)
        except TimeoutError:
            waited += step
            failed = failed_since(since, watched, mesh)
            if failed:
                on_failure(failed)
            if waited >= left:
                raise
            if reask is not None and waited >= REASK_S:
                reask()
                reask = None
