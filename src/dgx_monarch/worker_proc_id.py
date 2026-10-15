"""The per-start Monarch ProcId each Worker loop serves, and the driver's dial.

A loop that serves a bare ``tcp://ip:port`` URL takes Monarch's singleton
``service`` proc identity, the same on every start. A driver that outlives a
loop restart then keeps its old ordering session for the restarted host agent,
and its next attach stalls until the config-push budget runs out
(docs/TROUBLESHOOTING.md #2). So each loop start serves
``<proc-id>@tcp://ip:port`` with an id derived from the boot id, the loop's
pid and its start time: no restart can reuse one, and the value normally
starts with the 12 hex digits of the listener generation the probe already
prints (a host whose procfs will not read gets a random value instead).

Where the id travels: the loop writes it into its actor ledger loop record
before it serves (``cli/actor_ledger.publish_loop``). The listener-generation
probe (``cli/listener_generation.py``), which the driver already runs on every
host before every attach, reads that record for the process holding the
listener and prints the value. The driver dials only what the probe read for
that attempt, and refuses before dialing when a host has no usable id: a stale
or bare dial bounces at the host at once, yet the attach still waits out the
whole config-push budget.

The first message after a restart: this driver's link to the address still
holds the channel session the old process accepted. The first message it
sends there rides that session, the new process refuses the session, and the
message is lost; the next send replaces the session and gets through. A lost
config push is fatal for that id, because the new host agent then waits for it
before it handles anything later from this driver. So before the first dial to
a loop whose id changed since this process last dialed it, ``prime`` sends one
throwaway message to that address and waits ``PRIME_SETTLE_S``.

Leaf module: the standard library only at import. The two calls that build a
ProcId import the pinned torchmonarch when they run.
"""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

# The label torchmonarch's own per-host service-id helper uses.
LABEL = "service"
# The probe line field and the actor ledger loop record key.
FIELD = "proc_id"
ENTRY = "docs/TROUBLESHOOTING.md #2"
_VALUE = re.compile(r"[0-9a-f]{16}")

# Probe readings that are not an id. ``none`` and ``unknown`` keep the
# meaning they have for the listener generation.
BARE = "bare"                # the holder's loop record has no id: an older release serves the bare URL
UNPUBLISHED = "unpublished"  # no loop record names the process holding the listener
MALFORMED = "malformed"      # the holder's record carries something that is not an id
UNREADABLE = "unreadable"    # the ledger file would not read
NONE = "none"                # nothing holds the listener
UNKNOWN = "unknown"          # the probe could not tell who holds it, or did not answer
# Readings that are facts about the loop, as against a probe that saw nothing.
DEFINITE = frozenset({BARE, UNPUBLISHED, MALFORMED})
# The wait after the throwaway message: enough for the link to reconnect, be
# refused and be replaced. Monarch retried its reconnects about once a second
# in the 2026-10-15 CPU logs.
PRIME_SETTLE_S = 2.0
# The throwaway message's destination: no such proc exists on any host, and
# nothing waits for a reply.
_PRIME_PROC = "dgxm_prime"
_PRIME_ACTOR = "stale_session"


class WorkerProcIdError(RuntimeError):
    """A pre-dial refusal: a host's current id is unknown, so no attach ran.

    ``unknown`` names each such host and the reason, without the remedy or
    the claim that nothing dialed, for a reading taken after a dial failed.
    """

    unknown: str = ""


def derive(boot_id: str, pid: int, starttime: int) -> str:
    """One loop start's id value: 16 lowercase hex digits.

    The digest input is the one the listener generation hashes, so the value
    starts with that generation's 12 digits.
    """
    if not boot_id or pid <= 0 or starttime <= 0:
        raise ValueError("a loop id needs the boot id, a pid and a start time")
    digest = hashlib.sha256(f"{boot_id}|{pid}|{starttime}".encode())
    return digest.hexdigest()[:16]


def current(proc_root: str | None = None) -> str:
    """This process's id value, derived from its own procfs identity.

    A host whose procfs will not answer gets a random value instead: it is
    still new on every start, and the driver reads the published record, not
    the formula.
    """
    from .cli import proc_identity

    root = proc_identity.PROC_ROOT if proc_root is None else proc_root
    pid = os.getpid()
    stat = proc_identity.read_stat(pid, root)
    try:
        return derive(proc_identity.boot_id(root), pid, 0 if stat is None else stat[1])
    except ValueError:
        return f"{secrets.randbits(64):016x}"


def is_value(value: object) -> bool:
    return isinstance(value, str) and _VALUE.fullmatch(value) is not None


def proc_addr(address: str, value: str) -> str:
    """``<proc-id>@<address>``, built with the pinned torchmonarch's helpers."""
    if not is_value(value):
        raise ValueError(f"not a loop id value: {value!r}")
    from monarch._rust_bindings.monarch_hyperactor.proc import ProcId, Uid
    from monarch._src.job.service_identity import service_proc_addr

    return service_proc_addr(address, ProcId(Uid.instance_from_value(int(value, 16), LABEL)))


def problem(fields: Mapping[str, Any]) -> tuple[str, str]:
    """``(reason, remedy)`` for a reading that carries no usable id."""
    value = str(fields.get(FIELD, ""))
    if value == NONE or fields.get("gen") == NONE:
        return ("nothing holds its listener",
                "start the Worker services (`dgxm up`), then render again")
    if value == BARE:
        return ("its loop runs an older release that serves the bare address",
                "run the Worker services on this driver's release (`dgxm update`, or "
                "`dgxm restart` on a plain install; for Workers in release slots see "
                "docs/TROUBLESHOOTING.md #118), then render again")
    if value in (UNPUBLISHED, MALFORMED):
        return (f"no loop record publishes the id of the process holding its listener ({value})",
                "render again, which reads the id again; if it repeats, restart the "
                "Worker services (`dgxm restart`)")
    if value == UNREADABLE:
        return ("its actor ledger file would not read (unreadable)",
                "check that `~/.local/state/dgx-monarch/actor-procs.json` on that host is a "
                "readable JSON file, then render again; if it repeats, restart the Worker "
                "services (`dgxm restart`)")
    error = fields.get("error") or (value if value else "no reading")
    return (f"the probe did not read it ({error})",
            "fix the host or ssh fault, then render again")


def read_values(hosts: Sequence[Any],
                readings: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Each host's id value from this attempt's readings, in host order.

    Raises ``WorkerProcIdError`` naming every host without a usable id, before
    anything dials.
    """
    values: list[str] = []
    missing: list[str] = []
    unknown: list[str] = []
    for host in hosts:
        fields = readings.get(host.address) or {}
        value = fields.get(FIELD)
        if is_value(value):
            values.append(str(value))
            continue
        reason, remedy = problem(fields)
        missing.append(f"{host.name} ({host.address}): {reason}; {remedy}")
        unknown.append(f"{host.name} ({host.address}: {reason})")
    if missing:
        error = WorkerProcIdError(
            "no attach ran: the id each Worker loop serves on this start is unknown "
            f"for {'; '.join(missing)}. This ComfyUI session stays usable ({ENTRY}).")
        error.unknown = "; ".join(unknown)
        raise error
    return values


def dial_addresses(hosts: Sequence[Any],
                   readings: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Each host's ``<proc-id>@<address>`` from this attempt's readings."""
    return [proc_addr(host.address, value)
            for host, value in zip(hosts, read_values(hosts, readings), strict=True)]


def channel_addr(address: str) -> str:
    """Monarch's channel spelling of a ``tcp://ip:port`` address."""
    from .config import tcp_endpoint

    host, port = tcp_endpoint(address)
    return f"tcp:[{host}]:{port}" if ":" in host else f"tcp:{host}:{port}"


def prime(addresses: Sequence[str], *, settle_s: float = PRIME_SETTLE_S,
          sleep: Callable[[float], None] = time.sleep) -> list[str]:
    """Send one throwaway message to each address, then wait ``settle_s``.

    Each message goes where nothing listens for it and asks for no undeliverable
    return, so it costs nothing whether it is lost on the old session or
    dropped by the host. Returns the addresses it could not send to, and never
    raises: an unprimed loop still gets its attach.
    """
    try:
        from monarch._rust_bindings.monarch_hyperactor.actor import (
            PythonMessage,
            PythonMessageKind,
        )
        from monarch._rust_bindings.monarch_hyperactor.buffers import Buffer
        from monarch._rust_bindings.monarch_hyperactor.mailbox import PortId, PortRef
        from monarch._rust_bindings.monarch_hyperactor.proc import ActorAddr
        from monarch._src.actor.actor_mesh import context

        instance = context().actor_instance._as_rust()
    except Exception:
        return list(addresses)
    unsent: list[str] = []
    for address in addresses:
        try:
            actor = ActorAddr(addr=channel_addr(address), proc_name=_PRIME_PROC,
                              actor_name=_PRIME_ACTOR)
            port = PortRef(PortId(actor_id=actor, port=0))
            port.return_undeliverable = False
            port.send(instance, PythonMessage(
                PythonMessageKind.Result(None), Buffer().freeze(), []))
        except Exception:
            unsent.append(address)
    if len(unsent) < len(addresses):
        sleep(settle_s)
    return unsent


def read_dial(config: Any, *, runner: Any = None, timeout: float = 10.0) -> list[str]:
    """Probe every configured loop and return the dial strings, or refuse."""
    from .cli import listener_generation

    if runner is None:
        from .cli import lifecycle

        runner = lifecycle.run_on_host
    readings = listener_generation.fleet(config, runner=runner, timeout=timeout)
    return dial_addresses(config.hosts, readings)
