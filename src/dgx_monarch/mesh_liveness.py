"""Prove that a dead fleet's processes are gone before releasing its stop latch.

A timed-out ProcMesh stop leaves ownership unresolved. This probe reads the
loop-side actor ledger and reaper procfs scan through the host lifecycle
channel; supervision text alone cannot prove process exit. Only fleet-wide
``gone`` can release the eligible after-death latch.

* ``gone``: no live actor, live ledger row, or unreadable process.
* ``alive``: a live actor or ledger row.
* ``unknown``: unreadable evidence, malformed output, or an unreachable host.

Zombies are gone because they released their pages; stopped processes
(state T) remain alive and retain memory.

Remote scans use ``cli.lifecycle.run_on_host``'s subprocess deadline. Local
scans use one process-lifetime worker and a separate wait deadline.
``run_blocking_off_loop`` is unsuitable: it runs inline without a deadline
off-loop and creates a new thread per on-loop call. One shared worker bounds
thread use even when procfs hangs; later probes then return unknown.
"""
from __future__ import annotations

import json
import queue
import shlex
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import mesh_safety, mesh_teardown
from .log import get_logger

log = get_logger(__name__)

GONE = "gone"
ALIVE = "alive"
UNKNOWN = "unknown"

# Both budgets stay under the mesh creation timeout, so a probe can never be the
# reason an attach misses its own deadline. tests/test_doctrine_hygiene.py pins
# the ordering.
PROBE_HOST_BUDGET_S = 10.0
PROBE_FLEET_BUDGET_S = 20.0

# The floor between two probes of the same handle (rules in _still_answers).
PROBE_MIN_INTERVAL_S = 2.0

# Equal to cli/actor_reaper.LIVENESS_SCHEMA (tests/test_mesh_liveness.py asserts
# it). A line under another schema comes from another package version and reads
# as unknown.
_SCHEMA = 1
_RECONCILE_THREAD = "dgxm-reconcile-stop"
_STALE_HOST_MARKERS = ("unrecognized arguments", "No module named")


@dataclass(frozen=True)
class LivenessEvidence:
    """One fleet-wide answer, with the per-host answers that produced it."""

    answer: str
    per_host: tuple[tuple[str, str], ...]
    at: float
    detail: str = ""

    @property
    def age_s(self) -> float:
        """Seconds since the probe ran, read fresh every time.

        A property and not a field: the retire re-checks the age under the
        handle lock, and a stored number would let a `gone` recorded while a
        live lease blocked the release authorize one long afterwards.
        """
        return max(time.monotonic() - self.at, 0.0)


def host_script(config: Any, host: Any, addresses: tuple[str, ...]) -> str:
    """The actor sweep's host script (cli/actor_sweep.sweep_script), with --liveness.

    Keep the PYTHONPATH branch. A package sync never writes the managed tree on
    the host that runs the driver, so that host must import the checkout the
    driver runs. With the managed path instead, it fails on a missing module or
    an unknown flag and answers unknown every time.
    """
    from .cli import lifecycle
    from .runtime_provenance import SOURCE_ONLY_PYCACHE_PREFIX

    pythonpath = (
        shlex.quote(str(lifecycle._pkg_src_dir())) if lifecycle._is_local(host)
        else f'"$HOME/{lifecycle._MANAGED_SRC_REL}"'
    )
    flags = " ".join(f"--loop-address {shlex.quote(address)}"
                     for address in addresses)
    return f"""
set -u
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX={shlex.quote(SOURCE_ONLY_PYCACHE_PREFIX)}
export PYTHONPATH={pythonpath}
{lifecycle._pybin_shell(config.python_bin)}
"$PYBIN" -m dgx_monarch.cli.actor_reaper --liveness {flags}
"""


def _counts(payload: Any) -> dict[str, Any] | None:
    """The four counts and the loop flag, or None when the line is not ours."""
    if not isinstance(payload, dict) or payload.get("schema") != _SCHEMA:
        return None
    fields: dict[str, Any] = {}
    for name in ("actors", "zombies", "unreadable", "ledger_rows_alive"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        fields[name] = value
    loop = payload.get("loop_alive")
    fields["loop_alive"] = loop if isinstance(loop, bool) else None
    return fields


def _verdict(fields: dict[str, Any]) -> str:
    """Alive first, so a definite alive is logged as one.

    An unreadable process and a live actor both keep the latch, but only a live
    actor names something the operator can inspect.
    """
    if fields["actors"] or fields["ledger_rows_alive"]:
        return ALIVE
    if fields["unreadable"]:
        return UNKNOWN
    return GONE


def _loop_text(value: bool | None) -> str:
    return "none" if value is None else ("true" if value else "false")


def _log_probe(address: str, answer: str, fields: dict[str, Any] | None) -> None:
    # A host that did not answer prints -1 rather than 0: nobody counted zero.
    view = fields or {"actors": -1, "zombies": -1, "unreadable": -1,
                      "ledger_rows_alive": -1, "loop_alive": None}
    log.warning(
        "liveness probe host=%s answer=%s actors=%d zombies=%d unreadable=%d "
        "ledger_alive=%d loop_alive=%s",
        address, answer, view["actors"], view["zombies"], view["unreadable"],
        view["ledger_rows_alive"], _loop_text(view["loop_alive"]))


def probe_host(config: Any, host: Any, runner: Callable[..., Any],
               timeout_s: float = PROBE_HOST_BUDGET_S) -> str:
    """One host's answer. Every failure answers unknown."""
    address = str(getattr(host, "address", "") or getattr(host, "name", ""))
    addresses = tuple(str(entry.address)
                      for entry in (getattr(config, "hosts", ()) or ()))
    try:
        # Keep the float: rounding the remaining budget up to a whole second
        # lets the last host overrun PROBE_FLEET_BUDGET_S.
        result = runner(config, host, host_script(config, host, addresses),
                        timeout=max(0.05, float(timeout_s)))
    except Exception as exc:
        log.warning("liveness probe host=%s did not run: %r", address, exc)
        _log_probe(address, UNKNOWN, None)
        return UNKNOWN
    # Do not write `returncode or 0`: it reads a missing code as exit 0 and
    # parses an unfinished result.
    raw = getattr(result, "returncode", None)
    code = raw if isinstance(raw, int) else 1
    stderr = str(getattr(result, "stderr", "") or "")[-200:]
    lines = [line for line in str(getattr(result, "stdout", "") or "").splitlines()
             if line.strip()]
    fields: dict[str, Any] | None = None
    # One line exactly, or unknown: a host that prints a warning ahead of its
    # JSON is in a state this driver does not understand.
    if code == 0 and len(lines) == 1:
        try:
            fields = _counts(json.loads(lines[0]))
        except ValueError:
            fields = None
    if code != 0:
        if any(marker in stderr for marker in _STALE_HOST_MARKERS):
            # Two remedies: a remote host gets the package from a sync, the
            # driver's host from its own checkout (see host_script).
            log.warning(
                "liveness probe host=%s cannot answer: its package is missing or "
                "predates the liveness flag; run dgxm up to deploy it there, or "
                "restart ComfyUI if that host runs the driver",
                address)
        log.warning("liveness probe host=%s exit=%d stderr=%s",
                    address, code, stderr)
    answer = UNKNOWN if fields is None else _verdict(fields)
    _log_probe(address, answer, fields)
    return answer


_SCAN_NAME = "dgxm-liveness"
_SCAN_WEDGED = "the local liveness scanner is still stuck on an earlier scan"
_scan_jobs: queue.Queue = queue.Queue()
_scan_lock = threading.Lock()
_scan_thread: threading.Thread | None = None
_scan_busy = False


def _scan_worker() -> None:
    """Run offered scans one at a time, for the life of the process."""
    global _scan_busy
    while True:
        scanner, holder, done = _scan_jobs.get()
        try:
            holder.append((scanner(), None))
        except BaseException as exc:
            holder.append((None, exc))
        finally:
            with _scan_lock:
                _scan_busy = False
            done.set()


def _offer_scan(scanner: Callable[[], Any], holder: list, done: Any) -> bool:
    """Hand one scan to the shared worker, or refuse because it is stuck.

    A thread per probe would leak one live thread per hung procfs read for the
    life of ComfyUI. One worker cannot leak: a stuck scan holds it and the next
    probe is told so.
    """
    global _scan_thread, _scan_busy
    with _scan_lock:
        running = _scan_thread is not None and _scan_thread.is_alive()
        if _scan_busy and running:
            return False
        if not running:
            # A worker that died while marked busy would refuse every later
            # scan for the life of the process. Start a fresh one instead.
            _scan_thread = threading.Thread(
                target=_scan_worker, name=_SCAN_NAME, daemon=True)
            _scan_thread.start()
        _scan_busy = True
        _scan_jobs.put((scanner, holder, done))
    return True


def _local_answer(scanner: Callable[[], Any] | None) -> tuple[str, str]:
    """Run the in-process scan on the shared worker, under the host deadline.

    Never raises, and never waits on anything but its own deadline.
    """
    if scanner is None:
        from .cli import actor_reaper

        scanner = actor_reaper.liveness
    holder: list[tuple[Any, BaseException | None]] = []
    done = threading.Event()
    if not _offer_scan(scanner, holder, done):
        log.warning("liveness probe host=local answer=%s: %s", UNKNOWN, _SCAN_WEDGED)
        return UNKNOWN, _SCAN_WEDGED
    done.wait(timeout=PROBE_HOST_BUDGET_S)
    if not holder:
        return UNKNOWN, "the local liveness scan did not finish in time"
    payload, exc = holder[0]
    if exc is not None:
        return UNKNOWN, f"the local liveness scan failed ({type(exc).__name__})"
    fields = _counts(payload)
    if fields is None:
        return UNKNOWN, "the local liveness scan returned an unreadable answer"
    answer = _verdict(fields)
    _log_probe("local", answer, fields)
    return answer, ""


def probe_fleet(handle: Any, *, now: Callable[[], float] = time.monotonic,
                runner: Callable[..., Any] | None = None,
                scanner: Callable[[], Any] | None = None) -> LivenessEvidence:
    """Ask every configured host, then fold. Gone needs every host to say gone."""
    config = getattr(handle, "config", None)
    hosts = list(getattr(config, "hosts", ()) or ())
    # Two clocks: ``now`` is the budget clock a caller may inject, and the
    # stamp is always the real one, taken at the start. An injected stamp would
    # make the age the retire re-reads mean nothing, and a stamp taken at the
    # end would understate the first host's age by up to the fleet budget.
    started = now()
    stamp = time.monotonic()
    if not hosts:
        answer, detail = _local_answer(scanner)
        return LivenessEvidence(answer, (("local", answer),), stamp, detail)
    if runner is None:
        from .cli import lifecycle

        runner = lifecycle.run_on_host
    per_host: list[tuple[str, str]] = []
    for host in hosts:
        remaining = PROBE_FLEET_BUDGET_S - (now() - started)
        if remaining <= 0.0:
            per_host.append((str(host.address), UNKNOWN))
            continue
        per_host.append((str(host.address), probe_host(
            config, host, runner, min(PROBE_HOST_BUDGET_S, remaining))))
    answers = [answer for _address, answer in per_host]
    if ALIVE in answers:
        answer = ALIVE
    elif all(entry == GONE for entry in answers):
        answer = GONE
    else:
        answer = UNKNOWN
    return LivenessEvidence(answer, tuple(per_host), stamp)


def _still_answers(evidence: Any) -> LivenessEvidence | None:
    """Reuse a recent probe when its evidence or retry interval permits.

    Only ``gone`` remains reusable for the full evidence window; retirement
    rechecks its age under the handle lock. ``alive`` and ``unknown`` are retried
    after ``PROBE_MIN_INTERVAL_S``. Calls within that interval share the previous
    result instead of issuing SSH once per blocked render.
    """
    if not isinstance(evidence, LivenessEvidence):
        return None
    if evidence.age_s <= PROBE_MIN_INTERVAL_S:
        return evidence
    if (evidence.answer == GONE
            and evidence.age_s <= mesh_teardown.EVIDENCE_MAX_AGE_S):
        return evidence
    return None


def _fold_text(evidence: LivenessEvidence) -> str:
    return ",".join(f"{address}={answer}" for address, answer in evidence.per_host)


def try_release(handle: Any, *, runner: Callable[..., Any] | None = None,
                scanner: Callable[[], Any] | None = None) -> bool:
    """Clear an eligible stop-timeout latch after proving all workers are gone.

    Return immediately for other handle states. Callers must invoke this before
    taking mesh or lifecycle locks: the fleet probe can take 20 seconds and must
    not block other lifecycle operations under those locks.
    """
    if handle is None:
        return False
    if threading.current_thread().name == _RECONCILE_THREAD:
        return False
    try:
        verdict = mesh_safety.coherent_lifecycle_verdict(handle, RuntimeError)
    except (AttributeError, RuntimeError):
        return False
    if verdict != "blocked":
        return False
    if mesh_teardown.block_cause(handle) != mesh_teardown.TAG_STOP_TIMEOUT_AFTER_DEATH:
        return False
    evidence = _still_answers(getattr(handle, "_liveness_evidence", None))
    if evidence is None:
        evidence = probe_fleet(handle, runner=runner, scanner=scanner)
        handle._liveness_evidence = evidence
    log.warning("liveness fold answer=%s hosts=%s",
                evidence.answer, _fold_text(evidence))
    if evidence.answer != GONE:
        return False
    return mesh_teardown.retire_after_liveness_proof(handle, evidence)
