"""The driver's attach reads each loop's id first and dials only that.

docs/TROUBLESHOOTING.md #2. ``mesh_attach`` reads every loop's id with the
listener-generation probe immediately before each attempt, dials
``<proc-id>@<address>``, keeps every bookkeeping key on the bare address,
and refuses before anything dials when an id is unknown. The probe and the
attach are stubbed; no ssh or Monarch runs.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from dgx_monarch import attach_trace, mesh_attach, mesh_teardown, worker_proc_id
from dgx_monarch.cli import listener_generation

ADDRESS_A = "tcp://10.0.0.1:26600"
ADDRESS_B = "tcp://10.0.0.2:26600"
ID_1, ID_2 = "1111111111110001", "2222222222220002"


def _config(*addresses: str, auto_heal: bool = False):
    from dgx_monarch.config import ClusterConfig, HostConfig

    return ClusterConfig(
        hosts=tuple(HostConfig(name=f"h{index}", address=address)
                    for index, address in enumerate(addresses, 1)),
        client_bind="tcp://10.0.0.1:0", auto_heal=auto_heal,
        transport_security="trusted_fabric")


class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def _record(self, template, *args):
        self.lines.append(template % args if args else template)

    info = warning = error = _record

    def traced(self, event: str) -> list[dict[str, str]]:
        parsed = [attach_trace.parse(line) or {} for line in self.lines]
        return [fields for fields in parsed if fields.get("event") == event]


class _Factory:
    def __init__(self) -> None:
        self.safe_failures = 0

    def safe_transport_failure(self, handoff):
        self.safe_failures += 1
        if handoff is not None:
            handoff.transport_failure_safe = True

    def safe_failure_evidence(self, exc):
        return repr(exc)

    def safe_failure_message(self, exc):
        return repr(exc)


@pytest.fixture
def rig(monkeypatch):
    """mesh.py's runtime seams, with each probe reading queued per attempt."""
    import monarch.actor as monarch_actor

    monkeypatch.setattr(monarch_actor, "enable_transport", lambda *_a, **_k: None)
    monkeypatch.setattr(attach_trace, "_LAST_SEEN", {})
    released = SimpleNamespace(value=False, asked=[])

    def released_recently(**kwargs):
        released.asked.append(kwargs.get("addresses"))
        return released.value

    monkeypatch.setattr(mesh_teardown, "released_predecessor_recently", released_recently)
    log = _Log()
    readings: list[dict[str, dict[str, str]]] = []

    def fleet(config, **_kwargs):
        return readings.pop(0) if len(readings) > 1 else readings[0]

    monkeypatch.setattr(listener_generation, "fleet", fleet)
    primed: list[list[str]] = []
    monkeypatch.setattr(worker_proc_id, "prime",
                        lambda addresses: primed.append(list(addresses)) or [])
    dials: list[list[str]] = []
    raised: list[tuple[str, list[str], BaseException, dict[str, str]]] = []

    def build(results):
        def attach_once(dial):
            dials.append(list(dial))
            outcome = results.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        def raise_attach(addresses, exc, **detail):
            raised.append(("attach", list(addresses), exc, detail))
            raise RuntimeError("attach failed") from exc

        def raise_released(addresses, exc, *, auto_heal, restarted=None, **detail):
            raised.append(("released", list(addresses), exc, detail))
            raise RuntimeError("released attach failed") from exc

        return {
            "mesh_factory": _Factory(), "MeshAttachError": RuntimeError, "log": log,
            "_TRANSPORT_ENABLE_LOCK": threading.Lock(), "_TRANSPORT_POISON": None,
            "_TRANSPORT_BIND": None, "_attach_once": attach_once,
            "_heal_dead_loops": lambda *_a, **_k: None,
            "_raise_attach": raise_attach, "_raise_released_attach": raise_released,
        }

    return SimpleNamespace(build=build, log=log, readings=readings, dials=dials,
                           raised=raised, released=released, primed=primed)


def _reading(**ids: str) -> dict[str, dict[str, str]]:
    """One fleet reading: address -> the probe fields carrying that id."""
    address = {"a": ADDRESS_A, "b": ADDRESS_B}
    return {address[key]: {"gen": value[:12], "proc_id": value}
            for key, value in ids.items()}


def test_each_attempt_dials_the_ids_read_just_before_it(rig):
    """The loop restarted between the two attempts, so the second attempt must
    carry the second reading. An attach that skipped the read, or reused the
    first one, would dial the dead host agent's identity again."""
    rig.readings.extend([_reading(a=ID_1), _reading(a=ID_2)])
    runtime = rig.build([RuntimeError("stale id"), "HOSTS"])
    assert mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None) == "HOSTS"
    assert rig.dials == [[worker_proc_id.proc_addr(ADDRESS_A, ID_1)],
                         [worker_proc_id.proc_addr(ADDRESS_A, ID_2)]]
    phases = [fields["phase"] for fields in rig.log.traced("loop_generation")]
    assert phases == ["before_attach_1", "before_attach_2"]
    assert [fields["proc_id"] for fields in rig.log.traced("loop_generation")] == [ID_1, ID_2]
    assert rig.log.traced("proc_id_moved")[0]["addresses"] == ADDRESS_A
    # Attempt 2 is the first dial to the restarted loop, so it is primed.
    assert rig.primed == [[ADDRESS_A]]


def test_the_first_dial_to_a_loop_that_restarted_since_the_last_dial_is_primed(
        rig, monkeypatch):
    """The driver's link to a restarted loop still holds the old process's
    session, and the first message on it is lost. A lost config push leaves
    the new host agent waiting for it forever, so a throwaway message goes
    first. Skipping it is the stall this test pins."""
    order: list[str] = []
    monkeypatch.setattr(worker_proc_id, "prime", lambda addresses: order.append(
        f"prime {','.join(addresses)} after {len(rig.dials)} dial(s)") or [])
    rig.readings.extend([_reading(a=ID_1, b=ID_2), _reading(a=ID_2, b=ID_2)])
    runtime = rig.build(["HOSTS", "HOSTS"])
    config = _config(ADDRESS_A, ADDRESS_B)
    assert mesh_attach.attach_cluster(runtime, config, None) == "HOSTS"
    assert order == []  # a first-ever dial has no old session to lose a message on
    assert mesh_attach.attach_cluster(runtime, config, None) == "HOSTS"
    assert order == [f"prime {ADDRESS_A} after 1 dial(s)"]
    prime_line = rig.log.traced("proc_id_prime")[0]
    assert (prime_line["attempt"], prime_line["addresses"]) == ("1", ADDRESS_A)
    assert rig.dials[1] == [worker_proc_id.proc_addr(ADDRESS_A, ID_2),
                            worker_proc_id.proc_addr(ADDRESS_B, ID_2)]


def test_a_loop_dialed_again_with_the_same_id_is_not_primed(rig):
    rig.readings.append(_reading(a=ID_1))
    runtime = rig.build(["HOSTS", "HOSTS"])
    for _ in range(2):
        assert mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None) == "HOSTS"
    assert rig.primed == []
    assert rig.log.traced("proc_id_prime") == []


def test_a_loop_the_prime_could_not_reach_still_gets_its_attach(rig, monkeypatch):
    monkeypatch.setattr(worker_proc_id, "prime", lambda addresses: list(addresses))
    monkeypatch.setitem(mesh_attach._LAST_DIALED, ADDRESS_A, ID_1)
    rig.readings.append(_reading(a=ID_2))
    runtime = rig.build(["HOSTS"])
    assert mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None) == "HOSTS"
    assert rig.log.traced("proc_id_prime")[0]["unprimed"] == ADDRESS_A
    assert rig.dials == [[worker_proc_id.proc_addr(ADDRESS_A, ID_2)]]


def test_every_bookkeeping_key_stays_the_bare_address(rig):
    """Only the dial carries the id. The released check, the trace and the
    operator error all key on the configured address, so a restart that
    changes the id cannot flip the released answer or split one loop's
    trace history in two."""
    rig.readings.append(_reading(a=ID_1, b=ID_2))
    runtime = rig.build([RuntimeError("blip"), RuntimeError("blip")])
    with pytest.raises(RuntimeError, match="attach failed"):
        mesh_attach.attach_cluster(runtime, _config(ADDRESS_A, ADDRESS_B), None)
    assert rig.released.asked == [[ADDRESS_A, ADDRESS_B]]
    assert rig.log.traced("open")[0]["addresses"] == f"{ADDRESS_A},{ADDRESS_B}"
    assert {fields["address"] for fields in rig.log.traced("loop_generation")} == {
        ADDRESS_A, ADDRESS_B}
    assert rig.raised[0][:2] == ("attach", [ADDRESS_A, ADDRESS_B])
    assert all(entry.startswith("service<") for dial in rig.dials for entry in dial)


@pytest.mark.parametrize("fields", [
    {"gen": "none", "proc_id": "none"},
    {"gen": "abc", "proc_id": "bare"},
    {"gen": "abc", "proc_id": "unpublished"},
    {"gen": "unknown", "error": "TimeoutExpired"},
])
def test_an_unknown_id_refuses_before_anything_dials_and_leaves_the_session_usable(
        rig, fields):
    rig.readings.extend([{ADDRESS_A: fields}, _reading(a=ID_1)])
    runtime = rig.build(["HOSTS"])
    handoff = SimpleNamespace(transport_failure_safe=False)
    with pytest.raises(worker_proc_id.WorkerProcIdError, match="no attach ran"):
        mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), handoff)
    assert rig.dials == []
    assert runtime["_TRANSPORT_POISON"] is None
    assert handoff.transport_failure_safe is True
    close = rig.log.traced("close")[0]
    assert (close["outcome"], close["stage"]) == ("refused", "proc_id")
    # The next render reads the id again and attaches on the same transport.
    assert mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None) == "HOSTS"
    assert rig.dials == [[worker_proc_id.proc_addr(ADDRESS_A, ID_1)]]


def test_a_released_fleet_retries_once_when_a_loop_came_back_with_a_new_id(rig):
    """The loop restarted again between the read and the dial. The first
    attempt dialed the old id; the new one can still win, so the released
    fail-fast yields one more attempt for it."""
    rig.released.value = True
    rig.readings.extend([_reading(a=ID_1), _reading(a=ID_2)])
    runtime = rig.build([RuntimeError("stale id"), "HOSTS"])
    assert mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None) == "HOSTS"
    assert len(rig.dials) == 2 and rig.dials[1] == [worker_proc_id.proc_addr(ADDRESS_A, ID_2)]
    assert runtime["_TRANSPORT_POISON"] is None
    assert any("new id" in line for line in rig.log.lines)


def test_a_released_fleet_whose_ids_held_still_fails_fast(rig):
    """Same id after the failed attempt: the loop did not move, so a second
    dial cannot win (an outage longer than the delivery timeout)."""
    rig.released.value = True
    rig.readings.append(_reading(a=ID_1))
    runtime = rig.build([RuntimeError("closed channel")])
    with pytest.raises(RuntimeError, match="released attach failed"):
        mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None)
    assert len(rig.dials) == 1
    assert rig.log.traced("close")[0]["stage"] == "attach_1_released"
    assert rig.log.traced("proc_id_moved") == []


def test_a_reread_that_finds_no_id_after_a_failed_dial_does_not_dial_again(rig):
    """The loop went down during the first attempt. A second dial would only
    spend the config-push budget again, so the attach ends there. The first
    dial already ran, so the session is poisoned, and the error names that
    failure and the loop the reading found no id for, never the pre-dial
    gate's 'no attach ran' or 'stays usable'."""
    rig.readings.extend([_reading(a=ID_1), {ADDRESS_A: {"gen": "none", "proc_id": "none"}}])
    first = RuntimeError("timeout")
    runtime = rig.build([first])
    with pytest.raises(RuntimeError, match="attach failed"):
        mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None)
    assert len(rig.dials) == 1
    kind, _addresses, exc, detail = rig.raised[0]
    assert (kind, exc) == ("attach", first)
    assert detail == {"reread": f"h1 ({ADDRESS_A}: nothing holds its listener)"}
    close = rig.log.traced("close")[0]
    assert (close["outcome"], close["stage"]) == ("poisoned", "attach_1_reread_refused")
    assert [fields.get("attempt") for fields in rig.log.traced("phase")
            if fields["phase"] == "attach"] == ["1"]
    assert runtime["_TRANSPORT_POISON"] == "cluster attach failed: RuntimeError('timeout')"
    assert not any("retrying" in line for line in rig.log.lines)


def test_a_released_reread_that_finds_no_id_says_so_and_still_fails_fast(rig):
    rig.released.value = True
    rig.readings.extend([_reading(a=ID_1), {ADDRESS_A: {"gen": "abc", "proc_id": "bare"}}])
    first = RuntimeError("timeout")
    runtime = rig.build([first])
    with pytest.raises(RuntimeError, match="released attach failed"):
        mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None)
    assert len(rig.dials) == 1
    kind, _addresses, exc, detail = rig.raised[0]
    assert (kind, exc) == ("released", first)
    assert "older release" in detail["reread"]
    assert rig.log.traced("close")[0]["stage"] == "attach_1_released"


def test_the_operator_messages_after_a_failed_reread_tell_the_truth():
    """The dial ran and the session is poisoned, so neither message may say
    that no attach ran or that the session stays usable; both name the loop
    the reading found no id for."""
    reread = f"h1 ({ADDRESS_A}: nothing holds its listener)"
    texts = []
    for raise_it, kwargs in ((mesh_attach.raise_attach, {}),
                             (mesh_attach.raise_released_attach, {"auto_heal": False})):
        with pytest.raises(RuntimeError) as raised:
            raise_it(RuntimeError, [ADDRESS_A], TimeoutError(), reread=reread, **kwargs)
        texts.append(str(raised.value))
    for text in texts:
        assert "TimeoutError()" in text and reread in text
        assert "no attach ran" not in text and "stays usable" not in text
        assert "restart ComfyUI" in text.replace("Restart ComfyUI", "restart ComfyUI")
    assert "no loop came back with a new id" not in texts[1]
    with pytest.raises(RuntimeError, match="no further in-process retry runs"):
        mesh_attach.raise_attach(RuntimeError, [ADDRESS_A], TimeoutError())


def test_a_restarted_loop_the_prime_could_not_reach_is_named_in_a_warning(rig, monkeypatch):
    monkeypatch.setattr(worker_proc_id, "prime", lambda addresses: list(addresses))
    monkeypatch.setitem(mesh_attach._LAST_DIALED, ADDRESS_A, ID_1)
    rig.readings.append(_reading(a=ID_2))
    runtime = rig.build(["HOSTS"])
    assert mesh_attach.attach_cluster(runtime, _config(ADDRESS_A), None) == "HOSTS"
    warned = [line for line in rig.log.lines if "throwaway message" in line]
    assert len(warned) == 1 and ADDRESS_A in warned[0] and worker_proc_id.ENTRY in warned[0]


def test_the_released_message_names_the_delivery_timeout_fallback():
    with pytest.raises(RuntimeError) as raised:
        mesh_attach.raise_released_attach(
            RuntimeError, [ADDRESS_A], TimeoutError(), auto_heal=False)
    text = str(raised.value)
    assert "no loop came back with a new id" in text
    assert "delivery timeout" in text and "Restart ComfyUI" in text
    assert "cannot attach again" not in text
