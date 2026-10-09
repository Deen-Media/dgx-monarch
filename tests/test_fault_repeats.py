"""The repeat gate behind the unhandled fault hook.

The hook-level behavior (one whole report, one toast, counted repeats) is
pinned beside the other hook tests in ``test_teardown_and_preflight.py``.
"""
from __future__ import annotations

import threading

from dgx_monarch import fault_repeats

_FAULT = ("MeshFailure(event=Supervision event: actor <client> failed:\n"
          "  undeliverable message to cast.anon-0<abc>:castmessage<def>\n"
          "  \terror: delivery failure: ttl expired for cast.anon-0<abc>\n)")


class _Log:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self._lock = threading.Lock()

    def error(self, message: str, *args) -> None:
        with self._lock:
            self.lines.append(message % args)


def test_the_first_report_is_whole_and_repeats_are_counted_at_powers_of_two():
    gate, log = fault_repeats.RepeatGate(), _Log()
    verdicts = [gate.first_report(_FAULT, log) for _ in range(792)]
    assert verdicts[0] is True and not any(verdicts[1:])
    # The 2026-10-04 session, 792 identical faults: nine lines instead of 791.
    assert [line.split(" times: ")[0] for line in log.lines] == [
        f"cluster fault repeated {count}" for count in (2, 4, 8, 16, 32, 64, 128, 256, 512)]
    assert all("undeliverable message to cast.anon-0<abc>" in line for line in log.lines)


def test_a_different_fault_is_news_and_starts_a_new_count():
    gate, log = fault_repeats.RepeatGate(), _Log()
    other = _FAULT.replace("abc", "xyz")
    assert gate.first_report(_FAULT, log) is True
    assert gate.first_report(_FAULT, log) is False
    assert gate.first_report(other, log) is True
    # The earlier fault returning is a change too, so it is reported whole.
    assert gate.first_report(_FAULT, log) is True
    assert gate.first_report(_FAULT, log) is False
    assert [line.split(" times: ")[0] for line in log.lines] == [
        "cluster fault repeated 2", "cluster fault repeated 2"]


def test_two_gates_do_not_share_a_count():
    first, second, log = fault_repeats.RepeatGate(), fault_repeats.RepeatGate(), _Log()
    assert first.first_report(_FAULT, log) is True
    assert second.first_report(_FAULT, log) is True
    assert log.lines == []


def test_the_headline_names_the_failed_actor_and_the_reason():
    assert fault_repeats.headline(_FAULT) == (
        "MeshFailure(event=Supervision event: actor <client> failed: "
        "undeliverable message to cast.anon-0<abc>:castmessage<def>")
    assert fault_repeats.headline("one line") == "one line"
    assert len(fault_repeats.headline("x" * 500)) == 200


def test_concurrent_reports_lose_no_repeat():
    gate, log = fault_repeats.RepeatGate(), _Log()
    whole: list[bool] = []
    whole_lock = threading.Lock()

    def report() -> None:
        for _ in range(8):
            verdict = gate.first_report(_FAULT, log)
            with whole_lock:
                whole.append(verdict)

    threads = [threading.Thread(target=report) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert whole.count(True) == 1 and len(whole) == 64
    assert sorted(int(line.split()[3]) for line in log.lines) == [2, 4, 8, 16, 32, 64]
