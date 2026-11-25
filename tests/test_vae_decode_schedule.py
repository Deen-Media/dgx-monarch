"""The two-ended unit board decides who decodes each unit, and nothing else.

The driver takes units from the front in the stock loop's order; Worker ranks
claim from the back. These tests pin the claims, the lookahead that keeps the
driver's next unit at home, more ranks than units and the reverse, a single
unit, and every way a rank stops: a raise, a decline, a reply the loop refuses
and silence. A unit a rank gives up is always decoded here, so the output is
the reference either way (tests/test_vae_decode_split_comfy.py holds the bytes).
The reply budget holds a rank's claims to a window ahead of the loop and to a
byte total, stops it while full and resumes it as the loop drains replies.
"""
from __future__ import annotations

import threading
import time

import pytest

from dgx_monarch.vae_decode_schedule import UnitBoard, pump


def test_ranks_claim_from_the_back_and_never_the_drivers_next_unit():
    board = UnitBoard(5)
    assert [board.claim(1), board.claim(2), board.claim(1), board.claim(2)] == [4, 3, 2, 1]
    assert board.claim(1) is None  # unit 0 is the driver's next unit
    assert board.take(0, max_wait_s=1) == (False, None)
    board.deliver(1, 2, "one")
    assert board.take(1, max_wait_s=1) == (True, "one")
    assert board.stats.local == 1 and board.stats.remote == {2: 1}


def test_more_ranks_than_units_and_a_single_unit():
    board = UnitBoard(2)
    assert board.claim(1) == 1
    assert board.claim(2) is None and board.claim(3) is None
    single = UnitBoard(1)
    assert single.claim(1) is None
    assert single.take(0, max_wait_s=1) == (False, None)
    assert single.stats.local == 1 and not single.stats.remote


def test_a_unit_the_driver_passed_is_never_claimed():
    board = UnitBoard(4)
    assert board.take(0, max_wait_s=1) == (False, None)
    assert board.take(1, max_wait_s=1) == (False, None)
    assert board.claim(1) == 3
    assert board.claim(1) is None


def test_a_failed_rank_hands_back_every_unit_it_holds():
    board = UnitBoard(6)
    held = [board.claim(1), board.claim(1), board.claim(2)]
    assert held == [5, 4, 3]
    board.fail(1, "rank 1 declined: no file")
    assert board.claim(1) is None
    assert board.take(4, max_wait_s=1) == (False, None)
    assert board.take(5, max_wait_s=1) == (False, None)
    assert board.stats.fallback == 2
    assert [failure.cause for failure in board.stats.failures] == ["rank 1 declined: no file"]


def test_a_silent_rank_is_retired_after_the_wait():
    board = UnitBoard(3)
    assert board.claim(7) == 2
    started = time.monotonic()
    assert board.take(2, max_wait_s=0.3) == (False, None)
    assert time.monotonic() - started >= 0.3
    assert board.stats.failures[0].rank == 7 and "no answer" in board.stats.failures[0].cause
    board.deliver(2, 7, "late")  # a late answer changes nothing
    assert board.stats.remote == {}


def test_a_rejected_result_counts_as_decoded_here():
    board = UnitBoard(3)
    assert board.claim(1) == 2
    board.deliver(2, 1, "wrong dtype")
    assert board.take(2, max_wait_s=1) == (True, "wrong dtype")
    board.reject(2, "unit 2 came back as float16")
    assert board.stats.remote == {1: 0} and board.stats.local == 1 and board.stats.fallback == 1
    assert board.claim(1) is None


def test_an_interrupt_while_waiting_propagates():
    board = UnitBoard(3)
    assert board.claim(1) == 2

    class Interrupted(Exception):
        pass

    def interrupt():
        raise Interrupted

    with pytest.raises(Interrupted):
        board.take(2, max_wait_s=5, check_interrupt=interrupt)


def test_close_stops_claims():
    board = UnitBoard(4)
    board.close()
    assert board.closed and board.claim(1) is None


def _run(board, rank, replies, **kwargs):
    sent = []

    def send(rank_, unit):
        sent.append(unit)
        return unit

    def wait(handle, timeout_s):
        value = replies(handle)
        if isinstance(value, BaseException):
            raise value
        return value

    pump(board, rank, send, wait, first_timeout_s=1, unit_timeout_s=1, **kwargs)
    return sent


def test_the_pump_feeds_a_rank_until_nothing_is_left():
    board = UnitBoard(5)
    sent = _run(board, 1, lambda unit: {"output": unit * 10}, unwrap=lambda reply: reply["output"])
    assert sent == [4, 3, 2, 1]
    assert [board.take(unit, max_wait_s=1) for unit in range(5)] == [
        (False, None), (True, 10), (True, 20), (True, 30), (True, 40)]


def test_the_pump_retires_its_rank_on_a_raise_or_a_refused_reply():
    board = UnitBoard(5)
    _run(board, 1, lambda unit: RuntimeError("actor on the sibling died") if unit == 3 else unit)
    assert board.stats.failures[0].cause == "unit 3: RuntimeError: actor on the sibling died"
    assert board.take(3, max_wait_s=1) == (False, None)
    other = UnitBoard(4)
    _run(other, 2, lambda unit: unit, accept=lambda reply: "rank 2 declined: busy")
    assert other.stats.failures[0].cause == "rank 2 declined: busy"
    assert all(other.take(unit, max_wait_s=1) == (False, None) for unit in range(4))


def test_the_first_wait_gets_the_load_budget():
    board = UnitBoard(4)
    budgets = []

    def wait(handle, timeout_s):
        budgets.append(timeout_s)
        return handle

    pump(board, 1, lambda rank, unit: unit, wait, first_timeout_s=600, unit_timeout_s=300)
    assert budgets == [600, 300, 300]


def test_two_ranks_and_the_driver_share_one_board_without_loss():
    board = UnitBoard(40)
    lock = threading.Lock()
    done = []

    def worker(rank):
        def wait(handle, timeout_s):
            time.sleep(0.002)
            with lock:
                done.append((rank, handle))
            return handle
        pump(board, rank, lambda _rank, unit: unit, wait, first_timeout_s=5, unit_timeout_s=5)

    threads = [threading.Thread(target=worker, args=(rank,)) for rank in (1, 2)]
    for thread in threads:
        thread.start()
    seen = []
    for unit in range(40):
        remote, value = board.take(unit, max_wait_s=5)
        seen.append(value if remote else unit)
        time.sleep(0.003)
    for thread in threads:
        thread.join(5)
    assert seen == list(range(40))
    assert board.stats.local + sum(board.stats.remote.values()) == 40
    assert len({unit for _rank, unit in done}) == len(done)


def test_the_budget_holds_claims_to_a_window_and_resumes_as_the_loop_drains():
    board = UnitBoard(8, unit_bytes=[10] * 8, budget_bytes=30)  # three units ahead of the loop
    assert [board.claim(1), board.claim(1), board.claim(1)] == [3, 2, 1]
    assert board.claim(1) is None and board.reserved_bytes == 30
    resumed = []
    waiter = threading.Thread(target=lambda: resumed.append(board.wait_for_room(1)))
    waiter.start()
    time.sleep(0.3)
    assert waiter.is_alive() and board.stats.budget_waits == 1
    assert board.take(0, max_wait_s=1) == (False, None)  # the window moves, but unit 4 does not fit
    time.sleep(0.3)
    assert waiter.is_alive()
    board.deliver(1, 1, "one")
    assert board.take(1, max_wait_s=1) == (True, "one")  # the loop drains a reply below the budget
    waiter.join(2)
    assert resumed == [True]
    assert board.claim(1) == 5 and board.reserved_bytes == 30
    assert board.stats.peak_bytes == 30


def test_a_unit_larger_than_the_budget_still_runs_alone():
    board = UnitBoard(4, unit_bytes=[50] * 4, budget_bytes=30)
    assert board.claim(1) == 1
    assert board.claim(1) is None


def test_a_failed_rank_returns_its_reserved_bytes():
    board = UnitBoard(6, unit_bytes=[10] * 6, budget_bytes=20)
    assert [board.claim(1), board.claim(1)] == [2, 1]
    board.fail(1, "rank 1 declined: busy")
    assert board.reserved_bytes == 0
    assert board.claim(2) is None  # the failed units stay with the driver
    assert board.take(1, max_wait_s=1) == (False, None)


def test_with_no_units_left_the_waiting_rank_stops():
    board = UnitBoard(3, unit_bytes=[10] * 3, budget_bytes=10)
    assert board.claim(1) == 1
    for unit in range(3):
        board.take(unit, max_wait_s=0.1)
    assert board.wait_for_room(1) is False


def test_two_ranks_under_a_budget_never_hold_more_than_it():
    board = UnitBoard(40, unit_bytes=[10] * 40, budget_bytes=40)
    lock = threading.Lock()
    done = []

    def worker(rank):
        def wait(handle, timeout_s):
            time.sleep(0.004)
            with lock:
                done.append(handle)
            return handle
        pump(board, rank, lambda _rank, unit: unit, wait, first_timeout_s=5, unit_timeout_s=5)

    threads = [threading.Thread(target=worker, args=(rank,)) for rank in (1, 2)]
    for thread in threads:
        thread.start()
    seen = []
    for unit in range(40):
        remote, value = board.take(unit, max_wait_s=5)
        seen.append(value if remote else unit)
        time.sleep(0.004)
    for thread in threads:
        thread.join(5)
    assert seen == list(range(40))
    assert board.stats.peak_bytes <= 40
    assert sum(board.stats.remote.values()) > 4  # the ranks kept working past the first window
    assert board.stats.local + sum(board.stats.remote.values()) == 40
