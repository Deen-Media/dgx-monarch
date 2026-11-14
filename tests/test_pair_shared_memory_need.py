"""The pair's need counts the shared memory a rank has yet to map.

A Worker of a pair maps about 5.5 GiB of shared memory, about half of it in
the first seconds of its first forward, after the agreement has weighed the
room; ComfyUI's estimate does not count it. The need and the ask therefore add
``PAIR_SHARED_MEMORY_BYTES`` less what the Worker maps now, never under zero,
on a rank on a CUDA device. These tests use the rig of
``tests/test_backup_agreement.py`` with a stand-in integrated device.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from dgx_monarch.actor import backup_agreement, lora_partial_load
from test_backup_agreement import GIB, SAMPLING, _answer, _patcher, _peer, rig  # noqa: F401  (rig: the fixture)

BASE_NEED = int(1.15 * int(6.2 * GIB)) + GIB  # the rig's estimate, as the vote computes it


def _cuda_rank(monkeypatch, mapped_gib: float | None):
    """A patcher on a stand-in integrated CUDA device whose Worker maps ``mapped_gib`` of shared memory."""
    monkeypatch.setattr(lora_partial_load, "shares_host_memory", lambda patcher: True)
    monkeypatch.setattr(backup_agreement, "_mapped_shared_bytes",
                        lambda: None if mapped_gib is None else int(mapped_gib * GIB))
    patcher = _patcher()
    patcher.load_device = SimpleNamespace(type="cuda")
    return patcher


@pytest.mark.parametrize("mapped,unmapped", [
    (0.0, 5.6),      # a fresh Worker: nothing mapped yet
    (2.9, 2.7),      # after its load's end: the forward maps the rest
    (5.6, 0.0),      # a warm Worker: asked nothing more
    (7.0, 0.0),      # never under zero
    (None, 5.6),     # /proc cannot say: the whole figure
])
def test_the_need_and_the_ask_carry_what_the_rank_has_yet_to_map(rig, monkeypatch, mapped, unmapped):  # noqa: F811
    vote = backup_agreement.vote((_cuda_rank(monkeypatch, mapped),), SAMPLING, 0)
    assert vote.unmapped == pytest.approx(unmapped * GIB, abs=GIB / 1000)
    assert vote.need == BASE_NEED + vote.unmapped
    assert vote.ask == int(18.6 * GIB) + vote.unmapped


def test_a_fresh_rank_is_short_where_a_warm_one_is_not(rig, monkeypatch):  # noqa: F811
    rig.room = 20.0  # above the 18.6 GiB ask, under 18.6 + 2.7
    assert backup_agreement.vote((_cuda_rank(monkeypatch, 2.9),), SAMPLING, 0).short is True
    assert backup_agreement.vote((_cuda_rank(monkeypatch, 5.6),), SAMPLING, 0).short is False


def test_a_rank_off_a_cuda_device_carries_no_term(rig, monkeypatch):  # noqa: F811
    monkeypatch.setattr(backup_agreement, "_mapped_shared_bytes", lambda: 0)
    vote = backup_agreement.vote((_patcher(),), SAMPLING, 0)
    assert vote.unmapped == 0 and vote.need == BASE_NEED


def test_the_margin_after_a_free_weighs_the_term(rig, monkeypatch):  # noqa: F811
    """Freed to 12 GiB of room: enough for 8.1 GiB, not for 8.1 plus the 5.6 a fresh Worker has yet to map."""
    rig.room, rig.room_after = 9.0, 12.0
    mine = backup_agreement.vote((_cuda_rank(monkeypatch, 0.0),), SAMPLING, 0)
    with pytest.raises(backup_agreement.PairBackupRoomError) as raised:
        backup_agreement.settle(mine, _answer(mine.elements, _peer()),
                                lambda values: _answer(values, [1, backup_agreement._margin(40 * GIB, 7 * GIB, 1)]))
    message = str(raised.value)
    assert "rank 0 is 1.7 GiB short" in message
    assert "and 5.6 GiB of shared memory the rank had yet to map, which ComfyUI's estimate does not count" in message


def test_a_warm_rank_passes_the_same_margin(rig, monkeypatch):  # noqa: F811
    rig.room, rig.room_after = 9.0, 12.0
    mine = backup_agreement.vote((_cuda_rank(monkeypatch, 5.6),), SAMPLING, 0)
    backup_agreement.settle(mine, _answer(mine.elements, _peer()),
                            lambda values: _answer(values, [1, backup_agreement._margin(40 * GIB, 7 * GIB, 1)]))


def test_the_crossed_figures_name_the_term(rig, monkeypatch):  # noqa: F811
    import json

    rig.room = 40.0
    roomy = backup_agreement.vote((_patcher(),), SAMPLING, 0)
    short = {"rank": 1, "wrong": "", "room": 9 * GIB, "need": BASE_NEED + int(2.7 * GIB),
             "unmapped": int(2.7 * GIB)}
    with pytest.raises(backup_agreement.PairBackupRoomError) as raised:
        backup_agreement.settle(roomy, _answer(roomy.elements, _peer(not_short=0)),
                                lambda values: _answer(values, [1, backup_agreement._margin(
                                    9 * GIB, short["need"], 1)]),
                                lambda text: [text, json.dumps(short)])
    assert "Rank 1 read 9.0 GiB of room (its idle device pool counted) against 10.8 GiB needed (1.15 times" in (
        str(raised.value))
    assert "and 2.7 GiB of shared memory the rank had yet to map" in str(raised.value)


def test_the_reading_of_this_process_is_a_count_of_bytes():
    mapped = backup_agreement._mapped_shared_bytes()
    assert mapped is None or (isinstance(mapped, int) and mapped >= 0)
