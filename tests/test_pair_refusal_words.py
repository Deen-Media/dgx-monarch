"""The pair's refusal speaks in the words of the rank that is short or wrong.

MIN carries agreement and not identity, and the driver receives one rank's
error for the fleet, whichever arrives first. So a refusal after the agreement
crosses each rank's own facts once (``partial_load_guard._all_ranks_texts``),
and every rank's error leads with the causing rank's cause and figures and
carries its mark. These tests answer the exchange and the gather for the peer,
on the CPU, with the rig of ``tests/test_backup_agreement.py``.
"""
from __future__ import annotations

import json

import pytest

from dgx_monarch.actor import backup_agreement, lora_partial_load, rank_readiness
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag, read_rank
from test_backup_agreement import GIB, SAMPLING, _answer, _patcher, _peer, rig  # noqa: F401  (rig: the fixture)


def _gather(rank: int, peer: dict, calls: list):
    """A world of two whose other rank sends ``peer``: the answer in rank order."""
    def gather(text):
        calls.append(text)
        return [text, json.dumps(peer)] if rank == 0 else [json.dumps(peer), text]
    return gather


def _second(peer_ok: int, peer_margin: int):
    def exchange(values):
        return _answer(values, [peer_ok, peer_margin])
    return exchange


def test_the_wrong_rank_logs_its_own_refusal_and_its_peer_leads_with_it(rig, caplog):  # noqa: F811
    cause = "ComfyUI holds its weights as unpatched"
    # Rank 1: short, frees, and finds its model wrong.
    rig.room, rig.room_after, rig.wrong = 9.0, 24.0, cause
    wrong_side = backup_agreement.vote((_patcher(),), SAMPLING, 1)
    calls: list = []
    with caplog.at_level("WARNING"), pytest.raises(lora_partial_load.OffloadedInputQuantError) as own:
        backup_agreement.settle(wrong_side, _answer(wrong_side.elements, _peer()),
                                _second(1, backup_agreement._margin(30 * GIB, 7 * GIB, 0)),
                                _gather(1, {"rank": 0, "wrong": "", "room": 30 * GIB, "need": 7 * GIB}, calls))
    assert len(calls) == 1 and json.loads(calls[0])["wrong"] == cause
    assert cause in str(own.value) and "after rank 1 freed its weight backup" in str(own.value)
    assert "this rank refuses for the fleet" in caplog.text and cause in caplog.text
    # Rank 0: its model is right, and its refusal leads with rank 1's words and carries rank 1's mark.
    rig.wrong, rig.room, rig.room_after = None, 30.0, None
    healthy = backup_agreement.vote((_patcher(),), SAMPLING, 0)
    calls = []
    with pytest.raises(rank_readiness.PeerLoadFailureError) as peer:
        backup_agreement.settle(healthy, _answer(healthy.elements, _peer(not_short=0)),
                                _second(0, backup_agreement._margin(24 * GIB, 7 * GIB, 1)),
                                _gather(0, {"rank": 1, "wrong": cause, "room": 24 * GIB, "need": 7 * GIB}, calls))
    text, marked = read_rank(str(peer.value))
    tag = parse_leading_refusal_tag(text)
    assert marked == 1 and tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert text.index("rank 1 found its model is not as a full load leaves it") < text.index(cause)
    assert text.index("The model on rank 0 was right") < text.index("a peer rank could not load or verify")
    assert len(calls) == 1


def test_a_room_refusal_gives_the_short_ranks_figures_whichever_rank_raises(rig, caplog):  # noqa: F811
    """Rank 0 is roomy and raises for the fleet: the words are rank 1's, then its own."""
    rig.room = 40.0
    roomy = backup_agreement.vote((_patcher(),), SAMPLING, 0)
    short = {"rank": 1, "wrong": "", "room": 6 * GIB, "need": int((1.15 * 6.2 + 1) * GIB)}
    calls: list = []
    with caplog.at_level("INFO"), pytest.raises(backup_agreement.PairBackupRoomError) as raised:
        backup_agreement.settle(roomy, _answer(roomy.elements, _peer(not_short=0)),
                                _second(1, backup_agreement._margin(6 * GIB, short["need"], 1)),
                                _gather(0, short, calls))
    text, marked = read_rank(str(raised.value))
    assert marked == 1 and len(calls) == 1
    assert "rank 1 is 2.1 GiB short" in text
    assert text.index("Rank 1 read 6.0 GiB of room") < text.index("rank 0 read 40.0 GiB of room")
    assert "This rank read" not in text and "rank 1 is short, so the fleet refuses" in caplog.text


def test_the_short_rank_logs_its_refusal_at_warning(rig, caplog):  # noqa: F811
    rig.room, rig.room_after = 2.0, 6.0
    mine = backup_agreement.vote((_patcher(),), SAMPLING, 0)
    roomy = {"rank": 1, "wrong": "", "room": 40 * GIB, "need": 7 * GIB}
    with caplog.at_level("WARNING"), pytest.raises(backup_agreement.PairBackupRoomError):
        backup_agreement.settle(mine, _answer(mine.elements, _peer()),
                                _second(1, backup_agreement._margin(40 * GIB, 7 * GIB, 1)),
                                _gather(0, roomy, []))
    assert "this rank is the one short, so the fleet refuses" in caplog.text and "Rank 0 read 6.0" in caplog.text


def test_a_render_that_samples_crosses_no_words(rig):  # noqa: F811
    rig.room, rig.room_after = 9.0, 24.0
    mine = backup_agreement.vote((_patcher(),), SAMPLING, 0)
    calls: list = []
    backup_agreement.settle(mine, _answer(mine.elements, _peer()),
                            _second(1, backup_agreement._margin(30 * GIB, 7 * GIB, 1)),
                            _gather(0, {"rank": 1}, calls))
    roomy = backup_agreement.vote((_patcher(),), SAMPLING, 0)
    backup_agreement.settle(roomy, _answer(roomy.elements, _peer()), lambda values: values, _gather(0, {}, calls))
    assert calls == []


def test_a_gather_that_fails_leaves_each_rank_its_own_words(rig, caplog):  # noqa: F811
    rig.room, rig.room_after = 2.0, 6.0
    mine = backup_agreement.vote((_patcher(),), SAMPLING, 0)

    def broken(_text):
        raise RuntimeError("no group")

    with caplog.at_level("WARNING"), pytest.raises(backup_agreement.PairBackupRoomError) as raised:
        backup_agreement.settle(mine, _answer(mine.elements, _peer()),
                                _second(1, backup_agreement._margin(40 * GIB, 7 * GIB, 1)), broken)
    assert "This rank read 6.0 GiB of room" in str(raised.value) and read_rank(str(raised.value))[1] is None
    assert "could not be crossed" in caplog.text


def test_a_peer_with_unreadable_words_still_refuses_in_the_plain_words():
    with pytest.raises(rank_readiness.PeerLoadFailureError) as raised:
        rank_readiness.peer_failed()
    assert read_rank(str(raised.value))[1] is None and str(raised.value).endswith(
        rank_readiness.PEER_LOAD_TEXT)
