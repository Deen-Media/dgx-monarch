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
from dgx_monarch.actor.partial_load_guard import TEXT_BYTES
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
    assert text.index("The model on rank 0 was right") < text.index("Nothing was sampled and no rank went on")
    # Once the failing box is named, the close is that box's remedy, not the words for a box it cannot name.
    assert text.rstrip().endswith(rank_readiness.PEER_CAUSE_TAIL)
    assert "a peer rank could not load or verify" not in text and "Fix the cause on that box" not in text
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


def _wire(line: str) -> str:
    """The line as the gather returns it: UTF-8, cut at ``TEXT_BYTES``, zero padding gone."""
    sent = line.encode("utf-8")[:TEXT_BYTES]
    return (sent + b"\0" * (TEXT_BYTES - len(sent))).rstrip(b"\0").decode("utf-8", "ignore")


def _kept(line: str, cause: str) -> str:
    """What of ``cause`` the gathered line keeps, checked whole: it fits, parses and is a prefix."""
    assert len(line.encode("utf-8")) <= TEXT_BYTES and _wire(line) == line
    wrong = json.loads(_wire(line))["wrong"]
    assert wrong.endswith(backup_agreement.CUT_MARK)
    kept = wrong[:-len(backup_agreement.CUT_MARK)]
    assert cause.startswith(kept)
    return kept


def test_a_short_cause_crosses_whole():
    mine = backup_agreement.Vote(rank=1, room=9 * GIB, need=7 * GIB)
    facts = json.loads(backup_agreement._facts(mine, "ComfyUI holds its weights as unpatched"))
    assert facts["wrong"] == "ComfyUI holds its weights as unpatched" and facts["rank"] == 1


@pytest.mark.parametrize(("cause", "last"), [
    ('the key "blocks.0\\to_q" is short\n' * 60, None),  # quotes, backslashes and newlines grow when escaped
    ("é漢🙂" * 400, None),  # two, three and four bytes a character; the emoji escapes to twelve characters
    ("a" * 800 + "🙂" * 40, "🙂"),  # the cut falls among the four-byte characters
], ids=["escapes", "multi-byte", "cut-among-emoji"])
def test_a_long_cause_is_cut_at_a_character_and_marked(cause, last):
    mine = backup_agreement.Vote(rank=1, room=9 * GIB, need=7 * GIB, unmapped=2 * GIB)
    line = backup_agreement._facts(mine, cause)
    kept = _kept(line, cause)
    # As many whole characters as fit: the next one would not.
    assert kept and len(line) + len(json.dumps(cause[len(kept)])) - 2 > TEXT_BYTES
    assert last is None or kept.endswith(last)
    facts = json.loads(line)
    assert (facts["rank"], facts["room"], facts["need"], facts["unmapped"]) == (1, 9 * GIB, 7 * GIB, 2 * GIB)


def test_a_cut_cause_leads_the_peers_refusal_with_its_mark(rig):  # noqa: F811
    cause = "é漢🙂 the key is short; " * 120
    rig.room = 30.0
    healthy = backup_agreement.vote((_patcher(),), SAMPLING, 0)
    peer = json.loads(backup_agreement._facts(backup_agreement.Vote(rank=1, room=24 * GIB, need=7 * GIB), cause))

    def gather(text):
        return [text, json.dumps(peer)]

    with pytest.raises(rank_readiness.PeerLoadFailureError) as raised:
        backup_agreement.settle(healthy, _answer(healthy.elements, _peer(not_short=0)),
                                _second(0, backup_agreement._margin(24 * GIB, 7 * GIB, 1)), gather)
    text, marked = read_rank(str(raised.value))
    assert marked == 1 and backup_agreement.CUT_MARK in text
    start = text.index("freed their weight backup: ") + len("freed their weight backup: ")
    assert cause.startswith(text[start:text.index(backup_agreement.CUT_MARK)])


def test_the_facts_never_raise():
    class Broken:
        rank = "not a rank"

    assert backup_agreement._facts(Broken(), "x" * 5000) == "{}"  # type: ignore[arg-type]
    # A NUL escapes in the line, so the gather's zero padding never eats the cause.
    mine = backup_agreement.Vote(rank=1, room=9 * GIB, need=7 * GIB)
    assert _kept(backup_agreement._facts(mine, "\x00" * 5000), "\x00" * 5000)
