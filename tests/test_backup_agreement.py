"""The pair's backup agreement: the ranks free ComfyUI's weight backup together or not at all.

``actor/backup_agreement.py`` decides from what the guard's MIN exchange
brings back, and ``actor/partial_load_guard.py`` owns that exchange. These
tests stand in for the peer by answering the exchange, on the CPU, with no
ComfyUI: the patchers are plain objects and the two helpers that need ComfyUI
(the estimate and the free) are replaced.
"""
from __future__ import annotations

import weakref
from types import SimpleNamespace

import pytest

from dgx_monarch import mesh_builds
from dgx_monarch.actor import backup_agreement, lora_partial_load, partial_load_guard, rank_readiness
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag

GIB = 2 ** 30
SAMPLING = ((1, 16, 22, 104, 60), {"positive": []})
NEED = 1.15 * 6.2 + 1.0  # GiB: 1.15 times the estimate the rig gives, plus 1


class _Slot:
    def __init__(self):
        self.base_baked, self.unbake = False, None


def _patcher(backup=True, stored=True):
    slot = _Slot()
    model = SimpleNamespace(current_weight_patches_uuid="stack", model_lowvram=False)
    if stored:
        setattr(model, lora_partial_load.STORED, weakref.ref(slot))
    patcher = SimpleNamespace(model=model, backup={"w": object()} if backup else {}, patches={"w": [1]},
                              load_device=SimpleNamespace(type="cpu"), is_dynamic=lambda: False)
    patcher.slot = slot
    return patcher


@pytest.fixture
def rig(monkeypatch):
    """Room, ask and estimate set by the test; the free recorded, not run."""
    state = SimpleNamespace(room=20.0, ask=18.6, estimate=6.2, freed=[], wrong=None, room_after=None)

    def free(patcher, stored):
        state.freed.append(patcher)
        patcher.backup.clear()
        stored.base_baked = True
        if state.room_after is not None:
            state.room = state.room_after
        return "MemAvailable read 12.0 GiB before the free and 27.0 GiB after it (host trim 0.01 s)"

    monkeypatch.delenv(lora_partial_load.BACKUP_FREE_DISABLE_ENV, raising=False)
    monkeypatch.setattr(lora_partial_load, "_host_room", lambda: int(state.room * GIB))
    monkeypatch.setattr(lora_partial_load, "_sampling_ask",
                        lambda patcher, shape, conds: None if state.ask is None else int(state.ask * GIB))
    monkeypatch.setattr(backup_agreement, "_single_estimate",
                        lambda patcher, shape, conds: None if state.estimate is None else int(state.estimate * GIB))
    monkeypatch.setattr(lora_partial_load, "_free_backup", free)
    monkeypatch.setattr(lora_partial_load, "_what_is_wrong_after_the_route", lambda patcher, was: state.wrong)
    return state


def _peer(**sent):
    """A roomy peer's elements unless told otherwise: not short, can, switch off, a wide margin, none freed."""
    return [sent.get("not_short", 1), sent.get("can", 1), sent.get("switch_off", 1),
            sent.get("margin", backup_agreement._margin(60 * GIB, 7 * GIB, 1)), sent.get("not_freed", 1)]


def _answer(mine, peer):
    return [min(a, b) for a, b in zip(mine, peer, strict=True)]


def test_the_margin_carries_the_tightest_rank_through_a_min():
    tight = backup_agreement._margin(int(3.5 * GIB), int(7.2 * GIB), 0)
    roomy = backup_agreement._margin(40 * GIB, int(7.2 * GIB), 1)
    gib, rank = backup_agreement._read_margin(min(tight, roomy))
    assert rank == 0 and gib == pytest.approx(-3.7, abs=0.01)
    no_claim = backup_agreement._margin(0, None, 5)  # a rank that cannot tell claims room, not a shortfall
    assert backup_agreement._read_margin(no_claim) == (1024.0, 5)
    assert abs(tight) < 2 ** 31 and abs(backup_agreement._margin(0, None, 63)) < 2 ** 31  # an int32 carries it


def test_a_rank_votes_what_it_read_and_never_raises(rig, monkeypatch):
    patcher = _patcher()
    roomy = backup_agreement.vote((patcher,), SAMPLING, 0)
    assert roomy.elements[:3] == [1, 1, 1] and roomy.freeable == [(patcher, patcher.slot)]
    rig.room = 9.0
    short = backup_agreement.vote((patcher,), SAMPLING, 0)
    assert short.short is True and short.elements[:3] == [0, 1, 1]
    assert backup_agreement._read_margin(short.elements[3]) == (pytest.approx(9.0 - NEED, abs=0.01), 0)
    # A rank with no backup is never short, whatever its room: it has nothing the step could free.
    assert backup_agreement.vote((_patcher(backup=False),), SAMPLING, 0).elements[:3] == [1, 1, 1]
    # A patched model outside the store cannot be freed: the rank says so and keeps the count of elements.
    outside = backup_agreement.vote((_patcher(stored=False),), SAMPLING, 1)
    assert outside.can is False and outside.elements[:3] == [0, 0, 1] and "store" in outside.why_not
    monkeypatch.setenv(lora_partial_load.BACKUP_FREE_DISABLE_ENV, "1")
    assert backup_agreement.vote((patcher,), SAMPLING, 0).elements[2] == 0
    monkeypatch.delenv(lora_partial_load.BACKUP_FREE_DISABLE_ENV)

    def breaks():
        raise OSError("no reading")

    monkeypatch.setattr(lora_partial_load, "_host_room", breaks)
    blind = backup_agreement.vote((patcher,), SAMPLING, 0)
    assert len(blind.elements) == backup_agreement.ELEMENTS and blind.elements[:2] == [1, 0]
    assert backup_agreement._read_margin(blind.elements[3])[0] > 0 and "could not weigh" in blind.why_not


def test_nobody_short_frees_nothing_and_makes_no_second_exchange(rig):
    patcher = _patcher()
    mine = backup_agreement.vote((patcher,), SAMPLING, 0)
    calls = []
    backup_agreement.settle(mine, _answer(mine.elements, _peer()), lambda values: calls.append(values) or values)
    assert rig.freed == [] and calls == [] and patcher.backup


@pytest.mark.parametrize("who", ["this rank", "a peer"])
def test_any_rank_short_and_every_rank_frees(rig, who, caplog):
    patcher = _patcher()
    if who == "this rank":
        rig.room, rig.room_after = 9.0, 24.0
    mine = backup_agreement.vote((patcher,), SAMPLING, 0)
    peer = _peer(not_short=1 if who == "this rank" else 0)
    calls = []

    def exchange(values):
        calls.append(list(values))
        return _answer(values, [1, backup_agreement._margin(30 * GIB, 7 * GIB, 1)])

    with caplog.at_level("INFO"):
        backup_agreement.settle(mine, _answer(mine.elements, peer), exchange)
    assert rig.freed == [patcher] and patcher.backup == {} and patcher.slot.base_baked is True
    assert len(calls) == 1 and calls[0][0] == 1  # one more exchange, and this rank's model is as it should be
    assert "every rank freed ComfyUI's weight backup" in caplog.text
    assert ("this rank was short" if who == "this rank" else "a peer was short") in caplog.text
    assert "MemAvailable read 12.0 GiB before the free and 27.0 GiB after it" in caplog.text


def test_a_rank_that_cannot_free_keeps_the_fleet_and_says_what_stands(rig, caplog):
    patcher = _patcher()
    rig.room = 9.0
    mine = backup_agreement.vote((patcher,), SAMPLING, 0)
    calls = []
    with caplog.at_level("INFO"):
        backup_agreement.settle(mine, _answer(mine.elements, _peer(can=0)), lambda values: calls.append(values))
    assert rig.freed == [] and calls == [] and patcher.backup
    assert "every rank keeps its own" in caplog.text and "left the backup of 1 weight(s) standing" in caplog.text


def test_a_kept_fleet_that_is_short_of_the_samplers_need_refuses_in_words(rig):
    patcher = _patcher()
    rig.room = 3.5  # under the 8.1 GiB the sampler still needs
    mine = backup_agreement.vote((patcher,), SAMPLING, 0)
    with pytest.raises(backup_agreement.PairBackupRoomError) as raised:
        backup_agreement.settle(mine, _answer(mine.elements, _peer(can=0)), lambda values: values)
    message = str(raised.value)
    tag = parse_leading_refusal_tag(message)
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY
    assert "every rank kept its weight backup" in message and "rank 0 is 4.6 GiB short" in message
    assert "lora_low_rss on" in message and backup_agreement.PairBackupRoomError.fleet_wide is True


def test_a_free_that_leaves_too_little_refuses_on_what_it_left(rig):
    """The room is read after the free: a free that returns less than expected is caught by the reading."""
    patcher = _patcher()
    rig.room, rig.room_after = 2.0, 6.0  # freed, and still under 8.1
    mine = backup_agreement.vote((patcher,), SAMPLING, 0)
    sent = []

    def exchange(values):
        sent.append(list(values))
        return _answer(values, [1, backup_agreement._margin(30 * GIB, 7 * GIB, 1)])

    with pytest.raises(backup_agreement.PairBackupRoomError) as raised:
        backup_agreement.settle(mine, _answer(mine.elements, _peer()), exchange)
    assert rig.freed == [patcher]
    assert backup_agreement._read_margin(sent[0][1]) == (pytest.approx(6.0 - NEED, abs=0.01), 0)
    assert "every rank freed its weight backup" in str(raised.value) and "rank 0 is 2.1 GiB short" in str(raised.value)


def test_a_model_found_wrong_after_the_free_stops_every_rank(rig):
    patcher = _patcher()
    rig.room, rig.room_after, rig.wrong = 9.0, 24.0, "ComfyUI holds its weights as unpatched"
    mine = backup_agreement.vote((patcher,), SAMPLING, 0)
    sent = []

    def exchange(values):
        sent.append(list(values))
        return _answer(values, [1, backup_agreement._margin(30 * GIB, 7 * GIB, 1)])

    with pytest.raises(lora_partial_load.OffloadedInputQuantError) as raised:
        backup_agreement.settle(mine, _answer(mine.elements, _peer()), exchange)
    assert sent[0][0] == 0 and "ComfyUI holds its weights as unpatched" in str(raised.value)  # the peer is told
    # The peer's side of the same exchange: its own model is right, and it stops too.
    rig.wrong, rig.room, rig.room_after = None, 9.0, 24.0
    other = _patcher()
    theirs = backup_agreement.vote((other,), SAMPLING, 1)
    with pytest.raises(rank_readiness.PeerLoadFailureError):
        backup_agreement.settle(theirs, _answer(theirs.elements, _peer()), lambda values: [0, values[1]])


def test_the_switch_on_any_rank_turns_the_rule_off_for_the_fleet(rig, caplog):
    patcher = _patcher()
    rig.room = 3.5  # short of the ask and of the need: with the rule on this would free, or refuse
    mine = backup_agreement.vote((patcher,), SAMPLING, 0)
    calls = []
    with caplog.at_level("INFO"):
        backup_agreement.settle(mine, _answer(mine.elements, _peer(switch_off=0)), lambda v: calls.append(v))
    assert rig.freed == [] and calls == [] and patcher.backup
    assert "is off for this fleet" in caplog.text and "a peer" in caplog.text


def _guard(monkeypatch, peer):
    """A world of two whose exchange answers as ``peer`` would; returns what each collective sent."""
    sent: list[list[int]] = []
    flags: list[bool] = []

    def exchange(values):
        sent.append(list(values))
        return _answer(values, peer[:len(values)] if len(sent) == 1 else [1, _peer()[3]])

    monkeypatch.setattr(partial_load_guard, "_distributed_world", lambda: 2)
    monkeypatch.setattr(partial_load_guard, "_rank", lambda: 0)
    monkeypatch.setattr(partial_load_guard, "_all_ranks_min", exchange)
    monkeypatch.setattr(partial_load_guard, "_all_ranks_full", lambda flag: flags.append(flag) or True)
    return sent, flags


def test_a_render_that_does_not_free_makes_exactly_the_one_collective_it_made_before(rig, monkeypatch):
    patcher = _patcher()
    sent, flags = _guard(monkeypatch, [1, *_peer()])
    wrapper = partial_load_guard.make_prepare_sampling_guard(whole_model=True)
    result = wrapper(lambda *args, **kwargs: ("real model", "conds", []), patcher, *SAMPLING)
    assert result == ("real model", "conds", [])
    assert len(sent) == 1 and len(sent[0]) == 1 + backup_agreement.ELEMENTS and flags == []
    assert sent[0][0] == 1 and rig.freed == []
    # Where not every rank holds the whole model (FSDP) the exchange is the old one, one flag.
    plain = partial_load_guard.make_prepare_sampling_guard()
    plain(lambda *args, **kwargs: ("real model", "conds", []), patcher, *SAMPLING)
    assert len(sent) == 1 and flags == [True]


def test_a_render_that_frees_makes_one_collective_more_and_a_partial_rank_still_refuses(rig, monkeypatch):
    patcher = _patcher()
    rig.room, rig.room_after = 9.0, 24.0
    sent, _flags = _guard(monkeypatch, [1, *_peer()])
    wrapper = partial_load_guard.make_prepare_sampling_guard(whole_model=True)
    wrapper(lambda *args, **kwargs: ("real model", "conds", []), patcher, *SAMPLING)
    assert len(sent) == 2 and len(sent[1]) == 2 and rig.freed == [patcher]
    # A peer that loaded partly: the guard's own refusal, and nothing is freed on the way.
    other = _patcher()
    rig.room, rig.room_after = 9.0, 24.0
    sent, _flags = _guard(monkeypatch, [0, *_peer()])
    with pytest.raises(partial_load_guard.PartialLoadDivergenceError):
        wrapper(lambda *args, **kwargs: ("real model", "conds", []), other, *SAMPLING)
    assert len(sent) == 1 and other.backup


def test_a_fleet_that_freed_and_was_refused_is_refused_again_at_its_next_render(rig):
    """No backup is left to be short of after a free: the slot's own flag keeps the room check alive."""
    patcher = _patcher()
    rig.room, rig.room_after = 2.0, 6.0

    def exchange(values):
        return _answer(values, [1, backup_agreement._margin(30 * GIB, 7 * GIB, 1)])

    mine = backup_agreement.vote((patcher,), SAMPLING, 0)
    with pytest.raises(backup_agreement.PairBackupRoomError) as first:
        backup_agreement.settle(mine, _answer(mine.elements, _peer()), exchange)
    assert getattr(patcher.slot, backup_agreement.FREED_FLAG) is True and patcher.backup == {}
    assert "clear the Sparks" in str(first.value)
    # The same graph queued again, nothing given back in between.
    again = backup_agreement.vote((patcher,), SAMPLING, 0)
    assert again.short is False and again.freed_before is True and again.elements[4] == 0
    calls: list = []
    with pytest.raises(backup_agreement.PairBackupRoomError) as raised:
        backup_agreement.settle(again, _answer(again.elements, _peer()), lambda values: calls.append(values))
    assert calls == [] and rig.freed == [patcher]  # no second exchange, and nothing freed twice
    message = str(raised.value)
    assert "freed their weight backup at an earlier render of this model" in message
    assert "clear the Sparks" in message and "lora_low_rss on does not help" in message
    # A roomy peer that freed with it carries the flag too, and is refused by the same reduced margin.
    peer_side = _answer(_peer(not_freed=0), again.elements)
    roomy = backup_agreement.Vote(rank=1, room=60 * GIB, need=7 * GIB, freed_before=True)
    with pytest.raises(backup_agreement.PairBackupRoomError):
        backup_agreement.settle(roomy, peer_side, lambda values: calls.append(values))
    # Once the Sparks are cleared and the room is back, the same vote passes.
    rig.room = 20.0
    fine = backup_agreement.vote((patcher,), SAMPLING, 0)
    backup_agreement.settle(fine, _answer(fine.elements, _peer()), lambda values: calls.append(values))
    assert calls == []


def test_the_acceptance_fault_drives_the_refusal_through_the_second_exchange(rig, monkeypatch, caplog):
    """For a proof cell on real ranks: the named rank votes short at a roomy reading and reads its model wrong."""
    from dgx_monarch.actor import load_fault

    patcher = _patcher()
    roomy = backup_agreement.vote((patcher,), SAMPLING, 0)
    assert roomy.short is False
    monkeypatch.setenv(backup_agreement.FAULT_RANK_ENV, "0")
    assert backup_agreement.vote((patcher,), SAMPLING, 0).short is False  # not without the acceptance switch
    monkeypatch.setenv(load_fault.ACCEPTANCE_ENV, "1")
    assert backup_agreement.vote((patcher,), SAMPLING, 1).short is False  # another rank is not named
    mine = backup_agreement.vote((patcher,), SAMPLING, 0)
    assert mine.short is True
    sent = []

    def exchange(values):
        sent.append(list(values))
        return _answer(values, [1, backup_agreement._margin(30 * GIB, 7 * GIB, 1)])

    with caplog.at_level("WARNING"), pytest.raises(lora_partial_load.OffloadedInputQuantError) as raised:
        backup_agreement.settle(mine, _answer(mine.elements, _peer()), exchange)
    assert rig.freed == [patcher] and sent[0][0] == 0
    assert "acceptance fault" in str(raised.value) and "acceptance fault" in caplog.text


def test_a_fleet_on_two_builds_is_refused_before_dispatch():
    def fleet(*reports):
        return SimpleNamespace(worker_capabilities=list(reports))

    mesh_builds.refuse_two_builds(fleet({"world": 2, "source": "aaaa"}, {"world": 2, "source": "aaaa"}))
    mesh_builds.refuse_two_builds(SimpleNamespace())  # a handle with no setup replies makes no claim
    # A Worker that could not hash its source says so with an empty field: no claim, no refusal.
    mesh_builds.refuse_two_builds(fleet({"world": 2, "source": "aaaa"}, {"world": 2, "source": None}))
    # A Fleet's actors each run a world of one and share no collective: not compared.
    mesh_builds.refuse_two_builds(fleet({"world": 1, "source": "aaaa"}, {"world": 1, "source": "bbbb"}))
    with pytest.raises(mesh_builds.FleetTwoBuildsError) as raised:
        mesh_builds.refuse_two_builds(fleet({"world": 2, "source": "aaaa"}, {"world": 2, "source": "bbbb"}))
    message = str(raised.value)
    assert "Worker(s) 0 run aaaa" in message and "Worker(s) 1 run bbbb" in message
    assert "Nothing was dispatched" in message and "Use one release on every box" in message
    tag = parse_leading_refusal_tag(message)
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS


def test_a_worker_from_before_the_check_is_refused_beside_one_that_reports():
    """An older build sends no source field at all: beside a Worker that does, the exchanges would not match."""
    old_and_new = SimpleNamespace(worker_capabilities=[{"world": 2, "rank": 0, "source": "aaaa"},
                                                       {"world": 2, "rank": 0}])
    with pytest.raises(mesh_builds.FleetTwoBuildsError) as raised:
        mesh_builds.refuse_two_builds(old_and_new)
    message = str(raised.value)
    assert "Worker(s) 1 run a build from before this check" in message and "Worker(s) 0 run aaaa" in message
    # Two Workers that both predate the check state one build between them: nothing to refuse here.
    mesh_builds.refuse_two_builds(SimpleNamespace(worker_capabilities=[{"world": 2}, {"world": 2}]))
