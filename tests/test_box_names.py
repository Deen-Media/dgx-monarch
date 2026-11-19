"""The one rule for naming a box in text a user reads (box_names.py).

On 2026-11-05 the Old model in the way card and the capacity refusals named one
box three ways in one session: the cluster host name, the address cluster.toml
gives it, and "rank 0". These tests hold the rule and the texts that use it.
CPU only: no torch beyond the actor package import, no fleet.
"""
from __future__ import annotations

import ast
import re
import types
from pathlib import Path

import pytest

from dgx_monarch import box_names, capacity_agreement, driver_footprint, old_model
from dgx_monarch.actor.capacity_quote import RankPrice
from dgx_monarch.capacity_agreement import FleetCapacityError, RankAnswer, decide
from dgx_monarch.capacity_fit import StockFit

SRC = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"
_GIB = 2 ** 30
NEW = "qwen_image_2512_fp8_e4m3fn.safetensors"
OLD = "flux1-dev.safetensors"
# cluster.toml may name a box by any string, an address included; the Worker
# reports its own host name. Neither may reach a sentence a user reads.
CONFIGURED = ("fabric-a", "fabric-b")
REPORTED = ("worker-a", "worker-b")


def _head_is(name: str):
    return lambda host: host == name


def test_a_fleet_on_the_driver_box_alone_is_this_box():
    """A one-host cluster.toml on the driver's box reads as a local fleet does,
    so one box keeps one name whatever the Init mode."""
    assert box_names.host_words(("fabric-a",), _head_is("fabric-a")) == ["this box"]
    assert box_names.host_words(("fabric-b",), _head_is("fabric-a")) == ["the sibling"]
    assert [box_names.label(*pair) for pair in box_names.rank_boxes(
        ("fabric-a",), 2, 2, _head_is("fabric-a"))] == ["GPU 0 on this box", "GPU 1 on this box"]


def test_a_pair_is_the_head_and_the_sibling():
    assert box_names.host_words(CONFIGURED, _head_is("fabric-a")) == ["the head", "the sibling"]
    # The head need not be the first entry: the driver's own box is the head.
    assert box_names.host_words(CONFIGURED, _head_is("fabric-b")) == ["the sibling", "the head"]


def test_more_than_one_other_box_is_named_by_host():
    hosts = ("fabric-a", "fabric-b", "fabric-c")
    assert box_names.host_words(hosts, _head_is("fabric-a")) == [
        "the head", "the sibling at fabric-b", "the sibling at fabric-c"]
    # A driver box with no Worker: two other boxes, so neither is just "the sibling".
    assert box_names.host_words(CONFIGURED, _head_is("elsewhere")) == [
        "the sibling at fabric-a", "the sibling at fabric-b"]
    assert box_names.host_words(("fabric-b",), _head_is("elsewhere")) == ["the sibling"]


def test_ranks_take_their_box_and_name_a_gpu_only_where_a_box_holds_several():
    head = _head_is("fabric-a")
    assert box_names.rank_boxes(CONFIGURED, 2, 1, head) == [
        ("the head", None), ("the sibling", None)]
    assert [box_names.label(*pair) for pair in box_names.rank_boxes(CONFIGURED, 4, 2, head)] == [
        "GPU 0 on the head", "GPU 1 on the head", "GPU 0 on the sibling", "GPU 1 on the sibling"]
    assert box_names.rank_boxes((), 1, 1, head) == [("this box", None)]
    assert [box_names.label(*pair) for pair in box_names.rank_boxes((), 2, 1, head)] == [
        "GPU 0 on this box", "GPU 1 on this box"]
    assert box_names.rank_boxes(("fabric-a",), 2, 1, head)[1] == (box_names.UNLISTED, None)


def test_a_this_host_mesh_is_local_whatever_the_config_lists(monkeypatch):
    """Init mode=local on a rig with a cluster.toml: every rank is on this box."""
    monkeypatch.setattr(driver_footprint, "host_is_local", _head_is("fabric-a"))
    config = types.SimpleNamespace(hosts=tuple(
        types.SimpleNamespace(name=name) for name in CONFIGURED))
    local = types.SimpleNamespace(owns_hosts=False, config=config, gpus_per_host=1)
    assert box_names.fleet_labels(local, 1) == ["this box"]
    cluster = types.SimpleNamespace(owns_hosts=True, config=config, gpus_per_host=1)
    assert box_names.fleet_labels(cluster, 2) == ["the head", "the sibling"]


def test_a_name_that_opens_a_sentence_is_capitalized():
    assert box_names.opening("the sibling at fabric-b priced") == "The sibling at fabric-b priced"
    assert box_names.opening("") == ""


def _short(avail_gib: float, *, resident: str | None = OLD) -> RankPrice:
    fit = StockFit(False, True, 19 * _GIB, int(avail_gib * _GIB))
    return RankPrice(
        verdict="refuse", rung="stock_fits", priced=True, measured=dict(fit.measured()),
        would_load="swap" if resident else "fresh", checkpoint_bytes=19 * _GIB, slot="cond",
        holds_resident={"cond": "stock" if resident else None, "uncond": None},
        resident_names={"cond": resident, "uncond": None},
        resident_bytes={"cond": 22 * _GIB} if resident else {}, reason="short")


def _fits() -> RankPrice:
    fit = StockFit(True, True, 19 * _GIB, 200 * _GIB)
    return RankPrice(verdict="fits", rung="stock_fits", priced=True,
                     measured=dict(fit.measured()), would_load="fresh",
                     checkpoint_bytes=0, slot="cond", reason="fits")


def _pair(monkeypatch, *rows: RankPrice) -> list[RankAnswer]:
    """Answers named the way ``collect`` names them on a configured pair."""
    monkeypatch.setattr(driver_footprint, "host_is_local", _head_is(CONFIGURED[0]))
    handle = types.SimpleNamespace(config=types.SimpleNamespace(hosts=tuple(
        types.SimpleNamespace(name=name) for name in CONFIGURED)), gpus_per_host=1)
    named = box_names.fleet_boxes(handle, len(rows))
    return [RankAnswer(rank, REPORTED[rank], (row,), elapsed_s=0.01, box=box, gpu=gpu)
            for rank, (row, (box, gpu)) in enumerate(zip(rows, named, strict=True))]


def _plain(text: str) -> None:
    for word in (*CONFIGURED, *REPORTED, "rank 0", "rank 1", "(host"):
        assert word not in text, (word, text)


def test_the_old_model_refusal_and_its_card_say_the_head(monkeypatch, caplog):
    """A1b on 2026-11-05: the card read an address where the docs say the head."""
    answers = _pair(monkeypatch, _short(39.4), _fits())
    with caplog.at_level("INFO"), pytest.raises(old_model.OldModelRefusal) as raised:
        decide(answers, 2, unet_name=NEW)
    text = str(raised.value)
    _plain(text)
    assert "on any rank. The head priced a stock load of " + NEW in text
    assert ("The old model (flux1-dev) is still loaded on the head and "
            "qwen_image_2512_fp8_e4m3fn does not fit beside it.") in text
    offer = old_model.offer_of(raised.value)
    assert offer is not None
    assert old_model.card_body(offer) == (
        "The old model (flux1-dev) is still loaded on the head and qwen_image_2512_fp8_e4m3fn "
        "does not fit beside it. Clear the Sparks and run again?")
    assert old_model.card_numbers(offer).endswith(" GiB available on the head")
    line = caplog.text
    assert "old model in the way: the head=cond:refuse/" in line
    assert ", the sibling=cond:fits/stock_fits@" in line
    _plain(line)


def test_an_ordinary_refusal_names_both_boxes_plainly(monkeypatch):
    answers = _pair(monkeypatch, _short(2.0, resident=None), _fits())
    with pytest.raises(FleetCapacityError) as raised:
        decide(answers, 2, unet_name=NEW)
    text = str(raised.value)
    _plain(text)
    assert "The head priced a stock load" in text
    assert "The sibling took the stock_fits rung." in text


def test_one_configured_box_is_cleared_by_name():
    """One host in cluster.toml: the card asks to clear that box by the rule's name,
    "this box" on the driver's box, "the sibling" on the other one."""
    for head, word in (("fabric-a", "this box"), ("elsewhere", "the sibling")):
        (name,) = box_names.host_words(("fabric-a",), _head_is(head))
        box = old_model.ShortBox(name, ("flux1-dev",), 45 * _GIB, 39 * _GIB, 22 * _GIB)
        offer = old_model.ClearOffer(NEW, (box,), 1, False)
        assert old_model.card_body(offer).endswith(f"Clear {word} and run again?")
        assert old_model.log_line(offer).startswith(f"cleared {word} to make room for")


def test_no_capacity_sentence_names_a_rank_by_number():
    """The modules that word these texts build no "rank <n>", host or address sentence."""
    banned = re.compile(r"rank \{|\(host |host \{|\{[\w.]*address\}|\{[\w.]*\.name\}")
    for name in ("capacity_agreement.py", "old_model.py", "nodes/old_model_clear.py",
                 "fsdp_reload_price.py", "restart_reattach.py", "driver_footprint.py"):
        tree = ast.parse((SRC / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                source = ast.unparse(node)
                assert not banned.search(source), (name, source)


def test_the_agreement_imports_the_rule_rather_than_its_own_names():
    source = (SRC / "capacity_agreement.py").read_text()
    assert "LOCAL_HOST_NAME" not in source and "configured_host" not in source
    assert "box_names." in source
    assert capacity_agreement.RankAnswer(0).where == box_names.UNLISTED


# Round two: the texts the review of PR #143 found still naming a box.

_ADDRESSES = tuple(f"tcp://{name}:26600" for name in CONFIGURED)


def _fleet_config(monkeypatch) -> types.SimpleNamespace:
    """A pair whose cluster.toml names and addresses must never reach a sentence."""
    monkeypatch.setattr(driver_footprint, "host_is_local", _head_is(CONFIGURED[0]))
    return types.SimpleNamespace(config=types.SimpleNamespace(hosts=tuple(
        types.SimpleNamespace(name=name, address=address)
        for name, address in zip(CONFIGURED, _ADDRESSES, strict=True))))


def _plain_of_addresses(text: str) -> None:
    _plain(text)
    assert "tcp://" not in text and ":26600" not in text, text


def test_the_undeliverable_actor_refusal_names_the_box(monkeypatch):
    """Raised from inside the agreement's wait, under its guard (restart_reattach.py)."""
    from dgx_monarch import restart_reattach
    from dgx_monarch.delivery_faults import WorkerDeliveryError

    handle = _fleet_config(monkeypatch)
    with pytest.raises(WorkerDeliveryError) as raised:
        restart_reattach.refuse_undelivered(handle, [_ADDRESSES[1]], 1.2, 30.0)
    text = str(raised.value)
    _plain_of_addresses(text)
    assert "this fleet's Worker actor on the sibling undeliverable" in text
    assert "the Worker loop on that box restarted" in text


def test_the_move_refusals_name_the_boxes(monkeypatch):
    from dgx_monarch import restart_reattach
    from dgx_monarch.delivery_faults import WorkerDeliveryError

    handle = _fleet_config(monkeypatch)
    cases = (
        (restart_reattach.Moves(moved=_ADDRESSES[:1], kept=_ADDRESSES[1:]),
         "the Worker loops on the head serve new proc ids while those on the sibling "
         "still serve the id this mesh dialed"),
        (restart_reattach.Moves(kept=_ADDRESSES),
         "the actor on that box is gone or unreachable"),
        (restart_reattach.Moves(unknown=("the sibling (no id published)",)),
         "the id reading found none for the sibling (no id published)"),
    )
    for moves, words in cases:
        with pytest.raises(WorkerDeliveryError) as raised:
            restart_reattach._refuse_moves(handle, moves, RuntimeError("x"))
        _plain_of_addresses(str(raised.value))
        assert words in str(raised.value)


def test_the_retire_refusal_and_its_death_report_name_the_boxes(monkeypatch):
    from dgx_monarch import mesh_helpers, mesh_liveness, mesh_teardown, restart_reattach
    from dgx_monarch.delivery_faults import WorkerDeliveryError

    handle = _fleet_config(monkeypatch)
    handle._liveness_evidence = mesh_liveness.LivenessEvidence(
        mesh_liveness.ALIVE, ((_ADDRESSES[0], "alive"), (_ADDRESSES[1], "gone")), 0.0)
    monkeypatch.setattr(mesh_teardown, "block_cause_sentence", lambda _handle: "")
    moves = restart_reattach.Moves(moved=_ADDRESSES)
    with pytest.raises(WorkerDeliveryError) as raised:
        restart_reattach._refuse_retire(handle, moves, None)
    text = str(raised.value)
    _plain_of_addresses(text)
    assert "(on the head and the sibling)" in text
    assert "(the head: alive, the sibling: gone)" in text
    reports: list[str] = []

    def publish(_handle, _refused, report):
        reports.append(report)
        return object(), False, True, False

    monkeypatch.setattr(mesh_helpers, "publish_supervision_failure_state", publish)
    handle.teardown_complete = True
    assert restart_reattach._retire(handle, moves, RuntimeError("x")) == "stop_confirmed"
    assert reports == ["worker restart: the Worker loops on the head and the sibling serve "
                       "new proc ids, and messages to this fleet's actors were undeliverable"]


def test_the_loader_charge_notes_name_no_rank():
    """The loader refusal that carries the Old model card read "co-resident rank 0"."""
    profile = next(iter(driver_footprint.DRIVER_STACK_FAMILIES.values()))
    notes = [driver_footprint.estimate_driver_footprint(
        profile=profile, mem_available=100 * _GIB, weight_bytes=20 * _GIB,
        co_resident=co_resident, fsdp_world=world).notes["weights"]
        for co_resident, world in ((True, 0), (False, 0), (True, 2))]
    assert notes[0] == "a Worker on this box, not yet loaded"
    assert notes[1] == "the first Worker is on another box, not charged"
    assert notes[2].startswith("a Worker on this box, FSDP shard build: it holds ")
    assert not any("rank" in note for note in notes)


def test_a_reply_that_names_another_rank_keeps_both_numbers_in_the_log(caplog):
    """The user reads 'answered as another rank'; the numbers are the slicing evidence."""
    with caplog.at_level("WARNING"):
        answer = capacity_agreement._answer_from(1, {"host": REPORTED[1], "rank": 0,
                                                     "quotes": []}, 0.1)
    assert answer.silent == "answered as another rank"
    assert "cross-rank capacity: rank 1 answered as rank 0" in caplog.text


# After PR #142: the credited-swap and cast sentences name boxes by the same rule.

def _credited(avail_gib: float, arena_gib: float) -> RankPrice:
    """qwen over flux1-dev in one slot, priced after the Worker drops flux1-dev."""
    from dataclasses import replace

    fit = StockFit(False, True, 19 * _GIB, int(avail_gib * _GIB), credit_bytes=22 * _GIB,
                   dropped=OLD)
    fit = replace(fit, fits=fit.required_bytes <= fit.avail_bytes)
    return replace(_short(avail_gib), measured=dict(fit.measured()),
                   arena_bytes=int(arena_gib * _GIB))


def test_the_credited_swap_and_its_card_name_the_head(monkeypatch):
    answers = _pair(monkeypatch, _credited(10.0, 30.0), _fits())
    with pytest.raises(old_model.OldModelRefusal) as raised:
        decide(answers, 2, unet_name=NEW)
    text = str(raised.value)
    _plain(text)
    assert (f"The head priced a stock load of {NEW} after the Worker drops flux1-dev, "
            "which frees at least 22.0 GiB.") in text
    offer = old_model.offer_of(raised.value)
    assert offer is not None
    assert old_model.card_body(offer) == (
        "The head still holds about 30.0 GiB that earlier loads left behind, and "
        "qwen_image_2512_fp8_e4m3fn does not fit beside it. Clear the Sparks and run again?")
    assert old_model.card_numbers(offer).endswith(" GiB available on the head")


def test_both_boxes_left_behind_read_as_one_sentence(monkeypatch):
    answers = _pair(monkeypatch, _credited(10.0, 30.0), _credited(12.0, 40.0))
    with pytest.raises(old_model.OldModelRefusal) as raised:
        decide(answers, 2, unet_name=NEW)
    _plain(str(raised.value))
    assert ("The head still holds about 30.0 GiB and the sibling about 40.0 GiB that "
            "earlier loads left behind") in str(raised.value)


def test_a_credited_swap_no_clear_can_fix_names_the_box_and_the_whole_price(monkeypatch):
    answers = _pair(monkeypatch, _credited(10.0, 0.0), _fits())
    with pytest.raises(FleetCapacityError) as raised:
        decide(answers, 2, unet_name=NEW)
    text = str(raised.value)
    _plain(text)
    assert not isinstance(raised.value, old_model.OldModelRefusal)
    assert " GiB on the head, and a cleared box would have about " in text
    assert "That is the whole price, because a cleared box holds no copy" in text


def test_a_cast_load_names_the_box_that_priced_it(monkeypatch):
    from dataclasses import replace

    fit = StockFit(False, True, int(13.3 * _GIB), int(20.0 * _GIB), cast_bytes=int(26.6 * _GIB))
    cast = replace(_short(20.0, resident=None), measured=dict(fit.measured()))
    answers = _pair(monkeypatch, _fits(), cast)
    with pytest.raises(FleetCapacityError) as raised:
        decide(answers, 2, unet_name=NEW)
    text = str(raised.value)
    _plain(text)
    assert ("The sibling priced a stock load of " + NEW + ": the load needs ") in text
    assert "a 13.3 GiB file cast to a 26.6 GiB copy, plus the host copy" in text
    assert "The head took the stock_fits rung." in text
