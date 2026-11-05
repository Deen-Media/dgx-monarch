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


def test_one_configured_box_is_cleared_by_name(monkeypatch):
    """One host in cluster.toml: the card asks to clear that box, by its name."""
    monkeypatch.setattr(driver_footprint, "host_is_local", _head_is("fabric-a"))
    box = old_model.ShortBox(box_names.HEAD, ("flux1-dev",), 45 * _GIB, 39 * _GIB, 22 * _GIB)
    offer = old_model.ClearOffer(NEW, (box,), 1, False)
    assert old_model.card_body(offer).endswith("Clear the head and run again?")
    assert old_model.log_line(offer).startswith("cleared the head to make room for")


def test_no_capacity_sentence_names_a_rank_by_number():
    """The modules that word these texts build no "rank <n>" or host sentence."""
    banned = re.compile(r"rank \{|\(host |host \{")
    for name in ("capacity_agreement.py", "old_model.py", "nodes/old_model_clear.py",
                 "fsdp_reload_price.py"):
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
