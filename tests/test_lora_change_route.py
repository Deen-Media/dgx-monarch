"""A LoRA change on a held stock base: which path the Worker takes, and the card.

The 2026-11-05 hardware leg met two cases on the BF16 Wan-Animate 2 templates,
which keep LoRA residency stock (low-RSS and slab off).

B2: a LoRA added to a bare base already placed on the GPU. The in-place swap
spent 230 s copying the model off the GPU before ComfyUI re-applied the stack,
while a fresh load of base and LoRA took 13.1 s and its patch 4.0 s. The store
now drops the slot and loads fresh when ComfyUI holds no host copy of a weight
the new stack patches, at the same price.

B3: after a base-plus-LoRA load, each actor kept about 34 GiB of anonymous
memory (ComfyUI's host copy of the patched weights). A strength change was
priced as a credited hot swap, 38.6 GiB against 24.3 GiB, 14.3 GiB short, and
no card was raised, because the resident is the file the change asks for. The
refusal now offers the Old model in the way card when a clear makes room for
the price a fresh fleet pays. CPU only.
"""
from __future__ import annotations

import logging
import types
from dataclasses import replace

import pytest

from dgx_monarch import capacity_agreement, mesh_safety, old_model
from dgx_monarch.actor import actor_footprint, store_bake, worker_capacity
from dgx_monarch.actor import capacity_quote as cq
from dgx_monarch.actor import model_store as ms
from dgx_monarch.capacity_agreement import RankAnswer, decide
from dgx_monarch.capacity_fit import StockFit
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from store_ensure_helpers import rig as rig  # a fixture tests ask for by name.

_GIB = 2 ** 30
BASE = "wan_animate_2_bf16.safetensors"
LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
# The BF16 Wan-Animate 2 file (30.5 GiB) and B3's readings.
BF16_BYTES = 32_789_894_104
HEAD_AVAILABLE = int(24.3 * _GIB)
SIBLING_AVAILABLE = int(42.2 * _GIB)
# The head actor's anonymous memory above its fresh floor after the first
# queue: 34.1 GiB held, a few GiB of it the floor.
HEAD_ARENA = int(29.6 * _GIB)
CARD_BODY = ("The model loaded earlier (wan_animate_2_bf16) holds memory on head that this "
             "change needs. Clear the Sparks and run again?")


def _row(available: int, verdict: str, *, arena: int = HEAD_ARENA, resident: str = BASE,
         slot: str = "cond") -> cq.RankPrice:
    """A credited LoRA-change row, as the Worker sends it for B3's state."""
    fit = StockFit(False, True, BF16_BYTES, available, credit_bytes=BF16_BYTES)
    fit = StockFit(fit.required_bytes <= available, True, BF16_BYTES, available,
                   credit_bytes=BF16_BYTES)
    return cq.RankPrice(
        verdict=verdict, rung=cq.RUNG_EXPLICIT_STOCK, priced=True, measured=dict(fit.measured()),
        would_load=cq.WOULD_LOAD_SWAP, checkpoint_bytes=BF16_BYTES, slot=slot,
        holds_resident={"cond": "stock", "uncond": None} if slot == "cond"
        else {"cond": None, "uncond": "stock"},
        resident_names={"cond": resident, "uncond": None} if slot == "cond"
        else {"cond": None, "uncond": resident},
        resident_bytes={slot: BF16_BYTES}, arena_bytes=arena, reason="short")


def _b3(*, arena: int = HEAD_ARENA) -> list[RankAnswer]:
    return [RankAnswer(0, "host0", (_row(HEAD_AVAILABLE, "refuse", arena=arena),),
                       configured_host="head"),
            RankAnswer(1, "host1", (_row(SIBLING_AVAILABLE, "fits"),), configured_host="sibling")]


@pytest.fixture(autouse=True)
def _no_card_store(monkeypatch):
    """The card's registration needs a driver's model folders; these tests read the refusal."""
    from dgx_monarch.nodes import old_model_clear

    monkeypatch.setattr(old_model_clear, "card_possible", lambda _name: True)


# B3: the card for a change the copy loaded earlier blocks.

def test_b3_a_credited_change_14_3_gib_short_offers_the_card():
    head = _b3()[0].quotes[0]
    assert head.measured["required_gib"] == 38.6 and head.measured["headroom_gib"] == -14.3
    with pytest.raises(old_model.OldModelRefusal) as raised:
        decide(_b3(), 2, unet_name=BASE, keep=frozenset({BASE}))
    offer = old_model.offer_of(raised.value)
    assert offer is not None and offer.helps and offer.same_file
    (box,) = offer.boxes
    assert (box.box, box.old_models) == ("head", ("wan_animate_2_bf16",))
    # A fresh fleet holds no copy to credit: the clear must make room for 69.1 GiB.
    assert round(box.fresh_bytes / _GIB, 1) == 69.1
    assert box.gain_bytes == BF16_BYTES + HEAD_ARENA
    assert old_model.card_body(offer) == CARD_BODY
    assert old_model.card_numbers(offer) == "needs 38.6 GiB, 24.3 GiB available on head"
    text = str(raised.value)
    assert "rank 0 on head priced a LoRA change on wan_animate_2_bf16.safetensors" in text
    assert "14.3 GiB short" in text
    assert "The model loaded earlier (wan_animate_2_bf16) holds memory on head" in text
    assert '"Clear and retry" on the "Old model in the way" card' in text
    assert "Reset attached mesh" in text
    assert "does not fit beside it" not in text
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY
    assert tag.guard == "cross_rank_capacity" and not tag.waivable
    assert mesh_safety.is_stock_load_capacity_error(raised.value)
    assert old_model.log_line(offer) == (
        "cleared both Sparks so wan_animate_2_bf16 loads fresh; the copy loaded earlier "
        "held memory this change needed")


def test_b3_through_the_sample_agreement_where_the_request_names_the_held_file(monkeypatch):
    """The keep the sampler passes names the request's own file, as on the rig."""
    monkeypatch.setattr(capacity_agreement, "collect", lambda *_a, **_k: _b3())
    handle = types.SimpleNamespace(world=2, workers=object(), active_worker_args={})
    request = capacity_agreement.quote_request([
        {"unet_name": BASE, "options": {}, "loras": [{"name": LORA, "strength": 0.8}],
         "slot": "cond"}])
    with pytest.raises(old_model.OldModelRefusal) as raised:
        capacity_agreement.agree_or_refuse(handle, request, unet_name=BASE)
    assert raised.value.fleet is handle
    assert old_model.offer_of(raised.value) is not None


def test_the_same_file_card_needs_room_for_the_price_a_fresh_fleet_pays():
    """Gain that covers the credited shortfall but not the fresh one offers no card."""
    small = int(10.0 * _GIB)
    gain = BF16_BYTES + small
    assert gain > int(14.3 * _GIB) + old_model.CLEAR_MARGIN_BYTES   # the credited figure
    with pytest.raises(capacity_agreement.FleetCapacityError) as raised:
        decide(_b3(arena=small), 2, unet_name=BASE, keep=frozenset({BASE}))
    assert not isinstance(raised.value, old_model.OldModelRefusal)
    text = str(raised.value)
    assert "The model loaded earlier (wan_animate_2_bf16) holds memory on head" in text
    assert ("A clear would not make room: wan_animate_2_bf16 needs 69.1 GiB on head, and a "
            "cleared box would have about 62.8 GiB.") in text


def test_b2_state_is_admitted_and_raises_nothing():
    """B2's swap was admitted on both ranks: 56.7 GiB on the head, 74.8 on the sibling."""
    answers = [RankAnswer(0, "host0", (_row(int(56.7 * _GIB), "fits"),), configured_host="head"),
               RankAnswer(1, "host1", (_row(int(74.8 * _GIB), "fits"),), configured_host="sibling")]
    decide(answers, 2, unet_name=BASE, keep=frozenset({BASE}))


@pytest.mark.parametrize(("quote", "unet_name", "expected", "why"), [
    (_row(HEAD_AVAILABLE, "refuse"), BASE, True, "B3: a change to the held file"),
    (_row(HEAD_AVAILABLE, "refuse", resident="flux1-dev.safetensors"), BASE, False,
     "another model in the slot keeps today's rule"),
    (_row(HEAD_AVAILABLE, "refuse", slot="uncond"), "other_cond.safetensors", False,
     "the agreement names the cond model, so an uncond change keeps today's rule"),
    (_row(HEAD_AVAILABLE, "refuse"), "", False, "no name, no match"),
])
def test_only_the_slot_a_row_changes_in_place_leaves_the_kept_set(quote, unet_name, expected, why):
    assert old_model.changes_in_place(quote, unet_name) is expected, why
    reuse = replace(quote, would_load=cq.WOULD_LOAD_REUSE)
    assert old_model.changes_in_place(reuse, unet_name) is False, "a reuse is never a change"


def test_the_worker_quote_for_b3s_state_reaches_the_card(monkeypatch, tmp_path):
    """End to end on one rank: a real store holding base plus LoRA, a strength change."""
    from dgx_monarch.actor.store_detect import LivePrecisionEvidence
    from dgx_monarch.actor.stored_model import StoredModel

    store = ms.ModelStore()
    store.lora_low_rss = False
    store.slab_weights = False
    held_sig = ((LORA, 1.0),)
    store.current = StoredModel(
        base_key=(BASE, ()), request_key=(BASE, (), held_sig, "bf16"), base_patcher=object(),
        active_patcher=object(), quant_kind="bf16",
        precision_evidence=LivePrecisionEvidence("bf16", "bf16"), family="wan_animate2",
        artifact_identity={}, lora_sig=held_sig, residency_rung=cq.RUNG_EXPLICIT_STOCK)
    checkpoint = tmp_path / BASE
    with open(checkpoint, "wb") as handle:
        handle.truncate(BF16_BYTES)
    monkeypatch.setattr(ms, "resolve_model_path", lambda _kind, _name: str(checkpoint))
    monkeypatch.setattr(ms, "_slab_capable_path", lambda _path: False)
    monkeypatch.setattr(ms, "_to_comfy_model_options", lambda options: dict(options or {}))
    monkeypatch.setattr(worker_capacity, "_pinned_staging", lambda: False)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: HEAD_AVAILABLE)
    monkeypatch.setattr(actor_footprint, "arena_bytes", lambda *_a: HEAD_ARENA)
    worker = types.SimpleNamespace(rank=0, world=2, topology={}, _setup_key=("ready",),
                                   store=store, uma_reserve_gb=0.0)
    envelope = worker_capacity.quote(worker, {"specs": [
        {"unet_name": BASE, "options": {}, "loras": [{"name": LORA, "strength": 0.8}],
         "slot": "cond"}], "rescue_consent": None})
    head = cq.RankPrice.from_row(envelope["quotes"][0])
    assert (head.would_load, head.verdict) == ("swap", "refuse")
    assert head.measured["credit_bytes"] == BF16_BYTES and head.measured["headroom_gib"] == -14.3
    assert head.resident_names["cond"] == BASE and head.resident_bytes["cond"] == BF16_BYTES
    sibling = _row(SIBLING_AVAILABLE, "fits")
    with pytest.raises(old_model.OldModelRefusal) as raised:
        decide([RankAnswer(0, "host0", (head,), configured_host="head"),
                RankAnswer(1, "host1", (sibling,), configured_host="sibling")],
               2, unet_name=BASE, keep=frozenset({BASE}))
    assert old_model.card_body(old_model.offer_of(raised.value)) == CARD_BODY


# B2: the Worker's choice between an in-place swap and a fresh load.

def _placed(model, placed_bytes: int = BF16_BYTES) -> None:
    model.model_loaded_weight_memory = placed_bytes


def _patched_active(keys):
    def build(self, base, stack):
        return types.SimpleNamespace(tag="ACTIVE", model=base.model,
                                     patches={key: [("patch",)] for key in keys} if stack else {})
    return build


def test_b2_a_lora_added_to_a_placed_bare_base_loads_fresh(rig, monkeypatch, caplog):
    store, calls = rig
    store.lora_low_rss = False
    store.ensure(BASE, {}, None)
    bare = store.current
    _placed(bare.base_patcher.model)
    bare.base_patcher.backup = {}            # a bare base: ComfyUI holds no host copy
    monkeypatch.setattr(ms.ModelStore, "_build_active", _patched_active(["w1", "w2", "w3"]))
    with caplog.at_level(logging.INFO, logger=ms.log.name):
        _patcher, transition = store.ensure(BASE, {}, [{"name": LORA, "strength": 1.0}])
    assert transition == "load" and calls.load == 2
    assert store.current is not bare and store.current.lora_sig == ((LORA, 1.0),)
    assert calls.unload_all >= 1             # the drop ran before the fresh load
    assert ("a LoRA change on wan_animate_2_bf16.safetensors would copy the model off the GPU "
            "first (no host copy of 3 of the 3 weights the new stack patches)") in caplog.text


def test_a_strength_change_whose_host_copy_is_held_swaps_in_place(rig, monkeypatch):
    """B3's change, once it fits: the backup ComfyUI holds covers the stack."""
    store, calls = rig
    store.lora_low_rss = False
    monkeypatch.setattr(ms.ModelStore, "_build_active", _patched_active(["w1", "w2"]))
    store.ensure(BASE, {}, [{"name": LORA, "strength": 1.0}])
    held = store.current
    _placed(held.base_patcher.model)
    held.base_patcher.backup = {"w1": object(), "w2": object()}
    _patcher, transition = store.ensure(BASE, {}, [{"name": LORA, "strength": 0.8}])
    assert transition == "hot-swap" and calls.load == 1
    assert store.current is held and held.lora_sig == ((LORA, 0.8),)


def test_a_two_slot_store_keeps_the_in_place_swap(rig, monkeypatch):
    """A drop's global unload would push the other slot's model off the GPU too."""
    store, calls = rig
    store.lora_low_rss = False
    store.ensure("uncond_model.safetensors", {}, None, slot="uncond")
    store.ensure(BASE, {}, None)
    _placed(store.current.base_patcher.model)
    store.current.base_patcher.backup = {}
    monkeypatch.setattr(ms.ModelStore, "_build_active", _patched_active(["w1"]))
    _patcher, transition = store.ensure(BASE, {}, [{"name": LORA, "strength": 1.0}])
    assert transition == "hot-swap" and calls.load == 2


def _gate_case(**changes):
    model = types.SimpleNamespace(model_loaded_weight_memory=BF16_BYTES)
    base = types.SimpleNamespace(model=model, backup={})
    stored = types.SimpleNamespace(base_patcher=base, slab=None, residency_rung="explicit_stock")
    store = types.SimpleNamespace(_stored=lambda slot: None)
    active = types.SimpleNamespace(patches={"w1": [], "w2": []})
    for key, value in changes.items():
        target, attr = key.split("__")
        setattr({"model": model, "base": base, "stored": stored, "store": store,
                 "active": active}[target], attr, value)
    return store, stored, active


@pytest.mark.parametrize(("changes", "expected", "why"), [
    ({}, (2, 2), "B2: placed, no host copy, other slot empty"),
    ({"base__backup": {"w1": object()}}, (1, 2), "one weight still has no host copy"),
    ({"base__backup": {"w1": object(), "w2": object()}}, (0, 2), "the host copy covers it"),
    ({"model__model_loaded_weight_memory": 0}, (0, 2), "never placed: nothing to copy back"),
    ({"model__model_loaded_weight_memory": True}, (0, 2), "a flag is not a byte count"),
    ({"model__model_loaded_weight_memory": object()}, (0, 2), "a double reads as not placed"),
    ({"store___stored": lambda slot: object()}, (0, 2), "the other slot holds a model"),
    ({"stored__slab": object()}, (0, 2), "a slab resident keeps its own path"),
    ({"stored__residency_rung": cq.RUNG_COMFY_MANAGED}, (0, 2), "comfy-managed keeps no backup"),
    ({"base___dgxm_fsdp": True}, (0, 2), "an FSDP shard keeps its own path"),
    ({"base__backup": None}, (0, 2), "no backup dict to read"),
    ({"active__patches": None}, (0, 0), "no patch dict to read"),
])
def test_every_gate_keeps_the_swap_in_place(changes, expected, why):
    store, stored, active = _gate_case(**changes)
    assert store_bake.copies_off_the_gpu(store, stored, "cond", active) == expected, why
