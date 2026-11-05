"""A LoRA change on a base the slot already holds is priced as the hot swap it is.

The 2026-11-04 release test refused the Wan-Animate 2 memory-saver template on
a fresh pair of Sparks. Under an explicit topology the UNET Loader loads the
bare base at node time; the sampler then asks for the base plus its LoRA. The
quote read that as a swap and priced a whole stock load (the file, the host
copy it is placed from and the floor, 69.1 GiB) beside the base already held
(61.8 GiB available). The model store does not load the base again for a LoRA
change: it keeps the base and re-applies the stack (``ModelStore.ensure``,
``store_bake.hot_swap_transition``), which places at most the host copy of the
weights the stack patches. These tests pin the credited price and the cases
that must not get it. CPU only.
"""
from __future__ import annotations

import types

import pytest

from dgx_monarch import box_names, capacity_agreement, capacity_fit, mesh_safety
from dgx_monarch.actor import capacity_quote as cq
from dgx_monarch.actor import store_bake, worker_capacity
from dgx_monarch.capacity_agreement import RankAnswer
from dgx_monarch.capacity_fit import StockFit

_GIB = 2 ** 30
# The BF16 Wan-Animate 2 file and the head's reading at the 2026-11-04 refusal.
BF16_BYTES = 32_789_894_104
HEAD_AVAILABLE = int(61.8 * _GIB)
BASE = "wan_animate_2_bf16.safetensors"
HELD_KEY = repr((BASE, (), (), "bf16"))
WITH_LORA_KEY = repr((BASE, (), (("distill.safetensors", 1.0),), "bf16"))
LORA_STACK = [{"name": "distill.safetensors", "strength": 1.0}]


def _probe(_path, _options):
    fit = StockFit(True, True, BF16_BYTES, HEAD_AVAILABLE)
    return StockFit(fit.required_bytes <= HEAD_AVAILABLE, True, BF16_BYTES, HEAD_AVAILABLE)


def _held(**changes):
    entry = {"residency": "stock", "residency_rung": cq.RUNG_EXPLICIT_STOCK,
             "request_key": HELD_KEY, "swaps_in_place": True}
    entry.update(changes)
    return {"cond": entry, "uncond": None, "cleanup_failed_slots": []}


def _price(monkeypatch, *, snapshot=None, request_key=WITH_LORA_KEY, **overrides):
    monkeypatch.setattr(cq.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(capacity_fit, "slab_load_fit", lambda _p, _o, **_kw: capacity_fit.SlabFit(
        True, False, 0, None, skipped_reason="not under test"))
    monkeypatch.setattr(cq.os.path, "exists", lambda _p: True)
    monkeypatch.setattr(cq.os.path, "getsize", lambda _p: BF16_BYTES)
    kwargs = {
        "path": f"/models/{BASE}", "unet_name": BASE, "model_options": {},
        "slab_weights": False, "slab_capable_path": True, "lora_low_rss": False,
        "fsdp_launch": False, "blocked_reason": "", "authoritative_slab_retry": False,
        "memoized_family": lambda _path: None, "vouched_families": frozenset({"krea2"}),
        "rescue_consent": None, "fit_probe": _probe,
        "file_identity": lambda _path: "1:2:3:4:5", "compile_dit": lambda: False,
        "comfy_managed": lambda: False, "lora_stack": LORA_STACK,
        "store_snapshot": _held() if snapshot is None else snapshot,
        "request_key": request_key,
    }
    kwargs.update(overrides)
    return cq.price(**kwargs)


def test_the_memory_saver_lora_change_fits_beside_the_base_it_reuses(monkeypatch):
    """The refused 2026-11-04 render, priced again: 38.6 GiB against 61.8 GiB."""
    row = _price(monkeypatch)
    assert row.would_load == cq.WOULD_LOAD_SWAP
    assert (row.verdict, row.rung) == (cq.VERDICT_FITS, cq.RUNG_EXPLICIT_STOCK)
    assert row.priced is True
    assert row.measured["probe"] == capacity_fit.PROBE_STOCK_LOAD_FIT
    assert row.measured["credit_bytes"] == BF16_BYTES
    assert row.measured["required_gib"] == 38.6
    assert row.measured["mem_available_gib"] == 61.8
    # The same row without the credit is the refusal the release test met.
    assert _probe(None, None).required_gib == 69.1 and not _probe(None, None).fits


@pytest.mark.parametrize(("snapshot", "request_key", "overrides", "why"), [
    (_held(swaps_in_place=False), WITH_LORA_KEY, {}, "the store would reload this base"),
    (_held(swaps_in_place=None), WITH_LORA_KEY, {}, "an older Worker states nothing"),
    ({"cond": {k: v for k, v in _held()["cond"].items() if k != "swaps_in_place"},
      "uncond": None, "cleanup_failed_slots": []}, WITH_LORA_KEY, {}, "no field at all"),
    (_held(residency="slab"), WITH_LORA_KEY, {}, "a slab resident is not a stock copy"),
    (_held(request_key=repr(("other.safetensors", (), (), "bf16"))), WITH_LORA_KEY, {},
     "another checkpoint is a fresh load"),
    (_held(), repr((BASE, (("weight_dtype", "fp8_e4m3fn"),), (), "bf16")), {},
     "a dtype change reloads the file"),
    ({"cond": None, "uncond": None, "cleanup_failed_slots": []}, WITH_LORA_KEY, {},
     "an empty slot is a fresh load"),
    (_held(), WITH_LORA_KEY, {"slab_weights": True, "slab_capable_path": False},
     "explicit slab drops the stock resident before it loads"),
])
def test_only_an_in_place_change_of_the_same_stock_base_is_credited(
        monkeypatch, snapshot, request_key, overrides, why):
    row = _price(monkeypatch, snapshot=snapshot, request_key=request_key, **overrides)
    assert row.verdict != cq.VERDICT_FITS, why
    assert "credit_bytes" not in (row.measured or {}), why
    assert (row.measured or {}).get("required_gib") == 69.1, why


def test_a_credited_price_that_still_falls_short_says_what_it_charged():
    """The card names the held base and the host copy, never a second file copy."""
    fit = StockFit(False, True, BF16_BYTES, int(30.0 * _GIB), credit_bytes=BF16_BYTES)
    assert fit.required_gib == 38.6 and not fit.fits
    quote = cq.RankPrice(
        verdict=cq.VERDICT_REFUSE, rung=cq.RUNG_EXPLICIT_STOCK, priced=True,
        measured=dict(fit.measured()), would_load=cq.WOULD_LOAD_SWAP,
        holds_resident={"cond": "stock", "uncond": None},
        checkpoint_bytes=BF16_BYTES, slot="cond")
    text = capacity_agreement._priced_sentence(
        RankAnswer(0, "box0", (quote,), box=box_names.HEAD), quote, BASE)
    assert text == (
        f"The head priced a LoRA change on {BASE}: the change needs 38.6 GiB "
        "beside the 30.5 GiB copy of this file the Worker already holds, room for a host "
        "copy of the weights its LoRA stack changes and the floor. Only 30.0 GiB of "
        "unified memory is available, 8.6 GiB short.")
    assert "placed from" not in text and "slot" not in text
    plain = StockFit(False, True, BF16_BYTES, int(30.0 * _GIB))
    assert "credit_bytes" not in plain.measured()
    assert capacity_fit.stock_terms(30.5) == "(a 30.5 GiB file plus the host copy it is placed from)"
    assert "this file the Worker already holds" in capacity_fit.stock_terms(30.5, BF16_BYTES)


@pytest.mark.parametrize(("stored", "low_rss", "expected"), [
    ({"base_baked": False}, False, True),                         # classic hot swap clones the base
    ({"base_baked": True}, False, False),                         # a baked base cannot be re-cloned
    ({"unbake": object(), "lora_sig": (("a", 1.0),)}, True, True),  # lazy swap restores in place
    ({"unbake": None, "lora_sig": ()}, True, True),               # nothing baked yet
    ({"unbake": None, "lora_sig": (("a", 1.0),)}, True, False),   # baked with no un-bake record
    ({"slab_auto_retry": True}, False, False),                    # reloads into slab
])
def test_the_store_says_when_a_lora_change_keeps_the_base(stored, low_rss, expected):
    fields = {"slab_auto_retry": False, "base_baked": False, "unbake": None, "lora_sig": ()}
    fields.update(stored)
    assert store_bake.swaps_in_place(types.SimpleNamespace(**fields), low_rss) is expected
    assert store_bake.swaps_in_place(None, low_rss) is False


def test_the_worker_quote_credits_the_base_its_own_store_holds(monkeypatch, tmp_path):
    """End to end on one rank: a real ModelStore snapshot reaches the price."""
    from dgx_monarch.actor import model_store
    from dgx_monarch.actor.store_detect import LivePrecisionEvidence
    from dgx_monarch.actor.stored_model import StoredModel

    store = model_store.ModelStore()
    store.lora_low_rss = False
    store.slab_weights = False
    store.current = StoredModel(
        base_key=(BASE, ()), request_key=(BASE, (), (), "bf16"), base_patcher=object(),
        active_patcher=object(), quant_kind="bf16",
        precision_evidence=LivePrecisionEvidence("bf16", "bf16"), family="wan_animate2",
        artifact_identity={}, residency_rung=cq.RUNG_EXPLICIT_STOCK)
    assert store.snapshot()["cond"]["swaps_in_place"] is True
    checkpoint = tmp_path / BASE
    with open(checkpoint, "wb") as handle:
        handle.truncate(BF16_BYTES)
    monkeypatch.setattr(model_store, "resolve_model_path", lambda _kind, _name: str(checkpoint))
    monkeypatch.setattr(model_store, "_slab_capable_path", lambda _path: False)
    monkeypatch.setattr(model_store, "_to_comfy_model_options", lambda options: dict(options or {}))
    monkeypatch.setattr(worker_capacity, "_pinned_staging", lambda: False)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: HEAD_AVAILABLE)
    worker = types.SimpleNamespace(rank=0, world=2, topology={}, _setup_key=("ready",),
                                   store=store, uma_reserve_gb=0.0)
    envelope = worker_capacity.quote(worker, {"specs": [
        {"unet_name": BASE, "options": {}, "loras": LORA_STACK, "slot": "cond"}],
        "rescue_consent": None})
    row = envelope["quotes"][0]
    assert (row["would_load"], row["verdict"]) == ("swap", "fits")
    assert row["measured"]["credit_bytes"] == BF16_BYTES
    # A base the low-RSS bake consumed reloads instead: the store drops the slot
    # first, so the stock swap credit (2026-11-05) credits the same copy.
    store.current.base_baked = True
    assert store.snapshot()["cond"]["swaps_in_place"] is False
    reloaded = worker_capacity.quote(worker, {"specs": [
        {"unet_name": BASE, "options": {}, "loras": LORA_STACK, "slot": "cond"}],
        "rescue_consent": None})["quotes"][0]
    assert reloaded["verdict"] == "fits" and reloaded["measured"]["credit_bytes"] == BF16_BYTES


def test_a_credited_sentence_names_its_box_in_plain_words():
    """Every branch of the sentence, the credited one included, names the box
    the way the rest of the refusal does (box_names.py), never by host."""
    fit = StockFit(False, True, BF16_BYTES, int(30.0 * _GIB), credit_bytes=BF16_BYTES)
    quote = cq.RankPrice(
        verdict=cq.VERDICT_REFUSE, rung=cq.RUNG_EXPLICIT_STOCK, priced=True,
        measured=dict(fit.measured()), would_load=cq.WOULD_LOAD_SWAP,
        holds_resident={"cond": "stock", "uncond": None},
        checkpoint_bytes=BF16_BYTES, slot="cond")
    text = capacity_agreement._priced_sentence(
        RankAnswer(1, "box1", (quote,), box=box_names.SIBLING), quote, BASE)
    assert text.startswith(f"The sibling priced a LoRA change on {BASE}:")
    assert "box1" not in text and "rank" not in text
