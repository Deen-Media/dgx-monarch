"""The classic LoRA bake's price, the in-place bake's, and the rule that picks between them.

The measured rises (one GPU of a DGX Spark under its clock cap, max 2100 MHz,
each over the same model's load with no stack; docs/DESIGN.md section 5.9):
LTX 2.3 scaled FP8 with 1660 patched weights 12.34 GiB and with 480 patched
weights 5.03; MiniMax H3 INT8 ConvRot with 208 patched weights 1.21; Qwen
Image plain FP8 with 720 patched weights 0.84. The in-place bake rose 8.28
GiB for a 7.08 GiB stack and 2.25 for a 0.305 GiB one. The real files are not
in CI, so these tests price counts and synthetic headers. CPU only.
"""
from __future__ import annotations

import dataclasses
import types

import pytest

from dgx_monarch import capacity_agreement, capacity_fit, stock_words
from dgx_monarch import capacity_classic_bake as price
from dgx_monarch.actor import in_place_fallback as fallback
from dgx_monarch.capacity_fit import StockFit
from test_capacity_lora_bake import _resolver, _write_safetensors

GIB, MIB = 2 ** 30, 2 ** 20
FLOOR = capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES
STACK = [{"name": "stack.safetensors", "strength": 0.8}]
# LTX 2.3's file and its two stacks, as recorded.
LTX_BYTES = int(21.85 * GIB)
LTX_OWN_STACK = int(7.08 * GIB)


def _weights(count: int, kind: str = price.OTHER, nbytes: int = 16 * MIB):
    return [(kind, nbytes)] * count


@pytest.mark.parametrize("count, kind, measured_gib", [
    (1660, price.FP8_QUANTIZED, 12.34), (480, price.FP8_QUANTIZED, 5.03),
    (208, price.OTHER, 1.21), (720, price.OTHER, 0.84)])
def test_the_line_is_never_under_a_measured_rise(count, kind, measured_gib):
    charged = price.term_from_weights(_weights(count, kind))
    assert charged == price.CLASSIC_BAKE_FIXED_BYTES + price.CLASSIC_BAKE_BYTES_PER_WEIGHT * count
    assert charged >= measured_gib * GIB
    assert charged <= (measured_gib + 6.5) * GIB  # a cover, not a blank cheque


def test_the_line_sits_within_a_sixth_of_both_ltx_points():
    assert price.term_from_weights(_weights(480, price.FP8_QUANTIZED)) / (5.03 * GIB) < 1.11
    assert price.term_from_weights(_weights(1660, price.FP8_QUANTIZED)) / (12.34 * GIB) < 1.16


def test_no_patched_weight_charges_nothing():
    assert price.term_from_weights([]) == 0


def test_a_quantized_fp8_weight_past_the_largest_measured_takes_the_bytes_guard():
    limit = price.MEASURED_LARGEST_WEIGHT_BYTES[price.FP8_QUANTIZED]
    at_the_limit = _weights(40, price.FP8_QUANTIZED, limit)
    past_it = at_the_limit[:-1] + [(price.FP8_QUANTIZED, limit + 1)]
    line = price.CLASSIC_BAKE_FIXED_BYTES + price.CLASSIC_BAKE_BYTES_PER_WEIGHT * 40
    assert price.term_from_weights(at_the_limit) == line
    guard = int(price.CLASSIC_BAKE_GUARD_RATIO * (39 * limit + limit + 1))
    assert price.term_from_weights(past_it) == max(line, guard)


def test_full_precision_weights_have_no_measured_point_and_are_always_guarded():
    weights = _weights(148, price.FULL_PRECISION, 110 * MIB)  # a BF16 file's patched layers
    line = price.CLASSIC_BAKE_FIXED_BYTES + price.CLASSIC_BAKE_BYTES_PER_WEIGHT * 148
    guard = int(0.75 * 148 * 110 * MIB)
    assert guard > line and price.term_from_weights(weights) == guard
    assert price.FULL_PRECISION not in price.MEASURED_LARGEST_WEIGHT_BYTES


def test_int8_and_plain_fp8_weights_are_charged_by_the_line_alone():
    weights = _weights(208, price.OTHER, 94 * MIB)
    assert price.term_from_weights(weights) == (price.CLASSIC_BAKE_FIXED_BYTES
                                                + price.CLASSIC_BAKE_BYTES_PER_WEIGHT * 208)


def _checkpoint(tmp_path):
    tensors = {
        "blocks.0.attn.weight": ("F8_E4M3", [4096, 4096]),
        "blocks.0.attn.comfy_quant": ("U8", [8]),
        "blocks.0.plain.weight": ("F8_E4M3", [1024, 1024]),
        "blocks.0.mlp.weight": ("BF16", [1024, 1024]),
        "blocks.0.int.weight": ("I8", [1024, 1024]),
        "blocks.0.int.weight_scale": ("F32", [1]),
        "blocks.0.mlp.bias": ("BF16", [1024]),
    }
    path = tmp_path / "base.safetensors"
    _write_safetensors(path, tensors)
    return str(path), tensors


def _lora(tmp_path, stems, name="stack.safetensors"):
    entries = {}
    for stem in stems:
        entries[f"diffusion_model.{stem}.lora_down.weight"] = ("BF16", [4, 4])
        entries[f"diffusion_model.{stem}.lora_up.weight"] = ("BF16", [4, 4])
    path = tmp_path / name
    _write_safetensors(path, entries)
    return str(path)


def test_a_weight_is_classed_from_the_header_alone(tmp_path):
    path, _tensors = _checkpoint(tmp_path)
    tensors = price.read_safetensors_header(path).tensors
    assert price.weight_class("blocks.0.attn.weight", tensors) == price.FP8_QUANTIZED
    assert price.weight_class("blocks.0.plain.weight", tensors) == price.OTHER  # FP8 with no quant marker
    assert price.weight_class("blocks.0.mlp.weight", tensors) == price.FULL_PRECISION
    assert price.weight_class("blocks.0.int.weight", tensors) == price.OTHER


def test_the_price_counts_the_weights_the_stack_patches(tmp_path):
    path, _tensors = _checkpoint(tmp_path)
    lora = _lora(tmp_path, ["blocks.0.attn", "blocks.0.int"])
    resolve = _resolver({"stack.safetensors": lora})
    tensors = price.read_safetensors_header(path).tensors
    assert price.patched_weights(tensors, STACK, resolve) == {"blocks.0.attn.weight", "blocks.0.int.weight"}
    assert price.classic_bake_bytes(path, STACK, resolve) == (
        price.CLASSIC_BAKE_FIXED_BYTES + 2 * price.CLASSIC_BAKE_BYTES_PER_WEIGHT)
    assert price.classic_bake_bytes(path, None, resolve) == 0
    assert price.classic_bake_bytes(path, [], resolve) == 0


def test_a_stack_it_cannot_map_or_read_charges_every_weight(tmp_path):
    path, _tensors = _checkpoint(tmp_path)
    tensors = price.read_safetensors_header(path).tensors
    every = {key for key in tensors if key.endswith((".weight", ".bias"))}
    stray = _lora(tmp_path, ["blocks.9.nowhere"])
    assert price.patched_weights(tensors, STACK, _resolver({"stack.safetensors": stray})) == every

    def missing(_name):
        raise FileNotFoundError("no such LoRA")

    assert price.patched_weights(tensors, STACK, missing) == every
    assert price.classic_bake_bytes(path, STACK, missing) >= (
        price.CLASSIC_BAKE_FIXED_BYTES + len(every) * price.CLASSIC_BAKE_BYTES_PER_WEIGHT)


def test_a_checkpoint_whose_header_cannot_be_read_is_charged_on_its_whole_size(tmp_path):
    path = tmp_path / "base.ckpt"
    path.write_bytes(b"not a safetensors file" * 100)
    lora = _lora(tmp_path, ["blocks.0.attn"])
    charged = price.classic_bake_bytes(str(path), STACK, _resolver({"stack.safetensors": lora}))
    assert charged == price.CLASSIC_BAKE_FIXED_BYTES + int(0.75 * path.stat().st_size)


def test_the_in_place_bake_charges_the_stack_twice_and_its_working_memory():
    assert price.in_place_bake_bytes(LTX_OWN_STACK, stack_read=False) == 2 * LTX_OWN_STACK + 2 * GIB
    assert price.in_place_bake_bytes(LTX_OWN_STACK, stack_read=True) == LTX_OWN_STACK + 2 * GIB
    # Never under the measured rises: 8.28 GiB for the 7.08 GiB stack, 2.25 for the 0.305 GiB one.
    assert price.in_place_bake_bytes(LTX_OWN_STACK, stack_read=True) >= 8.28 * GIB
    assert price.in_place_bake_bytes(int(0.305 * GIB), stack_read=True) >= 2.25 * GIB


def _fit(avail_gib: float, size: int = LTX_BYTES) -> StockFit:
    fit = StockFit(True, True, size, int(avail_gib * GIB))
    return StockFit(fit.required_bytes <= fit.avail_bytes, True, size, int(avail_gib * GIB))


@pytest.fixture
def ltx(monkeypatch):
    """LTX's own stack by its recorded figures: 14.2 GiB of classic bake, 7.08 GiB of files."""
    term = price.term_from_weights(_weights(1660, price.FP8_QUANTIZED))
    monkeypatch.setattr(price, "classic_bake_bytes", lambda _path, stack, _resolve: term if stack else 0)
    monkeypatch.setattr(price, "stack_bytes", lambda stack, _resolve: LTX_OWN_STACK if stack else 0)
    return term


def test_a_stock_price_with_no_stack_is_unchanged(ltx):
    fit = _fit(60.0)
    assert price.priced(fit, "p", None, None) is fit
    assert fit.bake_bytes == 0 and "classic bake" not in fit.terms and "bake" not in fit.measured()


def test_the_classic_bake_is_added_to_the_stock_price_and_named(ltx):
    roomy = price.priced(_fit(75.8), "p", STACK, None, stack_read=True)
    assert roomy.bake_bytes == ltx and roomy.fits
    assert roomy.required_bytes == capacity_fit.stock_required_bytes(LTX_BYTES) + ltx
    assert "GiB for the LoRA stack's classic bake" in roomy.terms
    assert roomy.measured()["bake"] == "classic" and roomy.measured()["bake_bytes"] == ltx
    # The record: this load used 54.6 GiB of an allowed 45.9; the price now allows what it used.
    assert roomy.required_bytes - FLOOR >= 54.6 * GIB
    # A price read before the stack is read charges the stack's files as well.
    assert price.priced(_fit(75.8), "p", STACK, None).bake_bytes == ltx + LTX_OWN_STACK


def test_todays_admitted_but_short_load_is_now_refused_by_the_classic_price(ltx):
    short = price.priced(_fit(52.0), "p", STACK, None, stack_read=True)  # today's 50.9 admits this
    assert _fit(52.0).fits and not short.fits and short.in_place_bytes == 0
    assert short.in_place_note == ""  # the question of an in-place bake did not arise


def test_with_the_permission_the_in_place_price_is_taken_where_the_classic_one_misses(ltx):
    held = int(21.85 * GIB)
    taken = price.priced(_fit(30.0), "p", STACK, None, stack_read=True, held_base_bytes=held, in_place=True)
    assert taken.fits and taken.in_place_bytes == LTX_OWN_STACK + 2 * GIB
    assert taken.required_bytes == taken.in_place_bytes + FLOOR  # no file term and no credit
    assert "bake the LoRA stack in place" in taken.terms
    assert taken.measured()["bake"] == "in_place"
    # A later credit cannot lower or raise an in-place price.
    assert dataclasses.replace(taken, credit_bytes=held).required_bytes == taken.required_bytes


def test_the_classic_bake_is_kept_wherever_it_fits_even_with_the_permission(ltx):
    held = int(21.85 * GIB)
    kept = price.priced(_fit(44.0), "p", STACK, None, stack_read=True, held_base_bytes=held, in_place=True)
    assert kept.in_place_bytes == 0 and kept.bake_bytes == ltx
    assert kept.required_bytes - held <= kept.avail_bytes  # fits once the drop returns the held copy


def test_without_the_permission_the_classic_price_decides_alone_and_says_why(ltx):
    held = int(21.85 * GIB)
    refused = price.priced(_fit(30.0), "p", STACK, None, stack_read=True, held_base_bytes=held,
                           in_place=price.NO_GATE_PASS)
    assert not refused.fits and refused.in_place_bytes == 0
    assert "baking it in place would need" in refused.in_place_note and "identity-gate PASS" in refused.terms
    assert refused.measured()["in_place_note"] == refused.in_place_note
    block = refused.measured()
    assert stock_words.bake_sentences(block).startswith(" Baking it in place would need")
    assert "for the LoRA stack's classic bake" in stock_words.stock_second_term(block)


def test_when_neither_price_fits_the_refusal_names_both(ltx):
    refused = price.priced(_fit(9.0), "p", STACK, None, stack_read=True,
                           held_base_bytes=int(21.85 * GIB), in_place=True)
    assert not refused.fits and refused.in_place_bytes == 0
    assert "classic bake" in refused.terms and "which is not available either" in refused.terms


def test_an_empty_slot_prices_the_bare_load_and_then_the_bake_beside_it(ltx):
    taken = price.priced(_fit(60.0), "p", STACK, None, in_place=True, bare_load=True)
    stock = capacity_fit.stock_required_bytes(LTX_BYTES)
    assert taken.fits and taken.in_place_bytes == max(stock - FLOOR, LTX_BYTES + 2 * LTX_OWN_STACK + 2 * GIB)
    assert not price.priced(_fit(49.0), "p", STACK, None, in_place=True, bare_load=True).fits


def test_the_probe_wraps_the_stock_probe_with_this_loads_terms(ltx):
    probe = price.probe(STACK, None, stack_read=True, stock_load_fit=lambda _path, _options: _fit(75.8))
    assert probe("p", {}).bake_bytes == ltx


def test_a_quote_request_carries_the_permission_only_as_true():
    specs = [{"unet_name": "m", "options": None, "loras": STACK, "slot": "cond"}]
    assert capacity_agreement.quote_request(specs)["in_place_bake_proven"] is False
    assert capacity_agreement.quote_request(specs, None, True)["in_place_bake_proven"] is True
    assert capacity_agreement.quote_request(specs, None, "yes")["in_place_bake_proven"] is False  # type: ignore[arg-type]


# The Worker's side: the permission, the slot rule and the decision.

def _stored(**changes):
    stored = types.SimpleNamespace(
        base_key=("m.safetensors", ()), lora_sig=(), base_baked=False, unbake=None, baked_in_place=False,
        base_patcher=types.SimpleNamespace(model=types.SimpleNamespace(model_loaded_weight_memory=LTX_BYTES)))
    for name, value in changes.items():
        setattr(stored, name, value)
    return stored


def _store(stored=None, low_rss=False):
    slots = {"cond": stored}
    return types.SimpleNamespace(lora_low_rss=low_rss, slab_weights="auto", slots=slots,
                                 _stored=lambda slot: slots.get(slot))


@pytest.fixture
def stock_resident(monkeypatch):
    from dgx_monarch.actor import store_residency

    monkeypatch.setattr(store_residency, "describe", lambda _stored: {"residency": "stock"})


def test_the_permission_is_true_only_for_the_value_true_and_only_inside_its_block():
    assert not fallback.permitted()
    with fallback.permission({price.IN_PLACE_BAKE_PROVEN: True}):
        assert fallback.permitted()
        with fallback.permission({price.IN_PLACE_BAKE_PROVEN: "true"}):
            assert not fallback.permitted()
        assert fallback.permitted()
    assert not fallback.permitted()
    for request in (None, {}, {price.IN_PLACE_BAKE_PROVEN: 1}, {price.IN_PLACE_BAKE_PROVEN: False}, "x"):
        with fallback.permission(request):
            assert not fallback.permitted()


def test_the_permission_is_withdrawn_when_the_load_raises():
    with pytest.raises(RuntimeError):
        with fallback.permission({price.IN_PLACE_BAKE_PROVEN: True}):
            raise RuntimeError("the load failed")
    assert not fallback.permitted()


def test_a_slot_baked_in_place_is_not_reused_with_the_switch_off_and_no_permission():
    baked = _stored(baked_in_place=True, base_baked=True)
    assert fallback.must_reload(_store(baked), baked)
    assert not fallback.must_reload(_store(baked, low_rss=True), baked)  # the default reuses its own bakes
    assert not fallback.must_reload(_store(), _stored())  # a classic slot
    with fallback.permission({price.IN_PLACE_BAKE_PROVEN: True}):
        assert not fallback.must_reload(_store(baked), baked)


def test_only_a_bare_stock_base_is_baked_in_place(stock_resident):
    assert fallback.eligibility(_store(), _stored(), True, True) is True
    assert fallback.eligibility(_store(), _stored(), True, False) == price.NO_GATE_PASS
    assert fallback.eligibility(_store(), None, False, True) is True  # an empty slot loads bare first
    for held in (_stored(lora_sig=(("a", 1.0),)), _stored(base_baked=True), _stored(unbake=object())):
        assert fallback.eligibility(_store(), held, True, True) == price.HOLDS_A_STACK
    assert fallback.eligibility(_store(low_rss=True), _stored(), True, True) == ""


def test_a_base_on_another_residency_is_not_baked_in_place(monkeypatch):
    from dgx_monarch.actor import store_residency

    monkeypatch.setattr(store_residency, "describe", lambda _stored: {"residency": "slab"})
    assert fallback.eligibility(_store(), _stored(), True, True) == price.HOLDS_A_STACK
    sharded = _stored()
    sharded.base_patcher._dgxm_fsdp = True
    monkeypatch.setattr(store_residency, "describe", lambda _stored: {"residency": "stock"})
    assert fallback.eligibility(_store(), sharded, True, True) == price.HOLDS_A_STACK


@pytest.fixture
def worker_side(monkeypatch, ltx, stock_resident):
    """``take``'s seams: the model path, the stock probe's reading and the store's in-place bake."""
    from dgx_monarch.actor import capacity_quote, comfy_bridge, store_bake

    state = types.SimpleNamespace(avail_gib=30.0, baked=[], fail=False)
    monkeypatch.setattr(comfy_bridge, "resolve_model_path", lambda _kind, name: f"/models/{name}")
    monkeypatch.setattr(capacity_quote, "default_resolve_lora_path", lambda name: f"/loras/{name}")
    monkeypatch.setattr(capacity_fit, "stock_load_fit", lambda _path, _options: _fit(state.avail_gib))

    def lazy_swap_transition(store, slot, unet_name, lora_stack, base_key, identity, *,
                             request_artifact_identity, logger):
        state.baked.append((slot, unet_name, base_key))
        if state.fail:
            store.slots[slot] = None  # the real transition drops the slot when the bake fails
            return None
        stored = store._stored(slot)
        stored.lora_sig, stored.base_baked, stored.baked_in_place, stored.unbake = (("s", 0.8),), True, True, object()
        return "patcher", "hot-swap"

    monkeypatch.setattr(store_bake, "lazy_swap_transition", lazy_swap_transition)
    return state


def _take(store, load_bare=lambda: pytest.fail("the base must not be loaded again")):
    return fallback.take(store, "cond", "m.safetensors", None, STACK, ("m.safetensors", ()), {}, None,
                         lambda *_args: {}, load_bare)


def test_without_the_permission_the_fallback_is_never_taken(worker_side):
    store = _store(_stored())
    assert _take(store) is None and worker_side.baked == []


def test_with_the_permission_and_a_classic_miss_the_stack_is_baked_in_place(worker_side, caplog):
    store = _store(_stored())
    with fallback.permission({price.IN_PLACE_BAKE_PROVEN: True}), caplog.at_level("INFO"):
        assert _take(store) == ("patcher", "hot-swap")
    assert worker_side.baked == [("cond", "m.safetensors", ("m.safetensors", ()))]
    slot = store._stored("cond")
    assert slot.unbake is None and slot.base_baked and slot.baked_in_place  # the next change reloads from disk
    assert "baked it in place on the loaded model" in caplog.text
    assert "the identity gate has passed this combination's in-place bake" in caplog.text


def test_with_the_permission_the_classic_bake_is_still_taken_where_it_fits(worker_side):
    worker_side.avail_gib = 44.0  # the drop returns 21.85, and the classic change then fits
    with fallback.permission({price.IN_PLACE_BAKE_PROVEN: True}):
        assert _take(_store(_stored())) is None
    assert worker_side.baked == []


def test_neither_price_fitting_leaves_the_resident_untouched(worker_side):
    worker_side.avail_gib = 9.0
    store = _store(_stored())
    with fallback.permission({price.IN_PLACE_BAKE_PROVEN: True}):
        assert _take(store) is None
    assert worker_side.baked == [] and store._stored("cond") is not None


def test_a_base_that_carries_a_classic_stack_is_not_baked_in_place(worker_side):
    with fallback.permission({price.IN_PLACE_BAKE_PROVEN: True}):
        assert _take(_store(_stored(lora_sig=(("old", 1.0),)))) is None
    assert worker_side.baked == []


def test_the_default_and_a_request_with_no_stack_never_reach_the_rule(worker_side):
    with fallback.permission({price.IN_PLACE_BAKE_PROVEN: True}):
        assert _take(_store(_stored(), low_rss=True)) is None
        assert fallback.take(_store(_stored()), "cond", "m.safetensors", None, None, ("m.safetensors", ()), {},
                             None, lambda *_args: {}, lambda: None) is None
    assert worker_side.baked == []


def test_an_empty_slot_loads_the_base_bare_and_then_bakes(worker_side):
    worker_side.avail_gib = 60.0  # the classic load needs 72.2 here; the bare base needs 50.9
    store = _store(None)
    loaded = []

    def load_bare():
        loaded.append("bare")
        store.slots["cond"] = _stored(base_key=("m.safetensors", (), "fp8"))

    with fallback.permission({price.IN_PLACE_BAKE_PROVEN: True}):
        assert _take(store, load_bare) == ("patcher", "hot-swap")
    # The baked slot is keyed as the loaded base is, whose detected kind the request did not know.
    assert loaded == ["bare"] and worker_side.baked == [("cond", "m.safetensors", ("m.safetensors", (), "fp8"))]


def test_a_bare_load_that_did_not_leave_a_bare_stock_base_goes_to_the_fresh_load(worker_side):
    worker_side.avail_gib = 60.0
    store = _store(None)
    with fallback.permission({price.IN_PLACE_BAKE_PROVEN: True}):
        assert _take(store, lambda: None) is None
    assert worker_side.baked == []


def test_a_bake_that_fails_returns_to_the_fresh_load(worker_side):
    worker_side.fail = True
    store = _store(_stored())
    with fallback.permission({price.IN_PLACE_BAKE_PROVEN: True}):
        assert _take(store) is None
    assert len(worker_side.baked) == 1 and store._stored("cond") is None


def test_the_fresh_loads_probe_charges_the_classic_bake_and_never_prices_in_place(worker_side):
    held = _store(_stored())
    assert fallback.fresh_load_probe(_store(_stored(), low_rss=True), "cond", ("m.safetensors", ()), STACK) is None
    assert fallback.fresh_load_probe(held, "cond", ("m.safetensors", ()), None) is None
    worker_side.avail_gib = 30.0
    refused = fallback.fresh_load_probe(held, "cond", ("m.safetensors", ()), STACK)("/models/m", {})
    assert not refused.fits and refused.in_place_bytes == 0
    assert refused.bake_bytes == worker_side_term()  # the stack was read by the swap attempt
    assert "identity-gate PASS" in refused.in_place_note
    with fallback.permission({price.IN_PLACE_BAKE_PROVEN: True}):
        again = fallback.fresh_load_probe(held, "cond", ("m.safetensors", ()), STACK)("/models/m", {})
    assert again.in_place_bytes == 0 and "that is not available either" in again.in_place_note
    # An empty slot has read nothing yet, so the stack's files are charged as well.
    empty = fallback.fresh_load_probe(_store(None), "cond", ("m.safetensors", ()), STACK)("/models/m", {})
    assert empty.bake_bytes == worker_side_term() + LTX_OWN_STACK


def worker_side_term() -> int:
    return price.CLASSIC_BAKE_FIXED_BYTES + price.CLASSIC_BAKE_BYTES_PER_WEIGHT * 1660
