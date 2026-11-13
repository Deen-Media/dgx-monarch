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
    (1660, price.FP8_QUANTIZED, 12.34), (480, price.FP8_QUANTIZED, 5.03)])
def test_the_line_is_never_under_a_measured_scaled_fp8_rise(count, kind, measured_gib):
    charged = price.term_from_weights(_weights(count, kind))
    assert charged == price.CLASSIC_BAKE_FIXED_BYTES + price.CLASSIC_BAKE_BYTES_PER_WEIGHT * count
    assert charged >= measured_gib * GIB
    assert charged <= (measured_gib + 6.5) * GIB  # a cover, not a blank cheque


@pytest.mark.parametrize("count, kind, measured_gib", [
    (208, price.INT8_QUANTIZED, 1.21), (208, price.INT8_QUANTIZED, 0.99),
    (720, price.PLAIN_FP8, 0.84), (720, price.PLAIN_FP8, 1.74)])
def test_int8_convrot_and_plain_fp8_take_their_own_share_which_covers_their_measured_rises(count, kind, measured_gib):
    """MiniMax H3 INT8 ConvRot (208 patched weights) and Qwen Image plain FP8 (720): two reads each, 2026-11-12."""
    charged = price.term_from_weights(_weights(count, kind))
    assert charged == price.CLASSIC_BAKE_FIXED_BYTES + count * price.LIGHT_SHARE_BYTES[kind] >= measured_gib * GIB
    assert price.LIGHT_SHARE_BYTES == {price.INT8_QUANTIZED: 4 * MIB, price.PLAIN_FP8: 0}
    assert price.LIGHT_SHARE_BYTES[price.INT8_QUANTIZED] < price.CLASSIC_BAKE_BYTES_PER_WEIGHT


def test_the_line_sits_within_a_sixth_of_both_ltx_points():
    assert price.term_from_weights(_weights(480, price.FP8_QUANTIZED)) / (5.03 * GIB) < 1.11
    assert price.term_from_weights(_weights(1660, price.FP8_QUANTIZED)) / (12.34 * GIB) < 1.16


def test_no_patched_weight_charges_nothing():
    assert price.term_from_weights([]) == 0


def test_the_bytes_guard_is_off_and_one_constant_turns_it_on_for_a_format(monkeypatch):
    """No record needs the guard: both LTX stacks' largest patched weight is the 64 MiB it starts past."""
    limit = price.MEASURED_LARGEST_WEIGHT_BYTES[price.FP8_QUANTIZED]
    at_the_limit = _weights(60, price.FP8_QUANTIZED, limit)  # enough of them that the guard would bind
    past_it = [*at_the_limit[:-1], (price.FP8_QUANTIZED, limit + 1)]
    line = price.CLASSIC_BAKE_FIXED_BYTES + price.CLASSIC_BAKE_BYTES_PER_WEIGHT * 60
    guard = int(price.CLASSIC_BAKE_GUARD_RATIO * (59 * limit + limit + 1))
    assert guard > line and price.GUARDED_BY_BYTES == ()
    assert price.term_from_weights(at_the_limit) == line and price.term_from_weights(past_it) == line
    monkeypatch.setattr(price, "GUARDED_BY_BYTES", (price.FP8_QUANTIZED,))
    assert price.term_from_weights(at_the_limit) == line  # at the measured size: the line alone
    assert price.term_from_weights(past_it) == guard  # one byte past it: the guard
    assert price.term_from_weights(past_it, guard=False) == line  # the partial-load route's own rule


def test_full_precision_weights_are_charged_by_the_line_and_no_share_of_their_bytes(monkeypatch):
    """Two BF16 cells sit under the line (121 weights rose 1.00 GiB, 1261 at most 7.4), so no bytes are charged.

    One constant decides it: with the format named in ``GUARDED_BY_BYTES`` the
    same weights take the bytes guard.
    """
    weights = _weights(148, price.FULL_PRECISION, 110 * MIB)  # a BF16 file's patched layers
    line = price.CLASSIC_BAKE_FIXED_BYTES + price.CLASSIC_BAKE_BYTES_PER_WEIGHT * 148
    guard = int(0.75 * 148 * 110 * MIB)
    assert guard > line and price.term_from_weights(weights) == line
    assert price.term_from_weights(_weights(121, price.FULL_PRECISION, 140 * MIB)) >= 1.00 * GIB
    assert price.term_from_weights(_weights(1261, price.FULL_PRECISION, 24 * MIB)) >= 7.4 * GIB
    monkeypatch.setattr(price, "GUARDED_BY_BYTES", (price.FULL_PRECISION,))
    assert price.term_from_weights(weights) == guard  # no measured largest weight: guarded whatever the size


def test_a_weight_with_no_record_is_charged_by_the_line():
    weights = _weights(208, price.OTHER, 94 * MIB)
    assert price.term_from_weights(weights) == (price.CLASSIC_BAKE_FIXED_BYTES
                                                + price.CLASSIC_BAKE_BYTES_PER_WEIGHT * 208)


def test_the_stacks_files_are_charged_on_a_reading_taken_before_they_are_read_and_not_after():
    """Every class the same way: the files are part of what a load takes from a box that has not read them."""
    files = int(0.7 * GIB)
    fixed, per_weight = price.CLASSIC_BAKE_FIXED_BYTES, price.CLASSIC_BAKE_BYTES_PER_WEIGHT
    int8 = price.LIGHT_SHARE_BYTES[price.INT8_QUANTIZED]
    terms = {
        price.FP8_QUANTIZED: (_weights(480, price.FP8_QUANTIZED), fixed + 480 * per_weight),
        price.INT8_QUANTIZED: (_weights(480, price.INT8_QUANTIZED), fixed + 480 * int8),
        price.PLAIN_FP8: (_weights(720, price.PLAIN_FP8), fixed),
        price.FULL_PRECISION: (_weights(148, price.FULL_PRECISION), fixed + 148 * per_weight),
        price.OTHER: (_weights(9, price.OTHER, 0), fixed + 9 * per_weight),
    }
    for weights, term in terms.values():
        assert price.term_from_weights(weights) == term
        assert price.charge_from_weights(weights, files) == term + files  # an empty slot, and every quote
        assert price.charge_from_weights(weights, files, stack_on_host=True) == term  # after a declined hot swap
    assert price.charge_from_weights([], files) == files  # a stack that patches no weight still reads its files
    assert price.charge_from_weights([], files, stack_on_host=True) == 0


def test_a_file_listed_more_than_once_is_charged_as_often_as_the_worker_reads_it():
    """The line counts distinct patched keys; the Worker reads the file for each entry (block C, 2026-11-12)."""
    fixed, per_weight = price.CLASSIC_BAKE_FIXED_BYTES, price.CLASSIC_BAKE_BYTES_PER_WEIGHT
    weights = _weights(1660, price.FP8_QUANTIZED, 11 * MIB)
    line = fixed + 1660 * per_weight
    once, twice = LTX_OWN_STACK, 2 * LTX_OWN_STACK
    assert price.charge_from_weights(weights, once) == line + once
    assert price.charge_from_weights(weights, twice) == line + twice  # the count has not moved, the files have
    # From an empty Worker the single stack took 62.5 GiB and the doubled one 69.5: 7.0 apart, the file.
    assert (price.charge_from_weights(weights, twice) - price.charge_from_weights(weights, once)) == once
    assert price.charge_from_weights(weights, twice, stack_on_host=True) == line


def test_one_stray_key_of_another_format_does_not_change_how_a_stack_is_charged():
    """A stack is read by the format that holds most of its patched bytes."""
    fixed, per_weight = price.CLASSIC_BAKE_FIXED_BYTES, price.CLASSIC_BAKE_BYTES_PER_WEIGHT
    int8_share = price.LIGHT_SHARE_BYTES[price.INT8_QUANTIZED]
    small = _weights(288, price.FULL_PRECISION, 10 * 1024)
    # LTX's shape: scaled FP8 weights, their small keys, and a few BF16 weights, all on the line. One
    # unmarked FP8 key among them takes its own class's share (nothing); the small keys keep the line's.
    scaled = _weights(1344, price.FP8_QUANTIZED, 13 * MIB) + small + _weights(26, price.FULL_PRECISION, 20 * MIB)
    assert price.term_from_weights(scaled) == fixed + (1344 + 288 + 26) * per_weight
    assert price.term_from_weights([*scaled, (price.PLAIN_FP8, 13 * MIB)]) == fixed + (1344 + 288 + 26) * per_weight
    # An INT8 stack: its weights and its small keys take the INT8 share. One scaled FP8 weight among
    # them takes the line's share, and the small keys still take INT8's.
    int8 = _weights(480, price.INT8_QUANTIZED, 30 * MIB) + small
    assert price.term_from_weights(int8) == fixed + (480 + 288) * int8_share
    stray_scaled = [*int8, (price.FP8_QUANTIZED, 30 * MIB)]
    assert price.term_from_weights(stray_scaled) == fixed + (480 + 288) * int8_share + per_weight
    # A plain FP8 stack with the same small keys: nothing for either.
    assert price.term_from_weights(_weights(720, price.PLAIN_FP8, 26 * MIB) + small) == fixed
    # Half and half is neither format's: the small keys take the line's share, the dearer reading.
    even = _weights(10, price.INT8_QUANTIZED, 30 * MIB) + _weights(10, price.FP8_QUANTIZED, 30 * MIB)
    assert price.term_from_weights(even) == fixed + 10 * int8_share + 10 * per_weight
    assert price.term_from_weights(even + small) == fixed + 10 * int8_share + (10 + 288) * per_weight
    # Two light classes in one stack: the small keys follow the one that holds more bytes.
    both = _weights(10, price.INT8_QUANTIZED, 30 * MIB) + _weights(30, price.PLAIN_FP8, 30 * MIB) + small
    assert price.term_from_weights(both) == fixed + 10 * int8_share


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
    assert price.weight_class("blocks.0.plain.weight", tensors) == price.PLAIN_FP8  # FP8 with no quant marker
    assert price.weight_class("blocks.0.mlp.weight", tensors) == price.FULL_PRECISION
    assert price.weight_class("blocks.0.int.weight", tensors) == price.INT8_QUANTIZED  # INT8 with its scale


def test_the_price_counts_the_weights_the_stack_patches(tmp_path):
    path, _tensors = _checkpoint(tmp_path)
    lora = _lora(tmp_path, ["blocks.0.attn", "blocks.0.int"])
    resolve = _resolver({"stack.safetensors": lora})
    tensors = price.read_safetensors_header(path).tensors
    assert price.patched_weights(tensors, STACK, resolve) == {"blocks.0.attn.weight", "blocks.0.int.weight"}
    # The scaled FP8 weight carries the line's share; the INT8 ConvRot weight beside it carries its own.
    assert price.classic_bake_bytes(path, STACK, resolve) == (
        price.CLASSIC_BAKE_FIXED_BYTES + price.CLASSIC_BAKE_BYTES_PER_WEIGHT
        + price.LIGHT_SHARE_BYTES[price.INT8_QUANTIZED])
    both_shared = _lora(tmp_path, ["blocks.0.attn", "blocks.0.mlp"], name="shared.safetensors")
    assert price.classic_bake_bytes(path, [{"name": "shared.safetensors"}], _resolver(
        {"shared.safetensors": both_shared})) == (
        price.CLASSIC_BAKE_FIXED_BYTES + 2 * price.CLASSIC_BAKE_BYTES_PER_WEIGHT)
    assert price.classic_bake_bytes(path, None, resolve) == 0
    assert price.classic_bake_bytes(path, [], resolve) == 0


def test_comfyuis_other_lora_spellings_map_to_the_keys_they_patch(tmp_path):
    """A LoRA spelled ``.lora.down.weight`` named no suffix the mapping knew, and was counted as every weight.

    On hardware (2026-11-12) the Worker counted 240 patched weights for such a
    Qwen Image LoRA and the header mapping gave 1933, the whole model.
    """
    path, _tensors = _checkpoint(tmp_path)
    tensors = price.read_safetensors_header(path).tensors
    spellings = [(".lora.down.weight", ".lora.up.weight"), ("_lora.down.weight", "_lora.up.weight"),
                 (".lora_linear_layer.down.weight", ".lora_linear_layer.up.weight"),
                 (".lora_A.default.weight", ".lora_B.default.weight"), (".lora_A", ".lora_B")]
    for index, (down, up) in enumerate(spellings):
        name = f"spelled{index}.safetensors"
        _write_safetensors(tmp_path / name, {f"transformer.blocks.0.attn{down}": ("BF16", [4, 4]),
                                             f"transformer.blocks.0.attn{up}": ("BF16", [4, 4])})
        assert price.patched_weights(tensors, [{"name": name}], _resolver({name: str(tmp_path / name)})) == {
            "blocks.0.attn.weight"}, down
    _write_safetensors(tmp_path / "norms.safetensors", {
        "transformer.blocks.0.mlp.w_norm": ("BF16", [1024]), "transformer.blocks.0.mlp.b_norm": ("BF16", [1024]),
        "transformer.blocks.0.attn.set_weight": ("BF16", [4, 4])})
    assert price.patched_weights(tensors, [{"name": "norms.safetensors"}], _resolver(
        {"norms.safetensors": str(tmp_path / "norms.safetensors")})) == {
        "blocks.0.mlp.weight", "blocks.0.mlp.bias", "blocks.0.attn.weight"}


def test_a_stack_it_cannot_map_or_read_charges_every_weight(tmp_path):
    path, _tensors = _checkpoint(tmp_path)
    tensors = price.read_safetensors_header(path).tensors
    every = {key for key in tensors if key.endswith((".weight", ".bias"))}
    stray = _lora(tmp_path, ["blocks.9.nowhere"])
    assert price.patched_weights(tensors, STACK, _resolver({"stack.safetensors": stray})) == every

    def missing(_name):
        raise FileNotFoundError("no such LoRA")

    assert price.patched_weights(tensors, STACK, missing) == every
    # Every weight is priced by its own class. This header has an INT8 ConvRot weight and a plain FP8
    # one, which take their own shares (4 MiB and nothing). Most of its bytes are the scaled FP8
    # weight, so the rest is counted as that format is: the scaled FP8 weight, the BF16 weight and the
    # bias each carry the line's share.
    assert len(every) == 5
    assert price.classic_bake_bytes(path, STACK, missing) == (
        price.CLASSIC_BAKE_FIXED_BYTES + 3 * price.CLASSIC_BAKE_BYTES_PER_WEIGHT
        + price.LIGHT_SHARE_BYTES[price.INT8_QUANTIZED])


def test_a_checkpoint_whose_header_cannot_be_read_is_charged_as_the_largest_stack_measured(tmp_path):
    path = tmp_path / "base.ckpt"
    path.write_bytes(b"not a safetensors file" * 100)
    lora = _lora(tmp_path, ["blocks.0.attn"])
    charged = price.classic_bake_bytes(str(path), STACK, _resolver({"stack.safetensors": lora}))
    assert price.UNREADABLE_STACK_WEIGHTS == 1660  # LTX 2.3's own stack
    assert charged == price.CLASSIC_BAKE_FIXED_BYTES + 1660 * price.CLASSIC_BAKE_BYTES_PER_WEIGHT


def test_the_in_place_bake_charges_one_copy_of_the_stack_and_its_working_memory():
    assert price.in_place_bake_bytes(LTX_OWN_STACK) == LTX_OWN_STACK + 2 * GIB
    # Never under the measured rises: 8.28 GiB for the 7.08 GiB stack, 2.25 for the 0.305 GiB one.
    assert price.in_place_bake_bytes(LTX_OWN_STACK) >= 8.28 * GIB
    assert price.in_place_bake_bytes(int(0.305 * GIB)) >= 2.25 * GIB


def _fit(avail_gib: float, size: int = LTX_BYTES) -> StockFit:
    fit = StockFit(True, True, size, int(avail_gib * GIB))
    return StockFit(fit.required_bytes <= fit.avail_bytes, True, size, int(avail_gib * GIB))


@pytest.fixture
def ltx(monkeypatch):
    """LTX's own stack by its recorded figures: a 14.2 GiB term and 7.08 GiB of files.

    What a classic price adds for it on a reading taken before the file is
    read: 21.2 GiB, the term and the file. On a reading taken after
    (``stack_on_host``), the term alone.
    """
    term = price.term_from_weights(_weights(1660, price.FP8_QUANTIZED))
    charge = term + LTX_OWN_STACK

    def classic_charge(_path, stack, _resolve, *, stack_on_host=False):
        return 0 if not stack else term if stack_on_host else charge

    monkeypatch.setattr(price, "classic_charge", classic_charge)
    monkeypatch.setattr(price, "stack_bytes", lambda stack, _resolve: LTX_OWN_STACK if stack else 0)
    return charge  # what this stack's classic price adds from a box that has not read its file


HELD = int(21.85 * GIB)  # what dropping the bare LTX base returns


def test_a_stock_price_with_no_stack_is_unchanged(ltx):
    fit = _fit(60.0)
    assert price.priced(fit, "p", None, None) is fit
    assert fit.bake_bytes == 0 and "classic bake" not in fit.terms and "bake" not in fit.measured()


def test_the_classic_bake_and_the_stacks_files_are_added_to_the_stock_price_and_named(ltx):
    roomy = price.priced(_fit(75.8), "p", STACK, None)
    assert roomy.bake_bytes == ltx and roomy.fits
    assert roomy.required_bytes == capacity_fit.stock_required_bytes(LTX_BYTES) + ltx
    assert "GiB for the LoRA stack's classic bake" in roomy.terms
    assert roomy.measured()["bake"] == "classic" and roomy.measured()["bake_bytes"] == ltx
    # The record: this load used 54.6 GiB of an allowed 45.9; the price now allows what it used.
    assert roomy.required_bytes - FLOOR >= 54.6 * GIB


def test_the_bake_is_chosen_against_what_an_earlier_slot_leaves_and_the_reading_comes_back_as_read(ltx):
    """A pair's second model: the choice is made after the first slot's charge; the quote takes it off once."""
    read = int(80.0 * GIB)
    alone = price.priced(_fit(80.0), "p", STACK, None, in_place=True, bare_load=True)
    assert alone.fits and alone.in_place_bytes == 0 and alone.bake_bytes == ltx  # the classic bake fits the reading
    spent = int(25 * GIB)
    second = price.priced(_fit(80.0), "p", STACK, None, in_place=True, bare_load=True, charged=spent)
    assert second.in_place_bytes > 0 and second.fits and second.avail_bytes == read
    assert second.required_bytes <= read - spent < alone.required_bytes
    short = price.priced(_fit(80.0), "p", STACK, None, in_place=True, bare_load=True, charged=int(40 * GIB))
    assert not short.fits and short.in_place_bytes == 0 and short.avail_bytes == read
    assert "and 40.0 GiB was left beside this request's other model" in short.in_place_note
    # No earlier slot: the same choice as before, and the reading is named as read.
    assert price.priced(_fit(80.0), "p", STACK, None, in_place=True, bare_load=True, charged=0) == alone
    assert "GiB was read" in price.priced(_fit(40.0), "p", STACK, None, in_place=True, bare_load=True).in_place_note


def test_todays_admitted_but_short_load_is_now_refused_by_the_classic_price(ltx):
    short = price.priced(_fit(52.0), "p", STACK, None)  # today's 50.9 admits this
    assert _fit(52.0).fits and not short.fits and short.in_place_bytes == 0
    assert short.in_place_note == ""  # the question of an in-place bake did not arise


def test_with_the_permission_the_in_place_price_is_taken_where_the_classic_one_misses(ltx):
    taken = price.priced(_fit(30.0), "p", STACK, None, held_bytes=HELD, in_place=True)
    assert taken.fits and taken.in_place_bytes == LTX_OWN_STACK + 2 * GIB
    assert taken.required_bytes == taken.in_place_bytes + FLOOR  # no file term and no credit
    assert "bake the LoRA stack in place" in taken.terms
    assert taken.measured()["bake"] == "in_place"
    # A later credit cannot lower or raise an in-place price.
    assert dataclasses.replace(taken, credit_bytes=HELD).required_bytes == taken.required_bytes


def test_the_classic_bake_is_kept_wherever_it_fits_even_with_the_permission(ltx):
    kept = price.priced(_fit(51.0), "p", STACK, None, held_bytes=HELD, in_place=True)
    assert kept.in_place_bytes == 0 and kept.bake_bytes == ltx
    assert kept.required_bytes - HELD <= kept.avail_bytes  # fits once the drop returns the held copy
    # One GiB under that, it does not, and the in-place bake is taken.
    assert price.priced(_fit(49.5), "p", STACK, None, held_bytes=HELD, in_place=True).in_place_bytes > 0


def test_without_the_permission_the_classic_price_decides_alone_and_says_why(ltx):
    refused = price.priced(_fit(30.0), "p", STACK, None, held_bytes=HELD, in_place=price.GATE_HAS_NOT_PASSED)
    assert not refused.fits and refused.in_place_bytes == 0
    assert refused.in_place_note == "; " + price.GATE_HAS_NOT_PASSED and "identity-gate PASS" in refused.terms
    block = refused.measured()
    assert block["in_place_note"] == refused.in_place_note
    assert stock_words.bake_sentences(block).startswith(" Baking it in place was not tried")
    assert stock_words.bake_sentences(block).endswith("prove it.")
    assert "for the LoRA stack's classic bake" in stock_words.stock_second_term(block)
    assert " That includes " in stock_words.bake_sentences(block, name_the_term=True)


def test_when_neither_price_fits_the_refusal_names_both_and_the_reading(ltx):
    refused = price.priced(_fit(9.0), "p", STACK, None, held_bytes=HELD, in_place=True)
    assert not refused.fits and refused.in_place_bytes == 0
    assert "classic bake" in refused.terms
    assert "baking it in place would need 14.1 GiB beside the" in refused.terms
    assert "and 9.0 GiB was read" in refused.terms


def test_an_empty_slot_prices_the_bare_load_and_then_the_bake_beside_it(ltx):
    taken = price.priced(_fit(60.0), "p", STACK, None, in_place=True, bare_load=True)
    stock = capacity_fit.stock_required_bytes(LTX_BYTES)
    assert taken.fits and taken.in_place_bytes == max(stock - FLOOR, LTX_BYTES + LTX_OWN_STACK + 2 * GIB)
    assert not price.priced(_fit(49.0), "p", STACK, None, in_place=True, bare_load=True).fits


def test_a_slot_that_holds_something_else_is_credited_what_its_drop_returns(ltx):
    other = int(20 * GIB)
    # 35 read: the classic needs 72.1 less the 20 returned, the bare route 50.9 less the 20.
    taken = price.priced(_fit(35.0), "p", STACK, None, held_bytes=other, in_place=True, bare_load=True)
    assert taken.fits and taken.in_place_bytes == capacity_fit.stock_required_bytes(LTX_BYTES) - FLOOR - other
    # With the classic fitting once that model is gone, the classic is kept.
    assert price.priced(_fit(53.0), "p", STACK, None, held_bytes=other, in_place=True,
                        bare_load=True).in_place_bytes == 0


def test_the_probe_wraps_the_stock_probe_with_this_loads_terms(ltx):
    probe = price.probe(STACK, None, stock_load_fit=lambda _path, _options: _fit(75.8))
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
        base_patcher=types.SimpleNamespace(
            backup={}, model=types.SimpleNamespace(model_loaded_weight_memory=HELD)))
    for name, value in changes.items():
        setattr(stored, name, value)
    return stored


def _store(stored=None, low_rss=False):
    slots = {"cond": stored}
    dropped: list[str] = []

    def drop(slot):
        dropped.append(slot)
        slots[slot] = None

    return types.SimpleNamespace(lora_low_rss=low_rss, slab_weights="auto", slots=slots, dropped=dropped,
                                 _stored=lambda slot: slots.get(slot), _drop=drop)


@pytest.fixture(autouse=True)
def stock_resident(monkeypatch):
    from dgx_monarch.actor import store_residency

    monkeypatch.setattr(store_residency, "describe", lambda _stored: {"residency": "stock"})


ALLOW = {price.IN_PLACE_BAKE_PROVEN: True}


def test_the_permission_is_true_only_for_the_value_true_and_only_inside_its_block():
    assert not fallback.permitted()
    with fallback.permission(ALLOW):
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
        with fallback.permission(ALLOW):
            raise RuntimeError("the load failed")
    assert not fallback.permitted()


def test_a_slot_baked_in_place_is_not_reused_with_the_switch_off_and_no_permission():
    baked = _stored(baked_in_place=True, base_baked=True)
    assert fallback.must_reload(_store(baked), baked)
    assert not fallback.must_reload(_store(baked), baked, True)
    assert not fallback.must_reload(_store(baked, low_rss=True), baked)  # the default reuses its own bakes
    assert not fallback.must_reload(_store(), _stored())  # a classic slot
    with fallback.permission(ALLOW):
        assert not fallback.must_reload(_store(baked), baked)
        assert fallback.must_reload(_store(baked), baked, False)  # the quote's own permission decides there


def test_a_base_is_bare_only_by_the_stores_record_and_by_comfyuis_backup(monkeypatch):
    from dgx_monarch.actor import store_residency

    assert fallback.is_bare_stock(_stored())
    for held in (_stored(lora_sig=(("a", 1.0),)), _stored(base_baked=True), _stored(unbake=object())):
        assert not fallback.is_bare_stock(held)
    swapped_to_bare = _stored()
    swapped_to_bare.base_patcher.backup = {"blocks.0.attn.weight": object()}  # the old stack's originals
    assert not fallback.is_bare_stock(swapped_to_bare)
    sharded = _stored()
    sharded.base_patcher._dgxm_fsdp = True
    assert not fallback.is_bare_stock(sharded) and fallback.held_bytes(sharded) == 0
    monkeypatch.setattr(store_residency, "describe", lambda _stored: {"residency": "slab"})
    assert not fallback.is_bare_stock(_stored()) and fallback.held_bytes(_stored()) == 0


def test_the_route_is_the_same_arguments_for_every_caller():
    bare = _stored()
    assert fallback.route(_store(), bare, True, True) == {"held_bytes": HELD, "bare_load": False, "in_place": True}
    assert fallback.route(_store(), bare, True, False)["in_place"] == price.GATE_HAS_NOT_PASSED
    # Nothing held, another model held, or this base with a stack: the base is reloaded bare first.
    assert fallback.route(_store(), None, False, True) == {"held_bytes": 0, "bare_load": True, "in_place": True}
    assert fallback.route(_store(), bare, False, True) == {"held_bytes": HELD, "bare_load": True, "in_place": True}
    assert fallback.route(_store(), _stored(base_baked=True), True, True)["bare_load"] is True
    assert fallback.route(_store(low_rss=True), bare, True, True) is None  # the default: no classic bake


def test_the_quote_does_not_read_a_slot_the_store_will_reload_as_a_reuse():
    baked = _stored(baked_in_place=True, base_baked=True)
    snapshot = {"cond": {"request_key": "k", "swaps_in_place": True, "residency": "stock"}, "uncond": None}
    assert fallback.quote_snapshot(_store(baked), baked, snapshot, "cond", True) is snapshot
    priced_as_a_load = fallback.quote_snapshot(_store(baked), baked, snapshot, "cond", False)
    assert priced_as_a_load["cond"]["request_key"] is None and priced_as_a_load["cond"]["residency"] == "stock"
    assert snapshot["cond"]["request_key"] == "k"  # the caller's snapshot is not edited
    assert fallback.quote_snapshot(_store(), _stored(), snapshot, "cond", False) is snapshot


@pytest.fixture
def worker_side(monkeypatch, ltx):
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
    assert _take(store) is None and worker_side.baked == [] and store.dropped == []


def test_with_the_permission_and_a_classic_miss_the_stack_is_baked_in_place(worker_side, caplog):
    store = _store(_stored())
    with fallback.permission(ALLOW), caplog.at_level("INFO"):
        assert _take(store) == ("patcher", "hot-swap")
    assert worker_side.baked == [("cond", "m.safetensors", ("m.safetensors", ()))] and store.dropped == []
    slot = store._stored("cond")
    assert slot.unbake is None and slot.base_baked and slot.baked_in_place  # the next change reloads the base
    assert "baked it in place on the loaded model" in caplog.text
    assert "the identity gate has passed this combination's in-place bake" in caplog.text


def test_with_the_permission_the_classic_bake_is_still_taken_where_it_fits(worker_side):
    worker_side.avail_gib = 51.0  # the drop returns 21.85, and the classic change then fits
    with fallback.permission(ALLOW):
        assert _take(_store(_stored())) is None
    assert worker_side.baked == []


def test_neither_price_fitting_leaves_the_resident_untouched_and_logs_the_figures(worker_side, caplog):
    worker_side.avail_gib = 9.0
    store = _store(_stored())
    with fallback.permission(ALLOW), caplog.at_level("INFO"):
        assert _take(store) is None
    assert worker_side.baked == [] and store._stored("cond") is not None and store.dropped == []
    assert "does not fit here; baking it in place would need 14.1 GiB" in caplog.text


def test_a_base_that_carries_a_stack_is_dropped_and_reloaded_bare_never_baked_on(worker_side):
    worker_side.avail_gib = 35.0  # the bare route needs 50.9 less the 21.85 its drop returns
    for held in (_stored(lora_sig=(("old", 1.0),)), _stored(base_baked=True, baked_in_place=True)):
        store = _store(held)
        loaded = []

        def load_bare(store=store, loaded=loaded):
            assert store._stored("cond") is None  # the stacked base went first
            loaded.append("bare")
            store.slots["cond"] = _stored()

        with fallback.permission(ALLOW):
            assert _take(store, load_bare) == ("patcher", "hot-swap")
        assert store.dropped == ["cond"] and loaded == ["bare"]


def test_another_models_bare_base_is_dropped_and_this_base_loaded_never_baked_on(worker_side):
    """A bare stock base, but another model's: the slot holds this base only by the whole key."""
    worker_side.avail_gib = 35.0
    store = _store(_stored(base_key=("other.safetensors", ())))
    loaded = []

    def load_bare():
        assert store._stored("cond") is None  # the other model went first
        loaded.append("bare")
        store.slots["cond"] = _stored()

    with fallback.permission(ALLOW):
        assert _take(store, load_bare) == ("patcher", "hot-swap")
    assert store.dropped == ["cond"] and loaded == ["bare"]
    assert worker_side.baked == [("cond", "m.safetensors", ("m.safetensors", ()))]


def test_the_default_and_a_request_with_no_stack_never_reach_the_rule(worker_side):
    with fallback.permission(ALLOW):
        assert _take(_store(_stored(), low_rss=True)) is None
        assert fallback.take(_store(_stored()), "cond", "m.safetensors", None, None, ("m.safetensors", ()), {},
                             None, lambda *_args: {}, lambda: None) is None
    assert worker_side.baked == []


def test_an_empty_slot_loads_the_base_bare_and_then_bakes(worker_side):
    worker_side.avail_gib = 60.0  # the classic load needs 72.1 here; the bare base needs 50.9
    store = _store(None)
    loaded = []

    def load_bare():
        loaded.append("bare")
        store.slots["cond"] = _stored(base_key=("m.safetensors", (), "fp8"))

    with fallback.permission(ALLOW):
        assert _take(store, load_bare) == ("patcher", "hot-swap")
    # The baked slot is keyed as the loaded base is, whose detected kind the request did not know.
    assert loaded == ["bare"] and store.dropped == []
    assert worker_side.baked == [("cond", "m.safetensors", ("m.safetensors", (), "fp8"))]


def test_a_bare_load_that_did_not_leave_a_bare_stock_base_goes_to_the_fresh_load(worker_side):
    worker_side.avail_gib = 60.0
    store = _store(None)
    with fallback.permission(ALLOW):
        assert _take(store, lambda: None) is None
    assert worker_side.baked == []


def test_a_bare_load_that_came_up_on_another_residency_is_not_baked_on(worker_side, monkeypatch):
    """The base loaded bare, but not as a stock resident (comfy-managed, say): the fresh load takes over."""
    from dgx_monarch.actor import store_residency

    worker_side.avail_gib = 60.0
    store = _store(None)
    monkeypatch.setattr(store_residency, "describe", lambda stored: {"residency": stored.residency})

    def load_bare():
        store.slots["cond"] = _stored(residency="comfy_managed")

    with fallback.permission(ALLOW):
        assert _take(store, load_bare) is None
    assert worker_side.baked == [] and store._stored("cond") is not None


def test_a_bake_that_fails_returns_to_the_fresh_load(worker_side):
    worker_side.fail = True
    store = _store(_stored())
    with fallback.permission(ALLOW):
        assert _take(store) is None
    assert len(worker_side.baked) == 1 and store._stored("cond") is None


def test_the_fresh_loads_probe_charges_the_classic_bake_and_never_prices_in_place(worker_side, ltx):
    held = _store(_stored())
    assert fallback.fresh_load_probe(_store(_stored(), low_rss=True), "cond", STACK) is None
    assert fallback.fresh_load_probe(held, "cond", None) is None
    refused = fallback.fresh_load_probe(held, "cond", STACK)("/models/m", {})
    assert not refused.fits and refused.in_place_bytes == 0 and refused.bake_bytes == ltx
    assert refused.in_place_note == "; " + price.GATE_HAS_NOT_PASSED
    with fallback.permission(ALLOW):
        again = fallback.fresh_load_probe(held, "cond", STACK)("/models/m", {})
    assert again.in_place_bytes == 0 and again.in_place_note == "; " + fallback.TRIED


def test_after_a_declined_hot_swap_the_walls_charge_none_of_the_stacks_files_and_only_once(worker_side, ltx):
    """The swap read the stack before it declined; the fresh load prices against a box those bytes have left."""
    term = ltx - LTX_OWN_STACK
    store = _store(_stored())
    assert not fallback.stack_on_host(store, "cond", STACK)
    assert fallback.fresh_load_probe(store, "cond", STACK)("/models/m", {}).bake_bytes == ltx  # an empty slot
    fallback.note_stack_read(store, "cond", STACK)
    assert fallback.stack_on_host(store, "cond", STACK)
    assert not fallback.stack_on_host(store, "uncond", STACK)  # another slot's stack
    assert not fallback.stack_on_host(store, "cond", [{"name": "other.safetensors", "strength": 0.8}])
    assert not fallback.stack_on_host(store, "cond", None)
    assert fallback.fresh_load_probe(store, "cond", STACK)("/models/m", {}).bake_bytes == term
    # The probe read the note once: the next fresh load of this stack is charged its file again.
    assert not fallback.stack_on_host(store, "cond", STACK)
    assert fallback.fresh_load_probe(store, "cond", STACK)("/models/m", {}).bake_bytes == ltx
    # A probe for another stack, or on the default, clears a note it does not use.
    fallback.note_stack_read(store, "cond", STACK)
    assert fallback.fresh_load_probe(store, "cond", None) is None
    assert not fallback.stack_on_host(store, "cond", STACK)


def test_a_note_too_old_to_be_this_calls_is_not_believed(worker_side, ltx, monkeypatch):
    store = _store(_stored())
    fallback.note_stack_read(store, "cond", STACK)
    assert fallback.stack_on_host(store, "cond", STACK) and fallback._STACK_READ_MAX_AGE_S >= 60
    monkeypatch.setattr(fallback, "_STACK_READ_MAX_AGE_S", -1.0)  # every note is now older than allowed
    assert not fallback.stack_on_host(store, "cond", STACK)
    assert fallback.fresh_load_probe(store, "cond", STACK)("/models/m", {}).bake_bytes == ltx


def test_the_fallback_reads_the_same_note_and_keeps_the_classic_bake_where_it_then_fits(worker_side):
    worker_side.avail_gib = 46.0  # the classic change needs 50.3 with the file still to read, 43.2 with it read
    plain = _store(_stored())
    with fallback.permission(ALLOW):
        assert _take(plain) == ("patcher", "hot-swap")  # no note: the file is charged, so the bake goes in place
    worker_side.baked.clear()
    noted = _store(_stored())
    fallback.note_stack_read(noted, "cond", STACK)
    with fallback.permission(ALLOW):
        assert _take(noted) is None  # the classic bake fits the reading the file has already left
    assert worker_side.baked == [] and fallback.stack_on_host(noted, "cond", STACK)  # kept for the fresh load
    # Where the fallback does bake, the box is no longer as the swap left it, and the note goes.
    worker_side.avail_gib = 30.0
    with fallback.permission(ALLOW):
        assert _take(noted) == ("patcher", "hot-swap")
    assert not fallback.stack_on_host(noted, "cond", STACK)


def test_the_term_and_the_fallback_belong_to_a_change_that_loads_fresh():
    """What the quote asks of its snapshot is what ``ensure``'s order does: hot swap first, then ``take``."""
    bare, stacked = _stored(), _stored(lora_sig=(("held", 1.0),))
    assert fallback.loads_fresh(None, False, False, False)  # an empty slot, or another model
    assert fallback.loads_fresh(bare, True, False, False)  # the hot swap declines: nothing backs the patches
    assert fallback.loads_fresh(stacked, True, False, False)  # a stack with a file the held one lacks
    assert not fallback.loads_fresh(stacked, True, False, True)  # a strength change: re-applied from the backup
    assert not fallback.loads_fresh(bare, True, True, False)  # beside a held slot the hot swap stays in place
    unplaced = _stored()
    unplaced.base_patcher.model.model_loaded_weight_memory = 0
    assert not fallback.loads_fresh(unplaced, True, False, False)  # ComfyUI has placed none of it
    assert fallback.loads_fresh(_stored(base_baked=True), True, True, True)  # a baked base is never hot-swapped


def test_a_change_the_hot_swap_makes_in_place_keeps_todays_price(worker_side, monkeypatch):
    from dgx_monarch.actor import store_fsdp, worker_capacity

    monkeypatch.setattr(store_fsdp, "active", lambda _worker: False)
    stored, spec = _stored(), {"unet_name": "m.safetensors", "loras": STACK}
    probe = worker_capacity._classic_bake_probe
    assert probe(object(), _store(stored), stored, spec, True, True, other_held=True) is None
    stacked = _stored(lora_sig=(("held", 1.0),))
    assert probe(object(), _store(stacked), stacked, spec, True, True, covered=True) is None
    assert probe(object(), _store(stacked), stacked, spec, True, True) is not None


@pytest.mark.parametrize("reading, bake", [(80.0, "classic"), (51.0, "classic"), (30.0, "in_place"), (9.0, None)])
def test_the_quote_the_store_and_the_walls_pick_one_bake_at_one_reading(worker_side, monkeypatch, reading, bake):
    """One held-base stack change, one reading: the quote's probe, ``take`` and the fresh load's wall agree."""
    from dgx_monarch.actor import store_fsdp, worker_capacity

    monkeypatch.setattr(store_fsdp, "active", lambda _worker: False)
    worker_side.avail_gib = reading
    stored = _stored()
    store = _store(stored)
    spec = {"unet_name": "m.safetensors", "loras": STACK}
    quoted = worker_capacity._classic_bake_probe(object(), store, stored, spec, True, True)(
        "/models/m.safetensors", {})
    with fallback.permission(ALLOW):
        took = _take(store)
    if bake == "in_place":
        assert quoted.fits and quoted.measured()["bake"] == "in_place" and took == ("patcher", "hot-swap")
        return
    assert quoted.measured()["bake"] == "classic" and took is None and worker_side.baked == []
    # The classic verdict is one verdict: the quote's, credited the drop, and the wall's after it.
    worker_side.avail_gib = reading + HELD / GIB
    with fallback.permission(ALLOW):
        walled = fallback.fresh_load_probe(store, "cond", STACK)("/models/m.safetensors", {})
    assert walled.bake_bytes == quoted.bake_bytes
    assert walled.fits is (quoted.required_bytes - HELD <= quoted.avail_bytes) is (bake == "classic")


def test_the_quote_without_the_permission_prices_the_classic_bake_alone(worker_side, monkeypatch):
    from dgx_monarch.actor import store_fsdp, worker_capacity

    monkeypatch.setattr(store_fsdp, "active", lambda _worker: False)
    stored = _stored()
    spec = {"unet_name": "m.safetensors", "loras": STACK}
    quoted = worker_capacity._classic_bake_probe(object(), _store(stored), stored, spec, True, False)(
        "/models/m.safetensors", {})
    assert not quoted.fits and quoted.in_place_bytes == 0 and "identity-gate PASS" in quoted.in_place_note
    assert worker_capacity._classic_bake_probe(
        object(), _store(stored, low_rss=True), stored, spec, True, True) is None
    assert worker_capacity._classic_bake_probe(
        object(), _store(stored), stored, {"unet_name": "m.safetensors"}, True, True) is None
    monkeypatch.setattr(store_fsdp, "active", lambda _worker: True)
    assert worker_capacity._classic_bake_probe(object(), _store(stored), stored, spec, True, True) is None


def test_the_ladders_row_gives_the_wall_back_its_bake_term(ltx):
    from dgx_monarch.actor import capacity_quote, store_residency

    refused = price.priced(_fit(30.0), "p", STACK, None, in_place=price.GATE_HAS_NOT_PASSED)
    row = types.SimpleNamespace(measured=refused.measured(), reason="")
    rebuilt = store_residency._stock_fit(row)
    assert rebuilt.bake_bytes == ltx and rebuilt.in_place_note == refused.in_place_note
    assert rebuilt.required_bytes == refused.required_bytes and rebuilt.terms == refused.terms
    assert capacity_quote.RankPrice is not None


def test_the_partial_load_route_charges_the_line_for_the_layers_still_to_bake():
    from dgx_monarch.actor import lora_partial_load

    pending = [(f"k{index}", types.SimpleNamespace()) for index in range(148)]
    # 2 GiB plus 7.5 MiB a layer, beside ComfyUI's backups: no bytes guard on this route.
    assert lora_partial_load._bake_room(pending) == (
        price.CLASSIC_BAKE_FIXED_BYTES + 148 * price.CLASSIC_BAKE_BYTES_PER_WEIGHT)
    assert lora_partial_load._bake_room([]) == 0


def test_the_fleets_sentence_names_the_bake_and_the_in_place_bake_not_taken(ltx):
    from dgx_monarch.actor.capacity_quote import RUNG_EXPLICIT_STOCK, VERDICT_REFUSE, RankPrice
    from dgx_monarch.capacity_agreement import RankAnswer, _priced_sentence

    refused = price.priced(_fit(30.0), "p", STACK, None, in_place=price.GATE_HAS_NOT_PASSED)
    quote = RankPrice(VERDICT_REFUSE, RUNG_EXPLICIT_STOCK, priced=True, measured=refused.measured(),
                      checkpoint_bytes=LTX_BYTES, slot="cond", would_load="swap", holds_resident={})
    sentence = _priced_sentence(RankAnswer(rank=0, host="box", quotes=(quote,)), quote, "m.safetensors")
    assert "GiB for the LoRA stack's classic bake" in sentence
    assert "Baking it in place was not tried: no identity-gate PASS" in sentence
