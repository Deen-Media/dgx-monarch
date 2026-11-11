"""A load that replaces a model is priced after the Worker drops it (swap_credit.py).

Two hardware records, both the loader preflight pricing a new DiT beside the
one its own slot was about to drop:

- Row 5 of the 2026-11 Studio study: flux1-dev held (22.17 GiB), then the
  shipped LTX 2.3 template. The load window needed 80.84 GiB against 67.51
  available, 13.33 short, and the user met the Old model in the way card.
- 2026-11-11, MiniMax-H3-Image: the 37.5 GiB BF16 DiT resident, then the INT8
  file. 46.0 GiB needed against 34.1 available, 11.9 short, the same card.

The cross-rank agreement already credited such a swap in the Worker's own
price. These tests hold the same credit at the loader, taken from that price
and from nothing else: the rule (``swap_credit.from_answers``, over rows the
Worker's real ``credit_stock_swap`` wrote), both records before and after
(``loader_relief.relieve`` over the loader's real windows), the loader itself,
and the walls that must stay: a true shortage, a resident whose drop the
Worker's price does not credit, another slot's model, a load whose slot is not
known, and a price that charges no load.

CPU only: no torch, no CUDA, no ComfyUI beyond a stub folder_paths.
"""
from __future__ import annotations

import ast
import types
from dataclasses import replace
from pathlib import Path

import pytest

import test_capacity_old_model as rig
from dgx_monarch import (
    capacity_agreement,
    capacity_lora_bake,
    consent_pending,
    driver_footprint,
    mesh_safety,
    old_model,
    swap_credit,
)
from dgx_monarch.capacity_fit import StockFit
from dgx_monarch.nodes import loader_preflight, loader_relief
from test_capacity_old_model import NEW, _answer, _LiveHandle, _loader_rig, _stock

# The old-model rig: a stub folder_paths, a clean card list and memo, no settle wait.
_rig = rig._rig

REPO = Path(__file__).resolve().parents[1]
_GIB = 2 ** 30
FLUX1, LTX23 = "flux1-dev.safetensors", "ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors"
H3I_BF16, H3I_INT8 = "minimax_h3_image_pruned_bf16.safetensors", "minimax_h3_image_pruned_int8_convrot.safetensors"
LONGCAT = "longcat_image_bf16.safetensors"


def _held(name: str, gib: float, *, word: str | None = "stock", slot: str = "cond",
          would_load: str = "swap", arena_gib: float = 0.0, **changes):
    """The head Worker's row for a load into ``slot`` whose resident is ``name``.

    Its measured block is the one the Worker's own stock price writes: the real
    ``credit_stock_swap`` over the slot's snapshot entry, so a row carries a
    credit exactly where the agreement gives one.
    """
    snapshot = {"residency": word, "resident_bytes": int(gib * _GIB), "resident_name": name}
    fit = capacity_lora_bake.credit_stock_swap(
        StockFit(False, True, 1 * _GIB, 200 * _GIB), snapshot, {"would_load": would_load})
    row = _stock("fits", size_gib=1.0, avail_gib=200.0, resident=name, word=word,
                 resident_gib=gib, slot=slot, arena_gib=arena_gib)
    return replace(row, measured=dict(fit.measured()), would_load=would_load, **changes)


def _credit(*rows, slot: str = "cond") -> swap_credit.SwapCredit:
    return swap_credit.from_answers([_answer(index, row) for index, row in enumerate(rows)], slot)


def _answers(monkeypatch, *rows) -> list:
    """Make the driver host's Workers answer with ``rows``; returns the questions asked."""
    asked: list = []
    monkeypatch.setattr(capacity_agreement, "collect", lambda handle, request, world, *, deadline_s:
                        asked.append(request) or [_answer(index, row, box="this box")
                                                  for index, row in enumerate(rows)])
    return asked


# The rule.

def test_a_stock_resident_the_load_replaces_is_credited_what_the_workers_price_credited():
    credit = _credit(_held(FLUX1, 22.17))
    assert credit and credit.bytes == int(22.17 * _GIB) and credit.dropped == FLUX1


@pytest.mark.parametrize("word", ["slab", "comfy_managed", "fsdp", "in_flight", None])
def test_a_resident_the_workers_price_does_not_credit_gives_nothing(word):
    assert not _credit(_held(FLUX1, 22.17, word=word))


@pytest.mark.parametrize("would_load", ["reuse", "fresh", "unknown"])
def test_a_reuse_and_a_fresh_load_give_nothing(would_load):
    assert not _credit(_held(FLUX1, 22.17, would_load=would_load))


def test_only_the_slot_the_load_goes_into_is_credited():
    """A cond resident stays when the uncond loader loads: no credit for it."""
    cond_holds = _held(FLUX1, 22.17)
    assert not _credit(replace(cond_holds, slot="uncond", would_load="fresh"), slot="uncond")
    assert not _credit(cond_holds, slot="uncond"), "the Worker's row is for another slot"
    assert not _credit(cond_holds, slot=""), "a load whose slot is not known gets no credit"
    # The row's own check, not only the caller's filter over the rows.
    assert swap_credit._row_credit(cond_holds, "uncond") == (0, "")
    assert swap_credit._row_credit(cond_holds, "cond") == (int(22.17 * _GIB), FLUX1)


def test_nothing_but_the_workers_own_credit_counts():
    """The snapshot says the slot holds 26.6 GiB; only a price that named the drop is a credit."""
    row = _held(FLUX1, 26.6)
    bare = {key: value for key, value in row.measured.items() if key not in ("credit_bytes", "dropped")}
    assert not _credit(replace(row, measured=bare)), "resident_bytes alone is no credit"
    # A LoRA change that keeps its base carries a credit and names no drop.
    assert not _credit(replace(row, measured={**bare, "credit_bytes": 20 * _GIB}))
    # A row priced on a slab rung: the agreement prices that load with the old model held.
    assert not _credit(replace(row, use_slab=True))
    assert not _credit(replace(row, measured=None))


def test_a_cast_resident_is_credited_the_lower_bound_the_worker_reports():
    """The snapshot's figure for a cast copy is its lower bound, and that is what the price credits."""
    assert _credit(_held(FLUX1, 20.0)).bytes == 20 * _GIB


def test_several_ranks_on_the_box_give_one_workers_worth_and_all_must_report():
    assert _credit(_held(FLUX1, 22.0), _held(FLUX1, 12.0)).bytes == 12 * _GIB
    assert not _credit(_held(FLUX1, 22.0), _held(FLUX1, 12.0, word="slab"))
    assert not swap_credit.from_answers([], "cond")


def test_the_render_site_credit_is_zero_outside_its_block():
    assert swap_credit.credited(10 * _GIB) == 10 * _GIB and swap_credit.credited(None) is None
    with swap_credit.applying(_credit(_held(FLUX1, 22.0))):
        assert swap_credit.credited(10 * _GIB) == 32 * _GIB and swap_credit.credited(None) is None
    assert swap_credit.credited(10 * _GIB) == 10 * _GIB


# Both records, before and after, over the loader's own windows.

def _shape(family: str, *, avail: float, file: float, stack: float = 0.0,
           load_stack: float = 0.0, resident: bool = False) -> loader_preflight._Shape:
    return loader_preflight._Shape(
        profile=driver_footprint.DRIVER_STACK_FAMILIES[family], mem_available=int(avail * _GIB),
        weight_bytes=int(file * _GIB), weights_resident=resident, co_resident=True,
        floor_reserve=5 * _GIB, stack_bytes=int(stack * _GIB), load_stack_bytes=int(load_stack * _GIB))


def _stock_windows(priced):
    return loader_preflight._windows(priced, slab=False)


def _relieve(monkeypatch, shape, held_row, *, new: str, slot: str = "cond"):
    _answers(monkeypatch, held_row)
    mesh = types.SimpleNamespace(handle=_LiveHandle(), topology_preset="auto", worker_args={})
    before = max(window.shortfall for window in _stock_windows(shape))
    relief = loader_relief.relieve(shape, _stock_windows, mesh=mesh, unet_name=new,
                                   options={}, slot=slot)
    return before, relief


def test_row_5_flux1_dev_then_ltx_23_fits_once_the_drop_is_counted(monkeypatch):
    """80.84 GiB at the load window against 67.51: 13.33 short before, 8.84 to spare after."""
    # LTX 2.3 fp8: 21.86 GiB; its driver models 29.94 GiB; a deferred (auto) load.
    shape = _shape("ltx", avail=67.51, file=21.86, stack=29.94, load_stack=29.94)
    before, relief = _relieve(monkeypatch, shape, _held(FLUX1, 22.17), new=LTX23)
    assert before / _GIB == pytest.approx(13.33, abs=0.03)
    assert relief.fits and relief.credit.bytes == int(22.17 * _GIB) and relief.credit.dropped == FLUX1
    assert not relief.short_bytes and not relief.beside_short_bytes


def test_gh1_h3_image_int8_after_bf16_fits_once_the_drop_is_counted(monkeypatch):
    """46.0 GiB at the load window against 34.1: 11.9 short before, 25.6 to spare after."""
    # The INT8 file 19.5 GiB, an eager (uly2) load; text encoder, VAE and activations 16.0 GiB.
    shape = _shape("minimax_h3_image", avail=34.1, file=19.5, stack=16.0)
    before, relief = _relieve(monkeypatch, shape, _held(H3I_BF16, 37.5), new=H3I_INT8)
    assert before / _GIB == pytest.approx(11.85, abs=0.03)
    assert relief.fits and relief.credit.gib == "37.5"


def test_a_true_shortage_still_does_not_fit(monkeypatch):
    """The control row: flux1-dev then p-h3-image BF16, 119.79 GiB needed, 89.68 with the credit."""
    shape = _shape("minimax_h3", avail=67.51, file=37.69, stack=35.64, load_stack=35.64)
    before, relief = _relieve(monkeypatch, shape, _held(FLUX1, 22.17), new="p-h3-image_bf16.safetensors")
    assert before / _GIB == pytest.approx(52.28, abs=0.05)
    assert not relief.fits and relief.credit
    assert relief.short_bytes / _GIB == pytest.approx(30.11, abs=0.05)
    text = relief.sentence("p-h3-image_bf16.safetensors")
    assert text == (" The Worker on this box drops flux1-dev before this load, which frees at least "
                    "22.2 GiB, and p-h3-image_bf16 is still 30.1 GiB short after that.")


def test_a_deferred_load_needs_room_for_its_driver_models_beside_the_old_model(monkeypatch):
    """Under auto the driver builds the graph's models before the first render drops the old DiT."""
    shape = _shape("ltx", avail=30.0, file=21.86, stack=29.94, load_stack=29.94)
    _before, relief = _relieve(monkeypatch, shape, _held(FLUX1, 60.0), new=LTX23)
    assert not relief.short_bytes, "after the drop the DiT itself would fit"
    assert not relief.fits and relief.beside_short_bytes / _GIB == pytest.approx(4.94, abs=0.02)
    assert "this graph's driver models load first, beside flux1-dev" in relief.sentence(LTX23)
    # The same shortage on an eager load has no driver models to build before the drop.
    eager = _shape("ltx", avail=30.0, file=21.86, stack=29.94)
    assert _relieve(monkeypatch, eager, _held(FLUX1, 60.0), new=LTX23)[1].fits


def test_a_price_that_charges_no_load_takes_no_credit(monkeypatch):
    """The loader's memo reads the DiT as already resident: nothing is placed, so nothing is
    dropped for it. Only the driver models are short, and no drop may stand in for them."""
    shape = _shape("ltx", avail=20.0, file=21.86, stack=29.94, resident=True)
    asked = _answers(monkeypatch, _held(FLUX1, 60.0))
    mesh = types.SimpleNamespace(handle=_LiveHandle(), topology_preset="auto", worker_args={})
    relief = loader_relief.relieve(shape, _stock_windows, mesh=mesh, unet_name=LTX23,
                                   options={}, slot="cond")
    assert not relief.fits and not relief.credit and asked == []
    assert relief.asked is loader_relief.UNASKED, "the card's own question is still to be asked"


def test_no_slot_no_fleet_and_a_fault_mean_no_relief(monkeypatch):
    shape = _shape("ltx", avail=67.51, file=21.86)
    unasked = _relieve(monkeypatch, shape, _held(FLUX1, 22.17), new=LTX23, slot="")[1]
    assert not unasked.fits and not unasked.credit and unasked.asked is loader_relief.UNASKED

    def broken(*_a, **_k):
        raise RuntimeError("quote endpoint gone")

    monkeypatch.setattr(capacity_agreement, "collect", broken)
    mesh = types.SimpleNamespace(handle=_LiveHandle(), topology_preset="auto", worker_args={})
    relief = loader_relief.relieve(shape, _stock_windows, mesh=mesh, unet_name=LTX23,
                                   options={}, slot="cond")
    assert not relief.fits and not relief.credit and relief.asked is None


# The loader itself. The rig: a 20 GiB file, 47 GiB at the load window, 40 GiB read.

def _cards() -> list[str]:
    return [card["kind"] for card in consent_pending.pending_cards()]


def test_the_loader_lets_a_swap_through_with_no_card(monkeypatch, tmp_path, caplog):
    """ltx-i2v-av on 2026-11-04, longcat still held: 47 GiB needed, 40 available, 12 about to be dropped."""
    mesh = _loader_rig(monkeypatch, tmp_path, held_gib=12.0)
    asked = _answers(monkeypatch, _held(LONGCAT, 12.0))
    with caplog.at_level("INFO", logger="dgx_monarch.nodes.loader_relief"):
        assert loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond") is None
    assert _cards() == [] and len(asked) == 1
    assert asked[0]["specs"] == [{"unet_name": NEW, "options": {}, "loras": [], "slot": "cond"}]
    line = next(record.getMessage() for record in caplog.records if record.name.endswith("loader_relief"))
    assert ("fits on this box once the Worker there drops longcat_image_bf16, which frees at "
            "least 12.0 GiB; MemAvailable read 40.0 GiB") in line


def test_the_loader_without_a_slot_prices_as_before(monkeypatch, tmp_path):
    mesh = _loader_rig(monkeypatch, tmp_path, held_gib=12.0)
    _answers(monkeypatch, _held(LONGCAT, 12.0))
    with pytest.raises(old_model.LoaderOldModelError):
        loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None)
    assert _cards() == ["clear-retry"]


def test_a_true_shortage_still_refuses_and_says_what_was_counted(monkeypatch, tmp_path):
    """4 GiB held: 47 needed against 44 with the credit. No clear helps either: no arena to return."""
    mesh = _loader_rig(monkeypatch, tmp_path, held_gib=4.0)
    _answers(monkeypatch, _held(LONGCAT, 4.0))
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond")
    text = str(raised.value)
    assert "needs 47.0 GiB on this box and 40.0 GiB is available, 7.0 GiB short." in text
    assert ("The Worker on this box drops longcat_image_bf16 before this load, which frees at "
            "least 4.0 GiB, and qwen_image_2512_bf16 is still 3.0 GiB short after that.") in text
    assert "A clear would not make room" in text and "a cleared box holds no copy" in text
    assert "is still loaded on this box" not in text, "the price already dropped it"
    assert "clear-retry" not in _cards()


def test_a_clear_is_weighed_against_the_uncredited_price(monkeypatch, tmp_path):
    """What a clear returns beyond the drop is the arena: with enough of it the card still helps."""
    mesh = _loader_rig(monkeypatch, tmp_path, held_gib=4.0)
    _answers(monkeypatch, _held(LONGCAT, 4.0, arena_gib=6.0))
    with pytest.raises(old_model.LoaderOldModelError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond")
    offer = raised.value.offer
    assert offer.boxes[0].dropped == ("longcat_image_bf16",)
    assert offer.boxes[0].left_bytes == 6 * _GIB and offer.boxes[0].gain_bytes == 10 * _GIB
    assert offer.boxes[0].required_bytes == 47 * _GIB, "the uncredited price: a cleared box has nothing to drop"
    assert "still holds about 6.0 GiB that earlier loads left behind" in str(raised.value)
    assert _cards() == ["clear-retry"]


@pytest.mark.parametrize("word", ["slab", "fsdp"])
def test_a_slab_or_shard_resident_gets_no_credit_at_the_loader(monkeypatch, tmp_path, word):
    mesh = _loader_rig(monkeypatch, tmp_path, held_gib=12.0)
    _answers(monkeypatch, _held(LONGCAT, 12.0, word=word))
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond")
    assert "drops longcat_image_bf16 before this load" not in str(raised.value)


def test_the_uncond_loader_is_not_credited_the_cond_slots_model(monkeypatch, tmp_path):
    mesh = _loader_rig(monkeypatch, tmp_path, held_gib=12.0)
    asked = _answers(monkeypatch, _held(LONGCAT, 12.0))
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="uncond")
    assert "before this load" not in str(raised.value)
    assert asked[0]["specs"][0]["slot"] == "uncond", "the Workers are asked about the loader's own slot"


def test_the_dropped_models_memo_is_forgotten_so_it_is_priced_when_it_comes_back(monkeypatch, tmp_path):
    """A, then B on the credit, then A again: A is no longer 'already loaded'."""
    first, second = LONGCAT, NEW
    (tmp_path / "diffusion_models" / first).write_bytes(b"x")
    mesh = _loader_rig(monkeypatch, tmp_path, held_gib=20.0)
    nothing = _stock("fits", size_gib=1.0, avail_gib=200.0, resident=None)
    _answers(monkeypatch, nothing)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 60 * _GIB)
    assert loader_preflight.preflight_loader_footprint(mesh, first, {}, None, slot="cond") is None
    # B replaces A in the slot: 47 GiB needed, 40 available, A's 20 GiB about to be dropped.
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 40 * _GIB)
    _answers(monkeypatch, _held(first, 20.0))
    assert loader_preflight.preflight_loader_footprint(mesh, second, {}, None, slot="cond") is None
    # A again on a box with no room for it and nothing to drop: it must be priced, and refuse.
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 10 * _GIB)
    _answers(monkeypatch, nothing)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError):
        loader_preflight.preflight_loader_footprint(mesh, first, {}, None, slot="cond")
    # B, which the Worker now holds, is still read as loaded.
    assert loader_preflight.preflight_loader_footprint(mesh, second, {}, None, slot="cond") is None


def test_a_model_the_memo_reads_as_loaded_takes_no_credit_at_the_loader(monkeypatch, tmp_path):
    """The memo is written when a load is admitted, before a deferred load happens. A second
    price then charges no DiT. A box 2 GiB under its reserve must refuse as it did, whatever
    a Worker says it would drop."""
    mesh = _loader_rig(monkeypatch, tmp_path, held_gib=20.0)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 60 * _GIB)
    assert loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond") is None
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 3 * _GIB)
    _answers(monkeypatch, _held(LONGCAT, 20.0))
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond")
    assert "before this load" not in str(raised.value)


# A dual-model graph: each price charges one DiT alone, so a credit for one slot
# leaves the room the other slot's load takes (swap_credit.growth and less).

UNCOND = "ideogram4_uncond_bf16.safetensors"
_GRAPH = {"7": {"class_type": "DGXMonarchUncondUNETLoader", "inputs": {"unet_name": UNCOND}}}


def _other(new_gib: float, *, held_gib: float | None = None, would_load: str = "swap",
           word: str | None = "stock", slot: str = "uncond") -> object:
    """The Worker's row for the other slot's load: ``new_gib`` over a resident of ``held_gib``."""
    row = (_held("old_uncond.safetensors", held_gib, slot=slot, would_load=would_load, word=word)
           if held_gib else replace(_stock("fits", size_gib=1.0, avail_gib=200.0, resident=None, slot=slot),
                                    would_load=would_load))
    return replace(row, checkpoint_bytes=int(new_gib * _GIB))


def _answers_by_slot(monkeypatch, **rows) -> list:
    """One Worker on the driver host that answers each spec asked with its slot's row."""
    asked: list = []

    def collect(handle, request, world, *, deadline_s):
        asked.append(request)
        return [_answer(0, *(rows[spec["slot"]] for spec in request["specs"] if spec["slot"] in rows),
                        box="this box")]

    monkeypatch.setattr(capacity_agreement, "collect", collect)
    return asked


@pytest.mark.parametrize(("row", "grows_gib", "why"), [
    (_other(30.0, held_gib=22.0, would_load="reuse"), 0.0, "a reuse loads nothing"),
    (_other(30.0, held_gib=22.0), 8.0, "30 GiB over a 22 GiB stock resident its Worker drops"),
    (_other(10.0, held_gib=22.0), 0.0, "a smaller model adds nothing, and its gain is not counted"),
    (_other(30.0, would_load="fresh"), 30.0, "an empty slot: the whole file"),
    (_other(30.0, held_gib=22.0, word="slab"), 30.0, "a resident its Worker does not credit stays held"),
    (_other(30.0, would_load="unknown"), None, "a load the Worker cannot name"),
    (_other(0.0, held_gib=22.0), None, "a load with no size"),
])
def test_what_the_other_slots_load_adds_to_the_host(row, grows_gib, why):
    grows = swap_credit.growth([_answer(0, _held(FLUX1, 22.17), row)], "uncond")
    assert grows == (None if grows_gib is None else int(grows_gib * _GIB)), why


def test_a_cast_load_into_the_other_slot_is_sized_at_the_casts_copy():
    """A 20 GiB fp8 file cast to a 40 GiB copy over a 22 GiB resident grows the slot by 18, not by nothing."""
    row = _other(20.0, held_gib=22.0)
    cast = replace(row, measured={**row.measured, "cast_bytes": 40 * _GIB})
    assert swap_credit.growth([_answer(0, row)], "uncond") == 0
    assert swap_credit.growth([_answer(0, cast)], "uncond") == 18 * _GIB


def test_the_other_slots_growth_is_the_largest_a_rank_reports_and_none_with_no_answer():
    ranks = [_answer(0, _other(30.0, held_gib=22.0)), _answer(1, _other(30.0, held_gib=12.0))]
    assert swap_credit.growth(ranks, "uncond") == 18 * _GIB
    assert swap_credit.growth([], "uncond") is None
    # A rank that holds one slot only sends no row for the other: nothing lands on it there.
    assert swap_credit.growth([_answer(0, _held(FLUX1, 22.17))], "uncond") == 0


def test_a_credit_less_the_other_slots_growth():
    credit = swap_credit.SwapCredit(22 * _GIB, FLUX1, "this box")
    left = swap_credit.less(credit, 8 * _GIB)
    assert (left.bytes, left.set_aside, left.dropped, left.box) == (14 * _GIB, 8 * _GIB, FLUX1, "this box")
    assert left.gib == "22.0", "the sentence gives what the drop frees"
    assert left.note == ", of which 8.0 GiB is set aside for the model this graph loads into the other slot"
    assert swap_credit.less(credit, 0) is credit and credit.note == ""
    assert swap_credit.less(credit, -5 * _GIB) is credit, "a gain in the other slot is not counted"
    for grows in (22 * _GIB, 40 * _GIB, None):
        assert swap_credit.less(credit, grows) is swap_credit.NONE
    assert swap_credit.less(swap_credit.NONE, 0) is swap_credit.NONE


def test_the_graphs_other_loader_is_read_as_a_quote_spec():
    cond = {"3": {"class_type": "DGXMonarchUNETLoader", "inputs": {"unet_name": NEW, "weight_dtype": "fp8_e4m3fn"}}}
    graph = {**cond, **_GRAPH, "9": {"class_type": "KSampler", "inputs": {"unet_name": "not a loader"}}}
    assert loader_relief.other_slot_loads(graph, "cond") == [
        {"unet_name": UNCOND, "options": {}, "loras": [], "slot": "uncond"}]
    assert loader_relief.other_slot_loads(graph, "uncond") == [
        {"unet_name": NEW, "options": {"weight_dtype": "fp8_e4m3fn"}, "loras": [], "slot": "cond"}]
    assert loader_relief.other_slot_loads(cond, "cond") == [] and loader_relief.other_slot_loads(None, "cond") == []
    linked = {"7": {"class_type": "DGXMonarchUncondUNETLoader", "inputs": {"unet_name": ["4", 0]}}}
    assert loader_relief.other_slot_loads(linked, "cond") is None, "a name on a link cannot be asked"


def _dual(monkeypatch, shape, uncond_row, *, graph=_GRAPH):
    asked = _answers_by_slot(monkeypatch, cond=_held(FLUX1, 22.17), uncond=uncond_row)
    mesh = types.SimpleNamespace(handle=_LiveHandle(), topology_preset="auto", worker_args={})
    relief = loader_relief.relieve(shape, _stock_windows, mesh=mesh, unet_name=LTX23,
                                   options={}, slot="cond", prompt=graph)
    return relief, asked


def test_a_dual_swap_is_credited_less_what_the_other_slot_grows_by(monkeypatch):
    """Row 5's figures with a second DiT in the graph: 8.84 GiB to spare on the drop alone."""
    shape = _shape("ltx", avail=67.51, file=21.86, stack=29.94, load_stack=29.94)
    # The other slot swaps 22 GiB for 30: 8 GiB is set aside, and 0.84 is left to spare.
    relief, asked = _dual(monkeypatch, shape, _other(30.0, held_gib=22.0))
    assert relief.fits and relief.credit.bytes == int(22.17 * _GIB) - 8 * _GIB
    assert relief.credit.set_aside == 8 * _GIB
    assert [len(request["specs"]) for request in asked] == [1, 2], "one question more, with both loads"
    assert asked[1]["specs"] == [{"unet_name": LTX23, "options": {}, "loras": [], "slot": "cond"},
                                 {"unet_name": UNCOND, "options": {}, "loras": [], "slot": "uncond"}]
    assert asked[0]["specs"] == asked[1]["specs"][:1], "the card's own question is unchanged"
    # It swaps 22 GiB for 32: the load is 1.16 GiB short, and the refusal says where the room went.
    relief, _asked = _dual(monkeypatch, shape, _other(32.0, held_gib=22.0))
    assert not relief.fits and relief.short_bytes / _GIB == pytest.approx(1.16, abs=0.02)
    assert relief.sentence(LTX23).startswith(
        " The Worker on this box drops flux1-dev before this load, which frees at least 22.2 GiB, "
        "of which 10.0 GiB is set aside for the model this graph loads into the other slot, and ")


@pytest.mark.parametrize(("uncond_row", "credited", "why"), [
    (_other(30.0, held_gib=22.0, would_load="reuse"), True, "the other slot keeps its model"),
    (_other(10.0, held_gib=22.0), True, "the other slot shrinks"),
    (_other(30.0, would_load="fresh"), False, "a 30 GiB load into the empty slot takes the 22 GiB drop"),
    (_other(30.0, would_load="unknown"), False, "a load the Worker cannot name"),
])
def test_a_dual_swap_is_credited_only_what_the_other_slot_leaves(monkeypatch, uncond_row, credited, why):
    shape = _shape("ltx", avail=67.51, file=21.86, stack=29.94, load_stack=29.94)
    relief, _asked = _dual(monkeypatch, shape, uncond_row)
    assert bool(relief.credit) is credited and relief.fits is credited, why
    if credited:
        assert relief.credit.bytes == int(22.17 * _GIB) and not relief.credit.set_aside


def test_a_dual_swap_whose_other_load_cannot_be_read_or_priced_gets_no_credit(monkeypatch):
    shape = _shape("ltx", avail=67.51, file=21.86, stack=29.94, load_stack=29.94)
    linked = {"7": {"class_type": "DGXMonarchUncondUNETLoader", "inputs": {"unet_name": ["4", 0]}}}
    relief, asked = _dual(monkeypatch, shape, _other(10.0, held_gib=22.0), graph=linked)
    assert not relief.credit and not relief.fits and len(asked) == 1
    # The second question fails: no credit, and the card still has its own first answer.
    first = _answers_by_slot(monkeypatch, cond=_held(FLUX1, 22.17))

    def once(handle, request, world, *, deadline_s, real=capacity_agreement.collect):
        if len(request["specs"]) > 1:
            raise RuntimeError("quote endpoint gone")
        return real(handle, request, world, deadline_s=deadline_s)

    monkeypatch.setattr(capacity_agreement, "collect", once)
    mesh = types.SimpleNamespace(handle=_LiveHandle(), topology_preset="auto", worker_args={})
    relief = loader_relief.relieve(shape, _stock_windows, mesh=mesh, unet_name=LTX23,
                                   options={}, slot="cond", prompt=_GRAPH)
    assert not relief.credit and len(first) == 1
    assert relief.asked is not None and relief.asked is not loader_relief.UNASKED


def test_a_graph_with_one_dit_asks_once(monkeypatch):
    shape = _shape("ltx", avail=67.51, file=21.86, stack=29.94, load_stack=29.94)
    relief, asked = _dual(monkeypatch, shape, _other(30.0, held_gib=22.0), graph={
        "3": {"class_type": "DGXMonarchUNETLoader", "inputs": {"unet_name": LTX23}}})
    assert relief.fits and relief.credit.bytes == int(22.17 * _GIB) and len(asked) == 1


def test_the_loader_sets_the_other_slots_growth_aside_end_to_end(monkeypatch, tmp_path, caplog):
    """The rig's 47 GiB against 40 read and 12 to drop: 5 GiB to spare, so a second DiT that
    grows its slot by 4 GiB still fits and one that grows it by 6 does not."""
    mesh = _loader_rig(monkeypatch, tmp_path, held_gib=12.0)
    _answers_by_slot(monkeypatch, cond=_held(LONGCAT, 12.0), uncond=_other(26.0, held_gib=22.0))
    with caplog.at_level("INFO", logger="dgx_monarch.nodes.loader_relief"):
        assert loader_preflight.preflight_loader_footprint(mesh, NEW, {}, _GRAPH, slot="cond") is None
    assert any("frees at least 12.0 GiB, of which 4.0 GiB is set aside for the model this graph "
               "loads into the other slot; MemAvailable read 40.0 GiB" in record.getMessage()
               for record in caplog.records)
    loader_preflight.reset_memos()
    # The Worker's cond row as it really reads: it holds the uncond slot's 22 GiB model too.
    cond = replace(_held(LONGCAT, 12.0), holds_resident={"cond": "stock", "uncond": "stock"},
                   resident_names={"cond": LONGCAT, "uncond": "old_uncond.safetensors"},
                   resident_bytes={"cond": 12 * _GIB, "uncond": 22 * _GIB})
    _answers_by_slot(monkeypatch, cond=cond, uncond=_other(28.0, held_gib=22.0))
    with pytest.raises(old_model.LoaderOldModelError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, NEW, {}, _GRAPH, slot="cond")
    assert "of which 6.0 GiB is set aside" in str(raised.value) and "1.0 GiB short after that" in str(raised.value)
    # The card: a clear returns both models, 34 GiB, and the drop frees 12 of that whatever
    # part of it was set aside. So 22 GiB is what a clear adds, and it is the uncond model.
    box = raised.value.offer.boxes[0]
    assert (box.gain_bytes, box.left_bytes) == (34 * _GIB, 22 * _GIB)
    assert box.dropped == ("longcat_image_bf16",) and "old_uncond" in str(raised.value)


# The credit's own switch, and the unload's.

def test_the_credits_own_switch_gives_the_loader_its_price_as_it_was(monkeypatch, tmp_path):
    """``DGXM_DISABLE_SWAP_CREDIT=1``: the card, as before the credit, and no word of a drop."""
    mesh = _loader_rig(monkeypatch, tmp_path, held_gib=12.0)
    _answers(monkeypatch, _held(LONGCAT, 12.0))
    monkeypatch.setenv(swap_credit.DISABLE_ENV, "1")
    with pytest.raises(old_model.LoaderOldModelError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond")
    assert _cards() == ["clear-retry"] and "before this load" not in str(raised.value)


def test_the_unload_switch_leaves_the_credit_on_at_the_loader(monkeypatch, tmp_path):
    """The credit is a price, not a rescue: the loader and the sampler must agree under that switch."""
    from dgx_monarch import driver_unload

    mesh = _loader_rig(monkeypatch, tmp_path, held_gib=12.0)
    _answers(monkeypatch, _held(LONGCAT, 12.0))
    monkeypatch.setenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, "1")
    assert loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond") is None
    assert _cards() == []


def test_a_load_that_fits_asks_no_worker(monkeypatch, tmp_path):
    mesh = _loader_rig(monkeypatch, tmp_path, held_gib=12.0)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 60 * _GIB)

    def asked(*_a, **_k):
        raise AssertionError("a price that fits must not ask the Workers")

    monkeypatch.setattr(capacity_agreement, "collect", asked)
    assert loader_preflight.preflight_loader_footprint(mesh, NEW, {}, None, slot="cond") is None


def test_both_loader_nodes_and_the_requeue_price_name_their_slot():
    def slot_keywords(path: str) -> list[str]:
        tree = ast.parse((REPO / "src" / "dgx_monarch" / "nodes" / path).read_text())
        return [ast.unparse(keyword.value) for node in ast.walk(tree)
                if isinstance(node, ast.Call) and ast.unparse(node.func).endswith("preflight_loader_footprint")
                for keyword in node.keywords if keyword.arg == "slot"]

    assert slot_keywords("loaders.py") == ["self.SLOT"]
    assert slot_keywords("requeue_price.py") == ["str(getattr(spec, 'slot', '') or '')"]
