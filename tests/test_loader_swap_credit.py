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
