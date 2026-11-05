"""A loader dtype cast is priced like any other stock load.

On 2026-11-05 (dev 91fd15f) ``dgx-monarch-wan22-t2v`` loaded its second DiT,
``wan2.2_t2v_low_noise_14B_fp16.safetensors`` cast to bf16, on a head held at
12.2 GiB available. The probe skipped the cast, the Worker logged
``rung=stock_skipped unpriced``, the fleet agreement admitted it as
``no_claim/stock_fits@headroom unstated``, and only the vetting watchdog
stopped the box at 7.5 GiB. These tests hold the price that replaces the skip:
the copy the cast builds, read from the header, plus the host copy the load is
placed from and the floor, at the ladder, at the last wall and at the fleet
agreement.

CPU only. Checkpoints are sparse synthetic files: a header the real parser
accepts, truncated to the declared size, with no tensor bytes written.
"""
from __future__ import annotations

import json
import os
import struct

import pytest
import torch

from dgx_monarch import box_names, capacity_agreement, capacity_fit, mesh_safety
from dgx_monarch.actor import capacity_quote, store_residency
from dgx_monarch.actor.capacity_quote import RankPrice
from dgx_monarch.capacity_agreement import FleetCapacityError, RankAnswer, decide
from dgx_monarch.capacity_fit import StockFit
from dgx_monarch.consent_descriptor import SlabResidencyRescueOffer
from dgx_monarch.gate_audit_evidence import parse_measured
from dgx_monarch.refusal import parse_leading_refusal_tag
from dgx_monarch.safetensors_header import DTYPE_BITS

_GIB = 2 ** 30
# The wan2.2 14B fp16 DiT's element count, read off the real header on the head
# (1095 F16 tensors, 26.6 GiB of data). One tensor carries it here.
WAN22_ELEMENTS = 14_288_491_584
WAN22_NAME = "wan2.2_t2v_low_noise_14B_fp16.safetensors"
# Where B4 held the head while stage 2's load was admitted.
B4_AVAILABLE = int(12.2 * _GIB)
CAST_BLOCKER = "the loader casts this checkpoint's weight dtype"


def _write(path, tensors: dict) -> str:
    """A header-valid safetensors file with no tensor bytes written."""
    header: dict = {}
    cursor = 0
    for key, (dtype, shape) in tensors.items():
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * DTYPE_BITS[dtype] // 8
        header[key] = {"dtype": dtype, "shape": list(shape),
                       "data_offsets": [cursor, cursor + nbytes]}
        cursor += nbytes
    body = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
    os.truncate(path, 8 + len(body) + cursor)
    return str(path)


def _wan22(tmp_path, dtype: str = "F16") -> str:
    return _write(tmp_path / WAN22_NAME,
                  {"blocks.0.ffn.0.weight": (dtype, [WAN22_ELEMENTS])})


@pytest.fixture
def unified(monkeypatch):
    """An integrated GPU whose MemAvailable the test sets."""
    available = {"bytes": 100 * _GIB}
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: available["bytes"])
    return available


def _ladder(path: str, model_options: dict, **overrides):
    kwargs = {
        "path": path, "unet_name": os.path.basename(path), "model_options": model_options,
        "slab_weights": "auto", "slab_capable_path": True, "lora_low_rss": True,
        "fsdp_launch": False, "blocked_reason": "", "authoritative_slab_retry": False,
        "memoized_family": lambda _path: None, "vouched_families": frozenset({"krea2"}),
        "rescue_consent": None, "file_identity": lambda _path: "1:2:3:4:5",
        "compile_dit": lambda: False, "comfy_managed": lambda: False,
    }
    kwargs.update(overrides)
    return kwargs


# The estimate, from the header.

def test_a_cast_that_keeps_the_width_prices_what_an_uncast_load_prices(tmp_path):
    """Less the header, which the file carries and no device copy holds."""
    path = _wan22(tmp_path)
    size = os.path.getsize(path)
    cast = capacity_fit.cast_resident_bytes(path, torch.bfloat16)
    assert cast == WAN22_ELEMENTS * 2 and 0 < size - cast < 4096
    assert capacity_fit.stock_required_bytes(size, cast) == (
        capacity_fit.stock_required_bytes(size) - (size - cast))
    assert capacity_fit.stock_required_bytes(size, size) == capacity_fit.stock_required_bytes(size)
    # 26.6 GiB cast, 1.1 times the file for the host copy, the 5 GiB floor.
    assert capacity_fit.gib(capacity_fit.stock_required_bytes(size, cast)) == 60.9


def test_a_widening_cast_charges_more_than_the_file_and_a_narrowing_one_less(tmp_path):
    elements = 1 << 30
    fp8 = _write(tmp_path / "plain_fp8.safetensors", {"w": ("F8_E4M3", [elements])})
    bf16 = _write(tmp_path / "plain_bf16.safetensors", {"w": ("BF16", [elements])})
    assert capacity_fit.cast_resident_bytes(fp8, torch.bfloat16) == 2 * elements
    assert capacity_fit.cast_resident_bytes(bf16, torch.float8_e4m3fn) == elements
    size = os.path.getsize(fp8)
    widened = capacity_fit.stock_required_bytes(size, 2 * elements)
    assert widened == capacity_fit.stock_required_bytes(size) + (2 * elements - size)


def test_quantized_layers_integers_and_fp32_islands_keep_their_stored_size(tmp_path):
    """A quantized layer is named by its scale or its comfy_quant marker; ComfyUI
    keeps that payload under a cast. F32 is never charged below its stored size."""
    path = _write(tmp_path / "mixed.safetensors", {
        "blocks.0.q.weight": ("F8_E4M3", [1024, 1024]),
        "blocks.0.q.weight_scale": ("F32", [1]),
        "blocks.0.k.weight": ("F8_E4M3", [1024, 1024]),
        "blocks.0.k.comfy_quant": ("U8", [16]),
        "blocks.0.v.weight": ("I8", [1024, 1024]),
        "blocks.0.v.weight_scale": ("F32", [1024]),
        "blocks.0.norm.weight": ("F32", [1024]),
        "blocks.0.out.weight": ("F8_E4M3", [1024, 1024]),
    })
    stored = 3 * (1 << 20) + 4 + 16 + 4 * 1024 + 4 * 1024
    plain_fp8_cast = 2 * (1 << 20)  # blocks.0.out has no scale: a plain fp8 weight
    assert capacity_fit.cast_resident_bytes(path, torch.bfloat16) == stored + plain_fp8_cast
    assert capacity_fit.cast_resident_bytes(path, torch.float8_e4m3fn) == stored + (1 << 20)


def test_a_header_the_parser_cannot_read_prices_the_widest_copy_the_cast_can_build(tmp_path):
    """No castable element is narrower than a byte, so the file times the target width."""
    path = tmp_path / "model.ckpt"
    path.write_bytes(b"x" * 4096)
    assert capacity_fit.cast_resident_bytes(str(path), torch.bfloat16) == 2 * 4096
    assert capacity_fit.cast_resident_bytes(str(path), torch.float8_e4m3fn) == 4096


def test_the_target_width_reads_torch_dtypes_and_names_and_an_unknown_one_prices_high():
    assert capacity_fit.cast_width(torch.bfloat16) == 2
    assert capacity_fit.cast_width(torch.float8_e5m2) == 1
    assert capacity_fit.cast_width("fp8_e4m3fn") == 1
    assert capacity_fit.cast_width("bf16") == 2
    assert capacity_fit.cast_width("something_new") == 4


# The probe, the ladder and the last wall.

def test_the_stock_probe_applies_to_a_cast_and_says_so_in_its_block(tmp_path, unified):
    path = _wan22(tmp_path)
    fit = capacity_fit.stock_load_fit(path, {"dtype": torch.bfloat16})
    assert (fit.applies, fit.cast_bytes) == (True, WAN22_ELEMENTS * 2)
    block = fit.measured()
    assert block["cast_bytes"] == WAN22_ELEMENTS * 2 and block["cast_gib"] == 26.6
    assert block["required_gib"] == 60.9 and block["applies"]
    # An uncast block keeps the keys the ledger and the cards already read.
    assert "cast_bytes" not in capacity_fit.stock_load_fit(path, {}).measured()


def test_a_cast_load_on_a_short_box_refuses_with_the_priced_text(tmp_path, unified):
    """B4: the head at 12.2 GiB refuses stage 2 before the load, and offers no slab
    rescue, because a cast slab load keeps the cast copy beside the file."""
    unified["bytes"] = B4_AVAILABLE
    path = _wan22(tmp_path)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        store_residency.resolve(**_ladder(path, {"dtype": torch.bfloat16}))
    assert not isinstance(raised.value, SlabResidencyRescueOffer)
    text = str(raised.value)
    assert ("the load needs 60.9 GiB (a 26.6 GiB file cast to a 26.6 GiB copy, the host "
            "copy it is placed from and the 5.0 GiB host floor) and only 12.2 GiB") in text
    assert f"Slab rescue is unavailable here because {CAST_BLOCKER}" in text
    assert "What would fit: set the loader's weight_dtype to default" in text
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.guard == "stock_load_preflight" and not tag.waivable
    measured = parse_measured(text)
    assert measured is not None
    assert measured["mem_available_bytes"] - measured["headroom_bytes"] == (
        capacity_fit.stock_required_bytes(os.path.getsize(path), WAN22_ELEMENTS * 2))


def test_a_widening_cast_refuses_where_the_uncast_price_would_have_fit(tmp_path, unified):
    """The ladder rebuilds its fit from the row; the cast must survive that inverse,
    or the refusal would quote the uncast price."""
    path = _wan22(tmp_path, dtype="F8_E4M3")      # a plain fp8 file, 13.3 GiB
    size = os.path.getsize(path)
    unified["bytes"] = 40 * _GIB
    assert capacity_fit.stock_required_bytes(size) < 40 * _GIB
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        store_residency.resolve(**_ladder(path, {"dtype": torch.bfloat16}))
    needed = capacity_fit.gib(capacity_fit.stock_required_bytes(size, WAN22_ELEMENTS * 2))
    assert needed == 46.3
    assert (f"the load needs {needed} GiB (a 13.3 GiB file cast to a 26.6 GiB copy") in str(raised.value)


def test_a_cast_load_that_fits_is_priced_and_the_journal_says_so(tmp_path, unified):
    path = _wan22(tmp_path)
    decision = store_residency.resolve(**_ladder(path, {"dtype": torch.bfloat16}))
    assert (decision.use_slab, decision.rung) == (False, store_residency.RUNG_STOCK_FITS)
    assert decision.fit is not None and decision.fit.applies
    assert decision.fit.cast_bytes == WAN22_ELEMENTS * 2
    stored = type("Stored", (), {"slab": None, "residency_rung": decision.rung})()
    line = store_residency.summary_line(stored, decision)
    assert "rung=stock_fits priced=60.9GiB against 100.0GiB by worker_stock_load" in line
    assert "unpriced" not in line and "stock_skipped" not in line


def test_the_last_wall_reprices_a_cast_and_never_falls_back_to_the_bare_file_check(
        tmp_path, unified):
    path = _wan22(tmp_path)
    decision = store_residency.ResidencyDecision(
        False, store_residency.RUNG_STOCK_FITS, "stock", False)
    unified["bytes"] = B4_AVAILABLE
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        store_residency.preload_capacity_check(
            decision, path, WAN22_NAME, {"dtype": torch.bfloat16},
            preflight=lambda *_args: pytest.fail("the bare-file preflight ran"))
    assert ("needs 60.9 GiB (a 26.6 GiB file cast to a 26.6 GiB copy, the host copy it is "
            "placed from and the 5.0 GiB host floor), but only 12.2 GiB is available.") in str(raised.value)


# The fleet agreement.

def _quote(path: str, **overrides) -> RankPrice:
    return capacity_quote.price(**_ladder(path, {"dtype": torch.bfloat16}, **overrides))


def _answer(rank: int, row: RankPrice) -> RankAnswer:
    return RankAnswer(rank, f"box{rank}", (row,), elapsed_s=0.01,
                      box=box_names.HEAD if rank == 0 else box_names.SIBLING)


def test_the_agreement_no_longer_admits_a_cast_load_with_headroom_unstated(
        tmp_path, unified, caplog):
    path = _wan22(tmp_path)
    row = _quote(path)
    assert (row.verdict, row.priced) == (capacity_quote.VERDICT_FITS, True)
    assert row.measured is not None and row.measured["cast_bytes"] == WAN22_ELEMENTS * 2
    with caplog.at_level("INFO"):
        decide([_answer(0, row), _answer(1, row)], 2, unet_name=WAN22_NAME)
    assert "the head=cond:fits/stock_fits@39.1GiB" in caplog.text
    assert "headroom unstated" not in caplog.text
    assert "no_claim" not in caplog.text


def test_a_short_rank_refuses_a_cast_load_for_the_whole_fleet(tmp_path, unified):
    path = _wan22(tmp_path)
    roomy = _quote(path)
    unified["bytes"] = B4_AVAILABLE
    short = _quote(path)
    assert (short.verdict, short.blocker is not None) == (capacity_quote.VERDICT_REFUSE, True)
    with pytest.raises(FleetCapacityError) as raised:
        decide([_answer(0, short), _answer(1, roomy)], 2, unet_name=WAN22_NAME)
    text = str(raised.value)
    assert (f"The head priced a stock load of {WAN22_NAME}: the load needs 60.9 GiB, "
            "a 26.6 GiB file cast to a 26.6 GiB copy, plus the host copy it is placed from, "
            "and 12.2 GiB of unified memory is available, 48.7 GiB short.") in text
    assert "What would fit: set the loader's weight_dtype to default" in text
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.guard == capacity_agreement.GUARD


def test_a_consent_cannot_turn_a_cast_load_into_a_slab_rescue(tmp_path, unified):
    unified["bytes"] = B4_AVAILABLE
    row = _quote(_wan22(tmp_path), rescue_consent={"consent_id": "granted"})
    assert (row.verdict, row.use_slab) == (capacity_quote.VERDICT_REFUSE, False)
    assert row.blocker is not None and row.blocker[0].startswith(CAST_BLOCKER)


def test_a_hot_swap_on_a_cast_resident_credits_the_cast_copy_it_holds():
    """A LoRA change the store makes in place keeps the base: the credit is the
    copy the slot holds, which a widening cast makes larger than the file."""
    from dgx_monarch import capacity_lora_bake

    size, cast = 13 * _GIB, 26 * _GIB
    fit = StockFit(False, True, size, 40 * _GIB, cast_bytes=cast)
    key = repr(("m.safetensors", (("weight_dtype", "bf16"),), (("lora", 1.0),), "fp8"))
    held = {"residency": "stock", "swaps_in_place": True,
            "request_key": repr(("m.safetensors", (("weight_dtype", "bf16"),), (), "fp8"))}
    credited = capacity_lora_bake.credit_hot_swap(fit, held, key, {"would_load": "swap"}, "auto")
    assert credited is not fit and credited.credit_bytes == cast
    assert credited.required_bytes == capacity_fit.stock_required_bytes(size, cast) - cast


# Round two: a swap over the slot's own stock resident is priced after its drop.
# dgx-monarch-wan22-t2v loads both DiTs into the cond slot, one after the other,
# and the store drops the first before it loads the second (actor/store_load.py).
# The 91fd15f records: B4c, the release-test cell on a fresh fleet, ran stage 2's
# agreement with the head at 58.7 GiB and the sibling at 70.5, and the head rose
# to 88.5 GiB at the drop; B4b q2 and q3, warm re-queues, loaded each stage at its
# sampler with the head at 57.9 and 57.8 GiB. All of them rendered.

WAN22_HIGH = "wan2.2_t2v_high_noise_14B_fp16.safetensors"
CAST = {"weight_dtype": "bf16"}


def _endpoint_row(monkeypatch, tmp_path, *, want: str, held: str, available: float,
                  slab_weights: object = "auto", rung: str = "stock_fits",
                  want_options: dict | None = CAST, held_options: dict | None = CAST,
                  arena: float = 0.0, files: dict | None = None) -> RankPrice:
    """One rank's row from the Worker's own quote endpoint, its store holding ``held``.

    ``files`` maps a name to its (dtype, elements), or to None for a file with no
    header; every other name is a wan2.2 fp16 DiT.
    """
    import types

    from dgx_monarch.actor import actor_footprint, model_store, worker_capacity
    from dgx_monarch.actor.store_detect import _to_comfy_model_options

    def staged(name: str) -> str:
        path = tmp_path / name
        dtype, elements = (files or {}).get(name, ("F16", WAN22_ELEMENTS)) or (None, 0)
        if not path.exists():
            if dtype is None:
                path.write_bytes(b"")
                os.truncate(path, 10 * _GIB)      # a file the header parser cannot read
            else:
                _write(path, {"blocks.0.w": (dtype, [elements])})
        return str(path)

    paths = {name: staged(name) for name in (want, held)}
    monkeypatch.setattr(model_store, "resolve_model_path", lambda _kind, name: paths[name])
    monkeypatch.setattr(model_store, "_slab_capable_path", lambda _path: True)
    monkeypatch.setattr(worker_capacity, "_pinned_staging", lambda: False)
    monkeypatch.setattr(actor_footprint, "arena_bytes", lambda: int(arena * _GIB))
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: int(available * _GIB))
    held_key = model_store.normalize_options(held_options or {})
    snapshot = {"cond": {"residency": "stock", "residency_rung": rung, "quant": "fp16",
                         "request_key": repr((held, held_key, (), "fp16"))},
                "uncond": None, "cleanup_failed_slots": []}
    store = types.SimpleNamespace(
        slab_weights=slab_weights, lora_low_rss=True, slab_blocked_reason="",
        family_override=None, uncond=None, snapshot=lambda: dict(snapshot),
        current=types.SimpleNamespace(base_key=(held, held_key), fsdp_checkpoint_pin=None,
                                      slab_auto_retry=False,
                                      request_key=(held, held_key, (), "fp16")))
    worker = types.SimpleNamespace(rank=0, world=2, topology={}, _setup_key=("ready",),
                                   store=store, uma_reserve_gb=0.0)
    assert _to_comfy_model_options(CAST)["dtype"] is torch.bfloat16
    envelope = worker_capacity.quote(worker, {"specs": [
        {"unet_name": want, "options": want_options or {}, "loras": [], "slot": "cond"}],
        "rescue_consent": None})
    (row,) = envelope["quotes"]
    return RankPrice.from_row(row)


def _fleet(head: RankPrice, sibling: RankPrice) -> list[RankAnswer]:
    """The pair as ``collect`` names it; the Workers report their own host names."""
    return [RankAnswer(0, "the-head", (head,), elapsed_s=0.01, box=box_names.HEAD),
            RankAnswer(1, "the-sibling", (sibling,), elapsed_s=0.01, box=box_names.SIBLING)]


def test_the_release_cell_swap_is_priced_after_the_drop_and_admitted(tmp_path, monkeypatch, caplog):
    """B4c stage 2: 60.9 GiB priced whole against 58.7 refused 2.2 GiB short. The
    store drops stage 1's 26.6 GiB cast copy first, so the price credits it."""
    head = _endpoint_row(monkeypatch, tmp_path, want=WAN22_NAME, held=WAN22_HIGH, available=58.7)
    sibling = _endpoint_row(monkeypatch, tmp_path, want=WAN22_NAME, held=WAN22_HIGH, available=70.5)
    assert (head.would_load, head.rung, head.verdict) == ("swap", "stock_fits", "fits")
    assert head.measured is not None
    assert head.measured["credit_bytes"] == WAN22_ELEMENTS * 2 == head.resident_bytes["cond"]
    assert head.measured["required_gib"] == 34.3 and head.measured["headroom_gib"] == 24.4
    with caplog.at_level("INFO"):
        decide(_fleet(head, sibling), 2, unet_name=WAN22_NAME,
               keep=frozenset({WAN22_NAME, WAN22_HIGH}))
    assert "cross-rank capacity admitted: the head=cond:fits/stock_fits@24.4GiB" in caplog.text


def test_the_held_head_still_refuses_after_the_credit(tmp_path, monkeypatch):
    """B4: at 12.2 GiB the head is short even with stage 1's copy dropped."""
    head = _endpoint_row(monkeypatch, tmp_path, want=WAN22_NAME, held=WAN22_HIGH, available=12.2)
    sibling = _endpoint_row(monkeypatch, tmp_path, want=WAN22_NAME, held=WAN22_HIGH, available=68.5)
    assert (head.verdict, head.blocker is not None) == ("refuse", True)
    with pytest.raises(FleetCapacityError) as raised:
        decide(_fleet(head, sibling), 2, unet_name=WAN22_NAME,
               keep=frozenset({WAN22_NAME, WAN22_HIGH}))
    text = str(raised.value)
    assert "the-head" not in text and "rank 0" not in text
    assert (f"The head priced a stock load of {WAN22_NAME} after the Worker drops "
            "wan2.2_t2v_high_noise_14B_fp16, which frees at least 26.6 GiB. Beyond that "
            "the load needs 34.3 GiB, a 26.6 GiB file cast to a 26.6 GiB copy, plus the host copy "
            "it is placed from, and 12.2 GiB of unified memory is available, 22.1 GiB short.") in text
    assert "What would fit: set the loader's weight_dtype to default" in text
    assert "does not model the drop" not in text


@pytest.mark.parametrize(("label", "want", "held", "available", "slab_weights", "rung", "keep"), [
    ("q2 stage 1 at the sampler", WAN22_HIGH, WAN22_NAME, 57.9, False, "explicit_stock",
     {WAN22_HIGH}),
    ("the same swap at the loader's eager agreement", WAN22_HIGH, WAN22_NAME, 57.9, False,
     "explicit_stock", {WAN22_HIGH, WAN22_NAME}),
    ("q2 stage 1 on slab_weights auto", WAN22_HIGH, WAN22_NAME, 57.9, "auto", "stock_fits",
     {WAN22_HIGH}),
    ("stage 2 of the same render at its sampler", WAN22_NAME, WAN22_HIGH, 57.8, False,
     "explicit_stock", {WAN22_NAME}),
])
def test_a_warm_requeue_admits_each_stage_without_a_clear(
        tmp_path, monkeypatch, label, want, held, available, slab_weights, rung, keep):
    """B4b q2 and q3: each stage at its sampler swaps over the other stage's DiT.
    Priced as held, every stage asked for a clear (reviewer's probe_warm_requeue)."""
    head = _endpoint_row(monkeypatch, tmp_path, want=want, held=held, available=available,
                         slab_weights=slab_weights, rung=rung)
    sibling = _endpoint_row(monkeypatch, tmp_path, want=want, held=held, available=70.8,
                            slab_weights=slab_weights, rung=rung)
    assert head.verdict == "fits", label
    decide(_fleet(head, sibling), 2, unet_name=want, keep=frozenset(keep))


def test_a_cast_refused_on_explicit_stock_never_names_the_slab_lever(tmp_path, unified):
    """explicit_stock is what a gated render takes; slab_weights=on would not help a cast."""
    unified["bytes"] = B4_AVAILABLE
    path = _wan22(tmp_path)
    row = _quote(path, slab_weights=False)
    assert row.verdict == "refuse" and row.blocker is not None
    why, options = row.blocker
    assert "a slab load under a cast keeps the cast copy beside the file" in why
    assert "slab_weights=on" not in options
    assert options.startswith("set the loader's weight_dtype to default")
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        store_residency.resolve(**_ladder(path, {"dtype": torch.bfloat16}, slab_weights=False))
    assert "set slab_weights=on" not in str(raised.value)


def test_explicit_slab_refuses_a_cast_rather_than_admitting_it_unpriced(tmp_path, unified):
    unified["bytes"] = B4_AVAILABLE
    path = _wan22(tmp_path)
    row = _quote(path, slab_weights=True)
    assert (row.verdict, row.rung, row.use_slab) == ("refuse", "explicit", True)
    with pytest.raises(FleetCapacityError) as raised:
        decide([_answer(0, row), _answer(1, row)], 2, unet_name=WAN22_NAME)
    assert "slab_weights=on cannot load a checkpoint the loader casts" in str(raised.value)
    assert "What would fit: set slab_weights to off, or" in str(raised.value)
    assert "slab_weights to auto" not in str(raised.value)    # auto admits a vouched family's cast
    unified["bytes"] = 100 * _GIB      # roomy or not: the price cannot be taken
    with pytest.raises(mesh_safety.StockLoadCapacityError) as ladder:
        store_residency.resolve(**_ladder(path, {"dtype": torch.bfloat16}, slab_weights=True))
    assert "slab_weights=on cannot load a checkpoint the loader casts" in str(ladder.value)


# Round three: the words on a credited row, and two narrow routes (review of 54403be).
FLUX1 = "flux1-dev.safetensors"                 # 22.2 GiB bf16, the model in the way
ANIMATE = "wan_animate_2_bf16.safetensors"      # 30.5 GiB bf16, the load that replaces it
PLAIN_FP8 = "plain_fp8.safetensors"             # 13.3 GiB plain fp8
SWAP_FILES = {FLUX1: ("BF16", int(22.2 * _GIB) // 2), ANIMATE: ("BF16", int(30.5 * _GIB) // 2),
              PLAIN_FP8: ("F8_E4M3", WAN22_ELEMENTS)}


def _swap(monkeypatch, tmp_path, available: float, **kwargs) -> RankPrice:
    kwargs.setdefault("want_options", None)
    kwargs.setdefault("held_options", None)
    return _endpoint_row(monkeypatch, tmp_path, available=available, files=SWAP_FILES, **kwargs)


def test_a_rescue_card_on_a_credited_swap_names_the_model_the_worker_drops(tmp_path, monkeypatch):
    """R2-1: the consent text read "beside the 22.2 GiB copy of this file ... its LoRA
    stack changes" for a load that replaces flux1-dev and has no LoRA."""
    row = _swap(monkeypatch, tmp_path, 44.0, want=ANIMATE, held=FLUX1)
    assert row.verdict == "rescue" and row.measured is not None
    assert row.measured["dropped"] == FLUX1
    reason = row.descriptor["human_reason"]
    assert ("needs 46.9 GiB of unified memory under stock residency (a 30.5 GiB file plus "
            "the host copy it is placed from), after the Worker drops flux1-dev, which frees "
            "at least 22.2 GiB, and only 44.0 GiB is available.") in reason
    assert "LoRA" not in reason and "copy of this file" not in reason


def test_a_dtype_change_on_the_held_file_is_a_swap_not_a_lora_change(tmp_path, monkeypatch):
    """R2-2: the held 13.3 GiB plain fp8 file, asked again cast to bf16."""
    head = _swap(monkeypatch, tmp_path, 20.0, want=PLAIN_FP8, held=PLAIN_FP8,
                 want_options=CAST, slab_weights=False)
    sibling = _swap(monkeypatch, tmp_path, 90.0, want=PLAIN_FP8, held=PLAIN_FP8,
                    want_options=CAST, slab_weights=False)
    assert head.verdict == "refuse"
    with pytest.raises(FleetCapacityError) as raised:
        decide(_fleet(head, sibling), 2, unet_name=PLAIN_FP8, keep=frozenset({PLAIN_FP8}))
    text = str(raised.value)
    assert ("priced a stock load of plain_fp8.safetensors after the Worker drops plain_fp8, "
            "which frees at least 13.3 GiB. Beyond that the load needs 32.9 GiB, a 13.3 GiB "
            "file cast to a 26.6 GiB copy, plus the host copy it is placed from, and 20.0 GiB") in text
    assert "LoRA" not in text


def test_a_card_on_a_credited_swap_names_what_the_dropped_model_left_behind(tmp_path, monkeypatch):
    """R2-3: the price already took flux1-dev out, so what a clear returns beyond the
    drop is the memory its load left behind (a 12 GiB arena here)."""
    from dgx_monarch import old_model

    head = _swap(monkeypatch, tmp_path, 40.0, want=ANIMATE, held=FLUX1, slab_weights=False, arena=12.0)
    sibling = _swap(monkeypatch, tmp_path, 90.0, want=ANIMATE, held=FLUX1, slab_weights=False)
    with pytest.raises(old_model.OldModelRefusal) as raised:
        decide(_fleet(head, sibling), 2, unet_name=ANIMATE, keep=frozenset({ANIMATE}))
    text = str(raised.value)
    assert ("The head still holds about 12.0 GiB that earlier loads left behind, and "
            "wan_animate_2_bf16 does not fit beside it.") in text
    assert "the-head" not in text
    assert "The old model (flux1-dev) is still loaded" not in text
    offer = old_model.offer_of(raised.value)
    assert offer is not None and old_model.card_body(offer).startswith(
        "The head still holds about 12.0 GiB that earlier loads left behind")


def test_a_credited_swap_a_clear_cannot_fix_says_its_figure_is_the_whole_price(tmp_path, monkeypatch):
    """R2-3, the plain refusal: 46.9 GiB short of the drop's credit, 69.1 on a cleared box."""
    head = _swap(monkeypatch, tmp_path, 30.0, want=ANIMATE, held=FLUX1, slab_weights=False, arena=2.0)
    sibling = _swap(monkeypatch, tmp_path, 90.0, want=ANIMATE, held=FLUX1, slab_weights=False)
    with pytest.raises(FleetCapacityError) as raised:
        decide(_fleet(head, sibling), 2, unet_name=ANIMATE, keep=frozenset({ANIMATE}))
    text = str(raised.value)
    assert "Beyond that the load needs 46.9 GiB" in text
    assert ("A clear would not make room: wan_animate_2_bf16 needs 69.1 GiB on the head, and a "
            "cleared box would have about 52.2 GiB. That is the whole price, because a cleared "
            "box holds no copy for the Worker to drop.") in text


@pytest.mark.parametrize("arena", [0.0, 2.0])
def test_a_refusal_no_clear_can_fix_never_claims_memory_was_left_behind(tmp_path, monkeypatch, arena):
    """R3-1: with nothing (or too little) left behind, the plain refusal names no
    left-behind memory; the no-room sentence gives the whole price instead."""
    head = _swap(monkeypatch, tmp_path, 30.0, want=ANIMATE, held=FLUX1, slab_weights=False, arena=arena)
    sibling = _swap(monkeypatch, tmp_path, 90.0, want=ANIMATE, held=FLUX1, slab_weights=False)
    with pytest.raises(FleetCapacityError) as raised:
        decide(_fleet(head, sibling), 2, unet_name=ANIMATE, keep=frozenset({ANIMATE}))
    text = str(raised.value)
    assert "left behind" not in text and "still holds" not in text
    assert "That is the whole price, because a cleared box holds no copy" in text


def test_an_unreadable_header_credits_nothing_and_the_fleet_refuses_together(tmp_path, monkeypatch):
    """R2-4: two 10 GiB files with no header, both cast to bf16, head at 17 GiB. The price
    takes the worst case (the file times two); the credit takes nothing, so the agreement
    refuses instead of admitting what the head's own ladder refuses after its drop."""
    unreadable = {"a_fp16.ckpt": None, "b_fp16.ckpt": None}
    head = _endpoint_row(monkeypatch, tmp_path, want="b_fp16.ckpt", held="a_fp16.ckpt",
                         available=17.0, slab_weights=False, files=unreadable)
    assert head.resident_bytes["cond"] == 0
    assert head.verdict == "refuse" and "credit_bytes" not in (head.measured or {})
    assert head.measured is not None and head.measured["required_gib"] == 36.0


def test_explicit_slab_over_a_held_model_does_not_claim_a_price_it_never_took(tmp_path, monkeypatch):
    """R2-7: the explicit-slab cast refusal has no figures, so no as-held sentence."""
    head = _endpoint_row(monkeypatch, tmp_path, want=WAN22_NAME, held=WAN22_HIGH,
                         available=12.2, slab_weights=True)
    assert (head.verdict, head.rung, head.measured) == ("refuse", "explicit", None)
    with pytest.raises(FleetCapacityError) as raised:
        decide(_fleet(head, head), 2, unet_name=WAN22_NAME, keep=frozenset({WAN22_NAME}))
    assert "does not model the drop" not in str(raised.value)
    assert "slab_weights=on cannot load a checkpoint the loader casts" in str(raised.value)


# Round four: the two bot findings on e37b6fe.

def test_a_cast_resident_is_credited_no_more_than_its_drop_can_return(tmp_path, monkeypatch):
    """A credit must be a lower bound. ComfyUI builds fp32 auxiliaries at the cast's
    width on a copying load, and a widening cast is unmeasured, so each float tensor
    is credited at the smaller of its stored and its cast size."""
    held = "mixed_fp16_fp32.safetensors"
    elements = 1 << 30
    files = {PLAIN_FP8: SWAP_FILES[PLAIN_FP8], ANIMATE: SWAP_FILES[ANIMATE]}
    _write(tmp_path / held, {"blocks.0.w": ("F16", [elements]), "blocks.0.norm": ("F32", [elements])})
    mixed = _endpoint_row(monkeypatch, tmp_path, want=ANIMATE, held=held, available=40.0,
                          want_options=None, held_options=CAST, slab_weights=False, files=files)
    assert mixed.measured is not None
    assert mixed.measured["credit_bytes"] == 2 * elements + 2 * elements     # fp32 at bf16 width
    assert capacity_fit.cast_resident_bytes(str(tmp_path / held), torch.bfloat16) == 6 * elements
    widened = _endpoint_row(monkeypatch, tmp_path, want=ANIMATE, held=PLAIN_FP8, available=40.0,
                            want_options=None, held_options=CAST, slab_weights=False, files=files)
    assert widened.measured is not None
    assert widened.measured["credit_bytes"] == WAN22_ELEMENTS        # the stored fp8 bytes, not 2x
    assert capacity_fit.cast_resident_bytes(str(tmp_path / PLAIN_FP8), torch.bfloat16,
                                            lower=True) == WAN22_ELEMENTS


def test_a_resident_that_changes_under_the_quote_gets_no_credit(tmp_path, monkeypatch):
    """The endpoint runs outside the GPU lock. If the slot's resident changes between
    the facts and the snapshot, the snapshot's model must not inherit the old one's
    credit or its old-model figures."""
    import types

    from dgx_monarch.actor import actor_footprint, model_store, worker_capacity

    paths = {name: _write(tmp_path / name, {"blocks.0.w": (SWAP_FILES[name][0], [SWAP_FILES[name][1]])})
             for name in (FLUX1, ANIMATE)}
    small = _write(tmp_path / "small.safetensors", {"blocks.0.w": ("BF16", [1 << 28])})
    paths["small.safetensors"] = small
    monkeypatch.setattr(model_store, "resolve_model_path", lambda _kind, name: paths[name])
    monkeypatch.setattr(model_store, "_slab_capable_path", lambda _path: True)
    monkeypatch.setattr(worker_capacity, "_pinned_staging", lambda: False)
    monkeypatch.setattr(actor_footprint, "arena_bytes", lambda: 0)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 40 * _GIB)

    def resident(name):
        return types.SimpleNamespace(base_key=(name, ()), fsdp_checkpoint_pin=None,
                                     slab_auto_retry=False, request_key=(name, (), (), "bf16"))

    store = types.SimpleNamespace(slab_weights=False, lora_low_rss=True, slab_blocked_reason="",
                                  family_override=None, uncond=None, current=resident(FLUX1))

    def snapshot():
        store.current = resident("small.safetensors")       # a pipelined load lands here
        return {"cond": {"residency": "stock", "residency_rung": "explicit_stock", "quant": "bf16",
                         "request_key": repr(store.current.request_key)},
                "uncond": None, "cleanup_failed_slots": []}

    store.snapshot = snapshot
    worker = types.SimpleNamespace(rank=0, world=2, topology={}, _setup_key=("ready",),
                                   store=store, uma_reserve_gb=0.0)
    (row,) = worker_capacity.quote(worker, {"specs": [
        {"unet_name": ANIMATE, "options": {}, "loras": [], "slot": "cond"}],
        "rescue_consent": None})["quotes"]
    assert row["would_load"] == "swap" and "credit_bytes" not in (row["measured"] or {})
    assert row["resident_bytes"]["cond"] == 0 and row["resident_names"]["cond"] is None
