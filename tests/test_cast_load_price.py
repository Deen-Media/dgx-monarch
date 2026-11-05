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

from dgx_monarch import capacity_agreement, capacity_fit, mesh_safety
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
    return RankAnswer(rank, f"box{rank}", (row,), elapsed_s=0.01)


def test_the_agreement_no_longer_admits_a_cast_load_with_headroom_unstated(
        tmp_path, unified, caplog):
    path = _wan22(tmp_path)
    row = _quote(path)
    assert (row.verdict, row.priced) == (capacity_quote.VERDICT_FITS, True)
    assert row.measured is not None and row.measured["cast_bytes"] == WAN22_ELEMENTS * 2
    with caplog.at_level("INFO"):
        decide([_answer(0, row), _answer(1, row)], 2, unet_name=WAN22_NAME)
    assert "rank 0=cond:fits/stock_fits@39.1GiB" in caplog.text
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
    assert (f"rank 0 (host box0) priced a stock load of {WAN22_NAME}: the load needs 60.9 GiB, "
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
