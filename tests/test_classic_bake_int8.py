"""The classic bake's term for an INT8 ConvRot checkpoint: no per-weight share.

The line (2 GiB and 7.5 MiB a patched weight) is scaled FP8's. Two hardware
blocks read what an INT8 ConvRot load with its stack took beside the stock
price, on one GPU of a DGX Spark under its clock cap (max 2100 MHz),
2026-11-12: 0.5 GiB for Wan-Animate 2 with 1261 patched keys, 2.2 GiB for
MiniMax H3 with 208. The line charged them 11.2 and 3.5, and refused the
shipped Wan-Animate 2 template at readings where it rendered. These tests
hold the INT8 class on a header shaped like that file's (the real file is not
in CI), at the Worker's two walls and at the quote. CPU only.
"""
from __future__ import annotations

import os
import types

import pytest

from dgx_monarch import capacity_classic_bake as price
from dgx_monarch import capacity_fit, mesh_safety
from dgx_monarch.actor import capacity_quote as cq
from dgx_monarch.actor import in_place_fallback as fallback
from dgx_monarch.actor import model_store as ms
from store_ensure_helpers import rig as rig  # a fixture tests ask for by name.
from test_bake_price_quote_and_walls import _rows
from test_capacity_lora_bake import _resolver, _write_safetensors
from test_lora_backup_credit import LORA, _b3_store
from test_lora_backup_credit import quote as quote  # a fixture tests ask for by name.

GIB, MIB = 2 ** 30, 2 ** 20
STACK = [{"name": "a.safetensors", "strength": 1.0}]
BAKE_WORDS = "for the LoRA stack's classic bake"
BLOCKS = 40
# One block's twelve quantized linears, as the INT8 ConvRot Wan-Animate 2 header has them: (name, out, in).
LINEARS = ([(f"self_attn.{name}", 5120, 5120) for name in "qkvo"]
           + [(f"cross_attn.{name}", 5120, 5120) for name in ("q", "k", "v", "o", "k_img", "v_img")]
           + [("ffn.0", 13824, 5120), ("ffn.2", 5120, 13824)])
NORMS = ["self_attn.norm_q", "self_attn.norm_k", "cross_attn.norm_q", "cross_attn.norm_k", "cross_attn.norm_k_img"]
# Full-precision weights outside the blocks that the stack also patches; each is 1 MiB or more.
EMBEDDINGS = [("text_embedding.0", 5120, 4096), ("text_embedding.2", 5120, 5120), ("time_embedding.0", 5120, 256),
              ("time_embedding.2", 5120, 5120), ("time_projection.1", 30720, 5120)]
# What the stack patches: per block 12 weights, their 12 biases, 5 norm weights and norm3's two keys; then
# the embeddings' weights and biases, and 11 small keys of the head. 1261 in all, the hardware cell's count.
HEAD_KEYS = 11
PATCHED = BLOCKS * (len(LINEARS) * 2 + len(NORMS) + 2) + len(EMBEDDINGS) * 2 + HEAD_KEYS
LINE = price.CLASSIC_BAKE_FIXED_BYTES + PATCHED * price.CLASSIC_BAKE_BYTES_PER_WEIGHT
INT8_TERM = price.CLASSIC_BAKE_FIXED_BYTES + len(EMBEDDINGS) * price.CLASSIC_BAKE_BYTES_PER_WEIGHT


def _files(tmp_path, quantized_dtype="I8"):
    """A checkpoint header shaped like the INT8 ConvRot Wan-Animate 2 file, and a LoRA that patches 1261 of its keys."""
    tensors: dict = {}
    lora: dict = {}

    def pair(stem):
        lora[f"diffusion_model.{stem}.lora_down.weight"] = ("BF16", [4, 4])
        lora[f"diffusion_model.{stem}.lora_up.weight"] = ("BF16", [4, 4])

    for block in range(BLOCKS):
        for name, out, inner in LINEARS:
            stem = f"blocks.{block}.{name}"
            tensors[f"{stem}.weight"] = (quantized_dtype, [out, inner])
            tensors[f"{stem}.weight_scale"] = ("F32", [out, 1])
            tensors[f"{stem}.comfy_quant"] = ("U8", [72])
            tensors[f"{stem}.bias"] = ("BF16", [out])
            pair(stem)
            lora[f"diffusion_model.{stem}.diff_b"] = ("BF16", [out])
        for name in NORMS:
            tensors[f"blocks.{block}.{name}.weight"] = ("BF16", [5120])
            lora[f"diffusion_model.blocks.{block}.{name}.diff"] = ("BF16", [5120])
        tensors[f"blocks.{block}.norm3.weight"] = ("BF16", [5120])
        tensors[f"blocks.{block}.norm3.bias"] = ("BF16", [5120])
        lora[f"diffusion_model.blocks.{block}.norm3.diff"] = ("BF16", [5120])
        lora[f"diffusion_model.blocks.{block}.norm3.diff_b"] = ("BF16", [5120])
        tensors[f"blocks.{block}.modulation"] = ("BF16", [1, 6, 5120])  # neither a weight nor a bias: not patched
    for name, out, inner in EMBEDDINGS:
        tensors[f"{name}.weight"] = ("BF16", [out, inner])
        tensors[f"{name}.bias"] = ("BF16", [out])
        pair(name)
        lora[f"diffusion_model.{name}.diff_b"] = ("BF16", [out])
    for index in range(HEAD_KEYS):
        tensors[f"head.{index}.bias"] = ("BF16", [64])
        lora[f"diffusion_model.head.{index}.diff_b"] = ("BF16", [64])
    checkpoint, stack = tmp_path / f"wan_{quantized_dtype}.safetensors", tmp_path / "a.safetensors"
    _write_safetensors(checkpoint, tensors)
    _write_safetensors(stack, lora)
    return str(checkpoint), str(stack)


def test_an_int8_convrot_weight_is_its_own_class_and_needs_its_marker(tmp_path):
    path = tmp_path / "classes.safetensors"
    _write_safetensors(path, {
        "a.weight": ("I8", [8, 8]), "a.weight_scale": ("F32", [8, 1]),
        "b.weight": ("I8", [8, 8]), "b.comfy_quant": ("U8", [72]),
        "c.weight": ("I8", [8, 8]),  # INT8 with no quantization marker beside it: not ConvRot by the header
        "d.weight": ("F8_E4M3", [8, 8]), "d.weight_scale": ("F32", [1]),
        "e.weight": ("F8_E4M3", [8, 8]),  # FP8 with no marker: plain FP8
        "a.bias": ("BF16", [8])})
    tensors = price.read_safetensors_header(str(path)).tensors
    assert price.weight_class("a.weight", tensors) == price.INT8_QUANTIZED
    assert price.weight_class("b.weight", tensors) == price.INT8_QUANTIZED
    assert price.weight_class("c.weight", tensors) == price.OTHER
    assert price.weight_class("d.weight", tensors) == price.FP8_QUANTIZED
    assert price.weight_class("e.weight", tensors) == price.PLAIN_FP8
    assert price.weight_class("a.bias", tensors) == price.FULL_PRECISION
    assert price.NO_PER_WEIGHT_SHARE == (price.INT8_QUANTIZED, price.PLAIN_FP8)


def test_the_wan_animate_2_shaped_int8_stack_is_charged_the_fixed_part_not_the_line(tmp_path):
    checkpoint, stack = _files(tmp_path)
    resolve = _resolver({"a.safetensors": stack})
    tensors = price.read_safetensors_header(checkpoint).tensors
    keys = price.patched_weights(tensors, STACK, resolve)
    assert len(keys) == PATCHED == 1261
    assert sum(price.weight_class(key, tensors) == price.INT8_QUANTIZED for key in keys) == 480
    charged = price.classic_bake_bytes(checkpoint, STACK, resolve)
    # The fixed 2 GiB, and a share for each of the five large full-precision weights beside the INT8 ones.
    assert charged == INT8_TERM
    assert charged >= 0.5 * GIB  # what the hardware cell took beside the stock price
    assert LINE - charged > 9 * GIB  # what the line charged this stack beyond that (11.2 GiB in all)
    # The whole addition to the stock price: the term and the stack's file.
    assert price.classic_charge(checkpoint, STACK, resolve) == INT8_TERM + os.path.getsize(stack)


def test_the_same_header_as_scaled_fp8_keeps_the_line_and_is_not_charged_its_files_again(tmp_path):
    checkpoint, stack = _files(tmp_path, quantized_dtype="F8_E4M3")
    resolve = _resolver({"a.safetensors": stack})
    tensors = price.read_safetensors_header(checkpoint).tensors
    keys = price.patched_weights(tensors, STACK, resolve)
    quantized = [tensors[key].nbytes for key in keys if price.weight_class(key, tensors) == price.FP8_QUANTIZED]
    assert len(keys) == PATCHED and len(quantized) == 480
    # Every patched key counts, the biases included, as the line was fitted.
    assert price.classic_bake_bytes(checkpoint, STACK, resolve) == LINE
    assert price.classic_charge(checkpoint, STACK, resolve) == LINE  # the line holds the files
    # This file's largest quantized weight is past the measured 64 MiB; the bytes guard is off (no record).
    assert max(quantized) > price.MEASURED_LARGEST_WEIGHT_BYTES[price.FP8_QUANTIZED]


def test_the_same_header_as_plain_fp8_is_charged_as_int8_is(tmp_path):
    """FP8 weights with no quantization marker beside them: the fixed part and the files."""
    tensors: dict = {}
    lora: dict = {}
    for block in range(60):
        for name in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out"):
            stem = f"blocks.{block}.{name}"
            tensors[f"{stem}.weight"] = ("F8_E4M3", [3072, 3072])
            tensors[f"{stem}.bias"] = ("F8_E4M3", [3072])
            lora[f"diffusion_model.{stem}.lora_down.weight"] = ("BF16", [4, 4])
            lora[f"diffusion_model.{stem}.lora_up.weight"] = ("BF16", [4, 4])
            lora[f"diffusion_model.{stem}.diff_b"] = ("BF16", [3072])
    checkpoint, stack = tmp_path / "plain.safetensors", tmp_path / "a.safetensors"
    _write_safetensors(checkpoint, tensors)
    _write_safetensors(stack, lora)
    resolve = _resolver({"a.safetensors": str(stack)})
    read = price.read_safetensors_header(str(checkpoint)).tensors
    keys = price.patched_weights(read, STACK, resolve)
    assert len(keys) == 480 and {price.weight_class(key, read) for key in keys} == {price.PLAIN_FP8}
    assert price.classic_bake_bytes(str(checkpoint), STACK, resolve) == price.CLASSIC_BAKE_FIXED_BYTES
    assert price.classic_charge(str(checkpoint), STACK, resolve) == (
        price.CLASSIC_BAKE_FIXED_BYTES + os.path.getsize(stack))


def test_in_an_int8_stack_a_large_weight_of_another_format_keeps_its_share_and_a_small_key_does_not():
    int8 = [(price.INT8_QUANTIZED, 25 * MIB)] * 30  # most of the stack's bytes, as on the real files
    large = [(price.FULL_PRECISION, 40 * MIB)] * 2 + [(price.OTHER, 4 * MIB)]
    small = [(price.FULL_PRECISION, 10 * 1024)] * 4
    per_weight = price.CLASSIC_BAKE_BYTES_PER_WEIGHT
    assert price.term_from_weights(int8) == price.CLASSIC_BAKE_FIXED_BYTES
    assert price.term_from_weights(int8 + small) == price.CLASSIC_BAKE_FIXED_BYTES
    assert price.term_from_weights(int8 + large + small) == price.CLASSIC_BAKE_FIXED_BYTES + 3 * per_weight
    at_the_edge = [(price.FULL_PRECISION, price.SMALL_KEY_BYTES), (price.FULL_PRECISION, price.SMALL_KEY_BYTES - 1)]
    assert price.term_from_weights(int8 + at_the_edge) == price.CLASSIC_BAKE_FIXED_BYTES + per_weight
    # A scaled FP8 weight in the same stack keeps its share, and its own guard.
    scaled = [(price.FP8_QUANTIZED, 16 * MIB)] * 2
    assert price.term_from_weights(int8 + scaled) == price.CLASSIC_BAKE_FIXED_BYTES + 2 * per_weight


def test_a_stack_with_no_int8_weight_is_counted_as_it_was_fitted_small_keys_included():
    stack = [(price.FP8_QUANTIZED, 16 * MIB)] * 2 + [(price.FULL_PRECISION, 10 * 1024)] * 2
    assert price.term_from_weights(stack) == (price.CLASSIC_BAKE_FIXED_BYTES
                                              + 4 * price.CLASSIC_BAKE_BYTES_PER_WEIGHT)
    bf16 = [(price.FULL_PRECISION, 110 * MIB)] * 148 + [(price.FULL_PRECISION, 10 * 1024)] * 20
    assert price.term_from_weights(bf16) == (price.CLASSIC_BAKE_FIXED_BYTES
                                             + 168 * price.CLASSIC_BAKE_BYTES_PER_WEIGHT)
    # The partial-load route's own rule gives no class and no size, and keeps the line.
    pending = [(price.OTHER, 0)] * 268
    assert price.term_from_weights(pending, guard=False) == (price.CLASSIC_BAKE_FIXED_BYTES
                                                             + 268 * price.CLASSIC_BAKE_BYTES_PER_WEIGHT)


# The Worker's two walls on the real fresh load, and the quote, for the same file.

@pytest.fixture
def int8_load(rig, monkeypatch, tmp_path):
    """The real fresh load of the INT8-shaped file with its stack and the switch off; the test sets the reading."""
    store, calls = rig
    store.lora_low_rss = False
    store.slab_weights = False
    checkpoint, stack = _files(tmp_path)
    monkeypatch.setattr(ms, "resolve_model_path", lambda _kind, _name: checkpoint)
    monkeypatch.setattr(cq, "default_resolve_lora_path", lambda _name: stack)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    box = types.SimpleNamespace(available=0)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: box.available)
    stock = capacity_fit.stock_required_bytes(os.path.getsize(checkpoint))
    need = stock + INT8_TERM + os.path.getsize(stack)
    return store, calls, box, stock, need


def test_the_walls_admit_the_int8_load_under_what_the_line_asked(int8_load):
    """The hardware cell read 4.7 GiB under the line's price (44.8 against 49.5) and was refused."""
    store, calls, box, stock, need = int8_load
    by_the_line = stock + LINE
    assert need < by_the_line - int(4.7 * GIB)
    box.available = by_the_line - int(4.7 * GIB)
    store.ensure("wan.safetensors", {}, STACK)
    assert calls.load == 1 and store.current.lora_sig == (("a.safetensors", 1.0),)


def test_the_walls_still_refuse_the_int8_load_under_its_own_price_and_name_the_bake(int8_load):
    store, calls, box, _stock, need = int8_load
    box.available = need - GIB
    with pytest.raises(Exception) as raised:
        store.ensure("wan.safetensors", {}, STACK)
    assert BAKE_WORDS in str(raised.value) and calls.load == 0
    box.available = need + GIB
    store.ensure("wan.safetensors", {}, STACK)
    assert calls.load == 1


def test_the_quote_and_the_walls_charge_the_int8_file_one_figure(quote, monkeypatch, tmp_path):
    quote(_b3_store(), [(LORA, 1.0)])  # installs the fixture's seams
    checkpoint, stack = _files(tmp_path)
    monkeypatch.setattr(ms, "resolve_model_path", lambda _kind, _name: checkpoint)
    monkeypatch.setattr(cq, "default_resolve_lora_path", lambda _name: stack)
    store = ms.ModelStore()
    store.lora_low_rss = False
    store.slab_weights = False
    stock = capacity_fit.stock_required_bytes(os.path.getsize(checkpoint))
    bake = INT8_TERM + os.path.getsize(stack)
    spec = {"unet_name": "wan.safetensors", "options": {}, "loras": STACK, "slot": "cond"}
    row = _rows(store, monkeypatch, [spec], stock + bake + GIB)[0]
    assert row["verdict"] == cq.VERDICT_FITS and row["measured"]["bake"] == "classic"
    assert row["measured"]["bake_bytes"] == bake and row["measured"]["required_bytes"] == stock + bake
    walled = fallback.fresh_load_probe(store, "cond", STACK)(checkpoint, {})
    assert walled.bake_bytes == bake and walled.required_bytes == row["measured"]["required_bytes"]
    assert _rows(store, monkeypatch, [spec], stock + bake - GIB)[0]["verdict"] == cq.VERDICT_REFUSE
