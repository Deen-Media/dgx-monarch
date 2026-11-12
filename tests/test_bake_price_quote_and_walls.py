"""The classic bake's term where it is charged for real: the Worker's two walls and the Worker's quote.

``tests/test_classic_bake_price.py`` holds the arithmetic. These tests hold
the wiring on the real objects: the fresh load's two walls each refuse by a
price that names the bake, and the quote endpoint reads whether a stack
change is a fresh load from the slot's own ComfyUI backup, the fact the
store's hot swap acts on, so the two agree for every format. CPU only.
"""
from __future__ import annotations

import types

import pytest

from dgx_monarch import capacity_classic_bake as price
from dgx_monarch import capacity_fit, mesh_safety
from dgx_monarch.actor import capacity_quote as cq
from dgx_monarch.actor import in_place_fallback as fallback
from dgx_monarch.actor import model_store as ms
from dgx_monarch.actor import store_residency, worker_capacity
from dgx_monarch.actor.store_detect import LivePrecisionEvidence
from dgx_monarch.actor.stored_model import StoredModel
from store_ensure_helpers import rig as rig  # a fixture tests ask for by name.
from test_capacity_lora_bake import _resolver, _write_safetensors
from test_lora_backup_credit import BASE, LORA, NARROW, _b3_store
from test_lora_backup_credit import quote as quote  # a fixture tests ask for by name.

GIB = 2 ** 30
STACK = [{"name": "a.safetensors", "strength": 1.0}]
BAKE_WORDS = "for the LoRA stack's classic bake"
# The fixtures' checkpoints are sparse files with no header, so a stack is charged as the largest measured.
UNREAD_TERM = price.CLASSIC_BAKE_FIXED_BYTES + price.UNREADABLE_STACK_WEIGHTS * price.CLASSIC_BAKE_BYTES_PER_WEIGHT


# The Worker's own walls (the load that went 8.7 GiB past its price was admitted by both).

@pytest.fixture
def classic_load(rig, monkeypatch, tmp_path):
    """The real fresh load of a 20 GiB file with a stack and the switch off, on a box whose reading the test sets."""
    store, calls = rig
    store.lora_low_rss = False
    store.slab_weights = False
    checkpoint = tmp_path / "base.safetensors"
    with open(checkpoint, "wb") as handle:
        handle.truncate(20 * GIB)
    monkeypatch.setattr(ms, "resolve_model_path", lambda _kind, _name: str(checkpoint))
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    box = types.SimpleNamespace(available=0)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: box.available)
    stock = capacity_fit.stock_required_bytes(20 * GIB)
    return store, calls, box, stock


def test_the_fresh_loads_first_wall_refuses_by_a_price_that_names_the_bake(classic_load, monkeypatch):
    store, calls, box, stock = classic_load
    box.available = stock + GIB  # the stock price alone fits; with the bake's share it does not
    monkeypatch.setattr(store_residency, "preload_capacity_check",
                        lambda *_a, **_k: pytest.fail("the first wall admitted this load"))
    with pytest.raises(Exception) as raised:
        store.ensure("base.safetensors", {}, STACK)
    assert BAKE_WORDS in str(raised.value) and calls.load == 0
    # The same reading admits the same file with no stack: the bake is what refused it.
    monkeypatch.undo()


def test_the_fresh_loads_second_wall_refuses_by_a_price_that_names_the_bake(classic_load, monkeypatch):
    store, calls, box, stock = classic_load
    real_resolve = store_residency.resolve

    def resolve(**kwargs):
        box.available = stock + UNREAD_TERM + GIB  # the first wall admits
        decision = real_resolve(**kwargs)
        box.available = stock + GIB  # memory moved before the second wall reads
        return decision

    monkeypatch.setattr(store_residency, "resolve", resolve)
    with pytest.raises(Exception) as raised:
        store.ensure("base.safetensors", {}, STACK)
    assert "stock residency cannot load" in str(raised.value) and BAKE_WORDS in str(raised.value)
    assert calls.load == 0


def test_both_walls_admit_the_same_file_with_no_stack_or_on_the_default(classic_load):
    store, calls, box, stock = classic_load
    box.available = stock + GIB
    store.ensure("base.safetensors", {}, None)
    assert calls.load == 1
    store.lora_low_rss = True  # the default bakes in place and is charged nothing new
    store.ensure("base.safetensors", {}, STACK)
    assert store.current.lora_sig == (("a.safetensors", 1.0),)


def test_both_walls_admit_the_stack_once_the_reading_covers_its_bake(classic_load):
    store, calls, box, stock = classic_load
    box.available = stock + UNREAD_TERM + GIB
    store.ensure("base.safetensors", {}, STACK)
    assert calls.load == 1 and store.current.lora_sig == (("a.safetensors", 1.0),)


# The quote: a fresh load or a hot swap in place, read from the slot's own backup.

def _measured(head):
    return head.measured or {}


@pytest.mark.parametrize("quant", ["bf16", "int8", "fp8"])
def test_a_strength_change_on_the_held_stack_keeps_todays_price_for_every_format(quote, quant):
    """ComfyUI's backup holds every weight the held stack patches, whatever the format, so the store hot-swaps."""
    store = _b3_store(stored__quant_kind=quant)
    store.current.request_key = (BASE, (), ((LORA, 1.0),), quant)
    assert fallback.backup_covers(store.current, "/no/such/checkpoint", [{"name": LORA, "strength": 0.8}], str)
    head = quote(store, [(LORA, 0.8)])
    assert "bake" not in _measured(head) and "in_place_note" not in _measured(head)


@pytest.mark.parametrize("quant", ["bf16", "int8"])
def test_with_the_permission_the_quote_still_prices_no_bake_the_store_will_not_run(quote, monkeypatch, quant):
    quote(_b3_store(), [(LORA, 1.0)])  # installs the fixture's seams
    store = _b3_store(stored__quant_kind=quant)
    store.current.request_key = (BASE, (), ((LORA, 1.0),), quant)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 45 * GIB)
    worker = types.SimpleNamespace(rank=0, world=2, topology={}, _setup_key=("ready",), store=store,
                                   uma_reserve_gb=0.0)
    row = worker_capacity.quote(worker, {"specs": [
        {"unet_name": BASE, "options": {}, "loras": [{"name": LORA, "strength": 0.8}], "slot": "cond"}],
        "rescue_consent": None, "in_place_bake_proven": True})["quotes"][0]
    assert "bake" not in (row["measured"] or {})


def test_a_stack_the_backup_does_not_cover_is_priced_as_the_fresh_load_it_is(quote):
    head = quote(_b3_store(), [(NARROW, 1.0)])  # another file, and this fixture's headers cannot be read
    assert _measured(head)["bake"] == "classic"
    bare = _b3_store(stack=(), base__backup={})
    bare.current.lora_sig = ()
    assert _measured(quote(bare, [(LORA, 1.0)]))["bake"] == "classic"  # a bare base backs nothing


def test_beside_a_held_slot_the_quote_keeps_todays_price_as_the_store_keeps_the_hot_swap(quote):
    bare = _b3_store(stack=(), base__backup={})
    bare.current.lora_sig = ()
    assert _measured(quote(bare, [(LORA, 1.0)]))["bake"] == "classic"
    bare.uncond = StoredModel(
        base_key=("other.safetensors", ()), request_key=("other.safetensors", (), (), "bf16"),
        base_patcher=types.SimpleNamespace(model=types.SimpleNamespace(model_loaded_weight_memory=GIB), backup={}),
        active_patcher=types.SimpleNamespace(patches={}), quant_kind="bf16",
        precision_evidence=LivePrecisionEvidence("bf16", "bf16"), family="wan_animate2", artifact_identity={},
        residency_rung=cq.RUNG_EXPLICIT_STOCK)
    assert "bake" not in _measured(quote(bare, [(LORA, 1.0)]))


def _stored_with_backup(keys, lora_sig=(("held.safetensors", 1.0),), patches=None):
    backup = {key: object() for key in keys}
    return types.SimpleNamespace(
        lora_sig=lora_sig, base_patcher=types.SimpleNamespace(backup=backup),
        active_patcher=types.SimpleNamespace(patches=dict.fromkeys(keys if patches is None else patches)))


def _files(tmp_path, stems):
    checkpoint = tmp_path / "base.safetensors"
    _write_safetensors(checkpoint, {
        "blocks.0.attn.weight": ("I8", [64, 64]), "blocks.0.attn.weight_scale": ("F32", [1]),
        "blocks.0.mlp.weight": ("BF16", [64, 64]), "blocks.1.attn.weight": ("I8", [64, 64])})
    lora = tmp_path / "other.safetensors"
    entries = {}
    for stem in stems:
        entries[f"diffusion_model.{stem}.lora_down.weight"] = ("BF16", [4, 4])
        entries[f"diffusion_model.{stem}.lora_up.weight"] = ("BF16", [4, 4])
    _write_safetensors(lora, entries)
    return str(checkpoint), _resolver({"other.safetensors": str(lora)})


OTHER_FILE = [{"name": "other.safetensors", "strength": 1.0}]
BACKED = ["diffusion_model.blocks.0.attn.weight", "diffusion_model.blocks.0.mlp.weight"]


def test_another_file_on_layers_the_backup_holds_is_covered_and_one_layer_outside_is_not(tmp_path):
    checkpoint, resolve = _files(tmp_path, ["blocks.0.attn", "blocks.0.mlp"])
    assert fallback.backup_covers(_stored_with_backup(BACKED), checkpoint, OTHER_FILE, resolve)  # INT8 and BF16
    checkpoint, resolve = _files(tmp_path, ["blocks.0.attn", "blocks.1.attn"])
    assert not fallback.backup_covers(_stored_with_backup(BACKED), checkpoint, OTHER_FILE, resolve)


def test_what_the_backup_covers_is_read_as_the_store_reads_it(tmp_path):
    checkpoint, resolve = _files(tmp_path, ["blocks.0.attn"])
    held = _stored_with_backup(BACKED)
    assert fallback.backup_covers(held, checkpoint, None, resolve)  # no stack patches nothing
    assert fallback.backup_covers(held, checkpoint, [{"name": "held.safetensors", "strength": 0.5}], resolve)
    assert not fallback.backup_covers(_stored_with_backup([]), checkpoint, OTHER_FILE, resolve)  # a bare base
    # A held stack ComfyUI has not baked yet has patches its backup lacks: its own files are not covered.
    unbaked = _stored_with_backup(BACKED[:1], patches=BACKED)
    assert not fallback.backup_covers(unbaked, "/no/such/checkpoint", [{"name": "held.safetensors"}], resolve)
    # A header that cannot be read answers no, which prices the dearer fresh load.
    assert not fallback.backup_covers(held, "/no/such/checkpoint", OTHER_FILE, resolve)
    unreadable = types.SimpleNamespace(lora_sig=(), base_patcher=types.SimpleNamespace(backup=None))
    assert fallback.backup_covers(unreadable, checkpoint, OTHER_FILE, resolve)  # the store's own gate swaps in place


def test_the_partial_load_route_declines_when_the_bakes_room_is_missing(monkeypatch):
    """``make_whole`` asks for the backups' room and the bake's line; the line alone can decline it."""
    from dgx_monarch.actor import lora_partial_load

    stored = types.SimpleNamespace(base_baked=False, unbake=None, active_patcher=None)
    patcher = types.SimpleNamespace(model=types.SimpleNamespace(**{lora_partial_load.STORED: lambda: stored}))
    pending = [(f"k{index}", object()) for index in range(100)]
    monkeypatch.setattr(lora_partial_load, "shares_host_memory", lambda _patcher: True)
    monkeypatch.setattr(lora_partial_load, "_patched_off_device", lambda _patcher: pending)
    monkeypatch.setattr(lora_partial_load, "_room_needed", lambda _patcher, _pending: GIB)
    line = price.CLASSIC_BAKE_FIXED_BYTES + 100 * price.CLASSIC_BAKE_BYTES_PER_WEIGHT
    monkeypatch.setattr(lora_partial_load, "_host_room", lambda: GIB + line - 1)
    why_not = lora_partial_load.make_whole(patcher)
    assert why_not is not None and "patched layer(s) still off the device needs" in why_not
    assert stored.base_baked is False
