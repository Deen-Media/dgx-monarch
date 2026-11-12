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
from dgx_monarch.actor import store_bake, store_residency, worker_capacity
from dgx_monarch.actor.store_detect import LivePrecisionEvidence
from dgx_monarch.actor.stored_model import StoredModel
from store_ensure_helpers import rig as rig  # a fixture tests ask for by name.
from test_capacity_lora_bake import _resolver, _write_safetensors
from test_lora_backup_credit import BASE, BF16_BYTES, LORA, LORA_FILES, NARROW, _b3_store
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


# The base as the store compares it, and a pair's second model against the budget left.

def _rows(store, monkeypatch, specs, available, proven=False):
    """``worker_capacity.quote``'s rows for these specs at one reading, with the ``quote`` fixture's seams in."""
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: available)
    worker = types.SimpleNamespace(rank=0, world=2, topology={}, _setup_key=("ready",), store=store,
                                   uma_reserve_gb=0.0)
    return worker_capacity.quote(worker, {"specs": specs, "rescue_consent": None,
                                          "in_place_bake_proven": proven})["quotes"]


RECAST = {"weight_dtype": "default"}  # another base key by the store's rule, and no cast in the price
COVERED = [{"name": LORA, "strength": 0.8}]  # a strength change: the held backup covers it


def test_a_loader_option_changed_on_a_held_file_is_priced_as_the_fresh_load_the_store_makes(quote, monkeypatch):
    """The store compares the whole base key, so the held backup and the held copy are not this request's."""
    quote(_b3_store(), [(LORA, 1.0)])  # installs the fixture's seams
    store = _b3_store()
    same = {"unet_name": BASE, "options": {}, "loras": COVERED, "slot": "cond"}
    changed = dict(same, options=RECAST)
    assert ms.ModelStore.make_keys(BASE, RECAST, None, "")[0] != store.current.base_key
    assert "bake" not in _rows(store, monkeypatch, [same], 120 * GIB)[0]["measured"]  # the hot swap in place
    measured = _rows(store, monkeypatch, [changed], 120 * GIB)[0]["measured"]
    assert measured["bake"] == "classic" and measured["bake_bytes"] == UNREAD_TERM + LORA_FILES[LORA]
    # A reading the stock price fits and the bake's share does not: the quote refuses what the wall would.
    short = measured["required_bytes"] - measured["bake_bytes"] + GIB
    assert _rows(store, monkeypatch, [same], short)[0]["verdict"] == cq.VERDICT_FITS
    assert _rows(store, monkeypatch, [changed], short)[0]["verdict"] == cq.VERDICT_REFUSE


def test_with_the_permission_a_changed_option_prices_the_bare_reload_not_a_bake_on_the_held_copy(
        quote, monkeypatch):
    quote(_b3_store(), [(LORA, 1.0)])  # installs the fixture's seams
    bare = _b3_store(stack=(), base__backup={})
    bare.current.lora_sig = ()
    spec = {"unet_name": BASE, "options": {}, "loras": [{"name": LORA, "strength": 1.0}], "slot": "cond"}
    on_the_held_copy = price.in_place_bake_bytes(LORA_FILES[LORA])
    kept = _rows(bare, monkeypatch, [spec], 45 * GIB, proven=True)[0]["measured"]
    assert kept["bake"] == "in_place" and kept["bake_bytes"] == on_the_held_copy
    # Another base key: the store drops the held copy and loads the base bare first, at the stock price.
    changed = _rows(bare, monkeypatch, [dict(spec, options=RECAST)], 45 * GIB, proven=True)[0]["measured"]
    reload = capacity_fit.stock_required_bytes(BF16_BYTES) - capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES - BF16_BYTES
    assert changed["bake"] == "in_place" and changed["bake_bytes"] == reload > on_the_held_copy


def test_the_store_loads_fresh_when_a_loader_option_changes_on_a_held_file(classic_load, monkeypatch):
    """The store's side of the same rule, on the real ``ensure``: no hot swap, a fresh load at the walls."""
    store, calls, box, stock = classic_load
    box.available = stock + UNREAD_TERM + GIB
    store.ensure("base.safetensors", {}, STACK)
    asked = []
    real_swap = store_bake.hot_swap_transition

    def hot_swap(*args, **kwargs):
        asked.append("hot swap")
        return real_swap(*args, **kwargs)

    monkeypatch.setattr(store_bake, "hot_swap_transition", hot_swap)
    box.available = stock + GIB  # the stock price fits; the bake's share does not
    with pytest.raises(Exception) as raised:
        store.ensure("base.safetensors", RECAST, [{"name": "a.safetensors", "strength": 0.5}])
    assert asked == [] and BAKE_WORDS in str(raised.value) and calls.load == 1


def _base_identity(signature):
    return {"artifacts": [{"kind": "diffusion_models", "name": BASE, "signature": signature}]}


def test_the_slot_holds_a_base_by_the_whole_key_and_is_set_aside_when_its_file_moved(tmp_path):
    from dgx_monarch.gate_ledger import artifact_signature

    checkpoint = tmp_path / "base.safetensors"
    checkpoint.write_bytes(b"a" * 4096)
    key = ms.ModelStore.make_keys("base.safetensors", {}, None, "")[0]
    stored = types.SimpleNamespace(base_key=key, artifact_identity={})
    assert fallback.holds_base(stored, key)
    assert not fallback.holds_base(None, key) and not fallback.holds_base(stored, ())
    assert not fallback.holds_base(stored, ms.ModelStore.make_keys("base.safetensors", RECAST, None, "")[0])
    assert not fallback.checkpoint_moved(stored, str(checkpoint))  # no fingerprint recorded: nothing to compare
    stored.artifact_identity = _base_identity(artifact_signature(str(checkpoint)))
    assert not fallback.checkpoint_moved(stored, str(checkpoint))
    checkpoint.write_bytes(b"b" * 8192)  # rewritten under its own name: ``ensure`` sets such a slot aside
    assert fallback.checkpoint_moved(stored, str(checkpoint))
    checkpoint.unlink()
    assert fallback.checkpoint_moved(stored, str(checkpoint))  # gone: the dearer reading
    # The worker endpoint puts the two together: held for a swap, or set aside.
    spec = {"unet_name": "base.safetensors", "options": {}}
    stored.artifact_identity = {}
    assert worker_capacity._held_base(stored, spec, "base.safetensors", str(checkpoint)) == (True, False)
    assert worker_capacity._held_base(stored, dict(spec, options=RECAST), "base.safetensors", "p") == (False, False)
    assert worker_capacity._held_base(stored, dict(spec, options=7), "base.safetensors", "p") == (False, False)
    stored.artifact_identity = _base_identity("another file's fingerprint")
    assert worker_capacity._held_base(stored, spec, "base.safetensors", str(checkpoint)) == (False, True)


def test_a_slot_the_store_sets_aside_is_quoted_the_fresh_load_and_no_in_place_bake(quote, monkeypatch):
    """A checkpoint rewritten under its own name: ``ensure`` sets the slot aside and never asks ``take``."""
    from dgx_monarch.gate_ledger import artifact_signature

    quote(_b3_store(), [(LORA, 1.0)])  # installs the fixture's seams
    bare = _b3_store(stack=(), base__backup={})
    bare.current.lora_sig = ()
    spec = {"unet_name": BASE, "options": {}, "loras": [{"name": LORA, "strength": 1.0}], "slot": "cond"}
    bare.current.artifact_identity = _base_identity(artifact_signature(ms.resolve_model_path("", BASE)))
    kept = _rows(bare, monkeypatch, [spec], 45 * GIB, proven=True)[0]
    assert kept["verdict"] == cq.VERDICT_FITS and kept["measured"]["bake"] == "in_place"  # the file the slot loaded
    bare.current.artifact_identity = _base_identity("another file's fingerprint")
    moved = _rows(bare, monkeypatch, [spec], 45 * GIB, proven=True)[0]
    assert moved["verdict"] == cq.VERDICT_REFUSE and moved["measured"]["bake"] == "classic"
    assert moved["measured"]["in_place_note"] == "; " + fallback.SET_ASIDE
    # Without the permission the refusal says the gate has not passed, as for any slot.
    unproven = _rows(bare, monkeypatch, [spec], 45 * GIB)[0]["measured"]
    assert unproven["bake"] == "classic" and "identity-gate PASS" in unproven["in_place_note"]


def test_a_pairs_second_model_takes_the_bake_that_fits_the_budget_the_first_leaves(quote, monkeypatch, tmp_path):
    """The store bakes the second model beside the loaded first, so the quote chooses after the first's charge."""
    quote(_b3_store(), [(LORA, 1.0)])  # installs the fixture's seams
    small = tmp_path / "pair.safetensors"
    with open(small, "wb") as handle:
        handle.truncate(10 * GIB)
    monkeypatch.setattr(ms, "resolve_model_path", lambda _kind, _name: str(small))
    store = ms.ModelStore()
    store.lora_low_rss = False
    store.slab_weights = False
    stock = capacity_fit.stock_required_bytes(10 * GIB)
    classic = stock + UNREAD_TERM + LORA_FILES[LORA]
    specs = [{"unet_name": "first.safetensors", "options": {}, "slot": "cond"},
             {"unet_name": "second.safetensors", "options": {}, "slot": "uncond",
              "loras": [{"name": LORA, "strength": 1.0}]}]
    reading = 2 * stock + GIB  # beside the first model the bare base and its in-place bake fit; the classic does not
    assert classic <= reading < stock + classic
    first, second = _rows(store, monkeypatch, specs, reading, proven=True)
    assert first["verdict"] == cq.VERDICT_FITS and "bake" not in first["measured"]
    assert second["verdict"] == cq.VERDICT_FITS and second["measured"]["bake"] == "in_place"
    assert second["measured"]["mem_available_bytes"] == reading - stock  # the first's charge, taken off once
    # Priced alone at the same reading the same slot keeps the classic bake.
    alone = _rows(store, monkeypatch, specs[1:], reading, proven=True)[0]
    assert alone["verdict"] == cq.VERDICT_FITS and alone["measured"]["bake"] == "classic"
    # Without the permission the pair is refused, and the row says what was left and why no in-place bake.
    refused = _rows(store, monkeypatch, specs, reading)[1]
    assert refused["verdict"] == cq.VERDICT_REFUSE and refused["measured"]["bake"] == "classic"
    assert "identity-gate PASS" in refused["measured"]["in_place_note"]


def test_an_unchanged_request_is_not_quoted_a_reuse_once_a_file_it_names_changed_on_disk(quote, monkeypatch):
    """``ensure`` reuses on the key and on the files the slot loaded; read by the key alone it was quoted nothing."""
    from dgx_monarch.gate_ledger import artifact_signature

    quote(_b3_store(), [(LORA, 1.0)])  # installs the fixture's seams
    store = _b3_store()
    spec = {"unet_name": BASE, "options": {}, "loras": [{"name": LORA, "strength": 1.0}], "slot": "cond"}
    checkpoint = artifact_signature(ms.resolve_model_path("", BASE))
    lora = artifact_signature(cq.default_resolve_lora_path(LORA))

    def loaded(checkpoint_print, lora_print):
        return {"artifacts": [{"kind": "diffusion_models", "name": BASE, "signature": checkpoint_print},
                              {"kind": "loras", "name": LORA, "signature": lora_print}]}

    store.current.artifact_identity = loaded(checkpoint, lora)
    same = _rows(store, monkeypatch, [spec], 120 * GIB)[0]
    assert same["would_load"] == cq.WOULD_LOAD_REUSE and "bake" not in (same["measured"] or {})
    for moved in (loaded("another file's fingerprint", lora), loaded(checkpoint, "another file's fingerprint")):
        store.current.artifact_identity = moved
        row = _rows(store, monkeypatch, [spec], 120 * GIB)[0]
        # The store reloads: base and stack fresh, with the classic bake (this fixture's headers map nothing).
        assert row["would_load"] == cq.WOULD_LOAD_SWAP and row["measured"]["bake"] == "classic"
        assert row["measured"]["required_bytes"] > UNREAD_TERM
        assert "backup_bytes" not in row["measured"]  # a backup that no longer covers the stack is not credited
    # The moved checkpoint sets the slot aside for a changed request too: a fresh load, no credit for the backup.
    store.current.artifact_identity = loaded("another file's fingerprint", lora)
    aside = _rows(store, monkeypatch, [dict(spec, loras=COVERED)], 120 * GIB)[0]["measured"]
    assert aside["bake"] == "classic" and "backup_bytes" not in aside
    # A changed request on a slot whose files are as loaded keeps the price it had.
    store.current.artifact_identity = loaded(checkpoint, lora)
    strength = _rows(store, monkeypatch, [dict(spec, loras=COVERED)], 120 * GIB)[0]
    assert "bake" not in strength["measured"] and strength["measured"]["backup_bytes"] > 0
    # A LoRA rewritten to patch what the backup still covers is the stack change the store makes of it:
    # the hot swap in place, at a strength change's own price, the backup's credit included.
    store.current.artifact_identity = loaded(checkpoint, "another file's fingerprint")
    monkeypatch.setattr(fallback, "backup_covers", lambda *_args, **_kwargs: True)
    rewritten = _rows(store, monkeypatch, [spec], 120 * GIB)[0]
    assert rewritten["would_load"] == cq.WOULD_LOAD_SWAP and "bake" not in rewritten["measured"]
    assert rewritten["measured"]["required_bytes"] == strength["measured"]["required_bytes"]


def test_a_rewritten_lora_is_priced_as_a_stack_change_on_the_default_too(quote, monkeypatch):
    """With ``lora_low_rss`` on the store un-bakes and re-bakes in place: a swap's price, never a reuse's."""
    from dgx_monarch.gate_ledger import artifact_signature

    quote(_b3_store(), [(LORA, 1.0)])  # installs the fixture's seams
    store = _b3_store(store__lora_low_rss=True, stored__unbake=object(), stored__base_baked=True,
                      stored__baked_in_place=True)
    spec = {"unet_name": BASE, "options": {}, "loras": [{"name": LORA, "strength": 1.0}], "slot": "cond"}
    recorded = {"artifacts": [
        {"kind": "diffusion_models", "name": BASE, "signature": artifact_signature(ms.resolve_model_path("", BASE))},
        {"kind": "loras", "name": LORA, "signature": artifact_signature(cq.default_resolve_lora_path(LORA))}]}
    store.current.artifact_identity = recorded
    assert _rows(store, monkeypatch, [spec], 120 * GIB)[0]["would_load"] == cq.WOULD_LOAD_REUSE
    strength = _rows(store, monkeypatch, [dict(spec, loras=COVERED)], 120 * GIB)[0]
    recorded["artifacts"][1]["signature"] = "another file's fingerprint"
    rewritten = _rows(store, monkeypatch, [spec], 120 * GIB)[0]
    assert rewritten["would_load"] == cq.WOULD_LOAD_SWAP and "bake" not in rewritten["measured"]
    assert rewritten["measured"]["required_bytes"] == strength["measured"]["required_bytes"]


def test_a_pairs_second_model_reads_the_first_slot_as_held_as_the_store_will_find_it(quote, monkeypatch):
    """The conditional slot is empty at the quote and loads first; the store then hot-swaps the second in place."""
    quote(_b3_store(), [(LORA, 1.0)])  # installs the fixture's seams
    store = _b3_store()
    store.uncond, store.current = store.current, None
    first = {"unet_name": "first.safetensors", "options": {}, "slot": "cond"}
    second = {"unet_name": BASE, "options": {}, "slot": "uncond",
              "loras": [{"name": NARROW, "strength": 1.0}]}  # a file the held backup does not cover
    alone = _rows(store, monkeypatch, [second], 200 * GIB)[0]
    assert alone["slot"] == "uncond" and alone["measured"]["bake"] == "classic"  # no other model: a fresh load
    paired = _rows(store, monkeypatch, [first, second], 200 * GIB)[1]
    assert paired["slot"] == "uncond" and "bake" not in paired["measured"]  # beside the first: the hot swap


def test_one_rewritten_file_in_a_stack_of_two_held_files_ends_the_shortcut(tmp_path):
    from dgx_monarch.gate_ledger import artifact_signature

    checkpoint, _resolve = _files(tmp_path, ["blocks.0.attn"])
    one, two = tmp_path / "one.safetensors", tmp_path / "two.safetensors"

    def write(path, stems):
        entries = {}
        for stem in stems:
            entries[f"diffusion_model.{stem}.lora_down.weight"] = ("BF16", [4, 4])
            entries[f"diffusion_model.{stem}.lora_up.weight"] = ("BF16", [4, 4])
        _write_safetensors(path, entries)

    write(one, ["blocks.0.attn"])
    write(two, ["blocks.0.mlp"])
    resolve = _resolver({"one.safetensors": str(one), "two.safetensors": str(two)})
    stored = _stored_with_backup(BACKED, lora_sig=(("one.safetensors", 1.0), ("two.safetensors", 1.0)))
    stored.artifact_identity = {"artifacts": [
        {"kind": "diffusion_models", "name": "base.safetensors", "signature": "x"},
        {"kind": "loras", "name": "one.safetensors", "signature": artifact_signature(str(one))},
        {"kind": "loras", "name": "two.safetensors", "signature": artifact_signature(str(two))}]}
    both = [{"name": "one.safetensors", "strength": 0.5}, {"name": "two.safetensors", "strength": 0.5}]
    assert fallback.backup_covers(stored, "/no/such/checkpoint", both, resolve)  # by name, no header read
    assert not fallback.stack_files_moved(stored, both, resolve)
    write(two, ["blocks.0.mlp", "blocks.1.attn"])  # the second file alone, to a layer the backup lacks
    assert not fallback.backup_covers(stored, checkpoint, both, resolve)
    assert not fallback.backup_covers(stored, "/no/such/checkpoint", both, resolve)  # no shortcut by the first's name
    assert fallback.stack_files_moved(stored, both, resolve) and not fallback.stack_files_moved(stored, both[:1], resolve)


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


def test_a_held_file_rewritten_on_disk_under_its_own_name_is_read_again_not_trusted_by_name(tmp_path):
    """The store builds the stack from the file as it is now, so the quote must not take its name for its keys."""
    from dgx_monarch.gate_ledger import artifact_signature

    checkpoint, _resolve = _files(tmp_path, ["blocks.0.attn"])
    held_file = tmp_path / "held.safetensors"

    def write(stems):
        entries = {}
        for stem in stems:
            entries[f"diffusion_model.{stem}.lora_down.weight"] = ("BF16", [4, 4])
            entries[f"diffusion_model.{stem}.lora_up.weight"] = ("BF16", [4, 4])
        _write_safetensors(held_file, entries)

    write(["blocks.0.attn", "blocks.0.mlp"])
    resolve = _resolver({"held.safetensors": str(held_file)})
    stored = _stored_with_backup(BACKED)
    stored.artifact_identity = {"artifacts": [
        {"kind": "diffusion_models", "name": "base.safetensors", "signature": "x"},
        {"kind": "loras", "name": "held.safetensors", "signature": artifact_signature(str(held_file))}]}
    strength_change = [{"name": "held.safetensors", "strength": 0.5}]
    assert fallback.backup_covers(stored, "/no/such/checkpoint", strength_change, resolve)  # by name, no header read
    write(["blocks.0.attn", "blocks.0.mlp", "blocks.1.attn"])  # the same name, one more layer
    assert not fallback.backup_covers(stored, checkpoint, strength_change, resolve)  # mapped again: not covered
    write(["blocks.0.attn"])  # rewritten to patch less: still inside the backup, read from its headers
    assert fallback.backup_covers(stored, checkpoint, strength_change, resolve)
    held_file.unlink()
    assert not fallback.backup_covers(stored, checkpoint, strength_change, resolve)  # gone: not covered
    # A slot that recorded no fingerprint for the name has nothing to compare and keeps the shortcut.
    stored.artifact_identity = {}
    assert fallback.backup_covers(stored, "/no/such/checkpoint", strength_change, resolve)


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
