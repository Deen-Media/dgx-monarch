"""Where the real ``ModelStore.ensure`` asks for the in-place bake, and what the real swap marks.

The classic bake's term and its fallback belong to a stack change that ends
in a fresh load (``actor/in_place_fallback.py``). ``ensure``'s own order
gives that: reuse, the classic hot swap, then the fallback's one call before
anything is dropped, then the fresh load. These tests drive the real method
on the store rig and record that order. CPU only.
"""
from __future__ import annotations

import types

import pytest

from dgx_monarch.actor import in_place_fallback, store_bake
from dgx_monarch.actor import model_store as ms
from dgx_monarch.capacity_classic_bake import IN_PLACE_BAKE_PROVEN
from store_ensure_helpers import rig as rig  # a fixture tests ask for by name.

BASE = "base.safetensors"
STACK = [{"name": "a.safetensors", "strength": 1.0}]
ALLOW = {IN_PLACE_BAKE_PROVEN: True}


def _patched_active(keys):
    def build(self, base, stack):
        return types.SimpleNamespace(tag="ACTIVE", model=base.model,
                                     patches={key: [("patch",)] for key in keys} if stack else {})
    return build


@pytest.fixture
def classic(rig, monkeypatch):
    """A classic store that holds BASE bare and placed, with every ask of the swap and the fallback recorded."""
    store, calls = rig
    store.lora_low_rss = False
    store.ensure(BASE, {}, None)
    store.current.base_patcher.model.model_loaded_weight_memory = 1 << 30
    store.current.base_patcher.backup = {}
    monkeypatch.setattr(ms.ModelStore, "_build_active", _patched_active(["w1", "w2"]))
    asked: list[str] = []
    real_swap = store_bake.hot_swap_transition

    def hot_swap(*args, **kwargs):
        asked.append("hot swap")
        return real_swap(*args, **kwargs)

    monkeypatch.setattr(store_bake, "hot_swap_transition", hot_swap)
    answer = types.SimpleNamespace(value=None, loads_at_ask=None, bare=None)

    def take(store_, slot, unet_name, options, lora_stack, base_key, identity, logger, artifact_identity,
             load_bare):
        asked.append("take")
        answer.loads_at_ask = calls.load
        answer.bare = load_bare
        return answer.value

    monkeypatch.setattr(in_place_fallback, "take", take)
    return store, calls, asked, answer


def test_ensure_asks_the_hot_swap_first_then_the_fallback_before_any_drop(classic):
    store, calls, asked, answer = classic
    bare = store.current
    answer.value = ("IN PLACE", "hot-swap")
    assert store.ensure(BASE, {}, STACK) == ("IN PLACE", "hot-swap")
    assert asked == ["hot swap", "take"]
    assert calls.load == 1 and store.current is bare  # nothing was dropped or loaded when it was asked


def test_a_fallback_that_declines_is_followed_by_the_fresh_load(classic):
    store, calls, asked, answer = classic
    bare = store.current
    _patcher, transition = store.ensure(BASE, {}, STACK)
    assert asked == ["hot swap", "take"] and answer.loads_at_ask == 1
    assert transition == "load" and calls.load == 2 and store.current is not bare


def test_a_declined_hot_swap_notes_the_stack_it_read_for_the_fresh_load_that_follows(classic, monkeypatch):
    """The real swap reads the stack to count what it would copy; the walls then charge none of its files."""
    store, _calls, _asked, _answer = classic
    seen: dict = {}
    real_probe = in_place_fallback.fresh_load_probe

    def take(store_, slot, unet_name, options, lora_stack, *_rest):
        seen["at take"] = in_place_fallback.stack_on_host(store_, slot, lora_stack)
        return None

    def probe(store_, slot, lora_stack):
        seen["at the walls"] = in_place_fallback.stack_on_host(store_, slot, lora_stack)
        return real_probe(store_, slot, lora_stack)

    monkeypatch.setattr(in_place_fallback, "take", take)
    monkeypatch.setattr(in_place_fallback, "fresh_load_probe", probe)
    _patcher, transition = store.ensure(BASE, {}, STACK)
    assert transition == "load" and seen == {"at take": True, "at the walls": True}
    assert not in_place_fallback.stack_on_host(store, "cond", STACK)  # read once
    # The same stack into a slot the store holds nothing in: no swap ran, so nothing was read first.
    store._drop("cond")
    seen.clear()
    store.ensure(BASE, {}, STACK)
    assert seen == {"at take": False, "at the walls": False}


def test_a_hot_swap_that_stays_in_place_leaves_no_note(classic):
    store, _calls, _asked, _answer = classic
    in_place_fallback.note_stack_read(store, "cond", STACK)  # a note an earlier call left behind
    store.current.base_patcher.backup = {"w1": object(), "w2": object()}  # ComfyUI's backup covers the stack
    _patcher, transition = store.ensure(BASE, {}, STACK)
    assert transition == "hot-swap" and not in_place_fallback.stack_on_host(store, "cond", STACK)


def test_a_fresh_load_that_raised_leaves_no_note_for_the_requeue(classic, monkeypatch):
    """The note binds to one ``ensure`` call: a requeue of the same stack is charged its files.

    The swap declines and leaves its note; the fresh load raises before its
    walls price, with the slot emptied. The same request again finds an empty
    slot, runs no swap and reads nothing first, so nothing may be believed
    read. ``ensure`` clears the note on entry.
    """
    from dgx_monarch.actor import store_load

    store, _calls, _asked, _answer = classic
    seen: dict = {}
    real_probe = in_place_fallback.fresh_load_probe

    def failing_load(store_, slot, *_args, **_kwargs):
        store_._drop(slot)
        raise RuntimeError("the load failed before its walls priced")

    with monkeypatch.context() as patch:
        patch.setattr(store_load, "fresh_load", failing_load)
        with pytest.raises(RuntimeError, match="before its walls priced"):
            store.ensure(BASE, {}, STACK)
    assert store.current is None
    assert in_place_fallback.stack_on_host(store, "cond", STACK)  # what the failed call left on the store

    def take(store_, slot, unet_name, options, lora_stack, *_rest):
        seen["at take"] = in_place_fallback.stack_on_host(store_, slot, lora_stack)
        return None

    def probe(store_, slot, lora_stack):
        seen["at the walls"] = in_place_fallback.stack_on_host(store_, slot, lora_stack)
        return real_probe(store_, slot, lora_stack)

    monkeypatch.setattr(in_place_fallback, "take", take)
    monkeypatch.setattr(in_place_fallback, "fresh_load_probe", probe)
    _patcher, transition = store.ensure(BASE, {}, STACK)  # the requeue
    assert transition == "load" and seen == {"at take": False, "at the walls": False}


def test_every_ensure_call_starts_with_no_note(classic):
    store, _calls, _asked, _answer = classic
    in_place_fallback.note_stack_read(store, "uncond", STACK)  # left by a call that never reached its fresh load
    store.ensure(BASE, {}, None)  # a reuse: no swap, no fresh load
    assert not in_place_fallback.stack_on_host(store, "uncond", STACK)


def test_a_change_the_hot_swap_makes_in_place_never_asks_the_fallback(classic):
    store, calls, asked, _answer = classic
    store.current.base_patcher.backup = {"w1": object(), "w2": object()}  # ComfyUI's backup covers the stack
    _patcher, transition = store.ensure(BASE, {}, STACK)
    assert transition == "hot-swap" and asked == ["hot swap"] and calls.load == 1


def test_beside_a_held_slot_the_hot_swap_stays_in_place_and_the_fallback_is_not_asked(classic):
    store, _calls, asked, _answer = classic
    store.ensure("other.safetensors", {}, None, slot="uncond")
    asked.clear()
    _patcher, transition = store.ensure(BASE, {}, STACK)
    assert transition == "hot-swap" and asked == ["hot swap"]


def test_the_default_never_asks_the_classic_swap_and_the_real_fallback_declines(rig, monkeypatch):
    store, _calls = rig  # lora_low_rss on
    asked = []
    monkeypatch.setattr(store_bake, "hot_swap_transition", lambda *a, **k: asked.append("hot swap"))
    monkeypatch.setattr(store_bake, "lazy_swap_transition",
                        lambda *a, **k: asked.append("lazy swap") or ("P", "hot-swap"))
    store.ensure(BASE, {}, None)
    with in_place_fallback.permission(ALLOW):
        assert store.ensure(BASE, {}, STACK) == ("P", "hot-swap")
    assert asked == ["lazy swap"]


def test_the_bare_load_the_fallback_is_handed_is_the_real_ensure_with_no_stack(classic):
    store, _calls, asked, answer = classic
    store._drop("cond")
    assert store.current is None
    store.ensure(BASE, {}, STACK)  # take declines; the callable it was handed is kept
    store._drop("cond")
    asked.clear()
    answer.bare()
    assert store.current is not None and store.current.lora_sig == () and store.current.base_key[0] == BASE
    # The bare load passes the same point with no stack, where the real fallback declines at once
    # (test_classic_bake_price holds that), and it never reaches the classic swap.
    assert asked == ["take"]


def test_a_slot_baked_in_place_is_reused_only_with_the_switch_on_or_the_permission(classic):
    store, calls, asked, _answer = classic
    store.ensure(BASE, {}, STACK)  # a fresh classic load of the stack
    held = store.current
    held.base_baked = held.baked_in_place = True
    asked.clear()
    loads = calls.load
    with in_place_fallback.permission(ALLOW):
        assert store.ensure(BASE, {}, STACK)[1] == "reuse"
    assert asked == [] and calls.load == loads and store.current is held
    # The same request with no permission: not reused, never hot-swapped (it is baked), the
    # fallback asked before the drop, then the classic load.
    _patcher, transition = store.ensure(BASE, {}, STACK)
    assert transition == "load" and asked == ["take"] and calls.load == loads + 1
    assert store.current is not held and store.current.baked_in_place is False


def test_a_slot_ensure_set_aside_is_not_handed_to_the_fallback(classic, monkeypatch):
    """An identity that drifted discards the resident before reuse; the fallback must not bake on it."""
    from dgx_monarch.actor import store_identity

    store, _calls, asked, _answer = classic
    monkeypatch.setattr(store_identity, "base_artifact_identity", lambda identity: object())  # never equal
    _patcher, transition = store.ensure(BASE, {}, STACK)
    assert transition == "load" and asked == []


def test_the_real_in_place_swap_marks_the_slot_and_a_swap_to_no_stack_clears_the_mark(rig, monkeypatch):
    store, _calls = rig  # lora_low_rss on: the default's own in-place bake
    store.ensure(BASE, {}, None)
    stored = store.current
    assert stored.baked_in_place is False
    monkeypatch.setattr(ms.ModelStore, "_lazy_swap", lambda self, stored_, active, unet_name: "RECORD")
    result = store_bake.lazy_swap_transition(
        store, "cond", BASE, STACK, stored.base_key, stored.artifact_identity,
        request_artifact_identity=lambda _name, _stack: stored.artifact_identity, logger=ms.log)
    assert result is not None and result[1] == "hot-swap"
    assert stored.baked_in_place is True and stored.base_baked is True and stored.unbake == "RECORD"
    store_bake.lazy_swap_transition(
        store, "cond", BASE, None, stored.base_key, stored.artifact_identity,
        request_artifact_identity=lambda _name, _stack: stored.artifact_identity, logger=ms.log)
    assert stored.baked_in_place is False and stored.base_baked is False
