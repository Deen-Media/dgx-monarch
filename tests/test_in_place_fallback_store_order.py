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


class _Tensor:
    """Stands for a LoRA file's tensor: weak-referenceable, so a test can see whether it is still alive."""

    nbytes = 1 << 20


@pytest.fixture
def stack(classic, monkeypatch):
    """Every build of the stack reads fresh tensors, as ``_build_active`` reads the files; the builds are recorded."""
    import weakref

    builds = types.SimpleNamespace(tensors=[], alive_at_each_build=[])

    def build(self, base, stack_):
        builds.alive_at_each_build.append([ref() is not None for ref in builds.tensors])
        tensor = _Tensor()
        builds.tensors.append(weakref.ref(tensor))
        return types.SimpleNamespace(tag="ACTIVE", model=base.model,
                                     patches={key: [(1.0, tensor)] for key in ("w1", "w2")} if stack_ else {})

    monkeypatch.setattr(ms.ModelStore, "_build_active", build)
    return builds


def test_the_declined_swaps_stack_is_held_through_both_walls_and_let_go_before_the_files_are_read_again(
        classic, stack, monkeypatch):
    """Held through: the walls charge none of the files, by what the store holds and not by timing."""
    from dgx_monarch.actor import store_residency

    store, _calls, _asked, _answer = classic
    seen: dict = {}
    real_resolve, real_check = store_residency.resolve, store_residency.preload_capacity_check
    real_drop = store._drop

    def take(store_, slot, unet_name, options, lora_stack, *_rest):
        seen["at take"] = in_place_fallback.stack_on_host(store_, slot, lora_stack)
        return None

    def drop(slot, *args, **kwargs):  # the fresh load drops the old slot and collects before either wall
        real_drop(slot, *args, **kwargs)
        seen.setdefault("after the drop", stack.tensors[0]() is not None)

    def resolve(*args, **kwargs):
        seen["at the first wall"] = (in_place_fallback.stack_on_host(store, "cond", STACK),
                                     stack.tensors[0]() is not None)
        return real_resolve(*args, **kwargs)

    def check(*args, **kwargs):
        seen["at the second wall"] = (in_place_fallback.stack_on_host(store, "cond", STACK),
                                      stack.tensors[0]() is not None)
        return real_check(*args, **kwargs)

    monkeypatch.setattr(in_place_fallback, "take", take)
    monkeypatch.setattr(store, "_drop", drop)
    monkeypatch.setattr(store_residency, "resolve", resolve)
    monkeypatch.setattr(store_residency, "preload_capacity_check", check)
    _patcher, transition = store.ensure(BASE, {}, STACK)
    assert transition == "load"
    assert seen == {"at take": True, "after the drop": True,
                    "at the first wall": (True, True), "at the second wall": (True, True)}
    # Two builds: the swap's and the fresh load's. The swap's tensors were gone before the second read.
    assert stack.alive_at_each_build == [[], [False]]
    assert not in_place_fallback.stack_on_host(store, "cond", STACK)
    # The same stack into a slot the store holds nothing in: no swap ran, so nothing was read first.
    monkeypatch.setattr(store, "_drop", real_drop)
    store._drop("cond")
    seen.clear()
    monkeypatch.setattr(store_residency, "resolve", lambda *a, **k: (
        seen.__setitem__("at the first wall", in_place_fallback.stack_on_host(store, "cond", STACK)),
        real_resolve(*a, **k))[1])
    store.ensure(BASE, {}, STACK)
    assert seen == {"at take": False, "at the first wall": False, "at the second wall": (False, False)}


def test_a_stack_released_before_the_walls_read_is_charged_its_files(classic, stack, monkeypatch):
    """Released before the check: nothing holds the files' bytes, so the walls are told they are still to read."""
    from dgx_monarch.actor import store_residency

    store, _calls, _asked, _answer = classic
    seen: dict = {}
    real_resolve = store_residency.resolve

    def take(store_, slot, unet_name, options, lora_stack, *_rest):
        in_place_fallback.forget_stack_read(store_)  # as the in-place path does before it drops or loads
        return None

    def resolve(*args, **kwargs):
        seen["at the first wall"] = (in_place_fallback.stack_on_host(store, "cond", STACK),
                                     stack.tensors[0]() is not None)
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(in_place_fallback, "take", take)
    monkeypatch.setattr(store_residency, "resolve", resolve)
    store.ensure(BASE, {}, STACK)
    assert seen == {"at the first wall": (False, False)}


def test_a_hot_swap_that_stays_in_place_holds_nothing(classic):
    store, _calls, _asked, _answer = classic
    store.current.base_patcher.backup = {"w1": object(), "w2": object()}  # ComfyUI's backup covers the stack
    _patcher, transition = store.ensure(BASE, {}, STACK)
    assert transition == "hot-swap" and not in_place_fallback.stack_on_host(store, "cond", STACK)
    assert getattr(store, in_place_fallback._STACK_READ, None) is None


def test_a_fresh_load_that_raised_holds_nothing_for_the_requeue(classic, stack, monkeypatch):
    """The held stack never outlives its ``ensure`` call: a requeue of the same stack is charged its files.

    The swap declines and the store holds its stack; the fresh load raises
    before its walls price, with the slot emptied. ``ensure`` lets the stack
    go on the way out, so its bytes are back in the box before any quote or
    load reads it again, and the same request finds an empty slot, runs no
    swap and reads nothing first.
    """
    from dgx_monarch.actor import store_load

    store, _calls, _asked, _answer = classic
    seen: dict = {}
    real_probe = in_place_fallback.fresh_load_probe

    def failing_load(store_, slot, *_args, **_kwargs):
        seen["held when the load failed"] = in_place_fallback.stack_on_host(store_, slot, STACK)
        store_._drop(slot)
        raise RuntimeError("the load failed before its walls priced")

    with monkeypatch.context() as patch:
        patch.setattr(store_load, "fresh_load", failing_load)
        with pytest.raises(RuntimeError, match="before its walls priced"):
            store.ensure(BASE, {}, STACK)
    assert store.current is None and seen == {"held when the load failed": True}
    assert not in_place_fallback.stack_on_host(store, "cond", STACK)  # let go on the way out
    assert getattr(store, in_place_fallback._STACK_READ, None) is None
    import gc

    gc.collect()
    assert stack.tensors[0]() is None  # and its tensors with it

    def take(store_, slot, unet_name, options, lora_stack, *_rest):
        seen["at take"] = in_place_fallback.stack_on_host(store_, slot, lora_stack)
        return None

    def probe(store_, slot, lora_stack):
        seen["at the walls"] = in_place_fallback.stack_on_host(store_, slot, lora_stack)
        return real_probe(store_, slot, lora_stack)

    monkeypatch.setattr(in_place_fallback, "take", take)
    monkeypatch.setattr(in_place_fallback, "fresh_load_probe", probe)
    _patcher, transition = store.ensure(BASE, {}, STACK)  # the requeue
    assert transition == "load" and seen["at take"] is False and seen["at the walls"] is False


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
