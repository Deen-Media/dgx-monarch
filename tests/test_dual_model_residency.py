"""Keep both checkpoints resident through both dual-model prepares.

Each prepare registers the other checkpoint temporarily. Registrations are
staged to avoid cycles, and the partial-load guard checks every model in each
prepare.
"""
from __future__ import annotations

import math
import sys
import types

import pytest

from dgx_monarch.actor import partial_load_guard


class _Model:
    """The object comfy's ``is_clone`` compares by identity."""

    def __init__(self, name: str):
        self.name = name
        self.model_lowvram = False


class _Patcher:
    """Stand-in for ModelPatcher: additional models, options, and a clone."""

    def __init__(self, name: str = "m", model: _Model | None = None):
        self.model = model if model is not None else _Model(name)
        self.model_options: dict = {}
        self.additional: dict[str, list] = {}
        self.calls: list[tuple[str, str]] = []
        self.on_set = None
        self.name = name
        self.witness: list | None = None
        self.pair: tuple = ()

    def clone(self) -> _Patcher:
        """A comfy clone shares the underlying model, which is what is_clone reads."""
        return _Patcher(self.model.name, model=self.model)

    def set_additional_models(self, key, models):
        if self.on_set is not None:
            self.on_set(key)
        self.additional[key] = list(models)
        self._record("set", key)

    def remove_additional_models(self, key):
        self.additional.pop(key, None)
        self._record("remove", key)

    def get_additional_models(self) -> list:
        found: list = []
        for models in self.additional.values():
            found.extend(models)
        return found

    def get_additional_models_with_key(self, key) -> list:
        return self.additional.get(key, [])

    def _record(self, action: str, key: str) -> None:
        self.calls.append((action, key))
        if self.witness is not None:
            cond, uncond = self.pair
            self.witness.append(
                (self.name, action, key, tuple(cond.additional), tuple(uncond.additional)))


def _pair() -> tuple[_Patcher, _Patcher, list]:
    """A cond/uncond render clone pair that records every registration write."""
    cond, uncond = _Patcher("cond"), _Patcher("uncond")
    witness: list = []
    for patcher in (cond, uncond):
        patcher.witness = witness
        patcher.pair = (cond, uncond)
    return cond, uncond, witness


def _nested_additional(patcher) -> list:
    """ComfyUI's own gather, cycle guard included."""

    def evaluate(previous: list, cache: set) -> list:
        following: list = []
        for model in previous:
            for candidate in model.get_additional_models():
                if candidate not in cache:
                    following.append(candidate)
                    cache.add(candidate)
        if not following:
            return previous
        return previous + evaluate(following, cache)

    initial = patcher.get_additional_models()
    return evaluate(initial, set(initial))


class _Residency:
    """ComfyUI's load rule under DISABLE_SMART_MEMORY.

    ``load_models_gpu`` first drops every tracked entry that is a clone of a
    model in this call, then the free step runs with no keep-loaded set and,
    with smart memory off, unloads whatever is still tracked. So a checkpoint
    survives a prepare only when it is in that prepare's own model list.
    """

    def __init__(self, resident: list):
        self.tracked = list(resident)
        self.evicted: list[str] = []
        self.loads: list[tuple[str, ...]] = []

    def load_models_gpu(self, models: list) -> None:
        self.loads.append(tuple(patcher.model.name for patcher in models))
        for patcher in models:
            self.tracked = [t for t in self.tracked if t.model is not patcher.model]
        self.evicted.extend(tracked.model.name for tracked in self.tracked)
        self.tracked = list(models)


def _resident_base(patcher: _Patcher) -> _Patcher:
    """The store-loaded patcher comfy still tracks between renders."""
    return _Patcher(patcher.model.name, model=patcher.model)


def _install_fake_comfy(monkeypatch, outer_behavior):
    """Provide the two comfy modules the factory imports, with comfy's shapes."""

    class _WrappersMP:
        PREPARE_SAMPLING = "prepare_sampling"

    def add_wrapper_with_key(wrapper_type, key, wrapper, options, is_model_options=False):
        if is_model_options:
            options = options.setdefault("transformer_options", {})
        wrappers = options.setdefault("wrappers", {})
        wrappers.setdefault(wrapper_type, {}).setdefault(key, []).append(wrapper)

    def get_wrappers_with_key(wrapper_type, key, options, is_model_options=False):
        if is_model_options:
            options = options.get("transformer_options", {})
        return list(options.get("wrappers", {}).get(wrapper_type, {}).get(key, []))

    def get_all_wrappers(wrapper_type, options, is_model_options=False):
        if is_model_options:
            options = options.get("transformer_options", {})
        found: list = []
        for wrappers in options.get("wrappers", {}).get(wrapper_type, {}).values():
            found.extend(wrappers)
        return found

    extension = types.ModuleType("comfy.patcher_extension")
    extension.WrappersMP = _WrappersMP
    extension.add_wrapper_with_key = add_wrapper_with_key
    extension.get_wrappers_with_key = get_wrappers_with_key
    extension.get_all_wrappers = get_all_wrappers
    comfy = types.ModuleType("comfy")
    comfy.patcher_extension = extension
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", extension)

    class _FakeGuider:
        def __init__(self, model_patcher, uncond_model_patcher):
            self.model_patcher = model_patcher
            self.uncond_model_patcher = uncond_model_patcher
            self.cfg = 7.0

        def outer_sample(self, *args, **kwargs):
            return outer_behavior(self)

    module = types.ModuleType("comfy_extras.nodes_custom_sampler")
    module.Guider_DualModel = _FakeGuider
    parent = types.ModuleType("comfy_extras")
    parent.nodes_custom_sampler = module
    monkeypatch.setitem(sys.modules, "comfy_extras", parent)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_custom_sampler", module)
    return extension, _FakeGuider


def _prepare_sampling(extension, residency: _Residency, patcher: _Patcher):
    """Stock's prepare: gather nested additional models, load, hand them back."""

    def stock(model):
        models = _nested_additional(model)
        residency.load_models_gpu([model, *models])
        return (model.model, {}, models)

    wrappers = extension.get_all_wrappers(
        extension.WrappersMP.PREPARE_SAMPLING, patcher.model_options,
        is_model_options=True)

    def chain(index):
        def call(*args, **kwargs):
            if index >= len(wrappers):
                return stock(*args, **kwargs)
            return wrappers[index](chain(index + 1), *args, **kwargs)

        return call

    return chain(0)(patcher)


def _stock_body(extension, residency: _Residency):
    """Stock's two prepares in stock's order: uncond first, then cond."""

    def behavior(guider):
        if not math.isclose(guider.cfg, 1.0):
            _prepare_sampling(extension, residency, guider.uncond_model_patcher)
        _prepare_sampling(extension, residency, guider.model_patcher)
        return "sampled"

    return behavior


def _no_key_pair_coexists(witness: list) -> bool:
    return all(not (cond_keys and uncond_keys)
               for _who, _action, _key, cond_keys, uncond_keys in witness)


def test_the_bracket_stages_the_two_registrations_and_hands_off(monkeypatch):
    seen: list[tuple[tuple, tuple]] = []

    def outer(guider):
        # The uncond prepare's seam: only the uncond carries a registration.
        seen.append((tuple(guider.model_patcher.additional),
                     tuple(guider.uncond_model_patcher.additional)))
        swap, = extension.get_all_wrappers(
            extension.WrappersMP.PREPARE_SAMPLING,
            guider.uncond_model_patcher.model_options, is_model_options=True)
        # Comfy hands the wrapper the patcher being prepared, and the wrapper
        # reads what to swap off that patcher rather than off a closure.
        swap(lambda *a, **k: "loaded", guider.uncond_model_patcher)
        # The cond prepare's seam: the hand-off moved the registration over.
        seen.append((tuple(guider.model_patcher.additional),
                     tuple(guider.uncond_model_patcher.additional)))
        return "sampled"

    extension, _guider_cls = _install_fake_comfy(monkeypatch, outer)
    from dgx_monarch.actor.dual_model_guider import (
        _DUAL_COND_KEY,
        _DUAL_KEY,
        make_both_resident_guider,
    )

    cond, uncond, witness = _pair()
    guider = make_both_resident_guider(cond, uncond)
    assert guider.outer_sample() == "sampled"
    assert seen == [((), (_DUAL_COND_KEY,)), ((_DUAL_KEY,), ())]
    assert cond.additional == {} and uncond.additional == {}
    assert uncond.calls == [("set", _DUAL_COND_KEY), ("remove", _DUAL_COND_KEY),
                            ("remove", _DUAL_COND_KEY)]
    assert cond.calls == [("set", _DUAL_KEY), ("remove", _DUAL_KEY)]
    # ModelPatcher.clone() clones each additional model with no cycle guard, so
    # a cond and an uncond registered on each other would recurse without end.
    assert _no_key_pair_coexists(witness)


def test_the_handoff_wrapper_is_installed_once_under_its_own_key(monkeypatch):
    def outer(guider):
        return "sampled"

    extension, _guider_cls = _install_fake_comfy(monkeypatch, outer)
    from dgx_monarch.actor.dual_model_guider import _HANDOFF_KEY, make_both_resident_guider

    cond, uncond, _witness = _pair()
    guider = make_both_resident_guider(cond, uncond)
    guider.outer_sample()
    guider.outer_sample()
    installed = extension.get_wrappers_with_key(
        extension.WrappersMP.PREPARE_SAMPLING, _HANDOFF_KEY, uncond.model_options,
        is_model_options=True)
    assert len(installed) == 1, "comfy appends on a second add under one key"
    assert cond.model_options == {}, "the cond prepare must not fire the hand-off"


def test_a_second_guider_through_the_same_uncond_patcher_still_hands_off(monkeypatch):
    """Comfy has no remove-wrapper call, so the one wrapper must serve every render.

    Both call sites hand the guider a fresh per-render clone, so this never
    happens today. It is pinned because the wrapper outlives the guider that
    installed it: a wrapper that had closed over its own guider would fire
    for a dead render here, and the read-before-write install would leave the
    live render with no hand-off at all: its cond prepare would evict the
    uncond, the bf16 Ideogram4 device fault of 2026-08-21.
    """
    residency = _Residency([])

    def outer(guider):
        return _stock_body(extension, residency)(guider)

    extension, _guider_cls = _install_fake_comfy(monkeypatch, outer)
    from dgx_monarch.actor.dual_model_guider import (
        _DUAL_KEY,
        _HANDOFF_KEY,
        make_both_resident_guider,
    )

    first_cond, uncond, _witness = _pair()
    second_cond = _Patcher("cond2")
    assert make_both_resident_guider(first_cond, uncond).outer_sample() == "sampled"
    residency.loads.clear()
    assert make_both_resident_guider(second_cond, uncond).outer_sample() == "sampled"
    # The live render's cond prepare carries the uncond, exactly as the first
    # render's did, and the second render never wrote to the first's cond patcher.
    assert residency.loads == [("uncond", "cond2"), ("cond2", "uncond")]
    assert first_cond.calls == [("set", _DUAL_KEY), ("remove", _DUAL_KEY)]
    assert second_cond.calls == [("set", _DUAL_KEY), ("remove", _DUAL_KEY)]
    assert first_cond.additional == {} and second_cond.additional == {}
    installed = extension.get_wrappers_with_key(
        extension.WrappersMP.PREPARE_SAMPLING, _HANDOFF_KEY, uncond.model_options,
        is_model_options=True)
    assert len(installed) == 1


@pytest.mark.parametrize("handoff_first", [False, True])
def test_the_handoff_and_the_partial_load_guard_compose_either_way(
        monkeypatch, handoff_first):
    """Both wrappers sit on one seam, and the result does not depend on which is outer."""
    checked: list = []

    def fake_check(model, additional=()):
        checked.append((model, tuple(additional)))

    monkeypatch.setattr(partial_load_guard, "check", fake_check)
    extension, _guider_cls = _install_fake_comfy(monkeypatch, lambda guider: "sampled")
    from dgx_monarch.actor.dual_model_guider import (
        _DUAL_COND_KEY,
        _DUAL_KEY,
        _HANDOFF_KEY,
        handoff,
    )

    residency = _Residency([])
    cond, uncond, witness = _pair()
    uncond.set_additional_models(_DUAL_COND_KEY, [cond])
    installs = [
        (partial_load_guard.WRAPPER_KEY, partial_load_guard.make_prepare_sampling_guard()),
        (_HANDOFF_KEY, handoff),
    ]
    if handoff_first:
        installs.reverse()
    for key, wrapper in installs:
        extension.add_wrapper_with_key(
            extension.WrappersMP.PREPARE_SAMPLING, key, wrapper,
            uncond.model_options, is_model_options=True)

    result = _prepare_sampling(extension, residency, uncond)
    assert result == (uncond.model, {}, [cond]), "a wrapper altered the load result"
    assert residency.loads == [("uncond", "cond")]
    # The guard saw both checkpoints of this prepare, and the hand-off moved
    # the registration on for the cond prepare that follows.
    assert checked == [(uncond.model, (cond,))]
    assert uncond.additional == {} and cond.additional == {_DUAL_KEY: [uncond]}
    assert _no_key_pair_coexists(witness)


def test_neither_prepare_evicts_the_other_checkpoint(monkeypatch):
    residency = _Residency([])

    def outer(guider):
        return _stock_body(extension, residency)(guider)

    extension, _guider_cls = _install_fake_comfy(monkeypatch, outer)
    from dgx_monarch.actor.dual_model_guider import make_both_resident_guider

    cond, uncond, witness = _pair()
    # A warm render starts with both checkpoints tracked from the last one.
    residency.tracked = [_resident_base(cond), _resident_base(uncond)]
    guider = make_both_resident_guider(cond, uncond)
    assert guider.outer_sample() == "sampled"
    assert residency.loads == [("uncond", "cond"), ("cond", "uncond")]
    assert residency.evicted == [], (
        "a prepare unloaded the other checkpoint: that is the 74.0 s of a "
        "108.5 s warm Ideogram4 render issue #314 removes")
    assert _no_key_pair_coexists(witness)


def test_the_one_way_bracket_still_loses_the_cond_to_the_uncond_prepare(monkeypatch):
    """The one-way bracket (only the cond carries the uncond), kept as the regression a revert would bring back."""
    residency = _Residency([])

    def outer(guider):
        return _stock_body(extension, residency)(guider)

    extension, guider_cls = _install_fake_comfy(monkeypatch, outer)
    from dgx_monarch.actor.dual_model_guider import _DUAL_KEY

    cond, uncond, _witness = _pair()
    residency.tracked = [_resident_base(cond), _resident_base(uncond)]

    class _OneWay(guider_cls):
        def outer_sample(self, *args, **kwargs):
            self.model_patcher.set_additional_models(_DUAL_KEY, [self.uncond_model_patcher])
            try:
                return super().outer_sample(*args, **kwargs)
            finally:
                self.model_patcher.remove_additional_models(_DUAL_KEY)

    assert _OneWay(cond, uncond).outer_sample() == "sampled"
    assert residency.loads == [("uncond",), ("cond", "uncond")]
    assert residency.evicted == ["cond"]


def test_cfg_one_keeps_the_one_way_shape_and_installs_no_handoff(monkeypatch):
    residency = _Residency([])

    def outer(guider):
        return _stock_body(extension, residency)(guider)

    extension, _guider_cls = _install_fake_comfy(monkeypatch, outer)
    from dgx_monarch.actor.dual_model_guider import (
        _DUAL_COND_KEY,
        _DUAL_KEY,
        make_both_resident_guider,
    )

    cond, uncond, witness = _pair()
    residency.tracked = [_resident_base(cond), _resident_base(uncond)]
    guider = make_both_resident_guider(cond, uncond)
    guider.cfg = 1.0
    assert guider.outer_sample() == "sampled"
    # Stock skips the uncond prepare at cfg 1.0, so there is one prepare and
    # one direction to cover: the cond carries the uncond.
    assert residency.loads == [("cond", "uncond")]
    assert residency.evicted == []
    assert extension.get_all_wrappers(
        extension.WrappersMP.PREPARE_SAMPLING, uncond.model_options,
        is_model_options=True) == []
    assert cond.calls == [("set", _DUAL_KEY), ("remove", _DUAL_KEY)]
    assert uncond.calls == [("remove", _DUAL_COND_KEY)]
    assert cond.additional == {} and uncond.additional == {}
    assert _no_key_pair_coexists(witness)


def _boom(*_args, **_kwargs):
    raise RuntimeError("staged failure")


def test_both_registrations_are_removed_when_the_sample_raises(monkeypatch):
    def outer(guider):
        raise RuntimeError("mid-sample failure")

    _extension, _guider_cls = _install_fake_comfy(monkeypatch, outer)
    from dgx_monarch.actor.dual_model_guider import make_both_resident_guider

    cond, uncond, witness = _pair()
    guider = make_both_resident_guider(cond, uncond)
    with pytest.raises(RuntimeError, match="mid-sample failure"):
        guider.outer_sample()
    assert cond.additional == {} and uncond.additional == {}
    assert _no_key_pair_coexists(witness)


def test_a_raise_while_staging_leaves_no_registration_behind(monkeypatch):
    def outer(guider):
        return "sampled"

    _extension, _guider_cls = _install_fake_comfy(monkeypatch, outer)
    from dgx_monarch.actor.dual_model_guider import make_both_resident_guider

    cond, uncond, witness = _pair()
    uncond.on_set = _boom
    guider = make_both_resident_guider(cond, uncond)
    with pytest.raises(RuntimeError, match="staged failure"):
        guider.outer_sample()
    assert cond.additional == {} and uncond.additional == {}
    assert _no_key_pair_coexists(witness)


def test_a_raise_installing_the_handoff_still_drops_the_first_key(monkeypatch):
    def outer(guider):
        return "sampled"

    extension, _guider_cls = _install_fake_comfy(monkeypatch, outer)
    monkeypatch.setattr(extension, "add_wrapper_with_key", _boom)
    from dgx_monarch.actor.dual_model_guider import make_both_resident_guider

    cond, uncond, witness = _pair()
    guider = make_both_resident_guider(cond, uncond)
    with pytest.raises(RuntimeError, match="staged failure"):
        guider.outer_sample()
    assert cond.additional == {} and uncond.additional == {}
    assert _no_key_pair_coexists(witness)


def test_a_failed_uncond_load_leaves_no_registration_behind(monkeypatch):
    def outer(guider):
        wrapper, = extension.get_all_wrappers(
            extension.WrappersMP.PREPARE_SAMPLING,
            guider.uncond_model_patcher.model_options, is_model_options=True)
        return wrapper(_boom, guider.uncond_model_patcher)

    extension, _guider_cls = _install_fake_comfy(monkeypatch, outer)
    from dgx_monarch.actor.dual_model_guider import _DUAL_KEY, make_both_resident_guider

    cond, uncond, witness = _pair()
    guider = make_both_resident_guider(cond, uncond)
    with pytest.raises(RuntimeError, match="staged failure"):
        guider.outer_sample()
    assert cond.additional == {} and uncond.additional == {}
    assert cond.calls == [("remove", _DUAL_KEY)]
    assert _no_key_pair_coexists(witness)


class _RealModel:
    def __init__(self, lowvram: bool):
        self.model_lowvram = lowvram


class _LoadedPatcher:
    def __init__(self, lowvram: bool):
        self.model = _RealModel(lowvram)


def test_guard_reads_additional_models_too(monkeypatch):
    monkeypatch.setattr(partial_load_guard, "_distributed_world", lambda: 2)
    flags = []

    def fake_all_full(local_full):
        flags.append(local_full)
        return local_full

    monkeypatch.setattr(partial_load_guard, "_all_ranks_full", fake_all_full)
    partial_load_guard.check(
        _RealModel(False), additional=(_LoadedPatcher(False),))
    assert flags == [True]
    with pytest.raises(partial_load_guard.PartialLoadDivergenceError):
        partial_load_guard.check(
            _RealModel(False), additional=(_LoadedPatcher(True),))
    assert flags == [True, False]


def test_prepare_wrapper_forwards_the_additional_list(monkeypatch):
    captured = {}

    def fake_check(model, additional=()):
        captured["model"] = model
        captured["additional"] = additional

    monkeypatch.setattr(partial_load_guard, "check", fake_check)
    wrapper = partial_load_guard.make_prepare_sampling_guard()
    real, extras = _RealModel(False), [_LoadedPatcher(False)]

    def executor(*args, **kwargs):
        return (real, {"positive": []}, extras)

    assert wrapper(executor) == (real, {"positive": []}, extras)
    assert captured["model"] is real
    assert captured["additional"] == tuple(extras)
