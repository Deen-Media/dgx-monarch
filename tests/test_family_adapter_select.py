"""The Init node's `family_adapter` override: auto is untouched, forcing is bounded.

The override lets a finetune run under a family its checkpoint header cannot
prove. Two properties are pinned here:

1. `auto` emits no worker-args key, so the mesh spec, the gate capability
   context and the adapter decision equal those of a graph without the
   widget. The forced-path tests measure against that baseline.
2. Forcing is a claim, not evidence. It enters the capability context, so a
   forced run never shares a ledger row with an auto run of the same file; it
   never inherits family-level slab vouching; it never writes the family
   memo; and a family the loaded model is not an instance of refuses with a
   typed class P message before any kernel runs. So does a model its family's
   adapter refuses under `auto`: a forced family binds what `auto` binds, plus
   a derivative of a vetted class (docs/TROUBLESHOOTING.md #153).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from dgx_monarch import family_select
from dgx_monarch.adapters import SELECTABLE_FAMILIES


class _Alpha:
    """Stands in for one Comfy model class."""


class _AlphaChild(_Alpha):
    """The derivative subclass shape a finetune produces."""


class _Beta:
    """An unrelated architecture."""


class _Parent:
    """Stands in for Flux, Omnigen2, or WAN21."""


class _Child(_Parent):
    """Stands in for Chroma, Boogu, or SCAIL: a subclass with its own adapter
    and its own forward, registered ahead of its parent."""


class _StubAdapter:
    def __init__(self, family, broad=(), exact=()):
        self.family = family
        self.model_base_classes = broad
        self.exact_model_base_classes = exact
        self.injected = []

    def matches(self, base_model):
        """As the shipped adapters: a broad adapter takes every subclass, an
        exact-type one raises on any type but those it names. The forced path
        asks this about the model's nearest ComfyUI class."""
        import comfy.model_base as model_base

        from dgx_monarch.adapters import UnsupportedModelError

        names = (*self.model_base_classes, *self.exact_model_base_classes)
        owned = [getattr(model_base, name, None) for name in names]
        if not any(cls is not None and isinstance(base_model, cls) for cls in owned):
            return False
        if not self.exact_model_base_classes or any(
                type(base_model) is getattr(model_base, name, None)
                for name in self.exact_model_base_classes):
            return True
        raise UnsupportedModelError(
            f"{self.family} variant {type(base_model).__name__} is not vetted")

    def inject_usp(self, diffusion_model, ctx):
        self.injected.append(diffusion_model)


@pytest.fixture
def stub_registry(monkeypatch):
    """Register four families against a stub `comfy.model_base`."""
    import sys
    import types as pytypes

    from dgx_monarch import adapters as adapters_mod

    model_base = pytypes.ModuleType("comfy.model_base")
    model_base.Alpha = _Alpha
    model_base.Beta = _Beta
    model_base.Parent = _Parent
    model_base.Child = _Child
    comfy = sys.modules.get("comfy") or pytypes.ModuleType("comfy")
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_base", model_base)
    monkeypatch.setattr(comfy, "model_base", model_base, raising=False)

    alpha = _StubAdapter("alpha", exact=("Alpha",))
    beta = _StubAdapter("beta", broad=("Beta",))
    # Most specific first, the registry's own ordering rule.
    child = _StubAdapter("child", exact=("Child",))
    parent = _StubAdapter("parent", broad=("Parent",))
    monkeypatch.setattr(adapters_mod, "ADAPTERS", (alpha, beta, child, parent))
    monkeypatch.setattr(adapters_mod, "SELECTABLE_FAMILIES",
                        ("alpha", "beta", "child", "parent"))
    return SimpleNamespace(alpha=alpha, beta=beta, child=child, parent=parent)


def test_the_choice_list_is_derived_from_the_adapter_registry():
    """Hand-typing the list would silently omit a newly registered family."""
    from dgx_monarch.adapters import ADAPTERS
    from dgx_monarch.nodes.init import DGXMonarchInit

    values, meta = DGXMonarchInit.INPUT_TYPES()["optional"]["family_adapter"]
    assert values[0] == "auto"
    assert meta["default"] == "auto"
    assert meta["advanced"] is True
    assert tuple(values[1:]) == tuple(sorted({a.family for a in ADAPTERS}))
    # Fewer names than adapters: SCAIL2 and SCAIL both answer "wan_scail".
    assert len(values[1:]) < len(ADAPTERS)


def test_the_tooltip_states_what_the_override_cannot_fix():
    """The limits belong where the operator reads them, not only in the docs."""
    from dgx_monarch.nodes.init import DGXMonarchInit

    _values, meta = DGXMonarchInit.INPUT_TYPES()["optional"]["family_adapter"]
    tooltip = meta["tooltip"]
    for promise in ("cannot", "quantization", "slab_weights=on", "docs/MODELS.md"):
        assert promise in tooltip


def test_the_widget_is_appended_last_so_saved_workflows_keep_their_values():
    from dgx_monarch.nodes.init import DGXMonarchInit

    order = [*DGXMonarchInit.INPUT_TYPES()["required"],
             *DGXMonarchInit.INPUT_TYPES()["optional"]]
    assert order[-1] == "family_adapter"


def test_cluster_config_refuses_the_key_because_it_is_not_fleet_policy():
    """Which family a checkpoint is belongs to the graph that names the
    checkpoint. A cluster.toml default would force every file to one family."""
    from dgx_monarch import config_schema

    with pytest.raises(config_schema.ClusterConfigError) as excinfo:
        config_schema.validate_worker_args({"family_override": "krea2"})
    assert "family_override" in str(excinfo.value)


def _init_worker_args(monkeypatch, **kwargs):
    from dgx_monarch import first_render
    from dgx_monarch.nodes import init as init_mod

    handle = SimpleNamespace(world=1, config=None, config_fingerprint="local")
    monkeypatch.setattr(init_mod, "get_mesh", lambda **_kw: handle)
    monkeypatch.setattr(first_render, "nccl_deferred", lambda: None)
    (spec,) = init_mod.DGXMonarchInit().init(
        topology="auto", mode="local", **kwargs)
    return spec.worker_args


def test_auto_produces_the_worker_args_of_a_graph_without_this_widget(monkeypatch):
    """worker_args goes verbatim into every gate capability context, so a key
    written on auto too would change that context, and no PASS row already
    recorded would match it."""
    without = _init_worker_args(monkeypatch)
    explicit_auto = _init_worker_args(monkeypatch, family_adapter="auto")
    assert without == explicit_auto
    assert family_select.WORKER_ARG_KEY not in without


def test_auto_and_forced_runs_canonicalize_to_different_contexts(monkeypatch):
    """A forced run never reads an auto run's PASS, and an auto run never reads
    a forced one's."""
    from dgx_monarch import gate_ledger

    auto = _init_worker_args(monkeypatch, family_adapter="auto")
    forced = _init_worker_args(monkeypatch, family_adapter="krea2")
    assert forced[family_select.WORKER_ARG_KEY] == "krea2"
    assert auto != forced
    canonical_auto = gate_ledger._canonical_capability_context({"worker_args": auto})
    canonical_forced = gate_ledger._canonical_capability_context({"worker_args": forced})
    assert canonical_auto != canonical_forced


def test_forcing_the_family_detection_already_reads_changes_only_the_context(
        monkeypatch):
    """Forcing the family the sniff would have chosen anyway leaves the render
    policy alone: same topology, same sage choice. Only the context changes,
    as it must."""
    auto = _init_worker_args(monkeypatch, family_adapter="auto")
    forced = _init_worker_args(monkeypatch, family_adapter="krea2")
    assert {k: v for k, v in forced.items()
            if k != family_select.WORKER_ARG_KEY} == auto

    from dgx_monarch.topology import choose_auto_topology

    detected = choose_auto_topology("krea2", "fp8", 2.25, 2, cfg_value=1.0)
    override = choose_auto_topology(
        family_select.effective_family("krea2", "krea2"), "fp8", 2.25, 2,
        cfg_value=1.0)
    assert detected.topology == override.topology
    assert detected.sage == override.sage


def test_an_unreadable_family_still_routes_by_the_forced_value():
    """A header the signature table cannot read loses its table row, and
    naming the family restores it."""
    from dgx_monarch.topology import choose_auto_topology

    unreadable = choose_auto_topology("unknown", "fp8", 2.25, 2, cfg_value=1.0)
    forced = choose_auto_topology(
        family_select.effective_family("unknown", "krea2"), "fp8", 2.25, 2,
        cfg_value=1.0)
    assert "no krea2 table row" not in forced.reason
    assert forced.reason != unreadable.reason


def test_the_init_node_refuses_a_family_no_adapter_answers(monkeypatch):
    with pytest.raises(ValueError, match="names no dgx-monarch adapter"):
        _init_worker_args(monkeypatch, family_adapter="not-a-family")


@pytest.mark.parametrize("value", [None, {}, {"family_override": "auto"},
                                   {"family_override": ""},
                                   {"family_override": 7}, "not-a-dict"])
def test_absent_blank_and_auto_all_read_as_no_override(value):
    assert family_select.override_from_worker_args(value) is None


def test_a_named_family_reads_back_stripped():
    assert family_select.override_from_worker_args(
        {"family_override": " krea2 "}) == "krea2"


def test_a_worker_without_a_store_runs_the_auto_path():
    """Injection is exercised by probes and unit tests that stand a bare
    namespace in for the worker."""
    assert family_select.override_for_worker(SimpleNamespace()) is None
    assert family_select.override_for_worker(
        SimpleNamespace(store=SimpleNamespace(family_override="ltx"))) == "ltx"


def test_a_forced_family_binds_the_derivative_subclass_detection_refuses(
        stub_registry):
    """A subclass ComfyUI does not define, built on a class this family's
    adapter takes: detection's exact-type adapters raise on it, and the forced
    family binds it through the class it extends."""
    from dgx_monarch.adapters import adapter_for

    assert adapter_for(_AlphaChild(), "alpha") is stub_registry.alpha
    assert adapter_for(_Alpha(), "alpha") is stub_registry.alpha


def test_forcing_a_parent_family_over_a_child_family_refuses(stub_registry):
    """Chroma, Flux2 and LongCatImage subclass Flux; Boogu subclasses Omnigen2;
    the WAN2x variants subclass WAN21. A bare isinstance against the named
    family's classes would accept a child model and bind the parent adapter
    over a divergent forward. The owner is the first ancestry match across the
    whole registry, which is ordered most specific first."""
    from dgx_monarch.adapters import UnsupportedModelError, adapter_for

    with pytest.raises(UnsupportedModelError) as excinfo:
        adapter_for(_Child(), "parent")
    message = str(excinfo.value)
    assert "'parent'" in message and "_Child" in message and "child" in message
    assert not stub_registry.parent.injected


def test_a_child_model_still_binds_its_own_family(stub_registry):
    """The refusal above must not fire when a model is forced to its own family."""
    from dgx_monarch.adapters import adapter_for

    assert adapter_for(_Child(), "child") is stub_registry.child
    assert adapter_for(_Parent(), "parent") is stub_registry.parent


def test_a_derivative_of_a_child_binds_the_child_not_the_parent(stub_registry):
    """A finetune of a child family: ComfyUI builds a subclass of the child,
    and the owner is still the child adapter, not the parent."""
    from dgx_monarch.adapters import adapter_for

    grandchild = type("ChildFinetune", (_Child,), {})()
    assert adapter_for(grandchild, "child") is stub_registry.child


def test_the_real_registry_orders_every_known_parent_after_its_children():
    """The ordering the ancestry rule depends on, asserted against the shipped
    registry rather than the stub: a child family that subclasses a parent
    family must be registered ahead of it."""
    from dgx_monarch.adapters import ADAPTERS

    order = {adapter.family: index for index, adapter in enumerate(ADAPTERS)}
    for child, parent in (("chroma", "flux"), ("flux2", "flux"),
                          ("longcat", "flux"), ("boogu", "omnigen2"),
                          ("wan_scail", "wan"), ("wan_dancer", "wan")):
        assert order[child] < order[parent], (
            f"{child} subclasses {parent} in comfy.model_base and must be "
            "registered first, or a forced parent family would bind over it")


def test_a_forced_family_the_model_is_not_refuses_naming_both(stub_registry):
    """The claim is checked against the module tree ComfyUI built, so a wrong
    family refuses instead of ending in a CUDA assert or a mid-forward
    AttributeError."""
    from dgx_monarch.adapters import UnsupportedModelError, adapter_for

    with pytest.raises(UnsupportedModelError) as excinfo:
        adapter_for(_Beta(), "alpha")
    message = str(excinfo.value)
    assert "'alpha'" in message and "_Beta" in message
    assert not stub_registry.alpha.injected and not stub_registry.beta.injected


def test_the_bind_refusal_is_tagged_class_p_so_the_lease_retires_consumed(
        stub_registry):
    """The refusal is rank-symmetric: both ranks load the same bytes under the
    same fleet policy and refuse identically, so the tag lets the user fix the
    widget and re-queue on the same fleet instead of paying a recycle."""
    from dgx_monarch.adapters import UnsupportedModelError, adapter_for
    from dgx_monarch.refusal import RefusalClass, parse_refusal_tag

    with pytest.raises(UnsupportedModelError) as excinfo:
        adapter_for(_Beta(), "alpha")
    tag = parse_refusal_tag(str(excinfo.value))
    assert tag is not None
    assert tag.refusal_class is RefusalClass.PHYSICS
    assert tag.guard is None and tag.waivable is False


def test_an_unknown_family_string_refuses_before_it_ever_touches_comfy():
    """A name this build has never heard of is a graph error, not a model
    question, so it is answered without importing comfy.model_base."""
    from dgx_monarch.adapters import UnsupportedModelError, adapter_for

    with pytest.raises(UnsupportedModelError) as excinfo:
        adapter_for(object(), "definitely-not-a-family")
    message = str(excinfo.value)
    for family in SELECTABLE_FAMILIES:
        assert family in message


def test_auto_and_none_both_take_the_detection_path(monkeypatch):
    """`adapter_for` must not become a second decision for unforced graphs."""
    from dgx_monarch import adapters as adapters_mod

    sentinel = object()
    calls = []
    monkeypatch.setattr(adapters_mod, "get_adapter",
                        lambda model: calls.append(model) or sentinel)
    assert adapters_mod.adapter_for(sentinel) is sentinel
    assert adapters_mod.adapter_for(sentinel, None) is sentinel
    assert adapters_mod.adapter_for(sentinel, "auto") is sentinel
    assert len(calls) == 3


def test_the_registry_order_resolves_a_family_two_adapters_answer():
    """`wan_scail` names SCAIL2 and SCAIL; detection tries SCAIL2 first and the
    forced path must not disagree with it."""
    from dgx_monarch.adapters import ADAPTERS

    scail = [a for a in ADAPTERS if a.family == "wan_scail"]
    assert len(scail) == 2
    assert type(scail[0]).__name__ == "SCAIL2Adapter"


def test_a_forced_family_refuses_a_comfy_subclass_its_adapter_refuses_under_auto(
        stub_registry, monkeypatch):
    """The rule: a forced family binds a model exactly when its adapter takes
    the model's nearest ComfyUI class under `auto`. A subclass ComfyUI defines
    is its own nearest class, and an exact-type adapter refuses it: another
    upstream model, whose forward the adapter does not re-express. So does a
    derivative of that subclass. A derivative of a vetted class still binds,
    which is what the override is for."""
    import comfy.model_base as model_base

    from dgx_monarch.adapters import UnsupportedModelError, adapter_for, get_adapter
    from dgx_monarch.refusal import RefusalClass, parse_refusal_tag

    image = type("AlphaImage", (_Alpha,), {})
    monkeypatch.setattr(model_base, "AlphaImage", image, raising=False)
    for model, built in ((image(), "a AlphaImage, and"),
                         (type("AlphaImageFinetune", (image,), {})(),
                          "a AlphaImageFinetune, a derivative of AlphaImage, and")):
        with pytest.raises(UnsupportedModelError):
            get_adapter(model)  # auto refuses it
        with pytest.raises(UnsupportedModelError) as excinfo:
            adapter_for(model, "alpha")
        message = str(excinfo.value)
        assert "family_adapter='alpha' does not fit this checkpoint" in message
        assert f"ComfyUI built {built}" in message
        assert "does not take this AlphaImage model under 'auto' either" in message
        assert "alpha variant AlphaImage is not vetted" in message  # the adapter's own answer
        assert message.endswith("See docs/TROUBLESHOOTING.md #153.")
        tag = parse_refusal_tag(message)
        assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
        assert tag.guard is None and tag.waivable is False
    assert not stub_registry.alpha.injected
    # A derivative of the vetted class: auto's exact-type test raises on it, the force binds it.
    with pytest.raises(UnsupportedModelError):
        get_adapter(_AlphaChild())
    assert adapter_for(_AlphaChild(), "alpha") is stub_registry.alpha
    assert adapter_for(_Alpha(), "alpha") is stub_registry.alpha
    assert adapter_for(_Child(), "child") is stub_registry.child


def test_the_forced_path_honours_an_adapters_instance_test_through_the_view(
        stub_registry, monkeypatch):
    """Z-Image's adapter takes its named class only when the diffusion model
    carries learned pad tokens, because Lumina 2.0 builds the same class. The
    forced path asks the same test, on the model or, for a derivative, on a
    view of it as its ComfyUI class that shares its attributes."""
    from dgx_monarch.adapters import UnsupportedModelError, adapter_for

    def only_with_pads(base_model):
        return type(base_model) is _Beta and bool(getattr(base_model, "pads", None))

    monkeypatch.setattr(stub_registry.beta, "matches", only_with_pads)
    padded, bare = _Beta(), _Beta()
    padded.pads = 32
    assert adapter_for(padded, "beta") is stub_registry.beta
    with pytest.raises(UnsupportedModelError, match="does not take this _Beta model"):
        adapter_for(bare, "beta")
    finetune = type("BetaFinetune", (_Beta,), {})
    padded_finetune, bare_finetune = finetune(), finetune()
    padded_finetune.pads = 32
    assert adapter_for(padded_finetune, "beta") is stub_registry.beta
    assert type(padded_finetune) is finetune and padded_finetune.pads == 32  # the model is untouched
    with pytest.raises(UnsupportedModelError, match="a derivative of _Beta"):
        adapter_for(bare_finetune, "beta")


def test_a_forced_minimax_h3_does_not_bind_the_image_model(monkeypatch):
    """ComfyUI 0.45.0's MiniMaxH3Image subclasses MiniMaxH3 and is another
    model: no audio stream, its own output heads. A forced `minimax_h3` must
    refuse it class P, naming the class. This holds whichever check answers:
    the named entry today, or the owner check once an adapter for the image
    model is registered ahead of H3's. H3 itself and a derivative ComfyUI
    does not define still bind."""
    from dgx_monarch.adapters import ADAPTERS, UnsupportedModelError, adapter_for
    from dgx_monarch.refusal import RefusalClass, parse_refusal_tag
    from fake_model_base_helpers import install_fake_model_base

    model_base = install_fake_model_base(
        monkeypatch, {"MiniMaxH3": None, "MiniMaxH3Image": "MiniMaxH3"})
    with pytest.raises(UnsupportedModelError) as excinfo:
        adapter_for(model_base.MiniMaxH3Image(), "minimax_h3")
    message = str(excinfo.value)
    assert "'minimax_h3'" in message and "MiniMaxH3Image" in message
    tag = parse_refusal_tag(message)
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert tag.guard is None and tag.waivable is False
    video = next(adapter for adapter in ADAPTERS if adapter.family == "minimax_h3")
    assert adapter_for(model_base.MiniMaxH3(), "minimax_h3") is video
    finetune = type("MiniMaxH3Finetune", (model_base.MiniMaxH3,), {})
    assert adapter_for(finetune(), "minimax_h3") is video
    # The image model's own adapter registers ahead of H3's: the owner check answers,
    # naming it, and a forced minimax_h3_image binds it.
    assert "which the minimax_h3_image adapter owns" in message
    image = next(adapter for adapter in ADAPTERS if adapter.family == "minimax_h3_image")
    assert adapter_for(model_base.MiniMaxH3Image(), "minimax_h3_image") is image


def test_a_forced_minimax_h3_image_binds_its_class_and_refuses_a_comfy_subclass(monkeypatch):
    """The image adapter names its own model class. A forced ``minimax_h3_image``
    binds that class and a class a custom node derives from it. A subclass
    ComfyUI itself defines is another model: `auto` refuses it, and so does the
    force (docs/TROUBLESHOOTING.md #153). Neither family takes the other's model."""
    from dgx_monarch.adapters import UnsupportedModelError, adapter_for, get_adapter
    from fake_model_base_helpers import install_fake_model_base

    mb = install_fake_model_base(monkeypatch, {
        "MiniMaxH3": None, "MiniMaxH3Image": "MiniMaxH3",
        "MiniMaxH3ImageVariant": "MiniMaxH3Image"})
    image = mb.MiniMaxH3Image()
    assert _bound(image, "minimax_h3_image") == "MiniMaxH3ImageAdapter"
    assert type(get_adapter(image)).__name__ == "MiniMaxH3ImageAdapter"
    finetune = type("H3ImageFinetune", (mb.MiniMaxH3Image,), {})()
    assert _bound(finetune, "minimax_h3_image") == "MiniMaxH3ImageAdapter"
    message = _forced_refusal(mb.MiniMaxH3ImageVariant(), "minimax_h3_image")
    assert "ComfyUI built a MiniMaxH3ImageVariant, and" in message
    assert "minimax_h3_image variant MiniMaxH3ImageVariant is a MiniMaxH3Image subclass" in message
    derived = type("VariantFinetune", (mb.MiniMaxH3ImageVariant,), {})()
    assert "a VariantFinetune, a derivative of MiniMaxH3ImageVariant" in _forced_refusal(
        derived, "minimax_h3_image")
    with pytest.raises(UnsupportedModelError, match="family_adapter='minimax_h3_image'"):
        adapter_for(mb.MiniMaxH3(), "minimax_h3_image")


# ComfyUI 0.45.0's class tree for the three families whose forced bind was wider than
# `auto`: each refused class is another upstream model (docs/TROUBLESHOOTING.md #153).
_WAN_REFUSED = ("WAN21_Camera", "WAN21_CausalAR", "WAN21_HuMo", "WAN21_Vace",
                "WAN22_Animate", "WAN22_RelayVSR", "WAN22_S2V")
_FORCED_TREE = {
    "WAN21": None, "WAN22": "WAN21", "WAN21_FlowRVS": "WAN21",
    **dict.fromkeys(_WAN_REFUSED, "WAN21"),
    "HunyuanVideo": None, "HunyuanVideoI2V": "HunyuanVideo",
    "HunyuanVideoSkyreelsI2V": "HunyuanVideo", "HunyuanVideo15": "HunyuanVideo",
    "HunyuanImage21": None,
    "Lumina2": None, "MingImage": "Lumina2", "ZImagePixelSpace": "Lumina2",
}


def _forced_refusal(model, family):
    from dgx_monarch.adapters import UnsupportedModelError, adapter_for, get_adapter
    from dgx_monarch.refusal import RefusalClass, parse_refusal_tag

    with pytest.raises(UnsupportedModelError):
        get_adapter(model)  # what the force must not widen
    with pytest.raises(UnsupportedModelError) as excinfo:
        adapter_for(model, family)
    message = str(excinfo.value)
    tag = parse_refusal_tag(message)
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert tag.guard is None and tag.waivable is False
    assert f"family_adapter={family!r} does not fit this checkpoint" in message
    assert message.endswith("See docs/TROUBLESHOOTING.md #153.")
    return message


def _bound(model, family):
    from dgx_monarch.adapters import adapter_for

    adapter = adapter_for(model, family)
    assert adapter.family == family
    return type(adapter).__name__


def test_the_shipped_adapters_refuse_under_force_what_they_refuse_under_auto(monkeypatch):
    """The twelve classes a forced family bound although `auto` refuses them, on
    the shipped registry: every one refuses, with the built class named, and so
    does a derivative of one. What `auto` takes still binds, to the same
    adapter, and so does a derivative of a vetted class."""
    from dgx_monarch.adapters import get_adapter
    from fake_model_base_helpers import install_fake_model_base

    mb = install_fake_model_base(monkeypatch, _FORCED_TREE)
    for name in _WAN_REFUSED:
        message = _forced_refusal(getattr(mb, name)(), "wan")
        assert f"ComfyUI built a {name}, and" in message
        assert f"wan variant {name} is not in the dgx-monarch launch set" in message
    derived = type("VaceFinetune", (mb.WAN21_Vace,), {})()
    assert "a VaceFinetune, a derivative of WAN21_Vace" in _forced_refusal(derived, "wan")
    for name in ("WAN21", "WAN22", "WAN21_FlowRVS"):
        model = getattr(mb, name)()
        assert _bound(model, "wan") == "WanAdapter" == type(get_adapter(model)).__name__
    assert _bound(type("Wan22Finetune", (mb.WAN22,), {})(), "wan") == "WanAdapter"

    for name in ("HunyuanVideo", "HunyuanVideoI2V", "HunyuanVideoSkyreelsI2V"):
        message = _forced_refusal(getattr(mb, name)(), "hunyuan")
        assert f"hunyuan variant {name} is not in the dgx-monarch launch set" in message
        # The adapter's remedy no longer names topologies its own refusal covers.
        assert "cfg2 or single" not in message and "stock ComfyUI on one GPU" in message
    for name in ("HunyuanVideo15", "HunyuanImage21"):
        assert _bound(getattr(mb, name)(), "hunyuan") == "HunyuanAdapter"


def test_a_forced_zimage_takes_the_named_class_only_as_auto_does(monkeypatch):
    """Lumina 2.0 builds the class Z-Image's adapter names. `auto` tells them
    apart by the diffusion model's learned pad tokens, and the forced path asks
    the same test: with them it binds, also through a derivative; without them
    it refuses. Ming-Image, a subclass ComfyUI defines, refuses either way."""
    from fake_model_base_helpers import install_fake_model_base

    mb = install_fake_model_base(monkeypatch, _FORCED_TREE)
    zimage = SimpleNamespace(pad_tokens_multiple=32)
    lumina = SimpleNamespace(pad_tokens_multiple=None)

    def model(cls, diffusion_model):
        instance = cls()
        instance.diffusion_model = diffusion_model
        return instance

    assert _bound(model(mb.Lumina2, zimage), "zimage") == "ZImageAdapter"
    assert _bound(model(mb.ZImagePixelSpace, zimage), "zimage") == "ZImageAdapter"
    finetune = type("ZImageFinetune", (mb.Lumina2,), {})
    assert _bound(model(finetune, zimage), "zimage") == "ZImageAdapter"
    assert "ComfyUI built a Lumina2, and" in _forced_refusal(model(mb.Lumina2, lumina), "zimage")
    assert "a derivative of Lumina2" in _forced_refusal(model(finetune, lumina), "zimage")
    for diffusion_model in (zimage, lumina):
        message = _forced_refusal(model(mb.MingImage, diffusion_model), "zimage")
        assert "does not take this MingImage model under 'auto' either" in message


def _write_safetensors(path, metadata):
    """A minimal valid safetensors container carrying `metadata`."""
    header = {"__metadata__": metadata,
              "a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
    blob = json.dumps(header).encode()
    path.write_bytes(len(blob).to_bytes(8, "little") + blob + b"\x00" * 4)
    return str(path)


def test_a_configless_ltx_file_refuses_under_a_forced_ltx_family(tmp_path):
    """The LTX 2.5 header carries the model config and ComfyUI's constructor
    defaults disagree with it, so a file whose header lost the `config` key
    builds a structurally different model and still accepts most of the
    weights (docs/MODELS.md). Naming the family cannot put the config back."""
    from dgx_monarch.adapters.base import UnsupportedModelError

    path = _write_safetensors(tmp_path / "finetune.safetensors",
                              {"model_version": "2.5.0"})
    with pytest.raises(UnsupportedModelError) as excinfo:
        family_select.assert_override_admits_checkpoint("ltx", path)
    message = str(excinfo.value)
    assert "'config'" in message
    assert "structurally different model" in message


def test_the_configless_refusal_is_tagged_class_p_with_no_waiver(tmp_path):
    from dgx_monarch.adapters.base import UnsupportedModelError
    from dgx_monarch.refusal import RefusalClass, parse_refusal_tag

    path = _write_safetensors(tmp_path / "finetune.safetensors", {})
    with pytest.raises(UnsupportedModelError) as excinfo:
        family_select.assert_override_admits_checkpoint("ltx", path)
    tag = parse_refusal_tag(str(excinfo.value))
    assert tag is not None
    assert tag.refusal_class is RefusalClass.PHYSICS
    assert tag.guard is None and tag.waivable is False


def test_an_ltx_file_that_kept_its_config_is_admitted(tmp_path):
    path = _write_safetensors(tmp_path / "real.safetensors",
                              {"config": "{\"transformer\": {}}",
                               "model_version": "2.5.0"})
    family_select.assert_override_admits_checkpoint("ltx", path)


def test_other_families_are_not_asked_for_a_config(tmp_path):
    """Only LTX's constructor defaults disagree with its header."""
    path = _write_safetensors(tmp_path / "k.safetensors", {})
    for family in ("krea2", "flux2", "wan", "minimax_h3"):
        family_select.assert_override_admits_checkpoint(family, path)


def test_an_unreadable_file_leaves_the_loaders_own_report_in_place(tmp_path):
    """The loader reports a broken header; this guard covers a valid file that
    lost one key."""
    path = tmp_path / "broken.safetensors"
    path.write_bytes(b"not a safetensors container")
    family_select.assert_override_admits_checkpoint("ltx", str(path))
    family_select.assert_override_admits_checkpoint("ltx", str(tmp_path / "gone"))


def test_a_non_safetensors_path_is_left_alone(tmp_path):
    path = tmp_path / "legacy.ckpt"
    path.write_bytes(b"x")
    family_select.assert_override_admits_checkpoint("ltx", str(path))


def test_a_forced_family_never_reaches_the_vouched_auto_slab_rung():
    """A claim is not evidence. The vouched set records ceremony results about
    a family's real artifacts, so naming the family must not inherit them."""
    from dgx_monarch.actor import store_residency

    common = {
        "path": "/models/finetune.safetensors",
        "unet_name": "finetune.safetensors",
        "model_options": {},
        "slab_weights": "auto",
        "slab_capable_path": True,
        "lora_low_rss": True,
        "fsdp_launch": False,
        "blocked_reason": "",
        "authoritative_slab_retry": False,
        "memoized_family": lambda _path: "krea2",
        "vouched_families": frozenset({"krea2"}),
        "file_identity": lambda _path: "1:2:3:4:5",
    }
    vouched = store_residency.resolve(**common)
    assert vouched.use_slab is True

    forced = store_residency.resolve(**common, family_override="krea2")
    assert forced.use_slab is False
    assert forced.auto_retry_eligible is False


def test_an_explicit_slab_request_still_loads_slab_under_an_override():
    """The explicit path stays open, and it gates as its own combination with
    the cross-residency leg."""
    from dgx_monarch.actor import store_residency

    decision = store_residency.resolve(
        path="/models/finetune.safetensors", unet_name="finetune.safetensors",
        model_options={}, slab_weights=True, slab_capable_path=True,
        lora_low_rss=True, fsdp_launch=False, blocked_reason="",
        authoritative_slab_retry=False, memoized_family=lambda _p: None,
        vouched_families=frozenset({"krea2"}), file_identity=lambda _p: "1:2:3:4:5",
        family_override="krea2")
    assert decision.use_slab is True


def test_a_forced_load_does_not_write_the_family_memo():
    """The memo is keyed by file identity and read by the plain auto residency
    path, so a claimed family written here would grant auto-slab in a later
    session whose capability context carries no override at all."""
    source = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "src" / "dgx_monarch" / "actor" / "store_load.py"
    ).read_text()
    assert "if not family_override:\n            # A forced family is never memoized" in source
    assert source.count("memoize_family(path, family)") == 1


def test_the_effective_family_replaces_only_the_family():
    assert family_select.effective_family("longcat", None) == "longcat"
    assert family_select.effective_family("longcat", "flux") == "flux"
    assert family_select.effective_family("unknown", "krea2") == "krea2"


def test_a_forced_render_says_so_once_with_what_the_header_read(monkeypatch, caplog):
    """A forced krea2 run would otherwise inherit that row's HW status in
    silence. The warning names the forced and the sniffed family, for when a
    finetune misbehaves."""
    import logging

    from dgx_monarch.nodes import common as common_mod

    monkeypatch.setattr(common_mod, "_impl_warned", set())
    monkeypatch.setattr(common_mod, "sniff_checkpoint_for_topology",
                        lambda _path: ("longcat", "fp8"))
    folder_paths = __import__("types").ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, name: f"/models/{name}"
    monkeypatch.setitem(__import__("sys").modules, "folder_paths", folder_paths)

    model = SimpleNamespace(
        unet_name="finetune.safetensors", options={},
        mesh=SimpleNamespace(topology_preset="auto", world=2,
                             worker_args={"family_override": "krea2"}))
    latent = SimpleNamespace(shape=(1, 16, 128, 128), ndim=4)
    with caplog.at_level(logging.WARNING):
        topo, _sage, _reason = common_mod.resolve_topology(model, latent, 1.0, world=2)
    assert topo is not None
    text = caplog.text
    assert "'krea2'" in text and "'longcat'" in text
    assert "hardware evidence" in text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        common_mod.resolve_topology(model, latent, 1.0, world=2)
    assert "hardware evidence" not in caplog.text
    # The generic advisory must not fire on the second render either: its
    # text says the family came from the checkpoint header, which is untrue
    # for a forced render. _HW_VALIDATED_FAMILIES is empty, so without the
    # guard it would fire here for the forced krea2.
    assert "was selected from a checkpoint header" not in caplog.text


@pytest.fixture
def fresh_auto_warn(monkeypatch):
    """The warned set lives for the process, so every test needs its own."""
    monkeypatch.setattr(family_select, "_auto_warned", set())


def _warnings(caplog):
    return [record.getMessage() for record in caplog.records]


def test_a_configless_ltx_file_warns_on_the_auto_path(tmp_path, caplog,
                                                      fresh_auto_warn):
    """Auto makes no claim the file has to honor, so a missing key is a report,
    not a refusal: it proves the key is absent, never that the model ComfyUI
    built disagrees with this file."""
    import logging

    path = _write_safetensors(tmp_path / "finetune.safetensors",
                              {"model_version": "2.5.0"})
    with caplog.at_level(logging.WARNING):
        family_select.warn_auto_admits_checkpoint("ltx", path)
    messages = _warnings(caplog)
    assert len(messages) == 1
    assert "'config'" in messages[0]
    assert "Not refused here" in messages[0]


def test_the_auto_warning_fires_once_per_file_and_again_for_a_second_file(
    tmp_path, caplog, fresh_auto_warn,
):
    import logging

    first = _write_safetensors(tmp_path / "one.safetensors", {})
    second = _write_safetensors(tmp_path / "two.safetensors", {})
    with caplog.at_level(logging.WARNING):
        family_select.warn_auto_admits_checkpoint("ltx", first)
        family_select.warn_auto_admits_checkpoint("ltx", first)
        assert len(_warnings(caplog)) == 1
        family_select.warn_auto_admits_checkpoint("ltx", second)
    assert len(_warnings(caplog)) == 2


def test_an_ltx_file_that_kept_its_config_is_quiet(tmp_path, caplog,
                                                   fresh_auto_warn):
    import logging

    path = _write_safetensors(tmp_path / "real.safetensors",
                              {"config": "{\"transformer\": {}}",
                               "model_version": "2.5.0"})
    with caplog.at_level(logging.WARNING):
        family_select.warn_auto_admits_checkpoint("ltx", path)
    assert _warnings(caplog) == []


def test_other_families_are_never_asked_for_a_config_on_the_auto_path(
    tmp_path, caplog, fresh_auto_warn,
):
    """The two pruned int8-convrot MiniMax H3 files (fl2va, ref2va) carry no
    `config` (headers read 2026-10-07) and are the family's primary hardware
    evidence: ComfyUI sizes H3 and WAN from their tensors and merges the key
    only when it is there. LTX takes only its block count, head size and
    cross-attention width from tensor shapes and the rest of its geometry from
    the header."""
    import logging

    path = _write_safetensors(tmp_path / "k.safetensors", {})
    with caplog.at_level(logging.WARNING):
        for family in ("krea2", "flux2", "wan", "minimax_h3", "unknown"):
            family_select.warn_auto_admits_checkpoint(family, path)
    assert _warnings(caplog) == []


def test_an_unreadable_or_missing_or_legacy_file_is_quiet(tmp_path, caplog,
                                                          fresh_auto_warn):
    import logging

    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors container")
    legacy = tmp_path / "legacy.ckpt"
    legacy.write_bytes(b"x")
    with caplog.at_level(logging.WARNING):
        family_select.warn_auto_admits_checkpoint("ltx", str(broken))
        family_select.warn_auto_admits_checkpoint("ltx", str(tmp_path / "gone"))
        family_select.warn_auto_admits_checkpoint("ltx", str(legacy))
    assert _warnings(caplog) == []


def test_the_scope_stays_the_one_family_whose_header_carries_its_architecture():
    assert family_select._CONFIG_REQUIRED_FAMILIES == frozenset({"ltx"})
