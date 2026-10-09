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
   typed class P message before any kernel runs.
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

    def matches(self, _base_model):  # pragma: no cover - never called under force
        raise AssertionError(
            "the forced path must not call matches(): exact-type adapters raise "
            "there on exactly the subclasses an override exists to admit")

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
    """ComfyUI built a subclass of a class this family owns: detection's
    exact-type adapters raise on it, and the forced family binds it."""
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
