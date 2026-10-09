"""The two operator surfaces that turn the comfy-managed rung on: the cluster.toml
worker-arg schema and the Init node widget."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from comfy_managed_helpers import (  # noqa: F401  # autouse fixture import.
    REPO,
    _isolated_process_state,
)
from dgx_monarch import config_schema
from dgx_monarch.actor import comfy_dynamic


def test_the_schema_accepts_the_key_only_as_a_boolean():
    assert config_schema.validate_worker_args({"comfy_managed": True}) == {
        "comfy_managed": True}
    for value in ("on", "true", 1, "off"):
        with pytest.raises(config_schema.ClusterConfigError):
            config_schema.validate_worker_args({"comfy_managed": value})


def test_the_schema_refuses_the_two_combinations_the_init_node_refuses():
    with pytest.raises(config_schema.ClusterConfigError) as excinfo:
        config_schema.validate_worker_args(
            {"comfy_managed": True, "slab_weights": True})
    assert "slab_weights" in str(excinfo.value)
    with pytest.raises(config_schema.ClusterConfigError) as excinfo:
        config_schema.validate_worker_args(
            {"comfy_managed": True, "lora_low_rss": True})
    assert "lora_low_rss" in str(excinfo.value)


def test_the_schema_accepts_an_operator_who_asks_for_pinned_staging():
    """apply_policy never writes disable_pinned_memory, but an operator may set it
    false, and the schema must not refuse a shape the capacity wall prices."""
    assert config_schema.validate_worker_args({
        "comfy_managed": True, "slab_weights": False, "lora_low_rss": False,
        "disable_pinned_memory": False,
    })["comfy_managed"] is True


def test_the_schema_accepts_the_shape_apply_policy_produces():
    assert config_schema.validate_worker_args(
        comfy_dynamic.apply_policy({"comfy_managed": True}))["comfy_managed"] is True


def test_the_schema_still_accepts_slab_auto_beside_a_disabled_rung():
    assert config_schema.validate_worker_args(
        {"comfy_managed": False, "slab_weights": "auto"})["slab_weights"] == "auto"


def test_the_schema_accepts_slab_auto_beside_an_ENABLED_rung():
    """`auto` is a resolution policy, not a residency request, and apply_policy
    resolves it to off inside the worker. It is the documented default on
    unified memory, so refusing it on truthiness rejects a cluster.toml that
    spells it out and disagrees with the Init node, which refuses only `on`.
    """
    out = config_schema.validate_worker_args(
        {"comfy_managed": True, "slab_weights": "auto"})
    assert out["comfy_managed"] is True
    assert out["slab_weights"] == "auto"
    assert comfy_dynamic.apply_policy(dict(out))["slab_weights"] is False


def test_the_two_boundaries_refuse_exactly_the_same_combinations(monkeypatch):
    """The schema and the Init node must not disagree about what is legal, or
    an operator's cluster.toml is rejected for a combination the widget takes.
    """
    for slab in ("auto", "off"):
        _init(monkeypatch, comfy_managed="on", slab_weights=slab)
        assert config_schema.validate_worker_args(
            {"comfy_managed": True,
             "slab_weights": "auto" if slab == "auto" else False})
    with pytest.raises(ValueError):
        _init(monkeypatch, comfy_managed="on", slab_weights="on")
    with pytest.raises(config_schema.ClusterConfigError):
        config_schema.validate_worker_args(
            {"comfy_managed": True, "slab_weights": True})


def _init(monkeypatch, **kwargs):
    from dgx_monarch import first_render
    from dgx_monarch.nodes import init as init_mod

    handle = SimpleNamespace(world=1, config=None, config_fingerprint="local")
    monkeypatch.setattr(init_mod, "get_mesh", lambda **_kw: handle)
    monkeypatch.setattr(first_render, "nccl_deferred", lambda: None)
    (spec,) = init_mod.DGXMonarchInit().init(topology="auto", mode="local", **kwargs)
    return spec.worker_args


def test_the_widget_is_two_valued_and_defaults_off():
    from dgx_monarch.nodes.init import DGXMonarchInit

    optional = DGXMonarchInit.INPUT_TYPES()["optional"]
    values, meta = optional["comfy_managed"]
    assert values == ["off", "on"]
    assert meta["default"] == "off"
    assert meta["advanced"] is True
    tooltip = meta["tooltip"]
    assert "docs/TROUBLESHOOTING.md #62" in tooltip
    for promise in ("lora_low_rss", "slab_weights", "FSDP", "Attached mesh reset"):
        assert promise in tooltip
    # No `auto`: the operator turns the rung on, or it stays off.
    assert "auto" not in values


def test_the_widget_keeps_its_appended_slot_and_the_js_epoch_matches():
    """comfy_managed sits at position 17 (index 16) and never moves.

    ComfyUI stores widget values by position, so the pin is on this widget's own
    slot, not on being last: later widgets append after it. The general
    append-only rule lives in tests/test_init_widget_order.py.
    """
    from dgx_monarch.nodes.init import DGXMonarchInit

    order = [*DGXMonarchInit.INPUT_TYPES()["required"],
             *DGXMonarchInit.INPUT_TYPES()["optional"]]
    assert order.index("comfy_managed") == 16, (
        "comfy_managed moved off the slot it was appended to; every workflow "
        "saved since it shipped reads its value from that position (issue #75)."
    )
    js = (REPO / "web" / "js" / "dgx_monarch_widgets.js").read_text()
    assert '"comfy_managed"' in js, (
        "append a new INIT_ORDERS row carrying comfy_managed so workflows "
        "saved before this widget keep healing"
    )


def test_the_init_node_writes_all_three_keys_when_the_rung_is_on(monkeypatch):
    """With the rung on, the Init node writes comfy_managed and forces slab_weights and
    lora_low_rss off, matching what apply_policy enforces in the worker. The two False
    levers alone read as stock residency, so the projection blocker, the loader-site
    residency, the stock-stamp check and the auto-gate and quarantine risk predicates
    each test comfy_managed itself through residency_mode.requested."""
    args = _init(monkeypatch, comfy_managed="on")
    assert args["comfy_managed"] is True
    assert args["slab_weights"] is False
    assert args["lora_low_rss"] is False


def test_the_init_node_writes_no_key_at_all_when_the_rung_is_off(monkeypatch):
    """With the rung off the Init node writes no key at all, not False.

    worker_args is embedded verbatim in every gate capability context, so an
    unconditional key would change the canonical string of every deployed
    render and invalidate every PASS on the box.
    """
    args = _init(monkeypatch, comfy_managed="off")
    assert "comfy_managed" not in args
    assert "slab_weights" not in args
    assert "lora_low_rss" not in args


def test_the_init_node_defaults_to_off(monkeypatch):
    assert "comfy_managed" not in _init(monkeypatch)


def test_the_init_node_refuses_the_two_contradictory_combinations(monkeypatch):
    with pytest.raises(ValueError) as excinfo:
        _init(monkeypatch, comfy_managed="on", slab_weights="on")
    assert "slab_weights" in str(excinfo.value)
    with pytest.raises(ValueError) as excinfo:
        _init(monkeypatch, comfy_managed="on", lora_low_rss="on")
    assert "lora_low_rss" in str(excinfo.value)


def test_the_init_node_accepts_the_rung_beside_auto_levers(monkeypatch):
    args = _init(monkeypatch, comfy_managed="on", slab_weights="auto",
                 lora_low_rss="auto")
    assert args["comfy_managed"] is True
    assert args["slab_weights"] is False


def test_the_managed_block_runs_after_the_explicit_lever_resolution():
    """The explicit slab_weights and lora_low_rss writes come first, so the
    rung's forced False survives instead of being overwritten."""
    source = inspect.getsource(
        __import__("dgx_monarch.nodes.init", fromlist=["init"])
        .DGXMonarchInit._init_with_bootstrap_policy)
    assert source.index("slab_explicit") < source.index('worker_args["comfy_managed"]')
