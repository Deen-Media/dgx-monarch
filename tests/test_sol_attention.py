"""The sol-attn lever: every refusal, the off switch, and the waiver route.

The kernel is approximate and not identity preserving, so most of this file
tests what the lever refuses. Three properties matter most: no auto-table row
can select it, no render runs it without a stamped class-K waiver, and the CuTe
dispatch entry is applied only when the device and both optional packages
support it.
"""
from __future__ import annotations

import ast
import dataclasses
import pathlib
import types

import pytest
import torch

from dgx_monarch import accuracy_waiver
from dgx_monarch import topology as topology_mod
from dgx_monarch.adapters import sol_attention as sol
from dgx_monarch.adapters import sol_attention_guards as guards
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.gate_audit_vocab import KIND_CLASS, KIND_GUARD_PREFIXES, check_guard
from dgx_monarch.refusal import GUARDS, RefusalClass

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"
KIND = "waive-known-wrong:sol-attn"


def _bthd(rows=8, heads=2, dim=128, dtype=torch.bfloat16):
    return torch.zeros(1, rows, heads, dim, dtype=dtype)


# The name carries the tau, and only shipped names resolve.

def test_only_the_shipped_kernel_name_resolves_to_a_tau():
    assert sol.is_sol_kernel("SOL_ATTN_TAU1.0")
    assert not sol.is_sol_kernel("SAGE_AUTO")
    assert not sol.is_sol_kernel(None)
    assert sol.parse_sol_tau("SOL_ATTN_TAU1.0") == 1.0
    with pytest.raises(ValueError):
        sol.parse_sol_tau("TORCH_FLASH")


@pytest.mark.parametrize("name", ["SOL_ATTN_TAU0.5", "SOL_ATTN_TAU1.3", "SOL_ATTN_TAUx"])
def test_an_unshipped_tau_refuses_instead_of_rounding_to_a_shipped_one(name):
    """A tau is a measured setting, so an unmeasured one has no code path."""
    with pytest.raises(UnsupportedModelError) as excinfo:
        sol.parse_sol_tau(name)
    assert "SOL_ATTN_TAU1.0" in str(excinfo.value)


def test_every_shipped_tau_parses_to_its_own_value():
    """Each name is a distinct measured setting the ledger can bind to."""
    assert [sol.parse_sol_tau(n) for n in sol.SOL_SELECTABLE_KERNELS] == [
        0.6, 0.7, 1.0]


def test_the_validated_tau_set_ships_empty_and_grants_nothing():
    """The VALIDATED_ULYSSES_FOLDS shape: an allow-list that admits nobody."""
    assert dict(sol.SOL_VALIDATED_TAUS) == {}
    assert not sol.tau_is_vouched("minimax_h3", 1.0)
    with pytest.raises(TypeError):  # read-only, unlike the plain-dict fold table
        sol.SOL_VALIDATED_TAUS["minimax_h3"] = (1.0,)


# Off by default, structurally.

def test_no_auto_table_row_can_ever_name_this_kernel():
    """The auto table's only kernel field is the `sage` bool, so no row can say sol.

    This is the whole off-by-default guarantee. A row selects sage or not; no
    field of a row selects a kernel by name.
    """
    fields = {f.name for f in dataclasses.fields(topology_mod.AutoRule)}
    assert "sage" in fields
    assert not fields & {"attention", "kernel", "sol", "tau"}
    for rule in topology_mod.AUTO_TABLE:
        assert isinstance(rule.sage, bool)
        assert not sol.is_sol_kernel(rule.note)
    assert not sol.is_sol_kernel("SAGE_AUTO")


def test_the_widget_ships_exactly_the_selectable_sol_values():
    from dgx_monarch.nodes.init import _ATTENTION_KERNELS

    shipped = [name for name in _ATTENTION_KERNELS if sol.is_sol_kernel(name)]
    assert shipped == list(sol.SOL_SELECTABLE_KERNELS) == [
        "SOL_ATTN_TAU0.6", "SOL_ATTN_TAU0.7", "SOL_ATTN_TAU1.0"]
    assert _ATTENTION_KERNELS[0] == "TORCH_FLASH"  # the default is untouched


# One resolution helper, used at all three sites.

def test_an_explicit_sol_choice_beats_an_auto_sage_suggestion():
    from dgx_monarch.nodes.common import resolve_sample_attention

    assert resolve_sample_attention("SOL_ATTN_TAU1.0", True) == "SOL_ATTN_TAU1.0"
    assert resolve_sample_attention("SOL_ATTN_TAU1.0", False) == "SOL_ATTN_TAU1.0"
    assert resolve_sample_attention("TORCH_FLASH", True) == "SAGE_AUTO"
    assert resolve_sample_attention("TORCH_FLASH", False) == "TORCH_FLASH"


def test_every_resolution_site_goes_through_the_shared_helper():
    """Three sites feed one capability context; a fourth spelling reads stale.

    mesh_residency and mesh_capacity_consent compare the recorded value against
    the dispatched request for equality, so a site that canonicalized
    differently would make every residency and consent context look stale.
    """
    sites = ["nodes/render_submit.py", "nodes/auto_gate.py", "nodes/gate_ceremony.py"]
    for site in sites:
        text = (SRC / site).read_text()
        assert "resolve_sample_attention(" in text, site
        assert '"SAGE_AUTO" if' not in text, f"{site} still resolves inline"


# The family scope, bound where the worker injects attention.

def test_h3_is_the_only_family_in_scope():
    assert set(sol.SOL_ATTN_FAMILIES) == {"minimax_h3"}
    sol.assert_family_supported("minimax_h3")


@pytest.mark.parametrize("family,fragment", [
    ("ltx", "0.058"),
    ("krea2", "grouped-query"),
    ("wan", "no render ceremony"),
])
def test_another_family_refuses_with_its_own_reason_and_names_sage(family, fragment):
    with pytest.raises(UnsupportedModelError) as excinfo:
        sol.assert_family_supported(family)
    message = str(excinfo.value)
    assert fragment in message
    assert "SAGE_*" in message and "[dgxm:P" in message


def test_binding_a_scope_records_nothing_without_a_scope_to_record_into():
    sol.bind_sol_scope(object(), "wan")  # no sol_scope attribute: nothing recorded, no raise


def test_sequence_parallel_degree_one_refuses_at_construction():
    """At sp 1 the model keeps stock attention, so the sol label would be false.

    The check sits before load_sol_attn, so it answers on a box without the
    optional package, and before any bind: at sp 1 the worker never injects an
    implementation, so a first-call check could never fire.
    """
    with pytest.raises(UnsupportedModelError) as excinfo:
        sol.make_sol_usp_attention("SOL_ATTN_TAU1.0", True, ulysses=1, ring=1)
    assert "sequence-parallel degree of 1" in str(excinfo.value)


def test_the_scope_binds_the_family_the_worker_actually_loaded():
    implementation = types.SimpleNamespace(sol_scope=sol.SolScope())
    sol.bind_sol_scope(implementation, "minimax_h3")
    assert implementation.sol_scope.family == "minimax_h3"
    # Recording refuses nothing, so a second load of another family overwrites
    # rather than raising; a family outside the lever refuses on its first call.
    sol.bind_sol_scope(implementation, "ltx")
    assert implementation.sol_scope.family == "ltx"


def _dispatch_with(monkeypatch, kernel, built=None):
    """A real _AttentionDispatch carrying a stand-in implementation.

    The object the worker binds on is the dispatcher, never the implementation,
    so a double with the attribute already on it proves nothing. Only the
    construction is faked; the dispatcher is the shipped one.
    """
    from dgx_monarch import adapters
    from dgx_monarch.actor import attention_dispatch as dispatch_mod

    def fake_sol(kernel_name, sync, *, ulysses, ring, scope=None):
        def implementation(*_a, **_k):
            raise AssertionError("the stand-in is never called")

        implementation.sol_scope = scope
        if built is not None:
            built.append(implementation)
        return implementation

    def fake_other(kernel_name, sync):
        return lambda *_a, **_k: None

    monkeypatch.setattr(sol, "make_sol_usp_attention", fake_sol)
    monkeypatch.setattr(adapters, "make_usp_attention", fake_other)
    dispatch = dispatch_mod._AttentionDispatch()
    dispatch.configure(kernel, True, topology={"ulysses": 2, "ring": 1},
                       world=2, setup_generation=1)
    return dispatch


def test_the_worker_binds_on_the_object_it_actually_holds(monkeypatch):
    """The worker binds on the dispatcher, not the implementation.

    Hardware, 2026-08-15: every sol render refused "without binding a model
    family" because the bind looked for an attribute the dispatcher did not
    carry and returned in silence. Three unit tests missed it by passing a
    double that already had the attribute.
    """
    dispatch = _dispatch_with(monkeypatch, "SOL_ATTN_TAU1.0")
    sol.bind_sol_scope(dispatch, "minimax_h3")
    assert dispatch.sol_scope.family == "minimax_h3"


def test_the_bound_family_survives_an_implementation_rebuild(monkeypatch):
    """Binding and building have different lifetimes, so the scope outlives each build.

    The family is bound when a base model loads; the implementation is rebuilt
    whenever the configured dispatch changes within one setup generation, as a
    sync_ulysses flip does here. A rebuild under an already bound family must
    inherit it rather than refuse. A new setup generation forgets the family on
    purpose; that case and a kernel change have their own tests below.
    """
    built: list = []
    dispatch = _dispatch_with(monkeypatch, "SOL_ATTN_TAU1.0", built)
    sol.bind_sol_scope(dispatch, "minimax_h3")
    dispatch.configure("SOL_ATTN_TAU1.0", False, topology={"ulysses": 2, "ring": 1},
                       world=2, setup_generation=1)
    assert len(built) == 2 and built[1] is not built[0]
    assert built[1].sol_scope is dispatch.sol_scope
    assert built[1].sol_scope.family == "minimax_h3"


def test_binding_under_another_kernel_records_the_family(monkeypatch):
    """The worker records the family on every load, so a flip to sol has one to read."""
    dispatch = _dispatch_with(monkeypatch, "TORCH_FLASH")
    assert dispatch.sol_scope is not None
    sol.bind_sol_scope(dispatch, "krea2")  # records, refuses nothing
    assert dispatch.sol_scope.family == "krea2"


def test_a_model_loaded_under_flash_renders_under_sol_without_reloading(monkeypatch):
    """A widget flip on a resident model must still bind.

    Same setup generation, so nothing unloaded between the load and the flip.
    """
    built: list = []
    dispatch = _dispatch_with(monkeypatch, "TORCH_FLASH", built)
    sol.bind_sol_scope(dispatch, "minimax_h3")
    dispatch.configure("SOL_ATTN_TAU1.0", True, topology={"ulysses": 2, "ring": 1},
                       world=2, setup_generation=1)
    assert built and built[-1].sol_scope.family == "minimax_h3"


def test_a_retired_setup_generation_forgets_the_family(monkeypatch):
    """A teardown unloads the residents that bound it, so the family goes too."""
    dispatch = _dispatch_with(monkeypatch, "SOL_ATTN_TAU1.0")
    sol.bind_sol_scope(dispatch, "minimax_h3")
    dispatch.invalidate()
    assert dispatch.sol_scope.family is None


def test_the_first_call_refuses_a_family_that_never_bound():
    with pytest.raises(UnsupportedModelError) as excinfo:
        sol.sol_waiver_required(None, 1.0)
    assert "without binding a model family" in str(excinfo.value)


def test_the_first_call_refuses_a_family_outside_the_lever():
    with pytest.raises(UnsupportedModelError) as excinfo:
        sol.sol_waiver_required("krea2", 1.0)
    assert "grouped-query" in str(excinfo.value)
    assert sol.sol_waiver_required("minimax_h3", 1.0) is True


def test_the_wan_view_forwards_the_scope_to_its_dispatcher(monkeypatch):
    """Wan adapters hand back a view; a bind through it must still land."""
    from dgx_monarch.actor import attention_dispatch as dispatch_mod

    dispatch = _dispatch_with(monkeypatch, "SOL_ATTN_TAU1.0")
    view = dispatch_mod._WanAttentionView(dispatch, ("key",))
    sol.bind_sol_scope(view, "minimax_h3")
    assert dispatch.sol_scope.family == "minimax_h3"


# The layout guard, the one the kernel does not have.

def test_a_transposed_tensor_is_refused_rather_than_silently_computed():
    """(B, H, T, 128) is structurally valid (B, T, H, 128) to the kernel."""
    preflight = guards.CallPreflight(2)
    preflight.surface(None, None, None, 4, None, None, False, False, False,
                      None, None, None, {})
    assert preflight.expected_local_heads() == 2
    bhtd = torch.zeros(1, 2, 8, 128, dtype=torch.bfloat16)  # heads on axis 1
    with pytest.raises(UnsupportedModelError) as excinfo:
        guards.validate_bthd(bhtd, bhtd, bhtd, preflight.expected_local_heads())
    assert "wrong output that looks valid" in str(excinfo.value)


def test_the_layout_guard_refuses_when_no_head_count_was_declared():
    with pytest.raises(UnsupportedModelError) as excinfo:
        guards.CallPreflight(2).expected_local_heads()
    assert "layout guard" in str(excinfo.value)


@pytest.mark.parametrize("kwargs,fragment", [
    ({"enable_gqa": True}, "grouped-query"),
    ({"query_bias_groups": [(0, 1)]}, "ordered query groups"),
    ({"mask": object()}, "no attention mask"),
    ({"heads": 0}, "positive declared head count"),
    ({"heads": 3}, "divide across the ranks"),
])
def test_the_outer_surface_refuses_what_the_kernel_cannot_express(kwargs, fragment):
    call = {"q": None, "k": None, "v": None, "heads": 4, "mask": None,
            "attn_precision": None, "skip_reshape": False,
            "skip_output_reshape": False, "enable_gqa": False, "drop_rows": None,
            "kv_drop_rows": None, "query_bias_groups": None, "kwargs": {}}
    call.update(kwargs)
    with pytest.raises(UnsupportedModelError) as excinfo:
        guards.CallPreflight(2).surface(*call.values())
    assert fragment in str(excinfo.value)


@pytest.mark.parametrize("tensors,fragment", [
    ((torch.zeros(4), torch.zeros(4), torch.zeros(4)), "4D"),
    ((_bthd(), _bthd(rows=9), _bthd()), "self-attention only"),
    ((_bthd(dim=64), _bthd(dim=64), _bthd(dim=64)), "head dimension 128"),
    ((_bthd(dtype=torch.float16),) * 3, "bfloat16 only"),
])
def test_the_kernel_constraints_are_refused_not_worked_around(tensors, fragment):
    with pytest.raises(UnsupportedModelError) as excinfo:
        guards.validate_bthd(*tensors, 2)
    assert fragment in str(excinfo.value)


def test_a_cpu_tensor_refuses():
    with pytest.raises(UnsupportedModelError) as excinfo:
        guards.validate_bthd(_bthd(), _bthd(), _bthd(), 2)
    assert "CUDA tensors" in str(excinfo.value)


@pytest.mark.parametrize("override,fragment", [
    ({"causal": True}, "noncausal only"),
    ({"dropout_p": 0.1}, "no attention dropout"),
    ({"return_attn_probs": True}, "no attention probabilities"),
    ({"window_size": (16, 16)}, "no sliding window"),
    ({"alibi_slopes": object()}, "no alibi slopes"),
    ({"softcap": 1.0}, "no logit softcap"),
    ({"deterministic": True}, "no deterministic mode"),
    ({"joint_strategy": "front"}, "joint tensors"),
    ({"q_descale": object()}, "joint tensors"),
])
def test_every_ignored_option_is_refused_rather_than_dropped(override, fragment):
    call = {"dropout_p": 0.0, "causal": False, "window_size": (-1, -1),
            "alibi_slopes": None, "deterministic": False,
            "return_attn_probs": False, "attn_layer": None,
            "joint_tensor_key": None, "joint_tensor_value": None,
            "joint_strategy": "none", "q_descale": None, "k_descale": None,
            "v_descale": None, "softcap": 0.0}
    call.update(override)
    with pytest.raises(UnsupportedModelError) as excinfo:
        guards.validate_kernel_surface(**call)
    assert fragment in str(excinfo.value)


def test_ring_refuses_with_both_structural_reasons():
    with pytest.raises(UnsupportedModelError) as excinfo:
        guards.refuse_ring(2)
    message = str(excinfo.value)
    assert "log-sum-exp" in message and "whole key row" in message


# The sink scope.

def test_a_call_outside_a_sink_scope_refuses():
    with pytest.raises(UnsupportedModelError) as excinfo:
        sol._current_sink(64)
    assert "outside a sink scope" in str(excinfo.value)


def test_a_sequence_with_no_video_segment_refuses_instead_of_guessing():
    with sol.sol_sink_scope(None):
        with pytest.raises(UnsupportedModelError) as excinfo:
            sol._current_sink(64)
    assert "no video segment" in str(excinfo.value)


def test_a_sink_longer_than_the_attended_rows_refuses():
    with sol.sol_sink_scope(80):
        with pytest.raises(UnsupportedModelError) as excinfo:
            sol._current_sink(64)
    assert "exceeds" in str(excinfo.value)


def test_the_scopes_restore_what_they_replaced():
    with sol.sol_sink_scope(5):
        with sol.sol_sink_scope(9):
            assert sol._current_sink(64) == 9
        assert sol._current_sink(64) == 5
        with sol.sol_dense_scope(True):
            assert sol._SCOPE.dense
        assert not sol._SCOPE.dense
    assert not sol._SCOPE.active


def test_the_recipe_ships_as_constants_not_as_a_widget():
    """NVIDIA's published GB10 numbers: 10 of 50 steps and 2 blocks dense."""
    assert sol.SOL_DENSE_FIRST_FRAC == 0.2
    assert sol.SOL_DENSE_BLOCKS == 2
    from dgx_monarch.nodes.init import DGXMonarchInit

    widgets = DGXMonarchInit.INPUT_TYPES()
    names = set(widgets.get("optional", {})) | set(widgets.get("required", {}))
    assert not {"sol_tau", "sol_dense_blocks", "sol_dense_first_frac"} & names


def test_the_adapter_opens_both_scopes_around_the_block_loop():
    text = (SRC / "adapters/minimax_h3.py").read_text()
    assert "sol_sink_scope(video_start)" in text
    assert "sol_dense_scope(dense_step or index < SOL_DENSE_BLOCKS)" in text
    # The schedule has counted sampler steps since 2026-08-19; a revert to the
    # timestep threshold must fail here, not on the next hardware bench.
    assert "dense_step = sol_dense_step(" in text
    assert "dense_step = t_v <" not in text


def test_the_dense_schedule_counts_sampler_steps_not_timesteps():
    """The recipe is 10 of 50 steps. Under H3's shifted sigma curve a timestep
    threshold holds about two thirds of every render dense (fixed 2026-08-19)."""
    assert [sol.sol_dense_step((i, 20), 0.5) for i in range(6)] == [
        True, True, True, True, False, False]
    assert sol.sol_dense_step((9, 50), 0.5) is True
    assert sol.sol_dense_step((10, 50), 0.5) is False
    # The discriminating case: a late step whose timestep label still sits
    # under the threshold must run sparse. The timestep rule runs it dense.
    assert sol.sol_dense_step((12, 20), 0.15) is False


def test_the_dense_schedule_falls_back_to_the_timestep_and_warns_once(caplog):
    import logging

    sol._warned_step_fallback.clear()
    with caplog.at_level(logging.WARNING, logger=sol.log.name):
        handler = logging.Handler()
        records = []
        handler.emit = records.append
        sol.log.addHandler(handler)
        try:
            assert sol.sol_dense_step(None, 0.1) is True
            assert sol.sol_dense_step(None, 0.3) is False
        finally:
            sol.log.removeHandler(handler)
    warned = [r for r in records if "falling back to the timestep" in r.getMessage()]
    assert len(warned) == 1
    sol._warned_step_fallback.clear()


def test_the_adapter_reads_the_step_position_from_the_sampler_table():
    from dgx_monarch.adapters.minimax_h3 import _sampler_step_position

    table = torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0])
    current = torch.tensor([0.7])
    assert _sampler_step_position(
        {"sigmas": current, "sample_sigmas": table}) == (2, 4)
    assert _sampler_step_position({"sample_sigmas": table}) is None
    assert _sampler_step_position(
        {"sigmas": torch.tensor([0.55]), "sample_sigmas": table}) is None
    assert _sampler_step_position(None) is None


# The cute dispatch entry.

def _find_spec_for(present):
    def find_spec(name):
        return object() if name in present else None
    return find_spec


def test_the_dispatch_entry_is_added_only_on_a_gb10_with_both_packages(monkeypatch):
    from dgx_monarch.adapters import sol_attention_backend as backend

    interface = pytest.importorskip("sol_attn.interface")
    table = {(12, 0): "cute_sm120"}
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 1))
    monkeypatch.setattr(interface, "_CUTE_BACKENDS", table, raising=False)
    reason = backend._apply_cute_dispatch(_find_spec_for({"cutlass", "tvm_ffi"}))
    assert reason.startswith("applied")
    assert table[(12, 1)] == "cute_sm120"


@pytest.mark.parametrize("capability,present,fragment", [
    ((12, 0), {"cutlass", "tvm_ffi"}, "is not (12, 1)"),
    ((9, 0), {"cutlass", "tvm_ffi"}, "is not (12, 1)"),
    ((12, 1), {"tvm_ffi"}, "cutlass is not importable"),
    ((12, 1), {"cutlass"}, "tvm_ffi is not importable"),
    ((12, 1), {"tvm_ffi"}, "pip install nvidia-cutlass-dsl"),
])
def test_the_dispatch_entry_falls_back_with_a_reason(capability, present, fragment,
                                                     monkeypatch):
    from dgx_monarch.adapters import sol_attention_backend as backend

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    reason = backend._apply_cute_dispatch(_find_spec_for(present))
    assert reason.startswith("not applied") and fragment in reason


def test_a_missing_dsl_names_the_install_because_nothing_else_refuses(monkeypatch):
    """The Triton fallback is silent, so this reason must print the install.

    A missing sol-attn package refuses and names its install. A missing CuTe
    compiler refuses nothing: the render runs, slower, and only the logged
    backend says so. So this reason carries the install, and the absence
    refusal does not: that refusal names what the kernel needs to run, not
    what makes it fast.
    """
    from dgx_monarch.adapters import sol_attention_backend as backend
    from dgx_monarch.adapters.sol_attention_guards import SOL_INSTALL_SPEC

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 1))
    reason = backend._apply_cute_dispatch(_find_spec_for({"tvm_ffi"}))
    assert backend.CUTE_PATH_INSTALL["cutlass"] in reason
    assert "nvidia-cutlass-dsl" not in SOL_INSTALL_SPEC


def test_an_unreadable_device_or_table_never_raises(monkeypatch):
    from dgx_monarch.adapters import sol_attention_backend as backend

    def boom(*_args):
        raise RuntimeError("no driver")

    monkeypatch.setattr(torch.cuda, "get_device_capability", boom)
    assert "capability unreadable" in backend._apply_cute_dispatch(
        _find_spec_for({"cutlass", "tvm_ffi"}))
    interface = pytest.importorskip("sol_attn.interface")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 1))
    monkeypatch.setattr(interface, "_CUTE_BACKENDS", ["not", "a", "dict"],
                        raising=False)
    assert "not a dict" in backend._apply_cute_dispatch(
        _find_spec_for({"cutlass", "tvm_ffi"}))


def test_the_capability_is_never_spoofed():
    """Forcing (12, 0) builds sm_120a cubins that will not load on sm_121.

    The device capability is read and compared. Nothing in this module may
    assign to it, monkeypatch it, or report a literal capability in its place.
    """
    text = (SRC / "adapters/sol_attention_backend.py").read_text()
    assert "NEVER SPOOF THE REPORTED CAPABILITY" in text
    assert "setdefault" in text
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call):
            target = getattr(node.func, "attr", getattr(node.func, "id", ""))
            assert target not in ("setattr", "monkeypatch"), target
        if isinstance(node, ast.Assign):
            for assigned in node.targets:
                assert getattr(assigned, "attr", "") != "get_device_capability"


# The optional dependency, on every box.

def test_a_missing_package_names_both_installs():
    with pytest.raises(UnsupportedModelError) as excinfo:
        guards.refuse_missing_package("the sol-attn package")
    message = str(excinfo.value)
    assert "subdirectory=techniques/sparse_backends" in message
    assert "apache-tvm-ffi" in message
    assert "EVERY box in the mesh" in message


def test_the_driver_refuses_before_dispatch_when_one_rank_lacks_the_package():
    """A rank-asymmetric raise inside a collective abandons the lease."""
    from dgx_monarch.nodes import render_preflight

    handle = types.SimpleNamespace(worker_capabilities=[
        {"host": "box-a", "rank": 0, "sol_attn": True},
        {"host": "box-b", "rank": 1, "sol_attn": False},
    ])
    topo = types.SimpleNamespace(ulysses=2, ring=1)
    with pytest.raises(UnsupportedModelError) as excinfo:
        render_preflight.preflight_sol_attention(
            types.SimpleNamespace(unet_name="x.safetensors"), topo,
            "SOL_ATTN_TAU1.0", handle)
    message = str(excinfo.value)
    assert "box-b" in message and "box-a" not in message
    assert "strands its peers" in message


def test_the_driver_preflight_ignores_every_other_kernel():
    from dgx_monarch.nodes import render_preflight

    handle = types.SimpleNamespace(worker_capabilities=[{"sol_attn": False}])
    render_preflight.preflight_sol_attention(
        types.SimpleNamespace(unet_name="x.safetensors"),
        types.SimpleNamespace(ulysses=1, ring=1), "SAGE_AUTO", handle)


def test_the_driver_refuses_a_single_gpu_selection():
    from dgx_monarch.nodes import render_preflight

    with pytest.raises(UnsupportedModelError) as excinfo:
        render_preflight.preflight_sol_attention(
            types.SimpleNamespace(unet_name="x.safetensors"),
            types.SimpleNamespace(ulysses=1, ring=1), "SOL_ATTN_TAU1.0", None)
    assert "sequence-parallel degree of 1" in str(excinfo.value)


def test_the_init_node_refuses_the_selection_before_it_sets_a_worker_up(monkeypatch):
    """The eager-setup sites reach a worker before any render does.

    An explicit preset that keeps stock attention has to refuse at the node the
    operator is looking at. Inside setup the same fact unwinds through the
    rollback, which tells nobody what to change.
    """
    from dgx_monarch.nodes import init as init_mod

    setups: list = []
    handle = types.SimpleNamespace(
        world=2, config=None, config_fingerprint="local",
        ensure_setup=lambda *args, **kwargs: setups.append(args))
    monkeypatch.setattr(init_mod, "get_mesh", lambda **_kwargs: handle)

    with pytest.raises(UnsupportedModelError) as excinfo:
        init_mod.DGXMonarchInit().init(
            topology="dp2", mode="local", attention="SOL_ATTN_TAU1.0")
    assert "sequence-parallel degree of 1" in str(excinfo.value)
    assert setups == [], "the refusal has to land before setup, not inside it"


def test_the_loader_node_refuses_the_selection_before_it_loads(monkeypatch):
    """The loader is the other eager-setup site, and it loads weights.

    Same fact, same refusal, one node later: under an explicit preset the load
    is eager, so a selection this topology never injects must refuse before the
    checkpoint is pushed to a rank.
    """
    from dgx_monarch.nodes import loader_preflight, loaders
    from dgx_monarch.nodes.common import MeshSpec

    calls: list = []
    handle = types.SimpleNamespace(
        world=2, call_all=lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint",
                        lambda *_args, **_kwargs: object())
    monkeypatch.setattr(loaders, "ensure_live", lambda *_args, **_kwargs: handle)
    mesh = MeshSpec(handle=handle, topology_preset="dp2",
                    attention="SOL_ATTN_TAU1.0", sync_ulysses=True)

    with pytest.raises(UnsupportedModelError) as excinfo:
        loaders.DGXMonarchUNETLoader().load(mesh, "x.safetensors")
    assert "sequence-parallel degree of 1" in str(excinfo.value)
    assert calls == [], "the refusal has to land before the eager load"


def test_the_fleet_path_refuses_the_selection_before_worker_setup(monkeypatch):
    """Fleet is the third setup site, and every job it places is world-1.

    Sol-Attn can never host that degree, so the selection refuses at the
    driver before any worker setup token is taken; inside setup the same fact
    unwinds through the rollback with no name on it.
    """
    from dgx_monarch.nodes import fleet as fleet_mod

    setups: list = []
    monkeypatch.setattr(fleet_mod.mesh_setup, "ensure_request_setup",
                        lambda *args, **kwargs: setups.append(args))
    handle = types.SimpleNamespace(world=2)
    mesh = types.SimpleNamespace(attention="SOL_ATTN_TAU1.0", sync_ulysses=True)
    spec = types.SimpleNamespace(mesh=mesh)

    with pytest.raises(UnsupportedModelError) as excinfo:
        fleet_mod.DGXMonarchFleetKSampler()._fleet_bound(
            spec, {}, [], 1, 1.0, "euler", "simple", handle, object())
    assert "sequence-parallel degree of 1" in str(excinfo.value)
    assert setups == [], "the refusal has to land before the setup token"


def test_workers_report_the_capability_and_the_mesh_retains_it():
    text = (SRC / "actor/worker_env.py").read_text()
    assert '"sol_attn": sol_attn_available(),' in text
    mesh = (SRC / "mesh_setup.py").read_text()
    assert "handle.worker_capabilities = reports" in mesh


# The class-k waiver route.

def test_the_kind_is_class_k_and_can_never_be_automatic():
    assert KIND_CLASS[KIND] == "K"
    from dgx_monarch.consent_kinds import KIND_SPECS

    spec = KIND_SPECS[KIND]
    assert spec.auto_eligible is False
    assert spec.wired is True
    assert spec.measured  # class K must carry its measurement
    assert spec.default_guard == KIND_GUARD_PREFIXES[KIND] == "sol_attn"
    assert KIND in accuracy_waiver.KNOWN_WRONG_KINDS


def test_the_family_scoped_guard_belongs_to_the_kind():
    guard = sol.SOL_ATTN_H3_GUARD
    assert sol.SOL_WAIVER_GUARDS["minimax_h3"] == guard
    assert GUARDS[guard].refusal_class is RefusalClass.KNOWN_WRONG
    assert GUARDS[guard].consent_kind == KIND
    assert GUARDS[guard].waivable_now is True
    assert check_guard(KIND, guard) == guard
    assert accuracy_waiver.KNOWN_WRONG_GUARDS[guard] == KIND


def test_a_render_without_a_grant_refuses_and_offers_the_card(monkeypatch):
    monkeypatch.setattr(accuracy_waiver, "waived", lambda _guard: False)
    with pytest.raises(UnsupportedModelError) as excinfo:
        sol._assert_waived("minimax_h3")
    message = str(excinfo.value)
    assert "[dgxm:K guard=sol_attn:minimax_h3 waivable=1]" in message
    assert "NOT identity preserving" in message
    assert "DGXM_WAIVE_KNOWN_WRONG" in message
    assert "TROUBLESHOOTING.md #84" in message


def test_a_granted_waiver_lets_the_render_proceed(monkeypatch):
    seen = []
    monkeypatch.setattr(accuracy_waiver, "waived",
                        lambda guard: seen.append(guard) or True)
    sol._assert_waived("minimax_h3")
    assert seen == ["sol_attn:minimax_h3"]


def test_the_waiver_is_checked_on_every_call_not_once_per_build():
    """Grants and revocations must apply per render, like the ring-pad guard."""
    text = (SRC / "adapters/sol_attention.py").read_text()
    tree = ast.parse(text)
    run = next(node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == "_run")
    assert "_waiver_required" in ast.dump(run)
    assert "@torch.compiler.disable" in text


def test_an_unvouched_tau_takes_the_waiver_route_and_never_downgrades():
    text = (SRC / "adapters/sol_attention.py").read_text()
    assert "return not tau_is_vouched" in text
    for fallback in ("SAGE_AUTO\"", "TORCH_FLASH\"", "make_usp_attention("):
        assert fallback not in text, "the lever must refuse, never substitute"


# Dispatch never falls back under a sol name.

def test_a_typed_refusal_never_keeps_the_previous_kernel():
    text = (SRC / "actor/attention_dispatch.py").read_text()
    assert "except UnsupportedModelError:" in text
    tree = ast.parse(text)
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
    for node in handlers:
        names = [ast.unparse(h.type) for h in node.handlers if h.type is not None]
        if "UnsupportedModelError" in names:
            assert names.index("UnsupportedModelError") < names.index("Exception")


def test_configure_never_swallows_a_typed_sol_refusal(monkeypatch):
    """The swallow path keeps the previous kernel; a typed refusal must not.

    A same-generation kernel that fails to build may keep what is running. A
    refusal is an answer about this render, and the driver has already stamped
    its capability context with the kernel it asked for, so keeping flash under
    a sol name would render one kernel's math under another kernel's vouch.
    """
    from dgx_monarch.actor import attention_dispatch as dispatch_mod

    dispatch = dispatch_mod._AttentionDispatch()
    topology = {"ulysses": 2, "ring": 1, "cfg": 1, "dp": 1}
    monkeypatch.setattr("dgx_monarch.adapters.make_usp_attention",
                        lambda *_a, **_k: object())
    dispatch.configure("TORCH_FLASH", True, topology=topology, world=2,
                       setup_generation=1)
    established = dispatch._impl

    def refuse(*_args, **_kwargs):
        raise UnsupportedModelError("[dgxm:P] no")

    monkeypatch.setattr("dgx_monarch.adapters.sol_attention.make_sol_usp_attention",
                        refuse)
    with pytest.raises(UnsupportedModelError):
        dispatch.configure("SOL_ATTN_TAU1.0", True, topology=topology, world=2,
                           setup_generation=1)
    assert dispatch._impl is established  # unchanged, and never relabelled
    assert dispatch._kernel == "TORCH_FLASH"


# The ceremony does not run under this kernel.

def test_the_skip_reason_names_only_this_kernel():
    assert sol.sol_ceremony_skip_reason("TORCH_FLASH") is None
    assert sol.sol_ceremony_skip_reason("SAGE_AUTO") is None
    reason = sol.sol_ceremony_skip_reason("SOL_ATTN_TAU1.0")
    assert reason and "compares residency, not kernels" in reason


def test_the_auto_gate_returns_no_ceremony_and_writes_no_row():
    """No row: an unproven context already denies the risky levers.

    Writing an INCONCLUSIVE row would add a record without adding a decision,
    and `row_grants_skip` admits only the no-material kind anyway.
    """
    text = (SRC / "nodes/auto_gate.py").read_text()
    assert "sol_ceremony_skip_reason(resolved_attention)" in text
    tree = ast.parse(text)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "auto_gate_context")
    body = ast.unparse(fn)
    skip = body[body.index("skip = sol_ceremony_skip_reason"):]
    guarded = skip[:skip.index("\n    key,")] if "\n    key," in skip else skip[:400]
    assert "return None" in guarded
    for forbidden in ("record", "GateLedger", "quarantine"):
        assert forbidden not in guarded, f"the skip must not {forbidden}"


def test_an_aborted_ceremony_on_this_guard_leaves_both_levers_alone():
    """Mirrors the class-U rescue exemption: one named guard, read from the tag."""
    text = (SRC / "nodes/gate_ceremony.py").read_text()
    assert "is_sol_waiver_refusal(exc)" in text
    assert "not fsdp_proof_scope and not sol_waiver_abort" in text


def test_the_abort_exemption_reads_the_tag_not_the_message():
    from dgx_monarch.adapters.base import UnsupportedModelError as Refusal

    with pytest.raises(Refusal) as raised:
        sol._assert_waived("minimax_h3")
    assert sol.is_sol_waiver_refusal(Exception(str(raised.value)))
    # A ring-pad refusal is someone else's guard and claims nothing here.
    from dgx_monarch.adapters import base

    accuracy_waiver.clear()
    with pytest.raises(Refusal) as ring:
        base.assert_ulysses_only_padding(2, 1)
    assert not sol.is_sol_waiver_refusal(Exception(str(ring.value)))
    # An untagged crash that merely mentions the guard claims nothing either.
    assert not sol.is_sol_waiver_refusal(
        RuntimeError("CUDA error while running sol_attn:minimax_h3"))


def test_the_manual_gate_node_reports_not_run_instead_of_rendering(monkeypatch):
    from dgx_monarch.nodes import gate as gate_runtime
    from dgx_monarch.nodes.gate_node import DGXMonarchIdentityGate

    def never(*_args, **_kwargs):
        raise AssertionError("the ceremony must not run under a sol kernel")

    monkeypatch.setattr(gate_runtime, "run_identity_ceremony", never)
    monkeypatch.setattr(gate_runtime, "conditioning_for_wire", lambda value: value)
    model = types.SimpleNamespace(
        mesh=types.SimpleNamespace(attention="SOL_ATTN_TAU1.0"))
    latent = {"samples": object()}
    node = DGXMonarchIdentityGate()
    returned, report = node.gate(model, [], [], latent, 42, 2, 1.0, "euler",
                                 "simple")
    assert returned is latent  # nothing was rendered, so nothing is returned
    assert '"verdict": "NOT RUN"' in report
    assert "compares residency, not kernels" in report
    with pytest.raises(RuntimeError, match="NOT RUN"):
        node.gate(model, [], [], latent, 42, 2, 1.0, "euler", "simple",
                  strict=True)


def test_the_tau_set_documents_the_instrument_that_could_admit_one():
    """A ceremony can never admit a tau; only a cross-kernel measurement can."""
    doc = sol.tau_is_vouched.__doc__
    assert "NOT the first-use identity ceremony" in doc.replace("\n    ", " ")
    flat = doc.replace("\n    ", " ")
    assert "CROSS-KERNEL" in flat and "IDENTICAL topology" in flat
    assert "0.10 floor" in flat


def test_the_resolved_backend_is_logged_because_two_benches_ran_the_wrong_path():
    """A Triton run and a CuTe run are not the same claim, and look alike."""
    text = (SRC / "adapters/sol_attention.py").read_text()
    assert "get_sol_attn_backend(torch.cuda.current_device())" in text
    assert "backend=%s" in text
