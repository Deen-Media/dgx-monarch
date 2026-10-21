"""run_matrix's optional Skip Layer Guidance keys go through the carrier node.

benchmark/run_matrix.py holds a ModelSpec, not a ModelPatcher, so a cell's
``slg_*`` keys apply ``DGXMonarchSkipLayerGuidance().patch`` the way
``model_sampling_sd3_shift`` applies ``DGXMonarchModelSamplingSD3().patch``.
The hardware ladder runs the shipped CFG 3 recipe through these keys.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.nodes.common import MeshSpec, ModelSpec

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dgxm_benchmark_run_matrix_slg", REPO / "benchmark" / "run_matrix.py")
assert SPEC and SPEC.loader
run_matrix = importlib.util.module_from_spec(SPEC)
_argv_before = list(sys.argv)
_path_before = list(sys.path)
SPEC.loader.exec_module(run_matrix)
assert sys.argv == _argv_before
assert sys.path == _path_before


def _model() -> ModelSpec:
    mesh = MeshSpec(handle=SimpleNamespace(world=2), topology_preset="uly2",
                    attention="TORCH_FLASH", sync_ulysses=True)
    return ModelSpec(mesh=mesh, unet_name="moss_image_sft_bf16.safetensors")


def test_no_slg_key_means_no_node():
    assert run_matrix._skip_layer_guidance_settings({}) is None
    assert run_matrix._skip_layer_guidance_settings({"model_sampling_sd3_shift": 3.0}) is None
    model = _model()
    assert run_matrix._apply_skip_layer_guidance(model, {}) is model


def test_unset_layer_lists_mean_none_and_unset_percents_take_the_node_defaults():
    assert run_matrix._skip_layer_guidance_settings({"slg_single_layers": "22"}) == ("", "22", 0.0, 1.0)
    assert run_matrix._skip_layer_guidance_settings({"slg_end_percent": 1}) == ("", "", 0.0, 1.0)
    assert run_matrix._skip_layer_guidance_settings({
        "slg_double_layers": "7", "slg_single_layers": "8, 9",
        "slg_start_percent": 0.01, "slg_end_percent": 0.15,
    }) == ("7", "8, 9", 0.01, 0.15)


@pytest.mark.parametrize("settings, message", [
    ({"slg_single_layers": 22}, "slg_single_layers must be a string of at most 256 characters"),
    ({"slg_double_layers": "1" * 257}, "slg_double_layers must be a string of at most 256 characters"),
    ({"slg_start_percent": True}, "slg_start_percent must be a finite number from 0 through 1"),
    ({"slg_start_percent": "0"}, "slg_start_percent must be a finite number from 0 through 1"),
    ({"slg_end_percent": 1.5}, "slg_end_percent must be a finite number from 0 through 1"),
    ({"slg_end_percent": float("nan")}, "slg_end_percent must be a finite number from 0 through 1"),
    ({"slg_start_percent": -0.1}, "slg_start_percent must be a finite number from 0 through 1"),
])
def test_invalid_slg_values_refuse_and_name_the_key(settings, message):
    with pytest.raises(ValueError) as caught:
        run_matrix._skip_layer_guidance_settings(settings)
    assert str(caught.value) == message


def test_the_keys_are_applied_once_through_an_injected_patch_in_node_order():
    model, calls = object(), []

    def patch(current, double, single, start, end):
        calls.append((current, double, single, start, end))
        return ("patched",)

    assert run_matrix._apply_skip_layer_guidance(
        model, {"slg_single_layers": "22"}, patch=patch) == "patched"
    assert calls == [(model, "", "22", 0.0, 1.0)]


def test_the_default_patch_is_the_carrier_node_and_reaches_the_wire():
    patched = run_matrix._apply_skip_layer_guidance(_model(), {"slg_single_layers": "22"})
    assert patched.request_dict()["skip_layer_guidance"] == {
        "double_layers": "", "single_layers": "22", "start_percent": 0.0, "end_percent": 1.0}
    off = run_matrix._apply_skip_layer_guidance(_model(), {"slg_single_layers": ""})
    assert off.request_dict() == _model().request_dict()


def test_run_case_applies_the_skip_right_after_the_shift_and_before_topology():
    tree = ast.parse((REPO / "benchmark" / "run_matrix.py").read_text())
    run_case = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_case")
    calls = [ast.unparse(call.func) for call in ast.walk(run_case) if isinstance(call, ast.Call)]
    shift = calls.index("_apply_model_sampling_sd3")
    assert calls[shift + 1] == "_apply_skip_layer_guidance"
    assert calls.index("_apply_skip_layer_guidance") < calls.index("resolve_topology")
