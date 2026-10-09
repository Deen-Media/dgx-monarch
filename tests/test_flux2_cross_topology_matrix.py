"""benchmark/flux2_cross_topology_matrix.toml is well formed.

The toml parses and passes run_matrix.py's own case validators, without a
render. It is the candidate side of the offline Flux2 cross-topology
comparison (docs/VALIDATION.md, Flux2 stock-reference record, 2026-07-28).
"""
from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MATRIX = REPO / "benchmark" / "flux2_cross_topology_matrix.toml"

SPEC = importlib.util.spec_from_file_location(
    "dgxm_flux2_matrix_run_matrix", REPO / "benchmark" / "run_matrix.py",
)
assert SPEC and SPEC.loader
run_matrix = importlib.util.module_from_spec(SPEC)
_argv_before = list(sys.argv)
_path_before = list(sys.path)
SPEC.loader.exec_module(run_matrix)
assert sys.argv == _argv_before
assert sys.path == _path_before


def _parsed_matrix() -> dict:
    with MATRIX.open("rb") as f:
        return tomllib.load(f)


def _merged_cases() -> dict[str, dict]:
    matrix = _parsed_matrix()
    defaults = matrix["defaults"]
    return {case["name"]: {**defaults, **case} for case in matrix["case"]}


def test_matrix_parses_with_two_unique_cases():
    matrix = _parsed_matrix()
    names = [case["name"] for case in matrix["case"]]
    assert names == ["flux2_uly2fsdp_512_1step", "flux2_uly2fsdp_1024_20step"]
    assert len(set(names)) == len(names)


def test_cases_carry_the_canonical_flux2_config_verbatim():
    for merged in _merged_cases().values():
        assert merged["unet"] == "flux2-dev.safetensors"
        assert merged["te"] == {
            "name": "mistral_3_small_flux2_bf16.safetensors", "type": "flux2",
        }
        assert merged["topology"] == "uly2+fsdp"
        assert merged["sampling_path"] == "flux2"
        assert merged["seed"] == 42
        assert merged["cfg"] == 1.0
        assert merged["guidance"] == 4.0
        assert merged["sampler"] == "euler"
        assert merged["scheduler"] == "simple"
        assert merged["auto_gate"] == "off"
        assert merged["lora_low_rss"] == "off"
        assert merged["slab_weights"] == "off"
        assert merged["latent_channels"] == 128
        assert merged["latent_downscale"] == 16


def test_cases_validate_against_run_matrix_own_validators_without_rendering():
    for merged in _merged_cases().values():
        assert run_matrix._sampling_path(merged) == "flux2"
        assert run_matrix._init_options(merged) == {
            "auto_gate": "off", "lora_low_rss": "off", "slab_weights": "off",
        }
        assert run_matrix._model_sampling_sd3_shift(merged) is None
        assert run_matrix._latent_downscale(merged) == 16


def test_case_a_is_512_1step_and_case_b_is_1024_20step_with_flux2_latent_shape():
    by_name = _merged_cases()
    a = by_name["flux2_uly2fsdp_512_1step"]
    assert (a["width"], a["height"], a["steps"]) == (512, 512, 1)
    assert run_matrix._latent_shape(a, batch=1) == (1, 128, 32, 32)

    b = by_name["flux2_uly2fsdp_1024_20step"]
    assert (b["width"], b["height"], b["steps"]) == (1024, 1024, 20)
    assert run_matrix._latent_shape(b, batch=1) == (1, 128, 64, 64)


def test_no_case_sets_a_reference_field():
    # The reference leg runs in a standalone single-GPU harness outside this
    # repository, never as an in-matrix `reference` cell: that cell keeps the
    # candidate's mesh and model resident while stock loads beside them, which
    # one box cannot hold (docs/VALIDATION.md, same record).
    for merged in _merged_cases().values():
        assert "reference" not in merged
