"""benchmark/reports/wan_i2v_fidelity_matrix.toml: structure and settings checks.

CPU only, with no comfy import, so a typo in the matrix fails here instead of
on hardware.
"""
from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MATRIX_PATH = REPO / "benchmark" / "reports" / "wan_i2v_fidelity_matrix.toml"

SPEC = importlib.util.spec_from_file_location(
    "dgxm_wan_i2v_matrix_run_matrix", REPO / "benchmark" / "run_matrix.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load benchmark/run_matrix.py as a module")
run_matrix = importlib.util.module_from_spec(SPEC)
_argv_before = list(sys.argv)
_path_before = list(sys.path)
SPEC.loader.exec_module(run_matrix)
if sys.argv != _argv_before or sys.path != _path_before:
    raise RuntimeError("loading run_matrix.py mutated sys.argv or sys.path")


def _load() -> dict:
    with open(MATRIX_PATH, "rb") as fh:
        return tomllib.load(fh)


def test_matrix_parses():
    doc = _load()
    if "defaults" not in doc or "case" not in doc:
        raise AssertionError("matrix must have [defaults] and at least one [[case]]")
    if not doc["case"]:
        raise AssertionError("matrix has no cases")


def test_case_names_unique():
    doc = _load()
    names = [case["name"] for case in doc["case"]]
    if len(names) != len(set(names)):
        raise AssertionError(f"duplicate case names: {names}")


def test_references_point_at_earlier_case():
    doc = _load()
    seen: set[str] = set()
    for case in doc["case"]:
        ref = case.get("reference")
        if ref is not None and ref not in seen:
            raise AssertionError(
                f"case {case['name']!r} references {ref!r}, which is not an "
                "earlier case in the file (typo, or ordered after this cell?)"
            )
        seen.add(case["name"])


def test_dp2_reference_cells_are_shaped_for_the_gate():
    """A case another case names in reference= is the single-GPU-equivalent
    reference: topology=dp2, batch=2 and steps=1. run_case notes that a dp
    reference cell needs batch equal to the dp degree. _fidelity_gate scores a
    one-step candidate with gate_step_fidelity, not the strict same-math gate
    (benchmark/gates.py), so its reference must run one step too."""
    doc = _load()
    defaults = doc["defaults"]
    by_name = {case["name"]: case for case in doc["case"]}
    referenced = {case["reference"] for case in doc["case"] if case.get("reference")}
    for name in referenced:
        merged = {**defaults, **by_name[name]}
        if merged.get("topology") != "dp2":
            raise AssertionError(f"reference case {name!r} must set topology='dp2'")
        if int(merged.get("batch", 1)) != 2:
            raise AssertionError(f"reference case {name!r} must set batch=2")
        if int(merged.get("steps", 0)) != 1:
            raise AssertionError(
                f"reference case {name!r} must set steps=1 for the NRMS gate to trigger"
            )


def test_gate_cells_use_uly2_only():
    """No cfg2 or ring2 wan cell: AUTO_TABLE defines only ulysses=2 for wan."""
    doc = _load()
    for case in doc["case"]:
        if case.get("reference") and case.get("topology") != "uly2":
            raise AssertionError(
                f"case {case['name']!r} gates against a reference but topology "
                f"is {case.get('topology')!r}, not 'uly2'"
            )


def test_every_case_settings_pass_run_matrix_validators():
    """Every case's merged settings must pass run_matrix.py's pure validators."""
    doc = _load()
    defaults = doc["defaults"]
    for case in doc["case"]:
        merged = run_matrix._case_settings(case, defaults)
        run_matrix._sampling_path(merged)
        run_matrix._init_options(merged)
        run_matrix._model_sampling_sd3_shift(merged)
        run_matrix._latent_shape(merged, batch=int(merged.get("batch", 1)))
