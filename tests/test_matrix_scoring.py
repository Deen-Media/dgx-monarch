"""What the matrix runner scores, and in which form (pure CPU math).

The gate itself lives in benchmark/gates.py and is covered in test_gates.py.
This file covers the runner's own scorer: which cell it reads, what it does
when that cell is missing, and the order in which run_case reduces a returned
cond/uncond pair. run_case needs comfy and a live mesh, so the order is read
from the source rather than run.
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmark"))

SPEC = importlib.util.spec_from_file_location(
    "dgxm_benchmark_matrix_scoring", REPO / "benchmark" / "run_matrix.py"
)
assert SPEC and SPEC.loader
run_matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_matrix)


def _reference_row(latent: torch.Tensor) -> dict:
    return {"name": "ref", "status": "ok", "latent": latent}


def test_the_gate_reads_batch_row_zero_of_both_sides():
    torch.manual_seed(0)
    row_zero = torch.randn(1, 4, 16, 16)
    # A dp2 reference renders batch 2 and row 1 is another rank's own sample.
    reference = torch.cat([row_zero, torch.randn(1, 4, 16, 16)])
    candidate = row_zero + 1e-3 * torch.randn_like(row_zero)
    gate = run_matrix._fidelity_gate(
        candidate, {"reference": "ref", "steps": 1}, {"ref": _reference_row(reference)})
    assert gate["pass"], gate["detail"]
    assert gate["reference"] == "ref" and "nrms" in gate["detail"]


def test_the_gate_fails_a_candidate_that_missed_the_floor():
    torch.manual_seed(0)
    reference = torch.randn(1, 4, 16, 16)
    gate = run_matrix._fidelity_gate(
        torch.randn_like(reference), {"reference": "ref", "steps": 1},
        {"ref": _reference_row(reference)})
    assert not gate["pass"]


def test_the_gate_fails_a_reference_no_cell_rendered():
    gate = run_matrix._fidelity_gate(
        torch.zeros(1, 4, 8, 8), {"reference": "absent", "steps": 1}, {})
    assert not gate["pass"] and "not found" in gate["detail"]


def test_the_gate_fails_a_reference_cell_that_produced_no_latent():
    gate = run_matrix._fidelity_gate(
        torch.zeros(1, 4, 8, 8), {"reference": "ref", "steps": 1},
        {"ref": {"name": "ref", "status": "N/A"}})
    assert not gate["pass"] and "N/A" in gate["detail"]


def test_the_gate_takes_the_strict_math_bar_above_one_step():
    reference = torch.zeros(1, 4, 8, 8)
    gate = run_matrix._fidelity_gate(
        reference + 1.0, {"reference": "ref", "steps": 4},
        {"ref": _reference_row(reference)})
    assert not gate["pass"] and "max|diff|" in gate["detail"]


def _run_case_body() -> list[ast.stmt]:
    source = (REPO / "benchmark" / "run_matrix.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    return next(
        node.body for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_case"
    )


def _first_call_line(body: list[ast.stmt], name: str) -> int:
    lines = [
        node.lineno
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]
    assert lines, f"run_case never calls {name}"
    return min(lines)


@pytest.mark.parametrize("reader", ("_latent_statistics", "_latent_sha256"))
def test_run_case_reduces_a_cfg_pair_before_it_attests_the_latent(reader: str):
    # Every reader of the sample latent must see one form, the form the row
    # names, so a cfg pair cannot reach the statistics or the hash unreduced.
    body = _run_case_body()
    assert _first_call_line(body, "scored_latent") < _first_call_line(body, reader)
