"""Runs the seam behavioral contracts wherever comfy imports: the comfy-canary
job, or a local run with COMFY_DIR (default ../ComfyUI). Elsewhere
test_seam_contracts_against_comfy skips, because the daily canary workflow
enforces it; the comfy-free tests below run everywhere."""
import ast
import os
import sys
from pathlib import Path

import pytest

from module_location_helpers import from_checkout

REPO = Path(__file__).resolve().parents[1]
SUITE = REPO / "tests" / "canary" / "comfy_seam_contracts.py"


def _is_comfy_module(name):
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


def _from_checkout(module, comfy_dir):
    """Does this module's file live under the checkout?
    module_location_helpers.from_checkout says why the unprefixed modules a
    real comfy import adds must leave with it."""
    return from_checkout(module, comfy_dir)


def test_seam_contracts_against_comfy():
    preserved_modules = {
        name: module for name, module in sys.modules.items() if _is_comfy_module(name)
    }
    for name in preserved_modules:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    comfy_dir = os.path.abspath(os.environ.get("COMFY_DIR", "../ComfyUI"))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        # comfy.utils reaches comfy.cli_args during import. Ensure the CPU
        # device selection is active before pytest probes that module.
        sys.argv = ["pytest-comfy-seam", "--cpu"]
        options = pytest.importorskip(
            "comfy.options", reason="no ComfyUI on sys.path (canary job covers it)")
        options.enable_args_parsing()
        pytest.importorskip("comfy.utils")
        sys.path.insert(0, str(REPO / "tests" / "canary"))
        from comfy_seam_contracts import main

        main()
    finally:
        sys.argv = original_argv
        # Classify first, then pop: a namespace package's path re-resolves
        # through its parent in sys.modules while it is being read.
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or _from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved_modules)
        sys.path[:] = original_path


# The comfy-free half checks the suite's own shape on every CI matrix entry, so
# a seam that is declared and never called, or a transaction that stops
# asserting, fails before the daily canary runs.


def _suite_tree():
    return ast.parse(SUITE.read_text(), filename=str(SUITE))


def _seam_entries(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id != "SEAMS" or not isinstance(node.value, ast.Tuple):
            continue
        for element in node.value.elts:
            assert isinstance(element, ast.Tuple) and len(element.elts) == 2
            name, function = element.elts
            assert isinstance(name, ast.Constant) and isinstance(function, ast.Name)
            yield name.value, function.id
        return
    raise AssertionError("tests/canary/comfy_seam_contracts.py declares no SEAMS table")


def test_top_level_inventory_matches_the_executable_seam_table():
    """The canonical run-order prose must enumerate every SEAMS row exactly."""
    tree = _suite_tree()
    docstring = ast.get_docstring(tree)
    assert docstring is not None
    inventory = docstring.split("The seams, in run order:", 1)[1].split(
        "\n\nRuns in the comfy-canary workflow", 1
    )[0]
    documented = []
    for line in inventory.splitlines():
        number, separator, rest = line.strip().partition(". ")
        if separator and number.isdigit():
            documented.append((int(number), rest.split("  ", 1)[0]))
    executable = [
        (number, name)
        for number, (name, _function) in enumerate(_seam_entries(tree), start=1)
    ]
    assert documented == executable, (
        f"documented seam inventory {documented} != executable table {executable}"
    )


def test_every_declared_seam_names_a_real_transaction_that_asserts():
    tree = _suite_tree()
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    seams = list(_seam_entries(tree))
    assert len(seams) >= 10, (
        "the seam inventory shrank; a seam is only retired when the pack stops "
        f"riding it (docs/DESIGN.md 5.10), got {[name for name, _ in seams]}"
    )
    assert len({name for name, _ in seams}) == len(seams), "duplicate seam names"
    for name, attribute in seams:
        assert name and attribute in functions, f"seam {name!r} names no transaction"
        body = functions[attribute]
        asserts = [node for node in ast.walk(body) if isinstance(node, ast.Assert)]
        assert len(asserts) >= 3, (
            f"seam {name!r} carries {len(asserts)} assertions; a transaction that "
            "does not assert is a green run that proves nothing"
        )
        assert ast.get_docstring(body), f"seam {name!r} does not say what it pins"


def test_every_transaction_written_is_wired_into_the_table():
    """A transaction the SEAMS table never names never runs.

    Under the fix-extends-canary rule (docs/DESIGN.md 5.10) a fix adds a
    transaction; if its SEAMS row is forgotten, the daily canary stays green
    over a contract nobody checks.
    """
    tree = _suite_tree()
    written = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("_assert_")
        and node.name.endswith("_seam")
    }
    assert written, "the suite defines no _assert_*_seam transactions"
    wired = {attribute for _name, attribute in _seam_entries(tree)}
    assert written == wired, (
        f"transactions the SEAMS table never runs: {sorted(written - wired)}; "
        f"table rows naming no transaction: {sorted(wired - written)}"
    )


def test_every_seam_failure_names_its_seam():
    """A drift page must say which contract moved, not only that one did."""
    tree = _suite_tree()
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    unnamed = []
    for name, attribute in _seam_entries(tree):
        # The message may be a plain string, an implicit concatenation, or an
        # f-string; ast.unparse flattens all three for one substring test.
        for node in ast.walk(functions[attribute]):
            if isinstance(node, ast.Assert) and node.msg is not None:
                if name.split()[0] not in ast.unparse(node.msg):
                    unnamed.append(f"{attribute}: {ast.unparse(node.msg)[:70]}")
    assert not unnamed, (
        "assertion messages that do not name their seam:\n" + "\n".join(unnamed)
    )


def test_minimax_audio_seam_drives_the_stock_outer_forward_around_the_adapter():
    """Keep the H3 regression behavioral, not a removed-symbol existence check."""
    tree = _suite_tree()
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_assert_minimax_h3_audio_carry_seam"
    )
    source = ast.unparse(function)
    assert "MiniMaxH3Model.forward" in source
    assert "MiniMaxH3Adapter().inject_usp" in source
    assert "model.forward" in source
    assert "torch.allclose" in source and "torch.equal" in source
    assert "MixedLegacyOuter" in source
    assert "partial or mixed ComfyUI update" in source
    assert "time_shift_slope" not in source
