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


# Seam 52's source scan, on synthetic trees: every form a new inference-mode,
# inference-flag or grad-mode read upstream could take must come back as a site,
# so the seam goes red on it. The rows are the review's mutations of a ComfyUI
# copy (2026-11-10), each one written as the smallest file that holds it.


def _mode_scanner():
    import importlib.util

    spec = importlib.util.spec_from_file_location("comfy_seam_contracts_scan", SUITE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._inference_mode_sites


MODE_FORMS = {
    "a torch attribute in a method": (
        "comfy/model_base.py",
        "import torch\nclass Lumina2:\n    def extra_conds(self):\n"
        "        return torch.is_inference_mode_enabled()\n",
        {("Lumina2.extra_conds", "is_inference_mode_enabled"): 1}),
    "a renamed from-import": (
        "comfy/model_base.py",
        "from torch import is_inference_mode_enabled as _on\ndef extra_conds():\n"
        "    return _on()\n",
        {("<module>", "is_inference_mode_enabled"): 1}),
    "a bare from-import and its read": (
        "comfy/model_base.py",
        "from torch import is_grad_enabled\ndef process_conds():\n    return is_grad_enabled()\n",
        {("<module>", "is_grad_enabled"): 1, ("process_conds", "is_grad_enabled"): 1}),
    "an alias by assignment": (
        "comfy/model_base.py",
        "import torch\n_on = torch.is_inference_mode_enabled\n",
        {("<module>", "is_inference_mode_enabled"): 1}),
    "torch under another name": (
        "comfy/model_base.py",
        "import torch as _t\ndef extra_conds():\n    return _t.is_inference_mode_enabled()\n",
        {("extra_conds", "is_inference_mode_enabled"): 1}),
    "a tensor's inference flag": (
        "comfy/model_base.py",
        "def extra_conds(cross_attn):\n    return cross_attn.is_inference()\n",
        {("extra_conds", "is_inference"): 1}),
    "a name looked up with getattr": (
        "comfy/model_base.py",
        "import torch\ndef extra_conds():\n"
        "    return getattr(torch, 'is_inference_mode_enabled')()\n",
        {("extra_conds", "is_inference_mode_enabled"): 1}),
    "a decorator": (
        "comfy/utils.py",
        "import torch\n@torch.inference_mode()\ndef tiled():\n    pass\n",
        {("tiled", "inference_mode"): 1}),
    "a directory the first scan left out": (
        "app/__init__.py",
        "import torch\n_X = torch.is_inference_mode_enabled\n",
        {("<module>", "is_inference_mode_enabled"): 1}),
    "the executor package": (
        "comfy_execution/graph_utils.py",
        "def _mode():\n    import torch\n    return torch.inference_mode\n",
        {("_mode", "inference_mode"): 1}),
}


@pytest.mark.parametrize("form", sorted(MODE_FORMS))
def test_the_inference_mode_scan_finds_every_read_form(tmp_path, form):
    relative, source, sites = MODE_FORMS[form]
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text(source)
    found = _mode_scanner()(tmp_path)
    assert found == {(relative, scope, name): count for (scope, name), count in sites.items()}


def test_the_inference_mode_scan_skips_tests_packs_and_data_folders(tmp_path):
    read = "import torch\n_X = torch.is_inference_mode_enabled\n"
    for relative in ("tests/a.py", "tests-unit/b.py", "custom_nodes/pack/c.py", "models/d.py",
                     "input/e.py", "output/f.py", "user/g.py", "temp/h.py", ".ci/i.py",
                     "comfy/tests/j.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(read)
    (tmp_path / "comfy" / "kept.py").write_text(read)
    assert _mode_scanner()(tmp_path) == {("comfy/kept.py", "<module>", "is_inference_mode_enabled"): 1}
