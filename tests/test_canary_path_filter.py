"""The comfy canary's pull-request filter lists what its steps import.

A pull request runs the canary only when it touches a path in the filter. The
real-ComfyUI cases skip in ordinary CI, so a change to a harness they import
reaches comfy master only through this job. Every repo file outside src/ that
a step's test file imports, directly or through another such file, must be in
the filter. In the other direction, a test file in the filter must be one a
step runs or one of those imports, so the list carries no dead entries.

The check reads import statements, so it sees an import by name and nothing
loaded another way. It reads each file a step names whole. In a file the steps
only import, it skips the bodies of test functions, since pytest runs those
only in the files a step names.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "comfy-canary.yml"
# pytest loads conftest.py itself; no step names it and nothing imports it.
_IMPLICIT = {"tests/conftest.py"}
_STEP_FILE = re.compile(r"(?<![\w/])(tests/[\w/]+\.py)")
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _path_filter(workflow: dict) -> list[str]:
    # PyYAML reads the bare key `on` as the boolean True.
    triggers = workflow.get("on", workflow.get(True))
    return list(triggers["pull_request"]["paths"])


def _step_files(workflow: dict) -> set[str]:
    files = set()
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            run = step.get("run") or ""
            code = "\n".join(line for line in run.splitlines()
                             if not line.strip().startswith("#"))
            files.update(_STEP_FILE.findall(code))
    return files


def _resolve(parts: list[str], bases: list[Path]) -> str | None:
    for base in bases:
        for candidate in (base.joinpath(*parts).with_suffix(".py"),
                          base.joinpath(*parts, "__init__.py")):
            if candidate.is_file():
                return candidate.relative_to(REPO).as_posix()
    return None


def _unrun_tests(tree: ast.Module) -> set[int]:
    """The test functions pytest collects, by node id. pytest runs them only
    in a file a step names, never in a file a step merely imports."""
    tests = {id(node) for node in tree.body
             if isinstance(node, _FUNCTIONS) and node.name.startswith("test")}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            tests.update(id(member) for member in node.body
                         if isinstance(member, _FUNCTIONS) and member.name.startswith("test"))
    return tests


def _nodes(tree: ast.Module, skipped: set[int]):
    pending: list[ast.AST] = [tree]
    while pending:
        node = pending.pop()
        if id(node) not in skipped:
            yield node
            pending.extend(ast.iter_child_nodes(node))


def _imported_files(relative: str, *, root: bool) -> set[str]:
    """Repo files one file imports, at any depth in its body.

    pytest puts tests/ on sys.path and the sweep tests add the repo root, so a
    bare name resolves beside the file, then under tests/, then at the root.
    Canary code also puts benchmark/ on sys.path (the cfg combine seam in
    comfy_seam_contracts.py, for gates.py) and tools/ (template_widget_canary.py
    and benchmark/sweep/convert.py, for gen_templates.py), so a bare name
    resolves there last. The installed package lives under src/, which the
    filter covers whole, so its imports resolve to nothing here.

    A step root is read whole. In any other file the bodies of its test
    functions are skipped; its module level, helpers and fixtures are still
    read, because a step can call them.
    """
    path = REPO / relative
    bare_bases = [path.parent, REPO / "tests", REPO, REPO / "benchmark", REPO / "tools"]
    tree = ast.parse(path.read_text())
    found = set()
    for node in _nodes(tree, set() if root else _unrun_tests(tree)):
        if isinstance(node, ast.Import):
            names = [(alias.name.split("."), bare_bases) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package = path.parent
                for _ in range(node.level - 1):
                    package = package.parent
                stem = list(package.relative_to(REPO).parts)
                bases = [REPO]
            else:
                stem, bases = [], bare_bases
            module = stem + (node.module.split(".") if node.module else [])
            names = [(module, bases)] + [
                ([*module, alias.name], bases) for alias in node.names]
        else:
            continue
        for parts, bases in names:
            for depth in range(1, len(parts) + 1):
                resolved = _resolve(parts[:depth], bases)
                if resolved is not None:
                    found.add(resolved)
    return found


def _closure(roots: set[str]) -> set[str]:
    seen: set[str] = set()
    pending = sorted(roots)
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        pending.extend(sorted(_imported_files(current, root=current in roots) - seen))
    return seen


def _covered(path: str, patterns: list[str]) -> bool:
    return any(path == pattern
               or (pattern.endswith("/**") and path.startswith(pattern[:-2]))
               for pattern in patterns)


def test_every_import_of_a_canary_step_triggers_the_canary():
    workflow = _workflow()
    patterns = _path_filter(workflow)
    roots = _step_files(workflow)
    assert "tests/test_ideogram4_stock_equivalence.py" in roots, (
        "the step parser found no pytest files; the check would pass vacuously")
    missing = sorted(path for path in _closure(roots)
                     if not _covered(path, patterns))
    assert not missing, (
        "a canary step imports these files, so a change to them can break the "
        f"canary, but the pull-request filter does not list them: {missing}")


def test_every_listed_test_file_is_run_or_imported_by_a_step():
    workflow = _workflow()
    reachable = _closure(_step_files(workflow)) | _IMPLICIT
    dead = sorted(
        pattern for pattern in _path_filter(workflow)
        if pattern.endswith(".py")
        and pattern.startswith(("tests/", "benchmark/"))
        and pattern not in reachable)
    assert not dead, (
        f"no canary step runs or imports these filter entries: {dead}")


def test_the_pad_exclusion_harness_is_a_trigger_for_its_importers():
    """The 2026-10-12 review read this entry as a file no step runs. It is the
    two-rank harness four real-comfy files import, so it stays a trigger."""
    roots = _step_files(_workflow())
    importers = {
        root for root in roots
        if "tests/test_flux_pad_exclusion.py" in _imported_files(root, root=True)}
    assert importers >= {
        "tests/test_ernie_stock_equivalence.py",
        "tests/test_hunyuan_stock_equivalence.py",
        "tests/test_ideogram4_stock_equivalence.py",
        "tests/test_lens_comfy_exactness.py",
    }
    assert "tests/test_flux_pad_exclusion.py" not in roots
    assert "tests/test_flux_pad_exclusion.py" in _path_filter(_workflow())


def test_a_path_loaded_module_triggers_and_an_unrun_test_import_does_not():
    """The 2026-10-13 review found two gaps. The cfg combine seam loads
    benchmark/gates.py with benchmark/ on sys.path, which the guard did not
    resolve. test_sweep_matrix.py imports report.py only inside test
    functions, which no step runs, so report.py was a dead trigger."""
    workflow = _workflow()
    closure = _closure(_step_files(workflow))
    assert "benchmark/gates.py" in closure
    assert "tools/gen_templates.py" in closure
    assert "benchmark/sweep/report.py" in _imported_files("tests/test_sweep_matrix.py", root=True)
    assert "benchmark/sweep/report.py" not in _imported_files("tests/test_sweep_matrix.py", root=False)
    assert "benchmark/sweep/report.py" not in closure
    assert "benchmark/gates.py" in _path_filter(workflow)
    assert "benchmark/sweep/report.py" not in _path_filter(workflow)
