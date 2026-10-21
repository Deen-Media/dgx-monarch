"""Without its optional dependencies, collection leaves the Studio suite out
instead of stopping the whole run with import errors."""
from __future__ import annotations

import ast
import functools
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# Hide modules the way a missing install does (find_spec returns None) or the
# way an import hook refuses them (find_spec raises), then collect.
COLLECT = """
import importlib.abc, sys
mode, hidden = sys.argv[1], {name for name in sys.argv[2].split(",") if name}
if mode == "absent":
    for name in hidden:
        sys.modules[name] = None
else:
    class Refuse(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.partition(".")[0] in hidden:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
    sys.meta_path.insert(0, Refuse())
import pytest
sys.exit(pytest.main(sys.argv[3:]))
"""


def _module_scope_imports(path: Path):
    """Modules a test file imports outside its functions and classes."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for statement in ast.walk(node):
            if isinstance(statement, ast.Import):
                yield from (alias.name for alias in statement.names)
            elif isinstance(statement, ast.ImportFrom) and statement.module and not statement.level:
                yield statement.module
                yield from (f"{statement.module}.{alias.name}" for alias in statement.names)


def _reaches_studio(name: str) -> bool:
    # Pillow alone is left out: the brand and card tests import it directly,
    # predate Studio and need the dev extra's CairoSVG as well.
    return name.partition(".")[0] == "aiohttp" or (name + ".").startswith("dgx_monarch.studio.")


@functools.cache
def _studio_reachers() -> tuple[str, ...]:
    """The Studio modules and every other test module that imports aiohttp or
    the Studio package outside its functions."""
    return tuple(sorted({"test_studio_*.py"} | {
        path.name for path in (REPO / "tests").glob("test_*.py")
        if not path.name.startswith("test_studio_")
        and any(_reaches_studio(name) for name in _module_scope_imports(path))}))


@functools.cache
def _collect(mode: str, hidden: str) -> tuple[int, frozenset[str], frozenset[str], str]:
    """Exit code, modules with collection errors, modules collected, output."""
    path = os.pathsep.join(filter(None, [str(REPO / "src"), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", COLLECT, mode, hidden, "tests", "--collect-only", "-q",
         "-p", "no:cacheprovider", "-o", "python_files=" + " ".join(_studio_reachers())],
        cwd=REPO, env={**os.environ, "PYTHONPATH": path, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr
    errors = frozenset(re.findall(r"^ERROR (tests/\S+\.py)", output, re.MULTILINE))
    collected = frozenset(re.findall(r"^(tests/\S+\.py)::", output, re.MULTILINE))
    return result.returncode, errors, collected, output


@pytest.mark.parametrize("mode", ["absent", "refused"])
@pytest.mark.parametrize("hidden", ["aiohttp", "PIL"])
def test_collection_adds_no_errors_without_a_studio_dependency(mode, hidden):
    _code, baseline_errors, baseline_collected, _output = _collect("absent", "")
    assert any(name.startswith("tests/test_studio_") for name in baseline_collected)
    code, errors, collected, output = _collect(mode, hidden)
    # Errors this environment already has (for example a missing CairoSVG)
    # are not this guard's concern; a hidden Studio dependency adds none.
    assert errors <= baseline_errors, output
    assert not any(name.startswith("tests/test_studio_") for name in collected), output
    studio_modules = len(list((REPO / "tests").glob("test_studio_*.py")))
    assert (f"Left out {studio_modules} module(s) matching tests/test_studio_*.py: "
            f"{hidden} not installed") in output
    if not baseline_errors:
        # 5 means every collected module was left out or skipped.
        assert code in {0, 5}, output


def test_collection_runs_the_studio_suite_when_its_dependencies_are_installed():
    pytest.importorskip("aiohttp")
    pytest.importorskip("PIL")
    _code, _errors, collected, output = _collect("absent", "")
    assert "tests/test_studio_backend.py" in collected
    assert "matching tests/test_studio_*.py" not in output


def test_collection_adds_no_errors_without_rich():
    """The parity test reaches rich through dgx_monarch.tui.app, so without rich
    it is left out with the TUI record test instead of stopping collection."""
    _code, baseline_errors, _baseline, _output = _collect("absent", "")
    code, errors, collected, output = _collect("absent", "rich")
    assert errors <= baseline_errors, output
    for module in ("test_studio_presentation_parity.py", "test_tui_record_security.py"):
        assert f"tests/{module}" not in errors | collected, output
        assert f"Left out 1 module(s) matching tests/{module}: rich not installed" in output
    if not baseline_errors:
        assert code in {0, 5}, output


def test_explicit_non_studio_runs_do_not_report_left_out_studio_modules():
    """The summary counts what collection actually left out, so a run of one
    unrelated file without aiohttp names nothing."""
    path = os.pathsep.join(filter(None, [str(REPO / "src"), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", COLLECT, "absent", "aiohttp", "tests/test_public_docs.py",
         "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=REPO, env={**os.environ, "PYTHONPATH": path, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Left out" not in result.stdout


def test_dev_extra_keeps_the_studio_extra_bounds():
    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]

    def requirement(extra: str, name: str) -> str:
        return next(spec for spec in extras[extra] if re.match(rf"{name}\b", spec, re.IGNORECASE))

    for name in ("aiohttp", "pillow"):
        assert requirement("dev", name) == requirement("studio", name), name


def test_dev_extra_comment_says_what_happens_without_optional_dependencies():
    text = (REPO / "pyproject.toml").read_text()
    comment = text.split("[project.optional-dependencies]\n", 1)[1].split("\ndev = [", 1)[0]
    flat = " ".join(line.lstrip("#").strip() for line in comment.splitlines())
    assert "skip silently" not in flat
    assert "tests/conftest.py leaves out the Studio and TUI test modules that need it" in flat
