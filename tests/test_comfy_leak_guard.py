"""Exercise conftest's ComfyUI isolation guard through child pytest runs.

The temporary projects mirror this repository: pyproject.toml and __init__.py
at the root, conftest under tests, and no tests/__init__.py. The copied guard
is extracted between its marker comments and prefixed with required imports
and fixtures.

This layout exercises pytest's post-collection root/src path inserts and its
session hook proxy, which excludes the tests conftest. The guard must register
as a separate plugin to reach session-scoped fixtures. Additional checks
inspect the guard registered in this process.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

CONFTEST = Path(__file__).resolve().with_name("conftest.py")
START = "# --- comfy module guard"
END = "# --- end comfy module guard"
PLUGIN = "dgx_monarch_comfy_guard"

PYPROJECT = '''
[tool.pytest.ini_options]
testpaths = ["tests"]
'''

# The repository's root __init__.py, reduced to the sys.path insert it makes
# for the package under src when ComfyUI, or pytest's Package setup, runs it.
ROOT_INIT = '''
import os
import sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
'''

PRELUDE = '''
import os
import sys
import time
import types

import pytest

# A comfy-shaped stub present at collection, so the baseline holds one entry
# a module can remove or replace.
sys.modules.setdefault("comfy_extras", types.ModuleType("comfy_extras"))

# A stub ComfyUI checkout beside the suite: a directory holding
# comfy/options.py, as the guard recognizes one.
CHECKOUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "comfy_checkout")


@pytest.fixture(scope="session", autouse=True)
def _session_wide_stub():
    """A session-scoped fixture spans module boundaries; its entries are credited."""
    sys.modules["comfy_api"] = types.ModuleType("comfy_api")
    sys.path.append("/session/scoped/entry")
    yield
    sys.modules.pop("comfy_api", None)
    sys.path.remove("/session/scoped/entry")


@pytest.fixture(scope="session")
def session_comfy():
    """Import a stub comfy checkout for the rest of the session, and take it back exactly."""
    names = ("comfy", "comfy.options", "nodes")
    before = {name: sys.modules[name] for name in names if name in sys.modules}
    for name in names:
        sys.modules.pop(name, None)
    sys.path.insert(0, CHECKOUT)
    import comfy.options  # noqa: F401
    import nodes  # noqa: F401
    yield sys.modules["comfy"]
    for name in names:
        sys.modules.pop(name, None)
    sys.modules.update(before)
    sys.path.remove(CHECKOUT)


@pytest.fixture(scope="session")
def leaking_session_stub():
    """A session-scoped fixture that never puts back what its setup installed."""
    sys.modules["folder_paths"] = types.ModuleType("folder_paths")
    sys.path.append("/leaked/by/a/session/fixture")
    yield
'''

CHECKOUT_FILES = {
    "comfy/__init__.py": "",
    "comfy/options.py": "args_parsing = False\n",
    "nodes.py": "NODE_CLASS_MAPPINGS = {}\n",
}

CLEAN = '''
import sys
import types

import pytest


@pytest.fixture(scope="module")
def held_stub():
    """A module-scoped stub and path entry, taken back exactly at module end."""
    before = {name: sys.modules[name] for name in ("comfy", "folder_paths") if name in sys.modules}
    sys.modules["comfy"] = types.ModuleType("comfy")
    sys.modules["folder_paths"] = types.ModuleType("folder_paths")
    sys.path.insert(0, "/held/for/this/module")
    yield
    for name in ("comfy", "folder_paths"):
        sys.modules.pop(name, None)
    sys.modules.update(before)
    sys.path.remove("/held/for/this/module")


def test_function_scoped_stub(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy.model_base", types.ModuleType("comfy.model_base"))
    monkeypatch.setitem(sys.modules, "nodes", types.ModuleType("nodes"))
    monkeypatch.syspath_prepend("/function/scoped/entry")


def test_module_stub_is_live(held_stub):
    assert "comfy" in sys.modules and sys.path[0] == "/held/for/this/module"


def test_module_stub_still_live(held_stub):
    assert "folder_paths" in sys.modules


def test_session_stub_is_live():
    assert "comfy_api" in sys.modules and "/session/scoped/entry" in sys.path
'''

LEAKS_STUB = '''
import sys
import types


def test_leaves_a_stub_comfy():
    sys.modules["comfy"] = types.ModuleType("comfy")
    sys.modules["comfy.samplers"] = types.ModuleType("comfy.samplers")
'''

AFTER_LEAK = '''
import sys


def test_namespace_is_clean_again():
    assert "comfy" not in sys.modules
    assert "comfy.samplers" not in sys.modules
'''

ORPHAN = '''
import sys
import types


def test_leaves_an_orphan_submodule():
    sys.modules["comfy.model_base"] = types.ModuleType("comfy.model_base")
'''

PATH_LEAK = '''
import sys


def test_leaves_a_path_entry():
    sys.path.append("/leaked/path/entry")
'''

REMOVES = '''
import sys


def test_removes_a_baseline_entry():
    sys.modules.pop("comfy_extras")
'''

RESTORED = '''
import sys


def test_baseline_entry_is_back():
    assert "comfy_extras" in sys.modules
    assert "/leaked/path/entry" not in sys.path
'''

REPLACES = '''
import sys
import types


def test_replaces_a_baseline_entry():
    sys.modules["comfy_extras"] = types.ModuleType("comfy_extras")
'''

SESSION_IMPORT = '''
import sys

from conftest import CHECKOUT


def test_session_fixture_imports_the_stub_checkout(session_comfy):
    assert sys.modules["comfy"] is session_comfy
    assert sys.modules["comfy.options"].__file__.startswith(CHECKOUT)
    assert sys.modules["nodes"].__file__.startswith(CHECKOUT)
    assert sys.path[0] == CHECKOUT
'''

AFTER_SESSION_IMPORT = '''
import sys

from conftest import CHECKOUT


def test_the_session_fixtures_checkout_is_still_live():
    """Credited to the live fixture: the module boundary before this file raised nothing."""
    assert sys.modules["comfy.options"].__file__.startswith(CHECKOUT)
    assert CHECKOUT in sys.path


def test_a_leaking_session_fixture_is_credited_while_it_lives(leaking_session_stub):
    assert "folder_paths" in sys.modules
'''

SESSION_END = '''
import sys


def test_the_session_fixtures_finish_after_this_test():
    """The last test: every session-scoped fixture is finalized in its teardown."""
    assert "folder_paths" in sys.modules and "/leaked/by/a/session/fixture" in sys.path
'''

FAILS_MID_MODULE = '''
import sys
import types

import pytest


@pytest.fixture(scope="module")
def held_stub():
    sys.modules["comfy"] = types.ModuleType("comfy")
    yield
    sys.modules.pop("comfy", None)


def test_first_fails(held_stub):
    raise AssertionError("stop here with -x while the module fixture is live")


def test_second_never_runs(held_stub):
    pass
'''

TINY = '''
import os
import sys


def test_the_root_package_put_its_src_on_the_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.join(root, "src") in sys.path
'''


def _guard_block() -> str:
    source = CONFTEST.read_text(encoding="utf-8")
    start, end = source.index(START), source.index(END)
    assert start < end
    return source[start:end]


def _layout(root: Path, prelude: str, tests: dict[str, str]) -> Path:
    """The repository's shape: pyproject at the root, the root package, the suite under tests/."""
    root.mkdir()
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (root / "__init__.py").write_text(ROOT_INIT, encoding="utf-8")
    (root / "src").mkdir()
    suite = root / "tests"
    suite.mkdir()
    (suite / "conftest.py").write_text(prelude + "\n" + _guard_block(), encoding="utf-8")
    for name, body in tests.items():
        (suite / name).write_text(textwrap.dedent(body), encoding="utf-8")
    return root


def _project(tmp_path: Path) -> Path:
    """The suite, under a root named as this repository is, beside a stub comfy checkout."""
    root = _layout(tmp_path / "dgx-monarch", PRELUDE, {
        "test_a_clean.py": CLEAN,
        "test_b_leaks_stub.py": LEAKS_STUB,
        "test_c_after_the_leak.py": AFTER_LEAK,
        "test_d_orphan.py": ORPHAN,
        "test_e_path.py": PATH_LEAK,
        "test_f_removes.py": REMOVES,
        "test_g_restored.py": RESTORED,
        "test_h_replaces.py": REPLACES,
        "test_i_session_import.py": SESSION_IMPORT,
        "test_j_after_session_import.py": AFTER_SESSION_IMPORT,
        "test_k_session_end.py": SESSION_END,
        "fails_mid_module.py": FAILS_MID_MODULE,
    })
    for name, body in CHECKOUT_FILES.items():
        target = root / "comfy_checkout" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")
    return root


def _run(project: Path, *args: str) -> str:
    """One child pytest from the project root, with no PYTHONPATH, as a bare run has."""
    env = {name: value for name, value in os.environ.items()
           if name not in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONPATH")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args],
        cwd=project, env=env, text=True, capture_output=True, timeout=120, check=False,
    )
    return completed.stdout + completed.stderr


def test_every_leak_is_named_and_restored_and_clean_modules_pass(tmp_path):
    out = _run(_project(tmp_path))
    # Five leaking modules error at their last test's teardown, named by file.
    assert "ERROR tests/test_b_leaks_stub.py::test_leaves_a_stub_comfy" in out
    assert "tests/test_b_leaks_stub.py left comfy modules or sys.path changed" in out
    assert "added: comfy (stub, no file), comfy.samplers (stub, no file)" in out
    assert "ERROR tests/test_d_orphan.py::test_leaves_an_orphan_submodule" in out
    assert "added: comfy.model_base (stub, no file)" in out
    assert "ERROR tests/test_e_path.py::test_leaves_a_path_entry" in out
    assert "sys.path added: /leaked/path/entry" in out
    assert "ERROR tests/test_f_removes.py::test_removes_a_baseline_entry" in out
    assert "removed: comfy_extras" in out
    assert "ERROR tests/test_h_replaces.py::test_replaces_a_baseline_entry" in out
    assert "replaced: comfy_extras (now stub, no file)" in out
    # A session-scoped fixture that leaks is finalized in the last test's
    # teardown; the verdict there names the fixture beside what it left.
    assert "ERROR tests/test_k_session_end.py::test_the_session_fixtures_finish_after_this_test" in out
    assert "added: folder_paths (stub, no file)" in out
    assert "sys.path added: /leaked/by/a/session/fixture" in out
    assert ("the session-scoped fixture 'leaking_session_stub' finished here without putting "
            "back folder_paths, sys.path entry /leaked/by/a/session/fixture") in out
    # The clean module (function-scoped monkeypatch stubs, a module-scoped
    # stub taken back, a session-scoped stub still live), the modules that
    # read the restored baseline, and the two around the session-scoped
    # import of a stub checkout all pass: no cascade from one leak, and no
    # false positive from a wider fixture or from the root package's own
    # sys.path insert at the first test.
    for clean in ("test_a_clean.py", "test_c_after_the_leak.py", "test_g_restored.py",
                  "test_i_session_import.py", "test_j_after_session_import.py"):
        assert f"ERROR tests/{clean}" not in out, out
    assert "session_comfy" not in out, out
    assert "15 passed, 6 errors" in out, out


def test_a_selection_and_an_early_stop_raise_no_false_positive(tmp_path):
    project = _project(tmp_path)
    selected = _run(project, "-k", "test_namespace_is_clean_again")
    assert "1 passed" in selected and "error" not in selected, selected
    stopped = _run(project, "-x", "tests/fails_mid_module.py")
    assert "1 failed" in stopped and "error" not in stopped, stopped


@pytest.mark.parametrize("name", ["dgx-monarch", "monarch"])
def test_the_root_packages_own_src_insert_is_not_a_leak(tmp_path, name):
    """A bare run, no PYTHONPATH: the root __init__.py inserts src at the first test, after the baseline.

    Without the exemption the guard charges that insert to whichever file runs
    first (2026-10-05: 10 passed, 1 error for tests/test_abandoned_wedge_heal.py alone).
    The module asserts the insert happened, so the exemption, not its absence,
    is what passes.
    Under a root named as this repository is, pytest's Package setup also
    inserts the root; under one named as an identifier it inserts the root's
    parent instead, and both are the checkout's own.
    """
    project = _layout(tmp_path / name, "import os\nimport sys\nimport time\n\nimport pytest\n",
                      {"test_tiny.py": TINY})
    out = _run(project)
    assert "1 passed" in out and "error" not in out, out


def test_the_guard_is_a_plugin_of_its_own_that_the_sessions_hook_proxy_keeps(pytestconfig, request):
    """Why a session-scoped fixture cannot escape the guard, pinned in this process.

    pytest calls the fixture hooks through the hook proxy of the fixture's
    node; for a session- or package-scoped fixture that is the session, whose
    proxy asks pluggy to drop every conftest module not visible from the
    repository root. pluggy's `subset_hook_caller` drops only a plugin that
    has an attribute named for the hook, so a conftest module whose hook has
    a name of its own would also pass the proxy check below, by accident
    alone; the module and name asserts rule that case out. The guard is an
    object registered by name, not a conftest module, so that proxy keeps it
    whatever its methods are called.
    """
    manager = pytestconfig.pluginmanager
    plugin = manager.get_plugin(PLUGIN)
    assert plugin is not None and not isinstance(plugin, types.ModuleType)

    def guards(caller) -> list:
        return [impl for impl in caller.get_hookimpls() if impl.plugin is plugin]

    session = request.session
    at_the_session = session.gethookproxy(session.path)
    for hook, wrapper in (("pytest_fixture_setup", True), ("pytest_fixture_post_finalizer", False),
                          ("pytest_runtest_teardown", True), ("pytest_collection_finish", False)):
        (guard,) = guards(getattr(pytestconfig.hook, hook))
        assert guard.wrapper is wrapper and guard.function.__name__.startswith("pytest_comfy_guard_"), hook
        assert guard.function.__name__ != hook
        assert guards(getattr(at_the_session, hook)) == [guard], hook
    # The teardown wrapper runs around pytest's own teardown implementation,
    # inside which every fixture of the module, and of the session at the last
    # item, is finalized.
    (runner,) = [impl for impl in pytestconfig.hook.pytest_runtest_teardown.get_hookimpls()
                 if impl.plugin_name == "runner"]
    assert not runner.wrapper and not runner.hookwrapper
    # The baseline the process runs under: taken after collection, with the
    # checkout's root and src as the only path entries the comparison skips.
    root = os.path.realpath(str(pytestconfig.rootpath))
    assert plugin.baseline is not None and plugin.path is not None
    assert plugin.root == root and plugin.root_src == os.path.join(root, "src")
    assert plugin.compared_path([root, os.path.dirname(root), "/", os.path.join(root, "src")]) == []
    assert plugin.compared_path([os.path.join(root, "tests"), "/opt/ComfyUI"]) == [
        os.path.join(root, "tests"), "/opt/ComfyUI"]


def test_the_guard_reads_a_lazy_modules_own_namespace_without_importing(pytestconfig, monkeypatch):
    """A module-level __getattr__ must not run while the guard classifies sys.modules.

    transformers answers any name its namespace lacks by importing a
    submodule. On a hosted runner on 2026-10-05 the guard's getattr for
    __path__ ran one of those imports from inside teardown and errored 614
    tests. The guard reads the module's own namespace instead, so the lazy
    hook never fires.
    """
    plugin = pytestconfig.pluginmanager.get_plugin(PLUGIN)
    lazy = types.ModuleType("lazy_package_for_the_comfy_guard")

    def __getattr__(name):
        raise AssertionError(f"the guard asked a lazy module for {name}")

    lazy.__getattr__ = __getattr__
    monkeypatch.setitem(sys.modules, lazy.__name__, lazy)
    assert plugin.location(lazy) is None
    assert plugin.watches(lazy.__name__, lazy) is False
    assert plugin.own(object(), "__file__") is None
    assert lazy.__name__ not in plugin.snapshot()
