"""Exercise conftest's environment isolation guard through a child pytest run.

The generated project mirrors this repository: conftest under tests, no root
conftest, and no tests/__init__.py. Its cases cover environment leaks, the
tests following each leak, and supported writes through fixtures. Each case's
outcome checks that leaks fail and restoration prevents later failures.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

CONFTEST = Path(__file__).with_name("conftest.py")
SRC = Path(__file__).resolve().parents[1] / "src"
PLUGIN = "dgx_monarch_environ_guard"

# First module. Neither the node pack nor cv2 is imported when its first test
# starts, so that test also covers the variables their imports write.
CASES = '''
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _aa_autouse_named_to_sort_first(monkeypatch):
    monkeypatch.setenv("GUARD_AUTOUSE_FIRST", "1")


@pytest.fixture(autouse=True)
def zz_autouse_named_to_sort_last(monkeypatch):
    monkeypatch.setenv("GUARD_AUTOUSE_LAST", "1")


@pytest.fixture
def requested_fixture_sets_env(monkeypatch):
    monkeypatch.setenv("GUARD_REQUESTED", "1")
    monkeypatch.delenv("GUARD_PRESENT")


@pytest.fixture(scope="module")
def module_env():
    os.environ["GUARD_MODULE"] = "module"
    yield
    del os.environ["GUARD_MODULE"]


@pytest.fixture(scope="package")
def package_env():
    os.environ["GUARD_PACKAGE"] = "package"
    yield
    del os.environ["GUARD_PACKAGE"]


@pytest.fixture(scope="session")
def session_env():
    os.environ["GUARD_SESSION"] = "session"
    yield
    del os.environ["GUARD_SESSION"]


@pytest.fixture(scope="module")
def module_env_requested_late():
    os.environ["GUARD_MODULE_LATE"] = "late"
    yield
    del os.environ["GUARD_MODULE_LATE"]


@pytest.fixture(scope="session")
def session_env_set_up_by_a_module_fixture():
    os.environ["GUARD_SESSION_NESTED"] = "nested"
    yield
    del os.environ["GUARD_SESSION_NESTED"]


@pytest.fixture(scope="module")
def module_fixture_that_asks_for_another(request):
    request.getfixturevalue("session_env_set_up_by_a_module_fixture")
    os.environ["GUARD_MODULE_NESTING"] = "nesting"
    yield
    del os.environ["GUARD_MODULE_NESTING"]


@pytest.fixture(scope="module")
def module_fixture_first_set_up_under_isolation():
    os.environ["GUARD_UNDER_ISOLATION"] = "isolated"
    yield
    del os.environ["GUARD_UNDER_ISOLATION"]


@pytest.fixture(scope="module")
def module_fixture_over_the_same_variable():
    os.environ["GUARD_SAME_VARIABLE"] = "module"
    yield
    del os.environ["GUARD_SAME_VARIABLE"]


@pytest.fixture(scope="module")
def module_fixture_that_saves_and_restores():
    saved = os.environ.get("GUARD_MASKED")
    os.environ["GUARD_MASKED"] = "module"
    yield
    if saved is None:
        del os.environ["GUARD_MASKED"]
    else:
        os.environ["GUARD_MASKED"] = saved


@pytest.fixture
def teardown_raises_after_a_leak():
    yield
    os.environ["GUARD_LEAK_BESIDE_AN_ERROR"] = "beside"
    raise RuntimeError("the teardown's own failure")


def test_clean():
    assert os.environ["GUARD_AUTOUSE_FIRST"] == os.environ["GUARD_AUTOUSE_LAST"] == "1"


def test_the_node_pack_comes_from_this_checkout():
    import dgx_monarch

    assert Path(dgx_monarch.__file__).resolve().is_relative_to(Path(os.environ["GUARD_SRC"]))


def test_importing_cv2_inside_a_test_writes_nothing():
    # GUARD_LIBRARY_PATH_AT_START is the parent's LD_LIBRARY_PATH, when it has one.
    before = os.environ.get("LD_LIBRARY_PATH")
    assert before == os.environ.get("GUARD_LIBRARY_PATH_AT_START")
    try:
        import cv2  # noqa: F401
    except ImportError:
        pass
    assert os.environ.get("LD_LIBRARY_PATH") == before


def test_monkeypatch_in_the_body(monkeypatch):
    monkeypatch.setenv("GUARD_BODY", "1")
    monkeypatch.setenv("GUARD_PRESENT", "changed")
    monkeypatch.delenv("GUARD_PRESENT")


def test_monkeypatch_in_a_requested_fixture(requested_fixture_sets_env):
    assert "GUARD_PRESENT" not in os.environ


def test_module_fixture_first_use(module_env):
    assert os.environ["GUARD_MODULE"] == "module"


def test_package_and_session_fixtures_first_use(package_env, session_env):
    assert os.environ["GUARD_PACKAGE"] == "package"
    assert os.environ["GUARD_SESSION"] == "session"


def test_monkeypatch_over_a_module_fixtures_variable(module_env, monkeypatch):
    monkeypatch.setenv("GUARD_MODULE", "shadowed")


def test_module_fixture_requested_from_the_body(request, monkeypatch):
    monkeypatch.setenv("GUARD_BODY", "1")
    request.getfixturevalue("module_env_requested_late")
    assert os.environ["GUARD_MODULE_LATE"] == "late"


def test_module_fixture_sets_up_a_session_fixture_itself(module_fixture_that_asks_for_another):
    assert os.environ["GUARD_SESSION_NESTED"] == "nested"
    assert os.environ["GUARD_MODULE_NESTING"] == "nesting"


def test_leaks_a_new_variable(module_env):
    os.environ["GUARD_LEAK_ADDED"] = "added"


def test_after_the_added_leak_starts_clean(module_env):
    assert "GUARD_LEAK_ADDED" not in os.environ
    assert os.environ["GUARD_MODULE"] == "module"


def test_leaks_a_changed_and_a_removed_variable(module_env):
    os.environ["GUARD_MODULE"] = "changed under the fixture"
    del os.environ["GUARD_PRESENT"]


def test_after_the_changed_leak_starts_clean(module_env):
    assert os.environ["GUARD_MODULE"] == "module"
    assert os.environ["GUARD_PRESENT"] == "present"


def test_leaks_beside_a_teardown_error(teardown_raises_after_a_leak):
    pass


def test_after_the_teardown_error_starts_clean():
    assert "GUARD_LEAK_BESIDE_AN_ERROR" not in os.environ


@pytest.mark.xfail(reason="the body is meant to fail; the leak is not")
def test_leaks_inside_an_expected_failure():
    os.environ["GUARD_LEAK_UNDER_XFAIL"] = "hidden"
    assert False


def test_after_the_expected_failure_starts_clean():
    assert "GUARD_LEAK_UNDER_XFAIL" not in os.environ


def test_monkeypatch_then_a_module_fixture_over_the_same_variable(request, monkeypatch):
    # monkeypatch undoes its write at teardown, which also removes the value the
    # module fixture set after it: the next user would find the variable gone.
    monkeypatch.setenv("GUARD_SAME_VARIABLE", "temporary")
    request.getfixturevalue("module_fixture_over_the_same_variable")
    assert os.environ["GUARD_SAME_VARIABLE"] == "module"


def test_the_next_user_of_that_module_fixture_finds_its_value(module_fixture_over_the_same_variable):
    assert os.environ["GUARD_SAME_VARIABLE"] == "module"


def test_writes_a_variable_then_a_module_fixture_sets_it(request):
    # The fixture saves what it finds as the original, so the test's write
    # would come back at module end as if the fixture had put things right.
    os.environ["GUARD_MASKED"] = "written by the test"
    request.getfixturevalue("module_fixture_that_saves_and_restores")
    assert os.environ["GUARD_MASKED"] == "module"


def test_the_next_user_of_the_saving_fixture_is_clean(module_fixture_that_saves_and_restores):
    assert os.environ["GUARD_MASKED"] == "module"


def test_isolated_environ_contains_every_kind_of_write(isolated_environ):
    os.environ["GUARD_CONTAINED"] = "added"
    os.environ["GUARD_PRESENT"] = "changed"
    del os.environ["GUARD_ALSO_PRESENT"]


def test_after_isolated_environ_everything_is_back():
    assert "GUARD_CONTAINED" not in os.environ
    assert os.environ["GUARD_PRESENT"] == "present"
    assert os.environ["GUARD_ALSO_PRESENT"] == "also present"


def test_isolated_environ_undoes_a_module_fixture_set_up_from_the_body(request, isolated_environ):
    request.getfixturevalue("module_fixture_first_set_up_under_isolation")
    assert os.environ["GUARD_UNDER_ISOLATION"] == "isolated"


def test_the_next_user_of_the_fixture_set_up_under_isolation_finds_its_value(
        module_fixture_first_set_up_under_isolation):
    assert os.environ["GUARD_UNDER_ISOLATION"] == "isolated"


def test_module_fixtures_last_use(module_env, module_env_requested_late, package_env):
    assert os.environ["GUARD_MODULE"] == "module"
    assert os.environ["GUARD_MODULE_LATE"] == "late"
'''

# Second module. Its module-scoped fixture sets a variable and never puts it
# back.
LEAKING_FIXTURE = '''
import os

import pytest


@pytest.fixture(scope="module")
def leaking_module_fixture():
    os.environ["GUARD_MODULE_LEAK"] = "left behind"
    return "leaking"


def test_leaking_module_fixture_first_use(leaking_module_fixture):
    assert os.environ["GUARD_MODULE_LEAK"] == "left behind"
    assert os.environ["GUARD_SESSION"] == "session"


def test_leaking_module_fixture_ends_here(leaking_module_fixture):
    assert os.environ["GUARD_MODULE_LEAK"] == "left behind"
'''

# Third module, collected last. Its test is where the session ends.
AFTERWARDS = '''
import os


def test_after_the_wider_fixtures_everything_is_back():
    assert "GUARD_MODULE_LEAK" not in os.environ
    assert "GUARD_MODULE" not in os.environ
    assert "GUARD_MODULE_LATE" not in os.environ
    assert "GUARD_MODULE_NESTING" not in os.environ
    assert "GUARD_UNDER_ISOLATION" not in os.environ
    assert "GUARD_SAME_VARIABLE" not in os.environ
    assert "GUARD_MASKED" not in os.environ
    # Without a tests/__init__.py there is no package node, so pytest binds a
    # package-scoped fixture to the session; both live until the session ends.
    assert os.environ["GUARD_PACKAGE"] == "package"
    assert os.environ["GUARD_SESSION"] == "session"
    assert os.environ["GUARD_SESSION_NESTED"] == "nested"
    assert os.environ["GUARD_PRESENT"] == "present"
'''

# What each case must report. A leak fails its test at teardown, which pytest
# reports as an error beside the body's own outcome.
CLEAN, LEAKED, LEAKED_UNDER_XFAIL = {"PASSED"}, {"PASSED", "ERROR"}, {"XFAIL", "ERROR"}
EXPECTED = {
    "test_clean": CLEAN,
    "test_the_node_pack_comes_from_this_checkout": CLEAN,
    "test_importing_cv2_inside_a_test_writes_nothing": CLEAN,
    "test_monkeypatch_in_the_body": CLEAN,
    "test_monkeypatch_in_a_requested_fixture": CLEAN,
    "test_module_fixture_first_use": CLEAN,
    "test_package_and_session_fixtures_first_use": CLEAN,
    "test_monkeypatch_over_a_module_fixtures_variable": CLEAN,
    "test_module_fixture_requested_from_the_body": CLEAN,
    "test_module_fixture_sets_up_a_session_fixture_itself": CLEAN,
    "test_leaks_a_new_variable": LEAKED,
    "test_after_the_added_leak_starts_clean": CLEAN,
    "test_leaks_a_changed_and_a_removed_variable": LEAKED,
    "test_after_the_changed_leak_starts_clean": CLEAN,
    "test_leaks_beside_a_teardown_error": LEAKED,
    "test_after_the_teardown_error_starts_clean": CLEAN,
    "test_leaks_inside_an_expected_failure": LEAKED_UNDER_XFAIL,
    "test_after_the_expected_failure_starts_clean": CLEAN,
    "test_monkeypatch_then_a_module_fixture_over_the_same_variable": LEAKED,
    "test_the_next_user_of_that_module_fixture_finds_its_value": CLEAN,
    "test_writes_a_variable_then_a_module_fixture_sets_it": LEAKED,
    "test_the_next_user_of_the_saving_fixture_is_clean": CLEAN,
    "test_isolated_environ_contains_every_kind_of_write": CLEAN,
    "test_after_isolated_environ_everything_is_back": CLEAN,
    "test_isolated_environ_undoes_a_module_fixture_set_up_from_the_body": LEAKED,
    "test_the_next_user_of_the_fixture_set_up_under_isolation_finds_its_value": CLEAN,
    "test_module_fixtures_last_use": CLEAN,
    "test_leaking_module_fixture_first_use": CLEAN,
    "test_leaking_module_fixture_ends_here": LEAKED,
    "test_after_the_wider_fixtures_everything_is_back": CLEAN,
}


class ChildRun:
    def __init__(self, done: subprocess.CompletedProcess[str]) -> None:
        self.output = done.stdout + done.stderr
        self.returncode = done.returncode
        self.outcomes: dict[str, set[str]] = {}
        for kind, name in re.findall(r"^(PASSED|FAILED|ERROR|XFAIL) \S+::(\w+)", done.stdout,
                                     re.MULTILINE):
            self.outcomes.setdefault(name, set()).add(kind)
        # One section per teardown error, up to the next section rule, which
        # pytest shortens to one character a side around a long title.
        self.errors = dict(re.findall(
            r"^_+ ERROR at teardown of (\w+) _+\n(.*?)(?=^[_=]+ )",
            done.stdout, re.MULTILINE | re.DOTALL))


@pytest.fixture(scope="module")
def child(tmp_path_factory) -> ChildRun:
    """One child pytest over the generated suite, under a copy of the real conftest.

    The child imports the node pack from this checkout's ``src``, first on its
    PYTHONPATH as an absolute path; the parent's own entries follow, made
    absolute against the parent's working directory, since the child runs in
    the suite's directory.
    """
    root = tmp_path_factory.mktemp("environ_guard")
    suite = root / "tests"
    suite.mkdir()
    (suite / "conftest.py").write_bytes(CONFTEST.read_bytes())
    (suite / "test_a_cases.py").write_text(CASES, encoding="utf-8")
    (suite / "test_b_leaking_fixture.py").write_text(LEAKING_FIXTURE, encoding="utf-8")
    (suite / "test_c_afterwards.py").write_text(AFTERWARDS, encoding="utf-8")
    env = dict(os.environ, GUARD_PRESENT="present", GUARD_ALSO_PRESENT="also present",
               GUARD_SRC=str(SRC))
    for inherited in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "GUARD_LIBRARY_PATH_AT_START"):
        env.pop(inherited, None)
    if "LD_LIBRARY_PATH" in env:
        env["GUARD_LIBRARY_PATH_AT_START"] = env["LD_LIBRARY_PATH"]
    inherited_paths = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC)] + [str(Path(entry).resolve()) for entry in inherited_paths])
    done = subprocess.run(
        [sys.executable, "-m", "pytest", str(suite), "-q", "-rA", "--color=no",
         "-p", "no:cacheprovider", "--rootdir", str(root), "--basetemp", str(root / "basetemp")],
        cwd=suite, env=env, capture_output=True, text=True, timeout=300, check=False)
    return ChildRun(done)


def test_every_case_ends_the_way_the_guard_promises(child):
    """Clean tests pass, leaking tests fail, and nothing else is blamed.

    The clean cases show that the guard has no false positive: a variable set
    through monkeypatch from the body, from a requested fixture or from
    autouse fixtures named to sort before and after every fixture in the
    conftest; a module-, package- or session-scoped fixture that sets a
    variable for its own lifetime, requested from a signature, from a test body
    or from another fixture's body; and the imports of the node pack and of
    cv2, neither of which the child has loaded when its first test starts. The
    package and session cases run under the layout in which pytest's hook
    proxy for the session drops the conftest, so they pin that the guard is
    reached through a plugin of its own.
    """
    assert child.outcomes == EXPECTED, child.output[-6000:]
    assert child.returncode == 1
    assert set(child.errors) == {name for name, kinds in EXPECTED.items() if "ERROR" in kinds}


def test_a_leak_names_each_variable_with_its_value_before_and_after(child):
    added = child.errors["test_leaks_a_new_variable"]
    assert "left os.environ changed" in added
    assert "GUARD_LEAK_ADDED: unset -> 'added'" in added
    changed = child.errors["test_leaks_a_changed_and_a_removed_variable"]
    assert "GUARD_MODULE: 'module' -> 'changed under the fixture'" in changed
    assert "GUARD_PRESENT: 'present' -> unset" in changed
    # Only the leaked variables are named: nothing set through monkeypatch or
    # by a live wider fixture appears in any failure.
    everything = "".join(child.errors.values())
    for bystander in ("GUARD_AUTOUSE", "GUARD_BODY", "GUARD_REQUESTED", "GUARD_SESSION",
                      "GUARD_SESSION_NESTED", "GUARD_PACKAGE", "GUARD_MODULE_NESTING",
                      "GUARD_CONTAINED", "GUARD_SRC",
                      "HYPERACTOR", "TORCHINDUCTOR", "LD_LIBRARY_PATH"):
        assert bystander not in everything, bystander


def test_the_test_after_a_leak_starts_from_the_restored_environment(child):
    """The guard restores before it fails, so one leak cannot reach a later test."""
    for name in ("test_after_the_added_leak_starts_clean",
                 "test_after_the_changed_leak_starts_clean",
                 "test_after_the_teardown_error_starts_clean",
                 "test_after_the_expected_failure_starts_clean",
                 "test_the_next_user_of_that_module_fixture_finds_its_value",
                 "test_the_next_user_of_the_saving_fixture_is_clean",
                 "test_the_next_user_of_the_fixture_set_up_under_isolation_finds_its_value",
                 "test_after_the_wider_fixtures_everything_is_back"):
        assert child.outcomes[name] == CLEAN, child.output[-6000:]


def test_a_leak_beside_a_teardown_error_is_added_to_that_error(child):
    """The teardown's own failure stays the failure; the leak is added to it as a note."""
    section = child.errors["test_leaks_beside_a_teardown_error"]
    assert "RuntimeError: the teardown's own failure" in section
    assert "GUARD_LEAK_BESIDE_AN_ERROR: unset -> 'beside'" in section


def test_a_leak_inside_an_expected_failure_is_still_a_failure(child):
    """xfail covers the body's failure, not the leak, which is named as usual."""
    section = child.errors["test_leaks_inside_an_expected_failure"]
    assert "GUARD_LEAK_UNDER_XFAIL: unset -> 'hidden'" in section


def test_a_wider_fixture_that_never_puts_its_variable_back_is_named(child):
    """A module-scoped fixture owns what it set only while it lives."""
    section = child.errors["test_leaking_module_fixture_ends_here"]
    assert "GUARD_MODULE_LEAK: unset -> 'left behind'" in section
    assert "module-scoped fixture 'leaking_module_fixture' finished here" in section


def test_a_live_wider_fixture_whose_variable_the_test_removed_is_named(child):
    """isolated_environ undid a module fixture the body set up; the fixture is the owner."""
    section = child.errors["test_isolated_environ_undoes_a_module_fixture_set_up_from_the_body"]
    assert "GUARD_UNDER_ISOLATION: 'isolated' -> unset" in section
    assert ("module-scoped fixture 'module_fixture_first_set_up_under_isolation' holds it until "
            "it finishes") in section


def test_a_write_before_a_wider_fixture_claims_the_variable_is_the_tests_leak(child):
    """The test that wrote first is blamed, not the fixture that saved its value as the original."""
    section = child.errors["test_writes_a_variable_then_a_module_fixture_sets_it"]
    assert "GUARD_MASKED: unset -> 'written by the test'" in section
    assert ("changed before the module-scoped fixture 'module_fixture_that_saves_and_restores' "
            "set it, which holds it until it finishes") in section
    # The fixture handed that value back at module end; the guard put the
    # original back without blaming the module's last test.
    assert "test_module_fixtures_last_use" not in child.errors


def test_monkeypatch_undoing_a_live_wider_fixtures_value_is_named(child):
    """monkeypatch's own undo removed what the fixture set after it; the guard says so."""
    section = child.errors["test_monkeypatch_then_a_module_fixture_over_the_same_variable"]
    assert "GUARD_SAME_VARIABLE: 'module' -> unset" in section
    assert ("module-scoped fixture 'module_fixture_over_the_same_variable' set it during this "
            "test, after the test had changed it to 'temporary'") in section
    # The temporary value never outlives the test: the fixture's next user and
    # the module after it are clean, which test_every_case_ends_the_way_the_guard_promises
    # checks through their outcomes.


def test_the_guard_is_a_plugin_of_its_own_wrapping_the_hooks_every_fixture_runs_inside(
        pytestconfig, request):
    """Why no fixture order or scope can escape the guard, pinned on this process.

    pytest sets up and tears down every fixture inside its own plain
    implementations of ``pytest_runtest_setup`` and ``pytest_runtest_teardown``.
    The guard's two ends are wrappers around those hooks, so they run outside
    every fixture whatever its scope, name or position. The fixture hooks are
    called through the hook proxy of the fixture's node; for a session-scoped
    fixture that is the session, whose proxy drops every conftest module not
    visible from the repository root. The guard is an object registered by
    name, not a conftest module, so that proxy keeps it.
    """
    manager = pytestconfig.pluginmanager
    plugin = manager.get_plugin(PLUGIN)
    assert plugin is not None and not isinstance(plugin, types.ModuleType)

    def guards(caller) -> list:
        return [impl for impl in caller.get_hookimpls() if impl.plugin is plugin]

    session = request.session
    at_the_session = session.gethookproxy(session.path)
    for hook in ("pytest_runtest_setup", "pytest_runtest_teardown", "pytest_fixture_setup",
                 "pytest_runtest_makereport"):
        (guard,) = guards(getattr(pytestconfig.hook, hook))
        assert guard.wrapper and guard.function.__name__.startswith("pytest_environ_guard_"), hook
        assert guard.function.__name__ != hook
        assert guards(getattr(at_the_session, hook)) == [guard], hook
    for hook in ("pytest_runtest_setup", "pytest_runtest_teardown"):
        (runner,) = [impl for impl in getattr(pytestconfig.hook, hook).get_hookimpls()
                     if impl.plugin_name == "runner"]
        assert not runner.wrapper and not runner.hookwrapper, hook
    (finished,) = guards(pytestconfig.hook.pytest_fixture_post_finalizer)
    assert guards(at_the_session.pytest_fixture_post_finalizer) == [finished]
    (report,) = guards(pytestconfig.hook.pytest_runtest_makereport)
    assert report.tryfirst
