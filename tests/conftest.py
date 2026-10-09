"""Shared pytest settings, fixtures, and isolation guards.

Autouse fixtures isolate placeholder DNS, source-only imports, gate state,
and family memo files, and reject unexpected SSH calls. Plugins detect and
restore leaked ComfyUI modules, sys.path entries, and environment variables.
The sp and isolated_environ fixtures are opt-in. Collection excludes canary
scripts and freezes imported objects; settings also configure Hypothesis
and filter SWIG warnings.
"""

from __future__ import annotations

import gc
import os
import socket
import subprocess
import sys
import time
import warnings

import pytest
from hypothesis import settings

# Canaries are CI scripts the workflows run directly, not unit tests.
collect_ignore_glob = ["canary/*"]


def pytest_collection_finish(session):
    """Exclude collection-time objects from repeated garbage-collection scans.

    Unload tests call ``gc.collect()``. Freezing objects already retained by the
    session avoids rescanning torch, the node pack, and test modules on each
    call. Objects created by tests remain collectable. Frozen objects are also
    absent from ``gc.get_objects()`` and ``gc.get_referrers()``; no test uses
    those functions.
    """
    gc.collect()
    gc.freeze()

# SWIG can warn during interpreter finalization, after pytest's per-item
# filters expire. A process-wide filter covers only these three messages.
warnings.filterwarnings(
    "ignore",
    message=(r"builtin type (SwigPyObject|SwigPyPacked|swigvarlink) "
             r"has no __module__ attribute"),
    category=DeprecationWarning,
)

settings.register_profile(
    "ci",
    max_examples=64,
    derandomize=True,
    database=None,
    deadline=None,
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

# Read once at import, before any test patches gethostname: this box's own
# name keeps reaching the real resolver.
_THIS_HOST = socket.gethostname().lower()
_SOURCE_ONLY_ENV = ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX")


def _is_placeholder_host(host: object) -> bool:
    """A single-label name other than localhost, this box, or a numeric address."""
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str) or not host or "." in host or ":" in host:
        return False
    if host.lower() in {"localhost", _THIS_HOST}:
        return False
    try:
        socket.inet_aton(host)  # a parse, not a lookup: "1" and "0x7f000001"
    except OSError:
        return True
    return False


@pytest.fixture(autouse=True)
def _placeholder_hosts_get_the_hosted_runner_answer(monkeypatch):
    """Reject single-label placeholder hosts without network lookups.

    Names such as ``worker`` should be unresolvable in these tests. Returning
    ``gaierror`` immediately avoids resolver search-domain timeouts. Dotted
    names, addresses, ``localhost``, and this host's name use the real resolver.
    Tests can override either patched function for their own duration.
    """
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname

    def unresolvable(host: object) -> socket.gaierror:
        return socket.gaierror(socket.EAI_NONAME, f"placeholder host {host!r} does not resolve")

    def getaddrinfo(host, *args, **kwargs):
        if _is_placeholder_host(host):
            raise unresolvable(host)
        return real_getaddrinfo(host, *args, **kwargs)

    def gethostbyname(host):
        if _is_placeholder_host(host):
            raise unresolvable(host)
        return real_gethostbyname(host)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", gethostbyname)


@pytest.fixture(autouse=True)
def _source_only_imports_end_with_the_test():
    """Restore import flags and environment after source-only worker tests.

    ``worker_loop.main()`` enables source-only imports before parsing argv.
    Without restoration, even a CLI refusal changes subsequent tests and child
    processes. Deleting an absent variable with ``monkeypatch.delenv`` cannot
    undo a later write by production code, so this fixture saves both states.
    """
    flags = (sys.dont_write_bytecode, sys.pycache_prefix)
    saved = {name: os.environ.get(name) for name in _SOURCE_ONLY_ENV}
    yield
    sys.dont_write_bytecode, sys.pycache_prefix = flags
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@pytest.fixture(autouse=True)
def _quarantine_scope_starts_empty():
    """Isolate quarantine records and cached class-K denials between tests.

    These process-local records retain policy dictionaries and must not let one
    test's aborted ceremony deny a later test's render.
    """
    from dgx_monarch.nodes import gate_inconclusive, gate_quarantine_scope

    gate_quarantine_scope.reset_quarantine_scope()
    gate_inconclusive.forget_known_wrong_aborts()
    yield
    gate_quarantine_scope.reset_quarantine_scope()
    gate_inconclusive.forget_known_wrong_aborts()


@pytest.fixture(autouse=True)
def _family_memo_stays_out_of_the_operator_cache(monkeypatch, tmp_path):
    """Redirect family memo writes to a separate file for each test.

    ``ModelStore.ensure`` can memoize a family when a real checkpoint clears the
    size floor. Such writes must never modify the operator's worker cache.
    Patch the attribute for an already-imported module and the environment for
    later imports and child processes. Function scope prevents cross-test rows
    and contention on a shared memo lock.
    """
    memo = tmp_path / "family_memo.json"
    monkeypatch.setenv("DGXM_FAMILY_MEMO_PATH", str(memo))
    module = sys.modules.get("dgx_monarch.actor.model_store")
    if module is not None:
        # Import order decides this: a module imported later reads the
        # variable above, so a missing module here is already redirected.
        monkeypatch.setattr(module, "_FAMILY_MEMO_PATH", str(memo))


def _operator_memo_stamp() -> tuple[str, int | None]:
    """The default family memo's path and its mtime, or None when it is absent."""
    path = os.path.expanduser("~/.cache/dgx-monarch/family_memo.json")
    try:
        return path, os.stat(path).st_mtime_ns
    except OSError:
        return path, None


@pytest.fixture(scope="session", autouse=True)
def _the_battery_cannot_write_the_operator_memo():
    """Check whether the operator's family memo changed during the session.

    The mtime check covers indirect writes through ``ModelStore.ensure``. A
    concurrent worker or benchmark may also change the file, so the failure
    reports the change without attributing it to a test.
    """
    path, before = _operator_memo_stamp()
    yield
    _path, after = _operator_memo_stamp()
    assert after == before, (
        f"the family memo at {os.path.basename(path)} changed during this run "
        f"(mtime {before} -> {after}). Either a test wrote the cache a worker "
        "reads, or another process on this box did; check the autouse redirect "
        "above before trusting the second reading."
    )


@pytest.fixture
def sp(monkeypatch):
    """Patch the xfuser rank accessors read by ``base.shard_seq``.

    Only ``base.sp_world`` and ``base.sp_rank`` are patched. Adapters that import
    those names into their own modules need local fixtures that patch those
    bindings too; pytest selects the closest fixture. Lens currently tests only
    rank 0, so a rank-1 Lens case would also need an override.
    """
    from dgx_monarch.adapters import base

    def set_rank(world, rank):
        monkeypatch.setattr(base, "sp_world", lambda: world)
        monkeypatch.setattr(base, "sp_rank", lambda: rank)

    return set_rank


@pytest.fixture(autouse=True)
def _isolate_service_telemetry(monkeypatch):
    """Consumer tests must not probe the developer's configured Workers.

    Producer tests use their own collector with a mocked probe. Integration
    tests can override this snapshot with explicit fixture observations.
    """
    from dgx_monarch.service_observations import services

    monkeypatch.setattr(services, "snapshot", lambda: {"state": "unavailable"})


@pytest.fixture(autouse=True)
def _the_battery_never_shells_out_to_ssh(monkeypatch):
    """Fail a test that spawned ssh instead of standing in for the runner.

    Every fleet path in this repository reaches a worker through
    ``lifecycle.run_on_host``, which runs ``ssh`` for a host that is not this
    box. A test that leaves that seam unpatched sends real traffic at the
    placeholder addresses the battery configures. It usually passes anyway,
    because ssh exits 255 at once and the callers read a failed probe as
    unobserved, so nothing in the run says what happened.

    It records instead of raising: the attach reads its listener generations
    inside a ``try`` that swallows every exception, so a guard that raised
    would be absorbed by the code it is guarding and the test would still pass.
    """
    spawned: list[str] = []
    real = subprocess.run

    def record(args, *rest, **kwargs):
        first = args[0] if isinstance(args, (list, tuple)) and args else args
        if isinstance(first, str) and os.path.basename(first) == "ssh":
            spawned.append(" ".join(map(str, args)))
        return real(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "run", record)
    yield
    assert not spawned, (
        "this test spawned ssh; patch the host runner seam it reaches "
        f"({len(spawned)} call(s), first: {spawned[0][:120] if spawned else ''})"
    )


# --- comfy module guard ----------------------------------------------------
#
# Real-ComfyUI imports and stubs share module names. Leaking either can change
# later imports or make ImportError branches unreachable. At each module
# boundary, after fixture teardown, compare watched modules and sys.path with
# the post-collection baseline. Report leaks on the module's last test and
# restore the baseline before the next module runs.
#
# Watch comfy/comfy.*, all modules loaded from a ComfyUI checkout, and local
# copies or stubs using its top-level names (including folder_paths and comfy_*).
# Installed comfy_* distributions are ordinary lazy imports and are exempt.
# Session/package fixtures temporarily extend the baseline; their finalizers
# must restore their own changes or the final test reports the fixture's leak.
#
# Exempt the checkout root, its ancestors, and src by resolved path. pytest's
# Package setup inserts the root (or its parent for an identifier-named root)
# after collection, and __init__.py may then insert src. This exemption also
# means the guard cannot detect a test removing one of those entries.
#
# Register a separate plugin: the hook proxy for session/package fixtures
# excludes this conftest because it is not visible from the repository root.
# Use specname with distinct configure-function names so registering the
# environment guard below cannot replace this guard's registration hook.

COMFY_GUARD_PLUGIN = "dgx_monarch_comfy_guard"
_COMFY_GUARD_TOP_NAMES = frozenset({
    "folder_paths", "node_helpers", "nodes", "server", "execution", "latent_preview",
    "protocol", "utils", "app", "api_server", "middleware", "alembic_db", "cuda_malloc",
    "hook_breaker_ac10a0", "comfyui_version", "main",
})
_COMFY_GUARD_ABSENT = object()


class _ComfyGuard:
    """The guard's hooks and the state they share, registered as one plugin.

    ``baseline`` is the watched part of sys.modules, by name, and ``path`` the
    sys.path, that every module must hand back; both are None until collection
    has finished. ``root`` and ``root_src`` are the resolved paths the path
    comparison skips, with the root's ancestors. A wider fixture's claim is what the baseline held, or
    that it held nothing, for each name its setup changed, the path to return
    to once it finishes, and the live path before and after its setup, so
    its own entries can be told from another fixture's. ``notes`` names the
    wider fixtures that finished without putting their entries back, for the
    next verdict.
    """

    def __init__(self) -> None:
        self.baseline: dict[str, object] | None = None
        self.path: list[str] | None = None
        self.root = ""
        self.root_src = ""
        self.seconds = 0.0
        self.checkout_dirs: dict[str, bool] = {}
        self.verdicts: dict[str, tuple[int, bool]] = {}
        self.resolved: dict[str, bool] = {}
        self.claims: dict[object, tuple[dict[str, object], list[str], list[str], list[str]]] = {}
        self.notes: list[str] = []

    @staticmethod
    def own(module: object, attribute: str) -> object:
        """A module's own attribute, read without running a module-level __getattr__.

        Lazy packages such as transformers answer any name their namespace
        lacks by importing a submodule, so a plain getattr over sys.modules
        would run imports from inside the guard. A hosted runner hit this on
        2026-10-05: the import failed on a missing optional dependency.
        Objects without a namespace, and names the namespace lacks, read None.
        """
        try:
            return vars(module).get(attribute)
        except TypeError:
            return None

    @staticmethod
    def location(module: object) -> str | None:
        """The file, or first package directory, a module came from; None for a stub."""
        location = _ComfyGuard.own(module, "__file__")
        if not location:
            paths = _ComfyGuard.own(module, "__path__")
            if paths is not None:
                try:
                    location = next(iter(paths), None)
                except (KeyError, TypeError):
                    # A namespace package's path re-resolves through its parent
                    # in sys.modules, and raises once the parent is gone. The
                    # entries it last resolved are still on the object.
                    inner = getattr(paths, "_path", None)
                    location = inner[0] if isinstance(inner, list) and inner else None
        return location if isinstance(location, str) else None

    def under_checkout(self, directory: str) -> bool:
        """Does this directory, or one above it, hold comfy/options.py? Memoized."""
        known = self.checkout_dirs.get(directory)
        if known is None:
            parent = os.path.dirname(directory)
            known = os.path.isfile(os.path.join(directory, "comfy", "options.py")) or (
                parent != directory and self.under_checkout(parent))
            self.checkout_dirs[directory] = known
        return known

    def watches(self, name: str, module: object) -> bool:
        top = name.partition(".")[0]
        if top == "comfy":
            return True
        named = top in _COMFY_GUARD_TOP_NAMES or top.startswith("comfy_")
        location = self.location(module)
        if location is None:
            return named
        directory = os.path.dirname(os.path.abspath(location))
        if self.under_checkout(directory):
            return True
        parts = directory.split(os.sep)
        return named and "site-packages" not in parts and "dist-packages" not in parts

    def snapshot(self) -> dict[str, object]:
        """The watched entries of sys.modules, by name. Verdicts are memoized per module object."""
        watched: dict[str, object] = {}
        for name, module in list(sys.modules.items()):
            cached = self.verdicts.get(name)
            if cached is None or cached[0] != id(module):
                cached = (id(module), self.watches(name, module))
                self.verdicts[name] = cached
            if cached[1]:
                watched[name] = module
        return watched

    def describe(self, module: object) -> str:
        location = self.location(module)
        return "stub, no file" if location is None else location

    def is_own(self, entry: str) -> bool:
        """Is this sys.path entry the root, a directory above it, or the root's src? Memoized."""
        own = self.resolved.get(entry)
        if own is None:
            resolved = os.path.realpath(entry)
            own = self.resolved[entry] = (
                resolved == self.root_src or os.path.commonpath([self.root, resolved]) == resolved)
        return own

    def compared_path(self, path: list[str]) -> list[str]:
        """sys.path without the checkout's own entries."""
        return [entry for entry in path if not self.is_own(entry)]

    @pytest.hookimpl(specname="pytest_collection_finish")
    def pytest_comfy_guard_baseline(self, session):
        """Take the baseline after collection: every import collection made is in it."""
        self.root = os.path.realpath(str(session.config.rootpath))
        self.root_src = os.path.join(self.root, "src")
        self.resolved.clear()
        self.baseline = self.snapshot()
        self.path = list(sys.path)

    @pytest.hookimpl(specname="pytest_fixture_setup", wrapper=True)
    def pytest_comfy_guard_fixture_setup(self, fixturedef):
        """Credit a session- or package-scoped fixture with what its setup installs.

        Its modules and path entries join the baseline until it is finalized;
        the claim remembers what the baseline held before, or that it held
        nothing.
        """
        if fixturedef.scope in ("function", "class", "module") or self.baseline is None:
            return (yield)
        assert self.path is not None
        before = self.snapshot()
        live_path_before = list(sys.path)
        try:
            return (yield)
        finally:
            after = self.snapshot()
            held: dict[str, object] = {}
            for name in before.keys() | after.keys():
                if before.get(name, _COMFY_GUARD_ABSENT) is not after.get(name, _COMFY_GUARD_ABSENT):
                    held[name] = self.baseline.get(name, _COMFY_GUARD_ABSENT)
                    if name in after:
                        self.baseline[name] = after[name]
                    else:
                        self.baseline.pop(name, None)
            if held or list(sys.path) != live_path_before:
                # The baseline path becomes the path as this fixture left it;
                # the claim keeps the one to return to once it is finalized.
                self.claims[fixturedef] = (held, list(self.path), live_path_before, list(sys.path))
                self.path = list(sys.path)

    @pytest.hookimpl(specname="pytest_fixture_post_finalizer")
    def pytest_comfy_guard_fixture_finished(self, fixturedef):
        """A finished wider-scoped fixture's claim ends: the baseline wants its earlier entries back.

        pytest calls this after the fixture's own teardown, so an entry still
        different from what the baseline held is one the fixture did not put
        back; the next verdict names the fixture beside it.
        """
        claim = self.claims.pop(fixturedef, None)
        if claim is None or self.baseline is None:
            return
        held, path_before, live_before, live_after = claim
        left: list[str] = []
        for name, earlier in held.items():
            if earlier is _COMFY_GUARD_ABSENT:
                self.baseline.pop(name, None)
                if name in sys.modules:
                    left.append(name)
            else:
                self.baseline[name] = earlier
                if sys.modules.get(name) is not earlier:
                    left.append(name)
        # Only the entries this fixture's own setup added or removed, by
        # count, so a leak by another fixture still live is not laid on it.
        before, after, now = (self.compared_path(path) for path in (live_before, live_after, sys.path))
        for entry in sorted(set(before) | set(after)):
            was, became, is_now = before.count(entry), after.count(entry), now.count(entry)
            if (became > was and is_now > was) or (became < was and is_now < was):
                left.append(f"sys.path entry {entry}")
        self.path = path_before
        if left:
            self.notes.append(
                f"the {fixturedef.scope}-scoped fixture {fixturedef.argname!r} finished here "
                "without putting back " + ", ".join(sorted(left)))

    def verdict(self, item, nextitem) -> str | None:
        """At a module boundary, the difference from the baseline, after restoring it."""
        if self.baseline is None or self.path is None:
            return None
        module = item.getparent(pytest.Module)
        if nextitem is not None and nextitem.getparent(pytest.Module) is module:
            return None
        started = time.perf_counter()
        try:
            baseline = self.baseline
            notes, self.notes = self.notes, []
            now = self.snapshot()
            added = sorted(name for name in now if name not in baseline)
            removed = sorted(name for name in baseline if name not in now)
            replaced = sorted(name for name in now if name in baseline and now[name] is not baseline[name])
            path_now = self.compared_path(sys.path)
            path_base = self.compared_path(self.path)
            if not (added or removed or replaced) and path_now == path_base:
                return None
            where = item.nodeid.split("::", 1)[0]
            lines = [f"{where} left comfy modules or sys.path changed after its last test; "
                     "the guard has put the baseline back:"]
            if added:
                lines.append("  added: " + ", ".join(
                    f"{name} ({self.describe(now[name])})" for name in added))
            if removed:
                lines.append("  removed: " + ", ".join(removed))
            if replaced:
                lines.append("  replaced: " + ", ".join(
                    f"{name} (now {self.describe(now[name])})" for name in replaced))
            path_added = [entry for entry in path_now if path_now.count(entry) > path_base.count(entry)]
            path_removed = [entry for entry in path_base if path_base.count(entry) > path_now.count(entry)]
            if path_added:
                lines.append("  sys.path added: " + ", ".join(sorted(set(path_added))))
            if path_removed:
                lines.append("  sys.path removed: " + ", ".join(sorted(set(path_removed))))
            if path_now != path_base and not path_added and not path_removed:
                lines.append("  sys.path reordered")
            lines.extend("  " + note for note in notes)
            lines.append(
                "A test or fixture in this module installed or imported these and did not take "
                "them back. Restore exactly what you removed, drop what you inserted, and leave "
                "sys.path as you found it (tests/test_auto_latent_downscale.py::real_comfy shows "
                "the shape).")
            for name in added:
                sys.modules.pop(name, None)
            for name in removed + replaced:
                sys.modules[name] = baseline[name]
            if path_now != path_base:
                own = [entry for entry in sys.path if self.is_own(entry)]
                sys.path[:] = own + path_base
            return "\n".join(lines)
        finally:
            self.seconds += time.perf_counter() - started

    @pytest.hookimpl(specname="pytest_runtest_teardown", wrapper=True)
    def pytest_comfy_guard_teardown(self, item, nextitem):
        """Check the comfy namespace once pytest has torn down the module's fixtures.

        The comparison runs after the yield, so every fixture of the finished
        module, monkeypatch and module-scoped ones included, has been
        finalized, and so has every session-scoped fixture when the item is
        the last. The restore comes first, so one leak cannot reach the files
        after it.
        """
        try:
            result = yield
        except BaseException as error:
            verdict = self.verdict(item, nextitem)
            if verdict:
                error.add_note(verdict)
            raise
        verdict = self.verdict(item, nextitem)
        if verdict:
            pytest.fail(verdict, pytrace=False)
        return result


@pytest.hookimpl(specname="pytest_configure")
def pytest_comfy_guard_configure(config):
    """Register the guard as a plugin of its own, which no node's hook proxy can drop."""
    if not config.pluginmanager.has_plugin(COMFY_GUARD_PLUGIN):
        config.pluginmanager.register(_ComfyGuard(), name=COMFY_GUARD_PLUGIN)
# --- end comfy module guard -------------------------------------------------


# The environment guard, and the fixture that satisfies it.
#
# Environment changes affect later tests and child processes. Hook wrappers
# enclose pytest's setup and teardown, so checks run outside every fixture's
# lifetime, including monkeypatch and wider-scoped fixtures. A fixture cannot
# provide those ordering guarantees. Use a separately registered plugin for
# the same session/package hook-proxy reason as the ComfyUI guard above.
#
# Only PYTEST_CURRENT_TEST is exempt. The guard reads os.environ, so direct
# os.putenv writes are invisible (none exist in this repository). Wider fixtures
# own variables changed during setup; teardown-only writes are attributed to
# the finishing test without a fixture owner.

ENVIRON_GUARD_PLUGIN = "dgx_monarch_environ_guard"


def _environ_now() -> dict[str, str]:
    """os.environ without PYTEST_CURRENT_TEST, which pytest rewrites at every phase."""
    return {name: value for name, value in os.environ.items() if name != "PYTEST_CURRENT_TEST"}


def _environ_changes(
    before: dict[str, str], after: dict[str, str],
) -> dict[str, tuple[str | None, str | None]]:
    """Each variable that differs, with its value on both sides; None is unset."""
    if before == after:
        return {}
    return {name: (before.get(name), after.get(name))
            for name in sorted(before.keys() | after.keys())
            if before.get(name) != after.get(name)}


def _environ_put(target, name: str, value: str | None) -> None:
    if value is None:
        target.pop(name, None)
    else:
        target[name] = value


def _restore_environ(saved: dict[str, str]) -> dict[str, tuple[str | None, str | None]]:
    """Put os.environ back to ``saved`` exactly, and return what had changed."""
    changes = _environ_changes(saved, _environ_now())
    for name, (was, _now) in changes.items():
        _environ_put(os.environ, name, was)
    return changes


def _environ_value(value: str | None) -> str:
    return "unset" if value is None else repr(value)


@pytest.fixture
def isolated_environ():
    """Let this test's in-process code write os.environ, then put it all back.

    For a test that runs production code which sets the environment on purpose,
    as a worker does for its own process: a mesh bring-up plants the driver
    marker, ``setup_impl`` pins the CUDA device and the NCCL launch order.
    Every variable added is removed, and every one changed or removed gets its
    old value back.

    ``monkeypatch.setenv`` is still the tool for one variable the test sets
    itself. ``monkeypatch.delenv`` of a variable that is absent records nothing
    to undo, so it does not contain production code that sets the variable
    afterwards; this fixture does.
    """
    saved = _environ_now()
    yield
    _restore_environ(saved)


class EnvironmentLeak(pytest.fail.Exception):
    """The guard's verdict, kept a failure for a test marked xfail as well."""


class _EnvironGuard:
    """The guard's hooks and the state they share, registered as one plugin.

    ``expected`` is the environment the running test must hand back. A wider
    fixture's claim is each variable its setup changed with the value due back
    when the fixture finishes; ``tainted`` holds, for such a variable, a value
    the test had written before the fixture set it, which the fixture may hand
    back believing it the original. ``before_claim`` is this test's share of
    those, for its own verdict, and ``notes`` the owner clause the verdict adds
    to a variable.
    """

    def __init__(self) -> None:
        self.expected: dict[str, str] | None = None
        self.claims: dict[object, dict[str, str | None]] = {}
        self.tainted: dict[object, dict[str, str | None]] = {}
        self.setups: list[dict[str, str]] = []
        self.before_claim: dict[str, tuple[str | None, str | None, object]] = {}
        self.notes: dict[str, str] = {}
        self.baseline_imported = False

    @staticmethod
    def _describe(fixturedef) -> str:
        return f"{fixturedef.scope}-scoped fixture {fixturedef.argname!r}"

    def _baseline_imports(self) -> None:
        """Import environment-writing libraries before recording the baseline.

        The node pack sets attach timeout, execution-id, and compile-cache variables;
        importing it here makes them part of every test's baseline. The quarantine
        fixture would otherwise import it during the first test's setup.

        ComfyUI or xfuser may lazily import cv2, which changes LD_LIBRARY_PATH.
        Import it once here and restore that variable so later tests and child
        processes do not inherit an import-order-dependent library path.
        """
        if self.baseline_imported:
            return
        import dgx_monarch.nodes  # noqa: F401

        library_path = os.environ.get("LD_LIBRARY_PATH")
        try:
            import cv2  # noqa: F401
        except ImportError:
            pass
        finally:
            _environ_put(os.environ, "LD_LIBRARY_PATH", library_path)
        self.baseline_imported = True

    @pytest.hookimpl(specname="pytest_runtest_setup", wrapper=True)
    def pytest_environ_guard_setup(self):
        """Record the environment this test must hand back, before any fixture runs."""
        self._baseline_imports()
        self.notes.clear()
        self.before_claim.clear()
        self.expected = _environ_now()
        return (yield)

    @pytest.hookimpl(specname="pytest_fixture_setup", wrapper=True)
    def pytest_environ_guard_fixture_setup(self, fixturedef):
        """Book what a module-, class-, package- or session-scoped fixture sets.

        Such a fixture owns its variables for its own lifetime, which spans
        tests: they appear during the first test that uses it and go during the
        last. So its setup's changes join what the running test must hand back,
        and each variable is due back, when the fixture finishes, as the test
        found it. A variable the test had already changed when the outermost
        wider fixture in flight started is the test's own leak, booked for its
        verdict: the fixture may hand that value back as if it were the
        original.
        """
        if fixturedef.scope == "function":
            return (yield)
        before = _environ_now()
        self.setups.append(before)
        try:
            return (yield)
        finally:
            self.setups.pop()
            changes = _environ_changes(before, _environ_now())
            if changes and self.expected is not None:
                outermost = self.setups[0] if self.setups else before
                claim = self.claims.setdefault(fixturedef, {})
                for name, (_was, now) in changes.items():
                    if name not in claim:
                        owed = self.expected.get(name)
                        claim[name] = owed
                        found = outermost.get(name)
                        if found != owed:
                            self.tainted.setdefault(fixturedef, {})[name] = found
                            self.before_claim[name] = (owed, found, fixturedef)
                    _environ_put(self.expected, name, now)
                    # A fixture that set this one up from inside its own body
                    # did not make the change, so it must not claim it as well.
                    for outer in self.setups:
                        _environ_put(outer, name, now)

    @pytest.hookimpl(specname="pytest_fixture_post_finalizer")
    def pytest_environ_guard_fixture_finished(self, fixturedef):
        """A finished fixture's claim ends: its variables are due back as they were.

        pytest calls this after the fixture's own teardown. A fixture that did
        not put back what it set while setting up therefore fails the test it
        finished in, by name. One that hands back the value a test had written
        before it set the variable has done its part; that test was failed for
        the write, so the original goes back without a second verdict.
        """
        claim = self.claims.pop(fixturedef, None)
        tainted = self.tainted.pop(fixturedef, {})
        if not claim or self.expected is None:
            return
        for name, owed in claim.items():
            _environ_put(self.expected, name, owed)
            if name in tainted and os.environ.get(name) == tainted[name]:
                _environ_put(os.environ, name, owed)
            else:
                self.notes[name] = f"the {self._describe(fixturedef)} finished here without putting it back"

    def _verdict(self) -> str | None:
        """Restore whatever the finished test left changed, and say what it was."""
        expected, self.expected = self.expected, None
        if expected is None:
            return None
        changes = _restore_environ(expected)
        for name, (owed, found, fixturedef) in self.before_claim.items():
            if name in changes:
                self.notes[name] = (
                    f"the {self._describe(fixturedef)} set it during this test, after the test had "
                    f"changed it to {_environ_value(found)}; the teardown of that change removed the "
                    "fixture's value")
            else:
                changes[name] = (owed, found)
                self.notes[name] = (
                    f"changed before the {self._describe(fixturedef)} set it"
                    + (", which holds it until it finishes" if fixturedef in self.claims else ""))
        if not changes:
            return None
        for name in changes:
            if name not in self.notes:
                for fixturedef, claim in self.claims.items():
                    if name in claim:
                        self.notes[name] = f"the {self._describe(fixturedef)} holds it until it finishes"
        lines = ["this test left os.environ changed (before -> after); the guard has put it back:"]
        for name in sorted(changes):
            was, now = changes[name]
            note = self.notes.get(name)
            lines.append(f"  {name}: {_environ_value(was)} -> {_environ_value(now)}"
                         + (f" ({note})" if note else ""))
        lines.append(
            "Request the isolated_environ fixture in a test that runs code which writes the "
            "environment, or use monkeypatch.setenv for one variable the test sets itself. "
            "A wider fixture puts back what it set in its own teardown.")
        return "\n".join(lines)

    @pytest.hookimpl(specname="pytest_runtest_teardown", wrapper=True)
    def pytest_environ_guard_teardown(self):
        """Restore the environment a test changed, and fail that test.

        The comparison runs after every fixture of the test has been torn down,
        monkeypatch included, and after any wider fixture that ends with this
        test. The restore comes first, so one leak cannot reach the tests after
        it.
        """
        try:
            result = yield
        except BaseException as error:
            verdict = self._verdict()
            if verdict:
                error.add_note(verdict)
            raise
        verdict = self._verdict()
        if verdict:
            raise EnvironmentLeak(verdict, pytrace=False)
        return result

    @pytest.hookimpl(specname="pytest_runtest_makereport", wrapper=True, tryfirst=True)
    def pytest_environ_guard_makereport(self, call):
        """Keep the verdict a failure for a test marked xfail.

        pytest's skipping plugin turns every failed phase of such a test into an
        expected failure, teardown included, which would report a leak as XFAIL
        and never name it. This wrapper is the outermost, so it runs after that
        plugin and puts the outcome back.
        """
        report = yield
        if (call.when == "teardown" and call.excinfo is not None
                and isinstance(call.excinfo.value, EnvironmentLeak)):
            report.outcome = "failed"
            if hasattr(report, "wasxfail"):
                del report.wasxfail
        return report


@pytest.hookimpl(specname="pytest_configure")
def pytest_environ_guard_configure(config):
    """Register the guard as a plugin of its own, which no node's hook proxy can drop."""
    if not config.pluginmanager.has_plugin(ENVIRON_GUARD_PLUGIN):
        config.pluginmanager.register(_EnvironGuard(), name=ENVIRON_GUARD_PLUGIN)
