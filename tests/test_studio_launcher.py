"""The companion launcher: loopback bind, state lock, token delivery and startup refusals.

``web.run_app`` is replaced, so nothing binds a port. The real argument checks,
state directory, lock, store, driver binding and ``create_app`` all run.
"""
from __future__ import annotations

import fcntl
import io
import os
import sqlite3
import stat
import sys
import types
from types import SimpleNamespace

import pytest
from aiohttp import web

from dgx_monarch import studio_resource_validation
from dgx_monarch.studio import __main__ as launcher
from dgx_monarch.studio import server as server_module

OWNER_FILE = "owner-token"


class Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture
def launch(monkeypatch):
    record = SimpleNamespace(run=None, allowed_hosts=None, token=None, during=None, store=None)
    real_create_app = server_module.create_app

    def create_app(store, client, token, *, allowed_hosts, operations=None, poll=True):
        record.allowed_hosts, record.token, record.store = set(allowed_hosts), token, store
        return real_create_app(store, client, token, allowed_hosts=allowed_hosts, operations=operations, poll=False)

    def run_app(app, **kwargs):
        assert isinstance(app, web.Application)
        record.run = kwargs
        if record.during is not None:
            record.during()

    monkeypatch.setattr(server_module, "create_app", create_app)
    monkeypatch.setattr(web, "run_app", run_app)
    return record


def private(path):
    path.mkdir(mode=0o700, parents=True)
    path.chmod(0o700)
    return path


def lock_is_free(directory):
    descriptor = os.open(directory / "companion.lock", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(("host", "port"), [
    ("0.0.0.0", 8220), ("10.0.0.5", 8220), ("::", 8220), ("localhost", 8220),  # noqa: S104
    ("127.0.0.1", 0), ("127.0.0.1", 65536),
])
def test_launcher_refuses_a_non_loopback_host_or_a_bad_port_before_touching_state(tmp_path, launch, host, port):
    with pytest.raises(SystemExit) as stopped:
        launcher.main(["--host", host, "--port", str(port), "--state-dir", str(tmp_path / "state")])
    assert stopped.value.code == 2
    assert launch.run is None
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize(("host", "allowed"), [("127.0.0.1", "127.0.0.1:18220"), ("::1", "[::1]:18220")])
def test_launcher_binds_the_loopback_literal_and_allows_only_its_own_host(tmp_path, launch, host, allowed):
    state = tmp_path / "state"
    assert launcher.main(["--host", host, "--port", "18220", "--state-dir", str(state)]) == 0
    assert launch.allowed_hosts == {allowed}
    assert launch.run == {"host": host, "port": 18220, "access_log": None, "print": None}
    assert len(launch.token) >= 32


def test_launcher_holds_the_state_lock_while_serving_and_refuses_a_second_owner(tmp_path, launch):
    state = private(tmp_path / "state")
    observed = []
    launch.during = lambda: observed.append(lock_is_free(state))
    assert launcher.main(["--state-dir", str(state)]) == 0
    assert observed == [False]
    assert lock_is_free(state)

    descriptor = os.open(state / "companion.lock", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch.run = None
        with pytest.raises(SystemExit) as stopped:
            launcher.main(["--state-dir", str(state)])
    finally:
        os.close(descriptor)
    assert stopped.value.code == "A companion already owns this state directory."
    assert launch.run is None


def test_token_is_not_printed_to_a_pipe_and_lands_in_a_private_state_file(tmp_path, launch, capsys):
    state = tmp_path / "state"
    seen = {}

    def during():
        target = state / OWNER_FILE
        if target.exists():
            seen["mode"] = stat.S_IMODE(target.stat().st_mode)
            seen["text"] = target.read_text()

    launch.during = during
    assert launcher.main(["--state-dir", str(state)]) == 0
    output = capsys.readouterr().out
    assert launch.token not in output
    assert f"Owner token file (mode 0600): {state / OWNER_FILE}" in output
    assert seen == {"mode": 0o600, "text": launch.token + "\n"}
    assert not (state / OWNER_FILE).exists()
    assert launcher.OWNER_FILE == OWNER_FILE


def test_token_is_printed_only_to_a_terminal(tmp_path, launch, monkeypatch):
    terminal = Terminal()
    monkeypatch.setattr(sys, "stdout", terminal)
    state = tmp_path / "state"
    assert launcher.main(["--state-dir", str(state)]) == 0
    assert f"Owner token: {launch.token}" in terminal.getvalue()
    assert not (state / OWNER_FILE).exists()


def test_token_file_option_writes_an_owner_only_file_even_on_a_terminal(tmp_path, launch, monkeypatch):
    terminal = Terminal()
    monkeypatch.setattr(sys, "stdout", terminal)
    target = private(tmp_path / "secrets") / "studio-token"
    seen = {}
    launch.during = lambda: seen.update(mode=stat.S_IMODE(target.stat().st_mode), text=target.read_text())
    assert launcher.main(["--state-dir", str(tmp_path / "state"), "--token-file", str(target)]) == 0
    assert launch.token not in terminal.getvalue()
    assert str(target) in terminal.getvalue()
    assert seen == {"mode": 0o600, "text": launch.token + "\n"}
    assert not target.exists()
    assert list(target.parent.iterdir()) == []


def test_token_file_cleanup_never_removes_a_file_someone_else_put_there(tmp_path, launch):
    target = private(tmp_path / "secrets") / "studio-token"

    def replace():
        replacement = target.with_name("other")
        replacement.write_text("not the companion's")
        os.replace(replacement, target)

    launch.during = replace
    assert launcher.main(["--state-dir", str(tmp_path / "state"), "--token-file", str(target)]) == 0
    assert target.read_text() == "not the companion's"


def test_token_file_must_be_absolute(tmp_path, launch, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        launcher.main(["--state-dir", str(tmp_path / "state"), "--token-file", "token"])
    assert stopped.value.code == 2
    assert not (tmp_path / "state").exists() and not (tmp_path / "token").exists()


def test_an_earlier_token_left_by_a_killed_launch_is_replaced(tmp_path, launch):
    state = private(tmp_path / "state")
    (state / OWNER_FILE).write_text("a" * 43 + "\n")
    (state / OWNER_FILE).chmod(0o600)
    seen = {}
    launch.during = lambda: seen.update(text=(state / OWNER_FILE).read_text())
    assert launcher.main(["--state-dir", str(state)]) == 0
    assert seen == {"text": launch.token + "\n"}
    assert not (state / OWNER_FILE).exists()


@pytest.mark.parametrize("case", ["shared_directory", "symlinked_directory", "missing_directory",
                                  "state_file", "other_file", "small_private_file"])
def test_token_file_refusals_exit_cleanly_and_release_the_lock(tmp_path, launch, case):
    state = private(tmp_path / "state")
    if case == "shared_directory":
        shared = tmp_path / "shared"
        shared.mkdir(mode=0o755)
        shared.chmod(0o755)
        target, message = shared / "token", "must belong to this user with mode 0700"
    elif case == "symlinked_directory":
        (tmp_path / "link").symlink_to(private(tmp_path / "real"))
        target, message = tmp_path / "link" / "token", "must not be a symlink"
    elif case == "missing_directory":
        target, message = tmp_path / "absent" / "token", "existing directory"
    elif case == "state_file":
        target, message = state / "studio.sqlite3", "cannot replace a companion state file"
    elif case == "other_file":
        target = private(tmp_path / "keep") / "notes.txt"
        target.write_text("x" * 5000)
        target.chmod(0o600)
        message = "holds another file"
    else:
        # A small owner-only secret, such as a key in a 0700 directory, is not an earlier token.
        target = private(tmp_path / "keys") / "private_key"
        target.write_text("keep this owner-only secret\n")
        target.chmod(0o600)
        message = "holds another file"
    with pytest.raises(SystemExit) as stopped:
        launcher.main(["--state-dir", str(state), "--token-file", str(target)])
    assert isinstance(stopped.value.code, str) and message in stopped.value.code
    assert launch.run is None
    assert lock_is_free(state)
    if case == "other_file":
        assert target.read_text() == "x" * 5000
    if case == "small_private_file":
        assert target.read_text() == "keep this owner-only secret\n"
    if case == "state_file":
        sqlite3.connect(target).execute("SELECT count(*) FROM objects").fetchone()


@pytest.mark.parametrize("case", ["shared_mode", "symlink", "database_mode", "database_symlink", "schema"])
def test_store_refusals_exit_with_a_message_not_a_traceback(tmp_path, launch, case):
    state = tmp_path / "state"
    expected = None
    if case == "shared_mode":
        state.mkdir(mode=0o755)
        state.chmod(0o755)
        expected = "State directory must belong to this user with mode 0700."
    elif case == "symlink":
        state.symlink_to(private(tmp_path / "real"))
        expected = "State directories cannot be symlinks."
    elif case == "database_mode":
        private(state)
        (state / "studio.sqlite3").write_bytes(b"")
        (state / "studio.sqlite3").chmod(0o644)
        expected = "State database must be an owner-private regular file."
    elif case == "database_symlink":
        private(state)
        (state / "studio.sqlite3").symlink_to(tmp_path / "elsewhere.sqlite3")
    else:
        private(state)
        database = sqlite3.connect(state / "studio.sqlite3")
        database.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        database.execute("INSERT INTO meta VALUES ('schema','2')")
        database.commit()
        database.close()
        (state / "studio.sqlite3").chmod(0o600)
        expected = "Unsupported storage schema; export with the matching companion."
    with pytest.raises(SystemExit) as stopped:
        launcher.main(["--state-dir", str(state)])
    code = stopped.value.code
    assert isinstance(code, str)
    if expected is None:
        assert code.startswith("Cannot open the state database:")
    else:
        assert code == expected
    assert launch.run is None
    if case in {"shared_mode", "symlink"}:
        # The directory check runs before the lock file or database is created.
        directory = tmp_path / "real" if case == "symlink" else state
        assert sorted(path.name for path in directory.iterdir()) == []
    else:
        assert lock_is_free(state)


def test_a_driver_binding_refusal_also_exits_cleanly_and_releases_the_lock(tmp_path, launch):
    state = tmp_path / "state"
    assert launcher.main(["--state-dir", str(state), "--driver", "http://127.0.0.1:8188"]) == 0
    with pytest.raises(SystemExit) as stopped:
        launcher.main(["--state-dir", str(state), "--driver", "http://127.0.0.1:9188"])
    assert isinstance(stopped.value.code, str) and "different driver origin" in stopped.value.code
    assert lock_is_free(state)


def test_empty_xdg_state_home_counts_as_unset(tmp_path, launch, monkeypatch):
    home, elsewhere = tmp_path / "home", tmp_path / "launch"
    home.mkdir()
    elsewhere.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_STATE_HOME", "")
    monkeypatch.chdir(elsewhere)
    assert launcher.main([]) == 0
    assert launch.store.directory == home / ".local/state/dgx-monarch/studio"
    assert list(elsewhere.iterdir()) == []


def test_absolute_xdg_state_home_is_the_default_root(tmp_path, launch, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert launcher.main([]) == 0
    assert launch.store.directory == tmp_path / "xdg/dgx-monarch/studio"


def test_relative_xdg_state_home_is_refused(tmp_path, launch, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", "relative-state")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        launcher.main([])
    assert stopped.value.code == "XDG_STATE_HOME must be an absolute path; unset it or pass an absolute --state-dir."
    assert list(tmp_path.iterdir()) == []


def test_relative_state_dir_is_refused(tmp_path, launch, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        launcher.main(["--state-dir", "relative/studio"])
    assert stopped.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("missing", ["aiohttp", "PIL"])
def test_a_missing_studio_dependency_names_the_constrained_install(tmp_path, launch, monkeypatch, missing):
    monkeypatch.setitem(sys.modules, missing, None)
    if missing == "PIL":
        # Make the companion import its image module again so the missing package is seen.
        monkeypatch.delitem(sys.modules, "dgx_monarch.studio.server")
        monkeypatch.delitem(sys.modules, "dgx_monarch.studio.media_assets")
    with pytest.raises(SystemExit) as stopped:
        launcher.main(["--state-dir", str(tmp_path / "state")])
    assert stopped.value.code == launcher.INSTALL_HINT
    assert '-e "$REPO_DIR[studio]"' in launcher.INSTALL_HINT and "docs/INSTALL.md" in launcher.INSTALL_HINT
    assert "unconstrained" in launcher.INSTALL_HINT
    assert not (tmp_path / "state").exists()


def test_a_broken_studio_module_keeps_its_import_error(tmp_path, launch, monkeypatch):
    monkeypatch.setitem(sys.modules, "dgx_monarch.studio.server", types.ModuleType("dgx_monarch.studio.server"))
    with pytest.raises(ImportError) as raised:
        launcher.main(["--state-dir", str(tmp_path / "state")])
    assert "create_app" in str(raised.value)
    monkeypatch.setitem(sys.modules, "dgx_monarch.studio.operations", None)
    with pytest.raises(ModuleNotFoundError):
        launcher.main(["--state-dir", str(tmp_path / "state")])


def test_check_verifies_every_manifest_file_and_reports_the_extra(capsys):
    assert launcher.main(["--check"]) == 0
    output = capsys.readouterr().out
    table = studio_resource_validation.check_packaged_studio_resources()
    assert f"{len(table)} packaged resources match their manifest" in output
    assert "studio extra installed (aiohttp " in output and "Pillow " in output


def test_check_passes_without_the_extra_and_says_so(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "aiohttp", None)
    monkeypatch.setitem(sys.modules, "PIL", None)
    assert launcher.main(["--check"]) == 0
    assert "studio extra not installed (missing aiohttp, Pillow)" in capsys.readouterr().out


def test_check_fails_when_a_hashed_asset_is_missing(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    package = Path(studio_resource_validation.__file__).parent
    root = tmp_path / "package"
    for relative in ("studio/data", "studio/static"):
        shutil.copytree(package / relative, root / relative)
    shutil.copy2(package / "studio/resource_hashes.py", root / "studio/resource_hashes.py")
    stylesheet = next((root / "studio/static/assets").glob("*.css"))
    stylesheet.unlink()
    real = studio_resource_validation.check_packaged_studio_resources
    monkeypatch.setattr(studio_resource_validation, "check_packaged_studio_resources", lambda: real(root))
    with pytest.raises(SystemExit) as stopped:
        launcher.main(["--check"])
    assert isinstance(stopped.value.code, str)
    assert f"missing studio/static/assets/{stylesheet.name}" in stopped.value.code
