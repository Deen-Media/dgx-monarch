"""The companion launcher: loopback bind, state lock, token delivery, sign-in and startup refusals.

``web.run_app`` and ``sign_in.bind_listener`` are replaced, so nothing binds a
port unless a test asks for the real bind, and ``sign_in.open_in_browser`` is
a recorder, so no browser starts. The real argument checks, state directory,
lock, store, driver binding and ``create_app`` all run. The fixture clears the
display and session variables, so a shell with ``DISPLAY=:0`` runs the same
cases as a headless one.
"""
from __future__ import annotations

import errno
import fcntl
import io
import logging
import os
import shlex
import socket
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import types
from types import SimpleNamespace

import pytest
from aiohttp import web

from dgx_monarch import studio_resource_validation
from dgx_monarch.studio import __main__ as launcher
from dgx_monarch.studio import launch as launch_module
from dgx_monarch.studio import server as server_module
from dgx_monarch.studio import sign_in

OWNER_FILE = "owner-token"
SESSION_VARIABLES = ("DISPLAY", "WAYLAND_DISPLAY", "SSH_CONNECTION", "TMUX", "STY")
REAL_BIND = sign_in.bind_listener
REAL_OPEN = sign_in.open_in_browser


class Terminal(io.StringIO):
    def isatty(self):
        return True


class FakeSocket:
    """A bound listener that never touched the network."""

    def __init__(self, host, port):
        self.address, self.closed = (host, port), False

    def getsockname(self):
        return self.address

    def close(self):
        self.closed = True


@pytest.fixture
def launch(monkeypatch):
    record = SimpleNamespace(run=None, allowed_hosts=None, token=None, during=None, store=None, operations=None,
                             bound=[], sock=None, opened=[], browser=(sign_in.OPENED, None))
    real_create_app = server_module.create_app

    def create_app(store, client, token, *, allowed_hosts, operations=None, poll=True):
        record.allowed_hosts, record.token, record.store = set(allowed_hosts), token, store
        record.operations = operations
        return real_create_app(store, client, token, allowed_hosts=allowed_hosts, operations=operations, poll=False)

    def run_app(app, **kwargs):
        assert isinstance(app, web.Application)
        record.run = kwargs
        if record.during is not None:
            record.during()

    def bind_listener(host, port):
        record.bound.append((host, port))
        record.sock = FakeSocket(host, port)
        return record.sock

    def open_in_browser(path):
        record.opened.append(path)
        return record.browser

    for name in SESSION_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(server_module, "create_app", create_app)
    monkeypatch.setattr(web, "run_app", run_app)
    monkeypatch.setattr(sign_in, "bind_listener", bind_listener)
    monkeypatch.setattr(sign_in, "open_in_browser", open_in_browser)
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
    assert launch.bound == [(host, 18220)]
    assert launch.run == {"sock": launch.sock, "access_log": None, "print": None}
    assert launch.sock.closed
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
    # No display in the fixture, so no browser is sent the page and the link prints on its own line.
    assert f"Sign-in link:\n  http://127.0.0.1:8220/#token={launch.token}\n" in terminal.getvalue()
    assert not (state / OWNER_FILE).exists()
    assert not (state / sign_in.SIGN_IN_FILE).exists()


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


def lines_of(terminal):
    return terminal.getvalue().splitlines()


def link(token, port=8220):
    return f"http://127.0.0.1:{port}/#token={token}"


def state_names(state):
    return {path.name for path in state.iterdir()}


@pytest.mark.parametrize("cause", ["refused_by_the_kernel", "held_by_a_real_listener"])
def test_a_busy_port_exits_with_one_line_before_any_token_exists(tmp_path, launch, monkeypatch, capsys, cause):
    monkeypatch.setattr(sign_in, "bind_listener", REAL_BIND)
    state = tmp_path / "state"
    holder = None
    if cause == "refused_by_the_kernel":
        def busy(*args, **kwargs):
            raise OSError(errno.EADDRINUSE, "Address already in use")
        monkeypatch.setattr(sign_in.socket, "create_server", busy)
        port = 18220
    else:
        holder = socket.create_server(("127.0.0.1", 0))
        port = holder.getsockname()[1]
    try:
        with pytest.raises(SystemExit) as stopped:
            launcher.main(["--state-dir", str(state), "--port", str(port)])
    finally:
        if holder is not None:
            holder.close()
    assert stopped.value.code == sign_in.PORT_IN_USE
    assert "[studio] port" in sign_in.PORT_IN_USE and "--port for dgxm-studio" in sign_in.PORT_IN_USE
    assert capsys.readouterr() == ("", "")
    assert launch.run is None
    assert not {OWNER_FILE, sign_in.SIGN_IN_FILE} & state_names(state)
    assert lock_is_free(state)


def test_a_held_lock_is_retried_before_the_companion_refuses(tmp_path, launch):
    state = private(tmp_path / "state")
    descriptor = os.open(state / "companion.lock", os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # A probe that lets go within the retry window never ends a start.
    release = threading.Timer(0.1, os.close, (descriptor,))
    release.start()
    try:
        assert launcher.main(["--state-dir", str(state)]) == 0
    finally:
        release.join()
    assert launch.run is not None

    launch.run = None
    descriptor = os.open(state / "companion.lock", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        began = time.monotonic()
        with pytest.raises(SystemExit) as stopped:
            launcher.main(["--state-dir", str(state)])
        waited = time.monotonic() - began
    finally:
        os.close(descriptor)
    assert stopped.value.code == "A companion already owns this state directory."
    assert waited >= launcher.LOCK_RETRIES * launcher.LOCK_PAUSE
    assert launch.run is None


def test_a_driver_binding_refusal_happens_before_the_bind_and_names_both_origins(tmp_path, launch, capsys):
    state = tmp_path / "state"
    assert launcher.main(["--state-dir", str(state)]) == 0
    capsys.readouterr()
    launch.bound.clear()
    with pytest.raises(SystemExit) as stopped:
        launcher.main(["--state-dir", str(state), "--driver", "http://127.0.0.1:8191"])
    code = stopped.value.code
    assert isinstance(code, str)
    refusal, *extra = code.splitlines()
    assert refusal == ("This state directory belongs to a different driver origin. "
                       "Use its original driver or choose a separate --state-dir.")
    assert extra == launch_module.mismatch_lines(state, "http://127.0.0.1:8188", "http://127.0.0.1:8191")
    assert len(extra) == 2
    assert "http://127.0.0.1:8188" in extra[0] and "http://127.0.0.1:8191" in extra[0]
    assert extra[1] == launch_module.MOVE_STATE.format(state=shlex.quote(str(state)), suffix="8188")
    # Nothing was bound, printed or written for the refused launch.
    assert launch.bound == [] and capsys.readouterr().out == ""
    assert not {OWNER_FILE, sign_in.SIGN_IN_FILE} & state_names(state)
    assert lock_is_free(state)


@pytest.mark.parametrize("session", [{}, {"DISPLAY": ":0"}, {"DISPLAY": ":0", "SSH_CONNECTION": "203.0.113.7 51000 192.0.2.10 22"}])
def test_the_link_is_printed_only_to_a_terminal(tmp_path, launch, monkeypatch, capsys, caplog, session):
    caplog.set_level(logging.DEBUG)
    for name, value in session.items():
        monkeypatch.setenv(name, value)
    state = tmp_path / "state"
    seen = {}
    launch.during = lambda: seen.update(page=(state / sign_in.SIGN_IN_FILE).read_text())
    assert launcher.main(["--state-dir", str(state)]) == 0
    out, err = capsys.readouterr()
    for stream in (out, err):
        assert launch.token not in stream and "#token=" not in stream
    assert out.splitlines() == [
        "Studio: http://127.0.0.1:8220", f"Owner token file (mode 0600): {state / OWNER_FILE}",
        f"Sign-in page (mode 0600): {state / sign_in.SIGN_IN_FILE}"]
    assert link(launch.token) in seen["page"]
    assert launch.opened == []
    assert not any(launch.token in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("browser", ["opened", "not_a_browser", "no_opener"])
def test_the_link_prints_only_when_no_browser_was_sent_the_page(tmp_path, launch, monkeypatch, browser):
    monkeypatch.setenv("DISPLAY", ":0")
    terminal = Terminal()
    monkeypatch.setattr(sys, "stdout", terminal)
    handler = ".html opens with chat.desktop; the web browser is firefox.desktop. To change it: xdg-mime default firefox.desktop text/html"
    launch.browser = {"opened": (sign_in.OPENED, None), "not_a_browser": (sign_in.NOT_A_BROWSER, handler),
                      "no_opener": (sign_in.NO_OPENER, None)}[browser]
    state = tmp_path / "state"
    seen = {}
    launch.during = lambda: seen.update(page=(state / sign_in.SIGN_IN_FILE).is_file())
    assert launcher.main(["--state-dir", str(state)]) == 0
    head = ["Studio: http://127.0.0.1:8220", f"Owner token: {launch.token}"]
    if browser == "opened":
        assert lines_of(terminal) == [*head, sign_in.OPENED, sign_in.STOP_HINT]
        assert "#token=" not in terminal.getvalue()
    else:
        status = [sign_in.NOT_A_BROWSER, handler] if browser == "not_a_browser" else [sign_in.NO_OPENER]
        assert lines_of(terminal) == [*head, "Sign-in link:", "  " + link(launch.token), *status, sign_in.STOP_HINT]
    assert launch.opened == [state / sign_in.SIGN_IN_FILE]
    assert seen == {"page": True}
    assert not (state / sign_in.SIGN_IN_FILE).exists()


@pytest.mark.parametrize("session", [{}, {"DISPLAY": ":0"}, {"DISPLAY": ":0", "SSH_CONNECTION": "203.0.113.7 51000 192.0.2.10 22"}])
def test_management_never_makes_a_link_or_a_page(tmp_path, launch, monkeypatch, capsys, session):
    for name, value in session.items():
        monkeypatch.setenv(name, value)
    state = tmp_path / "state"
    seen = []
    launch.during = lambda: seen.append(state_names(state))
    terminal = Terminal()
    with monkeypatch.context() as patched:
        patched.setattr(sys, "stdout", terminal)
        assert launcher.main(["--state-dir", str(state), "--allow-management"]) == 0
    assert lines_of(terminal) == ["Studio: http://127.0.0.1:8220", f"Owner token: {launch.token}",
                                  sign_in.MANAGEMENT_NO_LINK, sign_in.STOP_HINT]
    assert "#token=" not in terminal.getvalue()
    # In pipe mode only the token file is written.
    assert launcher.main(["--state-dir", str(state), "--allow-management"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Studio: http://127.0.0.1:8220", f"Owner token file (mode 0600): {state / OWNER_FILE}"]
    assert sign_in.SIGN_IN_FILE not in seen[0] | seen[1] and OWNER_FILE not in seen[0] and OWNER_FILE in seen[1]
    # A terminal that chose --token-file gets the token file line alone, as in pipe mode.
    target = private(tmp_path / "secrets") / "studio-token"
    terminal = Terminal()
    with monkeypatch.context() as patched:
        patched.setattr(sys, "stdout", terminal)
        assert launcher.main(["--state-dir", str(state), "--allow-management", "--token-file", str(target)]) == 0
    assert lines_of(terminal) == ["Studio: http://127.0.0.1:8220", f"Owner token file (mode 0600): {target}"]
    assert sign_in.SIGN_IN_FILE not in seen[2]
    assert launch.opened == []


@pytest.mark.parametrize("session", [{"DISPLAY": ":0"}, {}, {"SSH_CONNECTION": "203.0.113.7 51000 192.0.2.10 22"}])
def test_token_file_on_a_terminal_never_prints_the_link(tmp_path, launch, monkeypatch, session):
    for name, value in session.items():
        monkeypatch.setenv(name, value)
    terminal = Terminal()
    monkeypatch.setattr(sys, "stdout", terminal)
    launch.browser = (sign_in.NO_OPENER, None)
    state = tmp_path / "state"
    target = private(tmp_path / "secrets") / "studio-token"
    assert launcher.main(["--state-dir", str(state), "--token-file", str(target)]) == 0
    text = terminal.getvalue()
    assert launch.token not in text and "#token=" not in text and "Sign-in link:" not in text
    assert lines_of(terminal)[:3] == ["Studio: http://127.0.0.1:8220", f"Owner token file (mode 0600): {target}",
                                      f"Sign-in page (mode 0600): {state / sign_in.SIGN_IN_FILE}"]
    assert lines_of(terminal)[-1] == sign_in.STOP_HINT


def test_sign_in_page_is_private_and_removed_at_exit(tmp_path, launch):
    state = tmp_path / "state"
    page = state / sign_in.SIGN_IN_FILE
    seen = {}

    def during():
        info = os.lstat(page)
        seen.update(mode=stat.S_IMODE(info.st_mode), owner=info.st_uid, regular=stat.S_ISREG(info.st_mode),
                    text=page.read_text())
    launch.during = during
    assert launcher.main(["--state-dir", str(state)]) == 0
    assert (seen["mode"], seen["owner"], seen["regular"]) == (0o600, os.getuid(), True)
    assert seen["text"] == sign_in.sign_in_page(link(launch.token))
    assert sign_in.SIGN_IN_PAGE.fullmatch(seen["text"])
    assert not page.exists()


def test_sign_in_page_cleanup_never_removes_a_replacement(tmp_path, launch):
    state = tmp_path / "state"
    page = state / sign_in.SIGN_IN_FILE

    def replace():
        replacement = state / "other"
        replacement.write_text("not the companion's")
        os.replace(replacement, page)
    launch.during = replace
    assert launcher.main(["--state-dir", str(state)]) == 0
    assert page.read_text() == "not the companion's"


def test_sign_in_page_refuses_a_foreign_file(tmp_path, launch):
    state = private(tmp_path / "state")
    page = state / sign_in.SIGN_IN_FILE
    page.write_text("<!doctype html><p>someone else's page</p>\n")
    page.chmod(0o600)
    with pytest.raises(SystemExit) as stopped:
        launcher.main(["--state-dir", str(state)])
    assert stopped.value.code == sign_in.SIGN_IN_TAKEN
    assert page.read_text() == "<!doctype html><p>someone else's page</p>\n"
    assert launch.run is None and lock_is_free(state)
    assert not (state / OWNER_FILE).exists()


def test_token_file_cannot_name_the_sign_in_page(tmp_path, launch):
    state = private(tmp_path / "state")
    with pytest.raises(SystemExit) as stopped:
        launcher.main(["--state-dir", str(state), "--token-file", str(state / sign_in.SIGN_IN_FILE)])
    assert isinstance(stopped.value.code, str) and "cannot replace a companion state file" in stopped.value.code
    assert sign_in.SIGN_IN_FILE in launcher.STATE_FILES
    assert launch.run is None and not (state / sign_in.SIGN_IN_FILE).exists()


def fake_desktop(answers, opened, exit_code=0):
    """Stand-ins for xdg-mime, xdg-open and PATH lookup."""
    def run(argv, *args, **kwargs):
        return subprocess.CompletedProcess(argv, 0, answers[argv[-1]], "")

    class Opener:
        def __init__(self, argv, **kwargs):
            opened.append((argv, kwargs))

        def wait(self, timeout=None):
            return exit_code

    return run, Opener, lambda name: f"/usr/bin/{name}"


@pytest.mark.parametrize("session", [
    {"DISPLAY": ":0"}, {"WAYLAND_DISPLAY": "wayland-0"},
    {"DISPLAY": ":0", "SSH_CONNECTION": "203.0.113.7 51000 192.0.2.10 22"},
    {"DISPLAY": "localhost:10.0", "SSH_CONNECTION": "203.0.113.7 51000 192.0.2.10 2222"},
    {"SSH_CONNECTION": "fe80::2%eth0 51000 fe80::1%eth0 22"},
    {"DISPLAY": ":0", "TMUX": "tmux-socket,1,0"}, {"DISPLAY": ":0", "STY": "1.pts-0.studio-host"},
    {},
], ids=["x11", "wayland", "ssh", "ssh_x11_port", "ssh_scoped", "tmux", "screen", "headless"])
def test_routes(tmp_path, launch, monkeypatch, session):
    for name, value in session.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LOGNAME", "operator")
    monkeypatch.setenv("USER", "operator")
    opened = []
    run, opener, which = fake_desktop({"text/html": "firefox.desktop\n", "x-scheme-handler/http": "firefox.desktop\n"},
                                      opened)
    monkeypatch.setattr(sign_in, "open_in_browser", lambda path: REAL_OPEN(path, run=run, popen=opener, which=which))
    terminal = Terminal()
    monkeypatch.setattr(sys, "stdout", terminal)
    state = tmp_path / "state"
    assert launcher.main(["--state-dir", str(state), "--port", "18220"]) == 0
    text, lines = terminal.getvalue(), lines_of(terminal)
    head = ["Studio: http://127.0.0.1:18220", f"Owner token: {launch.token}"]
    if "SSH_CONNECTION" not in session and ("DISPLAY" in session or "WAYLAND_DISPLAY" in session) \
            and not {"TMUX", "STY"} & set(session):
        # The handler check ran, then xdg-open got the page's file: URI, never the token.
        ((argv, kwargs),) = opened
        assert argv == ["/usr/bin/xdg-open", (state / sign_in.SIGN_IN_FILE).as_uri()]
        assert kwargs["start_new_session"] is True
        assert lines == [*head, sign_in.OPENED, sign_in.STOP_HINT]
        return
    assert opened == [] and not (state / sign_in.SIGN_IN_FILE).exists()
    assert lines[:4] == [*head, "Sign-in link:", "  " + link(launch.token, 18220)]
    assert lines[-1] == sign_in.STOP_HINT
    if "SSH_CONNECTION" not in session:
        assert lines[4:] == [sign_in.NO_DISPLAY, sign_in.STOP_HINT]
        return
    assert lines[4] == sign_in.SSH_SESSION and lines[-2] == sign_in.THEN_OPEN_THERE
    assert lines[4:-1] == sign_in.ssh_forward_hint(os.environ, 18220)
    assert "-L 18220:127.0.0.1:18220" in text
    if "fe80" in session["SSH_CONNECTION"]:
        assert "<host>" in text and "fe80" not in text and "%eth0" not in text
    else:
        assert "operator@192.0.2.10" in text
        assert ("-p 2222" in text) == session["SSH_CONNECTION"].endswith(" 2222")


def test_no_open_starts_no_browser_and_writes_no_page_on_a_terminal(tmp_path, launch, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    terminal = Terminal()
    monkeypatch.setattr(sys, "stdout", terminal)
    state = tmp_path / "state"
    seen = []
    launch.during = lambda: seen.append(state_names(state))
    assert launcher.main(["--state-dir", str(state), "--no-open"]) == 0
    assert lines_of(terminal) == ["Studio: http://127.0.0.1:8220", f"Owner token: {launch.token}",
                                  "Sign-in link:", "  " + link(launch.token), sign_in.STOP_HINT]
    assert launch.opened == []
    assert not {OWNER_FILE, sign_in.SIGN_IN_FILE} & seen[0]


def test_a_pipe_never_starts_a_browser_even_with_a_display(tmp_path, launch, monkeypatch, capsys):
    monkeypatch.setenv("DISPLAY", ":0")
    state = tmp_path / "state"
    seen = []
    launch.during = lambda: seen.append(state_names(state))
    assert launcher.main(["--state-dir", str(state)]) == 0
    assert launch.opened == []
    assert {OWNER_FILE, sign_in.SIGN_IN_FILE} <= seen[0]
    assert "#token=" not in capsys.readouterr().out


@pytest.mark.parametrize("config", ["broken", "missing", "shared"])
def test_a_broken_config_never_blocks_startup(tmp_path, launch, config):
    path = tmp_path / "cluster.toml"
    if config != "missing":
        path.write_text("[studio\n" if config == "broken" else '[studio]\ndriver_command = "bash /home/user/x.sh"\n')
        path.chmod(0o600 if config == "broken" else 0o666)
    assert launcher.main(["--state-dir", str(tmp_path / "state"), "--config", str(path)]) == 0
    assert launch.run is not None
    assert launch.operations.driver_start == {"command": None, "source": None, "config": str(path)}
