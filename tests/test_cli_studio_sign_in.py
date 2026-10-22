"""Sign-in helpers shared by the companion and dgxm studio; aiohttp-free, so no extra is needed."""
from __future__ import annotations

import errno
import os
import socket
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.studio import sign_in
from dgx_monarch.studio.contracts import StudioError

TOKEN = "T" * 21 + "_-" + "k" * 20
URL = f"http://127.0.0.1:8220/#token={TOKEN}"


@pytest.fixture
def state(tmp_path):
    directory = tmp_path / "state"
    directory.mkdir(mode=0o700)
    return directory


def _write_page(state: Path, url: str = URL) -> Path:
    path = state / sign_in.SIGN_IN_FILE
    sign_in.write_private_file(path, sign_in.sign_in_page(url), state, earlier=sign_in.SIGN_IN_PAGE,
                               taken=sign_in.SIGN_IN_TAKEN)
    return path


def _write_token(path: Path, state: Path, token: str = TOKEN) -> tuple[int, int]:
    return sign_in.write_private_file(path, token + "\n", state, earlier=sign_in.EARLIER_TOKEN,
                                      taken=sign_in.TOKEN_FILE_TAKEN)


# Link and page


def test_sign_in_link_brackets_an_ipv6_host():
    assert sign_in.sign_in_link("127.0.0.1", 8220, TOKEN) == URL
    assert sign_in.sign_in_link("::1", 8220, TOKEN) == f"http://[::1]:8220/#token={TOKEN}"
    assert sign_in.sign_in_link("[::1]", 8220, TOKEN) == f"http://[::1]:8220/#token={TOKEN}"


def test_link_lines_put_the_link_on_its_own_line():
    assert sign_in.link_lines(URL) == ["Sign-in link:", "  " + URL]


def test_sign_in_page_round_trips_and_refuses_other_urls(state):
    page = sign_in.sign_in_page(URL)
    assert page == (
        "<!doctype html>\n"
        '<meta charset="utf-8">\n'
        '<meta name="referrer" content="no-referrer">\n'
        f'<meta http-equiv="refresh" content="0; url={URL}">\n'
        "<title>DGX Monarch Studio</title>\n"
        f'<p><a href="{URL}" rel="noreferrer">Open DGX Monarch Studio</a></p>\n')
    path = _write_page(state)
    assert (path.stat().st_mode & 0o777) == 0o600
    url, written = sign_in.read_sign_in_page(path)
    assert url == URL and written == path.stat().st_mtime_ns
    ipv6 = f"http://[::1]:8220/#token={TOKEN}"
    assert sign_in.read_sign_in_page(_write_page(state, ipv6))[0] == ipv6
    for bad in (
            f"http://localhost:8220/#token={TOKEN}", f"https://127.0.0.1:8220/#token={TOKEN}",
            f"http://192.0.2.10:8220/#token={TOKEN}", f"http://127.0.0.1:8220/#token={TOKEN[:31]}",
            f"http://127.0.0.1:8220/?x=1#token={TOKEN}", f"http://127.0.0.1:8220/#token={TOKEN}\"",
            f"http://127.0.0.1:8220/#token={TOKEN}<", f"http://127.0.0.1:8220/#token={TOKEN}\n",
            f"http://127.0.0.1:8220/#token={'x' * 129}", "http://127.0.0.1:8220/"):
        with pytest.raises(ValueError):
            sign_in.sign_in_page(bad)


def test_a_page_with_two_different_urls_is_another_file(state):
    path = state / sign_in.SIGN_IN_FILE
    other = f"http://127.0.0.1:8221/#token={TOKEN}"
    path.write_text(sign_in.sign_in_page(URL).replace(f'href="{URL}"', f'href="{other}"'))
    path.chmod(0o600)
    with pytest.raises(StudioError) as raised:
        sign_in.read_sign_in_page(path)
    assert str(raised.value) == sign_in.SIGN_IN_TAKEN


# Private files


def test_reads_of_absent_files_are_none(state):
    assert sign_in.read_private_file(state / "missing", 10) is None
    assert sign_in.read_sign_in_page(state / sign_in.SIGN_IN_FILE) is None
    assert sign_in.read_token_file(state / sign_in.OWNER_FILE) is None
    assert sign_in.read_token_file(state / "no-such-directory" / sign_in.OWNER_FILE) is None


def test_private_reads_refuse_symlinks_other_owners_modes_and_shapes(state, monkeypatch, tmp_path):
    page = _write_page(state)
    token = state / sign_in.OWNER_FILE
    _write_token(token, state)
    assert sign_in.read_token_file(token)[0] == TOKEN

    def refused(path: Path) -> None:
        with pytest.raises(StudioError) as raised:
            sign_in.read_sign_in_page(path)
        assert (raised.value.code, str(raised.value)) == ("token_file", sign_in.SIGN_IN_TAKEN)
        with pytest.raises(StudioError) as raised:
            sign_in.read_token_file(path)
        assert (raised.value.code, str(raised.value)) == ("token_file", sign_in.TOKEN_FILE_TAKEN)

    link = state / "link"
    link.symlink_to(page)
    refused(link)
    directory = state / "directory"
    directory.mkdir()
    refused(directory)
    loose = tmp_path / "loose"
    loose.write_text(TOKEN + "\n")
    loose.chmod(0o640)
    refused(loose)
    large = state / "large"
    large.write_text("x" * 4097)
    large.chmod(0o600)
    refused(large)
    other = state / "other"
    other.write_text("not a token or a page\n")
    other.chmod(0o600)
    refused(other)
    binary = state / "binary"
    binary.write_bytes(b"\xff" * 40)
    binary.chmod(0o600)
    refused(binary)
    real = os.getuid()
    monkeypatch.setattr(sign_in.os, "getuid", lambda: real + 1)
    refused(page)
    refused(token)


def test_read_private_file_reports_shapes_as_value_errors(state):
    path = state / "data"
    path.write_bytes(b"12345")
    path.chmod(0o600)
    data, written = sign_in.read_private_file(path, 5)
    assert (data, written) == (b"12345", path.stat().st_mtime_ns)
    with pytest.raises(ValueError, match="larger than 4 bytes"):
        sign_in.read_private_file(path, 4)
    path.chmod(0o604)
    with pytest.raises(ValueError, match="not private"):
        sign_in.read_private_file(path, 5)


def test_private_reads_refuse_a_fifo_without_blocking(state):
    fifo = state / sign_in.SIGN_IN_FILE
    os.mkfifo(fifo, 0o600)
    outcome: list[BaseException] = []

    def read() -> None:
        for reader in (sign_in.read_sign_in_page, sign_in.read_token_file):
            try:
                reader(fifo)
            except BaseException as exc:  # recorded for the assertions below
                outcome.append(exc)

    worker = threading.Thread(target=read, daemon=True)
    worker.start()
    worker.join(5)
    assert not worker.is_alive(), "a FIFO blocked the read"
    assert [type(exc) for exc in outcome] == [StudioError, StudioError]


def test_a_token_file_replaces_only_an_earlier_token(state, tmp_path):
    path = state / sign_in.OWNER_FILE
    identity = _write_token(path, state)
    assert identity == (path.stat().st_dev, path.stat().st_ino)
    assert path.read_text() == TOKEN + "\n"
    assert (path.stat().st_mode & 0o777) == 0o600
    newer = "N" * 43
    _write_token(path, state, newer)
    assert sign_in.read_token_file(path)[0] == newer
    path.write_text("a secret that is not a token\n")
    path.chmod(0o600)
    with pytest.raises(StudioError) as raised:
        _write_token(path, state)
    assert str(raised.value) == sign_in.TOKEN_FILE_TAKEN
    assert path.read_text() == "a secret that is not a token\n"
    assert not list(state.glob(".*.tmp"))


def test_a_symlink_at_the_token_path_is_refused_and_its_target_untouched(state, tmp_path):
    target = tmp_path / "target"
    target.write_text("keep\n")
    path = state / sign_in.OWNER_FILE
    path.symlink_to(target)
    with pytest.raises(StudioError):
        _write_token(path, state)
    assert target.read_text() == "keep\n"


def test_the_writer_keeps_the_directory_rules(state, tmp_path):
    with pytest.raises(StudioError, match="existing directory"):
        _write_token(tmp_path / "missing" / "token", state)
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    with pytest.raises(StudioError, match="mode 0700"):
        _write_token(shared / "token", state)
    with pytest.raises(UnicodeEncodeError):
        sign_in.write_private_file(state / "x", "café", state, earlier=sign_in.EARLIER_TOKEN,
                                   taken=sign_in.TOKEN_FILE_TAKEN)


@pytest.mark.parametrize("name", sorted(sign_in.STATE_FILES))
def test_token_file_cannot_name_a_state_file(state, name):
    with pytest.raises(StudioError) as raised:
        _write_token(state / name, state)
    assert str(raised.value) == "The owner token file cannot replace a companion state file."
    assert not (state / name).exists()


def test_only_the_page_writer_may_write_sign_in_html(state):
    assert "sign-in.html" in sign_in.STATE_FILES
    path = _write_page(state)
    # A second start replaces its earlier page.
    _write_page(state, f"http://127.0.0.1:8221/#token={TOKEN}")
    assert sign_in.read_sign_in_page(path)[0] == f"http://127.0.0.1:8221/#token={TOKEN}"
    # The page pattern is the only key: the token writer is refused there.
    with pytest.raises(StudioError, match="cannot replace a companion state file"):
        _write_token(path, state)


def test_a_foreign_file_at_the_page_path_is_never_replaced(state):
    path = state / sign_in.SIGN_IN_FILE
    path.write_text("<html>something else</html>\n")
    path.chmod(0o600)
    with pytest.raises(StudioError) as raised:
        _write_page(state)
    assert str(raised.value) == sign_in.SIGN_IN_TAKEN
    assert path.read_text() == "<html>something else</html>\n"


# Listener


def test_bind_listener_holds_the_port_and_names_a_busy_one():
    listener = sign_in.bind_listener("127.0.0.1", 0)
    try:
        host, port = listener.getsockname()
        assert host == "127.0.0.1" and port > 0
        assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)
        with pytest.raises(SystemExit) as raised:
            sign_in.bind_listener("127.0.0.1", port)
        assert str(raised.value) == sign_in.PORT_IN_USE
    finally:
        listener.close()


def test_bind_listener_names_other_failures(monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError(errno.EACCES, "Permission denied (while attempting to bind)")

    monkeypatch.setattr(sign_in.socket, "create_server", fail)
    with pytest.raises(SystemExit) as raised:
        sign_in.bind_listener("::1", 1023)
    assert str(raised.value) == f"Cannot listen on [::1]:1023: {os.strerror(errno.EACCES)}."


def test_bind_listener_uses_ipv6_for_an_ipv6_host(monkeypatch):
    seen = {}

    def record(address, **kwargs):
        seen.update(address=address, **kwargs)
        return "listener"

    monkeypatch.setattr(sign_in.socket, "create_server", record)
    assert sign_in.bind_listener("::1", 8220) == "listener"
    assert seen == {"address": ("::1", 8220), "family": socket.AF_INET6, "backlog": 128}


# Browser route and handlers


@pytest.mark.parametrize(("environ", "route"), [
    ({"DISPLAY": ":0"}, "local"),
    ({"WAYLAND_DISPLAY": "wayland-0"}, "local"),
    ({"DISPLAY": ":0", "SSH_CONNECTION": "192.0.2.20 50000 192.0.2.10 22"}, "ssh"),
    ({"DISPLAY": "localhost:10.0", "SSH_CONNECTION": "192.0.2.20 50000 192.0.2.10 22"}, "ssh"),
    ({"SSH_CONNECTION": ""}, "ssh"),
    ({"DISPLAY": ":0", "TMUX": "tmux-socket,1,0"}, "none"),
    ({"DISPLAY": ":0", "STY": "1234.pts-0.studio-host"}, "none"),
    ({"DISPLAY": ""}, "none"),
    ({}, "none"),
])
def test_browser_route_reads_only_the_given_environment(monkeypatch, environ, route):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    assert sign_in.browser_route(environ) == route


def _runner(answers: dict[str, object]):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        answer = answers[argv[-1]]
        if isinstance(answer, BaseException):
            raise answer
        code, out = answer
        return subprocess.CompletedProcess(argv, code, out, "")

    run.calls = calls
    return run


class _Popen:
    def __init__(self, code: int | None = 0, error: OSError | None = None):
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.code, self.error = code, error

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.error is not None:
            raise self.error
        code = self.code

        def wait(timeout=None):
            if code is None:
                raise subprocess.TimeoutExpired(argv, timeout)
            return code

        return SimpleNamespace(wait=wait)


def _which(name: str) -> str | None:
    return f"/usr/bin/{name}"


BRAVE = (0, "brave-origin.desktop\n")


def test_browser_handlers_asks_xdg_mime_for_both_types():
    run = _runner({"text/html": BRAVE, "x-scheme-handler/http": BRAVE})
    assert sign_in.browser_handlers(run) == ("brave-origin.desktop", "brave-origin.desktop")
    assert [argv for argv, _ in run.calls] == [
        ["xdg-mime", "query", "default", "text/html"],
        ["xdg-mime", "query", "default", "x-scheme-handler/http"]]
    assert all(kwargs["timeout"] == 5 and kwargs["check"] is False for _, kwargs in run.calls)


@pytest.mark.parametrize("answers", [
    {"text/html": (0, "chat.desktop\n"), "x-scheme-handler/http": BRAVE},
    {"text/html": (0, "\n"), "x-scheme-handler/http": BRAVE},
    {"text/html": (1, "brave-origin.desktop\n"), "x-scheme-handler/http": BRAVE},
    {"text/html": FileNotFoundError("xdg-mime"), "x-scheme-handler/http": BRAVE},
    {"text/html": subprocess.TimeoutExpired(["xdg-mime"], 5), "x-scheme-handler/http": BRAVE},
    {"text/html": (0, "\x1b]0;title\x07.desktop\n"), "x-scheme-handler/http": BRAVE},
])
def test_open_in_browser_requires_one_desktop_entry_for_html_and_http(state, answers):
    page = _write_page(state)
    popen = _Popen()
    status, line = sign_in.open_in_browser(page, run=_runner(answers), popen=popen, which=_which)
    assert status == sign_in.NOT_A_BROWSER
    assert popen.calls == []
    if answers["text/html"] == (0, "chat.desktop\n"):
        assert line == (".html opens with chat.desktop; the web browser is brave-origin.desktop. "
                        "To change it: xdg-mime default brave-origin.desktop text/html")
    else:
        assert line == "xdg-mime could not name the .html or http handler."


def test_open_in_browser_passes_the_file_uri_never_the_token(state):
    page = _write_page(state)
    popen = _Popen(code=0)
    run = _runner({"text/html": BRAVE, "x-scheme-handler/http": BRAVE})
    assert sign_in.open_in_browser(page, run=run, popen=popen, which=_which) == (sign_in.OPENED, None)
    [(argv, kwargs)] = popen.calls
    assert argv == ["/usr/bin/xdg-open", page.as_uri()]
    assert kwargs["start_new_session"] is True and kwargs["close_fds"] is True
    assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
    for argument in [*argv, *(str(value) for value in kwargs.values())]:
        assert TOKEN not in argument and "#token=" not in argument
    for argv, _ in run.calls:
        assert TOKEN not in " ".join(argv)


def test_open_in_browser_counts_an_opener_that_keeps_running_as_sent(state):
    page = _write_page(state)
    run = _runner({"text/html": BRAVE, "x-scheme-handler/http": BRAVE})
    assert sign_in.open_in_browser(page, run=run, popen=_Popen(code=None), which=_which)[0] == sign_in.OPENED


def test_open_in_browser_without_xdg_open_reports_it(state):
    page = _write_page(state)
    run = _runner({"text/html": BRAVE, "x-scheme-handler/http": BRAVE})
    popen = _Popen()
    assert sign_in.open_in_browser(page, run=run, popen=popen, which=lambda _name: None) == (
        sign_in.NO_OPENER, None)
    assert popen.calls == []


def test_open_in_browser_reports_a_failed_opener(state):
    page = _write_page(state)
    run = _runner({"text/html": BRAVE, "x-scheme-handler/http": BRAVE})
    assert sign_in.open_in_browser(page, run=run, popen=_Popen(code=4), which=_which) == (sign_in.NO_OPENER, None)
    assert sign_in.open_in_browser(page, run=run, popen=_Popen(error=PermissionError(errno.EACCES, "denied")),
                                   which=_which) == (sign_in.NO_OPENER, None)


def test_open_plain_address_passes_no_token():
    popen = _Popen(code=0)
    assert sign_in.open_plain_address("http://127.0.0.1:8220/", popen=popen, which=_which) == sign_in.OPENED_PLAIN
    [(argv, kwargs)] = popen.calls
    assert argv == ["/usr/bin/xdg-open", "http://127.0.0.1:8220/"]
    assert kwargs["start_new_session"] is True
    assert sign_in.open_plain_address("http://127.0.0.1:8220/", popen=_Popen(code=3), which=_which) == (
        sign_in.NO_OPENER)
    assert sign_in.open_plain_address("http://127.0.0.1:8220/", popen=_Popen(), which=lambda _n: None) == (
        sign_in.NO_OPENER)
    for refused in (URL, "file:///home/user/sign-in.html", "http://127.0.0.1:8220/#"):
        with pytest.raises(ValueError):
            sign_in.open_plain_address(refused, popen=_Popen(), which=_which)


def test_terminal_stdout_counts_any_doubt_as_no(monkeypatch):
    monkeypatch.setattr(sign_in.sys, "stdout", SimpleNamespace(isatty=lambda: True))
    assert sign_in.terminal_stdout() is True
    monkeypatch.setattr(sign_in.sys, "stdout", None)
    assert sign_in.terminal_stdout() is False

    def closed():
        raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(sign_in.sys, "stdout", SimpleNamespace(isatty=closed))
    assert sign_in.terminal_stdout() is False
    monkeypatch.setattr(sign_in.sys, "stdout", object())
    assert sign_in.terminal_stdout() is False


# SSH host rule and lines


ACCEPTED = ["studio-host", "studio_head", "user@studio-host", "studio.example", "192.0.2.10", "2001:db8::5",
            "user@2001:db8::5", "u@::1", "Alice.B@studio-host", "_runner@studio-host", "a"]
REFUSED = ["-oProxyCommand=x", "a b", "a;b", "$(x)", "`id`", "a`b", "ssh://studio-host", "user@host@x",
           "host:22", "a%b", "_lead", "studio\x07host", "", "fe80::1%eth0", "fe80::1%`id`", "fe80::1%$(id)",
           "2001:db8::5%;id", "::1%-oX=y", "fe80::1%\n", "user@fe80::1%eth0", "studio-host\n", "studio-host.",
           "@studio-host", "-user@studio-host", "user name@studio-host", "[2001:db8::5]", "x" * 254,
           "u" * 33 + "@studio-host", "café", "studio-host -L"]


@pytest.mark.parametrize("text", ACCEPTED)
def test_ssh_destination_rule_accepts(text):
    assert sign_in.valid_ssh_destination(text) is True


@pytest.mark.parametrize("text", REFUSED)
def test_ssh_destination_rule(text):
    assert sign_in.valid_ssh_destination(text) is False


def test_forward_line_is_the_owner_form():
    # The owner's form with -o ExitOnForwardFailure=yes, so ssh exits instead of
    # opening a session without the tunnel when another program holds the port.
    assert sign_in.forward_line("user@192.0.2.10", 8220) == (
        "ssh -o ExitOnForwardFailure=yes -L 8220:127.0.0.1:8220 user@192.0.2.10")
    assert sign_in.forward_line("studio-host", 8221, ssh_port=2222) == (
        "ssh -p 2222 -o ExitOnForwardFailure=yes -L 8221:127.0.0.1:8221 studio-host")
    assert sign_in.forward_line("studio-host", 8220, target="::1") == (
        "ssh -o ExitOnForwardFailure=yes -L 8220:[::1]:8220 studio-host")
    assert sign_in.forward_line("studio-host", 8220, target="[::1]") == (
        "ssh -o ExitOnForwardFailure=yes -L 8220:[::1]:8220 studio-host")
    assert sign_in.forward_line(sign_in.HOST_PLACEHOLDER, 8220) == (
        "ssh -o ExitOnForwardFailure=yes -L 8220:127.0.0.1:8220 <host>")
    assert "-N" not in sign_in.forward_line("studio-host", 8220).split()


def test_ssh_session_destination_reads_the_third_field_and_port(monkeypatch):
    monkeypatch.setattr(sign_in.getpass, "getuser", lambda: "user")
    assert sign_in.ssh_session_destination({"SSH_CONNECTION": "192.0.2.20 50514 192.0.2.10 22"}) == (
        "user@192.0.2.10", None)
    assert sign_in.ssh_session_destination({"SSH_CONNECTION": "192.0.2.20 50514 192.0.2.10 2222"}) == (
        "user@192.0.2.10", 2222)
    assert sign_in.ssh_session_destination({"SSH_CONNECTION": "2001:db8::20 50514 2001:db8::5 22"}) == (
        "user@2001:db8::5", None)
    for value in ("", "192.0.2.20 50514 192.0.2.10", "192.0.2.20 50514 192.0.2.10 x",
                  "192.0.2.20 50514 192.0.2.10 0", "192.0.2.20 50514 192.0.2.10 65536",
                  "192.0.2.20 50514 192.0.2.10 22 extra", "192.0.2.20 50514 $(id) 22"):
        assert sign_in.ssh_session_destination({"SSH_CONNECTION": value}) is None, value
    assert sign_in.ssh_session_destination({}) is None
    monkeypatch.setattr(sign_in.getpass, "getuser", lambda: "user name")
    assert sign_in.ssh_session_destination({"SSH_CONNECTION": "192.0.2.20 50514 192.0.2.10 22"}) is None


def test_ssh_session_destination_drops_a_scoped_address(monkeypatch):
    monkeypatch.setattr(sign_in.getpass, "getuser", lambda: "user")
    for address in ("fe80::1%eth0", "fe80::1%`id`", "fe80::1%$(id)"):
        assert sign_in.ssh_session_destination({"SSH_CONNECTION": f"fe80::2 50514 {address} 22"}) is None


def test_another_way_line_names_the_port_twice():
    assert sign_in.another_way_line(8220) == (
        "If you reach this host another way, add -L 8220:127.0.0.1:8220 to the ssh command you use, or set a "
        "local forward of port 8220 to 127.0.0.1:8220 in your SSH app.")
    assert sign_in.another_way_line(8221, target="::1") == (
        "If you reach this host another way, add -L 8221:[::1]:8221 to the ssh command you use, or set a "
        "local forward of port 8221 to [::1]:8221 in your SSH app.")


def test_ssh_forward_hint_lines(monkeypatch):
    monkeypatch.setattr(sign_in.getpass, "getuser", lambda: "user")
    environ = {"SSH_CONNECTION": "192.0.2.20 50514 192.0.2.10 2222", "DISPLAY": ":0"}
    assert sign_in.ssh_forward_hint(environ, 8220) == [
        sign_in.SSH_SESSION,
        "  ssh -p 2222 -o ExitOnForwardFailure=yes -L 8220:127.0.0.1:8220 user@192.0.2.10",
        sign_in.another_way_line(8220),
        "Then open the sign-in link there."]
    scoped = {"SSH_CONNECTION": "fe80::2 50514 fe80::1%eth0 22"}
    assert sign_in.ssh_forward_hint(scoped, 8220) == [
        sign_in.SSH_SESSION,
        "  ssh -o ExitOnForwardFailure=yes -L 8220:127.0.0.1:8220 <host>",
        sign_in.NAME_THE_STUDIO_HOST,
        sign_in.another_way_line(8220),
        "Then open the sign-in link there."]
    assert "dgxm studio tunnel" in sign_in.NAME_THE_STUDIO_HOST


# Constants


DESIGN_TEXT = {
    "PORT_IN_USE": "Another process already listens on the companion port; stop it, or choose another port: "
                   "[studio] port in cluster.toml, or --port for dgxm-studio.",
    "SIGN_IN_TAKEN": "The sign-in page in the state directory holds another file; remove it, then start the "
                     "companion again.",
    "OPENED": "Sent the sign-in page to your web browser.",
    "OPENED_PLAIN": "Opened Studio in your web browser; paste the owner token into its form.",
    "NOT_A_BROWSER": "No browser opened: this desktop opens .html files with another program than your web "
                     "browser. Open the sign-in link above.",
    "NO_OPENER": "No browser opened: xdg-open is missing or failed. Open the sign-in link above.",
    "NO_DISPLAY": "No browser opened: no display here, or a tmux or screen session. Open the sign-in link above "
                  "in a browser on this computer, or run dgxm studio tunnel for another computer.",
    "SSH_SESSION": "No browser opened: this terminal is an SSH session. If the computer with the browser has no "
                   "tunnel yet, run this there:",
    "STOP_HINT": "Press Ctrl+C to stop Studio; dgxm studio up keeps it running instead.",
    "NOT_RUNNING": "Studio is not running. Start it with dgxm studio, or keep it running with dgxm studio up.",
    "NO_SIGN_IN_PAGE": "No sign-in page: a Studio started in a terminal printed its sign-in link or token there.",
    "NO_TOKEN_FILE": "No token file: a Studio started in a terminal printed its token there.",
    "NO_PIPE": "Refusing to print the token to a pipe or file; run this in a terminal or read the token file.",
    "MANAGEMENT_NO_LINK": "This Studio runs with --allow-management, so it never signs in by link; paste the "
                          "owner token into the page's form.",
    # The existing text of the companion's token-file refusal, now a constant.
    "TOKEN_FILE_TAKEN": "The owner token file path holds another file; choose a new path.",
    "OWNER_FILE": "owner-token",
    "SIGN_IN_FILE": "sign-in.html",
}


@pytest.mark.parametrize("name", sorted(DESIGN_TEXT))
def test_sign_in_constants_hold_the_design_text(name):
    assert getattr(sign_in, name) == DESIGN_TEXT[name]


def test_sign_in_patterns_and_state_files():
    assert sign_in.STATE_FILES == {"companion.lock", "studio.sqlite3", "studio.sqlite3-wal",
                                   "studio.sqlite3-shm", "sign-in.html"}
    assert sign_in.LINK.pattern == (
        r"http://(?:127\.\d{1,3}\.\d{1,3}\.\d{1,3}|\[::1\]):\d{1,5}/#token=[A-Za-z0-9_-]{32,128}")
    assert sign_in.EARLIER_TOKEN.fullmatch(TOKEN + "\n")
    assert not sign_in.EARLIER_TOKEN.fullmatch(TOKEN + "\n\n")
    assert not sign_in.EARLIER_TOKEN.fullmatch("x" * 31)
