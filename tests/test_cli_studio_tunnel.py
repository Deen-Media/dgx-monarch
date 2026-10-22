"""dgxm studio tunnel: the host rule, the port sources and the exact lines it prints.

Every case runs end to end through ``run_tunnel``. The home, the unit file,
the state directory, its lock and its sign-in page are real files under
``tmp_path``; only the user manager's answer (``unit_state.unit_status``)
and the readiness clocks are stood in for, so no test asks systemd or reads
the machine's clocks. ssh is never started: ``--run`` cases pass a recording
``execvp``. The token is fake.
"""
from __future__ import annotations

import fcntl
import io
import os
from pathlib import Path

import pytest

from dgx_monarch.studio import launch, sign_in, unit, unit_state
from dgx_monarch.studio.tunnel import (
    BAD_HOST,
    BROWSE,
    CONFIG_NOTE,
    FORWARD_FAILED,
    FROM_CONFIG,
    FROM_DEFAULT,
    FROM_FLAG,
    FROM_STUDIO,
    LINK_FOLLOWS,
    NAME_THE_HOST,
    NO_SSH,
    PORT_LINE,
    PORT_RANGE,
    RUN_THIS,
    SAME_ADDRESS,
    SIGN_IN_BY_FORM,
    SIGN_IN_BY_OPEN,
    run_tunnel,
)

FAKE_TOKEN = "fake_" + "0" * 38
SESSION = {"SSH_CONNECTION": "192.0.2.20 50022 192.0.2.10 22"}
SESSION_HOST = "user@192.0.2.10"
# The readiness clocks: wall time NOW and monotonic time MONO are the same instant.
NOW, MONO = 1_800_000_000.0, 5_000.0


class Rig:
    """One test's computer: a home, a state directory, a PATH holding ssh, and a stand-in user manager."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp = tmp_path
        self.home = tmp_path / "home"
        self.state = tmp_path / "xdg-state" / "dgx-monarch" / "studio"
        self.bin = tmp_path / "bin"
        self.work = tmp_path / "work"
        for directory in (self.home, self.state, self.bin, self.work):
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        ssh = self.bin / "ssh"
        ssh.write_text("#!/bin/sh\nexit 99\n")
        ssh.chmod(0o700)
        for name, value in (("HOME", str(self.home)), ("XDG_STATE_HOME", str(tmp_path / "xdg-state")),
                            ("PATH", str(self.bin)), ("LOGNAME", "user"), ("USER", "user")):
            monkeypatch.setenv(name, value)
        for name in ("DGXM_CLUSTER_TOML", "SSH_CONNECTION", "XDG_CONFIG_HOME", "XDG_RUNTIME_DIR"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.chdir(self.work)
        monkeypatch.setattr(unit_state, "_monotonic", lambda: MONO)
        monkeypatch.setattr(unit_state, "_realtime", lambda: NOW)
        self.status: unit_state.UnitStatus | None = None
        self.status_calls = 0
        monkeypatch.setattr(unit_state, "unit_status", self._unit_status)
        self.held: list[int] = []
        self.execs: list[tuple[str, list[str]]] = []

    def _unit_status(self, **_kwargs: object) -> unit_state.UnitStatus | None:
        self.status_calls += 1
        return self.status

    def close(self) -> None:
        for fd in self.held:
            os.close(fd)

    def _private(self, path: Path, text: str, age: float) -> None:
        path.write_text(text)
        path.chmod(0o600)
        os.utime(path, (NOW - age, NOW - age))

    def companion(self, port: int = 8232, host: str = "127.0.0.1", *, lock: bool = True, page: bool = True,
                  age: float = 10.0) -> str:
        """A companion's files in the state directory: its lock (held or free), its token file and page."""
        url = sign_in.sign_in_link(host, port, FAKE_TOKEN)
        fd = os.open(self.state / unit_state.LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
        if lock:
            fcntl.flock(fd, fcntl.LOCK_EX)
            self.held.append(fd)
        else:
            os.close(fd)
        self._private(self.state / sign_in.OWNER_FILE, FAKE_TOKEN + "\n", age)
        if page:
            self._private(self.state / sign_in.SIGN_IN_FILE, sign_in.sign_in_page(url), age)
        return url

    def service(self, port: int = 8231, *, management: bool = False, active: str = "active",
                started_ago: float = 60.0) -> None:
        """Our unit file, as dgxm studio up writes its first lines, and the user manager's answer."""
        path = self.home / ".config" / "systemd" / "user" / unit.UNIT_NAME
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_text("\n".join([
            unit.UNIT_MARKER,
            f"{unit.SETTING_PREFIX}state={self.state}",
            f"{unit.SETTING_PREFIX}port={port}",
            f"{unit.SETTING_PREFIX}driver=http://127.0.0.1:8188",
            f"{unit.SETTING_PREFIX}management={'yes' if management else 'no'}",
            "[Unit]", "Description=DGX Monarch Studio companion", "",
        ]))
        path.chmod(0o600)
        running = active == "active"
        self.status = unit_state.UnitStatus(
            load_state="loaded", active_state=active, sub_state="running" if running else "dead",
            main_pid=4242 if running else 0, control_group="/user.slice/dgxm-studio.service",
            fragment_path=str(path), start_monotonic_us=int((MONO - started_ago) * 1_000_000))

    def config(self, text: str, path: Path | None = None) -> Path:
        path = path or self.home / ".config" / "dgx-monarch" / "cluster.toml"
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_text(text)
        path.chmod(0o600)
        return path

    def call(self, host: str | None = None, *, port: int | None = None, run: bool = False,
             config: Path | None = None, environ: dict[str, str] | None = None,
             terminal: bool = True) -> tuple[int, list[str], list[str]]:
        out, err = io.StringIO(), io.StringIO()
        code = run_tunnel(host, port, run, config, environ={} if environ is None else environ,
                          terminal=terminal, stdout=out, stderr=err,
                          execvp=lambda file, argv: self.execs.append((file, list(argv))))
        return code, out.getvalue().splitlines(), err.getvalue().splitlines()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    made = Rig(tmp_path, monkeypatch)
    yield made
    made.close()


def block(port: int, source: str, forward: str, sign: list[str], target: str = "127.0.0.1") -> list[str]:
    """The design 7.3 lines, in order."""
    return [PORT_LINE.format(port=port, source=source), RUN_THIS, "  " + forward,
            sign_in.another_way_line(port, target=target), BROWSE.format(address=f"http://{target}:{port}/"),
            *sign, SAME_ADDRESS, FORWARD_FAILED]


def test_the_printed_lines_keep_their_text():
    # The docs quote these; a change here is a change there.
    assert PORT_LINE.format(port=8220, source=FROM_STUDIO) == "Port: 8220 (from the running Studio)"
    assert PORT_LINE.format(port=8220, source=FROM_FLAG) == "Port: 8220 (from --port)"
    assert PORT_LINE.format(port=8220, source=FROM_CONFIG) == "Port: 8220 (from [studio] port)"
    assert PORT_LINE.format(port=8220, source=FROM_DEFAULT) == (
        "Port: 8220 (default; pass --port if the Studio host sets another port in [studio])")
    assert RUN_THIS == "On the computer with the browser, run:"
    assert BROWSE.format(address="http://127.0.0.1:8220/") == "Then browse: http://127.0.0.1:8220/"
    assert LINK_FOLLOWS == "Sign in with this link in that browser:"
    assert SIGN_IN_BY_OPEN == ("Sign in with the link that dgxm studio open prints in an SSH session on the "
                               "Studio host.")
    assert SIGN_IN_BY_FORM == ("Sign in by pasting the owner token that dgxm studio token --value prints on "
                               "the Studio host.")
    assert CONFIG_NOTE.format(error="x") == "Note: x"
    assert NO_SSH == "ssh was not found on PATH."


def test_the_design_block_on_the_studio_host(rig):
    """The 7.3 example: an SSH session, a terminal, a live companion, no management."""
    url = rig.companion(port=8220)
    code, out, err = rig.call(environ=SESSION)
    assert (code, err) == (0, [])
    assert out == [
        "Port: 8220 (from the running Studio)",
        "On the computer with the browser, run:",
        "  ssh -o ExitOnForwardFailure=yes -L 8220:127.0.0.1:8220 user@192.0.2.10",
        "If you reach this host another way, add -L 8220:127.0.0.1:8220 to the ssh command you use, or set a "
        "local forward of port 8220 to 127.0.0.1:8220 in your SSH app.",
        "Then browse: http://127.0.0.1:8220/",
        "Sign in with this link in that browser:",
        "  " + url,
        "Keep 127.0.0.1 and the same port on both ends; Studio refuses localhost and any other port.",
        "If ssh says \"Could not request local forwarding\", another program on that computer holds the port and "
        "ssh opened the session without the tunnel: close it, free the port, and do not open the sign-in link "
        "until the tunnel works, because a page served on that port would read the token from the link.",
    ]


@pytest.mark.parametrize("host", [
    "studio-host", "studio_head", "user@studio-host", "studio.example", "192.0.2.10", "2001:db8::5",
    "user@2001:db8::5",
])
def test_accepted_hosts_reach_the_forward_line(rig, host):
    code, out, err = rig.call(host)
    assert (code, err) == (0, [])
    assert out == block(8220, FROM_DEFAULT, sign_in.forward_line(host, 8220), [SIGN_IN_BY_OPEN])
    assert out[2] == f"  ssh -o ExitOnForwardFailure=yes -L 8220:127.0.0.1:8220 {host}"


@pytest.mark.parametrize("host", [
    "-oProxyCommand=x", "a b", "a;b", "$(x)", "a`id`b", "ssh://studio-host", "user@host@x", "host:22", "a%b",
    "_lead", "studio\x1bhost", "", "fe80::1%eth0", "fe80::1%`id`", "fe80::1%$(id)", "2001:db8::5%;id",
    "::1%-oX=y", "fe80::1%\n", "user@fe80::1%eth0",
])
def test_refused_hosts_print_bad_host_and_nothing_else(rig, host):
    rig.companion()
    code, out, err = rig.call(host, run=True, environ=SESSION)
    assert (code, out, err) == (2, [], [BAD_HOST])
    assert rig.execs == []


@pytest.mark.parametrize("ssh_port,forward", [
    (22, "ssh -o ExitOnForwardFailure=yes -L 8220:127.0.0.1:8220 user@192.0.2.10"),
    (2222, "ssh -p 2222 -o ExitOnForwardFailure=yes -L 8220:127.0.0.1:8220 user@192.0.2.10"),
])
def test_the_derived_host_comes_from_the_ssh_session(rig, ssh_port, forward):
    code, out, err = rig.call(environ={"SSH_CONNECTION": f"192.0.2.20 50022 192.0.2.10 {ssh_port}"})
    assert (code, err) == (0, [])
    assert out == block(8220, FROM_DEFAULT, forward, [SIGN_IN_BY_OPEN])


@pytest.mark.parametrize("environ", [
    {"SSH_CONNECTION": "fe80::2%eth0 50022 fe80::1%eth0 22"},
    {"SSH_CONNECTION": "192.0.2.20 50022"},
    {},
])
def test_without_a_usable_session_the_host_must_be_named(rig, environ):
    rig.companion()
    code, out, err = rig.call(environ=environ)
    assert (code, out, err) == (2, [], [NAME_THE_HOST])


def test_a_named_host_never_reads_the_local_studio(rig, monkeypatch):
    """S4: a typed host may be run from a computer that runs its own Studio."""
    rig.companion(port=8232)
    rig.service(port=8231)

    def refuse(*_args, **_kwargs):
        raise AssertionError("a named host read the local Studio")

    for owner, name in ((unit, "read_owned_unit"), (unit, "unit_settings"), (unit_state, "unit_status"),
                        (unit_state, "lock_probe"), (unit_state, "wait_ready"),
                        (sign_in, "read_sign_in_page"), (launch, "default_state_dir")):
        monkeypatch.setattr(owner, name, refuse)
    code, out, err = rig.call("studio-host", environ=SESSION)
    assert (code, err) == (0, [])
    assert out == block(8220, FROM_DEFAULT, sign_in.forward_line("studio-host", 8220), [SIGN_IN_BY_OPEN])
    assert FAKE_TOKEN not in "\n".join(out)


def test_port_from_the_flag(rig):
    code, out, err = rig.call("studio-host", port=8240)
    assert (code, err) == (0, [])
    assert out == block(8240, FROM_FLAG, sign_in.forward_line("studio-host", 8240), [SIGN_IN_BY_OPEN])
    assert out[0] == "Port: 8240 (from --port)"


@pytest.mark.parametrize("table,port,source", [
    ("[studio]\nport = 8230\n", 8230, FROM_CONFIG),
    ("[studio]\nport = 8220\n", 8220, FROM_CONFIG),
    ("[studio]\ndriver = \"http://127.0.0.1:8188\"\n", 8220, FROM_DEFAULT),
    ("", 8220, FROM_DEFAULT),
])
def test_port_from_the_config_the_usual_search_finds(rig, table, port, source):
    rig.config(table)
    code, out, err = rig.call("studio-host")
    assert (code, err) == (0, [])
    assert out == block(port, source, sign_in.forward_line("studio-host", port), [SIGN_IN_BY_OPEN])


def test_port_from_a_typed_config_wins_over_the_search(rig):
    typed = rig.config("[studio]\nport = 8235\n", rig.work / "elsewhere.toml")
    rig.config("[studio]\nport = 8230\n")
    code, out, err = rig.call("studio-host", config=typed)
    assert (code, err) == (0, [])
    assert out == block(8235, FROM_CONFIG, sign_in.forward_line("studio-host", 8235), [SIGN_IN_BY_OPEN])
    assert out[0] == "Port: 8235 (from [studio] port)"


def test_port_defaults_without_a_config(rig):
    code, out, err = rig.call("studio-host")
    assert (code, err) == (0, [])
    assert out == block(8220, FROM_DEFAULT, sign_in.forward_line("studio-host", 8220), [SIGN_IN_BY_OPEN])
    assert out[0] == "Port: 8220 (default; pass --port if the Studio host sets another port in [studio])"


@pytest.mark.parametrize("kind", ["bad port", "not private", "invalid TOML", "missing typed file"])
def test_a_broken_config_prints_one_note_and_never_blocks(rig, kind):
    typed = None
    if kind == "bad port":
        rig.config("[studio]\nport = 80\n")
    elif kind == "not private":
        rig.config("[studio]\nport = 8230\n").chmod(0o606)
    elif kind == "invalid TOML":
        rig.config("[studio\nport = 8230\n")
    else:
        typed = rig.work / "missing.toml"
    code, out, err = rig.call("studio-host", config=typed)
    assert code == 0
    assert len(err) == 1 and err[0].startswith("Note: ")
    assert out == block(8220, FROM_DEFAULT, sign_in.forward_line("studio-host", 8220), [SIGN_IN_BY_OPEN])


def test_port_from_a_running_unit(rig):
    rig.service(port=8231)
    code, out, err = rig.call(environ=SESSION)
    assert (code, err) == (0, [])
    assert out == block(8231, FROM_STUDIO, sign_in.forward_line(SESSION_HOST, 8231), [SIGN_IN_BY_OPEN])


def test_port_from_a_terminal_companion_page(rig):
    url = rig.companion(port=8232)
    rig.config("[studio]\nport = 8230\n")
    code, out, err = rig.call(environ=SESSION)
    assert (code, err) == (0, [])
    assert out == block(8232, FROM_STUDIO, sign_in.forward_line(SESSION_HOST, 8232), [LINK_FOLLOWS, "  " + url])


def test_the_unit_names_the_state_directory(rig, monkeypatch):
    """S16: the unit's state line wins over this shell's XDG_STATE_HOME."""
    rig.service(port=8231)
    url = rig.companion(port=8231, lock=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(rig.tmp / "another-state"))
    code, out, err = rig.call(environ=SESSION)
    assert (code, err) == (0, [])
    assert out == block(8231, FROM_STUDIO, sign_in.forward_line(SESSION_HOST, 8231), [LINK_FOLLOWS, "  " + url])


def test_the_link_of_a_running_unit_needs_a_page_written_after_it_started(rig):
    rig.service(port=8231, started_ago=60.0)
    rig.companion(port=8231, lock=False, age=120.0)
    code, out, err = rig.call(environ=SESSION)
    assert (code, err) == (0, [])
    assert out == block(8231, FROM_STUDIO, sign_in.forward_line(SESSION_HOST, 8231), [SIGN_IN_BY_OPEN])
    assert FAKE_TOKEN not in "\n".join(out)
    url = rig.companion(port=8231, lock=False, age=10.0)
    code, out, err = rig.call(environ=SESSION)
    assert (code, err) == (0, [])
    assert out == block(8231, FROM_STUDIO, sign_in.forward_line(SESSION_HOST, 8231), [LINK_FOLLOWS, "  " + url])


def test_a_management_unit_signs_in_by_its_form(rig):
    rig.service(port=8231, management=True)
    rig.companion(port=8231, lock=False, page=False)
    code, out, err = rig.call(environ=SESSION)
    assert (code, err) == (0, [])
    assert out == block(8231, FROM_STUDIO, sign_in.forward_line(SESSION_HOST, 8231), [SIGN_IN_BY_FORM])
    assert FAKE_TOKEN not in "\n".join(out)


@pytest.mark.parametrize("case", ["not a terminal", "lock free", "stopped unit", "unit unknown", "no page",
                                  "page of another shape"])
def test_no_link_without_a_live_companion_on_a_terminal(rig, case):
    port, source = 8220, FROM_DEFAULT
    if case == "not a terminal":
        rig.companion(port=8232)
        port, source = 8232, FROM_STUDIO
    elif case == "lock free":
        rig.companion(port=8232, lock=False)
    elif case == "stopped unit":
        rig.service(port=8231, active="inactive")
        rig.companion(port=8231, lock=False)
    elif case == "unit unknown":
        rig.service(port=8231)
        rig.status = None
        rig.companion(port=8231, lock=False)
    elif case == "no page":
        rig.companion(port=8232, page=False)
    else:
        rig.companion(port=8232)
        (rig.state / sign_in.SIGN_IN_FILE).write_text("<p>another page</p>\n")
    code, out, err = rig.call(environ=SESSION, terminal=case != "not a terminal")
    assert (code, err) == (0, [])
    assert FAKE_TOKEN not in "\n".join(out)
    assert out == block(port, source, sign_in.forward_line(SESSION_HOST, port), [SIGN_IN_BY_OPEN])


def test_an_unsafe_state_directory_reads_as_no_live_studio(rig):
    rig.companion(port=8232)
    rig.state.chmod(0o755)
    try:
        code, out, err = rig.call(environ=SESSION)
    finally:
        rig.state.chmod(0o700)
    assert (code, err) == (0, [])
    assert out == block(8220, FROM_DEFAULT, sign_in.forward_line(SESSION_HOST, 8220), [SIGN_IN_BY_OPEN])


def test_a_typed_port_unlike_the_live_studio_drops_its_link(rig):
    rig.companion(port=8232)
    code, out, err = rig.call(port=8240, environ=SESSION)
    assert (code, err) == (0, [])
    assert out == block(8240, FROM_FLAG, sign_in.forward_line(SESSION_HOST, 8240), [SIGN_IN_BY_OPEN])


def test_a_typed_port_equal_to_the_live_studio_keeps_its_link(rig):
    url = rig.companion(port=8232)
    code, out, err = rig.call(port=8232, environ=SESSION)
    assert (code, err) == (0, [])
    assert out == block(8232, FROM_FLAG, sign_in.forward_line(SESSION_HOST, 8232), [LINK_FOLLOWS, "  " + url])


def test_a_companion_on_ipv6_loopback_gets_the_bracketed_target(rig):
    url = rig.companion(port=8233, host="::1")
    code, out, err = rig.call(environ=SESSION)
    assert (code, err) == (0, [])
    assert out[2] == "  ssh -o ExitOnForwardFailure=yes -L 8233:[::1]:8233 user@192.0.2.10"
    assert out[4] == "Then browse: http://[::1]:8233/"
    assert out == block(8233, FROM_STUDIO, sign_in.forward_line(SESSION_HOST, 8233, target="::1"),
                        [LINK_FOLLOWS, "  " + url], target="[::1]")


def test_run_execs_the_printed_argument_list_and_never_a_link(rig):
    rig.companion(port=8232)
    _code, printed, _err = rig.call(environ=SESSION)
    code, out, err = rig.call(run=True, environ=SESSION)
    assert (code, out) == (0, [])
    forward = sign_in.forward_line(SESSION_HOST, 8232)
    assert printed[2] == "  " + forward
    assert rig.execs == [("ssh", forward.split(" "))]
    assert " ".join(rig.execs[0][1]) == forward
    # Every line but the forward line, with the open sentence in place of the link.
    assert err == [line for line in block(8232, FROM_STUDIO, forward, [SIGN_IN_BY_OPEN]) if line != "  " + forward]
    assert FAKE_TOKEN not in "\n".join(err)


def test_run_with_a_named_host(rig):
    code, out, err = rig.call("user@studio-host", run=True)
    forward = sign_in.forward_line("user@studio-host", 8220)
    assert (code, out) == (0, [])
    assert rig.execs == [("ssh", ["ssh", "-o", "ExitOnForwardFailure=yes", "-L", "8220:127.0.0.1:8220",
                                  "user@studio-host"])]
    assert err == [line for line in block(8220, FROM_DEFAULT, forward, [SIGN_IN_BY_OPEN]) if line != "  " + forward]


def test_run_for_a_non_22_session_port_keeps_the_port_words(rig):
    code, _out, _err = rig.call(run=True, environ={"SSH_CONNECTION": "192.0.2.20 50022 192.0.2.10 2222"})
    assert code == 0
    assert rig.execs == [("ssh", ["ssh", "-p", "2222", "-o", "ExitOnForwardFailure=yes", "-L",
                                  "8220:127.0.0.1:8220", SESSION_HOST])]


def test_run_without_ssh_on_path(rig, monkeypatch):
    monkeypatch.setenv("PATH", str(rig.work))
    code, out, err = rig.call("studio-host", run=True)
    assert (code, out, err) == (1, [], [NO_SSH])
    assert rig.execs == []


@pytest.mark.parametrize("port", [0, 80, 1023, 65536])
def test_an_out_of_range_port_is_a_usage_error(rig, port):
    with pytest.raises(ValueError, match=r"^--port must be from 1024 to 65535$"):
        rig.call("studio-host", port=port)
    assert PORT_RANGE == "--port must be from 1024 to 65535"
    assert rig.execs == []


@pytest.mark.parametrize("port", [1024, 65535])
def test_the_port_range_is_inclusive(rig, port):
    code, out, err = rig.call("studio-host", port=port)
    assert (code, err) == (0, [])
    assert out[0] == PORT_LINE.format(port=port, source=FROM_FLAG)


def test_no_unit_means_no_user_manager_call(rig):
    rig.companion(port=8232)
    code, _out, _err = rig.call(environ=SESSION)
    assert code == 0
    assert rig.status_calls == 0
