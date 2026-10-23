"""dgxm studio: the parser, the one word, open and token, end to end through the CLI.

A fake user manager stands in for systemctl, journalctl, loginctl, xdg-mime,
xdg-open and the service's --check run, unit_state's clocks are replaced and
the display variables are cleared, so no test reads the machine's display,
clocks or user manager. Every token is fake.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import os
import socket
import sqlite3
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from dgx_monarch import runtime_provenance
from dgx_monarch.cli import arguments, comfy_ports, probe_certainty, studio_command
from dgx_monarch.cli import main as cli_main
from dgx_monarch.studio import launch, sign_in, unit, unit_paths, unit_state

TOKEN = "A" * 21 + "_" + "b" * 21  # 43 characters, the shape of secrets.token_urlsafe(32)
NEW_TOKEN = "C" * 21 + "-" + "d" * 21
DIGEST = "ab" * 32
EPOCH = 1_800_000_000.0  # the fake wall clock minus the fake monotonic clock
DRIVER = "http://127.0.0.1:8191"
DEFAULT = "http://127.0.0.1:8188"
DRIVER_UNIT = "comfyui-driver.service"
HOSTS = '[cluster]\nclient_bind = "tcp://10.0.0.1:0"\n\n[[hosts]]\nname = "n1"\naddress = "tcp://10.0.0.1:26600"\n'
PYTHON = "/home/user/venv/bin/python"
DIRECT = ("{{ path={python} ; argv[]={python} /home/user/ComfyUI/main.py --listen 127.0.0.1 --port {port} ; "
          "ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }}")
WRAPPER = ("{ path=/usr/bin/bash ; argv[]=/usr/bin/bash /home/user/start-comfyui.sh ; ignore_errors=no ; "
           "start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }")
REAL_UNIT_RUNTIME = unit_paths.unit_runtime


@dataclass
class Result:
    code: int
    out: str
    err: str

    @property
    def text(self) -> str:
        return self.out + self.err


class _Opened:
    def wait(self, timeout: float | None = None) -> int:
        return 0


class Rig:
    """One home directory, a fake user manager and a fake companion."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
        self.tmp, self.monkeypatch, self.capsys = tmp_path, monkeypatch, capsys
        self.home = tmp_path / "home"
        self.home.mkdir(mode=0o700)
        self.cwd = tmp_path / "cwd"
        self.cwd.mkdir()
        runtime_dir = tmp_path / "run"
        runtime_dir.mkdir(mode=0o700)
        for name in ("XDG_STATE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "DGXM_CLUSTER_TOML", "DISPLAY",
                     "WAYLAND_DISPLAY", "SSH_CONNECTION", "TMUX", "STY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv("HOME", str(self.home))
        monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
        monkeypatch.chdir(self.cwd)
        self.state = self.home / ".local" / "state" / "dgx-monarch" / "studio"
        self.unit_path = self.home / ".config" / "systemd" / "user" / unit.UNIT_NAME
        self.load, self.active, self.pid, self.start_us = "not-found", "inactive", 0, 0
        self.answers = True        # False: systemctl show times out
        self.start = "ok"          # what a restart does: ok, fail or hang
        self.stops = True          # False: disable --now exits 0 but the unit stays active
        self.driver: dict[str, str] | None = None
        self.driver_starts = True  # Wants= starts the driver unit with the Studio unit
        self.linger = "yes"
        self.handlers = ("firefox.desktop", "firefox.desktop")
        self.check = (0, "dgx-monarch Studio: 3 packaged resources match their manifest; studio extra installed")
        self.journal = ["Another process already listens on the companion port.", "exit status 1"]
        self.busy: list[str] | None = []
        self.port_busy = False
        self.terminal = True
        self.now = 1000.0
        self.later: list[tuple[float, Callable[[], None]]] = []
        self.calls: list[list[str]] = []
        self.fails: dict[str, tuple[int, str]] = {}  # systemctl verb: its exit code and stderr
        self.opened: list[list[str]] = []
        self.started: list[list[str]] = []
        self.locks: list[int] = []
        self.runtime = unit_paths.UnitRuntime(interpreter=Path(PYTHON), source=Path("/home/user/src"),
                                              dependencies=Path("/home/user/venv/lib/python3.12/site-packages"))
        self.digest = DIGEST
        monkeypatch.setattr(studio_command, "_run", self.run)
        monkeypatch.setattr(studio_command, "_popen", self.popen)
        monkeypatch.setattr(sign_in, "terminal_stdout", lambda: self.terminal)
        monkeypatch.setattr(unit_state, "_monotonic", lambda: self.now)
        monkeypatch.setattr(unit_state, "_realtime", lambda: self.now + EPOCH)
        monkeypatch.setattr(unit_state, "_sleep", self.sleep)
        monkeypatch.setattr(unit_state, "busy_processes", lambda status: self.busy)
        monkeypatch.setattr(unit_state, "port_in_use", lambda port: self.port_busy)
        monkeypatch.setattr(unit_state, "user_manager", lambda **_kw: "ok")
        monkeypatch.setattr(unit_paths, "unit_runtime", lambda environ=None: self.runtime)
        monkeypatch.setattr(runtime_provenance, "dgx_source_manifest_sha256", lambda root=None: self.digest)
        opened, plain = sign_in.open_in_browser, sign_in.open_plain_address
        monkeypatch.setattr(sign_in, "open_in_browser",
                            lambda path, **kw: opened(path, which=lambda name: f"/usr/bin/{name}", **kw))
        monkeypatch.setattr(sign_in, "open_plain_address",
                            lambda url, **kw: plain(url, which=lambda name: f"/usr/bin/{name}", **kw))
        monkeypatch.setattr("dgx_monarch.studio.__main__.main", self.companion)

    # The CLI.

    def cli(self, *argv: str) -> Result:
        self.capsys.readouterr()
        try:
            code: object = cli_main.main(list(argv))
        except SystemExit as exc:
            code = exc.code
        out, err = self.capsys.readouterr()
        if isinstance(code, str):
            return Result(1, out, err + code + "\n")
        return Result(code if isinstance(code, int) else 0, out, err)

    def companion(self, argv: list[str] | None = None) -> int:
        self.started.append(list(argv or []))
        return 0

    # Files.

    def config(self, studio: str = f'driver = "{DRIVER}"', *, hosts: bool = True, where: Path | None = None) -> Path:
        path = where or self.tmp / "cluster.toml"
        path.write_text((HOSTS if hosts else "") + (f"\n[studio]\n{studio}\n" if studio is not None else ""))
        path.chmod(0o600)
        return path

    def private(self, path: Path, text: str, at: float | None = None) -> None:
        unit_paths.make_private_directories(path.parent)
        descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            os.write(descriptor, text.encode())
        finally:
            os.close(descriptor)
        stamp = int(((self.now if at is None else at) + EPOCH) * 1e9)
        os.utime(path, ns=(stamp, stamp))

    def serve(self, token: str = TOKEN, *, state: Path | None = None, port: int = 8220, management: bool = False,
              at: float | None = None) -> None:
        """What a companion writes once it holds its port: the token file and, without management, the page."""
        state = state or self.state
        unit_paths.make_private_directories(state)
        os.close(os.open(state / "companion.lock", os.O_CREAT | os.O_RDWR, 0o600))  # the companion takes it first
        self.private(state / sign_in.OWNER_FILE, token + "\n", at)
        if not management:
            self.private(state / sign_in.SIGN_IN_FILE, sign_in.sign_in_page(sign_in.sign_in_link("127.0.0.1", port, token)), at)

    def bind(self, origin: str, state: Path | None = None) -> None:
        state = state or self.state
        unit_paths.make_private_directories(state)
        with sqlite3.connect(state / "studio.sqlite3") as db:
            db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            db.execute("INSERT INTO meta VALUES ('driver_origin', ?)", (origin,))
        db.close()

    def settings(self, config: Path | None = None, **changes: object) -> launch.StudioSettings:
        """What up resolves: [studio] driver from a config, else the default origin."""
        values: dict[str, object] = {
            "port": 8220, "driver": DRIVER if config else DEFAULT,
            "driver_source": "[studio] driver" if config else "default", "state_dir": self.state,
            "config": config, "config_source": "--config" if config else None, "driver_service": None,
            "allow_management": False}
        values.update(changes)
        return launch.StudioSettings(**values)  # type: ignore[arg-type]

    def install(self, settings: launch.StudioSettings | None = None, *, active: str = "active",
                started: float | None = None, **fields: str) -> None:
        """An owned unit file, as up wrote it, and the user manager's view of it."""
        text = unit.render_unit(settings or self.settings(), self.runtime, source_digest=self.digest)
        lines = []
        for line in text.splitlines():
            name = line[len(unit.SETTING_PREFIX):].partition("=")[0] if line.startswith(unit.SETTING_PREFIX) else ""
            lines.append(f"{unit.SETTING_PREFIX}{name}={fields[name]}" if name in fields else line)
        text = "\n".join(lines) + "\n"
        unit_paths.make_private_directories(self.unit_path.parent)
        self.unit_path.write_text(text)
        self.unit_path.chmod(0o600)
        self.load, self.active = "loaded", active
        if active in unit_state.ACTIVE_STATES:
            self.pid, self.start_us = 4242, int((self.now if started is None else started) * 1e6)

    def hold_lock(self, state: Path | None = None) -> None:
        """A companion in another terminal: its state lock, held until release()."""
        state = state or self.state
        unit_paths.make_private_directories(state)
        descriptor = os.open(state / "companion.lock", os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.locks.append(descriptor)

    def release(self) -> None:
        while self.locks:
            os.close(self.locks.pop())

    # The fake user manager.

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        due = [entry for entry in self.later if entry[0] <= self.now]
        self.later = [entry for entry in self.later if entry[0] > self.now]
        for _, action in due:
            action()

    def popen(self, argv: list[str], **_options: object) -> _Opened:
        self.opened.append(list(argv))
        return _Opened()

    def _studio_properties(self) -> dict[str, str]:
        running = self.active in unit_state.ACTIVE_STATES
        return {"LoadState": self.load, "ActiveState": self.active, "SubState": "running" if running else "dead",
                "MainPID": str(self.pid if running else 0),
                "ControlGroup": "/user.slice/app.slice/dgxm-studio.service",
                "FragmentPath": str(self.unit_path) if self.load == "loaded" else "",
                "ExecMainStartTimestampMonotonic": str(self.start_us), "DropInPaths": "", "NeedDaemonReload": "no"}

    def _restart(self) -> None:
        if self.start == "fail":
            self.active, self.pid = "failed", 0
            return
        self.now += 0.5
        self.active, self.pid, self.start_us = "active", 4243, int(self.now * 1e6)
        installed = unit.unit_settings(self.unit_path.read_text())
        if self.driver is not None and self.driver_starts and installed.get("driver-service"):
            self.driver["ActiveState"] = "active"
        if self.start == "ok":
            self.serve(NEW_TOKEN, state=Path(installed["state"]), port=int(installed["port"]),
                       management=installed["management"] == "yes")

    def run(self, argv: list[str], **_options: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        out, code = "", 0
        if argv[:2] == ["systemctl", "--user"]:
            verb = argv[2]
            if verb in self.fails:
                return subprocess.CompletedProcess(argv, self.fails[verb][0], "", self.fails[verb][1])
            if verb == "show":
                if not self.answers:
                    raise subprocess.TimeoutExpired(argv, 10)
                names = argv[argv.index("--no-pager") + 1].split("=", 1)[1].split(",")
                target = argv[-1]
                values = (self._studio_properties() if target == unit.UNIT_NAME else self.driver
                          if self.driver is not None and target == DRIVER_UNIT
                          else {"LoadState": "not-found", "ActiveState": "inactive"})
                # systemd 255 prints no ExecStart= at all for a unit it could not load.
                shown = [name for name in names if name != "ExecStart" or values.get("LoadState") == "loaded"]
                out = "".join(f"{name}={values.get(name, '')}\n" for name in shown)
            elif verb == "daemon-reload":
                self.load = "loaded" if unit.owned_unit_state(self.unit_path) == "ours" else "not-found"
            elif verb == "reset-failed" and self.active == "failed":
                self.active = "inactive"
            elif verb == "restart":
                self._restart()
            elif verb == "disable" and self.stops:
                self.active, self.pid = "inactive", 0
        elif argv[0] == "journalctl":
            out = "".join(line + "\n" for line in self.journal)
        elif argv[0] == "loginctl":
            out = self.linger + "\n"
        elif argv[0] == "xdg-mime":
            out = self.handlers[0 if argv[-1] == "text/html" else 1] + "\n"
        elif argv[0] == PYTHON and argv[-1] == "--check":
            code, out = self.check
        else:
            raise AssertionError(f"unexpected command {argv}")
        return subprocess.CompletedProcess(argv, code, out, "")

    def systemctl(self, verb: str) -> list[list[str]]:
        return [call for call in self.calls if call[:3] == ["systemctl", "--user", verb]]


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> Iterator[Rig]:
    made = Rig(tmp_path, monkeypatch, capsys)
    yield made
    made.release()


def _children(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    return next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices


def _studio_parsers() -> dict[str, argparse.ArgumentParser]:
    studio = _children(arguments.build_parser(defaultdict(lambda: lambda _args: 0)))["studio"]
    return {"studio": studio, **_children(studio)}


def _options(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {(action.option_strings[-1] if action.option_strings else action.dest): action
            for action in parser._actions if not isinstance(action, argparse._SubParsersAction)}


def test_studio_parser_shape():
    parsers = _studio_parsers()
    assert set(parsers) == {"studio", "up", "down", "open", "token", "tunnel"}
    shapes = {name: {option: (action.dest, action.default) for option, action in _options(parser).items()
                     if option != "--help"} for name, parser in parsers.items()}
    assert shapes == {
        "studio": {"--config": ("config", argparse.SUPPRESS), "--driver": ("driver", None),
                   "--state-dir": ("state_dir", None), "--allow-management": ("allow_management", False),
                   "--no-open": ("no_open", False)},
        "up": {"--config": ("config", argparse.SUPPRESS), "--allow-management": ("up_allow_management", False)},
        "down": {}, "open": {},
        "token": {"--value": ("value", False), "--rotate": ("rotate", False)},
        "tunnel": {"--config": ("config", argparse.SUPPRESS), "host": ("host", None), "--port": ("port", None),
                   "--run": ("run", False)},
    }
    assert _options(parsers["tunnel"])["host"].nargs == "?"
    assert _options(parsers["tunnel"])["--port"].type is int
    parser = arguments.build_parser(defaultdict(lambda: lambda _args: 0))
    assert parser.parse_args(["studio"]).studio_action is None
    with pytest.raises(SystemExit) as refused:
        parser.parse_args(["studio", "token", "--value", "--rotate"])
    assert refused.value.code == 2
    # The global --config stays in force unless one is typed after the word it belongs to.
    assert parser.parse_args(["--config", "X", "studio"]).config == "X"
    assert parser.parse_args(["--config", "X", "studio", "--config", "Y"]).config == "Y"
    assert parser.parse_args(["--config", "X", "studio", "up"]).config == "X"
    assert parser.parse_args(["studio", "up", "--config", "Y"]).config == "Y"
    assert parser.parse_args(["studio", "tunnel"]).host is None


@pytest.mark.parametrize("action", ["up", "down", "open", "token", "tunnel"])
@pytest.mark.parametrize("flag", [["--driver", DRIVER], ["--state-dir", "/home/user/state"],
                                  ["--allow-management"], ["--no-open"]])
def test_foreground_options_are_refused_with_an_action(rig: Rig, action: str, flag: list[str]):
    result = rig.cli("studio", *flag, action)
    assert result.code == 2
    assert result.err.strip() == f"dgxm studio: error: {studio_command.FOREGROUND_ONLY}"
    assert rig.calls == [] and not rig.unit_path.parent.exists()


def _foreground_cases(rig: Rig) -> list[tuple[list[str], list[str]]]:
    plain = rig.config(None)
    full = rig.config(f'driver = "{DRIVER}"\nport = 8230\ndriver_service = "{DRIVER_UNIT}"',
                      where=rig.tmp / "full.toml")
    state = ["--state-dir", str(rig.state)]
    return [
        (["--driver", DRIVER], ["--host", "127.0.0.1", "--port", "8220", "--driver", DRIVER, *state]),
        (["--config", str(plain)],
         ["--host", "127.0.0.1", "--port", "8220", "--driver", DEFAULT, *state, "--config", str(plain)]),
        (["--config", str(full)], ["--host", "127.0.0.1", "--port", "8230", "--driver", DRIVER, *state,
                                   "--config", str(full), "--driver-service", DRIVER_UNIT]),
        (["--config", str(full), "--driver", "http://127.0.0.1:8192"],
         ["--host", "127.0.0.1", "--port", "8230", "--driver", "http://127.0.0.1:8192", *state,
          "--config", str(full), "--driver-service", DRIVER_UNIT]),
        (["--driver", DRIVER, "--state-dir", str(rig.tmp / "elsewhere")],
         ["--host", "127.0.0.1", "--port", "8220", "--driver", DRIVER, "--state-dir", str(rig.tmp / "elsewhere")]),
        (["--driver", DRIVER, "--allow-management"],
         ["--host", "127.0.0.1", "--port", "8220", "--driver", DRIVER, *state, "--allow-management"]),
        (["--driver", DRIVER, "--no-open"], ["--host", "127.0.0.1", "--port", "8220", "--driver", DRIVER, *state,
                                             "--no-open"]),
    ]


def test_foreground_argv_is_exact(rig: Rig):
    rig.bind(DEFAULT)
    cases = _foreground_cases(rig)
    for typed, _expected in cases:
        result = rig.cli("studio", *typed)
        assert result.code == 0, result.text
    assert rig.started == [expected for _typed, expected in cases]
    assert rig.calls == []  # no unit installed, so no user manager call and no browser


def test_settings_never_probe_for_a_driver(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dgxm studio probed for a driver")

    for name, value in vars(comfy_ports).items():
        if callable(value) and getattr(value, "__module__", None) == comfy_ports.__name__:
            monkeypatch.setattr(comfy_ports, name, refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    config = rig.config()
    assert rig.cli("studio", "--config", str(config)).code == 0
    assert rig.started[-1][:6] == ["--host", "127.0.0.1", "--port", "8220", "--driver", DRIVER]


def test_the_source_lines_name_config_and_driver_origin(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    explicit = rig.config()
    assert rig.cli("studio", "--config", str(explicit)).out.splitlines()[:2] == [
        f"Config: {explicit}", f"Driver: {DRIVER} (from [studio] driver)"]
    monkeypatch.setenv("DGXM_CLUSTER_TOML", str(explicit))
    assert rig.cli("studio", "--driver", "http://127.0.0.1:8192").out.splitlines()[:2] == [
        f"Config: {explicit}", "Driver: http://127.0.0.1:8192 (from --driver)"]
    monkeypatch.delenv("DGXM_CLUSTER_TOML")
    local = rig.config(where=rig.cwd / "cluster.toml")
    assert rig.cli("studio").out.splitlines()[0] == f"Config: {local} (from the working directory)"
    local.unlink()
    (rig.home / ".config" / "dgx-monarch").mkdir(parents=True)
    user = rig.config(where=rig.home / ".config" / "dgx-monarch" / "cluster.toml")
    assert rig.cli("studio").out.splitlines()[0] == f"Config: {user}"
    user.unlink()
    rig.bind(DEFAULT)
    assert rig.cli("studio").out.splitlines()[:2] == [
        studio_command.NO_CONFIG, f"Driver: {DEFAULT} (default; this state directory is already bound to it)"]


def test_one_word_starts_with_a_local_mode_file(rig: Rig):
    config = rig.config(hosts=False)
    result = rig.cli("studio", "--config", str(config))
    assert result.code == 0
    assert result.out.splitlines()[:3] == [f"Config: {config}", launch.NO_HOSTS_NOTE,
                                           f"Driver: {DRIVER} (from [studio] driver)"]
    assert rig.started[-1][-2:] == ["--config", str(config)]


def test_one_word_refuses_a_fresh_state_directory_without_a_driver_origin(rig: Rig):
    result = rig.cli("studio")
    assert (result.code, result.err.strip()) == (1, launch.NO_DRIVER_ORIGIN)
    assert rig.started == [] and not rig.state.exists()
    rig.bind(DEFAULT)
    with sqlite3.connect(rig.state / "studio.sqlite3") as db:
        db.execute("DELETE FROM meta")
    db.close()
    assert rig.cli("studio").err.strip() == launch.NO_DRIVER_ORIGIN
    assert rig.started == []


def test_one_word_refuses_the_default_origin_when_the_binding_cannot_be_read(rig: Rig):
    # A first start that stopped inside the store leaves a zero-byte database.
    unit_paths.make_private_directories(rig.state)
    (rig.state / "studio.sqlite3").write_bytes(b"")
    result = rig.cli("studio")
    assert (result.code, result.err.strip()) == (1, launch.NO_DRIVER_ORIGIN)
    assert rig.started == []
    assert rig.cli("studio", "--driver", DRIVER).code == 0
    assert rig.started[-1][:6] == ["--host", "127.0.0.1", "--port", "8220", "--driver", DRIVER]


def test_one_word_starts_on_a_directory_bound_to_the_default(rig: Rig):
    rig.bind(DEFAULT)
    result = rig.cli("studio")
    assert result.code == 0
    assert rig.started == [["--host", "127.0.0.1", "--port", "8220", "--driver", DEFAULT,
                            "--state-dir", str(rig.state)]]


@pytest.mark.parametrize("active", ["active", "activating", "reloading", "deactivating"])
def test_one_word_acts_as_open_while_the_unit_runs(rig: Rig, active: str):
    config = rig.config()
    rig.install(rig.settings(config), active=active)
    rig.serve(at=rig.now + 0.1)
    result = rig.cli("studio", "--config", str(config))
    assert result.code == 0, result.text
    assert "Studio: http://127.0.0.1:8220" in result.out
    assert sign_in.link_lines(sign_in.sign_in_link("127.0.0.1", 8220, TOKEN)) == result.out.splitlines()[-3:-1]
    assert result.out.splitlines()[-1] == sign_in.NO_DISPLAY
    assert rig.started == []
    broken = rig.config('driver = "http://127.0.0.1:8191"\nlisten = 1')
    result = rig.cli("--config", str(broken), "studio")
    assert result.code == 0, result.text
    notes = [line for line in result.out.splitlines() if line.startswith("Note: ")]
    assert len(notes) == 1 and "unsupported key" in notes[0]
    assert result.out.splitlines()[0] == notes[0] and rig.started == []


def test_one_word_exits_2_on_a_broken_config_when_no_service_runs(rig: Rig):
    broken = rig.config("port = 80")
    result = rig.cli("studio", "--config", str(broken))
    assert result.code == 2
    assert result.err.startswith(f"error: {broken}: studio.port must be")
    assert rig.started == []


@pytest.mark.parametrize("flag", [["--driver", DRIVER], ["--state-dir", "/home/user/state"], ["--allow-management"]])
def test_one_word_refuses_launch_flags_while_the_unit_runs(rig: Rig, flag: list[str]):
    rig.install()
    rig.serve(at=rig.now + 0.1)
    result = rig.cli("studio", *flag)
    assert (result.code, result.err.strip()) == (1, launch.UNIT_RUNS)
    assert rig.started == []
    assert rig.cli("studio", "--no-open").code == 0


def test_one_word_prints_the_journal_then_starts_when_the_unit_failed(rig: Rig):
    rig.install(active="failed")
    rig.bind(DRIVER)
    result = rig.cli("studio", "--config", str(rig.config()))
    assert result.code == 0
    assert result.err.splitlines() == [unit.START_FAILED, *rig.journal]
    assert studio_command.INSTALLED_FAILED in result.out.splitlines()
    assert len(rig.started) == 1


def test_one_word_starts_in_the_terminal_when_the_unit_is_inactive_or_unknown(rig: Rig):
    config = rig.config()
    rig.install(active="inactive")
    result = rig.cli("studio", "--config", str(config))
    assert result.code == 0 and studio_command.INSTALLED_INACTIVE in result.out.splitlines()
    rig.answers = False
    result = rig.cli("studio", "--config", str(config))
    assert result.code == 0 and "installed but" not in result.out
    assert len(rig.started) == 2


def test_one_word_with_a_terminal_studio_running(rig: Rig):
    config = rig.config()
    rig.hold_lock()
    result = rig.cli("studio", "--config", str(config))
    assert (result.code, result.err.strip()) == (1, launch.TERMINAL_STUDIO)
    rig.serve()
    result = rig.cli("studio", "--config", str(config))
    assert result.code == 0
    assert "Studio: http://127.0.0.1:8220" in result.out and sign_in.NO_DISPLAY in result.out
    assert rig.started == []


def test_open_waits_for_a_page_newer_than_the_main_process_start(rig: Rig):
    rig.install()
    rig.later.append((rig.now + 0.5, lambda: rig.serve(at=rig.now)))
    result = rig.cli("studio", "open")
    assert result.code == 0, result.text
    assert sign_in.sign_in_link("127.0.0.1", 8220, TOKEN) in result.out
    assert 1000.5 <= rig.now < 1001


def test_open_ignores_a_leftover_page_older_than_the_start(rig: Rig):
    rig.serve(TOKEN, at=rig.now - 10)
    rig.install()
    rig.later.append((rig.now + 1, lambda: rig.serve(NEW_TOKEN, at=rig.now)))
    result = rig.cli("studio", "open")
    assert result.code == 0
    assert NEW_TOKEN in result.out and TOKEN not in result.out


def test_open_reports_unit_not_ready_at_the_deadline(rig: Rig):
    rig.install()
    result = rig.cli("studio", "open")
    assert (result.code, result.err.strip()) == (probe_certainty.UNKNOWN_EXIT, unit.UNIT_NOT_READY)
    assert rig.now >= 1030


def test_open_reads_the_state_directory_the_unit_names(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    elsewhere = rig.tmp / "service-state"
    rig.install(rig.settings(state_dir=elsewhere))
    rig.serve(state=elsewhere, at=rig.now + 0.1)
    monkeypatch.setenv("XDG_STATE_HOME", str(rig.tmp / "shell-state"))
    rig.terminal = False
    result = rig.cli("studio", "open")
    assert result.code == 0
    assert f"Sign-in page (mode 0600): {elsewhere / sign_in.SIGN_IN_FILE}" in result.out


def test_open_for_a_management_unit_prints_the_token_and_never_a_link(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    rig.install(rig.settings(allow_management=True))
    rig.serve(management=True, at=rig.now + 0.1)
    monkeypatch.setenv("DISPLAY", ":0")
    result = rig.cli("studio", "open")
    assert result.code == 0
    assert result.out.splitlines()[:5] == [
        "Studio: http://127.0.0.1:8220", f"Owner token file (mode 0600): {rig.state / sign_in.OWNER_FILE}",
        f"Owner token: {TOKEN}", sign_in.MANAGEMENT_NO_LINK, sign_in.OPENED_PLAIN]
    assert "#token=" not in result.text
    assert rig.opened == [["/usr/bin/xdg-open", "http://127.0.0.1:8220/"]]


def test_open_prints_the_link_only_to_a_terminal(rig: Rig):
    rig.install()
    rig.serve(at=rig.now + 0.1)
    rig.terminal = False
    result = rig.cli("studio", "open")
    assert result.out.splitlines()[:2] == ["Studio: http://127.0.0.1:8220",
                                           f"Sign-in page (mode 0600): {rig.state / sign_in.SIGN_IN_FILE}"]
    assert TOKEN not in result.text and "#token=" not in result.text
    rig.terminal = True
    result = rig.cli("studio", "open")
    assert result.out.splitlines()[1:4] == [*sign_in.link_lines(sign_in.sign_in_link("127.0.0.1", 8220, TOKEN)),
                                            sign_in.NO_DISPLAY]
    assert rig.opened == []


def test_open_sends_the_page_only_on_the_local_route_with_matching_handlers(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    rig.install()
    rig.serve(at=rig.now + 0.1)
    page = rig.state / sign_in.SIGN_IN_FILE
    monkeypatch.setenv("DISPLAY", ":0")
    result = rig.cli("studio", "open")
    assert result.out.splitlines()[1] == sign_in.OPENED and TOKEN not in result.text
    assert rig.opened == [["/usr/bin/xdg-open", page.as_uri()]]
    rig.handlers = ("chat.desktop", "firefox.desktop")
    result = rig.cli("studio", "open")
    link = sign_in.sign_in_link("127.0.0.1", 8220, TOKEN)
    assert result.out.splitlines()[1:5] == [*sign_in.link_lines(link), sign_in.NOT_A_BROWSER,
                                            sign_in.HANDLERS_DIFFER.format(html="chat.desktop", http="firefox.desktop")]
    monkeypatch.setenv("SSH_CONNECTION", "192.0.2.20 50000 192.0.2.10 22")
    result = rig.cli("studio", "open")
    assert result.out.splitlines()[1:4] == [*sign_in.link_lines(link), sign_in.SSH_SESSION]
    assert sign_in.THEN_OPEN_THERE in result.out
    monkeypatch.setenv("TMUX", "tmux-session")
    monkeypatch.delenv("SSH_CONNECTION")
    assert sign_in.NO_DISPLAY in rig.cli("studio", "open").out
    assert len(rig.opened) == 1


def test_open_refuses_a_stale_page_when_the_lock_is_free(rig: Rig):
    rig.serve()
    result = rig.cli("studio", "open")
    assert (result.code, result.err.strip()) == (1, sign_in.NOT_RUNNING)
    (rig.state / "companion.lock").unlink()
    assert rig.cli("studio", "open").err.strip() == sign_in.NOT_RUNNING
    rig.hold_lock()
    (rig.state / sign_in.SIGN_IN_FILE).unlink()
    assert rig.cli("studio", "open").err.strip() == sign_in.NO_SIGN_IN_PAGE


def test_open_reports_a_failed_unit_with_its_journal(rig: Rig):
    rig.install(active="failed")
    result = rig.cli("studio", "open")
    assert result.code == 1
    assert result.err.splitlines() == [unit.START_FAILED, *rig.journal]


def test_open_drift_note_names_the_fields_that_differ(rig: Rig):
    config = rig.config()
    rig.install(rig.settings(config), driver="http://127.0.0.1:8188", interpreter="/home/user/old/bin/python")
    rig.serve(at=rig.now + 0.1)
    result = rig.cli("--config", str(config), "studio", "open")
    assert result.code == 0
    assert result.out.splitlines()[-1] == f"{unit.DRIFT_NOTE} driver, interpreter"
    rig.install(rig.settings(config))
    rig.serve(at=rig.now + 0.1)
    assert unit.DRIFT_NOTE not in rig.cli("--config", str(config), "studio", "open").out


def test_open_stale_code_note_after_the_source_changes(rig: Rig):
    config = rig.config()
    rig.install(rig.settings(config))
    rig.serve(at=rig.now + 0.1)
    assert unit.STALE_CODE not in rig.cli("--config", str(config), "studio", "open").out
    rig.digest = "cd" * 32
    result = rig.cli("--config", str(config), "studio", "open")
    assert result.out.splitlines()[-1] == unit.STALE_CODE


def test_token_prints_the_path_and_value_only_on_a_terminal(rig: Rig):
    rig.install()
    rig.serve(at=rig.now + 0.1)
    result = rig.cli("studio", "token")
    assert result.out.splitlines() == [f"Owner token file (mode 0600): {rig.state / sign_in.OWNER_FILE}"]
    result = rig.cli("studio", "token", "--value")
    assert result.out.splitlines()[1] == f"Owner token: {TOKEN}"
    rig.terminal = False
    assert TOKEN not in rig.cli("studio", "token").text


def test_token_value_without_a_terminal_refuses(rig: Rig):
    rig.install()
    rig.serve(at=rig.now + 0.1)
    rig.terminal = False
    result = rig.cli("studio", "token", "--value")
    assert (result.code, result.err.strip()) == (1, sign_in.NO_PIPE)
    assert TOKEN not in result.text and rig.calls == []


def test_rotate_needs_the_service_and_waits_for_a_different_token(rig: Rig):
    result = rig.cli("studio", "token", "--rotate")
    assert (result.code, result.err.strip()) == (1, unit.NO_ROTATE)
    rig.install()
    rig.serve(at=rig.now + 0.1)
    result = rig.cli("studio", "token", "--rotate")
    assert result.code == 0, result.text
    verbs = [call[2] for call in rig.calls if call[:2] == ["systemctl", "--user"] and call[2] != "show"]
    assert verbs == ["reset-failed", "restart"]
    assert result.out.splitlines()[0] == studio_command.ROTATED
    assert sign_in.sign_in_link("127.0.0.1", 8220, NEW_TOKEN) in result.out
    assert TOKEN not in result.out and rig.opened == []


def test_rotate_prints_also_started_for_a_stopped_driver_unit(rig: Rig):
    rig.driver = {"LoadState": "loaded", "ActiveState": "inactive", "ExecStart": DIRECT.format(python=PYTHON, port=8191)}
    rig.install(rig.settings(driver_service=DRIVER_UNIT))
    rig.serve(at=rig.now + 0.1)
    result = rig.cli("studio", "token", "--rotate")
    assert result.out.splitlines()[:2] == [studio_command.ROTATED,
                                           f"Also started {DRIVER_UNIT}: the Studio service wants it."]
    rig.serve(at=rig.now + 0.1)
    assert "Also started" not in rig.cli("studio", "token", "--rotate").out


def test_rotate_reads_the_unit_again_under_the_lock(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    rig.install()
    rig.serve(at=rig.now + 0.1)
    real_lock = unit_state.unit_lock

    @contextlib.contextmanager
    def lock_after_a_down() -> Iterator[None]:
        # A down that finished while this rotate waited for the lock.
        rig.unit_path.unlink()
        rig.load, rig.active, rig.pid = "not-found", "inactive", 0
        with real_lock():
            yield

    monkeypatch.setattr(unit_state, "unit_lock", lock_after_a_down)
    result = rig.cli("studio", "token", "--rotate")
    assert (result.code, result.err.strip()) == (1, unit.NO_ROTATE)
    assert rig.systemctl("restart") == [] and rig.systemctl("reset-failed") == []


def test_rotate_reports_a_restart_systemd_refused(rig: Rig):
    rig.install()
    rig.serve(at=rig.now + 0.1)
    rig.fails["restart"] = (1, "Job for dgxm-studio.service failed.\nSee the journal for details.")
    result = rig.cli("studio", "token", "--rotate")
    assert result.code == 1
    assert result.err.splitlines() == [unit.START_FAILED, "See the journal for details.", *rig.journal]


def test_rotate_times_out_as_unknown(rig: Rig):
    rig.install()
    rig.serve(at=rig.now + 0.1)
    rig.start = "hang"
    result = rig.cli("studio", "token", "--rotate")
    assert (result.code, result.err.strip()) == (probe_certainty.UNKNOWN_EXIT, unit.NOT_READY)
    rig.answers = False
    result = rig.cli("studio", "token", "--rotate")
    assert (result.code, result.err.strip()) == (probe_certainty.UNKNOWN_EXIT, studio_command.LOST_ANSWER)


@pytest.mark.parametrize("argv", [[], ["--no-open"], ["open"], ["token"], ["token", "--rotate"], ["up"],
                                  ["tunnel"], ["down"]])
def test_no_studio_command_output_holds_the_token_without_a_terminal(rig: Rig, argv: list[str]):
    config = rig.config()
    rig.install(rig.settings(config))
    rig.serve(at=rig.now + 0.1)
    rig.terminal = False
    # Inside an SSH session tunnel derives the host and would print the link on a terminal.
    rig.monkeypatch.setenv("SSH_CONNECTION", "192.0.2.20 50000 192.0.2.10 22")
    result = rig.cli("--config", str(config), "studio", *argv)
    assert result.code == 0, result.text
    for secret in (TOKEN, NEW_TOKEN, "#token="):
        assert secret not in result.text


def test_tunnel_gets_the_typed_values_and_a_bad_port_is_a_usage_error(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    seen: list[tuple[object, ...]] = []

    def fake(host, port, run, config, **options):
        seen.append((host, port, run, config, options["terminal"]))
        if port is not None and not 1024 <= port <= 65535:
            raise ValueError("--port must be from 1024 to 65535")
        return 0

    monkeypatch.setattr(studio_command.tunnel, "run_tunnel", fake)
    assert rig.cli("--config", "/home/user/cluster.toml", "studio", "tunnel", "studio-host", "--run").code == 0
    assert seen[-1] == ("studio-host", None, True, Path("/home/user/cluster.toml"), True)
    assert rig.cli("studio", "tunnel").code == 0 and seen[-1] == (None, None, False, None, True)
    result = rig.cli("studio", "tunnel", "studio-host", "--port", "80")
    assert (result.code, result.err.strip()) == (2, "dgxm studio tunnel: error: --port must be from 1024 to 65535")
    assert rig.calls == []


def test_one_word_usage_errors_exit_2(rig: Rig):
    result = rig.cli("studio", "--state-dir", "relative/state")
    assert (result.code, result.err.strip()) == (2, f"dgxm studio: error: {launch.STATE_DIR_RELATIVE}")
    result = rig.cli("studio", "--driver", "http://0.0.0.0:8191")
    assert result.code == 2 and result.err.startswith("dgxm studio: error: studio.driver names a listen address")
    assert rig.started == [] and not rig.state.exists()
