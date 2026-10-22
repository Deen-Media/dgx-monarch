"""The always-on Studio unit primitives, with no user manager, display or clock of this machine.

Every systemctl, journalctl and loginctl call goes to a fake runner, and a
live subprocess fails the test. The unit directory lives under a temporary
XDG_CONFIG_HOME, the unit lock under a temporary XDG_STATE_HOME, and the
clocks, control groups and process table are replaced in unit_state.
Nothing here imports aiohttp or Pillow.
"""
from __future__ import annotations

import fcntl
import os
import shlex
import socket
import sqlite3
import stat
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import dgx_monarch
from dgx_monarch.studio import driver_service, sign_in, unit, unit_paths, unit_state
from dgx_monarch.studio.launch import StudioSettings
from dgx_monarch.studio.unit_paths import UnitRuntime

RUNTIME = UnitRuntime(interpreter=Path("/home/user/monarch-env/bin/python"),
                      source=Path("/home/user/projects/dgx-monarch/src"),
                      dependencies=Path("/home/user/monarch-env/lib/python3.12/site-packages"))
DIGEST = "0123456789abcdef" * 4
FIRST = "A" * 43
SECOND = "B" * 43

GOLDEN = """\
# dgxm-managed-studio-unit-v1
# dgxm-studio-state=/home/user/.local/state/dgx-monarch/studio
# dgxm-studio-port=8220
# dgxm-studio-driver=http://127.0.0.1:8191
# dgxm-studio-config=/home/user/.config/dgx-monarch/cluster.toml
# dgxm-studio-driver-service=comfyui-driver.service
# dgxm-studio-management=no
# dgxm-studio-interpreter=/home/user/monarch-env/bin/python
# dgxm-studio-pythonpath=/home/user/projects/dgx-monarch/src:/home/user/monarch-env/lib/python3.12/site-packages
# dgxm-studio-source=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
[Unit]
Description=DGX Monarch Studio companion
Documentation=https://github.com/Deen-Media/dgx-monarch/blob/main/docs/STUDIO.md#keep-studio-running
Wants=comfyui-driver.service
After=comfyui-driver.service
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
UnsetEnvironment=PYTHONHOME PYTHONUSERBASE PYTHONSTARTUP PYTHONINSPECT PYTHONWARNINGS LD_ASSUME_KERNEL LD_AUDIT \
LD_BIND_NOT LD_BIND_NOW LD_DEBUG LD_DEBUG_OUTPUT LD_DYNAMIC_WEAK LD_HWCAP_MASK LD_LIBRARY_PATH LD_ORIGIN_PATH \
LD_POINTER_GUARD LD_PREFER_MAP_32BIT_EXEC LD_PRELOAD LD_PROFILE LD_PROFILE_OUTPUT LD_SHOW_AUXV LD_TRACE_LOADED_OBJECTS \
LD_TRACE_PRELINKING LD_USE_LOAD_BIAS LD_VERBOSE LD_WARN GLIBC_TUNABLES GCONV_PATH LOCPATH NLSPATH
WorkingDirectory=/
ExecStart="/usr/bin/env" "-i" "HOME=%h" "USER=%u" "LOGNAME=%u" "XDG_RUNTIME_DIR=%t" "PATH=/usr/bin:/bin" \
"LANG=C.UTF-8" "PYTHONPATH=/home/user/projects/dgx-monarch/src:/home/user/monarch-env/lib/python3.12/site-packages" \
"PYTHONNOUSERSITE=1" "PYTHONSAFEPATH=1" "PYTHONDONTWRITEBYTECODE=1" "PYTHONPYCACHEPREFIX=/proc/self/fd/2147483647" \
"/home/user/monarch-env/bin/python" "-S" "-P" "-s" "-B" "-m" "dgx_monarch.studio" "--host" "127.0.0.1" "--port" \
"8220" "--driver" "http://127.0.0.1:8191" "--state-dir" "/home/user/.local/state/dgx-monarch/studio" "--config" \
"/home/user/.config/dgx-monarch/cluster.toml" "--driver-service" "comfyui-driver.service" "--no-open"
UMask=0077
NoNewPrivileges=yes
Restart=on-failure
RestartSec=3
KillMode=control-group
SendSIGKILL=yes
TimeoutStopSec=75

[Install]
WantedBy=default.target
"""


def settings(**changes: object) -> StudioSettings:
    values: dict[str, object] = {
        "port": 8220, "driver": "http://127.0.0.1:8191", "driver_source": "[studio] driver",
        "state_dir": Path("/home/user/.local/state/dgx-monarch/studio"),
        "config": Path("/home/user/.config/dgx-monarch/cluster.toml"), "config_source": "user config",
        "driver_service": "comfyui-driver.service", "allow_management": False}
    values.update(changes)
    return StudioSettings(**values)


def replaced(text: str, *pairs: tuple[str, str]) -> str:
    for old, new in pairs:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text


def exec_words(text: str) -> list[str]:
    line = next(line for line in text.splitlines() if line.startswith("ExecStart="))
    return shlex.split(line.removeprefix("ExecStart="))


def completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout, "")


@pytest.fixture(autouse=True)
def no_live_processes(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("a test reached a live process")

    monkeypatch.setattr(subprocess, "run", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A private home whose XDG directories are temporary; nothing reads the real ones."""
    root = tmp_path / "home"
    root.mkdir(mode=0o700)
    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    environ = {"HOME": str(root), "XDG_CONFIG_HOME": str(root / "config"), "XDG_DATA_HOME": str(root / "data"),
               "XDG_STATE_HOME": str(root / "state"), "XDG_RUNTIME_DIR": str(runtime)}
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    return SimpleNamespace(root=root, environ=environ, units=root / "config" / "systemd" / "user",
                           unit=root / "config" / "systemd" / "user" / unit.UNIT_NAME)


class Clock:
    """A monotonic and a wall clock that move only when the code sleeps."""

    def __init__(self, monkeypatch, start: float = 1000.0):
        self.now = start
        self.hooks: list = []
        monkeypatch.setattr(unit_state, "_monotonic", lambda: self.now)
        monkeypatch.setattr(unit_state, "_realtime", lambda: 1_700_000_000.0 + self.now)
        monkeypatch.setattr(unit_state, "_sleep", self.sleep)

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        for hook in list(self.hooks):
            hook(self.now)

    @staticmethod
    def wall_ns(monotonic: float) -> int:
        return int((1_700_000_000.0 + monotonic) * 1_000_000_000)


def status(active: str = "active", *, pid: int = 4242, start: float = 1000.5,
           group: str = "/user.slice/user-1000.slice/app.slice/dgxm-studio.service"):
    return unit_state.UnitStatus(load_state="loaded", active_state=active, sub_state="running", main_pid=pid,
                                 control_group=group, fragment_path="", start_monotonic_us=int(start * 1_000_000))


def write_private(path: Path, text: str, *, wall_ns: int | None = None) -> None:
    path.write_text(text)
    path.chmod(0o600)
    if wall_ns is not None:
        os.utime(path, ns=(wall_ns, wall_ns))


def write_token_and_page(state: Path, value: str, wall_ns: int) -> None:
    write_private(state / sign_in.OWNER_FILE, f"{value}\n", wall_ns=wall_ns)
    page = sign_in.sign_in_page(sign_in.sign_in_link("127.0.0.1", 8220, value))
    write_private(state / sign_in.SIGN_IN_FILE, page, wall_ns=wall_ns)


# The unit text


def test_unit_text_is_exact():
    assert unit.render_unit(settings(), RUNTIME, source_digest=DIGEST) == GOLDEN
    without_service = replaced(
        GOLDEN, ("# dgxm-studio-driver-service=comfyui-driver.service\n", "# dgxm-studio-driver-service=\n"),
        ("Wants=comfyui-driver.service\nAfter=comfyui-driver.service\n", ""),
        (' "--driver-service" "comfyui-driver.service"', ""))
    assert unit.render_unit(settings(driver_service=None), RUNTIME, source_digest=DIGEST) == without_service
    management = (("# dgxm-studio-management=no\n", "# dgxm-studio-management=yes\n"),
                  ('"--no-open"', '"--allow-management" "--no-open"'))
    assert unit.render_unit(settings(allow_management=True), RUNTIME, source_digest=DIGEST) == replaced(
        GOLDEN, *management)
    assert unit.render_unit(settings(driver_service=None, allow_management=True), RUNTIME,
                            source_digest=DIGEST) == replaced(without_service, *management)
    # A config-less Studio passes no --config and renders the field empty.
    no_config = unit.render_unit(settings(config=None), RUNTIME, source_digest=DIGEST)
    assert "# dgxm-studio-config=\n" in no_config and '"--config"' not in no_config
    # A non-editable install pins one import entry.
    single = UnitRuntime(RUNTIME.interpreter, RUNTIME.dependencies, RUNTIME.dependencies)
    assert '"PYTHONPATH=/home/user/monarch-env/lib/python3.12/site-packages"' in unit.render_unit(
        settings(), single, source_digest=DIGEST)
    with pytest.raises(ValueError, match="source digest"):
        unit.render_unit(settings(), RUNTIME, source_digest="not-a-digest")


def test_unit_quotes_percent_quote_and_backslash_in_paths():
    odd = Path('/home/user/state %h "quoted" back\\slash')
    text = unit.render_unit(settings(state_dir=odd), RUNTIME, source_digest=DIGEST)
    assert '"--state-dir" "/home/user/state %%h \\"quoted\\" back\\\\slash"' in text
    # The comment line keeps the value as written; systemd ignores comments.
    assert f"# dgxm-studio-state={odd}\n" in text
    assert unit.unit_settings(text)["state"] == str(odd)
    # Only the four specifier words keep a bare %.
    bare = [word for word in exec_words(text) if "%" in word and "%%" not in word]
    assert bare == ["HOME=%h", "USER=%u", "LOGNAME=%u", "XDG_RUNTIME_DIR=%t"]


@pytest.mark.parametrize("change", [
    {"state_dir": Path("/home/user/$HOME/studio")},
    {"state_dir": Path("/home/user/${XDG}/studio")},
    {"config": Path("/home/user/cluster$1.toml")},
    {"state_dir": Path("/home/user/state\nExecStartPre=/bin/false")},
    {"config": Path("/home/user/cluster\n.toml")},
])
def test_unit_refuses_dollar_signs_and_newlines_in_paths(change):
    with pytest.raises(SystemExit) as refused:
        unit.render_unit(settings(**change), RUNTIME, source_digest=DIGEST)
    message = str(refused.value)
    assert message.startswith(unit_paths.UNIT_PATH_CHARS)
    assert "\n" not in message
    dollar = UnitRuntime(Path("/home/user/$venv/bin/python"), RUNTIME.source, RUNTIME.dependencies)
    with pytest.raises(SystemExit, match=r"^The service cannot hold a path"):
        unit.render_unit(settings(), dollar, source_digest=DIGEST)
    with pytest.raises(SystemExit, match=r"^The service cannot hold a path"):
        unit_paths.check_runtime(dollar)


@pytest.mark.parametrize("runtime", [
    UnitRuntime(RUNTIME.interpreter, Path("/home/user/src:/home/user/evil"), RUNTIME.dependencies),
    UnitRuntime(RUNTIME.interpreter, RUNTIME.source, Path("/home/user/site:packages")),
    UnitRuntime(RUNTIME.interpreter, Path("/home/user/src\x01"), RUNTIME.dependencies),
    UnitRuntime(RUNTIME.interpreter, RUNTIME.source, Path("/home/user/site\x7fpackages")),
    UnitRuntime(Path("/home/user/venv\tbin/python"), RUNTIME.source, RUNTIME.dependencies),
])
def test_unit_refuses_control_characters_and_colons_in_pythonpath(runtime):
    for attempt in (lambda: unit.render_unit(settings(), runtime, source_digest=DIGEST),
                    lambda: unit_paths.check_runtime(runtime)):
        with pytest.raises(SystemExit) as refused:
            attempt()
        assert str(refused.value).startswith(unit_paths.UNIT_PATH_CHARS)
        assert all(32 <= ord(char) != 127 for char in str(refused.value))
    # A colon in the interpreter path is not a PYTHONPATH entry and stays legal.
    unit_paths.check_path_chars("/home/user/v:env/bin/python")


def test_unit_and_preflight_share_one_environment():
    text = unit.render_unit(settings(), RUNTIME, source_digest=DIGEST)
    words = exec_words(text)
    environment = unit_paths.unit_environment(RUNTIME)
    command = unit_paths.service_command(RUNTIME)
    assert words[:2] == ["/usr/bin/env", "-i"]
    assert words[2:2 + len(environment)] == environment
    assert words[2 + len(environment):2 + len(environment) + len(command)] == command
    filled = unit_paths.preflight_environment(RUNTIME, home="/home/user", user="user", runtime_dir="/run/user/1000")
    expected = {}
    for word in environment:
        name, _, value = word.partition("=")
        expected[name] = value.replace("%h", "/home/user").replace("%u", "user").replace("%t", "/run/user/1000")
    assert filled == expected
    assert filled["PYTHONPATH"] == RUNTIME.pythonpath and filled["LOGNAME"] == "user"
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return completed("dgx-monarch Studio 0.0.0: 9 packaged resources match their manifest; studio extra installed\n")

    assert unit_state.check_preflight(RUNTIME, filled, run=run)[0] == "ok"
    argv, kwargs = calls[0]
    assert argv == [*command, "--check"]
    assert kwargs["env"] == filled and kwargs["cwd"] == "/" and kwargs["timeout"] == 60


def test_unit_environment_carries_the_runtime_directory():
    environment = unit_paths.unit_environment(RUNTIME)
    assert "XDG_RUNTIME_DIR=%t" in environment
    assert {"HOME=%h", "USER=%u", "LOGNAME=%u"} <= set(environment)
    text = unit.render_unit(settings(), RUNTIME, source_digest=DIGEST)
    assert ' "XDG_RUNTIME_DIR=%t" ' in text
    filled = unit_paths.preflight_environment(RUNTIME, home="/home/user", user="user", runtime_dir="/run/user/4242")
    assert filled["XDG_RUNTIME_DIR"] == "/run/user/4242"
    # Nothing else from this shell reaches the service: env -i starts empty.
    assert set(filled) == {word.partition("=")[0] for word in environment}
    import pwd

    entry = pwd.getpwuid(os.getuid())
    assert unit_state.specifier_values({"XDG_RUNTIME_DIR": "/run/user/4242"}) == (
        entry.pw_dir, entry.pw_name, "/run/user/4242")
    assert unit_state.specifier_values({})[2] == f"/run/user/{os.getuid()}"


def test_check_preflight_outcomes():
    def answer(output: str, code: int = 0):
        return lambda argv, **kwargs: completed(output, code)

    assert unit_state.check_preflight(RUNTIME, {}, run=answer("ok line\n")) == ("ok", "ok line")
    missing = ("dgx-monarch Studio 0.0.0: 9 packaged resources match their manifest; studio extra not installed "
               "(missing aiohttp); install it as docs/INSTALL.md describes before starting the companion\n")
    outcome, line = unit_state.check_preflight(RUNTIME, {}, run=answer(missing))
    assert outcome == "failed" and line.startswith(f"{unit_state.CHECK_FAILED} dgx-monarch Studio")
    outcome, line = unit_state.check_preflight(RUNTIME, {}, run=answer("Traceback\nImportError: no module\n", 1))
    assert (outcome, line) == ("failed", f"{unit_state.CHECK_FAILED} ImportError: no module")
    assert unit_state.check_preflight(RUNTIME, {}, run=answer("", 3)) == (
        "failed", f"{unit_state.CHECK_FAILED} exit status 3")

    def slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 60)

    assert unit_state.check_preflight(RUNTIME, {}, run=slow) == ("timeout", unit_state.CHECK_TIMEOUT)

    def absent(argv, **kwargs):
        raise FileNotFoundError(2, "No such file or directory")

    assert unit_state.check_preflight(RUNTIME, {}, run=absent) == (
        "failed", f"{unit_state.CHECK_FAILED} No such file or directory")


# The runtime and its owners


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A fake interpreter, checkout and site-packages; the parents above the tree read as root's."""
    base = tmp_path / "tree"
    interpreter = base / "venv" / "bin" / "python"
    source = base / "checkout" / "src"
    site = base / "venv" / "lib" / "site-packages"
    for directory in (interpreter.parent, source / "dgx_monarch", site / "aiohttp", site / "PIL"):
        directory.mkdir(parents=True)
    for path in (base, *base.rglob("*")):
        if path.is_dir():
            path.chmod(0o755)
    interpreter.write_text("#!fake interpreter, never run\n")
    interpreter.chmod(0o755)
    (source / "dgx_monarch" / "__init__.py").write_text("")
    real_stat = os.stat
    outside = {"uid": 0}

    def described(path):
        if Path(path).is_relative_to(base):
            return real_stat(path)
        return SimpleNamespace(st_uid=outside["uid"], st_gid=0, st_mode=stat.S_IFDIR | 0o755)

    def spec(name):
        return SimpleNamespace(submodule_search_locations=[str(site / name)], origin=str(site / name / "__init__.py"))

    monkeypatch.setattr(unit_paths, "_stat", described)
    monkeypatch.setattr(unit_paths, "find_spec", spec)
    monkeypatch.setattr(sys, "executable", str(interpreter))
    monkeypatch.setattr(dgx_monarch, "__file__", str(source / "dgx_monarch" / "__init__.py"))
    return SimpleNamespace(base=base, interpreter=interpreter, source=source, site=site, outside=outside)


def test_unit_runtime_refuses_when_pythonpath_is_set(tree):
    with pytest.raises(SystemExit) as refused:
        unit_paths.unit_runtime({"PYTHONPATH": str(tree.source)})
    assert str(refused.value) == unit_paths.PYTHONPATH_SET
    # Empty counts as unset, and the runtime comes from this process.
    runtime = unit_paths.unit_runtime({"PYTHONPATH": ""})
    assert runtime == UnitRuntime(tree.interpreter, tree.source, tree.site)
    assert runtime.pythonpath == f"{tree.source}:{tree.site}"


def test_unit_runtime_refuses_a_missing_or_split_extra(tree, monkeypatch):
    from dgx_monarch.studio.__main__ import INSTALL_HINT

    monkeypatch.setattr(unit_paths, "find_spec", lambda name: None if name == "PIL" else SimpleNamespace(
        submodule_search_locations=[str(tree.site / name)], origin=None))
    with pytest.raises(SystemExit) as refused:
        unit_paths.unit_runtime({})
    assert str(refused.value) == f"The studio extra is not installed for {tree.interpreter}. {INSTALL_HINT}"
    other = tree.base / "elsewhere"
    monkeypatch.setattr(unit_paths, "find_spec", lambda name: SimpleNamespace(
        submodule_search_locations=[str((tree.site if name == "aiohttp" else other) / name)], origin=None))
    with pytest.raises(SystemExit) as refused:
        unit_paths.unit_runtime({})
    assert str(refused.value) == ("Studio's dependencies must be in one site-packages directory: "
                                  f"aiohttp in {tree.site}, Pillow in {other}.")


def test_unit_runtime_refuses_paths_another_account_can_write(tree):
    for path, mode in ((tree.base / "checkout", 0o777), (tree.site, 0o757), (tree.interpreter, 0o757),
                       (tree.base / "venv", 0o773)):
        before = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode)
        try:
            with pytest.raises(SystemExit) as refused:
                unit_paths.unit_runtime({})
            assert str(refused.value) == f"{unit_paths.UNIT_PATH_OWNER} {path}"
        finally:
            path.chmod(before)
    # A parent owned by another account refuses too, before anything is written.
    tree.outside["uid"] = os.geteuid() + 4242
    with pytest.raises(SystemExit) as refused:
        unit_paths.unit_runtime({})
    assert str(refused.value) == f"{unit_paths.UNIT_PATH_OWNER} /"
    tree.outside["uid"] = 0
    # A link is checked where it points as well as where it sits.
    target = tree.base / "shared"
    target.mkdir(mode=0o777)
    target.chmod(0o777)
    (target / "python").write_text("")
    (target / "python").chmod(0o755)
    tree.interpreter.unlink()
    tree.interpreter.symlink_to(target / "python")
    with pytest.raises(SystemExit) as refused:
        unit_paths.unit_runtime({})
    assert str(refused.value) == f"{unit_paths.UNIT_PATH_OWNER} {target}"


def test_unit_runtime_accepts_a_group_writable_path_whose_group_holds_only_the_owner(tree, monkeypatch):
    import pwd

    checkout = tree.base / "checkout"
    for path in (checkout, tree.source):
        path.chmod(0o775)  # the head's umask 002 checkout
    owner = pwd.getpwuid(os.geteuid()).pw_name
    group = checkout.stat().st_gid
    monkeypatch.setattr(driver_service.pwd, "getpwall", lambda: [SimpleNamespace(pw_name=owner, pw_gid=group)])
    monkeypatch.setattr(driver_service.grp, "getgrgid", lambda gid: SimpleNamespace(gr_mem=[]))
    assert unit_paths.unit_runtime({}).source == tree.source
    monkeypatch.setattr(driver_service.grp, "getgrgid", lambda gid: SimpleNamespace(gr_mem=[owner, "worker"]))
    with pytest.raises(SystemExit) as refused:
        unit_paths.unit_runtime({})
    assert str(refused.value) == f"{unit_paths.UNIT_PATH_OWNER} {checkout}"


def test_unit_directory_check_refuses_an_unsafe_systemd_user_directory(tmp_path, monkeypatch):
    units = tmp_path / "config" / "systemd" / "user"
    unit_paths.check_unit_directory(units, create=False)
    assert not units.parent.exists()
    unit_paths.check_unit_directory(units, create=True)
    for path in (tmp_path / "config", units.parent, units):
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    units.chmod(0o755)
    unit_paths.check_unit_directory(units, create=False)
    for mode in (0o775, 0o757):
        units.chmod(mode)
        with pytest.raises(SystemExit, match="must be a directory you own"):
            unit_paths.check_unit_directory(units, create=True)
    units.chmod(0o700)
    with monkeypatch.context() as patch, pytest.raises(SystemExit, match="must be a directory you own"):
        patch.setattr(os, "geteuid", lambda: os.getuid() + 4242)
        unit_paths.check_unit_directory(units, create=False)
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(real)
    a_file = tmp_path / "file"
    a_file.write_text("")
    for path in (linked, a_file):
        with pytest.raises(SystemExit) as refused:
            unit_paths.check_unit_directory(path, create=True)
        assert str(refused.value) == unit_paths.UNIT_DIR_UNSAFE


# Ownership, drop-ins and the readback


def test_ownership_probe_verdicts(home, monkeypatch):
    path = home.unit
    assert unit.unit_path() == path
    assert unit.owned_unit_state() == "absent" and unit.read_owned_unit() is None
    path.parent.mkdir(parents=True)
    ours = unit.render_unit(settings(), RUNTIME, source_digest=DIGEST)
    write_private(path, ours)
    assert unit.owned_unit_state() == "ours" and unit.read_owned_unit() == ours
    write_private(path, "# dgxm-managed-unit-v1\n[Unit]\n")
    assert unit.owned_unit_state() == "foreign" and unit.read_owned_unit() is None
    write_private(path, "[Unit]\n" + ours)
    assert unit.owned_unit_state() == "foreign"
    write_private(path, ours)
    with monkeypatch.context() as patch:
        patch.setattr(os, "geteuid", lambda: os.getuid() + 4242)
        assert unit.owned_unit_state() == "foreign"
    path.chmod(0o000)
    try:
        assert unit.owned_unit_state() in {"unknown", "ours"}  # root reads it anyway
    finally:
        path.chmod(0o600)
    target = home.root / "target.service"
    target.write_text(ours)
    path.unlink()
    path.symlink_to(target)
    assert unit.owned_unit_state() == "symlink" and unit.read_owned_unit() is None
    path.unlink()
    path.mkdir()
    assert unit.owned_unit_state() == "foreign"
    path.rmdir()
    write_private(path, ours.replace("\n", "\n" + "#" * 70000, 1))
    assert unit.owned_unit_state() == "foreign"
    # An explicit path is read as given, and a missing parent reads absent.
    assert unit.owned_unit_state(home.root / "nowhere" / unit.UNIT_NAME) == "absent"


def test_ownership_probe_refuses_a_unit_another_account_can_write(home, monkeypatch):
    import pwd

    path = home.unit
    path.parent.mkdir(parents=True)
    ours = unit.render_unit(settings(), RUNTIME, source_digest=DIGEST)
    write_private(path, ours)
    owner = pwd.getpwuid(os.geteuid()).pw_name
    group = path.stat().st_gid
    monkeypatch.setattr(driver_service.pwd, "getpwall", lambda: [SimpleNamespace(pw_name=owner, pw_gid=group)])
    monkeypatch.setattr(driver_service.grp, "getgrgid", lambda gid: SimpleNamespace(gr_mem=[]))
    path.chmod(0o664)  # group write, but the group holds only this user
    assert unit.owned_unit_state() == "ours" and unit.read_owned_unit() == ours
    monkeypatch.setattr(driver_service.grp, "getgrgid", lambda gid: SimpleNamespace(gr_mem=[owner, "worker"]))
    path.chmod(0o660)  # group write for a group that holds another account
    assert unit.owned_unit_state() == "foreign" and unit.read_owned_unit() is None
    path.chmod(0o602)  # write for every account
    assert unit.owned_unit_state() == "foreign"
    path.chmod(0o600)
    assert unit.owned_unit_state() == "ours"


def test_dropin_scan_covers_both_names_in_all_three_directories(home):
    plain = {"HOME": str(home.root), "XDG_RUNTIME_DIR": home.environ["XDG_RUNTIME_DIR"]}
    directories = unit.search_directories(plain)
    assert directories == [home.root / ".config/systemd/user", home.root / ".local/share/systemd/user",
                           Path(home.environ["XDG_RUNTIME_DIR"]) / "systemd/user"]
    planted = []
    for directory in directories:
        for name in unit.DROPIN_NAMES:
            dropin = directory / name
            dropin.mkdir(parents=True)
            (dropin / "50-override.conf").write_text("[Service]\nExecStart=\n")
            planted.append(dropin / "50-override.conf")
    assert unit.foreign_dropins(plain) == planted
    assert unit.DROPIN_NAMES == ("dgxm-studio.service.d", "dgxm-.service.d")
    # An XDG base directory that differs from its default is scanned as well.
    moved = home.root / "config" / "systemd" / "user" / "dgxm-.service.d"
    moved.mkdir(parents=True)
    (moved / ".hidden.conf").write_text("")
    assert moved / ".hidden.conf" in unit.foreign_dropins(home.environ)
    assert unit.search_directories({"HOME": str(home.root), "XDG_RUNTIME_DIR": "relative"})[-1] == (
        home.root / ".local/share/systemd/user")


def test_an_empty_dropin_directory_passes(home):
    for directory in unit.search_directories(home.environ):
        for name in unit.DROPIN_NAMES:
            (directory / name).mkdir(parents=True)
    assert unit.foreign_dropins(home.environ) == []
    # A file where a drop-in directory belongs is reported, never skipped.
    blocked = unit.search_directories(home.environ)[0] / "dgxm-studio.service.d"
    blocked.rmdir()
    blocked.write_text("")
    assert unit.foreign_dropins(home.environ) == [blocked]


def test_readback_scopes_drop_ins_by_directory(home):
    good = {"LoadState": "loaded", "FragmentPath": str(home.unit), "DropInPaths": "", "NeedDaemonReload": "no"}
    assert unit.judge_readback(good, home.unit, home.environ) == (None, [])
    vendor = "/usr/lib/systemd/user/service.d/10-timeout-abort.conf"
    detail, notes = unit.judge_readback({**good, "DropInPaths": vendor}, home.unit, home.environ)
    assert detail is None and notes == [f"{unit.VENDOR_DROPIN} {vendor}"]
    runtime = home.environ["XDG_RUNTIME_DIR"]
    for refused in (f"{home.root}/config/systemd/user.control/dgxm-studio.service.d/50-x.conf",
                    f"{home.root}/.config/systemd/user.control/service.d/50-x.conf",
                    f"{runtime}/systemd/generator/dgxm-studio.service.d/50-x.conf",
                    f"{runtime}/systemd/transient/service.d/50-x.conf",
                    f"{home.root}/data/systemd/user/service.d/50-x.conf",
                    "/etc/systemd/user/service.d/50-x.conf",
                    "/usr/lib/systemd/user/dgxm-.service.d/50-x.conf",
                    "/run/systemd/user/service.d/50-x.conf",
                    "relative/50-x.conf"):
        detail, notes = unit.judge_readback({**good, "DropInPaths": f"{vendor} {refused}"}, home.unit, home.environ)
        assert detail == f"DropInPaths={refused}", refused
        assert notes == [f"{unit.VENDOR_DROPIN} {vendor}"]
    for name, value in (("FragmentPath", "/etc/systemd/user/dgxm-studio.service"), ("NeedDaemonReload", "yes"),
                        ("LoadState", "not-found")):
        detail, _ = unit.judge_readback({**good, name: value}, home.unit, home.environ)
        assert detail == f"{name}={value}"
    detail, _ = unit.judge_readback({**good, "DropInPaths": "'unbalanced"}, home.unit, home.environ)
    assert detail == "DropInPaths='unbalanced"


def test_publish_is_atomic_private_and_never_follows_a_symlink(home, monkeypatch):
    text = unit.render_unit(settings(), RUNTIME, source_digest=DIGEST)
    assert unit.publish_unit(text) == "wrote"
    assert home.unit.read_text() == text
    assert stat.S_IMODE(home.unit.stat().st_mode) == 0o600
    assert stat.S_IMODE(home.units.stat().st_mode) == 0o700
    assert sorted(entry.name for entry in home.units.iterdir()) == [unit.UNIT_NAME]
    assert unit.publish_unit(text) == "unchanged"
    target = home.root / "elsewhere.service"
    target.write_text("someone else's\n")
    home.unit.unlink()
    home.unit.symlink_to(target)
    with pytest.raises(SystemExit) as refused:
        unit.publish_unit(text)
    assert str(refused.value) == unit.UNIT_SYMLINK
    assert target.read_text() == "someone else's\n" and home.unit.is_symlink()
    home.unit.unlink()
    write_private(home.unit, "[Unit]\nDescription=not ours\n")
    with pytest.raises(SystemExit) as refused:
        unit.publish_unit(text)
    assert str(refused.value) == f"{unit.FOREIGN_UNIT} {home.unit}"
    assert home.unit.read_text() == "[Unit]\nDescription=not ours\n"
    home.unit.unlink()
    # A link that appears after the ownership check is replaced by the rename; its target is untouched.
    real_check = unit.check_unit_directory

    def racing(path, *, create):
        real_check(path, create=create)
        if not home.unit.is_symlink():
            home.unit.symlink_to(target)

    monkeypatch.setattr(unit, "check_unit_directory", racing)
    assert unit.publish_unit(text) == "wrote"
    assert not home.unit.is_symlink() and home.unit.read_text() == text
    assert target.read_text() == "someone else's\n"
    assert stat.S_IMODE(home.unit.stat().st_mode) == 0o600
    assert sorted(entry.name for entry in home.units.iterdir()) == [unit.UNIT_NAME]


def test_restore_puts_back_the_previous_bytes(home):
    first = unit.render_unit(settings(), RUNTIME, source_digest=DIGEST)
    second = unit.render_unit(settings(port=8221), RUNTIME, source_digest=DIGEST)
    assert unit.publish_unit(first) == "wrote"
    assert unit.publish_unit(second) == "wrote"
    unit.restore_unit(first)
    assert home.unit.read_text() == first and stat.S_IMODE(home.unit.stat().st_mode) == 0o600
    unit.restore_unit(None)
    assert not home.unit.exists()
    write_private(home.unit, "[Unit]\n")
    unit.restore_unit(None)
    assert home.unit.read_text() == "[Unit]\n"  # a file dgxm did not write is never removed


def test_down_primitives_remove_only_what_dgxm_wrote(home, tmp_path):
    text = unit.render_unit(settings(), RUNTIME, source_digest=DIGEST)
    unit.remove_unit()  # absent: nothing to do
    unit.publish_unit(text)
    unit.remove_unit()
    assert not home.unit.exists()
    write_private(home.unit, "[Unit]\n")
    with pytest.raises(SystemExit, match=r"^Another file already uses"):
        unit.remove_unit()
    assert home.unit.exists()
    home.unit.unlink()
    gone = {"LoadState": "not-found", "ActiveState": "inactive"}
    assert unit.judge_removal(gone) is None
    assert unit.judge_removal({**gone, "ActiveState": "failed"}) is None
    assert unit.judge_removal({"LoadState": "loaded", "ActiveState": "active"}) == (
        "LoadState=loaded; ActiveState=active")
    write_private(home.unit, text)
    assert unit.judge_removal(gone) == str(home.unit)
    # Leftover token files go only while no companion holds the lock, and only in their exact shape.
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    write_token_and_page(state, FIRST, Clock.wall_ns(1000))
    (state / "companion.lock").write_text("")
    holder = os.open(state / "companion.lock", os.O_RDWR)
    try:
        fcntl.flock(holder, fcntl.LOCK_EX)
        assert unit_state.remove_leftovers(state) == []
    finally:
        os.close(holder)
    assert unit_state.remove_leftovers(state) == [state / sign_in.OWNER_FILE, state / sign_in.SIGN_IN_FILE]
    assert sorted(entry.name for entry in state.iterdir()) == ["companion.lock"]
    write_private(state / sign_in.OWNER_FILE, "not a token\n")
    write_private(state / sign_in.SIGN_IN_FILE, "<p>another page</p>\n")
    assert unit_state.remove_leftovers(state) == []
    assert (state / sign_in.OWNER_FILE).exists() and (state / sign_in.SIGN_IN_FILE).exists()


# Locks, the busy check and the lock probe


def test_unit_lock_refuses_while_held(home):
    lock = home.root / "state" / "dgx-monarch" / "studio-unit.lock"
    with unit_state.unit_lock():
        assert stat.S_IMODE(lock.stat().st_mode) == 0o600
        assert stat.S_IMODE(lock.parent.stat().st_mode) == 0o700
        with pytest.raises(SystemExit) as refused, unit_state.unit_lock():
            pass
        assert str(refused.value) == unit_state.UNIT_BUSY
    with unit_state.unit_lock():
        pass  # released on exit
    target = home.root / "target"
    target.write_text("keep\n")
    lock.unlink()
    lock.symlink_to(target)
    with pytest.raises(SystemExit, match=r"^The Studio unit lock must be"), unit_state.unit_lock():
        pass
    assert target.read_text() == "keep\n"
    lock.unlink()
    lock.parent.chmod(0o777)
    try:
        with pytest.raises(SystemExit, match=r"^The Studio unit lock must be"), unit_state.unit_lock():
            pass
    finally:
        lock.parent.chmod(0o700)


def test_busy_check(tmp_path, monkeypatch):
    cgroups, proc = tmp_path / "cgroup", tmp_path / "proc"
    monkeypatch.setattr(unit_state, "_CGROUP_ROOT", cgroups)
    monkeypatch.setattr(unit_state, "_PROC", proc)
    running = status(pid=100)
    procs = cgroups / running.control_group.lstrip("/") / "cgroup.procs"
    procs.parent.mkdir(parents=True)
    procs.write_text("100\n")
    assert unit_state.busy_processes(running) == []
    procs.write_text("100\n200\n300\n400\n")
    for pid, argv in ((200, ["/home/user/monarch-env/bin/python", "-P", "-s", "-B", "-m", "dgx_monarch", "update",
                             "--yes", "--extra"]),
                      (400, ["/usr/bin/rsync", "-a", "src/", "worker:dst/"])):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / "cmdline").write_bytes(b"\0".join(word.encode() for word in argv) + b"\0")
    # pid 300 ended between the two reads and is not in flight any more.
    assert unit_state.busy_processes(running) == ["pid 200: dgxm update --yes", "pid 400: rsync -a src/"]
    procs.write_text("100\nnot-a-pid\n")
    assert unit_state.busy_processes(running) is None
    procs.unlink()
    assert unit_state.busy_processes(running) is None
    for group in ("", "/", "relative/dgxm-studio.service", "/user.slice/../../etc"):
        assert unit_state.busy_processes(status(group=group)) is None


def test_the_lock_probe_never_creates_the_lock_and_refuses_a_symlink(tmp_path):
    state = tmp_path / "state"
    assert unit_state.lock_probe(state) == "absent" and not state.exists()
    state.mkdir(mode=0o700)
    assert unit_state.lock_probe(state) == "absent"
    assert list(state.iterdir()) == []
    lock = state / "companion.lock"
    lock.write_text("")
    lock.chmod(0o600)
    assert unit_state.lock_probe(state) == "free"
    holder = os.open(lock, os.O_RDWR)
    try:
        fcntl.flock(holder, fcntl.LOCK_EX)
        assert unit_state.lock_probe(state) == "held"
    finally:
        os.close(holder)
    assert unit_state.lock_probe(state) == "free"
    lock.unlink()
    target = tmp_path / "target"
    target.write_text("")
    lock.symlink_to(target)
    with pytest.raises(SystemExit, match=r"^Cannot use the state directory"):
        unit_state.lock_probe(state)
    lock.unlink()
    lock.mkdir()
    with pytest.raises(SystemExit) as refused:
        unit_state.lock_probe(state)
    assert str(refused.value) in {unit_state.STATE_PRIVATE} or str(refused.value).startswith("Cannot use")
    lock.rmdir()
    state.chmod(0o755)
    with pytest.raises(SystemExit) as refused:
        unit_state.lock_probe(state)
    assert str(refused.value) == unit_state.STATE_PRIVATE
    state.chmod(0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(state)
    with pytest.raises(SystemExit) as refused:
        unit_state.lock_probe(linked / "inner")
    assert str(refused.value) == unit_state.STATE_SYMLINK
    assert list(state.iterdir()) == []


# Readiness


def test_wait_ready_needs_a_file_newer_than_the_main_process_start(tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    # A page and token left by a process that crashed before the restart.
    write_token_and_page(state, FIRST, Clock.wall_ns(990.0))
    seen = []

    def status_fn():
        seen.append(clock.now)
        return status(start=1000.5)

    def appear(now):
        if now >= 1001.0 and (state / sign_in.OWNER_FILE).read_text() != f"{SECOND}\n":
            write_token_and_page(state, SECOND, Clock.wall_ns(1001.0))

    clock.hooks.append(appear)
    assert unit_state.wait_ready(state, management=False, after_monotonic=1000.0, previous_token=FIRST,
                                 status_fn=status_fn) == "ready"
    assert clock.now == pytest.approx(1001.0)
    assert len(seen) == 5  # 0.25 second polls until the new files appear
    # A start before the restart stamp is the old process, whatever its files say.
    clock.hooks.clear()
    assert unit_state.wait_ready(state, management=False, after_monotonic=1000.6, previous_token=None,
                                 timeout=1.0, status_fn=lambda: status(start=1000.5)) == "unknown"
    # One second of slack before the start; a file older than that is stale.
    clock.now = 1100.0
    write_token_and_page(state, SECOND, Clock.wall_ns(1049.6))
    assert unit_state.wait_ready(state, management=False, after_monotonic=None, previous_token=None,
                                 timeout=0.5, status_fn=lambda: status(start=1050.5)) == "ready"
    write_token_and_page(state, SECOND, Clock.wall_ns(1049.4))
    assert unit_state.wait_ready(state, management=False, after_monotonic=None, previous_token=None,
                                 timeout=0.5, status_fn=lambda: status(start=1050.5)) == "unknown"
    # The same token as before a restart is not ready, even when the file is new.
    write_token_and_page(state, SECOND, Clock.wall_ns(1060.0))
    assert unit_state.wait_ready(state, management=False, after_monotonic=None, previous_token=SECOND,
                                 timeout=0.5, status_fn=lambda: status(start=1050.5)) == "unknown"
    # Without management the page must be new too; with management only the token file counts.
    write_private(state / sign_in.SIGN_IN_FILE, (state / sign_in.SIGN_IN_FILE).read_text(),
                  wall_ns=Clock.wall_ns(1000.0))
    assert unit_state.wait_ready(state, management=False, after_monotonic=None, previous_token=None,
                                 timeout=0.5, status_fn=lambda: status(start=1050.5)) == "unknown"
    (state / sign_in.SIGN_IN_FILE).unlink()
    assert unit_state.wait_ready(state, management=True, after_monotonic=None, previous_token=None,
                                 timeout=0.5, status_fn=lambda: status(start=1050.5)) == "ready"
    # A main process that is gone, or a status read before the process started, is not ready.
    for unready in (status(pid=0, start=1050.5), status(start=0.0), status("inactive", start=1050.5)):
        assert unit_state.wait_ready(state, management=True, after_monotonic=None, previous_token=None,
                                     timeout=0.5, status_fn=lambda unready=unready: unready) == "unknown"
    # A file that holds another thing is never ready.
    write_private(state / sign_in.OWNER_FILE, "<html>\n", wall_ns=Clock.wall_ns(1060.0))
    assert unit_state.wait_ready(state, management=True, after_monotonic=None, previous_token=None,
                                 timeout=0.5, status_fn=lambda: status(start=1050.5)) == "unknown"


def test_wait_ready_reports_failed_and_unknown(tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    assert unit_state.wait_ready(state, management=False, after_monotonic=999.0, previous_token=None,
                                 status_fn=lambda: status("failed")) == "failed"
    assert clock.now == 1000.0
    assert unit_state.wait_ready(state, management=False, after_monotonic=999.0, previous_token=None,
                                 status_fn=lambda: status("inactive", pid=0)) == "failed"
    # Auto-restart reads activating with no main process: keep waiting, then unknown.
    assert unit_state.wait_ready(state, management=False, after_monotonic=999.0, previous_token=None,
                                 status_fn=lambda: status("activating", pid=0)) == "unknown"
    assert clock.now == pytest.approx(1030.0)
    # No answer from the user manager is unknown at the deadline, never failed.
    start = clock.now
    assert unit_state.wait_ready(state, management=False, after_monotonic=None, previous_token=None,
                                 status_fn=lambda: None) == "unknown"
    assert clock.now - start == pytest.approx(30.0)
    # The default status reader is this module's, looked up when the wait runs.
    monkeypatch.setattr(unit_state, "unit_status", lambda: status("failed"))
    assert unit_state.wait_ready(state, management=False, after_monotonic=None, previous_token=None) == "failed"


# Drift and stale code


def test_drift_fields_name_what_differs():
    installed = unit.unit_settings(unit.render_unit(settings(), RUNTIME, source_digest=DIGEST))
    assert installed == unit.rendered_settings(settings(), RUNTIME, DIGEST)
    assert list(installed) == list(unit.SETTING_FIELDS)
    assert unit.drift_fields(installed, unit.rendered_settings(settings(), RUNTIME, "f" * 64)) == []
    moved = UnitRuntime(Path("/home/user/other-env/bin/python"), RUNTIME.source, RUNTIME.dependencies)
    current = unit.rendered_settings(settings(driver="http://127.0.0.1:8188"), moved, DIGEST)
    assert unit.drift_fields(installed, current) == ["driver", "interpreter"]
    current = unit.rendered_settings(settings(driver_service=None, allow_management=True, config=None), RUNTIME,
                                     DIGEST)
    assert unit.drift_fields(installed, current) == ["config", "driver-service", "management"]
    # A unit without the comment lines differs in every field.
    assert unit.drift_fields({}, installed) == [name for name in unit.SETTING_FIELDS if name != "source"]
    assert unit.unit_settings("# dgxm-studio-port=1\n# dgxm-studio-port=2\n# dgxm-studio-other=3\n") == {"port": "1"}


def test_source_changed_follows_the_digest(tmp_path):
    from dgx_monarch.runtime_provenance import dgx_source_manifest_sha256

    package = tmp_path / "src" / "dgx_monarch"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n")
    (package / "helper.py").write_text("def helper():\n    return 1\n")
    digest = dgx_source_manifest_sha256(package)
    installed = {"pythonpath": f"{package.parent}:/home/user/monarch-env/lib/python3.12/site-packages",
                 "source": digest}
    assert unit.source_changed(installed) is False
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "helper.cpython-312.pyc").write_bytes(b"\0")
    assert unit.source_changed(installed) is False  # a cache the CLI writes changes nothing
    (package / "helper.py").write_text("def helper():\n    return 2\n")
    assert unit.source_changed(installed) is True
    assert unit.source_changed({**installed, "pythonpath": str(tmp_path / "missing")}) is None
    assert unit.source_changed({**installed, "pythonpath": "relative/src"}) is None
    assert unit.source_changed({"pythonpath": str(package.parent)}) is None
    assert unit.source_changed({}) is None


# The origin preflight and the driver unit


def test_origin_preflight_outcomes(tmp_path):
    state = tmp_path / "st?ate#1"
    origin, other = "http://127.0.0.1:8191", "http://127.0.0.1:8188"
    assert unit_state.origin_preflight(state, origin, True) == ("no_origin", None)
    assert unit_state.origin_preflight(state, origin, False) == ("ok", None)
    assert not state.exists()
    state.mkdir(mode=0o700)
    assert unit_state.origin_preflight(state, origin, True) == ("no_origin", None)
    database = state / "studio.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.close()
    assert unit_state.origin_preflight(state, origin, True) == ("no_origin", None)
    assert unit_state.origin_preflight(state, origin, False) == ("ok", None)
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO meta VALUES ('driver_origin', ?)", (origin,))
    connection.close()
    assert unit_state.origin_preflight(state, origin, True) == ("ok", None)
    assert unit_state.origin_preflight(state, other, False) == ("differs", origin)
    assert unit_state.origin_preflight(state, other, True) == ("differs", origin)
    entries = sorted(entry.name for entry in state.iterdir())
    database.write_bytes(b"not a database, just bytes" * 100)
    assert unit_state.origin_preflight(state, origin, False) == ("unreadable", None)
    database.unlink()
    database.mkdir()
    assert unit_state.origin_preflight(state, origin, True) == ("unreadable", None)
    database.rmdir()
    assert entries == ["studio.sqlite3"]  # a read-only open leaves no journal behind
    state.chmod(0o750)
    with pytest.raises(SystemExit) as refused:
        unit_state.origin_preflight(state, origin, True)
    assert str(refused.value) == unit_state.STATE_PRIVATE
    state.chmod(0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(state)
    with pytest.raises(SystemExit) as refused:
        unit_state.origin_preflight(linked, origin, True)
    assert str(refused.value) == unit_state.STATE_SYMLINK


def exec_start(executable: str, *argv: str) -> str:
    words = " ".join((executable, *argv))
    return (f"{{ path={executable} ; argv[]={words} ; ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; "
            "pid=0 ; code=(null) ; status=0/0 }")


def test_driver_unit_check_outcomes():
    calls = []
    python = "/home/user/venv/bin/python"
    main = "/home/user/ComfyUI/main.py"

    def answering(load="loaded", active="inactive", start=exec_start(python, main, "--listen", "127.0.0.1",
                                                                       "--port", "8191"), code=0):
        def run(argv, **kwargs):
            calls.append(argv)
            asked = argv[argv.index("--no-pager") + 1].removeprefix("--property=").split(",")
            # systemd 255 prints no ExecStart= at all for a unit it could not load.
            fields = {"LoadState": load, "ActiveState": active, **({"ExecStart": start} if load == "loaded" else {})}
            return completed("".join(f"{name}={fields[name]}\n" for name in asked if name in fields), code)
        return run

    origin = "http://127.0.0.1:8191"
    assert unit_state.driver_unit_check("comfyui-driver.service", origin, run=answering()) == ("ok", "", "inactive")
    assert calls[0] == ["systemctl", "--user", "show", "--no-pager", "--property=LoadState,ActiveState",
                        "--", "comfyui-driver.service"]
    assert calls[1] == ["systemctl", "--user", "show", "--no-pager", "--property=ExecStart",
                        "--", "comfyui-driver.service"]
    assert unit_state.driver_unit_check("comfyui-driver.service", "http://127.0.0.1:8188",
                                        run=answering(active="active")) == (
        "origin", unit.DRIVER_UNIT_ORIGIN, "active")
    assert unit_state.driver_unit_check("comfyui-driver.service", "http://127.0.0.1:8188", run=answering(
        start=exec_start(python, main))) == ("ok", "", "inactive")  # ComfyUI's defaults
    wrapper = exec_start("/home/user/bin/start-comfy")
    assert unit_state.driver_unit_check("comfyui-driver.service", origin, run=answering(start=wrapper)) == (
        "note", unit.DRIVER_UNIT_NOTE, "inactive")
    abbreviated = exec_start(python, main, "--list", "127.0.0.1")
    assert unit_state.driver_unit_check("comfyui-driver.service", origin, run=answering(start=abbreviated))[0] == (
        "note")
    assert unit_state.driver_unit_check("missing.service", origin, run=answering(load="not-found", start="")) == (
        "unknown_unit", f"{unit.DRIVER_UNIT_UNKNOWN} missing.service", "inactive")
    assert unit_state.driver_unit_check("masked.service", origin, run=answering(load="masked", start="")) == (
        "other_state", "[studio] driver_service names masked.service, whose LoadState is masked.", "inactive")
    calls.clear()
    assert unit_state.driver_unit_check("missing.service", origin, run=answering(load="not-found"))[0] == (
        "unknown_unit")
    assert len(calls) == 1  # no ExecStart read for a unit systemd could not load
    assert unit_state.driver_unit_check("comfyui-driver.service", origin, run=answering(code=1)) == (
        "no_answer", unit.NO_ANSWER, None)

    def slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 10)

    assert unit_state.driver_unit_check("comfyui-driver.service", origin, run=slow) == (
        "no_answer", unit.NO_ANSWER, None)


def test_driver_inspect_ignores_reverse_dependencies(tmp_path, monkeypatch):
    """A driver unit the Studio unit wants gains WantedBy= and Before=; the adapter never reads them."""
    from dgx_monarch.studio.driver_service import _MARKERS, _PROPERTIES, DriverService

    unit_root = tmp_path / "units"
    unit_root.mkdir()
    driver_unit = unit_root / "comfyui-driver.service"
    driver_unit.write_text("[Service]\nExecStart=/python /comfy/main.py\n")
    driver_unit.chmod(0o644)
    source_root = tmp_path / "ComfyUI"
    source_root.mkdir()
    for name, markers in _MARKERS.items():
        (source_root / name).write_text("\n".join(markers))
        (source_root / name).chmod(0o644)
    executable = tmp_path / "python3"
    executable.write_bytes(b"fake executable never invoked")
    executable.chmod(0o755)
    service = DriverService(driver_unit.name, origin="http://127.0.0.1:8191")
    service.owner_root = unit_root
    argv = [str(executable), str(source_root / "main.py"), "--listen", "127.0.0.1", "--port", "8191"]
    fields = dict.fromkeys(_PROPERTIES, "")
    fields.update(Id=driver_unit.name, LoadState="loaded", Transient="no", NeedDaemonReload="no",
                  FragmentPath=str(driver_unit), ExecStart=exec_start(*argv), ActiveState="inactive",
                  SubState="dead", MainPID="0", UnitFileState="enabled", Restart="no", KillMode="control-group")
    assert "WantedBy" not in _PROPERTIES and "Before" not in _PROPERTIES
    reverse = {**fields, "WantedBy": "default.target dgxm-studio.service", "Before": "dgxm-studio.service"}
    monkeypatch.setattr(service, "_show", lambda: deepcopy(reverse))
    identity = service.inspect()
    assert identity["state"] == "inactive" and identity["config"]["listen"]["port"] == 8191
    # A forward dependency on the Studio unit is still coupling, and still refused.
    monkeypatch.setattr(service, "_show", lambda: {**deepcopy(fields), "Wants": "dgxm-studio.service"})
    with pytest.raises(driver_service.StudioError, match="coupled"):
        service.inspect()


# The user manager, the journal, linger and the port


def test_user_manager_status_journal_and_linger_keep_no_answer_apart(monkeypatch):
    calls = []

    def run_with(stdout: str, code: int = 0):
        def run(argv, **kwargs):
            calls.append((argv, kwargs))
            return completed(stdout, code)
        return run

    which = {"systemctl": "/usr/bin/systemctl"}.get
    assert unit_state.user_manager(run=run_with(""), which=which) == "ok"
    assert calls[-1][0] == ["systemctl", "--user", "show-environment"]
    assert calls[-1][1]["stdout"] == subprocess.DEVNULL  # the manager's environment is never read
    assert unit_state.user_manager(run=run_with("", 1), which=which) == "absent"
    assert unit_state.user_manager(run=run_with(""), which=lambda name: None) == "absent"

    def slow(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 10)

    assert unit_state.user_manager(run=slow, which=which) == "no_answer"
    shown = ("LoadState=loaded\nActiveState=active\nSubState=running\nMainPID=4242\nControlGroup=/app.slice/x\n"
             "FragmentPath=/home/user/.config/systemd/user/dgxm-studio.service\n"
             "ExecMainStartTimestampMonotonic=1000500000\n")
    found = unit_state.unit_status(run=run_with(shown))
    assert found == unit_state.UnitStatus("loaded", "active", "running", 4242, "/app.slice/x",
                                          "/home/user/.config/systemd/user/dgxm-studio.service", 1000500000)
    assert unit_state.unit_active(found)
    assert calls[-1][0][-2:] == ["--", "dgxm-studio.service"] and calls[-1][1]["timeout"] == 10
    for active, expected in (("activating", True), ("reloading", True), ("deactivating", True),
                             ("inactive", False), ("failed", False)):
        assert unit_state.unit_active(status(active)) is expected
    absent = shown.replace("LoadState=loaded", "LoadState=not-found").replace("MainPID=4242", "MainPID=0")
    assert unit_state.unit_status(run=run_with(absent)).load_state == "not-found"
    assert unit_state.unit_status(run=run_with(shown, 1)) is None
    assert unit_state.unit_status(run=slow) is None
    assert unit_state.unit_status(run=run_with("LoadState=loaded\n")) is None
    assert unit_state.unit_status(run=run_with(shown.replace("MainPID=4242", "MainPID=many"))) is None
    # The default runner is looked up at call time, so a patched subprocess.run is what runs.
    monkeypatch.setattr(subprocess, "run", run_with(shown))
    assert unit_state.unit_status() == found
    lines = "".join(f"line {number}\n" for number in range(25)) + "bell\x07\n"
    assert unit_state.journal_tail(1_700_000_000.9, run=run_with(lines)) == [
        *(f"line {number}" for number in range(6, 25)), "bell\\x07"]
    assert calls[-1][0] == ["journalctl", "--user", "-u", "dgxm-studio.service", "--since", "@1700000000",
                            "-n", "20", "--no-pager", "-o", "cat"]
    assert unit_state.journal_tail(0, run=run_with("x\n", 1)) == []
    assert unit_state.journal_tail(0, run=slow) == []
    assert unit_state.linger_state(run=run_with("no\n")) == "no"
    assert calls[-1][0] == ["loginctl", "show-user", str(os.getuid()), "--property=Linger", "--value"]
    assert unit_state.linger_state(run=run_with("yes\n", 1)) is None
    assert unit_state.linger_state(run=slow) is None


def test_port_probe_binds_only_studio_loopback_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]
        assert unit_state.port_in_use(port) is True
    assert unit_state.port_in_use(port) is False
