"""dgxm studio settings: which cluster.toml, its [studio] table, and the companion argv."""
from __future__ import annotations

import argparse
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.config import find_config_path
from dgx_monarch.config_schema import ClusterConfigError
from dgx_monarch.studio import driver_service, launch

CLUSTER = '[cluster]\nclient_bind = "tcp://192.0.2.11:0"\n[[hosts]]\nname = "192.0.2.11"\n'
STUDIO = ('[studio]\ndriver = "http://127.0.0.1:8191"\nport = 8221\n'
          'driver_service = "comfyui-driver.service"\ndriver_command = "bash /home/user/start-comfyui.sh"\n')


def _private(path: Path, text: str, mode: int = 0o600) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    return path


@pytest.fixture
def places(tmp_path, monkeypatch):
    """A home, a working directory and an environment with no cluster.toml anywhere."""
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("DGXM_CLUSTER_TOML", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.chdir(work)
    return SimpleNamespace(home=home, work=work, user=home / ".config" / "dgx-monarch" / "cluster.toml",
                           here=work / "cluster.toml", tmp=tmp_path)


def _args(**values: object) -> argparse.Namespace:
    return argparse.Namespace(**values)


def _same(found: tuple[Path | None, str | None], explicit: str | None) -> None:
    expected = find_config_path(explicit)
    assert found[0] == (None if expected is None else Path(expected).expanduser().absolute())


def test_find_studio_config_labels_every_entry_like_find_config_path(places, monkeypatch):
    assert launch.find_studio_config(None) == (None, None)
    _same(launch.find_studio_config(None), None)

    _private(places.user, CLUSTER)
    assert launch.find_studio_config(None) == (places.user, "user config")
    _same(launch.find_studio_config(None), None)

    _private(places.here, CLUSTER)
    assert launch.find_studio_config(None) == (places.here, "working directory")
    _same(launch.find_studio_config(None), None)

    named = _private(places.tmp / "env" / "cluster.toml", CLUSTER)
    monkeypatch.setenv("DGXM_CLUSTER_TOML", str(named))
    assert launch.find_studio_config(None) == (named, "$DGXM_CLUSTER_TOML")
    _same(launch.find_studio_config(None), None)

    # A missing $DGXM_CLUSTER_TOML falls through to the next entry, as every dgxm command does.
    monkeypatch.setenv("DGXM_CLUSTER_TOML", str(places.tmp / "missing.toml"))
    assert launch.find_studio_config(None) == (places.here, "working directory")
    _same(launch.find_studio_config(None), None)

    explicit = _private(places.work / "chosen.toml", CLUSTER)
    assert launch.find_studio_config("chosen.toml") == (explicit, "--config")
    _same(launch.find_studio_config("chosen.toml"), "chosen.toml")
    assert launch.find_studio_config("") == (places.here, "working directory")


def test_an_explicit_missing_config_is_refused_as_every_dgxm_command_does(places):
    with pytest.raises(ClusterConfigError, match="does not exist"):
        launch.find_studio_config("missing.toml")
    with pytest.raises(ClusterConfigError, match="does not exist"):
        find_config_path("missing.toml")


def test_read_studio_config_defaults_without_a_file_or_a_table(places):
    defaults = {"driver": None, "port": 8220, "driver_service": None, "driver_command": None}
    assert launch.read_studio_config(None) == defaults
    assert launch.read_studio_config(_private(places.user, CLUSTER)) == defaults


def test_read_studio_config_returns_the_table(places):
    assert launch.read_studio_config(_private(places.user, CLUSTER + STUDIO)) == {
        "driver": "http://127.0.0.1:8191", "port": 8221, "driver_service": "comfyui-driver.service",
        "driver_command": "bash /home/user/start-comfyui.sh"}


def test_a_file_without_hosts_gives_the_note_and_no_exit(places):
    notes: list[str] = []
    table = launch.read_studio_config(_private(places.user, STUDIO), notes)
    assert table["driver"] == "http://127.0.0.1:8191"
    assert notes == [launch.NO_HOSTS_NOTE]
    assert launch.read_studio_config(_private(places.user, "hosts = []\n" + STUDIO), notes)["port"] == 8221
    # Without a notes list the note is simply not collected.
    assert launch.read_studio_config(places.user)["port"] == 8221


@pytest.mark.parametrize(("text", "fragment"), [
    (CLUSTER + "[studio]\ndriver = \"http://0.0.0.0:8191\"\n", "studio.driver names a listen address"),
    (CLUSTER + "[studio]\nunknown = 1\n", "[studio] contains unsupported key(s): unknown"),
    (CLUSTER + "[studioo]\n", "cluster.toml top level contains unsupported key(s): studioo"),
    ("[cluster\n", "invalid TOML"),
    (CLUSTER.replace('name = "192.0.2.11"', 'name = "192.0.2.11"\ngpus = 0'), "gpus must be a positive integer"),
    (STUDIO + "[cluster]\nclient_bnd = 1\n[[hosts]]\nname = \"192.0.2.11\"\n", "client_bnd"),
])
def test_read_studio_config_refuses_with_the_file_named(places, text, fragment):
    path = _private(places.user, text)
    with pytest.raises(ClusterConfigError) as raised:
        launch.read_studio_config(path)
    assert str(raised.value).startswith(f"{path}: ")
    assert fragment in str(raised.value)


def test_invalid_utf8_is_invalid_toml(places):
    path = places.user
    path.parent.mkdir(parents=True)
    path.write_bytes(b"[studio]\ndriver = \"\xff\"\n")
    path.chmod(0o600)
    with pytest.raises(ClusterConfigError, match="invalid TOML"):
        launch.read_studio_config(path)


def _not_private(path: Path) -> None:
    with pytest.raises(ClusterConfigError) as raised:
        launch.read_studio_config(path)
    assert str(raised.value) == f"{path}: {launch.CONFIG_NOT_PRIVATE}"


def test_read_studio_config_refuses_a_file_of_another_owner(places, monkeypatch):
    path = _private(places.user, CLUSTER)
    real = os.getuid()
    monkeypatch.setattr(launch.os, "getuid", lambda: real + 1)
    _not_private(path)


def test_read_studio_config_refuses_a_world_writable_file(places):
    _not_private(_private(places.user, CLUSTER, 0o646))


def _group(monkeypatch, *, others: bool) -> None:
    gid = os.getgid()
    members = ["alice"] if others else []
    monkeypatch.setattr(driver_service, "grp", SimpleNamespace(
        getgrgid=lambda _gid: SimpleNamespace(gr_mem=members)))
    monkeypatch.setattr(driver_service, "pwd", SimpleNamespace(
        getpwuid=lambda _uid: SimpleNamespace(pw_name="user"),
        getpwall=lambda: [SimpleNamespace(pw_name="user", pw_gid=gid)]))


def test_read_studio_config_refuses_a_group_writable_file_whose_group_holds_another_account(places, monkeypatch):
    path = _private(places.user, CLUSTER, 0o664)
    _group(monkeypatch, others=True)
    _not_private(path)


def test_read_studio_config_accepts_a_group_writable_file_whose_group_holds_only_the_owner(places, monkeypatch):
    path = _private(places.user, CLUSTER + STUDIO, 0o664)
    _group(monkeypatch, others=False)
    assert launch.read_studio_config(path)["port"] == 8221


def test_read_studio_config_refuses_a_symlink_a_directory_and_a_fifo(places):
    target = _private(places.tmp / "real.toml", CLUSTER)
    places.user.parent.mkdir(parents=True)
    places.user.symlink_to(target)
    _not_private(places.user)
    directory = places.tmp / "dir.toml"
    directory.mkdir()
    _not_private(directory)
    fifo = places.tmp / "fifo.toml"
    os.mkfifo(fifo, 0o600)
    result: list[BaseException] = []

    def read() -> None:
        try:
            launch.read_studio_config(fifo)
        except BaseException as exc:  # recorded for the assertion below
            result.append(exc)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    reader.join(5)
    assert not reader.is_alive(), "reading a FIFO blocked"
    assert isinstance(result[0], ClusterConfigError)
    assert str(result[0]) == f"{fifo}: {launch.CONFIG_NOT_PRIVATE}"


def test_flag_beats_table_beats_default_for_every_key(places):
    table = launch.read_studio_config(_private(places.user, CLUSTER + STUDIO))
    config, source = launch.find_studio_config(None)
    settings = launch.resolve_settings(_args(driver=None, state_dir=None, allow_management=False),
                                       config, source, table)
    assert settings == launch.StudioSettings(
        port=8221, driver="http://127.0.0.1:8191", driver_source="[studio] driver",
        state_dir=places.home / ".local" / "state" / "dgx-monarch" / "studio",
        config=places.user, config_source="user config", driver_service="comfyui-driver.service",
        allow_management=False)

    typed = launch.resolve_settings(
        _args(driver="http://127.0.0.1:8195", state_dir=places.tmp / "state", allow_management=True),
        config, source, table)
    assert (typed.driver, typed.driver_source) == ("http://127.0.0.1:8195", "--driver")
    assert typed.state_dir == places.tmp / "state"
    assert typed.allow_management is True

    bare = launch.resolve_settings(_args(), None, None, launch.read_studio_config(None))
    assert (bare.driver, bare.driver_source) == ("http://127.0.0.1:8188", "default")
    assert (bare.port, bare.config, bare.config_source, bare.driver_service) == (8220, None, None, None)
    assert bare.allow_management is False


def test_management_is_read_from_the_flag_of_the_command_given(places):
    table = launch.read_studio_config(None)
    up = launch.resolve_settings(_args(studio_action="up", allow_management=False, up_allow_management=True),
                                 None, None, table)
    assert up.allow_management is True
    # The foreground flag never reaches up, and up's flag never reaches the foreground start.
    assert not launch.resolve_settings(_args(studio_action="up", allow_management=True), None, None,
                                       table).allow_management
    assert not launch.resolve_settings(_args(studio_action=None, up_allow_management=True), None, None,
                                       table).allow_management


def test_driver_override_skips_only_the_loopback_rule(places):
    table = launch.read_studio_config(None)
    remote = launch.resolve_settings(_args(driver="http://192.0.2.10:8188"), None, None, table)
    assert remote.driver == "http://192.0.2.10:8188"
    for value, message in (
            ("http://192.0.2.10:8188/", "studio.driver must be written as 'http://192.0.2.10:8188'"),
            ("http://0.0.0.0:8188", "studio.driver names a listen address"),
            ("http://[fe80::1%25eth0]:8188", "with a host and no user, path, query, fragment or zone")):
        with pytest.raises(ValueError) as raised:
            launch.resolve_settings(_args(driver=value), None, None, table)
        assert message in str(raised.value)


def test_state_dir_and_config_are_absolute(places, monkeypatch):
    table = launch.read_studio_config(None)
    with pytest.raises(ValueError) as raised:
        launch.resolve_settings(_args(state_dir=Path("relative/state")), None, None, table)
    assert str(raised.value) == "--state-dir must be an absolute path"
    home_state = launch.resolve_settings(_args(state_dir=Path("~/state")), None, None, table)
    assert home_state.state_dir == places.home / "state"
    settings = launch.resolve_settings(_args(), Path("cluster.toml"), "--config", table)
    assert settings.config == places.work / "cluster.toml"
    assert settings.config.is_absolute()


def test_companion_argv_order_and_content(places):
    settings = launch.StudioSettings(
        port=8221, driver="http://127.0.0.1:8191", driver_source="[studio] driver",
        state_dir=Path("/home/user/.local/state/dgx-monarch/studio"),
        config=Path("/home/user/.config/dgx-monarch/cluster.toml"), config_source="user config",
        driver_service="comfyui-driver.service", allow_management=True)
    assert launch.companion_argv(settings, no_open=True) == [
        "--host", "127.0.0.1", "--port", "8221", "--driver", "http://127.0.0.1:8191",
        "--state-dir", "/home/user/.local/state/dgx-monarch/studio",
        "--config", "/home/user/.config/dgx-monarch/cluster.toml",
        "--driver-service", "comfyui-driver.service", "--allow-management", "--no-open"]
    bare = launch.StudioSettings(
        port=8220, driver="http://127.0.0.1:8188", driver_source="default",
        state_dir=Path("/home/user/state"), config=None, config_source=None, driver_service=None,
        allow_management=False)
    assert launch.companion_argv(bare, no_open=False) == [
        "--host", "127.0.0.1", "--port", "8220", "--driver", "http://127.0.0.1:8188",
        "--state-dir", "/home/user/state"]


def test_settings_never_probe_for_a_driver(places, monkeypatch):
    import socket

    from dgx_monarch.cli import comfy_ports

    def refuse(*_args, **_kwargs):
        raise AssertionError("settings opened a socket or probed for a driver")

    for name in dir(comfy_ports):
        if callable(getattr(comfy_ports, name)) and not name.startswith("__") and name[0].islower():
            monkeypatch.setattr(comfy_ports, name, refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    _private(places.user, CLUSTER + STUDIO)
    config, source = launch.find_studio_config(None)
    settings = launch.resolve_settings(_args(), config, source, launch.read_studio_config(config))
    assert launch.companion_argv(settings, no_open=True)[:6] == [
        "--host", "127.0.0.1", "--port", "8221", "--driver", "http://127.0.0.1:8191"]


def test_default_state_dir_keeps_the_companion_rule(places, monkeypatch):
    assert launch.default_state_dir() == places.home / ".local" / "state" / "dgx-monarch" / "studio"
    monkeypatch.setenv("XDG_STATE_HOME", "")
    assert launch.default_state_dir() == places.home / ".local" / "state" / "dgx-monarch" / "studio"
    monkeypatch.setenv("XDG_STATE_HOME", str(places.tmp / "xdg"))
    assert launch.default_state_dir() == places.tmp / "xdg" / "dgx-monarch" / "studio"
    monkeypatch.setenv("XDG_STATE_HOME", "relative")
    with pytest.raises(SystemExit) as raised:
        launch.default_state_dir()
    assert str(raised.value) == ("XDG_STATE_HOME must be an absolute path; unset it or pass an absolute "
                                 "--state-dir.")
    with pytest.raises(SystemExit):
        launch.resolve_settings(_args(), None, None, launch.read_studio_config(None))


class _Clock:
    @staticmethod
    def now(tz=None):
        return datetime(2026, 10, 22, 12, 30, 5, tzinfo=tz or UTC)


def test_mismatch_lines_name_both_origins_and_a_free_target(places, monkeypatch):
    monkeypatch.setattr(launch, "datetime", _Clock)
    state = places.tmp / "state" / "studio"
    state.mkdir(parents=True)
    bound, asked = "http://127.0.0.1:8188", "http://127.0.0.1:8191"
    assert launch.mismatch_lines(state, bound, asked) == [
        f"This state directory is bound to {bound}; this launch asked for {asked}.",
        "To use the new driver with fresh records, move the old ones aside, then start Studio again: "
        f"mv {state} {state}-8188"]
    # mv into an existing directory would move the state inside it; lstat counts a dangling symlink too.
    Path(f"{state}-8188").symlink_to(places.tmp / "nowhere")
    assert launch.mismatch_lines(state, bound, asked)[1].endswith(f"mv {state} {state}-8188-20261022123005")
    Path(f"{state}-8188-20261022123005").mkdir()
    assert launch.mismatch_lines(state, bound, asked) == [
        f"This state directory is bound to {bound}; this launch asked for {asked}."]
    assert launch.move_state_line(state, bound) == ""
    # A bound origin without a port takes the scheme's port.
    assert launch.move_state_line(state, "http://localhost").endswith(f"{state}-80")
    assert launch.move_state_line(state, "https://127.0.0.1").endswith(f"{state}-443")


def test_the_move_line_quotes_a_state_path_with_a_space(places):
    state = places.tmp / "my state"
    line = launch.move_state_line(state, "http://127.0.0.1:8188")
    assert line.endswith(f"mv '{state}' '{state}'-8188")
    assert launch.MOVE_STATE.format(state="S", suffix="X").endswith("mv S S-X")


def test_launch_constants_hold_the_design_text():
    assert launch.UNIT_RUNS == (
        "The always-on Studio is running; its settings come from its unit. Run dgxm studio down to use these "
        "flags here, or change cluster.toml and run dgxm studio up.")
    assert launch.TERMINAL_STUDIO == (
        "Studio already runs in another terminal on this account and printed its sign-in link or token there. "
        "Stop it there with Ctrl+C, or use dgxm studio up for an always-on Studio.")
    assert launch.NO_DRIVER_ORIGIN == (
        'No driver origin is set. Add driver to the [studio] table of cluster.toml, for example driver = '
        '"http://127.0.0.1:8188", or pass --driver. Studio keeps its records for the first driver address '
        "it uses.")
    assert launch.CONFIG_NOT_PRIVATE == (
        "Studio reads cluster.toml only when it is a regular file you own that no other account can write; "
        "fix its owner or mode, for example chmod 600.")
    assert launch.NO_HOSTS_NOTE == (
        "Note: this cluster.toml has no [[hosts]], so cluster actions stay unavailable in Studio.")
    assert launch.WORKDIR_CONFIG == (
        "dgxm studio up does not write a cluster.toml found in the working directory into a service; pass "
        "--config with its path, or keep it at ~/.config/dgx-monarch/cluster.toml.")
    assert launch.MOVE_STATE == (
        "To use the new driver with fresh records, move the old ones aside, then start Studio again: "
        "mv {state} {state}-{suffix}")
    assert launch.EXTRA_MISSING == "studio extra not installed"


def test_launch_imports_no_optional_dependency():
    import subprocess
    import sys

    code = ("import json, sys\nfor name in ('aiohttp', 'PIL'):\n    sys.modules[name] = None\n"
            "import dgx_monarch.studio.launch, dgx_monarch.studio.sign_in\n"
            "print(json.dumps(sorted(m for m in sys.modules if m.startswith('dgx_monarch.studio.'))))\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                            env={**os.environ, "PYTHONPATH": str(Path(launch.__file__).parents[2])})
    assert result.returncode == 0, result.stderr
    assert set(json.loads(result.stdout)) <= {
        "dgx_monarch.studio.launch", "dgx_monarch.studio.sign_in", "dgx_monarch.studio.driver_service",
        "dgx_monarch.studio.contracts"}
