"""dgxm studio up and down, and the uninstall guard, run through the CLI against a fake user manager.

The rig (tests/test_cli_studio_command.py) replaces systemctl, journalctl,
loginctl, the service's --check run and unit_state's clocks; the unit file,
the state directory and the locks are real files under a temporary home.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from dgx_monarch.cli import main as cli_main
from dgx_monarch.cli import probe_certainty, studio_command
from dgx_monarch.studio import launch, operations_lock, sign_in, unit, unit_paths, unit_state
from dgx_monarch.studio.contracts import StudioError
from test_cli_studio_command import (
    DIRECT,
    DRIVER,
    DRIVER_UNIT,
    NEW_TOKEN,
    PYTHON,
    REAL_UNIT_RUNTIME,
    TOKEN,
    WRAPPER,
    Rig,
)
from test_cli_studio_command import rig as rig

LINK = sign_in.sign_in_link("127.0.0.1", 8220, NEW_TOKEN)


def _snapshot(rig: Rig) -> set[str]:
    """Every file under the systemd user directory: up writes nothing there before every check passes."""
    root = rig.unit_path.parent
    return {str(path.relative_to(root)) for path in root.rglob("*")} if root.exists() else set()


def _up(rig: Rig, *extra: str, config: Path | None = None) -> object:
    return rig.cli("--config", str(config or rig.config()), "studio", "up", *extra)


def test_up_refuses_when_pythonpath_is_set(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(unit_paths, "unit_runtime", REAL_UNIT_RUNTIME)
    monkeypatch.setenv("PYTHONPATH", "/home/user/projects/dgx-monarch-worktree/src")
    result = _up(rig)
    assert (result.code, result.err.strip()) == (1, unit_paths.PYTHONPATH_SET)
    assert _snapshot(rig) == set() and rig.calls == []


def test_up_refuses_a_working_directory_config_and_accepts_it_by_name(rig: Rig):
    local = rig.config(where=rig.cwd / "cluster.toml")
    result = rig.cli("studio", "up")
    assert (result.code, result.err.strip()) == (1, launch.WORKDIR_CONFIG)
    assert result.out.splitlines() == [f"Config: {local} (from the working directory)"]
    assert _snapshot(rig) == set()
    result = rig.cli("studio", "up", "--config", "./cluster.toml")
    assert result.code == 0, result.text
    assert unit.unit_settings(rig.unit_path.read_text())["config"] == str(local)


def _refusals(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> dict[str, tuple[object, int, str]]:
    """Each check up makes before it writes, armed alone; the value is (arm, exit code, the refusal's start)."""
    def foreign() -> None:
        rig.unit_path.parent.mkdir(parents=True, mode=0o700)
        rig.unit_path.write_text("[Unit]\nDescription=someone else's\n")

    def dropin() -> None:
        directory = rig.home / ".config" / "systemd" / "user" / "dgxm-.service.d"
        unit_paths.make_private_directories(directory)
        (directory / "override.conf").write_text("[Service]\n")

    def held() -> None:
        rig.hold_lock()

    def symlink() -> None:
        rig.unit_path.parent.mkdir(parents=True, mode=0o700)
        rig.unit_path.symlink_to(rig.tmp / "elsewhere.service")

    def driver(port: int) -> None:
        rig.driver = {"LoadState": "loaded", "ActiveState": "active",
                      "ExecStart": DIRECT.format(python=PYTHON, port=port)}

    return {
        "root": (lambda: monkeypatch.setattr(os, "geteuid", lambda: 0), 1, unit.RUN_AS_ROOT),
        "no manager": (lambda: monkeypatch.setattr(unit_state, "user_manager", lambda **_kw: "absent"), 1,
                       unit.NO_USER_MANAGER),
        "silent manager": (lambda: monkeypatch.setattr(unit_state, "user_manager", lambda **_kw: "no_answer"),
                           probe_certainty.UNKNOWN_EXIT, unit.NO_ANSWER),
        "busy lock": (None, 1, unit_state.UNIT_BUSY),
        "broken config": (lambda: rig.config("port = 80"), 2, "error: "),
        "fresh state": (lambda: rig.config("port = 8220"), 1, launch.NO_DRIVER_ORIGIN),
        "bound elsewhere": (lambda: rig.bind("http://127.0.0.1:8188"), 1, studio_command.OTHER_ORIGIN),
        "check fails": (lambda: setattr(rig, "check", (1, "ModuleNotFoundError: No module named 'aiohttp'")), 1,
                        unit_state.CHECK_FAILED),
        "unknown driver unit": (lambda: rig.config(f'driver = "{DRIVER}"\ndriver_service = "{DRIVER_UNIT}"'), 1,
                                unit.DRIVER_UNIT_UNKNOWN),
        "driver unit elsewhere": (lambda: (driver(8188), rig.config(
            f'driver = "{DRIVER}"\ndriver_service = "{DRIVER_UNIT}"')), 1, unit.DRIVER_UNIT_ORIGIN),
        "foreign unit": (foreign, 1, unit.FOREIGN_UNIT),
        "symlinked unit": (symlink, 1, unit.UNIT_SYMLINK),
        "drop-in": (dropin, 1, unit.FOREIGN_DROPIN),
        "held state lock": (held, 1, unit.LOCK_HELD),
        "busy port": (lambda: setattr(rig, "port_busy", True), 1, sign_in.PORT_IN_USE),
        "silent status": (lambda: setattr(rig, "answers", False), probe_certainty.UNKNOWN_EXIT, unit.NO_ANSWER),
    }


REFUSALS = ["root", "no manager", "silent manager", "busy lock", "broken config", "fresh state", "bound elsewhere",
            "check fails", "unknown driver unit", "driver unit elsewhere", "foreign unit", "symlinked unit", "drop-in",
            "held state lock", "busy port", "silent status"]


@pytest.mark.parametrize("case", REFUSALS)
def test_up_order_writes_nothing_before_every_check_passes(rig: Rig, monkeypatch: pytest.MonkeyPatch, case: str):
    config = rig.config()
    arm, code, refusal = _refusals(rig, monkeypatch)[case]
    before = _snapshot(rig)
    if case == "busy lock":
        with unit_state.unit_lock():
            result = _up(rig, config=config)
    else:
        arm()
        before = _snapshot(rig)
        result = _up(rig, config=config)
    assert result.code == code, result.text
    assert result.err.startswith(refusal), result.err
    assert _snapshot(rig) == before
    writes = {"daemon-reload", "enable", "reset-failed", "restart", "disable"}
    assert not any(call[:2] == ["systemctl", "--user"] and call[2] in writes for call in rig.calls)
    assert rig.started == []


def test_unit_lock_serializes_up_down_and_rotate(rig: Rig):
    config = rig.config()
    rig.install(rig.settings(config))
    rig.serve(at=rig.now + 0.1)
    with unit_state.unit_lock():
        for argv in (["studio", "up"], ["studio", "down"], ["studio", "token", "--rotate"]):
            result = rig.cli("--config", str(config), *argv)
            assert (result.code, result.err.strip()) == (1, unit_state.UNIT_BUSY), argv
    assert rig.unit_path.exists()
    assert not any(call[2:3] in (["restart"], ["disable"]) for call in rig.calls)


def test_up_resets_failed_then_restarts_and_waits_for_a_new_token(rig: Rig):
    config = rig.config()
    rig.install(rig.settings(config), active="failed")
    rig.serve(TOKEN, at=rig.now - 50)
    result = _up(rig, config=config)
    assert result.code == 0, result.text
    verbs = [call[2] for call in rig.calls if call[:2] == ["systemctl", "--user"] and call[2] != "show"]
    assert verbs == ["daemon-reload", "enable", "reset-failed", "restart"]
    assert result.out.splitlines() == [
        f"Config: {config}", f"Driver: {DRIVER} (from [studio] driver)", f"Interpreter: {PYTHON}",
        "Source: /home/user/src", f"Unchanged {rig.unit_path}", "Started dgxm-studio.service on http://127.0.0.1:8220.",
        *sign_in.link_lines(LINK), studio_command.OPEN_HINT]
    assert TOKEN not in result.text


def test_up_writes_the_unit_on_a_fresh_install(rig: Rig):
    result = _up(rig)
    assert result.code == 0, result.text
    assert f"Wrote {rig.unit_path}" in result.out.splitlines()
    assert rig.unit_path.stat().st_mode & 0o777 == 0o600
    assert rig.unit_path.read_text().splitlines()[0] == unit.UNIT_MARKER


def test_up_reports_a_failed_unit_with_journal_lines(rig: Rig):
    rig.start = "fail"
    result = _up(rig)
    assert result.code == 1
    assert result.err.splitlines() == [unit.START_FAILED, *rig.journal]
    assert unit.owned_unit_state(rig.unit_path) == "ours"


def test_up_times_out_as_unknown(rig: Rig):
    rig.start = "hang"
    result = _up(rig)
    assert (result.code, result.err.strip()) == (probe_certainty.UNKNOWN_EXIT, unit.NOT_READY)


def test_up_refuses_an_unknown_driver_service(rig: Rig):
    result = _up(rig, config=rig.config(f'driver = "{DRIVER}"\ndriver_service = "{DRIVER_UNIT}"'))
    assert (result.code, result.err.strip()) == (1, f"{unit.DRIVER_UNIT_UNKNOWN} {DRIVER_UNIT}")
    assert _snapshot(rig) == set()


def test_up_refuses_a_driver_unit_that_serves_another_origin(rig: Rig):
    rig.driver = {"LoadState": "loaded", "ActiveState": "active", "ExecStart": DIRECT.format(python=PYTHON, port=8188)}
    result = _up(rig, config=rig.config(f'driver = "{DRIVER}"\ndriver_service = "{DRIVER_UNIT}"'))
    assert (result.code, result.err.strip()) == (1, unit.DRIVER_UNIT_ORIGIN)
    assert _snapshot(rig) == set()


def test_up_notes_a_wrapper_driver_unit_and_continues(rig: Rig):
    rig.driver = {"LoadState": "loaded", "ActiveState": "active", "ExecStart": WRAPPER}
    result = _up(rig, config=rig.config(f'driver = "{DRIVER}"\ndriver_service = "{DRIVER_UNIT}"'))
    assert result.code == 0, result.text
    assert unit.DRIVER_UNIT_NOTE in result.out.splitlines()
    assert f"Wrote {rig.unit_path} (starts with {DRIVER_UNIT})" in result.out.splitlines()
    assert "Also started" not in result.out


def test_up_refuses_a_state_directory_bound_elsewhere_with_the_move_line(rig: Rig):
    rig.bind("http://127.0.0.1:8188")
    result = _up(rig)
    assert result.code == 1
    assert result.err.splitlines() == [
        studio_command.OTHER_ORIGIN,
        launch.BOUND_ELSEWHERE.format(bound="http://127.0.0.1:8188", asked=DRIVER),
        launch.MOVE_STATE.format(state=rig.state, suffix="8188")]
    assert _snapshot(rig) == set()


def test_up_refuses_a_fresh_state_directory_without_a_driver_origin(rig: Rig):
    result = _up(rig, config=rig.config("port = 8220"))
    assert (result.code, result.err.strip()) == (1, launch.NO_DRIVER_ORIGIN)
    assert _snapshot(rig) == set() and not rig.state.exists()


def test_up_continues_when_the_binding_cannot_be_read(rig: Rig):
    unit_paths.make_private_directories(rig.state)
    (rig.state / "studio.sqlite3").write_bytes(b"not a database" * 64)
    result = _up(rig)
    assert result.code == 0, result.text
    assert studio_command.UNREADABLE_BINDING in result.out.splitlines()


def test_up_refuses_the_default_origin_when_the_binding_cannot_be_read(rig: Rig):
    # Only a readable binding equal to the default lets the default through.
    unit_paths.make_private_directories(rig.state)
    (rig.state / "studio.sqlite3").write_bytes(b"")
    result = _up(rig, config=rig.config("port = 8220"))
    assert (result.code, result.err.strip()) == (1, launch.NO_DRIVER_ORIGIN)
    assert _snapshot(rig) == set()


def test_up_restores_the_previous_unit_when_enable_fails(rig: Rig):
    config = rig.config()
    rig.install(rig.settings(config), driver="http://127.0.0.1:8192")
    before = rig.unit_path.read_text()
    rig.fails["enable"] = (1, "Failed to enable unit: Unit file is masked.")
    result = _up(rig, config=config)
    assert (result.code, result.err.strip()) == (
        1, f"{studio_command.ENABLE_FAILED} Failed to enable unit: Unit file is masked.")
    assert rig.unit_path.read_text() == before
    verbs = [call[2] for call in rig.calls if call[:2] == ["systemctl", "--user"] and call[2] != "show"]
    assert verbs == ["daemon-reload", "enable", "daemon-reload"]


def test_up_reports_a_restart_systemd_refused(rig: Rig):
    rig.fails["restart"] = (1, "Job for dgxm-studio.service failed because the control process exited.")
    result = _up(rig)
    assert result.code == 1
    assert result.err.splitlines() == [
        unit.START_FAILED, "Job for dgxm-studio.service failed because the control process exited.", *rig.journal]


def test_up_refuses_root(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    result = _up(rig)
    assert (result.code, result.err.strip()) == (1, unit.RUN_AS_ROOT)
    assert result.out == "" and rig.calls == []


def test_up_refuses_a_held_lock_and_a_busy_port(rig: Rig):
    rig.hold_lock()
    result = _up(rig)
    assert (result.code, result.err.strip()) == (1, unit.LOCK_HELD)
    rig.release()
    rig.port_busy = True
    result = _up(rig)
    assert result.code == 1
    assert result.err.splitlines() == [sign_in.PORT_IN_USE, "See which program holds it: ss -ltnp 'sport = :8220'"]
    assert _snapshot(rig) == set()


def test_up_prints_management_removed_when_the_flag_is_dropped(rig: Rig):
    config = rig.config()
    assert _up(rig, "--allow-management", config=config).code == 0
    assert unit.unit_settings(rig.unit_path.read_text())["management"] == "yes"
    rig.serve(TOKEN, at=rig.now + 0.1)
    result = _up(rig, config=config)
    assert result.code == 0, result.text
    assert result.out.splitlines()[-1] == unit.MANAGEMENT_REMOVED
    assert unit.unit_settings(rig.unit_path.read_text())["management"] == "no"


def test_up_prints_also_started_for_a_stopped_driver_unit(rig: Rig):
    rig.driver = {"LoadState": "loaded", "ActiveState": "inactive", "ExecStart": DIRECT.format(python=PYTHON, port=8191)}
    result = _up(rig, config=rig.config(f'driver = "{DRIVER}"\ndriver_service = "{DRIVER_UNIT}"'))
    assert result.code == 0, result.text
    started = result.out.splitlines().index("Started dgxm-studio.service on http://127.0.0.1:8220.")
    assert result.out.splitlines()[started + 1] == f"Also started {DRIVER_UNIT}: the Studio service wants it."
    rig.serve(TOKEN, at=rig.now + 0.1)
    assert "Also started" not in _up(rig, config=rig.config(
        f'driver = "{DRIVER}"\ndriver_service = "{DRIVER_UNIT}"')).out


def test_up_prints_no_link_for_management(rig: Rig):
    result = _up(rig, "--allow-management")
    assert result.code == 0, result.text
    token_file = rig.state / sign_in.OWNER_FILE
    assert result.out.splitlines()[-4:] == [f"Owner token file (mode 0600): {token_file}", f"Owner token: {NEW_TOKEN}",
                                            sign_in.MANAGEMENT_NO_LINK, unit.MANAGEMENT_WARNING]
    assert "#token=" not in result.text and not (rig.state / sign_in.SIGN_IN_FILE).exists()
    rig.terminal = False
    rig.serve(TOKEN, at=rig.now + 0.1, management=True)
    result = _up(rig, "--allow-management")
    assert NEW_TOKEN not in result.text and result.out.splitlines()[-2:] == [
        sign_in.MANAGEMENT_NO_LINK, unit.MANAGEMENT_WARNING]


def test_up_prints_the_linger_note(rig: Rig):
    rig.linger = "no"
    result = _up(rig)
    assert result.code == 0 and unit.LINGER_NOTE in result.out.splitlines()


def test_down_without_a_unit_is_a_no_op(rig: Rig):
    result = rig.cli("studio", "down")
    assert (result.code, result.out.strip()) == (0, unit.NO_UNIT)
    assert rig.systemctl("disable") == []


def test_down_refuses_a_foreign_unit_before_stopping_anything(rig: Rig):
    rig.unit_path.parent.mkdir(parents=True, mode=0o700)
    rig.unit_path.write_text("[Unit]\nDescription=someone else's\n")
    result = rig.cli("studio", "down")
    assert (result.code, result.err.strip()) == (1, f"{unit.FOREIGN_UNIT} {rig.unit_path}")
    assert rig.unit_path.exists() and rig.calls == []
    rig.unit_path.unlink()
    rig.unit_path.symlink_to(rig.tmp / "elsewhere.service")
    assert rig.cli("studio", "down").err.strip() == unit.UNIT_SYMLINK
    assert rig.calls == []


def test_down_reads_back_removes_leftovers_and_reports_dropins(rig: Rig):
    rig.install()
    rig.serve(at=rig.now + 0.1)
    dropins = rig.home / ".config" / "systemd" / "user" / "dgxm-.service.d"
    unit_paths.make_private_directories(dropins)
    (dropins / "override.conf").write_text("[Service]\n")
    rig.busy = ["pid 4250: dgxm update"]
    result = rig.cli("studio", "down")
    assert result.code == 1 and result.err.splitlines() == [unit_state.BUSY, "pid 4250: dgxm update"]
    assert rig.unit_path.exists() and rig.systemctl("disable") == []
    rig.busy = None
    assert rig.cli("studio", "down").err.strip() == unit_state.BUSY_UNKNOWN
    rig.busy = []
    result = rig.cli("studio", "down")
    assert result.code == 0, result.text
    assert rig.systemctl("disable") == [["systemctl", "--user", "disable", "--now", "--", unit.UNIT_NAME]]
    assert not rig.unit_path.exists()
    assert not (rig.state / sign_in.OWNER_FILE).exists() and not (rig.state / sign_in.SIGN_IN_FILE).exists()
    assert result.out.splitlines() == [
        studio_command.LEFT_DROPINS.format(paths=dropins / "override.conf"),
        studio_command.REMOVED.format(state=rig.state)]
    assert (dropins / "override.conf").exists()


def test_down_refuses_when_the_unit_is_still_loaded(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    rig.install(active="inactive")
    monkeypatch.setattr(unit, "remove_unit", lambda: None)
    result = rig.cli("studio", "down")
    assert result.code == 1 and result.err.startswith(unit.STILL_PRESENT)


def test_uninstall_refuses_while_the_studio_unit_is_installed(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    calls: list[object] = []
    monkeypatch.setattr(cli_main, "_load_config", lambda args: calls.append(args))
    rig.install()
    result = rig.cli("uninstall")
    assert (result.code, result.err.strip()) == (1, unit.UNINSTALL_STUDIO_FIRST)
    assert calls == [] and rig.calls == []
    rig.unit_path.write_text("[Unit]\nDescription=someone else's\n")
    result = rig.cli("uninstall")
    assert result.code == 0 and len(calls) == 1 and result.err == ""
    rig.unit_path.unlink()
    assert rig.cli("uninstall").code == 0 and len(calls) == 2


def test_up_records_its_settings_for_open_and_tunnel(rig: Rig):
    config = rig.config(f'driver = "{DRIVER}"\nport = 8230')
    assert rig.cli("--config", str(config), "studio", "up").code == 0
    installed = unit.unit_settings(rig.unit_path.read_text())
    assert {key: installed[key] for key in ("state", "port", "driver", "config", "management")} == {
        "state": str(rig.state), "port": "8230", "driver": DRIVER, "config": str(config), "management": "no"}


# The operations lock (studio/operations_lock.py): what the companion holds for each review and operation.

COMMANDS = {"up": ["studio", "up"], "down": ["studio", "down"], "rotate": ["studio", "token", "--rotate"]}


def _running(rig: Rig) -> list[str]:
    """A running unit of ours and the --config words every command below passes."""
    config = rig.config()
    rig.install(rig.settings(config))
    rig.serve(at=rig.now + 0.1)
    return ["--config", str(config)]


def _admits(rig: Rig) -> str:
    """What the companion answers a review or operation now: admitted, or its refusal."""
    try:
        operations_lock.release(operations_lock.hold(rig.state))
    except StudioError as exc:
        return f"{exc.code} {exc.status}: {exc}"
    return "admitted"


@pytest.mark.parametrize("name", list(COMMANDS))
def test_an_admitted_operation_with_no_child_yet_keeps_studio_running(rig: Rig, name: str):
    """The window the control group cannot see: an operation admitted whose child has not started."""
    config = _running(rig)
    admitted = operations_lock.hold(rig.state)  # what Operations.apply takes before it reserves the review
    try:
        result = rig.cli(*config, *COMMANDS[name])
    finally:
        operations_lock.release(admitted)
    assert rig.busy == [] and (result.code, result.err.strip()) == (1, unit_state.BUSY)
    assert rig.unit_path.exists()
    assert not any(call[2:3] in (["daemon-reload"], ["restart"], ["disable"]) for call in rig.calls)
    assert _admits(rig) == "admitted"  # the refusal let go of the lock
    assert rig.cli(*config, *COMMANDS[name]).code == 0


@pytest.mark.parametrize(("name", "verb"), [("up", "restart"), ("down", "disable"), ("rotate", "restart")])
def test_up_down_and_rotate_hold_the_operations_lock_through_the_restart_or_stop(
        rig: Rig, monkeypatch: pytest.MonkeyPatch, name: str, verb: str):
    """An operation that reaches the companion after the busy check is refused until the command ends."""
    config = _running(rig)
    seen: list[str] = []

    def run(argv: list[str], **options: object) -> object:
        if argv[:3] == ["systemctl", "--user", verb]:
            seen.append(_admits(rig))
        return rig.run(argv, **options)

    monkeypatch.setattr(studio_command, "_run", run)
    assert rig.cli(*config, *COMMANDS[name]).code == 0
    assert seen == [f"operation_unavailable 409: {operations_lock.DRAINING}"]
    assert _admits(rig) == "admitted"


@pytest.mark.parametrize("active", ["active", "inactive"])
def test_down_keeps_the_unit_when_systemd_refuses_the_stop(rig: Rig, active: str):
    rig.install(active=active)
    rig.serve(at=rig.now + 0.1)
    rig.fails["disable"] = (1, "Failed to disable unit.\nFailed to stop dgxm-studio.service: Access denied")
    result = rig.cli("studio", "down")
    assert result.code == 1
    assert result.err.splitlines() == [f"{unit.NOT_STOPPED} ActiveState={active}",
                                       "Failed to stop dgxm-studio.service: Access denied"]
    assert rig.unit_path.exists() and rig.systemctl("daemon-reload") == []
    assert (rig.state / sign_in.OWNER_FILE).exists()
    # The next down still finds the unit, rather than reporting that none is installed.
    del rig.fails["disable"]
    result = rig.cli("studio", "down")
    assert (result.code, result.out.splitlines()) == (0, [studio_command.REMOVED.format(state=rig.state)])
    assert not rig.unit_path.exists()


def test_down_keeps_the_unit_when_it_still_runs_after_a_clean_exit(rig: Rig):
    rig.install()
    rig.serve(at=rig.now + 0.1)
    rig.stops = False
    result = rig.cli("studio", "down")
    assert (result.code, result.err.splitlines()) == (1, [f"{unit.NOT_STOPPED} ActiveState=active"])
    assert rig.unit_path.exists() and rig.systemctl("daemon-reload") == []


def test_down_keeps_the_unit_when_the_stop_readback_gets_no_answer(rig: Rig, monkeypatch: pytest.MonkeyPatch):
    rig.install()
    rig.serve(at=rig.now + 0.1)

    def run(argv: list[str], **options: object) -> object:
        done = rig.run(argv, **options)
        if argv[:3] == ["systemctl", "--user", "disable"]:
            rig.answers = False
        return done

    monkeypatch.setattr(studio_command, "_run", run)
    result = rig.cli("studio", "down")
    assert (result.code, result.err.strip()) == (probe_certainty.UNKNOWN_EXIT, studio_command.LOST_ANSWER)
    assert rig.unit_path.exists() and rig.systemctl("daemon-reload") == []
