"""``dgxm studio``: the front door that starts Studio and manages its user service.

Settings come from cluster.toml and typed flags only; nothing probes for a driver. A token or a
sign-in link goes only to a terminal. A refusal prints to stderr and raises SystemExit, as the
companion does. No aiohttp and no companion module is imported at import time.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn
from urllib.parse import urlsplit

from .. import runtime_provenance
from ..config import ClusterConfigError
from ..studio import launch, operations_lock, sign_in, tunnel, unit, unit_paths, unit_state
from ..studio.contracts import StudioError
from . import probe_certainty

if TYPE_CHECKING:
    from ..config_schema import StudioTable

FOREGROUND_ONLY = ("--driver, --state-dir, --allow-management and --no-open apply only to dgxm studio itself; "
                   "give up its own --allow-management after the word up.")
LOST_ANSWER = "The user manager stopped answering; check systemctl --user status dgxm-studio.service."
ENABLE_FAILED = "systemctl --user enable dgxm-studio.service failed:"
INSTALLED_FAILED = "dgxm-studio.service is installed but failed; starting Studio in this terminal."
INSTALLED_INACTIVE = "dgxm-studio.service is installed but inactive; starting Studio in this terminal."
UNREADABLE_BINDING = "Could not read the state directory's driver binding; the service refuses at start if it differs."
OTHER_ORIGIN = ("This state directory belongs to a different driver origin. "
                "Use its original driver or choose a separate --state-dir.")
OPEN_HINT = "Run dgxm studio or dgxm studio open to open it in a browser."
ROTATED = "Issued a new token. Tabs signed in with the old one must use the new link."
REMOVED = "Stopped Studio and removed its unit. Studio's records stay in {state}; dgxm studio up starts it again."
LEFT_DROPINS = "Left drop-ins dgxm did not write: {paths}."
NO_CONFIG = ("Config: none found (looked at $DGXM_CLUSTER_TOML, ./cluster.toml and "
             "~/.config/dgx-monarch/cluster.toml); cluster actions stay unavailable.")
DRIVER_FROM = {"--driver": "from --driver", "[studio] driver": "from [studio] driver",
               "default": "default; this state directory is already bound to it"}
UNKNOWN = probe_certainty.UNKNOWN_EXIT
Config = tuple["Path | None", "str | None", "StudioTable", list[str], "ClusterConfigError | None"]
View = tuple[str, dict[str, str], "unit_state.UnitStatus | None"]


def run(args: argparse.Namespace) -> int:
    """Dispatch ``dgxm studio`` and its actions; the return value is the exit code."""
    action = args.studio_action
    if action and (args.driver is not None or args.state_dir is not None or args.allow_management or args.no_open):
        _usage(FOREGROUND_ONLY)
    try:
        if action in ("up", "down"):
            _manager()
            # The stack holds the operations lock, once taken, until the restart or stop has finished.
            with unit_state.unit_lock(), contextlib.ExitStack() as hold:
                return _install(args, hold) if action == "up" else _remove(hold)
        return {None: _one_word, "open": _open, "token": _token, "tunnel": _tunnel}[action](args)
    except StudioError as exc:
        _refuse(str(exc))


# Every systemctl, journalctl, loginctl and xdg-mime call goes through _run, and the one
# xdg-open start through _popen, so a test replaces two names.
def _run(argv: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, **options)


def _popen(argv: list[str], **options: Any) -> subprocess.Popen[bytes]:
    return subprocess.Popen(argv, **options)


def _say(*lines: str, err: bool = False) -> None:
    for line in lines:
        print(line, file=sys.stderr if err else sys.stdout, flush=True)


def _refuse(*lines: str, code: int = 1) -> NoReturn:
    _say(*lines, err=True)
    raise SystemExit(code)


def _usage(message: str, *path: str) -> NoReturn:
    """A usage error in argparse's words and exit code."""
    _refuse(f"{' '.join(('dgxm', 'studio', *path))}: error: {message}", code=2)


def _systemctl(*words: str, timeout: float = 30) -> subprocess.CompletedProcess[str] | None:
    """``systemctl --user`` by argv list; None when it did not answer in time."""
    try:
        return _run(["systemctl", "--user", *words], capture_output=True, text=True, timeout=timeout, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return None


def _status() -> unit_state.UnitStatus | None:
    return unit_state.unit_status(run=_run)


def _config(args: argparse.Namespace) -> Config:
    """The file, its source, its [studio] table and the Config line with its notes, or the held failure."""
    notes: list[str] = []
    try:
        path, source = launch.find_studio_config(getattr(args, "config", None))
        table = launch.read_studio_config(path, notes)
    except ClusterConfigError as exc:
        return None, None, launch.read_studio_config(None), [], exc
    line = NO_CONFIG if path is None else f"Config: {path}" + (
        " (from the working directory)" if source == "working directory" else "")
    return path, source, table, [line, *notes], None


def _driver_line(settings: launch.StudioSettings) -> str:
    return f"Driver: {settings.driver} ({DRIVER_FROM[settings.driver_source]})"


def _view() -> View:
    """Ownership of the unit path, its settings lines when it is ours, and its status (None: not ours, no answer)."""
    ownership = unit.owned_unit_state()
    if ownership != "ours":
        return ownership, {}, None
    text = unit.read_owned_unit()
    return ownership, unit.unit_settings(text) if text else {}, _status()


def _running(view: View) -> bool:
    return view[0] == "ours" and view[2] is not None and unit_state.unit_active(view[2])


def _state_dir(installed: dict[str, str]) -> Path:
    return Path(installed["state"]) if installed.get("state") else launch.default_state_dir()


def _journal(status: unit_state.UnitStatus | None) -> list[str]:
    """START_FAILED and the unit's journal lines since its main process started."""
    since = 0.0
    if status is not None and status.start_monotonic_us:
        since = unit_state._realtime() - unit_state._monotonic() + status.start_monotonic_us / 1e6
    return [unit.START_FAILED, *unit_state.journal_tail(since, run=_run)]


def _token_now(state: Path) -> str | None:
    with contextlib.suppress(StudioError, OSError):
        found = sign_in.read_token_file(state / sign_in.OWNER_FILE)
        return found[0] if found else None
    return None


def _one_word(args: argparse.Namespace) -> int:
    """The one word: open a running Studio, else start one in this terminal."""
    config = path, source, table, lines, failure = _config(args)
    settings, usage = None, ""
    if failure is None:
        _say(*lines)
        try:
            settings = launch.resolve_settings(args, path, source, table)
        except ValueError as exc:
            usage = str(exc)
        if settings is not None and settings.driver_source != "default":
            _say(_driver_line(settings))
    view = ownership, _installed, status = _view()
    if _running(view):
        if args.driver is not None or args.state_dir is not None or args.allow_management:
            _refuse(launch.UNIT_RUNS)
        if failure is not None:
            _say(f"Note: {failure}")
        return _open(args, view=view, config=config)
    if ownership == "ours" and status is not None and status.active_state in ("failed", "inactive"):
        failed = status.active_state == "failed"
        _say(*(_journal(status) if failed else []), err=True)
        _say(INSTALLED_FAILED if failed else INSTALLED_INACTIVE)
    if failure is not None:
        _refuse(f"error: {failure}", code=2)
    if settings is None:
        _usage(usage)
    state = settings.state_dir
    if unit_state.lock_probe(state) == "held":
        if sign_in.read_sign_in_page(state / sign_in.SIGN_IN_FILE) is None:
            _refuse(launch.TERMINAL_STUDIO)
        return _open(args, view=view, config=config, state=state)
    if settings.driver_source == "default":
        outcome, _bound = unit_state.origin_preflight(state, settings.driver, True)
        # Only a readable binding equal to the default lets the default through.
        if outcome in ("no_origin", "unreadable"):
            _refuse(launch.NO_DRIVER_ORIGIN)
        if outcome == "ok":
            _say(_driver_line(settings))
    from ..studio import __main__ as companion

    return companion.main(launch.companion_argv(settings, no_open=args.no_open))


def _live(view: View, target: Path, reader: Any, absent: str) -> tuple[str, int]:
    """The file's content and write time, only while a companion serves it."""
    ownership, installed, status = view
    if _running(view):
        outcome = unit_state.wait_ready(target.parent, management=installed.get("management") == "yes",
                                        after_monotonic=None, previous_token=None, status_fn=_status)
        if outcome == "failed":
            _refuse(*_journal(status))
        found = reader(target) if outcome == "ready" else None
        return found or _refuse(unit.UNIT_NOT_READY, code=UNKNOWN)
    found, held = reader(target), unit_state.lock_probe(target.parent) == "held"
    if found and held:
        return found
    if ownership == "ours" and status is not None and status.active_state == "failed":
        _refuse(*_journal(status))
    _refuse(absent if held else sign_in.NOT_RUNNING)


def _open(args: argparse.Namespace, *, view: View | None = None, config: Config | None = None,
          state: Path | None = None, browser: bool = True) -> int:
    """``open``: the running Studio's sign-in block; a browser starts only on the local route."""
    view = view or _view()
    installed = view[1]
    state = state or _state_dir(installed)
    # Only a running unit's own line says management; a stopped unit says nothing about what runs now.
    management = _running(view) and installed.get("management") == "yes"
    target = state / (sign_in.OWNER_FILE if management else sign_in.SIGN_IN_FILE)
    found = _live(view, target, *((sign_in.read_token_file, sign_in.NO_TOKEN_FILE) if management
                                  else (sign_in.read_sign_in_page, sign_in.NO_SIGN_IN_PAGE)))
    no_open = bool(getattr(args, "no_open", False)) or not browser
    if management:
        port = installed.get("port", "")
        _say(f"Studio: http://127.0.0.1:{port}", f"Owner token file (mode 0600): {target}")
        if sign_in.terminal_stdout():
            _say(f"Owner token: {found[0]}", sign_in.MANAGEMENT_NO_LINK)
            if not no_open and sign_in.browser_route(os.environ) == "local":
                _say(sign_in.open_plain_address(f"http://127.0.0.1:{port}/", popen=_popen))
    else:
        # Without management: the page to a browser on the local route, or the link to a terminal only.
        url, route = found[0], "off" if no_open else sign_in.browser_route(os.environ)
        _say(f"Studio: {url.split('/#', 1)[0]}")
        if not sign_in.terminal_stdout():
            _say(f"Sign-in page (mode 0600): {target}")
        elif route == "local":
            status, handler = sign_in.open_in_browser(target, run=_run, popen=_popen)
            _say(*([] if status == sign_in.OPENED else sign_in.link_lines(url)), status, *filter(None, [handler]))
        else:
            parts = urlsplit(url)
            forward = sign_in.ssh_forward_hint(os.environ, parts.port or 0, target=parts.hostname or "127.0.0.1")
            hints = {"ssh": forward, "none": [sign_in.NO_DISPLAY]}
            _say(*sign_in.link_lines(url), *hints.get(route, []))
    if view[0] == "ours":
        _notes(args, installed, config)
    return 0


def _notes(args: argparse.Namespace, installed: dict[str, str], config: Config | None) -> None:
    """The drift note and the stale-code note; a failure to compute either prints nothing."""
    path, source, table, _seen, failure = config or _config(args)
    if failure is not None and config is None:
        _say(f"Note: {failure}")
    with contextlib.suppress(SystemExit, ValueError, OSError, StudioError):
        if failure is None:
            # What up would write now; management comes only from a typed flag, so it keeps the installed value.
            asked = argparse.Namespace(studio_action="up", up_allow_management=installed.get("management") == "yes")
            current = launch.resolve_settings(asked, path, source, table)
            rendered = unit.rendered_settings(current, unit_paths.unit_runtime(os.environ), installed.get("source", ""))
            fields = unit.drift_fields(installed, rendered)
            if fields:
                _say(f"{unit.DRIFT_NOTE} {', '.join(fields)}")
    if unit.source_changed(installed):
        _say(unit.STALE_CODE)


def _token(args: argparse.Namespace) -> int:
    """``token``: the token file's path, the token on a terminal with --value, or a restart for
    --rotate under the unit lock that waits for a different token."""
    if args.value and not sign_in.terminal_stdout():
        _refuse(sign_in.NO_PIPE)
    if not args.rotate:
        view = _view()
        target = _state_dir(view[1]) / sign_in.OWNER_FILE
        found = _live(view, target, sign_in.read_token_file, sign_in.NO_TOKEN_FILE)
        _say(f"Owner token file (mode 0600): {target}", *([f"Owner token: {found[0]}"] if args.value else []))
        return 0
    _rotatable()
    with unit_state.unit_lock():
        # Read again under the lock: an up or down that finished in between changed what runs.
        installed, status = _rotatable()
        with operations_lock.quiesced(_state_dir(installed), status):
            driver, origin = installed.get("driver-service"), installed.get("driver", "")
            before = unit_state.driver_unit_check(driver, origin, run=_run)[2] if driver else None
            _restart(_state_dir(installed), installed.get("management") == "yes")
        _say(ROTATED, *_also_started(driver, origin, before))
    return _open(args, browser=False)


def _rotatable() -> tuple[dict[str, str], unit_state.UnitStatus]:
    """The settings and status of a running unit of ours, or the --rotate refusal."""
    view = ownership, installed, status = _view()
    if ownership == "ours" and status is None:
        _refuse(LOST_ANSWER, code=UNKNOWN)
    if status is None or not _running(view):
        _refuse(unit.NO_ROTATE)
    return installed, status


def _restart(state: Path, management: bool) -> None:
    """Remember the token, reset-failed, restart, then wait for a new token."""
    previous, reset = _token_now(state), _systemctl("reset-failed", "--", unit.UNIT_NAME)
    # unit_state's clocks, which a test replaces, so the stamps and the readiness check share one clock.
    epoch, started = unit_state._realtime(), unit_state._monotonic()
    restarted = None if reset is None else _systemctl("restart", "--", unit.UNIT_NAME, timeout=120)
    if restarted is None:
        _refuse(LOST_ANSWER, code=UNKNOWN)
    if restarted.returncode != 0:
        _refuse(unit.START_FAILED, *restarted.stderr.strip().splitlines()[-1:], *unit_state.journal_tail(epoch, run=_run))
    outcome = unit_state.wait_ready(state, management=management, after_monotonic=started,
                                    previous_token=previous, status_fn=_status)
    if outcome == "failed":
        _refuse(unit.START_FAILED, *unit_state.journal_tail(epoch, run=_run))
    if outcome != "ready":
        _refuse(unit.NOT_READY, code=UNKNOWN)


def _also_started(driver: str | None, origin: str, before: str | None) -> list[str]:
    """The ``Also started`` line, only when the driver unit was not active before and is active now."""
    if not driver or before is None or before in ("active", "activating", "reloading"):
        return []
    now = unit_state.driver_unit_check(driver, origin, run=_run)[2]
    return [f"Also started {driver}: the Studio service wants it."] if now in ("active", "activating") else []


def _manager() -> None:
    """Never as root, and only with a user manager that answers."""
    if os.geteuid() == 0:
        _refuse(unit.RUN_AS_ROOT)
    answer = unit_state.user_manager(run=_run)
    if answer == "absent":
        _refuse(unit.NO_USER_MANAGER)
    if answer != "ok":
        _refuse(unit.NO_ANSWER, code=UNKNOWN)


def _owned(ownership: str) -> None:
    """Only an absent path or our own regular file passes; unknown is never read as absent."""
    if ownership == "symlink":
        _refuse(unit.UNIT_SYMLINK)
    if ownership not in ("absent", "ours"):
        _refuse(f"{unit.FOREIGN_UNIT} {unit.unit_path()}")


def _install(args: argparse.Namespace, hold: contextlib.ExitStack) -> int:
    """Install and start the unit under the unit lock; nothing is written before every check passes."""
    path, source, table, lines, failure = _config(args)
    if failure is not None:
        _refuse(f"error: {failure}", code=2)
    _say(*lines)
    if source == "working directory":
        _refuse(launch.WORKDIR_CONFIG)
    try:
        settings = launch.resolve_settings(args, path, source, table)
    except ValueError as exc:
        _usage(str(exc), "up")
    from_default = settings.driver_source == "default"
    if not from_default:
        _say(_driver_line(settings))
    runtime = unit_paths.unit_runtime(os.environ)
    target, state, driver = unit.unit_path(), settings.state_dir, settings.driver_service
    unit_paths.check_unit_directory(target.parent, create=False)
    _say(f"Interpreter: {runtime.interpreter}", f"Source: {runtime.source}")
    outcome, bound = unit_state.origin_preflight(state, settings.driver, from_default)
    if outcome == "no_origin" or (outcome == "unreadable" and from_default):
        _refuse(launch.NO_DRIVER_ORIGIN)
    if outcome == "differs":
        _refuse(OTHER_ORIGIN, *launch.mismatch_lines(state, bound or "", settings.driver))
    if outcome == "unreadable":
        _say(UNREADABLE_BINDING)
    elif from_default:
        _say(_driver_line(settings))
    home, user, runtime_dir = unit_state.specifier_values(os.environ)
    environment = unit_paths.preflight_environment(runtime, home=home, user=user, runtime_dir=runtime_dir)
    checked, detail = unit_state.check_preflight(runtime, environment, run=_run)
    if checked != "ok":
        _refuse(detail, code=UNKNOWN if checked == "timeout" else 1)
    before = None
    if driver:
        verdict, detail, before = unit_state.driver_unit_check(driver, settings.driver, run=_run)
        if verdict not in ("ok", "note"):
            _refuse(detail, code=UNKNOWN if verdict == "no_answer" else 1)
        _say(*([detail] if verdict == "note" else []))
    ownership = unit.owned_unit_state()
    _owned(ownership)
    previous = unit.read_owned_unit() if ownership == "ours" else None
    installed = unit.unit_settings(previous) if previous else {}
    dropins = unit.foreign_dropins(os.environ)
    if dropins:
        _refuse(f"{unit.FOREIGN_DROPIN} {', '.join(map(str, dropins))}")
    status = _status()
    if status is None:
        _refuse(unit.NO_ANSWER, code=UNKNOWN)
    running = ownership == "ours" and unit_state.unit_active(status)
    if running:
        hold.enter_context(operations_lock.quiesced(_state_dir(installed), status))
    elif unit_state.lock_probe(state) == "held":
        _refuse(unit.LOCK_HELD)
    if (not running or installed.get("port") != str(settings.port)) and unit_state.port_in_use(settings.port):
        _refuse(sign_in.PORT_IN_USE, f"See which program holds it: ss -ltnp 'sport = :{settings.port}'")
    digest = runtime_provenance.dgx_source_manifest_sha256(runtime.source / "dgx_monarch")
    text = unit.render_unit(settings, runtime, source_digest=digest)
    verb = "Unchanged" if unit.publish_unit(text) == "unchanged" else "Wrote"
    _say(f"{verb} {target}" + (f" (starts with {driver})" if driver else ""))
    props = None if _systemctl("daemon-reload") is None else unit_state.show_unit(
        ("LoadState", "FragmentPath", "DropInPaths", "NeedDaemonReload"), run=_run)
    if props is None:
        _refuse(LOST_ANSWER, code=UNKNOWN)
    refusal, vendor = unit.judge_readback(props, target, os.environ)
    _say(*vendor)
    if refusal is not None:
        unit.restore_unit(previous)
        _systemctl("daemon-reload")
        _refuse(f"{unit.SHADOWED} {refusal}")
    enabled = _systemctl("enable", "--", unit.UNIT_NAME)
    if enabled is None:
        _refuse(LOST_ANSWER, code=UNKNOWN)
    if enabled.returncode != 0:
        unit.restore_unit(previous)
        _systemctl("daemon-reload")
        _refuse(" ".join([ENABLE_FAILED, *enabled.stderr.strip().splitlines()[-1:]]))
    _restart(state, settings.allow_management)
    if unit_state.linger_state(run=_run) != "yes":
        _say(unit.LINGER_NOTE)
    _say(f"Started {unit.UNIT_NAME} on http://127.0.0.1:{settings.port}.",
         *_also_started(driver, settings.driver, before))
    token_file, page, terminal = state / sign_in.OWNER_FILE, None, sign_in.terminal_stdout()
    if settings.allow_management:
        _say(f"Owner token file (mode 0600): {token_file}", *([f"Owner token: {_token_now(state)}"] if terminal else []),
             sign_in.MANAGEMENT_NO_LINK, unit.MANAGEMENT_WARNING)
    else:
        with contextlib.suppress(StudioError, OSError):
            page = sign_in.read_sign_in_page(state / sign_in.SIGN_IN_FILE) if terminal else None
        _say(*(sign_in.link_lines(page[0]) if page else [f"Owner token file (mode 0600): {token_file}"]), OPEN_HINT)
    if installed.get("management") == "yes" and not settings.allow_management:
        _say(unit.MANAGEMENT_REMOVED)
    return 0


def _remove(hold: contextlib.ExitStack) -> int:
    """Under the unit lock: stop and disable our unit, then remove it once it reads stopped; the state stays."""
    ownership = unit.owned_unit_state()
    if ownership == "absent":
        _say(unit.NO_UNIT)
        return 0
    _owned(ownership)
    installed = unit.unit_settings(unit.read_owned_unit() or "")
    status = _status()
    if status is None:
        _refuse(LOST_ANSWER, code=UNKNOWN)
    if unit_state.unit_active(status):
        hold.enter_context(operations_lock.quiesced(_state_dir(installed), status))
    stopped = _systemctl("disable", "--now", "--", unit.UNIT_NAME, timeout=120)
    after = None if stopped is None else _status()
    if stopped is None or after is None:
        _refuse(LOST_ANSWER, code=UNKNOWN)
    if stopped.returncode != 0 or unit_state.unit_active(after):
        _refuse(f"{unit.NOT_STOPPED} ActiveState={after.active_state}", *stopped.stderr.strip().splitlines()[-1:])
    unit.remove_unit()  # checks ownership again; a foreign file is never removed
    props = None if _systemctl("daemon-reload") is None else unit_state.show_unit(
        ("LoadState", "ActiveState"), run=_run)
    if props is None:
        _refuse(LOST_ANSWER, code=UNKNOWN)
    problem = unit.judge_removal(props)
    if problem is not None:
        _refuse(f"{unit.STILL_PRESENT} {problem}")
    state = _state_dir(installed)
    # A stop that ended in SIGKILL can leave both files; their token died with the process.
    with contextlib.suppress(SystemExit, OSError):
        unit_state.remove_leftovers(state)
    dropins = unit.foreign_dropins(os.environ)
    if dropins:
        _say(LEFT_DROPINS.format(paths=", ".join(map(str, dropins))))
    _say(REMOVED.format(state=state))
    return 0


def _tunnel(args: argparse.Namespace) -> int:
    """Hand the typed host, port and config to the tunnel helper."""
    try:
        typed = getattr(args, "config", None)
        return tunnel.run_tunnel(args.host, args.port, args.run, Path(typed) if typed else None,
                                 environ=os.environ, terminal=sign_in.terminal_stdout(),
                                 stdout=sys.stdout, stderr=sys.stderr)
    except ValueError as exc:
        _usage(str(exc), "tunnel")
