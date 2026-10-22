"""The always-on Studio user unit: its text, its owner check and its one file.

dgxm writes exactly one file, ``dgxm-studio.service`` in the systemd user
directory (``$XDG_CONFIG_HOME/systemd/user`` when that is set, else
``~/.config/systemd/user``, where the user manager looks for it), and never a
drop-in. The settings comment lines at the top of the unit are rendered from
the same settings and runtime as ``ExecStart``, so ``open``, ``token``,
``tunnel`` and the drift note read the service's settings without parsing a
command line. Standard library only, so the aiohttp-free CLI can import it.
"""
from __future__ import annotations

import errno
import os
import pwd
import re
import secrets
import shlex
import stat
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from ..cli.systemd_unit import quote
from .unit_paths import (
    SPECIFIER_WORDS,
    UNIT_DIR_UNSAFE,
    UnitRuntime,
    check_path_chars,
    check_unit_directory,
    printable,
    service_command,
    unit_environment,
)

if TYPE_CHECKING:
    from .launch import StudioSettings

UNIT_NAME = "dgxm-studio.service"
UNIT_MARKER = "# dgxm-managed-studio-unit-v1"
# systemd applies dgxm-.service.d to every unit whose name starts with dgxm-.
DROPIN_NAMES = ("dgxm-studio.service.d", "dgxm-.service.d")
SETTING_PREFIX = "# dgxm-studio-"
SETTING_FIELDS = ("state", "port", "driver", "config", "driver-service", "management", "interpreter", "pythonpath", "source")
DOCUMENTATION = "https://github.com/Deen-Media/dgx-monarch/blob/main/docs/STUDIO.md#keep-studio-running"

RUN_AS_ROOT = "Run dgxm studio up as the account that runs ComfyUI, not as root."
NO_USER_MANAGER = "No systemd user manager answers here; run dgxm studio in a terminal instead."
FOREIGN_UNIT = "Another file already uses the Studio unit name; dgxm studio up writes only a unit it wrote itself."
UNIT_SYMLINK = "The Studio unit path is a symbolic link; dgxm never writes through a link."
FOREIGN_DROPIN = "A drop-in changes the Studio unit; remove it, then run dgxm studio up again."
SHADOWED = "systemd loads the Studio unit from another file or with drop-ins; remove them, then run dgxm studio up again."
DRIVER_UNIT_UNKNOWN = "[studio] driver_service names a unit that the user manager does not know; install that unit first or remove the key."
DRIVER_UNIT_ORIGIN = "[studio] driver_service listens on another address or port than [studio] driver; make them match."
DRIVER_UNIT_NOTE = "Note: Studio cannot read the address from this driver unit's command; check that it serves [studio] driver. Studio's own driver controls and Studio update refuse such a unit."
LOCK_HELD = "A companion already runs on this state directory; stop it before installing the always-on Studio."
START_FAILED = "Studio did not start; its journal lines follow."
NOT_READY = "Studio has not reported ready after 30 seconds; check systemctl --user status dgxm-studio.service."
UNIT_NOT_READY = "dgxm-studio.service runs but has not written its sign-in page or token file yet; see journalctl --user -u dgxm-studio.service -n 20."
STILL_PRESENT = "The Studio unit is still loaded or active after removal; check systemctl --user status dgxm-studio.service."
NO_UNIT = "No always-on Studio is installed."
NO_ROTATE = "No always-on Studio to rotate; a Studio started with dgxm studio makes a new token each time it starts."
DRIFT_NOTE = "Note: the installed service differs from what dgxm studio up would write now; run dgxm studio up from the environment and config you want it to use. Fields that differ:"
STALE_CODE = "Note: the Studio code on disk changed after dgxm studio up started the service; run dgxm studio up to restart Studio on it."
MANAGEMENT_WARNING = "The service runs with --allow-management. Its token is code execution as the worker account on every host; the token file is a standing credential until the service restarts, and the page signs in only through its form."
MANAGEMENT_REMOVED = "Removed --allow-management from the service. Run dgxm studio up --allow-management to keep management."
LINGER_NOTE = "Linger is off for this user, so the service runs only while you are logged in: it starts at your first login and stops when your last session ends. To keep it running from boot, run: sudo loginctl enable-linger $USER"
UNINSTALL_STUDIO_FIRST = "uninstall refused: dgxm-studio.service is still installed; run dgxm studio down first, then dgxm uninstall."
# Filled at print time (design section 6.4 steps 2, 8 and 14).
NO_ANSWER = "The user manager did not answer; nothing was installed."
DRIVER_UNIT_STATE = "[studio] driver_service names {unit}, whose LoadState is {state}."
VENDOR_DROPIN = "Note: systemd also applies a drop-in that this system ships for every user service:"

# Drop-ins under these directories come with the distribution, such as the
# service.d/10-timeout-abort.conf some ship for every user service.
VENDOR_DIRECTORIES = (PurePosixPath("/usr/lib/systemd/user"), PurePosixPath("/usr/local/lib/systemd/user"))
_UNIT_LIMIT = 65536

_UNIT_TEMPLATE = """{marker}
{settings}[Unit]
Description=DGX Monarch Studio companion
Documentation={documentation}
{coupling}StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
UnsetEnvironment={unset}
WorkingDirectory=/
ExecStart={exec_start}
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


def _home(environ: Mapping[str, str]) -> Path:
    home = environ.get("HOME", "")
    return Path(home) if os.path.isabs(home) else Path(pwd.getpwuid(os.geteuid()).pw_dir)


def _base(environ: Mapping[str, str], name: str, default: Path) -> Path:
    """An XDG base directory; a relative value is ignored, as the specification says."""
    value = environ.get(name, "")
    return Path(value) if os.path.isabs(value) else default


def unit_path() -> Path:
    """The one file dgxm writes: in the user manager's own configuration directory."""
    config = _base(os.environ, "XDG_CONFIG_HOME", _home(os.environ) / ".config")
    return config / "systemd" / "user" / UNIT_NAME


def search_directories(environ: Mapping[str, str]) -> list[Path]:
    """The user-writable unit directories the drop-in scan reads, as the Worker scan does.

    ``~/.config/systemd/user``, ``~/.local/share/systemd/user`` and
    ``$XDG_RUNTIME_DIR/systemd/user`` when it is set; an XDG base directory
    that differs from its default adds its own entry, so neither shell
    environment can hide a drop-in.
    """
    home = _home(environ)
    candidates = [
        _base(environ, "XDG_CONFIG_HOME", home / ".config") / "systemd" / "user",
        home / ".config" / "systemd" / "user",
        _base(environ, "XDG_DATA_HOME", home / ".local" / "share") / "systemd" / "user",
        home / ".local" / "share" / "systemd" / "user",
    ]
    runtime = environ.get("XDG_RUNTIME_DIR", "")
    if os.path.isabs(runtime):
        candidates.append(Path(runtime) / "systemd" / "user")
    return list(dict.fromkeys(candidates))


def rendered_settings(settings: StudioSettings, runtime: UnitRuntime, source_digest: str) -> dict[str, str]:
    """The settings comment fields, in unit order; an absent value renders empty."""
    return {
        "state": str(settings.state_dir),
        "port": str(settings.port),
        "driver": settings.driver,
        "config": "" if settings.config is None else str(settings.config),
        "driver-service": settings.driver_service or "",
        "management": "yes" if settings.allow_management else "no",
        "interpreter": str(runtime.interpreter),
        "pythonpath": runtime.pythonpath,
        "source": source_digest,
    }


def _word(value: str) -> str:
    if value in SPECIFIER_WORDS:
        return f'"{value}"'
    check_path_chars(value)
    return quote(value)


def render_unit(settings: StudioSettings, runtime: UnitRuntime, *, source_digest: str) -> str:
    """The unit text; a value systemd would rewrite is refused, never escaped."""
    from ..cli.lifecycle_hardened import UNSET_ENVIRONMENT
    from ..config_schema import DRIVER_UNIT_PATTERN
    from .launch import companion_argv

    if re.fullmatch(r"[0-9a-f]{64}", source_digest) is None:
        raise ValueError("the source digest must be 64 lowercase hexadecimal characters")
    for entry in (runtime.source, runtime.dependencies):
        check_path_chars(str(entry), pythonpath_entry=True)
    service = settings.driver_service
    if service is not None and DRIVER_UNIT_PATTERN.fullmatch(service) is None:
        raise ValueError("driver_service must be a validated unit name")
    fields = rendered_settings(settings, runtime, source_digest)
    for value in fields.values():
        check_path_chars(value)
    words = ["/usr/bin/env", "-i", *unit_environment(runtime), *service_command(runtime),
             *companion_argv(settings, no_open=True)]
    return _UNIT_TEMPLATE.format(
        marker=UNIT_MARKER,
        settings="".join(f"{SETTING_PREFIX}{name}={value}\n" for name, value in fields.items()),
        documentation=DOCUMENTATION,
        coupling=f"Wants={service}\nAfter={service}\n" if service else "",
        unset=UNSET_ENVIRONMENT,
        exec_start=" ".join(_word(word) for word in words),
    )


def unit_settings(text: str) -> dict[str, str]:
    """The ``# dgxm-studio-<field>=`` lines of a unit; the first line for a field wins."""
    found: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith(SETTING_PREFIX):
            continue
        name, separator, value = line[len(SETTING_PREFIX):].partition("=")
        if separator and name in SETTING_FIELDS and name not in found:
            found[name] = value
    return found


UnitState = Literal["absent", "ours", "symlink", "foreign", "unknown"]


def _read_unit(path: Path) -> tuple[UnitState, str | None]:
    """Ownership by the first line, read on a descriptor that cannot follow a link."""
    try:
        info = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return "absent", None
    except OSError:
        return "unknown", None
    if stat.S_ISLNK(info.st_mode):
        return "symlink", None
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_size > _UNIT_LIMIT:
        return "foreign", None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return "absent", None
    except OSError as exc:
        return ("symlink" if exc.errno == errno.ELOOP else "unknown"), None
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino) or not stat.S_ISREG(opened.st_mode):
            return "unknown", None
        chunks: list[bytes] = []
        size = 0
        while size <= _UNIT_LIMIT:
            chunk = os.read(descriptor, 16384)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
    except OSError:
        return "unknown", None
    finally:
        os.close(descriptor)
    if size > _UNIT_LIMIT:
        return "foreign", None
    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError:
        return "foreign", None
    if text.split("\n", 1)[0] != UNIT_MARKER:
        return "foreign", None
    return "ours", text


def owned_unit_state(path: Path | None = None) -> UnitState:
    """``ours`` only for a regular file of this user whose first line is the Studio marker."""
    return _read_unit(unit_path() if path is None else path)[0]


def read_owned_unit(path: Path | None = None) -> str | None:
    """The unit text when the unit is ours, else ``None``."""
    return _read_unit(unit_path() if path is None else path)[1]


def foreign_dropins(environ: Mapping[str, str]) -> list[Path]:
    """Every entry in either drop-in directory of every search directory; dgxm writes none."""
    found: list[Path] = []
    for base in search_directories(environ):
        for name in DROPIN_NAMES:
            directory = base / name
            try:
                info = os.lstat(directory)
            except (FileNotFoundError, NotADirectoryError):
                continue
            except OSError:
                found.append(directory)
                continue
            if not stat.S_ISDIR(info.st_mode):
                found.append(directory)
                continue
            try:
                entries = sorted(os.listdir(directory))
            except OSError:
                found.append(directory)
                continue
            found.extend(directory / entry for entry in entries)
    return found


def _write(path: Path, text: str) -> None:
    """Replace the unit atomically; a rename replaces a racing link without touching its target."""
    check_unit_directory(path.parent, create=True)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        info = os.fstat(directory)
        if info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise SystemExit(UNIT_DIR_UNSAFE)
        temporary = f".{UNIT_NAME}.{secrets.token_hex(8)}.tmp"
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o600, dir_fd=directory)
        try:
            try:
                os.fchmod(descriptor, 0o600)
                data = memoryview(text.encode("utf-8"))
                while data:
                    data = data[os.write(descriptor, data):]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=directory)
            raise
        with suppress(OSError):
            os.fsync(directory)
    finally:
        os.close(directory)


def _refuse_unless_ours_or_absent(path: Path) -> UnitState:
    state, _ = _read_unit(path)
    if state == "symlink":
        raise SystemExit(UNIT_SYMLINK)
    if state not in {"absent", "ours"}:
        raise SystemExit(f"{FOREIGN_UNIT} {path}")
    return state


def publish_unit(text: str) -> Literal["wrote", "unchanged"]:
    """Write the unit unless the owned file already holds these bytes; never over a file dgxm did not write."""
    path = unit_path()
    _refuse_unless_ours_or_absent(path)
    if read_owned_unit(path) == text:
        return "unchanged"
    _write(path, text)
    return "wrote"


def restore_unit(previous: str | None) -> None:
    """Put back the bytes read before a publish, or remove the unit this run created."""
    path = unit_path()
    if previous is not None:
        _write(path, previous)
        return
    if owned_unit_state(path) == "ours":
        with suppress(FileNotFoundError):
            os.unlink(path)


def remove_unit() -> None:
    """Remove the owned unit file after ``disable --now``; a foreign file is never removed."""
    path = unit_path()
    if _refuse_unless_ours_or_absent(path) == "ours":
        with suppress(FileNotFoundError):
            os.unlink(path)


def _user_roots(environ: Mapping[str, str]) -> list[PurePosixPath]:
    """Directories this account can write that systemd reads units and drop-ins from."""
    home = _home(environ)
    roots = [
        _base(environ, "XDG_CONFIG_HOME", home / ".config") / "systemd",
        home / ".config" / "systemd",
        _base(environ, "XDG_DATA_HOME", home / ".local" / "share") / "systemd",
        home / ".local" / "share" / "systemd",
    ]
    runtime = environ.get("XDG_RUNTIME_DIR", "")
    if os.path.isabs(runtime):
        roots.append(Path(runtime) / "systemd")  # user.control, transient and the generator directories
    return [PurePosixPath(root) for root in dict.fromkeys(roots)]


def _dropin_refused(entry: str, environ: Mapping[str, str]) -> bool:
    path = PurePosixPath(entry)
    if not path.is_absolute() or ".." in path.parts or path.parent.name in DROPIN_NAMES:
        return True
    if any(path.is_relative_to(root) for root in _user_roots(environ)):
        return True
    if path.is_relative_to("/etc/systemd/user"):
        return True
    # Anything outside the vendor directories is refused too: unknown is never read as safe.
    return not any(path.is_relative_to(root) for root in VENDOR_DIRECTORIES)


def judge_readback(props: Mapping[str, str], expected: Path,
                   environ: Mapping[str, str]) -> tuple[str | None, list[str]]:
    """What the user manager loaded after ``daemon-reload``: a refusal detail or ``None``, and vendor notes.

    ``props`` holds ``LoadState``, ``FragmentPath``, ``DropInPaths`` and
    ``NeedDaemonReload``. A drop-in in either Studio drop-in directory, under
    a directory this account can write, or under ``/etc/systemd/user``
    refuses; one under the vendor directories prints a note and passes.
    """
    problems: list[str] = []
    if props.get("LoadState") != "loaded":
        problems.append(f"LoadState={props.get('LoadState', '')}")
    if props.get("FragmentPath") != str(expected):
        problems.append(f"FragmentPath={props.get('FragmentPath', '')}")
    if props.get("NeedDaemonReload") != "no":
        problems.append(f"NeedDaemonReload={props.get('NeedDaemonReload', '')}")
    notes: list[str] = []
    raw = props.get("DropInPaths", "")
    try:
        entries = shlex.split(raw)
    except ValueError:
        entries = []
        problems.append(f"DropInPaths={raw}")
    refused = [entry for entry in entries if _dropin_refused(entry, environ)]
    if refused:
        problems.append(f"DropInPaths={' '.join(refused)}")
    notes.extend(f"{VENDOR_DROPIN} {entry}" for entry in entries if entry not in refused)
    return ("; ".join(printable(problem) for problem in problems) if problems else None), notes


def judge_removal(props: Mapping[str, str], path: Path | None = None) -> str | None:
    """After removal the file is gone, ``LoadState`` reads not-found and the unit is inactive or failed."""
    path = unit_path() if path is None else path
    problems: list[str] = []
    if os.path.lexists(path):
        problems.append(str(path))
    if props.get("LoadState") != "not-found":
        problems.append(f"LoadState={props.get('LoadState', '')}")
    if props.get("ActiveState") not in {"inactive", "failed"}:
        problems.append(f"ActiveState={props.get('ActiveState', '')}")
    return "; ".join(problems) if problems else None


def drift_fields(installed: Mapping[str, str], current: Mapping[str, str]) -> list[str]:
    """The fields, other than ``source``, where the installed unit differs from what ``up`` would write now."""
    return [name for name in SETTING_FIELDS if name != "source" and installed.get(name) != current.get(name)]


def source_changed(installed: Mapping[str, str]) -> bool | None:
    """Whether the code under the unit's first import entry changed since ``up``; ``None`` when unknown."""
    from ..runtime_provenance import dgx_source_manifest_sha256

    expected = installed.get("source", "")
    first = installed.get("pythonpath", "").split(":", 1)[0]
    if not expected or not os.path.isabs(first):
        return None
    try:
        current = dgx_source_manifest_sha256(Path(first) / "dgx_monarch")
    except (OSError, RuntimeError, ValueError):
        return None
    return current != expected
