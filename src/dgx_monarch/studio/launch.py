"""Settings for dgxm studio: which cluster.toml, its [studio] table, and the companion argv.

Nothing here probes for a driver or opens a socket. The module imports no
aiohttp, so every dgxm studio action works without the studio extra.
"""
from __future__ import annotations

import argparse
import errno
import os
import shlex
import stat
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from ..config import default_config_paths, find_config_path, load_cluster_config
from ..config_schema import (
    DEFAULT_STUDIO_DRIVER,
    DEFAULT_STUDIO_PORT,
    ClusterConfigError,
    StudioTable,
    validate_driver_origin,
    validate_root_keys,
    validate_studio_table,
)
from .driver_service import _others_can_write

UNIT_RUNS = ("The always-on Studio is running; its settings come from its unit. Run dgxm studio down to use "
             "these flags here, or change cluster.toml and run dgxm studio up.")
TERMINAL_STUDIO = ("Studio already runs in another terminal on this account and printed its sign-in link or "
                   "token there. Stop it there with Ctrl+C, or use dgxm studio up for an always-on Studio.")
NO_DRIVER_ORIGIN = ("No driver origin is set. Add driver to the [studio] table of cluster.toml, for example "
                    "driver = \"http://127.0.0.1:8188\", or pass --driver. Studio keeps its records for the "
                    "first driver address it uses.")
CONFIG_NOT_PRIVATE = ("Studio reads cluster.toml only when it is a regular file you own that no other account "
                      "can write; fix its owner or mode, for example chmod 600.")
NO_HOSTS_NOTE = "Note: this cluster.toml has no [[hosts]], so cluster actions stay unavailable in Studio."
WORKDIR_CONFIG = ("dgxm studio up does not write a cluster.toml found in the working directory into a service; "
                  "pass --config with its path, or keep it at ~/.config/dgx-monarch/cluster.toml.")
# Filled at print time; TROUBLESHOOTING describes the two mismatch lines in prose.
MOVE_STATE = ("To use the new driver with fresh records, move the old ones aside, then start Studio again: "
              "mv {state} {state}-{suffix}")
BOUND_ELSEWHERE = "This state directory is bound to {bound}; this launch asked for {asked}."
# The dgxm-studio --check phrase for a missing extra; dgxm studio up refuses a service whose check prints it.
EXTRA_MISSING = "studio extra not installed"
XDG_STATE_RELATIVE = "XDG_STATE_HOME must be an absolute path; unset it or pass an absolute --state-dir."
STATE_DIR_RELATIVE = "--state-dir must be an absolute path"

CONFIG_SOURCES = ("--config", "$DGXM_CLUSTER_TOML", "working directory", "user config")
DRIVER_SOURCES = ("--driver", "[studio] driver", "default")


@dataclass(frozen=True)
class StudioSettings:
    """Everything dgxm studio and dgxm studio up hand to the companion; the bind host is always 127.0.0.1."""

    port: int
    driver: str               # verbatim, canonical
    driver_source: str        # one of DRIVER_SOURCES
    state_dir: Path           # absolute
    config: Path | None       # absolute
    config_source: str | None  # one of CONFIG_SOURCES
    driver_service: str | None
    allow_management: bool


def _defaults() -> StudioTable:
    return StudioTable(driver=None, port=DEFAULT_STUDIO_PORT, driver_service=None, driver_command=None)


def find_studio_config(explicit: str | None) -> tuple[Path | None, str | None]:
    """The cluster.toml every dgxm command would read, made absolute, and where it was found.

    The order is config.find_config_path's: an explicit path (a missing one
    raises ClusterConfigError), then $DGXM_CLUSTER_TOML, ./cluster.toml and
    ~/.config/dgx-monarch/cluster.toml. Returns (None, None) when none exists.
    """
    if explicit:
        found = find_config_path(explicit)
        return (None, None) if found is None else (found.expanduser().absolute(), "--config")
    environment = os.environ.get("DGXM_CLUSTER_TOML")
    labels = (["$DGXM_CLUSTER_TOML"] if environment else []) + ["working directory", "user config"]
    candidates = default_config_paths()
    if len(candidates) != len(labels):
        # The search order changed under this module; label nothing rather than mislabel.
        found = find_config_path(None)
        return (None, None) if found is None else (found.expanduser().absolute(), None)
    for candidate, label in zip(candidates, labels, strict=True):
        if candidate.is_file():
            return candidate.expanduser().absolute(), label
    return None, None


def _read_private_config(path: Path) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:  # O_NOFOLLOW met a symlink
            raise ClusterConfigError(f"{path}: {CONFIG_NOT_PRIVATE}") from exc
        raise ClusterConfigError(f"{path}: cannot read the file: {exc.strerror or exc}") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or _others_can_write(info):
            raise ClusterConfigError(f"{path}: {CONFIG_NOT_PRIVATE}")
        chunks = []
        while chunk := os.read(descriptor, 1 << 16):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def read_studio_config(config: Path | None, notes: list[str] | None = None) -> StudioTable:
    """Read the [studio] table of one private cluster.toml, with defaults filled in.

    No file, or a file without the table, gives the defaults. The file must be
    a regular file this user owns that no other account can write
    (CONFIG_NOT_PRIVATE). The root keys and the table are checked here; a file
    with [[hosts]] must also load through load_cluster_config. A file with no
    [[hosts]] is a local-mode file: NO_HOSTS_NOTE is appended to notes and
    reading goes on. Every refusal is a ClusterConfigError naming the file.
    """
    if config is None:
        return _defaults()
    path = Path(config)
    data = _read_private_config(path)
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ClusterConfigError(f"{path}: invalid TOML: {exc}") from exc
    validate_root_keys(raw, path)
    table = validate_studio_table(raw["studio"], path) if "studio" in raw else _defaults()
    if raw.get("hosts"):
        load_cluster_config(path)
    elif notes is not None:
        notes.append(NO_HOSTS_NOTE)
    return table


def default_state_dir() -> Path:
    """$XDG_STATE_HOME/dgx-monarch/studio, or ~/.local/state/dgx-monarch/studio.

    The operator-receipt rule: an empty XDG_STATE_HOME counts as unset, and a
    relative one exits with XDG_STATE_RELATIVE.
    """
    from ..cli.operator_receipt import _state_root

    try:
        return _state_root(None) / "dgx-monarch" / "studio"
    except ValueError as exc:
        raise SystemExit(XDG_STATE_RELATIVE) from exc


def resolve_settings(args: argparse.Namespace, config: Path | None, source: str | None,
                     table: StudioTable) -> StudioSettings:
    """Flag, then [studio], then default, for each value.

    Reads driver, state_dir, allow_management and up_allow_management from args
    with getattr defaults. Management comes from up_allow_management when
    args.studio_action is "up" and from allow_management otherwise, so a flag
    is honored only where the parser accepts it. A typed --driver passes the
    canonical, listen-address and zone rules but may name another computer.
    Raises ValueError for a usage error (a bad --driver, a relative
    --state-dir); the CLI turns it into parser.error.
    """
    typed = getattr(args, "driver", None)
    if typed is not None:
        driver, driver_source = validate_driver_origin(typed, local_only=False), "--driver"
    elif table["driver"] is not None:
        driver, driver_source = table["driver"], "[studio] driver"
    else:
        driver, driver_source = DEFAULT_STUDIO_DRIVER, "default"
    typed_state = getattr(args, "state_dir", None)
    if typed_state is not None:
        state_dir = Path(typed_state).expanduser()
        if not state_dir.is_absolute():
            raise ValueError(STATE_DIR_RELATIVE)
    else:
        state_dir = default_state_dir()
    if getattr(args, "studio_action", None) == "up":
        management = bool(getattr(args, "up_allow_management", False))
    else:
        management = bool(getattr(args, "allow_management", False))
    return StudioSettings(
        port=table["port"], driver=driver, driver_source=driver_source, state_dir=state_dir,
        config=None if config is None else Path(config).expanduser().absolute(),
        config_source=source if config is not None else None,
        driver_service=table["driver_service"], allow_management=management)


def companion_argv(settings: StudioSettings, *, no_open: bool) -> list[str]:
    """The dgxm-studio arguments for these settings, in a fixed order."""
    return ["--host", "127.0.0.1", "--port", str(settings.port), "--driver", settings.driver,
            "--state-dir", str(settings.state_dir),
            *(["--config", str(settings.config)] if settings.config else []),
            *(["--driver-service", settings.driver_service] if settings.driver_service else []),
            *(["--allow-management"] if settings.allow_management else []),
            *(["--no-open"] if no_open else [])]


def _origin_port(origin: str) -> int:
    parsed = urlsplit(origin)
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None:
        return port
    return 443 if parsed.scheme == "https" else 80


def _free(path: str) -> bool:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def move_state_line(state: Path, bound_origin: str) -> str:
    """MOVE_STATE filled for this state directory, or "" when no free target exists.

    The target is the directory name plus the bound origin's port (80 or 443
    when it has none). mv into an existing directory would move the state
    inside it, so an existing target (checked with lstat) gains a UTC time,
    and the line is offered only for a target that does not exist.
    """
    port = _origin_port(bound_origin)
    for suffix in (str(port), f"{port}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"):
        if _free(f"{state}-{suffix}"):
            return MOVE_STATE.format(state=shlex.quote(str(state)), suffix=suffix)
    return ""


def mismatch_lines(state: Path, bound: str, asked: str) -> list[str]:
    """The lines printed after a driver_identity_mismatch: both origins, then the mv line when one is free."""
    lines = [BOUND_ELSEWHERE.format(bound=bound, asked=asked)]
    move = move_state_line(state, bound)
    if move:
        lines.append(move)
    return lines
