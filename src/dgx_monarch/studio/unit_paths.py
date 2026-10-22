"""Where the always-on Studio unit imports its code from, and who may change it.

The interpreter, the source directory and the dependency directory come from
the process that runs ``dgxm studio up``. Every path the unit carries is
checked before anything is written: no ``$`` (systemd substitutes ``${NAME}``
inside an ``ExecStart`` word), no control character, no ``:`` inside a
``PYTHONPATH`` entry, and no directory on the way that another account can
write. One list of environment words feeds both the unit text and the
preflight run of the service command, so the two cannot drift. Standard
library only, so the aiohttp-free CLI can import it.
"""
from __future__ import annotations

import os
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

from ..runtime_provenance import SOURCE_ONLY_PYCACHE_PREFIX
from .driver_service import _others_can_write

PYTHONPATH_SET = "PYTHONPATH is set in this shell, so the service would import whatever it names; unset it, then run dgxm studio up again."
UNIT_PATH_OWNER = "The service would run code from a path that another account can change; fix the owner or mode of:"
UNIT_DIR_UNSAFE = "~/.config/systemd/user must be a directory you own that no other account can write."
UNIT_PATH_CHARS = "The service cannot hold a path that contains '$', a newline, or a ':' inside a PYTHONPATH entry:"
# Filled at refusal time; the first is followed by the companion's install hint.
EXTRA_NOT_INSTALLED = "The studio extra is not installed for {interpreter}."
SPLIT_DEPENDENCIES = ("Studio's dependencies must be in one site-packages directory: "
                      "aiohttp in {aiohttp}, Pillow in {pillow}.")

# The only unit words that keep a systemd specifier; every other word is quoted.
SPECIFIER_WORDS = frozenset({"HOME=%h", "USER=%u", "LOGNAME=%u", "XDG_RUNTIME_DIR=%t"})

# The owner walk reads metadata through this name, so a test can describe the
# parents of a temporary tree (which sit under a world-writable /tmp).
_stat = os.stat


@dataclass(frozen=True)
class UnitRuntime:
    """The interpreter and the two import directories the unit pins."""

    interpreter: Path
    source: Path
    dependencies: Path

    @property
    def pythonpath(self) -> str:
        """One entry for a non-editable install, where both directories are the same."""
        if self.source == self.dependencies:
            return str(self.source)
        return f"{self.source}:{self.dependencies}"


def printable(text: str) -> str:
    """One line of text, control characters shown as escapes."""
    return "".join(char if 32 <= ord(char) != 127 else f"\\x{ord(char):02x}" for char in text)


def check_path_chars(text: str, *, pythonpath_entry: bool = False) -> None:
    """Refuse a value the unit cannot carry verbatim; never escape a ``$``."""
    if "$" in text or any(ord(char) < 32 or ord(char) == 127 for char in text) or (pythonpath_entry and ":" in text):
        raise SystemExit(f"{UNIT_PATH_CHARS} {printable(text)}")


def _package_directory(name: str) -> Path | None:
    """The directory that holds an importable package, found without importing it."""
    try:
        spec = find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    locations = list(spec.submodule_search_locations or [])
    if locations:
        return Path(locations[0]).absolute().parent
    return Path(spec.origin).absolute().parent if spec.origin else None


def _owner_problem(path: Path) -> Path | None:
    """The first entry from ``/`` down to ``path`` that another account can change."""
    uid = os.geteuid()
    for entry in [*reversed(path.parents), path]:
        try:
            info = _stat(entry)
        except OSError:
            return entry
        if info.st_uid not in {0, uid} or _others_can_write(info):
            return entry
    return None


def _owner_paths(runtime: UnitRuntime) -> list[Path]:
    """Each path as the unit names it and as it resolves, so a link cannot hide a parent."""
    paths: list[Path] = []
    for path in (runtime.interpreter, runtime.source, runtime.dependencies):
        paths.append(path)
        try:
            paths.append(path.resolve())
        except (OSError, RuntimeError):
            raise SystemExit(f"{UNIT_PATH_OWNER} {printable(str(path))}") from None
    return list(dict.fromkeys(paths))


def check_runtime(runtime: UnitRuntime) -> None:
    """The character rule for every path, then the owner rule for each and its parents."""
    check_path_chars(str(runtime.interpreter))
    for entry in (runtime.source, runtime.dependencies):
        check_path_chars(str(entry), pythonpath_entry=True)
    for path in _owner_paths(runtime):
        problem = _owner_problem(path)
        if problem is not None:
            raise SystemExit(f"{UNIT_PATH_OWNER} {printable(str(problem))}")


def unit_runtime(environ: Mapping[str, str] = os.environ) -> UnitRuntime:
    """The runtime of this ``dgxm`` process, refused when the unit could not hold it safely."""
    if environ.get("PYTHONPATH"):
        raise SystemExit(PYTHONPATH_SET)
    import dgx_monarch

    # Unresolved, so a virtual environment's python link keeps its pyvenv.cfg meaning.
    interpreter = Path(sys.executable).absolute()
    source = Path(dgx_monarch.__file__).resolve().parent.parent
    aiohttp, pillow = _package_directory("aiohttp"), _package_directory("PIL")
    if aiohttp is None or pillow is None:
        from .__main__ import INSTALL_HINT

        raise SystemExit(f"{EXTRA_NOT_INSTALLED.format(interpreter=interpreter)} {INSTALL_HINT}")
    if aiohttp != pillow:
        raise SystemExit(SPLIT_DEPENDENCIES.format(aiohttp=aiohttp, pillow=pillow))
    runtime = UnitRuntime(interpreter=interpreter, source=source, dependencies=aiohttp)
    check_runtime(runtime)
    return runtime


def make_private_directories(path: Path) -> None:
    """Create each missing directory down to ``path`` with mode 0700."""
    missing: list[Path] = []
    probe = path
    while not os.path.lexists(probe) and probe != probe.parent:
        missing.append(probe)
        probe = probe.parent
    for entry in reversed(missing):
        try:
            os.mkdir(entry, 0o700)
        except FileExistsError:
            pass


def check_unit_directory(path: Path, *, create: bool) -> None:
    """The systemd user directory is a real directory of this user with no group or other write bit."""
    if create:
        try:
            make_private_directories(path)
        except OSError as exc:
            raise SystemExit(UNIT_DIR_UNSAFE) from exc
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        if create:
            raise SystemExit(UNIT_DIR_UNSAFE) from None
        return
    except OSError as exc:
        raise SystemExit(UNIT_DIR_UNSAFE) from exc
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o022:
        raise SystemExit(UNIT_DIR_UNSAFE)


def unit_environment(runtime: UnitRuntime) -> list[str]:
    """The words ``/usr/bin/env -i`` takes in the unit; the preflight builds its environment from them."""
    return ["HOME=%h", "USER=%u", "LOGNAME=%u", "XDG_RUNTIME_DIR=%t", "PATH=/usr/bin:/bin", "LANG=C.UTF-8",
            f"PYTHONPATH={runtime.pythonpath}", "PYTHONNOUSERSITE=1", "PYTHONSAFEPATH=1",
            "PYTHONDONTWRITEBYTECODE=1", f"PYTHONPYCACHEPREFIX={SOURCE_ONLY_PYCACHE_PREFIX}"]


def service_command(runtime: UnitRuntime) -> list[str]:
    """The isolated interpreter call the unit runs, before the companion's own options."""
    return [str(runtime.interpreter), "-S", "-P", "-s", "-B", "-m", "dgx_monarch.studio"]


def preflight_environment(runtime: UnitRuntime, *, home: str, user: str, runtime_dir: str) -> dict[str, str]:
    """The unit's environment with its specifiers filled in, for the preflight run."""
    filled = {"%h": home, "%u": user, "%t": runtime_dir}
    environment: dict[str, str] = {}
    for word in unit_environment(runtime):
        name, _, value = word.partition("=")
        environment[name] = filled[value] if word in SPECIFIER_WORDS else value
    return environment
