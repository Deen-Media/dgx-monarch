"""Launch the optional owner-only Studio host without importing ComfyUI."""
from __future__ import annotations

import argparse
import fcntl
import ipaddress
import os
import re
import secrets
import sqlite3
import stat
import sys
from contextlib import ExitStack, suppress
from pathlib import Path

from .contracts import StudioError

# The owner token lands here, mode 0600, when stdout is not a terminal.
OWNER_FILE = "owner-token"
STATE_FILES = frozenset({"companion.lock", "studio.sqlite3", "studio.sqlite3-wal", "studio.sqlite3-shm"})
# An existing token file is replaced only when it holds an earlier token, never another secret.
EARLIER_TOKEN = re.compile(rb"[A-Za-z0-9_-]{32,128}\n?")
STUDIO_DEPENDENCIES = frozenset({"aiohttp", "PIL"})
INSTALL_HINT = ("Studio requires the optional studio extra (aiohttp and Pillow). Install it with the constrained "
                'procedure in docs/INSTALL.md, using -e "$REPO_DIR[studio]" in both commands; '
                "do not run an unconstrained extras install.")


def _default_state_dir() -> Path:
    # The operator-receipt rule (cli/operator_receipt._state_root): an empty
    # XDG_STATE_HOME counts as unset, and a relative root is refused.
    from ..cli.operator_receipt import _state_root

    try:
        return _state_root(None) / "dgx-monarch" / "studio"
    except ValueError as exc:
        raise SystemExit("XDG_STATE_HOME must be an absolute path; unset it or pass an absolute --state-dir.") from exc


def _absolute(parser: argparse.ArgumentParser, value: Path | None, flag: str) -> Path | None:
    if value is None:
        return None
    path = value.expanduser()
    if not os.path.isabs(path):
        parser.error(f"{flag} must be an absolute path")
    return path


def _terminal_stdout() -> bool:
    try:
        return sys.stdout is not None and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _studio_extra() -> str:
    """Report the optional dependencies; a missing extra never fails the no-deps check."""
    from importlib import import_module

    found, missing = [], []
    for module, label in (("aiohttp", "aiohttp"), ("PIL", "Pillow")):
        try:
            imported = import_module(module)
        except ModuleNotFoundError as exc:
            if (exc.name or "").split(".")[0] != module:
                raise
            missing.append(label)
            continue
        found.append(f"{label} {getattr(imported, '__version__', 'unknown')}")
    if missing:
        return (f"studio extra not installed (missing {', '.join(missing)}); "
                "install it as docs/INSTALL.md describes before starting the companion")
    # With the extra present, a companion module that fails to import fails the check.
    import_module("dgx_monarch.studio.server")
    return f"studio extra installed ({', '.join(found)})"


def _check() -> int:
    from .. import __version__, studio_resource_validation

    try:
        table = studio_resource_validation.check_packaged_studio_resources()
    except (RuntimeError, OSError) as exc:
        raise SystemExit(f"Studio resources failed the packaged check: {exc}") from exc
    print(f"dgx-monarch Studio {__version__}: {len(table)} packaged resources match their manifest; {_studio_extra()}")
    return 0


def _claim_state(path: Path, cleanup: ExitStack) -> Path:
    """Check the owner-only directory and hold its lock before the store writes anything."""
    from .store import private_directory

    try:
        directory = private_directory(path)
        lock = os.open(directory / "companion.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise SystemExit(f"Cannot use the state directory {path}: {exc.strerror or exc}.") from exc
    cleanup.callback(os.close, lock)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("A companion already owns this state directory.") from exc
    return directory


def _write_owner_file(path: Path, token: str, state: Path) -> tuple[int, int]:
    """Replace ``path`` with a 0600 file that holds only the token.

    An existing file at ``path`` is replaced only when it holds an earlier launch's token.
    """
    directory = path.parent
    try:
        info = os.lstat(directory)
    except FileNotFoundError as exc:
        raise StudioError("token_file", "The owner token file must be in an existing directory.") from exc
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise StudioError("token_file", "The owner token file directory must belong to this user with "
                                        "mode 0700 and must not be a symlink.")
    if os.path.realpath(directory) == os.path.realpath(state) and path.name in STATE_FILES:
        raise StudioError("token_file", "The owner token file cannot replace a companion state file.")
    with suppress(FileNotFoundError):
        old = os.lstat(path)
        if (not stat.S_ISREG(old.st_mode) or old.st_uid != os.getuid() or stat.S_IMODE(old.st_mode) & 0o077
                or old.st_size > 4096 or EARLIER_TOKEN.fullmatch(path.read_bytes()) is None):
            raise StudioError("token_file", "The owner token file path holds another file; choose a new path.")
    temporary = directory / f".{path.name}.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    try:
        try:
            os.fchmod(descriptor, 0o600)
            os.write(descriptor, f"{token}\n".encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    except BaseException:
        with suppress(FileNotFoundError):
            os.unlink(temporary)
        raise
    written = os.lstat(path)
    return written.st_dev, written.st_ino


def _remove_owner_file(path: Path, identity: tuple[int, int]) -> None:
    # A token is valid for one launch; remove the file this launch wrote, never a replacement.
    with suppress(OSError):
        info = os.lstat(path)
        if (info.st_dev, info.st_ino) == identity:
            os.unlink(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dgxm-studio", description="Optional local Studio companion")
    parser.add_argument("--host", default="127.0.0.1", help="loopback bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8220)
    parser.add_argument("--driver", default="http://127.0.0.1:8188", help="existing ComfyUI driver origin")
    parser.add_argument("--state-dir", type=Path, help="absolute owner-only state directory (default: "
                        "$XDG_STATE_HOME/dgx-monarch/studio, or ~/.local/state/dgx-monarch/studio)")
    parser.add_argument("--token-file", type=Path, help="write the owner token to this absolute path, mode 0600, "
                        "instead of printing it; its directory must be owner-only")
    parser.add_argument("--config", type=Path, help="existing cluster configuration")
    parser.add_argument("--allow-management", action="store_true", help="authorize individually confirmed owner management operations")
    parser.add_argument("--driver-service", help="existing ComfyUI systemd user service; never a Worker unit")
    parser.add_argument("--check", action="store_true", help="check packaged assets and report versions without starting a server")
    args = parser.parse_args(argv)
    if args.check:
        return _check()
    try:
        if not ipaddress.ip_address(args.host).is_loopback or not 1 <= args.port <= 65535:
            parser.error("Studio binds only to a loopback IP and a port between 1 and 65535")
    except ValueError:
        parser.error("--host must be a loopback IP literal")
    state_dir = _absolute(parser, args.state_dir, "--state-dir") or _default_state_dir()
    token_file = _absolute(parser, args.token_file, "--token-file")
    try:
        from aiohttp import web

        from .comfy import ComfyClient
        from .operations import Operations
        from .server import create_app
        from .store import Store
    except ModuleNotFoundError as exc:
        # Only a missing optional dependency earns the install hint; a broken Studio module keeps its traceback.
        if (exc.name or "").split(".")[0] not in STUDIO_DEPENDENCIES:
            raise
        raise SystemExit(INSTALL_HINT) from exc
    with ExitStack() as cleanup:
        try:
            directory = _claim_state(state_dir, cleanup)
            try:
                store = Store(directory)
            except (OSError, sqlite3.DatabaseError) as exc:
                raise SystemExit(f"Cannot open the state database: {exc}.") from exc
            cleanup.callback(store.close)
            client = ComfyClient(args.driver)
            ops = Operations(store, client, config=args.config, allow_management=args.allow_management, driver_service=args.driver_service)
            token = secrets.token_urlsafe(32)
            host = f"[{args.host}]" if ":" in args.host else args.host
            app = create_app(store, client, token, allowed_hosts={f"{host}:{args.port}"}, operations=ops)
            print(f"Studio: http://{host}:{args.port}", flush=True)
            if token_file is None and _terminal_stdout():
                print(f"Owner token: {token}", flush=True)
            else:
                # Never send the token to a log, pipe or journal.
                target = token_file or directory / OWNER_FILE
                try:
                    identity = _write_owner_file(target, token, directory)
                except OSError as exc:
                    raise SystemExit(f"Cannot write the owner token file {target}: {exc.strerror or exc}.") from exc
                cleanup.callback(_remove_owner_file, target, identity)
                print(f"Owner token file (mode 0600): {target}", flush=True)
            web.run_app(app, host=args.host, port=args.port, access_log=None, print=None)
        except StudioError as exc:
            raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
