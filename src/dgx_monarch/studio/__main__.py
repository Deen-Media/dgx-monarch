"""Launch the optional owner-only Studio host without importing ComfyUI."""
from __future__ import annotations

import argparse
import fcntl
import ipaddress
import os
import secrets
import sqlite3
import time
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any

from ..config_schema import DEFAULT_STUDIO_DRIVER, DEFAULT_STUDIO_PORT
from . import driver_hint, launch, sign_in
from .contracts import StudioError

# Moved to launch and sign_in, where dgxm studio shares them; kept importable here.
from .launch import EXTRA_MISSING as EXTRA_MISSING
from .launch import default_state_dir as default_state_dir
from .sign_in import EARLIER_TOKEN as EARLIER_TOKEN
from .sign_in import OWNER_FILE as OWNER_FILE
from .sign_in import STATE_FILES as STATE_FILES
from .sign_in import terminal_stdout as terminal_stdout

STUDIO_DEPENDENCIES = frozenset({"aiohttp", "PIL"})
INSTALL_HINT = ("Studio requires the optional studio extra (aiohttp and Pillow). Install it with the constrained "
                'procedure in docs/INSTALL.md, using -e "$REPO_DIR[studio]" in both commands; '
                "do not run an unconstrained extras install.")
# A liveness probe holds the state lock for an instant; the claim retries this
# many times, this many seconds apart, before it refuses.
LOCK_RETRIES, LOCK_PAUSE = 10, 0.05


def _absolute(parser: argparse.ArgumentParser, value: Path | None, flag: str) -> Path | None:
    if value is None:
        return None
    path = value.expanduser()
    if not os.path.isabs(path):
        parser.error(f"{flag} must be an absolute path")
    return path


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
        return (f"{EXTRA_MISSING} (missing {', '.join(missing)}); "
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
    retries = LOCK_RETRIES
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return directory
        except BlockingIOError as exc:
            if retries == 0:
                raise SystemExit("A companion already owns this state directory.") from exc
        retries -= 1
        time.sleep(LOCK_PAUSE)


def _write(path: Path, text: str, state: Path, cleanup: ExitStack, *, earlier: Any, taken: str, cannot: str) -> None:
    """Write an owner-only file and remove it at exit, unless something else replaced it."""
    try:
        identity = sign_in.write_private_file(path, text, state, earlier=earlier, taken=taken)
    except OSError as exc:
        raise SystemExit(f"{cannot} {path}: {exc.strerror or exc}.") from exc
    cleanup.callback(_remove_private_file, path, identity)


def _remove_private_file(path: Path, identity: tuple[int, int]) -> None:
    # A token is valid for one launch; remove the file this launch wrote, never a replacement.
    with suppress(OSError):
        info = os.lstat(path)
        if (info.st_dev, info.st_ino) == identity:
            os.unlink(path)


def _mismatch(exc: StudioError, store: Any, state: Path, asked: str) -> str:
    """The binding refusal, then the bound and asked origins and the command that frees the name."""
    bound = store.bound_driver()
    lines = launch.mismatch_lines(state, bound, asked) if bound is not None else []
    return "\n".join([str(exc), *lines])


def _deliver(args: argparse.Namespace, token: str, state: Path, token_file: Path | None, cleanup: ExitStack) -> None:
    """Hand the token to a terminal, an owner-only file or the browser; never to a pipe, log or journal."""
    terminal = sign_in.terminal_stdout()
    if token_file is None and terminal:
        print(f"Owner token: {token}", flush=True)
    else:
        target = token_file or state / OWNER_FILE
        _write(target, f"{token}\n", state, cleanup, earlier=EARLIER_TOKEN, taken=sign_in.TOKEN_FILE_TAKEN,
               cannot="Cannot write the owner token file")
        print(f"Owner token file (mode 0600): {target}", flush=True)
    if args.allow_management:
        # This token is code execution on every host: no link, no page and no browser history.
        if token_file is None and terminal:
            print(sign_in.MANAGEMENT_NO_LINK, sign_in.STOP_HINT, sep="\n", flush=True)
        return
    url = sign_in.sign_in_link(args.host, args.port, token)
    route = sign_in.browser_route(os.environ)
    browser = terminal and not args.no_open and route == "local"
    page = state / sign_in.SIGN_IN_FILE
    if token_file is not None or not terminal or browser:
        _write(page, sign_in.sign_in_page(url), state, cleanup, earlier=sign_in.SIGN_IN_PAGE,
               taken=sign_in.SIGN_IN_TAKEN, cannot="Cannot write the sign-in page")
        if token_file is not None or not terminal:
            print(f"Sign-in page (mode 0600): {page}", flush=True)
    if not terminal:
        return
    # With --token-file the user chose a file over the terminal; the page line stands in for the link.
    link = sign_in.link_lines(url) if token_file is None else []
    if browser:
        status, handler = sign_in.open_in_browser(page)
        lines = [status] if status == sign_in.OPENED else [*link, status, *([handler] if handler else [])]
    else:
        lines = list(link)
        if not args.no_open and route == "ssh":
            lines += sign_in.ssh_forward_hint(os.environ, args.port)
        elif not args.no_open:
            lines.append(sign_in.NO_DISPLAY)
    print(*lines, sign_in.STOP_HINT, sep="\n", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dgxm-studio", description="Optional local Studio companion")
    parser.add_argument("--host", default="127.0.0.1", help="loopback bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_STUDIO_PORT)
    parser.add_argument("--driver", default=DEFAULT_STUDIO_DRIVER, help="existing ComfyUI driver origin")
    parser.add_argument("--state-dir", type=Path, help="absolute owner-only state directory (default: "
                        "$XDG_STATE_HOME/dgx-monarch/studio, or ~/.local/state/dgx-monarch/studio)")
    parser.add_argument("--token-file", type=Path, help="write the owner token to this absolute path, mode 0600, "
                        "instead of printing it; its directory must be owner-only")
    parser.add_argument("--config", type=Path, help="existing cluster configuration")
    parser.add_argument("--allow-management", action="store_true", help="authorize individually confirmed owner management operations")
    parser.add_argument("--driver-service", help="existing ComfyUI systemd user service; never a Worker unit")
    parser.add_argument("--no-open", action="store_true", help="never start a browser; print the sign-in link instead")
    parser.add_argument("--check", action="store_true", help="check packaged assets and report versions without starting a server")
    args = parser.parse_args(argv)
    if args.check:
        return _check()
    try:
        if not ipaddress.ip_address(args.host).is_loopback or not 1 <= args.port <= 65535:
            parser.error("Studio binds only to a loopback IP and a port between 1 and 65535")
    except ValueError:
        parser.error("--host must be a loopback IP literal")
    state_dir = _absolute(parser, args.state_dir, "--state-dir") or default_state_dir()
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
            # Read once, before the event loop: text the first-run page shows and Studio never runs.
            start = driver_hint.driver_start(client.origin, args.driver_service, args.config)
            ops = Operations(store, client, config=args.config, allow_management=args.allow_management,
                             driver_service=args.driver_service, driver_start=start)
            token = secrets.token_urlsafe(32)
            host = f"[{args.host}]" if ":" in args.host else args.host
            try:
                app = create_app(store, client, token, allowed_hosts={f"{host}:{args.port}"}, operations=ops)
            except StudioError as exc:
                if exc.code != "driver_identity_mismatch":
                    raise
                raise SystemExit(_mismatch(exc, store, directory, client.origin)) from exc
            # Bind before any token exists outside this process, so a link never names a port another process holds.
            sock = sign_in.bind_listener(args.host, args.port)
            cleanup.callback(sock.close)
            print(f"Studio: http://{host}:{args.port}", flush=True)
            _deliver(args, token, directory, token_file, cleanup)
            web.run_app(app, sock=sock, access_log=None, print=None)
        except StudioError as exc:
            raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
