"""Launch the optional owner-only Studio host without importing ComfyUI."""
from __future__ import annotations

import argparse
import fcntl
import ipaddress
import os
import secrets
from pathlib import Path

from .contracts import StudioError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dgxm-studio", description="Optional local Studio companion")
    parser.add_argument("--host", default="127.0.0.1", help="loopback bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8220)
    parser.add_argument("--driver", default="http://127.0.0.1:8188", help="existing ComfyUI driver origin")
    parser.add_argument("--state-dir", type=Path, default=Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "dgx-monarch/studio")
    parser.add_argument("--config", type=Path, help="existing cluster configuration")
    parser.add_argument("--allow-management", action="store_true", help="authorize individually confirmed owner management operations")
    parser.add_argument("--driver-service", help="existing ComfyUI systemd user service; never a Worker unit")
    parser.add_argument("--check", action="store_true", help="check packaged assets and report versions without starting a server")
    args = parser.parse_args(argv)
    if args.check:
        from importlib.resources import files

        from .. import __version__
        root = files("dgx_monarch.studio")
        if not root.joinpath("static/index.html").is_file() or not root.joinpath("data/parity.json").is_file():
            parser.error("Studio assets are missing; build the frontend before packaging")
        print(f"dgx-monarch Studio {__version__}: packaged assets present")
        return 0
    try:
        if not ipaddress.ip_address(args.host).is_loopback or not 1 <= args.port <= 65535:
            parser.error("Studio binds only to a loopback IP and a port between 1 and 65535")
    except ValueError:
        parser.error("--host must be a loopback IP literal")
    try:
        from aiohttp import web

        from .comfy import ComfyClient
        from .operations import Operations
        from .server import create_app
        from .store import Store
    except ImportError as exc:
        raise SystemExit("Studio requires the optional studio dependency: install dgx-monarch[studio].") from exc
    store = Store(args.state_dir)
    lock = os.open(store.directory / "companion.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(lock)
        store.close()
        raise SystemExit("A companion already owns this state directory.") from exc
    client = ComfyClient(args.driver)
    ops = Operations(store, client, config=args.config, allow_management=args.allow_management, driver_service=args.driver_service)
    token = secrets.token_urlsafe(32)
    host = f"[{args.host}]" if ":" in args.host else args.host
    print(f"Studio: http://{host}:{args.port}\nOwner token: {token}", flush=True)
    app = create_app(store, client, token, allowed_hosts={f"{host}:{args.port}"}, operations=ops)
    try:
        web.run_app(app, host=args.host, port=args.port, access_log=None, print=None)
    except StudioError as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        os.close(lock)
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
