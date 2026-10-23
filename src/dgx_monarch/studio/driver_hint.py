"""The start command the first-run page shows while no driver accepts a connection.

Studio shows this text to a token holder and never runs it. The read happens
once, before the event loop, and only reads files: it starts no process and
opens no socket. A configuration it cannot use never stops the companion; the
operations that need the configuration keep their own refusals.
"""
from __future__ import annotations

import os
import shlex
import sys
import tomllib
from pathlib import Path
from urllib.parse import urlsplit

from ..config_schema import ClusterConfigError

# The repository launcher, relative to a dgx-monarch checkout.
LAUNCHER = Path("scripts") / "comfy-driver.sh"
# The documented node-pack directory under a ComfyUI checkout (docs/INSTALL.md).
NODE_PACK = Path("custom_nodes") / "dgx-monarch"
# The loopback hosts a driver started by the launcher can listen on, by origin host.
LISTEN = {"127.0.0.1": "127.0.0.1", "localhost": "127.0.0.1", "::1": "::1"}
_CONFIG_FAULTS = (ClusterConfigError, OSError, tomllib.TOMLDecodeError, ValueError)


def _package_file() -> Path:
    """This package's ``__init__.py`` as Python found it, not resolved."""
    import dgx_monarch

    return Path(dgx_monarch.__file__ or "")


def _word(value: str | Path) -> str:
    return shlex.quote(os.path.expanduser(str(value)))


def _installed_checkout() -> tuple[Path, Path] | None:
    """The ComfyUI checkout and launcher of the INSTALL layout, from this package's own path."""
    package = _package_file().parent
    checkout = package.parent.parent
    if package.parent.name != "src" or checkout.parent.name != "custom_nodes":
        return None
    comfy = checkout.parent.parent
    script = checkout / LAUNCHER
    return (comfy, script) if (comfy / "main.py").is_file() and script.is_file() else None


def _configured_checkout(config: Path) -> tuple[Path, Path] | None:
    """The ComfyUI checkout that ``[cluster] comfy_dir`` names, with the launcher of its node pack."""
    from ..config import load_cluster_config

    comfy_dir = load_cluster_config(config).comfy_dir
    if not comfy_dir:
        return None
    comfy = Path(os.path.expanduser(comfy_dir))
    script = comfy / NODE_PACK / LAUNCHER
    return (comfy, script) if script.is_file() else None


def _launcher(origin: str, config: Path | None, usable: bool) -> str | None:
    """The repository launcher command for a loopback origin, or None."""
    parts = urlsplit(origin)
    try:
        port = parts.port or 80
    except ValueError:
        return None
    listen = LISTEN.get(parts.hostname or "")
    if parts.scheme != "http" or listen is None or not usable:
        return None
    found = _installed_checkout()
    if found is None and config is not None:
        try:
            found = _configured_checkout(config)
        except _CONFIG_FAULTS:
            found = None
    if found is None:
        return None
    comfy, script = found
    # The companion's own interpreter is absolute and sits beside the ComfyUI one on the
    # documented install; [cluster] python names the Worker interpreter and may be a bare word.
    words = ["bash", _word(script), "--comfy-dir", _word(comfy), "--python", _word(sys.executable),
             "--listen", listen, "--port", str(port), "--no-browser"]
    if config is not None:
        words += ["--cluster-config", _word(config)]
    return " ".join(words)


def driver_start(origin: str, driver_service: str | None, config: Path | None) -> dict[str, str | None]:
    """The command the page shows when no driver accepts a connection. Studio never runs it.

    Precedence: the companion's driver service, then ``[studio] driver_command``
    from its configuration, then the repository launcher for a loopback
    origin, else no command. ``config`` in the result is the companion's
    absolute configuration path, so the page can name the file the text came
    from. A configuration Studio's reader refuses yields no command from it,
    and no launcher command that would pass it along.
    """
    from .launch import read_studio_config

    path = config.expanduser().absolute() if config is not None else None
    named = str(path) if path is not None else None
    if driver_service:
        return {"command": f"systemctl --user start {shlex.quote(driver_service)}",
                "source": "driver_service", "config": named}
    usable = True
    if path is not None:
        try:
            table = read_studio_config(path, [])
        except _CONFIG_FAULTS:
            usable = False
        else:
            if table["driver_command"]:
                return {"command": table["driver_command"], "source": "driver_command", "config": named}
    command = _launcher(origin, path, usable)
    return {"command": command, "source": "launcher" if command else None, "config": named}
