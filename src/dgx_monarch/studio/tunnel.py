"""Print, or run, the one SSH command that reaches Studio from another computer.

Studio listens on 127.0.0.1 only and its Host check accepts only the literal
``127.0.0.1:<port>`` it bound, so the browser computer needs a local forward
of the same port to the same address. This module prints that line, the
address to browse and the sentences that name the usual mistakes. It imports
no aiohttp, adds no listener and never runs a shell.

The host rule, the session reader and the forward line live in ``sign_in``,
which the companion's SSH hint shares; this module only chooses the port and
the lines around them.
"""
from __future__ import annotations

import os
import shutil
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO
from urllib.parse import urlsplit

from ..config_schema import DEFAULT_STUDIO_PORT, ClusterConfigError
from . import launch, sign_in, unit, unit_state
from .contracts import StudioError

BAD_HOST = ("tunnel takes an SSH host name, an IP address or user@host; it refuses spaces, shell characters, "
            "an IPv6 zone (%) and a leading '-'.")
NAME_THE_HOST = ("Name the Studio host the way the browser computer reaches it, for example "
                 "dgxm studio tunnel studio-host or dgxm studio tunnel user@studio-host.")
NO_SSH = "ssh was not found on PATH."
SAME_ADDRESS = "Keep 127.0.0.1 and the same port on both ends; Studio refuses localhost and any other port."
FORWARD_FAILED = ("If ssh says \"Address already in use\" or \"Could not request local forwarding\", another "
                  "program on that computer holds the port: close the session if it opened, free the port, and do "
                  "not open the sign-in link until the tunnel works, because a page served on that port would read "
                  "the token from the link.")

# The other printed lines, kept here so the tests and the docs quote one text.
PORT_RANGE = "--port must be from 1024 to 65535"
PORT_LINE = "Port: {port} ({source})"
FROM_FLAG = "from --port"
FROM_STUDIO = "from the running Studio"
FROM_CONFIG = "from [studio] port"
FROM_DEFAULT = "default; pass --port if the Studio host sets another port in [studio]"
RUN_THIS = "On the computer with the browser, run:"
BROWSE = "Then browse: {address}"
LINK_FOLLOWS = "Sign in with this link in that browser:"
SIGN_IN_BY_OPEN = "Sign in with the link that dgxm studio open prints in an SSH session on the Studio host."
SIGN_IN_BY_FORM = "Sign in by pasting the owner token that dgxm studio token --value prints on the Studio host."
CONFIG_NOTE = "Note: {error}"


@dataclass(frozen=True)
class _Studio:
    """What this computer's live Studio tells the tunnel: its port, where it listens, its link."""

    port: int
    target: str = "127.0.0.1"
    url: str | None = None
    management: bool = False


def run_tunnel(host: str | None, port: int | None, run: bool, config: Path | None, *,
               environ: Mapping[str, str], terminal: bool, stdout: TextIO, stderr: TextIO,
               execvp: Callable[[str, list[str]], object] = os.execvp) -> int:
    """Print the tunnel lines for ``dgxm studio tunnel`` and return the exit code.

    ``host`` is the typed ``[user@]host`` or ``None``; ``port`` the typed
    ``--port`` or ``None``; ``config`` the typed ``--config`` path or ``None``.
    With ``run`` it prints every line but the forward line to ``stderr`` and
    replaces this process with ssh through ``execvp``, so it returns only
    when a test stands in for ``execvp`` (then 0). An out-of-range ``port``
    raises ``ValueError(PORT_RANGE)``, which the CLI turns into
    ``parser.error``. Exit codes: 0 printed, 1 ``NO_SSH``, 2 ``BAD_HOST`` or
    ``NAME_THE_HOST``.

    Only a host taken from ``SSH_CONNECTION`` proves that this computer is the
    Studio host, so only then does it read this computer's Studio for the
    port and the link. A typed host never reads the unit or the state
    directory: on a browser computer that runs its own Studio, that state
    belongs to the wrong Studio.
    """
    if port is not None and not 1024 <= port <= 65535:
        raise ValueError(PORT_RANGE)
    if host is not None:
        if not sign_in.valid_ssh_destination(host):
            print(BAD_HOST, file=stderr)
            return 2
        destination, ssh_port, local = host, None, None
    else:
        session = sign_in.ssh_session_destination(environ)
        if session is None:
            print(NAME_THE_HOST, file=stderr)
            return 2
        destination, ssh_port = session
        local = _local_studio()
    if run and shutil.which("ssh") is None:
        print(NO_SSH, file=stderr)
        return 1

    if port is not None:
        chosen, source = port, FROM_FLAG
    elif local is not None:
        chosen, source = local.port, FROM_STUDIO
    else:
        configured = _configured_port(config, stderr)
        chosen, source = (configured, FROM_CONFIG) if configured is not None else (DEFAULT_STUDIO_PORT, FROM_DEFAULT)
    # A typed --port that differs from the live Studio's names another Studio.
    studio = local if local is not None and local.port == chosen else None
    target = studio.target if studio is not None else "127.0.0.1"

    forward = sign_in.forward_line(destination, chosen, ssh_port=ssh_port, target=target)
    if studio is not None and studio.management:
        sign_lines = [SIGN_IN_BY_FORM]
    elif studio is not None and studio.url is not None and terminal and not run:
        sign_lines = [LINK_FOLLOWS, "  " + studio.url]
    else:
        sign_lines = [SIGN_IN_BY_OPEN]
    head = [PORT_LINE.format(port=chosen, source=source), RUN_THIS]
    tail = [sign_in.another_way_line(chosen, target=target), BROWSE.format(address=f"http://{target}:{chosen}/"),
            *sign_lines, SAME_ADDRESS, FORWARD_FAILED]
    if not run:
        print("\n".join([*head, "  " + forward, *tail]), file=stdout)
        return 0
    # The forward line is what runs, so the header that introduces it stays out too.
    print("\n".join([head[0], *tail]), file=stderr)
    stderr.flush()
    # The argument list is the printed line word by word, run with no shell:
    # the host rule refuses whitespace, so the split gives back exactly the
    # words the line was built from.
    argv = forward.split(" ")
    execvp(argv[0], argv)
    return 0


def _configured_port(config: Path | None, stderr: TextIO) -> int | None:
    """``[studio] port`` from the config the usual search finds, or ``None``.

    A broken or missing config prints one note line and never blocks: the
    tunnel line then carries the default port.
    """
    try:
        path, _source = launch.find_studio_config(str(config) if config is not None else None)
        if path is None:
            return None
        table = launch.read_studio_config(path)
    except (ClusterConfigError, OSError) as exc:
        print(CONFIG_NOTE.format(error=exc), file=stderr)
        return None
    # The table fills in the default; only a port the file names counts as its own.
    try:
        named = tomllib.loads(path.read_text(encoding="utf-8")).get("studio", {})
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    return table["port"] if isinstance(named, dict) and "port" in named else None


def _local_studio() -> _Studio | None:
    """This computer's live Studio, read the way ``dgxm studio open`` reads it, or ``None``.

    The state directory is the one our unit names, else the default. A
    running unit of ours gives its port from its ``# dgxm-studio-port=``
    line, and its sign-in page counts only when ``unit_state.wait_ready``
    finds it written after the main process started (asked once, never
    waited for). Without a running unit a companion counts as live only while
    it holds the state directory's lock, and its page gives the port. Nothing
    here starts or changes anything; a read that fails counts as no live
    Studio, so the printed line never depends on it.
    """
    text = unit.read_owned_unit()
    settings = unit.unit_settings(text) if text is not None else {}
    state = _state_directory(settings)
    status = unit_state.unit_status() if text is not None else None
    if status is not None and unit_state.unit_active(status):
        written = settings.get("port", "")
        port = int(written) if written.isascii() and written.isdigit() else 0
        if not 1024 <= port <= 65535:
            return None
        if settings.get("management") == "yes":
            return _Studio(port, management=True)
        ready = "unknown" if state is None else unit_state.wait_ready(
            state, management=False, after_monotonic=None, previous_token=None, timeout=0.0,
            status_fn=lambda: status)
        found = _page(state) if state is not None and ready == "ready" else None
        return found if found is not None and found.port == port else _Studio(port)
    if state is None:
        return None
    try:
        held = unit_state.lock_probe(state)
    except SystemExit:
        # The probe refuses a state directory of the wrong shape, as open does;
        # here that only means no live Studio to read.
        return None
    return _page(state) if held == "held" else None


def _state_directory(settings: Mapping[str, str]) -> Path | None:
    """The state directory our unit names, else the default one; ``None`` when there is none to read."""
    if settings.get("state"):
        return Path(settings["state"])
    try:
        return launch.default_state_dir()
    except SystemExit:
        return None  # a relative XDG_STATE_HOME: no state directory to read


def _page(state: Path) -> _Studio | None:
    """Port, listen address and link from the state directory's sign-in page."""
    try:
        page = sign_in.read_sign_in_page(state / sign_in.SIGN_IN_FILE)
    except StudioError:
        return None  # another file there; open refuses it, the tunnel just reads no link
    if page is None:
        return None
    parts = urlsplit(page[0])
    try:
        port = parts.port  # the link pattern allows five digits, so 99999 raises here
    except ValueError:
        return None
    if parts.hostname is None or port is None:
        return None
    target = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return _Studio(port, target, page[0])
