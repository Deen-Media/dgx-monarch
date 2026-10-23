"""Sign-in helpers shared by the companion and dgxm studio: the link, the page, private files, browsers.

The rule for every printer: a token or a sign-in link goes to a terminal, an
owner-only file, or the browser; never to a pipe, a log or a journal. This
module imports no aiohttp, so the CLI can import it without the studio extra.
"""
from __future__ import annotations

import errno
import getpass
import ipaddress
import os
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from .contracts import StudioError

# The owner token lands here, mode 0600, when stdout is not a terminal.
OWNER_FILE = "owner-token"
SIGN_IN_FILE = "sign-in.html"
# Files the companion keeps in its state directory; --token-file can never name one.
STATE_FILES = frozenset({"companion.lock", "studio.sqlite3", "studio.sqlite3-wal", "studio.sqlite3-shm",
                         SIGN_IN_FILE})
# An existing token file is replaced only when it holds an earlier token, never another secret.
EARLIER_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,128}\n?")
LINK = re.compile(r"http://(?:127\.\d{1,3}\.\d{1,3}\.\d{1,3}|\[::1\]):\d{1,5}/#token=[A-Za-z0-9_-]{32,128}")

_PAGE_HEAD = ('<!doctype html>\n<meta charset="utf-8">\n<meta name="referrer" content="no-referrer">\n'
              '<meta http-equiv="refresh" content="0; url=')
_PAGE_MIDDLE = '">\n<title>DGX Monarch Studio</title>\n<p><a href="'
_PAGE_TAIL = '" rel="noreferrer">Open DGX Monarch Studio</a></p>\n'
# The whole sign-in page: the URL once as a group and once as a back-reference.
SIGN_IN_PAGE = re.compile(re.escape(_PAGE_HEAD) + "(?P<url>" + LINK.pattern + ")" + re.escape(_PAGE_MIDDLE)
                          + "(?P=url)" + re.escape(_PAGE_TAIL))
_PAGE_LIMIT = 4096
_TOKEN_LIMIT = 4096

PORT_IN_USE = ("Another process already listens on the companion port; stop it, or choose another port: "
               "[studio] port in cluster.toml, or --port for dgxm-studio.")
SIGN_IN_TAKEN = ("The sign-in page in the state directory holds another file; remove it, then start the "
                 "companion again.")
# Refusal texts; the S105 waivers below are for names holding TOKEN, not secrets.
TOKEN_FILE_TAKEN = "The owner token file path holds another file; choose a new path."  # noqa: S105
OPENED = "Sent the sign-in page to your web browser."
OPENED_PLAIN = "Opened Studio in your web browser; paste the owner token into its form."
NOT_A_BROWSER = ("No browser opened: this desktop opens .html files with another program than your web "
                 "browser. Open the sign-in link above.")
NO_OPENER = "No browser opened: xdg-open is missing or failed. Open the sign-in link above."
NO_DISPLAY = ("No browser opened: no display here, or a tmux or screen session. Open the sign-in link above "
              "in a browser on this computer, or run dgxm studio tunnel for another computer.")
SSH_SESSION = ("No browser opened: this terminal is an SSH session. If the computer with the browser has no "
               "tunnel yet, run this there:")
STOP_HINT = "Press Ctrl+C to stop Studio; dgxm studio up keeps it running instead."
NOT_RUNNING = "Studio is not running. Start it with dgxm studio, or keep it running with dgxm studio up."
NO_SIGN_IN_PAGE = "No sign-in page: a Studio started in a terminal printed its sign-in link or token there."
NO_TOKEN_FILE = "No token file: a Studio started in a terminal printed its token there."  # noqa: S105
NO_PIPE = "Refusing to print the token to a pipe or file; run this in a terminal or read the token file."
MANAGEMENT_NO_LINK = ("This Studio runs with --allow-management, so it never signs in by link; paste the owner "
                      "token into the page's form.")
THEN_OPEN_THERE = "Then open the sign-in link there."
# The SSH session gave no usable address, so the forward line names <host>.
NAME_THE_STUDIO_HOST = ("Replace <host> with the name or address the computer with the browser uses for this "
                        "host, as for dgxm studio tunnel <host>.")
ANOTHER_WAY = ("If you reach this host another way, add -L {target}:{port}:{target}:{port} to the ssh command "
               "you use, or set a local forward in your SSH app that listens on {target} port {port} and goes to "
               "{target}:{port}.")
HANDLERS_DIFFER = ".html opens with {html}; the web browser is {http}. To change it: xdg-mime default {http} text/html"
HANDLERS_UNKNOWN = "xdg-mime could not name the .html or http handler."
HOST_PLACEHOLDER = "<host>"

_SSH_USER = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,31}")
_SSH_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,251}[A-Za-z0-9])?")
_DESKTOP_ENTRY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,250}\.desktop")
_HANDLER_TIMEOUT = 5
_OPENER_WAIT = 2

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def terminal_stdout() -> bool:
    """Whether standard output is a terminal; any doubt counts as no."""
    try:
        return sys.stdout is not None and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _bracketed(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def sign_in_link(host: str, port: int, token: str) -> str:
    """http://<host>:<port>/#token=<token>, an IPv6 host in brackets."""
    return f"http://{_bracketed(host)}:{port}/#token={token}"


def link_lines(url: str) -> list[str]:
    """The label and, on its own indented line, the link, so a narrow terminal never splits them."""
    return ["Sign-in link:", "  " + url]


def sign_in_page(url: str) -> str:
    """The sign-in page for a link that fully matches LINK; raises ValueError for any other URL."""
    if LINK.fullmatch(url) is None:
        raise ValueError("The sign-in page takes only a loopback Studio sign-in link.")
    return _PAGE_HEAD + url + _PAGE_MIDDLE + url + _PAGE_TAIL


def read_private_file(path: Path, limit: int) -> tuple[bytes, int] | None:
    """The bytes and st_mtime_ns of an owner-only regular file of at most limit bytes.

    Opened with O_NOFOLLOW and O_NONBLOCK and checked with fstat on the open
    descriptor, so a symlink or a FIFO swapped in can neither redirect nor
    hang the read. Returns None when the file is absent. Raises ValueError for
    any other shape (a symlink, not a regular file, another owner, group or
    other mode bits, too large) and OSError when it cannot be read.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError(f"{path} is a symlink") from exc
        raise
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"{path} is not a regular file")
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise ValueError(f"{path} is not private to this user")
        if info.st_size > limit:
            raise ValueError(f"{path} is larger than {limit} bytes")
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(descriptor, limit + 1 - size):
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                raise ValueError(f"{path} is larger than {limit} bytes")
        return b"".join(chunks), info.st_mtime_ns
    finally:
        os.close(descriptor)


def _private_text(path: Path, limit: int, taken: str) -> tuple[str, int] | None:
    try:
        found = read_private_file(path, limit)
        if found is None:
            return None
        return found[0].decode("ascii"), found[1]
    except (OSError, ValueError) as exc:  # UnicodeDecodeError is a ValueError
        raise StudioError("token_file", taken) from exc


def read_sign_in_page(path: Path) -> tuple[str, int] | None:
    """The sign-in URL and write time of a page this module wrote; None when absent.

    Any other file there raises StudioError("token_file", SIGN_IN_TAKEN).
    """
    found = _private_text(path, _PAGE_LIMIT, SIGN_IN_TAKEN)
    if found is None:
        return None
    match = SIGN_IN_PAGE.fullmatch(found[0])
    if match is None:
        raise StudioError("token_file", SIGN_IN_TAKEN)
    return match["url"], found[1]


def read_token_file(path: Path) -> tuple[str, int] | None:
    """The owner token and write time of a token file; None when absent.

    The file must fully match EARLIER_TOKEN; any other file there raises
    StudioError("token_file", TOKEN_FILE_TAKEN).
    """
    found = _private_text(path, _TOKEN_LIMIT, TOKEN_FILE_TAKEN)
    if found is None:
        return None
    if EARLIER_TOKEN.fullmatch(found[0]) is None:
        raise StudioError("token_file", TOKEN_FILE_TAKEN)
    return found[0].rstrip("\n"), found[1]


def remove_matching(path: Path, pattern: re.Pattern[str]) -> bool:
    """Remove an owner-only regular file whose whole content matches pattern; True when removed.

    The read checks the open descriptor, and the name must still point at the
    file that was read, so a file swapped in meanwhile stays.
    """
    try:
        before = os.lstat(path)
        found = read_private_file(path, max(_PAGE_LIMIT, _TOKEN_LIMIT))
        if found is None or pattern.fullmatch(found[0].decode("utf-8", "replace")) is None:
            return False
        after = os.lstat(path)
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            return False
        os.unlink(path)
    except (OSError, ValueError):
        return False
    return True


def write_private_file(path: Path, text: str, state: Path, *, earlier: re.Pattern[str],
                       taken: str) -> tuple[int, int]:
    """Replace path with a mode 0600 file holding text; return its (st_dev, st_ino).

    The directory must exist, belong to this user with mode 0700 and not be a
    symlink. A file already at path is replaced only when it is private and
    fully matches earlier (an earlier launch's file); anything else raises
    StudioError("token_file", taken). A path naming one of STATE_FILES in the
    state directory is refused, except the state directory's own sign-in.html
    when earlier is SIGN_IN_PAGE: that is the companion writing its page. The
    text is written to an O_EXCL temporary file, fsynced, then renamed over
    path.
    """
    data = text.encode("ascii")
    directory = path.parent
    try:
        info = os.lstat(directory)
    except FileNotFoundError as exc:
        raise StudioError("token_file", "The owner token file must be in an existing directory.") from exc
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise StudioError("token_file", "The owner token file directory must belong to this user with "
                                        "mode 0700 and must not be a symlink.")
    in_state = os.path.realpath(directory) == os.path.realpath(state)
    own_page = in_state and path.name == SIGN_IN_FILE and earlier is SIGN_IN_PAGE
    if in_state and path.name in STATE_FILES and not own_page:
        raise StudioError("token_file", "The owner token file cannot replace a companion state file.")
    old = _private_text(path, _TOKEN_LIMIT, taken)
    if old is not None and earlier.fullmatch(old[0]) is None:
        raise StudioError("token_file", taken)
    temporary = directory / f".{path.name}.{secrets.token_hex(8)}.tmp"
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        try:
            os.fchmod(descriptor, 0o600)
            view = memoryview(data)
            while view:
                view = view[os.write(descriptor, view):]
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


def bind_listener(host: str, port: int) -> socket.socket:
    """A bound, listening socket on host and port, before any token exists.

    socket.create_server sets SO_REUSEADDR (never SO_REUSEPORT) and
    IPV6_V6ONLY for an IPv6 host. A busy port exits with PORT_IN_USE; any
    other failure exits with one line naming the address.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        return socket.create_server((host, port), family=family, backlog=128)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise SystemExit(PORT_IN_USE) from exc
        reason = os.strerror(exc.errno) if exc.errno else str(exc)
        raise SystemExit(f"Cannot listen on {_bracketed(host)}:{port}: {reason}.") from exc


def browser_route(environ: Mapping[str, str]) -> Literal["local", "ssh", "none"]:
    """How this process may reach a browser, read only from environ.

    "ssh" whenever SSH_CONNECTION is present, whatever DISPLAY says; "none"
    inside tmux or screen (TMUX or STY present) or with neither DISPLAY nor
    WAYLAND_DISPLAY; "local" otherwise. There is no X11-forwarding route.
    """
    if "SSH_CONNECTION" in environ:
        return "ssh"
    if "TMUX" in environ or "STY" in environ:
        return "none"
    if not environ.get("DISPLAY") and not environ.get("WAYLAND_DISPLAY"):
        return "none"
    return "local"


def valid_ssh_destination(text: str) -> bool:
    """Whether text is host or user@host under the tunnel host rule.

    user matches [A-Za-z0-9_][A-Za-z0-9_.-]{0,31}. host is a name, an SSH
    config alias or an IPv4 address matching
    [A-Za-z0-9](?:[A-Za-z0-9._-]{0,251}[A-Za-z0-9])?, or a literal IPv6
    address with no zone: only text with no "%", whitespace or control
    character reaches ipaddress. The result is meant to be pasted into a shell.
    """
    if not isinstance(text, str) or not text.isascii() or text.count("@") > 1:
        return False
    user, _, host = text.rpartition("@")
    if "@" in text and _SSH_USER.fullmatch(user) is None:
        return False
    if _SSH_NAME.fullmatch(host):
        return True
    if not host or "%" in host or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in host):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return isinstance(address, ipaddress.IPv6Address) and address.scope_id is None


def ssh_session_destination(environ: Mapping[str, str]) -> tuple[str, int | None] | None:
    """user@address from SSH_CONNECTION's third field, and its fourth field when it is not 22.

    None when SSH_CONNECTION is absent or malformed, or the destination fails
    valid_ssh_destination (sshd can report fe80::1%eth0, which has a zone).
    """
    fields = environ.get("SSH_CONNECTION", "").split()
    if len(fields) != 4 or not fields[3].isascii() or not fields[3].isdigit():
        return None
    port = int(fields[3])
    if not 1 <= port <= 65535:
        return None
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        return None
    destination = f"{user}@{fields[2]}"
    if not valid_ssh_destination(destination):
        return None
    return destination, None if port == 22 else port


def forward_line(destination: str, port: int, *, ssh_port: int | None = None, target: str = "127.0.0.1") -> str:
    """ssh [-p N] -o ExitOnForwardFailure=yes -L <target>:<port>:<target>:<port> <destination>.

    The bind address in -L keeps the forward on that one loopback address of
    the browser computer, so when another program there already holds the
    port, ssh prints "Could not request local forwarding" and
    ExitOnForwardFailure ends the session. Without a bind address ssh listens
    on both 127.0.0.1 and ::1 and opens the session when either one is free.
    An IPv6 target is bracketed. The destination is printed as given; callers
    pass a value that passed valid_ssh_destination, or HOST_PLACEHOLDER.
    """
    bound = _bracketed(target)
    words = ["ssh", *(["-p", str(ssh_port)] if ssh_port is not None else []),
             "-o", "ExitOnForwardFailure=yes", "-L", f"{bound}:{port}:{bound}:{port}", destination]
    return " ".join(words)


def another_way_line(port: int, *, target: str = "127.0.0.1") -> str:
    """The sentence for a jump host, a router forward or an SSH app, naming the port on both ends."""
    return ANOTHER_WAY.format(port=port, target=_bracketed(target))


def ssh_forward_hint(environ: Mapping[str, str], port: int, *, target: str = "127.0.0.1") -> list[str]:
    """The lines a companion or dgxm studio open prints in an SSH session, after the link lines.

    SSH_SESSION, the indented forward line for the address this session
    used, the "another way" line, and THEN_OPEN_THERE. Without a usable
    address the forward line names HOST_PLACEHOLDER, followed by
    NAME_THE_STUDIO_HOST. ``target`` is the loopback address the companion
    listens on, which both ends of the forward use.
    """
    session = ssh_session_destination(environ)
    if session is None:
        lines = [SSH_SESSION, "  " + forward_line(HOST_PLACEHOLDER, port, target=target), NAME_THE_STUDIO_HOST]
    else:
        lines = [SSH_SESSION, "  " + forward_line(session[0], port, ssh_port=session[1], target=target)]
    return [*lines, another_way_line(port, target=target), THEN_OPEN_THERE]


def browser_handlers(run: Runner = subprocess.run) -> tuple[str, str] | None:
    """The desktop entries for .html files and http links, as xdg-mime names them.

    One argv-list call each with a 5-second timeout. None for a missing
    xdg-mime, a timeout, a non-zero exit, or an answer that is empty or not a
    desktop entry name (so nothing odd reaches the terminal).
    """
    answers = []
    for kind in ("text/html", "x-scheme-handler/http"):
        try:
            result = run(["xdg-mime", "query", "default", kind], stdin=subprocess.DEVNULL, capture_output=True,
                         text=True, encoding="utf-8", errors="replace", timeout=_HANDLER_TIMEOUT, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        answer = (result.stdout or "").strip()
        if result.returncode != 0 or _DESKTOP_ENTRY.fullmatch(answer) is None:
            return None
        answers.append(answer)
    return answers[0], answers[1]


def _start(argv: list[str], popen: Callable[..., Any]) -> bool:
    """Start argv detached from this terminal; a failure within two seconds counts as not started."""
    try:
        process = popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        close_fds=True, start_new_session=True)
    except OSError:
        return False
    try:
        return process.wait(timeout=_OPENER_WAIT) == 0
    except subprocess.TimeoutExpired:
        # xdg-open may stay attached to a browser it started, so the page counts as sent.
        return True


def open_in_browser(path: Path, *, run: Runner = subprocess.run, popen: Callable[..., Any] = subprocess.Popen,
                    which: Callable[[str], str | None] = shutil.which) -> tuple[str, str | None]:
    """Send the sign-in page to the web browser by its file: URI, never the token.

    Starts nothing unless xdg-mime names one desktop entry for both .html
    files and http links, because GNOME opens a file: URI with the .html
    handler. Returns (OPENED, None), (NOT_A_BROWSER, the handler line), or
    (NO_OPENER, None) for a missing or failing xdg-open.
    """
    handlers = browser_handlers(run)
    if handlers is None:
        return NOT_A_BROWSER, HANDLERS_UNKNOWN
    if handlers[0] != handlers[1]:
        return NOT_A_BROWSER, HANDLERS_DIFFER.format(html=handlers[0], http=handlers[1])
    opener = which("xdg-open")
    if opener is None:
        return NO_OPENER, None
    return (OPENED, None) if _start([opener, path.as_uri()], popen) else (NO_OPENER, None)


def open_plain_address(url: str, *, popen: Callable[..., Any] = subprocess.Popen,
                       which: Callable[[str], str | None] = shutil.which) -> str:
    """Open an http:// address with no token through xdg-open; OPENED_PLAIN or NO_OPENER.

    For a management companion, which never makes a link. Raises ValueError
    for a URL with a fragment or another scheme, so no token can pass here.
    """
    if not url.startswith("http://") or "#" in url:
        raise ValueError("open_plain_address takes a plain http:// address with no fragment.")
    opener = which("xdg-open")
    if opener is None:
        return NO_OPENER
    return OPENED_PLAIN if _start([opener, url], popen) else NO_OPENER
