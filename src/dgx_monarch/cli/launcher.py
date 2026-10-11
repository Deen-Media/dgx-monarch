"""Explicit local installation of an environment-bound CLI entry point.

This owner never invokes the target, installs dependencies, or operates Workers.
"""
from __future__ import annotations

import fcntl
import json
import os
import secrets
import shlex
import shutil
import stat
import sys
from pathlib import Path

_MARKER = "# dgx-monarch-launcher-v1 "
_MAX_BYTES = 16384


def _absolute(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or ".." in path.parts or any(ord(c) < 32 or ord(c) == 127 for c in str(path)):
        raise ValueError("launcher paths must be absolute and contain no traversal or control characters")
    return path


def _render(executable: Path, config: Path) -> bytes:
    metadata = json.dumps([str(executable), str(config)], ensure_ascii=True)
    return (
        "#!/bin/sh\n" + _MARKER + metadata + "\n"
        "# Explicit --config uses the CLI's own expansion and validation.\n"
        'case "${1-}" in\n'
        '  --config|--config=*) ;;\n'
        "  *) config=" + shlex.quote(str(config)) + "\n"
        '     [ -f "$config" ] || { printf "%s\\n" "dgxm launcher: saved cluster config is missing or not a file: $config" >&2; exit 2; }\n'
        '     set -- --config "$config" "$@" ;;\n'
        "esac\n"
        "exec " + shlex.quote(str(executable)) + ' "$@"\n'
    ).encode()


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _refuse_launcher_target(target: Path) -> None:
    """Require the environment entry point, never another managed launcher.

    Follow executable symlinks, but inspect bytes without executing the target.
    A damaged or edited marker still denotes a launcher; ownership validation
    for replacement is deliberately stricter than this chain-prevention check.
    """
    fd = os.open(target, os.O_RDONLY | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("selected dgxm executable is not a regular file")
        if b"dgx-monarch-launcher-" in os.read(fd, _MAX_BYTES + 1):
            raise ValueError("selected executable is a managed launcher; select the environment's dgxm executable")
    finally:
        os.close(fd)


def _read_owned(parent: int, name: str) -> tuple[bytes, tuple[int, ...]] | None:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    except FileNotFoundError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ValueError("refusing an unsafe or foreign launcher")
        data = os.read(fd, _MAX_BYTES + 1)
        lines = data.decode().splitlines()
        if len(data) > _MAX_BYTES or len(lines) < 2 or not lines[1].startswith(_MARKER):
            raise ValueError("refusing to replace an unmanaged command")
        values = json.loads(lines[1][len(_MARKER):])
        if not isinstance(values, list) or len(values) != 2 or not all(isinstance(v, str) for v in values):
            raise ValueError("refusing malformed launcher ownership metadata")
        if data != _render(*(_absolute(v) for v in values)):
            raise ValueError("refusing to replace a modified managed launcher")
        if _identity(os.fstat(fd)) != _identity(info):
            raise ValueError("launcher changed while reading")
        return data, _identity(info)
    finally:
        os.close(fd)


def _parent(path: Path, *, create: bool) -> int | None:
    """Walk directories without following symlinks, including the final parent."""
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    return None
                try:
                    os.mkdir(part, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ValueError("launcher directory must be owned by the current user and not writable by others")
        result, fd = fd, -1
        return result
    finally:
        if fd >= 0:
            os.close(fd)


def install(executable: str, config: str, destination: str, *, apply: bool = False) -> str:
    target, cluster, output = map(_absolute, (executable, config, destination))
    if not target.is_file() or not os.access(target, os.X_OK):
        raise ValueError("selected dgxm executable is missing or not executable")
    if not cluster.is_file():
        raise ValueError("selected cluster config is missing or not a file")
    if target.resolve() == output.resolve():
        raise ValueError("launcher cannot execute itself")
    _refuse_launcher_target(target)
    payload = _render(target, cluster)
    if len(payload) > _MAX_BYTES:
        raise ValueError("launcher paths exceed the supported size")
    parent = _parent(output.parent, create=apply)
    if parent is None:
        return "would create"
    lock = -1
    temporary = None
    try:
        if apply:
            lock = os.open(".dgxm-launcher.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=parent)
            info = os.fstat(lock)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError("unsafe launcher lock")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if _identity(os.stat(".dgxm-launcher.lock", dir_fd=parent, follow_symlinks=False)) != _identity(os.fstat(lock)):
                raise ValueError("launcher lock changed")
        before = _read_owned(parent, output.name)
        if before is not None and before[0] == payload and before[1][2] & stat.S_IXUSR:
            return "unchanged"
        if not apply:
            return "would replace managed launcher" if before else "would create"
        temporary = ".dgxm-launcher-" + secrets.token_hex(12)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o700, dir_fd=parent)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(payload)
                stream.flush()
            os.fchmod(fd, 0o755)
            os.fsync(fd)
        finally:
            os.close(fd)
        if _identity(os.stat(output.parent, follow_symlinks=False))[:2] != _identity(os.fstat(parent))[:2]:
            raise ValueError("launcher directory changed during installation")
        _refuse_launcher_target(target)
        if before is None:
            os.link(temporary, output.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        else:
            current = _read_owned(parent, output.name)
            if current != before:
                raise ValueError("launcher changed; preserving the replacement")
            # Cooperative lock plus exact identity/content check. POSIX rename
            # has a final check-to-rename window against non-cooperating writers.
            os.replace(temporary, output.name, src_dir_fd=parent, dst_dir_fd=parent)
            temporary = None
        os.fsync(parent)
        after = _read_owned(parent, output.name)
        if after is None or after[0] != payload:
            raise ValueError("launcher publication could not be verified")
        return "installed"
    finally:
        if temporary is not None:
            os.unlink(temporary, dir_fd=parent)
        if lock >= 0:
            os.close(lock)
        os.close(parent)


def run(args) -> int:
    try:
        result = install(args.executable, args.cluster_config, args.destination, apply=args.apply)
        destination = _absolute(args.destination)
        print(f"launcher: {result}: {destination}")
        selected = shutil.which("dgxm")
        if selected is not None and Path(selected).absolute() == destination:
            print("This process's PATH selects the launcher.")
        elif selected:
            print(f"PATH currently selects another dgxm: {selected}")
        else:
            print("dgxm is not currently discoverable on this process's PATH.")
        print("In your ordinary terminal, run `command -v dgxm` and `dgxm version`; "
              "clear any cached command lookup first (`hash -r` in Bash). "
              "Aliases and functions may override PATH. No shell startup files were changed.")
        return 0
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"launcher: {exc}", file=sys.stderr)
        return 2
