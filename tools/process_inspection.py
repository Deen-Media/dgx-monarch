#!/usr/bin/env python3
"""Check local inspection access or launch the reviewed foreground inspector."""
import argparse
import grp
import hashlib
import os
import pwd
import stat
import subprocess
import sys
import time
from pathlib import Path

MAX_PIDS = 65536
DEADLINE = 10
MAX_SOURCE = 256 * 1024
SOURCE = Path(__file__).resolve().parents[1] / 'src/dgx_monarch/cli/process_inspector.py'
ROOT_SCRIPT = r'''set -euo pipefail
export PATH=/usr/bin:/bin
[[ "${SUDO_UID:-}" =~ ^[1-9][0-9]*$ ]]
[[ "$2" =~ ^[0-9a-f]{64}$ ]]
inspector_dir=$(mktemp -d /run/dgxm-inspector.XXXXXX)
trap 'rm -f -- "$inspector_dir/inspector.py"; rmdir -- "$inspector_dir"' EXIT
install -o root -g root -m 0600 -- "$1" "$inspector_dir/inspector.py"
printf '%s  %s\n' "$2" "$inspector_dir/inspector.py" | sha256sum --check --status
env -i PATH=/usr/bin:/bin SUDO_UID="$SUDO_UID" \
  /usr/bin/python3 -I -S "$inspector_dir/inspector.py" --lifetime 1800
'''


def starttime(path):
    with path.open('rb') as stream:
        value = stream.read(8193)
    if len(value) > 8192:
        raise ValueError('Oversized process stat')
    tail = value.rsplit(b')', 1)
    if len(tail) != 2:
        raise ValueError('Invalid process stat')
    value = int(tail[1].split()[19])
    if value <= 0:
        raise ValueError('Invalid start time')
    return value


def process_identity(path):
    owner = path.stat().st_uid
    with (path / 'status').open('rb') as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError('Oversized process status')
    for line in raw.splitlines():
        if line.startswith(b'Uid:'):
            values = tuple(int(value) for value in line.split()[1:])
            if len(values) == 4 and all(value >= 0 for value in values):
                return owner, values
            break
    raise ValueError('Missing or malformed process UID information')


def confirmed_gone(path):
    try:
        path.stat()
    except FileNotFoundError:
        return True
    except OSError:
        pass
    return False


def check_access(proc=Path('/proc'), limit=MAX_PIDS, timeout=DEADLINE):
    uid = os.geteuid()
    counts = {'checked': 0, 'denied': 0, 'changed_or_unknown': 0}
    deadline = time.monotonic() + timeout
    scanned = 0
    try:
        with os.scandir(proc) as entries:
            for item in entries:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Inspection deadline reached')
                if not item.name.isdecimal():
                    continue
                scanned += 1
                if scanned > limit:
                    raise ValueError('Process limit reached')
                path = Path(item.path)
                try:
                    identity = process_identity(path)
                    if uid != identity[0] and uid not in identity[1]:
                        continue
                    before = starttime(path / 'stat')
                    for name in ('cmdline', 'environ'):
                        with (path / name).open('rb') as stream:
                            stream.read(1)
                    if starttime(path / 'stat') != before or process_identity(path) != identity:
                        counts['changed_or_unknown'] += 1
                    else:
                        counts['checked'] += 1
                except FileNotFoundError:
                    if not confirmed_gone(path):
                        counts['changed_or_unknown'] += 1
                except PermissionError:
                    counts['denied'] += 1
                except (OSError, ValueError, IndexError):
                    counts['changed_or_unknown'] += 1
    except (OSError, ValueError) as exc:
        return {'status': 'incomplete', 'reason': str(exc), **counts}
    status = 'readable' if counts['checked'] and not counts['denied'] and not counts['changed_or_unknown'] else 'unknown'
    return {'status': status, **counts}


def private_group(gid, uid):
    user = pwd.getpwuid(uid)
    group = grp.getgrgid(gid)
    return (
        gid == user.pw_gid
        and set(group.gr_mem) <= {user.pw_name}
        and all(entry.pw_uid == uid for entry in pwd.getpwall() if entry.pw_gid == gid)
    )


def validate_path(path, uid):
    metadata = path.stat()
    if metadata.st_uid not in (0, uid) or metadata.st_mode & stat.S_IWOTH:
        raise ValueError('Source or parent has unsafe ownership or write access')
    if metadata.st_mode & stat.S_IWGRP and not private_group(metadata.st_gid, uid):
        raise ValueError('Source or parent is writable by a shared group')


def source_hash():
    source = SOURCE.resolve(strict=True)
    for path in (source, *source.parents):
        validate_path(path, os.getuid())
    if not source.is_file():
        raise ValueError('Inspector source is not a regular file')
    with source.open('rb') as stream:
        data = stream.read(MAX_SOURCE + 1)
    if not data or len(data) > MAX_SOURCE:
        raise ValueError('Inspector source size is invalid')
    return source, hashlib.sha256(data).hexdigest()


def serve():
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        raise ValueError('--serve requires a foreground interactive terminal')
    source, digest = source_hash()
    # Fixed system tools only. No caller PATH or shell startup environment.
    for name in ('/usr/bin/sudo', '/bin/bash', '/usr/bin/python3'):
        path = Path(name).resolve(strict=True)
        for component in (path, *path.parents):
            metadata = component.stat()
            if metadata.st_uid != 0 or metadata.st_mode & 0o022:
                raise ValueError('Unsafe system executable')
    print('Starting the read-only inspector for up to 30 minutes. Keep this terminal open.', flush=True)
    return subprocess.run(
        ['/usr/bin/sudo', '/bin/bash', '-s', '--', str(source), digest],
        input=ROOT_SCRIPT, text=True, check=False,
        env={'PATH': '/usr/bin:/bin', 'TERM': 'dumb'},
    ).returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Check access only; this is the default')
    mode.add_argument('--serve', action='store_true', help='Run the reviewed inspector in this terminal with sudo')
    args = parser.parse_args(argv)
    if os.geteuid() == 0 or os.getuid() != os.geteuid():
        parser.error('Run as a normal user with matching real and effective UIDs, not with sudo')
    if args.serve:
        try:
            return serve()
        except KeyboardInterrupt:
            print('Inspector launch interrupted. Check its terminal state before continuing.', file=sys.stderr)
            return 130
        except (OSError, ValueError) as exc:
            print(f'Inspector not started: {exc}', file=sys.stderr)
            return 2
    result = check_access()
    print(f'Process inspection access: {result["status"]}.')
    print(f'Readable: {result["checked"]}; denied: {result["denied"]}; changed or unknown: {result["changed_or_unknown"]}.')
    print('This checks access only. It does not establish process ownership or absence.')
    return 0 if result['status'] == 'readable' else 2


if __name__ == '__main__':
    raise SystemExit(main())
