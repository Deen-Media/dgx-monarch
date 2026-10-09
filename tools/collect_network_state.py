#!/usr/bin/env python3
"""Collect private, read-only local network evidence. This does not prove isolation."""

import argparse
import json
import os
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

LIMIT = 256 * 1024
TIMEOUT = 15


def trusted_tool(name):
    """Use system executables, never a repository or user PATH entry."""
    for directory in ('/usr/sbin', '/usr/bin', '/sbin', '/bin'):
        candidate = Path(directory, name)
        if not candidate.exists():
            continue
        resolved = candidate.resolve(strict=True)
        for path in (resolved, *resolved.parents):
            metadata = path.stat()
            if metadata.st_uid != 0 or metadata.st_mode & 0o022:
                raise ValueError(f'Untrusted system executable: {name}')
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise ValueError(f'Not executable: {name}')
        return str(resolved)
    raise FileNotFoundError(f'Missing system tool: {name}')


def run_read(argv, timeout=TIMEOUT, limit=LIMIT, interactive=False):
    """Bound combined output and wall time, including descendants holding pipes."""
    result = {'command': argv, 'status': 'failed', 'stdout': '', 'stderr': ''}
    try:
        process = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, start_new_session=not interactive,
            env={'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LC_ALL': 'C'},
        )
    except OSError as exc:
        result['error'] = str(exc)
        return result
    streams = {'stdout': bytearray(), 'stderr': bytearray()}
    deadline = time.monotonic() + timeout
    total = 0
    failure = None
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ, 'stdout')
        selector.register(process.stderr, selectors.EVENT_READ, 'stderr')
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failure = 'timeout'
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = os.read(key.fileobj.fileno(), min(65536, limit - total + 1))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    room = limit - total
                    streams[key.data].extend(chunk[:room])
                    total += len(chunk)
                    if total > limit:
                        failure = 'output_limit'
                        break
                if failure:
                    break
            if not failure:
                try:
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    failure = 'timeout'
        finally:
            if failure or process.poll() is None:
                try:
                    process.kill() if interactive else os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            process.stdout.close()
            process.stderr.close()
    result.update({name: value.decode('utf-8', errors='replace') for name, value in streams.items()})
    result['returncode'] = process.returncode
    result['status'] = failure or ('ok' if process.returncode == 0 else 'failed')
    return result


def local_identity():
    identity = {}
    for name, path in {
        'hostname': '/proc/sys/kernel/hostname',
        'boot_id': '/proc/sys/kernel/random/boot_id',
    }.items():
        try:
            with open(path, 'rb') as stream:
                raw = stream.read(257)
            if not raw.strip() or len(raw) > 256:
                raise ValueError('Missing or oversized identity value')
            identity[name] = {'status': 'ok', 'value': raw.decode('utf-8').strip()}
        except (OSError, ValueError, UnicodeError) as exc:
            identity[name] = {'status': 'unknown', 'error': str(exc)}
    return identity


def utc_now():
    return datetime.now(UTC).isoformat()


def collect(use_sudo=False):
    started_at = utc_now()
    identity = local_identity()
    reads = {}
    specs = {
        'interfaces': ['-j', '-d', 'link', 'show'],
        'addresses': ['-j', 'address', 'show'],
        'ipv4_routes': ['-j', '-4', 'route', 'show', 'table', 'all'],
        'ipv6_routes': ['-j', '-6', 'route', 'show', 'table', 'all'],
        'ipv4_rules': ['-j', '-4', 'rule', 'show'],
        'ipv6_rules': ['-j', '-6', 'rule', 'show'],
    }
    for name, args in specs.items():
        reads[name] = command_read('ip', args)
    reads['nft_ruleset'] = command_read('nft', ['--json', 'list', 'ruleset'], use_sudo)
    for name, path in {
        'ipv4_forwarding': '/proc/sys/net/ipv4/ip_forward',
        'ipv6_forwarding': '/proc/sys/net/ipv6/conf/all/forwarding',
    }.items():
        try:
            with open(path, 'rb') as stream:
                raw = stream.read(33)
            if raw.strip() not in (b'0', b'1'):
                raise ValueError('Unexpected forwarding value')
            reads[name] = {'status': 'ok', 'path': path, 'stdout': raw.decode('ascii')}
        except (OSError, ValueError) as exc:
            reads[name] = {'status': 'failed', 'path': path, 'error': str(exc)}
    return {
        'schema': 'dgxm-local-network-state-v1',
        'started_at': started_at,
        'finished_at': utc_now(),
        'identity': identity,
        'scope': 'local host only; no scans or network connections',
        'isolation_verified': False,
        'notice': 'Successful reads do not establish a trusted inter-node boundary. Interpret locally.',
        'reads': reads,
    }


def command_read(tool, args, use_sudo=False):
    try:
        argv = [trusted_tool(tool), *args]
        if use_sudo:
            if tool != 'nft' or args != ['--json', 'list', 'ruleset']:
                raise ValueError('Elevation is restricted to the nft ruleset read')
            argv = [trusted_tool('sudo'), '--', *argv]
        result = run_read(argv, timeout=60 if use_sudo else TIMEOUT, interactive=use_sudo)
        if result['status'] == 'ok':
            try:
                json.loads(result['stdout'])
            except (ValueError, TypeError):
                result['status'] = 'invalid_json'
        return result
    except (OSError, ValueError) as exc:
        return {'status': 'unavailable', 'error': str(exc)}


def private_directory(requested):
    if requested is None:
        directory = Path(tempfile.mkdtemp(prefix='dgxm-network-'))
    else:
        directory = Path(requested).absolute()
        os.mkdir(directory, 0o700)
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    metadata = os.fstat(fd)
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        os.close(fd)
        raise ValueError('Output directory must be private and owned by the current user')
    return directory, fd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', help='New private output directory; must not exist')
    parser.add_argument('--sudo', action='store_true', help='Interactively authorize only the nft ruleset read')
    args = parser.parse_args(argv)
    if os.geteuid() == 0 or os.getuid() == 0:
        parser.error('Run as a normal user; do not run this script with sudo')
    if args.sudo and not (sys.stdin.isatty() and sys.stderr.isatty()):
        parser.error('--sudo requires an interactive terminal; passwords are handled by sudo')
    try:
        directory, fd = private_directory(args.output_dir)
        try:
            report = collect(args.sudo)
            output = os.open('report.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
            with os.fdopen(output, 'w') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')
        finally:
            os.close(fd)
    except (OSError, ValueError) as exc:
        print(f'Collection failed: {exc}', file=sys.stderr)
        return 2
    failures = [name for name, item in report['reads'].items() if item['status'] != 'ok']
    print(f'Local reads: {len(report["reads"]) - len(failures)}/{len(report["reads"])} completed.')
    if failures:
        print('Incomplete: ' + ', '.join(failures))
    print('Network isolation is not verified. Keep the report private; let your agent inspect it locally.')
    print(f'Report: {directory / "report.json"}')
    return 2 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
