"""Network collection tests use synthetic commands, never inspect the host network."""
import importlib.util
import json
import stat
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

SPEC = importlib.util.spec_from_file_location('network_collector', Path(__file__).parents[1] / 'tools' / 'collect_network_state.py')
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


def normal_user(monkeypatch):
    monkeypatch.setattr(collector.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(collector.os, 'getuid', lambda: 1000)


def test_private_report_and_summary(tmp_path, monkeypatch, capsys):
    # Keep real ownership for output validation; only root rejection is stubbed.
    monkeypatch.setattr(collector.os, 'geteuid', lambda: 1000)
    report = {'reads': {'synthetic': {'status': 'ok', 'stdout': 'PRIVATE RAW'}}, 'isolation_verified': False}
    monkeypatch.setattr(collector, 'collect', lambda sudo: report)
    directory = tmp_path / 'new'
    assert collector.main(['--output-dir', str(directory)]) == 0
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    output = directory / 'report.json'
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text()) == report
    assert 'PRIVATE RAW' not in capsys.readouterr().out


def test_existing_output_refused_before_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(collector.os, 'geteuid', lambda: 1000)
    collect = Mock(side_effect=AssertionError('must not collect'))
    monkeypatch.setattr(collector, 'collect', collect)
    assert collector.main(['--output-dir', str(tmp_path)]) == 2
    collect.assert_not_called()


def test_root_rejected(monkeypatch):
    monkeypatch.setattr(collector.os, 'geteuid', lambda: 0)
    with pytest.raises(SystemExit):
        collector.main([])


def test_sudo_requires_terminal(monkeypatch):
    normal_user(monkeypatch)
    monkeypatch.setattr(collector.sys.stdin, 'isatty', lambda: False)
    with pytest.raises(SystemExit):
        collector.main(['--sudo'])


def test_default_never_uses_sudo(monkeypatch):
    resolver = Mock(side_effect=lambda tool: '/usr/bin/' + tool)
    runner = Mock(return_value={'status': 'ok', 'stdout': '{}', 'stderr': ''})
    monkeypatch.setattr(collector, 'trusted_tool', resolver)
    monkeypatch.setattr(collector, 'run_read', runner)
    assert collector.command_read('nft', ['--json', 'list', 'ruleset'])['status'] == 'ok'
    resolver.assert_called_once_with('nft')
    assert runner.call_args.args[0] == ['/usr/bin/nft', '--json', 'list', 'ruleset']


def test_sudo_shape_and_restricted_command(monkeypatch):
    monkeypatch.setattr(collector, 'trusted_tool', lambda tool: '/usr/bin/' + tool)
    runner = Mock(return_value={'status': 'ok', 'stdout': '{}'})
    monkeypatch.setattr(collector, 'run_read', runner)
    collector.command_read('nft', ['--json', 'list', 'ruleset'], True)
    assert runner.call_args.args[0] == ['/usr/bin/sudo', '--', '/usr/bin/nft', '--json', 'list', 'ruleset']
    assert runner.call_args.kwargs['interactive'] is True
    runner.reset_mock()
    assert collector.command_read('nft', ['flush', 'ruleset'], True)['status'] == 'unavailable'
    runner.assert_not_called()


def test_permission_failure_not_success(monkeypatch):
    monkeypatch.setattr(collector, 'trusted_tool', lambda tool: '/usr/bin/' + tool)
    monkeypatch.setattr(collector, 'run_read', lambda *a, **kw: {'status': 'failed', 'returncode': 1, 'stderr': 'Operation not permitted'})
    assert collector.command_read('nft', ['--json', 'list', 'ruleset'])['status'] == 'failed'


def test_missing_tool_explicit(monkeypatch):
    monkeypatch.setattr(collector, 'trusted_tool', Mock(side_effect=FileNotFoundError('Missing system tool: nft')))
    assert collector.command_read('nft', [])['status'] == 'unavailable'


def test_invalid_json_refused(monkeypatch):
    monkeypatch.setattr(collector, 'trusted_tool', lambda tool: '/usr/bin/' + tool)
    monkeypatch.setattr(collector, 'run_read', lambda *a, **kw: {'status': 'ok', 'stdout': 'not json'})
    assert collector.command_read('nft', [])['status'] == 'invalid_json'


def test_output_bound():
    result = collector.run_read([sys.executable, '-c', 'print("x" * 100000)'], limit=1000)
    assert result['status'] == 'output_limit'
    assert len(result['stdout']) + len(result['stderr']) <= 1000


def test_timeout():
    result = collector.run_read([sys.executable, '-c', 'import time; time.sleep(5)'], timeout=0.05)
    assert result['status'] == 'timeout'


def test_successful_synthetic_command():
    result = collector.run_read([sys.executable, '-c', 'print("synthetic")'])
    assert result['status'] == 'ok'
    assert result['stdout'] == 'synthetic\n'


def test_symlink_output_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(collector.os, 'geteuid', lambda: 1000)
    destination = tmp_path / 'destination'
    destination.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(destination, target_is_directory=True)
    monkeypatch.setattr(collector, 'collect', Mock(side_effect=AssertionError('must not collect')))
    assert collector.main(['--output-dir', str(link)]) == 2
    assert not list(destination.iterdir())


def test_partial_report_returns_failure(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(collector.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(collector, 'collect', lambda sudo: {
        'reads': {'nft_ruleset': {'status': 'failed', 'stderr': 'private details'}},
        'isolation_verified': False,
    })
    assert collector.main(['--output-dir', str(tmp_path / 'partial')]) == 2
    summary = capsys.readouterr().out
    assert 'Incomplete: nft_ruleset' in summary
    assert 'private details' not in summary


def test_identity_and_times_are_private(tmp_path, monkeypatch, capsys):
    import io

    def fake_open(path, mode):
        values = {
            '/proc/sys/kernel/hostname': b'private-host\n',
            '/proc/sys/kernel/random/boot_id': b'private-boot\n',
            '/proc/sys/net/ipv4/ip_forward': b'0\n',
            '/proc/sys/net/ipv6/conf/all/forwarding': b'0\n',
        }
        return io.BytesIO(values[path])

    monkeypatch.setattr(collector, 'open', fake_open, raising=False)
    monkeypatch.setattr(collector, 'command_read', lambda *a: {'status': 'ok', 'stdout': '[]'})
    monkeypatch.setattr(collector.os, 'geteuid', lambda: 1000)
    clock = iter(['2026-10-09T10:00:00+00:00', '2026-10-09T10:00:01+00:00'])
    monkeypatch.setattr(collector, 'utc_now', lambda: next(clock))
    directory = tmp_path / 'identity'
    assert collector.main(['--output-dir', str(directory)]) == 0
    report = json.loads((directory / 'report.json').read_text())
    assert report['identity']['hostname']['value'] == 'private-host'
    assert report['identity']['boot_id']['value'] == 'private-boot'
    assert report['started_at'] < report['finished_at']
    stdout = capsys.readouterr().out
    for private in ('private-host', 'private-boot', '2026-10-09'):
        assert private not in stdout


def test_unknown_identity_does_not_prevent_command_reads(monkeypatch):
    monkeypatch.setattr(collector, 'open', Mock(side_effect=PermissionError('denied')), raising=False)
    command = Mock(return_value={'status': 'ok', 'stdout': '[]'})
    monkeypatch.setattr(collector, 'command_read', command)
    report = collector.collect()
    assert command.call_count == 7
    assert all(item['status'] == 'unknown' for item in report['identity'].values())


def _exercise_interactive_pty():
    import os
    import pty
    import select
    import signal
    import time

    pid, master = pty.fork()
    if pid == 0:
        try:
            fake = (
                'import os; f=os.open("/dev/tty", os.O_RDWR); '
                'os.write(f,b"VISIBLE_PROMPT:"); '
                'answer=os.read(f,128); print("accepted="+answer.decode().strip())'
            )
            result = collector.run_read([sys.executable, '-c', fake], timeout=3, interactive=True)
            os.write(1, ('RESULT=' + json.dumps(result) + '\n').encode())
            os._exit(0)
        except BaseException:
            os._exit(1)
    received = bytearray()
    sent = False
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.1)
            if not ready:
                continue
            try:
                chunk = os.read(master, 65536)
            except OSError:
                break
            if not chunk:
                break
            received.extend(chunk)
            if b'VISIBLE_PROMPT:' in received and not sent:
                os.write(master, b'synthetic-reply\n')
                sent = True
            if b'RESULT=' in received and b'\n' in received.split(b'RESULT=', 1)[1]:
                break
        assert sent, received.decode(errors='replace')
        payload = received.split(b'RESULT=', 1)[1].splitlines()[0]
        result = json.loads(payload)
        assert result['status'] == 'ok'
        assert result['stdout'] == 'accepted=synthetic-reply\n'
        assert 'VISIBLE_PROMPT' not in result['stdout'] + result['stderr']
    finally:
        os.close(master)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        os.waitpid(pid, 0)


def test_interactive_read_uses_controlling_terminal():
    import subprocess

    # Fork the PTY in a fresh process, outside pytest plugin background threads.
    subprocess.run(
        [sys.executable, '-c',
         'import runpy, sys; runpy.run_path(sys.argv[1])["_exercise_interactive_pty"]()',
         str(Path(__file__).resolve())],
        check=True, timeout=10, capture_output=True,
    )
