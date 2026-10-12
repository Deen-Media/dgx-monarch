"""Opt-in updater inspector evidence must be fresh at both activity guards."""
from __future__ import annotations

import contextlib
import io
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import update_activity, update_command
from dgx_monarch.cli.update_transaction import UpdateRequest, execute_verified_update
from dgx_monarch.config import ClusterConfig, HostConfig
from test_update_transaction import FakeOps


def config():
    return ClusterConfig(hosts=[HostConfig('worker-a', 'tcp://127.0.0.1:1'),
                                HostConfig('worker-b', 'tcp://127.0.0.1:2')])


def run_client(monkeypatch, report=None, error=None):
    """Run the embedded client body with a simulated authenticated API."""
    definition = ('import json,struct\nDGXM_INSPECTOR_SHA="a"*64\n'
                  'def _request_process_report(_sha):\n')
    definition += f'    raise {error}\n' if error else f'    return {report!r}\n'
    monkeypatch.setattr(update_activity, 'privileged_process_source', lambda enabled: definition)
    calls = []
    def run(_config, host, script, timeout):
        calls.append(host.name)
        assert 0 < timeout <= 15
        code = shlex.split(script)[-1]
        output = io.StringIO()
        exit_code = 0
        with contextlib.redirect_stdout(output):
            try:
                exec(compile(code, '<mock-inspector-client>', 'exec'), {})  # noqa: S102 - generated client with mocked API
            except SystemExit as error:
                exit_code = error.code
        return subprocess.CompletedProcess([], exit_code, output.getvalue(), '')
    monkeypatch.setattr(update_activity.lifecycle, 'run_on_host', run)
    return calls


def test_complete_reports_count_actors_on_every_host(monkeypatch):
    calls = run_client(monkeypatch, {'status': 'ok', 'unknown_count': 0, 'actors': [{'pid': 5}]})
    assert update_activity.privileged_actor_count(config()) == 2
    assert calls == ['worker-a', 'worker-b']


@pytest.mark.parametrize('error', ['FileNotFoundError()', 'ValueError("stale inspection report")',
                                   'ValueError("inspection identity mismatch")', 'PermissionError()'])
def test_missing_stale_or_unauthenticated_helper_blocks(monkeypatch, error):
    run_client(monkeypatch, error=error)
    assert update_activity.privileged_actor_count(config()) is None


def test_unknown_report_is_not_zero(monkeypatch):
    run_client(monkeypatch, {'status': 'unknown', 'unknown_count': 1, 'actors': []})
    assert update_activity.privileged_actor_count(config()) is None


@pytest.mark.parametrize('raw', ['{"actors":true}', '{"actors":-1}', '{"actors":8193}',
                                 '{"actors":0,"extra":1}', 'null', '[]', 'invalid'])
def test_noncanonical_remote_summary_is_unknown(monkeypatch, raw):
    monkeypatch.setattr(update_activity.lifecycle, 'run_on_host',
                        lambda *a, **k: subprocess.CompletedProcess([], 0, raw, ''))
    assert update_activity.privileged_actor_count(config()) is None


def system_ops(monkeypatch, tmp_path, enabled=True):
    ops = update_command.SystemUpdateOps(tmp_path, config(),
        privileged_process_inspection=enabled, release_manager=SimpleNamespace())
    monkeypatch.setattr(update_command, '_count_marked_actors', lambda _: 0)
    monkeypatch.setattr(update_command, '_comfy_candidate_count', lambda: 0)
    monkeypatch.setattr(update_command.comfy_ports, 'find_comfy_ports', lambda: {})
    monkeypatch.setattr(update_command.comfy_ports, 'driver_probe', lambda *_: (None, True))
    return ops


@pytest.mark.parametrize('second,code', [(None, 'activity_unknown'), (1, 'marked_actors')])
def test_new_or_lost_actor_report_before_stop_blocks_transaction(monkeypatch, tmp_path, second, code):
    real = system_ops(monkeypatch, tmp_path)
    reports = iter([0, second])
    monkeypatch.setattr(update_command, 'privileged_actor_count', lambda _: next(reports))
    ops = FakeOps()
    ops.activity_snapshot = real.activity_snapshot
    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)
    assert result.code == code
    assert 'stop' not in ops.events
    assert 'activate' not in ops.events


def test_option_preserves_existing_unknown_or_active_guard(monkeypatch, tmp_path):
    ops = system_ops(monkeypatch, tmp_path)
    monkeypatch.setattr(update_command, 'privileged_actor_count', lambda _: 0)
    monkeypatch.setattr(update_command, '_count_marked_actors', lambda _: None)
    assert ops.activity_snapshot().blockers() == ('activity_unknown',)
    monkeypatch.setattr(update_command, '_count_marked_actors', lambda _: 2)
    assert 'marked_actors' in ops.activity_snapshot().blockers()
    monkeypatch.setattr(update_command, '_count_marked_actors', lambda _: 0)
    monkeypatch.setattr(update_command, '_comfy_candidate_count', lambda: 1)
    assert 'comfy_running' in ops.activity_snapshot().blockers()


def test_default_does_not_consume_privileged_helper(monkeypatch, tmp_path):
    ops = system_ops(monkeypatch, tmp_path, enabled=False)
    monkeypatch.setattr(update_command, 'privileged_actor_count', lambda _: pytest.fail('not opted in'))
    assert ops.activity_snapshot().blockers() == ()


def test_helper_deadline_and_empty_fleet_are_unknown(monkeypatch):
    clock = iter([0.0, 46.0])
    monkeypatch.setattr(update_activity.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(update_activity.lifecycle, 'run_on_host', lambda *a, **k: pytest.fail('deadline exceeded'))
    assert update_activity.privileged_actor_count(config()) is None
    assert update_activity.privileged_actor_count(ClusterConfig()) is None


def test_missing_helper_explains_existing_read_only_start_path(monkeypatch, capsys):
    run_client(monkeypatch, error='FileNotFoundError()')
    assert update_activity.privileged_actor_count(config()) is None
    error = capsys.readouterr().err
    assert 'read-only inspectors running on every configured host' in error
    assert 'docs/INSTALL.md' in error


def test_cli_passes_explicit_flag_only_for_verified_update(monkeypatch, tmp_path):
    from dgx_monarch.cli import main as cli
    seen = []
    monkeypatch.setattr(cli, '_require_config', lambda _: config())
    monkeypatch.setattr(update_command, 'run_verified_update', lambda *a, **k: seen.append(k) or 0)
    args = SimpleNamespace(verify=True, target='origin/HEAD', yes=False, host=None,
                           receipt=None, privileged_process_inspection=True)
    assert cli.cmd_update(args) == 0
    assert seen[0]['privileged_process_inspection'] is True
    args.verify = False
    assert cli.cmd_update(args) == 2
    assert len(seen) == 1


def test_public_update_parser_exposes_flag_and_preserves_setup_default(monkeypatch):
    from dgx_monarch.cli import main as cli
    seen = []
    monkeypatch.setattr(cli, 'cmd_update', lambda args: seen.append(args) or 0)
    assert cli.main(['update', '--verify', '--privileged-process-inspection']) == 0
    assert seen[-1].privileged_process_inspection is True
    monkeypatch.setattr(cli, 'cmd_setup', lambda args: seen.append(args) or 0)
    assert cli.main(['setup']) == 0
    assert seen[-1].privileged_process_inspection is False
