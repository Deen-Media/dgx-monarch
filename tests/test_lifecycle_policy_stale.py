"""A Worker unit rendered for an earlier policy is stale, not foreign.

Every case runs the generated source probe against a private fake HOME. No
test here starts systemd, contacts a host, or executes a start or stop script.
"""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from dgx_monarch.cli import doctor, lifecycle, lifecycle_managed, update_installation
from dgx_monarch.cli.doctor_worker_context import worker_policy_row
from dgx_monarch.cli.lifecycle_managed import (
    POLICY_REMEDY,
    POLICY_STALE,
    POLICY_STALE_EXIT,
    legacy_policy_is_current,
    managed_source_probe,
    managed_start_guard,
    prepare_source_sync,
)
from dgx_monarch.cli.lifecycle_scripts import build_start_script
from dgx_monarch.cli.lifecycle_systemd import systemd_worker_unit
from dgx_monarch.cli.probe_certainty import FAIL, OK, WARN
from dgx_monarch.cli.systemd_unit import quote
from dgx_monarch.cli.update_transaction import Certainty, RepoSnapshot, UpdateRequest, execute_verified_update
from dgx_monarch.config import ClusterConfig, HostConfig
from dgx_monarch.monarch_launch import worker_policy_args
from dgx_monarch.monarch_policy import MonarchPolicy, parse_policy, policy_dict
from test_doctor_listener_generation import _config, cluster_rig  # noqa: F401  (fixture re-exported)
from test_update_transaction import TARGET, FakeOps

TOKEN = 'u-123456789abc-0123456789abcdef'  # noqa: S105 - fixture release identifier
NEW_TOKEN = 'u-abcdefabcdef-0011223344556677'  # noqa: S105 - fixture release identifier
SETUP = 's-0123456789abcdef-1'
SITE_PACKAGES = '/opt/worker/lib/python3.12/site-packages'
HOST = HostConfig(name='worker', address='tcp://10.0.0.2:26600')
BASE = ClusterConfig(hosts=(HOST,), python_bin=sys.executable, transport_security='trusted_fabric')
KINDS = ('canonical', 'hardened', 'tokenless', 'setup')
EDITS = {
    'metrics_on': lambda config: replace(config, monarch=parse_policy({'diagnostics': {'metrics': True}})),
    'startup_240': lambda config: replace(config, monarch=parse_policy({'startup_timeout_s': 240})),
    'target_nic': lambda config: replace(config, monarch=parse_policy({'rdma': {'target': 'nic:mlx5_0'}})),
    'latent_on': lambda config: replace(config, rdma_latent_return=True),
}


def _token(payload: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).decode()


OUT_OF_RANGE = _token({**policy_dict(MonarchPolicy()), 'startup_timeout_s': 1})
STALE_TOKEN = worker_policy_args(EDITS['metrics_on'](BASE), HOST)[1]
TAMPER = {
    'garbage_token': lambda text, token: text.replace(token, 'tampered-policy'),
    'default_not_canonical': lambda text, token: text.replace(token, 'e30='),
    'partial_payload': lambda text, token: text.replace(token, _token({'startup_timeout_s': 180})),
    'list_payload': lambda text, token: text.replace(token, 'W10='),
    'extra_argument': lambda text, token: text.replace(f'"{token}"', f'"{token}" "--unknown"'),
    'misordered_native': lambda text, token: text.replace(' "--runtime-policy"', ' "--native-rdma" "--runtime-policy"'),
    'stale_policy_and_body': lambda text, token: text.replace(token, STALE_TOKEN).replace('RestartSec=3', 'RestartSec=4'),
}


def suffix(config: ClusterConfig) -> str:
    return ' ' + ' '.join(quote(arg) for arg in worker_policy_args(config, HOST))


def render(kind: str, config: ClusterConfig = BASE, token: str = TOKEN) -> str:
    """The unit setup or a verified update writes for ``config``."""
    if kind == 'setup':
        lines = systemd_worker_unit(config, HOST, managed_source=True).splitlines()[1:]
        fence = next(line for line in lines if line.startswith('ExecStartPre='))
        lines.remove(fence)
        lines.insert(next(i for i, line in enumerate(lines) if line.startswith('ExecStart=')), fence)
        lines.insert(lines.index('RestartSec=3') + 1, 'TimeoutStopSec=20')
        text = '\n'.join([f'# dgxm-setup-token={SETUP.rsplit("-", 1)[0]} ordinal=1 source={"0" * 64}', *lines]) + '\n'
        return text.replace('PYTHONPATH=%h/.local/share/dgx-monarch/src',
                            f'PYTHONPATH=%h/.local/share/dgx-monarch/releases/{SETUP}/site')
    text = systemd_worker_unit(config, HOST, managed_source=True, update_token=token,
                               site_packages=SITE_PACKAGES if kind == 'hardened' else None)
    return text.replace(suffix(config) + '\n', '\n') if kind == 'tokenless' else text


def run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(['bash', '-s'], input=script, text=True, capture_output=True)


def probe_only(_config, _host, script, timeout=60):
    """Run a read-only probe; refuse to execute anything that drives systemd."""
    assert 'systemctl' not in script, 'a start or stop script reached the test runner'
    return run(script)


def verdict(config: ClusterConfig) -> tuple[int, str]:
    result = run(managed_source_probe(config, HOST))
    return result.returncode, result.stdout.strip()


@pytest.fixture
def installed(tmp_path, monkeypatch):
    """Publish one managed release link and its unit under a fake HOME."""
    monkeypatch.setenv('HOME', str(tmp_path))
    monkeypatch.setenv('XDG_RUNTIME_DIR', str(tmp_path / 'no-runtime'))

    def install(kind: str = 'canonical', text: str | None = None, token: str = TOKEN) -> Path:
        slot = SETUP if kind == 'setup' else token
        base = tmp_path / '.local/share/dgx-monarch'
        (base / 'releases' / slot / 'site').mkdir(parents=True, exist_ok=True)
        link = base / 'src'
        if link.is_symlink():
            link.unlink()
        link.symlink_to(f'releases/{slot}/site')
        unit = tmp_path / '.config/systemd/user/dgxm-worker.service'
        unit.parent.mkdir(parents=True, exist_ok=True)
        unit.write_text(render(kind) if text is None else text)
        for path in tmp_path.rglob('*'):
            if not path.is_symlink():
                path.chmod(0o700 if path.is_dir() else 0o600)
        return unit

    return install


class LogRecorder:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def warning(self, message: str, *args: object) -> None:
        self.lines.append(message % args)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('edit', sorted(EDITS))
def test_a_policy_edit_reads_stale_and_its_revert_reads_managed(installed, kind, edit):
    installed(kind)
    assert verdict(BASE) == (0, 'MANAGED')
    assert verdict(EDITS[edit](BASE)) == (POLICY_STALE_EXIT, POLICY_STALE)
    if kind != 'tokenless':  # a tokenless unit cannot carry the edited policy
        installed(kind, render(kind, EDITS[edit](BASE)))
        assert verdict(EDITS[edit](BASE)) == (0, 'MANAGED')
        assert verdict(BASE) == (POLICY_STALE_EXIT, POLICY_STALE)


def test_an_external_setup_checkout_reads_stale_too(tmp_path, monkeypatch):
    from test_lifecycle_managed import external_setup_layout

    config, host, _package, _unit, _txn = external_setup_layout(tmp_path, monkeypatch)
    assert run(managed_source_probe(config, host)).stdout.strip() == 'MANAGED'
    result = run(managed_source_probe(replace(config, rdma_latent_return=True), host))
    assert (result.returncode, result.stdout.strip()) == (POLICY_STALE_EXIT, POLICY_STALE)


@pytest.mark.parametrize('damage', sorted(TAMPER))
def test_a_malformed_policy_stays_failed_source_layout(installed, damage):
    installed('canonical', TAMPER[damage](render('canonical'), worker_policy_args(BASE, HOST)[1]))
    for config in (BASE, EDITS['metrics_on'](BASE)):
        assert verdict(config) == (1, 'FAILED_SOURCE_LAYOUT')


def test_a_tokenless_unit_is_current_only_under_the_default_policy(installed):
    installed('tokenless')
    assert legacy_policy_is_current(BASE, HOST)
    assert verdict(BASE) == (0, 'MANAGED')
    elsewhere = replace(BASE, monarch=parse_policy({'rdma': {'host_targets': {'other': 'nic:mlx5_0'}}}))
    assert legacy_policy_is_current(elsewhere, HOST)
    assert verdict(elsewhere) == (0, 'MANAGED')
    for edit in EDITS.values():
        assert not legacy_policy_is_current(edit(BASE), HOST)
        assert verdict(edit(BASE)) == (POLICY_STALE_EXIT, POLICY_STALE)


def _row(text: str, kind: str) -> dict:
    home = '/home/user'
    site = f'{home}/.local/share/dgx-monarch/' + (f'releases/{SETUP}/site' if kind == 'setup' else 'src')
    return {'unit': base64.b64encode(text.encode()).decode(), 'sha256': hashlib.sha256(text.encode()).hexdigest(),
            'site': site, 'site_packages': SITE_PACKAGES if kind == 'hardened' else '', 'home': home,
            'user': 'user', 'live': {'kind': 'symlink', 'target': f'{home}/.local/share/dgx-monarch/releases/{TOKEN}/site'}}


@pytest.mark.parametrize('kind', KINDS)
def test_update_capture_accepts_a_stale_unit_for_rerendering(monkeypatch, kind):
    monkeypatch.setattr(update_installation.lifecycle, '_is_local', lambda _host: False)
    row = _row(render(kind), kind)
    for config in (BASE, *(edit(BASE) for edit in EDITS.values())):
        assert update_installation.recognized_unit(config, HOST, row)


@pytest.mark.parametrize('damage', [*sorted(TAMPER), 'out_of_range'])
def test_update_capture_refuses_a_malformed_policy(monkeypatch, damage):
    monkeypatch.setattr(update_installation.lifecycle, '_is_local', lambda _host: False)
    token = worker_policy_args(BASE, HOST)[1]
    text = render('canonical').replace(token, OUT_OF_RANGE) if damage == 'out_of_range' else TAMPER[damage](render('canonical'), token)
    for config in (BASE, EDITS['metrics_on'](BASE)):
        assert not update_installation.recognized_unit(config, HOST, _row(text, 'canonical'))


def test_source_sync_warns_with_host_verdict_and_remedy(installed, monkeypatch):
    installed('canonical')
    recorder = LogRecorder()
    monkeypatch.setattr(lifecycle_managed, 'log', recorder)
    assert prepare_source_sync(BASE, HOST, False, probe_only) == (False, True)
    assert recorder.lines == []
    assert prepare_source_sync(EDITS['latent_on'](BASE), HOST, False, probe_only) == (False, False)
    [line] = recorder.lines
    assert line.startswith('worker: POLICY_STALE: ')
    assert POLICY_REMEDY in line and 'TROUBLESHOOTING.md #113' in line


def test_every_failed_source_probe_is_logged(installed, monkeypatch):
    unit = installed('canonical')
    unit.write_text(unit.read_text().replace('RestartSec=3', 'RestartSec=4'))
    recorder = LogRecorder()
    monkeypatch.setattr(lifecycle_managed, 'log', recorder)
    assert prepare_source_sync(BASE, HOST, False, probe_only) == (False, False)
    assert prepare_source_sync(BASE, HOST, False, lambda *_a: subprocess.CompletedProcess([], 255, '', '')) == (False, None)
    assert recorder.lines[0].startswith('worker: FAILED_SOURCE_LAYOUT: ')
    assert 'not observed' in recorder.lines[1]


def _record_lifecycle(monkeypatch, runner=probe_only) -> list[str]:
    events: list[str] = []
    monkeypatch.setattr(lifecycle, 'run_on_host', runner)
    monkeypatch.setattr(lifecycle, 'down', lambda _config: events.append('down') or True)
    monkeypatch.setattr(lifecycle, 'up', lambda _config, *, sync: events.append(f'up:{sync}') or True)
    monkeypatch.setattr(lifecycle.time, 'sleep', lambda _seconds: None)
    return events


@pytest.mark.parametrize('sync', [False, True])
def test_restart_refuses_a_stale_host_before_stopping_any_worker(installed, monkeypatch, capsys, sync):
    installed('canonical')
    events = _record_lifecycle(monkeypatch)
    assert lifecycle.restart(EDITS['metrics_on'](BASE), sync=sync) is False
    assert events == []
    out = capsys.readouterr().out
    assert '  worker: POLICY_STALE: ' in out and POLICY_REMEDY in out
    assert out.rstrip().endswith('restart refused before stopping any Worker service')
    assert lifecycle.restart(BASE, sync=sync) is True
    assert events == ['down', f'up:{sync}']


def test_restart_refuses_a_foreign_unit_before_stopping_any_worker(installed, monkeypatch, capsys):
    unit = installed('canonical')
    unit.write_text(unit.read_text().replace('RestartSec=3', 'RestartSec=4'))
    events = _record_lifecycle(monkeypatch)
    assert lifecycle.restart(BASE, sync=False) is False
    assert events == []
    assert 'worker: FAILED_SOURCE_LAYOUT: ' in capsys.readouterr().out


@pytest.mark.parametrize('answer', [OSError('no route'), subprocess.TimeoutExpired('ssh', 60), 255])
def test_an_unanswered_source_probe_keeps_the_existing_restart_order(monkeypatch, answer):
    def runner(_config, _host, _script, timeout=60):
        if isinstance(answer, BaseException):
            raise answer
        return subprocess.CompletedProcess([], answer, '', '')

    events = _record_lifecycle(monkeypatch, runner)
    assert lifecycle.restart(BASE, sync=False) is True
    assert events == ['down', 'up:False']


def test_up_refuses_a_stale_host_at_its_source_check(installed, monkeypatch, capsys):
    installed('canonical')
    scripts: list[str] = []
    recorder = LogRecorder()
    monkeypatch.setattr(lifecycle_managed, 'log', recorder)
    monkeypatch.setattr(lifecycle, '_is_local', lambda _host: False)
    monkeypatch.setattr(lifecycle, 'run_on_host', lambda *args, **kwargs: scripts.append(args[2]) or probe_only(*args))
    assert lifecycle.up(EDITS['latent_on'](BASE)) is False
    assert len(scripts) == 1
    assert 'worker: FAILED (package sync; not started; see the warning above)' in capsys.readouterr().out
    assert recorder.lines[0].startswith('worker: POLICY_STALE: ')


def test_up_without_sync_prints_the_stale_remedy_from_its_start_guard(installed, monkeypatch, capsys):
    installed('canonical')
    edited = EDITS['metrics_on'](BASE)

    def guard_only(config, host, script, timeout=60):
        # Execute only the guard the start payload opens with, never the payload.
        assert managed_start_guard(config, host, f'"$HOME/{lifecycle._MANAGED_SRC_REL}"') in script
        return run(managed_start_guard(config, host, '"$HOME/unused"') + 'echo GUARD_PASSED\n')

    monkeypatch.setattr(lifecycle, '_is_local', lambda _host: False)
    monkeypatch.setattr(lifecycle, 'run_on_host', guard_only)
    assert lifecycle.up(edited, sync=False) is False
    out = capsys.readouterr().out
    assert 'worker: FAILED (tcp://10.0.0.2:26600, nohup)' in out
    assert '    worker: POLICY_STALE: ' in out and POLICY_REMEDY in out and 'stdout:' not in out


def test_only_an_update_owned_start_accepts_a_stale_unit(installed):
    installed('canonical')
    edited = EDITS['metrics_on'](BASE)
    refused = run(managed_start_guard(edited, HOST, '/plain') + 'echo GUARD_PASSED\n')
    assert (refused.returncode, refused.stdout.strip()) == (1, POLICY_STALE)
    accepted = run(managed_start_guard(edited, HOST, '/plain', allow_stale=True) + 'echo "$DGXM_START_SOURCE"\n')
    assert accepted.returncode == 0
    assert accepted.stdout.strip().endswith('/.local/share/dgx-monarch/src')
    arguments = {'pattern': 'worker_loop', 'pythonpath': '"$HOME/src"', 'unit_name': 'dgxm-worker.service',
                 'log_path': '$HOME/log', 'pybin_shell': '', 'loop_args': ''}
    plain_start = build_start_script(edited, HOST, generation=None, **arguments)
    update_start = build_start_script(edited, HOST, generation=TOKEN, **arguments)
    assert f'{POLICY_STALE_EXIT}:{POLICY_STALE}) echo {POLICY_STALE}; exit 1;;' in plain_start
    assert f'echo {POLICY_STALE}; exit 1' not in update_start


def test_doctor_policy_row_names_each_verdict():
    assert worker_policy_row('h1', ['torchmonarch=x', 'dgxm_worker_policy=MANAGED'])[:2] == (OK, 'h1 worker policy')
    assert worker_policy_row('h1', ['dgxm_worker_policy=PLAIN'])[:2] == (OK, 'h1 worker policy')
    status, name, detail = worker_policy_row('h1', ['dgxm_worker_policy=POLICY_STALE'])
    assert (status, name) == (FAIL, 'h1 worker policy')
    assert detail.startswith('STALE: ') and POLICY_REMEDY in detail and 'TROUBLESHOOTING.md #113' in detail
    status, _name, detail = worker_policy_row('h1', ['torchmonarch=x'])
    assert status == WARN and detail.startswith('unobserved:')


@pytest.mark.parametrize(('verdict_line', 'status'), [('MANAGED', OK), ('POLICY_STALE', FAIL)])
def test_doctor_reports_the_policy_row_from_the_host_probe(cluster_rig, capsys, verdict_line, status):  # noqa: F811
    from dgx_monarch import TORCHMONARCH_PIN

    stdout = (f'torchmonarch={TORCHMONARCH_PIN} torch=2.test comfy=True nccl_proto=unset\n'
              f'fabric_iface=up rdma_active=2\ndgxm_worker_policy={verdict_line}\n')
    cluster_rig.setattr(doctor, 'run_on_host', lambda *_a, **_kw: subprocess.CompletedProcess([], 0, stdout, ''))
    rows = doctor._doctor_rows(_config('tcp://10.0.0.1:26600'))
    capsys.readouterr()
    [row] = [row for row in rows if row['name'] == 'h1 worker policy']
    assert row['status'] == status
    assert (row in doctor._failures(rows)) is (status == FAIL)


def test_update_activation_renders_the_current_policy_through_the_attested_writer(installed, monkeypatch):
    edited = EDITS['metrics_on'](BASE)
    prior = render('canonical')
    requests: list[dict] = []
    monkeypatch.setattr(update_installation, 'installation_script', lambda request: requests.append(request) or 'true')

    def runner(_config, _host, script, _timeout):
        assert 'true' in script  # the request never executes here
        return subprocess.CompletedProcess([], 0, 'ACTIVATED\n', '')

    snapshot = {'unit': base64.b64encode(prior.encode()).decode(), 'site_packages': ''}
    assert update_installation.change_installation(edited, HOST, NEW_TOKEN, snapshot, runner) is Certainty.SUCCEEDED
    [request] = requests
    assert request['operation'] == 'activate' and request['prior'] is snapshot and request['token'] == NEW_TOKEN
    unit = base64.b64decode(request['unit']).decode()
    assert unit == render('canonical', edited, NEW_TOKEN)
    assert suffix(edited) + '\n' in unit
    installed('canonical', unit, NEW_TOKEN)
    assert verdict(edited) == (0, 'MANAGED')
    assert verdict(BASE) == (POLICY_STALE_EXIT, POLICY_STALE)


def _same_target_ops() -> FakeOps:
    ops = FakeOps()
    ops.snapshots = [RepoSnapshot(TARGET, True, True), RepoSnapshot(TARGET, True, True)]
    return ops


def test_a_same_target_verified_update_runs_the_whole_transaction():
    ops = _same_target_ops()
    result = execute_verified_update(UpdateRequest('HEAD', assume_yes=True), ops)
    assert result.code == 'updated'
    assert ops.events == [
        'resolve:HEAD', f'repo:{TARGET}', 'activity', f'stage:{TARGET}', f'repo:{TARGET}', 'activity',
        'stop', 'activate', 'start_no_sync', 'doctor', 'cluster_smoke', 'readback', 'commit_driver',
        'finalize', 'cleanup',
    ]


def test_a_same_target_update_restores_the_prior_units_when_doctor_fails():
    from test_update_transaction import FAIL as FAILED

    ops = _same_target_ops()
    ops.doctor_result = FAILED
    result = execute_verified_update(UpdateRequest('HEAD', assume_yes=True), ops)
    assert result.code == 'doctor_failed'
    assert ops.events[-5:] == ['doctor', 'stop', 'compensate', 'start_no_sync', 'cleanup']
    assert result.receipt['status'] == 'failed'


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(repo), '-c', 'user.name=Test',
                                    '-c', 'user.email=test@example.test', *args], text=True).strip()


def test_head_target_resolves_to_the_installed_commit_even_ahead_of_origin(tmp_path):
    from dgx_monarch.cli.update_command import SystemUpdateOps

    remote = tmp_path / 'origin'
    remote.mkdir()
    _git(remote, 'init', '-b', 'main')
    _git(remote, 'commit', '--allow-empty', '-m', 'published')
    local = tmp_path / 'local'
    subprocess.run(['git', 'clone', str(remote), str(local)], check=True, capture_output=True)
    _git(local, 'commit', '--allow-empty', '-m', 'installed but unpublished')
    config_file = tmp_path / 'cluster.toml'
    config_file.write_text('# fixture\n')
    ops = SystemUpdateOps(local, ClusterConfig(source=str(config_file)))
    head = _git(local, 'rev-parse', 'HEAD')
    assert ops.resolve_target('HEAD') == head
    assert ops.repo_snapshot(head) == RepoSnapshot(head, True, True)
