"""Execute the generated Doctor preamble using inert distribution metadata."""
from __future__ import annotations

import ast
import os
import subprocess
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import doctor
from dgx_monarch.cli.doctor_worker_context import env_failure_row, worker_context
from dgx_monarch.cli.lifecycle import _pybin_shell
from dgx_monarch.cli.lifecycle_lock import locked_script
from dgx_monarch.cli.lifecycle_managed import managed_source_probe, probe_kind, probe_refusal
from dgx_monarch.cli.lifecycle_systemd import systemd_worker_unit
from dgx_monarch.cli.systemd_unit import quote
from dgx_monarch.config import ClusterConfig, HostConfig
from dgx_monarch.monarch_launch import worker_policy_args

TOKEN = 'u-123456789abc-0123456789abcdef'  # noqa: S105 - fixture identifier
# The release probe's line for the fixture's slot, whose package reads 1.1.0.
SLOT_LINE = f'dgxm_worker_release=SLOT {TOKEN} 1.1.0'
METADATA_PROBE = '''"$PYBIN" - <<'PYCODE'
import importlib.metadata as md, sys
print(md.version('torchmonarch'))
assert 'torch' not in sys.modules and 'monarch' not in sys.modules
PYCODE
'''


def distribution(site: Path, version: str) -> None:
    info = site / f'torchmonarch-{version}.dist-info'
    info.mkdir(parents=True)
    (info / 'METADATA').write_text(f'Name: torchmonarch\nVersion: {version}\n')


@pytest.fixture
def context(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir(mode=0o700)
    monkeypatch.setenv('HOME', str(home))
    env_dir = tmp_path / 'python'
    venv.EnvBuilder(with_pip=False).create(env_dir)
    python = env_dir / 'bin/python'
    site = next((env_dir / 'lib').glob('python*/site-packages'))
    distribution(site, '0.6.0')
    host = HostConfig(name='worker', address='tcp://192.0.2.2:26600')
    config = ClusterConfig(hosts=(host,), python_bin=str(python))
    base = home / '.local/share/dgx-monarch'
    managed = base / 'releases' / TOKEN / 'site'
    managed.mkdir(parents=True)
    distribution(managed, '0.7.0')
    (managed / 'dgx_monarch').mkdir()
    (managed / 'dgx_monarch' / '__init__.py').write_text('__version__ = "1.1.0"\n')
    (base / 'src').symlink_to(managed)
    unit = home / '.config/systemd/user/dgxm-worker.service'
    unit.parent.mkdir(parents=True)
    unit.write_text(systemd_worker_unit(config, host, managed_source=True, update_token=TOKEN))
    staging = tmp_path / 'driver-only-staging'
    distribution(staging, '0.8.0')
    for path in home.rglob('*'):
        if not path.is_symlink():
            path.chmod(0o700 if path.is_dir() else 0o600)
    return config, host, base, managed, unit, site, staging


def generated_prefix(config, host):
    # Evaluate Doctor's actual generated script expression without calling Doctor
    # (which performs hardware probes), then execute only its context preamble.
    tree = ast.parse(Path(doctor.__file__).read_text())
    expressions = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and isinstance(node.value, ast.JoinedStr)
                   and 'worker_context(config, host)' in ast.unparse(node.value)]
    assert len(expressions) == 1
    empty = SimpleNamespace(PROBE_SOURCE='', SWAP_PROBE_SOURCE='', probe_source=lambda _address: '')
    script = eval(compile(ast.Expression(expressions[0]), '<doctor-host-script>', 'eval'), {  # noqa: S307 - trusted checkout expression
        'config': config, 'host': host, '_pybin_shell': _pybin_shell, 'worker_context': worker_context,
        'comfy_dir_literal': repr('/fixture/comfy'), 'wanted_iface_literal': repr(''),
        'sol_attn_row': empty, 'doctor_nccl': empty, 'doctor_probes': empty, 'listener_generation': empty,
    })
    prefix, marker, _body = script.partition("\"$PYBIN\" - <<'EOF'")
    assert marker
    return prefix


def run(config, host, env):
    return subprocess.run(['bash', '-s'], input=generated_prefix(config, host) + METADATA_PROBE,
                          capture_output=True, text=True, timeout=15, env=env)


@pytest.mark.parametrize('transport', ['local-inherited', 'ssh-clean'])
@pytest.mark.parametrize('unit_kind', ['canonical', 'hardened', 'legacy'])
def test_generated_probe_uses_owned_release_not_base_or_driver_staging(context, transport, unit_kind):
    config, host, _base, _managed, unit, site, staging = context
    if unit_kind == 'hardened':
        unit.write_text(systemd_worker_unit(config, host, managed_source=True,
                                           update_token=TOKEN, site_packages=str(site)))
    elif unit_kind == 'legacy':
        suffix = ' ' + ' '.join(quote(arg) for arg in worker_policy_args(config, host))
        unit.write_text(unit.read_text().replace(suffix + '\n', '\n'))
    env = dict(os.environ)
    if transport == 'local-inherited':
        env['PYTHONPATH'] = str(staging)
    else:
        env.pop('PYTHONPATH', None)
    result = run(config, host, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['dgxm_worker_policy=MANAGED', SLOT_LINE, '0.7.0']
    assert (site / 'torchmonarch-0.6.0.dist-info/METADATA').is_file()


@pytest.mark.parametrize('layout', ['absent', 'plain-directory'])
def test_pre_setup_and_plain_install_keep_base_environment(context, layout):
    config, host, base, _managed, unit, _site, _staging = context
    (base / 'src').unlink()
    unit.unlink()
    if layout == 'plain-directory':
        (base / 'src').mkdir()
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    result = run(config, host, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['dgxm_worker_policy=PLAIN', 'dgxm_worker_release=PLAIN', '0.6.0']


@pytest.mark.parametrize('unit_kind', ['canonical', 'hardened', 'legacy'])
def test_stale_policy_still_inspects_owned_release_and_reports_the_verdict(context, unit_kind):
    from dataclasses import replace

    config, host, _base, _managed, unit, site, staging = context
    if unit_kind == 'hardened':
        unit.write_text(systemd_worker_unit(config, host, managed_source=True,
                                           update_token=TOKEN, site_packages=str(site)))
    elif unit_kind == 'legacy':
        suffix = ' ' + ' '.join(quote(arg) for arg in worker_policy_args(config, host))
        unit.write_text(unit.read_text().replace(suffix + '\n', '\n'))
    result = run(replace(config, rdma_latent_return=True), host, dict(os.environ, PYTHONPATH=str(staging)))
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['dgxm_worker_policy=POLICY_STALE', SLOT_LINE, '0.7.0']


@pytest.mark.parametrize('damage', ['missing-unit', 'changed-unit', 'unknown-link', 'unit-dropin'])
def test_unverified_source_never_falls_back_to_base_or_staging(context, damage):
    config, host, base, _managed, unit, _site, staging = context
    if damage == 'missing-unit':
        unit.unlink()
    elif damage == 'changed-unit':
        unit.write_text(unit.read_text().replace('RestartSec=3', 'RestartSec=4'))
    elif damage == 'unknown-link':
        (base / 'src').unlink()
        (base / 'src').symlink_to(staging)
    else:
        dropin = unit.parent / 'dgxm-worker.service.d'
        dropin.mkdir()
        (dropin / 'override.conf').write_text('[Service]\n')
    result = run(config, host, dict(os.environ, PYTHONPATH=str(staging)))
    assert result.returncode != 0
    assert result.stdout.splitlines() == ['FAILED_SOURCE_LAYOUT', 'dgxm_worker_probe_exit=1']
    status, name, detail = env_failure_row('worker', result)
    assert (status, name) == ('FAIL', 'worker env')
    assert detail.startswith('FAILED_SOURCE_LAYOUT: its managed source link or Worker unit is not one')
    assert '(exit' not in detail


# Doctor runs the probe without the lifecycle lock, so an exit with no verdict
# it accepts names FAILED_SOURCE_PROBE, never FAILED_LIFECYCLE_LOCK.
@pytest.mark.parametrize(('verdict', 'kind', 'code'), [
    ("printf 'MANAGED\\nPLAIN\\n'", 'FAILED_SOURCE_PROBE', 0),
    ("echo MANAGED; exit 1", 'FAILED_SOURCE_PROBE', 1),
    ("echo UNKNOWN", 'FAILED_SOURCE_PROBE', 0),
    ("echo POLICY_STALE", 'FAILED_SOURCE_PROBE', 0),
    ("echo MANAGED; exit 3", 'FAILED_SOURCE_PROBE', 3),
    ("echo POLICY_STALE; exit 1", 'FAILED_SOURCE_PROBE', 1),
    ("echo dgxm_worker_policy=MANAGED; exit 4", 'FAILED_SOURCE_PROBE', 4),
    ("echo FAILED_SOURCE_LAYOUT; exit 1", 'FAILED_SOURCE_LAYOUT', 1),
    ("exit 126", 'FAILED_WORKER_INTERPRETER', 126),
    ("exit 127", 'FAILED_WORKER_INTERPRETER', 127),
])
def test_failed_or_ambiguous_guard_never_runs_environment_probe(context, monkeypatch, verdict, kind, code):
    from dgx_monarch.cli import doctor_worker_context

    config, host, _base, _managed, _unit, _site, _staging = context
    monkeypatch.setattr(doctor_worker_context, 'managed_source_probe', lambda *_args: verdict)
    result = run(config, host, dict(os.environ))
    assert result.returncode == 1
    lines = result.stdout.splitlines()
    # The environment probe never ran: no version line and no policy verdict.
    assert lines[-1] == f'dgxm_worker_probe_exit={code}'
    assert not any(line.startswith('dgxm_worker_policy=') and line != 'dgxm_worker_policy=MANAGED'
                   for line in lines)
    assert not any(line[:1].isdigit() for line in lines)
    _status, _name, detail = env_failure_row('worker', result)
    assert detail.startswith(f'{kind}: ')
    # As in the lifecycle refusal, a layout verdict carries no exit code.
    assert ('(exit' not in detail) if kind == 'FAILED_SOURCE_LAYOUT' else (f'(exit {code})' in detail)
    assert 'FAILED_LIFECYCLE_LOCK' not in detail


def _lifecycle_probe(config, host, env):
    """The probe exactly as ``dgxm restart`` and ``dgxm up`` run it, under the lock."""
    result = subprocess.run(['bash', '-c', locked_script(config.python_bin, managed_source_probe(config, host))],
                            capture_output=True, text=True, timeout=15, env=env)
    return probe_kind(result), result


@pytest.mark.parametrize(('interpreter', 'code'), [('missing', 127), ('not-executable', 126)])
def test_doctor_names_an_unusable_interpreter_as_restart_and_up_do(context, tmp_path, interpreter, code):
    """A Worker interpreter the shell cannot run reads the same cause and exit code
    in Doctor's env row as in the lifecycle refusal, not FAILED_SOURCE_LAYOUT."""
    from dataclasses import replace

    config, host, *_rest = context
    python = tmp_path / 'worker-python'
    if interpreter == 'not-executable':
        python.write_text('#!/bin/sh\nexit 0\n')
        python.chmod(0o600)
    config = replace(config, python_bin=str(python))
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    lifecycle_kind, lifecycle = _lifecycle_probe(config, host, env)
    assert (lifecycle_kind, lifecycle.returncode) == ('FAILED_WORKER_INTERPRETER', code)
    result = run(config, host, env)
    assert result.returncode == 1
    assert result.stdout.splitlines()[-1] == f'dgxm_worker_probe_exit={code}'
    _status, _name, detail = env_failure_row('worker', result)
    cause = probe_refusal('worker', lifecycle_kind, lifecycle).split(';')[0].removeprefix('worker: ')
    assert detail.startswith(cause + '; the environment check did not run; stderr: ')
    assert cause == ('FAILED_WORKER_INTERPRETER: the shell could not run cluster.python there '
                     f'(missing or not executable), so the source check never ran (exit {code})')
    assert 'FAILED_SOURCE_LAYOUT' not in detail and str(tmp_path) not in detail


def test_a_failure_after_the_probe_keeps_the_generic_env_row():
    """An environment probe that fails after the source probe passed is not a
    source verdict: the row keeps the last output line and the exit code."""
    result = subprocess.CompletedProcess(
        [], 1, 'dgxm_worker_policy=MANAGED\n', 'Traceback\nImportError: no torch\n')
    assert env_failure_row('worker', result) == (
        'FAIL', 'worker env', 'remote check exited 1: ImportError: no torch')
    exited = subprocess.CompletedProcess([], 23, '', '')
    assert env_failure_row('worker', exited) == ('FAIL', 'worker env', 'remote check exited 23: no output')


def test_wrong_managed_pin_does_not_fall_back_to_matching_base(context):
    from dgx_monarch import TORCHMONARCH_PIN

    config, host, _base, managed, _unit, site, _staging = context
    (managed / 'torchmonarch-0.7.0.dist-info/METADATA').write_text('Name: torchmonarch\nVersion: 0.6.0\n')
    (site / 'torchmonarch-0.6.0.dist-info/METADATA').write_text(f'Name: torchmonarch\nVersion: {TORCHMONARCH_PIN}\n')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    result = run(config, host, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:2] == ['dgxm_worker_policy=MANAGED', SLOT_LINE]
    assert result.stdout.splitlines()[2:] == ['0.6.0'] != [TORCHMONARCH_PIN]
