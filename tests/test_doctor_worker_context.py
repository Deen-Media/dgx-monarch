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
from dgx_monarch.cli.doctor_worker_context import worker_context
from dgx_monarch.cli.lifecycle import _pybin_shell
from dgx_monarch.cli.lifecycle_systemd import systemd_worker_unit
from dgx_monarch.cli.systemd_unit import quote
from dgx_monarch.config import ClusterConfig, HostConfig
from dgx_monarch.monarch_launch import worker_policy_args

TOKEN = 'u-123456789abc-0123456789abcdef'  # noqa: S105 - fixture identifier
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
    assert result.stdout.strip() == '0.7.0'
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
    assert result.stdout.strip() == '0.6.0'


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
    assert result.stdout.strip() == 'FAILED_SOURCE_LAYOUT'



@pytest.mark.parametrize('verdict', ["printf 'MANAGED\\nPLAIN\\n'", "echo MANAGED; exit 1", "echo UNKNOWN"])
def test_failed_or_ambiguous_guard_never_runs_environment_probe(context, monkeypatch, verdict):
    from dgx_monarch.cli import doctor_worker_context

    config, host, _base, _managed, _unit, _site, _staging = context
    monkeypatch.setattr(doctor_worker_context, 'managed_source_probe', lambda *_args: verdict)
    result = run(config, host, dict(os.environ))
    assert result.returncode != 0
    assert result.stdout.strip() == 'FAILED_SOURCE_LAYOUT'


def test_wrong_managed_pin_does_not_fall_back_to_matching_base(context):
    from dgx_monarch import TORCHMONARCH_PIN

    config, host, _base, managed, _unit, site, _staging = context
    (managed / 'torchmonarch-0.7.0.dist-info/METADATA').write_text('Name: torchmonarch\nVersion: 0.6.0\n')
    (site / 'torchmonarch-0.6.0.dist-info/METADATA').write_text(f'Name: torchmonarch\nVersion: {TORCHMONARCH_PIN}\n')
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    result = run(config, host, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '0.6.0' != TORCHMONARCH_PIN
