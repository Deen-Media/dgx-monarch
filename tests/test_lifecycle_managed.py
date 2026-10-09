"""Immutable release slots survive ordinary lifecycle operations."""
from __future__ import annotations

import os
import pwd
import subprocess
import sys

import pytest

from dgx_monarch.cli import lifecycle, lifecycle_host, lifecycle_systemd
from dgx_monarch.cli.lifecycle_managed import managed_source_probe, managed_start_guard
from dgx_monarch.config import ClusterConfig, HostConfig

TOKEN = 'u-123456789abc-0123456789abcdef'  # noqa: S105 - fixture release identifier


def layout(tmp_path, monkeypatch, *, absolute=False):
    monkeypatch.setenv('HOME', str(tmp_path))
    host = HostConfig(name='worker', address='tcp://10.0.0.2:26600')
    config = ClusterConfig(hosts=(host,), python_bin=sys.executable)
    base = tmp_path / '.local/share/dgx-monarch'
    site = base / 'releases' / TOKEN / 'site'
    site.mkdir(parents=True)
    (site / 'keep.py').write_text('original release\n')
    (base / 'src').symlink_to(site if absolute else f'releases/{TOKEN}/site')
    unit = tmp_path / '.config/systemd/user/dgxm-worker.service'
    unit.parent.mkdir(parents=True)
    unit.write_text(lifecycle_systemd.systemd_worker_unit(
        config, host, managed_source=True, update_token=TOKEN))
    for path in tmp_path.rglob('*'):
        if not path.is_symlink():
            path.chmod(0o700 if path.is_dir() else 0o600)
    return config, host, base, site, unit


def run(script):
    return subprocess.run(['bash', '-s'], input=script, text=True, capture_output=True)


@pytest.mark.parametrize('absolute', [False, True])
def test_sync_preserves_update_slot(tmp_path, monkeypatch, absolute):
    config, host, base, site, _unit = layout(tmp_path, monkeypatch, absolute=absolute)
    monkeypatch.setattr(lifecycle, '_is_local', lambda host: False)
    calls = []
    def runner(config, host, script, timeout=60):
        calls.append(script)
        return run(script)
    monkeypatch.setattr(lifecycle, 'run_on_host', runner)
    assert lifecycle.sync_package(config, host) is True
    assert len(calls) == 1
    assert (site / 'keep.py').read_text() == 'original release\n'
    assert (base / 'src').is_symlink()


@pytest.mark.parametrize('damage', ['unit', 'token', 'outside', 'dropin', 'missing', 'parent'])
def test_foreign_layout_refused_before_sync(tmp_path, monkeypatch, damage):
    config, host, base, site, unit = layout(tmp_path, monkeypatch)
    if damage == 'unit':
        unit.write_text(unit.read_text().replace('RestartSec=3', 'RestartSec=4'))
    elif damage == 'token':
        unit.write_text(unit.read_text().replace(TOKEN, 'u-ffffffffffff-ffffffffffffffff'))
    elif damage == 'outside':
        (base / 'src').unlink()
        (base / 'src').symlink_to(tmp_path)
    elif damage == 'dropin':
        directory = unit.parent / 'dgxm-worker.service.d'
        directory.mkdir()
        (directory / 'foreign.conf').write_text('[Service]\n')
    elif damage == 'missing':
        unit.unlink()
    else:
        site.rename(base / 'saved')
        site.symlink_to(base / 'saved')
    result = run(managed_source_probe(config, host))
    assert result.returncode != 0 and result.stdout.strip() == 'FAILED_SOURCE_LAYOUT'
    monkeypatch.setattr(lifecycle, '_is_local', lambda host: False)
    monkeypatch.setattr(lifecycle, 'run_on_host', lambda c, h, script, timeout=60: run(script))
    assert lifecycle.sync_package(config, host) is False
    assert (base / 'src').is_symlink()


def test_managed_start_uses_installed_slot_not_checkout(tmp_path, monkeypatch):
    config, host, base, _site, unit = layout(tmp_path, monkeypatch, absolute=True)
    result = run(managed_start_guard(config, host, '/unrelated/checkout') + '\nprintf "%s" "$DGXM_START_SOURCE"')
    assert result.returncode == 0
    assert result.stdout == str(base / 'src')
    unit.unlink()
    result = run(managed_start_guard(config, host, '/unrelated/checkout') + '\necho STARTED')
    assert result.returncode != 0 and 'STARTED' not in result.stdout


def test_plain_source_is_still_syncable(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    host = HostConfig(name='worker', address='tcp://10.0.0.2:26600')
    config = ClusterConfig(hosts=(host,), python_bin=sys.executable)
    assert run(managed_source_probe(config, host)).stdout.strip() == 'PLAIN'


def test_managed_renderer_needs_no_locality(monkeypatch):
    monkeypatch.setattr(lifecycle_systemd, 'is_local', lambda h: pytest.fail('no lookup'))
    host = HostConfig(name='worker', address='tcp://10.0.0.2:26600')
    text = lifecycle_systemd.systemd_worker_unit(ClusterConfig(), host, managed_source=True, update_token=TOKEN)
    assert text.startswith(f'# dgxm-update-token={TOKEN}\n')
    assert 'ExecStartPre=' in text
    assert 'PYTHONPATH=%h/.local/share/dgx-monarch/src' in text


def test_unknown_locality_never_ssh_falls_back(monkeypatch):
    monkeypatch.setattr(lifecycle, '_is_local', lambda h: None)
    monkeypatch.setattr(lifecycle.subprocess, 'run', lambda *a, **kw: pytest.fail('no transport'))
    host = HostConfig(name='worker', address='tcp://10.0.0.2:26600')
    with pytest.raises(OSError, match='locality'):
        lifecycle.run_on_host(ClusterConfig(), host, 'echo mutation', require_known_locality=True)
    assert lifecycle.sync_package(ClusterConfig(), host) is None


def test_verified_pair_allows_local_but_refuses_other_user_and_aliases(monkeypatch):
    monkeypatch.setattr(lifecycle_host.socket, 'getaddrinfo', lambda *a: [])
    local = HostConfig(name='local', address='tcp://10.0.0.1:26600')
    remote = HostConfig(name='remote', address='tcp://10.0.0.2:26600')
    def probe(host):
        return host.name == 'local'
    assert lifecycle_host.remote_colocation_error((local, remote), probe) is None
    alias = HostConfig(name='alias', address='tcp://10.0.0.2:26601')
    assert 'duplicate' in lifecycle_host.remote_colocation_error((remote, alias), probe)
    other = HostConfig(name='local', address=local.address, ssh_user=pwd.getpwuid(os.geteuid()).pw_name + '-other')
    assert 'different SSH user' in lifecycle_host.remote_colocation_error((other, remote), probe)


def test_hardened_update_unit_preserves_isolation(tmp_path, monkeypatch):
    config, host, _base, _site, unit = layout(tmp_path, monkeypatch)
    text = lifecycle_systemd.systemd_worker_unit(
        config, host, managed_source=True, update_token=TOKEN,
        site_packages='/opt/worker/lib/python3.12/site-packages')
    unit.write_text(text)
    assert '"-i"' in text and '"-S" "-P" "-s" "-B"' in text
    assert 'UnsetEnvironment=' in text and 'KillMode=control-group' in text
    assert run(managed_source_probe(config, host)).stdout.strip() == 'MANAGED'
    unit.write_text(text.replace('"-S" ', ''))
    assert run(managed_source_probe(config, host)).returncode != 0


def test_setup_slot_is_authenticated(tmp_path, monkeypatch):
    config, host, base, site, unit = layout(tmp_path, monkeypatch)
    setup = 's-0123456789abcdef-1'
    new = base / 'releases' / setup
    site.parent.rename(new)
    (base / 'src').unlink()
    (base / 'src').symlink_to(f'releases/{setup}/site')
    text = lifecycle_systemd.systemd_worker_unit(config, host, managed_source=True)
    lines = text.splitlines()[1:]
    fence = next(line for line in lines if line.startswith('ExecStartPre='))
    lines.remove(fence)
    index = next(i for i, line in enumerate(lines) if line.startswith('ExecStart='))
    lines.insert(index, fence)
    lines.insert(lines.index('RestartSec=3') + 1, 'TimeoutStopSec=20')
    text = '\n'.join([f'# dgxm-setup-token=s-0123456789abcdef ordinal=1 source={"0" * 64}', *lines]) + '\n'
    text = text.replace('PYTHONPATH=%h/.local/share/dgx-monarch/src', f'PYTHONPATH=%h/.local/share/dgx-monarch/releases/{setup}/site')
    unit.write_text(text)
    assert run(managed_source_probe(config, host)).stdout.strip() == 'MANAGED'
    unit.write_text(text.replace('RestartSec=3', 'RestartSec=99'))
    assert run(managed_source_probe(config, host)).returncode != 0


def test_unknown_aliases_and_multiple_local_hosts_are_refused(monkeypatch):
    first = HostConfig(name='first', address='tcp://10.0.0.1:26600')
    second = HostConfig(name='second', address='tcp://10.0.0.2:26600')
    assert 'definite' in lifecycle_host.remote_colocation_error((first, second), lambda h: None)
    assert 'duplicate' in lifecycle_host.remote_colocation_error((first, second), lambda h: True)
    monkeypatch.setattr(lifecycle_host.socket, 'getaddrinfo', lambda *a: [(0, 0, 0, '', ('10.2.3.4', 0))])
    assert 'duplicate' in lifecycle_host.remote_colocation_error((first, second), lambda h: False)


def test_ordinary_up_keeps_release_slot(tmp_path, monkeypatch):
    config, _host, _base, site, _unit = layout(tmp_path, monkeypatch)
    from dataclasses import replace
    config = replace(config, transport_security='trusted_fabric')
    monkeypatch.setattr(lifecycle, '_is_local', lambda host: True)
    scripts = []
    def runner(config, host, script, timeout=60):
        scripts.append(script)
        if len(scripts) == 1:
            return run(script)
        assert 'DGXM_START_SOURCE=' in script
        return subprocess.CompletedProcess([], 0, 'STARTED_SYSTEMD\n', '')
    monkeypatch.setattr(lifecycle, 'run_on_host', runner)
    monkeypatch.setattr(lifecycle.actor_sweep, 'report_result', lambda *a: True)
    assert lifecycle.up(config) is True
    assert len(scripts) == 2
    assert (site / 'keep.py').read_text() == 'original release\n'


def external_setup_layout(tmp_path, monkeypatch):
    import hashlib
    import json

    from dgx_monarch.cli.setup_services import SetupServiceRequest
    from dgx_monarch.cli.setup_services_platform import SystemSetupServiceOps

    config, host, base, _site, unit = layout(tmp_path, monkeypatch)
    package = tmp_path / 'checkout with % and space/src/dgx_monarch'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('# fixture\n')
    token = 's-0123456789abcdef'  # noqa: S105 - fixture release identifier
    digest = '0' * 64
    request = SetupServiceRequest(config, package, digest, True, True)
    ops = SystemSetupServiceOps(request, local_detector=lambda h: True, token_factory=lambda: token)
    text = ops._unit(host, 1)
    unit.write_text(text)
    (base / 'src').unlink()
    (base / 'src').symlink_to(package.parent)
    txn = tmp_path / '.local/state/dgx-monarch/setup' / f'{token}-1'
    txn.mkdir(parents=True)
    info = txn.stat()
    common = {'token': token, 'ordinal': 1, 'source_manifest': digest,
              'txn_dev': info.st_dev, 'txn_ino': info.st_ino, 'managed_source': False}
    transaction = {**common, 'link_target': str(package.parent),
                   'unit_sha256': hashlib.sha256(text.encode()).hexdigest()}
    proof = {**common, 'site_dev': package.parent.stat().st_dev, 'site_ino': package.parent.stat().st_ino,
             'package_dev': package.stat().st_dev, 'package_ino': package.stat().st_ino}
    (txn / 'transaction.json').write_text(json.dumps(transaction))
    (txn / 'verified.json').write_text(json.dumps(proof))
    for path in tmp_path.rglob('*'):
        if not path.is_symlink():
            path.chmod(0o700 if path.is_dir() else 0o600)
    return config, host, package, unit, txn


def test_local_setup_external_checkout_remains_usable(tmp_path, monkeypatch):
    config, host, package, _unit, _txn = external_setup_layout(tmp_path, monkeypatch)
    result = run(managed_source_probe(config, host))
    assert result.returncode == 0 and result.stdout.strip() == 'MANAGED'
    monkeypatch.setattr(lifecycle, '_is_local', lambda h: True)
    monkeypatch.setattr(lifecycle, 'run_on_host', lambda c, h, script, timeout=60: run(script))
    assert lifecycle.sync_package(config, host) is True
    assert (package / '__init__.py').read_text() == '# fixture\n'


@pytest.mark.parametrize('damage', ['record', 'inode', 'unit', 'link', 'missing', 'symlink_record'])
def test_local_setup_external_checkout_needs_exact_proof(tmp_path, monkeypatch, damage):
    import json

    config, host, package, unit, txn = external_setup_layout(tmp_path, monkeypatch)
    if damage == 'record':
        proof = json.loads((txn / 'verified.json').read_text())
        proof['managed_source'] = True
        (txn / 'verified.json').write_text(json.dumps(proof))
    elif damage == 'inode':
        package.rename(package.with_name('old_package'))
        package.mkdir(mode=0o700)
    elif damage == 'unit':
        unit.write_text(unit.read_text().replace('RestartSec=3', 'RestartSec=4'))
    elif damage == 'link':
        link = tmp_path / '.local/share/dgx-monarch/src'
        link.unlink()
        link.symlink_to(tmp_path)
    elif damage == 'missing':
        (txn / 'transaction.json').unlink()
    else:
        path = txn / 'verified.json'
        path.rename(txn / 'saved.json')
        path.symlink_to(txn / 'saved.json')
    result = run(managed_source_probe(config, host))
    assert result.returncode != 0 and result.stdout.strip() == 'FAILED_SOURCE_LAYOUT'


@pytest.mark.parametrize(
    'directory,permissions,owner,gid,group_name,members,other_primary,accepted',
    [
        (True, 0o775, 1000, 1000, 'alice', (), False, True),
        (True, 0o775, 1000, 1000, 'alice', ('alice',), False, True),
        (True, 0o775, 1000, 1000, 'alice', ('bob',), False, False),
        (True, 0o775, 1000, 1000, 'alice', (), True, False),
        (True, 0o775, 1000, 1000, 'shared', (), False, False),
        (True, 0o775, 1000, 1001, 'alice', (), False, False),
        (True, 0o775, 1001, 1000, 'alice', (), False, False),
        (True, 0o777, 1000, 1000, 'alice', (), False, False),
        (False, 0o660, 1000, 1000, 'alice', (), False, False),
        (False, 0o600, 1000, 1000, 'alice', (), False, True),
    ],
)
def test_managed_ownership_private_primary_group_only(
    directory, permissions, owner, gid, group_name, members, other_primary, accepted,
):
    import shlex
    import stat
    from types import SimpleNamespace

    host = HostConfig(name='worker', address='tcp://10.0.0.2:26600')
    script = managed_source_probe(ClusterConfig(python_bin=sys.executable), host)
    code = shlex.split(script.partition(' -c ')[2])[0]
    namespace = {}
    exec(compile(code.partition('try: verdict()')[0], '<fixture ownership probe>', 'exec'), namespace)  # noqa: S102 - definitions from our generated probe
    users = [SimpleNamespace(pw_uid=1000, pw_gid=1000)]
    if other_primary:
        users.append(SimpleNamespace(pw_uid=1001, pw_gid=1000))
    namespace['os'] = SimpleNamespace(geteuid=lambda: 1000, getegid=lambda: 1000)
    namespace['pwd'] = SimpleNamespace(
        getpwuid=lambda uid: SimpleNamespace(pw_name='alice'), getpwall=lambda: users,
    )
    namespace['grp'] = SimpleNamespace(
        getgrgid=lambda group: SimpleNamespace(gr_name=group_name, gr_mem=members),
    )
    info = SimpleNamespace(st_uid=owner, st_gid=gid,
                           st_mode=(stat.S_IFDIR if directory else stat.S_IFREG) | permissions)
    path = SimpleNamespace(lstat=lambda: info)
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if accepted:
        namespace['safe'](path, kind)
    else:
        with pytest.raises(ValueError, match='ownership'):
            namespace['safe'](path, kind)
    info.st_uid = 1000
    info.st_mode = stat.S_IFLNK | 0o700
    with pytest.raises(ValueError, match='file type'):
        namespace['safe'](path, kind)


def test_managed_slot_with_private_group_parent_is_usable(tmp_path, monkeypatch):
    import grp

    user = pwd.getpwuid(os.geteuid()).pw_name
    group = grp.getgrgid(os.getegid())
    if group.gr_name != user or set(group.gr_mem) - {user} or any(
        entry.pw_uid != os.geteuid() for entry in pwd.getpwall() if entry.pw_gid == os.getegid()
    ):
        pytest.skip('integration fixture needs a private primary group')
    config, host, base, _site, _unit = layout(tmp_path, monkeypatch)
    base.chmod(0o775)
    result = run(managed_source_probe(config, host))
    assert result.returncode == 0 and result.stdout.strip() == 'MANAGED'
    base.chmod(0o777)
    assert run(managed_source_probe(config, host)).returncode != 0


def test_strict_managed_directory_needs_no_group_lookup():
    import shlex
    import stat
    from types import SimpleNamespace

    host = HostConfig(name='worker', address='tcp://10.0.0.2:26600')
    script = managed_source_probe(ClusterConfig(python_bin=sys.executable), host)
    code = shlex.split(script.partition(' -c ')[2])[0]
    namespace = {}
    exec(compile(code.partition('try: verdict()')[0], '<fixture ownership probe>', 'exec'), namespace)  # noqa: S102 - definitions from our generated probe
    def unexpected(*args):
        pytest.fail('strict directories must not depend on group lookup')
    namespace['os'] = SimpleNamespace(geteuid=lambda: 1000, getegid=unexpected)
    namespace['grp'] = SimpleNamespace(getgrgid=unexpected)
    namespace['pwd'] = SimpleNamespace(getpwuid=unexpected, getpwall=unexpected)
    for permissions in (0o755, 0o700):
        info = SimpleNamespace(st_uid=1000, st_gid=1000, st_mode=stat.S_IFDIR | permissions)
        namespace['safe'](SimpleNamespace(lstat=lambda info=info: info), stat.S_ISDIR)
