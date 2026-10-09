"""Synthetic access checks and mocked foreground launch; no privileged server."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SPEC = importlib.util.spec_from_file_location('inspection_tool', Path(__file__).parents[1] / 'tools/process_inspection.py')
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def process(root, pid='123'):
    directory = root / pid
    directory.mkdir()
    (directory / 'stat').write_text('123 (synthetic) ' + ' '.join(['0'] * 19 + ['42']))
    (directory / 'status').write_text('Uid: ' + ' '.join([str(os.geteuid())] * 4) + '\n')
    (directory / 'cmdline').write_text('PRIVATE COMMAND')
    (directory / 'environ').write_text('PRIVATE TOKEN')
    return directory


def test_check_readable_without_content(tmp_path):
    process(tmp_path)
    result = tool.check_access(tmp_path)
    assert result == {'status': 'readable', 'checked': 1, 'denied': 0, 'changed_or_unknown': 0}
    assert 'PRIVATE' not in str(result)


def test_permission_denied_stays_unknown(tmp_path, monkeypatch):
    process(tmp_path)
    original = Path.open

    def denied(path, *args, **kwargs):
        if path.name == 'environ':
            raise PermissionError('denied')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', denied)
    result = tool.check_access(tmp_path)
    assert result['status'] == 'unknown'
    assert result['denied'] == 1


def test_changed_identity_stays_unknown(tmp_path, monkeypatch):
    process(tmp_path)
    monkeypatch.setattr(tool, 'starttime', Mock(side_effect=[42, 43]))
    assert tool.check_access(tmp_path)['changed_or_unknown'] == 1


def test_scan_bounds(tmp_path):
    process(tmp_path)
    assert tool.check_access(tmp_path, limit=0)['status'] == 'incomplete'
    assert tool.check_access(tmp_path, timeout=0)['status'] == 'incomplete'


def test_empty_is_unknown(tmp_path):
    assert tool.check_access(tmp_path)['status'] == 'unknown'


def test_malformed_stat(tmp_path):
    directory = process(tmp_path)
    (directory / 'stat').write_text('invalid')
    assert tool.check_access(tmp_path)['status'] == 'unknown'


def test_root_rejected(monkeypatch):
    monkeypatch.setattr(tool.os, 'geteuid', lambda: 0)
    with pytest.raises(SystemExit):
        tool.main(['--check'])


def test_no_sudo_during_check(monkeypatch, capsys):
    monkeypatch.setattr(tool.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(tool.os, 'getuid', lambda: 1000)
    monkeypatch.setattr(tool, 'check_access', lambda: {'status': 'readable', 'checked': 1, 'denied': 0, 'changed_or_unknown': 0})
    runner = Mock(side_effect=AssertionError('No subprocess needed'))
    monkeypatch.setattr(tool.subprocess, 'run', runner)
    assert tool.main([]) == 0
    runner.assert_not_called()
    assert 'does not establish process ownership or absence' in capsys.readouterr().out


def test_serve_requires_terminal(monkeypatch):
    monkeypatch.setattr(tool.sys.stdin, 'isatty', lambda: False)
    with pytest.raises(ValueError, match='interactive terminal'):
        tool.serve()


def test_foreground_sudo_shape(monkeypatch):
    monkeypatch.setattr(tool.sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr(tool.sys.stderr, 'isatty', lambda: True)
    monkeypatch.setattr(tool, 'source_hash', lambda: (Path('/reviewed/process_inspector.py'), 'a' * 64))
    # Synthetic metadata; never launch sudo or a privileged server.
    monkeypatch.setattr(Path, 'resolve', lambda self, **kwargs: self)
    monkeypatch.setattr(Path, 'stat', lambda self: SimpleNamespace(st_uid=0, st_mode=0o100755))
    runner = Mock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(tool.subprocess, 'run', runner)
    assert tool.serve() == 0
    argv = runner.call_args.args[0]
    assert argv == ['/usr/bin/sudo', '/bin/bash', '-s', '--', '/reviewed/process_inspector.py', 'a' * 64]
    kwargs = runner.call_args.kwargs
    assert kwargs['env'] == {'PATH': '/usr/bin:/bin', 'TERM': 'dumb'}
    assert 'sha256sum --check --status' in kwargs['input']
    assert '--lifetime 1800' in kwargs['input']
    assert 'install -o root -g root -m 0600' in kwargs['input']
    assert "trap 'rm -f" in kwargs['input']
    assert 'env -i' in kwargs['input']
    assert 'stdout' not in kwargs and 'stderr' not in kwargs


def test_group_membership_checks(monkeypatch):
    user = SimpleNamespace(pw_uid=1000, pw_gid=1000, pw_name='owner')
    monkeypatch.setattr(tool.pwd, 'getpwuid', lambda uid: user)
    monkeypatch.setattr(tool.pwd, 'getpwall', lambda: [user])
    monkeypatch.setattr(tool.grp, 'getgrgid', lambda gid: SimpleNamespace(gr_mem=[]))
    assert tool.private_group(1000, 1000)
    monkeypatch.setattr(tool.grp, 'getgrgid', lambda gid: SimpleNamespace(gr_mem=['other']))
    assert not tool.private_group(1000, 1000)
    monkeypatch.setattr(tool.grp, 'getgrgid', lambda gid: SimpleNamespace(gr_mem=[]))
    monkeypatch.setattr(tool.pwd, 'getpwall', lambda: [user, SimpleNamespace(pw_uid=1001, pw_gid=1000)])
    assert not tool.private_group(1000, 1000)


def test_source_write_permission(tmp_path, monkeypatch):
    source = tmp_path / 'source.py'
    source.write_text('pass\n')
    source.chmod(0o666)
    with pytest.raises(ValueError, match='unsafe'):
        tool.validate_path(source, os.getuid())
    source.chmod(0o664)
    monkeypatch.setattr(tool, 'private_group', lambda gid, uid: False)
    with pytest.raises(ValueError, match='shared group'):
        tool.validate_path(source, os.getuid())
    monkeypatch.setattr(tool, 'private_group', lambda gid, uid: True)
    tool.validate_path(source, os.getuid())


def test_source_hash_is_bounded_and_covers_exact_bytes(tmp_path, monkeypatch):
    import hashlib

    source = tmp_path / 'inspector.py'
    source.write_bytes(b'# reviewed source\n')
    monkeypatch.setattr(tool, 'SOURCE', source)
    validations = Mock()
    monkeypatch.setattr(tool, 'validate_path', validations)
    selected, digest = tool.source_hash()
    assert selected == source
    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert validations.call_args_list[0].args[0] == source
    assert validations.call_count == len(source.parents) + 1
    monkeypatch.setattr(tool, 'MAX_SOURCE', 2)
    with pytest.raises(ValueError, match='size'):
        tool.source_hash()


def test_no_arbitrary_source_argument(monkeypatch):
    monkeypatch.setattr(tool.os, 'geteuid', lambda: 1000)
    with pytest.raises(SystemExit):
        tool.main(['--serve', '--source', '/unreviewed.py'])


def test_mixed_uids_rejected(monkeypatch):
    monkeypatch.setattr(tool.os, 'getuid', lambda: 1000)
    monkeypatch.setattr(tool.os, 'geteuid', lambda: 1001)
    with pytest.raises(SystemExit):
        tool.main(['--check'])


def test_ctrl_c_has_no_traceback_or_cleanup_claim(monkeypatch, capsys):
    monkeypatch.setattr(tool.os, 'getuid', lambda: 1000)
    monkeypatch.setattr(tool.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(tool, 'serve', Mock(side_effect=KeyboardInterrupt))
    assert tool.main(['--serve']) == 130
    error = capsys.readouterr().err
    assert 'interrupted' in error
    assert 'Traceback' not in error and 'cleaned' not in error


def test_vanished_directory_is_dropped(tmp_path, monkeypatch):
    import shutil

    directory = process(tmp_path)
    process(tmp_path, '124')
    original = tool.process_identity

    def disappear(path):
        if path == directory:
            shutil.rmtree(directory)
            raise FileNotFoundError('gone')
        return original(path)

    monkeypatch.setattr(tool, 'process_identity', disappear)
    result = tool.check_access(tmp_path)
    assert result['status'] == 'readable'
    assert result['changed_or_unknown'] == 0
    assert result['checked'] == 1


def test_missing_file_in_present_directory_is_unknown(tmp_path):
    directory = process(tmp_path)
    (directory / 'environ').unlink()
    result = tool.check_access(tmp_path)
    assert result['status'] == 'unknown'
    assert result['changed_or_unknown'] == 1


def test_root_owned_same_user_process_is_checked(tmp_path, monkeypatch):
    process(tmp_path)
    uid = os.geteuid()
    original = tool.process_identity
    monkeypatch.setattr(tool, 'process_identity', lambda path: (0, original(path)[1]))
    assert uid != 0
    assert tool.check_access(tmp_path)['checked'] == 1


def test_unrelated_root_process_is_skipped(tmp_path, monkeypatch):
    process(tmp_path)
    monkeypatch.setattr(tool, 'process_identity', lambda path: (0, (0, 0, 0, 0)))
    result = tool.check_access(tmp_path)
    assert result['checked'] == 0 and result['denied'] == 0


def test_changed_uid_tuple_stays_unknown(tmp_path, monkeypatch):
    process(tmp_path)
    uid = os.geteuid()
    monkeypatch.setattr(tool, 'process_identity', Mock(side_effect=[(uid, (uid,) * 4), (uid, (uid, uid, 0, uid))]))
    assert tool.check_access(tmp_path)['changed_or_unknown'] == 1
