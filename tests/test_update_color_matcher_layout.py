"""Narrow color-matcher bundled-test layout compatibility, without external reads."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import sys

import pytest

from dgx_monarch.cli import update_payload_ownership as ownership
from dgx_monarch.cli.update_driver_pin import _manifest_digest, _scan_tree


def record(path, data):
    return [path, 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('='), str(len(data))]


def write_records(info, rows):
    out = io.StringIO()
    csv.writer(out).writerows(rows)
    (info / 'RECORD').write_text(out.getvalue())


@pytest.fixture
def layout(tmp_path, monkeypatch):
    site = tmp_path / 'site'
    site.mkdir()
    pinned = {'monarch/__init__.py': b'', 'monarch/runtime.py': b'pinned runtime',
              'tests/test_cuda.py': b'pinned CUDA tests', 'tests/conftest.py': b'pinned fixtures'}
    for name, data in pinned.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    expected = {name: hashlib.sha256(data).hexdigest() for name, data in pinned.items()}
    assets = {name: ('fixture ' + name).encode() for name in ownership._COLOR_MATCHER_DATA}
    # Small fixtures exercise the same binding as the three published PNG records.
    monkeypatch.setattr(ownership, '_COLOR_MATCHER_DATA', {
        name: tuple(record('', data)[1:]) for name, data in assets.items()
    })
    info = site / 'color_matcher-0.6.0.dist-info'
    info.mkdir()
    metadata = b'Metadata-Version: 2.1\nName: color-matcher\nVersion: 0.6.0\n'
    (info / 'METADATA').write_bytes(metadata)
    rows = [record(info.name + '/METADATA', metadata)]
    files = {'tests/__init__.py': b'', 'tests/frames.py': b'foreign test helper',
             **{'tests/data/' + name: data for name, data in assets.items()}}
    for name, data in files.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        rows.append(record(name, data))
    rows.extend(record('../../../tests/data/' + name, data) for name, data in assets.items())
    rows.append(['../../../bin/__pycache__/cli.' + sys.implementation.cache_tag + '.pyc', '', ''])
    write_records(info, rows)
    return site, expected, info, rows


def test_complete_layout_keeps_pinned_digest_and_never_reads_external_paths(layout, monkeypatch):
    site, expected, _info, _rows = layout
    before = _manifest_digest(expected)
    read = ownership._read
    def in_site_only(path):
        assert path.is_relative_to(site)
        assert '..' not in path.parts
        return read(path)
    monkeypatch.setattr(ownership, '_read', in_site_only)
    assert _scan_tree(site, expected) == expected
    assert _manifest_digest(_scan_tree(site, expected)) == before


@pytest.mark.parametrize('change', ['wrong_owner', 'wrong_version', 'nonempty_init', 'init_hash', 'init_size',
                                   'asset_bytes', 'asset_record', 'external_hash', 'external_size',
                                   'extra_parent', 'site_reentry', 'absolute', 'embedded_parent',
                                   'other_png', 'other_cache_tag', 'cache_hash', 'duplicate_external',
                                   'duplicate_initializer', 'other_initializer', 'compiled_initializer'])
def test_layout_refuses_every_other_shape(layout, change):
    site, expected, info, rows = layout
    if change in ('wrong_owner', 'wrong_version'):
        metadata = (info / 'METADATA').read_bytes().replace(
            b'color-matcher' if change == 'wrong_owner' else b'0.6.0',
            b'other' if change == 'wrong_owner' else b'0.6.1')
        (info / 'METADATA').write_bytes(metadata)
        rows[0] = record(info.name + '/METADATA', metadata)
    elif change == 'nonempty_init':
        (site / 'tests/__init__.py').write_bytes(b'pass\n')
        rows[1] = record('tests/__init__.py', b'pass\n')
    elif change in ('init_hash', 'init_size'):
        rows[1][1 if change == 'init_hash' else 2] = 'invalid'
    elif change == 'asset_bytes':
        (site / 'tests/data/scotland_house.png').write_bytes(b'changed')
    elif change == 'asset_record':
        next(row for row in rows if row[0] == 'tests/data/scotland_house.png')[1] = 'invalid'
    elif change in ('external_hash', 'external_size'):
        next(row for row in rows if row[0] == '../../../tests/data/scotland_house.png')[1 if change == 'external_hash' else 2] = 'invalid'
    elif change in ('other_cache_tag', 'cache_hash'):
        rows[-1][0 if change == 'other_cache_tag' else 1] = '../../../bin/__pycache__/cli.other.pyc' if change == 'other_cache_tag' else 'sha256=invalid'
    elif change == 'duplicate_external':
        rows.append(list(rows[-1]))
    elif change == 'duplicate_initializer':
        rows.append(list(rows[1]))
    elif change in ('other_initializer', 'compiled_initializer'):
        name = 'monarch/new/__init__.py' if change == 'other_initializer' else 'tests/__init__.so'
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'')
        rows.append(record(name, b''))
    else:
        name = {'extra_parent': '../../../../tests/data/scotland_house.png',
                'site_reentry': '../../../lib/python3.12/site-packages/tests/data/scotland_house.png',
                'absolute': '/outside/scotland_house.png',
                'embedded_parent': 'tests/../monarch/runtime.py',
                'other_png': '../../../tests/data/other.png'}[change]
        rows.append(record(name, b''))
    write_records(info, rows)
    with pytest.raises((ownership.PayloadOwnershipError, OSError)):
        _scan_tree(site, expected)


@pytest.mark.parametrize('pinned_name', ['tests/__init__.py', 'tests.py', 'tests/__init__.so', 'tests/__init__.pyc'])
def test_pinned_test_initializer_or_module_never_gets_an_exception(layout, pinned_name):
    site, expected, _info, _rows = layout
    expected[pinned_name] = hashlib.sha256(b'').hexdigest()
    (site / pinned_name).write_bytes(b'')
    with pytest.raises(ownership.PayloadOwnershipError):
        _scan_tree(site, expected)


def test_initializer_symlink_and_duplicate_owner_are_refused(layout):
    site, expected, _info, _rows = layout
    path = site / 'tests/__init__.py'
    path.unlink()
    path.symlink_to(site / 'monarch/__init__.py')
    with pytest.raises((RuntimeError, OSError)):
        _scan_tree(site, expected)
    path.unlink()
    path.write_bytes(b'')
    other = site / 'other-1.dist-info'
    other.mkdir()
    metadata = b'Name: other\nVersion: 1\n'
    (other / 'METADATA').write_bytes(metadata)
    write_records(other, [record(other.name + '/METADATA', metadata), record('tests/__init__.py', b'')])
    with pytest.raises(ownership.PayloadOwnershipError):
        _scan_tree(site, expected)


def test_generic_namespace_guard_is_unchanged():
    expected = {'examples/demo.py': 'hash', 'tests/test_cuda.py': 'hash'}
    assert not ownership._namespace_path('examples/__init__.py', expected)
    assert not ownership._namespace_path('tests/__init__.py', expected)
    assert not ownership._empty_color_test_initializer('tests/__init__.py', b'', {'monarch/runtime.py': 'hash'})
