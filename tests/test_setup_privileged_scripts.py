"""Run the generated setup scripts with a fake in place of the request to the root-owned process inspector."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from dgx_monarch.cli import setup_process_inspection as inspection
from dgx_monarch.cli import setup_services_activation_scripts as activation
from dgx_monarch.cli import setup_services_scripts as scripts
from test_setup_services_scripts import TOKEN, _generation_path, _prepare, _run


def _mock_inspector(monkeypatch, *, unknown=False, unavailable=False):
    original = inspection.privileged_process_source

    def source(enabled):
        if not enabled:
            return original(False)
        replacement = '''
def _request_process_report(source_sha):
 if UNAVAILABLE: raise OSError('inspection unavailable')
 root=pathlib.Path(os.environ['DGXM_TEST_PROC'])
 workers=[]
 if (root/'123').is_dir():
  address_hash=hashlib.sha256(os.environ['DGXM_TEST_ADDRESS'].encode()).hexdigest()
  workers=[{'pid':123,'starttime':777,'address_hashes':[address_hash]}]
 return {'workers':workers,'actors':[],'unknown_count':UNKNOWN}
'''.replace("UNAVAILABLE", repr(unavailable)).replace("UNKNOWN", str(int(unknown)))
        return original(True) + replacement

    monkeypatch.setattr(scripts, "privileged_process_source", source)
    monkeypatch.setattr(activation, "privileged_process_source", source)


def _ownership(env, *, privileged):
    result = _run(scripts.build_ownership_script(
        sys.executable, env["DGXM_TEST_ADDRESS"], 1, local_source=True,
        privileged_process_inspection=privileged,
    ), env)
    assert result.returncode == 0, result.stderr
    return json.loads(next(line.removeprefix(scripts.STATE_MARKER)
                          for line in result.stdout.splitlines() if line.startswith(scripts.STATE_MARKER)))


def test_explicit_inspection_can_establish_otherwise_unreadable_absence(tmp_path, monkeypatch):
    env, _digest, _address, _candidate = _prepare(tmp_path, local=True)
    protected = Path(env["DGXM_TEST_PROC"]) / "456"
    protected.mkdir()
    (protected / "cmdline").write_bytes(b"protected-process\0")
    (protected / "environ").write_bytes(b"")
    (protected / "environ").chmod(0)
    if os.geteuid() == 0:
        pytest.skip("permission-denied fixture needs an unprivileged test user")
    assert _ownership(env, privileged=False)["actors_absent"] is None
    _mock_inspector(monkeypatch)
    observed = _ownership(env, privileged=True)
    assert observed["worker_absent"] is observed["actors_absent"] is True


@pytest.mark.parametrize("unavailable", [False, True])
def test_opted_in_inspection_never_falls_back_from_unknown(tmp_path, monkeypatch, unavailable):
    env, _digest, _address, _candidate = _prepare(tmp_path, local=True)
    assert _ownership(env, privileged=False)["actors_absent"] is True
    _mock_inspector(monkeypatch, unknown=True, unavailable=unavailable)
    observed = _ownership(env, privileged=True)
    assert observed["worker_absent"] is observed["actors_absent"] is None


def test_privileged_selection_is_bound_through_publication_and_compensation(tmp_path, monkeypatch):
    env, digest, address, _candidate = _prepare(tmp_path, local=False)
    _mock_inspector(monkeypatch)
    result = _run(scripts.build_activate_script(
        sys.executable, TOKEN, 1, digest,
        "[Unit]\nDescription=test\n[Service]\nExecStart=/bin/true\n", address,
        start_service=True, privileged_process_inspection=True,
    ), env)
    assert result.returncode == 0 and result.stdout.strip() == "PUBLISHED", result.stderr
    probe = _run(scripts.build_probe_activation_script(
        sys.executable, TOKEN, 1, digest, privileged_process_inspection=True,
    ), env)
    assert probe.stdout.strip() == "NEW", probe.stderr
    wrong_mode = _run(scripts.build_compensate_script(sys.executable, TOKEN, 1, digest), env)
    assert wrong_mode.stdout.strip() == "UNKNOWN"
    home = Path(env["HOME"])
    assert (home / ".local/share/dgx-monarch/src").is_symlink() and _generation_path(env).exists()
    _mock_inspector(monkeypatch, unavailable=True)
    expired = _run(scripts.build_compensate_script(
        sys.executable, TOKEN, 1, digest, privileged_process_inspection=True,
    ), env)
    assert expired.stdout.strip() == "UNKNOWN"
    assert (home / ".local/share/dgx-monarch/src").is_symlink() and _generation_path(env).exists()
    _mock_inspector(monkeypatch)
    settled = _run(scripts.build_compensate_script(
        sys.executable, TOKEN, 1, digest, privileged_process_inspection=True,
    ), env)
    assert settled.stdout.strip() == "COMPENSATED", settled.stderr
    assert not _generation_path(env).exists()
