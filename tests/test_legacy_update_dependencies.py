"""Plain updates refuse missing candidate dependencies before touching any pin."""
from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import legacy_update, update_support
from dgx_monarch.cli import legacy_update_dependencies as deps
from dgx_monarch.cli.update_dependency_diagnostics import report_result
from dgx_monarch.config import ClusterConfig, HostConfig


def fake_download(monkeypatch, *, name="torchmonarch", version="0.7.0", requirements=""):
    calls = []
    distribution = importlib.metadata.distribution

    def installed(name):
        if name == "torchmonarch":
            raise importlib.metadata.PackageNotFoundError(name)
        return distribution(name)
    monkeypatch.setattr(importlib.metadata, "distribution", installed)

    def download(argv, **kwargs):
        calls.append(argv)
        assert argv[2:4] == ["pip", "download"]
        assert "--no-deps" in argv and "--only-binary=:all:" in argv
        target = Path(argv[argv.index("--dest") + 1]) / "candidate.whl"
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("torchmonarch-0.7.0.dist-info/METADATA",
                             f"Name: {name}\nVersion: {version}\nRequires-Python: >=3.11\n{requirements}")
        return subprocess.CompletedProcess(argv, 0, "", "private pip URL")

    monkeypatch.setattr(deps.subprocess, "run", download)
    return calls


def test_exact_candidate_missing_requirement_is_not_installed(monkeypatch):
    calls = fake_download(monkeypatch, requirements="Requires-Dist: dgxm-test-missing>=1\n")
    assert deps.candidate_errors("0.7.0", ["torchmonarch==0.7.0"]) == ["dgxm-test-missing: missing"]
    assert len(calls) == 1 and "install" not in calls[0]
    assert not Path(calls[0][calls[0].index("--dest") + 1]).exists()


@pytest.mark.parametrize("name,version", [("other", "0.7.0"), ("torchmonarch", "0.6.0")])
def test_candidate_metadata_must_match_exact_pin(monkeypatch, name, version):
    fake_download(monkeypatch, name=name, version=version)
    assert deps.candidate_errors("0.7.0", []) == ["candidate wheel could not be verified"]


def test_emitted_probe_reuses_checker_without_installed_dgxm(monkeypatch, capsys):
    fake_download(monkeypatch, requirements="Requires-Dist: dgxm-test-missing>=1\n")
    monkeypatch.setattr(sys, "argv", ["probe", "0.7.0", "[]"])
    source = deps.probe_source()
    assert "import dgx_monarch" not in source and "import torch" not in source
    with pytest.raises(SystemExit) as error:
        exec(source, {})  # noqa: S102 - execute the locally generated probe with fake pip.
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out) == ["dgxm-test-missing: missing"]


def test_download_failure_discards_exception_and_pip_details(monkeypatch):
    fake_download(monkeypatch)
    def fail(*args, **kwargs):
        raise OSError("https://private-token@private.invalid/secret")
    monkeypatch.setattr(deps.subprocess, "run", fail)
    assert deps.candidate_errors("0.7.0", []) == ["candidate dependency inspection failed"]


def test_same_pin_checks_installed_metadata_without_network_or_mutation(monkeypatch):
    installed = SimpleNamespace(version="0.7.0", read_text=lambda _name:
        "Name: torchmonarch\nVersion: 0.7.0\nRequires-Python: >=3.11\n")
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _name: installed)
    def forbidden(*_args, **_kwargs):
        pytest.fail("same pin must not download or install")
    monkeypatch.setattr(deps.subprocess, "run", forbidden)
    assert deps.candidate_errors("0.7.0", ["torchmonarch==0.7.0"]) == []


def test_all_interpreters_checked_and_host_failure_blocks(monkeypatch, capsys):
    config = ClusterConfig(hosts=[HostConfig("private-one", "tcp://192.0.2.1:26600"),
                                 HostConfig("private-two", "tcp://192.0.2.2:26600")])
    monkeypatch.setattr(deps.subprocess, "run", lambda *_a, **_kw: subprocess.CompletedProcess([], 0, "[]", ""))
    seen = []
    def remote(_config, host, source, **kwargs):
        seen.append(host.name)
        assert "pip" in source and "--no-deps" in source
        return subprocess.CompletedProcess([], 1, '["xxhash: missing"]', "private detail")
    monkeypatch.setattr(deps.lifecycle, "run_on_host", remote)
    assert not deps.preflight("0.7.0", [], config)
    assert seen == ["private-one", "private-two"]
    output = capsys.readouterr().err
    assert "host_1: xxhash: missing" in output and "host_2: xxhash: missing" in output
    assert "private" not in output


@pytest.mark.parametrize("payload", [None, {}, ["secret https://private.invalid"], [True],
                                      ["numpy: installed https://secret, requires >=1"], ["bad/name: missing"]])
def test_untrusted_dependency_details_are_not_disclosed(payload, capsys):
    assert not report_result(subprocess.CompletedProcess([], 1, json.dumps(payload), "secret stderr"), "driver")
    assert capsys.readouterr().err == "  driver: candidate dependency inspection failed\n"


def test_version_remediation_is_bounded_and_actionable(capsys):
    assert not report_result(subprocess.CompletedProcess([], 1,
        '["numpy: installed 1.25.2, requires >=1.26", "pyarrow: missing"]', ""), "host_1")
    assert "numpy: installed 1.25.2, requires >=1.26" in capsys.readouterr().err


@pytest.mark.parametrize("admitted", [False, True])
def test_plain_update_admission_precedes_pin_and_service_changes(monkeypatch, tmp_path, capsys, admitted):
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies=["torchmonarch==0.7.0"]\n')
    events = []
    config = ClusterConfig(hosts=[HostConfig("worker", "tcp://192.0.2.1:26600")])
    def preflight(pin, roots, selected):
        assert pin == "0.7.0" and roots == ["torchmonarch==0.7.0"] and selected is config
        events.append("preflight")
        return admitted
    monkeypatch.setattr(legacy_update, "dependency_preflight", preflight)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.6.0")
    def run(argv, **kwargs):
        assert events[0] == "preflight" and admitted
        if "pip" in argv:
            assert "--no-deps" in argv
        events.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(legacy_update.subprocess, "run", run)
    monkeypatch.setattr(legacy_update.lifecycle, "ensure_torchmonarch_pin",
                        lambda *_args: events.append("worker_pin") or True)
    result = legacy_update._run_locked(SimpleNamespace(host=None), repo=tmp_path,
        load_config=lambda _args: config, driver_probe=lambda _host: (None, True))
    assert result == (0 if admitted else 1)
    if not admitted:
        assert events == ["preflight"]
        assert "before pin installation or Worker changes" in capsys.readouterr().err
    else:
        assert "worker_pin" in events


def test_verified_checker_uses_same_local_safe_details(monkeypatch, capsys):
    def run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 1, '["xxhash: missing"]', "private")
    assert not update_support.candidate_dependencies_available(ClusterConfig(), [], {}, run)
    assert capsys.readouterr().err == "  driver: xxhash: missing\n"
