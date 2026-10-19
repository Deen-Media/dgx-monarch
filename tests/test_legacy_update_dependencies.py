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
    # No host runs from a release slot; the slot probe runs before admission.
    monkeypatch.setattr(legacy_update, "release_slot_hosts",
                        lambda selected: events.append("slot_probe") or ([], []))
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "0.6.0")
    def run(argv, **kwargs):
        assert events[:2] == ["slot_probe", "preflight"] and admitted
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
        assert events == ["slot_probe", "preflight"]
        assert "before pin installation or Worker changes" in capsys.readouterr().err
    else:
        assert "worker_pin" in events


def test_verified_checker_uses_same_local_safe_details(monkeypatch, capsys):
    def run(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 1, '["xxhash: missing"]', "private")
    assert not update_support.candidate_dependencies_available(ClusterConfig(), [], {}, run)
    assert capsys.readouterr().err == "  driver: xxhash: missing\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _commit_file(repo: Path, name: str, text: str) -> None:
    (repo / name).write_text(text)
    _git(repo, "add", name)
    _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.test",
         "-c", "commit.gpgsign=false", "commit", "-q", "-m", name)


def _commit_pin(repo: Path, pin: str) -> None:
    _commit_file(repo, "pyproject.toml", f'[project]\ndependencies=["torchmonarch=={pin}"]\n')


def _checkout_at_its_upstream(tmp_path: Path) -> tuple[Path, Path]:
    """A clone at pin 0.6.0 whose upstream has not moved."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _commit_pin(origin, "0.6.0")
    local = tmp_path / "local"
    subprocess.run(["git", "clone", "-q", str(origin), str(local)], check=True,
                   capture_output=True)
    return origin, local


def _checkout_behind_its_upstream(tmp_path: Path) -> tuple[Path, Path]:
    """A clone at pin 0.6.0 whose upstream has since moved to 0.7.0."""
    origin, local = _checkout_at_its_upstream(tmp_path)
    _commit_pin(origin, "0.7.0")
    return origin, local


def _plain_update(local: Path) -> int:
    return legacy_update._run_locked(
        SimpleNamespace(host=None), repo=local, load_config=lambda _args: None,
        driver_probe=lambda _host: (None, True))


def _record_admission_merge_and_pip(monkeypatch, events: list, installed: str) -> None:
    """Admit every pin, run git for real, and record admission, merges and pip calls."""
    monkeypatch.setattr(legacy_update, "dependency_preflight",
                        lambda pin, _roots, _config: events.append(("admit", pin)) or True)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: installed)
    real_run = subprocess.run

    def run(argv, **kwargs):
        if argv[0] == "git":
            if "merge" in argv:
                events.append(("merge", argv[-1]))
            return real_run(argv, **kwargs)
        events.append(("pip", tuple(argv[3:5])))
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(legacy_update.subprocess, "run", run)


def test_plain_update_refusal_leaves_the_checkout_at_its_commit(monkeypatch, tmp_path, capsys):
    _origin, local = _checkout_behind_its_upstream(tmp_path)
    before = _git(local, "rev-parse", "HEAD")
    admitted = []
    monkeypatch.setattr(legacy_update, "dependency_preflight",
                        lambda pin, _roots, _config: admitted.append(pin) or False)

    assert _plain_update(local) == 1
    # Admission read the fetched commit's pin, not the working tree's.
    assert admitted == ["0.7.0"]
    assert _git(local, "rev-parse", "HEAD") == before
    assert "torchmonarch==0.6.0" in (local / "pyproject.toml").read_text()
    assert "stays at its current commit" in capsys.readouterr().err


def test_plain_update_fast_forwards_to_the_admitted_commit_only_after_admission(
        monkeypatch, tmp_path):
    origin, local = _checkout_behind_its_upstream(tmp_path)
    events: list = []
    _record_admission_merge_and_pip(monkeypatch, events, "0.7.0")

    assert _plain_update(local) == 0
    target = _git(origin, "rev-parse", "HEAD")
    assert _git(local, "rev-parse", "HEAD") == target
    assert events == [("admit", "0.7.0"), ("merge", target), ("pip", ("install", "-e"))]


def test_plain_update_at_its_upstream_checks_head_without_merging(monkeypatch, tmp_path, capsys):
    _origin, local = _checkout_at_its_upstream(tmp_path)
    before = _git(local, "rev-parse", "HEAD")
    events: list = []
    _record_admission_merge_and_pip(monkeypatch, events, "0.6.0")

    assert _plain_update(local) == 0
    assert _git(local, "rev-parse", "HEAD") == before
    assert events == [("admit", "0.6.0"), ("pip", ("install", "-e"))]
    assert "already contains its fetched upstream" in capsys.readouterr().out


def test_plain_update_with_local_commits_checks_head_not_the_older_upstream(
        monkeypatch, tmp_path, capsys):
    """`git pull --ff-only` leaves a checkout ahead of its upstream unmoved; so does this."""
    _origin, local = _checkout_at_its_upstream(tmp_path)
    _commit_pin(local, "0.7.0")
    before = _git(local, "rev-parse", "HEAD")
    assert _git(local, "rev-parse", "origin/main") != before
    events: list = []
    _record_admission_merge_and_pip(monkeypatch, events, "0.7.0")

    assert _plain_update(local) == 0
    # Admission read HEAD's pin, the commit that is installed, not the upstream's 0.6.0.
    assert events == [("admit", "0.7.0"), ("pip", ("install", "-e"))]
    assert _git(local, "rev-parse", "HEAD") == before
    assert "already contains its fetched upstream" in capsys.readouterr().out


@pytest.mark.parametrize("edit", ["unstaged", "staged", "assume-unchanged", "skip-worktree"])
def test_plain_update_refuses_a_local_pyproject_edit_the_fast_forward_would_keep(
        monkeypatch, tmp_path, capsys, edit):
    origin, local = _checkout_at_its_upstream(tmp_path)
    _commit_file(origin, "README.md", "the fetched commit leaves pyproject.toml alone\n")
    tracking = _git(local, "rev-parse", "origin/main")
    before = _git(local, "rev-parse", "HEAD")
    edited = '[project]\ndependencies=["torchmonarch==0.6.1"]\n'
    (local / "pyproject.toml").write_text(edited)
    if edit == "staged":
        _git(local, "add", "pyproject.toml")
    elif edit != "unstaged":
        _git(local, "update-index", f"--{edit}", "pyproject.toml")
    monkeypatch.setattr(legacy_update, "dependency_preflight",
                        lambda *_args: pytest.fail("admission ran over a local pyproject edit"))

    assert _plain_update(local) == 1
    assert _git(local, "rev-parse", "HEAD") == before
    assert _git(local, "rev-parse", "origin/main") == tracking  # refused before the fetch
    assert (local / "pyproject.toml").read_text() == edited
    assert "pyproject.toml has local changes" in capsys.readouterr().err
    # The escape is real: a fast-forward to this upstream keeps the edit in the tree.
    _git(local, "pull", "-q", "--ff-only")
    assert _git(local, "rev-parse", "HEAD") == _git(origin, "rev-parse", "HEAD")
    assert (local / "pyproject.toml").read_text() == edited


def test_plain_update_refuses_a_branch_without_upstream_before_admission(
        monkeypatch, tmp_path, capsys):
    _origin, local = _checkout_behind_its_upstream(tmp_path)
    _git(local, "checkout", "-q", "-b", "local-only")
    before = _git(local, "rev-parse", "HEAD")
    monkeypatch.setattr(legacy_update, "dependency_preflight",
                        lambda *_args: pytest.fail("admission ran without an upstream"))

    assert _plain_update(local) == 1
    assert _git(local, "rev-parse", "HEAD") == before
    assert "no single upstream" in capsys.readouterr().err


def test_plain_update_refuses_a_diverged_checkout_before_admission(
        monkeypatch, tmp_path, capsys):
    _origin, local = _checkout_behind_its_upstream(tmp_path)
    _commit_pin(local, "0.6.1")
    before = _git(local, "rev-parse", "HEAD")
    monkeypatch.setattr(legacy_update, "dependency_preflight",
                        lambda *_args: pytest.fail("admission ran for a diverged checkout"))

    assert _plain_update(local) == 1
    assert _git(local, "rev-parse", "HEAD") == before
    assert "diverged from its fetched upstream" in capsys.readouterr().err


def _run_plain_update(repo: Path, tmp_path: Path, driver_probe) -> int:
    """The whole plain update entry, with its lock kept under ``tmp_path``."""
    from dgx_monarch.cli.update_lock import UpdateLock

    return legacy_update.run(
        SimpleNamespace(host=None), load_config=lambda _args: None,
        driver_probe=driver_probe, repo=repo,
        lock_factory=lambda checkout: UpdateLock(checkout, state_root=tmp_path / "update-locks"))


def test_plain_update_refuses_a_linked_worktree_before_any_fetch(monkeypatch, tmp_path, capsys):
    _origin, local = _checkout_behind_its_upstream(tmp_path)
    tracking = _git(local, "rev-parse", "origin/main")
    worktree = tmp_path / "worktree"
    _git(local, "worktree", "add", "-q", str(worktree))
    assert (worktree / ".git").is_file()
    before = _git(worktree, "rev-parse", "HEAD")
    monkeypatch.setattr(legacy_update, "dependency_preflight",
                        lambda *_args: pytest.fail("admission ran for a linked worktree"))
    monkeypatch.setattr(legacy_update.subprocess, "run",
                        lambda *_args, **_kwargs: pytest.fail("git or pip ran for a linked worktree"))

    # The driver guard still comes first, as for every other refusal after it.
    assert _run_plain_update(worktree, tmp_path, lambda _host: (None, True)) == 1
    monkeypatch.undo()
    assert _git(local, "rev-parse", "origin/main") == tracking
    assert _git(worktree, "rev-parse", "HEAD") == before
    output = capsys.readouterr()
    assert output.out == ""
    assert f"{worktree} is a linked git worktree" in output.err
    # The remedy names what plain update acts on, since cd'ing elsewhere changes nothing.
    assert "the checkout this dgxm was installed from" in output.err
    assert "only a main checkout" in output.err
    assert "`dgxm update --verify`" in output.err
    assert "reinstall dgx-monarch from the main checkout" in output.err


@pytest.mark.parametrize("layout", ["no-git", "gitdir-file-without-commondir"])
def test_plain_update_without_a_git_directory_still_checks_the_working_tree(
        monkeypatch, tmp_path, capsys, layout):
    """A copy without Git, and a .git file that is not a linked worktree (a
    submodule's, say), keep the old path: no fetch, the working tree's pin."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text('[project]\ndependencies=["torchmonarch==0.6.0"]\n')
    if layout == "gitdir-file-without-commondir":
        (tmp_path / "modules" / "repo").mkdir(parents=True)
        (repo / ".git").write_text("gitdir: ../modules/repo\n")
    admitted = []
    monkeypatch.setattr(legacy_update, "dependency_preflight",
                        lambda pin, _roots, _config: admitted.append(pin) or False)
    monkeypatch.setattr(legacy_update.subprocess, "run",
                        lambda *_args, **_kwargs: pytest.fail("git or pip ran without a git directory"))

    assert _run_plain_update(repo, tmp_path, lambda _host: (None, True)) == 1
    assert admitted == ["0.6.0"]
    output = capsys.readouterr()
    assert "not a plain git checkout (no .git directory); skipping the fetch" in output.out
    assert "linked git worktree" not in output.err
