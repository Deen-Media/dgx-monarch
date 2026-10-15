"""Release slots across an upgrade: the Doctor row, the plain-update refusal and the remedies.

Guided setup puts remote Workers in ``releases/s-...`` slots and a verified
update puts every Worker in ``releases/u-...``. Plain ``dgxm update`` never
changes a slot (TROUBLESHOOTING #118), so Doctor compares each slot's package
version with the driver's, and plain update refuses before it fetches.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import TORCHMONARCH_PIN, __version__
from dgx_monarch.cli import doctor, legacy_update, lifecycle, pincheck, probe_certainty, release_slot
from dgx_monarch.cli.doctor_worker_context import worker_context, worker_release_row
from dgx_monarch.cli.lifecycle import _pybin_shell
from dgx_monarch.cli.lifecycle_managed import managed_source_probe
from dgx_monarch.cli.probe_certainty import FAIL, OK, WARN
from dgx_monarch.config import ClusterConfig, HostConfig, load_cluster_config
from test_doctor_listener_generation import _config, cluster_rig  # noqa: F401  (fixture re-exported)
from test_doctor_worker_context import generated_prefix
from test_legacy_update_dependencies import _checkout_behind_its_upstream, _git
from test_lifecycle_managed import TOKEN, external_setup_layout, layout
from test_lifecycle_managed import run as run_bash

SKEWED = "0.0.1"  # no release of this package carries this version
SETUP_SLOT = "s-0123456789abcdef-2"


def _write_version(site: Path, version: str) -> None:
    """The slot's package: the version line, then a line that exits if anything imports it."""
    (site / "dgx_monarch").mkdir(parents=True, exist_ok=True)
    (site / "dgx_monarch" / "__init__.py").write_text(
        f'__version__ = "{version}"\nraise SystemExit("the probe imported the slot package")\n')


def _repos(tmp_path: Path) -> tuple[Path, Path]:
    parent = tmp_path / "repos"
    parent.mkdir()
    return _checkout_behind_its_upstream(parent)


def _local_runner(calls: list[str] | None = None):
    """Run a host script with bash on this box, under the test's HOME."""
    real_run = subprocess.run

    def run(_config, _host, script, timeout=60):
        if calls is not None:
            calls.append(script)
        return real_run(["bash", "-s"], input=script, text=True, capture_output=True, timeout=timeout)
    return run


# --- the Doctor row --------------------------------------------------------------------


def test_release_row_fails_on_a_slot_whose_version_differs_from_the_driver():
    status, name, detail = worker_release_row("spark-b", [f"dgxm_worker_release=SLOT {TOKEN} {SKEWED}"])
    assert (status, name) == (FAIL, "spark-b worker release version")
    assert detail.startswith(f"spark-b runs dgx-monarch {SKEWED} from release slot {TOKEN}")
    assert f"the driver runs {__version__}" in detail
    assert "TROUBLESHOOTING.md #118" in detail and "`dgxm update --verify`" in detail


def test_release_row_passes_a_slot_at_the_driver_version():
    status, _name, detail = worker_release_row("spark-b", [f"dgxm_worker_release=SLOT {TOKEN} {__version__}"])
    assert status == OK
    assert detail == f"release slot {TOKEN} carries dgx-monarch {__version__}, as the driver does"


@pytest.mark.parametrize(("line", "status", "start"), [
    ("dgxm_worker_release=PLAIN", OK, "plain source directory"),
    ("dgxm_worker_release=OUTSIDE", OK, "the source link points outside releases/"),
    (f"dgxm_worker_release=SLOT {TOKEN} UNREADABLE", WARN, "unobserved:"),
    ("dgxm_worker_release=SLOT", WARN, "unobserved:"),
    (None, WARN, "unobserved:"),
])
def test_release_row_never_fails_a_plain_or_unread_host(line, status, start):
    lines = ["dgxm_worker_policy=PLAIN", *([line] if line else [])]
    got, _name, detail = worker_release_row("spark-b", lines)
    assert (got, detail.startswith(start)) == (status, True)


@pytest.mark.parametrize(("version", "status"), [(__version__, OK), (SKEWED, FAIL)])
def test_doctor_script_reads_an_update_slot_version_without_importing_it(tmp_path, monkeypatch, version, status):
    """Doctor's real host preamble reads the slot's package version beside the policy verdict."""
    config, host, _base, site, _unit = layout(tmp_path, monkeypatch)
    _write_version(site, version)
    result = run_bash(generated_prefix(config, host))
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines == ["dgxm_worker_policy=MANAGED", f"dgxm_worker_release=SLOT {TOKEN} {version}"]
    assert worker_release_row(host.name, lines)[0] == status


def test_the_probe_ignores_a_module_the_slot_site_would_shadow(tmp_path, monkeypatch):
    """Doctor exports PYTHONPATH to the slot site just before the probe; `-I` keeps it out.

    A slot's ``re.py`` (or ``ast.py``) would otherwise replace the standard
    module the probe imports, and could stop or fake the version line.
    """
    config, host, _base, site, _unit = layout(tmp_path, monkeypatch)
    _write_version(site, SKEWED)
    for name in ("re", "ast", "pathlib"):
        (site / f"{name}.py").write_text(f'raise SystemExit("the probe imported the slot {name}.py")\n')
    result = run_bash(generated_prefix(config, host))
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["dgxm_worker_policy=MANAGED", f"dgxm_worker_release=SLOT {TOKEN} {SKEWED}"]


def test_the_row_needs_only_what_a_1_1_0_guided_setup_left_and_no_config_key(tmp_path, monkeypatch):
    """A cluster.toml written for 1.1.0 and an `s-` slot holding only the package suffice.

    That is what the 1.1.0 updater leaves when it starts this release's Doctor
    after its pull: no unit, no dist-info, no release record, no new key.
    """
    home = tmp_path / "home"
    site = home / ".local/share/dgx-monarch/releases" / SETUP_SLOT / "site"
    _write_version(site, "1.1.0")
    (home / ".local/share/dgx-monarch/src").symlink_to(site)
    monkeypatch.setenv("HOME", str(home))
    toml = tmp_path / "cluster.toml"
    toml.write_text(
        '[cluster]\nclient_bind = "tcp://192.0.2.11:0"\nfabric_profile = "dgx-spark-pair"\n'
        'transport_security = "trusted_fabric"\nnccl_master_addr = "192.0.2.11"\n'
        f'nccl_master_port = 29777\npython = "{sys.executable}"\nssh_key = ""\n\n'
        '[[hosts]]\nname = "spark-b"\naddress = "tcp://192.0.2.12:26600"\ngpus = 1\n')
    config = load_cluster_config(toml)
    [host] = config.hosts
    assert release_slot.PROBE_COMMAND in worker_context(config, host)
    files = [path.relative_to(home).as_posix() for path in home.rglob("*") if path.is_file()]
    assert files == [f".local/share/dgx-monarch/releases/{SETUP_SLOT}/site/dgx_monarch/__init__.py"]
    result = run_bash(f"{_pybin_shell(config.python_bin)}\n{release_slot.PROBE_COMMAND}\n")
    assert result.stdout.splitlines() == [f"dgxm_worker_release=SLOT {SETUP_SLOT} 1.1.0"]
    assert __version__ != "1.1.0"
    status, _name, detail = worker_release_row(host.name, result.stdout.splitlines())
    assert status == FAIL and "spark-b runs dgx-monarch 1.1.0" in detail
    assert release_slot.release_slot_hosts(config, runner=_local_runner()) == ([(host, SETUP_SLOT)], [])


@pytest.mark.parametrize(("version", "status"), [(__version__, OK), (SKEWED, FAIL)])
def test_doctor_reports_the_release_row_for_each_host(cluster_rig, capsys, version, status):  # noqa: F811
    stdout = (f"torchmonarch={TORCHMONARCH_PIN} torch=2.test comfy=True nccl_proto=unset\n"
              "fabric_iface=up rdma_active=2\ndgxm_worker_policy=MANAGED\n"
              f"dgxm_worker_release=SLOT {TOKEN} {version}\n")
    cluster_rig.setattr(doctor, "run_on_host", lambda *_a, **_kw: subprocess.CompletedProcess([], 0, stdout, ""))
    rows = doctor._doctor_rows(_config("tcp://10.0.0.1:26600"))
    capsys.readouterr()
    [row] = [row for row in rows if row["name"] == "h1 worker release version"]
    assert row["status"] == status
    assert (row in doctor._failures(rows)) is (status == FAIL)
    # The env row's torchmonarch check is unchanged and still passes here.
    [env] = [row for row in rows if row["name"] == "h1 env"]
    assert env["status"] == OK


# --- the plain-update refusal ------------------------------------------------------------


def _plain_update(local: Path, config: ClusterConfig) -> int:
    return legacy_update._run_locked(
        SimpleNamespace(host=None), repo=local, load_config=lambda _args: config,
        driver_probe=lambda _host: (None, True))


def test_plain_update_refuses_a_release_slot_host_before_any_fetch(tmp_path, monkeypatch, capsys):
    config, host, _base, _site, _unit = layout(tmp_path / "home", monkeypatch)
    _origin, local = _repos(tmp_path)
    tracking, head = _git(local, "rev-parse", "origin/main"), _git(local, "rev-parse", "HEAD")
    monkeypatch.setattr(lifecycle, "run_on_host", _local_runner())
    monkeypatch.setattr(legacy_update, "dependency_preflight",
                        lambda *_args: pytest.fail("admission ran for a release-slot install"))

    assert _plain_update(local, config) == 2
    assert _git(local, "rev-parse", "origin/main") == tracking  # nothing was fetched
    assert _git(local, "rev-parse", "HEAD") == head
    err = capsys.readouterr().err
    assert f"{host.name} runs its Worker from release slot {TOKEN} (release_slot_worker)" in err
    assert "plain `dgxm update` cannot refresh a release slot" in err
    assert "`dgxm update --verify --target <ref>`" in err and "TROUBLESHOOTING.md #118" in err


def test_plain_update_passes_a_colocated_setup_link_that_reads_managed(tmp_path, monkeypatch, capsys):
    """Guided setup's link to a colocated checkout reads MANAGED, yet the pull updates it."""
    config, host, _package, _unit, _txn = external_setup_layout(tmp_path / "home", monkeypatch)
    assert run_bash(managed_source_probe(config, host)).stdout.strip() == "MANAGED"
    _origin, local = _repos(tmp_path)
    tracking = _git(local, "rev-parse", "origin/main")
    calls: list[str] = []
    monkeypatch.setattr(lifecycle, "run_on_host", _local_runner(calls))
    admitted = []
    monkeypatch.setattr(legacy_update, "dependency_preflight",
                        lambda pin, _roots, _config: admitted.append(pin) or False)

    assert _plain_update(local, config) == 1  # stops at admission, past the slot probe
    assert len(calls) == 1 and release_slot.PROBE_COMMAND in calls[0]
    assert admitted == ["0.7.0"]
    assert _git(local, "rev-parse", "origin/main") != tracking  # the fetch ran
    err = capsys.readouterr().err
    assert "release_slot" not in err and "stays at its current commit" in err


def _unreachable(*_args, **_kwargs):
    raise OSError("ssh: connect to host failed")


@pytest.mark.parametrize("answer", [
    _unreachable,
    # The configured interpreter is missing: bash exits 127 before the probe runs.
    lambda *_a, **_kw: subprocess.CompletedProcess([], 127, "", "python3: No such file or directory"),
    # The probe exited 0 but printed no verdict.
    lambda *_a, **_kw: subprocess.CompletedProcess([], 0, "garbage\n", ""),
    # A verdict line on a failed exit is not trusted.
    lambda *_a, **_kw: subprocess.CompletedProcess([], 1, "dgxm_worker_release=PLAIN\n", "Traceback"),
], ids=["no-answer", "exit-127", "no-verdict", "failed-exit"])
def test_plain_update_refuses_an_unread_host_as_unknown(tmp_path, monkeypatch, capsys, answer):
    _origin, local = _checkout_behind_its_upstream(tmp_path)
    tracking = _git(local, "rev-parse", "origin/main")
    config = ClusterConfig(hosts=(HostConfig("spark-b", "tcp://192.0.2.12:26600"),))
    monkeypatch.setattr(lifecycle, "run_on_host", answer)
    monkeypatch.setattr(legacy_update, "dependency_preflight",
                        lambda *_args: pytest.fail("admission ran with an unread host"))

    assert _plain_update(local, config) == probe_certainty.UNKNOWN_EXIT
    assert _git(local, "rev-parse", "origin/main") == tracking
    err = capsys.readouterr().err
    assert "the source link on spark-b could not be read (release_slot_unknown)" in err
    assert "release_slot_worker" not in err and "TROUBLESHOOTING.md #118" in err


def test_local_mode_plain_update_probes_no_host(tmp_path, monkeypatch):
    _origin, local = _checkout_behind_its_upstream(tmp_path)
    monkeypatch.setattr(legacy_update, "release_slot_hosts",
                        lambda *_args: pytest.fail("a configless update probed a host"))
    admitted = []
    monkeypatch.setattr(legacy_update, "dependency_preflight",
                        lambda pin, _roots, _config: admitted.append(pin) or False)
    assert _plain_update(local, ClusterConfig()) == 1
    assert admitted == ["0.7.0"]


# --- the pin skew warning ---------------------------------------------------------------


@pytest.mark.parametrize(("policy", "release", "in_slot"), [
    ("MANAGED", f"SLOT {TOKEN} 1.1.0", True),
    ("MANAGED", f"SLOT {SETUP_SLOT} 1.1.0", True),
    ("POLICY_STALE", f"SLOT {TOKEN} 1.1.0", True),
    # Guided setup's link to a colocated checkout reads MANAGED, yet plain update moves it.
    ("MANAGED", "OUTSIDE", False),
    ("PLAIN", "PLAIN", False),
])
def test_pin_warning_sends_only_a_release_slot_to_verify(monkeypatch, capsys, policy, release, in_slot):
    host = HostConfig(name="spark-b", address="tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    stdout = f"dgxm_worker_policy={policy}\ndgxm_worker_release={release}\n0.0.0rc0\n"
    monkeypatch.setattr(pincheck, "run_on_host", lambda *_a, **_kw: subprocess.CompletedProcess([], 0, stdout, ""))
    pincheck.warn_on_pin_mismatch(config, TORCHMONARCH_PIN)
    err = " ".join(capsys.readouterr().err.split())
    if in_slot:
        assert "the managed Worker release on spark-b carries torchmonarch 0.0.0rc0" in err
        assert "Fix: run `dgxm update --verify`" in err
        assert "a plain `dgxm update` never changes a Worker release slot" in err
        assert "TROUBLESHOOTING.md #118" in err
        assert "pulls the repo" not in err and "pip install" not in err
    else:
        assert "spark-b has torchmonarch 0.0.0rc0 installed" in err
        assert "Fix: run `dgxm update`, which pulls the repo" in err
        assert "with the python path from cluster.toml's [cluster] table" in err
        assert f"<python> -m pip install --no-deps torchmonarch=={TORCHMONARCH_PIN}" in err
        assert "--verify" not in err and "#118" not in err


# --- Kroma's cfg2 cell -------------------------------------------------------------------


def test_kroma_bf16_reaches_the_krea2_cfg2_row_and_the_matrix_says_untested():
    from dgx_monarch.topology import choose_auto_topology
    from test_docs_status_sync import (
        _AUTO_MARK,
        _DISTILLED_CFG1_ROWS,
        _MODE_COLUMN,
        _PENDING_MARK,
        _STARRED_MODE_GRANT,
        _support_matrix_rows,
    )

    # The vendor card allows CFG up to 1.5; a BF16 Kroma file reads family krea2.
    assert choose_auto_topology("krea2", "bf16", 1.5, 2, cfg_value=1.5).topology.cfg == 2
    assert choose_auto_topology("krea2", "bf16", 1.05, 2, cfg_value=1.5).topology.cfg == 1  # shipped 1024 graph
    assert "Kroma" not in _DISTILLED_CFG1_ROWS
    state, phrase = _STARRED_MODE_GRANT[("Kroma", "cfg2")]
    cells = _support_matrix_rows()["Kroma"]
    cell = cells[_MODE_COLUMN["cfg2"]]
    assert state == "unrun" and _AUTO_MARK in cell and _PENDING_MARK in cell
    assert phrase in cells[7] and "`uly2` keeps the tested path" in cells[7]
