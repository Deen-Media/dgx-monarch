"""Regression tests for install and operator surfaces.

Install pins, doctor, templates, cluster.toml rendering, the verified stop, the
TUI, browser assets, the line ledger and mesh_safety's import rule.
"""
from __future__ import annotations

import ast
import importlib.metadata
import shutil
import subprocess
import sys
import tomllib
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from dgx_monarch import TORCHMONARCH_PIN
from dgx_monarch.cli import doctor, lifecycle
from dgx_monarch.cli.main import cmd_top
from dgx_monarch.config import (
    ClusterConfig,
    HostConfig,
    load_cluster_config,
    render_cluster_toml,
)
from dgx_monarch.tui.data import Poller, RingStore
from line_limit_helpers import CEILINGS, DEFAULT_CAP

REPO = Path(__file__).resolve().parents[1]


def test_torchmonarch_pin_stays_synchronized_across_install_surfaces():
    with (REPO / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    dependency = next(
        item for item in project["dependencies"] if item.startswith("torchmonarch==")
    )
    expected = f"torchmonarch=={TORCHMONARCH_PIN}"
    assert dependency == expected
    assert expected in (REPO / "requirements.txt").read_text()
    assert f'MONARCH_PIN="{expected}"' in (REPO / "scripts" / "setup_env.sh").read_text()


def test_source_install_requirements_match_project_runtime_dependencies():
    def normalized(items):
        requirements = []
        for item in items:
            requirement = Requirement(item)
            requirements.append((
                canonicalize_name(requirement.name),
                tuple(sorted(requirement.extras)),
                str(requirement.specifier),
                requirement.url or "",
                str(requirement.marker or ""),
            ))
        return sorted(requirements)

    with (REPO / "pyproject.toml").open("rb") as stream:
        project_dependencies = tomllib.load(stream)["project"]["dependencies"]
    source_install_dependencies = [
        line.strip()
        for line in (REPO / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert normalized(source_install_dependencies) == normalized(project_dependencies)


def test_doctor_treats_local_torchmonarch_version_skew_as_failure(monkeypatch, capsys):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.test"
    torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: "999.0")
    monkeypatch.setattr(
        doctor,
        "_frontend_skew_row",
        lambda: {"status": doctor._OK, "name": "frontend", "detail": "stub"},
    )
    monkeypatch.setattr(doctor, "_spark_health_rows", lambda _config: [])
    monkeypatch.setattr(doctor.socket, "gethostname", lambda: "driver")
    monkeypatch.setattr(doctor.socket, "gethostbyname", lambda _name: "10.0.0.1")

    assert not doctor.run_doctor(None)
    output = capsys.readouterr().out
    assert "installed 999.0" in output
    assert "1 failures" in output


def test_doctor_remote_version_token_does_not_accept_postrelease_suffix():
    fields = doctor._env_tokens(
        "torchmonarch=0.5.0.post1 torch=2.test comfy=True nccl_proto=unset"
    )
    assert fields["torchmonarch"] == "0.5.0.post1"
    assert fields["torchmonarch"] != TORCHMONARCH_PIN


def test_doctor_rejects_remote_torchmonarch_postrelease(monkeypatch, capsys):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.test"
    torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(importlib.metadata, "version", lambda _name: TORCHMONARCH_PIN)
    monkeypatch.setattr(
        doctor,
        "_frontend_skew_row",
        lambda: {"status": doctor._OK, "name": "frontend", "detail": "stub"},
    )
    monkeypatch.setattr(doctor, "_spark_health_rows", lambda _config: [])
    monkeypatch.setattr(doctor.socket, "gethostname", lambda: "driver")
    monkeypatch.setattr(doctor.socket, "gethostbyname", lambda _name: "10.0.0.1")
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/rsync")
    monkeypatch.setattr(
        doctor,
        "_mesh_health_row",
        lambda: {"status": doctor._OK, "name": "mesh health", "detail": "stub"},
    )
    monkeypatch.setattr(
        doctor,
        "_tcp_probe",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        doctor,
        "passive_worker_health",
        lambda *_args, **_kwargs: {
            "running": True, "listening": True, "healthy": True,
        },
    )
    output = (
        "torchmonarch=0.5.0.post1 torch=2.test comfy=True nccl_proto=unset\n"
        "fabric_iface=up rdma_active=1\n"
        "loop=running\n"
    )
    monkeypatch.setattr(
        doctor,
        "run_on_host",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, output, ""),
    )
    config = ClusterConfig(
        hosts=(HostConfig(name="worker", address="tcp://10.0.0.2:26600"),),
        client_bind="tcp://10.0.0.1:0",
        fabric_profile="generic-roce",
        transport_security="trusted_fabric",
        source="test.toml",
    )

    assert not doctor.run_doctor(config)
    assert "version skew vs driver pin" in capsys.readouterr().out


def test_templates_match_generator_and_carry_named_widget_maps():
    result = subprocess.run(
        [sys.executable, str(REPO / "tools" / "gen_templates.py"), "--check"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope="module")
def generated_templates(tmp_path_factory):
    """One generator run into a scratch directory, shared by the tests below.

    The generator is deterministic and each test mutates its own copy, so a
    second run per test would re-import torch and the node pack only to write
    the same bytes. Every ``--check`` stays a separate process: its exit code
    and stderr are the contract under test.
    """
    output_dir = tmp_path_factory.mktemp("generated") / "templates"
    generated = subprocess.run(
        [sys.executable, str(REPO / "tools" / "gen_templates.py"), "--output-dir", str(output_dir)],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert generated.returncode == 0, generated.stderr
    return output_dir


def test_template_check_rejects_orphaned_json_and_jpeg(tmp_path, generated_templates):
    output_dir = tmp_path / "templates"
    command = [
        sys.executable,
        str(REPO / "tools" / "gen_templates.py"),
        "--output-dir",
        str(output_dir),
    ]
    shutil.copytree(generated_templates, output_dir)
    for template in output_dir.glob("*.json"):
        template.with_suffix(".jpg").touch()

    checked = subprocess.run(
        [*command, "--check"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert checked.returncode == 0, checked.stderr

    orphan_json = output_dir / "orphan.json"
    orphan_json.write_text("{}", encoding="utf-8")
    checked = subprocess.run(
        [*command, "--check"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert checked.returncode == 1
    assert f"{orphan_json} (unexpected)" in checked.stderr
    orphan_json.unlink()

    orphan_jpeg = output_dir / "orphan.jpg"
    orphan_jpeg.touch()
    checked = subprocess.run(
        [*command, "--check"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert checked.returncode == 1
    assert f"{orphan_jpeg} (unexpected)" in checked.stderr
    orphan_jpeg.unlink()

    declared_jpeg = next(output_dir.glob("*.jpg"))
    declared_jpeg.unlink()
    checked = subprocess.run(
        [*command, "--check"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert checked.returncode == 1
    assert f"{declared_jpeg} (missing)" in checked.stderr


def test_testing_only_templates_stay_pinned_to_the_generator(tmp_path, generated_templates):
    # Sweep fixtures are generated separately from public templates and have
    # no preview cards. A JPEG in the fixture directory is unexpected.
    output_dir = tmp_path / "templates"
    command = [
        sys.executable,
        str(REPO / "tools" / "gen_templates.py"),
        "--output-dir",
        str(output_dir),
    ]
    shutil.copytree(generated_templates, output_dir)

    fresh = output_dir / "fixtures" / "workflows" / "generated"
    committed = REPO / "tests" / "fixtures" / "workflows" / "generated"
    assert {path.name for path in fresh.glob("*.json")} == {
        path.name for path in committed.glob("*.json")
    }
    for path in sorted(fresh.glob("*.json")):
        assert path.read_text() == (committed / path.name).read_text(), (
            f"{path.name} differs from the generator; run tools/gen_templates.py"
        )

    orphan_jpeg = fresh / (next(iter(sorted(fresh.glob("*.json")))).stem + ".jpg")
    orphan_jpeg.touch()
    checked = subprocess.run(
        [*command, "--check"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
    )
    assert checked.returncode == 1
    assert f"{orphan_jpeg} (unexpected)" in checked.stderr


def test_template_check_does_not_create_a_missing_output_directory(tmp_path):
    output_dir = tmp_path / "missing-templates"
    checked = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "gen_templates.py"),
            "--output-dir",
            str(output_dir),
            "--check",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert checked.returncode == 1
    assert not output_dir.exists()
    assert f"{output_dir} (missing directory)" in checked.stderr


def test_render_cluster_toml_escapes_interactive_strings(tmp_path):
    text = render_cluster_toml(
        hosts=[('ssh-alias"quoted', "10.0.0.2", 1)], client_ip="10.0.0.1",
        fabric_profile="generic-roce", master_addr="10.0.0.1",
        python_bin='/opt/venv "gpu"/bin/python', ssh_key='~/.ssh/key"name',
    )
    path = tmp_path / "cluster.toml"
    path.write_text(text)
    config = load_cluster_config(path)
    assert config.hosts[0].name == 'ssh-alias"quoted'
    assert config.python_bin == '/opt/venv "gpu"/bin/python'
    assert config.ssh_key == '~/.ssh/key"name'


def test_systemd_down_requires_verified_stop(monkeypatch):
    host = HostConfig(name="worker", address="tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,))
    scripts = []

    def run_host(_config, _host, script, timeout=30):
        scripts.append(script)
        return subprocess.CompletedProcess([], 0, "FAILED_SYSTEMD_ACTIVE\n", "")

    monkeypatch.setattr(lifecycle, "run_on_host", run_host)
    assert not lifecycle.down(config)
    assert "FAILED_SYSTEMD_ACTIVE" in scripts[0]
    assert "FAILED_NOHUP_ACTIVE" in scripts[0]


def test_poller_uses_explicit_config(monkeypatch, tmp_path):
    path = tmp_path / "chosen.toml"
    path.write_text(render_cluster_toml(
        hosts=[("worker", "10.1.2.3", 1)], client_ip="10.1.2.1",
        fabric_profile="generic-roce", master_addr="10.1.2.3",
    ))
    poller = Poller(config_path=str(path))
    assert poller.loops == ("tcp://10.1.2.3:26600",)


@pytest.mark.parametrize("interval", [0.0, -1.0, 1e-9, 1e-320, 1e308, float("inf"), float("nan")])
def test_unsafe_interval_is_rejected_before_tui_import(interval, capsys):
    rc = cmd_top(SimpleNamespace(
        interval=interval, host=None, replay=None, record=None, theme="spark", config=None,
    ))
    assert rc == 2
    assert "between 0.1 and 3600 seconds" in capsys.readouterr().err
    with pytest.raises(ValueError, match=r"between 0\.1 and 3600 seconds"):
        RingStore(interval=interval)


@pytest.mark.parametrize("interval", [0.1, 1.0, 3600.0])
def test_ring_accepts_documented_poll_interval_bounds(interval):
    assert RingStore(interval=interval).interval == interval


def test_ring_replay_reports_bad_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"t": 1}\nnot-json\n')
    try:
        RingStore.load(str(path))
    except ValueError as exc:
        assert f"{path}:2" in str(exc)
    else:
        raise AssertionError("malformed replay should fail")


def test_browser_panel_has_single_flight_and_protected_action_header():
    source = (REPO / "web/js/dgx_monarch_panel.js").read_text()
    assert "refreshInFlight" in source
    assert '"X-DGXM-Action": "recycle"' in source
    assert "window.confirm" in source


def test_browser_listener_carries_both_driver_events_and_the_toast_opt_out():
    source = (REPO / "web/js/dgx_monarch.js").read_text()
    assert '"dgx-monarch.fault"' in source
    assert '"dgx-monarch.notice"' in source
    assert "detail.toast === false" in source
    assert source.count("app.extensionManager?.toast") == 2
    # A phase that runs for minutes holds its toast until the next notice
    # replaces it, and a non-PASS verdict is not styled as a success.
    assert "detail.sticky === true" in source
    assert 'detail.severity || "info"' in source
    assert "toast.remove(openNotice)" in source
    # Callers that are not the first-render spine title their own toast.
    assert 'detail.summary || "DGX Monarch: first render"' in source


def test_module_line_limit_exception_ledger():
    # The ledger is the numbers in tests/line_limit_helpers.py, which
    # tests/test_comfy_managed_pricing.py reads through ceiling_for. The
    # DESIGN.md section 7 table mirrors it by hand; no test reads that table.
    ceilings = CEILINGS
    src = REPO / "src" / "dgx_monarch"
    over = {}
    for path in src.rglob("*.py"):
        lines = len(path.read_text().splitlines())
        relative = path.relative_to(src).as_posix()
        if lines > DEFAULT_CAP:
            over[relative] = lines
    assert set(over) == set(ceilings), (
        f"line-limit exception ledger drift: actual={over}, ledger={ceilings}"
    )
    assert all(over[name] <= ceiling for name, ceiling in ceilings.items()), (
        f"line-limit ceiling exceeded: actual={over}, ceilings={ceilings}"
    )


def test_mesh_safety_remains_a_leaf_without_mesh_imports():
    path = REPO / "src" / "dgx_monarch" / "mesh_safety.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.append(node.module)
                imported.extend(
                    f"{node.module}.{alias.name}" for alias in node.names)
            else:
                imported.extend(alias.name for alias in node.names)
    assert "mesh" not in imported
    assert "dgx_monarch.mesh" not in imported
