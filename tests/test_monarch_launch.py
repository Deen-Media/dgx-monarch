"""Launch isolation and readiness are tested without spawning native actors."""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest

from dgx_monarch import monarch_launch as launch
from dgx_monarch.config import ClusterConfig, HostConfig
from dgx_monarch.monarch_policy import MonarchPolicy, RdmaPolicy, policy_from_token


@dataclass
class Command:
    executable: str
    cwd: object
    args: list[str]
    env: dict
    inherit_env: bool = True


@pytest.fixture
def commands(monkeypatch):
    monkeypatch.setitem(sys.modules, "monarch._rust_bindings.monarch_hyperactor.host_mesh",
                        SimpleNamespace(BootstrapCommand=Command))


def config():
    return ClusterConfig(hosts=(HostConfig("a", "tcp://192.0.2.1:26600"),
                                HostConfig("b", "tcp://192.0.2.2:26600")),
                         client_bind="tcp://192.0.2.1:0", python_bin="~/worker env/bin/python",
                         monarch=MonarchPolicy(rdma=RdmaPolicy(
                             host_targets=(("a", "nic:mlx5_0"), ("b", "nic:mlx5_1")))))


def test_local_launch_has_empty_base_and_no_secret_or_injected_paths(commands, monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-copy")
    monkeypatch.setenv("LD_PRELOAD", "must-not-load")
    monkeypatch.setenv("PYTHONPATH", "/foreign/controller/path")
    monkeypatch.setenv("DGXM_DRIVER", "123:456")
    cmd = launch.actor_bootstrap_command(config(), False)({"gpus": 0})
    assert cmd.executable == sys.executable and cmd.inherit_env is False
    assert not {"AWS_SECRET_ACCESS_KEY", "LD_PRELOAD"} & cmd.env.keys()
    assert cmd.env["PYTHONPATH"] != "/foreign/controller/path"
    assert cmd.env["PYTHONPATH"].split(os.pathsep)[0] == cmd.env["DGXM_PYTHONPATH"]
    assert cmd.env["DGXM_DRIVER"] == "123:456"
    assert cmd.env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert "-B" in cmd.args and "-s" in cmd.args


def test_remote_command_resolves_per_point_policy_and_keeps_host_environment(commands):
    command = launch.actor_bootstrap_command(config(), True)
    for index, expected in enumerate(("nic:mlx5_0", "nic:mlx5_1")):
        cmd = command({"hosts": index, "gpus": 0})
        assert cmd.inherit_env is True and cmd.executable == "/bin/sh"
        assert config().python_bin in cmd.args
        assert not {"PYTHONPATH", "DGXM_PYTHONPATH", "HOME"} & cmd.env.keys()
        assert policy_from_token(cmd.env["DGXM_MONARCH_POLICY"]).rdma.target == expected
        assert not policy_from_token(cmd.env["DGXM_MONARCH_POLICY"]).rdma.host_targets


def test_remote_shell_expands_home_without_evaluating_arguments(tmp_path):
    target = tmp_path / 'worker $(touch SHOULD_NOT_EXIST) env'
    target.mkdir()
    executable = target / "python"
    executable.symlink_to(sys.executable)
    supplied = "~/" + str(executable.relative_to(tmp_path))
    result = subprocess.run(["/bin/sh", "-c", launch._REMOTE_EXEC, "dgxm-bootstrap",
                             supplied, "-c", "import sys; print(sys.argv[1])", "$(not-a-command)"],
                            env={"HOME": str(tmp_path)}, text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "$(not-a-command)"
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


def test_readiness_uses_one_budget_and_requires_endpoint_response(monkeypatch):
    from dgx_monarch import mesh_runtime
    ticks = iter((10.0, 12.0, 14.0))
    monkeypatch.setattr("time.monotonic", lambda: next(ticks))
    # Avoid diagnostics consuming the fake budget clock.
    from contextlib import nullcontext

    from dgx_monarch import runtime_diagnostics
    monkeypatch.setattr(runtime_diagnostics, "phase", lambda *_a: nullcontext())
    observed = []
    monkeypatch.setattr(mesh_runtime, "get_off_loop", lambda *args: observed.append(args))
    procs = SimpleNamespace(initialized="procs")
    workers = SimpleNamespace(initialized="actors", status=SimpleNamespace(call=lambda: "status"))
    launch.wait_initialized(procs, 20.0, "procs")
    launch.wait_worker_ready(workers, 20.0)
    assert [(future, remaining) for future, remaining, _ in observed] == [
        ("procs", 10.0), ("actors", 8.0), ("status", 6.0)]


def test_expired_readiness_never_touches_native_future(monkeypatch):
    monkeypatch.setattr("time.monotonic", lambda: 20.0)
    class Mesh:
        @property
        def initialized(self):
            pytest.fail("expired work must not start")
    with pytest.raises(TimeoutError, match="deadline"):
        launch.wait_initialized(Mesh(), 20.0, "unused")


@pytest.mark.parametrize("failed_phase", ["proc", "actor", "status"])
def test_readiness_failure_rolls_back_only_owned_mesh(monkeypatch, failed_phase):
    from contextlib import nullcontext

    from monarch._src.actor import proc_mesh

    from dgx_monarch import mesh_factory, runtime_diagnostics

    monkeypatch.setattr(runtime_diagnostics, "phase", lambda *_a: nullcontext())
    monkeypatch.setitem(sys.modules, "dgx_monarch.actor", SimpleNamespace(GPUWorker=object))
    callbacks = []
    monkeypatch.setattr(proc_mesh, "register_proc_mesh_spawn_callback", callbacks.append)
    monkeypatch.setattr(proc_mesh, "unregister_proc_mesh_spawn_callback", callbacks.remove)
    calls = []
    class Future:
        def __init__(self, name): self.name = name
        def get(self, timeout=None):
            calls.append((self.name, timeout))
            if self.name == failed_phase:
                raise TimeoutError("readiness failed")
    workers = SimpleNamespace(initialized=Future("actor"), status=SimpleNamespace(call=lambda: Future("status")))
    procs = SimpleNamespace(initialized=Future("proc"), spawn=lambda *_a: workers,
                            stop=lambda _reason: Future("stop"))
    hosts = SimpleNamespace(spawn_procs=lambda **_kw: procs)
    handoff = mesh_factory.FleetHandoff()
    with pytest.raises(mesh_factory.WorkerSpawnError) as caught:
        mesh_factory.spawn_worker_fleet(config(), True, 1, lambda *_a: hosts, None, handoff)
    assert isinstance(caught.value.original, TimeoutError)
    assert caught.value.rollback_confirmed and handoff.cleanup_confirmed
    assert handoff.fleet is None and callbacks == []
    assert [name for name, _ in calls][-1] == "stop"
    assert calls[-1][1] == 60


def test_worker_generators_encode_same_resolved_policy():
    import shlex

    from dgx_monarch.cli.lifecycle import _loop_args
    from dgx_monarch.cli.lifecycle_systemd import systemd_worker_unit

    cfg = config()
    host = cfg.hosts[1]
    args = launch.worker_policy_args(cfg, host)
    assert args == shlex.split(_loop_args(host, cfg))[-2:]
    assert policy_from_token(args[1]).rdma.target == "nic:mlx5_1"
    for site in (None, "/worker/site-packages"):
        unit = systemd_worker_unit(cfg, host, managed_source=True, site_packages=site)
        assert '"--runtime-policy" "' + args[1] + '"' in unit


def test_worker_token_omits_the_driver_only_readiness_budget():
    """Only the driver reads startup_timeout_s; editing it must not change the Worker unit."""
    cfg = config()
    budget = replace(cfg, monarch=replace(cfg.monarch, startup_timeout_s=240))
    for host in cfg.hosts:
        args = launch.worker_policy_args(budget, host)
        assert args == launch.worker_policy_args(cfg, host)
        assert policy_from_token(args[1]).startup_timeout_s == MonarchPolicy().startup_timeout_s
    metrics = replace(budget.monarch, diagnostics=replace(budget.monarch.diagnostics, metrics=True))
    assert launch.worker_policy_args(replace(budget, monarch=metrics), cfg.hosts[0]) != launch.worker_policy_args(cfg, cfg.hosts[0])


@pytest.fixture
def driver_steps(monkeypatch):
    """Record prepare_driver's steps; the real native import still runs."""
    from dgx_monarch import mesh, monarch_capabilities, monarch_diagnostics, monarch_policy
    events: list = []
    diagnostics_state: dict = {'metrics': False, 'traces': False, 'available': True}
    monkeypatch.setattr(monarch_policy, 'apply_policy',
                        lambda _policy, host='', **kw: events.append(('apply', host, kw['native_enabled'])))
    monkeypatch.setattr(monarch_diagnostics, 'configure_policy',
                        lambda *_a: events.append('diagnostics') or diagnostics_state)
    monkeypatch.setattr(mesh, 'install_fault_hook', lambda: events.append('hook'))
    monkeypatch.setattr(monarch_capabilities, 'assert_supported_runtime', lambda: events.append('admit'))
    monkeypatch.setattr(monarch_policy, 'activate_native_policy', lambda: events.append('activate'))
    return events, diagnostics_state


def test_driver_admission_precedes_native_policy_activation(driver_steps):
    events, _ = driver_steps
    launch.prepare_driver(config())
    # The fault hook waits for this first Monarch import and precedes every transport call.
    assert events == [('apply', 'a', False), 'diagnostics', 'hook', 'admit', 'activate']


def test_driver_resolves_its_own_host_target_like_the_qualifier(driver_steps):
    events, _ = driver_steps
    native = replace(config(), rdma_latent_return=True)
    launch.prepare_driver(native)
    assert events[0] == ('apply', 'a', True)
    launch.prepare_driver(replace(native, client_bind='tcp://192.0.2.2:0'))
    assert events[5] == ('apply', 'b', True)


def test_native_admission_refuses_any_host_before_driver_policy(driver_steps):
    from dgx_monarch.config import ClusterConfigError
    events, _ = driver_steps
    partial = replace(config(), rdma_latent_return=True, monarch=MonarchPolicy(rdma=RdmaPolicy(
        host_targets=(("a", "nic:mlx5_0"),))))
    with pytest.raises(ClusterConfigError, match=r"host 'b' resolves to upstream selection.*host_targets"):
        launch.prepare_driver(partial)
    assert events == []
    multi = replace(partial, monarch=MonarchPolicy(rdma=RdmaPolicy(max_nics_per_buffer=2)))
    with pytest.raises(ClusterConfigError, match="selected target"):
        launch.prepare_driver(multi)
    assert events == []


@pytest.mark.parametrize(("reason", "hint"), [
    ("otlp_restart_required", "restart ComfyUI"),
    ("ambient_otlp_endpoint_conflict", "OTEL_EXPORTER_OTLP_ENDPOINT"),
    ("instrument_setup_failed", "metrics = false"),
])
def test_driver_refuses_requested_diagnostics_it_cannot_start(driver_steps, reason, hint):
    from dgx_monarch.config import ClusterConfigError
    from dgx_monarch.monarch_policy import DiagnosticsPolicy
    events, state = driver_steps
    state.clear()
    state.update(available=False, reason=reason)
    requested = replace(config(), monarch=MonarchPolicy(diagnostics=DiagnosticsPolicy(metrics=True)))
    with pytest.raises(ClusterConfigError, match=reason) as caught:
        launch.prepare_driver(requested)
    assert hint in str(caught.value) and "[monarch.diagnostics]" in str(caught.value)
    assert events == [('apply', 'a', False), 'diagnostics']
    events.clear()
    launch.prepare_driver(config())  # nothing requested: the same state never refuses
    assert events[-1] == 'activate'


def test_local_candidate_overlay_is_explicit_and_ambient_paths_are_not(commands, monkeypatch, tmp_path):
    import importlib.metadata

    candidate = tmp_path / 'candidate' / 'site-packages'
    monkeypatch.setattr(importlib.metadata, 'distribution', lambda name: SimpleNamespace(locate_file=lambda _: candidate))
    monkeypatch.setenv('PYTHONPATH', '/unverified/site')
    cmd = launch.actor_bootstrap_command(config(), False)({'gpus': 0})
    assert cmd.env['PYTHONPATH'].split(os.pathsep) == [cmd.env['DGXM_PYTHONPATH'], str(candidate)]


def test_disabled_actor_export_removes_coordinator_endpoint_before_startup(commands, monkeypatch):
    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://127.0.0.1:4318')
    cmd = launch.actor_bootstrap_command(config(), True)({'hosts': 0, 'gpus': 0})
    assert cmd.env['OTEL_EXPORTER_OTLP_ENDPOINT'] is None
    assert cmd.inherit_env is True
