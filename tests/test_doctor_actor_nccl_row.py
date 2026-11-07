"""The `actor NCCL backend` row: the torch actors import and the legacy NCCL pin.

The row builds the actor start environment with the launcher's own function,
so these cases change only the torch build string or that function.
"""
from __future__ import annotations

import pytest
import torch

from dgx_monarch import monarch_launch
from dgx_monarch.cli import actor_nccl_row, doctor
from dgx_monarch.cli.probe_certainty import FAIL, OK, WARN
from dgx_monarch.config import ClusterConfig, HostConfig
from test_doctor_hostname_row import _REMOTE_OK
from test_doctor_nccl_row import _run_doctor

LINE = "nccl_lib=23007 nccl_sha=a1b2c3d4e5f6\n"


def _cluster() -> ClusterConfig:
    return ClusterConfig(hosts=(HostConfig(name="head", address="tcp://10.0.0.2:26600"),
                                HostConfig(name="worker", address="tcp://10.0.0.3:26600")),
                         client_bind="tcp://10.0.0.1:0")


def test_the_launcher_pins_the_legacy_backend():
    assert monarch_launch.LEGACY_NCCL_ENV == {"TORCH_DIST_USE_NCCL2": "0"}


@pytest.mark.parametrize(("build", "effect"), [
    ("2.15.0+cu132", 'so "nccl" keeps torch\'s legacy ProcessGroupNCCL'),
    ("2.16.0", 'so "nccl" keeps torch\'s legacy ProcessGroupNCCL'),
    ("2.12.0+cu132", "which torch before 2.15 ignores"),
    ("2.test", 'so "nccl" keeps torch\'s legacy ProcessGroupNCCL'),
])
def test_the_pin_reads_ok_and_names_the_torch_build(build, effect):
    status, name, detail = actor_nccl_row.doctor_row(None, build)
    assert (status, name) == (OK, "actor NCCL backend")
    assert detail == (f"torch {build} here, and local actors import it; every actor starts "
                      f"with TORCH_DIST_USE_NCCL2=0, {effect}")


def test_a_cluster_points_at_each_hosts_env_row():
    status, _, detail = actor_nccl_row.doctor_row(_cluster(), "2.15.0+cu132")
    assert status == OK
    assert detail.startswith("torch 2.15.0+cu132 here, and each Worker host's actors import "
                             "the torch its env row shows; every actor starts with "
                             "TORCH_DIST_USE_NCCL2=0")


def test_the_row_reads_this_interpreters_torch_by_default():
    assert actor_nccl_row.doctor_row(None)[2].startswith(f"torch {torch.__version__} here")


@pytest.mark.parametrize(("build", "status"), [
    ("2.15.0+cu132", FAIL), ("2.test", FAIL), ("2.12.0+cu132", WARN)])
def test_a_missing_pin_fails_from_torch_2_15_and_warns_before(monkeypatch, build, status):
    original = monarch_launch.actor_environment

    def unpinned(config, host):
        environment = original(config, host)
        environment.pop("TORCH_DIST_USE_NCCL2")
        return environment

    monkeypatch.setattr(monarch_launch, "actor_environment", unpinned)
    row_status, _, detail = actor_nccl_row.doctor_row(None, build)
    assert row_status == status
    assert detail.endswith("the actor start environment lacks TORCH_DIST_USE_NCCL2=0, so torch "
                           "2.15 or later aborts the first process group under cudaMallocAsync "
                           "(docs/TROUBLESHOOTING.md #143)")


def test_an_environment_that_cannot_be_built_warns(monkeypatch):
    def broken(_config, _host):
        raise RuntimeError("policy")

    monkeypatch.setattr(monarch_launch, "actor_environment", broken)
    assert actor_nccl_row.doctor_row(None, "2.15.0+cu132") == (
        WARN, "actor NCCL backend",
        "unobserved: the actor start environment could not be built (RuntimeError); "
        "torch 2.15.0+cu132 here")


def test_doctor_prints_the_row_after_torch_cuda(monkeypatch, capsys):
    ok, output, _ = _run_doctor(monkeypatch, capsys, {"head": _REMOTE_OK + LINE,
                                                      "worker": _REMOTE_OK + LINE})
    assert ok is True
    lines = [line.strip() for line in output.splitlines()]
    index = next(i for i, line in enumerate(lines) if line.startswith("[ ok ] actor NCCL backend: "))
    assert lines[index - 1].startswith("[ ok ] torch CUDA: ")
    assert "every actor starts with TORCH_DIST_USE_NCCL2=0" in lines[index]


def test_the_row_has_a_bounded_updater_category():
    from dgx_monarch.cli import update_doctor_diagnostics

    assert update_doctor_diagnostics._CHECKS["actor NCCL backend"] == "actor_nccl_backend"
    assert doctor.actor_nccl_row is actor_nccl_row
