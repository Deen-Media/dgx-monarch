"""CPU-only contracts for the explicitly requested GPU smoke command."""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.cli import sol_smoke
from dgx_monarch.config import ClusterConfig, HostConfig


def _evidence(**changes):
    result = {"backend": "cute_sm120", "capability": [12, 1], "relative_l2": 0.003,
              "shape": [1, 4096, 1, 128], "dtype": "bfloat16", "scope": "dense-kernel-control"}
    result.update(changes)
    return result


def _completed(**changes):
    return subprocess.CompletedProcess([], 0, sol_smoke.RESULT_PREFIX + json.dumps(_evidence(**changes)), "")


def _config():
    return ClusterConfig(hosts=(HostConfig("head", "tcp://head:1"), HostConfig("peer", "tcp://peer:1")),
                         python_bin="~/comfyui-env/bin/python")


def test_explicit_maintenance_ack_required(monkeypatch):
    monkeypatch.setattr(sol_smoke, "run_on_host", lambda *a, **kw: pytest.fail("must not execute"))
    assert sol_smoke.cmd_sol_smoke(_config()) == 2


def test_every_configured_worker_is_checked_sequentially(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(sol_smoke, "run_on_host", lambda config, host, script, **kw:
                        calls.append((host.name, script, kw)) or _completed())
    assert sol_smoke.cmd_sol_smoke(_config(), idle_maintenance=True) == 0
    assert [call[0] for call in calls] == ["head", "peer"]
    assert all(call[2] == {"timeout": 300, "require_known_locality": True} for call in calls)
    assert "no render, sparse-attention fidelity" in capsys.readouterr().out


@pytest.mark.parametrize("failure, expected", [(subprocess.CompletedProcess([], 1, "", "secret error"), 1),
                                    (subprocess.TimeoutExpired("ssh", 300), 75), (_completed(relative_l2=0.5), 75)])
def test_failure_stops_fleet_and_does_not_print_raw_remote_output(monkeypatch, capsys, failure, expected):
    calls = []

    def run(config, host, *a, **kw):
        calls.append(host.name)
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(sol_smoke, "run_on_host", run)
    assert sol_smoke.cmd_sol_smoke(_config(), idle_maintenance=True) == expected
    assert calls == ["head"]
    assert "secret error" not in capsys.readouterr().out


@pytest.mark.parametrize("changes", [{"relative_l2": float("nan")}, {"relative_l2": True},
                                     {"backend": "triton"}, {"scope": "model-fidelity"},
                                     {"capability": [True, 1]}, {"shape": [1, 10, 1, 128]}])
def test_invalid_remote_evidence_refused(changes):
    with pytest.raises(ValueError):
        sol_smoke._parse_result(_completed(**changes).stdout)


def test_smoke_script_guards_before_cuda_under_lock():
    script = sol_smoke.smoke_script("~/env with spaces/bin/python")
    assert "acquire_lifecycle_lock(Path.home(), nonblocking=True)" in script
    assert script.index("require_idle()\n") < script.index("import torch\n")
    assert script.index("filter_worker_environment(os.environ)") < script.index("import torch\n")
    assert '"$PYBIN" -I -c ' in script
    assert "nvidia-smi" in script


def _cpu_torch(capability=(12, 1)):
    # Replace CUDA allocation with small CPU tensors while testing the actual
    # mathematical comparison and full-KV call arguments. No GPU API runs.
    fake = SimpleNamespace(**{name: getattr(torch, name) for name in
                              ("inference_mode", "bfloat16", "isfinite", "nn")})
    fake.cuda = SimpleNamespace(is_available=lambda: True, set_device=lambda n: None,
                                get_device_capability=lambda: capability,
                                current_device=lambda: 0, synchronize=lambda: None)
    fake.Generator = lambda **kw: torch.Generator()
    fake.randn = lambda shape, **kw: torch.randn((1, 16, 1, 128), dtype=kw["dtype"], generator=kw["generator"])
    return fake


def test_dense_control_calls_full_kv_and_measures_actual_output():
    seen = []

    def sol(q, k, v, **kw):
        seen.append(kw)
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)

    result = sol_smoke.kernel_control(_cpu_torch(), sol, lambda device: "cute_sm120", lambda: None)
    assert result["relative_l2"] == 0
    assert seen == [{"tau": 1.0, "thresh_type": "exact", "kv_splits": 1, "sink_start": 0, "sink_tokens": 4096}]


@pytest.mark.parametrize("value", [float("nan"), 100.0])
def test_nonfinite_or_inaccurate_output_fails(value):
    with pytest.raises(RuntimeError):
        sol_smoke.kernel_control(_cpu_torch(), lambda q, k, v, **kw: torch.full_like(q, value),
                                 lambda device: "cute_sm120", lambda: None)


def test_gb10_triton_fallback_refused_before_allocation():
    with pytest.raises(RuntimeError, match="CuTe"):
        sol_smoke.kernel_control(_cpu_torch(), lambda *a, **kw: pytest.fail("no allocation"),
                                 lambda device: "triton", lambda: None)


def test_fp32_output_cannot_be_reported_as_bf16_kernel_result():
    def sol(q, k, v, **kw):
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2).float()
    with pytest.raises(RuntimeError, match="dtype or device"):
        sol_smoke.kernel_control(_cpu_torch(), sol, lambda device: "cute_sm120", lambda: None)
