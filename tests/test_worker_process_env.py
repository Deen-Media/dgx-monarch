"""Credential containment at the worker-loop process boundary."""
from __future__ import annotations

import builtins
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import worker_loop, worker_process_env


def test_worker_environment_preserves_reviewed_runtime_and_compute_names():
    environment = {
        "HOME": "/home/worker",
        "USER": "worker",
        "PATH": "/usr/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TMPDIR": "/private/tmp",
        "DGXM_RDMA_MIN_BYTES": "8388608",
        "DGXM_RDMA_QP_SPLIT": "2",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_MODULE_LOADING": "LAZY",
        "CUDA_CACHE_MAXSIZE": "4294967296",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "XDIT_LOGGING_LEVEL": "WARN",
        "OMP_NUM_THREADS": "16",
    }

    assert worker_process_env.filter_worker_environment(environment) == 0
    assert set(environment) == {
        "HOME", "USER", "PATH", "LANG", "LC_ALL", "TMPDIR",
        "DGXM_RDMA_MIN_BYTES", "DGXM_RDMA_QP_SPLIT",
        "CUDA_DEVICE_ORDER", "CUDA_MODULE_LOADING", "CUDA_CACHE_MAXSIZE",
        "PYTORCH_CUDA_ALLOC_CONF",
        "XDIT_LOGGING_LEVEL", "OMP_NUM_THREADS",
    }


@pytest.mark.parametrize("name", [
    "OPENAI_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "GITHUB_TOKEN",
    "SSH_AUTH_SOCK",
    "HTTP_PROXY",
    "RUST_LOG",
    "DGXM_SSH_KEY",
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "UCX_NET_DEVICES",
    "CUDA_API_KEY",
    "NVIDIA_NGC_APIKEY",
    "GLOO_NGCAPIKEY",
    "NCCL_CLIENT_SECRET",
    "NCCL_CLIENTSECRET",
    "UCX_PSK",
    "UCX_CLUSTERPSK",
    "TORCH_AUTH_TOKEN",
    "TRITON_SESSION",
    "TORCH_LOGS",
    "IBV_FORK_SAFE",
    "MLX5_SHUT_UP_BF",
    "NCCL_NET_PLUGIN",
    "NCCL_DEBUG_FILE",
    "UCX_LOG_FILE",
    "GLOO_MODULE_DIR",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "TRITON_PTXAS_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDACXX",
    "HYPERACTOR_MESH_ATTACH_CONFIG_TIMEOUT",
    "MONARCH_HOME",
    "PYTHONSTARTUP",
    "NCCL_PROTO",
    "NCCL_P2P_DISABLE",
    "NCCL_LAUNCH_ORDER_IMPLICIT",
    "lowercase_name",
])
def test_worker_environment_rejects_secrets_unrecognized_and_injection_names(name: str):
    assert not worker_process_env.worker_environment_key_allowed(name)


def test_filter_does_not_read_rejected_values():
    class KeyOnlyEnvironment(dict[str, str]):
        def __getitem__(self, name: str) -> str:
            if name == "UNRELATED_SERVICE_VALUE":
                raise AssertionError("rejected environment value was read")
            return super().__getitem__(name)

    environment = KeyOnlyEnvironment({
        "PATH": "/usr/bin",
        "UNRELATED_SERVICE_VALUE": "opaque-placeholder",
    })
    assert worker_process_env.filter_worker_environment(environment) == 1
    assert environment == {"PATH": "/usr/bin"}


def test_secure_runtime_uses_private_xdg_child(tmp_path: Path):
    tmp_path.chmod(0o700)
    environment = {"HOME": str(tmp_path / "home"), "XDG_RUNTIME_DIR": str(tmp_path)}

    runtime = worker_process_env.secure_worker_runtime(environment)

    assert runtime == tmp_path / "dgx-monarch"
    assert stat.S_IMODE(runtime.stat().st_mode) == 0o700
    assert environment["XDG_RUNTIME_DIR"] == str(runtime)
    assert {environment[name] for name in ("TMPDIR", "TMP", "TEMP")} == {str(runtime)}


def test_secure_runtime_rejects_public_xdg_and_falls_back_under_home(tmp_path: Path):
    public = tmp_path / "public-runtime"
    public.mkdir(mode=0o755)
    public.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    environment = {"HOME": str(home), "XDG_RUNTIME_DIR": str(public)}

    runtime = worker_process_env.secure_worker_runtime(environment)

    assert runtime == home / ".local" / "state" / "dgx-monarch" / "runtime"
    assert stat.S_IMODE(runtime.stat().st_mode) == 0o700
    assert environment["TMPDIR"] == str(runtime)


def test_secure_runtime_refuses_symlink_at_final_path(tmp_path: Path):
    tmp_path.chmod(0o700)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "dgx-monarch").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="owned directory"):
        worker_process_env.secure_worker_runtime({"XDG_RUNTIME_DIR": str(tmp_path)})


def test_prepare_worker_process_environment_sets_umask_filters_and_secures_tmp(
    monkeypatch, tmp_path: Path,
):
    seen_umask: list[int] = []
    monkeypatch.setattr(worker_process_env.os, "umask", seen_umask.append)
    environment = {
        "HOME": str(tmp_path),
        "PATH": "/usr/bin",
        "RUST_LOG": "debug",
        "CUDA_MODULE_LOADING": "LAZY",
        "PYTHONPATH": "/ambient/one:/ambient/two",
        "DGXM_PYTHONPATH": "/ambient/other",
    }
    managed = tmp_path / "managed-src"
    managed.mkdir()

    removed, runtime = worker_process_env.prepare_worker_process_environment(managed, environment)

    assert seen_umask == [0o077]
    assert removed == 3
    assert "RUST_LOG" not in environment
    assert environment["PATH"] == "/usr/bin"
    assert environment["CUDA_MODULE_LOADING"] == "LAZY"
    assert environment["PYTHONPATH"] == str(managed)
    assert environment["DGXM_PYTHONPATH"] == str(managed)
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONPYCACHEPREFIX"] == "/proc/self/fd/2147483647"
    assert environment["TMPDIR"] == str(runtime)


def test_sanitized_environment_is_inherited_by_child_process(tmp_path: Path):
    source = Path(__file__).parents[1] / "src"
    probe = """
import os
import subprocess
import sys
from pathlib import Path
from dgx_monarch.cli import worker_process_env

managed = Path(worker_process_env.__file__).resolve().parents[2]
worker_process_env.prepare_worker_process_environment(managed)
check = (
    "FICTIONAL_SERVICE_TOKEN" not in os.environ
    and "RUST_LOG" not in os.environ
    and "NCCL_DEBUG" not in os.environ
    and os.environ.get("OMP_NUM_THREADS") == "16"
    and os.environ.get("PATH") == "/usr/bin:/bin"
    and os.environ.get("PYTHONPATH") == str(managed)
    and os.environ.get("DGXM_PYTHONPATH") == str(managed)
    and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
    and os.environ.get("PYTHONPYCACHEPREFIX") == "/proc/self/fd/2147483647"
)
child = subprocess.run(
    [sys.executable, "-c", "import os, sys; sys.exit('FICTIONAL_SERVICE_TOKEN' in os.environ)"],
    check=False,
)
raise SystemExit(0 if check and child.returncode == 0 else 1)
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env={
            "HOME": str(tmp_path),
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(source),
            "DGXM_PYTHONPATH": "/ambient/other",
            "NCCL_DEBUG": "WARN",
            "OMP_NUM_THREADS": "16",
            "RUST_LOG": "debug",
            "FICTIONAL_SERVICE_TOKEN": "opaque-placeholder",
        },
    )
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_worker_loop_prepares_environment_before_monarch_import(monkeypatch):
    events: list[str] = []
    managed_paths: list[Path] = []
    monkeypatch.setattr(sys, "argv", ["dgxm-worker", "--address", "tcp://127.0.0.1:26600"])
    monkeypatch.setattr(
        worker_process_env,
        "prepare_worker_process_environment",
        lambda path: (managed_paths.append(path), events.append("prepare")),
    )

    actor_module = SimpleNamespace(
        enable_transport=lambda _address: events.append("enable"),
        run_worker_loop_forever=lambda **_kwargs: events.append("run"),
    )
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "monarch.actor":
            assert events == ["prepare"]
            events.append("monarch-import")
            return actor_module
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    worker_loop.main()

    assert events == ["prepare", "monarch-import", "enable", "run"]
    assert managed_paths == [Path(worker_loop.__file__).resolve().parents[2]]
