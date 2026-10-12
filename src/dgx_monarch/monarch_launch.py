"""Explicit 0.7 process launch policy without controller environment leakage."""
from __future__ import annotations

import os
import sys
from typing import Any

from .cli.worker_process_env import worker_environment_key_allowed
from .monarch_policy import encode_policy, policy_environment
from .runtime_provenance import SOURCE_ONLY_PYCACHE_PREFIX

NATIVE_ENV = "DGXM_MONARCH_NATIVE_ENABLED"
# Resolve only the configured interpreter's leading ~/ on the destination host.
# All values travel as argv; none are evaluated as shell source.
_REMOTE_EXEC = 'py=$1; shift; case "$py" in "~/"*) py="$HOME/${py#??}";; esac; exec "$py" "$@"'


def worker_policy_args(config: Any, host: Any) -> list[str]:
    args = ["--runtime-policy", encode_policy(config.monarch, host.name)]
    if config.rdma_latent_return:
        args.append("--native-rdma")
    return args


def actor_bootstrap_command(config: Any, use_cluster: bool):
    """Return a per-Point command; remote source/executable stay host-owned."""
    def command(point: Any):
        from monarch._rust_bindings.monarch_hyperactor.host_mesh import BootstrapCommand

        host = config.hosts[point["hosts"]].name if use_cluster else ""
        environment: dict[str, str | None] = dict(policy_environment(config.monarch, host))
        if not config.monarch.diagnostics.otlp_endpoint:
            # An older coordinator may export logs. Remove its endpoint before
            # native child startup when this actor policy disables export.
            environment["OTEL_EXPORTER_OTLP_ENDPOINT"] = None
        environment[NATIVE_ENV] = "1" if config.rdma_latent_return else "0"
        environment["DGXM_DRIVER"] = os.environ.get("DGXM_DRIVER")
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPYCACHEPREFIX"] = SOURCE_ONLY_PYCACHE_PREFIX
        if use_cluster:
            # Worker startup already filters its environment. Inherit its managed
            # source and private runtime directories, never the controller's.
            return BootstrapCommand("/bin/sh", None, ["-c", _REMOTE_EXEC,
                "dgxm-bootstrap", config.python_bin, "-P", "-s", "-B", "-m",
                "monarch._src.actor.bootstrap_main"], environment, inherit_env=True)
        from .mesh_runtime import src_pythonpath

        local: dict[str, str | None] = {name: os.environ[name] for name in os.environ
                 if worker_environment_key_allowed(name)}
        local.update(environment)
        from importlib.metadata import distribution

        # Admission verified this loaded distribution; keep a candidate overlay
        # visible without copying arbitrary controller import paths.
        runtime_site = str(distribution("torchmonarch").locate_file(""))
        local["PYTHONPATH"] = os.pathsep.join(dict.fromkeys((src_pythonpath(), runtime_site)))
        local["DGXM_PYTHONPATH"] = src_pythonpath()
        return BootstrapCommand(sys.executable, None,
            ["-P", "-s", "-B", "-m", "monarch._src.actor.bootstrap_main"], local,
            inherit_env=False)
    return command


def wait_initialized(mesh: Any, deadline: float, label: str) -> None:
    """Observe eager readiness within one bring-up budget; never grant reuse."""
    import time

    from .mesh_runtime import get_off_loop
    from .runtime_diagnostics import phase

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Monarch startup readiness deadline expired")
    with phase("readiness"):
        get_off_loop(mesh.initialized, remaining, label)


def wait_worker_ready(workers: Any, deadline: float) -> None:
    """Prove constructor and endpoint dispatch readiness before publication."""
    import time

    from .mesh_runtime import get_off_loop
    from .runtime_diagnostics import phase

    wait_initialized(workers, deadline, "dgxm-workers-ready")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Monarch startup readiness deadline expired")
    with phase("readiness"):
        # Before setup, status returns without CUDA or host memory probes.
        get_off_loop(workers.status.call(), remaining, "dgxm-workers-status")


def prepare_driver(config: Any) -> None:
    """Apply policy before native import and admit the exact supported runtime."""
    from .monarch_capabilities import assert_supported_runtime
    from .monarch_diagnostics import configure_policy
    from .monarch_policy import activate_native_policy, apply_policy

    apply_policy(config.monarch, role="driver", native_enabled=config.rdma_latent_return)
    configure_policy(config.monarch.diagnostics)
    import monarch.actor  # noqa: F401 - load native bindings before identity admission

    assert_supported_runtime()
    activate_native_policy()
