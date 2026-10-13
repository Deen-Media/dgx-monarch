"""Explicit 0.7 process launch policy without controller environment leakage."""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from typing import Any

from .cli.worker_process_env import worker_environment_key_allowed
from .monarch_policy import MonarchPolicy, encode_policy, policy_environment
from .runtime_provenance import SOURCE_ONLY_PYCACHE_PREFIX

NATIVE_ENV = "DGXM_MONARCH_NATIVE_ENABLED"
# Resolve only the configured interpreter's leading ~/ on the destination host.
# All values travel as argv; none are evaluated as shell source.
_REMOTE_EXEC = 'py=$1; shift; case "$py" in "~/"*) py="$HOME/${py#??}";; esac; exec "$py" "$@"'


def worker_policy_args(config: Any, host: Any) -> list[str]:
    # Only the driver reads the readiness budget, so the Worker token carries
    # the default and a budget edit leaves the Worker units current.
    worker = replace(config.monarch, startup_timeout_s=MonarchPolicy().startup_timeout_s)
    args = ["--runtime-policy", encode_policy(worker, host.name)]
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


_DIAGNOSTICS_HINTS = {
    "otlp_restart_required": "Monarch loaded in this process before the endpoint could be "
                             "installed; restart ComfyUI",
    "ambient_otlp_endpoint_conflict": "OTEL_EXPORTER_OTLP_ENDPOINT in the ComfyUI environment names "
                                      "another endpoint; make otlp_endpoint match it, or unset it and "
                                      "restart ComfyUI",
}


Admission = tuple[dict[str, Any], str | None]


def _refuse_export(reason: str | None) -> None:
    """Raise the typed refusal for a requested export the driver cannot start."""
    if reason:
        from .config_schema import ClusterConfigError

        raise ClusterConfigError(
            f"[monarch.diagnostics] metrics or OTLP export cannot start on the driver ({reason}); "
            + _DIAGNOSTICS_HINTS.get(reason, 'set metrics = false and otlp_endpoint = "", or restart ComfyUI'))


def prepare_driver(config: Any) -> Admission | None:
    """Admit every participant's policy, apply it before native import, admit the runtime.

    Each refusal is a typed ClusterConfigError raised before transport commit.
    Monarch first loads here, so the fault hook is installed here too, ahead of
    every transport call. An endpoint the driver cannot install refuses before
    anything is written, so the running exporter and its thread stay as they
    were. Any later failure puts the earlier completed admission back through
    restore_driver. Returns the admission this one replaced, None on a first
    build, for the caller to restore if the fleet it admitted stays registered.
    """
    from .mesh import install_fault_hook
    from .monarch_capabilities import assert_supported_runtime
    from .monarch_diagnostics import configure_policy, requested_refusal, start_refusal
    from .monarch_policy import activate_native_policy, admit_native_targets, admitted, apply_policy
    from .monarch_policy_io import driver_host

    host, native = driver_host(config), config.rdma_latent_return
    if native:
        admit_native_targets(config.monarch, (host, *(entry.name for entry in config.hosts)))
    diagnostics = config.monarch.diagnostics
    _refuse_export(start_refusal(diagnostics))
    previous = admitted()
    try:
        apply_policy(config.monarch, host, role="driver", native_enabled=native)
        _refuse_export(requested_refusal(configure_policy(diagnostics), diagnostics))
        import monarch.actor  # noqa: F401 - load native bindings before identity admission

        install_fault_hook()
        assert_supported_runtime()
        activate_native_policy()
    except BaseException:
        restore_driver(previous)
        raise
    return previous


def restore_driver(previous: Admission | None) -> None:
    """Put back an admission prepare_driver replaced; None, from a first build, is a no-op.

    Only a fresh configure_policy restarts an exporter the replacing admission
    stopped. When the stopped thread is still inside a backend call, that
    restart is refused: monarch_diagnostics.status() then reads
    exporter_still_running and one ERROR names it, instead of a silent loss.
    """
    if previous is None:
        return
    from .monarch_diagnostics import configure_policy_logged
    from .monarch_policy import DiagnosticsPolicy, restore_admitted

    restore_admitted(previous)
    configure_policy_logged(DiagnosticsPolicy(**previous[0]["policy"]["diagnostics"]), "driver")
