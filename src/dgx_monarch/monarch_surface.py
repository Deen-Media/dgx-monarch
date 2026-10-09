"""Executable inventory of the torchmonarch compatibility surface dgx-monarch uses.

The weekly dependency canary snapshots this manifest under the exact pin and
then under the latest stable wheel, in one scratch environment, and compares
the two snapshots. This module needs only the standard library, so the probe
adds no dependency to the check it runs.

Raw signatures are evidence for a maintainer, not the compatibility verdict.
The verdict binds only the positional order, keyword names and variadic shape
that production calls require, so an annotation change shows in the report as
a change, not a blocker.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
import warnings
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from .monarch_semantics import (
    exception_observation as _exception_observation,
)
from .monarch_semantics import (
    probe_mesh_factory_private_ownership as _probe_mesh_factory_private_ownership,
)
from .monarch_semantics import (
    probe_python_task_asyncio as _probe_python_task_asyncio,
)
from .monarch_semantics import (
    probe_unhandled_hook_writable as _probe_unhandled_hook_writable,
)
from .monarch_semantics import (
    type_name as _type_name,
)
from .monarch_touchpoint_spec import (  # Public monarch_surface API.
    MonarchTouchpoint,
    _attr,
    _call,
    _class,
    _resolve,
    _signature_record,
)

SNAPSHOT_SCHEMA = 1


# Public imports, returned runtime objects, and the private surfaces that
# control the Future bridge or exact ProcMesh ownership during construction.
# Private ownership drift must block a pin bump: mesh_factory fails closed
# rather than guessing which process it may stop.
TOUCHPOINTS: tuple[MonarchTouchpoint, ...] = (
    _class("monarch.actor", "Actor"),
    _class("monarch.actor", "ActorError"),
    _call("monarch.actor", "endpoint", positional=("method",)),
    # Every long-running GPUWorker endpoint (actor/worker.py) has used this
    # since the 0.6.0 pin; a signature or removal drift here is a cancellation outage.
    _call("monarch.actor", "concurrent_endpoint", positional=("method",)),
    _call("monarch.actor", "this_host"),
    _call(
        "monarch.actor",
        "attach_to_workers",
        "ca",
        "workers",
        "name",
    ),
    _call("monarch.actor", "enable_transport", positional=("transport",)),
    _call(
        "monarch.actor",
        "run_worker_loop_forever",
        "ca",
        "address",
    ),
    _attr("monarch.actor", "unhandled_fault_hook"),
    _class("monarch.actor", "HostMesh"),
    _attr("monarch.actor", "HostMesh.initialized"),
    _call(
        "monarch.actor",
        "HostMesh.spawn_procs",
        "per_host",
        "bootstrap",
        positional=("self",),
    ),
    _class("monarch.actor", "ProcMesh"),
    _attr("monarch.actor", "ProcMesh.initialized"),
    _call(
        "monarch.actor",
        "ProcMesh.spawn",
        positional=("self", "name", "Class"),
        var_positional=True,
        var_keyword=True,
    ),
    _call("monarch.actor", "ProcMesh.stop", positional=("self", "reason")),
    # mesh_factory clones one HostMesh wrapper per spawn so callback/registry
    # observations can be attributed by exact host identity.  Keep the private
    # constructor and every callable registry seam in the executable manifest.
    _call(
        "monarch._src.actor.host_mesh",
        "HostMesh",
        positional=(
            "hy_host_mesh",
            "region",
            "stream_logs",
            "is_fake_in_process",
            "code_sync_proc_mesh",
        ),
        require_class=True,
    ),
    _attr("monarch._src.actor.host_mesh", "HostMesh.region"),
    _attr("monarch._src.actor.host_mesh", "HostMesh.stream_logs"),
    _attr("monarch._src.actor.host_mesh", "HostMesh.is_fake_in_process"),
    _class("monarch._src.actor.proc_mesh", "ProcMesh"),
    _call("monarch._src.actor.proc_mesh", "get_active_proc_meshes"),
    _call(
        "monarch._src.actor.proc_mesh",
        "register_proc_mesh_spawn_callback",
        positional=("callback",),
    ),
    _call(
        "monarch._src.actor.proc_mesh",
        "unregister_proc_mesh_spawn_callback",
        positional=("callback",),
    ),
    # ProcMesh.spawn returns this implementation object even though ActorMesh
    # is not re-exported from monarch.actor.
    _attr("monarch._src.actor.actor_mesh", "ActorMesh.extent"),
    _call(
        "monarch._src.actor.actor_mesh",
        "ActorMesh.slice",
        positional=("self",),
        var_keyword=True,
    ),
    _attr("monarch.actor", "Extent.labels"),
    _class("monarch.actor", "Endpoint"),
    _call(
        "monarch.actor",
        "Endpoint.call",
        positional=("self",),
        var_positional=True,
        var_keyword=True,
    ),
    _call(
        "monarch.actor",
        "Endpoint.call_one",
        positional=("self",),
        var_positional=True,
        var_keyword=True,
    ),
    _call(
        "monarch.actor",
        "Endpoint.broadcast",
        positional=("self",),
        var_positional=True,
        var_keyword=True,
    ),
    _class("monarch.actor", "Future"),
    _call("monarch.actor", "Future.get", "timeout", positional=("self",)),
    _call("monarch.actor", "Future.__await__", positional=("self",)),
    _class("monarch.actor", "ValueMesh"),
    _call("monarch.actor", "ValueMesh.items", positional=("self",)),
    _class("monarch.actor", "Channel"),
    _call("monarch.actor", "Channel.open"),
    _class("monarch.actor", "Port"),
    _call("monarch.actor", "Port.send", positional=("self", "obj")),
    _class("monarch.actor", "PortReceiver"),
    _call("monarch.actor", "PortReceiver.recv", positional=("self",)),
    _call(
        "monarch.rdma",
        "RDMABuffer",
        positional=("data",),
        require_class=True,
    ),
    _call(
        "monarch.rdma",
        "RDMABuffer.read_into",
        "timeout",
        positional=("self", "dst"),
    ),
    _call("monarch.rdma", "RDMABuffer.drop", positional=("self",)),
    _call("monarch.rdma", "is_ibverbs_available"),
    _class(
        "monarch._rust_bindings.monarch_hyperactor.pytokio",
        "PythonTask",
    ),
    _call(
        "monarch._rust_bindings.monarch_hyperactor.pytokio",
        "PythonTask.sleep",
        positional=("seconds",),
    ),
    _call(
        "monarch._rust_bindings.monarch_hyperactor.pytokio",
        "PythonTask.__await__",
        positional=("self",),
    ),
    _call(
        "monarch._rust_bindings.monarch_hyperactor.pytokio",
        "PythonTask.block_on",
        positional=("self",),
    ),
    _call(
        "monarch._rust_bindings.monarch_hyperactor.pytokio",
        "PythonTask.with_timeout",
        positional=("self", "seconds"),
    ),
)


def _manifest_digest() -> str:
    payload = json.dumps(
        [asdict(touchpoint) for touchpoint in TOUCHPOINTS],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def collect_touchpoints(
    import_module: Callable[[str], object] = importlib.import_module,
) -> dict[str, dict[str, Any]]:
    """Inspect every declared surface while preserving all failures in output."""

    modules: dict[str, object] = {}
    module_errors: dict[str, str] = {}
    observed: dict[str, dict[str, Any]] = {}
    for touchpoint in TOUCHPOINTS:
        module = modules.get(touchpoint.module)
        if module is None and touchpoint.module not in module_errors:
            try:
                module = import_module(touchpoint.module)
            except BaseException as exc:
                module_errors[touchpoint.module] = f"{type(exc).__name__}: {exc}"
            else:
                modules[touchpoint.module] = module
        if module is None:
            observed[touchpoint.path] = {
                "status": "missing",
                "kind": None,
                "signature": None,
                "detail": f"module import failed: {module_errors[touchpoint.module]}",
            }
            continue
        try:
            value = _resolve(module, touchpoint.attribute)
        except BaseException as exc:
            observed[touchpoint.path] = {
                "status": "missing",
                "kind": None,
                "signature": None,
                "detail": f"attribute lookup failed: {type(exc).__name__}: {exc}",
            }
            continue
        observed[touchpoint.path] = _signature_record(value, touchpoint)
    return observed


def _probe_future_get_asyncio(import_module: Callable[[str], object]) -> dict[str, Any]:
    actor = import_module("monarch.actor")
    pytokio = import_module("monarch._rust_bindings.monarch_hyperactor.pytokio")
    future_type: Any = actor.__dict__["Future"]  # type: ignore[attr-defined]
    python_task: Any = pytokio.__dict__["PythonTask"]  # type: ignore[attr-defined]

    async def probe() -> dict[str, Any]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = future_type(coro=python_task.sleep(0)).get(timeout=1)
            except BaseException as exc:
                observed: dict[str, Any] = _exception_observation(exc)
            else:
                observed = {
                    "outcome": "returned",
                    "result_type": _type_name(result),
                }
        observed["warnings"] = [
            {
                "category": (
                    f"{warning.category.__module__}."
                    f"{warning.category.__qualname__}"
                ),
                "message": str(warning.message),
            }
            for warning in caught
        ]
        return observed

    return asyncio.run(probe())


def _probe_actor_queue_dispatch_default(
    import_module: Callable[[str], object],
) -> dict[str, Any]:
    """Observe the runtime default for hyperactor's actor dispatch model.

    The out-of-band worker calls (cancel_sample, status, the pipelined
    artifact preflight) need a second endpoint served while sample() is in
    flight. Queue dispatch runs plain endpoints to completion one at a time,
    so a default flip (torchmonarch 0.6.0, meta-pytorch/monarch#4211) stops
    cancellation and live telemetry without an error unless the long-running
    endpoints are concurrent (actor/worker.py). Any change in the observed
    default, or the flag's removal (key_absent), is a semantic-drift blocker.
    """
    module = import_module("monarch._rust_bindings.monarch_hyperactor.config")
    get_global_config: Any = module.__dict__["get_global_config"]  # type: ignore[attr-defined]
    config = get_global_config()
    if not isinstance(config, dict):
        return {"outcome": "unexpected_config_type", "config_type": _type_name(config)}
    if "actor_queue_dispatch" not in config:
        return {"outcome": "key_absent"}
    return {
        "outcome": "observed",
        "actor_queue_dispatch": bool(config["actor_queue_dispatch"]),
    }


def collect_semantics(
    import_module: Callable[[str], object] = importlib.import_module,
) -> dict[str, dict[str, Any]]:
    probes: tuple[tuple[str, Callable[[Callable[[str], object]], dict[str, Any]]], ...] = (
        ("raw_python_task_await_in_asyncio", _probe_python_task_asyncio),
        ("public_future_get_in_asyncio", _probe_future_get_asyncio),
        ("unhandled_fault_hook_assignment", _probe_unhandled_hook_writable),
        ("actor_queue_dispatch_default", _probe_actor_queue_dispatch_default),
        ("mesh_factory_private_ownership", _probe_mesh_factory_private_ownership),
    )
    observations: dict[str, dict[str, Any]] = {}
    for name, probe in probes:
        try:
            observations[name] = probe(import_module)
        except BaseException as exc:
            observations[name] = {
                "outcome": "probe_error",
                "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
            }
    return observations


def collect_snapshot(
    *,
    import_module: Callable[[str], object] = importlib.import_module,
    distribution_version: Callable[[str], str] = importlib.metadata.version,
    include_semantics: bool = True,
) -> dict[str, Any]:
    try:
        version = distribution_version("torchmonarch")
    except BaseException as exc:
        version = f"unavailable ({type(exc).__name__}: {exc})"
    return {
        "schema": SNAPSHOT_SCHEMA,
        "distribution": "torchmonarch",
        "version": version,
        "manifest_digest": _manifest_digest(),
        "environment": {
            "python": platform.python_version(),
            "implementation": sys.implementation.name,
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "touchpoints": collect_touchpoints(import_module),
        "semantics": collect_semantics(import_module) if include_semantics else {},
    }


__all__ = [
    "SNAPSHOT_SCHEMA",
    "TOUCHPOINTS",
    "MonarchTouchpoint",
    "collect_semantics",
    "collect_snapshot",
    "collect_touchpoints",
]
