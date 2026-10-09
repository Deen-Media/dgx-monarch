"""Runtime semantic probes for the torchmonarch compatibility canary.

These complement the signature inventory in ``monarch_surface``. Only the
standard library is imported directly; probed Monarch modules arrive through
``import_module``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any


def type_name(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def exception_observation(exc: BaseException) -> dict[str, str]:
    return {
        "outcome": "raised",
        "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "message": str(exc),
    }


def probe_python_task_asyncio(
    import_module: Callable[[str], object],
) -> dict[str, str]:
    module = import_module("monarch._rust_bindings.monarch_hyperactor.pytokio")
    python_task: Any = module.__dict__["PythonTask"]  # type: ignore[attr-defined]

    async def probe() -> dict[str, str]:
        try:
            result = await python_task.sleep(0)
        except BaseException as exc:
            return exception_observation(exc)
        return {
            "outcome": "returned",
            "result_type": type_name(result),
        }

    return asyncio.run(probe())


def probe_unhandled_hook_writable(
    import_module: Callable[[str], object],
) -> dict[str, str]:
    actor: Any = import_module("monarch.actor")
    original = actor.unhandled_fault_hook

    def marker(_failure: object) -> None:
        return None

    try:
        actor.unhandled_fault_hook = marker
        if actor.unhandled_fault_hook is not marker:
            return {"outcome": "not_writable"}
    finally:
        actor.unhandled_fault_hook = original
    return {"outcome": "writable"}


def probe_mesh_factory_private_ownership(
    import_module: Callable[[str], object],
) -> dict[str, Any]:
    """Verify the private identity fields used for interrupted spawn recovery."""
    host_module: Any = import_module("monarch._src.actor.host_mesh")
    proc_module: Any = import_module("monarch._src.actor.proc_mesh")
    host_type: Any = host_module.__dict__["HostMesh"]
    proc_type: Any = proc_module.__dict__["ProcMesh"]
    inner = object()
    region = object()
    host = host_type(inner, region, False, False, None)
    proc = proc_type(object(), host, region, region, None)
    return {
        "outcome": "observed",
        "host_inner_identity": host._inner_host_mesh is inner,
        "host_proc_meshes_empty": host._proc_meshes == [],
        "host_proc_meshes_type": type_name(host._proc_meshes),
        "proc_host_identity": proc._host_mesh is host,
    }


__all__ = [
    "exception_observation",
    "probe_mesh_factory_private_ownership",
    "probe_python_task_asyncio",
    "probe_unhandled_hook_writable",
    "type_name",
]
