"""The Monarch names the teardown, the fault hook and the root-level guard rest on.

monarch_surface appends these to its TOUCHPOINTS, so the canary checks them
under the pin and under the latest wheel. The CPU smoke checks how they behave
(tests/canary/monarch_cpu_smoke.py), and the lifecycle_config_defaults probe
records the config defaults. Standard library only.
"""
from __future__ import annotations

from .monarch_touchpoint_spec import MonarchTouchpoint, _attr, _call, _class

LIFECYCLE_TOUCHPOINTS: tuple[MonarchTouchpoint, ...] = (
    # monarch_launch waits on the Worker mesh's own spawn; under direct spawn a
    # constructor failure fails it.
    _attr("monarch._src.actor.actor_mesh", "ActorMesh.initialized"),
    # worker_proc_id primes a restarted loop from the client instance.
    _attr("monarch._src.actor.actor_mesh", "Context.actor_instance"),
    _call("monarch._src.actor.actor_mesh", "Instance._as_rust", positional=("self",)),
    # The actor processes' module, which proc_identity and the Worker service
    # scripts match beside the HYPERACTOR_MESH_BOOTSTRAP_MODE marker.
    _call("monarch._src.actor.bootstrap_main", "invoke_main"),
    # The fault hook reads each fault's mesh name (retired_meshes).
    _class("monarch._rust_bindings.monarch_hyperactor.supervision", "MeshFailure"),
    _attr("monarch._rust_bindings.monarch_hyperactor.supervision", "MeshFailure.mesh_name"),
    # The internal meshes it keeps quiet, by actor class, and where Monarch spawns them.
    _class("monarch._src.actor.proc_mesh", "SetupActor"),
    _attr("monarch._src.actor.proc_mesh", "SetupActor.startup_actor_from_setup_function"),
    _class("monarch._src.actor.logging", "_LogLevelSetter"),
    _attr("monarch._src.actor.logging", "LoggingManager.spawn_log_level_actor"),
    # Where Monarch sets the root logger to INFO (mesh_runtime.kept_root_log_level).
    _attr("monarch._src.actor.proc_mesh", "ProcMesh._start_root_logging"),
    _attr("monarch._src.actor.logging", "LoggingManager.initialize_local_log_level"),
)

__all__ = ["LIFECYCLE_TOUCHPOINTS"]
