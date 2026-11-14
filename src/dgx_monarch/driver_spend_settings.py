"""The two loader settings a driver spend price reads, taken from the running graph as the loader takes them.

The DGX Monarch UNET Loader reads both from its ``mesh`` input, the Init node's
output (``nodes/loader_preflight``):

- the family forced on the Init node (``family_adapter``), from the mesh's own
  worker arguments (``family_select.override_from_worker_args``), over the
  checkpoint's sniff, and in place of a header it cannot read;
- the floor, ``driver_footprint.reserve_bytes`` of those arguments over the
  configured cluster policy (``consent_rescue.mesh_worker_args``), so the Init
  node's or the config's ``uma_reserve_gb`` raises it above 5 GiB.

A wrap holds no mesh. The running graph does hold the Init node the loader's
``mesh`` input links to, whose inputs the Init node turns into those arguments
(``nodes/init.py``), so the wrap reads the same two values there. The configured
policy is the attached mesh's; before a mesh is up (ComfyUI may run a spend
before the Init node on a first queue) it is read from the file the Init node
names, by ``get_mesh``'s rule. No Init node gives no settings, as before 1.5.0;
a configuration that cannot be read gives the Init node's own. A fault in the
sniff steps the price aside, as the loader steps aside for one.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

INIT_CLASS = "DGXMonarchInit"
FAMILY_INPUT = "family_adapter"
RESERVE_KEY = "uma_reserve_gb"  # the Init node's input and the worker argument share the name


class PricedLoader(NamedTuple):
    family: str
    worker_args: dict[str, Any]


def _literal(inputs: Mapping[str, Any], key: str, default: Any) -> Any:
    """A widget's value; a linked input (a list) is not known before it runs, so it reads as the default."""
    value = inputs.get(key, default)
    return default if isinstance(value, (list, tuple)) else value


def init_inputs(graph: Mapping[str, Any], loader: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The inputs of the Init node the loader's mesh comes from; None where it does not come from one."""
    link = (loader.get("inputs") or {}).get("mesh")
    node = graph.get(str(link[0])) if isinstance(link, (list, tuple)) and link else None
    if not isinstance(node, Mapping) or node.get("class_type") != INIT_CLASS:
        return None
    inputs = node.get("inputs")
    return inputs if isinstance(inputs, Mapping) else {}


def spec_args(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """The worker arguments a spend price reads, as the Init node sets them from these inputs."""
    from .family_select import AUTO, WORKER_ARG_KEY, override_from_worker_args

    args: dict[str, Any] = {}
    forced = override_from_worker_args({WORKER_ARG_KEY: _literal(inputs, FAMILY_INPUT, AUTO)})
    if forced is not None:
        args[WORKER_ARG_KEY] = forced
    reserve = _literal(inputs, RESERVE_KEY, 0.0)
    if isinstance(reserve, (int, float)) and not isinstance(reserve, bool) and reserve > 0:
        args[RESERVE_KEY] = float(reserve)
    return args


def configured(inputs: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    """``spec`` over the configured policy of the mesh the Init node attaches; ``spec`` alone where unread."""
    try:
        from .mesh import _MESH_LOCK, _MESHES

        with _MESH_LOCK:
            attached = [handle for handle in _MESHES.values()
                        if not (getattr(handle, "defunct", False) or getattr(handle, "teardown_complete", False))]
        if len(attached) == 1:
            return dict(attached[0].effective_worker_args(dict(spec)))
        from .config import find_config_path, load_cluster_config, local_config
        from .config_schema import actor_worker_args

        resolved = find_config_path(str(_literal(inputs, "config_path", "") or "") or None)
        mode = _literal(inputs, "mode", "auto")
        if mode == "cluster" or (mode == "auto" and resolved is not None):
            if resolved is None:
                return dict(spec)
            config = load_cluster_config(resolved)
        else:
            config = local_config(resolved)
        return {**actor_worker_args(config.worker_args), **spec}
    except Exception:
        return dict(spec)


def priced_loader(graph: Mapping[str, Any]) -> PricedLoader | None:
    """The first DGX Monarch UNET Loader of a family the loader prices a stack for, and the arguments it reads.

    By the loader's rules: a file the slab loader cannot open is not priced, a
    forced family stands over the sniff, and an unreadable header is the forced
    family or nothing.
    """
    import folder_paths

    from .adapters.detect import CheckpointSniffError, sniff_checkpoint
    from .driver_footprint import DRIVER_STACK_FAMILIES
    from .family_select import effective_family, override_from_worker_args
    from .nodes.loader_preflight import _slab_capable_path

    for node in graph.values():
        if not isinstance(node, Mapping) or not str(node.get("class_type", "")).startswith("DGXMonarch"):
            continue
        name = (node.get("inputs") or {}).get("unet_name")
        if not isinstance(name, str) or not name:
            continue
        path = folder_paths.get_full_path("diffusion_models", name)
        if not path or not _slab_capable_path(str(path)):
            continue
        inputs = init_inputs(graph, node)
        spec = spec_args(inputs) if inputs is not None else {}
        forced = override_from_worker_args(spec)
        try:
            family = effective_family(sniff_checkpoint(path)[0], forced)
        except CheckpointSniffError:
            if forced is None:
                continue
            family = forced
        if family in DRIVER_STACK_FAMILIES:
            return PricedLoader(family, configured(inputs, spec) if inputs is not None else {})
    return None
