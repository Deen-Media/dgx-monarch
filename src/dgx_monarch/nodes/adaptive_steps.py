"""Refuse an adaptive-step sampler where FSDP ties data-parallel groups into every call.

Under data parallelism each group samples its own slice of the batch. An
adaptive-step sampler sizes its steps from the error of that slice, so two
groups can make different numbers of model calls. Where FSDP shards the
model (``*+fsdp`` with dp above 1), every forward gathers each block's
weights from every rank of the world: ``adapters/fsdp.py`` calls
``fully_shard`` with no mesh, torch then builds one over the default group,
and each sharded module's pre-forward hook all-gathers its parameters before
it runs. A group with more calls then waits in that gather, and in the cond
batch agreement before it, for ranks that have finished, until the sample
deadline. No group of fewer ranks can carry those gathers, so nothing makes
the calls line up, and the driver refuses the combination, class P. Pure
data parallelism and the sharded layouts without FSDP keep rendering: their
groups share no collective (``actor/cond_batch_agreement.py``). Only the
explicit ``+fsdp`` presets reach it: ``auto`` never resolves FSDP
(``nodes/common.resolve_topology`` leaves ``fits_resident`` at True), so the
refusal never arises under ``auto``.

An explicit preset sets up and loads at its UNET loader, so the refusal comes
at three points:

* the UNET loader (``nodes/loaders.py``), before its setup and load, reads the
  graph: a dgx-monarch sampler ComfyUI is sure to run, whose model provably
  comes from that loader, and whose sampler is a literal ``sampler_name``, a
  KSamplerSelect naming one, or a SamplerDPMAdaptative node. What the walk
  cannot read makes no claim there;
* the sampler and Identity Gate nodes, before the first-use check, with the
  sampler in hand: they catch what the loader could not read, after that
  loader's setup and load;
* the render submit, with the topology the render resolves, before setup
  (``render_validation.validate_render_topology``).

ComfyUI has one sampler whose call count follows its data: ``dpm_adaptive``,
``DPMSolver.dpm_solver_adaptive`` in ``comfy/k_diffusion/sampling.py`` and the
only user of its ``PIDStepSizeController``. KSampler names it; Sampler Custom
reaches it as a SAMPLER object, which SamplerDPMAdaptative and KSamplerSelect
build with ``comfy.samplers.ksampler("dpm_adaptive", ...)``. The comfy seam
canary checks every reading on ComfyUI master. A third-party sampler whose
call count follows its data cannot be recognised and would stall the same way.
"""
from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any, cast

from ..refusal import RefusalClass, refusal

TROUBLESHOOTING = 156
# The KSampler names whose model call count follows the latent's own values.
ADAPTIVE_STEP_SAMPLERS: tuple[str, ...] = ("dpm_adaptive",)
# Stock nodes whose SAMPLER output a loader's graph walk can read.
SAMPLER_SELECT_NODE = "KSamplerSelect"  # its sampler_name input names the sampler
ADAPTIVE_SAMPLER_NODE = "SamplerDPMAdaptative"  # always the adaptive solver
_NAMED = "sampler {}"
_OBJECT = "Sampler Custom's SAMPLER, ComfyUI's dpm_adaptive solver,"


def _solver_functions() -> list[Any]:
    """Each function a SAMPLER object runs for an adaptive-step sampler: the
    k-diffusion solver itself, and the wrapper ``comfy.samplers.ksampler``
    builds around it. ComfyUI absent answers none."""
    try:
        import comfy.k_diffusion.sampling as k_sampling
        import comfy.samplers
    except Exception:
        return []
    found: list[Any] = []
    for name in ADAPTIVE_STEP_SAMPLERS:
        raw = getattr(k_sampling, f"sample_{name}", None)
        if raw is not None:
            found.append(raw)
        try:
            found.append(comfy.samplers.ksampler(name).sampler_function)
        except Exception:
            pass
    return found


def _runs_adaptive_solver(sampler: Any) -> bool:
    """Whether a SAMPLER object's function is ComfyUI's adaptive-step solver.

    ``ksampler`` builds a new closure on every call, so the wrapper is matched
    by its code object, which every closure of it shares.
    """
    function = getattr(sampler, "sampler_function", None)
    while isinstance(function, functools.partial):
        function = function.func
    if function is None:
        return False
    code = getattr(function, "__code__", None)
    return any(function is known or (code is not None and code is getattr(known, "__code__", None))
               for known in _solver_functions())


def adaptive_sampler(request: Mapping[str, Any]) -> str | None:
    """How a refusal names the request's adaptive-step sampler; None for any other."""
    name = request.get("sampler_name")
    if isinstance(name, str) and name in ADAPTIVE_STEP_SAMPLERS:
        return _NAMED.format(name)
    sampler = request.get("sampler_object")
    if sampler is not None and _runs_adaptive_solver(sampler):
        return _OBJECT
    return None


def ties_data_parallel_groups(topo: Any) -> bool:
    """Whether every rank shares each model call while dp groups sample apart."""
    return int(getattr(topo, "dp", 1) or 1) > 1 and bool(getattr(topo, "fsdp", False))


def _presets_without_dp(world: int) -> list[tuple[str, bool]]:
    """Each preset that leaves one data-parallel group at ``world``, and whether it shards."""
    from ..topology import PRESETS, topology_from_preset

    found = []
    for name in PRESETS:
        try:
            topo = topology_from_preset(name, world)
        except ValueError:
            continue  # does not tile this world
        if topo.dp == 1:
            found.append((name, bool(topo.fsdp)))
    return found


def _refuse(label: str, topo: Any) -> None:
    from ..adapters.base import UnsupportedModelError

    world = int(getattr(topo, "world", 0) or 0)
    without_dp = _presets_without_dp(world)
    names = ", ".join(name for name, _shards in without_dp)
    shards = any(sharded for _name, sharded in without_dp)
    remedies = ["a sampler with a fixed step count (any but dpm_adaptive, such as euler "
                "or dpmpp_2m)"]
    if shards:
        remedies.append(f"a topology preset without data parallelism ({names})")
    elif without_dp:
        remedies.append(f"a topology preset without data parallelism ({names}), only where "
                        "the model fits resident, since none of them shards it")
    remedies.append("a preset without +fsdp where the model fits resident, so each "
                    "data-parallel group runs on its own")
    needs_sampler = "" if shards else (
        f" No preset shards the model without data parallelism on {world} ranks, so a model "
        "that needs FSDP here needs another sampler.")
    every_rank = "both ranks" if world == 2 else f"each of the {world} ranks"
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"{label} cannot run under {topo.describe()} on {world} ranks. The layout splits "
        f"the batch over {int(topo.dp)} data-parallel groups, and FSDP gathers every model "
        f"call's weights from {every_rank}, so every rank must make the same number of "
        "model calls. dpm_adaptive picks its step sizes from the error of its own slice of "
        "the batch, so one group can make more calls than another and would wait for ranks "
        f"that have finished until the sample deadline. Use {', or '.join(remedies)}."
        f"{needs_sampler}",
        troubleshooting=TROUBLESHOOTING,
    ))


def refuse(request: Mapping[str, Any], topo: Any) -> None:
    """Refuse ``request`` under ``topo`` where its sampler cannot keep the calls aligned."""
    label = adaptive_sampler(request)
    if label is not None and ties_data_parallel_groups(topo):
        _refuse(label, topo)


def preflight_for_render(model: Any, request: Mapping[str, Any], latent: Mapping[str, Any],
                         cfg_value: Any, *, resolve_topology: Any = None) -> None:
    """The same refusal before a render's first-use check.

    Resolves the topology as the render will (``resolve_topology`` defaults to
    ``common._resolve_topology_for_latent``), and only for an adaptive-step
    sampler. A resolution that raises is left for the render's own resolution
    to raise, so this adds no new failure; the render submit refuses again
    with the topology it resolves, before setup.
    """
    if adaptive_sampler(request) is None:
        return
    if resolve_topology is None:
        from .common import _resolve_topology_for_latent as resolve_topology
    mesh = getattr(model, "mesh", None)
    world = getattr(getattr(mesh, "handle", None), "world", getattr(mesh, "world", None))
    try:
        topo, _sage, _reason = resolve_topology(
            model, latent, cfg_value, None if world is None else int(world))
    except Exception:
        return
    refuse(request, topo)


def _graph_sampler(prompt: Mapping, loader_id: str) -> str | None:
    """How a refusal names the first adaptive-step sampler the graph ties to
    ``loader_id``, among the dgx-monarch samplers ComfyUI is sure to run."""
    from .graph_live import is_link, live_nodes
    from .render_validation import _FLEET_SAMPLER, _node, _takes_loader

    live = live_nodes(prompt, sure=True)
    for node_id, node in prompt.items():
        if str(node_id) not in live or not isinstance(node, Mapping):
            continue
        class_type, inputs = node.get("class_type"), node.get("inputs")
        if (not isinstance(class_type, str) or not class_type.startswith("DGXMonarch")
                or class_type == _FLEET_SAMPLER or not isinstance(inputs, Mapping)
                or not _takes_loader(prompt, inputs, loader_id)):
            continue
        name = inputs.get("sampler_name")
        link = inputs.get("sampler")
        if is_link(link):
            source = _node(prompt, cast(list, link)[0]) or {}
            if source.get("class_type") == ADAPTIVE_SAMPLER_NODE:
                return _OBJECT
            if source.get("class_type") == SAMPLER_SELECT_NODE:
                name = (source.get("inputs") or {}).get("sampler_name")
        if isinstance(name, str) and name in ADAPTIVE_STEP_SAMPLERS:
            return _NAMED.format(name)
    return None


def preflight_loader_graph(mesh: Any, prompt: Any, loader_id: Any = None) -> None:
    """The same refusal at a UNET loader of an explicit preset, before its setup and load.

    Reads the topology as ``render_validation.preflight_graph_batch_divides_dp``
    does, from the preset and the mesh's world, and the sampler from the
    graph. ``auto``, an unknown world, a graph or loader it cannot read, and a
    sampler it cannot name make no claim; the sampler node answers those.
    """
    try:
        preset = getattr(mesh, "topology_preset", "auto")
        handle = getattr(mesh, "handle", None)
        if (preset == "auto" or handle is None or getattr(handle, "defunct", False)
                or loader_id is None or not isinstance(prompt, Mapping)):
            return
        world = int(getattr(handle, "world", 0) or 0)
        if world <= 0:
            return
        from ..topology import topology_from_preset

        topo = topology_from_preset(preset, world)
        if not ties_data_parallel_groups(topo):
            return
        label = _graph_sampler(prompt, str(loader_id))
    except Exception:  # unreadable evidence answers nothing here
        return
    if label is not None:
        _refuse(label, topo)
