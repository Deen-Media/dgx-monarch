"""Defer the UNET loader's placement when an identity gate will drop it unused.

With an explicit topology the loader places the model on every rank at node
execution. When the graph's first render then runs an identity ceremony, the
ceremony unloads every rank before its first proof render and loads the file
again (issue #157): one wasted full load per rank. The ceremony's unload
stays, because its first leg must come from a clean load from the file, so the
loader leaves the placement to that leg instead. The load then happens once,
inside the first proof render (or, on a graph with nothing to compare, the
gate's no-render swap cycle, nodes/gate_nothing_to_gate.py), and the gate's
trust is unchanged.

The decision reads the prompt from the loader to the nodes that render its
model, collecting the LoRA stack on the way, and asks what the first-use gate
(``auto_gate.maybe_auto_gate``) would do for each one. It defers only when a
ceremony will run. Any doubt keeps the eager load: a Fleet consumer, a
BasicScheduler (DGX Monarch), which loads the model itself before the
ceremony, a node this walk does not know, a linked LoRA widget, a chain that
reaches no loader, or any error. A wrong prediction costs nothing extra either way: an unneeded
deferral moves the load to the first render, and a missed one is the double
load this module removes.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..log import get_logger
from .graph_live import is_link, live_nodes

log = get_logger(__name__)

# The journal reason ``first_render.load_deferred`` prints. The gate loads it
# fresh for its first proof render, or, where it has nothing to compare, for
# its swap cycle with no proof render (nodes/gate_nothing_to_gate.py).
REASON = "the identity gate unloads every rank and loads it fresh before the first render"

# Read-only after import. Model transforms that keep the gate's combination.
_PASSTHROUGH = frozenset({
    "DGXMonarchModelSamplingMiniMaxH3", "DGXMonarchModelSamplingSD3",
    "DGXMonarchSkipLayerGuidance", "DGXMonarchMiniMaxH3ContextWindows",
    "DGXMonarchQwenImage21Cache",
})
_LORA = "DGXMonarchLoraLoader"
# Read-only after import: each loader class and the model slot it fills.
_LOADERS = {"DGXMonarchUNETLoader": "cond", "DGXMonarchUncondUNETLoader": "uncond"}
# Nodes that render the model through the first-use gate.
_GATED = frozenset({
    "DGXMonarchKSampler", "DGXMonarchKSamplerAdvanced", "DGXMonarchKSamplerPipeline",
    "DGXMonarchBasicGuider", "DGXMonarchCFGGuider", "DGXMonarchDualModelGuider",
})
# Its explicit ceremony unloads every rank at its baseline, whatever the ledger says.
_EXPLICIT_GATE = "DGXMonarchIdentityGate"
# Loads the model itself to compute its sigmas (``compute_sigmas`` calls
# ``store_fsdp.ensure``) before any ceremony, which then drops that load: a
# deferral saves nothing there, so such a graph keeps the eager load.
_LOADS_ITSELF = frozenset({"DGXMonarchBasicScheduler"})


def ceremony_drops_load(spec: Any, handle: Any, prompt: object, unique_id: object) -> bool:
    """Whether a gate will unload this loader's placement before the first render."""
    try:
        if not isinstance(prompt, Mapping) or unique_id is None:
            return False
        nodes = {str(key): node for key, node in prompt.items() if isinstance(node, Mapping)}
        live = live_nodes(nodes)
        model = _bound(spec, handle)
        for endpoint, slot_input, stack in _endpoints(nodes, live, str(unique_id)):
            if nodes[endpoint].get("class_type") == _EXPLICIT_GATE:
                return True
            if _ceremony_will_run(*_combination(nodes, endpoint, slot_input, model, stack)):
                return True
        return False
    except LookupError:  # the walk met something it cannot read
        return False
    except Exception as exc:
        log.debug("loader deferral not decided (%r); the loader places the model now", exc)
        return False


def _bound(spec: Any, handle: Any) -> Any:
    """The loader's spec on the live handle ``ensure_live`` returned."""
    from .gate_identity import model_with_worker_overrides

    return model_with_worker_overrides(spec, {}, handle=handle)


def _consumers(nodes: Mapping[str, Mapping], live: frozenset[str], source: str):
    for key, node in nodes.items():
        if key not in live:
            continue
        inputs = node.get("inputs")
        for name, value in (inputs.items() if isinstance(inputs, Mapping) else ()):
            if is_link(value) and str(value[0]) == source:
                yield key, name


def _lora_entry(node: Mapping) -> tuple[str, float]:
    inputs = node.get("inputs") or {}
    name, strength = inputs.get("lora_name"), inputs.get("strength_model")
    if not isinstance(name, str) or isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise LookupError("a LoRA widget is linked or unreadable")
    return name, float(strength)


def _endpoints(nodes, live, loader: str) -> list[tuple[str, str, list[tuple[str, float]]]]:
    """(rendering node, its model input, the LoRA stack on the way) for this loader."""
    found: list[tuple[str, str, list[tuple[str, float]]]] = []
    pending: list[tuple[str, list[tuple[str, float]]]] = [(loader, [])]
    seen: set[tuple[str, tuple]] = set()
    while pending:
        source, stack = pending.pop()
        if (source, tuple(stack)) in seen:
            continue
        seen.add((source, tuple(stack)))
        for key, name in _consumers(nodes, live, source):
            kind = nodes[key].get("class_type")
            if kind == _LORA and name == "model":
                pending.append((key, [*stack, _lora_entry(nodes[key])]))
            elif kind in _PASSTHROUGH and name == "model":
                pending.append((key, stack))
            elif (kind in _GATED or kind == _EXPLICIT_GATE) and name in ("model", "model_negative"):
                found.append((key, name, stack))
            elif kind in _LOADS_ITSELF:
                raise LookupError(f"{kind} loads the model itself before the ceremony")
            else:
                raise LookupError(f"{kind} consumes the model")  # Fleet, or a node this walk does not know
    return found


def _chain_to_loader(nodes: Mapping[str, Mapping], link: object) -> tuple[Mapping, list[tuple[str, float]]]:
    """Walk one model input back to its loader: (loader node, LoRA stack in load order)."""
    stack: list[tuple[str, float]] = []
    for _hop in range(64):
        if not is_link(link) or str(link[0]) not in nodes:  # type: ignore[index]
            raise LookupError("the model input reaches no node")
        node = nodes[str(link[0])]  # type: ignore[index]
        kind = node.get("class_type")
        if kind in _LOADERS:
            return node, stack[::-1]
        if kind == _LORA:
            stack.append(_lora_entry(node))
        elif kind not in _PASSTHROUGH:
            raise LookupError(f"{kind} sits on the model chain")
        link = (node.get("inputs") or {}).get("model")
    raise LookupError("the model chain does not end")


def _spec_from_loader(model: Any, node: Mapping, stack, slot: str) -> Any:
    from .common import ModelSpec

    inputs = node.get("inputs") or {}
    unet, dtype = inputs.get("unet_name"), inputs.get("weight_dtype", "default")
    if not isinstance(unet, str) or not isinstance(dtype, str):
        raise LookupError("a loader widget is linked")
    options = {} if dtype == "default" else {"weight_dtype": dtype}
    spec = ModelSpec(mesh=model.mesh, unet_name=unet, options=options, slot=slot)
    for name, strength in stack:
        spec = spec.with_lora(name, strength)
    return spec


def _combination(nodes, endpoint: str, slot_input: str, model: Any, stack) -> tuple[Any, dict | None]:
    """The gate's (model, unconditional request) for one rendering node."""
    for name, strength in stack:
        model = model.with_lora(name, strength)
    node = nodes[endpoint]
    if node.get("class_type") != "DGXMonarchDualModelGuider":
        return model, None
    inputs = node.get("inputs") or {}
    other = "model" if slot_input == "model_negative" else "model_negative"
    if other not in inputs:  # no second model wired: plain CFG on one model
        return model, None
    loader, other_stack = _chain_to_loader(nodes, inputs[other])
    other_slot = _LOADERS[str(loader.get("class_type"))]
    peer = _spec_from_loader(model, loader, other_stack, other_slot)
    if slot_input == "model":
        return model, peer.request_dict()
    return peer, model.request_dict()


def _ceremony_will_run(model: Any, uncond: dict | None) -> bool:
    """What ``maybe_auto_gate`` would decide, read without claiming anything."""
    from . import common, gate_inconclusive

    context = common._auto_gate_context(
        model, "ksampler", None, None, *(() if uncond is None else (uncond,)))
    if context is None:
        return False
    state, token = context
    verdict = common._process_gate_verdict(token)
    if verdict in ("FAIL", "INCONCLUSIVE", "ERROR"):
        return False
    if state in ("pass", "fail", gate_inconclusive.GRANTED_STATE):
        return False
    if token in common._AUTO_GATE_SESSION and (verdict != "PASS" or state in ("unknown", "stale")):
        return False
    return True
