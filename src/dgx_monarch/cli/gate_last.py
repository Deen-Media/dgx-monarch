"""Choose the prompt that `dgxm gate --last` gates and build its gate graph.

The CLI and Studio both call this module, so they gate the same prompt with the
same Identity Gate graph. It opens no network connection and imports neither
ComfyUI nor torch; its only read is the local gate report file. Callers fetch
history and submit the graph themselves.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

SAMPLERS = frozenset({"DGXMonarchKSampler", "DGXMonarchKSamplerAdvanced", "DGXMonarchKSamplerPipeline"})
GATE_INPUTS = ("model", "positive", "negative", "latent_image")


def latest_entry(history: object) -> tuple[str, Any] | None:
    """Return the newest finished prompt in a ComfyUI /history response.

    ComfyUI keeps history in the order prompts finished, and `max_items`
    returns the end of that order, so the last key is the newest prompt. The
    queue number cannot rank prompts: Queue Front stores it negated, and a
    number given in the request arrives as a float. None means no history.
    """
    if not isinstance(history, Mapping) or not history:
        return None
    identity = list(history)[-1]
    return str(identity), history[identity]


def source_graph(entry: object) -> dict[str, Any]:
    """The API graph ComfyUI stored with a history entry."""
    serialized = entry.get("prompt") if isinstance(entry, Mapping) else None
    if not isinstance(serialized, (list, tuple)) or len(serialized) < 3 or not isinstance(serialized[2], Mapping):
        raise ValueError("the last prompt in ComfyUI history carries no API graph")
    return dict(serialized[2])


def prompt_input(value: Any, default: Any, cast: Callable[[Any], Any]) -> Any:
    """Preserve a Comfy API link; coerce only literal widget values."""
    if isinstance(value, list) and len(value) == 2:
        return value
    return cast(default if value is None else value)


def identity_gate(source: Mapping[str, Any], run_id: str) -> tuple[str, dict[str, Any]] | None:
    """Swap the first Monarch sampler for the Identity Gate node.

    Returns the sampler's node id and a new graph, or None when the graph has
    no Monarch sampler. The source graph is never changed. A sampler without
    the four gate inputs, or with a seed or cfg the gate cannot read, raises
    ValueError or TypeError.
    """
    prompt = deepcopy(dict(source))
    target = next((node_id for node_id, node in prompt.items()
                   if isinstance(node, Mapping) and node.get("class_type") in SAMPLERS), None)
    if target is None:
        return None
    inputs = prompt[target].get("inputs")
    if not isinstance(inputs, Mapping) or not set(GATE_INPUTS) <= inputs.keys():
        raise ValueError("the last prompt's sampler lacks the model, positive, negative or latent_image input")
    prompt[target] = {"class_type": "DGXMonarchIdentityGate", "inputs": {
        **{name: inputs[name] for name in GATE_INPUTS},
        "noise_seed": prompt_input(inputs.get("noise_seed", inputs.get("seed", 42)), 42, int),
        "steps": 2, "cfg": prompt_input(inputs.get("cfg", 1.0), 1.0, float),
        "sampler_name": inputs.get("sampler_name", "euler"),
        "scheduler": inputs.get("scheduler", "simple"),
        "run_id": run_id,
    }}
    return target, prompt


def find_gate_report(report_dir: str, run_id: str) -> dict | None:
    """Find only the report produced by this gate run.

    A shared JSONL file can contain concurrent node runs and old verdicts, so
    accepting its last row can report success for the wrong prompt.
    """
    try:
        lines = (Path(report_dir) / "dgxm_gate_reports.jsonl").read_text().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            report = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(report, dict) and report.get("run_id") == run_id:
            return report
    return None
