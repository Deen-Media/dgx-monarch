"""Prompt-bound execution observations, distinct from immutable requested settings."""
from __future__ import annotations

import math
import re
from copy import deepcopy
from typing import Any

from .contracts import StudioError, canonical, timestamp

EVENTS = frozenset({"execution_start", "execution_cached", "executing", "progress", "progress_state", "execution_success",
                    "execution_error", "execution_interrupted"})
REFUSAL_HEAD_CHARS = 2000
# The same markers as dgx_monarch.consent_descriptor and dgx_monarch.refusal,
# repeated here because importing those modules loads the Worker stack.
DESCRIPTOR_BEGIN = "<<<DGXM-CONSENT-V1 "
DESCRIPTOR_END = " DGXM-CONSENT-END>>>"
DESCRIPTOR_CHARS = 4096 + len(DESCRIPTOR_BEGIN) + len(DESCRIPTOR_END)
_REMEDY = re.compile(r' Open the DGX Monarch panel and click "[^"\n]{1,200}" on the card for this model, then queue the '
                     r'render again\. Headless: set [A-Za-z0-9_]{1,64}=\S{1,64} on the driver and run again\.'
                     r'(?: See docs/TROUBLESHOOTING\.md #\d{1,4}\.)?| See docs/TROUBLESHOOTING\.md #\d{1,4}\.')


def bounded_refusal(text: str) -> str:
    """Bound a driver refusal while keeping its remedy sentence and consent descriptor.

    A long refusal keeps its head (with the class tag), then the panel or
    troubleshooting sentence and the machine-readable consent block, and says
    how much of the middle it left out.
    """
    if len(text) <= REFUSAL_HEAD_CHARS:
        return text
    rest, descriptor = text, ""
    start = text.find(DESCRIPTOR_BEGIN)
    end = text.find(DESCRIPTOR_END, start + 1) if start >= 0 else -1
    if start >= 0 and end >= 0 and text.count(DESCRIPTOR_BEGIN) == 1:
        block = text[start:end + len(DESCRIPTOR_END)]
        if len(block) <= DESCRIPTOR_CHARS:
            descriptor, rest = block, text[:start] + text[end + len(DESCRIPTOR_END):]
    remedies = [match.group(0) for match in _REMEDY.finditer(rest)][-1:]
    for remedy in remedies:
        index = rest.rfind(remedy)
        rest = rest[:index] + rest[index + len(remedy):]
    head = rest[:REFUSAL_HEAD_CHARS]
    omitted = len(rest) - len(head)
    parts = [head, f" [{omitted} characters omitted]" if omitted else "", *remedies,
             f" {descriptor}" if descriptor else ""]
    return "".join(parts)


def observation(message: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Parse one driver event; driver data that cannot be bounded is ignored, never raised."""
    try:
        return _observation(message)
    except (RecursionError, OverflowError, StudioError):
        return None


def _observation(message: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    kind, data = message.get("type"), message.get("data")
    if not isinstance(kind, str) or kind not in EVENTS or not isinstance(data, dict) or not isinstance(data.get("prompt_id"), str):
        return None
    # Never attribute unkeyed progress or the global last-render telemetry to a job.
    safe = {key: deepcopy(data[key]) for key in ("prompt_id", "timestamp", "node", "node_id", "node_type",
                                               "value", "max", "nodes", "exception_type") if key in data}
    if len(canonical(safe)) > 8192:
        return None
    event = {"type": kind, "data": safe}
    updates: dict[str, Any] = {"last_observed_at": timestamp(), "observation_source": "driver_event", "phase": kind}
    event_time = data.get("timestamp")
    when = event_time / 1000 if isinstance(event_time, (int, float)) and not isinstance(event_time, bool) else None
    if when is not None and not math.isfinite(when):
        return None
    if kind == "execution_start":
        updates.update(state="running", started_at=when)
    elif kind in {"progress", "progress_state", "executing", "execution_cached"}:
        updates["state"] = "running"
        if kind == "progress":
            value, maximum = data.get("value"), data.get("max")
            if (isinstance(value, (int, float)) and not isinstance(value, bool)
                    and isinstance(maximum, (int, float)) and not isinstance(maximum, bool) and maximum > 0
                    and 0 <= value <= maximum):
                updates["progress"] = {"value": value, "max": maximum, "node": data.get("node")}
        elif kind == "executing":
            updates["current_node"] = data.get("node")
        elif kind == "progress_state" and isinstance(data.get("nodes"), dict):
            updates["node_progress"] = deepcopy(data["nodes"])
    else:
        updates.update(state={"execution_success": "succeeded", "execution_error": "failed",
                              "execution_interrupted": "cancelled"}[kind], progress=None, ended_at=when)
        if kind == "execution_error":
            updates["error"] = {"code": str(data.get("exception_type", "execution_error"))[:200],
                                "message": bounded_refusal(str(data.get("exception_message", "Driver reported execution failure.")))}
    return event, updates


def model_identities(prompt: dict[str, Any]) -> list[dict[str, Any]]:
    names = {"ckpt_name", "unet_name", "model_name", "vae_name", "clip_name", "clip_name1", "clip_name2",
             "clip_name3", "lora_name", "control_net_name", "upscale_model", "model_path"}
    return [{"node_id": node_id, "input": key, "name": value, "fingerprint": None,
             "fingerprint_status": "not_reported_by_driver"}
            for node_id, node in prompt.items() for key, value in node.get("inputs", {}).items()
            if key in names and isinstance(value, str)]


def requested_topology(prompt: dict[str, Any]) -> list[dict[str, Any]]:
    names = {"world_size", "tp_size", "sp_size", "pp_size", "topology", "execution_mode", "attention_mode",
             "ulysses_degree", "ring_degree", "fsdp", "device", "target", "mode"}
    return [{"node_id": node_id, "class_type": node["class_type"], "parameters": {
        key: deepcopy(value) for key, value in node.get("inputs", {}).items() if key in names}}
        for node_id, node in prompt.items() if node["class_type"].startswith("DGXMonarch")]
