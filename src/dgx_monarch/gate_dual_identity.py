"""The ledger identity of a dual-model identity-gate verdict.

A dual-model render (Ideogram 4) runs two checkpoints: the conditional model in
the first slot and the unconditional model in the second. Its first-use proof
binds both, so its verdict needs a row identity no single-model verdict can
share: a combination key over both model specifications and an artifact digest
over both artifact sets, the conditional set first. A single-model PASS then
never answers a dual render, and a dual PASS never answers a render of either
model alone. The capability context is the single-model one: the key and the
digest are what set a dual row apart.

The driver writes these rows (nodes/gate_dual.py) and the Worker recomputes the
same values from the request's two wire specifications and its own two
artifact sets before it accepts a dual grant (mesh_residency.py).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from .gate_ledger import ArtifactSetSignature, artifact_set_signature


def _spec(spec: Mapping[str, Any]) -> list[Any]:
    """The fields ``gate_ledger.combo_key`` reads, in its order."""
    return [
        str(spec["unet_name"]),
        sorted(dict(spec.get("options") or {}).items()),
        sorted(str(entry["name"]) for entry in spec.get("loras") or []),
    ]


def dual_combo_key(model_spec: Mapping[str, Any], uncond_spec: Mapping[str, Any]) -> str:
    """The combination key of a dual-model render: both models, options and LoRA names."""
    payload = json.dumps(["dual-model", _spec(model_spec), _spec(uncond_spec)], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def dual_artifact_signature(artifact_sets: Sequence[Mapping[str, Any]]) -> ArtifactSetSignature:
    """One digest over both slots' artifact signatures, conditional slot first."""
    return artifact_set_signature(
        str(item["signature"]) for identity in artifact_sets for item in identity["artifacts"])


def dual_identity_summary(artifact_sets: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """The ``comfy`` and ``digest`` a dual grant names for its two artifact sets.

    Both sets come from one ComfyUI checkout, so they must name one commit.
    """
    if len(artifact_sets) != 2:
        raise RuntimeError("a dual-model Gate grant binds exactly two artifact sets")
    commits = {str(identity.get("comfy")) for identity in artifact_sets}
    if len(commits) != 1:
        raise RuntimeError("the two model slots name different ComfyUI commits")
    return {"comfy": commits.pop(), "digest": dual_artifact_signature(artifact_sets).current}
