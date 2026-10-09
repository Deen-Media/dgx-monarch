"""Bounded artifact identities used by the worker model-store cache."""
from __future__ import annotations

from collections.abc import Callable


def build_request_artifact_identity(
    unet_name: str,
    lora_stack: list[dict] | None,
    resolver: Callable[[str, str], str],
) -> dict:
    """Build bounded fingerprints for the checkpoint/LoRAs a request consumes."""
    from ..gate_ledger import artifact_set_signature, artifact_signature, comfy_commit

    artifacts = [{
        "kind": "diffusion_models",
        "name": unet_name,
        "signature": artifact_signature(resolver("diffusion_models", unet_name)),
    }]
    for entry in lora_stack or []:
        name = str(entry["name"])
        artifacts.append({
            "kind": "loras",
            "name": name,
            "signature": artifact_signature(resolver("loras", name)),
        })
    unresolved = [
        f"{item['kind']}/{item['name']}"
        for item in artifacts
        if item["signature"] in {"unreadable", "unstable"}
    ]
    if unresolved:
        raise RuntimeError(
            "cannot establish a stable model artifact identity for "
            + ", ".join(unresolved)
        )
    digest = artifact_set_signature(item["signature"] for item in artifacts).current
    return {"digest": digest, "comfy": comfy_commit(), "artifacts": artifacts}


def base_artifact_identity(identity: dict) -> tuple | None:
    """Checkpoint/Comfy portion of a request identity (LoRAs excluded)."""
    artifacts = identity.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return None
    return identity.get("comfy"), artifacts[0]


def assert_stable_identity(current: dict, expected: dict, phase: str) -> None:
    if current != expected:
        from ..mesh_safety import ArtifactBindingError

        raise ArtifactBindingError(f"model/LoRA identity changed while {phase}")
