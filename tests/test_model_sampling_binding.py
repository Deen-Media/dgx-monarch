"""Per-render model-sampling identity stays outside model residency.

The gate and Fleet paths bind the complete wire request, while the coarser
ledger combination and model-store keys describe only checkpoint residency. A
sampling patch must therefore invalidate an existing authorization without
forcing a model reload.
"""
from __future__ import annotations

import copy

import pytest

import dgx_monarch.mesh_safety as mesh_safety
from dgx_monarch.actor.model_store import ModelStore
from dgx_monarch.mesh_safety import (
    FLEET_RESIDENCY_CAPABILITY,
    ArtifactBindingError,
    assert_fleet_residency_grant,
    assert_request_artifact_binding,
    request_combo_key,
)


def _model_request(name: str, shift: float | None) -> dict:
    return {
        "unet_name": name,
        "options": {"weight_dtype": "default"},
        "loras": [{"name": "style.safetensors", "strength": 0.5}],
        "model_sampling": None if shift is None else {"kind": "sd3", "shift": shift},
    }


def _artifacts() -> list[dict]:
    return [
        {"digest": "primary", "comfy": "commit", "artifacts": []},
        {"digest": "uncond", "comfy": "commit", "artifacts": []},
    ]


def _fleet_gate_grant(model, artifacts, context, *, uncond_model=None):
    from dgx_monarch import __version__
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION, gate_verdict_token

    combo = mesh_safety.request_combo_key(model)
    identity = artifacts[0]
    return {
        "gate_protocol": GATE_PROTOCOL_VERSION,
        "dgx_monarch": __version__,
        "comfy": identity["comfy"],
        "artifact_digest": identity["digest"],
        "gate_token": list(gate_verdict_token(
            combo, identity["digest"], identity["comfy"], context)),
        "combo_key": combo,
        "model_request": copy.deepcopy(model),
        "uncond_model_request": copy.deepcopy(uncond_model),
        "artifact_sets": copy.deepcopy(artifacts),
        "capability_context": copy.deepcopy(context),
    }


def test_sampling_shift_is_not_part_of_combo_or_model_store_identity():
    image = _model_request("kandinsky-image.safetensors", 3.0)
    video = copy.deepcopy(image)
    video["model_sampling"]["shift"] = 5.0
    default = copy.deepcopy(image)
    default["model_sampling"] = None

    assert request_combo_key(image) == request_combo_key(video)
    assert request_combo_key(image) == request_combo_key(default)

    def store_keys(request: dict) -> tuple[tuple, tuple]:
        return ModelStore.make_keys(
            request["unet_name"],
            request["options"],
            request["loras"],
            "bf16",
        )

    assert store_keys(image) == store_keys(video)
    assert store_keys(image) == store_keys(default)


@pytest.mark.parametrize(
    ("slot", "message"),
    [
        ("model", "full model request changed"),
        ("uncond_model", "full unconditional model request changed"),
    ],
)
def test_gate_binding_rejects_sampling_shift_mutation(slot: str, message: str):
    primary = _model_request("kandinsky-image.safetensors", 3.0)
    uncond = _model_request("kandinsky-image-uncond.safetensors", 3.0)
    artifacts = _artifacts()
    binding = {
        "combo_key": request_combo_key(primary),
        "model_request": copy.deepcopy(primary),
        "uncond_model_request": copy.deepcopy(uncond),
        "artifact_sets": copy.deepcopy(artifacts),
    }
    request = {
        "model": copy.deepcopy(primary),
        "uncond_model": copy.deepcopy(uncond),
        "_dgxm_artifact_binding": binding,
    }
    assert assert_request_artifact_binding(request, artifacts)

    changed = copy.deepcopy(request)
    changed[slot]["model_sampling"]["shift"] = 5.0
    # The coarse combination remains reusable, but the exact ceremony does not.
    assert request_combo_key(changed["model"]) == binding["combo_key"]
    with pytest.raises(ArtifactBindingError, match=message):
        assert_request_artifact_binding(changed, artifacts)


def test_mesh_safety_aliases_resolve_old_namespace_monkeypatches(monkeypatch):
    model = _model_request("model.safetensors", None)
    binding = {
        "combo_key": "patched-combo",
        "model_request": copy.deepcopy(model),
        "uncond_model_request": None,
        "artifact_sets": [],
    }
    request = {
        "model": copy.deepcopy(model),
        "_dgxm_artifact_binding": binding,
    }
    monkeypatch.setattr(
        mesh_safety, "request_combo_key", lambda _model: "patched-combo")
    assert mesh_safety.assert_request_artifact_binding(request, [])

    fleet_request = {"model": copy.deepcopy(model), "_dgxm_fleet_job": True}
    monkeypatch.setattr(
        mesh_safety, "fleet_policy_is_risky", lambda *_args: False)
    assert not mesh_safety.assert_fleet_residency_grant(
        fleet_request, [], effective_worker_args={})

    monkeypatch.setattr(
        mesh_safety, "FLEET_RESIDENCY_CAPABILITY", "patched-capability")
    artifacts = [{"digest": "primary", "comfy": "commit", "artifacts": []}]
    context = {
        "capability": "patched-capability",
        "rank_world": 1,
        "worker_args": {},
    }
    fleet_request["_dgxm_fleet_residency_grant"] = _fleet_gate_grant(
        model, artifacts, context)
    assert mesh_safety.assert_fleet_residency_grant(
        fleet_request, artifacts, effective_worker_args={})


@pytest.mark.parametrize(
    ("slot", "message"),
    [
        ("model", "full fleet model request changed"),
        ("uncond_model", "full fleet unconditional model request changed"),
    ],
)
def test_fleet_grant_rejects_sampling_shift_mutation(slot: str, message: str):
    primary = _model_request("kandinsky-video.safetensors", 5.0)
    uncond = _model_request("kandinsky-video-uncond.safetensors", 5.0)
    artifacts = _artifacts()
    policy = {"slab_weights": True, "lora_low_rss": True}
    context = {
        "capability": FLEET_RESIDENCY_CAPABILITY,
        "rank_world": 1,
        "worker_args": copy.deepcopy(policy),
    }
    grant = _fleet_gate_grant(
        primary, artifacts, context, uncond_model=uncond)
    request = {
        "model": copy.deepcopy(primary),
        "uncond_model": copy.deepcopy(uncond),
        "_dgxm_fleet_job": True,
        "_dgxm_fleet_residency_grant": grant,
    }
    assert assert_fleet_residency_grant(
        request,
        artifacts,
        effective_worker_args=policy,
        rank_world=1,
    )

    changed = copy.deepcopy(request)
    changed[slot]["model_sampling"]["shift"] = 3.0
    # Full-request drift must be caught after the coarse combo still matches.
    assert request_combo_key(changed["model"]) == grant["combo_key"]
    with pytest.raises(RuntimeError, match=message):
        assert_fleet_residency_grant(
            changed,
            artifacts,
            effective_worker_args=policy,
            rank_world=1,
        )
