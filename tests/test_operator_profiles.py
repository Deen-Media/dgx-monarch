from __future__ import annotations

from dataclasses import replace

import pytest

from dgx_monarch.config_schema import ClusterConfigError
from dgx_monarch.operator_profiles import (
    HardwareObservation,
    ProfileRefusal,
    apply_profile,
    classify_hardware,
    compile_profile,
    resolve_profile,
)


def test_safe_profile_uses_stock_residency_and_keeps_native_rdma_off():
    resolution = resolve_profile("safe", hardware_class="homogeneous_uma")

    assert resolution.cluster_set == {"auto_heal": True, "rdma_latent_return": False}
    assert resolution.worker_args_set == {
        "lora_low_rss": False,
        "slab_weights": False,
    }
    assert resolution.worker_args_unset == ()


def test_balanced_uma_preserves_model_aware_omission_semantics():
    resolution = resolve_profile("balanced", hardware_class="homogeneous_uma")

    assert resolution.worker_args_set == {}
    assert resolution.worker_args_unset == ("lora_low_rss", "slab_weights")
    assert "auto" not in resolution.worker_args_set.values()


@pytest.mark.parametrize("hardware", ["discrete", "mixed", "unknown"])
def test_balanced_is_conservative_without_homogeneous_uma(hardware: str):
    resolution = resolve_profile("balanced", hardware_class=hardware)

    assert resolution.worker_args_set == {
        "lora_low_rss": False,
        "slab_weights": False,
    }
    assert resolution.worker_args_unset == ()


def test_advanced_requires_positive_homogeneous_uma_evidence():
    resolution = resolve_profile("advanced", hardware_class="homogeneous_uma")
    assert resolution.worker_args_set == {
        "lora_low_rss": True,
        "slab_weights": True,
    }

    for hardware in ("discrete", "mixed", "unknown"):
        with pytest.raises(ProfileRefusal) as caught:
            resolve_profile("advanced", hardware_class=hardware)
        assert caught.value.requested_profile == "advanced"
        assert caught.value.hardware_class == hardware
        assert caught.value.recommended_profile == "balanced"


def test_hardware_classification_requires_complete_matching_uma_evidence():
    assert classify_hardware([]) == "unknown"
    assert classify_hardware([HardwareObservation("GB10", True)]) == "homogeneous_uma"
    assert classify_hardware([
        HardwareObservation("GB10", True), HardwareObservation("GB10", True),
    ]) == "homogeneous_uma"
    assert classify_hardware([HardwareObservation(None, True)]) == "unknown"
    assert classify_hardware([HardwareObservation("GB10", None)]) == "unknown"
    assert classify_hardware([
        HardwareObservation("GB10", True), HardwareObservation("GB20", True),
    ]) == "mixed"
    assert classify_hardware([
        HardwareObservation("GB10", True), HardwareObservation("H100", False),
    ]) == "mixed"
    assert classify_hardware([
        HardwareObservation("H100", False), HardwareObservation(None, False),
    ]) == "discrete"


def test_compile_profile_defaults_to_balanced_and_unknown_is_safe():
    resolution = compile_profile()

    assert resolution.name == "balanced"
    assert resolution.hardware_class == "unknown"
    assert resolution.worker_args_set["slab_weights"] is False


def test_every_profile_has_separate_graph_recommendations_and_no_forbidden_output():
    forbidden = {
        "operator_profile", "comfy_managed", "compile_dit", "load_profile",
    }
    for name in ("safe", "balanced", "advanced"):
        resolution = resolve_profile(name, hardware_class="homogeneous_uma")
        emitted = set(resolution.cluster_set) | set(resolution.worker_args_set)
        emitted.update(resolution.worker_args_unset)
        assert not emitted & forbidden
        assert resolution.cluster_set["rdma_latent_return"] is False
        assert resolution.graph_recommendations == {
            "topology": "auto",
            "auto_gate": "first_use",
            "pipeline_depth": 1,
        }
        assert "operator_profile" not in apply_profile({}, resolution)


def test_apply_profile_preserves_unrelated_strict_config_and_input():
    original = {
        "cluster": {"client_bind": "tcp://192.0.2.1:0", "auto_heal": False},
        "hosts": [{"name": "spark-1", "address": "tcp://192.0.2.2:26600"}],
        "worker_args": {"swap_verify": 2, "lora_low_rss": True, "slab_weights": True},
        "fabric": {"custom": {"NCCL_DEBUG": "WARN"}},
    }

    rendered = apply_profile(
        original,
        resolve_profile("balanced", hardware_class="homogeneous_uma"),
    )

    assert original["cluster"]["auto_heal"] is False
    assert original["worker_args"]["slab_weights"] is True
    assert rendered == {
        "cluster": {
            "client_bind": "tcp://192.0.2.1:0",
            "auto_heal": True,
            "rdma_latent_return": False,
        },
        "hosts": original["hosts"],
        "worker_args": {"swap_verify": 2},
        "fabric": original["fabric"],
    }


def test_apply_profile_returns_original_dict_for_effective_noop():
    raw = {
        "cluster": {"auto_heal": True, "rdma_latent_return": False},
        "worker_args": {"slab_weights": False, "lora_low_rss": False},
    }

    rendered = apply_profile(raw, resolve_profile("safe", hardware_class="unknown"))

    assert rendered is raw


def test_apply_profile_never_persists_profile_or_receipt_metadata():
    resolution = resolve_profile("safe", hardware_class="unknown")
    receipt = {"operator_profile": resolution.name}
    rendered = apply_profile({"cluster": {}}, resolution)

    # The setup workflow records the profile name in its receipt; this compiler
    # never copies receipt metadata into the strict runtime config.
    assert receipt == {"operator_profile": "safe"}
    assert "receipt" not in rendered
    assert "operator_profile" not in rendered["cluster"]
    assert "operator_profile" not in rendered.get("worker_args", {})


def test_tampered_resolution_cannot_enable_rdma_or_emit_forbidden_keys():
    safe = resolve_profile("safe", hardware_class="unknown")
    with pytest.raises(ClusterConfigError, match="rdma_latent_return"):
        apply_profile({}, replace(safe, cluster_set={"auto_heal": True, "rdma_latent_return": True}))
    with pytest.raises(ClusterConfigError, match="forbidden"):
        apply_profile({}, replace(safe, worker_args_set={"comfy_managed": False}))


def test_existing_worker_policy_is_validated_after_profile_edits():
    advanced = resolve_profile("advanced", hardware_class="homogeneous_uma")

    with pytest.raises(ClusterConfigError, match="comfy_managed"):
        apply_profile({"worker_args": {"comfy_managed": True}}, advanced)


@pytest.mark.parametrize("name", ["fast", "", "BALANCED"])
def test_unknown_profile_is_rejected(name: str):
    with pytest.raises(ValueError, match="unknown operator profile"):
        resolve_profile(name)


def test_resolution_representation_is_deterministic():
    first = resolve_profile("balanced", hardware_class="homogeneous_uma").as_dict()
    second = resolve_profile("balanced", hardware_class="homogeneous_uma").as_dict()

    assert first == second
    assert list(first) == [
        "name",
        "hardware_class",
        "cluster_set",
        "worker_args_set",
        "worker_args_unset",
        "graph_recommendations",
        "notes",
    ]
