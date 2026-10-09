"""Adapter and render-node regressions."""
from __future__ import annotations

import sys
import types

import pytest
import torch

from dgx_monarch.adapters import attention_patches, base, pixeldit
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.cfg_parallel import make_cfg_parallel_wrapper
from dgx_monarch.adapters.zimage import ZImageAdapter
from dgx_monarch.nodes import (
    common,
    latent_outputs,
    render_result,
    render_submit,
    render_validation,
    samplers,
)
from dgx_monarch.nodes import gate as gate_mod
from dgx_monarch.nodes.render_validation import (
    PackedCfgParallelError,
    PackedDataParallelError,
)
from dgx_monarch.refusal import RefusalClass, parse_refusal_tag
from dgx_monarch.topology import Topology


def test_shard_seq_zero_pads_nondivisible_sequence_by_default(monkeypatch):
    monkeypatch.setattr(base, "sp_world", lambda: 2)
    monkeypatch.setattr(base, "sp_rank", lambda: 1)
    local, orig = base.shard_seq(torch.ones(1, 3, 1))
    assert orig == 3
    assert torch.equal(local, torch.tensor([[[1.0], [0.0]]]))


def test_adapter_can_explicitly_require_exact_divisibility(monkeypatch):
    monkeypatch.setattr(base, "sp_world", lambda: 2)
    monkeypatch.setattr(base, "sp_rank", lambda: 0)
    with pytest.raises(UnsupportedModelError, match="requires exact divisibility") as caught:
        base.shard_seq(torch.zeros(1, 3, 4), allow_padding=False)
    message = str(caught.value)
    assert "product matches the worker world" in message
    assert "mode=local" in message and "gpus_per_host=1" in message
    assert "topology 'single'" not in message


def test_ideogram_warns_and_drops_meaningful_text_padding_mask(monkeypatch):
    warnings = []
    monkeypatch.setattr(pixeldit._segment_mask_dropped, "warn", warnings.append)
    indicator = torch.full((1, 6), 9, dtype=torch.long)

    pixeldit._drop_padded_text_segment_mask(
        indicator, torch.tensor([[1, 1, 0, 0]]), l_text=4
    )

    assert torch.equal(indicator, torch.tensor([[9, 9, 0, 0, 9, 9]]))
    assert len(warnings) == 1
    assert "segment mask" in warnings[0]


def _finish(results, dp, monkeypatch):
    monkeypatch.setattr(render_result, "read_latent_result", lambda tensor: tensor)
    monkeypatch.setattr(
        render_result, "verify_cross_rank_signatures", lambda _results, _topo: None
    )
    return render_result._finish_render(results, types.SimpleNamespace(dp=dp), {})


def test_dp_custom_denoised_concatenates_in_dp_rank_order(monkeypatch):
    results = [
        {"rank": 1, "dp_rank": 1, "latent": torch.tensor([[2.0]]),
         "latent_extra": {"denoised": torch.tensor([[20.0]])}},
        {"rank": 0, "dp_rank": 0, "latent": torch.tensor([[1.0]]),
         "latent_extra": {"denoised": torch.tensor([[10.0]])}},
    ]
    out = _finish(results, 2, monkeypatch)
    assert torch.equal(out["samples"], torch.tensor([[1.0], [2.0]]))
    assert torch.equal(out["_dgxm_denoised"], torch.tensor([[10.0], [20.0]]))


def test_dp_custom_denoised_requires_every_leader(monkeypatch):
    results = [
        {"rank": 0, "dp_rank": 0, "latent": torch.tensor([[1.0]]),
         "latent_extra": {"denoised": torch.tensor([[10.0]])}},
        {"rank": 1, "dp_rank": 1, "latent": torch.tensor([[2.0]])},
    ]
    with pytest.raises(RuntimeError, match="missing denoised"):
        _finish(results, 2, monkeypatch)


@pytest.mark.parametrize(
    "second",
    [
        torch.ones(1, 3),
        torch.ones(1, 2, dtype=torch.float64),
    ],
)
def test_dp_leader_samples_reject_structure_or_dtype_drift(monkeypatch, second):
    with pytest.raises(RuntimeError, match="changed non-batch Tensor structure"):
        _finish([
            {"rank": 0, "dp_rank": 0, "latent": torch.zeros(1, 2)},
            {"rank": 1, "dp_rank": 1, "latent": second},
        ], 2, monkeypatch)


class NestedTensor:
    """Minimal stand-in for comfy.nested_tensor.NestedTensor (one tensor per modality)."""

    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


@pytest.fixture
def direct_nested_tensor_surface(monkeypatch):
    comfy = sys.modules.get("comfy") or types.ModuleType("comfy")
    nested = types.ModuleType("comfy.nested_tensor")
    nested.NestedTensor = NestedTensor
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", nested)
    monkeypatch.setattr(comfy, "nested_tensor", nested, raising=False)
    return NestedTensor


def test_nested_reconstruction_passes_direct_list_payload(monkeypatch):
    class StrictListNestedTensor:
        def __init__(self, tensors):
            if type(tensors) is not list:
                raise TypeError("direct modalities must be a list")
            self.tensors = tensors
            self.is_nested = True

        def unbind(self):
            return self.tensors

    comfy = sys.modules.get("comfy") or types.ModuleType("comfy")
    nested = types.ModuleType("comfy.nested_tensor")
    nested.NestedTensor = StrictListNestedTensor
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", nested)
    monkeypatch.setattr(comfy, "nested_tensor", nested, raising=False)
    source = StrictListNestedTensor([torch.zeros(1, 2), torch.ones(1, 3)])

    rebuilt = latent_outputs.concatenate_latent_batches([source], "packed result")

    assert type(rebuilt) is StrictListNestedTensor
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(rebuilt.unbind(), source.unbind(), strict=True)
    )


def test_nested_dp_refusal_uses_public_typed_contract(direct_nested_tensor_surface):
    model = types.SimpleNamespace(
        mesh=types.SimpleNamespace(topology_preset="dp2")
    )
    samples = NestedTensor((torch.zeros(1, 2), torch.zeros(1, 3)))

    with pytest.raises(
        ValueError,
        match="packed multi-modality latents cannot be data-parallel",
    ):
        render_validation.validate_render_topology(
            model,
            Topology(dp=2, world=2),
            samples,
        )


def test_nested_dp_refusal_declares_the_physics_class(direct_nested_tensor_surface):
    """A packed latent cannot be data-parallel at any setting, so the refusal
    carries the same class its CFG sibling does. Untagged, the sentence reads
    like a configuration complaint an operator could waive."""
    model = types.SimpleNamespace(
        mesh=types.SimpleNamespace(topology_preset="dp2")
    )
    samples = NestedTensor((torch.zeros(1, 2), torch.zeros(1, 3)))

    with pytest.raises(PackedDataParallelError) as raised:
        render_validation.validate_render_topology(
            model, Topology(dp=2, world=2), samples)

    tag = parse_refusal_tag(str(raised.value))
    assert tag.refusal_class is RefusalClass.PHYSICS
    assert tag.waivable is False


def test_nested_auto_dp_refusal_names_the_resolved_topology(
    direct_nested_tensor_surface,
):
    model = types.SimpleNamespace(
        mesh=types.SimpleNamespace(topology_preset="auto")
    )
    samples = NestedTensor((torch.zeros(2, 2), torch.zeros(2, 3)))

    with pytest.raises(
        ValueError,
        match=r"resolved topology for preset 'auto'.*cannot be data-parallel",
    ):
        render_validation.validate_render_topology(
            model,
            Topology(dp=2, world=4, ulysses=2),
            samples,
        )


def test_packed_dp_refuses_before_gate_or_policy_mutation(
    monkeypatch,
    direct_nested_tensor_surface,
):
    policy = {"lora_low_rss": True, "slab_weights": True}
    handle = types.SimpleNamespace(world=2)
    model = common.ModelSpec(
        mesh=common.MeshSpec(
            handle=handle,
            topology_preset="dp2",
            attention="TORCH_FLASH",
            sync_ulysses=False,
            worker_args=policy,
            auto_gate="first_use",
        ),
        unet_name="model.safetensors",
    )
    latent = {
        "samples": NestedTensor((torch.zeros(1, 2), torch.zeros(1, 3)))
    }

    def forbidden(name):
        def call(*_args, **_kwargs):
            pytest.fail(f"{name} ran before packed-DP refusal")

        return call

    for name in ("_maybe_auto_gate", "_quarantine_unproven_paths", "submit_render"):
        monkeypatch.setattr(common, name, forbidden(name))
    for name in ("_enforce_persisted_quarantine", "model_with_worker_overrides"):
        monkeypatch.setattr(render_submit, name, forbidden(name))

    before = dict(policy)
    with pytest.raises(
        ValueError,
        match="packed multi-modality latents cannot be data-parallel",
    ):
        common.run_render(
            model,
            {"kind": "ksampler", "steps": 2},
            latent,
            1.0,
            2,
        )
    assert model.mesh.worker_args is policy
    assert policy == before


def test_packed_dp_pipeline_refuses_before_gate_session_or_sequence_mutation(
    monkeypatch,
    direct_nested_tensor_surface,
):
    policy = {"lora_low_rss": True, "slab_weights": True}
    model = common.ModelSpec(
        mesh=common.MeshSpec(
            handle=types.SimpleNamespace(world=2),
            topology_preset="dp2",
            attention="TORCH_FLASH",
            sync_ulysses=False,
            worker_args=policy,
        ),
        unet_name="model.safetensors",
    )
    latent = {
        "samples": NestedTensor((torch.zeros(1, 2), torch.zeros(1, 3)))
    }
    pipe = common.RenderPipeline(depth=2)

    class ExistingPending:
        _state = "pending"
        cancelled = False
        abandoned = False

        def cancel(self):
            self.cancelled = True

        def result(self):
            raise RuntimeError("still running")

        def abandon(self):
            self.abandoned = True
            self._state = "closed"

    existing = ExistingPending()
    pipe._inflight.append(existing)
    monkeypatch.setattr(
        common,
        "auto_gate_required",
        lambda *_args, **_kwargs: pytest.fail("packed-DP request reached Gate lookup"),
    )
    monkeypatch.setattr(
        common,
        "submit_render",
        lambda *_args, **_kwargs: pytest.fail("packed-DP request reached submission"),
    )

    with pytest.raises(
        ValueError,
        match="packed multi-modality latents cannot be data-parallel",
    ):
        pipe.push(
            model,
            {"kind": "ksampler", "steps": 2},
            latent,
            1.0,
            2,
        )
    assert pipe._seq == 0
    assert not pipe._inflight
    assert existing.cancelled
    assert existing.abandoned
    assert policy == {"lora_low_rss": True, "slab_weights": True}


def test_packed_dp_identity_gate_refuses_before_mesh_session_or_quarantine(
    monkeypatch,
    direct_nested_tensor_surface,
):
    policy = {"lora_low_rss": True, "slab_weights": True}

    class Handle:
        world = 2

        def call_all(self, *_args, **_kwargs):
            pytest.fail("packed-DP Gate reached a worker side effect")

    handle = Handle()
    model = common.ModelSpec(
        mesh=common.MeshSpec(
            handle=handle,
            topology_preset="dp2",
            attention="TORCH_FLASH",
            sync_ulysses=False,
            worker_args=policy,
        ),
        unet_name="model.safetensors",
    )
    latent = {
        "samples": NestedTensor((torch.zeros(1, 2), torch.zeros(1, 3)))
    }
    monkeypatch.setattr(
        gate_mod,
        "ensure_live",
        lambda *_args, **_kwargs: pytest.fail("packed-DP Gate healed the mesh"),
    )
    monkeypatch.setattr(
        gate_mod,
        "_force_stock_quarantine",
        lambda *_args, **_kwargs: pytest.fail("packed-DP Gate quarantined policy"),
    )
    monkeypatch.setattr(
        gate_mod,
        "GateLedger",
        lambda *_args, **_kwargs: pytest.fail("packed-DP Gate opened the ledger"),
    )

    with pytest.raises(
        ValueError,
        match="packed multi-modality latents cannot be data-parallel",
    ):
        gate_mod.run_identity_ceremony(
            model,
            {"kind": "ksampler", "steps": 2},
            latent,
            1.0,
            2,
            "manual",
        )
    assert policy == {"lora_low_rss": True, "slab_weights": True}


def _packed_cluster_request(
    NestedTensor,
    *,
    cached_world: int = 2,
    topology_preset: str = "uly2",
):
    policy = {"lora_low_rss": True, "slab_weights": True}
    config = types.SimpleNamespace(
        hosts=(object(),), source="/cluster.toml", worker_args={})
    handle = types.SimpleNamespace(world=cached_world, config=config)
    model = common.ModelSpec(
        mesh=common.MeshSpec(
            handle=handle,
            topology_preset=topology_preset,
            attention="TORCH_FLASH",
            sync_ulysses=False,
            worker_args=policy,
            auto_gate="first_use",
        ),
        unet_name="model.safetensors",
    )
    latent = {
        "samples": NestedTensor((torch.zeros(1, 2), torch.zeros(1, 3)))
    }
    return model, latent, policy


def test_cluster_packed_preflight_uses_refreshed_smaller_world(
    direct_nested_tensor_surface,
):
    model, latent, policy = _packed_cluster_request(
        direct_nested_tensor_surface, cached_world=4)
    fresh = types.SimpleNamespace(world=2, config=model.mesh.handle.config)
    seen = []

    def ensure(_handle, *, mesh_preflight):
        seen.append(2)
        mesh_preflight(fresh.config, 2)
        return fresh

    bound, handle = common._bind_packed_render_model(
        model, latent, 1.0, ensure_live_fn=ensure)

    assert seen == [2]
    assert handle is fresh and bound.mesh.handle is fresh
    assert bound.mesh.worker_args is policy


@pytest.mark.parametrize(
    ("cached_world", "topology_preset", "configured_world", "expected_dp"),
    [(1, "single", 2, 2), (2, "uly2", 4, 2)],
)
def test_cluster_world_drift_refuses_inside_mesh_resolution_before_side_effects(
    monkeypatch,
    direct_nested_tensor_surface,
    cached_world,
    topology_preset,
    configured_world,
    expected_dp,
):
    from dgx_monarch import mesh as mesh_mod

    model, latent, _policy = _packed_cluster_request(
        direct_nested_tensor_surface,
        cached_world=cached_world,
        topology_preset=topology_preset,
    )
    old = model.mesh.handle
    old.gpus_per_host = 1
    old.n_hosts = cached_world
    old.comfy_dir = "/comfy"
    old.config_fingerprint = "old"
    old.shutdown = lambda **_kwargs: pytest.fail("unsafe config retired old mesh")
    current = types.SimpleNamespace(
        hosts=tuple(types.SimpleNamespace(gpus=1)
                    for _ in range(configured_world)),
        source="/cluster.toml",
        comfy_dir="",
        worker_args={},
    )
    key = mesh_mod._mesh_cache_key("/cluster.toml", "/comfy", True)
    monkeypatch.setattr(mesh_mod, "install_fault_hook", lambda: None)
    monkeypatch.setattr(mesh_mod, "find_config_path", lambda _path: "/cluster.toml")
    monkeypatch.setattr(mesh_mod, "load_cluster_config", lambda _path: current)
    monkeypatch.setattr(mesh_mod, "_detect_comfy_dir", lambda _path: "/comfy")
    monkeypatch.setattr(mesh_mod, "_coherent_lifecycle_verdict", lambda _handle: "live")
    monkeypatch.setattr(mesh_mod, "_MESHES", {key: old})
    monkeypatch.setattr(mesh_mod, "_MESH_CREATING", {})
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_MODE", "cluster")
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_CLAIMS", set())
    monkeypatch.setattr(
        mesh_mod,
        "spawn_worker_fleet",
        lambda *_args: pytest.fail("unsafe config spawned a replacement mesh"),
    )
    monkeypatch.setattr(
        mesh_mod,
        "config_fingerprint",
        lambda *_args: pytest.fail("preflight ran after cache transition began"),
    )

    with pytest.raises(
        PackedDataParallelError,
        match=rf"world {configured_world} leaves dp{expected_dp}",
    ):
        common._bind_packed_render_model(model, latent, 1.0)

    assert mesh_mod._MESHES == {key: old}
    assert mesh_mod._MESH_CREATING == {}
    assert mesh_mod._TRANSPORT_CLAIMS == set()


def test_packed_dp_pipeline_config_drift_refuses_before_gate_or_sequence(
    monkeypatch,
    direct_nested_tensor_surface,
):
    model, latent, policy = _packed_cluster_request(
        direct_nested_tensor_surface)
    calls = []

    def ensure(_handle, *, mesh_preflight):
        calls.append("resolve")
        mesh_preflight(model.mesh.handle.config, 4)
        pytest.fail("invalid packed world returned a mesh")

    monkeypatch.setattr(common, "ensure_live", ensure)
    monkeypatch.setattr(
        common,
        "auto_gate_required",
        lambda *_args, **_kwargs: pytest.fail("invalid packed world reached Gate"),
    )
    monkeypatch.setattr(
        common,
        "submit_render",
        lambda *_args, **_kwargs: pytest.fail("invalid packed world reached submission"),
    )
    pipe = common.RenderPipeline(depth=2)

    class ExistingPending:
        _state = "pending"
        cancelled = False
        abandoned = False

        def cancel(self):
            self.cancelled = True

        def result(self):
            raise RuntimeError("still running")

        def abandon(self):
            self.abandoned = True
            self._state = "closed"

    existing = ExistingPending()
    pipe._inflight.append(existing)

    with pytest.raises(PackedDataParallelError, match="world 4 leaves dp2"):
        pipe.push(model, {"kind": "ksampler"}, latent, 1.0, 2)

    assert calls == ["resolve"]
    assert pipe._seq == 0 and not pipe._inflight
    assert existing.cancelled and existing.abandoned
    assert policy == {"lora_low_rss": True, "slab_weights": True}


def test_packed_dp_gate_config_drift_refuses_before_session_or_ledger(
    monkeypatch,
    direct_nested_tensor_surface,
):
    model, latent, policy = _packed_cluster_request(
        direct_nested_tensor_surface)

    def ensure(_handle, *, mesh_preflight):
        mesh_preflight(model.mesh.handle.config, 4)
        pytest.fail("invalid packed world returned a mesh")

    monkeypatch.setattr(gate_mod, "ensure_live", ensure)
    monkeypatch.setattr(
        gate_mod,
        "RenderSession",
        lambda: pytest.fail("invalid packed world constructed a session"),
    )
    monkeypatch.setattr(
        gate_mod,
        "GateLedger",
        lambda *_args: pytest.fail("invalid packed world opened the ledger"),
    )
    monkeypatch.setattr(
        gate_mod,
        "_force_stock_quarantine",
        lambda *_args, **_kwargs: pytest.fail("invalid packed world changed policy"),
    )

    with pytest.raises(PackedDataParallelError, match="world 4 leaves dp2"):
        gate_mod.run_identity_ceremony(
            model, {"kind": "ksampler"}, latent, 1.0, 2, "manual")

    assert policy == {"lora_low_rss": True, "slab_weights": True}


def test_packed_dp_direct_submit_drift_precedes_request_session_and_setup(
    monkeypatch,
    direct_nested_tensor_surface,
):
    model, latent, policy = _packed_cluster_request(
        direct_nested_tensor_surface)
    request = {
        "kind": "ksampler",
        "_dgxm_normal_residency_grant": "must-survive-local-refusal",
    }

    def ensure(_handle, *, mesh_preflight):
        mesh_preflight(model.mesh.handle.config, 4)
        pytest.fail("invalid packed world returned a mesh")

    monkeypatch.setattr(common, "ensure_live", ensure)
    monkeypatch.setattr(
        render_submit,
        "claim_render_session",
        lambda *_args: pytest.fail("invalid packed world claimed a session"),
    )
    monkeypatch.setattr(
        render_submit.mesh_setup,
        "ensure_request_setup",
        lambda *_args: pytest.fail("invalid packed world reached setup"),
    )

    with pytest.raises(PackedDataParallelError, match="world 4 leaves dp2"):
        common.submit_render(
            model,
            request,
            latent,
            1.0,
            2,
            handoff=common.PendingRenderHandoff(),
        )

    assert request["_dgxm_normal_residency_grant"] == "must-survive-local-refusal"
    assert policy == {"lora_low_rss": True, "slab_weights": True}


def test_packed_dp_change_during_auto_gate_lookup_is_not_swallowed(
    monkeypatch,
    direct_nested_tensor_surface,
):
    model, latent, policy = _packed_cluster_request(
        direct_nested_tensor_surface)
    worlds = iter((2, 2, 4))

    def ensure(handle, *, mesh_preflight):
        world = next(worlds)
        mesh_preflight(handle.config, world)
        return handle

    monkeypatch.setattr(common, "ensure_live", ensure)
    monkeypatch.setattr(
        common,
        "submit_render",
        lambda *_args, **_kwargs: pytest.fail("invalid packed world reached submission"),
    )
    monkeypatch.setattr(
        common,
        "_quarantine_unproven_paths",
        lambda *_args: pytest.fail("typed packed refusal was quarantined"),
    )

    with pytest.raises(PackedDataParallelError, match="world 4 leaves dp2"):
        common.run_render(
            model, {"kind": "ksampler", "steps": 2}, latent, 1.0, 2)

    with pytest.raises(StopIteration):
        next(worlds)
    assert policy == {"lora_low_rss": True, "slab_weights": True}


def test_packed_dp_change_during_auto_gate_required_is_not_swallowed(
    monkeypatch,
    direct_nested_tensor_surface,
):
    model, latent, policy = _packed_cluster_request(
        direct_nested_tensor_surface)
    worlds = iter((2, 4))

    def ensure(handle, *, mesh_preflight):
        world = next(worlds)
        mesh_preflight(handle.config, world)
        return handle

    monkeypatch.setattr(common, "ensure_live", ensure)

    with pytest.raises(PackedDataParallelError, match="world 4 leaves dp2"):
        common.auto_gate_required(model, "ksampler", latent, 1.0)

    with pytest.raises(StopIteration):
        next(worlds)
    assert policy == {"lora_low_rss": True, "slab_weights": True}


def _drift_to_cfg(monkeypatch, model):
    """Resolution gains cfg between the outer and the inner bind.

    The outer bind reads a sequence topology and admits the render; the inner
    bind inside the auto-gate context reads cfg. Both auto-gate handlers wrap
    that inner bind in a broad `except`, so this is where the CFG refusal has
    to survive.
    """
    topologies = iter((Topology(world=2, ulysses=2, dp=1),
                       Topology(world=2, cfg=2, dp=1)))
    monkeypatch.setattr(
        common, "_resolve_topology_for_latent",
        lambda *_args, **_kwargs: (next(topologies), False, "drift"))

    def ensure(handle, *, mesh_preflight):
        mesh_preflight(handle.config, 2)
        return handle

    monkeypatch.setattr(common, "ensure_live", ensure)
    monkeypatch.setattr(
        common,
        "_quarantine_unproven_paths",
        lambda *_args: pytest.fail("typed packed refusal was quarantined"),
    )
    return model


def test_packed_cfg_change_during_auto_gate_required_is_not_swallowed(
    monkeypatch,
    direct_nested_tensor_surface,
):
    model, latent, policy = _packed_cluster_request(
        direct_nested_tensor_surface)
    _drift_to_cfg(monkeypatch, model)

    with pytest.raises(PackedCfgParallelError, match="cannot use cfg-parallel"):
        common.auto_gate_required(model, "ksampler", latent, 7.5)

    assert policy == {"lora_low_rss": True, "slab_weights": True}


def test_packed_cfg_change_during_maybe_auto_gate_is_not_swallowed(
    monkeypatch,
    direct_nested_tensor_surface,
):
    model, latent, policy = _packed_cluster_request(
        direct_nested_tensor_surface)
    _drift_to_cfg(monkeypatch, model)

    with pytest.raises(PackedCfgParallelError, match="cannot use cfg-parallel"):
        common._maybe_auto_gate(
            model, {"kind": "ksampler", "steps": 2}, latent, 7.5, 2)

    assert policy == {"lora_low_rss": True, "slab_weights": True}


@pytest.mark.parametrize(
    "video_shape",
    [(1, 128, 16, 11, 20), (1, 128, 16, 22, 40)],
    ids=["low-stage", "final-stage"],
)
def test_nested_custom_denoised_preserves_ltx_av_shapes_and_output_one(
    monkeypatch, direct_nested_tensor_surface, video_shape,
):
    audio_shape = (1, 8, 126, 16)
    samples = NestedTensor((
        torch.zeros(video_shape, dtype=torch.float32),
        torch.zeros(audio_shape, dtype=torch.float32),
    ))
    denoised = NestedTensor((
        torch.ones(video_shape, dtype=torch.float32),
        torch.ones(audio_shape, dtype=torch.float32),
    ))
    rendered = _finish([{
        "rank": 0,
        "dp_rank": 0,
        "latent": samples,
        "latent_extra": {"denoised": denoised},
    }], 1, monkeypatch)

    assert rendered["samples"] is samples
    assert rendered["_dgxm_denoised"] is denoised
    assert [tuple(part.shape) for part in rendered["samples"].unbind()] == [
        video_shape, audio_shape,
    ]

    monkeypatch.setattr(samplers, "run_render", lambda *_args, **_kwargs: dict(rendered))
    primary, denoised_output = samplers.DGXMonarchSamplerCustom().sample(
        types.SimpleNamespace(seed=7),
        {"model": object(), "spec": {"kind": "basic"}},
        object(),
        torch.tensor([1.0, 0.0]),
        {"samples": samples},
    )
    assert denoised_output["samples"] is denoised
    assert denoised_output["samples"] is not primary["samples"]
    assert all(
        primary_part.untyped_storage().data_ptr()
        != denoised_part.untyped_storage().data_ptr()
        for primary_part, denoised_part in zip(
            primary["samples"].unbind(), denoised_output["samples"].unbind(), strict=True
        )
    )


@pytest.mark.parametrize(
    ("denoised", "match"),
    [
        (
            NestedTensor((torch.ones(1, 2, 3), torch.ones(1, 3))),
            "modality does not match",
        ),
        (
            NestedTensor((torch.ones(1, 2, 2),)),
            "modality count",
        ),
        (
            NestedTensor((
                torch.ones(1, 2, 2, dtype=torch.float64),
                torch.ones(1, 3, dtype=torch.float64),
            )),
            "modality does not match",
        ),
        (
            NestedTensor((torch.ones(1, 2, 2), torch.ones(2, 3))),
            "mismatched batches",
        ),
    ],
)
def test_nested_custom_denoised_rejects_structure_drift(
    monkeypatch, direct_nested_tensor_surface, denoised, match,
):
    samples = NestedTensor((torch.zeros(1, 2, 2), torch.zeros(1, 3)))
    with pytest.raises(RuntimeError, match=match):
        _finish([{
            "rank": 0,
            "dp_rank": 0,
            "latent": samples,
            "latent_extra": {"denoised": denoised},
        }], 1, monkeypatch)


def test_nested_custom_denoised_rejects_alias_and_nonfinite(
    monkeypatch, direct_nested_tensor_surface,
):
    video = torch.zeros(1, 2, 2)
    audio = torch.zeros(1, 3)
    samples = NestedTensor((video, audio))
    with pytest.raises(RuntimeError, match="aliases"):
        _finish([{
            "rank": 0,
            "dp_rank": 0,
            "latent": samples,
            "latent_extra": {"denoised": NestedTensor((video, audio))},
        }], 1, monkeypatch)

    bad_audio = torch.ones(1, 3)
    bad_audio[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        _finish([{
            "rank": 0,
            "dp_rank": 0,
            "latent": samples,
            "latent_extra": {
                "denoised": NestedTensor((torch.ones(1, 2, 2), bad_audio))
            },
        }], 1, monkeypatch)


def test_flat_empty_views_detect_shared_storage_without_false_positive(monkeypatch):
    backing = torch.zeros(1, 2)
    shared_sample = backing[:, :0]
    shared_denoised = backing[:, 1:1]
    with pytest.raises(RuntimeError, match="aliases"):
        _finish([{
            "rank": 0,
            "dp_rank": 0,
            "latent": shared_sample,
            "latent_extra": {"denoised": shared_denoised},
        }], 1, monkeypatch)

    independent_sample = torch.empty(1, 0)
    independent_denoised = torch.empty(1, 0)
    rendered = _finish([{
        "rank": 0,
        "dp_rank": 0,
        "latent": independent_sample,
        "latent_extra": {"denoised": independent_denoised},
    }], 1, monkeypatch)
    assert rendered["samples"] is independent_sample
    assert rendered["_dgxm_denoised"] is independent_denoised


@pytest.mark.parametrize("packed", [False, True], ids=["flat", "packed"])
def test_shifted_distinct_storages_are_detected_as_aliases(
    monkeypatch, direct_nested_tensor_surface, packed,
):
    np = pytest.importorskip("numpy")
    backing = np.arange(16, dtype=np.float32)
    sample_leaf = torch.from_numpy(backing[:-1]).reshape(1, 15)
    denoised_leaf = torch.from_numpy(backing[1:]).reshape(1, 15)
    assert sample_leaf.untyped_storage() is not denoised_leaf.untyped_storage()
    assert sample_leaf.untyped_storage().data_ptr() != (
        denoised_leaf.untyped_storage().data_ptr()
    )
    samples = (
        NestedTensor((sample_leaf, torch.zeros(1, 3)))
        if packed
        else sample_leaf
    )
    denoised = (
        NestedTensor((denoised_leaf, torch.ones(1, 3)))
        if packed
        else denoised_leaf
    )

    with pytest.raises(RuntimeError, match="aliases"):
        _finish([{
            "rank": 0,
            "dp_rank": 0,
            "latent": samples,
            "latent_extra": {"denoised": denoised},
        }], 1, monkeypatch)


def test_packed_empty_views_detect_shared_storage_without_false_positive(
    monkeypatch, direct_nested_tensor_surface,
):
    zero_backing = torch.empty(0)
    zero_sample = zero_backing.view(1, 0)
    zero_denoised = zero_backing.view(1, 0)
    assert zero_sample.untyped_storage() is zero_denoised.untyped_storage()
    with pytest.raises(RuntimeError, match="aliases"):
        _finish([{
            "rank": 0,
            "dp_rank": 0,
            "latent": NestedTensor((zero_sample, torch.zeros(1, 3))),
            "latent_extra": {
                "denoised": NestedTensor((zero_denoised, torch.ones(1, 3)))
            },
        }], 1, monkeypatch)

    backing = torch.zeros(1, 2)
    shared_samples = NestedTensor((backing[:, :0], torch.zeros(1, 3)))
    shared_denoised = NestedTensor((backing[:, 1:1], torch.ones(1, 3)))
    with pytest.raises(RuntimeError, match="aliases"):
        _finish([{
            "rank": 0,
            "dp_rank": 0,
            "latent": shared_samples,
            "latent_extra": {"denoised": shared_denoised},
        }], 1, monkeypatch)

    independent_samples = NestedTensor((torch.empty(1, 0), torch.zeros(1, 3)))
    independent_denoised = NestedTensor((torch.empty(1, 0), torch.ones(1, 3)))
    rendered = _finish([{
        "rank": 0,
        "dp_rank": 0,
        "latent": independent_samples,
        "latent_extra": {"denoised": independent_denoised},
    }], 1, monkeypatch)
    assert rendered["samples"] is independent_samples
    assert rendered["_dgxm_denoised"] is independent_denoised


def test_nested_driver_refuses_dp_and_non_tensor_modalities(
    monkeypatch, direct_nested_tensor_surface,
):
    samples = NestedTensor((torch.zeros(1, 2), torch.zeros(1, 3)))
    with pytest.raises(RuntimeError, match="cannot be reconstructed across DP"):
        _finish([
            {"rank": 0, "dp_rank": 0, "latent": samples},
            {"rank": 1, "dp_rank": 1, "latent": NestedTensor((
                torch.ones(1, 2), torch.ones(1, 3),
            ))},
        ], 2, monkeypatch)

    with pytest.raises(RuntimeError, match="modalities must all be tensors"):
        _finish([{
            "rank": 0,
            "dp_rank": 0,
            "latent": NestedTensor((torch.zeros(1, 2), "audio")),
        }], 1, monkeypatch)


def test_driver_rejects_nested_subclass_and_same_named_spoof(
    monkeypatch, direct_nested_tensor_surface,
):
    class NestedSubclass(NestedTensor):
        pass

    spoof_type = type(
        "NestedTensor",
        (),
        {
            "__init__": lambda self, tensors: setattr(self, "tensors", list(tensors)),
            "unbind": lambda self: self.tensors,
        },
    )
    for value in (
        NestedSubclass((torch.zeros(1, 2), torch.zeros(1, 3))),
        spoof_type((torch.zeros(1, 2), torch.zeros(1, 3))),
    ):
        with pytest.raises(RuntimeError):
            _finish([{"rank": 0, "dp_rank": 0, "latent": value}], 1, monkeypatch)


def test_custom_node_fails_closed_when_packed_denoised_transport_is_missing(
    monkeypatch, direct_nested_tensor_surface,
):
    samples = NestedTensor((torch.zeros(1, 2), torch.zeros(1, 3)))
    monkeypatch.setattr(
        samplers, "run_render", lambda *_args, **_kwargs: {"samples": samples}
    )
    with pytest.raises(RuntimeError, match="missing reconstructed denoised x0"):
        samplers.DGXMonarchSamplerCustom().sample(
            types.SimpleNamespace(seed=7),
            {"model": object(), "spec": {"kind": "basic"}},
            object(),
            torch.tensor([1.0, 0.0]),
            {"samples": samples},
        )


def test_custom_node_zero_step_packed_fallback_clones_structure(
    monkeypatch, direct_nested_tensor_surface,
):
    samples = NestedTensor((torch.zeros(1, 2), torch.ones(1, 3)))
    monkeypatch.setattr(
        samplers, "run_render", lambda *_args, **_kwargs: {"samples": samples}
    )
    primary, denoised = samplers.DGXMonarchSamplerCustom().sample(
        types.SimpleNamespace(seed=7),
        {"model": object(), "spec": {"kind": "basic"}},
        object(),
        torch.tensor([0.0]),
        {"samples": samples},
    )
    for original, clone in zip(
        primary["samples"].unbind(), denoised["samples"].unbind(), strict=True
    ):
        assert torch.equal(clone, original)
        assert clone.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()


def test_custom_node_validates_schedule_and_clones_plain_fallback(monkeypatch):
    samples = torch.zeros(1, 2)
    monkeypatch.setattr(
        samplers, "run_render", lambda *_args, **_kwargs: {"samples": samples}
    )
    with pytest.raises(ValueError, match="finite, non-empty, 1-D floating Tensor"):
        samplers.DGXMonarchSamplerCustom().sample(
            types.SimpleNamespace(seed=7),
            {"model": object(), "spec": {"kind": "basic"}},
            object(),
            torch.tensor([]),
            {"samples": samples},
        )

    primary, denoised = samplers.DGXMonarchSamplerCustom().sample(
        types.SimpleNamespace(seed=7),
        {"model": object(), "spec": {"kind": "basic"}},
        object(),
        torch.tensor([1.0, 0.0]),
        {"samples": samples},
    )
    assert torch.equal(denoised["samples"], primary["samples"])
    assert denoised["samples"].untyped_storage().data_ptr() != (
        primary["samples"].untyped_storage().data_ptr()
    )


class _FakeZImageCfg:
    def __init__(self):
        self.recorded = None

    def _forward(self, x, timesteps, context, num_tokens, attention_mask=None,
                 transformer_options=None, **kwargs):
        self.recorded = {
            "context": context,
            "num_tokens": num_tokens,
            "attention_mask": attention_mask,
        }
        return context


def test_zimage_cfg_trim_rederives_num_tokens():
    model = _FakeZImageCfg()
    ZImageAdapter().inject_cfg_pad_forward(model)
    context = torch.ones(1, 10, 4)
    context[:, 6:] = 0
    model._forward(torch.zeros(1, 1, 2, 2), torch.zeros(1), context, 10)
    assert model.recorded["context"].shape[1] == 6
    assert model.recorded["num_tokens"] == 6


def _run_cfg_patch_probe(monkeypatch, hook, patch):
    import dgx_monarch.adapters.cfg_parallel as cfg_parallel

    monkeypatch.setattr(cfg_parallel, "cfg_world", lambda: 2)
    monkeypatch.setattr(cfg_parallel, "cfg_rank", lambda: 0)

    class Group:
        def all_gather(self, value, dim=0):
            assert dim == 0
            return value

    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.get_cfg_group = lambda: Group()
    monkeypatch.setitem(sys.modules, "xfuser", types.ModuleType("xfuser"))
    monkeypatch.setitem(sys.modules, "xfuser.core", types.ModuleType("xfuser.core"))
    monkeypatch.setitem(sys.modules, "xfuser.core.distributed", distributed)

    class Executor:
        def __init__(self):
            self.original = self.forward
            self.recorded = None

        def forward(self, x, transformer_options=None):
            raise AssertionError("signature only")

        def __call__(self, x, transformer_options=None):
            self.recorded = transformer_options
            return x

    executor = Executor()
    patches = patch if isinstance(patch, list) else [patch]
    options = {
        "cond_or_uncond": [0, 1],
        "patches": {hook: patches},
    }
    wrapper = make_cfg_parallel_wrapper(types.SimpleNamespace(family="flux"))
    result = wrapper(executor, torch.zeros(2, 1), options)
    return executor, result


def test_cfg_parallel_rejects_uncontracted_cross_condition_output_patch(monkeypatch):
    def nag_patch(value):
        return value

    with pytest.raises(UnsupportedModelError, match="NAG"):
        _run_cfg_patch_probe(monkeypatch, "attn1_output_patch", nag_patch)


def test_cfg_patch_contract_rejects_marked_callable_patches_container():
    def malformed_patches_container(value):
        return value

    setattr(
        malformed_patches_container,
        attention_patches.ATTENTION_PATCH_CAPABILITIES_ATTR,
        frozenset({attention_patches.CONDITION_SHARD_LOCAL_CAPABILITY}),
    )
    with pytest.raises(UnsupportedModelError, match="attn1_output_patch"):
        attention_patches.assert_cfg_attention_output_patches_safe(
            {"patches": malformed_patches_container}, "flux"
        )


def test_cfg_parallel_requires_every_output_patch_to_be_condition_local(monkeypatch):
    def local_patch(value):
        return value

    setattr(
        local_patch,
        attention_patches.ATTENTION_PATCH_CAPABILITIES_ATTR,
        frozenset({attention_patches.CONDITION_SHARD_LOCAL_CAPABILITY}),
    )

    def cross_condition_patch(value):
        return value

    with pytest.raises(UnsupportedModelError, match="condition_shard_local"):
        _run_cfg_patch_probe(
            monkeypatch,
            "attn1_output_patch",
            [local_patch, cross_condition_patch],
        )


@pytest.mark.parametrize(
    ("hook", "capability"),
    [
        ("attn1_patch", None),
        (
            "attn1_output_patch",
            attention_patches.CONDITION_SHARD_LOCAL_CAPABILITY,
        ),
    ],
)
def test_cfg_parallel_keeps_branch_local_attention_patches(
    monkeypatch, hook, capability
):
    def patch(value):
        return value

    if capability is not None:
        setattr(
            patch,
            attention_patches.ATTENTION_PATCH_CAPABILITIES_ATTR,
            frozenset({capability}),
        )
    executor, result = _run_cfg_patch_probe(monkeypatch, hook, patch)
    assert result.shape == (1, 1)
    assert executor.recorded["patches"][hook][0] is patch


@pytest.mark.parametrize(
    ("rank", "expected_rows", "expected_condition"),
    [
        (0, [[0.0], [1.0], [2.0]], (0, "conditional-uuid")),
        (1, [[3.0], [4.0], [5.0]], (1, "unconditional-uuid")),
    ],
)
def test_cfg_parallel_localizes_transformer_metadata(
    monkeypatch, rank, expected_rows, expected_condition
):
    import dgx_monarch.adapters.cfg_parallel as cfg_parallel

    monkeypatch.setattr(cfg_parallel, "cfg_world", lambda: 2)
    monkeypatch.setattr(cfg_parallel, "cfg_rank", lambda: rank)

    class Group:
        def all_gather(self, value, dim=0):
            assert dim == 0
            return value

    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.get_cfg_group = lambda: Group()
    core = types.ModuleType("xfuser.core")
    xfuser = types.ModuleType("xfuser")
    monkeypatch.setitem(sys.modules, "xfuser", xfuser)
    monkeypatch.setitem(sys.modules, "xfuser.core", core)
    monkeypatch.setitem(sys.modules, "xfuser.core.distributed", distributed)

    def patch_hook(value):
        return value

    original_options = {
        "cond_or_uncond": [0, 1],
        "uuids": ("conditional-uuid", "unconditional-uuid"),
        "nested": {"per_sample": torch.arange(6).view(6, 1)},
        "patches": {"attn": [patch_hook]},
        # Leading size happens to equal the model batch but this is a static
        # step schedule, not per-sample data.
        "sigmas": torch.arange(6),
        "note": "static",
    }

    class Executor:
        def __init__(self):
            self.recorded = None
            self.original = self.forward

        def forward(self, x, transformer_options=None):
            raise AssertionError("signature only")

        def __call__(self, x, transformer_options=None):
            self.recorded = (x, transformer_options)
            return x

    executor = Executor()
    wrapper = make_cfg_parallel_wrapper(None)
    wrapper(executor, torch.arange(6).view(6, 1).float(), original_options)

    local_x, local_options = executor.recorded
    expected_cond, expected_uuid = expected_condition
    assert torch.equal(local_x, torch.tensor(expected_rows))
    assert local_options["cond_or_uncond"] == [expected_cond]
    assert local_options["uuids"] == (expected_uuid,)
    assert (
        local_options["cond_or_uncond"][0], local_options["uuids"][0]
    ) == expected_condition
    assert torch.equal(
        local_options["nested"]["per_sample"],
        torch.tensor(expected_rows, dtype=torch.int64),
    )
    assert local_options["patches"]["attn"][0] is patch_hook
    assert local_options["sigmas"] is original_options["sigmas"]
    assert local_options["note"] == "static"
    assert original_options["cond_or_uncond"] == [0, 1]
    assert original_options["uuids"] == ("conditional-uuid", "unconditional-uuid")
    assert original_options["nested"]["per_sample"].shape[0] == 6


def test_basic_scheduler_heals_and_sets_up_fresh_handle(monkeypatch):
    calls = []

    class FreshHandle:
        world = 2

        def ensure_setup(self, topo, attention, sync_ulysses, worker_args):
            calls.append(("setup", topo.describe(), attention, sync_ulysses, worker_args))

        def call_all(self, *args, **kwargs):
            calls.append(("call", args, kwargs))
            return [torch.tensor([1.0, 0.0])]

    old_handle = object()
    fresh = FreshHandle()
    monkeypatch.setattr(samplers, "ensure_live", lambda handle: fresh)
    mesh = types.SimpleNamespace(
        handle=old_handle,
        topology_preset="ring2",
        attention="TORCH_FLASH",
        sync_ulysses=True,
        worker_args={"mmap_fallback": False},
    )
    model = types.SimpleNamespace(mesh=mesh, request_dict=lambda: {"unet_name": "model.sft"})

    sigmas, = samplers.DGXMonarchBasicScheduler().get_sigmas(model, "normal", 2, 1.0)
    assert torch.equal(sigmas, torch.tensor([1.0, 0.0]))
    assert calls[0][:3] == ("setup", "ring2", "TORCH_FLASH")
    assert calls[1][0] == "call"
    assert calls[1][1][0] == "compute_sigmas"


def test_packed_cfg_parallel_refuses_before_dispatch(direct_nested_tensor_surface):
    from dgx_monarch.nodes.render_validation import (
        PackedCfgParallelError,
        validate_render_topology,
    )

    model = types.SimpleNamespace(
        mesh=types.SimpleNamespace(topology_preset="cfg2")
    )
    latent = NestedTensor((torch.zeros(1, 2), torch.zeros(1, 3)))
    topo = Topology(cfg=2, world=2)
    with pytest.raises(PackedCfgParallelError, match="cannot use cfg-parallel"):
        validate_render_topology(model, topo, latent)


def test_packed_sequence_topology_passes_the_cfg_arm(direct_nested_tensor_surface):
    from dgx_monarch.nodes.render_validation import validate_render_topology

    model = types.SimpleNamespace(
        mesh=types.SimpleNamespace(topology_preset="uly2")
    )
    latent = NestedTensor((torch.zeros(1, 2), torch.zeros(1, 3)))
    validate_render_topology(model, Topology(ulysses=2, world=2), latent)
