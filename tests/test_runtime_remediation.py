"""Runtime hardening regressions: ledger integrity, grants, artifact identity, telemetry, setup, cluster attach."""

from __future__ import annotations

import hashlib
import os
import sys
import types

import pytest
import torch


def test_gate_ledger_recovers_matching_identity_after_other_artifact(tmp_path):
    from dgx_monarch.gate_ledger import GateLedger

    ledger = GateLedger(str(tmp_path))
    ledger.record("model", "artifact-a", "commit", "FAIL")
    ledger.record("model", "artifact-b", "commit", "PASS")

    assert ledger.lookup("model", "artifact-a", "commit") == "fail"
    assert ledger.lookup("model", "artifact-b", "commit") == "pass"


def test_gate_ledger_ignores_only_torn_jsonl_record(tmp_path):
    from dgx_monarch.gate_ledger import GateLedger

    ledger = GateLedger(str(tmp_path))
    ledger.record("model", "artifact", "commit", "FAIL")
    with open(ledger.path, "a") as stream:
        stream.write('{"key":"model"')

    assert ledger.lookup("model", "artifact", "commit") == "fail"
    assert len(ledger.entries()) == 1

    ledger.record("next-model", "next-artifact", "commit", "PASS")
    assert ledger.lookup("next-model", "next-artifact", "commit") == "pass"
    assert len(ledger.entries()) == 2


def test_gate_ledger_preserves_intact_rows_around_invalid_utf8(tmp_path):
    from dgx_monarch.gate_ledger import GateLedger

    ledger = GateLedger(str(tmp_path))
    ledger.record("before", "artifact-a", "commit", "FAIL")
    with open(ledger.path, "ab") as stream:
        stream.write(b'{"key":"damaged","verdict":"PASS","bad":"\xff"}\n')
    ledger.record("after", "artifact-b", "commit", "PASS")

    assert ledger.lookup("before", "artifact-a", "commit") == "fail"
    # A valid exact row written after the damage heals it: damage blocks
    # positive authority only when it is newer than the row.
    assert ledger.lookup("after", "artifact-b", "commit") == "pass"
    assert [entry["key"] for entry in ledger.entries()] == ["before", "after"]


def _normal_grant_request():
    from dgx_monarch import __version__
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION, gate_verdict_token
    from dgx_monarch.mesh_safety import request_combo_key

    model = {
        "unet_name": "model.safetensors",
        "options": {},
        "loras": [{"name": "adapter.safetensors", "strength": 1.0}],
    }
    identity = {
        "digest": "artifact-digest",
        "comfy": "comfy-commit",
        "artifacts": [{"kind": "diffusion_models", "name": "model.safetensors",
                       "signature": "model-signature"}],
    }
    topology = {"ulysses": 1, "ring": 1, "cfg": 1, "dp": 2, "fsdp": False}
    worker_args = {"slab_weights": True, "lora_low_rss": True}
    context = {
        "worker_args": dict(worker_args),
        "mesh_mode": "cluster",
        "config_source": "/cluster.toml",
        "config_fingerprint": "config-digest",
        "world": 2,
        "hosts": 2,
        "gpus_per_host": 1,
        "topology_preset": "auto",
        "attention": "TORCH_FLASH",
        "sync_ulysses": False,
        "resolved_topology": dict(topology),
        "resolved_attention": "TORCH_FLASH",
    }
    combo = request_combo_key(model)
    token = gate_verdict_token(
        combo, identity["digest"], identity["comfy"], context)
    setup_key = (1, 1, 1, 2, False, 2, "TORCH_FLASH", False)
    worker_args_key = (("lora_low_rss", "True"), ("slab_weights", "True"))
    grant = {
        "capability": "normal_render_residency",
        "gate_protocol": GATE_PROTOCOL_VERSION,
        "dgx_monarch": __version__,
        "comfy": identity["comfy"],
        "artifact_digest": identity["digest"],
        "gate_token": list(token),
        "combo_key": combo,
        "model_request": model,
        "uncond_model_request": None,
        "artifact_sets": [identity],
        "capability_context": context,
        "setup_generation": 7,
        "setup_key": setup_key,
        "worker_args_key": worker_args_key,
        "worker_topology": topology,
    }
    request = {
        "model": model,
        "sage_kernel": "TORCH_FLASH",
        "sync_ulysses": False,
        "_dgxm_normal_residency_mode": "required",
        "_dgxm_normal_residency_grant": grant,
    }
    return request, identity, worker_args, context, setup_key, worker_args_key, topology


def test_normal_residency_grant_binds_every_reconstructible_boundary():
    from dgx_monarch.mesh_safety import assert_normal_render_residency_grant

    request, identity, worker_args, context, setup_key, wa_key, topology = (
        _normal_grant_request())
    assert assert_normal_render_residency_grant(
        request,
        [identity],
        effective_worker_args=worker_args,
        expected_context=context,
        rank_world=2,
        setup_generation=7,
        setup_key=setup_key,
        worker_args_key=wa_key,
        worker_topology=topology,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda request: request["_dgxm_normal_residency_grant"].update(
            gate_protocol=-1),
        lambda request: request["_dgxm_normal_residency_grant"].update(
            setup_generation=8),
        lambda request: request.update(sage_kernel="SAGE_AUTO"),
        lambda request: request.pop("sync_ulysses"),
        lambda request: request["_dgxm_normal_residency_grant"][
            "worker_topology"].update(dp=1),
        lambda request: request["model"].update(unet_name="replacement.safetensors"),
    ],
)
def test_normal_residency_grant_rejects_tampering(mutation):
    import copy

    from dgx_monarch.mesh_safety import assert_normal_render_residency_grant

    request, identity, worker_args, _context, setup_key, wa_key, topology = (
        _normal_grant_request())
    request = copy.deepcopy(request)
    mutation(request)
    with pytest.raises(RuntimeError):
        assert_normal_render_residency_grant(
            request,
            [identity],
            effective_worker_args=worker_args,
            rank_world=2,
            setup_generation=7,
            setup_key=setup_key,
            worker_args_key=wa_key,
            worker_topology=topology,
        )


def test_risky_normal_render_requires_driver_mode_and_grant():
    from dgx_monarch.mesh_safety import assert_normal_render_residency_mode

    request, _identity, worker_args, *_rest = _normal_grant_request()
    request.pop("_dgxm_normal_residency_grant")
    with pytest.raises(RuntimeError, match="lost its required"):
        assert_normal_render_residency_mode(request, worker_args)
    request.pop("_dgxm_normal_residency_mode")
    with pytest.raises(RuntimeError, match="no driver-stamped"):
        assert_normal_render_residency_mode(request, worker_args)
    request["_dgxm_normal_residency_mode"] = "operator_off"
    assert assert_normal_render_residency_mode(request, worker_args) == "operator_off"


def test_rdma_native_preflight_failure_is_cached_and_safe(monkeypatch):
    import dgx_monarch.transfer as transfer

    native_marker = object()
    calls = []
    monkeypatch.setattr(transfer, "RDMABuffer", native_marker)
    monkeypatch.setattr(transfer, "_NATIVE_RDMA_BUFFER", native_marker)
    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "_RDMA_PREFLIGHT_RESULT", None)

    def failed_probe(*args, **kwargs):
        calls.append((args, kwargs))
        return types.SimpleNamespace(returncode=-6)  # child died by SIGABRT

    monkeypatch.setattr(transfer.subprocess, "run", failed_probe)
    assert transfer._rdma_usable() is False
    assert transfer._rdma_usable() is False
    assert len(calls) == 1


def test_salted_ports_remain_non_privileged_and_fleet_offsets_legal():
    from dgx_monarch.mesh_safety import bounded_master_port

    base = bounded_master_port(65535, pid=65535, generation=15, fleet_world=8)
    assert 1024 <= base <= 65535 - 39
    assert all(1024 <= base + 32 + rank <= 65535 for rank in range(8))


def test_salted_port_has_a_stable_integer_golden_value():
    from dgx_monarch.mesh_safety import bounded_master_port

    port = bounded_master_port(1024, pid=1, generation=1)
    assert type(port) is int
    assert port == 1041


def test_request_without_artifact_binding_is_explicitly_unbound():
    from dgx_monarch.mesh_safety import assert_request_artifact_binding

    assert assert_request_artifact_binding({"model": {"unet_name": "m"}}, []) is False


def test_host_specific_comfy_dir_wins_over_cluster_default():
    from dgx_monarch.config import ClusterConfig, HostConfig
    from dgx_monarch.mesh_safety import worker_comfy_dir

    config = ClusterConfig(
        hosts=(
            HostConfig("a", "tcp://10.0.0.1:1", comfy_dir="/srv/a/comfy"),
            HostConfig("b", "tcp://10.0.0.2:1"),
        ),
        comfy_dir="/srv/default/comfy",
    )
    assert worker_comfy_dir(config, 0, "/driver/comfy") == "/srv/a/comfy"
    assert worker_comfy_dir(config, 1, "/driver/comfy") == "/srv/default/comfy"


def test_artifact_parity_compares_workers_to_driver():
    from dgx_monarch.mesh_safety import assert_artifact_parity

    driver = [{"digest": "a", "comfy": "c", "artifacts": []}]
    workers = [
        {"host": "one", "rank": 0, "artifact_sets": driver},
        {"host": "two", "rank": 1, "artifact_sets": driver},
    ]
    assert_artifact_parity(workers, driver)
    workers[1] = {
        "host": "two",
        "rank": 1,
        "artifact_sets": [{"digest": "b", "comfy": "c", "artifacts": []}],
    }
    with pytest.raises(RuntimeError, match="artifact parity"):
        assert_artifact_parity(workers, driver)


def test_artifact_parity_fails_closed_without_worker_identity():
    from dgx_monarch.mesh_safety import assert_artifact_parity

    with pytest.raises(RuntimeError, match="no reference identity"):
        assert_artifact_parity([{"host": "one", "rank": 0}])


@pytest.mark.parametrize("signature", ["unreadable", "unstable"])
def test_request_artifact_identity_rejects_unproven_files(monkeypatch, signature):
    from dgx_monarch.actor import store_identity

    monkeypatch.setattr(
        "dgx_monarch.gate_ledger.artifact_signature", lambda _path: signature)
    with pytest.raises(RuntimeError, match="stable model artifact identity"):
        store_identity.build_request_artifact_identity(
            "model.safetensors", [], lambda kind, name: f"/{kind}/{name}")


def test_artifact_preflight_marker_is_request_local(monkeypatch):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.actor import model_store

    expected = {"digest": "a", "comfy": "c", "artifacts": []}
    monkeypatch.setattr(model_store, "request_artifact_identity", lambda *_args: expected)
    handle = mesh_mod.MeshHandle.__new__(mesh_mod.MeshHandle)
    handle.setup_generation = 3
    handle.active_worker_args = {"slab_weights": False, "lora_low_rss": False}
    calls = []

    def call_all(_endpoint, _specs, timeout_s):
        calls.append(timeout_s)
        return [
            {"host": "one", "rank": 0, "artifact_sets": [expected]},
            {"host": "two", "rank": 1, "artifact_sets": [expected]},
        ]

    handle.call_all = call_all
    first = {"model": {"unet_name": "m", "loras": []}}
    handle._verify_request_artifacts(first)
    handle._verify_request_artifacts(first)  # same dispatch object: marker avoids duplicate
    second = {"model": {"unet_name": "m", "loras": []}}
    handle._verify_request_artifacts(second)  # new render: fresh all-rank proof

    assert calls == [120, 120]
    assert first["_dgxm_artifact_sets"] == [expected]
    assert second["_dgxm_artifact_sets"] == [expected]


def test_ceremony_binding_rejects_replacement_between_render_legs(monkeypatch):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.actor import model_store
    from dgx_monarch.mesh_safety import request_combo_key

    first = {"digest": "first", "comfy": "c", "artifacts": []}
    replacement = {"digest": "replacement", "comfy": "c", "artifacts": []}
    current = {"identity": first}
    model = {"unet_name": "m", "options": {}, "loras": []}
    binding = {"combo_key": request_combo_key(model),
               "model_request": model, "artifact_sets": [first]}
    handle = mesh_mod.MeshHandle.__new__(mesh_mod.MeshHandle)
    handle.setup_generation = 1
    handle.active_worker_args = {"slab_weights": False, "lora_low_rss": False}
    calls = []

    def call_all(_endpoint, _specs, timeout_s):
        calls.append(timeout_s)
        return [{"host": "worker", "rank": 0,
                 "artifact_sets": [current["identity"]]}]

    handle.call_all = call_all
    monkeypatch.setattr(
        model_store, "request_artifact_identity", lambda *_args: current["identity"])
    first_leg = {"model": dict(model), "_dgxm_artifact_binding": binding}
    handle._verify_request_artifacts(first_leg)

    current["identity"] = replacement
    second_leg = {"model": dict(model), "_dgxm_artifact_binding": binding}
    with pytest.raises(RuntimeError, match="changed after the ceremony snapshot"):
        handle._verify_request_artifacts(second_leg)
    assert calls == [120]  # replacement refused before another worker dispatch


def test_ceremony_binding_deeply_snapshots_every_model_request_field(monkeypatch):
    import copy

    from dgx_monarch.gate_ledger import artifact_set_signature
    from dgx_monarch.mesh_safety import (
        ArtifactBindingError,
        assert_request_artifact_binding,
    )
    from dgx_monarch.nodes.gate_identity import capture

    primary_source = {
        "unet_name": "primary.sft",
        "options": {"dtype": {"name": "bf16"}},
        "loras": [
            {"name": "a.sft", "strength": 0.25},
            {"name": "b.sft", "strength": 0.75},
        ],
        "model_sampling": {"kind": "sd3", "shift": 3.0},
    }
    uncond_source = {
        "unet_name": "negative.sft",
        "options": {"dtype": {"name": "fp16"}},
        "loras": [{"name": "neg.sft", "strength": 0.4}],
        "model_sampling": {"kind": "sd3", "shift": 5.0},
    }

    def identity(name, loras):
        signatures = [f"sig:{name}", *(f"sig:{entry['name']}" for entry in loras or [])]
        return {
            "digest": artifact_set_signature(signatures).current,
            "comfy": "commit",
            "artifacts": [
                {"kind": "artifact", "name": str(index), "signature": signature}
                for index, signature in enumerate(signatures)
            ],
        }

    monkeypatch.setattr(
        "dgx_monarch.actor.model_store.request_artifact_identity", identity)
    model = types.SimpleNamespace(request_dict=lambda: primary_source)
    model_request, _key, _identity, _artifacts, binding = capture(
        model, uncond_source)

    # The graph's dictionaries, the ceremony request and its binding are three
    # independent copies, down to nested options and LoRA entries.
    primary_source["options"]["dtype"]["name"] = "changed"
    primary_source["loras"].reverse()
    uncond_source["loras"][0]["strength"] = 9.0
    assert model_request["options"]["dtype"]["name"] == "bf16"
    assert [entry["name"] for entry in model_request["loras"]] == ["a.sft", "b.sft"]
    assert binding["model_request"] is not model_request
    assert binding["uncond_model_request"]["loras"][0]["strength"] == 0.4

    baseline = {
        "model": copy.deepcopy(model_request),
        "uncond_model": copy.deepcopy(binding["uncond_model_request"]),
    }
    assert assert_request_artifact_binding(
        {**copy.deepcopy(baseline), "_dgxm_artifact_binding": binding},
        copy.deepcopy(binding["artifact_sets"]),
    )
    mutators = [
        lambda request: request["model"]["options"]["dtype"].update(name="fp8"),
        lambda request: request["model"]["loras"][0].update(strength=0.3),
        lambda request: request["model"]["loras"].reverse(),
        lambda request: request["model"]["model_sampling"].update(shift=4.0),
        lambda request: request["uncond_model"]["options"]["dtype"].update(name="bf16"),
        lambda request: request["uncond_model"]["loras"][0].update(strength=0.5),
    ]
    for mutate in mutators:
        changed = copy.deepcopy(baseline)
        mutate(changed)
        changed["_dgxm_artifact_binding"] = binding
        with pytest.raises(ArtifactBindingError):
            assert_request_artifact_binding(changed, copy.deepcopy(binding["artifact_sets"]))


def test_direct_fleet_submit_marks_and_rejects_risky_unconditional_model(monkeypatch):
    """submit_sample_to marks its own copy of the request as a fleet job, so no
    caller can skip the grant check for a low-RSS unconditional LoRA stack."""
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.actor import model_store

    handle = mesh_mod.MeshHandle.__new__(mesh_mod.MeshHandle)
    handle.setup_generation = 1
    handle.active_worker_args = {"slab_weights": False, "lora_low_rss": True}
    handle.config = types.SimpleNamespace(hosts=(), source="local")
    handle.config_fingerprint = "local"
    handle.world, handle.n_hosts, handle.gpus_per_host = 1, 1, 1
    request = {
        "model": {"unet_name": "primary", "options": {}, "loras": []},
        "uncond_model": {
            "unet_name": "negative", "options": {},
            "loras": [{"name": "negative-style", "strength": 0.5}],
        },
    }
    monkeypatch.setattr(
        model_store, "request_artifact_identity",
        lambda name, _loras: {"digest": name, "comfy": "c", "artifacts": []},
    )
    actor_payloads = []
    verify = handle._verify_request_artifacts

    def capture_and_verify(payload, *, worker_index):
        actor_payloads.append(payload)
        return verify(payload, worker_index=worker_index)

    handle._verify_request_artifacts = capture_and_verify

    with pytest.raises(RuntimeError, match="without an identity-gate grant"):
        handle.submit_sample_to(0, request)
    assert "_dgxm_fleet_job" not in request
    assert len(actor_payloads) == 1
    assert actor_payloads[0] is not request
    assert actor_payloads[0]["_dgxm_fleet_job"] is True


def test_fleet_grant_binds_full_unconditional_model_snapshot():
    import copy

    from dgx_monarch.mesh_safety import (
        FLEET_RESIDENCY_CAPABILITY,
        assert_fleet_residency_grant,
    )

    policy = {"slab_weights": True, "lora_low_rss": True}
    primary = {
        "unet_name": "primary", "options": {"dtype": {"name": "bf16"}},
        "loras": [],
        "model_sampling": {"kind": "sd3", "shift": 3.0},
    }
    uncond = {
        "unet_name": "negative", "options": {"dtype": {"name": "fp16"}},
        "loras": [{"name": "negative-style", "strength": 0.5}],
    }
    artifacts = [
        {"digest": "primary", "comfy": "c", "artifacts": []},
        {"digest": "negative", "comfy": "c", "artifacts": []},
    ]
    context = {
        "capability": FLEET_RESIDENCY_CAPABILITY,
        "rank_world": 1,
        "worker_args": copy.deepcopy(policy),
    }
    grant = _fleet_gate_grant(
        primary, artifacts, context, uncond_model=uncond)
    request = {
        "model": primary,
        "uncond_model": uncond,
        "_dgxm_fleet_job": True,
        "_dgxm_fleet_residency_grant": grant,
    }
    assert assert_fleet_residency_grant(
        request, artifacts, effective_worker_args=policy, rank_world=1)

    changed_primary = copy.deepcopy(request)
    changed_primary["model"]["model_sampling"]["shift"] = 5.0
    with pytest.raises(RuntimeError, match="full fleet model request changed"):
        assert_fleet_residency_grant(
            changed_primary, artifacts, effective_worker_args=policy, rank_world=1)

    uncond["options"]["dtype"]["name"] = "changed"
    uncond["loras"][0]["strength"] = 0.75
    assert grant["uncond_model_request"]["options"]["dtype"]["name"] == "fp16"
    assert grant["uncond_model_request"]["loras"][0]["strength"] == 0.5
    with pytest.raises(RuntimeError, match="full fleet unconditional model request changed"):
        assert_fleet_residency_grant(
            request, artifacts, effective_worker_args=policy, rank_world=1)


def test_safe_fleet_without_grant_is_not_misreported_as_authorized():
    from dgx_monarch.mesh_safety import assert_fleet_residency_grant

    request = {
        "model": {"unet_name": "primary", "options": {}, "loras": []},
        "_dgxm_fleet_job": True,
    }
    assert assert_fleet_residency_grant(
        request,
        [],
        effective_worker_args={"slab_weights": False, "lora_low_rss": False},
    ) is False


@pytest.mark.parametrize(
    ("context_field", "context_value", "runtime_rank", "runtime_policy"),
    [
        ("capability", "wrong-capability", 1, {"slab_weights": True}),
        ("rank_world", 2, 1, {"slab_weights": True}),
        ("rank_world", 1, 2, {"slab_weights": True}),
        ("worker_args", {"slab_weights": False}, 1, {"slab_weights": True}),
    ],
)
def test_fleet_grant_rejects_each_world_and_policy_context_axis(
        context_field, context_value, runtime_rank, runtime_policy):
    from dgx_monarch.mesh_safety import (
        FLEET_RESIDENCY_CAPABILITY,
        assert_fleet_residency_grant,
        request_combo_key,
    )

    model = {"unet_name": "primary", "options": {}, "loras": []}
    artifacts = [{"digest": "primary", "comfy": "c", "artifacts": []}]
    context = {
        "capability": FLEET_RESIDENCY_CAPABILITY,
        "rank_world": 1,
        "worker_args": {"slab_weights": True},
    }
    context[context_field] = context_value
    request = {
        "model": model,
        "_dgxm_fleet_job": True,
        "_dgxm_fleet_residency_grant": {
            "combo_key": request_combo_key(model),
            "model_request": model,
            "uncond_model_request": None,
            "artifact_sets": artifacts,
            "capability_context": context,
        },
    }

    with pytest.raises(RuntimeError, match="world-1 worker policy"):
        assert_fleet_residency_grant(
            request,
            artifacts,
            effective_worker_args=runtime_policy,
            rank_world=runtime_rank,
        )


def test_gate_swap_cycle_rejects_replacement_between_transitions(monkeypatch):
    from dgx_monarch.actor import gate_cycle, model_store

    expected = {"digest": "expected", "comfy": "c", "artifacts": []}
    replacement = {"digest": "replacement", "comfy": "c", "artifacts": []}
    identities = iter((expected, expected, replacement))
    ensure_calls = []
    store = types.SimpleNamespace(
        current=types.SimpleNamespace(slab=None, artifact_identity=expected),
        lora_low_rss=True,
    )

    def ensure(*_args, **_kwargs):
        ensure_calls.append(1)
        store.current = types.SimpleNamespace(slab=None, artifact_identity=expected)
        return object(), "reuse"

    store.ensure = ensure
    worker = types.SimpleNamespace(
        _setup_key=("live",), _setup_generation=4, rank=0, world=1, store=store,
        _inject_for_topology=lambda *_args: None,
    )
    monkeypatch.setattr(
        model_store, "request_artifact_identity", lambda *_args: next(identities))

    with pytest.raises(RuntimeError, match="changed after the ceremony snapshot"):
        gate_cycle.run(
            worker, "m", {}, [{"name": "lora", "strength": 1.0}], expected)
    assert ensure_calls == [1]


def test_fleet_replacement_between_authorization_and_submit_is_rejected(monkeypatch):
    """A PASS for bytes A cannot enable risky residency after the driver path
    is atomically replaced with bytes B, even when every worker also sees B."""
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.actor import model_store
    from dgx_monarch.mesh_safety import (
        FLEET_RESIDENCY_CAPABILITY,
        physical_capability_context,
    )

    old = {"digest": "old", "comfy": "c", "artifacts": [
        {"kind": "diffusion_models", "name": "m", "signature": "old"}]}
    new = {"digest": "new", "comfy": "c", "artifacts": [
        {"kind": "diffusion_models", "name": "m", "signature": "new"}]}
    policy = {"slab_weights": True, "lora_low_rss": True}
    handle = mesh_mod.MeshHandle.__new__(mesh_mod.MeshHandle)
    handle.setup_generation = 3
    handle.active_worker_args = {"slab_weights": False, "lora_low_rss": False}
    handle.active_worker_args = dict(policy)
    handle.config = types.SimpleNamespace(hosts=(), source="local")
    handle.config_fingerprint = "config-a"
    handle.world, handle.n_hosts, handle.gpus_per_host = 2, 1, 2
    model = {"unet_name": "m", "options": {}, "loras": []}
    context = {
        "capability": FLEET_RESIDENCY_CAPABILITY,
        "rank_world": 1,
        "worker_args": dict(policy),
        **physical_capability_context(handle),
    }
    request = {
        "model": model,
        "_dgxm_fleet_job": True,
        "_dgxm_fleet_residency_grant": _fleet_gate_grant(
            model, [old], context),
    }
    monkeypatch.setattr(model_store, "request_artifact_identity", lambda *_args: new)

    with pytest.raises(RuntimeError, match="changed after fleet residency authorization"):
        handle._verify_request_artifacts(request, worker_index=0)


def test_fleet_grant_still_requires_selected_worker_parity(monkeypatch):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.actor import model_store
    from dgx_monarch.mesh_safety import (
        FLEET_RESIDENCY_CAPABILITY,
        physical_capability_context,
    )

    driver = {"digest": "driver", "comfy": "c", "artifacts": []}
    worker_identity = {"digest": "worker", "comfy": "c", "artifacts": []}
    policy = {"slab_weights": True, "lora_low_rss": True}
    handle = mesh_mod.MeshHandle.__new__(mesh_mod.MeshHandle)
    handle.setup_generation = 3
    handle.active_worker_args = dict(policy)
    handle.config = types.SimpleNamespace(hosts=(), source="local")
    handle.config_fingerprint = "config-a"
    handle.world, handle.n_hosts, handle.gpus_per_host = 1, 1, 1
    model = {"unet_name": "m", "options": {}, "loras": []}
    context = {
        "capability": FLEET_RESIDENCY_CAPABILITY,
        "rank_world": 1,
        "worker_args": dict(policy),
        **physical_capability_context(handle),
    }
    request = {
        "model": model,
        "_dgxm_fleet_job": True,
        "_dgxm_fleet_residency_grant": _fleet_gate_grant(
            model, [driver], context),
    }
    endpoint = types.SimpleNamespace(call_one=lambda _specs: {
        "host": "worker", "rank": 0, "artifact_sets": [worker_identity]})
    handle.workers = types.SimpleNamespace(
        extent=types.SimpleNamespace(labels=("gpus",)),
        slice=lambda **_coords: types.SimpleNamespace(artifact_identity=endpoint),
    )
    handle._await_or_evict = lambda value, timeout_s: value
    monkeypatch.setattr(model_store, "request_artifact_identity", lambda *_args: driver)

    with pytest.raises(RuntimeError, match="artifact parity"):
        handle._verify_request_artifacts(request, worker_index=0)


def test_worker_rechecks_fleet_grant_after_driver_preflight(monkeypatch):
    from dgx_monarch.actor import worker as worker_mod
    from dgx_monarch.mesh_safety import FLEET_RESIDENCY_CAPABILITY

    before = {"digest": "before", "comfy": "c", "artifacts": []}
    after = {"digest": "after", "comfy": "c", "artifacts": []}
    policy = {"slab_weights": True, "lora_low_rss": True}
    model = {"unet_name": "m", "options": {}, "loras": []}
    context = {
        "capability": FLEET_RESIDENCY_CAPABILITY,
        "rank_world": 1,
        "worker_args": dict(policy),
    }
    request = {
        "model": model,
        "_dgxm_fleet_job": True,
        "_dgxm_artifact_sets": [before],
        "_dgxm_fleet_residency_grant": _fleet_gate_grant(
            model, [before], context),
    }
    worker = worker_mod.GPUWorker.__new__(worker_mod.GPUWorker)
    worker._setup_key = ("fleet",)
    worker.rank, worker.world = 0, 1
    worker._active_worker_args = dict(policy)
    worker.store = types.SimpleNamespace(
        ensure=lambda *_args, **_kwargs: pytest.fail("replacement reached model load"))
    monkeypatch.setattr(worker_mod, "request_artifact_identity", lambda *_args: after)

    with pytest.raises(RuntimeError, match="changed after fleet residency authorization"):
        worker_mod.GPUWorker._sample_impl.__wrapped__(worker, request)


def test_worker_binds_loaded_model_to_grant_after_execution_recheck(monkeypatch):
    """A replacement after the worker's fresh identity check but before
    ModelStore.ensure must not reach sampling either."""
    from dgx_monarch.actor import worker as worker_mod
    from dgx_monarch.mesh_safety import FLEET_RESIDENCY_CAPABILITY

    granted = {"digest": "granted", "comfy": "c", "artifacts": []}
    replacement = {"digest": "replacement", "comfy": "c", "artifacts": []}
    policy = {"slab_weights": True, "lora_low_rss": True}
    model = {"unet_name": "m", "options": {}, "loras": []}
    context = {
        "capability": FLEET_RESIDENCY_CAPABILITY,
        "rank_world": 1,
        "worker_args": dict(policy),
    }
    request = {
        "model": model,
        "_dgxm_fleet_job": True,
        "_dgxm_artifact_sets": [granted],
        "_dgxm_fleet_residency_grant": _fleet_gate_grant(
            model, [granted], context),
    }
    store = types.SimpleNamespace(current=None)

    def ensure(*_args, **_kwargs):
        store.current = types.SimpleNamespace(artifact_identity=replacement)
        return object(), "load"

    store.ensure = ensure
    worker = worker_mod.GPUWorker.__new__(worker_mod.GPUWorker)
    worker._setup_key = ("fleet",)
    worker.rank, worker.world = 0, 1
    worker._active_worker_args = dict(policy)
    worker.store = store
    monkeypatch.setattr(worker_mod, "request_artifact_identity", lambda *_args: granted)

    with pytest.raises(RuntimeError, match="after the artifact snapshot"):
        worker_mod.GPUWorker._sample_impl.__wrapped__(worker, request)


def test_unbake_identity_rejects_same_size_same_mtime_replacement(tmp_path):
    from dgx_monarch.actor.unbake import UnbakeRecord

    path = tmp_path / "model.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    path.write_bytes(b"AAAA")
    captured = os.stat(path)
    record = UnbakeRecord(
        path=str(path),
        file_size=captured.st_size,
        file_mtime_ns=captured.st_mtime_ns,
        file_dev=captured.st_dev,
        file_ino=captured.st_ino,
        file_ctime_ns=captured.st_ctime_ns,
    )
    replacement.write_bytes(b"BBBB")
    os.utime(replacement, ns=(captured.st_atime_ns, captured.st_mtime_ns))
    replacement.replace(path)

    assert not record.stat_ok()


def test_latent_signature_streaming_preserves_full_byte_digest(monkeypatch):
    import dgx_monarch.actor.worker_status as status

    monkeypatch.setattr(status, "_SIGNATURE_CHUNK_ELEMENTS", 7)
    tensor = torch.arange(35, dtype=torch.float32).reshape(1, 5, 7)
    signature = status._latent_signature(tensor)
    expected = hashlib.sha256(
        memoryview(tensor.contiguous().view(torch.uint8).numpy())
    ).hexdigest()

    assert signature["sha256"] == expected
    assert signature["mean"] == round(float(tensor.double().mean()), 6)
    assert signature["std"] == round(float(tensor.double().std()), 6)
    assert len(signature["projection"]) == 64


def test_worker_source_manifest_is_stable_and_path_free():
    import dgx_monarch.actor.worker_status as status

    status.source_manifest_sha256.cache_clear()
    first = status.source_manifest_sha256()
    second = status.source_manifest_sha256()

    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")
    assert second == first


def test_telemetry_events_carry_monotonic_process_local_watermarks(monkeypatch):
    from collections import deque

    import dgx_monarch.telemetry as telemetry

    monkeypatch.setattr(telemetry, "_EVENTS", deque(maxlen=256))
    monkeypatch.setattr(telemetry, "_EVENT_SEQ", 0)

    telemetry.emit("load", model="first")
    baseline = telemetry.event_sequence()
    telemetry.emit("load", model="second", seq=999, t=0)
    snapshot_seq, snapshot_events = telemetry.event_snapshot()

    assert baseline == 1
    assert snapshot_seq == 2
    assert [event["seq"] for event in snapshot_events] == [1, 2]
    assert [event["seq"] for event in telemetry.events_tail()] == [1, 2]
    assert telemetry.events_tail()[-1]["t"] > 0


def test_worker_status_event_watermark_and_tail_use_one_atomic_snapshot(monkeypatch):
    import dgx_monarch.actor.worker_status as status
    import dgx_monarch.telemetry as telemetry

    calls = []

    def snapshot(n):
        calls.append(n)
        return 7, [{"seq": 7, "kind": "load"}]

    monkeypatch.setattr(telemetry, "event_snapshot", snapshot)
    monkeypatch.setattr(
        telemetry,
        "event_sequence",
        lambda: (_ for _ in ()).throw(AssertionError("separate watermark read")),
    )
    monkeypatch.setattr(
        telemetry,
        "events_tail",
        lambda *_args: (_ for _ in ()).throw(AssertionError("separate tail read")),
    )
    monkeypatch.setattr(status, "source_manifest_sha256", lambda: "a" * 64)
    worker = types.SimpleNamespace(
        rank=None,
        world=None,
        topology={},
        store=types.SimpleNamespace(snapshot=lambda: {}),
        _setup_key=None,
    )

    result = status.status_impl(worker)

    assert calls == [32]
    assert result["event_seq"] == 7
    assert result["events"] == [{"seq": 7, "kind": "load"}]


def test_worker_provenance_baseline_is_rank_source_and_generation_bound(monkeypatch):
    import dgx_monarch.actor.comfy_bridge as comfy_bridge
    import dgx_monarch.actor.worker_status as status
    import dgx_monarch.telemetry as telemetry

    setup_snapshot = {"schema": 1, "phase": "ready"}
    post_snapshot = {
        "schema": 1,
        "phase": "post",
        "dgx_monarch": {"source_manifest_sha256": "b" * 64},
    }
    manifests = []
    monkeypatch.setattr(
        status,
        "runtime_provenance_snapshot",
        lambda manifest=None: manifests.append(manifest) or post_snapshot,
    )
    monkeypatch.setattr(telemetry, "event_sequence", lambda: 19)
    monkeypatch.setattr(comfy_bridge, "_CUSTOM_NODES_DISABLED", True)
    worker = types.SimpleNamespace(
        rank=1,
        world=2,
        topology={"ulysses": 2},
        _setup_key=("ready",),
        _setup_generation=4,
        _setup_provenance=(4, setup_snapshot),
    )

    artifact_manifest = [{"id": "model", "kind": "diffusion_models", "file": "m"}]
    assert status.provenance_baseline_impl(worker, 4, artifact_manifest) == {
        "rank": 1,
        "world": 2,
        "topology": {"ulysses": 2},
        "setup_generation": 4,
        "source_manifest_sha256": "b" * 64,
        "setup_provenance": setup_snapshot,
        "post_provenance": post_snapshot,
        "event_seq": 19,
    }
    assert manifests == [artifact_manifest]
    with pytest.raises(RuntimeError, match="generation is stale"):
        status.provenance_baseline_impl(worker, 3)


def test_worker_provenance_legacy_baseline_does_not_require_strict_comfy(
    monkeypatch,
):
    import dgx_monarch.actor.comfy_bridge as comfy_bridge
    import dgx_monarch.actor.worker_status as status
    import dgx_monarch.telemetry as telemetry

    monkeypatch.setattr(comfy_bridge, "_CUSTOM_NODES_DISABLED", False)
    monkeypatch.setattr(status, "source_manifest_sha256", lambda: "c" * 64)
    monkeypatch.setattr(telemetry, "event_sequence", lambda: 23)
    worker = types.SimpleNamespace(
        rank=0,
        world=1,
        topology={"ulysses": 1},
        _setup_key=("ready",),
        _setup_generation=5,
        _setup_provenance=None,
    )

    assert status.provenance_baseline_impl(worker, 5) == {
        "rank": 0,
        "world": 1,
        "topology": {"ulysses": 1},
        "setup_generation": 5,
        "source_manifest_sha256": "c" * 64,
        "event_seq": 23,
    }
    with pytest.raises(RuntimeError, match="requires custom nodes disabled"):
        status.provenance_baseline_impl(worker, 5, [])


@pytest.mark.parametrize("configured_mode", [True, "auto"])
def test_memory_status_reports_actual_stock_residency_after_slab_fallback(
        monkeypatch, configured_mode):
    import dgx_monarch.actor.worker_status as status

    comfy = types.ModuleType("comfy")
    memory_management = types.ModuleType("comfy.memory_management")
    memory_management.aimdo_enabled = False
    utils = types.ModuleType("comfy.utils")
    utils.DISABLE_MMAP = False
    comfy.memory_management = memory_management
    comfy.utils = utils
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.memory_management", memory_management)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)

    patcher = types.SimpleNamespace(
        backup={},
        model=types.SimpleNamespace(
            model_loaded_weight_memory=None,
            model_lowvram=False,
        ),
    )
    stored = types.SimpleNamespace(
        active_patcher=patcher,
        base_key=("future-dtype.safetensors", ()),
        quant_kind="unknown",
        lora_sig=(),
        slab=None,
        unbake=None,
        residency_rung="",   # not comfy_managed: weight_residency reads slab or cudaMalloc
    )
    worker = types.SimpleNamespace(
        store=types.SimpleNamespace(
            current=stored,
            slab_weights=configured_mode,
            lora_low_rss=False,
            swap_verify=2,
            verify_failures=0,
        ),
        uma_reserve_gb=0.0,
    )

    detail = status.memory_detail(worker)

    assert detail["weight_residency"] == "cudaMalloc"
    stored.slab = types.SimpleNamespace(telemetry=lambda: {"slab_gib": 1.0})
    assert status.memory_detail(worker)["weight_residency"] == "slab"


_FAKE_UNBAKE_RECORD = types.SimpleNamespace(
    mapped={}, quant={}, mapped_bytes=0, resident_count=0, resident_bytes=0)


@pytest.mark.parametrize("lora_low_rss", [True, False])
@pytest.mark.parametrize("unbake,lora_sig,expected", [
    (_FAKE_UNBAKE_RECORD, ("lora-a",), "low_rss"),  # un-baked at least once
    (None, ("lora-a",), "hot_swap"),                # loras resident, never un-baked
    (None, (), "n/a"),                              # no loras at all
])
def test_lora_mode_reports_actual_backing_not_store_policy(
        monkeypatch, unbake, lora_sig, expected, lora_low_rss):
    """lora_mode reports what backs the resident model, as weight_residency
    does, never the store-wide policy flag: a model that never un-baked reports
    hot_swap or n/a even with lora_low_rss=on."""
    import dgx_monarch.actor.worker_status as status

    comfy = types.ModuleType("comfy")
    memory_management = types.ModuleType("comfy.memory_management")
    memory_management.aimdo_enabled = False
    utils = types.ModuleType("comfy.utils")
    utils.DISABLE_MMAP = False
    comfy.memory_management = memory_management
    comfy.utils = utils
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.memory_management", memory_management)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils)

    patcher = types.SimpleNamespace(
        backup={},
        model=types.SimpleNamespace(model_loaded_weight_memory=None, model_lowvram=False),
    )
    stored = types.SimpleNamespace(
        active_patcher=patcher,
        base_key=("m.safetensors", ()),
        quant_kind="bf16",
        lora_sig=lora_sig,
        slab=None,
        unbake=unbake,
        residency_rung="",
    )
    worker = types.SimpleNamespace(
        store=types.SimpleNamespace(
            current=stored,
            slab_weights=False,
            lora_low_rss=lora_low_rss,   # varied: must not affect the result
            swap_verify=0,
            verify_failures=0,
        ),
        uma_reserve_gb=0.0,
    )

    assert status.memory_detail(worker)["lora_mode"] == expected


def test_render_progress_waits_for_every_pipeline_render():
    from dgx_monarch.telemetry import RenderProgress

    progress = RenderProgress()
    progress.start(10, {"model": "a"})
    progress.start(10, {"model": "b"})
    assert progress.snapshot()["active_renders"] == 2
    progress.finish()
    assert progress.snapshot()["active"] is True
    assert progress.snapshot()["active_renders"] == 1
    progress.finish()
    assert progress.snapshot()["active"] is False


def test_render_progress_tokens_cannot_finish_another_render():
    from dgx_monarch.telemetry import RenderProgress

    class StopNow(BaseException):
        pass

    progress = RenderProgress()
    first, second = object(), object()
    progress.start(10, {"model": "first"}, token=first)

    # Cleanup for a start that failed before registration is a no-op for the
    # already-active render.
    progress.finish(second)
    assert progress.snapshot()["active_renders"] == 1


    # Interrupt the second start at its one state commit, so its token never
    # registers. Finishing that token must leave only the first render live.
    original_setattr = RenderProgress.__setattr__
    armed = True

    def interrupt_state(self, name, value):
        nonlocal armed
        if self is progress and name == "_state" and armed:
            armed = False
            raise StopNow("start state publication interrupted")
        original_setattr(self, name, value)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(RenderProgress, "__setattr__", interrupt_state)
    try:
        with pytest.raises(StopNow):
            progress.start(10, {"model": "second"}, token=second)
    finally:
        monkeypatch.undo()
    progress.finish(second)

    assert progress.snapshot()["active_renders"] == 1
    progress.finish(first)
    assert progress.snapshot()["active"] is False


def test_render_progress_explicit_finish_is_state_idempotent():
    from dgx_monarch.telemetry import RenderProgress

    progress = RenderProgress()
    token = object()
    progress.start(3, token=token)
    progress.finish(token)
    completed = progress.snapshot()

    progress.finish(token)

    assert progress.snapshot() == completed
    assert completed["active"] is False
    assert completed["last_wall_s"] is not None


def test_render_progress_finish_retry_heals_interrupted_commit(monkeypatch):
    from dgx_monarch.telemetry import RenderProgress

    class StopNow(BaseException):
        pass

    progress = RenderProgress()
    token = object()
    progress.start(3, token=token)
    original_setattr = RenderProgress.__setattr__
    armed = True

    def interrupt_token_commit(self, name, value):
        nonlocal armed
        if self is progress and name == "_state" and armed:
            armed = False
            raise StopNow("finish token commit interrupted")
        original_setattr(self, name, value)

    monkeypatch.setattr(RenderProgress, "__setattr__", interrupt_token_commit)
    with pytest.raises(StopNow, match="finish token commit interrupted"):
        progress.finish(token)

    progress.finish(token)

    snapshot = progress.snapshot()
    assert snapshot["active"] is False
    assert snapshot["active_renders"] == 0
    assert snapshot["last_wall_s"] is not None


def test_progress_receiver_defers_channel_open_until_guarded_enter(monkeypatch):
    from monarch.actor import Channel

    from dgx_monarch.progress import ProgressReceiver

    opened = []
    sent = []

    class Future:
        def get(self, timeout=None):
            return {"done": True}

    class Receiver:
        def recv(self):
            return Future()

    class Port:
        def send(self, value):
            sent.append(value)

    monkeypatch.setattr(
        Channel, "open",
        lambda: (opened.append(True) or Port(), Receiver()),
    )

    progress = ProgressReceiver(2)
    assert opened == []
    progress.__enter__()
    assert opened == [True]
    progress.__exit__(None, None, None)
    assert sent == [{"done": True}]


def test_gpu_telemetry_uses_cuda_visible_device(monkeypatch):
    from dgx_monarch.telemetry import _visible_gpu_selector

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    assert _visible_gpu_selector() == "3"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-deadbeef")
    assert _visible_gpu_selector() == "GPU-deadbeef"


def test_uma_smart_memory_default_and_worker_args_are_reversible(monkeypatch):
    from dgx_monarch.actor import comfy_bridge

    fake = types.ModuleType("comfy.model_management")
    fake.EXTRA_RESERVED_VRAM = 0
    fake.NUM_STREAMS = 4
    fake.DISABLE_SMART_MEMORY = False
    fake.MAX_PINNED_MEMORY = 1024
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake)
    comfy = types.ModuleType("comfy")
    comfy.model_management = fake
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setattr(comfy_bridge, "_MEMORY_BASELINE_MODULE", None)

    defaults = comfy_bridge._uma_memory_defaults({}, integrated=True)
    assert defaults["disable_smart_memory"] is True
    comfy_bridge._apply_worker_args({
        "disable_async_offload": True,
        "disable_smart_memory": True,
        "disable_pinned_memory": True,
    })
    assert (fake.NUM_STREAMS, fake.DISABLE_SMART_MEMORY, fake.MAX_PINNED_MEMORY) == (0, True, -1)
    comfy_bridge._apply_worker_args({})
    assert (fake.NUM_STREAMS, fake.DISABLE_SMART_MEMORY, fake.MAX_PINNED_MEMORY) == (4, False, 1024)


def test_worker_artifact_identity_payload_is_stable(monkeypatch):
    from dgx_monarch.actor import model_store

    monkeypatch.setattr(model_store, "resolve_model_path", lambda kind, name: f"/{kind}/{name}")
    monkeypatch.setattr("dgx_monarch.gate_ledger.artifact_signature", lambda path: path)
    monkeypatch.setattr("dgx_monarch.gate_ledger.comfy_commit", lambda: "commit")
    identity = model_store.request_artifact_identity(
        "base.sft", [{"name": "style.sft", "strength": 0.5}])

    assert identity["comfy"] == "commit"
    assert [item["name"] for item in identity["artifacts"]] == ["base.sft", "style.sft"]
    # The digest is the untruncated sha256 composite, not the legacy hyphen join cut at 96 characters.
    assert len(identity["digest"]) == 64


def test_failed_old_setup_teardown_cannot_leave_a_reusable_stale_key():
    from dgx_monarch.actor import worker_env

    worker = types.SimpleNamespace(
        _setup_key=(2, "old"),
        _setup_cleanup_failed=False,
        rank=1,
        world=2,
        topology={"ulysses": 2},
        _latent_return=object(),
        _teardown_parallel_state=lambda: (_ for _ in ()).throw(
            RuntimeError("old group teardown failed")),
    )

    with pytest.raises(RuntimeError, match="old group teardown failed"):
        worker_env.teardown_existing_setup(worker)

    assert worker._setup_key is None
    assert worker._setup_provenance is None
    assert worker.rank is None and worker.world is None and worker.topology == {}
    assert worker._setup_cleanup_failed is True


def test_post_nccl_setup_failure_rolls_back_all_worker_state(monkeypatch, isolated_environ):
    from dgx_monarch.actor import comfy_bridge, worker_env

    class StopNow(BaseException):
        pass

    initialized = False
    destroyed = []

    def init_process_group(*args, **kwargs):
        nonlocal initialized
        initialized = True

    def destroy_process_group():
        nonlocal initialized
        initialized = False
        destroyed.append("pg")

    monkeypatch.setattr(worker_env, "ensure_comfy", lambda *args, **kwargs: None)
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", lambda args: dict(args))
    monkeypatch.setattr("dgx_monarch.config.fixup_fabric_ifaces", lambda env: (env, ""))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.distributed, "init_process_group", init_process_group)
    monkeypatch.setattr(worker_env.rendezvous, "generation_store", lambda *_a, **_k: None)
    monkeypatch.setattr(torch.distributed, "destroy_process_group", destroy_process_group)

    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.init_distributed_environment = lambda **kwargs: None
    distributed.initialize_model_parallel = lambda **kwargs: None
    core = types.ModuleType("xfuser.core")
    core.distributed = distributed
    xfuser = types.ModuleType("xfuser")
    xfuser.core = core
    monkeypatch.setitem(sys.modules, "xfuser", xfuser)
    monkeypatch.setitem(sys.modules, "xfuser.core", core)
    monkeypatch.setitem(sys.modules, "xfuser.core.distributed", distributed)

    teardown = []
    old_latent_return = object()
    worker = types.SimpleNamespace(
        _setup_key=("old",),
        _setup_cleanup_failed=False,
        _latent_return=old_latent_return,
        rank=None,
        world=None,
        topology={},
        _attn=types.SimpleNamespace(
            configure=lambda *_args, **_kwargs: (_ for _ in ()).throw(StopNow("attention interrupted"))
        ),
        _teardown_parallel_state=lambda: teardown.append("parallel"),
        _slab_mode_effective=lambda *_args: False,
        _nccl_launch_order_needed=lambda _topo: False,
        store=types.SimpleNamespace(
            family_override=None,
            set_lora_mode=lambda *_args: None,
            set_slab_mode=lambda *_args: None,
            swap_verify=2,
        ),
    )
    env = {
        "setup_generation": 2,
        "world": 1,
        "rank": 0,
        "master_addr": "127.0.0.1",
        "master_port": 23456,
        "topology": {"ulysses": 1, "ring": 1, "cfg": 1, "dp": 1, "fsdp": False},
        "fabric_env": {},
        "comfy_dir": "/comfy",
        "worker_args": {},
        "local_gpu_index": 0,
    }

    with pytest.raises(StopNow, match="attention interrupted"):
        worker_env.setup_impl(worker, env)
    assert worker._setup_key is None
    assert worker._setup_provenance is None
    assert worker.rank is None and worker.world is None and worker.topology == {}
    assert worker._latent_return is old_latent_return
    assert teardown == ["parallel", "parallel"]
    assert destroyed == ["pg"]

    ordinary = RuntimeError("ordinary attention setup failure")
    cleanup_cancel = KeyboardInterrupt("setup rollback cancelled")
    worker._setup_key = None
    worker._setup_cleanup_failed = False
    worker._attn.configure = lambda *_args, **_kwargs: (
        _ for _ in ()).throw(ordinary)
    worker._teardown_parallel_state = lambda: (
        _ for _ in ()).throw(cleanup_cancel)

    with pytest.raises(KeyboardInterrupt) as caught:
        worker_env.setup_impl(worker, env)

    assert caught.value is cleanup_cancel
    assert caught.value.__cause__ is ordinary
    assert worker._setup_key is None
    assert worker._setup_cleanup_failed is True
    assert destroyed == ["pg", "pg"]

    setup_cancel = KeyboardInterrupt("attention setup cancelled")
    ordinary_cleanup = RuntimeError("ordinary setup rollback failure")
    worker._setup_cleanup_failed = False
    worker._attn.configure = lambda *_args, **_kwargs: (
        _ for _ in ()).throw(setup_cancel)
    worker._teardown_parallel_state = lambda: (
        _ for _ in ()).throw(ordinary_cleanup)

    with pytest.raises(KeyboardInterrupt) as caught:
        worker_env.setup_impl(worker, env)

    assert caught.value is setup_cancel
    assert caught.value.__cause__ is ordinary_cleanup
    assert worker._setup_cleanup_failed is True
    assert destroyed == ["pg", "pg", "pg"]


def test_worker_setup_is_dirty_before_pre_nccl_side_effects(monkeypatch, isolated_environ):
    from dgx_monarch.actor import worker_env

    class StopNow(BaseException):
        pass

    worker = types.SimpleNamespace(
        _setup_key=None,
        _setup_cleanup_failed=False,
        rank=None,
        world=None,
        topology={},
    )
    env = {
        "setup_generation": 1,
        "world": 1,
        "rank": 0,
        "master_addr": "127.0.0.1",
        "master_port": 23456,
        "topology": {"ulysses": 1, "ring": 1, "cfg": 1, "dp": 1, "fsdp": False},
        "fabric_env": {},
        "comfy_dir": "/comfy",
        "worker_args": {},
        "local_gpu_index": 0,
    }

    def stop_before_comfy(*_args, **_kwargs):
        assert worker._setup_cleanup_failed is True
        raise StopNow("pre-NCCL setup interrupted")

    monkeypatch.setattr(worker_env, "ensure_comfy", stop_before_comfy)
    monkeypatch.setattr(
        "dgx_monarch.config.fixup_fabric_ifaces", lambda value: (value, ""))

    with pytest.raises(StopNow, match="pre-NCCL setup interrupted"):
        worker_env.setup_impl(worker, env)

    assert worker._setup_cleanup_failed is True


def _no_generation_probe(monkeypatch):
    """Stand in for the listener-generation reading the attach takes.

    The attach reads each loop's listener generation over the lifecycle
    runner before every attempt, and that runner is ssh for a remote host.
    These tests cover the transport and the poison, so the reading is canned
    and no test here reaches the network.
    """
    from dgx_monarch.cli import listener_generation

    monkeypatch.setattr(
        listener_generation, "fleet",
        lambda config, **_kwargs: {host.address: {"gen": "aaaaaaaaaaaa"}
                                   for host in config.hosts})


def test_failed_cluster_attach_poisons_irreversible_transport(monkeypatch):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.config import ClusterConfig, HostConfig

    _no_generation_probe(monkeypatch)

    actor = types.ModuleType("monarch.actor")
    actor.enable_transport = lambda _bind: None
    monkeypatch.setitem(sys.modules, "monarch.actor", actor)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_MODE", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_BIND", None)
    monkeypatch.setattr(
        mesh_mod, "_attach_once", lambda _addresses: (_ for _ in ()).throw(RuntimeError("attach")))
    config = ClusterConfig(
        hosts=(HostConfig("a", "tcp://10.0.0.1:26600"),),
        client_bind="tcp://10.0.0.2:0",
        auto_heal=False,
        transport_security="trusted_fabric",
    )

    with pytest.raises(mesh_mod.MeshAttachError, match="attach to workers"):
        mesh_mod._attach_cluster(config)
    assert mesh_mod._TRANSPORT_MODE is None
    assert mesh_mod._TRANSPORT_BIND == config.client_bind
    assert "cluster attach failed" in mesh_mod._TRANSPORT_POISON


def test_enable_transport_exception_poisons_even_if_native_init_was_partial(monkeypatch):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.config import ClusterConfig, HostConfig

    actor = types.ModuleType("monarch.actor")
    actor.enable_transport = lambda _bind: (_ for _ in ()).throw(
        RuntimeError("enable failed after mutation"))
    monkeypatch.setitem(sys.modules, "monarch.actor", actor)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_MODE", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_BIND", None)
    config = ClusterConfig(
        hosts=(HostConfig("a", "tcp://10.0.0.1:26600"),),
        client_bind="tcp://10.0.0.2:0",
        auto_heal=False,
        transport_security="trusted_fabric",
    )

    with pytest.raises(mesh_mod.MeshAttachError, match="partially initialized"):
        mesh_mod._attach_cluster(config)
    assert mesh_mod._TRANSPORT_MODE is None
    assert mesh_mod._TRANSPORT_BIND is None
    assert "enable failed after mutation" in mesh_mod._TRANSPORT_POISON


def test_pre_attach_heal_exception_is_retryable_after_transport_enable(monkeypatch):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.config import ClusterConfig, HostConfig

    _no_generation_probe(monkeypatch)

    actor = types.ModuleType("monarch.actor")
    enables = []
    actor.enable_transport = lambda bind: enables.append(bind)
    monkeypatch.setitem(sys.modules, "monarch.actor", actor)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_MODE", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_BIND", None)
    heals = iter((RuntimeError("heal exploded"), None))

    def heal(*_args):
        outcome = next(heals)
        if outcome is not None:
            raise outcome

    monkeypatch.setattr(mesh_mod, "_heal_dead_loops", heal)
    attached = object()
    monkeypatch.setattr(mesh_mod, "_attach_once", lambda _addresses: attached)
    config = ClusterConfig(
        hosts=(HostConfig("a", "tcp://10.0.0.1:26600"),),
        client_bind="tcp://10.0.0.2:0",
        auto_heal=True,
        transport_security="trusted_fabric",
    )

    with pytest.raises(mesh_mod.MeshAttachError, match="fix the lifecycle/SSH error and retry"):
        mesh_mod._attach_cluster(config)
    assert mesh_mod._TRANSPORT_MODE is None
    assert mesh_mod._TRANSPORT_BIND == config.client_bind
    assert mesh_mod._TRANSPORT_POISON is None

    assert mesh_mod._attach_cluster(config) is attached
    assert enables == ["tcp://10.0.0.2:0"]


def test_enabled_cluster_transport_rejects_a_different_client_bind(monkeypatch):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.config import ClusterConfig, HostConfig

    _no_generation_probe(monkeypatch)

    actor = types.ModuleType("monarch.actor")
    enables = []
    actor.enable_transport = lambda bind: enables.append(bind)
    monkeypatch.setitem(sys.modules, "monarch.actor", actor)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_MODE", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_BIND", None)
    attached = object()
    monkeypatch.setattr(mesh_mod, "_attach_once", lambda _addresses: attached)
    host = (HostConfig("a", "tcp://10.0.0.1:26600"),)

    first = ClusterConfig(
        hosts=host, client_bind="tcp://10.0.0.2:0", auto_heal=False,
        transport_security="trusted_fabric")
    second = ClusterConfig(
        hosts=host, client_bind="tcp://10.0.0.3:0", auto_heal=False,
        transport_security="trusted_fabric")

    assert mesh_mod._attach_cluster(first) is attached
    with pytest.raises(mesh_mod.MeshAttachError, match="requires a restart"):
        mesh_mod._attach_cluster(second)
    assert enables == ["tcp://10.0.0.2:0"]


def test_artifact_preflight_marker_hit_skips_identity_io(monkeypatch):
    """A marker hit must return before any identity I/O, since the sampled-window
    reads cost the most; the worker still derives identity and refuses drift."""
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.actor import model_store

    expected = {"digest": "a", "comfy": "c", "artifacts": []}
    identity_calls = []

    def identity(*_args):
        identity_calls.append(1)
        return expected

    monkeypatch.setattr(model_store, "request_artifact_identity", identity)
    handle = mesh_mod.MeshHandle.__new__(mesh_mod.MeshHandle)
    handle.setup_generation = 3
    handle.active_worker_args = {"slab_weights": False, "lora_low_rss": False}
    handle.call_all = lambda _endpoint, _specs, timeout_s: [
        {"host": "one", "rank": 0, "artifact_sets": [expected]},
    ]
    request = {"model": {"unet_name": "m", "loras": []}}
    handle._verify_request_artifacts(request)
    assert identity_calls == [1]
    handle._verify_request_artifacts(request)  # marker hit: no new identity read
    assert identity_calls == [1]


def test_unbake_stat_ok_survives_metadata_only_changes(tmp_path):
    """chmod and chown change ctime, not bytes; the record must stay valid, or
    a permission fix costs a full reload."""
    import os

    from dgx_monarch.actor.unbake import UnbakeRecord

    path = tmp_path / "model.safetensors"
    path.write_bytes(b"AAAA")
    captured = os.stat(path)
    record = UnbakeRecord(
        path=str(path),
        file_size=captured.st_size,
        file_mtime_ns=captured.st_mtime_ns,
        file_dev=captured.st_dev,
        file_ino=captured.st_ino,
        file_ctime_ns=captured.st_ctime_ns,
    )
    os.chmod(path, 0o600)
    assert record.stat_ok()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "lost its required"),
        ("stale", "READY setup generation"),
        ("tampered", "token does not match"),
    ],
)
def test_worker_rechecks_normal_grant_before_model_load(
    monkeypatch, mutation, message,
):
    from dgx_monarch.actor import worker as worker_mod

    request, identity, worker_args, _context, _setup_key, _wa_key, topology = (
        _normal_grant_request()
    )
    request["_dgxm_artifact_sets"] = [identity]
    grant = request["_dgxm_normal_residency_grant"]
    if mutation == "missing":
        request.pop("_dgxm_normal_residency_grant")
    elif mutation == "stale":
        grant["setup_generation"] = 8
    else:
        grant["gate_token"] = ["forged"]

    worker = worker_mod.GPUWorker.__new__(worker_mod.GPUWorker)
    worker._setup_key = ("ready",)
    worker._setup_generation = 7
    worker.rank, worker.world = 0, 2
    worker.topology = dict(topology)
    worker._active_worker_args = dict(worker_args)
    worker.store = types.SimpleNamespace(
        ensure=lambda *_args, **_kwargs: pytest.fail(
            "unauthorized request reached ModelStore.ensure"
        )
    )
    monkeypatch.setattr(
        worker_mod, "request_artifact_identity", lambda *_args: identity
    )

    with pytest.raises(RuntimeError, match=message):
        worker_mod.GPUWorker._sample_impl.__wrapped__(worker, request)


def _fleet_gate_grant(model, artifacts, context, *, uncond_model=None):
    from copy import deepcopy

    from dgx_monarch import __version__
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION, gate_verdict_token
    from dgx_monarch.mesh_safety import request_combo_key

    identity = artifacts[0]
    combo = request_combo_key(model)
    token = gate_verdict_token(
        combo, identity["digest"], identity["comfy"], context)
    return {
        "gate_protocol": GATE_PROTOCOL_VERSION,
        "dgx_monarch": __version__,
        "comfy": identity["comfy"],
        "artifact_digest": identity["digest"],
        "gate_token": list(token),
        "combo_key": combo,
        "model_request": deepcopy(model),
        "uncond_model_request": deepcopy(uncond_model),
        "artifact_sets": deepcopy(artifacts),
        "capability_context": deepcopy(context),
    }


def test_fleet_grant_rejects_driver_worker_source_manifest_mismatch(monkeypatch):
    import dgx_monarch.gate_ledger as ledger_mod
    from dgx_monarch.actor import worker as worker_mod
    from dgx_monarch.mesh_safety import FLEET_RESIDENCY_CAPABILITY

    source = ["a" * 64]
    monkeypatch.setattr(
        ledger_mod.runtime_provenance,
        "cached_dgx_source_manifest_sha256",
        lambda: source[0],
    )
    identity = {"digest": "granted", "comfy": "c", "artifacts": []}
    policy = {"slab_weights": True, "lora_low_rss": True}
    model = {"unet_name": "m", "options": {}, "loras": []}
    context = {
        "capability": FLEET_RESIDENCY_CAPABILITY,
        "rank_world": 1,
        "worker_args": dict(policy),
    }
    request = {
        "model": model,
        "_dgxm_fleet_job": True,
        "_dgxm_artifact_sets": [identity],
        "_dgxm_fleet_residency_grant": _fleet_gate_grant(
            model, [identity], context),
    }
    worker = worker_mod.GPUWorker.__new__(worker_mod.GPUWorker)
    worker._setup_key = ("fleet",)
    worker.rank, worker.world = 0, 1
    worker._active_worker_args = dict(policy)
    worker.store = types.SimpleNamespace(
        ensure=lambda *_args, **_kwargs: pytest.fail("source skew reached model load"))
    monkeypatch.setattr(
        worker_mod, "request_artifact_identity", lambda *_args: identity)

    source[0] = "b" * 64
    with pytest.raises(RuntimeError, match="grant token does not match"):
        worker_mod.GPUWorker._sample_impl.__wrapped__(worker, request)


def test_normal_grant_rejects_driver_worker_source_manifest_mismatch(monkeypatch):
    import dgx_monarch.gate_ledger as ledger_mod
    from dgx_monarch.actor import worker as worker_mod

    source = ["a" * 64]
    monkeypatch.setattr(
        ledger_mod.runtime_provenance,
        "cached_dgx_source_manifest_sha256",
        lambda: source[0],
    )
    request, identity, worker_args, _context, _setup_key, _wa_key, topology = (
        _normal_grant_request())
    request["_dgxm_artifact_sets"] = [identity]
    worker = worker_mod.GPUWorker.__new__(worker_mod.GPUWorker)
    worker._setup_key = ("ready",)
    worker._setup_generation = 7
    worker.rank, worker.world = 0, 2
    worker.topology = dict(topology)
    worker._active_worker_args = dict(worker_args)
    worker.store = types.SimpleNamespace(
        ensure=lambda *_args, **_kwargs: pytest.fail("source skew reached model load"))
    monkeypatch.setattr(
        worker_mod, "request_artifact_identity", lambda *_args: identity)

    source[0] = "b" * 64
    with pytest.raises(RuntimeError, match="grant token does not match"):
        worker_mod.GPUWorker._sample_impl.__wrapped__(worker, request)
