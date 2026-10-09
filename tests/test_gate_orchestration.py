"""Identity-gate orchestration and first-use pipeline regressions."""
import json
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

import dgx_monarch.gate_ledger as ledger_mod
import dgx_monarch.nodes.common as common
import dgx_monarch.nodes.gate as gate_mod
import dgx_monarch.nodes.render_quarantine as quarantine_mod
import dgx_monarch.nodes.render_submit as submit_mod
import dgx_monarch.nodes.samplers as samplers
import dgx_monarch.runtime_provenance as runtime_provenance
from dgx_monarch.actor import model_store as model_store_mod
from dgx_monarch.mesh import MeshHandle
from dgx_monarch.mesh_setup import SetupToken
from dgx_monarch.nodes import gate_process_state
from dgx_monarch.nodes.render_session import (
    ConcurrentRenderSessionError,
    RenderSession,
)
from gate_orchestration_helpers import (  # noqa: F401  # autouse fixture import.
    _clear_process_gate_verdicts,
    _cross_mode_rig,
    _identity,
    _run,
)


def _normal_authorization_rig(
    monkeypatch,
    tmp_path,
    lookup,
    session_verdict=None,
    *,
    session_pass_safe=True,
):
    from dgx_monarch.nodes import gate_identity
    from dgx_monarch.nodes.common import MeshSpec, ModelSpec
    from dgx_monarch.topology import Topology

    handle = SimpleNamespace(
        config=SimpleNamespace(worker_args={}, hosts=(), source=""),
        config_fingerprint="local",
        world=1,
        n_hosts=1,
        gpus_per_host=1,
    )
    handle.effective_worker_args = lambda requested: dict(requested)
    model = ModelSpec(
        mesh=MeshSpec(
            handle=handle,
            topology_preset="single",
            attention="TORCH_FLASH",
            sync_ulysses=False,
            worker_args={"slab_weights": True, "lora_low_rss": True},
            auto_gate="first_use",
        ),
        unet_name="model.safetensors",
        loras=({"name": "adapter.safetensors", "strength": 1.0},),
    )

    class Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_integrity(self, _key, _artifacts, _commit, context):
            state, entry = lookup(context)
            return ledger_mod.GateLedgerLookup(
                state, entry, session_pass_safe)

    monkeypatch.setattr(ledger_mod, "GateLedger", Ledger)
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(model_store_mod, "request_artifact_identity", _identity)
    authorization = gate_identity.authorize_normal_render(
        model,
        handle,
        gate_active=False,
        session_verdict=session_verdict or (lambda _token: None),
        resolved_topology=Topology(world=1),
        resolved_attention="TORCH_FLASH",
    )
    return model, handle, authorization


@pytest.mark.parametrize(
    ("state", "entry", "session", "expected_mode"),
    [
        ("pass", {"verdict": "PASS"}, None, "required"),
        ("unknown", None, "PASS", "required"),
        ("stale", {"verdict": "PASS"}, "PASS", "required"),
        ("inconclusive", {"verdict": "INCONCLUSIVE"}, "PASS", "stock"),
        ("unknown", {"verdict": "BROKEN"}, "PASS", "stock"),
        ("pass", {"verdict": "PASS"}, "FAIL", "stock"),
    ],
)
def test_normal_render_authority_obeys_durable_and_session_precedence(
    monkeypatch, tmp_path, state, entry, session, expected_mode,
):
    _model, _handle, authorization = _normal_authorization_rig(
        monkeypatch,
        tmp_path,
        lambda _context: (state, entry),
        session_verdict=lambda _token: session,
    )

    assert authorization.residency_mode == expected_mode
    assert (authorization.residency_grant is not None) == (
        expected_mode == "required")
    if expected_mode == "stock":
        assert authorization.worker_args["slab_weights"] is False
        assert authorization.worker_args["lora_low_rss"] is False


def test_normal_render_pass_is_scoped_to_concrete_topology(
    monkeypatch, tmp_path,
):
    from dgx_monarch.nodes import gate_identity
    from dgx_monarch.topology import Topology

    model, handle, first = _normal_authorization_rig(
        monkeypatch,
        tmp_path,
        lambda context: (
            ("pass", {"verdict": "PASS"})
            if context["resolved_topology"]["dp"] == 1
            else ("unknown", None)
        ),
    )
    second = gate_identity.authorize_normal_render(
        model,
        handle,
        gate_active=False,
        session_verdict=lambda _token: None,
        resolved_topology=Topology(world=2, dp=2),
        resolved_attention="TORCH_FLASH",
    )

    assert first.residency_mode == "required"
    assert second.residency_mode == "stock"


def test_normal_render_ledger_read_error_never_uses_cached_pass(
    monkeypatch, tmp_path,
):
    def unreadable(_context):
        raise ledger_mod.GateLedgerReadError("permission denied")

    _model, _handle, authorization = _normal_authorization_rig(
        monkeypatch,
        tmp_path,
        unreadable,
        session_verdict=lambda _token: "PASS",
    )

    assert authorization.residency_mode == "stock"
    assert authorization.residency_grant is None


def test_normal_render_unscoped_damage_never_uses_cached_pass(
    monkeypatch, tmp_path,
):
    _model, _handle, authorization = _normal_authorization_rig(
        monkeypatch,
        tmp_path,
        lambda _context: ("unknown", None),
        session_verdict=lambda _token: "PASS",
        session_pass_safe=False,
    )

    assert authorization.residency_mode == "stock"
    assert authorization.residency_grant is None
    assert authorization.worker_args["slab_weights"] is False
    assert authorization.worker_args["lora_low_rss"] is False


def test_auto_gate_ignores_cached_pass_when_ledger_damage_is_unhealed(
    monkeypatch,
):
    token = ("combo", "artifacts", "commit", "context")
    states = iter((("error", token), ("error", token), ("pass", token)))
    calls = []
    monkeypatch.setattr(common, "_auto_gate_context", lambda *_args: next(states))
    monkeypatch.setattr(
        gate_mod,
        "run_identity_ceremony",
        lambda *_args, **_kwargs: calls.append(True) or {
            "verdict": "PASS",
            "_gate_token": token,
            "_gate_tokens": [token],
        },
    )
    common._AUTO_GATE_SESSION[token] = "PASS"
    common._AUTO_GATE_ACTIVE.on = False

    assert common.auto_gate_required(SimpleNamespace()) is True
    result = common._maybe_auto_gate(
        SimpleNamespace(), {"kind": "ksampler", "steps": 2}, {}, 1.0, 2)

    assert result == "PASS"
    assert calls == [True]
    assert common._AUTO_GATE_SESSION[token] == "PASS"


def test_auto_gate_does_not_cache_unconsumable_pass_over_retesting(
    monkeypatch,
):
    token = ("combo", "artifacts", "commit", "context")
    calls = []
    monkeypatch.setattr(
        common, "_auto_gate_context", lambda *_args: ("inconclusive", token))
    monkeypatch.setattr(
        gate_mod,
        "run_identity_ceremony",
        lambda *_args, **_kwargs: calls.append(True) or {
            "verdict": "PASS",
            "_gate_token": token,
            "_gate_tokens": [token],
        },
    )
    common._AUTO_GATE_ACTIVE.on = False

    request = {"kind": "ksampler", "steps": 2}
    assert common._maybe_auto_gate(SimpleNamespace(), request, {}, 1.0, 2) == "ERROR"
    assert common._maybe_auto_gate(SimpleNamespace(), request, {}, 1.0, 2) == "ERROR"
    assert calls == [True]
    assert common._process_gate_verdict(token) == "ERROR"


def test_manual_gate_delegates_to_canonical_engine_and_enforces_strict(monkeypatch):
    calls = []
    stock = {"samples": torch.zeros(1)}

    def ceremony(model, request, latent, cfg_value, steps_hint, origin, run_id=""):
        calls.append((origin, run_id, request["noise_seed"]))
        return {
            "verdict": "PASS",
            "max_abs_latent_diff": 0.0,
            "run_id": run_id,
            "latent": stock,
        }

    monkeypatch.setattr(gate_mod, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(gate_mod, "run_identity_ceremony", ceremony)
    node = gate_mod.DGXMonarchIdentityGate()
    latent, report = node.gate(
        object(), [], [], {"samples": torch.ones(1)}, 17, 2, 1.0,
        "euler", "simple", run_id="cli-17",
    )
    assert latent is stock
    assert json.loads(report)["run_id"] == "cli-17"
    assert calls == [("explicit", "cli-17", 17)]

    def failed(*args, **kwargs):
        return {"verdict": "FAIL", "max_abs_latent_diff": 1.0, "latent": stock}

    monkeypatch.setattr(gate_mod, "run_identity_ceremony", failed)
    with pytest.raises(RuntimeError, match="identity gate FAIL"):
        node.gate(
            object(), [], [], {"samples": torch.ones(1)}, 18, 2, 1.0,
            "euler", "simple", strict=True,
        )


def test_identity_ceremony_rejects_dual_model_without_recording_a_pass():
    model = SimpleNamespace(mesh=SimpleNamespace(worker_args={
        "lora_low_rss": True, "slab_weights": True}))
    with pytest.raises(ValueError, match="proves one model only"):
        gate_mod.run_identity_ceremony(
            model,
            {"uncond_model": {"unet_name": "negative.sft", "loras": []}},
            {"samples": torch.zeros(1)}, 1.0, 2, "explicit",
        )
    assert model.mesh.worker_args == {"lora_low_rss": True, "slab_weights": True}


def test_identity_ceremony_owns_session_across_bound_helper(monkeypatch):
    handle = object.__new__(MeshHandle)
    handle.lock = threading.RLock()
    model = SimpleNamespace(mesh=SimpleNamespace(handle=handle))
    contender = RenderSession(timeout_s=0)
    expected = {"verdict": "PASS"}

    monkeypatch.setattr(gate_mod, "ensure_live", lambda value: value)

    def bound_helper(*args, **kwargs):
        with pytest.raises(
            ConcurrentRenderSessionError, match="another render session"
        ):
            contender.bind(handle)
        return expected

    monkeypatch.setattr(gate_mod, "_run_identity_ceremony_bound", bound_helper)

    result = gate_mod.run_identity_ceremony(
        model, {}, {"samples": torch.zeros(1)}, 1.0, 2, "explicit"
    )
    assert result is expected

    contender.bind(handle)
    contender.close()


def test_canonical_failure_quarantines_and_persists(monkeypatch, tmp_path):
    class Handle:
        setup_generation = 4
        setup_key = ("setup",)
        worker_args_key = ("worker",)
        world = 1

        def __init__(self):
            self.calls = []

        def call_all(self, method, *args, **kwargs):
            self.calls.append((method, args))
            if method == "provenance_baseline":
                return [{
                    "rank": 0,
                    "world": 1,
                    "setup_generation": args[0],
                    "source_manifest_sha256": (
                        runtime_provenance.cached_dgx_source_manifest_sha256()
                    ),
                }]
            if method == "gate_swap_cycle":
                return [{"rank": 0, "world": 1, "setup_generation": 4,
                         "conclusive": True, "transitions": ["lazy"],
                         "family": "unvouched", "slab_active": False}]
            return [{"ok": True}]

    class Ledger:
        def __init__(self):
            self.records = []
            self.retests = []

        def begin_retest_required(self, *args):
            self.retests.append(args)

        def record(self, *args):
            self.records.append(args)

    handle = Handle()
    ledger = Ledger()
    worker_args = {"lora_low_rss": True}
    model = SimpleNamespace(
        unet_name="model.safetensors",
        options={},
        loras=({"name": "test.safetensors", "strength": 1.0},),
        mesh=SimpleNamespace(handle=handle, worker_args=worker_args),
    )
    renders = iter([{"samples": torch.zeros(1)}, {"samples": torch.ones(1)}])

    def render(*args, **kwargs):
        handle.calls.append(("render", ()))
        return next(renders)

    monkeypatch.setattr(gate_mod, "run_render", render)
    monkeypatch.setattr(gate_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        gate_mod,
        "render_setup_token",
        lambda *_args: SetupToken(4, ("setup",), ("worker",)),
    )
    monkeypatch.setattr(gate_mod, "_combo_of", lambda value: ("combo", "artifacts"))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(gate_mod, "GateLedger", lambda directory: ledger)
    monkeypatch.setattr(model_store_mod, "request_artifact_identity", _identity)
    monkeypatch.setattr("dgx_monarch.telemetry.emit", lambda *args, **kwargs: None)

    result = gate_mod.run_identity_ceremony(
        model, {"noise_seed": 1, "steps": 2, "cfg": 1.0},
        {"samples": torch.zeros(1)}, 1.0, 2, "explicit", run_id="run-1",
    )
    assert result["verdict"] == "FAIL"
    assert worker_args["lora_low_rss"] is False
    assert [call[0] for call in handle.calls] == [
        "provenance_baseline", "unload", "render", "gate_swap_cycle", "render",
        "provenance_baseline", "apply_worker_args",
    ]
    assert ledger.records[0][3] == "FAIL"
    # The swap lineage diverged and slab was never resident, so the cross
    # leg never ran. Only the normal and Fleet contexts of the rendered policy
    # get rows; the explicit-slab siblings stay covered by the one preflight
    # RETESTING row, not a sticky FAIL for a capability no leg tested.
    assert [record[3] for record in ledger.records] == ["FAIL", "FAIL"]
    assert not any(record[4].get("stamped") for record in ledger.records)
    assert all(
        record[5]["worker_args"].get("slab_weights") is not True
        for record in ledger.records
    )
    report = json.loads((tmp_path / "dgxm_gate_reports.jsonl").read_text())
    assert report["run_id"] == "run-1" and report["verdict"] == "FAIL"


def test_ceremony_freezes_nested_request_and_latent_for_every_render(
    monkeypatch, tmp_path,
):
    class Handle:
        world = 1
        setup_generation = 4
        setup_key = ("setup",)
        worker_args_key = ("worker",)

        def call_all(self, method, *args, **kwargs):
            if method == "provenance_baseline":
                return [{
                    "rank": 0,
                    "world": 1,
                    "setup_generation": args[0],
                    "source_manifest_sha256": (
                        runtime_provenance.cached_dgx_source_manifest_sha256()
                    ),
                }]
            if method == "gate_swap_cycle":
                return [{"rank": 0, "world": 1, "setup_generation": 4,
                         "conclusive": True, "transitions": ["stock"],
                         "family": "unvouched", "slab_active": False}]
            return [{}]

    handle = Handle()
    model = SimpleNamespace(
        unet_name="model.safetensors",
        options={},
        loras=(),
        mesh=SimpleNamespace(handle=handle, worker_args={
            "lora_low_rss": False, "slab_weights": False}),
    )
    request = {
        "noise_seed": 7,
        "steps": 2,
        "cfg": 1.0,
        "nested": {"values": [1, 2]},
    }
    latent = {"samples": torch.tensor([3.0]), "nested": {"values": [4]}}
    seen = []

    def render(_model, render_request, render_latent, **_kwargs):
        seen.append((
            list(render_request["nested"]["values"]),
            render_latent["samples"].clone(),
            list(render_latent["nested"]["values"]),
        ))
        render_request["nested"]["values"].append(99)
        render_latent["nested"]["values"].append(99)
        if len(seen) == 1:
            # Mutating the caller's latent after capture must not change the
            # gate's frozen copy that later legs render from.
            latent["samples"].add_(100)
        return {"samples": torch.zeros(1)}

    monkeypatch.setattr(gate_mod, "run_render", render)
    monkeypatch.setattr(gate_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        gate_mod,
        "render_setup_token",
        lambda *_args: SetupToken(4, ("setup",), ("worker",)),
    )
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(model_store_mod, "request_artifact_identity", _identity)
    monkeypatch.setattr("dgx_monarch.telemetry.emit", lambda *args, **kwargs: None)

    result = gate_mod.run_identity_ceremony(
        model, request, latent, 1.0, 2, "explicit")

    # This fixture has no LoRA, slab or FSDP reload lineage to prove, so the
    # ceremony grants nothing and reads INCONCLUSIVE.
    assert result["verdict"] == "INCONCLUSIVE"
    assert len(seen) == 2
    assert all(values == [1, 2] for values, _samples, _nested in seen)
    assert all(torch.equal(samples, torch.tensor([3.0]))
               for _values, samples, _nested in seen)
    assert all(nested == [4] for _values, _samples, nested in seen)
    assert request["nested"]["values"] == [1, 2]
    assert torch.equal(latent["samples"], torch.tensor([103.0]))
    assert latent["nested"]["values"] == [4]


def test_transaction_snapshot_shares_tensor_baseline_but_isolates_metadata():
    from dgx_monarch.nodes.gate_identity import copy_transaction, freeze_transaction

    source = {
        "samples": torch.zeros(2),
        "nested": {"values": [1, 2]},
    }
    frozen = freeze_transaction(source)

    assert frozen is not source
    assert frozen["nested"] is not source["nested"]
    assert frozen["nested"]["values"] is not source["nested"]["values"]
    assert frozen["samples"].data_ptr() != source["samples"].data_ptr()
    leg = copy_transaction(frozen)
    assert leg["samples"].data_ptr() == frozen["samples"].data_ptr()


def test_transaction_snapshot_clones_and_shares_every_packed_modality(
    monkeypatch,
):
    from dgx_monarch.nodes.gate_identity import copy_transaction, freeze_transaction

    NestedTensor = _install_nested_tensor(monkeypatch)
    source_packed = _nested(NestedTensor, video=1.0, audio=2.0)
    source = {"samples": source_packed, "metadata": ["original"]}
    source["cycle"] = source

    frozen = freeze_transaction(source)
    frozen_parts = frozen["samples"].unbind()
    source_parts = source_packed.unbind()

    assert frozen["cycle"] is frozen
    assert frozen["samples"] is not source_packed
    assert frozen["samples"].tensors is not source_packed.tensors
    assert all(frozen_part.data_ptr() != source_part.data_ptr()
               for frozen_part, source_part in zip(
                   frozen_parts, source_parts, strict=True))

    leg = copy_transaction(frozen)
    leg_parts = leg["samples"].unbind()
    assert leg["samples"] is not frozen["samples"]
    assert leg["samples"].tensors is not frozen["samples"].tensors
    assert all(leg_part.data_ptr() == frozen_part.data_ptr()
               for leg_part, frozen_part in zip(
                   leg_parts, frozen_parts, strict=True))
    leg["samples"].tensors.pop()
    leg["metadata"].append("leg-only")
    assert len(frozen["samples"].unbind()) == 2
    assert frozen["metadata"] == ["original"]


@pytest.mark.parametrize("modality", [0, 1])
def test_transaction_guard_rejects_mutation_of_either_packed_modality(
    monkeypatch, modality,
):
    from dgx_monarch.nodes.gate_identity import (
        copy_transaction,
        freeze_transaction,
        require_transaction_unchanged,
        transaction_tensor_versions,
    )

    NestedTensor = _install_nested_tensor(monkeypatch)
    frozen = freeze_transaction({"samples": _nested(NestedTensor)})
    versions = transaction_tensor_versions(frozen)
    copy_transaction(frozen)["samples"].unbind()[modality].add_(1)

    with pytest.raises(RuntimeError, match="mutated a captured"):
        require_transaction_unchanged(versions)


def _mutate_tensor_without_version(tensor, surface):
    version = int(getattr(tensor, "_version", 0))
    if surface == "data":
        tensor.data.add_(1)
    elif surface == "storage":
        storage = tensor.untyped_storage()
        storage[0] = 1
    else:
        tensor.detach().numpy().reshape(-1)[0] = 1
    assert int(getattr(tensor, "_version", 0)) == version


@pytest.mark.parametrize("surface", ["data", "storage", "numpy"])
@pytest.mark.parametrize("modality", [0, 1])
def test_transaction_guard_rejects_unversioned_packed_modality_mutation(
    monkeypatch, surface, modality,
):
    from dgx_monarch.nodes.gate_identity import (
        copy_transaction,
        freeze_transaction,
        require_transaction_unchanged,
        transaction_tensor_versions,
    )

    NestedTensor = _install_nested_tensor(monkeypatch)
    frozen = freeze_transaction({"samples": _nested(NestedTensor)})
    versions = transaction_tensor_versions(frozen)
    target = copy_transaction(frozen)["samples"].unbind()[modality]
    _mutate_tensor_without_version(target, surface)

    with pytest.raises(RuntimeError, match="mutated a captured"):
        require_transaction_unchanged(versions)


def test_transaction_snapshot_rejects_malformed_or_derived_packed_wrapper(
    monkeypatch,
):
    from dgx_monarch.nodes.gate_identity import freeze_transaction

    NestedTensor = _install_nested_tensor(monkeypatch)

    class DerivedNestedTensor(NestedTensor):
        pass

    with pytest.raises(TypeError, match="at least one modality"):
        freeze_transaction({"samples": NestedTensor(())})
    with pytest.raises(TypeError, match="subclasses"):
        freeze_transaction({"samples": DerivedNestedTensor((torch.zeros(1),))})


def test_transaction_snapshot_and_fingerprint_reject_tensor_subclasses(
    monkeypatch,
):
    from dgx_monarch.nodes.gate_identity import freeze_transaction
    from dgx_monarch.nodes.gate_tensor_guard import transaction_tensor_fingerprint

    class LyingTensor(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if func is torch.equal:
                return True
            return super().__torch_function__(func, types, args, kwargs or {})

    lying = torch.zeros(1).as_subclass(LyingTensor)
    NestedTensor = _install_nested_tensor(monkeypatch)

    with pytest.raises(TypeError, match="Tensor subclasses"):
        freeze_transaction({"samples": lying})
    with pytest.raises(TypeError, match="must all be tensors"):
        freeze_transaction(
            {"samples": NestedTensor((lying, torch.zeros(1)))}
        )
    with pytest.raises(TypeError, match=r"exact torch.Tensor"):
        transaction_tensor_fingerprint(lying)


def test_transaction_guard_rejects_in_place_tensor_mutation():
    from dgx_monarch.nodes.gate_identity import (
        copy_transaction,
        freeze_transaction,
        require_transaction_unchanged,
        transaction_tensor_versions,
    )

    frozen = freeze_transaction({"samples": torch.zeros(2)})
    versions = transaction_tensor_versions(frozen)
    copy_transaction(frozen)["samples"].add_(1)

    with pytest.raises(RuntimeError, match="mutated a captured"):
        require_transaction_unchanged(versions)


@pytest.mark.parametrize("surface", ["data", "storage", "numpy"])
def test_transaction_guard_rejects_unversioned_plain_tensor_mutation(surface):
    from dgx_monarch.nodes.gate_identity import (
        copy_transaction,
        freeze_transaction,
        require_transaction_unchanged,
        transaction_tensor_versions,
    )

    frozen = freeze_transaction({"samples": torch.zeros(2)})
    versions = transaction_tensor_versions(frozen)
    target = copy_transaction(frozen)["samples"]
    _mutate_tensor_without_version(target, surface)

    with pytest.raises(RuntimeError, match="mutated a captured"):
        require_transaction_unchanged(versions)


def test_transaction_guard_rejects_zero_numel_storage_replacement():
    from dgx_monarch.nodes.gate_identity import (
        copy_transaction,
        freeze_transaction,
        require_transaction_unchanged,
        transaction_tensor_versions,
    )

    frozen = freeze_transaction({"samples": torch.empty(0)})
    versions = transaction_tensor_versions(frozen)
    target = copy_transaction(frozen)["samples"]
    version = int(getattr(target, "_version", 0))
    old_storage = int(target.untyped_storage()._cdata)
    target.data = torch.empty_like(target)

    assert int(getattr(target, "_version", 0)) == version
    assert int(target.untyped_storage()._cdata) != old_storage
    with pytest.raises(RuntimeError, match="mutated a captured"):
        require_transaction_unchanged(versions)


def test_transaction_baseline_is_versioned_under_inference_mode():
    from dgx_monarch.nodes.gate_identity import (
        freeze_transaction,
        transaction_tensor_versions,
    )

    with torch.inference_mode():
        source = {"samples": torch.ones(1)}
        frozen = freeze_transaction(source)

    assert frozen["samples"].data_ptr() != source["samples"].data_ptr()
    assert not frozen["samples"].is_inference()
    versions = transaction_tensor_versions(frozen)
    assert isinstance(versions[0][1], int)
    with torch.inference_mode():
        source["samples"].fill_(7)
    assert torch.equal(frozen["samples"], torch.ones(1))


def test_inconclusive_auto_gate_retries_only_after_context_changes(monkeypatch):
    token = {"value": ("combo", "args-a")}
    calls = []
    model = SimpleNamespace()
    monkeypatch.setattr(
        common, "_auto_gate_context",
        lambda model, kind, *_args: ("inconclusive", token["value"]),
    )
    monkeypatch.setattr(
        gate_mod, "run_identity_ceremony",
        lambda *args, **kwargs: calls.append(kwargs.get("origin")) or
        {"verdict": "INCONCLUSIVE"},
    )
    common._AUTO_GATE_ACTIVE.on = False
    common._AUTO_GATE_SESSION.clear()
    request = {"kind": "ksampler", "steps": 4}
    common._maybe_auto_gate(model, request, {}, 1.0, 4)
    common._maybe_auto_gate(model, request, {}, 1.0, 4)
    assert calls == ["auto_first_use"]
    assert common.auto_gate_required(model)

    token["value"] = ("combo", "args-b")
    common._maybe_auto_gate(model, request, {}, 1.0, 4)
    assert calls == ["auto_first_use", "auto_first_use"]
    common._AUTO_GATE_SESSION.clear()


def test_unknown_commit_pass_is_cached_only_for_this_process(monkeypatch):
    calls = []
    token = ("combo", "artifacts", "unknown", "args")
    model = SimpleNamespace()
    monkeypatch.setattr(
        common, "_auto_gate_context", lambda model, kind, *_args: ("stale", token),
    )
    monkeypatch.setattr(
        gate_mod, "run_identity_ceremony",
        lambda *args, **kwargs: calls.append(1) or
        {"verdict": "PASS", "_gate_token": token},
    )
    common._AUTO_GATE_ACTIVE.on = False
    common._AUTO_GATE_SESSION.clear()
    request = {"kind": "ksampler", "steps": 2}
    common._maybe_auto_gate(model, request, {}, 1.0, 2)
    common._maybe_auto_gate(model, request, {}, 1.0, 2)
    assert calls == [1]
    assert common._AUTO_GATE_SESSION[token] == "PASS"
    assert not common.auto_gate_required(model)
    common._AUTO_GATE_SESSION.clear()


def test_auto_gate_discards_pass_when_ceremony_snapshot_changed(monkeypatch):
    old = ("combo", "artifact-old", "commit", "context")
    new = ("combo", "artifact-new", "commit", "context")
    contexts = iter((("unknown", old), ("pass", new)))
    model = SimpleNamespace(mesh=SimpleNamespace(worker_args={
        "lora_low_rss": True, "slab_weights": True}))
    monkeypatch.setattr(common, "_auto_gate_context", lambda *_args: next(contexts))
    monkeypatch.setattr(
        gate_mod, "run_identity_ceremony",
        lambda *_args, **_kwargs: {"verdict": "PASS", "_gate_token": new},
    )

    class Pending:
        def result(self):
            return {"samples": torch.zeros(1)}

    def submit(*_args, **_kwargs):
        assert model.mesh.worker_args == {
            "lora_low_rss": False, "slab_weights": False}
        return Pending()

    monkeypatch.setattr(common, "submit_render", submit)
    common._AUTO_GATE_SESSION.clear()
    common._AUTO_GATE_RUNNING.clear()
    common.run_render(model, {"kind": "ksampler", "steps": 2}, {}, 1.0, 2)
    # The ceremony's token no longer names this dispatch, so its PASS is
    # discarded and the claimed token gets a process-local ERROR denial.
    assert common._AUTO_GATE_SESSION == {old: "ERROR"}
    assert gate_process_state._PROCESS_GATE_DENIALS == {old: "ERROR"}


def test_dual_model_render_never_reuses_primary_only_pass(monkeypatch):
    model = SimpleNamespace(mesh=SimpleNamespace(worker_args={
        "lora_low_rss": True, "slab_weights": True}))
    common._AUTO_GATE_SESSION.clear()
    common._AUTO_GATE_SESSION[("primary-pass",)] = "PASS"
    monkeypatch.setattr(
        common, "_auto_gate_context",
        lambda *_args: pytest.fail("dual-model request consulted primary-only ledger"),
    )

    class Pending:
        def result(self):
            return {"samples": torch.zeros(1)}

    def submit(dispatched_model, *_args, **_kwargs):
        assert dispatched_model is not model
        assert dispatched_model.mesh.worker_args == {
            "lora_low_rss": False, "slab_weights": False}
        assert model.mesh.worker_args == {
            "lora_low_rss": True, "slab_weights": True}
        return Pending()

    monkeypatch.setattr(common, "submit_render", submit)
    request = {
        "kind": "ksampler_advanced", "steps": 2,
        "uncond_model": {
            "unet_name": "negative.sft", "options": {"dtype": "bf16"},
            "loras": [{"name": "negative-style.sft", "strength": 0.5}],
        },
    }

    common.run_render(model, request, {}, 1.0, 2)
    assert model.mesh.worker_args == {"lora_low_rss": True, "slab_weights": True}
    assert common._AUTO_GATE_SESSION == {("primary-pass",): "PASS"}
    common._AUTO_GATE_SESSION.clear()


def test_pipeline_dual_model_bypasses_primary_pass_and_forces_stock(monkeypatch):
    model = SimpleNamespace(mesh=SimpleNamespace(worker_args={
        "lora_low_rss": True, "slab_weights": True}))
    monkeypatch.setattr(common, "auto_gate_required", lambda *_args: False)
    submitted = []

    class Pending:
        def result(self):
            return {"samples": torch.zeros(1)}

    def submit(dispatched_model, *_args, **_kwargs):
        submitted.append(1)
        assert dispatched_model is not model
        assert dispatched_model.mesh.worker_args == {
            "lora_low_rss": False, "slab_weights": False}
        assert model.mesh.worker_args == {
            "lora_low_rss": True, "slab_weights": True}
        return Pending()

    monkeypatch.setattr(common, "submit_render", submit)
    request = {
        "kind": "ksampler_advanced",
        "uncond_model": {
            "unet_name": "negative.sft", "options": {},
            "loras": [{"name": "negative-style.sft", "strength": 0.5}],
        },
    }
    pipeline = common.RenderPipeline(depth=2)
    pipeline.push(model, request, {}, 1.0, 2)

    assert submitted == [1]
    assert pipeline.drain()[0]["samples"].shape == (1,)
    assert model.mesh.worker_args == {"lora_low_rss": True, "slab_weights": True}


def test_auto_gate_session_cache_evicts_oldest_context(monkeypatch):
    token = {"value": ("combo-0",)}
    model = SimpleNamespace()
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_SESSION_LIMIT", 2)
    monkeypatch.setattr(
        common, "_auto_gate_context",
        lambda model, kind, *_args: ("unknown", token["value"]),
    )
    monkeypatch.setattr(
        gate_mod, "run_identity_ceremony",
        lambda *args, **kwargs: {"verdict": "PASS", "_gate_token": token["value"]},
    )
    common._AUTO_GATE_ACTIVE.on = False
    common._AUTO_GATE_SESSION.clear()
    try:
        for index in range(3):
            token["value"] = (f"combo-{index}",)
            common._maybe_auto_gate(
                model, {"kind": "ksampler", "steps": 2}, {}, 1.0, 2,
            )

        assert list(common._AUTO_GATE_SESSION) == [("combo-1",), ("combo-2",)]
    finally:
        common._AUTO_GATE_SESSION.clear()


def test_process_denial_survives_bounded_positive_cache_pressure():
    denied = ("denied",)
    common._record_process_gate_verdicts([denied], "FAIL")

    for index in range(gate_process_state._AUTO_GATE_SESSION_LIMIT + 32):
        common._record_process_gate_verdicts(
            [("passing", str(index))], "PASS")

    assert denied not in common._AUTO_GATE_SESSION
    assert common._process_gate_verdict(denied) == "FAIL"

    # Cache pressure never evicts the denial; a later PASS for this exact
    # token clears it.
    common._record_process_gate_verdicts([denied], "PASS")
    assert common._process_gate_verdict(denied) == "PASS"


@pytest.mark.parametrize("verdict", ["FAIL", "INCONCLUSIVE", "ERROR"])
def test_full_render_is_submitted_only_after_unproven_paths_are_quarantined(
    monkeypatch, verdict,
):
    events = []
    model = SimpleNamespace(mesh=SimpleNamespace(worker_args={
        "lora_low_rss": True, "slab_weights": True}))

    class Pending:
        def result(self):
            events.append("result")
            return {"samples": torch.zeros(1)}

    monkeypatch.setattr(
        common, "_maybe_auto_gate", lambda *args: events.append("gate") or verdict)

    def submit(*args, **kwargs):
        events.append("submit")
        assert model.mesh.worker_args["lora_low_rss"] is False
        assert model.mesh.worker_args["slab_weights"] is False
        return Pending()

    monkeypatch.setattr(common, "submit_render", submit)
    common.run_render(model, {"kind": "ksampler"}, {}, 1.0, 2)
    assert events == ["gate", "submit", "result"]


def test_auto_gate_preserves_cfg_zero(monkeypatch):
    captured = []
    token = ("cfg-zero",)
    model = SimpleNamespace()
    monkeypatch.setattr(
        common, "_auto_gate_context",
        lambda model, kind, *_args: ("unknown", token))
    monkeypatch.setattr(
        gate_mod, "run_identity_ceremony",
        lambda model, request, latent, cfg_value, *args, **kwargs:
        captured.append(cfg_value) or {"verdict": "PASS", "_gate_token": token},
    )
    common._AUTO_GATE_SESSION.clear()
    common._AUTO_GATE_RUNNING.clear()
    assert common._maybe_auto_gate(
        model, {"kind": "ksampler", "steps": 2}, {}, 0.0, 2) == "PASS"
    assert captured == [0.0]
    common._AUTO_GATE_SESSION.clear()


def test_gate_context_merges_cluster_and_init_worker_args():
    handle = SimpleNamespace(
        config=SimpleNamespace(worker_args={"compile_dit": True, "slab_weights": True},
                               hosts=(object(),), source="/configs/cluster.toml"),
        config_fingerprint="config-a",
        world=4, n_hosts=2, gpus_per_host=2,
    )
    model = SimpleNamespace(
        mesh=SimpleNamespace(
            handle=handle,
            worker_args={"slab_weights": False, "lora_low_rss": True},
            topology_preset="uly2", attention="TORCH_FLASH", sync_ulysses=True,
        )
    )
    context = gate_mod.gate_capability_context(model, handle)
    assert context["worker_args"] == {
        "compile_dit": True, "slab_weights": False, "lora_low_rss": True}
    assert (context["world"], context["hosts"], context["gpus_per_host"]) == (4, 2, 2)
    assert (context["mesh_mode"], context["config_source"], context["config_fingerprint"],
            context["topology_preset"], context["attention"], context["sync_ulysses"]) == (
                "cluster", "/configs/cluster.toml", "config-a", "uly2", "TORCH_FLASH", True)


def test_gate_compat_exports_and_effective_policy_monkeypatch(monkeypatch):
    from dgx_monarch import mesh_safety

    assert gate_mod.FLEET_RESIDENCY_CAPABILITY is mesh_safety.FLEET_RESIDENCY_CAPABILITY
    assert gate_mod.physical_capability_context is mesh_safety.physical_capability_context
    handle = SimpleNamespace(
        config=SimpleNamespace(worker_args={}, hosts=(), source=""),
        config_fingerprint="local", world=1, n_hosts=1, gpus_per_host=1,
    )
    model = SimpleNamespace(mesh=SimpleNamespace(
        handle=handle, worker_args={}, topology_preset="single",
        attention="TORCH_FLASH", sync_ulysses=True,
    ))
    monkeypatch.setattr(
        gate_mod, "_effective_worker_args",
        lambda *_args, **_kwargs: {"patched": True},
    )
    assert gate_mod.gate_capability_context(model, handle)["worker_args"] == {
        "patched": True}
    assert gate_mod.fleet_residency_capability_context(
        model, handle)["worker_args"] == {"patched": True}


def test_cross_mode_reference_model_carries_stock_policy_without_mutating_graph():
    mesh = common.MeshSpec(
        handle=object(), topology_preset="uly2", attention="TORCH_FLASH",
        sync_ulysses=True, worker_args={"lora_low_rss": True, "slab_weights": True})
    model = common.ModelSpec(mesh=mesh, unet_name="model.safetensors")
    stock = gate_mod._model_with_worker_overrides(model, {"slab_weights": False})
    assert stock.mesh.worker_args["slab_weights"] is False
    assert model.mesh.worker_args["slab_weights"] is True


def test_low_rss_off_does_not_suppress_slab_gate(monkeypatch, tmp_path):
    import dgx_monarch.gate_ledger as ledger_mod

    handle = SimpleNamespace(
        config=SimpleNamespace(worker_args={"slab_weights": True}),
        world=2, n_hosts=2, gpus_per_host=1,
    )
    model = SimpleNamespace(
        unet_name="model.safetensors", options={}, loras=({"name": "lora"},),
        mesh=SimpleNamespace(
            handle=handle, auto_gate="first_use",
            worker_args={"lora_low_rss": False}, topology_preset="uly2",
            attention="TORCH_FLASH", sync_ulysses=True,
        ),
    )

    class Ledger:
        def __init__(self, directory):
            pass

        def lookup_with_entry(self, *args):
            return "unknown", None

        def lookup_with_integrity(self, *args):
            return ledger_mod.GateLedgerLookup("unknown", None, True)

    monkeypatch.setattr(gate_mod, "_combo_of", lambda model: (
        "combo", SimpleNamespace(current="artifact")))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(common, "ensure_live", lambda value: value)
    monkeypatch.setattr(ledger_mod, "comfy_commit", lambda: "commit")
    monkeypatch.setattr(ledger_mod, "GateLedger", Ledger)
    context = common._auto_gate_context(model, "ksampler_advanced")
    assert context is not None
    first_token = context[1]
    assert first_token[-2:] == (
        str(ledger_mod.GATE_PROTOCOL_VERSION), str(ledger_mod.__version__))

    monkeypatch.setattr(
        ledger_mod, "GATE_PROTOCOL_VERSION", ledger_mod.GATE_PROTOCOL_VERSION + 1)
    changed = common._auto_gate_context(model, "ksampler_advanced")
    assert changed is not None and changed[1] != first_token

    model.loras = ()
    assert common._auto_gate_context(model, "ksampler") is not None

    handle.config.worker_args["slab_weights"] = False
    assert common._auto_gate_context(model, "ksampler") is None


def test_auto_gate_carries_known_fail_before_any_later_ledger_read(
    monkeypatch, tmp_path,
):
    import dgx_monarch.gate_ledger as ledger_mod

    handle = SimpleNamespace(
        config=SimpleNamespace(worker_args={}), world=1, n_hosts=1,
        gpus_per_host=1)
    model = SimpleNamespace(
        unet_name="model.safetensors", options={},
        loras=({"name": "lora.safetensors"},),
        mesh=SimpleNamespace(
            handle=handle, auto_gate="first_use",
            worker_args={"lora_low_rss": True, "slab_weights": True},
            topology_preset="single", attention="TORCH_FLASH",
            sync_ulysses=False),
    )
    lookups = []

    class Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_entry(self, *_args):
            lookups.append(True)
            if len(lookups) == 1:
                return "fail", {
                    "verdict": "FAIL",
                    "quarantine_levers": ["lora_low_rss", "slab_weights"],
                }
            return "unknown", None

        def lookup_with_integrity(self, *_args):
            state, entry = self.lookup_with_entry()
            return ledger_mod.GateLedgerLookup(state, entry, True)

    monkeypatch.setattr(gate_mod, "_combo_of", lambda _model: (
        "combo", SimpleNamespace(current="artifact")))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(common, "ensure_live", lambda value: value)
    monkeypatch.setattr(ledger_mod, "comfy_commit", lambda: "commit")
    monkeypatch.setattr(ledger_mod, "GateLedger", Ledger)

    state, _token = common._auto_gate_context(model, "ksampler")
    quarantine_mod._enforce_persisted_quarantine(model, handle)

    assert state == "fail"
    assert lookups == [True]
    assert model.mesh.worker_args == {
        "lora_low_rss": False, "slab_weights": False}


@pytest.mark.parametrize(
    ("ceremony_succeeds", "expected_required"),
    [(True, False), (False, True)],
)
def test_auto_gate_required_waits_for_running_ceremony(
    monkeypatch, ceremony_succeeds, expected_required,
):
    token = ("combo", "artifacts", "commit", "args")
    model = SimpleNamespace()
    ceremony_started = threading.Event()
    release_ceremony = threading.Event()
    checker_started = threading.Event()
    required = []

    monkeypatch.setattr(
        common, "_auto_gate_context", lambda model, kind, *_args: ("unknown", token),
    )

    def ceremony(*args, **kwargs):
        ceremony_started.set()
        release_ceremony.wait(timeout=2)
        if not ceremony_succeeds:
            raise RuntimeError("ceremony failed")
        return {"verdict": "PASS", "_gate_token": token}

    monkeypatch.setattr(gate_mod, "run_identity_ceremony", ceremony)
    common._AUTO_GATE_SESSION.clear()
    common._AUTO_GATE_RUNNING.clear()

    gate_thread = threading.Thread(
        target=common._maybe_auto_gate,
        args=(model, {"kind": "ksampler", "steps": 2}, {}, 1.0, 2),
    )

    def check_required():
        checker_started.set()
        required.append(common.auto_gate_required(model))

    check_thread = threading.Thread(target=check_required)
    try:
        gate_thread.start()
        assert ceremony_started.wait(timeout=1)
        check_thread.start()
        assert checker_started.wait(timeout=1)
        check_thread.join(timeout=0.05)
        assert check_thread.is_alive()

        release_ceremony.set()
        gate_thread.join(timeout=1)
        check_thread.join(timeout=1)
        assert not gate_thread.is_alive()
        assert not check_thread.is_alive()
        assert required == [expected_required]
    finally:
        release_ceremony.set()
        gate_thread.join(timeout=1)
        if check_thread.ident is not None:
            check_thread.join(timeout=1)
        with common._AUTO_GATE_CONDITION:
            common._AUTO_GATE_SESSION.clear()
            common._AUTO_GATE_RUNNING.clear()
            common._AUTO_GATE_CONDITION.notify_all()


def test_persisted_fail_quarantines_fresh_model_before_setup(monkeypatch, tmp_path):
    import dgx_monarch.gate_ledger as ledger_mod

    model = SimpleNamespace(
        unet_name="model.safetensors",
        loras=({"name": "lora.safetensors"},),
        mesh=SimpleNamespace(worker_args={"lora_low_rss": True}),
    )
    monkeypatch.setattr(gate_mod, "_combo_of", lambda _model: ("combo", "artifacts"))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(ledger_mod, "comfy_commit", lambda: "commit")

    class Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_entry(self, *args):
            # A FAIL with no entry, or an entry with no lever list, still
            # turns every guarded residency path off.
            return "fail", None

    monkeypatch.setattr(ledger_mod, "GateLedger", Ledger)
    common._AUTO_GATE_ACTIVE.on = False
    quarantine_mod._enforce_persisted_quarantine(model)
    assert model.mesh.worker_args["lora_low_rss"] is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("slab_weights", {"lora_low_rss": False, "slab_weights": False}),
        (["unknown"], {"lora_low_rss": False, "slab_weights": False}),
        ([[]], {"lora_low_rss": False, "slab_weights": False}),
        ([{}], {"lora_low_rss": False, "slab_weights": False}),
        ([], {"lora_low_rss": False, "slab_weights": False}),
        (["slab_weights"], {"lora_low_rss": False, "slab_weights": False}),
    ],
)
def test_persisted_fail_lever_metadata_is_diagnostic_and_fails_closed(raw, expected):
    model = SimpleNamespace(mesh=SimpleNamespace(worker_args={
        "lora_low_rss": True, "slab_weights": True}))

    common._apply_persisted_quarantine(
        model, {"verdict": "FAIL", "quarantine_levers": raw})

    assert model.mesh.worker_args == expected


def test_persisted_fail_lookup_uses_resolved_replacement_handle(
        monkeypatch, tmp_path):
    import dgx_monarch.gate_ledger as ledger_mod

    stale, replacement = object(), object()
    model = SimpleNamespace(
        unet_name="model.safetensors",
        loras=({"name": "lora.safetensors"},),
        mesh=SimpleNamespace(
            handle=stale, worker_args={"lora_low_rss": True}),
    )
    monkeypatch.setattr(gate_mod, "_combo_of", lambda _model: ("combo", "artifacts"))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(ledger_mod, "comfy_commit", lambda: "commit")
    monkeypatch.setattr(
        gate_mod, "_effective_worker_args",
        lambda _model, handle: {
            "lora_low_rss": True, "handle": id(handle)},
    )
    monkeypatch.setattr(
        gate_mod, "gate_capability_context",
        lambda _model, handle, **_kwargs: {"handle": id(handle)},
    )
    seen = []

    class Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_entry(self, _key, _artifacts, _commit, context):
            seen.append(context)
            return (
                ("fail", {"verdict": "FAIL", "quarantine_levers": ["lora_low_rss"]})
                if context == {"handle": id(replacement)}
                else ("unknown", None)
            )

    monkeypatch.setattr(ledger_mod, "GateLedger", Ledger)
    common._AUTO_GATE_ACTIVE.on = False

    quarantine_mod._enforce_persisted_quarantine(model, replacement)

    assert seen == [{"handle": id(replacement)}]
    assert model.mesh.worker_args["lora_low_rss"] is False


def test_persisted_quarantine_lookup_error_forces_both_paths_off(
    monkeypatch, tmp_path,
):
    import dgx_monarch.gate_ledger as ledger_mod

    model = SimpleNamespace(
        unet_name="model.safetensors",
        loras=({"name": "lora.safetensors"},),
        mesh=SimpleNamespace(worker_args={
            "lora_low_rss": True, "slab_weights": True}),
    )
    monkeypatch.setattr(gate_mod, "_combo_of", lambda _model: (
        "combo", "artifacts"))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(ledger_mod, "comfy_commit", lambda: "commit")

    class Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_entry(self, *_args):
            raise OSError("second ledger read failed")

    monkeypatch.setattr(ledger_mod, "GateLedger", Ledger)
    common._AUTO_GATE_ACTIVE.on = False

    quarantine_mod._enforce_persisted_quarantine(model)

    assert model.mesh.worker_args == {
        "lora_low_rss": False, "slab_weights": False}


def test_persisted_fail_forces_all_risky_levers_off_via_real_ledger(
    monkeypatch, tmp_path
):
    """With the real GateLedger, the FAIL verdict and its lever list come from
    one read, and the lever list is diagnostic: naming only slab_weights never
    grants the other risky residency path."""
    from dgx_monarch.gate_ledger import GateLedger

    worker_args = {"lora_low_rss": True, "slab_weights": True}
    model = SimpleNamespace(
        unet_name="model.safetensors",
        loras=({"name": "lora.safetensors"},),
        mesh=SimpleNamespace(worker_args=dict(worker_args)),
    )
    monkeypatch.setattr(gate_mod, "_combo_of", lambda _model: ("combo", "artifacts"))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(ledger_mod, "comfy_commit", lambda: "commit")
    context = {"worker_args": dict(model.mesh.worker_args)}
    GateLedger(str(tmp_path)).record(
        "combo", "artifacts", "commit", "FAIL",
        detail={"quarantine_levers": ["slab_weights"]}, context=context)

    common._AUTO_GATE_ACTIVE.on = False
    quarantine_mod._enforce_persisted_quarantine(model)
    assert model.mesh.worker_args["slab_weights"] is False
    assert model.mesh.worker_args["lora_low_rss"] is False


def test_persisted_quarantine_skips_when_fail_was_superseded(monkeypatch, tmp_path):
    """A newer non-FAIL row moves state and entry together: no enforcement,
    and never the FAIL's levers paired with the newer row."""
    from dgx_monarch.gate_ledger import GateLedger

    model = SimpleNamespace(
        unet_name="model.safetensors",
        loras=({"name": "lora.safetensors"},),
        mesh=SimpleNamespace(worker_args={"lora_low_rss": True, "slab_weights": True}),
    )
    monkeypatch.setattr(gate_mod, "_combo_of", lambda _model: ("combo", "artifacts"))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(ledger_mod, "comfy_commit", lambda: "commit")
    context = {"worker_args": dict(model.mesh.worker_args)}
    led = GateLedger(str(tmp_path))
    led.record("combo", "artifacts", "commit", "FAIL",
               detail={"quarantine_levers": ["slab_weights"]}, context=context)
    led.record("combo", "artifacts", "commit", "INCONCLUSIVE", context=context)

    common._AUTO_GATE_ACTIVE.on = False
    quarantine_mod._enforce_persisted_quarantine(model)
    assert model.mesh.worker_args["slab_weights"] is True
    assert model.mesh.worker_args["lora_low_rss"] is True


def test_progress_entry_failure_finishes_telemetry(monkeypatch):
    import dgx_monarch.telemetry as telemetry

    events = []

    class Progress:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise OSError("bind failed")

    class Topology:
        dp = 1

        def describe(self):
            return "test"

    handle = SimpleNamespace(ensure_setup=lambda *args, **kwargs: None)
    mesh = SimpleNamespace(
        handle=handle, attention="auto", sync_ulysses=False, worker_args={},
    )
    model = SimpleNamespace(mesh=mesh, unet_name="model", loras=(), request_dict=lambda: {})
    monkeypatch.setattr(common, "ensure_live", lambda value: value)
    monkeypatch.setattr(common, "resolve_topology", lambda *args: (Topology(), False, "test"))
    monkeypatch.setattr(submit_mod, "ProgressReceiver", Progress)
    monkeypatch.setattr(
        telemetry.render_progress, "start",
        lambda *args, **kwargs: events.append("start"))
    monkeypatch.setattr(
        telemetry.render_progress, "finish",
        lambda *_args: events.append("finish"))
    with pytest.raises(OSError, match="bind failed"):
        common.submit_render(
            model, {}, {"samples": torch.zeros(1)}, cfg_value=1.0, steps_hint=2,
        )
    assert events == ["start", "finish"]


@pytest.mark.parametrize(
    ("gate_requirements", "expected_depth"),
    [([True, True], 1), ([True, False], 3)],
    ids=["inconclusive-stays-sequential", "pass-restores-configured-depth"],
)
def test_pipeline_gate_verdict_controls_remaining_depth(
    monkeypatch, gate_requirements, expected_depth,
):
    sequential = []
    submitted = []
    depths = []
    requirements = iter(gate_requirements)

    def run_render(model, request, latent, cfg_value, steps_hint):
        sequential.append(request["noise_seed"])
        return {"samples": torch.tensor([[[[request["noise_seed"]]]]], dtype=torch.float32)}

    class Pipeline:
        def __init__(self, depth, on_step=None):
            depths.append(depth)
            self.requests = []

        def push(self, model, request, latent, cfg_value, steps_hint):
            submitted.append(request["noise_seed"])
            self.requests.append(request)

        def drain(self):
            return [
                {"samples": torch.tensor([[[[request["noise_seed"]]]]], dtype=torch.float32)}
                for request in self.requests
            ]

    monkeypatch.setattr(samplers, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(samplers, "auto_gate_required", lambda *args: next(requirements))
    monkeypatch.setattr(samplers, "run_render", run_render)
    monkeypatch.setattr(samplers, "RenderPipeline", Pipeline)
    model = SimpleNamespace(mesh=SimpleNamespace(pipeline_depth=3))
    latent = {"samples": torch.zeros(1, 1, 1, 1)}
    out, = samplers.DGXMonarchKSamplerPipeline().sample(
        model, "10,11,12", 2, 1.0, "euler", "simple", [], [], latent,
    )
    assert sequential == [10]
    assert submitted == [11, 12]
    assert depths == [expected_depth]
    assert out["samples"].flatten().tolist() == [10.0, 11.0, 12.0]


def _install_nested_tensor(monkeypatch):
    class NestedTensor:
        def __init__(self, tensors):
            self.tensors = list(tensors)
            self.is_nested = True

        @property
        def shape(self):
            raise AssertionError("the gate must compare every packed modality")

        def unbind(self):
            return self.tensors

    NestedTensor.__module__ = "comfy.nested_tensor"
    comfy = type(sys)("comfy")
    nested = type(sys)("comfy.nested_tensor")
    nested.NestedTensor = NestedTensor
    comfy.nested_tensor = nested
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", nested)
    return NestedTensor


def _nested(NestedTensor, *, video=0.0, audio=0.0):
    return NestedTensor((
        torch.full((1, 2, 2), video),
        torch.full((1, 3), audio),
    ))


def _fsdp_cycle(world: int = 2) -> list[dict]:
    return [
        {
            "rank": rank,
            "setup_generation": 4,
            "conclusive": True,
            "proof": "fsdp_clean_reload",
            "baseline_verified": True,
            "baseline_artifact_identity_verified": True,
            "baseline_family": "wan",
            "baseline_quant": "bf16",
            "baseline_live_dtype_profile": "all_bf16",
            "baseline_auxiliary_parameter_count": 0,
            "baseline_auxiliary_parameter_bytes": 0,
            "baseline_checkpoint_precision": "bf16",
            "baseline_slab_active": False,
            "baseline_fsdp_ready": True,
            "transitions": {"reload": "load"},
            "family": "wan",
            "quant": "bf16",
            "live_dtype_profile": "all_bf16",
            "auxiliary_parameter_count": 0,
            "auxiliary_parameter_bytes": 0,
            "checkpoint_precision": "bf16",
            "slab_active": False,
            "fsdp_ready": True,
            "artifact_identity_verified": True,
        }
        for rank in range(world)
    ]


def test_provenance_bracket_precedes_every_gate_pass_publication(
    monkeypatch, tmp_path,
):
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.zeros(1)],
        worker_args,
    )
    real_publish = common._record_process_gate_verdicts
    real_record = ledger.record

    def publish(tokens, verdict, ceremony=None):
        ledger.events.append(("process", verdict))
        return real_publish(tokens, verdict, ceremony)

    def record(*args):
        ledger.events.append(("ledger", args[3]))
        return real_record(*args)

    def attest(boundary, bound_handle):
        assert bound_handle is handle
        ledger.events.append(("provenance", boundary))

    monkeypatch.setattr(common, "_record_process_gate_verdicts", publish)
    ledger.record = record
    result = gate_mod.run_identity_ceremony(
        model,
        {"noise_seed": 1, "steps": 2, "cfg": 1.0},
        {"samples": torch.zeros(1)},
        1.0,
        2,
        "explicit",
        run_id="ordered",
        provenance_attestor=attest,
    )

    assert result["verdict"] == "PASS"
    ordered = ledger.events
    retest_index = next(
        index for index, event in enumerate(ordered) if event[0] == "retest"
    )
    baselines = [
        index for index, event in enumerate(ordered)
        if event == ("call", "provenance_baseline")
    ]
    assert len(baselines) == 2
    assert retest_index < ordered.index(("process", "INCONCLUSIVE"))
    assert ordered.index(("process", "INCONCLUSIVE")) < baselines[0]
    assert baselines[0] < ordered.index(("provenance", "pre"))
    assert ordered.index(("provenance", "pre")) < ordered.index(("call", "unload"))
    assert ordered.index(("call", "gate_swap_cycle")) < baselines[1]
    assert baselines[1] < ordered.index(("provenance", "post"))
    assert ordered.index(("provenance", "post")) < ordered.index(("process", "PASS"))
    assert ordered.index(("process", "PASS")) < ordered.index(
        ("ledger", "PASS")
    )


class _ProvenanceInterrupt(BaseException):
    pass


@pytest.mark.parametrize("boundary", ["pre", "post"])
@pytest.mark.parametrize("failure_type", [RuntimeError, _ProvenanceInterrupt])
def test_provenance_failure_leaves_gate_denied_and_quarantined(
    monkeypatch, tmp_path, boundary, failure_type,
):
    worker_args = {"lora_low_rss": True, "slab_weights": True}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.zeros(1), torch.zeros(1)],
        worker_args,
    )
    real_publish = common._record_process_gate_verdicts
    publications = []

    def publish(tokens, verdict, ceremony=None):
        publications.append((tuple(tokens), verdict))
        return real_publish(tokens, verdict, ceremony)

    def attest(current, bound_handle):
        assert bound_handle is handle
        if current == boundary:
            raise failure_type(f"injected {boundary} provenance failure")

    monkeypatch.setattr(common, "_record_process_gate_verdicts", publish)
    with pytest.raises(failure_type, match=f"injected {boundary}"):
        gate_mod.run_identity_ceremony(
            model,
            {"noise_seed": 1, "steps": 2, "cfg": 1.0},
            {"samples": torch.zeros(1)},
            1.0,
            2,
            "explicit",
            run_id="failed-provenance",
            provenance_attestor=attest,
        )

    assert ledger.retests
    assert ledger.records == []
    assert [verdict for _tokens, verdict in publications] == ["INCONCLUSIVE"]
    assert all(
        common._process_gate_verdict(token) == "INCONCLUSIVE"
        for token in publications[0][0]
    )
    assert worker_args["lora_low_rss"] is False
    assert worker_args["slab_weights"] is False
    if boundary == "pre":
        assert not any(
            method in {"unload", "gate_swap_cycle"}
            for method, _args in handle.calls
        )
        assert not any(method == "render" for method, _args in handle.calls)


def test_ceremony_durably_guards_every_potential_grant_before_unload(
    monkeypatch, tmp_path,
):
    worker_args = {"lora_low_rss": True}  # slab_weights omitted: auto on an integrated worker
    handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.zeros(1)],
        worker_args,
        cycle_response=[{
            "conclusive": True,
            "transitions": ["lazy"],
            "family": "unvouched",
            "slab_active": False,
        }],
    )

    result = _run(model)

    assert result["verdict"] == "PASS"
    assert ledger.events[0][0] == "retest"
    assert handle.calls[0][0] == "provenance_baseline"
    assert handle.calls[1][0] == "unload"
    guarded_contexts = ledger.retests[0][3]
    assert len(guarded_contexts) == 4
    assert [context["worker_args"].get("slab_weights")
            for context in guarded_contexts] == [None, None, True, True]
    # Slab never ran, so only the normal and Fleet contexts get PASS rows;
    # the explicit-on siblings stay denied by the preflight RETESTING row.
    assert [row[3] for row in ledger.records] == ["PASS", "PASS"]


def test_required_retest_guard_failure_aborts_before_any_ceremony_leg(
    monkeypatch, tmp_path,
):
    worker_args = {"lora_low_rss": True, "slab_weights": True}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.zeros(1), torch.zeros(1)],
        worker_args,
    )

    def fail_guard(*_args):
        raise ledger_mod.GateLedgerWriteError("guard fsync failed")

    monkeypatch.setattr(ledger, "begin_retest_required", fail_guard)

    with pytest.raises(ledger_mod.GateLedgerWriteError, match="guard fsync failed"):
        _run(model)

    assert not any(
        method in {"provenance_baseline", "unload", "gate_swap_cycle", "render"}
        for method, _args in handle.calls
    )
    assert model.mesh.worker_args == {
        "lora_low_rss": False,
        "slab_weights": False,
    }


def test_cross_mode_reference_passes_and_restores_slab(monkeypatch, tmp_path):
    """Slab requested, all three lineages agree: PASS with a recorded
    cross-mode verdict, stock push then slab restore."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), torch.zeros(1)], wa)
    result = _run(model)
    assert result["verdict"] == "PASS"
    assert result["cross_mode"] == {"verdict": "PASS", "max_abs_latent_diff": 0.0}
    applies = [args[0] for method, args in handle.calls if method == "apply_worker_args"]
    assert [a["slab_weights"] for a in applies] == [False, True]
    assert wa == {"lora_low_rss": True, "slab_weights": True}  # untouched
    assert ledger.records[0][4]["cross_mode"] == "PASS"
    # Only the 2 ceremony rows: an explicit-on ceremony never stamps the
    # omitted sibling. An absent key resolves per hardware (integrated
    # worker defaults inject "auto"; on discrete it stays absent, which is
    # stock), so stamping it would grant a policy context no leg ran.
    assert [record[3] for record in ledger.records] == ["PASS", "PASS"]
    assert not any(record[4].get("stamped") for record in ledger.records)
    normal_context, fleet_context = ledger.records[0][5], ledger.records[1][5]
    assert "topology_preset" in normal_context
    assert fleet_context["capability"] == "fleet_per_rank_residency"
    assert fleet_context["rank_world"] == 1
    assert fleet_context["worker_args"] == wa
    assert not ({"topology_preset", "attention", "sync_ulysses"} & set(fleet_context))
    rendered_requests = [args[1] for method, args in handle.calls if method == "render"]
    assert all(request["_dgxm_artifact_binding"]["artifact_sets"]
               == [_identity(model.unet_name, model.loras)]
               for request in rendered_requests)
    cycle_args = next(args for method, args in handle.calls if method == "gate_swap_cycle")
    assert cycle_args[-1] == _identity(model.unet_name, model.loras)


def test_nested_normal_ceremony_passes_and_serializes_without_latents(
    monkeypatch, tmp_path,
):
    NestedTensor = _install_nested_tensor(monkeypatch)
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    stock = _nested(NestedTensor)
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [stock, _nested(NestedTensor)], worker_args)

    result = _run(model)

    assert result["verdict"] == "PASS"
    assert result["latents_identical"] is True
    assert result["max_abs_latent_diff"] == 0.0
    assert result["latent"]["samples"] is stock
    assert ledger.records[0][3] == "PASS"
    report = json.loads((tmp_path / "dgxm_gate_reports.jsonl").read_text())
    assert report["verdict"] == "PASS"
    assert report["max_abs_latent_diff"] == 0.0
    assert "latent" not in report


def test_real_comfy_nested_tensor_runs_identity_ceremony(monkeypatch, tmp_path):
    comfy_nested = pytest.importorskip("comfy.nested_tensor")

    def packed(video=0.0, audio=0.0):
        return comfy_nested.NestedTensor((
            torch.full((1, 2, 2), video),
            torch.full((1, 3), audio),
        ))

    worker_args = {"lora_low_rss": True, "slab_weights": False}
    stock = packed()
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [stock, packed()], worker_args,
    )

    result = _run(model)

    assert result["verdict"] == "PASS"
    assert result["latents_identical"] is True
    assert result["max_abs_latent_diff"] == 0.0
    assert type(result["latent"]["samples"]) is comfy_nested.NestedTensor
    assert ledger.records[0][3] == "PASS"


def test_first_use_auto_gate_runs_production_nested_ceremony(
    monkeypatch, tmp_path,
):
    NestedTensor = _install_nested_tensor(monkeypatch)
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [_nested(NestedTensor), _nested(NestedTensor)],
        worker_args,
    )
    token = ("nested-auto-v7",)
    real_ceremony = gate_mod.run_identity_ceremony

    def ceremony(*args, **kwargs):
        assert kwargs["origin"] == "auto_first_use"
        result = real_ceremony(*args, **kwargs)
        # Token construction has its own tests. Fixed tokens keep this one on
        # the first-use path and the packed ceremony, not the rig's identity.
        result["_gate_token"] = token
        result["_gate_tokens"] = [token]
        return result

    monkeypatch.setattr(gate_mod, "run_identity_ceremony", ceremony)
    monkeypatch.setattr(
        common,
        "_auto_gate_context",
        lambda *_args: ("unknown", token),
    )
    previous_active = getattr(common._AUTO_GATE_ACTIVE, "on", False)
    common._AUTO_GATE_ACTIVE.on = False
    common._AUTO_GATE_SESSION.clear()
    common._AUTO_GATE_RUNNING.clear()
    try:
        verdict = common._maybe_auto_gate(
            model,
            {"kind": "ksampler", "noise_seed": 1, "steps": 4, "cfg": 1.0},
            {"samples": _nested(NestedTensor)},
            1.0,
            4,
        )

        assert verdict == "PASS"
        assert common._AUTO_GATE_SESSION[token] == "PASS"
        assert ledger.records[0][3] == "PASS"
        report = json.loads((tmp_path / "dgxm_gate_reports.jsonl").read_text())
        assert report["origin"] == "auto_first_use"
        assert report["latents_identical"] is True
        assert report["max_abs_latent_diff"] == 0.0
    finally:
        common._AUTO_GATE_ACTIVE.on = previous_active
        common._AUTO_GATE_SESSION.clear()
        common._AUTO_GATE_RUNNING.clear()


@pytest.mark.parametrize("modality", [0, 1])
def test_nested_normal_divergence_quarantines_low_rss(
    monkeypatch, tmp_path, modality,
):
    NestedTensor = _install_nested_tensor(monkeypatch)
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    risky = _nested(NestedTensor)
    risky.unbind()[modality].flatten()[0] = 2.0
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [_nested(NestedTensor), risky],
        worker_args,
    )

    result = _run(model)

    assert result["verdict"] == "FAIL"
    assert result["latents_identical"] is False
    assert result["max_abs_latent_diff"] == 2.0
    assert worker_args == {"lora_low_rss": False, "slab_weights": False}
    assert ledger.records[0][4]["quarantine_levers"] == ["lora_low_rss"]


def test_nested_slab_cross_mode_passes_all_modalities(monkeypatch, tmp_path):
    NestedTensor = _install_nested_tensor(monkeypatch)
    worker_args = {"lora_low_rss": True, "slab_weights": True}
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [_nested(NestedTensor), _nested(NestedTensor), _nested(NestedTensor)],
        worker_args,
    )

    result = _run(model)

    assert result["verdict"] == "PASS"
    assert result["cross_mode"] == {
        "verdict": "PASS",
        "max_abs_latent_diff": 0.0,
    }
    assert ledger.records[0][4]["cross_mode"] == "PASS"


@pytest.mark.parametrize("modality", [0, 1])
def test_nested_slab_cross_mode_divergence_returns_stock_and_quarantines_slab(
    monkeypatch, tmp_path, modality,
):
    NestedTensor = _install_nested_tensor(monkeypatch)
    worker_args = {"lora_low_rss": True, "slab_weights": True}
    stock_reference = _nested(NestedTensor)
    stock_reference.unbind()[modality].flatten()[0] = 4.0
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [_nested(NestedTensor), _nested(NestedTensor), stock_reference],
        worker_args,
    )

    result = _run(model)

    assert result["verdict"] == "FAIL"
    assert result["cross_mode"] == {
        "verdict": "FAIL",
        "max_abs_latent_diff": 4.0,
    }
    assert result["latent"]["samples"] is stock_reference
    assert worker_args == {"lora_low_rss": True, "slab_weights": False}
    assert ledger.records[0][4]["quarantine_levers"] == ["slab_weights"]


def test_nested_structure_mismatch_fails_and_quarantines(monkeypatch, tmp_path):
    NestedTensor = _install_nested_tensor(monkeypatch)
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    malformed_lineage = NestedTensor((torch.zeros(1),))
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [_nested(NestedTensor), malformed_lineage],
        worker_args,
    )

    result = _run(model)

    assert result["verdict"] == "FAIL"
    assert result["max_abs_latent_diff"] is None
    assert worker_args == {"lora_low_rss": False, "slab_weights": False}
    assert ledger.records[0][3] == "FAIL"


@pytest.mark.parametrize("modality", [0, 1])
def test_ceremony_rejects_render_mutation_of_either_packed_input_modality(
    monkeypatch, tmp_path, modality,
):
    NestedTensor = _install_nested_tensor(monkeypatch)
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [_nested(NestedTensor), _nested(NestedTensor)],
        worker_args,
    )

    def mutating_render(_model, _request, render_latent, **_kwargs):
        render_latent["samples"].unbind()[modality].add_(1)
        return {"samples": _nested(NestedTensor)}

    monkeypatch.setattr(gate_mod, "run_render", mutating_render)

    with pytest.raises(RuntimeError, match="mutated a captured"):
        _run(model, {"samples": _nested(NestedTensor)})

    assert ledger.records == []
    assert worker_args == {"lora_low_rss": False, "slab_weights": False}
    assert any(method == "apply_worker_args" for method, _args in handle.calls)


def test_ceremony_rechecks_a_retained_packed_reference_before_grant(
    monkeypatch, tmp_path,
):
    NestedTensor = _install_nested_tensor(monkeypatch)
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [_nested(NestedTensor), _nested(NestedTensor)],
        worker_args,
    )
    real_render = gate_mod.run_render
    retained = []

    def retaining_render(render_model, render_request, render_latent, **kwargs):
        retained.append(render_latent)
        return real_render(render_model, render_request, render_latent, **kwargs)

    monkeypatch.setattr(gate_mod, "run_render", retaining_render)
    real_effective_worker_args = gate_mod._effective_worker_args
    mutation_observed = False

    def mutate_after_render_checks(*args, **kwargs):
        nonlocal mutation_observed
        if len(retained) >= 2 and not mutation_observed:
            audio = retained[0]["samples"].unbind()[1]
            version = int(getattr(audio, "_version", 0))
            audio.data.add_(1)
            assert int(getattr(audio, "_version", 0)) == version
            mutation_observed = True
        return real_effective_worker_args(*args, **kwargs)

    monkeypatch.setattr(
        gate_mod, "_effective_worker_args", mutate_after_render_checks
    )

    with pytest.raises(RuntimeError, match="mutated a captured"):
        _run(model, {"samples": _nested(NestedTensor)})

    assert mutation_observed
    assert ledger.records == []
    assert worker_args == {"lora_low_rss": False, "slab_weights": False}
    assert any(method == "apply_worker_args" for method, _args in handle.calls)


def test_world1_nonfinite_nested_ceremony_fails_and_quarantines(
    monkeypatch, tmp_path,
):
    NestedTensor = _install_nested_tensor(monkeypatch)
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    infinity = _nested(NestedTensor, audio=float("inf"))
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [infinity, _nested(NestedTensor, audio=float("inf"))],
        worker_args,
    )

    result = _run(model)

    assert result["verdict"] == "FAIL"
    assert result["latents_identical"] is False
    assert result["max_abs_latent_diff"] is None
    assert worker_args == {"lora_low_rss": False, "slab_weights": False}
    assert ledger.records[0][3] == "FAIL"


def test_ceremony_persists_the_rendered_snapshot_not_a_post_final_replacement(
        monkeypatch, tmp_path):
    wa = {"lora_low_rss": True, "slab_weights": True}
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), torch.zeros(1)], wa)
    initial = _identity(model.unet_name, model.loras)
    replacement = {**initial, "digest": "replacement-digest"}
    calls = []

    def changing_identity(*_args):
        calls.append(1)
        return initial if len(calls) == 1 else replacement

    monkeypatch.setattr(
        model_store_mod, "request_artifact_identity", changing_identity)

    result = _run(model)

    assert result["verdict"] == "PASS"
    assert calls == [1]  # no identity re-read at ledger-record time
    assert all(record[1].current == initial["digest"] for record in ledger.records)


def test_identity_ceremony_rejects_policy_drift_before_pass(monkeypatch, tmp_path):
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [], worker_args,
        cycle_response=[{"conclusive": True, "transitions": ["lazy"],
                         "family": "unvouched", "slab_active": False}],
    )
    handle.config = SimpleNamespace(worker_args={"compile_dit": True})
    captured_policy = {"compile_dit": True, **worker_args}
    replacement_handle = SimpleNamespace(
        config=SimpleNamespace(worker_args={"compile_dit": False}))
    rendered = []

    def render(render_model, *_args, **_kwargs):
        rendered.append(
            (render_model.mesh.handle, dict(render_model.mesh.worker_args)))
        if len(rendered) == 1:
            model.mesh.worker_args["lora_low_rss"] = False
            handle.config.worker_args["compile_dit"] = False
            model.mesh.handle = replacement_handle
        return {"samples": torch.zeros(1)}

    monkeypatch.setattr(gate_mod, "run_render", render)

    with pytest.raises(RuntimeError, match="worker policy changed during the ceremony"):
        _run(model)

    assert any(method == "gate_swap_cycle" for method, _args in handle.calls)
    assert len(rendered) == 2
    assert all(render_handle is handle for render_handle, _policy in rendered)
    assert all(policy == captured_policy for _render_handle, policy in rendered)
    assert ledger.records == []


def test_cross_mode_divergence_fails_and_quarantines_slab(monkeypatch, tmp_path):
    """Swap lineages agree but slab diverges from stock residency: FAIL, and
    the quarantine lever is slab_weights; lora_low_rss stays on."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), torch.ones(1)], wa)
    result = _run(model)
    assert result["verdict"] == "FAIL"
    assert result["cross_mode"]["verdict"] == "FAIL"
    assert torch.equal(result["latent"]["samples"], torch.ones(1))
    assert wa["slab_weights"] is False
    assert wa["lora_low_rss"] is True
    quarantine = [args[0] for method, args in handle.calls
                  if method == "apply_worker_args"][-1]
    assert quarantine["slab_weights"] is False and quarantine["lora_low_rss"] is True
    assert ledger.records[0][3] == "FAIL"
    assert [record[3] for record in ledger.records] == ["FAIL", "FAIL"]
    assert ledger.records[1][5]["capability"] == "fleet_per_rank_residency"


def test_cross_residency_divergence_under_auto_revokes_the_explicit_on_sibling(
    monkeypatch, tmp_path,
):
    """With slab_weights auto the ceremony rendered under slab residency and
    that render diverged from stock, so the explicit-on sibling is what failed
    and takes the FAIL."""
    wa = {"lora_low_rss": True}
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), torch.ones(1)], wa,
        cycle_response=[{"conclusive": True, "transitions": ["lazy"],
                         "family": "krea2", "slab_active": True}])
    result = _run(model)
    assert result["verdict"] == "FAIL"
    assert result["cross_mode"]["verdict"] == "FAIL"
    assert [record[3] for record in ledger.records] == ["FAIL"] * 4
    stamped = [record for record in ledger.records
               if record[4].get("stamped") == "slab-mode-equivalence"]
    assert len(stamped) == 2
    assert all(record[5]["worker_args"]["slab_weights"] is True for record in stamped)


def test_lazy_swap_failure_under_auto_leaves_the_explicit_on_sibling_untested(
    monkeypatch, tmp_path,
):
    """Slab was resident and matched stock, but the swap lineage diverged. The
    finding is the lazy path, not residency, so the sibling that differs only
    in slab_weights keeps its preflight RETESTING row: denied for now, and
    re-gated on next use rather than quarantined for good."""
    wa = {"lora_low_rss": True}
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.ones(1), torch.zeros(1)], wa,
        cycle_response=[{"conclusive": True, "transitions": ["lazy"],
                         "family": "krea2", "slab_active": True}])
    result = _run(model)
    assert result["verdict"] == "FAIL"
    assert result["cross_mode"]["verdict"] == "PASS"
    assert [record[3] for record in ledger.records] == ["FAIL", "FAIL"]
    assert not any(record[4].get("stamped") for record in ledger.records)
    assert all(
        record[5]["worker_args"].get("slab_weights") is not True
        for record in ledger.records
    )
    # No sibling row is written, yet an older sibling PASS cannot stand: the
    # preflight RETESTING row covered both sibling contexts before the first
    # side effect, and nothing here supersedes it.
    blocked = [context for context in ledger.retests[0][3]
               if context["worker_args"].get("slab_weights") is True]
    assert len(blocked) == 2


def test_cross_mode_skipped_without_slab(monkeypatch, tmp_path):
    wa = {"lora_low_rss": True, "slab_weights": False}
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [torch.zeros(1), torch.zeros(1)], wa)
    result = _run(model)
    assert result["verdict"] == "PASS"
    assert result["cross_mode"] is None
    assert all(method != "apply_worker_args" for method, _ in handle.calls)


def test_fsdp_no_lora_clean_reload_can_earn_exact_capability_pass(
    monkeypatch,
    tmp_path,
):
    wa = {"lora_low_rss": False, "slab_weights": False}
    cycle = _fsdp_cycle()
    handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.zeros(1)],
        wa,
        cycle_response=cycle,
    )
    handle.world = 2
    model.loras = ()
    model.mesh.world = 2
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True

    result = _run(model)

    assert result["verdict"] == "PASS"
    assert result["proof_kind"] == "fsdp_clean_reload"
    assert result["cross_mode"] is None
    assert len(ledger.records) == 1
    assert all(record[3] == "PASS" for record in ledger.records)
    assert all(
        record[5]["resolved_topology"]["fsdp"] is True
        for record in ledger.records
        if "resolved_topology" in record[5]
    )


def test_fsdp_clean_reload_divergence_denies_without_vacuous_residency_quarantine(
    monkeypatch,
    tmp_path,
):
    wa = {"lora_low_rss": False, "slab_weights": False}
    cycle = _fsdp_cycle()
    handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.ones(1)],
        wa,
        cycle_response=cycle,
    )
    handle.world = 2
    model.loras = ()
    model.mesh.world = 2
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True

    result = _run(model)

    assert result["verdict"] == "FAIL"
    assert result["proof_kind"] == "fsdp_clean_reload"
    assert wa == {"lora_low_rss": False, "slab_weights": False}
    assert all(method != "apply_worker_args" for method, _args in handle.calls)
    assert handle.unload_calls == 2
    assert result["fsdp_cleanup_confirmed"] is True
    assert len(ledger.records) == 1
    assert all(record[4]["quarantine_levers"] == [] for record in ledger.records)


def test_incomplete_fsdp_reload_is_typed_and_never_stock_quarantines(
    monkeypatch,
    tmp_path,
):
    from dgx_monarch.nodes.gate_fsdp import FsdpGateProofError

    worker_args = {"lora_low_rss": False, "slab_weights": False}
    cycle = _fsdp_cycle()
    cycle[1]["fsdp_ready"] = False
    handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1)],
        worker_args,
        cycle_response=cycle,
    )
    handle.world = 2
    model.loras = ()
    model.mesh.world = 2
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True

    with pytest.raises(FsdpGateProofError) as raised:
        _run(model)

    assert raised.value.verdict == "INCONCLUSIVE"
    assert worker_args == {"lora_low_rss": False, "slab_weights": False}
    assert all(method != "apply_worker_args" for method, _args in handle.calls)
    assert handle.unload_calls == 2
    assert len(ledger.retests) == 1
    # The incomplete proof closes its own retest transaction (docs/TROUBLESHOOTING.md #92).
    assert len(ledger.records) == 1
    assert ledger.records[0][3] == "INCONCLUSIVE"


@pytest.mark.parametrize(
    "initial_response",
    [
        pytest.param({"unloaded": True}, id="malformed-non-list"),
        pytest.param(
            [{"unloaded": True}, {"unloaded": False}],
            id="negative-rank",
        ),
        pytest.param([{"unloaded": True}], id="missing-rank"),
    ],
)
def test_fsdp_baseline_requires_complete_positive_all_rank_unload(
    monkeypatch,
    tmp_path,
    initial_response,
):
    from dgx_monarch.nodes.gate_fsdp import FsdpGateProofError

    handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1)],
        {"lora_low_rss": False, "slab_weights": False},
        cycle_response=_fsdp_cycle(),
        initial_unload_response=initial_response,
    )
    handle.world = model.mesh.world = 2
    model.loras = ()
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True

    with pytest.raises(FsdpGateProofError) as raised:
        _run(model)

    assert raised.value.verdict == "INCONCLUSIVE"
    assert "baseline all-rank unload" in str(raised.value)
    assert handle.unload_calls == 2
    assert handle.dirty_latches == []
    assert all(method != "render" for method, _args in handle.calls)
    assert all(method != "gate_fsdp_reload_cycle" for method, _args in handle.calls)
    assert len(ledger.retests) == 1
    # An unproven baseline unload still closes its retest row (docs/TROUBLESHOOTING.md #92).
    assert len(ledger.records) == 1
    assert ledger.records[0][3] == "INCONCLUSIVE"


@pytest.mark.parametrize("failure_phase", ["baseline", "reload", "candidate"])
def test_aborted_fsdp_gate_unloads_every_rank_before_refusing(
    monkeypatch,
    tmp_path,
    failure_phase,
):
    from dgx_monarch.nodes.gate_fsdp import FsdpGateProofError

    failure = RuntimeError(f"{failure_phase} failed")
    renders = (
        [failure]
        if failure_phase == "baseline"
        else [torch.zeros(1)]
        if failure_phase == "reload"
        else [torch.zeros(1), failure]
    )
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        renders,
        {"lora_low_rss": False, "slab_weights": False},
        cycle_response=_fsdp_cycle(),
        cycle_error=failure if failure_phase == "reload" else None,
    )
    handle.world = 2
    model.loras = ()
    model.mesh.world = 2
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True

    with pytest.raises(FsdpGateProofError) as raised:
        _run(model)

    assert raised.value.__cause__ is failure
    assert handle.unload_calls == 2
    assert handle.dirty_latches == []


def test_aborted_fsdp_gate_latches_dirty_when_all_rank_unload_fails(
    monkeypatch,
    tmp_path,
):
    from dgx_monarch.nodes.gate_fsdp import FsdpGateProofError

    reload_error = RuntimeError("rank reload failed")
    cleanup_error = RuntimeError("rank unload failed")
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1)],
        {"lora_low_rss": False, "slab_weights": False},
        cycle_error=reload_error,
        cleanup_unload_error=cleanup_error,
    )
    handle.world = 2
    model.loras = ()
    model.mesh.world = 2
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True

    with pytest.raises(FsdpGateProofError) as raised:
        _run(model)

    assert handle.unload_calls == 2
    assert handle.dirty_latches == [("unload", 600.0, cleanup_error)]
    assert any("must be recycled" in note for note in raised.value.__notes__)


def test_fsdp_terminal_fail_latches_dirty_when_final_unload_fails(
    monkeypatch,
    tmp_path,
):
    cleanup_error = RuntimeError("terminal rank unload failed")
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.ones(1)],
        {"lora_low_rss": False, "slab_weights": False},
        cycle_response=_fsdp_cycle(),
        cleanup_unload_error=cleanup_error,
    )
    handle.world = model.mesh.world = 2
    model.loras = ()
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True

    result = _run(model)

    assert result["verdict"] == "FAIL"
    assert result["fsdp_cleanup_confirmed"] is False
    assert handle.unload_calls == 2
    assert handle.dirty_latches == [("unload", 600.0, cleanup_error)]
    assert handle.defunct is False


@pytest.mark.parametrize(
    "cleanup_response",
    [
        pytest.param([{"unloaded": True}], id="partial-world"),
        pytest.param(
            [{"unloaded": True}, {"unloaded": False}],
            id="negative-rank",
        ),
        pytest.param({"unloaded": True}, id="non-list"),
    ],
)
def test_fsdp_cleanup_requires_complete_positive_all_rank_evidence(
    monkeypatch,
    tmp_path,
    cleanup_response,
):
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.ones(1)],
        {"lora_low_rss": False, "slab_weights": False},
        cycle_response=_fsdp_cycle(),
        cleanup_unload_response=cleanup_response,
    )
    handle.world = model.mesh.world = 2
    model.loras = ()
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True

    result = _run(model)

    assert result["verdict"] == "FAIL"
    assert result["fsdp_cleanup_confirmed"] is False
    assert handle.unload_calls == 2
    assert len(handle.dirty_latches) == 1
    assert "incomplete evidence" in str(handle.dirty_latches[0][2])


@pytest.mark.parametrize("latch_failures", [1, 2])
def test_fsdp_cleanup_retries_dirty_latch_then_retires_if_exhausted(
    monkeypatch,
    tmp_path,
    latch_failures,
):
    from dgx_monarch.nodes.gate_fsdp import FsdpGateProofError

    reload_error = RuntimeError("rank reload failed")
    cleanup_error = RuntimeError("rank unload failed")
    latch_errors = [RuntimeError(f"latch {index}")
                    for index in range(latch_failures)]
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1)],
        {"lora_low_rss": False, "slab_weights": False},
        cycle_error=reload_error,
        cleanup_unload_error=cleanup_error,
        cleanup_latch_errors=latch_errors,
    )
    handle.world = model.mesh.world = 2
    model.loras = ()
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True

    with pytest.raises(FsdpGateProofError) as raised:
        _run(model)

    assert handle.unload_calls == 2
    assert handle.latch_attempts == 2
    assert handle.defunct is (latch_failures == 2)
    assert bool(handle.dirty_latches) is (latch_failures == 1)
    assert raised.value.__cause__ is reload_error


def test_late_fsdp_validation_baseexception_unloads_once_and_preserves_identity(
    monkeypatch,
    tmp_path,
):
    class GateCancelled(BaseException):
        def __bool__(self):
            raise AssertionError("active exception truthiness must not be evaluated")

    cancellation = GateCancelled("cancelled at the final transaction check")
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.zeros(1)],
        {"lora_low_rss": False, "slab_weights": False},
        cycle_response=_fsdp_cycle(),
    )
    handle.world = model.mesh.world = 2
    model.loras = ()
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True
    real_check = gate_mod.require_transaction_unchanged
    checks = 0

    def cancel_final_check(versions):
        nonlocal checks
        checks += 1
        if checks == 5:
            raise cancellation
        return real_check(versions)

    monkeypatch.setattr(
        gate_mod, "require_transaction_unchanged", cancel_final_check)

    with pytest.raises(GateCancelled) as raised:
        _run(model)

    assert raised.value is cancellation
    assert checks == 5
    assert handle.unload_calls == 2
    assert handle.latch_attempts == 0


def test_late_fsdp_validation_exception_is_typed_after_exact_cleanup(
    monkeypatch,
    tmp_path,
):
    from dgx_monarch.nodes.gate_fsdp import FsdpGateProofError

    failure = RuntimeError("late latent comparison failed")
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.zeros(1)],
        {"lora_low_rss": False, "slab_weights": False},
        cycle_response=_fsdp_cycle(),
    )
    handle.world = model.mesh.world = 2
    model.loras = ()
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True

    def fail_comparison(*_args):
        raise failure

    monkeypatch.setattr(gate_mod, "compare_latents", fail_comparison)

    with pytest.raises(FsdpGateProofError) as raised:
        _run(model)

    assert raised.value.__cause__ is failure
    assert handle.unload_calls == 2
    assert handle.latch_attempts == 0


def test_early_fsdp_baseexception_preserves_identity_and_skips_quarantine(
    monkeypatch,
    tmp_path,
):
    class GateCancelled(BaseException):
        pass

    cancellation = GateCancelled("cancelled before the first proof render")
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [],
        {"lora_low_rss": False, "slab_weights": False},
    )
    handle.world = 2
    model.loras = ()
    model.mesh.world = 2
    model.mesh.topology_preset = "uly2+fsdp"
    model.mesh.attention = "SDPA"
    model.mesh.sync_ulysses = True
    monkeypatch.setattr(
        gate_mod,
        "freeze_transaction",
        lambda *_args: (_ for _ in ()).throw(cancellation),
    )

    with pytest.raises(GateCancelled) as raised:
        _run(model)

    assert raised.value is cancellation
    assert handle.unload_calls == 1
    assert all(method != "apply_worker_args" for method, _args in handle.calls)


def test_no_lora_non_slab_worker_metadata_cannot_mint_generic_pass(
    monkeypatch,
    tmp_path,
):
    wa = {"lora_low_rss": False, "slab_weights": False}
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.zeros(1)],
        wa,
        cycle_response=[{
            "conclusive": True,
            "transitions": {"fake": "load"},
            "family": "wan",
            "slab_active": False,
        }],
    )
    model.loras = ()

    result = _run(model)

    assert result["verdict"] == "INCONCLUSIVE"
    assert result["proof_kind"] == "none"
    assert any(
        "no complete FSDP clean-reload proof" in reason
        for reason in result["inconclusive_reasons"]
    )
    assert all(record[3] == "INCONCLUSIVE" for record in ledger.records)


def test_cross_mode_error_leaves_verdict_inconclusive(monkeypatch, tmp_path):
    """An errored cross leg leaves slab exactness unproven, never passed, and
    the slab args are restored on the failure path too."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), RuntimeError("worker died")], wa)
    result = _run(model)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"]["verdict"] == "ERROR"
    applies = [args[0] for method, args in handle.calls if method == "apply_worker_args"]
    assert [a["slab_weights"] for a in applies] == [False, True]


def test_artifact_drift_in_cross_leg_aborts_and_forces_stock(monkeypatch, tmp_path):
    from dgx_monarch.mesh_safety import ArtifactBindingError

    wa = {"lora_low_rss": True, "slab_weights": True}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1),
         ArtifactBindingError("changed after the ceremony artifact snapshot")], wa)

    with pytest.raises(ArtifactBindingError, match="artifact snapshot"):
        _run(model)

    assert wa == {"lora_low_rss": False, "slab_weights": False}
    applies = [args[0] for method, args in handle.calls if method == "apply_worker_args"]
    assert applies[-1] == {"lora_low_rss": False, "slab_weights": False}
    assert ledger.records == []


def test_pass_under_auto_stamps_the_explicit_on_context(monkeypatch, tmp_path):
    """auto resolved to slab (vouched family, all-rank active): the PASS also
    covers the explicit-on policy, so toggling the widget must not re-gate."""
    wa = {"lora_low_rss": True}
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [torch.zeros(1)] * 3, wa,
        cycle_response=[{"conclusive": True, "transitions": ["lazy"],
                         "family": "krea2", "slab_active": True}])
    result = _run(model)
    assert result["verdict"] == "PASS"
    assert [record[3] for record in ledger.records] == ["PASS"] * 4
    stamped = [record for record in ledger.records
               if record[4].get("stamped") == "slab-mode-equivalence"]
    assert len(stamped) == 2
    assert all(record[5]["worker_args"]["slab_weights"] is True for record in stamped)


def test_config_supplied_auto_stamps_only_the_explicit_on_variant(monkeypatch, tmp_path):
    """slab_weights="auto" from cluster.toml (a string, key present) with
    all-rank slab residency stamps only the explicit-True variant, never a
    variant without the key, whose resolution depends on the hardware."""
    wa = {"lora_low_rss": True, "slab_weights": "auto"}
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [torch.zeros(1)] * 3, wa,
        cycle_response=[{"conclusive": True, "transitions": ["lazy"],
                         "family": "krea2", "slab_active": True}])
    result = _run(model)
    assert result["verdict"] == "PASS"
    stamped = [record for record in ledger.records
               if record[4].get("stamped") == "slab-mode-equivalence"]
    assert len(stamped) == 2
    assert all(record[5]["worker_args"]["slab_weights"] is True for record in stamped)
    assert all("slab_weights" in record[5]["worker_args"] for record in ledger.records)


def test_pass_with_unvouched_family_never_stamps(monkeypatch, tmp_path):
    """Explicit slab=on outside the vouched set: the PASS stays bound to its
    exact policy. Auto would resolve stock, which this ceremony never ran."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [torch.zeros(1)] * 3, wa,
        cycle_response=[{"conclusive": True, "transitions": ["lazy"],
                         "family": "unvouched", "slab_active": True}])
    result = _run(model)
    assert result["verdict"] == "PASS"
    assert len(ledger.records) == 2
    assert not any(record[4].get("stamped") for record in ledger.records)


def test_capacity_refusal_stays_inconclusive_with_no_pass_grant(monkeypatch, tmp_path):
    """A stock-load capacity refusal explains why the reference is missing; it
    never stands in for the stock lineage. The verdict is INCONCLUSIVE, no row
    can grant a PASS, and the slab policy is restored. On the auto path the
    wrapper then fails the session closed, as for any INCONCLUSIVE."""
    from dgx_monarch.mesh_safety import StockLoadCapacityError

    wa = {"lora_low_rss": True, "slab_weights": True}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1),
         StockLoadCapacityError("stock residency cannot load model.safetensors")], wa)
    result = _run(model)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"]["verdict"] == "CAPACITY"
    assert any("capacity" in reason for reason in result["inconclusive_reasons"])
    assert [record[3] for record in ledger.records] == ["INCONCLUSIVE", "INCONCLUSIVE"]
    assert not any(record[4].get("stamped") for record in ledger.records)
    applies = [args[0] for method, args in handle.calls if method == "apply_worker_args"]
    assert [a["slab_weights"] for a in applies] == [False, True]  # restored


def test_wrapped_capacity_refusal_is_classified_but_grants_nothing(monkeypatch, tmp_path):
    """Worker-side refusals arrive as ActorError text, and the class name in
    it carries the classification, as for ArtifactBindingError. The
    classification still grants no PASS."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    _handle, _ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1),
         RuntimeError("ActorError: StockLoadCapacityError: stock residency "
                      "cannot load model.safetensors")], wa)
    result = _run(model)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"]["verdict"] == "CAPACITY"


def test_generic_oom_in_reference_render_stays_inconclusive(monkeypatch, tmp_path):
    """An OOM during the reference leg's render proves nothing about the load,
    and an incomplete stock reference must never grant a PASS."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1),
         RuntimeError("torch.OutOfMemoryError: CUDA out of memory during denoise")], wa)
    result = _run(model)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"]["verdict"] == "ERROR"
    assert [record[3] for record in ledger.records] == ["INCONCLUSIVE", "INCONCLUSIVE"]
    applies = [args[0] for method, args in handle.calls if method == "apply_worker_args"]
    assert [a["slab_weights"] for a in applies] == [False, True]  # restored


def test_oom_during_stock_policy_apply_stays_inconclusive(monkeypatch, tmp_path):
    """An OOM while applying the stock policy never reaches the reference load,
    so the leg reads ERROR, not a CAPACITY refusal."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [torch.zeros(1), torch.zeros(1)], wa)
    original = handle.call_all
    state = {"raised": False}

    def call_all(method, *args, **kwargs):
        if method == "apply_worker_args" and not state["raised"]:
            state["raised"] = True
            handle.calls.append((method, args))
            raise RuntimeError("CUDA out of memory while pushing worker policy")
        return original(method, *args, **kwargs)

    handle.call_all = call_all
    result = _run(model)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"]["verdict"] == "ERROR"


def test_oom_cross_leg_under_auto_stays_inconclusive(monkeypatch, tmp_path):
    """With slab_weights auto the system chose slab, so memory exhaustion in
    the reference leg still leaves the fail-closed INCONCLUSIVE."""
    wa = {"lora_low_rss": True}
    _handle, _ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), RuntimeError("CUDA out of memory")], wa,
        cycle_response=[{"conclusive": True, "transitions": ["lazy"],
                         "family": "krea2", "slab_active": True}])
    result = _run(model)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"]["verdict"] == "ERROR"


def test_cross_mode_refuses_a_worker_that_stayed_in_slab(monkeypatch, tmp_path):
    """If any rank reports it did not leave slab residency, the leg errors
    instead of a vacuous slab-vs-slab render, and the verdict is INCONCLUSIVE."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [torch.zeros(1), torch.zeros(1)], wa)
    handle.apply_response = [{"slab_weights": True}]  # a rank stayed in slab
    result = _run(model)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"]["verdict"] == "ERROR"
    assert "did not leave slab residency" in result["cross_mode"]["detail"]
    # ranks that did flip must be restored to slab even on the refusal path
    applies = [args[0] for method, args in handle.calls if method == "apply_worker_args"]
    assert [a["slab_weights"] for a in applies] == [False, True]


def test_cross_mode_unloads_before_the_stock_reference(monkeypatch, tmp_path):
    """The stock leg must come from a fresh disk load, not whatever apply's
    side effects left resident."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), torch.zeros(1)], wa)
    _run(model)
    methods = [m for m, _ in handle.calls]
    stock_push = methods.index("apply_worker_args")
    assert methods[stock_push + 1] == "unload"
    assert methods[stock_push + 2] == "render"


def test_cross_mode_fail_quarantines_slab_even_when_swaps_inconclusive(monkeypatch, tmp_path):
    """A proven slab-vs-stock divergence must FAIL and quarantine slab even
    when the swap cycle produced nothing to compare; an INCONCLUSIVE there
    would silently keep a broken slab on."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), torch.ones(1)], wa)
    original = handle.call_all

    def call_all(method, *args, **kwargs):
        if method == "gate_swap_cycle":
            handle.calls.append((method, args))
            return [{"rank": 0, "world": handle.world,
                     "setup_generation": handle.setup_generation,
                     "conclusive": False, "reason": "no lazy path",
                     "family": "krea2", "slab_active": True}]
        return original(method, *args, **kwargs)

    handle.call_all = call_all
    result = _run(model)
    assert result["verdict"] == "FAIL"
    assert result["cross_mode"]["verdict"] == "FAIL"
    assert wa["slab_weights"] is False
    assert wa["lora_low_rss"] is True  # the lazy path was not disproven
    assert ledger.records[0][3] == "FAIL"


def test_both_levers_quarantine_when_both_findings_hold(monkeypatch, tmp_path):
    """In-mode divergence and cross-mode divergence: both levers flip."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    _handle, _ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.full((1,), 5.0), torch.ones(1)], wa)
    result = _run(model)
    assert result["verdict"] == "FAIL"
    assert wa["slab_weights"] is False
    assert wa["lora_low_rss"] is False


def test_cross_mode_triggers_on_worker_reported_residency(monkeypatch, tmp_path):
    """Under the Init node's slab_weights auto the driver's worker_args carry no
    slab key, because the family memo decides slab residency on the worker.
    The cross leg must key on the residency the cycle reports."""
    wa = {"lora_low_rss": True}  # no slab key: the auto default
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), torch.zeros(1)], wa)
    original = handle.call_all

    def call_all(method, *args, **kwargs):
        if method == "gate_swap_cycle":
            handle.calls.append((method, args))
            return [{"rank": 0, "world": handle.world,
                     "setup_generation": handle.setup_generation,
                     "conclusive": True, "transitions": ["lazy"],
                     "family": "krea2", "slab_active": True}]
        return original(method, *args, **kwargs)

    handle.call_all = call_all
    result = _run(model)
    assert result["verdict"] == "PASS"
    assert result["cross_mode"] == {"verdict": "PASS", "max_abs_latent_diff": 0.0}
    applies = [args[0] for method, args in handle.calls if method == "apply_worker_args"]
    assert [a.get("slab_weights") for a in applies] == [False, None]


def test_fresh_auto_slab_reloads_after_family_discovery_and_proves_residency(
        monkeypatch, tmp_path):
    """The first auto load discovers the family too late to slab-load.  The
    gate must replace B after one clean reload and prove that retry resident."""
    wa = {"lora_low_rss": True}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.ones(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)], wa)
    cycles = iter((False, True))
    original = handle.call_all

    def call_all(method, *args, **kwargs):
        if method == "gate_swap_cycle":
            handle.calls.append((method, args))
            return [{"rank": 0, "world": handle.world,
                     "setup_generation": handle.setup_generation,
                     "conclusive": True, "transitions": ["lazy"],
                     "family": "krea2", "slab_active": next(cycles)}]
        return original(method, *args, **kwargs)

    handle.call_all = call_all
    result = _run(model)

    assert result["verdict"] == "PASS"
    assert result["cross_mode"] == {"verdict": "PASS", "max_abs_latent_diff": 0.0}
    methods = [method for method, _args in handle.calls]
    assert methods[:7] == [
        "provenance_baseline", "unload", "render", "gate_swap_cycle",
        "unload", "render", "gate_swap_cycle",
    ]
    assert methods.count("gate_swap_cycle") == 2
    # 2 ceremony rows + 2 stamped for the explicit-on sibling (the retry
    # proved all-rank slab residency in a vouched family under auto).
    assert [record[3] for record in ledger.records] == ["PASS"] * 4
    stamped = [record for record in ledger.records if record[4].get("stamped")]
    assert all(record[5]["worker_args"]["slab_weights"] is True for record in stamped)


def test_compile_noop_auto_context_proves_only_the_discovered_slab_retry(
        monkeypatch, tmp_path):
    """The stock discovery cycle cannot grant the later Flux2 slab context."""
    wa = {"lora_low_rss": True, "compile_dit": True}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.ones(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)], wa)
    cycles = iter((False, True))
    original = handle.call_all

    def call_all(method, *args, **kwargs):
        if method == "gate_swap_cycle":
            handle.calls.append((method, args))
            return [{"rank": 0, "world": handle.world,
                     "setup_generation": handle.setup_generation,
                     "conclusive": True, "transitions": ["lazy"],
                     "family": "flux2", "slab_active": next(cycles)}]
        return original(method, *args, **kwargs)

    handle.call_all = call_all
    result = _run(model)

    assert result["verdict"] == "PASS"
    assert [method for method, _args in handle.calls].count("gate_swap_cycle") == 2
    assert [record[3] for record in ledger.records] == ["PASS"] * 4


def test_compile_blocks_auto_context_keeps_krea2_stock_without_slab_proof(
        monkeypatch, tmp_path):
    """compile_dit compiles krea2's blocks, which keeps it stock, so the gate asks for no slab proof."""
    wa = {"lora_low_rss": True, "compile_dit": True}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [torch.zeros(1), torch.zeros(1)], wa)
    original = handle.call_all

    def call_all(method, *args, **kwargs):
        if method == "gate_swap_cycle":
            handle.calls.append((method, args))
            return [{"rank": 0, "world": handle.world,
                     "setup_generation": handle.setup_generation,
                     "conclusive": True, "transitions": ["lazy"],
                     "family": "krea2", "slab_active": False}]
        return original(method, *args, **kwargs)

    handle.call_all = call_all
    result = _run(model)

    assert result["verdict"] == "PASS"
    assert [method for method, _args in handle.calls].count("gate_swap_cycle") == 1
    assert all(record[5]["worker_args"].get("compile_dit") is True for record in ledger.records)


@pytest.mark.parametrize("slab_weights", (None, True), ids=("auto", "explicit-on"))
def test_compile_blocks_rejects_an_active_krea2_slab(monkeypatch, tmp_path, slab_weights):
    wa = {"lora_low_rss": True, "compile_dit": True}
    if slab_weights is not None:
        wa["slab_weights"] = slab_weights
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [torch.zeros(1), torch.zeros(1)], wa)
    original = handle.call_all

    def call_all(method, *args, **kwargs):
        if method == "gate_swap_cycle":
            handle.calls.append((method, args))
            return [{"rank": 0, "world": handle.world,
                     "setup_generation": handle.setup_generation,
                     "conclusive": True, "transitions": ["lazy"],
                     "family": "krea2", "slab_active": True}]
        return original(method, *args, **kwargs)

    handle.call_all = call_all
    result = _run(model)

    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"] == {
        "verdict": "ERROR",
        "detail": "slab residency was active although compile_dit requires stock residency",
    }


def test_auto_unvouched_active_slab_is_a_policy_error(monkeypatch, tmp_path):
    wa = {"lora_low_rss": True}
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1)], wa)
    original = handle.call_all

    def call_all(method, *args, **kwargs):
        if method == "gate_swap_cycle":
            handle.calls.append((method, args))
            return [{"rank": 0, "world": handle.world,
                     "setup_generation": handle.setup_generation,
                     "conclusive": True, "transitions": ["lazy"],
                     "family": "unvouched", "slab_active": True}]
        return original(method, *args, **kwargs)

    handle.call_all = call_all
    result = _run(model)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"] == {
        "verdict": "ERROR",
        "detail": "slab residency was active outside the configured family policy",
    }


def test_auto_partial_unvouched_slab_cannot_pass(monkeypatch, tmp_path):
    wa = {"lora_low_rss": True}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), torch.zeros(1)], wa)
    handle.world = 2
    original = handle.call_all

    def call_all(method, *args, **kwargs):
        if method == "gate_swap_cycle":
            handle.calls.append((method, args))
            return [
                {"rank": 0, "world": 2, "setup_generation": 4,
                 "conclusive": True, "transitions": ["lazy"],
                 "family": "unvouched", "slab_active": True},
                {"rank": 1, "world": 2, "setup_generation": 4,
                 "conclusive": True, "transitions": ["lazy"],
                 "family": "unvouched", "slab_active": False},
            ]
        return original(method, *args, **kwargs)

    handle.call_all = call_all
    result = _run(model)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"]["verdict"] == "ERROR"
    # Negative evidence revokes both base capabilities and their deterministic
    # explicit-slab siblings; it never grants cross-mode equivalence.
    assert [record[3] for record in ledger.records] == ["INCONCLUSIVE"] * 4


def test_explicit_stock_context_refuses_unexpected_active_slab(monkeypatch, tmp_path):
    wa = {"lora_low_rss": True, "slab_weights": False}
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path, [torch.zeros(1), torch.zeros(1)], wa)
    original = handle.call_all

    def call_all(method, *args, **kwargs):
        if method == "gate_swap_cycle":
            handle.calls.append((method, args))
            return [{"rank": 0, "world": handle.world,
                     "setup_generation": handle.setup_generation,
                     "conclusive": True, "transitions": ["lazy"],
                     "family": "krea2", "slab_active": True}]
        return original(method, *args, **kwargs)

    handle.call_all = call_all
    result = _run(model)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"] == {
        "verdict": "ERROR",
        "detail": "slab residency was active outside the configured family policy"}


def test_slab_retry_cannot_downgrade_initial_family_obligation(monkeypatch, tmp_path):
    wa = {"lora_low_rss": True}
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), torch.zeros(1)], wa)
    cycles = iter(("krea2", "unvouched"))
    original = handle.call_all

    def call_all(method, *args, **kwargs):
        if method == "gate_swap_cycle":
            handle.calls.append((method, args))
            return [{"rank": 0, "world": handle.world,
                     "setup_generation": handle.setup_generation,
                     "conclusive": True, "transitions": ["lazy"],
                     "family": next(cycles), "slab_active": False}]
        return original(method, *args, **kwargs)

    handle.call_all = call_all
    result = _run(model)
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"] == {
        "verdict": "ERROR", "detail": "resident family changed during slab residency proof"}


def test_explicit_slab_without_all_rank_residency_never_passes_or_crosses(
        monkeypatch, tmp_path):
    """A rank can bypass the slab loader while slab is requested, as FSDP does.
    A lazy-swap proof in that state must not grant slab residency to any context."""
    wa = {"lora_low_rss": True, "slab_weights": True}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), torch.zeros(1)], wa)
    handle.world = 2
    original = handle.call_all

    def call_all(method, *args, **kwargs):
        if method == "gate_swap_cycle":
            handle.calls.append((method, args))
            return [
                {"rank": 0, "world": 2, "setup_generation": 4,
                 "conclusive": True, "transitions": ["lazy"],
                 "family": "krea2", "slab_active": True},
                {"rank": 1, "world": 2, "setup_generation": 4,
                 "conclusive": True, "transitions": ["lazy"],
                 "family": "krea2", "slab_active": False},
            ]
        return original(method, *args, **kwargs)

    handle.call_all = call_all
    result = _run(model)

    reason = "slab residency was requested but not exercised"
    assert result["verdict"] == "INCONCLUSIVE"
    assert result["cross_mode"] == {"verdict": "ERROR", "detail": reason}
    assert reason in result["inconclusive_reasons"]
    assert [method for method, _args in handle.calls].count("gate_swap_cycle") == 2
    assert [method for method, _args in handle.calls].count("render") == 3
    assert all(method != "apply_worker_args" for method, _args in handle.calls)
    assert [record[3] for record in ledger.records] == ["INCONCLUSIVE", "INCONCLUSIVE"]


def test_no_lora_slab_cross_mode_pass_is_conclusive(monkeypatch, tmp_path):
    wa = {"lora_low_rss": False, "slab_weights": True}
    handle, _ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1), torch.zeros(1)], wa)
    model.loras = ()
    original = handle.call_all

    def call_all(method, *args, **kwargs):
        if method == "gate_swap_cycle":
            handle.calls.append((method, args))
            return [{"rank": 0, "world": handle.world,
                     "setup_generation": handle.setup_generation,
                     "conclusive": False, "reason": "no lora stack",
                     "family": "krea2", "slab_active": True}]
        return original(method, *args, **kwargs)

    handle.call_all = call_all
    result = _run(model)
    assert result["verdict"] == "PASS"
    assert result["cross_mode"]["verdict"] == "PASS"
    assert wa == {"lora_low_rss": False, "slab_weights": True}


def test_temporary_worker_args_restore_never_masks_body_exception(monkeypatch):
    """MeshHandle.temporary_worker_args must surface the body's render or CUDA
    error even when the compensating restore also fails."""
    from dgx_monarch import mesh as mesh_mod

    handle = mesh_mod.MeshHandle.__new__(mesh_mod.MeshHandle)
    handle.config = SimpleNamespace(worker_args={})
    calls = []

    def apply(worker_args, timeout_s=120.0, merge_config=True):
        calls.append(dict(worker_args))
        if len(calls) > 1:
            raise RuntimeError("restore broadcast failed")
        return [{"ok": True}]

    handle.apply_worker_args = apply
    with pytest.raises(RuntimeError, match="the real render error"):
        with handle.temporary_worker_args({"a": 1}, {"slab_weights": False}):
            raise RuntimeError("the real render error")
    assert len(calls) == 2  # restore was still attempted

    calls.clear()
    with pytest.raises(RuntimeError, match="restore broadcast failed"):
        with handle.temporary_worker_args({"a": 1}, {"slab_weights": False}):
            pass  # the body succeeds, so the restore failure must propagate


def test_known_commit_pass_can_be_session_cached_when_persistence_fails(
    monkeypatch, tmp_path,
):
    token = ("combo", "artifacts", "commit", "args")
    calls = []
    ledger = ledger_mod.GateLedger(str(tmp_path))

    def fail_makedirs(*_args, **_kwargs):
        raise OSError("read-only ledger directory")

    def ceremony(*_args, **_kwargs):
        calls.append(1)
        ledger.record("combo", "artifacts", "commit", "PASS", context={"args": 1})
        return {"verdict": "PASS", "_gate_token": token}

    monkeypatch.setattr(common, "_auto_gate_context", lambda *_args: ("stale", token))
    monkeypatch.setattr(gate_mod, "run_identity_ceremony", ceremony)
    monkeypatch.setattr(ledger_mod.os, "makedirs", fail_makedirs)
    common._AUTO_GATE_ACTIVE.on = False
    common._AUTO_GATE_SESSION.clear()
    request = {"kind": "ksampler", "steps": 2}

    assert common._maybe_auto_gate(SimpleNamespace(), request, {}, 1.0, 2) == "PASS"
    assert common._maybe_auto_gate(SimpleNamespace(), request, {}, 1.0, 2) == "PASS"
    assert calls == [1]
    assert ledger.lookup("combo", "artifacts", "commit", {"args": 1}) == "unknown"
    common._AUTO_GATE_SESSION.clear()


def test_dual_model_authorization_never_mints_primary_only_grant(
    monkeypatch, tmp_path,
):
    from dgx_monarch.nodes import gate_identity
    from dgx_monarch.topology import Topology

    ledger_reads = []
    model, handle, first = _normal_authorization_rig(
        monkeypatch,
        tmp_path,
        lambda context: ledger_reads.append(context) or (
            "pass", {"verdict": "PASS"}),
        session_verdict=lambda _token: "PASS",
    )
    assert first.residency_mode == "required"
    assert len(ledger_reads) == 1
    primary_request = model.request_dict()

    authorization = gate_identity.authorize_normal_render(
        model,
        handle,
        uncond_model_request={
            "unet_name": "negative.safetensors",
            "options": {"dtype": "bf16"},
            "loras": [{"name": "negative-style.safetensors", "strength": 0.5}],
        },
        gate_active=False,
        session_verdict=lambda _token: "PASS",
        resolved_topology=Topology(world=1),
        resolved_attention="TORCH_FLASH",
    )

    assert len(ledger_reads) == 1
    assert authorization.model_request == primary_request
    assert authorization.residency_mode == "stock"
    assert authorization.residency_grant is None
    assert authorization.worker_args["slab_weights"] is False
    assert authorization.worker_args["lora_low_rss"] is False


@pytest.mark.parametrize(
    ("write_final", "expected_verdict", "expected_state"),
    [(True, "PASS", "pass"), (False, "ERROR", "inconclusive")],
)
def test_damaged_real_ledger_cached_pass_requires_exact_reproof(
    monkeypatch, tmp_path, write_final, expected_verdict, expected_state,
):
    from dgx_monarch.gate_ledger import ArtifactSetSignature, GateLedger
    from dgx_monarch.topology import Topology

    context = {"capability": "normal", "worker_args": {"slab_weights": True}}
    artifacts = ArtifactSetSignature("artifact", "artifact", True)
    handle = SimpleNamespace(
        config=SimpleNamespace(worker_args={}, hosts=(), source=""),
        config_fingerprint="local",
        world=1,
        n_hosts=1,
        gpus_per_host=1,
    )
    handle.effective_worker_args = lambda requested: dict(requested)
    model = SimpleNamespace(
        unet_name="model.safetensors",
        options={},
        loras=(),
        mesh=SimpleNamespace(
            handle=handle,
            auto_gate="first_use",
            worker_args={"slab_weights": True, "lora_low_rss": False},
            topology_preset="single",
            attention="TORCH_FLASH",
            sync_ulysses=False,
        ),
    )
    latent = {"samples": torch.zeros(1, 4, 8, 8)}
    ledger = GateLedger(str(tmp_path))
    ledger.record("other", "other-artifact", "commit", "FAIL")
    with open(ledger.path, "ab") as stream:
        stream.write(b'{"torn":')

    monkeypatch.setattr(common, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        common,
        "_resolve_topology_for_latent",
        lambda *_args: (Topology(world=1), False, "test"),
    )
    monkeypatch.setattr(gate_mod, "_combo_of", lambda _model: ("target", artifacts))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        gate_mod, "gate_capability_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(ledger_mod, "comfy_commit", lambda: "commit")

    state, token = common._auto_gate_context(
        model, "ksampler", latent, 1.0)
    assert state == "error"
    common._AUTO_GATE_SESSION[token] = "PASS"
    assert common.auto_gate_required(model, "ksampler", latent, 1.0) is True

    def ceremony(*_args, **_kwargs):
        if write_final:
            ledger.record("target", artifacts, "commit", "PASS", context=context)
        else:
            ledger.begin_retest_required(
                "target", artifacts, "commit", [context])
        return {
            "verdict": "PASS",
            "_gate_token": token,
            "_gate_tokens": [token],
        }

    monkeypatch.setattr(gate_mod, "run_identity_ceremony", ceremony)
    result = common._maybe_auto_gate(
        model, {"kind": "ksampler", "steps": 2}, latent, 1.0, 2)

    assert result == expected_verdict
    assert ledger.lookup("target", artifacts, "commit", context) == expected_state
    assert common._process_gate_verdict(token) == expected_verdict


def test_no_lora_cross_pass_cannot_override_explicit_slab_repeat_divergence(
    monkeypatch, tmp_path,
):
    worker_args = {"lora_low_rss": False, "slab_weights": True}
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.ones(1), torch.zeros(1)],  # B=0, A=1, stock=0
        worker_args,
        cycle_response=[{
            "conclusive": False,
            "no_material": True,
            "reason": "no lora stack: lazy swap is not applicable",
            "family": "krea2",
            "slab_active": True,
        }],
    )
    model.loras = ()

    result = _run(model)

    assert result["verdict"] == "INCONCLUSIVE"
    assert result["latents_identical"] is False
    assert result["max_abs_latent_diff"] == 1.0
    assert result["cross_mode"] == {
        "verdict": "PASS",
        "max_abs_latent_diff": 0.0,
    }
    assert "same-residency repeat diverged" in "; ".join(
        result["inconclusive_reasons"])
    assert [record[3] for record in ledger.records] == [
        "INCONCLUSIVE", "INCONCLUSIVE"]
    assert not any(record[3] == "PASS" for record in ledger.records)
    assert len(result["_gate_tokens"]) == 2
    assert {
        common._process_gate_verdict(token) for token in result["_gate_tokens"]
    } == {"INCONCLUSIVE"}


def test_normal_render_source_manifest_failure_forces_stock_without_grant(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(
        ledger_mod.runtime_provenance,
        "cached_dgx_source_manifest_sha256",
        lambda: (_ for _ in ()).throw(RuntimeError("manifest unavailable")),
    )
    _model, _handle, authorization = _normal_authorization_rig(
        monkeypatch,
        tmp_path,
        lambda _context: pytest.fail("ledger lookup must not run"),
        session_verdict=lambda _token: pytest.fail("session lookup must not run"),
    )

    assert authorization.residency_mode == "stock"
    assert authorization.residency_grant is None
    assert authorization.worker_args["slab_weights"] is False
    assert authorization.worker_args["lora_low_rss"] is False


def test_no_lora_cross_pass_cannot_stamp_auto_vouched_slab_sibling_passes(
    monkeypatch, tmp_path,
):
    worker_args = {"lora_low_rss": True}  # omitted slab mode resolves auto
    _handle, ledger, model = _cross_mode_rig(
        monkeypatch,
        tmp_path,
        [torch.zeros(1), torch.ones(1), torch.zeros(1)],  # B=0, A=1, stock=0
        worker_args,
        cycle_response=[{
            "conclusive": False,
            "no_material": True,
            "reason": "no lora stack: lazy swap is not applicable",
            "family": "krea2",
            "slab_active": True,
        }],
    )
    model.loras = ()

    result = _run(model)

    assert result["verdict"] == "INCONCLUSIVE"
    assert result["latents_identical"] is False
    assert result["cross_mode"]["verdict"] == "PASS"
    assert "same-residency repeat diverged" in "; ".join(
        result["inconclusive_reasons"])
    assert [record[3] for record in ledger.records] == ["INCONCLUSIVE"] * 4
    stamped = [
        record for record in ledger.records
        if record[4].get("stamped") == "slab-mode-equivalence"
    ]
    assert len(stamped) == 2
    assert all(record[5]["worker_args"]["slab_weights"] is True for record in stamped)
    assert not any(record[3] == "PASS" for record in ledger.records)
    assert len(result["_gate_tokens"]) == 4
    assert {
        common._process_gate_verdict(token) for token in result["_gate_tokens"]
    } == {"INCONCLUSIVE"}


@pytest.mark.parametrize(
    "failure",
    [
        "source-mismatch",
        "missing-rank",
        "duplicate-rank",
        "world-mismatch",
        "generation-mismatch",
    ],
)
def test_default_provenance_attestor_rejects_incomplete_or_mixed_cohort_preproof(
    monkeypatch, tmp_path, failure,
):
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1)], worker_args,
    )
    if failure == "source-mismatch":
        handle.provenance_source = "0" * 64
        match = "package source does not match driver"
    else:
        handle.world = model.mesh.world = 2
        rows = [
            {
                "rank": rank,
                "world": 2,
                "setup_generation": 4,
                "source_manifest_sha256": handle.provenance_source,
            }
            for rank in range(2)
        ]
        if failure == "missing-rank":
            rows.pop()
            match = "lacks complete world coverage"
        elif failure == "duplicate-rank":
            rows[1]["rank"] = 0
            match = "rank coverage is invalid"
        elif failure == "world-mismatch":
            rows[1]["world"] = 1
            match = "world is inconsistent"
        else:
            rows[1]["setup_generation"] = 3
            match = "setup generation is inconsistent"
        handle.provenance_response = rows

    with pytest.raises(RuntimeError, match=match):
        _run(model)

    assert ledger.retests and ledger.records == []
    assert handle.provenance_calls == 1
    assert handle.provenance_kwargs[0]["setup_token"].generation == 4
    assert not any(
        method in {"unload", "gate_swap_cycle", "gate_fsdp_reload_cycle", "render"}
        for method, _args in handle.calls
    )
    assert worker_args == {"lora_low_rss": False, "slab_weights": False}


@pytest.mark.parametrize("boundary", ["pre", "post"])
@pytest.mark.parametrize("failure_type", [RuntimeError, _ProvenanceInterrupt])
def test_default_provenance_rpc_failure_never_publishes_a_terminal_row(
    monkeypatch, tmp_path, boundary, failure_type,
):
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1)], worker_args,
    )
    failed_call = 1 if boundary == "pre" else 2

    def provenance(call, generation):
        if call == failed_call:
            raise failure_type(f"injected {boundary} baseline failure")
        return [{
            "rank": 0,
            "world": 1,
            "setup_generation": generation,
            "source_manifest_sha256": handle.provenance_source,
        }]

    handle.provenance_response = provenance
    with pytest.raises(failure_type, match=f"injected {boundary}"):
        _run(model)

    assert ledger.retests and ledger.records == []
    assert handle.provenance_calls == failed_call
    assert worker_args == {"lora_low_rss": False, "slab_weights": False}
    if boundary == "pre":
        assert not any(
            method in {"unload", "gate_swap_cycle", "render"}
            for method, _args in handle.calls
        )
    else:
        assert any(method == "gate_swap_cycle" for method, _args in handle.calls)


def test_post_provenance_source_mismatch_cannot_publish_a_terminal_row(
    monkeypatch, tmp_path,
):
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1)], worker_args,
    )

    def provenance(call, generation):
        return [{
            "rank": 0,
            "world": 1,
            "setup_generation": generation,
            "source_manifest_sha256": (
                handle.provenance_source if call == 1 else "0" * 64
            ),
        }]

    handle.provenance_response = provenance
    with pytest.raises(RuntimeError, match="post provenance package source"):
        _run(model)

    assert handle.provenance_calls == 2
    assert any(method == "gate_swap_cycle" for method, _args in handle.calls)
    assert ledger.retests and ledger.records == []
    assert worker_args == {"lora_low_rss": False, "slab_weights": False}


def test_post_provenance_rejects_setup_generation_drift_before_publication(
    monkeypatch, tmp_path,
):
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1)], worker_args,
    )

    def provenance(call, generation):
        if call == 1:
            handle.setup_generation = generation + 1
        return [{
            "rank": 0,
            "world": 1,
            "setup_generation": generation,
            "source_manifest_sha256": handle.provenance_source,
        }]

    handle.provenance_response = provenance
    with pytest.raises(RuntimeError, match="is no longer current"):
        _run(model)

    assert handle.provenance_calls == 1
    assert any(method == "gate_swap_cycle" for method, _args in handle.calls)
    assert ledger.retests and ledger.records == []
    assert worker_args == {"lora_low_rss": False, "slab_weights": False}


def test_injected_provenance_observer_cannot_bypass_mandatory_cohort_check(
    monkeypatch, tmp_path,
):
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1)], worker_args,
    )
    handle.provenance_source = "0" * 64
    observations = []

    with pytest.raises(RuntimeError, match="package source does not match driver"):
        gate_mod.run_identity_ceremony(
            model,
            {"noise_seed": 1, "steps": 2, "cfg": 1.0},
            {"samples": torch.zeros(1)},
            1.0,
            2,
            "explicit",
            provenance_attestor=lambda *args: observations.append(args),
        )

    assert observations == []
    assert ledger.retests and ledger.records == []
    assert handle.provenance_calls == 1


def test_fresh_ceremony_establishes_setup_after_denial_before_pre_attestation(
    monkeypatch, tmp_path,
):
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1)], worker_args,
    )
    handle.setup_generation = 0
    handle.setup_key = None
    handle.worker_args_key = None
    token = SetupToken(4, ("fresh-setup",), ("fresh-policy",))

    def establish(*_args):
        ledger.events.append(("setup", token.generation))
        handle.setup_generation = token.generation
        handle.setup_key = token.key
        handle.worker_args_key = token.worker_args_key
        return token

    monkeypatch.setattr(gate_mod, "render_setup_token", establish)
    result = _run(model)

    assert result["verdict"] == "PASS"
    retest = next(i for i, event in enumerate(ledger.events) if event[0] == "retest")
    setup = ledger.events.index(("setup", 4))
    pre = ledger.events.index(("call", "provenance_baseline"))
    unload = ledger.events.index(("call", "unload"))
    assert retest < setup < pre < unload
    assert all(
        kwargs["setup_token"] == token for kwargs in handle.provenance_kwargs
    )


def test_post_provenance_malformed_row_cannot_publish_a_terminal_row(
    monkeypatch, tmp_path,
):
    worker_args = {"lora_low_rss": True, "slab_weights": False}
    handle, ledger, model = _cross_mode_rig(
        monkeypatch, tmp_path,
        [torch.zeros(1), torch.zeros(1)], worker_args,
    )

    def provenance(call, generation):
        if call == 2:
            return [None]
        return [{
            "rank": 0,
            "world": 1,
            "setup_generation": generation,
            "source_manifest_sha256": handle.provenance_source,
        }]

    handle.provenance_response = provenance
    with pytest.raises(RuntimeError, match="post provenance row is malformed"):
        _run(model)

    assert handle.provenance_calls == 2
    assert ledger.retests and ledger.records == []
    assert worker_args == {"lora_low_rss": False, "slab_weights": False}
