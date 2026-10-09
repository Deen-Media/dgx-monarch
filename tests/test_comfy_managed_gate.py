"""Driver-side fail-closed behavior around the rung: normal-render authority,
fleet policy, the persisted quarantine and the first-use ceremony trigger."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from comfy_managed_helpers import (  # noqa: F401  # autouse fixture import.
    SRC,
    _isolated_process_state,
    _tag,
)
from dgx_monarch import mesh_residency, residency_mode
from dgx_monarch.refusal import RefusalClass


def test_the_rung_is_risky_on_its_own():
    """Unlike the other two residencies it cannot be turned off for one call,
    so there is no safe downgrade and it is risky whenever it is on."""
    assert mesh_residency.fleet_policy_is_risky(
        {"loras": []},
        {"comfy_managed": True, "slab_weights": False, "lora_low_rss": False},
    ) is True


def test_a_fully_quarantined_classic_policy_is_still_not_risky():
    assert mesh_residency.fleet_policy_is_risky(
        {"loras": []},
        {"comfy_managed": False, "slab_weights": False, "lora_low_rss": False},
    ) is False


def test_normal_render_stock_mode_refuses_the_rung_with_a_typed_message():
    """Dual-model renders and every ceremony failure land on the `stock` stamp.

    Without a typed arm here they die with a bare RuntimeError that names slab
    and lora and never mentions the rung, the recycle, or
    docs/TROUBLESHOOTING.md #62.
    """
    request = {"model": {"loras": []}, "_dgxm_normal_residency_mode": "stock"}
    effective = {"comfy_managed": True, "slab_weights": False, "lora_low_rss": False}
    with pytest.raises(residency_mode.ComfyManagedResidencyError) as excinfo:
        mesh_residency.assert_normal_render_residency_mode(request, effective)
    text = str(excinfo.value)
    assert _tag(text).refusal_class is RefusalClass.PHYSICS
    assert "comfy_managed" in text
    assert "docs/TROUBLESHOOTING.md #62" in text
    assert "instead" in text


def test_normal_render_stock_mode_keeps_its_classic_refusal():
    request = {"model": {"loras": ["a"]}, "_dgxm_normal_residency_mode": "stock"}
    with pytest.raises(RuntimeError) as excinfo:
        mesh_residency.assert_normal_render_residency_mode(
            request, {"slab_weights": "auto", "lora_low_rss": True})
    assert not isinstance(excinfo.value, residency_mode.ComfyManagedResidencyError)


def test_normal_render_authority_modes_are_otherwise_unchanged():
    request = {"model": {"loras": []}, "_dgxm_normal_residency_mode": "operator_off"}
    effective = {"comfy_managed": True, "slab_weights": False, "lora_low_rss": False}
    assert mesh_residency.assert_normal_render_residency_mode(
        request, effective) == "operator_off"


def _normal_authorization(monkeypatch, tmp_path, *, state, entry, worker_args):
    """The KSampler path, end to end, with an injected ledger verdict."""
    import dgx_monarch.gate_ledger as ledger_mod
    import dgx_monarch.nodes.gate as gate_mod
    from dgx_monarch.actor import model_store as model_store_mod
    from dgx_monarch.nodes import gate_identity
    from dgx_monarch.nodes.common import MeshSpec, ModelSpec
    from dgx_monarch.topology import Topology

    def _identity(unet_name, loras):
        artifacts = [{"kind": "diffusion_models", "name": unet_name,
                      "signature": "model-sig"}]
        return {
            "digest": ledger_mod.artifact_set_signature(["model-sig"]).current,
            "comfy": "known",
            "artifacts": artifacts,
        }

    handle = SimpleNamespace(
        config=SimpleNamespace(worker_args={}, hosts=(), source=""),
        config_fingerprint="local", world=1, n_hosts=1, gpus_per_host=1)
    handle.effective_worker_args = lambda requested: dict(requested)
    model = ModelSpec(
        mesh=MeshSpec(handle=handle, topology_preset="single",
                      attention="TORCH_FLASH", sync_ulysses=False,
                      worker_args=dict(worker_args), auto_gate="first_use"),
        unet_name="model.safetensors")

    class _Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_integrity(self, _key, _artifacts, _commit, _context):
            return ledger_mod.GateLedgerLookup(state, entry, True)

    monkeypatch.setattr(ledger_mod, "GateLedger", _Ledger)
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(model_store_mod, "request_artifact_identity", _identity)
    return gate_identity.authorize_normal_render(
        model, handle, gate_active=False, session_verdict=lambda _token: None,
        resolved_topology=Topology(world=1), resolved_attention="TORCH_FLASH")


def test_a_normal_render_can_actually_reach_the_rung(monkeypatch, tmp_path):
    """Reachability, not refusal: an exact PASS still authorizes the render.

    Without this test a one-token change at the dual-model arm of
    ``authorize_normal_render``, routing every comfy-managed graph into the
    unconditional stock stamp, would disable the feature through the
    KSampler while the refusal tests stay green, because the stock stamp is
    itself a shipped refusal.
    """
    authorization = _normal_authorization(
        monkeypatch, tmp_path, state="pass", entry={"verdict": "PASS"},
        worker_args={"comfy_managed": True, "slab_weights": False,
                     "lora_low_rss": False})
    assert authorization.residency_mode == "required"
    assert authorization.residency_grant is not None
    assert authorization.worker_args["comfy_managed"] is True
    # And the worker-side assertion accepts what the driver just stamped.
    request = {"model": {"loras": []},
               "_dgxm_normal_residency_mode": authorization.residency_mode,
               "_dgxm_normal_residency_grant": authorization.residency_grant}
    assert mesh_residency.assert_normal_render_residency_mode(
        request, {"comfy_managed": True, "slab_weights": False,
                  "lora_low_rss": False}) == "required"


def test_the_grants_capability_context_carries_the_rung(monkeypatch, tmp_path):
    """The PASS the render runs under is bound to a context that names the
    rung, so a later classic render cannot reuse it."""
    authorization = _normal_authorization(
        monkeypatch, tmp_path, state="pass", entry={"verdict": "PASS"},
        worker_args={"comfy_managed": True, "slab_weights": False,
                     "lora_low_rss": False})
    context = authorization.residency_grant["capability_context"]
    assert context["worker_args"]["comfy_managed"] is True


def test_a_normal_render_without_a_pass_takes_the_refusing_stock_stamp(
    monkeypatch, tmp_path,
):
    """The other half of the same path: no exact PASS means the stock stamp,
    which the worker-side assertion then refuses with the typed message."""
    authorization = _normal_authorization(
        monkeypatch, tmp_path, state="unknown", entry=None,
        worker_args={"comfy_managed": True, "slab_weights": False,
                     "lora_low_rss": False})
    assert authorization.residency_mode == "stock"
    request = {"model": {"loras": []},
               "_dgxm_normal_residency_mode": authorization.residency_mode}
    with pytest.raises(residency_mode.ComfyManagedResidencyError):
        mesh_residency.assert_normal_render_residency_mode(
            request, {"comfy_managed": True, "slab_weights": False,
                      "lora_low_rss": False})


def _fleet_environment(monkeypatch, *, state, effective):
    import dgx_monarch.gate_ledger as ledger_mod
    import dgx_monarch.nodes.gate as gate_mod
    from dgx_monarch.actor import model_store as model_store_mod
    from dgx_monarch.nodes.common import MeshSpec, ModelSpec

    identity = {
        "digest": ledger_mod.artifact_set_signature(["model-sig"]).current,
        "comfy": "commit",
        "artifacts": [{"kind": "diffusion_models", "name": "model.safetensors",
                       "signature": "model-sig"}],
    }

    class _Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_integrity(self, *_args):
            return ledger_mod.GateLedgerLookup(state, None, True)

    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: "/ledger")
    monkeypatch.setattr(ledger_mod, "GateLedger", _Ledger)
    monkeypatch.setattr(
        model_store_mod, "request_artifact_identity", lambda *_a: identity)
    handle = SimpleNamespace(effective_worker_args=lambda args: {**dict(args), **effective})
    mesh = MeshSpec(handle=handle, topology_preset="uly2", attention="TORCH_FLASH",
                    sync_ulysses=True, worker_args={})
    return ModelSpec(mesh=mesh, unet_name="model.safetensors"), handle


def test_fleet_refuses_a_rung_it_cannot_downgrade(monkeypatch):
    """Fleet cannot run a ceremony and cannot turn a bootstrap policy off for
    one call, so its fail-closed branch has nothing to downgrade to. That is
    why the refusal is class P and not class U: from inside a Fleet call there
    is no bypass."""
    from dgx_monarch.nodes.fleet_policy import _fleet_worker_policy

    spec, handle = _fleet_environment(
        monkeypatch, state="unknown", effective={"comfy_managed": True})
    with pytest.raises(residency_mode.ComfyManagedResidencyError) as excinfo:
        _fleet_worker_policy(spec, handle)
    text = str(excinfo.value)
    assert _tag(text).refusal_class is RefusalClass.PHYSICS
    assert "docs/TROUBLESHOOTING.md #62" in text
    assert "instead" in text
    assert "unknown" in text


def test_fleet_reads_the_effective_policy_not_the_graphs_own(monkeypatch):
    """A cluster.toml-configured rung never appears in the graph's worker_args.

    The risk predicate reads the merged view, so the call counts as risky; a
    refusal that read the graph's dict would then fall through to the stock
    stamp and nothing would refuse.
    """
    from dgx_monarch.nodes.fleet_policy import _fleet_worker_policy

    spec, handle = _fleet_environment(
        monkeypatch, state="unknown", effective={"comfy_managed": True})
    assert "comfy_managed" not in spec.mesh.worker_args
    with pytest.raises(residency_mode.ComfyManagedResidencyError):
        _fleet_worker_policy(spec, handle)


def test_fleet_still_refuses_when_the_merge_itself_raises(monkeypatch):
    """The seed must be the merged view, not the graph's own.

    ``policy`` is reassigned to the merged view only inside the try. If the
    import batch or _effective_worker_args raises above that line, a seed of
    only the graph's dict leaves a cluster.toml-only rung looking classic:
    the refusal never fires, the stock stamp goes out, and no other layer
    catches it, because assert_normal_render_residency_mode returns early for
    fleet jobs. That fails open in the case the merge exists for.
    """
    import dgx_monarch.nodes.gate as gate_mod
    from dgx_monarch.nodes.fleet_policy import _fleet_worker_policy

    spec, handle = _fleet_environment(
        monkeypatch, state="unknown", effective={"comfy_managed": True})
    handle.config = SimpleNamespace(worker_args={"comfy_managed": True})

    def _explode(*_args, **_kwargs):
        raise RuntimeError("ledger import failed")

    monkeypatch.setattr(gate_mod, "_effective_worker_args", _explode)
    assert "comfy_managed" not in spec.mesh.worker_args
    with pytest.raises(residency_mode.ComfyManagedResidencyError):
        _fleet_worker_policy(spec, handle)


def test_the_fleet_seed_survives_a_handle_with_no_config(monkeypatch):
    """The merged seed must not itself become a new way to raise."""
    from dgx_monarch.nodes.fleet_policy import _fleet_worker_policy

    spec, handle = _fleet_environment(
        monkeypatch, state="unknown", effective={"slab_weights": "auto"})
    assert not hasattr(handle, "config")
    authorization = _fleet_worker_policy(spec, handle)
    assert "comfy_managed" not in authorization.worker_args


def test_fleet_takes_an_exact_pass_for_the_comfy_managed_context(monkeypatch):
    """An exact PASS for the exact fleet capability context, which contains
    comfy_managed, authorizes the call: the same contract slab has."""
    from dgx_monarch.nodes.fleet_policy import _fleet_worker_policy

    spec, handle = _fleet_environment(
        monkeypatch, state="pass", effective={"comfy_managed": True})
    authorization = _fleet_worker_policy(spec, handle)
    assert authorization.residency_grant is not None


def test_fleet_never_stamps_the_key_onto_a_classic_call(monkeypatch):
    """The fleet fail-closed stamp must not add the comfy_managed key; it never
    writes comfy_managed=False.

    That stamp becomes the pushed worker args, so a comfy_managed=False on a
    live comfy-managed worker would trip the bootstrap latch after dispatch:
    the wrong refusal, at the wrong time.
    """
    from dgx_monarch.nodes.fleet_policy import _fleet_worker_policy

    spec, handle = _fleet_environment(
        monkeypatch, state="unknown", effective={"slab_weights": "auto"})
    authorization = _fleet_worker_policy(spec, handle)
    assert authorization.worker_args == {"slab_weights": False, "lora_low_rss": False}
    assert "comfy_managed" not in authorization.worker_args


def test_the_quarantine_covers_all_three_levers():
    from dgx_monarch.nodes import render_quarantine

    assert render_quarantine._QUARANTINE_LEVERS == (
        "lora_low_rss", "slab_weights", "comfy_managed")


def _quarantine(worker_args, entry=None):
    from dgx_monarch.nodes.common import MeshSpec, ModelSpec, _apply_persisted_quarantine

    mesh = MeshSpec(handle=None, topology_preset="single", attention="TORCH_FLASH",
                    sync_ulysses=True, worker_args=dict(worker_args))
    model = ModelSpec(mesh=mesh, unet_name="model.safetensors")
    levers = _apply_persisted_quarantine(model, entry or {})
    return model.mesh.worker_args, levers


def test_a_quarantine_turns_the_rung_off_when_the_graph_asked_for_it():
    args, levers = _quarantine({"comfy_managed": True, "slab_weights": "auto"})
    assert args["comfy_managed"] is False
    assert args["lora_low_rss"] is False
    assert args["slab_weights"] is False
    assert set(levers) == {"lora_low_rss", "slab_weights", "comfy_managed"}


def test_a_quarantine_does_not_introduce_the_key_on_a_classic_graph():
    """A persisted FAIL applies to classic renders too, so the key stays absent.

    Writing comfy_managed=False unconditionally would change the capability
    context of every deployed post-quarantine stock PASS and re-prove them all.
    """
    args, levers = _quarantine({"slab_weights": "auto", "lora_low_rss": True})
    assert "comfy_managed" not in args
    assert args["slab_weights"] is False
    assert args["lora_low_rss"] is False
    # The caller logs these by name. Naming a lever that was never written is
    # a false log line about the state of somebody's render.
    assert levers == ["lora_low_rss", "slab_weights"]


def test_a_pre_change_fail_row_is_the_exact_case(monkeypatch):
    """A gate FAIL reports only the two levers it can find against, so a
    two-lever row is the exact metadata and must not warn.

    Comparing the row against the three-lever quarantine table would make the
    exact branch unreachable for every row a ceremony can write and turn the
    expansion warning into permanent noise.
    """
    args, levers = _quarantine(
        {"comfy_managed": True},
        entry={"quarantine_levers": ["lora_low_rss", "slab_weights"]})
    assert args["comfy_managed"] is False
    assert len(levers) == 3
    assert residency_mode.quarantine_metadata_exact(
        ["lora_low_rss", "slab_weights"]) is True


def test_the_exact_branch_is_reachable_from_what_the_gate_actually_writes():
    """Read the writer, not a guess about it: nodes/gate_verdict.py appends
    these two names and no others, so those are the rows this predicate must
    accept."""
    source = (SRC / "nodes/gate_verdict.py").read_text()
    assert 'levers.append("slab_weights")' in source
    assert 'levers.append("lora_low_rss")' in source
    assert 'levers.append("comfy_managed")' not in source
    assert residency_mode.quarantine_metadata_exact(
        ["slab_weights", "lora_low_rss"]) is True
    assert residency_mode.quarantine_metadata_exact(
        ["lora_low_rss", "slab_weights", "comfy_managed"]) is True


@pytest.mark.parametrize("reported", [
    None,
    [],
    ["slab_weights"],                       # the cross-mode-only FAIL: partial
    ["lora_low_rss"],
    ["lora_low_rss", "slab_weights", "something_else"],
    ["lora_low_rss", 7],
    "lora_low_rss+slab_weights",
])
def test_a_row_that_does_not_explain_the_quarantine_still_warns(reported):
    assert residency_mode.quarantine_metadata_exact(reported) is False


def test_the_quarantine_risk_predicate_sees_the_rung():
    from dgx_monarch.nodes import render_quarantine

    source = inspect.getsource(render_quarantine._enforce_persisted_quarantine)
    assert "residency_mode.requested" in source or "comfy_managed" in source, (
        "the inline risk predicate must treat a comfy-managed graph as risky, "
        "or a quarantined combination renders under the rung unchecked"
    )


def test_the_projection_blocker_names_the_rung_not_the_slab_widget():
    """The operator did not set slab_weights=off; the comfy_managed widget did.

    Class C requires the refusal to name the real blocker, and this string is
    printed verbatim by the loader-site refusal.
    """
    from dgx_monarch.nodes.consent_projection import projection_blocker

    blocked = projection_blocker(
        {"comfy_managed": True, "slab_weights": False, "lora_low_rss": False})
    assert blocked
    assert "comfy" in blocked.lower()
    assert "the Init node's" not in blocked or "comfy_managed" in blocked
    # The classic wording is untouched for the classic case.
    classic = projection_blocker({"slab_weights": False})
    assert "slab_weights=off" in classic


def test_the_rung_alone_makes_a_first_use_render_gate_worthy(monkeypatch):
    """comfy_managed forces both classic levers off, so a trigger that read
    only those levers would see a risk-free render while authorization still
    demands a PASS for the comfy-managed capability context. first_use would
    then skip the one ceremony that can mint it, and the worker would refuse
    the stock fallback it cannot take (seen on hardware, 2026-08-05). The rung
    is itself the risky lever."""
    import types as _types

    from dgx_monarch.nodes import auto_gate as ag
    from dgx_monarch.nodes import common as common_mod
    from dgx_monarch.nodes import gate as gate_mod

    def build_model(worker_args):
        mesh = _types.SimpleNamespace(
            auto_gate="first_use", worker_args=dict(worker_args),
            topology_preset="auto", attention="TORCH_FLASH",
            sync_ulysses=True, handle=object())
        return _types.SimpleNamespace(mesh=mesh, loras=(),
                                      unet_name="model.safetensors")

    monkeypatch.setattr(
        common_mod, "ensure_live",
        lambda handle: _types.SimpleNamespace(world=2))
    monkeypatch.setattr(
        gate_mod, "_effective_worker_args",
        lambda model, handle: dict(model.mesh.worker_args))
    monkeypatch.setattr(
        gate_mod, "_combo_of",
        lambda model: ("combo", _types.SimpleNamespace(current="sig")))
    monkeypatch.setattr(gate_mod, "gate_capability_context",
                        lambda *a, **k: {"worker_args": {}})
    import dgx_monarch.gate_ledger as ledger_mod
    monkeypatch.setattr(ledger_mod, "comfy_commit", lambda: "commit")

    levers_off = {"slab_weights": False, "lora_low_rss": False}

    # Without the rung, both levers explicitly off is the classic risk-free
    # short-circuit: no ceremony consideration at all.
    assert ag.auto_gate_context(
        build_model(levers_off), "ksampler") is None

    # With the rung requested, the same lever state must reach the ledger
    # path: the returned (state, token) is what triggers the ceremony.
    result = ag.auto_gate_context(
        build_model({**levers_off, "comfy_managed": True}), "ksampler")
    assert result is not None
    state, _token = result
    assert state == "unknown"


def test_a_leverless_ceremony_is_conclusive_only_under_the_rung():
    """Classic doctrine forces a no-lineage ceremony INCONCLUSIVE; under the
    rung the a/b repeat plus in-path per-rank identity are the claim, so the
    a/b verdict decides. On hardware on 2026-08-05 the classic arm fired under
    comfy_managed: two clean proofs, INCONCLUSIVE, then the class P
    authorization refusal the ceremony exists to prevent."""
    from dgx_monarch import residency_mode

    prior = ["no lora stack: lazy swap is not applicable"]
    conclusive, reasons = residency_mode.leverless_ceremony_verdict(
        {"comfy_managed": True, "slab_weights": False, "lora_low_rss": False},
        list(prior))
    assert conclusive is True
    assert any("rung's own proof" in r for r in reasons)
    assert any("no swap or slab lineage" in r for r in reasons)

    conclusive, reasons = residency_mode.leverless_ceremony_verdict(
        {"slab_weights": False, "lora_low_rss": False}, list(prior))
    assert conclusive is False
    assert any("no LoRA/slab lineage" in r for r in reasons)
    # The string trap: a truthy string never turns the rung on.
    conclusive, _ = residency_mode.leverless_ceremony_verdict(
        {"comfy_managed": "on"}, [])
    assert conclusive is False
