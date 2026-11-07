"""Gate authority needs every rank on the driver's torch and CUDA build.

The review's case: a PASS is established, then a remote Worker is recreated
with only its torch changed. The driver's own context is unchanged, so the
driver alone would still find the PASS. Each rank now reports its build at
setup and in every Gate provenance row, the driver refuses Gate authority when
a report differs or lacks the build, and each Worker checks a grant's build
against its own (that half lives beside the other Worker grant checks in
test_runtime_remediation.py).
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from dgx_monarch import box_names, gate_runtime, mesh_residency, worker_runtime
from dgx_monarch.mesh_setup import SetupToken
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag

DRIVER = {"torch": "0.0.1+proven", "cuda": "1.1"}
MOVED = {"torch": "0.0.2+moved", "cuda": "1.2"}


@pytest.fixture(autouse=True)
def driver_build(monkeypatch):
    monkeypatch.setattr(gate_runtime, "runtime_build_identity", lambda: dict(DRIVER))
    monkeypatch.setattr(box_names, "_is_head", lambda name: name == "head-host")


def _pair(*builds, reported=True):
    """A head and a sibling, one GPU each, with each rank's setup report."""
    handle = SimpleNamespace(
        config=SimpleNamespace(
            worker_args={}, source="/configs/cluster.toml",
            hosts=(SimpleNamespace(name="head-host", address="10.0.0.1"),
                   SimpleNamespace(name="sibling-host", address="10.0.0.2"))),
        config_fingerprint="config-a", world=2, n_hosts=2, gpus_per_host=1)
    if reported:
        handle.worker_capabilities = [
            {"rank": rank, "host": f"host-{rank}", **({} if build is None else {"runtime": build})}
            for rank, build in enumerate(builds)]
    return handle


def test_a_pair_on_the_drivers_build_keeps_the_same_context():
    """The context string does not change, so a matched pair keeps its PASS rows."""
    matched = mesh_residency.physical_capability_context(_pair(DRIVER, DRIVER))
    unreported = mesh_residency.physical_capability_context(_pair(reported=False))
    assert matched["runtime"] == DRIVER
    assert matched == unreported


def test_a_sibling_on_another_build_refuses_gate_authority_naming_each_box():
    with pytest.raises(worker_runtime.WorkerRuntimeMismatch) as raised:
        mesh_residency.physical_capability_context(_pair(DRIVER, MOVED))
    text = str(raised.value)
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert "the driver runs torch 0.0.1+proven with CUDA 1.1" in text
    assert "the head runs torch 0.0.1+proven with CUDA 1.1" in text
    assert "the sibling runs torch 0.0.2+moved with CUDA 1.2" in text
    assert "Use one torch and CUDA build on every box" in text
    assert "docs/TROUBLESHOOTING.md #8" in text
    # Plain names only: no host name, address or rank number.
    assert "sibling-host" not in text and "10.0.0.2" not in text and "rank" not in text


def test_a_report_that_lacks_the_build_refuses_too():
    """Missing identity never authorizes: an older Worker reports no build."""
    with pytest.raises(worker_runtime.WorkerRuntimeMismatch, match="the sibling reports no torch build"):
        mesh_residency.physical_capability_context(_pair(DRIVER, None))


def test_a_report_list_that_misses_a_rank_refuses_and_names_the_silent_box():
    """Defence in depth beside each Worker's own grant check: a rank of the
    world that sent no setup report at all does not count as a match."""
    handle = _pair(DRIVER)
    with pytest.raises(worker_runtime.WorkerRuntimeMismatch,
                       match="the sibling sent no report of its build"):
        mesh_residency.physical_capability_context(handle)


def test_a_handle_with_no_setup_reports_yet_makes_no_claim():
    """Render submit authorizes before it sets up; the dispatch check after
    setup, the Worker's grant check and the ceremony rows answer again."""
    assert mesh_residency.physical_capability_context(_pair(reported=False))["runtime"] == DRIVER


def test_every_capability_context_refuses_the_mixed_pair():
    import dgx_monarch.nodes.gate as gate_mod

    handle = _pair(DRIVER, MOVED)
    model = SimpleNamespace(mesh=SimpleNamespace(
        handle=handle, worker_args={"lora_low_rss": True},
        topology_preset="uly2", attention="TORCH_FLASH", sync_ulysses=True))
    for build in (gate_mod.gate_capability_context, gate_mod.fleet_residency_capability_context):
        with pytest.raises(worker_runtime.WorkerRuntimeMismatch):
            build(model, handle)


def test_each_rank_reports_its_build_at_setup(monkeypatch):
    from dgx_monarch.actor import worker_env
    from dgx_monarch.adapters import sol_attention

    distributed = types.ModuleType("xfuser.core.distributed")
    for name in ("get_classifier_free_guidance_rank", "get_data_parallel_rank",
                 "get_sequence_parallel_rank"):
        setattr(distributed, name, lambda: 0)
    monkeypatch.setitem(sys.modules, "xfuser.core.distributed", distributed)
    monkeypatch.setattr(sol_attention, "sol_attn_available", lambda: False)
    worker = SimpleNamespace(rank=1, world=2, topology={"ulysses": 2})
    assert worker_env.setup_info(worker)["runtime"] == DRIVER


def _attest(handle, rows):
    from dgx_monarch import runtime_provenance
    from dgx_monarch.nodes.gate_provenance import ProofCohortAttestor

    source = runtime_provenance.cached_dgx_source_manifest_sha256()
    for row in rows:
        row.setdefault("source_manifest_sha256", source)
    handle.call_all = lambda method, *args, **kwargs: rows
    ProofCohortAttestor()("pre", handle, SetupToken(4, ("setup",), ("worker",)))


def _rows(*builds):
    return [{"rank": rank, "world": 2, "setup_generation": 4,
             **({} if build is None else {"runtime": build})}
            for rank, build in enumerate(builds)]


def test_the_ceremony_attests_every_ranks_build_before_any_proof(monkeypatch):
    """A ceremony on a mixed pair would record a PASS under the driver's build
    that a later matched pair reuses; its provenance rows stop it first."""
    from dgx_monarch import runtime_provenance
    from dgx_monarch.nodes.gate_provenance import GateProvenanceError

    monkeypatch.setattr(runtime_provenance, "cached_dgx_source_manifest_sha256", lambda: "a" * 64)
    _attest(_pair(reported=False), _rows(DRIVER, DRIVER))
    with pytest.raises(GateProvenanceError, match=r"the sibling runs torch 0\.0\.2"):
        _attest(_pair(reported=False), _rows(DRIVER, MOVED))
    with pytest.raises(GateProvenanceError, match="the sibling reports no torch build"):
        _attest(_pair(reported=False), _rows(DRIVER, None))


def _normal_authorization(monkeypatch, tmp_path, handle):
    import dgx_monarch.gate_ledger as ledger_mod
    import dgx_monarch.nodes.gate as gate_mod
    from dgx_monarch.actor import model_store as model_store_mod
    from dgx_monarch.nodes import gate_identity
    from dgx_monarch.nodes.common import MeshSpec, ModelSpec
    from dgx_monarch.topology import Topology

    def _identity(unet_name, loras):
        return {"digest": ledger_mod.artifact_set_signature(["model-sig"]).current,
                "comfy": "known",
                "artifacts": [{"kind": "diffusion_models", "name": unet_name,
                               "signature": "model-sig"}]}

    class _Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_integrity(self, *_args):
            return ledger_mod.GateLedgerLookup("pass", {"verdict": "PASS"}, True)

    handle.effective_worker_args = lambda requested: dict(requested)
    model = ModelSpec(
        mesh=MeshSpec(handle=handle, topology_preset="uly2", attention="TORCH_FLASH",
                      sync_ulysses=False, worker_args={"slab_weights": True},
                      auto_gate="first_use"),
        unet_name="model.safetensors")
    monkeypatch.setattr(ledger_mod, "GateLedger", _Ledger)
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(model_store_mod, "request_artifact_identity", _identity)
    return gate_identity.authorize_normal_render(
        model, handle, gate_active=False, session_verdict=lambda _token: None,
        resolved_topology=Topology(ulysses=2, world=2), resolved_attention="TORCH_FLASH")


def test_a_cached_pass_on_a_mixed_pair_refuses_instead_of_granting(monkeypatch, tmp_path):
    """The review's trigger at the driver: the PASS is there and the driver's
    context matches it, but the sibling's build moved. No grant, and no quiet
    fall back to stock: the user sees what to fix."""
    granted = _normal_authorization(monkeypatch, tmp_path, _pair(DRIVER, DRIVER))
    assert granted.residency_mode == "required" and granted.residency_grant is not None
    with pytest.raises(worker_runtime.WorkerRuntimeMismatch, match="the sibling runs"):
        _normal_authorization(monkeypatch, tmp_path, _pair(DRIVER, MOVED))


def test_fleet_refuses_the_mixed_pair_too(monkeypatch):
    import dgx_monarch.gate_ledger as ledger_mod
    import dgx_monarch.nodes.gate as gate_mod
    from dgx_monarch.actor import model_store as model_store_mod
    from dgx_monarch.nodes.common import MeshSpec, ModelSpec
    from dgx_monarch.nodes.fleet_policy import _fleet_worker_policy

    identity = {"digest": ledger_mod.artifact_set_signature(["model-sig"]).current,
                "comfy": "commit",
                "artifacts": [{"kind": "diffusion_models", "name": "model.safetensors",
                               "signature": "model-sig"}]}

    class _Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_integrity(self, *_args):
            return ledger_mod.GateLedgerLookup("pass", {"verdict": "PASS"}, True)

    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: "/ledger")
    monkeypatch.setattr(ledger_mod, "GateLedger", _Ledger)
    monkeypatch.setattr(model_store_mod, "request_artifact_identity", lambda *_a: identity)
    handle = _pair(DRIVER, MOVED)
    handle.effective_worker_args = lambda args: {**dict(args), "slab_weights": True}
    spec = ModelSpec(mesh=MeshSpec(handle=handle, topology_preset="uly2", attention="TORCH_FLASH",
                                   sync_ulysses=True, worker_args={}),
                     unet_name="model.safetensors")
    with pytest.raises(worker_runtime.WorkerRuntimeMismatch, match="the sibling runs"):
        _fleet_worker_policy(spec, handle)


def test_the_persisted_quarantine_read_does_not_hide_the_mixed_pair(monkeypatch, tmp_path):
    """Render submit reads the quarantine first; its broad handler would turn
    the refusal into a silent stock dispatch."""
    import dgx_monarch.gate_ledger as ledger_mod
    import dgx_monarch.nodes.gate as gate_mod
    from dgx_monarch.nodes import render_quarantine
    from dgx_monarch.nodes.common import MeshSpec, ModelSpec
    from dgx_monarch.topology import Topology

    monkeypatch.setattr(gate_mod, "_combo_of", lambda _model: (
        "k", ledger_mod.artifact_set_signature(["model-sig"])))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    handle = _pair(DRIVER, MOVED)
    handle.effective_worker_args = lambda args: dict(args)
    model = ModelSpec(
        mesh=MeshSpec(handle=handle, topology_preset="uly2", attention="TORCH_FLASH",
                      sync_ulysses=False, worker_args={"slab_weights": True}),
        unet_name="model.safetensors")
    with pytest.raises(worker_runtime.WorkerRuntimeMismatch):
        render_quarantine._enforce_persisted_quarantine(
            model, handle, resolved_topology=Topology(ulysses=2, world=2),
            resolved_attention="TORCH_FLASH")


def _first_use_model(monkeypatch, tmp_path, preset):
    """The default first-use auto gate's lookup, with a PASS in the ledger."""
    import dgx_monarch.gate_ledger as ledger_mod
    import dgx_monarch.nodes.gate as gate_mod
    from dgx_monarch.nodes import auto_gate, common
    from dgx_monarch.nodes.common import MeshSpec, ModelSpec

    class _Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_integrity(self, *_args):
            return ledger_mod.GateLedgerLookup("pass", {"verdict": "PASS"}, True)

    monkeypatch.setattr(ledger_mod, "GateLedger", _Ledger)
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(gate_mod, "_combo_of", lambda _model: (
        "k", ledger_mod.artifact_set_signature(["model-sig"])))
    monkeypatch.setattr(common, "ensure_live", lambda handle: handle)
    # The KSampler's lookup without a latent, so the explicit preset resolves
    # the topology and the runtime check is what answers.
    monkeypatch.setattr(common, "_auto_gate_context", lambda model, kind, _latent, cfg: (
        auto_gate.auto_gate_context(model, kind, None, cfg)))
    handle = _pair(DRIVER, MOVED)
    handle.defunct = False
    handle.effective_worker_args = lambda requested: dict(requested)
    return ModelSpec(
        mesh=MeshSpec(handle=handle, topology_preset=preset, attention="TORCH_FLASH",
                      sync_ulysses=False, worker_args={}, auto_gate="first_use"),
        unet_name="model.safetensors")


@pytest.mark.parametrize("preset", ["uly2", "uly2+fsdp"])
def test_the_first_use_auto_gate_refuses_the_mixed_pair_instead_of_going_stock(
        monkeypatch, tmp_path, preset):
    """The Init node's default path. Swallowed, the refusal became an ERROR
    verdict, a session quarantine and a quiet stock render; under FSDP it was
    rewrapped without the box names or the fix."""
    from dgx_monarch.nodes import auto_gate

    model = _first_use_model(monkeypatch, tmp_path, preset)
    with pytest.raises(worker_runtime.WorkerRuntimeMismatch, match="the sibling runs"):
        auto_gate.auto_gate_required(model, "ksampler")
    with pytest.raises(worker_runtime.WorkerRuntimeMismatch, match="the sibling runs") as raised:
        auto_gate.maybe_auto_gate(model, {"kind": "ksampler"}, {}, 1.0, 2)
    assert "Use one torch and CUDA build on every box" in str(raised.value)
    assert model.mesh.worker_args == {}


def _fleet_dispatch_handle(monkeypatch):
    from dgx_monarch import mesh_safety
    from dgx_monarch.actor import model_store

    monkeypatch.setattr(model_store, "request_artifact_identity",
                        lambda name, loras: {"digest": "d", "comfy": "c", "artifacts": []})
    monkeypatch.setattr(mesh_safety, "assert_artifact_parity", lambda results, expected: None)
    handle = _pair(DRIVER, MOVED)
    handle.setup_generation = 3
    handle.active_worker_args = {"slab_weights": False, "lora_low_rss": False}
    worker = SimpleNamespace(artifact_identity=SimpleNamespace(call_one=lambda specs: "future"))
    handle.workers = SimpleNamespace(
        extent=SimpleNamespace(labels=["hosts", "gpus"]), slice=lambda **_kw: worker)
    handle._await_or_evict = lambda future, timeout_s: {"ok": True}
    return handle


def test_a_stock_fleet_job_on_a_mixed_pair_still_dispatches(monkeypatch):
    """Option 1 takes Gate authority away from a mixed pair, not stock
    residency: a Fleet job with no grant builds no Gate context."""
    from dgx_monarch import mesh_rpc

    handle = _fleet_dispatch_handle(monkeypatch)
    request = {"model": {"unet_name": "m.safetensors", "loras": []}, "_dgxm_fleet_job": True}
    mesh_rpc.verify_request_artifacts(handle, request, worker_index=1)


def test_a_fleet_job_with_a_grant_on_a_mixed_pair_refuses(monkeypatch):
    from dgx_monarch import mesh_rpc

    handle = _fleet_dispatch_handle(monkeypatch)
    request = {"model": {"unet_name": "m.safetensors", "loras": []}, "_dgxm_fleet_job": True,
               "_dgxm_fleet_residency_grant": {"capability_context": {}}}
    with pytest.raises(worker_runtime.WorkerRuntimeMismatch, match="the sibling runs"):
        mesh_rpc.verify_request_artifacts(handle, request, worker_index=1)
