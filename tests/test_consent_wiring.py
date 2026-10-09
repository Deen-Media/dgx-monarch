"""The consent subsystem's module boundaries, driven end to end.

Each test crosses a boundary that one module's own tests cannot: a real slab
certificate folded into a v9 row, a worker-raised refusal turned into a panel
card by the driver frame that observes it, a granted memo projected into the
worker args a render dispatches with, and the two quarantine rules that outrank
all of them.
"""
from __future__ import annotations

import ast
import json
import struct
import sys
import types
from pathlib import Path

import pytest
import torch

from dgx_monarch import (
    consent_descriptor,
    consent_pending,
    consent_store,
    gate_audit,
    mesh_residency,
    mesh_safety,
    refusal,
)
from dgx_monarch.actor import store_residency
from dgx_monarch.actor.slab import WeightSlab
from dgx_monarch.capacity_fit import StockFit
from dgx_monarch.nodes import (
    consent_observe,
    consent_projection,
    consent_quarantine,
    consent_rescue,
    gate_identity,
)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "dgx_monarch"

DOES_NOT_FIT = StockFit(False, True, 62 * 2**30, 47 * 2**30)


def _write_safetensors(path, tensors):
    header, blobs, off = {}, [], 0
    for name, tensor in tensors.items():
        raw = tensor.detach().reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": "BF16", "shape": list(tensor.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    encoded = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for raw in blobs:
            handle.write(raw)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "diffusion_models" / "h3_fl2va_bf16.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_safetensors(path, {"blocks.0.w": torch.arange(4096, dtype=torch.bfloat16),
                              "blocks.0.b": torch.ones(64, dtype=torch.bfloat16)})
    return path


@pytest.fixture(autouse=True)
def _rig(monkeypatch, tmp_path):
    """A private consent store, a private ledger, and a stub folder_paths."""
    consent_pending.clear_all()
    monkeypatch.setattr(consent_store, "MEMO_PATH", str(tmp_path / "consent_memo.json"))
    for spec in consent_pending.KIND_SPECS.values():
        monkeypatch.delenv(spec.env_var, raising=False)
    monkeypatch.delenv(consent_pending.AUTO_RESCUE_ENV, raising=False)
    output = tmp_path / "output"
    output.mkdir(parents=True, exist_ok=True)
    fp = types.ModuleType("folder_paths")
    fp.get_full_path = lambda folder, name: (  # type: ignore[attr-defined]
        str(tmp_path / folder / name) if (tmp_path / folder / name).exists() else None)
    fp.get_output_directory = lambda: str(output)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    yield output
    consent_pending.clear_all()


def test_a_real_slab_certificate_folds_into_a_v9_capacity_row(checkpoint):
    """A real certificate feeds the fold: a hand-built literal in the fold's
    shape cannot show that any producer emits that shape."""
    slab = WeightSlab(str(checkpoint))
    try:
        wire = slab.certificate.public()
    finally:
        slab.close()
    from dgx_monarch.gate_audit_vocab import CERTIFICATE_WIRE_KEYS

    for key in CERTIFICATE_WIRE_KEYS:
        assert key in wire, f"a real certificate is missing the wire key {key}"

    folded = gate_audit.fold_rank_certificates([{"certificate": wire}], 1)
    assert folded is not None
    assert folded["complete"] is True
    assert folded["tensors_verified"] == folded["tensors_total"] == wire["tensor_count"]

    row = gate_audit.build_capacity_row(
        combo_key="a" * 24,
        capability_context={"worker_args": {"slab_weights": True}},
        certificate=folded,
        measured={"probe": "worker_stock_load", "checkpoint_bytes": 66_280_487_368,
                  "mem_available_bytes": 50 * 2**30, "headroom_bytes": -11 * 2**30},
        certificate_origin="rescue_load", unet_name="h3_fl2va_bf16.safetensors",
        file_identity="1:2:3:4:5", consent_id="b" * 32)
    assert row.verdict == gate_audit.CAPACITY_VERDICT
    assert row.detail["certificate"]["digest"] == folded["digest"]


def test_two_ranks_on_two_boxes_still_fold(checkpoint):
    """file_identity differs by construction across hosts; the binding is
    content derived, so a cross-Spark leg still produces one row."""
    slab = WeightSlab(str(checkpoint))
    try:
        wire = slab.certificate.public()
    finally:
        slab.close()
    sibling = dict(wire, file_identity="99:99:99:99:99")
    folded = gate_audit.fold_rank_certificates(
        [{"certificate": wire}, {"certificate": sibling}], 2)
    assert folded is not None and folded["ranks_certified"] == 2


def _offer(checkpoint, **overrides):
    kwargs = {
        "path": str(checkpoint),
        "unet_name": "h3_fl2va_bf16.safetensors",
        "model_options": {},
        "slab_weights": "auto",
        "slab_capable_path": True,
        "lora_low_rss": True,
        "fsdp_launch": False,
        "blocked_reason": "",
        "authoritative_slab_retry": False,
        "memoized_family": lambda _path: None,
        "vouched_families": frozenset({"krea2"}),
        "fit_probe": lambda _path, _options: DOES_NOT_FIT,
        "node_options": {},
        "lora_stack": [],
    }
    kwargs.update(overrides)
    with pytest.raises(consent_descriptor.SlabResidencyRescueOffer) as raised:
        store_residency.resolve(**kwargs)
    return raised.value


def test_a_worker_raised_offer_becomes_a_panel_card(checkpoint):
    """The descriptor keeps the decision on the worker and the request on the driver."""
    exc = _offer(checkpoint)
    card = consent_observe.observe_refusal(RuntimeError(f"ActorError: {exc}"))
    assert card is not None
    assert card["kind"] == "rescue-slab" and card["class"] == "C"
    assert card["primary"]["action"] == "accept"
    assert card["numbers"], "a class C card must name numbers"
    assert consent_pending.pending_count() == 1
    entry = consent_pending.peek_pending(card["key"], card["id"])
    assert entry is not None
    _combo, signature = consent_rescue.combo_identity(
        "h3_fl2va_bf16.safetensors", {}, str(checkpoint))
    assert entry.descriptor.artifacts_legacy == signature.legacy
    assert entry.descriptor.artifacts_legacy_complete is signature.legacy_complete


def test_the_card_a_worker_raised_is_keyed_where_the_driver_looks_it_up(checkpoint):
    """A memo written from a worker-raised card must be found by the driver
    sites that resolve consents, or the click grants nothing."""
    exc = _offer(checkpoint)
    card = consent_observe.observe_refusal(exc)
    assert card is not None
    assert card["key"] == consent_store.memo_key(
        "rescue-slab", str(checkpoint),
        consent_rescue.memo_context("h3_fl2va_bf16.safetensors", {}, ()))


def test_a_certificate_failure_never_becomes_a_card(checkpoint):
    from dgx_monarch.actor.slab_certificate import SlabCertificateError

    assert consent_observe.observe_refusal(
        SlabCertificateError("slab byte-verify FAILED for x")) is None
    assert consent_pending.pending_count() == 0


def test_the_offer_carries_its_numbers_over_the_wire(checkpoint):
    """The ceremony's capacity branch reads them from the message text, because
    Monarch's text wrapping drops the exception's attributes."""
    exc = _offer(checkpoint)
    measured = gate_audit.parse_measured(RuntimeError(f"ActorError: {exc}"))
    assert measured is not None
    assert measured["checkpoint_bytes"] == DOES_NOT_FIT.size_bytes
    assert measured["mem_available_bytes"] == DOES_NOT_FIT.avail_bytes
    assert measured["headroom_bytes"] < 0


def test_both_ladder_arms_carry_their_class_tag(checkpoint):
    offer = refusal.parse_refusal_tag(str(_offer(checkpoint)))
    assert offer is not None and offer.refusal_class.value == "C"
    assert offer.guard == "stock_load_preflight" and offer.waivable
    assert refusal.is_waivable(str(_offer(checkpoint)))

    with pytest.raises(mesh_safety.StockLoadCapacityError) as spent:
        store_residency.resolve(
            path=str(checkpoint), unet_name="h3_fl2va_bf16.safetensors", model_options={},
            slab_weights=False, slab_capable_path=True, lora_low_rss=True, fsdp_launch=False,
            blocked_reason="", authoritative_slab_retry=False,
            memoized_family=lambda _p: None, vouched_families=frozenset(),
            fit_probe=lambda _p, _o: DOES_NOT_FIT)
    tag = refusal.parse_refusal_tag(str(spent.value))
    assert tag is not None and tag.refusal_class.value == "C" and not tag.waivable


def test_a_quarantined_combination_names_the_quarantine_not_a_lever(checkpoint):
    """A persisted quarantine forces both levers off, so an arm that tested
    lora_low_rss first would send the user to turn a quarantined lever on."""
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        store_residency.resolve(
            path=str(checkpoint), unet_name="h3_fl2va_bf16.safetensors", model_options={},
            slab_weights=False, slab_capable_path=True, lora_low_rss=False,
            fsdp_launch=False, blocked_reason="", authoritative_slab_retry=False,
            memoized_family=lambda _p: None, vouched_families=frozenset(),
            fit_probe=lambda _p, _o: DOES_NOT_FIT)
    message = str(raised.value)
    assert "quarantine" in message
    assert "identity ceremony" in message


class _Mesh:
    def __init__(self, worker_args=None, handle=None):
        self.worker_args = dict(worker_args or {})
        self.handle = handle
        self.topology_preset = "auto"
        self.auto_gate = "first_use"
        self.attention = "TORCH_FLASH"
        self.sync_ulysses = True


class _Model:
    def __init__(self, mesh, unet_name="h3_fl2va_bf16.safetensors", loras=()):
        self.mesh = mesh
        self.unet_name = unet_name
        self.options = {}
        self.loras = tuple(loras)

    def request_dict(self):
        return {"unet_name": self.unet_name, "options": dict(self.options),
                "loras": [dict(entry) for entry in self.loras]}


def _grant(checkpoint, unet_name="h3_fl2va_bf16.safetensors",
           capability_context=None):
    context = consent_rescue.memo_context(unet_name, {}, ())
    combo, signature = consent_rescue.combo_identity(unet_name, {}, str(checkpoint))
    key = consent_store.memo_key("rescue-slab", str(checkpoint), context)
    record = consent_store.ConsentRecord(
        id="c" * 32, kind="rescue-slab", target_guard="stock_load_preflight",
        waiver_class="C", artifact=unet_name, path=str(checkpoint),
        file_identity=consent_store.file_identity(str(checkpoint)),
        memo_context=context, granted_at="2026-08-04 12:00:00", granted_epoch=0.0,
        consent_source="panel", reason="stock cannot fit this checkpoint",
        ledger_key="audit:waiver:rescue-slab:x", combo_key=combo,
        artifacts=signature.current, artifacts_legacy=signature.legacy,
        artifacts_legacy_complete=signature.legacy_complete,
        capability_context=capability_context, unet_name=unet_name)
    consent_store.grant(record, key)
    return key


def test_a_granted_consent_is_projected_into_the_render_worker_args(checkpoint):
    _grant(checkpoint)
    model = _Model(_Mesh({"slab_weights": "auto"}))
    assert consent_projection.project_for_render(model) is True
    assert model.mesh.worker_args["slab_weights"] is True


def test_a_revoked_consent_projects_nothing(checkpoint):
    key = _grant(checkpoint)
    consent_store.revoke(key)
    model = _Model(_Mesh({"slab_weights": "auto"}))
    assert consent_projection.project_for_render(model) is False
    assert model.mesh.worker_args["slab_weights"] == "auto"


def test_an_unprovable_ceremony_keeps_a_consented_slab_load(checkpoint):
    """The ceremony cannot prove what stock residency cannot load, so forcing
    the render back to stock causes the out-of-memory instead of avoiding it.

    The exemption keeps lora_low_rss on too, because slab residency requires
    it: slab_weights=True beside lora_low_rss=False is not a narrower
    quarantine but a residency the worker's own defaults turn back to stock,
    while the refusal blames the Init widget.
    """
    _grant(checkpoint)
    model = _Model(_Mesh({"slab_weights": True, "lora_low_rss": True}))
    gate_identity.quarantine_unproven_paths(model)
    assert model.mesh.worker_args["slab_weights"] is True
    assert model.mesh.worker_args["lora_low_rss"] is True


def test_a_consented_render_with_a_lora_stack_still_quarantines_both(checkpoint):
    """The exemption is all of the residency it authorizes or none of it, and
    a LoRA stack is a second unproven path the consent says nothing about."""
    _grant(checkpoint)
    model = _Model(_Mesh({"slab_weights": True, "lora_low_rss": True}),
                   loras=({"name": "a.safetensors"},))
    gate_identity.quarantine_unproven_paths(model)
    assert model.mesh.worker_args == {"slab_weights": False, "lora_low_rss": False}


def test_without_a_consent_an_unprovable_ceremony_still_forces_both_off():
    model = _Model(_Mesh({"slab_weights": True, "lora_low_rss": True}))
    gate_identity.quarantine_unproven_paths(model)
    assert model.mesh.worker_args == {"slab_weights": False, "lora_low_rss": False}


def test_a_persisted_gate_fail_revokes_the_memo_and_blocks_the_projection(checkpoint, _rig):
    """Quarantine outranks consent, and the panel must not keep showing a live
    card for a path the gate has proven unsafe."""
    from dgx_monarch.gate_ledger import GateLedger

    key = _grant(checkpoint)
    combo, artifacts = consent_rescue.combo_identity(
        "h3_fl2va_bf16.safetensors", {}, str(checkpoint))
    GateLedger(str(_rig)).record(
        combo, artifacts, "deadbeef", "FAIL",
        {"quarantine_levers": ["slab_weights", "lora_low_rss"]},
        {"worker_args": {"slab_weights": True}})
    model = _Model(_Mesh({"slab_weights": "auto"}))
    assert consent_projection.project_for_render(model) is False
    assert model.mesh.worker_args["slab_weights"] == "auto"
    assert consent_store.read()["consents"].get(key) is None


def test_a_newer_gate_pass_supersedes_fail_for_consent_projection(checkpoint, _rig):
    """A successful explicit re-test clears the older quarantine boundary."""
    from dgx_monarch.gate_ledger import GateLedger

    key = _grant(checkpoint)
    combo, artifacts = consent_rescue.combo_identity(
        "h3_fl2va_bf16.safetensors", {}, str(checkpoint))
    ledger = GateLedger(str(_rig))
    context = {"worker_args": {"slab_weights": True}}
    ledger.record(
        combo, artifacts, "deadbeef", "FAIL",
        {"quarantine_levers": ["slab_weights", "lora_low_rss"]}, context)
    ledger.record(combo, artifacts, "deadbeef", "PASS", context=context)

    model = _Model(_Mesh({"slab_weights": "auto"}))
    assert consent_projection.project_for_render(model) is True
    assert model.mesh.worker_args["slab_weights"] is True
    assert consent_store.read()["consents"].get(key) is not None


def test_pass_in_one_context_does_not_clear_another_context_fail(checkpoint, _rig):
    from dgx_monarch.gate_ledger import GateLedger

    key = _grant(checkpoint)
    combo, artifacts = consent_rescue.combo_identity(
        "h3_fl2va_bf16.safetensors", {}, str(checkpoint))
    ledger = GateLedger(str(_rig))
    ledger.record(
        combo, artifacts, "deadbeef", "FAIL",
        {"quarantine_levers": ["slab_weights"]}, {"worker_args": {"slab_weights": False}})
    ledger.record(
        combo, artifacts, "deadbeef", "PASS",
        context={"worker_args": {"slab_weights": True}})

    model = _Model(_Mesh({"slab_weights": "auto"}))
    assert consent_projection.project_for_render(model) is False
    assert consent_store.read()["consents"].get(key) is None


def test_context_bound_memo_ignores_an_unrelated_context_fail(checkpoint, _rig):
    from dgx_monarch.gate_ledger import GateLedger

    covered = {"worker_args": {"slab_weights": True}}
    key = _grant(checkpoint, capability_context=covered)
    combo, artifacts = consent_rescue.combo_identity(
        "h3_fl2va_bf16.safetensors", {}, str(checkpoint))
    GateLedger(str(_rig)).record(
        combo, artifacts, "deadbeef", "FAIL",
        {"quarantine_levers": ["slab_weights"]},
        {"worker_args": {"slab_weights": False}})

    model = _Model(_Mesh({"slab_weights": "auto"}))
    assert consent_projection.project_for_render(model) is True
    assert consent_store.read()["consents"].get(key) is not None


@pytest.mark.parametrize("superseder", ["legacy", "wrong_source"])
def test_unscoped_pass_cannot_clear_unscoped_fail_for_live_projection(
        checkpoint, _rig, superseder):
    from dgx_monarch import __version__
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION, GateLedger

    covered = {"worker_args": {"slab_weights": True}}
    key = _grant(checkpoint, capability_context=covered)
    combo, artifacts = consent_rescue.combo_identity(
        "h3_fl2va_bf16.safetensors", {}, str(checkpoint))
    ledger = GateLedger(str(_rig))
    ledger.record(combo, artifacts, "deadbeef", "FAIL")
    detail = ({} if superseder == "legacy" else {
        "gate_protocol": GATE_PROTOCOL_VERSION,
        "dgx_monarch": __version__,
        "dgx_source": "foreign-source",
    })
    ledger.record(combo, artifacts, "deadbeef", "PASS", detail=detail)

    model = _Model(_Mesh({"slab_weights": "auto"}))
    assert consent_projection.project_for_render(model) is False
    assert model.mesh.worker_args["slab_weights"] == "auto"
    assert consent_store.read()["consents"].get(key) is None


@pytest.mark.parametrize("stale_dimension", ["protocol", "package"])
def test_stale_contextual_pass_cannot_clear_fail_for_live_projection(
        checkpoint, _rig, monkeypatch, stale_dimension):
    from dgx_monarch import gate_ledger as ledger_mod

    context = {"worker_args": {"slab_weights": True}}
    key = _grant(checkpoint, capability_context=context)
    combo, artifacts = consent_rescue.combo_identity(
        "h3_fl2va_bf16.safetensors", {}, str(checkpoint))
    ledger = ledger_mod.GateLedger(str(_rig))
    # The FAIL comes from the running release, so the stale PASS is under test.
    # A superseded FAIL retires for both readers:
    # test_a_scoped_fail_from_a_superseded_release_stops_blocking_consent.
    ledger.record(combo, artifacts, "deadbeef", "FAIL", context=context)
    with monkeypatch.context() as stale:
        if stale_dimension == "protocol":
            stale.setattr(ledger_mod, "GATE_PROTOCOL_VERSION",
                          ledger_mod.GATE_PROTOCOL_VERSION - 1)
        else:
            stale.setattr(ledger_mod, "__version__", "stale-package")
        ledger.record(combo, artifacts, "deadbeef", "PASS", context=context)

    model = _Model(_Mesh({"slab_weights": "auto"}))
    assert consent_projection.project_for_render(model) is False
    assert model.mesh.worker_args["slab_weights"] == "auto"
    assert consent_store.read()["consents"].get(key) is None


@pytest.mark.parametrize("terminal", ["INCONCLUSIVE", "RETESTING"])
def test_foreign_source_terminal_cannot_mask_fail_for_live_projection(
        checkpoint, _rig, monkeypatch, terminal):
    from dgx_monarch import runtime_provenance
    from dgx_monarch.gate_ledger import GateLedger

    context = {"worker_args": {"slab_weights": True}}
    key = _grant(checkpoint, capability_context=context)
    combo, artifacts = consent_rescue.combo_identity(
        "h3_fl2va_bf16.safetensors", {}, str(checkpoint))
    ledger = GateLedger(str(_rig))
    ledger.record(combo, artifacts, "deadbeef", "FAIL", context=context)
    if terminal == "RETESTING":
        ledger.begin_retest_required(combo, artifacts, "deadbeef", [context])
    else:
        ledger.record(combo, artifacts, "deadbeef", terminal, context=context)
    monkeypatch.setattr(
        runtime_provenance, "cached_dgx_source_manifest_sha256",
        lambda: "foreign-runtime-source")

    model = _Model(_Mesh({"slab_weights": "auto"}))
    assert consent_projection.project_for_render(model) is False
    assert model.mesh.worker_args["slab_weights"] == "auto"
    assert consent_store.read()["consents"].get(key) is None


def test_unreadable_gate_ledger_denies_projection_without_revoking(
        checkpoint, _rig, monkeypatch):
    from dgx_monarch.gate_ledger import GateLedger

    key = _grant(checkpoint)

    def unreadable(_ledger):
        raise OSError("ledger unavailable")

    monkeypatch.setattr(GateLedger, "entries_with_integrity", unreadable)
    model = _Model(_Mesh({"slab_weights": "auto"}))

    assert consent_projection.project_for_render(model) is False
    assert model.mesh.worker_args["slab_weights"] == "auto"
    assert consent_store.read()["consents"].get(key) is not None


def test_a_silent_residency_gate_carries_its_reason_to_the_ladder():
    """Upstream gates turn slab residency off before the ladder runs, and
    slab_mode_reason says why. A reason that only a log line in another module
    records cannot be named in a class C refusal, so it travels with the mode."""
    from dgx_monarch.actor import worker_env

    assert worker_env.slab_mode_reason({"slab_weights": "auto", "lora_low_rss": True}, {}) == ""
    assert "FSDP" in worker_env.slab_mode_reason(
        {"slab_weights": "auto", "lora_low_rss": True}, {"fsdp": 2})
    assert worker_env.slab_mode_reason(
        {"slab_weights": "auto", "lora_low_rss": True, "compile_dit": True}, {}) == ""
    assert "lora_low_rss" in worker_env.slab_mode_reason(
        {"slab_weights": "auto", "lora_low_rss": False}, {})
    reason = worker_env.slab_mode_reason({"slab_weights": False}, {})
    with pytest.raises(mesh_safety.StockLoadCapacityError, match="worker args"):
        store_residency.resolve(
            path="/models/x.safetensors", unet_name="x.safetensors", model_options={},
            slab_weights=False, slab_capable_path=True, lora_low_rss=True, fsdp_launch=False,
            blocked_reason=reason, authoritative_slab_retry=False,
            memoized_family=lambda _p: None, vouched_families=frozenset(),
            fit_probe=lambda _p, _o: DOES_NOT_FIT)


def test_the_guard_registry_and_the_consent_registry_agree_in_both_directions():
    """A renamed guard or kind cannot land half applied.

    Class K guards carry a family scope, so a waiver granted for one family
    never silences the same guard in another: a raise site's registry entry is
    ``<prefix>:<scope>`` while the kind's default guard is the bare prefix. The
    test asserts that protocol-bound rule (gate_audit_vocab.KIND_GUARD_PREFIXES).
    """
    from dgx_monarch.consent_kinds import KIND_SPECS
    from dgx_monarch.gate_audit_vocab import KIND_GUARD_PREFIXES, check_guard

    for guard, spec in refusal.GUARDS.items():
        assert spec.consent_kind in KIND_SPECS, f"{guard} names an unknown kind"
        assert spec.refusal_class.value == KIND_SPECS[spec.consent_kind].refusal_class
        assert check_guard(spec.consent_kind, guard) == guard
    for kind, spec in KIND_SPECS.items():
        prefix = KIND_GUARD_PREFIXES.get(kind)
        if prefix is not None:
            assert spec.default_guard == prefix
            scoped = [g for g in refusal.GUARDS if g.split(":")[0] == prefix]
            if spec.wired:
                assert scoped, f"{kind} has no guard entry in refusal.GUARDS"
            for guard in scoped:
                assert refusal.GUARDS[guard].refusal_class.value == spec.refusal_class
            continue
        assert spec.default_guard in refusal.GUARDS, f"{kind} names an unknown guard"
        assert refusal.GUARDS[spec.default_guard].refusal_class.value == spec.refusal_class


def test_the_ladder_and_the_registry_spell_the_rescue_one_way():
    from dgx_monarch.consent_kinds import KIND_SPECS

    spec = KIND_SPECS["rescue-slab"]
    assert store_residency._PANEL_ACTION == spec.primary_label
    assert store_residency._RESCUE_SPEC.env_var == spec.env_var
    assert consent_descriptor.ENV_SLAB_RESCUE == spec.env_var
    assert consent_rescue.RESCUE_KIND == spec.kind == consent_descriptor.KIND_RESCUE_SLAB


@pytest.mark.parametrize("module", [
    "refusal.py", "consent_kinds.py", "consent_descriptor.py", "consent_pending.py",
    "consent_store.py", "capacity_fit.py", "accuracy_waiver.py",
])
def test_the_consent_leaves_never_import_the_mesh(module):
    """The adapters, the actor, the nodes and the driver endpoints all import
    these modules, so a mesh, torch or comfy import here loads that runtime into
    every importer (as in test_mesh_safety_remains_a_leaf_without_mesh_imports)."""
    tree = ast.parse((SRC / module).read_text())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            imported.append(base)
            imported.extend(f"{base}.{alias.name}" if base else alias.name
                            for alias in node.names)
    banned = {"mesh", "mesh_rpc", "mesh_setup", "dgx_monarch.mesh", "torch", "comfy"}
    hits = [name for name in imported
            if name in banned or name.split(".")[0] in {"torch", "comfy"}]
    assert not hits, f"{module} must stay a leaf, imports {hits}"


def test_the_projection_runs_for_an_ordinary_latent_not_only_a_packed_one(checkpoint):
    """Driven through the real render entry point, with a plain LATENT.

    `is_direct_nested_tensor` is False for every latent that is not a packed
    Comfy NestedTensor, so a projection placed under that guard would fire only
    for H3-class latents and the click-then-requeue loop would never terminate
    for anything else. Residency is a property of the checkpoint, not of the
    latent's container type.
    """
    from dgx_monarch.nodes import render_preflight

    _grant(checkpoint)
    model = _Model(_Mesh({"slab_weights": "auto"}))
    bound, handle = render_preflight.bind_packed_render_model(
        model, {"samples": torch.zeros(1, 4, 8, 8)}, 1.0,
        resolve_topology=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no topology")),
        ensure_live_fn=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no lifecycle")),
    )
    assert bound is model and handle is None      # the packed guard still returns early
    assert model.mesh.worker_args["slab_weights"] is True


def test_an_explicit_stock_request_is_terminal_for_a_consent(checkpoint):
    """docs/DESIGN.md section 5.9 invariant 2. The operator's widget and both
    quarantine paths write this value, so a consent that overwrote it would be
    the stale driver push that invariant names."""
    _grant(checkpoint)
    model = _Model(_Mesh({"slab_weights": False}))
    assert consent_projection.project_for_render(model) is False
    assert model.mesh.worker_args["slab_weights"] is False


def test_a_consent_never_projects_over_an_explicit_low_rss_off(checkpoint):
    """Slab residency requires low-rss, so publishing slab beside low_rss=off
    would be collapsed back to stock by the worker's own defaults and then
    refused with a message naming a lever the driver itself had just set."""
    _grant(checkpoint)
    model = _Model(_Mesh({"slab_weights": "auto", "lora_low_rss": False}))
    assert consent_projection.project_for_render(model) is False
    assert model.mesh.worker_args["slab_weights"] == "auto"


def test_a_projection_is_taken_back_for_a_checkpoint_it_does_not_cover(checkpoint, tmp_path):
    """One consent, one checkpoint. The policy dict is the fleet's and ComfyUI
    caches it for the session, so a projection never taken back would put every
    later checkpoint on the pre-gate slab path with no consent, no card and no
    ledger row (the explicit slab rung, RUNG_EXPLICIT, consults neither the
    stock fit probe nor the vouched set)."""
    _grant(checkpoint)
    other = tmp_path / "diffusion_models" / "some_other_model.safetensors"
    _write_safetensors(other, {"blocks.0.w": torch.ones(8, dtype=torch.bfloat16)})

    mesh = _Mesh({"slab_weights": "auto"})
    assert consent_projection.project_for_render(_Model(mesh)) is True
    assert mesh.worker_args["slab_weights"] is True

    consent_projection.project_for_render(_Model(mesh, unet_name=other.name))
    assert mesh.worker_args["slab_weights"] == "auto"


def test_a_dual_model_graph_keeps_the_first_loaders_projection(checkpoint, tmp_path):
    """Both checkpoints belong to one graph and one eager-load transaction, so
    the second loader node must not take the first one's projection back
    between the two loads."""
    _grant(checkpoint)
    other = tmp_path / "diffusion_models" / "uncond_model.safetensors"
    _write_safetensors(other, {"blocks.0.w": torch.ones(8, dtype=torch.bfloat16)})
    mesh = _Mesh({"slab_weights": "auto"})
    consent_projection.project_for_render(_Model(mesh))
    consent_projection.project_for_loader(
        mesh, other.name, {}, covered=frozenset({checkpoint.name, other.name}))
    assert mesh.worker_args["slab_weights"] is True


def test_an_environment_grant_writes_its_own_permanent_waiver_row(checkpoint, monkeypatch, _rig):
    """Every bypass writes a permanent row (docs/DESIGN.md section 5.9,
    invariant 1). The click path audits at click time; the headless variable
    has no click, so the row is written where the bypass fires, as
    docs/TROUBLESHOOTING.md #53 promises ("the same grant and the same
    permanent ledger row")."""
    from dgx_monarch.gate_audit_vocab import is_audit_row
    from dgx_monarch.gate_ledger import GateLedger

    consent_projection.reset_grant_audit()
    monkeypatch.setenv(consent_pending.KIND_SPECS["rescue-slab"].env_var, "1")
    mesh = _Mesh({"slab_weights": "auto"})
    assert consent_projection.project_for_render(_Model(mesh)) is True

    rows = [row for row in GateLedger(str(_rig)).entries() if is_audit_row(row)]
    assert [row["verdict"] for row in rows] == ["WAIVER"]
    assert rows[0]["waiver_kind"] == "rescue-slab"
    assert rows[0]["consent_source"] == "env"
    assert rows[0]["action"] == "grant"

    # Once per kind, combination and channel per fleet generation (a recycle
    # clears the set): every lookup scans the whole ledger, so a row per render
    # would slow every later lookup.
    consent_projection.project_for_render(_Model(_Mesh({"slab_weights": "auto"})))
    assert len([row for row in GateLedger(str(_rig)).entries() if is_audit_row(row)]) == 1


def test_a_consented_load_with_no_certificate_refuses_instead_of_warning(checkpoint):
    """Fail closed. Every pre-gate slab load needs a byte-verify certificate,
    and an older worker's snapshot has no `slab_certificate` key, so a missing
    key is how version skew arrives."""
    from dgx_monarch.actor.slab_certificate import SlabCertificateError

    load = consent_rescue.RescueLoad(
        combo_key="a" * 24, artifacts="b" * 64, unet_name=checkpoint.name,
        path=str(checkpoint), consent_id="c" * 32,
        measured={"probe": "worker_stock_load", "checkpoint_bytes": 1,
                  "mem_available_bytes": 1, "headroom_bytes": 0})
    unloaded = []

    class _Handle:
        def call_all(self, endpoint, **kwargs):
            unloaded.append(endpoint)
            return []

    with pytest.raises(SlabCertificateError) as caught:
        consent_rescue.record_rescue_row(
            load, [{"rank": 0, "cond": {"residency": "slab"}}], 1,
            handle=_Handle())
    assert "no byte-verify certificate" in str(caught.value)
    assert refusal.parse_refusal_tag(str(caught.value)).refusal_class.value == "P"
    assert unloaded == ["unload"]


def test_the_uncond_loader_folds_the_uncond_slots_certificate(checkpoint):
    """`ModelStore.snapshot` returns both slots, each with its own certificate.
    Folding "cond" for an uncond load would stamp a permanent row whose digest
    describes one checkpoint and whose file identity names another."""
    slab = WeightSlab(str(checkpoint))
    try:
        wire = slab.certificate.public()
    finally:
        slab.close()
    rows = [{"rank": 0, "cond": {"slab_certificate": None},
             "uncond": {"slab_certificate": wire}}]
    folded = consent_rescue._folded_certificate(
        consent_rescue.RescueLoad(combo_key="", artifacts="", unet_name="x", path="",
                                  consent_id="", measured={}),
        rows, 1, "uncond")
    assert folded is not None and folded["digest"]
    assert consent_rescue._folded_certificate(
        consent_rescue.RescueLoad(combo_key="", artifacts="", unet_name="x", path="",
                                  consent_id="", measured={}),
        rows, 1, "cond") is None


def test_a_worker_raised_card_with_no_memo_context_is_not_guessed(checkpoint):
    """A guessed memo key is worse than no card: the click would succeed, the
    memo would land where no lookup computes it, and the same card would come
    back on every queue forever. Skew fails closed."""
    fit = StockFit(False, True, 62 * 2**30, 47 * 2**30)
    descriptor = store_residency._descriptor(
        unet_name=checkpoint.name, path=str(checkpoint), fit=fit,
        model_options={}, file_identity=lambda _p: "1:2:3:4:5", family_hint="minimax_h3",
        node_options={"weight_dtype": "fp8_e4m3fn"}, lora_stack=[{"name": "a.safetensors"}])
    stripped = consent_descriptor.encode(
        consent_descriptor.replace(descriptor, memo_context={}))
    assert consent_observe.observe_refusal(RuntimeError(stripped)) is None
    assert consent_pending.pending_cards() == []


def test_the_wire_consent_id_is_the_width_the_permanent_row_validates():
    """One id across the refusal, the card, the memo and every row. A width the
    row's own validator refuses is silently replaced by a fresh random one, so
    the wire value and the audited value would differ for every worker card."""
    minted = consent_descriptor.consent_id("rescue-slab", "1:2:3:4:5", "abc")
    assert len(minted) == consent_pending.CONSENT_ID_HEX
    assert consent_pending._usable_consent_id(minted)


def test_the_certificate_fold_records_the_world_it_was_measured_against():
    """`ranks_expected == ranks_certified` is the invariant validate_certificate
    enforces; emitting the fold count for both would make it a comparison of a
    value with itself."""
    from dgx_monarch.gate_audit_evidence import fold_rank_certificates

    one = {"algorithm": "memcmp-tensor-sampled-v1", "digest": "d" * 64, "complete": True,
           "tensors_verified": 2, "tensors_total": 2, "bytes_verified": 8,
           "checkpoint_bytes": 8, "artifact_signature": "s" * 24, "file_identity": "1:2:3:4:5"}
    folded = fold_rank_certificates([{"certificate": one}], 1)
    assert folded is not None and folded["ranks_expected"] == 1
    assert fold_rank_certificates([{"certificate": one}], 2) is None


def test_a_consented_load_that_legitimately_fell_back_to_stock_does_not_refuse(checkpoint):
    """A dtype the slab loader cannot represent falls back to a stock load,
    which takes the stock capacity check on its own, so it fitted. There are no
    slab bytes to certify. Absence of a residency report is still fail-closed:
    that is an older worker, not a fallback."""
    load = consent_rescue.RescueLoad(
        combo_key="a" * 24, artifacts="b" * 64, unet_name=checkpoint.name,
        path=str(checkpoint), consent_id="c" * 32,
        measured={"probe": "worker_stock_load", "checkpoint_bytes": 1,
                  "mem_available_bytes": 1, "headroom_bytes": 0})
    assert consent_rescue.record_rescue_row(
        load, [{"rank": 0, "cond": {"residency": "stock", "slab_certificate": None}}], 1) is False


@pytest.mark.parametrize("results", [
    [{"rank": 0, "cond": {"residency": "stock"}}],
    [{"rank": 0, "cond": {"residency": "stock"}},
     {"rank": 0, "cond": {"residency": "stock"}}],
    [{"rank": 0, "cond": {"residency": "stock"}},
     {"rank": 1, "cond": "stock"}],
])
def test_partial_duplicate_or_malformed_stock_rescue_evidence_is_refused(checkpoint, results):
    """Stock fallback is an all-rank fact, never an observed-subset shortcut."""
    from dgx_monarch.actor.slab_certificate import SlabCertificateError

    load = consent_rescue.RescueLoad(
        combo_key="a" * 24, artifacts="b" * 64, unet_name=checkpoint.name,
        path=str(checkpoint), consent_id="c" * 32, measured={})
    unloaded = []

    class _Handle:
        def call_all(self, endpoint, **_kwargs):
            unloaded.append(endpoint)

    with pytest.raises(SlabCertificateError):
        consent_rescue.record_rescue_row(load, results, 2, handle=_Handle())
    assert unloaded == ["unload"]


def test_duplicate_rank_certificate_cannot_mint_a_capacity_row(checkpoint):
    """Two valid copies of rank zero are not an all-rank slab proof."""
    from dgx_monarch.actor.slab_certificate import SlabCertificateError

    slab = WeightSlab(str(checkpoint))
    try:
        certificate = slab.certificate.public()
    finally:
        slab.close()
    load = consent_rescue.RescueLoad(
        combo_key="a" * 24, artifacts="b" * 64, unet_name=checkpoint.name,
        path=str(checkpoint), consent_id="c" * 32, measured={})
    unloaded = []

    class _Handle:
        def call_all(self, endpoint, **_kwargs):
            unloaded.append(endpoint)

    results = [
        {"rank": 0, "cond": {"residency": "slab", "slab_certificate": certificate}},
        {"rank": 0, "cond": {"residency": "slab", "slab_certificate": certificate}},
    ]
    assert consent_rescue._folded_certificate(load, results, 2, "cond") is None
    with pytest.raises(SlabCertificateError):
        consent_rescue.record_rescue_row(load, results, 2, handle=_Handle())
    assert unloaded == ["unload"]


def test_out_of_order_unique_rank_certificates_still_fold(checkpoint):
    """Cohort validation binds membership, not transport response ordering."""
    slab = WeightSlab(str(checkpoint))
    try:
        certificate = slab.certificate.public()
    finally:
        slab.close()
    load = consent_rescue.RescueLoad(
        combo_key="", artifacts="", unet_name="x", path="", consent_id="", measured={})
    results = [
        {"rank": 1, "cond": {"slab_certificate": certificate}},
        {"rank": 0, "cond": {"slab_certificate": certificate}},
    ]
    folded = consent_rescue._folded_certificate(load, results, 2, "cond")
    assert folded is not None and folded["ranks_certified"] == 2


# Hardware acceptance, 2026-08-04: the card was clicked, the memo was written,
# the projection landed and the eager loader node slab-loaded 61.7 GiB. The
# render then died on the worker with "Slab rescue is unavailable here because
# slab_weights is off in the effective worker args": the no-Gate-PASS fallback
# in authorize_normal_render had set the projected lever back to False on its
# way to dispatch. The tests below keep a granted consent alive through that
# last driver site.

class _Handle:
    world = 1
    n_hosts = 1
    gpus_per_host = 1
    config_fingerprint = "local"

    def __init__(self):
        self.config = types.SimpleNamespace(worker_args={}, hosts=(), source="")

    def effective_worker_args(self, requested):
        return dict(requested)


_WORKER_TOPOLOGY = {"ulysses": 1, "ring": 1, "cfg": 1, "dp": 1, "fsdp": False}
_IDENTITY = {"digest": "b" * 64, "comfy": "known", "artifacts": [{"signature": "b" * 64}]}


def _authorize(model, monkeypatch, tmp_path, state="inconclusive"):
    """Drive authorize_normal_render with a ledger that holds no PASS.

    A checkpoint that only a capacity rescue can place never holds one: the
    cross-residency ceremony has no stock reference to compare against, so
    every consented render arrives in this state.
    """
    from dgx_monarch import gate_ledger
    from dgx_monarch.nodes import gate
    from dgx_monarch.topology import Topology

    artifacts = gate_ledger.ArtifactSetSignature("b" * 64, "b" * 64, True)
    real_ledger = gate_ledger.GateLedger

    class _Ledger:
        """No PASS for the render, but the real rows for the consent path's
        quarantine read: a consent lookup that could not see a persisted FAIL
        would test the exemption without the rule that a quarantine outranks
        every consent."""

        def __init__(self, directory):
            self._real = real_ledger(directory)

        def lookup_with_integrity(self, *_args):
            return gate_ledger.GateLedgerLookup(state, None, False)

        def entries_with_integrity(self):
            return self._real.entries_with_integrity()

        def matching_entry(self, *args, **kwargs):
            return self._real.matching_entry(*args, **kwargs)

    monkeypatch.setattr(
        gate_identity, "capture",
        lambda subject: (subject.request_dict(),
                         mesh_safety.request_combo_key(subject.request_dict()),
                         _IDENTITY, artifacts, {}))
    monkeypatch.setattr(gate_ledger, "GateLedger", _Ledger)
    monkeypatch.setattr(gate, "_ledger_dir", lambda: str(tmp_path))
    return gate_identity.authorize_normal_render(
        model, model.mesh.handle, gate_active=False,
        session_verdict=lambda _token: None,
        resolved_topology=Topology(world=1), resolved_attention="TORCH_FLASH")


def _dispatch(authorization, *, setup_generation=7, uncond=None):
    """The model, attention and residency fields render_submit.py stamps for this dispatch."""
    request = {
        "model": authorization.model_request,
        "sage_kernel": "TORCH_FLASH",
        "sync_ulysses": True,
        "_dgxm_normal_residency_mode": authorization.residency_mode,
    }
    if uncond is not None:
        request["uncond_model"] = uncond
    if authorization.residency_grant is not None:
        grant = dict(authorization.residency_grant)
        grant.update(setup_generation=setup_generation,
                     setup_key=("key", "TORCH_FLASH", True),
                     worker_args_key=("policy",),
                     worker_topology=dict(_WORKER_TOPOLOGY))
        request["_dgxm_normal_residency_grant"] = grant
    return request


def _consented_model(checkpoint, worker_args=None):
    """A render whose card has been clicked, with the projection already in."""
    _grant(checkpoint)
    mesh = _Mesh(worker_args or {"slab_weights": "auto", "lora_low_rss": True},
                 handle=_Handle())
    model = _Model(mesh)
    consent_projection.project_for_render(model)
    return model


def test_the_no_gate_pass_fallback_no_longer_erases_a_granted_rescue(
        checkpoint, monkeypatch, tmp_path):
    """The 2026-08-04 failure above: the projection must reach the dispatch
    snapshot and stay there, under a mode that names what authorized it."""
    model = _consented_model(checkpoint)
    assert model.mesh.worker_args["slab_weights"] is True

    authorization = _authorize(model, monkeypatch, tmp_path)

    assert authorization.residency_mode == mesh_residency.RESIDENCY_MODE_CAPACITY_CONSENT
    assert authorization.worker_args["slab_weights"] is True
    assert authorization.worker_args["lora_low_rss"] is True
    grant = authorization.residency_grant
    assert grant["capability"] == mesh_residency.CAPACITY_RESCUE_CONSENT_CAPABILITY
    assert grant["consent"]["kind"] == "rescue-slab"
    assert grant["consent"]["id"] == "c" * 32
    # A consent is not a verdict: it carries no ledger authority of any kind.
    assert "gate_token" not in grant and "gate_protocol" not in grant


def test_without_a_consent_the_fallback_still_forces_stock(monkeypatch, tmp_path):
    """The consent earns the exemption, not the ledger state: with no memo the fallback still forces stock."""
    model = _Model(_Mesh({"slab_weights": True, "lora_low_rss": True}, handle=_Handle()))
    authorization = _authorize(model, monkeypatch, tmp_path)

    assert authorization.residency_mode == "stock"
    assert authorization.worker_args["slab_weights"] is False
    assert authorization.worker_args["lora_low_rss"] is False
    assert authorization.residency_grant is None


def test_an_exact_gate_pass_still_outranks_the_consent_path(
        checkpoint, monkeypatch, tmp_path):
    """A proven combination keeps its own authority; the consent path is only
    ever the fallback, so a PASS must not be downgraded to a consent."""
    model = _consented_model(checkpoint)
    authorization = _authorize(model, monkeypatch, tmp_path, state="pass")

    assert authorization.residency_mode == "required"
    assert authorization.residency_grant["capability"] == "normal_render_residency"


def test_a_consented_dispatch_reaches_the_workers_ladder(
        checkpoint, monkeypatch, tmp_path):
    """Every hop: memo, projection, authorization, wire, worker recheck and
    ladder. The 2026-08-04 acceptance found the last hop missing."""
    from dgx_monarch.actor import worker_authorization

    model = _consented_model(checkpoint)
    authorization = _authorize(model, monkeypatch, tmp_path)
    request = _dispatch(authorization)

    assert mesh_safety.assert_normal_render_residency_mode(
        request, authorization.worker_args) == "capacity_consent"
    assert mesh_safety.assert_normal_render_residency_grant(
        request, [_IDENTITY], effective_worker_args=authorization.worker_args,
        rank_world=1, setup_generation=7, setup_key=("key", "TORCH_FLASH", True),
        worker_args_key=("policy",), worker_topology=dict(_WORKER_TOPOLOGY)) is True

    consent = worker_authorization.sample_rescue_consent(request)
    assert consent == {"id": "c" * 32, "kind": "rescue-slab", "consent_source": "panel"}

    # Resolve under `auto`, the value the projection can hand back between two
    # checkpoints: there, worker args alone cannot carry the consent.
    decision = store_residency.resolve(
        path=str(checkpoint), unet_name=checkpoint.name, model_options={},
        slab_weights="auto", slab_capable_path=True, lora_low_rss=True,
        fsdp_launch=False, blocked_reason="", authoritative_slab_retry=False,
        memoized_family=lambda _p: None, vouched_families=frozenset(),
        rescue_consent=consent, fit_probe=lambda _p, _o: DOES_NOT_FIT)
    assert decision.use_slab is True
    assert decision.rung == store_residency.RUNG_CONSENTED_RESCUE

    with pytest.raises(consent_descriptor.SlabResidencyRescueOffer):
        store_residency.resolve(
            path=str(checkpoint), unet_name=checkpoint.name, model_options={},
            slab_weights="auto", slab_capable_path=True, lora_low_rss=True,
            fsdp_launch=False, blocked_reason="", authoritative_slab_retry=False,
            memoized_family=lambda _p: None, vouched_families=frozenset(),
            rescue_consent=None, fit_probe=lambda _p, _o: DOES_NOT_FIT)


def test_the_sample_endpoint_hands_its_consent_to_the_model_store():
    """The 2026-08-04 acceptance failure was one missing argument at one call
    site, the render path's own load: ModelStore.ensure takes `rescue_consent`
    and the ladder has a rung for it, but nothing passed one. Pin the call
    site, because a behavioral test of `sample` cannot reach it without a GPU."""
    tree = ast.parse((SRC / "actor" / "sample_protocol.py").read_text())
    body = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "run_sample")
    passes_consent = [
        call for call in ast.walk(body)
        if isinstance(call, ast.Call)
        and any(kw.arg == "rescue_consent" for kw in call.keywords)
        and any(kw.arg == "slot" and getattr(kw.value, "value", None) == "cond"
                for kw in call.keywords)
    ]
    assert passes_consent, ("the sample path's cond load must pass rescue_consent; "
                            "without it a granted card never reaches the ladder")
    named = {node.attr for call in passes_consent for node in ast.walk(call)
             if isinstance(node, ast.Attribute)}
    assert "sample_rescue_consent" in named


def test_a_worker_loop_on_another_contract_refuses_and_names_the_skew(
        checkpoint, monkeypatch, tmp_path):
    """The one skew that could otherwise end in a silent stock load and an
    out-of-memory: a driver that believes it authorized slab residency talking
    to a loop that never read the field."""
    model = _consented_model(checkpoint)
    authorization = _authorize(model, monkeypatch, tmp_path)
    request = _dispatch(authorization)
    request["_dgxm_normal_residency_grant"]["consent_contract"] = 0

    with pytest.raises(RuntimeError) as raised:
        mesh_safety.assert_normal_render_residency_grant(
            request, [_IDENTITY], effective_worker_args=authorization.worker_args,
            rank_world=1)
    message = str(raised.value)
    assert "disagree about what a consent authorizes" in message
    assert "dgxm restart" in message


def test_an_older_worker_loop_cannot_silently_stock_load_a_consented_dispatch(
        checkpoint, monkeypatch, tmp_path):
    """A loop that predates this mode has no arm for it and falls through to
    the `elif risky:` refusal in its own frozen copy of the mode check. That
    arm fires only if the dispatched policy is risky, so pin the property the
    skew's fail-closed behavior rests on: a consented dispatch always is."""
    model = _consented_model(checkpoint)
    authorization = _authorize(model, monkeypatch, tmp_path)

    assert mesh_safety.fleet_policy_is_risky(
        authorization.model_request, authorization.worker_args) is True


def test_a_worker_on_this_contract_behaves_as_today_under_an_older_driver():
    """Reverse skew: an older driver stamps no envelope and no consent mode, so
    no consent arm is reached and the stock dispatch is unchanged."""
    from dgx_monarch.actor import worker_authorization

    request = {"model": {"unet_name": "h3.safetensors", "options": {}, "loras": []},
               "_dgxm_normal_residency_mode": "stock"}
    assert mesh_safety.assert_normal_render_residency_mode(
        request, {"slab_weights": False, "lora_low_rss": False}) == "stock"
    assert worker_authorization.sample_rescue_consent(request) is None
    assert mesh_safety.assert_normal_render_residency_grant(request, [_IDENTITY],
                                                            effective_worker_args={}) is False


def test_an_explicit_slab_off_gets_no_consent_exemption(
        checkpoint, monkeypatch, tmp_path):
    """docs/DESIGN.md section 5.9 invariant 2, at the authorization site: the
    driver never mints residency the projection itself refused to write."""
    model = _consented_model(checkpoint, worker_args={"slab_weights": False})
    authorization = _authorize(model, monkeypatch, tmp_path)

    assert authorization.residency_mode == "stock"
    assert authorization.residency_grant is None
    assert authorization.worker_args["slab_weights"] is False


def test_a_quarantined_combination_gets_no_consent_envelope(
        checkpoint, monkeypatch, tmp_path, _rig):
    """A quarantine outranks every consent, at this site too."""
    from dgx_monarch.gate_ledger import GateLedger

    model = _consented_model(checkpoint)
    combo, artifacts = consent_rescue.combo_identity(
        checkpoint.name, {}, str(checkpoint))
    GateLedger(str(_rig)).record(
        combo, artifacts, "deadbeef", "FAIL",
        {"quarantine_levers": ["slab_weights", "lora_low_rss"]},
        {"worker_args": {"slab_weights": True}})
    authorization = _authorize(model, monkeypatch, tmp_path)

    assert authorization.residency_mode == "stock"
    assert authorization.residency_grant is None


def test_a_consent_envelope_that_claims_gate_authority_is_refused(
        checkpoint, monkeypatch, tmp_path):
    model = _consented_model(checkpoint)
    authorization = _authorize(model, monkeypatch, tmp_path)
    request = _dispatch(authorization)
    request["_dgxm_normal_residency_grant"]["gate_token"] = ["forged"]

    with pytest.raises(RuntimeError, match="must not carry identity-gate authority"):
        mesh_safety.assert_normal_render_residency_grant(
            request, [_IDENTITY], effective_worker_args=authorization.worker_args)


def test_a_consent_authorizes_one_model_slot_and_no_lora_stack(
        checkpoint, monkeypatch, tmp_path):
    """Both are second unproven paths the rescue consent says nothing about."""
    model = _consented_model(checkpoint)
    authorization = _authorize(model, monkeypatch, tmp_path)

    dual = _dispatch(authorization, uncond={"unet_name": "other.safetensors",
                                            "options": {}, "loras": []})
    with pytest.raises(RuntimeError, match="exactly one model slot"):
        mesh_safety.assert_normal_render_residency_grant(
            dual, [_IDENTITY], effective_worker_args=authorization.worker_args)

    with_lora = _dispatch(authorization)
    with_lora["model"] = {**authorization.model_request,
                          "loras": [{"name": "a.safetensors", "strength": 1.0}]}
    with pytest.raises(RuntimeError, match="LoRA stack"):
        mesh_safety.assert_normal_render_residency_grant(
            with_lora, [_IDENTITY], effective_worker_args=authorization.worker_args)


def test_a_consented_dispatch_is_bound_to_its_setup_and_policy(
        checkpoint, monkeypatch, tmp_path):
    """The envelope travels in the Gate grant's request field so it inherits
    that binding; the binding must be enforced, not only carried."""
    model = _consented_model(checkpoint)
    authorization = _authorize(model, monkeypatch, tmp_path)
    request = _dispatch(authorization)

    with pytest.raises(RuntimeError, match="READY setup generation"):
        mesh_safety.assert_normal_render_residency_grant(
            request, [_IDENTITY], effective_worker_args=authorization.worker_args,
            setup_generation=8)
    with pytest.raises(RuntimeError, match="executing worker policy"):
        mesh_safety.assert_normal_render_residency_grant(
            request, [_IDENTITY],
            effective_worker_args={**authorization.worker_args, "swap_verify": 4})
    drifted = _dispatch(authorization)
    drifted["_dgxm_normal_residency_grant"]["artifact_digest"] = "c" * 64
    with pytest.raises(RuntimeError, match="identity summary"):
        mesh_safety.assert_normal_render_residency_grant(
            drifted, [_IDENTITY], effective_worker_args=authorization.worker_args)


def test_each_unreadable_quarantine_state_names_its_own_cause(_rig, monkeypatch):
    """Six causes, six sentences, all failing closed.

    On 2026-09-02 the H3 uly2+fsdp card said its quarantine state "could not be
    read unambiguously" and sent the operator to the output directory. Six
    conditions produce that state and each asks for a different action: a
    request with no complete artifact identity, a ledger this host cannot read,
    damage above the newest matching row, damage this read cannot scope to a
    row at all, a newest row bound to another gate protocol, package version or
    source build, and a verdict this reader does not recognize. The state stays
    unknown for every one of them.
    """
    from dgx_monarch import __version__
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION, GateLedger

    def decide(rows, damage=0, **row):
        return consent_quarantine._scope_decision(
            rows, damage, "ctx", "source", GATE_PROTOCOL_VERSION, __version__)

    bound = {"capability_context": "ctx", "gate_protocol": GATE_PROTOCOL_VERSION,
             "dgx_monarch": __version__, "dgx_source": "source"}
    malformed = consent_quarantine.quarantine_decision("", None)
    damaged = decide([(20, {"verdict": "PASS", **bound})], damage=25)
    unscoped_damage = decide((), damage=25)
    unbound = decide([(1, {"verdict": "PASS", **bound, "gate_protocol": -1})])
    unrecognized = decide([(1, {"verdict": "WOBBLE", **bound})])

    def unreadable(_ledger):
        raise OSError("ledger unavailable")

    monkeypatch.setattr(GateLedger, "entries_with_integrity", unreadable)
    unreadable_ledger = consent_quarantine.quarantine_decision(
        "combo", types.SimpleNamespace(current="a", legacy="a", legacy_complete=True))

    found = [malformed, unreadable_ledger, damaged, unscoped_damage, unbound,
             unrecognized]
    assert [item.state for item in found] == ["unknown"] * 6
    assert len({item.reason for item in found}) == 6
    assert "complete artifact identity" in malformed.reason
    assert "could not be read on this host" in unreadable_ledger.reason
    assert "damaged above the newest row" in damaged.reason
    assert "cannot tie it to a capability-scoped row" in unscoped_damage.reason
    assert "not authoritative for this gate protocol" in unbound.reason
    assert "verdict this reader does not recognize" in unrecognized.reason


def test_the_contextless_damage_sentence_does_not_contradict_the_scoped_read(
        _rig, monkeypatch):
    """One torn line below an authoritative scoped PASS, read two ways.

    The loader site passes no capability context, so it takes the aggregate
    branch. With damage on line 1 and the only matching row, an authoritative
    scoped PASS, on line 2, the aggregate read must not print "damaged above the
    newest row matching these exact artifacts": the scoped read of the same
    ledger says clear. The state must not move: a torn line could still have
    carried a FAIL in a scope with no surviving rows, and clearing is per scope.
    """
    from dgx_monarch import __version__, runtime_provenance
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION, GateLedger

    row = {"key": "combo", "artifacts": "a", "verdict": "PASS",
           "capability_context": "ctx", "gate_protocol": GATE_PROTOCOL_VERSION,
           "dgx_monarch": __version__, "dgx_source": "source"}
    monkeypatch.setattr(
        GateLedger, "entries_with_integrity", lambda _self: ([(2, row)], 1))
    monkeypatch.setattr(
        runtime_provenance, "cached_dgx_source_manifest_sha256", lambda: "source")
    artifacts = types.SimpleNamespace(current="a", legacy="a", legacy_complete=True)

    scoped = consent_quarantine.quarantine_decision("combo", artifacts, "ctx")
    contextless = consent_quarantine.quarantine_decision("combo", artifacts)

    assert scoped.state == "clear"
    assert contextless.state == "unknown"
    assert "above the newest row" not in contextless.reason
    assert "cannot tie it to a capability-scoped row" in contextless.reason


def test_a_scope_that_read_damage_above_its_own_row_keeps_that_sentence(
        _rig, monkeypatch):
    """The accurate cause outranks the unscoped one.

    Damage above a matching row is the sentence an operator can act on, so the
    aggregate picks it first; the unscoped sentence speaks only where no scope
    read that damage.
    """
    from dgx_monarch import __version__, runtime_provenance
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION, GateLedger

    row = {"key": "combo", "artifacts": "a", "verdict": "PASS",
           "capability_context": "ctx", "gate_protocol": GATE_PROTOCOL_VERSION,
           "dgx_monarch": __version__, "dgx_source": "source"}
    monkeypatch.setattr(
        GateLedger, "entries_with_integrity", lambda _self: ([(2, row)], 9))
    monkeypatch.setattr(
        runtime_provenance, "cached_dgx_source_manifest_sha256", lambda: "source")

    decision = consent_quarantine.quarantine_decision(
        "combo", types.SimpleNamespace(current="a", legacy="a", legacy_complete=True))

    assert decision.state == "unknown"
    assert "damaged above the newest row" in decision.reason


def test_no_unreadable_quarantine_cause_ever_reads_clear():
    """Every unknown constant fails closed, by construction and by name."""
    unknowns = [value for name, value in vars(consent_quarantine).items()
                if name.startswith("_UNKNOWN_")
                and isinstance(value, consent_quarantine.QuarantineDecision)]
    assert len(unknowns) == 6
    assert {item.state for item in unknowns} == {"unknown"}
    assert all(item.reason for item in unknowns)


def test_the_quarantine_reason_does_not_promise_a_re_run_an_unscoped_fail_ignores():
    """Both FAIL spellings deny, and the copy says which one a re-run clears.

    A gate re-run writes capability-scoped rows. The clearing branch requires a
    scope, and an unscoped FAIL outranks even an exact contextual PASS, so an
    unconditional "re-run to clear it" would send that operator to a re-run
    that lifts nothing.
    """
    from dgx_monarch import __version__
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION

    # A scoped row speaks only while it binds to the running release, so the
    # scoped fixture carries that binding. The unscoped row has none to carry.
    scoped = consent_quarantine._scope_decision(
        [(1, {"verdict": "FAIL", "gate_protocol": GATE_PROTOCOL_VERSION,
              "dgx_monarch": __version__})],
        0, "capability-context", "source", GATE_PROTOCOL_VERSION, __version__)
    unscoped = consent_quarantine._scope_decision(
        [(1, {"verdict": "FAIL"})], 0, None, "source",
        GATE_PROTOCOL_VERSION, __version__)

    assert scoped.state == "fail" and unscoped.state == "fail"
    assert "FAILED" in unscoped.reason
    for reason in (scoped.reason, unscoped.reason):
        assert "capability-scoped FAIL clears when you re-run" in reason
        assert "an unscoped one predates those rows and no later run lifts it" in reason


def _bound_fail(**extra):
    """A scoped FAIL row from the running release."""
    from dgx_monarch import __version__
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION

    return {"verdict": "FAIL", "capability_context": "ctx",
            "gate_protocol": GATE_PROTOCOL_VERSION, "dgx_monarch": __version__,
            **extra}


def _decide(rows, scope="ctx"):
    from dgx_monarch import __version__
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION

    return consent_quarantine._scope_decision(
        rows, 0, scope, "source", GATE_PROTOCOL_VERSION, __version__)


def test_a_scoped_fail_from_the_running_release_still_blocks_consent():
    """Retirement must never reach a measured FAIL from the running release."""
    assert _decide([(1, _bound_fail())]).state == "fail"
    assert _decide([(1, _bound_fail(max_abs_latent_diff=2.53))]).state == "fail"


def test_a_scoped_fail_from_a_superseded_release_stops_blocking_consent():
    """The gate retires this row on a protocol or package move, and so does consent.

    Otherwise one ledger gives two answers: `gate_ledger.lookup_with_integrity`
    selects a scoped row only under the running protocol and version, while
    this reader would keep it live permanently.
    """
    from dgx_monarch import __version__
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION

    older_protocol = dict(_bound_fail(), gate_protocol=GATE_PROTOCOL_VERSION - 1)
    older_package = dict(_bound_fail(), dgx_monarch=f"0.0.0-before-{__version__}")

    assert _decide([(1, older_protocol)]).state == "clear"
    assert _decide([(1, older_package)]).state == "clear"

    # A retired row is also not the newest row for anything: it must not come
    # back as an unreadable state, which would deny by another name.
    assert _decide([(1, {"verdict": "PASS", "capability_context": "ctx",
                         "gate_protocol": GATE_PROTOCOL_VERSION,
                         "dgx_monarch": __version__, "dgx_source": "source"}),
                    (2, older_protocol)]).state == "clear"

    # The unscoped bucket has no binding to compare and keeps outranking.
    assert consent_quarantine._scope_decision(
        [(1, {"verdict": "FAIL"})], 0, None, "source",
        GATE_PROTOCOL_VERSION, __version__).state == "fail"


def test_a_stamped_sibling_fail_whose_cross_leg_diverged_still_blocks_consent():
    """The gate copies a FAIL onto the explicit-slab sibling for this reason.

    A diverged cross-residency leg is slab evidence, because that leg is the
    part of a ceremony that runs slab residency. Nothing retires it early.
    """
    row = _bound_fail(stamped="slab-mode-equivalence", cross_mode="FAIL")

    assert _decide([(1, row)]).state == "fail"


def test_a_stamped_sibling_fail_with_no_diverged_cross_leg_stops_blocking():
    """The read applies the writer's rule to rows that predate it.

    Until 2026-08-12 a ceremony stamped its FAIL onto the explicit-slab
    siblings on every non-PASS, so a swap-lineage failure recorded a verdict
    against a capability nothing had run. Read as live, such a row blocks the
    rescue consent whose load would let a ceremony overwrite it, a deadlock
    nothing but an upgrade clears.
    """
    absent = _bound_fail(stamped="slab-mode-equivalence")
    passed_cross = _bound_fail(stamped="slab-mode-equivalence", cross_mode="PASS")

    assert _decide([(1, absent)]).state == "clear"
    assert _decide([(1, passed_cross)]).state == "clear"
    # The stamp is not a wildcard: an unstamped FAIL with no cross leg is an
    # ordinary in-mode failure and still blocks.
    assert _decide([(1, _bound_fail())]).state == "fail"


def test_retiring_a_row_never_reads_a_damaged_ledger_as_clear():
    """Unreadable evidence stays unreadable. Retirement does not grant.

    A torn ledger line could have carried anything, a FAIL for this exact
    scope included, so a scope with no readable row left reads unknown while
    the damage stands. The retired row's position does not settle the torn
    line either: a row too weak to deny is too weak to vouch for what the torn
    line said. This denies where the clean ledger clears.
    """
    from dgx_monarch import __version__
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION

    retired = _bound_fail(stamped="slab-mode-equivalence")

    def decide(damage):
        return consent_quarantine._scope_decision(
            [(20, retired)], damage, "ctx", "source",
            GATE_PROTOCOL_VERSION, __version__).state

    assert decide(0) == "clear"
    assert decide(15) == "unknown"
    assert decide(25) == "unknown"


def test_the_uncertified_slab_load_refusal_is_a_protocol_skew_not_known_wrong(
        checkpoint):
    """Class P, not class K.

    A certificate this driver cannot read is a protocol skew between driver and
    worker, not measured wrongness, and class K implies that a waiver could
    exist for what it refuses. A byte-verify mismatch is class P for the same
    reason (docs/DESIGN.md section 5.9), and the two are one fault seen from two
    sides. Class P owes the operator the working alternative, so the text names
    it and carries no class K no-waiver sentence.
    """
    from dgx_monarch.actor.slab_certificate import SlabCertificateError

    load = consent_rescue.RescueLoad(
        combo_key="a" * 24, artifacts="b" * 64, unet_name=checkpoint.name,
        path=str(checkpoint), consent_id="c" * 32, measured={})

    class _Handle:
        def call_all(self, endpoint, **_kwargs):
            return []

    with pytest.raises(SlabCertificateError) as caught:
        consent_rescue.record_rescue_row(
            load, [{"rank": 0, "cond": {"residency": "slab"}}], 1, handle=_Handle())
    text = str(caught.value)
    tag = refusal.parse_refusal_tag(text)
    assert tag.refusal_class is refusal.RefusalClass.PHYSICS
    assert tag.guard is None and tag.waivable is False
    assert "What works instead: restart the Worker service" in text
    assert "there is no waiver for this refusal" not in text
    assert "protocol skew rather than measured wrongness" in text
