"""Class-K accuracy waivers end to end: ring-pad in full, then sol-attn and
shard-quant.

The divisibility-pad guard on ring and hybrid topologies, in the adapter base
backstop and its H3-scoped twin, can be waived. The properties pinned here:

* with a grant the guard proceeds, the render result carries the
  rendered-under-waiver stamp, and the permanent v9 WAIVER row lands;
* with no grant the guard raises its typed refusal, with the panel sentence,
  the headless fallback and the machine descriptor that turns it into a
  one-click card;
* a card is never raised for a question a memo already answers;
* a stamp never rides a chained latent; a later leg carries it only as
  inherited provenance;
* a revoke brings the refusal back on the next render;
* no class-K kind is auto-eligible, and accuracy_waiver refuses to import
  otherwise;
* an identity-gate ceremony never renders under a waiver;
* every skew direction degrades to the refusal, never to a silent unstamped
  waive.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from dgx_monarch import accuracy_waiver, consent_pending, consent_store
from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.gate_ledger import GateLedger
from dgx_monarch.nodes import consent_routes, consent_waiver
from dgx_monarch.refusal import GUARDS, RefusalClass, is_waivable, parse_refusal_tag

RING_PAD = "ring_pad"
H3_RING_PAD = "ring_pad:minimax_h3"
SOL_ATTN = "sol_attn"
SHARD_QUANT = "shard_quant_scale"
CHROMA_SHARD_QUANT = "shard_quant_scale:chroma"
H3_SOL_ATTN = "sol_attn:minimax_h3"
RING_KIND = "waive-known-wrong:ring-pad"
PIXELDIT_KIND = "waive-known-wrong:pixeldit-sp"
UNET = "model.safetensors"
RING2 = {"ulysses": 1, "ring": 2, "cfg": 1, "dp": 1, "fsdp": False}
ULY2 = {"ulysses": 2, "ring": 1, "cfg": 1, "dp": 1, "fsdp": False}


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """A driver and a worker on one box: a checkpoint, a consent store and a ledger."""
    checkpoint = tmp_path / UNET
    checkpoint.write_bytes(b"weights" * 64)
    output = tmp_path / "output"
    output.mkdir()
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = (  # type: ignore[attr-defined]
        lambda _kind, name: str(checkpoint) if name == UNET else None)
    folder_paths.get_output_directory = lambda: str(output)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(consent_store, "MEMO_PATH", str(tmp_path / "memo.json"))
    monkeypatch.setattr(consent_routes, "_RATE",
                        {"tokens": float(consent_routes.RATE_LIMIT_N), "t": 0.0})
    for spec in consent_pending.KIND_SPECS.values():
        monkeypatch.delenv(spec.env_var, raising=False)
    monkeypatch.delenv(consent_pending.AUTO_RESCUE_ENV, raising=False)
    consent_pending.clear_all()
    accuracy_waiver.clear()
    consent_waiver.reset_pending_audit()
    monkeypatch.setattr(consent_waiver, "_gate_ceremony_active", lambda: False)
    yield types.SimpleNamespace(checkpoint=str(checkpoint), output=str(output),
                                ledger=GateLedger(str(output)))
    consent_pending.clear_all()
    accuracy_waiver.clear()
    consent_waiver.reset_pending_audit()


@pytest.fixture(autouse=True)
def _forget_unreadable_env():
    """The env warning dedups process-wide, so every test starts it empty."""
    accuracy_waiver._ENV_UNREADABLE.clear()
    yield
    accuracy_waiver._ENV_UNREADABLE.clear()


def _model():
    # The world that matters is the one on the wire, threaded to `stamp_request`
    # separately; nothing under test reads a world off the model.
    mesh = types.SimpleNamespace(worker_args={}, attention="TORCH_FLASH")
    return types.SimpleNamespace(unet_name=UNET, options={}, loras=(), mesh=mesh)


def _topo(topology=RING2):
    return types.SimpleNamespace(**topology)


def _grant_from_the_panel(rig, kind, topology=RING2, world=2, guard=None):
    """Grant a waiver the way the panel does: card, then one click."""
    from dgx_monarch.nodes.consent_rescue import artifacts_digest, combo_identity

    combo, artifacts = combo_identity(UNET, {}, rig.checkpoint)
    context = accuracy_waiver.waiver_context(
        combo_key=combo, topology=topology, world=world)
    card = consent_pending.register_pending(consent_pending.ConsentDescriptor(
        kind=kind, path=rig.checkpoint, memo_context=context, combo_key=combo,
        artifacts=artifacts_digest(artifacts), unet_name=UNET,
        target_guard=guard or "", loras=0))
    assert card is not None, "a wired class-K kind must raise a card"
    status, payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"]})
    assert status == 200 and payload["ok"], payload
    return card, combo


def _worker_request(rig, model, topology=RING2, world=2, render_id="r-1"):
    """A driver request, stamped the way the dispatch stamps it."""
    request = {"model": {"unet_name": UNET, "options": {}, "loras": []},
               "_dgxm_render_id": render_id}
    consent_waiver.stamp_request(request, model, _topo(topology), world, render_id)
    return request


def _bind_worker(topology=RING2, world=2, dispatch=True):
    """Bind as `store_fsdp.ensure` does; dispatch=True is the sample path's own cond load."""
    accuracy_waiver.bind_model(UNET, {}, [], topology, world, dispatch=dispatch)


def test_the_stamp_is_spelled_once_for_the_row_the_result_and_the_log():
    ring = accuracy_waiver.stamp_text(RING_KIND)
    pixeldit = accuracy_waiver.stamp_text(PIXELDIT_KIND)
    assert ring == ("rendered-under-waiver: waive-known-wrong:ring-pad "
                    "(left-edge artifacts, measured 2026-07-10)")
    assert pixeldit == (
        "rendered-under-waiver: waive-known-wrong:pixeldit-sp (1-step NRMS 0.265 on "
        "uly2 and 0.338 on ring2 against a certified dp2/single reference, floor "
        "0.10, measured 2026-07-13)")
    # The grant-row endpoint and the use-row render path read the same
    # generator, so a ledger row and a render's metadata carry the same sentence.
    assert consent_routes._stamp_for(consent_pending.KIND_SPECS[RING_KIND]) == ring
    assert consent_routes._stamp_for(consent_pending.KIND_SPECS["rescue-slab"]) is None


def test_the_measured_wrongness_is_retained_for_live_and_historical_kinds():
    ring = consent_pending.KIND_SPECS[RING_KIND]
    assert ring.wired and ring.style == "accuracy"
    assert ring.measured and "measured 2026-07-10" in ring.measured

    pixeldit = consent_pending.KIND_SPECS[PIXELDIT_KIND]
    assert pixeldit.wired is False and pixeldit.style == "accuracy"
    assert pixeldit.measured and "NRMS 0.265" in pixeldit.measured


def test_the_pad_guard_refuses_unchanged_with_no_grant():
    accuracy_waiver.clear()
    with pytest.raises(UnsupportedModelError) as raised:
        base.assert_ulysses_only_padding(2, 1)
    message = str(raised.value)
    tag = parse_refusal_tag(message)
    assert tag is not None
    assert tag.refusal_class is RefusalClass.KNOWN_WRONG
    assert tag.guard == RING_PAD and tag.waivable is True
    assert is_waivable(message)
    assert "measured 2026-07-10" in message
    assert 'click "Render under waiver (output stamped)"' in message
    assert "DGXM_WAIVE_KNOWN_WRONG=ring_pad" in message
    assert message.index("DGX Monarch panel") < message.index("Headless:")
    # Unpadded ring and pure ulysses are untouched by the waiver wiring.
    base.assert_ulysses_only_padding(2, 0)
    base.assert_ulysses_only_padding(1, 3)


def test_a_refusal_carries_the_descriptor_that_becomes_the_card(rig):
    from dgx_monarch import consent_descriptor
    from dgx_monarch.nodes import consent_observe

    accuracy_waiver.clear()
    _bind_worker()
    with pytest.raises(UnsupportedModelError) as raised:
        base.assert_ulysses_only_padding(2, 1)
    descriptor = consent_descriptor.parse(str(raised.value))
    assert descriptor is not None
    assert descriptor.kind == RING_KIND
    assert descriptor.unet_name == UNET
    assert descriptor.measured["probe"] == RING_PAD
    assert descriptor.memo_context["topology"] == "ring2"
    assert descriptor.memo_context["world"] == "2"
    assert descriptor.env_fallback == "DGXM_WAIVE_KNOWN_WRONG"

    # The same driver frame the rescue uses turns it into a card: no new channel.
    card = consent_observe.observe_refusal(
        RuntimeError(f"ActorError: UnsupportedModelError: {raised.value}"))
    assert card is not None
    assert card["kind"] == RING_KIND and card["class"] == "K"
    assert card["style"] == "accuracy"
    assert card["auto_eligible"] is False
    assert card["primary"]["label"] == "Render under waiver (output stamped)"
    assert "measured 2026-07-10" in card["measured"]


def test_an_unbound_worker_refuses_with_no_card_rather_than_a_guessed_one():
    from dgx_monarch import consent_descriptor

    accuracy_waiver.clear()  # nothing loaded on this thread yet
    with pytest.raises(UnsupportedModelError) as raised:
        base.assert_ulysses_only_padding(2, 1)
    assert consent_descriptor.DESCRIPTOR_BEGIN not in str(raised.value)


def test_a_granted_ring_pad_waiver_passes_the_guard_and_stamps_the_render(rig):
    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    model = _model()
    request = _worker_request(rig, model)
    assert list(request[accuracy_waiver.REQUEST_KEY]) == [RING_KIND]

    # Worker side: activate, load, then the guard proceeds.
    assert accuracy_waiver.activate(request) == 1
    _bind_worker()
    base.assert_ulysses_only_padding(2, 1)          # no raise: waived
    base.assert_ulysses_only_padding(2, 5)          # and idempotent per render
    spent = accuracy_waiver.stamps()
    assert len(spent) == 1
    assert spent[0]["guard"] == RING_PAD and spent[0]["kind"] == RING_KIND
    assert spent[0]["stamp"] == accuracy_waiver.stamp_text(RING_KIND)
    assert spent[0]["run_id"] == "r-1"

    # Driver side: the result carries the stamp and the row lands.
    out = consent_waiver.stamp_result({"samples": None}, [{"waived": spent}])
    stamped = out[accuracy_waiver.STAMPED_RESULT_KEY]
    assert [entry["stamp"] for entry in stamped] == [accuracy_waiver.stamp_text(RING_KIND)]
    assert stamped[0]["guard"] == RING_PAD
    assert stamped[0]["consent_id"] == spent[0]["consent_id"]

    rows = [row for row in rig.ledger.entries() if row["verdict"] == "WAIVER"]
    grant, use = rows
    assert grant["action"] == "grant" and grant["waiver_class"] == "K"
    assert use["action"] == "use"
    assert use["waiver_kind"] == RING_KIND
    assert use["target_guard"] == RING_PAD
    assert use["run_id"] == "r-1"
    assert use["stamp"] == accuracy_waiver.stamp_text(RING_KIND)
    assert use["consent_id"] == grant["consent_id"]
    assert use["memo_context"] == grant["memo_context"]
    # A waiver row is never authority: it lives in the audit key namespace.
    assert use["key"].startswith("audit:waiver:")


def test_one_kind_covers_the_backstop_and_its_family_scoped_twin(rig):
    """One grant covers both guards of the kind.

    Each permanent row still names the guard that fired, so the audit shows
    which guard was waived though the operator answered one card.
    """
    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    request = _worker_request(rig, _model(), render_id="r-3")
    accuracy_waiver.activate(request)
    _bind_worker()
    assert accuracy_waiver.waived(RING_PAD) is True
    assert accuracy_waiver.waived(H3_RING_PAD) is True
    assert sorted(entry["guard"] for entry in accuracy_waiver.stamps()) == [
        RING_PAD, H3_RING_PAD]
    consent_waiver.stamp_result({}, [{"waived": accuracy_waiver.stamps()}])
    guards = sorted(row["target_guard"] for row in rig.ledger.entries()
                    if row.get("action") == "use")
    assert guards == [RING_PAD, H3_RING_PAD]


def test_a_revoked_waiver_brings_the_refusal_straight_back(rig):
    card, _combo = _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    assert _worker_request(rig, _model()).get(accuracy_waiver.REQUEST_KEY)

    status, payload = consent_routes.handle_action(
        {"action": "revoke", "key": card["key"]})
    assert status == 200 and payload["ok"]

    request = _worker_request(rig, _model(), render_id="r-4")
    assert accuracy_waiver.REQUEST_KEY not in request
    assert accuracy_waiver.activate(request) == 0
    _bind_worker()
    with pytest.raises(UnsupportedModelError):
        base.assert_ulysses_only_padding(2, 1)
    assert accuracy_waiver.stamps() == []
    revoke = [row for row in rig.ledger.entries() if row.get("action") == "revoke"]
    assert revoke and revoke[0]["revoked_by"] == "user"


def test_a_waiver_is_scoped_to_the_topology_and_world_it_was_granted_for(rig):
    _grant_from_the_panel(rig, RING_KIND, topology=RING2, world=2, guard=H3_RING_PAD)
    # Same checkpoint, a different resolved topology: a different question.
    assert accuracy_waiver.REQUEST_KEY not in _worker_request(
        rig, _model(), topology=ULY2, render_id="r-5")
    assert accuracy_waiver.REQUEST_KEY not in _worker_request(
        rig, _model(), topology=RING2, world=4, render_id="r-6")
    assert accuracy_waiver.REQUEST_KEY in _worker_request(
        rig, _model(), render_id="r-7")


def test_no_active_class_k_kind_is_ever_auto_eligible(rig):
    """accuracy_waiver checks this at import and refuses to load otherwise."""
    for kind in accuracy_waiver.KNOWN_WRONG_KINDS:
        assert consent_pending.KIND_SPECS[kind].auto_eligible is False
    assert accuracy_waiver._assert_never_auto() is None


def test_the_standing_toggle_and_its_env_mirror_grant_no_accuracy_waiver(
        rig, monkeypatch):
    consent_store.set_auto_rescue(True)
    monkeypatch.setenv(consent_pending.AUTO_RESCUE_ENV, "1")
    request = _worker_request(rig, _model(), render_id="r-8")
    assert accuracy_waiver.REQUEST_KEY not in request
    assert consent_routes.consent_state()["auto_rescue"] is True


def test_the_headless_fallback_must_name_the_guard(rig, monkeypatch):
    monkeypatch.setenv("DGXM_WAIVE_KNOWN_WRONG", "1")
    assert accuracy_waiver.REQUEST_KEY not in _worker_request(
        rig, _model(), render_id="r-9")

    monkeypatch.setenv("DGXM_WAIVE_KNOWN_WRONG", "ring_pad")
    request = _worker_request(rig, _model(), render_id="r-10")
    grants = request[accuracy_waiver.REQUEST_KEY]
    assert list(grants) == [RING_KIND]
    assert grants[RING_KIND]["consent_source"] == "env"
    # An environment grant persists no memo, and still writes its use row.
    assert consent_store.read()["consents"] == {}
    accuracy_waiver.activate(request)
    _bind_worker()
    base.assert_ulysses_only_padding(2, 1)
    consent_waiver.stamp_result({}, [{"waived": accuracy_waiver.stamps()}])
    use = next(row for row in rig.ledger.entries() if row.get("action") == "use")
    assert use["consent_source"] == "env" and use["run_id"] == "r-10"


def test_an_identity_gate_ceremony_never_renders_under_a_waiver(rig, monkeypatch):
    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    monkeypatch.setattr(consent_waiver, "_gate_ceremony_active", lambda: True)
    request = _worker_request(rig, _model(), render_id="r-11")
    assert accuracy_waiver.REQUEST_KEY not in request
    assert accuracy_waiver.activate(request) == 0


def test_an_unreadable_ceremony_flag_is_treated_as_a_ceremony(rig, monkeypatch):
    """Fail closed: a waiver never rides a dispatch this site cannot classify."""
    import dgx_monarch.nodes.gate_process_state as gate_state

    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    monkeypatch.undo()  # undo the fixture's patches: this test needs the real reader
    # gate_process_state is the flag's home and nodes/common forwards the read,
    # so a name missing here is one the classification site cannot resolve.
    monkeypatch.delattr(gate_state, "_AUTO_GATE_ACTIVE")
    assert consent_waiver._gate_ceremony_active() is True
    assert accuracy_waiver.REQUEST_KEY not in _worker_request(
        rig, _model(), render_id="r-12")


def test_an_older_worker_degrades_to_the_refusal_not_a_silent_waive(rig):
    """New driver, old worker: the stamp is an unread request key.

    The old worker's guard has no waiver check, so it refuses, and its
    result carries no `waived` key. The driver must then stamp nothing and
    write no use row: absence is never evidence that a bypass happened.
    """
    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    request = _worker_request(rig, _model(), render_id="r-13")
    assert accuracy_waiver.REQUEST_KEY in request

    old_worker_result = [{"host": "worker", "rank": 0, "latent": None}]
    out = consent_waiver.stamp_result({"samples": None}, old_worker_result)
    assert accuracy_waiver.STAMPED_RESULT_KEY not in out
    assert [row for row in rig.ledger.entries() if row.get("action") == "use"] == []


def test_a_newer_worker_stamp_never_authorizes_an_older_driver(rig):
    """Reverse skew and every malformed shape: no grant, so no waive."""
    _bind_worker()
    for payload in ({"kind-from-the-future": {"id": "a" * 32, "kind": "x"}},
                    {RING_KIND: {"kind": RING_KIND}},
                    {RING_KIND: "not-a-mapping"},
                    {"rescue-slab": {"id": "a" * 32, "kind": "rescue-slab",
                                     "consent_source": "panel"}},
                    "not-a-mapping"):
        assert accuracy_waiver.activate(
            {"_dgxm_render_id": "r", accuracy_waiver.REQUEST_KEY: payload}) == 0
        assert accuracy_waiver.granted(RING_PAD) is None
        with pytest.raises(UnsupportedModelError):
            base.assert_ulysses_only_padding(2, 1)


def test_a_grant_never_outlives_its_dispatch_onto_a_reused_worker_thread(rig):
    """A load outside the authorized dispatch disarms whatever is installed.

    Worker threads are reused, and only the sample path's own cond load passes
    a rescue consent, so every other load (an eager loader push, a gate
    ceremony cycle, a scheduler query) is treated as outside the dispatch.
    """
    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    accuracy_waiver.activate(_worker_request(rig, _model(), render_id="r-15"))
    _bind_worker()
    assert accuracy_waiver.granted(RING_PAD) is not None

    _bind_worker(dispatch=False)   # an eager load lands on the same thread
    assert accuracy_waiver.granted(RING_PAD) is None
    assert accuracy_waiver.stamps() == []
    with pytest.raises(UnsupportedModelError):
        base.assert_ulysses_only_padding(2, 1)


def test_only_the_sample_paths_cond_load_counts_as_its_dispatch():
    """Every worker load goes through `store_fsdp.ensure`; pin how it classifies each caller."""
    import ast
    import inspect

    from dgx_monarch.actor import store_fsdp

    tree = ast.parse(inspect.getsource(store_fsdp.ensure))
    call = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "bind_model")
    dispatch = next(kw.value for kw in call.keywords if kw.arg == "dispatch")
    assert isinstance(dispatch, ast.Compare)
    assert dispatch.left.value == "rescue_consent"  # type: ignore[attr-defined]
    # The wrapper also decodes `ModelStore.ensure`'s positional signature by
    # index. Pin the indices to the real parameter order: inserting a parameter
    # before `lora_stack` would otherwise compute a combo key off the wrong
    # LoRA list, and the card would mint an id under a memo context the driver
    # never resolves, so every click would silently fail to take effect.
    from dgx_monarch.actor.model_store import ModelStore

    names = list(inspect.signature(ModelStore.ensure).parameters)[1:]
    assert names[:3] == ["unet_name", "options", "lora_stack"], names
    indexed = [node for node in ast.walk(call)
               if isinstance(node, ast.Subscript) and getattr(node.value, "id", "") == "args"]
    assert sorted(node.slice.value for node in indexed) == [0, 1, 2]  # type: ignore[attr-defined]


def test_a_dispatch_the_driver_cannot_join_writes_no_row_but_still_stamps(rig):
    """A stamp with no dispatch behind it never guesses a combination."""
    spent = [{"guard": RING_PAD, "kind": RING_KIND, "consent_id": "d" * 32,
              "consent_source": "panel", "run_id": "unknown-run",
              "stamp": accuracy_waiver.stamp_text(RING_KIND)}]
    out = consent_waiver.stamp_result({}, [{"waived": spent}])
    assert out[accuracy_waiver.STAMPED_RESULT_KEY][0]["guard"] == RING_PAD
    assert [row for row in rig.ledger.entries() if row.get("action") == "use"] == []


def test_every_rank_reporting_one_guard_is_one_waiver(rig):
    spent = {"guard": RING_PAD, "kind": RING_KIND, "consent_id": "d" * 32,
             "consent_source": "panel", "run_id": "r", "stamp": "s"}
    folded = consent_waiver.fold_result_stamps(
        [{"waived": [spent]}, {"waived": [spent]}, {}, {"waived": None}, "junk"])
    assert len(folded) == 1
    assert consent_waiver.fold_result_stamps(None) == []


def test_a_granted_pad_waiver_survives_the_wan_ring_surface_check(rig):
    """The pad guard alone decides a padded surface.

    The Wan ring TORCH_FLASH binding checks the Comfy attention surface right
    after it, and a second refusal there would answer the waiver with an
    untagged, cardless error after the click was spent.
    """
    from dgx_monarch.adapters.wan_ring_attention import _CallPreflight

    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    accuracy_waiver.activate(_worker_request(rig, _model(), render_id="r-20"))
    _bind_worker()
    preflight = _CallPreflight(2)
    with pytest.raises(UnsupportedModelError) as raised:
        preflight.surface(None, None, None, 8, None, None, False, False, False,
                          [0], None, None, {})
    # Past the pad guard and into the ordinary q/k/v checks, not a second
    # refusal about the padding itself.
    assert "unsupported Comfy attention surface" not in str(raised.value)
    assert [entry["guard"] for entry in accuracy_waiver.stamps()] == [RING_PAD]


def test_a_card_is_never_raised_for_a_question_already_answered(rig):
    """A ceremony leg renders with no waiver on purpose.

    The gate must never grade waived math, so the stamping site stands down
    while it runs and the guard refuses. That refusal must not put the card
    back in front of an operator who has already answered it.
    """
    from dgx_monarch.nodes import consent_observe

    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    accuracy_waiver.clear()
    _bind_worker()
    with pytest.raises(UnsupportedModelError) as raised:
        base.assert_ulysses_only_padding(2, 1)   # the ceremony leg's refusal
    again = consent_observe.observe_refusal(
        RuntimeError(f"ActorError: UnsupportedModelError: {raised.value}"))
    assert again is None
    assert consent_pending.pending_cards() == []
    # A capacity refusal under a live memo is a different fault: it keeps its
    # card, because there the click has not reached the load.
    assert consent_observe._already_granted("rescue-slab", rig.checkpoint, {}) is False


def test_a_waiver_stamp_never_rides_a_chained_latent(rig):
    """The stamp describes the render that produced a latent, not the next one.

    A sampler builds its output from its input latent, so an inherited stamp
    would mark a clean render as waived and would disagree with the ledger,
    which is the authoritative surface.
    """
    from dgx_monarch.nodes.render_preflight import latent_without_topology_metadata

    waived_output = {"samples": None,
                     accuracy_waiver.STAMPED_RESULT_KEY: [{"guard": RING_PAD}]}
    assert accuracy_waiver.STAMPED_RESULT_KEY not in latent_without_topology_metadata(
        waived_output)
    assert accuracy_waiver.STAMPED_RESULT_KEY in waived_output  # the input is untouched


def test_the_waiver_topology_label_is_the_operators_own_preset_spelling():
    """The scope string a waiver is keyed on is the one the widget shows.

    `accuracy_waiver` restates the spelling instead of importing `Topology`,
    which keeps it a leaf; pin the two together so the memo key can never start
    naming a topology the operator does not recognize.
    """
    from dgx_monarch.topology import PRESETS, Topology

    for preset, kwargs in PRESETS.items():
        topo = Topology(world=2, **kwargs).with_derived_dp()  # type: ignore[arg-type]
        wire = consent_waiver.topology_wire(topo)
        assert accuracy_waiver.topology_label(wire) == topo.describe(), preset
    assert accuracy_waiver.topology_label(None) == "single"
    assert accuracy_waiver.topology_label({"ring": "junk"}) == "single"


def test_the_live_class_k_guards_are_wired_and_agree_with_the_registry():
    assert set(accuracy_waiver.KNOWN_WRONG_GUARDS) == {
        RING_PAD, H3_RING_PAD, SOL_ATTN, H3_SOL_ATTN, SHARD_QUANT, CHROMA_SHARD_QUANT}
    for guard, kind in accuracy_waiver.KNOWN_WRONG_GUARDS.items():
        spec = GUARDS[guard]
        assert spec.refusal_class is RefusalClass.KNOWN_WRONG
        assert spec.waivable_now is True
        assert consent_pending.KIND_SPECS[kind].wired is True
        action = accuracy_waiver.panel_action(guard)
        assert action.env == "DGXM_WAIVE_KNOWN_WRONG"
        assert action.env_value in guard
    assert consent_pending.KIND_SPECS[PIXELDIT_KIND].wired is False
    assert PIXELDIT_KIND not in accuracy_waiver.KNOWN_WRONG_KINDS
    assert accuracy_waiver.granted("stock_load_preflight") is None


def test_the_use_row_shape_is_exactly_what_protocol_v9_froze(rig):
    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    request = _worker_request(rig, _model(), render_id="r-14")
    accuracy_waiver.activate(request)
    _bind_worker()
    base.assert_ulysses_only_padding(2, 1)
    consent_waiver.stamp_result({}, [{"waived": accuracy_waiver.stamps()}])
    raw = [json.loads(line) for line in
           open(rig.ledger.path, encoding="utf-8").read().splitlines()]
    use = next(row for row in raw if row.get("action") == "use")
    for field in ("waiver_kind", "waiver_class", "target_guard", "action",
                  "consent_id", "consent_source", "memo_fingerprint",
                  "memo_context", "file_identity", "unet_name", "loras",
                  "reason", "waived_at", "run_id", "stamp"):
        assert field in use, field
    assert use["waiver_class"] == "K"
    assert use["verdict"] == "WAIVER"
    assert len(use["consent_id"]) == 32


def test_a_reserved_class_k_guard_is_never_granted_by_its_kinds_memo():
    """`waivable_now=False` reserves a guard id; it must not proceed unasked.

    `refusal._validate` refuses to let a site declare such a guard waivable.
    The proceed side has to make the same cut, or an existing grant for the
    kind would silence a guard that offers no card.
    """
    from dgx_monarch.refusal import GuardSpec

    reserved = GuardSpec(RefusalClass.KNOWN_WRONG, False, RING_KIND,
                         "waiver designed, not wired in this release")
    derived = accuracy_waiver.known_wrong_guards({**GUARDS, "ring_pad:reserved": reserved})
    assert "ring_pad:reserved" not in derived
    assert set(derived) == set(accuracy_waiver.KNOWN_WRONG_GUARDS)


def test_a_reserved_guard_refuses_even_while_its_kind_is_granted(rig, monkeypatch):
    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    accuracy_waiver.activate(_worker_request(rig, _model(), render_id="r-21"))
    assert accuracy_waiver.granted(H3_RING_PAD) is not None   # the wired twin proceeds
    monkeypatch.delitem(accuracy_waiver.KNOWN_WRONG_GUARDS, H3_RING_PAD)
    assert accuracy_waiver.granted(H3_RING_PAD) is None
    assert accuracy_waiver.waived(H3_RING_PAD) is False


def test_a_grant_whose_channel_this_build_cannot_name_installs_nothing(rig):
    """An unreadable `consent_source` refuses instead of rendering unaudited.

    The permanent row validates the channel against a frozen vocabulary, so a
    channel checked only at row-build time would render, stamp the output and
    then fail to write the permanent audit row. The check belongs at the moment
    the authorization is installed.
    """
    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    request = _worker_request(rig, _model(), render_id="r-22")
    request[accuracy_waiver.REQUEST_KEY][RING_KIND]["consent_source"] = "sticky-note"
    assert accuracy_waiver.activate(request) == 0
    _bind_worker()
    with pytest.raises(UnsupportedModelError):
        base.assert_ulysses_only_padding(2, 1)


def test_the_dispatch_audit_window_is_retired_once_its_row_is_written(rig):
    """The bounded join window holds an entry per outstanding dispatch.

    Writing the use row retires the entry. The window is capped and never
    evicts a live entry, so an entry that is never retired holds its slot.
    """
    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    request = _worker_request(rig, _model(), render_id="r-23")
    accuracy_waiver.activate(request)
    _bind_worker()
    base.assert_ulysses_only_padding(2, 1)
    assert "r-23" in consent_waiver._PENDING_AUDIT
    consent_waiver.stamp_result({}, [{"waived": accuracy_waiver.stamps()}])
    assert consent_waiver._PENDING_AUDIT == {}


def test_retire_audit_is_idempotent_and_total(rig):
    """The close-path retire hook must never add a failure of its own."""
    consent_waiver._PENDING_AUDIT["r-30"] = {"combo_key": "c"}
    consent_waiver.retire_audit("r-30")
    consent_waiver.retire_audit("r-30")
    consent_waiver.retire_audit(None)
    consent_waiver.retire_audit(object())
    assert consent_waiver._PENDING_AUDIT == {}


def test_retire_audit_is_total_for_hostile_baseexception_stringification(rig):
    class StopNow(BaseException):
        pass

    cancellation = StopNow("render-id stringification interrupted")

    class HostileRenderId:
        def __str__(self):
            raise cancellation

    consent_waiver._PENDING_AUDIT["still-live"] = {"combo_key": "c"}

    consent_waiver.retire_audit(HostileRenderId())
    consent_waiver.retire_audit(HostileRenderId())

    assert consent_waiver._PENDING_AUDIT == {
        "still-live": {"combo_key": "c"}
    }


def test_a_render_that_never_stamps_retires_its_audit_facts_on_close(rig):
    """A refused, failed or granted-but-unfired render cannot leak its entry.

    ``stamp_result`` pops only the stamped-success path. Every other outcome
    goes through ``PendingRender._close``, and a submit that never built one
    goes through ``SubmitRenderGuard.cleanup``; both call ``retire_audit``.
    Nothing evicts an entry left behind, so it is held for the life of the
    process and 512 of them refuse the next waived dispatch.
    """
    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    _worker_request(rig, _model(), render_id="r-31")
    _worker_request(rig, _model(), render_id="r-32")
    assert "r-31" in consent_waiver._PENDING_AUDIT
    consent_waiver.retire_audit("r-31")
    assert "r-31" not in consent_waiver._PENDING_AUDIT
    assert "r-32" in consent_waiver._PENDING_AUDIT


def test_a_513th_outstanding_waived_dispatch_is_not_stamped(rig, caplog):
    """Live audit facts cannot be evicted before their mandatory use row lands."""
    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    for i in range(consent_waiver._PENDING_AUDIT_LIMIT):
        consent_waiver._PENDING_AUDIT[f"r-old-{i}"] = {"combo_key": "c"}
    with caplog.at_level("ERROR"):
        request = _worker_request(rig, _model(), render_id="r-new")
    assert accuracy_waiver.REQUEST_KEY not in request
    assert "r-old-0" in consent_waiver._PENDING_AUDIT
    assert "r-new" not in consent_waiver._PENDING_AUDIT
    assert len(consent_waiver._PENDING_AUDIT) == consent_waiver._PENDING_AUDIT_LIMIT
    assert any("audit window is full" in record.getMessage()
               and "r-new" in record.getMessage()
               for record in caplog.records)


def test_a_chained_leg_carries_inherited_provenance_without_claiming_a_waiver(rig):
    """Leg 2 is not waived, but its pixels descend from a waiver and say so.

    Dropping the incoming stamp would let a clean refiner pass ship pixels
    descended from measured-wrong math with nothing in its result, while the
    ledger holds leg 1's row: the two provenance channels would disagree.
    `inherited` keeps them in agreement without leg 2 claiming a waiver it
    never spent, and leg 2 writes no use row.
    """
    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    request = _worker_request(rig, _model(), render_id="r-24")
    accuracy_waiver.activate(request)
    _bind_worker()
    base.assert_ulysses_only_padding(2, 1)
    leg1 = consent_waiver.stamp_result({}, [{"waived": accuracy_waiver.stamps()}])
    accuracy_waiver.clear()

    leg2 = consent_waiver.stamp_result({}, [{"waived": []}], leg1)
    carried = leg2[accuracy_waiver.STAMPED_RESULT_KEY]
    assert [entry["guard"] for entry in carried] == [RING_PAD]
    assert carried[0]["inherited"] is True
    assert all(entry.get("inherited") is not True
               for entry in leg1[accuracy_waiver.STAMPED_RESULT_KEY])
    rows = [json.loads(line) for line in
            open(rig.ledger.path, encoding="utf-8").read().splitlines()]
    assert len([row for row in rows if row.get("action") == "use"]) == 1
    # A third leg keeps the chain rather than growing it once per hop.
    leg3 = consent_waiver.stamp_result({}, [{"waived": []}], leg2)
    assert leg3[accuracy_waiver.STAMPED_RESULT_KEY] == carried


def test_an_fsdp_ceremony_that_aborts_on_a_class_k_guard_names_the_way_out(rig):
    """An FSDP preset with a padded sequence cannot proceed; the abort must name the way out.

    A ceremony renders without waivers on purpose, so a granted waiver does not
    clear the guard there, and the FSDP clean-reload PASS is unconditional.
    "aborted before PASS" alone, with a waiver live in the panel, leaves the
    operator nothing to act on.
    """
    _bind_worker()
    with pytest.raises(UnsupportedModelError) as raised:
        base.assert_ulysses_only_padding(2, 1)
    wrapped = RuntimeError("ActorError")
    wrapped.__cause__ = raised.value
    reason = consent_waiver.ceremony_abort_reason(wrapped)
    assert "aborted before PASS" in reason
    assert RING_PAD in reason
    assert "without fsdp" in reason and "#55" in reason
    # An abort with no class-K cause keeps the plain sentence.
    assert consent_waiver.ceremony_abort_reason(RuntimeError("nccl timeout")) == (
        consent_waiver._FSDP_ABORT)


@pytest.mark.parametrize("link", ["cause", "context"])
@pytest.mark.parametrize("truthiness", ["falsey", "explosive"])
def test_ceremony_abort_reason_never_tests_linked_exception_truthiness(
    rig,
    link,
    truthiness,
):
    """A falsey or hostile linked refusal keeps its exact guard and remedy."""
    _bind_worker()
    with pytest.raises(UnsupportedModelError) as raised:
        base.assert_ulysses_only_padding(2, 1)

    class TruthinessAbort(BaseException):
        pass

    class HostileRefusal(UnsupportedModelError):
        bool_calls = 0

        def __bool__(self):
            self.bool_calls += 1
            if truthiness == "falsey":
                return False
            raise TruthinessAbort("exception truthiness must not be evaluated")

    refusal_error = HostileRefusal(str(raised.value))
    wrapped = RuntimeError("ActorError")
    if link == "cause":
        wrapped.__cause__ = refusal_error
        # An explicit cause must win even when it reports false truthiness.
        wrapped.__context__ = RuntimeError("irrelevant implicit context")
    else:
        wrapped.__context__ = refusal_error

    reason = consent_waiver.ceremony_abort_reason(wrapped)

    assert refusal_error.bool_calls == 0
    assert "aborted before PASS" in reason
    assert RING_PAD in reason
    assert "without fsdp" in reason and "#55" in reason


def test_ceremony_abort_reason_is_total_for_hostile_baseexception_diagnostics(
    monkeypatch,
):
    class StopNow(BaseException):
        pass

    string_failure = StopNow("exception stringification interrupted")
    logging_failure = StopNow("fallback logging interrupted")

    class HostileFailure(RuntimeError):
        def __str__(self):
            raise string_failure

    monkeypatch.setattr(
        consent_waiver.log,
        "warning",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(logging_failure),
    )

    reason = consent_waiver.ceremony_abort_reason(HostileFailure("gate failed"))

    assert reason == consent_waiver._FSDP_ABORT


def test_a_class_k_environment_value_that_names_nothing_is_said_out_loud(
        rig, monkeypatch, caplog):
    """Headless has no panel, so an unreadable value must log a warning."""
    spec = consent_pending.KIND_SPECS[RING_KIND]
    monkeypatch.setenv(spec.env_var, "1")
    with caplog.at_level("WARNING"):
        assert consent_pending.resolve(
            kind=RING_KIND, path=rig.checkpoint,
            context=accuracy_waiver.waiver_context(
                combo_key="c", topology=RING2, world=2)) is None
    said = [record.getMessage() for record in caplog.records
            if "names nothing this build can waive" in record.getMessage()]
    assert len(said) == 1
    # One variable carries every wired kind, so the one warning an operator
    # gets must name every spelling, not the arbitrary subset that warned first.
    for kind in accuracy_waiver.KNOWN_WRONG_KINDS:
        assert repr(kind) in said[0]
        assert repr(consent_pending.KIND_SPECS[kind].default_guard) in said[0]
    # The scoped guard id an operator would copy out of a waiver row is the
    # other likely mistake, and it is not accepted for any kind.
    for scoped, kind in ((H3_RING_PAD, RING_KIND), (H3_SOL_ATTN, SOL_KIND)):
        monkeypatch.setenv(spec.env_var, scoped)
        assert consent_pending._env_grants(
            consent_pending.KIND_SPECS[kind]) is False
    monkeypatch.setenv(spec.env_var, spec.default_guard)
    assert consent_pending._env_grants(spec) is True


def test_a_value_naming_one_wired_kind_never_warns_from_another_kinds_arm(
        rig, monkeypatch, caplog):
    """A value meant for one kind must not draw a warning from another kind's arm.

    Every wired class-K kind shares one variable and the driver resolves them
    all per dispatch, so the ring arm must not report a value meant for sol as
    unreadable while the sol waiver lands (a false warning, recorded 2026-08-18).
    """
    monkeypatch.setenv("DGXM_WAIVE_KNOWN_WRONG", SOL_ATTN)
    context = accuracy_waiver.waiver_context(
        combo_key="c", topology=ULY2, world=2)
    with caplog.at_level("WARNING"):
        assert consent_pending.resolve(
            kind=RING_KIND, path=rig.checkpoint, context=context) is None
        granted = consent_pending.resolve(
            kind=SOL_KIND, path=rig.checkpoint, context=context)
    assert granted is not None and granted.consent_source == "env"
    assert not [record for record in caplog.records
                if "names nothing this build can waive" in record.getMessage()]


def test_the_whole_driver_loop_grants_only_the_kind_the_value_names(
        rig, monkeypatch, caplog):
    """The driver's own path: live_grants resolves every kind in one pass."""
    monkeypatch.setenv("DGXM_WAIVE_KNOWN_WRONG", SOL_ATTN)
    with caplog.at_level("WARNING"):
        grants = consent_waiver.live_grants(
            path=rig.checkpoint, combo_key="c", topo=_topo(ULY2), world=2)
    assert list(grants) == [SOL_KIND]
    assert not [record for record in caplog.records
                if "names nothing this build can waive" in record.getMessage()]


def test_a_typo_beside_a_good_name_is_still_reported(rig, monkeypatch, caplog):
    """A kind-level suppression would grant sol and say nothing about the typo."""
    monkeypatch.setenv("DGXM_WAIVE_KNOWN_WRONG", f"{SOL_ATTN},ring_pda")
    with caplog.at_level("WARNING"):
        grants = consent_waiver.live_grants(
            path=rig.checkpoint, combo_key="c", topo=_topo(ULY2), world=2)
    assert list(grants) == [SOL_KIND]
    said = [record.getMessage() for record in caplog.records
            if "names nothing this build can waive" in record.getMessage()]
    assert len(said) == 1
    assert "'ring_pda'" in said[0] and repr(SOL_ATTN) not in said[0].split(";")[0]


def test_every_wired_kind_reads_its_own_two_spellings_and_nothing_else(
        monkeypatch):
    """The loop covers every wired kind, so a third class-K kind is checked with no edit here."""
    for kind in accuracy_waiver.KNOWN_WRONG_KINDS:
        spec = consent_pending.KIND_SPECS[kind]
        for accepted in (spec.default_guard, spec.kind):
            monkeypatch.setenv(spec.env_var, accepted)
            assert consent_pending._env_grants(spec) is True, accepted
        for refused in ("1", f"{spec.default_guard}:minimax_h3"):
            monkeypatch.setenv(spec.env_var, refused)
            assert consent_pending._env_grants(spec) is False, refused


def test_the_warning_dedups_on_the_names_it_could_not_read(monkeypatch, caplog):
    spec = consent_pending.KIND_SPECS[RING_KIND]
    with caplog.at_level("WARNING"):
        accuracy_waiver.warn_unreadable_env(spec, "junk")
        accuracy_waiver.warn_unreadable_env(spec, "junk")
        accuracy_waiver.warn_unreadable_env(spec, "other")
    assert len([record for record in caplog.records
                if "names nothing this build can waive" in record.getMessage()]) == 2


def test_a_scoped_ledger_guard_is_answered_with_the_spelling_to_type(caplog):
    """The likeliest mistake is copying target_guard out of a waiver row."""
    spec = consent_pending.KIND_SPECS[SOL_KIND]
    with caplog.at_level("WARNING"):
        accuracy_waiver.warn_unreadable_env(spec, H3_SOL_ATTN)
    said = [record.getMessage() for record in caplog.records
            if "names nothing this build can waive" in record.getMessage()]
    assert len(said) == 1
    assert f"Did you mean {SOL_ATTN!r}?" in said[0]


@pytest.mark.parametrize("retired", ["sp_unvalidated", "waive-known-wrong:pixeldit-sp"])
def test_a_retired_spelling_is_never_suggested_back_to_the_operator(caplog, retired):
    """docs/TROUBLESHOOTING.md #55 names both PixelDiT values as spellings that look right.

    The audit vocabulary still carries the retired prefix, so a suggestion read
    from there would answer the value with itself and leave headless looping.
    """
    spec = consent_pending.KIND_SPECS[SOL_KIND]
    with caplog.at_level("WARNING"):
        accuracy_waiver.warn_unreadable_env(spec, retired)
    said = [record.getMessage() for record in caplog.records
            if "names nothing this build can waive" in record.getMessage()]
    assert len(said) == 1
    assert "Did you mean" not in said[0]


def test_the_class_k_guard_is_hidden_from_the_compiler():
    """A compiled DiT must never bake one render's waiver into a graph."""
    assert getattr(base.assert_ulysses_only_padding, "_torchdynamo_disable", False)


def test_the_active_row_carries_the_narrow_context_the_memo_is_keyed_on(rig):
    """The panel row must name the topology docs/TROUBLESHOOTING.md #55 promises it names."""
    _grant_from_the_panel(rig, RING_KIND, topology=RING2, world=2, guard=H3_RING_PAD)
    contexts = {row["kind"]: row["context"] for row in consent_store.list_active()}
    assert contexts[RING_KIND] == {"topology": "ring2", "world": "2"}
    # The opaque combination hash stays off the wire: it identifies nothing to
    # a reader and the row already carries the artifact basename.
    assert all("combo_key" not in row["context"] for row in consent_store.list_active())


def _funnel_worker(topology=RING2, world=2):
    """A bare worker whose store records each load `store_fsdp.ensure` passes to it."""
    loads: list[tuple[str, str, bool]] = []

    class _Store:
        def ensure(self, unet_name, options, lora_stack, **kwargs):
            loads.append((str(unet_name), str(kwargs.get("slot", "cond")),
                          "rescue_consent" in kwargs))
            return object(), "reuse"

    worker = types.SimpleNamespace(
        store=_Store(), topology=dict(topology), world=world)
    return worker, loads


def test_the_real_ring2_dispatch_order_leaves_the_grant_live_at_the_guard(rig):
    """The load order of the failed class-K acceptance run on hardware
    (recorded 2026-08-05), through the real `store_fsdp.ensure`.

    Its worker journal records exactly two cond ensures on the reused GPU
    thread: the loader node's eager push (logged `load`, no rescue consent,
    outside any dispatch) and then the sample's own load (logged `reuse`,
    inside it). Pin that this order arms the guard rather than disarming it.
    The eager push happens before `activate`, so it has nothing to disarm, and
    neither load the dispatch performs is treated as foreign. An uncond load
    arms and disarms nothing: store_fsdp.ensure binds only the cond slot.
    """
    from dgx_monarch.actor import store_fsdp, worker_env

    worker, loads = _funnel_worker()
    # 1. the loader node's eager push, before any grant exists
    store_fsdp.ensure(worker, UNET, {}, [], slot="cond", on_base_loaded=None)
    assert accuracy_waiver.granted(H3_RING_PAD) is None

    # 2. the click, and the driver's stamp on the re-queued dispatch
    _grant_from_the_panel(rig, RING_KIND, guard=H3_RING_PAD)
    request = _worker_request(rig, _model(), render_id="r-20")
    assert accuracy_waiver.activate(request) == 1

    # 3. everything the dispatch itself loads
    store_fsdp.ensure(worker, UNET, {}, [], slot="cond", on_base_loaded=None,
                      rescue_consent=worker_env.sample_rescue_consent(request))
    store_fsdp.ensure(worker, "uncond.safetensors", {}, [], slot="uncond",
                      on_base_loaded=None)
    assert loads == [(UNET, "cond", False), (UNET, "cond", True),
                     ("uncond.safetensors", "uncond", False)]

    # 4. mid-forward, both guards of the one kind proceed
    assert accuracy_waiver.granted(H3_RING_PAD) is not None
    assert accuracy_waiver.waived(H3_RING_PAD) is True
    base.assert_ulysses_only_padding(2, 1)   # the backstop must not refuse
    assert sorted(entry["guard"] for entry in accuracy_waiver.stamps()) == [
        RING_PAD, H3_RING_PAD]
    out = consent_waiver.stamp_result({}, [{"waived": accuracy_waiver.stamps()}])
    assert out[accuracy_waiver.STAMPED_RESULT_KEY]

    # 5. a cond load with no rescue consent, here for another checkpoint, disarms
    store_fsdp.ensure(worker, "other.safetensors", {}, [], slot="cond")
    assert accuracy_waiver.granted(H3_RING_PAD) is None
    assert accuracy_waiver.stamps() == []
    with pytest.raises(UnsupportedModelError):
        base.assert_ulysses_only_padding(2, 1)


def test_the_sample_paths_own_cond_load_is_what_arms_the_guard():
    """`store_fsdp.ensure` classifies by `rescue_consent`; pin who passes it.

    test_only_the_sample_paths_cond_load_counts_as_its_dispatch pins the ensure
    side. The sample path must pass the kwarg on its own cond load, or that load
    would disarm the grants the same function installed, and it must install
    them before it loads anything. The scheduler's sigma query must not pass it,
    because that load belongs to no dispatch.
    """
    import ast
    import inspect
    import textwrap

    from dgx_monarch.actor.sample_protocol import run_sample
    from dgx_monarch.actor.worker import GPUWorker

    def cond_loads(func):
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        found = []
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and getattr(node.func, "attr", "") == "ensure"):
                continue
            named = {kw.arg for kw in node.keywords}
            slot = next((kw.value.value for kw in node.keywords
                         if kw.arg == "slot"), "cond")
            if slot == "cond":
                found.append((node.lineno, named))
        return found

    sample = cond_loads(run_sample)
    assert sample, "the sample path must still load through the funnel"
    assert all("rescue_consent" in named for _line, named in sample)

    tree = ast.parse(textwrap.dedent(inspect.getsource(run_sample)))
    activated = [node.lineno for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and getattr(node.func, "attr", "") == "activate_accuracy_waivers"]
    assert len(activated) == 1, "one dispatch installs its grants exactly once"
    assert activated[0] < min(line for line, _named in sample), (
        "the grants must be installed before the dispatch loads anything")

    sigmas = cond_loads(GPUWorker._compute_sigmas_impl)
    assert sigmas, "the scheduler query must still load through the funnel"
    assert not any("rescue_consent" in named for _line, named in sigmas)


def test_a_typed_refusal_does_not_wedge_the_fleet_before_the_waiver_is_spent(
        rig, monkeypatch):
    """The failure of the class-K acceptance run (recorded 2026-08-05): the
    refusal blocked its own remedy.

    Every class-K refusal ends by telling the operator to click the card and
    queue the render again, and queueing again is the only way a waiver is
    spent. On hardware the click landed and the driver stamped the grant onto
    the re-queue, but that dispatch never reached a worker: the refused render
    had retired its lease as abandoned, and an abandoned lease blocks every
    later sample until the fleet is recycled.

    A typed refusal is not an abandonment. It is decided from the request,
    before the first collective and before any rank packs a latent, and it
    comes back through a live actor, so the render is over on every rank and
    its lease is consumed.
    """
    from monarch.actor import ActorError

    from dgx_monarch import mesh_setup
    from dgx_monarch.nodes import pending as pending_mod
    from dgx_monarch.nodes.pending import PendingRender

    # The refusal the worker raises before the click, card descriptor included.
    _bind_worker()
    with pytest.raises(UnsupportedModelError) as raised:
        base.assert_ulysses_only_padding(2, 1)
    assert parse_refusal_tag(str(raised.value)).refusal_class is RefusalClass.KNOWN_WRONG

    retired: list[str] = []
    monkeypatch.setattr(mesh_setup, "release_sample",
                        lambda _future: retired.append("release"))
    monkeypatch.setattr(mesh_setup, "abandon_sample",
                        lambda _future: retired.append("abandon"))
    monkeypatch.setattr(pending_mod, "_throw_if_comfy_interrupted", lambda: None)
    from dgx_monarch import telemetry

    monkeypatch.setattr(telemetry.render_progress, "finish", lambda *_a: None)
    wrapped = ActorError(RuntimeError(str(raised.value)))

    def collect(*_args, **_kwargs):
        raise wrapped

    handle = types.SimpleNamespace(collect_sample=collect, lock=None)
    pending = PendingRender(
        handle, object(), types.SimpleNamespace(__exit__=lambda *_a: None),
        None, {}, 1.0, "r-21", lambda *_a: {})

    with pytest.raises(ActorError):
        pending.result()

    assert retired == ["release"], (
        "a typed refusal must not leave an abandoned lease: the re-queue that "
        "spends the waiver is the very next dispatch")
    assert pending._state == "closed"


SOL_KIND = "waive-known-wrong:sol-attn"


def _sol_refusal():
    """The refusal the kernel wrapper raises with no grant in scope."""
    from dgx_monarch.adapters import sol_attention

    with pytest.raises(UnsupportedModelError) as raised:
        sol_attention._assert_waived("minimax_h3")
    return raised.value


def test_the_sol_attn_waiver_walks_refuse_card_consent_requeue_stamp_render(
        rig, monkeypatch):
    """The whole state machine, in the order an operator meets it.

    Each step is a state the lever passes through on hardware. A refusal with
    no card, a card that cannot be accepted, or an accepted card that does not
    reach the dispatch leaves the operator no way to render, and a refusal that
    abandons its lease wedges the fleet on the re-queue.
    """
    from monarch.actor import ActorError

    from dgx_monarch import consent_observe, mesh_setup
    from dgx_monarch.nodes import pending as pending_mod
    from dgx_monarch.nodes.pending import PendingRender

    # 1. REFUSE. No grant in scope, so the kernel wrapper refuses class K.
    accuracy_waiver.clear()
    _bind_worker(topology=ULY2)
    refusal_error = _sol_refusal()
    tag = parse_refusal_tag(str(refusal_error))
    assert tag.refusal_class is RefusalClass.KNOWN_WRONG
    assert tag.guard == H3_SOL_ATTN and is_waivable(str(refusal_error))

    # 2. CARD. The same driver frame the ring-pad guard uses turns it into one.
    card = consent_observe.observe_refusal(
        ActorError(RuntimeError(str(refusal_error))))
    assert card is not None
    assert card["kind"] == SOL_KIND and card["class"] == "K"
    assert card["style"] == "accuracy" and card["auto_eligible"] is False
    assert "2026-08-15" in card["measured"]

    # 3. LEASE. The refused render is over, not stranded: consumed, so the
    #    re-queue that spends the waiver runs on the same fleet.
    retired: list[str] = []
    monkeypatch.setattr(mesh_setup, "release_sample",
                        lambda _future: retired.append("release"))
    monkeypatch.setattr(mesh_setup, "abandon_sample",
                        lambda _future: retired.append("abandon"))
    monkeypatch.setattr(pending_mod, "_throw_if_comfy_interrupted", lambda: None)
    from dgx_monarch import telemetry

    monkeypatch.setattr(telemetry.render_progress, "finish", lambda *_a: None)
    wrapped = ActorError(RuntimeError(str(refusal_error)))

    def collect(*_args, **_kwargs):
        raise wrapped

    handle = types.SimpleNamespace(collect_sample=collect, lock=None)
    render = PendingRender(
        handle, object(), types.SimpleNamespace(__exit__=lambda *_a: None),
        None, {}, 1.0, "sol-1", lambda *_a: {})
    with pytest.raises(ActorError):
        render.result()
    assert retired == ["release"], (
        "a typed refusal must not abandon its lease: the re-queue that spends "
        "the waiver is the very next dispatch")

    # 4. CONSENT. One click on the card the refusal raised.
    _grant_from_the_panel(rig, SOL_KIND, topology=ULY2, guard=H3_SOL_ATTN)

    # 5. RE-QUEUE AND STAMP. The driver puts the grant on the wire.
    request = _worker_request(rig, _model(), topology=ULY2, render_id="sol-2")
    assert list(request[accuracy_waiver.REQUEST_KEY]) == [SOL_KIND]

    # 6. RENDER. The worker activates it and the guard now proceeds, twice,
    #    because the same render calls attention once per block.
    accuracy_waiver.clear()
    assert accuracy_waiver.activate(request) == 1
    _bind_worker(topology=ULY2)
    from dgx_monarch.adapters import sol_attention

    sol_attention._assert_waived("minimax_h3")
    sol_attention._assert_waived("minimax_h3")

    # 7. STAMPED. One entry, not two: the ledger records the dispatch, not the
    #    call count.
    stamps = accuracy_waiver.stamps()
    assert len(stamps) == 1
    assert stamps[0]["guard"] == H3_SOL_ATTN and stamps[0]["kind"] == SOL_KIND
    assert accuracy_waiver.stamp_text(SOL_KIND).startswith("rendered-under-waiver")


def test_a_revoked_sol_waiver_refuses_the_next_render(rig):
    """Grants are read per call, so a revocation lands on the next render."""
    from dgx_monarch.adapters import sol_attention

    _grant_from_the_panel(rig, SOL_KIND, topology=ULY2, guard=H3_SOL_ATTN)
    request = _worker_request(rig, _model(), topology=ULY2, render_id="sol-3")
    accuracy_waiver.clear()
    assert accuracy_waiver.activate(request) == 1
    _bind_worker(topology=ULY2)
    sol_attention._assert_waived("minimax_h3")

    accuracy_waiver.clear()  # the next dispatch carries no grant
    _bind_worker(topology=ULY2)
    with pytest.raises(UnsupportedModelError):
        sol_attention._assert_waived("minimax_h3")


SHARD_KIND = "waive-known-wrong:shard-quant"
CFG2 = {"ulysses": 1, "ring": 1, "cfg": 2, "dp": 1, "fsdp": False}


def test_a_granted_class_k_memo_stays_live_until_it_is_revoked(rig):
    """A class-K grant is honored until it is revoked, not spent on first use.

    The stamp rides each request and never the fleet's worker args, so the
    driver re-resolves the grant on every dispatch and a revoke lands on the
    next render. Nothing consumes the memo, which is why one waived chroma cfg2
    cell in the 2026-09-06 sweep left three later cells of the same shape
    rendering under its grant.
    """
    card, _combo = _grant_from_the_panel(
        rig, SHARD_KIND, topology=CFG2, guard=CHROMA_SHARD_QUANT)
    model = _model()

    # Three dispatches, one click: every one of them carries the same grant.
    ids = []
    for number in range(3):
        request = _worker_request(rig, model, topology=CFG2,
                                  render_id=f"r-{number}")
        grants = request.get(accuracy_waiver.REQUEST_KEY) or {}
        assert SHARD_KIND in grants, f"dispatch {number} carried no grant"
        ids.append(grants[SHARD_KIND]["id"])
        consent_waiver.retire_audit(f"r-{number}")
    assert len(set(ids)) == 1, "one memo, so one consent id on every dispatch"

    # And the revoke is what ends it, on the next dispatch and not later.
    status, payload = consent_routes.handle_action(
        {"action": "revoke", "key": card["key"]})
    assert status == 200 and payload["ok"], payload
    after = _worker_request(rig, model, topology=CFG2, render_id="r-3")
    assert accuracy_waiver.REQUEST_KEY not in after


@pytest.mark.parametrize(("kind", "guard"), [
    (RING_KIND, H3_RING_PAD),
    ("waive-known-wrong:sol-attn", H3_SOL_ATTN),
])
def test_the_ring_pad_and_sol_attn_grants_live_until_revoked_too(rig, kind, guard):
    """The ring_pad and sol_attn grants have the same lifetime: nothing consumes
    the memo, so every dispatch of the combination, topology and world carries
    the grant, and the revoke ends it on the next."""
    card, _combo = _grant_from_the_panel(rig, kind, topology=RING2, guard=guard)
    model = _model()
    ids = []
    for number in range(3):
        request = _worker_request(rig, model, topology=RING2, render_id=f"k-{number}")
        grants = request.get(accuracy_waiver.REQUEST_KEY) or {}
        assert kind in grants, f"dispatch {number} carried no grant"
        ids.append(grants[kind]["id"])
        consent_waiver.retire_audit(f"k-{number}")
    assert len(set(ids)) == 1, "one memo, so one consent id on every dispatch"
    status, payload = consent_routes.handle_action(
        {"action": "revoke", "key": card["key"]})
    assert status == 200 and payload["ok"], payload
    after = _worker_request(rig, model, topology=RING2, render_id="k-3")
    assert accuracy_waiver.REQUEST_KEY not in after


@pytest.mark.parametrize(("kind", "guard"), [
    (RING_KIND, H3_RING_PAD),
    ("waive-known-wrong:sol-attn", H3_SOL_ATTN),
])
def test_the_ring_pad_and_sol_attn_grants_stop_at_their_topology_and_world(rig, kind, guard):
    _grant_from_the_panel(rig, kind, topology=RING2, guard=guard)
    model = _model()
    assert accuracy_waiver.REQUEST_KEY in _worker_request(
        rig, model, topology=RING2, render_id="s-a")
    consent_waiver.retire_audit("s-a")
    assert accuracy_waiver.REQUEST_KEY not in _worker_request(
        rig, model, topology=ULY2, render_id="s-b")
    consent_waiver.retire_audit("s-b")
    assert accuracy_waiver.REQUEST_KEY not in _worker_request(
        rig, model, topology=RING2, world=4, render_id="s-c")


def test_a_class_k_memo_does_not_reach_another_topology_or_world(rig):
    """The scope that does bound it: the combination, the topology and world."""
    _grant_from_the_panel(rig, SHARD_KIND, topology=CFG2, guard=CHROMA_SHARD_QUANT)
    model = _model()
    assert accuracy_waiver.REQUEST_KEY in _worker_request(
        rig, model, topology=CFG2, render_id="r-a")
    consent_waiver.retire_audit("r-a")
    assert accuracy_waiver.REQUEST_KEY not in _worker_request(
        rig, model, topology=ULY2, render_id="r-b")
    consent_waiver.retire_audit("r-b")
    assert accuracy_waiver.REQUEST_KEY not in _worker_request(
        rig, model, topology=CFG2, world=4, render_id="r-c")
