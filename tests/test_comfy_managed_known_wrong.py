"""A class-K guard that blocks the ceremony, under comfy-managed residency.

A ceremony strips accuracy waivers on purpose, so a waivable class-K guard
refuses both of its legs, and a combination whose first-use proof render meets
one can never reach PASS. Under `comfy_managed` a class-P answer before
dispatch would name "render this combination once with auto_gate on" as the
remedy, which is the loop the render is already in (found 2026-09-09).

What is pinned here:

* the class-K refusal and the card it carries answer first, instead of being
  swallowed into the forced-stock fallback the class-P check then reads;
* every other residency policy keeps a stock fallback, so the abort stays
  swallowed there;
* a live grant stops the class-K answer, and the residency check then names the
  guard rather than promising a ceremony that cannot run;
* with `auto_gate` off the same combination is authorized and its dispatch
  carries the waiver.
"""
from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace

import pytest

from dgx_monarch import accuracy_waiver, consent_pending, consent_store, mesh_residency
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.nodes import (
    auto_gate,
    common,
    consent_observe,
    consent_routes,
    consent_waiver,
    gate,
    gate_inconclusive,
    gate_process_state,
)
from dgx_monarch.nodes.gate_identity import NormalRenderAuthorization
from dgx_monarch.refusal import RefusalClass, parse_refusal_tag, refusal

UNET = "chroma-nvfp4.safetensors"
SHARD_QUANT = "shard_quant_scale"
CHROMA_SHARD_QUANT = "shard_quant_scale:chroma"
SHARD_QUANT_KIND = "waive-known-wrong:shard-quant"
CFG2 = {"ulysses": 1, "ring": 1, "cfg": 2, "dp": 1, "fsdp": False}
WORLD = 2
TOKEN = ("combo", "artifact", "known", "context")


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """One driver: a checkpoint, a consent store, and empty process state."""
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
    _reset()
    yield SimpleNamespace(checkpoint=str(checkpoint), output=str(output))
    _reset()


def _reset() -> None:
    consent_pending.clear_all()
    accuracy_waiver.clear()
    consent_waiver.reset_pending_audit()
    consent_waiver.reset_ceremony_blockers()


def _patch_auto_state(monkeypatch) -> None:
    """Give one test its own process-local gate state and context seam."""
    lock = threading.Lock()
    monkeypatch.setattr(
        gate_process_state, "_AUTO_GATE_ACTIVE", SimpleNamespace(on=False))
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_LOCK", lock)
    monkeypatch.setattr(
        gate_process_state, "_AUTO_GATE_CONDITION", threading.Condition(lock))
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_RUNNING", set())
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_SESSION", {})
    monkeypatch.setattr(gate_process_state, "_PROCESS_GATE_DENIALS", {})
    monkeypatch.setattr(gate_process_state, "_AUTO_GATE_WAIT_S", 1.0)
    monkeypatch.setattr(
        common, "_auto_gate_context", lambda *_args: ("unknown", TOKEN))


def _model(comfy_managed: bool = True):
    worker_args = {"comfy_managed": True} if comfy_managed else {}
    handle = SimpleNamespace(
        world=WORLD,
        effective_worker_args=lambda requested: {**worker_args, **dict(requested)},
    )
    mesh = SimpleNamespace(worker_args=dict(worker_args), handle=handle,
                           attention="TORCH_FLASH", auto_gate="first_use")
    return SimpleNamespace(unet_name=UNET, options={}, loras=(), mesh=mesh)


def _worker_refusal() -> UnsupportedModelError:
    """The refusal the sample path raises, built the way that site builds it."""
    accuracy_waiver.bind_model(UNET, {}, [], CFG2, WORLD, dispatch=True)
    text = refusal(
        RefusalClass.KNOWN_WRONG,
        "this chroma render quantizes its nvfp4 activations against one scale "
        "shared across the sharded group and still does not match a one-GPU "
        "render.",
        guard=CHROMA_SHARD_QUANT,
        waivable=True,
        panel_action=accuracy_waiver.panel_action(CHROMA_SHARD_QUANT),
        troubleshooting=95,
    ) + accuracy_waiver.card_tail(CHROMA_SHARD_QUANT, family_hint="chroma")
    accuracy_waiver.clear()
    return UnsupportedModelError(text)


def _combo(rig) -> str:
    from dgx_monarch.nodes.consent_rescue import combo_identity

    return combo_identity(UNET, {}, rig.checkpoint)[0]


def _grant_from_the_panel(rig) -> dict:
    """Grant the shard-quant waiver the way the panel does: card, then a click."""
    card = consent_observe.observe_refusal(_worker_refusal())
    assert card is not None, "the class-K refusal must carry a one-click card"
    status, payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"]})
    assert status == 200 and payload["ok"], payload
    return card


def _unwaivable_worker_refusal() -> UnsupportedModelError:
    """A class-K refusal for a guard no consent can clear in this release."""
    return UnsupportedModelError(refusal(
        RefusalClass.KNOWN_WRONG,
        "this render does not match a one-GPU render and no waiver is wired "
        "for the guard that measured it.",
        guard=CHROMA_SHARD_QUANT,
        waivable=False,
        troubleshooting=95,
    ))


def _run_blocked_ceremony(monkeypatch, model, aborts_with=None):
    """Drive one first-use ceremony that aborts on the class-K guard."""
    _patch_auto_state(monkeypatch)
    build = _worker_refusal if aborts_with is None else aborts_with
    monkeypatch.setattr(
        gate, "run_identity_ceremony",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(build()))
    return auto_gate.maybe_auto_gate(
        model, {"kind": "ksampler", "steps": 2}, {}, 1.0, 2)


def test_a_blocked_ceremony_answers_with_its_class_k_card_under_comfy_managed(
        rig, monkeypatch):
    """The refusal the operator can clear reaches them first.

    Without this the abort is swallowed, the render falls back to stock, and
    the comfy-managed residency check answers class P with a remedy that is the
    ceremony that just failed.
    """
    with pytest.raises(UnsupportedModelError) as raised:
        _run_blocked_ceremony(monkeypatch, _model(comfy_managed=True))
    tag = parse_refusal_tag(str(raised.value))
    assert tag is not None
    assert tag.refusal_class is RefusalClass.KNOWN_WRONG
    assert tag.guard == CHROMA_SHARD_QUANT and tag.waivable
    card = consent_observe.observe_refusal(raised.value)
    assert card is not None and card["kind"] == SHARD_QUANT_KIND
    assert consent_waiver.ceremony_blocker(_combo(rig)) == CHROMA_SHARD_QUANT


def test_a_granted_waiver_lets_the_authorization_carry_on_past_the_card(
        rig, monkeypatch):
    """A question already answered raises no second card and no refusal."""
    _grant_from_the_panel(rig)
    assert _run_blocked_ceremony(
        monkeypatch,
        _model(comfy_managed=True)) == gate_inconclusive.KNOWN_WRONG_ABORT_VERDICT
    assert consent_waiver.ceremony_blocker(_combo(rig)) == CHROMA_SHARD_QUANT


def test_every_other_residency_keeps_its_stock_fallback(rig, monkeypatch):
    """Only comfy-managed residency lacks the fallback that makes the swallow
    right: elsewhere the full render runs on stock and meets the guard itself."""
    assert _run_blocked_ceremony(
        monkeypatch,
        _model(comfy_managed=False)) == gate_inconclusive.KNOWN_WRONG_ABORT_VERDICT
    assert consent_waiver.ceremony_blocker(_combo(rig)) == CHROMA_SHARD_QUANT


def test_an_abort_with_no_waivable_guard_is_still_swallowed(rig, monkeypatch):
    """An untagged ceremony failure keeps the forced-stock fallback."""
    _patch_auto_state(monkeypatch)
    monkeypatch.setattr(
        gate, "run_identity_ceremony",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert auto_gate.maybe_auto_gate(
        _model(comfy_managed=True), {"kind": "ksampler", "steps": 2},
        {}, 1.0, 2) == "ERROR"
    assert consent_waiver.ceremony_blocker(_combo(rig)) is None


def test_a_class_k_guard_no_consent_can_clear_keeps_the_stock_fallback(
        rig, monkeypatch):
    """A guard with no wired consent has no card to offer, so nothing changes.

    The abort stays swallowed, nothing is remembered, and the generic class-P
    refusal is still the one this combination reads.
    """
    from dgx_monarch.residency_mode import ComfyManagedResidencyError

    model = _model(comfy_managed=True)
    assert _run_blocked_ceremony(
        monkeypatch, model, aborts_with=_unwaivable_worker_refusal) == "ERROR"
    assert consent_waiver.ceremony_blocker(_combo(rig)) is None
    consent_waiver.assert_ceremony_not_blocked(
        model, model.mesh.handle, _authorization(model))
    request = {"model": {"unet_name": UNET, "options": {}, "loras": []},
               "_dgxm_normal_residency_mode": "stock"}
    with pytest.raises(ComfyManagedResidencyError) as raised:
        mesh_residency.assert_normal_render_residency_mode(
            request, {"comfy_managed": True})
    assert "the driver could not authorize" in str(raised.value)


def _authorization(model, residency: str = "stock") -> NormalRenderAuthorization:
    return NormalRenderAuthorization(
        dict(model.mesh.worker_args),
        {"unet_name": UNET, "options": {}, "loras": []},
        None,
        residency,
    )


def test_the_residency_refusal_names_the_guard_and_stops_promising_a_ceremony(
        rig):
    model = _model(comfy_managed=True)
    consent_waiver.remember_ceremony_blocker(_combo(rig), CHROMA_SHARD_QUANT)
    from dgx_monarch.residency_mode import ComfyManagedResidencyError

    with pytest.raises(ComfyManagedResidencyError) as raised:
        consent_waiver.assert_ceremony_not_blocked(
            model, model.mesh.handle, _authorization(model))
    text = str(raised.value)
    assert parse_refusal_tag(text).refusal_class is RefusalClass.PHYSICS
    assert CHROMA_SHARD_QUANT in text
    assert "auto_gate widget to off" in text
    assert "render this combination once with auto_gate=first_use" not in text
    # The card expires after half an hour and a class-P refusal carries no
    # panel sentence, so the variable has to be in the text itself.
    assert f"DGXM_WAIVE_KNOWN_WRONG={SHARD_QUANT}" in text


def test_a_combination_with_no_remembered_blocker_reads_the_generic_refusal(rig):
    """Nothing remembered means nothing to name, so the generic class P stands."""
    model = _model(comfy_managed=True)
    consent_waiver.assert_ceremony_not_blocked(
        model, model.mesh.handle, _authorization(model))
    request = {"model": {"unet_name": UNET, "options": {}, "loras": []},
               "_dgxm_normal_residency_mode": "stock"}
    from dgx_monarch.residency_mode import ComfyManagedResidencyError

    with pytest.raises(ComfyManagedResidencyError) as raised:
        mesh_residency.assert_normal_render_residency_mode(
            request, {"comfy_managed": True})
    assert "the driver could not authorize" in str(raised.value)


def test_an_authorized_render_is_never_refused_by_the_blocker_check(rig):
    """A grant-backed or operator-off dispatch never reaches this refusal."""
    model = _model(comfy_managed=True)
    consent_waiver.remember_ceremony_blocker(_combo(rig), CHROMA_SHARD_QUANT)
    for residency in ("required", "operator_off", "gate_internal"):
        consent_waiver.assert_ceremony_not_blocked(
            model, model.mesh.handle, _authorization(model, residency))
    stock = _model(comfy_managed=False)
    consent_waiver.assert_ceremony_not_blocked(
        stock, stock.mesh.handle, _authorization(stock))


def test_with_auto_gate_off_the_waived_combination_is_authorized_and_stamped(
        rig, monkeypatch):
    """The remedy the refusal names, end to end on the driver side."""
    _grant_from_the_panel(rig)
    monkeypatch.setattr(consent_waiver, "_gate_ceremony_active", lambda: False)
    model = _model(comfy_managed=True)
    model.mesh.auto_gate = "off"
    consent_waiver.remember_ceremony_blocker(_combo(rig), CHROMA_SHARD_QUANT)
    consent_waiver.assert_ceremony_not_blocked(
        model, model.mesh.handle, _authorization(model, "operator_off"))
    request = {"model": {"unet_name": UNET, "options": {}, "loras": []},
               "_dgxm_normal_residency_mode": "operator_off",
               "_dgxm_render_id": "r-425"}
    assert consent_waiver.stamp_request(
        request, model, SimpleNamespace(**CFG2), WORLD, "r-425") == 1
    assert accuracy_waiver.REQUEST_KEY in request
    assert mesh_residency.assert_normal_render_residency_mode(
        request, {"comfy_managed": True}) == "operator_off"


def test_the_submit_asks_before_it_takes_a_setup_or_dispatches():
    """The blocker check runs before the setup lease and before dispatch, so a
    refusal leaves no lease taken and no rank changed."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"
              / "nodes" / "render_submit.py").read_text()
    asked = source.index("assert_ceremony_not_blocked")
    assert asked < source.index("ensure_request_setup")
    assert asked < source.index("submit_sample")
    assert source.index("authorize_normal_render(") < asked


class _ReachedSetup(Exception):
    """Raised by the setup stub: the submit got past the blocker check."""


def _driver_submit(monkeypatch, request):
    """One real submit down the driver spine, stopped at the setup lease."""
    import dgx_monarch.nodes.render_submit as submit_mod
    from dgx_monarch import mesh_safety, mesh_setup
    from render_sessions_helpers import _real_handle, _stub_direct_submit

    dispatched: list = []
    handle = _real_handle()
    spec = _stub_direct_submit(
        monkeypatch, handle,
        lambda *args, **kwargs: dispatched.append(args) or object())
    spec.mesh.worker_args["comfy_managed"] = True
    monkeypatch.setattr(
        mesh_setup, "ensure_request_setup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(_ReachedSetup()))
    model_request = {"unet_name": UNET, "options": {}, "loras": []}
    monkeypatch.setattr(
        submit_mod, "authorize_normal_render",
        lambda *_args, **_kwargs: NormalRenderAuthorization(
            dict(spec.mesh.worker_args), model_request, None, "stock"))
    consent_waiver.remember_ceremony_blocker(
        mesh_safety.request_combo_key(model_request), CHROMA_SHARD_QUANT)
    return spec, handle, dispatched, request


def test_the_submit_itself_refuses_before_it_takes_a_setup_or_dispatches(
        rig, monkeypatch):
    """The call is wired and live, not just written above the setup line."""
    import dgx_monarch.nodes.common as common
    from dgx_monarch.nodes.pending import PendingRenderHandoff
    from dgx_monarch.nodes.render_session import RenderSession
    from dgx_monarch.residency_mode import ComfyManagedResidencyError

    spec, handle, dispatched, request = _driver_submit(monkeypatch, {})
    with pytest.raises(ComfyManagedResidencyError) as raised:
        common.submit_render(spec, request, {"samples": object()},
                             cfg_value=1.0, steps_hint=2,
                             handoff=PendingRenderHandoff())
    assert CHROMA_SHARD_QUANT in str(raised.value)
    assert dispatched == []
    # The refusal releases the render session it claimed.
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_a_dual_model_submit_keeps_the_refusal_that_names_the_second_model(
        rig, monkeypatch):
    """A second unconditional model is stamped stock whatever a ceremony did.

    Turning auto_gate off does not change that, so a refusal that names the
    guard would point the operator at a remedy that cannot help. The generic
    one names the real cause.
    """
    import dgx_monarch.nodes.common as common
    from dgx_monarch.nodes.pending import PendingRenderHandoff
    from dgx_monarch.residency_mode import ComfyManagedResidencyError

    uncond = {"unet_name": UNET, "options": {}, "loras": []}
    spec, _handle, _dispatched, request = _driver_submit(
        monkeypatch, {"uncond_model": dict(uncond)})
    with pytest.raises(_ReachedSetup):
        common.submit_render(spec, request, {"samples": object()},
                             cfg_value=1.0, steps_hint=2,
                             handoff=PendingRenderHandoff())
    request = {"model": {"unet_name": UNET, "options": {}, "loras": []},
               "uncond_model": dict(uncond),
               "_dgxm_normal_residency_mode": "stock"}
    with pytest.raises(ComfyManagedResidencyError) as raised:
        mesh_residency.assert_normal_render_residency_mode(
            request, {"comfy_managed": True})
    assert "second (unconditional) model" in str(raised.value)


def test_the_remembered_blockers_are_bounded_and_forgettable():
    for index in range(consent_waiver._CEREMONY_BLOCKER_LIMIT + 8):
        consent_waiver.remember_ceremony_blocker(f"combo-{index}", SHARD_QUANT)
    assert len(consent_waiver._CEREMONY_BLOCKERS) == \
        consent_waiver._CEREMONY_BLOCKER_LIMIT
    assert consent_waiver.ceremony_blocker("combo-0") is None
    consent_waiver.reset_ceremony_blockers()
    assert consent_waiver.ceremony_blocker("combo-70") is None
