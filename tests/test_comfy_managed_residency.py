"""CPU contracts for opt-in comfy-managed residency.

ComfyUI's DynamicVRAM (comfy-aimdo) is an alternative to stock cudaMalloc and
zero-copy slab residency. This worker policy is never automatic.
The comfy-managed suites enforce:

* LoRA swap refusal at ``actor/store_fsdp.ensure`` before ComfyUI's manager
  can unpatch a model under pool pressure and replace a weight with a
  sentinel (docs/DESIGN.md section 5.9, "The residency ladder").
* A successful process-local bring-up before the rung is claimed.
* A worker-arg key only when enabled, preserving existing classic PASS keys.
* Fail-closed authorization and quarantine: this bootstrap policy cannot be
  disabled inside a live worker.
* Source checks for bring-up order, swap isolation, and excluded monkeypatches.

All environment, capacity, and import dependencies are injected or patched;
these tests run no ComfyUI, CUDA, or aimdo.
"""
from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest

from comfy_managed_helpers import (  # noqa: F401  # autouse fixture import.
    STACK,
    _isolated_process_state,
    _RecordingStore,
    _resolve,
    _tag,
    _tree,
    _worker,
)
from comfy_managed_helpers import rung_on as rung_on  # a fixture tests ask for by name.
from dgx_monarch import mesh_safety, refusal, residency_mode
from dgx_monarch.actor import store_fsdp, store_residency
from dgx_monarch.refusal import RefusalClass


def test_the_three_residency_modes_are_named_in_one_place():
    assert residency_mode.RESIDENCY_MODES == frozenset({
        residency_mode.MODE_STOCK,
        residency_mode.MODE_SLAB,
        residency_mode.MODE_COMFY_MANAGED,
    })
    assert residency_mode.MODE_COMFY_MANAGED == "comfy_managed"
    assert residency_mode.WORKER_ARG == "comfy_managed"
    assert residency_mode.ENV_ACTIVE == "DGXM_COMFY_MANAGED"
    assert residency_mode.TROUBLESHOOTING == 62


@pytest.mark.parametrize("worker_args", [
    None, {}, {"comfy_managed": False}, {"comfy_managed": "off"},
    {"comfy_managed": "on"}, {"comfy_managed": 1}, {"comfy_managed": "false"},
])
def test_requested_is_is_true_only(worker_args):
    """bool("off") is True, so only the literal boolean True turns the rung on.

    A truthy string from a hand-edited cluster.toml, a JSON 1, or an absent key
    all mean off.
    """
    assert residency_mode.requested(worker_args) is False


def test_requested_accepts_the_boolean_true():
    assert residency_mode.requested({"comfy_managed": True}) is True


def test_pricing_value_is_the_one_answer_both_footprint_sites_read():
    assert residency_mode.pricing_value(
        {"comfy_managed": True, "slab_weights": False}
    ) == residency_mode.MODE_COMFY_MANAGED
    assert residency_mode.pricing_value({"slab_weights": True}) is True
    assert residency_mode.pricing_value({"slab_weights": False}) is False
    assert residency_mode.pricing_value({"slab_weights": "auto"}) == "auto"
    assert residency_mode.pricing_value({}) is None
    assert residency_mode.pricing_value(None) is None


def test_charges_legacy_arena_is_true_only_for_explicit_slab_off():
    assert residency_mode.charges_legacy_arena(False) is True
    assert residency_mode.charges_legacy_arena(True) is False
    assert residency_mode.charges_legacy_arena("auto") is False
    assert residency_mode.charges_legacy_arena(None) is False
    # The third residency must not fall into the legacy branch by omission.
    assert residency_mode.charges_legacy_arena(residency_mode.MODE_COMFY_MANAGED) is False


def test_arena_omitted_note_names_the_actual_reason():
    managed = residency_mode.arena_omitted_note(residency_mode.MODE_COMFY_MANAGED)
    assert "comfy-managed" in managed
    assert "slab_weights is auto" not in managed
    assert "slab" in residency_mode.arena_omitted_note(True)
    assert "auto" in residency_mode.arena_omitted_note("auto")
    assert "auto" in residency_mode.arena_omitted_note(None)


def test_the_rungs_error_is_not_a_capacity_error():
    """A ceremony leg must never auto-skip these as CAPACITY.

    ``mesh_safety.is_stock_load_capacity_error`` is what makes a gate leg
    record a CAPACITY verdict and move on. Nothing this error carries is about
    capacity (the rung's capacity wall raises StockLoadCapacityError), so a
    ceremony that hits one of these must not treat it as a skippable
    box-is-full result.
    """
    exc = residency_mode.ComfyManagedResidencyError("nope")
    assert not isinstance(exc, mesh_safety.StockLoadCapacityError)
    assert mesh_safety.is_stock_load_capacity_error(exc) is False
    assert isinstance(exc, RuntimeError)


def test_no_new_guard_was_added_to_the_frozen_registry():
    """A new guard would extend the v9 consent vocabulary and burn the protocol.

    Both frozen tables are asserted here, next to the refusals they protect, so
    a later change that reaches for a `comfy_managed` guard fails with the
    reason rather than with a vocabulary drift message three files away.
    """
    from dgx_monarch.gate_audit_vocab import RESIDENCY_KINDS
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION

    assert "comfy_managed" not in refusal.GUARDS
    assert not any("comfy_managed" in name or "aimdo" in name for name in refusal.GUARDS)
    assert RESIDENCY_KINDS == frozenset({"slab"})
    assert GATE_PROTOCOL_VERSION == 13


def test_rung_zero_outranks_an_explicit_slab_request(rung_on):
    """Rung 0 is a property of the process, not of the request.

    Nothing below it can be true at the same time (the schema and the Init node
    both refuse comfy_managed beside a truthy slab_weights, and apply_policy
    forces both levers off), so resolving it first turns a contradiction into a
    decision instead of a race between two explicit answers.
    """
    decision = _resolve(slab_weights=True)
    assert decision.rung == store_residency.RUNG_COMFY_MANAGED
    assert decision.use_slab is False


def test_rung_zero_is_a_stock_shaped_decision_that_never_retries(rung_on):
    decision = _resolve()
    assert decision.use_slab is False
    assert decision.fit is None
    assert decision.auto_retry_eligible is False
    assert decision.consent_id == ""
    assert decision.reason


def test_rung_zero_never_touches_the_fit_probe(rung_on):
    def explode(_path, _options):
        raise AssertionError("the fit probe must not run under comfy-managed residency")

    assert _resolve(fit_probe=explode).rung == store_residency.RUNG_COMFY_MANAGED


def test_rung_zero_returns_above_the_slab_probe_too(rung_on, monkeypatch):
    """Managed placement owns capacity policy and never reaches a slab.

    The three slab rungs share one price, taken once near the top of the
    ladder. Rung 0 returns above that line: a managed worker pays neither
    probe, and its own wall is ``preload_capacity_check``.
    """
    from dgx_monarch import capacity_fit

    def explode(_path, _options, **_kwargs):
        raise AssertionError("the slab probe must not run under comfy-managed residency")

    monkeypatch.setattr(capacity_fit, "slab_load_fit", explode)
    decision = _resolve(slab_weights=True)
    assert decision.rung == store_residency.RUNG_COMFY_MANAGED
    assert decision.slab_fit is None


def test_the_residency_property_is_three_valued(rung_on):
    assert _resolve().residency == residency_mode.MODE_COMFY_MANAGED


def test_the_residency_property_keeps_its_two_classic_values():
    assert _resolve(slab_weights=True).residency == residency_mode.MODE_SLAB
    assert _resolve(slab_weights=False).residency == residency_mode.MODE_STOCK


def test_the_late_bound_probe_can_be_injected_without_the_environment():
    """Same seam discipline as compile_dit: the default binds through the
    module at call time, so a caller with better information substitutes it."""
    decision = _resolve(comfy_managed=lambda: True)
    assert decision.rung == store_residency.RUNG_COMFY_MANAGED
    assert _resolve(comfy_managed=lambda: False).rung != store_residency.RUNG_COMFY_MANAGED


def test_comfy_managed_active_reads_only_the_environment(monkeypatch):
    assert store_residency.comfy_managed_active() is False
    monkeypatch.setenv(residency_mode.ENV_ACTIVE, "1")
    assert store_residency.comfy_managed_active() is True
    assert residency_mode.active() is True
    monkeypatch.setenv(residency_mode.ENV_ACTIVE, "yes")
    assert store_residency.comfy_managed_active() is False


@pytest.mark.parametrize(("slab_weights", "expected_rung"), [
    (True, store_residency.RUNG_EXPLICIT),
    (False, store_residency.RUNG_EXPLICIT_STOCK),
    ("auto", store_residency.RUNG_STOCK_FITS),
])
def test_every_classic_rung_resolves_exactly_as_before_when_the_rung_is_off(
        slab_weights, expected_rung):
    assert _resolve(slab_weights=slab_weights).rung == expected_rung


def test_describe_reports_the_third_residency_from_the_rung(rung_on):
    stored = SimpleNamespace(
        slab=None, residency_rung=store_residency.RUNG_COMFY_MANAGED)
    evidence = store_residency.describe(stored)
    assert evidence["residency"] == residency_mode.MODE_COMFY_MANAGED
    assert evidence["residency_rung"] == store_residency.RUNG_COMFY_MANAGED
    assert evidence["slab_certificate"] is None


def test_describe_still_reports_stock_and_slab():
    stock = SimpleNamespace(slab=None, residency_rung=store_residency.RUNG_STOCK_FITS)
    assert store_residency.describe(stock)["residency"] == "stock"
    slab = SimpleNamespace(
        slab=SimpleNamespace(certificate=None), residency_rung=store_residency.RUNG_EXPLICIT)
    assert store_residency.describe(slab)["residency"] == "slab"
    assert store_residency.describe(None)["residency"] is None


def test_summary_line_renders_the_third_residency():
    stored = SimpleNamespace(
        slab=None, residency_rung=store_residency.RUNG_COMFY_MANAGED)
    assert store_residency.summary_line(stored) == (
        "residency=comfy_managed rung=comfy_managed")


def test_a_lora_stack_refuses_at_the_funnel_before_any_load(rung_on):
    """The sentinel-trap doctrine's main pin.

    A graph rendered LoRA-less under the rung and then given one LoRA must
    refuse at ``store_fsdp.ensure``, before ``ModelStore.ensure`` runs;
    otherwise the baked hot-swap arm puts comfy's ModelPatcherDynamic straight
    into a swap with nothing refusing. docs/DESIGN.md section 5.9 says why the
    yield sits there ("Where the yields live, and why it is not in the ladder").
    """
    worker = _worker()
    with pytest.raises(residency_mode.ComfyManagedResidencyError) as excinfo:
        store_fsdp.ensure(worker, "model.safetensors", {}, STACK)
    assert worker.store.calls == []
    tag = _tag(excinfo.value)
    assert tag is not None
    assert tag.refusal_class is RefusalClass.PHYSICS
    assert tag.guard is None
    assert tag.waivable is False
    text = str(excinfo.value)
    assert "docs/TROUBLESHOOTING.md #62" in text
    assert "instead" in text
    assert "nothing was quarantined" in text
    assert refusal.is_waivable(text) is False


def test_the_driver_answers_comfy_managed_fsdp_before_the_footprint_card(monkeypatch):
    """Order, not price: the policy refusal owns this cell.

    Comfy-managed residency and FSDP have never been run together, so no budget
    can change the answer. The worker's copy of that refusal needs a load_model
    call, and the driver's footprint card runs above ``ensure_live``, so on the
    61.7 GiB bf16 H3 artifact the card answered first (2026-09-02) and typed
    itself class K on an unreadable quarantine, offering slab residency on a
    load where comfy-managed residency owns placement and slab is not a lever.
    The 19.5 GiB fp8 and 31.7 GiB int8 rows of the same family cleared the
    budget and answered class P, so only the artifact big enough to trip the
    card lost its policy answer.
    """
    from types import SimpleNamespace

    from dgx_monarch.driver_footprint import DriverFootprintCapacityError
    from dgx_monarch.nodes import loader_preflight
    from dgx_monarch.nodes.loaders import DGXMonarchUNETLoader

    charged = []

    def explode(*args, **kwargs):
        charged.append(args)
        raise DriverFootprintCapacityError("the footprint card answered first")

    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint", explode)
    mesh = SimpleNamespace(topology_preset="uly2+fsdp", handle=None,
                           worker_args={"comfy_managed": True})
    with pytest.raises(residency_mode.ComfyManagedResidencyError) as excinfo:
        DGXMonarchUNETLoader().load(mesh, "h3_fl2va_bf16.safetensors")
    assert charged == [], "the capacity card priced a load policy already refused"
    text = str(excinfo.value)
    tag = _tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert tag.waivable is False
    # Byte for byte the worker's sentence: one home, so the two cannot drift.
    assert residency_mode.COMFY_MANAGED_FSDP_REFUSAL in text
    assert "docs/TROUBLESHOOTING.md #62" in text
    assert "driver-side footprint preflight" not in text


def test_the_driver_policy_refusal_leaves_every_other_cell_alone(monkeypatch):
    """It adds a refusal, it never removes one: an fsdp preset without the
    widget, and the widget without an fsdp preset, both fall through to the
    capacity estimate that owns them."""
    from types import SimpleNamespace

    from dgx_monarch.nodes import loader_preflight

    for preset, args in (("uly2+fsdp", {}), ("uly2", {"comfy_managed": True}),
                         ("auto", {"comfy_managed": True}),
                         ("uly2+fsdp", {"comfy_managed": "on"})):
        loader_preflight.preflight_comfy_managed_topology(
            SimpleNamespace(topology_preset=preset, handle=None, worker_args=args))


def test_the_lora_refusal_counts_the_stack_and_names_the_widget(rung_on):
    worker = _worker()
    with pytest.raises(residency_mode.ComfyManagedResidencyError) as excinfo:
        store_fsdp.ensure(worker, "model.safetensors", {}, [*STACK, *STACK])
    text = str(excinfo.value)
    assert "2" in text
    assert "comfy_managed" in text


def test_the_regression_that_made_the_yield_move(rung_on):
    """Load LoRA-less, then hand the same base the first LoRA.

    This is the sequence that reached ``_build_active`` on a
    ModelPatcherDynamic with no refusal: same unet, same options, so the store's
    base_key matches and the baked hot-swap arm fires. The funnel must refuse
    the second call even though the first was fine.
    """
    worker = _worker()
    store_fsdp.ensure(worker, "model.safetensors", {}, [])
    assert len(worker.store.calls) == 1
    with pytest.raises(residency_mode.ComfyManagedResidencyError):
        store_fsdp.ensure(worker, "model.safetensors", {}, STACK)
    assert len(worker.store.calls) == 1


def test_the_uncond_slot_is_guarded_too(rung_on):
    worker = _worker()
    with pytest.raises(residency_mode.ComfyManagedResidencyError):
        store_fsdp.ensure(worker, "model.safetensors", {}, STACK, slot="uncond")
    assert worker.store.calls == []


def test_the_lora_stack_reaches_the_guard_as_a_keyword_too(rung_on):
    worker = _worker()
    with pytest.raises(residency_mode.ComfyManagedResidencyError):
        store_fsdp.ensure(
            worker, "model.safetensors", options={}, lora_stack=STACK)
    assert worker.store.calls == []


def test_fsdp_refuses_at_the_funnel(rung_on):
    """Not the silent gate slab uses.

    slab_weights gates off silently under FSDP because its default is `auto`
    and a silent gate on a default is not a surprise. comfy_managed has no
    `auto` and is on only when the operator asks for it, so it refuses.
    """
    worker = _worker(topology={"fsdp": 2})
    with pytest.raises(residency_mode.ComfyManagedResidencyError) as excinfo:
        store_fsdp.ensure(worker, "model.safetensors", {}, [])
    assert worker.store.calls == []
    text = str(excinfo.value)
    assert "FSDP" in text
    assert "docs/TROUBLESHOOTING.md #62" in text
    assert _tag(text).refusal_class is RefusalClass.PHYSICS


def test_the_lora_cause_is_named_before_the_fsdp_cause(rung_on):
    """The sentinel-trap doctrine outranks the FSDP reason."""
    worker = _worker(topology={"fsdp": 2})
    with pytest.raises(residency_mode.ComfyManagedResidencyError) as excinfo:
        store_fsdp.ensure(worker, "model.safetensors", {}, STACK)
    text = str(excinfo.value)
    assert "LoRA" in text or "lora" in text
    assert "FSDP is active for this render" not in text


def test_the_funnel_is_inert_when_the_rung_is_off():
    worker = _worker(topology={"fsdp": 2})
    store_fsdp.ensure(worker, "model.safetensors", {}, STACK)
    args, kwargs = worker.store.calls[0]
    assert args[0] == "model.safetensors"
    assert kwargs["fsdp_launch"] is True


def test_the_residency_ladder_holds_no_second_belt():
    """The yields live only in the funnel, not also in the ladder.

    A duplicate in ``resolve`` would double every ledger row for a refusal the
    operator sees once, and would put the authoritative text in the arm that a
    baked hot-swap never reaches. A second check there needs its own ledger
    rows and its own reason in the same change.
    """
    tree = _tree("actor/store_residency.py")
    raised = {
        node.exc.func.attr if isinstance(node.exc.func, ast.Attribute)
        else getattr(node.exc.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
    }
    assert "ComfyManagedResidencyError" not in raised, (
        "the LoRA and FSDP yields belong to actor/store_fsdp.ensure, the one "
        "funnel every load path shares. store_residency.resolve is skipped "
        "entirely by the baked hot-swap arm the rung forces on."
    )


def test_build_active_is_unreachable_under_the_rung_with_a_stack(rung_on):
    """The store's clone-and-repatch path is where comfy's manager would land."""
    calls = []

    class _Store(_RecordingStore):
        def ensure(self, *args, **kwargs):  # pragma: no cover - must not run
            calls.append(args)
            raise AssertionError("_build_active would have run here")

    with pytest.raises(residency_mode.ComfyManagedResidencyError):
        store_fsdp.ensure(_worker(store=_Store()), "m.safetensors", {}, STACK)
    assert calls == []
