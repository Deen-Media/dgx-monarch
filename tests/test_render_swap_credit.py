"""The render-site checks count the model a pending load replaces (swap_credit.py).

Under ``auto`` the UNET loader defers its load to the first render, so the
render-site checks price it again: the driver footprint, the MiniMax H3
activation price and the reference and pose price each charge the pending DiT
against the driver host's MemAvailable. With an old model in the slot they
priced the new one beside it, as the loader did. So a loader that admits a swap
on the credit must be followed by a render site that counts the same drop, or
the refusal only moves from the loader to the sampler.

``driver_unload.check_with_unload`` counts the credit first, then unloads what
ComfyUI holds, then asks the actors to release their caches. These tests hold
that order, the sentence a refusal that still stands carries, and that the
credit exists only inside the rescue.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import test_driver_unload as rig
from dgx_monarch import capacity_agreement, driver_unload, mesh_safety, swap_credit
from dgx_monarch.nodes import loader_relief, render_preflight
from test_capacity_old_model import _answer
from test_driver_unload import DIT_BYTES, UNET, _comfy, _h3_latent, _h3_need, _h3_request, _h3_rig
from test_loader_swap_credit import _answers_by_slot, _held, _other

# The driver-unload rig: a fresh progress tracker, a UMA box, the box's reading and the journal.
_isolated, box, journal = rig._isolated, rig.box, rig.journal

_GIB = 2 ** 30
OLD = "flux1-dev.safetensors"
CREDIT = swap_credit.SwapCredit(22 * _GIB, OLD, "the head")


def _pending(box, need: int, seen: list | None = None):
    """A check that charges a pending load: it reads the credited figure."""
    def check():
        available = swap_credit.credited(box.mem)
        if seen is not None:
            seen.append(available)
        if available < need:
            raise mesh_safety.StockLoadCapacityError(
                f"[dgxm:C guard=activation_footprint_preflight waivable=0] probe: "
                f"{need / _GIB:.1f} GiB needed, {available / _GIB:.1f} GiB MemAvailable.")
    return check


def _credited(monkeypatch, credit=CREDIT) -> list:
    asked: list = []
    monkeypatch.setattr(loader_relief, "render_credit",
                        lambda model, request: asked.append((model, request)) or credit)
    return asked


def test_a_pending_load_fits_once_the_drop_is_counted_and_nothing_is_unloaded(monkeypatch, box, journal):
    box.mem = 40 * _GIB
    calls = _comfy(monkeypatch, box)
    asked = _credited(monkeypatch)
    seen: list = []
    driver_unload.check_with_unload([_pending(box, 55 * _GIB, seen)], subject="ltx.safetensors",
                                    swap=("model", "request"))
    assert seen == [40 * _GIB, 62 * _GIB] and asked == [("model", "request")]
    assert calls.unload == 0, "the credit alone made room: ComfyUI keeps its models"
    lines = [record.getMessage() for record in journal]
    assert lines == ["render preflight: ltx.safetensors fits once the Worker on the head drops "
                     "flux1-dev.safetensors before this load, which frees at least 22.0 GiB"]
    assert journal[0].levelno == logging.INFO
    assert swap_credit.credited(box.mem) == box.mem, "the credit ended with the rescue"


def test_a_check_that_passes_asks_no_worker(monkeypatch, box):
    box.mem = 60 * _GIB
    asked = _credited(monkeypatch)
    driver_unload.check_with_unload([_pending(box, 55 * _GIB)], swap=("model", "request"))
    assert asked == []


def test_the_unload_still_runs_when_the_credit_falls_short_and_both_count(monkeypatch, box, journal):
    """40 GiB read, 22 GiB to be dropped, 22.1 GiB of ComfyUI's models: 80 GiB needs all of it."""
    box.mem = 40 * _GIB
    calls = _comfy(monkeypatch, box)
    _credited(monkeypatch)
    seen: list = []
    driver_unload.check_with_unload([_pending(box, 80 * _GIB, seen)], subject="ltx.safetensors",
                                    swap=("model", "request"))
    assert calls.unload == 1
    assert seen[-1] == box.mem + 22 * _GIB and seen[-1] >= 80 * _GIB
    assert swap_credit.credited(box.mem) == box.mem


def test_a_render_still_short_says_what_was_counted(monkeypatch, box):
    box.mem = 40 * _GIB
    _comfy(monkeypatch, box, models=())
    _credited(monkeypatch)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        driver_unload.check_with_unload([_pending(box, 90 * _GIB)], swap=("model", "request"))
    text = str(raised.value)
    assert "90.0 GiB needed, 62.0 GiB MemAvailable." in text
    # The price prints the credited figure, so the sentence gives the box's own reading too.
    assert text.endswith("The figures above already count that the Worker on the head drops "
                         "flux1-dev.safetensors before this load, which frees at least 22.0 GiB. "
                         "The box itself read 40.0 GiB of MemAvailable.")
    assert swap_credit.credited(box.mem) == box.mem


def test_no_credit_leaves_the_rescue_as_it_was(monkeypatch, box):
    box.mem = 40 * _GIB
    _comfy(monkeypatch, box, models=())
    _credited(monkeypatch, swap_credit.NONE)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        driver_unload.check_with_unload([_pending(box, 55 * _GIB)], swap=("model", "request"))
    assert "already count" not in str(raised.value)
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        driver_unload.check_with_unload([_pending(box, 55 * _GIB)])


def test_the_unload_switch_leaves_the_credit_on_and_unloads_nothing(monkeypatch, box, journal):
    """The credit is a price, not a rescue. A deferred swap the loader admits on it must not
    refuse at the sampler because the unload is switched off."""
    box.mem = 40 * _GIB
    calls = _comfy(monkeypatch, box)
    asked = _credited(monkeypatch)
    monkeypatch.setenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, "1")
    driver_unload.check_with_unload([_pending(box, 55 * _GIB)], swap=("model", "request"))
    assert asked == [("model", "request")] and calls.unload == 0
    # Still short with the credit: the switch keeps ComfyUI's models, and the refusal says what counted.
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        driver_unload.check_with_unload([_pending(box, 70 * _GIB)], swap=("model", "request"))
    assert calls.unload == 0 and "already count that the Worker on the head drops" in str(raised.value)


def test_the_credit_has_its_own_switch_at_the_render_site(monkeypatch, tmp_path, box):
    """``DGXM_DISABLE_SWAP_CREDIT=1``: no Worker is asked, and the H3 render that the drop
    would let through refuses as it did before the credit."""
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    _live(rig_)
    box.mem = _h3_need(weights=DIT_BYTES) - 15 * _GIB
    _comfy(monkeypatch, box, models=())
    asked: list = []
    monkeypatch.setattr(capacity_agreement, "collect", lambda *_a, **_k:
                        asked.append(1) or [_answer(0, _held(OLD, 22.2), box="this box")])
    monkeypatch.setenv(swap_credit.DISABLE_ENV, "1")
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        render_preflight.activation_footprint_preflight_for_request(rig_.model, _request(), _h3_latent())
    assert asked == [] and "already count" not in str(raised.value)


def test_a_fault_gives_no_credit(monkeypatch, box):
    box.mem = 40 * _GIB
    _comfy(monkeypatch, box, models=())

    def broken(model, request):
        raise RuntimeError("no fleet")

    monkeypatch.setattr(loader_relief, "render_credit", broken)
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        driver_unload.check_with_unload([_pending(box, 55 * _GIB)], swap=("model", "request"))


def test_any_other_error_ends_the_credit_too(monkeypatch, box):
    box.mem = 40 * _GIB
    _credited(monkeypatch)
    calls = {"n": 0}

    def step():
        calls["n"] += 1
        if calls["n"] == 1:
            raise mesh_safety.StockLoadCapacityError("[dgxm:C guard=x waivable=0] short")
        raise ValueError("a bug in a check")

    with pytest.raises(ValueError):
        driver_unload.check_with_unload([step], swap=("model", "request"))
    assert swap_credit.credited(box.mem) == box.mem


# The real H3 price under auto, where the DiT load waits for this render.

def _live(rig_) -> None:
    """Make the rig's local fleet one the driver may ask: set up, with Workers."""
    handle = rig_.model.mesh.handle
    handle.setup_key, handle.workers, handle.defunct = ("ready",), object(), False
    handle.gpus_per_host = 1
    handle.config = SimpleNamespace(hosts=(), worker_args={})


def _request(loras=(), uncond: dict | None = None) -> dict:
    request = {**_h3_request(),
               "model": {"unet_name": UNET, "options": {}, "loras": [dict(entry) for entry in loras]}}
    return {**request, "uncond_model": uncond} if uncond is not None else request


def test_an_h3_render_whose_load_replaces_an_old_model_proceeds(monkeypatch, tmp_path, box, journal):
    """The H3 price charges the pending 19.5 GiB DiT. With 15 GiB missing and a 22.2 GiB
    stock model in the slot, the Worker's drop makes the room, and ComfyUI keeps its models."""
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    _live(rig_)
    box.mem = _h3_need(weights=DIT_BYTES) - 15 * _GIB
    calls = _comfy(monkeypatch, box)
    held = _held(OLD, 22.2)
    requests: list = []
    monkeypatch.setattr(capacity_agreement, "collect", lambda handle, request, world, *, deadline_s:
                        requests.append(request) or [_answer(0, held, box="this box")])
    # The question carries the render's own LoRA stack: without it the Worker would read a
    # reuse of the held base and stack as a swap, and credit a drop that never happens.
    stack = ({"name": "style.safetensors", "strength": 0.8},)
    render_preflight.activation_footprint_preflight_for_request(rig_.model, _request(stack), _h3_latent())
    assert calls.unload == 0 and rig_.rpc == []
    assert requests[0]["specs"] == [{"unet_name": UNET, "options": {}, "loras": list(stack), "slot": "cond"}]
    assert len(requests) == 1, "a render with one DiT asks once"
    line = next(record.getMessage() for record in journal)
    assert UNET in line and "drops flux1-dev.safetensors before this load" in line


@pytest.mark.parametrize("change", [{"would_load": "reuse"}, {"word": "slab"}])
def test_an_h3_render_with_nothing_to_drop_is_priced_as_before(monkeypatch, tmp_path, box, change):
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    _live(rig_)
    box.mem = _h3_need(weights=DIT_BYTES) - 15 * _GIB
    _comfy(monkeypatch, box, models=())
    held = _held(OLD, 22.2, **change)
    monkeypatch.setattr(capacity_agreement, "collect",
                        lambda *_a, **_k: [_answer(0, held, box="this box")])
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        render_preflight.activation_footprint_preflight_for_request(rig_.model, _request(), _h3_latent())
    assert "already count" not in str(raised.value)


def test_a_price_that_charges_no_load_takes_no_credit_at_the_render_site(monkeypatch, tmp_path, box):
    """An explicit preset: the loader placed the DiT, so the H3 price charges activations only.
    If the Worker still answers "swap" (a load the gate deferred, say), the drop it names is
    not room for activations: the box 2.3 GiB short refuses as it did before the credit."""
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="uly2")
    _live(rig_)
    box.mem = _h3_need(weights=0) - int(2.3 * _GIB)
    _comfy(monkeypatch, box, models=())
    held = _held(OLD, 22.2)
    monkeypatch.setattr(capacity_agreement, "collect",
                        lambda *_a, **_k: [_answer(0, held, box="this box")])
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        render_preflight.activation_footprint_preflight_for_request(rig_.model, _request(), _h3_latent())
    assert f"{box.mem / _GIB:.1f} GiB" in str(raised.value), "the figure is the box's own reading"
    # The rescue found a credit, and no price that charges a load took it: nothing says it did.
    assert "already count" not in str(raised.value)


@pytest.mark.parametrize(("uncond_gib", "outcome"), [
    (27.0, "fits"),         # the uncond slot grows by 5 GiB: 17.2 of the drop still covers 15
    (32.0, "short"),        # by 10: 12.2 is counted, and the refusal says where the rest went
    (50.0, "no credit"),    # by 28, more than the drop frees: the price as it was
])
def test_a_dual_model_render_sets_the_uncond_loads_growth_aside(
        monkeypatch, tmp_path, box, journal, uncond_gib, outcome):
    """The H3 price charges the cond DiT alone. 15 GiB missing, a 22.2 GiB stock model in the
    cond slot, and a second DiT that replaces a 22 GiB model in the uncond slot."""
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    _live(rig_)
    box.mem = _h3_need(weights=DIT_BYTES) - 15 * _GIB
    _comfy(monkeypatch, box, models=())
    asked = _answers_by_slot(monkeypatch, cond=_held(OLD, 22.2), uncond=_other(uncond_gib, held_gib=22.0))
    uncond = {"unet_name": "second.safetensors", "options": {}, "loras": []}

    def render():
        render_preflight.activation_footprint_preflight_for_request(
            rig_.model, _request(uncond=uncond), _h3_latent())

    if outcome == "fits":
        render()
        assert "of which 5.0 GiB is set aside for the model this graph loads into the other slot" in next(
            record.getMessage() for record in journal)
    else:
        with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
            render()
        assert ("of which 10.0 GiB is set aside" in str(raised.value)) is (outcome == "short")
        assert ("already count" in str(raised.value)) is (outcome == "short")
    assert asked[1]["specs"] == [{"unet_name": UNET, "options": {}, "loras": [], "slot": "cond"},
                                 {**uncond, "slot": "uncond"}]


def test_a_dual_model_render_whose_second_spec_cannot_be_read_gets_no_credit(monkeypatch, tmp_path, box):
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    _live(rig_)
    box.mem = _h3_need(weights=DIT_BYTES) - 15 * _GIB
    _comfy(monkeypatch, box, models=())
    asked = _answers_by_slot(monkeypatch, cond=_held(OLD, 22.2))
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        render_preflight.activation_footprint_preflight_for_request(
            rig_.model, _request(uncond={"options": {}}), _h3_latent())
    assert len(asked) == 1 and "already count" not in str(raised.value)


def test_a_true_shortage_at_the_render_site_still_refuses(monkeypatch, tmp_path, box):
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    _live(rig_)
    box.mem = _h3_need(weights=DIT_BYTES) - 40 * _GIB
    _comfy(monkeypatch, box, models=())
    held = _held(OLD, 22.2)
    monkeypatch.setattr(capacity_agreement, "collect",
                        lambda *_a, **_k: [_answer(0, held, box="this box")])
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        render_preflight.activation_footprint_preflight_for_request(rig_.model, _request(), _h3_latent())
    assert "already count that the Worker on this box drops flux1-dev.safetensors" in str(raised.value)


# The other two prices that read the credit: each takes it only beside its own weight charge.

@pytest.mark.parametrize("resident, counted", [(False, True), (True, False)])
def test_the_driver_footprint_counts_the_drop_only_for_a_load_it_charges(
        monkeypatch, box, resident, counted):
    """Weights the driver reads as loaded are charged nothing, so their price has no drop to count."""
    from dgx_monarch import driver_footprint

    box.mem = 40 * _GIB
    seen: list = []
    monkeypatch.setattr(driver_footprint, "estimate_driver_footprint", lambda **given:
                        seen.append(given) or SimpleNamespace(weight_bytes=0))
    monkeypatch.setattr(driver_footprint, "driver_footprint_preflight", lambda *_a, **_k: None)
    monkeypatch.setattr(driver_footprint, "ref_pixel_volume_from_latents", lambda *_a: (0, 0, 0))
    model = SimpleNamespace(unet_name=UNET, options=None, mesh=SimpleNamespace(
        topology_preset="auto", worker_args={}, handle=None))
    with swap_credit.applying(CREDIT) as applied:
        render_preflight._driver_footprint_preflight_for_request(
            model, {}, "minimax_h3", "", weights_loaded=resident)
    assert [given["weights_resident"] for given in seen] == [resident]
    assert seen[0]["mem_available"] == box.mem + (CREDIT.bytes if counted else 0)
    assert applied.used is counted


@pytest.mark.parametrize("resident, counted", [(False, True), (True, False)])
def test_the_reference_and_pose_price_counts_the_drop_only_for_a_load_it_charges(
        tmp_path, box, resident, counted):
    """One GiB short of the activations, a 20 GiB DiT: pending, the price needs 21 GiB more than
    the box reads and the 22 GiB drop covers it; loaded, the drop is no room for activations."""
    shape = (1, 16, 21, 60, 104)
    tokens = mesh_safety.ref_pose_token_counts(shape, (), None)[0]
    need = (tokens * mesh_safety.REF_POSE_TOKEN_FAMILIES["wan_scail"]
            * mesh_safety._ACTIVATION_BF16_BYTES * mesh_safety._ACTIVATION_BLOCK_FACTOR)
    box.mem = need - 1 * _GIB
    path = tmp_path / "scail.safetensors"
    with open(path, "wb") as handle:
        handle.truncate(20 * _GIB)          # sparse: a 20 GiB DiT, no disk, never read

    def price():
        mesh_safety.activation_footprint_preflight(
            "wan_scail", str(path), "scail", video_shape=shape, weights_resident=resident)

    with swap_credit.applying(CREDIT) as applied:
        if counted:
            price()
        else:
            with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
                price()
            assert f"exceeds {box.mem / _GIB:.1f} GiB available" in str(raised.value)
    assert applied.used is counted
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        price()  # outside the credit both cases refuse


def test_the_reference_and_pose_price_still_refuses_a_true_shortage_inside_the_credit(tmp_path, box):
    """A 20 GiB pending DiT, 1 GiB read and 10 GiB to drop: 11 GiB does not hold it."""
    box.mem = 1 * _GIB
    path = tmp_path / "scail.safetensors"
    with open(path, "wb") as handle:
        handle.truncate(20 * _GIB)
    small = swap_credit.SwapCredit(10 * _GIB, OLD, "the head")
    with swap_credit.applying(small), pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        mesh_safety.activation_footprint_preflight(
            "wan_scail", str(path), "scail", video_shape=(1, 16, 1, 2, 2))
    assert "exceeds 11.0 GiB available" in str(raised.value)


def test_the_reference_and_pose_price_takes_no_credit_for_weights_it_cannot_size(tmp_path, box):
    """No file to size is no weight charge, so the price has no drop to count."""
    shape = (1, 16, 21, 60, 104)
    box.mem = 1 * _GIB
    with swap_credit.applying(CREDIT) as applied, pytest.raises(mesh_safety.StockLoadCapacityError):
        mesh_safety.activation_footprint_preflight(
            "wan_scail", str(tmp_path / "missing.safetensors"), "scail", video_shape=shape)
    assert not applied.used


def test_the_driver_footprint_still_refuses_a_true_shortage_inside_the_credit(monkeypatch, tmp_path, box):
    """The real estimator: a 20 GiB pending DiT with 0.5 GiB read and 22 GiB to drop is short of
    the weights, their host copy and the reserve."""
    from dgx_monarch import driver_footprint

    box.mem = int(0.5 * _GIB)
    path = tmp_path / "new.safetensors"
    with open(path, "wb") as handle:
        handle.truncate(20 * _GIB)
    monkeypatch.setattr(driver_footprint, "rank0_co_resident", lambda mesh: True)
    monkeypatch.setattr(driver_footprint, "ref_pixel_volume_from_latents", lambda *_a: (0, 0, 0))
    model = SimpleNamespace(unet_name=UNET, options=None, mesh=SimpleNamespace(
        topology_preset="auto", worker_args={}, handle=None))
    with swap_credit.applying(CREDIT) as applied, pytest.raises(
            driver_footprint.DriverFootprintCapacityError):
        render_preflight._driver_footprint_preflight_for_request(
            model, {}, "minimax_h3", str(path), weights_loaded=False)
    assert applied.used
