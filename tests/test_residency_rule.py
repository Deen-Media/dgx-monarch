"""One residency rule for the render-site checks: what was placed, and no unload since.

The render-site checks credit the DiT as already in MemAvailable from the
session's memos: a finished render, a pipelined submission, a successful load,
and for two checks a pending load the driver footprint cleared. Every door
that unloads every rank (an identity ceremony's clean reload, the no-render
path, the slab proof, the cross-mode reference, the FSDP cleanup, a rescue's
dropped slot, a hard Clear VRAM) marks the load as still to come
(``pending_load_relief.note_unloaded``), but the memos still named the
checkpoint, so after a hard Clear VRAM the gate's capacity stop, the #52 and
H3 checks and the reference and pose check (#47) priced a DiT the Workers had
dropped as resident. And the #47 check never credited an explicit preset's
eager load, which the #52 and H3 checks do, so it charged that file twice.

The unload now drains the memos it makes untrue, and every site reads one
predicate (``pending_load_relief.dit_resident``). Memos stamped after the
unload still count: a retry after a failed first render, a pipeline's next
push, a load the no-render path made. Requests come from the real sampler
node, the unload from the real Clear VRAM node.

CPU only, on the driver-unload rig of ``test_driver_unload``.
"""
from __future__ import annotations

import json
import sys
import types

import pytest
import torch

import test_driver_unload as rig
from dgx_monarch import mesh_safety
from dgx_monarch.nodes import (
    common,
    gate_capacity_stop,
    ops,
    pending_load_relief,
    recycle_drain,
    render_preflight,
    samplers,
)
from dgx_monarch.nodes.common import ModelSpec
from test_driver_unload import DIT_BYTES, UNET, _comfy, _h3_latent, _h3_need, _h3_request, _h3_rig

_isolated, box, journal = rig._isolated, rig.box, rig.journal

_GIB = 2 ** 30


class _Checked(Exception):
    """Raised where run_render leaves the render-site checks behind it."""


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    pending_load_relief.reset()
    monkeypatch.setattr(common, "_claim_adoption_context",
                        lambda: (_ for _ in ()).throw(_Checked()))
    # The relief refuses nothing; these tests read the checks that do.
    monkeypatch.setattr(pending_load_relief, "relieve", lambda *_a, **_k: None)
    yield
    pending_load_relief.reset()


def _cond() -> list:
    """The text rows ``_h3_need`` prices (test_driver_unload._h3_request)."""
    return [[torch.zeros(1, _h3_request()["positive"][0][0].shape[1], 8), {}]]


def _render(model) -> None:
    """One queue press through the KSampler node, up to the end of the checks."""
    with pytest.raises(_Checked):
        samplers.DGXMonarchKSampler().sample(
            model, 1, 8, 1.0, "euler", "simple", _cond(), _cond(), _h3_latent())


def _sampler_request(monkeypatch, model) -> dict:
    seen: list[dict] = []
    monkeypatch.setattr(samplers, "run_render", lambda _model, request, *_a, **_k: (
        seen.append(request), (_ for _ in ()).throw(_Checked()))[1])
    _render(model)
    monkeypatch.setattr(samplers, "run_render", common.run_render)
    return seen[0]


def _hard_clear(monkeypatch, model) -> None:
    """Clear VRAM (DGX Monarch) at hard: every Worker unloads its models."""
    handle = model.mesh.handle
    monkeypatch.setattr(ops, "ensure_live", lambda _handle: handle)
    handle.call_all = lambda method, *_a, **_k: [{"rank": 0, "method": method}]
    response = ops.DGXMonarchClearVRAM().clear(model.mesh, "hard", include_driver=False)
    assert json.loads(response["result"][0])["workers"] == [{"rank": 0, "method": "clear_vram"}]


@pytest.fixture
def h3(monkeypatch, tmp_path, box):
    """H3 on the pair under auto, the DiT pending unless a memo says otherwise.
    The box reads room for the activations alone, 4 GiB short of the DiT beside
    them, and ComfyUI holds nothing for the rescue to free."""
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="auto")
    box.mem = _h3_need(weights=0) + 4 * _GIB
    assert box.mem < _h3_need(weights=DIT_BYTES)
    _comfy(monkeypatch, box, models=())
    return ModelSpec(mesh=rig_.model.mesh, unet_name=UNET)


# --- a1#1, altitude#1: an all-rank unload ends the memos' credit ---------------------------


def test_the_render_site_charges_a_dit_a_hard_clear_dropped(monkeypatch, h3):
    render_preflight.note_successful_render(UNET)  # a render of it succeeded
    _render(h3)                                    # credited: activations alone fit
    _hard_clear(monkeypatch, h3)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as refused:
        _render(h3)
    assert "Charged: DiT weights 19.5 GiB (not yet loaded)" in str(refused.value)


def test_the_gates_capacity_stop_charges_a_dit_a_hard_clear_dropped(monkeypatch, h3):
    request = _sampler_request(monkeypatch, h3)
    render_preflight.note_successful_render(UNET)
    gate_capacity_stop.ceremony_activation_preflight(h3, request, _h3_latent())
    _hard_clear(monkeypatch, h3)
    assert not gate_capacity_stop.dit_already_loaded(h3)
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        gate_capacity_stop.ceremony_activation_preflight(h3, request, _h3_latent())


@pytest.mark.parametrize("writer", [render_preflight.note_successful_render,
                                    render_preflight.note_submitted_render,
                                    pending_load_relief.note_placed])
def test_every_unload_door_drains_each_memo(h3, writer):
    """The unload doors all mark through ``note_unloaded``; a test in
    test_pending_load_relief holds that each one does. The #52 memo has its
    own test below, on a family whose check writes it."""
    writer(UNET)
    _render(h3)
    pending_load_relief.note_unloaded()
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        _render(h3)


def test_a_stale_word_never_beats_the_unload_mark():
    model = types.SimpleNamespace(unet_name=UNET, mesh=types.SimpleNamespace(topology_preset="auto"))
    pending_load_relief.note_unloaded()
    assert pending_load_relief.load_pending(model, True)
    pending_load_relief.note_placed(UNET)
    assert not pending_load_relief.load_pending(model, True)


# --- what still counts: memos stamped after the unload --------------------------------------


def test_a_pipelines_next_push_after_a_reset_keeps_its_credit(h3):
    recycle_drain.drop_residency_memos()
    render_preflight.note_submitted_render(UNET)  # nodes/pipeline.py, at the first push
    _render(h3)


LTX = "ltx-2.3-22b-distilled_transformer_only_fp8_scaled.safetensors"
LTX_BYTES = 22 * _GIB


@pytest.fixture
def ltx(monkeypatch, tmp_path, box):
    """LTX 2.3 under ``auto``: the driver footprint (#52) charges its pending
    22 GiB DiT and, once that clears, writes its own memo. ComfyUI holds
    nothing for the rescue to free."""
    path = tmp_path / LTX
    path.write_bytes(b"x")
    fake = types.ModuleType("folder_paths")
    fake.get_full_path = lambda _kind, _name: str(path)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint", lambda _path: ("ltx", "fp8"))
    monkeypatch.setattr("dgx_monarch.driver_footprint.file_size_bytes", lambda _path: LTX_BYTES)
    monkeypatch.setattr("dgx_monarch.render_memory_price.preflight_render_memory",
                        lambda *_a, **_k: None)
    for memo in ("_LAST_RENDERED_UNET", "_LAST_SUBMITTED_UNET", "_LAST_DRIVER_CHARGED_UNET"):
        monkeypatch.setattr(render_preflight, memo, None)
    _comfy(monkeypatch, box, models=())
    handle = rig._leased(owns_hosts=False, world=2, call_all=lambda *a, **k: [],
                         effective_worker_args=lambda requested=None: dict(requested or {}))
    mesh = types.SimpleNamespace(topology_preset="auto", worker_args={}, world=2, handle=handle)
    return ModelSpec(mesh=mesh, unet_name=LTX)


def _ltx_render(model) -> None:
    latent = {"samples": torch.zeros(1, 128, 4, 8, 8)}
    with pytest.raises(_Checked):
        samplers.DGXMonarchKSampler().sample(
            model, 1, 8, 1.0, "euler", "simple", [[torch.zeros(1, 8, 8), {}]],
            [[torch.zeros(1, 8, 8), {}]], latent)


PENDING_FITS = 30 * _GIB     # 22 GiB to place and the 5 GiB reserve
PLACED_READS = 8 * _GIB      # the same box once the DiT sits in it


def test_the_footprint_check_writes_its_memo_and_an_unload_drains_it(ltx, box):
    box.mem = PENDING_FITS
    _ltx_render(ltx)
    assert render_preflight._LAST_DRIVER_CHARGED_UNET == LTX
    box.mem = PLACED_READS
    _ltx_render(ltx)                         # a retry: the cleared load is credited
    pending_load_relief.note_unloaded()
    with pytest.raises(Exception, match="driver-side footprint preflight"):
        _ltx_render(ltx)


def test_a_retry_after_a_failed_first_render_after_a_reset_keeps_its_credit(ltx, box):
    """A reset drains everything; the first render's check clears the pending
    load and writes its memo, the render loads and fails, and the retry does
    not charge that file again."""
    recycle_drain.drop_residency_memos()
    box.mem = PENDING_FITS
    _ltx_render(ltx)
    box.mem = PLACED_READS
    _ltx_render(ltx)


def test_a_load_the_no_render_path_made_after_its_unload_counts(ltx, box):
    """The user's check wrote its memo, the no-render path unloaded every rank
    and loaded the file, and the render failed: the retry's DiT is loaded."""
    box.mem = PENDING_FITS
    _ltx_render(ltx)
    pending_load_relief.note_unloaded()      # nodes/gate_nothing_to_gate.py, its unload
    pending_load_relief.note_placed(LTX)     # and its load
    box.mem = PLACED_READS
    _ltx_render(ltx)


def test_a_ceremony_leg_after_its_unload_is_priced_on_the_reading_that_holds_the_freed_dit(
        ltx, box):
    """Under ``auto`` a warm model passes the user's check credited; the
    ceremony unloads every rank and its first leg's check charges the DiT again,
    against a reading that holds the 22 GiB the Workers freed."""
    render_preflight.note_successful_render(LTX)
    box.mem = PLACED_READS
    _ltx_render(ltx)
    pending_load_relief.note_unloaded()      # nodes/gate_ceremony.py, before leg B
    box.mem = PLACED_READS + LTX_BYTES
    _ltx_render(ltx)


# --- altitude#2: the reference and pose check credits an explicit preset's eager load ------


SCAIL = "wan_scail_bf16.safetensors"
SCAIL_GIB = 30
SHAPE = (1, 16, 21, 60, 104)


@pytest.fixture
def scail(monkeypatch, tmp_path, box):
    """Explicit uly2, Wan SCAIL BF16, a sparse 30 GiB file. The box reads the
    clip's activations and 5 GiB more: room for the render, not for the file
    again beside it."""
    path = tmp_path / SCAIL
    with open(path, "wb") as handle:
        handle.truncate(SCAIL_GIB * _GIB)
    fake = types.ModuleType("folder_paths")
    fake.get_full_path = lambda _kind, _name: str(path)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint", lambda _path: ("wan_scail", "bf16"))
    monkeypatch.setattr(render_preflight.latent_scale, "rendered_latent_shape",
                        lambda *_a: SHAPE)
    monkeypatch.setattr("dgx_monarch.render_memory_price.preflight_render_memory",
                        lambda *_a, **_k: None)
    for memo in ("_LAST_RENDERED_UNET", "_LAST_SUBMITTED_UNET", "_LAST_DRIVER_CHARGED_UNET"):
        monkeypatch.setattr(render_preflight, memo, None)
    tokens = mesh_safety.ref_pose_token_counts(SHAPE, (), None)[0]
    activations = (tokens * mesh_safety.REF_POSE_TOKEN_FAMILIES["wan_scail"]
                   * mesh_safety._ACTIVATION_BF16_BYTES * mesh_safety._ACTIVATION_BLOCK_FACTOR)
    box.mem = int(activations) + 5 * _GIB
    _comfy(monkeypatch, box, models=())
    handle = rig._leased(owns_hosts=False, world=2, call_all=lambda *a, **k: [],
                         effective_worker_args=lambda requested=None: dict(requested or {}))
    mesh = types.SimpleNamespace(topology_preset="uly2", worker_args={}, world=2, handle=handle)
    return ModelSpec(mesh=mesh, unet_name=SCAIL)


def _scail_render(model) -> None:
    latent = {"samples": torch.zeros(1, 16, 21, 60, 104)}
    with pytest.raises(_Checked):
        samplers.DGXMonarchKSampler().sample(
            model, 1, 8, 1.0, "euler", "simple", [[torch.zeros(1, 8, 8), {}]],
            [[torch.zeros(1, 8, 8), {}]], latent)


def test_the_reference_and_pose_check_credits_the_loaders_eager_placement(scail):
    pending_load_relief.note_placed(SCAIL)  # nodes/loaders.py, once its eager load returns
    _scail_render(scail)


def test_the_reference_and_pose_check_charges_a_deferred_or_dropped_placement(monkeypatch, scail):
    """A deferral placed nothing, and a hard Clear VRAM dropped what was placed."""
    pending_load_relief.note_deferred(SCAIL)
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        _scail_render(scail)
    pending_load_relief.note_placed(SCAIL)
    _hard_clear(monkeypatch, scail)
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        _scail_render(scail)


def test_the_h3_and_footprint_prices_keep_their_explicit_preset_credit(monkeypatch, tmp_path, box):
    """Unchanged: an explicit preset credits the DiT in the #52 and H3 prices."""
    rig_ = _h3_rig(monkeypatch, tmp_path, preset="uly2")
    box.mem = _h3_need(weights=0) + 1 * _GIB
    _comfy(monkeypatch, box, models=())
    _render(ModelSpec(mesh=rig_.model.mesh, unet_name=UNET))
