"""Driver preflight for minimax_h3's cfg-parallel and data-parallel refusals.

H3 uses a separate packed layout per conditioning and accepts only batch 1,
so cfg-parallel has no merged batch to split. Refuse on the driver before a
worker can fail mid-forward and strand its peer.

PackedDataParallelError in nodes/render_validation.py is the primary dp guard;
this preflight must also refuse dp > 1 if reached. It must allow uly2. The
call site dispatches by family, leaving degree checks to the adapter predicate.

CPU-only: folder_paths and sniff_checkpoint are monkeypatched, as in
tests/test_krea2_driver_preflight.py. No ComfyUI import is exercised.
"""
from __future__ import annotations

import types

import pytest

from dgx_monarch.adapters import detect as detect_mod
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.minimax_h3 import minimax_h3_topology_would_reject
from dgx_monarch.nodes.render_preflight import preflight_minimax_h3_topology
from dgx_monarch.topology import Topology
from preflight_fails_open_helpers import (
    assert_missing_checkpoint_file_fails_open,
    assert_sniff_checkpoint_raising_fails_open,
    install_fake_folder_paths,
)


def _fake_model():
    return types.SimpleNamespace(unet_name="minimax-h3-model.safetensors")


def _install_fake_folder_paths(monkeypatch, path="/models/minimax-h3-model.safetensors"):
    install_fake_folder_paths(monkeypatch, path)


def test_predicate_truth_table():
    assert minimax_h3_topology_would_reject(1, 1) is False
    assert minimax_h3_topology_would_reject(2, 1) is True
    assert minimax_h3_topology_would_reject(1, 2) is True
    assert minimax_h3_topology_would_reject(2, 2) is True


@pytest.mark.parametrize(
    "topo",
    [
        Topology(world=1),
        Topology(ulysses=2, world=2),
        Topology(ring=2, world=2),
        Topology(ulysses=2, ring=2, world=4),
    ],
)
def test_supported_topologies_never_raise_or_sniff(monkeypatch, topo):
    calls = []
    monkeypatch.setattr(
        detect_mod, "sniff_checkpoint",
        lambda path: calls.append(path) or ("minimax_h3", "bf16"))
    preflight_minimax_h3_topology(_fake_model(), topo)
    assert calls == []


def test_cfg2_raises_and_names_the_alternatives(monkeypatch):
    _install_fake_folder_paths(monkeypatch)
    monkeypatch.setattr(detect_mod, "sniff_checkpoint", lambda path: ("minimax_h3", "bf16"))
    with pytest.raises(UnsupportedModelError) as excinfo:
        preflight_minimax_h3_topology(_fake_model(), Topology(cfg=2, world=2))
    message = str(excinfo.value)
    assert "batch 1" in message
    assert "single" in message
    assert "uly" in message


def test_dp2_raises_as_the_backstop(monkeypatch):
    _install_fake_folder_paths(monkeypatch)
    monkeypatch.setattr(detect_mod, "sniff_checkpoint", lambda path: ("minimax_h3", "bf16"))
    with pytest.raises(UnsupportedModelError, match="data-parallel"):
        preflight_minimax_h3_topology(_fake_model(), Topology(dp=2, world=2))


def test_single_at_world_two_derives_dp_and_raises(monkeypatch):
    """The 'single' preset derives dp on a 2-rank mesh without saying so
    (docs/TROUBLESHOOTING.md #48, and #45 for Anima). The derived degree must
    reach the same refusal as an explicit dp2."""
    _install_fake_folder_paths(monkeypatch)
    monkeypatch.setattr(detect_mod, "sniff_checkpoint", lambda path: ("minimax_h3", "bf16"))
    from dgx_monarch.topology import topology_from_preset

    topo = topology_from_preset("single", 2)
    assert topo.dp == 2
    with pytest.raises(UnsupportedModelError):
        preflight_minimax_h3_topology(_fake_model(), topo)


def test_non_minimax_family_does_not_raise(monkeypatch):
    _install_fake_folder_paths(monkeypatch)
    monkeypatch.setattr(detect_mod, "sniff_checkpoint", lambda path: ("wan", "bf16"))
    preflight_minimax_h3_topology(_fake_model(), Topology(cfg=2, world=2))


def test_sniff_checkpoint_raising_fails_open(monkeypatch):
    assert_sniff_checkpoint_raising_fails_open(
        monkeypatch,
        preflight=preflight_minimax_h3_topology,
        model=_fake_model(),
        refusing_args=(Topology(cfg=2, world=2),),
        default_path="/models/minimax-h3-model.safetensors")


def test_missing_checkpoint_file_fails_open(monkeypatch):
    assert_missing_checkpoint_file_fails_open(
        monkeypatch,
        preflight=preflight_minimax_h3_topology,
        model=_fake_model(),
        refusing_args=(Topology(cfg=2, world=2),),
        family="minimax_h3")


def test_preflight_runs_before_worker_dispatch(monkeypatch):
    """Integration-shaped: _submit_render_guarded must raise from the preflight
    before it touches mesh_setup. The preflight runs after run_render's
    first-use gate ceremony, as the krea2 preflight does, but before mesh setup
    and every worker RPC."""
    from dgx_monarch.nodes import common, render_submit

    _install_fake_folder_paths(monkeypatch)
    monkeypatch.setattr(detect_mod, "sniff_checkpoint", lambda path: ("minimax_h3", "bf16"))

    def _fail(*_a, **_k):
        pytest.fail("worker dispatch must not run when the driver preflight refuses")

    monkeypatch.setattr(render_submit.mesh_setup, "prepare_sample", _fail, raising=False)
    monkeypatch.setattr(render_submit.mesh_setup, "submit_sample", _fail, raising=False)
    monkeypatch.setattr(render_submit.mesh_setup, "ensure_request_setup", _fail, raising=False)

    class _Handle:
        world = 2

        def request_setup(self, *_a, **_k):
            _fail()

    class _MeshSpec:
        handle = _Handle()
        topology_preset = "cfg2"
        attention = "auto"

        @property
        def world(self):
            return 2

    model = types.SimpleNamespace(
        mesh=_MeshSpec(), unet_name="minimax-h3-model.safetensors", options={})

    monkeypatch.setattr(common, "_bind_packed_render_model", lambda model, latent, cfg: (model, None))
    monkeypatch.setattr(render_submit, "model_for_request", lambda model, request: model)
    monkeypatch.setattr(common, "ensure_live", lambda handle: handle)
    monkeypatch.setattr(
        common, "_resolve_topology_for_latent",
        lambda model, latent, cfg, world: (Topology(cfg=2, world=2), False, "preset: cfg2"))

    from dgx_monarch.nodes.submit_guard import SubmitRenderGuard

    with pytest.raises(UnsupportedModelError, match="cfg-parallel"):
        render_submit._submit_render_guarded(
            model, {}, {"samples": None}, cfg_value=8.0, steps_hint=20,
            seq=0, depth=1, progress_on_step=None, handoff=None, guard=SubmitRenderGuard())
