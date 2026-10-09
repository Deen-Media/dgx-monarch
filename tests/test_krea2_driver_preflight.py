"""Driver-side preflight for krea2's reference-latents refusal (2026-07-28).

The mid-forward guard in adapters/krea2.py is checked only inside the rebound
forward, mid-sample on a worker, after mesh_setup.submit_sample has
dispatched. One rank raising there strands the mesh until a Recycle
(docs/TROUBLESHOOTING.md #43). This preflight makes the same decision at the
driver, before any worker RPC, through krea2_ref_latents_would_reject, the
predicate the mid-forward guard also calls, so the two cannot disagree.

CPU-only: folder_paths is a fake and sniff_checkpoint is monkeypatched, so no
ComfyUI module is imported.
"""
from __future__ import annotations

import types

import pytest

from dgx_monarch.adapters import detect as detect_mod
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.krea2 import krea2_reference_latents_summary
from dgx_monarch.nodes.render_preflight import (
    krea2_ref_preflight_summary,
    preflight_krea2_reference_latents,
)
from preflight_fails_open_helpers import (
    assert_missing_checkpoint_file_fails_open,
    assert_sniff_checkpoint_raising_fails_open,
    install_fake_folder_paths,
)


def _cond(reference_latents=None, reference_latents_method=None):
    d = {}
    if reference_latents is not None:
        d["reference_latents"] = reference_latents
    if reference_latents_method is not None:
        d["reference_latents_method"] = reference_latents_method
    return [["tensor", d]]


def test_summary_of_none_or_empty_is_false_none():
    assert krea2_reference_latents_summary(None) == (False, None)
    assert krea2_reference_latents_summary([]) == (False, None)


def test_summary_of_empty_dict_cond_is_false_none():
    assert krea2_reference_latents_summary([["tensor", {}]]) == (False, None)


def test_summary_reads_refs_without_method():
    cond = _cond(reference_latents=["ref"])
    assert krea2_reference_latents_summary(cond) == (True, None)


def test_summary_reads_method_without_refs():
    cond = _cond(reference_latents_method="index_timestep_zero")
    assert krea2_reference_latents_summary(cond) == (False, "index_timestep_zero")


def test_summary_reads_both_refs_and_method():
    cond = _cond(reference_latents=["ref"], reference_latents_method="index_timestep_zero")
    assert krea2_reference_latents_summary(cond) == (True, "index_timestep_zero")


def test_combined_summary_or_combines_refs_across_positive_and_negative():
    """Refs on the negative and a method on the positive combine to (True, method)."""
    positive = _cond(reference_latents_method="index_timestep_zero")
    negative = _cond(reference_latents=["ref"])
    assert krea2_ref_preflight_summary(positive, negative) == (True, "index_timestep_zero")


def test_combined_summary_handles_missing_negative():
    positive = _cond(reference_latents=["ref"], reference_latents_method="index_timestep_zero")
    assert krea2_ref_preflight_summary(positive, None) == (True, "index_timestep_zero")


class _Topo:
    def __init__(self, sequence_parallel=1, cfg=1):
        self.sequence_parallel = sequence_parallel
        self.cfg = cfg


def _fake_model():
    return types.SimpleNamespace(unet_name="krea2-model.safetensors")


def _install_fake_folder_paths(monkeypatch, path="/models/diffusion_models/krea2-model.safetensors"):
    install_fake_folder_paths(monkeypatch, path)


def test_no_ref_summary_never_raises_or_sniffs(monkeypatch):
    calls = []
    monkeypatch.setattr(detect_mod, "sniff_checkpoint", lambda path: calls.append(path) or ("krea2", "bf16"))
    preflight_krea2_reference_latents(_fake_model(), None, _Topo(sequence_parallel=2))
    assert calls == []


def test_uly2_method_resolved_krea2_raises(monkeypatch):
    _install_fake_folder_paths(monkeypatch)
    monkeypatch.setattr(detect_mod, "sniff_checkpoint", lambda path: ("krea2", "bf16"))
    with pytest.raises(UnsupportedModelError, match="reference latents"):
        preflight_krea2_reference_latents(
            _fake_model(), (True, "index_timestep_zero"), _Topo(sequence_parallel=2, cfg=1))


def test_cfg2_method_resolved_krea2_raises(monkeypatch):
    _install_fake_folder_paths(monkeypatch)
    monkeypatch.setattr(detect_mod, "sniff_checkpoint", lambda path: ("krea2", "bf16"))
    with pytest.raises(UnsupportedModelError, match="reference latents"):
        preflight_krea2_reference_latents(
            _fake_model(), (True, "index_timestep_zero"), _Topo(sequence_parallel=1, cfg=2))


def test_refs_without_method_does_not_raise(monkeypatch):
    _install_fake_folder_paths(monkeypatch)
    monkeypatch.setattr(detect_mod, "sniff_checkpoint", lambda path: ("krea2", "bf16"))
    preflight_krea2_reference_latents(
        _fake_model(), (True, None), _Topo(sequence_parallel=2, cfg=1))


def test_single_topology_does_not_raise(monkeypatch):
    _install_fake_folder_paths(monkeypatch)
    monkeypatch.setattr(detect_mod, "sniff_checkpoint", lambda path: ("krea2", "bf16"))
    preflight_krea2_reference_latents(
        _fake_model(), (True, "index_timestep_zero"), _Topo(sequence_parallel=1, cfg=1))


def test_dp2_topology_does_not_raise(monkeypatch):
    """dp2 is sp=1 and cfg=1: batch split only, stock forward per rank."""
    _install_fake_folder_paths(monkeypatch)
    monkeypatch.setattr(detect_mod, "sniff_checkpoint", lambda path: ("krea2", "bf16"))
    topo = _Topo(sequence_parallel=1, cfg=1)
    topo.dp = 2
    preflight_krea2_reference_latents(_fake_model(), (True, "index_timestep_zero"), topo)


def test_non_krea2_family_does_not_raise(monkeypatch):
    """qwen_image handles its own default_ref_method; this preflight covers
    family == 'krea2' only and stays silent for other families."""
    _install_fake_folder_paths(monkeypatch)
    monkeypatch.setattr(detect_mod, "sniff_checkpoint", lambda path: ("qwen_image", "bf16"))
    preflight_krea2_reference_latents(
        _fake_model(), (True, "index_timestep_zero"), _Topo(sequence_parallel=2, cfg=1))


def test_sniff_checkpoint_raising_fails_open(monkeypatch):
    assert_sniff_checkpoint_raising_fails_open(
        monkeypatch,
        preflight=preflight_krea2_reference_latents,
        model=_fake_model(),
        refusing_args=((True, "index_timestep_zero"), _Topo(sequence_parallel=2, cfg=1)),
        default_path="/models/diffusion_models/krea2-model.safetensors")


def test_missing_checkpoint_file_fails_open(monkeypatch):
    assert_missing_checkpoint_file_fails_open(
        monkeypatch,
        preflight=preflight_krea2_reference_latents,
        model=_fake_model(),
        refusing_args=((True, "index_timestep_zero"), _Topo(sequence_parallel=2, cfg=1)),
        family="krea2")


def test_preflight_runs_before_worker_dispatch(monkeypatch):
    """Integration-shaped: _submit_render_guarded must raise from the
    preflight before touching mesh_setup at all."""
    from dgx_monarch.nodes import common, render_submit

    _install_fake_folder_paths(monkeypatch)
    monkeypatch.setattr(detect_mod, "sniff_checkpoint", lambda path: ("krea2", "bf16"))

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
        topology_preset = "uly2"
        attention = "auto"

        @property
        def world(self):
            return 2

    model = types.SimpleNamespace(
        mesh=_MeshSpec(), unet_name="krea2-model.safetensors", options={})

    monkeypatch.setattr(common, "_bind_packed_render_model", lambda model, latent, cfg: (model, None))
    monkeypatch.setattr(render_submit, "model_for_request", lambda model, request: model)
    monkeypatch.setattr(common, "ensure_live", lambda handle: handle)
    # The fake handle is not a MeshHandle, so the real claim_render_session
    # runs lock-free (mesh_session._handle_state leaves test doubles
    # unregistered), and the dispatch stubs above are all it takes to prove no
    # worker RPC runs.
    monkeypatch.setattr(
        common, "_resolve_topology_for_latent",
        lambda model, latent, cfg, world: (_Topo(sequence_parallel=2, cfg=1), False, "preset: uly2"))

    request = {
        "_dgxm_krea2_ref_preflight": (True, "index_timestep_zero"),
    }
    latent = {"samples": None}

    from dgx_monarch.nodes.submit_guard import SubmitRenderGuard

    with pytest.raises(UnsupportedModelError, match="reference latents"):
        render_submit._submit_render_guarded(
            model, request, latent, cfg_value=8.0, steps_hint=20, seq=0, depth=1,
            progress_on_step=None, handoff=None, guard=SubmitRenderGuard())
