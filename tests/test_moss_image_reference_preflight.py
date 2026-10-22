"""Driver preflight for MOSS-Image reference images (#130).

ComfyUI 60aade0bb (2026-10-22) added edit LoRAs to MOSS-Image: its Text Encode
MOSS-Image Edit node attaches reference latents to the conditioning, and
stock's forward appends them, or keys and values computed once from them, to
the joint stream. No hardware leg has compared a cluster edit with stock, so
the driver refuses a MOSS-Image render that carries references on every
topology but one GPU, before any Worker call, with the text the sharded
forward raises on every rank. One GPU (mode=local with gpus_per_host=1) binds
nothing and runs stock's forward, so it passes.

CPU-only: folder_paths is a fake and sniff_checkpoint is monkeypatched, so no
ComfyUI module is imported.
"""
from __future__ import annotations

import types

import pytest

from dgx_monarch.adapters import detect as detect_mod
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.moss_image_seams import REFERENCE_IMAGES_REFUSAL
from dgx_monarch.nodes.model_patch_preflight import (
    preflight_moss_image_reference_images,
    renders_on_one_gpu,
)
from dgx_monarch.nodes.render_preflight import krea2_ref_preflight_summary
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from dgx_monarch.topology import Topology
from preflight_fails_open_helpers import (
    assert_missing_checkpoint_file_fails_open,
    assert_sniff_checkpoint_raising_fails_open,
    install_fake_folder_paths,
)

PATH = "/models/diffusion_models/moss_image_sft_bf16.safetensors"

# Every cluster topology a world-2 pair resolves, and FSDP on its own flag.
CLUSTER = {
    "uly2": Topology(ulysses=2, world=2),
    "ring2": Topology(ring=2, world=2),
    "cfg2": Topology(cfg=2, world=2),
    "dp2": Topology(dp=2, world=2),
    "uly2+fsdp": Topology(ulysses=2, fsdp=True, world=2),
    "ring2+fsdp": Topology(ring=2, fsdp=True, world=2),
}
ONE_GPU = Topology(world=1)


def _model():
    return types.SimpleNamespace(unet_name="moss_image_sft_bf16.safetensors")


def _sniff(monkeypatch, family="moss_image"):
    calls: list = []
    install_fake_folder_paths(monkeypatch, PATH)
    monkeypatch.setattr(detect_mod, "sniff_checkpoint",
                        lambda path: calls.append(path) or (family, "bf16"))
    return calls


def _edit_conditioning(method="index_timestep_zero_isolated"):
    """The keys TextEncodeMOSSImageEdit sets through conditioning_set_values."""
    return [["tensor", {"reference_latents": ["latent"], "reference_latents_method": method}]]


@pytest.mark.parametrize("name", sorted(CLUSTER))
def test_references_refuse_class_p_on_every_cluster_topology(monkeypatch, name):
    _sniff(monkeypatch)
    with pytest.raises(UnsupportedModelError) as caught:
        preflight_moss_image_reference_images(
            _model(), krea2_ref_preflight_summary(_edit_conditioning(), None), CLUSTER[name])
    assert str(caught.value) == REFERENCE_IMAGES_REFUSAL
    assert parse_leading_refusal_tag(str(caught.value)).refusal_class is RefusalClass.PHYSICS


@pytest.mark.parametrize("method", [None, "index", "index_timestep_zero",
                                    "index_timestep_zero_isolated",
                                    "index_timestep_zero_text_start"])
def test_every_reference_method_refuses_and_none_is_needed(monkeypatch, method):
    """Stock MOSS-Image uses references whatever the method, so presence decides."""
    _sniff(monkeypatch)
    with pytest.raises(UnsupportedModelError, match="carries reference images"):
        preflight_moss_image_reference_images(
            _model(), krea2_ref_preflight_summary(_edit_conditioning(method), None), CLUSTER["uly2"])


def test_references_on_the_negative_alone_refuse(monkeypatch):
    """The edit node's guide asks for it on the negative too; either side counts."""
    _sniff(monkeypatch)
    summary = krea2_ref_preflight_summary([["tensor", {}]], _edit_conditioning())
    with pytest.raises(UnsupportedModelError, match="carries reference images"):
        preflight_moss_image_reference_images(_model(), summary, CLUSTER["cfg2"])


def test_one_gpu_passes_without_reading_a_header(monkeypatch):
    calls = _sniff(monkeypatch)
    assert renders_on_one_gpu(ONE_GPU)
    preflight_moss_image_reference_images(
        _model(), krea2_ref_preflight_summary(_edit_conditioning(), None), ONE_GPU)
    assert calls == []


@pytest.mark.parametrize("name", sorted(CLUSTER))
def test_no_cluster_topology_counts_as_one_gpu(name):
    assert not renders_on_one_gpu(CLUSTER[name])


@pytest.mark.parametrize("summary", [None, (False, None), (False, "index_timestep_zero")],
                         ids=["no-summary", "no-references", "method-alone"])
def test_a_render_without_references_never_reads_a_header(monkeypatch, summary):
    calls = _sniff(monkeypatch)
    preflight_moss_image_reference_images(_model(), summary, CLUSTER["uly2"])
    assert calls == []


@pytest.mark.parametrize("family", ["krea2", "flux3", "qwen_image"])
def test_other_families_are_left_to_their_own_guards(monkeypatch, family):
    calls = _sniff(monkeypatch, family)
    preflight_moss_image_reference_images(_model(), (True, None), CLUSTER["uly2"])
    assert calls == [PATH]


def test_sniff_checkpoint_raising_fails_open(monkeypatch):
    assert_sniff_checkpoint_raising_fails_open(
        monkeypatch, preflight=preflight_moss_image_reference_images, model=_model(),
        refusing_args=((True, None), CLUSTER["uly2"]), default_path=PATH)


def test_missing_checkpoint_file_fails_open(monkeypatch):
    assert_missing_checkpoint_file_fails_open(
        monkeypatch, preflight=preflight_moss_image_reference_images, model=_model(),
        refusing_args=((True, None), CLUSTER["uly2"]), family="moss_image")


def test_preflight_runs_before_worker_dispatch(monkeypatch):
    """_submit_render_guarded refuses from this preflight before any mesh_setup call."""
    from dgx_monarch.nodes import common, render_submit
    from dgx_monarch.nodes.submit_guard import SubmitRenderGuard

    _sniff(monkeypatch)

    def _fail(*_a, **_k):
        pytest.fail("worker dispatch must not run when the driver preflight refuses")

    for name in ("prepare_sample", "submit_sample", "ensure_request_setup"):
        monkeypatch.setattr(render_submit.mesh_setup, name, _fail, raising=False)

    class _Handle:
        world = 2

        def request_setup(self, *_a, **_k):
            _fail()

    class _MeshSpec:
        handle = _Handle()
        topology_preset = "uly2"
        attention = "auto"
        world = 2

    model = types.SimpleNamespace(mesh=_MeshSpec(), unet_name="moss_image_sft_bf16.safetensors",
                                  options={})
    monkeypatch.setattr(common, "_bind_packed_render_model", lambda model, latent, cfg: (model, None))
    monkeypatch.setattr(render_submit, "model_for_request", lambda model, request: model)
    monkeypatch.setattr(common, "ensure_live", lambda handle: handle)
    monkeypatch.setattr(common, "_resolve_topology_for_latent",
                        lambda model, latent, cfg, world: (CLUSTER["uly2"], False, "preset: uly2"))
    request = {"_dgxm_krea2_ref_preflight": krea2_ref_preflight_summary(_edit_conditioning(), None)}
    with pytest.raises(UnsupportedModelError, match="carries reference images"):
        render_submit._submit_render_guarded(
            model, request, {"samples": None}, cfg_value=3.0, steps_hint=50, seq=0, depth=1,
            progress_on_step=None, handoff=None, guard=SubmitRenderGuard())
