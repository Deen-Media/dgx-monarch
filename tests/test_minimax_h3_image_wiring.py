"""MiniMax-H3-Image's driver wiring, from a real header to each family-keyed arm.

The family facts read the published headers as ``minimax_h3_image``
(adapters/minimax_h3_image_header.py). These tests start from a synthetic header in
the published layout (``minimax_h3_image_headers``) and the real ``sniff_checkpoint``,
not a stubbed family, and follow the family to every arm the adapter needs on the
driver: the auto row, the cfg and dp preflight, the sampling-node and context-window
refusals, the driver footprint profile and the activation rows. No ComfyUI import.
"""
from __future__ import annotations

import types

import pytest

import minimax_h3_image_headers as h3i
from dgx_monarch import driver_footprint, h3_activation, h3_context_windows, h3_rows, h3_sigma_shift, mesh_safety
from dgx_monarch.adapters import detect
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.minimax_h3_image import FAMILY, MINIMAX_H3_IMAGE_BATCH_CAPPED_MESSAGE
from dgx_monarch.nodes.render_preflight import preflight_minimax_h3_topology
from dgx_monarch.topology import Topology, choose_auto_topology
from preflight_fails_open_helpers import install_fake_folder_paths


@pytest.fixture(scope="module")
def sniffed(tmp_path_factory):
    """The family each published DiT layout reads as, through the real detector."""
    root = tmp_path_factory.mktemp("h3i")
    return {kind: (root / f"{kind}.safetensors", detect.sniff_checkpoint(
        str(h3i.write(root / f"{kind}.safetensors", kind, blocks=2)))[0]) for kind in h3i.KINDS}


def test_every_published_layout_sniffs_as_the_family(sniffed):
    assert {family for _path, family in sniffed.values()} == {FAMILY}


def test_auto_takes_row_55_uly2_for_the_family(sniffed):
    for _path, family in sniffed.values():
        decision = choose_auto_topology(family, "bf16", 1.0, 2)
        assert decision.rule.row == 55 and decision.topology.ulysses == 2


@pytest.mark.parametrize("topo", [Topology(cfg=2, world=2), Topology(dp=2, world=2)])
def test_the_driver_refuses_cfg2_and_dp2_from_the_real_header(monkeypatch, sniffed, topo):
    path, _family = sniffed["int8_convrot"]
    install_fake_folder_paths(monkeypatch, str(path))
    with pytest.raises(UnsupportedModelError) as caught:
        preflight_minimax_h3_topology(types.SimpleNamespace(unet_name=path.name), topo)
    assert str(caught.value) == MINIMAX_H3_IMAGE_BATCH_CAPPED_MESSAGE


def test_the_h3_nodes_refuse_the_sniffed_family(sniffed):
    family = sniffed["bf16"][1]
    assert "pdd_sigmas grid" in (h3_sigma_shift.family_refusal_reason(family) or "")
    assert h3_context_windows.family_refusal_reason(family) is not None


def test_the_driver_footprint_profile_prices_reference_pictures():
    profile = driver_footprint.DRIVER_STACK_FAMILIES[FAMILY]
    assert profile.family == FAMILY and profile.ref_extras_keys == (h3_rows.REFS_KEY,)
    assert "37.5" in profile.dit_note and "nvfp4_awq" in profile.dit_note


class _Fake:
    def __init__(self, shape):
        self.shape = shape


def _image_request(text=300, refs=2, ref_latent=(32, 32)):
    extras = {h3_rows.REFS_KEY: [{"kind": "image", "latent_h": ref_latent[0],
                                  "latent_w": ref_latent[1]} for _ in range(refs)]}
    return {"positive": [[_Fake((1, text, 5120)), extras]]}


def test_the_image_rows_are_text_references_and_one_frame_with_no_audio():
    latent = {"samples": _Fake((1, 24, 1, 64, 64))}  # 1024 square
    rows = h3_rows.rows_from_request(_image_request(), latent, image=True)
    assert rows.resolved and rows.audio == 0 and rows.latent_t == 1
    assert rows.video == h3_rows.frame_rows(64, 64) == 32 * 32
    assert rows.ref == 2 * h3_rows.ref_frame_rows(32, 32) and rows.text == 300
    # The video path still counts audio for the same latent.
    assert h3_rows.rows_from_request(_image_request(), latent).audio > 0


def test_the_activation_preflight_prices_the_family(monkeypatch):
    seen = []
    real = h3_activation.estimate_h3_activation

    def capture(**kwargs):
        seen.append(kwargs["rows"])
        return real(**kwargs)

    monkeypatch.setattr(h3_activation, "estimate_h3_activation", capture)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 120 * 2 ** 30)
    monkeypatch.setattr(h3_activation, "_co_resident_enough", lambda _mesh: True)
    monkeypatch.setattr(driver_footprint, "file_size_bytes", lambda _path: 2 ** 30)
    monkeypatch.delenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, raising=False)
    mesh = types.SimpleNamespace(topology_preset="uly2", worker_args={}, world=2,
                                 handle=types.SimpleNamespace(world=2))
    model = types.SimpleNamespace(unet_name="m.safetensors", mesh=mesh, options={})
    latent = {"samples": _Fake((1, 24, 1, 64, 64))}
    h3_activation.preflight_h3_activation_for_request(
        model, _image_request(), latent, family=FAMILY, path="/m.safetensors",
        unet_name="m.safetensors", memo_credit=True)
    assert len(seen) == 1 and seen[0].audio == 0 and seen[0].video == 1024


def _image_graph(width=2048, height=2048, extra=None):
    prompt = {"7": {"class_type": "MiniMaxH3ImageConditioning",
                    "inputs": {"width": width, "height": height, "prompt": "x"}}}
    prompt.update(extra or {})
    return prompt


def test_the_loader_site_charges_the_image_canvas():
    """The explicit-preset check before the load prices the image's floor and target rows."""
    from dgx_monarch.h3_calibration import H3_ACTIVATION_FLOOR_BYTES
    from dgx_monarch.nodes import loader_graph

    charged, why = loader_graph.graph_activation_bytes(_image_graph(), 2, family=FAMILY)
    expected = h3_activation.activation_bytes_for(
        h3_rows.H3Rows(video=64 * 64, frame_rows=64 * 64, latent_t=1, resolved=True), 2)
    assert charged == expected > H3_ACTIVATION_FLOOR_BYTES
    assert "the target frame only" in why
    # A side left at 0 reads as the node's 2048 default, with no images.
    assert loader_graph.graph_activation_bytes(_image_graph(0, 0), 2, family=FAMILY)[0] == charged
    assert loader_graph.activation_sp_degree(
        types.SimpleNamespace(topology_preset="uly2", world=2), FAMILY) >= 1


def test_each_family_reads_only_its_own_canvas_nodes():
    h3_canvas = {"3": {"class_type": "EmptyMiniMaxH3LatentAV",
                       "inputs": {"width": 1280, "height": 704, "length": 121}}}
    mixed = _image_graph(extra=h3_canvas)
    image_rows = h3_rows.rows_from_prompt_graph(mixed, image=True)
    video_rows = h3_rows.rows_from_prompt_graph(mixed)
    assert image_rows.audio == 0 and image_rows.latent_t == 1 and image_rows.video == h3_rows.frame_rows(128, 128) == 64 * 64
    assert video_rows.audio > 0 and video_rows.latent_t > 1
    assert h3_rows.rows_from_prompt_graph(_image_graph()).resolved is False
