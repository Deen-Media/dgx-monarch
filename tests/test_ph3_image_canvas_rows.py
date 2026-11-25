"""The loader prices the shipped p-h3-image graph's render activations.

p-h3-image runs H3's blocks (family ``minimax_h3``) on the latent
``PH3ImageConditioning`` returns: one frame on its canvas and the two audio
latents that frame spans (comfy_extras/nodes_minimax_h3.py, ``_av_latent(w, h,
1, 2)``), 8,285 rows at the template's 2912x2912. The loader's row read kept
only nodes named MiniMaxH3*, so it found no canvas in this graph and its
settled window charged no activations, the 11 GiB fixed floor included, where
the render site charges them one node later. These tests read the shipped
template through the loader's own price on the rig of ``test_graph_load_price``.
"""
from __future__ import annotations

import json

import pytest

from dgx_monarch import driver_footprint, h3_activation, h3_rows, mesh_safety
from dgx_monarch.nodes import loader_graph, loader_preflight
from test_graph_load_price import DIT, ROOT, _Mesh, _rig, _template, convert  # noqa: F401  # autouse rig.

_GIB = 2 ** 30


class _Fake:
    def __init__(self, shape):
        self.shape = shape


class _Nested:
    """The AV latent PH3ImageConditioning returns: video first, audio second."""

    def __init__(self, video, audio):
        self.tensors = [_Fake(video), _Fake(audio)]
        self.shape = video


def _render_site_rows(width: int, height: int) -> h3_rows.H3Rows:
    """The rows the render site reads from the node's latent, before text rows."""
    latent = {"samples": _Nested((1, 24, 1, height // 16, width // 16), (1, 32, 2, 2))}
    return h3_rows.rows_from_request({"positive": []}, latent)


def test_the_loader_reads_the_templates_canvas_as_the_render_site_does():
    rows = h3_rows.rows_from_prompt_graph(_template())
    assert rows.resolved and rows == _render_site_rows(2912, 2912)
    assert (rows.video, rows.audio, rows.total) == (91 * 91, 4, 8285)


@pytest.mark.parametrize("sp", [1, 2])
def test_the_settled_window_charges_the_templates_activations(sp):
    charged, note = h3_activation.graph_activation_bytes(_template(), sp, family="minimax_h3")
    assert charged == h3_activation.activation_bytes_for(_render_site_rows(2912, 2912), sp)
    assert "8285 declared rows" in note
    terms = loader_graph.driver_stack_terms(_template(), family="minimax_h3", sp_degree=sp)
    assert terms.activation_bytes == charged


def test_an_explicit_presets_load_refuses_where_dit_models_and_activations_do_not_fit(monkeypatch):
    """Under an explicit single preset the BF16 load window fits from about 84.2
    GiB; the settled window, with the render's activations, needs 90.9. At 88
    GiB the loader admitted the load. Its render site then reads about 14.7 GiB
    with the DiT and the 35.6 GiB of driver models in memory, short of the 12.6
    GiB of activations and the reserve, and passes only by unloading those
    driver models, the VAE the decode needs among them. The settled window
    prices that sum before anything loads."""
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 88 * _GIB)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(_Mesh("single"), DIT, {}, _template())
    assert "render activations 12.6 GiB" in str(raised.value)


@pytest.mark.parametrize("side", [(0, 0), (0, 2912), (2912, 0), (["40", 0], 2912), (2912, ["41", 1])])
def test_a_side_left_at_zero_reads_as_the_nodes_own_default(side):
    """0 is the node's own 2912; a linked side is unread and takes the same stand-in."""
    graph = _template()
    (node,) = [node for node in graph.values() if node["class_type"] == "PH3ImageConditioning"]
    node["inputs"]["width"], node["inputs"]["height"] = side
    assert h3_rows.rows_from_prompt_graph(graph) == _render_site_rows(2912, 2912)


@pytest.mark.parametrize("name", ["dgx-monarch-minimax-h3-image-t2i", "dgx-monarch-minimax-h3-image-edit"])
def test_the_minimax_h3_image_templates_are_read_as_before(name):
    """Their canvas node is the image family's own, read by the image arm."""
    graph = convert.convert(json.loads((ROOT / "example_workflows" / f"{name}.json").read_text()),
                            convert.repo_widget_names(ROOT))
    assert not h3_rows.rows_from_prompt_graph(graph).resolved
    rows = h3_rows.rows_from_prompt_graph(graph, image=True)
    assert rows.resolved and rows.audio == 0 and rows.latent_t == 1
