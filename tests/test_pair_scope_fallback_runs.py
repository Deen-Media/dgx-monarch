"""An equal pair leaves a pair-scoped cfg2 row only for a route this render can run.

Each pair-scoped cfg2 row (Krea2 15, Chroma 19 and 20, Qwen-Image 68, Boogu 79)
sends an equal pair to a Ulysses row, which must have a bytes-equal reading
against one GPU AND be one this very render can run. Each test here goes through
the sampler's tag (reference_preflight.sequence_split_refused, read from the
pair and the checkpoint header) and the driver's real route choice
(nodes/common.py), and the Qwen-Image Edit 2511 case also through the real
sharded forward that refuses and the cfg2 forward that runs.

What each fallback row refuses that cfg2 admits:
- Qwen-Image (row 68 to 71): any text attention mask, and references whose
  method is index_timestep_zero, named or, with none named, Edit 2511's
  default. Plain Qwen-Image, Edit 2509 (index) and 2512 carry neither.
- Chroma (rows 19 and 20 to 22 and the uly2 fallback): a regional mask that
  hides a key (ComfyUI passes the flux family a mask only beside
  attention_mask_img_shape; a text keep-mask never reaches the forward), and
  index_timestep_zero references.
- Boogu (row 79 to 78): a caption mask that masks a row (num_tokens then
  differs from the caption rows).
- Krea2 (row 15 to 12): nothing a driver can see. Its references refuse cfg2
  too (preflight_krea2_reference_latents), its sharded forward accepts a mask,
  and the model patches it refuses on Ulysses never reach a Worker.
"""
from __future__ import annotations

import json
import math
import struct
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import family_select, latent_scale
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.qwen_image import QwenImageAdapter
from dgx_monarch.cfg_pair import text_rows_differ
from dgx_monarch.nodes import common
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.nodes.reference_preflight import sequence_split_refused
from dgx_monarch.nodes.render_preflight import sampler_latent

_BYTES = {"BF16": 2, "F8_E4M3": 1, "U8": 1}


def _checkpoint(path, dtype, *extra_keys):
    """A small header: one weight of ``dtype`` and any marker keys, as BF16
    unless a key names its dtype (``name=U8``)."""
    layers = {"blocks.0.attn.to_q.weight": (dtype, [8, 8])}
    for key in extra_keys:
        name, _, kind = key.partition("=")
        layers[name] = (kind or "BF16", [1])
    tensors: dict = {"__metadata__": {}}
    data = bytearray()
    for name, (kind, shape) in layers.items():
        offset = len(data)
        data.extend(bytes(math.prod(shape) * _BYTES[kind]))
        tensors[name] = {"dtype": kind, "shape": shape, "data_offsets": [offset, len(data)]}
    header = json.dumps(tensors).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)
    return str(path)


@pytest.fixture
def files(monkeypatch, tmp_path):
    return checkpoint_files(monkeypatch, tmp_path)


def checkpoint_files(monkeypatch, tmp_path):
    """Name a checkpoint for the next model; every reader finds it."""
    chosen: dict = {}
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: chosen["path"]
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(latent_scale, "comfy_spatial_downscale", lambda _path: 8)

    def make(dtype="BF16", *extra_keys):
        chosen["path"] = _checkpoint(tmp_path / f"model-{len(chosen)}.safetensors", dtype,
                                     *extra_keys)
        return chosen["path"]
    return make


def _model(family, weight_dtype="default"):
    return ModelSpec(
        mesh=MeshSpec(handle=SimpleNamespace(world=2), topology_preset="auto",
                      attention="TORCH_FLASH", sync_ulysses=True,
                      worker_args={family_select.WORKER_ARG_KEY: family}),
        unet_name="model.safetensors", options={"weight_dtype": weight_dtype})


def _pair(extra_positive=None, extra_negative=None, rows=8):
    """A prompt and a negative of equal text rows, with extra conditioning keys."""
    generator = torch.Generator().manual_seed(rows)
    positive = [[torch.randn(1, rows, 16, generator=generator), dict(extra_positive or {})]]
    negative = [[torch.randn(1, rows, 16, generator=generator), dict(extra_negative or {})]]
    return positive, negative


def _route(model, positive, negative, px, cfg=4.0, world=2, batch=1):
    """The tags a sampler node writes, then the driver's own route choice."""
    assert not text_rows_differ(positive, negative)
    latent = sampler_latent({"samples": torch.zeros(batch, 16, px // 8, px // 8)}, "euler",
                            False, False, sequence_split_refused(model, positive, negative))
    topology, _sage, reason = common._resolve_topology_for_latent(model, latent, cfg, world)
    return topology.describe(), reason


def _refs(method=None):
    extra = {"reference_latents": [torch.zeros(1, 16, 1, 8, 8)]}
    if method is not None:
        extra["reference_latents_method"] = method
    return extra


def test_an_edit_2511_equal_pair_keeps_cfg2_where_its_ulysses_row_refuses(files):
    """The finding: Edit 2511 FP8 at 1.5 MP with references and the default
    method. Its Ulysses row (71) refuses index_timestep_zero references; cfg2's
    stock forward runs them, so auto keeps row 68 and the render runs."""
    files("F8_E4M3", "__index_timestep_zero__")
    model = _model("qwen_image")
    positive, negative = _pair(_refs(), _refs())
    route, reason = _route(model, positive, negative, 1216)
    assert route == "cfg2" and "table row 68" in reason
    assert "refuses what the pair carries, so it keeps cfg2" in reason

    # The route auto no longer takes refuses this very conditioning.
    import test_qwen_image_adapter as qa

    sharded = qa._usp_fake("index_timestep_zero")
    qa._inject_usp(sharded)
    with pytest.raises(UnsupportedModelError, match="index_timestep_zero"):
        sharded._forward(torch.zeros(1, 16, 1, 8, 8), torch.zeros(1), positive[0][0],
                         ref_latents=positive[0][1]["reference_latents"])
    # The route it takes runs it: cfg2 hands the references to stock's forward.
    stock = qa._FakeQwenCfg()
    QwenImageAdapter().inject_cfg_pad_forward(stock)
    stock._forward(torch.zeros(1, 16, 1, 8, 8), torch.zeros(1), positive[0][0],
                   ref_latents=positive[0][1]["reference_latents"])
    assert torch.equal(stock.recorded[-1]["context"], positive[0][0])


@pytest.mark.parametrize(("case", "markers", "extra", "route", "row"), [
    # Plain Qwen-Image and 2512: no references, no mask; Ulysses runs the pair.
    ("plain or 2512", (), None, "uly2", 71),
    # Edit 2509 defaults to index references, which Ulysses runs.
    ("edit 2509", (), _refs(), "uly2", 71),
    # Edit 2511 with the method set to index in the graph runs on Ulysses too.
    ("edit 2511, index named", ("__index_timestep_zero__",), _refs("index"), "uly2", 71),
    ("edit 2511, default", ("__index_timestep_zero__",), _refs(), "cfg2", 68),
    ("any file, timestep zero named", (), _refs("index_timestep_zero"), "cfg2", 68),
    # Qwen-Image's sharded forward refuses any text mask.
    ("a text mask", (), {"attention_mask": torch.ones(1, 8)}, "cfg2", 68),
])
def test_row_68_sends_an_equal_pair_only_where_ulysses_runs_it(files, case, markers, extra,
                                                                route, row):
    files("F8_E4M3", *markers)
    resolved, reason = _route(_model("qwen_image"), *_pair(extra, extra), 1216)
    assert resolved == route and f"table row {row}" in reason, case


_REGIONAL = torch.tensor([[0.0, 0.0, float("-inf"), 0.0]])  # hides one key
_KEEP = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])  # a text encoder's keep-mask


@pytest.mark.parametrize(("px", "extra", "route", "row"), [
    # A regional mask ComfyUI passes the forward (it carries the image shape).
    (1216, {"attention_mask": _REGIONAL, "attention_mask_img_shape": (1, 2)}, "cfg2", 19),
    (1024, {"attention_mask": _REGIONAL, "attention_mask_img_shape": (1, 2)}, "cfg2", 20),
    # One that hides nothing runs on Ulysses.
    (1216, {"attention_mask": torch.zeros(1, 4), "attention_mask_img_shape": (1, 2)},
     "uly2", 22),
    # A keep-mask with no image shape never reaches Chroma's forward.
    (1216, {"attention_mask": _KEEP}, "uly2", 22),
    (1024, {"attention_mask": _KEEP}, "uly2", 99),
])
def test_chromas_rows_send_an_equal_pair_only_where_ulysses_runs_it(files, px, extra, route,
                                                                     row):
    files("BF16")
    resolved, reason = _route(_model("chroma"), *_pair(extra, extra), px, cfg=3.5)
    assert resolved == route and f"table row {row}" in reason


def test_chroma_keeps_an_equal_pair_with_timestep_zero_references_on_cfg2(files):
    files("BF16")
    resolved, reason = _route(_model("chroma"), *_pair(_refs("index_timestep_zero"),
                                                       _refs("index_timestep_zero")), 1216,
                              cfg=3.5)
    assert resolved == "cfg2" and "table row 19" in reason


@pytest.mark.parametrize(("mask", "route", "row"), [
    (torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]), "cfg2", 79),
    (torch.ones(1, 8), "uly2", 78),
])
def test_row_79_sends_an_equal_pair_only_where_ulysses_runs_it(files, mask, route, row):
    files("F8_E4M3")
    extra = {"attention_mask": mask}
    resolved, reason = _route(_model("boogu"), *_pair(extra, extra), 1024)
    assert resolved == route and f"table row {row}" in reason


def test_row_15_has_nothing_ulysses_refuses_that_cfg2_runs(files):
    """A masked Krea2 BF16 equal pair runs on Ulysses (row 12), and Krea2
    references refuse cfg2 as well as Ulysses, so neither moves the route."""
    from dgx_monarch.nodes.reference_preflight import preflight_krea2_reference_latents

    files("BF16")
    model = _model("krea2")
    mask = {"attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])}
    assert _route(model, *_pair(mask, mask), 1216, cfg=3.5)[0] == "uly2"
    refs = _refs("index")
    resolved, reason = _route(model, *_pair(refs, refs), 1216, cfg=3.5)
    assert resolved == "uly2" and "table row 12" in reason
    cfg2 = SimpleNamespace(sequence_parallel=1, cfg=2)
    with pytest.raises(UnsupportedModelError):
        preflight_krea2_reference_latents(model, (True, "index"), cfg2)


@pytest.mark.parametrize(("family", "dtype", "markers", "weight", "px", "row4", "row2"), [
    # Chroma FP8 from 1.2 to 1.64 MP: row 19 keeps the pair, since row 21's
    # Ulysses there runs SAGE.
    ("chroma", "F8_E4M3", (), "default", 1216, 21, 19),
    # A BF16 Chroma the loader casts to fp8 (fp8_cast), below 1.2 MP.
    ("chroma", "BF16", (), "fp8_e4m3fn", 1024, 99, 20),
    # A block-scaled Boogu file: an F8 weight with a U8 block scale.
    ("boogu", "F8_E4M3", ("blocks.0.attn.to_q.weight_scale=U8",), "default", 1024, 78, 79),
])
def test_a_lift_keeps_an_equal_pair_on_cfg2_only_where_the_row_runs(files, family, dtype,
                                                                    markers, weight, px,
                                                                    row4, row2):
    """At world 4 the cfg2 row leaves two data-parallel ranks that batch 1
    cannot split, where auto refuses the row, so the equal pair takes the row
    its scope sends it to, as before the lift; at world 2, and at world 4 with
    batch 2, the lift keeps cfg2."""
    files(dtype, *markers)
    model = _model(family, weight)
    pair = _pair()
    route, reason = _route(model, *pair, px, cfg=3.5, world=4)
    assert route == "uly2+ring2" and f"table row {row4}" in reason
    assert (f"cfg2 table row {row2} would keep this equal pair, but at world 4 it leaves 2 "
            "data-parallel ranks that batch 1 cannot split") in reason
    route, reason = _route(model, *pair, px, cfg=3.5, world=2)
    assert route == "cfg2" and f"table row {row2}" in reason
    route, reason = _route(model, *pair, px, cfg=3.5, world=4, batch=2)
    assert route == "cfg2+dp2" and f"table row {row2}" in reason


def test_a_refused_equal_pair_at_world_4_meets_the_refusal_that_names_what_runs_it(files):
    """At world 4, batch 1, the cfg2 row leaves two spare ranks, and every
    sequence route refuses the pair's references, so no route runs it: auto
    refuses at once (where the Ulysses and ring route used to load the model
    and refuse in its forward) and names what does run it. A header the
    driver cannot read never gets this far: auto refuses it itself."""
    files("BF16")
    refs = _refs("index_timestep_zero")
    with pytest.raises(UnsupportedModelError) as caught:
        _route(_model("chroma"), *_pair(refs, refs), 1216, cfg=3.5, world=4)
    message = str(caught.value)
    assert "auto matched table row 19 for chroma" in message
    assert ("render at world 2 or on one GPU: every preset that tiles world 4 splits the "
            "sequence, and chroma's sharded forward refuses what this pair carries") in message
    assert "a multiple of 2" in message and "uly4, ring4" not in message
    route, reason = _route(_model("chroma"), *_pair(refs, refs), 1216, cfg=3.5, world=4,
                           batch=2)
    assert route == "cfg2+dp2" and "table row 19" in reason


def test_a_truncated_header_meets_autos_own_refusal_before_any_row(files):
    """The tag answers True where it cannot read the header, but auto cannot
    read it either and refuses before any row, so that answer never keeps a
    pair from a route it could run."""
    from dgx_monarch.adapters.detect import CheckpointSniffError

    path = files("F8_E4M3", "__index_timestep_zero__")
    with open(path, "rb") as handle:
        head = handle.read(20)
    with open(path, "wb") as handle:
        handle.write(head)
    model = _model("qwen_image")
    pair = _pair(_refs(), _refs())
    assert sequence_split_refused(model, *pair) is True
    for world in (2, 4):
        with pytest.raises(CheckpointSniffError):
            _route(model, *pair, 1216, world=world)


def test_class_k_names_no_sequence_preset_for_a_refused_pair():
    """A differing pair keeps the cfg2 row and meets class K at world 4, batch
    1. Every preset that tiles world 4 splits the sequence, so for a pair the
    sequence forward refuses, the refusal names what does run it."""
    from dgx_monarch.topology import choose_auto_topology

    for refused in (False, True):
        with pytest.raises(UnsupportedModelError) as caught:
            choose_auto_topology("chroma", "fp8", 1.5, 4, cfg_value=3.5, pair_differs=True,
                                 sequence_refused=refused)
        message = str(caught.value)
        assert "send a batch size that is a multiple of 2" in message
        assert ("uly4, ring4, uly2+cfg2" in message) is not refused
        if refused:
            assert ("render at world 2 or on one GPU: every preset that tiles world 4 "
                    "splits the sequence") in message


def test_an_unreadable_header_keeps_a_masked_equal_pair_on_cfg2(files, monkeypatch):
    """cfg2 runs every case above, so a header the driver cannot read keeps it."""
    from dgx_monarch.adapters import detect

    files("F8_E4M3")
    model = _model("qwen_image")

    def unreadable(_path):
        raise OSError("gone")

    monkeypatch.setattr(detect, "sniff_checkpoint", unreadable)
    mask = {"attention_mask": torch.ones(1, 8)}
    assert sequence_split_refused(model, *_pair(mask, mask)) is True
    assert sequence_split_refused(model, *_pair()) is False
