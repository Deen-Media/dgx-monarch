"""MiniMax H3 at four Sparks (issue #188): the uly4 fold, and the driver's ring-pad refusal.

``auto`` used to fold H3's spare ranks at world 4 into uly2+ring2, where the
adapter refuses every packed total the degree does not divide, inside the
first model call, after setup and the load. Two things change and this file
holds both. The fold table grants H3 uly4 at world 4 and nothing else moves:
every other row at every world, and every row at world 2, decides as it did
with the table empty. And where a ring layout remains (a typed ring preset,
or ``auto`` at an odd world or above 4), the driver counts each model call's
packed total and refuses a padded one before the gate, setup and the load,
with the Worker's own card (nodes/h3_ring_pad.py). The count is pinned to
stock's ``PackedLayout`` by canary seam 62; this file pins it by hand.
"""
from __future__ import annotations

import types

import pytest
import torch

import dgx_monarch.topology as topology_mod
from dgx_monarch import h3_rows
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.detect import detect_family_from_header
from dgx_monarch.nodes import common, h3_ring_pad, render_preflight
from dgx_monarch.refusal import RefusalClass, parse_refusal_tag
from dgx_monarch.topology import PRESETS, Topology, choose_auto_topology, topology_from_preset
from dgx_monarch.topology_table import AUTO_TABLE

H3 = "minimax_h3"
H3_IMAGE = "minimax_h3_image"

# --- the fold --------------------------------------------------------------


def test_the_reporters_shape_resolves_to_uly4():
    """INT8 ConvRot ref2v at 896x1152 (1.03 MP a frame), world 4, CFG 1, batch 1."""
    decision = choose_auto_topology(H3, "int8", 896 * 1152 / 1e6, 4, cfg_value=1.0, batch_size=1)
    topo = decision.topology
    assert (topo.ulysses, topo.ring, topo.cfg, topo.dp) == (4, 1, 1, 1)
    assert decision.rule.row == 53
    assert "folded into ulysses4" in decision.reason
    assert "have not reproduced" in decision.reason


@pytest.mark.parametrize(("world", "layout"), [
    (2, "uly2"), (3, "ring3"), (4, "uly4"), (5, "ring5"), (6, "uly2+ring3"), (8, "uly2+ring4"),
])
@pytest.mark.parametrize("quant", ["bf16", "fp8", "int8"])
@pytest.mark.parametrize("cfg", [1.0, 4.0])
def test_h3_auto_at_each_world(world, layout, quant, cfg):
    """Only world 4 has a grant; 56 heads do not split 3, 5 or 6 ways, and no
    reading covers uly8, so the other worlds keep the ring fold or pure ring."""
    decision = choose_auto_topology(H3, quant, 1.03, world, cfg_value=cfg, batch_size=1)
    assert decision.topology.describe() == layout
    decision.topology.validate()


def test_the_grant_names_h3_at_world_4_only():
    assert topology_mod.VALIDATED_ULYSSES_FOLDS == {H3: {4: (56,)}}
    assert all(56 % degree == 0 for degree in topology_mod.VALIDATED_ULYSSES_FOLDS[H3])


def test_h3_image_keeps_the_ring_fold_it_has_no_reading_for():
    decision = choose_auto_topology(H3_IMAGE, "bf16", 1.0, 4, cfg_value=1.0, batch_size=1)
    assert decision.topology.describe() == "uly2+ring2"


def test_a_batch_that_splits_still_keeps_dp_for_the_dp_refusal():
    """Batch 2 at world 4 divides the two spare ranks, so no fold runs and the
    driver's H3 dp refusal answers (render_preflight.preflight_minimax_h3_topology)."""
    decision = choose_auto_topology(H3, "bf16", 1.0, 4, cfg_value=1.0, batch_size=2)
    assert decision.topology.describe() == "uly2+dp2"


def _row_inputs(rule):
    """Inputs that reach ``rule``: a megapixel value inside its band, its first
    quant, CFG 1 and 4, and both answers of the pair and cfg++ tags."""
    mp = rule.mp_min + 0.05 if rule.mp_max == float("inf") else (rule.mp_min + rule.mp_max) / 2
    quant = rule.quants[0] if rule.quants else "bf16"
    for cfg in (1.0, 4.0):
        for pair_differs in (False, True):
            for keeps_uncond in (False, True):
                yield {"family": rule.family, "quant_kind": quant, "megapixels": mp,
                       "cfg_value": cfg, "pair_differs": pair_differs,
                       "keeps_uncond": keeps_uncond}


def _decide(kwargs, world, batch):
    try:
        decision = choose_auto_topology(world=world, batch_size=batch, **kwargs)
    except UnsupportedModelError as exc:
        return ("refused", str(exc))
    return (decision.topology, decision.rule.row, decision.reason)


@pytest.mark.parametrize("world", [2, 3, 4, 5, 6, 8])
def test_table_wide_only_h3_at_world_4_moves(monkeypatch, world):
    """Every row decides as it did with the table empty, at world 2 for every
    row, and at every other world for every family but H3 at world 4."""
    granted = {}
    for rule in AUTO_TABLE:
        for kwargs in _row_inputs(rule):
            for batch in (1, 2, 3):
                granted[(rule.row, tuple(sorted(kwargs.items())), batch)] = _decide(
                    kwargs, world, batch)
    monkeypatch.setattr(topology_mod, "VALIDATED_ULYSSES_FOLDS", {})
    moved = []
    for rule in AUTO_TABLE:
        for kwargs in _row_inputs(rule):
            for batch in (1, 2, 3):
                before = _decide(kwargs, world, batch)
                after = granted[(rule.row, tuple(sorted(kwargs.items())), batch)]
                if before[:2] != after[:2]:
                    moved.append((rule.family, batch, before[0], after[0]))
    if world != 4:
        assert moved == []
        return
    assert moved and all(family == H3 for family, *_ in moved)
    for _family, batch, before, after in moved:
        assert batch in (1, 3)
        assert before.describe() == "uly2+ring2" and after.describe() == "uly4"
    # Checkpoints with no row of their own move with the family they read as.
    for name, header in NAMED_CHECKPOINTS.items():
        assert detect_family_from_header(list(header["__tensors__"]), header) == H3, name


def _p_h3_image_header() -> dict:
    """p-h3-image's BF16 DiT header as the release lays it out (block 0 and the
    shared rows; tests/test_p_h3_image_support.py copies the same layout)."""
    tensors = {
        "adaln_t_table": {"dtype": "F32", "shape": [4097, 16]},
        "video_patch_proj.weight": {"dtype": "F32", "shape": [5376, 96]},
        "audio_patch_proj.weight": {"dtype": "F32", "shape": [5376, 32]},
        "condition_proj.weight": {"dtype": "BF16", "shape": [5376, 5120]},
        "blocks.0.adaln_proj.linear.weight": {"dtype": "F32", "shape": [96768, 16]},
        "blocks.0.attn.q_norm.weight": {"dtype": "BF16", "shape": [128]},
        "blocks.0.attn.qkv_proj.weight": {"dtype": "BF16", "shape": [21504, 5376]},
        "blocks.0.attn.out_proj.weight": {"dtype": "BF16", "shape": [5376, 7168]},
        "blocks.0.mlp.fc1.weight": {"dtype": "BF16", "shape": [28672, 5376]},
        "blocks.0.mlp.fc2.weight": {"dtype": "BF16", "shape": [5376, 14336]},
        "token_refiner.blocks.0.attn.qkv_proj.weight": {"dtype": "BF16", "shape": [21504, 5376]},
    }
    return {"__tensors__": tensors}


# Checkpoints that read as a table family and have no row of their own. Each
# takes its family's decisions, the world-4 fold included, so each is named here.
NAMED_CHECKPOINTS = {"p-h3-image": _p_h3_image_header()}


def test_p_h3_image_takes_the_h3_fold_through_the_same_dit_attention():
    """p-h3-image reads as minimax_h3 and runs H3's DiT: its one sharded
    attention has the 56 heads the fold names, so it takes uly4 at world 4
    with no reading of its own, and everything else H3 decides."""
    header = NAMED_CHECKPOINTS["p-h3-image"]
    tensors = header["__tensors__"]
    assert detect_family_from_header(list(tensors), header) == H3
    head_dim = tensors["blocks.0.attn.q_norm.weight"]["shape"][0]
    heads = tensors["blocks.0.attn.qkv_proj.weight"]["shape"][0] // (3 * head_dim)
    assert (heads,) == topology_mod.VALIDATED_ULYSSES_FOLDS[H3][4]
    for world, layout in ((2, "uly2"), (3, "ring3"), (4, "uly4"), (8, "uly2+ring4")):
        decision = choose_auto_topology(H3, "bf16", 8.48, world, cfg_value=1.0, batch_size=1)
        assert decision.topology.describe() == layout


def test_the_row_note_keeps_its_provenance_phrases():
    decision = choose_auto_topology(H3, "bf16", 1.0, 2, cfg_value=4.0)
    assert "packed video+audio single sequence" in decision.reason
    assert "typed-refuse on the driver" in decision.reason


def test_a_pairs_reason_speaks_of_the_pair_and_the_fold_says_where_it_stands():
    """The world-4 clause joins the logged reason only where the fold runs."""
    for world in (2, 3, 6, 8):
        reason = choose_auto_topology(H3, "int8", 1.03, world, cfg_value=1.0).reason
        assert "community four-Spark users" not in reason and "uly4" not in reason, world
    reason = choose_auto_topology(H3, "int8", 1.03, 4, cfg_value=1.0).reason
    assert reason.index("folded into ulysses4") < reason.index(
        "uly4 rests on what community four-Spark users reported")
    assert "have not reproduced" in reason


# --- the exact count -------------------------------------------------------


class _Nested:
    """The NestedTensor duck the driver reads: a video and an audio latent."""

    def __init__(self, video, audio):
        self.tensors = [video, audio]
        self.shape = video.shape


def _latent(latent_t=72, lat_h=72, lat_w=56, audio_t=405, **tags):
    return {"samples": _Nested(torch.zeros(1, 24, latent_t, lat_h, lat_w),
                               torch.zeros(1, 32, 2, audio_t)), **tags}


def _entry(text_rows, **extras):
    return [torch.zeros(1, text_rows, 8), extras]


def test_the_exact_count_by_hand():
    """896x1152, 243 frames: 72 latent frames of 36x28 rows, 405 audio frames.
    A still guide adds one frame of the target grid; an image reference adds its
    own floor-halved grid, a video reference its frames and two rows per audio
    frame."""
    geometry = h3_rows.exact_geometry(_latent())
    assert geometry == (72, 36 * 28, 405)
    target = 72 * 1008 + 2 * 405
    assert h3_rows.exact_call_total(_entry(101), geometry) == 101 + target
    guide = {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 72, 56)}
    image_ref = {"kind": "image", "latent_h": 71, "latent_w": 57}
    video_ref = {"kind": "video_audio", "ref_audio_t": 3, "latent_t": 2,
                 "latent_h": 24, "latent_w": 40}
    entry = _entry(101, minimax_keyframes=[guide], minimax_refs=[image_ref, video_ref])
    assert h3_rows.exact_call_total(entry, geometry) == (
        101 + target + 1008 + 35 * 28 + (2 * 3 + 2 * 12 * 20))


def test_the_image_model_counts_text_references_and_one_frame():
    geometry = h3_rows.exact_geometry({"samples": torch.zeros(1, 16, 1, 128, 96)}, image=True)
    assert geometry == (1, 64 * 48, 0)
    guide = {"resolved_frame_index": 0, "latent": torch.zeros(1, 24, 1, 128, 96)}
    entry = _entry(30, minimax_keyframes=[guide],
                   minimax_refs=[{"kind": "image", "latent_h": 64, "latent_w": 64}])
    assert h3_rows.exact_call_total(entry, geometry, image=True) == 30 + 3072 + 32 * 32


@pytest.mark.parametrize("entry", [
    [None, {}],
    _entry(5, minimax_refs=[{"kind": "a kind comfy has not named", "latent_h": 4}]),
    _entry(5, minimax_refs=[{"kind": "image", "latent_h": 4}]),
    _entry(5, minimax_refs=[{"kind": "image", "latent_h": True, "latent_w": 4}]),
    _entry(5, minimax_keyframes=[{"latent": object()}]),
    _entry(5, minimax_refs="not a list"),
    [torch.zeros(1, 5, 8), "not a mapping"],
])
def test_anything_unread_counts_nothing(entry):
    assert h3_rows.exact_call_total(entry, (2, 4, 3)) is None


def test_an_unreadable_latent_reads_no_grid():
    assert h3_rows.exact_geometry({"samples": torch.zeros(1, 24, 2, 4, 4)}) is None  # no audio
    assert h3_rows.exact_geometry({"samples": torch.zeros(4, 4)}, image=True) is None
    assert h3_rows.exact_geometry("not a latent") is None


# --- the preflight ---------------------------------------------------------


def _mesh(world, preset="auto", **kwargs):
    return common.MeshSpec(handle=types.SimpleNamespace(world=world), topology_preset=preset,
                           attention="TORCH_FLASH", sync_ulysses=False, worker_args={}, **kwargs)


def _model(world, preset="auto", **kwargs):
    return common.ModelSpec(mesh=_mesh(world, preset), unet_name="h3.safetensors", **kwargs)


@pytest.fixture
def h3_checkpoint(monkeypatch):
    """The driver reads the checkpoint as H3, the latent as built, and no waiver."""
    family = {"name": H3}
    monkeypatch.setattr(h3_ring_pad, "_family_and_path",
                        lambda _model: (family["name"], "/models/h3.safetensors"))
    monkeypatch.setattr(h3_ring_pad.latent_scale, "rendered_latent_shape",
                        lambda _path, samples, *_tags: tuple(samples.shape))
    waived = {"on": False, "asked": []}

    def guard_waived_for(model, topo, world):
        waived["asked"].append((topo.describe(), world, h3_ring_pad.GUARD))
        return waived["on"]

    monkeypatch.setattr(h3_ring_pad, "guard_waived_for", guard_waived_for)
    return types.SimpleNamespace(family=family, waived=waived)


def _request(positive_rows, negative_rows=None, cfg=1.0):
    request = {"kind": "ksampler", "cfg": cfg, "positive": [_entry(positive_rows)]}
    if negative_rows is not None:
        request["negative"] = [_entry(negative_rows)]
    return request


def _target_rows():
    latent_t, rows_per_frame, audio_t = h3_rows.exact_geometry(_latent())
    return latent_t * rows_per_frame + 2 * audio_t   # 73386, which is 2 modulo 4


def _text_for(total_mod, degree):
    """Text rows that leave the packed total at ``total_mod`` modulo ``degree``."""
    return next(rows for rows in range(1, 64) if (rows + _target_rows()) % degree == total_mod)


def _refusal(exc_info):
    tag = parse_refusal_tag(str(exc_info.value))
    assert tag is not None
    return tag


def test_world_3_auto_refuses_a_padded_total_with_the_workers_card(h3_checkpoint):
    model = _model(3)
    topo = choose_auto_topology(H3, "int8", 1.03, 3, cfg_value=1.0).topology
    assert topo.describe() == "ring3"
    with pytest.raises(UnsupportedModelError) as exc_info:
        h3_ring_pad.preflight(model, _request(_text_for(1, 3)), _latent(), topo, 3, 1.0)
    tag = _refusal(exc_info)
    assert tag.refusal_class is RefusalClass.KNOWN_WRONG
    assert tag.guard == "ring_pad:minimax_h3" and tag.waivable
    text = str(exc_info.value)
    assert "`auto` resolved ring3 at world 3" in text
    assert "needs 2 divisibility pad row(s) to split 3 ways" in text
    assert "No Ulysses preset fills world 3" in text
    assert "before the first-use check, setup and sampling" in text
    assert "#49" in text
    assert h3_checkpoint.waived["asked"] == [("ring3", 3, "ring_pad:minimax_h3")]


def test_a_total_the_degree_divides_renders_on_ring(h3_checkpoint):
    topo = choose_auto_topology(H3, "int8", 1.03, 3, cfg_value=1.0).topology
    h3_ring_pad.preflight(_model(3), _request(_text_for(0, 3)), _latent(), topo, 3, 1.0)


@pytest.mark.parametrize(("world", "preset", "names"), [
    (4, "ring4", "uly4"), (2, "ring2", "uly2"),
])
def test_a_typed_ring_preset_names_the_ulysses_preset_that_fits(h3_checkpoint, world, preset, names):
    topo = topology_from_preset(preset, world)
    with pytest.raises(UnsupportedModelError) as exc_info:
        h3_ring_pad.preflight(_model(world, preset), _request(_text_for(1, world)), _latent(),
                              topo, world, 1.0)
    text = str(exc_info.value)
    assert f"topology {preset} runs at world {world}" in text
    assert f"Set the Init node's topology to {names}," in text


def test_ulysses_layouts_are_never_checked(h3_checkpoint):
    for world, preset in ((2, "uly2"), (4, "uly4"), (2, "uly2+fsdp")):
        h3_ring_pad.preflight(_model(world, preset), _request(_text_for(1, world)), _latent(),
                              topology_from_preset(preset, world), world, 1.0)
    assert h3_checkpoint.waived["asked"] == []


def test_a_live_waiver_lets_the_render_through_to_the_worker(h3_checkpoint):
    h3_checkpoint.waived["on"] = True
    topo = topology_from_preset("ring4", 4)
    h3_ring_pad.preflight(_model(4, "ring4"), _request(_text_for(3, 4)), _latent(), topo, 4, 1.0)


@pytest.mark.parametrize(("cfg", "tags", "counted"), [
    (1.0, {}, False),
    (4.0, {}, True),
    (1.0, {render_preflight.CFG_KEEPS_UNCOND_METADATA_KEY: True}, True),
])
def test_the_negative_counts_only_when_its_call_runs(h3_checkpoint, cfg, tags, counted):
    """ComfyUI skips the uncond call at CFG 1.0 unless a cfg++ sampler keeps it."""
    topo = topology_from_preset("ring4", 4)
    request = _request(_text_for(0, 4), negative_rows=_text_for(2, 4), cfg=cfg)
    call = lambda: h3_ring_pad.preflight(  # noqa: E731
        _model(4, "ring4"), request, _latent(**tags), topo, 4, cfg)
    if not counted:
        call()
        return
    with pytest.raises(UnsupportedModelError, match="the negative prompt's packed sequence"):
        call()


@pytest.mark.parametrize(("kind", "counted"), [("cfg", True), ("basic", False), ("dual_model", False)])
def test_sampler_custom_reads_the_guiders_pair(h3_checkpoint, kind, counted):
    topo = topology_from_preset("ring4", 4)
    guider = {"kind": kind, "positive": [_entry(_text_for(0, 4))],
              "negative": [_entry(_text_for(1, 4))], "cfg": 3.0}
    request = {"kind": "custom", "guider": guider}
    call = lambda: h3_ring_pad.preflight(  # noqa: E731
        _model(4, "ring4"), request, _latent(), topo, 4, 3.0)
    if not counted:
        call()
        return
    with pytest.raises(UnsupportedModelError, match="negative"):
        call()


def test_context_windows_leave_the_check_to_the_worker(h3_checkpoint):
    """Each window packs its own total with the guides stock crops to it."""
    from dgx_monarch.h3_context_windows import freeze_context_windows

    windows = freeze_context_windows({"context_length": 124, "context_overlap": 39,
                                      "fuse_method": "pyramid", "freenoise": True,
                                      "split_conds_to_windows": False})
    model = _model(4, "ring4", context_windows=windows)
    h3_ring_pad.preflight(model, _request(_text_for(1, 4)), _latent(),
                          topology_from_preset("ring4", 4), 4, 1.0)


def test_unread_conditioning_leaves_the_check_to_the_worker(h3_checkpoint):
    request = {"kind": "ksampler", "cfg": 1.0, "positive": [[None, {}]]}
    h3_ring_pad.preflight(_model(4, "ring4"), request, _latent(),
                          topology_from_preset("ring4", 4), 4, 1.0)


def test_other_families_are_not_read(monkeypatch):
    monkeypatch.setattr(h3_ring_pad, "_family_and_path", lambda _model: None)
    h3_ring_pad.preflight(_model(4, "ring4"), _request(_text_for(1, 4)), _latent(),
                          topology_from_preset("ring4", 4), 4, 1.0)


def test_h3_image_refuses_class_p_as_its_worker_does(h3_checkpoint):
    h3_checkpoint.family["name"] = H3_IMAGE
    latent = {"samples": torch.zeros(1, 16, 1, 128, 96)}
    topo = choose_auto_topology(H3_IMAGE, "bf16", 3.1, 4, cfg_value=1.0).topology
    assert topo.describe() == "uly2+ring2"
    rows = next(r for r in range(1, 9) if (r + 64 * 48) % 4)
    with pytest.raises(UnsupportedModelError) as exc_info:
        h3_ring_pad.preflight(_model(4), _request(rows), latent, topo, 4, 1.0)
    tag = _refusal(exc_info)
    assert tag.refusal_class is RefusalClass.PHYSICS and tag.guard is None
    text = str(exc_info.value)
    # No report covers MiniMax-H3-Image on uly4, and the 2026-07-10 corruption
    # was not measured on this family, so its message says neither.
    assert ("Set the Init node's topology to uly4 (exact by construction; no run of it "
            "is on record),") in text
    assert "left-edge corruption" not in text
    assert h3_checkpoint.waived["asked"] == []


@pytest.mark.parametrize(("world", "preset", "marked"), [
    (2, "ring2", False), (4, "ring4", False), (8, "ring8", True),
])
def test_a_named_preset_no_run_covers_says_so(h3_checkpoint, world, preset, marked):
    """uly2 has the project's runs and H3's uly4 the reports its fold rests on;
    uly8 has neither, so the refusal marks it exact by construction and unrun."""
    with pytest.raises(UnsupportedModelError) as exc_info:
        h3_ring_pad.preflight(_model(world, preset), _request(_text_for(1, world)), _latent(),
                              Topology(ring=world, world=world), world, 1.0)
    text = str(exc_info.value)
    name = f"uly{world}"
    assert f"Set the Init node's topology to {name}" in text
    assert ("(exact by construction; no run of it is on record)" in text) is marked
    assert "the left-edge corruption measured on 2026-07-10" in text


@pytest.fixture
def consent_rig(tmp_path, monkeypatch):
    """The real consent modules on a temporary store, as tests/test_accuracy_waivers.py sets them up."""
    import sys

    from dgx_monarch import accuracy_waiver, consent_pending, consent_store
    from dgx_monarch.nodes import consent_routes, consent_waiver

    checkpoint = tmp_path / "h3.safetensors"
    checkpoint.write_bytes(b"weights" * 64)
    lora = tmp_path / "style.safetensors"
    lora.write_bytes(b"lora" * 64)
    files = {("diffusion_models", "h3.safetensors"): str(checkpoint),
             ("loras", "style.safetensors"): str(lora)}
    output = tmp_path / "output"
    output.mkdir()
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = (  # type: ignore[attr-defined]
        lambda kind, name: files.get((kind, name)))
    folder_paths.get_output_directory = lambda: str(output)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(consent_store, "MEMO_PATH", str(tmp_path / "memo.json"))
    monkeypatch.setattr(consent_routes, "_RATE",
                        {"tokens": float(consent_routes.RATE_LIMIT_N), "t": 0.0})
    for spec in consent_pending.KIND_SPECS.values():
        monkeypatch.delenv(spec.env_var, raising=False)
    monkeypatch.delenv(consent_pending.AUTO_RESCUE_ENV, raising=False)
    consent_pending.clear_all()
    accuracy_waiver.clear()
    consent_waiver.reset_pending_audit()
    monkeypatch.setattr(consent_waiver, "_gate_ceremony_active", lambda: False)
    monkeypatch.setattr(h3_ring_pad, "_family_and_path", lambda _model: (H3, str(checkpoint)))
    monkeypatch.setattr(h3_ring_pad.latent_scale, "rendered_latent_shape",
                        lambda _path, samples, *_tags: tuple(samples.shape))
    yield types.SimpleNamespace(routes=consent_routes, pending=consent_pending,
                                waiver=consent_waiver)
    consent_pending.clear_all()
    accuracy_waiver.clear()
    consent_waiver.reset_pending_audit()


def test_the_drivers_card_grants_the_waiver_its_next_dispatch_stamps(consent_rig):
    """Refuse on the driver, raise the card, accept it, then the same render
    passes the driver and the dispatch carries the grant to the Workers."""
    from dgx_monarch import accuracy_waiver, consent_descriptor

    topo = topology_from_preset("ring4", 4)
    model = _model(4, "ring4", loras=({"name": "style.safetensors", "strength": 0.8},))
    request = _request(_text_for(1, 4))
    with pytest.raises(UnsupportedModelError) as exc_info:
        h3_ring_pad.preflight(model, request, _latent(), topo, 4, 1.0)
    descriptor = consent_descriptor.parse(str(exc_info.value))
    assert descriptor is not None and descriptor.kind == "waive-known-wrong:ring-pad"
    assert descriptor.measured["probe"] == "ring_pad:minimax_h3"
    assert descriptor.memo_context["topology"] == "ring4"
    assert descriptor.memo_context["world"] == "4"

    cards = consent_rig.pending.pending_cards()
    assert [card["kind"] for card in cards] == ["waive-known-wrong:ring-pad"]
    status, payload = consent_rig.routes.handle_action(
        {"action": "accept", "key": cards[0]["key"], "id": cards[0]["id"]})
    assert status == 200 and payload["ok"], payload

    h3_ring_pad.preflight(model, request, _latent(), topo, 4, 1.0)   # passes now
    dispatch = {"_dgxm_render_id": "r-188"}
    assert consent_rig.waiver.stamp_request(dispatch, model, topo, 4, "r-188") == 1
    assert list(dispatch[accuracy_waiver.REQUEST_KEY]) == ["waive-known-wrong:ring-pad"]
    # The grant is scoped to the layout and world it was given for.
    with pytest.raises(UnsupportedModelError):
        h3_ring_pad.preflight(_model(2, "ring2", loras=model.loras), _request(_text_for(1, 2)),
                              _latent(), topology_from_preset("ring2", 2), 2, 1.0)


def test_the_headless_waiver_reaches_the_driver_too(consent_rig, monkeypatch):
    monkeypatch.setenv("DGXM_WAIVE_KNOWN_WRONG", "ring_pad")
    topo = topology_from_preset("ring4", 4)
    h3_ring_pad.preflight(_model(4, "ring4"), _request(_text_for(1, 4)), _latent(), topo, 4, 1.0)
    assert consent_rig.pending.pending_cards() == []


def test_the_waiver_lookup_is_stamp_requests(monkeypatch):
    """A live grant of the guard's kind lets the render through; a running
    ceremony carries no waiver, and a store fault grants none."""
    import sys

    from dgx_monarch import accuracy_waiver
    from dgx_monarch.nodes import consent_rescue, consent_waiver

    kind = accuracy_waiver.KNOWN_WRONG_GUARDS[h3_ring_pad.GUARD]
    monkeypatch.setitem(sys.modules, "folder_paths", types.SimpleNamespace(
        get_full_path=lambda _folder, name: f"/models/{name}"))
    monkeypatch.setattr(consent_rescue, "combo_identity",
                        lambda name, options, path, loras: (f"combo:{name}:{len(loras)}", None))
    asked = []

    def live_grants(*, path, combo_key, topo, world):
        asked.append((path, combo_key, topo.describe(), world))
        return {kind: {"id": "grant"}}

    monkeypatch.setattr(consent_waiver, "live_grants", live_grants)
    monkeypatch.setattr(consent_waiver, "_gate_ceremony_active", lambda: False)
    topo = topology_from_preset("ring4", 4)
    model = _model(4, "ring4", loras=({"name": "a", "strength": 1.0},))
    assert h3_ring_pad.guard_waived_for(model, topo, 4) is True
    assert asked == [("/models/h3.safetensors", "combo:h3.safetensors:1", "ring4", 4)]

    monkeypatch.setattr(consent_waiver, "_gate_ceremony_active", lambda: True)
    assert h3_ring_pad.guard_waived_for(model, topo, 4) is False

    def broken(**_kwargs):
        raise OSError("the store is unreadable")

    monkeypatch.setattr(consent_waiver, "_gate_ceremony_active", lambda: False)
    monkeypatch.setattr(consent_waiver, "live_grants", broken)
    assert h3_ring_pad.guard_waived_for(model, topo, 4) is False


def test_the_driver_raises_the_workers_guard():
    from dgx_monarch.adapters.minimax_h3 import H3_RING_PAD_GUARD

    assert h3_ring_pad.GUARD == H3_RING_PAD_GUARD


def test_the_fitting_presets_tile_the_world_on_pure_ulysses():
    assert h3_ring_pad.fitting_presets(2) == ["uly2"]
    assert h3_ring_pad.fitting_presets(4) == ["uly4"]
    assert h3_ring_pad.fitting_presets(8) == ["uly8"]
    for world in (3, 5, 6, 7):
        assert h3_ring_pad.fitting_presets(world) == []
    for world in (2, 4, 8):
        for name in h3_ring_pad.fitting_presets(world):
            topo = topology_from_preset(name, world)
            assert (topo.ring, topo.cfg, topo.dp) == (1, 1, 1) and 56 % topo.ulysses == 0
            assert name in PRESETS


# --- where it runs ---------------------------------------------------------


def _forbid(monkeypatch, module, *names):
    def forbidden(name):
        def call(*_args, **_kwargs):
            pytest.fail(f"{name} ran before the ring-pad refusal")
        return call

    for name in names:
        monkeypatch.setattr(module, name, forbidden(name))


def _auto_resolution(monkeypatch, module=common):
    def resolve(model, latent, cfg_value, world=None):
        world = model.mesh.world if world is None else world
        decision = choose_auto_topology(H3, "int8", 1.03, world, cfg_value=cfg_value)
        return decision.topology, decision.sage, decision.reason

    monkeypatch.setattr(module, "_resolve_topology_for_latent", resolve)


def _run_render_until_the_gate(monkeypatch):
    monkeypatch.setattr(render_preflight, "activation_footprint_preflight_for_request",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(common, "_bind_packed_render_model",
                        lambda model, _latent, _cfg, **_kw: (model, None))
    _auto_resolution(monkeypatch)


def test_run_render_refuses_before_the_gate_setup_and_the_load(monkeypatch, h3_checkpoint):
    """World 3 on auto: pure ring3, and a total of 1 modulo 3. The refusal
    lands at the gate's entry, before it reads its context or runs a proof
    render, and before submit_render, which owns setup and the dispatch that
    loads the model under auto."""
    _run_render_until_the_gate(monkeypatch)
    _forbid(monkeypatch, common, "_auto_gate_context", "_quarantine_unproven_paths",
            "submit_render")
    with pytest.raises(UnsupportedModelError) as exc_info:
        common.run_render(_model(3), _request(_text_for(1, 3)), _latent(), 1.0, 8)
    assert _refusal(exc_info).guard == "ring_pad:minimax_h3"


def test_run_render_lets_the_four_spark_render_through(monkeypatch, h3_checkpoint):
    """World 4 on auto resolves uly4, so the check passes, the gate reads its
    context and the render goes on to submit."""
    _run_render_until_the_gate(monkeypatch)
    reached = []

    class SubmitReached(Exception):
        pass

    def gate_context(*_args, **_kwargs):
        reached.append("gate")
        return None  # no ceremony to run

    def submit(*_args, **_kwargs):
        reached.append("submit")
        raise SubmitReached

    monkeypatch.setattr(common, "_auto_gate_context", gate_context)
    monkeypatch.setattr(common, "_quarantine_unproven_paths", lambda *_args, **_kw: None)
    monkeypatch.setattr(common, "submit_render", submit)
    with pytest.raises(SubmitReached):
        common.run_render(_model(4), _request(_text_for(1, 4)), _latent(), 1.0, 8)
    assert reached == ["gate", "submit"]
    assert h3_checkpoint.waived["asked"] == []


def test_the_gate_entry_checks_before_any_ceremony():
    """run_render always enters the gate (it decides there whether to prove),
    so the check sits at that entry, before the gate reads its context."""
    import inspect

    from dgx_monarch.nodes import auto_gate

    source = inspect.getsource(auto_gate.maybe_auto_gate)
    assert (source.index("h3_ring_pad.preflight_for_render(")
            < source.index("= _auto_gate_context("))
    run_render = inspect.getsource(common.run_render)
    assert run_render.index("_maybe_auto_gate(") < run_render.index("submit_render(")


def test_the_pipeline_checks_before_its_gate():
    import inspect

    from dgx_monarch.nodes import pipeline

    source = inspect.getsource(pipeline.RenderPipeline._push_bound)
    assert source.index("h3_ring_pad.preflight_for_render(") < source.index("_maybe_auto_gate(")


def test_a_resolution_that_raises_is_left_to_the_render(monkeypatch, h3_checkpoint):
    def resolve(*_args, **_kwargs):
        raise FileNotFoundError("the render's own resolution raises this")

    h3_ring_pad.preflight_for_render(_model(3), _request(_text_for(1, 3)), _latent(), 1.0,
                                     resolve_topology=resolve)


def test_preflight_for_render_reads_the_mesh_world(monkeypatch, h3_checkpoint):
    seen = []

    def resolve(model, latent, cfg_value, world):
        seen.append(world)
        return Topology(ring=world, world=world), False, ""

    with pytest.raises(UnsupportedModelError):
        h3_ring_pad.preflight_for_render(_model(5), _request(_text_for(1, 5)), _latent(), 1.0,
                                         resolve_topology=resolve)
    assert seen == [5]


# --- Wan SCAIL: the auto fold its adapter always refuses ----------------------


@pytest.mark.parametrize(("world", "layout", "names"), [
    (4, "uly2+ring2", "uly4"), (3, "ring3", None), (8, "uly2+ring4", "uly8"),
])
def test_auto_refuses_scails_ring_layouts_at_once(world, layout, names):
    """SCAIL's adapter refuses ring above a sequence-parallel world of two at
    injection, whatever the sequence, so the driver refuses first (#136)."""
    topo = choose_auto_topology("wan_scail", "bf16", 0.46, world, cfg_value=5.0).topology
    assert topo.describe() == layout
    with pytest.raises(UnsupportedModelError) as exc_info:
        h3_ring_pad.refuse_scail_ring(_model(world), topo, world)
    tag = _refusal(exc_info)
    assert tag.refusal_class is RefusalClass.PHYSICS and tag.guard is None
    text = str(exc_info.value)
    assert f"`auto` resolved {layout} at world {world}" in text
    assert "before the first-use check, setup and the load" in text
    assert "#136" in text
    if names:
        # No report shows SCAIL's uly4 or uly8 latent identical to a validated layout's.
        assert (f"Set the Init node's topology to {names} (exact by construction; no run "
                "of it is on record),") in text
    else:
        assert f"No Ulysses preset fills world {world}" in text


@pytest.mark.parametrize(("world", "preset"), [(2, "ring2"), (2, "uly2"), (4, "uly4"), (8, "uly8")])
def test_scail_layouts_its_adapter_runs_pass(world, preset):
    h3_ring_pad.refuse_scail_ring(_model(world, preset), topology_from_preset(preset, world), world)


def test_run_render_refuses_scails_auto_fold_before_the_gate(monkeypatch):
    monkeypatch.setattr(h3_ring_pad, "_family_and_path",
                        lambda _model: ("wan_scail", "/models/scail.safetensors"))
    monkeypatch.setattr(render_preflight, "activation_footprint_preflight_for_request",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(common, "_bind_packed_render_model",
                        lambda model, _latent, _cfg, **_kw: (model, None))

    def resolve(model, latent, cfg_value, world=None):
        decision = choose_auto_topology("wan_scail", "fp8", 0.46, world, cfg_value=cfg_value)
        return decision.topology, decision.sage, decision.reason

    monkeypatch.setattr(common, "_resolve_topology_for_latent", resolve)
    _forbid(monkeypatch, common, "_auto_gate_context", "_quarantine_unproven_paths",
            "submit_render")
    latent = {"samples": torch.zeros(1, 16, 21, 64, 112)}
    with pytest.raises(UnsupportedModelError, match="wan_scail: `auto` resolved uly2\\+ring2"):
        common.run_render(_model(4), {"kind": "ksampler", "cfg": 5.0}, latent, 5.0, 8)


# --- the Identity Gate node, and the head count at injection ------------------


def _gate_node_run(monkeypatch, model, rows):
    """Queue the Identity Gate node on ring4 at world 4 and record whether its
    ceremony, the first proof render, was reached."""
    from dgx_monarch.nodes import expander_prompt, requeue_price
    from dgx_monarch.nodes import gate as gate_mod
    from dgx_monarch.nodes.gate_node import DGXMonarchIdentityGate

    reached = []

    def ceremony(_model, _request, latent, *_args, **_kwargs):
        reached.append("ceremony")
        return {"latent": latent, "verdict": "PASS"}

    monkeypatch.setattr(gate_mod, "run_identity_ceremony", ceremony)
    monkeypatch.setattr(gate_mod, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(expander_prompt, "refuse_dead_expander", lambda *_a, **_k: None)
    monkeypatch.setattr(requeue_price, "price_cached_loader", lambda *_a, **_k: None)
    monkeypatch.setattr(common, "_resolve_topology_for_latent",
                        lambda *_a, **_k: (topology_from_preset("ring4", 4), False, ""))
    DGXMonarchIdentityGate().gate(model, [_entry(rows)], [_entry(rows)], _latent(),
                                  0, 2, 1.0, "euler", "simple")
    return reached


def test_the_identity_gate_node_refuses_a_padded_ring_before_its_proof_render(
        monkeypatch, h3_checkpoint):
    with pytest.raises(UnsupportedModelError) as exc_info:
        _gate_node_run(monkeypatch, _model(4, "ring4"), _text_for(1, 4))
    assert _refusal(exc_info).guard == "ring_pad:minimax_h3"
    assert h3_checkpoint.waived["asked"] == [("ring4", 4, "ring_pad:minimax_h3")]


def test_the_identity_gate_node_passes_a_live_waiver_and_an_unpadded_total(
        monkeypatch, h3_checkpoint):
    assert _gate_node_run(monkeypatch, _model(4, "ring4"), _text_for(0, 4)) == ["ceremony"]
    h3_checkpoint.waived["on"] = True
    assert _gate_node_run(monkeypatch, _model(4, "ring4"), _text_for(1, 4)) == ["ceremony"]


def test_injection_refuses_a_head_count_the_ulysses_degree_does_not_divide(monkeypatch):
    """The fold names 56 heads from the table; the adapter reads the file's own
    count before it binds anything and refuses in words, before the all-to-all."""
    from dgx_monarch.adapters import minimax_h3

    class Reached(Exception):
        pass

    def head_form(_final_layer):
        raise Reached

    monkeypatch.setattr(minimax_h3, "head_form", head_form)
    from dgx_monarch.adapters import flux3_packing

    monkeypatch.setattr(flux3_packing, "ulysses_degree", lambda: 4)
    ctx = types.SimpleNamespace(usp_attention=None, topology_sp=4)

    def dit(heads):
        block = types.SimpleNamespace(attn=types.SimpleNamespace(heads=heads))
        return types.SimpleNamespace(blocks=[block], final_layer=None)

    with pytest.raises(UnsupportedModelError) as exc_info:
        minimax_h3.MiniMaxH3Adapter().inject_usp(dit(42), ctx)
    tag = _refusal(exc_info)
    assert tag.refusal_class is RefusalClass.PHYSICS
    assert "42 attention heads, which the Ulysses degree 4 does not divide" in str(exc_info.value)
    with pytest.raises(Reached):  # 56 heads pass the check and go on to the head's form
        minimax_h3.MiniMaxH3Adapter().inject_usp(dit(56), ctx)
    with pytest.raises(Reached):  # no readable count: the check stands aside
        minimax_h3.MiniMaxH3Adapter().inject_usp(types.SimpleNamespace(final_layer=None), ctx)
