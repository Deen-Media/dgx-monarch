"""nodes/loader_preflight: the loader-site driver footprint preflight and the
slab rescue offer.

CPU only, no torch, no CUDA, no comfy beyond tiny stubs. Artifact sizes are
byte exact from the staged set, and MemAvailable values come from the recorded
2026-08-04 series (docs/VALIDATION.md, driver-side footprint window), so three
of that day's four reference-to-video legs replay here on paper: the one that
froze the host and two that completed (the pruned fp8 leg is not replayed).
This site has to separate them as the render site does.

The site this file covers is the one the render site cannot see.
Under an explicit topology preset the loader node loads eagerly, before the
driver has built its text encoder, its VAEs or any reference encode, so the
weight load and the whole pending driver stack are charged together here and
nowhere else.
"""
from __future__ import annotations

import ast
import logging
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

from dgx_monarch import (
    capacity_fit,
    consent_pending,
    consent_store,
    driver_footprint,
    h3_activation,
    mesh_safety,
)
from dgx_monarch.nodes import loader_graph, loader_preflight

REPO = Path(__file__).resolve().parents[1]

_GIB = 2 ** 30

# Staged artifact sizes, byte exact (`ls --block-size=1`). 5.2 GB and 0.6 GB
# are the decimal readings of the two VAEs; in GiB the pair is 5.41.
DIT_REF2VA_BF16 = 66_280_487_368          # 61.73 GiB, the leg that froze the box
DIT_FL2VA_BF16 = 66_280_487_368           # same class of artifact, keyframes only
DIT_REF2VA_INT8 = 34_038_894_550          # 31.70 GiB, ran clean with 0.7 GiB left
DIT_REF2VA_INT8_PRUNED = 20_970_379_616   # 19.53 GiB, ran clean
TE_NVFP4 = 15_687_142_551                 # 14.61 GiB on disk, 16.80 charged
VAE_VIDEO = 5_207_808_496                 # 4.85 GiB
VAE_AUDIO = 605_254_808                   # 0.56 GiB

# MemAvailable from the recorded series, at the sample before each eager load.
# 0722 in the names below is the leg that froze the host on 2026-08-04.
MEM_0722_PEAK = int(114.5 * _GIB)
MEM_LEG5 = int(114.7 * _GIB)
MEM_LEG11 = int(114.3 * _GIB)

REF2VA_BF16 = "minimax_h3_ref2va_bf16.safetensors"
REF2VA_INT8 = "minimax_h3_ref2va_int8_convrot.safetensors"
REF2VA_INT8_PRUNED = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
FL2VA_BF16 = "minimax_h3_fl2va_bf16.safetensors"
TE_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_VIDEO_NAME = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO_NAME = "minimax_h3_audio_vae_fp32.safetensors"

# Pretend sizes, real (tiny) files: the estimator reads sizes through
# driver_footprint.file_size_bytes, which is stubbed, while the artifact digest
# the consent card is joined by is computed from the file itself.
PRETEND_SIZE = {
    ("diffusion_models", REF2VA_BF16): DIT_REF2VA_BF16,
    ("diffusion_models", FL2VA_BF16): DIT_FL2VA_BF16,
    ("diffusion_models", REF2VA_INT8): DIT_REF2VA_INT8,
    ("diffusion_models", REF2VA_INT8_PRUNED): DIT_REF2VA_INT8_PRUNED,
    ("text_encoders", TE_NAME): TE_NVFP4,
    ("vae", VAE_VIDEO_NAME): VAE_VIDEO,
    ("vae", VAE_AUDIO_NAME): VAE_AUDIO,
}

# The driver stack the reference-video graph is charged. At the 5 GiB driver
# floor (since 2026-09-04) the freeze leg and the binding clean leg bound it to
# (47.8, 77.6] GiB in the resident window: at or below 47.8 the artifact that
# froze the host clears that window, and above 77.6 the artifact that ran clean
# with 0.7 GiB spare is refused there.
STACK_WITH_REF_VIDEO = int(TE_NVFP4 * 1.15) + VAE_VIDEO + VAE_AUDIO + 40 * _GIB


def _graph(*, dit: str, refs: str = "none", width: int = 512,
           height: int = 320, image_file: str = "") -> dict:
    """Nodes 1 to 6, 8 and 9 follow the ids of
    tests/fixtures/workflows/minimax_h3_t2va_uly2_api.json, in comfy's prompt
    format and with no negative prompt; `dit`, the canvas and `refs` set the
    rest, and the guide nodes reuse ids 7, 10 and 11."""
    prompt = {
        "1": {"class_type": "DGXMonarchInit",
              "inputs": {"topology": "uly2", "mode": "cluster"}},
        "2": {"class_type": "DGXMonarchUNETLoader",
              "inputs": {"mesh": ["1", 0], "unet_name": dit,
                         "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TE_NAME, "type": "minimax"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VIDEO_NAME}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUDIO_NAME}},
        "8": {"class_type": "EmptyMiniMaxH3LatentAV",
              "inputs": {"width": width, "height": height, "length": 5}},
        "9": {"class_type": "DGXMonarchKSampler",
              "inputs": {"model": ["2", 0], "positive": ["6", 0],
                         "latent_image": ["8", 0], "seed": 42}},
    }
    if refs == "video":
        # Autogrow inputs serialize as dicts keyed by index strings, and
        # `ref_image_size` is a widget on the same node.
        prompt["6"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0],
            "prompt": "a lighthouse", "width": width, "height": height, "length": 5,
            "ref_image_size": "match",
            "ref_images": {"ref_image_1": ["20", 0]},
            "ref_videos": {"ref_video_1": ["21", 0]}}}
    elif refs == "keyframes":
        prompt["6"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0],
            "prompt": "a lighthouse", "width": width, "height": height, "length": 5,
            "first_frame": ["20", 0], "last_frame": ["21", 0]}}
    else:
        prompt["6"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0],
            "prompt": "a lighthouse", "width": width, "height": height, "length": 5}}
    if refs.startswith("guide"):
        # Two nodes that carry an `image` input and encode nothing: the loader
        # image is a widget string, the scaler's is a link. Neither is a guide.
        prompt["7"] = {"class_type": "LoadImage",
                       "inputs": {"image": image_file or "example.png"}}
        prompt["11"] = {"class_type": "ImageScale",
                        "inputs": {"image": ["7", 0], "width": width}}
        guide = {"positive": ["6", 0], "latent": ["8", 0], "frame_idx": 62}
        if refs != "guide_audio":
            # "guide_file" wires the loader straight in, the shape the shipped
            # example uses and the only one whose length a file can prove.
            source = "7" if refs == "guide_file" else "11"
            guide.update({"vae": ["4", 0], "image": [source, 0]})
        if refs in ("guide_audio", "guide_both"):
            guide.update({"audio_vae": ["5", 0], "audio": ["12", 0]})
        prompt["10"] = {"class_type": "MiniMaxH3AddGuide", "inputs": guide}
        prompt["9"]["inputs"]["positive"] = ["10", 0]
    return prompt


class _Handle:
    """Enough MeshHandle for the config-only reads this site makes."""

    def __init__(self, config_worker_args: dict | None = None,
                 gpus_per_host: int = 1):
        self.owns_hosts = False          # rank 0 is co-resident with the driver
        self.gpus_per_host = gpus_per_host   # the pair's shape: one rank here
        self.config = types.SimpleNamespace(hosts=(),
                                            worker_args=config_worker_args or {})

    def effective_worker_args(self, worker_args: dict | None = None) -> dict:
        return {**dict(self.config.worker_args), **dict(worker_args or {})}


class _Mesh:
    def __init__(self, worker_args: dict | None = None,
                 config_worker_args: dict | None = None,
                 topology_preset: str = "uly2", world: int = 2,
                 gpus_per_host: int = 1):
        self.handle = _Handle(config_worker_args, gpus_per_host)
        self.topology_preset = topology_preset
        self.world = world          # MeshSpec.world, the handle's own world
        self.worker_args = dict(worker_args or {})


@pytest.fixture(autouse=True)
def _rig(monkeypatch, tmp_path):
    """A UMA box with a known MemAvailable, a stub folder_paths over real
    (tiny) artifact files, an H3 header sniff, and a consent subsystem whose
    store and pending registry are private to the test."""
    loader_preflight.reset_memos()
    consent_pending.clear_all()
    monkeypatch.delenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, raising=False)
    monkeypatch.delenv(h3_activation.GUIDE_FRAMES_ENV, raising=False)
    monkeypatch.delenv(consent_pending.AUTO_RESCUE_ENV, raising=False)
    for spec in consent_pending.KIND_SPECS.values():
        monkeypatch.delenv(spec.env_var, raising=False)
    monkeypatch.setattr(consent_store, "MEMO_PATH", str(tmp_path / "consent_memo.json"))
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: MEM_0722_PEAK)

    paths: dict[str, int] = {}
    for (folder, name), size in PRETEND_SIZE.items():
        target = tmp_path / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(name.encode())
        paths[str(target)] = size
    monkeypatch.setattr(driver_footprint, "file_size_bytes",
                        lambda path: paths.get(str(path), 0))

    fp = types.ModuleType("folder_paths")
    fp.get_full_path = lambda folder, name: (  # type: ignore[attr-defined]
        str(tmp_path / folder / name)
        if (folder, name) in PRETEND_SIZE else None)
    ledger_dir = tmp_path / "output"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    fp.get_output_directory = lambda: str(ledger_dir)  # type: ignore[attr-defined]
    inputs = tmp_path / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    fp.get_annotated_filepath = lambda name: str(inputs / name)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fp)

    from dgx_monarch.adapters import detect

    monkeypatch.setattr(detect, "sniff_checkpoint",
                        lambda path: ("minimax_h3", "bf16"))
    yield
    loader_preflight.reset_memos()
    consent_pending.clear_all()


def _refuse(mesh, unet_name, options=None, prompt=None):
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as excinfo:
        loader_preflight.preflight_loader_footprint(
            mesh, unet_name, options or {}, prompt)
    return str(excinfo.value)


def test_graph_walk_prices_the_text_encoder_the_vaes_and_a_reference_video():
    terms = loader_graph.driver_stack_terms(
        _graph(dit=REF2VA_BF16, refs="video"))
    assert terms.resolved
    assert terms.te_bytes == int(TE_NVFP4 * 1.15)
    assert terms.vae_bytes == VAE_VIDEO + VAE_AUDIO
    assert terms.ref_bytes == 40 * _GIB
    assert terms.video_refs == 1 and terms.image_refs == 1
    assert terms.te_artifacts == (TE_NAME,)
    assert set(terms.vae_artifacts) == {VAE_VIDEO_NAME, VAE_AUDIO_NAME}
    assert terms.total == STACK_WITH_REF_VIDEO


def test_a_widget_that_shares_a_reference_prefix_is_not_a_reference():
    """`ref_image_size` is a COMBO on the same node. Only linked inputs count."""
    terms = loader_graph.driver_stack_terms(
        _graph(dit=REF2VA_BF16, refs="video"))
    assert terms.image_refs == 1        # the Autogrow ref_image_1 link, not the combo


def test_keyframe_only_graphs_charge_no_reference_encode():
    """The 2026-08-04 measurement is a reference video shape. Charging
    keyframes from it would be an invented number in the refusing direction."""
    terms = loader_graph.driver_stack_terms(
        _graph(dit=FL2VA_BF16, refs="keyframes"))
    assert terms.image_refs == 2 and terms.video_refs == 0
    assert terms.ref_bytes == 0
    assert terms.total == int(TE_NVFP4 * 1.15) + VAE_VIDEO + VAE_AUDIO


@pytest.mark.parametrize("prompt", [None, {}, "not a graph", 42,
                                    {"1": "not a node"},
                                    {"1": {"inputs": None}}])
def test_an_unreadable_graph_charges_nothing_and_never_raises(prompt):
    terms = loader_graph.driver_stack_terms(prompt)
    assert terms.total == 0
    assert terms.resolved is (isinstance(prompt, dict) and bool(prompt))


def test_an_unresolvable_artifact_name_is_omitted_not_guessed():
    graph = _graph(dit=REF2VA_BF16)
    graph["3"]["inputs"]["clip_name"] = "a_text_encoder_that_is_not_staged.safetensors"
    terms = loader_graph.driver_stack_terms(graph)
    assert terms.te_bytes == 0 and terms.te_artifacts == ()
    assert terms.vae_bytes == VAE_VIDEO + VAE_AUDIO


# The anchored guide encode. Add Guide's inputs are named `image` and `audio`,
# which match no reference prefix, so the guide takes its own charge: without
# it a 1344x768 guide graph prices like a text-to-video graph with no media,
# while a five-frame encode at that canvas alone needs more than 94 GiB
# (docs/VALIDATION.md, 2026-08-14).

H3 = "minimax_h3"


def _terms(**kwargs):
    return loader_graph.driver_stack_terms(_graph(**kwargs), family=H3,
                                           sp_degree=2)


def test_an_anchored_guide_is_charged_the_encode_its_canvas_costs():
    terms = _terms(dit=REF2VA_INT8_PRUNED, refs="guide", width=1344, height=768)
    assert terms.guide_refs == 1
    assert terms.guide_bytes == 94 * _GIB
    assert "priced as a clip" in terms.guide_note


@pytest.mark.parametrize("width,height,charge", [
    (448, 256, 17),        # measured, completed in 1.8 s
    (512, 320, 67),        # between two points: rounds up to the next one
    (672, 384, 67),        # measured, completed in 4.7 s
    (1344, 768, 94),       # censored: the encode never completed
])
def test_the_charge_rounds_a_canvas_up_to_the_next_measured_probe_point(
        monkeypatch, width, height, charge):
    """The canvas ladder, at the five frames every one of its points measured."""
    monkeypatch.setenv(h3_activation.GUIDE_FRAMES_ENV, "5")
    terms = _terms(dit=REF2VA_INT8_PRUNED, refs="guide", width=width, height=height)
    assert terms.guide_bytes == charge * _GIB


@pytest.mark.parametrize("declared,charge", [
    ("5", 17),             # measured: 16.2 GiB, encode alone
    ("22", 34),            # measured: 33.4 GiB
    ("39", 48),            # measured: 47.5 GiB
    ("56", 48),            # past the sweep: the last measured length, undercounted
    ("", 48),              # undeclared: a link can carry any legal length
])
def test_clip_length_is_priced_where_a_run_swept_it(monkeypatch, declared, charge):
    """Length costs about a gigabyte a frame at 448x256, so the five-frame floor
    would undercharge a legal 39-frame clip by about 31 GiB."""
    if declared:
        monkeypatch.setenv(h3_activation.GUIDE_FRAMES_ENV, declared)
    terms = _terms(dit=REF2VA_INT8_PRUNED, refs="guide", width=448, height=256)
    assert terms.guide_bytes == charge * _GIB


def _write_image(tmp_path, name: str, frames: int):
    """A real file with a real frame count, written where the loader looks."""
    pytest.importorskip("PIL")
    from PIL import Image
    pages = [Image.new("RGB", (32, 32), (index * 40 % 255, 0, 0))
             for index in range(max(frames, 1))]
    path = tmp_path / "input" / name
    pages[0].save(path, save_all=frames > 1, append_images=pages[1:])
    return name


def test_a_file_that_proves_one_frame_renders_at_config_b_untouched(tmp_path):
    """The shipped guide example: one still image at 1344x768, whose encode
    measured 13.6 GiB on 2026-08-14. It must render with no environment variable
    set, and it does because the file itself proves the length the graph cannot."""
    name = _write_image(tmp_path, "still.png", 1)
    graph = _graph(dit=REF2VA_INT8_PRUNED, refs="guide_file", width=1344,
                   height=768, image_file=name)
    terms = loader_graph.driver_stack_terms(graph, family=H3, sp_degree=2)
    assert terms.guide_bytes == 14 * _GIB
    assert "1 frame(s) proved from the named file" in terms.guide_note
    loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_INT8_PRUNED, {}, graph)


@pytest.mark.parametrize("name,frames", [
    ("clip.png", 6),      # APNG does not hide behind an image extension
    ("clip.webp", 6),
    ("clip.gif", 6),
])
def test_an_animated_file_is_proved_a_clip_and_refused(tmp_path, name, frames):
    written = _write_image(tmp_path, name, frames)
    graph = _graph(dit=REF2VA_INT8_PRUNED, refs="guide_file", width=1344,
                   height=768, image_file=written)
    terms = loader_graph.driver_stack_terms(graph, family=H3, sp_degree=2)
    assert terms.guide_bytes == 94 * _GIB
    assert f"{frames} frame(s) proved" in terms.guide_note


@pytest.mark.parametrize("name,content", [
    ("garbage.png", b"not an image at all"),
    ("", None),                                   # nothing named
])
def test_evidence_that_does_not_open_keeps_the_worst_case(tmp_path, name, content):
    """A video through this input does not open as an image either, which is
    the fail-safe direction: unreadable costs the clip charge."""
    if content is not None:
        (tmp_path / "input" / name).write_bytes(content)
    graph = _graph(dit=REF2VA_INT8_PRUNED, refs="guide_file", width=1344,
                   height=768, image_file=name or "missing.png")
    terms = loader_graph.driver_stack_terms(graph, family=H3, sp_degree=2)
    assert terms.guide_bytes == 94 * _GIB
    assert "priced as a clip" in terms.guide_note


def test_proof_outranks_a_declaration_that_disagrees(tmp_path, monkeypatch):
    """A declaration binds a whole driver process, so a still-declaring driver
    would otherwise price a later clip graph as a still. The file settles it."""
    monkeypatch.setenv(h3_activation.GUIDE_FRAMES_ENV, "1")
    written = _write_image(tmp_path, "sneaky.webp", 6)
    graph = _graph(dit=REF2VA_INT8_PRUNED, refs="guide_file", width=1344,
                   height=768, image_file=written)
    terms = loader_graph.driver_stack_terms(graph, family=H3, sp_degree=2)
    assert terms.guide_bytes == 94 * _GIB
    assert "proved from the named file" in terms.guide_note


def test_a_source_that_names_no_file_still_takes_the_worst_case(tmp_path):
    """A scaler between the loader and the guide hides the file, so nothing is
    proved and the charge stays where it was."""
    _write_image(tmp_path, "still.png", 1)
    graph = _graph(dit=REF2VA_INT8_PRUNED, refs="guide", width=1344,
                   height=768, image_file="still.png")
    terms = loader_graph.driver_stack_terms(graph, family=H3, sp_degree=2)
    assert terms.guide_bytes == 94 * _GIB


def test_a_guide_is_priced_at_the_canvas_it_is_anchored_into():
    """A graph can carry a second, larger canvas. The guide is resized to the
    latent it is wired to, so pricing it off the other one refuses a render
    whose encode fits."""
    graph = _graph(dit=REF2VA_INT8_PRUNED, refs="guide", width=448, height=256)
    graph["30"] = {"class_type": "EmptyMiniMaxH3LatentAV",
                   "inputs": {"width": 1344, "height": 768, "length": 124}}
    terms = loader_graph.driver_stack_terms(graph, family=H3, sp_degree=2)
    assert terms.guide_bytes == 48 * _GIB          # the 448x256 it anchors into
    # An unreadable anchor falls back to the whole-graph worst case.
    graph["10"]["inputs"]["latent"] = "not a link"
    fallback = loader_graph.driver_stack_terms(graph, family=H3, sp_degree=2)
    assert fallback.guide_bytes == 94 * _GIB


def test_the_credit_clause_never_claims_a_surviving_charge_was_credited():
    terms = loader_graph.DriverStackTerms(
        te_bytes=0, vae_bytes=0, guide_bytes=94 * _GIB, activation_bytes=0,
        resolved=True, credited=True, credit_reason="comfy already holds them")
    clause = terms.budget_clause
    assert "guide encode" in clause and "driver stack" not in clause
    assert "comfy already holds them" in clause


def test_the_length_sweep_does_not_reach_canvases_it_never_ran_at():
    """One canvas carries the frame sweep. Above it the five-frame charge
    stands, and it already refuses this hardware."""
    terms = _terms(dit=REF2VA_INT8_PRUNED, refs="guide", width=672, height=384)
    assert terms.guide_bytes == 67 * _GIB


def test_above_the_largest_measured_canvas_the_charge_saturates():
    """A bigger canvas cannot cost less, but no run says how much more. The
    charge stops at the last number a run produced rather than print an
    extrapolation as evidence; it refuses this hardware either way."""
    terms = _terms(dit=REF2VA_INT8_PRUNED, refs="guide", width=2688, height=1536)
    assert terms.guide_bytes == 94 * _GIB


def test_the_clip_geometry_the_ladder_runs_is_admitted():
    """448x256 encodes in 16.2 GiB at five frames and 47.5 at thirty-nine, and
    both fit beside the pruned DiT and the text encoder. The wall must admit
    this graph even when it has to assume the longest measured clip."""
    terms = _terms(dit=REF2VA_INT8_PRUNED, refs="guide", width=448, height=256)
    assert terms.guide_bytes == 48 * _GIB
    loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_INT8_PRUNED, {},
        _graph(dit=REF2VA_INT8_PRUNED, refs="guide", width=448, height=256))


def test_an_image_input_on_a_node_that_encodes_nothing_is_not_a_guide():
    """`image` names an input on most nodes that touch pixels. Pricing them as
    encodes would refuse whole graphs that never build one."""
    terms = _terms(dit=REF2VA_INT8_PRUNED, width=1344, height=768)
    assert terms.guide_refs == 0 and terms.guide_bytes == 0
    assert "no guide image linked" in terms.guide_note


def test_a_guide_node_in_another_familys_graph_charges_nothing():
    """Add Guide raises on any latent that is not an H3 AV pair before it
    reaches the encode, so only an H3 graph can spend these bytes."""
    terms = loader_graph.driver_stack_terms(
        _graph(dit=REF2VA_INT8_PRUNED, refs="guide", width=1344, height=768),
        family="ltx", sp_degree=2)
    assert terms.guide_bytes == 0


def test_an_audio_only_guide_charges_no_video_encode():
    terms = _terms(dit=REF2VA_INT8_PRUNED, refs="guide_audio", width=1344, height=768)
    assert terms.guide_refs == 0 and terms.guide_bytes == 0


def test_a_declared_still_is_priced_from_the_measured_still(monkeypatch):
    """The proven-good render: one image anchored at 1344x768, which the
    acceptance ladder ran repeatedly at 60 to 62 GiB steady."""
    monkeypatch.setenv(h3_activation.GUIDE_FRAMES_ENV, "1")
    terms = _terms(dit=REF2VA_INT8_PRUNED, refs="guide", width=1344, height=768)
    assert terms.guide_bytes == 14 * _GIB
    assert "1 frame(s) declared" in terms.guide_note
    loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_INT8_PRUNED, {},
        _graph(dit=REF2VA_INT8_PRUNED, refs="guide", width=1344, height=768))


@pytest.mark.parametrize("declared", ["5", "39", "lots", "", "-2"])
def test_no_declaration_buys_a_clip_this_box_cannot_encode(monkeypatch, declared):
    # Config B is above the frame sweep, so every clip length takes the
    # censored canvas charge.
    """A declaration prices a term, it does not waive one."""
    monkeypatch.setenv(h3_activation.GUIDE_FRAMES_ENV, declared)
    terms = _terms(dit=REF2VA_INT8_PRUNED, refs="guide", width=1344, height=768)
    assert terms.guide_bytes == 94 * _GIB


def test_the_config_b_guide_clip_the_old_wall_admitted_is_refused_now():
    message = _refuse(_Mesh(), REF2VA_INT8_PRUNED,
                      prompt=_graph(dit=REF2VA_INT8_PRUNED, refs="guide",
                                    width=1344, height=768))
    assert "94.0 GiB" in message                       # the guide encode term
    assert "guide encode" in message
    assert h3_activation.GUIDE_FRAMES_ENV in message   # the still is one env away
    assert "docs/TROUBLESHOOTING.md #82" in message


def test_a_guide_encode_survives_the_residency_credits(monkeypatch):
    """A loaded model proves weights are inside the MemAvailable reading. It
    does not prove Add Guide has run: comfy can encode text, and so load the
    text encoder, well before it reaches the guide. Crediting the encode off
    that evidence would drop the charge from every guide graph whose text
    encoder has loaded."""
    mm = types.ModuleType("comfy.model_management")
    mm.current_loaded_models = [object()]  # type: ignore[attr-defined]
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    terms = loader_graph.credited_terms(
        _terms(dit=REF2VA_INT8_PRUNED, refs="guide", width=1344, height=768))
    assert terms.credited and terms.te_bytes == 0
    assert terms.guide_bytes == 94 * _GIB
    assert terms.total == terms.guide_bytes + terms.activation_bytes


def test_the_driver_preflight_escape_stands_the_guide_charge_down(monkeypatch):
    monkeypatch.setenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, "1")
    monkeypatch.setattr(driver_footprint, "_DISABLE_LOGGED", False)
    terms = _terms(dit=REF2VA_INT8_PRUNED, refs="guide", width=1344, height=768)
    assert terms.guide_bytes == 0


def test_the_0722_reference_shape_refuses_at_the_loader_node():
    message = _refuse(_Mesh(), REF2VA_BF16,
                      prompt=_graph(dit=REF2VA_BF16, refs="video"))
    assert "61.7 GiB" in message                      # the DiT term
    assert "16.8 GiB" in message                      # the text encoder term
    assert "5.4 GiB" in message                       # both VAEs
    assert "40.0 GiB" in message                      # the reference encode
    assert "114.5 GiB" in message and "5.0 GiB" in message
    for option in driver_footprint.DRIVER_STACK_FAMILIES["minimax_h3"].fitting_options:
        assert option in message
    assert driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV in message
    assert "docs/TROUBLESHOOTING.md #52" in message


@pytest.mark.parametrize("unet_name,mem", [
    (REF2VA_INT8, MEM_LEG11),            # 31.70 GiB, the binding clean leg
    (REF2VA_INT8_PRUNED, MEM_LEG5),      # 19.53 GiB
])
def test_the_clean_reference_legs_pass_untouched(monkeypatch, unet_name, mem):
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: mem)
    loader_preflight.preflight_loader_footprint(
        _Mesh(), unet_name, {}, _graph(dit=unet_name, refs="video"))


def test_the_refusal_leads_with_the_numbers_so_a_gate_leg_truncation_keeps_them():
    """Raised inside a first-use Gate leg this message is cut to 200 chars."""
    message = _refuse(_Mesh(), REF2VA_BF16,
                      prompt=_graph(dit=REF2VA_BF16, refs="video"))
    head = message[:200]
    assert "61.7 GiB to place on the driver host" in head
    assert "GiB usable" in head


# The FSDP shard-build price. Under a `*+fsdp` preset each rank builds one shard
# of the checkpoint and the streaming build assigns file-backed rows rather than
# copying the file. The whole file plus a legacy arena is the price of a stock
# load this placement never runs, and it refused the M6 A leg before the shard
# build (docs/VALIDATION.md, the driver's FSDP shard price, 2026-09-09).

FSDP_PRESET = "uly2+fsdp"
SHARD_BF16 = driver_footprint.fsdp_shard_build_bytes(DIT_REF2VA_BF16, 2)


def _fsdp_mesh(worker_args: dict | None = None, world: int = 2,
               gpus_per_host: int = 1) -> _Mesh:
    return _Mesh(worker_args, topology_preset=FSDP_PRESET, world=world,
                 gpus_per_host=gpus_per_host)


def test_an_fsdp_preset_at_world_2_is_priced_at_one_ranks_shard(caplog):
    """The same box, the same graph and the same artifact: the whole file
    refuses, one rank's shard admits and says so in the journal."""
    assert "61.7 GiB to place" in _refuse(
        _Mesh(), REF2VA_BF16, prompt=_graph(dit=REF2VA_BF16, refs="video"))
    with caplog.at_level(logging.INFO,
                         logger="dgx_monarch.nodes.loader_preflight"):
        assert loader_preflight.preflight_loader_footprint(
            _fsdp_mesh(), REF2VA_BF16, {},
            _graph(dit=REF2VA_BF16, refs="video")) is None
    line = "\n".join(record.getMessage() for record in caplog.records)
    assert "FSDP shard build priced" in line
    assert "at world 2, charging 0.53 of the file" in line
    assert f"{SHARD_BF16 / _GIB:.1f} GiB" in line


def test_the_shard_refusal_names_the_fraction_and_the_world(monkeypatch):
    """A head too small for a shard still refuses, and the refusal names the
    shard it priced: the shard changes the price, not the decision rule."""
    monkeypatch.setattr(mesh_safety, "mem_available_bytes",
                        lambda: int(30.0 * _GIB))
    message = _refuse(_fsdp_mesh(), REF2VA_BF16,
                      prompt=_graph(dit=REF2VA_BF16, refs="video"))
    assert f"{SHARD_BF16 / _GIB:.1f} GiB to place on the driver host" in message
    assert "0.50 of the 61.7 GiB file at world 2" in message
    assert "stock host copy 0.0 GiB" in message
    assert "never makes the full copy a legacy load retains" in message
    # Slab is not a lever here, so no rescue card is raised and the refusal
    # stays the unwaivable class C the decision rule already answered.
    from dgx_monarch.refusal import parse_refusal_tag

    assert "Slab residency maps the weights" not in message
    assert consent_pending.pending_cards() == []
    tag = parse_refusal_tag(message)
    assert tag is not None and tag.waivable is False


@pytest.mark.parametrize("kind", ["fp8", "int8", "mxfp8", "nvfp4", "fp32", ""])
def test_only_a_full_precision_file_takes_the_shard_price(monkeypatch, kind):
    """Only bf16 and fp16 files take the streaming build. fp8 and int8 files take
    FSDP2's direct wrap, where each block moves at full size (adapters/fsdp.py,
    the streaming-versus-direct branch); FSDP refuses the block-scaled kinds and
    an all-fp32 file, and a kind this host could not read shards nothing. Every
    one of them keeps the whole-file price."""
    from dgx_monarch.adapters import detect

    monkeypatch.setattr(detect, "sniff_checkpoint",
                        lambda path: ("minimax_h3", kind))
    message = _refuse(_fsdp_mesh(), REF2VA_BF16,
                      prompt=_graph(dit=REF2VA_BF16, refs="video"))
    assert "61.7 GiB to place" in message
    assert "at world" not in message


@pytest.mark.parametrize("slab", [True, False, "auto"])
def test_slab_policy_does_not_move_the_shard_price(slab):
    """`worker_env.slab_mode_effective` is False under FSDP, so a sharded rank
    takes no slab whatever it asked for, and M6 leg H built its shards with
    slab on."""
    assert loader_preflight.preflight_loader_footprint(
        _fsdp_mesh({"slab_weights": slab}), REF2VA_BF16, {},
        _graph(dit=REF2VA_BF16, refs="video")) is None


@pytest.mark.parametrize("world", [1, 0])
def test_an_fsdp_preset_at_world_1_keeps_the_whole_file_price(world):
    """World 1 shards nothing."""
    message = _refuse(_fsdp_mesh(world=world), REF2VA_BF16,
                      prompt=_graph(dit=REF2VA_BF16, refs="video"))
    assert "61.7 GiB to place" in message


@pytest.mark.parametrize("gpus_per_host", [2, 4, 0, None])
def test_a_host_holding_more_than_one_rank_keeps_the_whole_file_price(
        gpus_per_host):
    """The share is one rank's, and `rank0_co_resident` is a flag rather than
    a count. A host running two ranks builds two shards, which is the whole
    file at world 2, so a share applied there would admit what the box
    cannot hold. A `this_host()` mesh is that host: every rank is a child of
    the driver process. An unreadable count keeps the refusing number too."""
    message = _refuse(_fsdp_mesh(gpus_per_host=gpus_per_host), REF2VA_BF16,
                      prompt=_graph(dit=REF2VA_BF16, refs="video"))
    assert "61.7 GiB to place" in message
    assert "at world" not in message and "shard build" not in message


def test_an_auto_preset_is_never_priced_as_a_shard_build():
    """`choose_auto_topology` can add fsdp to a row that does not fit
    resident, and the loader node has not resolved it here. The unresolved
    reading is the one that charges more."""
    message = _refuse(_Mesh(topology_preset="auto"), REF2VA_BF16,
                      prompt=_graph(dit=REF2VA_BF16, refs="video"))
    assert "61.7 GiB to place" in message
    assert "at world" not in message


def test_a_stock_preset_keeps_the_arena_and_the_whole_file():
    """A stock preset's load window charges the file and the host copy it is
    placed from; only a declared shard build moves either term."""
    shape = loader_preflight._Shape(
        profile=driver_footprint.DRIVER_STACK_FAMILIES["minimax_h3"],
        mem_available=MEM_0722_PEAK, weight_bytes=DIT_REF2VA_BF16,
        weights_resident=False, co_resident=True, floor_reserve=5 * _GIB,
        stack_bytes=STACK_WITH_REF_VIDEO)
    transient, settled = loader_preflight._windows(shape, slab=False)
    assert transient.weight_bytes == DIT_REF2VA_BF16
    assert transient.arena_bytes == int(
        capacity_fit.STOCK_PLACEMENT_RATIO * DIT_REF2VA_BF16)
    assert "stock load host copy while it places the weights, slab_weights is off" in transient.notes["arena"]
    assert settled.weight_bytes == DIT_REF2VA_BF16

    sharded, _ = loader_preflight._windows(replace(shape, fsdp_world=2),
                                           slab=False)
    assert sharded.weight_bytes == SHARD_BF16 and sharded.arena_bytes == 0


def test_a_shard_admit_does_not_credit_a_later_whole_file_load():
    """The memo is keyed on the world as well as the name: a rank admitted
    holding half the file has not loaded the whole of it."""
    graph = _graph(dit=REF2VA_BF16, refs="video")
    assert loader_preflight.preflight_loader_footprint(
        _fsdp_mesh(), REF2VA_BF16, {}, graph) is None
    # The whole file plus the host copy it is placed from: the second load is
    # charged in full, where the artifact name alone would have credited it.
    whole_gib = round(
        DIT_REF2VA_BF16 * capacity_fit.STOCK_LOAD_TRANSIENT_FACTOR / _GIB, 1)
    assert f"{whole_gib} GiB to place" in _refuse(_Mesh(), REF2VA_BF16, prompt=graph)


def test_a_rescue_is_offered_and_registered_when_slab_is_what_would_fit():
    """fl2va bf16 under `auto`, which stock-loads this unvouched family: the
    legacy load arena is the difference between refusing and fitting, so the
    card is the remedy.

    The card is built by the real consent registry, so this also proves the
    descriptor this site emits is one the endpoint can audit: `describe()`
    refuses a descriptor with no combo key or no artifact digest.
    """
    spec = consent_pending.KIND_SPECS["rescue-slab"]
    message = _refuse(_Mesh(worker_args={"slab_weights": "auto"}), FL2VA_BF16,
                      prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))
    assert f'"{spec.primary_label}"' in message
    assert "DGX Monarch panel" in message
    assert "verified against the checkpoint" in message
    assert f"{spec.env_var}=1" in message

    cards = consent_pending.pending_cards()
    assert len(cards) == 1
    card = cards[0]
    assert card["kind"] == "rescue-slab" and card["class"] == "C"
    assert card["artifact"] == FL2VA_BF16
    assert card["primary"] == {"label": spec.primary_label, "action": "accept"}
    assert "GiB usable" in str(card["numbers"])


def test_the_card_is_reachable_on_the_DEFAULT_residency(monkeypatch):
    """The acceptance case queues `auto`, which is the Init node's default.

    The rescue is the difference between the legacy load arena and no arena, so
    a site that priced `auto` as if it were already slab would price the rescue
    identically to the refusal and could never offer the card at all.
    """
    message = _refuse(_Mesh(), FL2VA_BF16, prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))
    assert "DGX Monarch panel" in message
    cards = consent_pending.pending_cards()
    assert len(cards) == 1 and cards[0]["kind"] == "rescue-slab"


def test_an_auto_load_of_an_unvouched_family_is_charged_its_legacy_arena():
    """`auto` slab-loads only vouched families and stock-loads the rest, so an
    unvouched family under `auto` spends the stock placement price and must be
    charged for it. Uncharged, the 61.7 GiB artifact would pass a keyframe-only
    graph and then stock-load at 2.1x the file, about 130 GiB on a box with
    114.5 GiB available."""
    message = _refuse(_Mesh(), FL2VA_BF16, prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))
    arena_gib = round(
        capacity_fit.STOCK_PLACEMENT_RATIO * DIT_FL2VA_BF16 / _GIB, 1)
    assert f"stock host copy {arena_gib} GiB" in message
    assert f"at the {loader_preflight.WINDOW_TRANSIENT} window" in message
    # krea2 is vouched, so the same shape under `auto` charges no arena there.
    assert "krea2" in capacity_fit.SLAB_VOUCHED_FAMILIES
    # The signature takes the whole worker-args mapping, so a comfy-managed
    # graph resolves to its own residency here; an absent slab_weights means
    # `auto` and resolves against the vouched set.
    assert loader_preflight._effective_slab({}, "krea2") is True
    assert loader_preflight._effective_slab({}, "minimax_h3") is False


def test_the_settled_window_still_refuses_what_no_residency_can_hold():
    """Slab removes the arena, not the weights: when the graph's own stack
    leaves no room for the checkpoint itself, there is no card to offer."""
    message = _refuse(_Mesh(), REF2VA_BF16, prompt=_graph(dit=REF2VA_BF16, refs="video"))
    assert f"at the {loader_preflight.WINDOW_SETTLED} window" in message
    assert consent_pending.pending_cards() == []


def test_one_failed_load_registers_exactly_one_card_however_often_it_is_queued():
    for _ in range(3):
        _refuse(_Mesh(worker_args={"slab_weights": "auto"}), FL2VA_BF16,
                prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))
    cards = consent_pending.pending_cards()
    assert len(cards) == 1 and cards[0]["occurrences"] == 3


def test_the_refusal_carries_its_class_tag_across_a_wrapped_wire():
    from dgx_monarch.refusal import parse_refusal_tag

    message = _refuse(_Mesh(worker_args={"slab_weights": "auto"}), FL2VA_BF16,
                      prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))
    tag = parse_refusal_tag(str(RuntimeError(message)))
    assert tag is not None
    assert tag.refusal_class.value == "C"
    assert tag.guard == "loader_footprint_preflight" and tag.waivable


def test_tag_shaped_unet_name_keeps_loader_capacity_fail_closed(
        monkeypatch, tmp_path):
    """The generic estimator catch cannot swallow a completed capacity verdict."""
    from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag

    name = "MiniMax-H3/[dgxm:P] ref2va_bf16.safetensors"
    target = tmp_path / "diffusion_models" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"tag-shaped artifact")
    monkeypatch.setitem(PRETEND_SIZE, ("diffusion_models", name), DIT_REF2VA_BF16)
    file_size = driver_footprint.file_size_bytes
    monkeypatch.setattr(
        driver_footprint, "file_size_bytes",
        lambda path: DIT_REF2VA_BF16 if str(path) == str(target) else file_size(path))

    message = _refuse(
        _Mesh(), name, prompt=_graph(dit=name, refs="video"))
    tag = parse_leading_refusal_tag(message)
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY
    assert tag.guard == "loader_footprint_preflight"
    assert "Refuses MiniMax-H3/[dgxm;P] ref2va_bf16.safetensors" in message


def test_the_loader_site_offers_no_rescue_under_comfy_managed_residency():
    """The comfy-managed rung, end to end at the loader site rather than re-derived.

    A capacity rescue is a slab load, and the projection blocker refuses the
    projection under this rung, so a card here would be a button that leads
    nowhere, which class C forbids. The refusal has to arrive with no panel
    action and no rescue sentence, and it has to name the widget, not
    `slab_weights=off`, because the operator never set that.
    """
    from dgx_monarch.refusal import parse_refusal_tag

    managed = {"comfy_managed": True, "slab_weights": False, "lora_low_rss": False}
    # The shape that fits under no residency: the graph's own driver stack
    # leaves too little room for the weights, so removing the arena cannot
    # save it.
    message = _refuse(_Mesh(worker_args=managed), REF2VA_BF16,
                      prompt=_graph(dit=REF2VA_BF16, refs="video"))
    assert "DGX Monarch panel" not in message
    assert consent_pending.pending_cards() == []
    assert "It fits this graph at" not in message   # the _rescue_clause sentence
    tag = parse_refusal_tag(message)
    assert tag is not None and tag.refusal_class.value == "C"
    # The shape an `auto` graph gets a card for is not refused here at all: the
    # rung charges no legacy arena, so it fits. This and the blocker must both
    # hold.
    loader_preflight.reset_memos()
    consent_pending.clear_all()
    assert loader_preflight.preflight_loader_footprint(
        _Mesh(worker_args=managed), FL2VA_BF16, {},
        _graph(dit=FL2VA_BF16, refs="keyframes")) is None
    assert consent_pending.pending_cards() == []


def test_no_card_when_slab_cannot_change_the_verdict():
    """The 2026-08-04 freeze shape: the driver stack leaves too little room for
    the weights, so slab does not fit either. Class C then names the working
    alternatives instead of offering a button that leads nowhere."""
    from dgx_monarch.refusal import parse_refusal_tag

    message = _refuse(_Mesh(worker_args={"slab_weights": "auto"}), REF2VA_BF16,
                      prompt=_graph(dit=REF2VA_BF16, refs="video"))
    assert "DGX Monarch panel" not in message
    assert "use a pruned or quantized DiT artifact" in message
    assert consent_pending.pending_cards() == []
    tag = parse_refusal_tag(message)
    assert tag is not None and tag.waivable is False
    assert tag.guard == "loader_footprint_preflight"


def test_no_card_when_slab_is_already_the_requested_residency():
    message = _refuse(_Mesh(worker_args={"slab_weights": True}), REF2VA_BF16,
                      prompt=_graph(dit=REF2VA_BF16, refs="video"))
    assert "DGX Monarch panel" not in message
    assert consent_pending.pending_cards() == []


def test_a_granted_consent_stands_the_refusal_down(monkeypatch):
    grant = types.SimpleNamespace(consent_id="c-0123456789ab", granted_by="panel")
    monkeypatch.setattr(consent_pending, "resolve",
                        lambda *, kind, path, context: grant)
    loader_preflight.preflight_loader_footprint(
        _Mesh(worker_args={"slab_weights": "auto"}), FL2VA_BF16, {},
        _graph(dit=FL2VA_BF16, refs="keyframes"))
    assert consent_pending.pending_cards() == []


def test_a_granted_consent_projects_slab_into_the_worker_args_the_load_uses(monkeypatch):
    """The click has to change the residency, or it changed nothing at all.

    Standing the guard down without projecting would delete the only capacity
    protection on this path and admit a load the host cannot hold, through the
    button that promised to prevent it.
    """
    grant = types.SimpleNamespace(id="c-0123456789ab", consent_source="panel")
    monkeypatch.setattr(consent_pending, "resolve",
                        lambda *, kind, path, context: grant)
    mesh = _Mesh(worker_args={"slab_weights": "auto"})
    loader_preflight.preflight_loader_footprint(
        mesh, FL2VA_BF16, {}, _graph(dit=FL2VA_BF16, refs="keyframes"))
    assert mesh.worker_args == {"slab_weights": True}


def test_a_consent_is_never_honored_for_a_quarantined_combination(monkeypatch, tmp_path):
    """A protocol bump makes an old FAIL read stale to the contextual lookup,
    and the rescue path has no ceremony to re-derive it, so the check is
    protocol blind."""
    from dgx_monarch.gate_ledger import GateLedger

    grant = types.SimpleNamespace(id="c-0123456789ab", consent_source="panel")
    monkeypatch.setattr(consent_pending, "resolve",
                        lambda *, kind, path, context: grant)
    path = str(tmp_path / "diffusion_models" / FL2VA_BF16)
    combo, artifacts = loader_preflight._combo_identity(FL2VA_BF16, {}, path)
    ledger = GateLedger(str(tmp_path / "output"))
    ledger.record(combo, artifacts, "deadbeef", "FAIL",
                  {"quarantine_levers": ["slab_weights", "lora_low_rss"]},
                  {"worker_args": {"slab_weights": True}})
    mesh = _Mesh(worker_args={"slab_weights": "auto"})
    message = _refuse(mesh, FL2VA_BF16, prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))
    assert mesh.worker_args == {"slab_weights": "auto"}
    assert "no waiver for this refusal" in message
    assert "Identity Gate" in message
    assert consent_pending.pending_cards() == []


def test_the_whole_one_click_arc_on_the_default_residency(tmp_path):
    """A refused queue, its card, one grant and a re-queue end in slab residency.

    No monkeypatched consent here. The refusal registers the card, the memo is
    written under the card's own key as the accept endpoint writes it, and the
    next queue must project slab residency into the worker args the eager load
    dispatches with.
    """
    mesh = _Mesh()
    graph = _graph(dit=FL2VA_BF16, refs="keyframes")
    _refuse(mesh, FL2VA_BF16, prompt=graph)
    card = consent_pending.pending_cards()[0]
    pending = consent_pending.peek_pending(card["key"], card["id"])
    assert pending is not None
    descriptor = pending.descriptor

    path = str(tmp_path / "diffusion_models" / FL2VA_BF16)
    record = consent_store.ConsentRecord(
        id="c" * 32, kind="rescue-slab", target_guard="loader_footprint_preflight",
        waiver_class="C", artifact=FL2VA_BF16, path=path,
        file_identity=consent_store.file_identity(path),
        memo_context=loader_preflight.memo_context(FL2VA_BF16, {}),
        granted_at="2026-08-04 12:00:00", granted_epoch=0.0, consent_source="panel",
        reason="stock residency cannot fit this checkpoint",
        ledger_key="audit:waiver:rescue-slab:x", combo_key=descriptor.combo_key,
        artifacts=descriptor.artifacts, artifacts_legacy=descriptor.artifacts_legacy,
        artifacts_legacy_complete=descriptor.artifacts_legacy_complete,
        unet_name=FL2VA_BF16)
    consent_store.grant(record, card["key"])

    loader_preflight.reset_memos()
    loader_preflight.preflight_loader_footprint(mesh, FL2VA_BF16, {}, graph)
    assert mesh.worker_args == {"slab_weights": True}


def test_an_unreadable_consent_store_asks_again_rather_than_granting(monkeypatch):
    def _boom(**kwargs):
        raise OSError("store unreadable")

    monkeypatch.setattr(consent_pending, "resolve", _boom)
    message = _refuse(_Mesh(worker_args={"slab_weights": "auto"}), FL2VA_BF16,
                      prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))
    assert "DGX Monarch panel" in message


def test_a_card_that_cannot_be_built_never_swallows_the_refusal(monkeypatch):
    def _boom(descriptor):
        raise RuntimeError("pending registry is broken")

    monkeypatch.setattr(consent_pending, "register_pending", _boom)
    message = _refuse(_Mesh(worker_args={"slab_weights": "auto"}), FL2VA_BF16,
                      prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))
    assert "driver-side footprint preflight" in message
    assert "DGX Monarch panel" in message


def test_the_headless_env_fallback_grants_the_rescue(monkeypatch):
    """The panel is the primary route and the env var is the fallback, so the
    headless path must clear the same refusal without a click."""
    monkeypatch.setenv(consent_pending.KIND_SPECS["rescue-slab"].env_var, "1")
    loader_preflight.preflight_loader_footprint(
        _Mesh(worker_args={"slab_weights": "auto"}), FL2VA_BF16, {},
        _graph(dit=FL2VA_BF16, refs="keyframes"))


def test_a_stack_comfy_already_holds_is_credited_not_charged(monkeypatch):
    """Comfy schedules by hops from an output, not by declared input order, so
    both VAE loaders in the repo's own H3 graph run before this node. Charging
    a resident artifact again is the false refusal class C forbids."""
    mm = types.ModuleType("comfy.model_management")
    mm.current_loaded_models = [object()]  # type: ignore[attr-defined]
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    terms = loader_graph.credited_terms(
        loader_graph.driver_stack_terms(_graph(dit=REF2VA_BF16, refs="video")))
    assert terms.credited and terms.total == 0
    # So the resident window clears: those bytes are already inside the
    # MemAvailable reading. The load window still refuses: on 2026-08-04 the
    # eager load of this 61.7 GiB artifact took MemAvailable from 114.5 GiB to
    # 1.4 GiB, about 1.8x the file, through a load path the log did not record.
    message = _refuse(_Mesh(), REF2VA_BF16, prompt=_graph(dit=REF2VA_BF16, refs="video"))
    assert f"at the {loader_preflight.WINDOW_TRANSIENT} window" in message
    # A smaller artifact clears both windows with the stack credited.
    loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_INT8, {}, _graph(dit=REF2VA_INT8, refs="video"))


def test_the_second_loader_of_a_dual_model_graph_is_not_charged_twice():
    graph = _graph(dit=REF2VA_INT8_PRUNED, refs="keyframes")
    loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_INT8_PRUNED, {}, graph)
    terms = loader_graph.credited_terms(
        loader_graph.driver_stack_terms(graph))
    assert terms.credited and terms.total == 0


def test_a_checkpoint_this_site_already_cleared_is_not_charged_again(monkeypatch):
    graph = _graph(dit=REF2VA_INT8_PRUNED, refs="keyframes")
    loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_INT8_PRUNED, {}, graph)
    # Same checkpoint, a box that has since filled up: the weight term is not
    # charged twice for an eager load that already happened.
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 12 * _GIB)
    loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_INT8_PRUNED, {}, graph)


def test_the_memo_is_bounded():
    for index in range(loader_graph._MEMO_LIMIT + 20):
        loader_graph.remember(loader_preflight._CLEARED_UNETS, f"m{index}")
    assert len(loader_preflight._CLEARED_UNETS) == loader_graph._MEMO_LIMIT


def test_the_env_escape_disables_the_refusal(monkeypatch):
    monkeypatch.setenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, "1")
    monkeypatch.setattr(driver_footprint, "_DISABLE_LOGGED", False)
    loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_BF16, {}, _graph(dit=REF2VA_BF16, refs="video"))


def test_a_weight_dtype_cast_stands_the_whole_check_down():
    """A cast's resident size is not the file's, so the only charge available
    here would be the wrong number in the refusing direction."""
    loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_BF16, {"weight_dtype": "fp8_e4m3fn"},
        _graph(dit=REF2VA_BF16, refs="video"))


def test_a_discrete_gpu_is_never_bounded_by_host_memory(monkeypatch):
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: False)
    loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_BF16, {}, _graph(dit=REF2VA_BF16, refs="video"))


def test_an_unregistered_family_is_a_hard_no_op(monkeypatch):
    from dgx_monarch.adapters import detect

    monkeypatch.setattr(detect, "sniff_checkpoint", lambda path: ("krea2", "bf16"))
    loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_BF16, {}, _graph(dit=REF2VA_BF16, refs="video"))


def test_a_worker_not_on_the_driver_host_is_not_charged(monkeypatch):
    mesh = _Mesh()
    mesh.handle.owns_hosts = True
    mesh.handle.config = types.SimpleNamespace(
        hosts=(types.SimpleNamespace(name="a-host-that-is-not-this-box"),),
        worker_args={})
    monkeypatch.setattr(driver_footprint, "rank0_co_resident", lambda mesh_: False)
    loader_preflight.preflight_loader_footprint(
        mesh, REF2VA_BF16, {}, _graph(dit=REF2VA_BF16, refs="video"))


def test_a_broken_estimator_never_blocks_a_load(monkeypatch):
    def _boom(**kwargs):
        raise ValueError("estimator is broken")

    monkeypatch.setattr(driver_footprint, "estimate_driver_footprint", _boom)
    loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_BF16, {}, _graph(dit=REF2VA_BF16, refs="video"))


def test_an_operator_reserve_widens_the_budget_it_is_measured_against():
    """uma_reserve_gb from cluster.toml [worker_args] must reach this refusal
    too, or the driver preflight disagrees with the worker, whose warning and
    capacity walls read the same reserve."""
    mesh = _Mesh(config_worker_args={"uma_reserve_gb": 40.0})
    message = _refuse(mesh, REF2VA_INT8_PRUNED,
                      prompt=_graph(dit=REF2VA_INT8_PRUNED, refs="video"))
    assert "40.0 GiB reserve" in message


def test_off_linux_nothing_is_bounded(monkeypatch):
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: None)
    loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_BF16, {}, _graph(dit=REF2VA_BF16, refs="video"))


def _loader_body() -> list[ast.stmt]:
    tree = ast.parse((REPO / "src/dgx_monarch/nodes/loaders.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "DGXMonarchUNETLoader":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "load":
                    return item.body
    raise AssertionError("DGXMonarchUNETLoader.load not found")


def test_the_preflight_call_sits_above_ensure_live_and_outside_any_session():
    """Placement is the contract: a refusal after ensure_live has healed or
    spawned the fleet, or inside mutation_render_session after topology and
    worker args were pushed to every rank, refuses too late."""
    body = _loader_body()
    preflight_at = ensure_live_at = None
    for index, statement in enumerate(body):
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name == "preflight_loader_footprint" and preflight_at is None:
                preflight_at = index
            if name == "ensure_live" and ensure_live_at is None:
                ensure_live_at = index
    assert preflight_at is not None, "the loader-site preflight call is gone"
    assert ensure_live_at is not None
    assert preflight_at < ensure_live_at
    assert not isinstance(body[preflight_at], (ast.With, ast.AsyncWith))


def test_the_loader_declares_the_hidden_prompt_input_and_accepts_it():
    import inspect

    from dgx_monarch.nodes.loaders import (
        DGXMonarchUncondUNETLoader,
        DGXMonarchUNETLoader,
    )

    fp = types.ModuleType("folder_paths")
    fp.get_filename_list = lambda kind: ["stub.safetensors"]  # type: ignore[attr-defined]
    previous = sys.modules.get("folder_paths")
    sys.modules["folder_paths"] = fp
    try:
        for cls in (DGXMonarchUNETLoader, DGXMonarchUncondUNETLoader):
            hidden = cls.INPUT_TYPES()["hidden"]
            assert hidden["prompt"] == "PROMPT"
            assert list(hidden) == ["unique_id", "prompt"], "append-only"
            assert "prompt" in inspect.signature(cls.load).parameters
    finally:
        if previous is None:
            sys.modules.pop("folder_paths", None)
        else:
            sys.modules["folder_paths"] = previous


def test_a_granted_consent_reaches_an_eager_load_this_site_never_estimated(monkeypatch):
    """The estimator is family scoped; the worker's wall is not.

    `DRIVER_STACK_FAMILIES` limits coverage to registered families. The env
    escape, a dtype cast, a discrete GPU or no MemAvailable can skip this
    site entirely, while the worker refuses on file size against MemAvailable for
    any big checkpoint. So a card raised by a worker has to be honored for a
    load this site returned None for, or the click never reaches the load and
    the same card comes back forever.
    """
    from dgx_monarch.adapters import detect
    from dgx_monarch.nodes import consent_projection

    monkeypatch.setattr(detect, "sniff_checkpoint", lambda path: ("krea2", "bf16"))
    grant = types.SimpleNamespace(id="c" * 32, consent_source="panel", key="k", kind="rescue-slab")
    monkeypatch.setattr(consent_pending, "resolve", lambda *, kind, path, context: grant)

    mesh = _Mesh(worker_args={"slab_weights": "auto"})
    graph = _graph(dit=REF2VA_BF16, refs="video")
    assert loader_preflight.preflight_loader_footprint(mesh, REF2VA_BF16, {}, graph) is None
    assert mesh.worker_args["slab_weights"] == "auto"

    load = consent_projection.project_for_loader(mesh, REF2VA_BF16, {})
    assert mesh.worker_args["slab_weights"] is True
    assert load is not None and load.unet_name == REF2VA_BF16
    assert load.worker_args["slab_weights"] is True


def test_the_capacity_row_fingerprints_the_residency_the_load_actually_ran_under(monkeypatch):
    """The row's capability fingerprint is built from RescueLoad.worker_args.
    Capturing them before the projection would permanently audit a consented
    slab load as having run under the residency it was granted to change."""
    grant = types.SimpleNamespace(id="c" * 32, consent_source="panel", key="k", kind="rescue-slab")
    monkeypatch.setattr(consent_pending, "resolve", lambda *, kind, path, context: grant)
    mesh = _Mesh(worker_args={"slab_weights": "auto"})
    load = loader_preflight.preflight_loader_footprint(
        mesh, FL2VA_BF16, {}, _graph(dit=FL2VA_BF16, refs="keyframes"))
    assert load is not None
    assert load.worker_args["slab_weights"] is True
    assert mesh.worker_args["slab_weights"] is True
    assert len(load.artifacts) == 64, "the row's artifacts column is the 64 hex set digest"


def test_an_explicit_stock_request_is_not_met_with_a_card(monkeypatch):
    """The operator already answered this question on the Init node. Offering
    a card would ask them to overrule themselves, and projecting over it would
    be the stale driver push DESIGN section 5.9 invariant 2 forbids."""
    message = _refuse(_Mesh(worker_args={"slab_weights": False}), FL2VA_BF16,
                      prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))
    assert consent_pending.pending_cards() == []
    assert "DGX Monarch panel" not in message
    assert "slab_weights=off" in message


def test_the_quarantine_arm_never_offers_the_preflight_kill_switch(monkeypatch, tmp_path):
    """A message that hands over the kill switch one sentence before saying
    there is no waiver both denies and offers a bypass."""
    from dgx_monarch.gate_ledger import GateLedger

    path = str(tmp_path / "diffusion_models" / FL2VA_BF16)
    combo, artifacts = loader_preflight._combo_identity(FL2VA_BF16, {}, path)
    GateLedger(str(tmp_path / "output")).record(
        combo, artifacts, "deadbeef", "FAIL",
        {"quarantine_levers": ["slab_weights", "lora_low_rss"]},
        {"worker_args": {"slab_weights": True}})
    message = _refuse(_Mesh(worker_args={"slab_weights": "auto"}), FL2VA_BF16,
                      prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))
    assert "no waiver for this refusal" in message
    assert driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV not in message


def test_the_rescue_clause_names_the_panel_before_any_widget(monkeypatch):
    """The panel is the primary route and the env var the headless fallback
    (docs/DESIGN.md section 5.9), so a class C message names the panel first and
    the env second. A widget named inside the body lands before both."""
    message = _refuse(_Mesh(worker_args={"slab_weights": "auto"}), FL2VA_BF16,
                      prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))
    assert "slab_weights=on on the DGX Monarch Init node" not in message
    assert message.index("DGX Monarch panel") < message.index("Headless:")


# The headless caller notice. A load that fits says nothing, so without this
# line a wall that priced only the weights looks like one that priced the whole
# stack. The line reports; it never refuses.


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def unpriced_notice(monkeypatch):
    """This module's own log records, with the per-family memo emptied first.

    The memo has proc lifetime and the whole suite shares one process, so a
    test that reads it without replacing it reads whatever every earlier
    promptless call already consumed. The handler goes on the module logger
    directly because these loggers do not propagate.
    """
    from dgx_monarch.log import get_logger

    monkeypatch.setattr(loader_preflight, "_UNPRICED_STACK_WARNED", set())
    capture = _CaptureHandler()
    logger = get_logger("dgx_monarch.nodes.loader_preflight")
    logger.addHandler(capture)
    yield capture.records
    logger.removeHandler(capture)


def test_a_promptless_load_says_the_driver_stack_is_unpriced_once(unpriced_notice):
    """A caller that imported the node classes is told, in the driver log, that
    the wall charged the weights and nothing else. It says so once, because a
    per-load line would be noise in a benchmark."""
    mesh = _Mesh()
    assert loader_preflight.preflight_loader_footprint(
        mesh, REF2VA_INT8_PRUNED, {}, None) is None

    assert len(unpriced_notice) == 1
    record = unpriced_notice[0]
    assert record.levelno == logging.WARNING
    message = record.getMessage()
    assert "no readable graph, headless call" in message
    assert "minimax_h3" in message
    assert "docs/TROUBLESHOOTING.md #85" in message

    assert loader_preflight.preflight_loader_footprint(
        mesh, REF2VA_INT8_PRUNED, {}, None) is None
    assert len(unpriced_notice) == 1


def test_a_prompted_load_never_says_the_stack_is_unpriced(unpriced_notice):
    """On the browser path ComfyUI fills the hidden prompt on every queued
    render, so the wall priced the whole stack and there is nothing to report."""
    assert loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_INT8_PRUNED, {}, _graph(dit=REF2VA_INT8_PRUNED)) is None
    assert unpriced_notice == []


def test_a_second_family_gets_its_own_line(monkeypatch, unpriced_notice):
    """Why the memo is keyed by family and not a plain flag: the message names
    the family whose stack went unpriced, so one line for the whole process
    would name the first family and stay silent about the second."""
    from dgx_monarch.adapters import detect

    assert loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_INT8_PRUNED, {}, None) is None
    monkeypatch.setattr(detect, "sniff_checkpoint", lambda path: ("ltx", "bf16"))
    assert loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_INT8_PRUNED, {}, None) is None

    said = [record.getMessage() for record in unpriced_notice]
    assert len(said) == 2
    assert "minimax_h3" in said[0]
    assert "ltx" in said[1]


def _ltx_without_config(monkeypatch):
    """Sniff LTX from the staged artifact and report a header with no config."""
    from dgx_monarch import family_select
    from dgx_monarch.adapters import detect

    monkeypatch.setattr(detect, "sniff_checkpoint", lambda path: ("ltx", "bf16"))
    monkeypatch.setattr(detect, "sniff_metadata_keys",
                        lambda path: ("model_version",))
    monkeypatch.setattr(family_select, "_auto_warned", set())


def _configless_lines(caplog):
    return [r.getMessage() for r in caplog.records
            if "carries no 'config' metadata" in r.getMessage()]


def test_the_configless_report_survives_the_env_escape(monkeypatch, caplog):
    """The capacity switches decide whether an estimate may refuse. None of
    them has anything to say about the architecture the file will build, so
    the report sits above all of them."""
    import logging

    _ltx_without_config(monkeypatch)
    monkeypatch.setenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, "1")
    monkeypatch.setattr(driver_footprint, "_DISABLE_LOGGED", False)
    with caplog.at_level(logging.WARNING):
        loader_preflight.preflight_loader_footprint(
            _Mesh(), REF2VA_BF16, {}, _graph(dit=REF2VA_BF16, refs="video"))
    assert len(_configless_lines(caplog)) == 1


def test_the_configless_report_survives_a_discrete_gpu(monkeypatch, caplog):
    import logging

    _ltx_without_config(monkeypatch)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: False)
    with caplog.at_level(logging.WARNING):
        loader_preflight.preflight_loader_footprint(
            _Mesh(), REF2VA_BF16, {}, _graph(dit=REF2VA_BF16, refs="video"))
    assert len(_configless_lines(caplog)) == 1


def test_a_checkpoint_that_kept_its_config_says_nothing_at_the_loader_site(
    monkeypatch, caplog,
):
    import logging

    from dgx_monarch import family_select
    from dgx_monarch.adapters import detect

    monkeypatch.setattr(detect, "sniff_checkpoint", lambda path: ("ltx", "bf16"))
    monkeypatch.setattr(detect, "sniff_metadata_keys",
                        lambda path: ("config", "model_version"))
    monkeypatch.setattr(family_select, "_auto_warned", set())
    monkeypatch.setenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, "1")
    monkeypatch.setattr(driver_footprint, "_DISABLE_LOGGED", False)
    with caplog.at_level(logging.WARNING):
        loader_preflight.preflight_loader_footprint(
            _Mesh(), REF2VA_BF16, {}, _graph(dit=REF2VA_BF16, refs="video"))
    assert _configless_lines(caplog) == []


def _unstable(monkeypatch):
    """Make this box unable to fingerprint the checkpoint it is about to load.

    What a real one looks like: a file still being copied in, or rewritten
    under the reader, whose stat changes between the two reads
    `gate_artifacts.artifact_signature` takes.
    """
    from dgx_monarch import gate_ledger

    monkeypatch.setattr(gate_ledger, "artifact_signature", lambda path: "unstable")


def test_an_unstable_artifact_identity_refuses_where_master_admitted(monkeypatch):
    """The estimate has already refused when the identity is built. An identity
    error that reached the outer `except Exception: return None` would admit
    the load unpriced: an eager load on a host measured unable to hold it, with
    the refusal thrown away."""
    from dgx_monarch.refusal import parse_refusal_tag

    graph = _graph(dit=FL2VA_BF16, refs="keyframes")
    clean = _refuse(_Mesh(worker_args={"slab_weights": False}), FL2VA_BF16,
                    prompt=graph)
    loader_preflight.reset_memos()
    consent_pending.clear_all()

    _unstable(monkeypatch)
    mesh = _Mesh(worker_args={"slab_weights": False})
    message = _refuse(mesh, FL2VA_BF16, prompt=graph)

    tag = parse_refusal_tag(message)
    assert tag is not None
    assert tag.refusal_class.value == "C"
    assert tag.guard == "loader_footprint_preflight"
    assert tag.waivable is False
    assert "cannot read a stable identity" in message
    assert f"diffusion_models/{FL2VA_BF16}" in message      # it names the artifact
    assert consent_pending.pending_cards() == []
    # The stable-identity refusal's charged bytes: the same load, priced the
    # same way, down to the tag and every charged term. The identity decides
    # what rescue can be offered, never what the weights cost.
    priced = clean.split(" Slab residency is not offered here:")[0]
    assert " Charged: DiT weights " in priced
    assert message.startswith(priced)


def test_a_standing_grant_does_not_cover_an_unstable_artifact_identity(monkeypatch):
    """The grant is resolved by `live_grant`, which sits after this raise and so
    never runs here, and a consent is bound to artifact bytes this box cannot
    currently name. So the refusal stands and no slab residency is projected
    into the load."""
    _unstable(monkeypatch)
    grant = types.SimpleNamespace(id="c-0123456789ab", consent_source="panel")
    monkeypatch.setattr(consent_pending, "resolve",
                        lambda *, kind, path, context: grant)
    mesh = _Mesh(worker_args={"slab_weights": "auto"})
    message = _refuse(mesh, FL2VA_BF16,
                      prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))
    assert "cannot read a stable identity" in message
    assert mesh.worker_args == {"slab_weights": "auto"}
    assert consent_pending.pending_cards() == []


def test_an_unstable_identity_still_admits_a_load_that_fits(monkeypatch):
    """Only a completed over-budget estimate refuses here. An artifact nobody
    can fingerprint is not itself a capacity fault, so a load with room stays a
    no-op."""
    _unstable(monkeypatch)
    assert loader_preflight.preflight_loader_footprint(
        _Mesh(), REF2VA_INT8_PRUNED, {},
        _graph(dit=REF2VA_INT8_PRUNED)) is None


def _ledger_lines(tmp_path, *lines: bytes) -> None:
    """Append raw bytes, so a torn line can precede an intact row."""
    from dgx_monarch.gate_ledger import LEDGER_NAME

    path = tmp_path / "output" / LEDGER_NAME
    with path.open("ab") as handle:
        for line in lines:
            handle.write(line)


def _load_scope(mesh) -> dict:
    """The capability context the loader site reads its own scope under."""
    from dgx_monarch.nodes import consent_rescue

    context = consent_rescue.load_capability_context(mesh.worker_args)
    assert context is not None
    return context


def test_a_damaged_line_below_this_loads_own_scoped_row_stops_blocking(
        monkeypatch, tmp_path):
    """A torn line the contextless read cannot tie to any scope must not
    outrank an authoritative row for this load's own capability.

    The state is unchanged for everything the ledger recorded about these
    artifacts. Only the unknown is asked again, under the scope this load runs
    in, and there the damage is older than the newest matching row.
    """
    from dgx_monarch.gate_ledger import GateLedger
    from dgx_monarch.nodes import consent_rescue
    from dgx_monarch.refusal import parse_refusal_tag

    mesh = _Mesh(worker_args={"slab_weights": "auto"})
    path = str(tmp_path / "diffusion_models" / FL2VA_BF16)
    combo, artifacts = loader_preflight._combo_identity(FL2VA_BF16, {}, path)
    _ledger_lines(tmp_path, b'{"torn": \n')
    GateLedger(str(tmp_path / "output")).record(
        combo, artifacts, "deadbeef", "PASS", context=_load_scope(mesh))

    # The contextless read of this exact ledger still answers unknown; only the
    # scoped read clears it.
    assert ("damaged record" in
            consent_rescue.quarantine_reason(combo, artifacts))

    message = _refuse(mesh, FL2VA_BF16, prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))
    tag = parse_refusal_tag(message)
    assert tag is not None and tag.refusal_class.value == "C"
    assert "no waiver for this refusal" not in message
    assert "damaged record" not in message
    assert consent_pending.pending_cards()          # the rescue is on offer again


def test_a_damaged_line_still_blocks_where_the_row_carries_no_scope(
        monkeypatch, tmp_path):
    """A row with no capability context is in no scope this load can claim, so
    the torn line could still have carried a verdict for these artifacts, and
    the lookup says so."""
    from dgx_monarch.gate_ledger import GateLedger
    from dgx_monarch.refusal import parse_refusal_tag

    mesh = _Mesh(worker_args={"slab_weights": "auto"})
    path = str(tmp_path / "diffusion_models" / FL2VA_BF16)
    combo, artifacts = loader_preflight._combo_identity(FL2VA_BF16, {}, path)
    _ledger_lines(tmp_path, b'{"torn": \n')
    GateLedger(str(tmp_path / "output")).record(combo, artifacts, "deadbeef", "PASS")

    message = _refuse(mesh, FL2VA_BF16, prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))
    tag = parse_refusal_tag(message)
    assert tag is not None and tag.refusal_class.value == "C"
    assert "There is no waiver for this refusal" in message
    assert "evidence is readable" not in message
    assert consent_pending.pending_cards() == []


@pytest.mark.parametrize("decision_name", [
    "_UNKNOWN_QUERY", "_UNKNOWN_LEDGER", "_UNKNOWN_DAMAGE",
    "_UNKNOWN_UNSCOPED_DAMAGE", "_UNKNOWN_UNBOUND", "_UNKNOWN_VERDICT",
])
def test_each_unknown_ledger_decision_refuses_capacity_without_a_consent(
        monkeypatch, decision_name):
    """Unreadable evidence blocks the load but is never a measured FAIL.

    The six decisions are separate because their operator actions differ. At
    this boundary they share two safety properties: no decision becomes class
    K without a proven FAIL, and none opens the rescue panel that would grant
    slab residency while its evidence cannot be read.
    """
    from dgx_monarch.nodes import consent_quarantine, consent_rescue
    from dgx_monarch.refusal import parse_refusal_tag

    decision = getattr(consent_quarantine, decision_name)
    monkeypatch.setattr(
        consent_rescue, "quarantine_decision_for_load",
        lambda *_args: decision)
    mesh = _Mesh(worker_args={"slab_weights": "auto"})

    message = _refuse(
        mesh, FL2VA_BF16, prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))

    tag = parse_refusal_tag(message)
    assert tag is not None and tag.refusal_class.value == "C"
    assert decision.reason in message
    assert "No slab-residency consent can be granted" in message
    assert consent_pending.pending_cards() == []
    assert mesh.worker_args == {"slab_weights": "auto"}


@pytest.mark.parametrize(("verdict", "reason"), [
    ("INCONCLUSIVE", "was INCONCLUSIVE"),
    ("RETESTING", "is RETESTING"),
    ("PASS", "diagnostic-only"),
])
def test_readable_denied_ledger_verdicts_keep_their_reason_without_a_grant(
        tmp_path, verdict, reason):
    """A durable denial is not unreadable evidence or measured wrongness."""
    from dgx_monarch.gate_ledger import GateLedger
    from dgx_monarch.refusal import parse_refusal_tag

    mesh = _Mesh(worker_args={"slab_weights": "auto"})
    path = str(tmp_path / "diffusion_models" / FL2VA_BF16)
    combo, artifacts = loader_preflight._combo_identity(FL2VA_BF16, {}, path)
    ledger = GateLedger(str(tmp_path / "output"))
    if verdict == "RETESTING":
        ledger.begin_retest_required(combo, artifacts, "deadbeef", [_load_scope(mesh)])
    else:
        ledger.record(combo, artifacts, "deadbeef", verdict,
                      context=None if verdict == "PASS" else _load_scope(mesh))

    message = _refuse(
        mesh, FL2VA_BF16, prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))

    tag = parse_refusal_tag(message)
    assert tag is not None and tag.refusal_class.value == "C"
    assert reason in message
    assert "There is no waiver for this refusal" in message
    assert "evidence is readable" not in message
    assert consent_pending.pending_cards() == []


def test_a_proven_ledger_fail_remains_the_only_class_k_loader_refusal(monkeypatch):
    """A current identity-gate FAIL is measured wrongness, not uncertainty."""
    from dgx_monarch.nodes import consent_quarantine, consent_rescue
    from dgx_monarch.refusal import parse_refusal_tag

    monkeypatch.setattr(
        consent_rescue, "quarantine_decision_for_load",
        lambda *_args: consent_quarantine._FAILED)

    message = _refuse(
        _Mesh(), FL2VA_BF16, prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))

    tag = parse_refusal_tag(message)
    assert tag is not None and tag.refusal_class.value == "K"
    assert consent_quarantine._FAILED.reason in message
    assert "There is no waiver for this refusal" in message
    assert consent_pending.pending_cards() == []


def test_a_clear_ledger_decision_reaches_the_normal_capacity_rescue(monkeypatch):
    """Class C offers its fitting-strategy card when evidence is clear."""
    from dgx_monarch.nodes import consent_quarantine, consent_rescue
    from dgx_monarch.refusal import parse_refusal_tag

    monkeypatch.setattr(
        consent_rescue, "quarantine_decision_for_load",
        lambda *_args: consent_quarantine._CLEAR)

    message = _refuse(
        _Mesh(), FL2VA_BF16, prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))

    tag = parse_refusal_tag(message)
    assert tag is not None and tag.refusal_class.value == "C"
    assert consent_pending.pending_cards()


def test_a_gate_fail_in_its_own_ceremony_scope_still_quarantines(tmp_path):
    """The rule the scoped read must not break. A ceremony writes its FAIL
    under a context this site cannot build (a resolved topology and a live
    handle it runs above), so a lookup that asked only this load's scope would
    read clear and offer a card for a combination the gate has proven unsafe.
    The FAIL is settled on the contextless read, before any scope is asked.
    """
    from dgx_monarch.gate_ledger import GateLedger
    from dgx_monarch.refusal import parse_refusal_tag

    ceremony_scope = {
        "worker_args": {"slab_weights": True, "lora_low_rss": True},
        "mesh_mode": "cluster", "config_source": "cluster.toml",
        "config_fingerprint": "f" * 16, "world": 2, "hosts": 2,
        "gpus_per_host": 1, "topology_preset": "uly2", "attention": "sage",
        "sync_ulysses": True, "resolved_attention": "sage",
        "resolved_topology": {"ulysses": 2, "ring": 1, "cfg": 1, "dp": 1,
                              "fsdp": False},
    }
    path = str(tmp_path / "diffusion_models" / FL2VA_BF16)
    combo, artifacts = loader_preflight._combo_identity(FL2VA_BF16, {}, path)
    GateLedger(str(tmp_path / "output")).record(
        combo, artifacts, "deadbeef", "FAIL",
        {"quarantine_levers": ["slab_weights", "lora_low_rss"]}, ceremony_scope)

    mesh = _Mesh(worker_args={"slab_weights": "auto"})
    message = _refuse(mesh, FL2VA_BF16, prompt=_graph(dit=FL2VA_BF16, refs="keyframes"))
    tag = parse_refusal_tag(message)
    assert tag is not None and tag.refusal_class.value == "K"
    assert "no waiver for this refusal" in message
    assert mesh.worker_args == {"slab_weights": "auto"}
    assert consent_pending.pending_cards() == []


def test_worker_args_the_ledger_cannot_canonicalize_keep_the_blind_read(monkeypatch):
    """No scope is better than a made-up one: a value the ledger cannot encode
    has no canonical context, so the site keeps the contextless answer rather
    than ask under a context it cannot encode."""
    from dgx_monarch.nodes import consent_rescue

    assert consent_rescue.load_capability_context({"uma_reserve_gb": object()}) is None
    assert consent_rescue.load_capability_context({"slab_weights": "auto"}) == {
        "worker_args": {"slab_weights": True}}


def _ceremony_scope() -> dict:
    """A capability context shaped the way an identity-gate ceremony writes one.

    Twelve keys, a resolved topology and a live handle's physical identity: the
    loader site runs above `ensure_live` and can build none of it, which is why
    its own scope holds no ceremony row on any real ledger.
    """
    return {
        "worker_args": {"slab_weights": True, "lora_low_rss": True},
        "mesh_mode": "cluster", "config_source": "cluster.toml",
        "config_fingerprint": "f" * 16, "world": 2, "hosts": 2,
        "gpus_per_host": 1, "topology_preset": "uly2", "attention": "sage",
        "sync_ulysses": True, "resolved_attention": "sage",
        "resolved_topology": {"ulysses": 2, "ring": 1, "cfg": 1, "dp": 1,
                              "fsdp": False},
    }


def test_a_card_this_loads_scope_cleared_is_a_card_a_grant_can_honor(
        tmp_path, monkeypatch):
    """The two loader reads have to agree, or the click buys nothing.

    The pre-emptive read decides whether a card is offered and `live_grant`
    decides whether the click is honored, and both read this one ledger. A
    loader-raised card is granted with no capability context of its own, so if
    only the first read asks under this load's scope, the grant is refused on
    the next queue and the same card comes back on every queue.
    """
    from dgx_monarch import gate_ledger as ledger_mod
    from dgx_monarch.nodes import consent_rescue

    mesh = _Mesh()
    graph = _graph(dit=FL2VA_BF16, refs="keyframes")
    path = str(tmp_path / "diffusion_models" / FL2VA_BF16)
    combo, artifacts = loader_preflight._combo_identity(FL2VA_BF16, {}, path)
    ledger = ledger_mod.GateLedger(str(tmp_path / "output"))
    with monkeypatch.context() as stale:
        # The normal state after a version bump: a real ceremony's PASS, in its
        # own scope, no longer bound to the running release.
        stale.setattr(ledger_mod, "__version__", "stale-package")
        ledger.record(combo, artifacts, "deadbeef", "PASS", context=_ceremony_scope())
    assert "not authoritative" in consent_rescue.quarantine_reason(combo, artifacts)

    _refuse(mesh, FL2VA_BF16, prompt=graph)
    card = consent_pending.pending_cards()[0]
    pending = consent_pending.peek_pending(card["key"], card["id"])
    assert pending is not None
    descriptor = pending.descriptor
    consent_store.grant(consent_store.ConsentRecord(
        id="c" * 32, kind="rescue-slab", target_guard="loader_footprint_preflight",
        waiver_class="C", artifact=FL2VA_BF16, path=path,
        file_identity=consent_store.file_identity(path),
        memo_context=loader_preflight.memo_context(FL2VA_BF16, {}),
        granted_at="2026-09-04 12:00:00", granted_epoch=0.0, consent_source="panel",
        reason="stock residency cannot fit this checkpoint",
        ledger_key="audit:waiver:rescue-slab:x", combo_key=descriptor.combo_key,
        artifacts=descriptor.artifacts, artifacts_legacy=descriptor.artifacts_legacy,
        artifacts_legacy_complete=descriptor.artifacts_legacy_complete,
        unet_name=FL2VA_BF16), card["key"])

    loader_preflight.reset_memos()
    consent_pending.clear_all()
    load = loader_preflight.preflight_loader_footprint(mesh, FL2VA_BF16, {}, graph)
    assert load is not None                       # a RescueLoad, not a refusal
    assert mesh.worker_args == {"slab_weights": True}
    assert consent_pending.pending_cards() == []
