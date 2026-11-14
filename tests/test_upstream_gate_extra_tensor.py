"""An extra tensor in a file cannot move a verdict of the upstream gate.

The header detector matches needles at any dotted boundary, strips a prefix per
key and takes the first signature row that matches. ComfyUI tests exact keys
after one prefix strip for the whole file. So one extra tensor could rename
the family for the detector while ComfyUI still built the model the gate must
refuse: an L2P file with ``aux.dec_net.x.weight`` loaded as latent Z-Image,
and a MiniMax-H3-Image file with ``aux.audio_patch_proj.weight`` lost the
family's header refusals (no sigma grid, #149; a quantization table ComfyUI
cannot parse, #150). The gate's two family tests now read ComfyUI's exact keys first
(``adapters/gate_keys.py``), and the detector's reading can only add.

Every file is hand-made: a header and a few zero bytes. The last tests hold the
same names against ComfyUI's own detection; the comfy-canary job runs them.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from comfy_managed_helpers import _isolated_process_state  # noqa: F401  # autouse fixture import.
from dgx_monarch import upstream_gate
from dgx_monarch.adapters import detect, gate_keys
from test_dropped_scale_gate import _comfy_checkout
from test_dropped_scale_gate import real_comfy as real_comfy  # the real-ComfyUI fixture
from test_minimax_h3_image_comfy_detection import _FAMILY_SINCE, _comfy_version
from test_upstream_gate_damaged_header import DEEP, _assert_on_both_paths
from test_upstream_gate_damaged_header import models as models  # both funnels' fixture

L2P_NUMBER, GRID_NUMBER = upstream_gate.TROUBLESHOOTING, upstream_gate.H3_IMAGE_GRID_TROUBLESHOOTING
# What a MiniMax-H3-Image file with its grid gets: None, since the family runs on its own
# adapter and the file passes the gate (tests/test_upstream_gate_damaged_header.py holds the
# same line). The family's other header refusal, a quantization table ComfyUI cannot parse:
H3_IMAGE_WITH_GRID: int | None = None
QUANT_NUMBER = upstream_gate.H3_IMAGE_QUANT_METADATA_TROUBLESHOOTING
CONFIG = {"config": json.dumps({"transformer": {"pdd_steps": 4, "pdd_sigmas": [1.0, 0.5, 0.0]}})}
# Names with the shapes ComfyUI's detection reads its dimensions from. The gate reads names.
ZIMAGE = {
    "cap_embedder.0.weight": [8], "cap_embedder.1.weight": [3840, 8], "cap_embedder.1.bias": [3840],
    "noise_refiner.0.attention.k_norm.weight": [8], "x_pad_token": [1, 3840],
    "cap_pad_token": [1, 3840], "layers.0.attention.qkv.weight": [8, 8],
    "layers.0.ffn_norm1.weight": [8], "layers.1.attention.qkv.weight": [8, 8],
    "layers.1.ffn_norm1.weight": [8], "x_embedder.weight": [3840, 12],
}
L2P = {**ZIMAGE, "all_x_embedder.16-1.weight": [8, 8], "local_decoder.blocks.0.weight": [8, 8]}
DCT = {**ZIMAGE, "dec_net.cond_embed.weight": [8, 3840], "dec_net.final_layer.linear.weight": [12, 8],
       "dec_net.input_embedder.embedder.0.weight": [8, 28], "dec_net.res_blocks.0.weight": [8]}
H3_IMAGE = {
    "video_patch_proj.weight": [64, 96], "condition_proj.weight": [64, 32],
    "blocks.0.attn.qkv_proj.weight": [96, 64], "blocks.0.attn.q_norm.weight": [16],
    "blocks.0.mlp.fc1.weight": [128, 64], "final_layer.video_out.weight": [96, 64],
    "adaln_t_table": [9, 8], "rope.inv_freq": [4],
}
H3_VIDEO = {**H3_IMAGE, "audio_patch_proj.weight": [64, 16], "final_layer.audio_out.weight": [16, 64]}
# (names, metadata, the entry the file refuses with)
BASES: dict[str, tuple[dict[str, list[int]], dict[str, str] | None, int | None]] = {
    "l2p": (L2P, None, L2P_NUMBER),
    "h3-image": (H3_IMAGE, CONFIG, H3_IMAGE_WITH_GRID),
    "h3-image-no-grid": (H3_IMAGE, None, GRID_NUMBER),
    "h3-image-bad-table": (H3_IMAGE, {**CONFIG, "_quantization_metadata": "{not json"}, QUANT_NUMBER),
}


def _write(folder: Path, name: str, names: dict[str, list[int]] | list[str],
           metadata: dict[str, str] | None = None) -> str:
    header: dict = {"__metadata__": metadata} if metadata else {}
    offset = 0
    for key in names:  # the gate reads names and dtypes: one BF16 value each
        header[key] = {"dtype": "BF16", "shape": [1], "data_offsets": [offset, offset + 2]}
        offset += 2
    blob = json.dumps(header).encode()
    (folder / name).write_bytes(struct.pack("<Q", len(blob)) + blob + b"\0" * offset)
    return name


def _row_needles_as_extra_tensors() -> list[tuple[str, dict[str, list[int]]]]:
    """For each signature row, tensors under ``aux.`` that satisfy every needle of the row."""
    rows = []
    for index, (family, needles) in enumerate(detect._SIGNATURES):
        extra = {f"aux.{needle}weight" if needle.endswith(".") else f"aux.{needle}": [1]
                 for needle in needles}
        rows.append((f"{index}-{family}", extra))
    return rows


ROW_EXTRAS = _row_needles_as_extra_tensors()
# The reviewer's two, the absence needles at a boundary, by name.
NAMED_EXTRAS = {"aux.dec_net.x.weight": [1], "aux.audio_patch_proj.weight": [1],
                "aux.audio_patch_proj.x.weight": [1], "tail.dec_net.cond_embed.weight": [1]}


@pytest.mark.parametrize("extra", [*NAMED_EXTRAS, *(label for label, _ in ROW_EXTRAS)])
@pytest.mark.parametrize("base", sorted(BASES))
def test_an_extra_tensor_cannot_cancel_a_family_refusal_on_either_path(models, base, extra):
    """Each base against the two absence needles and against every signature row's
    needles as one extra tensor each: the driver and the Worker refuse as they refuse
    the file without it. ComfyUI's keys, the ones that decide what it builds, did not move."""
    names, metadata, number = BASES[base]
    reference = _assert_on_both_paths(models, _write(models, "plain.safetensors", names, metadata), number)
    added = {extra: NAMED_EXTRAS[extra]} if extra in NAMED_EXTRAS else dict(ROW_EXTRAS)[extra]
    name = _write(models, "extra.safetensors", {**names, **added}, metadata)
    assert _assert_on_both_paths(models, name, number) == reference


def test_the_fix_shows_on_the_detectors_own_reading(models):
    """What made the hole, pinned: the detector, which chooses topologies, still reads
    these files as something else, and the gate no longer asks it first."""
    l2p = [*L2P, "aux.dec_net.x.weight"]
    assert not detect.is_zimage_l2p_from_keys(l2p)
    assert gate_keys.reads_zimage_l2p(l2p, frozenset(l2p))
    image = [*H3_IMAGE, "aux.audio_patch_proj.weight"]
    assert detect.detect_family_from_keys(image) == "minimax_h3"
    assert gate_keys.reads_minimax_h3_image(image, frozenset(image))
    renamed = [*H3_IMAGE, "aux.txtfusion.x.weight"]
    assert detect.detect_family_from_keys(renamed) == "krea2"
    assert gate_keys.reads_minimax_h3_image(renamed, frozenset(renamed))


@pytest.mark.parametrize("prefix", detect._PREFIXES)
def test_the_absence_keys_under_a_prefix_follow_what_comfy_would_load(models, prefix):
    """The detector strips each of its prefixes per key, so ``<prefix>dec_net...`` read
    as the pixel decoder and ``<prefix>audio_patch_proj.weight`` as H3's audio path.
    ComfyUI strips one prefix for the file. Where the extra key starts with ``model.``
    its load keeps that one key alone and detects nothing, so the load fails by itself
    and the gate gives no verdict. Under any other prefix ComfyUI's keys are unmoved,
    and the refusal stands."""
    comfy_keeps_the_extra_key_alone = prefix.startswith("model.")
    for base, key in (("l2p", "dec_net.cond_embed.weight"), ("h3-image", "audio_patch_proj.weight"),
                      ("h3-image-no-grid", "audio_patch_proj.weight"),
                      ("h3-image-bad-table", "audio_patch_proj.weight")):
        names, metadata, number = BASES[base]
        name = _write(models, f"{base}.safetensors", {**names, prefix + key: [1]}, metadata)
        _assert_on_both_paths(models, name, None if comfy_keeps_the_extra_key_alone else number)


def test_the_l2p_refusal_answers_whatever_the_quantization_table_holds(models):
    """#91 is decided before any quantization metadata counts: the helper that names
    ComfyUI's keys never raises, on a table that is not JSON or one nested past the
    parser's limit, with or without the extra tensor."""
    for label, table in (("broken", "{not json"), ("deep", DEEP), ("null", "null")):
        for extra in ({}, {"aux.dec_net.x.weight": [1]}):
            name = _write(models, f"{label}.safetensors", {**L2P, **extra},
                          {"_quantization_metadata": table})
            _assert_on_both_paths(models, name, L2P_NUMBER)
    assert gate_keys.comfy_detection_keys(dict.fromkeys(L2P, "BF16"),
                                          {"_quantization_metadata": DEEP}) == frozenset(L2P)


def test_an_l2p_file_only_the_exact_keys_catch_still_logs_its_key_paths(models, caplog):
    """#91 tells the reader to confirm the spelling from the log. The strict sniff, which
    writes that line, reads this header and answers False, so the gate writes it."""
    name = _write(models, "l2p.safetensors", {**L2P, "aux.dec_net.x.weight": [1]})
    assert detect.sniff_zimage_l2p(str(models / name)) is False
    with caplog.at_level("INFO"):
        assert upstream_gate.refusal_for(str(models / name)) == (upstream_gate.ZIMAGE_L2P_REFUSAL, 91)
    assert any("carries the Z-Image L2P decoder (local_decoder.blocks.0.weight) and no dec_net head"
               in record.getMessage() for record in caplog.records)


SOUND = {
    "latent-zimage": (ZIMAGE, None),
    "dct-pixel-space": (DCT, None),
    "h3-video": (H3_VIDEO, CONFIG),
    "h3-video-under-comfys-prefix": ({f"model.diffusion_model.{key}": shape
                                      for key, shape in H3_VIDEO.items()}, CONFIG),
    "dct-under-comfys-prefix": ({f"model.diffusion_model.{key}": shape
                                 for key, shape in DCT.items()}, None),
    "latent-zimage + aux.audio_patch_proj.weight": ({**ZIMAGE, "aux.audio_patch_proj.weight": [1]}, None),
    "h3-video + aux.local_decoder.x.weight": ({**H3_VIDEO, "aux.local_decoder.x.weight": [1]}, CONFIG),
}


@pytest.mark.parametrize("case", sorted(SOUND))
def test_a_sound_file_keeps_no_verdict(models, case):
    """No new refusal for a file ComfyUI builds as a family that renders: latent and DCT
    Z-Image, H3 video, with ComfyUI's own prefix, and with a stray tensor of the other
    family's needle."""
    names, metadata = SOUND[case]
    _assert_on_both_paths(models, _write(models, "sound.safetensors", names, metadata), None)


@pytest.mark.parametrize("missing", gate_keys._H3_KEYS)
def test_one_h3_key_alone_draws_no_image_refusal(models, missing):
    """ComfyUI builds MiniMax H3 from both keys together. A file with one of them, no
    audio path and no sigma grid is no image model to ComfyUI or to the detector, so the
    family's header refusals stay out."""
    names = {key: shape for key, shape in H3_IMAGE.items() if key != missing}
    assert not gate_keys.reads_minimax_h3_image(list(names), frozenset(names))
    _assert_on_both_paths(models, _write(models, "one-key.safetensors", names, None), None)


# Real ComfyUI, CPU only. Keep these last in the file (the conftest comfy guard).
def _comfy_builds(comfy, names: dict[str, list[int]], metadata) -> str | None:
    """ComfyUI's ``image_model`` for these names, through its own prefix strip, or None."""
    import torch

    state = {name: torch.empty(shape, device="meta") for name, shape in names.items()}
    prefix = comfy.model_detection.unet_prefix_from_state_dict(state)
    stripped = comfy.utils.state_dict_prefix_replace(dict(state), {prefix: ""}, filter_keys=True)
    state = stripped if len(stripped) > 0 else state
    config = comfy.model_detection.detect_unet_config(state, "", metadata=metadata)
    if config is None:
        return None
    kind = config.get("image_model")
    return "latent-zimage" if kind == "lumina2" and config.get("z_image_modulation") else kind


def _comfy_names_the_image_family(comfy) -> bool:
    """Whether this ComfyUI builds ``minimax_h3_image`` from the gate's keys. It may not
    only on a ComfyUI known to predate 0.45.0, which added the family; an unknown or newer
    version must, so a ComfyUI that renames the family's keys fails here, not skips."""
    names = _comfy_builds(comfy, H3_IMAGE, CONFIG) == "minimax_h3_image"
    version = _comfy_version(_comfy_checkout()[0])
    assert names or (version is not None and version < _FAMILY_SINCE), (
        f"ComfyUI {version} no longer builds minimax_h3_image from the keys the gate anchors on")
    return names


def test_the_gate_refuses_wherever_comfy_still_builds_the_refused_model(real_comfy, tmp_path):
    """The invariant, against ComfyUI's own detection, over every file above and every
    prefix: where ComfyUI builds latent Z-Image from a file that carries the L2P decoder,
    the gate refuses #91; where it builds ``minimax_h3_image``, the gate answers what it
    answers for the same file without the extra tensor (#149 without the grid, the
    family's answer with it); and a sound file ComfyUI builds as H3 video or pixel-space
    Z-Image passes."""
    files: dict[str, tuple[dict[str, list[int]], dict[str, str] | None, int | None]] = {}
    for base, (names, metadata, number) in BASES.items():
        files[base] = (names, metadata, number)
        for extra, shape in NAMED_EXTRAS.items():
            files[f"{base} + {extra}"] = ({**names, extra: shape}, metadata, number)
        for label, extra in ROW_EXTRAS:
            files[f"{base} + row {label}"] = ({**names, **extra}, metadata, number)
        key = "dec_net.cond_embed.weight" if base == "l2p" else "audio_patch_proj.weight"
        for prefix in detect._PREFIXES:
            files[f"{base} + {prefix}{key}"] = ({**names, prefix + key: [1]}, metadata, number)
    files.update({label: (names, metadata, None) for label, (names, metadata) in SOUND.items()})
    has_image_family = _comfy_names_the_image_family(real_comfy)
    held = {"l2p": 0, "image": 0, "sound": 0}
    for label, (names, metadata, number) in files.items():
        built = _comfy_builds(real_comfy, names, metadata)
        path = tmp_path / "file.safetensors"
        _write(tmp_path, path.name, names, metadata)
        found = upstream_gate.refusal_for(str(path))
        answer = None if found is None else found[1]
        carries_l2p = any("local_decoder." in name for name in names)
        if built == "latent-zimage" and carries_l2p and label not in SOUND:
            assert answer == L2P_NUMBER, label
            held["l2p"] += 1
        elif built == "minimax_h3_image":
            assert answer == number, label
            held["image"] += 1
        elif label in SOUND:
            assert found is None, label
            assert built in ("latent-zimage", "zimage_pixel", "minimax_h3"), (label, built)
            held["sound"] += 1
    assert held["l2p"] >= 30 and held["sound"] == len(SOUND), held
    assert held["image"] >= 60 or not has_image_family, held


def test_comfy_tests_the_exact_keys_the_gate_anchors_on(real_comfy):
    """The four facts ``gate_keys`` rests on, read from ComfyUI's detection itself: one
    key off and ComfyUI stops building the model, or builds the other one."""
    assert _comfy_builds(real_comfy, L2P, None) == "latent-zimage"
    assert _comfy_builds(real_comfy, DCT, None) == "zimage_pixel"
    without = {key: shape for key, shape in L2P.items() if key != gate_keys._LUMINA2_KEYS[1]}
    assert _comfy_builds(real_comfy, without, None) != "latent-zimage"
    assert _comfy_builds(real_comfy, H3_VIDEO, CONFIG) == "minimax_h3"
    if not _comfy_names_the_image_family(real_comfy):
        pytest.skip("this ComfyUI predates MiniMax-H3-Image (ComfyUI 0.45.0 added it)")
    renamed = {**H3_IMAGE, "aux.audio_patch_proj.weight": [1]}
    assert _comfy_builds(real_comfy, renamed, CONFIG) == "minimax_h3_image"
