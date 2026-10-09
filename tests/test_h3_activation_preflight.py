"""The MiniMax H3 activation-footprint guard: h3_rows, h3_calibration and
h3_activation (docs/TROUBLESHOOTING.md #56).

CPU only, with comfy present only as small stubs. The three modules import no
torch and do integer arithmetic over duck-typed shapes, so the 2026-08-05 kill
and the hardware-proven world-2 run at the same shape both replay here.

Provenance for every number below:

* the two committed packed totals, docs/VALIDATION.md: Config A 512x320x5 is
  356 rows, Config B 1344x768x124 is 37,730 rows;
* the staged artifact sizes, byte exact, from the same file;
* the 2026-08-05 incident (docs/TROUBLESHOOTING.md #56): a Config B
  single-rank bf16 render recorded 103.3 GiB of anon RSS and was OOM-killed
  twice;
* the hardware-proven world-2 uly2 run at Config B, 727.6 s bf16, and the
  single-GPU Config B int8-convrot completion, both from docs/VALIDATION.md.

The acceptance surface is those last three: the shape that was killed must
refuse, and the two shapes that completed on this rig must not. Every
calibration constant is pinned against the window those requirements bound, so
an edit to the block factor fails here with the reason.
"""
from __future__ import annotations

import ast
import json
import math
import sys
import types
from pathlib import Path

import pytest

from dgx_monarch import (
    driver_footprint,
    h3_activation,
    h3_calibration,
    h3_rows,
    mesh_safety,
)
from dgx_monarch.nodes import loader_graph, loader_preflight, render_preflight
from dgx_monarch.refusal import GUARDS, parse_refusal_tag

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "dgx_monarch"

_GIB = 2 ** 30

# --- staged artifacts, byte exact (docs/VALIDATION.md) -----------------------
DIT_REF2VA_BF16 = 66_280_487_368          # 61.73 GiB, the 2026-08-05 kill
DIT_REF2VA_INT8 = 34_038_894_550          # 31.70 GiB, non-pruned int8-convrot
DIT_REF2VA_INT8_PRUNED = 20_970_379_616   # 19.53 GiB
TE_NVFP4 = 15_687_142_551                 # 14.61 GiB on disk, 16.80 charged
VAE_VIDEO = 5_207_808_496
VAE_AUDIO = 605_254_808

UNET_NAME = "minimax_h3_ref2va_bf16.safetensors"
TE_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_VIDEO_NAME = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO_NAME = "minimax_h3_audio_vae_fp32.safetensors"

# 65 characters. `unet_name` is comfy's path relative to diffusion_models, so
# a subfoldered, quant-suffixed name is the ordinary case, and the Gate-leg
# truncation test pins against it.
LONG_UNET_NAME = (
    "MiniMax-H3/minimax_h3_ref2va_bf16_convrot_int8_pruned.safetensors")

# --- h3_calibration.py postures, per rank on a 121.6 GiB DGX Spark ----------
# RESIDENT: an explicit preset (or a memo) already put the DiT, the text
# encoder and both VAEs in the reading. The vanilla single-process bf16
# reference run left a 28.4 GiB MemAvailable floor (docs/VALIDATION.md).
# h3_calibration.py reads it as the post-render floor at Config A, with about
# 4 GiB of activations live, so dispatch sees 32.4 GiB.
MEM_RESIDENT = int(32.4 * _GIB)
# PENDING: `auto`, so only the text encoder (16.7 GiB resident, not its 14.61
# GiB file) and the two VAEs (5.8 is their files' decimal-GB size, 5.41 GiB)
# are in the reading. That run's 114.7 GiB idle minus the two is 92.2 GiB.
MEM_PENDING = int(92.2 * _GIB)
# The single-GPU Config B int8-convrot completion. docs/VALIDATION.md (H3
# promotion, Leg 1) names pruned checkpoints for that stack; this fixture, like
# h3_calibration.py, charges the larger non-pruned file: 31.70 + 16.7 + 5.8 =
# 54.2 GiB resident, and 114.7 minus that is 60.5 GiB.
MEM_SINGLE_RANK_INT8 = int(60.5 * _GIB)
RESERVE = 4 * _GIB

# --- the two committed shapes ----------------------------------------------
CONFIG_A = (512, 320, 5)
CONFIG_B = (1344, 768, 124)
CONFIG_A_STREAMS = (320, 16, 20)      # video, audio, text
CONFIG_B_STREAMS = (37_296, 414, 20)
CONFIG_A_TOTAL = sum(CONFIG_A_STREAMS)        # 356
CONFIG_B_TOTAL = sum(CONFIG_B_STREAMS)        # 37,730
FRAME_ROWS_A = 160        # ceil(20/2) * ceil(32/2)
FRAME_ROWS_B = 1_008      # ceil(48/2) * ceil(84/2)

# The 2026-08-05 kill as a slope: 103.3 GiB anon minus 61.7 GiB of slab
# weights and 16.7 GiB of resident text encoder leaves 24.9 GiB of activations
# and runtime at 37,730 rows. It is a lower bound taken at a kill, so the
# shipped constant sits above it.
INCIDENT_UNATTRIBUTED = int(24.9 * _GIB)


def _rows(**fields: int) -> h3_rows.H3Rows:
    return h3_rows.H3Rows(resolved=True, **fields)


ROWS_A = _rows(video=320, audio=16, text=20, frame_rows=FRAME_ROWS_A, latent_t=2)
ROWS_B = _rows(video=37_296, audio=414, text=20, frame_rows=FRAME_ROWS_B,
               latent_t=37)


@pytest.fixture(autouse=True)
def _clean_module_state(monkeypatch):
    """Reset every once-only log flag and kill switch, which are process
    state, so test order cannot decide a verdict. Force gpu_is_integrated on:
    the probe returns False on a GPU-less runner, where a test that expects a
    refusal would fail."""
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.delenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, raising=False)
    monkeypatch.delenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, raising=False)
    monkeypatch.setattr(driver_footprint, "_DISABLE_LOGGED", False)
    for name, value in list(vars(h3_activation).items()):
        if "LOGGED" in name.upper() and isinstance(value, bool):
            monkeypatch.setattr(h3_activation, name, False)
    yield


def _estimate(rows: h3_rows.H3Rows, *, sp: int = 1, weight: int = 0,
              resident: bool = True, mem: int = MEM_RESIDENT,
              reserve: int = RESERVE) -> h3_activation.H3ActivationEstimate:
    """The keyword arguments `preflight_h3_activation_for_request` passes,
    except `weights_note`."""
    return h3_activation.estimate_h3_activation(
        rows=rows, sp_degree=sp, weight_bytes=weight,
        weights_resident=resident, mem_available=mem, reserve=reserve)


# 1. The row algebra, against the two committed packed totals.

class _Fake:
    """A tensor stand-in: `shape` is all h3_rows reads from it."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


class _Nested:
    """comfy's NestedTensor duck: `.shape` is tensors[0].shape, and the audio
    stream is the second tensor and reachable no other way."""

    def __init__(self, video: tuple[int, ...], audio: tuple[int, ...] | None = None):
        self.tensors = [_Fake(video)] + ([_Fake(audio)] if audio else [])
        self.shape = video


def _latent(canvas: tuple[int, int, int], *, audio: bool = True) -> dict:
    width, height, length = canvas
    latent_t = h3_rows.latent_t_for_frames(h3_rows.align_frame_count(length))
    audio_t = h3_rows.audio_t_for_frames(h3_rows.align_frame_count(length))
    video_shape = (1, 24, latent_t, height // 16, width // 16)
    return {"samples": _Nested(video_shape,
                               (1, 32, 2, audio_t) if audio else None)}


def _request(*, text: int = 20, extras: object | None = None,
             entries: list | None = None) -> dict:
    if entries is not None:
        return {"positive": entries}
    return {"positive": [[_Fake((1, text, 4096)), extras if extras is not None else {}]]}


def test_config_a_rows_match_the_committed_packed_total():
    """512x320x5 with a 20-token prompt is 356 rows, split 320/16/20: the
    packed total the 2026-08-04 promotion campaign recorded (docs/VALIDATION.md)."""
    rows = h3_rows.rows_from_request(_request(), _latent(CONFIG_A))
    if (rows.video, rows.audio, rows.text) != CONFIG_A_STREAMS:
        pytest.fail(f"Config A split is {(rows.video, rows.audio, rows.text)}, "
                    f"expected {CONFIG_A_STREAMS}")
    if rows.total != CONFIG_A_TOTAL or not rows.resolved:
        pytest.fail(f"Config A total is {rows.total}, expected {CONFIG_A_TOTAL}")
    if rows.frame_rows != FRAME_ROWS_A or rows.latent_t != 2:
        pytest.fail(f"Config A grid is {rows.frame_rows} x {rows.latent_t}")


def test_config_b_rows_match_the_committed_packed_total():
    """1344x768x124 is 37,730 rows, split 37,296/414/20: the shape whose
    single-rank render was killed twice and whose world-2 run completed."""
    rows = h3_rows.rows_from_request(_request(), _latent(CONFIG_B))
    if (rows.video, rows.audio, rows.text) != CONFIG_B_STREAMS:
        pytest.fail(f"Config B split is {(rows.video, rows.audio, rows.text)}, "
                    f"expected {CONFIG_B_STREAMS}")
    if rows.total != CONFIG_B_TOTAL:
        pytest.fail(f"Config B total is {rows.total}, expected {CONFIG_B_TOTAL}")
    if rows.frame_rows != FRAME_ROWS_B or rows.latent_t != 37:
        pytest.fail(f"Config B grid is {rows.frame_rows} x {rows.latent_t}")


def test_frame_rows_use_ceiling_halving_of_the_even_latent_grid():
    """comfy rounds lat_h/lat_w up to even before packing and _frame_grid then
    floor-divides by 2, so the composition is exactly ceil(lat/2). The H3
    canvas widgets step by 32, so every graph the nodes can produce lands on an
    even latent grid where ceil equals floor; the ceil form stays correct for a
    hand-built canvas that does not."""
    for lat_h, lat_w, want in ((20, 32, 160), (48, 84, 1008), (21, 21, 121),
                               (1, 1, 1), (3, 5, 6)):
        got = h3_rows.frame_rows(lat_h, lat_w)
        if got != want:
            pytest.fail(f"frame_rows({lat_h}, {lat_w}) is {got}, expected {want}")


def test_latent_t_and_frames_round_trip():
    """frames_for_latent_t inverts latent_t_for_frames on the 17k+5 grid, as
    driver_footprint._entry_geometry does. h3_rows restates the formula to stay
    stdlib-only, so this pins the two copies to each other."""
    for frames in (5, 22, 39, 56, 73, 90, 107, 124):
        latent_t = h3_rows.latent_t_for_frames(frames)
        if h3_rows.frames_for_latent_t(latent_t) != frames:
            pytest.fail(f"latent_t {latent_t} did not invert back to {frames}")
        entry = {"kind": "video", "latent_t": latent_t,
                 "latent_h": 20, "latent_w": 32}
        recovered, _h, _w = driver_footprint._entry_geometry(entry)
        if recovered != frames:
            pytest.fail(f"driver_footprint recovered {recovered} frames for "
                        f"latent_t {latent_t}, h3_rows says {frames}")
    for length, aligned in ((1, 5), (5, 5), (6, 22), (22, 22), (124, 124),
                            (125, 141)):
        if h3_rows.align_frame_count(length) != aligned:
            pytest.fail(f"align_frame_count({length}) is "
                        f"{h3_rows.align_frame_count(length)}, expected {aligned}")


def test_keyframe_rows_use_the_target_grid():
    """Keyframe blocks carry `resolved_frame_index` and a `latent` the node has
    already encoded onto the target canvas: it pops the raw image and encodes
    it before it sets these values, so the block that reaches this call never
    holds an `image`. Charging the target frame_rows per latent frame is
    therefore exact, and the block's own trailing dims are the target's. A
    first-and-last-frame guide is one latent frame each, which is this case."""
    extras = {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": _Fake((1, 24, 1, 48, 84))},
                                    {"resolved_frame_index": 123, "latent": _Fake((1, 24, 1, 48, 84))}]}
    rows = h3_rows.rows_from_request(_request(extras=extras), _latent(CONFIG_B))
    if rows.cond != 2 * FRAME_ROWS_B:
        pytest.fail(f"two keyframes must add {2 * FRAME_ROWS_B} rows, got {rows.cond}")
    if rows.total != CONFIG_B_TOTAL + 2 * FRAME_ROWS_B:
        pytest.fail(f"keyframe rows must reach the total: {rows.total}")


def test_anchored_guide_clip_charges_every_latent_frame():
    """An anchored guide can be a clip, not just a still: comfy crops the batch
    to 1 or 17k + 5 frames and encodes all of it, so the latent's frame count
    is the multiplier. One grid per guide would undercount a 39-frame clip (12
    latent frames) by eleven twelfths of its rows."""
    extras = {"minimax_keyframes": [{"resolved_frame_index": 0,
                                     "latent": _Fake((1, 24, 12, 48, 84))}]}
    rows = h3_rows.rows_from_request(_request(extras=extras), _latent(CONFIG_B))
    if rows.cond != 12 * FRAME_ROWS_B:
        pytest.fail(f"a 12-latent-frame guide clip is {12 * FRAME_ROWS_B} rows, "
                    f"got {rows.cond}")


def test_anchored_audio_guide_charges_two_rows_per_latent_frame():
    """An audio guide packs channel-major stereo, two rows per audio latent
    frame, and occupies no video grid. Charging it a target frame grid would
    overcount, the direction that can refuse a render which fits."""
    extras = {"minimax_keyframes": [{"resolved_frame_index": 24,
                                     "audio_latent": _Fake((1, 32, 2, 80))}]}
    rows = h3_rows.rows_from_request(_request(extras=extras), _latent(CONFIG_B))
    if rows.cond != 2 * 80:
        pytest.fail(f"an 80-frame audio guide is {2 * 80} rows, got {rows.cond}")


def test_anchored_guide_carrying_both_streams_charges_both():
    """One node can anchor an image and a soundtrack at the same frame, and
    chained nodes accumulate. Both stream charges add, per guide."""
    extras = {"minimax_keyframes": [
        {"resolved_frame_index": 0, "latent": _Fake((1, 24, 1, 48, 84))},
        {"resolved_frame_index": 90, "latent": _Fake((1, 24, 5, 48, 84)),
         "audio_latent": _Fake((1, 32, 2, 20))},
    ]}
    rows = h3_rows.rows_from_request(_request(extras=extras), _latent(CONFIG_B))
    want = FRAME_ROWS_B + 5 * FRAME_ROWS_B + 2 * 20
    if rows.cond != want:
        pytest.fail(f"two chained guides are {want} rows, got {rows.cond}")


def test_guide_without_readable_latents_charges_nothing():
    """Permissive, as the reference-block rule is: a block whose shapes cannot
    be read undercounts rather than refusing a render that fits. A guide dict
    carrying neither latent is the degenerate case."""
    extras = {"minimax_keyframes": [{"resolved_frame_index": 3},
                                    {"resolved_frame_index": 4, "latent": None},
                                    "junk"]}
    rows = h3_rows.rows_from_request(_request(extras=extras), _latent(CONFIG_A))
    if rows.cond != 0:
        pytest.fail(f"unreadable guides must charge nothing, got {rows.cond}")
    if rows.total != CONFIG_A_TOTAL:
        pytest.fail(f"the total must stay the plain canvas: {rows.total}")


def test_reference_blocks_count_image_video_and_audio_rows():
    """One image block, one video_audio block and one standalone audio block,
    each at its own arithmetic (PackedLayout.__init__)."""
    extras = {"minimax_refs": [
        {"kind": "image", "latent_h": 20, "latent_w": 32},
        {"kind": "video_audio", "latent_t": 2, "latent_h": 20, "latent_w": 32,
         "ref_audio_t": 8},
        {"kind": "audio", "ref_audio_t": 80},
    ]}
    rows = h3_rows.rows_from_request(_request(extras=extras), _latent(CONFIG_A))
    want = 160 + (2 * 8 + 2 * 160) + (2 * 80)
    if rows.ref != want:
        pytest.fail(f"reference rows are {rows.ref}, expected {want}")
    if rows.total != CONFIG_A_TOTAL + want:
        pytest.fail(f"reference rows must reach the total: {rows.total}")


def test_reference_rows_floor_the_latent_grid():
    """The target grid rounds up to even before packing; reference blocks do
    not. They reach _frame_grid with their raw wire values, and _frame_grid
    floors. On an odd reference latent dim a ceiling would over-count, which is
    the refusing direction class C forbids."""
    extras = {"minimax_refs": [{"kind": "image", "latent_h": 21, "latent_w": 21}]}
    rows = h3_rows.rows_from_request(_request(extras=extras), _latent(CONFIG_A))
    if rows.ref != 100:
        pytest.fail(f"an odd 21x21 reference latent is (21//2)^2 = 100 rows, "
                    f"got {rows.ref}")


def test_audio_rows_come_from_the_second_nested_tensor():
    """NestedTensor.shape reports the video tensor only. The audio stream is
    the second entry of `.tensors` and is reachable no other way."""
    latent = {"samples": _Nested((1, 24, 37, 48, 84), (1, 32, 2, 207))}
    rows = h3_rows.rows_from_request(_request(), latent)
    if rows.audio != 2 * 207:
        pytest.fail(f"audio rows are {rows.audio}, expected {2 * 207}")


def test_audio_rows_fall_back_to_the_frame_grid_without_an_audio_tensor():
    """pack_conditioning and tree_to_cpu are the only things between the node
    and this call, and a future container change must degrade, not crash. The
    derived path is exact for every shape the stock nodes produce."""
    for canvas, want in ((CONFIG_A, 2 * 8), (CONFIG_B, 2 * 207)):
        rows = h3_rows.rows_from_request(_request(), _latent(canvas, audio=False))
        if rows.audio != want:
            pytest.fail(f"derived audio rows for {canvas} are {rows.audio}, "
                        f"expected {want}")


def test_text_rows_are_the_max_positive_conditioning_length():
    """Max, not first: cond and uncond are separate model calls and the
    negative mirrors the positive per the node, so the positive bounds the
    peak, and a scheduled-conditioning graph carries several entries whose
    lengths differ."""
    entries = [[_Fake((1, 20, 4096)), {}], [_Fake((1, 77, 4096)), {}],
               [_Fake((1, 12, 4096)), {}]]
    rows = h3_rows.rows_from_request(_request(entries=entries), _latent(CONFIG_A))
    if rows.text != 77:
        pytest.fail(f"text rows are {rows.text}, expected the longest entry, 77")


def test_extras_are_scanned_across_every_positive_entry():
    """The scan reads every entry, not just the first: an entry with no extras
    must not hide the one that has them. The SCAIL extractor states the same
    reason for the same scan."""
    entries = [[_Fake((1, 20, 4096)), {}],
               [_Fake((1, 20, 4096)),
                {"minimax_refs": [{"kind": "audio", "ref_audio_t": 80}]}]]
    rows = h3_rows.rows_from_request(_request(entries=entries), _latent(CONFIG_A))
    if rows.ref != 160:
        pytest.fail(f"a later entry's extras must be counted: {rows.ref}")


def test_extras_are_maxed_across_entries_never_summed():
    """comfy's conditioning helper does not append by default and this
    family's nodes never ask it to, so the same extras list lands on every
    positive entry. Summing would multiply one reference block by the entry
    count: a two-entry positive carrying one Config B video_audio reference
    would add 37,710 phantom rows, 38.8 GiB at one rank, enough to refuse a
    render that fits. Each entry is its own model call building its own packed
    layout, so the peak is one layout and the count must not move with the
    entry count."""
    block = {"kind": "video_audio", "latent_t": 37, "latent_h": 48,
             "latent_w": 84, "ref_audio_t": 207}
    extras = {"minimax_refs": [block],
              "minimax_keyframes": [{"resolved_frame_index": 0,
                                     "latent": _Fake((1, 24, 1, 48, 84))}]}
    counts = set()
    for entry_count in (1, 2, 3):
        entries = [[_Fake((1, 20, 4096)), dict(extras)]
                   for _ in range(entry_count)]
        rows = h3_rows.rows_from_request(_request(entries=entries),
                                         _latent(CONFIG_B))
        counts.add((rows.ref, rows.cond))
    if len(counts) != 1:
        pytest.fail(f"the packed count moved with the entry count: {counts}")
    ref, cond = counts.pop()
    if ref != 2 * 207 + 37 * FRAME_ROWS_B:
        pytest.fail(f"one video_audio block is 2*207 + 37*1008 rows: {ref}")
    if cond != FRAME_ROWS_B:
        pytest.fail(f"one keyframe is one target frame: {cond}")


def test_several_reference_blocks_in_one_entry_still_sum():
    """Max across entries, sum within one: every block in one entry's list
    packs into that entry's own sequence, so they add there."""
    extras = {"minimax_refs": [{"kind": "audio", "ref_audio_t": 80},
                               {"kind": "audio", "ref_audio_t": 40}]}
    entries = [[_Fake((1, 20, 4096)), extras], [_Fake((1, 20, 4096)), extras]]
    rows = h3_rows.rows_from_request(_request(entries=entries), _latent(CONFIG_A))
    if rows.ref != 2 * 80 + 2 * 40:
        pytest.fail(f"blocks inside one entry must sum: {rows.ref}")


def test_malformed_extras_degrade_to_the_readable_terms():
    """Every parse surprise is a zero, never an exception, and never an
    over-count: a block whose ints are absent contributes nothing, which is
    the permissive direction."""
    cases: list[object] = [
        None, "junk", 7, _Fake((1, 2)),
        {"minimax_refs": None},
        {"minimax_refs": ["not a mapping", 7]},
        {"minimax_refs": [{"kind": "video"}]},          # no ints at all
        {"minimax_keyframes": "junk"},
    ]
    for extras in cases:
        rows = h3_rows.rows_from_request(_request(extras=extras), _latent(CONFIG_A))
        if rows.ref or rows.cond:
            pytest.fail(f"extras {extras!r} over-counted: ref={rows.ref}, "
                        f"cond={rows.cond}")
        if rows.video != 320 or rows.audio != 16:
            pytest.fail(f"extras {extras!r} lost the readable terms: {rows}")


def test_unresolvable_latent_returns_an_unresolved_row_count():
    """No video geometry means charge nothing, which the caller reads off
    `resolved` rather than off a zero total."""
    for latent in (None, {}, "junk", 7, {"samples": None},
                   {"samples": _Fake(None)},
                   {"samples": _Fake((1, 24, 2))}):       # fewer than 5 dims
        rows = h3_rows.rows_from_request(_request(), latent)
        if rows.resolved or rows.total:
            pytest.fail(f"latent {latent!r} must be unresolved, got {rows}")


def test_an_unreadable_latent_dimension_invents_no_video_rows():
    """A shape whose entries are not ints is a shape this module knows nothing
    about, so the video term must be omitted rather than guessed. Omitting is
    the permissive direction; guessing is the refusing one."""
    rows = h3_rows.rows_from_request(
        _request(), {"samples": _Fake((1, 24, 2, "x", 32))})
    if rows.video:
        pytest.fail(f"an unreadable spatial dim must charge no video rows: {rows}")


def test_batch_axis_is_absent_because_this_family_is_batch_one():
    """h3_rows has no batch multiplier, by structure: the adapter refuses cfg>1
    and dp>1 before dispatch and refuses a batched packed forward, so a batch
    axis above 1 cannot reach the sampler."""
    from dgx_monarch.adapters.minimax_h3 import minimax_h3_topology_would_reject

    if not minimax_h3_topology_would_reject(1, 2):
        pytest.fail("dp>1 must be refused for this family")
    if not minimax_h3_topology_would_reject(2, 1):
        pytest.fail("cfg>1 must be refused for this family")
    batched = {"samples": _Nested((4, 24, 2, 20, 32), (4, 32, 2, 8))}
    rows = h3_rows.rows_from_request(_request(), batched)
    if rows.total != CONFIG_A_TOTAL:
        pytest.fail(f"a batch axis must not multiply the packed total: "
                    f"{rows.total}")


def test_ref_extras_keys_match_the_driver_stack_profile():
    """h3_rows hard-codes the two extras keys rather than importing them, to
    stay stdlib-only. The two spellings must not drift."""
    profile = driver_footprint.DRIVER_STACK_FAMILIES["minimax_h3"]
    source = (SRC / "h3_rows.py").read_text()
    for key in profile.ref_extras_keys:
        if key not in source:
            pytest.fail(f"h3_rows.py must spell {key!r} exactly as the driver "
                        "stack profile does")
    if set(profile.ref_extras_keys) != {"minimax_refs", "minimax_keyframes"}:
        pytest.fail(f"the profile's keys moved: {profile.ref_extras_keys}")


# 2. Calibration: the constants, and the window they have to sit in.

def test_the_calibration_constants_are_pinned():
    """Changing any of these is a recalibration, which needs its own
    evidence."""
    expected = {
        "H3_HIDDEN_DIM": 5_376,
        "H3_SHARDED_ROW_BYTES": 1_032_192,
        "H3_REPLICATED_ROW_BYTES": 73_728,
    }
    for name, want in expected.items():
        got = getattr(h3_calibration, name)
        if got != want:
            pytest.fail(f"{name} is {got}, was calibrated at {want}; see "
                        "docs/TROUBLESHOOTING.md #56 before moving it")
    if h3_activation.H3_SHARDED_ROW_BYTES != 96 * h3_calibration.H3_HIDDEN_DIM * 2:
        pytest.fail("the sharded row cost must stay block_factor x hidden x 2")
    if h3_activation.H3_FAMILY != "minimax_h3":
        pytest.fail("the guard is family scoped to minimax_h3")


def _calibration_window() -> tuple[float, float]:
    """The intersection of every constraint the rig records, recomputed from
    the committed totals rather than from h3_calibration.py's arithmetic.

    Two requirements bound the sharded row cost S from opposite sides, and a
    third comes from the one single-rank Config B run that completed:

    * Config B at one rank at bf16 must refuse, in both postures: it is the
      shape that was killed twice.
    * Config B at world-2 uly2 must pass. It is hardware-proven at 727.6 s.
    * Config B at one rank on int8-convrot must pass under an explicit preset,
      charged at the non-pruned file (see MEM_SINGLE_RANK_INT8).
    """
    total = CONFIG_B_TOTAL
    per_rank_2 = -(-total // 2)
    replicated = h3_activation.H3_REPLICATED_ROW_BYTES
    floors: list[float] = []
    ceilings: list[float] = []
    for usable, weight in ((MEM_RESIDENT - RESERVE, 0),
                           (MEM_PENDING - RESERVE, DIT_REF2VA_BF16)):
        budget = usable - weight
        floors.append(budget / total - replicated)                  # refuse at N=1
        ceilings.append((budget - total * replicated) / per_rank_2)  # pass at N=2
    # The single-rank int8 completion ran an explicit preset, so its weights
    # were already inside the reading and nothing is charged for them.
    ceilings.append((MEM_SINGLE_RANK_INT8 - RESERVE) / total - replicated)
    return max(floors), min(ceilings)


def test_the_block_factor_sits_inside_its_calibration_window():
    """A new measurement moves the constant by changing an inequality here."""
    floor, ceiling = _calibration_window()
    sharded = h3_activation.H3_SHARDED_ROW_BYTES
    if not floor < sharded <= ceiling:
        pytest.fail(
            f"H3_SHARDED_ROW_BYTES {sharded} left its calibration window "
            f"({floor:.0f}, {ceiling:.0f}]: below it the Config B single-rank "
            "bf16 kill passes, above it a hardware-proven run refuses")
    centre = math.sqrt(floor * ceiling)
    if not 0.85 * centre <= sharded <= 1.15 * centre:
        pytest.fail(f"the window is a factor of two wide by construction, so "
                    f"the defensible pick is its geometric centre {centre:.0f}; "
                    f"{sharded} is not near it")


def test_the_block_factor_is_above_the_anatomy_floor_and_the_measured_slope():
    """Two lower bounds, both below the shipped constant.

    The anatomy floor is what one H3 block holds at once during an
    autograd-free forward: x, h, fc1 out, the swiglu product and fc2 out. In
    comfy.ldm.minimax.model.MLP, fc1 is Linear(hidden, 2 x ffn) and swiglu
    halves its output; at H3's hidden 5,376 and ffn 14,336 the five sum to
    1 + 1 + 16/3 + 8/3 + 1 = 11.0 x hidden elements. mesh_safety's generic
    placeholder factor (8) is below that floor, one reason it cannot be reused
    here.

    The measured slope is the 2026-08-05 kill's own unattributed bytes, and a
    constant calibrated to it would not have intercepted the incident it was
    calibrated on: it is a lower bound taken at a kill.
    """
    hidden = h3_calibration.H3_HIDDEN_DIM
    anatomy = 11 * hidden * 2
    sharded = h3_activation.H3_SHARDED_ROW_BYTES
    if sharded <= anatomy:
        pytest.fail(f"{sharded} is at or below the {anatomy}-byte anatomy floor")
    if mesh_safety._ACTIVATION_BLOCK_FACTOR * hidden * 2 >= anatomy:
        pytest.fail("the SCAIL placeholder is no longer below H3's anatomy "
                    "floor; restate why it still cannot be reused")
    measured = INCIDENT_UNATTRIBUTED / CONFIG_B_TOTAL - h3_activation.H3_REPLICATED_ROW_BYTES
    if sharded <= measured:
        pytest.fail(f"{sharded} is at or below the measured slope {measured:.0f}")
    if sharded / measured < 1.5:
        pytest.fail(f"the margin over the measured slope is only "
                    f"{sharded / measured:.2f}x; at the measurement itself the "
                    "single-rank Config B estimate equals the measured value "
                    "and passes")
    # Why the margin is needed, as arithmetic.
    at_the_measurement = CONFIG_B_TOTAL * (
        int(measured) + h3_activation.H3_REPLICATED_ROW_BYTES)
    if at_the_measurement > MEM_RESIDENT - RESERVE:
        pytest.fail("a constant equal to the measured slope would have "
                    "intercepted the incident; the margin argument needs a "
                    "new reason")


# 3. The estimate: what refuses, what must never refuse, and the per-rank rule.

@pytest.mark.parametrize(("posture", "mem", "weight", "resident"), [
    ("resident", MEM_RESIDENT, DIT_REF2VA_BF16, True),
    ("pending", MEM_PENDING, DIT_REF2VA_BF16, False),
])
def test_config_b_single_rank_bf16_refuses_in_both_postures(
        posture, mem, weight, resident):
    """The 2026-08-05 kill: 103.3 GiB of anon RSS, OOM-killed twice, when no
    capacity boundary the repo owned counted packed rows."""
    estimate = _estimate(ROWS_B, sp=1, weight=weight, resident=resident, mem=mem)
    if estimate.fits:
        pytest.fail(f"{posture}: the incident shape must refuse, "
                    f"{estimate.projected / _GIB:.2f} GiB against "
                    f"{estimate.usable / _GIB:.2f} GiB usable")


@pytest.mark.parametrize(("posture", "mem", "weight", "resident"), [
    ("resident", MEM_RESIDENT, DIT_REF2VA_BF16, True),
    ("pending", MEM_PENDING, DIT_REF2VA_BF16, False),
])
def test_config_b_uly2_bf16_fits_in_both_postures(posture, mem, weight, resident):
    """Hardware-proven at 727.6 s (docs/VALIDATION.md). A false refusal here
    would block the run the guard offers as a remedy: world 2."""
    estimate = _estimate(ROWS_B, sp=2, weight=weight, resident=resident, mem=mem)
    if not estimate.fits:
        pytest.fail(f"{posture}: a hardware-proven run was refused, "
                    f"{estimate.projected / _GIB:.2f} GiB against "
                    f"{estimate.usable / _GIB:.2f} GiB usable")


def test_the_tightest_surviving_cell_is_named_and_pinned():
    """Config B at world 2 under `auto`, at bf16, is the narrowest cell that
    still passes: the load has not happened yet, so the whole 61.7 GiB weight
    charge sits on top of the sharded activations. At this file's 4 GiB
    calibration reserve its margin is 1.07x, and it false-refuses whenever real
    MemAvailable at dispatch falls below about 86.5 GiB; the same shape under an
    explicit `uly2` preset runs the 1.37x resident posture instead. The reserve
    floor has been 5 GiB since 2026-09-04; docs/TROUBLESHOOTING.md #56 gives
    the figures at that floor. The 92.2 GiB MemAvailable the pending posture
    starts from is derived, not measured (see MEM_PENDING); the cell's first
    hardware pass, 2026-08-05, is in docs/TROUBLESHOOTING.md #56.
    """
    estimate = _estimate(ROWS_B, sp=2, weight=DIT_REF2VA_BF16, resident=False,
                         mem=MEM_PENDING)
    if not estimate.fits:
        pytest.fail(f"the tightest surviving cell must still pass: "
                    f"{estimate.projected / _GIB:.2f} GiB against "
                    f"{estimate.usable / _GIB:.2f} GiB usable")
    margin = estimate.usable / estimate.projected
    if not 1.0 < margin < 1.15:
        pytest.fail(f"this cell's margin moved to {margin:.3f}; it was 1.07 and "
                    "docs/TROUBLESHOOTING.md #56 prints that number")
    floor = estimate.projected + estimate.reserve
    if not 86.0 * _GIB < floor < 87.0 * _GIB:
        pytest.fail(f"the MemAvailable this cell needs moved to "
                    f"{floor / _GIB:.1f} GiB; #56 says about 86.5")
    if not _estimate(ROWS_B, sp=2, mem=MEM_RESIDENT).fits:
        pytest.fail("the explicit-preset sibling of this cell must stay the "
                    "comfortable one, or the doc's contrast is wrong")


@pytest.mark.parametrize(("posture", "mem", "resident"), [
    ("resident", MEM_RESIDENT, True), ("pending", MEM_PENDING, False)])
def test_config_a_single_rank_bf16_fits_in_both_postures(posture, mem, resident):
    """356 rows is 0.37 GiB of activations. The guard must never refuse here."""
    estimate = _estimate(ROWS_A, sp=1, weight=DIT_REF2VA_BF16,
                         resident=resident, mem=mem)
    if not estimate.fits:
        pytest.fail(f"{posture}: Config A must never refuse, "
                    f"{estimate.projected / _GIB:.2f} GiB against "
                    f"{estimate.usable / _GIB:.2f} GiB usable")


def test_config_b_single_rank_int8_fits():
    """The guard is not a blanket Config B ban: it refuses one combination of
    weights and activations, not a shape."""
    estimate = _estimate(ROWS_B, sp=1, weight=DIT_REF2VA_INT8_PRUNED,
                         resident=False, mem=MEM_PENDING)
    if not estimate.fits:
        pytest.fail(f"a pruned artifact at Config B must fit: "
                    f"{estimate.projected / _GIB:.2f} GiB against "
                    f"{estimate.usable / _GIB:.2f} GiB usable")


def test_config_b_single_rank_int8_with_an_explicit_preset_fits():
    """The single-GPU Config B int8-convrot completion (see
    MEM_SINGLE_RANK_INT8), charged at the non-pruned file: the tightest
    single-rank pass constraint. Its explicit preset means the weights were
    already resident, so nothing is charged for them."""
    estimate = _estimate(ROWS_B, sp=1, weight=DIT_REF2VA_INT8, resident=True,
                         mem=MEM_SINGLE_RANK_INT8)
    if estimate.weight_bytes:
        pytest.fail("an explicit preset credits the weight term")
    if not estimate.fits:
        pytest.fail(f"a hardware-proven single-rank Config B run was refused: "
                    f"{estimate.projected / _GIB:.2f} GiB against "
                    f"{estimate.usable / _GIB:.2f} GiB usable")


def test_sp_degree_divides_only_the_sharded_term():
    """Sequence parallel shards the packed row axis and nothing else. Weights
    are replicated on every rank under Ulysses, and so is the staging in the
    replicated floor."""
    previous = None
    for sp in (1, 2, 3, 4):
        estimate = _estimate(ROWS_B, sp=sp, weight=DIT_REF2VA_BF16, resident=False,
                             mem=MEM_PENDING)
        if estimate.rows_per_rank != -(-CONFIG_B_TOTAL // sp):
            pytest.fail(f"sp {sp}: rows/rank is {estimate.rows_per_rank}")
        if estimate.sharded_bytes != estimate.rows_per_rank * h3_activation.H3_SHARDED_ROW_BYTES:
            pytest.fail(f"sp {sp}: the sharded term is not rows/rank x S")
        if estimate.weight_bytes != DIT_REF2VA_BF16:
            pytest.fail(f"sp {sp}: sequence parallel must never divide weights")
        if previous is not None and estimate.sharded_bytes >= previous:
            pytest.fail(f"sp {sp}: the sharded term must fall with the degree")
        previous = estimate.sharded_bytes


def test_the_replicated_floor_is_charged_at_the_full_packed_total():
    """Several buffers in the H3 forward are indexed by the full packed total
    on every rank whatever the degree (the fp32 patchify and both patch
    projections, the full-length assembly buffer, the sp_gather output and the
    final layer's fp32 slices). Charging them per per-rank row would credit
    sharding that does not happen."""
    want = CONFIG_B_TOTAL * h3_activation.H3_REPLICATED_ROW_BYTES
    for sp in (1, 2, 4):
        estimate = _estimate(ROWS_B, sp=sp)
        if estimate.replicated_bytes != want:
            pytest.fail(f"sp {sp}: replicated staging is "
                        f"{estimate.replicated_bytes}, expected {want}")
        if estimate.activation_bytes != estimate.sharded_bytes + estimate.replicated_bytes:
            pytest.fail(f"sp {sp}: the two terms must sum to the activations")


def test_the_committed_activation_figures():
    """The activation bytes at the two committed shapes, pinned so a constant
    edit fails here with the figure it moved."""
    if _estimate(ROWS_A, sp=1).activation_bytes != 393_707_520:
        pytest.fail("Config A at one rank is 0.37 GiB of activations")
    if _estimate(ROWS_B, sp=1).activation_bytes != 41_726_361_600:
        pytest.fail("Config B at one rank is 38.86 GiB of activations")
    if _estimate(ROWS_B, sp=2).activation_bytes != 22_254_059_520:
        pytest.fail("Config B at world 2 is 20.73 GiB of activations")


def test_max_total_rows_is_reachable():
    """The offered bound must fit, and one row past it must not."""
    for sp, mem in ((1, MEM_RESIDENT), (2, MEM_RESIDENT), (2, MEM_PENDING)):
        estimate = _estimate(ROWS_B, sp=sp, mem=mem)
        bound = estimate.max_total_rows
        if bound <= 0:
            pytest.fail(f"sp {sp}: no bound to offer at "
                        f"{estimate.usable / _GIB:.2f} GiB usable")
        if not _estimate(_rows(video=bound), sp=sp, mem=mem).fits:
            pytest.fail(f"sp {sp}: the offered bound {bound} does not fit")
        if _estimate(_rows(video=bound + 1), sp=sp, mem=mem).fits:
            pytest.fail(f"sp {sp}: {bound} is not the largest total that fits")


def test_max_total_rows_is_reachable_at_a_budget_that_needs_the_ceiling_correction():
    """`fits` charges ceil(total/N) sharded rows; the closed form solves the
    linear relaxation, and floor-dividing that solution does not undo the
    ceiling. At the budget below the linear answer is 51,701 and it refuses,
    so an operator who shrank to the offered number would be refused again."""
    sp = 2
    usable = 30_494_490_624
    sharded = h3_activation.H3_SHARDED_ROW_BYTES
    replicated = h3_activation.H3_REPLICATED_ROW_BYTES
    linear = usable * sp // (sharded + replicated * sp)
    if linear != 51_701:
        pytest.fail(f"this fixture exists for the off-by-one; linear is {linear}")
    estimate = _estimate(ROWS_B, sp=sp, mem=usable + RESERVE)
    if estimate.usable != usable:
        pytest.fail(f"fixture drift: usable is {estimate.usable}")
    if estimate.max_total_rows != linear - 1:
        pytest.fail(f"the bound must be corrected down to {linear - 1}, got "
                    f"{estimate.max_total_rows}")
    if not _estimate(_rows(video=linear - 1), sp=sp, mem=usable + RESERVE).fits:
        pytest.fail("the corrected bound must fit")
    if _estimate(_rows(video=linear), sp=sp, mem=usable + RESERVE).fits:
        pytest.fail("the uncorrected bound must refuse; that is the defect")


def test_shortfall_is_measured_against_the_unclamped_budget():
    """Below the reserve `usable` clamps at 0, so `projected - usable` would
    understate the lever by exactly the clamp and the message would offer an
    amount that still refuses."""
    tight = _estimate(ROWS_B, sp=1, mem=int(2.0 * _GIB))
    if tight.usable != 0:
        pytest.fail("this fixture must sit below the reserve")
    if tight.shortfall <= tight.projected:
        pytest.fail(f"the shortfall must carry the clamp: {tight.shortfall}")
    healed = _estimate(ROWS_B, sp=1, mem=int(2.0 * _GIB) + tight.shortfall)
    if not healed.fits:
        pytest.fail("the offered shortfall must be the amount that fixes it")
    ordinary = _estimate(ROWS_B, sp=1, mem=MEM_RESIDENT)
    if ordinary.shortfall != ordinary.projected - ordinary.usable:
        pytest.fail("above the reserve the two spellings must agree")


def test_the_activation_term_does_not_move_with_the_weight_term():
    """The estimate is residency blind on the activation axis, which is why
    this refusal is not waivable: slab residency changes when weight bytes
    appear and how they are mapped, never how many activation bytes a packed
    sequence needs."""
    for weight in (0, DIT_REF2VA_BF16, DIT_REF2VA_INT8):
        for resident in (True, False):
            estimate = _estimate(ROWS_B, sp=1, weight=weight, resident=resident,
                                 mem=MEM_RESIDENT)
            if estimate.activation_bytes != _estimate(ROWS_B, sp=1).activation_bytes:
                pytest.fail("the activation term must not move with the weight "
                            f"term ({weight}, resident={resident})")


# 4. The refusal: class C, never bare, every number reachable.

def _refuse(estimate, unet_name: str = UNET_NAME):
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        h3_activation.h3_activation_preflight(estimate, unet_name=unet_name)
    return excinfo.value


def test_refusal_is_class_c_with_the_activation_guard_tag():
    """The guard id comes from the frozen vocabulary. Class C with no card: no
    consent can change this verdict."""
    exc = _refuse(_estimate(ROWS_B, sp=1))
    tag = parse_refusal_tag(str(exc))
    if tag is None:
        pytest.fail(f"the refusal must carry a class tag: {exc}")
    if tag.refusal_class.value != "C":
        pytest.fail(f"expected class C, got {tag.refusal_class}")
    if tag.guard != "activation_footprint_preflight":
        pytest.fail(f"expected the reserved guard id, got {tag.guard}")
    if tag.waivable:
        pytest.fail("residency changes when weight bytes appear, never how "
                    "many activation bytes a shape needs, so this is not "
                    "waivable")
    spec = GUARDS["activation_footprint_preflight"]
    if spec.waivable_now or spec.refusal_class.value != "C":
        pytest.fail(f"the frozen guard spec moved: {spec}")


def test_refusal_is_recognized_as_a_capacity_error():
    """Reusing StockLoadCapacityError keeps the ceremony's capacity auto-skip
    and the Gate-leg classification working unchanged, and a Monarch-wrapped
    copy still parses."""
    exc = _refuse(_estimate(ROWS_B, sp=1))
    if not mesh_safety.is_stock_load_capacity_error(exc):
        pytest.fail("the refusal must classify as a capacity error")
    wrapped = RuntimeError(
        f"Monarch actor failed: StockLoadCapacityError: {exc}")
    if not mesh_safety.is_stock_load_capacity_error(wrapped):
        pytest.fail("a Monarch-wrapped copy must still classify")


def test_refusal_names_every_fitting_option():
    """Class C forbids a bare refusal, so every fitting option stays in the
    message."""
    message = h3_activation.h3_refusal(_estimate(ROWS_B, sp=1), UNET_NAME)
    for option in h3_activation.H3_FITTING_OPTIONS:
        if option not in message:
            pytest.fail(f"missing fitting option {option!r}")
    for phrase in ("What would fit here", "packed total of at most",
                   "more free memory"):
        if phrase not in message:
            pytest.fail(f"missing {phrase!r} from a class C refusal")


def test_refusal_option_list_differs_from_the_driver_profile_on_purpose():
    """The two option lists differ on purpose; do not merge them.

    driver_footprint's H3 profile does not offer splitting across the pair:
    for a weight-bound refusal Ulysses and Ring replicate the checkpoint on
    every rank. FSDP admits this family's bf16 release, but its completed
    2026-09-09 render has no qualifying fidelity comparison (docs/VALIDATION.md).
    For an activation-bound refusal the split halves the sharded term, and
    that path is hardware-proven at this shape.
    """
    driver_options = driver_footprint.DRIVER_STACK_FAMILIES["minimax_h3"].fitting_options
    if h3_activation.SPLIT_SEQUENCE not in h3_activation.H3_FITTING_OPTIONS:
        pytest.fail("the activation refusal must offer the sequence split")
    if any("world 2" in option for option in driver_options):
        pytest.fail("the weight-bound profile must not offer a split that "
                    "provably refuses")
    for shared in (driver_footprint.PRUNED_DIT, driver_footprint.FREE_MEMORY):
        if shared not in h3_activation.H3_FITTING_OPTIONS:
            pytest.fail(f"{shared!r} must be reused verbatim so there is one "
                        "spelling of each option")


def test_refusal_names_the_packed_total_and_the_stream_split():
    """Five streams, named separately: the operator's lever differs per
    stream, and the 2026-08-05 kill passed a boundary that counted only the
    video stream."""
    rows = _rows(video=37_296, audio=414, text=20, cond=2 * FRAME_ROWS_B,
                 ref=160, frame_rows=FRAME_ROWS_B, latent_t=37)
    summary = rows.stream_summary()
    if summary != "37296 video, 414 audio, 20 text, 2016 keyframe, 160 reference":
        pytest.fail(f"the stream summary reads {summary!r}")
    estimate = _estimate(rows, sp=1)
    message = h3_activation.h3_refusal(estimate, UNET_NAME)
    if summary not in message:
        pytest.fail(f"the refusal must carry the stream split: {message}")
    for token in (f"{rows.total} rows", str(estimate.max_total_rows),
                  UNET_NAME, "rows/rank", "replicated"):
        if token not in message:
            pytest.fail(f"the refusal must name {token!r}: {message}")


def test_every_number_the_refusal_offers_is_reachable():
    """Both levers the message offers, taken at their word."""
    estimate = _estimate(ROWS_B, sp=1, weight=DIT_REF2VA_BF16, resident=False,
                         mem=MEM_PENDING)
    bound = estimate.max_total_rows
    at_the_bound = _estimate(_rows(video=bound), sp=1, weight=DIT_REF2VA_BF16,
                             resident=False, mem=MEM_PENDING)
    if not at_the_bound.fits:
        pytest.fail(f"the offered bound of {bound} rows does not fit")
    with_the_shortfall = _estimate(
        ROWS_B, sp=1, weight=DIT_REF2VA_BF16, resident=False,
        mem=MEM_PENDING + estimate.shortfall)
    if not with_the_shortfall.fits:
        pytest.fail("the offered shortfall must be the amount that fixes it")


def test_a_weight_charge_that_alone_overruns_offers_no_row_bound():
    """When the weight term alone clears the budget there is no packed total to
    offer, and printing `~0 rows` would offer a lever that changes nothing.
    The case is reachable on the render path, and the shortfall is then the
    only lever left."""
    estimate = _estimate(ROWS_A, sp=1, weight=DIT_REF2VA_BF16, resident=False,
                         mem=60 * _GIB)
    if estimate.max_total_rows:
        pytest.fail("this test is only worth running with no row bound left")
    message = h3_activation.h3_refusal(estimate, UNET_NAME)
    if "~0 rows" in message:
        pytest.fail(f"a zero row bound must not be offered as a lever: {message}")
    if "no packed total of any length" not in message:
        pytest.fail(f"the message must say no length clears this: {message}")
    if f"{estimate.shortfall / _GIB:.1f} GiB more free memory" not in message:
        pytest.fail(f"the shortfall stays the surviving lever: {message}")


def test_a_gate_leg_truncation_keeps_the_numbers():
    """Raised inside a first-use Gate leg the detail is `repr(exc)[:200]`, not
    the message, so 24 characters are gone before the message starts and the
    class tag takes 57 more. The variant below is the longest realistic one: an
    operator-set reserve, three-digit MemAvailable and projected figures, and a
    65-character subfoldered artifact name."""
    estimate = _estimate(ROWS_B, sp=1, weight=DIT_REF2VA_BF16, resident=False,
                         mem=int(114.5 * _GIB), reserve=48 * _GIB)
    exc = mesh_safety.StockLoadCapacityError(
        h3_activation.h3_refusal(estimate, LONG_UNET_NAME))
    detail = repr(exc)[:200]
    if len(LONG_UNET_NAME) < 60:
        pytest.fail("this test is only worth running at a realistic name length")
    for token in (f"{estimate.projected / _GIB:.1f}",
                  f"{estimate.usable / _GIB:.1f}",
                  f"{estimate.mem_available / _GIB:.1f}",
                  f"{estimate.reserve / _GIB:.1f}"):
        if token not in detail:
            pytest.fail(f"the truncated detail must still carry {token!r}: "
                        f"{detail}")


def test_the_reserve_names_its_source():
    """A large uma_reserve_gb can refuse a hardware-proven shape by design, so
    the message must say where the reserve came from."""
    operator = h3_activation.h3_refusal(
        _estimate(ROWS_B, sp=1, reserve=48 * _GIB), UNET_NAME)
    if "your uma_reserve_gb" not in operator:
        pytest.fail(f"an operator-set reserve must be named: {operator}")
    floor = h3_activation.h3_refusal(_estimate(ROWS_B, sp=1), UNET_NAME)
    if "the driver floor" not in floor:
        pytest.fail(f"the default reserve must be named as the floor: {floor}")


def test_the_refusal_names_the_escape_hatch_and_its_entry():
    """It names the shared activation kill switch and its entry, not the driver
    preflight's variable."""
    message = h3_activation.h3_refusal(_estimate(ROWS_B, sp=1), UNET_NAME)
    if mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV not in message:
        pytest.fail("the refusal must name the shared kill switch")
    if driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV in message:
        pytest.fail("this guard is not stood down by the driver preflight's "
                    "own variable, so naming it would mislead")
    if "docs/TROUBLESHOOTING.md #56" not in message:
        pytest.fail("the refusal must point at its troubleshooting entry")


def test_the_refusal_credits_a_resident_weight_term_in_words():
    """The message says why the weight term is zero, in the words the SCAIL
    guard in mesh_safety uses."""
    credited = h3_activation.h3_refusal(_estimate(ROWS_B, sp=1), UNET_NAME)
    if "resident, credited" not in credited:
        pytest.fail(f"a credited weight term must say so: {credited}")
    charged = h3_activation.h3_refusal(
        _estimate(ROWS_B, sp=1, weight=DIT_REF2VA_BF16, resident=False,
                  mem=MEM_PENDING), UNET_NAME)
    if "61.7" not in charged:
        pytest.fail(f"a charged weight term must print its GiB: {charged}")


# 5. Stand-downs and render-site wiring.

def _render_rig(monkeypatch, tmp_path, *, size: int, mem: int,
                preset: str = "auto", canvas: tuple[int, int, int] = CONFIG_B,
                worker_args: dict | None = None, options: dict | None = None,
                world: int = 1, config_worker_args: dict | None = None):
    """The shipped render path, with a stubbed size standing in for the
    artifact's bytes. The driver-footprint guard is stubbed out so only the
    guard under test can raise.

    `worker_args` is the live policy on the mesh and `config_worker_args` is
    cluster.toml's `[worker_args]`, reachable only through the handle's own
    merge. Both are wired because the guard reads the operator's reserve from
    either place, as the driver-footprint guard does."""
    checkpoint = tmp_path / UNET_NAME
    checkpoint.write_bytes(b"x")
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: str(checkpoint)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint",
                        lambda _path: ("minimax_h3", "bf16"))
    monkeypatch.setattr(driver_footprint, "file_size_bytes", lambda _path: size)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: mem)
    monkeypatch.setattr(driver_footprint, "driver_footprint_preflight",
                        lambda *_a, **_kw: None)
    monkeypatch.setattr(render_preflight, "_LAST_RENDERED_UNET", None)
    monkeypatch.setattr(render_preflight, "_LAST_SUBMITTED_UNET", None)
    monkeypatch.setattr(render_preflight, "_LAST_DRIVER_CHARGED_UNET", None)
    config_args = dict(config_worker_args or {})
    handle = types.SimpleNamespace(owns_hosts=False, world=world)
    handle.effective_worker_args = (
        lambda requested=None: {**config_args, **dict(requested or {})})
    mesh = types.SimpleNamespace(
        topology_preset=preset, worker_args=dict(worker_args or {}),
        world=world, handle=handle)
    model = types.SimpleNamespace(unet_name=UNET_NAME, mesh=mesh,
                                  options=dict(options or {}))
    return model, _latent(canvas)


def _run(model, latent, request=None):
    render_preflight.activation_footprint_preflight_for_request(
        model, request if request is not None else _request(), latent)


def _record_estimates(monkeypatch) -> list:
    """Every completed estimate, in order, without changing any verdict."""
    seen: list = []
    real = h3_activation.estimate_h3_activation

    def _capture(**kwargs):
        estimate = real(**kwargs)
        seen.append(estimate)
        return estimate

    monkeypatch.setattr(h3_activation, "estimate_h3_activation", _capture)
    return seen


def test_the_incident_shape_refuses_through_the_shipped_entry_point(
        monkeypatch, tmp_path):
    """End to end on the render path: Config B, one rank, bf16, `auto`."""
    model, latent = _render_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                                mem=MEM_PENDING)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        _run(model, latent)
    message = str(excinfo.value)
    for token in ("minimax_h3 activation preflight", "37730 rows",
                  "37296 video, 414 audio, 20 text", UNET_NAME,
                  mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV):
        if token not in message:
            pytest.fail(f"the refusal must name {token!r}: {message}")


def test_the_hardware_proven_world_2_run_dispatches(monkeypatch, tmp_path):
    """uly2 is an explicit preset, so this render runs in the resident posture
    and the weight term is credited: the loader node performed the load before
    the driver stack existed."""
    model, latent = _render_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                                mem=MEM_RESIDENT, preset="uly2", world=2)
    _run(model, latent)   # no raise


def test_config_a_dispatches_on_one_rank(monkeypatch, tmp_path):
    model, latent = _render_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                                mem=MEM_PENDING, canvas=CONFIG_A)
    _run(model, latent)   # no raise


def test_skips_every_other_family(monkeypatch, tmp_path):
    """Family scoped: a hard no-op for everything that is not minimax_h3."""
    model, latent = _render_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                                mem=1)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint",
                        lambda _path: ("krea2", "bf16"))
    seen = _record_estimates(monkeypatch)
    _run(model, latent)
    h3_activation.preflight_h3_activation_for_request(
        model, _request(), latent, family="krea2", path="x", unet_name="x",
        memo_credit=False)   # no raise
    if seen:
        pytest.fail("an unregistered family must never reach the estimator")


@pytest.mark.parametrize("break_it", ["discrete", "off_linux"])
def test_skips_discrete_gpu_and_off_linux(monkeypatch, tmp_path, break_it):
    """Off unified memory host MemAvailable does not bound anything, and off
    Linux there is no reading at all."""
    model, latent = _render_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                                mem=1)
    if break_it == "discrete":
        monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: False)
    else:
        monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: None)
    seen = _record_estimates(monkeypatch)
    _run(model, latent)
    if seen:
        pytest.fail(f"{break_it} must stand the guard down before the estimate")


def test_a_dtype_cast_still_charges_activations(monkeypatch, tmp_path):
    """A cast invalidates the weight term and nothing else: the packed stream
    carries the text-encoder output dtype, so the activation estimate does not
    depend on the cast. Standing the whole guard down would leave the incident
    shape reachable with a supported widget value."""
    model, latent = _render_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                                mem=MEM_RESIDENT,
                                options={"weight_dtype": "fp8_e4m3fn"})
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        _run(model, latent)
    message = str(excinfo.value)
    if "DiT weights 0.0 GiB" not in message:
        pytest.fail(f"a cast must stand the weight term down, not the guard: "
                    f"{message}")
    if "37730 rows" not in message:
        pytest.fail(f"the activation term must stay live under a cast: {message}")
    if h3_activation.CAST_NOTE not in message:
        pytest.fail(f"a zero weight term under a cast must say WHY, or it "
                    f"reads exactly like a resident credit: {message}")


def test_escape_hatch_env_stands_the_guard_down_and_warns_once(monkeypatch, tmp_path):
    """With the guard disabled the 2026-08-05 kill is reachable again, so the
    state must be greppable after an incident: WARNING, not INFO, once per
    driver process."""
    records: list[tuple] = []
    monkeypatch.setattr(h3_activation, "log", types.SimpleNamespace(
        warning=lambda msg, *args: records.append((msg, args))))
    monkeypatch.setenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, "1")
    model, latent = _render_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                                mem=MEM_PENDING)
    for _ in range(3):
        _run(model, latent)   # no raise
    if len(records) != 1:
        pytest.fail(f"expected exactly one WARNING, got {len(records)}")
    if mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV not in str(records[0]):
        pytest.fail(f"the WARNING must name the variable: {records[0]}")


def test_driver_preflight_env_does_not_stand_this_guard_down(monkeypatch, tmp_path):
    """Two guards, two kill switches: turning the driver footprint preflight
    off must not turn this guard off."""
    monkeypatch.setenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, "1")
    model, latent = _render_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                                mem=MEM_PENDING)
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        _run(model, latent)


def test_an_explicit_preset_credits_the_weight_term_on_a_cold_session(
        monkeypatch, tmp_path):
    """A double charge: on the first render of a session under an explicit
    preset every memo is unset, yet the loader node has already loaded the
    checkpoint eagerly, so its bytes are inside this MemAvailable reading.
    Charging them again false-refuses two hardware-proven runs."""
    seen = _record_estimates(monkeypatch)
    model, latent = _render_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                                mem=MEM_RESIDENT, preset="uly2", world=2,
                                canvas=CONFIG_A)
    _run(model, latent)
    if not seen:
        pytest.fail("the estimator must run for an explicit preset")
    if seen[0].weight_bytes:
        pytest.fail(f"an explicit preset credits the weight term, charged "
                    f"{seen[0].weight_bytes} bytes")


@pytest.mark.parametrize("memo", ["_LAST_RENDERED_UNET", "_LAST_SUBMITTED_UNET",
                                  "_LAST_DRIVER_CHARGED_UNET"])
def test_each_residency_memo_credits_the_weight_term(monkeypatch, tmp_path, memo):
    """A warm re-render, a depth>=2 pipelined submit and a retry after a failed
    render each leave the checkpoint resident under a different memo."""
    seen = _record_estimates(monkeypatch)
    model, latent = _render_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                                mem=MEM_RESIDENT, canvas=CONFIG_A)
    monkeypatch.setattr(render_preflight, memo, UNET_NAME)
    _run(model, latent)
    if not seen or seen[0].weight_bytes:
        pytest.fail(f"{memo} must credit the weight term: {seen}")


@pytest.mark.parametrize("where", ["live", "config"])
def test_the_operators_reserve_reaches_the_render_site_from_either_place(
        monkeypatch, tmp_path, where):
    """A large `uma_reserve_gb` can refuse a hardware-proven shape, and the
    operator can set it on the Init widget (the live worker args) or in
    cluster.toml `[worker_args]`, which reaches this guard only through the
    handle's own merge. The driver-footprint guard one line earlier merges
    both, so reading only the live args would leave two class C guards at one
    call site disagreeing about the operator's own knob."""
    args = {"uma_reserve_gb": 48.0}
    model, latent = _render_rig(
        monkeypatch, tmp_path, size=DIT_REF2VA_INT8_PRUNED,
        mem=int(60.4 * _GIB), preset="uly2", world=2,
        worker_args=args if where == "live" else None,
        config_worker_args=None if where == "live" else args)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        _run(model, latent)
    message = str(excinfo.value)
    if "your uma_reserve_gb" not in message:
        pytest.fail(f"the {where} reserve must be named as the operator's: "
                    f"{message}")
    if "48.0 GiB reserve" not in message:
        pytest.fail(f"the {where} reserve must reach the arithmetic: {message}")


def test_the_reserve_merge_is_advisory_and_never_raises(monkeypatch, tmp_path):
    """The merge is a convenience over a duck-typed handle, so a handle that
    raises must degrade to the live args and the driver floor, never take the
    guard down with it."""
    model, latent = _render_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                                mem=MEM_RESIDENT)

    def _boom(_requested=None):
        raise RuntimeError("no config here")

    model.mesh.handle.effective_worker_args = _boom
    with pytest.raises(mesh_safety.StockLoadCapacityError) as excinfo:
        _run(model, latent)
    if "the driver floor" not in str(excinfo.value):
        pytest.fail(f"a failed merge falls back to the floor: {excinfo.value}")


def test_an_unresolved_row_count_stands_this_guard_down(monkeypatch, tmp_path):
    """The activation guard owns the row term. With no readable geometry there
    is no row term, and charging the weight term alone would refuse on a
    question the driver-footprint guard already owns, in a message advertising
    zero rows and a bound the operator cannot act on."""
    model, _latent_ignored = _render_rig(monkeypatch, tmp_path,
                                         size=DIT_REF2VA_BF16, mem=MEM_RESIDENT)
    # 61.7 GiB pending against 28.4 usable: the weight term alone refuses, so
    # this passes only because the guard stands down without a row count.
    _run(model, {"samples": _Fake((1, 24, 2))})     # fewer than 5 dims


def test_the_belt_and_braces_re_checks_hold_on_their_own(monkeypatch):
    """`h3_activation_preflight` re-checks the kill switch and the integrated
    device so no caller can reach the raise around them. Called directly, with
    an over-budget estimate, each re-check has to stand it down by itself."""
    estimate = _estimate(ROWS_B, sp=1, mem=MEM_RESIDENT)
    if estimate.fits:
        pytest.fail("this test is only worth running on a refusing estimate")
    monkeypatch.setenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, "1")
    h3_activation.h3_activation_preflight(estimate, unet_name=UNET_NAME)
    monkeypatch.delenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: False)
    h3_activation.h3_activation_preflight(estimate, unet_name=UNET_NAME)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        h3_activation.h3_activation_preflight(estimate, unet_name=UNET_NAME)


def test_sp_degree_is_the_mesh_world():
    """For this family the divisor is a proof rather than an estimate: xfuser
    requires world == dp*ulysses*ring*cfg exactly, and cfg>1 and dp>1 are
    refused as class P before dispatch, so every H3 render that can dispatch at
    all has ulysses*ring == world."""
    if h3_activation.sp_degree_for_mesh(types.SimpleNamespace(world=2)) != 2:
        pytest.fail("the mesh world is the degree")
    handle_only = types.SimpleNamespace(handle=types.SimpleNamespace(world=4))
    if h3_activation.sp_degree_for_mesh(handle_only) != 4:
        pytest.fail("the handle's world is the fallback")
    for mesh in (None, object(), types.SimpleNamespace(world=0),
                 types.SimpleNamespace(world="two"),
                 types.SimpleNamespace(world=-3)):
        if h3_activation.sp_degree_for_mesh(mesh) != 1:
            pytest.fail(f"{mesh!r} must floor at 1")


def test_sp_degree_for_mesh_is_total():
    """It runs at the loader node above `ensure_live`, on a mesh that has not
    been healed, and its caller sits inside a blanket handler that would
    swallow a raise and turn off the whole loader-site guard."""

    class _Angry:
        @property
        def world(self):
            raise RuntimeError("the fleet is not live yet")

    if h3_activation.sp_degree_for_mesh(_Angry()) != 1:
        pytest.fail("a raising world must degrade to 1, never propagate")


def test_a_broken_estimator_never_blocks_a_render(monkeypatch, tmp_path):
    """Fails open on everything except a completed, over-budget estimate."""
    model, latent = _render_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                                mem=MEM_PENDING)

    def _boom(*_a, **_kw):
        raise ValueError("the row extractor is broken")

    monkeypatch.setattr(h3_rows, "rows_from_request", _boom)
    _run(model, latent)   # no raise


def test_a_remote_rank_zero_is_either_credited_or_the_omission_is_named():
    """The driver-footprint guard charges nothing when rank 0 is not on the
    driver host: a rank on another box places neither its weights nor its
    activations here. This guard must stand down the same way, or its module
    docstring must name the omission. No run on this rig reaches the case,
    since rank 0 shares the driver host here; if rank 0 ever moves to another
    box, the omission would be silent."""
    source = (SRC / "h3_activation.py").read_text()
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree) or ""
    calls_it = "rank0_co_resident" in source
    names_it = any(word in docstring.lower()
                   for word in ("co-resident", "co_resident", "driver host"))
    if not calls_it and not names_it:
        pytest.fail("either credit a remote rank 0 or state the omission in "
                    "the module docstring")


# 6. Placement: neither caller names the guard's modules or can route around it.

def test_render_site_samples_the_memos_before_the_driver_delegation():
    """Ordering trap: `_driver_footprint_preflight_for_request` stamps
    _LAST_DRIVER_CHARGED_UNET for this render when it clears a pending load, so
    reading the memos after it would credit this render for its own charge and
    zero the weight term on the first render of every session."""
    tree = ast.parse((SRC / "nodes" / "render_preflight.py").read_text())
    host = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
                and node.name == "activation_footprint_preflight_for_request")
    memo = [node.lineno for node in ast.walk(host)
            if isinstance(node, ast.Name) and node.id == "memo_credit"]
    delegation = [node.lineno for node in ast.walk(host)
                  if isinstance(node, ast.Call)
                  and getattr(node.func, "id", "") == "_driver_footprint_preflight_for_request"]
    h3 = [node.lineno for node in ast.walk(host)
          if isinstance(node, ast.Attribute)
          and node.attr == "preflight_h3_activation_for_request"]
    registry = [node.lineno for node in ast.walk(host)
                if isinstance(node, ast.Attribute)
                and node.attr == "REF_POSE_TOKEN_FAMILIES"]
    if not (memo and delegation and h3 and registry):
        pytest.fail(f"expected all four anchors: {memo}, {delegation}, {h3}, "
                    f"{registry}")
    if not min(memo) < min(delegation) < min(h3) < min(registry):
        pytest.fail(f"order must be memo, driver delegation, H3 delegation, "
                    f"registry guard: {min(memo)}, {min(delegation)}, "
                    f"{min(h3)}, {min(registry)}")


def test_neither_caller_names_the_new_modules():
    """nodes/common.py and nodes/pipeline.py are the two callers of the same
    function. Delegating from inside that function means neither can route
    around it, and neither has to know the calibration leaves exist."""
    common = (SRC / "nodes" / "common.py").read_text()
    pipeline = (SRC / "nodes" / "pipeline.py").read_text()
    for name, source in (("common.py", common), ("pipeline.py", pipeline)):
        if any(module in source for module in
               ("h3_activation", "h3_rows", "h3_calibration")):
            pytest.fail(f"{name} must not name the new modules; the delegation "
                        "lives inside render_preflight")
        if "activation_footprint_preflight_for_request" not in source:
            pytest.fail(f"{name} must still call the shared entry point")


def test_a_pipelined_submission_inherits_this_refusal(monkeypatch, tmp_path):
    """The source test above proves the delegation's place; this one proves its
    effect on the second caller. Pipelined submissions have run the SCAIL
    guard since 2026-07-28; moving the delegation into run_render would reopen
    this bypass while every other test in this file stayed green."""
    from dgx_monarch.nodes.pipeline import RenderPipeline

    model, latent = _render_rig(monkeypatch, tmp_path, size=DIT_REF2VA_BF16,
                                mem=MEM_RESIDENT)
    pipe = RenderPipeline(depth=2)
    with pytest.raises(mesh_safety.StockLoadCapacityError):
        pipe.push(model, _request(), latent, 4.5, 10)


def test_the_refusal_points_at_a_troubleshooting_entry_that_exists():
    """The shipped message names `docs/TROUBLESHOOTING.md #56`. A renumbering
    would leave it pointing at nothing with the suite green, so this reads the
    heading, as tests/test_docs_status_sync.py does for docs/TROUBLESHOOTING.md #28."""
    message = h3_activation.h3_refusal(_estimate(ROWS_B, sp=1), UNET_NAME)
    if "docs/TROUBLESHOOTING.md #56" not in message:
        pytest.fail(f"the refusal must name its entry: {message}")
    troubleshooting = (REPO / "docs" / "TROUBLESHOOTING.md").read_text()
    heading = "## 56. MiniMax H3 render refused: activation footprint preflight"
    if heading not in troubleshooting:
        pytest.fail(f"TROUBLESHOOTING.md has no {heading!r}")


def test_h3_modules_are_leaves():
    """From dgx_monarch, h3_activation may import mesh_safety,
    driver_footprint, refusal, log, h3_rows and h3_calibration only. h3_rows
    and h3_calibration import nothing from dgx_monarch, so they stay testable
    on CPU with hand-built fakes and pull no runtime into the driver's node
    path. None of the three imports torch or comfy."""
    roots = {"torch", "comfy", "comfy_extras", "folder_paths", "nodes",
             "subprocess", "mesh", "actor"}
    for name in ("h3_rows.py", "h3_calibration.py", "h3_activation.py"):
        path = SRC / name
        tree = ast.parse(path.read_text(), filename=str(path))
        imported: list[str] = []
        relative: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                imported.append(base)
                imported.extend(f"{base}.{alias.name}" if base else alias.name
                                for alias in node.names)
                if node.level and base:
                    relative.append(base.split(".")[0])
                elif node.level:
                    relative.extend(alias.name for alias in node.names)
        hits = [item for item in imported
                if item and item.split(".")[0] in roots]
        if hits:
            pytest.fail(f"{name} must stay a leaf, imports {hits}")
        if name in ("h3_rows.py", "h3_calibration.py") and (relative or any(
                item.startswith("dgx_monarch") for item in imported)):
            pytest.fail(f"{name} must import nothing from dgx_monarch: "
                        f"{relative or imported}")
        if name == "h3_activation.py":
            allowed = {"mesh_safety", "driver_footprint", "refusal", "h3_rows",
                       "h3_calibration", "log"}
            stray = [item for item in relative if item not in allowed]
            if stray:
                pytest.fail(f"h3_activation may not import {stray}")


def test_the_calibration_record_is_its_own_leaf():
    """A recalibration is an evidence change and a guard edit is a logic
    change, so they are different diffs. Both constants keep their spelling on
    h3_activation, because every caller and every message reads them there."""
    from dgx_monarch import h3_calibration

    for name in ("H3_SHARDED_ROW_BYTES", "H3_REPLICATED_ROW_BYTES"):
        if getattr(h3_activation, name) != getattr(h3_calibration, name):
            pytest.fail(f"{name} diverged between the calibration record and "
                        "the guard that reads it")
    source = (SRC / "h3_activation.py").read_text()
    if "_H3_BLOCK_FACTOR" in source:
        pytest.fail("the block factor and its derivation belong in "
                    "h3_calibration.py, not in the guard")


def test_h3_modules_stay_under_the_line_rule():
    """All three stay at or under 500 lines, so none needs a row in
    tests/line_limit_helpers.py. The guard's other files (nodes/render_preflight.py,
    nodes/loader_preflight.py, mesh_safety.py, driver_footprint.py) are held by
    test_module_line_limit_exception_ledger in tests/test_surface_remediation.py."""
    for name in ("h3_rows.py", "h3_calibration.py", "h3_activation.py"):
        lines = len((SRC / name).read_text().splitlines())
        if lines > 500:
            pytest.fail(f"{name} is {lines} lines; a file over 500 needs a "
                        "reviewed ledger row")


def test_minimax_h3_stays_out_of_the_ref_pose_registry():
    """The two registries stay disjoint, so the render-site driver footprint
    keeps the weight charge and nothing is charged twice. Two guards can both
    refuse the same render: they answer different questions against the same
    reading, and they are never summed."""
    if "minimax_h3" in mesh_safety.REF_POSE_TOKEN_FAMILIES:
        pytest.fail("minimax_h3 must stay out of REF_POSE_TOKEN_FAMILIES: that "
                    "estimator counts 5-D video-ish latents only and would "
                    "under-count ref2va and fl2va")
    if driver_footprint.weights_owned_by_activation_preflight("minimax_h3"):
        pytest.fail("the weight charge must stay with the driver footprint")


def test_the_reserved_guard_id_is_now_claimed():
    """The id was reserved in refusal.GUARDS and the frozen gate_audit_vocab
    tables, so claiming it edits neither. The name fits: this is the activation
    footprint preflight for a second family, whose row algebra lives in its own
    module."""
    ledger = json.loads((Path(__file__).with_name("refusal_class_ledger.json")).read_text())
    guard = "activation_footprint_preflight"
    if guard in ledger.get("reserved_guards", {}):
        pytest.fail("the guard id is claimed now and must leave reserved_guards")
    rows = [row for row in ledger.get("tagged", []) if row.get("guard") == guard]
    # The driver-side render-memory price joined this guard on 2026-09-02: it
    # reads comfy's own sample-time estimate, the same boundary under a
    # different reading, so it claims no new id. Set equality makes a third
    # site a reviewed edit.
    claiming = {row["site"] for row in rows}
    expected = {
        "h3_activation.py::h3_activation_preflight::StockLoadCapacityError#0",
        "render_memory_price.py::refuse_render_memory::RenderMemoryPriceError#0",
    }
    if claiming != expected:
        pytest.fail(f"the sites claiming {guard} moved: {sorted(claiming)}")
    for row in rows:
        if row["class"] != "C" or row["waivable"] is not False:
            pytest.fail(f"every row must be class C and not waivable: {row}")
    ceiling = ledger.get("grandfathered_ceiling", 0)
    if ceiling > 198:
        pytest.fail("the ceiling dated 2026-08-10 may only fall (DESIGN.md "
                    f"section 7); it reads {ceiling}")
    stray = [name for name in ("h3_activation.py", "h3_calibration.py", "h3_rows.py")
             if name in ledger.get("untagged_by_module", {})]
    if stray:
        pytest.fail("the H3 modules contain exactly one typed raise between them "
                    f"and it is tagged, so none may enter the backlog: {stray}")
    if ledger.get("new_sites_awaiting_a_class"):
        pytest.fail(f"no site may be left waiting for a class: "
                    f"{ledger['new_sites_awaiting_a_class']}")


# 7. The loader site: a load that fits is not a render that fits.

PRETEND_SIZE = {
    ("diffusion_models", UNET_NAME): DIT_REF2VA_INT8,
    ("text_encoders", TE_NAME): TE_NVFP4,
    ("vae", VAE_VIDEO_NAME): VAE_VIDEO,
    ("vae", VAE_AUDIO_NAME): VAE_AUDIO,
}
# The graph's own stack, before any activation term: text encoder at
# loader_graph's 1.15 resident ratio (1.144 measured, rounded up) plus both VAEs.
STACK_NO_ACTIVATIONS = int(TE_NVFP4 * 1.15) + VAE_VIDEO + VAE_AUDIO
# Chosen so the legacy load arena refuses the transient window (slab is the
# remedy), while the settled window fits without the render's activations and
# fails with them. The loader-site fold must preserve that distinction.
MEM_LOADER = 64_334_622_731


def _h3_graph(canvas: tuple[int, int, int], *, dit: str = UNET_NAME,
              second: tuple[int, int, int] | None = None,
              canvas_node: bool = True) -> dict:
    width, height, length = canvas
    prompt: dict = {
        "1": {"class_type": "DGXMonarchInit", "inputs": {"topology": "single"}},
        "2": {"class_type": "DGXMonarchUNETLoader",
              "inputs": {"unet_name": dit, "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TE_NAME, "type": "minimax"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VIDEO_NAME}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUDIO_NAME}},
    }
    if canvas_node:
        prompt["8"] = {"class_type": "EmptyMiniMaxH3LatentAV",
                       "inputs": {"width": width, "height": height,
                                  "length": length}}
    if second is not None:
        prompt["9"] = {"class_type": "MiniMaxH3ImageToVideo",
                       "inputs": {"clip": ["3", 0], "vae": ["4", 0],
                                  "width": second[0], "height": second[1],
                                  "length": second[2], "prompt": "a lighthouse"}}
    return prompt


class _Handle:
    def __init__(self):
        self.owns_hosts = False       # a this_host() mesh: rank 0 is co-resident
        self.config = types.SimpleNamespace(hosts=(), worker_args={})
        self.world = 1

    def effective_worker_args(self, worker_args=None) -> dict:
        return dict(worker_args or {})


class _Mesh:
    def __init__(self, worker_args: dict | None = None, world: int = 1):
        self.handle = _Handle()
        self.handle.world = world
        self.topology_preset = "single"
        self.worker_args = dict(worker_args or {})
        self.world = world


@pytest.fixture
def artifacts(monkeypatch, tmp_path):
    """A stub `folder_paths` over real (tiny) artifact files, reporting the
    staged sizes. Every graph walk needs this much and no more."""
    paths: dict[str, int] = {}
    for (folder, name), size in PRETEND_SIZE.items():
        target = tmp_path / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(name.encode())
        paths[str(target)] = size
    monkeypatch.setattr(driver_footprint, "file_size_bytes",
                        lambda path: paths.get(str(path), 0))
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda folder, name: (
        str(tmp_path / folder / name) if (folder, name) in PRETEND_SIZE else None)
    output = tmp_path / "output"
    output.mkdir(parents=True, exist_ok=True)
    folder_paths.get_output_directory = lambda: str(output)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    return tmp_path


@pytest.fixture
def loader_rig(monkeypatch, artifacts):
    """The shipped loader node: a UMA box with a known MemAvailable, an H3
    header sniff, and a consent subsystem whose store and pending registry are
    private to this test."""
    from dgx_monarch import consent_pending, consent_store
    from dgx_monarch.adapters import detect

    loader_preflight.reset_memos()
    consent_pending.clear_all()
    for spec in consent_pending.KIND_SPECS.values():
        monkeypatch.delenv(spec.env_var, raising=False)
    monkeypatch.delenv(consent_pending.AUTO_RESCUE_ENV, raising=False)
    monkeypatch.setattr(consent_store, "MEMO_PATH",
                        str(artifacts / "consent_memo.json"))
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: MEM_LOADER)
    monkeypatch.setattr(detect, "sniff_checkpoint",
                        lambda path: ("minimax_h3", "bf16"))
    yield consent_pending
    loader_preflight.reset_memos()
    consent_pending.clear_all()


def _terms(prompt, *, sp: int = 1, family: str = "minimax_h3"):
    return loader_graph.driver_stack_terms(prompt, sp_degree=sp, family=family)


def _refuse_loader(mesh, unet_name, prompt):
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as excinfo:
        loader_preflight.preflight_loader_footprint(mesh, unet_name, {}, prompt)
    return str(excinfo.value)


def test_loader_site_folds_graph_activations_into_the_settled_window(artifacts):
    """The graph can price video and audio rows exactly, from the three
    canvas-bearing stock nodes' literal widget ints."""
    terms = _terms(_h3_graph(CONFIG_B))
    want = 37_296 + 414        # no text, keyframe or reference rows here
    if terms.activation_bytes != _estimate(_rows(video=want), sp=1).activation_bytes:
        pytest.fail(f"the graph term is {terms.activation_bytes}, expected the "
                    "video and audio rows only")
    if terms.total != terms.te_bytes + terms.vae_bytes + terms.ref_bytes + terms.activation_bytes:
        pytest.fail("the activation term must be inside `total`, which is what "
                    "puts it in both the settled window and the rescue's")
    halved = _terms(_h3_graph(CONFIG_B), sp=2)
    if halved.activation_bytes >= terms.activation_bytes:
        pytest.fail("the sharded part must fall with the sequence-parallel "
                    "degree at the loader node too")
    if not terms.activation_note:
        pytest.fail("the term must carry a note naming what it could not see")


def test_loader_site_reads_the_largest_declared_canvas_not_the_sum(artifacts):
    """One render is one packed sequence, so the peak is the biggest declared
    canvas, not the sum of every canvas node in the graph."""
    biggest = _terms(_h3_graph(CONFIG_B))
    # Both orderings, because comfy's prompt dict has no guaranteed order and a
    # `largest` rule and a `last` rule are indistinguishable on one of them.
    for graph in (_h3_graph(CONFIG_A, second=CONFIG_B),
                  _h3_graph(CONFIG_B, second=CONFIG_A)):
        both = _terms(graph)
        if both.activation_bytes != biggest.activation_bytes:
            pytest.fail(f"expected the largest canvas alone: "
                        f"{both.activation_bytes} against "
                        f"{biggest.activation_bytes}")


def test_loader_site_charges_nothing_it_cannot_resolve_off_a_comfy_process(
        monkeypatch, artifacts):
    """`folder_paths` does not exist outside a comfy process, and letting that
    import propagate would turn off the whole loader-site refusal: its caller
    answers any exception by returning None. It degrades one term at a time
    instead."""
    monkeypatch.setitem(sys.modules, "folder_paths", None)   # import raises
    terms = _terms(_h3_graph(CONFIG_B))
    if terms.te_bytes or terms.vae_bytes:
        pytest.fail(f"nothing resolvable means nothing charged: {terms}")
    if not terms.resolved or not terms.activation_bytes:
        pytest.fail(f"the rest of the walk must survive the missing module, "
                    f"including the render this graph declares: {terms}")


def test_the_loader_refusal_never_prints_an_empty_activation_clause(artifacts):
    """The unresolved default is reachable: `prompt` is comfy's hidden input
    and the headless harness path calls the loader node directly, so comfy
    fills nothing in. An empty note would render the class C refusal's
    activation clause as bare parentheses."""
    terms = loader_graph.DriverStackTerms()
    if not terms.activation_note.strip():
        pytest.fail("the unresolved default must carry words, not an empty "
                    "string that a refusal prints as ()")
    profile = driver_footprint.DRIVER_STACK_FAMILIES["minimax_h3"]
    estimate = driver_footprint.estimate_driver_footprint(
        profile=profile, mem_available=MEM_LOADER, weight_bytes=DIT_REF2VA_BF16,
        weights_resident=False, co_resident=True, slab_weights=None,
        ref_pixels=0, ref_blocks=0, ref_frames=0, reserve=RESERVE)
    message = loader_preflight.loader_refusal(
        estimate, terms, unet_name=UNET_NAME, profile=profile,
        floor_reserve=RESERVE, rescue=None)
    if "()" in message:
        pytest.fail(f"no clause may render as empty parentheses: {message}")
    if terms.activation_note not in message:
        pytest.fail(f"the default note must reach the operator: {message}")


@pytest.mark.parametrize("prompt", [None, {}, "not a graph", 42,
                                    {"1": {"inputs": None}}])
def test_loader_site_activation_term_is_zero_without_a_readable_h3_canvas(artifacts, prompt):
    if _terms(prompt).activation_bytes:
        pytest.fail(f"an unreadable graph must charge nothing: {prompt!r}")
    no_canvas = _terms(_h3_graph(CONFIG_B, canvas_node=False))
    if no_canvas.activation_bytes:
        pytest.fail("a graph with no H3 canvas node must charge nothing")


def test_loader_site_activation_term_is_family_scoped(artifacts):
    """DRIVER_STACK_FAMILIES holds more than one family and a dual-model graph
    loads two checkpoints from one prompt, so another family must not inherit
    H3's activation charge from an H3 canvas node in the same graph."""
    if _terms(_h3_graph(CONFIG_B), family="krea2").activation_bytes:
        pytest.fail("another family must not inherit the H3 charge")
    if h3_activation.graph_activation_bytes(
            _h3_graph(CONFIG_B), 1, family="krea2")[0]:
        pytest.fail("graph_activation_bytes must be family scoped")
    default = loader_graph.driver_stack_terms(_h3_graph(CONFIG_B))
    if default.activation_bytes:
        pytest.fail("the default call must charge no activation term, or every "
                    "existing loader-site fixture silently changes value")


@pytest.mark.parametrize("env", ["activation", "driver"])
def test_loader_site_activation_term_stands_down_under_either_kill_switch(
        monkeypatch, artifacts, env):
    """An operator who turned the activation guard off must not meet its
    charge again at the loader node, and one who turned the driver preflight
    off must not meet an activation term inside it."""
    variable = (mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV if env == "activation"
                else driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV)
    monkeypatch.setenv(variable, "1")
    charged, note = h3_activation.graph_activation_bytes(
        _h3_graph(CONFIG_B), 1, family="minimax_h3")
    if charged:
        pytest.fail(f"{variable} must remove the term, charged {charged}")
    if not note:
        pytest.fail("a dropped term must still say why")
    if _terms(_h3_graph(CONFIG_B)).activation_bytes:
        pytest.fail(f"{variable} must reach the terms the loader site compares")


def test_loader_site_activation_term_survives_the_stack_credits(monkeypatch, artifacts):
    """A render's activations are never already spent, so the two residency
    credits that zero the text encoder, the VAEs and the reference encode must
    leave this term alone."""
    monkeypatch.setattr(loader_graph, "_comfy_has_loaded_models", lambda: True)
    terms = loader_graph.credited_terms(_terms(_h3_graph(CONFIG_B)))
    if not terms.credited:
        pytest.fail("this fixture must exercise the credit path")
    if terms.te_bytes or terms.vae_bytes or terms.ref_bytes:
        pytest.fail("the spent stack must be credited")
    if not terms.activation_bytes or terms.total != terms.activation_bytes:
        pytest.fail(f"the render's activations must stay charged: {terms}")


def test_loader_site_rescue_is_not_offered_when_slab_only_fixes_the_load(loader_rig):
    """Slab residency removes the legacy load arena and only the arena, so it
    fixes the transient window and nothing else. With the render folded into
    the settled window, a slab load that clears the load but leaves no room for
    the render is still refused, and no card is offered whose button cannot
    change the verdict."""
    prompt = _h3_graph(CONFIG_B)
    terms = _terms(prompt)
    without = MEM_LOADER - 4 * _GIB - STACK_NO_ACTIVATIONS
    if not DIT_REF2VA_INT8 <= without:
        pytest.fail("this fixture must fit the settled window WITHOUT the "
                    "render's activations, or it proves nothing")
    if DIT_REF2VA_INT8 <= without - terms.activation_bytes:
        pytest.fail("this fixture must fail the settled window WITH them")
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as excinfo:
        loader_preflight.preflight_loader_footprint(_Mesh(), UNET_NAME, {}, prompt)
    message = str(excinfo.value)
    if loader_rig.pending_cards():
        pytest.fail("no card may be offered whose button cannot change the "
                    "verdict")
    if "DGX Monarch panel" in message:
        pytest.fail(f"the text must not promise a card: {message}")
    tag = parse_refusal_tag(message)
    if tag is None or tag.refusal_class.value != "C" or tag.waivable:
        pytest.fail(f"expected an unwaivable class C refusal: {tag}")
    if "render activations" not in message:
        pytest.fail(f"the charged term must be named: {message}")
    if f"at the {loader_preflight.WINDOW_SETTLED} window" not in message:
        pytest.fail(f"the settled window is what refuses here: {message}")


def test_loader_site_offers_the_rescue_when_slab_clears_both_windows(loader_rig):
    """With the activation term present, the same graph at Config A leaves
    room for the render, so slab clears both windows and the card is the
    remedy."""
    prompt = _h3_graph(CONFIG_A)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as excinfo:
        loader_preflight.preflight_loader_footprint(_Mesh(), UNET_NAME, {}, prompt)
    message = str(excinfo.value)
    if "DGX Monarch panel" not in message:
        pytest.fail(f"slab clears both windows here, so the card is the "
                    f"remedy: {message}")
    cards = loader_rig.pending_cards()
    if len(cards) != 1 or cards[0]["kind"] != "rescue-slab":
        pytest.fail(f"exactly one rescue card must be registered: {cards}")


def test_a_broken_graph_extractor_never_disables_the_loader_footprint_guard(
        monkeypatch, loader_rig):
    """`driver_stack_terms` is called inside `preflight_loader_footprint`'s
    blanket handler, so a raise from the row extractor would not just drop the
    activation term: it would turn off the loader-site refusal, shipped
    2026-08-04, for every load."""

    def _boom(*_a, **_kw):
        raise RuntimeError("the graph extractor is broken")

    monkeypatch.setattr(h3_rows, "rows_from_prompt_graph", _boom)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError):
        loader_preflight.preflight_loader_footprint(
            _Mesh(), UNET_NAME, {}, _h3_graph(CONFIG_A))


def test_the_budget_clause_names_activations_only_when_it_charged_them(
        monkeypatch, loader_rig):
    """Two message rules. The settled-window budget clause fires
    whenever the stack is nonzero, so it must not name a term it did not
    charge; and because the credits leave activations charged, a credited stack
    can still be nonzero, in which case the credit reason must still be
    printed rather than lost to the first arm."""
    claim = "render activations this graph has not spent yet"
    # No canvas node, so no activation term, and a budget low enough that the
    # SETTLED window is the one that refuses.
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: int(57 * _GIB))
    silent = _refuse_loader(_Mesh(), UNET_NAME,
                            _h3_graph(CONFIG_B, canvas_node=False))
    if f"at the {loader_preflight.WINDOW_SETTLED} window" not in silent:
        pytest.fail(f"fixture drift: this arm must refuse at the settled "
                    f"window: {silent}")
    if claim in silent:
        pytest.fail(f"no canvas was readable, so no activations were charged "
                    f"and the budget clause must not name them: {silent}")
    # Now the other arm: comfy already holds the text encoder and the VAEs, so
    # the stack is credited, but the render's activations are never already
    # spent and stay charged.
    loader_preflight.reset_memos()
    monkeypatch.setattr(loader_graph, "_comfy_has_loaded_models", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: int(70 * _GIB))
    credited = _refuse_loader(_Mesh(), UNET_NAME, _h3_graph(CONFIG_B))
    reason = loader_graph.credited_terms(_terms(_h3_graph(CONFIG_B))).credit_reason
    if not reason:
        pytest.fail("fixture drift: this arm must exercise the credit path")
    if reason not in credited:
        pytest.fail(f"a credited stack that still charges activations must "
                    f"print why it was credited: {credited}")


# --- the kill switch honors only an explicit on value ---------------------

@pytest.mark.parametrize("value", ["0", "false", "off", "no", "", "  "])
def test_an_off_spelling_keeps_the_h3_activation_guard(monkeypatch, value):
    """The refusal tells the operator to set `=1`. `=0` must not disable it."""
    monkeypatch.setenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, value)
    monkeypatch.setattr(h3_activation, "_DISABLE_LOGGED", False)
    if h3_activation.preflight_disabled():
        pytest.fail(f"{value!r} turned the preflight off")


@pytest.mark.parametrize("value", ["1", "true", "ON", " yes "])
def test_an_on_spelling_disables_the_h3_activation_guard(monkeypatch, value):
    monkeypatch.setenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, value)
    monkeypatch.setattr(h3_activation, "_DISABLE_LOGGED", False)
    if not h3_activation.preflight_disabled():
        pytest.fail(f"{value!r} must take the documented bypass")


def test_the_h3_refusal_names_the_value_that_disables_it(monkeypatch):
    monkeypatch.delenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, raising=False)
    monkeypatch.setattr(h3_activation, "_DISABLE_LOGGED", False)
    monkeypatch.setenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, "1")
    if not h3_activation.preflight_disabled():
        pytest.fail("the instruction in the refusal text must be the behavior")
