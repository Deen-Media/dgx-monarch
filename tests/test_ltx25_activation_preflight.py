"""ltx25_activation and ltx25_calibration: the LTX dual-stream activation guard.

CPU only, no CUDA, no comfy beyond tiny stubs. The guard is arithmetic over
row counts, so the 2026-08-12 calibration series replays here and every ladder
shape that completed on this rig is checked against the shipped constants.

Provenance for every number below:

* the calibration series, docs/VALIDATION.md "LTX activation calibration
  (2026-08-12, two DGX Sparks)" (raw: out/ltx25_preflight/mem.tsv and fit.md):
  sibling MemAvailable at 1 s through two warm uly2 renders, 32,640 then
  57,600 video rows, int8 DiT and int8 text encoder;
* the promotion ladder, docs/VALIDATION.md "LTX 2.5 promotion evidence
  (2026-08-12, two DGX Sparks)" (raw: out/ltx25_ladder/ladder_state.md): the
  Config A and Config B legs that completed, on int8 and on bf16;
* the staged artifact sizes, byte exact, from the same ladder's model table;
* the row formulas, read from comfy: EmptyLTXVLatentVideo and
  latent_formats.LTXV for video, the 2.5 audio VAE metadata (16000/160/4 = 25
  latents/s) for audio.

Every shape this rig completed must fit, and a shape whose own arithmetic puts
it past the wall must refuse before any load. Both halves are pinned here, so
an edit to the block factor fails in this file with the reason, not on
hardware.
"""
from __future__ import annotations

import sys
import types

import pytest

from dgx_monarch import (
    driver_footprint,
    ltx25_activation,
    ltx25_calibration,
    mesh_safety,
)
from dgx_monarch.nodes import loader_graph, loader_preflight
from dgx_monarch.refusal import GUARDS, parse_refusal_tag

_GIB = 2 ** 30

# --- staged artifacts, byte exact (out/ltx25_ladder/ladder_state.md) --------
DIT_BF16 = 42_018_190_584          # 39.13 GiB
DIT_INT8 = 21_504_034_224          # 20.03 GiB
TE_BF16 = 26_263_860_594           # 24.46 GiB
TE_INT8 = 15_372_971_786           # 14.32 GiB
VAE_VIDEO = 1_472_223_346
VAE_AUDIO = 364_866_540

DIT_BF16_NAME = "ltx-2.5-22b-dev-transformer-bf16.safetensors"
DIT_INT8_NAME = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
TE_BF16_NAME = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
TE_INT8_NAME = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
VAE_VIDEO_NAME = "ltx-2.5-video-vae-bf16.safetensors"
VAE_AUDIO_NAME = "ltx-2.5-audio-vae-bf16.safetensors"

PRETEND_SIZE = {
    ("diffusion_models", DIT_BF16_NAME): DIT_BF16,
    ("diffusion_models", DIT_INT8_NAME): DIT_INT8,
    ("text_encoders", TE_BF16_NAME): TE_BF16,
    ("text_encoders", TE_INT8_NAME): TE_INT8,
    ("vae", VAE_VIDEO_NAME): VAE_VIDEO,
    ("vae", VAE_AUDIO_NAME): VAE_AUDIO,
}

RESERVE = 4 * _GIB

CONFIG_A = (960, 544, 121)
CONFIG_B = (1920, 1088, 121)
CONFIG_M2 = (2560, 1440, 121)
CONFIG_4K = (3840, 2176, 121)
CONFIG_4K_LONG = (3840, 2176, 361)
FPS = 24.0

ROWS_A_VIDEO = 8_160        # 16 latent frames x 17 x 30
ROWS_B_VIDEO = 32_640       # 16 x 34 x 60
ROWS_M2_VIDEO = 57_600      # 16 x 45 x 80
ROWS_4K_VIDEO = 130_560     # 16 x 68 x 120
ROWS_4K_LONG_VIDEO = 375_360  # 46 x 68 x 120
ROWS_AUDIO_121 = 126        # round(121/24 * 25)

# --- the calibration series (out/ltx25_preflight/fit.md) --------------------
# Sibling MemAvailable, one warm session, int8 DiT, uly2 world 2, 1 step.
SERIES_IDLE = 115.17
SERIES_POST_LOAD = 88.37       # weights + worker floor
SERIES_AFTER_M1 = 85.06        # + the 16,320 rows/rank peak
SERIES_AFTER_M2 = 81.68        # + the 28,800 rows/rank peak
MEASURED_FLOOR = (SERIES_IDLE - SERIES_POST_LOAD) * _GIB - DIT_INT8
MEASURED_ACT_M1 = (SERIES_POST_LOAD - SERIES_AFTER_M1) * _GIB
MEASURED_ACT_M2 = (SERIES_POST_LOAD - SERIES_AFTER_M2) * _GIB
# The one constraint a world-2 series can carry: S_video + 2*R_video.
MEASURED_ROW_SUM = (MEASURED_ACT_M2 - MEASURED_ACT_M1) / (
    (ROWS_M2_VIDEO - ROWS_B_VIDEO) // 2)


@pytest.fixture(autouse=True)
def _clean_module_state(monkeypatch):
    """Every once-only log flag and every kill switch is process state; reset
    it so test order can never decide a verdict. gpu_is_integrated is forced
    true because the probe returns False on a runner with no GPU."""
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.delenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, raising=False)
    monkeypatch.delenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, raising=False)
    monkeypatch.setattr(driver_footprint, "_DISABLE_LOGGED", False)
    monkeypatch.setattr(ltx25_activation, "_DISABLE_LOGGED", False)
    yield


def _rows(canvas, fps: float = FPS) -> ltx25_activation.LTXRows:
    width, height, length = canvas
    return ltx25_activation.LTXRows(
        video=ltx25_activation.video_rows(width, height, length),
        audio=ltx25_activation.audio_rows(length, fps), resolved=True)


def _graph(canvas, *, dit: str = DIT_BF16_NAME, te: str = TE_BF16_NAME,
           second=None, video_canvas: bool = True, audio_canvas: bool = True,
           batch: int = 1, guides: int = 0, guide_on: str = "10",
           guide_frames: int = 0, guide_latent: str | None = None) -> dict:
    """The committed LTX 2.5 template's canvas-bearing nodes, API format."""
    width, height, length = canvas
    graph: dict = {
        "3": {"class_type": "DGXMonarchUNETLoader",
              "inputs": {"unet_name": dit, "weight_dtype": "default"}},
        "4": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": te, "type": "ltxv"}},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VIDEO_NAME}},
        "9": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUDIO_NAME}},
        "12": {"class_type": "LTXVConcatAVLatent",
               "inputs": {"video_latent": ["10", 0], "audio_latent": ["11", 0]}},
    }
    if video_canvas:
        graph["10"] = {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": width, "height": height, "length": length,
            "batch_size": batch}}
    if audio_canvas:
        graph["11"] = {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "frames_number": length, "frame_rate": FPS, "batch_size": batch,
            "audio_vae": ["9", 0]}}
    if second is not None:
        width, height, length = second
        graph["30"] = {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": width, "height": height, "length": length,
            "batch_size": 1}}
    if guide_frames:
        # A guide clip whose length the graph does state, the one comfy-core
        # shape that carries it in a widget.
        graph["7"] = {"class_type": "RepeatImageBatch",
                      "inputs": {"image": ["70", 0], "amount": guide_frames}}
    # The first-and-last-frame shape: Add Guide chains through its own
    # latent output, so guide 1 reaches the canvas through guide 0.
    upstream = guide_latent if guide_latent is not None else guide_on
    for index in range(guides):
        graph[f"4{index}"] = {"class_type": "LTXVAddGuide", "inputs": {
            "positive": ["5", 0], "negative": ["6", 0], "vae": ["8", 0],
            "latent": [upstream, 0], "image": ["7", 0],
            "frame_idx": 0 if index == 0 else -1, "strength": 1.0}}
        upstream = f"4{index}"
    return graph


class _Handle:
    owns_hosts = False
    world = 2

    def effective_worker_args(self, args):
        return dict(args)


class _Mesh:
    def __init__(self, preset: str = "uly2", world: int = 2,
                 worker_args: dict | None = None):
        self.handle = _Handle()
        self.topology_preset = preset
        self.world = world
        self.worker_args = dict(worker_args or {})


@pytest.fixture
def artifacts(monkeypatch, tmp_path):
    """A stub `folder_paths` over real (tiny) files reporting staged sizes."""
    paths: dict[str, int] = {}
    for (folder, name), size in PRETEND_SIZE.items():
        target = tmp_path / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(name.encode())
        paths[str(target)] = size
    monkeypatch.setattr(driver_footprint, "file_size_bytes",
                        lambda path: paths.get(str(path), 0))
    module = types.ModuleType("folder_paths")
    module.get_full_path = lambda folder, name: (
        str(tmp_path / folder / name) if (folder, name) in PRETEND_SIZE else None)
    output = tmp_path / "output"
    output.mkdir(parents=True, exist_ok=True)
    module.get_output_directory = lambda: str(output)
    monkeypatch.setitem(sys.modules, "folder_paths", module)
    return tmp_path


@pytest.fixture
def loader_rig(monkeypatch, artifacts):
    """The shipped loader node: a UMA box with a known MemAvailable, an LTX
    header sniff, and a consent subsystem private to this test."""
    from dgx_monarch import consent_pending, consent_store
    from dgx_monarch.adapters import detect

    loader_preflight.reset_memos()
    consent_pending.clear_all()
    for spec in consent_pending.KIND_SPECS.values():
        monkeypatch.delenv(spec.env_var, raising=False)
    monkeypatch.delenv(consent_pending.AUTO_RESCUE_ENV, raising=False)
    monkeypatch.setattr(consent_store, "MEMO_PATH",
                        str(artifacts / "consent_memo.json"))
    monkeypatch.setattr(detect, "sniff_checkpoint", lambda path: ("ltx", "bf16"))
    monkeypatch.setattr(loader_graph, "_comfy_has_loaded_models", lambda: False)
    yield
    loader_preflight.reset_memos()
    consent_pending.clear_all()


def _set_mem(monkeypatch, gib: float) -> None:
    monkeypatch.setattr(mesh_safety, "mem_available_bytes",
                        lambda: int(gib * _GIB))


def _terms(prompt, *, sp: int = 1, family: str = "ltx"):
    return loader_graph.driver_stack_terms(prompt, sp_degree=sp, family=family)


def _refuse(mesh, unet_name, prompt) -> str:
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as excinfo:
        loader_preflight.preflight_loader_footprint(mesh, unet_name, {}, prompt)
    return str(excinfo.value)


# 1. The row algebra, against the formulas read out of comfy.

@pytest.mark.parametrize("canvas,want", [
    (CONFIG_A, ROWS_A_VIDEO), (CONFIG_B, ROWS_B_VIDEO),
    (CONFIG_M2, ROWS_M2_VIDEO), (CONFIG_4K, ROWS_4K_VIDEO),
    (CONFIG_4K_LONG, ROWS_4K_LONG_VIDEO),
])
def test_video_rows_match_the_latent_grid(canvas, want):
    """video rows = ((L-1)//8 + 1) * (H//32) * (W//32), comfy EmptyLTXVLatentVideo."""
    width, height, length = canvas
    got = ltx25_activation.video_rows(width, height, length)
    if got != want:
        pytest.fail(f"{canvas} gives {got} video rows, expected {want}")


def test_audio_rows_are_seconds_times_twenty_five():
    """25 latents/s = 16000 sample rate / 160 mel hop / 4 downsample."""
    if ltx25_activation.audio_rows(121, 24.0) != ROWS_AUDIO_121:
        pytest.fail("121 frames at 24 fps is 126 audio rows")
    # The comfy default that pads: 97 frames at 25 fps is 97 rows, odd.
    if ltx25_activation.audio_rows(97, 25.0) != 97:
        pytest.fail("97 frames at 25 fps is 97 audio rows")
    if ltx25_activation.audio_rows(121, 0):
        pytest.fail("a zero frame rate must charge nothing, not divide by zero")


def test_batch_multiplies_both_streams():
    """Batch multiplies activation streams one for one; LTX is not batch-1."""
    width, height, length = CONFIG_A
    if ltx25_activation.video_rows(width, height, length, 2) != 2 * ROWS_A_VIDEO:
        pytest.fail("batch must multiply video rows")
    if ltx25_activation.audio_rows(length, FPS, 3) != 3 * ROWS_AUDIO_121:
        pytest.fail("batch must multiply audio rows")


def test_each_stream_shards_and_pads_on_its_own():
    """adapters/ltx.py shards each stream with its own shard_seq call, so an odd
    audio total pads on the audio axis alone and cannot borrow an even video
    total's parity."""
    rows = ltx25_activation.LTXRows(video=6_409, audio=97, resolved=True)
    if rows.video_per_rank(2) != 3_205 or rows.audio_per_rank(2) != 49:
        pytest.fail("each axis rounds up on its own; pad rows are attended")


# 2. The calibration posture: the shipped constants against the series.

def test_the_shipped_row_sum_sits_just_above_the_measured_slope():
    """One world-2 series constrains S + 2R and nothing finer. The shipped
    pair must stay above it, so the guard never prices a row below what the
    hardware charged, and within 1.35x of it, so the margin stays small."""
    shipped = (ltx25_calibration.LTX_SHARDED_VIDEO_ROW_BYTES
               + 2 * ltx25_calibration.LTX_REPLICATED_VIDEO_ROW_BYTES)
    ratio = shipped / MEASURED_ROW_SUM
    if not 1.0 < ratio < 1.35:
        pytest.fail(
            f"shipped {shipped} B/row against the measured "
            f"{MEASURED_ROW_SUM:.0f} B/row is {ratio:.2f}x; the calibration "
            "wants a margin between 1.0 and 1.35")


def test_the_h3_slope_is_not_transferable_to_ltx():
    """Never borrow the H3 slope: h3_calibration's block factor at LTX's width
    is 2.7x the measured LTX cost and would refuse shapes this rig completes,
    so the LTX block factor is fitted."""
    from dgx_monarch import h3_calibration

    if (h3_calibration.H3_SHARDED_ROW_BYTES
            <= 2 * ltx25_calibration.LTX_SHARDED_VIDEO_ROW_BYTES):
        pytest.fail("the H3 row is no longer far above LTX's; recheck which "
                    "constant moved before trusting either")


def test_the_rank_floor_is_the_measured_worker_footprint():
    """26.80 GiB settled minus the 20.03 GiB file is CUDA context, torch, NCCL
    and worker runtime. Only about 0.85 GiB of it exists before the load, so
    the loader node charges memory not yet spent."""
    ratio = ltx25_calibration.LTX_RANK_FLOOR_BYTES / MEASURED_FLOOR
    if not 0.95 < ratio < 1.05:
        pytest.fail(f"the shipped floor is {ratio:.2f}x the measured "
                    f"{MEASURED_FLOOR / _GIB:.2f} GiB")


@pytest.mark.parametrize("canvas,per_rank,measured", [
    (CONFIG_B, 16_320, MEASURED_ACT_M1), (CONFIG_M2, 28_800, MEASURED_ACT_M2)])
def test_the_whole_charge_is_a_modest_margin_over_the_measured_footprint(
        canvas, per_rank, measured):
    """The loader charges floor + activations, the quantity the series
    measured, so this ratio is the calibration's safety factor."""
    rows = _rows(canvas)
    if rows.video_per_rank(2) != per_rank:
        pytest.fail(f"{canvas} is {rows.video_per_rank(2)} rows/rank, not {per_rank}")
    charged = ltx25_activation.activation_bytes(rows, 2)
    ratio = charged / (MEASURED_FLOOR + measured)
    if not 1.05 < ratio < 1.30:
        pytest.fail(f"{canvas} charges {charged / _GIB:.1f} GiB against a "
                    f"measured {(MEASURED_FLOOR + measured) / _GIB:.1f} GiB, "
                    f"{ratio:.2f}x")


def test_the_audio_row_is_derived_from_the_dimension_ratio():
    """126 audio rows against 130,560 video rows at 4K is too few to fit
    separately, so the audio terms are declared as half the video terms."""
    if (2 * ltx25_calibration.LTX_SHARDED_AUDIO_ROW_BYTES
            != ltx25_calibration.LTX_SHARDED_VIDEO_ROW_BYTES):
        pytest.fail("audio is 2048 wide against video's 4096; keep the ratio")
    if (2 * ltx25_calibration.LTX_REPLICATED_AUDIO_ROW_BYTES
            != ltx25_calibration.LTX_REPLICATED_VIDEO_ROW_BYTES):
        pytest.fail("the replicated audio term must follow the same ratio")


# 3. The verdicts this calibration is pinned to.

def _settled_need(canvas, sp, dit, te, *, te_credited):
    """The settled window's projection, exactly as loader_preflight forms it:
    the DiT against MemAvailable minus the reserve and the unspent stack."""
    activation = ltx25_activation.activation_bytes(_rows(canvas), sp)
    stack = activation if te_credited else (
        activation + int(te * loader_graph.TE_RESIDENT_RATIO)
        + VAE_VIDEO + VAE_AUDIO)
    return dit + RESERVE + stack


@pytest.mark.parametrize("name,canvas,sp,dit,te,avail,credited", [
    # Every leg the sealed ladder and the calibration series completed.
    ("ladder Config A int8 uly2", CONFIG_A, 2, DIT_INT8, TE_INT8, 94.6, True),
    ("ladder Config B int8 uly2", CONFIG_B, 2, DIT_INT8, TE_INT8, 94.6, True),
    ("ladder Config A bf16 uly2", CONFIG_A, 2, DIT_BF16, TE_BF16, 82.0, True),
    ("ladder Config B bf16 uly2", CONFIG_B, 2, DIT_BF16, TE_INT8, 94.6, True),
    ("calibration M2 int8 uly2", CONFIG_M2, 2, DIT_INT8, TE_INT8, 94.6, True),
    ("Config B single rank int8", CONFIG_B, 1, DIT_INT8, TE_INT8, 94.6, True),
    # Native 4K at 121 frames completed on hardware: it is not oversized.
    ("4K 121f bf16, cold graph", CONFIG_4K, 2, DIT_BF16, TE_BF16, 112.0, False),
    ("4K 121f bf16, warm encoder", CONFIG_4K, 2, DIT_BF16, TE_BF16, 80.5, True),
    ("4K 121f bf16, int8 encoder", CONFIG_4K, 2, DIT_BF16, TE_INT8, 94.4, True),
])
def test_every_shape_this_rig_completed_still_fits(
        name, canvas, sp, dit, te, avail, credited):
    need = _settled_need(canvas, sp, dit, te, te_credited=credited)
    if need > avail * _GIB:
        pytest.fail(f"{name} would now be refused: needs {need / _GIB:.1f} GiB "
                    f"against {avail} GiB available")


@pytest.mark.parametrize("name,dit,te,avail,credited", [
    ("cold graph, bf16 encoder", DIT_BF16, TE_BF16, 112.0, False),
    ("warm bf16 encoder", DIT_BF16, TE_BF16, 80.5, True),
    ("warm int8 encoder", DIT_BF16, TE_INT8, 94.4, True),
])
def test_the_oversized_4k_class_shape_refuses_in_every_execution_order(
        name, dit, te, avail, credited):
    """3840x2176 at 361 frames is 375,360 video rows. Comfy gives no order
    guarantee between the text-encoder loader and the DiT loader, so the
    refusal must hold whether or not the encoder is already in the reading,
    and on either encoder tier."""
    need = _settled_need(CONFIG_4K_LONG, 2, dit, te, te_credited=credited)
    margin = (need - avail * _GIB) / _GIB
    if margin < 10:
        pytest.fail(f"{name}: needs {need / _GIB:.1f} GiB against {avail} GiB, "
                    f"only {margin:.1f} GiB past the wall; the proof shape must "
                    "clear it by at least 10")


def test_the_refusal_threshold_is_text_encoder_dependent():
    """A shape near the line flips on the encoder tier alone, which is why the
    calibration comment states its verdicts per tier."""
    bf16 = _settled_need(CONFIG_4K_LONG, 2, DIT_BF16, TE_BF16, te_credited=False)
    int8 = _settled_need(CONFIG_4K_LONG, 2, DIT_BF16, TE_INT8, te_credited=False)
    if not 10 * _GIB < bf16 - int8 < 14 * _GIB:
        pytest.fail(f"the encoder tiers differ by {(bf16 - int8) / _GIB:.1f} GiB")


def test_the_row_bound_inverts_the_charge():
    """`max_video_rows` is the charge read backwards, so an operator can turn a
    shortfall into rows. It must never name a total that does not fit."""
    rows = _rows(CONFIG_4K_LONG)
    budget = ltx25_activation.activation_bytes(_rows(CONFIG_4K), 2)
    bound = ltx25_activation.max_video_rows(budget, 2, rows.audio)
    fits = ltx25_activation.LTXRows(video=bound, audio=rows.audio, resolved=True)
    if ltx25_activation.activation_bytes(fits, 2) > budget:
        pytest.fail(f"{bound} rows does not fit its own budget")
    over = ltx25_activation.LTXRows(video=bound + 2, audio=rows.audio, resolved=True)
    if ltx25_activation.activation_bytes(over, 2) <= budget:
        pytest.fail(f"{bound} rows is not the largest that fits")
    if ltx25_activation.max_video_rows(1, 2, rows.audio):
        pytest.fail("a budget below the rank floor admits no rows at all")


# 4. The loader-site wiring.

def test_the_graph_term_reaches_the_settled_window(artifacts):
    terms = _terms(_graph(CONFIG_B), sp=2)
    want = ltx25_activation.activation_bytes(_rows(CONFIG_B), 2)
    if terms.activation_bytes != want:
        pytest.fail(f"the graph term is {terms.activation_bytes}, expected {want}")
    if terms.total != (terms.te_bytes + terms.vae_bytes + terms.ref_bytes
                       + terms.activation_bytes):
        pytest.fail("the activation term must be inside `total`, which is what "
                    "puts it in the settled window")
    if not terms.activation_note:
        pytest.fail("the term must carry a note naming what it could not see")


def test_the_sharded_half_falls_with_the_sequence_degree(artifacts):
    one = _terms(_graph(CONFIG_B), sp=1).activation_bytes
    two = _terms(_graph(CONFIG_B), sp=2).activation_bytes
    if not two < one:
        pytest.fail("sequence parallelism must reduce the per-rank charge")
    floor = ltx25_calibration.LTX_RANK_FLOOR_BYTES
    replicated = (ROWS_B_VIDEO * ltx25_calibration.LTX_REPLICATED_VIDEO_ROW_BYTES
                  + ROWS_AUDIO_121 * ltx25_calibration.LTX_REPLICATED_AUDIO_ROW_BYTES)
    if two <= floor + replicated:
        pytest.fail("the replicated terms and the rank floor never divide")


def test_the_largest_canvas_wins_not_the_sum(artifacts):
    """Canvases in one graph are alternative or sequential renders, so the peak
    is the biggest, not the total. Both orderings, because comfy's prompt dict
    has no guaranteed order."""
    biggest = _terms(_graph(CONFIG_B), sp=2).activation_bytes
    for graph in (_graph(CONFIG_A, second=CONFIG_B),
                  _graph(CONFIG_B, second=CONFIG_A)):
        if _terms(graph, sp=2).activation_bytes != biggest:
            pytest.fail("expected the largest canvas alone")


@pytest.mark.parametrize("prompt", [None, {}, "not a graph", 42,
                                    {"1": {"inputs": None}}])
def test_an_unreadable_graph_charges_nothing(artifacts, prompt):
    if _terms(prompt).activation_bytes:
        pytest.fail(f"an unreadable graph must charge nothing: {prompt!r}")


def test_a_graph_with_no_ltx_canvas_charges_nothing(artifacts):
    graph = _graph(CONFIG_B, video_canvas=False, audio_canvas=False)
    if _terms(graph).activation_bytes:
        pytest.fail("no canvas node means no declared render to charge")


def test_a_video_only_graph_still_prices(artifacts):
    """LTXV without the audio stream is the same family and the same DiT."""
    graph = _graph(CONFIG_B, audio_canvas=False)
    rows = ltx25_activation.rows_from_prompt_graph(graph)
    if rows.video != ROWS_B_VIDEO or rows.audio:
        pytest.fail(f"expected video rows only, got {rows}")


# --- guides, the rows a canvas node never declares ------------------------

GUIDE_ROWS_A = 510          # 17 x 30, one latent frame of the Config A grid
GUIDE_ROWS_TEMPLATE = 880   # 22 x 40, the shipped templates' 1280x704 canvas


@pytest.mark.parametrize("canvas,per_guide", [
    (CONFIG_A, GUIDE_ROWS_A),
    ((1280, 704, 121), GUIDE_ROWS_TEMPLATE),
    (CONFIG_B, 2_040),
])
@pytest.mark.parametrize("guides", [1, 2])
def test_each_guide_adds_one_latent_frame_of_video_rows(canvas, per_guide, guides):
    """LTXV Add Guide always appends (comfy LTXVAddGuide.append_keyframe) and
    encodes at the canvas latent resolution (LTXVAddGuide.encode), so the
    appended frame is one grid of rows. Two guides is the first-and-last-frame
    shape."""
    base = ltx25_activation.rows_from_prompt_graph(_graph(canvas)).video
    rows = ltx25_activation.rows_from_prompt_graph(_graph(canvas, guides=guides))

    if rows.video != base + guides * per_guide:
        pytest.fail(f"expected {base + guides * per_guide} video rows, got {rows.video}")
    if rows.guides != guides or rows.guide_video != guides * per_guide:
        pytest.fail(f"the guide charge is not reported back: {rows}")


def test_the_wave_two_ladder_shape_prices_exactly():
    """The 2026-08-12 record: Config A is 8160 rows, one guide 8670, two 9180
    (docs/VALIDATION.md, "Image conditioning: image to video and first and last
    frame"; raw: out/ltx25_wave2/ladder_state.md, 'Shapes')."""
    got = [ltx25_activation.rows_from_prompt_graph(
        _graph(CONFIG_A, guides=n)).video for n in (0, 1, 2)]
    if got != [8_160, 8_670, 9_180]:
        pytest.fail(f"expected the ladder's own row counts, got {got}")


def test_a_guide_prices_against_the_canvas_it_attaches_to(artifacts):
    """A guide on the smaller canvas must not inflate the one that already wins.

    Peak memory is the largest canvas plus its own guides. Charging a guide to
    a canvas it never touches would refuse a render that fits.
    """
    rows = ltx25_activation.rows_from_prompt_graph(
        _graph(CONFIG_A, second=CONFIG_B, guides=1, guide_on="10"))

    if rows.video != ROWS_B_VIDEO:
        pytest.fail(f"the big canvas took a small canvas' guide: {rows}")
    if rows.guides or rows.guide_video:
        pytest.fail(f"the winning canvas carries no guide of its own: {rows}")


def test_a_guide_on_the_winning_canvas_is_still_charged(artifacts):
    """The other half of the same rule: attribution must not drop every guide."""
    rows = ltx25_activation.rows_from_prompt_graph(
        _graph(CONFIG_A, second=CONFIG_B, guides=1, guide_on="30"))

    if rows.video != ROWS_B_VIDEO + 2_040 or rows.guides != 1:
        pytest.fail(f"a guide on the winning canvas must be charged: {rows}")


def test_integer_key_multi_canvas_graph_charges_the_linked_canvas(artifacts):
    """Programmatic graphs may use integer node IDs instead of JSON strings.

    The guide link must resolve to the actual integer canvas key. Normalizing
    only the link target to text loses that association and silently omits the
    guide charge when more than one canvas is present.
    """
    graph = _graph(CONFIG_A, second=CONFIG_B, guides=1, guide_on="30")
    integer_graph = {
        int(node_id): {
            **node,
            "inputs": {
                name: (
                    [int(value[0]), *value[1:]]
                    if isinstance(value, list) and value and str(value[0]).isdigit()
                    else value
                )
                for name, value in node["inputs"].items()
            },
        }
        for node_id, node in graph.items()
    }

    rows = ltx25_activation.rows_from_prompt_graph(integer_graph)

    assert rows.video == ROWS_B_VIDEO + 2_040
    assert rows.guides == 1
    assert rows.unattributed_guides == 0


def test_string_link_reaches_a_canonical_integer_canvas_key(artifacts):
    graph = _graph(CONFIG_A, second=CONFIG_B, guides=1, guide_on="30")
    integer_keys = {int(node_id): node for node_id, node in graph.items()}

    rows = ltx25_activation.rows_from_prompt_graph(integer_keys)

    assert rows.video == ROWS_B_VIDEO + 2_040
    assert rows.guides == 1


def test_a_canvas_at_integer_node_id_zero_still_takes_its_guide(artifacts):
    """Node id 0 is a legal key and a falsy value.

    A guide that resolves to canvas 0 must keep it. Treating the resolved id as
    a truth value falls through to the lone-canvas fallback, which is None here
    because the graph declares two canvases, so the guide charge lands nowhere.
    """
    graph = _graph(CONFIG_A, second=CONFIG_B, guides=1, guide_on="30")
    zero_keyed = {(0 if node_id == "30" else int(node_id)): node
                  for node_id, node in graph.items()}
    zero_keyed[40]["inputs"]["latent"] = [0, 0]

    rows = ltx25_activation.rows_from_prompt_graph(zero_keyed)

    assert ltx25_activation.guide_canvas_id(zero_keyed, 40) == 0
    assert rows.video == ROWS_B_VIDEO + 2_040
    assert rows.guides == 1
    assert rows.unattributed_guides == 0


def test_noncanonical_decimal_link_does_not_alias_an_integer_key(artifacts):
    graph = _graph(CONFIG_A, second=CONFIG_B, guides=1, guide_on="030")
    integer_keys = {int(node_id): node for node_id, node in graph.items()}

    rows = ltx25_activation.rows_from_prompt_graph(integer_keys)

    assert rows.video == ROWS_B_VIDEO
    assert rows.guides == 0
    assert rows.unattributed_guides == 1


def test_a_guide_clip_that_states_its_length_charges_every_frame(artifacts):
    """The multi-frame case: comfy crops to 8n + 1 frames (LTXVAddGuide.encode),
    which is the rounding latent_frames already applies, so 17 pixel frames are
    3 latent frames and cost three grids rather than one."""
    rows = ltx25_activation.rows_from_prompt_graph(
        _graph(CONFIG_A, guides=1, guide_frames=17))

    if rows.guide_video != 3 * GUIDE_ROWS_A:
        pytest.fail(f"expected 3 latent frames of guide rows, got {rows}")
    if rows.video != ROWS_A_VIDEO + 3 * GUIDE_ROWS_A:
        pytest.fail(f"the clip's rows must reach the total: {rows}")


@pytest.mark.parametrize("frames,latent", [(1, 1), (8, 1), (9, 2), (17, 3), (25, 4)])
def test_the_guide_frame_derivation_follows_comfys_crop(frames, latent):
    """Pinned against comfy's own rule so a crop change is caught here."""
    got = ltx25_activation.latent_frames(frames)
    if got != latent:
        pytest.fail(f"{frames} pixel frames must be {latent} latent, got {got}")


def test_the_shipped_wiring_walks_the_guide_chain_to_its_canvas(artifacts):
    """The committed first-and-last-frame template's own shape.

    Add Guide returns the longer latent, so the second guide reaches the canvas
    through the first (example_workflows/dgx-monarch-ltx25-flf2v.json: guide 13
    takes canvas 12, guide 14 takes guide 13). Both must land on the canvas.
    """
    rows = ltx25_activation.rows_from_prompt_graph(_graph(CONFIG_A, guides=2))

    if rows.guides != 2 or rows.guide_video != 2 * GUIDE_ROWS_A:
        pytest.fail(f"both chained guides must reach the canvas: {rows}")
    if rows.unattributed_guides:
        pytest.fail(f"the shipped chain leaves nothing unattributed: {rows}")


def test_a_guide_off_the_chain_still_charges_the_only_canvas(artifacts):
    """The fallback arm: one canvas leaves no room for ambiguity.

    A graph may route a guide's latent through a node this leaf cannot follow.
    With a single canvas declared there is nothing else the guide could attach
    to, so it is charged rather than dropped; the ambiguity rule applies only
    when a second canvas makes the answer a guess.
    """
    rows = ltx25_activation.rows_from_prompt_graph(
        _graph(CONFIG_A, guides=1, guide_latent="12"))

    if rows.guides != 1 or rows.guide_video != GUIDE_ROWS_A:
        pytest.fail(f"the lone canvas must take the guide: {rows}")
    if rows.video != ROWS_A_VIDEO + GUIDE_ROWS_A or rows.unattributed_guides:
        pytest.fail(f"the charge must reach the total: {rows}")


def test_an_ambiguous_guide_is_named_rather_than_guessed(artifacts):
    """Two canvases and a guide reaching neither: charging a guess would price
    a canvas the guide never touched, so it is reported instead."""
    rows = ltx25_activation.rows_from_prompt_graph(
        _graph(CONFIG_A, second=CONFIG_B, guides=1, guide_latent="12"))

    if rows.video != ROWS_B_VIDEO or rows.guides:
        pytest.fail(f"an ambiguous guide must not be charged: {rows}")
    if rows.unattributed_guides != 1:
        pytest.fail(f"an ambiguous guide must be reported: {rows}")
    _, detail = ltx25_activation.graph_activation_bytes(
        _graph(CONFIG_A, second=CONFIG_B, guides=1, guide_latent="12"), 2,
        family="ltx")
    if "reach no canvas" not in detail:
        pytest.fail(f"the note must admit the uncharged guide: {detail}")


def test_a_guide_with_no_video_canvas_charges_nothing(artifacts):
    """No grid resolved means no row count, and an invented one would refuse a
    render that fits."""
    rows = ltx25_activation.rows_from_prompt_graph(
        _graph(CONFIG_A, video_canvas=False, guides=2))
    if rows.guides or rows.guide_video:
        pytest.fail(f"a guide must not charge without a canvas grid: {rows}")


def test_a_linked_geometry_widget_makes_the_whole_graph_unresolved(artifacts):
    """A width, height or length converted to an input carries a link, not a
    number.

    Skipping that canvas would drop every guide on it too, while the audio
    canvas beside it still resolves, so the preflight would price the video
    render off the audio rows alone: 5.4 GiB short of the same graph with
    literal widgets (2026-09-09).
    """
    literal = _graph(CONFIG_A, guides=1)
    linked = {node_id: (
        {**node, "inputs": {**node["inputs"], "width": ["99", 0]}}
        if node["class_type"] == "EmptyLTXVLatentVideo" else node)
        for node_id, node in literal.items()}

    rows = ltx25_activation.rows_from_prompt_graph(linked)

    if rows.resolved or rows.video or rows.audio:
        pytest.fail(f"an unreadable video canvas must resolve nothing: {rows}")
    charged, detail = ltx25_activation.graph_activation_bytes(linked, 2, family="ltx")
    if charged:
        pytest.fail(f"an unresolved graph must charge nothing: {charged}")
    if "readable geometry" not in detail:
        pytest.fail(f"the note must say why nothing was charged: {detail}")
    if ltx25_activation.graph_activation_bytes(literal, 2, family="ltx")[0] <= 0:
        pytest.fail("the same graph with literal widgets must still price")


def test_an_audio_only_graph_still_prices_its_own_stream(artifacts):
    """The unresolved rule covers only a video canvas that declares itself and
    cannot be read; a graph with no video canvas keeps its audio rows."""
    rows = ltx25_activation.rows_from_prompt_graph(
        _graph(CONFIG_A, video_canvas=False))

    if not rows.resolved or rows.video or not rows.audio:
        pytest.fail(f"an audio-only graph must keep its audio rows: {rows}")


def test_the_charge_reaches_the_settled_window_and_the_refusal_text(artifacts):
    """Both halves of the guide charge: the bytes reach the settled window, and
    the note names the guide rows."""
    plain = _terms(_graph(CONFIG_A), sp=2)
    guided = _terms(_graph(CONFIG_A, guides=2), sp=2)

    if guided.activation_bytes <= plain.activation_bytes:
        pytest.fail("two guides must cost more than none")
    _, detail = ltx25_activation.graph_activation_bytes(
        _graph(CONFIG_A, guides=2), 2, family="ltx")
    if "2 guide(s) add 1020" not in detail:
        pytest.fail(f"the refusal text must name the guide charge: {detail}")
    _, plain_detail = ltx25_activation.graph_activation_bytes(
        _graph(CONFIG_A), 2, family="ltx")
    if "guide" in plain_detail:
        pytest.fail(f"a guideless graph must claim no guide charge: {plain_detail}")


def test_the_activation_term_is_family_scoped(artifacts):
    """DRIVER_STACK_FAMILIES grows by one row for each family that registers,
    and a dual-model graph loads two checkpoints from one prompt, so no family
    may inherit another's rows."""
    if _terms(_graph(CONFIG_B), family="krea2").activation_bytes:
        pytest.fail("another family must not inherit the LTX charge")
    if ltx25_activation.graph_activation_bytes(
            _graph(CONFIG_B), 1, family="minimax_h3")[0]:
        pytest.fail("graph_activation_bytes must be family scoped")
    if loader_graph.graph_activation_bytes(
            _graph(CONFIG_B), 1, family="krea2")[0]:
        pytest.fail("the dispatcher must charge an unregistered family nothing")
    if loader_graph.driver_stack_terms(_graph(CONFIG_B)).activation_bytes:
        pytest.fail("the default call must charge no activation term")


def test_the_h3_charge_still_routes_through_the_shared_dispatcher():
    """One call site, two families: the LTX row must not displace H3's."""
    from dgx_monarch import h3_activation

    registered = loader_graph._ACTIVATION_FAMILIES
    if set(registered) != {h3_activation.H3_FAMILY, ltx25_activation.LTX_FAMILY}:
        pytest.fail(f"unexpected activation registry: {sorted(registered)}")


@pytest.mark.parametrize("preset,world,want", [
    ("uly2", 2, 2), ("ring2", 2, 2), ("uly2+fsdp", 2, 2), ("auto", 2, 2),
    ("cfg2", 2, 1), ("dp2", 2, 1), ("single", 2, 1), ("uly2", 1, 1),
])
def test_only_a_sequence_degree_divides_the_row_axis(preset, world, want):
    """cfg and dp give every rank the whole sequence, so they price at one.
    auto prices at 2, capped at world, because the ltx auto rule is ulysses 2 at
    every canvas size and the packed path refuses cfg and dp outright."""
    mesh = _Mesh(preset=preset, world=world)
    got = loader_graph.activation_sp_degree(mesh, "ltx")
    if got != want:
        pytest.fail(f"{preset} at world {world} priced at {got}, expected {want}")


def test_an_unreadable_mesh_prices_at_one():
    if ltx25_activation.sp_degree_for_mesh(object()) != 1:
        pytest.fail("unknown topology state must keep the larger charge")


@pytest.mark.parametrize("env", ["activation", "driver"])
def test_either_kill_switch_stands_the_charge_down(monkeypatch, artifacts, env):
    """An operator who turned either preflight off must not still be charged
    for it at the loader node."""
    name = (mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV if env == "activation"
            else driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV)
    monkeypatch.setenv(name, "1")
    terms = _terms(_graph(CONFIG_4K_LONG), sp=2)
    if terms.activation_bytes:
        pytest.fail(f"{name} must remove the charge entirely")
    if name not in terms.activation_note:
        pytest.fail(f"the note must say which switch removed it: "
                    f"{terms.activation_note}")


# 5. End to end at the shipped loader node.

def test_an_oversized_request_refuses_before_any_load(monkeypatch, loader_rig):
    """A 4K-class bf16 request gets a typed capacity refusal at the loader
    node, before anything is placed."""
    _set_mem(monkeypatch, 112.0)
    message = _refuse(_Mesh(), DIT_BF16_NAME, _graph(CONFIG_4K_LONG))
    tag = parse_refusal_tag(message)
    if tag is None or tag.refusal_class != "C":
        pytest.fail(f"expected a class C capacity refusal: {message}")
    if tag.guard != loader_preflight.LOADER_GUARD:
        pytest.fail(f"expected the existing loader guard, got {tag.guard!r}")
    if GUARDS[loader_preflight.LOADER_GUARD].refusal_class != "C":
        pytest.fail("the guard vocabulary must agree with the tag")
    for fragment in (str(ROWS_4K_LONG_VIDEO), "rows/rank", "render activations",
                     "shortfall converts straight into rows"):
        if fragment not in message:
            pytest.fail(f"the bound must reach the operator, missing "
                        f"{fragment!r}: {message}")
    if mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV not in message and (
            driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV not in message):
        pytest.fail(f"the refusal must name its escape hatch: {message}")


def test_a_config_a_render_still_clears_the_loader(monkeypatch, loader_rig):
    """Config A on the int8 DiT and int8 encoder at world 2: the guard must
    clear every render the ladder accepted."""
    _set_mem(monkeypatch, 112.0)
    graph = _graph(CONFIG_A, dit=DIT_INT8_NAME, te=TE_INT8_NAME)
    if loader_preflight.preflight_loader_footprint(
            _Mesh(), DIT_INT8_NAME, {}, graph) is not None:
        pytest.fail("a fitting Config A load must return None, not a rescue")


def test_native_4k_at_121_frames_is_not_oversized(monkeypatch, loader_rig):
    """This shape completed on hardware, so the guard must not refuse it;
    only the borrowed H3 slope called it oversized."""
    _set_mem(monkeypatch, 112.0)
    if loader_preflight.preflight_loader_footprint(
            _Mesh(), DIT_BF16_NAME, {}, _graph(CONFIG_4K)) is not None:
        pytest.fail("4K at 121 frames must clear the loader unaided")


def test_the_warm_re_render_is_not_charged_for_resident_weights(
        monkeypatch, loader_rig):
    """A cleared load memoizes its checkpoint, so the second loader node in a
    session prices the same graph without the DiT. The activation term stays,
    because activations are not spent at loader time."""
    _set_mem(monkeypatch, 112.0)
    graph = _graph(CONFIG_A, dit=DIT_INT8_NAME, te=TE_INT8_NAME)
    mesh = _Mesh()
    loader_preflight.preflight_loader_footprint(mesh, DIT_INT8_NAME, {}, graph)
    # The memo is keyed on the artifact and the FSDP world it was priced under
    # (0 is the whole-file price), so a load admitted at a shard price cannot
    # credit a later whole-file load (2026-09-09).
    if loader_preflight._memo_key(DIT_INT8_NAME, 0) not in loader_preflight._CLEARED_UNETS:
        pytest.fail("a cleared load must memoize its checkpoint")
    # Second pass at a memory level that only a credited weight term clears.
    _set_mem(monkeypatch, 26.0)
    if loader_preflight.preflight_loader_footprint(
            mesh, DIT_INT8_NAME, {}, graph) is not None:
        pytest.fail("the warm re-render must not be charged for its resident "
                    "weights a second time")


def test_the_stack_is_credited_once_across_a_dual_loader_graph(
        monkeypatch, loader_rig):
    """Two loader nodes in one graph name the same encoder and VAEs; charging
    that stack twice is a false refusal, so the memo credits it."""
    _set_mem(monkeypatch, 112.0)
    graph = _graph(CONFIG_A, dit=DIT_INT8_NAME, te=TE_INT8_NAME)
    mesh = _Mesh()
    loader_preflight.preflight_loader_footprint(mesh, DIT_INT8_NAME, {}, graph)
    terms = loader_graph.credited_terms(loader_graph.driver_stack_terms(
        graph, family="ltx", sp_degree=2))
    if terms.te_bytes or terms.vae_bytes:
        pytest.fail("the second loader node must credit the charged stack")
    if not terms.activation_bytes:
        pytest.fail("residency credits never clear the render activations")


def test_the_kill_switch_lets_the_oversized_request_through(
        monkeypatch, loader_rig):
    """The bypass is the shared activation env var, not a new one."""
    _set_mem(monkeypatch, 112.0)
    monkeypatch.setenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, "1")
    if loader_preflight.preflight_loader_footprint(
            _Mesh(), DIT_BF16_NAME, {}, _graph(CONFIG_4K_LONG)) is not None:
        pytest.fail("the documented bypass must clear this refusal")


def test_a_discrete_gpu_never_refuses(monkeypatch, loader_rig):
    """Host MemAvailable bounds nothing on a discrete device."""
    _set_mem(monkeypatch, 112.0)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: False)
    if loader_preflight.preflight_loader_footprint(
            _Mesh(), DIT_BF16_NAME, {}, _graph(CONFIG_4K_LONG)) is not None:
        pytest.fail("the guard applies to unified memory only")


def test_a_broken_estimator_never_blocks_a_load(monkeypatch, loader_rig):
    """Fail open everywhere except a complete over-budget estimate."""
    _set_mem(monkeypatch, 112.0)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("estimator is broken")

    monkeypatch.setattr(ltx25_activation, "rows_from_prompt_graph", _boom)
    terms = _terms(_graph(CONFIG_4K_LONG), sp=2)
    if terms.activation_bytes:
        pytest.fail("a broken row walk must charge nothing")
    if not terms.activation_note:
        pytest.fail("even the failure path owes the operator a note")


def test_a_linked_audio_frame_rate_omits_only_the_audio_term():
    # A comfy link in the frame_rate widget must not raise out of the graph
    # walk and stand the whole guard down; _rate treats it as zero and only
    # the audio term is omitted.
    assert ltx25_activation.audio_rows(121, ["3", 0]) == 0
    assert ltx25_activation.audio_rows(121, None) == 0
    assert ltx25_activation.audio_rows(121, 25.0) > 0


def test_sequence_degree_prices_ulysses_ring_not_world():
    class Mesh:
        topology_preset = "uly2+fsdp"
        world = 4

    assert ltx25_activation.sp_degree_for_mesh(Mesh()) == 2

    class Auto:
        topology_preset = "auto"
        world = 4

    assert ltx25_activation.sp_degree_for_mesh(Auto()) == 2

    class Ring:
        topology_preset = "ring2"
        world = 2

    assert ltx25_activation.sp_degree_for_mesh(Ring()) == 2


# --- the kill switch honors only an explicit on value ---------------------

@pytest.mark.parametrize("value", ["0", "false", "off", "no", "", "  "])
def test_an_off_spelling_keeps_the_ltx_activation_guard(monkeypatch, value):
    """The refusal tells the operator to set `=1`. `=0` must not disable it."""
    monkeypatch.setenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, value)
    monkeypatch.setattr(ltx25_activation, "_DISABLE_LOGGED", False)
    if ltx25_activation.preflight_disabled():
        pytest.fail(f"{value!r} turned the preflight off")


@pytest.mark.parametrize("value", ["1", "true", "ON", " yes "])
def test_an_on_spelling_disables_the_ltx_activation_guard(monkeypatch, value):
    monkeypatch.setenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, value)
    monkeypatch.setattr(ltx25_activation, "_DISABLE_LOGGED", False)
    if not ltx25_activation.preflight_disabled():
        pytest.fail(f"{value!r} must take the documented bypass")
