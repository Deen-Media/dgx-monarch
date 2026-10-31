"""The sample deadline grows with the render's packed rows per rank.

The driver used to stop waiting for a sample call after the larger of
``DGXM_SAMPLE_TIMEOUT`` (3600 s) and 120 s per step, so a 15 s MiniMax H3 clip
that samples for 80 minutes on one Spark was abandoned at the hour. The rule
now scales the per-step allowance by rows per rank for the families whose rows
the driver counts, with the old per-step floor, a ceiling, the env floor above
both, and the old rule wherever no row count exists.

No comfy, no GPU, no mesh: folder_paths, the sniff and the submit seams are
stubbed.
"""
from __future__ import annotations

import logging
import sys
import types
from contextlib import contextmanager

import pytest

from dgx_monarch import family_select, h3_rows, mesh_safety, sample_deadline
from dgx_monarch.adapters.detect import CheckpointSniffError
from dgx_monarch.nodes import render_rows
from dgx_monarch.topology import Topology

STEPS = 20

# Sampling on the pair and on one Spark, 1344x768, 20 steps, CFG 1, exact
# attention, pruned INT8 ConvRot (2026-10-30): packed rows per rank as the run
# records give them, and the measured sampling time in seconds.
MEASURED = {
    "pair 124 frames": (18_865, 7 * 60 + 31),
    "pair 243 frames": (36_895, 22 * 60 + 31),
    "pair 362 frames": (54_732, 45 * 60 + 19),
    "one Spark 243 frames": (73_790, 38 * 60 + 34),
    "one Spark 362 frames": (109_466, 79 * 60 + 33),
}


@pytest.fixture(autouse=True)
def _no_operator_floor(monkeypatch):
    monkeypatch.delenv(sample_deadline.SAMPLE_TIMEOUT_ENV, raising=False)


# 1. The rule at the measured points


@pytest.mark.parametrize("cell", sorted(MEASURED))
def test_every_measured_render_clears_its_deadline_at_least_twice_over(cell):
    rows_per_rank, sampled_s = MEASURED[cell]
    deadline = sample_deadline.sample_deadline(STEPS, rows_per_rank, 1, source="minimax_h3")
    assert deadline.seconds >= 2 * sampled_s, (cell, deadline)
    assert deadline.seconds <= sample_deadline.CEILING_S
    assert not deadline.capped


def test_the_15_second_clip_on_one_spark_gets_600_s_per_step():
    """The anchor: 239 s per step measured, 600 s allowed, 200 minutes in all,
    where the old rule gave 60 and the render needed 80."""
    rows_per_rank, sampled_s = MEASURED["one Spark 362 frames"]
    deadline = sample_deadline.sample_deadline(STEPS, rows_per_rank, 1, source="minimax_h3")
    assert deadline.step_s == sample_deadline.ANCHOR_STEP_S == 600.0
    assert deadline.seconds == STEPS * 600.0 == 12_000.0
    assert sampled_s / STEPS == pytest.approx(239, abs=0.5)
    old_rule = max(3600.0, STEPS * 120.0)
    assert old_rule < sampled_s < deadline.seconds


def test_the_5_second_clip_on_the_pair_keeps_todays_deadline():
    """Its rows ask for under 120 s per step, so the step floor and then the
    hour floor decide, exactly as before rows were counted."""
    rows_total = 2 * MEASURED["pair 124 frames"][0]
    deadline = sample_deadline.sample_deadline(STEPS, rows_total, 2, source="minimax_h3")
    assert deadline.rows_per_rank == 18_865
    assert deadline.step_s == sample_deadline.STEP_FLOOR_S
    assert deadline.seconds == max(3600.0, STEPS * 120.0) == 3600.0


def test_the_15_second_clip_on_the_pair_gets_100_minutes():
    """It sampled for 45 minutes and once passed the hour while the head's CPU
    was busy; its rows split over two ranks give half the one-Spark allowance."""
    deadline = sample_deadline.sample_deadline(STEPS, 109_463, 2, source="minimax_h3")
    assert deadline.rows_per_rank == 54_732
    assert deadline.seconds == pytest.approx(6000.0, abs=1.0)


def test_the_anchor_is_the_15_second_clips_packed_total():
    """109,466 rows: 107 latent frames of 1008 rows, 603 audio latents at two
    rows each, and a 404-token prompt."""
    video = (1, 24, h3_rows.latent_t_for_frames(362), 768 // 16, 1344 // 16)
    rows = h3_rows.rows_from_request(_h3_request(text=404), {"samples": _Shape(video)})
    assert (rows.video, rows.audio, rows.text) == (107_856, 1_206, 404)
    assert rows.total == sample_deadline.ANCHOR_ROWS_PER_RANK


# 2. Floor, step floor, ceiling and rounding


def test_a_small_render_keeps_the_hour_floor():
    deadline = sample_deadline.sample_deadline(4, 356, 1, source="minimax_h3")
    assert deadline.seconds == sample_deadline.DEFAULT_FLOOR_S == 3600.0


def test_the_step_floor_binds_when_rows_ask_for_less():
    deadline = sample_deadline.sample_deadline(40, 18_865, 1, source="minimax_h3")
    assert deadline.step_s == 120.0
    assert deadline.seconds == 40 * 120.0


@pytest.mark.parametrize(("steps", "rows"), [(40, 109_466), (20, 10 * 109_466)])
def test_the_ceiling_caps_the_row_scaled_deadline(steps, rows):
    deadline = sample_deadline.sample_deadline(steps, rows, 1, source="minimax_h3")
    assert steps * deadline.step_s > sample_deadline.CEILING_S
    assert deadline.seconds == sample_deadline.CEILING_S == 6 * 3600.0
    assert deadline.capped


def test_rows_per_rank_round_up_because_a_pad_row_costs_a_step():
    deadline = sample_deadline.sample_deadline(STEPS, 109_465, 2, source="minimax_h3")
    assert deadline.rows_per_rank == 54_733


def test_the_deadline_is_linear_in_rows_per_rank_above_the_step_floor():
    half = sample_deadline.sample_deadline(STEPS, 109_466 // 2, 1, source="minimax_h3")
    whole = sample_deadline.sample_deadline(STEPS, 109_466, 1, source="minimax_h3")
    assert half.step_s * 2 == pytest.approx(whole.step_s)


# 3. The operator's floor


def test_the_env_floor_raises_the_deadline(monkeypatch):
    monkeypatch.setenv(sample_deadline.SAMPLE_TIMEOUT_ENV, "14400")
    one_spark = sample_deadline.sample_deadline(STEPS, 109_466, 1, source="minimax_h3")
    pair = sample_deadline.sample_deadline(STEPS, 2 * 18_865, 2, source="minimax_h3")
    uncounted = sample_deadline.sample_deadline(STEPS)
    assert one_spark.seconds == pair.seconds == uncounted.seconds == 14_400.0
    assert one_spark.floor_from_env


def test_the_env_floor_wins_over_the_ceiling(monkeypatch):
    monkeypatch.setenv(sample_deadline.SAMPLE_TIMEOUT_ENV, "30000")
    deadline = sample_deadline.sample_deadline(100, 109_466, 1, source="minimax_h3")
    assert deadline.capped
    assert deadline.seconds == 30_000.0 > sample_deadline.CEILING_S


def test_a_lower_env_floor_lets_the_step_terms_decide(monkeypatch):
    monkeypatch.setenv(sample_deadline.SAMPLE_TIMEOUT_ENV, "60")
    assert sample_deadline.sample_deadline(STEPS).seconds == STEPS * 120.0
    assert sample_deadline.sample_deadline(STEPS, 109_466, 1).seconds == 12_000.0


@pytest.mark.parametrize("raw", ["", "an hour", "nan"])
def test_an_unreadable_env_floor_keeps_the_default_and_warns(monkeypatch, raw):
    monkeypatch.setenv(sample_deadline.SAMPLE_TIMEOUT_ENV, raw)
    with _records(sample_deadline.log) as records:
        deadline = sample_deadline.sample_deadline(STEPS)
    assert deadline.floor_s == 3600.0 and not deadline.floor_from_env
    assert [r for r in records if r.levelno == logging.WARNING
            and sample_deadline.SAMPLE_TIMEOUT_ENV in r.getMessage()]


# 4. No row count: the old rule, unchanged


@pytest.mark.parametrize("rows", [None, 0, -5])
@pytest.mark.parametrize("steps", [1, 20, 30, 50, 200])
def test_an_uncounted_render_keeps_the_old_rule(rows, steps):
    deadline = sample_deadline.sample_deadline(steps, rows, 2)
    assert deadline.seconds == max(3600.0, steps * 120.0)
    assert not deadline.counted and not deadline.capped
    # No ceiling either: 200 steps wait 24000 s, as they did.
    if steps == 200:
        assert deadline.seconds > sample_deadline.CEILING_S


# 5. The INFO line


def test_the_info_line_names_the_deadline_and_every_term():
    line = sample_deadline.describe(
        sample_deadline.sample_deadline(STEPS, 109_466, 1, source="minimax_h3"))
    assert line.startswith("sample deadline: 12000 s (3:20:00) from 20 steps x 600.0 s;")
    for term in ("floor 3600 s (default)", "109466 packed rows per rank",
                 "minimax_h3: 109466 rows over 1 rank(s)", "600 s per step at 109466",
                 "at least 120 s", "ceiling 21600 s"):
        assert term in line


def test_the_info_line_says_which_term_won(monkeypatch):
    capped = sample_deadline.describe(sample_deadline.sample_deadline(40, 109_466, 1, source="x"))
    floor = sample_deadline.describe(sample_deadline.sample_deadline(STEPS, 37_730, 2, source="x"))
    uncounted = sample_deadline.describe(sample_deadline.sample_deadline(50))
    assert "21600 s (6:00:00) from the ceiling" in capped
    assert "3600 s (1:00:00) from the floor" in floor
    assert "6000 s (1:40:00) from 50 steps x 120.0 s" in uncounted
    assert "the driver counts no packed rows for this render" in uncounted
    monkeypatch.setenv(sample_deadline.SAMPLE_TIMEOUT_ENV, "14400")
    raised = sample_deadline.describe(sample_deadline.sample_deadline(STEPS, 109_466, 1, source="x"))
    assert "14400 s (4:00:00) from the floor" in raised
    assert f"floor 14400 s ({sample_deadline.SAMPLE_TIMEOUT_ENV})" in raised


# 6. The row count the deadline reads


class _Shape:
    """A tensor stand-in: `shape` is all the counts read."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


def _h3_request(*, text: int, extras: dict | None = None) -> dict:
    return {"positive": [[_Shape((1, text, 4096)), extras or {}]]}


def _h3_latent(frames: int) -> dict:
    video = (1, 24, h3_rows.latent_t_for_frames(frames), 768 // 16, 1344 // 16)
    return {"samples": _Shape(video)}


def _model(unet_name: str = "model.safetensors", worker_args: dict | None = None):
    return types.SimpleNamespace(
        unet_name=unet_name,
        mesh=types.SimpleNamespace(worker_args=worker_args or {}))


def _install_checkpoint(monkeypatch, family, path="/models/model.safetensors"):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: path
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    def sniff(_path):
        if isinstance(family, BaseException):
            raise family
        return family, "int8"

    monkeypatch.setattr(render_rows, "sniff_checkpoint", sniff)


def test_minimax_h3_counts_its_packed_rows(monkeypatch):
    _install_checkpoint(monkeypatch, "minimax_h3")
    counted = render_rows.packed_rows_for_request(
        _model(), _h3_request(text=404), _h3_latent(362))
    assert counted == (109_466, "minimax_h3")


def test_the_reference_and_pose_families_count_the_preflights_tokens(monkeypatch):
    _install_checkpoint(monkeypatch, "wan_scail")
    reference = _Shape((1, 16, 1, 64, 64))
    pose = _Shape((1, 16, 4, 32, 32))
    request = {"positive": [(_Shape((1, 512, 4096)), {
        "reference_latents": [reference], "pose_video_latent": pose})]}
    video = (1, 16, 20, 64, 64)
    counted = render_rows.packed_rows_for_request(
        _model(), request, {"samples": _Shape(video)})
    expected = sum(mesh_safety.ref_pose_token_counts(video, [reference.shape], pose.shape))
    assert expected == 20 * 32 * 32 + 32 * 32 + 4 * 16 * 16
    assert counted == (expected, "wan_scail")


@pytest.mark.parametrize("family", ["flux", "wan", "ltx", "krea2"])
def test_a_family_without_a_row_count_returns_none(monkeypatch, family):
    _install_checkpoint(monkeypatch, family)
    assert render_rows.packed_rows_for_request(
        _model(), _h3_request(text=404), _h3_latent(362)) is None


def test_a_forced_family_counts_through_an_unreadable_header(monkeypatch):
    _install_checkpoint(monkeypatch, CheckpointSniffError("no header"))
    model = _model(worker_args={family_select.WORKER_ARG_KEY: "minimax_h3"})
    counted = render_rows.packed_rows_for_request(
        model, _h3_request(text=404), _h3_latent(362))
    assert counted == (109_466, "minimax_h3")


@pytest.mark.parametrize("sniff", [CheckpointSniffError("no header"), OSError("unreadable")])
def test_an_unreadable_header_returns_none_and_never_raises(monkeypatch, sniff):
    _install_checkpoint(monkeypatch, sniff)
    assert render_rows.packed_rows_for_request(
        _model(), _h3_request(text=404), _h3_latent(362)) is None


@pytest.mark.parametrize("path", [None, "/models/model.gguf"])
def test_a_missing_or_foreign_checkpoint_returns_none(monkeypatch, path):
    _install_checkpoint(monkeypatch, "minimax_h3", path=path)
    assert render_rows.packed_rows_for_request(
        _model(), _h3_request(text=404), _h3_latent(362)) is None


def test_no_folder_paths_returns_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "folder_paths", None)
    monkeypatch.setattr(render_rows, "sniff_checkpoint",
                        lambda _path: pytest.fail("sniffed without a checkpoint path"))
    assert render_rows.packed_rows_for_request(
        _model(), _h3_request(text=404), _h3_latent(362)) is None


@pytest.mark.parametrize("latent", [{}, {"samples": object()}, None])
def test_a_latent_without_a_shape_returns_none(monkeypatch, latent):
    _install_checkpoint(monkeypatch, "minimax_h3")
    assert render_rows.packed_rows_for_request(
        _model(), _h3_request(text=404), latent) is None


def test_a_malformed_reference_shape_returns_none(monkeypatch):
    _install_checkpoint(monkeypatch, "wan_scail")
    request = {"positive": [(_Shape((1, 512, 4096)), {
        "reference_latents": [_Shape((1, 16, 64))]})]}
    assert render_rows.packed_rows_for_request(
        _model(), request, {"samples": _Shape((1, 16, 20, 64, 64))}) is None


# 7. The deadline reaches the collect


@contextmanager
def _records(logger: logging.Logger):
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def _submit(monkeypatch, counted, topology: Topology | None = None):
    from dgx_monarch.nodes import common
    from dgx_monarch.nodes.pending import PendingRenderHandoff
    from render_sessions_helpers import _real_handle, _stub_direct_submit

    spec = _stub_direct_submit(monkeypatch, _real_handle(), lambda *_a, **_k: object())
    if topology is not None:
        monkeypatch.setattr(common, "resolve_topology", lambda *_a: (topology, False, "test"))
    seen = []

    def count(model, request, latent):
        seen.append((model.unet_name, request, latent))
        return counted

    monkeypatch.setattr(render_rows, "packed_rows_for_request", count)
    with _records(render_rows.log) as records:
        pending = common.submit_render(
            spec, {}, {"samples": object()}, cfg_value=1.0, steps_hint=STEPS,
            handoff=PendingRenderHandoff())
    try:
        lines = [r.getMessage() for r in records
                 if r.levelno == logging.INFO and r.getMessage().startswith("sample deadline:")]
        return pending._timeout, lines, seen
    finally:
        pending.abandon()


def test_a_counted_render_hands_the_row_scaled_deadline_to_the_collect(monkeypatch):
    timeout, lines, seen = _submit(monkeypatch, (109_466, "minimax_h3"))
    assert timeout == 12_000.0
    assert len(lines) == 1 and lines[0].startswith("sample deadline: 12000 s (3:20:00)")
    assert [name for name, _request, _latent in seen] == ["model.safetensors"]


def test_an_uncounted_render_hands_the_old_deadline_to_the_collect(monkeypatch):
    timeout, lines, _seen = _submit(monkeypatch, None)
    assert timeout == 3600.0
    assert len(lines) == 1 and "counts no packed rows" in lines[0]


def test_the_rows_divide_by_the_sequence_parallel_degree(monkeypatch):
    timeout, lines, _seen = _submit(
        monkeypatch, (109_466, "minimax_h3"), Topology(ulysses=2, world=2))
    assert timeout == 6000.0
    assert "109466 rows over 2 rank(s)" in lines[0]


def test_cfg_parallel_leaves_every_rank_all_the_rows(monkeypatch):
    timeout, lines, _seen = _submit(
        monkeypatch, (109_466, "wan_scail"), Topology(cfg=2, world=2))
    assert timeout == 12_000.0
    assert "109466 rows over 1 rank(s)" in lines[0]
