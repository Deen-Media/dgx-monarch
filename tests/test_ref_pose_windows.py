"""The reference and pose price (#47) counts the positive entries that can run in one step.

ComfyUI runs a conditioning entry only inside its timestep window
(``start_percent``/``end_percent``, as ConditioningSetTimestepRange writes
them), so entries whose windows do not overlap never share a model call: a
Wan SCAIL graph with two such entries, each carrying the same reference,
needs one entry's activations at a time, not both. Overlapping windows, and
entries with no window, can run together, and keep the sum. Every request is
built by the real sampler and guider nodes, KSampler and Sampler Custom alike,
and priced by the real check (``render_preflight._ref_pose_activation_preflight``).
"""
from __future__ import annotations

import itertools
import math
import random
from types import SimpleNamespace

import pytest
import torch

import test_driver_unload as rig
from dgx_monarch import mesh_safety
from dgx_monarch.nodes import common, guiders, render_preflight, render_rows, samplers
from dgx_monarch.nodes.common import ModelSpec
from test_driver_unload import _h3_rig

_isolated, box = rig._isolated, rig.box
_GIB = 2 ** 30
TEXT_ROWS = 64
VIDEO = (1, 16, 21, 64, 128)       # 43008 rows
REFERENCE = torch.zeros(1, 16, 1, 128, 128)   # 4096 rows, one picture shared by both entries
POSE = torch.zeros(1, 16, 21, 64, 128)        # 43008 rows
# One entry: 90112 rows x 80 KiB = 6.875 GiB; both: 94208 rows = 7.1875 GiB; the box reads 7 GiB.
READING = 7 * _GIB


class _Checked(Exception):
    """Raised where run_render leaves the render-site checks behind it."""


@pytest.fixture(autouse=True)
def _stop_after_the_checks(monkeypatch):
    monkeypatch.setattr(common, "_claim_adoption_context",
                        lambda: (_ for _ in ()).throw(_Checked()))
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: READING)


def _entry(window: tuple[float, float] | None) -> list:
    extras = {"reference_latents": [REFERENCE], "pose_video_latent": POSE}
    if window is not None:
        extras["start_percent"], extras["end_percent"] = window
    return [torch.zeros(1, TEXT_ROWS, 8), extras]


def _request(monkeypatch, tmp_path, kind: str, positive: list) -> dict:
    """The request a sampler node hands run_render for this positive conditioning."""
    model_rig = _h3_rig(monkeypatch, tmp_path, preset="uly2")
    model = ModelSpec(mesh=model_rig.model.mesh, unet_name=model_rig.model.unet_name)
    latent = {"samples": torch.zeros(*VIDEO)}
    negative = [[torch.zeros(1, TEXT_ROWS, 8), {}]]
    seen: list[dict] = []
    monkeypatch.setattr(samplers, "run_render", lambda _model, request, *_a, **_k: (
        seen.append(request), (_ for _ in ()).throw(_Checked()))[1])
    with pytest.raises(_Checked):
        if kind == "ksampler":
            samplers.DGXMonarchKSampler().sample(
                model, 1, 8, 2.0, "euler", "simple", positive, negative, latent)
        else:
            (guider,) = guiders.DGXMonarchCFGGuider().get_guider(model, positive, negative, 2.0)
            samplers.DGXMonarchSamplerCustom().sample(
                type("Noise", (), {"seed": 1})(), guider, object(),
                torch.tensor([1.0, 0.5, 0.0]), latent)
    return seen[0]


def _price(request: dict) -> None:
    """The real reference and pose check, the DiT already resident."""
    render_preflight._ref_pose_activation_preflight(
        "wan_scail", "/unused.safetensors", "scail.safetensors", request, VIDEO, True)


@pytest.mark.parametrize("second", [(0.6, 1.0),
                                    (0.501, 1.0)])   # a gap of the node's own 0.001 step
@pytest.mark.parametrize("kind", ["ksampler", "sampler_custom"])
def test_two_windows_that_never_meet_price_the_larger_entry(monkeypatch, tmp_path, kind, second):
    request = _request(monkeypatch, tmp_path, kind, [_entry((0.0, 0.5)), _entry(second)])
    assert render_rows.ref_pose_shapes(request) == ([REFERENCE.shape], POSE.shape)
    _price(request)  # 6.875 GiB within the 7 GiB reading: admitted


@pytest.mark.parametrize(("first", "second"), [((0.0, 0.6), (0.5, 1.0)),   # they overlap
                                               ((0.0, 0.5), (0.5, 1.0)),   # they meet at 0.5
                                               ((0.0, 0.5), (0.500005, 1.0)),  # closer than a sigma
                                               (None, (0.6, 1.0))])        # one runs throughout
@pytest.mark.parametrize("kind", ["ksampler", "sampler_custom"])
def test_windows_that_share_a_step_price_their_sum(monkeypatch, tmp_path, kind, first, second):
    request = _request(monkeypatch, tmp_path, kind, [_entry(first), _entry(second)])
    assert render_rows.ref_pose_shapes(request) == ([REFERENCE.shape] * 2, POSE.shape)
    with pytest.raises(mesh_safety.StockLoadCapacityError, match=r"estimated 7\.2 GiB"):
        _price(request)


@pytest.mark.parametrize("kind", ["ksampler", "sampler_custom"])
def test_one_entry_prices_as_before(monkeypatch, tmp_path, kind):
    request = _request(monkeypatch, tmp_path, kind, [_entry((0.2, 0.8))])
    assert render_rows.ref_pose_shapes(request) == ([REFERENCE.shape], POSE.shape)
    _price(request)


def test_a_sampler_custom_and_a_ksampler_request_price_the_same(monkeypatch, tmp_path):
    positive = [_entry((0.0, 0.5)), _entry((0.6, 1.0)), _entry((0.3, 0.7))]
    shapes = [render_rows.ref_pose_shapes(_request(monkeypatch, tmp_path, kind, positive))
              for kind in ("ksampler", "sampler_custom")]
    assert shapes[0] == shapes[1] == ([REFERENCE.shape] * 2, POSE.shape)


def test_an_inverted_window_never_runs_and_adds_nothing(monkeypatch, tmp_path):
    """ComfyUI skips an entry whose start comes after its end at every step. This
    one is inverted by far more than twice the window tolerance; a narrower one
    counts as running, which only over-charges."""
    request = _request(monkeypatch, tmp_path, "ksampler", [_entry((0.7, 0.2)), _entry(None)])
    assert render_rows.ref_pose_shapes(request) == ([REFERENCE.shape], POSE.shape)


@pytest.mark.parametrize("bounds", [
    pytest.param({"start_percent": math.nan}, id="NaN start"),
    pytest.param({"end_percent": math.nan}, id="NaN end"),
    pytest.param({"clip_start_percent": math.nan, "clip_end_percent": 1.0}, id="NaN clip start"),
    pytest.param({"start_percent": math.inf}, id="infinite start"),
    pytest.param({"start_percent": True}, id="a bool"),
    pytest.param({"start_percent": "0.5"}, id="a string"),
    pytest.param({"start_percent": 10 ** 400}, id="an int too large for a float"),
])
@pytest.mark.parametrize("kind", ["ksampler", "sampler_custom"])
def test_a_window_bound_that_is_not_a_finite_number_runs_throughout(
        monkeypatch, tmp_path, kind, bounds):
    """ComfyUI never skips an entry on a NaN bound (``percent_to_sigma`` gives
    NaN, which every comparison in ``get_area_and_mult`` reads as false), and
    ConditioningSetTimestepRange's range checks let NaN through."""
    entry = _entry((0.0, 0.5))
    entry[1].update(bounds)
    request = _request(monkeypatch, tmp_path, kind, [entry])
    assert render_rows.ref_pose_shapes(request) == ([REFERENCE.shape], POSE.shape)


SMALL_POSE = torch.zeros(1, 16, 1, 64, 128)  # 2048 rows


@pytest.mark.parametrize("kind", ["ksampler", "sampler_custom"])
def test_the_largest_pose_of_a_set_is_priced(monkeypatch, tmp_path, kind):
    """A with a large pose and no reference before C's window, B with a small
    pose throughout, C with a large pose and a reference: C's own call needs
    47,104 rows, and the first pose in entry order would have priced B's."""
    a = [torch.zeros(1, TEXT_ROWS, 8), {"pose_video_latent": POSE, "start_percent": 0.0,
                                         "end_percent": 0.4}]
    b = [torch.zeros(1, TEXT_ROWS, 8), {"pose_video_latent": SMALL_POSE}]
    c = [torch.zeros(1, TEXT_ROWS, 8), {"reference_latents": [REFERENCE], "pose_video_latent": POSE,
                                         "start_percent": 0.6, "end_percent": 1.0}]
    request = _request(monkeypatch, tmp_path, kind, [a, b, c])
    refs, pose = render_rows.ref_pose_shapes(request)
    assert (refs, pose) == ([REFERENCE.shape], POSE.shape)
    assert sum(mesh_safety.ref_pose_token_counts((1, 0, 0, 0, 0), refs, pose)) == 47_104


def _rows(refs: list, poses: list) -> int:
    """One set's rows: its references summed and its largest pose."""
    pose = max((mesh_safety.ref_pose_token_counts((1, 0, 0, 0, 0), (), p)[1] for p in poses),
               default=0)
    return mesh_safety.ref_pose_token_counts((1, 0, 0, 0, 0), refs)[0] + pose


# Two bounds closer than this are one step to the sampler (its float32 sigmas
# cannot tell them apart), so the oracle below reads every window as reaching
# this far past each of its bounds.
SIGMA_RESOLUTION = 1e-5


def test_the_tolerance_is_the_one_the_oracle_reads():
    assert render_rows.WINDOW_TOLERANCE == SIGMA_RESOLUTION


def _bound(rng: random.Random) -> float:
    """A window bound on the node's 0.01 grid, now and then nudged by less than
    a few sigma resolutions, so near-touching windows are drawn too."""
    nudge = rng.uniform(-3, 3) * SIGMA_RESOLUTION if rng.random() < 0.3 else 0.0
    return min(max(round(rng.random(), 2) + nudge, 0.0), 1.0)


def test_the_price_covers_every_set_that_can_share_a_step():
    """Random requests: the price is never below any entry's own call or any set
    of entries that can share a step, and equals the largest such set a scan of
    every window's reach and the points between finds."""
    rng = random.Random(197)  # noqa: S311 - a seeded test case generator, not a secret
    for _case in range(2000):
        entries, shaped = [], []
        for _ in range(rng.randint(1, 5)):
            refs = [(1, 16, 1, 2 * rng.randint(1, 32), 2 * rng.randint(1, 32))
                    for _ in range(rng.randint(0, 2))]
            pose = ((1, 16, rng.randint(1, 8), 2 * rng.randint(1, 32), 2 * rng.randint(1, 32))
                    if rng.random() < 0.7 else None)
            extras = {"reference_latents": [SimpleNamespace(shape=s) for s in refs]}
            if pose is not None:
                extras["pose_video_latent"] = SimpleNamespace(shape=pose)
            window = None
            if rng.random() < 0.7:
                start, end = sorted(_bound(rng) for _ in range(2))
                window = (start, end)
                extras["start_percent"], extras["end_percent"] = window
            entries.append([object(), extras])
            start, end = window or (0.0, 1.0)
            reach = (start - SIGMA_RESOLUTION, end + SIGMA_RESOLUTION)
            shaped.append((reach, refs, [pose] if pose is not None else []))
        refs, pose = render_rows.ref_pose_shapes({"positive": entries})
        price = sum(mesh_safety.ref_pose_token_counts((1, 0, 0, 0, 0), refs, pose))
        for _reach, own_refs, own_pose in shaped:
            assert price >= _rows(own_refs, own_pose)
        bounds = sorted({b for reach, _r, _p in shaped for b in reach})
        points = bounds + [(lo + hi) / 2 for lo, hi in itertools.pairwise(bounds)]
        scanned = 0
        for point in points:
            active = [(r, p) for (start, end), r, p in shaped if start <= point <= end]
            rows = _rows([s for r, _p in active for s in r], [s for _r, p in active for s in p])
            assert price >= rows
            scanned = max(scanned, rows)
        assert price == scanned
