"""Apply empty-latent size tags exactly as stock samplers do.

Stock samplers pass downscale_ratio_spacial and downscale_ratio_temporal to
comfy.sample.fix_empty_latent_channels. It rescales an all-zero latent onto the
model's grid while preserving the requested output size. pack_latent carries
both tags as non-tensor keys; omitting them can change the render dimensions
(docs/VALIDATION.md, 2026-10-05).

Recording stubs check that each sampler passes both tags positionally (None
when absent), fixes the latent once, and derives noise and batch slices from
that result. The equalizer reads the corrected shape without another fix;
output drops the tags while the request retains them.

CPU canaries compare prepared latents, noise and shape arithmetic against real
comfy.sample for mismatched, matched and absent tags. They skip without ComfyUI;
the comfy-canary workflow runs them against ComfyUI master.
"""
from __future__ import annotations

import os
import sys
import types

import pytest
import torch

from dgx_monarch.actor import sampling
from dgx_monarch.nodes import render_preflight, render_result
from dgx_monarch.sampling_contract import (
    LATENT_SIZE_TAG_KEYS,
    fixed_latent_shape,
    latent_size_tags,
    latent_without_size_tags,
)
from module_location_helpers import from_checkout

SPATIAL, TEMPORAL = LATENT_SIZE_TAG_KEYS


def _cond(batch: int = 1):
    return [[torch.zeros(batch, 3, 4), {}]]


def _ksampler_request(latent: dict, **extra) -> dict:
    return {
        "latent": latent,
        "positive": _cond(),
        "negative": _cond(),
        "sampler_name": "euler",
        "scheduler": "normal",
        "steps": 1,
        "cfg": 1.0,
        "noise_seed": 7,
        **extra,
    }


def _custom_request(latent: dict, **extra) -> dict:
    return {
        "latent": latent,
        "sigmas": torch.tensor([1.0, 0.0]),
        "noise_seed": 7,
        "guider": {"kind": "basic", "positive": _cond()},
        "sampler_object": object(),
        **extra,
    }


class _RecordingNoise:
    """Stock Noise_RandomNoise's generate_noise, recording the LATENT it gets."""

    def __init__(self, seed: int):
        self.seed = seed
        self.seen: list[dict] = []

    def generate_noise(self, input_latent):
        import comfy.sample

        self.seen.append(dict(input_latent))
        return comfy.sample.prepare_noise(
            input_latent["samples"], self.seed, input_latent.get("batch_index"))


@pytest.fixture
def single_rank(monkeypatch):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)


@pytest.fixture
def captured_guider(monkeypatch):
    """Stand-in guider: records what reaches guider.sample and echoes the latent."""
    seen: dict = {}

    class _Guider:
        def sample(self, noise, latent_image, sampler, sigmas, **kwargs):
            seen["noise"], seen["latent"] = noise, latent_image
            return latent_image

    monkeypatch.setattr(sampling, "build_guider", lambda *_args, **_kwargs: _Guider())
    return seen


# Part 1: the call contract, against a recording stub.
FIXED_SHAPE = (4, 2, 3, 5)  # unlike any input below


@pytest.fixture
def stub_comfy(monkeypatch):
    """A comfy whose fix records its arguments and returns a fresh tensor."""
    rec: dict = {"fix": [], "fixed": [], "noise_from": [], "sample": {}}
    sample_mod = types.ModuleType("comfy.sample")

    def fix(model, value, *tags):
        rec["fix"].append((model, value, tags))
        fixed = torch.zeros(value.shape[0], *FIXED_SHAPE[1:])
        rec["fixed"].append(fixed)
        return fixed

    def prepare_noise(value, seed, batch_index=None):
        rec["noise_from"].append(value)
        return torch.full_like(value, float(seed))

    def sample(_model, noise, *_args, **_kwargs):
        latent_image = _args[6]
        rec["sample"].update(noise=noise, latent=latent_image)
        return latent_image

    sample_mod.fix_empty_latent_channels = fix
    sample_mod.prepare_noise = prepare_noise
    sample_mod.sample = sample
    samplers_mod = types.ModuleType("comfy.samplers")
    samplers_mod.KSampler = types.SimpleNamespace(SAMPLERS=("euler",), SCHEDULERS=("normal",))
    nested_mod = types.ModuleType("comfy.nested_tensor")
    nested_mod.NestedTensor = type("NestedTensor", (), {})
    comfy_mod = types.ModuleType("comfy")
    comfy_mod.__path__ = []
    comfy_mod.sample, comfy_mod.samplers, comfy_mod.nested_tensor = (
        sample_mod, samplers_mod, nested_mod)
    for name, module in (("comfy", comfy_mod), ("comfy.sample", sample_mod),
                         ("comfy.samplers", samplers_mod),
                         ("comfy.nested_tensor", nested_mod)):
        monkeypatch.setitem(sys.modules, name, module)
    return rec


@pytest.mark.parametrize(("tags", "expected"), [
    ({}, (None, None)),
    ({SPATIAL: 8}, (8, None)),
    ({TEMPORAL: 1764}, (None, 1764)),
    ({SPATIAL: 32, TEMPORAL: 8}, (32, 8)),
])
def test_both_tags_reach_the_fix_positionally_as_stock_reads_them(
        stub_comfy, single_rank, tags, expected):
    model = object()
    raw = torch.zeros(4, 4, 8, 8)
    sampling.run_ksampler(model, _ksampler_request({"samples": raw, **tags}))
    assert stub_comfy["fix"] == [(model, raw, expected)]
    assert latent_size_tags({"samples": raw, **tags}) == expected


def test_ksampler_fixes_once_and_draws_noise_from_the_fixed_latent(stub_comfy, single_rank):
    sampling.run_ksampler(object(), _ksampler_request(
        {"samples": torch.zeros(4, 4, 8, 8), SPATIAL: 8}))
    fixed, = stub_comfy["fixed"]  # a second fix would rescale an empty latent again
    assert stub_comfy["noise_from"] == [fixed]
    assert stub_comfy["sample"]["latent"] is fixed
    assert tuple(stub_comfy["sample"]["noise"].shape) == FIXED_SHAPE


def test_ksampler_without_noise_still_samples_the_fixed_grid(stub_comfy, single_rank):
    sampling.run_ksampler(object(), _ksampler_request(
        {"samples": torch.zeros(4, 4, 8, 8), SPATIAL: 8}, advanced={"add_noise": False}))
    assert len(stub_comfy["fix"]) == 1
    assert tuple(stub_comfy["sample"]["noise"].shape) == FIXED_SHAPE


def test_dp_slices_of_noise_and_latent_come_from_one_fixed_batch(stub_comfy, monkeypatch):
    monkeypatch.setattr(sampling, "_dp_info", lambda: (1, 2))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: False)
    request = _ksampler_request({"samples": torch.zeros(4, 4, 8, 8), SPATIAL: 8})
    request["positive"] = request["negative"] = _cond(4)
    sampling.run_ksampler(object(), request)
    fixed, = stub_comfy["fixed"]
    assert stub_comfy["noise_from"] == [fixed]  # full-batch noise, then the slice
    assert tuple(stub_comfy["sample"]["latent"].shape) == (2, *FIXED_SHAPE[1:])
    assert stub_comfy["sample"]["noise"].shape == stub_comfy["sample"]["latent"].shape


def test_custom_noise_object_sees_the_fixed_latent_with_its_tags(
        stub_comfy, single_rank, captured_guider):
    noise = _RecordingNoise(seed=7)
    latent = {"samples": torch.zeros(4, 4, 8, 8), SPATIAL: 8, TEMPORAL: 4, "batch_index": [0, 1, 2, 3]}
    sampling.run_custom(object(), _custom_request(latent, noise_object=noise))
    fixed, = stub_comfy["fixed"]
    seen, = noise.seen
    # Stock SamplerCustomAdvanced hands generate_noise its copy of the LATENT
    # with the fixed samples and the tags still on it.
    assert seen["samples"] is fixed
    assert (seen[SPATIAL], seen[TEMPORAL], seen["batch_index"]) == (8, 4, [0, 1, 2, 3])
    assert captured_guider["latent"] is fixed
    assert latent["samples"] is not fixed  # the request latent is not written back


def test_custom_seeded_noise_draws_from_the_fixed_latent(stub_comfy, single_rank, captured_guider):
    sampling.run_custom(object(), _custom_request(
        {"samples": torch.zeros(4, 4, 8, 8), SPATIAL: 8}, add_noise=True))
    fixed, = stub_comfy["fixed"]
    assert stub_comfy["noise_from"] == [fixed]
    assert captured_guider["latent"] is fixed
    assert captured_guider["noise"].shape == fixed.shape


def test_custom_zero_step_schedule_returns_the_fixed_grid(stub_comfy, single_rank):
    samples, out = sampling.run_custom(object(), _custom_request(
        {"samples": torch.zeros(4, 4, 8, 8), SPATIAL: 8}, sigmas=torch.tensor([1.0])))
    assert tuple(samples.shape) == FIXED_SHAPE
    assert tuple(out["samples"].shape) == FIXED_SHAPE


def _run_sample_for_equalize(monkeypatch, request: dict, latent_format) -> list:
    """Drive run_sample on a pure-cfg2 rank with a pad+mask family."""
    import dgx_monarch.adapters as adapters_pkg
    from dgx_monarch.actor import sample_protocol, worker_env
    from dgx_monarch.adapters import quant_activation_scale

    seen: list = []
    monkeypatch.setattr(quant_activation_scale, "plan_for_topology", lambda *_a, **_k: False)
    patcher = types.SimpleNamespace(
        model=types.SimpleNamespace(),
        get_model_object=lambda name: {"latent_format": latent_format}[name])
    monkeypatch.setattr(sample_protocol, "dual_model_cfg2_slot", lambda _w, _r: None)
    monkeypatch.setattr(sample_protocol, "_post_load_readiness", lambda _w: None)
    monkeypatch.setattr(sample_protocol.store_fsdp, "ensure",
                        lambda *_args, **_kwargs: (patcher, "warm"))
    monkeypatch.setattr(worker_env, "verify_sample_artifact_authorization",
                        lambda _w, _r, _fn: [{"id": "cond"}])
    monkeypatch.setattr(worker_env, "activate_accuracy_waivers", lambda _r: None)
    monkeypatch.setattr(worker_env, "assert_resident_artifact_identity", lambda *_a: None)
    monkeypatch.setattr(worker_env, "sample_rescue_consent", lambda _r: {})
    monkeypatch.setattr(worker_env, "accuracy_waiver_stamps", lambda: [])
    monkeypatch.setattr(adapters_pkg, "adapter_for", lambda _model, _override=None:
                        types.SimpleNamespace(cfg_cond_padding="pad+mask"))
    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))

    def equalize(_adapter, positive, negative, latent_samples):
        seen.append(latent_samples)
        return positive, negative

    worker = types.SimpleNamespace(
        _setup_key=object(), topology={"cfg": 2, "ulysses": 1, "ring": 1, "dp": 1, "fsdp": False},
        world=2, rank=0, store=types.SimpleNamespace(family_override=None),
        _attn=None, _inject_for_topology=object(), _check_uma_reserve=lambda: None)
    sample_protocol.run_sample(
        worker, request, None, None,
        equalize_cond_lengths=equalize,
        model_sampling_render_clone=lambda patched, _value: patched,
        request_artifact_identity=lambda name, _loras: {"id": name},
        run_ksampler=lambda *_a, **_k: (torch.zeros(1, 2), None),
        run_custom=lambda *_a, **_k: (torch.zeros(1, 2), None),
        latent_signature=lambda _s: {"sig": 1})
    return seen


# The four attributes stock's fix reads, for a 16x 2D format (Flux2's shape).
FORMAT_16X = types.SimpleNamespace(latent_channels=128, spacial_downscale_ratio=16,
                                   temporal_downscale_ratio=1, latent_dimensions=2)


def test_cfg_pad_equalizer_reads_the_grid_the_sampler_will_run(stub_comfy, monkeypatch):
    """pad+mask sizes its joint-key bias from the image grid; it must be the
    rescaled one, read by shape arithmetic: no fix runs before the sampler's
    own, no second latent is built, and the request latent stays raw."""
    raw = torch.zeros(1, 4, 8, 8)
    request = _ksampler_request({"samples": raw, SPATIAL: 8}, kind="ksampler",
                                model={"unet_name": "model.safetensors"})
    seen, = _run_sample_for_equalize(monkeypatch, request, FORMAT_16X)
    assert tuple(seen.shape) == (1, 128, 4, 4)
    assert seen.device.type == "meta"  # a shape, no storage
    assert stub_comfy["fix"] == []
    assert request["latent"]["samples"] is raw
    assert request["latent"][SPATIAL] == 8


def test_driver_output_drops_the_tags_and_the_request_keeps_them(monkeypatch):
    latent = {"samples": torch.zeros(1, 4, 8, 8), SPATIAL: 8, TEMPORAL: 4,
              "batch_index": [0], "noise_mask": torch.ones(1, 1, 8, 8),
              render_preflight.LATENT_DOWNSCALE_METADATA_KEY: 16}
    # The request latent (render_submit, fleet) must still carry the tags.
    request_latent = render_preflight.latent_without_topology_metadata(latent)
    assert (request_latent[SPATIAL], request_latent[TEMPORAL]) == (8, 4)

    monkeypatch.setattr(render_result, "verify_cross_rank_signatures", lambda *_a: None)
    monkeypatch.setattr(render_result, "read_latent_result", lambda value: value)
    rendered = torch.ones(1, 128, 4, 4)
    out = render_result._finish_render(
        [{"rank": 0, "dp_rank": 0, "latent": rendered}], types.SimpleNamespace(dp=1), latent)
    assert out["samples"] is rendered
    assert SPATIAL not in out and TEMPORAL not in out  # stock pops both
    assert out["batch_index"] == [0] and "noise_mask" in out
    assert render_preflight.LATENT_DOWNSCALE_METADATA_KEY not in out
    assert latent_without_size_tags(latent).keys() == latent.keys() - set(LATENT_SIZE_TAG_KEYS)


# Part 2: real comfy.sample on CPU, Monarch against stock's own two lines.
def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


def _from_checkout(module: object, comfy_dir: str) -> bool:
    """Does this module's file live under the checkout? A real comfy import
    also adds node_helpers, nodes, execution, latent_preview, server and more
    under no comfy prefix, and those must leave with it."""
    return from_checkout(module, comfy_dir)


@pytest.fixture(scope="module")
def real_comfy():
    """Import real comfy.sample and comfy.latent_formats on CPU, then undo it.

    The teardown drops every comfy module and every module loaded from the
    checkout, then restores the snapshot, so a later test that probes "is comfy
    importable" does not inherit a real package. comfy.cli_args parses argv on
    import, so ``--cpu`` is forced.
    """
    preserved_modules = {
        name: module for name, module in sys.modules.items() if _is_comfy_module(name)
    }
    for name in preserved_modules:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    comfy_dir = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        sys.argv = ["pytest-worker-latent-size-tags", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            comfy_sample = pytest.importorskip("comfy.sample")
            latent_formats = pytest.importorskip("comfy.latent_formats")
            pytest.importorskip("comfy.samplers")
        finally:
            sys.argv = original_argv
        yield comfy_sample, latent_formats
    finally:
        # Classify first, then pop: a namespace package's path re-resolves
        # through its parent in sys.modules while it is being read.
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or _from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved_modules)
        sys.path[:] = original_path


class _Patcher:
    """The two things the fix and the adapter lookup read off a ModelPatcher."""

    def __init__(self, latent_format):
        self.latent_format = latent_format
        self.model = types.SimpleNamespace()

    def get_model_object(self, name):
        assert name == "latent_format"
        return self.latent_format


def _stock(comfy_sample, patcher, latent: dict, seed: int):
    """The fix and noise lines of nodes.py common_ksampler: the reference."""
    latent_image = comfy_sample.fix_empty_latent_channels(
        patcher, latent["samples"], latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None))
    noise = comfy_sample.prepare_noise(latent_image, seed, latent.get("batch_index"))
    return latent_image, noise


# (case id, latent format, raw samples shape, tags, stock's fixed shape)
CASES = [
    # EmptyLatentImage (8x) wired into a 16x model: half the grid, not double the size.
    ("8x-tag-on-16x-image", "Flux2", (1, 4, 16, 16), {SPATIAL: 8}, (1, 128, 8, 8)),
    # EmptyLatentImage into a pixel-space model: the full pixel grid.
    ("8x-tag-on-pixel-space", "ChromaRadiance", (1, 4, 16, 16), {SPATIAL: 8}, (1, 3, 128, 128)),
    # A temporal tag twice the format's ratio doubles the frame axis.
    ("temporal-tag-on-video", "Wan21", (1, 16, 3, 8, 8), {TEMPORAL: 8}, (1, 16, 6, 8, 8)),
    # Cases the tags must not change: no tags, and tags that already match.
    ("no-tag", "Flux2", (1, 4, 16, 16), {}, (1, 128, 16, 16)),
    ("matched-tag", "Flux2", (1, 4, 16, 16), {SPATIAL: 16}, (1, 128, 16, 16)),
    ("matched-video-tags", "Wan21", (1, 16, 3, 8, 8), {SPATIAL: 8, TEMPORAL: 4}, (1, 16, 3, 8, 8)),
    ("8x-tag-on-8x-video", "Wan21", (1, 4, 8, 8), {SPATIAL: 8}, (1, 16, 1, 8, 8)),
]
UNCHANGED = {"no-tag", "matched-tag", "matched-video-tags", "8x-tag-on-8x-video"}


@pytest.mark.parametrize(("case", "fmt", "shape", "tags", "expected"), CASES,
                         ids=[case[0] for case in CASES])
def test_ksampler_prepares_stocks_latent_and_noise(
        real_comfy, single_rank, monkeypatch, case, fmt, shape, tags, expected):
    comfy_sample, latent_formats = real_comfy
    patcher = _Patcher(getattr(latent_formats, fmt)())
    latent = {"samples": torch.zeros(shape), **tags}
    seen: dict = {}

    def capture(_model, noise, *args, **_kwargs):
        seen["noise"], seen["latent"] = noise, args[6]
        return args[6]

    monkeypatch.setattr(comfy_sample, "sample", capture)
    samples, out = sampling.run_ksampler(patcher, _ksampler_request(latent))

    stock_latent, stock_noise = _stock(comfy_sample, patcher, latent, seed=7)
    assert tuple(stock_latent.shape) == expected
    assert torch.equal(seen["latent"], stock_latent)
    assert torch.equal(seen["noise"], stock_noise)
    assert tuple(samples.shape) == tuple(out["samples"].shape) == expected
    if case in UNCHANGED:
        old = comfy_sample.fix_empty_latent_channels(patcher, latent["samples"])
        assert tuple(seen["latent"].shape) == tuple(old.shape)


@pytest.mark.parametrize(("case", "fmt", "shape", "tags", "expected"), CASES,
                         ids=[case[0] for case in CASES])
@pytest.mark.parametrize("noise_kind", ["noise_object", "seeded"])
def test_custom_prepares_stocks_latent_and_noise(
        real_comfy, single_rank, captured_guider, case, fmt, shape, tags, expected, noise_kind):
    comfy_sample, latent_formats = real_comfy
    patcher = _Patcher(getattr(latent_formats, fmt)())
    latent = {"samples": torch.zeros(shape), **tags}
    extra = ({"noise_object": _RecordingNoise(seed=7)} if noise_kind == "noise_object"
             else {"add_noise": True})
    sampling.run_custom(patcher, _custom_request(latent, **extra))

    # SamplerCustomAdvanced: fix, then RandomNoise over the fixed copy.
    stock_latent, stock_noise = _stock(comfy_sample, patcher, latent, seed=7)
    assert tuple(stock_latent.shape) == expected
    assert torch.equal(captured_guider["latent"], stock_latent)
    assert torch.equal(captured_guider["noise"], stock_noise)


def test_dp_rank_slices_stocks_full_batch(real_comfy, monkeypatch):
    comfy_sample, latent_formats = real_comfy
    monkeypatch.setattr(sampling, "_dp_info", lambda: (1, 2))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: False)
    patcher = _Patcher(latent_formats.Flux2())
    latent = {"samples": torch.zeros(2, 4, 16, 16), SPATIAL: 8}
    seen: dict = {}

    def capture(_model, noise, *args, **_kwargs):
        seen["noise"], seen["latent"] = noise, args[6]
        return args[6]

    monkeypatch.setattr(comfy_sample, "sample", capture)
    request = _ksampler_request(latent)
    request["positive"] = request["negative"] = _cond(2)
    sampling.run_ksampler(patcher, request)
    stock_latent, stock_noise = _stock(comfy_sample, patcher, latent, seed=7)
    assert torch.equal(seen["latent"], stock_latent[1:])
    assert torch.equal(seen["noise"], stock_noise[1:])
    assert tuple(seen["latent"].shape) == (1, 128, 8, 8)


def test_a_populated_latent_ignores_a_mismatched_tag(real_comfy, single_rank, monkeypatch):
    comfy_sample, latent_formats = real_comfy
    patcher = _Patcher(latent_formats.Flux2())
    populated = torch.ones(1, 128, 16, 16)
    seen: dict = {}

    def capture(_model, noise, *args, **_kwargs):
        seen["latent"] = args[6]
        return args[6]

    monkeypatch.setattr(comfy_sample, "sample", capture)
    sampling.run_ksampler(patcher, _ksampler_request({"samples": populated, SPATIAL: 8}))
    assert seen["latent"] is populated


@pytest.mark.parametrize(("case", "fmt", "shape", "tags", "expected"), CASES,
                         ids=[case[0] for case in CASES])
def test_the_shape_arithmetic_matches_stocks_fix(real_comfy, case, fmt, shape, tags, expected):
    """The driver's estimates and the worker's pad equalizer read this
    arithmetic instead of building a latent; it must equal stock's result,
    channel axis included (the equalizer's pixel-space refusal reads it)."""
    comfy_sample, latent_formats = real_comfy
    latent_format = getattr(latent_formats, fmt)()
    latent = {"samples": torch.zeros(shape), **tags}
    stock = comfy_sample.fix_empty_latent_channels(
        _Patcher(latent_format), latent["samples"], *latent_size_tags(latent))
    assert fixed_latent_shape(latent["samples"], latent_format,
                              *latent_size_tags(latent)) == tuple(stock.shape) == expected


def test_the_shape_arithmetic_never_rescales_a_populated_latent(real_comfy):
    comfy_sample, latent_formats = real_comfy
    latent_format = latent_formats.Wan21()
    populated = torch.ones(1, 4, 3, 8, 8)
    stock = comfy_sample.fix_empty_latent_channels(
        _Patcher(latent_format), populated, 16, 8)
    assert fixed_latent_shape(populated, latent_format, 16, 8) == tuple(stock.shape)
    assert tuple(stock.shape) == (1, 4, 3, 8, 8)
