"""MOSS-Image's cond branch: the Worker's bind and one reference cache per render.

Stock ``MOSSImage.extra_conds`` (comfy/model_base.py) computes the reference
images' keys and values once per conditioning through ``preprocess_refs``,
but only under ``torch.inference_mode()``, with no hooks and a method other
than ``index`` or ``index_timestep_zero``. Otherwise it hands the forward the
latents, and the forward reruns ``preprocess_refs`` on every model call.
ComfyUI enters inference mode around prompt execution, and a Worker's
sampling thread never does, so before the adapter's ``prepare_base_model`` a
one-GPU edit recomputed the references on both calls of every CFG step: the
same bits as stock, more work (the 2026-10-22 hardware ladder measured it).
The bind is FLUX.3's (tests/test_flux3_cond_branch.py): stock's
``extra_conds`` under inference mode, every inference tensor cloned out.

Every test runs real ComfyUI on the CPU, on the tiny vendor-layout model of
tests/test_moss_image_comfy_exactness.py. The Worker's own load hook binds a
one-GPU load. The Worker's KSampler path, on a thread outside inference mode,
calls ``preprocess_refs`` once per conditioning and never in the forward, and
its latent equals stock's sampler run under inference mode, method for method;
without the bind the same path recomputes the references on every call, to
the same latent. The bound conds equal stock's and hold no inference tensor.
The bind is idempotent and survives patcher and LoRA clones, and the
persistence canary takes it through the store's whole LoRA life cycle.

Off one GPU, a Worker whose process group holds more than one rank, the bind
refuses reference images with #130's class P text before stock's method
runs, so no rank calls ``preprocess_refs`` or the model for a render that
refuses, even when the driver could not read the header and let it through.
A group of one, which topology single and every Fleet KSampler job open,
keeps stock's cache, and a cluster render without references passes. The
text attention bias of ComfyUI 9fe7ef764 passes the bind on every topology:
stock's method only wraps it, and under sequence parallelism the Worker's
request scan before sampling and the sharded forward refuse it (#135).
"""
from __future__ import annotations

import inspect
import math
import os
import types
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.moss_image import MOSSImageAdapter
from dgx_monarch.adapters.moss_image_seams import REFERENCE_IMAGES_REFUSAL, worker_on_one_gpu
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from test_moss_image_comfy_exactness import (  # noqa: F401  (fixture re-exported)
    _LORA_UNITS,
    CHANNELS,
    VENDOR_TEXT_DIM,
    _base_model,
    _rank4_lora,
    comfy_moss,
)

_SINGLE = {"cfg": 1, "ulysses": 1, "ring": 1, "dp": 1, "fsdp": False}
# The two world-2 topologies that bind no sharded forward: before the cond
# bind refused, a driver that could not read the header let stock run the edit.
_UNSHARDED_CLUSTER = {"cfg2": {**_SINGLE, "cfg": 2}, "dp2": {**_SINGLE, "dp": 2}}
# Stock's reference methods (comfy_extras/nodes_moss_image.py's three edit
# formats and the default) and whether stock caches their keys and values.
METHODS = {None: True, "index": False, "index_timestep_zero": False,
           "index_timestep_zero_isolated": True, "index_timestep_zero_text_start": True}
STEPS, CFG, SEED = 3, 3.0, 5
POSITIVE_ROWS, NEGATIVE_ROWS = 12, 7   # unequal, so cond and uncond run as two calls


@pytest.fixture
def moss(request):
    """The exactness file's module-scoped ComfyUI import, on a tree with references."""
    moss = request.getfixturevalue("comfy_moss")
    if not hasattr(moss.model.MOSSImage, "preprocess_refs"):
        pytest.skip("this ComfyUI predates MOSS-Image reference images (60aade0bb)")
    return moss


def _inject(monkeypatch, patcher, topology=_SINGLE, world=1) -> None:
    """The Worker's load hook, on topology single by default; the registry picks the adapter."""
    from dgx_monarch.actor import partial_load_guard, store_fsdp
    from dgx_monarch.actor import worker as worker_mod

    monkeypatch.setattr(store_fsdp, "validate_injection", lambda *a, **k: None)
    monkeypatch.setattr(partial_load_guard, "install", lambda *a, **k: None)
    monkeypatch.setattr(worker_mod, "_maybe_compile_dit", lambda *a, **k: None)
    attn = SimpleNamespace(bind_capability=lambda *a: None, effective_kernel="TORCH_FLASH")
    worker = SimpleNamespace(topology=dict(topology), world=world, _attn=attn,
                             store=SimpleNamespace(family_override=None))
    worker_mod.GPUWorker._inject_for_topology(worker, patcher, "bf16", [], None)


def _conds(method="index_timestep_zero_isolated", *, refs=True, seed=3):
    """A positive and a negative of unequal lengths, each with the same reference."""
    generator = torch.Generator().manual_seed(seed)
    reference = torch.randn(1, CHANNELS, 2, 3, generator=generator)
    extra: dict = {}
    if refs:
        extra["reference_latents"] = [reference]
        if method is not None:
            extra["reference_latents_method"] = method
    positive = [[torch.randn(1, POSITIVE_ROWS, VENDOR_TEXT_DIM, generator=generator) * 0.1, dict(extra)]]
    negative = [[torch.randn(1, NEGATIVE_ROWS, VENDOR_TEXT_DIM, generator=generator) * 0.1, dict(extra)]]
    return positive, negative


def _count_preprocess_refs(patcher) -> list[bool]:
    """Each preprocess_refs call, as whether inference mode was on for it."""
    dm = patcher.model.diffusion_model
    stock = type(dm).preprocess_refs
    calls: list[bool] = []

    def counted(refs, *args, **kwargs):
        calls.append(torch.is_inference_mode_enabled())
        return stock(dm, refs, *args, **kwargs)

    dm.preprocess_refs = counted
    return calls


def _worker_sample(monkeypatch, patcher, positive, negative):
    """run_ksampler on a Worker render clone, on a fresh thread: inference mode off."""
    from dgx_monarch.actor import sampling

    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)
    request = {"latent": {"samples": torch.zeros(1, CHANNELS, 4, 4)},
               "positive": positive, "negative": negative, "sampler_name": "euler",
               "scheduler": "simple", "steps": STEPS, "cfg": CFG, "noise_seed": SEED}
    clone = sampling.model_sampling_render_clone(patcher, None)
    with ThreadPoolExecutor(1) as pool:
        assert not pool.submit(torch.is_inference_mode_enabled).result()
        samples, _out = pool.submit(sampling.run_ksampler, clone, request).result()
    return samples


def _stock_sample(patcher, positive, negative):
    """Stock's KSampler (nodes.common_ksampler's calls) inside inference mode, as a prompt runs."""
    import comfy.sample

    with torch.inference_mode():
        latent = comfy.sample.fix_empty_latent_channels(patcher, torch.zeros(1, CHANNELS, 4, 4))
        noise = comfy.sample.prepare_noise(latent, SEED, None)
        return comfy.sample.sample(patcher, noise, STEPS, CFG, "euler", "simple", positive, negative,
                                   latent, denoise=1.0, disable_pbar=True, seed=SEED)


def test_the_workers_load_hook_binds_extra_conds_on_a_one_gpu_load(monkeypatch, moss, tmp_path):
    import comfy.model_base

    patcher, _path = _base_model(moss, tmp_path)
    assert "extra_conds" not in vars(patcher.model)
    _inject(monkeypatch, patcher)
    bound = vars(patcher.model)["extra_conds"]
    cells = [cell.cell_contents for cell in bound.__func__.__closure__ or ()]
    assert comfy.model_base.MOSSImage.extra_conds in cells
    # Topology single binds no forward: the forward stays stock's.
    assert "_forward" not in vars(patcher.model.diffusion_model)


@pytest.mark.parametrize("method", list(METHODS), ids=[str(method) for method in METHODS])
def test_a_one_gpu_edit_computes_its_references_once_per_render_to_stocks_latent(monkeypatch, moss,
                                                                                tmp_path, method):
    """The pass-through after the Worker's load hook: two preprocess_refs calls a
    render (one per conditioning, in inference mode) where a caching method is set,
    none for the in-sequence methods, and stock's latent either way. Without the
    bind the same path calls preprocess_refs on every model call, to the same latent."""
    positive, negative = _conds(method)
    stock, _path = _base_model(moss, tmp_path)
    stock_calls = _count_preprocess_refs(stock)
    expected = _stock_sample(stock, positive, negative)

    bound, _path = _base_model(moss, tmp_path)
    _inject(monkeypatch, bound)
    bound_calls = _count_preprocess_refs(bound)
    samples = _worker_sample(monkeypatch, bound, positive, negative)

    unbound, _path = _base_model(moss, tmp_path)
    unbound_calls = _count_preprocess_refs(unbound)
    before = _worker_sample(monkeypatch, unbound, positive, negative)

    if METHODS[method]:
        assert stock_calls == bound_calls == [True, True]
        assert unbound_calls == [False] * (2 * STEPS)
    else:
        assert stock_calls == bound_calls == unbound_calls == []
    assert torch.isfinite(expected).all()
    assert torch.equal(samples, expected)
    assert torch.equal(before, expected)


def _walk(value):
    if isinstance(value, list):
        for item in value:
            yield from _walk(item)
    elif torch.is_tensor(value):
        yield value


CONDS_CASES = {**{str(method): {"method": method} for method in METHODS},
               "hooks": {"method": "index_timestep_zero_isolated", "hooks": object()},
               "no-reference": {"refs": False}}


@pytest.mark.parametrize("case", sorted(CONDS_CASES))
def test_the_bound_conds_equal_stocks_in_inference_mode_and_hold_no_inference_tensor(moss, tmp_path, case):
    """Off inference mode the bound call returns what stock returns inside it,
    every tensor an exact copy that is not an inference tensor; stock's own
    predicate keeps hooks and the in-sequence methods on the forward's branch.
    With no reference nothing of the model runs and the text tensor passes as is."""
    spec = dict(CONDS_CASES[case])
    hooks = spec.pop("hooks", None)
    positive, _negative = _conds(**spec)
    context, options = positive[0]
    kwargs = {"cross_attn": context, "noise": torch.zeros(1, CHANNELS, 4, 4),
              "device": torch.device("cpu"), **options}
    if hooks is not None:
        kwargs["hooks"] = hooks
    patcher, _path = _base_model(moss, tmp_path)
    model = patcher.model
    calls = _count_preprocess_refs(patcher)
    with torch.inference_mode():
        stock = model.extra_conds(**kwargs)
    MOSSImageAdapter().prepare_base_model(model)
    with ThreadPoolExecutor(1) as pool:
        bound = pool.submit(lambda: model.extra_conds(**kwargs)).result()

    cached = spec.get("refs", True) and hooks is None and METHODS[spec.get("method")]
    assert calls == ([True, True] if cached else [])
    assert set(bound) == set(stock)
    assert ("ref_kv" in bound) is bool(cached)
    assert ("ref_latents" in bound) is bool(spec.get("refs", True) and not cached)
    for key in bound:
        assert type(bound[key]) is type(stock[key]), key
        got, want = list(_walk(bound[key].cond)), list(_walk(stock[key].cond))
        assert len(got) == len(want) and all(torch.equal(a, b) for a, b in zip(got, want, strict=True)), key
        assert not any(tensor.is_inference() for tensor in got), key
    if not spec.get("refs", True):
        assert set(bound) == {"c_crossattn"} and bound["c_crossattn"].cond is context


def test_the_bind_is_idempotent_and_survives_a_patcher_clone_and_a_lora_clone(moss, tmp_path):
    import comfy.model_base
    import comfy.sd

    patcher, _path = _base_model(moss, tmp_path)
    for _ in range(3):
        MOSSImageAdapter().prepare_base_model(patcher.model)
    cells = [cell.cell_contents for cell in patcher.model.extra_conds.__func__.__closure__ or ()]
    assert comfy.model_base.MOSSImage.extra_conds in cells
    assert not any(isinstance(cell, types.MethodType) for cell in cells)
    positive, _negative = _conds()
    context, options = positive[0]
    kwargs = {"cross_attn": context, "noise": torch.zeros(1, CHANNELS, 4, 4),
              "device": torch.device("cpu"), **options}
    clone = patcher.clone()
    lora_clone, _clip = comfy.sd.load_lora_for_models(patcher, None, {}, 1.0, 0)
    for other in (patcher, clone, lora_clone):
        assert other.model is patcher.model
        with ThreadPoolExecutor(1) as pool:
            assert "ref_kv" in pool.submit(lambda m=other.model: m.extra_conds(**kwargs)).result()


def _inference_state(model) -> list[str]:
    return [name for name, value in model.state_dict(keep_vars=True).items()
            if torch.is_tensor(value) and value.is_inference()]


def test_the_bound_branch_survives_sample_bake_hot_swap_and_unbake(moss, tmp_path):
    """The persistence canary: preprocess_refs runs every block inside the
    inference-mode window, so a persistent tensor created there and written in
    place outside it would raise. Nothing raises, no state is an inference
    tensor, and the un-bake restores the file's own bytes."""
    import comfy.lora
    import comfy.sample
    from safetensors.torch import load_file

    from dgx_monarch.actor.model_store import ModelStore
    from dgx_monarch.actor.store_bake import _merge_and_free, lazy_swap
    from dgx_monarch.actor.unbake import live_tensor, restore_pristine

    patcher, path = _base_model(moss, tmp_path)
    MOSSImageAdapter().prepare_base_model(patcher.model)
    calls = _count_preprocess_refs(patcher)
    positive, negative = _conds()
    noise, latent = torch.randn(1, CHANNELS, 4, 4), torch.zeros(1, CHANNELS, 4, 4)

    def sample(model_patcher):
        with ThreadPoolExecutor(1) as pool:
            return pool.submit(comfy.sample.sample, model_patcher, noise, 2, CFG, "euler",
                               "simple", positive, negative, latent, seed=1).result()

    first = sample(patcher)
    assert torch.isfinite(first).all()
    assert calls == [True, True]
    assert _inference_state(patcher.model) == []

    key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
    patches = comfy.lora.load_lora(_rank4_lora(_LORA_UNITS), key_map)
    active = patcher.clone()
    active.add_patches(patches, 0.5)
    record = _merge_and_free(active, path)
    assert record is not None
    baked = sample(active)
    assert not torch.equal(baked, first)

    store = ModelStore()
    store.swap_verify = -1   # comfy's own bake verifies every key
    swapped = patcher.clone()
    swapped.add_patches(patches, 0.8)
    record = lazy_swap(store, SimpleNamespace(unbake=record, base_patcher=patcher),
                       swapped, os.path.basename(path))
    assert record is not None
    sample(swapped)
    assert _inference_state(patcher.model) == []

    restore_pristine(patcher.model, record)
    file_state = load_file(path)
    for key in {key[0] if isinstance(key, tuple) else key for key in patches}:
        assert torch.equal(live_tensor(patcher.model, key), file_state[key[len("diffusion_model."):]]), key
    assert torch.equal(sample(patcher), first)
    assert calls == [True] * 8


# --- Off one GPU: the bind refuses reference images before stock's method ------------


def _process_group(monkeypatch, world: int) -> None:
    """The process group a Worker's setup opens, at ``world`` ranks."""
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: world)


def _extra_conds_kwargs(**spec) -> dict:
    positive, _negative = _conds(**spec)
    context, options = positive[0]
    return {"cross_attn": context, "noise": torch.zeros(1, CHANNELS, 4, 4),
            "device": torch.device("cpu"), **options}


def _count_forwards(patcher) -> list[int]:
    dm = patcher.model.diffusion_model
    stock = dm.forward
    calls: list[int] = []

    def counted(*args, **kwargs):
        calls.append(1)
        return stock(*args, **kwargs)

    dm.forward = counted
    return calls


def test_only_no_process_group_or_a_group_of_one_counts_as_one_gpu(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    assert worker_on_one_gpu()
    _process_group(monkeypatch, 1)
    assert worker_on_one_gpu()
    _process_group(monkeypatch, 2)
    assert not worker_on_one_gpu()


@pytest.mark.parametrize("method", list(METHODS), ids=[str(method) for method in METHODS])
@pytest.mark.parametrize("name", sorted(_UNSHARDED_CLUSTER))
def test_a_cluster_worker_refuses_references_in_extra_conds_before_any_reference_block(
        monkeypatch, moss, tmp_path, name, method):
    """A world-2 Worker loaded on cfg2 or dp2, where no sharded forward binds,
    samples an edit the driver let through: #130's class P text, raised while
    ComfyUI builds the conditioning, with no preprocess_refs call and no model
    call, whatever the method."""
    import comfy.sample

    patcher, _path = _base_model(moss, tmp_path)
    _inject(monkeypatch, patcher, _UNSHARDED_CLUSTER[name], world=2)
    assert "extra_conds" in vars(patcher.model)
    refs, forwards = _count_preprocess_refs(patcher), _count_forwards(patcher)
    _process_group(monkeypatch, 2)
    positive, negative = _conds(method)
    noise, latent = torch.randn(1, CHANNELS, 4, 4), torch.zeros(1, CHANNELS, 4, 4)
    with ThreadPoolExecutor(1) as pool, pytest.raises(UnsupportedModelError) as caught:
        pool.submit(comfy.sample.sample, patcher, noise, STEPS, CFG, "euler", "simple",
                    positive, negative, latent, seed=SEED).result()
    assert str(caught.value) == REFERENCE_IMAGES_REFUSAL
    assert parse_leading_refusal_tag(str(caught.value)).refusal_class is RefusalClass.PHYSICS
    assert "TROUBLESHOOTING.md #130" in str(caught.value)
    assert refs == [] and forwards == []


@pytest.mark.parametrize("world", [2, 4])
def test_the_bound_call_refuses_before_stock_on_any_group_above_one(monkeypatch, moss, tmp_path,
                                                                     world):
    """uly2, ring2 and FSDP take the same bind: the refusal precedes stock's
    method, so the sharded forward's own guard is no longer the first."""
    patcher, _path = _base_model(moss, tmp_path)
    MOSSImageAdapter().prepare_base_model(patcher.model)
    refs = _count_preprocess_refs(patcher)
    _process_group(monkeypatch, world)
    with ThreadPoolExecutor(1) as pool, pytest.raises(UnsupportedModelError) as caught:
        pool.submit(lambda: patcher.model.extra_conds(**_extra_conds_kwargs())).result()
    assert str(caught.value) == REFERENCE_IMAGES_REFUSAL
    assert refs == []


@pytest.mark.parametrize("case", ["no-reference", "empty-list", "method-alone"])
def test_a_cluster_render_without_references_passes_the_bind(monkeypatch, moss, tmp_path, case):
    """Text to image on a cluster: stock's conds, the text tensor as is. An
    empty list and a method with no latents carry nothing, as stock reads them."""
    kwargs = _extra_conds_kwargs(refs=False)
    if case == "empty-list":
        kwargs["reference_latents"] = []
    elif case == "method-alone":
        kwargs["reference_latents_method"] = "index_timestep_zero_isolated"
    patcher, _path = _base_model(moss, tmp_path)
    MOSSImageAdapter().prepare_base_model(patcher.model)
    refs = _count_preprocess_refs(patcher)
    _process_group(monkeypatch, 2)
    with ThreadPoolExecutor(1) as pool:
        out = pool.submit(lambda: patcher.model.extra_conds(**kwargs)).result()
    assert "c_crossattn" in out and out["c_crossattn"].cond is kwargs["cross_attn"]
    assert "ref_kv" not in out and "ref_latents" not in out
    assert refs == []


def test_a_group_of_one_keeps_stocks_cache_to_stocks_latent(monkeypatch, moss, tmp_path):
    """Topology single and every Fleet job open a group of one: the Worker's
    path calls preprocess_refs once per conditioning, two a render, in
    inference mode, and lands on stock's latent."""
    positive, negative = _conds()
    stock, _path = _base_model(moss, tmp_path)
    expected = _stock_sample(stock, positive, negative)
    bound, _path = _base_model(moss, tmp_path)
    _inject(monkeypatch, bound)
    refs = _count_preprocess_refs(bound)
    _process_group(monkeypatch, 1)
    samples = _worker_sample(monkeypatch, bound, positive, negative)
    assert refs == [True, True]
    assert torch.equal(samples, expected)


# --- The text attention bias: the bind hands it to stock untouched (#135) ------------


def _biased_conds():
    """The no-VAE edit: a bias on the positive's vision tokens, none on the negative."""
    positive, negative = _conds(refs=False)
    bias = torch.zeros(1, POSITIVE_ROWS)
    bias[:, 2:6] = math.log(2.0)
    positive[0][1]["context_attention_bias"] = bias
    return positive, negative


def _bias_tree(moss) -> None:
    if "context_attention_bias" not in inspect.signature(moss.model.MOSSImage._forward).parameters:
        pytest.skip("this ComfyUI predates the MOSS-Image text attention bias (9fe7ef764)")


@pytest.mark.parametrize("name", sorted(_UNSHARDED_CLUSTER))
def test_a_cluster_worker_hands_an_attention_bias_to_stock_untouched(monkeypatch, moss, tmp_path,
                                                                    name):
    """Off one GPU the bind refuses no bias: stock's CONDRegular, the tensor as is."""
    import comfy.conds

    _bias_tree(moss)
    positive, _negative = _biased_conds()
    context, options = positive[0]
    patcher, _path = _base_model(moss, tmp_path)
    _inject(monkeypatch, patcher, _UNSHARDED_CLUSTER[name], world=2)
    assert "_forward" not in vars(patcher.model.diffusion_model)
    _process_group(monkeypatch, 2)
    kwargs = {"cross_attn": context, "noise": torch.zeros(1, CHANNELS, 4, 4),
              "device": torch.device("cpu"), **options}
    with ThreadPoolExecutor(1) as pool:
        out = pool.submit(lambda: patcher.model.extra_conds(**kwargs)).result()
    assert type(out["context_attention_bias"]) is comfy.conds.CONDRegular
    assert out["context_attention_bias"].cond is options["context_attention_bias"]
    assert "ref_kv" not in out and "ref_latents" not in out


def test_a_dp2_worker_samples_an_attention_bias_to_stocks_latent(monkeypatch, moss, tmp_path):
    """dp2 binds no sharded forward, so stock's applies the bias on a Worker
    thread outside inference mode, to the latent stock samples, and the bias
    moves that latent."""
    import comfy.sample

    _bias_tree(moss)
    positive, negative = _biased_conds()
    stock, _path = _base_model(moss, tmp_path)
    expected = _stock_sample(stock, positive, negative)
    assert not torch.equal(_stock_sample(stock, *_conds(refs=False)), expected)
    bound, _path = _base_model(moss, tmp_path)
    _inject(monkeypatch, bound, _UNSHARDED_CLUSTER["dp2"], world=2)
    assert "extra_conds" in vars(bound.model) and "_forward" not in vars(bound.model.diffusion_model)
    _process_group(monkeypatch, 2)

    def sample():
        latent = comfy.sample.fix_empty_latent_channels(bound, torch.zeros(1, CHANNELS, 4, 4))
        noise = comfy.sample.prepare_noise(latent, SEED, None)
        return comfy.sample.sample(bound, noise, STEPS, CFG, "euler", "simple", positive, negative,
                                   latent, denoise=1.0, disable_pbar=True, seed=SEED)

    with ThreadPoolExecutor(1) as pool:
        assert torch.equal(pool.submit(sample).result(), expected)
