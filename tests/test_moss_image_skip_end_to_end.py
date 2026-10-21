"""Skip Layer Guidance on MOSS-Image under uly2, from the node to the sharded forward.

This is the end-to-end test that moss_image's ``sp_honours_block_replace = True``
rests on (tests/test_skip_layer_guidance.py pins that no other family declares
it). Each step is the shipped code or ComfyUI's own:
the DGX Monarch node freezes the widgets into the ModelSpec, request_dict
carries them through a JSON round trip, each rank's Worker belt
(actor/guidance_render.py) installs ComfyUI's SkipLayerGuidanceDiTSimple on its
own render clone, and the function ComfyUI installed then runs one in-window
step with ComfyUI's real calc_cond_batch. That step makes two model calls, the
conditional pass with every block and the unconditional pass with block 1
skipped, and each call is the adapter's sharded forward on two rank threads.

The model is a tiny real model_base.MOSSImage that ComfyUI loads from a
vendor-layout file with fused qkv, in float32, so "equal" means what it means in
tests/test_moss_image_comfy_exactness.py, whose harness this file reuses. The
cond and uncond have different lengths (23 and 10 text rows on a 3 by 3
latent), so on two ranks one call holds a pad row and the other does not.
"""
from __future__ import annotations

import json
import sys
import threading
import uuid
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.actor import guidance_render
from dgx_monarch.actor.sampling import model_sampling_render_clone
from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.moss_image import MOSSImageAdapter
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.nodes.skip_layer_guidance import DGXMonarchSkipLayerGuidance
from dgx_monarch.refusal import RefusalClass
from dgx_monarch.skip_layer_guidance import skip_layer_guidance_refusal_reason
from test_flux3_comfy_exactness import _same, _StockKernelHarness
from test_flux_pad_exclusion import _LOCAL, _install_stubs
from test_moss_image_comfy_exactness import (  # noqa: F401  (fixture re-exported)
    CHANNELS,
    LAYERS,
    VENDOR_TEXT_DIM,
    WORLD,
    _tag,
    _vendor_file,
    comfy_moss,
)

SKIPPED = 1
GRID = (3, 3)
COND_ROWS, UNCOND_ROWS = 23, 10
WIDGETS = {"double_layers": "", "single_layers": str(SKIPPED), "start_percent": 0.0, "end_percent": 1.0}
ULY2 = SimpleNamespace(topology={"ulysses": 2, "ring": 1, "cfg": 1, "dp": 1, "fsdp": False},
                       store=SimpleNamespace(family_override=None))
CFG2 = SimpleNamespace(topology={"ulysses": 1, "ring": 1, "cfg": 2, "dp": 1, "fsdp": False},
                       store=SimpleNamespace(family_override=None))

# Which calc_cond_batch call the calling thread is inside, and which rank it is
# (unset on the main thread, the stock run). This file's own thread-local: the
# harness's _LOCAL can carry a rank on the main thread from an earlier test.
_PHASE = threading.local()


@pytest.fixture
def moss(request):
    """The exactness file's module-scoped ComfyUI import, under a local name."""
    return request.getfixturevalue("comfy_moss")


def _gates() -> None:
    pytest.importorskip("comfy.model_patcher")
    pytest.importorskip("comfy.samplers")


def _request() -> dict:
    """The wire a driver sends: the node, ModelSpec.request_dict, a JSON round trip."""
    mesh = MeshSpec(handle=SimpleNamespace(world=WORLD), topology_preset="uly2",
                    attention="TORCH_FLASH", sync_ulysses=True)
    patched, = DGXMonarchSkipLayerGuidance().patch(
        ModelSpec(mesh=mesh, unet_name="moss_tiny.safetensors"), **WIDGETS)
    request = json.loads(json.dumps({"model": patched.request_dict()}))
    assert request["model"]["skip_layer_guidance"] == WIDGETS
    return request


def _loader(comfy_ns, tmp_path):
    """Load the one vendor-layout file through ComfyUI as often as a test needs a model."""
    import comfy.model_base
    import comfy.sd

    path = _vendor_file(comfy_ns, tmp_path, dtype=torch.float32)

    def load():
        patcher = comfy.sd.load_diffusion_model(path, model_options={"dtype": torch.float32})
        assert type(patcher.model) is comfy.model_base.MOSSImage
        return patcher

    return load


def _sampler_conds(base_model, rows: int, seed: int) -> list[dict]:
    """One sampler-side cond entry, the two keys calc_cond_batch reads."""
    generator = torch.Generator().manual_seed(seed)
    model_conds = base_model.extra_conds(
        cross_attn=torch.randn(1, rows, VENDOR_TEXT_DIM, generator=generator),
        noise=torch.zeros(1, CHANNELS, *GRID), device=torch.device("cpu"))
    return [{"model_conds": model_conds, "uuid": uuid.UUID(int=seed)}]


def _spy_blocks(diffusion_model) -> dict[int, list]:
    """Record, per block, which calc_cond_batch call ran its forward."""
    runs: dict[int, list] = {index: [] for index in range(len(diffusion_model.layers))}
    for index, block in enumerate(diffusion_model.layers):
        block.register_forward_hook(
            lambda _module, _args, _out, index=index: runs[index].append(getattr(_PHASE, "call", None)))
    return runs


def _spy_calc_cond_batch(monkeypatch) -> list:
    """Wrap ComfyUI's calc_cond_batch, which SLG reads by module attribute, to label each call."""
    import comfy.samplers

    real = comfy.samplers.calc_cond_batch
    seen: list = []

    def labelled(model, conds, x_in, timestep, model_options):
        _PHASE.call = "cond" if conds[0] is not None else "uncond"
        seen.append((getattr(_PHASE, "who", "stock"), [c is not None for c in conds]))
        try:
            return real(model, conds, x_in, timestep, model_options)
        finally:
            _PHASE.call = None

    monkeypatch.setattr(comfy.samplers, "calc_cond_batch", labelled)
    return seen


def _render_clone(patcher, request, worker, adapter):
    """A Worker's render clone through run_sample's two steps: the shift clone, then the skip."""
    clone = model_sampling_render_clone(patcher, None)
    patched = guidance_render.apply_skip_layer_guidance(clone, request, worker, adapter)
    assert patched is not clone
    assert "sampler_calc_cond_batch_function" in patched.model_options
    assert "sampler_calc_cond_batch_function" not in clone.model_options
    patched.pre_run()   # the sampler's own step: calc_cond_batch reads model.current_patcher
    return patched


def _step(patched, cond, uncond, x):
    """One in-window step of the function ComfyUI's node installed."""
    function = patched.model_options["sampler_calc_cond_batch_function"]
    with torch.no_grad():
        return function({"conds": [cond, uncond], "input": x, "sigma": torch.tensor([0.5]),
                         "model": patched.model, "model_options": patched.model_options})


def test_the_carried_skip_runs_the_sharded_forward_to_stocks_bits(monkeypatch, moss, tmp_path):
    _gates()
    import comfy.ldm.moss_image.model as moss_model
    from comfy_extras import nodes_slg

    load = _loader(moss, tmp_path)
    request = _request()
    seen = _spy_calc_cond_batch(monkeypatch)

    # The stock reference: ComfyUI's own node on an unsharded clone, one thread.
    stock = nodes_slg.SkipLayerGuidanceDiTSimple.execute(
        model_sampling_render_clone(load(), None), WIDGETS["start_percent"], WIDGETS["end_percent"],
        double_layers=WIDGETS["double_layers"], single_layers=WIDGETS["single_layers"])[0]
    stock.pre_run()
    cond = _sampler_conds(stock.model, COND_ROWS, seed=11)
    uncond = _sampler_conds(stock.model, UNCOND_ROWS, seed=12)
    x = torch.randn(1, CHANNELS, *GRID, generator=torch.Generator().manual_seed(13))
    stock_runs = _spy_blocks(stock.model.diffusion_model)
    expected = _step(stock, cond, uncond, x)
    assert seen == [("stock", [True, False]), ("stock", [False, True])]

    # Every rank: its own Worker belt on its own render clone, then the sharded forward.
    ranks = [_render_clone(load(), request, ULY2, MOSSImageAdapter()) for _rank in range(WORLD)]
    harness = _StockKernelHarness(WORLD, moss_model.optimized_attention)
    _install_stubs(monkeypatch, harness)
    monkeypatch.setattr(sys.modules["xfuser.core.distributed"], "get_ring_parallel_world_size", lambda: 1)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    for patched in ranks:
        MOSSImageAdapter().inject_usp(patched.model.diffusion_model, InjectionContext(
            topology_sp=WORLD, usp_attention=dispatch, pure_ulysses=True))
    rank_runs = [_spy_blocks(patched.model.diffusion_model) for patched in ranks]
    outputs: list = [None] * WORLD
    errors: list = [None] * WORLD

    def body(rank):
        try:
            _LOCAL.rank = _PHASE.who = rank
            outputs[rank] = _step(ranks[rank], cond, uncond, x)
        except BaseException as exc:  # re-raised in the test body
            errors[rank] = exc
            harness.fabric.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(WORLD)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    for error in errors:
        if error is not None:
            raise error

    # ComfyUI's function made its two calls on every rank, the cond call first.
    for rank in range(WORLD):
        assert [pattern for who, pattern in seen if who == rank] == [[True, False], [False, True]]
    # Block 1 ran on the cond call only, in stock and on every rank; the others ran on both.
    for runs in (stock_runs, *rank_runs):
        assert runs[SKIPPED] == ["cond"]
        assert all(runs[index] == ["cond", "uncond"] for index in range(LAYERS) if index != SKIPPED)
    # The sharded override attended every block that ran, and no other.
    for rank in range(WORLD):
        assert len(harness.seen[rank]) == 2 * LAYERS - 1
    # Stock's bits, and every rank alike.
    assert len(expected) == len(outputs[0]) == 2
    for rank in range(1, WORLD):
        assert all(torch.equal(a, b) for a, b in zip(outputs[0], outputs[rank], strict=True))
    for got, want in zip(outputs[0], expected, strict=True):
        assert _same(got, want)


def test_the_workers_ksampler_loop_reaches_the_carried_skip_at_cfg_3(monkeypatch, moss, tmp_path):
    """run_ksampler, one Euler step at CFG 3 on a Worker's render clone, on one rank.

    The Worker's KSampler goes through comfy.sample.sample and ComfyUI's CFGGuider,
    whose sampling_function calls the function the skip installed. So block 1 runs
    once, on the conditional pass, and every other block runs twice. Were the
    hook bypassed, one batched call would run every block once.
    """
    _gates()
    from dgx_monarch.actor import sampling

    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)
    seen = _spy_calc_cond_batch(monkeypatch)
    clone = model_sampling_render_clone(_loader(moss, tmp_path)(), None)
    patched = guidance_render.apply_skip_layer_guidance(clone, _request(), ULY2, MOSSImageAdapter())
    runs = _spy_blocks(patched.model.diffusion_model)
    generator = torch.Generator().manual_seed(21)
    request = {
        "latent": {"samples": torch.zeros(1, CHANNELS, *GRID)},
        "positive": [[torch.randn(1, COND_ROWS, VENDOR_TEXT_DIM, generator=generator), {}]],
        "negative": [[torch.randn(1, UNCOND_ROWS, VENDOR_TEXT_DIM, generator=generator), {}]],
        "sampler_name": "euler", "scheduler": "simple", "steps": 1, "cfg": 3.0, "noise_seed": 7,
    }
    with torch.no_grad():
        samples, out = sampling.run_ksampler(patched, request)
    assert out is not None and torch.isfinite(samples).all()
    assert [pattern for _who, pattern in seen] == [[True, False], [False, True]]
    assert runs[SKIPPED] == ["cond"]
    assert all(runs[index] == ["cond", "uncond"] for index in range(LAYERS) if index != SKIPPED)


def test_run_samples_call_shape_resolves_the_moss_adapter_and_admits_uly2(moss, tmp_path):
    """run_sample hands the belt no adapter; it resolves the real model's own and admits the skip."""
    _gates()
    patched = _render_clone(_loader(moss, tmp_path)(), _request(), ULY2, None)
    assert type(patched.model).__name__ == "MOSSImage"


def test_the_same_request_on_cfg2_refuses_class_p_before_any_model_call(monkeypatch, moss, tmp_path):
    _gates()
    seen = _spy_calc_cond_batch(monkeypatch)
    patcher = _loader(moss, tmp_path)()
    runs = _spy_blocks(patcher.model.diffusion_model)
    clone = model_sampling_render_clone(patcher, None)
    with pytest.raises(UnsupportedModelError) as caught:
        guidance_render.apply_skip_layer_guidance(clone, _request(), CFG2, MOSSImageAdapter())
    text = str(caught.value)
    assert _tag(caught.value).refusal_class is RefusalClass.PHYSICS
    reason = skip_layer_guidance_refusal_reason(cfg=2, sp=1, family="moss_image",
                                                family_honours=True, dual_model=False)
    assert reason is not None and reason in text
    assert "docs/TROUBLESHOOTING.md #126" in text
    assert "sampler_calc_cond_batch_function" not in clone.model_options
    assert seen == []
    assert all(not calls for calls in runs.values())
