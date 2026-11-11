"""Anima's text branch: the Worker's bind, stock's model calls, and the persistence canary.

Stock ``Anima.extra_conds`` (comfy/model_base.py) runs the text through
``preprocess_text_embeds``, the ``llm_adapter`` and then a pad to 512 rows,
once per conditioning, but only under ``torch.inference_mode()``. Otherwise
it hands the forward the raw Qwen3 rows with ``t5xxl_ids`` and
``t5xxl_weights``, and the forward reruns the adapter and the pad on every
model call. ComfyUI enters inference mode around prompt execution and a
Worker's sampling thread never does, so before the adapter's
``prepare_base_model`` a Worker took the second branch. Rows of different
length do not concatenate, so a prompt pair whose raw rows differ ran as two
batch-1 calls a step where stock, at 512 rows on both sides, folds it into
one batch-2 call; on 2026-11-10 the shipped graph's one-GPU Worker render
read one-step NRMS 0.028 against stock on hardware. The bind is FLUX.3's
(tests/test_flux3_cond_branch.py).

Every test runs real ComfyUI on the CPU, on a tiny Anima that ComfyUI's own
loader detects. CPU kernels give the same bits at batch 1 and batch 2, so
here the latent cannot tell the branches apart; the tests compare what
reaches the model. The Worker's load hook binds on every topology. The bound
conds equal stock's under inference mode and hold no inference tensor. The
Worker's KSampler path, on a thread outside inference mode, makes stock's
model calls input for input and runs the adapter once per conditioning;
without the bind it makes two batch-1 calls a step on a pair whose rows
differ, runs the adapter on every call, and cfg's fold test answers no where
one GPU folds. The bind is idempotent and survives patcher and LoRA clones,
and the persistence canary takes it through the store's whole LoRA life
cycle with LoRA units on the adapter, which now runs in the inference-mode
window. Under a real one-rank FSDP2 wrap the adapter runs before the first
root forward, in inference mode, on parameters the wrap left whole. The
comfy-canary job runs this file against ComfyUI master and the newest release.
"""
from __future__ import annotations

import inspect
import os
import platform
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters.anima import AnimaAdapter
from module_location_helpers import from_checkout

COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
STEPS, CFG, SEED = 3, 4.0, 42
# Equal bits where the CPU GEMMs agree per row whatever the call's shape or the
# operands' memory (aarch64), else float32 rounding: the x86 canary runner's
# GEMMs can round a row apart at another batch or on a copied weight.
_EXACT_GEMM = platform.machine() == "aarch64"


def _same(candidate: torch.Tensor, reference: torch.Tensor) -> bool:
    if candidate.shape != reference.shape:
        return False
    if _EXACT_GEMM:
        return torch.equal(candidate, reference)
    return torch.equal(candidate, reference) or torch.allclose(
        candidate, reference, rtol=2e-5, atol=2e-5)
# Raw rows of different length on both streams, as on the shipped graph, so
# off the bind cond and uncond cannot share a model call.
PAIRS = {"differing": ((14, 12), (9, 7)), "equal": ((14, 12), None)}
_SINGLE = {"cfg": 1, "ulysses": 1, "ring": 1, "dp": 1, "fsdp": False}
_TOPOLOGIES = {"single": (_SINGLE, 1), "dp2": ({**_SINGLE, "dp": 2}, 2),
               "cfg2": ({**_SINGLE, "cfg": 2}, 2)}


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy():
    """Import real ComfyUI on the CPU, build one tiny Anima, then leave sys.modules as found."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before = list(sys.path)
    argv_before = sys.argv
    if os.path.isdir(COMFY_DIR) and COMFY_DIR not in sys.path:
        sys.path.insert(0, COMFY_DIR)
    try:
        sys.argv = ["pytest-anima-cond-branch", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            anima = pytest.importorskip("comfy.ldm.anima.model", reason="this ComfyUI predates Anima")
            import comfy.lora
            import comfy.model_base
            import comfy.ops
            import comfy.patcher_extension
            import comfy.sample
            import comfy.sd
        finally:
            sys.argv = argv_before
        yield SimpleNamespace(lora=comfy.lora, model_base=comfy.model_base, ops=comfy.ops,
                              patcher_extension=comfy.patcher_extension, sample=comfy.sample,
                              sd=comfy.sd, state=_tiny_state(comfy.ops, anima.Anima))
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, COMFY_DIR)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


@pytest.fixture(autouse=True)
def _drop_loaded_models(comfy):
    """Each test loads its own models of about 0.55 GB, and ComfyUI's model
    management keeps every sampled one listed; drop them after each test so
    the file's peak stays near one test's need."""
    yield
    import gc

    sys.modules["comfy.model_management"].unload_all_models()
    gc.collect()


def _tiny_state(ops, anima_class) -> dict:
    """A real cosmos-DiT Anima at 256 channels and two blocks. Its llm_adapter
    keeps the fixed 1024-wide, six-layer shape stock builds for every file."""
    model = anima_class(
        max_img_h=240, max_img_w=240, max_frames=128, in_channels=16, out_channels=16,
        patch_spatial=2, patch_temporal=1, concat_padding_mask=True, model_channels=256,
        num_blocks=2, num_heads=16, crossattn_emb_channels=1024, pos_emb_cls="rope3d",
        pos_emb_learnable=True, pos_emb_interpolation="crop", min_fps=1, max_fps=30,
        use_adaln_lora=True, adaln_lora_dim=256, rope_h_extrapolation_ratio=4.0,
        rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
        extra_per_block_abs_pos_emb=False, dtype=torch.float32, device="cpu",
        operations=ops.disable_weight_init)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return {key: value.contiguous() for key, value in model.state_dict().items()}


def _load(comfy, path=None):
    """ComfyUI's own diffusion-model load, from the shared state or a file."""
    patcher = (comfy.sd.load_diffusion_model(str(path)) if path is not None
               else comfy.sd.load_diffusion_model_state_dict(dict(comfy.state)))
    assert type(patcher.model) is comfy.model_base.Anima
    return patcher


def _cond(generator, rows: int, t5: int) -> list:
    """One conditioning as ComfyUI's Anima text encoder hands it on: Qwen3 rows
    and, beside them, the T5 ids and their weights."""
    return [[torch.randn(1, rows, 1024, generator=generator) * 0.5,
             {"t5xxl_ids": torch.randint(0, 32128, (t5,), generator=generator, dtype=torch.int32),
              "t5xxl_weights": torch.rand(t5, generator=generator) + 0.5}]]


def _pair(name: str = "differing"):
    generator = torch.Generator().manual_seed(3)
    first, second = PAIRS[name]
    positive = _cond(generator, *first)
    if second is None:
        tensor, options = positive[0]
        return positive, [[tensor.clone(), {key: value.clone() for key, value in options.items()}]]
    return positive, _cond(generator, *second)


def _extra_conds_kwargs(conditioning) -> dict:
    tensor, options = conditioning[0]
    return {"cross_attn": tensor, "device": torch.device("cpu"), **options}


def _off_mode(fn):
    """Run ``fn`` on a fresh thread, where inference mode is off, as on a Worker."""
    with ThreadPoolExecutor(1) as pool:
        assert not pool.submit(torch.is_inference_mode_enabled).result()
        return pool.submit(fn).result()


def _inject(monkeypatch, patcher, topology=_SINGLE, world=1) -> None:
    """The Worker's load hook; the registry picks the adapter."""
    from dgx_monarch.actor import partial_load_guard, store_fsdp
    from dgx_monarch.actor import worker as worker_mod

    monkeypatch.setattr(store_fsdp, "validate_injection", lambda *a, **k: None)
    monkeypatch.setattr(partial_load_guard, "install", lambda *a, **k: None)
    monkeypatch.setattr(worker_mod, "_maybe_compile_dit", lambda *a, **k: None)
    attn = SimpleNamespace(bind_capability=lambda *a: None, effective_kernel="TORCH_FLASH")
    worker = SimpleNamespace(topology=dict(topology), world=world, _attn=attn,
                             store=SimpleNamespace(family_override=None))
    worker_mod.GPUWorker._inject_for_topology(worker, patcher, "bf16", [], None)


class _Probe:
    """Every model call's inputs by value, each llm_adapter call's mode, and
    cfg's fold test on every cond batch the sampler builds."""

    def __init__(self, comfy, patcher):
        from dgx_monarch.adapters.cfg_dispatch import conds_fold

        self.calls: list[dict] = []
        self.adapter: list[bool] = []
        self.folds: list[bool] = []
        dm = patcher.model.diffusion_model

        def before_forward(_module, args, kwargs):
            self.calls.append({"x": args[0].clone(), "t": args[1].clone(),
                               "context": kwargs["context"].clone(),
                               "t5xxl_ids": "t5xxl_ids" in kwargs})

        def before_adapter(_module, _args):
            self.adapter.append(torch.is_inference_mode_enabled())

        def cond_batch(executor, *args, **kwargs):
            bound = inspect.signature(executor.original).bind_partial(*args, **kwargs)
            self.folds.append(conds_fold(bound.arguments["conds"], bound.arguments["x_in"]))
            return executor(*args, **kwargs)

        self.handles = (dm.register_forward_pre_hook(before_forward, with_kwargs=True),
                        dm.llm_adapter.register_forward_pre_hook(before_adapter))
        pe = comfy.patcher_extension
        pe.add_wrapper_with_key(pe.WrappersMP.CALC_COND_BATCH, "test-anima-probe", cond_batch,
                                patcher.model_options, is_model_options=True)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def batches(self) -> list[int]:
        return [int(call["x"].shape[0]) for call in self.calls]

    def same_calls_as(self, other: _Probe, equal=torch.equal) -> bool:
        """Every call's inputs against ``other``'s. Calls of the same shapes on
        the same weights compare by bits; ``equal=_same`` where the weights
        were copied (an FSDP wrap)."""
        return len(self.calls) == len(other.calls) and all(
            mine[key].shape == theirs[key].shape and equal(mine[key], theirs[key])
            for mine, theirs in zip(self.calls, other.calls, strict=True)
            for key in ("x", "t", "context"))


def _latent():
    return torch.zeros(1, 4, 8, 8)


def _stock_sample(comfy, patcher, positive, negative):
    """Stock's KSampler (nodes.common_ksampler's calls) inside inference mode, as a prompt runs."""
    with torch.inference_mode():
        latent = comfy.sample.fix_empty_latent_channels(patcher, _latent())
        noise = comfy.sample.prepare_noise(latent, SEED, None)
        out = comfy.sample.sample(patcher, noise, STEPS, CFG, "euler", "simple", positive, negative,
                                  latent, denoise=1.0, disable_pbar=True, seed=SEED)
    return out.clone()


def _worker_sample(monkeypatch, patcher, positive, negative):
    """run_ksampler on a Worker render clone, on a fresh thread: inference mode off."""
    from dgx_monarch.actor import sampling

    monkeypatch.setattr(sampling, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(sampling, "is_result_leader", lambda: True)
    request = {"latent": {"samples": _latent()}, "positive": positive, "negative": negative,
               "sampler_name": "euler", "scheduler": "simple", "steps": STEPS, "cfg": CFG,
               "noise_seed": SEED}
    clone = sampling.model_sampling_render_clone(patcher, None)
    samples, _out = _off_mode(lambda: sampling.run_ksampler(clone, request))
    return samples


def _walk(value):
    if isinstance(value, list):
        for item in value:
            yield from _walk(item)
    elif torch.is_tensor(value):
        yield value


# --- Stock's branch and the Worker's hook ------------------------------------------


def test_inference_mode_alone_picks_stocks_text_branch(comfy):
    """The premise, on the installed ComfyUI: in inference mode the text leaves
    extra_conds at 512 rows with nothing else; off it, the raw rows and the T5
    inputs the forward reads."""
    model = _load(comfy).model
    positive, _negative = _pair()
    kwargs = _extra_conds_kwargs(positive)
    with torch.inference_mode():
        stock = model.extra_conds(**kwargs)
    plain = _off_mode(lambda: model.extra_conds(**kwargs))
    assert set(stock) == {"c_crossattn"}
    assert tuple(stock["c_crossattn"].cond.shape) == (1, 512, 1024)
    assert set(plain) == {"c_crossattn", "t5xxl_ids", "t5xxl_weights"}
    assert plain["c_crossattn"].cond is kwargs["cross_attn"]
    assert tuple(plain["t5xxl_ids"].cond.shape) == (1, 12)


@pytest.mark.parametrize("name", sorted(_TOPOLOGIES))
def test_the_workers_load_hook_binds_extra_conds_on_every_topology(monkeypatch, comfy, name):
    topology, world = _TOPOLOGIES[name]
    patcher = _load(comfy)
    assert "extra_conds" not in vars(patcher.model)
    _inject(monkeypatch, patcher, topology, world)
    bound = vars(patcher.model)["extra_conds"]
    cells = [cell.cell_contents for cell in bound.__func__.__closure__ or ()]
    assert comfy.model_base.Anima.extra_conds in cells
    # No sequence parallelism here, so the forward stays stock's.
    assert "_forward" not in vars(patcher.model.diffusion_model)
    positive, _negative = _pair()
    out = _off_mode(lambda: patcher.model.extra_conds(**_extra_conds_kwargs(positive)))
    assert set(out) == {"c_crossattn"}


CONDS_CASES = {
    "prompt": {"t5": 12},
    # Past 512 T5 ids stock pads nothing, so the rows keep the ids' count.
    "long": {"t5": 520},
    # No T5 ids: stock passes the text tensor through in either mode.
    "no-ids": {"t5": None},
}


@pytest.mark.parametrize("case", sorted(CONDS_CASES))
def test_the_bound_conds_equal_stocks_in_inference_mode_and_hold_no_inference_tensor(comfy, case):
    """Off inference mode the bound call returns what stock returns inside it,
    every tensor an exact copy that is not an inference tensor."""
    t5 = CONDS_CASES[case]["t5"]
    generator = torch.Generator().manual_seed(11)
    if t5 is None:
        kwargs = {"cross_attn": torch.randn(1, 14, 1024, generator=generator),
                  "device": torch.device("cpu")}
    else:
        kwargs = _extra_conds_kwargs(_cond(generator, 14, t5))
    model = _load(comfy).model
    with torch.inference_mode():
        stock = model.extra_conds(**kwargs)
    AnimaAdapter().prepare_base_model(model)
    bound = _off_mode(lambda: model.extra_conds(**kwargs))
    assert set(bound) == set(stock) == {"c_crossattn"}
    rows = 14 if t5 is None else max(t5, 512)
    for key in bound:
        assert type(bound[key]) is type(stock[key]), key
        got, want = list(_walk(bound[key].cond)), list(_walk(stock[key].cond))
        assert len(got) == len(want) and all(torch.equal(a, b) for a, b in zip(got, want, strict=True))
        assert not any(tensor.is_inference() for tensor in got), key
    assert bound["c_crossattn"].cond.shape[1] == rows
    if t5 is None:
        assert bound["c_crossattn"].cond is kwargs["cross_attn"]


# --- The Worker's sampling path against stock's -------------------------------------


@pytest.mark.parametrize("pair", sorted(PAIRS))
def test_the_workers_ksampler_makes_stocks_model_calls_after_the_load_hook(monkeypatch, comfy, pair):
    """The Worker's path after its load hook: stock's model calls, every
    input bit for bit, the adapter twice a render in inference mode, and
    stock's latent. Without the hook a pair whose rows differ runs as two
    batch-1 calls a step and cfg's fold test answers no; an equal pair folds
    there too, but the adapter runs at batch 2 on every call. The latent is
    stock's either way on the CPU, so the calls are the check."""
    positive, negative = _pair(pair)
    stock = _load(comfy)
    stock_probe = _Probe(comfy, stock)
    expected = _stock_sample(comfy, stock, positive, negative)
    stock_probe.close()
    assert stock_probe.batches() == [2] * STEPS
    assert [call["context"].shape[1] for call in stock_probe.calls] == [512] * STEPS
    assert stock_probe.adapter == [True, True] and stock_probe.folds == [True] * STEPS

    bound = _load(comfy)
    _inject(monkeypatch, bound)
    bound_probe = _Probe(comfy, bound)
    samples = _worker_sample(monkeypatch, bound, positive, negative)
    bound_probe.close()
    assert bound_probe.same_calls_as(stock_probe)
    assert not any(call["t5xxl_ids"] for call in bound_probe.calls)
    assert bound_probe.adapter == [True, True] and bound_probe.folds == [True] * STEPS
    assert torch.equal(samples, expected)

    unbound = _load(comfy)
    unbound_probe = _Probe(comfy, unbound)
    before = _worker_sample(monkeypatch, unbound, positive, negative)
    unbound_probe.close()
    assert not unbound_probe.same_calls_as(stock_probe)
    assert all(call["t5xxl_ids"] for call in unbound_probe.calls)
    if pair == "differing":
        assert unbound_probe.batches() == [1] * (2 * STEPS)
        assert unbound_probe.adapter == [False] * (2 * STEPS)
        assert unbound_probe.folds == [False] * STEPS
    else:
        assert unbound_probe.batches() == [2] * STEPS
        assert unbound_probe.adapter == [False] * STEPS
        assert unbound_probe.folds == [True] * STEPS
    # Batch-1 calls against stock's batch-2 ones: bits on aarch64 only.
    assert _same(before, expected)


def test_the_bind_is_idempotent_and_survives_a_patcher_clone_and_a_lora_clone(comfy):
    patcher = _load(comfy)
    for _ in range(3):
        AnimaAdapter().prepare_base_model(patcher.model)
    cells = [cell.cell_contents for cell in patcher.model.extra_conds.__func__.__closure__ or ()]
    assert comfy.model_base.Anima.extra_conds in cells
    assert not any(isinstance(cell, types.MethodType) for cell in cells)
    positive, _negative = _pair()
    kwargs = _extra_conds_kwargs(positive)
    clone = patcher.clone()
    lora_clone, _clip = comfy.sd.load_lora_for_models(patcher, None, {}, 1.0, 0)
    for other in (patcher, clone, lora_clone):
        assert other.model is patcher.model
        assert set(_off_mode(lambda m=other.model: m.extra_conds(**kwargs))) == {"c_crossattn"}


# --- Under a real FSDP2 wrap ---------------------------------------------------------


def test_the_bound_adapter_runs_before_the_first_fsdp_forward_on_plain_parameters(
        monkeypatch, comfy, tmp_path):
    """The bind moves the llm_adapter out of the root forward and in front of
    it, where no FSDP pre-forward hook has gathered anything. One gloo rank on
    the CPU, the wrap adapters/fsdp.py makes (blocks, then the root, the
    auxiliary census ignored): the adapter's parameters stay plain tensors,
    it runs twice a render in inference mode before the first root forward,
    and every model call and the latent equal stock's unwrapped run."""
    import torch.distributed.fsdp
    from torch.distributed.device_mesh import init_device_mesh

    from dgx_monarch.adapters.fsdp_islands import auxiliary_module_parameters

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    positive, negative = _pair()
    stock = _load(comfy)
    stock_probe = _Probe(comfy, stock)
    expected = _stock_sample(comfy, stock, positive, negative)
    stock_probe.close()

    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    try:
        patcher = _load(comfy)
        model, dm = patcher.model, patcher.model.diffusion_model
        for parameter in dm.parameters():
            parameter.requires_grad = False
        ignored = auxiliary_module_parameters(dm, model)
        assert set(dm.llm_adapter.parameters()) <= ignored
        mesh = init_device_mesh("cpu", (1,))
        for block in dm.blocks:
            torch.distributed.fsdp.fully_shard(block, mesh=mesh, ignored_params=ignored)
        torch.distributed.fsdp.fully_shard(dm, mesh=mesh, reshard_after_forward=True,
                                           ignored_params=ignored)
        assert any(type(p).__name__ == "DTensor" for p in dm.blocks[0].parameters())
        events: list = []
        dm.register_forward_pre_hook(lambda *_a, **_k: events.append("forward"), with_kwargs=True)
        dm.llm_adapter.register_forward_pre_hook(lambda *_a: events.append((
            "adapter", torch.is_inference_mode_enabled(),
            all(type(p.data) is torch.Tensor for p in dm.llm_adapter.parameters()))))
        _inject(monkeypatch, patcher)
        probe = _Probe(comfy, patcher)
        samples = _worker_sample(monkeypatch, patcher, positive, negative)
        probe.close()
    finally:
        torch.distributed.destroy_process_group()
    first = events.index("forward")
    assert events[:first] == [("adapter", True, True)] * 2
    assert not any(isinstance(event, tuple) for event in events[first:])
    assert probe.same_calls_as(stock_probe, equal=_same)
    assert _same(samples, expected)


# --- The persistence canary ---------------------------------------------------------


def _inference_state(model) -> list[str]:
    return [name for name, value in model.state_dict(keep_vars=True).items()
            if torch.is_tensor(value) and value.is_inference()]


# Two adapter units, which the bind moves into the inference-mode window, and
# two DiT units, which stay in the forward: (unit, out features, in features).
_LORA_UNITS = (("llm_adapter.blocks.0.self_attn.q_proj", 1024, 1024),
               ("llm_adapter.out_proj", 1024, 1024),
               ("blocks.0.self_attn.q_proj", 256, 256),
               ("blocks.1.cross_attn.k_proj", 256, 1024))


def test_the_bound_branch_survives_sample_bake_hot_swap_and_unbake(comfy, tmp_path):
    """Nothing raises, no state is an inference tensor, and the un-bake
    restores the file's own bytes, with LoRA units on the adapter."""
    from safetensors.torch import load_file, save_file

    from dgx_monarch.actor.model_store import ModelStore
    from dgx_monarch.actor.store_bake import _merge_and_free, lazy_swap
    from dgx_monarch.actor.unbake import live_tensor, restore_pristine

    path = tmp_path / "anima_tiny.safetensors"
    save_file(comfy.state, str(path))
    patcher = _load(comfy, path)
    AnimaAdapter().prepare_base_model(patcher.model)
    probe = _Probe(comfy, patcher)
    positive, negative = _pair()
    latent = comfy.sample.fix_empty_latent_channels(patcher, _latent())
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(5))

    def sample(model_patcher):
        return _off_mode(lambda: comfy.sample.sample(
            model_patcher, noise, 2, CFG, "euler", "simple", positive, negative, latent, seed=1))

    first = sample(patcher)
    assert torch.isfinite(first).all()
    assert probe.adapter == [True, True]
    assert _inference_state(patcher.model) == []

    key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
    generator = torch.Generator().manual_seed(9)
    lora = {}
    for unit, out_features, in_features in _LORA_UNITS:
        lora[f"diffusion_model.{unit}.lora_up.weight"] = torch.randn(
            out_features, 4, generator=generator) * 0.1
        lora[f"diffusion_model.{unit}.lora_down.weight"] = torch.randn(
            4, in_features, generator=generator) * 0.1
    patches = comfy.lora.load_lora(lora, key_map)
    assert len(patches) == len(_LORA_UNITS)

    active = patcher.clone()
    active.add_patches(patches, 0.5)
    record = _merge_and_free(active, str(path))
    assert record is not None and len(record.mapped) + len(record.resident) == len(_LORA_UNITS)
    baked = sample(active)
    assert not torch.equal(baked, first)

    store = ModelStore()
    store.swap_verify = -1   # comfy's own bake verifies every key
    swapped = patcher.clone()
    swapped.add_patches(patches, 0.8)
    record = lazy_swap(store, SimpleNamespace(unbake=record, base_patcher=patcher),
                       swapped, path.name)
    assert record is not None
    sample(swapped)
    assert _inference_state(patcher.model) == []

    restore_pristine(patcher.model, record)
    file_state = load_file(str(path))
    for key in patches:
        name = key[0] if isinstance(key, tuple) else key
        assert torch.equal(live_tensor(patcher.model, name), file_state[name[len("diffusion_model."):]]), name
    # Restored bytes in new tensors: bits on aarch64, float32 rounding elsewhere.
    assert _same(sample(patcher), first)
    probe.close()
    assert probe.adapter == [True, True] * 4
