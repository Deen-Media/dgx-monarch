"""FLUX.3 Klein under pure Ulysses produces exactly the stock ComfyUI output (real model, CPU).

This builds comfy's own ``comfy.ldm.flux3.model.Flux3`` at toy width in
float32 (hidden 256, two heads, so the head dimension stays 128 for the four
32-wide rotary axes; one image block and two single blocks), runs its stock
forward once, then runs the adapter's bound forward on two rank threads. Only
the all-to-all, the sequence gather and the attention kernel are stubbed (the
harness of tests/test_flux_pad_exclusion.py), and the stub kernel is comfy's
own ``optimized_attention`` on the full gathered rows, so the attention math
is stock's. Each kernel call records its q, k and v, which must equal stock's
call for the same block, with no pad row and in stock's key order, and the
output must equal stock's. The comfy-canary job runs this file against comfy
master, because the bound forward re-expresses stock's prelude orchestration
and no other test fails when upstream changes it.

The cases cover an even text-to-image stream, an odd grid that pads both
phases (the family's named pad-exclusion probe, docs/ADAPTERS.md), a rank
whose image window holds only a pad, one reference whose segment the shard
boundary cuts, two references, batch 2, the step-distilled flow map with a
real two-row sigma schedule, fused qkv, and the base model's cached states
against the uncached path. Every fused-norm launch on every rank holds the
rows stock's launch for the same span holds, because comfy-kitchen's CUDA
kernel for it depends on the call's rows.

"Equal" follows tests/test_ernie_stock_equivalence.py: bit equality on this
rig's aarch64 CPU, where the GEMMs give the same bits per row at these row
counts, and float32 rounding (allclose 2e-5) on x86, whose BLAS may pick
another kernel for the halved row count. The ring control shows that an
attended pad row breaks that agreement, and a ring render that pads only one
phase refuses before any block runs. Rank against rank is always exact.

The LoRA half maps and bakes synthetic rank-4 LoRAs on a tiny real
``model_base.Flux3`` loaded from a file in the vendor key layout: ComfyUI's
fused key map, the store's per-key bake against ComfyUI's own merge, and the
FSDP bake of a census-replicated early-block key through actor/fsdp_lora.py.
The fixture forces ``--cpu`` because comfy.cli_args parses argv on import,
and skips where ComfyUI is not importable.
"""
from __future__ import annotations

import copy
import os
import platform
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import accuracy_waiver
from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.flux3 import Flux3Adapter
from dgx_monarch.adapters.stock_attention import default_attention_override
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from module_location_helpers import from_checkout
from test_flux_pad_exclusion import _LOCAL, _Harness, _install_stubs, _rank

WORLD = 2
COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
HIDDEN, HEADS, CHANNELS, TEXT_DIM = 256, 2, 128, 20480
_EXACT_GEMM = platform.machine() == "aarch64"


def _same(candidate: torch.Tensor, reference: torch.Tensor) -> bool:
    """Equal bits where the GEMMs are known to agree per row, else float32 rounding."""
    if candidate.shape != reference.shape:
        return False
    if _EXACT_GEMM:
        return torch.equal(candidate, reference)
    return torch.equal(candidate, reference) or torch.allclose(
        candidate, reference, rtol=2e-5, atol=2e-5)


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy_flux3():
    """Import real ComfyUI's FLUX.3 model, then leave sys.modules as found."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before = list(sys.path)
    argv_before = sys.argv
    if os.path.isdir(COMFY_DIR) and COMFY_DIR not in sys.path:
        sys.path.insert(0, COMFY_DIR)
    try:
        sys.argv = ["pytest-flux3-comfy-exactness", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            flux3_model = pytest.importorskip(
                "comfy.ldm.flux3.model", reason="this ComfyUI predates FLUX.3 (0.41.0)")
            import comfy.lora
            import comfy.model_patcher
            import comfy.sd
        finally:
            sys.argv = argv_before
        yield SimpleNamespace(ops=ops, model=flux3_model, lora=comfy.lora,
                              model_patcher=comfy.model_patcher, sd=comfy.sd)
    finally:
        # Classify first, then pop: a namespace package's path re-resolves
        # through its parent in sys.modules while it is being read.
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, COMFY_DIR)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def _tiny(comfy, *, fused=False, flow_map=False, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    model = comfy.model.Flux3(
        hidden_size=HIDDEN, num_heads=HEADS, depth=1, depth_single_blocks=2,
        fused_qkv=fused, flow_map=flow_map, dtype=dtype, device="cpu",
        operations=comfy.ops.disable_weight_init)
    # comfy builds state-dict parameters with torch.empty; a real load fills
    # them from a checkpoint, so a toy run fills them here.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.05)
    return model


# (grid, reference grids, text rows, batch): t2i even; the odd grid that pads
# both phases (25 image rows, 105 joint rows); a 1 x 1 grid whose image phase
# leaves rank 1 one pad row and no real row, so its first span is kept empty;
# one reference whose segment the boundary cuts (16 + 30 + 8 = 54, boundary at
# 27); two references with an odd joint stream (12 + 9 + 8 + 6 = 35, boundary
# at 18 inside them); batch 2 on an odd grid, where the shard is a strided chunk.
CASES = {
    "t2i-even": ((4, 4), (), 80, 1),
    "odd-grid": ((5, 5), (), 80, 1),
    "pad-only-rank": ((1, 1), (), 80, 1),
    "one-ref": ((4, 4), ((5, 6),), 8, 1),
    "two-refs": ((3, 4), ((3, 3), (2, 4)), 6, 1),
    "batch-2": ((3, 3), ((2, 3),), 80, 2),
}


def _inputs(grid, refs, text_rows, batch, seed=1):
    torch.manual_seed(seed)
    x = torch.randn(batch, CHANNELS, *grid)
    timestep = torch.tensor([0.6, 0.35][:batch])
    context = torch.randn(batch, text_rows, TEXT_DIM) * 0.1
    references = [torch.randn(1, CHANNELS, *shape) for shape in refs] or None
    return x, timestep, context, references


class _StockKernelHarness(_Harness):
    """Runs comfy's attention kernel on the full rows stock would hand it."""

    def __init__(self, world, kernel):
        super().__init__(world, strict=False)
        self.kernel = kernel
        self.seen: dict[int, list] = {rank: [] for rank in range(world)}

    def attend(self, entry, q, k, v):
        rank = _rank()
        local = q.shape[1]
        if entry == "xfuser":
            # xfuser's own call holds only this rank's queries; gather them so
            # the kernel sees stock's full rows, then keep this rank's share.
            q = torch.cat(self.fabric.exchange(q), dim=1)
        batch, rows, heads, dim = q.shape
        flat = [t.reshape(batch, -1, heads * dim) for t in (q, k, v)]
        self.seen[rank].append((entry, *flat))
        out = self.kernel(*flat, heads).reshape(batch, rows, heads, dim)
        return out[:, rank * local:(rank + 1) * local] if entry == "xfuser" else out


def _plain(value):
    """The tensor behind comfy's AttentionTensorContainer, or the tensor itself."""
    return value if isinstance(value, torch.Tensor) else value.take()


def _stock(monkeypatch, comfy, model, inputs, options, **states):
    """Stock's forward, recording every attention call's operands."""
    kernel = comfy.model.optimized_attention
    calls: list[tuple] = []

    def record(q, k, v, heads, *args, **kwargs):
        q, k, v = (_plain(t) for t in (q, k, v))
        calls.append((q, k, v))
        return kernel(q, k, v, heads, *args, **kwargs)

    monkeypatch.setattr(comfy.model, "optimized_attention", record)
    try:
        with torch.no_grad():
            out = model(*inputs, transformer_options=dict(options), **states)
    finally:
        monkeypatch.setattr(comfy.model, "optimized_attention", kernel)
    return out, calls, kernel


def _sharded(monkeypatch, model, inputs, options, kernel, *, ring_world=1, harness=None,
             **states):
    harness = harness if harness is not None else _StockKernelHarness(WORLD, kernel)
    _install_stubs(monkeypatch, harness)
    monkeypatch.setattr(sys.modules["xfuser.core.distributed"],
                        "get_ring_parallel_world_size", lambda: ring_world)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    models = [copy.deepcopy(model) for _ in range(WORLD)]
    for rank_model in models:
        Flux3Adapter().inject_usp(rank_model, InjectionContext(
            topology_sp=WORLD, usp_attention=dispatch, pure_ulysses=ring_world == 1))
    outputs: list = [None] * WORLD
    errors: list = [None] * WORLD

    def body(rank):
        try:
            _LOCAL.rank = rank
            with torch.no_grad():
                outputs[rank] = models[rank](*inputs, transformer_options=dict(options), **states)
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
    return outputs, harness


def _assert_stock_operands_and_output(model, stock_calls, harness, expected, outputs):
    sharded_blocks = len(model.content_mode_blocks["image"]) + len(model.single_blocks)
    stock_sharded = stock_calls[-sharded_blocks:]
    for rank in range(WORLD):
        seen = harness.seen[rank]
        assert len(seen) == sharded_blocks
        for (_entry, q, k, v), (stock_q, stock_k, stock_v) in zip(seen, stock_sharded, strict=True):
            # The same keys in the same order, no pad row, and the same queries and values.
            assert _same(k, stock_k)
            assert _same(q, stock_q)
            assert _same(v, stock_v)
    assert torch.equal(outputs[0], outputs[1])
    assert outputs[0].shape == expected.shape
    assert _same(outputs[0], expected)


@pytest.mark.parametrize("case", sorted(CASES))
def test_pure_ulysses_hands_every_kernel_stock_operands_and_returns_stock_bits(
    monkeypatch, comfy_flux3, case
):
    grid, refs, text_rows, batch = CASES[case]
    model = _tiny(comfy_flux3)
    inputs = _inputs(grid, refs, text_rows, batch)
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_flux3, model, inputs, {})
    outputs, harness = _sharded(monkeypatch, model, inputs, {}, kernel)
    _assert_stock_operands_and_output(model, stock_calls, harness, expected, outputs)
    image_rows = grid[0] * grid[1]
    joint_rows = image_rows + sum(h * w for h, w in refs) + text_rows
    # Each phase takes the exclusion exactly when its stream is odd.
    entries = [entry for entry, *_ in harness.seen[0]]
    assert entries[0] == ("exact" if image_rows % WORLD else "xfuser")
    assert set(entries[1:]) == {"exact" if joint_rows % WORLD else "xfuser"}


@pytest.mark.parametrize("refs", [(), ((5, 6),)], ids=["t2i", "edit"])
def test_the_step_distilled_flow_map_matches_stock(monkeypatch, comfy_flux3, refs):
    """A real two-row schedule: the next sigma is the jump target every rank embeds."""
    model = _tiny(comfy_flux3, flow_map=True)
    inputs = _inputs((5, 5), refs, 8, 1)
    options = {"sample_sigmas": torch.tensor([0.6, 0.0])}
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_flux3, model, inputs, options)
    outputs, harness = _sharded(monkeypatch, model, inputs, options, kernel)
    _assert_stock_operands_and_output(model, stock_calls, harness, expected, outputs)
    # The jump target moved the output: the flow map is live, not inert.
    without_target = copy.deepcopy(model)
    without_target.flow_map = False
    with torch.no_grad():
        plain = without_target(*inputs, transformer_options={})
    assert not torch.equal(plain, expected)


def test_a_missing_sample_schedule_refuses_class_p_where_stock_raises_keyerror(
    monkeypatch, comfy_flux3
):
    model = _tiny(comfy_flux3, flow_map=True)
    inputs = _inputs((4, 4), (), 8, 1)
    with pytest.raises(KeyError, match="sample_sigmas"), torch.no_grad():
        model(*inputs, transformer_options={})
    kernel = comfy_flux3.model.optimized_attention
    with pytest.raises(UnsupportedModelError, match="sigma schedule") as caught:
        _sharded(monkeypatch, model, inputs, {}, kernel)
    tag = parse_leading_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS


@pytest.mark.parametrize("grid", [(4, 4), (5, 5)], ids=["even", "odd"])
def test_fused_qkv_matches_stock(monkeypatch, comfy_flux3, grid):
    model = _tiny(comfy_flux3, fused=True)
    assert hasattr(model.single_blocks[0], "qkv")
    inputs = _inputs(grid, (), 80, 1)
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_flux3, model, inputs, {})
    outputs, harness = _sharded(monkeypatch, model, inputs, {}, kernel)
    _assert_stock_operands_and_output(model, stock_calls, harness, expected, outputs)


@pytest.mark.parametrize("refs", [(), ((5, 6),), ((3, 3), (2, 4))],
                         ids=["t2i", "one-ref", "two-refs"])
def test_cached_states_match_stock_cached_and_uncached(monkeypatch, comfy_flux3, refs):
    """The base model's extra_conds branch hands _forward its early states."""
    model = _tiny(comfy_flux3)
    x, timestep, context, references = _inputs((5, 5), refs, 8, 1)
    with torch.no_grad():
        txt_states, ref_states = model.preprocess_conds(context, references)
    states = {"txt_states": txt_states, "ref_states": ref_states,
              "ref_shapes": [tuple(r.shape[-2:]) for r in references] if references else None}
    inputs = (x, timestep, None, None)
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_flux3, model, inputs, {}, **states)
    sharded_blocks = len(model.content_mode_blocks["image"]) + len(model.single_blocks)
    assert len(stock_calls) == sharded_blocks   # no early block ran
    uncached, _calls, _kernel = _stock(
        monkeypatch, comfy_flux3, model, (x, timestep, context, references), {})
    assert torch.equal(expected, uncached)
    outputs, harness = _sharded(monkeypatch, model, inputs, {}, kernel, **states)
    _assert_stock_operands_and_output(model, stock_calls, harness, expected, outputs)


def test_the_early_blocks_run_the_stock_backend_and_the_sharded_blocks_the_override(
    monkeypatch, comfy_flux3
):
    """Under sequence parallelism comfy runs the text and reference early
    blocks from preprocess_conds with no options; the bind hands their
    attention the stock backend, and the sharded blocks get the USP override."""
    model = _tiny(comfy_flux3)
    inputs = _inputs((4, 4), ((2, 3),), 8, 1)
    _expected, _calls, kernel = _stock(monkeypatch, comfy_flux3, model, inputs, {})
    seen: list = []

    def record(*args, **kwargs):
        override = (kwargs.get("transformer_options") or {}).get("optimized_attention_override")
        seen.append((_rank(), override))
        return kernel(*args, **kwargs)

    monkeypatch.setattr(comfy_flux3.model, "optimized_attention", record)
    _sharded(monkeypatch, model, inputs, {}, kernel)
    for rank in range(WORLD):
        overrides = [override for who, override in seen if who == rank]
        early = len(model.txt_mode_blocks) + len(model.content_mode_blocks["image_cond"])
        assert overrides[:early] == [default_attention_override] * early
        sharded = overrides[early:]
        assert len(sharded) == len(model.content_mode_blocks["image"]) + len(model.single_blocks)
        assert all(o is not default_attention_override
                   and getattr(o, base.USP_ATTENTION_OVERRIDE_ATTR, False) for o in sharded)


def _is_subsequence(part: list, whole: list) -> bool:
    remaining = iter(whole)
    return all(any(item == candidate for candidate in remaining) for item in part)


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_fused_norm_launch_on_every_rank_holds_stocks_rows_for_its_span(
    monkeypatch, comfy_flux3, case
):
    """comfy-kitchen picks its fused AdaLN kernel by a call's rows, and on a
    CUDA device the kernels round a few outputs apart, so a rank's call must
    hold the rows stock's call for the same span holds. Each rank's launches
    are therefore stock's launches with only the spans the rank holds no row
    of left out (adapters/flux3_adaln_parity.py). The comfy-canary job runs
    this against comfy master, so a stock block that stops calling
    ``modulated_norm`` or changes its calls fails here by name."""
    grid, refs, text_rows, batch = CASES[case]
    model = _tiny(comfy_flux3)
    inputs = _inputs(grid, refs, text_rows, batch)
    kitchen = sys.modules["comfy.quant_ops"].ck
    adaln = kitchen.adaln
    launches: dict = {"stock": [], **{rank: [] for rank in range(WORLD)}}

    def record(x, scale, shift, eps=1e-6):
        who = "stock" if threading.current_thread() is threading.main_thread() else _rank()
        launches[who].append(tuple(x.shape[:2]))
        return adaln(x, scale, shift, eps)

    monkeypatch.setattr(kitchen, "adaln", record)
    expected, _calls, kernel = _stock(monkeypatch, comfy_flux3, model, inputs, {})
    outputs, _harness = _sharded(monkeypatch, model, inputs, {}, kernel)
    joint_rows = grid[0] * grid[1] + sum(h * w for h, w in refs) + text_rows
    for rank in range(WORLD):
        assert _is_subsequence(launches[rank], launches["stock"]), (rank, launches)
        # Each single block's first call covers the whole joint stream, as stock's does.
        assert launches[rank].count((batch, joint_rows)) == len(model.single_blocks)
    assert _same(outputs[0], expected)


def test_a_padded_ring_refuses_class_k_and_a_waived_ring_attends_the_pad_and_moves(
    monkeypatch, comfy_flux3
):
    """The negative control on the real model: ring and hybrid cannot exclude a
    pad row, so a padded render refuses class K before the first block, and
    under a granted ring_pad waiver the pad rows reach the kernel and the
    output moves. The exact match above therefore comes from the exclusion."""
    model = _tiny(comfy_flux3)
    inputs = _inputs((5, 5), (), 80, 1)
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_flux3, model, inputs, {})
    with pytest.raises(UnsupportedModelError, match="ring/hybrid") as caught:
        _sharded(monkeypatch, model, inputs, {}, kernel, ring_world=2)
    tag = parse_leading_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.KNOWN_WRONG
    monkeypatch.setattr(accuracy_waiver, "waived", lambda guard: guard == base.RING_PAD_GUARD)
    outputs, harness = _sharded(monkeypatch, model, inputs, {}, kernel, ring_world=2)
    stock_sharded = stock_calls[-3:]
    for rank in range(WORLD):
        for (_entry, _q, k, _v), (_sq, stock_k, _sv) in zip(
                harness.seen[rank], stock_sharded, strict=True):
            assert k.shape[1] == stock_k.shape[1] + 1
    assert not _same(outputs[0], expected)


# (grid, reference grids, text rows): 25 image rows and N = 25 + 25 + 80 = 130,
# so only the image phase pads at world 2; 16 image rows and
# N = 16 + 16 + 25 + 80 = 137, so only the joint phase pads.
ONE_PHASE_PADS = {
    "image-phase-only": ((5, 5), ((5, 5),), 80),
    "joint-phase-only": ((4, 4), ((4, 4), (5, 5)), 80),
}


@pytest.mark.parametrize("case", sorted(ONE_PHASE_PADS))
def test_a_ring_render_padded_in_one_phase_refuses_before_any_sharded_block(
    monkeypatch, comfy_flux3, case
):
    """The ring refusal counts both phases' pads up front: a render that pads
    only the joint phase must not run the image blocks and the image gather
    first, and one that pads only the image phase must not reach the kernel."""
    grid, refs, text_rows = ONE_PHASE_PADS[case]
    model = _tiny(comfy_flux3)
    inputs = _inputs(grid, refs, text_rows, 1)
    image_rows = grid[0] * grid[1]
    joint_rows = image_rows + sum(h * w for h, w in refs) + text_rows
    assert (image_rows % WORLD, joint_rows % WORLD) == (
        (1, 0) if case == "image-phase-only" else (0, 1))
    _expected, _calls, kernel = _stock(monkeypatch, comfy_flux3, model, inputs, {})
    entered: list = []
    for block in list(model.content_mode_blocks["image"]) + list(model.single_blocks):
        # Each rank's deep copy keeps the hook, so every rank reports here.
        block.register_forward_pre_hook(lambda module, args: entered.append(module))
    harness = _StockKernelHarness(WORLD, kernel)
    with pytest.raises(UnsupportedModelError, match="ring/hybrid") as caught:
        _sharded(monkeypatch, model, inputs, {}, kernel, ring_world=2, harness=harness)
    tag = parse_leading_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.KNOWN_WRONG
    assert entered == []
    assert harness.seen == {rank: [] for rank in range(WORLD)}


# --- LoRA: ComfyUI's key map and merge on a tiny real model_base.Flux3 ---


def _vendor_state_dict(model) -> dict[str, torch.Tensor]:
    """The tiny model's weights under the vendor file's key names.

    Every vendor DiT header stores the QK norms as ``*.norm.query_norm.scale``
    and ``*.norm.key_norm.scale``, which ComfyUI's Flux loader renames to
    ``.weight`` (supported_models.Flux.process_unet_state_dict). Writing those
    names makes a tiny file take the rename a real file takes.
    """
    return {key.replace("_norm.weight", "_norm.scale"): value.contiguous()
            for key, value in model.state_dict().items()}


def _base_model(comfy, tmp_path, *, fused, dtype=torch.float32, seed=0):
    """A tiny real model_base.Flux3, loaded from a vendor-layout file through
    ComfyUI's own diffusion-model load (the store's load call)."""
    from safetensors.torch import save_file

    tiny = _tiny(comfy, fused=fused, dtype=dtype, seed=seed)
    path = tmp_path / f"flux3_tiny_{'fused' if fused else 'split'}_{seed}.safetensors"
    save_file(_vendor_state_dict(tiny), str(path))
    patcher = comfy.sd.load_diffusion_model(str(path), model_options={"dtype": dtype})
    import comfy.model_base

    assert type(patcher.model) is comfy.model_base.Flux3
    assert patcher.model.diffusion_model.single_blocks[0].fused_qkv is fused
    return patcher, str(path)


def test_comfys_key_map_routes_split_projection_loras_to_fused_qkv_thirds(
    comfy_flux3, tmp_path
):
    patcher, _path = _base_model(comfy_flux3, tmp_path, fused=True)
    key_map = comfy_flux3.lora.model_lora_keys_unet(patcher.model, {})
    qkv = "diffusion_model.single_blocks.0.qkv.weight"
    for index, projection in enumerate(("q_proj", "k_proj", "v_proj")):
        expected = (qkv, (0, index * HIDDEN, HIDDEN))
        assert key_map[f"diffusion_model.single_blocks.0.{projection}"] == expected
        assert key_map[f"lora_unet_single_blocks_0_{projection}"] == expected
    split, _ = _base_model(comfy_flux3, tmp_path, fused=False)
    split_map = comfy_flux3.lora.model_lora_keys_unet(split.model, {})
    assert split_map["diffusion_model.single_blocks.0.q_proj"] == (
        "diffusion_model.single_blocks.0.q_proj.weight")


_LORA_UNITS = (
    "single_blocks.0.q_proj", "single_blocks.0.k_proj", "single_blocks.0.v_proj",
    "single_blocks.0.attn_out", "single_blocks.0.mlp_in",
    "content_mode_blocks.image.0.q_proj", "txt_mode_blocks.0.v_proj",
)


def _rank4_lora(units, *, seed=7, width=HIDDEN, rank=4) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    lora = {}
    for unit in units:
        out_rows = 2 * 3 * width if unit.endswith("mlp_in") else width
        lora[f"diffusion_model.{unit}.lora_up.weight"] = torch.randn(out_rows, rank) * 0.1
        lora[f"diffusion_model.{unit}.lora_down.weight"] = torch.randn(rank, width) * 0.1
        lora[f"diffusion_model.{unit}.alpha"] = torch.tensor(float(rank))
    return lora


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "split"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_the_stores_bake_equals_comfys_own_merge_on_every_patched_key(
    comfy_flux3, tmp_path, fused, dtype
):
    """The ambient-verify contract (actor/store_bake.py) on FLUX.3's own keys."""
    from dgx_monarch.actor.model_store import ModelStore
    from dgx_monarch.actor.unbake import live_tensor

    # Twin patchers on two loads of one file: comfy's merge reads one, the
    # store's in-place bake writes the other.
    reference, _path = _base_model(comfy_flux3, tmp_path, fused=fused, dtype=dtype)
    ours, _path = _base_model(comfy_flux3, tmp_path, fused=fused, dtype=dtype)
    key_map = comfy_flux3.lora.model_lora_keys_unet(ours.model, {})
    patches = comfy_flux3.lora.load_lora(_rank4_lora(_LORA_UNITS), key_map)
    assert len(patches) == len(_LORA_UNITS)
    cpu = torch.device("cpu")
    reference.add_patches(patches, 0.7)
    ours.add_patches(patches, 0.7)
    keys = sorted(ours.patches)
    if fused:
        assert "diffusion_model.single_blocks.0.qkv.weight" in keys
        assert len(ours.patches["diffusion_model.single_blocks.0.qkv.weight"]) == 3
    for key in keys:
        expected = reference.patch_weight_to_device(key, device_to=cpu, return_weight=True)
        ModelStore._bake_key(ours, key)
        baked = live_tensor(ours.model, key)
        assert baked.dtype == expected.dtype == dtype
        assert torch.equal(baked, expected), key


def test_a_q_only_delta_changes_only_qs_rows_of_the_fused_weight(comfy_flux3, tmp_path):
    from dgx_monarch.actor.model_store import ModelStore
    from dgx_monarch.actor.unbake import live_tensor

    patcher, _path = _base_model(comfy_flux3, tmp_path, fused=True)
    key = "diffusion_model.single_blocks.0.qkv.weight"
    before = live_tensor(patcher.model, key).clone()
    key_map = comfy_flux3.lora.model_lora_keys_unet(patcher.model, {})
    patches = comfy_flux3.lora.load_lora(_rank4_lora(("single_blocks.0.q_proj",)), key_map)
    assert list(patches) == [(key, (0, 0, HIDDEN))]
    active = patcher.clone()
    active.add_patches(patches, 1.0)
    ModelStore._bake_key(active, key)
    after = live_tensor(active.model, key)
    assert not torch.equal(after[:HIDDEN], before[:HIDDEN])
    assert torch.equal(after[HIDDEN:], before[HIDDEN:])


def test_the_fsdp_bake_on_bf16_equals_the_resident_bake_with_a_replicated_early_block_key(
    comfy_flux3, tmp_path
):
    """FSDP2 on a one-rank gloo group, wrapped as adapters/fsdp.py wraps FLUX.3:
    the five image blocks and the single blocks per block through the dotted
    block list, the early text and reference streams replicated through the
    class-scoped census. A key on txt_mode_blocks.0 then bakes whole
    (actor/fsdp_lora.py's replicated-parameter path) and a key on a sharded
    block through its shard; both
    equal ComfyUI's merge on an unwrapped twin."""
    import torch.distributed.fsdp
    from torch.distributed.device_mesh import init_device_mesh

    from dgx_monarch.actor import fsdp_lora
    from dgx_monarch.adapters.fsdp import _BLOCK_LIST_ATTRS
    from dgx_monarch.adapters.fsdp_islands import (
        auxiliary_module_parameters,
        resolve_module_path,
    )

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    reference, _path = _base_model(comfy_flux3, tmp_path, fused=False, dtype=torch.bfloat16)
    active, path = _base_model(comfy_flux3, tmp_path, fused=False, dtype=torch.bfloat16)
    model = active.model
    key_map = comfy_flux3.lora.model_lora_keys_unet(model, {})
    units = ("single_blocks.0.q_proj", "content_mode_blocks.image.0.v_proj",
             "txt_mode_blocks.0.v_proj", "content_mode_blocks.image_cond.0.attn_out")
    patches = comfy_flux3.lora.load_lora(_rank4_lora(units), key_map)
    keys = sorted(patches)
    assert len(keys) == len(units)
    cpu = torch.device("cpu")
    reference.add_patches(patches, 0.7)
    expected = {key: reference.patch_weight_to_device(key, device_to=cpu, return_weight=True)
                for key in keys}
    active.add_patches(patches, 0.7)
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        dm = model.diffusion_model
        for parameter in dm.parameters():
            parameter.requires_grad = False
        ignored = auxiliary_module_parameters(dm, model)
        wrapped = 0
        for attr in _BLOCK_LIST_ATTRS:
            blocks = resolve_module_path(dm, attr)
            for block in blocks or ():
                torch.distributed.fsdp.fully_shard(block, mesh=mesh, ignored_params=ignored)
                wrapped += 1
        assert wrapped == len(dm.content_mode_blocks["image"]) + len(dm.single_blocks)
        torch.distributed.fsdp.fully_shard(
            dm, mesh=mesh, reshard_after_forward=True, ignored_params=ignored)
        early = dm.txt_mode_blocks[0].v_proj.weight
        assert type(early) is torch.nn.Parameter
        assert type(dm.single_blocks[0].q_proj.weight).__name__ == "DTensor"
        fsdp_lora.apply_stack(SimpleNamespace(swap_verify=-1), active, path, None)
        for key in keys:
            live = fsdp_lora.live_tensor(model, key)
            local = live if type(live) is torch.nn.Parameter else fsdp_lora.local_shard(live)
            assert local.dtype == expected[key].dtype == torch.bfloat16
            assert torch.equal(local, expected[key]), key
    finally:
        torch.distributed.destroy_process_group()
