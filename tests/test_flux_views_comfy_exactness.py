"""UltraTex view conditioning under pure Ulysses returns stock ComfyUI's bits (real model, CPU).

ComfyUI 0.45.0 (commit 61e524b) taught the Flux forward three conditioning
inputs for the UltraTex LoRAs: ``view_grid`` tiles each sample's n views into
one image, ``token_keep`` (from ``preprocess_token_masks``, run once by
``model_base.Flux.extra_conds``) drops background tokens before the blocks and
restores them after, and ``reference_views`` places a reference over its
view's tile. Stock ``Flux._forward`` applies all three around ``forward_orig``,
which the Flux 2 adapter rebinds under sequence parallelism
(adapters/flux_views.py).

This builds comfy's own ``comfy.ldm.flux.model.Flux`` in FLUX.2's layout at
toy width, loads it through ComfyUI's own diffusion-model load into a real
``model_base.Flux2``, builds every input with that model's real
``extra_conds`` and comfy's own cond processing, runs stock's forward, then
runs the adapter's bound forward on two rank threads (the harness of
tests/test_flux3_comfy_exactness.py: only the all-to-all, the sequence gather
and the kernel are stubbed, and the stub kernel is comfy's own
``optimized_attention`` on the full gathered rows). Both ranks must return
stock's output and each other's, and every fused-norm launch on every rank
must hold the rows stock's launch for the same norm holds, because the token
drop changes those counts and comfy-kitchen's CUDA kernel depends on them.
The cases take each input alone and all three together, batch 1 and batch 2,
and streams whose image and joint lengths are even and odd, so both the
unpadded route and the pad-excluding route run. "Equal" follows the FLUX.3
file: bit equality on this rig's aarch64 CPU, float32 rounding on x86.

The FSDP case wraps the model as adapters/fsdp.py wraps it, on a one-rank gloo
group: its ``extra_conds`` runs ``preprocess_token_masks`` on the wrapped
model with every block parameter a DTensor, and returns the plain twin's
``token_keep``; the wrapped forward then returns the twin's output. The last
case shows why cfg2 and dp2 refuse: half a sample's views raise stock's own
``view_grid`` error. The file skips where ComfyUI predates UltraTex.
"""
from __future__ import annotations

import copy
import os
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.flux_family import Flux2Adapter
from module_location_helpers import from_checkout
from test_flux3_comfy_exactness import _same, _StockKernelHarness
from test_flux_pad_exclusion import _LOCAL, _install_stubs, _rank

WORLD = 2
COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
HIDDEN, HEADS, CHANNELS, TEXT_DIM = 256, 2, 128, 64
GRID = [(0, 0), (0, 2), (1, 0), (0, 1), (1, 1), (1, 2)]


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy_flux():
    """Import real ComfyUI's Flux model and loader, then leave sys.modules as found."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before = list(sys.path)
    argv_before = sys.argv
    if os.path.isdir(COMFY_DIR) and COMFY_DIR not in sys.path:
        sys.path.insert(0, COMFY_DIR)
    try:
        sys.argv = ["pytest-flux-views-comfy-exactness", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            flux_model = pytest.importorskip("comfy.ldm.flux.model")
            import comfy.ldm.flux.math
            import comfy.quant_ops
            import comfy.sd
        finally:
            sys.argv = argv_before
        if not hasattr(flux_model.Flux, "preprocess_token_masks"):
            pytest.skip("this ComfyUI predates UltraTex (commit 61e524b, 0.45.0): its Flux "
                        "forward reads no view input, so there is nothing to compare")
        yield SimpleNamespace(ops=ops, model=flux_model, sd=comfy.sd, kitchen=comfy.quant_ops.ck,
                              adaln=comfy.quant_ops.ck.adaln,
                              kernel=comfy.ldm.flux.math.optimized_attention)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, COMFY_DIR)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


@pytest.fixture(autouse=True)
def one_blas_thread():
    """One CPU thread for every GEMM. With many, the CPU BLAS splits a call's
    rows across threads by the row count, so a rank's half-height call can
    round a row differently from stock's whole one; that is the CPU's
    scheduling, not the sharded forward, and one thread takes it out. A
    hardware leg reads the GPU's own bits."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def _base(comfy, seed=0):
    """A real model_base.Flux2 from ComfyUI's own load of a toy FLUX.2-layout state dict."""
    torch.manual_seed(seed)
    model = comfy.model.Flux(
        in_channels=CHANNELS, out_channels=CHANNELS, vec_in_dim=None, context_in_dim=TEXT_DIM,
        hidden_size=HIDDEN, mlp_ratio=3.0, num_heads=HEADS, depth=1, depth_single_blocks=2,
        axes_dim=[32, 32, 32, 32], theta=2000, patch_size=1, qkv_bias=False,
        guidance_embed=False, txt_ids_dims=[3], global_modulation=True, mlp_silu_act=True,
        ops_bias=False, default_ref_method="index", ref_index_scale=10.0,
        dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.05)
    loaded = comfy.sd.load_diffusion_model_state_dict(
        model.state_dict(), model_options={"dtype": torch.float32}).model
    assert type(loaded).__name__ == "Flux2" and type(loaded.diffusion_model) is comfy.model.Flux
    return loaded


def _view_masks(views: int, size: int, cut: int) -> torch.Tensor:
    """Object masks over each view's pixels, anchored top left, that leave
    background tokens to drop: ComfyUI keeps a token any coverage reaches and
    dilates the kept set by one token right and down (``foreground_tokens``),
    so a block ``16 * k - 8`` pixels high keeps ``k + 1`` token rows. ``cut`` and
    the view's index set how many rows and columns stay background."""
    masks = torch.zeros(views, size * 16, size * 16)
    for index in range(views):
        rows = 16 * max(size - 1 - cut - (index % 2), 1) - 8
        cols = 16 * max(size - 1 - cut, 1) - 8
        masks[index, :rows, :cols] = 1.0
    return masks


def _case_kwargs(name: str, size: int, cut: int) -> tuple[list, dict]:
    """The grid and the extras UltraTex Conditioning puts on the conditioning, per case.

    Stock reads both other inputs against the grid: ``preprocess_token_masks``
    maps the masks over it and a view reference takes its tile's offset. So
    the ``token_keep`` and ``reference_views`` cases ride a one-view grid,
    which tiles nothing, and the ``token_keep`` case drops the grid from the
    forward's inputs (``_forward_inputs``).
    """
    torch.manual_seed(11)
    grid = GRID if name in ("view_grid", "all", "four_views") else [(0, 0)]
    if name == "four_views":
        grid = GRID[:4]
    rows = max(r for r, _ in grid) + 1
    cols = max(c for _, c in grid) + 1
    atlas = torch.randn(1, CHANNELS, rows * size, cols * size)
    reference = torch.randn(1, CHANNELS, size, size)
    extras: dict = {"reference_latents": [atlas, reference], "reference_latents_method": "index",
                    "view_grid": grid}
    if name in ("token_keep", "all", "four_views"):
        target = _view_masks(len(grid), size, cut)
        atlas_mask = torch.zeros(1, rows * size * 16, cols * size * 16)
        atlas_mask[:, :, : cols * size * 8] = 1.0
        extras["token_masks"] = {"target": target, "refs": [atlas_mask, None]}
        extras["background_latent"] = torch.randn(1, CHANNELS, size, size)
    if name in ("reference_views", "all", "four_views"):
        extras["reference_views"] = [None, 0]
    return grid, extras


def _forward_inputs(base_model, grid, extras, size, batch):
    """Stock's own extra_conds and cond processing for one model call of ``batch`` samples."""
    rows = len(grid) * batch
    torch.manual_seed(5)
    noise = torch.randn(rows, CHANNELS, size, size)
    conds = base_model.extra_conds(noise=noise, device=torch.device("cpu"),
                                   cross_attn=torch.randn(1, 7, TEXT_DIM), **extras)
    processed = {key: cond.process_cond(batch_size=rows, device=torch.device("cpu")).cond
                 for key, cond in conds.items()}
    context = processed.pop("c_crossattn")
    timestep = torch.linspace(0.8, 0.4, batch).repeat_interleave(len(grid))
    if "reference_views" not in extras and len(grid) == 1:
        processed.pop("view_grid", None)  # token_keep alone
    return (torch.randn(rows, CHANNELS, size, size), timestep, context), processed, conds


def _record_launches(monkeypatch, comfy) -> dict:
    """Every fused-norm launch's rows, keyed by the rank thread that made it."""
    adaln = comfy.adaln  # the kitchen's own op, never an earlier recorder
    launches: dict = {}

    def record(x, scale, shift, eps=1e-6):
        launches.setdefault(_rank(), []).append(int(x.shape[1]))
        return adaln(x, scale, shift, eps)

    monkeypatch.setattr(comfy.kitchen, "adaln", record)
    return launches


def _stock(monkeypatch, comfy, model, inputs, kwargs):
    launches = _record_launches(monkeypatch, comfy)
    with torch.no_grad():
        out = model(*inputs, transformer_options={}, **kwargs)
    return out, comfy.kernel, list(launches.get(0, []))


def _sharded(monkeypatch, comfy, model, inputs, kwargs, kernel):
    harness = _StockKernelHarness(WORLD, kernel)
    _install_stubs(monkeypatch, harness)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    models = [copy.deepcopy(model) for _ in range(WORLD)]
    for rank_model in models:
        Flux2Adapter().inject_usp(rank_model, InjectionContext(
            topology_sp=WORLD, usp_attention=dispatch, pure_ulysses=True))
    launches = _record_launches(monkeypatch, comfy)
    outputs: list = [None] * WORLD
    errors: list = [None] * WORLD

    def body(rank):
        try:
            _LOCAL.rank = rank
            with torch.no_grad():
                outputs[rank] = models[rank](*inputs, transformer_options={}, **kwargs)
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
    return outputs, harness, launches


def _streams(kwargs, size, refs, text):
    """(image rows, joint rows) of the stream stock's forward_orig takes."""
    keep = kwargs.get("token_keep")
    grid = kwargs.get("view_grid") or [(0, 0)]
    if keep is not None:
        image = int(keep["target"].numel())
        image += sum(int(r.numel()) if r is not None else int(ref.shape[-2] * ref.shape[-1])
                     for r, ref in zip(keep["refs"], refs, strict=True))
    else:
        image = len(grid) * size * size + sum(int(r.shape[-2] * r.shape[-1]) for r in refs)
    return image, image + text


# (case, view size in tokens, background cut, batch, parity of the image and
# joint streams stock's forward_orig takes: "even" runs xfuser's unpadded
# route in the single blocks, "odd" the pad-excluding one.)
CASES = {
    "view-grid-b1": ("view_grid", 3, 0, 1, "odd"),
    "view-grid-b2": ("view_grid", 4, 0, 2, "even"),
    "token-keep-even": ("token_keep", 4, 2, 1, "even"),
    "token-keep-odd": ("token_keep", 5, 1, 1, "odd"),
    "reference-views-b1": ("reference_views", 3, 0, 1, "odd"),
    "reference-views-b2": ("reference_views", 4, 0, 2, "even"),
    "all-b1-odd": ("all", 3, 1, 1, "odd"),
    "all-b1-even": ("all", 4, 2, 1, "even"),
    "all-b2": ("all", 3, 1, 2, "odd"),
    "four-views-b2": ("four_views", 4, 1, 2, "even"),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_pure_ulysses_returns_stock_bits_with_view_inputs(monkeypatch, comfy_flux, case):
    name, size, cut, batch, parity = CASES[case]
    base_model = _base(comfy_flux)
    grid, extras = _case_kwargs(name, size, cut)
    inputs, kwargs, _conds = _forward_inputs(base_model, grid, extras, size, batch)
    model = base_model.diffusion_model
    assert ("token_keep" in kwargs) == (name in ("token_keep", "all", "four_views"))
    assert ("reference_views" in kwargs) == (name in ("reference_views", "all", "four_views"))
    text = inputs[2].shape[1]  # Flux 2 front-pads the text to 512 rows
    image, joint = _streams(kwargs, size, extras["reference_latents"], text)
    if "token_keep" in kwargs:
        assert kwargs["token_keep"]["target"].numel() < len(grid) * size * size  # tokens drop
    assert {image % 2, joint % 2} == ({0} if parity == "even" else {1}), (image, joint)

    expected, kernel, stock_launches = _stock(monkeypatch, comfy_flux, model, inputs, kwargs)
    assert expected.shape == inputs[0].shape
    outputs, harness, launches = _sharded(monkeypatch, comfy_flux, model, inputs, kwargs, kernel)
    assert torch.equal(outputs[0], outputs[1])
    assert _same(outputs[0], expected)
    # One double and two single blocks: the double block's two image and two
    # text norms at the stock image and text rows, each single block's at the
    # joint rows, on every rank, then the final layer's on the gathered image.
    assert stock_launches == [image, text, image, text, joint, joint, image]
    assert launches == dict.fromkeys(range(WORLD), stock_launches)
    # The single blocks take xfuser's own call on an even joint stream and the
    # pad-excluding one on an odd stream; the double block, whose rank-major
    # [text, image] windows pure Ulysses puts back in stock's order, always
    # takes the exact one.
    entries = [entry for entry, *_ in harness.seen[0]]
    assert entries == ["exact", *(["xfuser" if parity == "even" else "exact"] * 2)]


def test_the_fsdp_wrapped_model_runs_preprocess_token_masks_and_the_forward_as_stock(
        monkeypatch, comfy_flux, tmp_path):
    """FSDP2 on a one-rank gloo group, wrapped as adapters/fsdp.py wraps Flux 2:
    every block list per block, then the whole model, the census's parameters
    ignored. ``extra_conds`` runs on the wrapped model while every block
    parameter is a DTensor that no pre-forward hook has gathered, and returns
    the plain twin's ``token_keep``; the forward then returns its output."""
    import torch.distributed.fsdp
    from torch.distributed.device_mesh import init_device_mesh

    from dgx_monarch.adapters.fsdp import _BLOCK_LIST_ATTRS
    from dgx_monarch.adapters.fsdp_islands import auxiliary_module_parameters, resolve_module_path

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    twin, wrapped = _base(comfy_flux), _base(comfy_flux)
    grid, extras = _case_kwargs("all", 4, 1)
    inputs, kwargs, _conds = _forward_inputs(twin, grid, extras, 4, 1)
    with torch.no_grad():
        expected = twin.diffusion_model(*inputs, transformer_options={}, **kwargs)
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        dm = wrapped.diffusion_model
        for parameter in dm.parameters():
            parameter.requires_grad = False
        ignored = auxiliary_module_parameters(dm, wrapped)
        assert not ignored  # nothing of Flux runs outside its forward
        for attr in _BLOCK_LIST_ATTRS:
            for block in resolve_module_path(dm, attr) or ():
                torch.distributed.fsdp.fully_shard(block, mesh=mesh, ignored_params=ignored)
        torch.distributed.fsdp.fully_shard(dm, mesh=mesh, reshard_after_forward=True,
                                           ignored_params=ignored)
        assert type(dm.double_blocks[0].img_attn.qkv.weight).__name__ == "DTensor"
        assert type(dm.img_in.weight).__name__ == "DTensor"
        got_inputs, got_kwargs, _ = _forward_inputs(wrapped, grid, extras, 4, 1)
        keep, want = got_kwargs["token_keep"], kwargs["token_keep"]
        assert torch.equal(keep["target"], want["target"])
        assert torch.equal(keep["background"], want["background"])
        assert torch.equal(keep["refs"][0], want["refs"][0]) and keep["refs"][1] is None
        with torch.no_grad():
            out = dm(*got_inputs, transformer_options={}, **got_kwargs)
        assert torch.equal(out, expected) or torch.allclose(out, expected, rtol=0, atol=1e-6)
    finally:
        torch.distributed.destroy_process_group()


def test_half_a_samples_views_raise_stocks_own_view_grid_error(comfy_flux):
    """What cfg2 at CFG 1 and dp2 hand each rank of a one-sample render: half its
    views. Stock refuses that batch, which is why both topologies refuse first."""
    base_model = _base(comfy_flux)
    grid, extras = _case_kwargs("all", 3, 1)
    inputs, kwargs, _conds = _forward_inputs(base_model, grid, extras, 3, 1)
    half = len(grid) // 2
    x, timestep, context = inputs
    with pytest.raises(ValueError, match="view_grid: the latent batch"), torch.no_grad():
        base_model.diffusion_model(x[:half], timestep[:half], context[:half],
                                   transformer_options={}, **kwargs)
