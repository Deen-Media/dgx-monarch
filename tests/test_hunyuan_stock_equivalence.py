"""Real ComfyUI HunyuanVideo: the two-rank pure-Ulysses forward produces exactly the stock output.

This builds comfy's own `HunyuanVideo` at toy sizes in float32, once with the
HunyuanImage 2.1 layout (2-D patch, byt5 glyph rows after the text) and once
with the HunyuanVideo 1.5 layout (3-D patch, cond-type embedding, vision and
byt5 rows before the text). Stock runs its own forward once. The adapter's
forward then runs on two rank threads through the real
`base.make_usp_attention` dispatch and full-axis call, with only the
all-to-all, the gather and the kernel stubbed (the harness of
tests/test_flux_pad_exclusion.py).

Both sides attend through one kernel, so the claim is about operands: every
kernel call, the TokenRefiner's included, gets exactly stock's queries, keys
and values in stock's order, each text-stream Linear sees stock's rows in
stock's layout, and the output equals stock with a max abs difference of 0.0.
The claim needs a GEMM that gives a row the same bits at the shard's row count
as at the full one, since the image-stream and single-block Linears still run
on shards; each case checks that on the CPU it runs on, at the row counts and
batch it shards, and skips where it does not hold.

Skipped where ComfyUI is not importable, as in the ordinary CI job; the
comfy-canary job runs it against comfy master.
comfy.cli_args parses argv on import, so the fixture forces ``--cpu``.
"""
from __future__ import annotations

import copy
import math
import os
import sys
import threading

import pytest
import torch

from dgx_monarch.adapters import chroma_text, hunyuan
from dgx_monarch.adapters.base import InjectionContext, make_usp_attention
from dgx_monarch.adapters.hunyuan import HunyuanAdapter
from module_location_helpers import from_checkout
from test_flux_pad_exclusion import _LOCAL, _Harness, _install_stubs, _rank

WORLD = 2
CONTEXT_DIM, HIDDEN, HEADS, VISION_DIM, BYT5_DIM = 12, 32, 2, 8, 1472
TEXT_LINEARS = ("txt_attn.qkv", "txt_attn.proj", "txt_mlp.0", "txt_mlp.2")
# The Linears a shard runs on half the rows: image rows in the double blocks,
# joint rows in the single blocks.
SHARDED_LINEARS = (("double_blocks", ("img_attn.qkv", "img_attn.proj", "img_mlp.0",
                                      "img_mlp.2")),
                   ("single_blocks", ("linear1", "linear2")))


def _comfy_dir() -> str:
    value = os.environ.get("COMFYUI_DIR") or os.environ.get("COMFY_DIR") or "~/ComfyUI"
    return os.path.abspath(os.path.expanduser(value))


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy_hunyuan():
    """Import real comfy on its CPU path, then put sys.modules back as found."""
    comfy_dir = _comfy_dir()
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        sys.argv = ["pytest-hunyuan-stock-equivalence", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI checkout on sys.path")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            model = pytest.importorskip("comfy.ldm.hunyuan_video.model")
        finally:
            sys.argv = original_argv
        yield ops, model
    finally:
        # Classify first, then pop: a namespace package's path re-resolves
        # through its parent in sys.modules while it is being read.
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


_MODELS: dict[tuple[bool, bool, int], torch.nn.Module] = {}


def _build(comfy_hunyuan, *, video: bool, guidance: bool, singles: int = 2):
    """A seeded toy HunyuanVideo; built once per layout and copied per run."""
    key = (video, guidance, singles)
    if key not in _MODELS:
        ops, model_module = comfy_hunyuan
        torch.manual_seed(7 + 2 * video + guidance)
        model = model_module.HunyuanVideo(
            dtype=torch.float32, device="cpu", operations=ops.disable_weight_init,
            in_channels=4, out_channels=4, vec_in_dim=None, context_in_dim=CONTEXT_DIM,
            hidden_size=HIDDEN, mlp_ratio=4.0, num_heads=HEADS, depth=2,
            depth_single_blocks=singles, axes_dim=[4, 6, 6] if video else [8, 8], theta=256,
            patch_size=[1, 1, 1] if video else [1, 1], qkv_bias=True,
            guidance_embed=guidance, byt5=True, meanflow=False,
            use_cond_type_embedding=video, vision_in_dim=VISION_DIM if video else None,
            meanflow_sum=False)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2:
                    fan_in = math.prod(parameter.shape[1:])
                    parameter.normal_(0.0, 1.0 / math.sqrt(fan_in))
                else:
                    parameter.normal_(0.0, 0.1)
        _MODELS[key] = model
    return copy.deepcopy(_MODELS[key])


def _kernel(q, k, v):
    """One attention for both sides, on (B, H, L, D) in one memory layout."""
    return torch.nn.functional.scaled_dot_product_attention(
        q.contiguous(), k.contiguous(), v.contiguous())


class _Recorder(_Harness):
    """Each rank's kernel operands in call order, in stock's (B, H, L, D) layout."""

    def __init__(self, world: int) -> None:
        super().__init__(world, strict=False)
        self.seen: dict[int, list] = {rank: [] for rank in range(world)}

    def attend(self, entry, q, k, v):
        local = q.shape[1]
        if entry == "xfuser":
            # xfuser's own Ulysses call gathers the query axis as it does the
            # keys, so its kernel sees the whole sequence too.
            q = torch.cat(self.fabric.exchange(q), dim=1)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        self.seen[_rank()].append((entry, q.clone(), k.clone(), v.clone()))
        out = _kernel(q, k, v).transpose(1, 2)
        if entry == "xfuser":
            out = out[:, _rank() * local:(_rank() + 1) * local]
        return out


def _stock_attention(sink: list):
    """Comfy's override seam: stock's own call, recorded, through the same kernel."""
    def override(func, q, k, v, heads, mask=None, skip_reshape=False,
                 skip_output_reshape=False, **kwargs):
        assert skip_reshape and mask is None and not skip_output_reshape
        sink.append(("stock", q.clone(), k.clone(), v.clone()))
        out = _kernel(q, k, v)
        return out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)
    return override


def _real_strides(value: torch.Tensor) -> tuple[int, ...]:
    """A size-1 axis's stride is never read, so it is left out."""
    return tuple(step for step, size in zip(value.stride(), value.shape, strict=True) if size != 1)


def _spy_text_linears(model, sink: list) -> None:
    """Record (block, name, shape, strides) for each text Linear call."""
    for index, block in enumerate(model.double_blocks):
        for name in TEXT_LINEARS:
            module = block.get_submodule(name)

            def spy(value, module=module, index=index, name=name):
                sink.append((index, name, tuple(value.shape), _real_strides(value)))
                return type(module).forward(module, value)
            module.forward = spy


def _sharded_rows(case) -> dict[str, int]:
    """The rows each block kind's sharded Linears cover in this case."""
    text = case["qwen"] + case["byt5"] + case.get("clip", 0)
    image = math.prod(case["grid"]) + (math.prod(case["ref"]) if case.get("ref") else 0)
    return {"double_blocks": image, "single_blocks": text + image}


def _shards_match_whole(module, batch: int, rows: int, generator) -> bool:
    """Each rank's rows, the last rank's zero-padded as shard_seq pads them,
    give the bits the whole stream gives."""
    local = -(-rows // WORLD)
    value = torch.randn(batch, rows, module.in_features, generator=generator)
    padded = torch.cat(
        [value, value.new_zeros(batch, local * WORLD - rows, module.in_features)], dim=1)
    shards = torch.cat([module(padded[:, rank * local:(rank + 1) * local].contiguous())
                        for rank in range(WORLD)], dim=1)
    return torch.equal(module(value), shards[:, :rows])


def _gemm_rows_are_invariant(model, case, batch: int) -> bool:
    """Does this CPU give each sharded Linear stock's bits at this case's row counts?"""
    generator = torch.Generator().manual_seed(3)
    rows = _sharded_rows(case)
    for kind, names in SHARDED_LINEARS:
        for block in getattr(model, kind):
            for name in names:
                if not _shards_match_whole(block.get_submodule(name), batch, rows[kind],
                                           generator):
                    return False
    return True


def _inputs(case, batch):
    generator = torch.Generator().manual_seed(11 + batch)
    video = case.get("video", False)
    x = torch.randn(batch, 4, *case["grid"], generator=generator)
    context = torch.randn(batch, case["qwen"], CONTEXT_DIM, generator=generator)
    kwargs = {"txt_byt5": (torch.randn(batch, case["byt5"], BYT5_DIM, generator=generator)
                           if case["byt5"] else None)}
    if case.get("guidance"):
        kwargs["guidance"] = torch.full((batch,), 3.5)
    if case.get("clip"):
        kwargs["clip_fea"] = torch.randn(batch, case["clip"], VISION_DIM, generator=generator)
    if case.get("ref"):
        kwargs["ref_latent"] = torch.randn(batch, 4, *case["ref"], generator=generator)
    timesteps = torch.rand(batch, generator=generator)
    assert video == (len(case["grid"]) == 3)
    return x, timesteps, context, kwargs


def _stock(model, x, timesteps, context, kwargs):
    calls, linears = [], []
    _spy_text_linears(model, linears)
    with torch.no_grad():
        out = model._forward(x, timesteps, context, transformer_options={
            "optimized_attention_override": _stock_attention(calls)}, **kwargs)
    return out, calls, linears


def _sharded(monkeypatch, models, x, timesteps, context, kwargs):
    monkeypatch.setattr(hunyuan, "sp_rank", _rank)
    monkeypatch.setattr(chroma_text, "sp_rank", _rank)
    monkeypatch.setattr(chroma_text, "sp_world", lambda: WORLD)
    harness = _Recorder(WORLD)
    _install_stubs(monkeypatch, harness)
    attention = make_usp_attention("TORCH_FLASH")
    linears: dict[int, list] = {rank: [] for rank in range(WORLD)}
    for rank, model in enumerate(models):
        _spy_text_linears(model, linears[rank])
        HunyuanAdapter().inject_usp(model, InjectionContext(
            topology_sp=WORLD, usp_attention=attention, pure_ulysses=True))
    outputs: list = [None] * WORLD
    errors: list = [None] * WORLD

    def body(rank):
        try:
            _LOCAL.rank = rank
            with torch.no_grad():
                outputs[rank] = models[rank]._forward(x, timesteps, context, transformer_options={
                    "optimized_attention_override": _stock_attention(harness.seen[rank])},
                    **kwargs)
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
    return outputs, harness.seen, linears


# Rows per stream: text is vision + byt5 + Qwen rows, image is reference plus
# latent rows, and the joint stream of the single loop is their sum.
CASES = {
    "image-even-text-even-image": {"qwen": 6, "byt5": 4, "grid": (4, 4)},      # 10, 16, 26
    "image-odd-text": {"qwen": 5, "byt5": 4, "grid": (4, 4)},                  # 9, 16, 25
    "image-odd-image": {"qwen": 6, "byt5": 4, "grid": (3, 3)},                 # 10, 9, 19
    "image-odd-text-odd-image": {"qwen": 5, "byt5": 4, "grid": (3, 5)},        # 9, 15, 24
    "image-no-glyphs": {"qwen": 7, "byt5": 0, "grid": (4, 4)},                 # 7, 16, 23
    "image-guidance": {"qwen": 5, "byt5": 4, "grid": (3, 3), "guidance": True},  # 9, 9, 18
    "video-vision-odd-text": {"qwen": 5, "byt5": 3, "grid": (2, 3, 3), "clip": 3,
                              "guidance": True, "video": True},                # 11, 18, 29
    "video-vision-even": {"qwen": 6, "byt5": 4, "grid": (2, 2, 3), "clip": 2,
                          "video": True},                                      # 12, 12, 24
    # The released 720p t2v checkpoint has double blocks only (its header,
    # read 2026-10-06), so the joined stream runs no block before the gather.
    "video-no-single-blocks": {"qwen": 5, "byt5": 4, "grid": (2, 3, 3), "video": True,
                               "singles": 0},                                  # 9, 18, 27
}
# Stock's own forward writes into an einops repeat view for a reference latent
# (model.py `ref_latent_ids[..., 0] = -1`) and raises at batch 2, so the
# reference case runs at batch 1 only.
REFERENCE_CASE = {"qwen": 5, "byt5": 3, "grid": (1, 3, 3), "ref": (1, 2, 2),
                  "video": True}                                               # 8, 13, 21


def _assert_stock(monkeypatch, comfy_hunyuan, case, batch):
    video = case.get("video", False)
    stock_model = _build(comfy_hunyuan, video=video, guidance=bool(case.get("guidance")),
                         singles=case.get("singles", 2))
    if not _gemm_rows_are_invariant(stock_model, case, batch):
        pytest.skip("this CPU's GEMM gives a row different bits at the shard's row count, so "
                    "the image-stream and single-block Linears, which stay sharded, cannot be "
                    "compared bit for bit here")
    models = [copy.deepcopy(stock_model) for _ in range(WORLD)]
    x, timesteps, context, kwargs = _inputs(case, batch)
    stock, stock_calls, stock_linears = _stock(stock_model, x, timesteps, context, kwargs)
    outputs, seen, linears = _sharded(monkeypatch, models, x, timesteps, context, kwargs)

    labels = ([f"token refiner {i}" for i in range(2)]
              + [f"double {i}" for i in range(len(stock_model.double_blocks))]
              + [f"single {i}" for i in range(len(stock_model.single_blocks))])
    assert len(stock_calls) == len(labels)
    for rank in range(WORLD):
        assert len(seen[rank]) == len(labels), rank
        for label, (_s, *stock_qkv), (entry, *rank_qkv) in zip(
                labels, stock_calls, seen[rank], strict=True):
            for name, ours, theirs in zip("qkv", rank_qkv, stock_qkv, strict=True):
                assert ours.shape == theirs.shape and torch.equal(ours, theirs), (
                    f"rank {rank} {label}: the kernel's {name} ({entry} path, "
                    f"shape {tuple(ours.shape)}) is not stock's {tuple(theirs.shape)}")
        assert linears[rank] == stock_linears, f"rank {rank}: a text Linear saw other rows"
        assert outputs[rank].shape == stock.shape
        assert (outputs[rank] - stock).abs().max().item() == 0.0
        assert torch.equal(outputs[rank], stock)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("name", list(CASES))
def test_two_rank_forward_matches_stock_bit_for_bit(monkeypatch, comfy_hunyuan, name, batch):
    _assert_stock(monkeypatch, comfy_hunyuan, CASES[name], batch)


def test_reference_latent_rows_match_stock_bit_for_bit(monkeypatch, comfy_hunyuan):
    _assert_stock(monkeypatch, comfy_hunyuan, REFERENCE_CASE, 1)


def test_real_comfy_block_selects_its_four_text_linears(comfy_hunyuan):
    ops, _model = comfy_hunyuan
    from comfy.ldm.flux.layers import DoubleStreamBlock

    def block(operations):
        return DoubleStreamBlock(16, 2, mlp_ratio=4.0, qkv_bias=True, dtype=torch.float32,
                                 device="cpu", operations=operations)

    for family in ("disable_weight_init", "manual_cast", "fp8_ops"):
        real = block(getattr(ops, family))
        selected = hunyuan._text_modules(real)
        assert [module for module, _ in selected] == [
            real.get_submodule(name) for name in TEXT_LINEARS], family
        assert [joint for _, joint in selected] == [False, True, False, False]
    mixed = block(ops.mixed_precision_ops({}, torch.float32))
    assert not isinstance(mixed.txt_attn.qkv, torch.nn.Linear)
    assert hunyuan._text_modules(mixed) == ()
