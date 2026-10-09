"""Real ComfyUI MageFlow against the distributed adapter, on CPU.

Two rank threads share a lock-step collective simulation; comfy's stock
MageFlowTransformer2DModel supplies every projection, modulation, RoPE and
attention block, at toy sizes to keep CPU cost low. Skips unless COMFY_DIR or
COMFYUI_DIR names a ComfyUI with the Mage model; the comfy-canary workflow
runs it.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest
import torch

from module_location_helpers import from_checkout


def _comfy_dir() -> Path | None:
    value = os.environ.get("COMFY_DIR") or os.environ.get("COMFYUI_DIR")
    if not value:
        return None
    path = Path(value)
    return path if (path / "comfy" / "ldm" / "mage_flow" / "model.py").is_file() else None


COMFY_DIR = _comfy_dir()
pytestmark = pytest.mark.skipif(COMFY_DIR is None, reason="current ComfyUI Mage source is unavailable")


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


def _from_checkout(module: object) -> bool:
    """Does this module's file live under the checkout? A real comfy import
    also adds node_helpers, nodes, execution, latent_preview, server and more
    under no comfy prefix, and those must leave with it."""
    return COMFY_DIR is not None and from_checkout(module, os.path.abspath(str(COMFY_DIR)))


@pytest.fixture(autouse=True)
def _isolate_comfy_modules():
    """Import against an empty comfy namespace and leave it as it was found.

    A stub `comfy` an earlier file leaves behind makes `import comfy.options`
    fail here with "'comfy' is not a package", and a restore that keeps only
    the names present before leaves the real tree's comfy_ and checkout
    modules loaded for later stub-based files (both seen 2026-10-05). So the
    snapshot goes back exactly: present names return, everything imported
    here goes.
    """
    preserved = {name: module for name, module in sys.modules.items() if _is_comfy_module(name)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before = list(sys.path)
    argv_before = list(sys.argv)
    try:
        yield
    finally:
        # Classify first, then pop: a namespace package's path re-resolves
        # through its parent in sys.modules while it is being read.
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or _from_checkout(module)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before
        sys.argv[:] = argv_before


def _prepare_comfy() -> None:
    assert COMFY_DIR is not None
    if str(COMFY_DIR) not in sys.path:
        sys.path.insert(0, str(COMFY_DIR))
    sys.argv = ["mage-flow-cpu-differential", "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()


def _pad_rows(segments: list[tuple[int, int]]) -> list[int]:
    world = 2
    chunk = sum(local for _, local in segments)
    rows: list[int] = []
    offset = 0
    for original, local in segments:
        for padded in range(original, local * world):
            rank, within = divmod(padded, local)
            rows.append(rank * chunk + offset + within)
        offset += local
    return sorted(rows)


class _SimulatedUSP:
    """Lock-step rank collective preserving the all-gather ordering exactly."""

    def __init__(self) -> None:
        self._barrier = threading.Barrier(2)
        self._lock = threading.Lock()
        self._rounds: dict[int, dict[int, torch.Tensor]] = {}

    def all_gather(self, rank: int, round_id: int, value: torch.Tensor) -> list[torch.Tensor]:
        with self._lock:
            self._rounds.setdefault(round_id, {})[rank] = value.detach().clone()
        self._barrier.wait()
        values = [self._rounds[round_id][item] for item in range(2)]
        self._barrier.wait()
        return values


def _rank_forward(rank, state, x, timestep, context, refs, rank_context, attention, output) -> None:
    assert COMFY_DIR is not None
    _prepare_comfy()
    import comfy.ops as ops
    from comfy.ldm.mage_flow.model import MageFlowTransformer2DModel

    from dgx_monarch.adapters.base import InjectionContext
    from dgx_monarch.adapters.mage_flow import MageFlowAdapter

    model = MageFlowTransformer2DModel(
            in_channels=4, out_channels=4, num_layers=1, attention_head_dim=8,
            num_attention_heads=2, joint_attention_dim=16, axes_dims_rope=(2, 2, 4),
            dtype=torch.float32, device=torch.device("cpu"), operations=ops.disable_weight_init,
    )
    model.load_state_dict(state)
    rank_context.rank = rank
    MageFlowAdapter().inject_usp(model, InjectionContext(2, attention))
    output[rank] = model._forward(x, timestep, context, ref_latents=refs).detach()


@pytest.mark.parametrize("reference_count", (0, 1, 3))
def test_real_comfy_stock_and_two_rank_usp_match(reference_count, tmp_path):
    assert COMFY_DIR is not None
    _prepare_comfy()
    import comfy.ops as ops
    from comfy.ldm.mage_flow.model import MageFlowTransformer2DModel
    from comfy.ldm.modules.attention import optimized_attention_masked

    torch.manual_seed(814)
    model = MageFlowTransformer2DModel(
        in_channels=4, out_channels=4, num_layers=1, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=16, axes_dims_rope=(2, 2, 4),
        dtype=torch.float32, device=torch.device("cpu"), operations=ops.disable_weight_init,
    )
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
    x = torch.randn(2, 4, 3, 5)  # odd axes exercise Mage's centring convention
    timestep = torch.tensor([0.25, 0.75])
    context = torch.randn(2, 3, 16)  # text length 3 forces a USP pad row
    refs = [torch.randn(2, 4, 3, 5) for _ in range(reference_count)]
    expected = model._forward(x, timestep, context, ref_latents=refs).detach()

    collective = _SimulatedUSP()
    import dgx_monarch.adapters.mage_flow as mage
    original = mage.shard_seq, mage.sp_gather, mage.sp_rank, mage.padded_row_indices
    rank_context = threading.local()
    rounds = {0: 0, 1: 0}
    rounds_lock = threading.Lock()

    def gathered(value):
        rank = rank_context.rank
        with rounds_lock:
            round_id = rounds[rank]
            rounds[rank] += 1
        return collective.all_gather(rank, round_id, value.contiguous())

    def shard(value, dim=1):
        original_length = value.shape[dim]
        padded = (2 - original_length % 2) % 2
        if padded:
            shape = list(value.shape)
            shape[dim] = padded
            value = torch.cat((value, value.new_zeros(shape)), dim=dim)
        return torch.chunk(value, 2, dim=dim)[rank_context.rank], original_length

    def gather(value, original_length, dim=1):
        return torch.cat(gathered(value), dim=dim).narrow(dim, 0, original_length)

    def full_attention(q, k, v, heads, mask=None, **kwargs):
        sequence_dim = 2 if q.ndim == 4 else 1
        q_full, k_full, v_full = (torch.cat(gathered(value), dim=sequence_dim) for value in (q, k, v))
        drops = kwargs.pop("drop_rows", None) or []
        keep = torch.ones(q_full.shape[sequence_dim], dtype=torch.bool)
        keep[drops] = False
        index = [slice(None)] * q_full.ndim
        index[sequence_dim] = keep
        active = tuple(index)
        result = optimized_attention_masked(q_full[active], k_full[active], v_full[active], heads, mask=mask, **kwargs)
        full = result.new_zeros(result.shape[0], q_full.shape[sequence_dim], result.shape[-1])
        full[:, keep] = result
        width = q.shape[sequence_dim]
        return full[:, rank_context.rank * width:(rank_context.rank + 1) * width]

    mage.shard_seq, mage.sp_gather, mage.sp_rank, mage.padded_row_indices = shard, gather, lambda: rank_context.rank, _pad_rows
    results: dict[int, torch.Tensor] = {}
    threads = [threading.Thread(
        target=_rank_forward,
        args=(rank, model.state_dict(), x, timestep, context, refs, rank_context, full_attention, results),
    ) for rank in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()
    mage.shard_seq, mage.sp_gather, mage.sp_rank, mage.padded_row_indices = original
    assert torch.allclose(results[0], expected, atol=2e-5, rtol=2e-5)
    assert torch.equal(results[0], results[1])


def _tiny_mage(state=None):
    import comfy.ops as ops
    from comfy.ldm.mage_flow.model import MageFlowTransformer2DModel

    model = MageFlowTransformer2DModel(
        in_channels=4, out_channels=4, num_layers=1, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=16, axes_dims_rope=(2, 2, 4),
        dtype=torch.float32, device=torch.device("cpu"), operations=ops.disable_weight_init,
    )
    if state is None:
        for parameter in model.parameters():
            torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
    else:
        model.load_state_dict(state)
    return model


def _cfg_inputs(batch=2):
    x = torch.randn(batch, 4, 3, 5)
    timestep = torch.tensor([0.25, 0.75])[:batch]
    refs = [torch.randn(batch, 4, 3, 5)]
    return x, timestep, refs


def test_real_comfy_cfg_equalizer_restores_each_unpadded_call():
    _prepare_comfy()
    from dgx_monarch.actor.sampling import equalize_cond_lengths
    from dgx_monarch.adapters.mage_flow import MageFlowAdapter

    torch.manual_seed(615)
    stock = _tiny_mage()
    padded = _tiny_mage(stock.state_dict())
    MageFlowAdapter().inject_cfg_pad_forward(padded)
    x, timestep, refs = _cfg_inputs(batch=1)
    long = torch.randn(1, 5, 16)
    short = torch.randn(1, 3, 16)
    positive, negative = equalize_cond_lengths(
        MageFlowAdapter(), [[long, {}]], [[short, {}]], torch.zeros(1, 4, 3, 5),
    )
    got_long = padded._forward(x, timestep, positive[0][0], ref_latents=refs)
    got_short = padded._forward(x, timestep, negative[0][0], ref_latents=refs)
    assert torch.allclose(got_long, stock._forward(x, timestep, long, ref_latents=refs), atol=2e-5, rtol=2e-5)
    assert torch.allclose(got_short, stock._forward(x, timestep, short, ref_latents=refs), atol=2e-5, rtol=2e-5)


def test_real_comfy_cfg_padding_handles_ragged_batch_and_preserves_real_mask():
    _prepare_comfy()
    from dgx_monarch.adapters.mage_flow import MageFlowAdapter

    torch.manual_seed(616)
    stock = _tiny_mage()
    padded = _tiny_mage(stock.state_dict())
    MageFlowAdapter().inject_cfg_pad_forward(padded)
    x, timestep, refs = _cfg_inputs()
    first, second = torch.randn(1, 5, 16), torch.randn(1, 3, 16)
    ragged = torch.cat((first, torch.cat((second, torch.zeros(1, 2, 16)), dim=1)), dim=0)
    got = padded._forward(x, timestep, ragged, ref_latents=refs)
    expected = torch.cat((
        stock._forward(x[:1], timestep[:1], first, ref_latents=[ref[:1] for ref in refs]),
        stock._forward(x[1:], timestep[1:], second, ref_latents=[ref[1:] for ref in refs]),
    ))
    assert torch.allclose(got, expected, atol=2e-5, rtol=2e-5)

    real_mask = torch.zeros(2, 5)
    captured = []
    original = type(padded)._forward

    def recording(self, *args, **kwargs):
        captured.append(kwargs.get("attention_mask"))
        return original(self, *args, **kwargs)

    type(padded)._forward = recording
    try:
        padded._forward(x, timestep, torch.cat((first, first)), attention_mask=real_mask, ref_latents=refs)
    finally:
        type(padded)._forward = original
    assert captured[-1] is real_mask


def test_real_comfy_cfg_uniform_trailing_pad_is_stock_equivalent():
    _prepare_comfy()
    from dgx_monarch.adapters.mage_flow import MageFlowAdapter

    torch.manual_seed(617)
    stock = _tiny_mage()
    padded = _tiny_mage(stock.state_dict())
    MageFlowAdapter().inject_cfg_pad_forward(padded)
    x, timestep, refs = _cfg_inputs()
    context = torch.randn(2, 3, 16)
    padded_context = torch.cat((context, torch.zeros(2, 2, 16)), dim=1)
    got = padded._forward(x, timestep, padded_context, ref_latents=refs)
    expected = stock._forward(x, timestep, context, ref_latents=refs)
    assert torch.allclose(got, expected, atol=2e-5, rtol=2e-5)
