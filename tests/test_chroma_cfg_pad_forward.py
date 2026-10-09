"""Chroma's cfg2 pad+mask hook: exactness against stock.

At cfg2 each rank holds batch 1 of one cond (rank 0 cond, rank 1 uncond).
When the two prompts differ in length, `actor/sampling.equalize_cond_lengths`
zero-pads the shorter one's text to the longer one's length and attaches an
additive bias over the joint [text, image] keys so the batch still concats
into one cfg-parallel call. Stock's forward on that padded, masked context
drifted from the single-GPU render 6 to 10 times further than any other
family's cfg2 or uly2 probe in the 2026-09 sweep (2026-09-02). The
equalizer's bias math itself is exact (CPU float32, same kernel either side),
so the gap is the kernel: the workers run comfy's pytorch attention, and the
additive mask sends the call to a different SDPA kernel than the unpadded
single-GPU call uses. `_chroma_cfg_pad_trim` restores kernel parity by
handing stock exactly what an unpadded single-GPU call would have built. On
the shipped pair, the one-step cfg2 probe against one GPU read 0.089551 bf16
and 0.161908 fp8_scaled before the trim and 0.0 after (docs/VALIDATION.md,
Chroma cfg2 pad+mask trim, 2026-10-01).

The first half tests `_chroma_cfg_pad_trim` directly: pure tensors, no comfy
import, always runs. The second half drives a real, tiny
`comfy.ldm.chroma.model.Chroma` (the production class, shrunk to blocks small
enough to run in milliseconds) through `ChromaAdapter().inject_cfg_pad_forward`
and proves `torch.equal` against stock on the unpadded call, for both the
short (padded) and long (still-rounded) rank, plus a refusal for a batch
whose real lengths disagree and a no-op when the equalizer never built a
mask at all. It skips where ComfyUI is not importable; the comfy-canary
job runs this file against comfy master, with COMFYUI_DIR naming the
checkout (it overrides the default ``~/ComfyUI``). comfy.cli_args parses
argv on import, so the fixture forces ``--cpu`` the way
tests/test_render_memory_price.py does, or a CPU-only runner dies on the
CUDA branch before any test runs.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.flux_family import ChromaAdapter, _chroma_cfg_pad_trim

# Part 1: the trim function alone, pure tensors.
DTYPE = torch.float32
MIN = torch.finfo(DTYPE).min


def _bias(real_lengths: list[int], longest: int, image_keys: int) -> torch.Tensor:
    """The exact additive bias `equalize_cond_lengths` builds for one batch:
    real text rows 0, padded text rows the dtype minimum, image keys 0."""
    rows = []
    key_pos = torch.arange(longest + image_keys)
    for rows_real in real_lengths:
        appended = (key_pos >= rows_real) & (key_pos < longest)
        rows.append(appended.to(DTYPE) * MIN)
    return torch.stack(rows).unsqueeze(1)  # (B, 1, longest+image_keys)


def test_a_missing_mask_is_a_no_op():
    context = torch.randn(1, 5, 4)
    trimmed, mask = _chroma_cfg_pad_trim(context, None)
    assert trimmed is context
    assert mask is None


def test_a_uniformly_padded_batch_trims_to_the_real_length_and_drops_the_mask():
    context = torch.randn(2, 7, 4)
    context[:, 3:] = 0.0  # the equalizer's own zero-fill for the pad rows
    mask = _bias([3, 3], longest=7, image_keys=9)
    trimmed, out_mask = _chroma_cfg_pad_trim(context, mask)
    assert torch.equal(trimmed, context[:, :3])
    assert out_mask is None


def test_a_batch_already_at_the_padded_length_needs_no_trim_but_drops_the_mask():
    """The rank holding the longer prompt: real length == padded length, so
    the mask is a pure no-op, and dropping it (rather than keeping an
    all-zero bias) is what keeps this rank off the masked-attention kernel."""
    context = torch.randn(1, 7, 4)
    mask = _bias([7], longest=7, image_keys=9)
    trimmed, out_mask = _chroma_cfg_pad_trim(context, mask)
    assert trimmed is context
    assert out_mask is None


def test_disagreeing_real_lengths_refuse_instead_of_guessing():
    context = torch.randn(2, 7, 4)
    mask = _bias([3, 5], longest=7, image_keys=9)
    with pytest.raises(UnsupportedModelError):
        _chroma_cfg_pad_trim(context, mask)


def test_a_mask_with_a_nonzero_image_zone_is_left_alone():
    """Not the equalizer's construction (a regional-conditioning mask that
    skipped it, sampling.py's "already ships its own"); do not guess."""
    context = torch.randn(1, 5, 4)
    mask = torch.zeros(1, 1, 5 + 9)
    mask[:, :, -1] = MIN  # a real bias somewhere in the image zone
    trimmed, out_mask = _chroma_cfg_pad_trim(context, mask)
    assert trimmed is context
    assert out_mask is mask


def test_a_mask_whose_text_zone_is_not_a_clean_prefix_is_left_alone():
    context = torch.randn(1, 5, 4)
    bias = torch.zeros(5)
    bias[1] = MIN  # a masked row in the middle, not a trailing pad
    mask = bias.reshape(1, 1, 5)
    trimmed, out_mask = _chroma_cfg_pad_trim(context, mask)
    assert trimmed is context
    assert out_mask is mask


def test_a_fully_masked_row_trims_to_zero_real_tokens():
    """The mask is ground truth, unlike content-sniffing: a row whose bias
    marks every text position as padding has zero real tokens, even though
    its content (random, not zero-filled here) is nonzero."""
    context = torch.randn(1, 6, 4)
    mask = _bias([0], longest=6, image_keys=9)
    trimmed, out_mask = _chroma_cfg_pad_trim(context, mask)
    assert trimmed.shape[1] == 0
    assert out_mask is None


# Part 2: the real, tiny Chroma, driven through the installed hook.
HIDDEN = 16
RAW_CHANNELS = 1
PATCH = 2
CONTEXT_DIM = 8


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy_chroma_classes():
    """Import real `comfy.ops` / `comfy.ldm.chroma.model`, then undo it.

    Same snapshot-and-restore as tests/test_slab_hook_canary.py and
    tests/test_comfy_seam_contracts.py: without this, `sys.path` keeps
    `~/ComfyUI` and `sys.modules` keeps a real `comfy` package for the rest
    of the pytest process, so a later test that probes "is comfy importable"
    (those two canaries, run from a cwd where their own relative `../ComfyUI`
    does not exist) gets a real answer instead of the "not on sys.path" skip
    it gets on its own. Module-scoped: one import pays for every test below,
    and the restore still runs before any later test file collects.
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
        sys.argv = ["pytest-chroma-cfg-pad", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            chroma_model = pytest.importorskip("comfy.ldm.chroma.model")
        finally:
            sys.argv = original_argv
        yield ops, chroma_model
    finally:
        for name in [name for name in sys.modules if _is_comfy_module(name)]:
            sys.modules.pop(name, None)
        sys.modules.update(preserved_modules)
        sys.path[:] = original_path


def _tiny_chroma(comfy_chroma_classes):
    ops, chroma_model = comfy_chroma_classes
    torch.manual_seed(0)
    model = chroma_model.Chroma(
        image_model=None,
        in_channels=RAW_CHANNELS * PATCH * PATCH,
        out_channels=RAW_CHANNELS * PATCH * PATCH,
        context_in_dim=CONTEXT_DIM, hidden_size=HIDDEN, mlp_ratio=2.0,
        num_heads=2, depth=2, depth_single_blocks=2, axes_dim=[2, 2, 4],
        theta=10000, patch_size=PATCH, qkv_bias=True, in_dim=64,
        out_dim=HIDDEN, hidden_dim=32, n_layers=1, txt_ids_dims=[2, 2, 4],
        vec_in_dim=None, dtype=torch.float32, device="cpu",
        operations=ops.disable_weight_init,
    )
    # comfy initializes state-dict parameters with torch.empty (ComfyUI's AGENTS.md);
    # a real load fills them from a checkpoint, so a toy run fills them here.
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return model


def _image_keys(h: int, w: int) -> int:
    return (h // PATCH) * (w // PATCH)


def _equalizer_pad_and_mask(tensor: torch.Tensor, rows: int, longest: int, image_keys: int):
    """The construction `equalize_cond_lengths` builds for rule 'pad+mask'
    (actor/sampling.py): zero-pad the text to `longest`, then an additive bias
    over the joint [text, image] keys. The caller picks `longest`; the
    equalizer aligns the joint extent to a multiple of 8."""
    padded = F.pad(tensor, (0, 0, 0, longest - rows))
    key_pos = torch.arange(longest + image_keys)
    appended = (key_pos >= rows) & (key_pos < longest)
    mask = (appended.to(tensor.dtype) * torch.finfo(tensor.dtype).min).reshape(1, 1, -1)
    return padded, mask


def _render_inputs(batch: int = 1, h: int = 6, w: int = 6):
    return (torch.randn(batch, RAW_CHANNELS, h, w), torch.rand(batch), torch.rand(batch))


def _install_spy(model):
    """Replace `model._forward` with one that records what it was called
    with, then delegates to the forward captured right before the swap.

    Output equality alone cannot catch a hook that let the padded, masked
    call through unchanged: on CPU both sides reach the same attention
    backend, so `stock(padded, mask)` already equals `stock(unpadded)` bit
    for bit (no kernel to swap, unlike the GPU). The spy reads the arguments
    stock's own forward receives, which is the claim: zero pad rows and no
    mask, not only a matching output.
    """
    real_forward = model._forward
    calls: list[tuple[int, object]] = []

    def spy(x, timestep, context, guidance, control=None, transformer_options=None, **kwargs):
        calls.append((int(context.shape[1]), kwargs.get("attention_mask")))
        return real_forward(x, timestep, context, guidance, control, transformer_options, **kwargs)

    model._forward = spy
    return calls


def test_the_padded_masked_short_prompt_matches_stock_on_the_unpadded_short_prompt(
    comfy_chroma_classes,
):
    """The claim, short leg: a rank holding the shorter prompt, padded and
    masked by the driver, must reproduce stock's unpadded call exactly."""
    model = _tiny_chroma(comfy_chroma_classes)
    x, timesteps, guidance = _render_inputs()
    image_keys = _image_keys(6, 6)
    long_rows, short_rows = 5, 3
    base_txt = torch.randn(1, long_rows, CONTEXT_DIM)
    short_txt = base_txt[:, :short_rows].clone()
    longest = -(-(long_rows + image_keys) // 8) * 8 - image_keys
    padded_short, mask_short = _equalizer_pad_and_mask(short_txt, short_rows, longest, image_keys)

    reference = model._forward(x, timesteps, short_txt, guidance, None, {})

    ChromaAdapter().inject_cfg_pad_forward(model)
    trimmed = model._forward(x, timesteps, padded_short, guidance, None, {},
                              attention_mask=mask_short)
    assert torch.equal(trimmed, reference)


def test_stock_actually_receives_the_trimmed_short_prompt_with_no_mask(comfy_chroma_classes):
    """The mechanism, not only the output (see `_install_spy`)."""
    model = _tiny_chroma(comfy_chroma_classes)
    x, timesteps, guidance = _render_inputs()
    image_keys = _image_keys(6, 6)
    long_rows, short_rows = 5, 3
    base_txt = torch.randn(1, long_rows, CONTEXT_DIM)
    short_txt = base_txt[:, :short_rows].clone()
    longest = -(-(long_rows + image_keys) // 8) * 8 - image_keys
    padded_short, mask_short = _equalizer_pad_and_mask(short_txt, short_rows, longest, image_keys)

    calls = _install_spy(model)
    ChromaAdapter().inject_cfg_pad_forward(model)
    model._forward(x, timesteps, padded_short, guidance, None, {}, attention_mask=mask_short)

    assert len(calls) == 1
    rows, mask = calls[0]
    assert rows == short_rows
    assert mask is None


def test_the_padded_masked_long_prompt_also_matches_stock(comfy_chroma_classes):
    """The claim, long leg: even the rank holding the longer prompt still
    gets rounded up to the joint 8-key alignment (sampling.py), so it also
    carries a real pad/mask the hook must trim back off exactly."""
    model = _tiny_chroma(comfy_chroma_classes)
    x, timesteps, guidance = _render_inputs()
    image_keys = _image_keys(6, 6)
    long_rows = 5
    long_txt = torch.randn(1, long_rows, CONTEXT_DIM)
    longest = -(-(long_rows + image_keys) // 8) * 8 - image_keys
    assert long_rows < longest, "the alignment rounding must still pad this leg"
    padded_long, mask_long = _equalizer_pad_and_mask(long_txt, long_rows, longest, image_keys)

    reference = model._forward(x, timesteps, long_txt, guidance, None, {})

    ChromaAdapter().inject_cfg_pad_forward(model)
    trimmed = model._forward(x, timesteps, padded_long, guidance, None, {},
                              attention_mask=mask_long)
    assert torch.equal(trimmed, reference)


def test_stock_actually_receives_the_trimmed_long_prompt_with_no_mask(comfy_chroma_classes):
    """The mechanism for the long leg, same reason as the short one above."""
    model = _tiny_chroma(comfy_chroma_classes)
    x, timesteps, guidance = _render_inputs()
    image_keys = _image_keys(6, 6)
    long_rows = 5
    long_txt = torch.randn(1, long_rows, CONTEXT_DIM)
    longest = -(-(long_rows + image_keys) // 8) * 8 - image_keys
    padded_long, mask_long = _equalizer_pad_and_mask(long_txt, long_rows, longest, image_keys)

    calls = _install_spy(model)
    ChromaAdapter().inject_cfg_pad_forward(model)
    model._forward(x, timesteps, padded_long, guidance, None, {}, attention_mask=mask_long)

    assert len(calls) == 1
    rows, mask = calls[0]
    assert rows == long_rows
    assert mask is None


def test_equal_length_prompts_never_build_a_mask_and_the_hook_stays_a_no_op(
    comfy_chroma_classes,
):
    """When lengths already match, `equalize_cond_lengths` returns the
    conditionings untouched and never builds a mask; the installed hook must
    not change the render in that case."""
    model = _tiny_chroma(comfy_chroma_classes)
    x, timesteps, guidance = _render_inputs()
    txt = torch.randn(1, 5, CONTEXT_DIM)

    reference = model._forward(x, timesteps, txt, guidance, None, {})

    ChromaAdapter().inject_cfg_pad_forward(model)
    out = model._forward(x, timesteps, txt, guidance, None, {})
    assert torch.equal(out, reference)


def test_a_batch_with_disagreeing_real_lengths_refuses_end_to_end(comfy_chroma_classes):
    """Two rows in one rank's batch (a conditioning-combine chunk, say) whose
    real lengths differ cannot be trimmed to one exact shape; the installed
    hook must refuse rather than attend a partial pad."""
    model = _tiny_chroma(comfy_chroma_classes)
    x, timesteps, guidance = _render_inputs(batch=2)
    image_keys = _image_keys(6, 6)
    longest = 8
    row_a, row_b = 3, 5
    txt_a = torch.randn(1, row_a, CONTEXT_DIM)
    txt_b = torch.randn(1, row_b, CONTEXT_DIM)
    padded_a, _ = _equalizer_pad_and_mask(txt_a, row_a, longest, image_keys)
    padded_b, _ = _equalizer_pad_and_mask(txt_b, row_b, longest, image_keys)
    context = torch.cat([padded_a, padded_b], dim=0)
    mask = _bias([row_a, row_b], longest, image_keys)

    ChromaAdapter().inject_cfg_pad_forward(model)
    with pytest.raises(UnsupportedModelError):
        model._forward(x, timesteps, context, guidance, None, {}, attention_mask=mask)
