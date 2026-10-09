"""CPU contracts for Chroma's full-row text projection scope.

The last test builds Chroma's real double block from ComfyUI's
disable_weight_init, manual_cast, fp8_ops and mixed_precision_ops Linear
families and pins which ones the projection patch selects. ComfyUI defines
cublas_ops only when the optional cublas_ops package imports, so the test
checks that family only then. It skips where ComfyUI is not importable; the
comfy-canary job runs this file against comfy master. comfy.cli_args parses
argv on import, so the fixture forces ``--cpu`` the way
tests/test_chroma_cfg_pad_forward.py does.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import chroma_text
from dgx_monarch.adapters.base import pad_seq_to_multiple


def _block(hidden=4):
    return SimpleNamespace(
        txt_attn=SimpleNamespace(qkv=torch.nn.Linear(hidden, hidden, dtype=torch.float64),
                                 proj=torch.nn.Linear(hidden, hidden, dtype=torch.float64)),
        txt_mlp=[torch.nn.Linear(hidden, hidden, dtype=torch.float64), None,
                 torch.nn.Linear(hidden, hidden, dtype=torch.float64)],
    )


def _run(monkeypatch, block, source, world, image_rows):
    padded, text_rows = pad_seq_to_multiple(source, world, dim=1)
    chunks = list(torch.chunk(padded, world, dim=1))
    outputs = {"proj": [], "mlp": []}
    for rank in range(world):
        def gather(value, length, dim=1, rank=rank):
            assert length == text_rows and dim == 1 and torch.equal(value, chunks[rank])
            return padded[:, :length]
        monkeypatch.setattr(chroma_text, "sp_gather", gather)
        monkeypatch.setattr(chroma_text, "sp_rank", lambda rank=rank: rank)
        monkeypatch.setattr(chroma_text, "sp_world", lambda: world)
        with chroma_text.full_row_text_projections(block, text_rows, image_rows,
                                                   pure_ulysses=True):
            outputs["proj"].append(block.txt_attn.proj(chunks[rank]))
            outputs["mlp"].append(block.txt_mlp[2](chunks[rank]))
    return {name: torch.cat(parts, dim=1)[:, :text_rows] for name, parts in outputs.items()}


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("rows", [1, 28, 44, 45, 100])
def test_full_rows_match_stock_view_and_mlp_is_contiguous(monkeypatch, batch, world, rows):
    torch.manual_seed(batch * 1000 + rows)
    block, image_rows = _block(), 17
    source = torch.randn(batch, rows, 4, dtype=torch.float64)
    proj_seen, mlp_seen = [], []
    proj_original, mlp_original = block.txt_attn.proj.forward, block.txt_mlp[2].forward
    block.txt_attn.proj.forward = lambda value: (proj_seen.append((value.stride(), value.is_contiguous()))
                                                  or proj_original(value))
    block.txt_mlp[2].forward = lambda value: (mlp_seen.append((value.stride(), value.is_contiguous()))
                                               or mlp_original(value))
    actual = _run(monkeypatch, block, source, world=world, image_rows=image_rows)
    stock_joint = torch.empty(batch, rows + image_rows, 4, dtype=torch.float64)
    stock_view = stock_joint[:, :rows]
    stock_view.copy_(source)
    assert torch.equal(actual["proj"], proj_original(stock_view))
    assert torch.equal(actual["mlp"], mlp_original(source))
    assert proj_seen and all(stride == stock_view.stride() and contiguous is stock_view.is_contiguous()
                             for stride, contiguous in proj_seen)
    assert mlp_seen and all(contiguous for _, contiguous in mlp_seen)
    if batch == 2:
        assert all(stride == (rows * 4, 4, 1) for stride, _ in mlp_seen)


def test_disabled_path_and_non_linear_modules_stay_unpatched():
    block = _block()
    originals = (block.txt_attn.qkv.forward, block.txt_attn.proj.forward,
                 block.txt_mlp[0].forward, block.txt_mlp[2].forward)
    with chroma_text.full_row_text_projections(block, 7, 9, pure_ulysses=False):
        assert (block.txt_attn.qkv.forward, block.txt_attn.proj.forward,
                block.txt_mlp[0].forward, block.txt_mlp[2].forward) == originals
    # Stands in for comfy's mixed_precision_ops Linear, which subclasses
    # torch.nn.Module rather than torch.nn.Linear.
    class MixedPrecisionLinear(torch.nn.Module):
        def forward(self, value): return value
    block.txt_attn.proj, block.txt_mlp[2] = MixedPrecisionLinear(), MixedPrecisionLinear()
    originals = (block.txt_attn.proj.forward, block.txt_mlp[2].forward)
    with chroma_text.full_row_text_projections(block, 7, 9, pure_ulysses=True):
        assert (block.txt_attn.proj.forward, block.txt_mlp[2].forward) == originals


def test_linear_subclass_with_plain_fp8_weights_takes_full_rows(monkeypatch):
    seen_rows = []
    # Like comfy's manual_cast Linear: an nn.Linear subclass that casts its
    # stored fp8 weight to the input dtype on every call.
    class CastLinear(torch.nn.Linear):
        def forward(self, value):
            seen_rows.append(value.shape[1])
            return torch.nn.functional.linear(
                value, self.weight.to(value.dtype), self.bias.to(value.dtype))
    def fp8(module):
        cast = CastLinear(4, 4, dtype=torch.float64)
        cast.load_state_dict(module.state_dict())
        cast.weight = torch.nn.Parameter(cast.weight.detach().to(torch.float8_e4m3fn),
                                         requires_grad=False)
        return cast
    block = _block()
    block.txt_attn.proj, block.txt_mlp[2] = fp8(block.txt_attn.proj), fp8(block.txt_mlp[2])
    selected = chroma_text._modules(block)
    assert [module for module, _ in selected] == [block.txt_attn.proj, block.txt_mlp[2]]
    torch.manual_seed(5)
    rows, image_rows = 5, 3
    source = torch.randn(1, rows, 4, dtype=torch.float64)
    actual = _run(monkeypatch, block, source, world=2, image_rows=image_rows)
    assert seen_rows == [rows] * 4
    stock_view = torch.empty(1, rows + image_rows, 4, dtype=torch.float64)[:, :rows]
    stock_view.copy_(source)
    assert torch.equal(actual["proj"], block.txt_attn.proj(stock_view))
    assert torch.equal(actual["mlp"], block.txt_mlp[2](source))
    assert "forward" not in vars(block.txt_attn.proj)
    assert "forward" not in vars(block.txt_mlp[2])


def test_restores_for_block_error_and_partial_install_failure():
    block = _block()
    original = block.txt_attn.proj.forward
    with pytest.raises(RuntimeError):
        with chroma_text.full_row_text_projections(block, 3, 5, pure_ulysses=True):
            raise RuntimeError("block failure")
    assert block.txt_attn.proj.forward == original
    assert "forward" not in vars(block.txt_attn.proj)
    class FailingLinear(torch.nn.Linear):
        armed = False
        def __setattr__(self, name, value):
            if name == "forward" and self.armed:
                raise RuntimeError("assignment failure")
            super().__setattr__(name, value)
    failed = FailingLinear(4, 4, dtype=torch.float64)
    failed.armed = True
    block.txt_mlp[2] = failed
    with pytest.raises(RuntimeError, match="assignment failure"):
        with chroma_text.full_row_text_projections(block, 3, 5, pure_ulysses=True):
            pass
    assert block.txt_attn.proj.forward == original
    assert "forward" not in vars(block.txt_attn.proj)


def test_restore_tolerates_a_forward_the_block_already_removed():
    block = _block()
    # The attention projection restores last, so its missing forward would
    # replace the block's own error with AttributeError.
    with pytest.raises(RuntimeError, match="block failure"):
        with chroma_text.full_row_text_projections(block, 3, 5, pure_ulysses=True):
            del block.txt_attn.proj.forward
            raise RuntimeError("block failure")
    assert "forward" not in vars(block.txt_attn.proj)
    assert "forward" not in vars(block.txt_mlp[2])
    # The MLP projection restores first; a missing forward there must not
    # stop the attention projection's restore.
    with chroma_text.full_row_text_projections(block, 3, 5, pure_ulysses=True):
        del block.txt_mlp[2].forward
    assert "forward" not in vars(block.txt_attn.proj)
    assert "forward" not in vars(block.txt_mlp[2])


def test_restore_reveals_class_forward_and_preserves_instance_override(monkeypatch):
    class RebindLinear(torch.nn.Linear):
        pass
    block = _block()
    block.txt_attn.proj = RebindLinear(4, 4, dtype=torch.float64)
    with chroma_text.full_row_text_projections(block, 3, 5, pure_ulysses=True):
        pass
    assert "forward" not in vars(block.txt_attn.proj)
    monkeypatch.setattr(RebindLinear, "forward", lambda _self, value: value + 1)
    assert torch.equal(block.txt_attn.proj(torch.zeros(1, 3, 4)), torch.ones(1, 3, 4))
    def override(value):
        return value + 2
    block.txt_attn.proj.forward = override
    with chroma_text.full_row_text_projections(block, 3, 5, pure_ulysses=True):
        pass
    assert vars(block.txt_attn.proj)["forward"] is override


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy_ops_and_chroma():
    """Import real `comfy.ops` / `comfy.ldm.chroma.model`, then undo it.

    Same snapshot-and-restore as tests/test_chroma_cfg_pad_forward.py, so a
    later test that probes "is comfy importable" gets the answer it gets alone.
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
        sys.argv = ["pytest-chroma-text-projections", "--cpu"]
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


def test_real_comfy_linear_families_select_as_documented(comfy_ops_and_chroma):
    """bf16 and plain fp8 Linear families are patched; mixed precision is not.

    Plain fp8 checkpoints without quantization metadata load through
    manual_cast or fp8_ops; checkpoints with metadata load through
    mixed_precision_ops, whose Linear is a torch.nn.Module.
    """
    ops, chroma_model = comfy_ops_and_chroma

    def double_block(operations):
        # Chroma's own construction: modulation comes from its approximator.
        return chroma_model.DoubleStreamBlock(
            8, 2, mlp_ratio=2.0, qkv_bias=True, modulation=False,
            dtype=torch.float32, device="cpu", operations=operations)

    for name in ("disable_weight_init", "manual_cast", "fp8_ops"):
        block = double_block(getattr(ops, name))
        selected = chroma_text._modules(block)
        assert len(selected) == 2, name
        assert selected[0][0] is block.txt_attn.proj and selected[0][1] is True, name
        assert selected[1][0] is block.txt_mlp[2] and selected[1][1] is False, name
    # Defined only when the optional cublas_ops package imports.
    if hasattr(ops, "cublas_ops"):
        assert issubclass(ops.cublas_ops.Linear, torch.nn.Linear)
    mixed = double_block(ops.mixed_precision_ops({}, torch.float32))
    assert not isinstance(mixed.txt_attn.proj, torch.nn.Linear)
    assert chroma_text._modules(mixed) == ()
