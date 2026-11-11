"""Real ComfyUI: slab residency hands ComfyAttention the pin a stock load hands it,
and ComfyUI's own load path puts the checkpoint's pin back before a forward.

On one rank a checkpoint's own attention pins now run as stock runs them, so
slab residency with pins runs for the first time (docs/ADAPTERS.md, attention
pins). The slab serves the checkpoint's state dict from its arena and keeps
each ``<block>.comfy_attention.config`` uint8 tensor on the host
(``actor/slab.py``), since ComfyAttention decodes it with ``Tensor.numpy()``.
This loads one tiny pinned file twice into real ``ComfyAttention`` modules,
once through ComfyUI's own ``load_torch_file`` and once through the slab's
``load_torch_file`` hook (``comfy_bridge.slab_load``), with the kernels reported
available so ComfyUI binds them as it does on GB10, and holds the two equal: the
config, the bound function, its tau, and what ``attention_pins`` reads from
each. The slab state dict is built as on GB10, where CUDA is available, and the
test records each tensor's placement: the pin configs stay on the host and a
weight does not. It calls ``load_state_dict`` itself, so it does not run the
slab's ``BaseModel.load_model_weights`` hook (``assign=True``), which casts only
parameters and buffers and leaves the uint8 configs as they are.

The one-GPU rule (``attention_pins_source.checkpoint_only``) also trusts ComfyUI
to put a staged LoRA's ``comfy_attention`` module back to the checkpoint's before
the next clone's forward: ``load_models_gpu`` detaches the old clone without
unpatching, and the new clone's ``partially_load`` runs ``unpatch_model``, which
restores the shared ``object_patches_backup``, then ``patch_model``. The second
test drives a real ``ModelPatcher`` through that path, from a LoRA clone back to
the checkpoint clone and back again, with the model already loaded, and checks
the module a forward sees and what the guard reads. An upstream change of that
order turns it red.

What a hardware leg must still show: a slab-resident pinned file on GB10 binds
and runs the kernel. Skipped where ComfyUI is not importable; the comfy-canary
job runs it against master and the newest release.
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import attention_pins
from module_location_helpers import from_checkout

PINS = {"blocks.0.attn.comfy_attention.config": [{"attention": "comfy_kitchen_int8"}],
        "blocks.1.attn.comfy_attention.config": [{"attention": "comfy_kitchen_sol", "tau": 0.5}]}


def _comfy_dir() -> str:
    value = os.environ.get("COMFYUI_DIR") or os.environ.get("COMFY_DIR") or "~/ComfyUI"
    return os.path.abspath(os.path.expanduser(value))


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture
def comfy():
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
        sys.argv = ["pytest-attention-pin-slab", "--cpu"]
        try:
            options = pytest.importorskip("comfy.options", reason="no ComfyUI checkout on sys.path")
            options.enable_args_parsing()
            utils = pytest.importorskip("comfy.utils")
            pytest.importorskip("comfy.model_base")
            attention = pytest.importorskip("comfy.ldm.modules.attention")
            patcher = pytest.importorskip("comfy.model_patcher")
            management = pytest.importorskip("comfy.model_management")
        finally:
            sys.argv = original_argv
        if not hasattr(attention, "ComfyAttention"):
            pytest.skip("this ComfyUI has no per-block attention pins")
        yield SimpleNamespace(utils=utils, attention=attention, patcher=patcher, management=management)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


def _tree(comfy) -> torch.nn.Module:
    root = torch.nn.Module()
    root.blocks = torch.nn.ModuleList()
    for _ in range(2):
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.comfy_attention = comfy.attention.ComfyAttention()
        block.attn.qkv = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        root.blocks.append(block)
    return root


def _write(path) -> str:
    from safetensors.torch import save_file

    state = {name: torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)
             for name, config in PINS.items()}
    for index in range(2):
        state[f"blocks.{index}.attn.qkv.weight"] = torch.full((4, 4), index + 1.0, dtype=torch.bfloat16)
    save_file(state, str(path))
    return str(path)


def _bound(tree) -> list[tuple]:
    out = []
    for block in tree.blocks:
        module = block.attn.comfy_attention
        function = module.function
        target = getattr(function, "func", function)
        out.append((module.config, module.method, getattr(target, "__name__", None),
                    dict(getattr(function, "keywords", None) or {})))
    return out


def test_a_slab_load_binds_each_pin_as_a_stock_load_does(comfy, tmp_path, monkeypatch):
    import comfy_kitchen

    from dgx_monarch.actor.comfy_bridge import slab_load

    # The kernels run on GB10, not on this CPU: report them available so
    # ComfyUI binds them, which is the branch a Worker load takes.
    monkeypatch.setattr(comfy.attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", True)
    monkeypatch.setattr(comfy_kitchen, "int8_attention_is_available", lambda *a, **k: True)
    monkeypatch.setattr(comfy_kitchen, "sol_attn_is_available", lambda *a, **k: True)
    path = _write(tmp_path / "pinned.safetensors")
    from dgx_monarch.actor import slab as slab_module

    # A CPU runner has no CUDA, so every slab tensor is a host tensor there. Build
    # the state dict as on GB10 and record where each tensor would be placed.
    placed: dict[str, bool] = {}
    real_state_dict = slab_module.WeightSlab.state_dict
    real_tensor = slab_module.WeightSlab._tensor

    def state_dict_as_on_gb10(self):
        names = {id(region): name for name, region in self.regions.items()}

        def tensor(region, *, cuda):
            placed[names[id(region)]] = cuda
            return real_tensor(self, region, cuda=False)

        with monkeypatch.context() as patch:
            patch.setattr(slab_module.torch.cuda, "is_available", lambda: True)
            patch.setattr(self, "_tensor", tensor)
            return real_state_dict(self)

    monkeypatch.setattr(slab_module.WeightSlab, "state_dict", state_dict_as_on_gb10)

    stock = _tree(comfy)
    stock.load_state_dict(comfy.utils.load_torch_file(path), strict=False)
    slabbed = _tree(comfy)
    with slab_load(path) as slab:
        state = comfy.utils.load_torch_file(path)
        assert slab.contains(state["blocks.0.attn.qkv.weight"].data_ptr()), "not slab-backed"
        for name in PINS:
            assert placed[name] is False, f"{name} left the host"
        assert placed["blocks.0.attn.qkv.weight"] is True
        slabbed.load_state_dict(state, strict=False)
    del state
    slab.close()

    assert _bound(slabbed) == _bound(stock)
    assert _bound(stock) == [
        (PINS["blocks.0.attn.comfy_attention.config"], "comfy_kitchen_int8",
         "attention_comfy_kitchen_int8", {}),
        (PINS["blocks.1.attn.comfy_attention.config"], "comfy_kitchen_sol",
         "attention_comfy_kitchen_sol", {"tau": 0.5}),
    ]
    # What the record and the one-rank rule read off each tree is the same too.
    read = {name: attention_pins.scan(tree) for name, tree in (("stock", stock), ("slab", slabbed))}
    assert read["slab"] == read["stock"]
    assert {pin.label for pin in read["stock"].values()} == {
        "comfy_kitchen_int8", "comfy_kitchen_sol tau=0.5"}
    assert all(pin.bound for pin in read["stock"].values())
    for tree in (stock, slabbed):
        patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=tree), object_patches={})
        attention_pins.scan_loaded(patcher)
        summary = attention_pins.summarize([patcher], sequence_parallel=False)
        assert summary is not None and summary.checkpoint_only
        attention_pins.assert_waived(summary, {}, alone=True)  # one rank: nothing asked


class _Seen(torch.nn.Module):
    """A block whose forward records the ``comfy_attention`` module it would hand
    ``optimized_attention`` as ``preferred_attention``, as ComfyUI's blocks do."""

    def __init__(self, comfy, seen: list):
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.comfy_attention = comfy.attention.ComfyAttention()
        self.attn.qkv = torch.nn.Linear(4, 4, bias=False)
        self.seen = seen

    def forward(self, x):
        self.seen.append(self.attn.comfy_attention)
        return self.attn.qkv(x)


class _Model(torch.nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def test_comfyui_puts_the_checkpoints_pin_back_before_the_next_clones_forward(comfy, monkeypatch):
    import comfy_kitchen

    monkeypatch.setattr(comfy.attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", True)
    monkeypatch.setattr(comfy_kitchen, "int8_attention_is_available", lambda *a, **k: True)
    monkeypatch.setattr(comfy_kitchen, "sol_attn_is_available", lambda *a, **k: True)
    seen: list = []
    dm = _Model([_Seen(comfy, seen) for _ in range(2)])
    dm.load_state_dict({"blocks.0.attn.comfy_attention.config": torch.tensor(
        list(json.dumps([{"attention": "comfy_kitchen_int8"}]).encode()), dtype=torch.uint8),
        **{f"blocks.{i}.attn.qkv.weight": torch.ones(4, 4) for i in range(2)}}, strict=False)
    root = torch.nn.Module()
    root.diffusion_model = dm
    cpu = torch.device("cpu")
    base = comfy.patcher.ModelPatcher(root, load_device=cpu, offload_device=cpu)
    attention_pins.scan_loaded(base)  # what fresh_load does for every load
    header = dm.blocks[0].attn.comfy_attention
    path = "diffusion_model.blocks.0.attn.comfy_attention"
    staged = torch.tensor(list(json.dumps([{"attention": "comfy_kitchen_sol", "tau": 0.5}]).encode()),
                          dtype=torch.uint8)
    lora = base.clone()
    assert lora.add_patches({f"{path}.config": ("config", (staged,))})
    own = base.clone()

    def forward_sees(clone):
        """Load the clone as a sample does, run a forward, return what block 0 saw."""
        comfy.management.load_models_gpu([clone])
        seen.clear()
        clone.model.diffusion_model(torch.ones(1, 4))
        return seen[0]

    def asks(clone) -> bool:
        summary = attention_pins.summarize([clone], sequence_parallel=False)
        assert summary is not None
        try:
            attention_pins.assert_waived(summary, {}, alone=True)
        except Exception as exc:
            assert attention_pins.SOL_ATTN_PIN_GUARD in str(exc)
            return True
        return False

    try:
        for _round in range(2):  # the second round starts with the model already loaded
            module = forward_sees(lora)
            assert module is not header
            assert attention_pins.block_pin(module).label == "comfy_kitchen_sol tau=0.5"
            assert asks(lora), "a LoRA's pin must keep class K in a world of one GPU"
            # Before ComfyUI unpatches, the LoRA's module is still in place; the guard
            # reads the backup ComfyUI will restore, so the checkpoint clone asks nothing.
            assert dm.blocks[0].attn.comfy_attention is not header
            assert not asks(own)
            module = forward_sees(own)
            assert module is header, "ComfyUI did not put the checkpoint's pin back"
            assert attention_pins.block_pin(module).label == "comfy_kitchen_int8"
            assert not asks(own)
            assert forward_sees(own) is header  # loading the loaded clone again changes nothing
    finally:
        comfy.management.unload_all_models()
