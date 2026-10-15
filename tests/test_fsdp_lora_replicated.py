"""LoRA under FSDP on a census-replicated parameter: actor/fsdp_lora.py bakes it whole.

The FSDP wrap keeps every module a family runs outside its diffusion forward
replicated: ``adapters/fsdp_islands.auxiliary_module_parameters`` names their
parameters, the wrap passes them as ``ignored_params``, and they stay plain
tensors. Before this path a LoRA key on one raised ``UnbakeError`` on every
``*+fsdp`` topology (H3's ``token_refiner``, LTXAV's connectors, Anima's
``llm_adapter``). It now bakes as a world-1 shard: the same pristine read and
the same comfy merge, written whole on every rank.

The stand-in model is a comfy ``model_base`` class named ``MiniMaxH3`` (the
census reads the class name) holding one sharded block Linear and one
``token_refiner`` Linear. The FSDP cases follow tests/test_fsdp_lora.py: real
FSDP2 on a one-rank gloo group with comfy's math stubbed, plus a rank 1 of 2
case on a synthetic shard, the only shape in which a replicated key and a
sharded key take different rows. The real-ComfyUI case at the end checks bit
equality with comfy's own ``patch_weight_to_device`` where comfy imports.
"""
from __future__ import annotations

import copy
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.actor import fsdp_lora
from dgx_monarch.actor.unbake import UnbakeError
from dgx_monarch.adapters.fsdp_islands import auxiliary_module_parameters
from dgx_monarch.refusal import RefusalClass, RefusalTag, parse_leading_refusal_tag
from module_location_helpers import from_checkout
from test_fsdp_lora import _Active, _FakeQuantParam, _install_comfy_stub, _write_safetensors

SHARDED = "diffusion_model.blocks.0.lin.weight"
REPLICATED = "diffusion_model.token_refiner.weight"


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.lin(x)


class _DiffusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block()])
        self.token_refiner = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class MiniMaxH3(torch.nn.Module):
    """The comfy model_base stand-in: the census keys on this class name."""

    def __init__(self):
        super().__init__()
        self.diffusion_model = _DiffusionModel()


def _pristine(model: MiniMaxH3) -> dict[str, torch.Tensor]:
    """Distinct rows per key, so a row offset or a swapped key shows."""
    weights = {
        SHARDED: torch.arange(16, dtype=torch.float32).reshape(4, 4),
        REPLICATED: torch.arange(16, 32, dtype=torch.float32).reshape(4, 4) / 8,
    }
    out = {}
    for key, value in weights.items():
        value = value.to(torch.bfloat16)
        with torch.no_grad():
            fsdp_lora.live_tensor(model, key).copy_(value)
        out[key] = value
    return out


def _write_file(tmp_path, pristine: dict[str, torch.Tensor]) -> str:
    path = tmp_path / "h3.safetensors"
    _write_safetensors(path, {key[len("diffusion_model."):]: value
                              for key, value in pristine.items()})
    return str(path)


def _wrap_one_rank(tmp_path, model: MiniMaxH3) -> MiniMaxH3:
    """The wrap adapters/fsdp.py makes: blocks, then the root, census ignored."""
    from torch.distributed.device_mesh import init_device_mesh

    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    mesh = init_device_mesh("cpu", (1,))
    dm = model.diffusion_model
    for p in dm.parameters():
        p.requires_grad = False
    ignored = auxiliary_module_parameters(dm, model)
    for block in dm.blocks:
        torch.distributed.fsdp.fully_shard(block, mesh=mesh, ignored_params=ignored)
    torch.distributed.fsdp.fully_shard(
        dm, mesh=mesh, reshard_after_forward=True, ignored_params=ignored)
    return model


def _expected(twin: MiniMaxH3, patches: dict) -> dict[str, torch.Tensor]:
    """The merge on an unwrapped twin through the stub's patch_weight_to_device."""
    reference = _Active(twin, patches)
    return {key: reference.patch_weight_to_device(key, return_weight=True) for key in patches}


def test_the_census_names_the_token_refiner_and_not_the_block():
    model = MiniMaxH3()
    replicated = fsdp_lora.replicated_parameter_ids(model)
    assert replicated == {id(model.diffusion_model.token_refiner.weight)}
    assert fsdp_lora.replicated_parameter_ids(torch.nn.Linear(2, 2)) == frozenset()


def test_a_replicated_key_captures_bakes_verifies_and_restores_whole(tmp_path, monkeypatch):
    import torch.distributed.fsdp

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    _install_comfy_stub(monkeypatch)
    model = MiniMaxH3()
    pristine = _pristine(model)
    path = _write_file(tmp_path, pristine)
    patches = {SHARDED: ["patch"], REPLICATED: ["patch"]}
    expected = _expected(copy.deepcopy(model), patches)
    try:
        model = _wrap_one_rank(tmp_path, model)
        refiner = model.diffusion_model.token_refiner.weight
        sharded = model.diffusion_model.blocks[0].lin.weight
        assert type(refiner) is torch.nn.Parameter
        assert type(sharded).__name__ == "DTensor"
        x = torch.ones(2, 4, dtype=torch.bfloat16)
        active = _Active(model, patches)
        store = SimpleNamespace(swap_verify=-1)  # the ambient verify on every key
        record = fsdp_lora.apply_stack(store, active, path, None)
        assert set(record.mapped) == {SHARDED, REPLICATED}
        assert active.patches == {}
        # The same parameter objects, written in place: whole on the census
        # module, through the local view on the shard.
        assert model.diffusion_model.token_refiner.weight is refiner
        assert model.diffusion_model.blocks[0].lin.weight is sharded
        assert torch.equal(refiner, expected[REPLICATED])
        assert torch.equal(fsdp_lora.local_shard(sharded), expected[SHARDED])
        with torch.no_grad():
            assert torch.equal(model.diffusion_model.token_refiner(x),
                               x @ expected[REPLICATED].t())
            assert torch.equal(model.diffusion_model(x), x @ expected[SHARDED].t())
        # A stack that drops both keys restores both from the file.
        record2 = fsdp_lora.apply_stack(store, _Active(model, {}), path, record)
        assert record2.mapped == {}
        assert torch.equal(refiner, pristine[REPLICATED])
        assert torch.equal(fsdp_lora.local_shard(sharded), pristine[SHARDED])
    finally:
        torch.distributed.destroy_process_group()


def _rank_one_of_two(model: MiniMaxH3, pristine: dict[str, torch.Tensor]) -> torch.Tensor:
    """Make the block weight a rank-1-of-2 shard: global shape, rows 2..4 local."""
    lin = model.diffusion_model.blocks[0].lin
    shard = torch.nn.Parameter(pristine[SHARDED].clone(), requires_grad=False)
    shard._local_tensor = pristine[SHARDED][2:4].clone()
    lin.weight = shard
    return shard


def test_rank_one_of_two_writes_its_rows_and_the_replicated_key_whole(tmp_path, monkeypatch):
    """At world 1 the two geometries coincide; at rank 1 of 2 they cannot."""
    _install_comfy_stub(monkeypatch)
    monkeypatch.setattr(fsdp_lora, "mesh_rank_world", lambda: (1, 2))
    model = MiniMaxH3()
    pristine = _pristine(model)
    path = _write_file(tmp_path, pristine)
    patches = {SHARDED: ["patch"], REPLICATED: ["patch"]}
    expected = _expected(copy.deepcopy(model), patches)
    shard = _rank_one_of_two(model, pristine)
    refiner = model.diffusion_model.token_refiner.weight
    store = SimpleNamespace(swap_verify=-1)
    record = fsdp_lora.apply_stack(store, _Active(model, patches), path, None)
    assert set(record.mapped) == {SHARDED, REPLICATED}
    assert torch.equal(shard._local_tensor, expected[SHARDED][2:4])
    assert torch.equal(refiner, expected[REPLICATED])
    fsdp_lora.apply_stack(store, _Active(model, {}), path, record)
    assert torch.equal(shard._local_tensor, pristine[SHARDED][2:4])
    assert torch.equal(refiner, pristine[REPLICATED])


def test_a_plain_tensor_outside_the_census_still_raises(tmp_path, monkeypatch):
    """An unexpectedly unsharded parameter stays an error, not a whole bake."""
    _install_comfy_stub(monkeypatch)
    monkeypatch.setattr(fsdp_lora, "mesh_rank_world", lambda: (0, 2))
    model = MiniMaxH3()
    path = _write_file(tmp_path, _pristine(model))
    with pytest.raises(UnbakeError, match="not FSDP-sharded"):
        fsdp_lora.apply_stack(SimpleNamespace(swap_verify=0),
                              _Active(model, {SHARDED: ["patch"]}), path, None)


def test_a_quantized_key_on_a_census_module_still_refuses_class_p(tmp_path, monkeypatch):
    _install_comfy_stub(monkeypatch)
    model = MiniMaxH3()
    path = _write_file(tmp_path, _pristine(model))
    model.diffusion_model.token_refiner = torch.nn.Module()
    model.diffusion_model.token_refiner.weight = _FakeQuantParam()
    with pytest.raises(UnbakeError, match="LoRA on quantized shards") as raised:
        fsdp_lora.apply_stack(SimpleNamespace(swap_verify=0),
                              _Active(model, {REPLICATED: ["patch"]}), path, None)
    assert parse_leading_refusal_tag(str(raised.value)) == RefusalTag(
        refusal_class=RefusalClass.PHYSICS, guard=None, waivable=False)


# Real comfy, CPU only. Keep it last in the file: the conftest comfy guard
# checks the namespace once this module's fixtures are torn down.
def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def real_comfy():
    """Import real comfy.lora and comfy.model_patcher on CPU, then undo it.

    A checkout named by COMFYUI_DIR or COMFY_DIR must import; otherwise the
    test skips where no ComfyUI is present. comfy.cli_args parses argv on
    import, so ``--cpu`` is forced.
    """
    named = os.environ.get("COMFYUI_DIR") or os.environ.get("COMFY_DIR")
    comfy_dir = os.path.abspath(named or os.path.expanduser("~/ComfyUI"))
    preserved = {name: module for name, module in sys.modules.items() if _is_comfy_module(name)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        sys.argv = ["pytest-fsdp-lora-replicated", "--cpu"]
        try:
            if named is None:
                pytest.importorskip(
                    "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            import comfy.options

            comfy.options.enable_args_parsing()
            import comfy.lora
            import comfy.model_patcher
        finally:
            sys.argv = original_argv
        yield SimpleNamespace(lora=comfy.lora, model_patcher=comfy.model_patcher)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


def test_the_replicated_bake_equals_comfys_own_merge(real_comfy, tmp_path):
    """Both keys, real LoRA patches, bf16 with comfy's key-seeded rounding: the
    FSDP bake equals ModelPatcher.patch_weight_to_device on an unwrapped twin."""
    import torch.distributed.fsdp

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    torch.manual_seed(0)
    model = MiniMaxH3()
    pristine = _pristine(model)
    path = _write_file(tmp_path, pristine)
    lora = {}
    for unit in ("block", "refiner"):
        lora[f"{unit}.lora_up.weight"] = torch.randn(4, 2)
        lora[f"{unit}.lora_down.weight"] = torch.randn(2, 4)
        lora[f"{unit}.alpha"] = torch.tensor(2.0)
    patches = real_comfy.lora.load_lora(lora, {"block": SHARDED, "refiner": REPLICATED})
    assert set(patches) == {SHARDED, REPLICATED}
    cpu = torch.device("cpu")
    reference = real_comfy.model_patcher.ModelPatcher(copy.deepcopy(model), cpu, cpu)
    reference.add_patches(patches, 0.7)
    expected = {key: reference.patch_weight_to_device(key, device_to=cpu, return_weight=True)
                for key in (SHARDED, REPLICATED)}
    assert not torch.equal(expected[REPLICATED], pristine[REPLICATED])
    active = real_comfy.model_patcher.ModelPatcher(model, cpu, cpu)
    active.add_patches(patches, 0.7)
    try:
        _wrap_one_rank(tmp_path, model)
        store = SimpleNamespace(swap_verify=-1)  # comfy's own verify on every key
        fsdp_lora.apply_stack(store, active, path, None)
        refiner = model.diffusion_model.token_refiner.weight
        assert type(refiner) is torch.nn.Parameter
        assert refiner.dtype == expected[REPLICATED].dtype == torch.bfloat16
        assert torch.equal(refiner, expected[REPLICATED])
        sharded = fsdp_lora.local_shard(model.diffusion_model.blocks[0].lin.weight)
        assert torch.equal(sharded, expected[SHARDED])
    finally:
        torch.distributed.destroy_process_group()
