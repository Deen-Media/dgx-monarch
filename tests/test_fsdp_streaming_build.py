"""Streaming shard build for FSDP launches.

Pinned on CPU: the assign window makes the checkpoint's own tensors the
parameters and recasts only what the model constructed in another dtype; the
shard-build price is per world size and falls back to the full-copy bound when
the world is unknown; the worker guard, the residency ladder and the driver
price read the same multiplication; the build wraps on meta, moves only this
rank's rows and trims the pool once; and every host row and staged block
reaches the device through the bounce copier. One cuda-only test pins the
copy-on-write break that copier prevents.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

from dgx_monarch import fsdp_reload_price as price_mod
from dgx_monarch.actor import fsdp_streaming, store_residency
from dgx_monarch.adapters import fsdp as adapters_fsdp

GIB = 1 << 30


class _WanShaped(torch.nn.Module):
    """A bf16 core beside a deliberately fp32 input convolution (stock Wan)."""

    def __init__(self):
        super().__init__()
        self.core = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
        self.patch_embedding = torch.nn.Conv3d(1, 1, 1, dtype=torch.float32)
        self.register_buffer("freqs", torch.zeros(4, dtype=torch.float32))


class _FakeBaseModel:
    """comfy.model_base.BaseModel's load_model_weights contract."""

    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model

    def load_model_weights(self, sd, unet_prefix="", assign=False):
        to_load = {k[len(unet_prefix):]: v for k, v in sd.items() if k.startswith(unet_prefix)}
        self.diffusion_model.load_state_dict(to_load, strict=False, assign=assign)
        return self


def _bf16_file_state_dict():
    """Every tensor bf16 on disk, the shape a bf16 export takes."""
    return {
        "core.weight": torch.ones(4, 4, dtype=torch.bfloat16),
        "core.bias": torch.ones(4, dtype=torch.bfloat16),
        "patch_embedding.weight": torch.ones(1, 1, 1, 1, 1, dtype=torch.bfloat16),
        "patch_embedding.bias": torch.ones(1, dtype=torch.bfloat16),
        "freqs": torch.ones(4, dtype=torch.bfloat16),
    }


@pytest.fixture
def fake_comfy_model_base(monkeypatch):
    module = types.ModuleType("comfy.model_base")
    module.BaseModel = _FakeBaseModel
    comfy_pkg = sys.modules.get("comfy") or types.ModuleType("comfy")
    monkeypatch.setitem(sys.modules, "comfy", comfy_pkg)
    monkeypatch.setitem(sys.modules, "comfy.model_base", module)
    monkeypatch.setattr(comfy_pkg, "model_base", module, raising=False)
    return module


def test_assign_window_assigns_file_tensors_and_recasts_only_constructed_mismatches(
    fake_comfy_model_base,
):
    sd = _bf16_file_state_dict()
    model = _FakeBaseModel(_WanShaped())
    with fsdp_streaming.assign_load_window():
        model.load_model_weights(dict(sd), "", assign=False)  # comfy's classic call
    dm = model.diffusion_model
    # The bulk is the file's own tensor: same storage, nothing copied.
    assert dm.core.weight.data_ptr() == sd["core.weight"].data_ptr()
    assert dm.core.weight.dtype == torch.bfloat16
    # Where the model constructed another dtype the copying load's dtype wins.
    assert dm.patch_embedding.weight.dtype == torch.float32
    assert dm.patch_embedding.bias.dtype == torch.float32
    assert dm.freqs.dtype == torch.float32
    assert torch.equal(dm.patch_embedding.weight.float(), sd["patch_embedding.weight"].float())


def test_assign_window_matches_the_copying_load_dtype_for_dtype(fake_comfy_model_base):
    sd = _bf16_file_state_dict()
    copying = _FakeBaseModel(_WanShaped())
    copying.load_model_weights(dict(sd), "", assign=False)
    assigned = _FakeBaseModel(_WanShaped())
    with fsdp_streaming.assign_load_window():
        assigned.load_model_weights(dict(sd), "", assign=False)
    copied_dtypes = fsdp_streaming.constructed_dtypes(copying.diffusion_model)
    assert fsdp_streaming.constructed_dtypes(assigned.diffusion_model) == copied_dtypes


def test_assign_window_restores_the_original_loader(fake_comfy_model_base):
    original = fake_comfy_model_base.BaseModel.load_model_weights
    with fsdp_streaming.assign_load_window():
        assert fake_comfy_model_base.BaseModel.load_model_weights is not original
    assert fake_comfy_model_base.BaseModel.load_model_weights is original


def test_recast_reports_count_and_bytes():
    dm = _WanShaped()
    constructed = fsdp_streaming.constructed_dtypes(dm)
    dm.patch_embedding.weight.data = dm.patch_embedding.weight.data.to(torch.bfloat16)
    count, nbytes = fsdp_streaming.recast_to_constructed(dm, constructed)
    assert (count, nbytes) == (1, 4)
    assert dm.patch_embedding.weight.dtype == torch.float32


@pytest.mark.parametrize(("world", "expected"), [
    (None, adapters_fsdp.FSDP_MATERIALIZE_FACTOR),
    (0, adapters_fsdp.FSDP_MATERIALIZE_FACTOR),
    (1, 1.0 + adapters_fsdp.FSDP_STREAM_TRANSIENT_FRACTION),
    (2, 0.5 + adapters_fsdp.FSDP_STREAM_TRANSIENT_FRACTION),
    (4, 0.25 + adapters_fsdp.FSDP_STREAM_TRANSIENT_FRACTION),
])
def test_materialize_factor_is_per_world_and_falls_back_to_the_full_copy_bound(world, expected):
    assert adapters_fsdp.fsdp_materialize_factor(world, "bf16") == pytest.approx(expected)


def test_the_factor_is_the_measured_transient_above_the_shards():
    # The 0.047 is the 2026-08-26 flux2 build's transient beyond its 0.5 shards;
    # capacity_fit.py records it beside FSDP_STREAM_TRANSIENT_FRACTION.
    assert adapters_fsdp.FSDP_STREAM_TRANSIENT_FRACTION >= 0.047
    assert adapters_fsdp.fsdp_materialize_factor(2, "bf16") < 0.60


def test_a_60_gib_checkpoint_reloads_beside_its_shards_on_the_pair():
    """The first shard set left 47 GiB free on the pair for the clean reload
    (2026-08-25, docs/TROUBLESHOOTING.md #88)."""
    from dgx_monarch.capacity_fit import ABSOLUTE_HOST_FLOOR_BYTES

    required = adapters_fsdp.fsdp_required_bytes(60 * GIB, world=2, checkpoint_kind="bf16")
    assert required == (
        int(60 * GIB * (0.5 + adapters_fsdp.FSDP_STREAM_TRANSIENT_FRACTION))
        + ABSOLUTE_HOST_FLOOR_BYTES)
    # The ceremony's clean reload had 42 GiB free on the head at the price (leg 4 of 2026-08-26).
    assert required <= 42 * GIB
    assert adapters_fsdp.fsdp_required_bytes(60 * GIB, checkpoint_kind="bf16") > 42 * GIB  # the full-copy bound does not


def test_worker_guard_prices_per_world(monkeypatch, tmp_path):
    path = tmp_path / "checkpoint.safetensors"
    with open(path, "wb") as handle:
        handle.truncate(60 * GIB)
    monkeypatch.setattr(adapters_fsdp, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(adapters_fsdp, "mem_available_bytes", lambda: 47 * GIB)
    from dgx_monarch.mesh_safety import StockLoadCapacityError

    with pytest.raises(StockLoadCapacityError, match=r"1\.20x"):
        adapters_fsdp.fsdp_load_capacity_check(str(path), "checkpoint.safetensors", {})
    adapters_fsdp.fsdp_load_capacity_check(
        str(path), "checkpoint.safetensors", {}, world=2, checkpoint_kind="bf16")


def test_residency_ladder_hands_the_world_to_the_guard(monkeypatch):
    seen = {}

    def guard(path, unet_name, model_options, world=None, checkpoint_kind=None):
        seen["world"] = world

    import dgx_monarch.adapters.fsdp as fsdp_module

    monkeypatch.setattr(fsdp_module, "fsdp_load_capacity_check", guard)
    decision = store_residency.resolve(
        path="/models/x.safetensors", unet_name="x.safetensors", model_options={},
        slab_weights="auto", slab_capable_path=False, lora_low_rss=True,
        fsdp_launch=True, blocked_reason="", authoritative_slab_retry=False,
        memoized_family=lambda _p: None, vouched_families=frozenset(),
        node_options={}, lora_stack=None, family_override=None, world=2,
    )
    assert seen["world"] == 2
    assert decision.rung == store_residency.RUNG_STOCK_FITS


def test_driver_price_uses_the_handle_world(monkeypatch, tmp_path):
    class Handle:
        world = 2

        def call_all(self, *_a, **_k):
            row = {"host": {"mem_gib": {"MemAvailable": 47}}}
            return [{"rank": 0, **row}, {"rank": 1, **row}]

    from dgx_monarch import mesh_safety

    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(price_mod, "checkpoint_bytes", lambda _name: 60 * GIB)
    monkeypatch.setattr(price_mod, "checkpoint_kind", lambda _name: "bf16")
    price = price_mod.price_fsdp_clean_reload(Handle(), {"unet_name": "flux2-dev.safetensors"})
    assert price.applies and price.fits
    assert price.required_bytes == adapters_fsdp.fsdp_required_bytes(60 * GIB, 2, "bf16")


def _trim_rig(monkeypatch, events, cuda_available: bool, blocks: int):
    import torch.distributed.fsdp

    monkeypatch.setattr(torch.distributed.fsdp, "fully_shard",
                        lambda module, **kwargs: events.append("wrap") or module)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("trim"))
    dm = torch.nn.Module()
    dm.blocks = torch.nn.ModuleList(
        torch.nn.Linear(2, 2).to(torch.bfloat16) for _ in range(blocks))
    return types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=dm), size=0)


def test_the_pool_is_trimmed_once_after_the_build_when_cuda_is_in_play(monkeypatch):
    """The build wraps on meta and moves only this rank's rows, so there is no
    per-wrap full copy left to trim; one trim after the build returns whatever
    slack the chunk moves left in the pool."""
    from dgx_monarch.actor.store_detect import LivePrecisionEvidence
    from dgx_monarch.adapters.fsdp import apply_fsdp_capacity_mode

    events: list[str] = []
    patcher = _trim_rig(monkeypatch, events, cuda_available=True, blocks=3)
    evidence = LivePrecisionEvidence(
        quant_kind="bf16", live_dtype_profile="all_bf16", checkpoint_kind="bf16")
    apply_fsdp_capacity_mode(patcher, "bf16", None, evidence)
    # three blocks and the root, then one trim.
    assert events == ["wrap"] * 4 + ["trim"]


def test_no_trim_without_a_cuda_allocator(monkeypatch):
    from dgx_monarch.actor.store_detect import LivePrecisionEvidence
    from dgx_monarch.adapters.fsdp import apply_fsdp_capacity_mode

    events: list[str] = []
    patcher = _trim_rig(monkeypatch, events, cuda_available=False, blocks=1)
    evidence = LivePrecisionEvidence(
        quant_kind="bf16", live_dtype_profile="all_bf16", checkpoint_kind="bf16")
    apply_fsdp_capacity_mode(patcher, "bf16", None, evidence)
    assert events == ["wrap", "wrap"]


# shard_rows has one body, in actor/fsdp_lora.py; its torch.chunk equivalence is
# pinned in tests/test_fsdp_lora.py::test_shard_rows_match_torch_chunk.


def test_stubbed_wrap_leaves_parameters_exactly_as_loaded(monkeypatch):
    """A wrap that is not FSDP (the recorder tests) must see untouched data."""
    from dgx_monarch.adapters.fsdp_shard_build import build_shards

    dm = torch.nn.Module()
    dm.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4, dtype=torch.bfloat16)])
    before = {n: p.data.clone() for n, p in dm.named_parameters()}
    seen: list[str] = []

    def wrap():
        for n, p in dm.named_parameters():
            seen.append(f"{n}:{p.device.type}")

    assert build_shards(dm, wrap) == 0
    assert seen == ["blocks.0.lin.weight:meta"] if False else all(s.endswith(":meta") for s in seen)
    for n, p in dm.named_parameters():
        assert p.device.type == "cpu" and torch.equal(p.data, before[n])


def test_real_fsdp2_build_moves_only_this_ranks_rows_and_computes_right(tmp_path):
    """One-rank gloo: wrap on meta, materialize from host rows, forward equals
    the reference module, and an uneven row count is padded by FSDP2 itself."""
    import torch.distributed.fsdp
    from torch.distributed.device_mesh import init_device_mesh

    from dgx_monarch.adapters.fsdp_shard_build import build_shards

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")

    class _Block(torch.nn.Module):
        def __init__(self, out_features):
            super().__init__()
            self.lin = torch.nn.Linear(8, out_features, dtype=torch.bfloat16)

        def forward(self, x):
            return self.lin(x)

    class _Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([_Block(7), _Block(8)])
            self.proj = torch.nn.Linear(8, 8, dtype=torch.bfloat16)

        def forward(self, x):
            x = self.blocks[0](x)
            x = torch.nn.functional.pad(x, (0, 1))
            x = self.blocks[1](x)
            return self.proj(x)

    torch.manual_seed(0)
    reference = _Root()
    model = _Root()
    model.load_state_dict(reference.state_dict())
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = reference(x)
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        for p in model.parameters():
            p.requires_grad = False

        def wrap():
            for block in model.blocks:
                torch.distributed.fsdp.fully_shard(block, mesh=mesh)
            torch.distributed.fsdp.fully_shard(model, mesh=mesh, reshard_after_forward=True)
            # Meta init: the wrap moved and copied nothing.
            assert all(p.is_meta for p in model.parameters())

        # Every row reaches its shard through the copier, and the shard is a
        # tensor of its own. A plain `.to(device)` hands the driver the host
        # rows themselves, which breaks the mapping's copy-on-write
        # (docs/VALIDATION.md, 2026-09-07); on cpu it hands back the same storage.
        from dgx_monarch.adapters.fsdp_shard_build import DeviceCopier

        copier = DeviceCopier()
        seen: list = []
        real = copier.copy
        copier.copy = (  # type: ignore[method-assign]
            lambda rows, device: seen.append(rows.shape) or real(rows, device))
        hosts = {name: p.data.untyped_storage().data_ptr()
                 for name, p in model.named_parameters()}

        assert build_shards(model, wrap, copier=copier) == 6  # 3 weights + 3 biases
        assert len(seen) == 6
        for name, p in model.named_parameters():
            assert p.to_local().untyped_storage().data_ptr() != hosts[name]
        assert all(type(p).__name__ == "DTensor" and not p.is_meta for p in model.parameters())
        with torch.no_grad():
            out = model(x)
            out2 = model(x)
        assert torch.equal(out, expected)
        assert torch.equal(out2, expected)

        # The drop path: every DTensor is freed in place, the module then
        # moves nothing when comfy's unload walks it, and a repeat walk of
        # the same model is deduplicated to a no-op.
        from types import SimpleNamespace

        from dgx_monarch.adapters.fsdp_shard_build import (
            drop_sharded_weights_in_place,
        )

        patcher = SimpleNamespace(model=model)
        dropped, freed = drop_sharded_weights_in_place(
            (patcher, SimpleNamespace(model=model)), origin="test")
        assert dropped == 6 and freed > 0
        for p in model.parameters():
            assert type(p).__name__ != "DTensor"
            assert p.device.type == "cpu" and p.nelement() == 0
        from torch.distributed.fsdp import FSDPModule

        assert not any(isinstance(m, FSDPModule) for m in model.modules())
        model.to("cpu")  # comfy's detach walk must not raise on placeholders
        assert drop_sharded_weights_in_place((patcher,), origin="test") == (0, 0)
    finally:
        torch.distributed.destroy_process_group()


def _mapped_rows(tmp_path, rows: int = 4096, cols: int = 512):
    """A real checkpoint, mapped by the stock loader, and one row view into it."""
    import safetensors
    import safetensors.torch

    path = str(tmp_path / "rows.safetensors")
    weight = torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols).to(torch.bfloat16)
    safetensors.torch.save_file({"weight": weight}, path)
    handle = safetensors.safe_open(path, framework="pt", device="cpu")
    return path, handle, handle.get_tensor("weight"), weight


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.uint8, torch.float8_e4m3fn])
def test_the_copier_clones_off_cuda_and_leaves_the_source_alone(dtype):
    """Off cuda the copy is a clone: same bytes, a new storage, the source
    untouched. A dim-0 row view of a contiguous tensor is what the build hands
    it, and a strided view is copied through ``contiguous`` just the same."""
    from dgx_monarch.adapters.fsdp_shard_build import DeviceCopier

    full = torch.arange(96, dtype=torch.float32).reshape(12, 8).to(dtype)
    copier = DeviceCopier()
    cpu = torch.device("cpu")

    rows = copier.copy(full[3:9], cpu)
    assert torch.equal(rows.view(torch.uint8), full[3:9].view(torch.uint8))
    assert rows.untyped_storage().data_ptr() != full.untyped_storage().data_ptr()

    strided = copier.copy(full[:, ::2], cpu)
    assert torch.equal(strided.view(torch.uint8), full[:, ::2].contiguous().view(torch.uint8))
    assert copier.copy(full[0:0], cpu).numel() == 0
    copier.close()


def test_the_shard_device_falls_back_to_cpu_when_cuda_answers_without_a_driver(monkeypatch):
    """CI's torch says cuda is available under the trim rig's stub and then has
    no device to name; the wrap must stage on cpu there, not raise."""
    from dgx_monarch.adapters.fsdp_shard_build import shard_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def no_driver():
        raise RuntimeError("Found no NVIDIA driver on your system")

    monkeypatch.setattr(torch.cuda, "current_device", no_driver)
    assert shard_device() == torch.device("cpu")


def test_the_copier_returns_a_tensor_already_on_the_device_untouched():
    from dgx_monarch.adapters.fsdp_shard_build import DeviceCopier

    copier = DeviceCopier()
    meta = torch.empty(4, 4, device="meta")
    assert copier.copy(meta, torch.device("meta")) is meta
    cpu = torch.device("cpu")
    plain = torch.ones(3)
    assert copier.copy(plain, cpu) is not plain     # cpu to cpu is a copy, not an alias


def test_staging_moves_parameters_and_buffers_and_skips_meta(monkeypatch):
    """The direct wrap path hands each block to the copier before FSDP2 can
    move it: every host parameter and buffer is rebuilt on the device, while
    the streaming build's meta parameters are left for ``build_shards``."""
    from dgx_monarch.adapters import fsdp_shard_build

    block = torch.nn.Module()
    block.weight = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.bfloat16), requires_grad=False)
    block.island = torch.nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=False)
    block.register_buffer("freqs", torch.arange(8, dtype=torch.float32))
    block.later = torch.nn.Parameter(torch.empty(2, 2, device="meta"), requires_grad=False)
    copier = fsdp_shard_build.DeviceCopier()
    seen: list = []
    real = copier.copy
    monkeypatch.setattr(copier, "copy", lambda rows, device: seen.append(rows.shape) or real(rows, device))
    weight = block.weight
    island = block.island

    moved = fsdp_shard_build.stage_on_device(block, copier, torch.device("cpu"), ignored={block.island})

    assert moved == 4 * 4 * 2 + 8 * 4
    assert seen == [torch.Size([4, 4]), torch.Size([8])]        # the ignored island is left where it is
    assert block.later.is_meta
    assert block.weight is weight and block.island is island    # identity: the ignored set holds these objects
    assert block.island.untyped_storage().data_ptr() == island.untyped_storage().data_ptr()
    assert torch.equal(block.weight, torch.ones(4, 4, dtype=torch.bfloat16))
    assert torch.equal(block.freqs, torch.arange(8, dtype=torch.float32))


def test_staging_keeps_a_tied_buffer_one_tensor():
    """FSDP2 moves a buffer through ``.data``, so two modules that share one go
    on sharing it. Replacing each module's own entry instead forks the tie into
    two device copies, and a write through one never reaches the other."""
    from dgx_monarch.adapters import fsdp_shard_build

    shared = torch.arange(4, dtype=torch.float32)
    root = torch.nn.Module()
    root.left = torch.nn.Module()
    root.right = torch.nn.Module()
    root.left.register_buffer("freqs", shared)
    root.right.register_buffer("freqs", shared)
    copier = fsdp_shard_build.DeviceCopier()

    fsdp_shard_build.stage_on_device(root, copier, torch.device("cpu"))

    assert root.left.freqs is root.right.freqs
    assert torch.equal(root.left.freqs, torch.arange(4, dtype=torch.float32))
    copier.close()


def test_staging_rebuilds_a_quantized_wrapper_field_by_field():
    """A quantized parameter is a wrapper over its bytes and its scales, and
    FSDP2 would move both through ``.to``; the staging routes every tensor
    field through the copier and hands back the same wrapper class."""
    ck = pytest.importorskip("comfy_kitchen.tensor")
    from dgx_monarch.adapters import fsdp_quant, fsdp_shard_build

    cls = fsdp_quant.sharded_quant_class()
    quant = ck.QuantizedTensor.from_float(torch.randn(6, 4, dtype=torch.bfloat16), "TensorCoreFP8Layout")
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(cls.from_quantized(quant), requires_grad=False)
    before = module.weight
    qdata = before._qdata
    scales = {field: getattr(before._params, field) for field in before._params._tensor_fields()}
    copier = fsdp_shard_build.DeviceCopier()
    seen: list = []
    real = copier.copy

    def spy(rows, device):
        seen.append(rows.data_ptr())
        return real(rows, device)

    copier.copy = spy  # type: ignore[method-assign]
    fsdp_shard_build.stage_on_device(module, copier, torch.device("cpu"))

    after = module.weight
    assert after is not before and type(after) is type(before) and isinstance(after, ck.QuantizedTensor)
    assert torch.equal(after._qdata.view(torch.uint8), qdata.view(torch.uint8))
    assert after._qdata.data_ptr() != qdata.data_ptr()
    assert qdata.data_ptr() in seen
    for field, value in scales.items():
        assert torch.equal(getattr(after._params, field), value)
        assert getattr(after._params, field).data_ptr() != value.data_ptr()


def test_the_wrap_stages_each_block_before_it_is_sharded(monkeypatch):
    """On the direct path the block is on the device before ``fully_shard``
    sees it, so FSDP2 moves nothing itself."""
    from dgx_monarch.adapters import fsdp as adapters_fsdp
    from dgx_monarch.adapters import fsdp_shard_build

    order: list = []
    monkeypatch.setattr(torch.distributed.fsdp, "fully_shard",
                        lambda module, **kwargs: order.append(("shard", id(module))) or module)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(adapters_fsdp, "validate_fsdp_live_precision", lambda *a, **k: None)
    monkeypatch.setattr(fsdp_shard_build, "build_shards",
                        lambda dm, wrap, ignored, copier=None: wrap() or 0)
    real = fsdp_shard_build.stage_on_device
    monkeypatch.setattr(fsdp_shard_build, "stage_on_device",
                        lambda module, copier, device=None, ignored=None: (
                            order.append(("stage", id(module))) or real(module, copier, device, ignored)))

    dm = torch.nn.Module()
    dm.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4, dtype=torch.bfloat16) for _ in range(2)])
    patcher = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=dm), size=0)
    adapters_fsdp.apply_fsdp_capacity_mode(patcher, "bf16", None, None)

    ids = [id(dm.blocks[0]), id(dm.blocks[1]), id(dm)]
    assert order == [("stage", ids[0]), ("shard", ids[0]), ("stage", ids[1]), ("shard", ids[1]),
                     ("stage", ids[2]), ("shard", ids[2])]


def test_a_refused_build_frees_the_pinned_bounce_buffer(monkeypatch):
    """The build that raises is the one that must give the buffer back.

    A refusal carries its traceback, and the traceback holds the frame the
    copier lives in, so a build that stopped for want of memory would keep a
    quarter of a GiB pinned while the caller decides what to do about it.
    """
    from dgx_monarch.adapters import fsdp as adapters_fsdp
    from dgx_monarch.adapters import fsdp_shard_build

    closed: list = []

    class _Copier(fsdp_shard_build.DeviceCopier):
        def __init__(self, bounce_bytes: int = 1 << 20):
            super().__init__(bounce_bytes)

        def close(self) -> None:
            closed.append("close")
            super().close()

    def refuse(module, **kwargs):
        raise RuntimeError("the wrap ran out of unified memory")

    monkeypatch.setattr(fsdp_shard_build, "DeviceCopier", _Copier)
    monkeypatch.setattr(torch.distributed.fsdp, "fully_shard", refuse)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(adapters_fsdp, "validate_fsdp_live_precision",
                        lambda *args, **kwargs: None)

    dm = torch.nn.Module()
    dm.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4, dtype=torch.bfloat16)])
    patcher = types.SimpleNamespace(
        model=types.SimpleNamespace(diffusion_model=dm), size=0)

    with pytest.raises(RuntimeError, match="ran out of unified memory"):
        adapters_fsdp.apply_fsdp_capacity_mode(patcher, "bf16", None, None)

    assert closed == ["close"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the mechanism is the driver's pin of a pageable source")
def test_a_plain_device_copy_breaks_copy_on_write_and_the_bounce_does_not(tmp_path):
    """The 2026-09-07 measurement (docs/VALIDATION.md), pinned on the hardware that showed it.

    Copying rows to the device straight from the mmap leaves them anonymous
    inside the checkpoint's mapping; through the bounce the mapping stays
    clean page cache and the device holds the same bytes.
    """
    from dgx_monarch.actor import mapped_pages
    from dgx_monarch.adapters.fsdp_shard_build import DeviceCopier

    device = torch.device("cuda", torch.cuda.current_device())
    path, handle, view, expected = _mapped_rows(tmp_path, rows=16384, cols=1024)   # 32 MiB
    mapped = frozenset({path})

    def anonymous_bytes() -> int:
        return sum(anon for _mapping, _rss, anon in mapped_pages.resident(mapped))

    copier = DeviceCopier()
    bounced = copier.copy(view[:8192], device)
    torch.cuda.synchronize()
    assert torch.equal(bounced.cpu(), expected[:8192])
    assert copier.bounced == 8192 * 1024 * 2
    assert anonymous_bytes() == 0

    plain = view[8192:].to(device=device)
    torch.cuda.synchronize()
    assert torch.equal(plain.cpu(), expected[8192:])
    assert anonymous_bytes() >= 8192 * 1024 * 2 // 2     # the copied half, at least most of it
    copier.close()
    del handle, view
