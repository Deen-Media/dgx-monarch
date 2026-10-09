"""The FSDP materialize window: released before the load, and priced per rank.

Measurement M6 (2026-09-05 and 2026-09-07) found the shard price held and the
placement did not: the head kept a second copy of its shard anonymous inside
the checkpoint mapping, so ComfyUI partial-loaded the model and the
partial-load divergence guard refused class C (docs/VALIDATION.md, 2026-09-07
record). The driver breaks the mapping's copy-on-write under a device copy;
the bounce copy that prevents it is pinned in tests/test_fsdp_streaming_build.py.

Two claims, pinned here on CPU: no mapped file page survives the load, and the
bytes ComfyUI is charged are this rank's shard. The comfy-side arithmetic those
numbers feed is pinned against the installed comfy in
tests/canary/comfy_seam_contracts.py, seam 30.
"""
from __future__ import annotations

import gc
import mmap
import os
import sys
import types
import weakref

import pytest
import torch

from dgx_monarch.actor import comfy_shard_size, fsdp_streaming, mapped_pages
from store_ensure_helpers import rig as rig  # a fixture the tests below ask for

GIB = 1 << 30


def _checkpoint(tmp_path, name: str = "dit.safetensors"):
    """A real safetensors file, its cache clean, so a read maps clean pages."""
    import safetensors.torch

    path = str(tmp_path / name)
    safetensors.torch.save_file(
        {"block.weight": torch.zeros(2048, 512, dtype=torch.float32),
         "island": torch.ones(64, 64, dtype=torch.float32),
         "freqs": torch.arange(256, dtype=torch.float32)}, path)
    handle = os.open(path, os.O_RDONLY)
    os.fsync(handle)
    os.close(handle)
    return path


def _model_from_file(path):
    """What a shard build leaves: replicated islands and buffers, still mapped.

    The sharded rows are gone by then, moved to the device and rebuilt as
    DTensors, so what is modelled here is only what the build had no reason to
    rebuild. One of those tensors is enough to hold the whole file open.
    """
    import safetensors

    module = torch.nn.Module()
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        island = handle.get_tensor("island")
        freqs = handle.get_tensor("freqs")
    module.register_parameter("island", torch.nn.Parameter(island, requires_grad=False))
    module.register_buffer("freqs", freqs)
    module.island.sum()          # fault the pages the way a read would
    module.freqs.sum()
    return module


def test_one_file_backed_buffer_keeps_the_whole_checkpoint_mapped(tmp_path):
    """The fault this fixes, stated as the kernel states it.

    Nothing about the sharded rows is wrong. What kept 37 to 46 GiB charged to
    the head (2026-09-05) was a handful of tensors the build never rebuilt,
    each of them a view into one mapping of the entire file.
    """
    path = _checkpoint(tmp_path)
    module = _model_from_file(path)

    mapped = frozenset({path})
    assert any(row.path == path for row in mapped_pages.file_mappings())
    assert mapped_pages.resident_bytes(mapped) > 0
    assert mapped_pages.backing_paths([module.island, module.freqs]) == mapped


def test_the_release_unmaps_the_checkpoint_and_changes_no_byte(tmp_path):
    """After the release the file is not mapped here at all, and the tensors
    that were backed by it still read exactly what the file holds."""
    path = _checkpoint(tmp_path)
    module = _model_from_file(path)
    before = mapped_pages.resident_bytes(frozenset({path}))

    reported = fsdp_streaming.release_materialize_window(module, path)

    assert reported == before > 0
    assert [row for row in mapped_pages.file_mappings() if row.path == path] == []
    assert mapped_pages.resident_bytes(frozenset({path})) == 0
    assert torch.equal(module.island, torch.ones(64, 64))
    assert torch.equal(module.freqs, torch.arange(256, dtype=torch.float32))


def test_the_release_measures_the_file_the_kernel_names_not_the_loader_s_alias(
        tmp_path):
    """An FSDP load hands ComfyUI a `/proc/self/fd` symlink so the proven inode
    is the one it opens (actor/fsdp_checkpoint_pin.py). The kernel names the
    resolved file in its mapping table, so a release that measured the loader's
    own path literally would match nothing and report a clean window over a
    mapping that never moved."""
    real = _checkpoint(tmp_path)
    module = _model_from_file(real)
    owner = open(real, "rb", buffering=0)
    alias = str(tmp_path / "checkpoint.safetensors")
    os.symlink(f"/proc/{os.getpid()}/fd/{owner.fileno()}", alias)
    try:
        assert mapped_pages.resident_bytes(frozenset({alias})) == 0   # the trap
        reported = fsdp_streaming.release_materialize_window(module, alias)
    finally:
        owner.close()

    assert reported > 0
    assert mapped_pages.resident_bytes(frozenset({real})) == 0


def test_the_release_reaches_the_patcher_s_model(tmp_path):
    """The call site hands it a ModelPatcher, not a module. Walking the patcher
    itself would find no parameters and report a clean release over a mapping
    that never moved."""
    path = _checkpoint(tmp_path)
    patcher = types.SimpleNamespace(model=_model_from_file(path))

    fsdp_streaming.release_materialize_window(patcher, path)

    assert mapped_pages.resident_bytes(frozenset({path})) == 0


def test_the_release_never_raises_into_the_load_path(tmp_path):
    """A load that would have succeeded must not fail on an accounting step."""
    path = _checkpoint(tmp_path)

    class _Hostile:
        @property
        def model(self):
            raise RuntimeError("no model here")

    assert fsdp_streaming.release_materialize_window(_Hostile(), path) == 0


def test_the_release_reads_the_copy_on_write_bytes_while_the_mapping_is_whole(
        tmp_path, monkeypatch):
    """The copy-on-write figure is what the hardware legs accept on, so it has
    to be read before the copies. A mapping only carries those pages while
    something still holds it, and the copies are what let it go, so a reading
    taken afterwards is zero for the load that has the fault."""
    path = _checkpoint(tmp_path)
    module = _model_from_file(path)
    module.island.data[0, 0] = 7.0        # a copy-on-write page in the mapping
    warned: list = []
    monkeypatch.setattr(fsdp_streaming.log, "warning",
                        lambda message, *args: warned.append(message % args))

    fsdp_streaming.release_materialize_window(module, path)

    assert [line for line in warned if "copy-on-write" in line]
    assert float(module.island[0, 0]) == 7.0
    assert mapped_pages.resident_bytes(frozenset({path})) == 0


def _named_mapping(pointer: int):
    """The kernel's own row for this address, whatever the row names."""
    with open("/proc/self/maps") as handle:
        for line in handle:
            row = mapped_pages._row(line)
            if row is not None and row.holds(pointer):
                return row
    return None


def _anonymous_at(mapping) -> int:
    """What smaps counts as anonymous inside one mapping."""
    inside = False
    with open("/proc/self/smaps") as handle:
        for line in handle:
            if mapped_pages._SMAPS_HEADER.match(line):
                row = mapped_pages._row(line)
                inside = row is not None and row.start == mapping.start
            elif inside and line.startswith("Anonymous:"):
                return int(line.split()[1]) * 1024
    return 0


def test_a_mapping_that_is_not_a_file_is_never_a_checkpoint():
    """A pathname is not proof of a checkpoint, and the drop must read a file.

    The kernel names plenty of mappings that hold no checkpoint. Treating one
    as the checkpoint copies a tensor that needed no copy and hands a range
    nothing here owns to madvise.
    """
    handle = os.open("/dev/zero", os.O_RDWR)
    try:
        region = mmap.mmap(handle, 1 << 20, flags=mmap.MAP_PRIVATE,
                           prot=mmap.PROT_READ | mmap.PROT_WRITE)
    finally:
        os.close(handle)
    try:
        region.write(b"\x07" * (1 << 20))
        tensor = torch.frombuffer(region, dtype=torch.uint8)
        assert _named_mapping(tensor.untyped_storage().data_ptr()) is not None
        assert mapped_pages.backing_paths([tensor]) == frozenset()
        assert mapped_pages.resident_bytes(frozenset({"/dev/zero"})) == 0
        assert mapped_pages.drop_mapped_pages(frozenset({"/dev/zero"})) == 0
        assert int(tensor.sum()) == 7 * (1 << 20)
        del tensor
    finally:
        region.close()


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="the driver's own mapping of pinned host memory")
def test_pinned_host_memory_is_not_a_checkpoint_to_release():
    """Measured on the rig 2026-09-09: the driver backs pinned host memory with
    a private mapping of a device node, and smaps counts none of it anonymous.
    So the anonymous rule alone would not have kept the release out of it: the
    detach would have copied a pinned weight into pageable memory, unpinning it,
    and the drop would have handed the driver's range to madvise."""
    pinned = torch.zeros(4 << 20, dtype=torch.uint8, pin_memory=True)
    pinned.fill_(7)
    module = torch.nn.Module()
    module.register_buffer("weights", pinned)
    mapping = _named_mapping(pinned.untyped_storage().data_ptr())

    assert mapping is not None and _anonymous_at(mapping) == 0    # the trap
    assert mapped_pages.backing_paths([pinned]) == frozenset()
    assert mapped_pages.detach_from_files(module) == (0, 0, frozenset())
    assert module.weights is pinned and pinned.is_pinned()
    assert int(pinned.sum()) == 7 * (4 << 20)


def test_a_page_written_through_the_mapping_is_never_dropped(tmp_path):
    """MADV_DONTNEED discards a copy-on-write page instead of writing it back,
    so the drop is refused for any mapping the kernel counts as anonymous. A
    dirty page of the file's own cache is not that page."""
    import safetensors

    path = _checkpoint(tmp_path)
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor("block.weight")
    tensor.sum()
    mapped = mapped_pages.backing_paths([tensor])
    assert mapped == frozenset({path})
    assert mapped_pages.drop_mapped_pages(mapped) > 0        # clean, so dropped
    assert torch.equal(tensor, torch.zeros(2048, 512))       # and re-read whole

    tensor[0, 0] = 7.0                                       # now written through
    assert mapped_pages.drop_mapped_pages(mapped) == 0
    assert float(tensor[0, 0]) == 7.0


class _FakeDTensor:
    """A DTensor's size contract: ``nbytes`` is the global tensor's, and
    ``_local_tensor`` is the rows this rank holds."""

    def __init__(self, local: torch.Tensor, world: int):
        self._local_tensor = local
        self.nbytes = local.nbytes * world


class _ShardedModule:
    def __init__(self, entries: dict):
        self._entries = entries

    def state_dict(self):
        return dict(self._entries)


def _stock_module_size(module):
    """comfy/model_management.py module_size, verbatim at pin 3216c62e."""
    module_mem = 0
    sd = module.state_dict()
    for k in sd:
        t = sd[k]
        module_mem += t.nbytes
    return module_mem


@pytest.fixture
def fake_model_management(monkeypatch):
    stub = types.ModuleType("comfy.model_management")
    stub.module_size = _stock_module_size
    comfy = types.ModuleType("comfy")
    comfy.model_management = stub
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", stub)
    yield stub
    comfy_shard_size.uninstall()


def test_comfy_is_charged_this_rank_s_shard_and_not_the_world_s(fake_model_management):
    """The number M6 caught. comfy sums the state dict's ``nbytes`` per module,
    and a DTensor answers for the whole logical tensor, so a rank holding half
    a 61.7 GiB checkpoint was charged 61.7 and offloaded what it already had."""
    local = torch.zeros(1024, 512, dtype=torch.bfloat16)
    block = _ShardedModule({"weight": _FakeDTensor(local, world=2),
                            "bias": torch.zeros(512, dtype=torch.bfloat16)})
    world_bytes = fake_model_management.module_size(block)

    assert comfy_shard_size.install() is True
    local_bytes = fake_model_management.module_size(block)

    assert world_bytes == local.nbytes * 2 + 512 * 2
    assert local_bytes == local.nbytes + 512 * 2
    assert local_bytes * 2 > world_bytes > local_bytes


def test_a_model_that_is_not_sharded_is_charged_exactly_what_comfy_charged_it(
        fake_model_management):
    """The install is process wide, so this is the property that makes it safe:
    only a DTensor's answer moves. Every other model keeps stock's number."""
    plain = _ShardedModule({"weight": torch.zeros(320, 64, dtype=torch.float16),
                            "bias": torch.zeros(64, dtype=torch.float16)})
    stock = _stock_module_size(plain)

    comfy_shard_size.install()

    assert fake_model_management.module_size(plain) == stock


def test_the_install_is_idempotent_and_reversible(fake_model_management):
    """A second FSDP load in the same process must not wrap the shim in itself,
    which would charge a shard of a shard."""
    assert comfy_shard_size.install() is True
    once = fake_model_management.module_size
    assert comfy_shard_size.install() is False
    assert fake_model_management.module_size is once

    assert comfy_shard_size.uninstall() is True
    assert fake_model_management.module_size is _stock_module_size
    assert comfy_shard_size.uninstall() is False


def test_a_comfy_without_the_function_is_not_a_failed_load(monkeypatch):
    """A stand-in comfy carries no model_management. The price is then whatever
    that stand-in charges, and the load goes on."""
    stub = types.ModuleType("comfy.model_management")
    comfy = types.ModuleType("comfy")
    comfy.model_management = stub
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", stub)

    assert comfy_shard_size.install() is False


def test_both_numbers_comfy_decides_by_land_on_the_shard(fake_model_management):
    """comfy reads two numbers and they have to agree.

    ``ModelPatcher.model_size()`` returns the cached ``size``, which the wrap
    writes, and ``ModelPatcher.load`` charges each module
    ``module_size``. With the first at the shard's size and the second at the
    world's, ``partially_load`` takes the full path and the block loop still
    offloads inside it. The rules both numbers feed are pinned against the
    installed comfy in tests/canary/comfy_seam_contracts.py, seam 30; what this
    asserts is which side of them each number falls on.
    """
    from dgx_monarch.adapters.fsdp_quant import local_nbytes

    local = torch.zeros(4096, 1024, dtype=torch.bfloat16)
    blocks = [_ShardedModule({"weight": _FakeDTensor(local, world=2)})
              for _ in range(8)]
    comfy_shard_size.install()

    patcher_size = sum(local_nbytes(entry)
                       for block in blocks
                       for entry in block.state_dict().values())
    charged = sum(fake_model_management.module_size(block) for block in blocks)
    assert charged == patcher_size

    # The band the fix is for: memory enough for the shard, not for the file.
    offered = int(patcher_size * 1.4)
    assert offered > patcher_size            # full_load, so nothing is offloaded
    assert offered < patcher_size * 2        # and the world's size would refuse


def test_the_wrap_installs_the_shard_aware_sum(monkeypatch, fake_model_management):
    """The wrap is the one moment a rank learns it holds a shard, so it is
    where comfy is told, beside the cached size it already sets."""
    from dgx_monarch.adapters import fsdp as adapters_fsdp

    installed: list = []
    monkeypatch.setattr(comfy_shard_size, "install",
                        lambda: bool(installed.append("install")))
    monkeypatch.setattr(torch.distributed.fsdp, "fully_shard",
                        lambda module, **kwargs: module)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(adapters_fsdp, "validate_fsdp_live_precision",
                        lambda *args, **kwargs: None)

    dm = torch.nn.Module()
    dm.blocks = torch.nn.ModuleList(
        [torch.nn.Linear(4, 4, dtype=torch.bfloat16) for _ in range(2)])
    patcher = types.SimpleNamespace(
        model=types.SimpleNamespace(diffusion_model=dm), size=0)

    adapters_fsdp.apply_fsdp_capacity_mode(patcher, "bf16", None, None)

    assert installed == ["install"]
    assert patcher.size > 0


def test_the_window_is_released_before_the_model_reaches_comfy(
        rig, monkeypatch, tmp_path):
    """ComfyUI decides how much of a model to move from the memory it can see,
    and mapped pages are charged against that, so the release has to happen
    before the store publishes a model any sampler could hand to
    ``load_models_gpu``.
    """
    from dgx_monarch.actor import model_store as ms
    from store_ensure_helpers import _write_one_tensor_safetensors

    store, _calls = rig
    path = tmp_path / "model.safetensors"
    _write_one_tensor_safetensors(path, "BF16")
    monkeypatch.setattr(ms, "resolve_model_path", lambda *_args: str(path))
    monkeypatch.setattr(ms, "stock_load_preflight", lambda *_args: None)

    order: list = []
    monkeypatch.setattr(fsdp_streaming, "release_materialize_window",
                        lambda model, load_path: order.append("release") or 0)
    build = store._build_active
    monkeypatch.setattr(store, "_build_active",
                        lambda base, stack: order.append("publish") or build(base, stack))

    store.ensure("model.safetensors", None, None, fsdp_launch=True,
                 on_base_loaded=lambda *args: order.append("shards"))

    assert order == ["shards", "release", "publish"]


def test_a_load_that_is_not_sharded_releases_no_window(rig, monkeypatch, tmp_path):
    """A stock load leaves the pread backend in place and materializes no mmap
    window, so there is nothing to release and the walk is not paid for."""
    from dgx_monarch.actor import model_store as ms
    from store_ensure_helpers import _write_one_tensor_safetensors

    store, _calls = rig
    path = tmp_path / "model.safetensors"
    _write_one_tensor_safetensors(path, "BF16")
    monkeypatch.setattr(ms, "resolve_model_path", lambda *_args: str(path))
    monkeypatch.setattr(ms, "stock_load_preflight", lambda *_args: None)

    released: list = []
    monkeypatch.setattr(fsdp_streaming, "release_materialize_window",
                        lambda model, load_path: released.append(load_path) or 0)

    store.ensure("model.safetensors", None, None)

    assert released == []


def test_the_detach_lets_the_last_reference_die(tmp_path):
    """The copy is what frees the mapping, so the copies must not themselves be
    views of it. A clone that aliased its source would leave the file mapped
    and report a release that did not happen."""
    path = _checkpoint(tmp_path)
    module = _model_from_file(path)
    source = weakref.ref(module.freqs)

    count, copied, paths = mapped_pages.detach_from_files(module)
    gc.collect()

    assert count == 2 and copied > 0 and paths == frozenset({path})
    assert source() is None
    assert mapped_pages.backing_paths(
        [module.island, module.freqs]) == frozenset()


def test_the_detach_frees_a_buffer_two_modules_share(tmp_path):
    """A shared buffer has to leave the file once, under both its names.

    ``named_buffers`` names a shared buffer once by default, so a detach that
    walks it replaces one owner's entry and leaves the other holding the
    mapping. The release then reports a copy it has not finished: the file stays
    mapped for the model's life, ComfyUI's own move of that buffer breaks the
    mapping's copy-on-write again, and the two modules no longer share one
    tensor.
    """
    path = _checkpoint(tmp_path)
    import safetensors

    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        shared = handle.get_tensor("freqs")
    shared.sum()
    root = torch.nn.Module()
    root.left = torch.nn.Module()
    root.right = torch.nn.Module()
    root.left.register_buffer("freqs", shared)
    root.right.register_buffer("freqs", shared)
    del shared

    count, copied, paths = mapped_pages.detach_from_files(root)
    gc.collect()

    assert count == 1 and copied == 256 * 4 and paths == frozenset({path})
    assert root.left.freqs is root.right.freqs
    assert torch.equal(root.left.freqs, torch.arange(256, dtype=torch.float32))
    assert mapped_pages.backing_paths(
        [root.left.freqs, root.right.freqs]) == frozenset()
    assert mapped_pages.resident_bytes(frozenset({path})) == 0
