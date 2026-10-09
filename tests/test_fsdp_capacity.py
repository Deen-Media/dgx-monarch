"""FSDP capacity mode: apply_fsdp_capacity_mode, its launch and precision
guards, the ignored-params set, and the driver-side FSDP checkpoint preflight
in nodes/loaders.py.

CPU only. The fully_shard_recorder fixture patches
torch.distributed.fsdp.fully_shard and torch.distributed.get_world_size, so the
real function body runs with no process group or GPU: fsdp.py imports
fully_shard inside the function, so patching the module attribute before the
call is enough. Three tests run real wraps on a one-rank gloo group: two
behavioural torch canaries and the outside-forward crash test.
"""
import json
import os
import struct
import sys
import types
from contextlib import nullcontext

import pytest
import torch
import torch.distributed.fsdp

from dgx_monarch import adapters
from dgx_monarch.actor import store_fsdp
from dgx_monarch.actor import worker as worker_module
from dgx_monarch.actor.store_detect import (
    LivePrecisionEvidence,
    _detect_live_precision,
    _detect_quant_kind,
)
from dgx_monarch.actor.worker import GPUWorker
from dgx_monarch.adapters import fsdp as fsdp_module
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.detect import (
    FsdpLaunchQuantProof,
    sniff_fsdp_launch_quant,
)
from dgx_monarch.adapters.fsdp import (
    _BLOCK_LIST_ATTRS,
    ALL_BF16_PROFILE,
    WAN_FP32_INGRESS_PROFILE,
    apply_fsdp_capacity_mode,
    fsdp_precision_profile_is_admitted,
    validate_fsdp_launch_loras,
    validate_fsdp_launch_quant,
)
from dgx_monarch.adapters.fsdp_islands import auxiliary_module_parameters
from dgx_monarch.mesh_safety import ArtifactBindingError, StockLoadCapacityError
from dgx_monarch.nodes import loaders
from dgx_monarch.refusal import RefusalClass, parse_refusal_tag
from dgx_monarch.safetensors_header import SafetensorsFileIdentity
from dgx_monarch.topology import Topology


class _Recorder:
    def __init__(self):
        self.calls = []
        self.kwargs = []

    def __call__(self, module, **kwargs):
        self.calls.append(module)
        self.kwargs.append(kwargs)
        return module


_FAKE_FSDP_FILE_IDENTITY = SafetensorsFileIdentity(
    file_dev=1,
    file_ino=2,
    file_size=3,
    file_mtime_ns=4,
    file_ctime_ns=5,
)


def _fsdp_quant_proof(quant_kind: str | None) -> FsdpLaunchQuantProof:
    return FsdpLaunchQuantProof(
        quant_kind=quant_kind,
        file_identity=_FAKE_FSDP_FILE_IDENTITY if quant_kind is not None else None,
    )


def test_worker_enables_strict_store_preflight_only_for_fsdp():
    seen = []
    store = types.SimpleNamespace(
        ensure=lambda **kwargs: seen.append(kwargs) or (object(), "load")
    )
    store_fsdp.ensure(types.SimpleNamespace(store=store, topology={}))
    store_fsdp.ensure(
        types.SimpleNamespace(store=store, topology={"fsdp": True})
    )
    assert seen == [{}, {"fsdp_launch": True}]


def test_fsdp_capacity_remedy_does_not_price_a_quantized_file_at_the_shard_factor(
    monkeypatch, tmp_path,
):
    """The remedy never offers an fp8 or int8 file at the streaming build's price.

    A quantized checkpoint keeps the direct wrap, which commits the whole file,
    so a remedy pricing it at 0.58 of its size (1/world + 0.08 at world 2)
    would send a refused operator to a load that can commit far more. It must
    not say FSDP refuses quantized files either: FSDP admits them and shards
    their stored bytes.
    """
    checkpoint = tmp_path / "large.safetensors"
    checkpoint.write_bytes(b"x")
    monkeypatch.setattr(fsdp_module, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(fsdp_module, "mem_available_bytes", lambda: 0)
    monkeypatch.setattr(fsdp_module.os.path, "getsize", lambda _path: 8 << 30)

    with pytest.raises(StockLoadCapacityError) as raised:
        fsdp_module.fsdp_load_capacity_check(str(checkpoint), "large.safetensors", {})

    message = str(raised.value)
    assert "supported fp8/int8 checkpoint" not in message
    assert "helps only by being smaller" in message
    assert "the direct path, which holds the whole file at once" in message
    assert "refuses quantized files" not in message


@pytest.fixture
def fully_shard_recorder(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(torch.distributed.fsdp, "fully_shard", recorder)
    # apply_fsdp_capacity_mode reads dist.get_world_size() for its log line,
    # and no process group exists here.
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    return recorder


class _FakePatcher:
    def __init__(self, model):
        self.model = types.SimpleNamespace(diffusion_model=model)
        self.size = 0


def _fake_diffusion_model(attrs: dict[str, int], dtype=None) -> torch.nn.Module:
    """A real nn.Module whose block lists are nn.ModuleLists of `count`
    nn.Linear(2, 2) blocks, bf16 unless `dtype` says otherwise, so the wrap's
    requires_grad and local-bytes bookkeeping run on real parameters."""
    dm = torch.nn.Module()
    for attr, count in attrs.items():
        blocks = torch.nn.ModuleList(torch.nn.Linear(2, 2) for _ in range(count))
        blocks.to(dtype=dtype or torch.bfloat16)
        setattr(dm, attr, blocks)
    return dm


def _bf16_precision(patcher) -> LivePrecisionEvidence:
    return LivePrecisionEvidence(
        quant_kind="bf16",
        live_dtype_profile="all_bf16",
        checkpoint_kind="bf16",
    )


# family -> {block-list attr name: block count}. The names come from each
# family's adapter forward (`self.<attr>` block loops): wan_family.py,
# flux_family.py, hunyuan.py, qwen_image.py, lens.py, kandinsky5.py,
# pixeldit_comfy.py.
_FAMILY_BLOCK_LISTS = {
    "wan": {"blocks": 2},
    "flux": {"double_blocks": 2, "single_blocks": 2},
    "hunyuan": {"double_blocks": 2, "single_blocks": 2},
    "qwen_image": {"transformer_blocks": 2},
    "lens": {"transformer_blocks": 2},
    # The _BLOCK_LIST_ATTRS comment in adapters/fsdp.py says what breaks without these.
    "kandinsky5": {"visual_transformer_blocks": 2, "text_transformer_blocks": 2},
    "pixeldit_comfy": {"patch_blocks": 2, "pixel_blocks": 2},
}


@pytest.mark.parametrize("family", sorted(_FAMILY_BLOCK_LISTS))
def test_every_named_block_list_gets_per_block_wrapping(fully_shard_recorder, family):
    attrs = _FAMILY_BLOCK_LISTS[family]
    dm = _fake_diffusion_model(attrs)
    patcher = _FakePatcher(dm)

    apply_fsdp_capacity_mode(patcher, "bf16", None, _bf16_precision(patcher))

    # Every block of every named list gets its own wrap, beyond the root
    # catch-all, so a forward all-gathers one block at a time instead of the
    # whole model at once.
    for attr in attrs:
        for block in getattr(dm, attr):
            assert block in fully_shard_recorder.calls
    assert dm in fully_shard_recorder.calls  # the top-level catch-all wrap still runs
    assert patcher._dgxm_fsdp is True


def test_the_root_wrap_reshards_after_forward_and_blocks_keep_the_default(
    fully_shard_recorder,
):
    """Without reshard_after_forward=True on the root wrap, every parameter
    outside the block lists stays full size on every rank between renders.
    Blocks keep FSDP2's default. The comment above the root fully_shard call
    in adapters/fsdp.py gives the reason and the 2026-08-25 measurement."""
    dm = _fake_diffusion_model({"blocks": 3})
    patcher = _FakePatcher(dm)

    apply_fsdp_capacity_mode(patcher, "bf16", None, _bf16_precision(patcher))

    assert fully_shard_recorder.calls == [*dm.blocks, dm]
    assert fully_shard_recorder.kwargs == [
        {}, {}, {}, {"reshard_after_forward": True},
    ]


def test_torch_still_defaults_the_root_to_not_resharding():
    """Canary for the kwarg above: it matters only while torch leaves
    reshard_after_forward unset by default. If a torch bump sets a default,
    re-read what the root does before trusting the kwarg."""
    import inspect

    parameter = inspect.signature(
        torch.distributed.fsdp.fully_shard
    ).parameters["reshard_after_forward"]
    assert parameter.default is None


def test_torch_still_leaves_only_the_root_gathered_without_the_flag(tmp_path):
    """The behavioural half of the canary above, and the half that matters.

    An unset reshard_after_forward is not "off": torch reads it as auto, and
    lazy init then drops the post-forward mesh for the root state's own
    parameter group alone, which leaves those parameters gathered. Passing True
    skips that branch. A torch that kept the signature default but changed
    either half would pass the check above while the kwarg stopped doing
    anything, so run the real wraps on a one-rank gloo group and read the
    parameters back. The file store keeps this off any port.
    """
    import torch.nn as nn
    from torch.distributed.device_mesh import init_device_mesh

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return self.lin(x)

    class _Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([_Block(), _Block()])
            self.txt_in = nn.Linear(8, 8)

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.txt_in(x)

    def wrap_then_read(mesh, **root_kwargs):
        """(root still gathered, root param class, one block still gathered)."""
        model = _Root()
        for block in model.blocks:
            torch.distributed.fsdp.fully_shard(block, mesh=mesh)
        torch.distributed.fsdp.fully_shard(model, mesh=mesh, **root_kwargs)
        with torch.no_grad():
            model(torch.randn(2, 8))
        root_group = model._get_fsdp_state()._fsdp_param_group
        block_group = model.blocks[0]._get_fsdp_state()._fsdp_param_group
        return (
            root_group.is_unsharded,
            type(model.txt_in.weight).__name__,
            block_group.is_unsharded,
        )

    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_path / 'process_group'}",
        rank=0,
        world_size=1,
    )
    try:
        mesh = init_device_mesh("cpu", (1,))
        # Without the flag the root keeps its own parameters gathered after the
        # forward, while every block has resharded its own.
        assert wrap_then_read(mesh) == (True, "Parameter", False)
        # With the flag the root reshards too, and the blocks do not change.
        assert wrap_then_read(mesh, reshard_after_forward=True) == (
            False, "DTensor", False,
        )
    finally:
        torch.distributed.destroy_process_group()


def test_torch_still_defaults_forward_prefetch_off():
    """Canary for adapters/fsdp.py's set_modules_to_forward_prefetch([]) at
    prefetch depth 1.

    That call is a defensive pin on a torch default: explicit forward prefetch
    is opt-in and its list starts empty. A torch that ships explicit prefetch
    on by default fails here instead of leaving the comment in adapters/fsdp.py
    wrong."""
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    assert FSDPState()._states_to_forward_prefetch == []


def test_fsdp_ready_markers_are_absent_after_partial_fully_shard_failure(
    monkeypatch,
):
    calls = []

    def fail_on_second_module(module, **_kwargs):
        calls.append(module)
        if len(calls) == 2:
            raise RuntimeError("injected partial fully_shard failure")
        return module

    monkeypatch.setattr(
        torch.distributed.fsdp,
        "fully_shard",
        fail_on_second_module,
    )
    dm = _fake_diffusion_model({"blocks": 2})
    patcher = _FakePatcher(dm)

    with pytest.raises(RuntimeError, match="partial fully_shard failure"):
        apply_fsdp_capacity_mode(
            patcher, "bf16", None, _bf16_precision(patcher)
        )

    assert calls == [dm.blocks[0], dm.blocks[1]]
    assert not hasattr(patcher, "_dgxm_fsdp")
    assert not hasattr(patcher, "_dgxm_fsdp_ready")


def test_fsdp_ready_markers_wait_for_distributed_state_confirmation(
    monkeypatch,
    fully_shard_recorder,
):
    patcher = _FakePatcher(_fake_diffusion_model({"blocks": 1}))
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda: (_ for _ in ()).throw(RuntimeError("process group unavailable")),
    )

    with pytest.raises(RuntimeError, match="process group unavailable"):
        apply_fsdp_capacity_mode(
            patcher, "bf16", None, _bf16_precision(patcher)
        )

    assert fully_shard_recorder.calls
    assert not hasattr(patcher, "_dgxm_fsdp")
    assert not hasattr(patcher, "_dgxm_fsdp_ready")


def test_fsdp_requires_checkpoint_bound_precision_evidence_before_sharding(
    fully_shard_recorder,
):
    patcher = _FakePatcher(_fake_diffusion_model({"blocks": 1}))

    with pytest.raises(UnsupportedModelError, match="missing or malformed"):
        apply_fsdp_capacity_mode(patcher, "bf16", None)

    unbound = LivePrecisionEvidence("bf16", "all_bf16")
    with pytest.raises(UnsupportedModelError, match="not bound to a checkpoint of the detected kind"):
        apply_fsdp_capacity_mode(patcher, "bf16", None, unbound)

    assert fully_shard_recorder.calls == []
    assert not hasattr(patcher, "_dgxm_fsdp_ready")


def test_family_block_list_names_are_all_registered():
    # Every name the table uses is in fsdp.py's _BLOCK_LIST_ATTRS. A missing
    # one also fails the per-block test, through the no-shardable-lists
    # refusal or a missing wrap; this names the table as the cause.
    for attrs in _FAMILY_BLOCK_LISTS.values():
        for attr in attrs:
            assert attr in _BLOCK_LIST_ATTRS


def test_unrecognized_block_list_names_still_raise_typed(fully_shard_recorder):
    # A model with no block-list name fsdp.py recognizes raises
    # UnsupportedModelError instead of wrapping nothing.
    dm = _fake_diffusion_model({"totally_unknown_blocks": 2})
    patcher = _FakePatcher(dm)

    with pytest.raises(UnsupportedModelError, match="no shardable block lists"):
        apply_fsdp_capacity_mode(
            patcher, "bf16", None, _bf16_precision(patcher)
        )
    assert fully_shard_recorder.calls == []


def test_llm_adapter_parameters_are_ignored_on_every_wrap(fully_shard_recorder):
    """Anima's llm_adapter, run off the denoise path, never enters a collective.

    Its parameters ride ``ignored_params`` on every wrap, block and root alike,
    so FSDP2 leaves them plain replicated tensors and the module runs outside
    the denoise path with no peer waiting on an all-gather.
    """
    dm = _fake_diffusion_model({"blocks": 2})
    dm.llm_adapter = torch.nn.Linear(2, 2).to(torch.bfloat16)
    patcher = _FakePatcher(dm)

    apply_fsdp_capacity_mode(patcher, "bf16", None, _bf16_precision(patcher))

    adapter_params = set(dm.llm_adapter.parameters())
    assert fully_shard_recorder.calls == [*dm.blocks, dm]
    for kwargs in fully_shard_recorder.kwargs:
        assert kwargs["ignored_params"] == adapter_params
    assert fully_shard_recorder.kwargs[-1]["reshard_after_forward"] is True
    assert patcher._dgxm_fsdp is True
    # The adapter is not a block list, so it was never wrapped on its own.
    assert dm.llm_adapter not in fully_shard_recorder.calls


def test_h3_text_ingress_rides_ignored_params_and_the_line_states_its_bytes(
    fully_shard_recorder, caplog,
):
    """ComfyUI calls ``preprocess_text_embeds`` from ``extra_conds``, outside
    the diffusion forward, where no FSDP2 pre-forward hook has gathered
    anything, and H3 runs ``condition_proj`` then ``token_refiner`` there.
    Sharded by the root wrap they are DTensors meeting a plain text state: on
    the 61.7 GiB bf16 minimax_h3 leg at uly2+fsdp, world 2 (2026-09-09), every
    rank died on the first addmm. Both ride the ignored set, and the capacity
    line states the bytes a rank holds whole.
    """
    dm = _fake_diffusion_model({"blocks": 2})
    dm.condition_proj = torch.nn.Linear(4, 4).to(torch.bfloat16)
    dm.token_refiner = torch.nn.Linear(4, 4).to(torch.bfloat16)
    patcher = _FakePatcher(dm)

    with caplog.at_level("INFO", logger="dgx_monarch.adapters.fsdp"):
        apply_fsdp_capacity_mode(patcher, "bf16", None, _bf16_precision(patcher))

    ingress = set(dm.condition_proj.parameters()) | set(dm.token_refiner.parameters())
    assert len(ingress) == 4
    assert fully_shard_recorder.calls == [*dm.blocks, dm]
    for kwargs in fully_shard_recorder.kwargs:
        assert kwargs["ignored_params"] == ingress
    assert dm.condition_proj not in fully_shard_recorder.calls
    assert dm.token_refiner not in fully_shard_recorder.calls
    ingress_bytes = sum(p.numel() * p.element_size() for p in ingress)
    line = next(record.getMessage() for record in caplog.records
                if record.getMessage().startswith("FSDP capacity mode: sharded"))
    assert f"4 auxiliary-module parameters ignored, {ingress_bytes / 2**30:.3f} GiB" in line


def test_every_supported_family_runs_its_outside_forward_modules_unsharded(
    fully_shard_recorder,
):
    """The whole census, family by family, on a stand-in carrying its modules.

    One flat tuple feeds the wrap, but each name is declared under the comfy
    class whose ``extra_conds`` reaches it, so a family added to the table is
    held to the same rule here without a second test.
    """
    from dgx_monarch.adapters.fsdp_islands import (
        OUTSIDE_FORWARD_MODULE_ATTRS,
        auxiliary_module_parameters,
    )

    assert set(OUTSIDE_FORWARD_MODULE_ATTRS) == {"Anima", "MiniMaxH3", "LTXAV"}
    for model_base_class, attrs in OUTSIDE_FORWARD_MODULE_ATTRS.items():
        fully_shard_recorder.calls.clear()
        fully_shard_recorder.kwargs.clear()
        dm = _fake_diffusion_model({"blocks": 2})
        for attr in attrs:
            setattr(dm, attr, torch.nn.Linear(4, 4).to(torch.bfloat16))
        expected = {p for attr in attrs for p in getattr(dm, attr).parameters()}
        assert auxiliary_module_parameters(dm) == expected, model_base_class
        patcher = _FakePatcher(dm)

        apply_fsdp_capacity_mode(patcher, "bf16", None, _bf16_precision(patcher))

        assert fully_shard_recorder.calls == [*dm.blocks, dm], model_base_class
        for kwargs in fully_shard_recorder.kwargs:
            assert kwargs["ignored_params"] == expected, model_base_class
        for attr in attrs:
            assert getattr(dm, attr) not in fully_shard_recorder.calls, model_base_class


def test_the_census_tuple_is_the_table_and_no_name_is_a_block_list():
    """The flat census is the table, and two silent errors fail here: a name
    declared twice, and a name that is also a block-list attribute, which would
    put a whole stack of blocks in the ignored set and leave it unsharded."""
    from dgx_monarch.adapters.fsdp_islands import (
        AUXILIARY_MODULE_ATTRS,
        OUTSIDE_FORWARD_MODULE_ATTRS,
    )

    flat = [attr for attrs in OUTSIDE_FORWARD_MODULE_ATTRS.values() for attr in attrs]
    assert AUXILIARY_MODULE_ATTRS == tuple(dict.fromkeys(flat))
    assert len(set(flat)) == len(flat), f"a name is declared twice: {flat}"
    collisions = set(AUXILIARY_MODULE_ATTRS) & set(_BLOCK_LIST_ATTRS)
    assert not collisions, f"block lists named as outside-forward modules: {collisions}"


def _patcher_for_comfy_class(class_name: str, dm) -> object:
    """A patcher whose ``model`` is an instance of a class of that name, the
    way comfy hands the wrap a ``model_base`` instance."""
    model_class = type(class_name, (), {})
    patcher = _FakePatcher(dm)
    patcher.model = model_class()
    patcher.model.diffusion_model = dm
    return patcher


def test_a_family_that_owns_a_census_name_on_its_forward_path_keeps_it_sharded(
    fully_shard_recorder,
):
    """LTXV's forward calls `caption_projection`, so replicating it for LTXV
    would hold the whole module on every rank for no gain; LTXAV, which
    reaches it from `extra_conds`, still replicates it. The comment above
    FORWARD_PATH_NAME_OWNERS in adapters/fsdp_islands.py gives the reason."""
    from dgx_monarch.adapters.fsdp_islands import FORWARD_PATH_NAME_OWNERS

    assert FORWARD_PATH_NAME_OWNERS["LTXV"] == ("caption_projection",)
    for class_name, replicated in (("LTXV", False), ("LTXAV", True)):
        fully_shard_recorder.calls.clear()
        fully_shard_recorder.kwargs.clear()
        dm = _fake_diffusion_model({"transformer_blocks": 2})
        dm.caption_projection = torch.nn.Linear(4, 4).to(torch.bfloat16)
        patcher = _patcher_for_comfy_class(class_name, dm)

        apply_fsdp_capacity_mode(patcher, "bf16", None, _bf16_precision(patcher))

        projection = set(dm.caption_projection.parameters())
        for kwargs in fully_shard_recorder.kwargs:
            ignored = kwargs.get("ignored_params", set())
            assert (ignored == projection) is replicated, class_name
        # Sharded for LTXV means the block wraps still ran and the projection
        # was left to the root wrap, not that the model went unwrapped.
        assert fully_shard_recorder.calls == [*dm.transformer_blocks, dm], class_name


def test_a_declared_outside_forward_name_outranks_an_inherited_exclusion():
    """The exclusion is read down the whole class chain, so a family that
    subclasses an owner and does reach the module from ``extra_conds`` must
    keep it replicated. Declaring wins over owning, whatever the order."""
    from dgx_monarch.adapters.fsdp_islands import (
        AUXILIARY_MODULE_ATTRS,
        OUTSIDE_FORWARD_MODULE_ATTRS,
        outside_forward_module_attrs,
    )

    owner = type("LTXV", (), {})
    assert "caption_projection" not in outside_forward_module_attrs(owner())
    assert outside_forward_module_attrs(None) == AUXILIARY_MODULE_ATTRS
    assert outside_forward_module_attrs(type("Unread", (), {})()) == (
        AUXILIARY_MODULE_ATTRS)
    for name in OUTSIDE_FORWARD_MODULE_ATTRS:
        declaring = type(name, (owner,), {})
        kept = outside_forward_module_attrs(declaring())
        assert set(OUTSIDE_FORWARD_MODULE_ATTRS[name]) <= set(kept), name


def test_the_ignored_set_is_islands_and_outside_forward_modules_and_scalars(
    fully_shard_recorder,
):
    """The ignored set is the union of three sources, and one model can
    present all three: fp32 islands (H3's projection islands), modules run
    outside the forward (H3's text ingress), and zero-dimensional parameters,
    which FSDP2 refuses outright (PiD's ``log_alpha``)."""
    from dgx_monarch.adapters.fsdp_islands import FP32_ISLANDS_PROFILE

    dm = _fake_diffusion_model({"blocks": 2})
    # A bf16 core large enough that the island sits inside the 5% ceiling.
    dm.core = torch.nn.Parameter(
        torch.zeros(4096, dtype=torch.bfloat16), requires_grad=False)
    dm.condition_proj = torch.nn.Linear(4, 4).to(torch.bfloat16)
    dm.token_refiner = torch.nn.Linear(4, 4).to(torch.bfloat16)
    dm.video_patch_proj = torch.nn.Parameter(
        torch.ones(4, dtype=torch.float32), requires_grad=False)
    dm.log_alpha = torch.nn.Parameter(
        torch.tensor(0.5, dtype=torch.bfloat16), requires_grad=False)
    patcher = _FakePatcher(dm)

    apply_fsdp_capacity_mode(patcher, "bf16", None, LivePrecisionEvidence(
        quant_kind="bf16",
        live_dtype_profile=FP32_ISLANDS_PROFILE,
        checkpoint_kind="bf16",
        auxiliary_parameter_count=1,
        auxiliary_parameter_bytes=16,
    ))

    expected = (
        {dm.video_patch_proj, dm.log_alpha}
        | set(dm.condition_proj.parameters())
        | set(dm.token_refiner.parameters())
    )
    assert len(expected) == 6
    for kwargs in fully_shard_recorder.kwargs:
        assert kwargs["ignored_params"] == expected


def test_replicated_auxiliary_bytes_ride_inside_the_flat_shard_build_fraction():
    """The price reads a file size and a world, never a module census, so
    replicating a family's outside-forward modules has to fit the flat
    transient fraction (capacity_fit.py). Measured from the checkpoint headers
    2026-09-09. The build's in-flight chunk is freed before comfy moves the
    replicated modules to the device, so the fraction covers the larger of the
    two, not their sum."""
    from dgx_monarch.capacity_fit import (
        ABSOLUTE_HOST_FLOOR_BYTES,
        FSDP_STREAM_TRANSIENT_FRACTION,
        fsdp_required_bytes,
    )

    allocator_slack = 0.0165  # flux2 2026-08-26: 0.047 measured over a 0.0305 block
    for name, size_gib, auxiliary_gib, block_fraction in (
        ("minimax_h3 bf16", 61.728, 1.487, 0.0195),
        ("ltx 2.5 bf16", 39.132, 3.755, 0.0184),
    ):
        replicated = (1 - 1 / 2) * auxiliary_gib / size_gib
        assert max(replicated, block_fraction + allocator_slack) < (
            FSDP_STREAM_TRANSIENT_FRACTION), name
    # The completed 2026-09-09 minimax_h3 bf16 rerun (world 2, slab and Gate
    # off) read 31.6 GiB of weights per rank: half the file, plus the half of
    # 1.487 GiB a rank holds whole instead of sharded.
    assert round(61.728 / 2 + 1.487 / 2, 1) == 31.6
    size = int(61.728 * 2**30)
    assert fsdp_required_bytes(size, 2, "bf16") == (
        int(size * (0.5 + FSDP_STREAM_TRANSIENT_FRACTION))
        + ABSOLUTE_HOST_FLOOR_BYTES)


def test_fp32_islands_are_ignored_not_sharded(fully_shard_recorder):
    """Every fp32 parameter is replicated through ``ignored_params``."""
    from dgx_monarch.adapters.fsdp_islands import FP32_ISLANDS_PROFILE

    dm = _fake_diffusion_model({"blocks": 2})
    # A bf16 core large enough that 16 bytes of fp32 sit inside the 5% ceiling.
    dm.core = torch.nn.Parameter(
        torch.zeros(4096, dtype=torch.bfloat16), requires_grad=False)
    # A scale-shift table inside a block and one at the root, LTX-style.
    dm.blocks[0].scale_shift = torch.nn.Parameter(
        torch.zeros(2, dtype=torch.float32), requires_grad=False)
    dm.scale_shift_table = torch.nn.Parameter(
        torch.zeros(2, dtype=torch.float32), requires_grad=False)
    patcher = _FakePatcher(dm)
    evidence = LivePrecisionEvidence(
        quant_kind="bf16",
        live_dtype_profile=FP32_ISLANDS_PROFILE,
        auxiliary_parameter_count=2,
        auxiliary_parameter_bytes=16,
        checkpoint_kind="bf16",
    )

    apply_fsdp_capacity_mode(patcher, "bf16", None, evidence)

    islands = {dm.blocks[0].scale_shift, dm.scale_shift_table}
    assert fully_shard_recorder.calls == [*dm.blocks, dm]
    for kwargs in fully_shard_recorder.kwargs:
        assert kwargs["ignored_params"] == islands
    assert patcher._dgxm_fsdp is True


def test_scalar_parameters_are_ignored_not_sharded(fully_shard_recorder):
    """FSDP2 refuses a scalar outright, so every 0-dim parameter is replicated.

    The PiD 1024-to-4096 checkpoint carries a zero-dimensional ``log_alpha``,
    and on 2026-09-03 the direct wrap crashed its load untyped with
    "fully_shard doesn't support scalar parameters. Change log_alpha to a 1D
    tensor with numel equal to 1". A scalar has no dim 0 to chunk, so it joins
    ``ignored_params`` on every wrap and keeps the value and device it loaded
    with.
    """
    dm = _fake_diffusion_model({"blocks": 2})
    dm.log_alpha = torch.nn.Parameter(
        torch.tensor(0.5, dtype=torch.bfloat16), requires_grad=False)
    # A scalar inside a block is ignored on the block wrap too, not just at root.
    dm.blocks[0].gate = torch.nn.Parameter(
        torch.tensor(-1.25, dtype=torch.bfloat16), requires_grad=False)
    patcher = _FakePatcher(dm)

    apply_fsdp_capacity_mode(patcher, "bf16", None, _bf16_precision(patcher))

    scalars = {dm.log_alpha, dm.blocks[0].gate}
    assert fully_shard_recorder.calls == [*dm.blocks, dm]
    for kwargs in fully_shard_recorder.kwargs:
        assert scalars <= kwargs["ignored_params"]
    # Value and device survive the wrap untouched: still plain Parameters,
    # still the numbers the checkpoint loaded.
    assert dm.log_alpha.ndim == 0 and dm.blocks[0].gate.ndim == 0
    assert float(dm.log_alpha) == 0.5
    assert float(dm.blocks[0].gate) == -1.25
    assert dm.log_alpha.device.type == "cpu"
    assert dm.blocks[0].gate.device.type == "cpu"
    assert patcher._dgxm_fsdp is True


def test_the_census_and_the_wrap_agree_about_the_scalar(fully_shard_recorder):
    """The bytes census counts every ignored scalar in full, on every rank.

    ``patcher.size`` is one of the numbers comfy's planner prices the load by,
    and ``apply_fsdp_capacity_mode`` rebuilds it from ``local_nbytes`` over the
    live parameters. A replicated scalar is not divided by world, so the
    census must carry its whole ``element_size()``; the streaming shard build
    must leave the same parameter alone. This test pins wrap, build and census
    to one another rather than to a number.
    """
    from dgx_monarch.adapters.fsdp_quant import local_nbytes
    from dgx_monarch.adapters.fsdp_shard_build import build_shards

    dm = _fake_diffusion_model({"blocks": 2})
    dm.log_alpha = torch.nn.Parameter(
        torch.tensor(0.5, dtype=torch.bfloat16), requires_grad=False)
    patcher = _FakePatcher(dm)

    apply_fsdp_capacity_mode(patcher, "bf16", None, _bf16_precision(patcher))

    ignored = fully_shard_recorder.kwargs[-1]["ignored_params"]
    assert dm.log_alpha in ignored
    # The census: every live parameter's own bytes, scalar included at full size.
    assert patcher.size == sum(local_nbytes(p) for p in dm.parameters())
    assert local_nbytes(dm.log_alpha) == dm.log_alpha.element_size()

    # The build agrees: handed the wrap's own ignored set, it materializes
    # nothing for the scalar and returns it unchanged.
    before = dm.log_alpha
    materialized = build_shards(dm, lambda: None, ignored)
    assert materialized == 0  # the stubbed wrap produced no DTensors
    assert dm.log_alpha is before
    assert float(dm.log_alpha) == 0.5


def test_an_fp32_scalar_is_ignored_once_not_counted_twice(fully_shard_recorder):
    """An fp32 scalar is both an island and a scalar: ignored once, counted once."""
    from dgx_monarch.adapters.fsdp_islands import FP32_ISLANDS_PROFILE

    dm = _fake_diffusion_model({"blocks": 2})
    dm.core = torch.nn.Parameter(
        torch.zeros(4096, dtype=torch.bfloat16), requires_grad=False)
    dm.log_alpha = torch.nn.Parameter(
        torch.tensor(0.5, dtype=torch.float32), requires_grad=False)
    patcher = _FakePatcher(dm)
    evidence = LivePrecisionEvidence(
        quant_kind="bf16",
        live_dtype_profile=FP32_ISLANDS_PROFILE,
        auxiliary_parameter_count=1,
        auxiliary_parameter_bytes=4,
        checkpoint_kind="bf16",
    )

    apply_fsdp_capacity_mode(patcher, "bf16", None, evidence)

    for kwargs in fully_shard_recorder.kwargs:
        assert kwargs["ignored_params"] == {dm.log_alpha}
    assert patcher.size == sum(
        p.numel() * p.element_size() for p in dm.parameters())


def test_uniform_fp16_core_is_admitted(fully_shard_recorder):
    from dgx_monarch.adapters.fsdp_islands import ALL_FP16_PROFILE

    dm = _fake_diffusion_model({"blocks": 1}, dtype=torch.float16)
    patcher = _FakePatcher(dm)
    assert _detect_quant_kind(patcher, "bf16") == "fp16"
    evidence = LivePrecisionEvidence(
        quant_kind="fp16",
        live_dtype_profile=ALL_FP16_PROFILE,
        checkpoint_kind="fp16",
    )

    apply_fsdp_capacity_mode(patcher, "fp16", None, evidence)

    assert fully_shard_recorder.calls == [*dm.blocks, dm]
    assert all("ignored_params" not in kwargs for kwargs in fully_shard_recorder.kwargs)
    assert patcher._dgxm_fsdp is True


def test_mixed_frozen_dtypes_wrap_and_islands_stay_plain_under_real_fsdp(tmp_path):
    """Canary for three torch facts, on a one-rank gloo group.

    First, FSDP2's "uniform original parameter dtype" assertion covers
    trainable parameters only: a frozen block mixing bf16 and fp32 wraps and
    runs. Second, ``ignored_params`` leaves a parameter as a plain tensor on
    the module while its siblings become DTensors, and the forward still runs.
    Third, a scalar parameter is refused outright unless it is ignored, which
    the scalar replication in apply_fsdp_capacity_mode relies on.
    """
    import torch.nn as nn
    from torch.distributed.device_mesh import init_device_mesh

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8, dtype=torch.bfloat16)
            self.scale_shift = nn.Parameter(torch.ones(8, dtype=torch.float32))

        def forward(self, x):
            return self.lin(x) * self.scale_shift.to(x.dtype)

    class _Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([_Block(), _Block()])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_path / 'process_group'}",
        rank=0,
        world_size=1,
    )
    try:
        mesh = init_device_mesh("cpu", (1,))
        x = torch.randn(2, 8, dtype=torch.bfloat16)

        mixed = _Root()
        for p in mixed.parameters():
            p.requires_grad = False
        for block in mixed.blocks:
            torch.distributed.fsdp.fully_shard(block, mesh=mesh)
        torch.distributed.fsdp.fully_shard(mixed, mesh=mesh, reshard_after_forward=True)
        with torch.no_grad():
            mixed(x)  # would raise the uniform-dtype assertion if trainable
        assert type(mixed.blocks[0].scale_shift).__name__ == "DTensor"

        ignored = _Root()
        for p in ignored.parameters():
            p.requires_grad = False
        islands = {block.scale_shift for block in ignored.blocks}
        for block in ignored.blocks:
            torch.distributed.fsdp.fully_shard(block, mesh=mesh, ignored_params=islands)
        torch.distributed.fsdp.fully_shard(
            ignored, mesh=mesh, reshard_after_forward=True, ignored_params=islands)
        with torch.no_grad():
            out = ignored(x)
        assert out.shape == (2, 8)
        assert type(ignored.blocks[0].scale_shift).__name__ == "Parameter"
        assert type(ignored.blocks[0].lin.weight).__name__ == "DTensor"

        # The third torch fact: real fully_shard refuses a scalar outright, and
        # ``ignored_params`` lets one through: _fsdp_init._get_managed_states
        # skips the ignored set before it calls _verify_managed_param. That
        # order is the whole fix, so both directions are asserted, and the fix
        # fails if the filter moves after the check.
        class _Scalared(_Root):
            def __init__(self):
                super().__init__()
                self.log_alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.bfloat16))

            def forward(self, x):
                return super().forward(x) * self.log_alpha.to(x.dtype)

        crashes = _Scalared()
        for p in crashes.parameters():
            p.requires_grad = False
        with pytest.raises(ValueError, match="scalar parameters"):
            torch.distributed.fsdp.fully_shard(crashes, mesh=mesh)

        admitted = _Scalared()
        for p in admitted.parameters():
            p.requires_grad = False
        keep = {block.scale_shift for block in admitted.blocks} | {admitted.log_alpha}
        for block in admitted.blocks:
            torch.distributed.fsdp.fully_shard(block, mesh=mesh, ignored_params=keep)
        torch.distributed.fsdp.fully_shard(
            admitted, mesh=mesh, reshard_after_forward=True, ignored_params=keep)
        with torch.no_grad():
            assert admitted(x).shape == (2, 8)
        assert type(admitted.log_alpha).__name__ == "Parameter"
        assert admitted.log_alpha.ndim == 0
        assert float(admitted.log_alpha) == 0.5
        assert admitted.log_alpha.device.type == "cpu"
    finally:
        torch.distributed.destroy_process_group()


def test_an_outside_forward_module_crashes_under_real_fsdp_until_it_is_ignored(tmp_path):
    """The 2026-09-09 minimax_h3 crash and its fix, on a one-rank gloo group.

    The stand-in carries H3's two text-ingress modules and a method that runs
    them the way ``preprocess_text_embeds`` does. Sharded by the root wrap, the
    call raises the production error. In the ignored set the same call
    returns, the weights stay plain Parameters, and the blocks are still
    DTensors.
    """
    import torch.nn as nn
    from torch.distributed.device_mesh import init_device_mesh

    if torch.distributed.is_initialized():
        pytest.skip("a process group is already initialized in this session")

    class _Ingress(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(
                [nn.Linear(8, 8, dtype=torch.bfloat16) for _ in range(2)])
            self.condition_proj = nn.Linear(8, 8, dtype=torch.bfloat16)
            self.token_refiner = nn.Linear(8, 8, dtype=torch.bfloat16)

        def preprocess_text_embeds(self, text_states):
            return self.token_refiner(self.condition_proj(text_states))

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    def wrap(model, ignored):
        for parameter in model.parameters():
            parameter.requires_grad = False
        extra = {"ignored_params": ignored} if ignored else {}
        for block in model.blocks:
            torch.distributed.fsdp.fully_shard(block, mesh=mesh, **extra)
        torch.distributed.fsdp.fully_shard(
            model, mesh=mesh, reshard_after_forward=True, **extra)

    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_path / 'process_group'}",
        rank=0,
        world_size=1,
    )
    try:
        mesh = init_device_mesh("cpu", (1,))
        text = torch.randn(2, 8, dtype=torch.bfloat16)

        crashes = _Ingress()
        wrap(crashes, None)
        assert type(crashes.condition_proj.weight).__name__ == "DTensor"
        with pytest.raises(RuntimeError, match=r"mixed torch\.Tensor and DTensor"):
            with torch.no_grad():
                crashes.preprocess_text_embeds(text)

        admitted = _Ingress()
        wrap(admitted, auxiliary_module_parameters(admitted))
        assert type(admitted.condition_proj.weight).__name__ == "Parameter"
        assert type(admitted.token_refiner.weight).__name__ == "Parameter"
        assert type(admitted.blocks[0].weight).__name__ == "DTensor"
        with torch.no_grad():
            assert admitted.preprocess_text_embeds(text).shape == (2, 8)
            assert admitted(text).shape == (2, 8)
    finally:
        torch.distributed.destroy_process_group()


def test_llm_adapter_guard_does_not_fire_when_absent(fully_shard_recorder):
    # A model with no llm_adapter, the common case, still wraps.
    dm = _fake_diffusion_model({"blocks": 1})
    patcher = _FakePatcher(dm)

    apply_fsdp_capacity_mode(patcher, "bf16", None, _bf16_precision(patcher))

    assert patcher._dgxm_fsdp is True


def test_llm_adapter_guard_checked_before_quant_and_lora_are_moot():
    # The quant launch check is the first line of apply_fsdp_capacity_mode, so
    # an unsupported kind refuses before the model, llm_adapter included, is
    # read.
    dm = _fake_diffusion_model({"blocks": 1})
    dm.llm_adapter = torch.nn.Module()
    patcher = _FakePatcher(dm)

    with pytest.raises(UnsupportedModelError, match="bf16"):
        apply_fsdp_capacity_mode(patcher, "nvfp4", None)


@pytest.mark.parametrize(("dtype", "expected_kind"), [
    (torch.float32, "fp32"),
    (torch.float64, "unknown"),
])
def test_explicit_bf16_live_kind_guard_runs_before_sharding(
    fully_shard_recorder,
    dtype,
    expected_kind,
):
    dm = _fake_diffusion_model({"blocks": 1}, dtype=dtype)
    patcher = _FakePatcher(dm)
    # ``bf16`` is the option-derived kind for an explicit BF16 request. Live
    # inspection remains authoritative and must not turn that request into an
    # FSDP grant when Comfy produced another dtype.
    quant_kind = _detect_quant_kind(patcher, "bf16")

    assert quant_kind == expected_kind
    with pytest.raises(
        UnsupportedModelError,
        match=rf"supports bf16, fp16, fp8 and int8 checkpoints at launch \(got quant={expected_kind}\)",
    ):
        apply_fsdp_capacity_mode(patcher, quant_kind, None)

    assert fully_shard_recorder.calls == []
    assert not hasattr(patcher, "_dgxm_fsdp")


@pytest.mark.parametrize(
    ("quant_kind", "remediation"),
    [
        ("fp32", "only as islands inside a bf16 core"),
        ("nvfp4", "no registered shard layout"),
        ("mxfp8", "no registered shard layout"),
        ("unknown", "could not prove the checkpoint's precision"),
    ],
)
def test_fsdp_quant_validator_rejects_non_bf16(quant_kind, remediation):
    with pytest.raises(
        UnsupportedModelError,
        match=rf"supports bf16, fp16, fp8 and int8 checkpoints at launch \(got quant={quant_kind}\)",
    ) as exc_info:
        validate_fsdp_launch_quant(quant_kind)
    assert remediation in str(exc_info.value)


@pytest.mark.parametrize(
    ("profile", "count", "size", "checkpoint", "expected"),
    [
        pytest.param(ALL_BF16_PROFILE, 0, 0, "bf16", True, id="all-bf16"),
        pytest.param(
            WAN_FP32_INGRESS_PROFILE, 2, 8, "bf16", True, id="audited-wan"
        ),
        pytest.param(
            ALL_BF16_PROFILE, False, 0, "bf16", False, id="boolean-count"
        ),
        pytest.param(
            WAN_FP32_INGRESS_PROFILE, 2, 0, "bf16", False, id="empty-auxiliary"
        ),
        pytest.param(
            WAN_FP32_INGRESS_PROFILE,
            2,
            4 * 1024 * 1024 + 1,
            "bf16",
            False,
            id="oversized-auxiliary",
        ),
        pytest.param(
            WAN_FP32_INGRESS_PROFILE, 2, 8, "fp16", False, id="wrong-checkpoint"
        ),
        pytest.param(
            "unreviewed_mixed_precision", 0, 0, "bf16", False, id="unknown-profile"
        ),
        # An fp16 file that Comfy materializes as bf16 on this hardware: the
        # live core is bf16, the row keeps saying which file it came from.
        pytest.param(ALL_BF16_PROFILE, 0, 0, "fp16", True, id="bf16-core-from-fp16-file"),
        pytest.param(ALL_BF16_PROFILE, 0, 0, "fp8", False, id="bf16-core-from-quant-file"),
    ],
)
def test_adapter_owned_fsdp_precision_profile_admission_is_exact(
    profile,
    count,
    size,
    checkpoint,
    expected,
):
    assert fsdp_precision_profile_is_admitted(
        profile,
        count,
        size,
        checkpoint,
    ) is expected


def test_fsdp_accepts_detector_owned_wan_bf16_core_profile(
    monkeypatch,
    fully_shard_recorder,
):
    from dgx_monarch.actor import store_detect

    dm = _fake_diffusion_model({"blocks": 1}, dtype=torch.bfloat16)
    patcher = _FakePatcher(dm)
    evidence = LivePrecisionEvidence(
        quant_kind="bf16",
        live_dtype_profile=WAN_FP32_INGRESS_PROFILE,
        auxiliary_parameter_count=2,
        auxiliary_parameter_bytes=8,
        checkpoint_kind="bf16",
    )
    monkeypatch.setattr(
        store_detect,
        "_detect_live_precision",
        lambda *_args: evidence.bind_checkpoint(None, "bf16"),
    )

    apply_fsdp_capacity_mode(
        patcher,
        "bf16",
        None,
        evidence,
    )

    assert patcher._dgxm_fsdp is True
    assert patcher._dgxm_fsdp_ready is True
    assert fully_shard_recorder.calls


def test_worker_runs_one_live_precision_scan_after_preparation(
    monkeypatch,
    fully_shard_recorder,
):
    from dgx_monarch.actor import store_detect

    patcher = _FakePatcher(
        _fake_diffusion_model({"blocks": 1}, dtype=torch.bfloat16)
    )
    evidence = _bf16_precision(patcher)
    scans = []

    def detect(observed_patcher, option_kind):
        scans.append((observed_patcher, option_kind))
        return evidence.bind_checkpoint(None, "bf16")

    monkeypatch.setattr(store_detect, "_detect_live_precision", detect)
    monkeypatch.setattr(adapters, "get_adapter", lambda _model: object())
    monkeypatch.setattr(worker_module, "_maybe_compile_dit", lambda _model: None)
    worker = types.SimpleNamespace(
        topology={"fsdp": True, "ulysses": 1, "ring": 1, "cfg": 1},
    )

    GPUWorker._inject_for_topology(worker, patcher, "bf16", None, evidence)

    assert scans == [(patcher, "bf16")]
    assert patcher._dgxm_fsdp_ready is True
    assert fully_shard_recorder.calls


def test_worker_live_scan_observes_same_patcher_model_replacement(
    monkeypatch,
    fully_shard_recorder,
):
    patcher = _FakePatcher(
        _fake_diffusion_model({"blocks": 1}, dtype=torch.bfloat16)
    )
    evidence = _bf16_precision(patcher)
    replacement = _fake_diffusion_model(
        {"blocks": 1},
        dtype=torch.float32,
    )

    class ReplacingAdapter:
        def inject_usp(self, _diffusion_model, _ctx):
            patcher.model.diffusion_model = replacement

    monkeypatch.setattr(adapters, "get_adapter", lambda _model: ReplacingAdapter())
    monkeypatch.setattr(worker_module, "_maybe_compile_dit", lambda _model: None)
    worker = types.SimpleNamespace(
        topology={"fsdp": True, "ulysses": 2, "ring": 1, "cfg": 1},
        _attn=types.SimpleNamespace(bind_capability=lambda *_a: None, effective_kernel="TORCH_FLASH"),
    )

    with pytest.raises(UnsupportedModelError, match="no longer matches"):
        GPUWorker._inject_for_topology(
            worker,
            patcher,
            "bf16",
            None,
            evidence,
        )

    assert fully_shard_recorder.calls == []
    assert all(
        parameter.requires_grad
        for parameter in patcher.model.diffusion_model.parameters()
    )
    assert not hasattr(patcher, "_dgxm_fsdp_ready")


def test_worker_live_scan_observes_dtype_mutation_during_compile_preparation(
    monkeypatch,
    fully_shard_recorder,
):
    patcher = _FakePatcher(
        _fake_diffusion_model({"blocks": 1}, dtype=torch.bfloat16)
    )
    evidence = _bf16_precision(patcher)
    monkeypatch.setattr(adapters, "get_adapter", lambda _model: object())
    monkeypatch.setattr(
        worker_module,
        "_maybe_compile_dit",
        lambda model: model.to(dtype=torch.float32),
    )
    worker = types.SimpleNamespace(
        topology={"fsdp": True, "ulysses": 1, "ring": 1, "cfg": 1},
    )

    with pytest.raises(UnsupportedModelError, match="no longer matches"):
        GPUWorker._inject_for_topology(
            worker,
            patcher,
            "bf16",
            None,
            evidence,
        )

    assert fully_shard_recorder.calls == []
    assert all(
        parameter.requires_grad
        for parameter in patcher.model.diffusion_model.parameters()
    )
    assert not hasattr(patcher, "_dgxm_fsdp_ready")


def test_fsdp_rejects_fabricated_wan_profile_before_sharding(
    fully_shard_recorder,
):
    dm = _fake_diffusion_model({"blocks": 1}, dtype=torch.bfloat16)
    patcher = _FakePatcher(dm)
    fabricated = LivePrecisionEvidence(
        quant_kind="bf16",
        live_dtype_profile=WAN_FP32_INGRESS_PROFILE,
        auxiliary_parameter_count=2,
        auxiliary_parameter_bytes=8,
        checkpoint_kind="bf16",
    )

    with pytest.raises(UnsupportedModelError, match="no longer matches"):
        apply_fsdp_capacity_mode(patcher, "bf16", None, fabricated)

    assert fully_shard_recorder.calls == []
    assert not hasattr(patcher, "_dgxm_fsdp_ready")


def test_fsdp_lora_validator_rejects_nonempty_stack():
    with pytest.raises(UnsupportedModelError, match="needs lora_low_rss on"):
        validate_fsdp_launch_loras(("adapter.safetensors",), lora_low_rss=False)
    # Under lora_low_rss the shard-aware bake carries the stack.
    validate_fsdp_launch_loras(("adapter.safetensors",), lora_low_rss=True)


def test_worker_lora_guard_runs_before_any_sharding(fully_shard_recorder):
    patcher = _FakePatcher(_fake_diffusion_model({"blocks": 1}))

    with pytest.raises(UnsupportedModelError, match="needs lora_low_rss on"):
        apply_fsdp_capacity_mode(
            patcher,
            "bf16",
            [{"name": "adapter.safetensors", "strength": 0.5}],
            lora_low_rss=False,
        )

    assert fully_shard_recorder.calls == []
    assert not hasattr(patcher, "_dgxm_fsdp")


# LoRA checkpoint admission. The lever check above asks whether lora_low_rss
# is on; these ask whether the checkpoint's own bytes admit a LoRA bake. A
# checkpoint that fails refuses even with the lever on, and the card does not
# blame the lever.
# Where the evidence's shape matters it comes from the real detector run on a
# fake model shaped like the checkpoint: validate_fsdp_live_precision's rescan
# rejects a hand-built one that does not match ("no longer matches") before
# the admission check runs.

def test_worker_lora_admission_refuses_quantized_shards_even_with_the_lever_on(
    fully_shard_recorder,
):
    dm = _fake_diffusion_model({"blocks": 1})
    dm.blocks[0].quant_format = "float8_e4m3fn"  # comfy-kitchen wrapper marker
    patcher = _FakePatcher(dm)
    evidence = _detect_live_precision(patcher, "bf16").bind_checkpoint("fp8", "fp8")
    assert evidence.quantized_shards is True

    with pytest.raises(UnsupportedModelError, match="not admitted for this checkpoint"):
        apply_fsdp_capacity_mode(
            patcher,
            "fp8",
            [{"name": "adapter.safetensors", "strength": 0.5}],
            evidence,
            lora_low_rss=True,  # the operator turned it on; the card must not blame it
        )

    assert fully_shard_recorder.calls == []
    assert not hasattr(patcher, "_dgxm_fsdp")


def test_worker_lora_admission_admits_fp32_islands_pending_the_in_bake_check(
    fully_shard_recorder,
):
    """The live-evidence backstop does not pre-refuse a fp32-islands profile.
    No header or coarse live signal can tell an island that stays live fp32,
    safe to bake, from one that casts, a live/file mismatch that
    actor/fsdp_lora.py's in-bake _file_tensor check refuses. The module
    docstring of adapters/fsdp_lora_admission.py names a checkpoint of each."""
    from dgx_monarch.adapters.fsdp_islands import FP32_ISLANDS_PROFILE

    dm = _fake_diffusion_model({"blocks": 50})  # a bf16 core big enough that
    dm.island = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))  # one fp32 tensor fits the 5% ceiling
    patcher = _FakePatcher(dm)
    evidence = _detect_live_precision(patcher, "bf16").bind_checkpoint("bf16", "bf16")
    assert evidence.live_dtype_profile == FP32_ISLANDS_PROFILE

    apply_fsdp_capacity_mode(
        patcher,
        "bf16",
        [{"name": "adapter.safetensors", "strength": 0.5}],
        evidence,
        lora_low_rss=True,
    )

    assert patcher._dgxm_fsdp is True


def test_worker_lora_admission_admits_a_plain_fp8_dtype_checkpoint(fully_shard_recorder, monkeypatch):
    """Qwen-Image's literal float8_e4m3fn weights: no comfy-kitchen wrapper."""
    from dgx_monarch.adapters import fsdp_quant

    # A venv without comfy_kitchen (CI's) cannot import the real class, and plain
    # fp8 weights are never QuantizedTensors, so the lookup gets a stand-in. The
    # wrapper class is never built: one built on the stand-in would outlive this
    # test in fsdp_quant's module cache and break every later user of it.
    monkeypatch.setattr(fsdp_quant, "_quantized_tensor_class", lambda: type("NoQuantizedTensor", (), {}))
    monkeypatch.setattr(fsdp_quant, "sharded_quant_class", lambda: None)
    dm = _fake_diffusion_model({"blocks": 1}, dtype=torch.float8_e4m3fn)
    patcher = _FakePatcher(dm)
    evidence = _detect_live_precision(patcher, "bf16").bind_checkpoint("fp8", "fp8")
    assert evidence.quant_kind == "fp8"
    assert evidence.quantized_shards is False

    apply_fsdp_capacity_mode(
        patcher,
        "fp8",
        [{"name": "adapter.safetensors", "strength": 0.5}],
        evidence,
        lora_low_rss=True,
    )

    assert patcher._dgxm_fsdp is True


def test_fsdp_capacity_mode_pins_offload_to_the_load_device(fully_shard_recorder):
    """A hot-swap clone inherits the pin, so comfy's uuid-mismatch unpatch moves no shard."""
    patcher = _FakePatcher(_fake_diffusion_model({"blocks": 1}))
    patcher.load_device = torch.device("cuda", 0)
    patcher.offload_device = torch.device("cpu")

    apply_fsdp_capacity_mode(patcher, "bf16", None, _bf16_precision(patcher))

    assert patcher._dgxm_fsdp_ready is True
    assert patcher.offload_device == torch.device("cuda", 0)


def test_worker_lora_admission_admits_a_uniform_bf16_checkpoint(fully_shard_recorder):
    patcher = _FakePatcher(_fake_diffusion_model({"blocks": 1}))

    apply_fsdp_capacity_mode(
        patcher,
        "bf16",
        [{"name": "adapter.safetensors", "strength": 0.5}],
        _bf16_precision(patcher),
        lora_low_rss=True,
    )

    assert patcher._dgxm_fsdp is True


@pytest.mark.parametrize(
    ("quant_kind", "lora_stack", "match"),
    [
        ("nvfp4", None, "supports bf16, fp16, fp8 and int8 checkpoints"),
        ("bf16", [{"name": "adapter.safetensors"}], "LoRA on FSDP-wrapped"),
    ],
)
def test_worker_launch_guard_precedes_adapter_or_compile_mutation(
    monkeypatch,
    quant_kind,
    lora_stack,
    match,
):
    events = []

    def forbidden(name):
        def call(*_args, **_kwargs):
            events.append(name)
            pytest.fail(f"{name} ran before the FSDP launch guard")

        return call

    monkeypatch.setattr(adapters, "get_adapter", forbidden("adapter injection"))
    monkeypatch.setattr(worker_module, "_maybe_compile_dit", forbidden("compile"))
    monkeypatch.setattr(
        fsdp_module,
        "apply_fsdp_capacity_mode",
        forbidden("sharding"),
    )
    worker = types.SimpleNamespace(
        topology={"fsdp": True}, store=types.SimpleNamespace(lora_low_rss=False))

    with pytest.raises(UnsupportedModelError, match=match):
        GPUWorker._inject_for_topology(
            worker,
            types.SimpleNamespace(model=object()),
            quant_kind,
            lora_stack,
            None,
        )
    assert events == []


def test_lora_loader_rejects_explicit_fsdp_when_the_graph_turns_low_rss_off():
    model = types.SimpleNamespace(
        mesh=types.SimpleNamespace(
            topology_preset="uly2+fsdp",
            world=2,
            worker_args={"lora_low_rss": False},
        )
    )

    with pytest.raises(UnsupportedModelError, match="needs lora_low_rss on"):
        loaders.DGXMonarchLoraLoader().load_lora(
            model,
            "adapter.safetensors",
            0.5,
        )


def test_lora_loader_passes_explicit_fsdp_through_when_the_worker_decides():
    """No graph override: the cluster policy and the worker's check decide."""
    seen = []
    model = types.SimpleNamespace(
        mesh=types.SimpleNamespace(topology_preset="uly2+fsdp", world=2, worker_args={}),
        with_lora=lambda name, strength: seen.append((name, strength)) or "spec",
    )
    assert loaders.DGXMonarchLoraLoader().load_lora(model, "adapter.safetensors", 0.5) == ("spec",)
    assert seen == [("adapter.safetensors", 0.5)]


def _write_fsdp_dtype_checkpoint(path, dtype: str) -> None:
    width = {"BF16": 2, "F8_E4M3": 1}[dtype]
    header = json.dumps({
        "weight": {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [0, width],
        }
    }).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(width))


def test_lora_loader_rejects_quantized_shards_even_with_low_rss_on(
    tmp_path, monkeypatch,
):
    """With lora_low_rss on in the graph, this checkpoint's own property still
    refuses, on a card that does not tell the operator to change the lever."""
    path = tmp_path / "chroma_scaled_fp8.safetensors"
    header = json.dumps({
        "scaled_fp8": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]},
        "weight": {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [1, 2]},
    }).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(2))

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: str(path)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    model = types.SimpleNamespace(
        unet_name="chroma_scaled_fp8.safetensors",
        mesh=types.SimpleNamespace(
            topology_preset="uly2+fsdp",
            world=2,
            worker_args={"lora_low_rss": True},
        ),
    )

    with pytest.raises(
        UnsupportedModelError, match="not admitted for this checkpoint",
    ) as raised:
        loaders.DGXMonarchLoraLoader().load_lora(model, "adapter.safetensors", 0.5)

    assert "needs lora_low_rss on" not in str(raised.value)


def test_lora_loader_admits_fp32_islands_pending_the_in_bake_check(
    tmp_path, monkeypatch,
):
    """The driver-side header check does not cover fp32-islands checkpoints
    (adapters/detect.fsdp_lora_admission_property and
    docs/TROUBLESHOOTING.md #17 say why), so this checkpoint passes to the
    graph's own lever check. Whether the load then bakes or refuses typed in
    actor/fsdp_lora.py's in-bake check depends on architecture-specific
    live-cast behaviour this preflight cannot see."""
    path = tmp_path / "krea2_raw.safetensors"
    core_elements = 1000
    core_bytes = 2 * core_elements
    header = json.dumps({
        "core": {"dtype": "BF16", "shape": [core_elements],
                 "data_offsets": [0, core_bytes]},
        "island": {"dtype": "F32", "shape": [1],
                   "data_offsets": [core_bytes, core_bytes + 4]},
    }).encode()
    path.write_bytes(
        struct.pack("<Q", len(header)) + header + bytes(core_bytes + 4))

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: str(path)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    seen = []
    model = types.SimpleNamespace(
        unet_name="krea2_raw.safetensors",
        mesh=types.SimpleNamespace(
            topology_preset="cfg2+fsdp",
            world=2,
            worker_args={"lora_low_rss": True},
        ),
        with_lora=lambda name, strength: seen.append((name, strength)) or "spec",
    )

    assert loaders.DGXMonarchLoraLoader().load_lora(
        model, "adapter.safetensors", 0.5) == ("spec",)
    assert seen == [("adapter.safetensors", 0.5)]


def test_lora_loader_admits_a_plain_fp8_checkpoint(tmp_path, monkeypatch):
    """Qwen-Image's literal e4m3fn weights: the property check admits them, and
    the lever check after it in the LoRA loader decides."""
    path = tmp_path / "qwen_fp8.safetensors"
    _write_fsdp_dtype_checkpoint(path, "F8_E4M3")

    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: str(path)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    seen = []
    model = types.SimpleNamespace(
        unet_name="qwen_fp8.safetensors",
        mesh=types.SimpleNamespace(
            topology_preset="uly2+fsdp", world=2,
            worker_args={"lora_low_rss": True}),
        with_lora=lambda name, strength: seen.append((name, strength)) or "spec",
    )
    assert loaders.DGXMonarchLoraLoader().load_lora(
        model, "adapter.safetensors", 0.5) == ("spec",)
    assert seen == [("adapter.safetensors", 0.5)]


def _explicit_fsdp_mesh(side_effects: list[str]):
    class Handle:
        world = 2

        def call_all(self, *_args, **_kwargs):
            side_effects.append("rpc")
            pytest.fail("checkpoint drift reached worker RPC")

    return types.SimpleNamespace(
        handle=Handle(),
        world=2,
        topology_preset="uly2+fsdp",
        attention="TORCH_FLASH",
        sync_ulysses=False,
        worker_args={},
    )


def _forbid_loader_side_effects(monkeypatch, side_effects: list[str]) -> None:
    from dgx_monarch import mesh_setup
    from dgx_monarch.nodes import render_session

    def forbid(name):
        def call(*_args, **_kwargs):
            side_effects.append(name)
            pytest.fail(f"checkpoint drift reached {name}")

        return call

    def ensure_without_side_effect(_handle, *, mesh_preflight):
        assert mesh_preflight is not None
        mesh_preflight(types.SimpleNamespace(), 2)
        return forbid("mesh healing")()

    monkeypatch.setattr(loaders, "ensure_live", ensure_without_side_effect)
    monkeypatch.setattr(
        mesh_setup,
        "ensure_request_setup",
        forbid("mesh setup"),
    )
    monkeypatch.setattr(
        render_session,
        "mutation_render_session",
        forbid("render session"),
    )


def _assert_post_sniff_drift_refused(
    monkeypatch,
    resolved_path,
    mutate,
) -> None:
    side_effects: list[str] = []
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: str(resolved_path)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    _forbid_loader_side_effects(monkeypatch, side_effects)

    assert_binding = loaders._assert_explicit_fsdp_checkpoint_binding

    def mutate_then_assert(binding):
        mutate()
        assert_binding(binding)

    monkeypatch.setattr(
        loaders,
        "_assert_explicit_fsdp_checkpoint_binding",
        mutate_then_assert,
    )

    with pytest.raises(
        ArtifactBindingError,
        match="changed after the driver FSDP checkpoint header preflight",
    ):
        loaders.DGXMonarchUNETLoader().load(
            _explicit_fsdp_mesh(side_effects),
            "model.safetensors",
            "default",
        )
    assert side_effects == []


def test_explicit_fsdp_atomic_replacement_after_sniff_is_driver_local(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "model.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    _write_fsdp_dtype_checkpoint(path, "BF16")
    _write_fsdp_dtype_checkpoint(replacement, "F8_E4M3")

    _assert_post_sniff_drift_refused(
        monkeypatch,
        path,
        lambda: replacement.replace(path),
    )
    assert sniff_fsdp_launch_quant(path) == "fp8"


def test_explicit_fsdp_resolver_remap_is_driver_local(tmp_path, monkeypatch):
    initial = tmp_path / "initial.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    _write_fsdp_dtype_checkpoint(initial, "BF16")
    _write_fsdp_dtype_checkpoint(replacement, "F8_E4M3")
    # Five resolutions, three above ensure_live: the upstream gate
    # (loader_preflight.preflight_upstream_gated_artifact), the loader-site
    # footprint preflight (loader_preflight.preflight_loader_footprint) and the
    # unconditional consent projection after it
    # (consent_projection.project_for_loader). Then the FSDP header preflight
    # binds `initial`, and the binding assertion sees the swap.
    resolved = iter((str(initial), str(initial), str(initial), str(initial),
                     str(replacement)))
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: next(resolved)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    side_effects: list[str] = []
    _forbid_loader_side_effects(monkeypatch, side_effects)

    with pytest.raises(
        ArtifactBindingError,
        match="changed after the driver FSDP checkpoint header preflight",
    ):
        loaders.DGXMonarchUNETLoader().load(
            _explicit_fsdp_mesh(side_effects),
            "model.safetensors",
            "default",
        )
    assert side_effects == []
    assert sniff_fsdp_launch_quant(initial) == "bf16"
    assert sniff_fsdp_launch_quant(replacement) == "fp8"


def test_explicit_fsdp_symlink_retarget_after_sniff_is_driver_local(
    tmp_path,
    monkeypatch,
):
    initial = tmp_path / "initial.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    path = tmp_path / "model.safetensors"
    next_link = tmp_path / "next.safetensors"
    _write_fsdp_dtype_checkpoint(initial, "BF16")
    _write_fsdp_dtype_checkpoint(replacement, "F8_E4M3")
    path.symlink_to(initial)
    next_link.symlink_to(replacement)

    _assert_post_sniff_drift_refused(
        monkeypatch,
        path,
        lambda: next_link.replace(path),
    )
    assert sniff_fsdp_launch_quant(path) == "fp8"


def test_explicit_fsdp_disappearance_after_sniff_is_driver_local(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "model.safetensors"
    _write_fsdp_dtype_checkpoint(path, "BF16")

    _assert_post_sniff_drift_refused(monkeypatch, path, path.unlink)
    assert not path.exists()


def test_explicit_fsdp_in_place_metadata_drift_is_driver_local(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "model.safetensors"
    _write_fsdp_dtype_checkpoint(path, "BF16")

    def change_metadata():
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    _assert_post_sniff_drift_refused(monkeypatch, path, change_metadata)
    assert sniff_fsdp_launch_quant(path) == "bf16"


@pytest.mark.parametrize(
    ("replacement_kind", "expected_kind"),
    [("fp8", "fp8"), ("malformed", "unknown")],
)
def test_explicit_fsdp_atomic_replacement_during_sniff_is_driver_local(
    tmp_path,
    monkeypatch,
    replacement_kind,
    expected_kind,
):
    from dgx_monarch import safetensors_header as header_module

    path = tmp_path / "model.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    _write_fsdp_dtype_checkpoint(path, "BF16")
    if replacement_kind == "fp8":
        _write_fsdp_dtype_checkpoint(replacement, "F8_E4M3")
    else:
        replacement.write_bytes(b"not a safetensors checkpoint")
    side_effects: list[str] = []
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: str(path)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    _forbid_loader_side_effects(monkeypatch, side_effects)

    read_exact = header_module._read_exact
    replaced = False
    header_reads = 0

    def read_then_replace(fd, offset, length):
        # The FSDP launch proof is the third header read of this load: the
        # upstream gate reads the header first and the loader-site footprint
        # preflight sniffs the family second, both above ensure_live
        # (nodes/loader_preflight.py). Replacing on an earlier read would swap
        # the file before the sniff this test is about starts.
        nonlocal replaced, header_reads
        data = read_exact(fd, offset, length)
        if offset == 8:
            header_reads += 1
            if header_reads >= 3 and not replaced:
                replaced = True
                replacement.replace(path)
        return data

    monkeypatch.setattr(header_module, "_read_exact", read_then_replace)
    with pytest.raises(
        UnsupportedModelError,
        match=r"supports bf16, fp16, fp8 and int8 checkpoints at launch \(got quant=unknown\)",
    ):
        loaders.DGXMonarchUNETLoader().load(
            _explicit_fsdp_mesh(side_effects),
            "model.safetensors",
            "default",
        )
    assert replaced is True
    assert side_effects == []
    assert sniff_fsdp_launch_quant(path) == expected_kind


@pytest.mark.parametrize("quant_kind", ["fp32", "nvfp4", "unknown"])
@pytest.mark.parametrize("weight_dtype", ["default", "bf16"])
def test_explicit_fsdp_checkpoint_preflight_is_typed_and_driver_local(
    monkeypatch,
    quant_kind,
    weight_dtype,
):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda kind, name: f"/models/{kind}/{name}"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(
        loaders,
        "sniff_fsdp_launch_quant_proof",
        lambda _path: _fsdp_quant_proof(quant_kind),
    )

    with pytest.raises(
        UnsupportedModelError,
        match=rf"supports bf16, fp16, fp8 and int8 checkpoints at launch \(got quant={quant_kind}\)",
    ):
        loaders._preflight_explicit_fsdp_checkpoint(
            Topology(ulysses=2, fsdp=True, world=2),
            "ltx-fp8.safetensors",
            weight_dtype,
        )


@pytest.mark.parametrize("proof_kind", ["bf16", "fp16"])
@pytest.mark.parametrize("weight_dtype", ["default", "bf16"])
def test_explicit_fsdp_checkpoint_preflight_accepts_bf16(
    monkeypatch,
    weight_dtype,
    proof_kind,
):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda kind, name: f"/models/{kind}/{name}"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(
        loaders,
        "sniff_fsdp_launch_quant_proof",
        lambda _path: _fsdp_quant_proof(proof_kind),
    )

    binding = loaders._preflight_explicit_fsdp_checkpoint(
        Topology(ulysses=2, fsdp=True, world=2),
        "ltx-bf16.safetensors",
        weight_dtype,
    )
    assert binding is not None
    assert binding.file_identity == _FAKE_FSDP_FILE_IDENTITY


@pytest.mark.parametrize(("weight_dtype", "expected_options"), [
    ("default", {}),
    ("bf16", {"weight_dtype": "bf16"}),
])
def test_explicit_fsdp_stable_bf16_progresses_after_bound_preflight(
    tmp_path,
    monkeypatch,
    weight_dtype,
    expected_options,
):
    from dgx_monarch import mesh_setup
    from dgx_monarch.nodes import render_session

    path = tmp_path / "model.safetensors"
    _write_fsdp_dtype_checkpoint(path, "BF16")
    events: list[str] = []

    folder_paths = types.ModuleType("folder_paths")

    def resolve(_kind, _name):
        events.append("resolve")
        return str(path)

    folder_paths.get_full_path = resolve
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    class Handle:
        world = 2

        def call_all(self, endpoint, unet_name, options, loras, **kwargs):
            events.append("rpc")
            assert (endpoint, unet_name, options, loras) == (
                "load_model",
                "model.safetensors",
                expected_options,
                [],
            )
            assert kwargs["setup_token"] is setup_token
            return [{"host": "driver", "transition": "load"}]

    handle = Handle()
    mesh = types.SimpleNamespace(
        handle=handle,
        world=2,
        topology_preset="uly2+fsdp",
        attention="TORCH_FLASH",
        sync_ulysses=False,
        worker_args={},
    )

    def ensure(candidate, *, mesh_preflight):
        assert candidate is handle
        assert mesh_preflight is not None
        events.append("ensure:start")
        mesh_preflight(types.SimpleNamespace(), 2)
        events.append("ensure:complete")
        return handle

    setup_token = object()

    def ensure_setup(candidate, *_args, **_kwargs):
        assert candidate is handle
        events.append("setup")
        return setup_token

    def mutation_session(candidate):
        assert candidate is handle
        events.append("session")
        return nullcontext()

    monkeypatch.setattr(loaders, "ensure_live", ensure)
    monkeypatch.setattr(mesh_setup, "ensure_request_setup", ensure_setup)
    monkeypatch.setattr(render_session, "mutation_render_session", mutation_session)

    (model,) = loaders.DGXMonarchUNETLoader().load(
        mesh,
        "model.safetensors",
        weight_dtype,
    )

    assert model.unet_name == "model.safetensors"
    assert model.mesh is mesh
    assert model.options == expected_options
    assert events == [
        # Three driver-side resolutions above ensure_live, in this order: the
        # upstream gate, which refuses an artifact no world can run before
        # anything else reads a byte; the loader-site footprint preflight
        # (nodes/loader_preflight.py), which puts a capacity refusal before the
        # fleet is healed; and the unconditional consent projection after it
        # (nodes/consent_projection), which lets a granted card reach an eager
        # load the estimator stood down for. The next two are the FSDP header
        # preflight and its binding assertion, inside the mesh preflight
        # callback.
        "resolve",
        "resolve",
        "resolve",
        "ensure:start",
        "resolve",
        "resolve",
        "ensure:complete",
        "session",
        "setup",
        "rpc",
    ]


@pytest.mark.parametrize("header_kind", ["bf16", "fp16"])
@pytest.mark.parametrize(
    "weight_dtype",
    ["fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
)
def test_explicit_fsdp_weight_dtype_override_is_treated_as_fp8(
    monkeypatch,
    header_kind,
    weight_dtype,
):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda kind, name: f"/models/{kind}/{name}"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(
        loaders,
        "sniff_fsdp_launch_quant_proof",
        lambda _path: pytest.fail("FP8 override read the checkpoint header"),
    )

    with pytest.raises(
        UnsupportedModelError,
        match=r"no file-backed bytes",
    ):
        loaders._preflight_explicit_fsdp_checkpoint(
            Topology(ulysses=2, fsdp=True, world=2),
            f"ltx-{header_kind}.safetensors",
            weight_dtype,
        )


def test_non_fsdp_checkpoint_preflight_does_not_read_the_header(monkeypatch):
    monkeypatch.setattr(
        loaders,
        "sniff_fsdp_launch_quant_proof",
        lambda _path: pytest.fail("non-FSDP preflight read the checkpoint"),
    )

    binding = loaders._preflight_explicit_fsdp_checkpoint(
        Topology(ulysses=2, world=2),
        "unused.safetensors",
        "default",
    )
    assert binding is None


@pytest.mark.parametrize(("weight_dtype", "error"), [
    (None, TypeError),
    (1, TypeError),
    (True, TypeError),
    ("fp16", ValueError),
    ("BF16", ValueError),
])
def test_loader_rejects_malformed_weight_dtype_before_mesh_effects(
    monkeypatch,
    weight_dtype,
    error,
):
    monkeypatch.setattr(
        loaders,
        "ensure_live",
        lambda *_args, **_kwargs: pytest.fail("invalid dtype reached mesh acquisition"),
    )
    mesh = types.SimpleNamespace(topology_preset="auto", handle=object())

    with pytest.raises(error, match="weight_dtype"):
        loaders.DGXMonarchUNETLoader().load(
            mesh,
            "model.safetensors",
            weight_dtype,
        )


def test_loader_dtype_choices_append_explicit_bf16_without_reordering(monkeypatch):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_filename_list = lambda _kind: ["model.safetensors"]
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    choices = loaders.DGXMonarchUNETLoader.INPUT_TYPES()["required"][
        "weight_dtype"
    ][0]
    assert choices == [
        "default",
        "fp8_e4m3fn",
        "fp8_e4m3fn_fast",
        "fp8_e5m2",
        "bf16",
    ]


@pytest.mark.parametrize("suffix", [".ckpt", ".bin"])
def test_explicit_fsdp_legacy_checkpoint_refuses_without_source_proof(
    monkeypatch,
    suffix,
):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, name: f"/models/{name}"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(
        loaders,
        "sniff_fsdp_launch_quant_proof",
        lambda path: (
            _fsdp_quant_proof(None)
            if path.endswith(suffix)
            else pytest.fail("wrong path")
        ),
    )
    with pytest.raises(
        UnsupportedModelError,
        match=r"supports bf16, fp16, fp8 and int8 checkpoints at launch \(got quant=unknown\)",
    ):
        loaders._preflight_explicit_fsdp_checkpoint(
            Topology(ulysses=2, fsdp=True, world=2),
            f"legacy{suffix}",
            "default",
        )


@pytest.mark.parametrize("suffix", [".ckpt", ".bin"])
def test_explicit_bf16_fsdp_legacy_checkpoint_fails_closed(
    monkeypatch,
    suffix,
):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, name: f"/models/{name}"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(
        loaders,
        "sniff_fsdp_launch_quant_proof",
        lambda _path: pytest.fail("unprovable explicit BF16 read a legacy header"),
    )

    with pytest.raises(
        UnsupportedModelError,
        match=r"supports bf16, fp16, fp8 and int8 checkpoints at launch \(got quant=unknown\)",
    ):
        loaders._preflight_explicit_fsdp_checkpoint(
            Topology(ulysses=2, fsdp=True, world=2),
            f"legacy{suffix}",
            "bf16",
        )


def test_auto_loader_keeps_checkpoint_header_deferred(monkeypatch):
    handle = types.SimpleNamespace(world=2)
    mesh = types.SimpleNamespace(
        handle=handle,
        world=2,
        topology_preset="auto",
        attention="TORCH_FLASH",
        sync_ulysses=False,
        worker_args={},
    )
    healed = []
    monkeypatch.setattr(
        loaders,
        "sniff_fsdp_launch_quant_proof",
        lambda _path: pytest.fail("auto loader read the checkpoint header"),
    )
    monkeypatch.setattr(
        loaders,
        "ensure_live",
        lambda candidate, *, mesh_preflight: (
            pytest.fail("auto loader supplied a mesh preflight")
            if mesh_preflight is not None
            else healed.append(candidate) or candidate
        ),
    )

    (model,) = loaders.DGXMonarchUNETLoader().load(
        mesh,
        "model.safetensors",
        "default",
    )
    assert healed == [handle]
    assert model.mesh is mesh


def test_explicit_fsdp_fp8_override_rejects_legacy_before_header_read(monkeypatch):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, name: f"/models/{name}"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(
        loaders,
        "sniff_fsdp_launch_quant_proof",
        lambda _path: pytest.fail("FP8 legacy refusal read a checkpoint header"),
    )

    with pytest.raises(
        UnsupportedModelError,
        match=r"no file-backed bytes",
    ):
        loaders._preflight_explicit_fsdp_checkpoint(
            Topology(ulysses=2, fsdp=True, world=2),
            "legacy.ckpt",
            "fp8_e4m3fn",
        )


def test_unet_loader_refuses_before_setup_or_rpc(monkeypatch):
    from dgx_monarch import mesh_setup

    class Handle:
        world = 2

        def call_all(self, *_args, **_kwargs):
            pytest.fail("FSDP preflight reached worker RPC")

    handle = Handle()
    mesh = types.SimpleNamespace(
        handle=handle,
        world=2,
        topology_preset="uly2+fsdp",
        attention="TORCH_FLASH",
        sync_ulysses=False,
        worker_args={},
    )
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, name: f"/models/{name}"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    def ensure_without_healing(_handle, *, mesh_preflight):
        assert mesh_preflight is not None
        mesh_preflight(types.SimpleNamespace(), 2)
        pytest.fail("FSDP preflight healed the mesh")

    monkeypatch.setattr(loaders, "ensure_live", ensure_without_healing)
    monkeypatch.setattr(
        loaders,
        "sniff_fsdp_launch_quant_proof",
        lambda _path: _fsdp_quant_proof("fp32"),
    )
    monkeypatch.setattr(
        mesh_setup,
        "ensure_request_setup",
        lambda *_args, **_kwargs: pytest.fail("FSDP preflight reached mesh setup"),
    )

    with pytest.raises(
        UnsupportedModelError,
        match=r"supports bf16, fp16, fp8 and int8 checkpoints at launch \(got quant=fp32\)",
    ):
        loaders.DGXMonarchUNETLoader().load(
            mesh,
            "model.safetensors",
            "default",
        )


def test_one_fsdp_multiplication_across_every_caller():
    """The price is one function object, not a copy per caller.

    A value-equality assertion inside one module cannot catch a re-export
    replaced by a second definition, and a second definition is how the worker
    guard and the ceremony reload price start charging two prices for one
    placement. Identity catches it.
    """
    from dgx_monarch import capacity_fit, fsdp_reload_price

    assert fsdp_module.fsdp_required_bytes is capacity_fit.fsdp_required_bytes
    assert fsdp_reload_price.fsdp_required_bytes is capacity_fit.fsdp_required_bytes
    assert fsdp_module.fsdp_materialize_factor is capacity_fit.fsdp_materialize_factor
    assert fsdp_module.FSDP_MATERIALIZE_FACTOR == capacity_fit.FSDP_MATERIALIZE_FACTOR
    assert (fsdp_module.FSDP_STREAM_TRANSIENT_FRACTION
            == capacity_fit.FSDP_STREAM_TRANSIENT_FRACTION)


def _class_p(exc_info) -> str:
    text = str(exc_info.value)
    tag = parse_refusal_tag(text)
    assert tag is not None, text
    assert tag.refusal_class is RefusalClass.PHYSICS and tag.guard is None
    return text


def _rescan_answers(monkeypatch, evidence, profile=None):
    """The detector's live rescan reads the profile's own core kind."""
    from dgx_monarch.actor import store_detect

    live_profile = profile or evidence.live_dtype_profile
    live = LivePrecisionEvidence(
        quant_kind=fsdp_module._PROFILE_LIVE_KIND.get(live_profile, evidence.quant_kind),
        live_dtype_profile=live_profile,
        auxiliary_parameter_count=evidence.auxiliary_parameter_count,
        auxiliary_parameter_bytes=evidence.auxiliary_parameter_bytes,
    )
    monkeypatch.setattr(store_detect, "_detect_live_precision", lambda _p, _k: live)


@pytest.mark.parametrize("quant_kind", ["nvfp4", "mxfp8", "fp32", "unknown"])
def test_every_launch_quant_refusal_is_class_p(quant_kind):
    with pytest.raises(UnsupportedModelError) as exc_info:
        validate_fsdp_launch_quant(quant_kind)
    _class_p(exc_info)


def test_the_wan_fp16_pair_under_fsdp_refuses_class_p_and_names_the_kind(monkeypatch):
    """The fp16 Wan pair's fp32 ingress layers read the Wan profile, which is
    audited for a bf16 core only; the 2026-09 sweep saw this refusal untagged."""
    evidence = LivePrecisionEvidence(
        quant_kind="fp16", live_dtype_profile=fsdp_module.WAN_FP32_INGRESS_PROFILE,
        auxiliary_parameter_count=2, auxiliary_parameter_bytes=8, checkpoint_kind="fp16")
    _rescan_answers(monkeypatch, evidence)

    with pytest.raises(UnsupportedModelError) as exc_info:
        fsdp_module.validate_fsdp_live_precision(object(), "fp16", evidence)

    text = _class_p(exc_info)
    assert "got a fp16 checkpoint with 2 ingress parameters, 8 bytes" in text
    assert "resident topology" in text and "bf16 checkpoint" in text


@pytest.mark.parametrize(
    ("quant_kind", "evidence", "rescan_profile"),
    [
        pytest.param("bf16", None, None, id="malformed-evidence"),
        pytest.param(
            "fp16",
            LivePrecisionEvidence("bf16", fsdp_module.ALL_BF16_PROFILE, checkpoint_kind="bf16"),
            None, id="kind-mismatch"),
        pytest.param(
            "bf16",
            LivePrecisionEvidence("bf16", fsdp_module.ALL_BF16_PROFILE, checkpoint_kind=None),
            None, id="unbound-checkpoint"),
        pytest.param(
            "bf16",
            LivePrecisionEvidence("bf16", fsdp_module.ALL_BF16_PROFILE, checkpoint_kind="bf16"),
            fsdp_module.ALL_FP16_PROFILE, id="rescan-drift"),
        pytest.param(
            "bf16",
            LivePrecisionEvidence("bf16", fsdp_module.ALL_BF16_PROFILE, 1, 8, "bf16"),
            None, id="uniform-with-auxiliary"),
        pytest.param(
            "fp16",
            LivePrecisionEvidence("fp16", fsdp_module.FP32_ISLANDS_PROFILE, 1, 8, "fp16"),
            None, id="islands-on-fp16"),
    ],
)
def test_every_live_precision_refusal_is_class_p(monkeypatch, quant_kind, evidence, rescan_profile):
    if evidence is not None:
        _rescan_answers(monkeypatch, evidence, rescan_profile)
    with pytest.raises(UnsupportedModelError) as exc_info:
        fsdp_module.validate_fsdp_live_precision(object(), quant_kind, evidence)
    _class_p(exc_info)
