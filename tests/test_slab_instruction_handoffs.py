"""Slab construction and hook install and restore keep an owner at each tested instruction boundary."""
from __future__ import annotations

import dis
import json
import os
import struct
import sys
import types
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import pytest
import torch

from dgx_monarch.actor import comfy_bridge, slab_arena, slab_lifetime
from dgx_monarch.actor.slab import WeightSlab
from dgx_monarch.actor.slab_arena import (
    _Arena,
    _finish_prepublished,
    _OwnedDescriptor,
    _SlabHookRestore,
)
from slab_lifetime_helpers import reset_slab_lifetime


class _InstructionAbort(BaseException):
    pass


@pytest.fixture(autouse=True)
def _isolated_lifetime(monkeypatch):
    reset_slab_lifetime(monkeypatch)


@pytest.fixture
def checkpoint(tmp_path) -> str:
    tensor = torch.arange(16, dtype=torch.float32).to(torch.bfloat16)
    raw = tensor.view(torch.uint8).numpy().tobytes()
    header = json.dumps({
        "weight": {
            "dtype": "BF16",
            "shape": list(tensor.shape),
            "data_offsets": [0, len(raw)],
        },
    }).encode()
    path = tmp_path / "handoff.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + raw)
    return str(path)


def _after_call(code, needle: str) -> int:
    instructions = list(dis.get_instructions(code))
    load_index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.argval == needle
        and instruction.opname in {
            "LOAD_ATTR", "LOAD_FAST", "LOAD_GLOBAL", "LOAD_METHOD",
        }
    )
    call_index = next(
        index for index in range(load_index + 1, len(instructions))
        if instructions[index].opname == "CALL"
    )
    return instructions[call_index + 1].offset


def _after_instruction(code, opname: str, argval: str) -> int:
    instructions = list(dis.get_instructions(code))
    index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname == opname and instruction.argval == argval
    )
    return instructions[index + 1].offset


def _interrupt_once(
    code,
    target: int,
    invoke: Callable[[], Any],
    *,
    propagates: bool = True,
) -> _InstructionAbort:
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6) if monitoring.get_tool(index) is None)
    primary = _InstructionAbort(f"instruction {target} interrupted")

    def interrupt(_code, instruction_offset):
        if instruction_offset == target:
            monitoring.set_local_events(tool_id, code, 0)
            raise primary

    monitoring.use_tool_id(tool_id, "dgxm-slab-handoff-test")
    monitoring.register_callback(tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(tool_id, code, monitoring.events.INSTRUCTION)
    caught = None
    try:
        invoke()
    except _InstructionAbort as error:
        caught = error
    finally:
        monitoring.set_local_events(tool_id, code, 0)
        monitoring.register_callback(tool_id, monitoring.events.INSTRUCTION, None)
        monitoring.free_tool_id(tool_id)
    if propagates:
        assert caught is primary
    else:
        assert caught is None
    return primary


def _interrupt_after_call(
    code,
    needle: str,
    invoke: Callable[[], Any],
) -> _InstructionAbort:
    return _interrupt_once(code, _after_call(code, needle), invoke)


def _record_memfds(monkeypatch) -> tuple[list[int], Callable[..., int]]:
    created: list[int] = []
    healthy = slab_arena.os.memfd_create

    def create(name, flags):
        fd = healthy(name, flags)
        created.append(fd)
        return fd

    monkeypatch.setattr(slab_arena.os, "memfd_create", create)
    return created, healthy


def _assert_fd_closed(fd: int) -> None:
    with pytest.raises(OSError):
        os.fstat(fd)


def _stub_comfy_hooks(monkeypatch):
    utils = types.ModuleType("comfy.utils")

    def original_ltf(*_args, **_kwargs):
        return None

    utils.load_torch_file = original_ltf
    model_base = types.ModuleType("comfy.model_base")

    def original_lmw(*_args, **_kwargs):
        return None

    model_base.BaseModel = type(
        "BaseModel", (), {"load_model_weights": original_lmw})
    comfy = types.ModuleType("comfy")
    comfy.utils, comfy.model_base = utils, model_base
    for name, module in (
        ("comfy", comfy),
        ("comfy.utils", utils),
        ("comfy.model_base", model_base),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return comfy, utils, model_base, original_ltf, original_lmw


def test_slab_constructor_failure_never_closes_a_preexisting_handoff(
    monkeypatch,
):
    from dgx_monarch.actor import slab as slab_module

    _stub_comfy_hooks(monkeypatch)
    existing = types.SimpleNamespace(closed=0)
    existing.close = lambda: setattr(existing, "closed", existing.closed + 1)
    handoff = [existing]
    primary = RuntimeError("new slab construction failed")
    monkeypatch.setattr(
        slab_module,
        "WeightSlab",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )

    with pytest.raises(RuntimeError) as caught:
        with comfy_bridge.slab_load("/new.safetensors", handoff=handoff):
            pytest.fail("failed constructor entered slab context")

    assert caught.value is primary
    assert handoff == [existing]
    assert existing.closed == 0


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_slab_load_constructor_return_has_durable_exact_owner(
    checkpoint,
    monkeypatch,
):
    utils = types.ModuleType("comfy.utils")
    utils.load_torch_file = lambda *_args, **_kwargs: None
    model_base = types.ModuleType("comfy.model_base")
    model_base.BaseModel = type("BaseModel", (), {"load_model_weights": lambda *a, **k: None})
    comfy = types.ModuleType("comfy")
    comfy.utils, comfy.model_base = utils, model_base
    for name, module in (("comfy", comfy), ("comfy.utils", utils),
                         ("comfy.model_base", model_base)):
        monkeypatch.setitem(sys.modules, name, module)
    created, _healthy = _record_memfds(monkeypatch)
    handoff: list[WeightSlab] = []

    _interrupt_after_call(
        comfy_bridge.slab_load.__wrapped__.__code__,
        "WeightSlab",
        lambda: comfy_bridge.slab_load(checkpoint, handoff=handoff).__enter__(),
    )

    assert len(created) == 1
    _assert_fd_closed(created[0])
    assert handoff == []
    assert not slab_lifetime.cleanup_pending()


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_primary_arena_return_is_owned_before_weight_slab_assignment(
    checkpoint,
    monkeypatch,
):
    created, _healthy = _record_memfds(monkeypatch)
    handoff: list[WeightSlab] = []

    _interrupt_after_call(
        WeightSlab.__init__.__code__,
        "_Arena",
        lambda: WeightSlab(checkpoint, handoff=handoff),
    )

    assert len(created) == 1
    _assert_fd_closed(created[0])
    assert handoff == []
    assert not slab_lifetime.cleanup_pending()


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_annex_arena_return_is_owned_before_weight_slab_assignment(
    checkpoint,
):
    slab = WeightSlab(checkpoint)
    slab.confirm_handoff()
    module = torch.nn.Module()
    module.register_parameter(
        "outside",
        torch.nn.Parameter(torch.ones(32, dtype=torch.float32), requires_grad=False),
    )

    _interrupt_after_call(
        WeightSlab.reabsorb.__code__,
        "_Arena",
        lambda: slab.reabsorb(module),
    )

    assert slab.annex is None
    assert len(slab._annex_handoff) == 1
    annex = slab._annex_handoff[0]
    owned_fd = annex.fd
    os.fstat(owned_fd)
    assert any(
        owned is annex
        for owned, _error in slab_lifetime._RETAINED_FAILED_LOAD_RESOURCES
    )
    slab.close()
    _assert_fd_closed(owned_fd)
    assert slab._annex_handoff == []
    assert not slab_lifetime.cleanup_pending()


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_unknown_memfd_return_poison_is_durable_and_never_retried(monkeypatch):
    created, _healthy = _record_memfds(monkeypatch)
    handoff: list[_Arena] = []

    _interrupt_after_call(
        _Arena.__init__.__code__,
        "memfd_create",
        lambda: _Arena("dgxm-interrupted-memfd", 4096, handoff=handoff),
    )

    assert len(created) == 1
    os.fstat(created[0])
    assert len(handoff) == 1
    arena = handoff[0]
    assert arena.fd == -1 and arena.acquisition_uncertain
    assert slab_lifetime.cleanup_poisoned()
    assert slab_lifetime._RETAINED_FAILED_LOAD_RESOURCES[0][0] is arena
    with pytest.raises(RuntimeError, match="reset the Attached mesh"):
        arena.close()

    # Production cannot know this numeric FD, so it cannot retry it. The test
    # recorded it only so the descriptor does not leak into later tests.
    os.close(created[0])
    slab_lifetime._RETAINED_FAILED_LOAD_RESOURCES.clear()
    slab_lifetime._FAILED_LOAD_CLEANUP_POISONED = False


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("boundary", ["file owner", "descriptor owner"])
def test_checkpoint_source_return_is_finalizable_at_every_store_boundary(
    checkpoint,
    monkeypatch,
    boundary,
):
    opened: list[int] = []
    healthy_file_io = slab_arena.io.FileIO

    def tracked_file_io(*args, **kwargs):
        file = healthy_file_io(*args, **kwargs)
        opened.append(file.fileno())
        return file

    monkeypatch.setattr(slab_arena.io, "FileIO", tracked_file_io)
    slab: WeightSlab | None = None
    if boundary == "file owner":
        code = _OwnedDescriptor.open.__func__.__code__
        needle = "FileIO"

        def invoke():
            return _OwnedDescriptor.open(checkpoint, "test source")
    else:
        slab = WeightSlab(checkpoint)
        slab.confirm_handoff()
        code = WeightSlab._read_all.__code__
        needle = "open"
        invoke = slab._read_all

    try:
        _interrupt_after_call(code, needle, invoke)
        assert opened
        _assert_fd_closed(opened[-1])
        assert not slab_lifetime.cleanup_pending()
    finally:
        if slab is not None:
            slab.close()


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_loader_return_keeps_exact_slab_in_external_handoff(monkeypatch):
    class Slab:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    class Base:
        offload_device = "cpu"

    created: list[Slab] = []

    @contextmanager
    def fake_slab_load(_path, *, handoff=None):
        slab = Slab()
        created.append(slab)
        if handoff is not None:
            handoff.append(slab)
        yield slab

    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: "cuda:0"
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setattr(comfy_bridge, "slab_load", fake_slab_load)
    handoff: list[Any] = []

    def caller():
        result = comfy_bridge.load_diffusion_model_slab(
            "/fake/model.safetensors", {}, lambda *_args, **_kwargs: Base(),
            handoff=handoff,
        )
        return result

    _interrupt_after_call(
        caller.__code__, "load_diffusion_model_slab", caller)

    assert len(created) == 1
    assert handoff == [created[0]]
    assert created[0].closed == 0
    created[0].close()
    handoff.clear()


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("attribute", ["load_torch_file", "load_model_weights"])
def test_hook_install_store_interruption_restores_both_hooks(
    checkpoint,
    monkeypatch,
    attribute,
):
    _comfy, utils, model_base, original_ltf, original_lmw = _stub_comfy_hooks(
        monkeypatch)
    handoff: list[WeightSlab] = []

    _interrupt_once(
        comfy_bridge.slab_load.__wrapped__.__code__,
        _after_instruction(
            comfy_bridge.slab_load.__wrapped__.__code__, "STORE_ATTR", attribute),
        lambda: comfy_bridge.slab_load(checkpoint, handoff=handoff).__enter__(),
    )

    assert utils.load_torch_file is original_ltf
    assert model_base.BaseModel.load_model_weights is original_lmw
    assert handoff == []
    assert not slab_lifetime.cleanup_pending()


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize(
    ("method_name", "attribute"),
    [
        ("_restore_ltf", "load_torch_file"),
        ("_restore_lmw", "load_model_weights"),
    ],
)
@pytest.mark.parametrize("opname", ["STORE_ATTR", "LOAD_ATTR"])
def test_hook_restore_retries_store_and_identity_boundaries(
    monkeypatch,
    method_name,
    attribute,
    opname,
):
    _comfy, utils, model_base, original_ltf, original_lmw = _stub_comfy_hooks(
        monkeypatch)
    slab = types.SimpleNamespace(_close_guard=None)
    restore = _SlabHookRestore(
        utils, model_base.BaseModel, original_ltf, original_lmw, slab)
    restore.bind()
    utils.load_torch_file = lambda *_args, **_kwargs: "slab"
    model_base.BaseModel.load_model_weights = lambda *_args, **_kwargs: "slab"
    method = getattr(_SlabHookRestore, method_name)

    _interrupt_once(
        method.__code__,
        _after_instruction(method.__code__, opname, attribute),
        restore.close,
    )

    assert utils.load_torch_file is original_ltf
    assert model_base.BaseModel.load_model_weights is original_lmw
    assert slab._close_guard is restore
    restore.close()
    assert slab._close_guard is None


def test_hook_restore_retry_promotes_later_cancellation():
    ordinary = RuntimeError("first restore failed")
    cancellation = KeyboardInterrupt("retry cancelled")

    class Utils:
        def __init__(self):
            object.__setattr__(self, "errors", iter((ordinary, cancellation)))
            object.__setattr__(self, "load_torch_file", None)

        def __setattr__(self, name, value):
            if name == "load_torch_file" and value == "original":
                raise next(self.errors)
            object.__setattr__(self, name, value)

    utils = Utils()
    restore = _SlabHookRestore(
        utils,
        types.SimpleNamespace(load_model_weights="original-lmw"),
        "original",
        "original-lmw",
        types.SimpleNamespace(_close_guard=None),
    )

    error = restore._restore_ltf()

    assert error is cancellation
    assert any("first restore failed" in note for note in cancellation.__notes__)


@pytest.mark.parametrize("same_object", [False, True])
def test_hook_restore_close_is_cancellation_first_without_self_cause(
    monkeypatch,
    same_object,
):
    ordinary = RuntimeError("load_torch_file restore failed")
    cancellation = KeyboardInterrupt("load_model_weights restore cancelled")
    restore = _SlabHookRestore(
        types.SimpleNamespace(load_torch_file=None),
        types.SimpleNamespace(load_model_weights=None),
        object(),
        object(),
        types.SimpleNamespace(_close_guard=None),
    )
    if same_object:
        ordinary = cancellation
    monkeypatch.setattr(restore, "_restore_ltf", lambda: ordinary)
    monkeypatch.setattr(restore, "_restore_lmw", lambda: cancellation)

    with pytest.raises(KeyboardInterrupt) as caught:
        restore.close()

    assert caught.value is cancellation
    assert caught.value.__cause__ is not cancellation


def test_finish_prepublished_promotes_restore_cancellation(monkeypatch):
    primary = RuntimeError("slab load failed")
    cancellation = KeyboardInterrupt("hook restore cancelled")
    retained = []
    resource = types.SimpleNamespace(
        close=lambda: (_ for _ in ()).throw(cancellation))
    monkeypatch.setattr(
        slab_lifetime,
        "retain_failed_load_resource",
        lambda owned, error=None: retained.append((owned, error)),
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        _finish_prepublished(resource, primary)

    assert caught.value is cancellation
    assert caught.value.__cause__ is primary
    assert retained == [(resource, cancellation)]


def test_failed_hook_restore_guards_slab_until_later_confirmed_restore(
    checkpoint,
    monkeypatch,
):
    class PrimaryAbort(BaseException):
        pass

    class RestoreAbort(BaseException):
        pass

    events: list[str] = []
    utils_policy: dict[str, Any] = {"block": False, "original": None}

    class GuardedUtils(types.ModuleType):
        def __setattr__(self, name, value):
            if (
                name == "load_torch_file"
                and utils_policy["block"]
                and value is utils_policy["original"]
            ):
                events.append("ltf blocked")
                raise RestoreAbort("load_torch_file restore blocked")
            if name == "load_torch_file" and value is utils_policy["original"]:
                events.append("ltf restored")
            super().__setattr__(name, value)

    meta_policy: dict[str, Any] = {"block": False, "original": None}

    class GuardedMeta(type):
        def __setattr__(cls, name, value):
            if (
                name == "load_model_weights"
                and meta_policy["block"]
                and value is meta_policy["original"]
            ):
                events.append("lmw blocked")
                raise RestoreAbort("load_model_weights restore blocked")
            if name == "load_model_weights" and value is meta_policy["original"]:
                events.append("lmw restored")
            super().__setattr__(name, value)

    def original_ltf(*_args, **_kwargs):
        return None

    utils = GuardedUtils("comfy.utils")
    utils.load_torch_file = original_ltf

    class BaseModel(metaclass=GuardedMeta):
        def load_model_weights(self, *_args, **_kwargs):
            return None

    original_lmw = BaseModel.load_model_weights
    utils_policy.update(block=True, original=original_ltf)
    meta_policy.update(block=True, original=original_lmw)
    model_base = types.ModuleType("comfy.model_base")
    model_base.BaseModel = BaseModel
    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda: events.append("global unload")
    mm.soft_empty_cache = lambda: None
    comfy = types.ModuleType("comfy")
    comfy.utils, comfy.model_base, comfy.model_management = utils, model_base, mm
    for name, module in (
        ("comfy", comfy),
        ("comfy.utils", utils),
        ("comfy.model_base", model_base),
        ("comfy.model_management", mm),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    primary = PrimaryAbort("body interrupted")
    handoff: list[WeightSlab] = []
    with pytest.raises(PrimaryAbort) as caught:
        with comfy_bridge.slab_load(checkpoint, handoff=handoff) as slab:
            arena = slab.arena
            raise primary

    assert caught.value is primary
    assert not arena.closed
    assert handoff == [slab]
    assert slab._close_guard is not None
    restore = slab._close_guard
    assert any(
        owned is restore
        for owned, _error in slab_lifetime._RETAINED_FAILED_LOAD_RESOURCES
    )
    assert any(
        owned is slab for owned, _error in slab_lifetime._RETAINED_FAILED_LOAD_SLABS
    )
    assert slab_lifetime.cleanup_pending()

    healthy_arena_close = arena.close

    def observe_arena_close():
        events.append("arena closed")
        healthy_arena_close()

    monkeypatch.setattr(arena, "close", observe_arena_close)
    utils_policy["block"] = False
    meta_policy["block"] = False
    slab_lifetime.release_after_explicit_unload()

    assert events.index("ltf restored") < events.index("arena closed")
    assert events.index("lmw restored") < events.index("arena closed")
    assert utils.load_torch_file is original_ltf
    assert BaseModel.load_model_weights is original_lmw
    assert arena.closed
    assert handoff == []
    assert not slab_lifetime.cleanup_pending()


def test_nested_provisional_confirmation_keeps_poison_until_last_exact_owner():
    outer = object()
    inner = object()
    outer_token = slab_lifetime.prepublish_resource(outer)
    inner_token = slab_lifetime.prepublish_resource(inner)

    assert outer_token is False and inner_token is True
    slab_lifetime.confirm_prepublished_resource(outer, outer_token)
    assert slab_lifetime.cleanup_poisoned()
    assert slab_lifetime._RETAINED_FAILED_LOAD_RESOURCES == [(inner, None)]
    slab_lifetime.confirm_prepublished_resource(inner, inner_token)
    assert not slab_lifetime.cleanup_pending()


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_prepublish_return_interruption_keeps_exact_process_root(monkeypatch):
    events: list[str] = []

    class Owner:
        def close(self):
            events.append("closed")

    owner = Owner()
    token = None

    def publish():
        nonlocal token
        token = slab_lifetime.prepublish_resource(owner)

    _interrupt_after_call(publish.__code__, "prepublish_resource", publish)

    assert token is None
    assert slab_lifetime.cleanup_poisoned()
    assert slab_lifetime._RETAINED_FAILED_LOAD_RESOURCES == [(owner, None)]
    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda: None
    mm.soft_empty_cache = lambda: None
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    slab_lifetime.release_after_explicit_unload()
    assert events == ["closed"]
    assert not slab_lifetime.cleanup_pending()


def test_lifo_release_forgets_self_confirmed_owners_by_identity(monkeypatch):
    events: list[str] = []

    class Owner:
        def __init__(self, name):
            self.name = name
            self.token = slab_lifetime.prepublish_resource(self)

        def close(self):
            events.append(self.name)
            slab_lifetime.confirm_prepublished_resource(self, self.token)

    outer = Owner("outer")
    inner = Owner("inner")
    mm = types.ModuleType("comfy.model_management")
    mm.unload_all_models = lambda: None
    mm.soft_empty_cache = lambda: None
    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)

    slab_lifetime.release_after_explicit_unload()

    assert events == ["inner", "outer"]
    assert outer is not inner
    assert not slab_lifetime.cleanup_pending()
