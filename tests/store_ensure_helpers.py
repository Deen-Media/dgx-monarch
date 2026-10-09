"""Comfy load doubles, the ModelStore rig and the instruction-abort drivers
shared by the store-ensure suites."""
import dis
import json
import struct
import sys
import types
import weakref
from types import SimpleNamespace

import pytest

from dgx_monarch.actor import model_store as ms
from dgx_monarch.actor.model_store import ModelStore
from dgx_monarch.actor.store_detect import LivePrecisionEvidence
from slab_lifetime_helpers import reset_slab_lifetime

STACK = [{"name": "a.safetensors", "strength": 1.0}]


def _run_with_instruction_abort(code, target, primary, operation):
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target:
            monitoring.set_local_events(tool_id, code, 0)
            raise primary

    monitoring.use_tool_id(tool_id, "dgxm-model-store-handoff-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(tool_id, code, monitoring.events.INSTRUCTION)
    try:
        operation()
    finally:
        monitoring.set_local_events(tool_id, code, 0)
        monitoring.register_callback(
            tool_id, monitoring.events.INSTRUCTION, None)
        monitoring.free_tool_id(tool_id)


def _run_with_recovery_abort(
    code,
    primary_target,
    recovery_target,
    primary,
    recovery,
    operation,
):
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)
    primary_raised = False

    def interrupt(_code, instruction_offset):
        nonlocal primary_raised
        if instruction_offset == primary_target and not primary_raised:
            primary_raised = True
            raise primary
        if instruction_offset == recovery_target and primary_raised:
            monitoring.set_local_events(tool_id, code, 0)
            raise recovery

    monitoring.use_tool_id(tool_id, "dgxm-model-store-recovery-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(tool_id, code, monitoring.events.INSTRUCTION)
    try:
        operation()
    finally:
        monitoring.set_local_events(tool_id, code, 0)
        monitoring.register_callback(
            tool_id, monitoring.events.INSTRUCTION, None)
        monitoring.free_tool_id(tool_id)


def _target_after_named_call(code, name, *, latest_source=False):
    instructions = list(dis.get_instructions(code))
    loads = [
        index for index, instruction in enumerate(instructions)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_ATTR"}
        and instruction.argval == name
    ]
    load_index = (
        max(
            loads,
            key=lambda index: instructions[index].positions.lineno or -1,
        )
        if latest_source else loads[0]
    )
    call_index = next(
        index for index in range(load_index + 1, len(instructions))
        if instructions[index].opname == "CALL"
    )
    return instructions[call_index + 1]


def _write_one_tensor_safetensors(path, dtype: str) -> None:
    item_bytes = {"BF16": 2, "F16": 2}[dtype]
    header = json.dumps({
        "weight": {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [0, item_bytes],
        },
    }).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(item_bytes))


class _FakeBase:
    def __init__(self):
        self.model = SimpleNamespace(diffusion_model="DIT_MODULE")
        self.offload_device = "cpu"


class _FakeSlab:
    def __init__(self, on_close=None):
        self.closed = 0
        self.reabsorbed = []
        self.total_gib = 24.5
        self.on_close = on_close

    def reabsorb(self, module, prefix=""):
        self.reabsorbed.append(module)
        return {"reabsorbed": 3, "reabsorbed_gib": 1.0, "skipped": 0,
                "quant_stray_gib": 0.0}

    def close(self):
        if self.on_close is not None:
            self.on_close()
        self.closed += 1

    def telemetry(self):
        return {"slab_gib": self.total_gib}


@pytest.fixture
def rig(monkeypatch):
    reset_slab_lifetime(monkeypatch)
    calls = SimpleNamespace(
        load=0, unload_all=0, soft_empty=0, slabs=[], slab_refs=[],
        base_refs=[], live_models=[], close_guard=None, close_events=[],
        model_options=[],
    )

    mm_stub = types.ModuleType("comfy.model_management")
    mm_stub.get_torch_device = lambda: "cuda:0"
    mm_stub.soft_empty_cache = lambda: setattr(
        calls, "soft_empty", calls.soft_empty + 1)
    def unload_all_models():
        calls.unload_all += 1
        calls.live_models.clear()

    mm_stub.unload_all_models = unload_all_models

    comfy = types.ModuleType("comfy")
    comfy_sd = types.ModuleType("comfy.sd")

    def load_diffusion_model(path, model_options=None):
        calls.load += 1
        calls.model_options.append(model_options)
        base = _FakeBase()
        calls.base_refs.append(weakref.ref(base))
        calls.live_models.append(base)
        return base

    comfy_sd.load_diffusion_model = load_diffusion_model
    comfy.sd = comfy_sd
    comfy.model_management = mm_stub
    for name, mod in (("comfy", comfy), ("comfy.sd", comfy_sd),
                      ("comfy.model_management", mm_stub)):
        monkeypatch.setitem(sys.modules, name, mod)

    from contextlib import contextmanager

    from dgx_monarch.actor import comfy_bridge

    def record_close():
        if calls.close_guard is not None:
            calls.close_guard()
        calls.close_events.append("close")

    @contextmanager
    def fake_slab_load(path, *, handoff=None):
        slab = _FakeSlab(record_close)
        calls.slabs.append(slab)
        calls.slab_refs.append(weakref.ref(slab))
        if handoff is not None:
            handoff.append(slab)
        try:
            yield slab
        except BaseException:
            slab.close()
            raise

    monkeypatch.setattr(comfy_bridge, "slab_load", fake_slab_load)
    monkeypatch.setattr(ms, "resolve_model_path",
                        lambda kind, name: f"/fake/{name}")
    monkeypatch.setattr(ms, "_detect_checkpoint_kind", lambda _path, _kind: None)
    monkeypatch.setattr(
        "dgx_monarch.gate_ledger.artifact_signature", lambda path: path)
    monkeypatch.setattr(
        ms,
        "_detect_live_precision",
        lambda _patcher, _kind: LivePrecisionEvidence("bf16", "all_bf16"),
    )
    monkeypatch.setattr(ms, "_detect_family", lambda p: "krea2")
    monkeypatch.setattr(ms, "_merge_and_free", lambda active, base_path=None: None)
    monkeypatch.setattr(
        ModelStore, "_build_active",
        lambda self, base, stack: SimpleNamespace(tag="ACTIVE", model=base.model))

    store = ModelStore()
    store.lora_low_rss = True
    return store, calls
