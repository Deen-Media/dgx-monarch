"""FLUX.3's cond branch: the prepare_base_model bind, the Worker hook, and the
inference-tensor persistence canary.

Stock ``Flux3.extra_conds`` runs the early text and reference blocks once per
conditioning only under ``torch.inference_mode()``, which ComfyUI enters
around prompt execution and a Worker's sampling thread never does. The
adapter's ``prepare_base_model`` runs stock ``extra_conds`` under inference
mode and clones every inference tensor out. The first half here needs no
ComfyUI: the Worker calls the hook on every load path, before injection, an
adapter without the hook is left alone, and the clone reaches tensors inside
lists. The second half runs real ComfyUI
on the CPU: the bind is idempotent, survives a patcher clone, leaves the flow
map on stock's in-forward branch, and survives the store's whole LoRA life
cycle. That last case is the persistence canary. The clone covers the
returned conds, and nothing else yet proves that no persistent tensor is
created in the inference-mode window and written in place outside it (a cast
cache, an un-bake path). So a tiny real Flux3 is saved in the vendor key
layout, loaded through ComfyUI's diffusion-model load, sampled from a thread
outside inference mode, baked at LoRA strength 0.5, hot-swapped to 0.8 and
un-baked from the file, and every restored weight must equal the file's
bytes. The comfy-canary job runs this file against ComfyUI master.
"""
from __future__ import annotations

import ast
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from module_location_helpers import from_checkout

ACTOR = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch" / "actor"
COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))


# --- The Worker hook (no ComfyUI) -------------------------------------------------


def _inject(monkeypatch, adapter, topology):
    import dgx_monarch.adapters as adapters_pkg
    from dgx_monarch.actor import partial_load_guard, store_fsdp
    from dgx_monarch.actor import worker as worker_mod
    from dgx_monarch.adapters import attention_capability, quant_activation_scale, sol_attention

    monkeypatch.setattr(store_fsdp, "validate_injection", lambda *a, **k: None)
    monkeypatch.setattr(partial_load_guard, "install", lambda *a, **k: None)
    monkeypatch.setattr(worker_mod, "_maybe_compile_dit", lambda *a, **k: None)
    monkeypatch.setattr(adapters_pkg, "adapter_for", lambda *a, **k: adapter)
    monkeypatch.setattr(attention_capability, "assert_kernel_capability", lambda *a, **k: None)
    monkeypatch.setattr(sol_attention, "bind_sol_scope", lambda *a, **k: None)
    monkeypatch.setattr(quant_activation_scale, "install", lambda *a, **k: None)
    attn = SimpleNamespace(bind_capability=lambda *a: None, effective_kernel="TORCH_FLASH")
    fake_self = SimpleNamespace(topology=topology, world=2, _attn=attn,
                                store=SimpleNamespace(family_override=None))
    base_patcher = SimpleNamespace(
        model=SimpleNamespace(diffusion_model=object()), model_options={})
    worker_mod.GPUWorker._inject_for_topology(fake_self, base_patcher, "bf16", [], None)
    return base_patcher


class _HookedAdapter:
    family = "flux3"
    cfg_parallel_supported = True
    cfg_cond_padding = "none"
    attention_head_dim = 128

    def __init__(self):
        self.events: list = []

    def prepare_base_model(self, model):
        self.events.append(("prepare", model))

    def inject_usp(self, diffusion_model, ctx):
        self.events.append(("inject", diffusion_model))


_SINGLE = {"cfg": 1, "ulysses": 1, "ring": 1, "dp": 1, "fsdp": False}
_ULY2 = {"cfg": 1, "ulysses": 2, "ring": 1, "dp": 1, "fsdp": False}
_DP2 = {"cfg": 1, "ulysses": 1, "ring": 1, "dp": 2, "fsdp": False}


@pytest.mark.parametrize("topology", [_SINGLE, _ULY2, _DP2], ids=["single", "uly2", "dp2"])
def test_the_worker_runs_the_hook_once_per_load_on_every_topology(monkeypatch, topology):
    adapter = _HookedAdapter()
    base_patcher = _inject(monkeypatch, adapter, topology)
    prepares = [event for event in adapter.events if event[0] == "prepare"]
    assert prepares == [("prepare", base_patcher.model)]
    # Before any sequence-parallel install: the bind depends on none of them.
    assert adapter.events[0][0] == "prepare"
    if topology["ulysses"] > 1:
        assert adapter.events[1:] == [("inject", base_patcher.model.diffusion_model)]


def test_an_adapter_without_the_hook_is_left_alone(monkeypatch):
    class _Plain:
        family = "krea2"
        cfg_parallel_supported = True
        cfg_cond_padding = "none"

    _inject(monkeypatch, _Plain(), _SINGLE)
    assert not hasattr(_Plain, "prepare_base_model")


def test_every_load_path_hands_the_store_the_worker_injection_as_its_callback():
    """Resident, dual-model slots, the Gate cycle and the FSDP Gate cycle all
    load through ModelStore.ensure with ``on_base_loaded``; each must name the
    Worker's ``_inject_for_topology``, which runs the hook."""
    callbacks = []
    for path in sorted(ACTOR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.keyword) and node.arg == "on_base_loaded":
                if isinstance(node.value, ast.Constant) and node.value.value is None:
                    continue   # a default in a signature
                callbacks.append((path.name, ast.unparse(node.value)))
    files = {name for name, _ in callbacks}
    assert {"worker.py", "sample_protocol.py", "gate_cycle.py", "gate_fsdp_cycle.py"} <= files
    assert all(value.endswith("._inject_for_topology") for _name, value in callbacks), callbacks


def test_the_clone_walks_lists_so_no_listed_inference_tensor_survives():
    """Flux2's latent format hands references through unchanged, so no FLUX.3
    cond holds a listed inference tensor today; the walk keeps a future latent
    format that builds the reference list under inference mode covered."""
    from dgx_monarch.adapters.flux3 import _normal

    with torch.inference_mode():
        first, second = torch.ones(2), torch.arange(3.0)
    plain = torch.zeros(1)
    out = _normal([first, [second, plain], "text", None])
    assert not out[0].is_inference() and not out[1][0].is_inference()
    assert torch.equal(out[0], first) and torch.equal(out[1][0], second)
    assert out[1][1] is plain
    assert out[2:] == ["text", None]


# --- Real ComfyUI -----------------------------------------------------------------


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy():
    """Import real ComfyUI with FLUX.3 on the CPU, then leave sys.modules as found."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before = list(sys.path)
    argv_before = sys.argv
    if os.path.isdir(COMFY_DIR) and COMFY_DIR not in sys.path:
        sys.path.insert(0, COMFY_DIR)
    try:
        sys.argv = ["pytest-flux3-cond-branch", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            pytest.importorskip("comfy.ldm.flux3.model", reason="this ComfyUI predates FLUX.3")
            import comfy.lora
            import comfy.model_base
            import comfy.ops
            import comfy.sample
            import comfy.sd
            from comfy.ldm.flux3 import model as flux3_model
        finally:
            sys.argv = argv_before
        yield SimpleNamespace(lora=comfy.lora, model_base=comfy.model_base, ops=comfy.ops,
                              sample=comfy.sample, sd=comfy.sd, flux3=flux3_model)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, COMFY_DIR)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def _write_vendor_file(comfy, path: Path, *, flow_map=False, seed=0) -> Path:
    """A tiny real Flux3 under the vendor files' key names (the QK norms as
    ``*_norm.scale``, which ComfyUI's Flux loader renames on load)."""
    from safetensors.torch import save_file

    torch.manual_seed(seed)
    tiny = comfy.flux3.Flux3(hidden_size=256, num_heads=2, depth=1, depth_single_blocks=2,
                             flow_map=flow_map, dtype=torch.float32, device="cpu",
                             operations=comfy.ops.disable_weight_init)
    with torch.no_grad():
        for parameter in tiny.parameters():
            parameter.normal_(std=0.05)
    state = {key.replace("_norm.weight", "_norm.scale"): value.contiguous()
             for key, value in tiny.state_dict().items()}
    assert any(key.endswith(".norm.query_norm.scale") for key in state)
    save_file(state, str(path))
    return path


def _load(comfy, path: Path):
    patcher = comfy.sd.load_diffusion_model(str(path))
    assert type(patcher.model) is comfy.model_base.Flux3
    return patcher


def _conds(model, *, refs=True):
    torch.manual_seed(3)
    kwargs = {"cross_attn": torch.randn(1, 80, 20480) * 0.1, "device": torch.device("cpu")}
    if refs:
        kwargs["reference_latents"] = [torch.randn(1, 128, 3, 4)]
    with ThreadPoolExecutor(1) as pool:   # a Worker's sampling thread: inference mode off
        return pool.submit(lambda: model.extra_conds(**kwargs)).result()


def test_the_bind_is_idempotent_and_wraps_the_class_function(comfy, tmp_path):
    from dgx_monarch.adapters.flux3 import Flux3Adapter

    patcher = _load(comfy, _write_vendor_file(comfy, tmp_path / "base.safetensors"))
    model = patcher.model
    assert "txt_states" not in _conds(model)   # stock, off inference mode
    for _ in range(3):
        Flux3Adapter().prepare_base_model(model)
    bound = model.extra_conds
    cells = [cell.cell_contents for cell in bound.__func__.__closure__ or ()]
    assert comfy.model_base.Flux3.extra_conds in cells
    assert not any(isinstance(cell, types.MethodType) for cell in cells)
    first, second = _conds(model), _conds(model)
    assert {"txt_states", "ref_states", "ref_shapes"} <= set(first)
    assert torch.equal(first["txt_states"].cond, second["txt_states"].cond)


def test_the_bound_output_equals_stocks_cache_and_holds_no_inference_tensor(comfy, tmp_path):
    """Off inference mode the bound call returns stock's cached states, each
    an exact copy: outside inference mode an inference tensor refuses any
    in-place write, and ``.to()`` with an unchanged dtype hands it on as is."""
    from dgx_monarch.adapters.flux3 import Flux3Adapter

    patcher = _load(comfy, _write_vendor_file(comfy, tmp_path / "base.safetensors"))
    torch.manual_seed(3)
    kwargs = {"cross_attn": torch.randn(1, 80, 20480) * 0.1, "device": torch.device("cpu"),
              "reference_latents": [torch.randn(1, 128, 3, 4)]}
    with torch.inference_mode():
        stock = patcher.model.extra_conds(**kwargs)
    assert stock["txt_states"].cond.is_inference()
    Flux3Adapter().prepare_base_model(patcher.model)
    bound = _conds(patcher.model)
    assert set(bound) == set(stock) == {"txt_states", "ref_states", "ref_shapes"}
    for key in ("txt_states", "ref_states"):
        assert torch.equal(bound[key].cond, stock[key].cond)
        assert not bound[key].cond.is_inference(), key
    assert bound["ref_shapes"].cond == stock["ref_shapes"].cond == [(3, 4)]


def test_the_bind_survives_a_patcher_clone_and_a_lora_clone(comfy, tmp_path):
    from dgx_monarch.adapters.flux3 import Flux3Adapter

    patcher = _load(comfy, _write_vendor_file(comfy, tmp_path / "base.safetensors"))
    Flux3Adapter().prepare_base_model(patcher.model)
    clone = patcher.clone()
    lora_clone, _clip = comfy.sd.load_lora_for_models(patcher, None, {}, 1.0, 0)
    for other in (clone, lora_clone):
        assert other.model is patcher.model
        assert "txt_states" in _conds(other.model)


def test_the_flow_map_stays_on_stocks_in_forward_branch(comfy, tmp_path):
    from dgx_monarch.adapters.flux3 import Flux3Adapter

    patcher = _load(comfy, _write_vendor_file(comfy, tmp_path / "sd.safetensors", flow_map=True))
    assert patcher.model.diffusion_model.flow_map
    Flux3Adapter().prepare_base_model(patcher.model)
    out = _conds(patcher.model)
    assert {"c_crossattn", "ref_latents"} <= set(out)
    assert not {"txt_states", "ref_states", "ref_shapes"} & set(out)
    tensors = [out["c_crossattn"].cond, *out["ref_latents"].cond]
    assert not any(tensor.is_inference() for tensor in tensors)


def _no_inference_state(model) -> list[str]:
    return [name for name, value in model.state_dict(keep_vars=True).items()
            if torch.is_tensor(value) and value.is_inference()]


def test_the_bound_branch_survives_sample_bake_hot_swap_and_unbake(comfy, tmp_path):
    """The persistence canary: nothing raises, nothing persistent is an
    inference tensor, and the un-bake restores the file's own bytes."""
    from safetensors.torch import load_file

    from dgx_monarch.actor.model_store import ModelStore
    from dgx_monarch.actor.store_bake import _merge_and_free, lazy_swap
    from dgx_monarch.actor.unbake import live_tensor, restore_pristine
    from dgx_monarch.adapters.flux3 import Flux3Adapter

    path = _write_vendor_file(comfy, tmp_path / "flux3_tiny.safetensors")
    patcher = _load(comfy, path)
    Flux3Adapter().prepare_base_model(patcher.model)
    torch.manual_seed(5)
    reference = torch.randn(1, 128, 4, 4)
    positive = [[torch.randn(1, 80, 20480) * 0.1, {"reference_latents": [reference]}]]
    negative = [[torch.zeros(1, 80, 20480), {"reference_latents": [reference]}]]
    noise, latent = torch.randn(1, 128, 4, 4), torch.zeros(1, 128, 4, 4)

    def sample(model_patcher):
        with ThreadPoolExecutor(1) as pool:
            return pool.submit(comfy.sample.sample, model_patcher, noise, 2, 3.5, "euler",
                               "simple", positive, negative, latent, seed=1).result()

    first = sample(patcher)
    assert torch.isfinite(first).all()
    assert _no_inference_state(patcher.model) == []

    key_map = comfy.lora.model_lora_keys_unet(patcher.model, {})
    torch.manual_seed(9)
    lora = {}
    for unit in ("single_blocks.0.q_proj", "txt_mode_blocks.0.v_proj",
                 "content_mode_blocks.image_cond.0.attn_out"):
        lora[f"diffusion_model.{unit}.lora_up.weight"] = torch.randn(256, 4) * 0.1
        lora[f"diffusion_model.{unit}.lora_down.weight"] = torch.randn(4, 256) * 0.1
    # A key the vendor rename reaches: the norm scale the file names `.scale`.
    lora["diffusion_model.txt_mode_blocks.0.norm.query_norm.diff"] = torch.randn(128) * 0.01
    patches = comfy.lora.load_lora(lora, key_map)
    assert len(patches) == 4

    active = patcher.clone()
    active.add_patches(patches, 0.5)
    record = _merge_and_free(active, str(path))
    assert record is not None and len(record.mapped) + len(record.resident) == 4
    baked = sample(active)
    assert not torch.equal(baked, first)

    store = ModelStore()
    store.swap_verify = -1   # comfy's own bake verifies every key
    swapped = patcher.clone()
    swapped.add_patches(patches, 0.8)
    record = lazy_swap(store, SimpleNamespace(unbake=record, base_patcher=patcher),
                       swapped, path.name)
    assert record is not None
    sample(swapped)
    assert _no_inference_state(patcher.model) == []

    restore_pristine(patcher.model, record)
    file_state = load_file(str(path))
    for key in patches:
        name = key[len("diffusion_model."):].replace("_norm.weight", "_norm.scale")
        assert torch.equal(live_tensor(patcher.model, key), file_state[name]), key
    assert torch.equal(sample(patcher), first)
