"""The Worker side of a split VAE decode: decline, decode, account, release.

``vae_decode`` never raises: a unit this rank cannot decode exactly comes back
as a decline with its reason. A decoded unit returns as a CPU tensor copied
into pages the CPU wrote first. The one resident decode VAE shows in
``status()`` with its bytes, and ``clear_vram`` (either level) and ``unload``
release it. The Worker decodes under the driver's attention functions and
torch backend flags and puts its own back after.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import vae_decode_env
from dgx_monarch.actor import vae_decode
from dgx_monarch.vae_decode_plan import pretouched_cpu


def _request(**overrides):
    request = {
        "schema": 1, "kind": "multidim", "unit": 3, "input": torch.zeros(1, 4, 2, 5, 5, dtype=torch.bfloat16),
        "vae": {"folder": "vae", "name": "wan_2.1_vae.safetensors", "signature": "sig", "metadata": None,
                "dtype": "torch.bfloat16"},
        "output_dtype": "torch.float32", "env": {"vae_class": "Fake"},
    }
    request.update(overrides)
    return request


def _worker(**overrides):
    values = {"rank": 1, "_setup_key": ("topology",), "_active_worker_args": {}, "uma_reserve_gb": 0.0}
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("worker,request_overrides,expected", [
    (_worker(_setup_key=None), {}, "this rank has not completed setup"),
    (_worker(_active_worker_args={"comfy_managed": True}), {}, "comfy_managed residency is on"),
    (_worker(), {"schema": 2}, "not a schema-1 VAE decode unit"),
    (_worker(), {"kind": "audio"}, "names no known unit kind"),
    (_worker(), {"input": [1, 2, 3]}, "must arrive as a CPU tensor"),
    (_worker(), {"vae": {"folder": "vae", "name": "../../etc/passwd", "dtype": "torch.float32"}},
     "not a plain name under the vae folders"),
    (_worker(), {"vae": {"folder": "loras", "name": "x.safetensors", "dtype": "torch.float32"}},
     "not a plain name under the vae folders"),
    (_worker(), {"output_dtype": "torch.int8"}, "names no floating dtype"),
])
def test_a_unit_this_rank_cannot_decode_is_declined_not_raised(worker, request_overrides, expected):
    reply = vae_decode.run(worker, _request(**request_overrides))
    assert reply["ok"] is False and expected in reply["reason"]
    assert reply["schema"] == 1 and reply["rank"] == 1 and "seconds" in reply


class _Fsm(torch.nn.Module):
    def decode(self, latent):
        return latent.float().repeat_interleave(2, dim=-1)

    def _adaptive_decode(self, window):
        return window.repeat_interleave(4, dim=2)


def _held(env=None):
    vae = SimpleNamespace(first_stage_model=_Fsm(), device=torch.device("cpu"), vae_dtype=torch.bfloat16,
                          memory_used_decode=lambda shape, dtype: 1 << 20)
    return vae_decode.ResidentVae(key=("vae", "wan_2.1_vae.safetensors", "sig", "torch.bfloat16", "m"), vae=vae,
                                  label="vae/wan_2.1_vae.safetensors", bytes=254 << 20, dtype="torch.bfloat16",
                                  env=env or {"vae_class": "Fake"}, load_s=2.5)


@pytest.fixture
def comfy_stub(monkeypatch):
    management = types.ModuleType("comfy.model_management")
    management.soft_empty_cache = lambda: None
    comfy = types.ModuleType("comfy")
    comfy.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    monkeypatch.setattr(vae_decode, "_available_bytes", lambda: 64 << 30)
    return management


def test_a_decoded_unit_comes_back_on_the_wire_as_the_stock_function_returns_it(monkeypatch, comfy_stub):
    held = _held()
    monkeypatch.setattr(vae_decode, "_resident", lambda worker, spec: (held, True))
    latent = torch.randn(1, 4, 2, 5, 5).to(torch.bfloat16)
    reply = vae_decode.run(_worker(), _request(input=latent))
    assert reply["ok"] is True and reply["loaded"] is True and reply["load_s"] == 2.5
    assert set(reply) == {"schema", "host", "rank", "unit", "ok", "output", "loaded", "load_s", "seconds"}
    assert reply["output"].device.type == "cpu" and reply["output"].dtype == torch.float32
    assert torch.equal(reply["output"], latent.float().repeat_interleave(2, dim=-1))
    chunk = vae_decode.run(_worker(), _request(kind="h3_chunk", output_dtype=None, input=latent))
    assert chunk["ok"] is True and chunk["output"].dtype == torch.bfloat16
    assert torch.equal(chunk["output"], latent.repeat_interleave(4, dim=2))


def test_a_different_decode_environment_is_declined(monkeypatch, comfy_stub):
    held = _held(env={"vae_class": "Fake", "torch": "2.12.0+cu132", "comfy": "426f5eea5abc"})
    monkeypatch.setattr(vae_decode, "_resident", lambda worker, spec: (held, False))
    reply = vae_decode.run(_worker(), _request(env={"vae_class": "Fake", "torch": "2.11.0", "comfy": "426f5eea5abc"}))
    assert reply["ok"] is False
    assert reply["reason"] == "torch '2.12.0+cu132' differs from the driver's '2.11.0'"


def test_a_unit_that_does_not_fit_is_declined(monkeypatch, comfy_stub):
    held = _held()
    held.vae.memory_used_decode = lambda shape, dtype: 30 << 30
    monkeypatch.setattr(vae_decode, "_resident", lambda worker, spec: (held, False))
    monkeypatch.setattr(vae_decode, "_available_bytes", lambda: 32 << 30)
    reply = vae_decode.run(_worker(), _request())
    assert reply["ok"] is False
    assert reply["reason"] == ("not enough memory to decode the unit: 30.0 GiB plus a 5.0 GiB floor "
                               "against 32.0 GiB available")


def test_the_resident_is_reused_for_the_same_file_and_dtype(monkeypatch, caplog):
    loads = []

    def load(worker, spec):
        loads.append(spec.name)
        held = _held()
        held.key = (spec.folder, spec.name, spec.signature, str(spec.dtype), vae_decode._metadata_digest(spec.metadata))
        return held

    monkeypatch.setattr(vae_decode, "_load", load)
    monkeypatch.setattr(vae_decode, "_empty_cache", lambda: None)
    worker = _worker()
    first = vae_decode._validated(_request())
    assert vae_decode._resident(worker, first)[1] is True
    assert vae_decode._resident(worker, first)[1] is False
    other = vae_decode._validated(_request(vae={**_request()["vae"], "name": "Wan2_2_VAE_bf16.safetensors"}))
    assert vae_decode._resident(worker, other)[1] is True
    assert loads == ["wan_2.1_vae.safetensors", "Wan2_2_VAE_bf16.safetensors"]
    assert "released vae/wan_2.1_vae.safetensors" in caplog.text


def test_the_resident_shows_in_status_and_clear_vram_and_unload_release_it(monkeypatch):
    from dgx_monarch.actor import worker as worker_mod

    monkeypatch.setattr(vae_decode, "_empty_cache", lambda: None)
    actor = worker_mod.GPUWorker.__new__(worker_mod.GPUWorker)
    actor._setup_key = None
    actor.store = SimpleNamespace(unload_all=lambda: None, snapshot=lambda: {})
    actor._vae_resident = _held()
    assert vae_decode.status_block(actor) == {"resident": "vae/wan_2.1_vae.safetensors", "bytes": 254 << 20,
                                             "dtype": "torch.bfloat16"}
    actor._clear_vram_impl("soft")
    assert actor._vae_resident is None and vae_decode.status_block(actor) == {"resident": None, "bytes": 0}
    actor._vae_resident = _held()
    actor._unload_impl()
    assert actor._vae_resident is None


def test_a_teardown_that_releases_models_drops_the_decode_vae(monkeypatch):
    """teardown_group(release_models=True) frees every model, the decode VAE included.

    A teardown that keeps the models is for a process about to stop, whose exit
    frees them; it keeps the decode VAE as it keeps the DiT.
    """
    from dgx_monarch.actor import worker as worker_mod

    parallel_state = types.ModuleType("xfuser.core.distributed.parallel_state")
    parallel_state.destroy_model_parallel = lambda: None
    parallel_state.destroy_distributed_environment = lambda: None
    for name in ("xfuser", "xfuser.core", "xfuser.core.distributed"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, parallel_state.__name__, parallel_state)
    monkeypatch.setattr(vae_decode, "_empty_cache", lambda: None)
    unloads = []
    actor = worker_mod.GPUWorker.__new__(worker_mod.GPUWorker)
    actor.store = SimpleNamespace(unload_all=lambda: unloads.append("dit"), current=None, uncond=None)
    actor._attn = SimpleNamespace(invalidate=lambda: None)

    def set_up():
        actor._setup_key, actor._setup_generation, actor._setup_provenance = ("topology",), 3, None
        actor._setup_cleanup_failed = False
        actor.rank, actor.world, actor.topology = 1, 2, {"ulysses": 2}
        actor._vae_resident = _held()

    set_up()
    reply = actor._teardown_group_impl(True)
    assert reply["torn_down"] is True and reply["cleanup_state"] == "UNSETUP"
    assert unloads == ["dit"] and actor._vae_resident is None
    set_up()
    reply = actor._teardown_group_impl(False)
    assert reply["cleanup_state"] == "DIRTY" and actor._vae_resident is not None


def test_status_reports_the_decode_vae_before_setup(monkeypatch):
    from dgx_monarch.actor import worker_status

    worker = SimpleNamespace(rank=1, world=2, topology={}, store=SimpleNamespace(snapshot=lambda: {}),
                             _setup_key=None, _vae_resident=None)
    monkeypatch.setattr(worker_status, "source_manifest_sha256", lambda: "x")
    info = worker_status.status_impl(worker)
    assert info["vae_decode"] == {"resident": None, "bytes": 0}


def test_output_buffers_are_pre_touched_from_the_cpu_before_the_copy(monkeypatch):
    order = []
    zero, copy = torch.Tensor.zero_, torch.Tensor.copy_
    monkeypatch.setattr(torch.Tensor, "zero_", lambda self: (order.append("zero"), zero(self))[1])
    monkeypatch.setattr(torch.Tensor, "copy_", lambda self, src, *a, **k: (order.append("copy"), copy(self, src, *a, **k))[1])
    source = torch.randn(3, 5).to(torch.bfloat16)
    out = pretouched_cpu(source)
    assert order == ["zero", "copy"]
    assert out.device.type == "cpu" and out.dtype == torch.bfloat16 and torch.equal(out, source)
    assert out.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


def test_the_worker_decodes_under_the_drivers_attention_and_flags_then_restores(monkeypatch):
    def driver_choice(q, k, v, heads, **kwargs):
        return q

    def own_choice(q, k, v, heads, **kwargs):
        return v

    module = types.ModuleType("fake_vae_module")
    module.optimized_attention = own_choice
    driver_module = types.ModuleType("fake_attention_module")
    driver_module.driver_choice = driver_choice
    driver_choice.__module__, driver_choice.__qualname__ = "fake_attention_module", "driver_choice"
    monkeypatch.setitem(sys.modules, "fake_vae_module", module)
    monkeypatch.setitem(sys.modules, "fake_attention_module", driver_module)
    block = torch.nn.Module()
    block.optimized_attention = own_choice
    fsm = torch.nn.Sequential(block)
    before = torch.backends.cudnn.benchmark
    driver_env = {"attention": {"global:fake_vae_module": {"name": "fake_attention_module.driver_choice",
                                                           "registry": None},
                                "attr:optimized_attention": {"name": "fake_attention_module.driver_choice",
                                                             "registry": None}},
                  "flags": {"cudnn.benchmark": not before}}
    with vae_decode_env.adopted(fsm, driver_env):
        assert module.optimized_attention is driver_choice and block.optimized_attention is driver_choice
        assert torch.backends.cudnn.benchmark is (not before)
    assert module.optimized_attention is own_choice and block.optimized_attention is own_choice
    assert torch.backends.cudnn.benchmark is before


def test_an_attention_kernel_this_host_cannot_run_is_a_decline(monkeypatch):
    module = types.ModuleType("fake_vae_module_2")
    module.optimized_attention = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "fake_vae_module_2", module)
    monkeypatch.setattr(vae_decode_env, "_registry", lambda: {"pytorch": object()})
    env = {"attention": {"global:fake_vae_module_2": {"name": "comfy.ldm.modules.attention.attention_sage",
                                                      "registry": "sage"}}}
    with pytest.raises(vae_decode_env.Decline, match="attention 'sage' is not available on this host"):
        with vae_decode_env.adopted(torch.nn.Sequential(), env):
            pass


def test_the_compared_fields_and_their_text():
    driver = {"comfy": "a", "torch": "2.12.0", "cuda": "13.2", "kitchen": "0.2.40", "vae_class": "x.WanVAE"}
    assert vae_decode_env.mismatches(driver, dict(driver)) == []
    here = {**driver, "kitchen": "0.2.37"}
    assert vae_decode_env.mismatches(driver, here) == ["kitchen '0.2.37' differs from the driver's '0.2.40'"]
    assert vae_decode_env.COMPARED == ("comfy", "torch", "cuda", "kitchen", "vae_class")
