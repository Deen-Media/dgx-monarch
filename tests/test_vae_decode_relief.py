"""vae_decode_relief.py: before a VAE decode priced above free memory, the driver frees what it needs.

On by default at node import on a GPU that shares the host's memory. These
tests hold the trigger (ComfyUI's own price and reserve against its reading),
what is kept loaded (the VAE in hand, every diffusion model), when everything
is kept (inside a sampler, a render in flight, the unload switch, DynamicVRAM
off), that the caller's decode is stock's own call with the caller's own
arguments, how it fails open, and the gate. On real ComfyUI (CPU, random
weights; skipped where ComfyUI is not importable) a decode gives the same
tensor with the relief acting as without it.
"""
from __future__ import annotations

import os
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import driver_uma_memory, driver_unload, mesh_safety, vae_decode_relief
from module_location_helpers import from_checkout

GIB = 2 ** 30


class _BaseModel:
    """Stands in for ``comfy.model_base.BaseModel``: what marks a diffusion model."""


class _Flux(_BaseModel):
    pass


class _TextEncoder:
    pass


class _ClipVision:
    pass


@pytest.fixture(autouse=True)
def _a_driver_with_dynamic_vram(monkeypatch):
    """The state every case starts from: the rule unset, no sampler running, DynamicVRAM on, no switch."""
    monkeypatch.setitem(vae_decode_relief._STATE, "state", vae_decode_relief.UNSET)
    monkeypatch.setattr(vae_decode_relief._SAMPLING, "depth", 0, raising=False)
    monkeypatch.setitem(sys.modules, "comfy.memory_management", SimpleNamespace(aimdo_enabled=True))
    monkeypatch.setitem(sys.modules, "comfy.model_base", SimpleNamespace(BaseModel=_BaseModel))
    monkeypatch.delenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, raising=False)
    monkeypatch.setattr(driver_unload, "busy_reason", lambda handle: "")
    monkeypatch.setattr(vae_decode_relief, "_SETTLE_S", 0.0)


class _Vae:
    """What the relief reads of a VAE: its price, its dtype, its device and its patcher."""

    def __init__(self, price=4 * GIB, latent_dim=3):
        self.latent_dim = latent_dim
        self.vae_dtype = "bf16"
        self.device = "cuda:0"
        self.patcher = SimpleNamespace(model=object())
        self.priced = []
        self._price = price

    def memory_used_decode(self, shape, dtype):
        self.priced.append((shape, dtype))
        if isinstance(self._price, Exception):
            raise self._price
        return self._price


def _loaded(inner):
    """One entry of ``current_loaded_models``: a loaded model whose patcher holds ``inner``."""
    return SimpleNamespace(model=SimpleNamespace(model=inner))


def _model_management(readings, *, held=(), inference=GIB, reserve=GIB // 2, ran=None):
    """ComfyUI's model management as the relief reads it; ``readings`` are the free figures in order."""
    left = list(readings)
    mm = SimpleNamespace(asked=[], freed=[], current_loaded_models=list(held),
                         minimum_inference_memory=lambda: inference,
                         extra_reserved_memory=lambda: reserve)

    def get_free_memory(device=None, torch_free_too=False):
        mm.asked.append(device)
        figure = left.pop(0) if len(left) > 1 else left[0]
        if isinstance(figure, Exception):
            raise figure
        return figure

    def free_memory(memory_required, device, keep_loaded=(), for_dynamic=False):
        mm.freed.append((memory_required, device, list(keep_loaded), for_dynamic))
        if ran is not None:
            ran.append("free_memory")
        return []

    def soft_empty_cache():
        if ran is not None:
            ran.append("soft_empty_cache")

    mm.get_free_memory, mm.free_memory, mm.soft_empty_cache = get_free_memory, free_memory, soft_empty_cache
    return mm


def _available(monkeypatch, *figures):
    """MemAvailable as ``mesh_safety`` reads it, one figure a read."""
    left = list(figures)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: left.pop(0) if len(left) > 1 else left[0])


def _decode_spy(ran=None):
    """A stand-in for ``VAE.decode`` that records its calls and returns one object."""
    calls, answer = [], object()

    def decode(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        if ran is not None:
            ran.append("decode")
        return answer

    return decode, calls, answer


def _wrapped(monkeypatch, mm, ran=None):
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    decode, calls, answer = _decode_spy(ran)
    return vae_decode_relief.with_relief(decode), calls, answer


LATENT = torch.zeros(1, 4, 2, 8, 8)


def test_the_need_is_comfyuis_own_price_and_its_loaders_reserve():
    latent = torch.zeros(1, 16, 22, 104, 60)
    vae = _Vae(price=5 * GIB)
    mm = _model_management([0], inference=GIB, reserve=GIB // 2)
    assert vae_decode_relief.decode_need(mm, vae, latent) == (5 * GIB, 5 * GIB + GIB // 2)
    assert vae.priced == [((1, 16, 22, 104, 60), "bf16")]
    # A small decode still needs ComfyUI's inference floor, as load_models_gpu takes the larger.
    assert vae_decode_relief.decode_need(mm, _Vae(price=GIB // 4), latent) == (GIB // 4, GIB)
    # ComfyUI's decode drops the frame axis of a 5-D latent for a 2-D VAE before it prices it.
    image_vae = _Vae(latent_dim=2)
    vae_decode_relief.decode_need(mm, image_vae, torch.zeros(2, 16, 1, 104, 60))
    assert image_vae.priced == [((2, 16, 104, 60), "bf16")]
    image_vae.priced.clear()
    vae_decode_relief.decode_need(mm, image_vae, torch.zeros(2, 16, 104, 60))
    assert image_vae.priced == [((2, 16, 104, 60), "bf16")]
    # A ComfyUI that raises its reserve or its floor is followed, not copied.
    assert vae_decode_relief.decode_need(_model_management([0], inference=9 * GIB, reserve=8 * GIB),
                                         _Vae(price=5 * GIB), latent) == (5 * GIB, 13 * GIB)


def test_a_decode_that_fits_is_left_alone(monkeypatch, caplog):
    ran = []
    mm = _model_management([5 * GIB], held=[_loaded(_TextEncoder())], ran=ran)
    wrapped, calls, answer = _wrapped(monkeypatch, mm, ran)
    vae = _Vae(price=4 * GIB)
    with caplog.at_level("INFO"):
        assert wrapped(vae, LATENT) is answer
    assert ran == ["decode"] and calls == [(vae, (LATENT,), {})]
    assert mm.asked == ["cuda:0"] and mm.freed == [] and "decode relief" not in caplog.text
    assert vae_decode_relief.state() == vae_decode_relief.UNSET


def test_a_decode_priced_above_the_reading_frees_what_it_needs_and_then_decodes_with_the_callers_arguments(
        monkeypatch, caplog):
    ran = []
    vae = _Vae(price=4 * GIB)
    own, flux = SimpleNamespace(model=vae.patcher), _loaded(_Flux())
    held = [_loaded(_TextEncoder()), own, flux, _loaded(_ClipVision())]
    mm = _model_management([2 * GIB, 30 * GIB], held=held, ran=ran)
    _available(monkeypatch, 14 * GIB, 46 * GIB)
    wrapped, calls, answer = _wrapped(monkeypatch, mm, ran)
    options = {"tile": 1}
    with caplog.at_level("INFO"):
        assert wrapped(vae, LATENT, vae_options=options) is answer
    # ComfyUI's own classic-loader call, for the loader's own requirement, before the decode.
    assert ran == ["free_memory", "soft_empty_cache", "decode"]
    ((need, device, kept, for_dynamic),) = mm.freed
    assert need == 4 * GIB + GIB // 2 and device == "cuda:0" and for_dynamic is False
    # The VAE in hand and the diffusion model stay loaded, whoever is using them.
    assert kept == [own, flux]
    (call,) = calls
    assert call[0] is vae and call[1][0] is LATENT and call[2]["vae_options"] is options
    assert caplog.text.count("decode relief:") == 1
    line = next(record.getMessage() for record in caplog.records if "decode relief:" in record.getMessage())
    assert "priced this VAE decode at 4.0 GiB (4.5 GiB with its reserve) against 2.0 GiB free" in line
    assert "from the 2 idle models it held (_TextEncoder, _ClipVision)" in line
    assert "MemAvailable read 14.0 GiB before that and 46.0 GiB after it" in line
    assert line.endswith("the decode starts with 30.0 GiB free")
    # By name too.
    assert wrapped(vae, samples_in=LATENT) is answer and calls[-1][2] == {"samples_in": LATENT}


def test_a_decode_still_short_after_the_relief_runs_as_comfyui_runs_it_and_the_line_says_so(monkeypatch, caplog):
    mm = _model_management([GIB, 3 * GIB], held=[_loaded(_TextEncoder())])
    _available(monkeypatch, 8 * GIB, 10 * GIB)
    wrapped, _calls, answer = _wrapped(monkeypatch, mm)
    with caplog.at_level("INFO"):
        assert wrapped(_Vae(price=4 * GIB), LATENT) is answer
    assert "from the 1 idle model it held (_TextEncoder)" in caplog.text
    assert caplog.records[-1].getMessage().endswith(
        "the decode is still priced above the 3.0 GiB free and runs as ComfyUI runs it")


@pytest.mark.parametrize("held", ["nothing", "the vae in hand", "a diffusion model", "both"])
def test_with_nothing_idle_nothing_is_freed_and_one_line_says_so(monkeypatch, caplog, held):
    """On a box that stays tight a decode must not free its own VAE, nor a model a sampler may be using."""
    vae = _Vae(price=4 * GIB)
    entries = {"nothing": [], "the vae in hand": [SimpleNamespace(model=vae.patcher)],
               "a diffusion model": [_loaded(_Flux())],
               "both": [SimpleNamespace(model=vae.patcher), _loaded(_Flux())]}[held]
    ran = []
    mm = _model_management([GIB], held=entries, ran=ran)
    wrapped, _calls, answer = _wrapped(monkeypatch, mm, ran)
    with caplog.at_level("INFO"):
        assert wrapped(vae, LATENT) is answer
        assert wrapped(vae, LATENT) is answer
    assert ran == ["decode", "decode"] and mm.freed == []
    assert caplog.text.count("decode relief:") == 1
    assert "ComfyUI held no idle model here to free, so the decode runs as ComfyUI runs it" in caplog.text


def test_an_entry_whose_model_is_gone_is_neither_kept_nor_named(monkeypatch):
    vae = _Vae()
    dead = SimpleNamespace(model=None)
    mm = _model_management([0], held=[dead, _loaded(_TextEncoder()), _loaded(_Flux())])
    keep, idle = vae_decode_relief.kept_and_idle(mm, vae)
    assert keep == [mm.current_loaded_models[2]] and idle == ["_TextEncoder"]
    # A patcher with no inner model is named by its own class and counts as idle.
    bare = SimpleNamespace(model=_ClipVision())
    mm.current_loaded_models = [bare]
    assert vae_decode_relief.kept_and_idle(mm, vae) == ([], ["_ClipVision"])


def test_a_decode_inside_a_running_sampler_is_a_preview_and_frees_nothing(monkeypatch, caplog):
    """Preview nodes call VAE.decode at every step; freeing there could unload what the sampler uses."""
    ran = []
    mm = _model_management([GIB], held=[_loaded(_TextEncoder())], ran=ran)
    wrapped, _calls, answer = _wrapped(monkeypatch, mm, ran)
    vae = _Vae(price=4 * GIB)
    seen = []

    def sample(self, noise, callback=None):
        seen.append(vae_decode_relief.sampling_depth())
        for _step in range(3):
            assert wrapped(vae, LATENT) is answer  # the per-step preview
        return "latent"

    guarded = vae_decode_relief.with_sampler_guard(sample)
    assert guarded.__name__ == "sample" and guarded.__wrapped__ is sample
    with caplog.at_level("INFO"):
        assert guarded(object(), "noise", callback=None) == "latent"
    assert seen == [1] and vae_decode_relief.sampling_depth() == 0
    assert ran == ["decode"] * 3 and mm.freed == []
    # Said once, not at every step.
    assert caplog.text.count("decode relief:") == 1
    assert "the driver kept what it holds because a sampler is running in this thread" in caplog.text
    # Once the sampler has returned, a decode is relieved as usual.
    _available(monkeypatch, 8 * GIB, 30 * GIB)
    assert wrapped(vae, LATENT) is answer and len(mm.freed) == 1


def test_the_sampler_count_is_per_thread_nests_and_comes_back_after_an_error():
    depths = []

    def inner(self):
        depths.append(vae_decode_relief.sampling_depth())
        raise RuntimeError("the sampler failed")

    guarded_inner = vae_decode_relief.with_sampler_guard(inner)

    def outer(self):
        depths.append(vae_decode_relief.sampling_depth())
        with pytest.raises(RuntimeError):
            guarded_inner(self)
        depths.append(vae_decode_relief.sampling_depth())
        other = []
        thread = threading.Thread(target=lambda: other.append(vae_decode_relief.sampling_depth()))
        thread.start()
        thread.join()
        return other

    assert vae_decode_relief.with_sampler_guard(outer)(object()) == [0]  # another thread is not sampling
    assert depths == [1, 2, 1] and vae_decode_relief.sampling_depth() == 0


@pytest.mark.parametrize("why", [
    "a render is still running in this driver",
    "the driver could not read whether a render is running",
])
def test_a_render_in_flight_keeps_everything_and_says_so_once(monkeypatch, caplog, why):
    """A pipelined render decodes one result while later ones sample."""
    asked = []
    monkeypatch.setattr(driver_unload, "busy_reason", lambda handle: asked.append(handle) or why)
    ran = []
    mm = _model_management([GIB], held=[_loaded(_TextEncoder())], ran=ran)
    wrapped, _calls, answer = _wrapped(monkeypatch, mm, ran)
    with caplog.at_level("INFO"):
        assert wrapped(_Vae(price=4 * GIB), LATENT) is answer
        assert wrapped(_Vae(price=4 * GIB), LATENT) is answer
    assert ran == ["decode", "decode"] and mm.freed == [] and asked == [None, None]
    assert caplog.text.count("decode relief:") == 1
    assert f"the driver kept what it holds because {why}" in caplog.text
    assert vae_decode_relief.state() == vae_decode_relief.UNSET


def test_the_drivers_unload_switch_is_honoured(monkeypatch, caplog):
    """DGXM_DISABLE_DRIVER_UNLOAD=1 means the driver unloads none of ComfyUI's models on its own."""
    monkeypatch.setenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, "1")
    mm = _model_management([GIB], held=[_loaded(_TextEncoder())])
    wrapped, _calls, answer = _wrapped(monkeypatch, mm)
    with caplog.at_level("INFO"):
        assert wrapped(_Vae(price=4 * GIB), LATENT) is answer
    assert mm.freed == []
    assert "kept what it holds because the driver runs with DGXM_DISABLE_DRIVER_UNLOAD=1" in caplog.text


@pytest.mark.parametrize("memory_management", [SimpleNamespace(aimdo_enabled=False), SimpleNamespace(), None])
def test_with_dynamic_vram_off_comfyuis_own_loader_unloads_and_the_relief_does_nothing(
        monkeypatch, caplog, memory_management):
    """The gap is DynamicVRAM's: the classic loader already unloads for a decode, and copies as it does."""
    if memory_management is None:
        monkeypatch.delitem(sys.modules, "comfy.memory_management")
    else:
        monkeypatch.setitem(sys.modules, "comfy.memory_management", memory_management)
    mm = _model_management([GIB], held=[_loaded(_TextEncoder())])
    wrapped, _calls, answer = _wrapped(monkeypatch, mm)
    vae = _Vae(price=4 * GIB)
    with caplog.at_level("INFO"):
        assert wrapped(vae, LATENT) is answer
    assert mm.freed == [] and mm.asked == [] and vae.priced == [] and "decode relief" not in caplog.text


@pytest.mark.parametrize("broken", ["reading", "floor", "busy_reason", "model_base", "free_memory", "available"])
def test_any_failure_steps_aside_once_and_the_decode_in_hand_still_runs(monkeypatch, caplog, broken):
    ran = []
    mm = _model_management([RuntimeError("reading broke")] if broken == "reading" else [GIB, 2 * GIB],
                           held=[_loaded(_TextEncoder())], ran=ran)
    _available(monkeypatch, 8 * GIB, 9 * GIB)

    def broke(*_args, **_kwargs):
        raise RuntimeError(f"{broken} broke")

    if broken == "floor":
        mm.minimum_inference_memory = broke
    elif broken == "busy_reason":
        monkeypatch.setattr(driver_unload, "busy_reason", broke)
    elif broken == "model_base":
        monkeypatch.delitem(sys.modules, "comfy.model_base")
    elif broken == "free_memory":
        mm.free_memory = broke
    elif broken == "available":
        monkeypatch.setattr(mesh_safety, "mem_available_bytes", broke)
    wrapped, _calls, answer = _wrapped(monkeypatch, mm, ran)
    vae = _Vae(price=4 * GIB)
    with caplog.at_level("INFO"):
        assert wrapped(vae, LATENT) is answer
    assert ran[-1] == "decode"
    assert caplog.text.count("decode relief stepped aside") == 1
    assert vae_decode_relief.state() == vae_decode_relief.STEPPED_ASIDE
    # From then on every call is stock's and the relief does no work at all.
    ran.clear()
    vae.priced.clear()
    asked = len(mm.asked)
    assert wrapped(vae, LATENT) is answer
    assert ran == ["decode"] and vae.priced == [] and len(mm.asked) == asked
    assert caplog.text.count("decode relief stepped aside") == 1


def test_a_vae_that_cannot_price_its_decode_is_that_decodes_matter(monkeypatch, caplog):
    """ComfyUI's own decode meets the same call; one odd VAE must not stand the rule down for the process."""
    ran = []
    mm = _model_management([GIB, 30 * GIB], held=[_loaded(_TextEncoder())], ran=ran)
    _available(monkeypatch, 8 * GIB, 30 * GIB)
    wrapped, _calls, answer = _wrapped(monkeypatch, mm, ran)
    with caplog.at_level("INFO"):
        assert wrapped(_Vae(price=NotImplementedError("no generic decode")), LATENT) is answer
        assert wrapped(_Vae(price=NotImplementedError("no generic decode")), LATENT) is answer
    assert ran == ["decode", "decode"] and vae_decode_relief.state() == vae_decode_relief.UNSET
    assert caplog.text.count("could not be read") == 1 and "stepped aside" not in caplog.text
    # The next VAE that can price is relieved as before.
    assert wrapped(_Vae(price=4 * GIB), LATENT) is answer
    assert ran[-3:] == ["free_memory", "soft_empty_cache", "decode"]


def test_what_is_not_an_ordinary_error_passes_through(monkeypatch):
    """ComfyUI's interrupt is a BaseException: a cancelled prompt must stay cancelled."""

    class Interrupt(BaseException):
        pass

    def interrupted(*_args, **_kwargs):
        raise Interrupt

    mm = _model_management([GIB], held=[_loaded(_TextEncoder())])
    mm.free_memory = interrupted
    _available(monkeypatch, 8 * GIB)
    wrapped, calls, _answer = _wrapped(monkeypatch, mm)
    with pytest.raises(Interrupt):
        wrapped(_Vae(price=4 * GIB), LATENT)
    assert calls == [] and vae_decode_relief.state() == vae_decode_relief.UNSET


def test_a_call_it_cannot_read_goes_straight_to_comfyui(monkeypatch):
    ran = []
    mm = _model_management([GIB], held=[_loaded(_TextEncoder())], ran=ran)
    wrapped, calls, answer = _wrapped(monkeypatch, mm, ran)
    vae = _Vae()
    assert wrapped(vae, latent=LATENT) is answer  # a name this pack version does not know
    assert ran == ["decode"] and mm.asked == [] and calls[0][2].keys() == {"latent"}
    # And with ComfyUI's model management not imported in this process there is nothing to read.
    monkeypatch.delitem(sys.modules, "comfy.model_management")
    assert wrapped(vae, LATENT) is answer and mm.asked == []


def _comfy_classes(monkeypatch):
    class VAE:
        def decode(self, samples_in, vae_options=None):
            return samples_in

        def decode_tiled(self, samples):
            return samples

        def encode(self, pixel_samples):
            return pixel_samples

    class CFGGuider:
        def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None,
                   disable_pbar=False, seed=None):
            return vae_decode_relief.sampling_depth()

    monkeypatch.setitem(sys.modules, "comfy.sd", SimpleNamespace(VAE=VAE))
    monkeypatch.setitem(sys.modules, "comfy.samplers", SimpleNamespace(CFGGuider=CFGGuider))
    return VAE, CFGGuider


def test_install_wraps_the_sampler_and_the_decode_once_and_nothing_else(monkeypatch):
    VAE, CFGGuider = _comfy_classes(monkeypatch)
    stock_decode, stock_tiled, stock_encode, stock_sample = VAE.decode, VAE.decode_tiled, VAE.encode, CFGGuider.sample
    vae_decode_relief.install()
    first_decode, first_sample = VAE.decode, CFGGuider.sample
    assert first_decode is not stock_decode and first_decode.__name__ == "decode"
    assert first_decode.__wrapped__ is stock_decode and first_sample.__wrapped__ is stock_sample
    assert CFGGuider().sample("noise", "latent", "sampler", "sigmas") == 1  # counted while it runs
    assert vae_decode_relief.sampling_depth() == 0
    vae_decode_relief.install()  # a second install does not stack
    assert VAE.decode is first_decode and CFGGuider.sample is first_sample
    assert VAE.decode_tiled is stock_tiled and VAE.encode is stock_encode  # neither is touched
    vae_decode_relief.uninstall()
    assert VAE.decode is stock_decode and CFGGuider.sample is stock_sample
    assert vae_decode_relief.state() == vae_decode_relief.UNSET
    vae_decode_relief.uninstall()  # and undoing twice is harmless
    assert VAE.decode is stock_decode and CFGGuider.sample is stock_sample


def test_the_decode_is_never_wrapped_without_the_sampler_guard(monkeypatch):
    """A ComfyUI whose sampler entry is gone gets no relief at all, never a relief that cannot see a sampler."""
    VAE, _guider = _comfy_classes(monkeypatch)
    stock_decode = VAE.decode
    monkeypatch.setitem(sys.modules, "comfy.samplers", SimpleNamespace())
    with pytest.raises(AttributeError):
        vae_decode_relief.install()
    assert VAE.decode is stock_decode
    monkeypatch.delitem(sys.modules, "comfy.samplers")
    with pytest.raises(ImportError):
        vae_decode_relief.install()
    assert VAE.decode is stock_decode


def _comfy_stand_in(monkeypatch, *, total):
    """Stand-ins for what the gate reads: ComfyUI's modules in this process and the integrated probe."""
    model_management = SimpleNamespace(get_torch_device=lambda: "cuda:0")
    monkeypatch.setattr(driver_uma_memory, "comfy_modules", lambda: (model_management, object()))

    def probe(_torch, device):
        assert device == "cuda:0"
        if isinstance(total, Exception):
            raise total
        return total

    monkeypatch.setattr(driver_uma_memory, "integrated_total_bytes", probe)
    installs = []
    monkeypatch.setattr(vae_decode_relief, "install", lambda: installs.append("install"))
    return installs


def test_on_by_default_on_an_integrated_gpu_and_only_there(monkeypatch, caplog):
    monkeypatch.delenv(vae_decode_relief.DISABLE_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    installs = _comfy_stand_in(monkeypatch, total=124 * GIB)
    with caplog.at_level("INFO"):
        assert vae_decode_relief.install_driver() == vae_decode_relief.INSTALLED
    assert installs == ["install"] and vae_decode_relief.state() == vae_decode_relief.INSTALLED
    assert "free what the decode needs from the models it holds idle" in caplog.text
    assert vae_decode_relief.DISABLE_ENV in caplog.text
    # A discrete GPU or the CPU: ComfyUI's own decode, nothing installed.
    installs = _comfy_stand_in(monkeypatch, total=None)
    assert vae_decode_relief.install_driver() == vae_decode_relief.NOT_INTEGRATED and installs == []
    assert vae_decode_relief.state() == vae_decode_relief.NOT_INTEGRATED
    installs = _comfy_stand_in(monkeypatch, total=RuntimeError("no device"))
    assert vae_decode_relief.install_driver() == vae_decode_relief.PROBE_FAILED and installs == []
    monkeypatch.setattr(sys, "platform", "darwin")
    installs = _comfy_stand_in(monkeypatch, total=124 * GIB)
    assert vae_decode_relief.install_driver() == vae_decode_relief.NOT_LINUX and installs == []


def test_its_own_switch_turns_it_off_and_the_readings_switch_does_not(monkeypatch, caplog):
    monkeypatch.setattr(sys, "platform", "linux")
    installs = _comfy_stand_in(monkeypatch, total=124 * GIB)
    monkeypatch.setenv(vae_decode_relief.DISABLE_ENV, "1")
    with caplog.at_level("INFO"):
        assert vae_decode_relief.install_driver() == vae_decode_relief.OFF and installs == []
    assert vae_decode_relief.DISABLE_ENV in caplog.text and vae_decode_relief.state() == vae_decode_relief.OFF
    assert vae_decode_relief.DISABLE_ENV == "DGXM_DISABLE_VAE_DECODE_RELIEF"
    monkeypatch.delenv(vae_decode_relief.DISABLE_ENV)
    monkeypatch.setenv(driver_uma_memory.DISABLE_ENV, "1")
    assert driver_uma_memory.install_driver() == "off"
    assert vae_decode_relief.install_driver() == vae_decode_relief.INSTALLED and installs == ["install"]


def test_the_driver_hook_never_raises(monkeypatch):
    monkeypatch.delenv(vae_decode_relief.DISABLE_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    _comfy_stand_in(monkeypatch, total=124 * GIB)

    def broken():
        raise ImportError("No module named 'comfy'")

    monkeypatch.setattr(vae_decode_relief, "install", broken)
    assert vae_decode_relief.install_driver() == vae_decode_relief.NO_COMFY
    assert vae_decode_relief.state() == vae_decode_relief.NO_COMFY


class _RecordComfyImports:
    """A meta-path finder that records every attempt to import ComfyUI and finds nothing."""

    def __init__(self):
        self.seen = []

    def find_spec(self, name, path=None, target=None):
        if name == "comfy" or name.startswith("comfy."):
            self.seen.append(name)
        return None


def test_the_hook_imports_nothing_of_comfyui_itself(monkeypatch):
    """The finder records an attempt and finds nothing; the hook reads only what is already imported."""
    monkeypatch.delenv(vae_decode_relief.DISABLE_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    for name in [n for n in sys.modules if n == "comfy" or n.startswith("comfy.")]:
        monkeypatch.delitem(sys.modules, name)
    finder = _RecordComfyImports()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    assert vae_decode_relief.install_driver() == vae_decode_relief.NO_COMFY
    with pytest.raises(ImportError):
        vae_decode_relief.install()
    assert finder.seen == []


def test_the_entry_point_installs_the_relief_after_the_reading_and_the_encode_rule():
    import ast
    import pathlib

    tree = ast.parse((pathlib.Path(__file__).resolve().parents[1] / "__init__.py").read_text())
    calls = [f"{stmt.value.func.value.id}.{stmt.value.func.attr}" for stmt in tree.body
             if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
             and isinstance(stmt.value.func, ast.Attribute) and isinstance(stmt.value.func.value, ast.Name)]
    assert "vae_decode_relief.install_driver" in calls
    assert (calls.index("driver_uma_memory.install_driver") < calls.index("vae_encode_item.install_driver")
            < calls.index("vae_decode_relief.install_driver"))


def test_the_surface_canary_holds_the_calls_the_relief_makes():
    """An upstream change of a call the relief makes fails the daily canary, not a user's decode."""
    from dgx_monarch import comfy_surface

    paths = {touchpoint.path: touchpoint for touchpoint in comfy_surface.TOUCHPOINTS}
    assert paths["comfy.sd.VAE.decode"].positional == ("self", "samples_in")
    assert paths["comfy.model_management.free_memory"].positional == ("memory_required", "device")
    for name in ("comfy.model_management.minimum_inference_memory",
                 "comfy.model_management.extra_reserved_memory",
                 "comfy.model_management.soft_empty_cache", "comfy.samplers.CFGGuider.sample"):
        assert name in paths

    def renamed(self, latent):
        return latent

    with pytest.raises(RuntimeError):
        comfy_surface._assert_touchpoint(renamed, paths["comfy.sd.VAE.decode"])
    comfy_surface._assert_touchpoint(lambda self, samples_in, vae_options=None: samples_in,
                                     paths["comfy.sd.VAE.decode"])
    # The classic loader's call must still take the two keywords the relief passes.
    with pytest.raises(RuntimeError):
        comfy_surface._assert_touchpoint(lambda memory_required, device: [],
                                         paths["comfy.model_management.free_memory"])
    comfy_surface._assert_touchpoint(
        lambda memory_required, device, keep_loaded=(), for_dynamic=False, pins_required=0: [],
        paths["comfy.model_management.free_memory"])


@pytest.fixture(scope="module")
def comfy_sd():
    """Real ComfyUI's comfy.sd with --cpu, restored afterwards."""
    def is_comfy(name):
        return name == "folder_paths" or name == "comfy" or name.startswith(("comfy.", "comfy_"))

    preserved = {name: module for name, module in sys.modules.items() if is_comfy(name)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before, argv_before = list(sys.path), sys.argv
    comfy_dir = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    try:
        sys.argv = ["pytest-vae-decode-relief", "--cpu"]
        try:
            options = pytest.importorskip("comfy.options", reason="no ComfyUI on sys.path")
            options.enable_args_parsing()
            sd = pytest.importorskip("comfy.sd")
            pytest.importorskip("comfy.samplers")
        finally:
            sys.argv = argv_before
        yield sd
    finally:
        if "comfy.sd" in sys.modules and "comfy.samplers" in sys.modules:
            vae_decode_relief.uninstall()
        gone = [name for name, module in list(sys.modules.items())
                if is_comfy(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


@pytest.fixture
def real_comfy(comfy_sd, monkeypatch):
    """The autouse stand-ins give way to real ComfyUI's own modules, with DynamicVRAM read as on.

    ComfyUI's own flag is left as it is: with ``--cpu`` its allocator is not
    set up, and switching the flag would send ComfyUI's loader down that path.
    Only the relief's own reading of it is answered yes.
    """
    import comfy.memory_management
    import comfy.model_base
    import comfy.model_management as mm

    monkeypatch.setitem(sys.modules, "comfy.model_base", comfy.model_base)
    monkeypatch.setitem(sys.modules, "comfy.memory_management", comfy.memory_management)
    monkeypatch.setattr(vae_decode_relief, "_dynamic_vram_on", lambda: True)
    return mm


def _toy_vae(sd):
    """ComfyUI's plain AutoencoderKL layout, named by one key; every weight random."""
    vae = sd.VAE(sd={"decoder.conv_in.weight": torch.zeros(512, 4, 3, 3)})
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in vae.first_stage_model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return vae


def test_on_real_comfyui_a_decode_is_the_same_tensor_with_the_relief_acting(comfy_sd, real_comfy, monkeypatch,
                                                                           caplog):
    """The relief frees through ComfyUI's own call and then runs ComfyUI's own decode: same bits."""
    mm = real_comfy
    torch.set_num_threads(2)
    vae = _toy_vae(comfy_sd)
    # One item: its batch is 1 whatever memory reads, so the decodes below differ in the relief alone.
    latent = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(8))
    plain = vae.decode(latent).float().cpu().clone()

    # What the relief reads exists on real ComfyUI and prices this decode.
    assert "memory_used_decode" in vars(vae)  # an instance attribute, which no touchpoint can hold
    price, need = vae_decode_relief.decode_need(mm, vae, latent)
    assert price == int(vae.memory_used_decode(tuple(latent.shape), vae.vae_dtype)) > 0
    assert need == int(max(mm.minimum_inference_memory(), price + mm.extra_reserved_memory()))

    stock_decode, stock_encode = comfy_sd.VAE.decode, comfy_sd.VAE.encode
    freed = []
    real_free = mm.free_memory

    def recording(memory_required, device, *args, **kwargs):
        freed.append((memory_required, kwargs.get("keep_loaded"), kwargs.get("for_dynamic")))
        return real_free(memory_required, device, *args, **kwargs)

    vae_decode_relief.install()
    try:
        assert getattr(comfy_sd.VAE.decode, "_dgxm_decode_relief", False) is True
        assert comfy_sd.VAE.decode.__wrapped__ is stock_decode and comfy_sd.VAE.encode is stock_encode
        # A second model ComfyUI holds idle: another VAE that has decoded once.
        idle = _toy_vae(comfy_sd)
        idle.decode(latent)
        roomy = vae.decode(latent).float().cpu().clone()
        assert torch.equal(roomy, plain)
        held = vae_decode_relief.kept_and_idle(mm, vae)
        assert [entry.model for entry in held[0]] == [vae.patcher] and len(held[1]) == 1
        # Read as short of the decode's price: the relief has ComfyUI free for it, keeping the VAE
        # in hand, and ComfyUI's own decode gives the same tensor.
        monkeypatch.setattr(mm, "free_memory", recording)
        monkeypatch.setattr(mm, "get_free_memory",
                            lambda dev=None, torch_free_too=False: (0, 0) if torch_free_too else 0)
        with caplog.at_level("INFO"):
            relieved = vae.decode(latent).float().cpu().clone()
            by_name = vae.decode(samples_in=latent).float().cpu().clone()
    finally:
        monkeypatch.undo()
        vae_decode_relief.uninstall()
    assert comfy_sd.VAE.decode is stock_decode
    assert vae_decode_relief.state() == vae_decode_relief.UNSET  # it did not step aside on real ComfyUI
    ours = [call for call in freed if call[2] is False and call[1] is not None]
    assert ours and ours[0][0] == need and [entry.model for entry in ours[0][1]] == [vae.patcher]
    assert "decode relief:" in caplog.text and "stepped aside" not in caplog.text and idle is not None
    assert torch.equal(relieved, plain) and torch.equal(by_name, plain)


def test_real_comfyui_still_reads_as_the_relief_mirrors(comfy_sd, real_comfy):
    """The trigger and the guard rest on lines a signature cannot show; their text is the drift surface."""
    import inspect

    import comfy.memory_management
    import comfy.model_base
    import comfy.samplers

    mm = real_comfy
    assert list(inspect.signature(comfy_sd.VAE.decode).parameters)[:2] == ["self", "samples_in"]
    decode = inspect.getsource(comfy_sd.VAE.decode)
    assert "if self.latent_dim == 2 and samples_in.ndim == 5:" in decode
    assert "memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)" in decode
    assert "model_management.load_models_gpu([self.patcher], memory_required=memory_used" in decode
    loader = inspect.getsource(mm.load_models_gpu)
    assert "extra_mem = max(inference_memory, memory_required + extra_reserved_memory())" in loader
    assert "if free_mem < minimum_memory_required:" in loader
    free = inspect.getsource(mm.free_memory)
    assert "if shift_model not in keep_loaded and not shift_model.is_dead():" in free
    assert "is_dynamic() and for_dynamic" in free
    assert isinstance(vars(comfy.memory_management)["aimdo_enabled"], bool)
    assert inspect.isclass(comfy.model_base.BaseModel)
    assert "sample" in vars(comfy.samplers.CFGGuider)
    vae = _toy_vae(comfy_sd)
    assert vars(vae)["latent_dim"] == 2 and hasattr(vae, "vae_dtype") and hasattr(vae, "device")
    assert vae.patcher is not None and not isinstance(vae.patcher.model, comfy.model_base.BaseModel)


def test_on_real_comfyui_the_sampler_entry_is_guarded_and_a_decode_inside_a_guard_frees_nothing(
        comfy_sd, real_comfy, monkeypatch):
    """Install puts the guard on ComfyUI's real sampler entry, and a real decode under a guard is left alone.

    The decode here runs inside the guard the relief installs, on a stand-in
    for the sampler's body. A whole ComfyUI sampler with a preview callback is
    the hardware row's and a later CPU test's: it needs a model to sample.
    """
    import comfy.samplers

    mm = real_comfy
    vae = _toy_vae(comfy_sd)
    latent = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(9))
    other = _toy_vae(comfy_sd)
    other.decode(latent)
    plain = vae.decode(latent).float().cpu().clone()
    freed = []
    stock_sample = comfy.samplers.CFGGuider.sample
    vae_decode_relief.install()
    try:
        guarded = comfy.samplers.CFGGuider.sample
        assert getattr(guarded, "_dgxm_sampler_guard", False) is True and guarded.__wrapped__ is stock_sample
        monkeypatch.setattr(mm, "free_memory", lambda *args, **kwargs: freed.append(kwargs) or [])
        monkeypatch.setattr(mm, "get_free_memory",
                            lambda dev=None, torch_free_too=False: (0, 0) if torch_free_too else 0)
        preview = vae_decode_relief.with_sampler_guard(lambda self: vae.decode(latent).float().cpu().clone())
        inside = preview(object())
        assert freed == []
        # The same decode outside any sampler is relieved: the other VAE is idle.
        outside = vae.decode(latent).float().cpu().clone()
        assert len(freed) == 1 and freed[0]["for_dynamic"] is False
    finally:
        monkeypatch.undo()
        vae_decode_relief.uninstall()
    assert comfy.samplers.CFGGuider.sample is stock_sample
    assert torch.equal(inside, plain) and torch.equal(outside, plain) and other is not None
    assert vae_decode_relief.sampling_depth() == 0
