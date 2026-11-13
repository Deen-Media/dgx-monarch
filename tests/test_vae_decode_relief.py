"""vae_decode_relief.py: before a VAE decode priced above free memory, the driver frees what it needs.

On by default at node import on a GPU that shares the host's memory. These
tests hold the trigger (ComfyUI's own price and reserve against its reading),
what is kept loaded (the VAE in hand, every diffusion model), when everything
is kept (inside a sampler, the unload switch, DynamicVRAM off), that the
pack's shared busy check is not asked, that the caller's decode is stock's own call with the caller's own
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
from test_anima_cond_branch import CFG, SEED, STEPS, _pair, _same, _tiny_state

GIB = 2 ** 30
# A free-memory figure under any decode's requirement, and above zero: ComfyUI divides by it.
SHORT_READING = 64 * 2 ** 20


class _Flux:
    """A diffusion model."""


class _TextEncoder:
    """A text encoder: the one kind of model the relief frees from."""


class _SecondTextEncoder(_TextEncoder):
    pass


class _ClipVision:
    pass


class _ControlNet:
    pass


@pytest.fixture(autouse=True)
def _a_driver_with_dynamic_vram(monkeypatch):
    """The state every case starts from: the rule unset, no sampler running, DynamicVRAM on, no switch."""
    monkeypatch.setitem(vae_decode_relief._STATE, "state", vae_decode_relief.UNSET)
    monkeypatch.setitem(vae_decode_relief._SAMPLING, "depth", 0)
    monkeypatch.setitem(sys.modules, "comfy.memory_management", SimpleNamespace(aimdo_enabled=True))
    monkeypatch.delenv(driver_unload.DRIVER_UNLOAD_DISABLE_ENV, raising=False)
    monkeypatch.setattr(vae_decode_relief, "_SETTLE_S", 0.0)
    monkeypatch.setitem(vae_decode_relief._SERVING, "handle", None)  # no fleet noted unless a case notes one


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


def _loaded(inner, device="cuda:0"):
    """One entry of ``current_loaded_models``: a loaded model whose patcher holds ``inner``.

    ComfyUI marks a text encoder's patcher with ``is_clip``; every other patcher carries it False.
    """
    return SimpleNamespace(model=SimpleNamespace(model=inner, is_clip=isinstance(inner, _TextEncoder)),
                           device=device)


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


def test_a_reading_equal_to_the_need_is_not_short(monkeypatch):
    """ComfyUI's loader frees only when free memory is under the need; at the need itself it does not."""
    mm = _model_management([4 * GIB + GIB // 2], held=[_loaded(_TextEncoder())])
    wrapped, _calls, answer = _wrapped(monkeypatch, mm)
    assert wrapped(_Vae(price=4 * GIB), LATENT) is answer
    assert mm.freed == []


def test_a_reading_above_the_price_and_under_price_plus_reserve_is_short(monkeypatch):
    """The reserve is part of the need, as in ComfyUI's loader, so the relief acts inside that band."""
    mm = _model_management([4 * GIB + GIB // 4, 30 * GIB], held=[_loaded(_TextEncoder())])
    _available(monkeypatch, 8 * GIB, 30 * GIB)
    wrapped, _calls, answer = _wrapped(monkeypatch, mm)
    assert wrapped(_Vae(price=4 * GIB), LATENT) is answer
    assert len(mm.freed) == 1


def test_a_decode_priced_above_the_reading_frees_what_it_needs_and_then_decodes_with_the_callers_arguments(
        monkeypatch, caplog):
    ran = []
    vae = _Vae(price=4 * GIB)
    own, flux = SimpleNamespace(model=vae.patcher), _loaded(_Flux())
    clip_vision, control = _loaded(_ClipVision()), _loaded(_ControlNet())
    held = [_loaded(_TextEncoder()), own, flux, clip_vision, _loaded(_SecondTextEncoder()), control]
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
    # Everything that is not a text encoder stays loaded, whoever is using it: the VAE in hand, the
    # diffusion model, CLIP vision, a ControlNet.
    assert kept == [own, flux, clip_vision, control]
    (call,) = calls
    assert call[0] is vae and call[1][0] is LATENT and call[2]["vae_options"] is options
    assert caplog.text.count("decode relief:") == 1
    line = next(record.getMessage() for record in caplog.records if "decode relief:" in record.getMessage())
    assert "priced this VAE decode at 4.0 GiB (4.5 GiB with its reserve) against 2.0 GiB free" in line
    assert "from the 2 text encoders it held (_TextEncoder, _SecondTextEncoder)" in line
    assert "MemAvailable read 14.0 GiB before that and 46.0 GiB after it" in line
    assert line.endswith("the decode starts with 30.0 GiB free")
    # By name too.
    assert wrapped(vae, samples_in=LATENT) is answer and calls[-1][2] == {"samples_in": LATENT}


def test_the_decided_fallback_frees_every_idle_model_and_keeps_the_same_models(monkeypatch):
    """One line switches the amount asked for; what is kept loaded does not change with it."""
    assert vae_decode_relief.FREE_ALL_IDLE is False  # the decision of 2026-11-12: the requirement, no more
    vae = _Vae(price=4 * GIB)
    own, flux = SimpleNamespace(model=vae.patcher), _loaded(_Flux())
    monkeypatch.setattr(vae_decode_relief, "FREE_ALL_IDLE", True)
    mm = _model_management([2 * GIB, 30 * GIB], held=[own, _loaded(_TextEncoder()), flux])
    _available(monkeypatch, 14 * GIB, 46 * GIB)
    wrapped, _calls, answer = _wrapped(monkeypatch, mm)
    assert wrapped(vae, LATENT) is answer
    ((asked, device, kept, for_dynamic),) = mm.freed
    assert asked == 1e30 and device == "cuda:0" and for_dynamic is False and kept == [own, flux]


def test_a_decode_still_short_after_the_relief_runs_as_comfyui_runs_it_and_the_line_says_so(monkeypatch, caplog):
    mm = _model_management([GIB, 3 * GIB], held=[_loaded(_TextEncoder())])
    _available(monkeypatch, 8 * GIB, 10 * GIB)
    wrapped, _calls, answer = _wrapped(monkeypatch, mm)
    with caplog.at_level("INFO"):
        assert wrapped(_Vae(price=4 * GIB), LATENT) is answer
    assert "from the 1 text encoder it held (_TextEncoder)" in caplog.text
    assert caplog.records[-1].getMessage().endswith(
        "the decode is still priced above the 3.0 GiB free and runs as ComfyUI runs it")


@pytest.mark.parametrize("held", ["nothing", "the vae in hand", "a diffusion model", "what a sampler may use",
                                  "a text encoder on another device", "a marker that is not True"])
def test_with_no_text_encoder_to_free_from_nothing_is_freed_and_one_line_says_so(monkeypatch, caplog, held):
    """Only a text encoder is freed. Anything a sampler could be using is kept, whoever calls."""
    vae = _Vae(price=4 * GIB)
    uncertain = _loaded(_ClipVision())
    uncertain.model.is_clip = 1  # not ComfyUI's own True: not a model this rule is certain of
    entries = {"nothing": [], "the vae in hand": [SimpleNamespace(model=vae.patcher)],
               "a diffusion model": [_loaded(_Flux())],
               "what a sampler may use": [SimpleNamespace(model=vae.patcher), _loaded(_Flux()),
                                          _loaded(_ControlNet()), _loaded(_ClipVision()), _loaded(object())],
               "a text encoder on another device": [_loaded(_TextEncoder(), device="cpu")],
               "a marker that is not True": [uncertain]}[held]
    ran = []
    mm = _model_management([GIB], held=entries, ran=ran)
    wrapped, _calls, answer = _wrapped(monkeypatch, mm, ran)
    with caplog.at_level("INFO"):
        assert wrapped(vae, LATENT) is answer
        assert wrapped(vae, LATENT) is answer
    assert ran == ["decode", "decode"] and mm.freed == []
    assert caplog.text.count("decode relief:") == 1
    assert ("ComfyUI held no text encoder here to free from"
            f"{vae_decode_relief.NO_FLEET}, so the decode runs as ComfyUI runs it") in caplog.text


def test_an_entry_whose_model_is_gone_is_neither_kept_nor_named(monkeypatch):
    vae = _Vae()
    dead = SimpleNamespace(model=None)
    mm = _model_management([0], held=[dead, _loaded(_TextEncoder()), _loaded(_Flux())])
    keep, idle = vae_decode_relief.kept_and_idle(mm, vae)
    assert keep == [mm.current_loaded_models[2]] and idle == ["_TextEncoder"]
    # A text encoder's patcher with no inner model is named by its own class.
    bare = SimpleNamespace(model=SimpleNamespace(is_clip=True))
    mm.current_loaded_models = [bare]
    assert vae_decode_relief.kept_and_idle(mm, vae) == ([], ["SimpleNamespace"])
    # The VAE in hand is kept even if something marked its patcher as a text encoder.
    vae.patcher.is_clip = True
    own = SimpleNamespace(model=vae.patcher)
    mm.current_loaded_models = [own]
    assert vae_decode_relief.kept_and_idle(mm, vae) == ([own], [])


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
    assert "the driver kept what it holds because a sampler is running, and a decode made while it runs" \
           in caplog.text
    # Once the sampler has returned, a decode is relieved as usual.
    _available(monkeypatch, 8 * GIB, 30 * GIB)
    assert wrapped(vae, LATENT) is answer and len(mm.freed) == 1


def test_the_sampler_count_is_seen_from_every_thread_nests_and_comes_back_after_an_error():
    """A decode from another thread while a sampler runs must read the sampler as running too."""
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

    assert vae_decode_relief.with_sampler_guard(outer)(object()) == [1]  # another thread sees it running
    assert depths == [1, 2, 1] and vae_decode_relief.sampling_depth() == 0


def test_the_relief_does_not_ask_the_packs_shared_busy_check(monkeypatch, caplog):
    """That check read busy after a finished render on hardware (2026-11-12); this rule has its own tests.

    The shared check guards an unload of every model and a call to the
    Workers. The relief frees neither a diffusion model nor the VAE in hand,
    calls no Worker, and stands down inside a sampler, so a busy answer from
    the shared check must not stop it.
    """
    asked = []
    monkeypatch.setattr(driver_unload, "busy_reason",
                        lambda handle: asked.append(handle) or "a call to the Workers is still in flight")
    mm = _model_management([GIB, 30 * GIB], held=[_loaded(_TextEncoder())])
    _available(monkeypatch, 8 * GIB, 30 * GIB)
    wrapped, _calls, answer = _wrapped(monkeypatch, mm)
    with caplog.at_level("INFO"):
        assert wrapped(_Vae(price=4 * GIB), LATENT) is answer
    assert asked == [] and len(mm.freed) == 1 and "kept what it holds" not in caplog.text
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


@pytest.mark.parametrize("broken", ["reading", "floor", "held", "free_memory", "available"])
def test_any_failure_steps_aside_once_and_the_decode_in_hand_still_runs(monkeypatch, caplog, broken):
    ran = []
    mm = _model_management([RuntimeError("reading broke")] if broken == "reading" else [GIB, 2 * GIB],
                           held=[_loaded(_TextEncoder())], ran=ran)
    _available(monkeypatch, 8 * GIB, 9 * GIB)

    def broke(*_args, **_kwargs):
        raise RuntimeError(f"{broken} broke")

    if broken == "floor":
        mm.minimum_inference_memory = broke
    elif broken == "held":
        mm.current_loaded_models = 7  # not a list of loaded models at all
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
    assert "free what the decode needs from the text encoders it holds" in caplog.text
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
    Only the relief's own reading of it is answered yes. Each test's models
    are dropped after it, so the next one counts only what it loaded itself.
    """
    import gc

    import comfy.memory_management
    import comfy.model_management as mm

    monkeypatch.setitem(sys.modules, "comfy.memory_management", comfy.memory_management)
    monkeypatch.setattr(vae_decode_relief, "_dynamic_vram_on", lambda: True)
    yield mm
    mm.unload_all_models()
    gc.collect()


def _toy_vae(sd, channels=4):
    """ComfyUI's plain AutoencoderKL layout, named by one key; every weight random."""
    vae = sd.VAE(sd={"decoder.conv_in.weight": torch.zeros(512, channels, 3, 3)})
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
        # A second model ComfyUI holds idle, carrying the one mark the relief reads on a text
        # encoder's patcher. (A real text encoder is a tokenizer and a checkpoint; the mark is
        # what the rule keys on, and real ComfyUI's unload of it is what is under test.)
        idle = _toy_vae(comfy_sd)
        assert idle.patcher.is_clip is False and vae.patcher.is_clip is False  # ComfyUI's own default
        idle.patcher.is_clip = True
        idle.decode(latent)
        roomy = vae.decode(latent).float().cpu().clone()
        assert torch.equal(roomy, plain)
        held = vae_decode_relief.kept_and_idle(mm, vae)
        assert [entry.model for entry in held[0]] == [vae.patcher] and len(held[1]) == 1
        # Read as short of the decode's price. ComfyUI's own decode also reads this figure (it
        # slices its attention by it), so the reference is stock's decode under the same reading:
        # the two then differ in the relief alone.
        monkeypatch.setattr(mm, "get_free_memory", lambda dev=None, torch_free_too=False: (
            (SHORT_READING, 0) if torch_free_too else SHORT_READING))
        short = stock_decode(vae, latent).float().cpu().clone()
        # The relief has ComfyUI free for the decode, keeping the VAE in hand, and ComfyUI's own
        # decode gives the same tensor.
        monkeypatch.setattr(mm, "free_memory", recording)
        with caplog.at_level("INFO"):
            relieved = vae.decode(latent).float().cpu().clone()
            by_name = vae.decode(samples_in=latent).float().cpu().clone()
        state_while_installed = vae_decode_relief.state()
    finally:
        monkeypatch.undo()
        vae_decode_relief.uninstall()
    assert comfy_sd.VAE.decode is stock_decode
    assert state_while_installed != vae_decode_relief.STEPPED_ASIDE  # read before uninstall resets it
    ours = [call for call in freed if call[2] is False and call[1] is not None]
    assert ours and ours[0][0] == need and [entry.model for entry in ours[0][1]] == [vae.patcher]
    assert "decode relief:" in caplog.text and "stepped aside" not in caplog.text and idle is not None
    assert torch.equal(relieved, short) and torch.equal(by_name, short)


def test_real_comfyui_still_reads_as_the_relief_mirrors(comfy_sd, real_comfy):
    """The trigger and the guard rest on lines a signature cannot show; their text is the drift surface."""
    import inspect

    import comfy.memory_management
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
    assert "sample" in vars(comfy.samplers.CFGGuider)
    # The one mark the relief frees by: ComfyUI sets it on a text encoder's patcher and nowhere else.
    assert "self.patcher.is_clip = True" in inspect.getsource(comfy_sd.CLIP.__init__)
    vae = _toy_vae(comfy_sd)
    assert vars(vae)["latent_dim"] == 2 and hasattr(vae, "vae_dtype") and hasattr(vae, "device")
    assert vae.patcher is not None and vae.patcher.is_clip is False


def test_on_real_comfyui_the_sampler_entry_is_guarded_and_a_decode_inside_a_guard_frees_nothing(
        comfy_sd, real_comfy, monkeypatch):
    """Install puts the guard on ComfyUI's real sampler entry, and a real decode under a guard is left alone.

    The decode here runs inside the guard the relief installs, on a stand-in
    for the sampler's body. The test below runs a whole ComfyUI sampler with a
    preview callback.
    """
    import comfy.samplers

    mm = real_comfy
    vae = _toy_vae(comfy_sd)
    latent = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(9))
    other = _toy_vae(comfy_sd)
    other.patcher.is_clip = True  # the mark of a text encoder, as in the test above
    other.decode(latent)
    vae.decode(latent)
    freed = []
    stock_sample, stock_decode = comfy.samplers.CFGGuider.sample, comfy_sd.VAE.decode
    vae_decode_relief.install()
    try:
        guarded = comfy.samplers.CFGGuider.sample
        assert getattr(guarded, "_dgxm_sampler_guard", False) is True and guarded.__wrapped__ is stock_sample
        monkeypatch.setattr(mm, "get_free_memory", lambda dev=None, torch_free_too=False: (
            (SHORT_READING, 0) if torch_free_too else SHORT_READING))
        plain = stock_decode(vae, latent).float().cpu().clone()  # stock, under the same short reading
        monkeypatch.setattr(mm, "free_memory", lambda *args, **kwargs: freed.append(kwargs) or [])
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


def test_on_real_comfyui_a_whole_sampler_with_a_preview_decode_frees_nothing_while_it_runs(
        comfy_sd, real_comfy, monkeypatch, caplog):
    """A real ComfyUI sampler run whose step callback decodes a preview, as a previewer does.

    Every preview is priced above the reading while a text encoder sits idle:
    the state in which the relief acts. Inside the sampler it must not. The
    count is up at every step, nothing is freed, and the sampled latent and
    every preview equal stock's. The same decode after the sampler returns is
    relieved, and the sampled model is among the models kept.
    """
    anima =pytest.importorskip("comfy.ldm.anima.model", reason="this ComfyUI predates Anima")
    import comfy.ops
    import comfy.sample
    import comfy.samplers

    mm = real_comfy
    torch.set_num_threads(2)
    vae = _toy_vae(comfy_sd, channels=16)  # the tiny model's latent is 16 channels with a frame axis
    idle = _toy_vae(comfy_sd)
    idle.patcher.is_clip = True  # the mark of a text encoder, as in the tests above
    idle.decode(torch.zeros(1, 4, 8, 8))
    patcher = comfy_sd.load_diffusion_model_state_dict(_tiny_state(comfy.ops, anima.Anima))
    positive, negative = _pair()
    freed, seen, previews = [], [], []
    real_free = mm.free_memory

    def recording(memory_required, device, *args, **kwargs):
        freed.append(kwargs)
        return real_free(memory_required, device, *args, **kwargs)

    def preview(step, x0, x, total_steps):
        seen.append((vae_decode_relief.sampling_depth(), len(vae_decode_relief.kept_and_idle(mm, vae)[1]),
                     len(freed)))
        previews.append((x0.clone(), vae.decode(x0).float().cpu().clone()))

    def sample():
        del seen[:], previews[:]
        with torch.inference_mode():
            latent = comfy.sample.fix_empty_latent_channels(patcher, torch.zeros(1, 4, 8, 8))
            noise = comfy.sample.prepare_noise(latent, SEED, None)
            out = comfy.sample.sample(patcher, noise, STEPS, CFG, "euler", "simple", positive, negative,
                                      latent, denoise=1.0, disable_pbar=True, seed=SEED, callback=preview)
        return out.clone(), list(seen), list(previews)

    stock_sample, stock_decode = comfy.samplers.CFGGuider.sample, comfy_sd.VAE.decode
    try:
        # The sampler reads this figure too (its cond batches), so both runs read the same one.
        monkeypatch.setattr(mm, "get_free_memory", lambda dev=None, torch_free_too=False: (
            (SHORT_READING, 0) if torch_free_too else SHORT_READING))
        monkeypatch.setattr(mm, "free_memory", recording)
        stock_latent, stock_seen, stock_previews = sample()
        assert freed == []  # stock frees nothing on this device, so every call below is the relief's
        vae_decode_relief.install()
        with caplog.at_level("INFO"):
            latent, inside, ours = sample()
            freed_by_the_sampler_run = list(freed)
            last = ours[-1][0]
            with torch.inference_mode():
                after = vae.decode(last).float().cpu().clone()
        kept = [entry.model for entry in freed[0]["keep_loaded"]] if freed else []
    finally:
        monkeypatch.undo()
        vae_decode_relief.uninstall()
    assert comfy.samplers.CFGGuider.sample is stock_sample and comfy_sd.VAE.decode is stock_decode
    # Stock, with no guard: the count stays down. With the relief installed the real sampler
    # entry is the guarded one, so every preview sees the count up and an idle text encoder held.
    assert len(stock_seen) == len(inside) >= STEPS
    assert all(depth == 0 for depth, _, _ in stock_seen)
    assert all(row == (1, 1, 0) for row in inside), inside
    assert freed_by_the_sampler_run == []
    assert vae_decode_relief.sampling_depth() == 0
    # The same decode once the sampler has returned is relieved, and keeps the sampled model.
    assert len(freed) == 1 and freed[0]["for_dynamic"] is False
    assert vae.patcher in kept and any(getattr(model, "model", None) is patcher.model for model in kept)
    assert all(torch.isfinite(image).all() for _, image in ours)
    assert _same(latent, stock_latent)
    assert all(_same(mine[0], theirs[0]) and _same(mine[1], theirs[1])
               for mine, theirs in zip(ours, stock_previews, strict=True))
    assert _same(after, stock_previews[-1][1])
    said = [record for record in caplog.records if vae_decode_relief.SAMPLER_RUNNING in record.getMessage()]
    assert len(said) == 1 and "stepped aside" not in caplog.text and idle is not None


def test_on_real_comfyui_a_preview_decoded_from_another_thread_frees_nothing(comfy_sd, real_comfy, monkeypatch):
    """The same real sampler, with each preview decoded on a thread the step callback starts and joins.

    The count is process-wide, so that thread sees a sampler running. A count
    kept per thread would read 0 there and free the text encoder under the
    sampler at the first step.
    """
    anima = pytest.importorskip("comfy.ldm.anima.model", reason="this ComfyUI predates Anima")
    import comfy.ops
    import comfy.sample
    import comfy.samplers

    mm = real_comfy
    torch.set_num_threads(2)
    vae = _toy_vae(comfy_sd, channels=16)
    idle = _toy_vae(comfy_sd)
    idle.patcher.is_clip = True  # the mark of a text encoder, as in the tests above
    idle.decode(torch.zeros(1, 4, 8, 8))
    patcher = comfy_sd.load_diffusion_model_state_dict(_tiny_state(comfy.ops, anima.Anima))
    positive, negative = _pair()
    freed, seen = [], []
    real_free = mm.free_memory

    def recording(memory_required, device, *args, **kwargs):
        freed.append(kwargs)
        return real_free(memory_required, device, *args, **kwargs)

    def preview(step, x0, x, total_steps):
        box = []

        def other():
            with torch.inference_mode():  # per thread: a previewer's own thread enters it itself
                box.append((vae_decode_relief.sampling_depth(), len(freed),
                            vae.decode(x0).float().cpu().clone()))

        thread = threading.Thread(target=other, name="preview-on-another-thread")
        thread.start()
        thread.join()
        seen.append(box[0])

    def sample():
        del seen[:]
        with torch.inference_mode():
            latent = comfy.sample.fix_empty_latent_channels(patcher, torch.zeros(1, 4, 8, 8))
            noise = comfy.sample.prepare_noise(latent, SEED, None)
            out = comfy.sample.sample(patcher, noise, STEPS, CFG, "euler", "simple", positive, negative,
                                      latent, denoise=1.0, disable_pbar=True, seed=SEED, callback=preview)
        return out.clone(), list(seen)

    stock_sample = comfy.samplers.CFGGuider.sample
    try:
        monkeypatch.setattr(mm, "get_free_memory", lambda dev=None, torch_free_too=False: (
            (SHORT_READING, 0) if torch_free_too else SHORT_READING))
        monkeypatch.setattr(mm, "free_memory", recording)
        stock_latent, stock_seen = sample()
        assert freed == []
        vae_decode_relief.install()
        latent, inside = sample()
        during = list(freed)
    finally:
        monkeypatch.undo()
        vae_decode_relief.uninstall()
    assert comfy.samplers.CFGGuider.sample is stock_sample
    assert len(inside) == len(stock_seen) >= STEPS
    assert all(depth == 1 and count == 0 for depth, count, _ in inside), [row[:2] for row in inside]
    assert during == [] and vae_decode_relief.sampling_depth() == 0 and idle is not None
    assert _same(latent, stock_latent)
    assert all(_same(mine[2], theirs[2]) for mine, theirs in zip(inside, stock_seen, strict=True))
