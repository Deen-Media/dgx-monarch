"""vae_decode_relief.py: before a VAE decode priced above free memory, the driver gives back what is idle.

On by default at node import on a GPU that shares the host's memory. These
tests hold the trigger (ComfyUI's own price and reserve against its reading),
the order of the two steps, when it must keep everything, that the caller's
decode is stock's own call with the caller's own arguments, how it fails
open, and the gate. On real ComfyUI's ``VAE`` (CPU, random weights; skipped
where ComfyUI is not importable) a decode gives the same tensor with the
relief acting as without it.
"""
from __future__ import annotations

import gc
import os
import sys
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import driver_uma_memory, driver_unload, vae_decode_relief
from module_location_helpers import from_checkout

GIB = 2 ** 30


@pytest.fixture(autouse=True)
def _the_state_starts_unset(monkeypatch):
    monkeypatch.setitem(vae_decode_relief._STATE, "state", vae_decode_relief.UNSET)
    monkeypatch.setitem(vae_decode_relief._MESH, "ref", None)


class _Vae:
    """What the relief reads of a VAE: its price, its dtype and its device."""

    def __init__(self, price=4 * GIB, latent_dim=3):
        self.latent_dim = latent_dim
        self.vae_dtype = "bf16"
        self.device = "cuda:0"
        self.priced = []
        self._price = price

    def memory_used_decode(self, shape, dtype):
        self.priced.append((shape, dtype))
        if isinstance(self._price, Exception):
            raise self._price
        return self._price


def _model_management(readings, *, inference=GIB, reserve=GIB // 2):
    """ComfyUI's model management as the relief reads it; ``readings`` are the free figures in order."""
    left = list(readings)
    asked = []

    def get_free_memory(device=None, torch_free_too=False):
        asked.append(device)
        figure = left.pop(0) if len(left) > 1 else left[0]
        if isinstance(figure, Exception):
            raise figure
        return figure

    return SimpleNamespace(get_free_memory=get_free_memory, asked=asked,
                           minimum_inference_memory=lambda: inference,
                           extra_reserved_memory=lambda: reserve)


class _Mesh:
    """A stand-in mesh handle that can be referenced weakly."""


def _driver(monkeypatch, *, busy="", held=("TextEncoder", "OtherVae"), trimmable=True, raising=None):
    """Stand-ins for ``driver_unload``'s five calls; returns the order they ran in."""
    ran = []

    def call(name, answer):
        def stand_in(*args):
            ran.append(name)
            if raising == name:
                raise RuntimeError(f"{name} broke")
            return answer(*args) if callable(answer) else answer
        return stand_in

    monkeypatch.setattr(driver_unload, "busy_reason", call("busy_reason", busy))
    monkeypatch.setattr(driver_unload, "held_models",
                        call("held_models", [(object(), name) for name in held]))
    monkeypatch.setattr(driver_unload, "unload", call("unload", lambda _mm, pairs: driver_unload.Unloaded(
        tuple(name for _entry, name in pairs), 14 * GIB, 46 * GIB)))
    monkeypatch.setattr(driver_unload, "trimmable", call("trimmable", trimmable))
    monkeypatch.setattr(driver_unload, "trim_actors",
                        call("trim_actors", driver_unload.Trimmed(1, 46 * GIB, 55 * GIB)))
    return ran


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
    ran = _driver(monkeypatch)
    mm = _model_management([5 * GIB])
    wrapped, calls, answer = _wrapped(monkeypatch, mm, ran)
    vae, latent = _Vae(price=4 * GIB), torch.zeros(1, 4, 2, 8, 8)
    with caplog.at_level("INFO"):
        assert wrapped(vae, latent) is answer
    assert ran == ["decode"] and calls == [(vae, (latent,), {})]
    assert mm.asked == ["cuda:0"] and "decode relief" not in caplog.text
    assert vae_decode_relief.state() == vae_decode_relief.UNSET


def test_a_decode_priced_above_the_reading_unloads_and_then_decodes_with_the_callers_arguments(
        monkeypatch, caplog):
    ran = _driver(monkeypatch)
    mm = _model_management([2 * GIB, 30 * GIB])
    wrapped, calls, answer = _wrapped(monkeypatch, mm, ran)
    vae, latent, options = _Vae(price=4 * GIB), torch.zeros(1, 4, 2, 8, 8), {"tile": 1}
    with caplog.at_level("INFO"):
        assert wrapped(vae, latent, vae_options=options) is answer
    # The unload comes before the decode; the Workers are not asked, the unload was enough.
    assert ran == ["busy_reason", "held_models", "unload", "decode"]
    (call,) = calls
    assert call[0] is vae and call[1][0] is latent and call[2]["vae_options"] is options
    assert caplog.text.count("decode relief:") == 1
    line = next(record.getMessage() for record in caplog.records if "decode relief:" in record.getMessage())
    assert "priced this VAE decode at 4.0 GiB (4.5 GiB with its reserve) against 2.0 GiB free" in line
    assert "unloaded the 2 models (TextEncoder, OtherVae)" in line
    assert "14.0 GiB before the unload and 46.0 GiB after it" in line
    assert line.endswith("the decode starts with 30.0 GiB free")
    # By name too.
    assert wrapped(vae, samples_in=latent) is answer and calls[-1][2] == {"samples_in": latent}


def test_the_workers_are_asked_only_when_the_unload_was_not_enough(monkeypatch, caplog):
    mesh = _Mesh()
    vae_decode_relief.note_mesh(mesh)
    ran = _driver(monkeypatch)
    wrapped, _calls, answer = _wrapped(monkeypatch, _model_management([GIB, 3 * GIB, 12 * GIB]), ran)
    with caplog.at_level("INFO"):
        assert wrapped(_Vae(price=4 * GIB), torch.zeros(1, 4, 2, 8, 8)) is answer
    assert ran == ["busy_reason", "held_models", "unload", "trimmable", "trim_actors", "decode"]
    line = caplog.records[-1].getMessage()
    assert "; then asked 1 actor to release cached memory: MemAvailable read 46.0 GiB before that and " \
           "55.0 GiB after it" in line
    assert line.endswith("the decode starts with 12.0 GiB free")
    # With no idle mesh the driver's own half still runs, and the line says the decode is still short.
    caplog.clear()
    ran = _driver(monkeypatch, trimmable=False)
    wrapped, _calls, answer = _wrapped(monkeypatch, _model_management([GIB, 3 * GIB]), ran)
    with caplog.at_level("INFO"):
        assert wrapped(_Vae(price=4 * GIB), torch.zeros(1, 4, 2, 8, 8)) is answer
    assert ran == ["busy_reason", "held_models", "unload", "trimmable", "decode"]
    assert caplog.records[-1].getMessage().endswith(
        "the decode is still priced above the 3.0 GiB free and runs as ComfyUI runs it")


def test_the_mesh_the_relief_asks_is_the_one_the_last_render_used(monkeypatch):
    mesh = _Mesh()
    vae_decode_relief.note_mesh(mesh)
    asked = []
    _driver(monkeypatch)
    monkeypatch.setattr(driver_unload, "busy_reason", lambda handle: asked.append(("busy", handle)) or "")
    monkeypatch.setattr(driver_unload, "trimmable", lambda handle: asked.append(("trimmable", handle)) or True)
    monkeypatch.setattr(driver_unload, "trim_actors", lambda handle: asked.append(("trim", handle))
                        or driver_unload.Trimmed(2, GIB, 2 * GIB))
    wrapped, _calls, _answer = _wrapped(monkeypatch, _model_management([GIB, 2 * GIB, 9 * GIB]))
    wrapped(_Vae(price=4 * GIB), torch.zeros(1, 4, 2, 8, 8))
    assert asked == [("busy", mesh), ("trimmable", mesh), ("trim", mesh)]


def test_the_mesh_is_held_weakly_and_a_handle_that_cannot_be_is_not_kept():
    mesh = _Mesh()
    vae_decode_relief.note_mesh(mesh)
    assert vae_decode_relief._mesh() is mesh
    del mesh
    gc.collect()
    assert vae_decode_relief._mesh() is None  # the relief never keeps a retired mesh alive
    vae_decode_relief.note_mesh(_Mesh())
    vae_decode_relief.note_mesh(None)
    assert vae_decode_relief._mesh() is None
    vae_decode_relief.note_mesh(7)  # an int takes no weak reference: not kept, and no error
    assert vae_decode_relief._mesh() is None


@pytest.mark.parametrize("why", [
    "a render is still running in this driver",
    "a render result is still held on the mesh",
    "an abandoned render may still be running on the Workers",
    "a call to the Workers is still in flight",
])
def test_a_render_in_flight_keeps_everything_and_says_so_once(monkeypatch, caplog, why):
    """A pipelined render decodes one result while later ones sample: their models and caches are in use."""
    ran = _driver(monkeypatch, busy=why)
    wrapped, _calls, answer = _wrapped(monkeypatch, _model_management([GIB]), ran)
    with caplog.at_level("INFO"):
        assert wrapped(_Vae(price=4 * GIB), torch.zeros(1, 4, 2, 8, 8)) is answer
        assert wrapped(_Vae(price=4 * GIB), torch.zeros(1, 4, 2, 8, 8)) is answer
    assert ran == ["busy_reason", "decode", "busy_reason", "decode"]
    assert caplog.text.count("decode relief:") == 1
    assert f"the driver kept what it holds because {why}" in caplog.text
    assert vae_decode_relief.state() == vae_decode_relief.UNSET


def test_with_nothing_idle_the_decode_runs_as_comfyui_runs_it_and_one_line_says_so(monkeypatch, caplog):
    ran = _driver(monkeypatch, held=(), trimmable=False)
    wrapped, _calls, answer = _wrapped(monkeypatch, _model_management([GIB]), ran)
    with caplog.at_level("INFO"):
        assert wrapped(_Vae(price=4 * GIB), torch.zeros(1, 4, 2, 8, 8)) is answer
        assert wrapped(_Vae(price=4 * GIB), torch.zeros(1, 4, 2, 8, 8)) is answer
    assert "unload" not in ran and "trim_actors" not in ran and ran.count("decode") == 2
    assert caplog.text.count("decode relief:") == 1
    assert "ComfyUI held no model here to unload and no idle mesh to ask" in caplog.text


def test_a_trim_that_fails_is_a_skip_and_not_a_fault(monkeypatch, caplog):
    """A mesh another session owns, or a call that did not answer, must not stand the rule down."""
    vae_decode_relief.note_mesh(mesh := _Mesh())
    ran = _driver(monkeypatch, raising="trim_actors")
    wrapped, _calls, answer = _wrapped(monkeypatch, _model_management([GIB, 3 * GIB]), ran)
    with caplog.at_level("INFO"):
        assert wrapped(_Vae(price=4 * GIB), torch.zeros(1, 4, 2, 8, 8)) is answer
    assert ran[-2:] == ["trim_actors", "decode"] and mesh is not None
    assert vae_decode_relief.state() == vae_decode_relief.UNSET and "stepped aside" not in caplog.text
    assert "could not ask the Workers to release cached memory (RuntimeError: trim_actors broke)" in caplog.text


@pytest.mark.parametrize("broken", ["reading", "busy_reason", "held_models", "unload", "trimmable", "floor"])
def test_any_other_failure_steps_aside_once_and_the_decode_in_hand_still_runs(monkeypatch, caplog, broken):
    ran = _driver(monkeypatch, raising=broken)
    mm = _model_management([RuntimeError("reading broke")] if broken == "reading" else [GIB, 2 * GIB])
    if broken == "floor":
        def gone():
            raise AttributeError("minimum_inference_memory")
        mm.minimum_inference_memory = gone
    wrapped, _calls, answer = _wrapped(monkeypatch, mm, ran)
    vae = _Vae(price=4 * GIB)
    with caplog.at_level("INFO"):
        assert wrapped(vae, torch.zeros(1, 4, 2, 8, 8)) is answer
    assert ran[-1] == "decode"
    assert caplog.text.count("decode relief stepped aside") == 1
    assert vae_decode_relief.state() == vae_decode_relief.STEPPED_ASIDE
    # From then on every call is stock's and the relief does no work at all.
    ran.clear()
    vae.priced.clear()
    asked = len(mm.asked)
    assert wrapped(vae, torch.zeros(1, 4, 2, 8, 8)) is answer
    assert ran == ["decode"] and vae.priced == [] and len(mm.asked) == asked
    assert caplog.text.count("decode relief stepped aside") == 1


def test_a_vae_that_cannot_price_its_decode_is_that_decodes_matter(monkeypatch, caplog):
    """ComfyUI's own decode meets the same call; one odd VAE must not stand the rule down for the process."""
    ran = _driver(monkeypatch)
    wrapped, _calls, answer = _wrapped(monkeypatch, _model_management([GIB, 30 * GIB]), ran)
    with caplog.at_level("INFO"):
        assert wrapped(_Vae(price=NotImplementedError("no generic decode")), torch.zeros(1, 4, 8, 8)) is answer
        assert wrapped(_Vae(price=NotImplementedError("no generic decode")), torch.zeros(1, 4, 8, 8)) is answer
    assert ran == ["decode", "decode"] and vae_decode_relief.state() == vae_decode_relief.UNSET
    assert caplog.text.count("could not be read") == 1 and "stepped aside" not in caplog.text
    # The next VAE that can price is relieved as before.
    assert wrapped(_Vae(price=4 * GIB), torch.zeros(1, 4, 2, 8, 8)) is answer
    assert ran[-4:] == ["busy_reason", "held_models", "unload", "decode"]


def test_what_is_not_an_ordinary_error_passes_through(monkeypatch):
    """ComfyUI's interrupt is a BaseException: a cancelled prompt must stay cancelled."""

    class Interrupt(BaseException):
        pass

    ran = _driver(monkeypatch)

    def interrupted(_mm, _pairs):
        raise Interrupt

    monkeypatch.setattr(driver_unload, "unload", interrupted)
    wrapped, calls, _answer = _wrapped(monkeypatch, _model_management([GIB]), ran)
    with pytest.raises(Interrupt):
        wrapped(_Vae(price=4 * GIB), torch.zeros(1, 4, 2, 8, 8))
    assert calls == [] and vae_decode_relief.state() == vae_decode_relief.UNSET


def test_a_call_it_cannot_read_goes_straight_to_comfyui(monkeypatch):
    ran = _driver(monkeypatch)
    mm = _model_management([GIB])
    wrapped, calls, answer = _wrapped(monkeypatch, mm, ran)
    vae = _Vae()
    assert wrapped(vae, latent=torch.zeros(1, 4, 8, 8)) is answer  # a name this pack version does not know
    assert ran == ["decode"] and mm.asked == [] and calls[0][2].keys() == {"latent"}
    # And with ComfyUI's model management not imported in this process there is nothing to read.
    monkeypatch.delitem(sys.modules, "comfy.model_management")
    assert wrapped(vae, torch.zeros(1, 4, 8, 8)) is answer and mm.asked == []


def test_the_wrapper_keeps_the_functions_name_and_installs_once(monkeypatch):
    class VAE:
        def decode(self, samples_in, vae_options=None):
            return samples_in

        def decode_tiled(self, samples):
            return samples

        def encode(self, pixel_samples):
            return pixel_samples

    stock_decode, stock_tiled, stock_encode = VAE.decode, VAE.decode_tiled, VAE.encode
    monkeypatch.setitem(sys.modules, "comfy.sd", SimpleNamespace(VAE=VAE))
    vae_decode_relief.install()
    first = VAE.decode
    assert first is not stock_decode and first.__name__ == "decode" and first.__wrapped__ is stock_decode
    vae_decode_relief.install()  # a second install does not stack
    assert VAE.decode is first
    assert VAE.decode_tiled is stock_tiled and VAE.encode is stock_encode  # neither is touched
    vae_decode_relief.note_mesh(_Mesh())
    vae_decode_relief.uninstall()
    assert VAE.decode is stock_decode and vae_decode_relief.state() == vae_decode_relief.UNSET
    assert vae_decode_relief._mesh() is None
    vae_decode_relief.uninstall()  # and undoing twice is harmless
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
    assert "gives back the models it holds idle" in caplog.text and vae_decode_relief.DISABLE_ENV in caplog.text
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


def test_a_synchronous_render_names_its_mesh_once_its_result_is_in():
    """The relief asks the Workers of the mesh the last render used, so the render must say which.

    Held by source: ``run_render`` notes the mesh after ``pending.result()``
    has returned, which is after the result's own cleanup has retired its
    lease and finished the driver's progress tracker.
    """
    import pathlib

    source = (pathlib.Path(__file__).resolve().parents[1] / "src" / "dgx_monarch" / "nodes"
              / "common.py").read_text()
    result = source.index("result = pending.result()")
    note = source.index("vae_decode_relief.note_mesh(")
    assert result < note < source.index("return result", result)


def test_the_surface_canary_holds_the_calls_the_relief_makes():
    """An upstream change of the decode call or of the loader's two figures fails the daily canary."""
    from dgx_monarch import comfy_surface

    paths = {touchpoint.path: touchpoint for touchpoint in comfy_surface.TOUCHPOINTS}
    assert paths["comfy.sd.VAE.decode"].positional == ("self", "samples_in")
    for name in ("minimum_inference_memory", "extra_reserved_memory", "unload_all_models"):
        assert f"comfy.model_management.{name}" in paths

    def renamed(self, latent):
        return latent

    with pytest.raises(RuntimeError):
        comfy_surface._assert_touchpoint(renamed, paths["comfy.sd.VAE.decode"])
    comfy_surface._assert_touchpoint(lambda self, samples_in, vae_options=None: samples_in,
                                     paths["comfy.sd.VAE.decode"])


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
        finally:
            sys.argv = argv_before
        yield sd
    finally:
        if "comfy.sd" in sys.modules:
            vae_decode_relief.uninstall()
        gone = [name for name, module in list(sys.modules.items())
                if is_comfy(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def _toy_vae(sd):
    """ComfyUI's plain AutoencoderKL layout, named by one key; every weight random."""
    vae = sd.VAE(sd={"decoder.conv_in.weight": torch.zeros(512, 4, 3, 3)})
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in vae.first_stage_model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return vae


def test_on_real_comfyui_a_decode_is_the_same_tensor_with_the_relief_acting(comfy_sd, monkeypatch, caplog):
    """The relief unloads through ComfyUI's own calls and then runs ComfyUI's own decode: same bits."""
    import comfy.model_management as mm

    torch.set_num_threads(2)
    vae = _toy_vae(comfy_sd)
    # One item: its batch is 1 whatever memory reads, so the two decodes below differ in the
    # relief alone. (Several images would also change ComfyUI's own batch count with the reading.)
    latent = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(8))
    plain = vae.decode(latent).float().cpu().clone()

    # What the relief reads exists on real ComfyUI and prices this decode.
    price, need = vae_decode_relief.decode_need(mm, vae, latent)
    assert price == int(vae.memory_used_decode(tuple(latent.shape), vae.vae_dtype)) > 0
    assert need == int(max(mm.minimum_inference_memory(), price + mm.extra_reserved_memory()))

    stock_decode, stock_encode = comfy_sd.VAE.decode, comfy_sd.VAE.encode
    vae_decode_relief.install()
    try:
        assert getattr(comfy_sd.VAE.decode, "_dgxm_decode_relief", False) is True
        assert comfy_sd.VAE.decode.__wrapped__ is stock_decode and comfy_sd.VAE.encode is stock_encode
        # With room, nothing is unloaded.
        unloads = []
        real_unload = mm.unload_all_models
        monkeypatch.setattr(mm, "unload_all_models", lambda: unloads.append("unload") or real_unload())
        roomy = vae.decode(latent).float().cpu().clone()
        assert unloads == [] and torch.equal(roomy, plain)
        # Read as short of the decode's price: the relief unloads what ComfyUI holds, the VAE in
        # hand among it, and ComfyUI's own decode loads it again and gives the same tensor.
        reading = mm.get_free_memory
        monkeypatch.setattr(mm, "get_free_memory",
                            lambda dev=None, torch_free_too=False: (0, 0) if torch_free_too else 0)
        held_before = len(mm.current_loaded_models)
        with caplog.at_level("INFO"):
            relieved = vae.decode(latent).float().cpu().clone()
            by_name = vae.decode(samples_in=latent).float().cpu().clone()
        monkeypatch.setattr(mm, "get_free_memory", reading)
    finally:
        vae_decode_relief.uninstall()
    assert comfy_sd.VAE.decode is stock_decode
    assert vae_decode_relief.state() == vae_decode_relief.UNSET  # it did not step aside on real ComfyUI
    assert held_before >= 1 and unloads and "decode relief:" in caplog.text
    assert "stepped aside" not in caplog.text
    assert torch.equal(relieved, plain) and torch.equal(by_name, plain)


def test_real_comfyuis_decode_still_prices_and_loads_as_the_relief_mirrors(comfy_sd):
    """The trigger is ComfyUI's own two lines; their text is the drift surface a signature cannot show."""
    import inspect

    import comfy.model_management as mm

    assert list(inspect.signature(comfy_sd.VAE.decode).parameters)[:2] == ["self", "samples_in"]
    decode = inspect.getsource(comfy_sd.VAE.decode)
    assert "if self.latent_dim == 2 and samples_in.ndim == 5:" in decode
    assert "memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)" in decode
    assert "model_management.load_models_gpu([self.patcher], memory_required=memory_used" in decode
    loader = inspect.getsource(mm.load_models_gpu)
    assert "extra_mem = max(inference_memory, memory_required + extra_reserved_memory())" in loader
    assert "if free_mem < minimum_memory_required:" in loader
    vae = _toy_vae(comfy_sd)
    assert vars(vae)["latent_dim"] == 2 and hasattr(vae, "vae_dtype") and hasattr(vae, "device")
