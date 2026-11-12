"""vae_encode_item.py: the driver encodes a batch of images one image a call.

On by default at node import on a GPU that shares the host's memory. These
tests hold the gate, what the wrapper does and when it steps aside, and, on
real ComfyUI's ``VAE`` (CPU, random weights; skipped where ComfyUI is not
importable), that stock's latent moves with the free-memory reading while the
rule's does not and is stock's own encode of each image alone.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import driver_uma_memory, vae_encode_item
from module_location_helpers import from_checkout

GIB = 2 ** 30


@pytest.fixture(autouse=True)
def _the_state_starts_unset(monkeypatch):
    monkeypatch.setitem(vae_encode_item._STATE, "state", vae_encode_item.UNSET)


def _image_vae(**changes):
    return SimpleNamespace(**{"latent_dim": 2, "not_video": False, "extra_1d_channel": None, **changes})


def test_only_a_batch_of_images_for_an_image_vae_is_split():
    images = torch.zeros(6, 8, 8, 3)
    assert vae_encode_item.takes_items(_image_vae(), images)
    assert not vae_encode_item.takes_items(_image_vae(), torch.zeros(1, 8, 8, 3))   # one image
    assert not vae_encode_item.takes_items(_image_vae(), torch.zeros(6, 8))         # not an image batch
    assert not vae_encode_item.takes_items(_image_vae(), torch.zeros(2, 6, 8, 8, 3))  # five dimensions
    assert not vae_encode_item.takes_items(_image_vae(), object())                  # no shape at all
    assert not vae_encode_item.takes_items(_image_vae(latent_dim=3), images)        # a video VAE: frames
    # An image VAE built on 3-D convolutions: N images are N items of one frame each.
    assert vae_encode_item.takes_items(_image_vae(latent_dim=3, not_video=True), images)
    assert not vae_encode_item.takes_items(_image_vae(latent_dim=1), images)        # 1-D audio
    assert not vae_encode_item.takes_items(_image_vae(extra_1d_channel=16), images)  # audio with a 2-D latent
    # ComfyUI sets audio_sample_rate on every VAE, so it must not switch the rule off.
    assert vae_encode_item.takes_items(_image_vae(audio_sample_rate=44100), images)


@pytest.mark.parametrize("missing", ["latent_dim", "extra_1d_channel"])
def test_a_vae_without_an_attribute_the_test_reads_is_left_whole(missing):
    """If ComfyUI renames one, a video VAE must not start to read as an image VAE."""
    vae = _image_vae()
    delattr(vae, missing)
    assert not vae_encode_item.takes_items(vae, torch.zeros(6, 8, 8, 3))
    three_d = _image_vae(latent_dim=3, not_video=True)
    del three_d.not_video
    assert not vae_encode_item.takes_items(three_d, torch.zeros(6, 8, 8, 3))


def _encode_spy():
    seen = []

    def encode(self, pixel_samples):
        seen.append(tuple(pixel_samples.shape))
        return pixel_samples.sum(dim=(1, 2, 3), keepdim=True)

    return encode, seen


def test_each_image_goes_through_the_original_alone_and_in_order():
    encode, seen = _encode_spy()
    wrapped = vae_encode_item.by_item(encode)
    pixels = torch.arange(6 * 4 * 4 * 3, dtype=torch.float32).reshape(6, 4, 4, 3)
    expected = pixels.sum(dim=(1, 2, 3), keepdim=True)
    assert torch.equal(wrapped(_image_vae(), pixels), expected)
    assert seen == [(1, 4, 4, 3)] * 6
    seen.clear()
    assert torch.equal(wrapped(_image_vae(), pixel_samples=pixels), expected)  # by the parameter's name
    assert seen == [(1, 4, 4, 3)] * 6
    seen.clear()
    assert wrapped(_image_vae(latent_dim=3), pixels).shape[0] == 6 and seen == [(6, 4, 4, 3)]  # video: whole
    seen.clear()
    assert wrapped(_image_vae(), pixels[:1]).shape[0] == 1 and seen == [(1, 4, 4, 3)]  # one image: one call
    assert vae_encode_item.state() == vae_encode_item.UNSET  # nothing stepped aside


def test_an_error_comfyui_raises_on_an_image_is_passed_on_and_nothing_runs_again(caplog):
    calls = []

    def encode(self, pixel_samples):
        calls.append(int(pixel_samples.shape[0]))
        if len(calls) == 3:
            raise MemoryError("out of memory on image 3")
        return pixel_samples.sum(dim=(1, 2, 3), keepdim=True)

    wrapped = vae_encode_item.by_item(encode)
    with caplog.at_level("WARNING"), pytest.raises(MemoryError, match="image 3"):
        wrapped(_image_vae(), torch.zeros(6, 4, 4, 3))
    assert calls == [1, 1, 1]  # no whole-batch call after stock's own error
    assert caplog.text == "" and vae_encode_item.state() == vae_encode_item.UNSET
    wrapped(_image_vae(), torch.zeros(2, 4, 4, 3))
    assert calls[3:] == [1, 1]  # and the rule is still on for the next call


class _SliceFails:
    shape = (6, 4, 4, 3)

    def __getitem__(self, _index):
        raise RuntimeError("no slices today")


def _not_a_tensor(self, pixel_samples):
    return {"samples": pixel_samples}


def _two_rows_an_image(self, pixel_samples):
    return torch.zeros(2 * pixel_samples.shape[0], 1)


@pytest.mark.parametrize("case", ["an unknown argument", "an unknown keyword", "results that are not tensors",
                                  "a result that is not one row an image", "a batch that cannot be sliced"])
def test_it_steps_aside_once_when_comfyui_does_not_behave_as_expected(case, caplog):
    """Drift in ComfyUI's call must not break a render: stock's whole batch answers, with one warning."""
    calls = []

    def original(self, pixel_samples, *args, **kwargs):
        calls.append((pixel_samples.shape[0], args, tuple(kwargs)))
        if case == "results that are not tensors":
            return _not_a_tensor(self, pixel_samples)
        if case == "a result that is not one row an image":
            return _two_rows_an_image(self, pixel_samples)
        if case == "a batch that cannot be sliced":
            return "stock's answer"
        return pixel_samples.sum(dim=(1, 2, 3), keepdim=True)

    wrapped = vae_encode_item.by_item(original)
    pixels = _SliceFails() if case == "a batch that cannot be sliced" else torch.zeros(6, 4, 4, 3)
    call = {"an unknown argument": lambda: wrapped(_image_vae(), pixels, 0.5),
            "an unknown keyword": lambda: wrapped(_image_vae(), pixels, tiled=True)}.get(
                case, lambda: wrapped(_image_vae(), pixels))
    with caplog.at_level("WARNING"):
        call()
    assert calls[-1][0] == 6  # the answer is stock's own call on the whole batch
    assert caplog.text.count("stepped aside") == 1
    assert vae_encode_item.state() == vae_encode_item.STEPPED_ASIDE
    # An argument the rule does not know is never dropped: no call on one image is made without it.
    if case == "an unknown argument":
        assert calls == [(6, (0.5,), ())]
    if case == "an unknown keyword":
        assert calls == [(6, (), ("tiled",))]
    # From then on every call is stock's, a plain batch included, and nothing more is logged.
    calls.clear()
    wrapped(_image_vae(), torch.zeros(4, 4, 4, 3))
    assert calls == [(4, (), ())] and caplog.text.count("stepped aside") == 1


def _comfy_stand_in(monkeypatch, *, total):
    """Stand-ins for what the gate reads: ComfyUI's modules in this process and the integrated probe."""
    model_management = SimpleNamespace(get_torch_device=lambda: "cuda:0")
    monkeypatch.setattr(driver_uma_memory, "_comfy", lambda: (model_management, object()))

    def probe(_torch, device):
        assert device == "cuda:0"
        if isinstance(total, Exception):
            raise total
        return total

    monkeypatch.setattr(driver_uma_memory, "integrated_total_bytes", probe)
    installs = []
    monkeypatch.setattr(vae_encode_item, "install", lambda: installs.append("install"))
    return installs


def test_on_by_default_on_an_integrated_gpu_and_only_there(monkeypatch, caplog):
    monkeypatch.delenv(vae_encode_item.DISABLE_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    installs = _comfy_stand_in(monkeypatch, total=124 * GIB)
    with caplog.at_level("INFO"):
        assert vae_encode_item.install_driver() == vae_encode_item.INSTALLED
    assert installs == ["install"] and vae_encode_item.state() == vae_encode_item.INSTALLED
    assert "one image a call" in caplog.text and vae_encode_item.DISABLE_ENV in caplog.text
    # A discrete GPU or the CPU: ComfyUI's own batches, nothing installed.
    installs = _comfy_stand_in(monkeypatch, total=None)
    assert vae_encode_item.install_driver() == vae_encode_item.NOT_INTEGRATED and installs == []
    assert vae_encode_item.state() == vae_encode_item.NOT_INTEGRATED
    installs = _comfy_stand_in(monkeypatch, total=RuntimeError("no device"))
    assert vae_encode_item.install_driver() == vae_encode_item.PROBE_FAILED and installs == []
    monkeypatch.setattr(sys, "platform", "darwin")
    installs = _comfy_stand_in(monkeypatch, total=124 * GIB)
    assert vae_encode_item.install_driver() == vae_encode_item.NOT_LINUX and installs == []


def test_its_own_switch_turns_it_off_and_the_readings_switch_does_not(monkeypatch, caplog):
    monkeypatch.setattr(sys, "platform", "linux")
    installs = _comfy_stand_in(monkeypatch, total=124 * GIB)
    monkeypatch.setenv(vae_encode_item.DISABLE_ENV, "1")
    with caplog.at_level("INFO"):
        assert vae_encode_item.install_driver() == vae_encode_item.OFF and installs == []
    assert vae_encode_item.DISABLE_ENV in caplog.text and vae_encode_item.state() == vae_encode_item.OFF
    monkeypatch.delenv(vae_encode_item.DISABLE_ENV)
    monkeypatch.setenv(driver_uma_memory.DISABLE_ENV, "1")
    assert driver_uma_memory.install_driver() == "off"
    assert vae_encode_item.install_driver() == vae_encode_item.INSTALLED and installs == ["install"]


def test_the_driver_hook_never_raises(monkeypatch):
    monkeypatch.delenv(vae_encode_item.DISABLE_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    _comfy_stand_in(monkeypatch, total=124 * GIB)

    def broken():
        raise ImportError("No module named 'comfy'")

    monkeypatch.setattr(vae_encode_item, "install", broken)
    assert vae_encode_item.install_driver() == vae_encode_item.NO_COMFY
    assert vae_encode_item.state() == vae_encode_item.NO_COMFY


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
    monkeypatch.delenv(vae_encode_item.DISABLE_ENV, raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    for name in [n for n in sys.modules if n == "comfy" or n.startswith("comfy.")]:
        monkeypatch.delitem(sys.modules, name)
    finder = _RecordComfyImports()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    assert vae_encode_item.install_driver() == vae_encode_item.NO_COMFY
    with pytest.raises(ImportError):
        vae_encode_item.install()
    assert finder.seen == []


def test_the_entry_point_installs_the_rule_after_the_reading():
    import ast
    import pathlib

    tree = ast.parse((pathlib.Path(__file__).resolve().parents[1] / "__init__.py").read_text())
    calls = [f"{stmt.value.func.value.id}.{stmt.value.func.attr}" for stmt in tree.body
             if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
             and isinstance(stmt.value.func, ast.Attribute) and isinstance(stmt.value.func.value, ast.Name)]
    assert "vae_encode_item.install_driver" in calls
    assert calls.index("driver_uma_memory.install_driver") < calls.index("vae_encode_item.install_driver")


def test_the_surface_canary_holds_the_call_the_wrapper_makes():
    """An upstream change of the encode call fails the daily canary, not a user's render."""
    from dgx_monarch import comfy_surface

    (touchpoint,) = [t for t in comfy_surface.TOUCHPOINTS if t.path == "comfy.sd.VAE.encode"]
    assert touchpoint.positional == ("self", "pixel_samples")

    def renamed(self, images):
        return images

    with pytest.raises(RuntimeError):
        comfy_surface._assert_touchpoint(renamed, touchpoint)
    comfy_surface._assert_touchpoint(lambda self, pixel_samples, tiled=False: pixel_samples, touchpoint)


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
        sys.argv = ["pytest-vae-encode-item", "--cpu"]
        try:
            options = pytest.importorskip("comfy.options", reason="no ComfyUI on sys.path")
            options.enable_args_parsing()
            sd = pytest.importorskip("comfy.sd")
        finally:
            sys.argv = argv_before
        yield sd
    finally:
        if "comfy.sd" in sys.modules:
            vae_encode_item.uninstall()
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


def _under_readings(vae, call, items):
    """``call()`` with the VAE's free-memory reading faked to fit 1, 2, 3 and all ``items`` a batch.

    Returns the result under each reading, and the batch sizes the VAE's own model was handed.
    """
    stock_reading = vae.patcher.get_free_memory
    model_encode = vae.first_stage_model.encode
    results, sizes = {}, {}
    try:
        for batch in (1, 2, 3, items):
            handed = sizes.setdefault(batch, [])

            def recording(pixels_in, *args, handed=handed, **kwargs):
                handed.append(int(pixels_in.shape[0]))
                return model_encode(pixels_in, *args, **kwargs)

            vae.first_stage_model.encode = recording
            vae.patcher.get_free_memory = lambda device, batch=batch: batch * call.need + call.need // 2
            results[batch] = call().float().cpu()
    finally:
        vae.patcher.get_free_memory = stock_reading
        vae.first_stage_model.encode = model_encode
    return results, sizes


def test_stocks_encode_follows_the_reading_and_the_rules_is_stocks_own_encode_of_each_image(comfy_sd):
    torch.set_num_threads(2)
    vae = _toy_vae(comfy_sd)
    pixels = torch.rand(6, 64, 64, 3, generator=torch.Generator().manual_seed(6))

    def encode():
        return vae.encode(pixels)

    encode.need = vae.memory_used_encode(tuple(pixels.movedim(-1, 1).shape), vae.vae_dtype)
    stock, stock_batches = _under_readings(vae, encode, 6)
    # Stock: the batches follow the reading. On a GPU the latent then differs from split to split
    # (56 to 64 percent of BF16 values apart, 2026-11-11). On a CPU whether float32 bits move
    # depends on the wheel, so this test holds the batches and the equalities, not that inequality.
    assert stock_batches == {1: [1] * 6, 2: [2, 2, 2], 3: [3, 3], 6: [6]}

    stock_encode, stock_decode = comfy_sd.VAE.encode, comfy_sd.VAE.decode
    vae_encode_item.install()
    try:
        assert getattr(comfy_sd.VAE.encode, "_dgxm_encode_item", False) is True
        vae_encode_item.install()  # a second install does not stack
        assert comfy_sd.VAE.encode.__wrapped__ is stock_encode
        assert comfy_sd.VAE.decode is stock_decode  # decode is not touched
        ruled, ruled_batches = _under_readings(vae, encode, 6)
        by_name = vae.encode(pixel_samples=pixels).float().cpu()
    finally:
        vae_encode_item.uninstall()
    assert comfy_sd.VAE.encode is stock_encode
    assert vae_encode_item.state() == vae_encode_item.UNSET  # it did not step aside on real ComfyUI
    # The rule: the VAE's model is handed one image at a time whatever memory reads, so the
    # result is one tensor.
    assert ruled_batches == {batch: [1] * 6 for batch in (1, 2, 3, 6)}
    for batch in (2, 3, 6):
        assert torch.equal(ruled[1], ruled[batch]), batch
    assert torch.equal(by_name, ruled[1])
    # It adds nothing of its own: image i's latent is stock's own encode of image i alone.
    alone = torch.cat([vae.encode(pixels[index:index + 1]).float().cpu() for index in range(6)], dim=0)
    assert torch.equal(ruled[1], alone)
    assert torch.equal(ruled[1], stock[1])
    # A stock box with room hands the model all six at once (stock_batches[6] above), which is
    # where its bits part from these; the docs say so.


def test_real_comfyuis_vae_has_the_attributes_and_the_branch_the_image_test_reads(comfy_sd):
    """The image test is stock's own branch on three attributes; their names are the drift surface.

    A touchpoint reads a signature and cannot see instance attributes or a
    function body, so they are held here, on a real instance and by source.
    """
    import inspect

    vae = _toy_vae(comfy_sd)
    assert vars(vae)["latent_dim"] == 2 and vars(vae)["not_video"] is False
    assert vars(vae)["extra_1d_channel"] is None
    assert list(inspect.signature(comfy_sd.VAE.encode).parameters) == ["self", "pixel_samples"]
    source = inspect.getsource(comfy_sd.VAE.encode)
    assert "if self.latent_dim == 3 and pixel_samples.ndim < 5:" in source
    assert "if not self.not_video:" in source

    encode, seen = _encode_spy()
    wrapped = vae_encode_item.by_item(encode)
    pixels = torch.zeros(6, 16, 16, 3)
    wrapped(vae, pixels)
    assert seen == [(1, 16, 16, 3)] * 6  # a real image VAE: one image a call
    seen.clear()
    vae.latent_dim = 3  # as ComfyUI marks a video VAE: its frames stay one call
    wrapped(vae, pixels)
    assert seen == [(6, 16, 16, 3)]
    seen.clear()
    vae.not_video = True  # as it marks an image VAE built on 3-D convolutions
    wrapped(vae, pixels)
    assert seen == [(1, 16, 16, 3)] * 6
    seen.clear()
    vae.latent_dim, vae.not_video, vae.extra_1d_channel = 2, False, 16  # as it marks an audio VAE
    wrapped(vae, pixels)
    assert seen == [(6, 16, 16, 3)]
