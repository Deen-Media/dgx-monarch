"""driver_spend_price: a node's spend on the driver that the loader did not charge, priced where it happens.

A re-queue whose UNET Loader ComfyUI kept, or a node ComfyUI runs before the
loader, spends on the driver's box with no charge before it. These tests run
the wraps on the CPU against stand-ins shaped as ComfyUI's classes: a v3 Add
Guide and Reference to Video with ``execute`` as a classmethod, v1 loaders with
a ``FUNCTION`` method, and a v3 loader, all in a stand-in node mapping. The
executing prompt, the running graph, the loader's per-term marks and the box's
reading are set by each test. Canary seam 64 (tests/canary/comfy_seam_contracts.py)
runs the wrap through ComfyUI's own executor, and
``tests/test_driver_spend_price_marks.py`` runs the marks through the loader.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from dgx_monarch import (
    config,
    driver_footprint,
    driver_spend_marks,
    driver_spend_price,
    driver_spend_settings,
    driver_unload,
    mesh,
    mesh_safety,
    vae_decode_relief,
)
from dgx_monarch.adapters import SELECTABLE_FAMILIES, detect
from dgx_monarch.nodes import graph_load_price, loader_graph
from dgx_monarch.refusal import RefusalClass, parse_refusal_tag

GIB = 2 ** 30
PROMPT = "6d1c-prompt"
GRAPH = {"2": {"class_type": "DGXMonarchUNETLoader", "inputs": {"unet_name": "h3.safetensors"}}}
SIZES = {"te.safetensors": 1000, "te2.safetensors": 3000, "dit.safetensors": 5000,
         "a.safetensors": 100, "b.safetensors": 200, "c.safetensors": 400}


class _Guide:
    """ComfyUI's Add Guide (MiniMax H3) in shape: a classmethod ``execute`` that records each call."""

    calls: list = []

    @classmethod
    def execute(cls, positive, latent, frame_idx, vae=None, audio_vae=None, image=None, audio=None):
        cls.calls.append((positive, latent, frame_idx, vae, image))
        return ("guided", latent)


class _Reference:
    calls: list = []

    @classmethod
    def execute(cls, clip, prompt, width, height, length, ref_image_size="match", vae=None,
                audio_vae=None, ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None):
        cls.calls.append(ref_videos)
        return ("referenced",)


class _ClipLoader:
    """ComfyUI's v1 Load CLIP in shape."""

    FUNCTION = "load_clip"
    calls: list = []

    def load_clip(self, clip_name, type="stable_diffusion", device="default"):
        self.calls.append(clip_name)
        return ("clip",)


class _DualClip:
    FUNCTION = "load_clip"
    calls: list = []

    def load_clip(self, clip_name1, clip_name2, type, device="default"):
        self.calls.append((clip_name1, clip_name2))
        return ("clip",)


class _Unet:
    FUNCTION = "load_unet"
    calls: list = []

    def load_unet(self, unet_name, weight_dtype):
        self.calls.append(unet_name)
        return ("model",)


class _Triple:
    """ComfyUI's v3 TripleCLIPLoader in shape."""

    calls: list = []

    @classmethod
    def execute(cls, clip_name1, clip_name2, clip_name3):
        cls.calls.append((clip_name1, clip_name2, clip_name3))
        return ("clip",)


MAPPING = {driver_spend_price.GUIDE_CLASS: _Guide, driver_spend_price.REFERENCE_CLASS: _Reference,
           "CLIPLoader": _ClipLoader, "DualCLIPLoader": _DualClip, "UNETLoader": _Unet, "TripleCLIPLoader": _Triple}


def _latent(width: int, height: int, frames: int = 31):
    """An H3 AV latent as Add Guide reads it: the video stream first, its grid at 1/16 of the canvas."""
    return {"samples": SimpleNamespace(tensors=[SimpleNamespace(shape=(1, 24, frames, height // 16, width // 16))])}


def _image(frames: int):
    return SimpleNamespace(shape=(frames, 64, 64, 3))


class _Box:
    """MemAvailable as the test sets it, and how often it was read."""

    def __init__(self, value: int):
        self.value = value
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return self.value


@pytest.fixture
def rig(monkeypatch):
    """The rule installed on stand-ins, an unmarked prompt with a priced graph, and a roomy box."""
    for cls in MAPPING.values():
        cls.calls = []
        method = "execute" if "execute" in cls.__dict__ else cls.FUNCTION
        monkeypatch.setattr(cls, method, cls.__dict__[method])
    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace(NODE_CLASS_MAPPINGS=dict(MAPPING)))
    monkeypatch.setattr(driver_spend_price, "_STATE", {"state": driver_spend_price.INSTALLED, "warned": ""})
    monkeypatch.setattr(driver_spend_marks, "_CHARGED", {})
    monkeypatch.setattr(driver_spend_price, "_TOLD", {})
    monkeypatch.setattr(graph_load_price, "_PRICED_PROMPTS", {})
    monkeypatch.delenv(driver_spend_price.DISABLE_ENV, raising=False)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(driver_footprint, "preflight_disabled", lambda: False)
    box = _Box(100 * GIB)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", box)
    monkeypatch.setattr(driver_spend_price, "_pool_free_bytes", lambda: 0)
    context = {"prompt": PROMPT, "node": "7"}
    monkeypatch.setattr(driver_spend_price, "_context", lambda: (context["prompt"], context["node"]))
    graphs = {"graph": GRAPH}
    monkeypatch.setattr(driver_spend_price, "running_graph", lambda prompt_id: graphs["graph"])
    family = {"name": "minimax_h3"}
    monkeypatch.setattr(driver_spend_price, "priced_loader", lambda graph: (
        None if family["name"] is None else driver_spend_settings.PricedLoader(family["name"], {})))
    activations = {"bytes": 11 * GIB}
    monkeypatch.setattr(loader_graph, "graph_activation_bytes",
                        lambda graph, sp, family: (activations["bytes"], ""))
    monkeypatch.setattr(loader_graph, "_artifact_bytes", lambda folders, name: SIZES.get(name, 0))
    monkeypatch.setattr(vae_decode_relief, "_serving", lambda: (None, vae_decode_relief.NO_FLEET))
    monkeypatch.setattr(driver_unload, "comfy_model_management", lambda: None)
    monkeypatch.setattr(driver_unload, "busy_reason", lambda handle: "")
    # DynamicVRAM on, as for both 2026-11-13 readings; one test turns it off.
    monkeypatch.setitem(sys.modules, "comfy.memory_management", SimpleNamespace(aimdo_enabled=True))
    assert sorted(driver_spend_price.install()) == sorted(MAPPING)
    return SimpleNamespace(box=box, context=context, graphs=graphs, family=family, activations=activations,
                           floor=driver_footprint.reserve_bytes(None))


def _guide(frames: int, width: int = 1344, height: int = 768, vae=None):
    latent = _latent(width, height)
    return _Guide.execute("positive", latent, 0, vae=vae, image=_image(frames)), latent


def _refused(exc_info) -> str:
    text = str(exc_info.value)
    tag = parse_refusal_tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY
    assert tag.guard == "driver_footprint_preflight" and not tag.waivable
    return text


def _mark(term: str, figure: int) -> None:
    driver_spend_marks.record(PROMPT, term, figure)


# The frames Add Guide encodes.

@pytest.mark.parametrize(("batch", "frames"), [
    (1, 1), (4, 1), (5, 5), (6, 5), (21, 5), (22, 22), (38, 22), (39, 39), (40, 39)])
def test_the_frames_are_add_guides_own_cut(batch, frames):
    assert driver_spend_price.guide_frames(batch) == frames


# A spend the loader charged: never refused.

def test_a_guide_the_loader_charged_at_its_size_is_priced_nowhere_and_says_so_once(rig, caplog):
    _mark(driver_spend_price.GUIDE_TERM, 25 * GIB)  # a still's 14, held as the larger of it and 25 of activations
    rig.box.value = 1 * GIB
    with caplog.at_level(logging.INFO, logger="dgx_monarch.driver_spend_price"):
        _guide(1)
        _guide(1)
    assert len(_Guide.calls) == 2 and rig.box.reads == 0
    assert caplog.text.count("the loader priced this prompt; nothing priced here") == 1


def test_a_marked_guide_steps_aside_before_comfy_has_loaded_its_vae(rig, monkeypatch):
    """The loader charges the VAE among the models, not in the guide's mark: before Add Guide's first
    encode ComfyUI has not loaded the VAE, and the mark is held to the encode alone."""
    patcher = SimpleNamespace(model_size=lambda: 5 * GIB)
    monkeypatch.setitem(sys.modules, "comfy.model_management", SimpleNamespace(current_loaded_models=[]))
    _mark(driver_spend_price.GUIDE_TERM, int(15.9 * GIB))  # P1's shape: max(14, 15.9) of activations
    rig.box.value = 1 * GIB
    _guide(1, vae=SimpleNamespace(patcher=patcher))
    assert len(_Guide.calls) == 1 and rig.box.reads == 0


def test_a_guide_the_loader_charged_below_its_size_is_priced_live(rig):
    """A still's mark does not cover the clip queued after it: the mark is per term and per figure."""
    _mark(driver_spend_price.GUIDE_TERM, 14 * GIB)
    rig.box.value = 20 * GIB
    with pytest.raises(driver_spend_price.DriverSpendCapacityError):
        _guide(5, width=448, height=256)
    assert _Guide.calls == []


def test_a_mark_on_one_term_leaves_another_priced(rig):
    _mark(driver_spend_price.GUIDE_TERM, 100 * GIB)
    rig.box.value = 30 * GIB
    with pytest.raises(driver_spend_price.DriverSpendCapacityError):
        _Reference.execute("clip", "a prompt", 1344, 768, 124, ref_videos={"ref_video_1": _image(5)})
    _mark(driver_spend_price.REFERENCE_TERM, 40 * GIB)
    _Reference.execute("clip", "a prompt", 1344, 768, 124, ref_videos={"ref_video_1": _image(5)})
    assert len(_Reference.calls) == 1


def test_what_the_loader_charged_is_recorded_per_term():
    terms = loader_graph.DriverStackTerms(resolved=True, guide_bytes=14 * GIB, activation_bytes=25 * GIB,
                                          guide_first=True, ref_bytes=40 * GIB)
    prompt = {"3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "te.safetensors"}},
              "2": {"class_type": "DGXMonarchUNETLoader", "inputs": {"unet_name": "dit.safetensors"}}}
    store: dict = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(driver_spend_marks, "_CHARGED", store)
        patch.setattr(loader_graph, "_artifact_bytes", lambda folders, name: SIZES.get(name, 0))
        patch.setattr("dgx_monarch.nodes.graph_live.live_nodes", lambda graph: frozenset(graph))
        driver_spend_marks.note_charged(PROMPT, terms, prompt)
        assert driver_spend_price.charged(PROMPT, driver_spend_price.GUIDE_TERM) == 25 * GIB
        assert driver_spend_price.charged(PROMPT, driver_spend_price.REFERENCE_TERM) == 40 * GIB
        assert driver_spend_price.charged(PROMPT, driver_spend_price.node_term("3")) == int(
            1000 * loader_graph.TE_RESIDENT_RATIO)
        assert driver_spend_price.charged(PROMPT, driver_spend_price.node_term("2")) == 0
        store.clear()
        driver_spend_marks.note_charged(PROMPT, replace(terms, credited=True), prompt)
        assert driver_spend_price.charged(PROMPT, driver_spend_price.node_term("3")) == 0
        assert driver_spend_price.charged(PROMPT, driver_spend_price.GUIDE_TERM) == 25 * GIB


def test_the_loader_records_only_where_it_admits_a_graph_it_read(monkeypatch):
    noted: list = []
    monkeypatch.setattr(driver_spend_marks, "note_charged", lambda *args: noted.append(args))
    monkeypatch.setattr(graph_load_price, "executing_prompt_id", lambda: PROMPT)
    read = loader_graph.DriverStackTerms(resolved=True)
    graph_load_price.note_admitted(read, graph_load_price.SITE_SAMPLER, {})
    graph_load_price.note_admitted(loader_graph.DriverStackTerms(), graph_load_price.SITE_LOADER, {})
    assert noted == []
    graph_load_price.note_admitted(read, graph_load_price.SITE_LOADER, {"1": {}})
    assert noted == [(PROMPT, read, {"1": {}})]


# An unmarked spend: refused only under the floor after every rescue.

def test_a_still_with_room_is_admitted_with_one_line_and_runs_as_stock(rig, caplog):
    with caplog.at_level(logging.INFO, logger="dgx_monarch.driver_spend_price"):
        result, latent = _guide(1)
    assert result == ("guided", latent) and len(_Guide.calls) == 1
    assert _Guide.calls[0][1] is latent and _Guide.calls[0][2] == 0
    assert ("Add Guide (MiniMax H3) for this still at 1344x768 admitted: 14.0 GiB charged against 95.0 GiB "
            "of room above the 5.0 GiB floor (MemAvailable 100.0 GiB") in caplog.text


@pytest.mark.parametrize("reached", [False, True], ids=["loader not reached", "loader reached"])
def test_a_clip_short_of_room_refuses_before_anything_encodes(rig, reached):
    if reached:
        graph_load_price._PRICED_PROMPTS[f"{PROMPT}\x00h3.safetensors"] = None
    rig.box.value = 20 * GIB
    with pytest.raises(driver_spend_price.DriverSpendCapacityError) as exc_info:
        _guide(5, width=448, height=256)
    text = _refused(exc_info)
    assert _Guide.calls == []
    assert ("Add Guide (MiniMax H3) for this 5-frame clip at 448x256 needs 18.0 GiB on the driver's box, "
            "and the box has 15.0 GiB above its 5.0 GiB floor") in text
    if reached:
        assert "The UNET Loader priced this prompt but did not charge this at its size" in text
    else:
        assert "ComfyUI kept its output from an earlier queue, or runs this node before it" in text
    assert "Nothing was encoded. Use a shorter clip or a smaller canvas, or free memory on this box." in text
    assert "itemises the whole graph's figures" in text
    assert driver_spend_price.DISABLE_ENV not in text


def test_the_line_is_the_floor_after_the_rescues(rig, monkeypatch):
    """Admitted at exactly the floor plus the charge; refused a byte under it, after the unload ran."""
    unloads: list = []
    monkeypatch.setattr(driver_unload, "comfy_model_management", lambda: object())
    monkeypatch.setattr(driver_unload, "held_models", lambda mm: [(object(), "the text encoder")])

    def unload(mm, held):
        unloads.append(1)
        return driver_unload.Unloaded(("the text encoder",), rig.box.value, rig.box.value)

    monkeypatch.setattr(driver_unload, "unload", unload)
    rig.box.value = rig.floor + 14 * GIB
    _guide(1)
    assert unloads == [] and len(_Guide.calls) == 1
    rig.box.value = rig.floor + 14 * GIB - 1
    with pytest.raises(driver_spend_price.DriverSpendCapacityError):
        _guide(1)
    assert unloads == [1] and len(_Guide.calls) == 1


def test_a_clip_whose_activations_do_not_resolve_carries_no_margin(rig):
    rig.activations["bytes"] = 0
    rig.box.value = 22 * GIB  # 17 GiB plus the floor fits; 18 would not
    _guide(5, width=448, height=256)
    assert len(_Guide.calls) == 1


def test_a_rescue_runs_before_the_refusal_and_its_room_admits(rig, monkeypatch):
    rig.box.value = 15 * GIB
    unloaded: list = []

    def unload(mm, held):
        unloaded.append([name for _entry, name in held])
        before, rig.box.value = rig.box.value, rig.box.value + 10 * GIB
        return driver_unload.Unloaded(("the text encoder",), before, rig.box.value)

    monkeypatch.setattr(driver_unload, "comfy_model_management", lambda: object())
    monkeypatch.setattr(driver_unload, "held_models", lambda mm: [(object(), "the text encoder")])
    monkeypatch.setattr(driver_unload, "unload", unload)
    _guide(1)
    assert unloaded == [["the text encoder"]] and len(_Guide.calls) == 1


def _vae(monkeypatch, loaded: bool):
    """A VAE of 3 GiB, held loaded by ComfyUI's model management or not."""
    patcher = SimpleNamespace(model_size=lambda: 3 * GIB)
    held = [SimpleNamespace(model=patcher)] if loaded else []
    monkeypatch.setitem(sys.modules, "comfy.model_management", SimpleNamespace(current_loaded_models=held))
    return SimpleNamespace(patcher=patcher), held


def test_with_dynamic_vram_an_unloaded_vae_costs_the_first_encode_nothing(rig, monkeypatch):
    """On 2026-11-13 a fresh driver's first encode, which loads the VAE, raised the driver's figure 0.60 GiB."""
    vae, _held = _vae(monkeypatch, loaded=False)
    rig.box.value = 19 * GIB  # 14 plus the floor: fits, though the VAE is not loaded
    _guide(1, vae=vae)
    assert len(_Guide.calls) == 1


def test_a_trim_that_frees_enough_for_the_first_figure_admits(rig, monkeypatch):
    """A rescue that did not take the VAE does not raise the bar: the VAE was never loaded, so nothing reloads."""
    vae, _held = _vae(monkeypatch, loaded=False)
    fleet = object()
    monkeypatch.setattr(vae_decode_relief, "_serving", lambda: (fleet, ""))
    monkeypatch.setattr(driver_unload, "trimmable", lambda handle: handle is fleet)

    def trim(handle, level=driver_unload.TRIM_LEVEL):
        before, rig.box.value = rig.box.value, rig.box.value + 2 * GIB
        return driver_unload.Trimmed(2, before, rig.box.value)

    monkeypatch.setattr(driver_unload, "trim_actors", trim)
    rig.box.value = 18 * GIB  # 13 above the floor, short of 14; the trim frees 2
    _guide(1, vae=vae)
    assert len(_Guide.calls) == 1


def test_an_unload_that_took_the_vae_charges_it_again(rig, monkeypatch):
    vae, held = _vae(monkeypatch, loaded=True)

    def unload(mm, models):
        held.clear()  # the unload took the VAE ComfyUI held
        before, rig.box.value = rig.box.value, rig.box.value + 2 * GIB
        return driver_unload.Unloaded(("the VAE",), before, rig.box.value)

    monkeypatch.setattr(driver_unload, "comfy_model_management", lambda: object())
    monkeypatch.setattr(driver_unload, "held_models", lambda mm: [(object(), "the VAE")])
    monkeypatch.setattr(driver_unload, "unload", unload)
    rig.box.value = 18 * GIB  # short; the unload frees 2 GiB (16 above the floor), and the VAE's 3 come back
    with pytest.raises(driver_spend_price.DriverSpendCapacityError) as exc_info:
        _guide(1, vae=vae)
    assert "needs 17.0 GiB" in _refused(exc_info) and _Guide.calls == []


def test_a_refusal_with_models_held_is_true_of_the_box_after_the_unload_and_the_trim(rig, monkeypatch):
    """The restated promise where the loader would credit what ComfyUI holds: the driver unloads it and trims
    the Workers first, and a refusal that stands is true of the box after both."""
    fleet, rescues = object(), []
    monkeypatch.setattr(vae_decode_relief, "_serving", lambda: (fleet, ""))
    monkeypatch.setattr(driver_unload, "trimmable", lambda handle: handle is fleet)

    def unload(mm, models):
        rescues.append("unload")
        before, rig.box.value = rig.box.value, rig.box.value + 4 * GIB
        return driver_unload.Unloaded(("text encoder",), before, rig.box.value)

    def trim(handle, level=driver_unload.TRIM_LEVEL):
        rescues.append("trim")
        before, rig.box.value = rig.box.value, rig.box.value + 1 * GIB
        return driver_unload.Trimmed(2, before, rig.box.value)

    monkeypatch.setattr(driver_unload, "comfy_model_management", lambda: object())
    monkeypatch.setattr(driver_unload, "held_models", lambda mm: [(object(), "text encoder")])
    monkeypatch.setattr(driver_unload, "unload", unload)
    monkeypatch.setattr(driver_unload, "trim_actors", trim)
    rig.box.value = 8 * GIB
    with pytest.raises(driver_spend_price.DriverSpendCapacityError) as exc_info:
        _guide(1)
    text = _refused(exc_info)
    assert rescues == ["unload", "trim"] and _Guide.calls == []
    assert 14 * GIB + rig.floor > rig.box.value == 13 * GIB  # the spend and the floor exceed the box after both
    assert f"has {(rig.box.value - rig.floor) / GIB:.1f} GiB above its {rig.floor / GIB:.1f} GiB floor" in text


def test_without_dynamic_vram_the_probes_figures_stand(rig, monkeypatch):
    """Both 2026-11-13 readings were taken with DynamicVRAM on. Without it an unloaded VAE is charged at the first
    encode, as load_models_gpu copies it to the device, and a clip carries the 6 GiB margin."""
    monkeypatch.setitem(sys.modules, "comfy.memory_management", SimpleNamespace(aimdo_enabled=False))
    vae, _held = _vae(monkeypatch, loaded=False)
    rig.box.value = 21 * GIB  # 14 plus the floor fits; 14 plus the VAE's 3 does not
    with pytest.raises(driver_spend_price.DriverSpendCapacityError) as exc_info:
        _guide(1, vae=vae)
    assert "needs 17.0 GiB" in _refused(exc_info)
    rig.box.value = 27 * GIB  # 17 plus 6 plus the floor is 28
    with pytest.raises(driver_spend_price.DriverSpendCapacityError) as exc_info:
        _guide(5, width=448, height=256)
    assert "needs 23.0 GiB" in _refused(exc_info)


def test_a_reference_video_is_charged_the_loaders_figure(rig):
    rig.box.value = 30 * GIB
    with pytest.raises(driver_spend_price.DriverSpendCapacityError) as exc_info:
        _Reference.execute("clip", "a prompt", 1344, 768, 124, ref_videos={"ref_video_1": _image(5)})
    assert "Reference to Video (MiniMax H3) for its reference video needs 40.0 GiB" in _refused(exc_info)
    _Reference.execute("clip", "a prompt", 1344, 768, 124, ref_images={"ref_image_1": _image(1)})
    assert _Reference.calls == [None]


# Where nothing is priced.

@pytest.mark.parametrize("case", ["no prompt", "no graph", "a family the loader does not price",
                                  "the switch", "an unreadable latent", "no image"])
def test_what_prices_nothing_runs_as_stock(rig, monkeypatch, case):
    rig.box.value = 1 * GIB
    latent: object = _latent(448, 256)
    image: object = _image(5)
    if case == "no prompt":
        rig.context["prompt"] = None
    elif case == "no graph":
        rig.graphs["graph"] = None
    elif case == "a family the loader does not price":
        rig.family["name"] = None
    elif case == "the switch":
        monkeypatch.setenv(driver_spend_price.DISABLE_ENV, "1")
    elif case == "an unreadable latent":
        latent = {"samples": object()}
    else:
        image = None
    _Guide.execute("positive", latent, 0, image=image)
    assert len(_Guide.calls) == 1


# The stock loader nodes.

def test_load_clip_is_charged_its_file_at_the_loaders_ratio(rig):
    rig.context["node"] = "3"
    charge = int(1000 * loader_graph.TE_RESIDENT_RATIO)
    rig.box.value = rig.floor + charge - 1
    with pytest.raises(driver_spend_price.DriverSpendCapacityError) as exc_info:
        _ClipLoader().load_clip("te.safetensors", type="minimax")
    text = _refused(exc_info)
    assert "CLIPLoader loading te.safetensors needs" in text and "Nothing was loaded. Pick a smaller file" in text
    assert _ClipLoader.calls == []
    rig.box.value = rig.floor + charge
    assert _ClipLoader().load_clip("te.safetensors", type="minimax") == ("clip",)


def test_several_files_are_summed(rig):
    rig.context["node"] = "4"
    rig.box.value = rig.floor + int(4000 * loader_graph.TE_RESIDENT_RATIO) - 1
    with pytest.raises(driver_spend_price.DriverSpendCapacityError):
        _DualClip().load_clip("te.safetensors", "te2.safetensors", "flux")
    rig.box.value = rig.floor + int(700 * loader_graph.TE_RESIDENT_RATIO) - 1
    with pytest.raises(driver_spend_price.DriverSpendCapacityError):
        _Triple.execute(clip_name1="a.safetensors", clip_name2="b.safetensors", clip_name3="c.safetensors")
    rig.box.value = rig.floor + int(700 * loader_graph.TE_RESIDENT_RATIO)
    assert _Triple.execute(clip_name1="a.safetensors", clip_name2="b.safetensors",
                           clip_name3="c.safetensors") == ("clip",)


def test_a_model_on_the_driver_is_charged_its_file_whole(rig):
    rig.context["node"] = "5"
    rig.box.value = rig.floor + 4999
    with pytest.raises(driver_spend_price.DriverSpendCapacityError):
        _Unet().load_unet("dit.safetensors", "default")
    rig.box.value = rig.floor + 5000
    assert _Unet().load_unet("dit.safetensors", "default") == ("model",)


def test_a_file_comfy_already_holds_is_charged_again(rig, monkeypatch):
    """load_clip never shares a file with a CLIP ComfyUI holds: a second copy is a real spend."""
    rig.context["node"] = "3"
    monkeypatch.setitem(sys.modules, "comfy.model_management",
                        SimpleNamespace(current_loaded_models=[SimpleNamespace(model=object())]))
    rig.box.value = rig.floor + int(1000 * loader_graph.TE_RESIDENT_RATIO) - 1
    with pytest.raises(driver_spend_price.DriverSpendCapacityError):
        _ClipLoader().load_clip("te.safetensors")


def test_a_load_node_the_loader_charged_is_priced_nowhere(rig):
    rig.context["node"] = "3"
    _mark(driver_spend_price.node_term("3"), int(1000 * loader_graph.TE_RESIDENT_RATIO))
    rig.box.value = 1 * GIB
    assert _ClipLoader().load_clip("te.safetensors") == ("clip",)
    rig.context["node"] = "9"  # another Load CLIP of the same file, which the loader did not charge
    with pytest.raises(driver_spend_price.DriverSpendCapacityError):
        _ClipLoader().load_clip("te.safetensors")


def test_a_load_with_no_graph_in_hand_prices_nothing(rig):
    rig.box.value = 1 * GIB
    rig.graphs["graph"] = None
    assert _ClipLoader().load_clip("te.safetensors") == ("clip",)


# Installing.

def test_a_second_install_wraps_nothing_more(rig):
    guide, clip = _Guide.__dict__["execute"], _ClipLoader.__dict__["load_clip"]
    assert driver_spend_price.install() == []
    assert _Guide.__dict__["execute"] is guide and _ClipLoader.__dict__["load_clip"] is clip
    assert getattr(guide.__func__, driver_spend_price._KEY) and getattr(clip, driver_spend_price._KEY)


@pytest.mark.parametrize("marker", driver_spend_price.WORKER_ENV_MARKERS)
def test_a_worker_installs_nothing_and_says_so(monkeypatch, caplog, marker):
    monkeypatch.setattr(driver_spend_price, "_STATE", {"state": driver_spend_price.UNSET, "warned": ""})
    monkeypatch.setenv(marker, "1")
    installs: list = []
    monkeypatch.setattr(driver_spend_price, "install", lambda: installs.append(1) or [])
    with caplog.at_level(logging.INFO, logger="dgx_monarch.driver_spend_price"):
        assert driver_spend_price.install_driver() == driver_spend_price.WORKER and installs == []
    assert "this process is a Worker's actor" in caplog.text


def test_the_drivers_own_pythonpath_marker_does_not_make_it_a_worker(monkeypatch):
    """The driver sets DGXM_PYTHONPATH at every bring-up, and a ComfyUI-Manager restart re-execs it with it."""
    from dgx_monarch import driver_uma_memory

    monkeypatch.setattr(driver_spend_price, "_STATE", {"state": driver_spend_price.UNSET, "warned": ""})
    for marker in driver_spend_price.WORKER_ENV_MARKERS:
        monkeypatch.delenv(marker, raising=False)
    monkeypatch.delenv(driver_spend_price.DISABLE_ENV, raising=False)
    monkeypatch.setenv("DGXM_PYTHONPATH", "/somewhere/src")
    monkeypatch.setattr(sys, "platform", "linux")

    def no_comfy():
        raise ImportError("no ComfyUI in this test")

    monkeypatch.setattr(driver_uma_memory, "comfy_modules", no_comfy)
    assert driver_spend_price.install_driver() == driver_spend_price.NO_COMFY


def test_a_class_another_pack_registered_over_a_wrapped_name_is_named_once(rig, caplog):
    class Other:
        @classmethod
        def execute(cls, **kwargs):
            return None

    class Inherits(_ClipLoader):
        pass

    sys.modules["nodes"].NODE_CLASS_MAPPINGS[driver_spend_price.GUIDE_CLASS] = Other
    sys.modules["nodes"].NODE_CLASS_MAPPINGS["UNETLoader"] = Inherits
    with caplog.at_level(logging.WARNING, logger="dgx_monarch.driver_spend_price"):
        driver_spend_price.warn_if_unwrapped()
        driver_spend_price.warn_if_unwrapped()
    assert caplog.text.count("MiniMaxH3AddGuide in ComfyUI's node mapping is not wrapped (another pack") == 1
    assert caplog.text.count("UNETLoader in ComfyUI's node mapping is not wrapped (its node method is "
                             "inherited") == 1


def test_a_reused_prompt_id_does_not_carry_an_earlier_queues_marks(rig, monkeypatch):
    """ComfyUI takes a client's own prompt id: a mark is keyed by the running item's queue number too."""
    item = {"number": 1}
    monkeypatch.setattr(driver_spend_marks, "running_item", lambda prompt_id: (item["number"], prompt_id, GRAPH))
    _mark(driver_spend_price.GUIDE_TERM, 100 * GIB)
    rig.box.value = 1 * GIB
    _guide(1)  # the same queue: marked, steps aside
    item["number"] = 2  # the same id sent again, a new queue whose loader ComfyUI kept
    with pytest.raises(driver_spend_price.DriverSpendCapacityError):
        _guide(1)
    assert len(_Guide.calls) == 1


@pytest.mark.parametrize("fault", ["_active", "_context", "charged", "spend_of"])
def test_a_fault_in_the_price_steps_aside_and_runs_stock_once(rig, monkeypatch, fault):
    def broken(*_args):
        raise RuntimeError(f"{fault} is broken")

    if fault == "spend_of":
        monkeypatch.setattr(driver_spend_price, "_spend_for", lambda name, arguments: broken)
    else:
        monkeypatch.setattr(driver_spend_price, fault, broken)
    rig.box.value = 1 * GIB
    assert _guide(5, width=448, height=256)[0][0] == "guided"
    assert len(_Guide.calls) == 1


def test_a_v1_instance_made_before_the_install_reaches_the_wrap(rig, monkeypatch):
    """ComfyUI may hold a v1 node's instance from before this pack loaded: the method is the class's."""
    monkeypatch.setattr(_ClipLoader, "load_clip", _ClipLoader.load_clip.__wrapped__)
    made_before = _ClipLoader()
    assert driver_spend_price.install() == ["CLIPLoader"]
    rig.context["node"] = "3"
    rig.box.value = 1 * GIB
    with pytest.raises(driver_spend_price.DriverSpendCapacityError):
        made_before.load_clip("te.safetensors")
    assert _ClipLoader.calls == []


def test_the_refusal_is_one_the_driver_rescues_and_not_the_stock_load_family():
    error = driver_spend_price.DriverSpendCapacityError("x")
    assert driver_unload.is_capacity_refusal(error)
    assert not mesh_safety.is_stock_load_capacity_error(error)


def test_the_running_graph_is_read_from_comfys_queue_by_prompt_id(monkeypatch):
    graph = {"1": {"class_type": "X", "inputs": {}}}
    items = [(0, "other", {"9": {}}, {}, [], {}), (1, PROMPT, graph, {}, [], {})]
    queue = SimpleNamespace(get_current_queue_volatile=lambda: (items, []))
    server = SimpleNamespace(PromptServer=SimpleNamespace(instance=SimpleNamespace(prompt_queue=queue)))
    monkeypatch.setitem(sys.modules, "server", server)
    assert driver_spend_price.running_graph(PROMPT) is graph
    assert driver_spend_price.running_graph("absent") is None and driver_spend_price.running_graph(None) is None
    monkeypatch.setitem(sys.modules, "server", SimpleNamespace())
    assert driver_spend_price.running_graph(PROMPT) is None


# The loader's two settings, read from the running graph's Init node by the real reader.

def _settings(rig, monkeypatch, init: dict | None, sniffed: object = ("minimax_h3", "int8"),
              config_args: dict | None = None, attached: dict | None = None) -> None:
    """The real reader on a graph whose loader takes its mesh from an Init node with these inputs (None: no
    Init node), a header that sniffs as ``sniffed`` (an exception: unreadable), the configured policy
    ``config_args`` of the file the Init node names, and a mesh attached with policy ``attached``."""
    graph = {"2": {"class_type": "DGXMonarchUNETLoader", "inputs": {"mesh": ["1", 0], "unet_name": "h3.safetensors"}}}
    if init is not None:
        graph["1"] = {"class_type": "DGXMonarchInit", "inputs": {"topology": "auto", "mode": "auto", **init}}
    rig.graphs["graph"] = graph
    monkeypatch.setattr(driver_spend_price, "priced_loader", driver_spend_settings.priced_loader)
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(
        get_full_path=lambda folder, name: f"/models/{folder}/{name}"))

    def sniff(path):
        if isinstance(sniffed, Exception):
            raise sniffed
        return sniffed

    monkeypatch.setattr(detect, "sniff_checkpoint", sniff)
    monkeypatch.setattr(config, "find_config_path", lambda explicit=None: None)
    monkeypatch.setattr(config, "local_config", lambda path=None: SimpleNamespace(worker_args=dict(config_args or {})))
    handles = {}
    if attached is not None:
        handles["key"] = SimpleNamespace(effective_worker_args=lambda spec: {**attached, **dict(spec or {})})
    monkeypatch.setattr(mesh, "_MESHES", handles)


@pytest.mark.parametrize("sniffed", [("unknown", "int8"), detect.CheckpointSniffError("no header")],
                         ids=["a header that reads unknown", "a header that cannot be read"])
def test_a_header_the_loader_cannot_name_with_a_priced_family_forced_on_init_is_priced(rig, monkeypatch, sniffed):
    rig.box.value = 10 * GIB  # short of a still's 14 plus the floor
    _settings(rig, monkeypatch, {"family_adapter": "minimax_h3"}, sniffed=sniffed)
    with pytest.raises(driver_spend_price.DriverSpendCapacityError):
        _guide(1)
    _settings(rig, monkeypatch, {"family_adapter": "auto"}, sniffed=sniffed)
    _guide(1)  # on auto the loader prices nothing for this header, and nor does the wrap
    assert len(_Guide.calls) == 1


def test_a_forced_family_the_loader_does_not_price_prices_nothing(rig, monkeypatch):
    unpriced = next(f for f in SELECTABLE_FAMILIES if f not in driver_footprint.DRIVER_STACK_FAMILIES)
    rig.box.value = 10 * GIB
    _settings(rig, monkeypatch, {"family_adapter": unpriced})  # the header reads H3, the forced family wins
    _guide(1)
    assert len(_Guide.calls) == 1


def test_with_no_init_node_the_sniff_alone_decides_at_the_default_floor(rig, monkeypatch):
    rig.box.value = 10 * GIB
    _settings(rig, monkeypatch, None)
    with pytest.raises(driver_spend_price.DriverSpendCapacityError) as exc_info:
        _guide(1)
    assert "above its 5.0 GiB floor" in _refused(exc_info)
    _settings(rig, monkeypatch, None, sniffed=detect.CheckpointSniffError("no header"))
    _guide(1)
    assert len(_Guide.calls) == 1


def test_a_file_the_loader_does_not_open_is_not_priced_even_when_forced(rig, monkeypatch):
    rig.box.value = 10 * GIB
    _settings(rig, monkeypatch, {"family_adapter": "minimax_h3"}, sniffed=detect.CheckpointSniffError("gguf"))
    rig.graphs["graph"]["2"]["inputs"]["unet_name"] = "h3.gguf"
    _guide(1)
    assert len(_Guide.calls) == 1


def test_a_24_gib_reserve_moves_the_line_and_the_words_state_it(rig, monkeypatch):
    """The floor is reserve_bytes of the Init node's arguments, as at the loader. The rescues run the step
    again against that one floor, and this process's free pool adds to the room whatever the reserve."""
    _settings(rig, monkeypatch, {"uma_reserve_gb": 24.0})
    reserve = 24 * GIB
    rig.box.value = reserve + 14 * GIB
    _guide(1)
    assert len(_Guide.calls) == 1
    rig.box.value = reserve + 14 * GIB - 1  # admitted at the default floor, under the reserve here
    with pytest.raises(driver_spend_price.DriverSpendCapacityError) as exc_info:
        _guide(1)
    assert "has 14.0 GiB above its 24.0 GiB floor" in _refused(exc_info) and len(_Guide.calls) == 1
    monkeypatch.setattr(driver_spend_price, "_pool_free_bytes", lambda: 2 * GIB)
    rig.box.value = reserve + 12 * GIB
    _guide(1)
    assert len(_Guide.calls) == 2

    def unload(mm, models):
        before, rig.box.value = rig.box.value, rig.box.value + 1 * GIB
        return driver_unload.Unloaded(("text encoder",), before, rig.box.value)

    monkeypatch.setattr(driver_unload, "comfy_model_management", lambda: object())
    monkeypatch.setattr(driver_unload, "held_models", lambda mm: [(object(), "text encoder")])
    monkeypatch.setattr(driver_unload, "unload", unload)
    rig.box.value = reserve + 10 * GIB  # the unload frees 1, the pool adds 2: 13 above the floor, short of 14
    with pytest.raises(driver_spend_price.DriverSpendCapacityError) as exc_info:
        _guide(1)
    assert "has 13.0 GiB above its 24.0 GiB floor" in _refused(exc_info) and len(_Guide.calls) == 2


@pytest.mark.parametrize(("init", "config_args", "attached", "floor"), [
    ({}, {"uma_reserve_gb": 24.0}, None, 24),  # the config file the Init node names, before a mesh is up
    ({"uma_reserve_gb": 30.0}, {"uma_reserve_gb": 24.0}, None, 30),  # the Init node's over the config's
    ({}, {"uma_reserve_gb": 9.0}, {"uma_reserve_gb": 24.0}, 24),  # the attached mesh's policy first
    ({"uma_reserve_gb": 2.0}, None, None, 5),  # never under the 5 GiB floor
    ({"uma_reserve_gb": ["40", 0]}, None, None, 5),  # a linked input is not known before it runs
])
def test_the_floor_reads_the_init_node_over_the_configured_policy(rig, monkeypatch, init, config_args, attached,
                                                                  floor):
    _settings(rig, monkeypatch, init, config_args=config_args, attached=attached)
    rig.box.value = 1 * GIB
    with pytest.raises(driver_spend_price.DriverSpendCapacityError) as exc_info:
        _guide(1)
    assert f"above its {floor}.0 GiB floor" in _refused(exc_info)


def test_a_config_that_cannot_be_read_leaves_the_init_nodes_reserve(rig, monkeypatch):
    _settings(rig, monkeypatch, {"uma_reserve_gb": 12.0})

    def unreadable(explicit=None):
        raise config.ClusterConfigError("no such file")

    monkeypatch.setattr(config, "find_config_path", unreadable)
    rig.box.value = 1 * GIB
    with pytest.raises(driver_spend_price.DriverSpendCapacityError) as exc_info:
        _guide(1)
    assert "above its 12.0 GiB floor" in _refused(exc_info)


def test_a_marked_spend_steps_aside_whatever_the_reserve(rig, monkeypatch):
    """A mark is the loader's charge for the term against the spend's own size; no reserve enters it."""
    _settings(rig, monkeypatch, {"uma_reserve_gb": 30.0})
    _mark(driver_spend_price.GUIDE_TERM, 14 * GIB)
    rig.box.value = 1 * GIB
    _guide(1)
    assert len(_Guide.calls) == 1 and rig.box.reads == 0


def test_the_inputs_read_are_the_init_nodes_own(monkeypatch):
    """A rename on the Init node would leave the wrap reading defaults; it reads these names."""
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_filename_list=lambda kind: []))
    from dgx_monarch.nodes.init import DGXMonarchInit
    from dgx_monarch.nodes.loaders import DGXMonarchUNETLoader

    types = DGXMonarchInit.INPUT_TYPES()
    names = {**types["required"], **types.get("optional", {})}
    for name in (driver_spend_settings.FAMILY_INPUT, driver_spend_settings.RESERVE_KEY, "config_path", "mode"):
        assert name in names, name
    assert "mesh" in DGXMonarchUNETLoader.INPUT_TYPES()["required"]
    assert DGXMonarchInit.RETURN_TYPES == DGXMonarchUNETLoader.INPUT_TYPES()["required"]["mesh"]
