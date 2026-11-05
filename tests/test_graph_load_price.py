"""Price every model a graph loads at the loader preflight, and lead with the reason.

The 2026-11-05 hardware leg ran the shipped p-h3-image template (BF16 DiT,
prompt upsampler on) on one Spark. Its loader preflight passed: the load window
charged the DiT and its host copy and nothing else, because an eager load peaks
before the driver builds its models. Under ``auto`` the load waits for the
first render, after the upsampler, the text encoder and the VAE are loaded, so
the Worker refused the stock load at the sampler, 2.0 GiB short, and the
prompt error read "Actor call dgxm_worker.sample failed." with the reason at
the end of a traceback. These tests replay that graph on paper with the
manifest's byte sizes and the leg's 110.8 GiB, and pin the sampler's wording.

CPU only: tiny files stand in for the models, and their sizes are stubbed.
"""
from __future__ import annotations

import json
import struct
import sys
import types
from pathlib import Path

import pytest

from dgx_monarch import consent_pending, consent_store, driver_footprint, mesh_safety
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.nodes import (
    fleet,
    graph_live,
    loader_graph,
    loader_preflight,
    loaders,
    refusal_lead,
    samplers,
)
from dgx_monarch.refusal import (
    PanelAction,
    RefusalClass,
    parse_leading_refusal_tag,
    parse_refusal_tag,
    refusal,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.sweep import convert  # noqa: E402

_GIB = 2 ** 30
DIT = "p-h3-image_bf16.safetensors"
DIT_INT8 = "p-h3-image_int8_convrot.safetensors"
DIT_W4A8 = "p-h3-image_w4a8.safetensors"
TE = "p-h3-image_text_encoder_bf16.safetensors"
UPSAMPLER = "p-h3-image_prompt_upsampler_bf16.safetensors"
UPSAMPLER_INT8 = "p-h3-image_prompt_upsampler_int8_convrot.safetensors"
UPSAMPLER_W4A8 = "p-h3-image_prompt_upsampler_w4a8.safetensors"
VAE = "p-h3-image_vae_bf16.safetensors"
# Byte sizes from example_workflows/artifacts.toml and, for the two upsampler
# quantizations no template names, the files the head holds.
SIZES = {
    ("diffusion_models", DIT): 40_468_411_160,
    ("diffusion_models", DIT_INT8): 21_213_068_840,
    ("diffusion_models", DIT_W4A8): 13_362_068_864,
    ("text_encoders", TE): 9_138_027_824,
    ("text_encoders", UPSAMPLER): 19_306_305_552,
    ("text_encoders", UPSAMPLER_INT8): 9_748_117_724,
    ("text_encoders", UPSAMPLER_W4A8): 6_724_096_116,
    ("vae", VAE): 5_559_036_680,
    # A local fp8 copy no graph here names, so the manifest does not list it.
    ("diffusion_models", "p-h3-image_fp8.safetensors"): 20_234_205_580,
    # Kromoss, a MOSS-Image Base fine-tune, for the LoRA route.
    ("diffusion_models", "kromoss_bf16.safetensors"): 10_164_655_808,
    ("diffusion_models", "kromoss_int8_convrot.safetensors"): 5_413_698_958,
    ("text_encoders", "qwen3.5_9b_bf16.safetensors"): 19_306_312_328,
    ("vae", "flux2-vae_small.safetensors"): 249_521_340,
    # An upsampler kept in a subfolder, beside only its INT8 copy.
    ("text_encoders", "sub/p-h3-image_prompt_upsampler_bf16.safetensors"): 19_306_305_552,
    ("text_encoders", "sub/p-h3-image_prompt_upsampler_int8_convrot.safetensors"): 9_748_117_724,
}
KINDS = {DIT: "bf16", DIT_INT8: "int8", DIT_W4A8: "w4a8", "p-h3-image_fp8.safetensors": "fp8",
         "kromoss_bf16.safetensors": "bf16", "kromoss_int8_convrot.safetensors": "int8"}
# Head MemAvailable at the cell 9 submit, 2026-11-05.
MEM_CELL9 = int(110.8 * _GIB)


def _write_safetensors(path: Path, keys: tuple[str, ...] = ("blocks.0.weight",)) -> None:
    """A real header over a few bytes: the twin checks read the names in it."""
    header = json.dumps({key: {"dtype": "BF16", "shape": [1], "data_offsets": [i * 2, i * 2 + 2]}
                         for i, key in enumerate(keys)}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * (2 * len(keys)))


def _template() -> dict:
    graph = json.loads((ROOT / "example_workflows" / "dgx-monarch-p-h3-image.json").read_text())
    return convert.convert(graph, convert.repo_widget_names(ROOT))


def _switch_id(graph: dict) -> str:
    (key,) = [key for key, node in graph.items() if node["class_type"] == "ComfySwitchNode"]
    return key


class _Handle:
    def __init__(self, hosts: tuple = ()):
        self.owns_hosts = bool(hosts)     # a local mesh: rank 0 is a child of the driver
        self.gpus_per_host = 1
        self.config = types.SimpleNamespace(hosts=hosts, worker_args={})

    def effective_worker_args(self, worker_args=None):
        return dict(worker_args or {})


class _Mesh:
    def __init__(self, preset: str = "auto", world: int = 1, hosts: tuple = (),
                 worker_args: dict | None = None):
        self.handle = _Handle(hosts)
        self.topology_preset = preset
        self.world = world
        self.worker_args = dict(worker_args or {})


def _pair(preset: str = "auto") -> _Mesh:
    """The head and the sibling; rank 0 is on this box, the driver's."""
    hosts = (types.SimpleNamespace(name="localhost"), types.SimpleNamespace(name="sibling.invalid"))
    return _Mesh(preset, world=2, hosts=hosts)


@pytest.fixture(autouse=True)
def _rig(monkeypatch, tmp_path):
    loader_preflight.reset_memos()
    consent_pending.clear_all()
    monkeypatch.delenv(driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV, raising=False)
    monkeypatch.setattr(consent_store, "MEMO_PATH", str(tmp_path / "consent_memo.json"))
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: MEM_CELL9)
    sizes: dict[str, int] = {}
    for (folder, name), size in SIZES.items():
        target = tmp_path / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_safetensors(target)
        sizes[str(target)] = size
    monkeypatch.setattr(driver_footprint, "file_size_bytes", lambda path: sizes.get(str(path), 0))
    fp = types.ModuleType("folder_paths")
    fp.get_full_path = lambda folder, name: (  # type: ignore[attr-defined]
        str(tmp_path / folder / name) if (folder, name) in SIZES else None)
    fp.get_filename_list = lambda folder: sorted(  # type: ignore[attr-defined]
        name for (where, name) in SIZES if where == folder)
    (tmp_path / "output").mkdir()
    fp.get_output_directory = lambda: str(tmp_path / "output")  # type: ignore[attr-defined]
    fp.get_annotated_filepath = lambda name: str(tmp_path / "input" / name)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    from dgx_monarch.adapters import detect

    monkeypatch.setattr(detect, "sniff_checkpoint", lambda path: (
        "moss_image" if Path(path).name.startswith("kromoss") else "minimax_h3",
        KINDS[Path(path).name]))
    yield
    loader_preflight.reset_memos()
    consent_pending.clear_all()


def _refuse(mesh, graph) -> str:
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(mesh, DIT, {}, graph)
    return str(raised.value)


# --- one rank: the shipped template refuses before any load ----------------

def test_one_rank_with_the_upsampler_on_refuses_before_any_load_and_names_what_fits():
    """Slab residency places no host copy and fits, so the rescue card is offered too."""
    graph = _template()
    text = _refuse(_Mesh(), graph)
    assert text.startswith(
        "[dgxm:C guard=loader_footprint_preflight waivable=1] p-h3-image_bf16.safetensors "
        "needs 119.8 GiB on this box and 110.8 GiB is available, 9.0 GiB short. It loads "
        f"at the first render, beside {TE}, {UPSAMPLER} and {VAE}, which this graph loads "
        "on the driver first. Nothing was loaded.")
    assert (f"{DIT_INT8} (19.8 GiB) or {DIT_W4A8} (12.4 GiB) in place of {DIT}") in text
    assert (f"{UPSAMPLER_INT8} (9.1 GiB) or {UPSAMPLER_W4A8} (6.3 GiB) in place of "
            f"{UPSAMPLER}") in text
    assert (f"If/Else Switch (node {_switch_id(graph)}) set to false would fit too: the "
            f"graph then does not load {UPSAMPLER}.") in text
    assert ("On a pair of Sparks with the Init node's topology set to uly2+fsdp (auto never "
            "picks it), the head would hold 20.0 GiB of the DiT and the load fits") in text
    assert "Identity Gate PASS on uly2+fsdp first" in text
    assert "Refuses p-h3-image_bf16.safetensors at the load window" in text
    assert "35.6 GiB for the driver models this graph loads before its first render" in text
    assert parse_refusal_tag(text) is not None, "exactly one tag, or no reader finds the class"
    assert "Load with slab residency" in text
    assert [card["kind"] for card in consent_pending.pending_cards()] == ["rescue-slab"]
    assert len(text) < 4000


def test_the_loader_node_refuses_before_the_fleet_is_touched(monkeypatch):
    def touched(*_args, **_kwargs):
        raise AssertionError("the fleet was touched before the refusal")

    monkeypatch.setattr(loaders, "ensure_live", touched)
    monkeypatch.setattr(loader_preflight, "preflight_upstream_gated_artifact", lambda _name: None)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loaders.DGXMonarchUNETLoader().load(_Mesh(), DIT, prompt=_template())
    assert str(raised.value).startswith("[dgxm:C guard=loader_footprint_preflight")


def test_the_upsampler_switched_off_is_not_charged_and_the_load_is_admitted():
    graph = _template()
    graph[_switch_id(graph)]["inputs"]["switch"] = False
    terms = loader_graph.driver_stack_terms(graph, family="minimax_h3")
    assert terms.te_artifacts == (TE,)
    assert terms.te_bytes == int(SIZES[("text_encoders", TE)] * loader_graph.TE_RESIDENT_RATIO)
    assert loader_preflight.preflight_loader_footprint(_Mesh(), DIT, {}, graph) is None


def test_only_the_twins_that_fit_are_named(monkeypatch):
    """At 75 GiB the INT8 DiT's load (82.1 GiB) does not fit and the W4A8's does."""
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 75 * _GIB)
    text = _refuse(_Mesh(), _template())
    assert f"{DIT_W4A8} (12.4 GiB) in place of {DIT}" in text
    assert DIT_INT8 not in text and UPSAMPLER_INT8 not in text and "If/Else Switch" not in text


def test_an_eager_preset_keeps_the_driver_models_out_of_the_load_window():
    """An explicit preset loads at the loader node, before the driver builds its
    models, so its load window stays the DiT and its host copy."""
    assert loader_preflight.preflight_loader_footprint(
        _Mesh("single"), DIT, {}, _template()) is None


def test_a_cast_dit_is_priced_at_its_copy_beside_the_graph(monkeypatch):
    """A cast to fp8 builds a copy half the file; the Worker's stock rung prices
    that copy beside the host copy at the file's own width (#142), and so do the
    loader's windows. No twin or shard build is named for a cast, and no slab card."""
    from dgx_monarch import capacity_fit

    monkeypatch.setattr(capacity_fit, "cast_resident_bytes",
                        lambda path, dtype, *, lower=False: SIZES[("diffusion_models", DIT)] // 2)
    fp8 = {"weight_dtype": "fp8_e4m3fn"}
    # 18.8 + 41.5 + 35.6 + 5.0 = 100.9 GiB fits the cell 9 box.
    assert loader_preflight.preflight_loader_footprint(_Mesh(), DIT, fp8, _template()) is None
    loader_preflight.reset_memos()
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 95 * _GIB)
    with pytest.raises(driver_footprint.DriverFootprintCapacityError) as raised:
        loader_preflight.preflight_loader_footprint(_Mesh(), DIT, fp8, _template())
    text = str(raised.value)
    assert "needs 100.9 GiB on this box and 95.0 GiB is available, 5.9 GiB short" in text
    assert "cast to a 18.8 GiB copy of the 37.7 GiB file" in text
    assert DIT_INT8 not in text and "uly2+fsdp" not in text
    assert f"{UPSAMPLER_W4A8} (6.3 GiB) in place of {UPSAMPLER}" in text
    assert consent_pending.pending_cards() == []


# --- the pair: Ulysses holds the whole DiT on this box, a shard build half --

def test_on_the_pair_auto_holds_the_whole_dit_here_and_refuses_the_same():
    text = _refuse(_pair(), _template())
    assert "needs 119.8 GiB on the head and 110.8 GiB is available, 9.0 GiB short" in text
    assert "topology set to uly2+fsdp (auto never picks it)" in text


def test_on_the_pair_a_declared_shard_build_fits_beside_the_upsampler():
    assert loader_preflight.preflight_loader_footprint(
        _pair("uly2+fsdp"), DIT, {}, _template()) is None


# --- which nodes run --------------------------------------------------------

def _switch_graph(choice: object, *, on_true: bool = True) -> dict:
    graph = {
        "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": UPSAMPLER}},
        "2": {"class_type": "TextGenerateV2", "inputs": {"clip": ["1", 0], "prompt": "x"}},
        "3": {"class_type": "PrimitiveString", "inputs": {"value": "x"}},
        "4": {"class_type": "ComfySwitchNode",
              "inputs": {"switch": choice, "on_false": ["3", 0]}},
        "5": {"class_type": "PreviewAny", "inputs": {"source": ["4", 0]}},
    }
    if on_true:
        graph["4"]["inputs"]["on_true"] = ["2", 0]
    return graph


@pytest.mark.parametrize("choice,live", [
    (True, {"1", "2", "3", "4", "5"} - {"3"}),
    (False, {"3", "4", "5"}),
    (["9", 0], {"1", "2", "3", "4", "5"}),      # a linked switch: either branch may run
])
def test_a_literal_switch_runs_only_the_branch_it_selects(choice, live):
    assert graph_live.live_nodes(_switch_graph(choice)) == frozenset(live)


def test_a_switch_with_one_branch_open_keeps_the_other():
    """With nothing on on_true the switch runs on_false; the dangling Generate
    Text is a sink, and a sink is kept, since the walk cannot tell an output
    node from a node nothing reads."""
    live = graph_live.live_nodes(_switch_graph(True, on_true=False))
    assert {"3", "4", "5"} <= live and "2" in live


def test_a_stock_unet_and_a_checkpoint_on_the_driver_are_priced(monkeypatch, tmp_path):
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": DIT_W4A8}},
        "2": {"class_type": "DGXMonarchUNETLoader", "inputs": {"unet_name": DIT}},
        "3": {"class_type": "Sink", "inputs": {"a": ["1", 0], "b": ["2", 0]}},
    }
    terms = loader_graph.driver_stack_terms(graph)
    assert terms.model_artifacts == (DIT_W4A8,)
    assert terms.model_bytes == SIZES[("diffusion_models", DIT_W4A8)]


# --- sampler refusals lead the prompt error --------------------------------

def _ring_pad_text() -> str:
    return refusal(
        RefusalClass.KNOWN_WRONG,
        "minimax_h3: this packed sequence needs 1 divisibility pad row(s), and the exact pad "
        "exclusion is Ulysses-only.",
        guard="ring_pad:minimax_h3", waivable=True,
        panel_action=PanelAction(label="Waive for this render", env="DGXM_WAIVE_KNOWN_WRONG",
                                 env_value="ring_pad"),
        troubleshooting=49)


def _actor_error(inner: Exception) -> Exception:
    from monarch.actor import ActorError

    return ActorError(inner, "Actor call dgxm_worker.sample failed.")


def test_a_class_k_sampler_refusal_leads_the_prompt_error(monkeypatch):
    wrapper = _actor_error(UnsupportedModelError(_ring_pad_text()))

    def refused(*_args, **_kwargs):
        raise wrapper

    monkeypatch.setattr(samplers, "run_render", refused)
    import torch

    guider = {"model": object(), "spec": {}, "cfg": 1.0}
    with pytest.raises(UnsupportedModelError) as raised:
        samplers.DGXMonarchSamplerCustom().sample(
            types.SimpleNamespace(seed=42), guider, object(), torch.tensor([1.0, 0.0]),
            {"samples": torch.zeros(1, 16, 1, 8, 8)})
    text = str(raised.value)
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.KNOWN_WRONG
    assert 'Open the DGX Monarch panel and click "Waive for this render"' in text
    assert "Actor call" not in text and parse_refusal_tag(text) is not None
    assert raised.value.__cause__ is wrapper, "the remote traceback stays in the chain"


_LEAD = refusal_lead.lead_with_refusal(lambda: None).__code__


@pytest.mark.parametrize("node,method", [
    (samplers.DGXMonarchKSampler, "sample"),
    (samplers.DGXMonarchKSamplerPipeline, "sample"),
    (samplers.DGXMonarchKSamplerAdvanced, "sample"),
    (samplers.DGXMonarchSamplerCustom, "sample"),
    (fleet.DGXMonarchFleetKSampler, "fleet"),
    (loaders.DGXMonarchUNETLoader, "load"),
    (loaders.DGXMonarchUncondUNETLoader, "load"),
])
def test_every_node_that_meets_a_worker_refusal_leads_with_it(node, method):
    assert node.FUNCTION == method
    assert getattr(node, method).__code__ is _LEAD


def test_an_untagged_worker_crash_keeps_its_wrapper():
    crash = _actor_error(RuntimeError("CUDA error while loading models/[dgxm:P]x.safetensors"))
    assert refusal_lead.surfaced(crash) is None


# --- forced stock says why, and names the gate ------------------------------

def _worker_stock_refusal(monkeypatch) -> str:
    """The Worker's own words for a stock load that arrived with slab_weights off."""
    from dgx_monarch import capacity_fit
    from dgx_monarch.actor import rescue_offer, store_residency, worker_env

    monkeypatch.setattr(store_residency, "_host", lambda: "head")
    blocker = rescue_offer.rescue_blocker(
        slab_capable_path=True, lora_low_rss=False, fsdp_launch=False, compile_dit=False,
        explicit_stock=True, blocked_reason=worker_env.slab_mode_reason({}, {}))
    fit = capacity_fit.StockFit(False, True, SIZES[("diffusion_models", DIT)], int(82.1 * _GIB))
    return refusal(RefusalClass.CAPACITY, store_residency._no_rescue_text(DIT, fit, blocker),
                   guard="stock_load_preflight", waivable=False, troubleshooting=53)


def test_the_worker_still_words_explicit_stock_as_the_driver_matches_it(monkeypatch):
    text = _worker_stock_refusal(monkeypatch)
    assert refusal_lead.SLAB_OFF_BLOCKER in text
    assert refusal_lead.EXPLICIT_STOCK_OPTIONS in text


def test_a_forced_stock_refusal_names_the_gate_not_the_setting(monkeypatch):
    wrapper = _actor_error(mesh_safety.StockLoadCapacityError(_worker_stock_refusal(monkeypatch)))

    @refusal_lead.lead_with_refusal
    def render():
        refusal_lead.note_forced_stock("stale")
        raise wrapper

    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        render()
    text = str(raised.value)
    assert text.startswith("[dgxm:C guard=stock_load_preflight waivable=0] stock residency "
                           "cannot load p-h3-image_bf16.safetensors on head: the load needs 84.1 GiB")
    assert ("Slab residency was not used, although the Init node sets slab_weights=on: this "
            "combination has no Identity Gate PASS for its current state (stale), so the "
            "driver ran this render on stock residency.") in text
    assert ("What would fit: run Identity Gate (DGX Monarch) on this model with "
            "slab_weights=on in a graph whose stock reference fits the box that refused, for "
            "example with "
            "an optional driver model such as a prompt upsampler switched off") in text
    assert "set slab_weights=on" not in text and "effective worker args" not in text
    assert parse_refusal_tag(text) is not None
    assert refusal_lead.forced_stock_state() is None, "the note ends with the node call"


def test_a_forced_stock_cast_never_names_the_gate(monkeypatch):
    """Under a cast a slab saves nothing (#142), so the Worker names neither
    slab_weights=on nor the gate, and the driver's rewording leaves it be."""
    from dgx_monarch import capacity_fit
    from dgx_monarch.actor import rescue_offer, store_residency, worker_env

    monkeypatch.setattr(store_residency, "_host", lambda: "head")
    skipped = capacity_fit.SlabFit(True, False, 0, None, 0, capacity_fit.SKIP_SLAB_DTYPE_CAST)
    blocker = rescue_offer.rescue_blocker(
        slab_capable_path=True, lora_low_rss=False, fsdp_launch=False, compile_dit=False,
        explicit_stock=True, blocked_reason=worker_env.slab_mode_reason({}, {}),
        slab_fit=skipped)
    fit = capacity_fit.StockFit(False, True, SIZES[("diffusion_models", DIT)], int(82.1 * _GIB))
    text = refusal(RefusalClass.CAPACITY, store_residency._no_rescue_text(DIT, fit, blocker),
                   guard="stock_load_preflight", waivable=False, troubleshooting=53)
    assert refusal_lead.forced_stock_text(text, "stale") == text
    assert "Identity Gate" not in text and "set slab_weights=on" not in text


def test_without_the_note_the_worker_sentence_is_only_unwrapped(monkeypatch):
    inner = _worker_stock_refusal(monkeypatch)

    @refusal_lead.lead_with_refusal
    def render():
        raise _actor_error(mesh_safety.StockLoadCapacityError(inner))

    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        render()
    assert str(raised.value) == inner


@pytest.mark.parametrize("asked,noted", [(True, "stale"), ("auto", None), (False, None)])
def test_the_driver_notes_forced_stock_only_where_slab_was_asked_for(monkeypatch, asked, noted):
    from dgx_monarch.nodes import consent_projection, consent_rescue, gate_inconclusive

    monkeypatch.setattr(consent_rescue, "render_consent_wire", lambda _model: None)
    monkeypatch.setattr(gate_inconclusive, "dispatch_grants_skip", lambda *_a: False)
    seen: list = []

    @refusal_lead.lead_with_refusal
    def authorize():
        result = consent_projection.capacity_consent_authorization(
            object(), {"slab_weights": asked}, {"loras": []}, {}, "combo", {},
            types.SimpleNamespace(state="stale"), ("token",), None)
        seen.append(refusal_lead.forced_stock_state())
        return result

    assert authorize() is None
    assert seen == [noted]
