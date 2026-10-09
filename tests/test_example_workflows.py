"""Lint for every template in example_workflows/. CPU-only: it never runs a
graph or touches hardware.

Each file must be valid JSON with a complete UI-format envelope and a unique
root ID that can key its widgets in Nodes 2, place only known node classes,
wire only nodes that exist and record each link on both endpoints, carry a
named widget map on every DGX Monarch node, and store a randomize control after
every seed. A graph that places a DGX Monarch node holds exactly one Init, and
a shipped file also ships its preview card. Later checks cover the
docs/MODELS.md row map, the docs/QUICKSTART.md claims, pinned Init topologies,
the testing-only LoRA chain, chroma prompt token counts, the Note sentences
that must match an adapter, and the Wan-Animate 2 motion and residency
settings.

ComfyUI's template browser serves this folder as-is, so a broken file reaches a
user as a broken template with no error anywhere earlier. tools/gen_templates.py
writes these files and tests/test_surface_remediation.py pins them to it; this
lint states what a template must satisfy however it was written.

tests/fixtures/workflows/generated/ holds the sweep harness's graphs. These are outside
the template browser and need no preview card or docs/MODELS.md row. Every structural check runs over both
sets; the card, row and pinned-topology checks stay on the shipped set.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_COUNTER_SPEC = importlib.util.spec_from_file_location(
    "dgxm_tools_count_t5_tokens", REPO / "tools" / "count_t5_tokens.py"
)
assert _COUNTER_SPEC and _COUNTER_SPEC.loader
count_t5_tokens = importlib.util.module_from_spec(_COUNTER_SPEC)
sys.modules[_COUNTER_SPEC.name] = count_t5_tokens
_COUNTER_SPEC.loader.exec_module(count_t5_tokens)
TEMPLATE_DIR = REPO / "example_workflows"
TESTING_DIR = REPO / "tests" / "fixtures" / "workflows" / "generated"
TEMPLATES = sorted(TEMPLATE_DIR.glob("*.json"))
TESTING_TEMPLATES = sorted(TESTING_DIR.glob("*.json"))
EVERY_TEMPLATE = TEMPLATES + TESTING_TEMPLATES

# The frontend's LGraph.serialize writes these ten keys for every saved graph.
# The frontend's 0.4 schema requires last_node_id, last_link_id, nodes, links and
# version and accepts the rest as optional (comfyui_frontend_package 1.53.6,
# checked 2026-10-07); a template carries all ten, as a saved graph does.
_ENVELOPE_KEYS = frozenset({
    "id", "revision", "last_node_id", "last_link_id",
    "nodes", "links", "groups", "config", "extra", "version",
})

# Every node class a template, shipped or testing-only, may place: dgx-monarch's
# own nodes plus the stock ComfyUI nodes their families need. Extend it in the
# commit that adds the template. It is the templates' allowlist, not a node
# registry.
_KNOWN_NODE_TYPES = frozenset({
    "DGXMonarchInit",
    "DGXMonarchUNETLoader",
    "DGXMonarchUncondUNETLoader",
    "DGXMonarchLoraLoader",
    "DGXMonarchModelSamplingSD3",
    "DGXMonarchKSampler",
    "DGXMonarchKSamplerAdvanced",
    "DGXMonarchFleetKSampler",
    "DGXMonarchIdentityGate",
    "DGXMonarchClearVRAM",
    "DGXMonarchCFGGuider",
    "DGXMonarchDualModelGuider",
    "DGXMonarchSamplerCustom",
    "BerniniConditioning",
    "CLIPLoader",
    "DualCLIPLoader",
    "CLIPTextEncode",
    "TextEncodeMageFlowEdit",
    "TextEncodeQwenImage21",
    "QwenImage21Cache",
    "CLIPTextEncodeKandinsky5",
    "CLIPVisionEncode",
    "CLIPVisionLoader",
    "ConditioningZeroOut",
    "FluxGuidance",
    "InstructPixToPixConditioning",
    "PiDConditioning",
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyChromaRadianceLatentImage",
    "EmptyFlux2LatentImage",
    "EmptyHunyuanImageLatent",
    "EmptyHunyuanLatentVideo",
    "EmptyHunyuanVideo15Latent",
    "EmptyLTXVLatentVideo",
    "EmptyMiniMaxH3LatentAV",
    "HunyuanRefinerLatent",
    "GetVideoComponents",
    "GetImageSize",
    "ImageFromBatch",
    "ImageScale",
    "ImageToMask",
    "LatentCutToBatch",
    "LatentUpscaleModelLoader",
    "MaskToImage",
    "MiniMaxH3AddGuide",
    "Ideogram4Scheduler",
    "KSamplerSelect",
    "LoadAudio",
    "LoadImage",
    "LoadImageMask",
    "JoinImageWithAlpha",
    "LoadVideo",
    "LTXVAddGuide",
    "LTXVAudioVAEDecode",
    "LTXVConcatAVLatent",
    "LTXVConditioning",
    "LTXVCropGuides",
    "LTXVEmptyLatentAudio",
    "LTXVImgToVideo",
    "LTXVLatentUpsampler",
    "LTXVScheduler",
    "LTXVSeparateAVLatent",
    "ManualSigmas",
    "MiniMaxH3ImageToVideo",
    "RandomNoise",
    "VAELoader",
    "VAEDecode",
    "VAEDecodeAudio",
    "VAEDecodeTiled",
    "VAEEncode",
    "WanDancerEncodeAudio",
    "WanDancerPadKeyframesList",
    "WanDancerVideo",
    "WanImageToVideo",
    "WanAnimate2ToVideo",
    "WanSCAILToVideo",
    "TrimVideoLatent",
    "SaveImage",
    "SaveWEBM",
    "CreateVideo",
    "SaveVideo",
    "SaveAudioAdvanced",
    "Note",
})

_IDS = [path.name for path in TEMPLATES]
_EVERY_ID = [str(path.relative_to(REPO)) for path in EVERY_TEMPLATE]

# The testing-only set names its LoRA variants with this suffix, and only those
# carry the placeholder LoRA. dgx-monarch-test-wan-scail2 and
# dgx-monarch-test-wan-animate2-int8 keep the LoRA of the shipped graph they
# copy, because that LoRA is part of the shipped recipe (tools/gen_templates.py
# says so in the scail2 purpose line and the Wan-Animate 2 Note).
_LORA_VARIANT_SUFFIX = "-lora.json"
_TEST_LORA_NAME = "test_lora_placeholder.safetensors"
_TEST_LORA_STRENGTH = 0.8


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_the_template_folder_ships_templates():
    # The checks parametrized over the glob pass on an empty folder, so this
    # test fails on it by name.
    assert TEMPLATES, f"no templates found under {TEMPLATE_DIR}"


def test_the_testing_only_folder_ships_templates():
    # The checks parametrized over EVERY_TEMPLATE pass on an empty testing-only
    # set, so this test fails on it by name.
    assert TESTING_TEMPLATES, f"no templates found under {TESTING_DIR}"


@pytest.mark.parametrize("path", TEMPLATES, ids=_IDS)
def test_every_template_ships_its_preview_card(path: Path):
    # The browser builds an extension template's tile image from the workflow
    # name with the extension hardcoded to jpg, so a template with no sibling
    # card gets a 404 and the frontend's generic placeholder tile, with no error
    # the packager can see (comfyui_frontend_package 1.53.6, checked
    # 2026-10-07). MANIFEST.in ships *.jpg for this.
    card = path.with_suffix(".jpg")
    assert card.exists(), f"{path.name}: no preview card at {card.name}"
    with open(card, "rb") as fh:
        head = fh.read(3)
    assert head == b"\xff\xd8\xff", f"{card.name}: not a JPEG (starts {head!r})"


@pytest.mark.parametrize("path", EVERY_TEMPLATE, ids=_EVERY_ID)
def test_template_is_json_with_the_ui_envelope(path: Path):
    document = _load(path)
    missing = _ENVELOPE_KEYS - set(document)
    assert not missing, f"{path.name}: envelope is missing {sorted(missing)}"
    assert isinstance(document["nodes"], list) and document["nodes"]
    assert isinstance(document["links"], list)


def test_every_template_id_can_key_its_widgets_in_nodes2():
    """Keep root graph IDs compatible with ComfyUI frontend 1.53's WidgetId grammar.

    Nodes 2 stores a widget as ``graphId:nodeId:name`` and accepts exactly
    three nonempty colon-delimited fields.  A colon in the workflow's legacy
    0.4 root ID makes every widget key invalid, so the browser silently omits
    all widgets, including stock ones.
    """
    workflow_ids = []
    for path in EVERY_TEMPLATE:
        document = _load(path)
        workflow_id = document["id"]
        assert isinstance(workflow_id, str) and workflow_id, (
            f"{path.name}: empty workflow ID cannot scope Nodes 2 widgets"
        )
        workflow_ids.append(workflow_id)
        for node in document["nodes"]:
            for index, _value in enumerate(node.get("widgets_values") or []):
                widget_key = f"{workflow_id}:{node['id']}:widget-{index}"
                assert len(widget_key.split(":")) == 3 and all(
                    widget_key.split(":")
                ), (
                    f"{path.name}: root ID {workflow_id!r} makes Nodes 2 "
                    f"widget {node['id']!r}/{index} unkeyable"
                )
    assert len(workflow_ids) == len(set(workflow_ids)), (
        "workflow IDs must be unique so Nodes 2 widget state cannot cross "
        "between templates"
    )


@pytest.mark.parametrize("path", EVERY_TEMPLATE, ids=_EVERY_ID)
def test_template_places_only_known_node_classes(path: Path):
    for node in _load(path)["nodes"]:
        assert "type" in node, f"{path.name}: node {node.get('id')!r} has no type"
        assert node["type"] in _KNOWN_NODE_TYPES, (
            f"{path.name}: node {node['id']!r} places unrecognized class "
            f"{node['type']!r}; add it here in the commit that adds the template"
        )


@pytest.mark.parametrize("path", EVERY_TEMPLATE, ids=_EVERY_ID)
def test_monarch_nodes_carry_a_named_widget_map(path: Path):
    # widgets_values is positional: inserting a widget shifts every value a
    # template saved after it. On load the browser extension prefers the name
    # map, so a template without one loads shifted values after the next
    # insertion (tests/test_init_widget_order.py). One name per positional
    # value: an all-connection node like the custom sampler carries an empty
    # map, and a widget-bearing node that lost its map reads as zero names for
    # N values here.
    for node in _load(path)["nodes"]:
        if not node["type"].startswith("DGXMonarch"):
            continue
        where = f"{path.name}: node {node['id']!r} ({node['type']})"
        assert isinstance(node.get("widgets_values"), list), (
            f"{where} has no positional widgets_values"
        )
        named = node.get("dgxm_widgets")
        assert isinstance(named, dict), (
            f"{where} has no dgxm_widgets name->value map"
        )
        assert len(named) == len(node["widgets_values"]), (
            f"{where}: dgxm_widgets carries {len(named)} names for "
            f"{len(node['widgets_values'])} positional values"
        )


# The seed widget of each node shape a template can place, and the value the
# frontend stores for a seed that advances. Extend it in the commit that places
# a new seed-bearing class.
_SEED_WIDGETS = {
    "DGXMonarchKSampler": "seed",
    "DGXMonarchKSamplerAdvanced": "noise_seed",
    "DGXMonarchFleetKSampler": "noise_seed",
    "DGXMonarchIdentityGate": "noise_seed",
    "RandomNoise": "noise_seed",
}
_CONTROL_WIDGET = "control_after_generate"
_CONTROL_VALUE = "randomize"


@pytest.mark.parametrize("path", EVERY_TEMPLATE, ids=_EVERY_ID)
def test_every_seed_ships_a_randomize_control(path: Path):
    # The frontend gives a seed its own control widget and stores that control
    # in the next positional slot: it attaches one to any INT input asking for
    # it and, when the input does not say, to any INT named seed or noise_seed
    # (comfyui_frontend_package 1.53.6, useIntWidget.ts, checked 2026-10-07),
    # creates it directly after the seed widget, and serializes node.widgets in order. A template
    # that leaves the slot out hands the control the next widget's value, which
    # is no control name, so the frontend falls back to `fixed`: press Queue a
    # second time and the same graph comes back out of the execution cache with
    # no image and no error. The seed values stay as shipped, so a first render
    # still reproduces.
    for node in _load(path)["nodes"]:
        seed = _SEED_WIDGETS.get(node["type"])
        if seed is None:
            continue
        where = f"{path.name}: node {node['id']!r} ({node['type']})"
        values = node["widgets_values"]
        named = node.get("dgxm_widgets")
        if isinstance(named, dict):
            names = list(named)
            assert seed in names, f"{where}: no {seed} widget in dgxm_widgets"
            slot = names.index(seed) + 1
            assert names[slot:slot + 1] == [_CONTROL_WIDGET], (
                f"{where}: {seed} is followed by {names[slot:slot + 1]}, not "
                f"the {_CONTROL_WIDGET} the frontend stores there"
            )
            assert named[_CONTROL_WIDGET] == _CONTROL_VALUE, (
                f"{where}: dgxm_widgets holds "
                f"{named[_CONTROL_WIDGET]!r} for {_CONTROL_WIDGET}"
            )
        else:
            # Stock nodes carry no name map; RandomNoise is seed then control.
            slot = 1
            assert len(values) == 2 and isinstance(values[0], int), (
                f"{where}: expected a seed and its control, got {values!r}"
            )
        assert values[slot] == _CONTROL_VALUE, (
            f"{where}: positional slot {slot} holds {values[slot]!r}; a seed "
            f"that does not store {_CONTROL_VALUE!r} loads as fixed and makes "
            "a second Queue press a silent no-op"
        )


@pytest.mark.parametrize("path", EVERY_TEMPLATE, ids=_EVERY_ID)
def test_every_template_holds_a_seed_the_pin_covers(path: Path):
    # A template with no seed the check above recognizes is either a graph that
    # renders nothing or a new sampler shape missing from _SEED_WIDGETS; the
    # second would pass the check above while shipping the fixed-seed bug.
    types = [node["type"] for node in _load(path)["nodes"]]
    assert any(name in _SEED_WIDGETS for name in types), (
        f"{path.name}: no node in _SEED_WIDGETS; add the sampler shape it "
        "places, with the seed widget name its class declares"
    )


@pytest.mark.parametrize("path", EVERY_TEMPLATE, ids=_EVERY_ID)
def test_every_link_joins_nodes_that_exist(path: Path):
    document = _load(path)
    node_ids = {node["id"] for node in document["nodes"]}
    for link in document["links"]:
        link_id, origin, _origin_slot, target, _target_slot, _kind = link
        assert origin in node_ids, (
            f"{path.name}: link {link_id} starts at node {origin!r}, which is not "
            "in the file"
        )
        assert target in node_ids, (
            f"{path.name}: link {link_id} ends at node {target!r}, which is not "
            "in the file"
        )


@pytest.mark.parametrize("path", EVERY_TEMPLATE, ids=_EVERY_ID)
def test_every_link_id_is_recorded_on_both_endpoints(path: Path):
    # Litegraph reads the wire from three places; a link listed in only one of
    # them draws a graph that runs differently from the one on screen.
    document = _load(path)
    by_id = {node["id"]: node for node in document["nodes"]}
    for link in document["links"]:
        link_id, origin, origin_slot, target, target_slot, _kind = link
        assert link_id in by_id[origin]["outputs"][origin_slot]["links"], (
            f"{path.name}: link {link_id} is missing from node {origin!r} output "
            f"slot {origin_slot}"
        )
        assert by_id[target]["inputs"][target_slot]["link"] == link_id, (
            f"{path.name}: node {target!r} input slot {target_slot} does not "
            f"point at link {link_id}"
        )


@pytest.mark.parametrize("path", EVERY_TEMPLATE, ids=_EVERY_ID)
def test_a_monarch_graph_holds_exactly_one_init(path: Path):
    # Every mesh node takes its handle from Init, and two Init nodes in one
    # graph would ask for two meshes on hardware that has one.
    types = [node["type"] for node in _load(path)["nodes"]]
    if not any(name.startswith("DGXMonarch") for name in types):
        pytest.skip("stock-only template")
    assert types.count("DGXMonarchInit") == 1, (
        f"{path.name}: expected exactly one DGXMonarchInit, found "
        f"{types.count('DGXMonarchInit')}"
    )


MODELS = REPO / "docs" / "MODELS.md"

# One entry per support-matrix row in docs/MODELS.md, keyed by the row's model
# cell: the template that opens that family, or None for a row that carries a
# dated no-template note instead. Extend it in the commit that adds a template.
# A MODELS.md row with no entry here fails below, so no supported family ships
# without a template or a dated note.
TEMPLATE_FOR_ROW: dict[str, str | None] = {
    # image
    "Krea2 (RAW+Turbo)": "dgx-monarch-krea2-t2i.json",
    "Chroma": "dgx-monarch-chroma-t2i.json",
    "Radiance": "dgx-monarch-radiance-t2i.json",
    "Ideogram4": "dgx-monarch-ideogram4-t2i.json",
    "Flux 1 Dev": "dgx-monarch-flux1-t2i.json",
    "Flux 1 Schnell": "dgx-monarch-flux1-t2i.json",
    "Flux2": "dgx-monarch-flux2-t2i.json",
    "LongCat-Image": "dgx-monarch-longcat-t2i.json",
    "HunyuanImage 2.1 (+refiner)": "dgx-monarch-hunyuan-image.json",
    "Qwen-Image": "dgx-monarch-qwen-image-t2i.json",
    "Mage-Flow (T2I/Edit, quality + Turbo)": "dgx-monarch-mage-flow-t2i.json",
    "Qwen Image 2.1": "dgx-monarch-qwen-image21-t2i.json",
    "Ernie-Image": "dgx-monarch-ernie-t2i.json",
    "Z-Image latent": "dgx-monarch-zimage-t2i.json",
    "Z-Image DCT PixelSpace": "dgx-monarch-zimage-dct-t2i.json",
    "Lens": "dgx-monarch-lens-t2i.json",
    "Omnigen2": "dgx-monarch-omnigen2-t2i.json",
    "Anima": "dgx-monarch-anima-t2i.json",
    "Boogu": "dgx-monarch-boogu-t2i.json",
    # One row, two templates: the row maps to the t2i graph, and
    # dgx-monarch-pid-4k ships beside it unmapped, as the extra LTX 2.5
    # image-conditioning templates do.
    "PixelDiT / PiD": "dgx-monarch-pixeldit-t2i.json",
    "Kandinsky5-Image": "dgx-monarch-kandinsky5-image.json",
    # video
    "LTX 2.3 (t2v)": "dgx-monarch-ltx-t2v.json",
    "LTX 2.3 (i2v/AV packed)": "dgx-monarch-ltx-i2v-av.json",
    "LTX 2.5 (t2v/AV packed)": "dgx-monarch-ltx25-t2v.json",
    "Wan 2.1 (t2v)": "dgx-monarch-wan-t2v.json",
    "Wan 2.2 (t2v, high/low)": "dgx-monarch-wan22-t2v.json",
    "Wan 2.2 (i2v)": "dgx-monarch-wan22-i2v.json",
    "Wan 2.1 (i2v)": "dgx-monarch-wan21-i2v.json",
    "Wan FlowRVS": "dgx-monarch-wan-flowrvs.json",
    "Wan Bernini-R": "dgx-monarch-wan-bernini.json",
    "Wan SCAIL Preview": "dgx-monarch-wan-scail.json",
    "Wan SCAIL2": "dgx-monarch-wan-scail2.json",
    "WanDancer": "dgx-monarch-wandancer.json",
    "Wan-Animate 2": "dgx-monarch-wan-animate2.json",
    "HunyuanVideo 1.5 (+SR)": "dgx-monarch-hunyuan-video.json",
    "CogVideoX 1.5 (t2v)": "dgx-monarch-cogvideox-t2v.json",
    "CogVideoX 1.5 (i2v)": "dgx-monarch-cogvideox-i2v.json",
    "CogVideoX 1.5 (inpaint)": None,
    "Kandinsky5-Video Lite": "dgx-monarch-kandinsky5-video-lite.json",
    "Kandinsky5-Video Pro": "dgx-monarch-kandinsky5-video-pro.json",
    "MiniMax H3 (t2va/fl2va/ref2va AV)": "dgx-monarch-minimax-h3-t2va.json",
}

# Every row without a template must explicitly state that gap.
_NO_TEMPLATE = re.compile(r"\bno template yet\b", re.IGNORECASE)


def _matrix_rows() -> dict[str, str]:
    """{model cell: notes cell} for every support-matrix row. A matrix row is
    the eight-column shape (model, five modes, status, notes); the shorter
    not-supported table and every heading fall out on the column count."""
    rows: dict[str, str] = {}
    for line in MODELS.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 8 or cells[0] in ("Model", ""):
            continue
        if set(cells[1]) <= {"-", ":"}:  # the header underline
            continue
        rows[cells[0]] = cells[7]
    return rows


def test_the_row_pin_matches_the_support_matrix():
    rows = set(_matrix_rows())
    assert rows, "no support-matrix rows parsed out of docs/MODELS.md"
    missing = sorted(rows - set(TEMPLATE_FOR_ROW))
    stale = sorted(set(TEMPLATE_FOR_ROW) - rows)
    assert not missing, (
        "docs/MODELS.md rows with no entry in TEMPLATE_FOR_ROW; ship a template "
        f"or write the explicit no-template note: {missing}"
    )
    assert not stale, f"TEMPLATE_FOR_ROW names rows MODELS.md no longer has: {stale}"


def test_every_supported_family_ships_a_template_or_an_explicit_gap():
    failures = []
    for model, notes in sorted(_matrix_rows().items()):
        template = TEMPLATE_FOR_ROW[model]
        if template is None:
            if not _NO_TEMPLATE.search(notes):
                failures.append(
                    f"{model}: no template and no 'no template yet' "
                    "note in its Notes cell"
                )
            continue
        if not (TEMPLATE_DIR / template).is_file():
            failures.append(f"{model}: pinned template {template} is not in the tree")
        if _NO_TEMPLATE.search(notes):
            failures.append(
                f"{model}: ships {template} and still carries a no-template note"
            )
    assert not failures, "\n".join(failures)


QUICKSTART = REPO / "docs" / "QUICKSTART.md"

# The five graphs that carry no family of their own. They share one Flux 1 Dev
# stack and name none of it in their Notes, so QUICKSTART names that stack in
# prose instead of naming each graph.
_ANY_FAMILY_TEMPLATES = frozenset({
    "dgx-monarch-quickstart",
    "dgx-monarch-lora-low-rss",
    "dgx-monarch-fleet",
    "dgx-monarch-identity-gate",
    "dgx-monarch-dual-spark-split",
})

_MODEL_FILE = re.compile(r"[\w.\-]+\.(?:safetensors|sft|gguf|ckpt|pt)\b")


def _quickstart_prose() -> str:
    return " ".join(QUICKSTART.read_text(encoding="utf-8").split())


def _graphs_whose_note_names_no_model_file() -> set[str]:
    """Templates that load model files and name none of them in a Note.

    A reader who opens one of these has no list to stage against, so
    QUICKSTART has to name the graph and say where to take the list from.
    """
    quiet = set()
    for path in TEMPLATES:
        doc = _load(path)
        loaded, notes = set(), []
        for node in doc["nodes"]:
            if node["type"] == "Note":
                notes.append((node.get("widgets_values") or [""])[0])
                continue
            for value in node.get("widgets_values") or []:
                if isinstance(value, str) and _MODEL_FILE.fullmatch(value):
                    loaded.add(value)
        text = " ".join(notes)
        if loaded and not any(name in text for name in loaded):
            quiet.add(path.stem)
    return quiet


def test_quickstart_names_every_graph_whose_note_lists_no_model_file():
    """QUICKSTART tells a new reader which Notes carry a file list.

    It says a family graph names its files, lists the Flux 1 Dev stack the
    five any-family graphs share, and names each LTX 2.5 graph whose Note
    lists no file. A template that stops naming its files, or a new one that
    never did, makes the page wrong for the reader who does not yet know
    MODELS.md.
    """
    prose = _quickstart_prose()
    missing = sorted(
        stem for stem in _graphs_whose_note_names_no_model_file() - _ANY_FAMILY_TEMPLATES
        if f"`{stem}`" not in prose
    )
    assert not missing, (
        "docs/QUICKSTART.md says a family graph names the model files it "
        "expects; these name none and the page does not list them: "
        f"{missing}. Put the file list in the Note through "
        "tools/gen_templates.py, or name the graph on the page."
    )


def test_quickstart_promises_a_graph_for_every_row_while_no_row_waives_one():
    """While QUICKSTART promises a graph for every row, no row may waive one.

    A row may take a dated no-template waiver instead of a template, and the
    first one makes that sentence false; nothing else in the suite reads it.
    The test skips while the page makes no such promise.
    """
    claim = (
        "carries a starting graph for every family in the "
        "[MODELS.md](MODELS.md) support matrix"
    )
    if claim not in _quickstart_prose():
        pytest.skip("docs/QUICKSTART.md no longer promises a graph for every row")
    waived = sorted(row for row, template in TEMPLATE_FOR_ROW.items() if template is None)
    assert not waived, (
        "docs/QUICKSTART.md promises a starting graph for every support-matrix "
        f"family and these rows carry the no-template waiver: {waived}. Ship the "
        "template or rewrite the sentence."
    )


# Beyond `auto` and what AUTO_TABLE resolves for its own canvas, a template may
# pin a topology a hardware grant covers. Keyed by docs/MODELS.md row, each
# with the grant it cites.
_RECORDED_GRANT_SCOPE: dict[str, frozenset[str]] = {
    # 2026-07-28: cross-rank identity plus 1-step NRMS 0.007 against an
    # independent single-GPU reference. `auto` never adds FSDP, so this scope
    # is unreachable through `auto` and the template has to name it.
    "Flux2": frozenset({"uly2+fsdp"}),
    # AUTO_TABLE row 44: the 4096-dim Pro model runs explicit uly2+fsdp for
    # capacity.
    "Kandinsky5-Video Pro": frozenset({"uly2+fsdp"}),
}

# The latent nodes the megapixel figure is read from. A 2-rank auto decision
# depends on megapixels, so a canvas on the wrong side of a crossover fails here.
# EmptyChromaRadianceLatentImage builds a 3-channel tensor at the output size,
# so width times height is its megapixel figure too. No shipped pixel-space
# template pins a topology yet; the entry waits for the first one.
_LATENT_TYPES = ("EmptyLatentImage", "EmptySD3LatentImage", "EmptyFlux2LatentImage",
                 "EmptyChromaRadianceLatentImage",
                 "EmptyHunyuanImageLatent", "EmptyHunyuanLatentVideo",
                 "EmptyHunyuanVideo15Latent", "EmptyLTXVLatentVideo",
                 "EmptyMiniMaxH3LatentAV")


def _rows_for_template(name: str) -> list[str]:
    return sorted(row for row, template in TEMPLATE_FOR_ROW.items() if template == name)


def _init_node(doc: dict) -> dict | None:
    for node in doc["nodes"]:
        if node["type"] == "DGXMonarchInit":
            return node
    return None


def _graph_shape(doc: dict) -> tuple[float | None, float | None]:
    """(megapixels, cfg) as the driver would read them off this graph."""
    megapixels = cfg = None
    for node in doc["nodes"]:
        widgets = node.get("dgxm_widgets") or {}
        if cfg is None and "cfg" in widgets:
            cfg = float(widgets["cfg"])
        if megapixels is None and node["type"] in _LATENT_TYPES:
            values = node.get("widgets_values") or []
            if len(values) >= 2 and all(isinstance(v, (int, float)) for v in values[:2]):
                megapixels = float(values[0]) * float(values[1]) / 1e6
    return megapixels, cfg


@pytest.mark.parametrize("path", TEMPLATES, ids=_IDS)
def test_a_pinned_init_topology_matches_auto_or_a_recorded_grant(path: Path):
    """`auto`, what auto resolves to here, or a topology a grant covers.

    An operator runs a template without reading the matrix, so a hand-pinned
    topology is a hardware claim. Any other pin recommends a path nothing
    measured.
    """
    from dgx_monarch.topology import choose_auto_topology
    from test_docs_status_sync import _SUPPORT_ROW_CONTRACT

    doc = _load(path)
    init = _init_node(doc)
    if init is None:
        pytest.skip("stock-only template")
    pinned = (init.get("dgxm_widgets") or {}).get("topology")
    assert pinned, f"{path.name}: DGXMonarchInit carries no topology widget"
    if pinned == "auto":
        return

    rows = _rows_for_template(path.name)
    assert rows, (
        f"{path.name}: pins topology {pinned!r} but no docs/MODELS.md row points at "
        "this template, so its family and evidence cannot be checked. Map the row "
        "in TEMPLATE_FOR_ROW or ship the template on `auto`."
    )
    families = {family for (_section, row), (family, _status)
                in _SUPPORT_ROW_CONTRACT.items() if row in rows}
    megapixels, cfg = _graph_shape(doc)
    assert megapixels is not None, (
        f"{path.name}: pins topology {pinned!r} but ships no latent this lint can "
        "size; auto resolution depends on the canvas"
    )

    allowed = {"auto"}
    for family in families:
        allowed.add(
            choose_auto_topology(family, "fp8", megapixels, 2, cfg_value=cfg)
            .topology.describe()
        )
        allowed.add(
            choose_auto_topology(family, "bf16", megapixels, 2, cfg_value=cfg)
            .topology.describe()
        )
    for row in rows:
        allowed |= set(_RECORDED_GRANT_SCOPE.get(row, ()))

    assert pinned in allowed, (
        f"{path.name}: Init pins topology {pinned!r}, which is neither `auto`, the "
        f"topology auto resolves to for {sorted(families)} at {megapixels:.2f} MP and "
        f"cfg {cfg}, nor a recorded grant scope for {rows}. Allowed: {sorted(allowed)}"
    )


# What a model edge can end at, and what it may pass through on the way. The
# splice rule is one hop: a LoRA loader takes the model straight off the UNET
# loader, so a shift node or a second loader may follow it, never precede it.
_MODEL_SINKS = frozenset({
    "DGXMonarchKSampler", "DGXMonarchKSamplerAdvanced", "DGXMonarchFleetKSampler",
    "DGXMonarchIdentityGate", "DGXMonarchCFGGuider", "DGXMonarchDualModelGuider",
})
_MODEL_PASSTHROUGH = frozenset({"DGXMonarchLoraLoader", "DGXMonarchModelSamplingSD3"})

_LORA_VARIANTS = [path for path in TESTING_TEMPLATES
                  if path.name.endswith(_LORA_VARIANT_SUFFIX)]
_LORA_IDS = [path.name for path in _LORA_VARIANTS]


def _model_consumers(doc: dict, node_id: int) -> list[dict]:
    """Every node that reads this node's first output."""
    by_id = {node["id"]: node for node in doc["nodes"]}
    return [by_id[link[3]] for link in doc["links"]
            if link[1] == node_id and link[2] == 0]


def _reaches_a_sampler(doc: dict, node_id: int) -> bool:
    seen, queue = set(), [node_id]
    while queue:
        for consumer in _model_consumers(doc, queue.pop()):
            if consumer["type"] in _MODEL_SINKS:
                return True
            if consumer["type"] in _MODEL_PASSTHROUGH and consumer["id"] not in seen:
                seen.add(consumer["id"])
                queue.append(consumer["id"])
    return False


def test_the_lora_variants_exist():
    # The filename suffix is the pin; an empty list passes every check below.
    assert _LORA_VARIANTS, f"no *{_LORA_VARIANT_SUFFIX} template under {TESTING_DIR}"


@pytest.mark.parametrize("path", _LORA_VARIANTS, ids=_LORA_IDS)
def test_a_lora_variant_chains_its_loader_between_unet_and_sampler(path: Path):
    # A loader wired anywhere else renders the base model and reports a LoRA
    # leg, the one failure a LoRA sweep cannot see in its own output.
    doc = _load(path)
    by_id = {node["id"]: node for node in doc["nodes"]}
    origin = {link[0]: by_id[link[1]] for link in doc["links"]}
    loaders = [node for node in doc["nodes"]
               if node["type"] == "DGXMonarchLoraLoader"]
    assert loaders, f"{path.name}: names a LoRA variant but places no loader"

    for node in doc["nodes"]:
        if node["type"] != "DGXMonarchUNETLoader":
            continue
        consumers = {consumer["type"] for consumer in _model_consumers(doc, node["id"])}
        assert consumers == {"DGXMonarchLoraLoader"}, (
            f"{path.name}: node {node['id']!r} hands its model to {sorted(consumers)}; "
            "the LoRA loader belongs directly after the UNET loader"
        )

    for loader in loaders:
        where = f"{path.name}: node {loader['id']!r}"
        named = loader["dgxm_widgets"]
        assert named["lora_name"] == _TEST_LORA_NAME, (
            f"{where}: lora_name is {named['lora_name']!r}; the testing set ships "
            f"the placeholder {_TEST_LORA_NAME}"
        )
        assert named["strength_model"] == _TEST_LORA_STRENGTH, (
            f"{where}: strength_model is {named['strength_model']!r}, not "
            f"{_TEST_LORA_STRENGTH}"
        )
        source = origin.get(loader["inputs"][0]["link"])
        assert source is not None and source["type"] in (
            "DGXMonarchUNETLoader", "DGXMonarchLoraLoader"), (
            f"{where}: takes its model from "
            f"{source['type'] if source else 'nothing'}"
        )
        assert _reaches_a_sampler(doc, loader["id"]), (
            f"{where}: its model reaches no sampler, so nothing renders through "
            "the LoRA"
        )


# Sequence parallel shards the chroma text stream. Since 2026-09-03 an odd
# count is padded under Ulysses and its pad rows are dropped from every
# attention call. Ring and hybrid have no full-sequence point to drop them at
# and stop with the waivable ring_pad card, so the shipped prompts stay even
# (since 2026-09-02). Only the tokenizer knows what a prompt counts, so these
# tests read tools/count_t5_tokens.py; the real check needs ComfyUI's tokenizer
# file, which CI does not have. Radiance is in the set because ChromaRadiance
# subclasses Chroma and runs the same adapter.
_CHROMA_TEMPLATES = [path for path in EVERY_TEMPLATE
                     if "chroma" in path.name or "radiance" in path.name]
_SHIPPED_CHROMA = TEMPLATE_DIR / "dgx-monarch-chroma-t2i.json"
_SHIPPED_RADIANCE = TEMPLATE_DIR / "dgx-monarch-radiance-t2i.json"


class _FakeTokenizer:
    """The tokenizers encode contract over whitespace, for CI."""

    class _Encoding:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text: str):
        # One id per whitespace-separated word plus the single end token the
        # chroma tokenizer appends to the whole stream.
        return self._Encoding(list(range(len(text.split()) + 1)))


def test_the_chroma_set_is_not_empty():
    # Every count below is parametrized over this list; an empty one passes.
    assert _CHROMA_TEMPLATES, "no chroma template found to count"
    assert _SHIPPED_CHROMA in _CHROMA_TEMPLATES
    assert _SHIPPED_RADIANCE in _CHROMA_TEMPLATES


def test_the_token_counter_refuses_prompts_comfy_splits():
    # Weighted text and embeddings become several segments, each tokenized on
    # its own, so a whole-string count stops matching what comfy builds. The
    # counter has to say so rather than report a number.
    fake = _FakeTokenizer()
    assert count_t5_tokens.count_tokens("two words", fake) == 3
    for prompt in ("a (weighted:1.2) prompt", "a [bracketed] prompt",
                   "embedding:something"):
        with pytest.raises(ValueError):
            count_t5_tokens.count_tokens(prompt, fake)


def test_the_token_counter_refuses_a_tokenizer_that_misses_the_known_count():
    # A tokenizer that does not reproduce the count the runtime named when it
    # refused the shipped prompt is the wrong tokenizer, and every number it
    # gives after that is wrong quietly.
    with pytest.raises(RuntimeError):
        count_t5_tokens.calibrate(_FakeTokenizer())


def test_the_token_counter_reads_every_text_encode_of_a_template():
    # The reader, not the tokenizer: it must find both prompts of the shipped
    # graph, in node order, or an odd one hides behind a missing node.
    found = count_t5_tokens.prompts_in(_SHIPPED_CHROMA)
    assert len(found) == 2, f"read {len(found)} prompts, not the cond/uncond pair"
    assert all(text.strip() for _, text in found), "a chroma prompt reads blank"


def _real_tokenizer():
    path = count_t5_tokens.tokenizer_path()
    if not path.is_file():
        pytest.skip(f"no ComfyUI T5 tokenizer at {path}")
    try:
        return count_t5_tokens.load_tokenizer(path)
    except ImportError:
        pytest.skip("the tokenizers package is not installed")


@pytest.mark.parametrize("path", _CHROMA_TEMPLATES,
                         ids=[p.name for p in _CHROMA_TEMPLATES])
def test_every_chroma_prompt_counts_even(path: Path):
    # An odd count stops ring and hybrid at the ring_pad card, on every quant,
    # in a graph that is otherwise correct, so nothing else in this file
    # catches it.
    tokenizer = _real_tokenizer()
    count_t5_tokens.calibrate(tokenizer)
    for node_id, text in count_t5_tokens.prompts_in(path):
        count = count_t5_tokens.count_tokens(text, tokenizer)
        assert count % 2 == 0, (
            f"{path.name}: node {node_id} counts {count} T5 tokens; sequence "
            "parallel divides the chroma text stream, and a count it does not "
            "divide takes the waivable ring_pad card on ring and hybrid. "
            "Reword it and run tools/count_t5_tokens.py"
        )


def test_the_shipped_chroma_pair_stays_unequal():
    # The shipped graph is the one that exercises cfg2's pad-and-mask path.
    # Equal lengths would skip the padding and leave that path covered only by
    # a testing-only template.
    tokenizer = _real_tokenizer()
    count_t5_tokens.calibrate(tokenizer)
    counts = [count_t5_tokens.count_tokens(text, tokenizer)
              for _, text in count_t5_tokens.prompts_in(_SHIPPED_CHROMA)]
    assert len(counts) == 2 and counts[0] != counts[1], (
        f"the shipped chroma prompts count {counts}; keep them different so "
        "cfg2 still pads and masks the shorter side"
    )


def test_the_shipped_chroma_note_states_the_counts_it_ships():
    # The Note is where a user reads the counts, so it has to carry the ones
    # the graph ships. The generator's other three restatements are checked
    # below.
    tokenizer = _real_tokenizer()
    count_t5_tokens.calibrate(tokenizer)
    counts = [count_t5_tokens.count_tokens(text, tokenizer)
              for _, text in count_t5_tokens.prompts_in(_SHIPPED_CHROMA)]
    notes = [node["widgets_values"][0] for node in _load(_SHIPPED_CHROMA)["nodes"]
             if node["type"] == "Note"]
    assert len(notes) == 1, f"the shipped chroma graph carries {len(notes)} notes"
    claimed = re.search(r"shipped pair counts (\d+) and (\d+)", notes[0])
    assert claimed, "the Note no longer states the shipped pair's token counts"
    assert [int(claimed.group(1)), int(claimed.group(2))] == counts, (
        f"the Note says {claimed.group(1)} and {claimed.group(2)}; the prompts "
        f"count {counts}. Rerun tools/count_t5_tokens.py and fix every place "
        "tools/gen_templates.py restates them"
    )


def _shipped_chroma_counts():
    tokenizer = _real_tokenizer()
    count_t5_tokens.calibrate(tokenizer)
    return [count_t5_tokens.count_tokens(text, tokenizer)
            for _, text in count_t5_tokens.prompts_in(_SHIPPED_CHROMA)]


def _flat_generator_source() -> str:
    """tools/gen_templates.py with its literal and comment wrapping removed."""
    raw = (REPO / "tools" / "gen_templates.py").read_text(encoding="utf-8")
    joined = re.sub(r'"\s*\n\s*"', "", raw)       # adjacent string literals
    joined = re.sub(r"\n\s*#\s?", " ", joined)    # wrapped comment lines
    return re.sub(r"\s+", " ", joined)


def test_every_restatement_of_the_shipped_chroma_counts_agrees():
    # Four sentences in the generator carry the shipped counts: the Note the
    # test above reads out of the graph, the comment above the testing-only
    # pair, that pair's purpose line, and the comment above the negative. No
    # other test reads the last three, so only this one catches them going
    # stale on a prompt edit.
    counts = _shipped_chroma_counts()
    assert len(counts) == 2
    positive, negative = counts
    flat = _flat_generator_source()
    restated = re.findall(r"shipped pair[^.]{0,80}?(\d+) and (\d+)", flat)
    assert len(restated) == 3, (
        f"tools/gen_templates.py restates the shipped pair {len(restated)} "
        "times; every one of them has to carry the counted numbers"
    )
    for said in restated:
        assert [int(said[0]), int(said[1])] == counts, (
            f"tools/gen_templates.py says {said[0]} and {said[1]}; the prompts "
            f"count {counts}. Rerun tools/count_t5_tokens.py and fix every "
            "sentence that restates them"
        )
    against = re.search(r"(\d+) tokens against the positive's (\d+)", flat)
    assert against, "the negative's comment no longer states the two counts"
    assert [int(against.group(1)), int(against.group(2))] == [negative, positive], (
        f"the negative's comment says {against.group(1)} against "
        f"{against.group(2)}; the prompts count {negative} and {positive}"
    )


def _vouched_families() -> set[str]:
    """The families whose adapter names a pad-row exclusion probe.

    Imported here rather than at module scope: the adapters pull torch in, and
    most tests in this file read JSON alone.
    """
    from dgx_monarch.adapters import ADAPTERS

    return {adapter.family for adapter in ADAPTERS
            if getattr(adapter, "usp_pad_exclusion_probe", None)}


# The clause both chroma Notes carry for a stream the sequence-parallel degree
# does not divide.
_PAD_CLAUSE = "pad rows are dropped from every attention call"
# Wording that promises the refusal chroma stopped giving on 2026-09-03.
_REFUSAL_PROMISES = ("has to encode to an even token count",
                     "must each divide the split",
                     "cannot divide by two")


@pytest.mark.parametrize("path", [_SHIPPED_CHROMA, _SHIPPED_RADIANCE],
                         ids=["chroma", "radiance"])
def test_a_pad_vouched_note_states_the_pad_path_and_promises_no_refusal(path: Path):
    # The Note is where an operator reads this rule, so it has to say what the
    # adapter does. ChromaAdapter has named a pad-exclusion probe since
    # 2026-09-03: a stream the degree does not divide is padded, those rows are
    # dropped, and the render goes on. A Note that still asks for an even count
    # sends the operator to reword a prompt that would have rendered (both
    # Notes were corrected on 2026-09-09).
    assert "chroma" in _vouched_families(), (
        "chroma names no pad-exclusion probe any more, so both Notes have to "
        "go back to stating the divisibility refusal"
    )
    notes = [node["widgets_values"][0] for node in _load(path)["nodes"]
             if node["type"] == "Note"]
    assert len(notes) == 1, f"{path.name} carries {len(notes)} notes"
    assert _PAD_CLAUSE in notes[0], (
        f"{path.name}: the Note has to say the pad rows of a stream that does "
        "not divide are dropped from every attention call, and it does not"
    )
    for promise in _REFUSAL_PROMISES:
        assert promise not in notes[0], (
            f"{path.name}: the Note still says {promise!r}, which chroma "
            "stopped doing on 2026-09-03"
        )


_SHIPPED_FLUX1 = TEMPLATE_DIR / "dgx-monarch-flux1-t2i.json"


def test_the_shipped_flux1_note_matches_the_probe_the_adapter_names():
    """The Note has to say what the adapter does with a stream that does not
    divide, because that is the sentence a user rewords a prompt over.

    Flux has named a pad-exclusion probe since 2026-09-09, so an odd stream
    pads and renders. A Note that says it refuses sends a user to shorten a
    prompt that would have rendered. The Note also has to name the day the
    fidelity legs ran and say they passed, because a reader decides on that
    whether to trust a padded render.
    """
    from dgx_monarch.adapters.flux_family import FluxAdapter

    assert FluxAdapter.usp_pad_exclusion_probe is not None, (
        "flux hands its probe back, so this Note has to promise the refusal "
        "again; rewrite it in tools/gen_templates.py and rerun the generator"
    )
    notes = [node["widgets_values"][0] for node in _load(_SHIPPED_FLUX1)["nodes"]
             if node["type"] == "Note"]
    assert len(notes) == 1, f"the shipped flux1 graph carries {len(notes)} notes"
    note = notes[0]
    assert "refuses instead of padding" not in note, (
        "the flux1 Note still tells a user an odd stream refuses; flux pads it "
        "and drops the pad rows since 2026-09-09 (issue #376)"
    )
    assert "pad rows are excluded from every attention call" in note, (
        "the flux1 Note no longer says what happens to a stream that does not "
        "divide the sequence-parallel degree"
    )
    assert "unmeasured" not in note, (
        "the flux1 Note still reads a padded render as unmeasured; the "
        "fidelity legs in benchmark/reports/flux_family_pad_fidelity_matrix"
        ".toml ran on 2026-09-09 and passed"
    )
    assert "ran on 2026-09-09 and passed" in note, (
        "the flux1 Note no longer names the day the fidelity legs for the "
        "padded path ran, which is the number a reader trusts a padded render "
        "on"
    )


@pytest.mark.parametrize(
    ("name", "steps", "cfg", "sampler_name", "expects_lora", "expects_release"),
    (("dgx-monarch-wan-animate2.json", 6, 1.0, "lcm", True, False),
     ("dgx-monarch-wan-animate2-bfloat16-warm.json", 6, 1.0, "lcm", True, True),
     ("dgx-monarch-wan-animate2-memory-saver.json", 6, 1.0, "lcm", True, True),
     ("dgx-monarch-wan-animate2-base-quality.json", 20, 1.0, "euler", False, False),
     ("dgx-monarch-wan-animate2-distilled.json", 10, 1.0, "lcm", False, False)),
)
def test_wan_animate2_templates_keep_the_native_motion_contract(
    name: str, steps: int, cfg: float, sampler_name: str, expects_lora: bool,
    expects_release: bool,
) -> None:
    """Pin the stock Animate2 conditioning rather than reducing it to Wan I2V."""
    doc = _load(TEMPLATE_DIR / name)
    nodes = {node["type"]: [] for node in doc["nodes"]}
    for node in doc["nodes"]:
        nodes.setdefault(node["type"], []).append(node)

    animate, = nodes["WanAnimate2ToVideo"]
    inputs = {item["name"]: item["link"] for item in animate["inputs"]}
    for required in ("reference_image", "pose_video", "clip_vision_output",
                     "positive_pose", "clip_vision_output_pose"):
        assert inputs[required] is not None, f"{name}: missing {required}"
    scale, = nodes["ImageScale"]
    assert scale["widgets_values"] == ["lanczos", 480, 832, "disabled"]
    canvas, = nodes["GetImageSize"]
    canvas_inputs = {item["name"]: item["link"] for item in canvas["inputs"]}
    assert canvas_inputs["image"] is not None
    scale_link = next(link for link in doc["links"] if link[0] == canvas_inputs["image"])
    assert scale_link[1:3] == [scale["id"], 0]
    assert animate["widgets_values"][:4] == [480, 832, 81, 1]
    for dimension, slot in (("width", 0), ("height", 1)):
        link_id = inputs[dimension]
        assert link_id is not None, f"{name}: {dimension} must follow Canvas Image Scale"
        link = next(link for link in doc["links"] if link[0] == link_id)
        assert link[1:3] == [canvas["id"], slot]
    pose_link = next(link for link in doc["links"] if link[0] == inputs["pose_video"])
    assert pose_link[1:3] == [scale["id"], 0]
    assert "video_frame_offset" in inputs

    sampler, = nodes["DGXMonarchKSampler"]
    widgets = sampler["dgxm_widgets"]
    assert widgets["steps"] == steps
    assert widgets["cfg"] == cfg
    assert widgets["sampler_name"] == sampler_name
    assert widgets["scheduler"] == "simple"
    if name == "dgx-monarch-wan-animate2-base-quality.json":
        assert widgets["cfg"] == 1.0, (
            "upstream sample_guide_scale=0 is conditional-only, while Comfy CFG 0 "
            "would select the unconditional prediction"
        )
    assert bool(nodes.get("DGXMonarchLoraLoader")) is expects_lora
    assert bool(nodes.get("DGXMonarchClearVRAM")) is expects_release
    init, = nodes["DGXMonarchInit"]
    assert init["dgxm_widgets"]["slab_weights"] == "off"
    assert init["dgxm_widgets"]["lora_low_rss"] == "off"
    assert init["dgxm_widgets"]["auto_gate"] == "first_use"
    assert init["dgxm_widgets"]["topology"] == "auto"
    note = next(n for n in doc["nodes"] if n["type"] == "Note")["widgets_values"][0]
    assert "runs the identity gate" not in note, (
        f"{name}: slab and low-RSS are off and no FSDP is chosen, so no first-use "
        "ceremony runs and the Note must not promise one"
    )
    assert "first-use gate to prove" in note
    assert not nodes.get("WanAnimate2Cache")
    assert not nodes.get("ModelSamplingSD3")
    assert nodes.get("DGXMonarchModelSamplingSD3")

    trim, = nodes["TrimVideoLatent"]
    trim_inputs = {item["name"]: item["link"] for item in trim["inputs"]}
    assert trim_inputs["trim_amount"] is not None
    video, = nodes["CreateVideo"]
    video_inputs = {item["name"]: item["link"] for item in video["inputs"]}
    assert video_inputs["fps"] is not None, f"{name}: export must use driving FPS"
    assert video_inputs["audio"] is not None, f"{name}: export must retain driving audio"
    assert nodes.get("SaveVideo")


# The sweep graphs copy a shipped Wan-Animate 2 Init and keep only a sweep Note,
# so the Note checks above stay on the shipped set; their residency pins live here.
_WAN_ANIMATE2_SWEEP = (
    "dgx-monarch-test-wan-animate2-int8.json",
    "dgx-monarch-test-wan-animate2-distilled-int8.json",
)
_WAN_ANIMATE2 = [path for path in TEMPLATES if "wan-animate2" in path.name]
_WAN_ANIMATE2 += [TESTING_DIR / name for name in _WAN_ANIMATE2_SWEEP]
_WAN_ANIMATE2_IDS = [str(path.relative_to(REPO)) for path in _WAN_ANIMATE2]


@pytest.mark.parametrize("name", _WAN_ANIMATE2_SWEEP)
def test_wan_animate2_sweep_graphs_keep_slab_and_low_rss_off(name: str) -> None:
    init = _init_node(_load(TESTING_DIR / name))
    assert init is not None, f"{name}: no DGXMonarchInit node"
    assert init["dgxm_widgets"]["slab_weights"] == "off"
    assert init["dgxm_widgets"]["lora_low_rss"] == "off"


@pytest.mark.parametrize("path", _WAN_ANIMATE2, ids=_WAN_ANIMATE2_IDS)
def test_wan_animate2_residency_leaves_first_use_nothing_to_prove(
    path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shipped Notes promise a first render with no gate. With slab, low-RSS
    and comfy_managed off and no FSDP topology, the first-use trigger finds no
    risky residency, whether or not the graph chains a LoRA."""
    from types import SimpleNamespace

    from dgx_monarch import first_render
    from dgx_monarch.nodes import auto_gate, common
    from dgx_monarch.nodes import init as init_module

    doc = _load(path)
    handle = SimpleNamespace(world=2)
    monkeypatch.setattr(init_module, "get_mesh", lambda **_kwargs: handle)
    monkeypatch.setattr(first_render, "nccl_deferred", lambda: None)
    monkeypatch.setattr(common, "ensure_live", lambda value: value)
    widgets = {**_init_node(doc)["dgxm_widgets"], "auto_gate": "first_use"}
    (mesh,) = init_module.DGXMonarchInit().init(**widgets)
    has_lora = any(node["type"] == "DGXMonarchLoraLoader" for node in doc["nodes"])
    model = SimpleNamespace(
        mesh=mesh, loras=({"name": "lora"},) if has_lora else (),
        unet_name="model.safetensors")
    assert auto_gate.auto_gate_context(model, "ksampler") is None


def test_accelerated_wan_default_keeps_actors_warm_and_memory_saver_releases():
    """Warm INT8 is the default; recycle is an explicit BF16 fallback."""
    warm = _load(TEMPLATE_DIR / "dgx-monarch-wan-animate2.json")
    warm_nodes = {node["type"]: node for node in warm["nodes"]}
    warm_links = {link[0]: link for link in warm["links"]}
    assert "DGXMonarchClearVRAM" not in warm_nodes

    def warm_source(node_type, input_name):
        port = next(item for item in warm_nodes[node_type]["inputs"]
                    if item["name"] == input_name)
        return warm_links[port["link"]][1:3]

    assert warm_source("VAEDecode", "samples") == [warm_nodes["TrimVideoLatent"]["id"], 0]

    warm_bf16 = _load(TEMPLATE_DIR / "dgx-monarch-wan-animate2-bfloat16-warm.json")
    bf16_nodes = {node["type"]: node for node in warm_bf16["nodes"]}
    bf16_links = {link[0]: link for link in warm_bf16["links"]}
    soft = bf16_nodes["DGXMonarchClearVRAM"]
    soft_samples = next(item for item in soft["inputs"] if item["name"] == "samples")
    bf16_decode = bf16_nodes["VAEDecode"]
    decode_samples = next(item for item in bf16_decode["inputs"] if item["name"] == "samples")
    assert soft["dgxm_widgets"] == {"level": "soft", "include_driver": True}
    assert bf16_links[soft_samples["link"]][1:3] == [bf16_nodes["TrimVideoLatent"]["id"], 0]
    assert bf16_links[decode_samples["link"]][1:3] == [soft["id"], 1]
    assert bf16_nodes["DGXMonarchUNETLoader"]["dgxm_widgets"]["unet_name"] == "wan_animate_2_bf16.safetensors"

    doc = _load(TEMPLATE_DIR / "dgx-monarch-wan-animate2-memory-saver.json")
    nodes = {node["type"]: node for node in doc["nodes"]}
    links = {link[0]: link for link in doc["links"]}

    def source(node_type, input_name):
        port = next(item for item in nodes[node_type]["inputs"]
                    if item["name"] == input_name)
        return links[port["link"]][1:3]

    release = nodes["DGXMonarchClearVRAM"]
    assert release["dgxm_widgets"] == {"level": "recycle", "include_driver": False}
    assert source("TrimVideoLatent", "samples") == [nodes["DGXMonarchKSampler"]["id"], 0]
    assert source("DGXMonarchClearVRAM", "samples") == [nodes["TrimVideoLatent"]["id"], 0]
    assert source("DGXMonarchClearVRAM", "mesh") == [nodes["DGXMonarchInit"]["id"], 0]
    assert source("VAEDecode", "samples") == [release["id"], 1]
    assert source("VAEDecode", "vae") == [nodes["VAELoader"]["id"], 0]
    unet, = [node for node in doc["nodes"] if node["type"] == "DGXMonarchUNETLoader"]
    assert unet["dgxm_widgets"]["unet_name"] == "wan_animate_2_bf16.safetensors"
