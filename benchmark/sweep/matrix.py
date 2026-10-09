"""Derive and label sweep cells from templates, artifacts, and runtime guards.

Call runtime guards instead of copying their rules. Each cell records a label
and its basis:

* ``guard``: a runtime refusal;
* ``disk``: a missing checkpoint or unidentified family;
* ``name-token``: block-scaled layout inferred from the filename;
* ``header``: checkpoint and LoRA metadata;
* ``config``: a template excluded by configuration;
* ``token-count`` or ``token-count-unavailable``: prompt divisibility;
* ``per-cond-dispatch`` or ``cond-fold-unknown``: the expected CFG path for
  unequal prompts;
* ``open-defect``: a known runtime defect (see OPEN_DEFECT).

Block-scaled layouts need filename inference because their runtime kind is
available only after loading the model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from dgx_monarch import upstream_gate  # noqa: E402
from dgx_monarch.adapters import (  # noqa: E402
    ADAPTERS,
    DUAL_MODEL_CFG_FAMILIES,
    attention_capability,
    minimax_h3,
    omnigen2,
    quant_activation_scale,
    sol_attention,
)
from dgx_monarch.adapters.cfg_parallel import (  # noqa: E402
    assert_cfg_parallel_supported,
    assert_sp_quant_supported,
    cfg_dispatches_per_cond,
)
from dgx_monarch.adapters.detect import (  # noqa: E402
    fsdp_lora_admission_property,
    sniff_checkpoint_for_topology,
    sniff_fsdp_launch_quant,
)
from dgx_monarch.adapters.fsdp import validate_fsdp_launch_loras, validate_fsdp_launch_quant  # noqa: E402
from dgx_monarch.adapters.fsdp_lora_admission import refuse_fsdp_lora_checkpoint_property  # noqa: E402
from dgx_monarch.capacity_lora_bake import patched_base_keys  # noqa: E402
from dgx_monarch.latent_scale import spatial_megapixels  # noqa: E402
from dgx_monarch.mesh_runtime import ATTACH_INIT_WAIT_S  # noqa: E402
from dgx_monarch.mesh_safety import REF_POSE_TOKEN_FAMILIES  # noqa: E402
from dgx_monarch.nodes.common import resolve_sample_attention  # noqa: E402
from dgx_monarch.nodes.render_validation import (  # noqa: E402
    indivisible_batch_refusal,
    preflight_graph_batch_divides_dp,
)
from dgx_monarch.refusal import parse_refusal_tag  # noqa: E402
from dgx_monarch.safetensors_header import read_safetensors_header  # noqa: E402
from dgx_monarch.topology import choose_auto_topology, topology_from_preset  # noqa: E402

from .convert import probe_support  # noqa: E402

UNET_CLASSES = ("DGXMonarchUNETLoader", "DGXMonarchUncondUNETLoader")
LORA_CLASS, INIT_CLASS = "DGXMonarchLoraLoader", "DGXMonarchInit"
PACKED_CLASSES = frozenset({"LTXVConcatAVLatent", "EmptyMiniMaxH3LatentAV",
                            "MiniMaxH3ImageToVideo", "MiniMaxH3AddGuide"})
LENGTH_WIDGETS = ("length", "video_frames", "frames_number")
RESIDENCY_TOGGLES = frozenset({"slab_weights", "lora_low_rss", "comfy_managed"})
# The stock CLIP loaders and the widgets on each that name a text-encoder file.
# test_text_encoder_loader_classes_pin_against_the_artifact_manifest_widgets
# pins them against tools/check_artifacts.py LOADER_WIDGETS, which also names
# the "text_encoders" models subfolder both resolve against; the seam "loader
# model folders" in tests/canary/comfy_seam_contracts.py pins that folder to comfy.
TEXT_ENCODER_LOADER_CLASSES: dict[str, tuple[str, ...]] = {
    "CLIPLoader": ("clip_name",),
    "DualCLIPLoader": ("clip_name1", "clip_name2"),
}
# The bar a comparison of two full warm renders is scored at. The step-nrms
# floor was calibrated for a one-step probe and does not apply to a full
# render: ideogram4 bf16 uly2 read 0.198 as a full render and 0.000 at one step
# (2026-09-03). A full render carries no floor, so the number is measured and
# reported and never scored.
FULL_RENDER_BAR = "full-render-nrms"
# The bar a cell running a kernel the runtime already knows to be wrong is
# recorded at. A sol kernel is an approximation the class K waiver admits, so
# its deviation from the reference render is the deliverable rather than a
# doubt: the number carries no floor and is never scored.
WAIVED_BAR = "waived-nrms"
# A token comes before every token it contains, so a shorter match never takes
# part of a longer one.
QUANT_TOKENS = ("convrot_int8mixed", "convrot_int8", "int8_convrot", "fp8_e4m3fn",
                "fp8_scaled", "nvfp4_mixed", "int8mixed", "fp8mixed", "mxfp8",
                "nvfp4", "bf16", "fp16", "fp32", "int8", "fp8")
# A block-scaled file under FSDP refuses at one of two sites, by whether the
# file is uniform, and both answer class P. Every one of these headers sniffs
# fp8 at launch, so the live detector decides: it reports the kind of the first
# quantized module it meets (actor/store_detect.py). A uniform file names its
# own block-scaled kind, which the launch contract refuses (adapters/fsdp.py
# validate_fsdp_launch_quant). A mixed file names the admitted fp8 and meets
# the shard-layout check (adapters/fsdp_quant.py wrap_quantized_parameters).
# The test is a substring, so nvfp4_mixed stays ahead of nvfp4: the two meet
# different refusal sites.
BLOCK_SCALED = {"nvfp4_mixed": "refuse:P", "nvfp4": "refuse:P", "mxfp8": "refuse:P"}
# Words that name neither the model nor its precision; a stem keeping one
# lands in its own base and never meets its sibling quants.
RESIDUE_TOKENS = ("transformer_only", "defaultloader", "hybrid_large",
                  "comfy", "simple", "final")
RESIDUE_PATTERNS = (r"rev\d+",)
ADAPTER_BY_FAMILY = {adapter.family: adapter for adapter in ADAPTERS}
# A checkpoint this large can meet the stock-load capacity preflight before
# any sample-time guard the matrix models, so its cell is flagged rather than
# scored against the guard that would have answered on a smaller file.
CAPACITY_RISK_GIB = 50.0
# The 33.0 GiB Flux2 fp8mixed file is a separate measured rig boundary, flagged
# 2026-09-11. The constant is that file's size after the sweep's one-decimal
# rounding, not an estimate of the host's live memory: capacity answered before
# the cfg2 class-P guard, and the same artifact also exhausted the stock rung
# under auto. A changed file size, quant, or family needs its own evidence
# before it earns the flag.
FLUX2_FP8MIXED_CAPACITY_BOUNDARY_GIB = 33.0
# Families whose fp32-stored weights comfy casts to bf16 at load, so a LoRA that
# patches one meets the in-bake dtype refusal under FSDP. Measured: krea2
# (krea2_raw_bf16 + its turbo LoRA, 15 such keys, debt re-run 2026-09-29).
# A family whose fp32 weights stay fp32 live bakes fine, so it joins only with
# its own evidence.
FSDP_LORA_LIVE_CAST_FAMILIES = frozenset({"krea2"})
# uly2 and ring2 are measured too: the 2026-09-29 debt re-run (comfy a7169322)
# met the same partial_load_divergence refusal on six flux2 LoRA cells there.
FLUX2_FP8MIXED_CAPACITY_PRESETS = frozenset({"auto", "cfg2", "ring2", "uly2", "uly2+fsdp"})

# Families whose USP forward shards the text stream and the image stream apart,
# so their sequence-parallel label is the prompt's token count, which the graph
# does not carry. Which of them refuses a divisibility pad row is the adapter's
# own probe attribute, never a name written here: without a probe the shard
# refuses class P (docs/TROUBLESHOOTING.md entry 77), and with one the pad rows
# are excluded before every kernel (chroma since 2026-09-03; flux, flux2 and
# longcat since 2026-09-09), which is exact under pure ulysses and has no
# full-sequence point under ring.
_FLUX_FAMILY = tuple(adapter for adapter in ADAPTERS
                     if type(adapter).__module__.endswith("flux_family"))
SP_EXACT_TEXT_FAMILIES = frozenset(
    adapter.family for adapter in _FLUX_FAMILY
    if adapter.usp_pad_exclusion_probe is None)
SP_PAD_VOUCHED_FAMILIES = frozenset(
    adapter.family for adapter in _FLUX_FAMILY
    if adapter.usp_pad_exclusion_probe is not None)
COND_SLOTS = ("positive", "negative")
TEXT_WIDGETS = ("text", "prompt")
# comfy splits a prompt carrying weight or embedding syntax into segments and
# tokenizes each apart, which this reader does not do, so a prompt carrying
# either is left uncounted rather than counted wrong.
WEIGHT_SYNTAX = re.compile(r"[()\[\]]|embedding:")

# The basis for a known runtime defect the matrix carries until its fix lands.
# No rule uses it. A rule that does names the defect, and is deleted when the
# fix lands.
OPEN_DEFECT = "open-defect"


class TextStream(NamedTuple):
    """The tokenizer file for one family's text stream and its pad floor.

    ``floor`` is comfy's own ``min_length`` for that stream: a shorter prompt
    reaches the model padded up to it, so the floor is the length the adapter
    sees. ``file`` is relative to the tokenizer directory in the sweep config.
    """

    file: str
    floor: int


T5_TOKENIZER = "t5_tokenizer/tokenizer.json"
# The stream whose length decides a label, per family, with the comfy
# tokenizer that builds it. Only families whose tokenizer comfy ships as a
# `tokenizers` file are here; every other family counts nothing and keeps the
# label its guards give. Anima is a partial answer: unequal T5 ids do split the
# model call, but equal ones still need its qwen3 stream to match, and comfy
# ships that tokenizer as vocab and merges files, which this reader cannot load.
TEXT_STREAMS = {
    "chroma": TextStream(T5_TOKENIZER, 1),     # text_encoders/pixart_t5.py
    "anima": TextStream(T5_TOKENIZER, 1),      # text_encoders/anima.py
    "cogvideo": TextStream(T5_TOKENIZER, 226),  # text_encoders/cogvideo.py
    "flux": TextStream(T5_TOKENIZER, 256),     # text_encoders/flux.py
}

DEFAULTS: dict = {
    "driver": "http://127.0.0.1:8191", "sibling": "", "comfy_dir": "~/ComfyUI",
    "out_dir": "~/dgx-monarch-sweep", "world": 2, "exclude": [],
    "template_dirs": ["example_workflows"],
    "image_presets": ["auto", "uly2", "cfg2", "ring2", "dp2",
                      "uly2+fsdp", "cfg2+fsdp", "ring2+fsdp"],
    "video_presets": ["auto", "uly2", "ring2", "uly2+fsdp"],
    "image_attention": ["TORCH_FLASH", "TORCH_CUDNN", "SAGE_AUTO"],
    "video_attention": ["TORCH_FLASH"], "video_attention_full": {},
    "sol_kernels": ["SOL_ATTN_TAU0.7"], "sol_families": ["minimax_h3"],
    "toggles": {}, "toggle_presets": {}, "toggle_combos": [],
    # Levers the runtime schedules work with and never changes a number by.
    # They are held to the pixel against the auto cell, so any difference is a
    # finding.
    "inert_levers": ["pipeline_depth", "sync_ulysses"],
    "loras": {}, "lora_off": {}, "artifact_groups": {},
    "reference": {"probe_steps": 1, "nrms_floor": 0.10}, "mem_floor_gib": 60.0,
    "cooldown_s": 20.0,
    "image_timeout_s": 3600, "video_timeout_s": 7200,
    # Templates the current ComfyUI admits and the adapter cannot run. Their
    # cells are derived and labelled skip:upstream-gated.
    "upstream_gated": [],
    # Empty derives it from comfy_dir, which is where comfy keeps the files.
    "tokenizer_dir": "",
    # Seconds to hold after a fleet replacement before the next attach.
    # 0 follows monarch's own attach budget; see attach_pace_default.
    "attach_pace_s": 0.0,
}

log = logging.getLogger(__name__)


def tokenizer_dir(config: dict) -> Path:
    """Where the comfy tokenizer files live for this rig."""
    named = config.get("tokenizer_dir") or ""
    return Path(named).expanduser() if named \
        else Path(config["comfy_dir"]) / "comfy" / "text_encoders"


ATTACH_TIMEOUT_ENV = "HYPERACTOR_MESH_ATTACH_CONFIG_TIMEOUT"


def attach_pace_default() -> float:
    """Return the minimum delay between fleet replacement and the next attach.

    Use the larger of HYPERACTOR_MESH_ATTACH_CONFIG_TIMEOUT and the runtime's
    ATTACH_INIT_WAIT_S (that timeout plus 10 s, with a 70 s minimum). The package
    default of 60 s gives 70 s; the sweep driver's 180 s setting gives 190 s.
    Read the runner's environment, so export the setting there as well as on the
    driver. A shorter delay could expire during monarch's attach budget.
    """
    named = os.environ.get(ATTACH_TIMEOUT_ENV, "")
    try:
        return max(float(ATTACH_INIT_WAIT_S), float(named.strip().rstrip("s")))
    except ValueError:
        if named:
            log.warning("%s=%r is not a number of seconds, so the attach pace "
                        "stays at %s s", ATTACH_TIMEOUT_ENV, named, ATTACH_INIT_WAIT_S)
        return float(ATTACH_INIT_WAIT_S)


def load_config(path: Path) -> dict:
    config = dict(DEFAULTS)
    config.update(tomllib.loads(path.read_text()))
    config["comfy_dir"] = Path(config["comfy_dir"]).expanduser()
    config["out_dir"] = Path(config["out_dir"]).expanduser()
    dirs = [Path(entry).expanduser() for entry in config["template_dirs"]]
    config["template_dirs"] = [d if d.is_absolute() else REPO / d for d in dirs]
    config["model_dir"] = config["comfy_dir"] / "models" / "diffusion_models"
    config["tokenizer_dir"] = tokenizer_dir(config)
    config["attach_pace_s"] = float(config["attach_pace_s"]) or attach_pace_default()
    return config


class TokenCounter:
    """Count a prompt the way comfy's tokenizer for this family counts it.

    comfy drops the tokenizer's own end token from each word it tokenizes and
    appends exactly one at the end of the stream, then pads up to that
    stream's floor (comfy/sd1_clip.py SDTokenizer.tokenize_with_weights). The
    tokenizer file comfy ships appends the same end token to a whole prompt, so
    the count is that length raised to the floor. Calibration: the chroma
    prompt the runtime refused at uly2 counts 101 here, which is the length
    that refusal named (2026-09-02).

    A count nobody can read is not guessed. A family with no shipped tokenizer
    file, a missing file, and a prompt carrying comfy's weight syntax all
    answer None, and the caller keeps the label its guards gave.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._tokenizers: dict[str, object] = {}

    def __call__(self, family: str, text: str) -> int | None:
        stream = TEXT_STREAMS.get(family)
        if stream is None or WEIGHT_SYNTAX.search(text):
            return None
        tokenizer = self._tokenizer(stream.file)
        if tokenizer is None:
            return None
        return max(len(tokenizer.encode(text).ids), stream.floor)

    def _tokenizer(self, name: str):
        """The loaded tokenizer, or None, said once per file rather than per cell."""
        if name not in self._tokenizers:
            path = self.directory / name
            try:
                from tokenizers import Tokenizer

                self._tokenizers[name] = Tokenizer.from_file(str(path))
            except Exception as exc:  # tokenizers raises a bare Exception
                log.warning("token counts unavailable for %s: %s", path, exc)
                self._tokenizers[name] = None
        return self._tokenizers[name]


def split_quant(stem: str) -> tuple[str, str]:
    """Split a checkpoint stem into its artifact base and its quant tokens.

    Separators normalise first, so ``int8-convrot`` meets the ``int8_convrot``
    token, and residue words drop after, so ``-comfy-`` in one sibling's name
    does not file it away from the other."""
    low = re.sub(r"[^a-z0-9]+", "_", stem.lower())
    tokens: list[str] = []
    while True:
        for candidate in QUANT_TOKENS:
            if candidate in low:
                tokens.append(candidate)
                low = low.replace(candidate, "", 1)
                break
        else:
            break
    for word in RESIDUE_TOKENS:
        low = low.replace(word, "")
    for pattern in RESIDUE_PATTERNS:
        low = re.sub(pattern, "", low)
    return re.sub(r"[^a-z0-9]+", "", low), "+".join(tokens)


def sampler_keeps_uncond(sampler_name: str) -> bool:
    """Whether this sampler still runs the unconditional pass at CFG 1.0.

    comfy drops that pass at CFG 1.0 unless ``disable_cfg1_optimization`` is
    set (comfy/samplers.py), and at comfy a7169322 every sampler that sets it
    is a cfg++ sampler, named with a ``cfg_pp`` suffix
    (comfy/k_diffusion/sampling.py). Keeping the pass means comfy folds cond
    and uncond into one call of batch 2, which a cfg topology splits across
    its two ranks (krea2 at cfg2, 2026-09-02).
    """
    return isinstance(sampler_name, str) and sampler_name.endswith("cfg_pp")


def _prompt_behind(graph: dict, link: object, seen: frozenset = frozenset()) -> str | None:
    """The prompt one conditioning link carries, or None when the walk ends.

    A slot may link through a conditioning node (guides, image conditioning,
    an LTX shot list), which returns each stream on the slot it took it from,
    so the walk follows the input named for the output slot it arrived on.
    Following any input instead would hand the negative slot the positive
    prompt on every template that carries one of those nodes.
    """
    if not (isinstance(link, list) and len(link) == 2 and str(link[0]) in graph):
        return None
    node_id, slot = str(link[0]), link[1]
    if node_id in seen:
        return None
    inputs = graph[node_id]["inputs"]
    for name in TEXT_WIDGETS:
        if isinstance(inputs.get(name), str):
            return inputs[name]
    if set(COND_SLOTS) <= set(inputs) and isinstance(slot, int) and slot < len(COND_SLOTS):
        return _prompt_behind(graph, inputs[COND_SLOTS[slot]], seen | {node_id})
    links = [value for value in inputs.values() if isinstance(value, list)]
    return _prompt_behind(graph, links[0], seen | {node_id}) if len(links) == 1 else None


def prompt_texts(graph: dict) -> dict[str, str]:
    """The prompt each conditioning slot of the first sampler resolves to.

    Only a sampler of this pack's own carries the pair under test; the
    conditioning nodes above it carry the same slot names. A second sampler
    stage renders the same pair, so the first one answers for the graph. A pair
    the walk cannot resolve is no pair at all: counting one side twice would
    say two streams match when nothing was read.
    """
    samplers = [node_id for node_id, node in graph.items()
                if node["class_type"].startswith("DGXMonarch")
                and set(COND_SLOTS) <= set(node["inputs"])]
    if not samplers:
        return {}
    sampler = min(samplers, key=int)
    found = {slot: _prompt_behind(graph, graph[sampler]["inputs"][slot])
             for slot in COND_SLOTS}
    return {} if any(text is None for text in found.values()) else found


def text_token_counts(prompts: dict[str, str], family: str, count_tokens) -> dict[str, int]:
    """Token count per prompt, or {} when any one of them cannot be counted."""
    counts = {slot: count_tokens(family, text) for slot, text in prompts.items()}
    return {} if not counts or any(count is None for count in counts.values()) else counts


# The stock load nodes that name a file in comfy's input folder, by the input
# the converted graph carries it on. Comfy's own validation rejects a template
# naming media the rig does not have, cell after cell, so the matrix prunes it
# and says which files are missing.
MEDIA_INPUTS = {"LoadImage": "image", "LoadImageMask": "image",
                "LoadVideo": "file", "LoadAudio": "audio"}
# Comfy annotates a file picked from another folder. The annotation is
# stripped before lookup; a subfolder path resolves under the input folder.
MEDIA_ANNOTATIONS = (" [input]", " [output]", " [temp]")


def missing_media(graph: dict, input_dir: Path) -> list[str]:
    """The media files this graph names that the input folder does not hold.

    An input converted to a link carries a list rather than a name, so it
    claims nothing here: the file it resolves to is not in the graph. A folder
    that does not exist answers nothing either, since every name would read as
    missing and the whole video half of the sweep would prune itself.
    """
    if not input_dir.is_dir():
        return []
    missing = []
    for node in graph.values():
        widget = MEDIA_INPUTS.get(node.get("class_type", ""))
        name = node.get("inputs", {}).get(widget) if widget else None
        if not isinstance(name, str) or not name:
            continue
        for annotation in MEDIA_ANNOTATIONS:
            name = name[:-len(annotation)] if name.endswith(annotation) else name
        if not (input_dir / name).exists():
            missing.append(name)
    return sorted(set(missing))


def graph_facts(graph: dict) -> dict:
    """What the matrix needs to know about one API graph."""
    unets = {nid: node["inputs"]["unet_name"] for nid, node in graph.items()
             if node["class_type"] in UNET_CLASSES and "unet_name" in node["inputs"]}
    loras = [nid for nid, node in graph.items() if node["class_type"] == LORA_CLASS]
    classes = {node["class_type"] for node in graph.values()}
    width = height = 0
    cfg, batch, video = None, 1, False
    batch_node, sampler = "", ""
    for node_id, node in graph.items():
        inputs = node["inputs"]
        if isinstance(inputs.get("sampler_name"), str):
            sampler = inputs["sampler_name"]
        if isinstance(inputs.get("width"), int) and isinstance(inputs.get("height"), int) \
                and inputs["width"] * inputs["height"] > width * height:
            width, height = inputs["width"], inputs["height"]
        if isinstance(inputs.get("cfg"), (int, float)):
            cfg = float(inputs["cfg"])
        if isinstance(inputs.get("batch_size"), int) and not batch_node:
            batch, batch_node = inputs["batch_size"], node_id
        video = video or any(name in inputs for name in LENGTH_WIDGETS)
    # The probe rewrite decides the probe leg, not a class-name prefix: the
    # step count of an ideogram4 or LTX graph lives in a scheduler node of the
    # family's own, and a staged graph gets one step per stage.
    probe, probe_reason = probe_support(graph)
    # The graph's pixel size through the runtime's own megapixel definition:
    # the driver reads the latent grid times comfy's ratio for the model, the
    # same number wherever the latent node divides by that ratio.
    return {
        "unets": unets, "cond_node": min(unets, key=int) if unets else "",
        "loras": sorted(loras, key=int), "megapixels": spatial_megapixels(height, width),
        "dual_model": any(graph[nid]["class_type"] == UNET_CLASSES[1] for nid in unets),
        "packed": bool(classes & PACKED_CLASSES), "audio": "SaveAudioAdvanced" in classes,
        "cfg": cfg, "batch": batch, "probe": probe, "probe_reason": probe_reason,
        "sampler": sampler, "sampler_keeps_uncond": sampler_keeps_uncond(sampler),
        "batch_node": batch_node, "batch_widget": bool(batch_node),
        "klass": "video" if video else "image",
        "prompts": prompt_texts(graph),
        # The graph itself, for the rules that hand it to a runtime check
        # rather than restate what the check would read. It rides in the facts
        # and never in a cell: a cell is an identity and a label, not a copy of
        # the template.
        "graph": graph,
    }


def resident_text_encoder_gib(graph: dict, comfy_dir: Path) -> float:
    """On-disk GiB of every text-encoder file this graph's CLIP loaders place.

    Local mode runs the driver and the worker on the same box (docs/SWEEP.md
    section 1), so a stock DiT load there competes against whatever the CLIP
    loader already put in MemAvailable, not against an empty one. Only
    models/text_encoders is read: a name that is not a file there, even one
    comfy finds in its legacy models/clip folder, prices as absent.
    """
    te_dir = comfy_dir / "models" / "text_encoders"
    total = 0
    for node in graph.values():
        widgets = TEXT_ENCODER_LOADER_CLASSES.get(node.get("class_type", ""))
        if not widgets:
            continue
        for widget in widgets:
            name = (node.get("inputs") or {}).get(widget)
            path = te_dir / name if isinstance(name, str) else None
            if path is not None and path.is_file():
                total += path.stat().st_size
    return round(total / 2**30, 1)


def artifact_sets(unets: dict[str, str], files: list[Path],
                  extras: dict[str, list[str]]) -> tuple[list[dict[str, str]], list[str]]:
    """One artifact set per quant every loader in the graph can supply.

    ``extras`` names the loader it extends: a key is the checkpoint that loader
    ships, and the files under it join that loader's pool whatever they are
    called. A key no loader carries is reported rather than dropped."""
    by_base: dict[str, dict[str, str]] = {}
    for path in files:
        base, quant = split_quant(path.stem)
        by_base.setdefault(base, {})[quant] = path.name
    loader_base = {nid: split_quant(Path(name).stem)[0] for nid, name in unets.items()}
    extra_pruned: list[str] = []
    for stem, names in (extras or {}).items():
        owner = split_quant(Path(stem).stem)[0]
        if owner not in set(loader_base.values()):
            extra_pruned.append(f"artifact_groups key {stem!r} names no loader in this graph")
            continue
        for extra in names:
            by_base.setdefault(owner, {})[split_quant(Path(extra).stem)[1]] = extra
    shared: set[str] | None = None
    for base in loader_base.values():
        available = set(by_base.get(base, {}))
        shared = available if shared is None else (shared & available)
    pruned = extra_pruned + [
        f"{by_base[base][quant]} (not every loader in this graph has this quant)"
        for base in sorted(set(loader_base.values()))
        for quant in sorted(set(by_base.get(base, {})) - (shared or set()))]
    sets = [{nid: by_base[loader_base[nid]][quant] for nid in unets}
            for quant in sorted(shared or set())]
    return sets, pruned


class ConfigError(Exception):
    """The sweep TOML says something this harness cannot read."""


def template_extras(config: dict, template: str, facts: dict) -> dict[str, list[str]]:
    """The artifact_groups entry for one template, in the dict form. An array
    names extras for the base loader; the table keys each loader's checkpoint."""
    extras = config["artifact_groups"].get(template, {})
    if isinstance(extras, dict):
        return extras
    if isinstance(extras, list):
        return {facts["unets"][facts["cond_node"]]: list(extras)}
    raise ConfigError(f"artifact_groups.{template!r} is a {type(extras).__name__}; write an "
                      "array of extras for the base loader, or a table keyed by each "
                      "loader's own checkpoint")


def refusal_label(text: str) -> str | None:
    """The refusal class a message carries, or None when it carries no lone tag."""
    tag = parse_refusal_tag(text)
    return f"refuse:{tag.refusal_class.value}" if tag else None


def _label_from(exc: BaseException) -> str:
    return refusal_label(str(exc)) or "refuse:untyped"


def _guard(call, *args, **kwargs) -> str | None:
    try:
        call(*args, **kwargs)
    except Exception as exc:
        return _label_from(exc)
    return None


def loader_mesh(preset: str, world: int) -> SimpleNamespace:
    """What the loader node's own preflight reads off a live mesh.

    The preset and the world are the whole of it: the check resolves the
    topology from the two and makes no claim on an auto preset or a handle
    that names no world.
    """
    return SimpleNamespace(topology_preset=preset,
                           handle=SimpleNamespace(world=world, defunct=False))


def batch_graph(cell: dict, facts: dict) -> dict:
    """The graph as the runner queues it, for a check that reads the batch.

    ``run.patch_graph`` writes a toggled batch into the one node the matrix
    read it from, so a cell the batch toggle made runs a graph the file on disk
    does not carry. A check handed that file would read the old literal and
    answer for a cell nobody runs.
    """
    node, graph = cell.get("batch_node") or "", facts["graph"]
    if not cell.get("batch_toggled") or node not in graph:
        return graph
    inputs = {**graph[node]["inputs"], "batch_size": cell["batch"]}
    return {**graph, node: {**graph[node], "inputs": inputs}}


def label_cell(cell: dict, facts: dict, world: int) -> tuple[str, str, bool, list[str]]:
    """Return (label, basis, waiver, waivable guards), cheapest fatal rule first."""
    if cell.get("upstream_refusal"):
        # This is the loader's actual header predicate, ahead of every
        # topology and capacity question. A Z-Image family word does not
        # decide it: only the L2P layout (local_decoder and no dec_net) does.
        return "refuse:P", "guard", False, []
    if cell.get("upstream_gated"):
        # The adapter cannot run this artifact and the current comfy admits it
        # anyway, so a render here thrashes the host rather than refusing. No
        # cell of this template runs until upstream lands.
        return "skip:upstream-gated", "config", False, []
    if not cell["unets"]:
        return "skip:no-artifact", "disk", False, []
    family, quant, levers = cell["family"], cell["quant"], cell["levers"]
    if family == "unknown":
        # No family means no family gate armed. A render here proves nothing.
        return "skip:unknown-family", "disk", False, []
    adapter = ADAPTER_BY_FAMILY.get(family)
    if levers.get("comfy_managed") == "on" and (levers.get("slab_weights") == "on"
                                                or levers.get("lora_low_rss") == "on"):
        return "refuse:untyped", "guard", False, []
    try:
        topo = (choose_auto_topology(family, quant, facts["megapixels"], world,
                                     cfg_value=facts["cfg"],
                                     batch_size=cell["batch"]).topology
                if cell["preset"] == "auto" else topology_from_preset(cell["preset"], world))
    except Exception as exc:
        return _label_from(exc), "guard", False, []
    # ring_pad is waivable and fires on token counts only the driver knows.
    guards = ["ring_pad"] if topo.ring > 1 else []
    # The dual_model_cfg grant in actor/worker.py _inject_for_topology, clause
    # for clause. It reads the adapter flag and never the graph, so the uncond
    # loader is a separate question below.
    dual = (family in DUAL_MODEL_CFG_FAMILIES
            and topo.cfg == 2 and topo.ulysses * topo.ring == 1
            and world == 2 and not topo.fsdp)
    # actor/sample_protocol.py dual_model_cfg2_slot: one checkpoint per rank,
    # so the split guider needs the second loader this graph did not place.
    if dual and not facts["dual_model"]:
        return "refuse:P", "guard", False, guards
    # Load order first, sample order after. The header check runs on the driver
    # at the loader node; the worker then refuses comfy-managed residency before
    # it reads a weight, reads the live quant and the LoRA stack, and binds the
    # adapter last (nodes/loaders.py, actor/store_fsdp.py, actor/worker.py).
    #
    # The batch check is the first of them. nodes/loaders.py runs it above the
    # footprint card and above every residency policy, because no budget and
    # no consent can change an indivisible batch. It reads the literal batch out
    # of the graph the way the driver does, so a latent source the driver
    # cannot price makes no claim here either (2026-09-04: cell 2b172a2d45c8
    # scored FINDING for its label while the runtime answered class P).
    if topo.dp > 1 and (found := _guard(preflight_graph_batch_divides_dp,
                                        loader_mesh(cell["preset"], world),
                                        batch_graph(cell, facts))):
        return found, "guard", False, guards
    if topo.fsdp and (found := _guard(validate_fsdp_launch_quant,
                                      cell["launch_quant"] or "unknown")):
        return found, "guard", False, guards
    if levers.get("comfy_managed") == "on" and (cell["loras"] or topo.fsdp or facts["dual_model"]):
        return "refuse:P", "guard", False, guards
    if topo.fsdp:
        if found := next((label for token, label in BLOCK_SCALED.items()
                          if token in cell["artifact_quant"]), ""):
            return found, "name-token", False, guards
        # On an explicit FSDP preset the LoRA loader's driver-side preflight
        # (nodes/loaders.py _preflight_fsdp_lora_checkpoint_property) refuses a
        # stack on a comfy-kitchen quantized checkpoint (scaled fp8, fp8mixed,
        # int8, with or without convrot) before dispatch, whatever
        # lora_low_rss is set to. An auto preset skips that preflight, and the
        # worker backstop (fsdp_lora_admission.refuse_unless_fsdp_lora_admits,
        # called from adapters/fsdp.apply_fsdp_capacity_mode) answers the same
        # class from live evidence. Either way the label is class P.
        if cell["loras"] and cell.get("fsdp_lora_admission") and (
                found := _guard(refuse_fsdp_lora_checkpoint_property,
                                cell["fsdp_lora_admission"])):
            return found, "guard", False, guards
        # nodes/gate_identity.py authorize_normal_render authorizes every
        # dual-model dispatch with both residency levers off, so its LoRA stack
        # meets the worker check at lora_low_rss False whatever the lever asked.
        low_rss = False if facts["dual_model"] else \
            {"on": True, "off": False}.get(levers.get("lora_low_rss"))
        if found := _guard(validate_fsdp_launch_loras, cell["loras"], low_rss):
            return found, "guard", False, guards
        # actor/fsdp_lora.py's in-bake dtype check refuses class P when a patched
        # weight is stored fp32 but runs bf16. Whether comfy casts it is family
        # model code the header cannot show, so only a family measured to cast
        # takes the label (FSDP_LORA_LIVE_CAST_FAMILIES); the low-RSS guard above
        # answers first.
        if (cell["loras"] and cell.get("fsdp_lora_f32_patched")
                and family in FSDP_LORA_LIVE_CAST_FAMILIES):
            return "refuse:P", "header", False, guards
    if family == "omnigen2":
        if found := _guard(omnigen2._assert_ring_only_topology, topo.ulysses):
            return found, "guard", False, guards
    if adapter is not None:
        if found := (_guard(assert_cfg_parallel_supported, adapter, topo.cfg, dual)
                     or _guard(assert_sp_quant_supported, adapter,
                               topo.ulysses * topo.ring, quant)):
            return found, "guard", False, guards
    # The worker proves the kernel carries this family's head dimension before it
    # binds the sharded forward, substituting one that does where the table
    # offers it and refusing class P where nothing carries it.
    # ``resolved_kernel`` has already applied the substitution, so this asks the
    # published-limit half about the kernel that will run, and restates nothing.
    if adapter is not None and topo.ulysses * topo.ring > 1 and (
            found := _guard(attention_capability.assert_static_capability,
                            resolved_kernel(cell, facts, world), family,
                            adapter.attention_head_dim)):
        return found, "guard", False, guards
    # Sample time from here: nothing below is read until the sampler runs.
    if family == "minimax_h3" and minimax_h3.minimax_h3_topology_would_reject(topo.cfg, topo.dp):
        return _label_from(RuntimeError(minimax_h3.MINIMAX_H3_BATCH_CAPPED_MESSAGE)), \
            "guard", False, guards
    if facts["packed"] and (topo.cfg > 1 or topo.dp > 1):
        return "refuse:P", "guard", False, guards
    if topo.dp > 1 and cell["batch"] % topo.dp:
        # The graph named no literal the loader node could read, so the answer
        # arrives at the render submit instead, with the real tensor in hand
        # (nodes/render_validation.py validate_render_topology). Both sites
        # build one refusal, so the class is the same and the load is the only
        # difference.
        return _label_from(indivisible_batch_refusal(cell["preset"], topo, cell["batch"])), \
            "guard", False, guards
    # The sharded nvfp4 activation bar (adapters/quant_activation_scale.py). It
    # is read on the sample path before the sampler runs, so it answers ahead of
    # every wrapper below, and it reads the family's measurement rather than the
    # hook's coverage: a family in that table refuses on every topology the
    # shared scale reduces across, which is sp or cfg and never dp or dm-cfg2.
    # The guard name comes from the runtime's own table, so a second measured
    # family needs no edit here.
    shard_quant_guard = quant_activation_scale.SHARD_QUANT_GUARDS.get(family)
    if shard_quant_guard and "nvfp4" in cell["artifact_quant"] \
            and quant_activation_scale.plan_for_topology(
                {"ulysses": topo.ulysses, "ring": topo.ring, "cfg": topo.cfg},
                dual_model_cfg=dual):
        return "refuse:K", "guard", True, [*guards, shard_quant_guard]
    # At cfg 1.0 comfy runs no uncond pass, so the batched wrapper a cfg
    # topology installs never gets the two slices it splits. A cfg++ sampler
    # keeps the pass (sampler_keeps_uncond), so both ranks work. An fsdp cell
    # answers untagged before that tag is ever read: the FSDP authorization
    # gate refuses first, and under auto_gate=first_use the render that meets
    # this guard is that gate's own proof, whose abort denies every later cell
    # in the process.
    if topo.cfg > 1 and not dual and facts["cfg"] is not None \
            and abs(facts["cfg"] - 1.0) < 1e-6 and not facts["sampler_keeps_uncond"]:
        return "refuse:untyped" if topo.fsdp else "refuse:P", "guard", False, guards
    if topo.fsdp and facts["dual_model"] and not cell["loras"] \
            and cell["auto_gate"] == "first_use":
        return "refuse:untyped", "guard", False, guards
    # One rule reads the encoded prompt, which the graph does not carry: a
    # family that shards its text stream apart from its image stream needs
    # every text stream to divide the SP degree. It refuses typed, under FSDP
    # as well: the clean-reload proof settles a typed refusal inside itself, so
    # a cell at auto_gate first_use answers on attempt 1 the class it answers
    # at auto_gate off (2026-09-04: anima cfg2+fsdp cell ac07102eaf6e scored
    # FINDING for an untyped label). Where this rule sits against the sol block
    # below is unexercised: no family that shards text exactly takes a sol
    # kernel under the shipped sol_families.
    #
    # No cfg cell refuses for two prompts of different length. On a family that
    # pads no conditioning comfy can make two model calls of batch 1
    # (resolved_cfg_path says when), which the slice cannot cut, and the
    # per-cond dispatch (adapters/cfg_dispatch.py) catches that pair on every
    # family. Only the basis below says which path the cell was expected to take.
    basis = "guard"
    counts = sorted((cell.get("text_tokens") or {}).values())
    vouched = family in SP_PAD_VOUCHED_FAMILIES
    sp_rule = ((family in SP_EXACT_TEXT_FAMILIES or vouched)
               and topo.ulysses * topo.ring > 1)
    if topo.cfg > 1 and family not in DUAL_MODEL_CFG_FAMILIES and adapter is not None \
            and len(set(facts["prompts"].values())) > 1:
        # Two texts that differ. A family publishing the real caption length as
        # a per-cond constant is certain to dispatch; without one the cell
        # renders either way and the matrix cannot tell which path
        # (resolved_cfg_path says why).
        basis = ("per-cond-dispatch" if cfg_dispatches_per_cond(adapter)
                 else "cond-fold-unknown")
    if sp_rule:
        if not counts:
            # Nothing here could count the prompts (no tokenizer file for the
            # stream, weight syntax, or a prompt pair the walk could not
            # resolve), so the label stays what the other rules gave and the
            # basis says why.
            basis = "token-count-unavailable"
        else:
            basis = "token-count"
            padded = any(length % (topo.ulysses * topo.ring) for length in counts)
            if padded and not vouched:
                return "refuse:P", basis, False, guards
            if padded and topo.ring > 1:
                # The exclusion is a full-sequence point ulysses has and ring
                # does not, so a padded stream there takes the same waivable
                # card base.assert_ulysses_only_padding raises (guard ring_pad).
                return "refuse:K", basis, True, guards
    if sol_attention.is_sol_kernel(cell["attention"]):
        if topo.ulysses * topo.ring < 2 or topo.ring > 1:
            return "refuse:P", "guard", False, guards
        if found := _guard(sol_attention.assert_family_supported, family):
            return found, "guard", False, guards
        if not sol_attention.tau_is_vouched(family, sol_attention.parse_sol_tau(cell["attention"])):
            # The card names the guard, so the runner can match it.
            return "refuse:K", "guard", True, [*guards, "sol_attn"]
    return "render", basis, False, guards


ID_KEYS = ("template", "session", "preset", "mode", "gpus_per_host", "attention",
           "unets", "loras", "levers", "batch", "auto_gate")


def fp32_patched_count(cond: Path, config: dict, lora_names: list[str]) -> int:
    """fp32-stored checkpoint weights the named LoRA files patch, from headers only."""
    if not lora_names or not cond.is_file():
        return 0
    loras = config["comfy_dir"] / "models" / "loras"
    try:
        tensors = read_safetensors_header(str(cond)).tensors
    except Exception:
        return 0
    keys = patched_base_keys(tensors, [str(loras / name) for name in lora_names]) or set()
    return sum(tensors[key].dtype == "F32" for key in keys)


def capacity_risk_basis_for(family: str, checkpoint_gib: float,
                            artifact_quant: str = "", *, session: str = "",
                            world: int = 0, auto_gate: str = "", preset: str = "",
                            resident_te_gib: float = 0.0) -> str | None:
    """Identify empirical capacity rules that invalidate an earlier verdict.

    The ref-pose and 50 GiB rules set no basis, preserving existing resumability.
    The Flux2 fp8mixed boundary and local text-encoder overhead do set one, so
    records predating those rules rerun. Local mode shares a host between driver
    and worker: resident CLIP weights reduce memory available for the DiT. Cluster
    mode places the encoder on the driver and is excluded from that adjustment.

    These rules apply only to the recorded artifact, precision and run mode. They
    do not predict live free memory. See the 2026-09-28 local-memory record.
    """
    if (family == "flux2" and artifact_quant == "fp8mixed"
            and checkpoint_gib == FLUX2_FP8MIXED_CAPACITY_BOUNDARY_GIB
            and session == "cluster" and world == 2 and auto_gate == "first_use"
            and preset in FLUX2_FP8MIXED_CAPACITY_PRESETS):
        return "flux2-fp8mixed-33gib-observed"
    if (session == "local" and resident_te_gib > 0
            and checkpoint_gib < CAPACITY_RISK_GIB <= checkpoint_gib + resident_te_gib):
        return "local-stock-load-priced-with-resident-text-encoder"
    return None


def capacity_risk_for(family: str, checkpoint_gib: float,
                      artifact_quant: str = "", *, session: str = "",
                      resident_te_gib: float = 0.0, **context: object) -> bool:
    """Whether a capacity boundary may answer before the guard the label reads.

    The ref-pose rule reads the runtime's own family table
    (mesh_safety.REF_POSE_TOKEN_FAMILIES); the 50 GiB rule reads the file size
    (CAPACITY_RISK_GIB). The Flux2 and local text-encoder branches are observed
    boundaries and cannot price live memory beyond the bytes named.
    ``resident_te_gib`` prices into the 50 GiB line only for a local cell
    (capacity_risk_basis_for says why).
    """
    priced_te = resident_te_gib if session == "local" else 0.0
    return bool(family in REF_POSE_TOKEN_FAMILIES
                or checkpoint_gib + priced_te >= CAPACITY_RISK_GIB
                or capacity_risk_basis_for(family, checkpoint_gib, artifact_quant,
                                           session=session, resident_te_gib=resident_te_gib,
                                           **context))


def cell_id(cell: dict) -> str:
    payload = json.dumps({key: cell[key] for key in ID_KEYS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def resolved_kernel(cell: dict, facts: dict, world: int) -> str:
    """Return the attention kernel after the runtime's two substitutions.

    First, an auto row with SAGE enabled replaces non-sol kernels with SAGE_AUTO
    (nodes/common.resolve_sample_attention). Explicit presets have no auto row.
    Then sharded attention replaces a kernel that cannot handle the family's head
    dimension (adapters/attention_capability.effective_kernel). CFG and single
    modes keep stock per-rank attention and skip the second substitution.
    """
    sage, topo = False, None
    try:
        if cell["preset"] == "auto":
            row = choose_auto_topology(cell["family"], cell["quant"], facts["megapixels"],
                                       world, cfg_value=facts["cfg"],
                                       batch_size=cell["batch"])
            sage, topo = row.sage, row.topology
        else:
            topo = topology_from_preset(cell["preset"], world)
    except Exception:  # a row or preset that refuses; label_cell already scored it
        pass
    resolved = resolve_sample_attention(cell["attention"], sage)
    if topo is None or topo.ulysses * topo.ring < 2:
        return resolved
    adapter = ADAPTER_BY_FAMILY.get(cell["family"])
    return attention_capability.effective_kernel(
        resolved, cell["family"], getattr(adapter, "attention_head_dim", None))


def resolved_cfg_degree(cell: dict, facts: dict, world: int) -> int:
    """The cfg degree this cell runs, not the one its preset names.

    An "auto" row resolves against the family, the quant and the graph; an
    explicit preset is literal. A row that refuses resolves to 1, because
    label_cell already scored that refusal and no wall will be measured.
    """
    try:
        topo = (choose_auto_topology(cell["family"], cell["quant"], facts["megapixels"],
                                     world, cfg_value=facts["cfg"],
                                     batch_size=cell["batch"]).topology
                if cell["preset"] == "auto" else topology_from_preset(cell["preset"], world))
    except Exception:
        return 1
    return int(topo.cfg)


def resolved_cfg_path(cell: dict, facts: dict, world: int) -> str:
    """Predict ``slice``, ``dispatch``, ``either``, or ``none`` for CFG execution.

    ``slice`` splits one batched cond/uncond call across ranks; ``dispatch`` sends
    one non-concatenating conditioning to each rank. ``none`` means CFG degree is
    below two. ``either`` means the graph cannot establish the runtime path.

    Identical prompt text has matching shape. Different text proves dispatch only
    when the family exposes differing caption lengths as a per-cond constant.
    Otherwise ComfyUI may concatenate: CONDCrossAttn repeats unequal lengths when
    their LCM is at most four times the shorter, and padding families equalize
    lengths first. The runtime decides per step using the actual cond objects.
    """
    if int(cell.get("resolved_cfg") or 1) < 2:
        return "none"
    if len(set(facts["prompts"].values())) < 2:
        return "slice"
    adapter = ADAPTER_BY_FAMILY.get(cell["family"])
    if adapter is not None and cfg_dispatches_per_cond(adapter):
        return "dispatch"
    return "either"


def collapse_twins(cells: list[dict]) -> tuple[list[dict], int]:
    """Keep one cell per resolved kernel and move the references onto it."""
    groups: dict[str, list[dict]] = {}
    for cell in cells:
        groups.setdefault(json.dumps({**{name: cell[name] for name in ID_KEYS},
                                      "attention": cell["resolved_attention"]},
                                     sort_keys=True), []).append(cell)
    keep, moved = set(), {}
    for group in groups.values():
        # Never a sampled-out survivor: the run would skip the whole group.
        winner = next((cell for cell in group if not cell["sampled_out"]), group[0])
        keep.add(winner["id"])
        moved.update({cell["id"]: winner["id"] for cell in group if cell is not winner})
    kept = [cell for cell in cells if cell["id"] in keep]
    for cell in kept:
        # Only where a reference names one: a resident-leg spec carries no cell
        # key, and writing one would hand the runner a null to compare against.
        if cell["reference"].get("cell") in moved:
            cell["reference"]["cell"] = moved[cell["reference"]["cell"]]
    return kept, len(moved)


def _make(template: str, facts: dict, unets: dict[str, str], attention: str,
          preset: str, session: str, config: dict, **over) -> dict:
    # auto_gate is an Init widget the runtime reads and a field label_cell
    # reads, so it resolves once here rather than staying among the levers.
    levers = dict(over.pop("levers", {}))
    batch = over.pop("batch", facts["batch"])
    cell = {
        "template": template, "session": session, "preset": preset, "attention": attention,
        "mode": "local" if session == "local" else "cluster",
        "gpus_per_host": 1 if session == "local" else 0, "unets": unets,
        "loras": over.pop("loras", []), "levers": levers, "batch": batch,
        "auto_gate": levers.pop("auto_gate", over.pop("auto_gate", "first_use")),
        "klass": facts["klass"], "audio": facts["audio"], "probe": facts["probe"],
        # Why this cell has no probe leg, so the finding that scores its full
        # render says what it could not be measured against.
        "probe_reason": facts["probe_reason"],
        "batch_node": facts["batch_node"], "batch_toggled": batch != facts["batch"],
        "timeout_s": config["video_timeout_s"] if facts["klass"] == "video"
        else config["image_timeout_s"],
    }
    cell.update(over)
    capacity_context = {
        "session": cell["session"], "world": int(config.get("world") or 0),
        "auto_gate": cell["auto_gate"], "preset": cell["preset"],
        # Priced only for a local cell (capacity_risk_basis_for says why).
        "resident_te_gib": float(cell.get("resident_te_gib") or 0.0),
    }
    cell["capacity_risk_basis"] = capacity_risk_basis_for(
        cell["family"], float(cell.get("checkpoint_gib") or 0.0),
        cell.get("artifact_quant") or "", **capacity_context)
    cell["capacity_risk"] = capacity_risk_for(
        cell["family"], float(cell.get("checkpoint_gib") or 0.0),
        cell.get("artifact_quant") or "", **capacity_context)
    cell["id"] = cell_id(cell)
    return cell


def known_wrong_kernel(cell: dict) -> bool:
    """Whether this cell runs a kernel the runtime already knows to be wrong.

    Two readings of the same cell, because the name and the label can part. The
    declared kernel is a sol kernel, or the matrix labelled the cell for the
    class K sol-attn waiver, which is the guard that admits an approximate
    attention. A kernel that earns that waiver under some later name is still
    caught by the second reading.
    """
    return bool(sol_attention.is_sol_kernel(cell["attention"])
                or "sol_attn" in (cell.get("waiver_guards") or []))


def _numeric_bar(cell: dict, steps: int) -> str:
    """The bar for a comparison whose two sides may hold different numbers."""
    if known_wrong_kernel(cell):
        # The waiver admits the deviation, so measuring it is the whole cell.
        return WAIVED_BAR
    # A template the probe rewrite cannot cut renders a full trajectory, which
    # the one-step floor does not answer for (FULL_RENDER_BAR).
    return "step-nrms" if steps else FULL_RENDER_BAR


def _reference(cell: dict, auto_id: str, local_id: str, probe_steps: int,
               inert: frozenset = frozenset()) -> dict:
    """What this cell is scored against, and at which bar.

    The bar follows what the comparison can prove, and every lever cell earns
    one it can PASS: a bar no cell can pass reports no verdict.

    A residency toggle moves where the weights live, and an inert scheduling
    lever moves when the work runs. Neither changes an arithmetic result, so
    both are held to ``bit-identical`` on the full warm render against the auto
    cell, and a difference there is the finding the wave is looking for.

    Every other lever changes the arithmetic, so it is scored the way a
    cross-topology cell is: this cell's one-step probe leg against the auto
    cell's, at the floor the null A/B calibrated for one step. A cell holding
    both kinds takes that numeric bar; ``bit-identical`` needs every lever it
    holds to be inert or residency. Where a numeric bar applies, a sol kernel
    takes WAIVED_BAR.
    """
    steps = probe_steps if cell["probe"] else 0
    if "+fsdp" in cell["preset"]:
        return {"kind": "resident-leg", "preset": cell["preset"].replace("+fsdp", ""),
                "bar": "bit-identical"}
    if cell["levers"]:
        if set(cell["levers"]) <= (RESIDENCY_TOGGLES | inert):
            # Held to the pixel, so no probe leg: the full warm render is the
            # comparison, and one step would prove less of it.
            return {"kind": "cell", "cell": auto_id, "bar": "bit-identical"}
        return {"kind": "cell", "cell": auto_id, "bar": _numeric_bar(cell, steps),
                "probe_steps": steps}
    if cell["batch_toggled"]:
        # No cell renders the same frame count, so there is nothing to score.
        return {"kind": "none", "bar": "reference"}
    if cell["session"] == "local":
        # The reference renders the probe leg the cluster cells compare to.
        return {"kind": "none", "bar": "reference", "probe_steps": steps}
    return {"kind": "cell", "cell": local_id, "bar": _numeric_bar(cell, steps),
            "probe_steps": steps}


def build_cells(config: dict, count_tokens=None) -> tuple[list[dict], list[str], list[str]]:
    """Cells, pruned notes, and the checkpoint files no template claims.

    ``count_tokens`` is the prompt counter, so a caller with no tokenizer file
    on the box can hand its own.
    """
    graph_dir = config["out_dir"] / "graphs"
    model_dir = config["model_dir"]
    files = sorted(model_dir.glob("*.safetensors")) if model_dir.is_dir() else []
    count_tokens = count_tokens or TokenCounter(tokenizer_dir(config))
    cells: list[dict] = []
    pruned: list[str] = []
    claimed: set[str] = set()
    world = int(config["world"])
    probe_steps = int(config["reference"].get("probe_steps", 1))
    inert = frozenset(config["inert_levers"])
    shadowed: dict[tuple[str, str], list[str]] = {}
    for path in files:
        shadowed.setdefault(split_quant(path.stem), []).append(path.name)
    pruned += [f"{names[-1]} shadows {', '.join(names[:-1])} at the same base and quant"
               for names in shadowed.values() if len(names) > 1]
    for path in sorted(graph_dir.glob("*.json")):
        template = path.stem
        if template in config["exclude"]:
            continue
        graph = json.loads(path.read_text())
        facts = graph_facts(graph)
        if not facts["unets"]:
            pruned.append(f"{template}: no diffusion-model loader")
            continue
        te_gib = resident_text_encoder_gib(graph, config["comfy_dir"])
        absent = missing_media(graph, config["comfy_dir"] / "input")
        if absent:
            pruned.append(f"{template}: names media the input folder does not hold "
                          f"({', '.join(absent)}), so comfy rejects the graph and none "
                          "of its cells can run; add the files to the input folder, or "
                          "use its testing-only twin on synthetic media if it has one")
            continue
        sets, dropped = artifact_sets(facts["unets"], files,
                                      template_extras(config, template, facts))
        pruned.extend(f"{template}: {line}" for line in dropped)
        if not sets:
            sets = [{}]
        for names in sets:
            claimed.update(names.values())
        candidates = [config["model_dir"] / names[facts["cond_node"]] for names in sets if names]
        candidates.append(config["model_dir"] / facts["unets"][facts["cond_node"]])
        family = next((sniff_checkpoint_for_topology(path)[0]
                       for path in candidates if path.is_file()), "unknown")
        if family == "unknown":
            pruned.append(f"{template}: the header sniff names no family, so every "
                          "family gate stays off; the row is skipped")
        gated = template in config["upstream_gated"]
        if gated:
            pruned.append(f"{template}: upstream-gated by the sweep config, so its "
                          "cells are skipped; the adapter cannot run this artifact "
                          "and comfy admits it anyway")
        text_tokens = text_token_counts(facts["prompts"], family, count_tokens)
        if not facts["probe"]:
            # The reason names the node type, so the next template that lands
            # a scheduler of its own says which one to add to STEP_WIDGETS.
            pruned.append(f"{template}: no one-step probe leg, so the cross-topology score "
                          f"is two full warm renders at the {FULL_RENDER_BAR} bar; "
                          f"{facts['probe_reason']}")
        # Every kernel is derived; the ones outside `wanted` ride as sampled_out
        # rather than vanishing from the count.
        kernels = list(config["image_attention"])
        wanted = list(config["video_attention_full"].get(template)
                      or config["video_attention"]) if facts["klass"] == "video" else list(kernels)
        if family in config["sol_families"]:
            kernels += list(config["sol_kernels"])
            wanted += list(config["sol_kernels"])
        kernels += [kernel for kernel in wanted if kernel not in kernels]
        presets = config["video_presets"] if facts["klass"] == "video" \
            else config["image_presets"]
        lora_names = config["loras"].get(template, [])
        if lora_names and len(lora_names) != len(facts["loras"]):
            # A short list would drop loaders out of the graph and score the row
            # as a stack the template never places.
            pruned.append(f"{template}: {len(facts['loras'])} LoRA loader(s) against "
                          f"{len(lora_names)} configured name(s), so the lora-on arm "
                          "is dropped: " + ", ".join(lora_names))
            lora_names = []
        stack = [{"node": nid, "name": name, "strength": 0.8}
                 for nid, name in zip(facts["loras"], lora_names, strict=True)] \
            if facts["loras"] and lora_names else []
        # A "-lora" template is its family's shipped graph plus LoRA loaders, so
        # its lora-off arm renders what the shipped graph's row already renders.
        # dgx-monarch-test-chroma-lora sets the 44-token prompt pair instead, so
        # its lora-off arm renders what the dgx-monarch-test-chroma-even row
        # already renders. The [lora_off] table keeps an arm on request.
        keep_off = config["lora_off"].get(template, not template.endswith("-lora"))
        lora_options: list[list[dict]] = [[]] if keep_off or not stack else []
        if stack:
            lora_options.append(stack)
        if stack and not keep_off:
            pruned.append(f"{template}: lora-off arm dropped by the [lora_off] table or, "
                          "for a -lora template, by default; set the template true in "
                          "[lora_off] to keep it")
        for names in sets:
            cond = config["model_dir"] / names.get(facts["cond_node"], "")
            row = dict(facts)
            common = {
                "family": family,
                "quant": sniff_checkpoint_for_topology(cond)[1] if cond.is_file() else "bf16",
                "launch_quant": sniff_fsdp_launch_quant(cond) if cond.is_file() else None,
                "artifact_quant": split_quant(cond.stem)[1] if names else "",
                # The size the load-time capacity preflight prices, not a label:
                # it is out of ID_KEYS, so no cell id moves when a file is swapped.
                "checkpoint_gib": round(cond.stat().st_size / 2**30, 1)
                if cond.is_file() else 0.0,
                # The resident CLIP loader's GiB, priced into a local cell's
                # capacity risk alongside checkpoint_gib; one figure per
                # template, so it rides beside every artifact set the same way.
                "resident_te_gib": te_gib,
                # The header-only property that keeps a LoRA stack off this
                # checkpoint under FSDP
                # (adapters/detect.fsdp_lora_admission_property), or None when
                # admitted. Read once per artifact set so label_cell restates
                # no marker of its own.
                "fsdp_lora_admission": fsdp_lora_admission_property(cond)
                if cond.is_file() else None,
                # How many fp32-stored weights the template's LoRA stack patches:
                # the worker's in-bake check refuses those class P under FSDP.
                "fsdp_lora_f32_patched": fp32_patched_count(cond, config, lora_names),
                # Also out of ID_KEYS, so a new count moves no settled cell id.
                "text_tokens": text_tokens,
                # Read the exact header predicate the loader calls. Retain
                # only its boolean answer in the journal: local model paths
                # do not belong in sweep records.
                "upstream_refusal": upstream_gate.refuses(str(cond)),
                "upstream_gated": gated,
            }
            base_local = [_make(template, row, names, kernels[0], "single", "local",
                                config, loras=loras, **common)["id"]
                          for loras in lora_options]
            for attention in kernels:
                sampled = attention not in wanted
                for index, loras in enumerate(lora_options):
                    local = _make(template, row, names, attention, "single", "local",
                                  config, loras=loras, **common)
                    auto = _make(template, row, names, attention, "auto", "cluster",
                                 config, loras=loras, **common)
                    batch = [local, auto]
                    batch += [_make(template, row, names, attention, preset, "cluster",
                                    config, loras=loras, **common)
                              for preset in presets if preset != "auto"]
                    for name, values in config["toggles"].items():
                        if name == "batch_size" and not facts["batch_widget"]:
                            continue
                        # A lever rides "auto" unless the config names the
                        # presets it belongs on. Auto never selects FSDP
                        # (nodes/common.py), so the FSDP rules fire only on a
                        # preset that names it.
                        for over in config["toggle_presets"].get(name, ["auto"]):
                            if over != "auto" and over not in presets:
                                continue
                            batch += [_make(template, row, names, attention, over, "cluster",
                                            config, loras=loras, **common,
                                            **({"batch": value} if name == "batch_size"
                                               else {"levers": {name: value}}))
                                      for value in values]
                    batch += [_make(template, row, names, attention,
                                    combo.get("preset", "auto"), "cluster", config,
                                    loras=loras, **common,
                                    levers={key: value for key, value in combo.items()
                                            if key != "preset"})
                              for combo in config["toggle_combos"]
                              if combo.get("preset", "auto") in ("auto", *presets)]
                    # A sol kernel refuses at world 1, so a sol cell takes the
                    # base-kernel local render as its reference.
                    anchor = base_local[index] if sol_attention.is_sol_kernel(attention) else local["id"]
                    for cell in batch:
                        cell["sampled_out"] = sampled
                        world_here = 1 if cell["session"] == "local" else world
                        cell["label"], cell["basis"], cell["waiver"], \
                            cell["waiver_guards"] = label_cell(cell, row, world_here)
                        cell["reference"] = _reference(cell, auto["id"], anchor,
                                                       probe_steps, inert)
                        cell["resolved_attention"] = resolved_kernel(cell, row, world_here)
                        cell["resolved_cfg"] = resolved_cfg_degree(cell, row, world_here)
                        cell["cfg_path"] = resolved_cfg_path(cell, row, world_here)
                    cells.extend(batch)
    cells, collapsed = collapse_twins(cells)
    if collapsed:
        pruned.append(f"{collapsed} cells collapsed onto a twin: the auto row's sage "
                      "flag, or a head dimension the declared kernel cannot "
                      "carry, resolves their declared kernels to one")
    unmatched = sorted({path.name for path in files} - claimed)
    return cells, pruned, unmatched


def run_matrix_module():
    """The committed redactor, loaded by path so it parses no argv of ours."""
    spec = importlib.util.spec_from_file_location(
        "dgxm_sweep_run_matrix", REPO / "benchmark" / "run_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def counts_table(cells: list[dict]) -> str:
    lines = []
    for axis in ("label", "klass", "family"):
        lines += [f"\n### by {axis}\n", "| value | cells |", "|---|--:|"]
        lines += [f"| {value} | {count} |" for value, count
                  in sorted(Counter(str(cell[axis]) for cell in cells).items())]
    lines.append(f"\ntotal {len(cells)} cells, "
                 f"{sum(1 for cell in cells if cell['sampled_out'])} sampled out\n")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.sweep.matrix",
        description="Derive and label the sweep cells. Writes cells.jsonl and a counts table.")
    parser.add_argument("--config", required=True, help="sweep TOML (see sweep.example.toml)")
    parser.add_argument("--dry-run", action="store_true", help="derive from the graphs on disk and print the counts only")
    parser.add_argument("--convert", action="store_true", help="convert the templates first (driver if up, repo widget map if not)")
    args = parser.parse_args(argv)
    # Auto resolution logs a row per call, thousands of times over a matrix.
    for name, logger in logging.root.manager.loggerDict.items():
        if name.startswith("dgx_monarch") and isinstance(logger, logging.Logger):
            logger.setLevel(logging.WARNING)
    config = load_config(Path(args.config))
    if args.convert:
        from .convert import convert_dir, widget_source

        names, source = widget_source(config["driver"])
        graphs, failed = convert_dir(config["template_dirs"], config["out_dir"] / "graphs", names)
        print(f"converted {len(graphs)} templates with the {source} widget map"
              + "".join(f"\nnot converted: {line}" for line in failed))
    try:
        cells, pruned, unmatched = build_cells(config)
    except ConfigError as exc:
        parser.error(str(exc))
    table = counts_table(cells)
    table += f"\n### pruned ({len(pruned)})\n\n" + "\n".join(f"- {line}" for line in pruned)
    table += (f"\n\n### unmatched checkpoint files ({len(unmatched)})\n\n"
              + "\n".join(f"- {name}" for name in unmatched) + "\n")
    print(table)
    if args.dry_run:
        return 0
    (out := config["out_dir"]).mkdir(parents=True, exist_ok=True)
    with open(out / "cells.jsonl", "w") as handle:
        for cell in cells:
            handle.write(json.dumps(cell, sort_keys=True) + "\n")
    # The counts table is text an operator pastes elsewhere, so it passes the
    # redactor and the write refuses when a private value survives.
    run_matrix = run_matrix_module()
    payload = {"model_dir": str(config["model_dir"]), "out_dir": str(out),
               "comfy_dir": str(config["comfy_dir"]), "driver": config["driver"],
               "sibling": config["sibling"], "table": table}
    secrets = list(run_matrix.private_path_secrets(payload))
    table = run_matrix.sanitize_report(payload)["table"]
    if leaked := run_matrix.leaked_secrets(table, secrets):
        parser.error(f"{len(leaked)} private value(s) survive in the counts table")
    (out / "cells_counts.md").write_text(table)
    print(f"wrote {out / 'cells.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
