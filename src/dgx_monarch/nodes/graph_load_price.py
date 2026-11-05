"""The loader preflight's words: the refusal, its plain lead, and what would fit.

``loader_preflight`` prices the windows; this module says what they mean. Under
``auto`` the UNET loader defers the load to the first render, after the driver
has loaded every model the graph loads, so that load window carries them
(docs/DESIGN.md section 5.9, "Two windows, not one sum"). A refusal leads with
one plain sentence, the figures inside the first 200 characters, then names
what would fit: same-stem files on disk here in place of the DiT or of a driver
model, a lazy switch that skips a model, and a pair's ``uly2+fsdp`` preset,
each tried in every window before it is named. No ComfyUI import at module
scope; a fault while naming alternatives names none, and the refusal stands.
"""
from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from typing import Any

from .. import consent_pending, driver_footprint
from ..log import get_logger
from .graph_live import SWITCH_TITLES, is_link, live_nodes
from .loader_graph import TE_RESIDENT_RATIO, DriverStackTerms

log = get_logger(__name__)

_GIB = 2 ** 30
RESCUE_KIND = "rescue-slab"
WINDOW_TRANSIENT = "load"
WINDOW_SETTLED = "resident"
DIT_FOLDERS = ("diffusion_models", "unet")
_TE_FOLDERS = ("text_encoders", "clip", "clip_vision")
_VAE_FOLDERS = ("vae",)
_MODEL_FOLDERS = ("checkpoints", "diffusion_models", "unet")
# The streaming shard build's kinds (adapters/fsdp.py): only these halve the
# DiT on a pair, so only these name the pair's preset.
_SHARDED_KINDS = frozenset({"bf16", "fp16"})
PAIR_PRESET = "uly2+fsdp"
# A precision or quantization token at the end of a file stem: the files that
# differ only here are one model in another precision, as MODELS lists them.
_PRECISION = re.compile(
    r"[-_.](?:bf16|fp16|fp32|f16|f32|fp8(?:[-_](?:e4m3fn|e5m2|scaled|mixed))*|"
    r"fp8mixed|int8(?:[-_]convrot)?|int4|w4a8|w4a16|w8a8|nvfp4|mxfp8|"
    r"q\d(?:_k)?(?:_[msl])?)$", re.IGNORECASE)
_MAX_TWINS = 3
_MAX_MODELS = 2
# The most this text may take of refusal()'s 4,000 characters: the sentences a
# raise site adds after it (an Old model card's, a quarantine's, a session
# blocker's, the panel's) stay under 900, so past this the alternatives go.
TEXT_BUDGET = 3000


def _gib(value: float) -> str:
    return f"{value / _GIB:.1f}"


def _join(items: Iterable[str], word: str = "and") -> str:
    items = list(items)
    if len(items) < 2:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {word} " + items[-1]


def load_is_deferred(mesh: Any) -> bool:
    """Whether the UNET loader defers this load to the first render.

    Under ``auto`` it does (``nodes/loaders.py``, ``first_render.load_deferred``):
    the topology is resolved per render, so the load waits for it, and by then
    every node upstream of the sampler has run.
    """
    return str(getattr(mesh, "topology_preset", "auto") or "auto") == "auto"


def rescue_clause(estimate: driver_footprint.DriverFootprintEstimate) -> str:
    """The rescue evidence and projected footprint; ``refusal()`` adds the panel words."""
    spec = consent_pending.KIND_SPECS.get(RESCUE_KIND)
    evidence = getattr(spec, "evidence", None) or (
        "Slab residency maps the weights from the file instead of copying them.")
    return f"{evidence} It fits this graph at {_gib(estimate.projected)} GiB."


def unstable_identity_clause(names: tuple[str, ...]) -> str:
    """Why a refused load with no readable artifact identity gets no rescue."""
    return (f"Slab residency cannot be offered here either: this host cannot read a "
            f"stable identity for {', '.join(names)}, so no identity-gate row and no "
            "consent can be tied to these bytes. A checkpoint still being copied in, "
            "or rewritten while it was read, has no fingerprint to bind. Wait for the "
            "copy to finish, then queue the render again.")


def _budget_note(window: str, terms: DriverStackTerms, load_models: int | None) -> str:
    if window != WINDOW_TRANSIENT:
        return terms.budget_clause
    if load_models is None:
        return ("the driver stack is not in this window: the eager load peaks before "
                "the text encoder, the VAEs and the reference encodes are built")
    credit = f" ({terms.credit_reason})" if terms.credited else ""
    return (f"{_gib(load_models)} GiB for the driver models this graph loads before its "
            f"first render, where an auto topology loads the DiT{credit}")


def loader_refusal(
    estimate: driver_footprint.DriverFootprintEstimate, terms: DriverStackTerms, *,
    unet_name: str, profile: driver_footprint.DriverStackProfile, floor_reserve: int,
    rescue: driver_footprint.DriverFootprintEstimate | None, window: str = WINDOW_SETTLED,
    bypass: bool = True, lead: str = "", fits: str = "", load_models: int | None = None,
) -> str:
    """A capacity refusal with its charged terms and remedies, stripped at both ends.

    ``lead`` is the plain sentence and ``fits`` what would fit; the budget
    figures follow, for a reader who wants the terms. ``fits`` is left out when
    the whole would pass ``TEXT_BUDGET``. ``load_models`` is the driver models a
    deferred load window carries, None for an eager load.
    """
    notes = estimate.notes
    ref_note = (f"{terms.video_refs} reference video block(s), single-shape "
                f"calibration 2026-08-04, not extrapolated"
                if terms.ref_bytes else
                ("no reference video in this graph" if terms.resolved
                 else "no graph available at the loader node"))
    models = (f"; driver-side models {_gib(terms.model_bytes)} GiB "
              f"({', '.join(terms.model_artifacts)})" if terms.model_bytes else "")
    body = (
        f"driver-side footprint preflight at the loader node: "
        f"{_gib(estimate.projected)} GiB to place on the driver host, "
        f"{_gib(estimate.usable)} GiB usable ({_gib(estimate.mem_available)} GiB "
        f"MemAvailable minus a {_gib(floor_reserve)} GiB reserve and "
        f"{_budget_note(window, terms, load_models)}). Refuses {unet_name} at the {window} "
        f"window. Charged: DiT weights "
        f"{_gib(estimate.weight_bytes)} GiB ({notes.get('weights', 'not charged')}); "
        f"stock host copy {_gib(estimate.arena_bytes)} GiB "
        f"({notes.get('arena', 'not charged')}); text encoder "
        f"{_gib(terms.te_bytes)} GiB ({', '.join(terms.te_artifacts) or 'none named'}, "
        f"at the 1.15 resident ratio, rounded up from the 1.144 measured on 2026-08-05)"
        f"{models}; VAEs {_gib(terms.vae_bytes)} GiB "
        f"({', '.join(terms.vae_artifacts) or 'none named'}); reference encode "
        f"{_gib(terms.ref_bytes)} GiB ({ref_note}); guide encode "
        f"{_gib(terms.guide_bytes)} GiB ({terms.guide_note}); render activations "
        f"{_gib(terms.activation_bytes)} GiB ({terms.activation_note}). "
        f"{driver_footprint.what_would_fit(estimate, profile)}"
        f"Options: {'; '.join(profile.fitting_options)}. "
        # Name the kill switch only when no rescue card or stronger quarantine
        # applies. refusal() appends panel and per-kind headless instructions.
        + (rescue_clause(rescue) if rescue is not None else
           f"Set {driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV}=1 to turn this "
           f"preflight off for every render (docs/TROUBLESHOOTING.md #52)." if bypass else ""))
    if len(lead) + len(fits) + len(body) > TEXT_BUDGET:
        fits = ""
    return " ".join(part.strip() for part in (lead, fits, body) if part.strip())


def plain_lead(estimate: driver_footprint.DriverFootprintEstimate, *, window: str,
               unet_name: str, models: tuple[str, ...], deferred: bool) -> str:
    """One plain sentence: what does not fit, by how much, and why at this window."""
    name = os.path.basename(unet_name)
    subject = (f"{name} needs" if estimate.weight_bytes
               else f"Beside {name}, this graph's driver models need")
    if window == WINDOW_TRANSIENT and deferred and models:
        why = (f"It loads at the first render, beside {_join(models)}, which this graph "
               "loads on the driver first.")
    elif window == WINDOW_TRANSIENT:
        why = ("That is the peak of its load, the weights beside the host copy they are "
               "placed from." if estimate.arena_bytes else "That is the peak of its load.")
    else:
        why = "That is the DiT in memory beside what this graph still needs on the driver."
    return (f"{subject} {_gib(estimate.projected + estimate.reserve)} GiB on this box and "
            f"{_gib(estimate.mem_available)} GiB is available, {_gib(estimate.shortfall)} "
            f"GiB short. {why} Nothing was loaded.")


def _stem(name: str) -> str | None:
    match = _PRECISION.search(os.path.splitext(os.path.basename(name))[0])
    return os.path.splitext(os.path.basename(name))[0][:match.start()] if match else None


def twins_on_disk(folders: tuple[str, ...], name: str, below: int) -> list[tuple[str, int]]:
    """Files on disk here that hold the same model in another precision, under ``below`` bytes.

    Same folder and subfolder, same extension, and the same stem once a
    precision token is taken off the end. Largest first, the closest copy.
    """
    stem = _stem(name)
    if not stem:
        return []
    import folder_paths

    where, ext = os.path.dirname(name), os.path.splitext(name)[1].lower()
    found: dict[str, int] = {}
    for folder in folders:
        try:
            listed = folder_paths.get_filename_list(folder)
        except Exception:  # an unknown folder lists nothing
            continue
        for other in listed:
            if (other == name or other in found or os.path.dirname(other) != where
                    or os.path.splitext(other)[1].lower() != ext or _stem(other) != stem):
                continue
            size = driver_footprint.file_size_bytes(folder_paths.get_full_path(folder, other))
            if 0 < size < below:
                found[other] = size
    return sorted(found.items(), key=lambda item: -item[1])


def _same_model(path: str, family: str) -> str | None:
    """The twin's quant kind when its header reads the same family, else None."""
    from ..adapters.detect import sniff_checkpoint

    try:
        sniffed, kind = sniff_checkpoint(path)
    except Exception:  # a header that cannot be read is never named
        return None
    return kind if sniffed == family else None


def _dit_twins(shape: Any, fits: Callable[[Any], bool], unet_name: str,
               family: str) -> list[str]:
    import folder_paths

    named = []
    for twin, size in twins_on_disk(DIT_FOLDERS, unet_name, shape.weight_bytes):
        kind = _same_model(str(folder_paths.get_full_path("diffusion_models", twin)), family)
        world = shape.fsdp_world if kind in _SHARDED_KINDS else 0
        if kind is not None and fits(replace(shape, weight_bytes=size, fsdp_world=world)):
            named.append(f"{os.path.basename(twin)} ({_gib(size)} GiB)")
        if len(named) == _MAX_TWINS:
            break
    return named


def _model_twins(shape: Any, fits: Callable[[Any], bool], terms: DriverStackTerms,
                 deferred: bool) -> list[str]:
    from .loader_graph import _artifact_bytes

    clauses = []
    rows: list[tuple[str, tuple[str, ...], float]] = [
        (name, _TE_FOLDERS, TE_RESIDENT_RATIO) for name in terms.te_artifacts]
    rows += [(name, _MODEL_FOLDERS, 1.0) for name in terms.model_artifacts]
    rows += [(name, _VAE_FOLDERS, 1.0) for name in terms.vae_artifacts]
    for name, folders, ratio in rows if terms.models_bytes else ():
        current = _artifact_bytes(folders, name)
        named = []
        for twin, size in twins_on_disk(folders, name, current):
            delta = int((size - current) * ratio)
            if fits(replace(shape, stack_bytes=shape.stack_bytes + delta,
                            load_stack_bytes=shape.load_stack_bytes + (delta if deferred else 0))):
                named.append(f"{os.path.basename(twin)} ({_gib(size)} GiB)")
            if len(named) == _MAX_TWINS:
                break
        if named:
            clauses.append(f"{_join(named, 'or')} in place of {os.path.basename(name)}")
        if len(clauses) == _MAX_MODELS:
            break
    return clauses


def _switch_clause(shape: Any, fits: Callable[[Any], bool], terms: DriverStackTerms,
                   prompt: object, stack_for: Callable[[object], DriverStackTerms],
                   deferred: bool) -> str:
    """The first literal switch whose other setting skips a priced model and fits."""
    if not isinstance(prompt, Mapping) or not terms.models_bytes:
        return ""
    live = live_nodes(prompt)
    for key, node in prompt.items():
        inputs = node.get("inputs") if isinstance(node, Mapping) else None
        if (str(key) not in live or node.get("class_type") not in SWITCH_TITLES
                or not isinstance(inputs, Mapping) or not isinstance(inputs.get("switch"), bool)
                or not all(is_link(inputs.get(side)) for side in ("on_true", "on_false"))):
            continue
        flipped = {**prompt, key: {**node, "inputs": {**inputs, "switch": not inputs["switch"]}}}
        other = stack_for(flipped)
        dropped = [name for name in terms.models if name not in other.models]
        load_delta = other.models_bytes - terms.models_bytes
        if dropped and fits(replace(
                shape, stack_bytes=shape.stack_bytes + other.total - terms.total,
                load_stack_bytes=shape.load_stack_bytes + (load_delta if deferred else 0))):
            return (f"{SWITCH_TITLES[node['class_type']]} (node {key}) set to "
                    f"{str(not inputs['switch']).lower()} would fit too: the graph then does "
                    f"not load {_join(os.path.basename(name) for name in dropped)}.")
    return ""


def _pair_clause(shape: Any, fits: Callable[[Any], bool], quant_kind: str) -> str:
    if (shape.fsdp_world or not shape.co_resident or shape.weights_resident
            or quant_kind not in _SHARDED_KINDS):
        return ""
    if not fits(replace(shape, fsdp_world=2)):
        return ""
    share = driver_footprint.fsdp_shard_build_bytes(shape.weight_bytes, 2)
    return (f"On a pair of Sparks the {PAIR_PRESET} preset would fit: this box would hold "
            f"{_gib(share)} GiB of the DiT, where Ulysses and ring hold all of it, and an "
            f"FSDP render needs an Identity Gate PASS on {PAIR_PRESET} first.")


def alternatives(shape: Any, *, fits: Callable[[Any], bool], unet_name: str, family: str,
                 quant_kind: str, terms: DriverStackTerms, deferred: bool, prompt: object,
                 stack_for: Callable[[object], DriverStackTerms], pair: bool) -> str:
    """What would fit in place of what does not; empty when nothing does.

    ``fits`` prices a changed shape in every window. Best effort: a fault names
    nothing and the refusal stands without it.
    """
    try:
        swaps = []
        if shape.co_resident and not shape.weights_resident:
            dit = _dit_twins(shape, fits, unet_name, family)
            if dit:
                swaps.append(f"{_join(dit, 'or')} in place of {os.path.basename(unet_name)}")
        swaps += _model_twins(shape, fits, terms, deferred)
        parts = [f"On disk here, these would fit: {'; '.join(swaps)}."] if swaps else []
        parts.append(_switch_clause(shape, fits, terms, prompt, stack_for, deferred))
        parts.append(_pair_clause(shape, fits, quant_kind) if pair else "")
        return " ".join(part for part in parts if part)
    except Exception as exc:  # naming alternatives never replaces the refusal
        log.warning("loader preflight could not name what would fit (%r)", exc)
        return ""
