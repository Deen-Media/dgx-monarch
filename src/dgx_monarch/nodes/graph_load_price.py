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
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from .. import box_names, consent_pending, driver_footprint
from ..log import get_logger
from .graph_live import SWITCH_TITLES, is_link, live_nodes
from .graph_twins import _SHARDED_KINDS, _gib, _join, dit_twins, model_twins
from .loader_graph import DriverStackTerms, credited_terms, remember

log = get_logger(__name__)

_GIB = 2 ** 30
RESCUE_KIND = "rescue-slab"
# The driver models the lead names before it says how many more.
_MAX_NAMED = 4
WINDOW_TRANSIENT = "load"
WINDOW_SETTLED = "resident"
PAIR_PRESET = "uly2+fsdp"
# The most this text may take of refusal()'s 4,000 characters, beside what a
# raise site adds after it (an Old model card's, a quarantine's, a session
# blocker's, the panel's). Past it the budget terms are cut in the middle, and
# the alternatives go only when fewer than _MIN_TERMS characters of terms would
# stay; the lead and the remedy always stay. ``bounded`` caps the whole.
TEXT_BUDGET = 3000
_MIN_TERMS = 800
# Where the price runs: the UNET loader, or the first node that renders a queue
# whose loader output ComfyUI kept from an earlier queue (nodes/requeue_price.py):
# a sampler, or Identity Gate (DGX Monarch).
SITE_LOADER = "loader"
SITE_SAMPLER = "sampler"
SITE_GATE = "gate"
RENDER_SITES = frozenset({SITE_SAMPLER, SITE_GATE})
# Each render site's node as the words name it; read-only after import.
_SITE_NODES = {SITE_SAMPLER: "the sampler", SITE_GATE: "Identity Gate (DGX Monarch)"}
# A held encoder's resident size, as measured on 2026-08-05: the price charges it
# at loader_graph.TE_RESIDENT_RATIO, this rounded up, and credits it no higher.
HELD_ENCODER_RATIO = 1.144


def _site_words(site: str) -> str:
    node = _SITE_NODES.get(site)
    return (f"at {node} (the UNET Loader's output came from an earlier queue)" if node
            else "at the loader node")
# Session lifetime, FIFO-capped through loader_graph.remember: the ComfyUI prompt
# id and checkpoint of each price that ran, so a sampler prices only a prompt
# whose loader did not. A lost entry prices a prompt twice; it never skips one.
_PRICED_PROMPTS: dict[str, None] = {}


def site_terms(terms: DriverStackTerms, site: str) -> DriverStackTerms:
    """The terms the price charges at ``site``, held models credited.

    At a render site the graph's encoders have run, so ComfyUI's held bytes are
    read as encoders at their measured resident size, which MemAvailable already
    counts. The encodes have run too, and the render-site checks that run next
    price the activations, crediting what the Workers retain and freeing what
    ComfyUI can reload (``render_preflight``), so charging them here would refuse
    a re-queue the box just rendered.
    """
    if site not in RENDER_SITES:
        return credited_terms(terms)
    return replace(credited_terms(terms, HELD_ENCODER_RATIO), ref_bytes=0, guide_bytes=0,
                   activation_bytes=0, guide_note=f"encoded before {_SITE_NODES[site]}",
                   activation_note="the render-site checks price them next")


def log_alternatives(terms: DriverStackTerms, unet_name: str, avail: int, sum_fits: bool) -> None:
    """The one line an admitted load that took the alternative charge leaves (issue #187).

    The admitting path prints no card, so the journal says what charging the
    guide encode and the render activations as alternatives changed: what the
    sum would have said of the same load at the same reading.
    """
    log.info("loader footprint preflight: %s admitted with its %s GiB guide encode and %s GiB of render "
             "activations charged as alternatives (%s); charged together they would %s; MemAvailable %s GiB",
             unet_name, _gib(terms.guide_bytes), _gib(terms.activation_bytes), terms.guide_order_note,
             "fit too" if sum_fits else "have refused it", _gib(avail))


def load_window_models(terms: DriverStackTerms, site: str) -> tuple[int, tuple[str, ...]]:
    """The driver models a deferred load meets, in bytes and by name.

    At a render site a VAE still charged is not held, so it loads for the decode,
    after the DiT places: the settled window carries it, and the load window not.
    """
    if site not in RENDER_SITES:
        return terms.models_bytes, terms.models
    return terms.models_bytes - terms.vae_bytes, (*terms.te_artifacts, *terms.model_artifacts)


def executing_prompt_id() -> str | None:
    """The ComfyUI prompt this thread is executing, or None (headless, no ComfyUI)."""
    try:
        from comfy_execution.utils import get_executing_context

        prompt_id = getattr(get_executing_context(), "prompt_id", None)
    except Exception:
        return None
    return prompt_id if isinstance(prompt_id, str) and prompt_id else None


def note_priced(unet_name: str) -> None:
    """Record that the graph price ran for this checkpoint in the executing prompt."""
    prompt_id = executing_prompt_id()
    if prompt_id is not None:
        remember(_PRICED_PROMPTS, f"{prompt_id}\x00{unet_name}")


def priced_this_prompt(unet_name: str) -> bool | None:
    """Whether the graph price ran for this checkpoint in this prompt; None without a prompt id."""
    prompt_id = executing_prompt_id()
    return None if prompt_id is None else f"{prompt_id}\x00{unet_name}" in _PRICED_PROMPTS


def driver_box(mesh: Any) -> str:
    """The driver's box in ``box_names``'s words: the head, or this box when the
    fleet runs on the driver's box alone (a local fleet, or a cluster.toml that
    lists only this box). The loader's own sentences name no rank, host or address.
    """
    handle = getattr(mesh, "handle", None)
    if handle is None or getattr(handle, "owns_hosts", True) is False:
        return box_names.LOCAL
    hosts = [str(getattr(host, "name", "") or "")
             for host in getattr(getattr(handle, "config", None), "hosts", None) or ()]
    words = box_names.host_words(hosts, driver_footprint.host_is_local) if hosts else []
    return box_names.LOCAL if not words or set(words) == {box_names.LOCAL} else box_names.HEAD


def load_is_deferred(mesh: Any) -> bool:
    """Whether the UNET loader defers this load to the first render.

    Under ``auto`` it does (``nodes/loaders.py``, ``first_render.load_deferred``):
    the topology is resolved per render, so the load waits for it, and by then
    every node upstream of the sampler has run.
    """
    return str(getattr(mesh, "topology_preset", "auto") or "auto") == "auto"


def cast_copy_bytes(path: str, options: Mapping[str, Any] | None, *, mesh: Any,
                    slab: object) -> int | None:
    """The copy a loader dtype cast builds, priced as the Worker's stock rung does.

    0 for no cast. None where no price claims a cast (docs/DESIGN.md section
    5.9): slab, comfy-managed and a declared shard build stage a cast in ways
    no leg has measured, and a dtype this build cannot read sizes nothing.
    """
    if not driver_footprint.dtype_cast_requested(options):
        return 0
    from ..topology import declares_fsdp

    if slab is not False or declares_fsdp(getattr(mesh, "topology_preset", "auto")):
        return None
    from .. import capacity_fit
    from ..loader_options import weight_dtype_from_options

    try:
        dtype = weight_dtype_from_options(options)
    except (TypeError, ValueError):
        return None
    return capacity_fit.cast_resident_bytes(path, dtype)


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
    windows: tuple[driver_footprint.DriverFootprintEstimate, ...] = (), cast: bool = False,
    site: str = SITE_LOADER,
) -> str:
    """A capacity refusal with its charged terms and remedies, stripped at both ends.

    ``lead`` is the plain sentence and ``fits`` what would fit; the budget
    figures follow, for a reader who wants the terms. ``fits`` is left out when
    the whole would pass ``TEXT_BUDGET``. ``load_models`` is the driver models a
    deferred load window carries, None for an eager load. ``windows`` are both
    windows, so the DiT ceiling quoted is one that clears every failing window.
    """
    notes = estimate.notes
    ref_note = (f"{terms.video_refs} reference video block(s), single-shape "
                f"calibration 2026-08-04, not extrapolated"
                if terms.ref_bytes else
                f"encoded before {_SITE_NODES[site]}" if site in RENDER_SITES and terms.video_refs else
                ("no reference video in this graph" if terms.resolved
                 else "no graph available at the loader node"))
    models = (f"; driver-side models {_gib(terms.model_bytes)} GiB "
              f"({', '.join(terms.model_artifacts)})" if terms.model_bytes else "")
    body = (
        f"driver-side footprint preflight {_site_words(site)}: "
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
        f"{what_would_fit(windows or (estimate,), profile, cast=cast)}"
        f"Options: {'; '.join(profile.fitting_options)}. ")
    # Name the kill switch only when no rescue card or stronger quarantine
    # applies. refusal() appends panel and per-kind headless instructions.
    tail = (rescue_clause(rescue) if rescue is not None else
            f"Set {driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV}=1 to turn this "
            f"preflight off for every render (docs/TROUBLESHOOTING.md #52)." if bypass else "")
    room = TEXT_BUDGET - len(lead) - len(fits) - len(tail) - 3
    if room < _MIN_TERMS:
        fits, room = "", TEXT_BUDGET - len(lead) - len(tail) - 2
    if len(body) > room:
        log.warning("loader refusal shortened for the prompt; the whole of it: %s %s", body, tail)
        body = body[:max(room - 60, 0)].rsplit(" ", 1)[0] + " ... (the rest is in the driver log). "
    return " ".join(part.strip() for part in (lead, fits, body, tail) if part.strip())


def fits_box(window: driver_footprint.DriverFootprintEstimate) -> bool:
    """Whether the window fits MemAvailable, the stack in its reserve included.

    ``DriverFootprintEstimate.fits`` clamps the budget at zero, so a window with
    no DiT to place (resident, or rank 0 elsewhere) fits whatever its driver
    models need; the hardware's second-run near-OOM was that case.
    """
    return window.projected + window.reserve <= window.mem_available


def bounded(text: str, limit: int = 3500) -> str:
    """``text`` cut in the middle to ``limit`` characters, the whole of it logged.

    ``refusal()`` raises past 4,000 characters, and the loader preflight admits
    on any error it does not own, so an over-long refusal would let the load
    through; the tag and the panel sentences fit in what this leaves.
    """
    if len(text) <= limit:
        return text
    log.warning("loader refusal shortened for the prompt; the whole of it: %s", text)
    marker = " ... (cut here; the whole refusal is in the driver log) ... "
    keep = max(limit - len(marker), 0)
    return text[:keep // 2] + marker + text[len(text) - (keep - keep // 2):]


def what_would_fit(windows: tuple[driver_footprint.DriverFootprintEstimate, ...],
                   profile: driver_footprint.DriverStackProfile, *, cast: bool = False) -> str:
    """``driver_footprint.what_would_fit`` over every failing window at once.

    The ceiling is the smallest any failing window allows and the memory the
    largest any is short, so neither figure names what another window refuses.
    """
    failing = [window for window in windows if not fits_box(window)] or list(windows)
    low = min(failing, key=lambda window: window.dit_ceiling)
    short = _gib(max(window.shortfall for window in failing))
    if cast:
        # A cast's ceiling would be a copy size, not a file size: name memory only.
        return f"What would fit here: {short} GiB more free memory. "
    return driver_footprint.what_would_fit(low, profile).replace(
        f"{_gib(low.shortfall)} GiB more free memory", f"{short} GiB more free memory", 1)


def plain_lead(estimate: driver_footprint.DriverFootprintEstimate, *, window: str,
               unet_name: str, models: tuple[str, ...], deferred: bool,
               box: str = box_names.LOCAL, site: str = SITE_LOADER) -> str:
    """One plain sentence: what does not fit, by how much, and why at this window.

    At a render site the driver has loaded the graph's own models by then, so the
    last sentence says what the Workers did not load, and why the price ran there.
    """
    name = os.path.basename(unet_name)
    subject = (f"{name} needs" if estimate.weight_bytes
               else f"Beside {name}, this graph's driver models need")
    if window == WINDOW_TRANSIENT and deferred and models:
        named = list(models[:_MAX_NAMED]) + (
            [f"{len(models) - _MAX_NAMED} more"] if len(models) > _MAX_NAMED else [])
        why = (f"It loads at the first render, beside {_join(named)}, which this graph "
               "loads on the driver first.")
    elif window == WINDOW_TRANSIENT:
        why = ("That is the peak of its load, the weights beside the host copy they are "
               "placed from." if estimate.arena_bytes else "That is the peak of its load.")
    else:
        why = "That is the DiT in memory beside what this graph still needs on the driver."
    loaded = (f"The UNET Loader's output came from an earlier queue, so {_SITE_NODES[site]} "
              "priced this graph, and the Workers loaded nothing." if site in RENDER_SITES
              else "Nothing was loaded.")
    return (f"{subject} {_gib(estimate.projected + estimate.reserve)} GiB on {box} and "
            f"{_gib(estimate.mem_available)} GiB is available, {_gib(estimate.shortfall)} "
            f"GiB short. {why} {loaded}")


def graph_loras(prompt: object) -> tuple[str, ...]:
    """The LoRAs this graph's DGX Monarch LoRA loaders apply, on the branches it runs."""
    if not isinstance(prompt, Mapping):
        return ()
    live = live_nodes(prompt)
    return tuple(str(node["inputs"]["lora_name"]) for key, node in prompt.items()
                 if str(key) in live and isinstance(node, Mapping)
                 and str(node.get("class_type", "")).startswith("DGXMonarch")
                 and isinstance(node.get("inputs"), Mapping)
                 and isinstance(node["inputs"].get("lora_name"), str))


def _switch_clause(shape: Any, fits: Callable[[Any], bool], terms: DriverStackTerms,
                   prompt: object, stack_for: Callable[[object], DriverStackTerms],
                   load_of: Callable[[DriverStackTerms], int]) -> str:
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
        if dropped and fits(replace(
                shape, stack_bytes=shape.stack_bytes + other.total - terms.total,
                load_stack_bytes=shape.load_stack_bytes + load_of(other) - load_of(terms))):
            return (f"{SWITCH_TITLES[node['class_type']]} (node {key}) set to "
                    f"{str(not inputs['switch']).lower()} would fit too: the graph then does "
                    f"not load {_join(os.path.basename(name) for name in dropped)}.")
    return ""


def _pair_clause(shape: Any, fits: Callable[[Any], bool], quant_kind: str,
                 loras: tuple[str, ...]) -> str:
    """The pair's shard build, named with the setting it needs, or nothing.

    ``auto`` never resolves to it, so the text names the Init node's topology.
    A LoRA on FSDP shards has its own launch rules and first-use gate, so a
    graph with one is not offered it.
    """
    if (shape.fsdp_world or not shape.co_resident or shape.weights_resident
            or quant_kind not in _SHARDED_KINDS or loras):
        return ""
    if not fits(replace(shape, fsdp_world=2)):
        return ""
    share = driver_footprint.fsdp_shard_build_bytes(shape.weight_bytes, 2)
    return (f"On a pair of Sparks with the Init node's topology set to {PAIR_PRESET} "
            f"(auto never picks it), {box_names.HEAD} would hold {_gib(share)} GiB of the DiT and "
            f"the load fits; Ulysses and ring hold all of it. Under the default "
            f"auto_gate=first_use, its first render there runs the identity gate's "
            f"clean-reload proof before it samples.")


def alternatives(shape: Any, *, fits: Callable[[Any], bool], unet_name: str, family: str,
                 quant_kind: str, terms: DriverStackTerms, deferred: bool, prompt: object,
                 stack_for: Callable[[object], DriverStackTerms], pair: bool,
                 mesh: Any = None, site: str = SITE_LOADER) -> str:
    """What would fit in place of what does not; empty when nothing does.

    ``fits`` prices a changed shape in every window, its load window as ``site``
    charges it. Best effort: a fault names nothing and the refusal stands without it.
    """
    def load_of(stack: DriverStackTerms) -> int:
        return load_window_models(stack, site)[0] if deferred else 0

    try:
        swaps = []
        # A cast prices another file at another copy; its twins and the shard
        # build (which refuses a live cast) are not named.
        cast = bool(getattr(shape, "cast_bytes", 0))
        loras = graph_loras(prompt)
        if shape.co_resident and not shape.weights_resident and not cast:
            dit = dit_twins(shape, fits, unet_name, family, loras=loras, mesh=mesh)
            if dit:
                swaps.append(f"{_join(dit, 'or')} in place of {unet_name}")
        swaps += model_twins(shape, fits, terms, load_of, prompt, stack_for)
        parts = [f"On disk here, these would fit: {'; '.join(swaps)}."] if swaps else []
        parts.append(_switch_clause(shape, fits, terms, prompt, stack_for, load_of))
        parts.append(_pair_clause(shape, fits, quant_kind, loras) if pair and not cast else "")
        return " ".join(part for part in parts if part)
    except Exception as exc:  # naming alternatives never replaces the refusal
        log.warning("loader preflight could not name what would fit (%r)", exc)
        return ""
