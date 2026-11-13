"""Price driver memory that a prompt graph has not made resident yet.

Text encoder and VAE names reach this leaf through Comfy's hidden API-format
``prompt`` input. The result covers those artifacts, reference encoding, guide
encoding, render activations, and residency credits; ``loader_preflight``
applies policy.

Graph and path parse failures omit a charge instead of raising. This preserves
the class C failure direction: uncertain evidence cannot create a false
capacity refusal.

The guide encode and the render activations are a sum unless the graph proves
the guide encodes, and ComfyUI returns its memory, before any sampler runs;
then they are alternatives and the larger one is charged (``guide_order``,
issue #187). An order that cannot be read keeps the sum.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace

from .. import driver_footprint, h3_activation, ltx25_activation
from ..constants import GUIDER_TYPE, MODEL_TYPE
from .graph_live import is_link as _is_link
from .graph_live import live_nodes

_GIB = 2 ** 30

# One row per family whose render activations the loader site prices, each a
# (charge, sequence-degree) pair; read-only after import. Every charge helper
# is total and re-checks its own family, so a graph that names two families
# cannot inherit the other one's rows. An unregistered family charges nothing.
_ACTIVATION_FAMILIES = {
    h3_activation.H3_FAMILY: (h3_activation.graph_activation_bytes,
                              h3_activation.sp_degree_for_mesh),
    h3_activation.H3_IMAGE_FAMILY: (h3_activation.graph_activation_bytes,
                                    h3_activation.sp_degree_for_mesh),
    ltx25_activation.LTX_FAMILY: (ltx25_activation.graph_activation_bytes,
                                  ltx25_activation.sp_degree_for_mesh),
}


# Keyed by family like the table above, and read-only after import: each family
# owns its guide-encode price.
_GUIDE_ENCODE_FAMILIES = {
    h3_activation.H3_FAMILY: h3_activation.graph_guide_encode_bytes,
}


def activation_sp_degree(mesh: object, family: str = "") -> int:
    """Resolve the degree that shards this family's activation row axis.

    Families divide their rows by different degrees: H3 dispatches only
    sequence parallel, while an explicit ltx cfg or dp preset shards no row at
    all. An unregistered family takes the H3 resolution.
    """
    registered = _ACTIVATION_FAMILIES.get(family)
    resolve = registered[1] if registered else h3_activation.sp_degree_for_mesh
    return resolve(mesh)


def graph_activation_bytes(prompt: object, sp_degree: int, *,
                           family: str) -> tuple[int, str]:
    """Dispatch the loader-site activation charge to the family that owns it."""
    registered = _ACTIVATION_FAMILIES.get(family)
    if registered is None:
        return 0, "not a token-priced family"
    return registered[0](prompt, sp_degree, family=family)


def graph_guide_encode_bytes(prompt: object,
                             guides: tuple[tuple[object, int], ...],
                             *, family: str) -> tuple[int, str]:
    """Dispatch the guide-encode charge to the family that owns it."""
    registered = _GUIDE_ENCODE_FAMILIES.get(family)
    if registered is None:
        return 0, "not a guide-anchoring family"
    return registered(prompt, guides, family=family)


def _gib(value: float) -> str:
    return f"{value / _GIB:.1f}"


# Measured 2026-08-05 (docs/TROUBLESHOOTING.md #56): a killed H3 render's RSS
# breakdown read a 16.7 GiB resident text encoder for the 14.61 GiB nvfp4 file,
# ratio 1.144. Rounding up biases this known term toward refusal.
TE_RESIDENT_RATIO = 1.15

# Single-shape calibration, not a per-pixel model. On 2026-08-04, Config A
# used one reference image and a video truncated to five frames. Its reference
# path cost about 40 GiB beyond the 16.8 GiB encoder and 5.4 GiB VAEs.
# Vision and VAE construction workspaces dominate, so pixel scaling is invalid.
# At the 5 GiB driver floor (capacity_floor.ABSOLUTE_HOST_FLOOR_BYTES, since
# 2026-09-04), the 2026-08-04 legs bound the total S = T + V + E to (47.8, 77.6]
# GiB; this value gives 62.2 GiB. Do not extrapolate or lower it without
# another measurement.
REF_VIDEO_ENCODE_BYTES = 40 * _GIB

# Charge only reference video. The measurement cannot isolate image/keyframe
# cost, so estimating it would risk a false refusal.
IMAGE_REF_BYTES = 0

# Graph vocabulary, matched on input key rather than class_type, so a custom or
# renamed loader node that keeps comfy's input names still prices correctly.
# CLIP Vision files resolve in their own folder and are priced as encoders.
_TE_NAME_KEYS = ("clip_name", "text_encoder_name")
_VAE_NAME_KEYS = ("vae_name",)
_TE_FOLDERS = ("text_encoders", "clip", "clip_vision")
_VAE_FOLDERS = ("vae",)
# Whole models the driver loads itself: a checkpoint loader's file, and a stock
# UNET loader's. A DGX Monarch loader's `unet_name` loads on the Workers and is
# the DiT term, so its class prefix keeps it out of this charge. Read-only after import.
_DRIVER_MODEL_KEYS = {"ckpt_name": ("checkpoints",),
                      "unet_name": ("diffusion_models", "unet")}
_WORKER_LOADER_PREFIX = "DGXMonarch"
# Count only linked reference inputs. This excludes widgets such as
# ``ref_image_size`` even though their names share a reference prefix.
_VIDEO_REF_PREFIXES = ("ref_video",)
_IMAGE_REF_PREFIXES = ("ref_image", "first_frame", "last_frame")

# Guide anchoring is keyed on class type, the one place a name is not enough:
# ``image`` names an input on most nodes that touch pixels, and pricing every
# one as an encode would refuse families that encode nothing. A seam canary
# holds these names against the node's own schema.
_GUIDE_NODE_CLASSES = ("MiniMaxH3AddGuide",)
_GUIDE_IMAGE_KEYS = ("image",)
# The canvas a guide is resized to before it encodes, named by a link.
_GUIDE_LATENT_KEYS = ("latent",)
# The widget a loader carries its filename on, so a guide's length can be
# proved from the file rather than trusted from a node name or an extension.
_GUIDE_FILE_KEYS = ("image",)
# Priced at zero like an image reference: audio encodes through the small VAE.
_GUIDE_AUDIO_KEYS = ("audio",)

# The guide's order against the samplers (``guide_order``). ComfyUI with
# DynamicVRAM empties its device pool after every node: the executor's
# ``finally`` calls ``reset_cast_buffers``, which ends in ``soft_empty_cache``.
# So a guide that encodes before every sampler has returned its memory before
# rank 0 samples. A node with one of these inputs linked uses a model, and it
# counts as one that samples unless ComfyUI's registry shows that it returns
# only a patched model, a guider or a sigma schedule (stock's types and the
# pack's). Read-only after import.
_MODEL_INPUT_KEYS = ("model", "guider")
_NO_RENDER_OUTPUTS = frozenset({"MODEL", MODEL_TYPE, "GUIDER", GUIDER_TYPE, "SIGMAS"})
# Samplers whose renders can overlap other driver work, or whose overlap on one
# box this price does not read, each with the words a budget clause gives.
_OVERLAPPING_SAMPLERS = {
    "DGXMonarchKSamplerPipeline": ("the pipeline sampler can sample one render while "
                                   "the next one's guide encodes"),
    "DGXMonarchFleetKSampler": ("the fleet sampler's renders can overlap, which this "
                                "price does not read"),
}
GUIDE_FIRST = "it encodes and ComfyUI returns its memory before any sampler runs"
NO_POOL_RESET = ("ComfyUI here runs without DynamicVRAM, so it keeps the encode's "
                 "memory until the decode")
MODEL_BEFORE_GUIDE = "a node that uses a model runs before a guide encodes"
SAMPLER_BEFORE_GUIDE = "a sampler in this graph can run before a guide encodes"
ORDER_UNREAD = "the graph does not show where the guide encodes"

_MEMO_LIMIT = 64
# Every write to a capped memo holds this lock: `remember` on the prompt thread,
# and `clear_memos` on the executor thread that drains a recycle. Without it, a
# clear that lands inside an eviction raises out of `remember`, and the loader
# preflight's fail-open handler then drops the memo entry.
# Reads are single membership tests and take no lock.
_MEMO_LOCK = threading.Lock()
# Session lifetime, keyed by artifact names already charged. This avoids double
# charging dual-model graphs. FIFO eviction and recycle reset are safe: stale
# entries omit a charge, while lost entries only create a loud over-charge refusal.
_CHARGED_STACK_ARTIFACTS: dict[str, None] = {}


def clear_memos(*stores: dict[str, None]) -> None:
    """Empty capped memos under the lock that ``remember`` holds."""
    with _MEMO_LOCK:
        for store in stores:
            store.clear()


def reset_charged_artifacts() -> None:
    clear_memos(_CHARGED_STACK_ARTIFACTS)


def remember(store: dict[str, None], key: str) -> None:
    """Insertion-ordered set capped at ``_MEMO_LIMIT``, oldest entry evicted first.

    The eviction also tolerates a writer that holds no lock: an emptied or
    resized memo ends the step instead of raising out of ``remember``, which
    the loader preflight's fail-open handler would turn into a lost entry.
    """
    with _MEMO_LOCK:
        store[key] = None
        while len(store) > _MEMO_LIMIT:
            try:
                oldest = next(iter(store))
            except (RuntimeError, StopIteration):
                continue  # the memo changed size under the iterator: re-read it
            store.pop(oldest, None)


@dataclass(frozen=True, slots=True)
class DriverStackTerms:
    """Driver-side memory this graph has yet to allocate.

    ``resolved=False`` means no readable graph, including headless calls with
    no hidden prompt. Those calls charge only the weight term.
    """

    te_bytes: int = 0
    vae_bytes: int = 0
    ref_bytes: int = 0
    te_artifacts: tuple[str, ...] = ()
    vae_artifacts: tuple[str, ...] = ()
    # Checkpoints and stock UNETs the driver loads itself, at file size.
    model_bytes: int = 0
    model_artifacts: tuple[str, ...] = ()
    video_refs: int = 0
    image_refs: int = 0
    guide_refs: int = 0
    guide_bytes: int = 0
    guide_note: str = "no graph available at the loader node"
    resolved: bool = False
    credited: bool = False
    credit_reason: str = ""
    activation_bytes: int = 0
    # This text reaches class C refusals when headless calls omit ``prompt``.
    # Keep it nonempty so the activation clause remains actionable.
    activation_note: str = "no graph available at the loader node"
    # Every guide encodes, and ComfyUI returns its memory, before any sampler
    # runs (``guide_order``): the guide and the activations are then
    # alternatives. ``guide_order_note`` says why, either way.
    guide_first: bool = False
    guide_order_note: str = ""

    @property
    def encode_and_render_bytes(self) -> int:
        """The guide encode and the render activations: the larger of the two
        when ``guide_first``, both otherwise."""
        if self.guide_first:
            return max(self.guide_bytes, self.activation_bytes)
        return self.guide_bytes + self.activation_bytes

    @property
    def total(self) -> int:
        return self.models_bytes + self.ref_bytes + self.encode_and_render_bytes

    @property
    def models_bytes(self) -> int:
        """The models this graph loads on the driver, without encodes or activations."""
        return self.te_bytes + self.vae_bytes + self.model_bytes

    @property
    def models(self) -> tuple[str, ...]:
        return (*self.te_artifacts, *self.model_artifacts, *self.vae_artifacts)

    @property
    def budget_clause(self) -> str:
        """Describe the settled budget without claiming absent charges.

        Credit clears the text encoder, VAE and reference encode charges
        (``credited_terms``). It never clears render activations or a guide
        encode, so a credited result names what survives and keeps its
        ``credit_reason`` either way. With both a guide and activations, the
        clause says whether they are charged as alternatives or together, and
        why (``guide_order_note``).
        """
        total = self.total
        if not total:
            return (f"no driver stack charged ({self.credit_reason})"
                    if self.credited else "no driver stack charged")
        both = bool(self.guide_bytes and self.activation_bytes)
        guide, renders = self.guide_bytes, self.activation_bytes
        order = ""
        if both and self.guide_first and guide > renders:
            renders, order = 0, (f"; its {_gib(self.activation_bytes)} GiB of render "
                                 "activations come after ComfyUI returns the encode's "
                                 "memory, so only the larger of the two is charged")
        elif both and self.guide_first:
            guide, order = 0, (f"; its {_gib(self.guide_bytes)} GiB guide encode runs "
                               "first and ComfyUI returns that memory before sampling, "
                               "so only the larger of the two is charged")
        elif both and self.guide_order_note:
            order = f"; both are charged because {self.guide_order_note}"
        carried = " and ".join(name for name, charged in (
            ("driver stack", self.models_bytes + self.ref_bytes),
            ("guide encode", guide),
            ("render activations", renders)) if charged)
        credit = (f", with the text encoder, VAE and reference encode charges "
                  f"credited ({self.credit_reason})" if self.credited else "")
        return (f"{_gib(total)} GiB of {carried} this graph has not spent "
                f"yet{credit}{order}")


def _links_under(value: object) -> int:
    """Count links through one Autogrow container level.

    Comfy serializes Autogrow as a mapping keyed by index strings.
    """
    if _is_link(value):
        return 1
    if isinstance(value, Mapping):
        return sum(1 for item in value.values() if _is_link(item))
    if isinstance(value, (list, tuple)):
        return sum(1 for item in value if _is_link(item))
    return 0


def _artifact_bytes(folders: tuple[str, ...], name: str) -> int:
    """Return artifact size, or 0 when Comfy cannot resolve it.

    Guard the import for headless callers. Missing evidence omits the charge.
    """
    try:
        import folder_paths
    except Exception:
        return 0

    for folder in folders:
        try:
            path = folder_paths.get_full_path(folder, name)
        except Exception:  # an unknown folder must not raise here
            path = None
        if path:
            return driver_footprint.file_size_bytes(path)
    return 0


def named_file_frames(prompt: object, node_id: object) -> int:
    """Frames in the file this node names, or 0 when nothing proves a count.

    The file is opened and its frames counted, so an animated PNG or WebP
    cannot pass as a still behind an image extension, and a video that no
    image decoder opens reads unreadable rather than cheap. Missing files,
    unreadable files and sources that name no file all return 0, which the
    guide price reads as the declared length or else the worst case.

    A file replaced between this preflight read and execution is charged on
    the proof taken, the same window every other pre-execution read here has.
    """
    if not isinstance(prompt, Mapping) or node_id is None:
        return 0
    node = prompt.get(node_id)
    if node is None:
        node = prompt.get(str(node_id))
    inputs = node.get("inputs") if isinstance(node, Mapping) else None
    if not isinstance(inputs, Mapping):
        return 0
    for key in _GUIDE_FILE_KEYS:
        name = inputs.get(key)
        if not isinstance(name, str) or not name:
            continue
        try:
            import folder_paths
            from PIL import Image

            path = folder_paths.get_annotated_filepath(name)
            if not path or not os.path.isfile(path):
                return 0
            with Image.open(path) as image:
                frames = getattr(image, "n_frames", 1)
            return int(frames) if isinstance(frames, int) and frames > 0 else 0
        except Exception:  # unreadable evidence charges the worst case
            return 0
    return 0


def comfy_returns_pool_after_each_node() -> bool:
    """Whether ComfyUI in this process empties its device pool after every node.

    It does with DynamicVRAM on: the executor's ``finally`` calls
    ``reset_cast_buffers``, which ends in ``soft_empty_cache``
    (tests/canary/comfy_seam_contracts.py holds both). Off, or with no ComfyUI
    in this process, an encode's freed blocks stay reserved until the decode.
    """
    try:
        import comfy.memory_management as memory_management
    except Exception:  # no ComfyUI here: nothing returns the pool
        return False
    return getattr(memory_management, "aimdo_enabled", False) is True


def _class_outputs(class_type: object) -> tuple[str, ...] | None:
    """The output types ComfyUI's registry gives a node class, or None when it cannot say."""
    try:
        import nodes as comfy_nodes

        cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(class_type)
        outputs = None if cls is None else cls.RETURN_TYPES
    except Exception:  # no registry, or a class it cannot read
        return None
    if not isinstance(outputs, (list, tuple)):
        return None
    return tuple(str(kind) for kind in outputs)


def _uses_model(node: Mapping) -> bool:
    inputs = node.get("inputs")
    return isinstance(inputs, Mapping) and any(
        _is_link(inputs.get(key)) for key in _MODEL_INPUT_KEYS)


def _may_sample(node: Mapping) -> bool:
    """A node that uses a model, unless the registry shows every output is one
    that carries no render: a model patch, a guider and a scheduler use a
    model and sample nothing. Identity Gate (DGX Monarch) returns a latent, and
    a class the registry cannot place counts as sampling."""
    if not _uses_model(node):
        return False
    outputs = _class_outputs(node.get("class_type"))
    return not outputs or not set(outputs) <= _NO_RENDER_OUTPUTS


def guide_order(prompt: Mapping, live: frozenset[str],
                guides: tuple[str, ...]) -> tuple[bool, str]:
    """Whether every guide encodes, and returns its memory, before any sampler
    runs; the words for either answer.

    Four things must hold. ComfyUI here returns its pool after every node. No
    sampler that overlaps other driver work is in the graph. No guide has a
    node that uses a model among its ancestors (a second pass, a guide image
    decoded from a render), read with both branches of an undecided switch.
    And every node that may sample has every guide among the nodes ComfyUI is
    certain to run before it, so no sampler can run first. Any doubt, and any
    fault, keeps the sum.
    """
    try:
        if not comfy_returns_pool_after_each_node():
            return False, NO_POOL_RESET
        nodes = {str(key): node for key, node in prompt.items() if isinstance(node, Mapping)}
        for key in sorted(live):
            words = _OVERLAPPING_SAMPLERS.get(str(nodes.get(key, {}).get("class_type", "")))
            if words:
                return False, words
        for guide in guides:
            if any(_uses_model(nodes[key]) for key in live_nodes(nodes, roots=[guide])
                   if key != guide and key in nodes):
                return False, MODEL_BEFORE_GUIDE
        for key in sorted(live):
            node = nodes.get(key)
            if node is None or not _may_sample(node):
                continue
            ran_first = live_nodes(nodes, roots=[key], sure=True)
            if not all(guide in ran_first for guide in guides):
                return False, SAMPLER_BEFORE_GUIDE
        return True, GUIDE_FIRST
    except Exception:  # an unread order keeps the sum
        return False, ORDER_UNREAD


def driver_stack_terms(prompt: object, *, family: str = "",
                       sp_degree: int = 1) -> DriverStackTerms:
    """Price unresolved driver-stack and render-activation memory.

    Activations remain charged after residency credits because they are not
    allocated at loader time. The graph exposes video and audio rows, and a guide's
    encode is priced from the canvas it is resized to, but a linked reference
    or keyframe's own geometry is unavailable until execution; render-site
    checks catch that permissive under-count later.

    ``family`` prevents unrelated canvas nodes from inheriting another model
    family's activation charge. An unknown family charges none. With both a
    guide and activations charged, ``guide_order`` decides whether they are
    alternatives, and the guide's note says which and why.
    """
    if not isinstance(prompt, Mapping) or not prompt:
        return DriverStackTerms()
    activation_bytes, activation_note = graph_activation_bytes(
        prompt, sp_degree, family=family)
    te_bytes = vae_bytes = model_bytes = 0
    te_names: list[str] = []
    vae_names: list[str] = []
    model_names: list[str] = []
    video_refs = image_refs = 0
    guide_latents: list[tuple[object, int]] = []
    guide_nodes: list[str] = []
    live = live_nodes(prompt)
    for key, node in prompt.items():
        inputs = node.get("inputs") if isinstance(node, Mapping) else None
        if not isinstance(inputs, Mapping) or str(key) not in live:
            continue
        node_key = str(key)
        on_driver = not str(node.get("class_type", "")).startswith(_WORKER_LOADER_PREFIX)
        guide_node = (isinstance(node, Mapping)
                      and node.get("class_type") in _GUIDE_NODE_CLASSES)
        if guide_node:
            anchor = next((inputs[key] for key in _GUIDE_LATENT_KEYS
                           if _is_link(inputs.get(key))), None)
            canvas = anchor[0] if isinstance(anchor, (list, tuple)) else None
            for key, value in inputs.items():
                if not isinstance(key, str) or not key.startswith(_GUIDE_IMAGE_KEYS):
                    continue
                source = value[0] if isinstance(value, (list, tuple)) else None
                links = _links_under(value)
                if links and node_key not in guide_nodes:
                    guide_nodes.append(node_key)
                guide_latents.extend(
                    (canvas, named_file_frames(prompt, source))
                    for _ in range(links))
        for key, value in inputs.items():
            if not isinstance(key, str):
                continue
            if isinstance(value, str) and value:
                if key.startswith(_TE_NAME_KEYS):
                    size = _artifact_bytes(_TE_FOLDERS, value)
                    if size:
                        te_bytes += size
                        te_names.append(os.path.basename(value))
                elif key.startswith(_VAE_NAME_KEYS):
                    size = _artifact_bytes(_VAE_FOLDERS, value)
                    if size:
                        vae_bytes += size
                        vae_names.append(os.path.basename(value))
                elif key in _DRIVER_MODEL_KEYS and on_driver:
                    size = _artifact_bytes(_DRIVER_MODEL_KEYS[key], value)
                    if size:
                        model_bytes += size
                        model_names.append(os.path.basename(value))
                continue
            if guide_node and key.startswith(_GUIDE_IMAGE_KEYS):
                continue                 # counted with its canvas above
            if key.startswith(_VIDEO_REF_PREFIXES):
                video_refs += _links_under(value)
            elif key.startswith(_IMAGE_REF_PREFIXES):
                image_refs += _links_under(value)
    guide_bytes, guide_note = graph_guide_encode_bytes(
        prompt, tuple(guide_latents), family=family)
    guide_first, order_note = False, ""
    if guide_bytes and activation_bytes:
        guide_first, order_note = guide_order(prompt, live, tuple(guide_nodes))
        if guide_first:
            charged = ("this guide encode" if guide_bytes > activation_bytes else
                       f"the {_gib(activation_bytes)} GiB of render activations")
            guide_note = (f"{guide_note}; only the larger of this and the render "
                          f"activations is charged, here {charged}, since {order_note}")
        else:
            guide_note = (f"{guide_note}; charged beside the render activations because "
                          f"{order_note}")
    return DriverStackTerms(
        te_bytes=int(te_bytes * TE_RESIDENT_RATIO),
        vae_bytes=vae_bytes,
        ref_bytes=(REF_VIDEO_ENCODE_BYTES if video_refs
                   else IMAGE_REF_BYTES * image_refs),
        te_artifacts=tuple(te_names), vae_artifacts=tuple(vae_names),
        model_bytes=model_bytes, model_artifacts=tuple(model_names),
        video_refs=video_refs, image_refs=image_refs, resolved=True,
        guide_refs=len(guide_latents), guide_bytes=guide_bytes,
        guide_note=guide_note,
        activation_bytes=activation_bytes, activation_note=activation_note,
        guide_first=guide_first, guide_order_note=order_note,
    )


def unet_names(prompt: object) -> frozenset[str]:
    """Return all checkpoints named by this graph's loader nodes.

    The full set keeps checkpoint-scoped consent stable across dual-model eager
    loads in one prompt.
    """
    if not isinstance(prompt, Mapping):
        return frozenset()
    found: set[str] = set()
    for node in prompt.values():
        inputs = node.get("inputs") if isinstance(node, Mapping) else None
        if not isinstance(inputs, Mapping):
            continue
        value = inputs.get("unet_name")
        if isinstance(value, str) and value:
            found.add(value)
    return frozenset(found)


def _comfy_has_loaded_models() -> bool:
    """Whether Comfy already has a driver-side model in use.

    Comfy execution order can run VAE or text loaders first. A model comfy is
    using already shows in the MemAvailable reading, so charging it again
    would create a class C false refusal. ``credited_terms`` credits what
    the entries hold, up to the whole stack (``_comfy_held_bytes``).

    A loaded but not yet used artifact is absent from that list and can be
    over-charged once. The artifact memo covers repeats; missing every charge
    would permit unsafe loads.
    """
    try:
        import comfy.model_management as mm

        return bool(getattr(mm, "current_loaded_models", None))
    except Exception:  # no comfy means nothing comfy has loaded
        return False


def _comfy_held_bytes() -> int | None:
    """The bytes of the models ComfyUI holds, or None when it cannot say.

    It cannot say which files they came from, so the credit below never
    exceeds this: a held encoder can stand in for this graph's, never for more.
    """
    try:
        import comfy.model_management as mm

        return sum(int(loaded.model_memory()) for loaded in mm.current_loaded_models
                   if loaded.model is not None)
    except Exception:  # an unreadable size credits as before, the whole stack
        return None


def _credit_held(terms: DriverStackTerms, held: int, ratio: float = 1.0) -> DriverStackTerms:
    """Credit ``held`` bytes against the graph's models; the rest stays charged.

    The reference encode stays charged too: with models still to load, it has
    not run either. ``ratio`` reads held bytes as encoders resident at that ratio.
    """
    te = max(terms.te_bytes - int(held * ratio), 0)
    left = max(held - int(terms.te_bytes / ratio), 0)
    vae = max(terms.vae_bytes - left, 0)
    left = max(left - terms.vae_bytes, 0)
    return replace(terms, te_bytes=te, vae_bytes=vae,
                   model_bytes=max(terms.model_bytes - left, 0), credited=True,
                   credit_reason=(f"comfy already holds {_gib(held)} GiB of loaded models, "
                                  f"credited against the {_gib(terms.models_bytes)} GiB of "
                                  "models this graph loads"))


def credited_terms(terms: DriverStackTerms, ratio: float = 1.0) -> DriverStackTerms:
    """Apply the residency credits: comfy's loaded models, then earlier charges.

    A guide encode survives both credits, like render activations. The first
    credit fires on any loaded model. It credits the whole stack only where
    what ComfyUI holds covers it (MemAvailable already counts that much), and
    otherwise only the bytes held; a loaded model does not show that Add Guide
    has run.
    """
    if not terms.resolved:
        return terms
    if _comfy_has_loaded_models():
        held = _comfy_held_bytes()
        if held is not None and held < terms.models_bytes:
            return _credit_held(terms, held, ratio)
        return replace(terms, te_bytes=0, vae_bytes=0, model_bytes=0, ref_bytes=0,
                       credited=True,
                       credit_reason="comfy already holds loaded models, so "
                                     "the preflight assumes MemAvailable counts this stack")
    named = terms.models
    if named and all(name in _CHARGED_STACK_ARTIFACTS for name in named):
        return replace(terms, te_bytes=0, vae_bytes=0, model_bytes=0, ref_bytes=0,
                       credited=True,
                       credit_reason="an earlier loader node charged "
                                     "this stack")
    return terms


def note_stack_charged(terms: DriverStackTerms) -> None:
    """Record a cleared charge, so the next loader node credits it."""
    for name in terms.models:
        remember(_CHARGED_STACK_ARTIFACTS, name)
