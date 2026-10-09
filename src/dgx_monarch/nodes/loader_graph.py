"""Price driver memory that a prompt graph has not made resident yet.

Text encoder and VAE names reach this leaf through Comfy's hidden API-format
``prompt`` input. The result covers those artifacts, reference encoding, guide
encoding, render activations, and residency credits; ``loader_preflight``
applies policy.

Graph and path parse failures omit a charge instead of raising. This preserves
the class C failure direction: uncertain evidence cannot create a false
capacity refusal.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace

from .. import driver_footprint, h3_activation, ltx25_activation

_GIB = 2 ** 30

# One row per family whose render activations the loader site prices, each a
# (charge, sequence-degree) pair; read-only after import. Every charge helper
# is total and re-checks its own family, so a graph that names two families
# cannot inherit the other one's rows. An unregistered family charges nothing.
_ACTIVATION_FAMILIES = {
    h3_activation.H3_FAMILY: (h3_activation.graph_activation_bytes,
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
_TE_NAME_KEYS = ("clip_name", "text_encoder_name")
_VAE_NAME_KEYS = ("vae_name",)
_TE_FOLDERS = ("text_encoders", "clip")
_VAE_FOLDERS = ("vae",)
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

_MEMO_LIMIT = 64
# Session lifetime, keyed by artifact names already charged. This avoids double
# charging dual-model graphs. FIFO eviction and recycle reset are safe: stale
# entries omit a charge, while lost entries only create a loud over-charge refusal.
_CHARGED_STACK_ARTIFACTS: dict[str, None] = {}


def reset_charged_artifacts() -> None:
    _CHARGED_STACK_ARTIFACTS.clear()


def remember(store: dict[str, None], key: str) -> None:
    """Insertion-ordered set capped at ``_MEMO_LIMIT``, oldest entry evicted first."""
    store[key] = None
    while len(store) > _MEMO_LIMIT:
        store.pop(next(iter(store)))


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

    @property
    def total(self) -> int:
        return (self.te_bytes + self.vae_bytes + self.ref_bytes
                + self.guide_bytes + self.activation_bytes)

    @property
    def budget_clause(self) -> str:
        """Describe the settled budget without claiming absent charges.

        Credit clears the text encoder, VAE and reference encode charges
        (``credited_terms``). It never clears render activations or a guide
        encode, so a credited result names what survives and keeps its
        ``credit_reason`` either way.
        """
        total = self.total
        if not total:
            return (f"no driver stack charged ({self.credit_reason})"
                    if self.credited else "no driver stack charged")
        carried = " and ".join(name for name, charged in (
            ("driver stack", self.te_bytes + self.vae_bytes + self.ref_bytes),
            ("guide encode", self.guide_bytes),
            ("render activations", self.activation_bytes)) if charged)
        credit = (f", with the text encoder, VAE and reference encode charges "
                  f"credited ({self.credit_reason})" if self.credited else "")
        return (f"{_gib(total)} GiB of {carried} this graph has not spent "
                f"yet{credit}")


def _is_link(value: object) -> bool:
    """Comfy API-format link: ``[node_id, output_index]``."""
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and isinstance(value[0], (str, int))
            and isinstance(value[1], int) and not isinstance(value[1], bool))


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


def driver_stack_terms(prompt: object, *, family: str = "",
                       sp_degree: int = 1) -> DriverStackTerms:
    """Price unresolved driver-stack and render-activation memory.

    Activations remain charged after residency credits because they are not
    allocated at loader time. The graph exposes video and audio rows, and a guide's
    encode is priced from the canvas it is resized to, but a linked reference
    or keyframe's own geometry is unavailable until execution; render-site
    checks catch that permissive under-count later.

    ``family`` prevents unrelated canvas nodes from inheriting another model
    family's activation charge. An unknown family charges none.
    """
    if not isinstance(prompt, Mapping) or not prompt:
        return DriverStackTerms()
    activation_bytes, activation_note = graph_activation_bytes(
        prompt, sp_degree, family=family)
    te_bytes = vae_bytes = 0
    te_names: list[str] = []
    vae_names: list[str] = []
    video_refs = image_refs = 0
    guide_latents: list[tuple[object, int]] = []
    for node in prompt.values():
        inputs = node.get("inputs") if isinstance(node, Mapping) else None
        if not isinstance(inputs, Mapping):
            continue
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
                guide_latents.extend(
                    (canvas, named_file_frames(prompt, source))
                    for _ in range(_links_under(value)))
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
                continue
            if guide_node and key.startswith(_GUIDE_IMAGE_KEYS):
                continue                 # counted with its canvas above
            if key.startswith(_VIDEO_REF_PREFIXES):
                video_refs += _links_under(value)
            elif key.startswith(_IMAGE_REF_PREFIXES):
                image_refs += _links_under(value)
    guide_bytes, guide_note = graph_guide_encode_bytes(
        prompt, tuple(guide_latents), family=family)
    return DriverStackTerms(
        te_bytes=int(te_bytes * TE_RESIDENT_RATIO),
        vae_bytes=vae_bytes,
        ref_bytes=(REF_VIDEO_ENCODE_BYTES if video_refs
                   else IMAGE_REF_BYTES * image_refs),
        te_artifacts=tuple(te_names), vae_artifacts=tuple(vae_names),
        video_refs=video_refs, image_refs=image_refs, resolved=True,
        guide_refs=len(guide_latents), guide_bytes=guide_bytes,
        guide_note=guide_note,
        activation_bytes=activation_bytes, activation_note=activation_note,
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
    would create a class C false refusal. Any one entry is enough, so
    ``credited_terms`` assumes the reading already counts the whole stack.

    A loaded but not yet used artifact is absent from that list and can be
    over-charged once. The artifact memo covers repeats; missing every charge
    would permit unsafe loads.
    """
    try:
        import comfy.model_management as mm

        return bool(getattr(mm, "current_loaded_models", None))
    except Exception:  # no comfy means nothing comfy has loaded
        return False


def credited_terms(terms: DriverStackTerms) -> DriverStackTerms:
    """Apply the residency credits: comfy's loaded models, then earlier charges.

    A guide encode survives both credits, like render activations. The first
    credit fires on any loaded model and assumes MemAvailable already counts
    the stack; a loaded model does not show that Add Guide has run.
    """
    if not terms.resolved:
        return terms
    if _comfy_has_loaded_models():
        return replace(terms, te_bytes=0, vae_bytes=0, ref_bytes=0,
                       credited=True,
                       credit_reason="comfy already holds loaded models, so "
                                     "the preflight assumes MemAvailable counts this stack")
    named = (*terms.te_artifacts, *terms.vae_artifacts)
    if named and all(name in _CHARGED_STACK_ARTIFACTS for name in named):
        return replace(terms, te_bytes=0, vae_bytes=0, ref_bytes=0,
                       credited=True,
                       credit_reason="an earlier loader node charged "
                                     "this stack")
    return terms


def note_stack_charged(terms: DriverStackTerms) -> None:
    """Record a cleared charge, so the next loader node credits it."""
    for name in (*terms.te_artifacts, *terms.vae_artifacts):
        remember(_CHARGED_STACK_ARTIFACTS, name)
