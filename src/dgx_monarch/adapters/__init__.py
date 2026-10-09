"""Adapter registry (DESIGN.md §5.3).

Adapters own all per-model conditionals: no `if model_class == ...` exists
outside this package (DESIGN.md §7). Most specific adapters are listed first;
the first isinstance match wins.
"""
from __future__ import annotations

from .anima import AnimaAdapter
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    make_usp_attention,
)
from .boogu import BooguAdapter
from .cogvideo import CogVideoXAdapter
from .ernie import ErnieAdapter
from .flux_family import ChromaAdapter, Flux2Adapter, FluxAdapter, LongCatAdapter
from .hunyuan import HunyuanAdapter
from .kandinsky5 import Kandinsky5Adapter
from .krea2 import Krea2Adapter
from .lens import LensAdapter
from .ltx import LTXAdapter
from .mage_flow import MageFlowAdapter
from .minimax_h3 import MiniMaxH3Adapter
from .omnigen2 import Omnigen2Adapter
from .pixeldit import Ideogram4Adapter
from .pixeldit_comfy import PixelDiTAdapter
from .qwen_image import QwenImageAdapter
from .qwen_image21 import QwenImage21Adapter
from .wan_animate2 import WanAnimate2Adapter
from .wan_family import WanAdapter
from .wan_ring_attention import make_wan_ring_usp_attention
from .wan_variants import SCAIL2Adapter, SCAILAdapter, WanDancerAdapter
from .zimage import ZImageAdapter

# Order matters: an adapter whose Comfy class subclasses another family's comes
# first. Chroma, Flux2 and LongCatImage subclass Flux and precede FluxAdapter,
# which owns the typed reject for unclaimed Flux subclasses. MageFlow and
# QwenImage21 subclass QwenImage, whose adapter raises on any subclass. Krea2,
# Ideogram4 and MiniMaxH3 subclass BaseModel directly, and no adapter names
# BaseModel, so their place is free.
ADAPTERS: tuple[Adapter, ...] = (
    Krea2Adapter(),
    Ideogram4Adapter(),
    PixelDiTAdapter(),
    MageFlowAdapter(),
    QwenImage21Adapter(),
    QwenImageAdapter(),
    LensAdapter(),
    ChromaAdapter(),
    Flux2Adapter(),
    LongCatAdapter(),
    FluxAdapter(),
    HunyuanAdapter(),
    ZImageAdapter(),
    LTXAdapter(),
    # MiniMaxH3's matches() raises on any MiniMaxH3 subclass, so an adapter for
    # one must be registered before this entry.
    MiniMaxH3Adapter(),
    # Boogu subclasses Omnigen2, whose exact-type matches() raises on a Boogu
    # instance. The Wan variants subclass WAN21 and precede WanAdapter; SCAIL2
    # subclasses SCAIL and precedes it.
    BooguAdapter(),
    Omnigen2Adapter(),
    SCAIL2Adapter(),
    SCAILAdapter(),
    WanAnimate2Adapter(),
    WanDancerAdapter(),
    WanAdapter(),
    ErnieAdapter(),
    CogVideoXAdapter(),
    Kandinsky5Adapter(),
    AnimaAdapter(),
)


# The family strings a user may name in the Init node's `family_adapter`
# widget, derived here so a newly registered family cannot be forgotten. Fewer
# entries than ADAPTERS: SCAIL2 and SCAIL both answer "wan_scail".
SELECTABLE_FAMILIES: tuple[str, ...] = tuple(sorted({a.family for a in ADAPTERS}))

# Family strings whose adapter runs the dual-model cfg-parallel (dm-cfg2) split
# guider at cfg2 (Ideogram4). Both the driver's selection preflight and
# the worker's sample-time backstop resolve a sniffed family to this grant.
DUAL_MODEL_CFG_FAMILIES: frozenset[str] = frozenset(
    a.family for a in ADAPTERS if getattr(a, "dual_model_cfg_supported", False))


def family_supports_dual_model_cfg(family: str) -> bool:
    """Whether a family runs the dm-cfg2 split guider (grant carried by its adapter)."""
    return family in DUAL_MODEL_CFG_FAMILIES


def get_adapter(base_model) -> Adapter:
    for adapter in ADAPTERS:
        if adapter.matches(base_model):
            return adapter
    raise UnsupportedModelError(
        f"model class {type(base_model).__name__} has no dgx-monarch adapter. "
        f"Supported families: krea2, mage_flow, qwen_image21, qwen_image, lens, chroma, flux, flux2, longcat, "
        f"hunyuan, zimage, ideogram4, pixeldit_comfy, ltx, minimax_h3, boogu, omnigen2, "
        f"wan_scail, wan_animate2, wan_dancer, wan, ernie, cogvideo, kandinsky5, anima "
        "(docs/MODELS.md)."
    )


def _accepted_class_names(adapter: Adapter) -> tuple[str, ...]:
    """Every `comfy.model_base` name this adapter will bind to under a force."""
    return (*adapter.model_base_classes, *adapter.exact_model_base_classes)


def _ancestry_owner(base_model) -> Adapter | None:
    """The most specific registered adapter this model is an instance of.

    ADAPTERS is ordered most specific first (the comments there say why), so
    the first ancestry match is the owner, not merely an adapter that accepts
    the model. Walking the whole registry, not one family's entries, keeps a
    parent family from binding over a child family's divergent forward.
    """
    import comfy.model_base as model_base

    for adapter in ADAPTERS:
        for name in _accepted_class_names(adapter):
            cls = getattr(model_base, name, None)
            if cls is not None and isinstance(base_model, cls):
                return adapter
    return None


def _forced_adapter(base_model, family: str) -> Adapter:
    """Bind `family`'s adapter to `base_model` by Comfy-class ancestry.

    It never calls ``matches()``: the exact-type adapters raise there on a
    subclass (see the Omnigen2/Boogu and MiniMaxH3 notes on ADAPTERS), and an
    override is meant to admit a derivative subclass such as a finetune. The
    adapter re-expresses the forward of the module tree Comfy built, so binding
    is safe only when that tree is an instance of a class this family owns and
    no more specific family owns it.

    Registry order resolves a family named by more than one adapter, so
    ``wan_scail`` keeps SCAIL2 ahead of SCAIL as detection does.
    """
    from ..refusal import RefusalClass, refusal

    if not any(adapter.family == family for adapter in ADAPTERS):
        # Checked before touching Comfy: an unknown name is a graph error, not
        # a model question.
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"family_adapter={family!r} names no dgx-monarch adapter. Use one of: "
            f"{', '.join(SELECTABLE_FAMILIES)}, or leave family_adapter on 'auto' "
            "so detection chooses (docs/MODELS.md).",
        ))
    owner = _ancestry_owner(base_model)
    if owner is None:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"family_adapter={family!r} does not fit this checkpoint. ComfyUI built a "
            f"{type(base_model).__name__}, which is not an instance of any class a "
            "dgx-monarch adapter owns, so no adapter re-expresses its forward, and "
            "naming a family would patch a module tree that family was not built for. "
            "Use a checkpoint of a supported architecture (docs/MODELS.md), or render "
            "this one in plain ComfyUI without the DGX Monarch nodes instead.",
        ))
    if owner.family != family:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"family_adapter={family!r} does not fit this checkpoint. ComfyUI built a "
            f"{type(base_model).__name__}, which the {owner.family} adapter owns. "
            f"A {owner.family} model can be an instance of the {family} classes and "
            f"still re-express a different forward, which is why {owner.family} is "
            "registered ahead of it; binding the named family here would compute "
            "the wrong math. Use "
            f"{owner.family!r}, or set family_adapter back to 'auto' so "
            "detection chooses (docs/MODELS.md).",
        ))
    return owner


def adapter_for(base_model, family_override: str | None = None) -> Adapter:
    """The one adapter decision. `family_override` None or "auto" means detect.

    Every site that needs the bound adapter calls this, so injection and the
    render-time reads agree on the family. A disagreement would raise the
    override's own refusal mid-render, on a model whose forward is already
    patched.
    """
    if not family_override or family_override == "auto":
        return get_adapter(base_model)
    return _forced_adapter(base_model, str(family_override))


def detect_family(base_model) -> str:
    """Return the detected adapter family, or raise its typed refusal.

    Never turn an unsupported subclass into a made-up family string: callers
    use this value for topology and residency policy before adapter injection.
    """
    return get_adapter(base_model).family


__all__ = [
    "ADAPTERS",
    "SELECTABLE_FAMILIES",
    "Adapter",
    "InjectionContext",
    "UnsupportedModelError",
    "adapter_for",
    "detect_family",
    "get_adapter",
    "make_usp_attention",
    "make_wan_ring_usp_attention",
]
