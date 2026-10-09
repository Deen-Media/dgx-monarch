"""Guider nodes: build guider spec dicts consumed by DGXMonarchSamplerCustom.

Guider objects hold model references, so the driver never builds real comfy
guiders. The workers materialize them from these specs against their
resident models (actor/sampling.build_guider).

DGXMonarchDualModelGuider is the Ideogram4 asymmetric-CFG guider: the positive
pass runs the conditional model, the negative pass runs a separate
unconditional model (with no text conditioning when no negative is wired).
With no second model wired it is a plain CFG guider on one model.
"""
from __future__ import annotations

from ..constants import GUIDER_TYPE, MODEL_TYPE, NODE_CATEGORY
from .common import ModelSpec, conditioning_for_wire
from .render_preflight import krea2_ref_preflight_summary


class DGXMonarchBasicGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "conditioning": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = (GUIDER_TYPE,)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = NODE_CATEGORY

    def get_guider(self, model: ModelSpec, conditioning):
        spec = {"kind": "basic", "positive": conditioning_for_wire(conditioning)}
        ref_summary = krea2_ref_preflight_summary(conditioning, None)
        return ({"model": model, "spec": spec, "cfg": 1.0, "uncond_model": None,
                 "_dgxm_krea2_ref_preflight": ref_summary},)


class DGXMonarchCFGGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            },
        }

    RETURN_TYPES = (GUIDER_TYPE,)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = NODE_CATEGORY

    def get_guider(self, model: ModelSpec, positive, negative, cfg):
        spec = {
            "kind": "cfg",
            "positive": conditioning_for_wire(positive),
            "negative": conditioning_for_wire(negative),
            "cfg": float(cfg),
        }
        ref_summary = krea2_ref_preflight_summary(positive, negative)
        return ({"model": model, "spec": spec, "cfg": float(cfg), "uncond_model": None,
                 "_dgxm_krea2_ref_preflight": ref_summary},)


class DGXMonarchDualModelGuider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE, {"tooltip": "Conditional model; it runs the positive pass."}),
                "positive": ("CONDITIONING",),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            },
            "optional": {
                "model_negative": (MODEL_TYPE, {"tooltip": "Unconditional model from the Uncond loader. "
                                                "Leave it unconnected for plain CFG on one model."}),
                "negative": ("CONDITIONING", {"tooltip": "Leave it unconnected to run the negative pass "
                                                 "with no text conditioning."}),
            },
        }

    RETURN_TYPES = (GUIDER_TYPE,)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = NODE_CATEGORY

    def get_guider(self, model: ModelSpec, positive, cfg, model_negative: ModelSpec | None = None,
                   negative=None):
        if model_negative is not None and model_negative.slot != "uncond":
            raise ValueError(
                "model_negative must come from the DGX Monarch UNCOND loader "
                "(DGXMonarchUncondUNETLoader): the workers keep it in the second model slot. "
                "Wiring the regular loader here would evict the conditional model instead."
            )
        ref_summary = krea2_ref_preflight_summary(positive, negative)
        if model_negative is None:
            spec = {
                "kind": "cfg",
                "positive": conditioning_for_wire(positive),
                "negative": conditioning_for_wire(negative) if negative is not None else [[None, {}]],
                "cfg": float(cfg),
            }
            return ({"model": model, "spec": spec, "cfg": float(cfg), "uncond_model": None,
                     "_dgxm_krea2_ref_preflight": ref_summary},)

        spec = {
            "kind": "dual_model",
            "positive": conditioning_for_wire(positive),
            "negative": conditioning_for_wire(negative) if negative is not None else None,
            "cfg": float(cfg),
        }
        return ({
            "model": model,
            "spec": spec,
            "cfg": float(cfg),
            "uncond_model": model_negative.request_dict(),
            "_dgxm_krea2_ref_preflight": ref_summary,
        },)
