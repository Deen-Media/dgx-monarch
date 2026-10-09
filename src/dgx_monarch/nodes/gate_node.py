"""ComfyUI node surface for the DGX Monarch identity gate."""
from __future__ import annotations

import json
from typing import Any

from ..constants import MODEL_TYPE, NODE_CATEGORY
from .samplers import _sampler_names, _scheduler_names


class DGXMonarchIdentityGate:
    """Compare swap-produced weights with a stock fresh-load render."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "noise_seed": (
                    "INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Identity does not depend on the step count: 2 steps "
                        "gate as conclusively as 20 and finish sooner.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                "sampler_name": (_sampler_names(),),
                "scheduler": (_scheduler_names(),),
            },
            "optional": {
                "strict": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "advanced": True,
                        "tooltip": "Raise on any verdict but PASS, NOT RUN included, "
                        "instead of only reporting it, so the node works as a CI "
                        "assertion (ComfyUI-test-framework / comfyci compatible).",
                    },
                ),
                "run_id": (
                    "STRING",
                    {
                        "default": "",
                        "advanced": True,
                        "tooltip": "Optional ID that `dgxm gate` sets to find this run's report.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent (stock lineage)", "report")
    FUNCTION = "gate"
    CATEGORY = NODE_CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def gate(
        self,
        model: Any,
        positive,
        negative,
        latent_image,
        noise_seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        strict=False,
        run_id="",
        unique_id=None,
    ):
        # Resolve nodes.gate at call time so a monkeypatch of its seams reaches this node.
        from . import gate as runtime

        request = {
            "kind": "ksampler_advanced",
            "positive": runtime.conditioning_for_wire(positive),
            "negative": runtime.conditioning_for_wire(negative),
            "noise_seed": int(noise_seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": 1.0,
            "advanced": {
                "add_noise": True,
                "start_at_step": 0,
                "end_at_step": None,
                "return_with_leftover_noise": False,
            },
        }
        from ..adapters.sol_attention import sol_ceremony_skip_reason

        mesh = getattr(model, "mesh", None)
        skip = sol_ceremony_skip_reason(getattr(mesh, "attention", None))
        if skip is not None:
            report = json.dumps(
                {"verdict": "NOT RUN", "reason": skip, "origin": "explicit"},
                indent=1)
            if strict:
                raise RuntimeError(
                    "DGX Monarch identity gate NOT RUN: " + skip)
            return latent_image, report
        result = runtime.run_identity_ceremony(
            model,
            request,
            dict(latent_image),
            float(cfg),
            int(steps),
            origin="explicit",
            run_id=str(run_id),
        )
        latent = result["latent"]
        report = json.dumps(
            {
                key: value for key, value in result.items()
                if key != "latent" and not key.startswith("_")
            },
            indent=1,
        )
        if strict and result["verdict"] != "PASS":
            raise RuntimeError(
                f"DGX Monarch identity gate {result['verdict']}: "
                f"max latent diff {result['max_abs_latent_diff']}")
        return latent, report
