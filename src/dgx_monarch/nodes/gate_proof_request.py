"""The short proof request a first-use ceremony renders, built from the user's own.

The automatic gate proves a combination with the request the render sends,
cut to at most two steps. A KSampler request becomes a KSampler Advanced call
of up to two steps, as it always has. A Sampler Custom request keeps its own
call: the same guider, SAMPLER and NOISE objects and seed, with its sigma
schedule cut to the first two steps. Both replay through ``run_render`` like
any other render, so every leg of the ceremony makes the model calls the
render it vouches for makes.
"""
from __future__ import annotations

from typing import Any

from ..sampling_contract import custom_schedule_steps

# The request kinds the first-use gate covers: every sampler node's kind
# (nodes/samplers.py). Read-only after import.
AUTO_GATED_KINDS = frozenset({"ksampler", "ksampler_advanced", "custom"})

# The most denoise steps a proof render takes.
PROOF_STEPS = 2


def proof_request(request: dict) -> tuple[dict, int] | None:
    """Return the proof request and its step count, or None when there is nothing to render.

    None means a Sampler Custom schedule of one sigma: it runs no denoise
    step, so a proof would compare two copies of the input latent. The
    caller runs no ceremony; the render then finds no PASS and dispatches
    stock, or refuses under FSDP, which is the fail-closed answer.
    """
    light = dict(request)
    if request.get("kind") == "custom":
        sigmas = request["sigmas"]
        steps = custom_schedule_steps(sigmas)
        if steps == 0:
            return None
        if steps > PROOF_STEPS:
            light["sigmas"] = sigmas[:PROOF_STEPS + 1].clone()
        return light, min(steps, PROOF_STEPS)
    light["steps"] = min(PROOF_STEPS, int(request.get("steps") or PROOF_STEPS))
    light["advanced"] = {
        "add_noise": True,
        "start_at_step": 0,
        "end_at_step": None,
        "return_with_leftover_noise": False,
    }
    light["kind"] = "ksampler_advanced"
    return light, light["steps"]


def report_fields(request: dict) -> dict[str, Any]:
    """The sampling fields a gate report row names for this proof request.

    A Sampler Custom request names no sampler, scheduler or step count, and
    its cfg sits in the guider, so the row reads them from the call itself.
    """
    if request.get("kind") == "custom":
        sigmas = request.get("sigmas")
        guider = request.get("guider") or {}
        try:
            steps: int | None = custom_schedule_steps(sigmas)
        except ValueError:  # the report must never turn a verdict into an abort
            steps = None
        return {
            "seed": request.get("noise_seed"),
            "steps": steps,
            "cfg": guider.get("cfg", 1.0) if isinstance(guider, dict) else None,
            "sampler": "custom",
            "scheduler": None,
        }
    return {
        "seed": request.get("noise_seed"),
        "steps": request.get("steps"),
        "cfg": request.get("cfg"),
        "sampler": request.get("sampler_name"),
        "scheduler": request.get("scheduler"),
    }
