"""Sampler nodes: stock ComfyUI sampler semantics on the cluster.

Widget names, orders and defaults mirror the stock KSampler /
KSamplerAdvanced / SamplerCustomAdvanced nodes so swapping a workflow to the
cluster is a node swap, not a rebuild.
"""
from __future__ import annotations

from ..accuracy_waiver import STAMPED_RESULT_KEY
from ..constants import GUIDER_TYPE, MODEL_TYPE, NODE_CATEGORY
from ..mesh import ensure_live
from ..sampling_contract import custom_schedule_steps
from ..topology import topology_from_preset
from .common import (
    ModelSpec,
    RenderPipeline,
    _restore_requested_levers,
    auto_gate_required,
    conditioning_for_wire,
    run_render,
)
from .consent_waiver import validate_inherited_stamps
from .latent_outputs import clone_latent_samples, is_direct_nested_tensor
from .render_preflight import krea2_ref_preflight_summary
from .sampler_wire import (  # noqa: F401  # Public samplers API.
    _aggregate_render_outputs,
    _canonical_noise_for_wire,
    _module_source_path,
)


def _sampler_result(*outputs: dict):
    """Return normal sampler values, with waiver dispatch ids in prompt history.

    Comfy records a node's ``ui`` payload in this prompt's history. A class-K
    use row is keyed by the dispatch id ``submit_render`` generates; publishing
    that id here lets a sweep join a ledger use row to this prompt, not to any
    row written while the cell was running. With no waiver the return stays
    the plain tuple.
    """
    runs: set[str] = set()
    for output in outputs:
        entries = output.get(STAMPED_RESULT_KEY, ())
        if not isinstance(entries, (list, tuple)):
            continue
        for entry in entries:
            if isinstance(entry, dict) and isinstance(entry.get("run_id"), str):
                if entry["run_id"] and not entry.get("inherited"):
                    runs.add(entry["run_id"])
    if not runs:
        return outputs
    return {"ui": {"dgxm_waiver_runs": sorted(runs)}, "result": outputs}


def _sampler_names():
    import comfy.samplers

    return comfy.samplers.KSampler.SAMPLERS


def _scheduler_names():
    import comfy.samplers

    return comfy.samplers.KSampler.SCHEDULERS


class DGXMonarchKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE, {"tooltip": "mesh model handle from the DGX Monarch loader"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (_sampler_names(),),
                "scheduler": (_scheduler_names(),),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = NODE_CATEGORY

    def sample(self, model: ModelSpec, seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise=1.0, unique_id=None):
        request = {
            "kind": "ksampler",
            "positive": conditioning_for_wire(positive),
            "negative": conditioning_for_wire(negative),
            "noise_seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": float(denoise),
            "_dgxm_krea2_ref_preflight": krea2_ref_preflight_summary(positive, negative),
        }
        out = run_render(model, request, dict(latent_image), cfg_value=float(cfg), steps_hint=int(steps))
        out.pop("_dgxm_denoised", None)
        return _sampler_result(out)


class DGXMonarchKSamplerPipeline:
    """Batched seed sweep with cross-render pipelining (F2, DESIGN.md section
    5.6). The Init node's pipeline_depth sets the overlap; operator guidance is
    in DESCRIPTION. Each seed's latent is byte-identical to DGXMonarchKSampler,
    and the first seed of an ungated combination renders through the
    sequential path, so the sweep cannot dispatch before auto-gating runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE, {"tooltip": "mesh model handle from the DGX Monarch loader"}),
                "seeds": ("STRING", {"default": "0, 1, 2, 3", "multiline": True,
                          "tooltip": "Seeds separated by commas or whitespace, at most 1024. "
                          "Each seed renders one image; the node returns them as one batched LATENT."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (_sampler_names(),),
                "scheduler": (_scheduler_names(),),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = NODE_CATEGORY
    DESCRIPTION = (
        "Renders one image per seed and returns them together as one batch.\n\n"
        "With pipeline_depth at 2 or more on the DGX Monarch Init node, it overlaps "
        "the next render's conditioning transfer, dispatch and latent return with "
        "the current render's compute. That saves time only when those steps are a "
        "large share of each render: video, very large latents, or many fast renders.\n\n"
        "For an ordinary image seed sweep it is not faster than the normal KSampler: "
        "set batch_size on the Empty Latent instead, which is simpler and as fast. "
        "Use this node to render specific seeds (such as 42, 1337, 9000) one by one, "
        "or when the whole batch does not fit in memory at once.\n\n"
        "Leave pipeline_depth at 1, the default, to render the seeds one after "
        "another."
    )

    @staticmethod
    def _parse_seeds(seeds: str) -> list[int]:
        import re

        seed_text = str(seeds).strip()
        if len(seed_text) > 65536:
            raise ValueError("DGXMonarchKSamplerPipeline: seed input is too large")
        toks = [t for t in re.split(r"[,\s]+", seed_text) if t]
        if not toks:
            raise ValueError("DGXMonarchKSamplerPipeline: no seeds given")
        if len(toks) > 1024:
            raise ValueError(
                "DGXMonarchKSamplerPipeline: at most 1024 renders may be queued per prompt")
        parsed = [int(t) for t in toks]
        if any(seed < 0 or seed > 0xFFFFFFFFFFFFFFFF for seed in parsed):
            raise ValueError(
                "DGXMonarchKSamplerPipeline: seeds must be integers in 0..2^64-1")
        return parsed

    def sample(self, model: ModelSpec, seeds, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise=1.0, unique_id=None):
        validate_inherited_stamps(latent_image)
        from .. import adoption_evidence

        adoption_evidence.require_inactive_context("DGXMonarchKSamplerPipeline")
        seed_list = self._parse_seeds(seeds)
        steps = int(steps)
        pos = conditioning_for_wire(positive)
        neg = conditioning_for_wire(negative)
        ref_summary = krea2_ref_preflight_summary(positive, negative)

        # One aggregate ProgressBar across the whole sweep (n renders * steps),
        # advanced by every render's per-step messages through the on_step hook
        # (renders stream concurrently under depth>1, so count steps, not one
        # render's absolute position).
        total = len(seed_list) * steps
        try:
            from comfy.utils import ProgressBar

            bar = ProgressBar(total)
        except Exception:
            bar = None
        counter = {"n": 0}

        def on_step(_msg):
            if bar is not None:
                counter["n"] += 1
                bar.update_absolute(min(counter["n"], total), total)

        depth = int(getattr(model.mesh, "pipeline_depth", 1) or 1)
        if not 1 <= depth <= 8:
            raise ValueError("DGXMonarchKSamplerPipeline: pipeline_depth must be in 1..8")
        def request_for(seed):
            return {
                "kind": "ksampler",
                "positive": pos,
                "negative": neg,
                "noise_seed": int(seed),
                "steps": steps,
                "cfg": float(cfg),
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": float(denoise),
                "_dgxm_krea2_ref_preflight": ref_summary,
            }

        # Render the first seed through the normal sequential path when this
        # combination still needs its first-use ceremony. Otherwise a deep
        # pipeline could dispatch the whole sweep before auto-gating runs.
        outs = []
        remaining = seed_list
        # The risk read below decides whether the first seed runs sequentially,
        # so this combination's own levers have to be back first.
        _restore_requested_levers(model)
        if auto_gate_required(
                model, "ksampler", latent_image, float(cfg)):
            outs.append(run_render(
                model, request_for(seed_list[0]), dict(latent_image),
                cfg_value=float(cfg), steps_hint=steps,
            ))
            remaining = seed_list[1:]
            # The first render may have cached PASS or INCONCLUSIVE. PASS may
            # restore the configured overlap; INCONCLUSIVE stays cached but
            # keeps the rest of this sweep strictly sequential.
            if remaining and auto_gate_required(
                    model, "ksampler", latent_image, float(cfg)):
                depth = 1
            if bar is not None:
                counter["n"] = steps
                bar.update_absolute(counter["n"], total)

        pipe = RenderPipeline(depth=depth, on_step=on_step)
        for seed in remaining:
            pipe.push(model, request_for(seed), dict(latent_image),
                      cfg_value=float(cfg), steps_hint=steps)
        outs.extend(pipe.drain())
        for o in outs:
            o.pop("_dgxm_denoised", None)
        batched = _aggregate_render_outputs(
            outs, "KSampler Pipeline results")
        return _sampler_result(batched)


class DGXMonarchKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "add_noise": (["enable", "disable"],),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (_sampler_names(),),
                "scheduler": (_scheduler_names(),),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = NODE_CATEGORY

    def sample(self, model: ModelSpec, add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, start_at_step, end_at_step,
               return_with_leftover_noise, unique_id=None):
        request = {
            "kind": "ksampler_advanced",
            "positive": conditioning_for_wire(positive),
            "negative": conditioning_for_wire(negative),
            "noise_seed": int(noise_seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": 1.0,
            "advanced": {
                "add_noise": add_noise == "enable",
                "start_at_step": int(start_at_step),
                "end_at_step": int(end_at_step) if int(end_at_step) < 10000 else None,
                "return_with_leftover_noise": return_with_leftover_noise == "enable",
            },
            "_dgxm_krea2_ref_preflight": krea2_ref_preflight_summary(positive, negative),
        }
        out = run_render(model, request, dict(latent_image), cfg_value=float(cfg), steps_hint=int(steps))
        out.pop("_dgxm_denoised", None)
        return _sampler_result(out)


class DGXMonarchBasicScheduler:
    """Stock BasicScheduler semantics computed on the workers (the driver
    holds no model, so SIGMAS come from the resident model's sampling object).
    Needs an explicit topology on the Init node: `auto` resolves at the first
    render, after sigmas are needed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "scheduler": (_scheduler_names(),),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = NODE_CATEGORY

    def get_sigmas(self, model: ModelSpec, scheduler, steps, denoise):
        mesh = model.mesh
        if mesh.topology_preset == "auto":
            raise RuntimeError(
                "DGXMonarchBasicScheduler needs an explicit topology on the Init node: "
                "auto picks one at the first render, after the sigmas are needed. "
                "Set topology to a preset such as ring2, uly2 or cfg2."
            )
        handle = ensure_live(mesh.handle)
        topo = topology_from_preset(mesh.topology_preset, handle.world)
        from .. import mesh_setup
        from .render_session import mutation_render_session

        with mutation_render_session(handle):
            setup_token = mesh_setup.ensure_request_setup(
                handle, topo, mesh.attention, mesh.sync_ulysses,
                mesh.worker_args)
            results = handle.call_all(
                "compute_sigmas", model.request_dict(), scheduler, int(steps),
                float(denoise), timeout_s=1800,
                **mesh_setup.token_kwargs(setup_token))
        # Under dm-cfg2 the non-leader rank returns an empty schedule (it holds
        # no conditional checkpoint), and call_all results are not guaranteed
        # rank-ordered, so take the first populated schedule rather than [0].
        populated = [r for r in results if r is not None and r.numel() > 0]
        return (populated[0] if populated else results[0],)


class DGXMonarchSamplerCustom:
    """SamplerCustomAdvanced semantics: NOISE + GUIDER(spec) + stock SAMPLER +
    SIGMAS. NOISE/SAMPLER/SIGMAS objects go to the workers as they are, with
    one exception: stock RandomNoise/DisableNoise is first rebuilt under its
    canonical importable module, and third-party noise stays by reference.
    The worker wraps the stock SAMPLER and SIGMAS instead of reimplementing
    them (actor/sampling.py). NOISE runs worker-side after
    fix_empty_latent_channels, seed-deterministic on every rank.

    Despite its registered name, this node mirrors stock
    SamplerCustomAdvanced's interface (noise/guider/sampler/sigmas/latent),
    not SamplerCustom's; the name is semver-frozen API (DESIGN.md section 5.7)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": (GUIDER_TYPE,),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = NODE_CATEGORY

    def sample(self, noise, guider, sampler, sigmas, latent_image, unique_id=None):
        model: ModelSpec = guider["model"]
        latent = dict(latent_image)
        wire_noise = _canonical_noise_for_wire(noise)
        schedule_steps = custom_schedule_steps(sigmas)

        request = {
            "kind": "custom",
            "guider": guider["spec"],
            "uncond_model": guider.get("uncond_model"),
            "sampler_object": sampler,
            "sigmas": sigmas.detach().cpu(),
            "noise_seed": int(getattr(noise, "seed", 0) or 0),
            "noise_object": wire_noise,
            "_dgxm_krea2_ref_preflight": guider.get("_dgxm_krea2_ref_preflight"),
        }
        steps_hint = max(schedule_steps, 1)
        out = run_render(model, request, latent, cfg_value=guider.get("cfg"), steps_hint=steps_hint)

        denoised = out.pop("_dgxm_denoised", None)
        denoised_out = dict(out)
        if denoised is None:
            if is_direct_nested_tensor(
                out["samples"], "custom sampler output"
            ) and schedule_steps != 0:
                raise RuntimeError(
                    "packed custom sampler output is missing reconstructed denoised x0"
                )
            denoised = clone_latent_samples(
                out["samples"], "custom sampler fallback output"
            )
        if denoised is out["samples"]:
            raise RuntimeError("custom sampler denoised output aliases the primary samples")
        denoised_out["samples"] = denoised
        return _sampler_result(out, denoised_out)
