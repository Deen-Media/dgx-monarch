"""Loader nodes: describe what should be resident on the workers.

DiT weights never enter the driver process. With an explicit topology the base
model loads eagerly at node execution, except a dm-cfg2 slot, which defers to
the sample path; with `auto` the load happens on the first render (topology
must exist before load so USP injection binds at load). LoRA nodes only extend
the request signature. The model store (DESIGN.md section 5.5) turns a LoRA
tweak into a hot-swap, never a reload.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .. import first_render
from ..adapters.detect import sniff_fsdp_launch_quant_proof
from ..adapters.fsdp import (
    refuse_fsdp_live_cast,
    validate_fsdp_explicit_bf16_source,
    validate_fsdp_launch_loras,
    validate_fsdp_launch_quant,
)
from ..constants import MESH_TYPE, MODEL_TYPE, NODE_CATEGORY
from ..loader_options import SUPPORTED_WEIGHT_DTYPES, validate_weight_dtype
from ..log import get_logger
from ..mesh import ensure_live
from ..mesh_safety import ArtifactBindingError
from ..safetensors_header import SafetensorsFileIdentity
from ..topology import Topology, topology_from_preset
from . import consent_projection, loader_graph, loader_preflight, render_validation
from .common import MeshSpec, ModelSpec

log = get_logger(__name__)

# Widget vocabulary, read-only after import.
_WEIGHT_DTYPES = list(SUPPORTED_WEIGHT_DTYPES)


@dataclass(frozen=True, slots=True)
class _FSDPCheckpointBinding:
    path: str
    unet_name: str
    file_identity: SafetensorsFileIdentity


def _filename_list(kind: str):
    import folder_paths

    return folder_paths.get_filename_list(kind)


def _preflight_explicit_fsdp_checkpoint(
    topo: Topology,
    unet_name: str,
    weight_dtype: str,
) -> _FSDPCheckpointBinding | None:
    """Keep the FSDP precision refusal typed and local to the driver."""
    weight_dtype = validate_weight_dtype(weight_dtype)
    if not topo.fsdp:
        return None

    import folder_paths

    path = folder_paths.get_full_path("diffusion_models", unet_name)
    if path is None:
        raise FileNotFoundError(
            f"model {unet_name!r} not found on the driver; FSDP launch needs its "
            "checkpoint header"
        )
    validate_fsdp_explicit_bf16_source(str(path), weight_dtype)
    if weight_dtype.startswith("fp8"):
        refuse_fsdp_live_cast(weight_dtype)
    proof = sniff_fsdp_launch_quant_proof(path)
    if proof.quant_kind is None:
        validate_fsdp_launch_quant("unknown")
    else:
        validate_fsdp_launch_quant(proof.quant_kind)
    if proof.file_identity is None:
        raise ArtifactBindingError(
            f"could not bind the driver FSDP checkpoint header for model {unet_name!r}"
        )
    return _FSDPCheckpointBinding(
        path=str(path),
        unet_name=unet_name,
        file_identity=proof.file_identity,
    )


def _assert_explicit_fsdp_checkpoint_binding(
    binding: _FSDPCheckpointBinding,
) -> None:
    """Refuse when the logical model no longer names the admitted inode."""
    import folder_paths

    try:
        resolved = folder_paths.get_full_path(
            "diffusion_models", binding.unet_name)
        if resolved is None:
            raise FileNotFoundError(binding.unet_name)
        current = SafetensorsFileIdentity.from_stat(os.stat(os.fspath(resolved)))
    except (OSError, TypeError) as exc:
        raise ArtifactBindingError(
            f"model {binding.unet_name!r} changed after the driver FSDP "
            "checkpoint header preflight"
        ) from exc
    if current != binding.file_identity:
        raise ArtifactBindingError(
            f"model {binding.unet_name!r} changed after the driver FSDP "
            "checkpoint header preflight"
        )


class DGXMonarchUNETLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh": (MESH_TYPE,),
                "unet_name": (_filename_list("diffusion_models"),),
                "weight_dtype": (_WEIGHT_DTYPES, {"default": "default", "advanced": True}),
            },
            # `prompt` is comfy's published whole-graph hidden input. No
            # dgx-monarch request names the text encoder or the VAEs, so the
            # graph is the only place the driver-side footprint preflight can
            # price them before they are resident.
            "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"},
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = NODE_CATEGORY
    SLOT = "cond"
    ENDPOINT = "load_model"

    def _defers_for_dual_model_cfg2(self, mesh: MeshSpec, unet_name: str,
                                    topo: Topology) -> bool:
        """Whether this eager load is a dm-cfg2 slot to defer to the sample path.

        True only for an explicit cfg2 world-2 topology whose checkpoint family
        runs the split guider (Ideogram4). Then each rank loads one checkpoint at
        sample time (cond on rank 0, uncond on rank 1); an eager call_all here
        would load this slot on every rank and defeat the residency halving. An
        unreadable header falls to the normal eager load, never a refusal.
        """
        if not (topo.cfg == 2 and topo.ulysses == 1 and topo.ring == 1
                and topo.world == 2):
            return False
        try:
            import folder_paths

            from ..adapters import family_supports_dual_model_cfg
            from ..adapters.detect import CheckpointSniffError, sniff_checkpoint
            from ..family_select import effective_family, override_from_worker_args

            path = folder_paths.get_full_path("diffusion_models", unet_name)
            if path is None:
                return False
            forced = override_from_worker_args(getattr(mesh, "worker_args", None))
            try:
                family = effective_family(sniff_checkpoint(str(path))[0], forced)
            except CheckpointSniffError:
                return False
            return family_supports_dual_model_cfg(family)
        except Exception:
            return False

    def load(self, mesh: MeshSpec, unet_name: str, weight_dtype: str = "default",
             unique_id=None, prompt=None):
        weight_dtype = validate_weight_dtype(weight_dtype)
        options = {}
        if weight_dtype != "default":
            options["weight_dtype"] = weight_dtype
        spec = ModelSpec(mesh=mesh, unet_name=unet_name, options=options, slot=self.SLOT)

        # Heal a mesh an earlier worker death evicted. Explicit loaders reassert
        # their requested topology and bind the eager load to that generation.
        # The callback sees the freshly parsed prospective world and runs before
        # cache transition, old-fleet shutdown, actor spawn, setup, or RPC.
        mesh_preflight = None
        if mesh.topology_preset != "auto":
            def validate_explicit_mesh(_config, world: int) -> None:
                topo = topology_from_preset(mesh.topology_preset, int(world))
                binding = _preflight_explicit_fsdp_checkpoint(
                    topo, unet_name, weight_dtype)
                if binding is not None:
                    _assert_explicit_fsdp_checkpoint_binding(binding)

            mesh_preflight = validate_explicit_mesh
        # Runs first: this artifact cannot run on any world, topology or
        # budget, so it answers before the comfy-managed policy card, the
        # footprint card, and ensure_live healing or spawning a fleet.
        loader_preflight.preflight_upstream_gated_artifact(unet_name)
        # This runs above the footprint card; preflight_graph_batch_divides_dp says why.
        render_validation.preflight_graph_batch_divides_dp(mesh, prompt)
        # Above the footprint card, which would otherwise offer slab residency
        # where slab is not a lever (see preflight_comfy_managed_topology).
        loader_preflight.preflight_comfy_managed_topology(mesh)
        # Above ensure_live: a capacity refusal must land before any mesh, gate
        # or session side effect, so before the fleet is healed or spawned and
        # before mutation_render_session pushes topology and worker args to
        # every rank. Inside that session it would still be typed and caught,
        # but only after a full bring-up. run_render and the pipeline keep the
        # same placement.
        rescue_load = loader_preflight.preflight_loader_footprint(
            mesh, unet_name, options, prompt)
        if rescue_load is None:
            # Always project consent: worker capacity checks may offer rescue even
            # when this family-scoped driver estimator made no claim. Otherwise an
            # accepted card would never affect the load.
            rescue_load = consent_projection.project_for_loader(
                mesh, unet_name, options,
                covered=loader_graph.unet_names(prompt))
        handle = ensure_live(mesh.handle, mesh_preflight=mesh_preflight)
        if mesh.topology_preset != "auto":
            topo = topology_from_preset(mesh.topology_preset, handle.world)
            from .. import capacity_agreement, mesh_setup
            from . import render_preflight
            from .render_session import mutation_render_session

            render_preflight.preflight_sol_sequence_parallel(topo, mesh.attention)
            if self._defers_for_dual_model_cfg2(mesh, unet_name, topo):
                # Setup stays eager (topology is known), but the load defers to
                # the first render so the sample path loads one checkpoint per
                # rank. record_rescue_row is skipped with the load, mirroring the
                # auto path (no call_all results to certify).
                with mutation_render_session(handle):
                    mesh_setup.ensure_request_setup(
                        handle, topo, mesh.attention, mesh.sync_ulysses,
                        mesh.worker_args)
                first_render.load_deferred(self.ENDPOINT, reason="explicit cfg2 dm-cfg2")
                return (spec,)
            # Setup and eager adoption are one residency transaction. No
            # Fleet/pipeline gap may replace topology between these calls.
            with mutation_render_session(handle):
                setup_token = mesh_setup.ensure_request_setup(
                    handle, topo, mesh.attention, mesh.sync_ulysses,
                    mesh.worker_args)
                # After ensure_request_setup: before setup a rank's slab policy
                # is still the previous session's, and a consent projected
                # above is not yet visible to it. The SLOT rides the request
                # because the uncond loader inherits this method, so a slotless
                # quote would price an uncond load against the cond slot.
                capacity_agreement.agree_load(
                    handle, unet_name, options, [], self.SLOT)
                results = handle.call_all(
                    self.ENDPOINT, unet_name, options, [], timeout_s=1800,
                    **mesh_setup.token_kwargs(setup_token))
            for r in results:
                log.info("%s: %s -> %s", self.ENDPOINT, r.get("host"), r.get("transition"))
            # The SLOT matters: the worker's snapshot carries a certificate for
            # both model slots, and the uncond loader inherits this method, so
            # folding "cond" here would stamp the uncond row with the cond
            # checkpoint's digest.
            loader_preflight.record_rescue_row(
                rescue_load, results, handle.world, slot=self.SLOT, handle=handle)
        else:
            first_render.load_deferred(self.ENDPOINT)
        return (spec,)


class DGXMonarchUncondUNETLoader(DGXMonarchUNETLoader):
    """Second model slot for dual-model asymmetric CFG (Ideogram4)."""

    SLOT = "uncond"
    ENDPOINT = "load_uncond_model"


def _preflight_fsdp_lora_checkpoint_property(unet_name: object) -> None:
    """Refuse a LoRA stack on an unadmitted FSDP checkpoint before dispatch.

    Header-only, so this answers before the worker's own launch contract and
    before any first-use ceremony can run: a comfy-kitchen quantized
    checkpoint never admits a LoRA stack under FSDP, on every
    ``lora_low_rss``/``auto_gate`` setting. fp32 islands alone do not refuse
    here: the header cannot show whether an island casts live, so only the
    in-bake ``_file_tensor`` check in ``actor/fsdp_lora.py`` refuses one that
    does. A missing or non-string name fails open: the worker-side backstop
    (``adapters.fsdp_lora_admission.refuse_unless_fsdp_lora_admits``) and the
    lever check right after this call remain.
    """
    if not isinstance(unet_name, str):
        return
    import folder_paths

    path = folder_paths.get_full_path("diffusion_models", unet_name)
    if path is None:
        return  # resolve_topology's own FileNotFoundError covers this later
    from ..adapters.detect import fsdp_lora_admission_property
    from ..adapters.fsdp_lora_admission import refuse_fsdp_lora_checkpoint_property

    property_name = fsdp_lora_admission_property(path)
    if property_name is not None:
        refuse_fsdp_lora_checkpoint_property(property_name)


class DGXMonarchLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "lora_name": (_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = NODE_CATEGORY

    def load_lora(self, model: ModelSpec, lora_name: str, strength_model: float, unique_id=None):
        if strength_model == 0.0:
            return (model,)
        if model.mesh.topology_preset != "auto":
            topo = topology_from_preset(
                model.mesh.topology_preset,
                model.mesh.world,
            )
            if topo.fsdp:
                _preflight_fsdp_lora_checkpoint_property(
                    getattr(model, "unet_name", None))
                # The graph's own worker overrides are visible here; the
                # cluster policy and the worker's check decide the rest.
                requested = getattr(model.mesh, "worker_args", None) or {}
                validate_fsdp_launch_loras(
                    (lora_name,), lora_low_rss=requested.get("lora_low_rss"))
        return (model.with_lora(lora_name, strength_model),)
