"""GPUWorker: the per-GPU Monarch actor (DESIGN.md §5.2).

Endpoint surface; Monarch preserves every typed exception for the driver.
  setup / apply_worker_args / teardown_group: NCCL + xfuser bring-up, the
    worker-local knob re-apply that leaves live groups alone, and destruction
  load_model / load_uncond_model / gate_swap_cycle / gate_fsdp_reload_cycle /
    unload: §5.5 model store, plus the gate ceremony's swap and reload legs
  sample / cancel_sample: the denoise loop; leader streams progress and
    returns latents, cancel records even while the sample is still queued
  compute_sigmas: stock BasicScheduler against the resident sampling object
  artifact_identity: file/Comfy identity for the driver's pipelined preflight
  capacity_quote: this rank's price for a load, as rows the driver decides
  ack_latent_handoff: retire one generation-bound RDMA descriptor owner
  status / renew_client_lease: snapshot and lease heartbeat, both lock-free
  provenance_baseline: setup-bound GPU-queue/source/event watermark
  clear_vram: soft/hard cache clearing
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import functools
import os
import socket
from typing import Any

# torchmonarch >= 0.6.0 runs plain endpoints inline on the dispatch loop.
# All _gpu_lock users and potentially slow lock-free calls (artifact_identity,
# capacity_quote) must be concurrent endpoints so they cannot block cancellation
# or status. Use concurrent dispatch consistently for lock users: its start-order
# guarantee preserves FIFO lock acquisition in message order.
# Only short lock-free endpoints stay plain: status, renew_client_lease,
# cancel_sample, and registry-only RDMA ACK. They may run before an earlier
# concurrent body starts; _pending_cancels handles cancellation while queued.
from monarch.actor import Actor, concurrent_endpoint, endpoint

from .. import actor_lifetime
from ..loader_options import (
    validate_model_spec_weight_dtype,
    weight_dtype_from_options,
)
from ..log import get_logger
from ..model_sampling import normalize_model_sampling
from ..transfer import LatentReturn
from . import sample_protocol, store_fsdp, worker_env, worker_status
from .cancellation import CancellableSampleMixin
from .failure import cleanup_on_failure
from .gate_fsdp_cycle import FsdpGateCycleMixin
from .model_store import ModelStore, request_artifact_identity
from .rdma_handoff import RDMAHandoffMixin
from .sampling import (
    equalize_cond_lengths,
    model_sampling_render_clone,
    run_custom,
    run_ksampler,
)
from .worker_compile import compile_dit_allowed
from .worker_env import _AttentionDispatch, _maybe_compile_dit
from .worker_status import _latent_signature

log = get_logger(__name__)


class GPUWorker(RDMAHandoffMixin, FsdpGateCycleMixin, CancellableSampleMixin, Actor):
    def __init__(self) -> None:
        self.store = ModelStore()
        self.rank: int | None = None
        self.world: int | None = None
        self.topology: dict[str, Any] = {}  # instance lifetime: replaced whole by each setup
        self._setup_key: tuple | None = None
        self._setup_generation: int | None = None
        self._setup_provenance: tuple[int, dict] | None = None
        self._setup_cleanup_failed = False
        self._attn = _AttentionDispatch()
        self._latent_return = LatentReturn("message")
        self._init_rdma_handoffs()
        self._active_worker_args: dict = {}  # instance lifetime: the knobs the last setup applied
        # Serialize GPU/store/parallel-state mutation; one executor thread pins
        # every CUDA/NCCL call while async endpoints keep supervision responsive.
        self._gpu_lock = asyncio.Lock()
        self._gpu_exec = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dgxm-gpu")
        self._init_cancellation()

    async def _on_gpu(self, fn, *args):
        """Run a blocking GPU, CUDA or NCCL body on the GPU thread, in this actor's context.

        run_in_executor does not copy contextvars, and RDMABuffer(...) reads
        context().actor_instance (monarch/_src/rdma/rdma.py), which falls back to
        the client context when unset. The copy below is the one asyncio.to_thread
        makes; without it an RDMA latent return binds to the client, with no error.
        """
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(
            self._gpu_exec, functools.partial(ctx.run, fn, *args))

    @concurrent_endpoint
    async def setup(self, env: dict) -> dict:
        async with self._gpu_lock:
            return await self._on_gpu(self._setup_impl, env)

    @concurrent_endpoint
    async def apply_worker_args(self, worker_args: dict) -> dict:
        """Re-apply worker-local knobs (memory posture, safetensors backend, LoRA
        residency, DiT compile and others) without touching NCCL or xfuser. The
        mesh calls this when only worker_args changed, so a memory toggle
        mid-session never tears down the parallel groups the next render needs."""
        async with self._gpu_lock:
            return await self._on_gpu(self._apply_worker_args_impl, worker_args)

    def _apply_worker_args_impl(self, worker_args: dict) -> dict:
        return worker_env.apply_worker_args_impl(self, worker_args)

    _nccl_launch_order_needed = staticmethod(worker_env.nccl_launch_order_needed)
    _slab_mode_effective = staticmethod(worker_env.slab_mode_effective)

    def _setup_impl(self, env: dict) -> dict:
        return worker_env.setup_impl(self, env)

    def _setup_info(self) -> dict:
        return worker_env.setup_info(self)

    def _teardown_parallel_state(self) -> None:
        worker_env.teardown_parallel_state(self)

    @concurrent_endpoint
    async def artifact_identity(self, model_specs: list[dict]) -> dict:
        """Report local file/Comfy identities without touching CUDA state.

        It stays outside ``_gpu_lock`` so a pipelined driver's safety
        preflight can run while the actor's dedicated GPU thread is denoising.
        It reads bounded file windows only and never mutates the model store.
        """
        return {
            "host": socket.gethostname(),
            "rank": self.rank,
            "artifact_sets": [
                request_artifact_identity(spec["unet_name"], spec.get("loras"))
                for spec in model_specs
            ],
        }

    @concurrent_endpoint
    async def capacity_quote(self, request: dict) -> dict:
        """Price a load on this rank without loading it, and never raise.

        Outside ``_gpu_lock`` and off the GPU thread for the same reason as
        ``artifact_identity``: it mutates nothing, takes no CUDA work, and a
        fleet-wide question must not queue behind a denoise. A capacity refusal
        is data here and the driver owns the raise, because a worker raise
        crosses the actor boundary as a traceback where the driver needs a row
        naming the rank.
        """
        from . import worker_capacity

        return worker_capacity.quote(self, request)

    def _check_uma_reserve(self) -> None:
        worker_status.check_uma_reserve(self)

    def _inject_for_topology(
        self,
        base_patcher,
        quant_kind: str,
        lora_stack,
        precision_evidence,
    ) -> None:
        """Bind the family adapter onto a freshly loaded base (once per load)."""
        from ..adapters import (
            adapter_for,
            attention_capability,
            cfg_dispatch,
            cfg_parallel,
            quant_activation_scale,
            sol_attention,
        )
        from ..adapters.fsdp import apply_fsdp_capacity_mode
        from ..family_select import override_for_worker
        from . import attention_context, partial_load_guard
        store_fsdp.validate_injection(
            self,
            quant_kind,
            lora_stack,
        )
        partial_load_guard.install(base_patcher, getattr(self, "world", None))
        cfg_degree = int(self.topology.get("cfg", 1))
        adapter = adapter_for(base_patcher.model, override_for_worker(self))
        sp = int(self.topology.get("ulysses", 1)) * int(self.topology.get("ring", 1))
        # dm-cfg2: a dual_model_cfg family (ig4) runs the split guider at
        # cfg2/sp1/world2, one model per rank, so grant cfg-parallel here and
        # keep the batched cond/uncond wrapper off below (a batch-1 dual-model
        # forward would trip the wrapper's batch < world guard).
        # Mirrors sample_protocol._dual_model_cfg2_topology, fsdp clause included:
        # the slot decision and the adapter grant must never diverge.
        dual_model_cfg = (
            getattr(adapter, "dual_model_cfg_supported", False)
            and cfg_degree == 2 and sp == 1
            and int(getattr(self, "world", 0) or 0) == 2
            and not bool(self.topology.get("fsdp")))
        cfg_parallel.assert_cfg_parallel_supported(
            adapter, cfg_degree, dual_model_cfg=dual_model_cfg)
        cfg_parallel.assert_sp_quant_supported(
            adapter, sp, quant_kind, cfg=cfg_degree, dual_model_cfg=dual_model_cfg)

        if sp > 1:
            family = getattr(adapter, "family", "unknown")
            head_dim = getattr(adapter, "attention_head_dim", None)
            # The family decides which kernel can carry its head; bind it before
            # the view is taken, so a substitution reaches the object the render
            # attends through.
            self._attn.bind_capability(family, head_dim)
            attention = getattr(adapter, "attention_dispatch", lambda value: value)(self._attn)
            # Bind on the object this render attends through: the Wan overrides
            # return their own view instead of the dispatch. The worker's adapter
            # is the authority; the driver only sniffed a family from the
            # checkpoint. _AttentionDispatch.sol_scope says why the family is
            # recorded under every kernel.
            sol_attention.bind_sol_scope(attention, family)
            # Rank for rank, before the forward binds and before any collective.
            attention_capability.assert_kernel_capability(
                self._attn.effective_kernel, family, head_dim, base_patcher.model)
            ctx = attention_context.injection_context(self.topology, sp, attention)
            adapter.inject_usp(base_patcher.model.diffusion_model, ctx)
        elif cfg_degree > 1 and hasattr(adapter, "inject_cfg_pad_forward"):
            # cfg-parallel with asymmetric prompts: the pad-aware forward is a
            # no-op without padding, so installing it is always safe.
            adapter.inject_cfg_pad_forward(base_patcher.model.diffusion_model)

        # The shared nvfp4 activation scale wraps Linears after injection and
        # before compilation; the sample path reads the coverage it records.
        quant_activation_scale.install(
            base_patcher.model.diffusion_model, self.topology,
            family=getattr(adapter, "family", "model"), dual_model_cfg=dual_model_cfg)

        # Slab-resident bases stay eager; _maybe_compile_dit compiles others only under compile_dit.
        if compile_dit_allowed(base_patcher):
            _maybe_compile_dit(base_patcher.model.diffusion_model)
        elif os.environ.get("DGXM_COMPILE_DIT") == "1":
            log.info("compile_dit: slab-resident model stays eager")

        if cfg_degree > 1 and not dual_model_cfg:
            # The seam is the adapter's declaration, not one constant: comfy
            # applies DIFFUSION_MODEL wrappers only where the family's own
            # forward builds the executor, so three families take APPLY_MODEL
            # instead and the rest keep the seam they were measured on.
            cfg_parallel.install_cfg_parallel_wrapper(
                adapter, base_patcher.model_options)
            # And, for every family, the seam one call further out: comfy hands
            # every family's cond list to CALC_COND_BATCH before any model call,
            # so each rank can run its own conditioning where the conds do not
            # concatenate and there is no batched call to slice. A render
            # whose conds do fold passes through to the slice untouched.
            cfg_dispatch.install_cond_dispatch_wrapper(
                adapter, base_patcher.model_options)

        if self.topology.get("fsdp"):
            apply_fsdp_capacity_mode(
                base_patcher,
                quant_kind,
                lora_stack,
                precision_evidence,
                prefetch_depth=int(getattr(self, "fsdp_prefetch_depth", 1) or 1),
                lora_low_rss=(None if getattr(self, "store", None) is None
                              else bool(self.store.lora_low_rss)),
            )

    @concurrent_endpoint
    async def load_model(self, unet_name: str, options: dict | None = None,
                         lora_stack: list | None = None) -> dict:
        weight_dtype_from_options(options)
        async with self._gpu_lock:
            return await self._on_gpu(self._load_model_impl, unet_name, options, lora_stack)

    @cleanup_on_failure
    def _load_model_impl(self, unet_name: str, options: dict | None = None,
                         lora_stack: list | None = None) -> dict:
        if self._setup_key is None:
            raise RuntimeError("load_model before setup(): the Init node must run first")
        _, transition = store_fsdp.ensure(
            self, unet_name, options, lora_stack, slot="cond", on_base_loaded=self._inject_for_topology)
        return {
            "host": socket.gethostname(),
            "rank": self.rank,
            "transition": transition,
            "artifact_sets": [request_artifact_identity(unet_name, lora_stack)],
            **self.store.snapshot(),
        }

    @concurrent_endpoint
    async def load_uncond_model(self, unet_name: str, options: dict | None = None,
                                lora_stack: list | None = None) -> dict:
        weight_dtype_from_options(options)
        async with self._gpu_lock:
            return await self._on_gpu(self._load_uncond_model_impl, unet_name, options, lora_stack)

    @cleanup_on_failure
    def _load_uncond_model_impl(self, unet_name: str, options: dict | None = None,
                                lora_stack: list | None = None) -> dict:
        if self._setup_key is None:
            raise RuntimeError("load_uncond_model before setup(): the Init node must run first")
        _, transition = store_fsdp.ensure(
            self, unet_name, options, lora_stack, slot="uncond",
            on_base_loaded=self._inject_for_topology)
        return {
            "host": socket.gethostname(),
            "rank": self.rank,
            "transition": transition,
            "artifact_sets": [request_artifact_identity(unet_name, lora_stack)],
            **self.store.snapshot(),
        }

    @concurrent_endpoint
    async def gate_swap_cycle(self, unet_name: str, options: dict | None = None,
                              lora_stack: list | None = None,
                              expected_artifact_identity: dict | None = None) -> dict:
        weight_dtype_from_options(options)
        async with self._gpu_lock:
            return await self._on_gpu(
                self._gate_swap_cycle_impl, unet_name, options, lora_stack,
                expected_artifact_identity)

    @cleanup_on_failure
    def _gate_swap_cycle_impl(self, unet_name: str, options: dict | None = None,
                              lora_stack: list | None = None,
                              expected_artifact_identity: dict | None = None) -> dict:
        from .gate_cycle import run

        return run(self, unet_name, options, lora_stack, expected_artifact_identity)

    @concurrent_endpoint
    async def unload(self) -> dict:
        async with self._gpu_lock:
            return await self._on_gpu(self._unload_impl)

    def _unload_impl(self) -> dict:
        self.store.unload_all()
        # Do not force-free RDMA ownership here: tokenized HandoffRegistry
        # entries retain backing until exact generation/token ACK or actor
        # recycle. Only legacy no-handoff LatentReturn retention is bounded to
        # pipeline depth. A generation lease blocks replacement through reads.
        return {"host": socket.gethostname(), "unloaded": True}

    @cleanup_on_failure
    def _sample_impl(self, request: dict, progress_port=None, cancel_event=None) -> dict:
        # The seams are read here, from this module's globals, at every call,
        # so a patch on actor.worker reaches the body that lives in
        # sample_protocol.py. Passing them as bare names is what keeps that true.
        return sample_protocol.run_sample(
            self, request, progress_port, cancel_event,
            equalize_cond_lengths=equalize_cond_lengths,
            model_sampling_render_clone=model_sampling_render_clone,
            request_artifact_identity=request_artifact_identity,
            run_ksampler=run_ksampler,
            run_custom=run_custom,
            latent_signature=_latent_signature,
        )

    @concurrent_endpoint
    async def compute_sigmas(self, model_spec: dict, scheduler: str, steps: int, denoise: float = 1.0):
        """Stock BasicScheduler semantics against the resident model's sampling
        object (the driver holds no model, so SIGMAS come from the workers)."""
        validate_model_spec_weight_dtype(model_spec)
        async with self._gpu_lock:
            return await self._on_gpu(self._compute_sigmas_impl, model_spec, scheduler, steps, denoise)

    def _dual_model_cfg2_nonleader(self, model_spec: dict) -> bool:
        """Whether this rank is a non-leader cfg rank under dm-cfg2.

        Under dm-cfg2 only cfg rank 0 holds the conditional checkpoint. A
        non-leader returns an empty schedule instead of loading it, and the
        driver keeps the first non-empty one (nodes/samplers.py).
        """
        topo = self.topology
        sp = int(topo.get("ulysses", 1)) * int(topo.get("ring", 1))
        if (int(topo.get("cfg", 1)) != 2 or sp != 1 or int(topo.get("dp", 1)) != 1
                or int(getattr(self, "world", 0) or 0) != 2):
            return False
        from ..adapters import family_supports_dual_model_cfg
        from ..adapters.base import cfg_rank

        if cfg_rank() == 0:
            return False
        family = sample_protocol._sniff_request_family(self, model_spec)
        return family is not None and family_supports_dual_model_cfg(family)

    @cleanup_on_failure
    def _compute_sigmas_impl(self, model_spec: dict, scheduler: str, steps: int, denoise: float = 1.0):
        if self._setup_key is None:
            raise RuntimeError("compute_sigmas before setup(): the Init node must run first")
        import comfy.samplers

        if self._dual_model_cfg2_nonleader(model_spec):
            import torch

            return torch.FloatTensor([])
        model_sampling = normalize_model_sampling(model_spec.get("model_sampling"))

        patcher, _ = store_fsdp.ensure(
            self, model_spec["unet_name"], model_spec.get("options"),
            model_spec.get("loras"), slot="cond", on_base_loaded=self._inject_for_topology)

        total_steps = int(steps)
        if denoise < 1.0:
            if denoise <= 0.0:
                import torch

                return torch.FloatTensor([])
            total_steps = int(steps / denoise)
        render_patcher = model_sampling_render_clone(patcher, model_sampling)
        sampling_object = render_patcher.get_model_object("model_sampling")
        sigmas = comfy.samplers.calculate_sigmas(
            sampling_object, scheduler, total_steps).cpu()
        return sigmas[-(int(steps) + 1):]

    @endpoint
    async def status(self) -> dict:
        # Run inline without the GPU lock so status stays responsive during a
        # render: it mutates nothing and takes no CUDA work. The reads are not
        # free (host stats hit /proc and sysfs per call; the first call per
        # process hashes the source tree, cached after), so keep anything
        # unbounded out of worker_status.status_impl.
        return self._status_impl()

    def _status_impl(self) -> dict:
        return worker_status.status_impl(self)

    @endpoint
    async def renew_client_lease(self, lease_s: float) -> None:
        # Plain @endpoint for the same reason as status(): instant, lock-free,
        # and touching no worker state, so a client heartbeat can never queue
        # behind a render. The reaper it holds off runs on its own thread.
        actor_lifetime.renew(lease_s)

    @concurrent_endpoint
    async def provenance_baseline(
        self, setup_generation: int, artifact_manifest: object = None
    ) -> dict:
        """Linearize a source/event baseline behind prior GPU mutations.

        Unlike diagnostic ``status()``, this waits for the GPU queue, and the
        impl rejects a call whose setup generation no longer matches.
        """
        async with self._gpu_lock:
            if artifact_manifest is None:
                return await self._on_gpu(
                    self._provenance_baseline_impl,
                    setup_generation,
                )
            return await self._on_gpu(
                self._provenance_baseline_impl,
                setup_generation,
                artifact_manifest,
            )

    def _provenance_baseline_impl(
        self, setup_generation: int, artifact_manifest: object = None
    ) -> dict:
        return worker_status.provenance_baseline_impl(
            self, setup_generation, artifact_manifest
        )

    def _memory_detail(self) -> dict:
        return worker_status.memory_detail(self)

    @concurrent_endpoint
    async def clear_vram(self, level: str = "soft") -> dict:
        """soft: free allocator cache; hard: also unload every resident model."""
        async with self._gpu_lock:
            return await self._on_gpu(self._clear_vram_impl, level)

    def _clear_vram_impl(self, level: str = "soft") -> dict:
        if self._setup_key is None:
            # Before setup there is no allocator to clear, and touching CUDA here
            # would bind the wrong device (worker_status.status_impl).
            return self.status_snapshot()
        import comfy.model_management as mm

        if level == "hard":
            self.store.unload_all()
            # RDMA ownership is left alone, as in _unload_impl.
        mm.soft_empty_cache()
        import gc

        gc.collect()
        return self.status_snapshot()

    def status_snapshot(self) -> dict:
        """The host and model-store summary clear_vram returns; it touches no CUDA."""
        return {"host": socket.gethostname(), "models": self.store.snapshot()}

    @concurrent_endpoint
    async def teardown_group(self, release_models: bool = True) -> dict:
        async with self._gpu_lock:
            return await self._on_gpu(self._teardown_group_impl, release_models)

    def _teardown_group_impl(self, release_models: bool = True) -> dict:
        torn_down = worker_env.teardown_existing_setup(self, release_models)
        return {
            "host": socket.gethostname(),
            "torn_down": torn_down,
            "cleanup_state": "DIRTY" if getattr(self, "_setup_cleanup_failed", False) else "UNSETUP",
        }
