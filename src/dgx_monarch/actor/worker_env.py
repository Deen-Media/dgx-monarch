"""GPUWorker environment setup and teardown for ComfyUI, NCCL, and xFuser."""
from __future__ import annotations

import os
import socket
import time

from ..accuracy_waiver import RESULT_KEY as _ACCURACY_WAIVER_RESULT_KEY
from ..log import get_logger
from ..transfer import LatentReturn
from ..transfer_utils import (
    failure_summary,
    raise_with_distinct_cause,
    reconcile_error,
    safe_note,
)
from . import fsdp_checkpoint_pin, load_fault, rendezvous
from .attention_dispatch import _AttentionDispatch as _AttentionDispatch
from .comfy_bridge import ensure_comfy
from .worker_authorization import (
    accuracy_waiver_stamps as accuracy_waiver_stamps,
)
from .worker_authorization import (
    activate_accuracy_waivers as activate_accuracy_waivers,
)
from .worker_authorization import (
    assert_resident_artifact_identity as assert_resident_artifact_identity,
)
from .worker_authorization import (
    sample_rescue_consent as sample_rescue_consent,
)
from .worker_authorization import (
    verify_sample_artifact_authorization as verify_sample_artifact_authorization,
)
from .worker_compile import maybe_compile_dit as _maybe_compile_dit  # noqa: F401

log = get_logger(__name__)
ACCURACY_WAIVER_RESULT_KEY = _ACCURACY_WAIVER_RESULT_KEY


def apply_worker_args_impl(worker, worker_args: dict) -> dict:
    from .comfy_bridge import (
        _CUSTOM_NODES_DISABLED,
        _apply_worker_args,
        _uma_memory_defaults,
    )

    if _CUSTOM_NODES_DISABLED is not None and (
        bool(worker_args.get("disable_custom_nodes"))
        is not _CUSTOM_NODES_DISABLED
    ):
        raise RuntimeError(
            "disable_custom_nodes is a bootstrap policy; reset the Attached mesh "
            "before changing it"
        )

    wa = _uma_memory_defaults(dict(worker_args or {}))
    _apply_worker_args(wa)
    compile_changed = (os.environ.get("DGXM_COMPILE_DIT") == "1") is not bool(
        wa.get("compile_dit"))
    # The shared activation scale wraps forwards at load, so a resident built
    # under the previous setting cannot serve this one either.
    from ..adapters.quant_activation_scale import env_from_worker_args

    scale_changed = env_from_worker_args(wa, getattr(worker, "_active_worker_args", None))
    if wa.get("compile_dit"):
        os.environ["DGXM_COMPILE_DIT"] = "1"
    else:
        os.environ.pop("DGXM_COMPILE_DIT", None)
    if wa.get("load_profile"):
        os.environ["DGXM_LOAD_PROFILE"] = "1"
    else:
        os.environ.pop("DGXM_LOAD_PROFILE", None)
    # Residency setters invalidate only their own state; NCCL groups remain live.
    from ..family_select import override_from_worker_args

    # The adapter binds once per load, so a family change drops every resident.
    family_override = override_from_worker_args(wa)
    family_changed = family_override != worker.store.family_override
    if family_changed:
        log.info("family adapter: %s -> %s (residents dropped)",
                 worker.store.family_override or "auto", family_override or "auto")
        worker.store.family_override = family_override
    worker.store.set_lora_mode(bool(wa.get("lora_low_rss")))
    _topology = getattr(worker, "topology", None) or {}
    worker.store.set_slab_mode(worker._slab_mode_effective(wa, _topology),
                               slab_mode_reason(wa, _topology))
    worker.store.swap_verify = int(wa.get("swap_verify", worker.store.swap_verify))
    worker.fsdp_prefetch_depth = int(wa.get("fsdp_prefetch_depth", 1))
    worker.uma_reserve_gb = float(wa.get("uma_reserve_gb", getattr(worker, "uma_reserve_gb", 0.0)))
    worker.store.uma_reserve_gb = worker.uma_reserve_gb
    if compile_changed or family_changed or scale_changed:
        # Each binds at model load, so prior residents cannot serve this policy.
        worker.store.unload_all()
    else:
        worker.store.release_retained_cleanup()
    worker._active_worker_args = dict(worker_args or {})
    return {"lora_low_rss": worker.store.lora_low_rss,
            "slab_weights": worker.store.slab_weights}


def nccl_launch_order_needed(topo: dict) -> bool:
    """Return whether FSDP and model parallelism share ranks.

    That topology needs ``NCCL_LAUNCH_ORDER_IMPLICIT`` to prevent communicator
    deadlock. NCCL latches it at the process's first launch, so setup enables it
    for every topology, including an initial single-group setup.
    """
    model_parallel = (int(topo.get("ulysses", 1)) * int(topo.get("ring", 1))
                      * int(topo.get("cfg", 1)))
    return bool(topo.get("fsdp")) and model_parallel > 1


def slab_mode_reason(wa: dict, topology: dict) -> str:
    """Explain why effective residency fell back to stock.

    The model store carries this reason into capacity refusals. Empty means no
    compatibility gate disabled slab residency.
    """
    if not wa.get("slab_weights"):
        return "slab_weights is off in the effective worker args"
    if topology.get("fsdp"):
        return ("FSDP is active for this render and reshards weights into DTensors, "
                "so a slab would be an unused extra copy")
    if not wa.get("lora_low_rss"):
        return ("lora_low_rss is off in the effective worker args, and slab residency "
                "requires it")
    return ""


def slab_mode_effective(wa: dict, topology: dict) -> bool | str:
    """Resolve slab policy against topology: ``False``, ``True`` or ``"auto"``.

    Auto defers the family decision to model load time.
    """
    want = wa.get("slab_weights")
    if not want:
        return False
    if topology.get("fsdp"):
        log.info("slab_weights: disabled under FSDP (weights shard to DTensors)")
        return False
    # Compile compatibility depends on the loaded model's blocks, so the
    # residency ladder decides it before each fresh load from the family memo:
    # an unknown or compilable family stays stock, and a known eager no-op may
    # keep the requested slab policy.
    return "auto" if want == "auto" else True


def teardown_existing_setup(worker, release_models: bool = True) -> bool:
    """Invalidate a published setup before tearing its process groups down.

    Clear setup identity first so an interrupted teardown cannot be reused as
    a matching live setup. When teardown raises or keeps the models, the
    cleanup-failed latch stays set until an explicit teardown or recycle.
    """
    had_setup = worker._setup_key is not None
    was_dirty = bool(getattr(worker, "_setup_cleanup_failed", False))
    if not had_setup and not was_dirty:
        return False
    # Publish dirty state before clearing identity so interruption cannot
    # acknowledge surviving groups as clean.
    worker._setup_cleanup_failed = True
    worker._setup_key = None
    worker._setup_generation = None
    worker._setup_provenance = None
    worker.rank, worker.world, worker.topology = None, None, {}
    # Retired RDMA handoffs stay owned until their generation-bound ACK or actor recycle.
    if release_models:
        worker._teardown_parallel_state()
    else:  # the caller stops this process next; its exit frees the weights
        teardown_parallel_state(worker, release_models=False)
    worker._setup_cleanup_failed = not release_models  # held models stay dirty
    return had_setup or was_dirty


def _setup_key(env: dict, fabric: dict[str, str]) -> tuple:
    """Identity for persistent setup state; live worker policy is separate."""
    return (
        env["setup_generation"], env["rank"], env["world"], env["master_addr"],
        env["master_port"], tuple(sorted(env["topology"].items())), tuple(sorted(fabric.items())),
        os.path.expanduser(env["comfy_dir"]), env.get("gpus_per_host", 1),
        env.get("local_gpu_index", 0), env.get("attention", "TORCH_FLASH"),
        bool(env.get("sync_ulysses", True)),
        bool(env.get("rdma_latent_return", False)), env.get("rdma_min_bytes"),
        int(env.get("pipeline_depth", 1) or 1),
    )


def setup_impl(worker, env: dict) -> dict:
    """Bring up NCCL and xFuser parallel state on one GPU.

    env keys: setup_generation, rank, world, master_addr, master_port, topology{...},
    fabric_env{...}, comfy_dir, worker_args{...}, gpus_per_host, local_gpu_index,
    attention, sync_ulysses, rdma_latent_return, rdma_min_bytes, pipeline_depth,
    nccl_timeout_s.

    Setup is idempotent only for an exact persistent identity; live worker
    policy is applied separately. A different key retires bound forwards.
    """
    if getattr(worker, "_setup_cleanup_failed", False):
        raise RuntimeError(
            "the last NCCL setup was not fully torn down; tear down or recycle "
            "the worker before another setup"
        )
    from ..config_schema import MAX_GPUS_PER_HOST, MAX_WORLD_SIZE

    world = env.get("world")
    rank = env.get("rank")
    setup_generation = env.get("setup_generation")
    gpus_per_host = env.get("gpus_per_host", 1)
    local_gpu_index = env.get("local_gpu_index", 0)
    if isinstance(world, bool) or not isinstance(world, int) or not 1 <= world <= MAX_WORLD_SIZE:
        raise ValueError(f"worker setup world must be in 1..{MAX_WORLD_SIZE} (got {world!r})")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world:
        raise ValueError(f"worker setup rank must be in 0..{world - 1} (got {rank!r})")
    if (isinstance(setup_generation, bool)
            or not isinstance(setup_generation, int)
            or setup_generation < 1):
        raise ValueError(
            "worker setup generation must be a positive integer "
            f"(got {setup_generation!r})")
    if (isinstance(gpus_per_host, bool) or not isinstance(gpus_per_host, int)
            or not 1 <= gpus_per_host <= MAX_GPUS_PER_HOST):
        raise ValueError(
            f"worker setup gpus_per_host must be in 1..{MAX_GPUS_PER_HOST} "
            f"(got {gpus_per_host!r})")
    if (isinstance(local_gpu_index, bool) or not isinstance(local_gpu_index, int)
            or not 0 <= local_gpu_index < gpus_per_host):
        raise ValueError(
            f"worker setup local_gpu_index must be in 0..{gpus_per_host - 1} "
            f"(got {local_gpu_index!r})")
    from ..config_schema import validate_fabric_env

    fabric = validate_fabric_env(env.get("fabric_env") or {}, "fabric_env")
    key = _setup_key(env, fabric)
    if worker._setup_key == key:
        return worker._setup_info()
    had_existing_setup = worker._setup_key is not None

    # Validation above changes no state; only a valid replacement tears the old setup down.
    from ..config import fixup_fabric_ifaces

    fabric, rail_note = fixup_fabric_ifaces(fabric)
    if rail_note:
        log.warning("%s", rail_note)

    # A setup-key change retires the prior generation before any new policy or
    # environment can be applied. A later bootstrap refusal must not leave the
    # old key authorizing samples against partially changed process state.
    if had_existing_setup:
        teardown_existing_setup(worker)
    # Re-arm immediately before the first new-generation mutation so any
    # failure remains explicitly dirty until teardown/recycle settles it.
    worker._setup_cleanup_failed = True
    load_fault.arm(rank)

    # Apply device and fabric environment before CUDA or ComfyUI imports. Each
    # worker replaces a configured rail only when that interface is absent or down.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(env.get("local_gpu_index", 0))
    os.environ.update(fabric)
    ensure_comfy(
        os.path.expanduser(env["comfy_dir"]),
        env.get("worker_args"),
        gpus_per_host=int(env.get("gpus_per_host") or 1),
    )
    # Resolve auto low-RSS policy per host; mode changes rebuild residents.
    from .comfy_bridge import _uma_memory_defaults as _uma_defaults

    _wa_eff = _uma_defaults(dict(env.get("worker_args") or {}))
    from ..family_select import override_from_worker_args

    # Setup starts with no resident, so this is a plain assignment; the eviction
    # on a later change belongs to apply_worker_args_impl above.
    worker.store.family_override = override_from_worker_args(_wa_eff)
    worker.store.set_lora_mode(bool(_wa_eff.get("lora_low_rss")))
    worker.store.set_slab_mode(worker._slab_mode_effective(_wa_eff, env["topology"]),
                               slab_mode_reason(_wa_eff, env["topology"]))
    worker.store.swap_verify = int(_wa_eff.get("swap_verify", worker.store.swap_verify))
    worker.fsdp_prefetch_depth = int(_wa_eff.get("fsdp_prefetch_depth", 1))
    worker.uma_reserve_gb = worker.store.uma_reserve_gb = float(_wa_eff.get("uma_reserve_gb", 0.0))
    if _wa_eff.get("load_profile"):
        os.environ["DGXM_LOAD_PROFILE"] = "1"
    else:
        os.environ.pop("DGXM_LOAD_PROFILE", None)
    from ..adapters.quant_activation_scale import env_from_worker_args

    env_from_worker_args(env.get("worker_args") or {})
    if (env.get("worker_args") or {}).get("compile_dit"):
        # Persist compile policy for the later load endpoint in this process.
        os.environ["DGXM_COMPILE_DIT"] = "1"
    else:
        os.environ.pop("DGXM_COMPILE_DIT", None)

    from datetime import timedelta

    import torch
    import torch.distributed as dist
    from xfuser.core.distributed import init_distributed_environment, initialize_model_parallel

    # Using all 20 Spark cores starves host I/O and dispatch. The measured
    # baseline reserves four cores unless the operator overrides it.
    torch.set_num_threads(int(os.environ.get("DGXM_NUM_THREADS", "16")))

    topo = env["topology"]
    rank, world = int(env["rank"]), int(env["world"])
    # Set before communicator creation on every setup and never clear by topology:
    # NCCL latches this value at its first launch, including single-group launches.
    # The single GPU executor supplies deterministic host order. See
    # nccl_launch_order_needed for the FSDP/model-parallel requirement.
    os.environ["NCCL_LAUNCH_ORDER_IMPLICIT"] = "1"
    t0 = time.perf_counter()
    try:
        if not dist.is_initialized():
            dist.init_process_group(
                "nccl",
                rank=rank, world_size=world,
                store=rendezvous.generation_store(env, rank, world, setup_generation),
                timeout=timedelta(seconds=int(env.get("nccl_timeout_s", 600))),
            )
        init_distributed_environment(rank=rank, world_size=world, local_rank=0)
        initialize_model_parallel(
            data_parallel_degree=int(topo.get("dp", 1)),
            sequence_parallel_degree=int(topo.get("ulysses", 1)) * int(topo.get("ring", 1)),
            classifier_free_guidance_degree=int(topo.get("cfg", 1)),
            ulysses_degree=int(topo.get("ulysses", 1)),
            ring_degree=int(topo.get("ring", 1)),
        )
        # Publish setup identity only after attention, transfer, metadata, and
        # CUDA introspection all succeed.
        worker._attn.configure(
            env.get("attention", "TORCH_FLASH"),
            bool(env.get("sync_ulysses", True)),
            topology=topo,
            world=world,
            setup_generation=setup_generation,
        )
        latent_return = LatentReturn(
            # LatentReturn's own default is "rdma", so every construction site
            # must spell the off value; env.get's False keeps an omitted key off.
            "rdma" if env.get("rdma_latent_return", False) else "message",
            min_bytes=env.get("rdma_min_bytes"),
            keepalive_depth=int(env.get("pipeline_depth", 1) or 1),
        )
        worker.rank, worker.world, worker.topology = rank, world, dict(topo)
        # The residency ladder prices an FSDP shard build per rank.
        worker.store.world = world
        from .comfy_bridge import _CUSTOM_NODES_DISABLED

        setup_provenance = None
        if _CUSTOM_NODES_DISABLED is True:
            from .worker_status import runtime_provenance_snapshot

            setup_provenance = runtime_provenance_snapshot()
        info = worker._setup_info()
        info.update({
            "bringup_s": round(time.perf_counter() - t0, 2),
            "cuda": torch.cuda.get_device_name(0),
        })
        worker._latent_return = latent_return
        worker._setup_generation = setup_generation
        worker._setup_provenance = (
            (setup_generation, setup_provenance)
            if setup_provenance is not None
            else None
        )
        worker._setup_key = key
        worker._active_worker_args = dict(env.get("worker_args") or {})
        worker._setup_cleanup_failed = False
    except BaseException as setup_exc:
        # Tear down every partial group before surfacing the setup failure;
        # otherwise retries can wedge on already-initialized state.
        worker._setup_cleanup_failed = True
        worker._setup_key = None
        worker._setup_generation = None
        worker._setup_provenance = None
        worker.rank, worker.world, worker.topology = None, None, {}
        cleanup_failed = False
        failure: BaseException = setup_exc
        failure_cause: BaseException | None = None
        try:
            worker._teardown_parallel_state()
        except BaseException as teardown_exc:
            cleanup_failed = True
            failure, failure_cause = reconcile_error(
                failure,
                teardown_exc,
                "post-failure parallel-state teardown also failed",
            )
            try:
                log.warning(
                    "post-failure parallel-state teardown: %s",
                    failure_summary(teardown_exc),
                )
            except BaseException:
                pass
        initialized = False
        try:
            initialized = dist.is_initialized()
        except BaseException as probe_exc:
            cleanup_failed = True
            failure, failure_cause = reconcile_error(
                failure,
                probe_exc,
                "post-failure process-group state probe also failed",
            )
        if initialized:
            try:
                dist.destroy_process_group()
            except BaseException as pg_exc:
                cleanup_failed = True
                failure, failure_cause = reconcile_error(
                    failure,
                    pg_exc,
                    "post-failure process-group teardown also failed",
                )
                try:
                    log.warning(
                        "post-failure PG teardown: %s",
                        failure_summary(pg_exc),
                    )
                except BaseException:
                    pass
        worker._setup_cleanup_failed = cleanup_failed
        worker._active_worker_args = {}
        if failure is setup_exc and failure_cause is None:
            raise
        raise_with_distinct_cause(failure, failure_cause)

    log.info("setup complete: rank %d/%d, topology %s", rank, world, topo)
    return info


def setup_info(worker) -> dict:
    from xfuser.core.distributed import (
        get_classifier_free_guidance_rank,
        get_data_parallel_rank,
        get_sequence_parallel_rank,
    )

    from ..adapters.sol_attention import sol_attn_available

    return {
        "host": socket.gethostname(),
        "rank": worker.rank,
        "world": worker.world,
        "topology": dict(worker.topology),
        # Optional kernels are reported per rank so the driver can refuse a
        # rank-asymmetric selection before dispatch. A raise inside a
        # collective on one rank only abandons the lease and strands its peers.
        "sol_attn": sol_attn_available(),
        "sp_rank": get_sequence_parallel_rank(),
        "cfg_rank": get_classifier_free_guidance_rank(),
        "dp_rank": get_data_parallel_rank(),
    }


def teardown_parallel_state(worker, release_models: bool = True) -> None:
    from xfuser.core.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )

    # Attempt every stage. The distributed-environment finalizer owns the torch
    # world group and must run even after model-parallel teardown fails.
    attention = getattr(worker, "_attn", None)
    invalidate_attention = getattr(attention, "invalidate", lambda: None)
    unload = worker.store.unload_all if release_models else (
        lambda: fsdp_checkpoint_pin.close_store_pins(worker.store))
    stages = (
        ("model unload", unload),
        ("attention dispatch", invalidate_attention),
        ("model-parallel groups", destroy_model_parallel),
        ("distributed environment", destroy_distributed_environment),
    )
    errors: list[tuple[str, BaseException]] = []
    for name, destroy in stages:
        try:
            destroy()
        except BaseException as exc:
            errors.append((name, exc))
            try:
                log.warning("%s teardown failed: %s", name, failure_summary(exc))
            except BaseException:
                pass
    if errors:
        detail = "; ".join(
            f"{name}: {failure_summary(exc)}" for name, exc in errors
        )
        interruption = next(
            (exc for _name, exc in errors if not isinstance(exc, Exception)), None
        )
        if interruption is not None:
            safe_note(interruption, "parallel-state teardown incomplete", detail)
            cause = next(
                (exc for _name, exc in errors if exc is not interruption),
                None,
            )
            raise_with_distinct_cause(interruption, cause)
        raise RuntimeError(f"parallel-state teardown incomplete ({detail})") from errors[0][1]
