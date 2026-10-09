"""Price the second load required by an FSDP clean-reload proof.

Read every rank through the read-only, setup-independent ``status`` endpoint,
then apply the worker FSDP price to the lowest MemAvailable. Telemetry rounds
to 0.01 GiB, introducing about 5 MiB of uncertainty. The pre-unload reading
assumes an in-process unload returns no memory to the host pool; if that
changes, the estimate becomes conservative and the next prompt reprices.

Make no claim for dtype casts, a non-integrated driver GPU, an unmeasurable
checkpoint, or any rank without readable MemAvailable. The integrated-device
check covers only the driver: a discrete driver disables pricing for the
fleet, while an integrated driver can also price a discrete worker.

Refusals use the worker guard's class and id. They are not waivable: that
guard's rescue selects slab residency, which FSDP disables. Price rows grant
no authority.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import mesh_safety
from .adapters.fsdp import fsdp_required_bytes
from .loader_options import weight_dtype_from_options
from .log import get_logger
from .refusal import RefusalClass, parse_refusal_tag, refusal
from .transfer_utils import failure_summary, safe_call

log = get_logger(__name__)

# Frozen in ``gate_audit_vocab.MEASURED_PROBES`` and ``refusal.GUARDS``: the
# price mirrors that worker guard on the driver side, so it reports the same
# probe name and refuses under the same guard id.
PROBE = "stock_load_preflight"

# The operator entry both sentences point at.
TROUBLESHOOTING_ENTRY = 88

# The ledger row's vocabulary; both values are already in use elsewhere. The
# name says STOP so no import can confuse it with
# ``gate_audit_vocab.CAPACITY_VERDICT`` ("CAPACITY_CERTIFIED"), an audit
# verdict on a different key.
CAPACITY_CROSS_MODE = "CAPACITY"
CAPACITY_STOP_VERDICT = "INCONCLUSIVE"

STATUS_TIMEOUT_S = 60.0

_SKIP_DTYPE_CAST = "a loader dtype cast changes the resident size"
_SKIP_DISCRETE = "the driver CUDA device is not integrated"
_SKIP_NO_CHECKPOINT = "the checkpoint file could not be measured here"
_SKIP_NO_STATUS = "no rank reported a readable MemAvailable"


@dataclass(frozen=True, slots=True)
class ReloadPrice:
    """Whether one more load of this checkpoint fits on the leanest rank."""

    applies: bool
    fits: bool
    checkpoint_bytes: int = 0
    required_bytes: int = 0
    mem_available_bytes: int | None = None
    rank: int | None = None
    skipped_reason: str = ""

    @staticmethod
    def no_claim(reason: str) -> ReloadPrice:
        return ReloadPrice(False, True, skipped_reason=reason)

    @property
    def checkpoint_gib(self) -> float:
        return round(self.checkpoint_bytes / (1 << 30), 1)

    @property
    def required_gib(self) -> float:
        return round(self.required_bytes / (1 << 30), 1)

    @property
    def mem_available_gib(self) -> float | None:
        if self.mem_available_bytes is None:
            return None
        return round(self.mem_available_bytes / (1 << 30), 1)

    @property
    def headroom_bytes(self) -> int:
        if self.mem_available_bytes is None:
            return 0
        return int(self.mem_available_bytes - self.required_bytes)

    def measured(self) -> dict[str, Any]:
        """Measurement block in the field names capacity rows already carry."""
        return {
            "probe": PROBE,
            "checkpoint_bytes": int(self.checkpoint_bytes),
            "mem_available_bytes": (None if self.mem_available_bytes is None
                                    else int(self.mem_available_bytes)),
            "required_bytes": int(self.required_bytes),
            "headroom_bytes": self.headroom_bytes,
            "weights_gib": self.checkpoint_gib,
            "required_gib": self.required_gib,
            "mem_available_gib": self.mem_available_gib,
            "rank": self.rank,
        }


def checkpoint_bytes(unet_name: str) -> int:
    """Size of the named diffusion model on disk, or 0 when unmeasurable."""
    try:
        import folder_paths

        from .driver_footprint import file_size_bytes

        return file_size_bytes(folder_paths.get_full_path("diffusion_models", unet_name))
    except Exception:
        return 0


def checkpoint_kind(unet_name: str) -> str | None:
    """Stored precision for the shared FSDP build price, or unknown."""
    try:
        import folder_paths

        from .adapters.detect import sniff_fsdp_launch_quant_proof

        path = folder_paths.get_full_path("diffusion_models", unet_name)
        return sniff_fsdp_launch_quant_proof(path).quant_kind if path else None
    except Exception:
        return None


def _rank_mem_available_bytes(row: Any) -> int | None:
    """MemAvailable in bytes from one worker status row, or None."""
    if not isinstance(row, Mapping):
        return None
    host = row.get("host")
    mem = host.get("mem_gib") if isinstance(host, Mapping) else None
    value = mem.get("MemAvailable") if isinstance(mem, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        return None
    return int(value * (1 << 30))


def _leanest_rank(rows: Any, world: Any) -> tuple[int, int] | None:
    """(rank, MemAvailable bytes) for the rank with the least left."""
    if not isinstance(rows, list) or not rows:
        return None
    if isinstance(world, int) and not isinstance(world, bool) and world >= 1 and len(rows) != world:
        return None
    leanest: tuple[int, int] | None = None
    for index, row in enumerate(rows):
        available = _rank_mem_available_bytes(row)
        if available is None:
            return None
        rank = row.get("rank") if isinstance(row, Mapping) else None
        named = rank if isinstance(rank, int) and not isinstance(rank, bool) else index
        if leanest is None or available < leanest[1]:
            leanest = (named, available)
    return leanest


def _requests_a_cast(options: Any) -> bool:
    """Whether this request changes the resident dtype, under either name.

    A driver request names ``weight_dtype``; the ComfyUI options the worker
    builds from it name ``dtype``. The guard this price mirrors reads the
    worker name and skips itself on any cast, so a price that read only that
    name would price every driver-side cast the guard lets through. This reads
    both names and counts an option it cannot read as a cast, so the price
    makes no claim it cannot justify.
    """
    if isinstance(options, Mapping) and options.get("dtype") is not None:
        return True
    try:
        return weight_dtype_from_options(options) != "default"
    except Exception:
        return True


def price_fsdp_clean_reload(handle: Any, model_request: Mapping[str, Any]) -> ReloadPrice:
    """Price one more load of this checkpoint against every rank's free memory."""
    options = model_request.get("options") or {}
    if _requests_a_cast(options):
        return ReloadPrice.no_claim(_SKIP_DTYPE_CAST)
    if not mesh_safety.gpu_is_integrated():
        return ReloadPrice.no_claim(_SKIP_DISCRETE)
    size = checkpoint_bytes(str(model_request.get("unet_name") or ""))
    if size <= 0:
        return ReloadPrice.no_claim(_SKIP_NO_CHECKPOINT)
    try:
        rows = handle.call_all("status", timeout_s=STATUS_TIMEOUT_S)
    except Exception as exc:
        safe_call(log.warning,
                  "FSDP clean-reload price could not read fleet status (%r); "
                  "the ceremony proceeds unpriced", exc)
        return ReloadPrice.no_claim(_SKIP_NO_STATUS)
    leanest = _leanest_rank(rows, getattr(handle, "world", None))
    if leanest is None:
        return ReloadPrice.no_claim(_SKIP_NO_STATUS)
    rank, available = leanest
    kind = checkpoint_kind(str(model_request.get("unet_name") or ""))
    required = fsdp_required_bytes(size, getattr(handle, "world", None), kind)
    price = ReloadPrice(True, required <= available, size, required, available, rank)
    # Log both outcomes. If the price logged only on a refusal, a price that
    # fits would look like one that made no claim, and the per-rank numbers
    # would exist only for proofs that failed.
    safe_call(log.info,
              "FSDP clean-reload price: %s. Leanest rank %s reports %.1f GiB "
              "available; one more load of the %.1f GiB checkpoint needs about "
              "%.1f GiB, leaving %.1f GiB",
              "fits" if price.fits else "does not fit", price.rank,
              price.mem_available_gib, price.checkpoint_gib, price.required_gib,
              price.headroom_bytes / (1 << 30))
    return price


def capacity_classification(exc: BaseException) -> str:
    """Return a bounded failure classification without copying exception text.

    ActorError representations can contain remote paths and hostnames. Store
    only the exception type and any refusal class/guard in the durable ledger;
    retain full details in the driver log and operator message.
    """
    name = type(exc).__name__[:64]
    try:
        tag = parse_refusal_tag(failure_summary(exc))
    except Exception:
        tag = None
    if tag is None:
        return f"{name}/untagged"
    return f"{name}/class-{tag.refusal_class.value} guard={tag.guard or 'none'}"


def operator_sentence(
    price: ReloadPrice, unet_name: str, *, stage: str = "preflight",
) -> str:
    """Describe the capacity refusal at the correct stage of the proof.

    Preflight runs before mutation. Repricing runs after the baseline render,
    which may have loaded the driver's text encoder. Its message must therefore
    acknowledge that work has already occurred.
    """
    if stage == "reprice":
        middle = (
            "The ceremony stopped before its reload cycle. It had already "
            "unloaded your shards, loaded the model again and run one "
            "baseline render; this stop releases that residency and does not "
            "start the reload. Re-queueing runs the proof again and pays for "
            "that render again. "
        )
    else:
        middle = (
            "The proof was not started: nothing was "
            "unloaded and nothing was rendered. "
        )
    return refusal(
        RefusalClass.CAPACITY,
        f"the FSDP clean-reload proof for {unet_name} needs one more load of "
        f"the {price.checkpoint_gib:.1f} GiB checkpoint beside the shards "
        f"already resident, about {price.required_gib:.1f} GiB, but rank "
        f"{price.rank} reports only {price.mem_available_gib:.1f} GiB of "
        "unified memory available. "
        f"{middle}"
        "A checkpoint this large cannot be proved in place on unified memory. A "
        "fresh-process proof was declined on 2026-08-26: only a whole-fleet "
        "replace frees the memory a load keeps after its unload, and a replace "
        "unloads every model. Set auto_gate=off on the Init node: it is the "
        "intended path for this checkpoint, not a fallback, and it runs this "
        "render ungated. What else fits: free unified memory that is not this "
        "render's own model (another resident model, or the driver's text "
        "encoder), or a smaller bf16 or fp16 checkpoint (an fp8 or int8 file "
        "is priced at 1.2 times its size).",
        guard=PROBE, waivable=False, troubleshooting=TROUBLESHOOTING_ENTRY,
    )


def _in_flight_sentence(unet_name: str, detail: str) -> str:
    return refusal(
        RefusalClass.CAPACITY,
        f"the FSDP clean-reload proof for {unet_name} was refused for "
        f"capacity while it ran: {detail} Your shards were already unloaded, "
        "so re-queueing runs the whole proof again, its baseline render "
        "included. What fits: free unified memory that is not "
        "this render's own model, a smaller checkpoint, or auto_gate=off on "
        "the Init node, which skips the ceremony rather than passing it.",
        guard=PROBE, waivable=False, troubleshooting=TROUBLESHOOTING_ENTRY,
    )


def record_capacity_stop(
    ledger: Any,
    key: str,
    artifacts: Any,
    commit: Any,
    capability_context: Any,
    unet_name: str,
    *,
    origin: str,
    run_id: str,
    price: ReloadPrice | None = None,
    exc: BaseException | None = None,
    stage: str = "preflight",
) -> str:
    """Record one INCONCLUSIVE capacity row and return the operator sentence.

    A priced stop carries the numbers the driver measured. An in-flight stop
    records ``measured: None`` and a bounded classification of what refused,
    never the refusing text (see :func:`capacity_classification`); the
    operator reads that text in the sentence and the driver log. The worker
    stock, slab and FSDP launch walls append a measured tag to their refusals,
    which this row does not parse.
    """
    detail = "" if exc is None else failure_summary(exc)[:200]
    classification = "" if exc is None else capacity_classification(exc)
    sentence = (operator_sentence(price, unet_name, stage=stage)
                if price is not None
                else _in_flight_sentence(unet_name, detail))
    row = {
        "model": unet_name,
        "loras": 0,
        "origin": origin,
        "run_id": run_id,
        "max_abs_latent_diff": None,
        "quarantine_levers": [],
        "cross_mode": CAPACITY_CROSS_MODE,
        "inconclusive_reasons": [
            ("the FSDP clean-reload proof was repriced after the baseline "
             "render and does not fit"
             if stage == "reprice"
             else "the FSDP clean-reload proof was priced and does not fit")
            if price is not None
            else "the FSDP clean-reload proof was refused for capacity while it ran"
        ],
        "measured": None if price is None else price.measured(),
        "capacity_detail": classification or None,
    }
    try:
        ledger.record(key, artifacts, commit, CAPACITY_STOP_VERDICT, row,
                      capability_context)
    except Exception as record_exc:
        safe_call(log.warning,
                  "could not persist the FSDP clean-reload capacity row: %r",
                  record_exc)
    if detail:
        safe_call(log.error,
                  "FSDP clean-reload proof was refused in flight by %s: %s",
                  classification, detail)
    safe_call(log.error, "FSDP clean-reload proof stopped on capacity: %s", sentence)
    return sentence
