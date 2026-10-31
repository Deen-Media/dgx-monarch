"""Price the next load an FSDP clean-reload proof makes.

Read every rank through the read-only, setup-independent ``status`` endpoint:
the host's MemAvailable, rounded to 0.01 GiB, and the Worker's torch reserved
bytes, ``vram.reserved_gib``, rounded to 0.1 GiB.

The proof unloads every rank before each load it prices. The ceremony unloads
the user's shards before the baseline load, and the cycle unloads the baseline
before the reload. So the price credits each rank's resident shard pool: torch
reserved less 0.05 GiB of rounding, capped at one share of the checkpoint. The
cap keeps sampling reservations out of the credit. A rank fits when the price
is at most its MemAvailable plus that credit, and the leanest rank after the
credit decides. A missing or malformed reading, or an unknown world, credits
nothing. On 2026-10-13 both unloads of a 24.5 GiB bf16 proof returned the
whole share to the host (docs/VALIDATION.md, FSDP calibration).

The Worker guard prices the same load again after the real unload
(``actor/store_residency.preload_capacity_check``) and stays decisive. A credit
the unload does not honor becomes that guard's in-flight class C refusal, not
an over-commit. A preflight refusal still unloads and renders nothing.

Make no claim for dtype casts, a non-integrated driver GPU, an unmeasurable
checkpoint, or any rank without readable MemAvailable. The integrated-device
check covers only the driver: a discrete driver disables pricing for the
fleet, while an integrated driver can also price a discrete worker.

Refusals use the worker guard's class and id. They are not waivable: that
guard's rescue selects slab residency, which FSDP disables. Price rows grant
no authority.
"""
from __future__ import annotations

import math
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

# ``vram.reserved_gib`` is rounded to 0.1 GiB, so this much comes off each
# reading before it is credited: the credit never exceeds what torch holds.
_RESERVED_ROUNDING_GIB = 0.05


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
    # The leanest rank's resident shard pool, which the proof drops before
    # this load. Last, so the positional constructors keep their meaning.
    credit_bytes: int = 0

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
    def credit_gib(self) -> float:
        return round(self.credit_bytes / (1 << 30), 1)

    @property
    def headroom_bytes(self) -> int:
        """MemAvailable plus the credit, less the price."""
        if self.mem_available_bytes is None:
            return 0
        return int(self.mem_available_bytes + self.credit_bytes - self.required_bytes)

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
            "credit_bytes": int(self.credit_bytes),
            "credit_gib": self.credit_gib,
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


def _share_bytes(size: int, world: Any) -> int:
    """One rank's share of the checkpoint, or 0 when the world is unknown."""
    if isinstance(world, bool) or not isinstance(world, int) or world < 1:
        return 0
    return int(size) // world


def _rank_credit_bytes(row: Any, share_bytes: int) -> int:
    """The resident shard pool this rank's next unload drops, in bytes.

    Torch reserved less the rounding margin, capped at one share. A missing,
    None, boolean, negative or non-numeric reading credits 0.
    """
    vram = row.get("vram") if isinstance(row, Mapping) else None
    value = vram.get("reserved_gib") if isinstance(vram, Mapping) else None
    if share_bytes <= 0 or isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    try:
        reading = float(value) - _RESERVED_ROUNDING_GIB
    except OverflowError:
        return 0
    if not math.isfinite(reading) or reading <= 0:
        return 0
    capped = min(reading, share_bytes / (1 << 30))
    return min(int(capped * (1 << 30)), share_bytes)


def _leanest_rank(
    rows: Any, world: Any, share_bytes: int = 0,
) -> tuple[int, int, int] | None:
    """(rank, MemAvailable bytes, credit bytes) for the rank with the least
    left once its resident shard pool is credited."""
    if not isinstance(rows, list) or not rows:
        return None
    if isinstance(world, int) and not isinstance(world, bool) and world >= 1 and len(rows) != world:
        return None
    leanest: tuple[int, int, int] | None = None
    for index, row in enumerate(rows):
        available = _rank_mem_available_bytes(row)
        if available is None:
            return None
        credit = _rank_credit_bytes(row, share_bytes)
        rank = row.get("rank") if isinstance(row, Mapping) else None
        named = rank if isinstance(rank, int) and not isinstance(rank, bool) else index
        if leanest is None or available + credit < leanest[1] + leanest[2]:
            leanest = (named, available, credit)
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
    world = getattr(handle, "world", None)
    leanest = _leanest_rank(rows, world, _share_bytes(size, world))
    if leanest is None:
        return ReloadPrice.no_claim(_SKIP_NO_STATUS)
    rank, available, credit = leanest
    kind = checkpoint_kind(str(model_request.get("unet_name") or ""))
    required = fsdp_required_bytes(size, world, kind)
    price = ReloadPrice(True, required <= available + credit, size, required,
                        available, rank, credit_bytes=credit)
    # Log both outcomes. If the price logged only on a refusal, a price that
    # fits would look like one that made no claim, and the per-rank numbers
    # would exist only for proofs that failed. Keep the text up to "leaving"
    # stable: probe parsers read it.
    safe_call(log.info,
              "FSDP clean-reload price: %s. Leanest rank %s reports %.1f GiB "
              "available; one more load of the %.1f GiB checkpoint needs about "
              "%.1f GiB, leaving %.1f GiB. It counted %.1f GiB available plus "
              "%.1f GiB of resident shards the proof drops first",
              "fits" if price.fits else "does not fit", price.rank,
              price.mem_available_gib, price.checkpoint_gib, price.required_gib,
              price.headroom_bytes / (1 << 30), price.mem_available_gib,
              price.credit_gib)
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


def classification_class(detail: object) -> str | None:
    """The refusal class one :func:`capacity_classification` value names.

    None for an untagged, malformed or missing value, so a reader that needs a
    typed refusal fails closed on anything this module did not write.
    """
    if not isinstance(detail, str):
        return None
    _name, marker, rest = detail.partition("/class-")
    named = rest.split(" ", 1)[0] if marker else ""
    return named if named in {member.value for member in RefusalClass} else None


def operator_sentence(
    price: ReloadPrice, unet_name: str, *, stage: str = "preflight",
) -> str:
    """Describe the capacity refusal at the correct stage of the proof.

    Preflight runs before mutation. Repricing runs after the baseline render,
    which may have loaded the driver's text encoder. Its message must therefore
    acknowledge that work has already occurred. Both name the credit applied.
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
    counted = (
        "That estimate already counts this render's own shards, up to one "
        "share. What is short is other memory, such as the driver's text "
        "encoder or another resident model, or this render's sampling "
        "reservations, which the estimate does not count. "
        if price.credit_bytes > 0 else ""
    )
    return refusal(
        RefusalClass.CAPACITY,
        f"the FSDP clean-reload proof for {unet_name} needs one more load of "
        f"the {price.checkpoint_gib:.1f} GiB checkpoint, about "
        f"{price.required_gib:.1f} GiB, but rank {price.rank} reports only "
        f"{price.mem_available_gib:.1f} GiB of unified memory available plus "
        f"{price.credit_gib:.1f} GiB of resident shards the proof drops first. "
        f"{middle}{counted}"
        "Set auto_gate=off on the Init node: for a checkpoint that still does "
        "not fit, it is the intended path, not a fallback, and it runs this "
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
