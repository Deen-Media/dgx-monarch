"""Driver mesh safety checks without Monarch object dependencies.

Includes lifecycle classification, teardown fault attribution, artifact
comparison, rendezvous-port selection, per-host paths, and stock-load and
reference/pose activation preflights. It never imports ``mesh.py``.
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterable
from typing import Any

from . import absorb_fire as _absorb
from . import mesh_residency as _mesh_residency
from .config import ClusterConfig
from .error_utils import failure_summary

FLEET_RESIDENCY_CAPABILITY = _mesh_residency.FLEET_RESIDENCY_CAPABILITY
NORMAL_RENDER_RESIDENCY_CAPABILITY = _mesh_residency.NORMAL_RENDER_RESIDENCY_CAPABILITY
assert_fleet_residency_grant = _mesh_residency.assert_fleet_residency_grant
assert_normal_render_residency_grant = _mesh_residency.assert_normal_render_residency_grant
assert_normal_render_residency_mode = _mesh_residency.assert_normal_render_residency_mode
assert_request_artifact_binding = _mesh_residency.assert_request_artifact_binding
fleet_policy_is_risky = _mesh_residency.fleet_policy_is_risky
physical_capability_context = _mesh_residency.physical_capability_context
request_combo_key = _mesh_residency.request_combo_key

# A deliberate stop can race status polls. Reason-marked faults are attributable;
# transport-only faults require an in-flight stop or a short post-stop grace
# window with no live or creating mesh.
_TEARDOWN_REASON_MARKERS = ("dgx-monarch recycle", "dgx-monarch client detach",
                            "dgx-monarch failed setup rollback",
                            "dgx-monarch partial bring-up rollback")
# Keep the transport fault markers narrow: obsolete or unobserved variants stay
# out, so absorption never depends on stale text.
_TEARDOWN_FAULT_MARKERS = ("undeliverable", "delivery failure:",
                           "ttl expired for", "channel closed")
_TEARDOWN_GRACE_S = 30.0
# Token absorption authority is time-bounded. An interrupt after
# procs.stop().get() returns but before outcome publication can strand a token.
# The 240 s bound is the 180 s recycle deadline in
# `mesh_recycle.recycle_detailed`, which contains its 60 s group teardown and
# 60 s proc stop, plus scheduling margin. A plain `MeshHandle.shutdown` waits
# at most 2 x 60 + 30 s. An outcome-less token never authorizes reuse or
# another stop at any age.
TOKEN_AUTHORITY_S = 240.0
_TEARDOWN_STATE_LOCK = threading.Lock()
# One atomic snapshot: (token-to-issued-at mapping, grace-opened, version).
# Each transition publishes a new dict and triple through one name binding, so
# asynchronous readers cannot see a partial transition. The lock serializes
# writers; readers consume the immutable binding lock-free.
_TEARDOWN_STATE: tuple[dict[int, float], float, int] = ({}, float("-inf"), 0)


class PriorTeardownUnknownError(RuntimeError):
    """A prior stop may have completed, so another stop is unauthorized."""


def begin_deliberate_teardown(token: int) -> None:
    """Mark one handle's proc stop in flight.

    Call inside the try whose ``finally`` publishes the outcome and then calls
    ``end_deliberate_teardown``. The token scopes suppression to this teardown.
    """
    global _TEARDOWN_STATE
    with _TEARDOWN_STATE_LOCK:
        tokens, grace, version = _TEARDOWN_STATE
        _TEARDOWN_STATE = ({**tokens, token: time.monotonic()}, grace, version + 1)


def end_deliberate_teardown(token: int, stop_confirmed: bool) -> None:
    """Close the mark and open grace only after a confirmed stop.

    A failed stop leaves process state unknown, so later faults remain visible.
    Calling without a published begin mark is safe.
    """
    global _TEARDOWN_STATE
    with _TEARDOWN_STATE_LOCK:
        tokens, grace, version = _TEARDOWN_STATE
        _TEARDOWN_STATE = ({k: v for k, v in tokens.items() if k != token},
                           time.monotonic() if stop_confirmed else grace,
                           version + 1)


def clear_stale_token(token: int) -> None:
    """Clear a stranded token after the handle outcome is already published.

    Never open grace retroactively when the stop-success instant was lost.
    """
    end_deliberate_teardown(token, stop_confirmed=False)


def note_deliberate_teardown() -> None:
    """Refresh grace after a confirmed teardown absorbs a straggler."""
    global _TEARDOWN_STATE
    with _TEARDOWN_STATE_LOCK:
        tokens, _grace, version = _TEARDOWN_STATE
        _TEARDOWN_STATE = (tokens, time.monotonic(), version + 1)


def teardown_state_version() -> int:
    """Return the version used to bracket coherent correlation snapshots."""
    return _TEARDOWN_STATE[2]


def active_teardown_tokens() -> frozenset[int]:
    return frozenset(_TEARDOWN_STATE[0])


def token_within_authority(token: int) -> bool:
    """Return whether a present token still has fault-absorption authority."""
    issued = _TEARDOWN_STATE[0].get(token)
    return issued is not None and _absorb.verdict(
        "token-authority", time.monotonic() - issued <= TOKEN_AUTHORITY_S)


def token_present(token: int) -> bool:
    """Return whether a token exists at any age.

    Presence alone blocks reuse; age limits only fault absorption.
    """
    return token in _TEARDOWN_STATE[0]


def coherent_lifecycle_verdict(
    handle: Any,
    error_type: type[Exception],
) -> str:
    """Read the completed, blocked, defunct, dirty and token state coherently.

    Equal token-version reads plus identical handle-field snapshots bracket a
    consistent view; either kind of transition forces a retry. This stays
    lock-free with respect to ``handle.lock``: recycle holds it while awaiting
    the registry lock, so taking it here would reverse lock order and deadlock.
    """
    for _ in range(8):
        version = teardown_state_version()
        before = (
            bool(getattr(handle, "teardown_complete", False)),
            getattr(handle, "replacement_blocked", None),
            bool(handle.defunct),
            getattr(handle, "setup_cleanup_state", None) is not None,
        )
        token = token_present(id(handle))
        after = (
            bool(getattr(handle, "teardown_complete", False)),
            getattr(handle, "replacement_blocked", None),
            bool(handle.defunct),
            getattr(handle, "setup_cleanup_state", None) is not None,
        )
        if teardown_state_version() != version or before != after:
            continue
        complete, blocked, defunct, dirty = after
        if complete:
            verdict = "completed"
        elif blocked:
            verdict = "blocked"
        elif defunct or token or dirty:
            verdict = "unresolved"
        else:
            verdict = "live"
        return verdict
    raise error_type(
        "mesh lifecycle state kept changing while deciding reuse; retry")


def teardown_outcome_published(handle: Any) -> bool:
    """Return whether the stop published a confirmed or blocked outcome."""
    return bool(getattr(handle, "teardown_complete", False)
                or getattr(handle, "replacement_blocked", None))


def fault_correlation_state(
    handles: Iterable[Any],
    creation_in_progress: bool,
) -> tuple[bool, bool, bool]:
    """Return liveness, creation, and authoritative in-flight-stop state."""
    live = in_flight = False
    for handle in handles:
        for _ in range(8):
            before = (
                bool(getattr(handle, "teardown_complete", False)),
                getattr(handle, "replacement_blocked", None),
            )
            authoritative = token_within_authority(id(handle))
            after = (
                bool(getattr(handle, "teardown_complete", False)),
                getattr(handle, "replacement_blocked", None),
            )
            if before == after:
                break
        else:
            # Churn cannot authorize fault suppression. Treat the handle as
            # live so the caller keeps an unattributed transport fault visible.
            live = True
            continue
        complete, blocked = after
        if authoritative and not complete and not blocked:
            in_flight = True
        elif not complete:
            live = True
    return (live, creation_in_progress, in_flight)


def deliberate_teardown_in_progress() -> bool:
    return bool(_TEARDOWN_STATE[0])


def is_reason_marked_teardown(text: str) -> bool:
    """Return whether explicit stop-reason text proves fault attribution."""
    return _absorb.hits("reason-marked", text, _TEARDOWN_REASON_MARKERS)


def is_deliberate_teardown_fault(
    text: str,
    *,
    teardown_in_progress: bool = False,
    creation_in_progress: bool = False,
    live_mesh_exists: bool = False,
) -> bool:
    """Return whether a transport fault belongs to deliberate teardown.

    Explicit stop reasons absorb first. Non-transport faults never absorb. Any
    other live mesh or creation keeps ambiguous transport faults visible. An
    authoritative in-flight stop absorbs its own racing faults; otherwise only
    the confirmed-stop grace window covers late stragglers. ``live_mesh_exists``
    must exclude identities in ``active_teardown_tokens()``.
    """
    if is_reason_marked_teardown(text):
        return True
    if not _absorb.hits("fault-marker", text, _TEARDOWN_FAULT_MARKERS):
        return False
    if creation_in_progress or live_mesh_exists:
        return False
    if teardown_in_progress:
        _absorb.fire("in-flight")
        return True
    return _absorb.fire_if("grace-window", time.monotonic() - _TEARDOWN_STATE[1] < _TEARDOWN_GRACE_S)


def is_memory_exhaustion(exc: BaseException) -> bool:
    """Recognize local and wrapped allocator failures for diagnostics only.

    The markers cover typed OOM, CUDA runtime text, and C++ ``bad_alloc``.
    Location within a Gate leg cannot be inferred from this result.
    """
    cause = exc.__cause__
    text = f"{type(exc).__name__} {exc} {'' if cause is None else cause}"
    return any(marker in text for marker in ("OutOfMemoryError", "out of memory",
        "cudaErrorMemoryAllocation", "bad_alloc"))


class StockLoadCapacityError(RuntimeError):
    """Typed integrated-memory refusal from a load or activation boundary."""


def is_stock_load_capacity_error(exc: BaseException) -> bool:
    """Recognize local and Monarch-wrapped stock-load capacity refusals."""
    # Use the unique class name as the remote machine-readable marker.
    return (isinstance(exc, StockLoadCapacityError)
            or "StockLoadCapacityError" in failure_summary(exc))


def mem_available_bytes() -> int | None:
    """Kernel MemAvailable (includes reclaimable cache); None off-Linux."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def gpu_is_integrated() -> bool:
    """CUDA integrated/UMA device probe; False on discrete or unknown."""
    try:
        import torch

        return bool(getattr(torch.cuda.get_device_properties(0),
                            "is_integrated", False))
    except Exception:
        return False


def stock_load_preflight(path: str, unet_name: str, model_options: dict) -> None:
    """Refuse a stock load that cannot fit on an integrated/UMA device.

    Stock weights consume roughly file size in the shared UMA pool; exceeding
    MemAvailable can invoke the OS OOM killer before Python can classify an
    exception. Discrete devices and dtype casts are excluded because file size
    does not bound their resident allocation.
    """
    import os

    if model_options.get("dtype") is not None or not gpu_is_integrated():
        return
    size = os.path.getsize(path) if os.path.exists(path) else 0
    avail = mem_available_bytes()
    if avail is not None and size > avail:
        raise StockLoadCapacityError(
            f"stock residency cannot load {unet_name}: weights are "
            f"{size / 2**30:.1f} GiB but only {avail / 2**30:.1f} GiB of "
            "unified memory is available; the load would be killed out "
            "of memory. slab_weights=on (zero-copy residency) may still fit.")


# Families that append reference or pose rows before DiT blocks need a
# token-aware activation estimate. Keys are the family names header-only
# detection returns; values are hidden dimensions. New entries require
# hardware calibration of _ACTIVATION_BLOCK_FACTOR.
REF_POSE_TOKEN_FAMILIES: dict[str, int] = {
    # SCAIL variants share dim=5120 and token concatenation order.
    "wan_scail": 5120,
    # Animate2 has a lockstep driving-pose branch and the same hidden width.
    "wan_animate2": 5120,
    # MiniMax H3 is excluded because its packed audio, text, and reference row
    # algebra is estimated in h3_activation.py (docs/TROUBLESHOOTING.md #56).
}

# Coarse bf16 buffers live for one inference block: Q, K, V, attention output,
# MLP intermediate, and residual. The factor is a generous placeholder, not
# calibrated on hardware (docs/TROUBLESHOOTING.md #47). Higher values refuse more.
_ACTIVATION_BLOCK_FACTOR = 8
_ACTIVATION_BF16_BYTES = 2

# Bypasses a false positive from this uncalibrated estimator
# (docs/TROUBLESHOOTING.md #47). h3_activation, ltx25_activation and
# render_memory_price read the same variable, so it also turns off their
# preflights and drops the H3 and LTX activation charges from the loader-site
# footprint (nodes/loader_graph.driver_stack_terms); every other capacity and
# Gate boundary stays active.
ACTIVATION_PREFLIGHT_DISABLE_ENV = "DGXM_DISABLE_ACTIVATION_PREFLIGHT"

_TRUTHY_ENV = frozenset({"1", "true", "yes", "on"})


def env_enabled(name: str) -> bool:
    """Whether an env override is set to an explicit on value.

    Only ``1``, ``true``, ``yes`` and ``on`` count, in any case: a refusal that
    tells an operator to set ``=1`` must not also honor ``=0``.
    """
    return os.environ.get(name, "").strip().lower() in _TRUTHY_ENV


def _token_count_5d(shape: Iterable[int]) -> int:
    """Count ``(B,C,T,H,W)`` tokens under patch size ``(1,2,2)``.

    Temporal length is unchanged; spatial axes use ceiling division. Accepting
    any five-int iterable keeps the helper torch-free.
    """
    b, _c, t, h, w = (int(v) for v in shape)
    return max(b, 1) * t * -(-h // 2) * -(-w // 2)


def activation_footprint_preflight(
    family: str,
    path: str,
    unet_name: str,
    *,
    video_shape: Iterable[int],
    ref_shapes: Iterable[Iterable[int]] = (),
    pose_shape: Iterable[int] | None = None,
    sp_degree: int = 1,
    weights_resident: bool = False,
) -> None:
    """Refuse an oversized ref/pose activation footprint before worker RPC.

    Inputs are existing ``(B,C,T,H,W)`` latent shapes, not pixel dimensions.
    ``sp_degree`` is Ulysses times Ring: only activation rows are divided;
    weights remain fully resident on every rank. The check applies only to
    registered families on positive UMA detection and can be disabled with
    ACTIVATION_PREFLIGHT_DISABLE_ENV (docs/TROUBLESHOOTING.md #47).
    """
    import os

    hidden_dim = REF_POSE_TOKEN_FAMILIES.get(family)
    if (hidden_dim is None or not gpu_is_integrated()
            or env_enabled(ACTIVATION_PREFLIGHT_DISABLE_ENV)):
        return
    # Resident weights already reduce MemAvailable; counting them again would
    # double-charge every warm render.
    weight_bytes = 0 if weights_resident else (
        os.path.getsize(path) if os.path.exists(path) else 0)
    video_tokens = _token_count_5d(video_shape) + sum(
        _token_count_5d(s) for s in ref_shapes)
    pose_tokens = _token_count_5d(pose_shape) if pose_shape is not None else 0
    total_tokens = video_tokens + pose_tokens
    tokens_per_rank = -(-total_tokens // max(int(sp_degree), 1))
    bytes_per_token = hidden_dim * _ACTIVATION_BF16_BYTES * _ACTIVATION_BLOCK_FACTOR
    activation_bytes = tokens_per_rank * bytes_per_token
    estimate = weight_bytes + activation_bytes
    avail = mem_available_bytes()
    if avail is None or estimate <= avail:
        return
    max_tokens_per_rank = max((avail - weight_bytes) // max(bytes_per_token, 1), 0)
    weights_label = "resident, credited" if weights_resident else f"{weight_bytes / 2**30:.1f} GiB"
    raise StockLoadCapacityError(
        f"activation footprint preflight refuses {unet_name}: estimated "
        f"{estimate / 2**30:.1f} GiB (weights {weights_label} + "
        f"~{activation_bytes / 2**30:.1f} GiB activations for {total_tokens} "
        f"tokens [{video_tokens} video+ref, {pose_tokens} pose] over "
        f"{sp_degree} rank(s)) exceeds {avail / 2**30:.1f} GiB available. "
        "Reduce the resolution, frame count or reference-image count until "
        f"tokens/rank <= ~{max_tokens_per_rank}, free memory, or set "
        f"{ACTIVATION_PREFLIGHT_DISABLE_ENV}=1 to bypass this preflight "
        "(docs/TROUBLESHOOTING.md #47).")


class ArtifactBindingError(RuntimeError):
    """A multi-phase operation no longer sees its authorized artifact set."""


def is_artifact_binding_error(exc: BaseException) -> bool:
    """Recognize local and Monarch-wrapped artifact binding failures."""
    # A Monarch-wrapped remote failure arrives as ActorError text, so the
    # class name is the marker. Never match free text: a third-party message
    # that mentions "artifact snapshot" must not reroute the ceremony's errors.
    return (
        isinstance(exc, ArtifactBindingError)
        or "ArtifactBindingError" in failure_summary(exc))


def bounded_master_port(base_port: int, pid: int, generation: int,
                        fleet_world: int = 0) -> int:
    """Salt a rendezvous port while keeping every derived fleet port legal."""
    # Fleet ranks use base + 32 + rank. Choose the salted base from the range
    # that leaves those offsets below 65536; modulo handles an operator base at
    # the top of the legal config range without producing a privileged port or
    # overflow.
    largest_offset = 32 + fleet_world - 1 if fleet_world else 0
    upper = 65535 - largest_offset
    lower = 1024
    if upper < lower:
        raise ValueError(f"fleet world {fleet_world} leaves no legal TCP rendezvous port")
    salt = (int(pid) % 16) * 16 + (int(generation) % 16)
    span = upper - lower + 1
    return lower + ((int(base_port) - lower + salt) % span)


def assert_artifact_parity(results: list[dict], expected: list[dict] | None = None) -> None:
    """Refuse a distributed operation when workers see different bytes/code."""
    if not results:
        raise RuntimeError("distributed artifact parity check returned no worker results")
    reference = expected if expected is not None else results[0].get("artifact_sets")
    if reference is None:
        raise RuntimeError(
            "distributed artifact parity check received no reference identity"
        )
    mismatches = [
        result for result in results
        if result.get("artifact_sets") != reference
    ]
    if not mismatches:
        return
    identities = ([{"host": "driver", "rank": None, "artifact_sets": expected}]
                  if expected is not None else []) + [
        {
            "host": result.get("host"),
            "rank": result.get("rank"),
            "artifact_sets": result.get("artifact_sets"),
        }
        for result in results
    ]
    raise RuntimeError(
        "distributed artifact parity check failed: model/LoRA bytes or the "
        "ComfyUI commit differ between hosts; stage the same files and ComfyUI "
        f"commit on every host (identities: {identities!r})"
    )


def worker_comfy_dir(config: ClusterConfig, host_idx: int, fallback: str) -> str:
    """Resolve a worker's Comfy root with host override precedence."""
    if config.hosts and config.hosts[host_idx].comfy_dir:
        return config.hosts[host_idx].comfy_dir
    return config.comfy_dir or fallback
