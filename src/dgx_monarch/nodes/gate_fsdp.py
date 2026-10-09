"""Driver validation for Identity Gate's FSDP clean-reload proof."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .. import mesh_safety as _mesh_safety
from ..adapters.fsdp import (
    FSDP_ADMITTED_CHECKPOINT_KINDS,
    fsdp_precision_profile_is_admitted,
)
from ..fsdp_proof_scope import (  # Public gate_fsdp compatibility API.
    FSDP_PROOF_SCOPE,
    topology_requires_fsdp_proof,
)
from ..transfer_utils import safe_note
from .gate_abort import (  # Public gate_fsdp compatibility API.
    aborted_proof_error,  # noqa: F401
    ceremony_waiver_boundary_error,  # noqa: F401
    close_aborted_proof,  # noqa: F401
    log_aborted_proof,  # noqa: F401
    record_proof_stop,  # noqa: F401
    retract_process_gate_verdicts,  # noqa: F401
    settled_refusal_tag,  # noqa: F401
)

FSDP_PROOF_KIND = "fsdp_clean_reload"


class FsdpGateProofError(RuntimeError):
    """Typed refusal to execute FSDP without an exact clean-reload PASS."""

    def __init__(self, message: str, *, verdict: str = "ERROR") -> None:
        self.verdict = str(verdict)
        super().__init__(message)


def token_requires_fsdp_proof(token: tuple[str, ...]) -> bool:
    """Whether a canonical Gate token names the FSDP clean-reload scope."""
    if len(token) < 4 or not isinstance(token[3], str):
        return False
    try:
        context = json.loads(token[3])
    except (TypeError, ValueError):
        return False
    return (
        isinstance(context, dict)
        and context.get("proof_scope") == FSDP_PROOF_SCOPE
    )


def token_resolves_fsdp(token: tuple[str, ...]) -> bool:
    """Whether a Gate token's context resolved a sharded topology, LoRA or not.

    FSDP has no stock residency to fall back to, so an aborted ceremony on
    such a context surfaces its own cause instead of forcing stock.
    """
    try:
        context = json.loads(token[3]) if len(token) >= 4 else None
    except (TypeError, ValueError):
        return False
    topology = context.get("resolved_topology") if isinstance(context, dict) else None
    return isinstance(topology, dict) and bool(topology.get("fsdp"))


def require_confirmed_fsdp_unload(
    responses: Any,
    handle: Any,
    required: bool,
    *,
    phase: str,
) -> None:
    """Require positive evidence that one unload reached every rank."""
    if not required:
        return
    world = getattr(handle, "world", None)
    if (
        type(world) is not int
        or world < 1
        or type(responses) is not list
        or len(responses) != world
        or any(
            type(row) is not dict or row.get("unloaded") is not True
            for row in responses
        )
    ):
        raise FsdpGateProofError(
            f"FSDP Gate {phase} all-rank unload returned incomplete evidence",
            verdict="INCONCLUSIVE",
        )


def model_declares_fsdp_proof(model: Any, handle: Any) -> bool:
    """Resolve the explicit request scope before a Gate session mutates state."""
    preset = getattr(getattr(model, "mesh", None), "topology_preset", None)
    if not isinstance(preset, str) or preset == "auto":
        return False
    try:
        from ..topology import topology_from_preset

        topology = topology_from_preset(preset, int(getattr(handle, "world", 0)))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
    return topology_requires_fsdp_proof(topology, getattr(model, "loras", ()))


def _latch_or_retire_failed_cleanup(
    handle: Any,
    primary: BaseException,
    cleanup_exc: BaseException,
    *,
    logger: Any,
    timeout_s: float,
) -> None:
    """Publish a bounded DIRTY latch or retire a handle that cannot latch."""
    safe_note(
        primary,
        "FSDP Gate all-rank unload was not confirmed; mesh must be recycled",
        cleanup_exc,
    )
    latched = False
    for attempt in range(2):
        try:
            handle._latch_ambiguous_mutation(
                "unload", timeout_s, cleanup_exc
            )
        except BaseException as latch_exc:
            safe_note(
                primary,
                f"FSDP Gate cleanup DIRTY latch attempt {attempt + 1} failed",
                latch_exc,
            )
        else:
            latched = True
            break
    retired = False
    if not latched:
        for attempt in range(2):
            try:
                handle.defunct = True
                retired = getattr(handle, "defunct", False) is True
            except BaseException as retire_exc:
                safe_note(
                    primary,
                    f"FSDP Gate defunct fallback attempt {attempt + 1} failed",
                    retire_exc,
                )
            if retired:
                break
        if not retired:
            safe_note(
                primary,
                "FSDP Gate cleanup state publication failed",
                "restart the driver before any possible handle reuse",
            )
    try:
        logger.error(
            "FSDP Gate cleanup was not confirmed (%r); the mesh is %s "
            "and must be recycled",
            cleanup_exc,
            "DIRTY" if latched else "defunct" if retired else "unsafe",
        )
    except BaseException as log_exc:
        safe_note(primary, "FSDP Gate cleanup logging failed", log_exc)


def _fleet_already_evicted(handle: Any) -> bool:
    """Whether this handle's lifecycle already belongs to someone else.

    An unreadable state reads as live, so the caller still unloads.
    """
    try:
        return _mesh_safety.coherent_lifecycle_verdict(handle, RuntimeError) != "live"
    except BaseException:
        return False


def cleanup_aborted_fsdp_proof(
    handle: Any,
    primary: BaseException,
    *,
    logger: Any,
    state: dict[str, bool],
    timeout_s: float = 600.0,
) -> bool:
    """Exactly once, unload every rank or make the handle non-reusable."""
    if state.get("cleanup_attempted", False):
        if "cleanup_confirmed" in state:
            return state["cleanup_confirmed"]
        interrupted = RuntimeError(
            "FSDP Gate cleanup completion was interrupted"
        )
        _latch_or_retire_failed_cleanup(
            handle, primary, interrupted, logger=logger, timeout_s=timeout_s)
        state["cleanup_confirmed"] = False
        return False
    state["cleanup_attempted"] = True
    if _fleet_already_evicted(handle):
        # The eviction owns this fleet: the proc stop frees its residency, and
        # an unload sent to it can only fail into a DIRTY latch that then
        # refuses everything else touching the handle.
        state["cleanup_confirmed"] = False
        safe_note(
            primary,
            "the fleet was already evicted; the proc stop frees its residency",
            "no FSDP Gate unload was dispatched",
        )
        try:
            logger.info(
                "FSDP Gate cleanup skipped: this fleet is already evicted, so "
                "the proc stop frees its residency")
        except BaseException as log_exc:
            safe_note(primary, "FSDP Gate cleanup logging failed", log_exc)
        return False
    try:
        responses = handle.call_all("unload", timeout_s=timeout_s)
        world = getattr(handle, "world", None)
        if (
            type(world) is not int
            or world < 1
            or type(responses) is not list
            or len(responses) != world
            or any(
                type(row) is not dict or row.get("unloaded") is not True
                for row in responses
            )
        ):
            raise RuntimeError(
                "FSDP Gate all-rank unload returned incomplete evidence"
            )
    except BaseException as cleanup_exc:
        _latch_or_retire_failed_cleanup(
            handle, primary, cleanup_exc, logger=logger, timeout_s=timeout_s)
        state["cleanup_confirmed"] = False
        return False
    state["cleanup_confirmed"] = True
    return True


def cleanup_rejected_fsdp_pass(
    handle: Any,
    primary: BaseException,
    *,
    logger: Any,
    state: dict[str, bool],
) -> bool:
    """Clean the exact ceremony handle after rejecting its provisional PASS."""
    if state.get("cleanup_attempted", False):
        return state.get("cleanup_confirmed", False)
    if handle is not None:
        return cleanup_aborted_fsdp_proof(
            handle, primary, logger=logger, state=state)
    state.update(cleanup_attempted=True, cleanup_confirmed=False)
    safe_note(
        primary,
        "automatic FSDP Gate rejected-PASS cleanup lost its handle",
        "restart the driver before retrying",
    )
    try:
        logger.error(
            "automatic FSDP rejected-PASS cleanup has no exact handle; "
            "restart the driver before retrying"
        )
    except BaseException as log_exc:
        safe_note(primary, "automatic FSDP cleanup logging failed", log_exc)
    return False


def arm_fsdp_return_guard(
    guard: dict[str, Any],
    result: dict,
    *,
    proof_required: bool,
) -> bool:
    """Arm the cleanup guard for an FSDP PASS the ceremony returned unarmed
    (``gate_session`` arms its own); return whether the result is FSDP."""
    result_is_fsdp = result.get("proof_kind") == FSDP_PROOF_KIND
    if (
        result.get("verdict") == "PASS"
        and (proof_required or result_is_fsdp)
        and not guard.get("cleanup_required", False)
    ):
        guard.update({
            "cleanup_required": True,
            "accepted": False,
            "handle": result.get("_gate_handle"),
            "state": {
                "cleanup_attempted": bool(
                    result.get("_fsdp_cleanup_attempted", False)),
                "cleanup_confirmed": bool(
                    result.get("_fsdp_cleanup_confirmed", False)),
            },
        })
    return result_is_fsdp


@dataclass
class FsdpReloadProof:
    cycle: list[dict]
    conclusive: bool
    reasons: list[str]


def establish(raw_cycle: list, handle) -> FsdpReloadProof:
    """Require complete, unique, exact worker evidence for one FSDP reload."""
    reasons: set[str] = set()
    raw_world = getattr(handle, "world", None)
    expected_world = (
        raw_world
        if type(raw_world) is int
        else 0
    )
    raw_is_exact_list = type(raw_cycle) is list
    raw_length = len(raw_cycle) if raw_is_exact_list else 0
    cycle = (
        [row for row in raw_cycle if type(row) is dict]
        if raw_is_exact_list
        else []
    )
    exact_response_shape = (
        raw_is_exact_list
        and raw_length == expected_world
        and len(cycle) == raw_length
    )
    if not exact_response_shape:
        reasons.add("FSDP reload cycle response is malformed")
    if expected_world < 1 or not exact_response_shape:
        reasons.add("FSDP reload cycle did not report every rank")

    ranks = [row.get("rank") for row in cycle]
    if (
        any(type(rank) is not int for rank in ranks)
        or len(set(ranks)) != len(ranks)
        or set(ranks) != set(range(expected_world))
    ):
        reasons.add("FSDP reload cycle rank evidence is incomplete or duplicated")

    expected_generation = getattr(handle, "setup_generation", None)
    generations = [row.get("setup_generation") for row in cycle]
    if (
        type(expected_generation) is not int
        or expected_generation < 1
        or any(
            type(value) is not int
            or value < 1
            or value != expected_generation
            for value in generations
        )
    ):
        reasons.add("FSDP reload cycle setup generation is invalid or drifted")

    families = [row.get("family") for row in cycle]
    if (
        not families
        or any(type(value) is not str or not value for value in families)
        or any(value != families[0] for value in families)
    ):
        reasons.add("FSDP reload cycle family evidence is missing or drifted")

    profiles = [row.get("live_dtype_profile") for row in cycle]
    if (
        not profiles
        or any(type(value) is not str or not value for value in profiles)
        or any(value != profiles[0] for value in profiles)
    ):
        reasons.add("FSDP reload cycle live precision profile is missing or drifted across ranks")
    precision_shapes = [
        (
            row.get("live_dtype_profile"),
            row.get("auxiliary_parameter_count"),
            row.get("auxiliary_parameter_bytes"),
        )
        for row in cycle
    ]
    if (
        not precision_shapes
        or any(
            type(value[0]) is not str
            or not value[0]
            or type(value[1]) is not int
            or type(value[2]) is not int
            for value in precision_shapes
        )
        or any(value != precision_shapes[0] for value in precision_shapes)
    ):
        reasons.add("FSDP reload cycle precision evidence is incomplete or drifted across ranks")

    for row in cycle:
        proof = row.get("proof")
        baseline_family = row.get("baseline_family")
        family = row.get("family")
        baseline_quant = row.get("baseline_quant")
        quant = row.get("quant")
        baseline_profile = row.get("baseline_live_dtype_profile")
        profile = row.get("live_dtype_profile")
        baseline_checkpoint = row.get("baseline_checkpoint_precision")
        checkpoint = row.get("checkpoint_precision")
        transitions = row.get("transitions")
        exact_transition = (
            type(transitions) is dict
            and len(transitions) == 1
            and all(type(key) is str for key in transitions)
            and tuple(transitions) == ("reload",)
            and type(transitions.get("reload")) is str
            and transitions.get("reload") == "load"
        )
        if (
            row.get("conclusive") is not True
            or type(proof) is not str
            or not proof
            or proof != FSDP_PROOF_KIND
            or row.get("baseline_verified") is not True
            or row.get("baseline_artifact_identity_verified") is not True
            or row.get("baseline_fsdp_ready") is not True
            or row.get("baseline_slab_active") is not False
            or type(baseline_family) is not str
            or not baseline_family
            or type(family) is not str
            or not family
            or baseline_family != family
            or type(baseline_quant) is not str
            or baseline_quant not in FSDP_ADMITTED_CHECKPOINT_KINDS
            or type(baseline_checkpoint) is not str
            or baseline_checkpoint != baseline_quant
            or not exact_transition
            or row.get("artifact_identity_verified") is not True
            or row.get("fsdp_ready") is not True
            or row.get("slab_active") is not False
            or type(quant) is not str
            or quant not in FSDP_ADMITTED_CHECKPOINT_KINDS
            or type(checkpoint) is not str
            or checkpoint != quant
        ):
            reasons.add(
                "FSDP reload cycle lacks a freshly loaded, ready shard set of an admitted checkpoint kind"
            )
        count = row.get("auxiliary_parameter_count")
        size = row.get("auxiliary_parameter_bytes")
        baseline_count = row.get("baseline_auxiliary_parameter_count")
        baseline_size = row.get("baseline_auxiliary_parameter_bytes")
        if (
            type(baseline_profile) is not str
            or not baseline_profile
            or type(profile) is not str
            or not profile
            or baseline_profile != profile
            or type(count) is not int
            or type(size) is not int
            or type(baseline_count) is not int
            or type(baseline_size) is not int
            or baseline_count != count
            or baseline_size != size
        ):
            reasons.add("FSDP baseline/reload precision evidence is incomplete or drifted")
        if not fsdp_precision_profile_is_admitted(
            profile,
            count,
            size,
            checkpoint,
        ):
            reasons.add(
                "FSDP reload cycle has incomplete or unaudited precision evidence"
            )

    return FsdpReloadProof(
        cycle=cycle,
        conclusive=not reasons,
        reasons=sorted(reasons),
    )
