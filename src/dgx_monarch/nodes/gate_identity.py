"""Identity-gate transactions, capability contexts, render authorization and stock quarantine."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

from ..fsdp_proof_scope import topology_resolves_fsdp
from ..gate_ledger import ArtifactSetSignature, gate_verdict_token
from ..log import get_logger
from ..mesh_safety import FLEET_RESIDENCY_CAPABILITY, physical_capability_context
from ..transfer_utils import safe_call

# Both re-exported for nodes/gate_verdict, which reads them at call time as
# runtime["_gate_identity"].<name> through the nodes/gate namespace.
from .consent_rescue import revoke_consents_after_gate_fail as revoke_consents_after_gate_fail
from .gate_fsdp import (
    FSDP_PROOF_SCOPE,
    FsdpGateProofError,
    topology_requires_fsdp_proof,
)
from .gate_inconclusive import record_inconclusive as record_inconclusive
from .gate_tensor_guard import transaction_tensor_fingerprint

log = get_logger(__name__)


@dataclass(frozen=True)
class NormalRenderAuthorization:
    """One exact model/policy snapshot and its optional Gate capability."""

    worker_args: dict
    model_request: dict
    residency_grant: dict | None
    residency_mode: str


def _transaction_tensor_memo(value: Any, *, clone: bool) -> dict[int, Any]:
    """Map every unique tensor leaf to itself or one owned baseline clone."""
    try:
        import torch
    except ImportError:  # pragma: no cover - gate execution always has torch
        return {}
    from .latent_identity import direct_nested_tensor_parts

    memo: dict[int, Any] = {}
    seen: set[int] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(item, torch.Tensor):
            if type(item) is not torch.Tensor:
                raise TypeError(
                    "identity-gate transaction Tensor subclasses are not supported"
                )
            if clone:
                # Comfy commonly invokes nodes under inference_mode.  Disable
                # it for the owned baseline so PyTorch provides the mutation
                # version counter that guards every ceremony leg.
                with torch.inference_mode(False):
                    memo[identity] = item.clone()
            else:
                memo[identity] = item
        elif (parts := direct_nested_tensor_parts(item)) is not None:
            pending.extend(parts)
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            pending.extend(item)
    return memo


def freeze_transaction(value: Any) -> Any:
    """Own one immutable tensor baseline and deeply isolate its containers."""
    return copy.deepcopy(value, _transaction_tensor_memo(value, clone=True))


def copy_transaction(value: Any) -> Any:
    """Copy per-leg containers while sharing the guarded frozen tensor leaves."""
    return copy.deepcopy(value, _transaction_tensor_memo(value, clone=False))


def transaction_tensor_versions(
    value: Any,
) -> tuple[tuple[Any, int, bytes], ...]:
    """Capture version counters plus storage fingerprints for the frozen baseline."""
    return tuple(
        (
            tensor,
            int(getattr(tensor, "_version", 0)),
            transaction_tensor_fingerprint(tensor),
        )
        for tensor in _transaction_tensor_memo(value, clone=False).values()
    )


def require_transaction_unchanged(
    versions: tuple[tuple[Any, int, bytes], ...],
) -> None:
    for tensor, version, fingerprint in versions:
        if (
            int(getattr(tensor, "_version", 0)) != version
            or transaction_tensor_fingerprint(tensor) != fingerprint
        ):
            raise RuntimeError(
                "identity-gate render mutated a captured request/latent tensor; "
                "discarding the verdict, because its legs may not have run on the same inputs"
            )


def resolve_effective_worker_args(
    model: Any,
    handle: Any = None,
    *,
    requested_worker_args: dict | None = None,
) -> dict:
    """Merge cluster policy with the graph's explicit worker overrides."""
    handle = handle or model.mesh.handle
    requested = (dict(model.mesh.worker_args) if requested_worker_args is None
                 else dict(requested_worker_args))
    if hasattr(handle, "effective_worker_args"):
        return handle.effective_worker_args(requested)
    config_args = getattr(getattr(handle, "config", None), "worker_args", {}) or {}
    return {**dict(config_args), **requested}


def gate_capability_context(
    model: Any,
    handle: Any = None,
    *,
    effective_worker_args: dict | None = None,
    resolved_topology: Any = None,
    resolved_attention: str | None = None,
) -> dict:
    """Canonical capability identity to which a persisted PASS is bound."""
    handle = handle or model.mesh.handle
    effective = (
        resolve_effective_worker_args(model, handle)
        if effective_worker_args is None
        else dict(effective_worker_args)
    )
    physical = physical_capability_context(handle)
    topology_context = None
    if resolved_topology is not None:
        topology_context = {
            name: (resolved_topology.get(name)
                   if isinstance(resolved_topology, dict)
                   else getattr(resolved_topology, name))
            for name in ("ulysses", "ring", "cfg", "dp", "fsdp")
        }
    context = {
        "worker_args": effective,
        "mesh_mode": physical["mesh_mode"],
        "config_source": physical["config_source"],
        "config_fingerprint": physical["config_fingerprint"],
        "world": physical["physical_world"],
        "hosts": physical["hosts"],
        "gpus_per_host": physical["gpus_per_host"],
        "topology_preset": str(getattr(model.mesh, "topology_preset", "unknown")),
        "attention": str(getattr(model.mesh, "attention", "unknown")),
        "sync_ulysses": bool(getattr(model.mesh, "sync_ulysses", True)),
        "resolved_topology": topology_context,
        "resolved_attention": resolved_attention,
    }
    if (
        topology_context is not None
        and topology_requires_fsdp_proof(
            topology_context,
            getattr(model, "loras", ()),
        )
    ):
        # The FSDP reload proof is a distinct authority scope. It cannot match
        # any earlier residency-only capability row.
        context["proof_scope"] = FSDP_PROOF_SCOPE
    return context


def fleet_residency_capability_context(
    model: Any,
    handle: Any = None,
    *,
    effective_worker_args: dict | None = None,
) -> dict:
    """Capability identity for Fleet's independent world-1 residency path."""
    handle = handle or model.mesh.handle
    effective = (
        resolve_effective_worker_args(model, handle)
        if effective_worker_args is None
        else dict(effective_worker_args)
    )
    return {
        "capability": FLEET_RESIDENCY_CAPABILITY,
        "rank_world": 1,
        "worker_args": effective,
        **physical_capability_context(handle),
    }


def quarantine_unproven_paths(model, cause: BaseException | None = None) -> None:
    """Disable residency optimizations that lack a completed identity proof.

    ``unproven_quarantine_levers`` preserves two supported exceptions: a LoRA-free
    capacity rescue keeps slab and its required low-RSS mode, and an explicit
    FSDP LoRA render keeps its required low-RSS mode. Capacity rescue covers an
    unavailable stock reference, not a measured divergence. Measured FAIL handling
    separately disables the implicated settings and revokes contradicted consent.

    ``gate_quarantine_scope`` interprets ``cause``: a class-K accuracy refusal
    changes no residency setting; other aborts affect only their own combination.
    """
    from .consent_rescue import unproven_quarantine_levers
    from .gate_quarantine_scope import write_scoped_quarantine

    write_scoped_quarantine(model, unproven_quarantine_levers(model), cause)


def force_stock_quarantine(model, handle, timeout_s: float = 600.0,
                           cause: BaseException | None = None) -> list:
    """Publish both graph levers off and push that complete policy fail-closed.

    A measured class K abort keeps every lever the graph asked for, so there
    is no new policy to push and no cached key to invalidate.
    """
    from .gate_quarantine_scope import measured_abort_guard

    quarantine_unproven_paths(model, cause)
    if measured_abort_guard(cause) is not None:
        return []
    from ..mesh import MeshHandle

    if isinstance(handle, MeshHandle) and getattr(handle, "setup_key", None) is None:
        # No READY workers exist to mutate. Clear the driver's fast-path marker
        # and let the next setup carry the graph's stock policy; DIRTY here
        # would force a recycle with no remote side effect to undo.
        with handle.lock:
            handle.worker_args_key = None
            handle.active_worker_args = {}
        return []
    try:
        return apply_worker_policy(
            handle, resolve_effective_worker_args(model, handle),
            timeout_s=timeout_s)
    except BaseException as exc:
        # A failed or interrupted quarantine push must never release the
        # ceremony owner with risky worker residency still authorized.
        # retire_failed_quarantine decides which failures latch DIRTY.
        from .gate_quarantine_latch import retire_failed_quarantine

        retire_failed_quarantine(handle, exc, timeout_s)
        raise


def model_with_worker_overrides(
    model: Any, overrides: dict, *, handle: Any = None
) -> Any:
    """Clone a graph model with a per-dispatch policy and optional pinned handle."""
    worker_args = {**dict(model.mesh.worker_args), **dict(overrides)}
    mesh_updates = {"worker_args": worker_args}
    if handle is not None:
        mesh_updates["handle"] = handle
    if hasattr(model, "__dataclass_fields__") and hasattr(model.mesh, "__dataclass_fields__"):
        return replace(model, mesh=replace(model.mesh, **mesh_updates))
    cloned, mesh = copy.copy(model), copy.copy(model.mesh)
    object.__setattr__(mesh, "worker_args", worker_args)
    if handle is not None:
        object.__setattr__(mesh, "handle", handle)
    object.__setattr__(cloned, "mesh", mesh)
    return cloned


def model_for_request(model, request: dict):
    """Use a stock-policy clone for dual-model requests, without graph mutation."""
    if model is None or request.get("uncond_model") is None:
        return model
    worker_args = model.mesh.worker_args
    if (worker_args.get("lora_low_rss") is False
            and worker_args.get("slab_weights") is False):
        return model
    log.warning("dual-model request: forcing stock residency for both model slots")
    return model_with_worker_overrides(
        model, {"lora_low_rss": False, "slab_weights": False})


def capture(model, uncond_model_request: dict | None = None) -> tuple[
    dict, str, dict, ArtifactSetSignature, dict
]:
    """Return model request, combo, detailed identity, aggregate, and binding."""
    from ..actor.model_store import request_artifact_identity
    from ..gate_ledger import artifact_set_signature
    from ..mesh_safety import request_combo_key

    raw_request = model.request_dict() if hasattr(model, "request_dict") else {
        "unet_name": model.unet_name,
        "options": dict(model.options),
        "loras": [dict(entry) for entry in model.loras],
    }
    model_request = copy.deepcopy(raw_request)
    identity = request_artifact_identity(
        model_request["unet_name"], model_request.get("loras"))
    uncond_request = copy.deepcopy(uncond_model_request)
    artifact_sets = [identity]
    if uncond_request is not None:
        artifact_sets.append(request_artifact_identity(
            uncond_request["unet_name"], uncond_request.get("loras")))
    artifacts = artifact_set_signature(
        item["signature"] for item in identity["artifacts"])
    if artifacts.current != identity["digest"]:
        raise RuntimeError("identity-gate artifact snapshot is inconsistent")
    key = request_combo_key(model_request)
    binding = {
        "combo_key": key,
        "model_request": copy.deepcopy(model_request),
        "uncond_model_request": copy.deepcopy(uncond_request),
        "artifact_sets": copy.deepcopy(artifact_sets),
    }
    return model_request, key, identity, artifacts, binding


def bind_request(request: dict, binding: dict) -> dict:
    return {**dict(request), "_dgxm_artifact_binding": copy.deepcopy(binding)}


def authorize_normal_render(
    model: Any,
    handle: Any,
    *,
    uncond_model_request: dict | None = None,
    gate_active: bool,
    session_verdict,
    resolved_topology: Any,
    resolved_attention: str,
) -> NormalRenderAuthorization:
    """Authorize risky first-use residency for one immutable dispatch snapshot.

    A ledger/session PASS is only a source of authority.  The returned grant
    carries the exact detailed artifact and capability snapshot so driver and
    worker preflight can reject any later byte, policy, or model drift.
    """
    requested = dict(model.mesh.worker_args)
    model_request = copy.deepcopy(model.request_dict())
    effective = resolve_effective_worker_args(
        model, handle, requested_worker_args=requested)
    from ..mesh_safety import fleet_policy_is_risky

    fsdp_proof_required = topology_requires_fsdp_proof(resolved_topology, model_request.get("loras"))
    fsdp_lora = topology_resolves_fsdp(resolved_topology) and not fsdp_proof_required

    # A Gate ceremony proves exactly one model, so a dual-model render dispatches with
    # both residency levers off, or refuses under automatic FSDP. Never let a primary-only
    # PASS grant its second slot, even when the caller skipped model_for_request.
    if uncond_model_request is not None:
        if (
            (fsdp_proof_required or fsdp_lora)
            and not gate_active
            and getattr(model.mesh, "auto_gate", "off") == "first_use"
        ):
            raise FsdpGateProofError(
                "automatic FSDP execution is unavailable for a dual-model "
                "request because the clean-reload Gate proves one model only"
            )
        return NormalRenderAuthorization(
            {**requested, "slab_weights": False, "lora_low_rss": False},
            model_request,
            None,
            "stock",
        )

    if ((not fleet_policy_is_risky(model_request, effective)
            and not fsdp_proof_required)
            or gate_active
            or getattr(model.mesh, "auto_gate", "off") != "first_use"):
        mode = "gate_internal" if gate_active else (
            "operator_off"
            if getattr(model.mesh, "auto_gate", "off") != "first_use"
            else "stock")
        return NormalRenderAuthorization(requested, model_request, None, mode)

    try:
        from .. import __version__
        from ..gate_ledger import GATE_PROTOCOL_VERSION, GateLedger
        from .gate import _ledger_dir

        captured_request, key, identity, artifacts, _binding = capture(model)
        # ``model_request`` and capture must describe one graph snapshot. A
        # custom graph payload that changes between the two reads gets no risky
        # grant: the render falls back to stock, or refuses under automatic FSDP.
        if captured_request != model_request:
            raise RuntimeError("normal-render model request changed during authorization")
        context = gate_capability_context(
            model,
            handle,
            effective_worker_args=effective,
            resolved_topology=resolved_topology,
            resolved_attention=resolved_attention,
        )
        if context["resolved_topology"] is None:
            raise RuntimeError("normal-render authorization has no concrete topology")
        token = gate_verdict_token(
            key, artifacts.current, identity["comfy"], context)
        lookup = GateLedger(_ledger_dir()).lookup_with_integrity(
            key, artifacts, identity["comfy"], context)
        state, entry = lookup.state, lookup.entry
        process_verdict = session_verdict(token)
        process_denied = process_verdict in (
            "FAIL", "INCONCLUSIVE", "ERROR")
        if (not process_denied and (state == "pass"
                or (lookup.session_pass_safe
                    and (entry is None or entry.get("verdict") == "PASS")
                    and state in ("unknown", "stale")
                    and process_verdict == "PASS"))):
            return NormalRenderAuthorization(
                requested,
                model_request,
                {
                    "capability": "normal_render_residency",
                    "gate_protocol": GATE_PROTOCOL_VERSION,
                    "dgx_monarch": __version__,
                    "comfy": identity["comfy"],
                    "artifact_digest": artifacts.current,
                    "gate_token": list(token),
                    "combo_key": key,
                    "model_request": copy.deepcopy(model_request),
                    "uncond_model_request": None,
                    "artifact_sets": [copy.deepcopy(identity)],
                    "capability_context": copy.deepcopy(context),
                },
                "required",
            )
        if fsdp_proof_required or fsdp_lora:
            log.error(
                "normal render cannot use an exact FSDP Gate PASS (ledger state "
                "%s); denying automatic FSDP execution", state)
            refusal_verdict = (
                str(process_verdict)
                if process_denied
                else "FAIL"
                if state == "fail"
                else "INCONCLUSIVE"
            )
            raise FsdpGateProofError(
                "automatic FSDP execution requires an exact "
                f"{'first-use' if fsdp_lora else 'clean-reload'} Gate PASS (ledger={state}, "
                f"process={process_verdict})" + (
                    "; LoRA on FSDP shards has no stock residency to fall back to, so the "
                    "render is refused rather than dispatched with lora_low_rss off" if fsdp_lora else ""),
                verdict=refusal_verdict,
            )
        from .consent_projection import capacity_consent_authorization
        consented = capacity_consent_authorization(
            model, requested, model_request, identity, key, context, lookup, token, process_verdict)
        if consented is not None:
            return consented
    except FsdpGateProofError:
        raise
    except Exception as exc:
        if fsdp_proof_required or fsdp_lora:
            raise FsdpGateProofError(
                "automatic FSDP Gate authorization could not prove the exact "
                "clean-reload capability",
            ) from exc
        safe_call(
            log.error,
            "normal-render Gate authorization failed (%r); forcing stock residency",
            exc,
        )
    return NormalRenderAuthorization(
        {**requested, "slab_weights": False, "lora_low_rss": False},
        model_request,
        None,
        "stock",
    )


def apply_worker_policy(handle, worker_args: dict, timeout_s: float = 600.0) -> list:
    """Apply a complete effective policy and synchronize the driver's key."""
    if hasattr(handle, "apply_worker_args"):
        return handle.apply_worker_args(worker_args, timeout_s=timeout_s, merge_config=False)
    return handle.call_all("apply_worker_args", worker_args, timeout_s=timeout_s)


@contextmanager
def temporary_worker_policy(handle, original: dict, overrides: dict):
    """Failure-atomic temporary policy, including partial broadcast failures."""
    if hasattr(handle, "temporary_worker_args"):
        with handle.temporary_worker_args(original, overrides, timeout_s=600.0) as applied:
            yield applied
        return
    target = {**original, **overrides}
    body_failed = False
    try:
        applied = apply_worker_policy(handle, target)
        yield applied
    except BaseException:
        body_failed = True
        raise
    finally:
        # This branch repeats MeshHandle.temporary_worker_args for handles that lack it. A
        # broadcast can fail after changing only some ranks, so always restore, even when
        # the target apply raised. Never let a restore failure mask the body's exception:
        # the operator needs the render or CUDA error. This assumes the handle's
        # apply_worker_args drops the driver's key before it sends, as MeshHandle's does,
        # so a failed restore makes the next render re-apply the full policy.
        try:
            apply_worker_policy(handle, original)
        except BaseException as restore_exc:
            safe_call(
                log.error, "temporary worker policy restore failed: %r", restore_exc)
            if not body_failed:
                raise


def equivalent_slab_mode_contexts(model, handle, slab_proof, verdict, cross_mode,
                                  effective_worker_args: dict,
                                  resolved_topology: Any = None,
                                  resolved_attention: str | None = None) -> list[dict]:
    """The explicit-on sibling context an auto-mode slab PASS also covers.

    One direction only, auto to on. A ceremony whose effective policy was not
    explicit True, and whose workers all reported slab residency in a vouched
    family, has already run what explicit True resolves to on these workers:
    ``worker_env.slab_mode_effective`` (FSDP) and ``capacity_quote.price``
    (compile) gate True as they gate auto, and this context binds both. The
    reverse is never stamped: an omitted key resolves per hardware (integrated
    worker defaults inject "auto"; discrete leaves it absent, which is stock),
    so an explicit-on ceremony proves nothing about the sibling's residency.
    Stamp only on worker-reported state plus a passed cross-residency leg,
    never on a driver-side prediction.

    Revocation is bounded the same way. A FAIL reaches the sibling only when
    the cross-residency leg itself diverged, because that leg is the only part
    of the ceremony slab residency ran. A swap-lineage FAIL with no failing
    cross leg says nothing about a sibling it never rendered.
    """
    effective = dict(effective_worker_args)
    if effective.get("slab_weights") is True:
        return []  # on -> auto is hardware-dependent; never stamped
    if verdict == "PASS":
        from ..actor.model_store import SLAB_VOUCHED_FAMILIES

        if (not slab_proof.active
                or not (cross_mode and cross_mode.get("verdict") == "PASS")
                or slab_proof.family not in SLAB_VOUCHED_FAMILIES):
            return []
    elif verdict == "FAIL" and not (cross_mode and cross_mode.get("verdict") == "FAIL"):
        # A lazy-swap-only FAIL never exercised slab residency. A ledger FAIL
        # is sticky for these artifact bytes and terminal for the automatic
        # path, so stamping it here would convert "untested" into "proven
        # broken" and force both residency levers off for good. The preflight
        # RETESTING rows already deny both siblings and read back as a retest,
        # which keeps the untested capability denied and recoverable.
        return []
    # What reaches here revokes the deterministic explicit-on sibling: a
    # diverged cross-residency leg, or an INCONCLUSIVE/ERROR that reads back
    # as a retest, even when this ceremony could not rediscover the earlier
    # PASS's family/residency facts. Negative evidence never grants
    # equivalence. The retest is the whole revocation: the session cache
    # carries no denial for the sibling of a no-material ceremony, since that
    # ceremony never ran slab residency (gate_inconclusive.session_denial_tokens).
    from .gate import fleet_residency_capability_context, gate_capability_context

    variant = {**effective, "slab_weights": True}
    return [
        gate_capability_context(
            model,
            handle,
            effective_worker_args=variant,
            resolved_topology=resolved_topology,
            resolved_attention=resolved_attention,
        ),
        fleet_residency_capability_context(model, handle, effective_worker_args=variant),
    ]
