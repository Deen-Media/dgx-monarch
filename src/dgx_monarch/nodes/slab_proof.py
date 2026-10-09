"""Establish whether an identity ceremony exercised slab residency."""
from __future__ import annotations

from dataclasses import dataclass

from .. import mesh_setup
from .setup_binding import render_setup_token


@dataclass
class SlabProof:
    b: dict
    cycle: list[dict]
    complete: bool
    conclusive: bool
    reasons: list[str]
    expected: bool
    active: bool
    family: str | None
    error: str | None = None


def _cycle_state(raw_cycle: list, handle) -> tuple[
    list[dict], bool, bool, list[str], bool, bool, str | None
]:
    reasons: set[str] = set()
    raw_world = getattr(handle, "world", None)
    expected_world = raw_world if type(raw_world) is int else 0
    expected_generation = getattr(handle, "setup_generation", None)
    raw_is_exact_list = type(raw_cycle) is list
    raw_length = len(raw_cycle) if raw_is_exact_list else 0
    cycle = (
        [result for result in raw_cycle if type(result) is dict]
        if raw_is_exact_list
        else []
    )
    complete_shape = (
        raw_is_exact_list
        and expected_world > 0
        and raw_length == expected_world
        and len(cycle) == raw_length
    )
    if not complete_shape:
        reasons.add("gate swap cycle did not report every rank")
    ranks = [result.get("rank") for result in cycle]
    if (
        any(type(rank) is not int for rank in ranks)
        or len(set(ranks)) != len(ranks)
        or set(ranks) != set(range(expected_world))
    ):
        reasons.add("gate swap cycle rank evidence is incomplete or duplicated")
    worlds = [result.get("world") for result in cycle]
    if (
        type(raw_world) is not int
        or raw_world < 1
        or any(type(world) is not int or world != expected_world for world in worlds)
    ):
        reasons.add("gate swap cycle world evidence is inconsistent")
    generations = [result.get("setup_generation") for result in cycle]
    if (
        type(expected_generation) is not int
        or expected_generation < 1
        or any(
            type(generation) is not int
            or generation < 1
            or generation != expected_generation
            for generation in generations
        )
    ):
        reasons.add("gate swap cycle setup generation drifted")
    complete = complete_shape and not reasons
    families = [result.get("family") for result in cycle]
    family = (str(families[0]) if complete and families and isinstance(families[0], str)
              and all(value == families[0] for value in families) else None)
    reasons.update(str(result["reason"]) for result in cycle if result.get("reason"))
    if family is None:
        reasons.add("workers did not report one consistent resident family")
    conclusive = complete and family is not None and all(
        bool(result.get("conclusive")) for result in cycle)
    reported_active = [bool(result.get("slab_active")) for result in cycle]
    active = complete and all(reported_active)
    return (
        cycle,
        complete,
        conclusive,
        sorted(reasons),
        active,
        any(reported_active),
        family,
    )


def establish(model, model_request: dict, request: dict, artifact_binding: dict,
              ceremony_identity: dict, latent: dict, cfg_value: float,
              steps_hint: int, handle, worker_args: dict, b: dict,
              raw_cycle: list, render_fn) -> SlabProof:
    """Retry once on a fresh load if slab was expected but not active on every rank, then require it."""
    from ..actor.model_store import SLAB_VOUCHED_FAMILIES
    from .gate_identity import bind_request, copy_transaction

    frozen_request = request
    frozen_latent = latent

    def state(values):
        return _cycle_state(values, handle)

    def cycle_again():
        token = render_setup_token(model, frozen_latent, cfg_value, handle)
        return handle.call_all(
            "gate_swap_cycle", model_request["unet_name"],
            dict(model_request.get("options") or {}),
            [dict(entry) for entry in model_request.get("loras") or []],
            ceremony_identity, timeout_s=1800,
            **mesh_setup.token_kwargs(token))

    cycle, complete, conclusive, reasons, active, any_active, family = state(raw_cycle)
    mode = worker_args.get("slab_weights")
    from ..compile_policy import compile_dit_blocks_slab

    compile_blocks_slab = compile_dit_blocks_slab(
        family, bool(worker_args.get("compile_dit")))

    def slab_expected(resident_family: str | None) -> bool:
        if compile_dit_blocks_slab(resident_family, bool(worker_args.get("compile_dit"))):
            return False
        auto = mode is not False and mode is not True
        return mode is True or (auto and resident_family in SLAB_VOUCHED_FAMILIES)

    auto = mode is not False and mode is not True

    def active_policy_mismatch() -> bool:
        return any_active and (
            mode is False or compile_blocks_slab
            or (auto and family not in SLAB_VOUCHED_FAMILIES))

    policy_mismatch = active_policy_mismatch()
    expected = slab_expected(family)
    family_changed = False
    if not policy_mismatch and expected and not (active and family is not None):
        initial_family = family
        handle.call_all("unload", timeout_s=600)
        b = render_fn(
            model,
            bind_request(copy_transaction(frozen_request), artifact_binding),
            copy_transaction(frozen_latent),
            cfg_value=cfg_value, steps_hint=steps_hint)
        cycle, complete, conclusive, reasons, active, any_active, family = state(cycle_again())
        compile_blocks_slab = compile_dit_blocks_slab(
            family, bool(worker_args.get("compile_dit")))
        policy_mismatch = active_policy_mismatch()
        expected = expected or slab_expected(family)
        family_changed = initial_family is not None and family != initial_family

    error = None
    if policy_mismatch:
        error = (
            "slab residency was active although compile_dit requires stock residency"
            if compile_blocks_slab
            else "slab residency was active outside the configured family policy"
        )
        reasons = sorted({*reasons, error})
        conclusive = False
    elif family_changed:
        error = "resident family changed during slab residency proof"
        reasons = sorted({*reasons, error})
        conclusive = False
    elif expected and not (active and family is not None):
        error = "slab residency was requested but not exercised"
        reasons = sorted({*reasons, error})
        conclusive = False
    return SlabProof(
        b=b, cycle=cycle, complete=complete, conclusive=conclusive, reasons=reasons,
        expected=expected, active=active, family=family, error=error)
