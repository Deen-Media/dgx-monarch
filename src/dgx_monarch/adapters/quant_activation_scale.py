"""Match stock NVFP4 activation scales across sequence and CFG shards.

Without a static ``input_scale``, NVFP4 uses
``amax(abs(x)) / (448 * 6)`` over the activation tensor. Computing that value
independently on shards can miss outliers and produce different quantization.
Reduce local amax values across the groups that partition one stock model call,
then divide in the activation dtype to match ComfyUI's operation order.

Only NVFP4 needs this reduction: FP8 defaults to a constant, MXFP8 scales
32-element blocks within each row, and INT8 does not quantize activations.
``plan_for_call`` omits the CFG reduction when each CFG rank already runs a
complete stock call for its conditioning. Matching the scale alone does not
establish full-render fidelity; family guards still apply.
"""
from __future__ import annotations

import os
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from .. import accuracy_waiver
from ..log import get_logger
from ..refusal import RefusalClass, refusal
from . import cfg_dispatch
from .base import UnsupportedModelError

log = get_logger(__name__)
# Off switch and first-step trace, both read once per install and both failing
# open, like the env-gated compile in actor/worker_compile.py. The worker arg
# beside each variable reaches every rank, as it must: the off switch decides
# whether a rank issues a collective. The bare variable is a single-box aid. Set
# on one box of a pair, it refuses on both only for an unwaived chroma render
# (the class-K bar is read in the load guard, and the readiness exchange carries
# a refusing rank's not-ready flag). Any other render that runs a reducer stalls
# at the first one: one rank waits in a max all-reduce its peer never issues.
ENABLE_ENV = "DGXM_SHARED_ACT_SCALE"
LOG_ENV = "DGXM_LOG_ACT_SCALE"
ENABLE_ARG = "shared_act_scale"
LOG_ARG = "log_act_scale"
# The private marks, so a reader tells this build's writer from comfy's own
# state (the shape of USP_ATTENTION_OVERRIDE_ATTR in adapters/base.py).
WRAPPED_ATTR = "_dgxm_shared_activation_scale"
COVERAGE_ATTR = "_dgxm_shared_activation_scale_coverage"
# How each comfy layout picks an activation scale with no `input_scale`, keyed
# by the layout class name because that is what `module.layout_type` carries.
# "whole_tensor" is the only rule a shard can move, and the canary in
# tests/canary/comfy_seam_contracts.py maps comfy's own formats onto this table.
ACTIVATION_SCALE_RULES: Mapping[str, str] = {
    "TensorCoreFP8Layout": "constant",
    "TensorCoreFP8E4M3Layout": "constant",
    "TensorCoreFP8E5M2Layout": "constant",
    "TensorCoreMXFP8Layout": "per_block",
    "TensorWiseINT8Layout": "not_quantized",
    "TensorCoreConvRotW4A4Layout": "not_quantized",
    "AsymW4A8Int8Layout": "not_quantized",
    "TensorCoreNVFP4Layout": "whole_tensor",
}
SHARD_DEPENDENT_LAYOUTS = frozenset(
    name for name, rule in ACTIVATION_SCALE_RULES.items() if rule == "whole_tensor")
# One family-scoped guard per family measured past the fidelity floor, so a
# waiver cannot silence another. Krea2 nvfp4 reads under the floor on uly2, so
# it is not here and the class-K bar leaves it alone.
SHARD_QUANT_GUARD = "shard_quant_scale"
SHARD_QUANT_CHROMA_GUARD = "shard_quant_scale:chroma"
SHARD_QUANT_GUARDS: Mapping[str, str] = MappingProxyType({
    "chroma": SHARD_QUANT_CHROMA_GUARD,
})


def enabled_now() -> bool:
    """Whether the shared scale is on for the install about to run."""
    return os.environ.get(ENABLE_ENV, "1") != "0"


def env_from_worker_args(worker_args: Mapping[str, Any],
                         previous: Mapping[str, Any] | None = None) -> bool:
    """Write both variables from the worker args; report an enable change.

    A key the last call carried and this one drops pops its variable, so
    removing the key restores the default. A key neither call carried is left
    alone, because the bare variable is also a single-box aid. The hook wraps
    forwards at load, so a caller holding residents evicts on a True return.
    """
    was = enabled_now()
    for key, name in ((ENABLE_ARG, ENABLE_ENV), (LOG_ARG, LOG_ENV)):
        if key in worker_args:
            os.environ[name] = "1" if worker_args.get(key) else "0"
        elif previous is not None and key in previous:
            os.environ.pop(name, None)
    return was is not enabled_now()


def _nvfp4_divisor() -> float:
    """The constant comfy divides the nvfp4 amax by, read where comfy reads it."""
    try:
        from comfy_kitchen import float_utils

        return float(float_utils.F8_E4M3_MAX) * float(float_utils.F4_E2M1_MAX)
    except Exception:  # pragma: no cover - the shipped value, for a box with no ck
        return 448.0 * 6.0


NVFP4_SCALE_DIVISOR: float = _nvfp4_divisor()


def _quantized_tensor_class() -> type | None:
    """comfy's activation type, imported from comfy_kitchen, never from comfy."""
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        return QuantizedTensor
    except Exception:
        return None


@dataclass(frozen=True, slots=True)
class SharedScalePlan:
    """The reducers one model call runs, in the order every rank runs them."""

    reducers: tuple[tuple[str, Callable[[torch.Tensor], torch.Tensor]], ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.reducers)

    def __bool__(self) -> bool:
        return bool(self.reducers)


def _sp_reduce(tensor: torch.Tensor) -> torch.Tensor:
    from xfuser.core.distributed import get_sp_group

    return get_sp_group().all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)


def _cfg_reduce(tensor: torch.Tensor) -> torch.Tensor:
    from xfuser.core.distributed import get_cfg_group

    return get_cfg_group().all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)


_REDUCERS: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "sp": _sp_reduce, "cfg": _cfg_reduce}


def plan_for_topology(
    topology: Mapping[str, Any] | None,
    *,
    dual_model_cfg: bool = False,
    reducers: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
) -> SharedScalePlan:
    """The reducers this topology needs, decided once at inject time.

    A dp rank runs a complete independent model call on its own latents, and
    under dm-cfg2 each rank runs a full batch call for its own checkpoint, so
    each is already its own single-GPU reference and a max across the pair would
    move both away from it. Neither gets a reducer. Both reducers cover the
    combined case with no new group handle: max is associative and idempotent,
    and each rank sits in exactly one SP group and one cfg group.
    """
    values = dict(topology or {})

    def degree(name: str) -> int:
        try:
            return int(values.get(name, 1) or 1)
        except (TypeError, ValueError):
            return 1

    table = dict(_REDUCERS) if reducers is None else dict(reducers)
    plan: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = []
    if degree("ulysses") * degree("ring") > 1 and "sp" in table:
        plan.append(("sp", table["sp"]))
    if degree("cfg") > 1 and not dual_model_cfg and "cfg" in table:
        plan.append(("cfg", table["cfg"]))
    return SharedScalePlan(tuple(plan))


DISPATCH_EXCLUDED = frozenset({"cfg"})  # sp stays: a rank holds part of its cond


def plan_for_call(plan: SharedScalePlan) -> SharedScalePlan:
    """Omit CFG reduction when each rank runs an independent stock cond call.

    This applies to per-cond dispatch and to families whose CFG-pad forward
    restores stock inputs when the original pair would not fold. Only those
    families record ``cfg_pair_folds``. Reducing across CFG in these cases
    would combine activation maxima that stock computes separately; retain
    any SP reduction within each call.
    """
    if not plan.reducers:
        return plan
    if not cfg_dispatch.dispatch_in_progress() and cfg_dispatch.cfg_pair_folds():
        return plan
    return SharedScalePlan(tuple(
        entry for entry in plan.reducers if entry[0] not in DISPATCH_EXCLUDED))


@torch.compiler.disable
def shared_activation_scale(
    input: torch.Tensor, plan: SharedScalePlan, divisor: float,
    trace: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """The scale a single GPU would have taken over the whole tensor.

    The amax is reduced, not the scale, and the divide runs in the activation
    dtype, because that is comfy's own order. Max is exact in any dtype and a
    bfloat16 value survives bfloat16 to float32 to bfloat16 unchanged, so this
    comes out bit identical to comfy's fallback; reducing a float32 scale
    instead lands one bfloat16 ulp away. The scale stays a device tensor, and an
    all-reduce with async_op False blocks the stream rather than the host, so
    only `trace` ever syncs the host, and it adds no second collective.
    """
    flat = input.reshape(-1, input.shape[-1])
    local = (torch.zeros(1, device=input.device, dtype=torch.float32)
             if flat.numel() == 0 else flat.abs().amax().to(torch.float32).reshape(1))
    # Cloned because xfuser's all_reduce returns its own input and its docstring
    # says to assume it modifies it: without the copy the trace would print the
    # reduced amax under both labels. One element, so the copy is free.
    amax = local.clone()
    for _name, reduce in plan.reducers:
        amax = reduce(amax)
    if trace is not None:
        trace.extend((local, amax))
    return amax.to(input.dtype).reshape(()) / divisor


def comfy_would_quantize(module: Any, input: torch.Tensor) -> bool:
    """Reproduce comfy's own gate, so no collective is issued for nothing.

    An unbaked LoRA sets `weight_function`; comfy then dequantizes and never
    quantizes the activation. Every fact read here is rank replicated, so the
    gate is symmetric and no rank can wedge the group by branching alone.
    """
    if input.ndim < 2:
        return False
    if getattr(module, "_full_precision_mm", False):
        return False
    if getattr(module, "comfy_force_cast_weights", False):
        return False
    if len(getattr(module, "weight_function", ()) or ()):
        return False
    if len(getattr(module, "bias_function", ()) or ()):
        return False
    quantized = _quantized_tensor_class()
    return not (quantized is not None and isinstance(input, quantized))


@dataclass(frozen=True, slots=True)
class ScaleCoverage:
    """What one install covered, for the class-K guard to read."""

    family: str
    reducers: tuple[str, ...]
    wrapped: int
    declined: int
    shard_dependent: int
    enabled: bool

    @property
    def covered(self) -> bool:
        """Whether every module needing a shared scale got one.

        The declined count alone: the off switch declines each module it would
        have wrapped, and a static `input_scale` needs nothing either way.
        """
        return self.declined == 0


def _takes_an_input_scale(module: Any) -> bool:
    """Whether comfy would read an `input_scale` this wrapper writes.

    A module carrying `pre_quant_scale` applies that smoothing before the amax,
    so the statistic belongs to a product this wrapper never sees. An expert
    bank quantizes with no scale argument, so one written for it would be
    dropped while the collective was still paid; `num_experts` is what comfy
    sets on the bank and not on the Linear, and `isinstance` cannot tell them
    apart because comfy declares both as plain Modules.
    """
    if getattr(module, "pre_quant_scale", None) is not None:
        return False
    return getattr(module, "num_experts", None) is None


def _original_forward(module: Any):
    """The forward this wrapper replaces, never another copy of this wrapper."""
    bound = module.__dict__.get("forward")
    if bound is not None:
        return bound
    return type(module).forward.__get__(module, type(module))


def _wrap(module: Any, path: str, plan: SharedScalePlan, divisor: float,
          trace: bool) -> None:
    original = _original_forward(module)
    logged = [not trace]

    def forward(self, input, *args, **kwargs):
        if not comfy_would_quantize(self, input):
            return original(input, *args, **kwargs)
        record: list[torch.Tensor] | None = None if logged[0] else []
        active = plan_for_call(plan)  # per call, so the trace names what ran
        from . import mage_nvfp4_scale
        scale = shared_activation_scale(mage_nvfp4_scale.real_input(path, input), active, divisor, record)
        if record is not None:
            logged[0] = True
            _trace(path, record, scale, active)
        self.input_scale = scale
        try:
            return original(input, *args, **kwargs)
        finally:
            # state_dict writes input_scale as an extra quant param, so a per
            # call value must never reach a bake or a checkpoint.
            try:
                del self.input_scale
            except AttributeError:  # pragma: no cover - the finally is the point
                pass

    # A MethodType on the instance, the shape Adapter.bind uses, set here rather
    # than through it: the rebound-site registry validates a bind against a stock
    # comfy method, and the factory builds this class per call, so none names it.
    module.forward = types.MethodType(forward, module)
    # The reducer names rather than a bare flag, so a second install can tell a
    # shared-scale wrapper from one the off switch left with no reducer.
    setattr(module, WRAPPED_ATTR, plan.names)


@torch.compiler.disable
def _trace(path, record, scale, plan) -> None:
    """One line per wrapped Linear per rank, for the first step only.

    The only host sync in this module, and it runs under the knob alone.
    """
    local, reduced = record
    log.info("shared activation scale: %s local_amax=%.6g reduced_amax=%.6g "
             "scale=%.6g reducers=%s", path, float(local), float(reduced),
             float(scale), ",".join(plan.names) or "none")


def install(
    diffusion_model: Any,
    topology: Mapping[str, Any] | None,
    *,
    family: str = "model",
    dual_model_cfg: bool = False,
    reducers: Mapping[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
) -> ScaleCoverage:
    """Wrap every nvfp4 Linear that needs a shared scale, and record coverage.

    A module is wrapped only when its layout is shard dependent, it ships no
    static `input_scale` (one that does is already shard invariant), comfy would
    read a scale written onto it, and it is not already wrapped. With the off
    switch set the scan still counts, so the class-K guard reads a switched-off
    render as uncovered; under the trace knob it also wraps, with an empty plan,
    which issues no collective and takes comfy's own fallback scale unchanged,
    so the "before" leg is stock math with the per-rank amax printed.
    """
    plan = plan_for_topology(topology, dual_model_cfg=dual_model_cfg, reducers=reducers)
    enabled = enabled_now()
    trace = os.environ.get(LOG_ENV) == "1"
    divisor = NVFP4_SCALE_DIVISOR
    shard_dependent = wrapped = declined = 0
    # No module tree means no quantized Linear, so nothing to scan.
    walk = getattr(diffusion_model, "named_modules", None)
    for path, module in (walk() if callable(walk) else ()):
        if getattr(module, "layout_type", None) not in SHARD_DEPENDENT_LAYOUTS:
            continue
        shard_dependent += 1
        if getattr(module, "input_scale", None) is not None:
            continue
        mark = getattr(module, WRAPPED_ATTR, None)
        if mark is not None:
            # A second install rebinds nothing; report what the first one did,
            # so a repeated load callback reads the same coverage.
            wrapped += 1 if mark else 0
            declined += 0 if mark else 1
            continue
        if not plan:
            # Unsharded: every rank sees the whole tensor, so nothing is missing
            # and nothing is declined.
            continue
        if not _takes_an_input_scale(module):
            declined += 1
            continue
        if not enabled:
            if trace:
                _wrap(module, path, SharedScalePlan(()), divisor, trace)
            declined += 1
            continue
        _wrap(module, path, plan, divisor, trace)
        wrapped += 1
    coverage = ScaleCoverage(family=str(family), reducers=plan.names, wrapped=wrapped,
                             declined=declined, shard_dependent=shard_dependent,
                             enabled=enabled)
    try:
        setattr(diffusion_model, COVERAGE_ATTR, coverage)
    except AttributeError:  # an object with no attribute dict holds no record
        log.debug("shared activation scale: coverage not recorded on %r",
                  type(diffusion_model).__name__)
    if shard_dependent:
        log.info("shared activation scale: %d/%d nvfp4 linears wrapped, %d declined, "
                 "reducers=%s", wrapped, shard_dependent, declined,
                 ",".join(plan.names) or "none")
    # The TensorWise INT8 layout repair does not depend on topology; it runs
    # here because this is the post-load hook for quantized Linears.
    from . import int8_linear_layout
    int8_linear_layout.install(diffusion_model)
    return coverage


def coverage_of(diffusion_model: Any) -> ScaleCoverage | None:
    """The coverage this build's install recorded on a model, or None."""
    coverage = getattr(diffusion_model, COVERAGE_ATTR, None)
    return coverage if isinstance(coverage, ScaleCoverage) else None


# What a sharded chroma nvfp4 render reads, with the shared scale on and off.
# The card in consent_kinds.py carries its own copy, which is what the stamp
# and the audit row print; docs/VALIDATION.md holds the legs behind both.
CHROMA_MEASUREMENT = (
    "1-step NRMS 0.120 on uly2 and 0.129 on cfg2 with the shared scale on, "
    "0.131 and 0.134 without it, against a bf16 baseline of 0.032, floor 0.10, "
    "measured 2026-09-05")
REMEDY = ("a single or dp topology where each rank sees the whole tensor, or "
          "render this model from an fp8, mxfp8 or int8 artifact, whose "
          "activation scales do not depend on the shard.")


def _refusal_text(coverage: ScaleCoverage) -> str:
    """What this render's nvfp4 activations do, and what it reads for it."""
    family = coverage.family
    if coverage.covered:
        return (f"This {family} render quantizes its nvfp4 activations against one "
                "scale shared across the sharded group, so each rank quantizes its rows "
                "exactly as one GPU would, and the image still does not match a one-GPU "
                f"render: {CHROMA_MEASUREMENT}. The rest of that gap is not "
                f"explained yet. Use {REMEDY}")
    # The off switch is named only when it is the cause: an operator who never
    # set it would go hunting for it, and nobody can edit a layer out of a
    # checkpoint.
    first = (f"Turn the shared scale back on ({ENABLE_ARG} = true in "
             f"worker_args, or {ENABLE_ENV} unset on every worker), or use"
             if not coverage.enabled else "Use")
    return (f"{coverage.declined} nvfp4 layer(s) of this {family} render would "
            "quantize their activations against a scale each rank computes from "
            "its own shard, which is measured wrong math: the rank that misses "
            "the outlier picks a scale about twenty times too small. The shared "
            f"scale removes that term and {family} still reads past the floor "
            f"with it on ({CHROMA_MEASUREMENT}), so this topology is refused "
            f"either way. {first} {REMEDY}")


@torch.compiler.disable
def assert_shard_quant_scale_covered(diffusion_model: Any) -> None:
    """Refuse a sharded nvfp4 render of a family measured past the floor.

    The bar is the measurement and not the coverage. Chroma reads 0.120 with the
    shared scale on every layer, over the 0.10 floor, so the shared scale is
    necessary and not sufficient and the topology is refused either way; a
    coverage gap only changes what the message says. A family with no
    measurement past the floor is refused nothing and a gap there is a warning.
    The deciding facts are rank identical and are read before this render's
    first collective, so every rank raises or none does, and the waiver is read
    on every call, so revoking a grant stops the next render.
    """
    coverage = coverage_of(diffusion_model)
    if coverage is None or not coverage.reducers or not coverage.shard_dependent:
        return
    family = coverage.family
    if family not in SHARD_QUANT_GUARDS:
        if not coverage.covered:
            log.warning("shared activation scale: %d nvfp4 linears on %s are "
                        "quantizing against a per-rank scale; this family has no "
                        "measurement past the fidelity floor, so the render "
                        "proceeds", coverage.declined, family)
        return
    if accuracy_waiver.waived(SHARD_QUANT_GUARDS[family]):
        return
    # Named as a constant rather than read out of the mapping, so the refusal
    # ledger's static walk can claim the guard this site raises. A second family
    # names its own constant here, beside the row it adds to refusal.GUARDS.
    raise UnsupportedModelError(refusal(
        RefusalClass.KNOWN_WRONG,
        _refusal_text(coverage),
        guard=SHARD_QUANT_CHROMA_GUARD,
        waivable=True,
        panel_action=accuracy_waiver.panel_action(SHARD_QUANT_CHROMA_GUARD),
        troubleshooting=95,
    ) + accuracy_waiver.card_tail(SHARD_QUANT_CHROMA_GUARD, family_hint=family))
