"""Integrate optional Sol-Attn sparse attention with the dispatch contract.

Call the installed ``sol_attn`` package through its public API and validate
preconditions it cannot check. Missing dependencies raise a typed refusal;
no kernel implementation is copied into this repository.

Build xFuser long-context attention on an opaque sentinel and replace its
``ring_attn_fn``. Both plain and pad-exclusion paths call it with
``(B, T, H, D)`` after head scatter. The sentinel cannot resolve to another
kernel if binding is lost, and every call checks the binding again.

Sol is off by default and never selected by ``AUTO_TABLE``. It is approximate,
so each render requires a class-K accuracy waiver and records its use. Never
silently fall back to a different kernel. Encode ``tau`` in the kernel name
so mesh policy, capability context, and gate ledger identify the same setting.
See docs/TROUBLESHOOTING.md #84.
"""
from __future__ import annotations

import math
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import Final

import torch

from .. import accuracy_waiver
from ..log import get_logger
from ..refusal import RefusalClass, refusal
from . import sol_attention_guards as guards
from .base import UnsupportedModelError, _make_usp_attention_callable
from .sol_attention_backend import enable_cute_on_gb10, load_sol_attn, sol_attn_available  # noqa: F401

log = get_logger(__name__)

# The kernel name carries its own tau. Parsing lives here and nowhere else.
SOL_KERNEL_PREFIX: Final = "SOL_ATTN_TAU"
# Every widget value this build ships; each tau is its own measured setting.
SOL_SELECTABLE_KERNELS: Final = ("SOL_ATTN_TAU0.6", "SOL_ATTN_TAU0.7", "SOL_ATTN_TAU1.0")

# Pinned explicitly: the installed default and the published docs disagree.
SOL_THRESH_TYPE: Final = "diag"
# 2 and 4 are SM90 only; GB10 takes 1.
SOL_KV_SPLITS: Final = 1

# Families whose geometry and evidence admit this kernel: only H3, the one
# family measured under this kernel on real activations. Growing this set means
# a guard, a ceremony and its own evidence.
SOL_ATTN_FAMILIES: Final = frozenset({"minimax_h3"})

# Why each excluded family is excluded, in one sentence, for the refusal text.
SOL_FAMILY_REASONS: Final = MappingProxyType({
    "ltx": ("its uly2 and ring2 image-conditioning legs measured 0.058 to 0.099 "
            "one-step video NRMS against a 0.10 floor on 2026-08-12, so sequence "
            "parallelism has already spent the fidelity budget a sparse kernel would need."),
    "krea2": ("it uses grouped-query attention, 48 query heads over 12 KV "
              "heads, and this kernel takes one head count for q, k and v, so "
              "reaching it means expanding K and V, the memory traffic this "
              "kernel is built to avoid."),
})
SOL_FAMILY_DEFAULT_REASON: Final = (
    "no render ceremony has measured this kernel on that family's activations, "
    "and routing density is content dependent, so a setting vouched on one "
    "family is not vouched on another."
)

# Per family, the taus a render ceremony has admitted. With no entry for a
# family, every selection takes the class-K waiver route.
SOL_VALIDATED_TAUS: Mapping[str, tuple[float, ...]] = MappingProxyType({})

# Per family, the class-K guard whose waiver a render under this kernel needs.
# A family without a guard cannot be waived, which is why growing
# SOL_ATTN_FAMILIES also means growing the frozen guard vocabulary.
# The guard name is spelled as a module constant because the refusal ledger
# resolves it statically: a class-K raise must declare a guard the AST can read,
# so a second family means a second constant and a second raise site, as the
# two ring-pad guards do.
SOL_ATTN_H3_GUARD = "sol_attn:minimax_h3"
SOL_WAIVER_GUARDS: Mapping[str, str] = MappingProxyType({
    "minimax_h3": SOL_ATTN_H3_GUARD,
})

# NVIDIA's GB10 recipe for H3, as configuration rather than a widget: their
# published config/minimax_h3/gb10_kernel_sol.toml (upstream, not in this tree)
# keeps the first 10 of 50 denoising steps and the first 2 transformer blocks
# dense, with an exact key sink over the non-video prefix. Changing these is an
# evidence change, not a logic change, as h3_calibration.py's are.
SOL_DENSE_FIRST_FRAC: Final = 0.2
SOL_DENSE_BLOCKS: Final = 2


def sol_dense_step(position: tuple[int, int] | None, t_v: float) -> bool:
    """Return whether the current step uses dense attention under the GB10 recipe.

    Keep the first ``ceil(SOL_DENSE_FIRST_FRAC * total)`` steps dense.
    ``position`` supplies the index and total from the sampler's sigma table.
    If that table is unreadable, use the timestep label and warn once: H3's
    shifted sigma schedule can keep substantially more steps dense this way.
    """
    if position is not None:
        index, total = position
        return index < math.ceil(SOL_DENSE_FIRST_FRAC * total)
    if not _warned_step_fallback:
        _warned_step_fallback.add("warned")
        log.warning(
            "sol-attn dense schedule: no sampler sigma table is readable, "
            "falling back to the timestep threshold; under a shifted sigma "
            "curve this runs more steps dense than the recipe intends")
    return t_v < SOL_DENSE_FIRST_FRAC


# Warn once per worker process; a Recycle respawns actors, so the next warns.
_warned_step_fallback: set = set()


def is_sol_kernel(name: object) -> bool:
    """True for any kernel name this module owns."""
    return isinstance(name, str) and name.startswith(SOL_KERNEL_PREFIX)


def parse_sol_tau(name: str) -> float:
    """Read the tau a sol kernel name carries, refusing an unshipped one."""
    if not is_sol_kernel(name):
        raise ValueError(f"{name!r} is not a sol-attn kernel name")
    if name not in SOL_SELECTABLE_KERNELS:
        guards.refuse_unselectable_kernel(name, SOL_SELECTABLE_KERNELS)
    return float(name[len(SOL_KERNEL_PREFIX):])


def tau_is_vouched(family: str, tau: float) -> bool:
    """Return whether cross-kernel evidence admits this family's tau.

    Admission requires a CROSS-KERNEL sol-versus-dense comparison on
    IDENTICAL topology, scored with one-step latent NRMS against the 0.10 floor.
    The shipped table is empty; the lowest measured value was 0.223 at tau 0.6.

    This is NOT the first-use identity ceremony, which compares residency
    under one kernel and cannot qualify a tau. Until cross-kernel evidence
    admits it, every render requires the class-K waiver.
    """
    return tau in SOL_VALIDATED_TAUS.get(family, ())


def assert_family_supported(family: str) -> None:
    """Refuse a family this lever is not scoped to, naming the reason."""
    if family in SOL_ATTN_FAMILIES:
        return
    guards.refuse_family(
        family, SOL_FAMILY_REASONS.get(family, SOL_FAMILY_DEFAULT_REASON))


def sol_waiver_required(family: str | None, tau: float) -> bool:
    """Refuse an unbound or unscoped family, else say whether a waiver is owed."""
    if family is None:
        guards.refuse_geometry(
            "the render reached the kernel without binding a model family, "
            "so neither the family scope nor its accuracy waiver could be "
            "checked.")
    assert_family_supported(family)
    return not tau_is_vouched(family, tau)


# Why an identity ceremony is not run under this kernel, in the one sentence the
# log line and the gate node's report both use.
SOL_CEREMONY_SKIP_REASON = (
    "the first-use identity ceremony compares residency, not kernels: every leg "
    "reuses one frozen request, so both legs would run sol-attn and the "
    "comparison could only measure this kernel against itself. A PASS there "
    "would record known-wrong math as proven, so the ceremony does not run. The "
    "render still gets no risky-residency grant without one, and it still needs "
    "its class-K accuracy waiver"
)


def sol_ceremony_skip_reason(resolved_attention: object) -> str | None:
    """The reason this context skips its ceremony, or None for every other.

    Nothing is written and nothing is quarantined; the stated limit that
    follows is in docs/TROUBLESHOOTING.md #84.
    """
    return SOL_CEREMONY_SKIP_REASON if is_sol_kernel(resolved_attention) else None


def is_sol_waiver_refusal(exc: object) -> bool:
    """Whether an aborted ceremony aborted on this lever's class-K guard.

    Read from the tag, never the message: a name cannot claim the exemption.
    """
    from ..refusal import parse_leading_refusal_tag

    text = getattr(exc, "args", (None,))[0] if getattr(exc, "args", None) else None
    for candidate in (text, str(exc)):
        tag = parse_leading_refusal_tag(candidate) if isinstance(candidate, str) else None
        guard = getattr(tag, "guard", None)
        if isinstance(guard, str) and guard in SOL_WAIVER_GUARDS.values():
            return True
    return False


class SolScope:
    """Family binding, set where the worker injects the adapter.

    The kernel is selected before the adapter is built, so the family is not
    known where the implementation is constructed. It is known at the one call
    every family passes through, including the Wan overrides that bypass the
    base adapter's dispatch hook, so the scope is bound there and read back at
    call time, the same way the head count is. One scope serves every
    implementation the dispatcher builds (attention_dispatch._persistent_sol_scope).
    """

    __slots__ = ("family",)

    def __init__(self) -> None:
        self.family: str | None = None


def bind_sol_scope(implementation, family: str) -> None:
    """Record the family this load bound, under every kernel and refusing none."""
    scope = getattr(implementation, "sol_scope", None)
    if scope is not None:
        scope.family = family


class _CallScope(threading.local):
    """The sink prefix and dense schedule for the forward on this thread."""

    def __init__(self) -> None:
        self.active = False
        self.sink_tokens: int | None = 0
        self.dense = False


_SCOPE = _CallScope()


@contextmanager
def sol_dense_scope(dense: bool) -> Iterator[None]:
    """Run the enclosed attention calls dense (torch SDPA) instead of sparse.

    The adapter decides, since only it knows the step and block index; the
    wrapper obeys per call. The dense path takes the same tensors with exact
    math, so the schedule is the only variable.
    """
    previous = _SCOPE.dense
    _SCOPE.dense = bool(dense)
    try:
        yield
    finally:
        _SCOPE.dense = previous


@contextmanager
def sol_sink_scope(sink_tokens: int | None) -> Iterator[None]:
    """Declare the contiguous key prefix this forward keeps exact.

    The kernel takes exactly one such range. The adapter owns the offset because
    only it knows the packed order; the value is a full-sequence coordinate,
    which every rank shares after the head scatter, and divisibility pads are a
    tail so excluding them cannot shift it.

    The scope is mandatory: a sol call outside one refuses (``_current_sink``),
    because it would run a different configuration from the published recipe.
    """
    if sink_tokens is not None and (
            isinstance(sink_tokens, bool) or not isinstance(sink_tokens, int)
            or sink_tokens < 0):
        raise ValueError("sol-attn sink_tokens must be a non-negative integer")
    previous = (_SCOPE.active, _SCOPE.sink_tokens)
    _SCOPE.active, _SCOPE.sink_tokens = True, sink_tokens
    try:
        yield
    finally:
        _SCOPE.active, _SCOPE.sink_tokens = previous


def _current_sink(attended_rows: int) -> int:
    """The declared exact-key prefix, validated against the attended rows."""
    if not _SCOPE.active:
        guards.refuse_geometry(
            "attention ran outside a sink scope, so the exact-key prefix this "
            "configuration depends on was never declared.")
    if _SCOPE.sink_tokens is None:
        guards.refuse_geometry(
            "this packed sequence has no video segment, so it carries no "
            "contiguous non-video prefix to keep exact, and the published "
            "recipe this lever ships cannot be applied to it.")
    if _SCOPE.sink_tokens > attended_rows:
        guards.refuse_geometry(
            f"the exact-key prefix of {_SCOPE.sink_tokens} rows exceeds the "
            f"{attended_rows} attended rows.")
    return _SCOPE.sink_tokens


class _Sentinel:
    """Opaque non-AttnType key: yunchang cannot resolve it to any kernel."""

    __slots__ = ()


def _assert_waived(family: str) -> None:
    """Refuse a render under known-wrong math without a stamped waiver.

    Checked on every call so grants and revocations apply per render, and
    ``waived`` fires its ledger record once per dispatch rather than once per
    block. The lever is class K because the kernel is not identity preserving:
    the output is a different composition, not a slightly noisier one.
    """
    if accuracy_waiver.waived(SOL_WAIVER_GUARDS[family]):
        return
    raise UnsupportedModelError(refusal(
        RefusalClass.KNOWN_WRONG,
        "the sol-attn kernel is approximate by construction and is NOT "
        "identity preserving: it re-weights which key blocks are computed "
        "exactly, which moves the sampling trajectory. Measured on this "
        "hardware, a render under it is coherent and sharp but a different "
        "composition from the same seed under exact attention (2026-08-15). "
        "No tau has passed a render ceremony, so this is known-wrong math and "
        "refuses by default. Render it under an accuracy waiver, which stamps "
        "the output, or pick a SAGE_* kernel or TORCH_FLASH for exact framing.",
        guard=SOL_ATTN_H3_GUARD,
        waivable=True,
        panel_action=accuracy_waiver.panel_action(SOL_ATTN_H3_GUARD),
        troubleshooting=84,
    ) + accuracy_waiver.card_tail(SOL_ATTN_H3_GUARD, family_hint=family))


def make_sol_usp_attention(kernel: str, sync_ulysses: bool, *,
                           ulysses: int, ring: int, scope: SolScope | None = None):
    """Build the USP attention callable that runs the installed sol kernel."""
    tau = parse_sol_tau(kernel)
    if ring != 1:
        guards.refuse_ring(ring)
    if ulysses < 1:
        raise ValueError("sol-attn ulysses degree must be positive")
    if ulysses * ring < 2:
        guards.refuse_geometry(
            "it is a sequence-parallel kernel and this render has a "
            "sequence-parallel degree of 1, so the model would use stock "
            "attention while the capability context recorded sol.")
    sol_attn, get_sol_attn_backend = load_sol_attn()
    cute = enable_cute_on_gb10()

    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
    from yunchang.kernels import AttnType, select_flash_attn_impl

    sentinel = _Sentinel()
    preflight = guards.CallPreflight(ulysses)
    scope = SolScope() if scope is None else scope

    def _waiver_required() -> bool:
        """Whether this render still needs a class-K waiver to proceed."""
        return sol_waiver_required(scope.family, tau)

    def _run(q, k, v, softmax_scale):
        """The kernel call itself, or its exact stand-in on a dense step."""
        if _waiver_required():  # backstop; the outer guard refuses first
            _assert_waived(str(scope.family))
        guards.validate_bthd(q, k, v, preflight.expected_local_heads())
        sink_tokens = _current_sink(int(q.shape[1]))
        if _SCOPE.dense:
            qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
            out = torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, scale=softmax_scale)
            return out.transpose(1, 2).contiguous()
        return sol_attn(
            q.contiguous(), k.contiguous(), v.contiguous(),
            scale=softmax_scale, tau=tau, thresh_type=SOL_THRESH_TYPE,
            kv_splits=SOL_KV_SPLITS, sink_tokens=sink_tokens, sink_start=0)

    class _SolProcessor:
        """Inner-attention fallback yunchang resolves instead of the sentinel.

        Mirrors the Wan ring construction: with an attn_processor present,
        select_flash_attn_impl returns the processor rather than resolving the
        sentinel, which is what lets a non-AttnType key survive worker setup.
        The ulysses fwd-only stage consumes only the output; sol provides no
        log-sum-exp, so the lse slot is None and ring degrees above one are
        refused before this processor can ever serve them.
        """

        __slots__ = ()

        def __call__(self, q, k, v, *, dropout_p=0.0, softmax_scale=None,
                     causal=False, window_size=(-1, -1), softcap=0.0,
                     alibi_slopes=None, return_softmax=False, **kwargs):
            if kwargs:
                guards.refuse_geometry(
                    "the inner attention received arguments this kernel does "
                    f"not take ({', '.join(sorted(kwargs))}).")
            guards.validate_kernel_surface(
                dropout_p=dropout_p, causal=causal, window_size=window_size,
                alibi_slopes=alibi_slopes, deterministic=False,
                return_attn_probs=bool(return_softmax), attn_layer=None,
                joint_tensor_key=None, joint_tensor_value=None,
                joint_strategy="none", q_descale=None, k_descale=None,
                v_descale=None, softcap=softcap)
            return _run(q, k, v, softmax_scale), None

    processor = _SolProcessor()
    if isinstance(sentinel, AttnType) or select_flash_attn_impl(
        sentinel, stage="fwd-only", attn_processor=processor
    ) is not processor:
        raise RuntimeError("yunchang does not honor the sol processor fallback")
    usp_attn = xFuserLongContextAttention(
        use_sync=sync_ulysses, attn_type=sentinel, attn_processor=processor)
    if getattr(usp_attn, "attn_type", None) is not sentinel:
        raise RuntimeError("xFuser did not retain the sol attention binding")
    if not callable(getattr(usp_attn, "ring_attn_fn", None)):
        raise RuntimeError("xFuser did not expose a callable attention path")

    @torch.compiler.disable
    def sol_ring_attn_fn(q, k, v, dropout_p=0.0, softmax_scale=None,
                         causal=False, window_size=(-1, -1), alibi_slopes=None,
                         deterministic=False, return_attn_probs=False,
                         group=None, attn_type=None, attn_processor=None,
                         attn_layer=None, joint_tensor_key=None,
                         joint_tensor_value=None, joint_strategy="none",
                         q_descale=None, k_descale=None, v_descale=None):
        guards.validate_kernel_surface(
            dropout_p=dropout_p, causal=causal, window_size=window_size,
            alibi_slopes=alibi_slopes, deterministic=deterministic,
            return_attn_probs=return_attn_probs, attn_layer=attn_layer,
            joint_tensor_key=joint_tensor_key,
            joint_tensor_value=joint_tensor_value,
            joint_strategy=joint_strategy, q_descale=q_descale,
            k_descale=k_descale, v_descale=v_descale)
        ring_world = _ring_world(group)
        if ring_world != 1:
            guards.refuse_ring(ring_world)
        return _run(q, k, v, softmax_scale)

    usp_attn.ring_attn_fn = sol_ring_attn_fn
    if usp_attn.ring_attn_fn is not sol_ring_attn_fn:
        raise RuntimeError("xFuser did not retain the sol attention path")
    implementation = _make_usp_attention_callable(
        usp_attn, surface_validator=preflight.surface)

    @torch.compiler.disable
    def guarded_implementation(*args, **kwargs):
        # Refuse before the head scatter. Both inner paths reach the kernel
        # only after an all-to-all, so checking there would collect every
        # rank's heads and then throw them away. The family is bound at
        # injection, before any sampling collective, so it is readable here on
        # the first call. The inner check stays as a backstop and costs a dict
        # lookup: waived() records once per dispatch, not once per call.
        if _waiver_required():
            _assert_waived(str(scope.family))
        # A lost binding must never run another kernel under a sol name. The
        # host decides this, not the request: a foreign override can land on
        # one box and not its sibling, so the raise stays untagged and costs one
        # Recycle (docs/TROUBLESHOOTING.md #84). Tagging it would retire the
        # lease consumed while the other rank is still inside the collective.
        if getattr(usp_attn, "ring_attn_fn", None) is not sol_ring_attn_fn:
            raise RuntimeError("the sol-attn kernel binding changed after construction")
        result = implementation(*args, **kwargs)
        if getattr(usp_attn, "ring_attn_fn", None) is not sol_ring_attn_fn:
            raise RuntimeError("the sol-attn kernel binding changed during a call")
        return result

    guarded_implementation.sol_scope = scope
    backend = "unreported"
    try:
        backend = str(get_sol_attn_backend(torch.cuda.current_device()))
    except Exception as exc:
        backend = f"unreported ({type(exc).__name__})"
    log.info(
        "sol-attn kernel %s: tau=%s thresh_type=%s kv_splits=%s backend=%s "
        "ulysses=%s cute=%s (the first call of each shape pays a kernel "
        "autotune of seconds, not milliseconds)",
        kernel, tau, SOL_THRESH_TYPE, SOL_KV_SPLITS, backend, ulysses, cute)
    return guarded_implementation


def _ring_world(group) -> int:
    if group is None:
        return 1
    import torch.distributed as dist

    return int(dist.get_world_size(group))
