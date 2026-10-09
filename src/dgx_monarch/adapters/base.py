"""Adapter protocol and shared sequence-parallel plumbing.

Adapters bind family forwards to xFuser attention, declare padding and mask
rules, and install idempotently at model load. Each family requires cross-rank
identity and single-GPU fidelity evidence; see docs/ADAPTERS.md.
"""
from __future__ import annotations

import time
import types
from dataclasses import dataclass
from typing import overload

import torch

from .. import accuracy_waiver
from ..log import get_logger
from ..refusal import RefusalClass, refusal

log = get_logger(__name__)


class ThrottledWarning:
    """Rate-limit repeated denoise warnings without silencing later workflows."""

    def __init__(self, interval_s: float = 60.0) -> None:
        self._interval = interval_s
        self._last = 0.0

    def warn(self, message: str, *args) -> None:
        now = time.monotonic()
        if now - self._last >= self._interval:
            self._last = now
            log.warning(message, *args)


class UnsupportedModelError(RuntimeError):
    """Typed refusal for an unsupported model or feature."""


def _mask_is_noop(mask) -> bool:
    """Return whether a boolean or additive mask leaves every key visible."""
    if mask.dtype == torch.bool:
        return bool(mask.all())
    return not bool(mask.any())


def pad_seq_to_multiple(t: torch.Tensor, multiple: int, dim: int = 1) -> tuple[torch.Tensor, int]:
    """Zero-pad `dim` to a multiple of `multiple`; returns (padded, original_len)."""
    orig = t.size(dim)
    short = (multiple - orig % multiple) % multiple
    if short == 0:
        return t, orig
    pad_shape = list(t.shape)
    pad_shape[dim] = short
    return torch.cat([t, t.new_zeros(pad_shape)], dim=dim), orig


def sp_world() -> int:
    from xfuser.core.distributed import get_sequence_parallel_world_size

    return get_sequence_parallel_world_size()


def sp_rank() -> int:
    from xfuser.core.distributed import get_sequence_parallel_rank

    return get_sequence_parallel_rank()


@overload
def shard_seq(
    t: torch.Tensor,
    dim: int = 1,
    *,
    allow_padding: bool = True,
    name: str | None = None,
) -> tuple[torch.Tensor, int]: ...


@overload
def shard_seq(
    t: None,
    dim: int = 1,
    *,
    allow_padding: bool = True,
    name: str | None = None,
) -> tuple[None, int]: ...


def shard_seq(
    t: torch.Tensor | None,
    dim: int = 1,
    *,
    allow_padding: bool = True,
    name: str | None = None,
) -> tuple[torch.Tensor | None, int]:
    """Pad to a multiple of the SP degree and return this rank's contiguous chunk.

    Every aligned token tensor must take the same zero pad, later trimmed by
    `sp_gather`. A maskless kernel would attend pad rows after modulation;
    adapters with evidence for exact Ulysses exclusion pass their indices as
    `drop_rows`. Set `allow_padding=False` only for a stronger validated family
    invariant; `name` identifies that family's stream. The second result is the pre-pad length.
    """
    if t is None:
        return None, 0
    world = sp_world()
    orig = t.size(dim)
    if orig % world:
        if not allow_padding:
            subject = f"{name} stream" if name else "sequence"
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                f"{subject} length {orig} on dim {dim} is not divisible by the "
                f"sequence-parallel degree {world}, and this adapter requires exact "
                "divisibility. Use a topology whose SP degree divides this stream and "
                "whose product matches the worker world. For a stock one-GPU path, "
                "run mode=local with gpus_per_host=1.",
            ))
        t, _ = pad_seq_to_multiple(t, world, dim=dim)
    local = torch.chunk(t, world, dim=dim)[sp_rank()]
    return local, orig


def sp_gather(t: torch.Tensor, orig_len: int, dim: int = 1) -> torch.Tensor:
    """All-gather sequence chunks and restore the declared original length."""
    from xfuser.core.distributed import get_sp_group

    out = get_sp_group().all_gather(t.contiguous(), dim=dim)
    return out.narrow(dim, 0, orig_len)


def padded_row_indices(segments: list[tuple[int, int]]) -> list[int]:
    """Map segment pad rows into Ulysses' rank-interleaved full sequence.

    `segments` contains `(original, local_chunk)` in per-rank concat order.
    Padding can span multiple late ranks, so every synthetic row is mapped.
    """
    world = sp_world()
    chunk = sum(local for _, local in segments)
    rows: list[int] = []
    offset = 0
    for orig, local in segments:
        for padded_index in range(orig, local * world):
            rank, within = divmod(padded_index, local)
            rows.append(rank * chunk + offset + within)
        offset += local
    return sorted(rows)


def cfg_world() -> int:
    from xfuser.core.distributed import get_classifier_free_guidance_world_size

    return get_classifier_free_guidance_world_size()


def cfg_rank() -> int:
    from xfuser.core.distributed import get_classifier_free_guidance_rank

    return get_classifier_free_guidance_rank()


def make_usp_attention(attn_type_name: str, sync_ulysses: bool = True):
    """Build a Comfy attention callable backed by xFuser.

    Input is `(B,H,L,D)` when `skip_reshape`, otherwise `(B,L,H*D)`. Output is
    `(B,L,H*D)` unless output reshape is skipped. USP kernels accept no arbitrary
    bias: no-op masks are dropped exactly and effective masks raise a typed
    refusal. A family that expresses its bias as ordered query groups instead
    passes `query_bias_groups`, applied under Ulysses where every rank holds the
    whole key axis. `TORCH_FLASH` is the Init node default. The `TORCH_*` kernels
    need only torch; FA, FA3, AITER, FLASHINFER, NPU and the SAGE kernels need their
    own packages (yunchang.kernels). docs/VALIDATION.md records kernel results by family.
    """
    from xfuser.core.long_ctx_attention import xFuserLongContextAttention
    from yunchang.kernels import AttnType

    try:
        attn_type = AttnType[attn_type_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown attention kernel {attn_type_name!r}; valid: {[a.name for a in AttnType]}"
        ) from exc

    usp_attn = xFuserLongContextAttention(use_sync=sync_ulysses, attn_type=attn_type)
    return _make_usp_attention_callable(usp_attn)


def _make_usp_attention_callable(
    usp_attn, *, surface_validator=None, qkv_validator=None
):

    @torch.compiler.disable
    def _ulysses_full(q_l, k_l, v_l, drop_rows, kv_drop_rows, groups, sequence_order):
        """Attend at Ulysses' full-sequence point, past the divisibility pads.

        The head scatter hands every rank the whole token axis in rank order,
        which is the one place both things below are exact. Pad rows are dropped
        there and written back as zeros so `sp_gather` trims them, because after
        modulation a zero embedding is not a zero key and can corrupt the output. Per-query-group biases apply there for the same
        reason: the groups and the key axis are both complete and the weights do
        not vary by head (adapters/usp_query_bias.py). A padded stream whose
        keys come from a different padded stream passes both row sets.
        """
        from .usp_full_axis import run_full_axis
        from .usp_query_bias import query_group_attention

        if sequence_order is not None:
            return run_full_axis(usp_attn, q_l, k_l, v_l, drop_rows,
                                 kv_drop_rows, groups, sequence_order,
                                 query_group_attention)
        return run_full_axis(usp_attn, q_l, k_l, v_l, drop_rows, kv_drop_rows,
                             groups, None, query_group_attention)

    def usp_attention(q, k, v, heads, mask=None, attn_precision=None,
                      skip_reshape=False, skip_output_reshape=False,
                      enable_gqa=False, drop_rows=None, kv_drop_rows=None,
                      query_bias_groups=None, sequence_order=None, **kwargs):
        if surface_validator is not None:
            surface_validator(
                q, k, v, heads, mask, attn_precision, skip_reshape,
                skip_output_reshape, enable_gqa, drop_rows, kv_drop_rows,
                query_bias_groups, kwargs,
            )
        if mask is not None and not _mask_is_noop(mask):
            # Effective masks must route to cfg or single, never disappear.
            raise UnsupportedModelError(
                "this render carries a non-trivial attention mask, which the sharded "
                "attention kernel cannot apply. Run it on a cfg or single topology; "
                "a USP render that dropped it would be silently wrong (docs/MODELS.md)."
            )
        if skip_reshape:
            if any(t.ndim != 4 for t in (q, k, v)):
                raise UnsupportedModelError(
                    "skip_reshape attention requires q/k/v shaped (B, H, L, D)"
                )
            b, h, _, d = q.shape
            if h != heads:
                raise UnsupportedModelError(
                    f"attention declared {heads} query heads but q carries {h}"
                )
            if k.shape[0] != b or v.shape[0] != b or k.shape[-1] != d or v.shape[-1] != d:
                raise UnsupportedModelError(
                    "q/k/v batch and head dimensions must agree before USP attention"
                )
            # (B, H, L, D) -> (B, L, H, D)
            q_l, k_l, v_l = (t.transpose(1, 2) for t in (q, k, v))
        else:
            if any(t.ndim != 3 for t in (q, k, v)):
                raise UnsupportedModelError(
                    "attention requires q/k/v shaped (B, L, H*D)"
                )
            b, _, inner = q.shape
            h = heads
            if inner % h:
                raise UnsupportedModelError(
                    f"query inner dimension {inner} is not divisible by {h} heads"
                )
            d = inner // h
            if k.shape[0] != b or v.shape[0] != b:
                raise UnsupportedModelError("q/k/v batch dimensions must agree")

            def _reshape(t, name):
                if t.shape[-1] % d:
                    raise UnsupportedModelError(
                        f"{name} inner dimension {t.shape[-1]} is not divisible by head dimension {d}"
                    )
                return t.reshape(b, -1, t.shape[-1] // d, d)

            q_l = q.reshape(b, -1, h, d)
            k_l = _reshape(k, "key")
            v_l = _reshape(v, "value")

        kv_heads = k_l.shape[2]
        if v_l.shape[2] != kv_heads:
            raise UnsupportedModelError(
                f"key/value head counts differ ({kv_heads} vs {v_l.shape[2]})"
            )
        if kv_heads != h:
            if not enable_gqa:
                raise UnsupportedModelError(
                    f"attention has {h} query heads and {kv_heads} K/V heads but did not "
                    "set enable_gqa=True"
                )
            if h % kv_heads:
                raise UnsupportedModelError(
                    f"GQA query heads ({h}) must be divisible by K/V heads ({kv_heads})"
                )
            repeats = h // kv_heads
            k_l = k_l.repeat_interleave(repeats, dim=2)
            v_l = v_l.repeat_interleave(repeats, dim=2)

        if qkv_validator is not None:
            qkv_validator(q_l, k_l, v_l)

        padded = bool(drop_rows) or bool(kv_drop_rows)
        ring_world = 0
        if padded or query_bias_groups or sequence_order is not None:
            from xfuser.core.distributed import get_ring_parallel_world_size

            ring_world = get_ring_parallel_world_size()
        if padded:
            assert_ulysses_only_padding(
                ring_world, len(drop_rows or ()) + len(kv_drop_rows or ()))
        if query_bias_groups:
            assert_ulysses_only_query_bias(ring_world)
        if sequence_order is not None and ring_world != 1:
            # The joint-order descriptor is pure-Ulysses only; ring and hybrid keep their path.
            sequence_order = None
        if sequence_order is not None or query_bias_groups or (padded and ring_world == 1):
            # A self-attention pads one stream, so its keys drop where its
            # queries do; a cross attention between two padded streams does not.
            out = _ulysses_full(q_l, k_l, v_l, drop_rows,
                                drop_rows if kv_drop_rows is None else kv_drop_rows,
                                query_bias_groups, sequence_order)
        else:
            # A ring waiver permits attended pads; full-sequence indices cannot
            # be applied exactly to rank-local ring slices.
            out = usp_attn(None, q_l, k_l, v_l)  # (B, L_local, H, D)

        if not isinstance(out, torch.Tensor):
            raise UnsupportedModelError("USP attention returned a non-tensor output")
        if skip_output_reshape:
            return out.transpose(1, 2)
        return out.reshape(b, -1, h * d)

    return usp_attention


RING_PAD_GUARD = "ring_pad"


@torch.compiler.disable
def assert_ulysses_only_padding(ring_world: int, n_pad_rows: int) -> None:
    """Refuse pad rows that ring or hybrid attention cannot exclude.

    Ring has no full-sequence exclusion point, and attended pads produced
    measured left-edge corruption. A default-off class-K waiver is checked on
    every call so grants and revocations apply per render; compilation is
    disabled so one decision cannot persist into another render.
    """
    if ring_world > 1 and n_pad_rows:
        if accuracy_waiver.waived(RING_PAD_GUARD):
            return
        raise UnsupportedModelError(refusal(
            RefusalClass.KNOWN_WRONG,
            "this sequence length needs divisibility padding, and the exact "
            "pad exclusion is ulysses-only; ring/hybrid attention would "
            "attend the synthetic pad rows (left-edge corruption, "
            "measured 2026-07-10). Use a uly* or single topology, or a resolution "
            "and prompt lengths whose token counts divide the SP degree "
            "(docs/ADAPTERS.md).",
            guard=RING_PAD_GUARD,
            waivable=True,
            panel_action=accuracy_waiver.panel_action(RING_PAD_GUARD),
            troubleshooting=21,
        ) + accuracy_waiver.card_tail(RING_PAD_GUARD))


@torch.compiler.disable
def assert_ulysses_only_query_bias(ring_world: int) -> None:
    """Refuse a query-group bias that ring or hybrid attention cannot apply.

    Ulysses hands each rank the whole key axis, so a bias over that axis applies
    exactly. Ring never does: its ranks see the keys one block at a time, and
    the kernel that walks those blocks takes no bias argument (yunchang
    ring_flash_attn_func). There is nothing to waive, because no ring rank can
    be handed the weights.

    Divisibility padding is allowed with a bias. A biased stream shards alone,
    which puts its pads at the tail of the token axis, past every group
    boundary, so dropping them leaves each group biasing the tokens it named
    (adapters/usp_pad_exclusion.assert_pads_are_a_tail holds that layout).
    """
    if ring_world == 1:
        return
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        "this render biases attention between token groups, and ring or hybrid "
        "attention shards the key axis, so the weights cannot be applied where "
        "the kernel reads them. Use a pure-ulysses topology (uly2 on a two-rank "
        "fleet), or topology 'single' (mode=local with gpus_per_host=1) for a "
        "stock one-GPU run.",
        troubleshooting=80,
    ))


# The private mark on the override this build installs. Comfy exposes the same
# option key to graphs, so a reader has to tell the two writers apart
# (adapters/attention_patches.py).
USP_ATTENTION_OVERRIDE_ATTR = "_dgxm_usp_attention_override"


@dataclass
class InjectionContext:
    """Everything an adapter needs at injection time."""
    topology_sp: int              # ulysses*ring degree
    usp_attention: object         # the dispatcher's setup-bound attention callable
    pure_ulysses: bool = False    # permits the joint-order full-axis treatment


def usp_options(transformer_options: dict, usp_attention, drop_rows=None,
                sequence_order=None) -> dict:
    """Copy options with a block-loop-scoped attention override.

    `drop_rows` carries gathered-sequence pad indices into each scoped call.
    """
    options = dict(transformer_options)

    def override(_func, *args, **kwargs):
        kwargs.pop("_inside_attn_wrapper", None)
        kwargs.pop("transformer_options", None)
        if sequence_order is not None:
            kwargs["sequence_order"] = sequence_order
        return usp_attention(*args, drop_rows=drop_rows, **kwargs)

    setattr(override, USP_ATTENTION_OVERRIDE_ATTR, True)
    options["optimized_attention_override"] = override
    return options


class Adapter:
    """One model family. Subclasses override the class attributes and inject_usp."""

    family = "unknown"
    # Comfy model_base class names matched by isinstance; the registry tries
    # adapters most specific first.
    model_base_classes: tuple[str, ...] = ()
    # Optional exact types accepted after the broad gate.
    exact_model_base_classes: tuple[str, ...] = ()
    # Comfy-owned attributes consulted by exceptional detection gates.
    model_detection_attrs: tuple[str, ...] = ()
    # Cond-padding rule for cfg-parallel with asymmetric prompts:
    #   "none": conds already concatenate or the model tolerates them
    #   "pad": right-pad the shorter c_crossattn with zeros
    #   "pad+mask": add a key-padding attention mask
    #   "pad+text-mask": add a text-only attention mask
    cfg_cond_padding = "none"
    # True when inject_cfg_pad_forward trims the driver's uniform pad back off
    # at a batch-1 cfg rank, so the rank makes one GPU's call for its cond;
    # only then does the original pair's fold decide nvfp4's cfg scale
    # (docs/VALIDATION.md).
    cfg_pad_restores_stock_call = False
    # The named probe whose measured result vouches this family's exact Ulysses
    # pad-row exclusion. None keeps the divisibility refusal for a family whose
    # shard runs through flux_shared._family_shard (docs/TROUBLESHOOTING.md #77);
    # a family that wires the exclusion into its own forward (Krea2, Lens, LTX
    # and MiniMax H3 among them) does not declare it.
    usp_pad_exclusion_probe: str | None = None
    # Families whose guidance runs separate model calls have no batched
    # cond/uncond call for cfg-parallel to split.
    cfg_parallel_supported = True
    # Which comfy seam the cfg-parallel split rides. ComfyUI has no central
    # application point for WrappersMP.DIFFUSION_MODEL: each family's own
    # forward builds the executor over it, and three do not. BaseModel.
    # apply_model builds one over WrappersMP.APPLY_MODEL for every family, so
    # a forward that routes no DIFFUSION_MODEL executor declares "apply_model"
    # and the split reaches the model call there.
    cfg_split_seam = "diffusion_model"
    # The per-cond constant that keeps this family's cond and uncond out of one
    # batched call whenever the two prompts differ. comfy compares a
    # CONDConstant by value (conds.py CONDConstant.can_concat), so no driver
    # padding can equalize one; on a cfg topology a differing pair takes the
    # per-cond dispatch (adapters/cfg_dispatch.py). None where the family
    # publishes no such constant.
    cfg_batch_constant: str | None = None
    # Extra keys whose leading dimension is a shared token sequence rather than
    # image batch. DP carries these values whole on every rank.
    dp_cond_exempt_keys: frozenset[str] = frozenset()
    # This family's attention head dimension, where the model file fixes one.
    # Read by the worker's kernel substitution and inject-time capability check,
    # and by the sweep matrix (adapters/attention_capability.py). None turns all
    # three off, the live probe included, since each needs a head dimension;
    # that suits a family whose head dimension varies with the checkpoint.
    attention_head_dim: int | None = None

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        for name in self.model_base_classes:
            cls = getattr(model_base, name, None)
            if cls is not None and isinstance(base_model, cls):
                return True
        return False

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        raise NotImplementedError

    def attention_dispatch(self, dispatch):
        """Select this family instance's setup-bound attention callable."""
        return dispatch

    @staticmethod
    def bind(module, method_name: str, func) -> None:
        """Bind `func` to the instance; repeated load callbacks remain idempotent."""
        setattr(module, method_name, types.MethodType(func, module))
