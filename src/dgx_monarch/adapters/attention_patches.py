"""Validate distributed scope for ComfyUI attention hooks and overrides.

ComfyUI stores hooks in ``transformer_options["patches"]``. Sequence
parallelism changes their token scope; CFG parallelism changes their branch
scope. Flux (including Chroma), Hunyuan, and Wan-Animate 2 share these
capability checks. Krea2 refuses attention hooks under sequence parallelism
regardless of declared capabilities.

``transformer_options["optimized_attention_override"]`` is a separate,
single-writer interface. Reject graph overrides that conflict with the
adapter's own override.
"""
from __future__ import annotations

from ..refusal import RefusalClass, refusal
from .base import USP_ATTENTION_OVERRIDE_ATTR, UnsupportedModelError

ATTENTION_PATCH_CAPABILITIES_ATTR = "_dgxm_attention_patch_capabilities"
SEQUENCE_SHARD_LOCAL_CAPABILITY = "sequence_shard_local"
CONDITION_SHARD_LOCAL_CAPABILITY = "condition_shard_local"
_ATTN1_PATCH_HOOKS = ("attn1_patch", "attn1_output_patch")
_MALFORMED_PATCHES = object()


def _entries(transformer_options: dict, hook: str) -> tuple[object, ...]:
    """Return installed callables, treating a malformed singleton as present."""
    patches = transformer_options.get("patches", {})
    if not isinstance(patches, dict):
        return (_MALFORMED_PATCHES,)
    entries = patches.get(hook, ())
    if entries is None:
        return ()
    if isinstance(entries, (list, tuple)):
        return tuple(entries)
    return (entries,)


def _has_capability(patch: object, capability: str) -> bool:
    """Read the explicit per-callable dgx-monarch attention-patch contract."""
    if not callable(patch):
        return False
    try:
        capabilities = getattr(patch, ATTENTION_PATCH_CAPABILITIES_ATTR, ())
    except Exception:
        return False
    return (
        isinstance(capabilities, (list, tuple, set, frozenset))
        and capability in capabilities
    )


def assert_usp_attention_patches_safe(
    transformer_options: dict,
    family: str,
    hook_names: tuple[str, ...] = _ATTN1_PATCH_HOOKS,
) -> None:
    """Reject attention hooks whose semantics are not sequence-shard-local.

    Under USP, Comfy invokes these hooks on rank-local q/k/v or output rows.
    A hook opts in by placing ``"sequence_shard_local"`` in the collection at
    ``_dgxm_attention_patch_capabilities`` on its callable.  That is a
    correctness contract: it must not interpret local rows as global offsets
    or require another sequence rank's values.
    """
    for hook in hook_names:
        entries = _entries(transformer_options, hook)
        if entries and not all(
            _has_capability(patch, SEQUENCE_SHARD_LOCAL_CAPABILITY)
            for patch in entries
        ):
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                f"{family}: {hook} is installed without the explicit "
                f"{SEQUENCE_SHARD_LOCAL_CAPABILITY!r} capability. Sequence-parallel "
                "attention calls this hook on each rank's own token rows, so an unmarked "
                "hook could silently use the wrong global offsets. Remove the attention "
                "patch. Run this workflow on cfg-parallel for a branch-local input hook, "
                "or on topology 'single' (mode=local with gpus_per_host=1) for either hook.",
            ))


def assert_no_foreign_attention_override(transformer_options: dict, family: str) -> None:
    """Reject a second writer to Comfy's attention-override option key.

    ``ModelPatcher.set_model_optimized_attention`` and the attention-backend node
    write the key the distributed adapters own.  An adapter that binds a
    replacement forward never reads the key, so a foreign override would be
    dropped; an adapter that threads ``usp_options`` would overwrite it.  Either
    way one writer is lost silently, so the render refuses while both intents
    are still recoverable.
    """
    override = transformer_options.get("optimized_attention_override")
    if override is None or getattr(override, USP_ATTENTION_OVERRIDE_ATTR, False):
        return
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"{family}: transformer_options already carries an "
        "'optimized_attention_override' that dgx-monarch did not install (from a "
        "ComfyUI attention-backend node or another custom node). Sequence-parallel "
        "attention owns that hook, so one of the two would be silently dropped. "
        "Remove the attention-backend node. Run this workflow on topology 'single' "
        "(mode=local with gpus_per_host=1) when you need that backend.",
    ))


def assert_cfg_attention_output_patches_safe(
    transformer_options: dict, family: str
) -> None:
    """Reject output hooks not declared safe on one CFG branch per rank.

    Input q/k/v hooks remain allowed by default: many operate independently on
    one condition branch while retaining the full token sequence.  Output hooks
    such as NAG can compare cond and uncond, however, and silently no-op after
    cfg-parallel localizes ``cond_or_uncond``.  A hook opts in by declaring the
    ``"condition_shard_local"`` capability on its callable.
    """
    hook = "attn1_output_patch"
    entries = _entries(transformer_options, hook)
    if entries and not all(
        _has_capability(patch, CONDITION_SHARD_LOCAL_CAPABILITY)
        for patch in entries
    ):
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"{family}: {hook} is installed without the explicit "
            f"{CONDITION_SHARD_LOCAL_CAPABILITY!r} capability. CFG-parallel runs one "
            "condition branch per rank, so an output hook that compares conditions, "
            "such as NAG, would silently do nothing or read incomplete inputs. Remove "
            "the attention patch. Run this workflow on topology 'single' (mode=local "
            "with gpus_per_host=1) when the patch is required.",
        ))
