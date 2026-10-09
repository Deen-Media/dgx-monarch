"""Driver-side checks: topology and batch before dispatch, cross-rank identity before output is trusted."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import TypeGuard, cast

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from ..topology import Topology, topology_from_preset
from .latent_outputs import is_direct_nested_tensor

log = get_logger(__name__)


class PackedDataParallelError(ValueError):
    """A public driver refusal for packed latents resolved with DP > 1."""


class PackedCfgParallelError(ValueError):
    """A public driver refusal for packed latents resolved with CFG-parallel."""


# Physics refusals about packed latents. A caller that catches broadly around a
# topology bind must re-raise these: no gate or consent setting renders this
# shape on this topology, so a fallback would hide the only useful message.
PACKED_TOPOLOGY_REFUSALS = (PackedDataParallelError, PackedCfgParallelError)


# Shared class-P message for loader-time graph checks and render-time
# tensor checks. An indivisible batch has no consent bypass; both sites
# must name the same remedy (docs/TROUBLESHOOTING.md #16).
INDIVISIBLE_BATCH = (
    "{label} on world {world} preserves its literal model-parallel degree and "
    "therefore leaves dp{dp}, but latent batch {batch} is not divisible by "
    "{dp}. Increase batch to a multiple of {dp}, choose a preset whose "
    "ulysses*ring*cfg degree consumes the full world, or use a smaller mesh."
)


def _preset_label(preset: object) -> str:
    if preset == "auto":
        return f"resolved topology for preset {preset!r}"
    return f"explicit topology preset {preset!r}"


def _topology_label(model) -> str:
    return _preset_label(model.mesh.topology_preset)


def indivisible_batch_refusal(preset: object, topo: Topology, batch: int) -> ValueError:
    """The refusal both sites raise, built once and tagged once."""
    return ValueError(refusal(RefusalClass.PHYSICS, INDIVISIBLE_BATCH.format(
        label=_preset_label(preset), world=topo.world, dp=topo.dp, batch=batch)))


def validate_render_topology(model, topo: Topology, latent_samples) -> None:
    """Reject unusable explicit DP/CFG shapes on the driver before NCCL/setup."""
    if getattr(topo, "cfg", 1) > 1 and is_direct_nested_tensor(
        latent_samples, "input latent samples"
    ):
        raise PackedCfgParallelError(refusal(
            RefusalClass.PHYSICS,
            f"{_topology_label(model)} on world {topo.world} resolves "
            f"cfg{topo.cfg}; packed multi-modality latents cannot use "
            "cfg-parallel (the cond/uncond batch cannot split across a nested "
            "audio+video latent). Run CFG > 1 on a sequence topology instead: "
            "uly2 or ring2 runs the cond and uncond passes on every rank.",
        ))
    if topo.dp <= 1:
        return
    topology_label = _topology_label(model)
    if is_direct_nested_tensor(latent_samples, "input latent samples"):
        raise PackedDataParallelError(refusal(
            RefusalClass.PHYSICS,
            f"{topology_label} on world {topo.world} leaves dp{topo.dp}; "
            "packed multi-modality latents cannot be data-parallel. Choose a preset whose ulysses*ring "
            "degree uses the full world (packed latents cannot use cfg-parallel either), or use a smaller mesh.",
        ))
    batch = int(latent_samples.shape[0])
    if batch % topo.dp:
        raise indivisible_batch_refusal(model.mesh.topology_preset, topo, batch)


# ComfyUI's own latent sources whose batch this walk can read from the graph
# without executing anything. A batch_size wired from another node, a custom
# latent source, or a chain with a node in between makes no claim:
# validate_render_topology still fires at the submit with the real tensor, so
# no claim here costs a late answer, never a wrong one.
#
# The census covers shipped and testing-only templates so supported
# latent sources receive a pre-load check. It includes TextEncodeQwenImage21
# and WanAnimate2ToVideo; test_mage_flow_templates.py covers Mage Flow.
LITERAL_BATCH_LATENT_CLASSES = frozenset({
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyHunyuanLatentVideo",
    "EmptyMochiLatentVideo",
    "EmptyLTXVLatentVideo",
    "EmptyCosmosLatentVideo",
    "EmptyAceStepLatentAudio",
    "EmptyFlux2LatentImage",
    "EmptyChromaRadianceLatentImage",
    "EmptyHunyuanImageLatent",
    "EmptyHunyuanVideo15Latent",
    # These native conditioning nodes emit their sampling latent on output 2.
    # Qwen's schema has no batch input and always creates one latent; the other
    # two expose a literal batch_size widget.
    "TextEncodeMageFlowEdit",
    "TextEncodeQwenImage21",
    "WanAnimate2ToVideo",
})

# Only the output a dgx-monarch sampler's latent_image links to counts. The
# three conditioning nodes below also expose positive and negative outputs;
# treating either as a latent source would turn a malformed graph into an early
# batch claim about data the sampler cannot consume. Read-only after import.
_LITERAL_BATCH_LATENT_OUTPUT_SLOTS = {
    **dict.fromkeys(LITERAL_BATCH_LATENT_CLASSES, 0),
    "TextEncodeMageFlowEdit": 2,
    "TextEncodeQwenImage21": 2,
    "WanAnimate2ToVideo": 2,
}

# Empty-latent sources the census knows and this walk cannot price: the node
# states width, height and length and no batch_size, so there is no literal to
# read. EmptyMiniMaxH3LatentAV builds a packed audio+video latent, which
# ``validate_render_topology`` refuses above dp1 and above cfg1, and that
# refusal is the one the operator wants; a divisibility claim here would answer
# the wrong question. Membership closes the census and makes no claim.
NO_LITERAL_BATCH_LATENT_CLASSES = frozenset({
    "EmptyMiniMaxH3LatentAV",
})

# The census reads the union of both sets; ``_literal_batches`` reads only the
# priced one.
CENSUSED_LATENT_CLASSES = LITERAL_BATCH_LATENT_CLASSES | NO_LITERAL_BATCH_LATENT_CLASSES
_LATENT_INPUT_KEY = "latent_image"


def _node(prompt: Mapping, node_id: object) -> Mapping | None:
    found = prompt.get(node_id)
    if found is None:
        found = prompt.get(str(node_id))
    return found if isinstance(found, Mapping) else None


def _literal_batches(prompt: object) -> list[int]:
    """Batch sizes this graph states as literals, in graph order.

    Only a dgx-monarch sampler's ``latent_image`` link is followed, and only
    straight into an allowlisted latent output.  A source with a literal
    positive ``batch_size`` is priced directly; Qwen Image 2.1 is the one
    native provider whose pinned schema always emits batch one. Everything
    else is absent from the result.
    """
    if not isinstance(prompt, Mapping):
        return []
    found: list[int] = []
    for node in prompt.values():
        inputs = node.get("inputs") if isinstance(node, Mapping) else None
        class_type = node.get("class_type") if isinstance(node, Mapping) else None
        if not isinstance(inputs, Mapping) or not isinstance(class_type, str):
            continue
        if not class_type.startswith("DGXMonarch"):
            continue
        link = inputs.get(_LATENT_INPUT_KEY)
        if not (isinstance(link, (list, tuple)) and len(link) == 2):
            continue
        source = _node(prompt, link[0])
        if source is None:
            continue
        source_type = source.get("class_type")
        if not isinstance(source_type, str) or source_type not in LITERAL_BATCH_LATENT_CLASSES:
            continue
        if link[1] != _LITERAL_BATCH_LATENT_OUTPUT_SLOTS[source_type]:
            continue
        source_inputs = source.get("inputs")
        if not isinstance(source_inputs, Mapping):
            continue
        if source_type == "TextEncodeQwenImage21":
            # The pinned native schema has no batch input and creates exactly
            # one latent. Do not infer a value if a future/dynamic graph adds
            # one: that graph needs an explicit validator update.
            if "batch_size" not in source_inputs:
                found.append(1)
            continue
        batch = source_inputs.get("batch_size")
        if type(batch) is int and batch > 0:
            found.append(batch)
    return found


def preflight_graph_batch_divides_dp(mesh: object, prompt: object) -> None:
    """Reject a literal batch that cannot divide across explicit DP ranks.

    Run before capacity checks: slab rescue cannot repair an indivisible batch.
    Automatic topology, an unavailable world size, or an unreadable latent chain
    makes no claim here. ``validate_render_topology`` checks the resolved topology
    and real tensor again at submission.
    """
    try:
        preset = getattr(mesh, "topology_preset", "auto")
        handle = getattr(mesh, "handle", None)
        if preset == "auto" or handle is None or getattr(handle, "defunct", False):
            return
        world = int(getattr(handle, "world", 0) or 0)
        if world <= 0:
            return
        topo = topology_from_preset(preset, world)
        if topo.dp <= 1:
            return
        batches = _literal_batches(prompt)
    except Exception:  # unreadable evidence answers nothing, here or later
        return
    for batch in batches:
        if batch % topo.dp:
            raise indivisible_batch_refusal(preset, topo, batch)


_SIGNATURE_DIGEST = re.compile(r"[0-9a-f]{64}")
_SIGNATURE_PROJECTION_SIZE = 64
_EMPTY_SIGNATURE_DIGEST = (
    "e3b0c44298fc1c149afbf4c8996fb924"
    "27ae41e4649b934ca495991b7852b855"
)


def _finite_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _valid_signature(stats: dict) -> bool:
    shape = stats.get("shape")
    dtype = stats.get("dtype")
    numel = stats.get("numel")
    mean = stats.get("mean")
    std = stats.get("std")
    digest = stats.get("sha256")
    projection = stats.get("projection")
    if (
        not isinstance(shape, list)
        or any(type(size) is not int or size < 0 for size in shape)
        or not isinstance(dtype, str)
        or not dtype
        or type(numel) is not int
        or numel < 0
        or numel != math.prod(shape)
        or not _finite_number(mean)
        or not _finite_number(std)
        or std < 0
        or not isinstance(digest, str)
        or _SIGNATURE_DIGEST.fullmatch(digest) is None
        or not isinstance(projection, list)
        or len(projection) != _SIGNATURE_PROJECTION_SIZE
        or any(not _finite_number(value) for value in projection)
    ):
        return False
    if numel == 0 and (
        mean != 0
        or std != 0
        or digest != _EMPTY_SIGNATURE_DIGEST
        or any(value != 0 for value in projection)
    ):
        return False
    if numel > 0 and digest == _EMPTY_SIGNATURE_DIGEST:
        return False
    if numel == 1 and std != 0:
        return False
    return True


def _finite(stats: dict, projection: list) -> bool:
    try:
        return all(math.isfinite(value) for value in (
            stats.get("mean"), stats.get("std"), *projection))
    except TypeError:
        return False


def _signature_parts(value: object) -> list[dict] | None:
    if isinstance(value, dict):
        return [value] if value else None
    if (isinstance(value, list) and value
            and all(isinstance(part, dict) and part for part in value)):
        return cast(list[dict], value)
    return None


def verify_cross_rank_signatures(results: list, topo: Topology) -> None:
    """Require complete rank evidence, then compare each DP group."""
    topo.validate()
    world, dp = topo.world, topo.dp
    if len(results) != world:
        raise RuntimeError(
            "cross-rank latent identity FAILED: expected "
            f"{world} rank results, got {len(results)}. This render's rank "
            "inventory is incomplete; do not trust the output."
        )

    groups: dict[int, list[tuple[int, list[dict]]]] = {}
    ranks: list[int] = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise RuntimeError(
                "cross-rank latent identity FAILED: result at index "
                f"{index} is not a mapping. Do not trust the output."
            )
        rank = result.get("rank")
        dp_rank = result.get("dp_rank")
        if type(rank) is not int:
            raise RuntimeError(
                "cross-rank latent identity FAILED: result at index "
                f"{index} has a missing or non-integer rank identity. "
                "Do not trust the output."
            )
        if type(dp_rank) is not int:
            raise RuntimeError(
                "cross-rank latent identity FAILED: rank "
                f"{rank} has a missing or non-integer DP rank identity. "
                "Do not trust the output."
            )
        parts = _signature_parts(result.get("latent_stats"))
        if parts is None or any(not _valid_signature(part) for part in parts):
            raise RuntimeError(
                "cross-rank latent identity FAILED: rank "
                f"{rank} returned missing or malformed latent signatures. "
                "Do not trust the output."
            )
        ranks.append(rank)
        groups.setdefault(dp_rank, []).append((rank, parts))

    expected_ranks = list(range(world))
    actual_ranks = sorted(ranks)
    if actual_ranks != expected_ranks:
        raise RuntimeError(
            "cross-rank latent identity FAILED: expected global ranks "
            f"{expected_ranks}, got {actual_ranks}. This render's rank "
            "inventory is incomplete or duplicated; do not trust the output."
        )
    expected_dp_ranks = list(range(dp))
    actual_dp_ranks = sorted(groups)
    if actual_dp_ranks != expected_dp_ranks:
        raise RuntimeError(
            "cross-rank latent identity FAILED: expected DP ranks "
            f"{expected_dp_ranks}, got {actual_dp_ranks}. This render's DP "
            "inventory is incomplete or invalid; do not trust the output."
        )
    expected_group_size = world // dp
    for dp_rank, group in sorted(groups.items()):
        if len(group) != expected_group_size:
            raise RuntimeError(
                "cross-rank latent identity FAILED: DP group "
                f"{dp_rank} returned {len(group)} rank results, expected "
                f"{expected_group_size}. Do not trust the output."
            )
    for dp_rank, group in sorted(groups.items()):
        for rank, _parts in group:
            expected_dp_rank = rank // topo.model_parallel
            if dp_rank != expected_dp_rank:
                raise RuntimeError(
                    "cross-rank latent identity FAILED: global rank "
                    f"{rank} reported DP rank {dp_rank}, expected "
                    f"{expected_dp_rank} for the pinned topology. Do not trust "
                    "the output."
                )

    compared, all_exact, tolerance = 0, True, 1e-3
    for dp_rank, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        reference_rank, references = group[0]
        for rank, parts in group[1:]:
            if len(parts) != len(references):
                raise RuntimeError(
                    "cross-rank latent identity FAILED: rank "
                    f"{rank} vs rank {reference_rank} in dp group {dp_rank} returned "
                    f"{len(parts)} modality signatures, expected {len(references)}. "
                    "This render's packed latent diverged; do not trust the output.")
            for modality, (reference, stats) in enumerate(zip(references, parts, strict=True)):
                compared += 1
                reference_meta = (reference.get("shape"), reference.get("dtype"),
                                  reference.get("numel"))
                metadata = (stats.get("shape"), stats.get("dtype"), stats.get("numel"))
                ref_projection, projection = (reference.get("projection") or [],
                                               stats.get("projection") or [])
                reference_finite = _finite(reference, ref_projection)
                finite = reference_finite and _finite(stats, projection)
                if (finite and metadata == reference_meta
                        and stats.get("sha256") == reference.get("sha256")):
                    continue
                all_exact = False
                scale = max(abs(reference["std"]), 1e-3) if reference_finite else 1.0
                drift = (float("inf") if not finite or len(projection) != len(ref_projection)
                         or not projection else max(
                             abs(stats["mean"] - reference["mean"]),
                             abs(stats["std"] - reference["std"]),
                             *(abs(a - b) for a, b in zip(
                                 projection, ref_projection, strict=True))) / scale)
                if metadata != reference_meta or not math.isfinite(drift) or drift > tolerance:
                    raise RuntimeError(
                        f"cross-rank latent identity FAILED: rank {rank} vs rank "
                        f"{reference_rank} in dp group {dp_rank}, modality {modality} "
                        f"(metadata {metadata} vs {reference_meta}, relative drift {drift:.2e}). "
                        "This render's parallel math diverged; do not trust the output. "
                        "Report it with the model and topology (docs/ADAPTERS.md, Acceptance gates).")
    if compared:
        log.info("cross-rank latent identity: %d comparisons across %d DP groups %s",
                 compared, len(groups), "bit-identical" if all_exact else
                 f"identical within numerical noise (<{tolerance:.0e} relative)")
