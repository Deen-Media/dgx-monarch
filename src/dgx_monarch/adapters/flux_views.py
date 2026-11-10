"""UltraTex view conditioning on the Flux family: where it runs, and the refusal elsewhere.

ComfyUI 0.45.0 (commit 61e524b) added UltraTex, LoRAs for FLUX.2 Klein 9B that
paint a mesh texture as a grid of views, and its UltraTex Conditioning node
(``comfy_extras/nodes_ultratex.py``). Beside the reference latents the node puts
three inputs on the conditioning that the Flux forward reads: ``view_grid``,
the tile each of a sample's n views takes in one image; ``token_masks``, which
``model_base.Flux.extra_conds`` turns once per conditioning into ``token_keep``
through the diffusion model's ``preprocess_token_masks``; and
``reference_views``, the tile each reference sits over. Stock
``comfy.ldm.flux.model.Flux._forward`` applies all three around its call to
``forward_orig``: it tiles each run of n consecutive batch rows into one grid,
keeps only the foreground tokens of the target and the references, places the
view references, and after the call restores the background prediction and
splits the grid back into views.

The sharded forward rebinds ``forward_orig`` alone, so under Ulysses stock's
own code does all of that and the shard sees the stream stock's
``forward_orig`` sees. ``tests/test_flux_views_comfy_exactness.py`` holds the
uly2 forward equal to stock's against a real ComfyUI, and the FSDP2 wrapped
model's ``preprocess_token_masks`` call equal to the plain model's with nothing
gathered (``fsdp_islands.MODULE_FREE_OUTSIDE_FORWARD_CALLS``). cfg-parallel
and data parallelism split the batch, whose rows are the views: a split inside
one sample raises stock's ``view_grid`` error mid-render, and a split between
samples runs batch-1 calls no test has compared with one GPU's call. Ring
attention is never stock's math, and no Ulysses degree above 2 is proven. So a
render of these families whose conditioning carries any of the three runs on
one GPU, uly2 or uly2+fsdp. Every other topology refuses class P before
dispatch on the driver and again on every Worker rank before sampling
(docs/TROUBLESHOOTING.md #148), and auto skips the cfg rows for it
(``topology.choose_auto_topology``).

Older ComfyUI has none of these names, so a conditioning there never carries
them and nothing here changes a render.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..refusal import RefusalClass, refusal

# The conditioning keys stock's Flux forward reads for UltraTex, as
# model_base.Flux.extra_conds reads them: present and not None.
VIEW_COND_KEYS = ("view_grid", "token_masks", "reference_views")
# The adapters whose diffusion model is comfy.ldm.flux.model.Flux. Chroma and
# Radiance run their own, which ignores these keys.
VIEW_FAMILIES = frozenset({"flux", "flux2", "longcat"})
# The latent tag the sampler, Gate and Sampler Custom nodes set, so every site
# that resolves auto reads the same answer (nodes/render_preflight.sampler_latent).
VIEWS_METADATA_KEY = "_dgxm_flux_views"
TROUBLESHOOTING = 148
_WORKING = (
    "Use topology uly2, or uly2+fsdp where the model does not fit resident, on a two-GPU "
    "mesh, or run this workflow on one GPU (mode=local with gpus_per_host=1, topology "
    "single), where ComfyUI's own forward runs.")


def conditioning_carries_views(*conditionings: Any) -> bool:
    """Whether any entry of these CONDITIONING lists carries a view input.

    Reads raw lists and packed ones alike: ``pack_conditioning`` keeps each
    ``[tensor, extras]`` entry and only moves its tensors to the CPU. A
    malformed list or entry carries nothing.
    """
    for conditioning in conditionings:
        for entry in conditioning if isinstance(conditioning, list | tuple) else ():
            extras = entry[1] if isinstance(entry, list | tuple) and len(entry) > 1 else None
            if isinstance(extras, Mapping) and any(
                    extras.get(key) is not None for key in VIEW_COND_KEYS):
                return True
    return False


def request_carries_views(request: Any) -> bool:
    """Whether a render request's prompts carry a view input: its own pair, or the
    guider's on Sampler Custom. The driver and every Worker rank read the same."""
    if not isinstance(request, Mapping):
        return False
    prompts = [request.get("positive"), request.get("negative")]
    guider = request.get("guider")
    if isinstance(guider, Mapping):
        prompts += [guider.get("positive"), guider.get("negative")]
    return conditioning_carries_views(*prompts)


def _degree(topology: Any, name: str) -> int:
    value = topology.get(name, 1) if isinstance(topology, Mapping) else getattr(topology, name, 1)
    return int(value or 1)


def runs_views(topology: Any) -> bool:
    """Whether a topology is proven for view conditioning: no cfg, dp or ring degree,
    and Ulysses at 1 or 2, FSDP or not. Takes a ``Topology`` or a Worker's dict."""
    return (all(_degree(topology, name) == 1 for name in ("cfg", "dp", "ring"))
            and _degree(topology, "ulysses") <= 2)


def describe(topology: Any) -> str:
    """The preset-style name of a ``Topology`` or a Worker's topology dict."""
    parts = [f"{name}{_degree(topology, key)}" for key, name in (
        ("ulysses", "uly"), ("ring", "ring"), ("cfg", "cfg"), ("dp", "dp"))
        if _degree(topology, key) > 1]
    fsdp = topology.get("fsdp") if isinstance(topology, Mapping) else getattr(topology, "fsdp", False)
    return "+".join(parts + (["fsdp"] if fsdp else [])) or "single"


def refuse_unproven_views(family: str | None, topology: Any, request: Any, *, preset: Any = None) -> None:
    """Refuse a view-conditioned render of a Flux-family model off a proven topology.

    The driver calls it after the topology resolves, with the family from the
    checkpoint header (nodes/model_patch_preflight.py), and every Worker rank
    calls it after the load, with the loaded adapter's family and its own
    topology (actor/sample_protocol.py). Each reads the same request and issues
    no collective, so every rank refuses alike before the first exchange.
    """
    if family not in VIEW_FAMILIES or runs_views(topology) or not request_carries_views(request):
        return
    from .base import UnsupportedModelError

    chosen = f"topology preset {preset!r}" if preset is not None else "this render's topology"
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"{family}: this render's conditioning carries UltraTex view inputs (view_grid, "
        f"token_masks or reference_views, from ComfyUI's UltraTex Conditioning node), and "
        f"{chosen} resolves {describe(topology)}. Stock Flux tiles each sample's views, which "
        "are consecutive batch rows, into one image inside its forward. cfg-parallel and data "
        "parallelism split that batch across ranks, ring attention is not stock's math, and "
        "only one GPU and Ulysses at degree 2 are proven equal to stock for these inputs. "
        f"{_WORKING}",
        troubleshooting=TROUBLESHOOTING,
    ))


def refuse_auto_views(family: str, topology: Any, world: int) -> None:
    """Refuse auto where no proven topology tiles the world (world above 2)."""
    if family not in VIEW_FAMILIES or runs_views(topology):
        return
    from .base import UnsupportedModelError

    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"auto: this {family} render's conditioning carries UltraTex view inputs, and on world "
        f"{world} auto resolves {describe(topology)}, which splits the batch of views or runs "
        "ring attention. No topology proven for these inputs tiles that world. "
        f"{_WORKING}",
        troubleshooting=TROUBLESHOOTING,
    ))
