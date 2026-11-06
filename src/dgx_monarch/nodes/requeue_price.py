"""Price a re-queue whose UNET loader ComfyUI kept from an earlier queue, at the sampler.

ComfyUI keeps a node's output while nothing upstream of it changes, so on a
same-driver re-queue the UNET loader does not run, and the graph price it runs
(``loader_preflight.preflight_loader_footprint``) never sees the new prompt.
On 2026-11-05 a re-queue that switched the p-h3-image prompt upsampler on was
not priced for it, and rendered only because the box had room. The sampler is
the first DGX Monarch node that renders such a queue, at every world and before
any session or Worker call, so it runs the same price itself, and so does
Identity Gate (DGX Monarch) before its ceremony (docs/DESIGN.md section 5.9,
"A re-queue the loader does not see is priced where it renders").

Only under ``auto``, whose loader defers the load to the first render; only for
a prompt whose loader did not price this checkpoint (``graph_load_price``
records the ComfyUI prompt id of every price); and only for what is not yet
held: the loader memo credits a DiT this session loaded, and the graph walk
credits what ComfyUI holds and no more (``loader_graph.credited_terms``). No
prompt id or no graph prices nothing, the loader price's own direction.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from ..old_model import OldModelRefusal
from . import graph_load_price, loader_graph, loader_preflight, old_model_clear


def _due(spec: Any, prompt: Mapping) -> bool:
    """Whether this sampler must price ``spec``: a deferred load no price has seen."""
    mesh, unet_name = getattr(spec, "mesh", None), getattr(spec, "unet_name", None)
    return (mesh is not None and isinstance(unet_name, str) and bool(unet_name)
            and graph_load_price.load_is_deferred(mesh)
            and unet_name in loader_graph.unet_names(prompt)
            and graph_load_price.priced_this_prompt(unet_name) is False)


def price_cached_loader(model: Any, prompt: object, uncond: object = None,
                        site: str = graph_load_price.SITE_SAMPLER) -> None:
    """Run the loader's graph price for each DiT this node renders, when its loader did not.

    ``site`` is the node: a sampler, or Identity Gate (DGX Monarch), which
    ComfyUI runs on every queue and which renders the loader's model too.

    ``uncond`` is a dual-model guider's second checkpoint, as the request
    carries it; it loads on the same mesh, and its own loader prices it alone,
    so it is priced here alone too. The refusal is the loader's: its sentence,
    figures first, its remedies and its cards. An Old model in the way refusal
    takes Always clear and retry as at the loader: one clear, then the same
    price on the fresh fleet, uncaught.
    """
    if not isinstance(prompt, Mapping):
        return
    specs = [model]
    if isinstance(uncond, Mapping) and getattr(model, "mesh", None) is not None:
        specs.append(SimpleNamespace(mesh=model.mesh, unet_name=uncond.get("unet_name"),
                                     options=uncond.get("options")))
    for spec in specs:
        if _due(spec, prompt):
            _price(spec, prompt, site)


def _price(spec: Any, prompt: Mapping, site: str) -> None:
    mesh = spec.mesh

    def price() -> None:
        loader_preflight.preflight_loader_footprint(
            mesh, spec.unet_name, getattr(spec, "options", None), prompt, site=site)

    try:
        price()
    except OldModelRefusal as exc:
        from .render_session import active_render_session

        if not old_model_clear.clear_for_retry(
                mesh.handle, exc, inherited=active_render_session() is not None):
            raise
        price()
