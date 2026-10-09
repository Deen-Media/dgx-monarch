"""Generation-bound setup authorization for node-side follow-up RPCs."""
from __future__ import annotations

from .. import mesh_setup


def render_setup_token(model, latent: dict, cfg_value: float | None, handle):
    """Reassert a render's topology/policy and return its exact setup token."""
    from ..mesh import MeshHandle
    from .common import _resolve_topology_for_latent

    if not isinstance(handle, MeshHandle):
        return None

    spec = model.mesh
    topology, _sage, _reason = _resolve_topology_for_latent(
        model, latent, cfg_value, handle.world)
    return mesh_setup.ensure_request_setup(
        handle, topology, spec.attention, spec.sync_ulysses, spec.worker_args)
