"""Node-side spelling of the render-session API implemented in `mesh_session`.

Only the names below cross this boundary, and node callers import them from
here, so this facade stays live.
"""
from __future__ import annotations

from ..mesh_session import (  # noqa: F401  # Public render_session compatibility API.
    ConcurrentRenderSessionError,
    RenderSession,
    RetiredRenderHandleError,
    claim_render_session,
    mutation_render_session,
    require_mutation_authority,
)
