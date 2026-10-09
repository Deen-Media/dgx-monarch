"""Pure compile/residency compatibility shared by driver and workers."""
from __future__ import annotations

# The block compiler (actor/worker_compile.py) compiles only a `blocks` list,
# and these Flux-family models name theirs differently, so compile does nothing
# for them. Any other family, an unknown one included, is treated as compiled.
KNOWN_COMPILE_NOOP_FAMILIES = frozenset({"chroma", "flux", "flux2", "longcat"})


def compile_dit_is_known_noop(family: str | None) -> bool:
    """Whether a detected family has no block surface this compiler supports."""
    return family in KNOWN_COMPILE_NOOP_FAMILIES


def compile_dit_blocks_slab(family: str | None, requested: bool) -> bool:
    """Whether this compile request must keep a detected family stock-resident."""
    return bool(requested) and not compile_dit_is_known_noop(family)
