"""Pure compile/residency compatibility shared by driver and workers."""
from __future__ import annotations

# The block compiler (actor/worker_compile.py) compiles only a top-level `blocks`
# list. On ComfyUI 0.43.0 eight families build a DiT that has one: krea2,
# minimax_h3, wan, wan_scail, wan_animate2, wan_dancer, cogvideo and anima. Every
# family below names its block lists otherwise, so compile does nothing for it
# and its requested slab residency stays available:
#   transformer_blocks: qwen_image, qwen_image21, mage_flow, lens, ltx
#   double_blocks and single_blocks: flux, flux2, chroma, longcat, hunyuan
#   layers: ideogram4, moss_image, ernie; with refiners: zimage, omnigen2
#   staged or split lists: boogu, flux3, pixeldit_comfy, kandinsky5
# Any other family, an unknown one included, is treated as compiled and stays
# stock-resident under compile_dit. tests/test_compile_surface.py holds every
# registered family to one side and, given a ComfyUI checkout, checks each side
# against its source.
KNOWN_COMPILE_NOOP_FAMILIES = frozenset({
    "boogu", "chroma", "ernie", "flux", "flux2", "flux3", "hunyuan", "ideogram4",
    "kandinsky5", "lens", "longcat", "ltx", "mage_flow", "moss_image", "omnigen2",
    "pixeldit_comfy", "qwen_image", "qwen_image21", "zimage",
})


def compile_dit_is_known_noop(family: str | None) -> bool:
    """Whether a detected family has no block surface this compiler supports."""
    return family in KNOWN_COMPILE_NOOP_FAMILIES


def compile_dit_blocks_slab(family: str | None, requested: bool) -> bool:
    """Whether this compile request must keep a detected family stock-resident."""
    return bool(requested) and not compile_dit_is_known_noop(family)
