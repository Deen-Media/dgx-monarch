# Brand assets

The butterfly in `web/brand/monarch.svg` is the editable master. It was drawn
from the project owner's artwork. Its background and wing divisions are
transparent.

Use the black version on a light background and `monarch-dark.svg` on a dark
background. The small variants have heavier circuit lines and rings for
sidebar icons. Monochrome versions are also included. The README wordmark uses
outlined lettering so it does not depend on a reader's fonts.

`tools/gen_brand_assets.py` exports these variants, the desktop icon,
transparent PNGs at 256 and 1024 pixels, and `docs/media/monarch-social.png`.
The social image is a 1200 x 630 repository preview. Workflow cards use the
shared dark SVG and are exported as the JPEG files ComfyUI expects.

Install the project's dev dependencies, Cairo, and DejaVu Sans, then run:

```bash
python tools/gen_brand_assets.py
python tools/gen_template_cards.py
```

Both commands accept `--check` to check exports without writing. SVG exports must match exactly. Raster checks allow small antialiasing
differences. Regenerate and review exports after changing text or geometry.
The editable SVG uses shapes and paths, with no embedded photograph.

The sidebar adapts to the interface text color. Its attention badge and all
status colors remain separate from the logo. The terminal mark has Unicode
and ASCII forms, a monochrome option, and an animation toggle. It stays still
in idle, paused, and replay views.

`docs/media/dgxm-top.svg` shows the actual dashboard replaying a synthetic
sample in a 120 by 46 terminal. Install the `[tui]` extra and run
`python tools/gen_dgxm_top_image.py` to regenerate it after a dashboard change.
The README shows `dgxm-top.png`. Whenever the SVG changes, render the PNG
again from it at the same size (1482 by 1172) in a browser, for example
`firefox --headless --screenshot dgxm-top.png --window-size=1482,1172` on the
SVG's file URL. A browser loads the SVG's Fira Code font and draws the rail
graphs from a fallback font; CairoSVG has no fallback, so it draws those
graphs as empty boxes. The example readings are not benchmark results.
