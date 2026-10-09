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

The TUI images in `docs/media/dgxm-top.svg` and `dgxm-top.png` show the
actual dashboard replaying sample data. The PNG keeps the same terminal-cell
proportions for the README. Its example readings are not benchmark results.
