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

## Studio screenshot

`docs/media/dgxm-studio.png` shows Studio's Create view with the CPU test
fixture of `studio/tests/mock_driver.py`. It is 1440 by 900 pixels, captured at
device pixel ratio 1, and it shows no real render: the fixture runs no model,
and no job was queued. Capture it again after a visible Studio layout change,
on the bundle you intend to ship, with any WebDriver BiDi client:

1. Build the bundle and regenerate the inventory as `studio/README.md`
   describes. The `studio/static/` entries of `RESOURCE_SHA256` in
   `src/dgx_monarch/studio/resource_hashes.py` name the bundle the image shows.
2. Make a directory that only you can read, outside the repository:

   ```bash
   SHOT="$HOME/.local/state/dgxm-studio-fixture/screenshot"
   mkdir -p "$SHOT/profile" && chmod 700 "$HOME/.local/state/dgxm-studio-fixture" "$SHOT" "$SHOT/profile"
   ```

   Run steps 3 to 5 from the repository root, each in its own terminal with
   `SHOT` set the same way.
3. Start the fixture and export its workflow:

   ```bash
   python3 studio/tests/mock_driver.py --port 18189 --export-workflow "$SHOT/studio-cpu-workflow.json"
   ```

4. Check that `ss -ltn 'sport = :8220'` prints nothing, so the header shows the
   port a reader uses. Start a companion with a new state directory and its
   output sent to a file. It then writes its owner token and `sign-in.html` to
   the state directory instead of printing them:

   ```bash
   python -m dgx_monarch.studio --port 8220 --driver http://127.0.0.1:18189 \
     --state-dir "$SHOT/state" --no-open > "$SHOT/companion.log"
   ```

5. Start Firefox headless with WebDriver BiDi and a new profile that holds three
   preferences:

   ```bash
   printf '%s\n' 'user_pref("ui.useOverlayScrollbars", 1);' 'user_pref("ui.prefersReducedMotion", 1);' \
     'user_pref("browser.shell.checkDefaultBrowser", false);' > "$SHOT/profile/user.js"
   firefox --headless --no-remote --profile "$SHOT/profile" --remote-debugging-port=9333
   ```

   The first preference removes the light scrollbar bands that headless
   Firefox draws. The second stops transitions, so the capture never catches
   one halfway.
6. With your BiDi client, open `about:blank` first: a new profile starts on
   Firefox's own start page, which a BiDi client may not resize. Then set the
   viewport to 1440 by 900 at device pixel ratio 1, read the address in
   `$SHOT/state/sign-in.html` and navigate to it. Wait until the workspace
   shows and the token form does not, then check that `location.hash` is
   empty.
7. Import `$SHOT/studio-cpu-workflow.json` through the
   `.workflow-actions input[type=file]` file input. Replace the prompt text,
   with key input, by exactly this sentence:

   ```text
   A lighthouse on a rocky coast at dusk, soft warm light
   ```

   Queue nothing. Move the pointer away from every control, wait 2 seconds,
   and save `browsingContext.captureScreenshot` as `$SHOT/raw.png`. The header
   reads `127.0.0.1:8220 / connected`.
8. Write the PNG again without metadata. Pillow writes no text chunks unless
   asked:

   ```bash
   cd "$SHOT" && python -c "from PIL import Image; Image.open('raw.png').save('dgxm-studio.png', format='PNG', optimize=True)"
   ```

9. Walk the PNG chunks. The file may hold only `IHDR`, `IDAT` and `IEND`
   chunks (`PLTE` is allowed) and at most 1,048,576 bytes;
   `tests/test_public_docs.py` and `tests/test_readme_studio.py` check the same
   after you copy it. Record `sha256sum "$SHOT/dgxm-studio.png"`.
10. Copy it to `docs/media/dgxm-studio.png`, stage it and run
    `python tools/leak_check.py`. Then look at the pixels, which the scanner
    cannot read: no host name, private address, home path, client or
    model-adapter name, personal name, tool attribution, console output or
    error banner.
11. Stop the browser, the companion and the fixture. The companion removes its
    token file and `sign-in.html` when it stops. Delete `$SHOT` when you
    finish.

The current image was captured on 2026-10-22 with these steps in Mozilla
Firefox 158.0, build 20261007201254, headless on Linux for arm64, at 1440 by
900 and device pixel ratio 1. The profile held the three preferences of step 5;
the fixture, companion and prompt were those of steps 3, 4 and 7. It shows the
bundle that `RESOURCE_SHA256` names:

| File | SHA-256 |
|---|---|
| `studio/static/index.html` | `84ab96c19208b78e7ab568d0b5706ceed06d3d7ab4cf6dd89ade63b429a9db72` |
| `studio/static/assets/index-DNoOs9Sm.css` | `36e26466b55bed1f628f85576de439d43a6be8e2b1f862207e4e57604de0e25a` |
| `studio/static/assets/index-Db4Yq18N.js` | `9bbb542027ab717e60a29b58f4d16beccd3b19ffa7a17d578fea70933ae61c34` |

The image is 164,962 bytes, with SHA-256
`2d6fbb5c05edcc3638a8b74f15a590e09cac59897c15651fdfb4a64ab1411379`. It shows
no real render: the fixture ran no model, and no job was queued.
