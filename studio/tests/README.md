# CPU HTTP integration fixture

`mock_driver.py` is a deliberately synthetic, loopback-only Comfy-compatible
HTTP fixture. It has one named `StudioCPUFixture` node, performs no inference,
and returns a small deterministic RGB PNG. It cannot contact hardware or execute
management operations. Its telemetry explicitly identifies the CPU fixture.

The fixture answers in the shapes ComfyUI uses. POST `/prompt` keeps a
client-supplied prompt id and refuses one that is not a canonical UUID,
`/history/{prompt_id}` stays empty until the prompt finishes, and uploads stay
in memory and can be read back through `/view`. Output descriptors carry only a
file name, subfolder and type, because ComfyUI reports no width, height, frame
rate or duration for the files it saves.

Keep fixture files and evidence in a directory that only you can read, outside
the repository. Do not use a temporary directory that the host clears:

```sh
SCRATCH="$HOME/.local/state/dgxm-studio-fixture"
mkdir -p "$SCRATCH" && chmod 700 "$SCRATCH"
```

Start the fixture and export a workflow:

```sh
python3 studio/tests/mock_driver.py --port 18189 --export-workflow "$SCRATCH/studio-cpu-workflow.json"
```

In a Python environment with the optional Studio dependency installed, start a
separate companion in a terminal, with its own state directory:

```sh
python -m dgx_monarch.studio --port 18221 --driver http://127.0.0.1:18189 --state-dir "$SCRATCH/companion"
```

Do not pass a cluster configuration, a driver service, or `--allow-management`.
Put the companion's owner token in a mode-0600 file, then run:

```sh
python3 studio/tests/http_smoke.py --companion http://127.0.0.1:18221 --token-file "$SCRATCH/token"
```

The smoke verifies workflow inspection, submission preview, idempotent duplicate
submission, request reconciliation, terminal job state, immutable workflow/seed,
and authenticated artifact retrieval through the real companion HTTP routes.
It refuses drivers without the explicit fixture node and refuses enabled
mutation controls. It does not invoke the available read-only diagnostics. It
prints a JSON report whose check flags come from the companion's responses, and
exits 1 when any check fails. `tests/test_studio_tooling.py` runs the same smoke
against an in-process companion on every CPU test run.

## Browser checks

The [interface QA record](../../docs/STUDIO_INTERFACE_QA.md#browser-evidence)
lists checks that a person runs in a browser against this fixture. Run them on
the bundle you intend to ship, not an earlier build, and repeat them after any
change to `studio/src`.

1. Build the bundle and regenerate the inventory as `studio/README.md`
   describes. The `studio/static/assets/` entries of `RESOURCE_SHA256` in
   `src/dgx_monarch/studio/resource_hashes.py` name the bundle under test and
   give its SHA-256 hashes.
2. Start the fixture with a 30-second job, so that generation stays visible,
   and a short MP4 of your own for the video row:

   ```sh
   python3 studio/tests/mock_driver.py --port 18189 --job-seconds 30 \
     --video "$SCRATCH/clip.mp4" --export-workflow "$SCRATCH/studio-cpu-workflow.json"
   ```

3. Start the companion as above, open `http://127.0.0.1:18221`, enter the token
   and import the exported workflow.
4. Write down the date, the browser name and version, the operating system, the
   viewport or device, and the bundle hashes from step 1.
5. Work through each check below. Keep screenshots, fetched files and
   measurements in a new mode-0700 evidence directory outside the repository,
   called `$EVIDENCE` below.
6. List each check's outcome in one results file in that directory. Cite the
   file in the QA record by its SHA-256 (`sha256sum results.json`) and by a
   description of where it is kept, never by a temporary path.

The checks:

- **Responsive routes.** Open all eight routes at 360x800, 390x844, 768x1024
  and 1024x600 viewports. For the page and for `main`, compare `scrollWidth`
  with `clientWidth` in the developer console and record any overflow.
- **Phone controls.** At 390x844, read the computed font size and height of
  the inputs and selects.
- **Palette and motion.** Open the command palette, confirm that the search
  field takes focus, and time the reveal. Turn on reduced motion in the
  browser's rendering settings and confirm that dialogs, backdrops and
  disclosures no longer animate. Record a performance trace while a disclosure
  closes and note the longest frame interval.
- **Image previews.** On the Assets tab, select PNGs of 120x480, 480x120,
  256x256, 1024x64 and 64x1024 pixels, and measure each preview's aspect ratio
  and centering in its frame.
- **Image upload.** Upload the 120x480 portrait. Fetch the stored copy with
  `curl -o "$EVIDENCE/upload.png" "http://127.0.0.1:18189/view?type=input&subfolder=studio&filename=NAME"`,
  where `NAME` is the file name Studio shows, and confirm that it decodes to
  120x480 RGBA, which is 230,400 bytes.
- **Mask alignment.** Paint a mask at the center of the portrait, upload it,
  fetch it the same way, and record its bounding box and the alpha at the
  center and at a corner. ComfyUI's LoadImage reads transparent pixels as the
  masked area.
- **Video controls.** Select the video output. Check play, pause, mute, loop
  and the metadata panel on the phone layout. Studio offers frame stepping only
  when a record carries a frame rate, and ComfyUI reports none for the videos it
  saves, so record whether stepping is disabled. Do not add a frame rate to the
  fixture to test stepping.
- **Generation activity.** Submit the workflow and, while the 30-second job
  runs, confirm that the viewer border animates without blocking clicks. With
  reduced motion on, confirm that the border and the logo stop animating.
- **Editing and saved state.** Review a two-cell grid, edit it, and confirm that
  the confirmed submit state clears. Rename a saved preset while another draft
  is active and confirm that the preset keeps its own values. Enter two lines in
  a multiline field and confirm that both remain.
- **Operation review.** Review the read-only version operation and open its
  durable receipt. Do not apply any host operation.
- **Sign-in link.** Run it at 1440x900 and 390x844, each time with a new
  browser profile and a new state directory. With the fixture running, start
  a companion with its output sent to a file, so that it writes its owner
  token and a sign-in page to its state directory instead of printing them:

  ```sh
  python -m dgx_monarch.studio --port 18222 --driver http://127.0.0.1:18189 \
    --state-dir "$SCRATCH/link" --no-open > "$EVIDENCE/link.log"
  ```

  Open `$SCRATCH/link/sign-in.html` by its `file://` path. Expect the
  connected workspace without typing, an empty `location.hash`, and
  `sessionStorage['dgxm.studio.token.v1']` equal to the token in
  `$SCRATCH/link/owner-token`. Compare the two by SHA-256 and never write
  either value down. Record the page's requests, for example through the
  WebDriver BiDi `network.beforeRequestSent` event: no request URL and no
  `Referer` header may contain the token. Reload and confirm that the tab
  stays connected. Stop the companion, start it again with the same command
  and open the old link: the page shows "This sign-in link was not accepted:
  it is from an earlier start of Studio, or it was cut short when copied. Run
  dgxm studio open on the Studio host for a new link." and the token form.
  Close the browser, copy the profile's `places.sqlite` with its
  `places.sqlite-wal` file, and record what
  `SELECT url FROM moz_places WHERE url LIKE '%token=%'` returns, with each
  token replaced by its SHA-256. Last, start a companion on port 18223 with
  `--allow-management`, another new state directory and its output sent to a
  file: no `sign-in.html` appears in its state directory, and
  `http://127.0.0.1:18223` shows the token form.
- **Driver not running.** Run it at 1440x900 and 390x844, each time with a
  new browser profile and a new state directory. Write a configuration that
  only you can read and write:

  ```sh
  (umask 077 && printf '%s\n' '[studio]' 'driver = "http://127.0.0.1:18190"' \
    'driver_command = "echo start-the-fixture"' > "$SCRATCH/dnr.toml")
  ```

  So that the browser holds a saved draft, first start a fixture on port
  18190 and, in another terminal, a companion that uses the configuration.
  Sign in with its link, import the exported workflow, then stop both:

  ```sh
  python3 studio/tests/mock_driver.py --port 18190
  python -m dgx_monarch.studio --port 18224 --driver http://127.0.0.1:18190 --config "$SCRATCH/dnr.toml" \
    --state-dir "$SCRATCH/dnr" --no-open > "$EVIDENCE/dnr.log"
  ```

  Start the companion again with the same command while nothing listens on
  port 18190, and sign in with its new `sign-in.html`. Expect the Driver not
  running panel with the command `echo start-the-fixture`, a source line
  that names `$SCRATCH/dnr.toml`, the paragraph about another address, a
  disabled Review generation button and no red error banner. Then start the
  fixture on port 18190 again: within 5 seconds the panel is gone, a request
  to `/api/v1/catalog` succeeds, the button is enabled, and the restored draft
  shows its inputs without a reload.

Stop both fixture processes and delete the token file when you finish. These
checks are synthetic integration evidence only; they are not GPU generation,
accessibility, fidelity or performance proof. See `docs/STUDIO_ACCEPTANCE.md`
for the acceptance that remains.
