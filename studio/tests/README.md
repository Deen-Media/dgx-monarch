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

The [interface QA record](../../docs/STUDIO_INTERFACE_QA.md#browser-evidence-2026-10-28)
lists checks that a person runs in a browser against this fixture and the event
fixture below. Run them on
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

   That fixture reports no step count and never fails a prompt. For the Step
   bar, App Mode inputs and Failure details and Queue link rows, also start the
   event fixture, which serves the same routes, sends the events ComfyUI sends,
   refuses a prompt whose text holds `QA-FAIL-p` or `QA-FAIL-c` and answers
   `/object_info` with the recorded schema of the packaged templates. Start a
   second companion on it, with a state directory of its own. Run both in the
   Python environment where dgx_monarch is installed:

   ```sh
   python studio/tests/event_driver.py --port 18195 --job-seconds 30 \
     --schema tests/fixtures/studio/object_info_bundled.json
   python -m dgx_monarch.studio --port 18225 --driver http://127.0.0.1:18195 --state-dir "$SCRATCH/events"
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
- **Viewer status line.** Submit the workflow and watch the line above the
  viewer through the 30-second job. Expect Queued, then Running with the
  elapsed time as m:ss beside a sweeping bar labelled "Job progress, no step
  count reported", then "Finished in 0:30" with no bar. The telemetry panel's
  "This tab's job" row repeats the line, and its raw sample stays closed. With
  reduced motion on, the bar stands still as a dim full-width bar.
- **Step bar.** On the event fixture, submit the workflow. Expect "step N of
  20 on node 1" beside the elapsed time and a bar whose fill is N twentieths of
  its track. Once node 2 starts, the step text goes and the bar sweeps until
  the job finishes.
- **Newest result.** When a job finishes, the viewer must show its last stored
  output without a click, and Details must name the job. The recent outputs
  strip holds at most eight job outputs and no upload or mask, marks the one in
  view, and shows thumbnails for images and the video mark for videos.
- **Review summary and expiry.** Open Review generation. Expect "Ready to
  submit. Confirm below.", the workflow, model files, seeds, prompt text,
  topology, checks still to run and the time the review expires, 10 minutes
  after it was made, with the immutable record closed until opened and Submit
  off with its reason until the box is ticked. Leave one review open for more
  than 10 minutes: it must then say it expired and keep Submit off, also with
  the box ticked.
- **Seeds.** In a new browser profile, new drafts use the Random policy and say
  that review chooses the seeds. After a run, the review, the viewer footer,
  the Queue row and the clipboard after Copy seed in the gallery must hold the
  same seed. Use this seed puts it in the seed field with the Fixed policy as
  one change that one Undo reverses.
- **Workflow filter and palette.** Type in Create's workflow filter: the select
  keeps the packaged workflows whose title or file name holds every word, in
  title order, and says so when none matches. Escape clears the filter, Enter
  loads the first match while no packaged workflow is chosen and does nothing
  once one is, and Clear empties the box and keeps focus there. In the command
  palette, an empty search shows the LOAD A WORKFLOW group with its hint, and a
  typed title lists its matches; choosing one loads it in Create.
- **Prompt styles and Grids how-to.** Save two styles in Library, one plain and
  one with `{prompt}`, append both in Create and bind the prompt input. Expect
  the explanation and its worked example, Move up off on the first style and
  Move down off on the last with "Already first" and "Already last", the
  review's prompt text in each order, and Undo restoring the order. On Grids,
  expect the five-step how-to, empty value fields in a new profile, and a line
  under Validate every cell for no workflow, no axis, empty, invalid, non-list
  and empty-list values.
- **App Mode inputs.** On the event fixture, load each packaged template: Create
  must open on the Inputs tab with the inputs of the template's App Mode layout,
  `extra.linearData`, in that order, as typed controls with no alert. On this
  fixture, whose schema lacks the templates' nodes, the same order shows with
  each input marked as missing from `object_info`.
- **Failure details and Queue link.** On the event fixture, submit prompts whose
  text holds `QA-FAIL-p` and then `QA-FAIL-c`. Expect "Failed after 0:10" with
  the refusal class and first sentence, Open Queue and a Troubleshooting link,
  and for the second the consent request, the panel button's name and Review
  consent, which opens Operations on `consent.apply` with the job's request.
  Open Queue opens the job's row expanded and in view, and a second visit to
  Queue opens no row. Then post 45, 60 and 100 more prompts straight to the
  event fixture's `/prompt`, so that 45, 105 and 205 newer jobs exist: Queue
  turns to the job's history page, then pins its row above the history, and at
  205, with the job outside the companion's 200-job snapshot, the status line
  still shows the failed job and its link makes Queue read the job by its
  identity, a `GET /api/v1/jobs?job_id=` request.
- **Queue thumbnails.** With 30 or more finished jobs, open Queue and record
  the thumbnail requests, for example through the WebDriver BiDi
  `network.beforeRequestSent` event: a row more than 200 pixels below the
  visible area asks for no thumbnail until scrolling brings it near. Thumbnails
  are 64 pixels square; on this fixture each job's newest output is its video,
  so the event fixture's image outputs show the pictures.
- **Large seeds.** Post a prompt whose seed is 18446744073709551615 straight to
  the fixture, as text so that no tool rounds it:

  ```sh
  curl -sS -H 'Content-Type: application/json' http://127.0.0.1:18189/prompt --data-binary \
    '{"prompt": {"1": {"class_type": "StudioCPUFixture", "inputs": {"prompt": "QA large seed", "seed": 18446744073709551615, "strength": 1.0}}}}'
  ```

  Once the companion records the job, Copy named inputs, Copy API prompt and
  Export provenance must keep every digit, and Copy seed must write the digits
  alone. Reuse settings, from the card and the viewer, and Use this seed must
  refuse, name `1.seed` with its digits and leave the draft unchanged.
- **Gallery actions.** Open a card's Actions with a real click: the menu opens
  above the card's buttons on an opaque background, inside its card, and a
  second click on Actions closes it without running any item.
- **Grids style axes.** With saved styles, open Saved prompt-style axes on Grids
  at 360x800 and 390x844: neither the page nor the main area may scroll
  sideways, and every Use as axis button stays in view.
- **Panels.** The telemetry panel has no close button and the event console
  only Clear view. The footer's Telemetry and Console toggles show and hide
  them with matching `aria-expanded`, and so do the Settings checkboxes.

Stop the fixture processes and delete the token file when you finish. These
checks are synthetic integration evidence only; they are not GPU generation,
accessibility, fidelity or performance proof. See `docs/STUDIO_ACCEPTANCE.md`
for the acceptance that remains.
