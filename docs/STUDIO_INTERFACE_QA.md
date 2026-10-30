# Studio interface QA

Date: 2026-10-20; browser rows repeated on 2026-10-21, twice on 2026-10-22 and on 2026-10-28, with two rows added on 2026-10-22 and fourteen on 2026-10-28. Status: **scoped local validation; the twelve browser rows and fourteen new ones ran headless on the 2026-10-28 release candidate bundle; the three faults found there are fixed and re-checked on rebuilt bundles; draft acceptance remains incomplete**.

This pass inventoried every authored interactive control in `studio/src` and
traced its handler, conditional visibility, disabled state and recovery path.
The browser checks below used a disposable synthetic CPU fixture. Source
inspection and browser execution are separate evidence: this record does **not**
claim every control and state combination was manually exercised.

## Source coverage

At this audit snapshot, the Svelte sources contain 127 button tags, 32 selects,
54 inputs, seven textareas, 18 disclosure summaries, two links and two video
elements. These are authored tags, not runtime control counts: loops repeat
controls for inputs, artifacts, jobs, records, hosts and grid cells. Native video
controls and pointer interactions were reviewed separately. Later changes added
controls that these counts and this audit do not cover: the job review summary
with its record disclosure, Use this seed in the viewer footer and a second copy
action on gallery cards. The counts also predate these changes: the telemetry
panel and the event console lost their close buttons, Create gained Move up and
Move down for each applied style, Library lost its style mode select, and Grids
gained a how-to and a line that says why Validate every cell is disabled. The
2026-10-28 browser round ran rows on each control these changes added; this
source audit has not traced them.

| Surface | Controls and state branches inspected |
|---|---|
| Shell and connection | All eight routes; home; project selector; presentation; palette; telemetry and console toggles; console clearing; token form; reconnect; notices; cross-tab draft conflict; pending-request reconciliation and quarantine. |
| Palette and review dialogs | Search, navigation and operation choices, no matches, opening/closing, keyboard entry, focus restoration, consent, busy and unavailable states, expired review and retained request identity. Added after this audit and browser-checked on 2026-10-28: the job review opens with a summary, keeps the record closed, says why Submit is off, says when it has expired and closes after a submit, and the palette has a Load a workflow group. A refused or unknown submit outcome was not tried. |
| Generation | Workflow selection/import/export/undo; operation and seed choices; all five control sections; typed choices, numbers, booleans and text; required/unresolved fields; prompt bindings; style removal; lineage; submission review. Added after this audit and browser-checked on 2026-10-28: new drafts default to Random, seed fields say when review chooses their value, and the workflow list has a filter box with a Clear button and lists workflows in title order. |
| Workflow layout and App Mode | Prominence, labels, grouping and order; explicit rebinding; raw JSON repair; invalid JSON; metadata preservation; required unresolved selections; unchanged-source validation and intentional source edits. |
| Assets and LoRAs | File selection/replacement/clearing; input binding; mask size, brush, erase, all/clear and upload; stale upload completion; LoRA search, add/remove, strength, enable and reorder; insertion loader and consumer boundaries. |
| Output viewer | Fit, actual size, zoom, panning, fullscreen, metadata, download, reuse/edit/upscale, side-by-side/wipe comparison, difference disclosure, video playback, frame stepping, mute, loop, synchronization and decode failures. Added after this audit and browser-checked on 2026-10-28: the footer shows the recorded seed and Use this seed, and a failed job's status line shows its refusal class and reason, the driver's remedy or consent request, a troubleshooting link and Review consent. Grant and retry, which needs `--allow-management`, was not run. |
| Gallery | Search, media/sort and expanded filters; project membership; card actions; comparison; provenance/clipboard/download; virtual scrolling, pagination and bounded windows. Added after this audit and browser-checked on 2026-10-28: Copy seed and Copy named inputs replace the combined copy action. |
| Queue and runtime evidence | Search/state filters, refresh, job expansion, cancellation/retry availability, full provenance, evidence type and pages, recorded-fact disclosures and retained request identities. Added after this audit and browser-checked on 2026-10-28: rows show the recorded seeds, a 64 pixel output thumbnail and a failure reason, and the expanded row shows the full failure and its consent request. Review consent, tried from the viewer's status line, opens Operations with the job's request; matching it to pending requests was not run, since a companion without `--allow-management` reads no pending list. |
| Grids and Fleet | Axes and typed values; named styles; seed policy; exclusions; preview/confirmation/dispatch; saved runs and experiments; per-cell details; selective retries; manifests; Fleet task selection and evidence. |
| Operations | Capability selection, all parameter shapes, host add/remove, review, exact-plan consent, pending recovery, unavailable/refused states and full receipt disclosure/export. |
| Project library | All six record kinds; filtering, membership, apply/edit/export/save/cancel, revisions, bounded pages, merge/copy imports, conflict choices, source repairs and import uncertainty. |
| Settings and telemetry | Layout width, panel visibility, draft import/export, disconnect, capability disclosure, metric selection, sample inspection, table disclosure and history export. |

After this audit the viewer gained a job status line with a Queue link and a
step bar, and it now shows a finished job's result unless the user chose
another. The recent outputs strip shows thumbnails of job outputs only, and the
telemetry panel gained a row for this tab's job and keeps the raw sample
closed. The 2026-10-28 browser round ran rows on them, though not on the case
where the user chose another result; no source pass has traced them.

The control audit found and corrected stale workflow repair text, loss of typed
edits when validating unchanged source, incomplete project selectors, repeated
file-selection handling, multiline editing and record updates that could replace
saved content with an unrelated active draft. Grid definition edits invalidate
the confirmed preview. Full project catalogs are bounded and shared by client
and revision rather than fetched on every snapshot poll.

## Browser evidence

This section records the rounds up to 2026-10-22. The
[2026-10-28 round](#browser-evidence-2026-10-28) on the release candidate
follows it, with its own table.

On 2026-10-22 all twelve rows below ran again on the bundle rebuilt after two
review fixes: Review generation takes empty text in a required input, and a
library import that the companion refuses with `record_window_limit` no longer
stays pending. The rows followed the
[browser procedure](../studio/tests/README.md#browser-checks).
The bundle was the set that `RESOURCE_SHA256` named on that date:
`index-BAFK232V.js`
(`2a35c594d11aff6563cfe2ec9b89982556580db46e08a889219a0e8e9b50f505`),
`index-DNoOs9Sm.css`
(`36e26466b55bed1f628f85576de439d43a6be8e2b1f862207e4e57604de0e25a`) and
`index.html` (`f122a293b96e90a39f28ee0cd9d47f05e901d083549e1523b0abf3541b5e1f5b`).
The companion served those bytes unchanged. The browser was Mozilla Firefox
158.0, build 20261007201254, headless on Ubuntu 24.04.5 LTS for arm64, driven
through WebDriver BiDi with real pointer and key input. The first ten rows ran
at 360×800, 390×844, 768×1024 and 1024×600 viewports, and the motion rows ran
again in a profile that reports reduced motion. The Sign-in link and Driver not
running rows ran at 1440×900 and 390×844, each with companions, a fixture and
new browser profiles of its own. No run logged a console message, and in each
of the six browsers the launch rows started, a capture self-test caught both a
`console.error` call and an uncaught exception. The results file holds every
measurement and names a screenshot for each row and viewport. It is kept in the
maintainer's private evidence store, outside the repository and the packages,
and its SHA-256 is
`3addd3b09e1fb092e2208e53b3b9cf44b4af594093eefe4175c03b52bd69cb8f`. Every row
passed at each viewport where its check applies. The video row first ran before
any fixture job had finished and found no video to play, a harness order, not a
Studio fault. It ran again after the generation row, and so did the Responsive
routes and Phone controls rows, so that they opened every route after jobs,
uploads, grids, presets and operation reviews existed. The results file holds
those later runs.

The first 2026-10-22 run used the launch bundle, `index-Db4Yq18N.js`
(`9bbb542027ab717e60a29b58f4d16beccd3b19ffa7a17d578fea70933ae61c34`), and
every row passed (results SHA-256
`7aaa73cb7f9e0edc5aad4875affd98f90139eaf65bd5fad35c75da434e4373c6`). Two
first attempts in that run failed because of the harness, not Studio: the video
row ran before any fixture job had finished, and the Sign-in link checks looked
for the stale-link sentence inside the token card instead of the page's error
notice.

The 2026-10-21 runs used earlier builds of the bundle. The first found the
video metadata panel at zero height at 1024×600; the viewer fix that the video
row describes was made, and the ten rows then passed on the rebuilt bundle
(results SHA-256
`4298bdb94bda062ccd2652648c9b1403e2df6379e4abbe943ac85ba25764797e`). Runs
before 2026-10-21 recorded no browser, browser version or bundle hash.

| Check | Observed result and scope |
|---|---|
| Responsive routes | All eight routes were opened at all four viewports after jobs, uploads, grids, presets and operation reviews existed. Page and main-content horizontal overflow measured zero for every route. Arbitrary user content and other expanded states remain additional combinations. The controls added since are measured in the 2026-10-28 round below. |
| Phone controls | At 360 and 390 pixels wide, all 34 visible inputs, selects and textareas across the eight routes used 16px text and at least 44px height. The rule applies up to 760 pixels wide, so the 768×1024 and 1024×600 viewports keep desktop sizing. Native phone operating-system behavior was not tested. |
| Palette and motion | The Load a workflow group came after this run; the 2026-10-28 round covers it. Palette search held focus two frames after a pointer click or Ctrl+K opened the palette, and took focus 3ms to 5ms after a scripted open. Escape closed the palette and returned focus to its button. The dialog and backdrop reveals were declared at 200ms, and the dialog's finished 201ms to 203ms after opening. With reduced motion enabled, neither animated. A disclosure-closing sample captured 35 to 39 frames with a longest interval of 17.1ms to 17.5ms in headless Firefox's software compositor; this is local interaction evidence, not a frame-rate guarantee. |
| Image previews | Portrait 120×480, landscape 480×120, square 256×256, wide 1024×64 and tall 64×1024 previews filled their frames at every viewport, with at most 0.053 percent aspect-ratio error and 0.017px centering error. |
| Image upload | At every viewport the uploaded portrait came back from the fixture as 120×480 RGBA, 230,400 decoded bytes, with pixels and file bytes identical to the source. |
| Mask alignment | Canvas bounds matched the source preview at every viewport. A pointer press at the canvas center gave a transparent box 32 or 33 pixels a side, `(44, 224, 76, 256)` at three viewports and `(43, 224, 76, 256)` at 390×844, centered within 0.03 pixels of the press. Center alpha was 0 and corner alpha 255, matching the inverse-alpha mask convention. |
| Video controls | A 2-second 24fps H.264 clip decoded, played, paused, muted and looped in the viewer. Frame stepping stayed disabled, because ComfyUI reports no frame rate for the videos it saves. Details opened the metadata panel at its full 260px at every viewport. The panel no longer shrinks, and in a short window the viewer column scrolls to it; before that fix the panel collapsed to zero height at 1024×600. This does not establish synchronization across codecs or real devices. |
| Generation activity | While a 30-second fixture job ran, the viewer border ran a 2.4-second opacity animation and passed clicks through to the viewer at every viewport. The logo animated at 768 and 1024 pixels wide; the phone layout hides it. Reduced motion stopped both. The status line and its sweeping bar came later; the 2026-10-28 round covers them, the still bar under reduced motion included. |
| Editing and saved state | Editing a reviewed two-cell grid removed its confirmation and submit control. Renaming a preset while a different draft was active kept the preset's original values. Enter kept two lines in the prompt field, also after a tab switch. |
| Operation review | The read-only version review and its durable record were opened at every viewport. No host operation was applied. The receipt list shows the oldest receipt first. |
| Sign-in link | Opening `sign-in.html` by its `file://` path connected the workspace without typing in 0.30 to 0.31 seconds. `location.hash` was empty, and the token the tab kept matched the companion's token file by SHA-256. None of the 30 requests recorded at each viewport carried the token in its request line or a `Referer` header; Firefox reported the document address with its fragment in that event, and a browser never sends the fragment. A reload stayed connected. After a restart, the old link showed the stale-link sentence in the page's error notice, above the token form. The closed profile's history database held one row for `token=`: the old link, visited twice, so the link stays in browser history. A companion started with `--allow-management` wrote no `sign-in.html` and showed the token form. |
| Driver not running | Against a closed driver port and a mode-0600 configuration whose `driver_command` is `echo start-the-fixture`, the page showed the Driver not running panel with that command, a source line naming the configuration file, the paragraph about another address and a disabled Review generation button, with no red banner, at both viewports. The catalog answered 503 with `driver_not_running`, and the restored draft waited uninspected. 1.78 seconds after the fixture listened on that port, the panel was gone, the button was enabled and the draft's three fields showed, without a reload; the catalog then answered 200. |

On 2026-10-22 the live service ran on the driver Spark of the maintainer's rig,
from the code of the `dgxm studio` launch, with a mode-0600 copy of
cluster.toml whose `[studio]` table named the driver at `127.0.0.1:8191`, port
8220 and the driver's start command. `dgxm studio up` wrote, enabled and
started `dgxm-studio.service`; a second `up` left the unit file unchanged,
issued a new token and started a new process. The unit file was mode 600 with
its marker line, loaded, active and enabled, with no drop-ins, no
`--allow-management` and no `Wants=`. Port 8220 belonged to its main process,
the token file and `sign-in.html` were mode 600 in a mode-700 directory, and
the journal named the address and both files but never the token. The
companion answered `/` with 200, its Content-Security-Policy and
`Referrer-Policy: no-referrer`. It refused a request without the token with
401 and the host name `localhost` with 403, reported the configured start
command and configuration file, and read the running driver as connected.
With its output sent to a file, `dgxm studio open` wrote the sign-in page and
printed its path, and against the real cluster.toml it named the fields that
differ. In a terminal it printed the link and opened no browser, because that
desktop opens `.html` files with another program than its web browser. The one
word acted as `open`. `--driver` beside the running unit, `token --value` to a
file and `tunnel` without a host refused with their texts. `tunnel studio-host`
and `tunnel` inside an SSH session printed the bound forward and no link.
`dgxm studio down` stopped and removed the unit and its link, left no token
file or page, and kept `studio.sqlite3` bound to the driver origin; a second
`down` changed nothing. No output, log or record held the token. The results
file is kept in the maintainer's private evidence store, and its SHA-256 is
`9b57551debc32d832399fa966151796d7c739f2c3e8596616b16fadd70a2d3f8`.
`dgxm studio token --rotate` and a restart after the process is killed are
**NOT RUN** on a rig.

Headless Firefox could not show Chromium or Safari behavior, native phone
operating-system behavior, native select pop-ups or video control internals, or
frame timing under hardware composition. The run set select values and fired
`change` instead of using the pop-up. Firefox 158 lacks `interpolate-size`, so
it never applies the disclosure transition, and the run could not show that
transition's reduced-motion change.

The earlier [acceptance record](STUDIO_ACCEPTANCE.md) separately records corrupt
request quarantine, invalid draft refusal, corrupt image/video errors, frame
controls before valid metadata, effective-setting reuse and keyboard recovery.
Those observations are not expanded into untested claims here.

## Browser evidence, 2026-10-28

On 2026-10-28 the twelve rows of the
[browser procedure](../studio/tests/README.md#browser-checks) ran on the release
candidate, and fourteen new rows covered the controls that had no browser row.
The bundle under test was the set that `RESOURCE_SHA256` named at commit
`0511b0f`: `index-CDzfDgak.js`
(`f8dc110216c2e7db8329981eacf18efcf40824a0e4718b857ba2dc0302ab2e9a`),
`index-iQ6-YgZG.css`
(`883db08d214e7c51dd3afb91989236d7f1d6aa08b6d996490dd50c2931b57225`) and
`index.html` (`455daffa9062a26f2b8b83172d98a5650eaaf0a106480c3c10028b43de878682`).
The round's companions for that bundle served those bytes unchanged. The browser
was Mozilla Firefox 159.0, build 20261022150525, headless on Ubuntu 24.04.5 LTS
for arm64, driven through WebDriver BiDi with real pointer and key input. Its profiles turned off
hardware video decoding, WebGL and GPU compositing, and no process the round
started could see a CUDA device. The rows ran at 360×800, 390×844, 768×1024 and
1024×600 viewports. The motion and job rows ran again in a profile that reports
reduced motion, a review waited out its expiry at 390×844 in a profile of its
own, and the Sign-in link and Driver not running rows ran at 1440×900 and
390×844 with companions, a fixture and new browser profiles of their own.

Most rows used the CPU fixture with a 30-second job, a 2-second 24fps 320×240
H.264 clip made for this round, and a companion on port 18221. That fixture has
no event socket and writes only successful history entries, so it cannot report
a step count or fail a prompt. The step bar, failure details, the Queue link and
App Mode with typed widgets ran instead against a QA-side subclass of the same
fixture, with its own companion on port 18225. This change commits that fixture
as `studio/tests/event_driver.py`, and the last re-run used it. For each
prompt it sends the events ComfyUI v0.43.0 sends: `execution_start`,
`executing` for node 1, `progress` 1 to 20 of 20 on node 1, `executing` for
node 2, which reports no progress, then `execution_success` or
`execution_error`. It fails a prompt that names one of two refusals built with
dgx_monarch's own refusal and consent builders, answers `/object_info` with the
recorded bundled schema and, as run here, returns PNG outputs only. For the large seed rows, a
prompt whose seed is 18446744073709551615 went straight to the CPU fixture's
`/prompt`, and the companion recorded it as an imported job.

No row logged a console message, and the browsers' own logs show no page error
besides the self-test's. In each of the thirteen browsers, a capture self-test
caught both a `console.error` call and an uncaught exception. A scan of the
evidence for the 18 owner tokens the browsers used found none outside the token
files. The results file holds every measurement and names a screenshot for each
row and viewport. It is kept in the maintainer's private evidence store,
outside the repository and the packages, and its SHA-256 is
`5bb8ac669b793811c4533e5a134e11e2eae1af31691f3dc479e3740ffe88a992`.

Six harness checks were wrong at first, and the results file marks each
correction. On the event fixture a job can read Running before the first sample,
since its start event arrives at once. Strip tiles past the strip's own scroll
edge stay unloaded by design, so the reduced-motion job rows at 360×800 and
390×844 ran again. A row that the snapshot holds is pinned in Queue without a
request. The rebuilt menu's scroll-box check now brings the whole card into view
first. The last re-run's class P text is the committed fixture's, and the job
listing that placed its oldest job stops at 200 rows.

The round found three faults from before this release train, and this change
fixes all three:

1. An open gallery Actions menu lay over its own Actions summary at every
   viewport, with no background, because `--panel` is defined nowhere. A real
   click on the summary landed on the menu, so it stayed open, and at
   768×1024 and 1024×600 the click ran Save output membership; an earlier probe
   ran Copy named inputs the same way. The menu now opens above the card's
   buttons on the surface color, at most 220 pixels tall, so it stays inside its
   card and the gallery's scroll box never clips it.
2. With saved styles, opening Grids' Saved prompt-style axes made the main area
   scroll sideways, by 71 pixels at 360×800 and 41 pixels at 390×844, because a
   style's record id cannot break and pushed Use as axis A and Use as axis B
   past the edge. The line now wraps anywhere.
3. Once later jobs pushed this tab's job out of the companion's 200-job
   snapshot, 205 of them here, the status line went empty, so a failed job's
   line, its failure text and its Open Queue link disappeared from Create while
   the job remained in Queue. The tracker read the followed job only from the
   snapshot. It now reads the job by its identity when a truncated snapshot
   leaves it out: once per snapshot while the job may still change, once after
   it has ended, and never while the snapshot carries it.

The first two fixes rebuilt the bundle as `index-DabLBaeh.js`
(`71af9af581370fd5bc5a6b7e938da9484e4f499f1a7529fd5f9d831e60cb41fc`),
`index-CEknEx1W.css`
(`c14bbf38052bfda21bf2e0e2866881b9883b63ec0363aac85a29050e4df45e09`) and
`index.html` (`ec5f0fcc86ad518187b35bff7118b714222bef37b0c891ee2b1a87e44da17aa8`).
The same toolchain writes the release candidate's bundle byte for byte from the
unchanged source, so the two style changes are the whole difference. A
companion started from this change's tree served the rebuilt bytes. On it the
Gallery actions row passed at the four viewports and 1440×900, and the style
axes, seeds, large seed, Responsive routes and Phone controls rows passed at
the four viewports, as did one full job at 390×844.

The third fix gave the shipped bundle: `index-DVitiTw1.js`
(`8304db137947757aa73c228ce57d00f41a2682c0cfcfe310bb1488416c2b53ad`), the same
`index-CEknEx1W.css` and `index.html`
(`07d963762ba3cba985be4e02f024ae4e73420153ebdbd352981ed286d92af069`). Two
companions started from this change's tree served those bytes, one on the CPU
fixture and one on `studio/tests/event_driver.py`. On them the Viewer status
line, Step bar and Failure details and Queue link rows passed again at the four
viewports, with the telemetry row for this tab's job and Review consent, and
the Queue link passed with 45, 105 and 205 newer jobs at 1024×600.

| Check | Observed result and scope |
|---|---|
| Responsive routes | Page and main-content horizontal overflow measured zero on all eight routes at all four viewports, after jobs, failures, uploads, grids, styles, presets and operation reviews existed and with every disclosure closed. The newer controls fit their boxes: the status line was 16.5px tall for a finished job, the viewer footer kept its seed and buttons in view, and on the event fixture, with failed jobs, Create and Queue measured zero too, with a consent request's status line wrapped to 395px at 360×800 and Queue reason lines wrapped beside 64×64 pixel thumbnails. A new check of each control's right edge flagged two Grids buttons at 360 and 390 pixels wide inside the closed Saved prompt-style axes disclosure, which led to fault 2. |
| Phone controls | At 360 and 390 pixels wide, all 35 visible inputs, selects and textareas across the eight routes used 16px text and at least 44px height, one more than in the last run, on Create. The 768×1024 and 1024×600 viewports keep desktop sizing. |
| Palette and motion | Palette search held focus after a pointer click and after Ctrl+K, and took focus 2ms to 5ms after a scripted open. Escape closed the palette and returned focus to its button. The dialog reveal was declared at 200ms and finished 201ms to 216ms after opening; with reduced motion neither the dialog nor the backdrop animated. A disclosure-closing sample captured 35 or 36 frames with a longest interval of 17.2ms to 17.6ms. Firefox 159 still lacks `interpolate-size`, so the disclosure transition never applies. |
| Image previews | The five test images filled their frames at every viewport, with at most 0.053 percent aspect-ratio error and 0.017px centering error. |
| Image upload | At every viewport the uploaded portrait came back from the fixture as 120×480 RGBA, 230,400 decoded bytes, with pixels and file bytes identical to the source. |
| Mask alignment | Canvas bounds matched the preview. A press at the canvas center gave a transparent box of `(44, 224, 76, 256)` at 768×1024 and 1024×600 and `(43, 224, 76, 256)` at 360×800 and 390×844, centered within 0.022 pixels of the press, with center alpha 0 and corner alpha 255. |
| Video controls | The clip decoded, played, paused, muted and looped in the viewer. Frame stepping stayed disabled, because ComfyUI reports no frame rate for the videos it saves, and Details opened the metadata panel at its full 260px at every viewport. |
| Generation activity | While a 30-second job ran, the viewer border ran its 2.4-second animation and passed clicks through to the viewer at every viewport, and the logo animated at 768 and 1024 pixels wide. With reduced motion both stood still, and the status line's bar became a dim full-width bar at opacity 0.35. |
| Editing and saved state | Editing a reviewed two-cell grid removed its confirmation and submit control. Renaming a preset while a different draft was active kept the preset's values. Enter kept two lines in the prompt field, also after a tab switch. |
| Operation review | The read-only version review and its durable record opened at every viewport, and each new receipt joined the end of the list. No host operation was applied. |
| Sign-in link | The `sign-in.html` page connected the workspace without typing in 0.27 seconds at both viewports. `location.hash` was empty and the stored token matched the token file by SHA-256. None of the 42 and 30 recorded requests carried the token in its request line or a `Referer` header. A reload stayed connected, an old link after a restart showed the stale-link sentence above the token form, the closed profile's history held one `token=` row for the old link, visited twice, and a companion started with `--allow-management` wrote no `sign-in.html` and showed the token form. |
| Driver not running | With nothing on the driver port, the page showed the panel with `echo start-the-fixture`, the source line naming the configuration file, the paragraph about another address and a disabled Review generation button, with no red banner, and the catalog answered 503 `driver_not_running`. 1.77 and 1.78 seconds after the fixture listened, the panel was gone, the button was enabled and the draft's three fields showed without a reload; the catalog then answered 200. |
| Viewer status line | After Submit the dialog closed within 0.22 seconds and the line read Queued, then Running with the elapsed time as m:ss, counting from 0:00 or 0:01 to 0:28 to 0:31 on the companion's clock, beside a sweeping bar labelled "Job progress, no step count reported". It then read "Finished in 0:30", the driver's own start to end time, and the bar went away. The telemetry panel's "This tab's job" row matched the line while running and after, and its raw sample stayed closed. The shipped bundle repeated all of it at the four viewports. |
| Step bar | On the event fixture the line read "Running 0:02" with "step 2 of 20 on node 1" beside it, rising to step 20, with a determinate bar whose fill matched the step share of its 314 to 590 pixel track within 0.01 pixels. When node 2 began, the step text and the count went away and the bar swept until the job finished. The shipped bundle repeated it at the four viewports. |
| Newest result | Within 0.03 seconds of "Finished", the viewer showed the job's last stored output without a click, and the metadata panel named the job. The recent outputs strip held up to eight job outputs and no upload or mask, marked the one in view, and showed 128×96 thumbnails for images. A video tile showed the video mark and "Preview unavailable", because the companion makes thumbnails from images only, and tiles past the strip's own scroll edge waited, unloaded, as off-screen thumbnails do. |
| Review summary and expiry | The review opened on "Ready to submit. Confirm below." with the workflow, model files, seeds, prompt text, topology, checks still to run and "This review expires at 1:33:43 AM, 10 minutes after it was made. After that, review again." and similar, with the immutable record closed until opened. Submit stayed off with "Submit is off until you check the box above." until the box was ticked. A review left open past its 600 seconds then read "Blocked: this review expired at 1:29:52 AM.", its Expiry line turned to the past tense, and Submit stayed off with the expiry reason even with the box ticked. |
| Seeds | New drafts used the Random policy, with notes saying review chooses the seeds. After a run, the review's seed, the viewer footer, the Queue row and Copy seed's clipboard all held the same digits. Use this seed put them in the seed field with the Fixed policy and said so, and one Undo restored Random and the old value. |
| Workflow filter and palette | Create listed all 71 packaged workflows by title in title order. "flux" kept the six that match by title or file name, a filter with no match said so, Escape cleared the filter, Enter loaded the first "z-image" match while no packaged workflow was chosen and did nothing once one was, and Clear emptied the box and kept focus there. The palette showed its LOAD A WORKFLOW group with a hint on an empty search, listed the 14 matches for "wan", and loaded the chosen one in Create. |
| Prompt styles and Grids how-to | Create explained style order with its worked example. With two saved styles applied, the ends of the list had Move up or Move down off with "Already first" and "Already last", review showed "((a red fox, oil painting))", Move up turned it into "((a red fox)), oil painting", and Undo restored the order. Grids opened on its five-step how-to with empty value fields, and the line under Validate every cell named each missing or malformed value, and the missing workflow in a new profile, until the axis was valid. |
| App Mode inputs | At 1024×600, each of the 71 packaged templates opened Create on the Inputs tab with its App Mode inputs in the order of its layout, 8 to 21 inputs each, all typed controls and no alert, against the recorded schema; Flux 1 text to image did the same at the other three viewports. Against the CPU fixture, whose schema lacks the templates' nodes, Flux 1 text to image showed the same ten inputs in order, each marked as missing from `object_info`. |
| Failure details and Queue link | A class P refusal read "Failed after 0:10 · Refused, class P (physics): ..." with its first sentence, Open Queue and "Troubleshooting #135". A class C rescue added the consent request, the panel button's name, Review consent and why this companion cannot grant it; Review consent opened Operations on `consent.apply` with the job, model and consent id, and nothing was applied. Open Queue opened the job's row expanded with its summary in view within 0.33 seconds, a second visit to Queue opened no row, and the row showed the full refusal. With 45 newer jobs Queue turned to History page 2 for it; with 105, outside the first 100-row load, it pinned the row above the history from the snapshot. With 205, past the companion's 200-job snapshot, the release candidate's status line went empty (fault 3). On the shipped bundle it kept "Failed after 0:10" and its link after the tracker read the job once by its identity, and the link made Queue read the job by its identity and pin its row, open and in view within 0.32 seconds. |
| Queue thumbnails | With 30 to 37 rows, Queue fetched thumbnails only for rows within reach when it opened, and none for the 23 to 26 rows further down until scrolling brought them near. Where the page scrolls, at 360 and 390 pixels wide, a row's request began 135 to 193 pixels below the visible area. Where the main area scrolls, at 768 and 1024 pixels wide, it began only once the row was 8 to 66 pixels inside the visible area, so the 200 pixel margin does not reach inside the main area. Thumbnails measured 64×64 pixels. On the event fixture they showed the images; on the CPU fixture each job's newest output is its video, so its rows showed the video mark. |
| Large seeds | For the job with seed 18446744073709551615, Copy named inputs, Copy API prompt and Export provenance kept every digit: the 101-byte and 167-byte clipboard texts and the exported file, which holds the seed in the prompt and the effective prompt, parsed back to the exact number, with no rounded form. Copy seed wrote the digits alone and the viewer footer showed them. Reuse settings, from the card and from the viewer, refused with "Settings not loaded: 1.seed is 18446744073709551615." and why, Use this seed refused with its own message, and the draft stayed unchanged each time. |
| Gallery actions | Fault 1 above, on the release candidate at all four viewports. On the rebuilt bundle a real click on Actions opened the menu above the buttons, 8 pixels or more below the card's top, on an opaque background, and a second click on Actions closed it with no action and no clipboard change, at all four viewports and 1440×900. |
| Grids style axes | Fault 2 above, at 360×800 and 390×844 on the release candidate; the list fit at 768×1024 and 1024×600. On the rebuilt bundle the open list fit at all four viewports, with every Use as axis button in view and under the pointer. |
| Panels | The telemetry panel has no close button and the event console only Clear view. The footer's Telemetry and Console toggles showed and hid them with matching `aria-expanded`, and the Settings checkboxes did the same at every viewport. |

**NOT RUN:** a step count or a failure on the CPU fixture, which cannot produce
either; Grant and retry, which needs `--allow-management` and a host operation;
Reconcile in Queue for an unknown outcome; the review expiry at any viewport
but 390×844; and Chromium, Safari, physical phones, assistive technology and GPU
rendering, as before.

## Tester round 4 checks, 2026-10-30

Each change for the fourth round of tester reports ran a headless check of the
controls it changed, on the bundle its own branch builds, against the CPU
fixtures, in Firefox 159.0 headless on Ubuntu 24.04.5 LTS for arm64 through
WebDriver BiDi with real pointer and key input, as in the
[2026-10-28 round](#browser-evidence-2026-10-28). These rows do not repeat the
browser procedure, and each names its bundle and its results file.

| Check | Observed result and scope |
|---|---|
| Send to edit | Bundle `index-CJY0lZqa.js` (`8f040e40c6ec6d4e3da8cc9b0aac789970ea9107b7874fe1a69178b08ae284b7`), `index-C_Y8FigT.css` (`4928dd9879020d7cd45433566c28335b4102077279e692f727c67e0afa0e31e8`) and `index.html` (`8a5bb97fc7d9f22041a625f9fcdcf6a41de19a594375d63aa996a511b39c44d5`), at 1440×900 and 390×844, with 2-second jobs. On `studio/tests/event_driver.py` serving the recorded bundled schema without `ConditioningZeroOut`, `TextEncodeMageFlowEdit` and `TextEncodeQwenImage21`, so that FLUX.3 Klein Base image editing is the one packaged editing workflow it runs, Send to edit on a FLUX.3 Klein Base text to image output opened no dialog. It loaded the editing workflow, set the Operation to Edit, opened Inputs with the prompt and bound a `studio/` copy to node 8's image, listed and valid, with the output and the copy in the lineage. The fixture's input store held the copy as an RGBA PNG with the output's pixels. One Undo restored the text to image draft. After a second Send to edit, Generate submitted the edit with the copy as node 8's image, and after Clear bound input on the Assets tab, Generate and Review settings were off with "Edit needs an image in the image input on node 8." under Generate once the five-second submit line cleared. With the full recorded schema, the dialog listed the six packaged editing workflows in title order with focus on the first, and the stored draft stayed the same while it was open and after Cancel. Choosing the masked RGBA reference edit bound node 6's image and left the mask loader on node 7 alone, and Send to edit on that edit's output kept the workflow with no dialog and added the new output and copy to the lineage. On `studio/tests/mock_driver.py`, which runs no packaged editing workflow, the dialog said why and named where editing workflows come from, with focus on Open Create, and the draft and the selected output stayed as they were. Pages and dialogs measured no horizontal overflow, and no console error appeared. The results file, kept in the maintainer's private evidence store, has SHA-256 `f604f08babf235d33dacb48a9aa6af7549d2d07a631f366cd26687dc5f07573a`. |

## Remaining acceptance

Physical iOS and Android devices, Safari, Chromium, assistive technology,
comprehensive keyboard and contrast coverage, every control/state combination,
full long-session performance and real hardware/service workflows remain
**NOT RUN**. Browser viewport sizing is not physical-device qualification.
Representative inference, Worker acknowledgements, native return and fidelity
acceptance retain their separate gates in
[Studio acceptance](STUDIO_ACCEPTANCE.md) and the
[performance record](STUDIO_PERFORMANCE.md).
