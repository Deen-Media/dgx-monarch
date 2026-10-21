# Studio interface QA

Date: 2026-10-20; browser rows repeated on 2026-10-21. Status: **scoped local validation; the browser rows ran headless on the shipped bundle; draft acceptance remains incomplete**.

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
controls and pointer interactions were reviewed separately.

| Surface | Controls and state branches inspected |
|---|---|
| Shell and connection | All eight routes; home; project selector; presentation; palette; telemetry and console toggles; console clearing; token form; reconnect; notices; cross-tab draft conflict; pending-request reconciliation and quarantine. |
| Palette and review dialogs | Search, navigation and operation choices, no matches, opening/closing, keyboard entry, focus restoration, consent, busy and unavailable states, expired review and retained request identity. |
| Generation | Workflow selection/import/export/undo; operation and seed choices; all five control sections; typed choices, numbers, booleans and text; required/unresolved fields; prompt bindings; style removal; lineage; submission review. |
| Workflow layout and App Mode | Prominence, labels, grouping and order; explicit rebinding; raw JSON repair; invalid JSON; metadata preservation; required unresolved selections; unchanged-source validation and intentional source edits. |
| Assets and LoRAs | File selection/replacement/clearing; input binding; mask size, brush, erase, all/clear and upload; stale upload completion; LoRA search, add/remove, strength, enable and reorder; insertion loader and consumer boundaries. |
| Output viewer | Fit, actual size, zoom, panning, fullscreen, metadata, download, reuse/edit/upscale, side-by-side/wipe comparison, difference disclosure, video playback, frame stepping, mute, loop, synchronization and decode failures. |
| Gallery | Search, media/sort and expanded filters; project membership; card actions; comparison; provenance/clipboard/download; virtual scrolling, pagination and bounded windows. |
| Queue and runtime evidence | Search/state filters, refresh, job expansion, cancellation/retry availability, full provenance, evidence type and pages, recorded-fact disclosures and retained request identities. |
| Grids and Fleet | Axes and typed values; named styles; seed policy; exclusions; preview/confirmation/dispatch; saved runs and experiments; per-cell details; selective retries; manifests; Fleet task selection and evidence. |
| Operations | Capability selection, all parameter shapes, host add/remove, review, exact-plan consent, pending recovery, unavailable/refused states and full receipt disclosure/export. |
| Project library | All six record kinds; filtering, membership, apply/edit/export/save/cancel, revisions, bounded pages, merge/copy imports, conflict choices, source repairs and import uncertainty. |
| Settings and telemetry | Layout width, panel visibility, draft import/export, disconnect, capability disclosure, metric selection, sample inspection, table disclosure and history export. |

The control audit found and corrected stale workflow repair text, loss of typed
edits when validating unchanged source, incomplete project selectors, repeated
file-selection handling, multiline editing and record updates that could replace
saved content with an unrelated active draft. Grid definition edits invalidate
the confirmed preview. Full project catalogs are bounded and shared by client
and revision rather than fetched on every snapshot poll.

## Browser evidence

On 2026-10-21 the ten rows below were repeated on the shipped bundle, following
the [browser procedure](../studio/tests/README.md#browser-checks). A first run on
the previous bundle found the video metadata panel at zero height at 1024×600.
The viewer fix that the video row describes was made, the bundle was rebuilt, and
all ten rows ran again on the rebuilt bundle; the results below come from that
second run. The bundle was the set that `RESOURCE_SHA256` names:
`index-DPkdXJyt.js`
(`15b3abf8f7656f30c14a5fe77eff749abd022907f8d736962eb7913557930566`),
`index-Daeeztpe.css`
(`8d9d95443d0baecd7bc1b2391fff05559bc6e8fc81daf4ba2e40ed0df51e2a82`) and
`index.html` (`f202c2777cc63909e74bbb18ac5ebc8b1b961a30102325dcef64a70f62db2152`).
The companion served those bytes unchanged. The browser was Mozilla Firefox
158.0, build 20261007201254, headless on Ubuntu 24.04.5 LTS for arm64, driven
through WebDriver BiDi with real pointer and key input. Every row ran at
360×800, 390×844, 768×1024 and 1024×600 viewports, and the motion rows ran again
in a profile that reports reduced motion. No run logged a console message. The
results file holds every measurement and names a screenshot for each row and
viewport. It is kept in the maintainer's private evidence store, outside the
repository and the packages, and its SHA-256 is
`4298bdb94bda062ccd2652648c9b1403e2df6379e4abbe943ac85ba25764797e`. Every row
passed at each viewport where its check applies. The earlier runs
that these rows replace used earlier builds of the bundle and recorded no
browser, browser version or bundle hash.

| Check | Observed result and scope |
|---|---|
| Responsive routes | All eight routes were opened at all four viewports after jobs, uploads, grids, presets and operation reviews existed. Page and main-content horizontal overflow measured zero for every route. Arbitrary user content and other expanded states remain additional combinations. |
| Phone controls | At 360 and 390 pixels wide, all 32 visible inputs, selects and textareas across the eight routes used 16px text and at least 44px height. The rule applies up to 760 pixels wide, so the 768×1024 and 1024×600 viewports keep desktop sizing. Native phone operating-system behavior was not tested. |
| Palette and motion | Palette search held focus two frames after a pointer click or Ctrl+K opened the palette, and took focus 3ms to 12ms after a scripted open. Escape closed the palette and returned focus to its button. The dialog and backdrop reveals were declared at 200ms, and the dialog's finished 201ms to 218ms after opening. With reduced motion enabled, neither animated. A disclosure-closing sample captured 35 or 36 frames with a longest interval of 17.2ms to 17.5ms in headless Firefox's software compositor; this is local interaction evidence, not a frame-rate guarantee. |
| Image previews | Portrait 120×480, landscape 480×120, square 256×256, wide 1024×64 and tall 64×1024 previews filled their frames at every viewport, with at most 0.053 percent aspect-ratio error and 0.017px centering error. |
| Image upload | At every viewport the uploaded portrait came back from the fixture as 120×480 RGBA, 230,400 decoded bytes, with pixels and file bytes identical to the source. |
| Mask alignment | Canvas bounds matched the source preview at every viewport. A pointer press at the canvas center gave a transparent box 32 to 33 pixels a side, such as `(44, 224, 76, 256)`, centered within 0.24 pixels of the press. Center alpha was 0 and corner alpha 255, matching the inverse-alpha mask convention. |
| Video controls | A 2-second 24fps H.264 clip decoded, played, paused, muted and looped in the viewer. Frame stepping stayed disabled, because ComfyUI reports no frame rate for the videos it saves. Details opened the metadata panel at its full 260px at every viewport. The panel no longer shrinks, and in a short window the viewer column scrolls to it; before that fix the panel collapsed to zero height at 1024×600. This does not establish synchronization across codecs or real devices. |
| Generation activity | While a 30-second fixture job ran, the viewer border ran a 2.4-second opacity animation and passed clicks through to the viewer at every viewport. The logo animated at 768 and 1024 pixels wide; the phone layout hides it. Reduced motion stopped both. |
| Editing and saved state | Editing a reviewed two-cell grid removed its confirmation and submit control. Renaming a preset while a different draft was active kept the preset's original values. Enter kept two lines in the prompt field, also after a tab switch. |
| Operation review | The read-only version review and its durable record were opened at every viewport. No host operation was applied. The receipt list shows the oldest receipt first. |

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

## Remaining acceptance

Physical iOS and Android devices, Safari, Chromium, assistive technology,
comprehensive keyboard and contrast coverage, every control/state combination,
full long-session performance and real hardware/service workflows remain
**NOT RUN**. Browser viewport sizing is not physical-device qualification.
Representative inference, Worker acknowledgements, native return and fidelity
acceptance retain their separate gates in
[Studio acceptance](STUDIO_ACCEPTANCE.md) and the
[performance record](STUDIO_PERFORMANCE.md).
