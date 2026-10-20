# Studio interface QA

Date: 2026-10-20. Status: **scoped local validation; draft acceptance remains incomplete**.

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

| Check | Observed result and scope |
|---|---|
| Responsive routes | All eight routes were opened at 360×800, 390×844, 768×1024 and 1024×600 browser viewport sizes. Page and main-content horizontal overflow measured zero in those fixture states. Expanded controls and arbitrary user content remain additional state combinations. |
| Phone controls | Inspected phone inputs and selects used 16px text and 44px height. Native phone operating-system behavior was not tested. |
| Palette and motion | Palette search received focus immediately; its reveal was 200ms. With reduced motion enabled, dialog, backdrop and disclosure animations were absent. A short disclosure-closing sample captured 23 frames with a maximum interval of 16.7ms; this is local interaction evidence, not a general frame-rate guarantee. |
| Image previews | Portrait 120×480, landscape 480×120, square 256×256, wide 1024×64 and tall 64×1024 previews had zero measured aspect-ratio and centering error in the tested layout. |
| Image upload | The uploaded portrait decoded to the exact expected 230,400 RGBA bytes at 120×480. |
| Mask alignment | Canvas bounds matched the source preview. A centered click produced an uploaded mask bounding box of `(43, 224, 76, 257)`, with center alpha 0 and corner alpha 255, matching the inverse-alpha mask convention. |
| Video controls | A 24fps fixture advanced to approximately 0.041666 seconds with Next frame and returned to zero with Previous frame. Mute and loop toggles worked; video metadata expanded on the phone layout. This does not establish synchronization across every codec or real device. |
| Generation activity | The whole viewer border used a 2.4-second opacity animation while generation was active and did not intercept pointer events. Reduced motion disabled both the activity frame and logo animation. |
| Editing and saved state | Editing a reviewed grid invalidated its confirmed submit state. Renaming a preset while a different draft was active retained the preset's original values. Enter preserved two lines in the multiline field. |
| Operation review | A review and durable receipt were inspected. No host operation was applied. |

The earlier [acceptance record](STUDIO_ACCEPTANCE.md) separately records corrupt
request quarantine, invalid draft refusal, corrupt image/video errors, frame
controls before valid metadata, effective-setting reuse and keyboard recovery.
Those observations are not expanded into untested claims here.

## Remaining acceptance

Physical iOS and Android devices, Safari, assistive technology, comprehensive
keyboard and contrast coverage, every control/state combination, full long-session
performance and real hardware/service workflows remain **NOT RUN**. Browser
viewport sizing is not physical-device qualification. Representative inference,
Worker acknowledgements, native return and fidelity acceptance retain their
separate gates in [Studio acceptance](STUDIO_ACCEPTANCE.md) and the
[performance record](STUDIO_PERFORMANCE.md).
