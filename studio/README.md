# Monarch Studio frontend

This directory owns the Svelte 5 / TypeScript presentation layer. The production
companion serves the committed Vite output in `src/dgx_monarch/studio/static/`.
Node is used only to develop, validate and rebuild these assets.

```sh
cd studio
npm ci
npm run check
npm test
npm run build
cd ..
python3 tools/gen_studio_inventory.py
python3 tools/gen_studio_inventory.py --check
```

The lockfile pins the build dependencies. Rebuild and commit `static/` after every
source change, together with the files the inventory tool regenerates:
`src/dgx_monarch/studio/resource_hashes.py`, `src/dgx_monarch/studio/data/*.json`
and `web/js/dgx_monarch_presentation.js`. The Python source manifest binds the
hash table to exact resource bytes for installation and cleanup verification.
Asset filenames are content hashed; the bundle has no external fonts, scripts,
CDN URLs or runtime dependencies. Vite's development server is
optional and is not part of production startup. Serve the production bundle from
the Python companion to use its same-origin authenticated API.

The client negotiates schema major version 1, sends `X-DGXM-Schema: 1.0`, and
keeps the bearer token, which arrives through a sign-in link or the token form,
in tab-scoped session storage. Rendered workflow metadata is text, never HTML.
There is no arbitrary shell-command input.
Operation controls come from capability parameter schemas and require the exact
preview digest. Workflow controls come from inspected typed bindings; unresolved
and connected inputs are not silently replaced with strings.

Unsent drafts and layout preferences are stored in browser local storage. Another
tab's draft write pauses automatic saves until the user explicitly chooses a
version. Shared record writes carry revision numbers. Export a draft to retain a
copy independently of browser storage. Changing a draft never edits a job's frozen
submission. A timed-out submission retains its idempotency key and disables fresh
submission until the companion can reconcile its outcome.

The client keeps at most 400 event lines / 128 KiB. Gallery and queue history use
pages of up to 100 records. Saved record windows and selector catalogs are
bounded to 2,000 records and 8 MiB of serialized JSON; oversized catalogs give an
explicit error and direct users to paginated library browsing. Snapshot and job
history rows are summaries; full job and operation provenance loads on demand.
The gallery mounts
only visible rows, requests bounded 320-pixel thumbnails, and caps each thumbnail
response at 512 KiB. A thumbnail downloads only once it comes within 200 pixels
of the visible area, so a long Queue page or outputs strip does not start every
preview at once. Full media loads only for selection or explicit download and
has a 64 MiB byte limit. Object URLs are revoked after use. Polling pauses in hidden
tabs and backs off to 30 seconds after failures. Metric history retains 240 points
per series and at most 256 series, preserving source timestamps and missing-value
gaps. These are implementation limits, not measured long-session acceptance.

The Assets tab uploads PNG, JPEG or WebP inputs and painted PNG masks through the
companion's 2 MiB upload contract. Binding selection is explicit; the returned
server filename is the actual workflow value. Painted mask pixels export with
transparent alpha, matching Comfy LoadImage's inverse-alpha mask output. Clear
editor, replace source, clear mask, and clear bound input are separate actions.

Gallery filters cover provenance, operation, model, date, status, project,
media kind and dimensions. Dimension filters match only records that carry a
size, such as uploads: ComfyUI does not report the width or height of the
outputs it saves. Images support shared zoom and wipe comparison;
videos support leader/follower time and playback-rate synchronization, duration
clamping, and frame stepping when frame rate is recorded. This does not promise
frame alignment between different encodings or durations. Unsupported thumbnails
remain labelled placeholders.

Grids validate complete typed combinations in the backend, expose each complete
cell preview on demand, and require confirmation before grouped dispatch through
the existing Comfy queue. Durable outcomes, linked retries of selected failed or
cancelled cells, saved-run resume, output comparison and manifest export remain
available after dispatch. Fixed and incrementing grid seeds are explicit; grid
previews never silently randomize seeds. History views retain active jobs while
browsing older records.

`docs/STUDIO_ACCEPTANCE.md` remains the authority for complete-product acceptance,
including measured performance, real-driver behavior and hardware qualification.
Neither a passing CPU test nor a rendered control establishes those outcomes.

The presentation layer is divided into `views/` for view-owned state,
`components/` for reusable controls and shell regions, and typed records in
`contracts.ts`. `connection.svelte.ts` owns polling and reconciled snapshots;
`submission.svelte.ts` owns reviewed submission identities and uncertain outcomes.
`App.svelte` coordinates navigation and these modules: `draftAutosave.svelte.ts`
owns draft autosave and tab conflicts, `inspectionGuard.ts` orders workflow
inspections and job reviews, `assetEditorState.svelte.ts` keeps the asset
editor across tab and view switches, and `operationOpener.svelte.ts` holds the
operation Operations opens next. `jobFailure.ts` reads a failed job's recorded
error for the status line and Queue. `workflowTraits.ts` reads what a graph
loads, makes and saves for Create's Operation select and Output line, and
`frameCounts.ts` finds the clip frame counts that the Duration control sets
together. `navigation.ts` defines the rail and the Utilities tabs: Grids, Fleet
and Operations keep their route ids, and the `utilities` alias opens the tab
used last. `hostHealth.ts` reads the telemetry panel's per-host overview and
`chartChoice.ts` its Host and Metric choices; `fleetSetup.ts` decides what Set
up Fleet loads. All TypeScript/Svelte source modules are below 500 lines. CSS
is split by surface under `styles/`.

The [CPU HTTP fixture](tests/README.md) exercises the companion's actual HTTP
contract without a ComfyUI installation, GPU, or management permissions.
