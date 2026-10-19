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
source change, together with the regenerated `resource_hashes.py`. The Python
source manifest binds this table to exact resource bytes for installation and
cleanup verification. Asset filenames are content hashed; the bundle has no external
fonts, scripts, CDN URLs or runtime dependencies. Vite's development server is
optional and is not part of production startup. Serve the production bundle from
the Python companion to use its same-origin authenticated API.

The client negotiates schema major version 1, sends `X-DGXM-Schema: 1.0`, and
retains a user-entered bearer token in tab-scoped session storage. Rendered
workflow metadata is text, never HTML. There is no arbitrary shell-command input.
Operation controls come from capability parameter schemas and require the exact
preview digest. Workflow controls come from inspected typed bindings; unresolved
and connected inputs are not silently replaced with strings.

Unsent drafts and layout preferences are stored in browser local storage. Another
tab's draft write pauses automatic saves until the user explicitly chooses a
version. Shared record writes carry revision numbers. Export a draft to retain a
copy independently of browser storage. Changing a draft never edits a job's frozen
submission. A timed-out submission retains its idempotency key and disables fresh
submission until the companion can reconcile its outcome.

The client keeps at most 400 event lines / 128 KiB, requests 40 gallery records per
page from its bounded snapshot, and fetches only selected full media. Each media
fetch has a 64 MiB byte limit. Object URLs are revoked when selection changes or the
viewer unmounts. Polling pauses in hidden tabs and backs off to 30 seconds after
failures. The activity mark reflects accepted/running work, pauses in hidden tabs,
and is static for reduced motion. These are implementation limits, not measured
long-session performance acceptance.

`docs/STUDIO_ACCEPTANCE.md` is the authority for complete-product acceptance and
known gaps. In particular, gallery pagination currently uses metadata cards, not
thumbnail generation; workflow asset inputs are typed fields rather than a mask
editor; telemetry is an inspectable source sample rather than finished charts.
The parameter-grid view validates all combinations and persists an experiment,
but each cell requires its own explicit submission review. Never interpret a
missing control adapter or unmeasured acceptance criterion as a completed feature.

The presentation layer is divided into `views/` for view-owned state,
`components/` for reusable controls and shell regions, and typed records in
`contracts.ts`. `connection.svelte.ts` owns polling and reconciled snapshots;
`submission.svelte.ts` owns reviewed submission identities and uncertain outcomes.
`App.svelte` coordinates unsent drafts and navigation. All TypeScript/Svelte source
modules are below 500 lines. CSS is split by surface under `styles/`.

The [CPU HTTP fixture](tests/README.md) exercises the companion's actual HTTP
contract without a ComfyUI installation, GPU, or management permissions.
