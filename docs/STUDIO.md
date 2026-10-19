# DGX Monarch Studio (draft)

Studio is an optional Svelte interface served by a separate Python companion.
The companion uses the existing ComfyUI queue and Monarch nodes. It does not
load models, own a mesh, schedule GPU work, or change ComfyUI source. The CLI,
TUI and ComfyUI panel remain available independently.

This implementation is a draft against v1.2.0 source commit
`5604ac77799f1f3e1a077a2f841dfce59fbcb892`. The full agreed product contract is
[STUDIO_SPEC.md](STUDIO_SPEC.md); the [acceptance record](STUDIO_ACCEPTANCE.md)
identifies incomplete features and unperformed checks. An inventory row is
evidence of source coverage, not proof of a working Studio operation.
The [measurement record](STUDIO_PERFORMANCE.md) separates packaged asset sizes
from unperformed host and browser performance acceptance.

## Install and start

Follow [the existing installation procedure](INSTALL.md) with the `studio`
extra added to the selected source installation. It adds only `aiohttp`; the
normal Torch, Monarch and xFuser constraints still apply. Node is needed only
when rebuilding the frontend, not when installing or running a built checkout,
wheel or source archive.

```sh
dgxm-studio --check
dgxm-studio --driver http://127.0.0.1:8188
```

Open the printed loopback URL and enter the printed owner token. Treat the
token as a local password. It changes on each launch. Studio binds only to a
loopback IP, verifies Host and Origin, requires bearer authentication for every
API request and disables cross-origin access. It does not open a public port,
configure a reverse proxy or install a service. For a remote browser, use an
operator-managed encrypted tunnel retaining the configured Host. ComfyUI
itself must remain on a trusted local connection or authenticated HTTPS.

Pass an existing `--config /absolute/path/cluster.toml` to use the supported
cluster reporting actions. Management actions additionally require
`--allow-management` and an explicit review/apply confirmation in the browser.
There is no arbitrary shell, SSH, Python or URL operation API. Read the exact
operation description before applying it. Worker lifecycle, client-owned actor
recycle, stock driver `/free`, and ComfyUI driver lifecycle are separate actions.
Backend idle, consent, refusal and recovery rules remain authoritative.

The optional `--driver-service` setting names an already installed owner-managed
ComfyUI systemd user unit. It neither installs a unit nor grants permissions.
The companion refuses unrecognized driver ownership and does not interpret
Worker-service controls as driver controls. Management controls expose typed
review plans and retained receipts. Update executes the existing verified
transaction in a frozen controller using the reviewed driver interpreter;
uncertain outcomes suppress automatic restoration and retain recovery data.

The generated inventory maps all 101 CLI command and option rows to typed
actions, parameters, selectors, launch context or presentation behavior.
Diagnostics capture and capabilities, last-gate inspection, reap grace,
Worker synchronization choices and recording/replay controls have explicit
mappings. Inventory validation rejects missing mappings and mismatched
advertised defaults or choices. Live availability still depends on the
selected configuration, permissions and backend guards.

## Working with workflows

Choose one of the 66 packaged workflows or import a ComfyUI UI workflow/API
prompt. Supported inputs appear as typed controls; connected inputs remain
graph connections. The advanced view exposes all supported primitive fields.
App Mode labels and layout can change without changing binding identity.

Compilation preserves the original workflow and extension metadata. Unknown
widgets, ambiguous bindings and unsupported graph constructs refuse execution
with a repair reason. Repeated nested subgraphs have separate execution and
input-binding identities, so changing one instance leaves the others intact.
Bindings identify the root graph, host-instance path, node and typed input;
legacy definition bindings remain aliases only when they identify one instance.
Nested promoted scalar widgets follow explicit serialized mappings. Ambiguous,
missing or unsupported mappings require repair. Studio does not execute
imported widget JavaScript.

Preview resolves values and seed policy and freezes the submission. Styles
require explicit positive/negative prompt bindings. LoRA stacks fill and extend
a single supported loader chain. For a workflow without slots, select the
loader and exact connected consumers that should receive the new chain.
Core loaders require `MODEL` sockets; `DGXMonarchLoraLoader` requires
`DGXM_MODEL`. A loader with text-encoder support also requires the corresponding
`CLIP` connections. Other branches keep their original inputs. Preview freezes
the inserted node identities, connection boundaries, asset names and strengths.
Multiple ambiguous chains or incompatible typed connections refuse execution.
File names never establish model compatibility.
Execution-time guards still check the actual model and topology. A schema-only
preview is not a memory-admission, topology or identity-gate PASS.

Submitted jobs keep requested and effective settings, the workflow and its
digest, adapter version and lineage. Missing fingerprints and execution facts
are labelled unavailable. Reproduction reconstructs configuration; it does not
promise deterministic pixels across hardware and software versions.

Parameter grids use typed workflow inputs or named style stacks as axes. A
style-axis value is an ordered list of saved style IDs; an empty list selects
no styles. Each cell freezes the selected definitions and revisions before
expanding its prompt. Later library edits do not change that submission or its
retry. Fixed seed mode preserves draft seeds; incrementing mode requires an
explicit seed binding and base value. Invalid cells remain visible and block
dispatch until corrected or explicitly excluded. Each accepted cell has its
own immutable submission and outcome, with selective failed/cancelled retries
through the existing ComfyUI queue.

## Runtime evidence

Expand a queue entry's Monarch evidence view to inspect per-render outcomes
or exact runtime events. These records use the Comfy prompt ID, executing node
ID and list index, plus a Monarch render ID and optional Fleet item index.
Capture requires ComfyUI's `get_executing_context()` boundary. If that API or an
executing context is absent, Studio leaves the facts unavailable. It never
assigns the global last-render or last-Fleet snapshot to a particular job.

A dispatch observation establishes that the driver's guards and dispatch
returned; it does not establish Worker acceptance or successful output.
Validated Worker results supply their actual setup generation, topology,
software versions and available process-lifetime source manifest. Dispatch
attention is labelled as the driver's selection; Worker results report both
selected and effective attention where available. Model/LoRA identities reuse
the existing parity-validated start/middle/end fingerprints, not full-file
content hashes. Latent statistics retain the runtime's full-byte digest.

Output validation, resource retirement and graph completion are separate facts.
A render or Fleet item is complete only after its output is validated and its
existing resource retirement succeeds. Native owner ACKs are recorded only
after the existing validator checks the response count, generation/token and
exactly one owner; they are not inferred from a returned tensor or graph
success. The journal stores a SHA-256 identity for an owner token, never the
raw capability. The separate opt-in resident-adoption evidence is not copied
into this journal. Unknown outcomes remain unknown. Late verified facts may
enrich an ended job without changing its terminal graph state or submitted
settings.

The companion polls the Monarch-owned `/dgxm/execution-events` route and grants
a 30-second observer lease. Without a lease, hooks skip context lookup,
fingerprinting and event serialization; no driver journal is written to disk.
Requests already marked for observation can still finish collecting their
Worker identity after the companion stops. Ordinary callback and context
handling remain, so this is not a claim of zero runtime overhead.

Driver replay retains at most 8,192 events and 8 MiB of encoded event payload.
Event data is limited to 32 KiB plus sequencing metadata; a page budgets 2 MiB
for event data and at most 128 rows, with additional sequencing and response
metadata. These are payload
bounds, not measurements of Python process memory. Producer restarts, expired
leases, eviction and omitted events create explicit coverage gaps. Capture is
not retroactive, and reconnecting cannot reconstruct a missing native ACK.

The owner-private companion database retains received events and complete
per-render records. Deduplication and producer watermarks survive replay-window
and display-summary eviction. Each job displays at most 32 render summaries,
limited to 4 KiB each, with complete outcome counts and a count of omitted
summaries. Full outcomes and events remain available through the evidence view
and authenticated `/api/v1/jobs/{id}/runtime` pages. Durable history is not
silently pruned; its long-session disk growth remains a measurement gate.
Runtime records can contain Worker hostnames and model filenames. The driver's
existing trusted-network/authentication boundary and the companion's owner
boundary protect those details.

## State, reconciliation and limits

Owner state defaults to `$XDG_STATE_HOME/dgx-monarch/studio`, or
`~/.local/state/dgx-monarch/studio`. Use `--state-dir` to choose another private
directory. Directories must be owner-only (0700); the SQLite database uses 0600,
WAL and full synchronous transactions. A process lock permits one companion per
state directory. A schema mismatch refuses startup; this first schema has no
older migration. Preserve the directory to preserve shared records.

Projects, presets, styles, sessions, imported workflows and experiments use
optimistic revisions. A stale save returns a conflict, retaining the newer
record. Browser-local drafts and layout survive connection loss. Exported
records retain unknown metadata. Durable jobs and operation receipts are not
silently deleted. Expired previews can be recreated; review a new preview after
capabilities change.

Saved records support explicit project memberships and project filtering;
artifact memberships have their own revision checks. A record can be shared
across projects without moving its source assets. Portable library export
includes records, extension metadata and source references, for the entire
library or one project. Import first creates a review plan. Preserve-ID mode
requires an explicit keep/replace decision for conflicts; copy mode creates
new record IDs and remaps project/style references. Applying the reviewed
digest checks revisions again and commits atomically with an idempotency key.
Missing media or job references remain on a repair list. Import does not copy
media, dispatch jobs or recreate missing source provenance. Library documents
are limited to 4 MiB.

Save a durable grid as a named experiment to retain its definition, observed
cell outcomes and source references in the project library. Loading the
definition replaces the editable draft and axes and requires a fresh preview;
resuming a local run reads its existing jobs. An imported experiment can retain
its frozen definition while its original run is unavailable. Saving or importing
an experiment never submits those historical jobs again.

A request has a caller idempotency key and a server-issued identity. Repeat
requests must have identical bodies. After a lost response, reconcile by the
key; never invent a new key and resubmit an uncertain job. The companion
reconciles its submissions and other interfaces' jobs from driver queue and
history snapshots. External jobs contain only provenance the driver supplied.
Driver history eviction can leave an old unknown outcome unresolved.

Events have durable cursors and server timestamps. A replay gap requires a new
snapshot. Terminal job state never regresses; a late submission response cannot
replace a running state with queued. Restarted in-flight operations become
unknown and are never automatically re-executed. Queue deletion records a
cancellation request; without a definitive backend terminal event it does not
claim cancellation completed. Running-job cancellation uses the driver's
atomic prompt-specific cancellation route when available. Older drivers refuse
that request; the adapter never falls back to a global interrupt.
Each job retains one durable cancellation intent, including after a lost
acknowledgement or companion restart. Repeating that intent cannot dispatch it
again. A reviewed management operation is single-use; its original request key
reconciles the receipt. Fresh execution requires a fresh review.

Published limits are 4 MiB per JSON request/driver response, 4,096 replay events
and 8 MiB of replay payload, 400 browser console lines/128 KiB, 240 chart samples and
one hour, a 32 MiB thumbnail cache budget, a 64 MiB media display limit, and
1,024 grid combinations. Snapshots return compact windows of up to 200 rows per
kind under a shared byte budget, prioritizing unresolved jobs and reporting
omitted counts. Job history returns summaries; full job provenance and operation
receipts load by identity when opened or exported. Artifact lists omit only the
derived search index. Record and history pages are limited to 8 MiB; cursors
advance by actual returned rows. Record storage permits 2,000 editable records
per kind. The browser bounds loaded record windows and catalogs to 8 MiB and
2,000 rows. Thumbnail
decoding accepts at most 16 megapixels and 8 MiB per source image, returns at
most 320 pixels per edge and runs at most two decodes concurrently. Input
uploads accept 2 MiB encoded source files and preserve mask alpha after
orientation normalization. Live previews retain at most 16 identity-bound
frames. Grid previews allow at most 32 MiB of complete cell submissions.
Random seeds honor the installed input bounds and JavaScript's exact integer
range so the generated seed can be reused without rounding.

## Development

```sh
cd studio
npm ci
npm run check
npm test
npm run build
cd ..
python tools/gen_studio_inventory.py
python tools/gen_studio_inventory.py --check
python -m pytest -q tests/test_studio_inventory.py tests/test_studio_workflows.py \
  tests/test_studio_preview.py tests/test_studio_backend.py
python -m build
```

The checked-in Vite output is under `src/dgx_monarch/studio/static`, outside
ComfyUI's `WEB_DIRECTORY`. Both package formats include the compiled assets,
inventory and workflow catalog. The source archive also includes the complete
frontend source and lockfile. Builds do not download assets at runtime or use a
CDN. Release acceptance requires the full specification, not just these checks.

After each bundle build, regenerate `studio/resource_hashes.py` with the
inventory tool. Its literal SHA-256 table is covered by the Python source
manifest. Setup and update validation check the listed resource bytes;
source-sync cleanup permits those exact files only after verifying the reviewed
source digest. Extra files, changed assets and symlinks remain refusal cases.

## Troubleshooting

If the page is unavailable, start or recover the companion itself. A browser
cannot restart its entirely unavailable host. If only the driver is unavailable,
the page and durable records remain accessible. A stale telemetry timestamp is
not a live observation. If a schema changes, refresh capabilities and re-preview
the job; if a record revision conflicts, reload and explicitly reapply changes.

Do not delete state, resubmit uncertain jobs, bypass guards or restart Workers
as a connection-repair shortcut. An operation receipt reports only the boundary
that actually completed. A CLI exit code or a mocked test is not render fidelity
or hardware acceptance.
