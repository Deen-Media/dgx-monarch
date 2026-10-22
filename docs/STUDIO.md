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

The [interface QA record](STUDIO_INTERFACE_QA.md) distinguishes control-source
inspection from browser checks, including phone and tablet viewport layouts.
Physical-device and comprehensive accessibility acceptance remain open.

## Install and start

Follow [the existing installation procedure](INSTALL.md) with the `studio`
extra added to the selected source installation. It adds `aiohttp` and Pillow; the
normal Torch, Monarch and xFuser constraints still apply. Node is needed only
when rebuilding the frontend, not when installing or running a built checkout,
wheel or source archive.

```sh
dgxm-studio --check
dgxm-studio --driver http://127.0.0.1:8188
```

`dgxm-studio --check` compares every packaged Studio file with its recorded
SHA-256 and reports whether the studio extra is installed; it needs neither the
extra nor a driver.

Open the printed loopback URL. When standard output is a terminal, the companion
prints the owner token there. Otherwise, for example when output goes to a file,
a pipe or a service journal, it writes the token to `owner-token` (mode 0600) in
the state directory and prints only that path. `--token-file /absolute/path`
writes the token to another file instead, whose directory must belong to you
with mode 0700. The companion removes the file it wrote when it exits. Treat the
token as a local password. It changes on each launch, and it can land in
terminal scrollback, the token file and the browser's session storage.

Studio binds only to a loopback IP, verifies Host and Origin, requires bearer
authentication for every API request and disables cross-origin access. It does
not open a public port, configure a reverse proxy or install a service. The Host
check accepts only the literal bind address and port, so `localhost` is refused.
For a remote browser, the one working shape is an SSH tunnel to the same port,
`ssh -L <port>:127.0.0.1:<port> <host>`, browsed at `http://127.0.0.1:<port>`.
TLS termination, Host rewriting and a different local port are not supported;
the companion refuses their Host or Origin. ComfyUI itself must stay on a
trusted local connection or behind HTTPS with a valid server certificate. The
companion sends no credentials to the driver, so a front end that requires
client credentials refuses it.
Each state directory is bound to one canonical driver origin. A different
driver requires a separate `--state-dir`; this prevents old artifact names from
being served by an unrelated driver. Existing unbound draft databases adopt the
selected origin on their first upgraded start, so use their original driver.

Pass an existing `--config /absolute/path/cluster.toml` to use the supported
cluster reporting actions; a relative `--config` resolves against the launch
directory at startup. Management actions additionally require
`--allow-management` and an explicit review/apply confirmation in the browser.
Without the flag only these read-only actions run: version, status, doctor, reap
preview, gate inspect, gate repair preview, top replay, diagnostics and
diagnostics capabilities. A diagnostics request that names an output path writes
a file, so it counts as a management action.

Without `--allow-management`, the owner token controls jobs, outputs, saved
records and the read-only actions above. With it, setup and init run a browser-supplied interpreter on every
host and update runs a browser-named checkout, so the token is code execution as
the worker account on every host;
[THREAT_MODEL.md](THREAT_MODEL.md#browser-to-studio-companion) records its scope
and where it can land. Browser JSON never becomes command text. Read the exact
operation description before applying it. Worker lifecycle, client-owned actor
recycle, stock driver `/free`, and ComfyUI driver lifecycle are separate
actions. Backend idle, consent, refusal and recovery rules remain authoritative.

Studio runs each management command with `python -P -s -B`, so a dgx_monarch
directory in the launch directory cannot replace the installed package, and it
records the version the command itself printed. A doctor repair receipt path
must be absolute. An exit code of 75 (an unanswered critical probe) records
unknown, not failed. A 4xx refusal from the driver's consent or recycle route
records failed with the driver's short reason; a 5xx or a lost response records
unknown. Consent apply reads the driver's pending consent requests at review.
Leave key and id empty to list them; with one pending request the review selects
it. Studio offers accept only; dismiss, revoke and the automatic rescue setting
stay in the ComfyUI panel. A Sol install review with no plan token reuses the
staging of a live review of the same plan. Each review first removes, on every
host, staging that only expired or replaced Studio reviews used; staging an
apply used and staging for a token you supplied stay.

The optional `--driver-service` setting names an already installed owner-managed
ComfyUI systemd user unit. It neither installs a unit nor grants permissions.
The companion refuses unrecognized driver ownership and does not interpret
Worker-service controls as driver controls. The selected unit must serve the
`--driver` origin: Studio reads its ComfyUI `--listen`, `--port` and TLS options
(ComfyUI's default is 127.0.0.1 port 8188) and refuses a unit that serves
another address, port or scheme. A wildcard listen address matches only a
loopback or local origin. Management controls expose typed review plans and
retained receipts.

Update needs both `--config` and `--driver-service`, so in Studio it needs
ComfyUI to run from an existing systemd user unit rather than the foreground
launcher; `dgxm update` on the command line has no such requirement. Without
either setting, Studio lists update as unavailable and names the missing one. It
refuses a checkout other than the one the driver unit's ComfyUI imports, the
target of its `custom_nodes/dgx-monarch`, and refuses when the companion's own
dgx_monarch source differs from that checkout. The default verified mode runs
the existing verified transaction. Legacy mode runs the original in-place
update, and its review first refuses a cluster whose Workers run from release
slots. Both modes run in a frozen controller with the driver unit's interpreter.
Studio stops the driver unit first and starts it again only after the update
succeeds or the backend refuses before it changes anything; any other outcome
leaves the driver stopped and keeps the controller and the backend's reason for
recovery.

The generated inventory maps all 101 CLI command and option rows to typed
actions, parameters, selectors, launch context or presentation behavior.
Diagnostics capture and capabilities, last-gate inspection, reap grace,
Worker synchronization choices and recording/replay controls have explicit
mappings. Inventory validation rejects missing mappings and mismatched
advertised defaults or choices. Panel handlers and TUI bindings meet the same
closed check: each maps to a Studio operation, route or equivalent control, or
carries a written waiver. Node schema and node input rows record source
coverage only, because Studio reads them from the driver's live object_info. Live availability still depends on the
selected configuration, permissions and backend guards.

## Workspace controls and media

Use the navigation rail or Ctrl/Cmd+K to open a view. The command palette takes
keyboard focus immediately; Tab chooses a result and Enter opens it. Escape
closes dialogs or leaves presentation mode. Project selectors include the full
bounded project catalog, including projects outside the recent snapshot.

Phone layouts use a scrolling bottom navigation rail, larger touch controls and
stacked panels. Expanded viewer details remain reachable without clipping the
output area. Dialogs and disclosures use short transitions; generation adds a
subtle activity border around the viewer. Reduced-motion preferences disable
these animations and the activity logo.

Image previews retain their source proportions across portrait, landscape and
extreme aspect ratios. The mask canvas follows the displayed source bounds, so
painted positions map to the uploaded mask. An uploaded input or mask can be bound to
image upload widgets that read the input store. It joins that widget's choices
for the request that names it, at inspection and review; any other value outside
the driver's list is still refused. Studio refuses an empty mask before upload. Video frame controls wait for valid
metadata; native playback, mute and loop controls remain available. The
[interface QA record](STUDIO_INTERFACE_QA.md) describes the tested fixtures and
browser viewports without extending those results to every device or codec.

An output's `created_at` is the time the driver's history reports it finished;
when history carries no completion time it is the time the companion imported
it. `observed_at` is always the import time, and `created_at_source` says which
applied. Width and height filters apply to uploaded inputs and masks, because
ComfyUI does not report the size of generated outputs; the gallery shows those
filters only when a loaded output carries its size. Intermediate previews have
their own status option, and uploads have their own media kinds.

## Working with workflows

Choose one of the 71 packaged workflows or import a ComfyUI UI workflow/API
prompt. Supported inputs appear as typed controls; connected inputs remain
graph connections. The advanced view exposes all supported primitive fields.
App Mode labels and layout can change without changing binding identity.
Layout edits also update the workflow repair text. Validating unchanged source
preserves edits made in typed controls; intentional JSON changes use the edited
workflow's defaults. Changing a dynamic choice, such as the codec or encoding
mode of SaveVideo, inspects the workflow again with the draft's values: the
controls for the new choice's inputs appear, those of the old choice leave the
draft, and review accepts edits to the new inputs. Inside a subgraph, Studio
lists only the inputs of the saved choice. Multiline fields retain line breaks, and import controls
allow selecting the same file again after a success or error.

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
An invalid LoRA strength, a removed LoRA or an out-of-range inserted loader is
refused at review with `lora_binding`, naming the slot and input.
The LoRA list comes from the driver catalog Studio reads at connect. Studio
reads it again when the driver connects again or the companion restarts, and
once more after a failed read while the driver stays connected. Refresh list on
the LoRA tab reads it on demand, for example after adding LoRA files.
File names never establish model compatibility.
Execution-time guards still check the actual model and topology. A schema-only
preview is not a memory-admission, topology or identity-gate PASS.

Submitted jobs keep requested and effective settings, the workflow and its
digest, adapter version and lineage. Missing fingerprints and execution facts
are labelled unavailable. Reproduction reconstructs configuration; it does not
promise deterministic pixels across hardware and software versions.
Reuse loads the recorded executable prompt with its resolved seeds as a fixed
draft. Already expanded styles and inserted LoRAs remain in that prompt and are
not applied twice; original workflow metadata is retained separately.

A review binds to the definitions of the node classes its prompt uses, without
their file lists, so adding a model, LoRA or input file does not void it.
Submission still checks each selected file or option against the live driver and
refuses a missing one with `choice_unavailable`. A confirmed retry does not need
a fresh review: Studio reads the driver schema again, checks the frozen
submission against it the same way and keeps the frozen request unchanged. It
records the driver version and runtime the retry ran against in `retry_runtime`,
which a grid's full record also shows per cell, and comparisons of a retry read
those recorded versions.

Parameter grids use typed workflow inputs or named style stacks as axes. A
style-axis value is an ordered list of saved style IDs; an empty list selects
no styles. Each cell freezes the selected definitions and revisions before
expanding its prompt. Later library edits do not change that submission or its
retry. Fixed seed mode preserves draft seeds; incrementing mode requires an
explicit seed binding and base value. Invalid cells remain visible and block
dispatch until corrected or explicitly excluded. Each accepted cell has its
own immutable submission and outcome, with selective failed/cancelled retries
through the existing ComfyUI queue.
Changing a grid definition invalidates its reviewed preview and confirmation;
validate the edited cells again before submission. Grids keeps an identity of
the draft and grid fields each preview came from. When Grids opens again, or the
page reloads, it restores a saved preview only if the draft and fields still
match; otherwise it drops the preview and asks for validation again. A dispatch
whose outcome is unknown keeps its preview and request ID so it can be
reconciled, and Grids says when that review came from an earlier draft.

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
`~/.local/state/dgx-monarch/studio`. An empty `XDG_STATE_HOME` counts as unset.
The companion refuses a relative `XDG_STATE_HOME` or `--state-dir`, so launching
from another directory never opens a second database. Use `--state-dir` to
choose another private directory. Directories must be owner-only (0700); the
SQLite database uses 0600, WAL and full synchronous transactions. A process lock
permits one companion per state directory. A schema mismatch refuses startup;
this first schema has no older migration. A state directory, lock or database
refusal exits with one line. Preserve the directory to preserve shared records.

Projects, presets, styles, sessions, imported workflows and experiments use
optimistic revisions. A stale save returns a conflict, retaining the newer
record. Browser-local drafts and layout survive connection loss. If a saved
draft cannot be restored, Studio pauses automatic saves and offers the stored
text for export before anything replaces it. If the browser refuses a write,
Studio says so and offers an export of the current draft. A second tab raises a
conflict only when it saves a different draft. Exported records retain unknown
metadata. Durable jobs and operation receipts are not silently deleted. Expired
previews can be recreated; review a new preview after capabilities change. The
companion deletes expired job and grid previews at startup and on later preview
writes, at most once every five minutes, and keeps the preview each dispatched
grid was reviewed from. A submit of a deleted preview is refused as expired.
Each preview and each job stores its workflow once, and reads return the same
requested, effective and workflow fields. A stored workflow stays while any
preview or job names it.

Renaming a saved preset, session or workflow, or changing its membership,
preserves its saved content. Save a new record to capture the current workspace.

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
are limited to 4 MiB. Exports leave out each artifact's search text, which the companion
derives from the job that made it. One export holds at most 2,000 rows per
collection, the most one import accepts, and refuses beyond that or beyond
4 MiB, naming the collection to split. Import keeps a reference only when it
needs repair, once, with its reason.

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
The companion issues each prompt identity (a UUID) before it submits, so a lost
or malformed acknowledgement reconciles through that prompt's exact history.

A job the driver no longer holds is settled once, from exact evidence: it is
missing from a successful queue snapshot and its own history lookup answers with
no entry. A queued job whose cancellation the driver acknowledged then becomes
cancelled; any other such job becomes unknown with the reason
`absent_from_driver` and is no longer looked up. The recent-history window is
never used as proof. The adapter halves that window when a response exceeds the
4 MiB budget and doubles it again after three pages in a row under half that
budget; a doubled window that overflows waits twice as long before the next try.
It also queries missing known prompts individually. A history failure shows a
warning while successful queue observations remain available. When the queue
response passes its budget, the driver stays connected, the queue shows
`over_budget`, and history reconciliation continues without settling absent
jobs. A driver row that cannot be read is skipped and counted in the driver
state, and a non-finite number in a driver response reads as null.

A 4xx refusal of a prompt marks its job failed with the failing nodes (node,
class, input and type only); a 5xx, a lost response or an unreadable
acknowledgement stays unknown. A prompt the driver queued with node errors stays
queued with a warning. The driver state carries `code` and `since` while it is
unavailable, `queue.state`, the current `history.limit`, a count of skipped
driver rows, and the event watcher's state while it waits to reconnect.
Background failures are logged and retried with a growing pause of up to 30
seconds.

Events have durable cursors and server timestamps. A replay gap requires a new
snapshot. The snapshot is authoritative. The event feed is a notification log:
it names created jobs, operations and artifacts and the fields an update
changed. A field too large for an event is listed under `omitted_fields`; read
it from the snapshot or the record. Terminal job state never regresses; a late
submission response cannot replace a running state with queued. Restarted
in-flight operations become unknown and are never automatically re-executed.
Queue deletion records a cancellation request. It settles as cancelled only when
the driver acknowledged it while the job was queued and the prompt has left both
the queue and its history. Running-job cancellation uses the driver's atomic
prompt-specific cancellation route when available. On a driver without that
route, a queued job falls back to queue deletion under the same cancellation
intent; a running job is refused and its intent and key stay unspent. The
adapter never falls back to a global interrupt. Each job retains one durable
cancellation intent, including after a lost acknowledgement or companion
restart. Repeating that intent cannot dispatch it again. A reviewed management
operation is single-use; its original request key reconciles the receipt. Fresh
execution requires a fresh review.

The capabilities response publishes the companion's request, event, console,
chart, thumbnail, page, preview, grid and `/object_info` limits in its `limits`
object. A JSON request and a driver response are limited to 4 MiB, except the
driver's `/object_info` schema, which has its own 16 MiB budget
(`object_info_bytes`); a refusal names the endpoint and its budget. The
companion's browser event feed replays at most 4,096 events and 8 MiB of event
payload (`event_count` and `event_bytes`), and each event is limited to 64 KiB
(`event_max_bytes`). Other limits are 400 browser
console lines/128 KiB, 240 chart samples and one hour, a 32 MiB thumbnail cache
budget, a 64 MiB media display limit, and 1,024 grid combinations. Every
companion error, including the 4 MiB request limit and an unexpected failure,
returns JSON `{"error": {"code", "message"}}`. Workflow and draft import read
files up to twice `request_bytes` and refuse an encoded request above
`request_bytes`.

Snapshots return compact windows of up to 200 rows per kind under a shared byte
budget, prioritizing unresolved jobs and reporting omitted counts. Job history
returns summaries; full job provenance and operation receipts load by identity
when opened or exported. Artifact lists omit only the derived search index.
Record and history pages are limited to 8 MiB; cursors advance by actual
returned rows. Record storage permits 2,000 editable records per kind. The
browser bounds loaded record windows and catalogs to 8 MiB and 2,000 rows, and
the companion refuses a save or a library import that would take a kind's
catalog past that 8 MiB window before it commits. A new record keeps one identity per form until its
save is confirmed, so a retried save that already committed reads as saved.

Studio decodes at most 16 megapixels (4096 x 4096 pixels) per image, for input
uploads and thumbnails alike, and checks that from the file header before it
decodes. Input uploads accept PNG, JPEG or WebP files of at most 2 MiB, keep
mask alpha, and store an orientation-normalized RGBA PNG; a larger image is
refused with a request to downscale it. Thumbnails return at most 320 pixels per
edge and run at most two decodes at once. A 65 MiB ceiling on the normalized
upload and on a thumbnail's source file is only a backstop that no 8-bit image
within the pixel limit reaches. Live previews retain at most 16 identity-bound
frames. Expanded style text and complete submission previews each have a 4 MiB
JSON budget.

Grid expansion is bounded before copying parameters. All cell previews and the
grid review commit together under a combined 32 MiB budget; failure leaves no
partial previews. Previews and jobs store each workflow once, so the budget
counts each distinct workflow once and each cell for its compiled prompt and
settings. The same budget bounds what submitting the grid writes: a job for each
cell and the grid run. Grids whose compiled prompt passes about 30 KB reach that
byte budget before 1,024 cells. Excluded cells are not expanded, so they have no
preview to inspect. Grid history and polling use compact views
under an 8 MiB budget, with full records fetched only for explicit export.
Random seeds honor the installed input bounds and JavaScript's exact integer
range so the generated seed can be reused without rounding.
Definitive pre-acceptance refusals release pending review controls. Incomplete
or malformed acknowledgements, authentication failures and transport timeouts
retain the exact request identity for reconciliation. Corrupt optional draft or
layout storage cannot clear that protection.
Unreadable saved operation, grid or import identities block new requests until
the user checks durable records and explicitly clears the unreadable entry.
Storage access failures leave authentication available in memory, but requests
requiring a saved recovery identity remain blocked. Malformed imported draft
controls are rejected before replacing the working draft.
A 401 while polling returns to the token form, because a restarted companion
issues a new token. A workflow inspection or job review is discarded, with a
notice, when the workflow, its values or the connection change while it runs;
edits to other draft controls do not discard an inspection.

Snapshot refreshes preserve the event replay cursor so bursts spanning several
pages reach the console. Evicted replay gaps are labelled explicitly. Media
decoding failures retain a download action, and frame stepping waits for valid
duration and frame-rate metadata.

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

After each bundle build, regenerate `src/dgx_monarch/studio/resource_hashes.py`
and `web/js/dgx_monarch_presentation.js` with the inventory tool. Its literal SHA-256 table is covered by the Python source
manifest. Setup and update validation check the listed resource bytes;
source-sync cleanup permits those exact files only after verifying the reviewed
source digest. Extra files, changed assets and symlinks remain refusal cases.

## Troubleshooting

If the page is unavailable, start or recover the companion itself. A browser
cannot restart its entirely unavailable host. If only the driver is unavailable,
the page and durable records remain accessible. A stale telemetry timestamp is
not a live observation. If a schema changes, refresh capabilities and re-preview
the job; if a record revision conflicts, reload and explicitly reapply changes.
For startup, Host, Origin and token refusals, see
[TROUBLESHOOTING #123](TROUBLESHOOTING.md#123-studio-refuses-to-start-or-refuses-the-browsers-host-origin-or-token).

Do not delete state, resubmit uncertain jobs, bypass guards or restart Workers
as a connection-repair shortcut. An operation receipt reports only the boundary
that actually completed. A CLI exit code or a mocked test is not render fidelity
or hardware acceptance.
