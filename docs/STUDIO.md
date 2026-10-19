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

## Working with workflows

Choose one of the 66 packaged workflows or import a ComfyUI UI workflow/API
prompt. Supported inputs appear as typed controls; connected inputs remain
graph connections. The advanced view exposes all supported primitive fields.
App Mode labels and layout can change without changing binding identity.

Compilation preserves the original workflow and extension metadata. Unknown
widgets, ambiguous bindings and unsupported graph constructs refuse execution
with a repair reason. Nested serialized subgraph interfaces compile into
separate execution identities. Ambiguous edits to shared definitions and
unsupported promoted-widget encodings require an explicitly resolved API
prompt. Studio does not execute imported widget JavaScript.

Preview resolves values and seed policy and freezes the submission. Styles
require explicit positive/negative prompt bindings. LoRA stacks bind only to
existing supported loader slots in a single model chain; an unsupported branch
or missing slot refuses. File names never establish model compatibility.
Execution-time guards still check the actual model and topology. A schema-only
preview is not a memory-admission, topology or identity-gate PASS.

Submitted jobs keep requested and effective settings, the workflow and its
digest, adapter version and lineage. Missing fingerprints and execution facts
are labelled unavailable. Reproduction reconstructs configuration; it does not
promise deterministic pixels across hardware and software versions.

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

Published limits are 4 MiB per JSON request/driver response, 4,096 replay events
and 8 MiB of replay payload, 1,000 console lines/256 KiB, 600 chart samples and
one hour, a 32 MiB thumbnail cache budget, a 64 MiB media display limit, and
1,024 grid combinations. API snapshots return the newest 200 rows per kind;
record storage permits 2,000 editable records per kind. History uses stable
pagination; the browser retains at most 2,000 loaded rows per window. Thumbnail
decoding accepts at most 16 megapixels and 8 MiB per source image, returns at
most 320 pixels per edge and runs at most two decodes concurrently. Input
uploads accept 2 MiB encoded source files and preserve mask alpha after
orientation normalization. Live previews retain at most 16 identity-bound
frames. Grid previews allow at most 32 MiB of complete cell submissions.

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
