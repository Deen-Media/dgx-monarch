# Studio acceptance record

Status: **incomplete draft**, not release acceptance. Baseline:
`5604ac77799f1f3e1a077a2f841dfce59fbcb892` (v1.2.0).

The original [specification](STUDIO_SPEC.md) remains the complete deliverable.
Unimplemented work below is not a proposed reduction of that scope.

| Required acceptance | Draft evidence and remaining work |
|---|---|
| Complete interface parity | Generated source inventory covers 18 node schemas, 124 ordered inputs, 66 workflows, config, panel and TUI actions. All 101 CLI command/option rows have explicit action, parameter, selector, context or presentation mappings; checks reject missing mappings and mismatched defaults/choices. Typed workflow controls work for supported graphs. Source coverage and CPU tests do **not** establish live cross-interface acceptance. |
| Bundled optional companion | Python entry point, authenticated loopback host, compiled Svelte assets and explicit package data implemented. Exact reviewed update uses a frozen controller, retained transaction lock, selected driver interpreter and durable receipts. Real shutdown/update/restart transaction acceptance is **NOT RUN**. |
| Image/video/edit/upscale/Fleet outputs | Existing-node graph compilation and queue adapters implemented and mocked. Job-bound runtime fingerprint capture is implemented. Representative hardware outputs, runtime identity verification against a live Worker cohort and fidelity acceptance are **NOT RUN**. |
| Cross-interface semantic identity | Shared presentation schema and regression fixtures implemented. Complete names/defaults/formatting/topology parity needs cross-interface acceptance. |
| Race and recovery correctness | Durable idempotency, immutable submissions, optimistic revisions, replay gaps, restart unknown states, atomic prompt-specific cancellation, prompt-bound events, bounded preview frames, and per-prompt history reconciliation have CPU regressions. Older drivers without atomic cancellation refuse that operation. Monarch dispatch, validated Worker result, native owner ACK and individual Fleet outcome hooks have CPU regressions. Real Worker/ACK/Fleet coverage remains **NOT RUN**. |
| Workspace features | Mask/input uploads, bounded thumbnails, virtualized gallery, paginated queue, synchronized comparisons and telemetry charts are implemented. Durable grids include frozen named-style axes, per-cell inspection and selective retry. LoRA chains support typed insertion and extension. Saved records and artifacts have project membership; reviewed library import/export preserves definitions and source references. Complete end-to-end acceptance remains **NOT RUN**. |
| App Mode | Canonical/legacy identities, metadata preservation, primitive/reroute conversion, nested virtual interfaces and separate repeated-instance bindings have fixtures. Nested promoted scalar widgets resolve through explicit serialized mappings. Ambiguous legacy references, duplicate mappings and unsupported custom serialization refuse with a repair reason. Broader source-version fixtures remain an acceptance requirement; arbitrary widget JavaScript is outside the agreed scope. |
| Security | Token/Host/Origin/content-type checks, body limits, fixed driver routes, safe artifact descriptors, typed confirmed actions and stale-plan rejection have CPU tests. Full abuse testing and deployment review remain required. |
| Accessibility | Typed labels, focus styling, responsive layouts, reduced motion and keyboard actions implemented. Browser smoke is recorded in the PR; comprehensive keyboard/screen-reader/contrast/hidden-tab acceptance is **NOT RUN**. |
| Performance and retention | Explicit event, console and media limits implemented. Spark/browser overhead, long-session growth and variance-based regression budgets are **NOT RUN**. No zero-overhead or fastest-performance claim is made. |

Completed local feature work includes:

- Explicit mappings for all 101 CLI command and option rows, including
  diagnostics, last-gate inspection, reap grace, Worker synchronization and
  recording/replay defaults.
- Named style-stack grid axes that freeze ordered definitions and revisions;
  later style edits cannot change accepted cells or their retries.
- Typed LoRA insertion at selected consumer connections and extension of one
  existing chain, including Monarch's distinct `DGXM_MODEL` contract.
- Independent editing of repeated nested graph instances and explicit nested
  promoted-widget mapping, with unique legacy aliases retained.
- Project memberships, paginated record selection and reviewed library
  import/export with conflict/revision checks and unresolved source references.
- Saved experiment definitions linked to durable grids, with observed outcome
  references and explicit loading into a new draft for fresh validation.

Typed reviewed launcher, setup/init, Sol installation/smoke, update, uninstall,
and telemetry recording/replay adapters have CPU or mocked transaction tests.
Real service acceptance is still **NOT RUN**. An update targets the selected
runtime checkout and Worker cohort; it does not replace a separately installed,
currently running companion wheel.

The remaining release gates include the hardware and real-service cases below,
full browser/accessibility workflows, broader supported workflow-version
fixtures, cross-interface semantic checks, deployment/abuse review and measured
performance/retention budgets. A local implementation or source mapping does
not close those gates.

Job-bound Monarch evidence now uses explicit Comfy execution context and a
leased in-memory driver journal, with durable companion event/per-render
records. Tests cover exact prompt joins, late facts on ended jobs, conflicting
sequence refusal, replay and lease gaps, bounded summaries, native ACK
validation and scratch-thread identity, plus real PendingRender and Fleet
collection hooks using CPU fakes. See
[`test_execution_events.py`](../tests/test_execution_events.py) and
[`test_studio_runtime_observations.py`](../tests/test_studio_runtime_observations.py).
Correlation was also reviewed against ComfyUI source commit
`29e28e2b488e7739c351c331d73da25aa435da5c`: synchronous and asynchronous node
invocations both establish `CurrentNodeContext`. This source inspection is not
a live distributed generation result.

Runtime acceptance still requires representative real image/video/edit/upscale
and Fleet workloads, including native return, cancellation, typed refusal,
cleanup failure, reconnect and driver-restart cases. Verify job/render identity,
actual Worker generation/source identity, effective topology and attention,
validated output and exact owner ACK independently. Confirm that an absent
context or lost observation stays unavailable. The collector has no authority
to waive the runtime's class P/C/U/K guards or declare hardware acceptance.

The [performance record](STUDIO_PERFORMANCE.md) must compare an inactive
observer, active idle lease and observed generation against the same backend
baseline. Include a cold Worker fingerprint capture, subsequent captures,
lease expiry during an in-flight request, bounded driver replay and companion
database growth. Encoded payload limits and CPU tests do not establish host
RSS, generation overhead or a long-session disk budget.

The browser was checked against the explicitly synthetic CPU HTTP fixture:
workflow import, editable controls, immutable submission review, completed job,
two-cell named-style grid, durable outcomes and thumbnail gallery. Follow-up
checks covered project creation, negative-prompt/style-mode preservation,
revision-bound import conflicts, missing negative-input refusal, saving a grid
as an experiment and restoring its definition without dispatching more jobs.
The final installed wheel also served those saved project/experiment records
with its driver offline and Node absent from PATH. These checks do not establish
real video synchronization, screen-reader behavior, complete keyboard coverage
or the performance rows above.

All tests in this draft are CPU or mocked service tests unless explicitly
identified otherwise. They do not authorize live maintenance, grant capacity,
waive class P/K guards or establish native RDMA, ACK or model fidelity.

Local verification on 2026-10-19: the final complete CPU harness passed
13,588 tests with 39 skipped. Frontend tests passed all 22 cases with zero
Svelte/TypeScript errors or warnings. Ruff, whole-package mypy (475 source
files), generated inventory, trust citations, protected-test inventory and leak
checks passed. Wheel and source archive matched 478 runtime/package resource
files byte for byte. The isolated wheel served the final assets and its
authenticated API with Node absent from PATH.

The full CPU harness took 317 seconds on this local ARM64 host and emitted its
warning for exceeding the 120-second test-battery target. This is a test-runtime
warning, not a measurement of generation overhead. The observer microbenchmark
and its retained-allocation limits are recorded separately in the performance
record. No real hardware or production service acceptance was run.

The brand follow-up imports the canonical repository small dark SVG. The
welcome screen retains its headline without the unrelated orbit illustration. It
passed 22 frontend tests, zero Svelte/TypeScript diagnostics, all 418 Studio
packaging/resource/inventory cases and the brand-generation check. Browser
inspection confirmed the rendered SVG matches the source geometry and colors.
The full CPU result above belongs to implementation commit `ba31a02`.

The subsequent PR-wide quality pass added regressions for single-use operation
reviews, cancellation replay and interrupted local mutators, explicit local-only
diagnostics, immutable prompt identities, concurrent schema/runtime capture,
schema-bounded random seeds, reconnect races, and exact library references.
Live preview updates preserve memberships and extensions. Snapshot/history
summaries and byte-bounded library windows retain full provenance through
explicit detail reads. Streamed media carries the same security headers as
ordinary responses.

Compiled-browser checks against the synthetic CPU fixture verified queue detail
loading, restored prompt/seed through Gallery reuse, and a full read-only version
receipt. The browser reported no console errors. These do not close the
hardware, real-service, accessibility or long-session acceptance gates above.

Quality-pass validation on 2026-10-19: the complete CPU harness passed
13,652 tests with 39 skipped in 315.11 seconds (317 seconds for the wrapper;
its 120-second target warning remains advisory). The final Studio-only suite,
including the pagination regression added after full-suite collection, passed
all 851 cases. All 35 frontend tests passed, with zero Svelte/TypeScript
diagnostics. Ruff, mypy (479 source files), inventory, protected-test, trust,
artifact, brand/template and leak checks passed. Both installed package formats
passed `dgxm-studio --check`; wheel and source archive matched 482 runtime and
resource files. The installed wheel passed the actual synthetic HTTP generation
and idempotency smoke with Node absent from PATH.
