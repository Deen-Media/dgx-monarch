# Studio acceptance record

Status: **incomplete draft**, not release acceptance. Baseline:
`5604ac77799f1f3e1a077a2f841dfce59fbcb892` (v1.2.0).

The original [specification](STUDIO_SPEC.md) remains the complete deliverable.
Unimplemented work below is not a proposed reduction of that scope.

| Required acceptance | Draft evidence and remaining work |
|---|---|
| Complete interface parity | Generated source inventory covers 19 node schemas, 129 ordered inputs, 71 workflows, config, panel and TUI actions. All 101 CLI command/option rows have explicit action, parameter, selector, context or presentation mappings; checks reject missing mappings and mismatched defaults/choices. Closed registries map all 7 panel handlers and 9 TUI bindings to a Studio operation, route or equivalent, or waive them with a reason (consent dismiss and revoke, the automatic rescue setting, and the terminal quit, pause, animation and theme keys); a new handler or binding fails generation. Node schema and node input rows record source coverage only, because Studio reads them from the live object_info, so no closed check yet fails when such a backend feature lacks a Studio representation. Typed workflow controls work for supported graphs. Source coverage and CPU tests do **not** establish live cross-interface acceptance. |
| Bundled optional companion | Python entry point, authenticated loopback host, compiled Svelte assets and explicit package data implemented. `dgxm-studio --check` hashes every packaged Studio file against the manifest. Update requires `--config` and `--driver-service` in both modes; without either, Studio lists update as unavailable and names the missing setting, so it is unavailable when ComfyUI runs from the documented foreground launcher rather than a systemd user unit. The CLI update is unaffected. It binds the selected checkout to the driver's `custom_nodes/dgx-monarch` target, defaults to the verified transaction, and runs in a frozen controller with the driver unit's interpreter, a retained transaction lock and durable receipts. A CPU test runs Studio's review and execution through the real verified transaction against a scratch checkout and a local bare origin, with scripted Worker steps. Real shutdown/update/restart transaction acceptance against a live driver unit and Workers is **NOT RUN**. |
| Image/video/edit/upscale/Fleet outputs | Existing-node graph compilation and queue adapters implemented and mocked. All 71 bundled workflows inspect and compile in CPU tests against a recorded driver schema, including upload loaders, V3 dynamic combos and Autogrow inputs. An uploaded input or mask binds to image upload widgets through inspection, preview and submission (CPU end-to-end test); a live driver run with bundled image workflows is **NOT RUN**. Job-bound runtime fingerprint capture is implemented. Representative hardware outputs, runtime identity verification against a live Worker cohort and fidelity acceptance are **NOT RUN**. |
| Cross-interface semantic identity | Shared presentation schema and regression fixtures implemented. The memory bar's slab rule is shared by `dgxm top`, the sidebar and Studio's charts and is pinned by Python, browser panel and Studio parity tests. Complete names/defaults/formatting/topology parity needs cross-interface acceptance. |
| Race and recovery correctness | Durable idempotency, immutable submissions, optimistic revisions, replay gaps, restart unknown states, atomic prompt-specific cancellation, prompt-bound events, bounded preview frames, and per-prompt history reconciliation have CPU regressions. Dequeued cancellations, driver restarts and history clears settle from exact per-prompt evidence, a lost acknowledgement reconciles through the companion-issued identity, and the polling observer and event watcher survive storage faults; all have CPU regressions, including a polling lifecycle test against an in-process driver. On a driver without atomic cancellation, a queued job falls back to queue deletion and a running job is refused with its intent unspent. Monarch dispatch, validated Worker result, native owner ACK and individual Fleet outcome hooks have CPU regressions. Real Worker/ACK/Fleet coverage and the lifecycle hardware pass at this head (render, recycle, respawn, a Fleet wave, `dgxm restart` and `dgxm doctor` with Worker activation) remain **NOT RUN**. |
| Workspace features | Mask/input uploads, bounded thumbnails, virtualized gallery, paginated queue, synchronized comparisons and telemetry charts are implemented. Durable grids include frozen named-style axes, per-cell inspection and selective retry. LoRA chains support typed insertion and extension; LoRA model compatibility has no data source and stays pending backend validation. Saved records and artifacts have project membership; reviewed library import/export preserves definitions and source references. Complete end-to-end acceptance remains **NOT RUN**. |
| App Mode | Canonical/legacy identities, metadata preservation, primitive/reroute conversion, nested virtual interfaces and separate repeated-instance bindings have fixtures. Nested promoted scalar widgets resolve through explicit serialized mappings. Ambiguous legacy references, duplicate mappings and unsupported custom serialization refuse with a repair reason. Broader source-version fixtures remain an acceptance requirement; arbitrary widget JavaScript is outside the agreed scope. |
| Security | Token/Host/Origin/content-type checks, body limits, fixed driver routes, safe artifact descriptors, typed confirmed actions (every mutating action, both retries and one HTTP case refuse an unconfirmed apply with no reservation) and stale-plan rejection have CPU tests. [THREAT_MODEL.md](THREAT_MODEL.md#browser-to-studio-companion) records the companion's two boundaries and the owner token's scope. Full abuse testing and deployment review remain required. |
| Accessibility | Typed labels, focus styling, responsive layouts, reduced motion and keyboard actions implemented. The [interface QA record](STUDIO_INTERFACE_QA.md) separates authored-control inspection from scoped browser checks. Comprehensive keyboard/screen-reader/contrast/hidden-tab acceptance is **NOT RUN**. |
| Performance and retention | Explicit event, console and media limits implemented. Expired job and grid previews are swept at startup and on later preview writes. Spark/browser overhead, long-session growth and variance-based regression budgets are **NOT RUN**. No zero-overhead or fastest-performance claim is made. |

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
Real service acceptance is still **NOT RUN**. An update targets the checkout the
driver imports and the Worker cohort. Studio refuses it when the companion's own
dgx_monarch source differs from that checkout, so a separately installed
companion wheel must be reinstalled after an update.

The remaining release gates include the hardware and real-service cases below,
full browser/accessibility workflows, broader supported workflow-version
fixtures, cross-interface semantic checks, deployment/abuse review and measured
performance/retention budgets. A local implementation or source mapping does not
close those gates.

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

Browser checks against the synthetic CPU HTTP fixture (workflow import, editable
controls, submission review, a two-cell named-style grid, durable outcomes, the
thumbnail gallery, project and experiment records, and the request quarantine
screens) ran on earlier bundle builds and recorded no browser name, browser
version or bundle hash. Repeating them on the shipped bundle with the [browser
procedure](../studio/tests/README.md#browser-checks) is a follow-up. The
synthetic HTTP smoke itself now runs in every CPU test run against an in-process
companion (`tests/test_studio_tooling.py`). These checks do not establish real
video synchronization, screen-reader behavior, complete keyboard coverage or the
performance rows above.

All tests in this draft are CPU or mocked service tests unless explicitly
identified otherwise. They do not authorize live maintenance, grant capacity,
waive class P/K guards or establish native RDMA, ACK or model fidelity.

## Current result

Measured on 2026-10-21 (UTC) on the tree that the first set of PR 63 review-fix
commits records on top of `a0afbb0aa4fb3d7bb1522bfe49353438d097ee00`, before
those commits were made. This result covers that tree only; any later commit
must be measured again before this section applies to it. The full CPU suite
passed 14,374 tests with 27 skipped; `tests/test_brand_assets.py` and
`tests/test_template_cards.py` were left out because the measuring host lacks
CairoSVG and defusedxml, and the suite took 366 seconds against its advisory
120-second target. All 258 frontend tests passed with zero Svelte/TypeScript
errors or warnings. They server-render Generation, the asset editor, the field,
the status banners, the token form, Queue, JobProvenance, Gallery and the
viewer, and run the shipped grid dispatch, grid retry and Operations apply
handlers, but no test types into, clicks or paints a mounted DOM, so typing,
focus and canvas paint are covered only by helper and action tests. Ruff,
whole-package mypy (491 source files), the generated inventory, trust citations,
the protected-test inventory, the refusal ledger, templates, the artifact
manifest, the dependency check and the leak check passed. The wheel and the
source archive, each installed without dependencies, passed
`dgxm-studio --check`, which hashed every packaged Studio file against the
manifest, and their 494 installed package files matched byte for byte. Hosted CI
results for the pushed head must be checked separately. No hardware, live driver
or production service was exercised.
