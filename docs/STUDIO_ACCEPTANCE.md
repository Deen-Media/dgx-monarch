# Studio acceptance record

Status: **incomplete draft**, not release acceptance. Baseline:
`5604ac77799f1f3e1a077a2f841dfce59fbcb892` (v1.2.0).

The original [specification](STUDIO_SPEC.md) remains the complete deliverable.
Unimplemented work below is not a proposed reduction of that scope.

| Required acceptance | Draft evidence and remaining work |
|---|---|
| Complete interface parity | Generated source inventory covers 19 node schemas, 129 ordered inputs, 71 workflows, config, panel and TUI actions. All 126 CLI command/option rows are mapped: 101 have explicit action, parameter, selector, context or presentation mappings, and the 25 rows of `dgxm studio` carry a written waiver, because they start, stop, open or reach the companion itself and the page cannot run them for itself; checks reject missing mappings and mismatched defaults/choices. Closed registries map all 7 panel handlers and 9 TUI bindings to a Studio operation, route or equivalent, or waive them with a reason (consent dismiss and revoke, the automatic rescue setting, and the terminal quit, pause, animation and theme keys); a new handler or binding fails generation. Node schema and node input rows record source coverage only, because Studio reads them from the live object_info, so no closed check yet fails when such a backend feature lacks a Studio representation. Typed workflow controls work for supported graphs. Source coverage and CPU tests do **not** establish live cross-interface acceptance. |
| Bundled optional companion | Python entry point, authenticated loopback host, compiled Svelte assets and explicit package data implemented. `dgxm-studio --check` hashes every packaged Studio file against the manifest. Update requires `--config` and `--driver-service` in both modes; without either, Studio lists update as unavailable and names the missing setting, so it is unavailable when ComfyUI runs from the documented foreground launcher rather than a systemd user unit. The CLI update is unaffected. It binds the selected checkout to the driver's `custom_nodes/dgx-monarch` target, defaults to the verified transaction, and runs in a frozen controller with the driver unit's interpreter, a retained transaction lock and durable receipts. A CPU test runs Studio's review and execution through the real verified transaction against a scratch checkout and a local bare origin, with scripted Worker steps. Real shutdown/update/restart transaction acceptance against a live driver unit and Workers is **NOT RUN**. The project owner kept the `--driver-service` rule, and the maintainer's rig runs ComfyUI from the foreground launcher with no systemd user unit, so Studio offers no update there. `dgxm studio` starts the companion with the driver origin, port, driver unit and start command of the `[studio]` table in cluster.toml, and signs the browser in with a link that carries the token in its fragment, through an owner-only `sign-in.html` page for a browser it starts; a companion started with `--allow-management` never makes a link. `dgxm studio up` installs, enables and starts an opt-in user unit, `dgxm-studio.service`, which the project owner approved on 2026-10-22, and `dgxm studio down` removes it. `dgxm studio tunnel` prints the SSH line and the address to browse from another computer, and the page shows a Driver not running panel with the command that starts the driver. CPU tests cover the config rules, the printing rules, the unit text, its checks and the tunnel lines. The Sign-in link and Driver not running browser rows passed on 2026-10-22 in headless Firefox 158.0 at 1440×900 and 390×844 ([interface QA record](STUDIO_INTERFACE_QA.md#browser-evidence)). A live service start on the maintainer's rig is **NOT RUN**, held for the project owner's go: `dgxm studio up` with a private copy of cluster.toml, the installed unit and its files, the companion's HTTP answers, `dgxm studio open` with and without a terminal, `dgxm studio token --rotate`, a restart after the process is killed, and `dgxm studio down`. The one-URL pattern, a Studio unit that starts a driver unit, is **NOT RUN** on that rig, which has no driver unit. |
| Image/video/edit/upscale/Fleet outputs | Existing-node graph compilation and queue adapters implemented and mocked. All 71 bundled workflows inspect and compile in CPU tests against a recorded driver schema, including upload loaders, V3 dynamic combos and Autogrow inputs. An uploaded input or mask binds to image upload widgets through inspection, preview and submission (CPU end-to-end test); a live driver run with bundled image workflows is **NOT RUN**. Job-bound runtime fingerprint capture is implemented. Representative hardware outputs, runtime identity verification against a live Worker cohort and fidelity acceptance are **NOT RUN**. |
| Cross-interface semantic identity | Shared presentation schema and regression fixtures implemented. The memory bar's slab rule is shared by `dgxm top`, the sidebar and Studio's charts and is pinned by Python, browser panel and Studio parity tests. Complete names/defaults/formatting/topology parity needs cross-interface acceptance. |
| Race and recovery correctness | Durable idempotency, immutable submissions, optimistic revisions, replay gaps, restart unknown states, atomic prompt-specific cancellation, prompt-bound events, bounded preview frames, and per-prompt history reconciliation have CPU regressions. Dequeued cancellations, driver restarts and history clears settle from exact per-prompt evidence, a lost acknowledgement reconciles through the companion-issued identity, and the polling observer and event watcher survive storage faults; all have CPU regressions, including a polling lifecycle test against an in-process driver. On a driver without atomic cancellation, a queued job falls back to queue deletion and a running job is refused with its intent unspent. Monarch dispatch, validated Worker result, native owner ACK and individual Fleet outcome hooks have CPU regressions. The lifecycle hardware pass ran on 2026-10-21 at `ff8dab83`, which carries these hooks, on the maintainer's two-Spark rig. The driver ran from the foreground launcher with no Studio companion, and a scripted reader held the observer lease through the driver's execution-events route from the first Recycle on. A 50-step MOSS-Image render with no lease returned ok. A Recycle under a telemetry poll of 4.5 requests per second returned `recycled` in 3.81 seconds, and all 67 polls answered HTTP 200. A second Recycle 2 seconds later returned `no_live_mesh` in 0.06 seconds, the quiet no-op, and added nothing to the driver log. The respawn render with the lease live matched the first render bit for bit in latent and pixels. The shipped Fleet workflow, with a widget name map added to one loader node for the test tool, ran 4 jobs in 2 waves on 2 GPUs. The lease journal recorded dispatch, Worker result, validation and completion events for the respawn render and each Fleet item, with no failure event and no gap. `dgxm restart` exited 0, both Workers came back on the same release slot, and `dgxm doctor` reported 44 checks with 0 failures. The driver log and both Worker journals held 0 ERROR lines. The head GPU ran under its 2100 MHz stability cap. That pass did not repeat the recycle sequence without a lease, did not run Worker slot activation in the restart (activation last ran in this commit's verified update), and could not reach the RDMA and native ACK hooks, because native latent return is off on the rig. Those parts, native ACK coverage, and real Worker and Fleet coverage through Studio remain **NOT RUN**. |
| Workspace features | Mask/input uploads, bounded thumbnails, virtualized gallery, paginated queue, synchronized comparisons and telemetry charts are implemented. Durable grids include frozen named-style axes, per-cell inspection and selective retry. LoRA chains support typed insertion and extension; LoRA model compatibility has no data source and stays pending backend validation. Saved records and artifacts have project membership; reviewed library import/export preserves definitions and source references. Complete end-to-end acceptance remains **NOT RUN**. |
| App Mode | Canonical/legacy identities, metadata preservation, primitive/reroute conversion, nested virtual interfaces and separate repeated-instance bindings have fixtures. Nested promoted scalar widgets resolve through explicit serialized mappings. Ambiguous legacy references, duplicate mappings and unsupported custom serialization refuse with a repair reason. Broader source-version fixtures remain an acceptance requirement; arbitrary widget JavaScript is outside the agreed scope. |
| Security | Token/Host/Origin/content-type checks, body limits, fixed driver routes, safe artifact descriptors, typed confirmed actions (every mutating action, both retries and one HTTP case refuse an unconfirmed apply with no reservation) and stale-plan rejection have CPU tests. [THREAT_MODEL.md](THREAT_MODEL.md#browser-to-studio-companion) records the companion's two boundaries and the owner token's scope. Full abuse testing and deployment review remain required. |
| Accessibility | Typed labels, focus styling, responsive layouts, reduced motion and keyboard actions implemented. The [interface QA record](STUDIO_INTERFACE_QA.md) separates authored-control inspection from scoped browser checks. Its twelve browser rows ran on 2026-10-22 in headless Firefox 158.0 on the bundle of the `dgxm studio` launch, and all twelve passed at every viewport where their check applies. On 2026-10-21 a first run of the ten earlier rows found the video metadata panel at zero height at 1024×600, before the viewer fix. Comprehensive keyboard/screen-reader/contrast/hidden-tab acceptance is **NOT RUN**. |
| Performance and retention | Explicit event, console and media limits implemented. Expired job and grid previews are swept at startup and on later preview writes. Spark/browser overhead, long-session growth and variance-based regression budgets are **NOT RUN**. No zero-overhead or fastest-performance claim is made. |

Completed local feature work includes:

- Explicit mappings for 101 CLI command and option rows, including
  diagnostics, last-gate inspection, reap grace, Worker synchronization and
  recording/replay defaults, and a written waiver for the 25 rows of
  `dgxm studio`, which start, stop, open or reach the companion itself: 126
  rows in all.
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
screens) first ran on earlier bundle builds and recorded no browser name,
browser version or bundle hash. On 2026-10-21 the ten rows of the
[interface QA record](STUDIO_INTERFACE_QA.md#browser-evidence) ran again with
the [browser procedure](../studio/tests/README.md#browser-checks), in headless
Firefox 158.0 at four viewports and again with reduced motion. A first run found
the video metadata panel at zero height at 1024×600. The viewer now keeps the
panel's height and scrolls in a short window, and all ten rows passed on the
rebuilt bundle at every viewport where their check applies. On 2026-10-22 the
twelve rows, the ten and the new Sign-in link and Driver not running rows, ran
on the bundle of the `dgxm studio` launch and all passed; the two new rows ran
at 1440×900 and 390×844 with companions and browser profiles of their own
(results SHA-256
`7aaa73cb7f9e0edc5aad4875affd98f90139eaf65bd5fad35c75da434e4373c6`). Both runs
repeated workflow import, editable controls, submission review and the review
of a two-cell grid. Neither repeated the named-style grid axis, grid dispatch
and its durable outcomes, the thumbnail gallery contents, project and
experiment records or the quarantine screens, so those still rest on the
earlier builds. Headless Firefox could not show Chromium or
Safari behavior, native phone operating-system behavior, native select pop-ups,
video control internals, the disclosure transition that Firefox does not apply,
or frame timing under hardware composition. The
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
