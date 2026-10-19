# Studio acceptance record

Status: **incomplete draft**, not release acceptance. Baseline:
`5604ac77799f1f3e1a077a2f841dfce59fbcb892` (v1.2.0).

The original [specification](STUDIO_SPEC.md) remains the complete deliverable.
Unimplemented work below is not a proposed reduction of that scope.

| Required acceptance | Draft evidence and remaining work |
|---|---|
| Complete interface parity | Generated source inventory covers 18 node schemas, 124 ordered inputs, 66 workflows, CLI options, config, panel and TUI actions. Typed workflow controls work for supported graphs. Inventory coverage does **not** establish full working parity. Missing management adapters are explicitly identified in capabilities. |
| Bundled optional companion | Python entry point, authenticated loopback host, compiled Svelte assets and explicit package data implemented. Exact reviewed update uses a frozen controller, retained transaction lock, selected driver interpreter and durable receipts. Real shutdown/update/restart transaction acceptance is **NOT RUN**. |
| Image/video/edit/upscale/Fleet outputs | Existing-node graph compilation and queue adapters implemented and mocked. Representative hardware outputs, immutable runtime fingerprints and fidelity acceptance are **NOT RUN**. |
| Cross-interface semantic identity | Shared presentation schema and regression fixtures implemented. Complete names/defaults/formatting/topology parity needs cross-interface acceptance. |
| Race and recovery correctness | Durable idempotency, immutable submissions, optimistic revisions, replay gaps, restart unknown states, atomic prompt-specific cancellation, prompt-bound events, bounded preview frames, and per-prompt history reconciliation have CPU regressions. Older drivers without atomic cancellation refuse that operation. Full worker-ACK/Fleet event coverage remains unverified. |
| Workspace features | Mask/input uploads, bounded thumbnail generation, virtualized gallery, paginated queue, synchronized comparison controls, telemetry charts, durable grid dispatch and selective failed-cell retry are implemented. Grid axes for named style stacks, graph insertion beyond existing LoRA slots, and complete end-to-end coverage remain outstanding. |
| App Mode | Canonical/legacy identities, metadata preservation, graph-aware inspection, primitive/reroute conversion, nested virtual input/output links and distinct subgraph execution IDs have fixtures. Ambiguous edits to repeated definitions and unsupported promoted/custom widget serialization refuse. Broader source-version fixtures and explicit supported-widget adapters remain required; arbitrary widget parity is outside the agreed scope. |
| Security | Token/Host/Origin/content-type checks, body limits, fixed driver routes, safe artifact descriptors, typed confirmed actions and stale-plan rejection have CPU tests. Full abuse testing and deployment review remain required. |
| Accessibility | Typed labels, focus styling, responsive layouts, reduced motion and keyboard actions implemented. Browser smoke is recorded in the PR; comprehensive keyboard/screen-reader/contrast/hidden-tab acceptance is **NOT RUN**. |
| Performance and retention | Explicit event, console and media limits implemented. Spark/browser overhead, long-session growth and variance-based regression budgets are **NOT RUN**. No zero-overhead or fastest-performance claim is made. |

Concrete implementation gaps still include:

- Full CLI option parity: diagnostics, last-gate inspection, reap grace,
  Worker synchronization choices and identical recording defaults/ranges.
- Style-stack grid axes and insertion of additional LoRA loader slots.
- Instance-specific editing of repeated subgraph definitions and nested
  promoted-widget adapters beyond the supported scalar forms.
- Job-bound Monarch admission, effective topology, Worker acknowledgements
  and durable individual Fleet-item results. Generic Comfy execution events
  do not supply these facts; this is missing integration work, not hardware proof.
- Project membership for every shared record and round-trip library import.

Typed reviewed launcher, setup/init, Sol installation/smoke, update, uninstall,
and telemetry recording/replay adapters now have CPU or mocked transaction
tests. This does not establish full option-by-option CLI parity or real service
acceptance. An update targets the selected runtime checkout and Worker cohort;
it does not replace a separately installed, currently running companion wheel.

The browser was checked against the explicitly synthetic CPU HTTP fixture:
workflow import, editable controls, immutable submission review, completed job,
two-cell grid dispatch, durable outcomes and thumbnail gallery. These checks do
not establish real video synchronization, screen-reader behavior, complete
keyboard coverage or the performance rows below.

All tests in this draft are CPU or mocked service tests unless explicitly
identified otherwise. They do not authorize live maintenance, grant capacity,
waive class P/K guards or establish native RDMA, ACK or model fidelity.

Local verification on 2026-10-19: full CPU suite 13,391 passed / 39 skipped;
final Studio-specific suite 604 passed; frontend 16 passed with zero
Svelte/TypeScript diagnostics. Ruff, whole-package mypy, generated inventory,
trust citations, protected-test inventory and leak checks passed. Wheel and
source archive matched 464 tracked package files byte for byte. The isolated
wheel served final assets and its authenticated API with Node absent from PATH.
