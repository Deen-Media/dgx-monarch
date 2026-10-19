# Studio acceptance record

Status: **incomplete draft**, not release acceptance. Baseline:
`5604ac77799f1f3e1a077a2f841dfce59fbcb892` (v1.2.0).

The original [specification](STUDIO_SPEC.md) remains the complete deliverable.
Unimplemented work below is not a proposed reduction of that scope.

| Required acceptance | Draft evidence and remaining work |
|---|---|
| Complete interface parity | Generated source inventory covers 18 node schemas, 124 ordered inputs, 66 workflows, CLI options, config, panel and TUI actions. Typed workflow controls work for supported graphs. Inventory coverage does **not** establish full working parity. Missing management adapters are explicitly identified in capabilities. |
| Bundled optional companion | Python entry point, authenticated loopback host, compiled Svelte assets and explicit package data implemented. Driver shutdown does not terminate the host. Real shutdown/update/restart transaction acceptance is **NOT RUN**; exact update adapter remains incomplete. |
| Image/video/edit/upscale/Fleet outputs | Existing-node graph compilation and queue adapters implemented and mocked. Representative hardware outputs, immutable runtime fingerprints and fidelity acceptance are **NOT RUN**. |
| Cross-interface semantic identity | Shared presentation schema and regression fixtures implemented. Complete names/defaults/formatting/topology parity needs cross-interface acceptance. |
| Race and recovery correctness | Durable idempotency, immutable submissions, optimistic revisions, replay gaps, restart unknown states and queued deletion have CPU regressions. Running-job cancellation and full backend event reconciliation need completion and race testing. |
| Workspace features | Generation controls, viewer, queue, galleries, comparison, records, styles, bound LoRA slots, grid preview, command palette and presentation mode implemented. Full mask editing, virtualized thumbnail library, synchronized media comparison and experiment dispatch/selective cell retry remain incomplete. |
| App Mode | Canonical/legacy identities, metadata preservation, graph-aware inspection and primitive/reroute conversion have fixtures. Nested UI subgraph execution and arbitrary custom widgets are refused. Supported subgraph execution still needs implementation; arbitrary widget parity is outside the agreed scope. |
| Security | Token/Host/Origin/content-type checks, body limits, fixed driver routes, safe artifact descriptors, typed confirmed actions and stale-plan rejection have CPU tests. Full abuse testing and deployment review remain required. |
| Accessibility | Typed labels, focus styling, responsive layouts, reduced motion and keyboard actions implemented. Browser smoke is recorded in the PR; comprehensive keyboard/screen-reader/contrast/hidden-tab acceptance is **NOT RUN**. |
| Performance and retention | Explicit event, console and media limits implemented. Spark/browser overhead, long-session growth and variance-based regression budgets are **NOT RUN**. No zero-overhead or fastest-performance claim is made. |

Missing management work includes exact reviewed launcher/setup/Sol/update
transactions, uninstall ownership and receipt settlement, and complete
recording/replay. Each needs a typed adapter preserving its existing guards;
enabling a generic command proxy would not satisfy this requirement.

All tests in this draft are CPU or mocked service tests unless explicitly
identified otherwise. They do not authorize live maintenance, grant capacity,
waive class P/K guards or establish native RDMA, ACK or model fidelity.
