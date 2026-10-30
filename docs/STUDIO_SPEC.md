# DGX Monarch Studio Implementation Specification

Status: original specification, written against Monarch v1.2.0, commit 5604ac77799f1f3e1a077a2f841dfce59fbcb892. This document defines the complete deliverable and acceptance criteria. Studio and its companion now exist in this tree, and from v1.5 on Studio is a **supported** interface of DGX Monarch: [STUDIO.md](STUDIO.md) describes what it does, and the [acceptance record](STUDIO_ACCEPTANCE.md) records which criteria it meets and which remain open. Sentences below that describe the backend as it stood when this specification was written say so.

## Product contract

Build DGX Monarch Studio as an optional, bundled interface for the full supported Monarch experience. Studio, the CLI, the TUI, and the ComfyUI panel are equal entry points to the same backend. Installing or using Studio must not make the other interfaces secondary, introduce a Studio-only execution engine, or require an interface switch to complete a supported operation.

The complete deliverable includes image and video generation, editing, upscaling, workflow execution, Fleet controls, operational diagnostics, gallery and comparison tools, parameter grids, prompt styles, presets, projects, and session recovery. All features specified here are included in the implementation and release acceptance. Dependencies describe engineering order, not separate product releases or deferred features.

Preserve the repository's existing rule against altering ComfyUI source. Studio lives entirely in Monarch-owned code and uses supported integration boundaries. Do not fork, patch, or monkey-patch ComfyUI to make the interface work. Preserve Monarch's existing, deliberately designed backend runtime integrations and their inventories; this requirement does not retroactively prohibit those mechanisms. Arbitrary third-party widget parity and replacement of ComfyUI's graph-authoring environment are outside the supported-feature claim.

## Architecture and packaging

Use Svelte, TypeScript, and Vite for a statically compiled application. Ship versioned, content-hashed assets with the Python distribution. Node is a development/build dependency only. Production installation, startup, and ordinary use must not require Node, a CDN, or a separate frontend development server. Add explicit wheel package-data and source-archive coverage. Keep the app bundle outside WEB_DIRECTORY, where ComfyUI loads files as extension modules.

Svelte suits this bounded, reactive workspace and compiles components into browser JavaScript; Vite produces the deployable static bundle. This is a maintainability choice, not a claim that it is universally faster. React could implement the same design. Consider Solid only if measurements identify a relevant problem; a Rust backend rewrite is unwarranted without an established bottleneck. [Svelte overview](https://svelte.dev/docs/svelte/overview), [Vite production builds](https://vite.dev/guide/build.html).

Separate the implementation into a presentation layer, a typed client contract, Monarch-owned service adapters, and the existing backend execution and safety layers. Components never invoke shell commands, infer job state from log text, or reproduce admission and recovery decisions. Adapters translate approved operations into existing backend calls and return authoritative results.

When this specification was written, the backend mounted GET /dgxm/telemetry, GET /dgxm/metrics, POST /dgxm/recycle, GET /dgxm/consents, and POST /dgxm/consent through ComfyUI's PromptServer integration. Those were the existing routes, not a complete Studio API, and the job, catalog, artifact, session, capability, and operation contracts described below were new engineering work. The implementation adds GET /dgxm/execution-events to the driver and serves the Studio API from the companion; the [threat model](THREAT_MODEL.md#browser-to-comfyui-driver) lists the current driver routes and [STUDIO.md](STUDIO.md) describes the companion. Fleet uses the DGXMonarchFleetKSampler graph node; do not invent an existing Fleet REST service. [Runtime routes](https://github.com/Deen-Media/dgx-monarch/blob/v1.2.0/src/dgx_monarch/nodes/routes.py), [consent routes](https://github.com/Deen-Media/dgx-monarch/blob/v1.2.0/src/dgx_monarch/nodes/consent_routes.py).

Provide an optional Monarch-owned Python management companion, independent of the ComfyUI driver, as Studio’s authoritative static host and operation-stream origin. This is required because the existing update command refuses a live ComfyUI driver. The companion must keep the page, reviewed operation, receipt, and progress available while stopping, updating, and restarting that driver. Use constrained adapters to existing ComfyUI endpoints and Monarch operations. Publish versioned schemas and capability negotiation; reject incompatible clients explicitly.

The companion is absent or inactive when Studio is not used. Studio installs no service; an operator who runs the companion under a service manager writes and owns that unit. Existing CLI/core behavior remains unchanged. Give it only existing owner privileges and explicitly authorized service permissions. Persist operation identity and receipts independently of the driver, reconcile after restart, and distinguish driver failure from companion failure. No arbitrary command strings, inferred privilege escalation, or new permission grants. Amended 2026-10-22 with the owner's approval: `dgxm studio up` installs one systemd user unit for the companion, `dgxm-studio.service`, and `dgxm studio down` removes it. While that unit is installed the companion runs when Studio is not in use. `dgxm studio` joins the CLI and `[studio]` joins cluster.toml for it; other CLI commands and the core keep their behavior. Studio installs nothing else ([STUDIO.md](STUDIO.md#keep-studio-running)).

## Required parity inventory

Before wiring controls, generate a checked-in inventory from the implementation, CLI help, TUI actions, panel actions, nodes, configuration, and tests. Each row records the operation or setting, source symbol, stable identity, data type, allowed values, default, units, applicability, validation, consent or guard, preview/apply behavior, recovery behavior, and presentation in every interface.

Cover all 18 published node schemas and 66 example workflows at the pinned baseline, every CLI option, and all supported generation families, model and workflow controls, LoRA parameters, preview controls, execution modes, telemetry, Fleet, cache/memory operations, configuration, diagnostics, and administrative operations. Resolve conflicting defaults and terminology in the shared backend contract rather than hiding disagreement in Studio. Add a test for each row and a coverage check that fails when a supported backend feature lacks a Studio representation.

Distinguish availability from omission. A control can be unavailable because the current model, topology, installed capability, permission, or service state does not support it. Explain that reason in place. Every supported advanced parameter must remain editable through a typed advanced panel even when it has no specialized visual component.

## Shared state and contracts

Define versioned records for capabilities, generation drafts, immutable job submissions, job state, artifacts, catalog entries, workflows and bindings, telemetry samples, operations, presets, prompt styles, projects, and sessions. Use stable IDs rather than labels or paths as cross-component identity. Preserve unknown extension metadata on import/export.

The backend owns execution state, effective configuration, guards, queue ordering, worker health, and results. Studio owns unsent draft edits and local presentation preferences. A change submitted through another interface must appear in Studio through the same authoritative event stream or reconciliation snapshot.

Every accepted job stores the effective prompt, negative prompt where supported, resolved seed or per-item seeds, model identities and available fingerprints, ordered LoRA stack, complete effective parameters, workflow snapshot and digest, adapter/schema versions, relevant software versions, execution topology, and parent artifact lineage. Store requested and effective values when normalization changes them. Reproduction means reconstructing that configuration; do not promise universally deterministic images across hardware, versions, or nondeterministic kernels.

Use server-issued IDs and idempotency keys for submission. After a timeout, query the request's status before retrying. An unknown outcome remains visibly unknown until reconciled. Never silently duplicate a possibly accepted job. Events require job and worker identity, sequence or cursor information, server timestamps, and a documented snapshot/replay strategy. Duplicated, delayed, and out-of-order messages must not regress state.

## Workspace and navigation

The default workspace has a generation-control column, a central image/video viewer, and collapsible Spark telemetry and live-console regions. Provide resizable panels, persistent layout preferences, sensible minimum widths, and compact layouts without hiding required actions. Queue and gallery remain one action away. Dedicated Fleet, diagnostics, project, and settings views share the same shell and selected execution context.

The header identifies the connected host, selected project, active model/workflow, execution target, connection freshness, and current activity. Dirty draft changes are distinct from submitted settings. Selecting an older result never overwrites the current draft without an explicit reuse or remix action.

A searchable command palette provides navigation and the same typed actions exposed elsewhere, with keyboard shortcuts, availability reasons, and matching confirmation flows. Presentation mode maximizes the viewer and optionally hides prompts, paths, logs, and telemetry without changing backend state. Exiting it restores the previous layout.

## Generation and editing

Render controls from the capability inventory, organized around the selected operation and workflow. Include prompts, models, dimensions, aspect ratios, sampling parameters, seed policy, batch controls, input assets, masks, denoise/edit strength, video duration or frames, frame rate, and upscale controls wherever supported. Do not fabricate parameters for models that lack them.

Image, video, edit, and upscale operations belong in the same workspace. Reuse an output as an input through an explicit handoff that preserves lineage. Input and mask editors must expose dimensions, orientation, replacement and clearing behavior, and the actual asset submitted. Video inputs and outputs display duration, dimensions, frame rate, and format when known; missing metadata is labelled unknown.

Validation combines immediate field feedback with authoritative backend validation and a new Studio submission preview. Fleet currently has no preview/apply planner; implement this validation adapter without describing it as an existing service. Show applicable memory, topology, or resource warnings before submission. Never convert a warning into silent parameter reduction. A disabled submission button explains the blocking fields or guard and links to the relevant correction.

Presets store reusable generation configurations and clearly state which fields they replace. Prompt styles are separate named prompt transformations. Show the resulting prompt before submission, define style ordering, and preserve the unexpanded draft plus effective submitted text. Applying either must support undo and must not alter an existing job.

## LoRA browser and stack

Provide a searchable LoRA catalog with available metadata, model compatibility, path or stable identifier, and optional locally available previews. Support filtering and a clear missing/unavailable state. Search and thumbnail loading are bounded and lazy. Never imply that a preview or filename establishes compatibility.

The selected stack supports adding, ordering where meaningful, enabled toggles, removal, and numeric strength entry. Expose separate model and text-encoder strengths when the backend supports them; otherwise show the supported combined control. Use backend limits, steps, and defaults. Preserve disabled entries and their values in the draft and preset.

Resolve assets and effective strengths at submission and include them in the immutable snapshot. Every stack change affects the next submission only. The queue inspector continues to show the stack used by its selected job, including entries subsequently removed from the draft or local catalog.

## Viewer and live execution

The central viewer supports fit, actual pixels, zoom, pan, fullscreen, image download, and metadata inspection. Video supports playback, seeking, frame inspection where feasible, mute, looping, and truthful loading/error states. Do not auto-download or decode an entire gallery of full-resolution media.

Clearly distinguish intermediate previews, final outputs, and failed or incomplete artifacts. Associate every frame with its job, output identity, and timestamp. Respect backend preview settings and overhead constraints. Browser reconnect must not display a previous job's preview as current.

Progress reflects real backend phases and known denominators. Show step counts when supplied, elapsed phase time, and the current operation. Use an indeterminate state when completion percentage is unavailable. Estimated remaining time must be labelled an estimate and based on comparable observed work. Never animate invented percentage progress.

## Queue and artifact library

Queue entries show operation, submission time, status, target, thumbnail, and frozen effective settings. Expand an entry for its phases, logs, topology, warnings, and outputs. Support every existing queue action, including cancellation and retry where available, through the backend's established semantics. Distinguish a requested cancellation from confirmed cancellation, and explain whether completed outputs are retained.

Retry creates a new, linked submission with explicit configuration and seed behavior. Editing queued settings must use supported backend behavior or create a replacement through an explicit cancel-and-resubmit flow. Batch and grid jobs retain parent/child identities, per-item outcomes, and selective retry without rerunning successful items accidentally.

The gallery provides virtualized thumbnail browsing, search, filters, sorting, selection, project membership, and persistent provenance. Filters include operation, model, date, status, and available metadata. Reuse reconstructs settings; remix creates an editable derivative with lineage. Copy prompt, copy seed, load workflow, send to edit/upscale, download, and compare are direct actions. Missing source models or inputs produce a repair list, never a fabricated successful reconstruction.

## Comparison and parameter grids

Include side-by-side and image wipe comparisons with synchronized zoom where appropriate. Expose parameter differences and distinguish changed prompts, seeds, models, LoRAs, workflows, and topology. Video comparison supports clearly labelled playback synchronization and handles differing durations without suggesting nonexistent frame alignment.

Parameter grids expose named axes from the active typed parameter schema, including supported seeds, styles, strengths, and workflow inputs. Validate every combination, show the total job count, expected resource implications, and execution target before submission. Keep the chosen seed policy explicit. Invalid combinations require correction or explicit exclusion before execution.

Preserve the experiment definition and each cell's complete submission and outcome. Provide labelled axes, cell inspection, selective retry, comparisons, and export of the grid manifest and results. Bound concurrent execution through backend scheduling and admission controls. Studio must not create a second scheduler or bypass Fleet constraints.

## Projects and session continuity

Projects group drafts, workflows, presets, styles, experiments, artifacts, and notes without moving or duplicating source assets implicitly. Sessions capture the current project, draft, selected result, panel state, and active job references. Define storage ownership, retention, schema migration, and export/import formats explicitly.

Persist recoverable draft changes locally and durable shared records through the host service. Support interrupted writes and version conflicts. Multi-tab edits must not silently overwrite newer saved data. On restore, reconcile jobs with the backend before offering retry or control actions.

Show connected, reconnecting, stale, and unavailable states distinctly. Retain the last known sample with its timestamp and age, rather than displaying it as live. Use capped reconnect backoff and a visible retry control. A service restart may change process identity, so reconcile identifiers and capabilities before resuming subscriptions.

The companion keeps Studio operational while ComfyUI is stopped or restarting. If the companion itself is unreachable, preserve local state and explain the bootstrap or service recovery needed. No interface can restart its own entirely unavailable host through an offline button. This unavoidable initial-bootstrap boundary must not become a CLI fallback for supported operations while the companion is running.

## Fleet and operational controls

Provide the complete supported Fleet surface and make its semantics explicit. DGXMonarchFleetKSampler distributes independent, world-1 sampling jobs round-robin across ranks, with at most one job per rank in each wave. It shares model, source latent, steps, CFG, sampler, and scheduler. Optional positive-conditioning entries precede prompt-line jobs; job i uses base seed + i. Negative prompts must be one broadcast value or exactly one per job. Preserve the 1,024-job limit and return LATENT results in input order. Heterogeneous model/settings/size per Fleet job are not an existing capability.

Expose participating workers, readiness, supported topology and execution settings without equating Fleet queue distribution with Monarch’s distributed single-render path. Existing Fleet has no pause, reorder, or individual retry API. Implement any required Studio queue behavior through explicitly new guarded orchestration, retaining the correct graph-task boundary and refusing unsupported in-flight mutations.

Expose the actual validation result, admission decision, execution identity, worker acknowledgements, failure, and recovery details available from Monarch. The existing render.fleet record is a capped last-queue snapshot with timing rows, not durable history or reliable per-job terminal status. Implement durable Studio records and reconcile actual execution events; an absent ended_s must not be interpreted as a specific terminal outcome. Display unavailable or stale workers and explain topology constraints before submission. Keep the requested topology and effective topology in job provenance. Never silently degrade to a different mode or target.

Operational actions reuse applicable plan/apply, consent, idle-maintenance, drain, and recovery guards. The complete management inventory includes version, diagnostics, launcher plan/check/apply, sol-install, sol-smoke, Doctor, init/setup, up/down/restart, reap, status, service installation, update, uninstall, top recording/replay, and gate inspection/repair. Expose typed review/apply/status adapters through the companion, preserving exact plan digests, receipts, acknowledgements, and validation. Doctor repairs remain allowlisted. Existing /dgxm/recycle resets client-owned actors while persistent workers remain running; stock /free has separate behavior. Never merge these into an ambiguous reset action. [CLI operation inventory](https://github.com/Deen-Media/dgx-monarch/blob/v1.2.0/src/dgx_monarch/cli/arguments.py).

Preserve refusal classes P (physics, no bypass), C (capacity, fitting alternatives first), U (unproven), and K (known-wrong, only specifically waivable). GUARDS.waivable_now governs actual consent availability. Preserve exact pending consent key/id, artifact/context PASS binding, quarantine, output stamps, rank capacity agreement, topology/dtype/scale/LoRA/FSDP guards, RenderSession and lifecycle locks, policy-stale checks, and native-return latches. Auto-rescue never authorizes class K. Poisoned transport requires driver restart. [Guard authority](https://github.com/Deen-Media/dgx-monarch/blob/v1.2.0/src/dgx_monarch/refusal.py).

## Telemetry and semantic consistency

Create one shared, versioned telemetry definition for metric IDs, names, units, source, aggregation, missing-value behavior, thresholds, freshness rules, and semantic colors. Apply it to Studio, CLI, TUI, and the ComfyUI panel. Presentation adapters may differ in layout, never in meaning. Runtime safety thresholds come from backend policy. Presentation thresholds and formatters belong in a separate shared presentation contract, never masquerading as safety guards.

Use the approved memory tokens: GPU #22d3ee, slab #34d399, retained pool #fb923c, anonymous #c084fc, other #64748b, and reclaimable cache #facc15; free memory remains uncolored. Display explicit labels and legends. Establish which categories overlap before drawing a stacked total, especially on shared-memory hardware; never add overlapping counters into a misleading physical-memory total.

Use the approved hex values in true-color terminals. Define the 256-color fallback indexes, in the same category order, as 45, 79, 208, 141, 244, and 220. Basic ANSI falls back to cyan, green, yellow, magenta, white, and yellow; monochrome uses labels and consistent ordering. Pool/cache remain distinguishable through labels when colors collapse. Share this mapping in generated presentation assets and test it. Treat unavailable as unknown, not zero. Use the same conversions and precision everywhere: memory GiB, rail throughput decimal GB/s, utilization %, power W, temperature °C, durations s, and clocks MHz or an explicitly converted GHz display. Replace ambiguous current G labels. Preserve source sampling intervals.

Close existing parity gaps actively: TUI lacks some render.last topology, auto-row, last-step, and attention-pin details shown by the panel, and currently combines first-host RX/TX where the panel separates directions per host/rail. Adopt complete shared fields. Reconcile the TUI-only >20 GiB available-memory presentation threshold with the panel through the shared contract. Use ready/idle/rendering/busy #4ade80, degraded/legacy-active #facc15, and blocked/unknown #f87171 status tokens with accessible icons and green/yellow/red terminal equivalents. Keep these distinct from branding lime. [Current TUI behavior](https://github.com/Deen-Media/dgx-monarch/blob/v1.2.0/docs/TUI.md).

Include Spark utilization, memory, temperatures, power, relevant worker metrics, and execution phases only where the backend supplies them. Show phase-duration timelines inspired by Grafana's inspectable histories, with measured start/end times and incomplete phases identified. Tooltips explain metric meaning and supported actions, without speculative diagnosis.

## Visual system and accessibility

Use charcoal #111318, soft white #f3f5f7, and selective lime #87f900 branding. Lime is a brand/action accent and must never substitute for a memory category. Define separate status, focus, selection, and warning tokens shared semantically across interfaces. Validate contrast rather than assuming supplied colors work for every text/background pairing.

Include a small Monarch butterfly activity mark as a lightweight SVG with CSS transforms. Animation is bounded, reflects actual activity, pauses when the tab is hidden, and becomes static with reduced-motion preferences. No decorative particle system, perpetual high-frequency canvas loop, or GPU-intensive effect.

Support keyboard-only operation, visible focus, screen-reader names, accessible dialogs, non-color status cues, scalable text, and reduced motion. Restore focus after dialogs and navigation. Shortcuts must not interfere with typing in prompt fields. Ensure chart data and warnings remain understandable without hover or animation.

## Workflow compatibility and App Mode

Import ComfyUI workflows with preserved metadata and graph identity. Official App Mode support starts at frontend version 1.41.13. Build a versioned compatibility adapter, with source-version fixtures and an internal stable binding contract, rather than depending on ComfyUI frontend stores at runtime. [Official App Mode guide](https://docs.comfy.org/interface/app-mode).

Interpret extra.linearData inputs as tuples of identity, widgetName, and optional configuration such as height and description; interpret outputs as node identities and preserve extra.linearMode. Current canonical widget identities use graphId:nodeId:name with encoded segments. Legacy node/widget pairs and subgraphs require graph-aware resolution. Never resolve by display label alone. [App Mode store](https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/src/stores/appModeStore.ts), [widget identity](https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/src/types/widgetId.ts), [graph resolution](https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/src/utils/litegraphUtil.ts).

Include a workflow configuration view for choosing prominent controls, editing display labels, grouping and ordering fields, and saving the layout with the workflow. Preserve stable binding identities through label changes and keep every supported advanced input reachable.

The internal binding records source identity, graph context, typed input, ordering, display configuration, validation, output mapping, and resolution status. Preserve unresolved bindings and unknown metadata in the imported record. Explain missing, ambiguous, deleted, or unsupported widgets and block unsafe execution instead of silently dropping a required input. Provide an explicit rebinding editor for supported types.

Test renames, deleted nodes, duplicate names, nested graphs, legacy IDs, custom widgets, and image/video outputs. Export/import must preserve the original workflow information required for continued graph editing. A supported custom widget needs an explicit typed adapter; arbitrary JavaScript widget parity and a drop-in Svelte SDK are not promised.

## Included design references

Include an Invoke-inspired browsable gallery and result-driven reuse; Grafana-inspired phase-duration inspection; a Linear-inspired command palette; and AUTOMATIC1111-inspired parameter grids, prompt styles, seed handling, and result-to-generation handoffs. These are concrete behaviors specified above, not decorative references.

Borrow SwarmUI's discoverable named workflow parameters, grouping, and dynamically available grid axes. Its SwarmInput mapping supplies useful examples of titles, types, defaults, ranges, descriptions, ordering, and groups. Its embedded ComfyUI integration and workflow/output transformations are not the coupling model for Studio. Its backend scheduling must not be described as equivalent to Monarch Fleet or distributed execution. [Advanced usage](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Advanced%20Usage.md), [product behavior](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Why%20Use%20Swarm.md), [multi-GPU behavior](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Using%20More%20GPUs.md), [workflow mapping](https://github.com/mcmonkeyprojects/SwarmUI/blob/master/src/BuiltinExtensions/ComfyUIBackend/Assets/comfy_workflow_editor_helper.js).

## Security and resource budgets

Use typed, allowlisted operations. Do not expose arbitrary shell, SSH, Python, or filesystem command execution. Reuse backend authorization and plan/apply guards, with explicit confirmation for destructive and administrative actions. No new permission grants are implied by installing Studio.

Preserve existing trusted local/LAN exposure defaults. Do not open public internet access automatically. Specify authentication, authorization, origin checks, CSRF protection where applicable, transport security expectations, input limits, safe artifact paths, and log redaction. Imported metadata, SVG, filenames, and prompts are untrusted content; never execute or render them as active markup.

Bound console retention by both line count and bytes, chart histories by time and sample count, thumbnail caches by bytes, and event replay by retention. Publish defaults and limits. Virtualize long lists, downsample chart rendering, release media/object URLs, and pause decorative/background work when hidden. Backend sampling must remain independent of browser redraw frequency.

Measure overhead on the Spark and on the browser client separately. Compare Studio closed, idle, generating, viewing video, and showing full telemetry/console against repeatable backend baselines. Record generation timings, memory, CPU/GPU load, network traffic, client responsiveness, and long-session growth. Set and enforce regression budgets using measured baseline variance; do not claim fastest performance or zero overhead without evidence.

## Integration dependencies and acceptance

Implement against a pinned repository revision. First establish the parity inventory, shared schemas, telemetry definitions, and regression fixtures. These unlock service adapters, packaging, and the interface components. Workflow adapters, persistence, gallery provenance, and Fleet integration consume those same contracts. Integrate and test the entire specified product together before declaring completion.

Acceptance requires all of the following:

1. Every supported inventory row is usable in Studio with the same defaults, guards, and backend result as existing interfaces. No supported feature requires switching UI.
2. Production assets run from the Python package without runtime Node; ComfyUI source remains unchanged. The companion remains usable through driver shutdown, update, and restart, preserves receipts, and stays optional.
3. Representative image, video, edit, upscale, custom-workflow, and Fleet jobs complete with verified outputs and immutable provenance.
4. Cross-interface tests confirm identical names, units, colors, thresholds, operation identities, preview behavior, and recovery decisions.
5. Queue edits, retries, disconnects, timeouts, duplicate events, restarts, and cancellation races never silently duplicate jobs or overwrite submitted settings.
6. Gallery reuse, LoRA stacks, presets, styles, comparisons, grids, projects, session restore, command palette, and presentation mode pass end-to-end tests.
7. App Mode fixtures cover supported frontend versions, legacy identities, graph changes, custom widgets, unresolved inputs, and image/video outputs without metadata loss.
8. Security tests cover unauthorized operations, invalid origins, path traversal, oversized input, untrusted metadata, and required confirmations.
9. Keyboard, screen-reader, contrast, reduced-motion, responsive-layout, and hidden-tab tests pass.
10. Bounded retention and measured performance budgets hold during long-running generation, telemetry, console, and media sessions.

Deliver the source, reproducible asset build, packaged installation, schemas and compatibility policy, completed parity matrix, fixtures, automated tests, operational documentation, and measured performance report. Record real incompatibilities and their supported resolution. A missing agreed feature is an incomplete deliverable, not an implied future phase.