# Studio workflow compatibility fixtures

These are synthetic graph fixtures, not captured successful Comfy executions.
Their node schemas are provided by the tests. Unknown metadata deliberately
contains inert markup to ensure round trips do not discard extension data.

`app_mode_1_41_13.json` exercises the `[nodeId, widgetName]` input tuples used
by the [v1.41.13 App Mode store](https://github.com/Comfy-Org/ComfyUI_frontend/blob/v1.41.13/src/stores/appModeStore.ts).

`app_mode_canonical.json` exercises the canonical widget identity and optional
configuration contract inspected on 2026-10-19 in the upstream
[widget identity implementation](https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/src/types/widgetId.ts)
and [App Mode store](https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/src/stores/appModeStore.ts).
Its frontendVersion field is a fixture contract label, not a released version.

Tests derive deleted-node, duplicate-label, nested-graph, legacy-path,
unknown-widget, image/video-output, rebinding and cycle cases from these graphs.
Nested graphs are inspected and retained; executing subgraph serialization and
custom JavaScript widgets requires a separately validated adapter. They are
refused here. A frontend-exported API prompt is the supported execution input
for graphs whose UI-only transformations are not implemented by Studio.
