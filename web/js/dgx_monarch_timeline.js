// Record the last completed prompt's per-node timeline from ComfyUI events.
// Only the submitting tab receives these events; other tabs show no timeline.
// This module connects the event stream to dgx_monarch_timeline_model.js without
// adding routes or polling.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createTimelineRecorder } from "./dgx_monarch_timeline_model.js";

// The Fleet node is one node, so the recorder below gives it one row. Its
// per-job rows come from the driver instead, in the fleet block of the
// telemetry the panel already polls; this file re-exports their reader so the
// panel imports the whole timeline from one module.
export { fleetJobRows } from "./dgx_monarch_timeline_model.js";

const EVENT_END = ["execution_success", "execution_error", "execution_interrupted"];

let listening = false;

function nodeLabel(id) {
    try {
        const graph = app?.graph;
        const node = graph?.getNodeById?.(id) || graph?.getNodeById?.(Number(id));
        const label = node?.title || node?.type || node?.comfyClass;
        if (label) return String(label);
    } catch (e) {
        // a frontend that renames the graph accessor loses the label, not the row
    }
    return `node ${id}`;
}

const recorder = createTimelineRecorder({
    labelFor: nodeLabel,
    clock: () => performance.now(),
});

export function startTimelineRecording() {
    if (listening) return;
    listening = true;
    try {
        api.addEventListener("execution_start",
            (event) => recorder.open(event.detail?.prompt_id ?? null));
        api.addEventListener("executing", (event) => recorder.executing(event.detail));
        for (const name of EVENT_END) api.addEventListener(name, () => recorder.end());
    } catch (e) {
        listening = false;
    }
}

export function lastTimeline() {
    return recorder.last();
}
