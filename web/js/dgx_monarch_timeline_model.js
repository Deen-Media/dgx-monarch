// Convert ComfyUI event details into per-node timeline rows.
// Keep this model independent of ComfyUI so Node.js tests can exercise it.

// A runaway subgraph must not grow the tab's heap: rows past this are dropped
// and the tail is marked truncated.
export const MAX_ROWS = 250;

export function executingNodeId(detail) {
    // ComfyUI's frontend dispatches `executing` with the node id itself as the
    // detail (`display_node || node`), and null when the prompt's last node is
    // done, so reading detail.node alone would record nothing. A message-shaped
    // detail is read too, in case a later frontend sends the whole message.
    if (detail === null || detail === undefined) return null;
    if (typeof detail === "object") {
        const id = detail.display_node ?? detail.node;
        return id === null || id === undefined ? null : id;
    }
    return detail;
}

export function jobSpan(started, ended) {
    // Start and end in seconds from the moment the wave opened, on the
    // driver's clock, so rows from different boxes never compare two clocks.
    // A start with no end was dispatched and nothing came back: it is still
    // running, or a Stop cancelled it, and the span claims no more than that.
    if (!Number.isFinite(started)) {
        return Number.isFinite(ended) ? `to +${ended.toFixed(2)}s` : "";
    }
    if (!Number.isFinite(ended)) return `+${started.toFixed(2)}s`;
    return `+${started.toFixed(2)}s to +${ended.toFixed(2)}s`;
}

export function fleetJobRows(block) {
    // Convert telemetry render.fleet into rows in dispatch order. Retain both
    // rank-derived labels and responding host names so rows match memory cards.
    // Keep unanswered jobs as no-result rows, including jobs stopped mid-fleet.
    // host_reported identifies replies from a host other than the destination.
    if (!block || typeof block !== "object") return null;
    const jobs = Array.isArray(block.jobs) ? block.jobs : [];
    const rows = [];
    for (const job of jobs) {
        if (!job || typeof job !== "object") continue;
        const index = Number(job.job);
        const host = typeof job.host === "string" ? job.host : "";
        const box = String(job.box || host || "box unknown");
        // A job with no reply has no wall. Do not coerce with Number():
        // Number(null) is 0, which the panel would print as a measured 0.00s.
        // An unknown wall stays null and the panel prints a dash.
        const seconds = Number.isFinite(job.wall_s) ? job.wall_s : null;
        const started = Number.isFinite(job.started_s) ? job.started_s : null;
        const ended = Number.isFinite(job.ended_s) ? job.ended_s : null;
        const open = started !== null && ended === null;
        const elsewhere = typeof job.host_reported === "string"
            && job.host_reported !== "";
        rows.push({
            label: `job ${Number.isFinite(index) ? index + 1 : "?"} · ${box}`
                + (host && host !== box ? ` · ${host}` : "")
                + (job.prompt ? ` · ${job.prompt}` : "")
                + (elsewhere ? " · answered from another box" : "")
                + (open ? " · no result" : ""),
            box,
            host,
            elsewhere,
            started,
            ended,
            span: jobSpan(started, ended),
            seconds,
        });
    }
    const count = Number(block.count);
    const at = Number(block.t);
    return {
        rows,
        count: Number.isFinite(count) ? count : rows.length,
        truncated: Boolean(block.truncated),
        // When the wave opened; the panel's fleet heading shows it.
        at: Number.isFinite(at) && at > 0 ? at : null,
    };
}

export function createTimelineRecorder(options) {
    const settings = options || {};
    const labelFor = settings.labelFor || ((id) => `node ${id}`);
    const clock = settings.clock || (() => Date.now());
    const limit = settings.maxRows || MAX_ROWS;
    let recording = null;   // the prompt executing now
    let finished = null;    // the last prompt that ended

    function open(promptId) {
        recording = { promptId: promptId ?? null, rows: [], truncated: false };
    }

    function closeOpenRow(at) {
        const rows = recording ? recording.rows : null;
        const row = rows && rows.length ? rows[rows.length - 1] : null;
        if (row && row.seconds === null) row.seconds = (at - row.t0) / 1000;
    }

    function executing(detail) {
        const at = clock();
        // `executing` carries no prompt id, so a node event that arrives with
        // nothing open (execution_start missed on a reconnect) opens its own.
        if (!recording) open(null);
        closeOpenRow(at);
        const id = executingNodeId(detail);
        if (id === null) return;
        if (recording.rows.length >= limit) {
            recording.truncated = true;
            return;
        }
        recording.rows.push({ label: labelFor(id), t0: at, seconds: null });
    }

    function end() {
        if (!recording) return;
        closeOpenRow(clock());
        if (recording.rows.length) finished = recording;
        recording = null;
    }

    // The last completed prompt: {truncated, rows: [{label, seconds}]}.
    function last() {
        if (!finished) return null;
        return {
            truncated: finished.truncated,
            rows: finished.rows.map((row) => ({
                label: row.label,
                seconds: row.seconds === null ? 0 : row.seconds,
            })),
        };
    }

    return { open, executing, end, last };
}
