// Fixed diagnostic display only: measurements never establish readiness.
const PHASES = new Set(["attach", "setup", "queue_wait", "compute", "collect",
    "rdma_register", "rdma_read", "rdma_drop", "ack", "recycle", "spawn", "readiness", "cancel"]);
const OUTCOMES = new Set(["ok", "error", "timeout", "cancelled", "unknown"]);
const MODES = new Set(["none", "message", "rdma"]);

export function diagnosticsPanelModel(payload) {
    if (!payload || typeof payload !== "object" || payload.telemetry_error) {
        return { available: false, summary: "Phase timings unavailable or stale" };
    }
    const block = payload.diagnostics;
    if (!block || block.available !== true || block.schema !== 1) {
        return { available: false, summary: "Phase timings unavailable" };
    }
    const rows = (Array.isArray(block.recent) ? block.recent.slice(-4) : [])
        .filter(row => row && PHASES.has(row.phase) && OUTCOMES.has(row.outcome)
            && MODES.has(row.return_mode) && typeof row.duration_s === "number"
            && Number.isFinite(row.duration_s) && row.duration_s >= 0 && row.duration_s <= 86400)
        .map(row => `${row.phase.replaceAll("_", " ")} ${row.duration_s.toFixed(2)}s`
            + (row.outcome === "ok" ? "" : ` (${row.outcome})`));
    return { available: true, summary: rows.length
        ? `Recent driver phases: ${rows.join(" · ")}` : "No completed driver phases recorded" };
}
