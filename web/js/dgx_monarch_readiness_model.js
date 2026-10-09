// Pure readiness projection for the browser panel. This module has no DOM,
// ComfyUI, network, or mutation capability. It emits only fixed lifecycle
// rows; server-provided remediation stays outside the model, so no server
// reply can become a browser control.

const LAYERS = [
    ["worker_service", "Worker service"],
    ["attached_mesh", "Attached mesh"],
    ["render_session", "Render session"],
];
const OVERALL = new Set(["ready", "degraded", "blocked", "unknown"]);
const STATES = new Set(["ready", "idle", "active", "blocked", "unknown"]);

function plainObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : null;
}

function boundedText(value) {
    if (typeof value !== "string") return null;
    const text = value.replace(/\s+/g, " ").trim();
    return text ? text.slice(0, 400) : null;
}

function unknown(source, notice) {
    return {
        schemaVersion: null,
        source,
        overall: "unknown",
        notice,
        rows: LAYERS.map(([id, label]) => ({
            id, label, state: "unknown",
            detail: "Readiness telemetry is missing or malformed.",
        })),
    };
}

function requiredOverall(rows) {
    const states = rows.map((row) => row.state);
    if (states.includes("blocked")) return "blocked";
    if (states.includes("unknown")) return "unknown";
    if (states.includes("active")) return "degraded";
    return "ready";
}

function canonical(report) {
    if (!report || report.schema_version !== 1 || !OVERALL.has(report.overall)) return null;
    const lifecycle = plainObject(report.lifecycle);
    if (!lifecycle) return null;
    const rows = [];
    for (const [id, label] of LAYERS) {
        const layer = plainObject(lifecycle[id]);
        const state = layer?.state;
        const detail = boundedText(layer?.detail);
        if (!STATES.has(state) || detail === null) return null;
        rows.push({ id, label, state, detail });
    }
    const required = requiredOverall(rows);
    // The one accepted mismatch: a telemetry refresh error may report unknown
    // over ready or degraded rows. Any other is rejected, so no report can
    // look healthier than its rows.
    if (report.overall !== required
            && !(report.overall === "unknown" && ["ready", "degraded"].includes(required))) {
        return null;
    }
    return {
        schemaVersion: 1,
        source: "canonical",
        overall: report.overall,
        notice: report.overall === "unknown" && required !== "unknown"
            ? "Telemetry freshness could not be established."
            : null,
        rows,
    };
}

function legacyWorkers(value) {
    if (!Array.isArray(value) || !value.length) {
        return ["unknown", "No worker service observation is available."];
    }
    let blocked = false;
    let unknownState = false;
    for (const item of value) {
        const worker = plainObject(item);
        if (!worker || worker.status_error || worker.health_error || worker.error) {
            unknownState = true;
            continue;
        }
        let observed = false;
        for (const key of ["healthy", "up", "running", "listening"]) {
            if (!(key in worker)) continue;
            observed = true;
            if (typeof worker[key] !== "boolean") unknownState = true;
            else if (!worker[key]) blocked = true;
        }
        for (const [key, good, bad] of [["loop", "running", "stopped"],
                                       ["port", "open", "closed"]]) {
            if (!(key in worker)) continue;
            observed = true;
            const state = typeof worker[key] === "string" ? worker[key].trim().toLowerCase() : "";
            if (state === bad) blocked = true;
            else if (state !== good) unknownState = true;
        }
        // An attached actor response does not attest that the persistent
        // worker service itself is healthy and independently attachable.
        if (!observed) unknownState = true;
    }
    if (blocked) return ["blocked", "At least one worker service is not ready."];
    if (unknownState) return ["unknown", "Worker service health could not be confirmed."];
    const noun = value.length === 1 ? "service" : "services";
    return ["ready", `${value.length} worker ${noun} ready.`];
}

function count(value) {
    return Number.isInteger(value) && value >= 0 ? value : null;
}

function legacyMesh(value) {
    const mesh = plainObject(value);
    if (!mesh) return ["unknown", "Attached mesh state is unavailable."];
    const state = typeof mesh.state === "string" ? mesh.state.trim().toLowerCase() : null;
    const verdict = typeof mesh.verdict === "string" ? mesh.verdict.trim().toLowerCase() : null;
    const leases = count(mesh.active_leases ?? 0);
    const abandoned = count(mesh.abandoned_samples ?? 0);
    const abandonedLeases = count(mesh.abandoned_leases ?? 0);
    const busyPhase = mesh.busy_phase ?? "";
    const poisoned = mesh.poisoned ?? false;
    const replacement = mesh.replacement_blocked ?? false;
    const knownState = ["ok", "idle", "busy", "dirty", "poisoned", "unknown"].includes(state);
    const knownVerdict = ["none", "completed", "live", "blocked",
                          "unresolved", "unknown"].includes(verdict);
    if (state === null || verdict === null || leases === null || abandoned === null
            || abandonedLeases === null || typeof busyPhase !== "string"
            || typeof poisoned !== "boolean"
            || !(typeof replacement === "boolean" || typeof replacement === "string")
            || !knownState || !knownVerdict) {
        return ["unknown", "Attached mesh state is malformed."];
    }
    const blocked = ["dirty", "blocked", "poisoned"].includes(state)
        || ["blocked", "poisoned"].includes(verdict)
        || (verdict === "unresolved" && !busyPhase.trim())
        || abandoned > 0 || abandonedLeases > 0 || poisoned
        || replacement === true || (typeof replacement === "string" && replacement.trim());
    if (blocked) return ["blocked", "The attached mesh is not safe to reuse."];
    if (state === "unknown" || verdict === "unknown") {
        return ["unknown", "Attached mesh state is unavailable."];
    }
    if (state === "idle") {
        if (!["none", "completed"].includes(verdict) || busyPhase.trim()
                || (verdict === "none" && leases > 0)) {
            return ["unknown", "Attached mesh state is inconsistent."];
        }
        return ["idle", "No mesh is attached; the next render can attach one."];
    }
    if (state === "busy") {
        return verdict === "unresolved" && Boolean(busyPhase.trim())
            ? ["active", "The attached mesh has active work."]
            : ["unknown", "Attached mesh state is inconsistent."];
    }
    if (state === "ok" && verdict === "live" && !busyPhase.trim()) {
        return leases > 0
            ? ["active", "The attached mesh has active work."]
            : ["ready", "The attached mesh is ready for work."];
    }
    return ["unknown", "Attached mesh reports a state, verdict and busy phase that do not agree."];
}

function legacyRender(value) {
    const render = plainObject(value);
    if (!render) {
        return ["unknown", "Render session state is unavailable."];
    }
    const activeCount = render.active_renders === undefined
        ? null : count(render.active_renders);
    let active = render.active;
    if (active === undefined && activeCount !== null) active = activeCount > 0;
    if (typeof active !== "boolean" || (render.active_renders !== undefined
            && (activeCount === null || active !== (activeCount > 0)))) {
        return ["unknown", "Render session state is malformed."];
    }
    return active
        ? ["active", "A render session is active."]
        : ["idle", "No render session is active."];
}

function compatibility(payload) {
    const values = [legacyWorkers(payload.workers), legacyMesh(payload.mesh),
                    legacyRender(payload.render)];
    const rows = LAYERS.map(([id, label], index) => ({
        id, label, state: values[index][0], detail: values[index][1],
    }));
    const telemetryError = payload.telemetry_error !== undefined
        && payload.telemetry_error !== null && payload.telemetry_error !== false;
    return {
        schemaVersion: null,
        source: "compatibility",
        overall: telemetryError ? "unknown" : requiredOverall(rows),
        notice: "Compatibility view from older telemetry; restart ComfyUI so the server matches this panel.",
        rows,
    };
}

export function readinessPanelModel(payload) {
    const telemetry = plainObject(payload);
    if (!telemetry) return unknown("invalid", "Telemetry payload is missing or malformed.");
    if (!Object.prototype.hasOwnProperty.call(telemetry, "readiness")) {
        return compatibility(telemetry);
    }
    const model = canonical(plainObject(telemetry.readiness));
    return model || unknown("invalid", "Readiness payload is malformed; showing unknown state.");
}
