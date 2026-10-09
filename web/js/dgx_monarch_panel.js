// dgx-monarch sidebar panel: cluster state inside ComfyUI.
// Reads /dgxm/telemetry without becoming a second monarch client. Its two
// explicit maintenance actions use ComfyUI's own origin and any login it enforces.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
    computeMemorySegments,
    selectMemoryHostCards,
} from "./dgx_monarch_segments.js";
import {
    consentPendingCount,
    consentZones,
    setConsentRedrawHook,
    startConsentPolling,
} from "./dgx_monarch_consent.js";
import {
    fleetJobRows,
    lastTimeline,
    startTimelineRecording,
} from "./dgx_monarch_timeline.js";
import { readinessPanelModel } from "./dgx_monarch_readiness_model.js";
import { button, el } from "./dgx_monarch_ui.js";
import { brandHeader, brandMark, updateBrand } from "./dgx_monarch_brand.js";

const POLL_MS = 2500;
let refreshInFlight = false;
const actionState = {
    free: { busy: false, label: "Free driver memory" },
    recycle: { busy: false, label: "Reset attached mesh" },
};

function dot(ok) {
    return el("span", `color:${ok ? "#4ade80" : "#f87171"};`, "●");
}

function legendItem(parent, color, label, title) {
    const swatch = el("span", `color:${color};`, "■");
    if (title) swatch.title = title;
    parent.appendChild(swatch);
    parent.appendChild(document.createTextNode(` ${label} `));
}

function bar(segments, total) {
    // segments: [[gib, color], ...]
    const wrap = el("div", "display:flex;height:10px;border-radius:5px;overflow:hidden;" +
        "background:var(--comfy-input-bg,#222);margin:4px 0;");
    for (const [gib, color] of segments) {
        const pct = Math.max(0, Math.min(100, (gib / total) * 100));
        if (pct > 0.5) wrap.appendChild(el("div", `width:${pct}%;background:${color};`));
    }
    return wrap;
}

function hostCard(name, stats, meta) {
    const card = el("div",
        "border:1px solid var(--border-color,#333);border-radius:6px;padding:8px;margin:6px 0;");
    const gpu = stats.gpu || {};
    const mm = meta?.memory || null;
    const { total, used, gpu: gpuMem, slab, pool, anon: anonSeg, other,
            reclaimable: reclaim, free } = computeMemorySegments(stats, meta);

    const title = el("div", "font-weight:600;display:flex;justify-content:space-between;");
    title.appendChild(el("span", "", name));
    title.appendChild(el("span", "opacity:.7;font-weight:400;",
        `used ${used.toFixed(1)} / ${total.toFixed(1)}G`));
    card.appendChild(title);
    // Six colored categories plus the background partition MemTotal exactly
    // (the accounting itself lives in dgx_monarch_segments.js).
    card.appendChild(bar([[gpuMem, "#22d3ee"], [slab, "#34d399"], [pool, "#fb923c"],
                          [anonSeg, "#c084fc"], [other, "#64748b"],
                          [reclaim, "#facc15"]], total));
    const legend = el("div", "font-size:11px;opacity:.75;");
    legendItem(legend, "#22d3ee", `gpu ${gpuMem.toFixed(1)}G (models+CUDA)`);
    legendItem(legend, "#34d399", `slab ${slab.toFixed(1)}G`);
    legendItem(legend, "#fb923c", `pool ${pool.toFixed(1)}G`,
        "Retained allocator pool: the load-staging high-water mark, kept after model unload. It returns " +
        "to the OS only when its process exits; the Reset attached mesh button ends worker actor processes.");
    legendItem(legend, "#c084fc", `anon ${anonSeg.toFixed(1)}G`);
    legendItem(legend, "#64748b", `other ${other.toFixed(1)}G`);
    legendItem(legend, "#facc15", `cache ${reclaim.toFixed(1)}G reclaimable`);
    legend.appendChild(el("span", "opacity:.6;", `□ free ${free.toFixed(1)}G`));
    card.appendChild(legend);
    if (gpu.util_pct !== undefined) {
        card.appendChild(el("div", "font-size:11px;opacity:.75;margin-top:2px;",
            `GPU ${gpu.util_pct}% · ${((gpu.clock_mhz || 0) / 1000).toFixed(2)} GHz · ` +
            `${gpu.power_w ?? "?"} W · ${gpu.temp_c ?? "?"}°C`));
    }
    if (mm) {
        const fails = mm.swap_verify_failures || 0;
        const line = el("div", "font-size:11px;margin-top:2px;");
        line.appendChild(document.createTextNode(`${mm.lora_mode || ""}` +
            (mm.unbake_file_backed_gib != null
                ? ` · unbake ${mm.unbake_file_backed_gib}G file-backed` : "") +
            " · verify fails "));
        line.appendChild(el("span", `color:${fails ? "#f87171" : "#4ade80"};`, String(fails)));
        card.appendChild(line);
    }
    return card;
}

function timelineRow(label, seconds, span) {
    // A fleet row passes a span on the driver's clock; a node row passes none.
    // An unknown wall prints as a dash: 0.00s would read as a measured zero.
    const line = el("div", "display:flex;justify-content:space-between;gap:8px;");
    line.appendChild(el("span",
        "overflow:hidden;text-overflow:ellipsis;white-space:nowrap;", label));
    if (span) line.appendChild(el("span", "opacity:.55;flex:0 0 auto;", span));
    const wall = Number.isFinite(seconds) ? `${seconds.toFixed(2)}s` : "-";
    line.appendChild(el("span", "opacity:.7;flex:0 0 auto;", wall));
    return line;
}

function timelineBody() {
    // The recorder only listens to ComfyUI's event stream. If it fails, only
    // these rows are lost; the 2.5 s refresh around them still draws.
    let recorded = null;
    try {
        recorded = lastTimeline();
    } catch (e) {
        recorded = null;
    }
    if (!recorded || !recorded.rows.length) {
        return el("div", "opacity:.6;",
            "no node timeline yet; this tab records the prompts it submits");
    }
    const box = el("div", "");
    for (const row of recorded.rows) box.appendChild(timelineRow(row.label, row.seconds));
    if (recorded.truncated) box.appendChild(el("div", "opacity:.6;", "... truncated"));
    return box;
}

function fleetBody(fleet) {
    // A Fleet node fans its jobs across every GPU but is one node to the
    // recorder above, so its per-job rows come from the driver's block
    // (fleetJobRows says what each row holds). When rows are missing, the
    // heading says how many of the wave's jobs it lists.
    let view = null;
    try {
        view = fleetJobRows(fleet);
    } catch (e) {
        view = null;
    }
    if (!view || !view.rows.length) return null;
    const box = el("div", "margin-top:4px;");
    const shown = view.rows.length;
    // Keep the timestamp visible because these rows outlive their render.
    const when = view.at ? ` ${new Date(view.at * 1000).toLocaleTimeString()}` : "";
    let heading = `fleet wave${when}`;
    if (view.truncated) heading += ` · first ${shown} of ${view.count}`;
    else if (shown < view.count) heading += ` · ${shown} of ${view.count}`;
    box.appendChild(el("div", "opacity:.6;", heading));
    for (const row of view.rows) {
        box.appendChild(timelineRow(row.label, row.seconds, row.span));
    }
    return box;
}

let detailsZone = null;
let detailsTimeline = null;
let detailsLedger = null;

function ensureDetailsZone() {
    if (detailsZone) return;
    // One element for the page's lifetime, like the consent zones: the panel
    // appends the same element every tick, so root.replaceChildren() moves it
    // rather than rebuilding it, and a block the operator opened stays open.
    detailsZone = el("details", "margin-top:6px;font-size:11px;");
    detailsZone.appendChild(
        el("summary", "cursor:pointer;opacity:.75;", "timeline and gate ledger"));
    detailsTimeline = el("div", "");
    detailsLedger = el("div", "margin-top:4px;opacity:.75;");
    detailsZone.appendChild(detailsTimeline);
    detailsZone.appendChild(detailsLedger);
}

function renderDetails(ledger, fleet) {
    // The node timeline and the full gate ledger, read after a render, sit in
    // a collapsed block, so the gate chips can hide without either becoming
    // unreachable.
    ensureDetailsZone();
    const jobs = fleetBody(fleet);
    detailsTimeline.replaceChildren(
        ...(jobs ? [timelineBody(), jobs] : [timelineBody()]));
    const combos = ledger?.combinations || {};
    const summary = Object.entries(combos).map(([verdict, n]) => `${verdict}:${n}`).join(" ");
    detailsLedger.textContent =
        `gate ledger: ${summary || "no combinations recorded"}` +
        ` · waivers:${ledger?.waivers || 0}`;
    return detailsZone;
}

function readinessCard(telemetry) {
    const model = readinessPanelModel(telemetry);
    const colors = {
        ready: "#4ade80", idle: "#4ade80", active: "#facc15",
        degraded: "#facc15", blocked: "#f87171", unknown: "#f87171",
    };
    const card = el("div", "border:1px solid var(--border-color,#333);border-radius:6px;" +
        "padding:8px;margin:6px 0;font-size:12px;");
    const heading = el("div", "font-weight:600;", "Readiness ");
    heading.appendChild(el("span", `color:${colors[model.overall]};`, model.overall.toUpperCase()));
    card.appendChild(heading);
    if (model.notice) card.appendChild(el("div", "opacity:.65;margin-top:2px;", model.notice));
    for (const row of model.rows) {
        const line = el("div", "display:grid;grid-template-columns:112px 70px 1fr;" +
            "gap:6px;margin-top:3px;");
        line.appendChild(el("span", "font-weight:600;", row.label));
        line.appendChild(el("span", `color:${colors[row.state]};`, row.state.toUpperCase()));
        line.appendChild(el("span", "opacity:.75;", row.detail));
        card.appendChild(line);
    }
    return card;
}

function replaceWithUnknown(root, zones) {
    // Telemetry and consent fail independently, and a wedged mesh is when a
    // rescue card matters most, so both consent zones stay on this path.
    const down = document.createDocumentFragment();
    down.appendChild(brandHeader(root));
    down.appendChild(el("div", "padding:10px;opacity:.7;",
        "telemetry unreachable or malformed; current readiness is unknown"));
    down.appendChild(zones.pending);
    down.appendChild(readinessCard(null));
    down.appendChild(zones.active);
    root.replaceChildren(down);
}

async function refresh(root) {
    if (refreshInFlight) return;
    refreshInFlight = true;
    const zones = consentZones();
    try {
        await refreshOnce(root, zones);
    } catch (e) {
        replaceWithUnknown(root, zones);
    } finally {
        refreshInFlight = false;
    }
}

async function refreshOnce(root, zones) {
    const resp = await api.fetchApi(
        "/dgxm/telemetry", { signal: AbortSignal.timeout(5000) });
    if (!resp.ok) throw new Error(`telemetry HTTP ${resp.status}`);
    const t = await resp.json();
    const frag = document.createDocumentFragment();

    frag.appendChild(brandHeader(root));
    const head = el("div", "padding:2px 0 6px;font-size:12px;");
    const render = t.render || {};
    head.appendChild(dot(true));
    head.appendChild(document.createTextNode(` driver · ComfyUI telemetry ${t.comfy || "?"}`));
    frag.appendChild(head);

    // The pending consent zone sits directly under the header, so a pending
    // card never needs a scroll. It renders nothing when nothing is pending.
    frag.appendChild(zones.pending);
    frag.appendChild(readinessCard(t));

    const r = el("div", "border:1px solid var(--border-color,#333);border-radius:6px;" +
        "padding:8px;margin:6px 0;font-size:12px;");
    if (render.active) {
        const step = render.step || 0, steps = render.steps || 1;
        r.appendChild(el("div", "font-weight:600;",
            `rendering ${render.model || ""}; step ${step}/${steps}` +
            (render.sec_per_step ? ` · ${render.sec_per_step}s/it` : "")));
        const track = el("div", "height:8px;border-radius:4px;background:var(--comfy-input-bg,#222);margin-top:4px;");
        track.appendChild(el("div",
            `height:100%;width:${Math.min(100, (step / steps) * 100)}%;background:#22d3ee;border-radius:4px;`));
        r.appendChild(track);
    } else if (render.last) {
        // A finished render keeps its model, steps and whether it paid a
        // first-use identity gate on screen, not only its wall time.
        const last = render.last;
        r.appendChild(el("div", "font-weight:600;",
            `last render · ${last.model || "model unknown"}`));
        const steps = last.steps ? `${last.steps} steps` : "steps unknown";
        r.appendChild(el("div", "opacity:.75;margin-top:2px;",
            `${last.wall_s ?? "?"}s · ${steps}` +
            (last.ceremony ? " · paid a first-use identity gate" : "")));
    } else {
        r.appendChild(el("div", "",
            "idle" + (render.last_wall_s ? ` · last render ${render.last_wall_s}s` : "")));
    }
    r.appendChild(renderDetails(t.ledger, render.fleet));
    frag.appendChild(r);

    for (const { name, stats, meta } of selectMemoryHostCards(t)) {
        frag.appendChild(hostCard(name, stats, meta));
    }

    // gates
    const combos = t.ledger?.combinations || {};
    const waivers = t.ledger?.waivers || 0;
    const verdicts = Object.entries(combos);
    const ceremonyRender = Boolean(render.ceremony) || Boolean(render.last?.ceremony);
    // An all-PASS ledger belongs to the render that paid for the ceremony, not
    // to every render after it, so it hides once a plain render has finished.
    // Any other verdict, or a waiver, always shows; the full ledger stays one
    // click away in the render block.
    const passOnly = verdicts.length > 0 && verdicts.every(([verdict]) => verdict === "PASS");
    const settled = passOnly && !waivers && Boolean(render.last) && !ceremonyRender;
    if ((verdicts.length || waivers) && !settled) {
        const g = el("div", "font-size:12px;margin:6px 0;");
        g.appendChild(document.createTextNode(
            ceremonyRender ? "gates for this render: " : "gates: "));
        for (const [verdict, count] of verdicts) {
            const color = verdict === "PASS" ? "#4ade80" : verdict === "FAIL" ? "#f87171" : "#facc15";
            g.appendChild(el("span", `color:${color};`, `${verdict}:${count}`));
            g.appendChild(document.createTextNode(" "));
        }
        // Waiver rows have their own ledger key namespace, so they are counted
        // beside the verdict chips, never inside one.
        if (waivers) g.appendChild(el("span", "color:#fb923c;", `waivers:${waivers}`));
        frag.appendChild(g);
    }

    // recent events
    const events = (t.events || []).slice(-5).reverse();
    if (events.length) {
        const box = el("div", "font-size:11px;opacity:.8;margin:6px 0;");
        for (const ev of events) {
            const when = new Date(ev.t * 1000).toLocaleTimeString();
            const body = Object.entries(ev).filter(([k]) => !["t", "kind"].includes(k))
                .map(([k, v]) => `${k}=${v}`).join(" ");
            box.appendChild(el("div", ev.kind === "quarantine" || ev.kind === "audit_fail" ||
                ev.kind === "uma_reserve_breach"
                ? "color:#f87171;" : "", `${when}  ${ev.kind}  ${body}`));
        }
        frag.appendChild(box);
    }

    // Active consents and waivers, plus the auto-rescue toggle: a collapsed
    // record, kept below the events and away from a card's one button.
    frag.appendChild(zones.active);

    // Actions use a consent card's row style, so the two button rows in this
    // column line up.
    const actions = el("div", "display:flex;gap:8px;align-items:center;margin:6px 0;");
    const freeBtn = button(actionState.free.label, null);
    freeBtn.disabled = actionState.free.busy;
    freeBtn.onclick = async () => {
        if (actionState.free.busy) return;
        actionState.free.busy = true;
        actionState.free.label = "freeing…";
        freeBtn.disabled = true;
        freeBtn.textContent = actionState.free.label;
        try {
            const response = await api.fetchApi("/free", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ unload_models: true, free_memory: true }),
                signal: AbortSignal.timeout(30000),
            });
            if (!response.ok) throw new Error(`free HTTP ${response.status}`);
            actionState.free.label = "free queued ✓";
        } catch (e) {
            actionState.free.label = "free failed";
        }
        freeBtn.textContent = actionState.free.label;
        setTimeout(() => {
            actionState.free = { busy: false, label: "Free driver memory" };
            refresh(root);
        }, 2000);
    };
    actions.appendChild(freeBtn);
    const recycleBtn = button(actionState.recycle.label, "#fb923c");
    recycleBtn.disabled = actionState.recycle.busy;
    recycleBtn.title = "Stops only the client-owned attached mesh and its model-holding " +
        "worker actor processes. Persistent DGX Monarch worker services stay running and " +
        "attachable. Loaded models and the retained pool return to the OS; the next render " +
        "attaches a fresh mesh. Active renders and result leases refuse the reset without " +
        "stopping the mesh.";
    recycleBtn.onclick = async () => {
        if (actionState.recycle.busy) return;
        if (!window.confirm(
                "Reset the attached mesh? Persistent DGX Monarch worker services stay " +
                "running and attachable. This stops only the client-owned mesh and its " +
                "model-holding actor processes, frees their models, and makes the next " +
                "render pay the full mesh/model startup cost. Active work is refused " +
                "without stopping the mesh.")) return;
        actionState.recycle.busy = true;
        actionState.recycle.label = "resetting mesh…";
        recycleBtn.disabled = true;
        recycleBtn.textContent = actionState.recycle.label;
        try {
            const r = await api.fetchApi("/dgxm/recycle", {
                method: "POST",
                headers: { "X-DGXM-Action": "recycle" },
                // The server's _RECYCLE_TIMEOUT_S (260 s) plus 10 s, so a
                // server-side 503 arrives before the client aborts unanswered.
                signal: AbortSignal.timeout(270000),
            });
            const body = await r.json().catch(() => ({}));
            if (r.ok && body.ok) {
                actionState.recycle.label = "mesh reset ✓";
            } else if (r.status === 409 && body.retryable &&
                       body.status === "active_work") {
                actionState.recycle.label = "render session active; wait";
            } else if (r.status === 409 && body.retryable &&
                       body.status === "lifecycle_busy") {
                actionState.recycle.label = "attached mesh changing; wait";
            } else if (r.status === 409 && body.retryable) {
                actionState.recycle.label = "attached mesh busy; wait";
            } else if (["prior_teardown_unknown", "proc_stop_failed",
                        "proc_stop_timed_out", "overall_timed_out"].includes(body.status)) {
                actionState.recycle.label = "mesh outcome unknown; inspect";
            } else if (r.ok && body.status === "no_live_mesh") {
                actionState.recycle.label = "no attached mesh";
            } else if (r.status === 429 && body.status === "rate_limited") {
                actionState.recycle.label = "reset already active; wait";
            } else if (r.ok) {
                actionState.recycle.label = "mesh response unknown; inspect";
            } else if (r.status >= 500) {
                actionState.recycle.label = "mesh outcome unknown; inspect";
            } else {
                actionState.recycle.label = "mesh reset refused";
            }
        } catch (e) {
            actionState.recycle.label = "request outcome unknown; inspect";
        }
        actionState.recycle.busy = false;
        recycleBtn.disabled = false;
        recycleBtn.textContent = actionState.recycle.label;
    };
    actions.appendChild(recycleBtn);
    frag.appendChild(actions);

    root.replaceChildren(frag);
}

let decorateTimer = null;

function decorateTabButton() {
    // Replace the tab button's contents while preserving its listeners and
    // aria-label tooltip. Reapply after theme or tab redraws; the selector
    // limits updates to this tab's buttons.
    const pending = consentPendingCount();
    for (const btn of document.querySelectorAll(
            'button[aria-label="DGX Monarch cluster status"]')) {
        // A presence check, not a flag: an in-place strip redraw can restore
        // the default icon children on the same button element.
        let stack = btn.querySelector("[data-dgxm-stack]");
        if (!stack) {
            btn.replaceChildren();
            stack = el("div",
                "display:flex;flex-direction:column;align-items:center;line-height:1.2;");
            stack.dataset.dgxmStack = "1";
            stack.appendChild(brandMark(btn, true));
            btn.appendChild(stack);
        }
        updateBrand(stack.querySelector("[data-dgxm-size]"), btn);
        // Write the pending-consent badge here, not from the consent poll: an
        // in-place strip redraw replaces this button's children, and only this
        // periodic run is sure to see it.
        let badge = stack.querySelector("[data-dgxm-badge]");
        if (pending > 0) {
            if (!badge) {
                badge = el("span", "font-size:9px;color:#facc15;");
                badge.dataset.dgxmBadge = "1";
                stack.appendChild(badge);
            }
            const count = String(pending);
            if (badge.textContent !== count) badge.textContent = count;
        } else if (badge) {
            badge.remove();
        }
    }
}

app.registerExtension({
    name: "dgx_monarch.panel",
    setup() {
        let timer = null;
        const manager = app.extensionManager;
        if (!manager?.registerSidebarTab) {
            console.warn("[dgx-monarch] sidebar API unavailable; panel disabled");
            return;
        }
        // Decorate on appearance and every 2 s afterwards. Keep this timer
        // through destroy(): closing the panel leaves the tab button visible.
        if (!decorateTimer) {
            requestAnimationFrame(decorateTabButton);
            setTimeout(decorateTabButton, 250);
            decorateTimer = setInterval(decorateTabButton, 2000);
        }
        // The consent poll also runs for the app's lifetime: the user who most
        // needs the badge and the toast has the panel closed. The call is
        // idempotent, so the consent module's own setup() may have started it.
        startConsentPolling();
        // Recording also runs for the app's lifetime, from page load, so the
        // timeline is there the first time the operator opens the panel.
        startTimelineRecording();
        setConsentRedrawHook(decorateTabButton);
        manager.registerSidebarTab({
            id: "dgx-monarch",
            icon: "pi pi-server",
            title: "DGX Monarch",
            tooltip: "DGX Monarch cluster status",
            type: "custom",
            render: (container) => {
                // Set the base font size once here: a block with no size of its
                // own would otherwise use the frontend default, which is larger
                // than every sized block around it.
                const root = el("div", "padding:8px;font-family:inherit;font-size:12px;");
                container.appendChild(root);
                refresh(root);
                if (timer) clearInterval(timer);
                timer = setInterval(() => refresh(root), POLL_MS);
            },
            destroy: () => {
                if (timer) clearInterval(timer);
                timer = null;
            },
        });
    },
});
