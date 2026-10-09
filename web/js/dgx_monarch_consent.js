// Render server-authored consent cards and submit the selected response.
// Poll /dgxm/consents throughout the app's lifetime, including while the panel
// is closed. The server supplies titles, risks, evidence, figures and button
// labels; new consent kinds need no client-side copy.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { consentPanelModel } from "./dgx_monarch_consent_model.js";
import { button, el } from "./dgx_monarch_ui.js";

const POLL_MS = 2500;
const GET_TIMEOUT_MS = 5000;
const POST_TIMEOUT_MS = 15000;
const CAPACITY = "#facc15";
const ACCURACY = "#f87171";
const GRANTED = "#4ade80";
const AUTO_KEY = "#auto_rescue";

let consentState = { payload: null, error: null };
let consentInFlight = false;
// Suppresses only this zone's re-render while a POST is in flight, so a card
// cannot be destroyed between mousedown and click. The panel's own refresh is
// never blocked: its progress bar and host cards keep updating through a click.
let consentBusy = 0;
let pollTimer = null;
let lastSignature = "";
let redrawHook = null;
let pendingZone = null;
let activeZone = null;
let activeSummary = null;
let activeBody = null;
let accuracyOpened = false;
// Keys arrive from the consent payload. Null-prototype maps keep special
// property names such as "__proto__" inert even against a malformed server.
const justAccepted = Object.create(null);
const busyState = Object.create(null);
const announced = new Set();

function short(error) {
    return String((error && error.message) || error).slice(0, 120);
}

// Preserve the full error below the controls: the server puts remedies last,
// where short()'s 120-character truncation could remove them.
function detail(error) {
    return String((error && error.message) || error).slice(0, 400);
}

function ensureZones() {
    if (pendingZone) return;
    // Stable container nodes: the panel appends these same elements every tick,
    // so root.replaceChildren() MOVES them instead of destroying live buttons.
    pendingZone = el("div", "");
    activeZone = el("details", "margin:6px 0;font-size:12px;");
    activeZone.style.display = "none";
    activeSummary = el("summary", "cursor:pointer;opacity:.85;", "Consents and waivers");
    activeBody = el("div", "padding:4px 0 2px;");
    activeZone.appendChild(activeSummary);
    activeZone.appendChild(activeBody);
}

export function consentZones() {
    ensureZones();
    return { pending: pendingZone, active: activeZone };
}

function localState(key) {
    if (!Object.hasOwn(busyState, key)) {
        busyState[key] = { busy: false, label: null, error: null };
    }
    return busyState[key];
}

async function postConsent(body) {
    const response = await api.fetchApi("/dgxm/consent", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-DGXM-Action": "consent" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(POST_TIMEOUT_MS),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || !payload.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    return payload;
}

const VERBS = {
    accept: ["allowing…", "allow"],
    dismiss: ["dismissing…", "dismiss"],
    revoke: ["revoking…", "revoke"],
    auto_rescue: ["saving…", "save the setting"],
};

async function answer(key, body, onDone) {
    const state = localState(key);
    if (state.busy) return;
    const verbs = VERBS[body.action] || VERBS.accept;
    state.busy = true;
    state.label = verbs[0];
    state.error = null;
    consentBusy += 1;
    renderZones(true);
    try {
        await postConsent(body);
        state.label = null;
        if (onDone) onDone();
    } catch (error) {
        state.label = null;
        state.error = `Could not ${verbs[1]}: ${detail(error)}`;
        // Tie the failure to this card's id. The driver gives the card a new id
        // when it raises it again after the old one was answered, expired or
        // dropped, and a stale error must not sit under the new card.
        state.labelId = body.id || null;
    } finally {
        state.busy = false;
        consentBusy -= 1;
        renderZones(true);
        refreshConsents();
    }
}

function cardNode(row) {
    const card = row.card;
    const accent = card.style === "accuracy" ? ACCURACY : CAPACITY;
    const box = el("div",
        `border:1px solid ${accent};border-radius:6px;padding:8px;margin:6px 0;font-size:12px;`);
    if (card.style === "accuracy") {
        box.appendChild(el("div", `font-size:11px;color:${ACCURACY};`, "⚠ accuracy waiver"));
    }
    box.appendChild(el("div", "font-weight:600;", String(card.title || "")));
    box.appendChild(el("div", "font-size:11px;opacity:.75;",
        `${card.artifact || ""}${card.numbers ? ` · ${card.numbers}` : ""}`));
    box.appendChild(el("div", "margin-top:4px;", String(card.risk || "")));
    if (card.evidence) {
        box.appendChild(el("div", "font-size:11px;opacity:.75;margin-top:2px;", String(card.evidence)));
    }
    if (card.measured) {
        box.appendChild(el("div", `font-size:11px;color:${ACCURACY};margin-top:2px;`, String(card.measured)));
    }
    const controls = el("div", "margin-top:6px;display:flex;gap:8px;align-items:center;");
    const primary = button(row.label || card.primary.label, accent);
    primary.disabled = row.busy;
    primary.onclick = () => answer(card.key, { action: "accept", key: card.key, id: card.id }, () => {
        const message = card.style === "accuracy"
            ? `Accuracy waiver recorded: ${card.title}`
            : `Allowed: ${card.title}`;
        justAccepted[card.key] = { card, at: Date.now(), message };
        announced.delete(card.id);
    });
    controls.appendChild(primary);
    if (card.dismissible) {
        const dismiss = button("Dismiss", null);
        dismiss.disabled = row.busy;
        dismiss.title = "Hide this card. Dismissing is not consent: nothing is saved and the "
            + "card returns the next time the render fails.";
        dismiss.onclick = () => answer(card.key, { action: "dismiss", key: card.key, id: card.id });
        controls.appendChild(dismiss);
    }
    if (card.occurrences > 1) {
        controls.appendChild(el("span", "font-size:11px;opacity:.6;", `asked ${card.occurrences} times`));
    }
    box.appendChild(controls);
    if (row.error) {
        box.appendChild(el("div", `font-size:11px;color:${ACCURACY};margin-top:4px;`, row.error));
    }
    return box;
}

function acceptedNode(row) {
    const accuracy = row.card?.style === "accuracy";
    const accent = accuracy ? ACCURACY : GRANTED;
    const box = el("div",
        `border:1px solid ${accent};border-radius:6px;padding:8px;margin:6px 0;font-size:12px;`);
    if (accuracy) {
        box.appendChild(el("div", `font-size:11px;color:${ACCURACY};`,
            "⚠ accuracy waiver; output remains rendered under known-wrong math"));
    }
    box.appendChild(el("div", "", row.message));
    const controls = el("div", "margin-top:6px;");
    const again = button("Queue this graph again", accent);
    // This queues the workflow open in the tab now, as the Queue button does;
    // no stored prompt is replayed.
    again.title = "Queues the workflow open in this tab. If you switched workflow tabs since "
        + "the render failed, switch back first.";
    again.onclick = async () => {
        if (typeof app.queuePrompt !== "function") {
            again.textContent = "Press Queue to run again.";
            return;
        }
        again.disabled = true;
        try {
            await app.queuePrompt(0, 1);
            again.textContent = "queued ✓";
        } catch (error) {
            again.textContent = `could not queue: ${short(error)}`;
            again.disabled = false;
        }
    };
    controls.appendChild(again);
    box.appendChild(controls);
    return box;
}

function activeRowNode(row) {
    const line = el("div",
        "display:flex;justify-content:space-between;align-items:center;font-size:11px;padding:2px 0;gap:8px;");
    // The context (contextLabel in the model) tells two grants for one
    // checkpoint apart: without it the ring2 and uly2+ring2 waivers read alike.
    const text = `${row.style === "accuracy" ? "⚠ " : ""}${row.kind} · ${row.artifact}`
        + (row.context ? ` · ${row.context}` : "")
        + (row.stale ? " · checkpoint changed since you allowed this" : "")
        + (row.grantedBy ? ` · ${row.grantedBy}` : "");
    const style = row.style === "accuracy" ? `color:${ACCURACY};` : "";
    line.appendChild(el("span", row.stale ? `opacity:.6;${style}` : style, text));
    const revoke = button(
        busyState[row.key] && busyState[row.key].label ? busyState[row.key].label : "Revoke",
        ACCURACY, true);
    revoke.disabled = row.busy;
    revoke.onclick = () => answer(row.key, { action: "revoke", key: row.key, id: row.id });
    line.appendChild(revoke);
    const failed = busyState[row.key] && busyState[row.key].error;
    if (!failed) return line;
    const wrap = el("div", "");
    wrap.appendChild(line);
    wrap.appendChild(el("div", `font-size:11px;color:${ACCURACY};`, failed));
    return wrap;
}

function toggleNode(model) {
    const wrap = el("div", "margin-top:6px;");
    const label = el("label", "display:flex;align-items:center;gap:6px;font-size:12px;");
    label.title = "When stock residency cannot fit a checkpoint, load it with slab residency "
        + "without asking. Every rescue is still byte-verified, and the ledger records each "
        + "checkpoint it rescues. Accuracy waivers are never automatic.";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = model.autoRescue;
    box.disabled = model.autoRescueBusy || model.autoRescueEnv;
    box.onchange = () => answer(AUTO_KEY, { action: "auto_rescue", value: box.checked });
    label.appendChild(box);
    label.appendChild(el("span", "", "Auto-allow capacity rescue (slab residency)"));
    wrap.appendChild(label);
    if (model.autoRescueEnv) {
        wrap.appendChild(el("div", "font-size:11px;opacity:.6;",
            "DGXM_AUTO_RESCUE turns this on for the driver process; the panel cannot turn it off"));
    }
    const state = busyState[AUTO_KEY];
    const note = state && !state.busy && (state.error || state.label);
    if (note) {
        wrap.appendChild(el("div", `font-size:11px;color:${ACCURACY};`, note));
    }
    return wrap;
}

function signature(model) {
    return JSON.stringify([
        model.rows.map((row) => [row.type, row.key || "", row.detail || "", row.label || "",
            row.error || "", row.busy ? 1 : 0, (row.card && row.card.occurrences) || 0]),
        model.active.map((row) => [row.key, row.stale ? 1 : 0, row.busy ? 1 : 0,
            (busyState[row.key] && busyState[row.key].label) || "",
            (busyState[row.key] && busyState[row.key].error) || ""]),
        model.autoRescue, model.autoRescueBusy, model.autoRescueEnv,
        (busyState[AUTO_KEY] && busyState[AUTO_KEY].label) || "",
        (busyState[AUTO_KEY] && busyState[AUTO_KEY].error) || "",
    ]);
}

function renderZones(force) {
    ensureZones();
    if (consentBusy > 0 && !force) return;
    const model = consentPanelModel(consentState, {
        justAccepted, busy: busyState, now: Date.now(),
    });
    const current = signature(model);
    // Nothing changed: leave the existing nodes alone so a live button keeps
    // its identity across polls.
    if (!force && current === lastSignature) return;
    lastSignature = current;

    const cards = document.createDocumentFragment();
    for (const row of model.rows) {
        if (row.type === "card") cards.appendChild(cardNode(row));
        else if (row.type === "accepted") cards.appendChild(acceptedNode(row));
        else if (row.type === "error") {
            cards.appendChild(el("div", "font-size:11px;opacity:.7;margin:6px 0;",
                `consent state unavailable: ${row.detail}`));
        }
    }
    pendingZone.replaceChildren(cards);

    activeZone.style.display = model.hasPayload ? "" : "none";
    // Expand when live accuracy waivers first appear so persistent grants are
    // visible. Do not reopen on every poll after the user collapses the section.
    if (model.accuracyActive && !accuracyOpened) {
        activeZone.open = true;
        accuracyOpened = true;
    } else if (!model.accuracyActive) {
        accuracyOpened = false;
    }
    const waived = model.accuracyActive
        ? ` · ${model.accuracyActive} accuracy waiver${model.accuracyActive > 1 ? "s" : ""}`
        : "";
    activeSummary.textContent = `Consents and waivers (${model.activeCount}${waived})`;
    activeSummary.style.color = model.accuracyActive ? ACCURACY : "";
    const body = document.createDocumentFragment();
    for (const row of model.active) body.appendChild(activeRowNode(row));
    if (!model.activeCount) {
        body.appendChild(el("div", "font-size:11px;opacity:.6;", "no active consents"));
    }
    body.appendChild(toggleNode(model));
    activeBody.replaceChildren(body);
}

function announce() {
    const payload = consentState.payload || {};
    const pending = Array.isArray(payload.pending) ? payload.pending : [];
    const live = new Set(pending.map((card) => card && card.id));
    for (const id of Array.from(announced)) if (!live.has(id)) announced.delete(id);
    for (const card of pending) {
        if (!card || !card.id || announced.has(card.id)) continue;
        announced.add(card.id);
        const toast = app.extensionManager?.toast;
        if (toast && typeof toast.add === "function") {
            // The toast may arrive before the card. Distinguish a class K
            // accuracy waiver from a routine capacity request.
            const accuracy = card.style === "accuracy";
            toast.add({
                severity: accuracy ? "error" : "warn",
                summary: accuracy
                    ? "DGX Monarch refused known-wrong math"
                    : "DGX Monarch needs one click",
                detail: accuracy
                    ? `${card.title}. Measured: ${card.measured || "see the panel"}. `
                      + "Waiving it stamps the output rendered-under-waiver."
                    : `${card.title}. Open the DGX Monarch panel to allow it.`,
                life: 10000,
            });
        }
    }
}

function forgetStaleLabels(payload) {
    // Clear the label and error once the card they were recorded against is
    // gone or has a new id (the driver raised it again after the old one was
    // answered, expired or dropped), so the button says what it does, not what
    // went wrong with the old card.
    const served = Object.create(null);
    for (const card of (payload && payload.pending) || []) served[card.key] = card.id;
    for (const [key, state] of Object.entries(busyState)) {
        if (!state || state.busy || !(state.label || state.error)) continue;
        if (state.labelId && served[key] !== state.labelId) {
            state.label = null;
            state.error = null;
            state.labelId = null;
        }
    }
}


async function refreshConsents() {
    if (consentInFlight) return;
    consentInFlight = true;
    try {
        const response = await api.fetchApi("/dgxm/consents", {
            signal: AbortSignal.timeout(GET_TIMEOUT_MS),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        consentState = { payload: await response.json(), error: null };
        forgetStaleLabels(consentState.payload);
    } catch (error) {
        // Keep the last good payload so a failed poll does not drop a card mid-read.
        consentState = { payload: consentState.payload, error: short(error) };
    } finally {
        consentInFlight = false;
    }
    announce();
    renderZones(false);
    if (typeof redrawHook === "function") {
        try {
            redrawHook();
        } catch (error) {
            console.warn("[dgx-monarch] consent redraw hook failed:", short(error));
        }
    }
}

export function consentPendingCount() {
    return consentPanelModel(consentState, { justAccepted, busy: busyState, now: Date.now() }).badge;
}

export function setConsentRedrawHook(hook) {
    redrawHook = typeof hook === "function" ? hook : null;
}

export function startConsentPolling() {
    if (pollTimer) return;
    ensureZones();
    refreshConsents();
    // Never cleared, like the panel's tab decoration: a user whose sidebar is
    // closed still gets the badge and the toast.
    pollTimer = setInterval(refreshConsents, POLL_MS);
}

app.registerExtension({
    name: "dgx_monarch.consent",
    setup() {
        startConsentPolling();
    },
});
