// Pure consent-panel model shared by the sidebar and Node.js tests.
// Keep display decisions here so tests/test_consent_panel_model.py covers them.
// No ComfyUI or DOM imports: this module must load in Node.js and the browser.
//
// The server supplies card content and actions; this model selects and orders
// rows. New consent kinds require no client-side copy.

export const ACCEPTED_TTL_MS = 15000;

function plainObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function own(object, key) {
    return Object.hasOwn(object, key) ? object[key] : undefined;
}

// consent_store.list_active says why each active row shows its context;
// Keep row-level Revoke consistent with docs/TROUBLESHOOTING.md #55.
function contextLabel(value) {
    return Object.entries(plainObject(value))
        .filter(([, v]) => typeof v === "string" && v)
        .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
        .map(([k, v]) => `${k} ${v}`)
        .join(", ");
}

function cardIsRenderable(card) {
    return Boolean(card && typeof card === "object" && card.key && card.id
        && card.primary && typeof card.primary.label === "string");
}

// state: { payload: <GET /dgxm/consents body>|null, error: string|null }
// ui:    { justAccepted: {key: {card, at, message}}, busy: {key: {busy, label, error}},
//          now: epoch ms, acceptedTtlMs }
export function consentPanelModel(state = {}, ui = {}) {
    const payload = plainObject(state.payload);
    const justAccepted = plainObject(ui.justAccepted);
    const busy = plainObject(ui.busy);
    const now = typeof ui.now === "number" && Number.isFinite(ui.now) ? ui.now : 0;
    const ttl = typeof ui.acceptedTtlMs === "number" && Number.isFinite(ui.acceptedTtlMs)
        ? ui.acceptedTtlMs
        : ACCEPTED_TTL_MS;

    // Accepted rows are local UI state, not server state: they must survive the
    // poll that drops the pending entry, or the user's confirmation is wiped
    // before it can be read.
    const accepted = [];
    const acceptedKeys = new Set();
    for (const [key, entry] of Object.entries(justAccepted)) {
        const record = plainObject(entry);
        const at = typeof record.at === "number" && Number.isFinite(record.at) ? record.at : 0;
        if (now - at > ttl) continue;
        acceptedKeys.add(key);
        accepted.push({
            type: "accepted",
            key,
            at,
            card: plainObject(record.card),
            message: typeof record.message === "string" && record.message
                ? record.message
                : "Allowed.",
        });
    }
    accepted.sort((a, b) => b.at - a.at);

    const cards = [];
    for (const card of Array.isArray(payload.pending) ? payload.pending : []) {
        if (!cardIsRenderable(card) || acceptedKeys.has(card.key)) continue;
        const local = plainObject(own(busy, card.key));
        cards.push({
            type: "card",
            key: card.key,
            card,
            busy: Boolean(local.busy),
            label: typeof local.label === "string" && local.label ? local.label : null,
            // The error stays out of the label: the button says what it does,
            // and the line under it says what went wrong and how to fix it.
            error: typeof local.error === "string" && local.error ? local.error : null,
        });
    }

    const rows = accepted.concat(cards);
    if (state.error) rows.push({ type: "error", detail: String(state.error) });
    if (!rows.length) rows.push({ type: "empty" });

    // Stale rows (checkpoint replaced since the grant) sort last: they are
    // history, and the fresh authorizations are what a user revokes.
    const active = (Array.isArray(payload.active) ? payload.active : [])
        .filter((row) => row && typeof row === "object" && row.key)
        .map((row) => ({
            key: String(row.key),
            id: typeof row.id === "string" ? row.id : "",
            kind: typeof row.kind === "string" ? row.kind : "consent",
            // The consent route adds `style` to active rows but falls back to
            // capacity for a kind it does not know, so a class K row is an
            // accuracy waiver either way.
            style: row.style === "accuracy" || row.class === "K" ? "accuracy" : "capacity",
            artifact: typeof row.artifact === "string" ? row.artifact : "",
            context: contextLabel(row.context),
            grantedAt: typeof row.granted_at === "string" ? row.granted_at : "",
            // The server sends `consent_source`. No driver sends `granted_by`;
            // only the Node.js check's fixtures use it. With neither, the row
            // shows nothing there rather than "undefined".
            grantedBy: typeof row.granted_by === "string"
                ? row.granted_by
                : (typeof row.consent_source === "string" ? row.consent_source : ""),
            stale: Boolean(row.stale),
            busy: Boolean(plainObject(own(busy, row.key)).busy),
        }));
    active.sort((a, b) => Number(a.stale) - Number(b.stale));

    return {
        rows,
        badge: cards.length,
        active,
        activeCount: active.length,
        // Live accuracy waivers; renderZones in dgx_monarch_consent.js says why
        // they open the section.
        accuracyActive: active.filter((row) => row.style === "accuracy" && !row.stale).length,
        autoRescue: Boolean(payload.auto_rescue) || Boolean(payload.auto_rescue_env),
        // DGXM_AUTO_RESCUE (the headless setting) wins over the stored toggle,
        // so the checkbox shows the effective state and says who set it.
        autoRescueEnv: Boolean(payload.auto_rescue_env),
        autoRescueBusy: Boolean(plainObject(own(busy, "#auto_rescue")).busy),
        hasPayload: Boolean(state.payload),
    };
}
