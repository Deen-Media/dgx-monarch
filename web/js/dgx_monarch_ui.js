// Shared sidebar element helper and button styles for the panel and consent zone.
//
// Classic-mode node widgets are painted on the graph canvas using shared global
// sizes and colors; these DOM styles cannot affect them.
//
// Keep ComfyUI imports and module-level DOM work out of this file: the frontend
// loads every WEB_DIRECTORY file as an extension module.

export function el(tag, style, text) {
    const node = document.createElement(tag);
    if (style) node.style.cssText = style;
    if (text !== undefined) node.textContent = text;
    return node;
}

// `accent` tints the border and the label; omit it for a neutral control.
// `compact` is the density for a button inside an 11px row (the active-consent
// list), where a full-size control would set the row height.
export function button(label, accent, compact) {
    const size = compact
        ? "padding:2px 8px;font-size:11px;"
        : "padding:4px 10px;font-size:12px;";
    return el("button",
        `${size}border-radius:4px;border:1px solid ${accent || "var(--border-color,#444)"};` +
        `background:var(--comfy-input-bg,#222);color:${accent || "inherit"};cursor:pointer;`,
        label);
}
