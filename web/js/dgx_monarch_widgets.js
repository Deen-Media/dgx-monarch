// Restore saved widget values after node schemas change.
// ComfyUI saves a positional array. Inserting a widget shifts later values:
// combos can reject them, but number and boolean widgets may silently accept them.
//
// DGX Monarch nodes also save dgxm_widgets, a name-to-value map that takes
// precedence on load. Older saves use the Init node's widget-order history:
// array length identifies each layout except disable_mmap (see INIT_ORDERS).
//
// tests/test_init_widget_order.py pins the latest layout to INPUT_TYPES and
// requires append-only widget additions.
import { app } from "../../scripts/app.js";

// Saved Init widget orders. Append new layouts by
// copying the last row and appending the new widget; keep arrays literal for
// the sync test.
//
// The length-10 layout also matches saves with disable_mmap.
// Renaming it to mmap_fallback preserved its position and BOOLEAN type but
// inverted its meaning. Migration copies the value unchanged: true selects
// the mmap loader now, whereas it selected direct reads in those old saves.
const INIT_ORDERS = [
    ["topology", "mode", "attention", "config_path", "gpus_per_host",
     "reserve_vram_gb", "sync_ulysses"],
    ["topology", "mode", "attention", "config_path", "gpus_per_host",
     "reserve_vram_gb", "sync_ulysses", "compile_dit"],
    ["topology", "mode", "attention", "config_path", "gpus_per_host",
     "reserve_vram_gb", "sync_ulysses", "compile_dit", "pipeline_depth"],
    ["topology", "mode", "attention", "config_path", "gpus_per_host",
     "reserve_vram_gb", "sync_ulysses", "compile_dit", "pipeline_depth",
     "mmap_fallback"],
    ["topology", "mode", "attention", "config_path", "gpus_per_host",
     "reserve_vram_gb", "sync_ulysses", "compile_dit", "pipeline_depth",
     "mmap_fallback", "lora_low_rss"],
    ["topology", "mode", "attention", "config_path", "gpus_per_host",
     "reserve_vram_gb", "sync_ulysses", "compile_dit", "pipeline_depth",
     "mmap_fallback", "lora_low_rss", "auto_gate", "swap_verify",
     "uma_reserve_gb"],
    ["topology", "mode", "attention", "config_path", "gpus_per_host",
     "reserve_vram_gb", "sync_ulysses", "compile_dit", "pipeline_depth",
     "mmap_fallback", "lora_low_rss", "slab_weights", "auto_gate",
     "swap_verify", "uma_reserve_gb"],
    ["topology", "mode", "attention", "config_path", "gpus_per_host",
     "reserve_vram_gb", "sync_ulysses", "compile_dit", "pipeline_depth",
     "mmap_fallback", "lora_low_rss", "slab_weights", "auto_gate",
     "load_profile", "swap_verify", "uma_reserve_gb"],
    ["topology", "mode", "attention", "config_path", "gpus_per_host",
     "reserve_vram_gb", "sync_ulysses", "compile_dit", "pipeline_depth",
     "mmap_fallback", "lora_low_rss", "slab_weights", "auto_gate",
     "load_profile", "swap_verify", "uma_reserve_gb", "comfy_managed"],
    ["topology", "mode", "attention", "config_path", "gpus_per_host",
     "reserve_vram_gb", "sync_ulysses", "compile_dit", "pipeline_depth",
     "mmap_fallback", "lora_low_rss", "slab_weights", "auto_gate",
     "load_profile", "swap_verify", "uma_reserve_gb", "comfy_managed",
     "family_adapter"],
];

// lora_low_rss was BOOLEAN before it gained "auto", so an old save holds
// true/false where the combo now wants "on"/"off". Returns undefined for a
// value the combo does not offer.
function coerce(widget, value) {
    const options = widget.options?.values;
    if (Array.isArray(options)) {
        if (options.includes(value)) return value;
        if (typeof value === "boolean") {
            const mapped = value ? "on" : "off";
            if (options.includes(mapped)) return mapped;
        }
        return undefined;
    }
    return value;
}

function widgetDefaults(nodeData) {
    const defaults = {};
    for (const group of ["required", "optional"]) {
        for (const [name, spec] of Object.entries(nodeData.input?.[group] ?? {})) {
            const [type, config] = Array.isArray(spec) ? spec : [spec, undefined];
            if (config && "default" in config) defaults[name] = config.default;
            else if (Array.isArray(type)) defaults[name] = type[0];
        }
    }
    return defaults;
}

// Fires once per Init node the legacy heal realigns, so a workflow with several
// gets one toast each; the log line names the node id.
function notifyMigration(node, detail) {
    console.info(`[dgx-monarch] ${node.type} #${node.id}: ${detail}`);
    const toast = app.extensionManager?.toast;
    if (toast && typeof toast.add === "function") {
        toast.add({
            severity: "info",
            summary: "DGX Monarch: widget values migrated",
            detail: `${detail} Save the workflow again to keep the fix.`,
            life: 8000,
        });
    }
}

app.registerExtension({
    name: "dgx_monarch.widget_drift",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!String(nodeData?.category ?? "").startsWith("DGX Monarch")) return;
        const defaults = widgetDefaults(nodeData);

        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (o) {
            onSerialize?.apply(this, arguments);
            if (!this.widgets?.length) return;
            o.dgxm_widgets = {};
            for (const w of this.widgets) o.dgxm_widgets[w.name] = w.value;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (o) {
            onConfigure?.apply(this, arguments);
            const widgets = this.widgets ?? [];
            if (!widgets.length) return;
            const byName = {};
            for (const w of widgets) byName[w.name] = w;

            const restore = (entries, why) => {
                const named = new Set();
                for (const [name, value] of entries) {
                    const w = byName[name];
                    if (!w) continue;
                    named.add(name);
                    const v = coerce(w, value);
                    if (v !== undefined) w.value = v;
                    // A value the combo does not offer (a removed option, say)
                    // may be a shifted value from the positional pass, so it
                    // falls back to the default when widgetDefaults has one.
                    else if (w.name in defaults) w.value = defaults[w.name];
                }
                // Widgets the save never knew about got shifted values from
                // the positional pass, so put their defaults back.
                for (const w of widgets) {
                    if (!named.has(w.name) && w.name in defaults) {
                        w.value = defaults[w.name];
                    }
                }
                if (why) notifyMigration(this, why);
            };

            const saved = o?.dgxm_widgets;
            if (saved && typeof saved === "object" && !Array.isArray(saved)) {
                restore(Object.entries(saved), null); // by-name is exact; stay quiet
                return;
            }
            // A save without the map: only the Init node has a layout history.
            if (nodeData.name !== "DGXMonarchInit") return;
            const arr = o?.widgets_values;
            if (!Array.isArray(arr) || arr.length === widgets.length) return;
            const order = INIT_ORDERS.find((names) => names.length === arr.length);
            if (!order) return;
            restore(order.map((name, i) => [name, arr[i]]),
                    `realigned ${arr.length} widget values from an older save.`);
        };
    },
});
