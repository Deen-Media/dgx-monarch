// Show driver fault and notice events as ComfyUI websocket toasts.
// See DESIGN.md sections 5.1 and 5.8 for the event contracts.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "dgx_monarch.faults",
    setup() {
        api.addEventListener("dgx-monarch.fault", (event) => {
            const message =
                (event.detail && event.detail.message) || String(event.detail || "unknown fault");
            const toast = app.extensionManager?.toast;
            if (toast && typeof toast.add === "function") {
                toast.add({
                    severity: "error",
                    summary: "DGX Monarch cluster fault",
                    detail: message,
                    life: 10000,
                });
            } else {
                console.error("[dgx-monarch] cluster fault:", message);
            }
        });

        let openNotice = null;
        api.addEventListener("dgx-monarch.notice", (event) => {
            const detail = event.detail || {};
            const message = detail.message || "";
            // Phases that opt out (the gate's 2 to 4 proof renders and its
            // cross-residency check) still reach the sidebar event tail,
            // `dgxm top` and the log; only the toast is skipped, so the gate
            // does not stack toasts on the canvas.
            if (!message || detail.toast === false) return;
            const toast = app.extensionManager?.toast;
            if (!toast || typeof toast.add !== "function") {
                console.info("[dgx-monarch]", message);
                return;
            }
            // A sticky notice (the identity gate runs for minutes) stays on
            // screen until the next shown notice replaces it.
            if (openNotice && typeof toast.remove === "function") {
                toast.remove(openNotice);
            }
            openNotice = null;
            // Not every notice is about the first render (queue-time graph
            // advice sends its own title); a payload with no summary, from an
            // older driver, gets the first-render title.
            const item = {
                severity: detail.severity || "info",
                summary: detail.summary || "DGX Monarch: first render",
                detail: message,
            };
            if (detail.sticky === true) {
                openNotice = item;
            } else {
                item.life = 15000;
            }
            toast.add(item);
        });
    },
});
