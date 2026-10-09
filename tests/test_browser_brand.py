"""Exercise sidebar branding with a small DOM and the real panel setup."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

WEB = Path(__file__).parents[1] / "web" / "js"


def test_brand_theme_fallback_and_closed_panel_badge(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for browser behavior checks")
    (tmp_path / "package.json").write_text('{"type":"module"}')
    for name in ("dgx_monarch_ui.js", "dgx_monarch_brand.js", "dgx_monarch_panel.js"):
        source = (WEB / name).read_text()
        if name == "dgx_monarch_panel.js":
            import re

            source = re.sub(
                r'import \{[\s\S]*?\} from "([^"]+)";',
                lambda m: m.group() if m.group(1) in (
                    "./dgx_monarch_ui.js", "./dgx_monarch_brand.js"
                ) else m.group().replace(m.group(1), "./mocks.js"),
                source,
            )
        (tmp_path / name).write_text(source)
    (tmp_path / "mocks.js").write_text("""
export const app = {
    registerExtension(extension) { globalThis.extension = extension; },
    extensionManager: { registerSidebarTab(tab) { globalThis.tab = tab; } }
};
export const api = { fetchApi: async () => ({ ok: false, status: 503 }) };
export const consentPendingCount = () => globalThis.pending;
export const consentZones = () => ({ pending: document.createElement('div'), active: document.createElement('div') });
export const setConsentRedrawHook = hook => { globalThis.redraw = hook; };
export const startConsentPolling = () => {};
export const startTimelineRecording = () => {};
export const computeMemorySegments = () => ({});
export const selectMemoryHostCards = () => [];
export const readinessPanelModel = () => ({ state: 'unknown', title: 'Unknown', rows: [] });
export const fleetJobRows = () => [];
export const lastTimeline = () => null;
""")
    driver = r"""
import assert from 'node:assert/strict';
class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; this.style = {}; this.attributes = {}; this.textContent = ''; }
    appendChild(child) { child.parent = this; this.children.push(child); return child; }
    replaceChildren(...children) { this.children = []; children.forEach(child => this.appendChild(child)); }
    setAttribute(key, value) { this.attributes[key] = value; }
    querySelector(selector) {
        const match = child => selector === 'img' ? child.tag === 'img' :
            Object.hasOwn(child.dataset, selector.slice(6, -1).replace(/-([a-z])/g, (_, c) => c.toUpperCase()));
        for (const child of this.children) { if (match(child)) return child; const found = child.querySelector(selector); if (found) return found; }
        return null;
    }
    remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
}
const button = new Element('button');
button.setAttribute('aria-label', 'DGX Monarch cluster status');
const click = () => 42;
button.onclick = click;
globalThis.document = {
    createElement: tag => new Element(tag),
    createDocumentFragment: () => new Element('fragment'),
    createTextNode: text => Object.assign(new Element('text'), { textContent: text }),
    querySelectorAll: selector => selector === 'button[aria-label="DGX Monarch cluster status"]' ? [button] : []
};
let color = 'rgb(230, 230, 230)';
globalThis.getComputedStyle = () => ({ color });
globalThis.requestAnimationFrame = callback => callback();
globalThis.setTimeout = () => 1;
const intervals = new Map();
let serial = 0;
globalThis.setInterval = callback => { intervals.set(++serial, callback); return serial; };
globalThis.clearInterval = id => intervals.delete(id);
globalThis.pending = 2;
const brand = await import('./dgx_monarch_brand.js');
await import('./dgx_monarch_panel.js');
extension.setup();
assert.equal(tab.tooltip, 'DGX Monarch cluster status');
assert.equal(button.onclick, click);
assert.equal(button.attributes['aria-label'], tab.tooltip);
const mark = button.querySelector('[data-dgxm-size]');
const image = mark.querySelector('img');
assert.ok(image.src.endsWith('/brand/monarch-small-dark.svg'));
assert.equal(image.alt, '');
assert.equal(mark.attributes['aria-hidden'], 'true');
assert.equal(button.querySelector('[data-dgxm-badge]').textContent, '2');
color = 'rgb(30, 30, 30)';
redraw();
assert.ok(image.src.endsWith('/brand/monarch-small.svg'));
image.onerror();
assert.equal(image.hidden, true);
assert.equal(mark.children[1].hidden, false);
assert.equal(mark.children[1].textContent, 'DGX');
image.onload();
assert.equal(mark.children[1].hidden, true);
assert.equal(image.hidden, false);
const header = brand.brandHeader(button);
assert.equal(header.children[1].children[0].textContent, 'DGX Monarch');
assert.ok(header.querySelector('img').src.endsWith('/brand/monarch.svg'));
tab.destroy();
pending = 3;
for (const tick of intervals.values()) tick();
assert.equal(button.querySelector('[data-dgxm-badge]').textContent, '3');
button.replaceChildren();
redraw();
assert.equal(button.querySelector('[data-dgxm-badge]').textContent, '3');
assert.ok(button.querySelector('img'));
pending = 0;
redraw();
assert.equal(button.querySelector('[data-dgxm-badge]'), null);
assert.equal(button.onclick, click);
"""
    script = tmp_path / "driver.js"
    script.write_text(driver)
    result = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, json.dumps({"stdout": result.stdout, "stderr": result.stderr})
