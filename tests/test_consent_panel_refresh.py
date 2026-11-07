"""The shipped consent panel, run under Node with a minimal DOM and a fake ComfyUI api.

Review F5: a poll that replaces a card or an active grant under the same key and
count must rebuild the row, so the next click names the id the driver serves now.
The model test covers row selection; this one drives web/js/dgx_monarch_consent.js
itself, its click closures included. It skips when node is not installed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

PANEL = Path(__file__).parents[1] / "web" / "js" / "dgx_monarch_consent.js"

KEY = "8b1f4c" + "0" * 26
GRANT = "c4e2a9" + "0" * 26


def card(pending_id: str, **fields) -> dict:
    return {"id": pending_id, "key": KEY, "kind": "rescue-slab", "class": "C", "style": "capacity",
            "title": "Load a_checkpoint.safetensors with slab residency",
            "artifact": "a_checkpoint.safetensors", "risk": "Stock residency cannot fit it.",
            "primary": {"label": "Load with slab residency", "action": "accept"},
            "dismissible": True, "occurrences": 1, "seen_at": 1754305327.4, **fields}


def grant(grant_id: str, source: str) -> dict:
    return {"key": GRANT, "id": grant_id, "kind": "rescue-slab", "artifact": "b.safetensors",
            "granted_at": "2026-08-04 13:22:07", "consent_source": source, "stale": False}


DRIVER = r"""
import { registerHooks } from 'node:module';
const steps = JSON.parse(process.argv[1]);
const panel = process.argv[2];
registerHooks({
  resolve(specifier, context, next) {
    const name = specifier.match(/\/scripts\/(app|api)\.js$/)?.[1];
    if (name) return { url: `data:text/javascript,export const ${name} = globalThis.fake_${name};`,
                       shortCircuit: true };
    return next(specifier, context);
  },
});
// Each node gets a serial when it is made, so a test can tell a kept button from a rebuilt one.
let made = 0;
class Node {
  constructor(tag) { Object.assign(this, { tag, children: [], style: {}, textContent: '', serial: ++made }); }
  appendChild(child) { this.children.push(...(child.tag === '#fragment' ? child.children : [child])); return child; }
  replaceChildren(...nodes) { this.children = []; for (const node of nodes) this.appendChild(node); }
}
globalThis.document = {
  createElement: (tag) => new Node(tag),
  createDocumentFragment: () => new Node('#fragment'),
};
let payload = null, poll = null;
const posted = [];
globalThis.setInterval = (run) => { poll = run; return 1; };
globalThis.fake_app = { registerExtension() {} };
globalThis.fake_api = {
  async fetchApi(path, options = {}) {
    if (options.method === 'POST') {
      posted.push(JSON.parse(options.body));
      return { ok: true, status: 200, json: async () => ({ ok: true }) };
    }
    return { ok: true, status: 200, json: async () => structuredClone(payload) };
  },
};
const { consentZones, startConsentPolling } = await import(panel);
const settle = async () => { for (let i = 0; i < 20; i++) await new Promise((r) => setTimeout(r, 0)); };
const walk = (node) => [node, ...node.children.flatMap(walk)];
const text = (zone) => walk(zone).map((node) => node.textContent).filter(Boolean).join(' | ');
const out = [];
for (const step of steps) {
  payload = step.payload;
  if (poll) poll(); else startConsentPolling();
  await settle();
  const { pending, active } = consentZones();
  // Read the zones before any click: a click re-renders them whatever the poll did.
  const shown = { pending: text(pending), active: text(active), buttons: {} };
  for (const [name, zone] of [['pending', pending], ['active', active]])
    for (const node of walk(zone).filter((node) => node.tag === 'button'))
      shown.buttons[`${name}:${node.textContent}`] = node.serial;
  const sent = posted.length;
  if (step.click) {
    const zone = step.click.zone === 'active' ? active : pending;
    const target = walk(zone).find((node) => node.tag === 'button' && node.textContent === step.click.label);
    target.onclick();
    await settle();
  }
  out.push({ ...shown, posted: posted.slice(sent) });
}
process.stdout.write(JSON.stringify(out));
"""


def _run_panel(steps: list[dict]) -> list[dict]:
    node = shutil.which("node")
    if node is None:  # pragma: no cover - environment dependent
        pytest.skip("node not installed; consent panel check skipped")
    probe = subprocess.run([node, "--input-type=module", "-e",
                            "import('node:module').then((m) => process.exit(m.registerHooks ? 0 : 1))"],
                           capture_output=True, timeout=30, check=False)
    if probe.returncode != 0:  # pragma: no cover - environment dependent
        pytest.skip("node lacks module.registerHooks (Node 22.15 or later); consent panel check skipped")
    result = subprocess.run(
        [node, "--input-type=module", "-e", DRIVER, json.dumps(steps), PANEL.as_uri()],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if result.returncode != 0:  # pragma: no cover - surfaced only on a real break
        raise AssertionError(f"node driver failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _payload(pending=(), active=()) -> dict:
    return {"pending": list(pending), "active": list(active), "auto_rescue": False}


def test_a_replaced_pending_card_submits_the_new_id():
    first, replaced = _run_panel([
        {"payload": _payload([card("p-old")])},
        # Another tab dismissed the card and the render raised it again: same key
        # and count, a fresh id.
        {"payload": _payload([card("p-new")]),
         "click": {"zone": "pending", "label": "Load with slab residency"}},
    ])
    assert "Load a_checkpoint.safetensors with slab residency" in first["pending"]
    assert replaced["posted"] == [{"action": "accept", "key": KEY, "id": "p-new"}]
    # The replaced card is a new node, so its button closes over the new id.
    button = "pending:Load with slab residency"
    assert replaced["buttons"][button] != first["buttons"][button]


def test_a_replaced_card_dismisses_the_new_id_and_shows_new_server_text():
    _first, changed = _run_panel([
        {"payload": _payload([card("p-1")])},
        {"payload": _payload([card("p-2", detail="The head holds about 9 GiB an earlier load left.")]),
         "click": {"zone": "pending", "label": "Dismiss"}},
    ])
    assert "The head holds about 9 GiB an earlier load left." in changed["pending"]
    assert changed["posted"] == [{"action": "dismiss", "key": KEY, "id": "p-2"}]


def test_a_regranted_active_row_shows_and_revokes_the_new_grant():
    _first, regranted = _run_panel([
        {"payload": _payload(active=[grant("c-old", "panel")])},
        {"payload": _payload(active=[grant("c-new", "auto")]),
         "click": {"zone": "active", "label": "Revoke"}},
    ])
    assert "rescue-slab \u00b7 b.safetensors \u00b7 auto" in regranted["active"]
    assert regranted["posted"] == [{"action": "revoke", "key": GRANT, "id": "c-new"}]


def test_an_unchanged_poll_keeps_the_live_nodes():
    """The signature still holds a button across idle polls, so a click is never lost mid-press."""
    first, same = _run_panel([
        {"payload": _payload([card("p-1")], [grant("c-1", "panel")])},
        {"payload": _payload([card("p-1")], [grant("c-1", "panel")]),
         "click": {"zone": "pending", "label": "Load with slab residency"}},
    ])
    # The poll before the click kept every button node: the same objects, not rebuilt copies.
    assert first["buttons"] and same["buttons"] == first["buttons"]
    assert set(first["buttons"]) >= {"pending:Load with slab residency", "pending:Dismiss", "active:Revoke"}
    assert same["posted"] == [{"action": "accept", "key": KEY, "id": "p-1"}]
