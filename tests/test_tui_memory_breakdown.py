"""The memory bar draws for every host whose report cannot hide a slab.

Workers report slab_gib only while a slab holds the model, so stock and
comfy-managed residency, idle Workers, driver-only boxes and dgxm top's local
fallback used to lose the bar and the retained-pool gauge. A Worker that sent
memory_error, or slab residency without a size, stays unknown.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dgx_monarch.studio.presentation import SLAB_RULE, presentation_contract, slab_reading
from dgx_monarch.tui import data as tui_data
from dgx_monarch.tui.app import DgxmTopApp
from dgx_monarch.tui.presentation import breakdown_slab

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web/js"

STATS = {
    "host": "node-0001", "t": 1000.0,
    "mem_gib": {"MemTotal": 121.6, "MemAvailable": 60.0, "MemFree": 40.0, "AnonPages": 30.0,
                "Shmem": 13.0, "Cached": 20.0, "Buffers": 0.1},
    "gpu_proc_gib": 25.0, "pool_gib": 4.0,
    "gpu": {"util_pct": 0, "clock_mhz": 2100, "power_w": 10.0, "temp_c": 40},
}
BASE = {"safetensors_backend": "pread", "aimdo_enabled": True, "disable_mmap": False}
RESIDENT = {**BASE, "patcher_class": "ModelPatcher", "model": "sample-model", "lora_mode": "n/a",
            "swap_verify_failures": 0}

# (case, Worker telemetry entry or None for a host no Worker reports, bar drawn)
CASES = [
    ("slab resident", {"memory": {**RESIDENT, "weight_residency": "slab", "slab_gib": 12.0}}, True),
    ("stock cudaMalloc resident", {"memory": {**RESIDENT, "weight_residency": "cudaMalloc"}}, True),
    ("comfy managed resident", {"memory": {**RESIDENT, "weight_residency": "comfy_managed"}}, True),
    ("idle Worker", {"memory": dict(BASE)}, True),
    ("driver-only host", None, True),
    ("memory_error", {"memory_error": "RuntimeError: memory detail failed"}, False),
    ("no memory report", {}, False),
    ("slab residency without a size", {"memory": {**RESIDENT, "weight_residency": "slab"}}, False),
    ("non-finite slab size", {"memory": {**RESIDENT, "weight_residency": "slab", "slab_gib": None}}, False),
    ("resident model without residency", {"memory": dict(RESIDENT)}, False),
]


def _meta(worker):
    """What tui/data.py keeps for a Worker host, or None for a host no Worker reports."""
    if worker is None:
        return None
    return {"vram": None, "memory": worker.get("memory"), "memory_error": worker.get("memory_error"),
            "nccl": None, "models": None}


@pytest.mark.parametrize(("case", "worker", "drawn"), CASES, ids=[case for case, _, _ in CASES])
def test_slab_rule_draws_a_known_zero_and_keeps_a_hidden_slab_unknown(case, worker, drawn):
    slab = slab_reading(_meta(worker))
    assert (slab is not None) is drawn
    if case == "slab resident":
        assert slab == 12.0
    elif drawn:
        assert slab == 0.0
    assert (breakdown_slab(STATS, _meta(worker)) is not None) is drawn


def test_missing_host_counters_still_hide_the_breakdown():
    stats = {**STATS, "mem_gib": {k: v for k, v in STATS["mem_gib"].items() if k != "AnonPages"}}
    assert breakdown_slab(stats, None) is None


def _telemetry(workers):
    """A /dgxm/telemetry payload in the route's shape, one Worker host per entry."""
    rows = []
    for index, worker in enumerate(workers):
        if worker is None:
            continue
        rows.append({"host": {**STATS, "host": f"node-{index + 1:04d}"}, **worker})
    return {"t": 1000.0, "comfy": "sample", "driver_host": {**STATS, "host": "node-driver"},
            "render": {"active": False}, "ledger": {}, "events": [], "workers": rows}


def _draw(monkeypatch, telemetry):
    poller = tui_data.Poller(loops=())
    monkeypatch.setattr(poller, "_driver_telemetry", lambda: telemetry)
    tick = poller.tick()
    return tick, DgxmTopApp().draw_hosts(tick).plain


@pytest.mark.parametrize(("case", "worker", "drawn"), CASES, ids=[case for case, _, _ in CASES])
def test_dgxm_top_draws_the_bar_per_case_through_the_real_poller(monkeypatch, case, worker, drawn):
    tick, text = _draw(monkeypatch, _telemetry([worker]))
    lines = text.splitlines()
    # The driver's own box has no Worker report, so it always draws.
    assert any(line.lstrip().startswith("iver [") for line in lines)
    if worker is None:
        assert "0001" not in text
        return
    host = next(line for line in lines if line.lstrip().startswith("0001"))
    if drawn:
        assert "[" in host and "used" in host and "breakdown unknown" not in host
        assert "pool 4.0 GiB (retained)" in text
    else:
        assert "memory breakdown unknown" in host and "[" not in host
    if case == "memory_error":
        assert tick["workers"]["node-0001"]["memory_error"] == "RuntimeError: memory detail failed"


def test_driver_only_box_beside_a_worker_box_draws_both_bars(monkeypatch):
    _, text = _draw(monkeypatch, _telemetry([{"memory": {**RESIDENT, "weight_residency": "cudaMalloc"}}]))
    rows = [line for line in text.splitlines() if "used" in line]
    assert len(rows) == 2 and all("[" in row for row in rows)
    assert "breakdown unknown" not in text


def test_driver_down_local_fallback_draws_the_bar_and_pool_gauge(monkeypatch):
    from dgx_monarch import telemetry

    monkeypatch.setattr(telemetry, "host_stats", lambda: dict(STATS))
    tick, text = _draw(monkeypatch, None)
    assert tick["driver_up"] is False and tick["hosts"]["node-0001"]["source"] == "local"
    assert "[" in text and "pool 4.0 GiB (retained)" in text
    assert "breakdown unknown" not in text


def test_contract_row_carries_the_slab_rule():
    metrics = presentation_contract()["metrics"]
    slab = next(m for m in metrics if m["id"] == "host.memory.slab")
    assert slab["missing"] == "slab_rule" and slab["missing_rule"] == SLAB_RULE
    assert all(m["missing"] == "unknown" for m in metrics if m is not slab)


def _node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the browser panel checks")
    return node


def test_browser_slab_rule_matches_the_python_rule():
    metas = [_meta(worker) for _, worker, _ in CASES] + [
        [], "text", {"memory": []}, {"memory": {"slab_gib": -1}}, {"memory": {"slab_gib": True}},
        {"memory": {"weight_residency": 3}}, {"memory_error": None, "memory": dict(BASE)}]
    module = WEB / "dgx_monarch_presentation.js"
    program = (f"import {{ slabReading }} from {json.dumps(module.as_uri())};\n"
               f"process.stdout.write(JSON.stringify({json.dumps(metas)}.map(slabReading)));")
    result = subprocess.run([_node(), "--input-type=module", "-e", program], capture_output=True,
                            text=True, timeout=30, check=True)
    assert json.loads(result.stdout) == [slab_reading(meta) for meta in metas]


def test_studio_chart_slab_rule_matches_the_python_rule():
    """Studio's own metric history applies the rule the served contract names."""
    assert presentation_contract()["metrics"] and any(
        metric.get("missing") == "slab_rule" for metric in presentation_contract()["metrics"])
    workers = [worker for _, worker, _ in CASES if worker is not None] + [
        [], "text", {"memory": []}, {"memory": {"slab_gib": -1}}, {"memory": {"slab_gib": True}},
        {"memory": {"weight_residency": 3}}, {"memory_error": None, "memory": dict(BASE)}]
    module = ROOT / "studio/src/telemetryHistory.ts"
    program = (f"import {{ slabReading }} from {json.dumps(module.as_uri())};\n"
               f"const values = {json.dumps(workers)}.map(slabReading);\n"
               "values.push(slabReading(undefined));\n"
               "process.stdout.write(JSON.stringify(values));")
    result = subprocess.run([_node(), "--input-type=module", "-e", program], capture_output=True,
                            text=True, timeout=30, check=True)
    # The Python rule reads tui/data.py's meta; Studio reads the Worker entry itself.
    expected = [slab_reading(_meta(worker) if isinstance(worker, dict) else worker) for worker in workers]
    assert json.loads(result.stdout) == [*expected, slab_reading(None)]


PANEL_DRIVER = r"""
class Node {
    constructor(tag) { this.tag = tag; this.children = []; this.style = { cssText: "" }; this.dataset = {};
                       this.attributes = {}; this.own = ""; this.title = ""; }
    get textContent() { return this.own + this.children.map((child) => child.textContent).join(""); }
    set textContent(value) { this.own = String(value); this.children = []; }
    appendChild(child) { this.children.push(child); return child; }
    replaceChildren(...children) { this.children = []; children.forEach((child) => this.appendChild(child)); }
    setAttribute(key, value) { this.attributes[key] = value; }
    querySelector() { return null; }
    addEventListener() {}
    remove() {}
}
globalThis.document = {
    createElement: (tag) => new Node(tag), createDocumentFragment: () => new Node("#fragment"),
    createTextNode: (text) => Object.assign(new Node("#text"), { own: String(text) }),
    querySelectorAll: () => [],
};
globalThis.window = globalThis;
globalThis.getComputedStyle = () => ({ color: "rgb(230, 230, 230)" });
globalThis.payloads = JSON.parse(process.argv[2]);
const { refreshOnce } = await import("./web/js/dgx_monarch_panel.js");
const cards = (node, found = []) => {
    const title = node.children[0];
    if ((title?.style.cssText || "").startsWith("font-weight:600;display:flex;justify-content:space-between;"))
        found.push({ name: title.children[0].textContent,
                     bar: node.children.some((child) => child.style.cssText.includes("height:10px")),
                     text: node.textContent });
    node.children.forEach((child) => cards(child, found));
    return found;
};
const out = [];
for (const payload of payloads) {
    globalThis.payload = payload;
    const root = new Node("div");
    await refreshOnce(root, { pending: new Node("div"), active: new Node("div") });
    out.push(cards(root));
}
process.stdout.write(JSON.stringify(out));
"""

# ComfyUI's own modules, and the brand mark, which needs image elements this
# small DOM does not model. Every memory and presentation module is the real one.
PANEL_STUBS = {
    "web/js/dgx_monarch_brand.js": ("export const updateBrand = () => {};\n"
                                    "export const brandMark = () => document.createElement('span');\n"
                                    "export const brandHeader = () => document.createElement('div');\n"),
    "scripts/app.js": "export const app = { registerExtension() {}, extensionManager: { registerSidebarTab() {} } };\n",
    "scripts/api.js": ("export const api = { fetchApi: async () => ({ ok: true, json: async () => globalThis.payload }),"
                       " addEventListener() {} };\n"),
}


def _panel_cards(tmp_path, payloads):
    """Run the shipped panel's real refresh against each telemetry payload."""
    node = _node()
    ext = tmp_path / "ext"
    (ext / "web/js").mkdir(parents=True)
    (ext / "scripts").mkdir()
    (ext / "package.json").write_text('{"type":"module"}')
    for path in WEB.glob("*.js"):
        source = path.read_text()
        if path.name == "dgx_monarch_panel.js":
            source += "\nexport { refreshOnce };\n"
        (ext / "web/js" / path.name).write_text(source)
    for name, text in PANEL_STUBS.items():
        (ext / name).write_text(text)
    (ext / "driver.js").write_text(PANEL_DRIVER)
    result = subprocess.run([node, "driver.js", json.dumps(payloads)], cwd=ext, capture_output=True,
                            text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout)


def test_the_panel_draws_the_bar_per_case_through_its_real_refresh(tmp_path):
    payloads = [_telemetry([worker]) for _, worker, _ in CASES]
    for (case, worker, drawn), cards in zip(CASES, _panel_cards(tmp_path, payloads), strict=True):
        by_name = {card["name"]: card for card in cards}
        # The driver's own box has no Worker report, so it always draws.
        assert by_name["node-driver (driver)"]["bar"], case
        if worker is None:
            assert set(by_name) == {"node-driver (driver)"}, case
            continue
        card = by_name["node-0001"]
        assert card["bar"] is drawn, case
        assert ("memory breakdown unknown" in card["text"]) is not drawn, case
        if drawn:
            assert re.search(r"pool 4\.0 GiB", card["text"]), case


def test_the_panel_keeps_a_memory_error_unknown_even_beside_a_memory_report(tmp_path):
    worker = {"memory": {**RESIDENT, "weight_residency": "cudaMalloc"}, "memory_error": "RuntimeError: partial"}
    healthy = {"memory": {**RESIDENT, "weight_residency": "cudaMalloc"}}
    payload = _telemetry([worker])
    payload["workers"].append({"host": {**STATS, "host": "node-0001"}, **healthy})
    cards = _panel_cards(tmp_path, [payload])[0]
    ranks = [card for card in cards if card["name"] == "node-0001"]
    assert [card["bar"] for card in ranks] == [False, True]
