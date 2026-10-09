"""tools/campaign_stats.py: the campaign exports and the report built from them.

The fixture is three synthetic records in the runner's shape: a one-GPU
reference, a pair cell that names it, and an earlier attempt of the pair cell
that refused class C. Machine names, the address and the private LoRA are
invented here and built at run time, so this file carries none of them as a
literal the leak checker would have to allow.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"dgxm_tools_{name}", REPO / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


campaign_stats = _load("campaign_stats")
leak_check = _load("leak_check")

HEAD, SIBLING = "node" + "alpha", "node" + "beta"
ADDRESS = ".".join(["192", "168", "40", "2"])
PRIVATE_LORA = "client" + "_style_v3.safetensors"
PUBLIC_LORA = "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors"
PAGE = "docs/campaign-results/fixture-2026-09.md"
COUNTERS = ("MemTotal", "MemFree", "MemAvailable", "Cached", "Buffers", "Shmem", "AnonPages", "Dirty",
            "Writeback", "Active(anon)", "Inactive(anon)", "Active(file)", "Inactive(file)", "Unevictable",
            "Mlocked", "SReclaimable", "SwapTotal", "SwapFree")


def _memory(available: float, anon: float) -> dict:
    values = dict.fromkeys(COUNTERS, 1.0)
    values.update(MemTotal=121.6, MemAvailable=available, AnonPages=anon, SwapTotal=0.0, SwapFree=0.0)
    return {c: {"min": v - 0.5 if c == "MemAvailable" else v, "peak": v, "last": v} for c, v in values.items()}


def _leg(wall: float, outcome: str = "render", message: str = "", hosts=("head", SIBLING)) -> dict:
    return {"prefix": "sweep_leg", "cached_seeded": [], "exception_type": "" if outcome == "render" else
            "dgx_monarch.capacity_agreement.FleetCapacityError", "wall_s": wall, "outcome": outcome,
            "message": message, "frames": 1,
            "memory": {h: _memory(60.0 + i, 20.0 - i) for i, h in enumerate(hosts)}}


def _record(cell_id: str, session: str, *, reference: str = "", legs: dict, loras: list, journal: dict,
            verdict: str = "PASS", observed: str = "render", started: str = "2026-09-20 10:00:00") -> dict:
    return {
        "cell": {"id": cell_id, "family": "chroma", "template": "dgx-monarch-test-chroma-lora", "klass": "image",
                 "label": "render", "quant": "fp8", "artifact_quant": "fp8_scaled", "launch_quant": "fp8",
                 "attention": "TORCH_FLASH", "resolved_attention": "TORCH_FLASH",
                 "preset": "single" if session == "local" else "auto", "mode": session, "session": session,
                 "levers": {}, "loras": loras, "text_tokens": {"positive": 44, "negative": 44}, "audio": False,
                 "reference": {"kind": "cell", "cell": reference, "bar": "step-nrms", "probe_steps": 1}
                 if reference else {"kind": "none", "bar": "reference"},
                 "unets": {"3": "Chroma1-HD-fp8_scaled.safetensors"}, "waiver": False, "waiver_guards": []},
        "pin": {"comfy_commit": "3216c62e" * 5, "monarch_commit": "a6df3d4d" * 5, "source_digest": "191c6831f086ad2c",
                "torch": "2.12.0+cu132", "nccl": "2.29.7", "gate_protocol_version": "13", "world": 2},
        "started": started, "legs": legs, "graph_digest": "db9ce96eb94841bc",
        "reset": {"fleet_kept": True, "available_gib": 68.0, "need_gib": 60.0}, "floor_reset": {},
        "findings": [], "notes": [], "cleared_stale_files": 0,
        "memory_floor": {"readings": {"head": 68.0, SIBLING: 77.0}, "errors": {}, "floor_gib": 60.0, "cleared": True},
        "ledger_rows": [{"verdict": "RETESTING", "blocked_contexts": [json.dumps({
            "resolved_topology": {"cfg": 1, "dp": 1, "fsdp": False, "ring": 1, "ulysses": 2},
            "config_source": "/home/operator/.config/dgx-monarch/cluster.toml"})]},
            {"verdict": "PASS", "max_abs_latent_diff": 0.0}] if session == "cluster" else [],
        "journal": journal, "observed": observed, "verdict": verdict,
        "fidelity": {"frames": 1, "nrms": 0.0153, "max_abs": 25, "identical": False, "bar": "step-nrms"}
        if reference else {},
    }


def _journal(host_name: str, *lines: str) -> list[str]:
    return [f"[dgx-monarch {host_name}] INFO [actor=<root>.<w{{'hosts': 0/2}}>] {line}" for line in lines]


def _write_cells(cells: Path) -> None:
    cells.mkdir(parents=True)
    lora = [{"name": PRIVATE_LORA, "node": "12", "strength": 0.8}, {"name": PUBLIC_LORA, "node": "13", "strength": 1.0}]
    records = {
        "aaaaaaaaaaa1.json": _record(
            "aaaaaaaaaaa1", "local", legs={"cold": _leg(70.0), "warm": _leg(46.0), "probe": _leg(4.0)}, loras=lora,
            journal={"head": _journal(HEAD, f"model store [cond]: load -- {PRIVATE_LORA} in 3.0s")}),
        "bbbbbbbbbbb2.json": _record(
            "bbbbbbbbbbb2", "cluster", reference="aaaaaaaaaaa1", loras=lora,
            legs={"cold": _leg(50.0), "warm": _leg(23.0), "probe": _leg(4.0)},
            journal={"head": _journal(HEAD, "USP attention kernel: TORCH_FLASH (sync_ulysses=True)",
                                      "slab residency: 13.80 GiB slab, reabsorbed 174 strays (0.6 GiB)"),
                     SIBLING: _journal(SIBLING, f"failed to connect, remote=[{ADDRESS}]:15752")}),
        "bbbbbbbbbbb2.attempt1.json": _record(
            "bbbbbbbbbbb2", "cluster", reference="aaaaaaaaaaa1", loras=lora, verdict="PASS-capacity",
            observed="refuse:C", started="2026-09-19 09:00:00",
            legs={"cold": _leg(2.0, "refuse:C", f"[dgxm:C guard=cross_rank_capacity waivable=0] rank 0 (host {HEAD})"
                                                 f' File "/home/operator/ComfyUI/comfy/x.py" {PRIVATE_LORA}')},
            journal={"head": [], SIBLING: []}),
    }
    for name, record in records.items():
        (cells / name).write_text(json.dumps(record), encoding="utf-8")


def _run(tmp_path: Path, root_name: str, check: bool = False) -> tuple[Path, list[str]]:
    cells = tmp_path / "cells"
    if not cells.exists():
        _write_cells(cells)
    root = tmp_path / root_name
    drift = campaign_stats.run({"fixture_2026-09": cells}, root, check, None)
    return root, drift


def _export(root: Path) -> Path:
    return root / "benchmark" / "reports" / "fixture_2026-09_cells.csv.gz"


def test_a_rerun_on_the_same_records_is_byte_identical(tmp_path):
    first, _ = _run(tmp_path, "one")
    second, _ = _run(tmp_path, "two")
    for relative in ("benchmark/reports/fixture_2026-09_cells.csv.gz", "docs/CAMPAIGN_RESULTS.md",
                     "docs/CAMPAIGN_COLUMNS.md", PAGE):
        assert (first / relative).read_bytes() == (second / relative).read_bytes(), relative
    data = _export(first).read_bytes()
    assert data[4:8] == b"\0\0\0\0", "gzip header must carry mtime 0"


def test_one_row_per_record_and_one_sorted_column_per_field(tmp_path):
    root, _ = _run(tmp_path, "out")
    rows = campaign_stats.read_export(_export(root))
    assert [(r["cell_id"], r["attempt"]) for r in rows] == [("aaaaaaaaaaa1", "0"), ("bbbbbbbbbbb2", "0"),
                                                             ("bbbbbbbbbbb2", "1")]
    header = list(rows[0])
    assert header[:4] == list(campaign_stats.IDENTITY)
    assert header[4:] == sorted(header[4:])
    pair = rows[1]
    for leg in ("cold", "warm", "probe"):
        for host in ("head", "sibling"):
            for stat in ("min", "peak", "last"):
                for counter in COUNTERS:
                    assert f"legs.{leg}.memory.{host}.{counter}.{stat}" in header
    assert pair["legs.warm.wall_s"] == "23.0" and pair["fidelity.nrms"] == "0.0153"
    assert pair["derived.topology"] == "auto:uly2" and pair["derived.journal_kernels"] == "TORCH_FLASH"
    assert pair["derived.slab_gib"] == "13.8" and pair["ledger.count"] == "2"
    assert json.loads(pair["ledger.verdict"]) == ["RETESTING", "PASS"]
    assert rows[2]["derived.cold.refusal_class"] == "C"
    assert rows[2]["derived.cold.refusal_guard"] == "cross_rank_capacity"


def test_private_names_are_replaced_everywhere_and_public_ones_kept(tmp_path):
    root, _ = _run(tmp_path, "out")
    outputs = [gzip.decompress(_export(root).read_bytes())]
    outputs += [(root / name).read_bytes() for name in ("docs/CAMPAIGN_RESULTS.md", "docs/CAMPAIGN_COLUMNS.md", PAGE)]
    for data in outputs:
        for secret in (HEAD, SIBLING, ADDRESS, PRIVATE_LORA, PRIVATE_LORA.removesuffix(".safetensors"),
                       "/home/", "operator"):
            assert secret.encode() not in data, secret
        assert not leak_check.scan_bytes("output", data)
    rows = campaign_stats.read_export(_export(root))
    assert json.loads(rows[0]["cell.loras.name"]) == ["lora_1.safetensors", PUBLIC_LORA]
    assert "lora_1.safetensors" in rows[0]["journal.head.text"]
    assert "<ip>" in rows[1]["journal.sibling.text"]
    assert "rank 0 (host head)" in rows[2]["legs.cold.message"]
    assert "~/ComfyUI" in rows[2]["legs.cold.message"]
    assert "~/.config" in rows[1]["ledger.blocked_contexts"]


def test_the_report_pairs_a_cell_with_its_one_gpu_reference(tmp_path):
    root, _ = _run(tmp_path, "out")
    text = (root / PAGE).read_text()
    assert '<a id="chroma-fixture_2026-09"></a>chroma' in text
    assert "(campaign-results/fixture-2026-09.md)" in (root / "docs" / "CAMPAIGN_RESULTS.md").read_text()
    # 46.0 s on one GPU against 23.0 s on the pair.
    index = _load("campaign_stats_index")
    summary = json.loads(index.summary_json({"fixture_2026-09": campaign_stats.read_export(_export(root))}))
    pair = next(g for g in summary["fixture_2026-09"]["groups"] if g["topology"] == "auto:uly2")
    assert pair["ref_warm_s"]["median"] == 46.0
    assert pair["walls_s"]["warm"]["median"] == 23.0
    assert pair["speedup"]["median"] == 2.0
    assert "| <a id=\"chroma-fixture_2026-09\"></a>chroma | 2 | 2 | 0 | 0 | 0 | 0 | 1 | 0 |" in text
    assert "2100 MHz" in text


def test_check_passes_on_fresh_output_and_fails_on_drift(tmp_path):
    root, _ = _run(tmp_path, "out")
    assert campaign_stats.run({"fixture_2026-09": tmp_path / "cells"}, root, True, None) == []
    assert campaign_stats.run({}, root, True, None) == []
    doc = root / PAGE
    doc.write_text(doc.read_text().replace("2100 MHz", "2500 MHz"))
    assert campaign_stats.run({}, root, True, None) == [PAGE]
    assert campaign_stats.main(["--check", "--root", str(root)]) == 1


def test_the_nccl_stamp_note_shows_only_for_build_value_records_before_the_change():
    report = _load("campaign_stats_report")

    def header(*rows: dict) -> str:
        records = [{"attempt": "0", **row} for row in rows]
        return "\n".join(report.campaign_header("fixture_2026-09", records))

    old = {"pin.nccl_source": "torch-runtime", "started": "2026-09-20 10:00:00"}
    loaded = {"pin.nccl_source": "loaded-library", "started": "2026-10-07 10:00:00"}
    later_fallback = {"pin.nccl_source": "torch-runtime", "started": "2026-10-07 10:00:00"}
    assert report.NCCL_STAMP_NOTE in header(old)
    assert report.NCCL_STAMP_NOTE in header(old, loaded)
    assert report.NCCL_STAMP_NOTE not in header(loaded)
    assert report.NCCL_STAMP_NOTE not in header(later_fallback)
    assert report.NCCL_STAMP_NOTE not in header({"pin.nccl_source": "torch-runtime"})


def test_the_committed_report_matches_the_committed_exports():
    """CI holds the published pages to the exports without the private records."""
    assert campaign_stats.run({}, REPO, True, None) == []


def test_family_overview_keeps_failed_and_unscored_records():
    report = _load("campaign_stats_report")
    rows = [
        {"cell.family": "example", "verdict": "PASS", "observed": "render", "fidelity.nrms": "0.0"},
        {"cell.family": "example", "verdict": "CHECK", "observed": "render",
         "fidelity.bar": "step-nrms", "fidelity.nrms": "0.25"},
        {"cell.family": "example", "verdict": "CHECK", "observed": "render", "fidelity.bar": "step-nrms"},
        {"cell.family": "example", "verdict": "FINDING", "observed": "crash"},
        {"cell.family": "example", "verdict": "PASS-capacity", "observed": "refuse:C"},
    ]
    text = "\n".join(report.family_overview(rows, "Fixture"))
    assert '| <a id="example-fixture"></a>example | 5 | 1 | 1 | 2 | 1 | 1 | 2 | 1 |' in text
