"""`dgxm gate --last` and Studio's gate.last gate the same prompt with one graph.

The CLI asks ComfyUI for `history?max_items=1`; Studio reads the bounded
/history window. ComfyUI keeps history in completion order, so both must take
the newest entry, whatever queue number ComfyUI stored with it.
"""
from __future__ import annotations

import ast
import asyncio
import io
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import gate_last
from dgx_monarch.cli import main as cli
from dgx_monarch.studio import management_observe
from dgx_monarch.studio.contracts import StudioError

RUN = uuid.UUID(int=7)

SCHEMA = {
    "Inputs": {"input": {"required": {}}, "output": ["MODEL", "CONDITIONING", "LATENT"]},
    "DGXMonarchIdentityGate": {"input": {"required": {
        "model": ["MODEL", {}], "positive": ["CONDITIONING", {}], "negative": ["CONDITIONING", {}],
        "latent_image": ["LATENT", {}], "noise_seed": ["INT", {}], "steps": ["INT", {}],
        "cfg": ["FLOAT", {}], "sampler_name": [["euler", "heun"], {}], "scheduler": [["simple", "karras"], {}],
        "run_id": ["STRING", {}]}}, "output": ["LATENT"], "output_node": True},
}


def graph(seed: int, sampler: str = "euler") -> dict:
    return {"1": {"class_type": "Inputs", "inputs": {}}, "2": {"class_type": "DGXMonarchKSampler", "inputs": {
        "model": ["1", 0], "positive": ["1", 1], "negative": ["1", 1], "latent_image": ["1", 2],
        "noise_seed": seed, "cfg": 1.5, "sampler_name": sampler, "scheduler": "simple"}}}


def entry(number, identity: str, seed: int, sampler: str = "euler") -> dict:
    # ComfyUI stores [number, prompt_id, graph, extra_data, outputs_to_execute].
    return {"prompt": [number, identity, graph(seed, sampler), {}, ["2"]],
            "status": {"completed": True, "status_str": "success"}, "outputs": {}}


# Normal order, a Queue Front prompt (number negated) and an explicit number
# (cast to float by ComfyUI's /prompt route), each placed last in turn.
ORDERS = {
    "ordered_int": [entry(1, "first", 101), entry(-5, "front", 202, "heun"), entry(2, "newest", 303)],
    "queue_front_last": [entry(1, "first", 101), entry(2, "second", 303), entry(-5, "newest", 202, "heun")],
    "float_number_last": [entry(1, "first", 101), entry(-5, "front", 202, "heun"), entry(2.0, "newest", 404)],
}


def comfy_history(history: dict, max_items: int | None) -> dict:
    """ComfyUI's `PromptQueue.get_history` window: the last entries in completion order."""
    offset = len(history) - max_items if max_items is not None else 0
    return dict(list(history.items())[offset:])


def run_cli(monkeypatch, tmp_path: Path, history: dict) -> tuple[int, dict]:
    submitted = []

    def urlopen(target, timeout=None):
        url = getattr(target, "full_url", target)
        if url == "http://127.0.0.1:8195/history?max_items=1":
            return io.BytesIO(json.dumps(comfy_history(history, 1)).encode())
        if url == "http://127.0.0.1:8195/prompt":
            submitted.append(json.loads(target.data)["prompt"])
            return io.BytesIO(b'{"prompt_id": "gate-run"}')
        if url == "http://127.0.0.1:8195/history/gate-run":
            return io.BytesIO(b'{"gate-run": {"status": {"status_str": "success"}}}')
        raise OSError(f"unexpected request {url}")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    args = SimpleNamespace(gate_action="last", host="127.0.0.1:8195", report_dir=str(tmp_path), timeout=10, apply=False)
    code = cli._cmd_gate_impl(args)
    assert len(submitted) == 1
    return code, submitted[0]


def run_studio(tmp_path: Path, history: object) -> dict:
    class Client:
        async def request(self, method, path, data=None):
            assert method == "GET"
            if path == "/history":
                # A driver answer that is not a mapping reaches gate_plan unchanged.
                return comfy_history(history, 200) if isinstance(history, dict) else history
            assert path == "/object_info"
            return SCHEMA

    return asyncio.run(management_observe.gate_plan(Client(), {"report_dir": str(tmp_path), "timeout_s": 10}))


@pytest.mark.parametrize("order", sorted(ORDERS))
def test_cli_and_studio_gate_the_same_entry_with_the_same_graph(order, monkeypatch, tmp_path):
    history = {row["prompt"][1]: row for row in ORDERS[order]}
    monkeypatch.setattr(uuid, "uuid4", lambda: RUN)
    (tmp_path / "dgxm_gate_reports.jsonl").write_text(json.dumps({"run_id": RUN.hex, "verdict": "PASS"}) + "\n")

    code, cli_prompt = run_cli(monkeypatch, tmp_path, history)
    plan = run_studio(tmp_path, history)

    assert code == 0
    assert plan["source_prompt_id"] == "newest"
    newest = history["newest"]["prompt"][2]["2"]["inputs"]
    assert cli_prompt["2"]["inputs"]["noise_seed"] == newest["noise_seed"]
    assert plan["prompt"] == cli_prompt
    assert plan["prompt"]["2"]["class_type"] == "DGXMonarchIdentityGate"
    assert plan["prompt"]["2"]["inputs"]["run_id"] == RUN.hex
    assert plan["sampler_node"] == "2"


def test_studio_refuses_a_newest_prompt_without_a_sampler_instead_of_reaching_back(tmp_path):
    newest = entry(3, "newest", 1)
    newest["prompt"][2]["2"]["class_type"] = "KSampler"
    history = {"older": entry(1, "older", 101), "newest": newest}
    with pytest.raises(StudioError) as caught:
        run_studio(tmp_path, history)
    assert (caught.value.code, caught.value.status) == ("gate_sampler", 409)


@pytest.mark.parametrize("seed", ["not a number", {"nested": True}])
def test_studio_refuses_an_unusable_seed_with_a_typed_refusal(seed, tmp_path):
    newest = entry(1, "newest", 1)
    newest["prompt"][2]["2"]["inputs"]["noise_seed"] = seed
    with pytest.raises(StudioError) as caught:
        run_studio(tmp_path, {"newest": newest})
    assert (caught.value.code, caught.value.status) == ("gate_sampler", 409)


@pytest.mark.parametrize("history", [{}, [], None, {"newest": {"prompt": [1, "newest"]}}])
def test_studio_refuses_empty_or_graphless_history(history, tmp_path):
    with pytest.raises(StudioError) as caught:
        run_studio(tmp_path, history)
    assert caught.value.code == "gate_history"


UPLOAD = "studio/0123abcd.png"
UPLOAD_SCHEMA = {**SCHEMA, "LoadImage": {"input": {"required": {"image": [["example.png"], {"image_upload": True}]}},
                                         "output": ["IMAGE"]}}


def test_studio_gates_a_last_prompt_that_loaded_a_stored_upload(tmp_path):
    """A render whose LoadImage read a Studio upload gates as `dgxm gate --last` gates it."""
    from contextlib import closing

    from dgx_monarch.studio.contracts import digest
    from dgx_monarch.studio.management import Management
    from dgx_monarch.studio.store import Store
    from dgx_monarch.studio.workflows import WorkflowError

    source = graph(5)
    source["3"] = {"class_type": "LoadImage", "inputs": {"image": UPLOAD}}
    history = {"newest": {"prompt": [1, "newest", source, {}, ["2"]],
                          "status": {"completed": True, "status_str": "success"}, "outputs": {}}}

    class Client:
        origin = "http://127.0.0.1:8188"

        async def request(self, method, path, data=None):
            return history if path == "/history" else UPLOAD_SCHEMA

    parameters = {"report_dir": str(tmp_path), "timeout_s": 10}
    with closing(Store(tmp_path / "state")) as store:
        store.create("artifacts", {"id": "upload", "kind": "input", "job_id": None, "binding_value": UPLOAD})
        plan = asyncio.run(Management(store, Client(), None, None).gate_preview(parameters))
    assert plan["prompt"]["3"] == {"class_type": "LoadImage", "inputs": {"image": UPLOAD}}
    # The review names the driver's own schema, and a file this companion never stored still refuses.
    assert plan["schema_digest"] == digest(UPLOAD_SCHEMA)
    with pytest.raises(WorkflowError, match="not an available choice"):
        asyncio.run(management_observe.gate_plan(Client(), parameters))


def test_latest_entry_follows_insertion_order_not_queue_number():
    history = {row["prompt"][1]: row for row in ORDERS["queue_front_last"]}
    assert gate_last.latest_entry(history)[0] == "newest"
    assert gate_last.latest_entry({}) is None
    assert gate_last.latest_entry([["not", "a", "mapping"]]) is None


def test_identity_gate_leaves_the_source_graph_unchanged():
    source = graph(9)
    before = json.loads(json.dumps(source))
    target, prompt = gate_last.identity_gate(source, "run")
    assert source == before
    assert target == "2" and prompt["2"]["inputs"]["steps"] == 2
    prompt["2"]["inputs"]["model"].append("mutated")
    assert source["2"]["inputs"]["model"] == ["1", 0]


def test_identity_gate_refuses_a_sampler_without_gate_inputs():
    source = graph(9)
    del source["2"]["inputs"]["latent_image"]
    with pytest.raises(ValueError, match="latent_image"):
        gate_last.identity_gate(source, "run")
    assert gate_last.identity_gate({"1": {"class_type": "KSampler", "inputs": {}}}, "run") is None


def test_gate_last_imports_no_network_comfy_or_torch():
    tree = ast.parse(Path(gate_last.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"__future__", "json", "collections", "copy", "pathlib", "typing"}


def test_cli_keeps_its_gate_helper_names():
    assert cli._find_gate_report is gate_last.find_gate_report
    assert cli._prompt_input is gate_last.prompt_input
