"""A concurrent request cannot replace another preview's capability context."""
import asyncio
from copy import deepcopy

import pytest

from dgx_monarch.studio import jobs as jobs_module
from dgx_monarch.studio.contracts import StudioError, digest
from dgx_monarch.studio.experiments import GridExperiments
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.store import Store


def schema(version):
    return {"Render": {"input": {"required": {
        "text": ["STRING", {"default": version}],
        "seed": ["INT", {"default": 1, "min": 0, "max": 10}]}}, "output": [], "output_node": True}}


class OverlappingDriver:
    def __init__(self):
        self.catalogs = 0
        self.stats = 0
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()

    async def request(self, method, path):
        if path == "/object_info":
            self.catalogs += 1
            return schema("A" if self.catalogs == 1 else "B")
        assert path == "/system_stats"
        self.stats += 1
        version = "A" if self.stats == 1 else "B"
        if version == "A":
            self.first_started.set()
            await self.release_first.wait()
        return {"system": {"comfyui_version": version}}


@pytest.mark.parametrize("kind", ["job", "grid"])
def test_overlapping_previews_use_their_own_schema_and_runtime(tmp_path, kind):
    async def scenario():
        store = Store(tmp_path / "state")
        try:
            driver = OverlappingDriver()
            jobs = Jobs(store, driver)
            grids = GridExperiments(jobs)
            draft = {"workflow": {"nodes": [{"id": 1, "type": "Render", "widgets_values": []}]}}
            request = {"draft": draft, "axes": [{"name": "root:1:seed", "values": [2, 3]}]}
            first = asyncio.create_task(jobs.preview(draft) if kind == "job" else grids.preview(request))
            await driver.first_started.wait()
            second = await jobs.preview(draft)
            driver.release_first.set()
            first = await first
            previews = [first] if kind == "job" else [store.get("previews", row["preview_id"]) for row in first["cells"]]
            assert first["schema_digest"] == digest(schema("A"))
            for preview in previews:
                assert preview["prompt"]["1"]["inputs"]["text"] == "A"
                assert preview["schema_digest"] == digest(schema("A"))
                assert preview["versions"]["driver"] == "A"
            assert second["prompt"]["1"]["inputs"]["text"] == "B"
            assert second["versions"]["driver"] == "B"
            # The newer catalog still invalidates review of the older context.
            with pytest.raises(StudioError, match="capabilities changed"):
                if kind == "job":
                    jobs.reviewed_snapshot(first["id"])
                else:
                    grids.submit({"preview_id": first["id"], "idempotency_key": "old-grid"})
            assert not store.all("jobs")
        finally:
            store.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("low,high,current,expected", [
    (5, 7, 6, 7), (0, 2**64 - 1, 1, 2**53 - 1), (1.25, 3.75, 2, 3), (-3, -1, -2, -1),
])
def test_random_seeds_respect_declared_integer_bounds(tmp_path, monkeypatch, low, high, current, expected):
    store = Store(tmp_path / "state")
    try:
        jobs = Jobs(store, object())
        jobs.schema = {"Render": {"input": {"required": {
            "seed": ["INT", {"min": low, "max": high}], "noise_seed": ["BOOLEAN"]}},
            "output": [], "output_node": True},
            "Combo": {"input": {"required": {"seed": [[1, 2]]}}, "output": [], "output_node": True}}
        request = {"workflow": {
            "1": {"class_type": "Render", "inputs": {"seed": current, "noise_seed": True}},
            "2": {"class_type": "Combo", "inputs": {"seed": 1}}}, "seed_policy": "random"}
        original = deepcopy(request)
        monkeypatch.setattr(jobs_module.secrets, "randbelow", lambda count: count - 1)
        preview = jobs.build_preview(request)
        assert preview["prompt"]["1"]["inputs"]["seed"] == expected
        assert preview["prompt"]["1"]["inputs"]["noise_seed"] is True
        assert preview["prompt"]["2"]["inputs"]["seed"] == 1
        assert preview["effective"]["seeds"] == {"1.seed": expected}
        assert request == original
    finally:
        store.close()


@pytest.mark.parametrize("low,high", [(2**53, 2**64 - 1), (-2**64, -2**53)])
def test_random_seed_refuses_installed_range_outside_browser_precision(tmp_path, low, high):
    store = Store(tmp_path / "state")
    try:
        jobs = Jobs(store, object())
        jobs.schema = {"Render": {"input": {"required": {"seed": ["INT", {"min": low, "max": high}]}},
                                  "output": [], "output_node": True}}
        with pytest.raises(StudioError, match="browser JSON can represent exactly"):
            jobs.build_preview({"workflow": {"1": {"class_type": "Render", "inputs": {"seed": low}}},
                                "seed_policy": "random"})
        assert not store.all("previews")
    finally:
        store.close()


def test_random_seed_lower_bound_also_preserves_browser_precision(monkeypatch):
    monkeypatch.setattr(jobs_module.secrets, "randbelow", lambda count: 0)
    assert jobs_module._random_seed({"min": -2**64, "max": 0}) == -(2**53 - 1)


def test_random_seed_rejects_boolean_schema_bound(tmp_path):
    store = Store(tmp_path / "state")
    try:
        jobs = Jobs(store, object())
        jobs.schema = {"Render": {"input": {"required": {"seed": ["INT", {"min": True, "max": 7}]}},
                                  "output": [], "output_node": True}}
        with pytest.raises(StudioError, match="finite numeric bounds"):
            jobs.build_preview({"workflow": {"1": {"class_type": "Render", "inputs": {"seed": 5}}},
                                "seed_policy": "random"})
        assert not store.all("previews")
    finally:
        store.close()
