"""Exercise the real companion HTTP routes against mock_driver.py, never a GPU.

Usage: python http_smoke.py --companion http://127.0.0.1:18221 --token-file /tmp/token
The companion must target the isolated mock driver and have no management access.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

from mock_driver import WORKFLOW


def run(origin: str, token: str) -> dict:
    def call(path: str, data=None, raw=False):
        request = Request(origin + "/api/v1" + path,
                          data=None if data is None else json.dumps(data).encode(),
                          headers={"Authorization": "Bearer " + token,
                                   "X-DGXM-Schema": "1.0", "Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            payload = response.read()
            return payload if raw else json.loads(payload)

    capabilities = call("/capabilities")
    assert capabilities["schema_version"].split(".")[0] == "1"
    catalog = call("/catalog")
    assert "StudioCPUFixture" in catalog["object_info"], "Refusing a non-fixture driver"
    assert not any(operation["available"] and operation["id"] != "version"
                   for operation in capabilities["operations"]), "Refusing enabled management"
    inspection = call("/workflows/inspect", {"workflow": WORKFLOW})
    assert not inspection["errors"], inspection["errors"]
    values = {binding["id"]: binding["value"] for binding in inspection["bindings"] if binding["resolved"]}
    preview = call("/jobs/preview", {"workflow": WORKFLOW, "values": values,
                                    "operation": "image", "seed_policy": "fixed",
                                    "styles": [], "lora_stack": [], "parent_artifact_ids": []})
    key = str(uuid.uuid4())
    submission = {"preview_id": preview["id"], "idempotency_key": key}
    first = call("/jobs", submission)
    replay = call("/jobs", submission)
    assert first["id"] == replay["id"], "Idempotent replay duplicated the job"
    reconciled = call("/jobs/request/" + key)
    assert reconciled["id"] == first["id"]
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        snapshot = call("/snapshot")
        job = next(job for job in snapshot["jobs"] if job["id"] == first["id"])
        if job["state"] == "succeeded":
            break
        assert job["state"] not in {"failed", "cancelled"}, job
        time.sleep(.25)
    else:
        raise AssertionError("Fixture job did not reach succeeded within 15 seconds")
    artifact = next(artifact for artifact in snapshot["artifacts"] if artifact["job_id"] == job["id"])
    image = call("/artifacts/" + artifact["id"] + "/file", raw=True)
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert job["submission"]["prompt"]["1"]["inputs"]["seed"] == 7
    assert job["submission"]["workflow"] == WORKFLOW
    return {"evidence": "synthetic CPU HTTP fixture; no inference or hardware acceptance",
            "job_state": job["state"], "duplicate_submit_same_id": True,
            "request_reconciled": True, "immutable_workflow_preserved": True,
            "artifact_kind": artifact["kind"], "artifact_bytes": len(image)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companion", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    if not args.companion.startswith("http://127.0.0.1:"):
        parser.error("This fixture is restricted to local loopback companions")
    print(json.dumps(run(args.companion.rstrip("/"), args.token_file.read_text().strip()), indent=2))


if __name__ == "__main__":
    main()
