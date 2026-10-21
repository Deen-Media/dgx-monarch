"""Exercise the real companion HTTP routes against mock_driver.py, never a GPU.

Usage: python http_smoke.py --companion http://127.0.0.1:18221 --token-file TOKEN_FILE
The companion must target the isolated mock driver and have no management access.
The script prints a JSON report whose check flags come from the companion's own
responses, and exits 1 when any check fails.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from mock_driver import WORKFLOW

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def require(condition: bool, message: str) -> None:
    """Refuse explicitly; unlike assert, this still runs under python -O."""
    if not condition:
        raise RuntimeError(message)


def field(value: Any, *path: str) -> Any:
    """The nested value at path, or None when any level is missing."""
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def checks(*, first: dict, replay: dict, reconciled: dict, summary: dict, detail: dict,
           image: bytes) -> dict[str, bool]:
    """Each evidence flag, computed from what the companion returned."""
    submission = field(detail, "submission") or {}
    job_id = first.get("id")
    return {
        "duplicate_submit_same_id": job_id is not None and replay.get("id") == job_id,
        "request_reconciled": job_id is not None and reconciled.get("id") == job_id,
        "summary_omits_workflow": summary.get("summary") is True and "workflow" not in (summary.get("submission") or {}),
        "immutable_workflow_preserved": submission.get("workflow") == WORKFLOW,
        "submitted_seed_preserved": field(submission, "prompt", "1", "inputs", "seed") == WORKFLOW["1"]["inputs"]["seed"],
        "artifact_is_png": image.startswith(PNG_SIGNATURE),
    }


def run(origin: str, token: str, *, timeout: float = 15.0) -> dict:
    def call(path: str, data=None, raw=False):
        request = Request(origin + "/api/v1" + path,
                          data=None if data is None else json.dumps(data).encode(),
                          headers={"Authorization": "Bearer " + token,
                                   "X-DGXM-Schema": "1.0", "Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            payload = response.read()
            return payload if raw else json.loads(payload)

    capabilities = call("/capabilities")
    require(str(capabilities.get("schema_version", "")).split(".")[0] == "1", "Refusing an unsupported schema major")
    catalog = call("/catalog")
    require("StudioCPUFixture" in (catalog.get("object_info") or {}), "Refusing a non-fixture driver")
    require(not any(operation.get("available") and operation.get("requires_confirmation", True)
                    for operation in capabilities.get("operations", [])), "Refusing enabled management")
    inspection = call("/workflows/inspect", {"workflow": WORKFLOW})
    require(not inspection.get("errors"), f"Fixture workflow inspection failed: {inspection.get('errors')}")
    values = {binding["id"]: binding["value"] for binding in inspection["bindings"] if binding["resolved"]}
    preview = call("/jobs/preview", {"workflow": WORKFLOW, "values": values,
                                    "operation": "image", "seed_policy": "fixed",
                                    "styles": [], "lora_stack": [], "parent_artifact_ids": []})
    key = str(uuid.uuid4())
    submission = {"preview_id": preview["id"], "idempotency_key": key}
    first = call("/jobs", submission)
    replay = call("/jobs", submission)
    reconciled = call("/jobs/request/" + key)
    deadline = time.monotonic() + timeout
    while True:
        snapshot = call("/snapshot")
        job = next((row for row in snapshot.get("jobs", []) if row.get("id") == first.get("id")), None)
        require(job is not None, "The submitted fixture job is missing from the snapshot")
        if job["state"] == "succeeded":
            break
        require(job["state"] not in {"failed", "cancelled"}, f"Fixture job ended as {job['state']}")
        require(time.monotonic() < deadline, f"Fixture job did not reach succeeded within {timeout:g} seconds")
        time.sleep(.25)
    artifact = next((row for row in snapshot.get("artifacts", []) if row.get("job_id") == job["id"]), None)
    require(artifact is not None, "The succeeded fixture job has no artifact")
    image = call("/artifacts/" + artifact["id"] + "/file", raw=True)
    detail = call("/jobs/" + job["id"])
    results = checks(first=first, replay=replay, reconciled=reconciled, summary=job, detail=detail, image=image)
    return {"evidence": "synthetic CPU HTTP fixture; no inference or hardware acceptance",
            "job_state": detail.get("state"), **results, "passed": all(results.values()),
            "artifact_kind": artifact.get("kind"), "artifact_bytes": len(image)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companion", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    if not args.companion.startswith("http://127.0.0.1:"):
        parser.error("This fixture is restricted to local loopback companions")
    report = run(args.companion.rstrip("/"), args.token_file.read_text().strip())
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
