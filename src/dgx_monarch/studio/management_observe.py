"""Bounded diagnostics and reviewed identity-gate execution using fixed APIs."""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from typing import Any

from .. import diagnostic_session
from ..cli.gate_last import find_gate_report as _find_gate_report
from ..cli.gate_last import identity_gate, latest_entry, source_graph
from .contracts import StudioError, digest
from .management_config import absolute
from .submission_guard import referenced_uploads, upload_view
from .workflows import compile_workflow


def diagnostics_plan(parameters: dict[str, Any]) -> dict[str, Any]:
    diagnostic_session.Capture(**{key: parameters[key] for key in ("duration_s", "max_events", "max_bytes")})
    if parameters["output"]:
        diagnostic_session.validate_destination(absolute(parameters["output"]))
    return {"limits": {key: parameters[key] for key in ("duration_s", "max_events", "max_bytes")},
            "scope": "Existing telemetry only; no actor attach or ownership inference."}


async def diagnostics(client: Any, parameters: dict[str, Any], checkpoint: Callable[[dict], None]) -> tuple[str, dict]:
    capture = diagnostic_session.Capture(**{key: parameters[key] for key in ("duration_s", "max_events", "max_bytes")})
    try:
        while not capture.done:
            try:
                async with asyncio.timeout(min(2.0, capture.remaining_s)):
                    capture.ingest(await client.request("GET", "/dgxm/telemetry"))
            except (StudioError, TimeoutError):
                capture.gap("fetch_failed")
            if not capture.done:
                await asyncio.sleep(min(0.5, capture.remaining_s))
    except asyncio.CancelledError:
        capture.gap("interrupted")
        raise
    finally:
        checkpoint({"source": "diagnostic_session", "report": capture.report()})
    output = capture.export(absolute(parameters["output"])) if parameters["output"] else None
    report = capture.report()
    return ("unknown" if report["capture"]["partial"] else "succeeded",
            {"source": "diagnostic_session", "report": report, "output": str(output) if output else None})


async def gate_plan(client: Any, parameters: dict[str, Any], uploads: frozenset[str] = frozenset()) -> dict[str, Any]:
    """Gate the prompt `dgxm gate --last` would gate: the last /history entry.

    ``uploads`` are the input files this companion stored; a last prompt that
    loaded one validates against them, as the job that rendered it did.
    """
    absolute(parameters["report_dir"])
    latest = latest_entry(await client.request("GET", "/history"))
    if latest is None:
        raise StudioError("gate_history", "Render a supported Monarch workflow before gating it.", 409)
    identity, entry = latest
    run_id = uuid.uuid4().hex
    try:
        original = source_graph(entry)
    except ValueError as exc:
        raise StudioError("gate_history", "The last prompt in history has no usable API graph.", 409) from exc
    try:
        built = identity_gate(original, run_id)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StudioError("gate_sampler", "The latest sampler lacks gate inputs or has a seed or cfg the gate cannot use.", 409) from exc
    if built is None:
        raise StudioError("gate_sampler", "The last prompt has no supported Monarch sampler.", 409)
    target, prompt = built
    schema = await client.request("GET", "/object_info")
    prompt = compile_workflow(prompt, upload_view(schema, referenced_uploads(prompt, uploads)), {})
    return {"source_prompt_id": identity, "source_digest": digest(original), "prompt": prompt,
            "run_id": run_id, "schema_digest": digest(schema), "sampler_node": target,
            "scope": "Run the reviewed Identity Gate graph; retain its own prompt and report identities."}


async def gate(client: Any, parameters: dict[str, Any], plan: dict[str, Any],
               checkpoint: Callable[[dict], None]) -> tuple[str, dict]:
    if digest(await client.request("GET", "/object_info")) != plan["schema_digest"]:
        raise StudioError("gate_schema", "Driver schema changed after review.", 409)
    receipt: dict[str, Any] = {"source": "identity_gate", "run_id": plan["run_id"],
                               "source_prompt_id": plan["source_prompt_id"], "phase": "submission_requested"}
    checkpoint(receipt)
    accepted = await client.request("POST", "/prompt", {"prompt": plan["prompt"]})
    if not isinstance(accepted, dict) or not isinstance(accepted.get("prompt_id"), str):
        return "unknown", receipt
    prompt_id = accepted["prompt_id"]
    receipt.update(prompt_id=prompt_id, phase="accepted")
    checkpoint(receipt)
    deadline = time.monotonic() + parameters["timeout_s"]
    while time.monotonic() < deadline:
        history = await client.history_prompt(prompt_id)
        if isinstance(history, dict) and prompt_id in history:
            status = history[prompt_id].get("status", {})
            if status.get("completed") is True or status.get("status_str") in {"success", "error"}:
                report = await asyncio.to_thread(_find_gate_report, str(absolute(parameters["report_dir"])), plan["run_id"])
                receipt.update(phase="completed", backend_status=status, report=report)
                verdict = report.get("verdict") if isinstance(report, dict) else None
                return ("succeeded" if verdict == "PASS" and status.get("status_str") == "success" else
                        "failed" if verdict == "FAIL" else "unknown", receipt)
        await asyncio.sleep(min(3.0, max(0, deadline - time.monotonic())))
    return "unknown", {**receipt, "phase": "deadline", "recovery": "Inspect the retained prompt id before retrying."}
