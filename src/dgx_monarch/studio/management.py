"""Dispatch typed Studio management adapters without browser-selected commands."""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from ..config import load_cluster_config
from ..tui.data import Poller, RingStore, _private_record_stream
from . import management_config, management_observe, management_reports, management_sol
from .comfy import ComfyClient
from .contracts import StudioError, canonical, identifier, timestamp
from .driver_service import DriverService
from .management_fields import parse
from .management_update import Updates, blocking
from .store import Store, private_directory

RECORD_BYTES = 4 * 1024 * 1024


class RecordedPoller(Poller):
    """Reuse TUI normalization with an already-authenticated bounded sample."""
    sample: dict[str, Any]

    def _driver_telemetry(self) -> dict[str, Any]:
        return self.sample


class Management:
    def __init__(self, store: Store, client: ComfyClient, config: Path | None, driver: DriverService | None):
        self.store, self.client, self.config = store, client, config
        self.updates = Updates(config, driver, client.origin, store.directory)

    def preview(self, action: str, values: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        parameters = parse(action, values)
        path: Path | None
        if action == "update":
            plan = self.updates.preview(parameters)
        elif action == "top.record":
            plan = {"samples": parameters["samples"], "interval_s": parameters["interval_s"],
                    "format": "dgxm-top-jsonl", "byte_limit": RECORD_BYTES}
            if parameters["path"]:
                path = management_config.absolute(parameters["path"])
                prior = recording_bytes(path) if path.exists() else b""
                plan.update(path=str(path), prior_sha256=hashlib.sha256(prior).hexdigest())
        elif action == "top.replay":
            if bool(parameters["path"]) == bool(parameters["recording_id"]):
                raise StudioError("recording", "Select exactly one saved recording or absolute replay path.")
            if parameters["path"]:
                path = management_config.absolute(parameters["path"])
                plan = {"path": str(path), "recording_id": None, "sha256": hashlib.sha256(recording_bytes(path)).hexdigest()}
            else:
                recording = self.store.get("recordings", parameters["recording_id"])
                plan = {"recording_id": recording["id"], "sha256": recording["sha256"], "samples": recording["samples"]}
                if recording.get("path"):
                    plan["path"] = recording["path"]
        elif action in {"gate.inspect", "gate.repair.preview", "gate.repair.apply"}:
            plan = management_reports.gate_plan(action, parameters)
        elif action == "diagnostics":
            plan = management_observe.diagnostics_plan(parameters)
        elif action == "diagnostics.capabilities":
            from ..diagnostic_session import capabilities
            plan = capabilities()
        else:
            path = self.selected_path(action, parameters)
            if path is None:
                raise StudioError("operation_config", "Select a configuration path before this operation.")
            if action in {"sol-install", "sol-smoke"}:
                plan = management_sol.preview(action, load_cluster_config(path), parameters)
            else:
                plan = management_config.preview(action, path, parameters)
        return parameters, plan

    def selected_path(self, action: str, parameters: dict[str, Any]) -> Path | None:
        value = parameters.get("output") if action == "setup" else parameters.get("cluster_config") if action == "launcher" else None
        if value:
            return management_config.absolute(value)
        if action in {"setup", "init"} and self.config is None:
            return Path.home() / ".config/dgx-monarch/cluster.toml"
        return self.config

    async def gate_preview(self, parameters: dict[str, Any]) -> dict[str, Any]:
        return await management_observe.gate_plan(self.client, parameters)

    async def execute(self, operation: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        action, parameters, plan = operation["action"], operation["parameters"], operation["adapter_plan"]
        if action == "update":
            def checkpoint(receipt: dict[str, Any]) -> None:
                self.store.update("operations", operation["id"], {"receipt": receipt})
            return await self.updates.execute(parameters, plan, checkpoint)
        if action == "top.record":
            return await self.record(parameters, operation["id"], plan)
        if action == "top.replay":
            return await blocking(self.replay, plan)
        if action in {"diagnostics", "gate.last"}:
            def checkpoint(receipt: dict[str, Any]) -> None:
                self.store.update("operations", operation["id"], {"receipt": receipt})
            if action == "diagnostics":
                return await management_observe.diagnostics(self.client, parameters, checkpoint)
            return await management_observe.gate(self.client, parameters, plan, checkpoint)
        if action in {"gate.inspect", "gate.repair.preview", "gate.repair.apply"}:
            return await blocking(management_reports.gate_execute, action, parameters, plan)
        if action == "diagnostics.capabilities":
            return "succeeded", {"source": "diagnostic_session", "report": plan}
        path = self.selected_path(action, parameters)
        if path is None:
            raise StudioError("operation_config", "The configuration selection is unavailable.")
        if action in {"sol-install", "sol-smoke"}:
            return await blocking(management_sol.execute, action, load_cluster_config(path), parameters, plan)
        return await blocking(management_config.execute, action, path, parameters, plan)

    async def record(self, parameters: dict[str, Any], operation_id: str, plan: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        directory = private_directory(self.store.directory / "recordings")
        recording_id = identifier()
        external = bool(parameters.get("path"))
        path = management_config.absolute(parameters["path"]) if external else directory / f"{recording_id}.jsonl"
        prior = recording_bytes(path) if external and path.exists() else b""
        if external and (not plan or hashlib.sha256(prior).hexdigest() != plan["prior_sha256"]):
            raise StudioError("recording_changed", "Append destination changed after review.", 409)
        poller = RecordedPoller(loops=[])
        retained, size = 0, len(prior)
        sha = hashlib.sha256(prior)
        receipt = {"source": "telemetry_recording", "recording_id": recording_id, "samples": 0, "complete": False}
        self.store.update("operations", operation_id, {"receipt": receipt})
        self.store.create("recordings", {"id": recording_id, "sha256": sha.hexdigest(), "samples": 0,
                                         "bytes": size, "created_at": timestamp(), "complete": False,
                                         **({"path": str(path)} if external else {})})
        try:
            with _private_record_stream(str(path), append=external) as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                if external and (os.fstat(stream.fileno()).st_size != size or recording_bytes(path) != prior):
                    raise StudioError("recording_changed", "Append destination changed while opening it.", 409)
                for index in range(parameters["samples"]):
                    try:
                        poller.sample = await self.client.request("GET", "/dgxm/telemetry")
                        if not isinstance(poller.sample, dict) or not poller.sample:
                            raise StudioError("telemetry_shape", "Telemetry is unavailable.")
                        tick = poller.tick()
                        tick["source_telemetry"] = poller.sample
                    except StudioError:
                        tick = poller.unknown_tick("driver_unavailable")
                    # A valid final JSON record may omit its newline. Append
                    # its separator together with the first new sample so the
                    # byte budget and retained digest cover the exact write.
                    separator = "\n" if index == 0 and prior and not prior.endswith(b"\n") else ""
                    line = (separator + canonical(tick) + "\n").encode()
                    if size + len(line) > RECORD_BYTES:
                        break
                    if os.fstat(stream.fileno()).st_size != size:
                        raise StudioError("recording_changed", "Another writer changed the recording.", 409)
                    stream.write(line.decode())
                    stream.flush()
                    os.fsync(stream.fileno())
                    sha.update(line)
                    size += len(line)
                    retained += 1
                    if index + 1 < parameters["samples"]:
                        await asyncio.sleep(parameters["interval_s"])
        finally:
            record = self.store.update("recordings", recording_id,
                                       {"sha256": sha.hexdigest(), "samples": retained + len(prior.splitlines()), "bytes": size,
                                        "complete": retained == parameters["samples"]})
            receipt = {"source": "telemetry_recording", "recording_id": recording_id, **record}
            self.store.update("operations", operation_id, {"receipt": receipt})
        return "succeeded" if record["complete"] else "unknown", receipt

    def replay(self, plan: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        recording_id = plan["recording_id"]
        if not plan.get("path") and (not isinstance(recording_id, str) or re.fullmatch(r"[a-f0-9-]{36}", recording_id) is None):
            raise StudioError("recording", "Invalid recording identity.")
        path = management_config.absolute(plan["path"]) if plan.get("path") else self.store.directory / "recordings" / f"{recording_id}.jsonl"
        raw = recording_bytes(path, validate=False)
        if hashlib.sha256(raw).hexdigest() != plan["sha256"]:
            raise StudioError("recording_changed", "Recording changed after review.", 409)
        ring = RingStore()
        for line in raw.splitlines():
            tick = json.loads(line)
            if not isinstance(tick, dict) or "t" not in tick:
                raise StudioError("recording", "Malformed telemetry recording.")
            ring.push(tick)
        return "succeeded", {"source": "telemetry_replay", "recording_id": recording_id,
                             "sha256": plan["sha256"], "samples": list(ring.ticks), "replay": True}


def recording_bytes(path: Path, *, validate: bool = True) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_size > RECORD_BYTES:
            raise StudioError("recording", "Recording is not a bounded owner-private regular file.")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(RECORD_BYTES + 1)
        if len(raw) > RECORD_BYTES:
            raise StudioError("recording", "Recording exceeds the byte budget.")
        for line in raw.splitlines() if validate else []:
            tick = json.loads(line)
            if not isinstance(tick, dict) or "t" not in tick:
                raise StudioError("recording", "Malformed telemetry recording.")
        return raw
    finally:
        os.close(descriptor)
