"""Dispatch typed Studio management adapters without browser-selected commands."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from ..config import load_cluster_config
from ..tui.data import Poller, RingStore, _private_record_stream
from . import management_config, management_sol
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
        if action == "update":
            plan = self.updates.preview(parameters)
        elif action == "top.record":
            plan = {"samples": parameters["samples"], "interval_s": parameters["interval_s"],
                    "format": "dgxm-top-jsonl", "byte_limit": RECORD_BYTES}
        elif action == "top.replay":
            recording = self.store.get("recordings", parameters["recording_id"])
            plan = {"recording_id": recording["id"], "sha256": recording["sha256"], "samples": recording["samples"]}
        else:
            if self.config is None:
                raise StudioError("operation_config", "Select a configuration path before this operation.")
            if action in {"sol-install", "sol-smoke"}:
                plan = management_sol.preview(action, load_cluster_config(self.config), parameters)
            else:
                plan = management_config.preview(action, self.config, parameters)
        return parameters, plan

    async def execute(self, operation: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        action, parameters, plan = operation["action"], operation["parameters"], operation["adapter_plan"]
        if action == "update":
            def checkpoint(receipt: dict[str, Any]) -> None:
                self.store.update("operations", operation["id"], {"receipt": receipt})
            return await self.updates.execute(parameters, plan, checkpoint)
        if action == "top.record":
            return await self.record(parameters, operation["id"])
        if action == "top.replay":
            return await blocking(self.replay, plan)
        if self.config is None:
            raise StudioError("operation_config", "The configuration selection is unavailable.")
        if action in {"sol-install", "sol-smoke"}:
            return await blocking(management_sol.execute, action, load_cluster_config(self.config), parameters, plan)
        return await blocking(management_config.execute, action, self.config, parameters, plan)

    async def record(self, parameters: dict[str, Any], operation_id: str) -> tuple[str, dict[str, Any]]:
        directory = private_directory(self.store.directory / "recordings")
        recording_id = identifier()
        path = directory / f"{recording_id}.jsonl"
        poller = RecordedPoller(loops=[])
        retained, size = 0, 0
        sha = hashlib.sha256()
        receipt = {"source": "telemetry_recording", "recording_id": recording_id, "samples": 0, "complete": False}
        self.store.update("operations", operation_id, {"receipt": receipt})
        self.store.create("recordings", {"id": recording_id, "sha256": sha.hexdigest(), "samples": 0,
                                         "bytes": 0, "created_at": timestamp(), "complete": False})
        try:
            with _private_record_stream(str(path), append=False) as stream:
                for index in range(parameters["samples"]):
                    try:
                        poller.sample = await self.client.request("GET", "/dgxm/telemetry")
                        if not isinstance(poller.sample, dict) or not poller.sample:
                            raise StudioError("telemetry_shape", "Telemetry is unavailable.")
                        tick = poller.tick()
                        tick["source_telemetry"] = poller.sample
                    except StudioError:
                        tick = poller.unknown_tick("driver_unavailable")
                    line = (canonical(tick) + "\n").encode()
                    if size + len(line) > RECORD_BYTES:
                        break
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
                                       {"sha256": sha.hexdigest(), "samples": retained, "bytes": size,
                                        "complete": retained == parameters["samples"]})
            receipt = {"source": "telemetry_recording", "recording_id": recording_id, **record}
            self.store.update("operations", operation_id, {"receipt": receipt})
        return "succeeded" if record["complete"] else "unknown", receipt

    def replay(self, plan: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        recording_id = plan["recording_id"]
        if not isinstance(recording_id, str) or re.fullmatch(r"[a-f0-9-]{36}", recording_id) is None:
            raise StudioError("recording", "Invalid recording identity.")
        path = self.store.directory / "recordings" / f"{recording_id}.jsonl"
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_size > RECORD_BYTES:
                raise StudioError("recording", "Recording is not a bounded owner-private regular file.")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                raw = stream.read(RECORD_BYTES + 1)
            if hashlib.sha256(raw).hexdigest() != plan["sha256"]:
                raise StudioError("recording_changed", "Recording changed after review.", 409)
        finally:
            os.close(descriptor)
        ring = RingStore()
        for line in raw.splitlines():
            tick = json.loads(line)
            if not isinstance(tick, dict) or "t" not in tick:
                raise StudioError("recording", "Malformed telemetry recording.")
            ring.push(tick)
        return "succeeded", {"source": "telemetry_replay", "recording_id": recording_id,
                             "sha256": plan["sha256"], "samples": list(ring.ticks), "replay": True}
