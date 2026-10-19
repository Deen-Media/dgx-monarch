"""Fixed-endpoint adapter to an operator-selected ComfyUI driver."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from .contracts import MAX_BODY_BYTES, MAX_EVENT_BYTES, StudioError, bounded_key

GET_PATHS = frozenset({"/object_info", "/queue", "/history", "/system_stats", "/dgxm/telemetry", "/dgxm/consents"})
POST_PATHS = frozenset({"/prompt", "/queue", "/interrupt", "/free", "/dgxm/recycle", "/dgxm/consent"})


def safe_descriptor(value: dict[str, Any]) -> dict[str, str]:
    filename, subfolder = value.get("filename"), value.get("subfolder", "")
    if (not isinstance(filename, str) or not filename or filename in {".", ".."}
            or PurePosixPath(filename).name != filename or "\\" in filename or ":" in filename):
        raise StudioError("artifact_path", "Artifact filename is not a safe basename.")
    if not isinstance(subfolder, str) or "\\" in subfolder or ":" in subfolder or PurePosixPath(subfolder).is_absolute():
        raise StudioError("artifact_path", "Artifact subfolder must be relative.")
    if any(part in {"..", "."} for part in subfolder.split("/")) and subfolder:
        raise StudioError("artifact_path", "Artifact traversal is forbidden.")
    if any(ord(char) < 32 or ord(char) == 127 for char in filename + subfolder):
        raise StudioError("artifact_path", "Artifact path contains control characters.")
    kind = value.get("type", "output")
    if not isinstance(kind, str) or kind not in {"input", "output", "temp"}:
        raise StudioError("artifact_path", "Unknown artifact store.")
    return {"filename": filename, "subfolder": subfolder, "type": kind}


class ComfyClient:
    def __init__(self, url: str, *, bearer: str | None = None):
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            raise StudioError("driver_url", "Use an HTTP(S) driver origin without credentials.")
        if parts.path not in {"", "/"} or parts.query or parts.fragment:
            raise StudioError("driver_url", "Driver URL must be an origin without a path, query or fragment.")
        self.origin = url.rstrip("/")
        self.bearer = bearer
        self.session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        trace = aiohttp.TraceConfig()
        async def no_redirect(session: Any, context: Any, params: Any) -> None:
            raise StudioError("driver_redirect", "Driver redirects are not permitted.", 502)
        trace.on_request_redirect.append(no_redirect)
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20),
                                             headers={"Authorization": f"Bearer {self.bearer}"} if self.bearer else {},
                                             trace_configs=[trace])

    async def close(self) -> None:
        if self.session:
            await self.session.close()

    async def request(self, method: str, path: str, data: Any = None) -> Any:
        allowed = GET_PATHS if method == "GET" else POST_PATHS if method == "POST" else frozenset()
        if path not in allowed:
            raise StudioError("driver_endpoint", "The driver endpoint is not allowlisted.")
        return await self._request(method, path, data)

    async def execution_events(self, after: int, producer: str | None = None) -> Any:
        if type(after) is not int or after < 0:
            raise ValueError("Invalid runtime event cursor")
        from urllib.parse import urlencode

        query = {"after": str(after)}
        if producer is not None:
            query["producer"] = bounded_key(producer, "event producer")
        return await self._request("GET", "/dgxm/execution-events?" + urlencode(query))

    async def history_prompt(self, prompt_id: str) -> Any:
        return await self._request("GET", "/history/" + bounded_key(prompt_id, "prompt id"))

    async def cancel_prompt(self, prompt_id: str) -> dict[str, Any]:
        # This upstream API checks identity and interrupts under the queue mutex.
        # /interrupt is intentionally not used: even its prompt_id variant races.
        result = await self._request("POST", "/api/jobs/" + bounded_key(prompt_id, "prompt id") + "/cancel", {})
        if not isinstance(result, dict) or type(result.get("cancelled")) is not bool:
            raise StudioError("cancellation_unknown", "Driver did not confirm a prompt-specific cancellation request.", 503)
        return result

    async def stream_events(self, client_id: str) -> AsyncIterator[dict[str, Any]]:
        from .live_preview import FRAME_BYTES, parse_frame

        if self.session is None:
            raise StudioError("driver_offline", "Driver is unavailable.", 503)
        try:
            async with self.session.ws_connect(self.origin + "/ws", params={"clientId": bounded_key(client_id)},
                                               heartbeat=15, max_msg_size=FRAME_BYTES + 16392) as socket:
                await socket.send_json({"type": "feature_flags", "data": {"supports_preview_metadata": True}})
                async for message in socket:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        if len(message.data.encode()) > MAX_EVENT_BYTES:
                            continue
                        value = json.loads(message.data)
                        if isinstance(value, dict):
                            yield value
                    elif message.type == aiohttp.WSMsgType.BINARY:
                        frame = parse_frame(message.data)
                        if frame is not None:
                            yield frame
                    elif message.type == aiohttp.WSMsgType.ERROR:
                        raise StudioError("driver_events", "Driver event stream disconnected.", 503)
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            raise StudioError("driver_events", "Driver event stream disconnected; reconcile snapshots.", 503) from exc

    async def _request(self, method: str, path: str, data: Any = None) -> Any:
        if self.session is None:
            raise StudioError("driver_offline", "The driver connection is not initialized.", 503)
        params = {"max_items": "200"} if path == "/history" else None
        try:
            action = {"/dgxm/recycle": "recycle", "/dgxm/consent": "consent"}.get(path)
            headers = {"Origin": self.origin, "X-DGXM-Action": action} if action else None
            async with self.session.request(method, self.origin + path, json=data if method == "POST" else None,
                                            params=params, headers=headers, allow_redirects=False) as response:
                raw = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    raw.extend(chunk)
                    if len(raw) > MAX_BODY_BYTES:
                        raise StudioError("driver_response_limit", "Driver response exceeds its bounded size.", 502)
                if response.status >= 300:
                    # Raw backend errors may contain prompts, credentials or local paths.
                    raise StudioError("driver_rejected", f"Driver returned HTTP {response.status}; inspect its local diagnostics.", 502)
                # ComfyUI queue deletion returns an empty successful response.
                return json.loads(raw) if raw else {}
        except (aiohttp.ClientError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise StudioError("driver_unknown", "The driver did not return a complete response; reconcile before retrying.", 503) from exc

    async def media(self, descriptor: dict[str, Any]) -> aiohttp.ClientResponse:
        if self.session is None:
            raise StudioError("driver_offline", "Driver is unavailable.", 503)
        response = await self.session.get(self.origin + "/view", params=safe_descriptor(descriptor), allow_redirects=False)
        if response.status != 200:
            response.release()
            raise StudioError("artifact_unavailable", "Driver cannot serve this artifact.", 404)
        return response
