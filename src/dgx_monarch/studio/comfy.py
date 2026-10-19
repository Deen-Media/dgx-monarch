"""Fixed-endpoint adapter to an operator-selected ComfyUI driver."""
from __future__ import annotations

import json
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from .contracts import MAX_BODY_BYTES, StudioError

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
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20),
                                             headers={"Authorization": f"Bearer {self.bearer}"} if self.bearer else {})

    async def close(self) -> None:
        if self.session:
            await self.session.close()

    async def request(self, method: str, path: str, data: Any = None) -> Any:
        allowed = GET_PATHS if method == "GET" else POST_PATHS if method == "POST" else frozenset()
        if path not in allowed:
            raise StudioError("driver_endpoint", "The driver endpoint is not allowlisted.")
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
                return json.loads(raw)
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
