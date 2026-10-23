"""Fixed-endpoint adapter to an operator-selected ComfyUI driver."""
from __future__ import annotations

import errno
import unicodedata
from collections.abc import AsyncIterator
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from .. import finite_json
from .contracts import MAX_BODY_BYTES, MAX_EVENT_BYTES, OBJECT_INFO_BYTES, StudioError, bounded_key

GET_PATHS = frozenset({"/object_info", "/queue", "/history", "/system_stats", "/dgxm/telemetry", "/dgxm/consents"})
POST_PATHS = frozenset({"/prompt", "/queue", "/interrupt", "/free", "/dgxm/recycle", "/dgxm/consent"})
# A refusal is a refusal at any size; only this much of its body is read.
ERROR_BODY_BYTES = 256 * 1024
NODE_ERROR_ROWS = 64
DETAIL_CHARACTERS = 300
DETAIL_PATHS = frozenset({"/dgxm/consent", "/dgxm/recycle"})
HISTORY_WINDOW = 200
# A page at most half the body cap can double without a forecast overflow.
# The window doubles after this many such pages in a row, and the count
# doubles (up to the limit) each time a doubled window overflows again.
HISTORY_REGROW_BYTES = MAX_BODY_BYTES // 2
HISTORY_REGROW_PAGES = 3
HISTORY_HOLD_LIMIT = 64
# Only a refused connect is an observed "nothing listens"; every other failure stays unknown.
DRIVER_NOT_RUNNING = "No driver accepted a connection at the driver origin; start the driver."


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


def _route(path: str) -> str:
    """Name an endpoint in a refusal without echoing identities or queries."""
    route = path.split("?", 1)[0]
    if route.startswith("/history/"):
        return "/history/{prompt_id}"
    if route.startswith("/api/jobs/"):
        return "/api/jobs/{prompt_id}/cancel"
    return route


def _clean(value: Any, limit: int) -> str | None:
    """Bounded text with control and format characters removed."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    text = "".join(" " if unicodedata.category(char)[0] == "C" else char for char in str(value))
    return " ".join(text.split())[:limit] or None


def node_error_rows(body: Any) -> list[dict[str, str | None]]:
    """Name each refused input of a prompt; values, messages and details stay with the driver."""
    rows: list[dict[str, str | None]] = []
    if not isinstance(body, dict):
        return rows
    errors = body.get("node_errors")
    for node_id, entry in (errors.items() if isinstance(errors, dict) else ()):
        reasons = entry.get("errors") if isinstance(entry, dict) else None
        for reason in reasons if isinstance(reasons, list) else ():
            if len(rows) == NODE_ERROR_ROWS:
                return rows
            if not isinstance(reason, dict):
                continue
            extra = reason.get("extra_info")
            rows.append({"node_id": _clean(node_id, 128), "class_type": _clean(entry.get("class_type"), 128),
                         "input_name": _clean(extra.get("input_name"), 128) if isinstance(extra, dict) else None,
                         "type": _clean(reason.get("type"), 64)})
    error = body.get("error")
    if not rows and isinstance(error, dict) and error.get("type") is not None:
        # A graph-level refusal (a missing node class, no outputs) names no input.
        extra = error.get("extra_info")
        extra = extra if isinstance(extra, dict) else {}
        rows.append({"node_id": _clean(extra.get("node_id"), 128), "class_type": _clean(extra.get("class_type"), 128),
                     "input_name": None, "type": _clean(error.get("type"), 64)})
    return rows


def _parsed(raw: bytes) -> Any:
    try:
        return finite_json.loads(raw)
    except (ValueError, RecursionError, OverflowError):
        return None


def _short_message(raw: bytes) -> str | None:
    """The driver's own short refusal text: a JSON detail, or a plain-text body."""
    body = _parsed(raw)
    if isinstance(body, dict):
        text = body.get("detail", body.get("error"))
        text = text.get("message") if isinstance(text, dict) else text
    else:
        text = raw.decode("utf-8", "replace") if body is None else None
    return _clean(text, DETAIL_CHARACTERS) if isinstance(text, str) else None


def rejection(path: str, status: int, raw: bytes) -> StudioError:
    """A driver refusal carrying its HTTP status and a bounded, cleaned detail."""
    # Raw backend errors may contain prompts, credentials or local paths.
    error = StudioError("driver_rejected", f"Driver returned HTTP {status}; inspect its local diagnostics.", 502)
    error.upstream_status = status
    if path == "/prompt":
        body = _parsed(raw)
        error.upstream_detail = node_error_rows(body) if isinstance(body, dict) else None
    elif path in DETAIL_PATHS:
        error.upstream_detail = _short_message(raw)
    return error


class ComfyClient:
    def __init__(self, url: str):
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            raise StudioError("driver_url", "Use an HTTP(S) driver origin without credentials.")
        if parts.path not in {"", "/"} or parts.query or parts.fragment:
            raise StudioError("driver_url", "Driver URL must be an origin without a path, query or fragment.")
        self.origin = url.rstrip("/")
        self.session: aiohttp.ClientSession | None = None
        # The driver state reads history_limit to show a shrunk window.
        self.history_limit = HISTORY_WINDOW
        self._history_quiet = 0
        self._history_hold = HISTORY_REGROW_PAGES
        self._history_regrown = False

    async def start(self) -> None:
        trace = aiohttp.TraceConfig()
        async def no_redirect(session: Any, context: Any, params: Any) -> None:
            raise StudioError("driver_redirect", "Driver redirects are not permitted.", 502)
        trace.on_request_redirect.append(no_redirect)
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20), trace_configs=[trace])

    async def close(self) -> None:
        if self.session:
            await self.session.close()

    async def request(self, method: str, path: str, data: Any = None) -> Any:
        allowed = GET_PATHS if method == "GET" else POST_PATHS if method == "POST" else frozenset()
        if path not in allowed:
            raise StudioError("driver_endpoint", "The driver endpoint is not allowlisted.")
        if method == "GET" and path == "/history":
            return await self._history()
        return await self._request(method, path, data)

    async def _history(self) -> Any:
        # Workflow payloads vary widely: a count bound alone does not fit the
        # byte budget. Halve the window on an oversized page and double it
        # after a run of small pages, so large rows that have aged out of the
        # window cannot keep it small for the life of the process.
        while True:
            try:
                value, size = await self._exchange("GET", "/history", None, history_limit=self.history_limit)
            except StudioError as exc:
                if exc.code != "driver_response_limit" or self.history_limit == 1:
                    raise
                if self._history_regrown:
                    self._history_hold = min(self._history_hold * 2, HISTORY_HOLD_LIMIT)
                self._history_regrown = False
                self._history_quiet = 0
                self.history_limit = max(1, self.history_limit // 2)
                continue
            self._history_regrown = False
            if self.history_limit >= HISTORY_WINDOW:
                self._history_quiet, self._history_hold = 0, HISTORY_REGROW_PAGES
            elif size > HISTORY_REGROW_BYTES:
                self._history_quiet = 0
            else:
                self._history_quiet += 1
                if self._history_quiet >= self._history_hold:
                    self.history_limit = min(HISTORY_WINDOW, self.history_limit * 2)
                    self._history_quiet, self._history_regrown = 0, True
            return value

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
                        value = finite_json.loads(message.data)
                        if isinstance(value, dict):
                            yield value
                    elif message.type == aiohttp.WSMsgType.BINARY:
                        frame = parse_frame(message.data)
                        if frame is not None:
                            yield frame
                    elif message.type == aiohttp.WSMsgType.ERROR:
                        raise StudioError("driver_events", "Driver event stream disconnected.", 503)
        except (aiohttp.ClientError, TimeoutError, ValueError, RecursionError, OverflowError) as exc:
            raise StudioError("driver_events", "Driver event stream disconnected; reconcile snapshots.", 503) from exc

    async def _request(self, method: str, path: str, data: Any = None) -> Any:
        value, _size = await self._exchange(method, path, data)
        return value

    async def _exchange(self, method: str, path: str, data: Any, *,
                        history_limit: int = HISTORY_WINDOW) -> tuple[Any, int]:
        """Return the parsed body and its size in bytes."""
        if self.session is None:
            raise StudioError("driver_offline", "The driver connection is not initialized.", 503)
        params = {"max_items": str(history_limit)} if path == "/history" else None
        budget = OBJECT_INFO_BYTES if path == "/object_info" else MAX_BODY_BYTES
        try:
            action = {"/dgxm/recycle": "recycle", "/dgxm/consent": "consent"}.get(path)
            headers = {"Origin": self.origin, "X-DGXM-Action": action} if action else None
            async with self.session.request(method, self.origin + path, json=data if method == "POST" else None,
                                            params=params, headers=headers, allow_redirects=False) as response:
                refused = response.status >= 300
                raw = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    raw.extend(chunk)
                    if refused and len(raw) > ERROR_BODY_BYTES:
                        break
                    if len(raw) > budget:
                        raise StudioError("driver_response_limit", f"Driver response from {_route(path)} exceeds "
                                          f"its {budget // (1024 * 1024)} MiB budget.", 502)
                if refused:
                    raise rejection(path, response.status, bytes(raw[:ERROR_BODY_BYTES]))
                # ComfyUI queue deletion returns an empty successful response.
                return (finite_json.loads(raw) if raw else {}), len(raw)
        except StudioError:
            raise
        except (aiohttp.ClientError, TimeoutError, ValueError, RecursionError, OverflowError) as exc:
            if (isinstance(exc, aiohttp.ClientConnectorError)
                    and getattr(exc.os_error, "errno", None) == errno.ECONNREFUSED):
                # Nothing accepted the connection, so the request never reached a driver.
                raise StudioError("driver_not_running", DRIVER_NOT_RUNNING, 503) from exc
            # ValueError covers malformed JSON, bad UTF-8 and integers past the
            # digit limit; RecursionError covers JSON nested too deeply.
            raise StudioError("driver_unknown", "The driver did not return a complete response; "
                              "reconcile before retrying.", 503) from exc

    async def media(self, descriptor: dict[str, Any]) -> aiohttp.ClientResponse:
        if self.session is None:
            raise StudioError("driver_offline", "Driver is unavailable.", 503)
        response = await self.session.get(self.origin + "/view", params=safe_descriptor(descriptor), allow_redirects=False)
        if response.status != 200:
            response.release()
            raise StudioError("artifact_unavailable", "Driver cannot serve this artifact.", 404)
        return response
