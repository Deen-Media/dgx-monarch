"""CPU-only event fixture: mock_driver.py plus the driver events ComfyUI sends.

This is not an inference engine. It serves mock_driver.py's routes unchanged and
adds what that fixture leaves out, so the browser checks can see a step count, a
node that reports no steps, and a refused prompt. Bind only to loopback; do not
point it at a real driver, cluster config or management service.

- ``GET /ws`` opens a WebSocket. For each prompt it sends the events ComfyUI
  v0.43.0 sends: ``execution_start`` and ``execution_cached``, ``executing`` for
  node 1, ``progress`` 1 to 20 of 20 on node 1 over the first 40 percent of the
  job, ``executing`` for node 2 (which reports no steps) at 47 percent, then
  ``execution_success`` or ``execution_error``, then ``executing`` with no node.
  As in ComfyUI, a prompt sent with a ``client_id`` gets every event on that
  client's socket only, and a prompt without one gets ``progress`` and
  ``executing`` on every socket and no ``execution_*`` event. A socket gets only
  the events due after it connects.
- A prompt whose text holds ``QA-FAIL-p`` or ``QA-FAIL-c`` stops at one third of
  the job with a refusal built by dgx_monarch's own builders: a synthetic class P
  refusal that cites TROUBLESHOOTING #135, or the class C capacity rescue with its
  panel remedy and consent descriptor. Its history entry says error and lists no
  outputs, as ComfyUI writes one.
- ``--schema`` answers ``/object_info`` with a recorded schema, such as
  ``tests/fixtures/studio/object_info_bundled.json``, plus the fixture node, so
  packaged templates inspect with typed widgets.

Run it in the Python environment where dgx_monarch is installed, which the
refusals need. The companion reads it like any driver.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import re
import struct
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
STEPS = 20
# Fractions of the job: steps from 1/30 to about 0.41, node 2 at 0.47, a refusal at 1/3.
FIRST_STEP, STEP_GAP, SECOND_NODE, FAIL_AT = 1 / 30, 0.02, 14 / 30, 1 / 3
MARKER = re.compile(r"QA-FAIL-([pc])\b")
MAX_FRAME = 1 << 20


def load_fixture():
    """mock_driver.py from this directory, loaded by path so its CLI stays the one entry point."""
    spec = importlib.util.spec_from_file_location("mock_driver", Path(__file__).with_name("mock_driver.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def refusals() -> dict[str, dict[str, str]]:
    """The two refusal texts, built as dgx_monarch builds them; see test_studio_failure_parse.py."""
    from dgx_monarch import consent_descriptor
    from dgx_monarch.refusal import PanelAction, RefusalClass, refusal

    headline = ("capacity rescue available for flux/dev.safetensors on spark-a: stock residency needs 52.1 GiB "
                "of unified memory and only 40.2 GiB is available.")
    descriptor = consent_descriptor.ConsentDescriptor(
        version=consent_descriptor.CONSENT_DESCRIPTOR_VERSION,
        kind=consent_descriptor.KIND_RESCUE_SLAB,
        consent_id=consent_descriptor.consent_id("rescue-slab", "size:35000000000", "c0ffee00c0ffee00"),
        unet_name="flux/dev.safetensors",
        file_identity="size:35000000000",
        context_fingerprint="c0ffee00c0ffee00",
        family_hint="flux",
        measured={"checkpoint_bytes": 35_000_000_000, "mem_available_bytes": 43_000_000_000, "host": "spark-a"},
        memo_context={"combo_key": "flux/dev.safetensors"},
        human_reason=("flux/dev.safetensors needs 52.1 GiB of unified memory under stock residency and only "
                      "40.2 GiB is available. Zero-copy slab residency fits it."),
        evidence=consent_descriptor.EVIDENCE_BYTE_VERIFIED,
        panel_action="Load with slab residency",
        env_fallback=consent_descriptor.ENV_SLAB_RESCUE,
    )
    physics = refusal(
        RefusalClass.PHYSICS,
        "CPU fixture: a synthetic class P refusal for the browser checks. It stands for a render the sharded "
        "attention kernel cannot run, such as a MOSS-Image text attention bias under sequence parallelism; "
        "nothing ran on a GPU.",
        troubleshooting=135)
    capacity = refusal(
        RefusalClass.CAPACITY, f"{headline} Zero-copy slab residency fits this checkpoint.",
        guard="stock_load_preflight", waivable=True,
        panel_action=PanelAction("Load with slab residency", consent_descriptor.ENV_SLAB_RESCUE),
        troubleshooting=12) + "\n" + consent_descriptor.encode(descriptor)
    return {"p": {"exception_type": "RuntimeError", "exception_message": physics},
            "c": {"exception_type": "RuntimeError", "exception_message": capacity}}


def read_exact(stream, count: int) -> bytes:
    data = b""
    while len(data) < count:
        chunk = stream.read(count - len(data))
        if not chunk:
            raise EOFError
        data += chunk
    return data


def read_frame(stream) -> tuple[int, bytes]:
    first, second = read_exact(stream, 2)
    opcode, masked, length = first & 0x0F, second & 0x80, second & 0x7F
    if length == 126:
        length = struct.unpack(">H", read_exact(stream, 2))[0]
    elif length == 127:
        length = struct.unpack(">Q", read_exact(stream, 8))[0]
    if length > MAX_FRAME:
        raise ValueError("frame too large")
    mask = read_exact(stream, 4) if masked else b""
    payload = read_exact(stream, length)
    if masked:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return opcode, payload


def frame(opcode: int, payload: bytes) -> bytes:
    size = len(payload)
    if size < 126:
        head = bytes([0x80 | opcode, size])
    elif size < 65536:
        head = bytes([0x80 | opcode, 126]) + struct.pack(">H", size)
    else:
        head = bytes([0x80 | opcode, 127]) + struct.pack(">Q", size)
    return head + payload


def build(fixture, texts: dict[str, dict[str, str]]):
    """A handler class over the fixture's, with the events and refusals described above."""
    lock = threading.Lock()
    meta: dict[str, dict] = {}

    def error_data(identity: str, info: dict) -> dict:
        refused = texts[info["fail"]]
        return {"prompt_id": identity, "node_id": "1", "node_type": "StudioCPUFixture", "executed": [],
                "exception_message": refused["exception_message"], "exception_type": refused["exception_type"],
                "traceback": [], "current_inputs": {}, "current_outputs": []}

    def schedule(identity: str, job: dict, info: dict, seconds: float) -> list[tuple[float, str, dict]]:
        start, length = job["created"], info["duration"]

        def stamp(offset: float) -> int:
            return int((start + offset) * 1000)

        events = [(0.0, "execution_start", {"prompt_id": identity, "timestamp": stamp(0.0)}),
                  (0.0, "execution_cached", {"nodes": [], "prompt_id": identity, "timestamp": stamp(0.0)}),
                  (0.0, "executing", {"node": "1", "display_node": "1", "prompt_id": identity})]
        for step in range(1, STEPS + 1):
            at = (FIRST_STEP + (step - 1) * STEP_GAP) * seconds
            if at < length:
                events.append((at, "progress", {"value": step, "max": STEPS, "prompt_id": identity, "node": "1"}))
        if SECOND_NODE * seconds < length:
            events.append((SECOND_NODE * seconds, "executing", {"node": "2", "display_node": "2", "prompt_id": identity}))
        if info["fail"]:
            events.append((length, "execution_error", {**error_data(identity, info), "timestamp": stamp(length)}))
        else:
            events.append((length, "execution_success", {"prompt_id": identity, "timestamp": stamp(length)}))
        events.append((length, "executing", {"node": None, "prompt_id": identity}))
        return events

    class Handler(fixture.Handler):
        schema: dict | None = None

        def finished(self, job: dict) -> bool:
            info = meta.get(job["row"][1], {})
            return time.time() - job["created"] >= info.get("duration", self.job_seconds)

        def entry(self, identity: str, job: dict) -> dict:
            info = meta.get(identity, {})
            if not info.get("fail"):
                return super().entry(identity, job)
            started, ended = int(job["created"] * 1000), int((job["created"] + info["duration"]) * 1000)
            return {"prompt": job["row"], "outputs": {}, "status": {
                "status_str": "error", "completed": False, "messages": [
                    ["execution_start", {"prompt_id": identity, "timestamp": started}],
                    ["execution_error", {**error_data(identity, info), "timestamp": ended}]]}}

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/ws":
                return self.websocket()
            if path == "/object_info" and self.schema is not None:
                return self.respond(self.schema)
            return super().do_GET()

        def prompt(self, body: object):
            with lock:
                before = set(fixture.JOBS)
                super().prompt(body)
                for identity in set(fixture.JOBS) - before:
                    found = MARKER.search(json.dumps(body.get("prompt")))
                    client = body.get("client_id")
                    meta[identity] = {"client": client if isinstance(client, str) else None,
                                      "fail": found.group(1) if found else None,
                                      "duration": self.job_seconds * (FAIL_AT if found else 1.0)}

        def due(self, now: float):
            with lock:
                jobs = [(identity, job, meta.get(identity)) for identity, job in fixture.JOBS.items()]
            for identity, job, info in jobs:
                if info is None:
                    continue
                for index, (at, kind, data) in enumerate(schedule(identity, job, info, self.job_seconds)):
                    if job["created"] + at <= now:
                        yield (identity, index), job["created"] + at, kind, data, info["client"]

        def websocket(self):
            key = self.headers.get("Sec-WebSocket-Key", "")
            if self.headers.get("Upgrade", "").lower() != "websocket" or not key:
                return self.respond({"error": "fixture route unavailable"}, status=404)
            # RFC 6455 fixes SHA-1 for the handshake; it protects nothing here.
            digest = hashlib.sha1((key + GUID).encode(), usedforsecurity=False).digest()
            accept = base64.b64encode(digest).decode()
            self.wfile.write(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                              f"Sec-WebSocket-Accept: {accept}\r\n\r\n").encode())
            self.wfile.flush()
            self.close_connection = True
            client = parse_qs(urlsplit(self.path).query).get("clientId", [None])[0]
            send_lock, closed = threading.Lock(), threading.Event()

            def send(opcode: int, payload: bytes) -> None:
                with send_lock:
                    self.wfile.write(frame(opcode, payload))
                    self.wfile.flush()

            def reader() -> None:
                # Answer pings, which aiohttp's heartbeat sends, and stop on a close frame.
                try:
                    while not closed.is_set():
                        opcode, payload = read_frame(self.rfile)
                        if opcode == 0x8:
                            try:
                                send(0x8, payload[:2])
                            except OSError:
                                pass
                            break
                        if opcode == 0x9:
                            send(0xA, payload)
                except (OSError, EOFError, ValueError):
                    pass
                finally:
                    closed.set()

            sent = {event_key for event_key, *_rest in self.due(time.time())}
            threading.Thread(target=reader, daemon=True).start()
            try:
                send(0x1, json.dumps({"type": "status", "data": {
                    "status": {"exec_info": {"queue_remaining": 0}}, "sid": client}}).encode())
                while not closed.is_set():
                    ready = []
                    for event_key, when, kind, data, target in self.due(time.time()):
                        if event_key in sent:
                            continue
                        sent.add(event_key)
                        if kind.startswith("execution_"):
                            if target is not None and target == client:
                                ready.append((when, kind, data))
                        elif target is None or target == client:
                            ready.append((when, kind, data))
                    for _when, kind, data in sorted(ready, key=lambda item: item[0]):
                        send(0x1, json.dumps({"type": kind, "data": data}).encode())
                    closed.wait(0.05)
            except OSError:
                pass
            finally:
                closed.set()

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--port", type=int, default=18195)
    parser.add_argument("--job-seconds", type=float, default=30.0,
                        help="seconds each prompt runs before it completes (default: 30)")
    parser.add_argument("--video", type=Path, help="an MP4 file to serve as a second output of every prompt")
    parser.add_argument("--schema", type=Path, help="a recorded /object_info answer to serve with the fixture node")
    parser.add_argument("--export-workflow", type=Path, help="write the explicit CPU fixture workflow JSON")
    args = parser.parse_args()
    if not 1 <= args.job_seconds <= 3600:
        parser.error("--job-seconds must be between 1 and 3600")
    try:
        texts = refusals()
    except ImportError:
        parser.error("dgx_monarch is not importable; run this fixture in the Studio Python environment")
    fixture = load_fixture()
    handler = build(fixture, texts)
    handler.job_seconds = args.job_seconds
    handler.video = args.video.read_bytes() if args.video else None
    if args.schema:
        handler.schema = {**json.loads(args.schema.read_text()), **fixture.SCHEMA}
    if args.export_workflow:
        args.export_workflow.write_text(json.dumps(fixture.WORKFLOW, indent=2))
    print(f"CPU-only event fixture: http://127.0.0.1:{args.port}; ComfyUI-shaped events, no inference or "
          "management", flush=True)
    fixture.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
