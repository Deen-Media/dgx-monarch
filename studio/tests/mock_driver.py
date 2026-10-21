"""CPU-only HTTP fixture for manually checking the real Studio companion/browser.

This is not an inference engine. Every returned image and telemetry value is
synthetic and explicitly marked as a fixture. Bind only to loopback; do not point
this fixture at a real driver, cluster config, or management service.

The routes follow the ComfyUI shapes the companion reads: POST /prompt honors a
client-supplied canonical prompt_id, GET /history/{prompt_id} returns {} until
the prompt finishes, and output descriptors carry only filename, subfolder and
type, because ComfyUI's savers report no width, height, frame rate or duration.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
import uuid
import zlib
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlsplit

SCHEMA = {
    "StudioCPUFixture": {
        "input": {"required": {
            "prompt": ["STRING", {"default": "LOCAL CPU FIXTURE - NOT GPU"}],
            "seed": ["INT", {"default": 7, "min": 0, "max": 1000000}],
            "strength": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0}],
        }},
        "output": ["IMAGE"], "output_node": True,
    }
}
WORKFLOW = {"1": {"class_type": "StudioCPUFixture", "inputs": {
    "prompt": "LOCAL CPU FIXTURE - NOT GPU", "seed": 7, "strength": 1.0,
}}}
# The exact descriptor shape ComfyUI's SaveImage and SaveVideo report.
OUTPUT = {"filename": "cpu-fixture.png", "subfolder": "", "type": "output"}
VIDEO_OUTPUT = {"filename": "cpu-fixture.mp4", "subfolder": "", "type": "output"}
MAX_BODY_BYTES = 16 * 1024 * 1024
JOBS: dict[str, dict] = {}
UPLOADS: dict[tuple[str, str], bytes] = {}


def image_bytes() -> bytes:
    """A tiny deterministic RGB test pattern; generated entirely on the CPU."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    rows = []
    for y in range(96):
        row = bytearray([0])
        for x in range(128):
            row.extend((20 + x, 30 + y * 2, 65 + ((x // 16 + y // 16) % 2) * 80))
        rows.append(row)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 128, 96, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(b"".join(rows))) + chunk(b"IEND", b"")


def canonical_prompt_id(value: object) -> str | None:
    """The id ComfyUI accepts: a UUID string in canonical lowercase form."""
    if not isinstance(value, str):
        return None
    try:
        return value if str(uuid.UUID(value)) == value else None
    except ValueError:
        return None


class Handler(BaseHTTPRequestHandler):
    # Seconds a prompt stays in queue_running before it appears in history.
    job_seconds = 1.0
    # Bytes of an owner-supplied video served as a second output, or None.
    video: bytes | None = None

    def log_message(self, *_args):
        pass

    def respond(self, body, content_type="application/json", status=200):
        data = json.dumps(body).encode() if content_type == "application/json" else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def finished(self, job: dict) -> bool:
        return time.time() - job["created"] >= self.job_seconds

    def entry(self, identity: str, job: dict) -> dict:
        outputs = {"1": {"images": [dict(OUTPUT)]}}
        if self.video is not None:
            outputs["2"] = {"images": [dict(VIDEO_OUTPUT)], "animated": [True]}
        started, done = int(job["created"] * 1000), int((job["created"] + self.job_seconds) * 1000)
        messages = [["execution_start", {"prompt_id": identity, "timestamp": started}],
                    ["execution_success", {"prompt_id": identity, "timestamp": done}]]
        return {"prompt": job["row"], "outputs": outputs,
                "status": {"status_str": "success", "completed": True, "messages": messages}}

    def history(self, query: dict[str, list[str]]) -> dict:
        done = [(identity, job) for identity, job in JOBS.items() if self.finished(job)]
        limit = query.get("max_items", [None])[0]
        if limit is not None and limit.isdigit():
            done = done[max(0, len(done) - int(limit)):]
        return {identity: self.entry(identity, job) for identity, job in done}

    def do_GET(self):
        parts = urlsplit(self.path)
        path, query = parts.path, parse_qs(parts.query)
        if path == "/object_info":
            return self.respond(SCHEMA)
        if path == "/fixture-workflow":
            return self.respond(WORKFLOW)
        if path == "/queue":
            rows = [job["row"] for job in JOBS.values() if not self.finished(job)]
            return self.respond({"queue_running": rows, "queue_pending": []})
        if path == "/history":
            return self.respond(self.history(query))
        if path.startswith("/history/"):
            identity = path.removeprefix("/history/")
            job = JOBS.get(identity)
            return self.respond({identity: self.entry(identity, job)} if job and self.finished(job) else {})
        if path == "/view":
            return self.view(query)
        if path == "/dgxm/telemetry":
            return self.respond({"fixture": True, "source": "LOCAL CPU HTTP FIXTURE - NOT GPU", "render": {"state": "idle"}})
        return self.respond({"error": "fixture route unavailable"}, status=404)

    def view(self, query: dict[str, list[str]]):
        filename = query.get("filename", [""])[0]
        kind, subfolder = query.get("type", ["output"])[0], query.get("subfolder", [""])[0]
        if kind == "input" and (subfolder, filename) in UPLOADS:
            return self.respond(UPLOADS[(subfolder, filename)], "image/png")
        if kind == "output" and filename == VIDEO_OUTPUT["filename"] and self.video is not None:
            return self.respond(self.video, "video/mp4")
        if kind == "output" and filename == OUTPUT["filename"]:
            return self.respond(image_bytes(), "image/png")
        return self.respond({"error": "fixture file unavailable"}, status=404)

    def body(self) -> bytes | None:
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= MAX_BODY_BYTES:
            self.respond({"error": "fixture request too large"}, status=413)
            return None
        return self.rfile.read(length)

    def do_POST(self):
        raw = self.body()
        if raw is None:
            return None
        if self.path == "/prompt":
            try:
                body = json.loads(raw)
            except ValueError:
                return self.respond({"error": "fixture prompt is not JSON"}, status=400)
            return self.prompt(body)
        if self.path == "/upload/image":
            return self.upload(raw)
        return self.respond({"error": "fixture mutation unavailable"}, status=404)

    def prompt(self, body: object):
        if not isinstance(body, dict) or "prompt" not in body:
            return self.respond({"error": {"type": "no_prompt", "message": "No prompt provided",
                                           "details": "", "extra_info": {}}, "node_errors": {}}, status=400)
        supplied = body.get("prompt_id")
        identity = str(uuid.uuid4()) if supplied is None else canonical_prompt_id(supplied)
        if identity is None:
            return self.respond({"error": {"type": "invalid_prompt_id", "message": "prompt_id must be a valid UUID",
                                           "details": "", "extra_info": {}}, "node_errors": {}}, status=400)
        number = len(JOBS)
        JOBS[identity] = {"created": time.time(), "row": [number, identity, body["prompt"], body.get("extra_data", {}), ["1"]]}
        return self.respond({"prompt_id": identity, "number": number, "node_errors": {}})

    def upload(self, raw: bytes):
        header = ("Content-Type: " + self.headers.get("Content-Type", "") + "\r\n\r\n").encode()
        message = BytesParser(policy=email_policy).parsebytes(header + raw)
        fields = {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}
        image = fields.get("image")
        name = PurePosixPath(image.get_filename() or "").name if image is not None else ""
        subfolder = fields["subfolder"].get_content().strip() if "subfolder" in fields else ""
        if not name or name in {".", ".."} or ".." in subfolder.split("/"):
            return self.respond({"error": "fixture upload needs a safe image file"}, status=400)
        UPLOADS[(subfolder, name)] = image.get_payload(decode=True)
        return self.respond({"name": name, "subfolder": subfolder, "type": "input"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18188)
    parser.add_argument("--export-workflow", type=Path, help="write the explicit CPU fixture workflow JSON")
    parser.add_argument("--job-seconds", type=float, default=1.0,
                        help="seconds each prompt stays running before it completes (default: 1)")
    parser.add_argument("--video", type=Path, help="an MP4 file to serve as a second output of every prompt")
    args = parser.parse_args()
    if not 0 <= args.job_seconds <= 3600:
        parser.error("--job-seconds must be between 0 and 3600")
    if args.export_workflow:
        args.export_workflow.write_text(json.dumps(WORKFLOW, indent=2))
    Handler.job_seconds = args.job_seconds
    Handler.video = args.video.read_bytes() if args.video else None
    print(f"CPU-only fixture: http://127.0.0.1:{args.port}; no inference or management", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
