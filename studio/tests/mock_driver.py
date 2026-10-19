"""CPU-only HTTP fixture for manually checking the real Studio companion/browser.

This is not an inference engine. Every returned image and telemetry value is
synthetic and explicitly marked as a fixture. Bind only to loopback; do not point
this fixture at a real driver, cluster config, or management service.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

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
JOBS: dict[str, dict] = {}


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


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def respond(self, body, content_type="application/json", status=200):
        data = json.dumps(body).encode() if content_type == "application/json" else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/object_info":
            return self.respond(SCHEMA)
        if path == "/fixture-workflow":
            return self.respond(WORKFLOW)
        if path == "/queue":
            rows = [job["row"] for job in JOBS.values() if time.time() - job["created"] < 1]
            return self.respond({"queue_running": rows, "queue_pending": []})
        if path == "/history":
            return self.respond({identity: {
                "prompt": job["row"],
                "outputs": {"1": {"images": [{"filename": "cpu-fixture.png", "subfolder": "", "type": "output", "width": 128, "height": 96}]}},
                "status": {"completed": True, "status_str": "success", "messages": []},
            } for identity, job in JOBS.items() if time.time() - job["created"] >= 1})
        if path == "/view":
            return self.respond(image_bytes(), "image/png")
        if path == "/dgxm/telemetry":
            return self.respond({"fixture": True, "source": "LOCAL CPU HTTP FIXTURE - NOT GPU", "render": {"state": "idle"}})
        return self.respond({"error": "fixture route unavailable"}, status=404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        if self.path == "/prompt":
            identity = str(uuid.uuid4())
            JOBS[identity] = {"created": time.time(), "row": [len(JOBS), identity, body["prompt"], body.get("extra_data", {}), ["1"]]}
            return self.respond({"prompt_id": identity, "number": len(JOBS), "node_errors": {}})
        return self.respond({"error": "fixture mutation unavailable"}, status=404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18188)
    parser.add_argument("--export-workflow", help="write the explicit CPU fixture workflow JSON")
    args = parser.parse_args()
    if args.export_workflow:
        from pathlib import Path
        Path(args.export_workflow).write_text(json.dumps(WORKFLOW, indent=2))
    print(f"CPU-only fixture: http://127.0.0.1:{args.port}; no inference or management", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
