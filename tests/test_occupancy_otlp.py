"""Occupancy reaches an OTLP collector as gauges holding the level, across a restart.

torchmonarch 0.8.0's meter makes gauges, so each occupancy value is set to the
level it reads. The probe drives the driver's real policy admission and metrics
exporter into torchmonarch's real OTLP exporter; a loopback sink here decodes
the posts. With metrics turned off and on again, and the level changed while
they were off, the collector reads the level before, during and after.
"""
from __future__ import annotations

import gzip
import json
import os
import struct
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

PROBE = Path(__file__).with_name("occupancy_otlp_probe.py")
SRC = Path(__file__).resolve().parents[1] / "src"
NAME = "dgxm.mesh.active_leases"


def _fields(buf: bytes):
    """(number, value) for one protobuf message; LEN values stay bytes."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = buf[i:i + 8], i + 8
        elif wire == 2:
            size, i = _varint(buf, i)
            value, i = buf[i:i + size], i + size
        elif wire == 5:
            value, i = buf[i:i + 4], i + 4
        else:
            raise ValueError(f"unexpected wire type {wire}")
        yield number, value


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, i
        shift += 7


def _metrics(body: bytes) -> list[tuple[str, str, list[float]]]:
    """(name, kind, point values) for each metric in an ExportMetricsServiceRequest."""
    found = []
    for resource in (v for n, v in _fields(body) if n == 1):
        for scope in (v for n, v in _fields(resource) if n == 2):
            for metric in (v for n, v in _fields(scope) if n == 2):
                name, kind, values = "", "", []
                for number, value in _fields(metric):
                    if number == 1:
                        name = value.decode()
                    elif number in (5, 7):  # Gauge, Sum
                        kind = "gauge" if number == 5 else "sum"
                        for point in (v for n, v in _fields(value) if n == 1):
                            for pn, pv in _fields(point):
                                if pn == 4:
                                    values.append(struct.unpack("<d", pv)[0])
                                elif pn == 6:
                                    values.append(float(struct.unpack("<q", pv)[0]))
                found.append((name, kind, values))
    return found


class _Sink(BaseHTTPRequestHandler):
    seen: list = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        if self.path.endswith("/v1/metrics"):
            _Sink.seen.extend(entry for entry in _metrics(body) if entry[0] == NAME)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *_args):
        return None


def test_the_collector_reads_a_gauge_holding_the_level_across_a_restart(tmp_path):
    pytest.importorskip("monarch.actor")
    _Sink.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Sink)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("MONARCH_", "HYPERACTOR_", "OTEL_", "DGXM_MONARCH_POLICY"))}
    environment.update(
        HOME=str(tmp_path), XDG_RUNTIME_DIR=str(runtime_dir), CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=os.pathsep.join(filter(None, (str(SRC), os.environ.get("PYTHONPATH")))),
        OTEL_EXPORTER_OTLP_ENDPOINT=f"http://127.0.0.1:{server.server_port}")
    try:
        proc = subprocess.run([sys.executable, str(PROBE)], capture_output=True, text=True,
                              timeout=180, env=environment)
    finally:
        server.shutdown()
        server.server_close()
    assert proc.returncode == 0, f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == {"first": True, "again": True}
    assert _Sink.seen, "no occupancy point reached the collector"
    assert {kind for _name, kind, _values in _Sink.seen} == {"gauge"}, _Sink.seen
    # One stream per name: the restarted exporter's gauge is the same stream.
    assert all(len(values) == 1 for _name, _kind, values in _Sink.seen), _Sink.seen
    levels = [values[0] for _name, _kind, values in _Sink.seen]
    in_order = [level for i, level in enumerate(levels) if i == 0 or level != levels[i - 1]]
    # 3 while on, still 3 while off, 5 at the restart's first sample, then 0.
    assert in_order == [3.0, 5.0, 0.0], levels
