"""Contracts for the opt-in, root-authenticated setup inspection client."""

from __future__ import annotations

import hashlib
import json
import stat
import struct
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import setup_process_inspection as subject

UID = 1000
BOOT_ID = "test-boot-id"
SOURCE = "a" * 64
ADDRESS = "tcp://192.0.2.10:26600"


def _report(**changes):
    value = {
        "schema": 1,
        "status": "ok",
        "uid": UID,
        "boot_id": BOOT_ID,
        "start_monotonic_ns": 100,
        "end_monotonic_ns": 190,
        "source_sha256": SOURCE,
        "workers": [],
        "actors": [],
        "unknown_count": 0,
    }
    value.update(changes)
    return value


def _worker(pid=41, hashes=None):
    return {
        "pid": pid,
        "starttime": 7,
        "address_hashes": hashes if hashes is not None else [hashlib.sha256(ADDRESS.encode()).hexdigest()],
    }


def test_checked_report_accepts_only_fresh_same_identity_facts():
    report = _report(workers=[_worker()], actors=[{"pid": 42, "starttime": 8}])
    assert subject._checked_process_report(report, UID, BOOT_ID, SOURCE, 100, 200) is report
    edge = _report(start_monotonic_ns=100, end_monotonic_ns=100)
    assert subject._checked_process_report(edge, UID, BOOT_ID, SOURCE, 100, 5_000_000_100) is edge


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"schema": True}, "identity"),
        ({"uid": True}, "identity"),
        ({"boot_id": "other"}, "identity"),
        ({"source_sha256": "b" * 64}, "identity"),
        ({"end_monotonic_ns": 200 - 5_000_000_001}, "stale"),
        ({"start_monotonic_ns": 99}, "stale"),
        ({"unknown_count": True}, "certainty"),
        ({"status": "unknown"}, "certainty"),
    ],
)
def test_checked_report_rejects_noncanonical_identity_and_freshness(changes, match):
    with pytest.raises(ValueError, match=match):
        subject._checked_process_report(_report(**changes), UID, BOOT_ID, SOURCE, 100, 200)


@pytest.mark.parametrize(
    "workers",
    [
        [_worker(), _worker()],
        [_worker(hashes=["f" * 63])],
        [_worker(hashes=["F" * 64])],
        [_worker(hashes=["a" * 64, "a" * 64])],
        [_worker(hashes=["f" * 64, "0" * 64])],
        [{"pid": True, "starttime": 7, "address_hashes": []}],
        [{"pid": 41, "starttime": True, "address_hashes": []}],
    ],
)
def test_checked_report_rejects_duplicate_processes_and_bad_endpoint_hashes(workers):
    with pytest.raises(ValueError):
        subject._checked_process_report(_report(workers=workers), UID, BOOT_ID, SOURCE, 100, 200)


def test_checked_report_rejects_boolean_actor_lifetime_fields():
    with pytest.raises(ValueError):
        subject._checked_process_report(
            _report(actors=[{"pid": 42, "starttime": True}]), UID, BOOT_ID, SOURCE, 100, 200
        )


def test_unknown_report_preserves_observed_busy_facts(monkeypatch):
    report = _report(status="unknown", unknown_count=1, workers=[_worker()], actors=[])
    monkeypatch.setattr(subject, "_request_process_report", lambda _source: report)

    worker, actors, any_worker = subject._privileged_process_facts(SOURCE, ADDRESS, 41)

    assert (worker, actors, any_worker) == (True, None, True)


class _Path:
    def __init__(self, infos, text=""):
        self._infos = iter(infos if isinstance(infos, list) else [infos])
        self._last = None
        self._text = text

    def lstat(self):
        try:
            self._last = next(self._infos)
        except StopIteration:
            pass
        return self._last

    def read_text(self):
        return self._text


class _Connection:
    def __init__(self, chunks, peer_uid=0):
        self.chunks = list(chunks)
        self.peer_uid = peer_uid
        self.sent = b""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def settimeout(self, _value):
        return None

    def connect(self, _path):
        return None

    def getsockopt(self, *_args):
        return struct.pack("3i", 99, self.peer_uid, 88)

    def sendall(self, value):
        self.sent += value

    def shutdown(self, _how):
        return None

    def recv(self, _size):
        return self.chunks.pop(0) if self.chunks else b""


def _mode(kind, permissions):
    return kind | permissions


def _expired_clock():
    values = iter([0.0, 11.0])
    return lambda: next(values)


def _wire_request(monkeypatch, report, *, peer_uid=0, chunks=None, drift=False, monotonic=None):
    root = SimpleNamespace(st_mode=_mode(stat.S_IFDIR, 0o755), st_uid=0)
    helper = SimpleNamespace(st_mode=_mode(stat.S_IFDIR, 0o711), st_uid=0)
    socket_infos = [
        SimpleNamespace(st_mode=_mode(stat.S_IFSOCK, 0o600), st_uid=UID, st_dev=1, st_ino=2),
        SimpleNamespace(st_mode=_mode(stat.S_IFSOCK, 0o600), st_uid=UID, st_dev=1, st_ino=3 if drift else 2),
    ]
    paths = {
        "/run": _Path(root),
        "/run/dgxm-process-inspection": _Path(helper),
        f"/run/dgxm-process-inspection/{UID}.sock": _Path(socket_infos),
        "/proc/sys/kernel/random/boot_id": _Path(root, BOOT_ID + "\n"),
    }
    monkeypatch.setattr(subject.os, "geteuid", lambda: UID)
    monkeypatch.setattr(subject.pathlib, "Path", lambda value: paths[str(value)])
    connection = _Connection(chunks if chunks is not None else [json.dumps(report).encode()], peer_uid)
    monkeypatch.setattr(subject.socket, "socket", lambda *_args: connection)
    ticks = iter([100, 200])
    monkeypatch.setattr(subject.time, "monotonic_ns", lambda: next(ticks))
    monkeypatch.setattr(subject.time, "monotonic", monotonic or (lambda: 0.0))
    return connection


def test_request_authenticates_root_peer_and_returns_valid_report(monkeypatch):
    connection = _wire_request(monkeypatch, _report())

    report = subject._request_process_report(SOURCE)

    assert report["uid"] == UID
    assert connection.sent == b"dgxm-process-inspection-v1\n"


def test_request_rejects_non_root_peer(monkeypatch):
    _wire_request(monkeypatch, _report(), peer_uid=UID)

    with pytest.raises(ValueError, match="not root"):
        subject._request_process_report(SOURCE)


@pytest.mark.parametrize(
    ("chunks", "drift", "monotonic", "error"),
    [
        ([b"{"], False, None, ValueError),
        ([b"x" * ((1 << 20) + 1)], False, None, ValueError),
        (None, True, None, ValueError),
        (None, False, _expired_clock, TimeoutError),
    ],
)
def test_request_rejects_truncated_oversize_stale_or_drifted_transport(
    monkeypatch, chunks, drift, monotonic, error
):
    clock = monotonic() if monotonic is not None else None
    _wire_request(monkeypatch, _report(), chunks=chunks, drift=drift, monotonic=clock)

    with pytest.raises(error):
        subject._request_process_report(SOURCE)


def test_transport_failures_are_unknown_without_erasing_observed_busy_positive(monkeypatch):
    report = _report(status="unknown", unknown_count=1, workers=[_worker()])
    monkeypatch.setattr(subject, "_request_process_report", lambda _source: report)
    assert subject._privileged_process_facts(SOURCE, ADDRESS, 41)[0] is True

    monkeypatch.setattr(subject, "_request_process_report", lambda _source: (_ for _ in ()).throw(TimeoutError()))
    assert subject._privileged_process_facts(SOURCE, ADDRESS, 41) == (None, None, None)
