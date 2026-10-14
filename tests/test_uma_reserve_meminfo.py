"""check_uma_reserve reads /proc/meminfo inside a with block.

The read used to sit in a bare for statement, which left the file to the
garbage collector. Each case here checks that the file is closed when the
function returns, on the early break, on a full read and on a parse error.
"""
from __future__ import annotations

import io
import types

import pytest

from dgx_monarch import telemetry
from dgx_monarch.actor import worker_status


class _Meminfo(io.StringIO):
    opened: list[_Meminfo] = []

    def __init__(self, text: str) -> None:
        super().__init__(text)
        _Meminfo.opened.append(self)


class _Log:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, fmt, *args):
        self.warnings.append(str(fmt) % args)


@pytest.fixture
def rig(monkeypatch):
    _Meminfo.opened = []
    emitted: list[tuple[str, dict]] = []
    recorder = _Log()
    monkeypatch.setattr(worker_status, "log", recorder)
    monkeypatch.setattr(telemetry, "emit",
                        lambda kind, **fields: emitted.append((kind, fields)))

    def install(text: str) -> None:
        def fake_open(path, *args, **kwargs):
            assert path == "/proc/meminfo"
            return _Meminfo(text)

        monkeypatch.setattr(worker_status, "open", fake_open, raising=False)

    return types.SimpleNamespace(install=install, emitted=emitted, log=recorder)


def _kib(gib: float) -> int:
    return int(gib * 2**20)


def test_a_breach_warns_emits_and_closes_the_file(rig):
    rig.install(f"MemTotal: {_kib(120)} kB\nMemAvailable: {_kib(8)} kB\n"
                "Buffers: 1 kB\n")

    worker_status.check_uma_reserve(types.SimpleNamespace(uma_reserve_gb=16.0))

    assert len(_Meminfo.opened) == 1 and _Meminfo.opened[0].closed
    assert len(rig.log.warnings) == 1
    assert rig.emitted == [("uma_reserve_breach",
                            {"available_gib": 8.0, "reserve_gib": 16.0})]


def test_headroom_stays_quiet_and_closes_the_file(rig):
    rig.install(f"MemAvailable: {_kib(64)} kB\n")

    worker_status.check_uma_reserve(types.SimpleNamespace(uma_reserve_gb=16.0))

    assert len(_Meminfo.opened) == 1 and _Meminfo.opened[0].closed
    assert rig.log.warnings == [] and rig.emitted == []


def test_a_missing_or_garbled_row_closes_the_file(rig):
    rig.install("MemTotal: 1 kB\n")
    worker_status.check_uma_reserve(types.SimpleNamespace(uma_reserve_gb=16.0))
    rig.install("MemAvailable: lots kB\n")
    worker_status.check_uma_reserve(types.SimpleNamespace(uma_reserve_gb=16.0))

    assert len(_Meminfo.opened) == 2
    assert all(handle.closed for handle in _Meminfo.opened)
    assert rig.log.warnings == [] and rig.emitted == []


def test_no_reserve_reads_nothing(rig):
    rig.install("MemAvailable: 1 kB\n")

    worker_status.check_uma_reserve(types.SimpleNamespace(uma_reserve_gb=0.0))

    assert _Meminfo.opened == []
