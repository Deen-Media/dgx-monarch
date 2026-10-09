"""Dirty worker setup state crosses telemetry as a boolean verdict only."""
from __future__ import annotations

from types import SimpleNamespace

from dgx_monarch.actor import worker_status


class _Store:
    def snapshot(self):
        return {}


def _worker(*, cleanup_failed: bool):
    return SimpleNamespace(
        rank=0,
        world=1,
        topology={},
        store=_Store(),
        _attn=None,
        _setup_key=None,
        _setup_cleanup_failed=cleanup_failed,
    )


def test_worker_status_publishes_only_the_dirty_setup_verdict(monkeypatch):
    monkeypatch.setattr(worker_status, "source_manifest_sha256", lambda: "a" * 64)

    clean = worker_status.status_impl(_worker(cleanup_failed=False))
    dirty = worker_status.status_impl(_worker(cleanup_failed=True))

    assert clean["setup_cleanup_failed"] is False
    assert dirty["setup_cleanup_failed"] is True
    assert "cleanup_error" not in dirty
    assert "setup_error" not in dirty
