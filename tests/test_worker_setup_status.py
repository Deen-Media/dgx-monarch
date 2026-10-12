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


def test_worker_status_exposes_only_available_return_ownership(monkeypatch):
    from dgx_monarch.rdma_ownership import HandoffRegistry
    from dgx_monarch.return_diagnostics import ReturnCounts

    monkeypatch.setattr(worker_status, "source_manifest_sha256", lambda: "a" * 64)
    worker = _worker(cleanup_failed=False)
    fake_status = worker_status.status_impl(worker)["return_ownership"]
    assert "live_handoffs" not in fake_status
    worker._latent_handoffs = HandoffRegistry()
    worker._return_counts = ReturnCounts()
    handoff = {}
    worker._latent_handoffs.publish(handoff, 1, 1)
    status = worker_status.status_impl(worker)["return_ownership"]
    assert status["live_handoffs"] == 1
    assert status["tombstones"] == 0
    assert status["return_counts"] == []
    assert worker._latent_handoffs.owns(handoff)
