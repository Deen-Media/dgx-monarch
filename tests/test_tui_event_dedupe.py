"""Chronological bounded event dedupe for the repeated telemetry tail."""
from dgx_monarch.tui.data import RingStore


def test_large_batch_keeps_its_latest_unchanged_tail_deduped():
    ring = RingStore()
    events = [
        {"t": float(index), "kind": "notice", "seq": index}
        for index in range(4097)
    ]

    assert ring.dedupe_events(events) == events
    assert len(ring._seen_events) == 4096

    # Eviction drops the least recently seen key, so a resent tail of 2049
    # events, more than half the 4096 limit, stays deduped. Trimming the seen set
    # to an arbitrary half would let at least one of them reappear on the next poll.
    assert ring.dedupe_events(events[-2049:]) == []
    assert len(ring._seen_events) == 4096


def test_repeated_tail_refreshes_recency_before_new_event_eviction():
    ring = RingStore()
    events = [
        {"t": float(index), "kind": "notice", "seq": index}
        for index in range(4096)
    ]
    ring.dedupe_events(events)

    assert ring.dedupe_events([events[0]]) == []
    ring.dedupe_events([{"t": 4096.0, "kind": "notice", "seq": 4096}])
    assert ring.dedupe_events([events[0]]) == []


def test_malformed_event_times_and_kinds_are_still_stably_deduped():
    ring = RingStore()
    events = [
        {"t": "never", "kind": ["not", "hashable"], "note": "one"},
        {"t": {"nested": True}, "kind": "notice", "note": "two"},
        {"t": float("nan"), "kind": "notice", "note": "three"},
    ]

    assert ring.dedupe_events(events) == events
    assert ring.dedupe_events(events) == []
