"""The progress receiver's join guard reads a private CPython thread field.

The read goes through a cast, so mypy no longer checks it. These tests pin the
field on each Python that CI runs.
"""
from __future__ import annotations

import threading

from dgx_monarch.progress import _thread_started


def test_a_thread_that_never_started_is_not_joined():
    thread = threading.Thread(target=lambda: None)
    assert not _thread_started(thread)


def test_a_started_thread_is_joined():
    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join()
    assert _thread_started(thread)


def test_the_private_started_event_alone_still_counts():
    # Set the event without an ident, so only the second check can pass.
    thread = threading.Thread(target=lambda: None)
    thread._started.set()
    assert thread.ident is None
    assert _thread_started(thread)
