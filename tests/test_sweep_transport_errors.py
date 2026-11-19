"""A driver answer that urllib cannot read is a transport error at every sweep call site.

urllib raises http.client's IncompleteRead (a body shorter than its
Content-Length) and BadStatusLine (a malformed status line) as they are: neither
is an OSError or a ValueError. Each sweep call that talks to the driver must
still answer, or end the session, the way its docstring says, instead of
ending the runner with a traceback and no run summary. A local socket plays
the driver and writes the bytes of each broken answer.
"""
from __future__ import annotations

import socketserver
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from benchmark.sweep import driver

SHORT = (b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\n"
         b"Connection: close\r\n\r\n{\"prompt_id\": ")
GARBLED = b"NOT AN HTTP ANSWER\r\n\r\n"
BROKEN = {"IncompleteRead": SHORT, "BadStatusLine": GARBLED}


@contextmanager
def answering(reply: bytes) -> Iterator[str]:
    """A driver at the returned URL that answers every request with ``reply`` and hangs up."""
    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            # Read the whole request first, so the answer never races the request's body.
            length = 0
            while (line := self.rfile.readline()) not in (b"\r\n", b""):
                name, _, value = line.decode("latin-1").partition(":")
                if name.strip().lower() == "content-length":
                    length = int(value)
            self.rfile.read(length)
            self.wfile.write(reply)

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("broken", sorted(BROKEN))
def test_each_call_answers_an_unreadable_driver_answer_as_a_transport_error(broken):
    with answering(BROKEN[broken]) as url:
        # The cache reset answers with the error, so run.cache_wedge ends the session.
        answer = driver.free_cache(url)
        assert broken in answer and answer not in ("idle", "still busy")
        assert broken in driver.drain(url, "prompt")
        assert broken in driver.recycle(url)
        assert broken in driver.grant_pending_consent(url, {"waiver_guards": ()})["detail"]
        assert broken in driver.dismiss_pending_consent(url)["detail"]
        # A lost submit wedges the session, as the docstring says.
        with pytest.raises(driver.SessionWedged, match=broken):
            driver.queue_and_wait(url, {}, "client", 30)


@pytest.mark.parametrize("broken", sorted(BROKEN))
def test_a_history_poll_that_cannot_be_read_is_a_poll_transport_state(broken, monkeypatch):
    with answering(BROKEN[broken]) as url:
        monkeypatch.setattr(driver, "post", lambda *_args, **_kwargs: {"prompt_id": "prompt"})
        _wall, entry, state, prompt_id = driver.queue_and_wait(url, {}, "client", 30)
        assert entry is None and prompt_id == "prompt"
        assert state.startswith("poll transport ") and broken in state


@pytest.mark.parametrize("broken", sorted(BROKEN))
def test_the_converter_falls_back_to_the_repository_widgets_when_the_driver_cannot_be_read(broken):
    from benchmark.sweep import convert

    with answering(BROKEN[broken]) as url:
        names, source = convert.widget_source(url)
    assert source == "repo"
    assert "mode" in names("DGXMonarchInit")
