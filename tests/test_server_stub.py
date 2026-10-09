"""Headless PromptServer stub and the worker's preload failure hint.

Custom node packs decorate routes on server.PromptServer.instance at import
time; workers have no server, so ensure_prompt_server_stub gives them an
inert one.
"""
import sys
import types

import pytest

from dgx_monarch.actor.comfy_bridge import _preload_failure_hint
from dgx_monarch.actor.server_stub import ensure_prompt_server_stub


@pytest.fixture
def fake_server(monkeypatch):
    server_mod = types.ModuleType("server")

    class PromptServer:
        pass

    server_mod.PromptServer = PromptServer
    monkeypatch.setitem(sys.modules, "server", server_mod)
    return server_mod


def test_installs_inert_instance(fake_server):
    pytest.importorskip("aiohttp")
    assert ensure_prompt_server_stub() is True
    instance = fake_server.PromptServer.instance
    assert instance is not None

    # The import-time surface packs touch:
    @instance.routes.post("/reslyf/settings")
    async def handler(request):
        return None

    assert not instance.app.router.frozen  # Packs may probe this before add_routes.
    instance.send_sync("event", {"x": 1})
    instance.send_progress_text("t", "node")
    assert "custom_nodes_from_web" not in instance.supports  # RES4LYF tests membership in this list.
    assert instance.client_id is None


def test_idempotent_and_respects_existing_instance(fake_server):
    pytest.importorskip("aiohttp")
    sentinel = object()
    fake_server.PromptServer.instance = sentinel
    assert ensure_prompt_server_stub() is True
    assert fake_server.PromptServer.instance is sentinel  # a real server is never replaced


def test_fail_open_when_server_module_missing(monkeypatch):
    # sys.modules[name] = None makes `import name` raise ImportError: the
    # worker env without a comfy checkout on sys.path.
    monkeypatch.setitem(sys.modules, "server", None)
    assert ensure_prompt_server_stub() is False


def test_preload_failure_hint_reports_direct_import_error():
    hint = _preload_failure_hint(ImportError("worker-only dependency missing"))

    assert "python dependency is missing" in hint
    assert sys.executable in hint
    assert "PromptServer surface" not in hint


@pytest.mark.parametrize("link", ["cause", "context"])
def test_preload_failure_hint_prefers_chained_import_error(link: str):
    dependency = ModuleNotFoundError("No module named 'worker_only_dependency'")
    stub_gap = AttributeError("PromptServer.instance is unavailable")
    if link == "cause":
        stub_gap.__cause__ = dependency
    else:
        stub_gap.__context__ = dependency

    hint = _preload_failure_hint(stub_gap)

    assert "python dependency is missing" in hint
    assert sys.executable in hint
    assert "PromptServer surface" not in hint


def test_preload_failure_hint_does_not_skip_falsey_import_cause():
    class FalseyImportError(ImportError):
        def __bool__(self):
            return False

    wrapper = RuntimeError("pack preload failed")
    wrapper.__cause__ = FalseyImportError("worker dependency missing")
    wrapper.__context__ = AttributeError("PromptServer.instance is unavailable")

    hint = _preload_failure_hint(wrapper)

    assert "python dependency is missing" in hint
    assert "PromptServer surface" not in hint


def test_preload_failure_hint_does_not_test_stub_cause_truthiness():
    class ExplosiveStubGap(AttributeError):
        def __bool__(self):
            raise AssertionError("exception truthiness was evaluated")

    wrapper = RuntimeError("pack preload failed")
    wrapper.__cause__ = ExplosiveStubGap("PromptServer.instance is unavailable")

    hint = _preload_failure_hint(wrapper)

    assert "PromptServer surface" in hint
    assert "_HeadlessPromptServer" in hint


def test_preload_failure_hint_reports_standalone_prompt_server_gap():
    hint = _preload_failure_hint(
        AttributeError("PromptServer.instance is unavailable")
    )

    assert "PromptServer surface" in hint
    assert "_HeadlessPromptServer" in hint


def test_preload_failure_hint_falls_back_to_skip_list():
    assert _preload_failure_hint(RuntimeError("pack initialization failed")) == (
        "add it to DGXM_SKIP_NODE_PACKS so the worker skips it"
    )
