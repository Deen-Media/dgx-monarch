"""Cancellation must not release a live local management mutator."""

import asyncio
from types import SimpleNamespace

import pytest

from dgx_monarch.studio.operations import Operations
from dgx_monarch.studio.store import Store


@pytest.mark.parametrize("action,pid_checkpoint_fails", [
    ("driver.stop", False), ("reap.apply", False), ("reap.apply", True),
])
def test_cancel_keeps_lock_until_owner_settlement(tmp_path, monkeypatch, action, pid_checkpoint_fails):
    async def scenario():
        store = Store(tmp_path / "state")
        service = "comfy.service" if action == "driver.stop" else None
        operations = Operations(store, object(), allow_management=True, driver_service=service)
        started, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def perform(*args):
            started.set()
            await release.wait()
            settled.set()
            return "succeeded", {"verification": "verified", "source": "owner_service"}

        operations.driver_binding = SimpleNamespace(perform=perform)

        class Output:
            async def read(self, size):
                started.set()
                await release.wait()
                return b""

        class Process:
            pid = 1234
            stdout = Output()

            async def wait(self):
                assert release.is_set()
                settled.set()
                return 0

        async def spawn(*args, **kwargs):
            return Process()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        update = store.update

        def checkpoint(kind, identity, changes):
            if pid_checkpoint_fails and "process_id" in changes:
                raise OSError("synthetic checkpoint failure")
            if "receipt" in changes:
                assert settled.is_set(), "receipt published before owner settlement"
                assert operations.execution_lock.locked(), "receipt published after lock release"
            return update(kind, identity, changes)

        monkeypatch.setattr(store, "update", checkpoint)
        operation = store.create("operations", {
            "id": "owned", "action": action, "state": "applying", "driver_service": service,
            "config_fingerprint": None, "driver_identity": {"state": "active"}, "parameters": {"grace_s": 0},
        })
        task = asyncio.create_task(operations.execute(operation))
        try:
            await asyncio.wait_for(started.wait(), 2)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                assert not task.done() and not settled.is_set()
                assert operations.execution_lock.locked()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert settled.is_set() and not operations.execution_lock.locked()
        result = store.get("operations", "owned")
        assert result["state"] == "unknown"
        assert result["receipt"]["error"] == "operation_interrupted"
        if not pid_checkpoint_fails:
            assert result["receipt"]["source"] == ("owner_service" if service else "existing_cli")
        store.close()

    asyncio.run(scenario())
