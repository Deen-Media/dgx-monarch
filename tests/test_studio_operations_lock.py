"""The operations lock between the companion and dgxm studio up, down and token --rotate.

The companion holds ``operations.lock`` shared from a review's admission until
the operation settles; the commands take it exclusive before their busy check
and keep it until their restart or stop ends. These tests use real files and
real flock calls; the unit's control group is replaced, so no test reads the
machine's cgroups, and no systemd unit is touched. Each Store is built
without the companion's launch, so the lock file starts missing, as in a state
directory a companion of an earlier release used.
"""
from __future__ import annotations

import asyncio
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from dgx_monarch.studio import operations_lock, unit_state
from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.operations import Operations
from dgx_monarch.studio.store import Store

STATUS = unit_state.UnitStatus(load_state="loaded", active_state="active", sub_state="running", main_pid=4242,
                               control_group="/user.slice/app.slice/dgxm-studio.service",
                               fragment_path="/home/user/.config/systemd/user/dgxm-studio.service",
                               start_monotonic_us=1_000_000)


@pytest.fixture
def cgroup(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """The processes the busy check reads besides the main one; [None] stands for an unreadable group."""
    found: list[Any] = []
    monkeypatch.setattr(unit_state, "busy_processes", lambda status: None if found == [None] else list(found))
    return found


def _refusal(state: Path) -> str | None:
    """What up, down or --rotate would print when it tries to stop Studio now; None when it may."""
    try:
        with operations_lock.quiesced(state, STATUS):
            return None
    except SystemExit as exc:
        return str(exc.code)


def _review(operations: Operations) -> tuple[dict[str, Any], dict[str, Any]]:
    review = operations.preview({"action": "driver.free"})
    return review, {"idempotency_key": "first", "digest": review["digest"], "confirmed": True}


def test_an_operation_applied_after_the_busy_check_is_refused_and_keeps_its_review(tmp_path: Path, cgroup: list[Any]):
    """The race: the command checked, then a tab applies before the restart. The companion refuses it."""
    async def scenario() -> None:
        store = Store(tmp_path / "state")
        operations = Operations(store, object(), allow_management=True)
        review, request = _review(operations)
        ran: list[str] = []

        async def execute(operation: dict[str, Any]) -> None:
            ran.append(operation["id"])

        operations.execute = execute  # type: ignore[method-assign]
        with operations_lock.quiesced(store.directory, STATUS):
            with pytest.raises(StudioError) as refused:
                operations.apply(review["id"], request)
            assert (refused.value.code, refused.value.status, str(refused.value)) == (
                "operation_unavailable", 409, operations_lock.DRAINING)
            with pytest.raises(StudioError, match="stopping or restarting"):
                await operations.prepare({"action": "driver.free"})
        # Nothing was reserved, so the same review applies once the command has ended.
        assert store.get("operations", review["id"]).get("consumed_by") is None
        assert store.request("operations", "first", {**request, "operation_id": review["id"]}) is None
        accepted = operations.apply(review["id"], request)
        await asyncio.gather(*operations.tasks)
        assert ran == [accepted["id"]]
        store.close()

    asyncio.run(scenario())


def test_an_admitted_operation_with_no_child_yet_refuses_the_command_until_it_settles(tmp_path: Path,
                                                                                     cgroup: list[Any]):
    """The control group shows nothing while an update stops the driver; the shared hold still counts."""
    async def scenario() -> None:
        store = Store(tmp_path / "state")
        operations = Operations(store, object(), allow_management=True)
        review, request = _review(operations)
        release = asyncio.Event()

        async def execute(operation: dict[str, Any]) -> None:
            await release.wait()

        operations.execute = execute  # type: ignore[method-assign]
        operations.apply(review["id"], request)
        await asyncio.sleep(0)
        assert cgroup == [] and _refusal(store.directory) == unit_state.BUSY
        cgroup.append("pid 4250: dgxm update")
        assert _refusal(store.directory) == f"{unit_state.BUSY}\npid 4250: dgxm update"
        cgroup.clear()
        release.set()
        await asyncio.gather(*operations.tasks)
        assert _refusal(store.directory) is None
        store.close()

    asyncio.run(scenario())


def test_a_review_in_preparation_refuses_the_command_until_it_is_published(tmp_path: Path, cgroup: list[Any]):
    async def scenario() -> None:
        store = Store(tmp_path / "state")
        operations = Operations(store, object(), allow_management=True)
        release = asyncio.Event()

        async def prepare(request: dict[str, Any]) -> dict[str, Any]:
            await release.wait()
            return {}

        operations._prepare = prepare  # type: ignore[method-assign]
        preparing = asyncio.create_task(operations.prepare({"action": "driver.free"}))
        await asyncio.sleep(0)
        assert _refusal(store.directory) == unit_state.BUSY
        release.set()
        await preparing
        assert _refusal(store.directory) is None
        store.close()

    asyncio.run(scenario())


def test_an_apply_that_is_refused_or_replayed_returns_its_hold(tmp_path: Path, cgroup: list[Any]):
    async def scenario() -> None:
        store = Store(tmp_path / "state")
        operations = Operations(store, object(), allow_management=True)
        review, request = _review(operations)

        async def execute(operation: dict[str, Any]) -> None:
            return None

        operations.execute = execute  # type: ignore[method-assign]
        accepted = operations.apply(review["id"], request)
        await asyncio.gather(*operations.tasks)
        assert operations.apply(review["id"], request)["id"] == accepted["id"]  # a replay starts nothing
        with pytest.raises(StudioError, match="already dispatched"):
            operations.apply(review["id"], {**request, "idempotency_key": "second"})
        assert not operations.tasks and _refusal(store.directory) is None
        store.close()

    asyncio.run(scenario())


def test_the_busy_check_still_reads_the_control_group(tmp_path: Path, cgroup: list[Any]):
    state = tmp_path / "state"
    Store(state).close()
    cgroup.append("pid 4251: dgxm status")
    assert _refusal(state) == f"{unit_state.BUSY}\npid 4251: dgxm status"
    cgroup[:] = [None]
    assert _refusal(state) == unit_state.BUSY_UNKNOWN
    cgroup.clear()
    assert _refusal(state) is None


def test_the_command_makes_a_missing_lock_and_never_a_missing_state_directory(tmp_path: Path, cgroup: list[Any]):
    """A companion of an earlier release never holds the lock, so the control group lines still decide."""
    state = tmp_path / "state"
    Store(state).close()
    cgroup.append("pid 4250: dgxm update")
    assert _refusal(state) == f"{unit_state.BUSY}\npid 4250: dgxm update"
    info = (state / operations_lock.OPERATIONS_LOCK).lstat()
    assert stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600
    cgroup.clear()
    assert _refusal(state) is None
    assert _refusal(tmp_path / "never-started") is None and not (tmp_path / "never-started").exists()


def test_the_command_refuses_a_lock_path_it_cannot_trust(tmp_path: Path, cgroup: list[Any]):
    state = tmp_path / "state"
    Store(state).close()
    (state / operations_lock.OPERATIONS_LOCK).symlink_to(tmp_path / "elsewhere")
    assert _refusal(state).startswith(f"Cannot use the state directory {state}: ")
    assert not (tmp_path / "elsewhere").exists()


def test_a_command_that_dies_holding_the_lock_leaves_the_companion_admitting(tmp_path: Path):
    """The kernel drops a dead process's flock, so a crashed up, down or --rotate needs no deadline."""
    state = tmp_path / "state"
    Store(state).close()
    operations_lock.release(operations_lock.hold(state))
    holder = ("import fcntl, os, sys, time\n"
              "descriptor = os.open(sys.argv[1], os.O_RDWR)\n"
              "fcntl.flock(descriptor, fcntl.LOCK_EX)\n"
              "print('held', flush=True)\n"
              "time.sleep(60)\n")
    child = subprocess.Popen([sys.executable, "-I", "-c", holder, str(state / operations_lock.OPERATIONS_LOCK)],
                             stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout is not None and child.stdout.readline().strip() == "held"
        with pytest.raises(StudioError, match="stopping or restarting"):
            operations_lock.hold(state)
    finally:
        child.kill()
        child.wait(timeout=10)
    operations_lock.release(operations_lock.hold(state))
