"""A Sol install review reuses the host staging it reserved, and releases what it abandoned.

Each review without a plan token stages a pip resolution and wheel build on
every host. An identical review while the first is live reads that staging back
instead of building again. Staging that only expired or replaced Studio reviews
used is removed on each host at the next review; staging an apply used, or
that the owner supplied by token, stays.
"""
from __future__ import annotations

import os
import subprocess
import sys
from contextlib import closing
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import lifecycle, sol_staging
from dgx_monarch.config import ClusterConfig, HostConfig
from dgx_monarch.studio import management, management_sol
from dgx_monarch.studio.contracts import timestamp
from dgx_monarch.studio.management_fields import parse
from dgx_monarch.studio.operations import Operations
from dgx_monarch.studio.store import Store

CLUSTER = ClusterConfig(hosts=(HostConfig("alpha", "tcp://192.0.2.1:26600"), HostConfig("beta", "tcp://192.0.2.2:26600")),
                        python_bin=sys.executable)
PLAN = {"digest": "d" * 64, "allowed": [], "changes": []}


class Hosts:
    """Records every staging call; a token listed in `broken` no longer reads back."""

    def __init__(self):
        self.calls, self.discards, self.broken, self.unreachable = [], [], set(), False

    def host(self, config, host, action, token, allowed, expected=""):
        self.calls.append((host.name, action, token))
        if action == "read" and token in self.broken:
            return {"status": "failed", "reason": "plan changed"}
        return {"status": "ok", "plan": {**PLAN, "allowed": list(allowed)}}

    def discard(self, config, host, tokens):
        self.discards.append((host.name, sorted(tokens)))
        return None if self.unreachable and host.name == "beta" else dict.fromkeys(tokens, "removed")

    def actions(self):
        return [(action, token) for name, action, token in self.calls if name == "alpha"]


@pytest.fixture
def hosts(monkeypatch):
    value = Hosts()
    monkeypatch.setattr(management_sol.sol_install, "_host", value.host)
    monkeypatch.setattr(management_sol.sol_staging, "discard", value.discard)
    monkeypatch.setattr(management, "load_cluster_config", lambda path: CLUSTER)
    return value


@pytest.fixture
def operations(tmp_path):
    config = tmp_path / "cluster.toml"
    config.write_text("# reviewed\n")
    with closing(Store(tmp_path / "state")) as store:
        yield Operations(store, SimpleNamespace(origin="http://127.0.0.1:8188"), config=config, allow_management=True)


def review(ops, **values):
    plan = ops.preview({"action": "sol-install", "parameters": {"idle_maintenance": True, **values}})
    assert plan["available"], plan["reason"]
    return plan


def expire(ops, plan):
    ops.store.update("operations", plan["id"], {"expires_at": timestamp() - 1})


def test_identical_live_review_reads_back_the_same_staging(operations, hosts):
    first = review(operations)
    second = review(operations)
    token = first["adapter_plan"]["token"]
    assert second["adapter_plan"]["token"] == token
    assert second["adapter_plan"]["staging_reused"] is True and first["adapter_plan"]["staging_reused"] is False
    assert second["adapter_plan"]["fleet_digest"] == first["adapter_plan"]["fleet_digest"]
    assert hosts.actions() == [("plan", token), ("read", token)]
    assert hosts.discards == []


def test_a_different_plan_stages_its_own_token(operations, hosts):
    first = review(operations)
    second = review(operations, allow_change=["cuda-python"])
    assert second["adapter_plan"]["token"] != first["adapter_plan"]["token"]
    assert [action for action, _ in hosts.actions()] == ["plan", "plan"]


def test_expired_review_staging_is_removed_on_every_host_before_staging_again(operations, hosts):
    first = review(operations)
    old = first["adapter_plan"]["token"]
    expire(operations, first)
    second = review(operations)
    new = second["adapter_plan"]["token"]
    assert new != old and hosts.actions()[-1] == ("plan", new)
    assert hosts.discards == [("alpha", [old]), ("beta", [old])]
    assert second["adapter_plan"]["staging_cleanup"] == [
        {"token": old, "state": "removed", "hosts": [{"ordinal": 0, "outcome": "removed"}, {"ordinal": 1, "outcome": "removed"}]}]
    assert operations.store.get("operations", first["id"])["staging_discard"]["state"] == "removed"
    review(operations)  # The removal is recorded, so it is not repeated.
    assert len(hosts.discards) == 2


def test_replaced_staging_is_removed_even_before_its_review_expires(operations, hosts):
    first = review(operations)
    old = first["adapter_plan"]["token"]
    hosts.broken.add(old)
    second = review(operations)  # The old staging no longer reads back, so a fresh plan replaces it.
    assert second["adapter_plan"]["token"] != old and second["adapter_plan"]["staging_reused"] is False
    assert hosts.discards == []
    review(operations)
    assert hosts.discards == [("alpha", [old]), ("beta", [old])]


def test_an_unknown_removal_is_retried_at_the_next_review(operations, hosts):
    first = review(operations)
    expire(operations, first)
    hosts.unreachable = True
    second = review(operations)
    assert second["adapter_plan"]["staging_cleanup"][0]["state"] == "unknown"
    assert "staging_discard" not in operations.store.get("operations", first["id"])
    hosts.unreachable = False
    review(operations)
    assert operations.store.get("operations", first["id"])["staging_discard"]["state"] == "removed"


def test_applied_or_operator_staging_is_never_removed(operations, hosts, monkeypatch):
    applied = review(operations)

    async def execute(operation):
        pass
    monkeypatch.setattr(operations, "execute", execute)
    import asyncio

    async def consume():
        operations.apply(applied["id"], {"digest": applied["digest"], "confirmed": True, "idempotency_key": "sol-apply"})
        await asyncio.gather(*operations.tasks)
    asyncio.run(consume())
    supplied = review(operations, plan="a" * 32)
    assert supplied["adapter_plan"]["staging_owner"] == "operator"
    expire(operations, applied)
    expire(operations, supplied)
    review(operations)
    assert hosts.discards == []


def test_incomplete_studio_staging_is_removed_at_the_next_review(operations, hosts, monkeypatch):
    monkeypatch.setattr(management_sol.sol_install, "_host", lambda *args: None)
    blocked = operations.preview({"action": "sol-install", "parameters": {"idle_maintenance": True}})
    assert blocked["available"] is False and "removes this incomplete staging" in blocked["adapter_plan"]["recovery"]
    monkeypatch.setattr(management_sol.sol_install, "_host", hosts.host)
    review(operations)
    assert hosts.discards == [("alpha", [blocked["adapter_plan"]["token"]]), ("beta", [blocked["adapter_plan"]["token"]])]


def test_discard_script_refuses_anything_but_plan_tokens():
    for tokens in ([], ["../escape"], ["A" * 32], ["a" * 32] * 1 + [f"{index:032x}" for index in range(16)]):
        with pytest.raises(ValueError):
            sol_staging.discard_script(CLUSTER, tokens)
    script = sol_staging.discard_script(CLUSTER, ["b" * 32, "a" * 32])
    assert "-I -B -" in script and "acquire_lifecycle_lock(Path.home(), True)" in script
    assert repr(["a" * 32, "b" * 32]) in script


def staged(home, token, *, readback=False, mode=0o700):
    root = home / ".local/state/dgx-monarch/sol-install" / token
    (root / "wheels").mkdir(parents=True)
    (root / "wheels" / "package.whl").write_bytes(b"wheel")
    if readback:
        (root / "readback.json").write_text("{}")
    root.chmod(mode)
    return root


def test_host_script_removes_only_plain_private_studio_staging(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    removed = staged(home, "a" * 32)
    kept = staged(home, "b" * 32, readback=True)
    shared = staged(home, "c" * 32, mode=0o770)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious").write_text("keep")
    (home / ".local/state/dgx-monarch/sol-install" / ("d" * 32)).symlink_to(outside)
    for part in (home / ".local", home / ".local/state", home / ".local/state/dgx-monarch",
                 home / ".local/state/dgx-monarch/sol-install"):
        part.chmod(0o700)

    def run_on_host(config, host, script, timeout=60):
        environment = {**os.environ, "HOME": str(home)}
        return subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, timeout=timeout, env=environment)

    monkeypatch.setattr(lifecycle, "run_on_host", run_on_host)
    tokens = ["a" * 32, "b" * 32, "c" * 32, "d" * 32, "e" * 32]
    outcome = sol_staging.discard(CLUSTER, CLUSTER.hosts[0], tokens)
    assert outcome == {"a" * 32: "removed", "b" * 32: "kept", "c" * 32: "unsafe", "d" * 32: "unsafe", "e" * 32: "absent"}
    assert not removed.exists() and kept.exists() and shared.exists()
    assert (outside / "precious").read_text() == "keep"


@pytest.mark.parametrize("stdout,code", [("", 0), ("DGXM_SOL_STAGING=not json\n", 0),
                                         ('DGXM_SOL_STAGING={"status": "failed"}\n', 1),
                                         ('DGXM_SOL_STAGING={"status": "ok", "outcome": {}}\n', 0),
                                         ('DGXM_SOL_STAGING={"status": "ok", "outcome": {"%s": "gone"}}\n' % ("a" * 32), 0)])
def test_unreadable_host_answers_are_unknown(monkeypatch, stdout, code):
    monkeypatch.setattr(lifecycle, "run_on_host",
                        lambda *args, **kwargs: subprocess.CompletedProcess([], code, stdout=stdout, stderr=""))
    assert sol_staging.discard(CLUSTER, CLUSTER.hosts[0], ["a" * 32]) is None


def test_transport_loss_is_unknown(monkeypatch):
    def lost(*args, **kwargs):
        raise subprocess.TimeoutExpired("ssh", 120)
    monkeypatch.setattr(lifecycle, "run_on_host", lost)
    assert sol_staging.discard(CLUSTER, CLUSTER.hosts[0], ["a" * 32]) is None


def test_direct_preview_keeps_its_existing_contract(hosts):
    plan = management_sol.preview("sol-install", CLUSTER, parse("sol-install", {"idle_maintenance": True}))
    assert plan["ready"] is True and plan["staging_owner"] == "studio" and "staging_cleanup" not in plan
