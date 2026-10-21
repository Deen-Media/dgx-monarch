"""An absent Studio config cannot discover a different remote target."""

import asyncio
from types import SimpleNamespace

import pytest

from dgx_monarch import config
from dgx_monarch.cli import actor_sweep, doctor
from dgx_monarch.studio import local_commands
from dgx_monarch.studio.operations import Operations
from dgx_monarch.studio.store import Store


@pytest.mark.parametrize("action", ["doctor", "reap.preview", "reap.apply"])
def test_local_adapters_never_discover_env_or_working_directory_config(tmp_path, monkeypatch, action):
    unrelated = tmp_path / "cluster.toml"
    unrelated.write_text("An unrelated remote cluster must never be read")
    monkeypatch.setenv("DGXM_CLUSTER_TOML", str(unrelated))
    monkeypatch.chdir(tmp_path)

    def discovery_trap(*args, **kwargs):
        pytest.fail("Studio local operation attempted implicit config discovery")

    monkeypatch.setattr(config, "find_config_path", discovery_trap)
    calls = []

    def local_doctor(selected, *, as_json):
        calls.append((selected, as_json))
        return 0

    def local_reap(selected, *, dry_run, grace_s):
        calls.append((selected, dry_run, grace_s))
        return True

    monkeypatch.setattr(doctor, "doctor_exit", local_doctor)
    monkeypatch.setattr(actor_sweep, "reap_command", local_reap)
    assert local_commands.main([action, "--grace", "2.5"]) == 0
    assert calls == ([(None, True)] if action == "doctor" else [(None, action == "reap.preview", 2.5)])


@pytest.mark.parametrize("action", ["doctor", "reap.preview", "reap.apply"])
@pytest.mark.parametrize("selected_config", [False, True])
def test_dispatch_uses_local_adapter_only_without_explicit_config(tmp_path, monkeypatch, action, selected_config):
    async def scenario():
        store = Store(tmp_path / "state")
        config_path = tmp_path / "selected.toml" if selected_config else None
        operations = Operations(store, object(), config=config_path)
        store.create("operations", {"id": "operation"})
        calls = []

        async def wait():
            return 0

        async def spawn(argv0, *args, **kwargs):
            calls.append(list(args))
            return SimpleNamespace(pid=123, stdout=None, wait=wait)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        await operations.execute_owned({"id": "operation", "action": action, "parameters": {"grace_s": 2.5}})
        module = "dgx_monarch" if selected_config else "dgx_monarch.studio.local_commands"
        assert calls[0][:5] == ["-P", "-s", "-B", "-m", module]
        if selected_config:
            assert calls[0][5:7] == ["--config", str(config_path)]
        else:
            assert calls[0][5] == action and "--config" not in calls[0]
        store.close()

    asyncio.run(scenario())
