"""Optional attention commands require explicit configuration and authority."""
from __future__ import annotations

import sys
from types import ModuleType

from dgx_monarch.cli import main as cli


def test_install_plan_and_apply_forward_exact_reviewed_choices(monkeypatch):
    module = ModuleType("dgx_monarch.cli.sol_install")
    calls = []
    config = object()
    module.run = lambda value, **choices: calls.append((value, choices)) or 75
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cli, "_require_config", lambda args: config)
    assert cli.main(["sol-install"]) == 75
    assert calls.pop() == (config, {
        "apply": False, "plan": None, "plan_digest": None,
        "allow_change": (), "idle_maintenance": False,
    })
    assert cli.main(["sol-install", "--apply", "--plan", "token", "--plan-digest", "digest",
                     "--allow-change", "protobuf", "--idle-maintenance"]) == 75
    assert calls.pop() == (config, {
        "apply": True, "plan": "token", "plan_digest": "digest",
        "allow_change": ("protobuf",), "idle_maintenance": True,
    })


def test_smoke_is_a_separate_explicit_gpu_command(monkeypatch):
    module = ModuleType("dgx_monarch.cli.sol_smoke")
    calls = []
    config = object()
    module.cmd_sol_smoke = lambda value, **choices: calls.append((value, choices)) or 0
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cli, "_require_config", lambda args: config)
    assert cli.main(["sol-smoke"]) == 0
    assert calls.pop() == (config, {"idle_maintenance": False, "timeout_s": 300})
    assert cli.main(["sol-smoke", "--idle-maintenance", "--timeout-s", "420"]) == 0
    assert calls.pop() == (config, {"idle_maintenance": True, "timeout_s": 420})
