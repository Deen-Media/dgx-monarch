"""Package maintenance is plan bound, no-deps, explicit and fail closed."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import sol_install as command
from dgx_monarch.cli import sol_install_payload as payload
from dgx_monarch.config import HostConfig


def config():
    return SimpleNamespace(python_bin="/venv/bin/python", hosts=[
        HostConfig(name="a", address="192.0.2.1:1", ssh_user="u"),
        HostConfig(name="b", address="192.0.2.2:1", ssh_user="u"),
    ])


def report(name="protobuf", version="6.33.6"):
    return {"install": [{"metadata": {"name": name, "version": version},
                         "download_info": {"url": f"https://example.invalid/{name}.whl",
                                           "archive_info": {"hashes": {"sha256": "a" * 64}}}}]}


@pytest.mark.parametrize("name", ["torch", "torchmonarch", "triton", "xfuser", "yunchang", "nvidia-cudnn-cu13", "nvidia-nccl-cu13"])
def test_protected_package_cannot_be_overridden(name):
    with pytest.raises(RuntimeError, match="protected"):
        payload.validate_plan(report(name), {name: "1"}, {name})


def test_existing_protobuf_change_is_explicit_and_exact():
    with pytest.raises(RuntimeError, match="allow-change"):
        payload.validate_plan(report(), {"protobuf": "7.35.0"}, set())
    rows = payload.validate_plan(report(), {"protobuf": "7.35.0"}, {"protobuf"})
    assert rows[0]["old"] == "7.35.0" and rows[0]["new"] == "6.33.6"


@pytest.mark.parametrize("name", ["cuda-python", "cuda-bindings", "cuda-pathfinder", "nvidia-cuda-nvdisasm"])
def test_new_compiler_bindings_can_be_planned_but_replacement_needs_allowlist(name):
    assert payload.validate_plan(report(name), {}, set())
    with pytest.raises(RuntimeError, match="allow-change"):
        payload.validate_plan(report(name), {name: "0"}, set())


def test_sol_source_revision_must_be_full_pinned_commit():
    value = report("sol-attn", "0.5.0")
    value["install"][0]["download_info"] = {
        "url": "https://github.com/NVlabs/Sana", "vcs_info": {"vcs": "git", "commit_id": "other"}}
    with pytest.raises(RuntimeError, match="revision"):
        payload.validate_plan(value, {}, set())
    value["install"][0]["download_info"]["vcs_info"]["commit_id"] = payload.SOL_SHA
    assert payload.SOL_SHA in payload.validate_plan(value, {}, set())[0]["url"]


def test_dependencies_must_be_wheels_not_unreviewed_builds():
    value = report()
    value["install"][0]["download_info"]["url"] = "https://example.invalid/source.tar.gz"
    with pytest.raises(RuntimeError, match="HTTPS wheel"):
        payload.validate_plan(value, {}, set())


def _wheel(tmp_path, members=()):
    path = tmp_path / "protobuf-6.33.6-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("protobuf-6.33.6.dist-info/METADATA", "Name: protobuf\nVersion: 6.33.6\n")
        for member in members:
            archive.writestr(member, "content")
    return path


def test_staged_wheel_cannot_overwrite_protected_torch_files(tmp_path, monkeypatch):
    monkeypatch.setattr(payload.sysconfig, "get_path", lambda key: "/site")
    with pytest.raises(RuntimeError, match="overlaps"):
        payload.wheel_identity(_wheel(tmp_path, ["torch/__init__.py"]), {"/site/torch/__init__.py"})


def test_staged_wheel_rejects_escape_and_data_destinations(tmp_path):
    for member in ("../escape", "protobuf.data/data/tool"):
        with pytest.raises(RuntimeError):
            payload.wheel_identity(_wheel(tmp_path, [member]), set())


def test_apply_requires_explicit_review_and_maintenance(monkeypatch):
    monkeypatch.setattr(command, "_host", lambda *args: pytest.fail("remote work before authority"))
    assert command.run(config(), apply=True) == 1
    assert command.run(config(), apply=True, plan="a" * 32, plan_digest="b" * 64) == 1


def test_changed_fleet_plan_never_applies(monkeypatch):
    actions = []
    def host(*args):
        actions.append(args[2])
        return {"status": "ok", "plan": {"digest": "a" * 64, "allowed": [], "changes": []}}
    monkeypatch.setattr(command, "_host", host)
    assert command.run(config(), apply=True, plan="c" * 32, plan_digest="bad", idle_maintenance=True) == 1
    assert actions == ["read", "read"]


def test_partial_fleet_apply_stops_and_returns_unknown(monkeypatch):
    cfg = config()
    plan = {"digest": "a" * 64, "allowed": [], "changes": []}
    expected = command.fleet_digest(cfg, "c" * 32, set(), {"a": plan, "b": plan})
    def host(_config, target, action, *_args):
        if action == "apply" and target.name == "b":
            return None
        return {"status": "ok", "plan": plan}
    monkeypatch.setattr(command, "_host", host)
    assert command.run(cfg, apply=True, plan="c" * 32, plan_digest=expected, idle_maintenance=True) == 75


def test_remote_source_is_self_contained_and_nonblocking():
    script = command.host_script(config(), "plan", "a" * 32, set())
    source = script.split("<<'DGXM_SOL_PAYLOAD'\n", 1)[1].rsplit("\nDGXM_SOL_PAYLOAD", 1)[0]
    compile(source, "remote-sol", "exec")
    assert "acquire_lifecycle_lock(Path.home(), True)" in source
    assert "from ." not in source
    assert '"--no-index", "--no-deps"' in source


def test_apply_rechecks_environment_before_pip(monkeypatch):
    plan = {"digest": "a" * 64, "allowed": [], "python": payload.sys.executable,
            "before": {"torch": "1"}, "protected": "old"}
    monkeypatch.setattr(payload, "read_plan", lambda token: plan)
    monkeypatch.setattr(payload, "installed", lambda: {"torch": "2"})
    monkeypatch.setattr(payload, "run_pip", lambda *args: pytest.fail("pip before recheck"))
    with pytest.raises(RuntimeError, match="environment changed"):
        payload.apply_host("a" * 32, "a" * 64, set())


def test_private_plan_rejects_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / ".local").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(RuntimeError, match="unsafe staging"):
        payload.private_plan("a" * 32, create=True)


def test_digest_detects_manifest_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = payload.private_plan("a" * 32, True)
    plan = {"token": "a" * 32, "allowed": []}
    plan["digest"] = payload.digest(plan)
    path = root / "plan.json"
    path.write_text(json.dumps(plan))
    path.chmod(0o600)
    assert payload.read_plan("a" * 32)["digest"] == plan["digest"]
    plan["allowed"] = ["protobuf"]
    path.write_text(json.dumps(plan))
    with pytest.raises(RuntimeError, match="changed"):
        payload.read_plan("a" * 32)


def test_allowed_replacement_with_shared_ownership_is_refused(monkeypatch):
    distributions = [SimpleNamespace(metadata={"Name": name}, files=["tests/__init__.py"],
                                    locate_file=lambda item: Path("/site") / item)
                     for name in ("protobuf", "unrelated")]
    monkeypatch.setattr(payload.md, "distributions", lambda: distributions)
    monkeypatch.setattr(payload, "inventory_path", lambda distribution, item, **kwargs: Path("/site") / item)
    with pytest.raises(RuntimeError, match="overlaps another"):
        payload.preserved_paths({"protobuf"})


def test_wheel_script_destination_cannot_overwrite_existing_command(tmp_path, monkeypatch):
    monkeypatch.setattr(payload.sysconfig, "get_path", lambda key: "/fake/bin" if key == "scripts" else "/site")
    with pytest.raises(RuntimeError, match="overlaps"):
        payload.wheel_identity(_wheel(tmp_path, ["protobuf.data/scripts/tool"]), {"/fake/bin/tool"})


def test_apply_installs_exact_staged_artifacts_without_dependencies_and_reads_back(tmp_path, monkeypatch):
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel = _wheel(wheel_dir)
    monkeypatch.setattr(payload.sysconfig, "get_path", lambda key: str(tmp_path / key))
    before = {"torch": "2.12.0", "protobuf": "7.35.0"}
    after = {"torch": "2.12.0", "protobuf": "6.33.6"}
    plan = {"digest": "a" * 64, "allowed": ["protobuf"], "python": payload.sys.executable,
            "before": before, "protected": "protected", "baseline_violations": [],
            "changes": [{"name": "protobuf", "old": "7.35.0", "new": "6.33.6"}],
            "artifacts": [{"file": wheel.name, "sha256": payload.file_hash(wheel)}]}
    monkeypatch.setattr(payload, "read_plan", lambda token: plan)
    monkeypatch.setattr(payload, "private_plan", lambda token: tmp_path)
    calls = []
    monkeypatch.setattr(payload, "installed", lambda: after if calls else before)
    monkeypatch.setattr(payload, "protected_state", lambda: ("protected", set()))
    monkeypatch.setattr(payload, "preserved_paths", lambda changing, **kwargs: (set(), set()))
    monkeypatch.setattr(payload, "replacement_paths", lambda name: set())
    monkeypatch.setattr(payload, "require_idle", lambda: None)
    monkeypatch.setattr(payload, "verify_requirements", lambda names: None)
    monkeypatch.setattr(payload, "requirement_violations", lambda versions: [])
    monkeypatch.setattr(payload, "run_pip", lambda args: calls.append(args))
    assert payload.apply_host("a" * 32, "a" * 64, {"protobuf"}) == plan
    assert calls == [["install", "--no-index", "--no-deps", str(wheel)]]


def test_changed_wheel_refuses_before_idle_or_install(tmp_path, monkeypatch):
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel = _wheel(wheel_dir)
    plan = {"digest": "a" * 64, "allowed": [], "python": payload.sys.executable,
            "before": {}, "protected": "protected",
            "changes": [{"name": "protobuf", "new": "6.33.6"}],
            "artifacts": [{"file": wheel.name, "sha256": "wrong"}]}
    monkeypatch.setattr(payload, "read_plan", lambda token: plan)
    monkeypatch.setattr(payload, "private_plan", lambda token: tmp_path)
    monkeypatch.setattr(payload, "installed", dict)
    monkeypatch.setattr(payload, "protected_state", lambda: ("protected", set()))
    monkeypatch.setattr(payload, "preserved_paths", lambda changing, **kwargs: (set(), set()))
    monkeypatch.setattr(payload, "replacement_paths", lambda name: set())
    monkeypatch.setattr(payload, "run_pip", lambda args: pytest.fail("changed artifact installed"))
    with pytest.raises(RuntimeError, match="artifact changed"):
        payload.apply_host("a" * 32, "a" * 64, set())


def test_selected_root_package_transition_is_part_of_reviewed_plan():
    rows = payload.validate_plan(report("apache-tvm-ffi", "0.1.14.post1"),
                                 {"apache-tvm-ffi": "0.1.13.post3"}, set())
    assert rows[0]["old"] == "0.1.13.post3"


def test_resolver_archive_hash_is_required_and_retained():
    value = report()
    assert payload.validate_plan(value, {}, set())[0]["archive_sha256"] == "a" * 64
    value["install"][0]["download_info"].pop("archive_info")
    with pytest.raises(RuntimeError, match="SHA256"):
        payload.validate_plan(value, {}, set())


def test_staged_download_must_match_resolver_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(payload, "private_plan", lambda *args: tmp_path)
    monkeypatch.setattr(payload, "scheme_roots", lambda: set())
    monkeypatch.setattr(payload, "installed", lambda: {"torch": "2"})
    monkeypatch.setattr(payload, "protected_state", lambda: ("hash", set()))
    monkeypatch.setattr(payload, "preserved_paths", lambda *args, **kwargs: (set(), set()))
    monkeypatch.setattr(payload, "replacement_paths", lambda name: set())
    monkeypatch.setattr(payload, "requirement_violations", lambda *args: [])
    def pip(args):
        if args[0] == "install":
            (tmp_path / "resolve.json").write_text(json.dumps(report()))
        else:
            _wheel(tmp_path / "wheels")
    monkeypatch.setattr(payload, "run_pip", pip)
    with pytest.raises(RuntimeError, match="resolver SHA256"):
        payload.plan_host("a" * 32, set())


def test_reverse_dependency_new_conflict_detected_without_promoting_baseline(monkeypatch):
    distribution = SimpleNamespace(metadata={"Name": "existing-app"}, requires=["protobuf>=7", "missing-existing"])
    monkeypatch.setattr(payload.md, "distributions", lambda: [distribution])
    before = set(payload.requirement_violations({"protobuf": "7.35.0"}))
    after = set(payload.requirement_violations({"protobuf": "6.33.6"}))
    assert after - before == {"existing-app:protobuf"}
    assert before == {"existing-app:missing-existing"}


def test_record_paths_normalize_and_cannot_escape_scheme(tmp_path, monkeypatch):
    site, scripts = tmp_path / "lib" / "site", tmp_path / "bin"
    site.mkdir(parents=True)
    scripts.mkdir()
    monkeypatch.setattr(payload, "scheme_roots", lambda: {site, scripts})
    distribution = SimpleNamespace(locate_file=lambda item: site / item)
    assert payload.inventory_path(distribution, "../../bin/tool") == scripts / "tool"
    with pytest.raises(RuntimeError, match="escapes"):
        payload.inventory_path(distribution, "../../../unrelated")


def test_generated_entrypoint_cannot_overwrite_preserved_or_unowned_command(tmp_path, monkeypatch):
    scripts = tmp_path / "bin"
    scripts.mkdir()
    (scripts / "torchrun").write_text("user command")
    monkeypatch.setattr(payload.sysconfig, "get_path", lambda key: str(scripts if key == "scripts" else tmp_path / "site"))
    wheel = _wheel(tmp_path)
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("protobuf-6.33.6.dist-info/entry_points.txt", "[console_scripts]\ntorchrun=example:main\n")
    with pytest.raises(RuntimeError, match="overlaps"):
        payload.wheel_identity(wheel, set())


def test_unowned_existing_package_file_is_not_overwritten(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    (site / "unowned.py").write_text("user file")
    monkeypatch.setattr(payload.sysconfig, "get_path", lambda key: str(site))
    with pytest.raises(RuntimeError, match="overlaps"):
        payload.wheel_identity(_wheel(tmp_path, ["unowned.py"]), set())


def test_pip_ignores_destination_redirection_and_configuration(monkeypatch):
    from types import SimpleNamespace

    for key in ("PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PIP_CONFIG_FILE"):
        monkeypatch.setenv(key, "/unrelated")
    captured = {}
    def run(args, **kwargs):
        captured.update(args=args, **kwargs)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(payload.subprocess, "run", run)
    payload.run_pip(["install", "--no-deps", "/staged.whl"])
    assert captured["args"][1:6] == ["-I", "-m", "pip", "--isolated", "--require-virtualenv"]
    assert captured["env"]["PIP_CONFIG_FILE"] == payload.os.devnull
    assert not any(key in captured["env"] for key in ("PIP_TARGET", "PIP_PREFIX", "PIP_USER"))


def test_nonvenv_installation_is_refused(monkeypatch):
    monkeypatch.setattr(payload.sys, "prefix", payload.sys.base_prefix)
    with pytest.raises(RuntimeError, match="virtual environment"):
        payload.scheme_roots()


def test_failed_pip_keeps_readback_when_metadata_and_protected_files_are_unreadable(tmp_path, monkeypatch):
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel = _wheel(wheel_dir)
    monkeypatch.setattr(payload.sysconfig, "get_path", lambda key: str(tmp_path / key))
    plan = {"digest": "a" * 64, "allowed": [], "python": payload.sys.executable,
            "before": {}, "protected": "protected", "baseline_violations": [],
            "changes": [{"name": "protobuf", "new": "6.33.6"}],
            "artifacts": [{"file": wheel.name, "sha256": payload.file_hash(wheel)}]}
    failed = False
    def inventory():
        if failed:
            raise OSError("broken metadata")
        return {}
    def protected():
        if failed:
            raise OSError("missing protected file")
        return "protected", set()
    def pip(args):
        nonlocal failed
        failed = True
        raise RuntimeError("pip failure")
    monkeypatch.setattr(payload, "read_plan", lambda token: plan)
    monkeypatch.setattr(payload, "private_plan", lambda token: tmp_path)
    monkeypatch.setattr(payload, "installed", inventory)
    monkeypatch.setattr(payload, "protected_state", protected)
    monkeypatch.setattr(payload, "preserved_paths", lambda *args, **kwargs: (set(), set()))
    monkeypatch.setattr(payload, "replacement_paths", lambda name: set())
    monkeypatch.setattr(payload, "require_idle", lambda: None)
    monkeypatch.setattr(payload, "run_pip", pip)
    with pytest.raises(RuntimeError, match="pip failed"):
        payload.apply_host("a" * 32, "a" * 64, set())
    readback = json.loads((tmp_path / "readback.json").read_text())
    assert readback == {"pip_completed": False, "packages_match": None,
                        "requirements_match": None, "protected_unchanged": None}


def test_new_wheel_member_cannot_follow_existing_directory_symlink(tmp_path, monkeypatch):
    site, elsewhere = tmp_path / "site", tmp_path / "elsewhere"
    site.mkdir()
    elsewhere.mkdir()
    (site / "namespace").symlink_to(elsewhere, target_is_directory=True)
    monkeypatch.setattr(payload.sysconfig, "get_path", lambda key: str(site))
    with pytest.raises(RuntimeError, match="symlink"):
        payload.wheel_identity(_wheel(tmp_path, ["namespace/new.py"]), set())
    assert list(elsewhere.iterdir()) == []


def test_staged_wheels_cannot_claim_the_same_new_file(tmp_path, monkeypatch):
    monkeypatch.setattr(payload.sysconfig, "get_path", lambda key: str(tmp_path / key))
    first = _wheel(tmp_path, ["shared.py"])
    second = tmp_path / "another-1.0-py3-none-any.whl"
    with zipfile.ZipFile(second, "w") as archive:
        archive.writestr("another-1.0.dist-info/METADATA", "Name: another\nVersion: 1.0\n")
        archive.writestr("shared.py", "other owner")
    staged: set[str] = set()
    payload.wheel_identity(first, set(), staged_destinations=staged)
    with pytest.raises(RuntimeError, match="staged distributions overlap"):
        payload.wheel_identity(second, set(), staged_destinations=staged)


def test_replacement_ownership_is_scoped_to_each_distribution(monkeypatch, tmp_path):
    distributions = {name: SimpleNamespace(files=[f"{name}.py"], metadata={"Name": name})
                     for name in ("first", "second")}
    monkeypatch.setattr(payload, "distribution_inventory", lambda: distributions)
    monkeypatch.setattr(payload, "inventory_path", lambda dist, item: tmp_path / item)
    assert payload.replacement_paths("first") == {str(tmp_path / "first.py")}
    assert str(tmp_path / "second.py") not in payload.replacement_paths("first")


def test_apply_failure_keeps_bounded_refusal_reason(monkeypatch, capsys):
    cfg = config()
    plan = {"digest": "a" * 64, "allowed": [], "changes": []}
    expected = command.fleet_digest(cfg, "c" * 32, set(), {"a": plan, "b": plan})
    def host(_config, target, action, *_args):
        if action == "apply":
            return {"status": "failed", "reason": "process_observation_unknown"}
        return {"status": "ok", "plan": plan}
    monkeypatch.setattr(command, "_host", host)
    assert command.run(cfg, apply=True, plan="c" * 32, plan_digest=expected, idle_maintenance=True) == 75
    assert "apply/readback incomplete: process_observation_unknown" in capsys.readouterr().out


def test_dependency_permission_refusal_names_the_required_allow_change():
    with pytest.raises(RuntimeError, match=r"^protobuf requires --allow-change$"):
        payload.validate_plan(report("protobuf"), {"protobuf": "7.35.0"}, set())


def test_protected_package_refusal_names_the_distribution():
    with pytest.raises(RuntimeError, match=r"^protected package torch cannot change$"):
        payload.validate_plan(report("torch"), {}, {"torch"})


def test_invalid_resolver_name_never_becomes_a_diagnostic():
    with pytest.raises(RuntimeError, match=r"^invalid resolver distribution name$"):
        payload.validate_plan(report("private/path?secret"), {}, set())


def test_resolver_can_identify_nonprotected_transition_before_permission_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(payload, "private_plan", lambda *args: tmp_path)
    monkeypatch.setattr(payload, "scheme_roots", lambda: set())
    monkeypatch.setattr(payload, "installed", lambda: {"torch": "2.12.0", "protobuf": "7.35.0"})
    monkeypatch.setattr(payload, "protected_state", lambda: ("hash", set()))
    def pip(args):
        assert args[0] == "install" and "--dry-run" in args
        constraints = (tmp_path / "constraints.txt").read_text()
        assert "torch==2.12.0" in constraints and "protobuf" not in constraints
        (tmp_path / "resolve.json").write_text(json.dumps(report()))
    monkeypatch.setattr(payload, "run_pip", pip)
    with pytest.raises(RuntimeError, match="protobuf requires --allow-change"):
        payload.plan_host("a" * 32, set())
    assert not (tmp_path / "wheels").exists()


def _venv_inventory_fixture(tmp_path, monkeypatch):
    prefix = tmp_path / "venv"
    site = prefix / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    (prefix / "bin").mkdir()
    prefix.chmod(0o755)
    for path in prefix.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
    monkeypatch.setattr(payload.sys, "prefix", str(prefix))
    monkeypatch.setattr(payload.sys, "base_prefix", "/system-python")
    monkeypatch.setattr(payload.sysconfig, "get_path", lambda key: str(prefix / "bin" if key == "scripts" else site))
    return prefix, SimpleNamespace(locate_file=lambda item: site / item)


@pytest.mark.parametrize("record", [
    "../../../include/site/python3.12/greenlet/greenlet.h",
    "../../../share/man/man1/ttx.1",
    "../../../share/man/man1/isympy.1",
])
def test_owned_venv_header_and_manpage_records_are_valid(tmp_path, monkeypatch, record):
    prefix, distribution = _venv_inventory_fixture(tmp_path, monkeypatch)
    expected = prefix / record.removeprefix("../../../")
    expected.parent.mkdir(parents=True)
    for path in prefix.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
    expected.write_text("installed payload")
    assert payload.inventory_path(distribution, record) == expected
    with pytest.raises(RuntimeError, match="escapes"):
        payload.inventory_path(distribution, "../../../private-user-file")
    with pytest.raises(RuntimeError, match="escapes"):
        payload.inventory_path(distribution, "/usr/include/python3.12/Python.h")


@pytest.mark.parametrize("kind", ["symlink", "writable"])
def test_unsafe_venv_share_inventory_root_is_refused(tmp_path, monkeypatch, kind):
    prefix, _ = _venv_inventory_fixture(tmp_path, monkeypatch)
    if kind == "symlink":
        (prefix / "share").symlink_to(tmp_path, target_is_directory=True)
    else:
        (prefix / "share").mkdir()
        (prefix / "share").chmod(0o777)
    with pytest.raises(RuntimeError, match="unsafe virtual environment scheme"):
        payload.scheme_roots()


def test_absent_header_share_roots_are_not_created_or_accepted(tmp_path, monkeypatch):
    prefix, distribution = _venv_inventory_fixture(tmp_path, monkeypatch)
    roots = payload.scheme_roots()
    assert prefix / "include" not in roots and prefix / "share" not in roots
    assert not (prefix / "include").exists() and not (prefix / "share").exists()
    with pytest.raises(RuntimeError, match="escapes"):
        payload.inventory_path(distribution, "../../../include/new.h")
