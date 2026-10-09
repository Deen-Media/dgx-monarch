"""Strict source/runtime provenance regression tests."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types
from importlib import machinery, metadata
from pathlib import Path

import pytest

from dgx_monarch import runtime_identity as identity
from dgx_monarch import runtime_module_origins as module_origins
from dgx_monarch import runtime_provenance as provenance


def _git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )


def _committed_package(tmp_path: Path) -> tuple[Path, Path]:
    checkout = tmp_path / "checkout"
    package = checkout / "src" / "dgx_monarch"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (checkout / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    _git(checkout, "init")
    _git(checkout, "config", "user.email", "tests.invalid")
    _git(checkout, "config", "user.name", "tests")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "fixture")
    return checkout, package


def test_source_only_import_policy_redirects_and_disables_bytecode(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    monkeypatch.setattr(sys, "pycache_prefix", None)
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    monkeypatch.delenv("PYTHONPYCACHEPREFIX", raising=False)

    provenance.enforce_source_only_imports()

    assert sys.dont_write_bytecode is True
    assert sys.pycache_prefix == provenance.SOURCE_ONLY_PYCACHE_PREFIX
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    assert os.environ["PYTHONPYCACHEPREFIX"] == provenance.SOURCE_ONLY_PYCACHE_PREFIX


def test_managed_source_runtime_version_is_always_source_bound(
    monkeypatch: pytest.MonkeyPatch,
):
    from dgx_monarch import __version__

    queried: list[str] = []

    def fake_version(name: str) -> str:
        assert name != "dgx-monarch"
        queried.append(name)
        if name == "xfuser":
            raise metadata.PackageNotFoundError(name)
        return f"exact-{name}"

    monkeypatch.setattr(metadata, "version", fake_version)
    assert identity.runtime_versions() == {
        "python": identity.platform.python_version(),
        "dgx-monarch": __version__,
        "torch": "exact-torch",
        "torchmonarch": "exact-torchmonarch",
        "xfuser": "missing",
        "yunchang": "exact-yunchang",
    }
    assert queried == ["torch", "torchmonarch", "xfuser", "yunchang"]


def test_comfy_inventory_keeps_nested_code_but_excludes_root_data(tmp_path: Path):
    root = tmp_path / "ComfyUI"
    included = root / "comfy" / "ldm" / "models" / "core.py"
    api_included = root / "comfy_api" / "input" / "routes.py"
    excluded = root / "models" / "ignored.py"
    custom = root / "custom_nodes" / "third_party.py"
    for path in (included, api_included, excluded, custom):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"VALUE = {path.name!r}\n", encoding="utf-8")

    disabled = provenance.source_manifest_sha256(
        root, scope="comfyui", custom_nodes_disabled=True
    )
    excluded.write_text("VALUE = 'changed data'\n", encoding="utf-8")
    custom.write_text("VALUE = 'changed disabled node'\n", encoding="utf-8")
    assert (
        provenance.source_manifest_sha256(
            root, scope="comfyui", custom_nodes_disabled=True
        )
        == disabled
    )

    included.write_text("VALUE = 'changed core'\n", encoding="utf-8")
    assert (
        provenance.source_manifest_sha256(
            root, scope="comfyui", custom_nodes_disabled=True
        )
        != disabled
    )


def test_direct_pyc_and_included_symlink_directory_are_rejected(tmp_path: Path):
    root = tmp_path / "package"
    root.mkdir()
    (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "shadow.pyc").write_bytes(b"not trusted bytecode")
    with pytest.raises(RuntimeError, match=r"sourceless \.pyc"):
        provenance.source_manifest_sha256(root, scope="dgx_monarch")

    (root / "shadow.pyc").unlink()
    target = tmp_path / "external"
    target.mkdir()
    (target / "plugin.py").write_text("VALUE = 2\n", encoding="utf-8")
    (root / "linked").symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink directory"):
        provenance.source_manifest_sha256(root, scope="dgx_monarch")


def test_source_inventory_rejects_unreadable_included_subtree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "package"
    blocked = root / "blocked"
    blocked.mkdir(parents=True)
    (root / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
    (blocked / "hidden.py").write_text("VALUE = 2\n", encoding="utf-8")
    real_scandir = os.scandir

    def guarded_scandir(path):
        if Path(path) == blocked:
            raise PermissionError("included subtree is unreadable")
        return real_scandir(path)

    monkeypatch.setattr(provenance.os, "scandir", guarded_scandir)
    with pytest.raises(PermissionError, match="included subtree is unreadable"):
        provenance.source_manifest_sha256(root, scope="dgx_monarch")


def test_git_head_inventory_rejects_ignored_importable_source(tmp_path: Path):
    checkout, package = _committed_package(tmp_path)
    facts, tracked = provenance._git_checkout_facts(
        checkout,
        tracked_prefix="src/dgx_monarch",
        scope="dgx_monarch",
        custom_nodes_disabled=False,
    )
    assert facts["dirty"] is False
    assert tracked == frozenset({"__init__.py"})

    (package / "ignored.py").write_text("VALUE = 'ignored by Git'\n", encoding="utf-8")
    facts_after, tracked_after = provenance._git_checkout_facts(
        checkout,
        tracked_prefix="src/dgx_monarch",
        scope="dgx_monarch",
        custom_nodes_disabled=False,
    )
    assert facts_after["dirty"] is False
    with pytest.raises(RuntimeError, match="files tracked at HEAD"):
        provenance.source_manifest_sha256(
            package,
            scope="dgx_monarch",
            expected_paths=tracked_after,
        )


def test_empty_git_prefix_discovers_root_and_nested_sources(tmp_path: Path):
    checkout = tmp_path / "ComfyUI"
    nested = checkout / "comfy" / "ldm" / "core.py"
    nested.parent.mkdir(parents=True)
    (checkout / "folder_paths.py").write_text("VALUE = 1\n", encoding="utf-8")
    nested.write_text("VALUE = 2\n", encoding="utf-8")
    _git(checkout, "init")
    _git(checkout, "config", "user.email", "tests.invalid")
    _git(checkout, "config", "user.name", "tests")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "fixture")

    facts, tracked = provenance._git_checkout_facts(
        checkout,
        tracked_prefix="",
        scope="comfyui",
        custom_nodes_disabled=True,
    )
    assert facts["dirty"] is False
    assert tracked == frozenset({"folder_paths.py", "comfy/ldm/core.py"})
    assert len(
        provenance.source_manifest_sha256(
            checkout,
            scope="comfyui",
            custom_nodes_disabled=True,
            expected_paths=tracked,
        )
    ) == 64


def _committed_comfy_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "ComfyUI"
    sources = {
        "comfy/core.py": "VALUE = 'included'\n",
        "comfy/custom_nodes/nested.py": "VALUE = 'nested included'\n",
        "custom_nodes/plugin.py": "VALUE = 'excluded when disabled'\n",
        "custom_nodes_backup.py": "VALUE = 'similarly named included'\n",
    }
    for relative, content in sources.items():
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(checkout, "init")
    _git(checkout, "config", "user.email", "tests.invalid")
    _git(checkout, "config", "user.name", "tests")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "fixture")
    return checkout


def _checkout_dirty(
    checkout: Path, *, scope: str = "comfyui", custom_nodes_disabled: bool
) -> bool:
    facts, _tracked = provenance._git_checkout_facts(
        checkout,
        tracked_prefix="",
        scope=scope,
        custom_nodes_disabled=custom_nodes_disabled,
    )
    return bool(facts["dirty"])


def test_disabled_comfy_status_ignores_only_top_level_custom_nodes(
    tmp_path: Path,
):
    checkout = _committed_comfy_checkout(tmp_path)
    plugin = checkout / "custom_nodes" / "plugin.py"
    plugin.write_text("VALUE = 'tracked unstaged change'\n", encoding="utf-8")
    assert _checkout_dirty(checkout, custom_nodes_disabled=True) is False

    _git(checkout, "add", "custom_nodes/plugin.py")
    assert _checkout_dirty(checkout, custom_nodes_disabled=True) is False

    (checkout / "custom_nodes" / "untracked.py").write_text(
        "VALUE = 'untracked change'\n", encoding="utf-8"
    )
    assert _checkout_dirty(checkout, custom_nodes_disabled=True) is False

    (checkout / "comfy" / "core.py").write_text(
        "VALUE = 'included change'\n", encoding="utf-8"
    )
    assert _checkout_dirty(checkout, custom_nodes_disabled=True) is True


@pytest.mark.parametrize(
    ("scope", "custom_nodes_disabled"),
    [("comfyui", False), ("dgx_monarch", True)],
)
def test_top_level_custom_node_changes_remain_dirty_outside_disabled_comfy_policy(
    tmp_path: Path, scope: str, custom_nodes_disabled: bool
):
    checkout = _committed_comfy_checkout(tmp_path)
    (checkout / "custom_nodes" / "plugin.py").write_text(
        "VALUE = 'policy must include this change'\n", encoding="utf-8"
    )

    assert (
        _checkout_dirty(
            checkout,
            scope=scope,
            custom_nodes_disabled=custom_nodes_disabled,
        )
        is True
    )


def test_disabled_comfy_status_keeps_nested_custom_nodes_dirty(tmp_path: Path):
    checkout = _committed_comfy_checkout(tmp_path)
    (checkout / "comfy" / "custom_nodes" / "nested.py").write_text(
        "VALUE = 'nested paths are included'\n", encoding="utf-8"
    )

    assert _checkout_dirty(checkout, custom_nodes_disabled=True) is True


def test_disabled_comfy_status_keeps_similarly_named_root_dirty(tmp_path: Path):
    checkout = _committed_comfy_checkout(tmp_path)
    (checkout / "custom_nodes_backup.py").write_text(
        "VALUE = 'literal exclusion must not match this'\n", encoding="utf-8"
    )

    assert _checkout_dirty(checkout, custom_nodes_disabled=True) is True


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
@pytest.mark.parametrize(
    ("path", "scope", "custom_nodes_disabled", "expected_dirty"),
    [
        ("custom_nodes/plugin.py", "comfyui", True, False),
        ("comfy/core.py", "comfyui", True, True),
        ("comfy/custom_nodes/nested.py", "comfyui", True, True),
        ("custom_nodes_backup.py", "comfyui", True, True),
        ("custom_nodes/plugin.py", "comfyui", False, True),
        ("custom_nodes/plugin.py", "dgx_monarch", True, True),
    ],
)
def test_disabled_comfy_hidden_index_flags_follow_literal_policy_boundary(
    tmp_path: Path,
    flag: str,
    path: str,
    scope: str,
    custom_nodes_disabled: bool,
    expected_dirty: bool,
):
    checkout = _committed_comfy_checkout(tmp_path)
    _git(checkout, "update-index", flag, path)

    assert (
        _checkout_dirty(
            checkout,
            scope=scope,
            custom_nodes_disabled=custom_nodes_disabled,
        )
        is expected_dirty
    )


@pytest.mark.parametrize("direction", ["excluded_to_included", "included_to_excluded"])
def test_disabled_comfy_status_keeps_cross_boundary_renames_dirty(
    tmp_path: Path, direction: str
):
    checkout = _committed_comfy_checkout(tmp_path)
    if direction == "excluded_to_included":
        source = "custom_nodes/plugin.py"
        destination = "comfy/renamed_plugin.py"
    else:
        source = "comfy/core.py"
        destination = "custom_nodes/renamed_core.py"
    _git(checkout, "mv", source, destination)

    assert _checkout_dirty(checkout, custom_nodes_disabled=True) is True


def test_tracked_native_extension_participates_in_manifest(tmp_path: Path):
    checkout, package = _committed_package(tmp_path)
    extension = package / f"native{machinery.EXTENSION_SUFFIXES[0]}"
    extension.write_bytes(b"first native payload")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "native")
    _facts, tracked = provenance._git_checkout_facts(
        checkout,
        tracked_prefix="src/dgx_monarch",
        scope="dgx_monarch",
        custom_nodes_disabled=False,
    )
    assert extension.name in tracked
    first = provenance.source_manifest_sha256(
        package, scope="dgx_monarch", expected_paths=tracked
    )
    extension.write_bytes(b"second native payload")
    second = provenance.source_manifest_sha256(
        package, scope="dgx_monarch", expected_paths=tracked
    )
    assert first != second


def test_broken_git_marker_never_downgrades_to_managed_copy(tmp_path: Path):
    root = tmp_path / "broken"
    root.mkdir()
    (root / ".git").write_text("gitdir: /missing\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Git query failed"):
        provenance._git_checkout_facts(
            root,
            tracked_prefix="src/dgx_monarch",
            scope="dgx_monarch",
            custom_nodes_disabled=False,
            allow_managed_copy=True,
        )

    (root / ".git").unlink()
    facts, tracked = provenance._git_checkout_facts(
        root,
        tracked_prefix="src/dgx_monarch",
        scope="dgx_monarch",
        custom_nodes_disabled=False,
        allow_managed_copy=True,
    )
    assert facts == {"mode": "managed_copy"}
    assert tracked is None


def test_runtime_git_queries_ignore_ambient_repository_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout, _package = _committed_package(tmp_path / "expected")
    poison, poison_package = _committed_package(tmp_path / "poison")
    (poison_package / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(poison, "add", ".")
    _git(poison, "commit", "-m", "poison")
    expected_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    poison_commit = subprocess.run(
        ["git", "-C", str(poison), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert poison_commit != expected_commit
    monkeypatch.setenv("GIT_DIR", str(poison / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(poison))
    monkeypatch.setenv("GIT_INDEX_FILE", str(poison / ".git" / "index"))

    facts, _tracked = provenance._git_checkout_facts(
        checkout,
        tracked_prefix="src/dgx_monarch",
        scope="dgx_monarch",
        custom_nodes_disabled=False,
    )

    assert facts["commit"] == expected_commit


def test_runtime_git_discovery_uses_current_path_but_invocation_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prefix = tmp_path / "operator-prefix" / "bin"
    prefix.mkdir(parents=True)
    executable = prefix / "git"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(prefix))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "poison.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "poison-tree"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "poison-index"))
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        calls.append((command, environment))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    provenance._git_executable.cache_clear()
    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    try:
        provenance._run_git(tmp_path, "status")
    finally:
        provenance._git_executable.cache_clear()

    assert len(calls) == 1
    command, child_environment = calls[0]
    assert command[0] == str(executable.resolve())
    assert Path(command[0]).is_absolute()
    assert child_environment["PATH"] == os.defpath
    assert "GIT_DIR" not in child_environment
    assert "GIT_WORK_TREE" not in child_environment
    assert "GIT_INDEX_FILE" not in child_environment


def test_runtime_git_checkout_accepts_prefix_only_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout, _package = _committed_package(tmp_path / "repo")
    system_git = shutil.which("git")
    assert system_git is not None
    prefix = tmp_path / "operator-prefix" / "bin"
    prefix.mkdir(parents=True)
    (prefix / "git").symlink_to(Path(system_git).resolve())
    monkeypatch.setenv("PATH", str(prefix))

    provenance._git_executable.cache_clear()
    try:
        facts, _tracked = provenance._git_checkout_facts(
            checkout,
            tracked_prefix="src/dgx_monarch",
            scope="dgx_monarch",
            custom_nodes_disabled=False,
        )
    finally:
        provenance._git_executable.cache_clear()

    assert facts["mode"] == "git"


def test_runtime_git_rejects_executable_identity_drift_during_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prefix = tmp_path / "operator-prefix" / "bin"
    prefix.mkdir(parents=True)
    executable = prefix / "git"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    replacement = prefix / "git.replacement"
    replacement.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    replacement.chmod(0o755)
    monkeypatch.setenv("PATH", str(prefix))

    def drifting_run(
        command: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        replacement.replace(executable)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    provenance._git_executable.cache_clear()
    monkeypatch.setattr(provenance.subprocess, "run", drifting_run)
    try:
        with pytest.raises(RuntimeError, match="Git query failed"):
            provenance._run_git(tmp_path, "status")
    finally:
        provenance._git_executable.cache_clear()


def test_runtime_git_queries_ignore_local_replace_refs(tmp_path: Path):
    checkout, package = _committed_package(tmp_path)
    first_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (package / "second.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "-m", "second")
    second_commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _git(checkout, "replace", second_commit, first_commit)

    facts, tracked = provenance._git_checkout_facts(
        checkout,
        tracked_prefix="src/dgx_monarch",
        scope="dgx_monarch",
        custom_nodes_disabled=False,
    )

    assert facts == {"mode": "git", "commit": second_commit, "dirty": False}
    assert tracked == frozenset({"__init__.py", "second.py"})


def test_strict_runtime_rejects_dirty_git_facts():
    with pytest.raises(RuntimeError, match="not clean"):
        provenance._require_clean_git(
            {"mode": "git", "dirty": True}, scope="fixture"
        )
    provenance._require_clean_git(
        {"mode": "managed_copy"}, scope="fixture"
    )


def test_managed_pythonpath_rejects_every_sibling_entry(tmp_path: Path):
    package = tmp_path / "managed-src" / "dgx_monarch"
    package.mkdir(parents=True)
    provenance._validate_managed_pythonpath(package)

    (package.parent / "folder_paths.py").write_text("VALUE = 'shadow'\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected sibling"):
        provenance._validate_managed_pythonpath(package)


def test_resolved_artifact_identity_binds_actual_comfy_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifact = tmp_path / "alternate" / "model.safetensors"
    artifact.parent.mkdir()
    artifact.write_bytes(b"model")
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda kind, name: (
        str(artifact) if (kind, name) == ("diffusion_models", artifact.name) else None
    )
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)

    rows = provenance._resolved_artifact_identities(
        [{"id": "model", "kind": "diffusion_models", "file": artifact.name}]
    )
    assert [row["id"] for row in rows] == ["model"]
    assert len(str(rows[0]["identity_sha256"])) == 64


@pytest.mark.parametrize("malformed", [{}, "", 0, False, (), b""])
def test_runtime_snapshot_rejects_falsey_malformed_artifact_manifest(
    malformed: object,
):
    with pytest.raises(ValueError, match="artifact manifest is invalid"):
        provenance.runtime_provenance_snapshot(
            malformed, custom_nodes_disabled=True
        )


def test_custom_node_bootstrap_policy_is_fixed_for_process_lifetime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, isolated_environ: None
):
    from dgx_monarch.actor import comfy_bridge

    comfy_dir = tmp_path / "ComfyUI"
    (comfy_dir / "comfy").mkdir(parents=True)
    (comfy_dir / "comfy" / "sd.py").write_text("", encoding="utf-8")
    comfy = types.ModuleType("comfy")
    comfy.__path__ = [str(comfy_dir / "comfy")]
    model_management = types.ModuleType("comfy.model_management")
    comfy.model_management = model_management
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management)
    monkeypatch.setattr(comfy_bridge, "_BOOTSTRAPPED", None)
    monkeypatch.setattr(comfy_bridge, "_CUSTOM_NODES_DISABLED", None)
    monkeypatch.setattr(comfy_bridge, "_patch_uma_free_memory", lambda *_args: None)
    monkeypatch.setattr(comfy_bridge, "_instrument_gpu_load", lambda: None)
    monkeypatch.setattr(comfy_bridge, "_uma_memory_defaults", lambda args: dict(args))
    monkeypatch.setattr(comfy_bridge, "_apply_worker_args", lambda _args: None)
    loaded = []
    monkeypatch.setattr(
        comfy_bridge, "_load_custom_node_modules", lambda: loaded.append(True)
    )
    # ensure_comfy puts the checkout on sys.path; this copy is torn down with
    # the test, so the temporary directory does not stay importable.
    monkeypatch.setattr(sys, "path", list(sys.path))

    comfy_bridge.ensure_comfy(
        str(comfy_dir), {"disable_custom_nodes": True}, gpus_per_host=1
    )
    assert loaded == []
    assert comfy_bridge._CUSTOM_NODES_DISABLED is True
    with pytest.raises(RuntimeError, match="different custom-node policy"):
        comfy_bridge.ensure_comfy(str(comfy_dir), {}, gpus_per_host=1)


def test_private_init_flag_reaches_mesh_policy_without_public_widget(
    monkeypatch: pytest.MonkeyPatch,
):
    from dgx_monarch.nodes import init as init_module

    handle = types.SimpleNamespace(world=2)
    monkeypatch.setattr(init_module, "get_mesh", lambda **_kwargs: handle)
    inputs = init_module.DGXMonarchInit.INPUT_TYPES()
    assert "disable_custom_nodes" not in inputs["optional"]

    (mesh,) = init_module.DGXMonarchInit()._init_with_bootstrap_policy(
        topology="auto",
        mode="cluster",
        disable_custom_nodes=True,
    )
    assert mesh.worker_args["disable_custom_nodes"] is True


def _loaded_source_module(
    name: str, path: Path, *, package_paths: list[Path] | None = None
) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__cached__ = f"{provenance.SOURCE_ONLY_PYCACHE_PREFIX}{path}.pyc"
    if package_paths is not None:
        module.__path__ = [str(item) for item in package_paths]
    return module


def _origin_fixture(tmp_path: Path) -> tuple[Path, Path, frozenset[str], frozenset[str]]:
    dgx_root = tmp_path / "managed" / "dgx_monarch"
    comfy_root = tmp_path / "ComfyUI"
    dgx_worker = dgx_root / "actor" / "worker.py"
    comfy_init = comfy_root / "comfy" / "__init__.py"
    comfy_sampler = comfy_root / "comfy" / "samplers.py"
    for path in (dgx_worker, comfy_init, comfy_sampler):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("VALUE = 1\n", encoding="utf-8")
    return (
        dgx_root,
        comfy_root,
        frozenset({"actor/worker.py"}),
        frozenset({"comfy/__init__.py", "comfy/samplers.py"}),
    )


def test_loaded_module_origins_bind_exact_attested_trees(tmp_path: Path):
    dgx_root, comfy_root, dgx_allowed, comfy_allowed = _origin_fixture(tmp_path)
    modules = (
        (
            "comfy",
            _loaded_source_module(
                "comfy",
                comfy_root / "comfy" / "__init__.py",
                package_paths=[comfy_root / "comfy"],
            ),
        ),
        (
            "comfy.samplers",
            _loaded_source_module(
                "comfy.samplers", comfy_root / "comfy" / "samplers.py"
            ),
        ),
        (
            "dgx_monarch.actor.worker",
            _loaded_source_module(
                "dgx_monarch.actor.worker", dgx_root / "actor" / "worker.py"
            ),
        ),
    )

    digest = module_origins.loaded_module_origins_sha256(
        dgx_root=dgx_root,
        comfy_root=comfy_root,
        dgx_allowed_paths=dgx_allowed,
        comfy_allowed_paths=comfy_allowed,
        source_cache_prefix=provenance.SOURCE_ONLY_PYCACHE_PREFIX,
        _module_items=modules,
    )

    assert len(digest) == 64


@pytest.mark.parametrize(
    "case",
    [
        "foreign_comfy",
        "foreign_comfy_api_nodes",
        "foreign_comfy_config",
        "foreign_comfy_utils",
        "foreign_comfy_alembic",
        "foreign_comfy_hook_breaker",
        "excluded_comfy",
        "foreign_dgx",
    ],
)
def test_loaded_module_origins_reject_foreign_or_excluded_sources(
    tmp_path: Path, case: str
):
    dgx_root, comfy_root, dgx_allowed, comfy_allowed = _origin_fixture(tmp_path)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    excluded = comfy_root / "custom_nodes" / "model_management.py"
    excluded.parent.mkdir()
    excluded.write_text("VALUE = 3\n", encoding="utf-8")
    selected = {
        "foreign_comfy": (
            "comfy.samplers",
            foreign / "module.py",
        ),
        "foreign_comfy_api_nodes": (
            "comfy_api_nodes.video",
            foreign / "module.py",
        ),
        "foreign_comfy_config": (
            "comfy_config.folder_paths",
            foreign / "module.py",
        ),
        "foreign_comfy_utils": ("utils.extra", foreign / "module.py"),
        "foreign_comfy_alembic": ("alembic_db.rev", foreign / "module.py"),
        "foreign_comfy_hook_breaker": (
            "hook_breaker_ac10a0",
            foreign / "module.py",
        ),
        "excluded_comfy": (
            "comfy.model_management",
            excluded,
        ),
        "foreign_dgx": (
            "dgx_monarch.actor.worker",
            foreign / "module.py",
        ),
    }
    name, path = selected[case]

    with pytest.raises(RuntimeError, match=r"source tree|attested import surface"):
        module_origins.loaded_module_origins_sha256(
            dgx_root=dgx_root,
            comfy_root=comfy_root,
            dgx_allowed_paths=dgx_allowed,
            comfy_allowed_paths=comfy_allowed,
            source_cache_prefix=provenance.SOURCE_ONLY_PYCACHE_PREFIX,
            _module_items=((name, _loaded_source_module(name, path)),),
        )


def test_loaded_module_origins_reject_namespace_path_injection(tmp_path: Path):
    dgx_root, comfy_root, dgx_allowed, comfy_allowed = _origin_fixture(tmp_path)
    injected = tmp_path / "injected-comfy"
    injected.mkdir()
    comfy = _loaded_source_module(
        "comfy",
        comfy_root / "comfy" / "__init__.py",
        package_paths=[comfy_root / "comfy", injected],
    )

    with pytest.raises(RuntimeError, match="package path is not exact"):
        module_origins.loaded_module_origins_sha256(
            dgx_root=dgx_root,
            comfy_root=comfy_root,
            dgx_allowed_paths=dgx_allowed,
            comfy_allowed_paths=comfy_allowed,
            source_cache_prefix=provenance.SOURCE_ONLY_PYCACHE_PREFIX,
            _module_items=(("comfy", comfy),),
        )


def test_loaded_module_origins_reject_pre_policy_bytecode_cache(tmp_path: Path):
    dgx_root, comfy_root, dgx_allowed, comfy_allowed = _origin_fixture(tmp_path)
    sampler = _loaded_source_module(
        "comfy.samplers", comfy_root / "comfy" / "samplers.py"
    )
    sampler.__cached__ = str(comfy_root / "comfy" / "__pycache__" / "samplers.pyc")

    with pytest.raises(RuntimeError, match="predates the source-only"):
        module_origins.loaded_module_origins_sha256(
            dgx_root=dgx_root,
            comfy_root=comfy_root,
            dgx_allowed_paths=dgx_allowed,
            comfy_allowed_paths=comfy_allowed,
            source_cache_prefix=provenance.SOURCE_ONLY_PYCACHE_PREFIX,
            _module_items=(("comfy.samplers", sampler),),
        )
