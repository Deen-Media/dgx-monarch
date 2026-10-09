"""Contracts for the torchmonarch surface canary, run on fake modules and the committed pin snapshot."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.monarch_surface import (
    SNAPSHOT_SCHEMA,
    TOUCHPOINTS,
    MonarchTouchpoint,
    collect_snapshot,
    collect_touchpoints,
)
from dgx_monarch.monarch_surface_report import (
    compare_snapshots,
    render_comparison_markdown,
    snapshot_blockers,
)

REPO = Path(__file__).resolve().parents[1]

PIN_SEMANTICS = {
    "public_future_get_in_asyncio": {
        "outcome": "returned",
        "result_type": "builtins.tuple",
        "warnings": [
            {
                "category": "builtins.UserWarning",
                "message": (
                    "Blocking get() was called from a running asyncio event loop. "
                    "It is a synchronous, blocking call that can freeze the loop "
                    "and deadlock; use as_asyncio() (or await) instead."
                ),
            }
        ],
    },
    "raw_python_task_await_in_asyncio": {
        "exception_type": "builtins.RuntimeError",
        "message": (
            "Attempting to __await__ a PythonTask when the asyncio event loop is active. "
            "PythonTask objects should only be awaited in coroutines passed to "
            "PythonTask.from_coroutine"
        ),
        "outcome": "raised",
    },
    "unhandled_fault_hook_assignment": {"outcome": "writable"},
    # torchmonarch 0.6.0 defaults to queue dispatch, and long-running GPUWorker
    # endpoints use @concurrent_endpoint (actor/worker.py). False or a missing
    # flag changes dispatch semantics and must produce REVIEW REQUIRED.
    "actor_queue_dispatch_default": {
        "outcome": "observed",
        "actor_queue_dispatch": True,
    },
    "mesh_factory_private_ownership": {
        "outcome": "observed",
        "host_inner_identity": True,
        "host_proc_meshes_empty": True,
        "host_proc_meshes_type": "builtins.list",
        "proc_host_identity": True,
    },
}


class _Namespace(SimpleNamespace):
    pass


def _fake_signature(touchpoint: MonarchTouchpoint) -> inspect.Signature:
    parameters = [
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in touchpoint.positional
    ]
    if touchpoint.var_positional:
        parameters.append(inspect.Parameter("args", inspect.Parameter.VAR_POSITIONAL))
    parameters.extend(
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
        for name in touchpoint.keywords
    )
    if touchpoint.var_keyword:
        parameters.append(inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(parameters)


def _fake_callable(touchpoint: MonarchTouchpoint):
    def value(*args, **kwargs):
        del args, kwargs

    value.__signature__ = _fake_signature(touchpoint)
    return value


def _fake_modules() -> dict[str, object]:
    modules = {touchpoint.module: _Namespace() for touchpoint in TOUCHPOINTS}
    for touchpoint in sorted(TOUCHPOINTS, key=lambda item: item.attribute.count(".")):
        parent = modules[touchpoint.module]
        parts = touchpoint.attribute.split(".")
        for part in parts[:-1]:
            if not hasattr(parent, part):
                setattr(parent, part, _Namespace())
            parent = getattr(parent, part)
        leaf = parts[-1]
        if hasattr(parent, leaf):
            continue
        if touchpoint.require_class:
            value = type(leaf, (), {})
            if touchpoint.callable:
                value.__signature__ = _fake_signature(touchpoint)
        elif touchpoint.callable:
            value = _fake_callable(touchpoint)
        else:
            value = object()
        setattr(parent, leaf, value)
    return modules


def _resolve(root: object, attribute: str) -> object:
    value = root
    for part in attribute.split("."):
        value = getattr(value, part)
    return value


def _delete(root: object, attribute: str) -> None:
    parts = attribute.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    delattr(parent, parts[-1])


def _snapshot(version: str = "0.5.0") -> dict:
    modules = _fake_modules()
    snapshot = collect_snapshot(
        import_module=modules.__getitem__,
        distribution_version=lambda _name: version,
        include_semantics=False,
    )
    snapshot["semantics"] = copy.deepcopy(PIN_SEMANTICS)
    return snapshot


def test_manifest_is_unique_nonvacuous_and_covers_every_required_family():
    paths = [touchpoint.path for touchpoint in TOUCHPOINTS]
    assert len(paths) == len(set(paths))
    assert len(paths) >= 40
    assert {
        "monarch.actor.ActorError",
        "monarch.actor.this_host",
        "monarch.actor.attach_to_workers",
        "monarch.actor.HostMesh.spawn_procs",
        "monarch.actor.ProcMesh.spawn",
        "monarch._src.actor.host_mesh.HostMesh",
        "monarch._src.actor.host_mesh.HostMesh.region",
        "monarch._src.actor.host_mesh.HostMesh.stream_logs",
        "monarch._src.actor.host_mesh.HostMesh.is_fake_in_process",
        "monarch._src.actor.proc_mesh.ProcMesh",
        "monarch._src.actor.proc_mesh.get_active_proc_meshes",
        "monarch._src.actor.proc_mesh.register_proc_mesh_spawn_callback",
        "monarch._src.actor.proc_mesh.unregister_proc_mesh_spawn_callback",
        "monarch.actor.Endpoint.call",
        "monarch.actor.Endpoint.call_one",
        "monarch.actor.Endpoint.broadcast",
        "monarch.actor.Future.get",
        "monarch.actor.ValueMesh.items",
        "monarch.actor.Channel.open",
        "monarch.actor.Port.send",
        "monarch.actor.PortReceiver.recv",
        "monarch.rdma.RDMABuffer.read_into",
        "monarch.rdma.RDMABuffer.drop",
        "monarch.rdma.is_ibverbs_available",
        "monarch.actor.unhandled_fault_hook",
        "monarch._rust_bindings.monarch_hyperactor.pytokio.PythonTask.__await__",
    } <= set(paths)


def test_manifest_helper_has_only_standard_library_imports():
    imported_roots = set()
    for filename in (
        "monarch_semantics.py",
        "monarch_surface.py",
        "monarch_surface_report.py",
        "monarch_touchpoint_spec.py",
    ):
        source = (REPO / "src" / "dgx_monarch" / filename).read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported_roots.add(node.module.split(".", 1)[0])
    assert not imported_roots - sys.stdlib_module_names


def test_complete_fake_surface_is_compatible():
    modules = _fake_modules()
    observed = collect_touchpoints(modules.__getitem__)
    assert set(observed) == {touchpoint.path for touchpoint in TOUCHPOINTS}
    assert all(record["status"] == "ok" for record in observed.values())


@pytest.mark.parametrize("distribution", [None, "monarch", ""])
def test_snapshot_rejects_missing_or_wrong_distribution(distribution):
    snapshot = _snapshot()
    snapshot["distribution"] = distribution
    assert any("distribution" in blocker for blocker in snapshot_blockers(snapshot))


@pytest.mark.parametrize(
    "version",
    [None, "", "   ", 500, "unavailable (PackageNotFoundError: absent metadata)"],
)
def test_snapshot_rejects_untrustworthy_distribution_version(version):
    snapshot = _snapshot()
    snapshot["version"] = version
    assert any("trustworthy" in blocker for blocker in snapshot_blockers(snapshot))


@pytest.mark.parametrize("touchpoint", TOUCHPOINTS, ids=lambda item: item.path)
def test_deleting_any_touchpoint_names_that_surface(touchpoint: MonarchTouchpoint):
    modules = _fake_modules()
    _delete(modules[touchpoint.module], touchpoint.attribute)
    record = collect_touchpoints(modules.__getitem__)[touchpoint.path]
    assert record["status"] != "ok"
    assert record["detail"]


@pytest.mark.parametrize(
    "touchpoint",
    [touchpoint for touchpoint in TOUCHPOINTS if touchpoint.callable],
    ids=lambda item: item.path,
)
def test_incompatible_callable_shape_names_that_surface(touchpoint: MonarchTouchpoint):
    modules = _fake_modules()
    parent = modules[touchpoint.module]
    parts = touchpoint.attribute.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    leaf = parts[-1]
    if touchpoint.positional:
        incompatible = MonarchTouchpoint(
            touchpoint.module,
            touchpoint.attribute,
            ("wrong", *touchpoint.positional[1:]),
            touchpoint.keywords,
            callable=True,
            require_class=touchpoint.require_class,
            var_positional=touchpoint.var_positional,
            var_keyword=touchpoint.var_keyword,
        )
        signature = _fake_signature(incompatible)
        value = getattr(parent, leaf)
        value.__signature__ = signature
    elif touchpoint.keywords or touchpoint.var_positional or touchpoint.var_keyword:
        value = getattr(parent, leaf)
        value.__signature__ = inspect.Signature()
    else:
        setattr(parent, leaf, object())
    record = collect_touchpoints(modules.__getitem__)[touchpoint.path]
    assert record["status"] == "incompatible"
    assert record["detail"]


def test_comparator_reports_fake_missing_surface_as_blocker():
    baseline = _snapshot()
    candidate = copy.deepcopy(baseline)
    candidate["touchpoints"].pop("monarch.rdma.RDMABuffer.read_into")
    comparison = compare_snapshots(baseline, candidate)
    assert not comparison.compatible
    assert any("RDMABuffer.read_into" in blocker for blocker in comparison.blockers)


def test_comparator_reports_compatible_raw_signature_drift_without_blocking():
    baseline = _snapshot()
    candidate = copy.deepcopy(baseline)
    candidate["touchpoints"]["monarch.actor.this_host"]["signature"] = (
        "() -> monarch.actor.HostMesh"
    )
    comparison = compare_snapshots(baseline, candidate)
    assert comparison.compatible
    assert any("this_host" in change for change in comparison.changes)


def test_comparator_blocks_fake_semantic_drift_and_renders_report():
    baseline = _snapshot()
    candidate = copy.deepcopy(baseline)
    candidate["version"] = "0.6.0"
    candidate["semantics"]["public_future_get_in_asyncio"] = {
        "outcome": "raised",
        "exception_type": "builtins.RuntimeError",
        "message": "use await",
        "warnings": [],
    }
    comparison = compare_snapshots(baseline, candidate)
    report = render_comparison_markdown(baseline, candidate, comparison)
    assert not comparison.compatible
    assert any("semantic drift" in blocker for blocker in comparison.blockers)
    assert "REVIEW REQUIRED" in report
    assert "0.5.0" in report and "0.6.0" in report
    assert "never changes the repository pin" in report


def test_comparator_blocks_private_ownership_field_drift():
    baseline = _snapshot()
    candidate = copy.deepcopy(baseline)
    candidate["version"] = "0.6.1"
    candidate["semantics"]["mesh_factory_private_ownership"] = {
        "outcome": "probe_error",
        "exception_type": "builtins.AttributeError",
        "message": "ProcMesh has no attribute '_host_mesh'",
    }
    comparison = compare_snapshots(baseline, candidate)
    assert not comparison.compatible
    assert any(
        "semantic drift in mesh_factory_private_ownership" in blocker
        for blocker in comparison.blockers
    )


def test_checked_pin_snapshot_is_complete_and_matches_manifest():
    # A pin bump that does not regenerate this baseline fails here, instead of
    # shipping stale semantics under the new version number.
    from dgx_monarch import TORCHMONARCH_PIN

    path = (
        REPO / "tests" / "fixtures"
        / f"torchmonarch_pin_{TORCHMONARCH_PIN.replace('.', '_')}.json"
    )
    snapshot = json.loads(path.read_text())
    assert snapshot["schema"] == SNAPSHOT_SCHEMA
    assert snapshot["version"] == TORCHMONARCH_PIN
    assert not snapshot_blockers(snapshot)
    assert set(snapshot["touchpoints"]) == {
        touchpoint.path for touchpoint in TOUCHPOINTS
    }
    assert snapshot["semantics"] == PIN_SEMANTICS
