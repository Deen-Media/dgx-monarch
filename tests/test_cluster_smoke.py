"""Hermetic contracts for the reusable client-owned cluster smoke."""

from __future__ import annotations

import ast
import json
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

from dgx_monarch import mesh_setup
from dgx_monarch.cli import cluster_smoke as smoke
from dgx_monarch.mesh_setup import SetupVerdictError
from dgx_monarch.nodes.gate_provenance import GateProvenanceError

_VERDICT_TYPES = (GateProvenanceError, SetupVerdictError)


class _Status(Enum):
    RECYCLED = "recycled"
    ACTIVE_WORK = "active_work"


@dataclass(frozen=True)
class _Outcome:
    status: _Status


@dataclass(frozen=True)
class _Config:
    source: str
    world_size: int
    comfy_dir: str
    transport_security: str = "trusted_fabric"


@dataclass(frozen=True)
class _Topology:
    world: int
    dp: int
    ulysses: int = 1
    ring: int = 1
    cfg: int = 1
    fsdp: bool = False

    def validate(self) -> None:
        if self.world != self.dp * self.ulysses * self.ring * self.cfg:
            raise ValueError("invalid topology")


class _Attestor:
    def __init__(self, calls: list[str], control: dict[str, Any]) -> None:
        self.calls = calls
        self.control = control

    def __call__(self, boundary: str, handle: Any, token: Any) -> None:
        assert handle is self.control["primary"]
        assert token is self.control["token"]
        self.calls.append(f"attest:{boundary}")
        error = self.control.get(f"attest_{boundary}_error")
        if error is not None:
            raise error


class _Handle:
    def __init__(
        self,
        name: str,
        config: _Config,
        comfy_dir: Path,
        calls: list[str],
        rows: list[dict[str, Any]],
    ) -> None:
        self.name = name
        self.config = config
        self.comfy_dir = str(comfy_dir)
        self.world = config.world_size
        self.topology: _Topology | None = None
        self.teardown_complete = False
        self.replacement_blocked: str | None = None
        self.defunct = False
        self.setup_cleanup_state = None
        self.sample_leases: dict[object, object] = {}
        self.abandoned_sample_leases: dict[object, object] = {}
        self.rows = rows
        self.calls = calls
        self.recycle_status = _Status.RECYCLED
        self.recycle_confirms = True
        self.recycle_error: BaseException | None = None
        self.status_error: BaseException | None = None
        self.recycle_outcome: object | None = None

    def call_all(self, endpoint: str, *, timeout_s: float) -> list[dict[str, Any]]:
        assert endpoint == "status" and timeout_s == 60
        self.calls.append("status")
        if self.status_error is not None:
            raise self.status_error
        return self.rows

    def recycle_detailed(self) -> object:
        self.calls.append(f"recycle:{self.name}")
        if self.recycle_error is not None:
            raise self.recycle_error
        outcome = self.recycle_outcome if self.recycle_outcome is not None else _Outcome(self.recycle_status)
        if self.recycle_status is _Status.RECYCLED and self.recycle_confirms:
            self.teardown_complete = True
            self.replacement_blocked = None
        return outcome


def _topology_row(world: int) -> dict[str, object]:
    return {
        "ulysses": 1,
        "ring": 1,
        "cfg": 1,
        "dp": world,
        "fsdp": False,
    }


def _status_rows(world: int, digest: str) -> list[dict[str, Any]]:
    return [
        {
            "host": f"secret-host-{rank}",
            "rank": rank,
            "world": world,
            "topology": _topology_row(world),
            "nccl": True,
            "vram": {"free_gib": 1.0},
            "models": {},
            "event_seq": rank,
            "source_manifest_sha256": digest,
        }
        for rank in range(world)
    ]


@dataclass
class _Rig:
    path: Path
    comfy_dir: Path
    config: _Config
    primary: _Handle
    dependencies: smoke.ClusterSmokeDependencies
    calls: list[str]
    control: dict[str, Any]
    mesh_kwargs: dict[str, Any]


def _rig(tmp_path: Path, *, world: int = 2, configured_comfy: bool = False) -> _Rig:
    path = tmp_path / "cluster.toml"
    path.write_text("[cluster]\n")
    comfy_dir = tmp_path / "ComfyUI"
    comfy_dir.mkdir()
    digest = "a" * 64
    config = _Config(
        str(path.resolve()),
        world,
        str(comfy_dir) if configured_comfy else "",
    )
    calls: list[str] = []
    control: dict[str, Any] = {"token": object(), "source": digest}
    rows = _status_rows(world, digest)
    primary = _Handle("primary", config, comfy_dir, calls, rows)
    control.update({"primary": primary, "selected": primary})
    mesh_kwargs: dict[str, Any] = {}

    def load_config(actual: Path) -> _Config:
        calls.append("load_config")
        assert actual == path.resolve()
        error = control.get("config_error")
        if error is not None:
            raise error
        return config

    def select_mesh(**kwargs: Any) -> tuple[_Handle, bool]:
        calls.append("select_mesh")
        mesh_kwargs.update(kwargs)
        preflight = kwargs.get("mesh_preflight")
        if control.get("skip_preflight") is not True:
            assert callable(preflight)
            candidate = control.get("preflight_config", config)
            candidate_world = control.get("preflight_world", world)
            preflight(candidate, candidate_world)
        error = control.get("attach_error")
        if error is not None:
            raise error
        return primary, control.get("created_here", True) is True

    def make_topology(actual_world: int) -> _Topology:
        calls.append("make_topology")
        return _Topology(actual_world, actual_world)

    def ensure_setup(handle: _Handle, topology: _Topology) -> tuple[Any, Any]:
        calls.append("ensure_setup")
        assert handle is primary
        error = control.get("setup_error")
        if error is not None:
            raise error
        handle.topology = topology
        return control["selected"], control["token"]

    def source_digest() -> str:
        calls.append("source_digest")
        error = control.get("source_error")
        if error is not None:
            raise error
        return control["source"]

    dependencies = smoke.ClusterSmokeDependencies(
        load_config=load_config,
        select_mesh=select_mesh,
        make_topology=make_topology,
        ensure_setup=ensure_setup,
        attestor_factory=lambda: _Attestor(calls, control),
        source_digest=source_digest,
        recycle_outcome_type=_Outcome,
        recycle_status_type=_Status,
        recycled_status=_Status.RECYCLED,
        verdict_error_types=_VERDICT_TYPES,
    )
    return _Rig(
        path,
        comfy_dir,
        config,
        primary,
        dependencies,
        calls,
        control,
        mesh_kwargs,
    )


def test_happy_path_proves_dp_status_source_cohort_and_owned_cleanup(tmp_path):
    rig = _rig(tmp_path)

    result = smoke.run_cluster_smoke(
        rig.path,
        rig.comfy_dir,
        dependencies=rig.dependencies,
    )

    assert result == smoke.ClusterSmokeResult(2, "a" * 64, 2, 2, 1)
    assert rig.primary.topology == _Topology(world=2, dp=2)
    assert rig.calls == [
        "load_config",
        "select_mesh",
        "make_topology",
        "ensure_setup",
        "source_digest",
        "attest:pre",
        "status",
        "attest:post",
        "recycle:primary",
    ]
    assert rig.mesh_kwargs["config_path"] == str(rig.path.resolve())
    assert rig.mesh_kwargs["mode"] == "cluster"
    assert rig.mesh_kwargs["gpus_per_host"] == 0
    assert rig.mesh_kwargs["comfy_dir"] == str(rig.comfy_dir.resolve())
    assert callable(rig.mesh_kwargs["mesh_preflight"])
    assert rig.primary.teardown_complete

    encoded = json.dumps(result.as_dict(), sort_keys=True)
    assert '"result": "PASS"' in encoded
    assert '"topology": "dp2"' in encoded
    assert "secret-host" not in encoded
    assert str(tmp_path) not in encoded
    assert "setup_generation" not in encoded and "token" not in encoded


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("world", True),
        ("data_parallel", 1),
        ("nccl", False),
        ("status_rank_coverage", True),
        ("source_rank_coverage", "2"),
        ("source_manifest_sha256", None),
        ("source_cohort_boundaries", 1),
        ("owned_procmeshes", 0),
        ("teardown_confirmed", False),
    ],
)
def test_exact_pass_evidence_rejects_malformed_or_incomplete_fields(field, value):
    payload = smoke.ClusterSmokeResult(2, "a" * 64, 2, 2, 1).as_dict()
    assert smoke.exact_pass_evidence(payload, 2, "a" * 64)
    payload[field] = value
    assert not smoke.exact_pass_evidence(payload, 2, "a" * 64)


def test_optional_comfy_dir_uses_and_validates_configured_binding(tmp_path):
    rig = _rig(tmp_path, configured_comfy=True)

    result = smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert result.world == 2
    assert rig.mesh_kwargs["comfy_dir"] == ""


def test_comfy_binding_mismatch_fails_and_still_recycles_owned_mesh(tmp_path):
    rig = _rig(tmp_path)
    other = tmp_path / "other-comfy"
    other.mkdir()
    rig.primary.comfy_dir = str(other)

    with pytest.raises(smoke.ClusterSmokeError, match="comfy_binding_mismatch"):
        smoke.run_cluster_smoke(
            rig.path,
            rig.comfy_dir,
            dependencies=rig.dependencies,
        )

    assert rig.calls[-1] == "recycle:primary"


def test_preexisting_cached_mesh_is_neither_claimed_nor_recycled(tmp_path):
    rig = _rig(tmp_path)
    rig.control["created_here"] = False

    with pytest.raises(smoke.ClusterSmokeError, match="mesh_not_created_here"):
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert rig.calls == ["load_config", "select_mesh"]
    assert rig.primary.topology is None
    assert rig.primary.teardown_complete is False


def test_setup_replacement_is_claimed_refused_and_recycled_newest_first(tmp_path):
    rig = _rig(tmp_path)
    replacement = _Handle(
        "replacement",
        rig.config,
        rig.comfy_dir,
        rig.calls,
        _status_rows(2, "a" * 64),
    )
    rig.control["selected"] = replacement

    with pytest.raises(smoke.ClusterSmokeError, match="setup_replaced_mesh_handle"):
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert rig.calls[-2:] == ["recycle:replacement", "recycle:primary"]
    assert replacement.teardown_complete and rig.primary.teardown_complete
    assert "status" not in rig.calls


def test_cleanup_attempts_every_claimed_handle_after_untyped_newest_outcome(tmp_path):
    rig = _rig(tmp_path)
    replacement = _Handle("replacement", rig.config, rig.comfy_dir, rig.calls, _status_rows(2, "a" * 64))
    replacement.recycle_outcome = object()
    rig.control["selected"] = replacement

    with pytest.raises(smoke.ClusterSmokeError, match="cleanup_outcome_untyped"):
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert rig.calls[-2:] == ["recycle:replacement", "recycle:primary"]
    assert rig.primary.teardown_complete


def test_cleanup_preserves_cancellation_but_still_attempts_older_claim(tmp_path):
    rig = _rig(tmp_path)
    replacement = _Handle("replacement", rig.config, rig.comfy_dir, rig.calls, _status_rows(2, "a" * 64))
    interrupted = KeyboardInterrupt("stop")
    replacement.recycle_error = interrupted
    rig.control["selected"] = replacement

    with pytest.raises(KeyboardInterrupt) as raised:
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert raised.value is interrupted
    assert rig.calls[-2:] == ["recycle:replacement", "recycle:primary"]


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda rows: rows.__setitem__(1, dict(rows[0])), "status_rank_invalid"),
        (lambda rows: rows[0].__setitem__("world", 3), "status_world_invalid"),
        (lambda rows: rows[0].__setitem__("topology", {}), "status_topology_invalid"),
        (lambda rows: rows[0].__setitem__("nccl", False), "status_nccl_invalid"),
        (lambda rows: rows[0].__setitem__("vram", None), "status_nccl_invalid"),
        (lambda rows: rows[0].__setitem__("vram_error", "hidden"), "status_nccl_invalid"),
        (lambda rows: rows[0].__setitem__("models", []), "status_models_invalid"),
        (lambda rows: rows[0].__setitem__("event_seq", -1), "status_event_sequence_invalid"),
        (lambda rows: rows[0].__setitem__("source_manifest_sha256", "b" * 64), "status_source_mismatch"),
    ],
)
def test_status_rows_fail_closed_and_cleanup(tmp_path, mutation, code):
    rig = _rig(tmp_path)
    mutation(rig.primary.rows)

    with pytest.raises(smoke.ClusterSmokeError, match=code):
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert rig.calls[-1] == "recycle:primary"


def test_status_requires_complete_world_coverage(tmp_path):
    rig = _rig(tmp_path)
    rig.primary.rows.pop()

    with pytest.raises(smoke.ClusterSmokeError, match="status_coverage_incomplete"):
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert rig.primary.teardown_complete


@pytest.mark.parametrize("boundary", ["pre", "post"])
def test_an_untyped_attestor_fault_stays_unknown_and_cleanup_remains_mandatory(
    tmp_path, boundary
):
    rig = _rig(tmp_path)
    rig.control[f"attest_{boundary}_error"] = RuntimeError("private detail")

    with pytest.raises(smoke.ClusterSmokeError, match=f"provenance_{boundary}_unknown") as raised:
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert raised.value.unknown is True
    assert "private detail" not in str(raised.value)
    assert rig.calls[-1] == "recycle:primary"


@pytest.mark.parametrize("boundary", ["pre", "post"])
def test_a_decided_provenance_mismatch_is_a_definite_failure(tmp_path, boundary):
    """The mismatch the attestor decided keeps the authority a verdict carries.

    It is deterministic and it is the target release's fault, so the verified
    update may restore the prior release over it. The attestor's own RPC is the
    reason this cannot be read off the phase alone: a lost
    `provenance_baseline` arrives from the same call.
    """
    rig = _rig(tmp_path)
    rig.control[f"attest_{boundary}_error"] = GateProvenanceError(
        f"Gate {boundary} provenance package source does not match driver"
    )

    with pytest.raises(smoke.ClusterSmokeError, match=f"provenance_{boundary}_failed") as raised:
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert raised.value.unknown is False
    assert "does not match driver" not in str(raised.value)
    assert rig.calls[-1] == "recycle:primary"


@pytest.mark.parametrize("boundary", ["pre", "post"])
def test_a_lost_provenance_baseline_stays_unknown(tmp_path, boundary):
    """The ten minute RPC in the same call proves nothing when it times out."""
    rig = _rig(tmp_path)
    rig.control[f"attest_{boundary}_error"] = TimeoutError("provenance_baseline timed out")

    with pytest.raises(smoke.ClusterSmokeError, match=f"provenance_{boundary}_unknown") as raised:
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert raised.value.unknown is True
    assert "timed out" not in raised.value.code
    assert rig.calls[-1] == "recycle:primary"


def test_a_decided_setup_verdict_is_definite_and_a_lost_setup_is_not(tmp_path):
    """Same pair one phase earlier, where `ensure_setup` is the surface."""
    rig = _rig(tmp_path)
    rig.control["setup_error"] = SetupVerdictError("distributed setup has not completed")

    with pytest.raises(smoke.ClusterSmokeError, match="setup_failed") as decided:
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert decided.value.unknown is False
    assert "distributed setup" not in str(decided.value)
    assert rig.calls[-1] == "recycle:primary"

    (tmp_path / "lost").mkdir()
    lost = _rig(tmp_path / "lost")
    lost.control["setup_error"] = TimeoutError("rank setup never answered")

    with pytest.raises(smoke.ClusterSmokeError, match="setup_unknown") as unknown:
        smoke.run_cluster_smoke(lost.path, dependencies=lost.dependencies)

    assert unknown.value.unknown is True
    assert lost.calls[-1] == "recycle:primary"


@pytest.mark.parametrize(
    ("phase", "error", "code", "exit_code"),
    [
        ("provenance_pre", GateProvenanceError("mismatch"), "provenance_pre_failed", 1),
        ("provenance_pre", TimeoutError("lost"), "provenance_pre_unknown", 75),
        ("provenance_post", GateProvenanceError("mismatch"), "provenance_post_failed", 1),
        ("provenance_post", TimeoutError("lost"), "provenance_post_unknown", 75),
        ("setup", SetupVerdictError("no setup"), "setup_failed", 1),
        ("setup", TimeoutError("lost"), "setup_unknown", 75),
    ],
)
def test_smoke_main_exit_codes_separate_a_verdict_from_a_lost_run(
    tmp_path, monkeypatch, capsys, phase, error, code, exit_code
):
    """What the verified update reads: 1 restores, 75 stops fail-closed."""
    rig = _rig(tmp_path)
    if phase == "setup":
        rig.control["setup_error"] = error
    else:
        rig.control[f"attest_{phase.removeprefix('provenance_')}_error"] = error
    original = smoke.run_cluster_smoke
    monkeypatch.setattr(
        smoke,
        "run_cluster_smoke",
        lambda config_path, **_kwargs: original(config_path, dependencies=rig.dependencies),
    )

    assert smoke.smoke_main(str(rig.path)) == exit_code
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {
        "result": "FAIL",
        "error": code,
    }


def test_the_real_setup_dependency_decides_or_leaves_the_outcome_unknown(tmp_path):
    """`ensure_setup_token` is what the production dependency calls."""

    class _SetupHandle:
        def __init__(self, error: BaseException | None = None) -> None:
            self.lock = threading.RLock()
            self.setup_cleanup_state = None
            self.setup_key: tuple | None = None
            self.worker_args_key: tuple | None = None
            self.setup_generation = 7
            self.error = error

        def ensure_setup(self, *_args: Any, **_kwargs: Any) -> None:
            if self.error is not None:
                raise self.error

    ready = _SetupHandle()
    ready.setup_key = ("dp2",)
    ready.worker_args_key = ()
    assert mesh_setup.ensure_setup_token(ready).generation == 7

    with pytest.raises(SetupVerdictError):
        mesh_setup.ensure_setup_token(_SetupHandle())

    lost = _SetupHandle(TimeoutError("rank setup never answered"))
    with pytest.raises(TimeoutError):
        mesh_setup.ensure_setup_token(lost)


def test_cleanup_non_success_overrides_body_failure_and_is_exposed(tmp_path):
    rig = _rig(tmp_path)
    rig.primary.rows = []
    rig.primary.recycle_status = _Status.ACTIVE_WORK

    with pytest.raises(smoke.ClusterSmokeError, match="cleanup_not_confirmed") as raised:
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert any("body also failed" in note for note in raised.value.__notes__)
    assert rig.calls.count("recycle:primary") == 1


@pytest.mark.parametrize(
    ("configure", "code"),
    [
        (lambda handle: setattr(handle, "recycle_error", RuntimeError("secret")), "cleanup_recycle_failed"),
        (lambda handle: setattr(handle, "recycle_outcome", object()), "cleanup_outcome_untyped"),
        (lambda handle: setattr(handle, "recycle_confirms", False), "cleanup_not_confirmed"),
    ],
)
def test_cleanup_requires_typed_confirmed_teardown(tmp_path, configure, code):
    rig = _rig(tmp_path)
    configure(rig.primary)

    with pytest.raises(smoke.ClusterSmokeError, match=code):
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert rig.calls.count("recycle:primary") == 1


def test_body_cancellation_identity_survives_successful_cleanup(tmp_path):
    rig = _rig(tmp_path)
    interrupted = KeyboardInterrupt("cancelled")
    rig.control["source_error"] = interrupted

    with pytest.raises(KeyboardInterrupt) as raised:
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert raised.value is interrupted
    assert rig.primary.teardown_complete


def test_cleanup_cancellation_identity_outranks_ordinary_body_failure(tmp_path):
    rig = _rig(tmp_path)
    interrupted = KeyboardInterrupt("cleanup cancelled")
    rig.primary.rows = []
    rig.primary.recycle_error = interrupted

    with pytest.raises(KeyboardInterrupt) as raised:
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert raised.value is interrupted
    assert any("body also failed" in note for note in interrupted.__notes__)


@pytest.mark.parametrize(
    ("attribute", "value", "code"),
    [
        ("transport_security", "", "trusted_fabric_required"),
        ("world_size", 0, "config_world_invalid"),
        ("source", "/definitely/missing", "config_path_invalid"),
    ],
)
def test_config_contract_refuses_before_attach(tmp_path, attribute, value, code):
    rig = _rig(tmp_path)
    object.__setattr__(rig.config, attribute, value)

    with pytest.raises(smoke.ClusterSmokeError, match=code):
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert "select_mesh" not in rig.calls
    assert not rig.primary.teardown_complete


@pytest.mark.parametrize(
    ("control_key", "control_value", "code"),
    [
        ("skip_preflight", True, "mesh_preflight_missing"),
        ("preflight_world", 3, "config_changed_during_attach"),
        ("preflight_config", object(), "config_changed_during_attach"),
    ],
)
def test_attach_requires_exact_single_config_preflight(tmp_path, control_key, control_value, code):
    rig = _rig(tmp_path)
    rig.control[control_key] = control_value

    with pytest.raises(smoke.ClusterSmokeError, match=code):
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    if control_key == "skip_preflight":
        assert rig.primary.teardown_complete
    else:
        assert not rig.primary.teardown_complete


def test_public_error_is_structured_and_disclosure_safe():
    error = smoke.ClusterSmokeError("status_failed")

    assert error.as_dict() == {"result": "FAIL", "error": "status_failed"}


def test_module_is_lazy_source_only_and_never_stops_attached_hosts():
    source_path = Path(smoke.__file__ or "")
    source = source_path.read_text()
    tree = ast.parse(source)
    top_imports = {node.module for node in tree.body if isinstance(node, ast.ImportFrom)}

    assert top_imports <= {
        "__future__",
        "collections.abc",
        "dataclasses",
        "pathlib",
        "typing",
    }
    enforce = source.index("runtime_provenance.enforce_source_only_imports()")
    assert enforce < source.index("from ..mesh import get_mesh")
    assert "hosts.shutdown" not in source
    assert ".recycle_detailed()" in source
    assert "mesh_setup.ensure_setup_token" in source
    assert "ProofCohortAttestor" in source
    assert 'handle.call_all("status", timeout_s=60)' in source
    # The production wiring of the two verdict types, which only a real
    # dependency build can reach.
    assert (
        "verdict_error_types=(GateProvenanceError, mesh_setup.SetupVerdictError)"
        in source
    )


def _bare_runtime_error_raises(module: str) -> dict[str, int]:
    """Every `raise RuntimeError(...)` in one module, counted by function."""
    source_root = Path(smoke.__file__ or "").resolve().parents[1]
    tree = ast.parse((source_root / module).read_text())
    counts: dict[str, int] = {}
    stack: list[str] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack.append(node.name)
            for child in ast.iter_child_nodes(node):
                visit(child)
            stack.pop()
            return
        if isinstance(node, ast.Raise) and node.exc is not None:
            call = node.exc if isinstance(node.exc, ast.Call) else None
            target = call.func if call is not None else node.exc
            if getattr(target, "id", None) == "RuntimeError":
                qual = ".".join(stack) or "<module>"
                counts[qual] = counts.get(qual, 0) + 1
        for child in ast.iter_child_nodes(node):
            visit(child)

    for node in tree.body:
        visit(node)
    return counts


def test_a_new_verdict_on_either_surface_cannot_arrive_as_a_bare_runtime_error():
    """The two surfaces the smoke grades by exception type, pinned.

    A verdict that arrives as a bare `RuntimeError` is indistinguishable from
    the transport faults raised beside it, so this smoke would grade it
    unknown and the verified update would stop instead of restoring. The
    attestor is at zero and stays there. The `mesh_setup` raises left untyped
    all refuse a dispatch rather than decide a setup outcome, so no smoke
    phase can reach one as a verdict; grading them unknown is the safe
    direction. A count may fall (lower it here); a new raise fails here.
    """
    assert _bare_runtime_error_raises("nodes/gate_provenance.py") == {}
    assert _bare_runtime_error_raises("mesh_setup.py") == {
        "require_endpoint": 3,
        "dispatch_endpoint": 2,
        "_dispatch_sample_bound": 3,
    }


def test_a_lost_status_call_is_an_unknown_refusal_not_a_verdict(tmp_path):
    """A 60s RPC timeout is not the cluster failing its smoke.

    A raw fault reaching `_normalize_failure` that is not a typed verdict is
    one nobody decided, so it carries the unknown class and its phase's
    `_unknown` code. The smoke's own refusals arrive typed and stay definite.
    """
    rig = _rig(tmp_path)
    rig.primary.status_error = TimeoutError("status RPC timed out")

    with pytest.raises(smoke.ClusterSmokeError, match="status_unknown") as raised:
        smoke.run_cluster_smoke(rig.path, dependencies=rig.dependencies)

    assert raised.value.unknown is True
    assert "timed out" not in raised.value.code
    assert rig.primary.teardown_complete

    # A row the smoke itself refused stays a definite FAIL.
    (tmp_path / "second").mkdir()
    clean = _rig(tmp_path / "second")
    clean.primary.rows[0]["nccl"] = False
    with pytest.raises(smoke.ClusterSmokeError, match="status_nccl_invalid") as typed:
        smoke.run_cluster_smoke(clean.path, dependencies=clean.dependencies)
    assert typed.value.unknown is False


def test_smoke_main_reports_an_unproven_run_with_the_unknown_exit(tmp_path, capsys):
    """The wrapper the verified update runs, and the only thing it can read."""
    rig = _rig(tmp_path)
    rig.primary.status_error = TimeoutError("status RPC timed out")
    original = smoke.run_cluster_smoke

    def run(config_path, **_kwargs):
        return original(config_path, dependencies=rig.dependencies)

    smoke.run_cluster_smoke = run
    try:
        assert smoke.smoke_main(str(rig.path)) == 75
        payload = json.loads(capsys.readouterr().out.splitlines()[-1])
        assert payload == {"result": "FAIL", "error": "status_unknown"}

        (tmp_path / "second").mkdir()
        refused = _rig(tmp_path / "second")
        refused.primary.rows[0]["nccl"] = False

        def run_refused(config_path, **_kwargs):
            return original(config_path, dependencies=refused.dependencies)

        smoke.run_cluster_smoke = run_refused
        assert smoke.smoke_main(str(refused.path)) == 1
        assert json.loads(capsys.readouterr().out.splitlines()[-1])["error"] == (
            "status_nccl_invalid")
    finally:
        smoke.run_cluster_smoke = original
