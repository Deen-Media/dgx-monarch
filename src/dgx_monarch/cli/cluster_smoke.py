"""Check cluster setup and source identity using only smoke-owned meshes.

Accelerator, Monarch and worker-runtime imports occur only when
:func:`run_cluster_smoke` builds its default dependencies, keeping this module
importable by CPU-only tools.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

_DIGEST_LENGTH = 64
_ATTENTION = "TORCH_FLASH"


class ClusterSmokeError(RuntimeError):
    """A disclosure-safe smoke refusal or unknown outcome.

    ``unknown`` distinguishes a failed check from a run that lost its evidence,
    such as through transport failure. Only a definite refusal can authorize
    rollback.
    """

    def __init__(self, code: str, *, unknown: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.unknown = unknown

    def as_dict(self) -> dict[str, object]:
        return {"result": "FAIL", "error": self.code}


@dataclass(frozen=True)
class ClusterSmokeResult:
    world: int
    source_digest: str
    status_rank_coverage: int
    source_rank_coverage: int
    owned_procmeshes: int

    def as_dict(self) -> dict[str, object]:
        topology = "single" if self.world == 1 else f"dp{self.world}"
        return {
            "result": "PASS",
            "world": self.world,
            "topology": topology,
            "data_parallel": self.world,
            "nccl": True,
            "status_rank_coverage": self.status_rank_coverage,
            "source_manifest_sha256": self.source_digest,
            "source_rank_coverage": self.source_rank_coverage,
            "source_cohort_boundaries": 2,
            "owned_procmeshes": self.owned_procmeshes,
            "teardown_confirmed": True,
        }


def exact_pass_evidence(payload: object, expected_world: int, source_digest: str) -> bool:
    """Validate the complete PASS fields consumed by verified update."""
    return (
        isinstance(payload, dict)
        and payload.get("result") == "PASS"
        and type(payload.get("world")) is int
        and payload["world"] == expected_world
        and type(payload.get("data_parallel")) is int
        and payload["data_parallel"] == expected_world
        and payload.get("nccl") is True
        and type(payload.get("status_rank_coverage")) is int
        and payload["status_rank_coverage"] == expected_world
        and type(payload.get("source_rank_coverage")) is int
        and payload["source_rank_coverage"] == expected_world
        and type(payload.get("source_manifest_sha256")) is str
        and payload["source_manifest_sha256"] == source_digest
        and type(payload.get("source_cohort_boundaries")) is int
        and payload["source_cohort_boundaries"] == 2
        and type(payload.get("owned_procmeshes")) is int
        and payload["owned_procmeshes"] > 0
        and payload.get("teardown_confirmed") is True
    )


@dataclass(frozen=True)
class ClusterSmokeDependencies:
    """Injectable boundary around accelerator and lifecycle APIs."""

    load_config: Callable[[Path], Any]
    select_mesh: Callable[..., tuple[Any, bool]]
    make_topology: Callable[[int], Any]
    ensure_setup: Callable[[Any, Any], tuple[Any, Any]]
    attestor_factory: Callable[[], Any]
    source_digest: Callable[[], str]
    recycle_outcome_type: type[Any]
    recycle_status_type: type[Any]
    recycled_status: Any
    # Inject setup and source-verification result types to preserve the
    # required lazy import order.
    verdict_error_types: tuple[type[BaseException], ...]


def _fail(code: str) -> NoReturn:
    raise ClusterSmokeError(code)


def _safe_note(primary: BaseException, label: str) -> None:
    try:
        primary.add_note(label)
    except BaseException:
        pass


def _canonical_file(value: str | Path) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, Path)):
        _fail("config_path_invalid")
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        _fail("config_path_invalid")
    if not path.is_file():
        _fail("config_path_invalid")
    return path


def _canonical_directory(value: object, code: str) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, Path)) or not str(value):
        _fail(code)
    try:
        path = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        _fail(code)
    if not path.is_dir():
        _fail(code)
    return path


def _canonical_digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail("source_digest_invalid")
    return value


def _validate_topology(topology: Any, world: int) -> dict[str, object]:
    expected: dict[str, object] = {
        "ulysses": 1,
        "ring": 1,
        "cfg": 1,
        "dp": world,
        "fsdp": False,
    }
    for name, value in expected.items():
        actual = getattr(topology, name, None)
        if type(value) is int:
            if type(actual) is not int or actual != value:
                _fail("data_parallel_topology_invalid")
        elif actual is not value:
            _fail("data_parallel_topology_invalid")
    topology_world = getattr(topology, "world", None)
    if type(topology_world) is not int or topology_world != world:
        _fail("data_parallel_topology_invalid")
    validator = getattr(topology, "validate", None)
    if not callable(validator):
        _fail("data_parallel_topology_invalid")
    validator()
    return expected


def _validate_status_rows(
    rows: object,
    *,
    world: int,
    topology: dict[str, object],
    source_digest: str,
) -> int:
    if not isinstance(rows, list) or len(rows) != world:
        _fail("status_coverage_incomplete")
    ranks: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            _fail("status_row_invalid")
        rank = row.get("rank")
        if type(rank) is not int or not 0 <= rank < world or rank in ranks:
            _fail("status_rank_invalid")
        if type(row.get("world")) is not int or row["world"] != world:
            _fail("status_world_invalid")
        row_topology = row.get("topology")
        if not isinstance(row_topology, dict) or set(row_topology) != set(topology):
            _fail("status_topology_invalid")
        for name, value in topology.items():
            actual = row_topology.get(name)
            if type(value) is int:
                if type(actual) is not int or actual != value:
                    _fail("status_topology_invalid")
            elif actual is not value:
                _fail("status_topology_invalid")
        if row.get("nccl") is not True or not isinstance(row.get("vram"), dict) or "vram_error" in row:
            _fail("status_nccl_invalid")
        if not isinstance(row.get("models"), dict):
            _fail("status_models_invalid")
        event_seq = row.get("event_seq")
        if type(event_seq) is not int or event_seq < 0:
            _fail("status_event_sequence_invalid")
        if row.get("source_manifest_sha256") != source_digest:
            _fail("status_source_mismatch")
        ranks.add(rank)
    if ranks != set(range(world)):
        _fail("status_coverage_incomplete")
    return len(ranks)


class _OwnedProcMeshes:
    """Track only this smoke run's ProcMeshes, deduplicated by object identity."""

    def __init__(self) -> None:
        self._handles: list[Any] = []
        self._attempted: set[int] = set()

    @property
    def count(self) -> int:
        return len(self._handles)

    def claim(self, handle: Any) -> None:
        if handle is None:
            _fail("mesh_handle_missing")
        if all(existing is not handle for existing in self._handles):
            self._handles.append(handle)

    def cleanup(self, dependencies: ClusterSmokeDependencies) -> tuple[int, BaseException | None]:
        confirmed = 0
        failure: BaseException | None = None
        for handle in reversed(self._handles):
            candidate: BaseException | None = None
            if getattr(handle, "teardown_complete", None) is True:
                confirmed += 1
                continue
            identity = id(handle)
            if identity in self._attempted:
                candidate = ClusterSmokeError("cleanup_attempt_repeated")
            else:
                self._attempted.add(identity)
                try:
                    outcome = handle.recycle_detailed()
                except BaseException as exc:
                    candidate = exc if not isinstance(exc, Exception) else ClusterSmokeError("cleanup_recycle_failed")
                else:
                    if not isinstance(outcome, dependencies.recycle_outcome_type):
                        candidate = ClusterSmokeError("cleanup_outcome_untyped")
                    else:
                        status = getattr(outcome, "status", None)
                        if (
                            not isinstance(status, dependencies.recycle_status_type)
                            or status is not dependencies.recycled_status
                            or getattr(handle, "teardown_complete", None) is not True
                            or getattr(handle, "replacement_blocked", None) is not None
                        ):
                            candidate = ClusterSmokeError("cleanup_not_confirmed")
                        else:
                            confirmed += 1
            if candidate is not None:
                if failure is None:
                    failure = candidate
                elif isinstance(failure, Exception) and not isinstance(candidate, Exception):
                    _safe_note(candidate, "another ProcMesh cleanup also failed")
                    failure = candidate
                else:
                    _safe_note(failure, "another ProcMesh cleanup also failed")
        return confirmed, failure


def _default_dependencies() -> ClusterSmokeDependencies:
    # Source-only enforcement must precede every operational mesh/Comfy import.
    from .. import runtime_provenance

    runtime_provenance.enforce_source_only_imports()

    from .. import mesh_setup
    from ..config import load_cluster_config
    from ..mesh import get_mesh
    from ..mesh_recycle import RecycleOutcome, RecycleStatus
    from ..nodes.gate_provenance import GateProvenanceError, ProofCohortAttestor
    from ..topology import Topology

    def ensure_setup(handle: Any, topology: Any) -> tuple[Any, Any]:
        token = mesh_setup.ensure_setup_token(
            handle,
            topology,
            _ATTENTION,
            True,
            {},
        )
        return handle, token

    def select_mesh(**kwargs: Any) -> tuple[Any, bool]:
        claim = object()
        handle = get_mesh(**kwargs, _selection_claim=claim)
        return handle, getattr(handle, "_selection_claim", None) is claim

    return ClusterSmokeDependencies(
        load_config=load_cluster_config,
        select_mesh=select_mesh,
        make_topology=lambda world: Topology(dp=world, world=world),
        ensure_setup=ensure_setup,
        attestor_factory=ProofCohortAttestor,
        source_digest=runtime_provenance.cached_dgx_source_manifest_sha256,
        recycle_outcome_type=RecycleOutcome,
        recycle_status_type=RecycleStatus,
        recycled_status=RecycleStatus.RECYCLED,
        verdict_error_types=(GateProvenanceError, mesh_setup.SetupVerdictError),
    )


def _normalize_failure(
    error: BaseException,
    phase: str,
    verdict_types: tuple[type[BaseException], ...] = (),
) -> BaseException:
    """Map a failure to a disclosure-safe phase code while preserving certainty.

    Typed verdicts retain the called surface's definite outcome and rollback
    authority. Untyped faults, including transport losses and RPC timeouts, become
    unknown because they do not establish cluster failure. The smoke's own
    refusals are already ``ClusterSmokeError`` instances.
    """
    if isinstance(error, ClusterSmokeError) or not isinstance(error, Exception):
        return error
    if verdict_types and isinstance(error, verdict_types):
        return ClusterSmokeError(f"{phase}_failed")
    return ClusterSmokeError(f"{phase}_unknown", unknown=True)


def smoke_main(config_path: str) -> int:
    """Print one smoke result for the update wrapper; exit 1 on a refusal, 75 when unknown."""
    import json

    from .probe_certainty import UNKNOWN_EXIT

    try:
        print(json.dumps(run_cluster_smoke(config_path).as_dict(), sort_keys=True))
    except ClusterSmokeError as exc:
        print(json.dumps(exc.as_dict(), sort_keys=True))
        return UNKNOWN_EXIT if exc.unknown else 1
    return 0


def _prefer_cleanup_failure(primary: BaseException | None, cleanup: BaseException) -> BaseException:
    if primary is None:
        return cleanup
    if not isinstance(primary, Exception):
        _safe_note(primary, "cluster smoke ProcMesh cleanup also failed")
        return primary
    if not isinstance(cleanup, Exception):
        _safe_note(cleanup, "cluster smoke body also failed")
        return cleanup
    _safe_note(cleanup, "cluster smoke body also failed")
    return cleanup


def run_cluster_smoke(
    config_path: str | Path,
    comfy_dir: str | Path | None = None,
    *,
    dependencies: ClusterSmokeDependencies | None = None,
) -> ClusterSmokeResult:
    """Attach, establish DP/NCCL, attest all ranks, then recycle owned procs."""
    owners = _OwnedProcMeshes()
    runtime = dependencies
    primary: BaseException | None = None
    evidence: tuple[int, str, int] | None = None
    phase = "input"
    try:
        path = _canonical_file(config_path)
        phase = "dependencies"
        runtime = _default_dependencies() if runtime is None else runtime
        phase = "config"
        config = runtime.load_config(path)
        source = _canonical_file(getattr(config, "source", ""))
        if source != path:
            _fail("config_source_mismatch")
        world = getattr(config, "world_size", None)
        if type(world) is not int or world < 1:
            _fail("config_world_invalid")
        if getattr(config, "transport_security", None) != "trusted_fabric":
            _fail("trusted_fabric_required")

        explicit_comfy = None if comfy_dir is None else _canonical_directory(comfy_dir, "comfy_dir_invalid")
        configured_comfy = getattr(config, "comfy_dir", "")
        expected_comfy = explicit_comfy
        if expected_comfy is None and configured_comfy:
            expected_comfy = _canonical_directory(configured_comfy, "configured_comfy_dir_invalid")

        preflight_calls = 0

        def mesh_preflight(candidate: Any, candidate_world: int) -> None:
            nonlocal preflight_calls
            preflight_calls += 1
            if preflight_calls != 1:
                _fail("mesh_preflight_repeated")
            if candidate != config or type(candidate_world) is not int or candidate_world != world:
                _fail("config_changed_during_attach")

        phase = "attach"
        selection = runtime.select_mesh(
            config_path=str(path),
            mode="cluster",
            gpus_per_host=0,
            comfy_dir=str(explicit_comfy) if explicit_comfy is not None else "",
            mesh_preflight=mesh_preflight,
        )
        if not isinstance(selection, tuple) or len(selection) != 2:
            _fail("mesh_selection_invalid")
        handle, created_here = selection
        if created_here is not True:
            _fail("mesh_not_created_here")
        owners.claim(handle)
        if preflight_calls != 1:
            _fail("mesh_preflight_missing")
        if getattr(handle, "config", None) != config:
            _fail("mesh_config_mismatch")
        handle_world = getattr(handle, "world", None)
        if type(handle_world) is not int or handle_world != world:
            _fail("mesh_world_mismatch")
        actual_comfy = _canonical_directory(getattr(handle, "comfy_dir", None), "comfy_binding_invalid")
        if expected_comfy is not None and actual_comfy != expected_comfy:
            _fail("comfy_binding_mismatch")

        topology = runtime.make_topology(world)
        topology_row = _validate_topology(topology, world)
        phase = "setup"
        setup_result = runtime.ensure_setup(handle, topology)
        if not isinstance(setup_result, tuple) or len(setup_result) != 2:
            _fail("setup_result_invalid")
        selected, setup_token = setup_result
        owners.claim(selected)
        if selected is not handle:
            _fail("setup_replaced_mesh_handle")
        if setup_token is None or getattr(handle, "topology", None) != topology:
            _fail("setup_identity_invalid")

        source_digest = _canonical_digest(runtime.source_digest())
        attestor = runtime.attestor_factory()
        phase = "provenance_pre"
        attestor("pre", handle, setup_token)
        phase = "status"
        rows = handle.call_all("status", timeout_s=60)
        status_coverage = _validate_status_rows(
            rows,
            world=world,
            topology=topology_row,
            source_digest=source_digest,
        )
        phase = "provenance_post"
        attestor("post", handle, setup_token)
        if (
            getattr(handle, "defunct", None) is not False
            or getattr(handle, "teardown_complete", None) is not False
            or getattr(handle, "replacement_blocked", None) is not None
            or getattr(handle, "setup_cleanup_state", None) is not None
            or getattr(handle, "sample_leases", None) != {}
            or getattr(handle, "abandoned_sample_leases", None) != {}
        ):
            _fail("mesh_not_settled")
        evidence = (world, source_digest, status_coverage)
    except BaseException as exc:
        verdicts = () if runtime is None else runtime.verdict_error_types
        primary = _normalize_failure(exc, phase, verdicts)

    confirmed = 0
    if runtime is not None:
        confirmed, cleanup_error = owners.cleanup(runtime)
        if cleanup_error is not None:
            primary = _prefer_cleanup_failure(primary, cleanup_error)
    if primary is not None:
        raise primary from None
    if evidence is None or confirmed != owners.count or owners.count != 1:
        _fail("cleanup_not_confirmed")
    world, source_digest, status_coverage = evidence
    return ClusterSmokeResult(
        world=world,
        source_digest=source_digest,
        status_rank_coverage=status_coverage,
        source_rank_coverage=world,
        owned_procmeshes=owners.count,
    )
