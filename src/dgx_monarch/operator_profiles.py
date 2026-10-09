"""Conservative operator-profile compilation for guided setup.

Profiles are an authoring convenience, not a new runtime policy surface.  A
profile resolves only to keys already accepted by ``cluster.toml`` plus graph
recommendations that the setup UI may present separately. The profile name
itself must never be persisted into the strict cluster config.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from .config_schema import ClusterConfigError, validate_worker_args

ProfileName: TypeAlias = Literal["safe", "balanced", "advanced"]
HardwareClass: TypeAlias = Literal[
    "homogeneous_uma", "discrete", "mixed", "unknown"
]
GraphValue: TypeAlias = str | int | bool

PROFILE_NAMES: tuple[ProfileName, ...] = ("safe", "balanced", "advanced")
HARDWARE_CLASSES: tuple[HardwareClass, ...] = (
    "homogeneous_uma",
    "discrete",
    "mixed",
    "unknown",
)

_RESIDENCY_KEYS = ("lora_low_rss", "slab_weights")
_FORBIDDEN_EMISSIONS = frozenset({
    "operator_profile",
    "comfy_managed",
    "compile_dit",
    "load_profile",
})
_GRAPH_RECOMMENDATIONS: dict[str, GraphValue] = {
    "topology": "auto",
    "auto_gate": "first_use",
    "pipeline_depth": 1,
}


@dataclass(frozen=True)
class HardwareObservation:
    """One probed accelerator cohort member.

    ``uma`` must be observed, never inferred from a model name. ``None`` means
    the probe could not establish the memory topology.
    A non-empty model name is also required before advanced mode can regard a
    UMA fleet as homogeneous.
    """

    model: str | None
    uma: bool | None


@dataclass(frozen=True)
class ProfileResolution:
    """Strict config edits and separate graph advice produced by a profile."""

    name: ProfileName
    hardware_class: HardwareClass
    cluster_set: dict[str, bool]
    worker_args_set: dict[str, bool]
    worker_args_unset: tuple[str, ...]
    graph_recommendations: dict[str, GraphValue]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a stable, JSON-ready representation for plans and receipts."""
        return {
            "name": self.name,
            "hardware_class": self.hardware_class,
            "cluster_set": dict(self.cluster_set),
            "worker_args_set": dict(self.worker_args_set),
            "worker_args_unset": list(self.worker_args_unset),
            "graph_recommendations": dict(self.graph_recommendations),
            "notes": list(self.notes),
        }


class ProfileRefusal(ValueError):
    """A requested profile cannot be safely compiled for observed hardware."""

    requested_profile: ProfileName
    hardware_class: HardwareClass
    recommended_profile: ProfileName

    def __init__(
        self,
        requested_profile: ProfileName,
        hardware_class: HardwareClass,
        reason: str,
        *,
        recommended_profile: ProfileName = "balanced",
    ) -> None:
        self.requested_profile = requested_profile
        self.hardware_class = hardware_class
        self.recommended_profile = recommended_profile
        self.reason = reason
        super().__init__(
            f"profile {requested_profile!r} is unavailable for {hardware_class}: "
            f"{reason}; use {recommended_profile!r}"
        )


def classify_hardware(observations: Sequence[HardwareObservation]) -> HardwareClass:
    """Classify probe results without treating missing evidence as capability."""
    if not observations:
        return "unknown"

    known_uma = {observation.uma for observation in observations if observation.uma is not None}
    if known_uma == {True, False}:
        return "mixed"
    if any(observation.uma is None for observation in observations):
        return "unknown"
    if known_uma == {False}:
        return "discrete"

    # Every member is positively identified as UMA at this point.  Advanced
    # residency is still refused unless every model is known and identical.
    models = tuple(
        observation.model.strip() if isinstance(observation.model, str) else ""
        for observation in observations
    )
    if not all(models):
        return "unknown"
    if len(set(models)) != 1:
        return "mixed"
    return "homogeneous_uma"


def resolve_profile(
    name: str = "balanced",
    *,
    hardware_class: str = "unknown",
) -> ProfileResolution:
    """Compile an operator profile into existing strict configuration keys.

    ``advanced`` is capability-gated: unknown, mixed or discrete hardware gets
    a typed refusal (``ProfileRefusal``), never an optimistic config. Native
    RDMA latent return stays off in every profile while that path is held.
    """
    if name not in PROFILE_NAMES:
        raise ValueError(
            f"unknown operator profile {name!r}; choose one of {', '.join(PROFILE_NAMES)}"
        )
    if hardware_class not in HARDWARE_CLASSES:
        raise ValueError(
            f"unknown hardware class {hardware_class!r}; "
            f"choose one of {', '.join(HARDWARE_CLASSES)}"
        )
    profile = cast(ProfileName, name)
    hardware = cast(HardwareClass, hardware_class)

    cluster_set = {"auto_heal": True, "rdma_latent_return": False}
    worker_set: dict[str, bool]
    worker_unset: tuple[str, ...]
    notes: tuple[str, ...]

    if profile == "safe":
        worker_set = {"lora_low_rss": False, "slab_weights": False}
        worker_unset = ()
        notes = (
            "Uses stock residency on every hardware class.",
            "Native RDMA latent return stays off until it requalifies on torchmonarch 0.7.0 or newer.",
        )
    elif profile == "balanced" and hardware == "homogeneous_uma":
        worker_set = {}
        worker_unset = _RESIDENCY_KEYS
        notes = (
            "Removes both residency keys, so the worker's model-aware defaults decide them.",
            "Native RDMA latent return stays off until it requalifies on torchmonarch 0.7.0 or newer.",
        )
    elif profile == "balanced":
        worker_set = {"lora_low_rss": False, "slab_weights": False}
        worker_unset = ()
        notes = (
            f"Uses stock residency because the hardware class is {hardware}.",
            "Native RDMA latent return stays off until it requalifies on torchmonarch 0.7.0 or newer.",
        )
    else:
        if hardware != "homogeneous_uma":
            raise ProfileRefusal(
                profile,
                hardware,
                "advanced residency requires a positively identified homogeneous UMA fleet",
            )
        worker_set = {"lora_low_rss": True, "slab_weights": True}
        worker_unset = ()
        notes = (
            "Enables low-RSS loading and slab residency for a verified homogeneous UMA fleet.",
            "This is an expert profile; it may need a fresh identity-gate proof.",
            "Native RDMA latent return stays off until it requalifies on torchmonarch 0.7.0 or newer.",
        )

    _validate_emission(cluster_set, worker_set, worker_unset)
    return ProfileResolution(
        name=profile,
        hardware_class=hardware,
        cluster_set=cluster_set,
        worker_args_set=worker_set,
        worker_args_unset=worker_unset,
        graph_recommendations=dict(_GRAPH_RECOMMENDATIONS),
        notes=notes,
    )


def compile_profile(
    name: str = "balanced",
    *,
    observations: Sequence[HardwareObservation] = (),
) -> ProfileResolution:
    """Resolve a profile directly from conservative hardware observations."""
    return resolve_profile(name, hardware_class=classify_hardware(observations))


def apply_profile(
    raw_config: Mapping[str, object],
    resolution: ProfileResolution,
) -> dict[str, object]:
    """Apply a resolution without mutating or rewriting an effective no-op.

    Only ``[cluster]`` and ``[worker_args]`` keys named by the resolution are
    touched.  Other strict config content is retained exactly.  When every
    requested edit is already effective, the original dict object is returned
    when possible so callers can avoid a format-only rewrite and Gate digest
    churn.
    """
    _validate_emission(
        resolution.cluster_set,
        resolution.worker_args_set,
        resolution.worker_args_unset,
    )
    cluster = _table(raw_config, "cluster")
    worker_args = _table(raw_config, "worker_args")

    next_cluster = dict(cluster)
    next_cluster.update(resolution.cluster_set)
    next_worker_args = dict(worker_args)
    for key in resolution.worker_args_unset:
        next_worker_args.pop(key, None)
    next_worker_args.update(resolution.worker_args_set)
    validate_worker_args(next_worker_args, context="profile output [worker_args]")

    if next_cluster == cluster and next_worker_args == worker_args:
        return raw_config if isinstance(raw_config, dict) else dict(raw_config)

    rendered = dict(raw_config)
    if next_cluster != cluster:
        rendered["cluster"] = next_cluster
    if next_worker_args != worker_args:
        rendered["worker_args"] = next_worker_args
    return rendered


def _table(raw_config: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = raw_config.get(name, {})
    if not isinstance(value, Mapping):
        raise ClusterConfigError(f"profile input [{name}] must be a table")
    if not all(isinstance(key, str) for key in value):
        raise ClusterConfigError(f"profile input [{name}] keys must be strings")
    return cast(Mapping[str, object], value)


def _validate_emission(
    cluster_set: Mapping[str, object],
    worker_args_set: Mapping[str, object],
    worker_args_unset: Sequence[str],
) -> None:
    allowed_cluster = {"auto_heal", "rdma_latent_return"}
    unexpected_cluster = set(cluster_set) - allowed_cluster
    if unexpected_cluster:
        raise ClusterConfigError(
            "profile attempted unsupported [cluster] output: "
            + ", ".join(sorted(unexpected_cluster))
        )
    if cluster_set.get("auto_heal") is not True:
        raise ClusterConfigError("profiles must enable cluster.auto_heal")
    if cluster_set.get("rdma_latent_return") is not False:
        raise ClusterConfigError("profiles must keep cluster.rdma_latent_return disabled")

    emitted = set(worker_args_set) | set(worker_args_unset)
    forbidden = emitted & _FORBIDDEN_EMISSIONS
    if forbidden:
        raise ClusterConfigError(
            "profile attempted forbidden policy output: " + ", ".join(sorted(forbidden))
        )
    unexpected_worker = emitted - set(_RESIDENCY_KEYS)
    if unexpected_worker:
        raise ClusterConfigError(
            "profile attempted unsupported [worker_args] output: "
            + ", ".join(sorted(unexpected_worker))
        )
    overlap = set(worker_args_set) & set(worker_args_unset)
    if overlap:
        raise ClusterConfigError(
            "profile cannot set and unset the same key: " + ", ".join(sorted(overlap))
        )
    validate_worker_args(dict(worker_args_set), context="profile [worker_args] edits")
