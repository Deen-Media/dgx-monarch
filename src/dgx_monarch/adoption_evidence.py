"""Strict, opt-in evidence for request-bound resident-model adoption.

The ordinary render path does not activate this protocol.  A trusted caller
uses :func:`resident_adoption_evidence_context` around one render submission;
the worker then emits bounded, path-free evidence in that sample's result.
This module must expose no endpoint, command, arbitrary path, or persistence
surface.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import threading
import uuid
import weakref
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

CONTEXT_REQUEST_KEY = "_dgxm_resident_adoption_context"
RESULT_KEY = "_dgxm_adoption_evidence"
CONTEXT_SCHEMA = "dgx-monarch-resident-adoption-context-v1"
EVIDENCE_SCHEMA = "dgx-monarch-resident-adoption-evidence-v1"

_CONTEXT_KEYS = frozenset({
    "schema",
    "prompt_id",
    "capsule_sha256",
    "pre_attestation_sha256",
    "leg_nonce",
})
_EVIDENCE_KEYS = frozenset({
    "schema",
    "context_sha256",
    "request_sha256",
    "render_id_sha256",
    "runtime_sha256",
    "setup_generation",
    "rank",
    "world",
    "models",
})
_MODEL_EVIDENCE_KEYS = frozenset({
    "slot",
    "resident_identity_sha256",
    "weight_residency",
    "lora_low_rss",
    "artifacts",
})
_ARTIFACT_EVIDENCE_KEYS = frozenset({
    "kind",
    "ordinal",
    "sha256",
    "resolved_path_sha256",
    "resolved_stat_sha256",
})
_HEX = frozenset("0123456789abcdef")
_FULL_HASH_CHUNK_BYTES = 4 * 1024 * 1024
_MAX_ARTIFACTS_PER_MODEL = 32
_FULL_HASH_CACHE_MAX = 64
_FULL_HASH_CACHE: OrderedDict[tuple[str, str], str] = OrderedDict()
_FULL_HASH_CACHE_LOCK = threading.Lock()


class ResidentAdoptionEvidenceError(RuntimeError):
    """The opt-in adoption-evidence contract could not be established."""


def _require_exact_keys(value: object, expected: frozenset[str], label: str) -> dict:
    if type(value) is not dict:
        raise ResidentAdoptionEvidenceError(f"{label} must be a plain object")
    actual = set(value)
    if actual != expected:
        raise ResidentAdoptionEvidenceError(
            f"{label} fields must be exactly {sorted(expected)!r}")
    if not all(type(key) is str for key in value):
        raise ResidentAdoptionEvidenceError(f"{label} fields must be strings")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (type(value) is not str or len(value) != 64
            or any(char not in _HEX for char in value)):
        raise ResidentAdoptionEvidenceError(
            f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_json(value: object, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ResidentAdoptionEvidenceError(
            f"{label} is not canonical JSON data") from exc


def _domain_sha256(domain: bytes, value: object, label: str) -> str:
    return hashlib.sha256(domain + b"\0" + _canonical_json(value, label)).hexdigest()


@dataclass(frozen=True)
class ResidentAdoptionContext:
    prompt_id: str
    capsule_sha256: str
    pre_attestation_sha256: str
    leg_nonce: str

    def to_wire(self) -> dict[str, str]:
        return {
            "schema": CONTEXT_SCHEMA,
            "prompt_id": self.prompt_id,
            "capsule_sha256": self.capsule_sha256,
            "pre_attestation_sha256": self.pre_attestation_sha256,
            "leg_nonce": self.leg_nonce,
        }

    @property
    def sha256(self) -> str:
        return context_sha256(self.to_wire())


def validate_context(value: object) -> ResidentAdoptionContext:
    """Parse the exact wire schema without retaining caller-owned structures."""
    raw = _require_exact_keys(value, _CONTEXT_KEYS, "resident-adoption context")
    if raw["schema"] != CONTEXT_SCHEMA:
        raise ResidentAdoptionEvidenceError(
            f"resident-adoption context schema must be {CONTEXT_SCHEMA!r}")
    prompt_id = raw["prompt_id"]
    if type(prompt_id) is not str:
        raise ResidentAdoptionEvidenceError(
            "resident-adoption prompt_id must be a canonical UUID")
    try:
        canonical_prompt_id = str(uuid.UUID(prompt_id))
    except (ValueError, AttributeError):
        raise ResidentAdoptionEvidenceError(
            "resident-adoption prompt_id must be a canonical UUID") from None
    if prompt_id != canonical_prompt_id:
        raise ResidentAdoptionEvidenceError(
            "resident-adoption prompt_id must be a canonical UUID")
    return ResidentAdoptionContext(
        prompt_id=prompt_id,
        capsule_sha256=_require_sha256(
            raw["capsule_sha256"], "resident-adoption capsule_sha256"),
        pre_attestation_sha256=_require_sha256(
            raw["pre_attestation_sha256"],
            "resident-adoption pre_attestation_sha256"),
        leg_nonce=_require_sha256(
            raw["leg_nonce"], "resident-adoption leg_nonce"),
    )


def context_sha256(value: object) -> str:
    context = (
        value if isinstance(value, ResidentAdoptionContext)
        else validate_context(value)
    )
    return _domain_sha256(
        b"dgx-monarch:resident-adoption-context:v1",
        context.to_wire(),
        "resident-adoption context",
    )


def render_id_sha256(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 32
        or any(char not in _HEX for char in value)
    ):
        raise ResidentAdoptionEvidenceError(
            "resident-adoption render identity is malformed")
    return hashlib.sha256(
        b"dgx-monarch:resident-adoption-render:v1\0"
        + value.encode("ascii")
    ).hexdigest()


def request_sha256(
    request: dict,
    *,
    expected_context_sha256: str,
) -> str:
    context_digest = _require_sha256(
        expected_context_sha256,
        "resident-adoption request context sha256",
    )
    return _domain_sha256(
        b"dgx-monarch:resident-adoption-request:v1",
        {
            "context_sha256": context_digest,
            "render_id_sha256": render_id_sha256(
                request.get("_dgxm_render_id")),
            "kind": request.get("kind"),
            "model": request["model"],
            "uncond_model": request.get("uncond_model"),
            "artifact_sets": request.get("_dgxm_artifact_sets"),
            "residency_mode": request.get("_dgxm_normal_residency_mode"),
        },
        "resident-adoption request",
    )


@dataclass
class _ResidentAdoptionScope:
    context: ResidentAdoptionContext
    consumed: bool = False
    active: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


_ACTIVE_CONTEXT: ContextVar[_ResidentAdoptionScope | None] = ContextVar(
    "dgx_monarch_resident_adoption_context", default=None)


@contextmanager
def resident_adoption_evidence_context(
    *,
    prompt_id: str,
    capsule_sha256: str,
    pre_attestation_sha256: str,
    leg_nonce: str,
) -> Iterator[str]:
    """Scope exact adoption evidence to render submissions in this context.

    The yielded digest lets the caller bind the returned private evidence to
    its independently authorized context without exposing that context in the
    public latent result.
    """
    if _ACTIVE_CONTEXT.get() is not None:
        raise ResidentAdoptionEvidenceError(
            "resident-adoption evidence contexts cannot be nested")
    context = validate_context({
        "schema": CONTEXT_SCHEMA,
        "prompt_id": prompt_id,
        "capsule_sha256": capsule_sha256,
        "pre_attestation_sha256": pre_attestation_sha256,
        "leg_nonce": leg_nonce,
    })
    scope = _ResidentAdoptionScope(context)
    token = _ACTIVE_CONTEXT.set(scope)
    try:
        yield context.sha256
    finally:
        with scope.lock:
            scope.active = False
        _ACTIVE_CONTEXT.reset(token)


def active_context_wire() -> dict[str, str] | None:
    scope = _ACTIVE_CONTEXT.get()
    return None if scope is None else scope.context.to_wire()


def consume_active_context_wire() -> dict[str, str] | None:
    """Consume the active proof context for exactly one render submission."""
    scope = _ACTIVE_CONTEXT.get()
    if scope is None:
        return None
    with scope.lock:
        if not scope.active:
            raise ResidentAdoptionEvidenceError(
                "resident-adoption context is no longer active")
        if scope.consumed:
            raise ResidentAdoptionEvidenceError(
                "resident-adoption context already authorized one render")
        scope.consumed = True
    return scope.context.to_wire()


def _claim_api():
    class Claim:
        __slots__ = ("__weakref__",)

    claims: weakref.WeakKeyDictionary[Claim, dict[str, str]] = (
        weakref.WeakKeyDictionary()
    )
    claims_lock = threading.Lock()

    def claim_active_context() -> object | None:
        """Mint one opaque handoff after atomically claiming the active scope."""
        wire = consume_active_context_wire()
        if wire is None:
            return None
        claim = Claim()
        with claims_lock:
            claims[claim] = wire
        return claim

    def consume_context_claim(value: object | None) -> dict[str, str] | None:
        """Unwrap an authentic handoff exactly once."""
        if value is None:
            return None
        if type(value) is not Claim:
            raise ResidentAdoptionEvidenceError(
                "resident-adoption context claim is not authentic")
        with claims_lock:
            wire = claims.pop(value, None)
            if wire is None:
                raise ResidentAdoptionEvidenceError(
                    "resident-adoption context claim was already consumed")
        return copy.deepcopy(wire)

    return claim_active_context, consume_context_claim


claim_active_context, consume_context_claim = _claim_api()
del _claim_api


def require_inactive_context(operation: str) -> None:
    """Reject multi-render surfaces before they can consume proof authority."""
    if _ACTIVE_CONTEXT.get() is not None:
        raise ResidentAdoptionEvidenceError(
            f"resident-adoption evidence does not support {operation}")


@contextmanager
def suspend_active_context() -> Iterator[None]:
    """Keep nested gate/control renders outside the caller's one proof leg."""
    scope = _ACTIVE_CONTEXT.get()
    if scope is None:
        yield
        return
    token = _ACTIVE_CONTEXT.set(None)
    try:
        yield
    finally:
        _ACTIVE_CONTEXT.reset(token)


def _stable_stat_record(value: os.stat_result) -> dict[str, int]:
    return {
        "dev": int(value.st_dev),
        "ino": int(value.st_ino),
        "mode": int(value.st_mode),
        "nlink": int(value.st_nlink),
        "uid": int(value.st_uid),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _advise_full_read_dontneed(fd: int, offset: int = 0, length: int = 0) -> None:
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or dontneed is None:
        return
    try:
        advise(fd, offset, length, dontneed)
    except OSError:
        pass


def stable_file_identity(path: str, *, label: str) -> dict[str, str]:
    """Hash one exact regular file and return path-free identity evidence.

    Error text names only the bounded caller-provided role label, never
    ``path`` or an ``OSError`` representation, which could disclose model paths.
    """
    if (
        type(path) is not str
        or not os.path.isabs(path)
        or os.path.normpath(path) != path
    ):
        raise ResidentAdoptionEvidenceError(
            f"resident artifact {label} path was not direct and absolute")
    fd: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        before_path = os.lstat(path)
        fd = os.open(path, flags)
        before_fd = os.fstat(fd)
        before = _stable_stat_record(before_fd)
        if (
            not stat.S_ISREG(before_fd.st_mode)
            or before_fd.st_nlink != 1
            or before_fd.st_uid != os.getuid()
            or _stable_stat_record(before_path) != before
        ):
            raise ResidentAdoptionEvidenceError(
                f"resident artifact {label} was not one owner-held regular file")

        path_digest = hashlib.sha256(path.encode("utf-8")).hexdigest()
        stat_digest = hashlib.sha256(
            _canonical_json(before, f"resident artifact {label} stat")
        ).hexdigest()
        cache_key = (path_digest, stat_digest)
        with _FULL_HASH_CACHE_LOCK:
            content_digest = _FULL_HASH_CACHE.get(cache_key)
            if content_digest is not None:
                _FULL_HASH_CACHE.move_to_end(cache_key)
        if content_digest is None:
            digest = hashlib.sha256()
            offset = 0
            while offset < before_fd.st_size:
                chunk = os.pread(
                    fd,
                    min(_FULL_HASH_CHUNK_BYTES, before_fd.st_size - offset),
                    offset,
                )
                if not chunk:
                    raise ResidentAdoptionEvidenceError(
                        f"resident artifact {label} had a short full-content read")
                digest.update(chunk)
                consumed_offset = offset
                offset += len(chunk)
                _advise_full_read_dontneed(
                    fd, consumed_offset, len(chunk)
                )
            content_digest = digest.hexdigest()

        after_fd = os.fstat(fd)
        after_path = os.lstat(path)
        if (
            _stable_stat_record(after_fd) != before
            or _stable_stat_record(after_path) != before
        ):
            raise ResidentAdoptionEvidenceError(
                f"resident artifact {label} changed during full-content identity")
        with _FULL_HASH_CACHE_LOCK:
            _FULL_HASH_CACHE[cache_key] = content_digest
            _FULL_HASH_CACHE.move_to_end(cache_key)
            while len(_FULL_HASH_CACHE) > _FULL_HASH_CACHE_MAX:
                _FULL_HASH_CACHE.popitem(last=False)
        return {
            "sha256": content_digest,
            "resolved_path_sha256": path_digest,
            "resolved_stat_sha256": stat_digest,
        }
    except ResidentAdoptionEvidenceError:
        raise
    except OSError:
        raise ResidentAdoptionEvidenceError(
            f"resident artifact {label} could not be stable-read") from None
    finally:
        if fd is not None:
            _advise_full_read_dontneed(fd)
            try:
                os.close(fd)
            except OSError:
                pass


def stable_file_sha256(path: str, *, label: str) -> str:
    """Compatibility helper returning the full-content digest only."""
    return stable_file_identity(path, label=label)["sha256"]


def _model_artifact_paths(spec: dict, resolver: Any) -> list[tuple[str, str]]:
    rows = [(
        "diffusion_models",
        resolver("diffusion_models", spec["unet_name"]),
    )]
    for entry in spec.get("loras") or []:
        rows.append(("loras", resolver("loras", entry["name"])))
    if len(rows) > _MAX_ARTIFACTS_PER_MODEL:
        raise ResidentAdoptionEvidenceError(
            "resident-adoption evidence exceeds the per-model artifact bound")
    return rows


def _resident_for_slot(worker: Any, slot: str) -> Any:
    return worker.store.uncond if slot == "uncond" else worker.store.current


def build_worker_evidence(
    worker: Any,
    request: dict,
    expected_artifacts: list[dict],
    *,
    resolver: Any,
    identity_fn: Any,
) -> dict | None:
    """Build worker evidence after the caller has adopted and checked residents."""
    if CONTEXT_REQUEST_KEY not in request:
        return None
    context = validate_context(request[CONTEXT_REQUEST_KEY])
    specs = [("cond", request["model"])]
    if request.get("uncond_model") is not None:
        specs.append(("uncond", request["uncond_model"]))
    if len(specs) != len(expected_artifacts):
        raise ResidentAdoptionEvidenceError(
            "resident-adoption model/artifact cardinality changed")

    model_rows = []
    for model_index, ((slot, spec), expected) in enumerate(
        zip(specs, expected_artifacts, strict=True)
    ):
        resident = _resident_for_slot(worker, slot)
        if resident is None or resident.artifact_identity != expected:
            raise ResidentAdoptionEvidenceError(
                f"resident-adoption {slot} identity changed before evidence")
        artifact_rows = []
        for ordinal, (kind, path) in enumerate(
            _model_artifact_paths(spec, resolver)
        ):
            artifact_rows.append({
                "kind": kind,
                "ordinal": ordinal,
                **stable_file_identity(
                    path, label=f"model-{model_index}-artifact-{ordinal}"),
            })
        # Bracket the full reads with the same worker-local bounded identity
        # check that authorized the resident.  This catches request/file drift
        # before sampler entry without trusting the returned full hashes alone.
        current = identity_fn(spec["unet_name"], spec.get("loras"))
        if current != expected or resident.artifact_identity != expected:
            raise ResidentAdoptionEvidenceError(
                f"resident-adoption {slot} identity changed during evidence")
        model_rows.append({
            "slot": slot,
            "resident_identity_sha256": _domain_sha256(
                b"dgx-monarch:resident-artifact-identity:v1",
                expected,
                f"resident-adoption {slot} identity",
            ),
            # Storage backing for the low-RSS proof. Comfy-managed residency changes
            # placement policy, not backing, and does not add a third value to this
            # signed schema. Worker status reports the three-valued residency mode.
            "weight_residency": (
                "slab" if getattr(resident, "slab", None) is not None
                else "cudaMalloc"
            ),
            "lora_low_rss": bool(getattr(worker.store, "lora_low_rss", False)),
            "artifacts": artifact_rows,
        })

    setup_generation = getattr(worker, "_setup_generation", None)
    rank = getattr(worker, "rank", None)
    world = getattr(worker, "world", None)
    if (type(setup_generation) is not int or setup_generation < 0
            or type(rank) is not int or rank < 0
            or type(world) is not int or world <= 0 or rank >= world):
        raise ResidentAdoptionEvidenceError(
            "resident-adoption runtime rank/setup identity is unavailable")
    context_digest = context.sha256
    render_id_digest = render_id_sha256(request.get("_dgxm_render_id"))
    request_digest = request_sha256(
        request, expected_context_sha256=context_digest)
    runtime_digest = _domain_sha256(
        b"dgx-monarch:resident-adoption-runtime:v1",
        {
            "setup_generation": setup_generation,
            "topology": dict(getattr(worker, "topology", {}) or {}),
            "worker_args": dict(getattr(worker, "_active_worker_args", {}) or {}),
        },
        "resident-adoption runtime",
    )
    return {
        "schema": EVIDENCE_SCHEMA,
        "context_sha256": context_digest,
        "request_sha256": request_digest,
        "render_id_sha256": render_id_digest,
        "runtime_sha256": runtime_digest,
        "setup_generation": setup_generation,
        "rank": rank,
        "world": world,
        "models": model_rows,
    }


def _validate_artifact_evidence(value: object, *, ordinal: int) -> dict:
    row = _require_exact_keys(
        value, _ARTIFACT_EVIDENCE_KEYS, "resident artifact evidence")
    if row["kind"] not in {"diffusion_models", "loras"}:
        raise ResidentAdoptionEvidenceError(
            "resident artifact evidence kind is invalid")
    if type(row["ordinal"]) is not int or row["ordinal"] != ordinal:
        raise ResidentAdoptionEvidenceError(
            "resident artifact evidence order is invalid")
    for digest_name in (
        "sha256",
        "resolved_path_sha256",
        "resolved_stat_sha256",
    ):
        _require_sha256(
            row[digest_name], f"resident artifact evidence {digest_name}")
    return copy.deepcopy(row)


def _validate_model_evidence(value: object, *, index: int) -> dict:
    row = _require_exact_keys(
        value, _MODEL_EVIDENCE_KEYS, "resident model evidence")
    expected_slot = "cond" if index == 0 else "uncond"
    if row["slot"] != expected_slot:
        raise ResidentAdoptionEvidenceError(
            "resident model evidence slot order is invalid")
    _require_sha256(
        row["resident_identity_sha256"],
        "resident model identity sha256",
    )
    if row["weight_residency"] not in {"cudaMalloc", "slab"}:
        raise ResidentAdoptionEvidenceError(
            "resident model evidence weight_residency is invalid")
    if type(row["lora_low_rss"]) is not bool:
        raise ResidentAdoptionEvidenceError(
            "resident model evidence lora_low_rss must be boolean")
    artifacts = row["artifacts"]
    if type(artifacts) is not list or not (
        1 <= len(artifacts) <= _MAX_ARTIFACTS_PER_MODEL
    ):
        raise ResidentAdoptionEvidenceError(
            "resident model evidence artifacts are unbounded or empty")
    copied = copy.deepcopy(row)
    copied["artifacts"] = [
        _validate_artifact_evidence(item, ordinal=ordinal)
        for ordinal, item in enumerate(artifacts)
    ]
    return copied


def validate_result_evidence(
    results: list[dict],
    expected_context_sha256: str | None,
    *,
    expected_request_sha256: str | None = None,
    expected_render_id_sha256: str | None = None,
    expected_setup_generation: int | None = None,
) -> list[dict]:
    """Require absent evidence normally, or one coherent row from every rank."""
    present = [
        result.get("resident_adoption_evidence")
        for result in results
        if result.get("resident_adoption_evidence") is not None
    ]
    if expected_context_sha256 is None:
        if present:
            raise ResidentAdoptionEvidenceError(
                "workers returned unrequested resident-adoption evidence")
        return []
    expected_context_sha256 = _require_sha256(
        expected_context_sha256, "expected resident-adoption context sha256")
    expected_request_sha256 = _require_sha256(
        expected_request_sha256,
        "expected resident-adoption request sha256",
    )
    expected_render_id_sha256 = _require_sha256(
        expected_render_id_sha256,
        "expected resident-adoption render identity sha256",
    )
    if (
        type(expected_setup_generation) is not int
        or expected_setup_generation < 0
    ):
        raise ResidentAdoptionEvidenceError(
            "expected resident-adoption setup generation is invalid")
    if len(present) != len(results):
        raise ResidentAdoptionEvidenceError(
            "resident-adoption evidence is missing from one or more ranks")

    validated = []
    for result, value in zip(results, present, strict=True):
        row = _require_exact_keys(
            value, _EVIDENCE_KEYS, "resident-adoption evidence")
        if row["schema"] != EVIDENCE_SCHEMA:
            raise ResidentAdoptionEvidenceError(
                f"resident-adoption evidence schema must be {EVIDENCE_SCHEMA!r}")
        for digest_name in (
            "context_sha256",
            "request_sha256",
            "render_id_sha256",
            "runtime_sha256",
        ):
            _require_sha256(
                row[digest_name],
                f"resident-adoption evidence {digest_name}",
            )
        if (
            row["context_sha256"] != expected_context_sha256
            or row["request_sha256"] != expected_request_sha256
            or row["render_id_sha256"] != expected_render_id_sha256
            or row["setup_generation"] != expected_setup_generation
        ):
            raise ResidentAdoptionEvidenceError(
                "resident-adoption evidence does not match the driver request")
        rank = result.get("rank")
        if type(rank) is not int or type(row["rank"]) is not int or row["rank"] != rank:
            raise ResidentAdoptionEvidenceError(
                "resident-adoption evidence rank does not match the sample result")
        if (type(row["world"]) is not int or row["world"] != len(results)
                or type(row["setup_generation"]) is not int
                or row["setup_generation"] < 0):
            raise ResidentAdoptionEvidenceError(
                "resident-adoption evidence runtime cardinality is invalid")
        models = row["models"]
        if type(models) is not list or len(models) not in {1, 2}:
            raise ResidentAdoptionEvidenceError(
                "resident-adoption evidence model cardinality is invalid")
        copied = copy.deepcopy(row)
        copied["models"] = [
            _validate_model_evidence(item, index=index)
            for index, item in enumerate(models)
        ]
        validated.append(copied)

    validated.sort(key=lambda row: row["rank"])
    if [row["rank"] for row in validated] != list(range(len(results))):
        raise ResidentAdoptionEvidenceError(
            "resident-adoption evidence ranks are incomplete or duplicated")
    def shared_view(value: dict) -> dict:
        shared = copy.deepcopy(value)
        shared.pop("rank")
        for model in shared["models"]:
            for artifact in model["artifacts"]:
                artifact.pop("resolved_path_sha256")
                artifact.pop("resolved_stat_sha256")
        return shared

    reference = shared_view(validated[0])
    if any(shared_view(row) != reference for row in validated[1:]):
        raise ResidentAdoptionEvidenceError(
            "resident-adoption evidence disagrees across ranks")
    return validated
