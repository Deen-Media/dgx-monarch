"""Request-bound resident-adoption evidence stays strict, scoped, and path-free."""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import threading
import types

import pytest
import torch

from dgx_monarch import adoption_evidence
from dgx_monarch.actor import worker as worker_mod
from dgx_monarch.nodes import common, render_result, render_submit

_PROMPT_ID = "123e4567-e89b-12d3-a456-426614174000"
_CAPSULE = "a" * 64
_PRE_ATTESTATION = "b" * 64
_NONCE = "c" * 64


def _context() -> dict[str, str]:
    return {
        "schema": adoption_evidence.CONTEXT_SCHEMA,
        "prompt_id": _PROMPT_ID,
        "capsule_sha256": _CAPSULE,
        "pre_attestation_sha256": _PRE_ATTESTATION,
        "leg_nonce": _NONCE,
    }


def test_context_manager_is_exact_scoped_and_yields_its_digest():
    assert adoption_evidence.active_context_wire() is None
    with adoption_evidence.resident_adoption_evidence_context(
        prompt_id=_PROMPT_ID,
        capsule_sha256=_CAPSULE,
        pre_attestation_sha256=_PRE_ATTESTATION,
        leg_nonce=_NONCE,
    ) as digest:
        assert adoption_evidence.active_context_wire() == _context()
        assert digest == adoption_evidence.context_sha256(_context())
        with pytest.raises(
            adoption_evidence.ResidentAdoptionEvidenceError,
            match="cannot be nested",
        ):
            with adoption_evidence.resident_adoption_evidence_context(
                prompt_id=_PROMPT_ID,
                capsule_sha256=_CAPSULE,
                pre_attestation_sha256=_PRE_ATTESTATION,
                leg_nonce=_NONCE,
            ):
                pass
    assert adoption_evidence.active_context_wire() is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(extra="not-allowed"),
        lambda value: value.update(prompt_id=_PROMPT_ID.replace("-", "")),
        lambda value: value.update(capsule_sha256="A" * 64),
        lambda value: value.update(pre_attestation_sha256=False),
        lambda value: value.update(leg_nonce="c" * 62),
    ],
)
def test_malformed_context_is_rejected(mutate):
    value = _context()
    mutate(value)
    with pytest.raises(adoption_evidence.ResidentAdoptionEvidenceError):
        adoption_evidence.validate_context(value)


def test_stable_full_content_hash_and_worker_evidence_are_path_free(tmp_path):
    checkpoint = tmp_path / "private-model-name.sft"
    lora = tmp_path / "private-lora-name.sft"
    checkpoint.write_bytes(b"checkpoint-full-content")
    lora.write_bytes(b"lora-full-content")
    expected = {
        "digest": "bounded-digest",
        "comfy": "comfy-commit",
        "artifacts": [
            {
                "kind": "diffusion_models",
                "name": checkpoint.name,
                "signature": "bounded-checkpoint",
            },
            {
                "kind": "loras",
                "name": lora.name,
                "signature": "bounded-lora",
            },
        ],
    }
    resident = types.SimpleNamespace(
        artifact_identity=expected,
        slab=object(),
    )
    worker = types.SimpleNamespace(
        store=types.SimpleNamespace(
            current=resident,
            uncond=None,
            lora_low_rss=True,
        ),
        _setup_generation=7,
        rank=0,
        world=1,
        topology={"ulysses": 1, "ring": 1, "cfg": 1, "dp": 1},
        _active_worker_args={"lora_low_rss": True, "slab_weights": True},
    )
    paths = {
        ("diffusion_models", checkpoint.name): str(checkpoint),
        ("loras", lora.name): str(lora),
    }
    request = {
        "kind": "ksampler",
        "model": {
            "unet_name": checkpoint.name,
            "options": {},
            "loras": [{"name": lora.name, "strength": 0.5}],
            "model_sampling": None,
        },
        "_dgxm_render_id": "d" * 32,
        "_dgxm_normal_residency_mode": "required",
        adoption_evidence.CONTEXT_REQUEST_KEY: _context(),
    }

    result = adoption_evidence.build_worker_evidence(
        worker,
        request,
        [expected],
        resolver=lambda kind, name: paths[(kind, name)],
        identity_fn=lambda _name, _loras: expected,
    )

    assert result is not None
    artifacts = result["models"][0]["artifacts"]
    assert [(row["kind"], row["ordinal"], row["sha256"]) for row in artifacts] == [
        (
            "diffusion_models",
            0,
            hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        ),
        ("loras", 1, hashlib.sha256(lora.read_bytes()).hexdigest()),
    ]
    assert all(len(row["resolved_path_sha256"]) == 64 for row in artifacts)
    assert all(len(row["resolved_stat_sha256"]) == 64 for row in artifacts)
    assert result["models"][0]["weight_residency"] == "slab"
    encoded = json.dumps(result, sort_keys=True)
    assert str(tmp_path) not in encoded
    assert checkpoint.name not in encoded
    assert lora.name not in encoded


def test_full_content_identity_rejects_atomic_path_drift_without_disclosure(
    tmp_path, monkeypatch
):
    path = tmp_path / "secret-model.sft"
    replacement = tmp_path / "replacement.sft"
    path.write_bytes(b"abcdefgh")
    replacement.write_bytes(b"ABCDEFGH")
    real_pread = os.pread
    reads = 0
    monkeypatch.setattr(adoption_evidence, "_FULL_HASH_CHUNK_BYTES", 4)

    def drift(fd, count, offset):
        nonlocal reads
        chunk = real_pread(fd, count, offset)
        reads += 1
        if reads == 1:
            replacement.replace(path)
        return chunk

    monkeypatch.setattr(adoption_evidence.os, "pread", drift)
    with pytest.raises(
        adoption_evidence.ResidentAdoptionEvidenceError,
        match="changed during full-content identity",
    ) as raised:
        adoption_evidence.stable_file_sha256(
            str(path), label="model-0-artifact-0")
    assert str(path) not in str(raised.value)
    assert path.name not in str(raised.value)


def test_full_content_read_error_suppresses_private_path(monkeypatch):
    private_path = "/private/models/do-not-disclose.sft"

    def fail_open(path, flags):
        raise OSError(2, "missing", path)

    monkeypatch.setattr(adoption_evidence.os, "open", fail_open)
    with pytest.raises(
        adoption_evidence.ResidentAdoptionEvidenceError,
        match="could not be stable-read",
    ) as raised:
        adoption_evidence.stable_file_sha256(
            private_path, label="model-0-artifact-0")
    assert private_path not in str(raised.value)


def test_full_content_identity_rejects_symlinks_and_multiple_links(tmp_path):
    target = tmp_path / "target.sft"
    symlink = tmp_path / "symlink.sft"
    hardlink = tmp_path / "hardlink.sft"
    target.write_bytes(b"identity")
    symlink.symlink_to(target)
    with pytest.raises(adoption_evidence.ResidentAdoptionEvidenceError):
        adoption_evidence.stable_file_identity(
            str(symlink), label="model-0-artifact-0")
    hardlink.hardlink_to(target)
    with pytest.raises(
        adoption_evidence.ResidentAdoptionEvidenceError,
        match="one owner-held regular file",
    ):
        adoption_evidence.stable_file_identity(
            str(target), label="model-0-artifact-0")


def test_full_hash_cache_is_exact_stat_keyed_and_advises_each_chunk(
    tmp_path, monkeypatch
):
    path = tmp_path / "cached.sft"
    path.write_bytes(b"abcdefgh")
    advice = []
    monkeypatch.setattr(adoption_evidence, "_FULL_HASH_CHUNK_BYTES", 4)
    monkeypatch.setattr(
        adoption_evidence,
        "_advise_full_read_dontneed",
        lambda _fd, offset=0, length=0: advice.append((offset, length)),
    )
    with adoption_evidence._FULL_HASH_CACHE_LOCK:
        adoption_evidence._FULL_HASH_CACHE.clear()

    first = adoption_evidence.stable_file_identity(
        str(path), label="model-0-artifact-0")
    assert advice[:2] == [(0, 4), (4, 4)]
    monkeypatch.setattr(
        adoption_evidence.os,
        "pread",
        lambda *_args: pytest.fail("exact stat cache reread file bytes"),
    )
    second = adoption_evidence.stable_file_identity(
        str(path), label="model-0-artifact-0")
    assert second == first


def test_absent_context_does_not_resolve_or_hash_artifacts():
    assert adoption_evidence.build_worker_evidence(
        object(),
        {"model": {"unet_name": "unused"}},
        [],
        resolver=lambda *_args: pytest.fail("ordinary render resolved proof path"),
        identity_fn=lambda *_args: pytest.fail("ordinary render reread proof identity"),
    ) is None


def test_worker_builds_evidence_after_resident_assert_and_before_sampler(monkeypatch):
    events = []
    expected = {"digest": "resident", "comfy": "commit", "artifacts": []}
    patcher = types.SimpleNamespace(model_options={})

    class Store:
        current = None
        uncond = None
        lora_low_rss = False

        def ensure(self, *_args, **_kwargs):
            events.append("ensure")
            self.current = types.SimpleNamespace(
                artifact_identity=expected,
                slab=None,
            )
            return patcher, "load"

    worker = worker_mod.GPUWorker.__new__(worker_mod.GPUWorker)
    worker._setup_key = ("ready",)
    worker._setup_generation = 3
    worker.rank, worker.world = 0, 1
    worker.topology = {"ulysses": 1, "ring": 1, "cfg": 1, "dp": 1}
    worker._active_worker_args = {}
    worker.store = Store()
    worker._check_uma_reserve = lambda: None

    monkeypatch.setattr(
        worker_mod.worker_env,
        "verify_sample_artifact_authorization",
        lambda *_args: [expected],
    )

    def assert_resident(*_args):
        events.append("assert")

    def build(*_args, **_kwargs):
        events.append("evidence")
        return {"bounded": True}

    def sample(*_args, **_kwargs):
        events.append("sample")
        return torch.zeros(1), None

    monkeypatch.setattr(
        worker_mod.worker_env,
        "assert_resident_artifact_identity",
        assert_resident,
    )
    monkeypatch.setattr(adoption_evidence, "build_worker_evidence", build)
    monkeypatch.setattr(
        worker_mod, "model_sampling_render_clone", lambda value, _spec: value)
    monkeypatch.setattr(worker_mod, "run_ksampler", sample)
    monkeypatch.setattr(
        worker_mod, "_latent_signature", lambda _value: {"sha256": "0" * 64})
    monkeypatch.setattr("dgx_monarch.actor.sampling._dp_info", lambda: (0, 1))
    from dgx_monarch.actor import comfy_bridge

    monkeypatch.setattr(comfy_bridge, "gpu_load_seconds_reset", lambda: 0.0)
    request = {
        "kind": "ksampler",
        "model": {
            "unet_name": "model.sft",
            "options": {},
            "loras": [],
            "model_sampling": None,
        },
        adoption_evidence.CONTEXT_REQUEST_KEY: _context(),
    }

    result = worker_mod.GPUWorker._sample_impl.__wrapped__(worker, request)

    assert events == ["ensure", "assert", "evidence", "sample"]
    assert result["resident_adoption_evidence"] == {"bounded": True}


def _worker_evidence(rank: int, world: int = 2) -> dict:
    return {
        "schema": adoption_evidence.EVIDENCE_SCHEMA,
        "context_sha256": adoption_evidence.context_sha256(_context()),
        "request_sha256": "1" * 64,
        "render_id_sha256": "2" * 64,
        "runtime_sha256": "3" * 64,
        "setup_generation": 4,
        "rank": rank,
        "world": world,
        "models": [{
            "slot": "cond",
            "resident_identity_sha256": "4" * 64,
            "weight_residency": "cudaMalloc",
            "lora_low_rss": False,
            "artifacts": [{
                "kind": "diffusion_models",
                "ordinal": 0,
                "sha256": "5" * 64,
                "resolved_path_sha256": f"{rank + 9:x}" * 64,
                "resolved_stat_sha256": f"{rank + 7:x}" * 64,
            }],
        }],
    }


def test_finish_render_requires_all_ranks_and_attaches_private_evidence(monkeypatch):
    monkeypatch.setattr(
        render_result, "verify_cross_rank_signatures", lambda _results, _topo: None)
    monkeypatch.setattr(render_result, "read_latent_result", lambda value: value)
    results = [
        {
            "rank": 1,
            "dp_rank": 0,
            "latent": None,
            "resident_adoption_evidence": _worker_evidence(1),
        },
        {
            "rank": 0,
            "dp_rank": 0,
            "latent": torch.ones(1),
            "resident_adoption_evidence": _worker_evidence(0),
        },
    ]
    expected = adoption_evidence.context_sha256(_context())

    out = render_result._finish_render(
        results,
        types.SimpleNamespace(dp=1),
        {},
        expected_adoption_context_sha256=expected,
        expected_adoption_request_sha256="1" * 64,
        expected_adoption_render_id_sha256="2" * 64,
        expected_adoption_setup_generation=4,
    )

    assert torch.equal(out["samples"], torch.ones(1))
    assert [row["rank"] for row in out[adoption_evidence.RESULT_KEY]] == [0, 1]
    results[0].pop("resident_adoption_evidence")
    with pytest.raises(
        adoption_evidence.ResidentAdoptionEvidenceError,
        match="missing from one or more ranks",
    ):
        render_result._finish_render(
            results,
            types.SimpleNamespace(dp=1),
            {},
            expected_adoption_context_sha256=expected,
            expected_adoption_request_sha256="1" * 64,
            expected_adoption_render_id_sha256="2" * 64,
            expected_adoption_setup_generation=4,
        )


def test_finish_render_rejects_cross_rank_adoption_disagreement(monkeypatch):
    monkeypatch.setattr(
        render_result, "verify_cross_rank_signatures", lambda _results, _topo: None)
    first = _worker_evidence(0)
    second = _worker_evidence(1)
    second["models"][0]["artifacts"][0]["sha256"] = "6" * 64
    results = [
        {
            "rank": 0,
            "dp_rank": 0,
            "latent": torch.ones(1),
            "resident_adoption_evidence": first,
        },
        {
            "rank": 1,
            "dp_rank": 0,
            "latent": None,
            "resident_adoption_evidence": second,
        },
    ]

    with pytest.raises(
        adoption_evidence.ResidentAdoptionEvidenceError,
        match="disagrees across ranks",
    ):
        render_result._finish_render(
            results,
            types.SimpleNamespace(dp=1),
            {},
            expected_adoption_context_sha256=(
                adoption_evidence.context_sha256(_context())),
            expected_adoption_request_sha256="1" * 64,
            expected_adoption_render_id_sha256="2" * 64,
            expected_adoption_setup_generation=4,
        )


def test_ordinary_finish_preserves_latent_shape_bytes_and_result_surface(
    monkeypatch,
):
    monkeypatch.setattr(
        render_result, "verify_cross_rank_signatures", lambda _results, _topo: None)
    monkeypatch.setattr(render_result, "read_latent_result", lambda value: value)
    samples = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
    expected_bytes = samples.numpy().tobytes()

    out = render_result._finish_render(
        [{"rank": 0, "dp_rank": 0, "latent": samples}],
        types.SimpleNamespace(dp=1),
        {"noise_mask": "preserved"},
    )

    assert out["samples"].shape == samples.shape
    assert out["samples"].numpy().tobytes() == expected_bytes
    assert out["noise_mask"] == "preserved"
    assert adoption_evidence.RESULT_KEY not in out


def test_run_render_suspends_context_for_gate_but_restores_it_for_submit(
    monkeypatch,
):
    observations = []
    monkeypatch.setattr(
        common, "_bind_packed_render_model", lambda model, *_args: (model, None))
    monkeypatch.setattr(common, "model_for_request", lambda model, _request: model)

    def gate(*_args):
        observations.append(("gate", adoption_evidence.active_context_wire()))
        return "PASS"

    class Pending:
        def result(self):
            return {"samples": "done"}

        def abandon(self):
            pass

    def submit(*_args, **_kwargs):
        observations.append(("submit", adoption_evidence.active_context_wire()))
        assert type(_kwargs["_claimed_adoption_context"]) is not dict
        return Pending()

    monkeypatch.setattr(common, "_maybe_auto_gate", gate)
    monkeypatch.setattr(common, "submit_render", submit)

    with common.resident_adoption_evidence_context(
        prompt_id=_PROMPT_ID,
        capsule_sha256=_CAPSULE,
        pre_attestation_sha256=_PRE_ATTESTATION,
        leg_nonce=_NONCE,
    ):
        assert common.run_render(object(), {}, {}, None, 1) == {
            "samples": "done"}

    assert observations == [("gate", None), ("submit", _context())]


def test_run_render_rejects_consumed_context_before_binding_or_gate(monkeypatch):
    monkeypatch.setattr(
        common,
        "_bind_packed_render_model",
        lambda *_args: pytest.fail("consumed context touched render binding"),
    )
    monkeypatch.setattr(
        common,
        "_maybe_auto_gate",
        lambda *_args: pytest.fail("consumed context entered automatic Gate"),
    )
    with common.resident_adoption_evidence_context(
        prompt_id=_PROMPT_ID,
        capsule_sha256=_CAPSULE,
        pre_attestation_sha256=_PRE_ATTESTATION,
        leg_nonce=_NONCE,
    ):
        adoption_evidence.claim_active_context()
        with pytest.raises(
            adoption_evidence.ResidentAdoptionEvidenceError,
            match="already authorized one render",
        ):
            common.run_render(object(), {}, {}, None, 1)


def test_claim_is_opaque_one_use_and_rejects_forgery_before_binding(monkeypatch):
    monkeypatch.setattr(
        common,
        "_bind_packed_render_model",
        lambda *_args: pytest.fail("invalid claim touched render binding"),
    )
    # submit_render consumes the claim once, up front, so its abandoned-wedge
    # heal retry reuses the same evidence; a forged claim fails there, before
    # any binding or guard side effect.
    with pytest.raises(
        adoption_evidence.ResidentAdoptionEvidenceError,
        match="not authentic",
    ):
        render_submit.submit_render(
            object(),
            {},
            {},
            None,
            1,
            _claimed_adoption_context=_context(),
        )

    with common.resident_adoption_evidence_context(
        prompt_id=_PROMPT_ID,
        capsule_sha256=_CAPSULE,
        pre_attestation_sha256=_PRE_ATTESTATION,
        leg_nonce=_NONCE,
    ):
        claim = adoption_evidence.claim_active_context()
    for attribute, value in (
        ("_wire", _context()),
        ("_consumed", False),
        ("_lock", threading.Lock()),
    ):
        with pytest.raises(AttributeError):
            setattr(claim, attribute, value)
    assert adoption_evidence.consume_context_claim(claim) == _context()
    with pytest.raises(AttributeError):
        claim._consumed = False
    with pytest.raises(
        adoption_evidence.ResidentAdoptionEvidenceError,
        match="already consumed",
    ):
        adoption_evidence.consume_context_claim(claim)


def test_context_is_one_shot_across_threads_and_copied_contexts():
    copied = None
    outcomes = []
    with common.resident_adoption_evidence_context(
        prompt_id=_PROMPT_ID,
        capsule_sha256=_CAPSULE,
        pre_attestation_sha256=_PRE_ATTESTATION,
        leg_nonce=_NONCE,
    ):
        copied = contextvars.copy_context()

        def consume():
            try:
                outcomes.append(adoption_evidence.consume_active_context_wire())
            except adoption_evidence.ResidentAdoptionEvidenceError as exc:
                outcomes.append(type(exc).__name__)

        contexts = [contextvars.copy_context(), contextvars.copy_context()]
        threads = [
            threading.Thread(target=context.run, args=(consume,))
            for context in contexts
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert sum(isinstance(value, dict) for value in outcomes) == 1
    assert outcomes.count("ResidentAdoptionEvidenceError") == 1
    assert copied is not None
    with pytest.raises(
        adoption_evidence.ResidentAdoptionEvidenceError,
        match="no longer active",
    ):
        copied.run(adoption_evidence.consume_active_context_wire)


def test_ordinary_render_strips_stale_adoption_evidence(monkeypatch):
    monkeypatch.setattr(
        render_result, "verify_cross_rank_signatures", lambda _results, _topo: None)
    monkeypatch.setattr(render_result, "read_latent_result", lambda value: value)
    samples = torch.ones(1)
    out = render_result._finish_render(
        [{"rank": 0, "dp_rank": 0, "latent": samples}],
        types.SimpleNamespace(dp=1),
        {
            "samples": torch.zeros(1),
            adoption_evidence.RESULT_KEY: {"stale": True},
        },
    )
    assert adoption_evidence.RESULT_KEY not in out


def test_render_pipeline_rejects_active_context_before_binding(monkeypatch):
    from dgx_monarch.nodes.pipeline import RenderPipeline

    monkeypatch.setattr(
        common,
        "_bind_packed_render_model",
        lambda *_args: pytest.fail("pipeline touched render binding"),
    )
    pipeline = RenderPipeline(depth=2)
    with common.resident_adoption_evidence_context(
        prompt_id=_PROMPT_ID,
        capsule_sha256=_CAPSULE,
        pre_attestation_sha256=_PRE_ATTESTATION,
        leg_nonce=_NONCE,
    ), pytest.raises(
        adoption_evidence.ResidentAdoptionEvidenceError,
        match="does not support RenderPipeline",
    ):
        pipeline.push(object(), {}, {}, None, 1)
