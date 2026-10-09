"""Tests for the residency ladder in actor/store_residency.py.

Rules gate accuracy, never availability: a checkpoint stock residency cannot
hold gets a slab rescue offer wherever slab can run. These tests pin the
resolution order, the consent descriptor the driver parses out of the refusal
text, the two independent quarantine backstops, the stock and slab prices, and
the first-load-stock memo rule: a consent bypasses that rule's consequence,
never the memo, which is written from the loaded model and not the header sniff.
"""
import ast
import json
import struct
import sys
import types
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import capacity_fit, consent_descriptor, mesh_safety
from dgx_monarch.actor import model_store as ms
from dgx_monarch.actor import store_residency, worker_compile
from dgx_monarch.actor.model_store import ModelStore
from dgx_monarch.actor.store_detect import LivePrecisionEvidence
from dgx_monarch.capacity_fit import StockFit
from dgx_monarch.consent_descriptor import SlabResidencyRescueOffer
from dgx_monarch.constants import TRANSITION_LOAD
from slab_lifetime_helpers import reset_slab_lifetime

FITS = StockFit(True, True, 20 * 2**30, 90 * 2**30)
DOES_NOT_FIT = StockFit(False, True, 62 * 2**30, 47 * 2**30)
SKIPPED = StockFit(True, False, 0, None, "a loader dtype cast changes the resident size")

STACK = [{"name": "a.safetensors", "strength": 1.0}]

# The shipped flux2-dev bf16 checkpoint, 60.02 GiB, the artifact both stock
# kills below were measured on.
FLUX2_BF16_BYTES = 64_446_596_128


@pytest.fixture(autouse=True)
def _isolated_slab_lifetime(monkeypatch):
    """Keep any refused load's module-global ownership state inside this file."""
    reset_slab_lifetime(monkeypatch)


def _resolve(**overrides):
    kwargs = {
        "path": "/models/h3_fl2va_bf16.safetensors",
        "unet_name": "h3_fl2va_bf16.safetensors",
        "model_options": {},
        "slab_weights": "auto",
        "slab_capable_path": True,
        "lora_low_rss": True,
        "fsdp_launch": False,
        "blocked_reason": "",
        "authoritative_slab_retry": False,
        "memoized_family": lambda _path: None,
        "vouched_families": frozenset({"krea2"}),
        "rescue_consent": None,
        "fit_probe": lambda _path, _options: FITS,
        "file_identity": lambda _path: "1:2:3:4:5",
        "compile_dit": lambda: False,
    }
    kwargs.update(overrides)
    return store_residency.resolve(**kwargs)


def test_explicit_slab_loads_slab_for_any_family():
    """Slab is both the rescue and an explicit choice, and the choice needs no vouching."""
    probe_calls = []
    decision = _resolve(
        slab_weights=True,
        fit_probe=lambda path, options: probe_calls.append(path) or DOES_NOT_FIT,
    )
    assert (decision.use_slab, decision.rung) == (True, store_residency.RUNG_EXPLICIT)
    assert probe_calls == []


def test_explicit_stock_stays_stock_when_it_fits():
    decision = _resolve(slab_weights=False)
    assert (decision.use_slab, decision.rung) == (
        False, store_residency.RUNG_EXPLICIT_STOCK)


def test_vouched_auto_loads_slab_without_touching_the_fit_probe():
    probe_calls = []
    decision = _resolve(
        memoized_family=lambda _path: "krea2",
        fit_probe=lambda path, options: probe_calls.append(path) or FITS,
    )
    assert (decision.use_slab, decision.rung) == (True, store_residency.RUNG_VOUCHED_AUTO)
    # A vouched family must not be gated on MemAvailable jitter.
    assert probe_calls == []


def test_authoritative_retry_does_not_read_the_family_memo():
    reads = []
    decision = _resolve(
        authoritative_slab_retry=True,
        memoized_family=lambda path: reads.append(path),
    )
    assert decision.rung == store_residency.RUNG_VOUCHED_AUTO
    assert reads == []


def test_auto_stock_when_it_fits():
    decision = _resolve(memoized_family=lambda _path: "minimax_h3")
    assert (decision.use_slab, decision.rung) == (False, store_residency.RUNG_STOCK_FITS)
    assert decision.auto_retry_eligible is True


def test_auto_stock_when_the_probe_does_not_apply():
    decision = _resolve(fit_probe=lambda _path, _options: SKIPPED)
    assert (decision.use_slab, decision.rung) == (False, store_residency.RUNG_STOCK_FITS)


def test_auto_does_not_fit_offers_a_rescue():
    with pytest.raises(SlabResidencyRescueOffer) as raised:
        _resolve(fit_probe=lambda _path, _options: DOES_NOT_FIT)
    message = str(raised.value)
    assert "Open the DGX Monarch panel" in message
    assert '"Load with slab residency"' in message
    assert message.index("panel") < message.index(consent_descriptor.ENV_SLAB_RESCUE)
    assert "62.0 GiB" in message and "47.0 GiB" in message


def test_a_family_memo_is_not_a_consent():
    """A memoized but unvouched family still lands on the offer, not on slab."""
    with pytest.raises(SlabResidencyRescueOffer):
        _resolve(memoized_family=lambda _path: "minimax_h3",
                 fit_probe=lambda _path, _options: DOES_NOT_FIT)


def test_consent_resolves_to_a_slab_load():
    decision = _resolve(fit_probe=lambda _path, _options: DOES_NOT_FIT,
                        rescue_consent={"id": "abc", "source": "panel"})
    assert (decision.use_slab, decision.rung) == (
        True, store_residency.RUNG_CONSENTED_RESCUE)
    assert decision.auto_retry_eligible is False
    assert decision.consent_id


@pytest.mark.parametrize(("overrides", "blocker"), [
    ({"compile_dit": lambda: True}, "compile_dit"),
    ({"lora_low_rss": False}, "lora_low_rss"),
    ({"slab_capable_path": False}, ".safetensors"),
])
def test_an_unavailable_rescue_refuses_with_the_reason_and_the_options(overrides, blocker):
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        _resolve(fit_probe=lambda _path, _options: DOES_NOT_FIT,
                 rescue_consent={"id": "abc"}, **overrides)
    message = str(raised.value)
    assert blocker in message
    assert "What would fit:" in message
    assert consent_descriptor.DESCRIPTOR_BEGIN not in message   # no card is offered
    assert not consent_descriptor.is_slab_rescue_offer(raised.value)


def test_fsdp_launch_prices_the_shard_build_transient_not_the_stock_fit(
        monkeypatch, tmp_path):
    # An FSDP rank never steadies at a full resident copy, so the ladder's
    # stock full-copy fit is the wrong predicate. The FSDP transient check is
    # the whole capacity question: when it admits, the load proceeds stock (with
    # the mmap load window) even where stock residency would not fit, and when
    # it refuses, the refusal is its own text.
    from dgx_monarch.adapters import fsdp

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x" * 1000)
    monkeypatch.setattr(fsdp, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(fsdp, "mem_available_bytes", lambda: 6 * (1 << 30))
    decision = _resolve(fsdp_launch=True, path=str(checkpoint),
                        fit_probe=lambda _path, _options: DOES_NOT_FIT)
    assert (decision.use_slab, decision.rung) == (
        False, store_residency.RUNG_STOCK_FITS)
    assert decision.auto_retry_eligible is False

    monkeypatch.setattr(fsdp, "mem_available_bytes", lambda: 1100)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        _resolve(fsdp_launch=True, path=str(checkpoint),
                 fit_probe=lambda _path, _options: DOES_NOT_FIT)
    message = str(raised.value)
    assert "checkpoint kind is unproven" in message
    assert "5.0 GiB host floor" in message
    # The ladder re-raises the FSDP check's sentence, so it must arrive tagged
    # once: tagging a tagged text escapes the inner tag and prints the escape.
    from dgx_monarch import refusal as refusal_module

    assert message.count("[dgxm:") == 1
    tag = refusal_module.parse_refusal_tag(message)
    assert tag is not None and tag.guard == "stock_load_preflight"
    assert tag.waivable is False


def test_explicit_stock_never_offers_even_with_a_consent_present():
    """The worker-side backstop: a stale or buggy consent push cannot rescue
    into a lever an identity gate turned off."""
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        _resolve(slab_weights=False, fit_probe=lambda _path, _options: DOES_NOT_FIT,
                 rescue_consent={"id": "abc"})
    message = str(raised.value)
    assert consent_descriptor.DESCRIPTOR_BEGIN not in message
    assert "quarantine" in message
    assert consent_descriptor.parse(message) is None


def test_explicit_stock_names_the_reason_an_upstream_gate_carried_forward():
    with pytest.raises(mesh_safety.StockLoadCapacityError,
                       match="the identity ceremony quarantined slab_weights"):
        _resolve(slab_weights=False, fit_probe=lambda _path, _options: DOES_NOT_FIT,
                 blocked_reason="the identity ceremony quarantined slab_weights")


def test_every_refusal_still_reads_as_a_capacity_error_over_the_wire():
    """Subclassing does not survive the Monarch wire, so both mechanisms hold."""
    with pytest.raises(SlabResidencyRescueOffer) as raised:
        _resolve(fit_probe=lambda _path, _options: DOES_NOT_FIT)
    assert mesh_safety.is_stock_load_capacity_error(raised.value)
    wrapped = RuntimeError(
        f"ActorError: SlabResidencyRescueOffer: {raised.value}")
    assert mesh_safety.is_stock_load_capacity_error(wrapped)
    assert consent_descriptor.is_slab_rescue_offer(wrapped)


def test_rescue_offer_descriptor_round_trips():
    with pytest.raises(SlabResidencyRescueOffer) as raised:
        _resolve(fit_probe=lambda _path, _options: DOES_NOT_FIT,
                 family_hint="minimax_h3")
    descriptor = consent_descriptor.parse(str(raised.value))
    assert descriptor is not None
    assert descriptor.kind == consent_descriptor.KIND_RESCUE_SLAB
    assert descriptor.unet_name == "h3_fl2va_bf16.safetensors"
    assert descriptor.file_identity == "1:2:3:4:5"
    assert descriptor.family_hint == "minimax_h3"
    assert descriptor.evidence == consent_descriptor.EVIDENCE_BYTE_VERIFIED
    assert descriptor.panel_action == "Load with slab residency"
    assert descriptor.measured["weights_gib"] == DOES_NOT_FIT.size_gib
    assert descriptor.measured["mem_available_gib"] == DOES_NOT_FIT.avail_gib
    assert descriptor.measured["headroom_gib"] == DOES_NOT_FIT.headroom_gib
    assert descriptor.measured["fits"] is False
    assert json.dumps(descriptor.public(), allow_nan=False)
    assert consent_descriptor.parse(consent_descriptor.encode(descriptor)) == descriptor


def test_consent_id_is_deterministic_and_scoped():
    def offer(**overrides):
        with pytest.raises(SlabResidencyRescueOffer) as raised:
            _resolve(fit_probe=lambda _path, _options: DOES_NOT_FIT, **overrides)
        parsed = consent_descriptor.parse(str(raised.value))
        assert parsed is not None
        return parsed

    first = offer()
    assert offer().consent_id == first.consent_id          # idempotent: one card
    other_file = offer(file_identity=lambda _path: "9:9:9:9:9")
    assert other_file.consent_id != first.consent_id
    other_context = offer(model_options={"weight_dtype": "bf16"})
    assert other_context.consent_id != first.consent_id


@pytest.mark.parametrize("text", [
    "",
    "no descriptor here",
    consent_descriptor.DESCRIPTOR_BEGIN + '{"version":1}',
    consent_descriptor.DESCRIPTOR_BEGIN + "{}" + consent_descriptor.DESCRIPTOR_END,
    consent_descriptor.DESCRIPTOR_BEGIN + "not json" + consent_descriptor.DESCRIPTOR_END,
    consent_descriptor.DESCRIPTOR_BEGIN + '{"version":2}' + consent_descriptor.DESCRIPTOR_END,
])
def test_parse_is_total_and_fails_closed(text):
    assert consent_descriptor.parse(text) is None


def test_two_descriptors_in_one_text_are_refused():
    with pytest.raises(SlabResidencyRescueOffer) as raised:
        _resolve(fit_probe=lambda _path, _options: DOES_NOT_FIT)
    one = str(raised.value)
    assert consent_descriptor.parse(one) is not None
    # The parser must never pick one of two consents in silence.
    assert consent_descriptor.parse(one + "\n" + one) is None


def test_parse_refuses_duplicate_keys_and_non_finite_constants():
    duplicate = (consent_descriptor.DESCRIPTOR_BEGIN
                 + '{"version":1,"version":1,"kind":"rescue-slab"}'
                 + consent_descriptor.DESCRIPTOR_END)
    assert consent_descriptor.parse(duplicate) is None
    infinite = (consent_descriptor.DESCRIPTOR_BEGIN
                + '{"version":1,"kind":"rescue-slab","measured":{"a":Infinity}}'
                + consent_descriptor.DESCRIPTOR_END)
    assert consent_descriptor.parse(infinite) is None


def test_parse_refuses_an_oversized_payload():
    payload = json.dumps({"version": 1, "kind": "rescue-slab",
                          "pad": "x" * consent_descriptor.MAX_DESCRIPTOR_BYTES})
    text = consent_descriptor.DESCRIPTOR_BEGIN + payload + consent_descriptor.DESCRIPTOR_END
    assert consent_descriptor.parse(text) is None


def test_unknown_kinds_are_refused_and_the_reserved_waiver_grammar_is_not():
    assert consent_descriptor.valid_kind("rescue-slab")
    assert consent_descriptor.valid_kind("waive-known-wrong:ring-pad")
    assert not consent_descriptor.valid_kind("rescue-everything")
    assert not consent_descriptor.valid_kind("waive-known-wrong:")
    assert not consent_descriptor.valid_kind("waive-known-wrong:" + "x" * 64)


def test_stock_load_fit_mirrors_the_preflight_skips(tmp_path, monkeypatch):
    path = tmp_path / "m.safetensors"
    path.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 1024)
    tight = capacity_fit.stock_load_fit(str(path), {})
    assert (tight.applies, tight.fits) == (True, False)

    cast = capacity_fit.stock_load_fit(str(path), {"dtype": torch.bfloat16})
    assert (cast.applies, cast.fits) == (False, True)

    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: False)
    discrete = capacity_fit.stock_load_fit(str(path), {})
    assert (discrete.applies, discrete.fits) == (False, True)

    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: None)
    blind = capacity_fit.stock_load_fit(str(path), {})
    assert (blind.applies, blind.fits) == (False, True)


def test_the_ladder_is_pure():
    """No torch, no comfy, no CUDA: the whole ladder runs on a CPU-only box."""
    source = Path(store_residency.__file__).read_text()
    imported = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not [name for name in imported
                if name.split(".")[0] in {"torch", "comfy"}]


def _write_one_tensor_safetensors(path) -> None:
    header = json.dumps({
        "weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
    }).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(2))


class _FakeBase:
    def __init__(self):
        self.model = SimpleNamespace(diffusion_model="DIT_MODULE")
        self.offload_device = "cpu"


class _FakeCertificate:
    def public(self):
        return {"summary": "v1/sampled/deadbeefdeadbeef/1t/1v/1q/0.0MiB",
                "mode": "sampled", "verified_tensors": 1}


class _FakeSlab:
    total_gib = 24.5
    certificate = _FakeCertificate()

    def reabsorb(self, module, prefix=""):
        return {"reabsorbed": 0, "reabsorbed_gib": 0.0, "skipped": 0,
                "quant_stray_gib": 0.0}

    def close(self):
        return None

    def telemetry(self):
        return {"slab_gib": self.total_gib}


@pytest.fixture
def rig(monkeypatch, tmp_path):
    calls = SimpleNamespace(load=0, slabs=[], memo_writes=[], fit=DOES_NOT_FIT, path="")

    mm_stub = types.ModuleType("comfy.model_management")
    mm_stub.get_torch_device = lambda: "cuda:0"
    mm_stub.soft_empty_cache = lambda: None
    mm_stub.unload_all_models = lambda: None
    comfy = types.ModuleType("comfy")
    comfy_sd = types.ModuleType("comfy.sd")

    def load_diffusion_model(path, model_options=None):
        calls.load += 1
        base = _FakeBase()
        weakref.ref(base)
        return base

    comfy_sd.load_diffusion_model = load_diffusion_model
    comfy.sd = comfy_sd
    comfy.model_management = mm_stub
    for name, module in (("comfy", comfy), ("comfy.sd", comfy_sd),
                         ("comfy.model_management", mm_stub)):
        monkeypatch.setitem(sys.modules, name, module)

    from contextlib import contextmanager

    from dgx_monarch.actor import comfy_bridge

    @contextmanager
    def fake_slab_load(path, *, handoff=None):
        slab = _FakeSlab()
        calls.slabs.append(slab)
        if handoff is not None:
            handoff.append(slab)
        yield slab

    checkpoint = tmp_path / "m.safetensors"
    _write_one_tensor_safetensors(checkpoint)
    calls.path = str(checkpoint)

    monkeypatch.setattr(comfy_bridge, "slab_load", fake_slab_load)
    monkeypatch.setattr(ms, "resolve_model_path", lambda _kind, _name: str(checkpoint))
    monkeypatch.setattr(ms, "_detect_checkpoint_kind", lambda _path, _kind: None)
    monkeypatch.setattr("dgx_monarch.gate_ledger.artifact_signature", lambda path: path)
    monkeypatch.setattr(ms, "_detect_live_precision",
                        lambda _patcher, _kind: LivePrecisionEvidence("bf16", "all_bf16"))
    # The loaded truth deliberately differs from any header sniff.
    monkeypatch.setattr(ms, "_detect_family", lambda _base: "minimax_h3")
    monkeypatch.setattr(ms, "memoized_family", lambda _path: None)
    monkeypatch.setattr(ms, "memoize_family",
                        lambda path, family: calls.memo_writes.append((path, family)))
    monkeypatch.setattr(ms, "_merge_and_free", lambda active, base_path=None: None)
    monkeypatch.setattr(ModelStore, "_build_active",
                        lambda self, base, stack: SimpleNamespace(tag="ACTIVE", model=base.model))
    monkeypatch.setattr(capacity_fit, "stock_load_fit", lambda _path, _options: calls.fit)

    store = ModelStore()
    store.lora_low_rss = True
    store.slab_weights = "auto"
    return store, calls


def test_consent_bypasses_the_first_load_stock_consequence(rig):
    store, calls = rig

    _active, transition = store.ensure(
        "m.safetensors", None, STACK, rescue_consent={"id": "abc", "source": "panel"})

    assert transition == TRANSITION_LOAD
    assert store.current.slab is calls.slabs[0]          # the first load is slab
    assert store.current.residency_rung == store_residency.RUNG_CONSENTED_RESCUE
    # The memo is written from the loaded model, never from the sniff, and the
    # consent grants nothing beyond this load's residency.
    assert calls.memo_writes == [(calls.path, "minimax_h3")]
    assert store.current.slab_auto_retry is False
    assert "minimax_h3" not in ms.SLAB_VOUCHED_FAMILIES

    described = store.snapshot()["cond"]
    assert described["residency"] == "slab"
    assert described["residency_rung"] == store_residency.RUNG_CONSENTED_RESCUE
    assert described["slab_certificate"]["mode"] == "sampled"


def test_fresh_slab_load_marks_the_patcher_before_the_injection_callback(rig):
    """The worker's compile seam consumes actual load truth, not a header guess."""
    store, _calls = rig
    seen = []

    store.ensure(
        "m.safetensors", None, STACK,
        rescue_consent={"id": "abc", "source": "panel"},
        on_base_loaded=lambda base, *_args: seen.append(
            worker_compile.compile_dit_allowed(base)),
    )

    assert seen == [False]


def test_a_dispatched_consent_reaches_this_load_through_the_worker(rig):
    """A dispatched consent reaches the render's own load, end to end on the worker.

    A render's lazy load runs inside `sample`, which reads its consent off the
    validated dispatch envelope and hands it to the store through the one
    wrapper every worker endpoint calls. Drive that chain: a wire envelope in, a
    slab-resident first load out.
    """
    from dgx_monarch.actor import store_fsdp, worker_env
    from dgx_monarch.mesh_residency import CAPACITY_RESCUE_CONSENT_CAPABILITY

    store, calls = rig
    request = {
        "model": {"unet_name": "m.safetensors", "options": {}, "loras": STACK},
        "_dgxm_normal_residency_mode": "capacity_consent",
        "_dgxm_normal_residency_grant": {
            "capability": CAPACITY_RESCUE_CONSENT_CAPABILITY,
            "consent": {"id": "c" * 32, "kind": "rescue-slab",
                        "consent_source": "panel"},
        },
    }
    worker = SimpleNamespace(store=store, topology={})

    _active, transition = store_fsdp.ensure(
        worker, "m.safetensors", None, STACK, slot="cond",
        rescue_consent=worker_env.sample_rescue_consent(request))

    assert transition == TRANSITION_LOAD
    assert store.current.residency_rung == store_residency.RUNG_CONSENTED_RESCUE
    assert store.current.slab is calls.slabs[0]


def test_a_render_that_carries_no_consent_reads_none(rig):
    """The reader must not invent a grant from a gate grant, a fleet grant, or a
    request that carries nothing at all."""
    from dgx_monarch.actor import worker_env

    assert worker_env.sample_rescue_consent({"model": {}}) is None
    assert worker_env.sample_rescue_consent(
        {"_dgxm_normal_residency_grant": {"capability": "normal_render_residency",
                                          "gate_token": ["t"]}}) is None
    assert worker_env.sample_rescue_consent(
        {"_dgxm_normal_residency_grant": "not-a-dict"}) is None


def test_without_a_consent_the_same_load_offers_a_rescue(rig):
    store, calls = rig
    with pytest.raises(SlabResidencyRescueOffer):
        store.ensure("m.safetensors", None, STACK)
    assert calls.load == 0
    assert calls.slabs == []
    assert store.current is None


def test_a_fitting_stock_load_is_unchanged_and_records_its_rung(rig):
    store, calls = rig
    calls.fit = FITS

    _active, transition = store.ensure("m.safetensors", None, STACK)

    assert transition == TRANSITION_LOAD
    assert store.current.slab is None
    assert store.current.residency_rung == store_residency.RUNG_STOCK_FITS
    assert store.snapshot()["cond"]["residency"] == "stock"
    assert store.snapshot()["cond"]["slab_certificate"] is None


def test_a_certificate_failure_aborts_the_load_and_flips_no_lever(rig, monkeypatch):
    """Fail-closed on bytes, silent on trust: corruption is a storage fault,
    not a model-correctness finding, so it quarantines nothing."""
    from dgx_monarch.actor import comfy_bridge
    from dgx_monarch.actor.slab_certificate import SlabCertificateError

    store, calls = rig

    class Failing:
        def __enter__(self):
            raise SlabCertificateError("slab byte-verify FAILED for m.safetensors")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        comfy_bridge,
        "slab_load",
        lambda _path, *, handoff=None: Failing(),
    )

    with pytest.raises(SlabCertificateError):
        store.ensure("m.safetensors", None, STACK, rescue_consent={"id": "abc"})

    assert store.current is None
    assert store.slab_weights == "auto"      # no residency lever was flipped
    assert store.slab_blocked_reason == ""
    assert calls.memo_writes == []           # nothing was learned from a refused load
    assert not mesh_safety.is_stock_load_capacity_error(
        SlabCertificateError("slab byte-verify FAILED"))


def test_set_slab_mode_carries_a_blocking_reason_forward(rig):
    store, _calls = rig
    store.set_slab_mode(False, reason="the identity ceremony quarantined slab_weights")
    assert store.slab_blocked_reason == "the identity ceremony quarantined slab_weights"
    store.set_slab_mode("auto")
    assert store.slab_blocked_reason == ""


def test_stock_load_fit_prices_the_file_plus_the_host_copy_it_is_placed_from(
        tmp_path, monkeypatch):
    # The price is the file plus the host copy a stock load is placed from,
    # because a load's high water never sits below what it holds while it
    # places every weight. Two kills on the same 60.02 GiB bf16 file bound the
    # price from below: 2026-08-12 killed at 69.23 GiB available, and 2026-09-02
    # admitted at 107.46 GiB available, climbed to 1.694x the file with 10.58
    # GiB left, then stopped answering. Neither sets the current price: a full
    # placement measured on 2026-10-01 took 2.08x a 33.0 GiB fp8mixed file, and
    # the 2.1 factor set from it sits above every partial-load reading, both
    # kills included.
    path = tmp_path / "w.safetensors"
    path.write_bytes(b"x" * 1024)
    gib = float(1 << 30)
    monkeypatch.setattr(capacity_fit.os.path, "getsize", lambda _p: FLUX2_BF16_BYTES)
    monkeypatch.setattr(capacity_fit.mesh_safety, "gpu_is_integrated", lambda: True)

    monkeypatch.setattr(
        capacity_fit.mesh_safety, "mem_available_bytes", lambda: int(69.23 * gib)
    )
    first_kill = capacity_fit.stock_load_fit(str(path), {})
    assert first_kill.applies and not first_kill.fits

    monkeypatch.setattr(
        capacity_fit.mesh_safety, "mem_available_bytes", lambda: int(107.46 * gib)
    )
    second_kill = capacity_fit.stock_load_fit(str(path), {})
    assert second_kill.applies and not second_kill.fits

    # The placement price is one number in one home: the driver charges the
    # same share of the same file for the same placement.
    assert capacity_fit.STOCK_LOAD_TRANSIENT_FACTOR == 1.0 + capacity_fit.STOCK_PLACEMENT_RATIO

    monkeypatch.setattr(
        capacity_fit.mesh_safety, "mem_available_bytes",
        lambda: int(2 * capacity_fit.STOCK_LOAD_TRANSIENT_FACTOR * FLUX2_BF16_BYTES),
    )
    roomy = capacity_fit.stock_load_fit(str(path), {})
    assert roomy.applies and roomy.fits


def test_the_issue_443_measured_placement_refuses_and_fits_at_its_bounds(
        tmp_path, monkeypatch):
    """The 2026-10-01 flux2 fp8mixed measurement, pinned directly: a 33.0 GiB
    file needs ``int(2.1 * size) + 5 GiB floor``. A 1.85x price admitted ranks
    at 76.9 and 68.8 GiB available that then partial-loaded into the divergence
    guard instead of refusing up front; 2.1x refuses at 68.8 and fits at 77.3."""
    path = tmp_path / "flux2-fp8mixed.safetensors"
    path.write_bytes(b"x" * 1024)
    gib = float(1 << 30)
    size_bytes = int(33.0 * gib)
    monkeypatch.setattr(capacity_fit.os.path, "getsize", lambda _p: size_bytes)
    monkeypatch.setattr(capacity_fit.mesh_safety, "gpu_is_integrated", lambda: True)

    required = int(2.1 * size_bytes) + capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES
    assert capacity_fit.stock_required_bytes(size_bytes) == required

    monkeypatch.setattr(
        capacity_fit.mesh_safety, "mem_available_bytes", lambda: int(68.8 * gib))
    refused = capacity_fit.stock_load_fit(str(path), {})
    assert refused.applies and not refused.fits

    monkeypatch.setattr(
        capacity_fit.mesh_safety, "mem_available_bytes", lambda: int(77.3 * gib))
    fits = capacity_fit.stock_load_fit(str(path), {})
    assert fits.applies and fits.fits


def test_cfg2_single_model_rank_is_refused_at_the_measured_kill(tmp_path, monkeypatch):
    """The 2026-09-02 cfg2 kill, priced through the whole worker ladder.

    flux2 is not a dual-model cfg family, so each rank of a cfg2 world places
    one full copy of the same file. That leg ran with slab off, so each rank
    takes the explicit-stock branch. The head refused at 73.06 GiB available
    and the peer, reading 107.46 GiB, was admitted and died. Both ranks must
    refuse, and the evidence must carry a negative headroom.
    """
    path = tmp_path / "flux2-dev.safetensors"
    path.write_bytes(b"x" * 1024)
    gib = float(1 << 30)
    available = int(107.46 * gib)
    monkeypatch.setattr(capacity_fit.os.path, "getsize", lambda _p: FLUX2_BF16_BYTES)
    monkeypatch.setattr(capacity_fit.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(
        capacity_fit.mesh_safety, "mem_available_bytes", lambda: available)

    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        _resolve(path=str(path), unet_name=path.name, slab_weights=False,
                 world=2, fit_probe=None)
    text = str(raised.value)
    required = capacity_fit.stock_required_bytes(FLUX2_BF16_BYTES)
    assert available - required < 0
    assert f'"headroom_bytes":{available - required}' in text
    # The sentence quotes what the load needs, not what the file weighs: a card
    # reading "60.0 GiB does not fit in 107.5 GiB" reads as a bug. 131.0 GiB is
    # 2.1 times the file plus the 5.0 GiB absolute floor.
    required_gib = capacity_fit.gib(required)
    assert f"the load needs {required_gib} GiB (a 60.0 GiB file" in text
    assert "107.5 GiB of unified memory is available" in text


def test_refused_transient_fit_reports_negative_headroom(tmp_path, monkeypatch):
    # The refusal evidence must explain itself: a load refused on the
    # transient reports the factored requirement and negative headroom,
    # never a positive headroom beside a refusal.
    path = tmp_path / "w.safetensors"
    path.write_bytes(b"x" * 1024)
    monkeypatch.setattr(capacity_fit.os.path, "getsize", lambda _p: 60_020_000_000)
    monkeypatch.setattr(capacity_fit.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(
        capacity_fit.mesh_safety, "mem_available_bytes", lambda: 69_230_000_000
    )
    fit = capacity_fit.stock_load_fit(str(path), {})
    assert not fit.fits
    measured = fit.measured()
    assert measured["required_bytes"] > measured["mem_available_bytes"]
    assert measured["headroom_bytes"] < 0
    assert fit.headroom_gib is not None and fit.headroom_gib < 0


# The floor is the one ``slab_load_fit`` charges. A fixture at 4 GiB pins a card
# shape no production path can produce, and its assertions then read as the
# shipped wall's arithmetic when they are the fixture's own.
_FLOOR = capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES
SLAB_FITS = capacity_fit.SlabFit(True, True, 20 * 2**30, 90 * 2**30, _FLOOR)
SLAB_SHORT = capacity_fit.SlabFit(False, True, 62 * 2**30, 47 * 2**30, _FLOOR)


def _slab_probe(monkeypatch, fit):
    """Replace the slab probe and record every call the ladder makes."""
    calls = []

    def probe(path, options, *, reserve_bytes=None, **_kw):
        calls.append(path)
        return fit

    monkeypatch.setattr(capacity_fit, "slab_load_fit", probe)
    return calls


def test_the_vouched_auto_rung_prices_the_slab_and_not_the_stock_load(monkeypatch):
    """The vouched auto rung calls the slab probe and not the stock probe.

    A vouched family must not be gated on MemAvailable jitter through a price
    it never pays, but the slab price it does pay is taken.
    """
    slab_calls = _slab_probe(monkeypatch, SLAB_FITS)
    stock_calls = []
    decision = _resolve(
        memoized_family=lambda _path: "krea2",
        fit_probe=lambda path, options: stock_calls.append(path) or FITS,
    )
    assert (decision.use_slab, decision.rung) == (True, store_residency.RUNG_VOUCHED_AUTO)
    assert stock_calls == []
    assert len(slab_calls) == 1


def test_the_slab_probe_runs_exactly_once_per_resolve(monkeypatch):
    """Two reads of /proc/meminfo inside one resolve could disagree."""
    for overrides in ({"slab_weights": True},
                      {"memoized_family": lambda _path: "krea2"},
                      {"fit_probe": lambda _p, _o: DOES_NOT_FIT,
                       "rescue_consent": {"id": "abc"}},
                      {"fit_probe": lambda _p, _o: FITS}):
        calls = _slab_probe(monkeypatch, SLAB_FITS)
        _resolve(**overrides)
        assert len(calls) == 1, overrides


def test_a_slab_decision_carries_the_price_it_was_admitted_on(monkeypatch):
    """Equality, not identity: the decision is rebuilt from the row's own block.

    The ladder wraps ``capacity_quote.price``, so the fit reaches it through
    ``slab_measured``. The round trip has to be exact or the card would quote
    one number and the probe another.
    """
    _slab_probe(monkeypatch, SLAB_FITS)
    assert _resolve(slab_weights=True).slab_fit == SLAB_FITS
    assert _resolve(memoized_family=lambda _path: "krea2").slab_fit == SLAB_FITS
    consented = _resolve(fit_probe=lambda _p, _o: DOES_NOT_FIT,
                         rescue_consent={"id": "abc"})
    assert consented.slab_fit == SLAB_FITS
    assert consented.fit == DOES_NOT_FIT


@pytest.mark.parametrize("overrides", [
    {"slab_weights": True},
    {"memoized_family": lambda _path: "krea2"},
    {"authoritative_slab_retry": True},
])
def test_a_slab_rung_that_does_not_fit_is_refused_and_offers_no_card(
        monkeypatch, overrides):
    """Every rung that returns slab meets the wall, and none of them gets a card.

    Slab is the last rung, so a consent button here could not change the
    answer. The refusal names the stock price beside the slab one so nobody
    retries with slab_weights=off into a bigger wall. The consented rescue rung
    is the fourth slab return and it takes the same wall, but it can only be
    reached through a stock miss, so a short slab answers it at the rescue seam
    below rather than here.
    """
    _slab_probe(monkeypatch, SLAB_SHORT)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        _resolve(**overrides)
    message = str(raised.value)
    assert "slab residency cannot load h3_fl2va_bf16.safetensors" in message
    assert "this load needs 67.0 GiB (a 62.0 GiB file plus a 5.0 GiB floor" in message
    assert "only 47.0 GiB of unified memory is available, 20.0 GiB short" in message
    stock_gib = capacity_fit.gib(capacity_fit.stock_required_bytes(SLAB_SHORT.size_bytes))
    assert f"it needs {stock_gib} GiB for the same file" in message
    assert "guard=slab_load_preflight waivable=0" in message
    assert consent_descriptor.DESCRIPTOR_BEGIN not in message
    assert not consent_descriptor.is_slab_rescue_offer(raised.value)
    assert "nothing was loaded and nothing was quarantined" in message


def test_a_consent_memo_is_not_spent_on_a_slab_that_does_not_fit(monkeypatch):
    """The consent is untouched, and the refusal is the plain capacity one.

    A grant cannot make unified memory appear, so a consented load whose slab
    is short reads the same sentence as an unconsented one and the grant is
    neither used nor revoked.
    """
    _slab_probe(monkeypatch, SLAB_SHORT)
    consent = {"id": "abc", "source": "panel"}
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        _resolve(fit_probe=lambda _p, _o: DOES_NOT_FIT, rescue_consent=consent)
    assert "zero-copy slab residency does not fit either" in str(raised.value)
    assert consent == {"id": "abc", "source": "panel"}


def test_the_rescue_seam_refuses_when_neither_residency_fits(monkeypatch):
    """The primary refusal is the stock price, and no card is offered.

    The blocker arm is checked last, after every policy answer, so a render
    that has both a policy blocker and a short slab reads the policy one.
    """
    _slab_probe(monkeypatch, SLAB_SHORT)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        _resolve(fit_probe=lambda _p, _o: DOES_NOT_FIT)
    message = str(raised.value)
    assert "stock residency cannot load h3_fl2va_bf16.safetensors" in message
    assert "Slab rescue is unavailable here because zero-copy slab residency does not fit" in message
    assert "it needs 67.0 GiB (the file plus a 5.0 GiB floor) against the same 47.0 GiB" in message
    assert "What would fit: 20.0 GiB more free memory" in message
    assert "guard=stock_load_preflight" in message
    assert consent_descriptor.DESCRIPTOR_BEGIN not in message
    assert not consent_descriptor.is_slab_rescue_offer(raised.value)

    # A policy blocker still wins: it is the reason no rescue exists.
    with pytest.raises(mesh_safety.StockLoadCapacityError) as policy:
        _resolve(fit_probe=lambda _p, _o: DOES_NOT_FIT, compile_dit=lambda: True)
    assert "compile_dit" in str(policy.value)
    assert "zero-copy slab residency does not fit" not in str(policy.value)


def test_the_slab_refusal_parses_back_to_the_numbers_it_priced(monkeypatch):
    from dgx_monarch import gate_audit_evidence

    _slab_probe(monkeypatch, SLAB_SHORT)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        _resolve(slab_weights=True)
    measured = gate_audit_evidence.parse_measured(str(raised.value))
    assert measured is not None
    assert measured["probe"] == "worker_slab_load"
    assert measured["checkpoint_bytes"] == SLAB_SHORT.size_bytes
    assert measured["mem_available_bytes"] == SLAB_SHORT.avail_bytes
    assert measured["headroom_bytes"] == (
        SLAB_SHORT.avail_bytes - SLAB_SHORT.required_bytes)
    assert measured["headroom_bytes"] < 0


def test_the_slab_price_is_never_larger_than_the_stock_price(tmp_path, monkeypatch):
    """The capped floor, swept. The arena cap keeps the slab price under the
    stock one at every size, and the absolute host floor sits above the cap on
    both prices, so 2.1 * S plus the floor is over S plus the floor at every S.
    """
    path = tmp_path / "w.safetensors"
    path.write_bytes(b"x" * 16)
    gib = 1 << 30
    monkeypatch.setattr(capacity_fit.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(capacity_fit.mesh_safety, "mem_available_bytes",
                        lambda: 200 * gib)
    sizes = [gib // 2, gib, 2 * gib, 4 * gib, int(4.7 * gib), 5 * gib,
             10 * gib, 24 * gib, 60 * gib, 120 * gib]
    for size in sizes:
        monkeypatch.setattr(capacity_fit.os.path, "getsize", lambda _p, s=size: s)
        slab = capacity_fit.slab_load_fit(str(path), {})
        stock = capacity_fit.stock_load_fit(str(path), {})
        assert slab.required_bytes <= stock.required_bytes, size
        # No reserve is set, so the absolute floor is the whole floor: the inner
        # term is the 4 GiB default clipped by the cap, always under it.
        assert slab.floor_bytes == capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES, size


def test_a_stored_dtype_the_slab_cannot_wrap_makes_no_claim(tmp_path, monkeypatch):
    """The skip protects the fallback: a dtype the slab cannot represent stock-loads.

    The fallback at ``comfy_bridge`` fires on the dtype the checkpoint stores,
    read from the same header ``WeightSlab.__init__`` reads, not on a requested
    cast.
    """
    path = tmp_path / "future.safetensors"
    header = json.dumps({
        "weight": {"dtype": "F6_E3M2", "shape": [4], "data_offsets": [0, 3]},
    }).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(3))
    monkeypatch.setattr(capacity_fit.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(capacity_fit.mesh_safety, "mem_available_bytes", lambda: 1)

    fit = capacity_fit.slab_load_fit(str(path), {})
    assert (fit.applies, fit.fits) == (False, True)
    assert "cannot wrap a dtype this checkpoint stores" in fit.skipped_reason

    # A cast is a different skip and names the annex, not the fallback.
    cast = capacity_fit.slab_load_fit(str(path), {"dtype": torch.bfloat16})
    assert (cast.applies, cast.fits) == (False, True)
    assert "annex" in cast.skipped_reason

    # A container this parser rejects outright is not a capability miss.
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"x" * 1024)
    monkeypatch.setattr(capacity_fit.os.path, "getsize", lambda _p: 8)
    monkeypatch.setattr(capacity_fit.mesh_safety, "mem_available_bytes", lambda: 1 << 40)
    assert capacity_fit.slab_load_fit(str(broken), {}).applies


def test_an_operator_reserve_above_the_floor_is_what_the_card_quotes(tmp_path, monkeypatch):
    from dgx_monarch.actor import store_slab_admit

    path = tmp_path / "w.safetensors"
    path.write_bytes(b"x" * 16)
    gib = 1 << 30
    monkeypatch.setattr(capacity_fit.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(capacity_fit.os.path, "getsize", lambda _p: 60 * gib)
    monkeypatch.setattr(capacity_fit.mesh_safety, "mem_available_bytes", lambda: 66 * gib)

    default = capacity_fit.slab_load_fit(str(path), {})
    assert default.floor_bytes == capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES
    assert default.fits
    reserved = capacity_fit.slab_load_fit(str(path), {}, reserve_bytes=10 * gib)
    assert reserved.floor_bytes == 10 * gib and not reserved.fits

    card = store_slab_admit.card(reserved, "w.safetensors", "this host",
                                reserve_bytes=10 * gib)
    assert "plus the 10.0 GiB this host reserves (uma_reserve_gb)" in card
    assert "above the default floor" in card
    assert "plus a 10.0 GiB floor" not in card
    # The larger of the two figures is charged, and the cap still applies.
    assert capacity_fit.slab_load_fit(
        str(path), {}, reserve_bytes=1
    ).floor_bytes == capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES


def test_a_reserve_the_arena_cap_clips_is_still_named_on_the_card(tmp_path, monkeypatch):
    """An operator who set the figure and reads a smaller one back needs both
    numbers, or the card looks like a host that never read the setting."""
    from dgx_monarch.actor import store_slab_admit

    path = tmp_path / "w.safetensors"
    path.write_bytes(b"x" * 16)
    gib = 1 << 30
    monkeypatch.setattr(capacity_fit.mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(capacity_fit.os.path, "getsize", lambda _p: 10 * gib)
    monkeypatch.setattr(capacity_fit.mesh_safety, "mem_available_bytes", lambda: 12 * gib)

    clipped = capacity_fit.slab_load_fit(str(path), {}, reserve_bytes=20 * gib)
    assert clipped.floor_bytes == int(capacity_fit.LEGACY_ARENA_RATIO * 10 * gib)
    card = store_slab_admit.card(clipped, "w.safetensors", "this host",
                                 reserve_bytes=20 * gib)
    assert "8.5 GiB floor" in card
    assert "20.0 GiB this host reserves (uma_reserve_gb)" in card
    assert "capped by the arena a stock load would have retained" in card


def test_the_operator_reserve_reaches_both_slab_seams(monkeypatch):
    """One floor, or the two walls of one load charge different prices.

    The rung prices the slab and ``preload_capacity_check`` prices it again at
    the last monarch-owned moment, so a reserve that reaches one and not the
    other admits at one wall and refuses at the next.
    """
    seen = []

    def probe(_path, _options, *, reserve_bytes=None, **_kw):
        seen.append(reserve_bytes)
        return SLAB_FITS

    monkeypatch.setattr(capacity_fit, "slab_load_fit", probe)
    gib = 1 << 30
    decision = _resolve(slab_weights=True, reserve_bytes=10 * gib)
    store_residency.preload_capacity_check(
        decision, "/models/m.safetensors", "m.safetensors", {},
        preflight=lambda *_a: None, reserve_bytes=10 * gib)
    assert seen == [10 * gib, 10 * gib]


def test_final_stock_wall_reprices_the_full_stock_fit(monkeypatch):
    """A memory drop after selection still charges the placement price and
    host floor."""
    short = StockFit(False, True, 10 * (1 << 30), 15 * (1 << 30))
    monkeypatch.setattr(capacity_fit, "stock_load_fit", lambda *_args: short)
    decision = store_residency.ResidencyDecision(
        False, store_residency.RUNG_STOCK_FITS, "stock", False)

    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        store_residency.preload_capacity_check(
            decision, "/models/m.safetensors", "m.safetensors", {},
            preflight=lambda *_args: pytest.fail("bare-file preflight ran"))

    assert f"{short.required_gib} GiB" in str(raised.value)
    assert "5.0 GiB host floor" in str(raised.value)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        pytest.param("bf16", "shards plus one block", id="bf16"),
        pytest.param("fp8", "direct wrapper keeps the full checkpoint live", id="fp8"),
        pytest.param("int8", "direct wrapper keeps the full checkpoint live", id="int8"),
        pytest.param(None, "checkpoint kind is unproven", id="unknown"),
    ],
)
def test_final_fsdp_wall_reprices_the_shared_kind_aware_price(
        monkeypatch, tmp_path, kind, expected):
    checkpoint = tmp_path / "m.safetensors"
    with open(checkpoint, "wb") as handle:
        handle.truncate(10 * (1 << 30))
    required = capacity_fit.fsdp_required_bytes(10 * (1 << 30), 2, kind)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: required - 1)
    decision = store_residency.ResidencyDecision(
        False, store_residency.RUNG_STOCK_FITS, "fsdp", False)

    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        store_residency.preload_capacity_check(
            decision, str(checkpoint), "m.safetensors", {},
            preflight=lambda *_args: pytest.fail("stock preflight ran"),
            fsdp_launch=True, fsdp_world=2, fsdp_checkpoint_kind=kind)

    from dgx_monarch.gate_audit_evidence import parse_measured

    measured = parse_measured(raised.value)
    assert measured is not None
    assert measured["checkpoint_bytes"] == 10 * (1 << 30)
    assert measured["mem_available_bytes"] == required - 1
    assert expected in str(raised.value)


def test_the_card_names_no_lever_it_cannot_prove_exists(monkeypatch):
    """Some families' FSDP path itself refuses, so the fsdp clause is conditional
    and no caller sets it today."""
    from dgx_monarch.actor import store_slab_admit

    plain = store_slab_admit.card(SLAB_SHORT, "m.safetensors", "this host")
    assert "fsdp" not in plain
    assert "fsdp preset on world 2" in store_slab_admit.card(
        SLAB_SHORT, "m.safetensors", "this host", fsdp_lever=True)
    assert "fsdp" not in store_slab_admit.rescue_blocker(SLAB_SHORT)[1]


def test_a_skipped_slab_probe_admits_and_claims_nothing(monkeypatch):
    """A probe that could not run makes no claim in either direction."""
    skipped = capacity_fit.SlabFit(True, False, 0, None, 0, "no meminfo here")
    _slab_probe(monkeypatch, skipped)
    decision = _resolve(slab_weights=True)
    assert decision.rung == store_residency.RUNG_EXPLICIT
    # The decision's reason names the rung the ladder took. A probe that made
    # no claim reports why to the journal and does not rewrite that sentence.
    assert decision.reason == "slab_weights=on was requested"
    assert decision.slab_fit is None
    with pytest.raises(SlabResidencyRescueOffer):
        _resolve(fit_probe=lambda _p, _o: DOES_NOT_FIT)


def test_the_stock_fits_rung_names_whether_it_actually_priced(monkeypatch):
    """``rung=stock_fits`` alone cannot tell a priced load from a skipped probe.

    The 2026-09-02 head is recorded as ``rung=stock_fits`` at 108.75 GiB
    available against a 111.0 GiB requirement, and nobody can say from that
    line which it was. With a decision, the line names the price or the skip.
    """
    _slab_probe(monkeypatch, SLAB_FITS)
    stored = SimpleNamespace(slab=None, residency_rung=store_residency.RUNG_STOCK_FITS)

    priced = store_residency.summary_line(
        stored, _resolve(fit_probe=lambda _p, _o: FITS))
    skipped = store_residency.summary_line(
        stored, _resolve(fit_probe=lambda _p, _o: SKIPPED))

    assert priced != skipped
    assert "rung=stock_fits" in priced
    assert f"priced={FITS.required_gib}GiB against 90.0GiB by worker_stock_load" in priced
    assert "reason=stock residency fits this checkpoint" in priced
    assert "rung=stock_skipped" in skipped
    assert " unpriced" in skipped
    assert "reason=a loader dtype cast changes the resident size" in skipped


def test_a_slab_rung_records_the_slab_probes_figure(monkeypatch):
    _slab_probe(monkeypatch, SLAB_FITS)
    stored = SimpleNamespace(slab=None, residency_rung=store_residency.RUNG_EXPLICIT)
    line = store_residency.summary_line(stored, _resolve(slab_weights=True))
    assert "priced=25.0GiB against 90.0GiB by worker_slab_load" in line
    assert "rung=explicit" in line


def test_the_fsdp_rung_records_the_shard_build_figure(monkeypatch, tmp_path):
    """The FSDP rung's wall prices in its own units.

    ``StockFit`` cannot carry a shard-build price, so the row's own block does,
    and the journal quotes it. Without it the line reads ``unpriced`` for a
    load the check measured.
    """
    from dgx_monarch.adapters import fsdp

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x" * 1000)
    _slab_probe(monkeypatch, SLAB_FITS)
    monkeypatch.setattr(fsdp, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(fsdp, "mem_available_bytes", lambda: 6 * (1 << 30))
    decision = _resolve(fsdp_launch=True, path=str(checkpoint),
                        unet_name="model.safetensors", world=2)

    stored = SimpleNamespace(slab=None, residency_rung=store_residency.RUNG_STOCK_FITS)
    line = store_residency.summary_line(stored, decision)
    assert " unpriced" not in line
    assert "by stock_load_preflight" in line
    assert "reason=FSDP launch" in line


def test_the_summary_line_without_a_decision_is_unchanged():
    """Without a decision, a stock load's line holds only the residency and rung
    fields: no price and no reason."""
    stored = SimpleNamespace(slab=None, residency_rung=store_residency.RUNG_STOCK_FITS)
    assert store_residency.summary_line(stored) == "residency=stock rung=stock_fits"


def test_a_cold_box_reads_as_cold_rather_than_as_full(monkeypatch):
    """The memo was read here and came back empty, so say so on the card.

    Residency could not be resolved from this box's own record. Another rank
    that has loaded the same file may resolve differently, and an operator
    reading two different answers for one render needs the reason.
    """
    _slab_probe(monkeypatch, SLAB_FITS)
    with pytest.raises(SlabResidencyRescueOffer) as raised:
        _resolve(fit_probe=lambda _p, _o: DOES_NOT_FIT)
    message = str(raised.value)
    assert "This host has no record of loading this file" in message
    assert "another rank that has loaded it may resolve differently" in message
    # The card never claims a later load "needs no card": consenting once writes
    # a row, but an unvouched detected family returns the identical card.
    assert "needs no card" not in message


@pytest.mark.parametrize("overrides", [
    {"memoized_family": lambda _path: "minimax_h3"},
    {"family_override": "krea2"},
])
def test_the_cold_clause_is_absent_wherever_the_memo_was_not_the_reason(
        monkeypatch, overrides):
    """The clause is set only in the branch that read the memo.

    With an unvouched family the memo answered, so the box is not cold. A
    forced family never reads the memo: the override turns auto off before the
    vouched rung is tried.
    """
    _slab_probe(monkeypatch, SLAB_FITS)
    with pytest.raises(SlabResidencyRescueOffer) as raised:
        _resolve(fit_probe=lambda _p, _o: DOES_NOT_FIT, **overrides)
    assert "no record of loading this file" not in str(raised.value)


def test_the_cold_clause_never_reaches_a_refusal_that_offers_nothing(monkeypatch):
    """A card clause on a refusal with no card is noise an operator cannot use."""
    _slab_probe(monkeypatch, SLAB_FITS)
    with pytest.raises(mesh_safety.StockLoadCapacityError) as raised:
        _resolve(fit_probe=lambda _p, _o: DOES_NOT_FIT, lora_low_rss=False)
    assert "no record of loading this file" not in str(raised.value)
    assert _resolve(slab_weights=True).rung == store_residency.RUNG_EXPLICIT


def test_the_ladder_table_rows_869_and_873_still_read_as_the_record_has_them(
        monkeypatch):
    """Ladder rows 869 and 873, asserted together.

    Row 869: auto with a vouched family takes the vouched auto rung and loads
    slab; the slab probe is called and the stock fit probe is not. Row 873: auto
    with an unvouched family gets the rescue offer, and no consent is invented
    from a memo row.
    """
    slab_calls = _slab_probe(monkeypatch, SLAB_FITS)
    stock_calls = []
    row_869 = _resolve(memoized_family=lambda _path: "krea2",
                       fit_probe=lambda p, o: stock_calls.append(p) or FITS)
    assert (row_869.use_slab, row_869.rung) == (True, store_residency.RUNG_VOUCHED_AUTO)
    assert stock_calls == [] and len(slab_calls) == 1

    with pytest.raises(SlabResidencyRescueOffer) as raised:
        _resolve(memoized_family=lambda _path: "minimax_h3",
                 fit_probe=lambda _p, _o: DOES_NOT_FIT)
    assert consent_descriptor.is_slab_rescue_offer(raised.value)
    assert "minimax_h3" not in ms.SLAB_VOUCHED_FAMILIES
