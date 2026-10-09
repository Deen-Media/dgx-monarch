"""Regression coverage for identity, output ordering, masking and worker lifecycle.

Cross-rank identity detects content changes that preserve mean and standard
deviation. DP outputs join in dp_rank order. USP accepts only no-op masks.
Global unload marks a surviving slot for reload. Worker-loop Python paths
remain quoted and support home-directory expansion.
"""
import pytest
import torch

from dgx_monarch.actor.worker import _latent_signature
from dgx_monarch.adapters.base import _mask_is_noop
from dgx_monarch.cli.lifecycle import _loop_args, _pybin_shell
from dgx_monarch.config import HostConfig
from dgx_monarch.nodes import render_result, render_validation
from dgx_monarch.topology import Topology


def test_latent_signature_has_full_digest_and_projection():
    sig = _latent_signature(torch.arange(4096, dtype=torch.float32).reshape(1, 4, 32, 32))
    assert sig["shape"] == [1, 4, 32, 32]
    assert sig["dtype"] == "float32" and sig["numel"] == 4096
    assert "mean" in sig and "std" in sig
    assert len(sig["sha256"]) == 64
    assert len(sig["projection"]) == 64


def test_permutation_shares_stats_but_differs_in_full_content_fields():
    t = torch.randn(1, 4, 64, 64)
    perm = t.flatten()[torch.randperm(t.numel())].reshape(t.shape)
    a, b = _latent_signature(t), _latent_signature(perm)
    # A permutation preserves mean/std but changes exact and tolerant content
    # signatures because every tensor position contributes.
    assert a["mean"] == b["mean"] and a["std"] == b["std"]
    assert a["sha256"] != b["sha256"]
    assert a["projection"] != b["projection"]


def test_swap_outside_old_probe_changes_full_projection():
    t = torch.arange(4096, dtype=torch.float32)
    changed = t.clone()
    # 100 and 101 fall between the positions a 64-point evenly spaced probe would read.
    changed[100], changed[101] = changed[101].clone(), changed[100].clone()
    a, b = _latent_signature(t), _latent_signature(changed)
    assert a["mean"] == b["mean"] and a["std"] == b["std"]
    assert a["sha256"] != b["sha256"]
    assert a["projection"] != b["projection"]


def _finish(results, dp, monkeypatch, *, world=None):
    monkeypatch.setattr(render_result, "read_latent_result", lambda v: v)  # v is the tensor itself
    world = len(results) if world is None else world
    topo = Topology(ulysses=world // dp, dp=dp, world=world)
    return render_result._finish_render(results, topo, {})


def _different_digest(signature):
    zeros = "0" * 64
    return zeros if signature["sha256"] != zeros else "1" * 64


def test_identity_gate_rejects_incomplete_rank_inventory_before_output_handling(monkeypatch):
    t = torch.randn(1, 4, 8, 8)
    results = [{
        "rank": 0,
        "dp_rank": 0,
        "latent": t,
        "latent_stats": _latent_signature(t),
    }]

    with pytest.raises(RuntimeError, match="expected 2 rank results"):
        _finish(results, dp=1, world=2, monkeypatch=monkeypatch)


def test_identity_gate_rejects_non_mapping_rank_result(monkeypatch):
    with pytest.raises(RuntimeError, match="index 0 is not a mapping"):
        _finish([object()], dp=1, monkeypatch=monkeypatch)


@pytest.mark.parametrize(
    ("field", "value"),
    [("rank", None), ("rank", True), ("dp_rank", None), ("dp_rank", False)],
)
def test_identity_gate_requires_exact_integer_rank_identities(
    monkeypatch, field, value,
):
    t = torch.randn(1, 4, 8, 8)
    result = {
        "rank": 0,
        "dp_rank": 0,
        "latent": t,
        "latent_stats": _latent_signature(t),
    }
    result[field] = value

    with pytest.raises(RuntimeError, match="non-integer"):
        _finish([result], dp=1, monkeypatch=monkeypatch)


def test_identity_gate_rejects_duplicate_global_rank_inventory(monkeypatch):
    t = torch.randn(1, 4, 8, 8)
    sig = _latent_signature(t)
    results = [
        {"rank": 0, "dp_rank": 0, "latent": t, "latent_stats": sig},
        {"rank": 0, "dp_rank": 0, "latent": None, "latent_stats": sig},
    ]

    with pytest.raises(RuntimeError, match="incomplete or duplicated"):
        _finish(results, dp=1, monkeypatch=monkeypatch)


def test_identity_gate_rejects_missing_dp_group(monkeypatch):
    a, b = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
    results = [
        {"rank": 0, "dp_rank": 1, "latent": a, "latent_stats": _latent_signature(a)},
        {"rank": 1, "dp_rank": 1, "latent": b, "latent_stats": _latent_signature(b)},
    ]

    with pytest.raises(RuntimeError, match="expected DP ranks"):
        _finish(results, dp=2, monkeypatch=monkeypatch)


def test_identity_gate_rejects_split_dp_identity_in_model_parallel_topology(monkeypatch):
    t = torch.randn(1, 4, 8, 8)
    sig = _latent_signature(t)
    results = [
        {"rank": 0, "dp_rank": 0, "latent": t, "latent_stats": sig},
        {"rank": 1, "dp_rank": 1, "latent": None, "latent_stats": sig},
    ]

    # world2/dp1 is an Ulysses- or Ring-shaped model-parallel topology. Trusting
    # a worker that claims dp1 would split the signatures into singleton groups
    # and skip the cross-rank comparison in silence.
    with pytest.raises(RuntimeError, match=r"expected DP ranks \[0\], got \[0, 1\]"):
        _finish(results, dp=1, monkeypatch=monkeypatch)


def test_identity_gate_rejects_wrong_dp_group_cardinality(monkeypatch):
    tensors = [torch.randn(1, 4, 8, 8) for _ in range(4)]
    results = [
        {
            "rank": rank,
            "dp_rank": 0 if rank < 3 else 1,
            "latent": tensor if rank in (0, 3) else None,
            "latent_stats": _latent_signature(tensor),
        }
        for rank, tensor in enumerate(tensors)
    ]

    with pytest.raises(RuntimeError, match="DP group 0 returned 3 rank results, expected 2"):
        _finish(results, dp=2, monkeypatch=monkeypatch)


def test_identity_gate_rejects_balanced_world2_dp_rank_swap(monkeypatch):
    tensors = [torch.randn(1, 4, 8, 8) for _ in range(2)]
    results = [
        {
            "rank": rank,
            "dp_rank": 1 - rank,
            "latent": tensor,
            "latent_stats": _latent_signature(tensor),
        }
        for rank, tensor in enumerate(tensors)
    ]

    with pytest.raises(RuntimeError, match="global rank 1 reported DP rank 0, expected 1"):
        _finish(results, dp=2, monkeypatch=monkeypatch)


def test_identity_gate_rejects_balanced_world4_dp_group_swap(monkeypatch):
    tensors = [torch.randn(1, 4, 8, 8) for _ in range(4)]
    results = [
        {
            "rank": rank,
            "dp_rank": 1 if rank < 2 else 0,
            "latent": tensor if rank in (0, 2) else None,
            "latent_stats": _latent_signature(tensor),
        }
        for rank, tensor in enumerate(tensors)
    ]

    with pytest.raises(RuntimeError, match="global rank 2 reported DP rank 0, expected 1"):
        _finish(results, dp=2, monkeypatch=monkeypatch)


def test_identity_gate_accepts_exact_world4_dp_mapping(monkeypatch):
    slices = [torch.zeros(1, 4, 8, 8), torch.ones(1, 4, 8, 8)]
    signatures = [_latent_signature(tensor) for tensor in slices]
    results = [
        {"rank": 0, "dp_rank": 0, "latent": slices[0], "latent_stats": signatures[0]},
        {"rank": 1, "dp_rank": 0, "latent": None, "latent_stats": signatures[0]},
        {"rank": 2, "dp_rank": 1, "latent": slices[1], "latent_stats": signatures[1]},
        {"rank": 3, "dp_rank": 1, "latent": None, "latent_stats": signatures[1]},
    ]

    out = _finish(results, dp=2, monkeypatch=monkeypatch)
    assert torch.equal(out["samples"], torch.cat(slices))


@pytest.mark.parametrize("stats", [None, {}, [], [{}], [{"ok": True}, {}]])
def test_identity_gate_rejects_empty_or_malformed_signature_evidence(monkeypatch, stats):
    results = [{"rank": 0, "dp_rank": 0, "latent": None, "latent_stats": stats}]

    with pytest.raises(RuntimeError, match="missing or malformed latent signatures"):
        _finish(results, dp=1, monkeypatch=monkeypatch)


def test_identity_gate_rejects_incomplete_world_one_signature_schema(monkeypatch):
    results = [{
        "rank": 0,
        "dp_rank": 0,
        "latent": None,
        "latent_stats": {"present": True},
    }]

    with pytest.raises(RuntimeError, match="missing or malformed latent signatures"):
        _finish(results, dp=1, monkeypatch=monkeypatch)


def test_identity_gate_rejects_malformed_dp_singleton_signature(monkeypatch):
    tensor = torch.randn(1, 4, 8, 8)
    results = [
        {
            "rank": 0,
            "dp_rank": 0,
            "latent": tensor,
            "latent_stats": _latent_signature(tensor),
        },
        {
            "rank": 1,
            "dp_rank": 1,
            "latent": tensor,
            "latent_stats": {"shape": list(tensor.shape)},
        },
    ]

    with pytest.raises(RuntimeError, match="missing or malformed latent signatures"):
        _finish(results, dp=2, monkeypatch=monkeypatch)


def test_identity_gate_rejects_paired_empty_digest_and_projection(monkeypatch):
    tensor = torch.randn(1, 4, 8, 8)
    signature = _latent_signature(tensor)
    signature["sha256"] = ""
    signature["projection"] = []
    results = [
        {"rank": 0, "dp_rank": 0, "latent": tensor, "latent_stats": signature},
        {"rank": 1, "dp_rank": 0, "latent": None, "latent_stats": signature},
    ]

    with pytest.raises(RuntimeError, match="missing or malformed latent signatures"):
        _finish(results, dp=1, monkeypatch=monkeypatch)


@pytest.mark.parametrize(
    "case",
    [
        "missing_key",
        "bool_dimension",
        "negative_dimension",
        "numel_mismatch",
        "empty_dtype",
        "bool_numel",
        "nonfinite_mean",
        "negative_std",
        "short_digest",
        "uppercase_digest",
        "nonhex_digest",
        "short_projection",
        "bool_projection",
        "nonfinite_projection",
    ],
)
def test_identity_gate_rejects_noncanonical_signature_fields(monkeypatch, case):
    tensor = torch.randn(1, 4, 8, 8)
    signature = _latent_signature(tensor)
    if case == "missing_key":
        signature.pop("sha256")
    elif case == "bool_dimension":
        signature["shape"] = [True, *signature["shape"][1:]]
    elif case == "negative_dimension":
        signature["shape"] = [-1, *signature["shape"][1:]]
    elif case == "numel_mismatch":
        signature["numel"] += 1
    elif case == "empty_dtype":
        signature["dtype"] = ""
    elif case == "bool_numel":
        signature["numel"] = True
    elif case == "nonfinite_mean":
        signature["mean"] = float("nan")
    elif case == "negative_std":
        signature["std"] = -1.0
    elif case == "short_digest":
        signature["sha256"] = "0" * 63
    elif case == "uppercase_digest":
        signature["sha256"] = "A" * 64
    elif case == "nonhex_digest":
        signature["sha256"] = "g" * 64
    elif case == "short_projection":
        signature["projection"] = signature["projection"][:-1]
    elif case == "bool_projection":
        signature["projection"][0] = True
    elif case == "nonfinite_projection":
        signature["projection"][0] = float("inf")

    results = [{"rank": 0, "dp_rank": 0, "latent": tensor, "latent_stats": signature}]
    with pytest.raises(RuntimeError, match="missing or malformed latent signatures"):
        _finish(results, dp=1, monkeypatch=monkeypatch)


def test_identity_gate_does_not_echo_malformed_signature_values(monkeypatch):
    private_value = "private-artifact-identifier"
    results = [{
        "rank": 0,
        "dp_rank": 0,
        "latent": None,
        "latent_stats": private_value,
    }]

    with pytest.raises(RuntimeError, match="malformed latent signatures") as error:
        _finish(results, dp=1, monkeypatch=monkeypatch)
    assert private_value not in str(error.value)


def test_identity_gate_accepts_world_one_with_complete_signature(monkeypatch):
    t = torch.randn(1, 4, 8, 8)
    results = [{
        "rank": 0,
        "dp_rank": 0,
        "latent": t,
        "latent_stats": _latent_signature(t),
    }]

    out = _finish(results, dp=1, monkeypatch=monkeypatch)
    assert torch.equal(out["samples"], t)


@pytest.mark.parametrize("tensor", [torch.tensor(2.0), torch.empty(0, 3)])
def test_identity_gate_accepts_scalar_and_zero_numel_signatures(tensor):
    results = [{
        "rank": 0,
        "dp_rank": 0,
        "latent": tensor,
        "latent_stats": _latent_signature(tensor),
    }]

    render_validation.verify_cross_rank_signatures(results, Topology())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mean", 1.0),
        ("std", 1.0),
        ("sha256", "0" * 64),
        ("projection", [1.0, *([0.0] * 63)]),
    ],
)
def test_identity_gate_rejects_impossible_zero_numel_signature(field, value):
    tensor = torch.empty(0, 3)
    signature = _latent_signature(tensor)
    signature[field] = value
    results = [{
        "rank": 0,
        "dp_rank": 0,
        "latent": tensor,
        "latent_stats": signature,
    }]

    with pytest.raises(RuntimeError, match="missing or malformed latent signatures"):
        render_validation.verify_cross_rank_signatures(results, Topology())


def test_identity_gate_rejects_impossible_single_element_std():
    tensor = torch.tensor([2.0])
    signature = _latent_signature(tensor)
    signature["std"] = 1.0
    results = [{
        "rank": 0,
        "dp_rank": 0,
        "latent": tensor,
        "latent_stats": signature,
    }]

    with pytest.raises(RuntimeError, match="missing or malformed latent signatures"):
        render_validation.verify_cross_rank_signatures(results, Topology())


def test_identity_gate_rejects_empty_digest_for_nonempty_signature():
    tensor = torch.tensor([2.0])
    signature = _latent_signature(tensor)
    signature["sha256"] = _latent_signature(torch.empty(0))["sha256"]
    results = [{
        "rank": 0,
        "dp_rank": 0,
        "latent": tensor,
        "latent_stats": signature,
    }]

    with pytest.raises(RuntimeError, match="missing or malformed latent signatures"):
        render_validation.verify_cross_rank_signatures(results, Topology())


def test_identity_gate_fails_on_permutation(monkeypatch):
    # With dp 1 every rank gathers the same full latent and only the leader
    # returns it, so equal shape, mean and std with different content must raise.
    t = torch.randn(1, 4, 32, 32)
    perm = t.flatten()[torch.randperm(t.numel())].reshape(t.shape)
    results = [
        {"rank": 0, "dp_rank": 0, "latent": t, "latent_stats": _latent_signature(t)},
        {"rank": 1, "dp_rank": 0, "latent": None, "latent_stats": _latent_signature(perm)},
    ]
    try:
        _finish(results, dp=1, monkeypatch=monkeypatch)
        raise AssertionError("gate accepted a permuted-content divergence")
    except RuntimeError as e:
        assert "identity FAILED" in str(e)


def test_identity_gate_passes_on_true_match(monkeypatch):
    t = torch.randn(1, 4, 32, 32)
    sig = _latent_signature(t)
    results = [  # leader returns the latent; the rest report stats only
        {"rank": 0, "dp_rank": 0, "latent": t, "latent_stats": sig},
        {"rank": 1, "dp_rank": 0, "latent": None, "latent_stats": sig},
        {"rank": 2, "dp_rank": 0, "latent": None, "latent_stats": sig},
    ]
    out = _finish(results, dp=1, monkeypatch=monkeypatch)
    assert torch.equal(out["samples"], t)


def test_identity_gate_accepts_documented_cross_box_wobble(monkeypatch):
    t = torch.randn(1, 4, 32, 32)
    reference = _latent_signature(t)
    near = dict(reference)
    near["sha256"] = _different_digest(reference)
    near["projection"] = list(reference["projection"])
    near["projection"][0] += 8e-4 * max(abs(reference["std"]), 1e-3)
    results = [
        {"rank": 0, "dp_rank": 0, "latent": t, "latent_stats": reference},
        {"rank": 1, "dp_rank": 0, "latent": None, "latent_stats": near},
    ]
    out = _finish(results, dp=1, monkeypatch=monkeypatch)
    assert torch.equal(out["samples"], t)


def test_identity_gate_rejects_just_above_cross_box_tolerance(monkeypatch):
    t = torch.randn(1, 4, 32, 32)
    reference = _latent_signature(t)
    outside = dict(reference)
    outside["sha256"] = _different_digest(reference)
    outside["projection"] = list(reference["projection"])
    outside["projection"][0] += 1.2e-3 * max(abs(reference["std"]), 1e-3)
    results = [
        {"rank": 0, "dp_rank": 0, "latent": t, "latent_stats": reference},
        {"rank": 1, "dp_rank": 0, "latent": None, "latent_stats": outside},
    ]
    with pytest.raises(RuntimeError, match="identity FAILED"):
        _finish(results, dp=1, monkeypatch=monkeypatch)


@pytest.mark.parametrize(
    ("field", "value"),
    [("mean", float("nan")), ("std", float("inf")), ("projection", float("nan"))],
)
def test_identity_gate_rejects_non_finite_signature_values(monkeypatch, field, value):
    t = torch.randn(1, 4, 32, 32)
    reference = _latent_signature(t)
    invalid = dict(reference)
    invalid["sha256"] = _different_digest(reference)
    invalid["projection"] = list(reference["projection"])
    if field == "projection":
        invalid["projection"][0] = value
    else:
        invalid[field] = value
    results = [
        {"rank": 0, "dp_rank": 0, "latent": t, "latent_stats": reference},
        {"rank": 1, "dp_rank": 0, "latent": None, "latent_stats": invalid},
    ]
    with pytest.raises(RuntimeError, match="identity FAILED"):
        _finish(results, dp=1, monkeypatch=monkeypatch)


def test_identity_gate_rejects_swap_outside_the_old_probe(monkeypatch):
    t = torch.arange(4096, dtype=torch.float32).reshape(1, 4, 32, 32)
    changed = t.clone().flatten()
    # Neither position is one a 64-point evenly spaced probe would read, and the
    # distant swap sits well above the documented 1e-3 cross-box tolerance.
    changed[100], changed[3000] = changed[3000].clone(), changed[100].clone()
    changed = changed.reshape(t.shape)
    results = [
        {"rank": 0, "dp_rank": 0, "latent": t, "latent_stats": _latent_signature(t)},
        {"rank": 1, "dp_rank": 0, "latent": None,
         "latent_stats": _latent_signature(changed)},
    ]
    with pytest.raises(RuntimeError, match="identity FAILED"):
        _finish(results, dp=1, monkeypatch=monkeypatch)


def test_identity_gate_checks_ranks_within_each_dp_group(monkeypatch):
    a = torch.randn(1, 4, 8, 8)
    b = torch.randn(1, 4, 8, 8)
    divergent = b.flip(-1)
    results = [
        {"rank": 0, "dp_rank": 0, "latent": a, "latent_stats": _latent_signature(a)},
        {"rank": 1, "dp_rank": 0, "latent": None, "latent_stats": _latent_signature(a)},
        {"rank": 2, "dp_rank": 1, "latent": b, "latent_stats": _latent_signature(b)},
        {"rank": 3, "dp_rank": 1, "latent": None,
         "latent_stats": _latent_signature(divergent)},
    ]
    with pytest.raises(RuntimeError, match="dp group 1"):
        _finish(results, dp=2, monkeypatch=monkeypatch)


def test_identity_gate_checks_every_packed_modality(monkeypatch):
    video = torch.randn(1, 4, 8, 8)
    audio = torch.randn(1, 2, 16)
    divergent_audio = audio.flip(-1)
    results = [
        {"rank": 0, "dp_rank": 0, "latent": video,
         "latent_stats": [_latent_signature(video), _latent_signature(audio)]},
        {"rank": 1, "dp_rank": 0, "latent": None,
         "latent_stats": [_latent_signature(video), _latent_signature(divergent_audio)]},
    ]
    with pytest.raises(RuntimeError, match="modality 1"):
        _finish(results, dp=1, monkeypatch=monkeypatch)


def test_dp_leaders_concatenate_in_dp_rank_order(monkeypatch):
    # Leaders arrive reversed; the batch must reconstruct in dp_rank order.
    slice0 = torch.zeros(1, 4, 8, 8)
    slice1 = torch.ones(1, 4, 8, 8)
    results = [
        {"rank": 1, "dp_rank": 1, "latent": slice1, "latent_stats": _latent_signature(slice1)},
        {"rank": 0, "dp_rank": 0, "latent": slice0, "latent_stats": _latent_signature(slice0)},
    ]
    # Singleton DP groups have no pairwise comparison, but their complete rank
    # inventory and signature evidence are still validated.
    out = _finish(results, dp=2, monkeypatch=monkeypatch)
    assert torch.equal(out["samples"][0], slice0[0])  # dp_rank 0 first
    assert torch.equal(out["samples"][1], slice1[0])  # dp_rank 1 second


def test_mask_is_noop():
    assert _mask_is_noop(torch.zeros(1, 1, 4, 4))              # additive all-zero
    assert _mask_is_noop(torch.ones(1, 1, 4, 4, dtype=torch.bool))  # boolean all-True
    assert not _mask_is_noop(torch.tensor([[0.0, -1e4]]))      # additive, masks a key
    assert not _mask_is_noop(torch.tensor([[True, False]]))    # boolean, drops a key


def test_store_marks_other_slot_gpu_evicted():
    from dgx_monarch.actor.model_store import ModelStore
    s = ModelStore()
    s.current, s.uncond = object(), object()  # both slots resident
    # Dropping or hot-swapping cond runs comfy's global unload, which evicts
    # uncond's GPU weights too; cond re-materializes here.
    s._mark_other_slot_evicted("cond")
    assert s._gpu_evicted == {"uncond"}
    # a reuse of uncond would clear it (that reuse pays the reload)
    s._gpu_evicted.discard("uncond")
    # nothing to flag when the other slot is empty
    solo = ModelStore()
    solo.current = object()
    solo._mark_other_slot_evicted("cond")
    assert solo._gpu_evicted == set()


def test_injected_model_is_gc_only_and_needs_pin_release():
    # USP injection binds diffusion_model._forward = MethodType(fn, model), a
    # bound method stored on the object it points back to: a reference cycle,
    # so only the garbage collector frees the model. ensure() must not keep a
    # local pinning the outgoing model across _drop's gc.collect(), or it stays
    # resident (~2x memory) until the next model change. This reproduces the
    # cycle and checks that releasing the pin lets a collect reclaim it.
    import gc
    import types
    import weakref

    class _Model:  # stand-in for an injected diffusion_model
        pass

    def _fake_forward(self):
        return None

    m = _Model()
    m._forward = types.MethodType(_fake_forward, m)   # the injection cycle
    ref = weakref.ref(m)
    stored = m                                        # ensure's local pin
    del m

    gc.collect()
    assert ref() is not None, "a pinned injected model must survive a collect (the bug condition)"
    del stored                                        # the fix: release the pin before the drop's collect
    gc.collect()
    assert ref() is None, "once the pin is released, the collect must reclaim the injected cycle"


def test_merge_free_clears_backup_patches_and_buffers():
    # low_rss is merge-and-free: bake, then drop the ~model-sized backup and the
    # baked patch specs so re-loads never double-bake. Guard the exact fields it
    # frees (a fake patcher stands in for comfy's ModelPatcher; load_models_gpu is
    # the only comfy call, patched out here).
    import dgx_monarch.actor.model_store as ms

    class FakeModel:
        pass

    class FakePatcher:
        def __init__(self):
            self.model = FakeModel()
            self.backup = {"a": 1, "b": 2}
            self.backup_buffers = {"c": 3}
            self.patches = {"a": [1], "b": [2]}

    calls = {}

    class FakeMM:
        def load_models_gpu(self, models, force_full_load=False):
            calls["loaded"] = (list(models), force_full_load)

    import sys
    # Install both names and put both back exactly. With setdefault, a loaded
    # real comfy package would stay and the import would reach the real
    # model_management through the package attribute, not this fake. A bare
    # `comfy` left behind makes `import comfy.options` fail in later real-comfy
    # files ("'comfy' is not a package"), and the conftest comfy module guard
    # reports the leak against this file.
    stubbed = ("comfy", "comfy.model_management")
    before = {name: sys.modules[name] for name in stubbed if name in sys.modules}
    sys.modules["comfy"] = type(sys)("comfy")
    fake_mm = FakeMM()
    sys.modules["comfy.model_management"] = fake_mm
    try:
        p = FakePatcher()
        ms._merge_and_free(p)
        assert calls["loaded"][1] is True            # the bake is a forced full load
        assert p.backup == {} and p.backup_buffers == {} and p.patches == {}
    finally:
        for name in stubbed:
            sys.modules.pop(name, None)
        sys.modules.update(before)


def test_nccl_key_is_shape_only_not_worker_args():
    # Flipping a memory toggle (lora_low_rss) must not change the NCCL setup
    # key; otherwise the mesh tears down live USP groups mid-session and the
    # next render dies on a stale process group. The key is a pure function of
    # the parallel shape; worker_args can't reach it.
    import inspect

    from dgx_monarch.mesh import _nccl_setup_key
    params = set(inspect.signature(_nccl_setup_key).parameters)
    assert "worker_args" not in params
    assert {"topology", "attention", "sync_ulysses"} == params


def test_worker_has_apply_worker_args_endpoint():
    # the mesh calls workers.apply_worker_args when only worker_args change
    # (no NCCL teardown); the endpoint must exist or that path AttributeErrors.
    from dgx_monarch.actor.worker import GPUWorker
    assert hasattr(GPUWorker, "apply_worker_args")


def test_build_active_is_bound_instance_method():
    # _build_active reads self.lora_low_rss and must be a bound method.
    # Calling it needs ComfyUI, so this unit test checks the declaration.
    import inspect

    from dgx_monarch.actor.model_store import ModelStore
    raw = inspect.getattr_static(ModelStore, "_build_active")
    assert not isinstance(raw, staticmethod), "_build_active must be a bound method, not static"
    assert next(iter(inspect.signature(ModelStore._build_active).parameters)) == "self"


def test_store_lora_mode_switch():
    from dgx_monarch.actor.model_store import ModelStore
    s = ModelStore()
    assert s.lora_low_rss is False          # default: hot_swap (baked)
    s.set_lora_mode(True)                    # no resident model: flips without a drop
    assert s.lora_low_rss is True
    s.set_lora_mode(True)                    # idempotent
    assert s.lora_low_rss is True
    s.set_lora_mode(False)
    assert s.lora_low_rss is False


def test_loop_args_is_the_pgrep_tail():
    host = HostConfig(name="h", address="tcp://192.0.2.11:26600")
    # No python binary here: the script prepares it. The argv tail ends with
    # worker_health.loop_pattern, which loop_regex anchors for pgrep and pkill.
    assert _loop_args(host) == "-m dgx_monarch.cli.worker_loop --address tcp://192.0.2.11:26600"


def test_pybin_shell_quotes_spaces_and_expands_tilde():
    # A ~/venv path must launch: shlex.quote alone suppresses ~ expansion, and
    # the worker would not start.
    snip = _pybin_shell("~/monarch-env/bin/python")
    assert "PYBIN='~/monarch-env/bin/python'" in snip     # quoted
    assert '"~/"*) PYBIN="$HOME/${PYBIN#\\~/}"' in snip    # ...then ~-expanded in shell
    # a path with spaces is quoted as one token
    assert "PYBIN='/home/u/my env/bin/python'" in _pybin_shell("/home/u/my env/bin/python")
