"""One activation scale for the sharded group.

CPU only, no CUDA and no process group: the reducers are callables the plan
carries, so two fake ranks run in one process and the collective is a function
that stashes each caller's amax and replays the max.

comfy itself is not importable in the unit environment, so the fake Linear
below reproduces the two lines of comfy's forward that matter
(``getattr(self, 'input_scale', None)`` then ``QuantizedTensor.from_float``),
and the canary in tests/canary/comfy_seam_contracts.py checks those facts
against real comfy where it is importable.
"""
from __future__ import annotations

import pytest
import torch

ck = pytest.importorskip("comfy_kitchen.tensor")

from dgx_monarch.adapters import cfg_dispatch  # noqa: E402
from dgx_monarch.adapters import quant_activation_scale as qas  # noqa: E402

QuantizedTensor = ck.QuantizedTensor
NVFP4 = "TensorCoreNVFP4Layout"


@pytest.fixture(autouse=True)
def _cfg_pair_folds_starts_true():
    """Reset the fold fact to its default (fold) around each test, as
    `_restore_the_knobs` below does for the two variables."""
    cfg_dispatch.record_cfg_pair_lengths([])
    yield
    cfg_dispatch.record_cfg_pair_lengths([])


@pytest.fixture(autouse=True)
def _restore_the_knobs():
    """Both variables come back exactly as they were, set or unset.

    monkeypatch.delenv records nothing for a key that was already absent, so a
    test that writes one would otherwise leak it into every test after it.
    """
    import os

    before = {name: os.environ.get(name) for name in (qas.ENABLE_ENV, qas.LOG_ENV)}
    yield
    for name, value in before.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class _Linear(torch.nn.Module):
    """comfy's quantized Linear, reduced to the gate and the two lines under test."""

    def __init__(self, layout_type=NVFP4, *, input_scale=None, pre_quant_scale=None,
                 weight_function=(), quantizes=True):
        super().__init__()
        self.layout_type = layout_type
        self._full_precision_mm = False
        self.comfy_force_cast_weights = False
        self.weight_function = list(weight_function)
        self.bias_function = []
        self._quantizes = quantizes
        if input_scale is not None:
            self.input_scale = input_scale
        if pre_quant_scale is not None:
            self.pre_quant_scale = pre_quant_scale

    def forward(self, value):
        if not self._quantizes or self.layout_type != NVFP4:
            return value
        flat = value.reshape(-1, value.shape[-1])
        scale = getattr(self, "input_scale", None)
        return QuantizedTensor.from_float(flat, self.layout_type, scale=scale).dequantize()


class _Experts(_Linear):
    """comfy's MoEExperts: the same layout, but it quantizes with no scale.

    Its `num_experts` attribute marks it, so the installer declines it
    (`_takes_an_input_scale` in adapters/quant_activation_scale.py says why).
    """

    def __init__(self, num_experts=4):
        super().__init__(NVFP4)
        self.num_experts = num_experts

    def forward(self, value):
        flat = value.reshape(-1, value.shape[-1])
        return QuantizedTensor.from_float(flat, self.layout_type).dequantize()


class _Tree(torch.nn.Module):
    def __init__(self, **children):
        super().__init__()
        for name, child in children.items():
            setattr(self, name, child)


def _collecting_reducer():
    """A reducer that collects each rank's amax, then replays the max of them."""
    seen: list[torch.Tensor] = []

    def reduce(tensor):
        seen.append(tensor.clone())
        return torch.stack(seen).amax(dim=0) if len(seen) > 1 else tensor

    return reduce, seen


def _replay_reducer(shared):
    def reduce(tensor):
        return shared.clone()

    return reduce


def _in_place_reducer(shared):
    """The shape xfuser's all_reduce has: it modifies its input and returns it."""
    def reduce(tensor):
        tensor.copy_(torch.maximum(tensor, shared))
        return tensor

    return reduce


def _full_tensor_reference(x):
    return QuantizedTensor.from_float(
        x.reshape(-1, x.shape[-1]), NVFP4).dequantize()


def _two_rank_rows(x, reducer):
    """Run the wrapped forward once per rank half and return the two outputs."""
    rows = []
    for half in (x[:32], x[32:]):
        module = _Linear()
        plan = qas.plan_for_topology({"ulysses": 2}, reducers={"sp": reducer})
        tree = _Tree(proj=module)
        qas.install(tree, {"ulysses": 2}, family="chroma",
                    reducers={"sp": reducer})
        assert plan.names == ("sp",)
        rows.append(tree.proj(half))
    return rows


def test_the_shared_scale_makes_each_ranks_rows_bit_identical_to_one_gpu():
    torch.manual_seed(0)
    x = torch.randn(64, 64, dtype=torch.bfloat16)
    x[40, 7] = 30.0  # the outlier lives in rank 1's half, so rank 0 misses it

    collect, seen = _collecting_reducer()
    _two_rank_rows(x, collect)
    assert len(seen) == 2
    assert not torch.equal(seen[0], seen[1])  # the two local amax differ
    shared = torch.stack(seen).amax(dim=0)

    rank0, rank1 = _two_rank_rows(x, _replay_reducer(shared))
    reference = _full_tensor_reference(x)
    assert torch.equal(rank0, reference[:32])
    assert torch.equal(rank1, reference[32:])


def test_the_shared_scale_is_bitwise_the_scale_one_gpu_computes():
    """Reduce the amax and divide in the input dtype, not the other way round."""
    torch.manual_seed(1)
    x = torch.randn(64, 64, dtype=torch.bfloat16)
    x[40, 7] = 30.0
    shared_amax = x.abs().amax().to(torch.float32).reshape(1)
    plan = qas.plan_for_topology({"ulysses": 2}, reducers={"sp": _replay_reducer(shared_amax)})

    scale = qas.shared_activation_scale(x[:32], plan, qas.NVFP4_SCALE_DIVISOR)
    reference = QuantizedTensor.from_float(x, NVFP4).params.scale
    assert torch.equal(scale.to(torch.float32), reference)
    # The float32 divide a naive implementation would take is a different number.
    wrong = (shared_amax.reshape(()) / qas.NVFP4_SCALE_DIVISOR)
    assert not torch.equal(wrong, reference)


def test_without_the_reduction_the_missing_rank_diverges():
    """The control: a rank that takes its scale from its own half fails the
    comparison the shared scale passes."""
    torch.manual_seed(0)
    x = torch.randn(64, 64, dtype=torch.bfloat16)
    x[40, 7] = 30.0
    module = _Linear()
    local = module(x[:32])
    reference = _full_tensor_reference(x)[:32]
    error = (local.float() - reference.float()).abs().max()
    assert error > 1e-2


def _mixed_tree():
    return _Tree(
        nvfp4=_Linear(NVFP4),
        nvfp4_static=_Linear(NVFP4, input_scale=torch.tensor(0.5)),
        nvfp4_smoothed=_Linear(NVFP4, pre_quant_scale=torch.ones(4)),
        nvfp4_experts=_Experts(),
        fp8=_Linear("TensorCoreFP8E4M3Layout", quantizes=False),
        fp8_e5m2=_Linear("TensorCoreFP8E5M2Layout", quantizes=False),
        mxfp8=_Linear("TensorCoreMXFP8Layout", quantizes=False),
        int8=_Linear("TensorWiseINT8Layout", quantizes=False),
        convrot=_Linear("TensorCoreConvRotW4A4Layout", quantizes=False),
        asym=_Linear("AsymW4A8Int8Layout", quantizes=False),
        plain=torch.nn.Linear(4, 4, dtype=torch.bfloat16),
    )


def _counting_reducer():
    calls = [0]

    def reduce(tensor):
        calls[0] += 1
        return tensor

    return reduce, calls


def test_only_the_plain_nvfp4_module_is_wrapped_and_the_rest_pay_nothing():
    tree = _mixed_tree()
    originals = {name: child.forward for name, child in tree.named_children()}
    reduce, calls = _counting_reducer()
    coverage = qas.install(tree, {"ulysses": 2}, family="chroma",
                           reducers={"sp": reduce, "cfg": reduce})

    assert coverage.wrapped == 1
    assert coverage.shard_dependent == 4
    # Declined: the smoothed module (comfy applies pre_quant_scale before the
    # amax, so a scale from the raw input is wrong) and the expert bank (comfy
    # ignores a written scale). The static-scale module is already shard invariant.
    assert coverage.declined == 2
    assert coverage.reducers == ("sp",)
    assert getattr(tree.nvfp4, qas.WRAPPED_ATTR, None) == ("sp",)
    for name, child in tree.named_children():
        if name == "nvfp4":
            assert child.forward is not originals[name]
        else:
            assert child.forward == originals[name]

    x = torch.randn(8, 4, dtype=torch.bfloat16)
    for name, child in tree.named_children():
        if name != "nvfp4":
            child(x)
    assert calls[0] == 0
    tree.nvfp4(x)
    assert calls[0] == 1
    assert not hasattr(tree.nvfp4, "input_scale")  # restored after the call


def test_an_unsharded_plan_wraps_nothing_and_builds_no_reducer():
    for topology, dual in (({}, False), ({"dp": 2}, False), ({"cfg": 2}, True)):
        tree = _mixed_tree()
        originals = {name: child.forward for name, child in tree.named_children()}
        reduce, calls = _counting_reducer()
        coverage = qas.install(tree, topology, family="chroma", dual_model_cfg=dual,
                               reducers={"sp": reduce, "cfg": reduce})
        assert coverage.wrapped == 0, topology
        assert coverage.reducers == (), topology
        for name, child in tree.named_children():
            assert child.forward == originals[name], (topology, name)
        tree.nvfp4(torch.randn(8, 4, dtype=torch.bfloat16))
        assert calls[0] == 0, topology


def test_installing_twice_wraps_once():
    tree = _mixed_tree()
    reduce, calls = _counting_reducer()
    first = qas.install(tree, {"ulysses": 2}, family="chroma", reducers={"sp": reduce})
    wrapped = tree.nvfp4.forward
    second = qas.install(tree, {"ulysses": 2}, family="chroma", reducers={"sp": reduce})
    assert (first.wrapped, second.wrapped) == (1, 1)
    assert tree.nvfp4.forward is wrapped
    tree.nvfp4(torch.randn(8, 4, dtype=torch.bfloat16))
    assert calls[0] == 1  # one reduction, not two stacked wrappers


def test_a_module_comfy_would_not_quantize_pays_no_collective():
    tree = _Tree(nvfp4=_Linear(NVFP4, weight_function=[lambda w: w]))
    reduce, calls = _counting_reducer()
    coverage = qas.install(tree, {"ulysses": 2}, family="chroma", reducers={"sp": reduce})
    assert coverage.wrapped == 1
    tree.nvfp4(torch.randn(8, 4, dtype=torch.bfloat16))
    assert calls[0] == 0
    # A 1-D activation is the other half of comfy's gate.
    assert qas.comfy_would_quantize(_Linear(), torch.randn(4)) is False


def test_an_expert_bank_is_declined_rather_than_paid_for():
    """comfy's MoEExperts quantizes with no scale, so a written one is lost."""
    tree = _Tree(experts=_Experts())
    original = tree.experts.forward
    reduce, calls = _counting_reducer()
    coverage = qas.install(tree, {"ulysses": 2}, family="chroma",
                           reducers={"sp": reduce})
    assert (coverage.wrapped, coverage.declined) == (0, 1)
    assert coverage.covered is False  # so the refusal counts it among the declined layers
    assert tree.experts.forward == original
    tree.experts(torch.randn(8, 4, dtype=torch.bfloat16))
    assert calls[0] == 0


def test_the_off_switch_alone_leaves_every_forward_stock(monkeypatch):
    """No wrapper at all, so the cost leg measures an untouched process."""
    monkeypatch.setenv(qas.ENABLE_ENV, "0")
    tree = _mixed_tree()
    originals = {name: child.forward for name, child in tree.named_children()}
    reduce, calls = _counting_reducer()
    coverage = qas.install(tree, {"ulysses": 2}, family="chroma",
                           reducers={"sp": reduce})
    assert (coverage.wrapped, coverage.declined, coverage.enabled) == (0, 3, False)
    assert coverage.covered is False  # so the refusal names the off switch
    for name, child in tree.named_children():
        assert child.forward == originals[name], name
    tree.nvfp4(torch.randn(8, 4, dtype=torch.bfloat16))
    assert calls[0] == 0


def test_the_off_switch_under_the_trace_knob_keeps_the_before_leg_readable(
        monkeypatch):
    """Hook off plus trace on: stock math, and the per-rank amax printed."""
    monkeypatch.setenv(qas.ENABLE_ENV, "0")
    monkeypatch.setenv(qas.LOG_ENV, "1")
    tree = _mixed_tree()
    reduce, calls = _counting_reducer()
    coverage = qas.install(tree, {"ulysses": 2}, family="chroma",
                           reducers={"sp": reduce})
    assert (coverage.wrapped, coverage.declined) == (0, 3)
    # Wrapped, but with no reducers: that is what the empty plan records.
    assert getattr(tree.nvfp4, qas.WRAPPED_ATTR, None) == ()

    x = torch.randn(8, 4, dtype=torch.bfloat16)
    # Bit for bit the rows a process with no hook at all produces, and not one
    # collective: the empty plan takes comfy's own whole-tensor fallback.
    assert torch.equal(tree.nvfp4(x), _Linear()(x))
    assert calls[0] == 0


def test_an_unsharded_install_declines_nothing_because_nothing_was_needed():
    tree = _mixed_tree()
    coverage = qas.install(tree, {}, family="chroma")
    assert (coverage.wrapped, coverage.declined, coverage.shard_dependent) == (0, 0, 4)
    assert coverage.covered is True


def test_the_log_knob_prints_one_line_per_wrapped_layer_for_the_first_step(
        monkeypatch, caplog):
    monkeypatch.setenv(qas.LOG_ENV, "1")
    tree = _Tree(proj=_Linear())
    shared = torch.tensor([9.0])
    qas.install(tree, {"ulysses": 2}, family="chroma",
                reducers={"sp": _replay_reducer(shared)})
    x = torch.randn(8, 4, dtype=torch.bfloat16)
    with caplog.at_level("INFO", logger="dgx_monarch.adapters.quant_activation_scale"):
        tree.proj(x)
        tree.proj(x)
    lines = [record.getMessage() for record in caplog.records
             if "local_amax" in record.getMessage()]
    assert len(lines) == 1, lines  # the first step only, not every step
    assert "reduced_amax=9" in lines[0]
    assert f"local_amax={float(x.abs().amax()):.6g}" in lines[0]
    assert "reducers=sp" in lines[0]


def test_the_trace_reads_the_local_amax_when_the_collective_reduces_in_place(
        monkeypatch, caplog):
    """xfuser's all_reduce returns its own input, so the statistic is cloned.

    Without the clone the trace stores two references to one tensor and prints
    the reduced amax under both labels, which makes the hardware legs unreadable
    while the scale itself stays right.
    """
    monkeypatch.setenv(qas.LOG_ENV, "1")
    torch.manual_seed(3)
    tree = _Tree(proj=_Linear())
    qas.install(tree, {"ulysses": 2}, family="chroma",
                reducers={"sp": _in_place_reducer(torch.tensor([9.0]))})
    x = torch.randn(8, 4, dtype=torch.bfloat16)
    with caplog.at_level("INFO", logger="dgx_monarch.adapters.quant_activation_scale"):
        tree.proj(x)
    line = next(record.getMessage() for record in caplog.records
                if "local_amax" in record.getMessage())
    assert f"local_amax={float(x.abs().amax()):.6g}" in line
    assert "reduced_amax=9" in line


def test_the_off_switch_still_traces_each_ranks_own_amax(monkeypatch, caplog):
    """The hardware "before" leg runs with the hook off and reads this line to
    size how far the ranks' amax disagree."""
    monkeypatch.setenv(qas.ENABLE_ENV, "0")
    monkeypatch.setenv(qas.LOG_ENV, "1")
    torch.manual_seed(4)
    tree = _Tree(proj=_Linear())
    qas.install(tree, {"ulysses": 2}, family="chroma")
    x = torch.randn(8, 4, dtype=torch.bfloat16)
    with caplog.at_level("INFO", logger="dgx_monarch.adapters.quant_activation_scale"):
        tree.proj(x)
    line = next(record.getMessage() for record in caplog.records
                if "local_amax" in record.getMessage())
    amax = f"{float(x.abs().amax()):.6g}"
    assert f"local_amax={amax}" in line and f"reduced_amax={amax}" in line
    assert "reducers=none" in line


def test_a_reinstall_reports_what_the_first_install_did(monkeypatch):
    """The mark carries the reducers, so a second walk counts them the same."""
    monkeypatch.setenv(qas.ENABLE_ENV, "0")
    tree = _mixed_tree()
    first = qas.install(tree, {"ulysses": 2}, family="chroma")
    second = qas.install(tree, {"ulysses": 2}, family="chroma")
    assert (first.wrapped, first.declined) == (second.wrapped, second.declined)
    assert second.covered is False


def test_the_worker_args_write_both_variables_and_report_an_enable_change(
        monkeypatch):
    """The fleet ships worker_args to every worker; that is the one channel a
    collective can be turned off through without wedging its peer."""
    import os

    os.environ.pop(qas.ENABLE_ENV, None)
    os.environ.pop(qas.LOG_ENV, None)
    # Absent keys leave a hand-set variable alone.
    assert qas.env_from_worker_args({}) is False
    assert qas.ENABLE_ENV not in os.environ

    assert qas.env_from_worker_args({qas.ENABLE_ARG: False}) is True
    assert os.environ[qas.ENABLE_ENV] == "0"
    assert qas.enabled_now() is False
    # The same setting twice is not a change, so it evicts no resident.
    assert qas.env_from_worker_args({qas.ENABLE_ARG: False}) is False

    assert qas.env_from_worker_args({qas.ENABLE_ARG: True, qas.LOG_ARG: True}) is True
    assert (os.environ[qas.ENABLE_ENV], os.environ[qas.LOG_ENV]) == ("1", "1")
    assert qas.enabled_now() is True

    # A key the last call carried and this one drops restores the default, so
    # removing it from the config turns the hook back on.
    previous = {qas.ENABLE_ARG: False, qas.LOG_ARG: True}
    qas.env_from_worker_args(previous)
    assert os.environ[qas.ENABLE_ENV] == "0"
    assert qas.env_from_worker_args({}, previous) is True
    assert qas.ENABLE_ENV not in os.environ and qas.LOG_ENV not in os.environ
    assert qas.enabled_now() is True


def test_both_knobs_are_worker_args_the_config_accepts():
    from dgx_monarch.config_schema import ClusterConfigError, validate_worker_args

    assert validate_worker_args({qas.ENABLE_ARG: False, qas.LOG_ARG: True}) == {
        qas.ENABLE_ARG: False, qas.LOG_ARG: True}
    with pytest.raises(ClusterConfigError, match="boolean true/false"):
        validate_worker_args({qas.ENABLE_ARG: "off"})


@pytest.mark.parametrize("topology,dual,expected", [
    ({"ulysses": 2}, False, ("sp",)),
    ({"ring": 2}, False, ("sp",)),
    ({"ulysses": 2, "ring": 2}, False, ("sp",)),
    ({"cfg": 2}, False, ("cfg",)),
    ({"cfg": 2, "ulysses": 2}, False, ("sp", "cfg")),
    ({"dp": 2}, False, ()),
    ({}, False, ()),
    ({"cfg": 2}, True, ()),
])
def test_the_plan_table(topology, dual, expected):
    plan = qas.plan_for_topology(topology, dual_model_cfg=dual)
    assert plan.names == expected
    assert bool(plan) is bool(expected)


def _naming_reducers():
    """Reducers that record which name ran, in the order the plan runs them."""
    applied: list[str] = []

    def named(name):
        def reduce(tensor):
            applied.append(name)
            return tensor

        return reduce

    return {"sp": named("sp"), "cfg": named("cfg")}, applied


def _dispatch_scope():
    from dgx_monarch.adapters import cfg_dispatch

    return cfg_dispatch._dispatch_scope()


def test_a_dispatched_cond_drops_the_cfg_reducer_and_keeps_sp():
    """Each rank runs a whole cond of its own, so its amax is its own."""
    reducers, applied = _naming_reducers()
    tree = _Tree(proj=_Linear())
    qas.install(tree, {"cfg": 2, "ulysses": 2}, family="chroma", reducers=reducers)
    x = torch.randn(8, 4, dtype=torch.bfloat16)

    tree.proj(x)
    assert applied == ["sp", "cfg"]  # one batched call, sliced by the cfg group

    applied.clear()
    with _dispatch_scope():
        tree.proj(x)
    assert applied == ["sp"]  # this rank's cond, sharded only by tokens


def test_a_folded_cfg_call_keeps_the_cfg_reducer():
    """cfg2 alone: outside a dispatch the group still slices one batched call."""
    reducers, applied = _naming_reducers()
    tree = _Tree(proj=_Linear())
    qas.install(tree, {"cfg": 2}, family="chroma", reducers=reducers)
    x = torch.randn(8, 4, dtype=torch.bfloat16)

    tree.proj(x)
    assert applied == ["cfg"]
    applied.clear()
    with _dispatch_scope():
        tree.proj(x)
    assert applied == []
    # The scope ends with the block: the next call reduces again.
    tree.proj(x)
    assert applied == ["cfg"]


def test_a_pair_that_would_not_fold_drops_the_cfg_reducer_and_keeps_sp():
    """Chroma's cfg-pad trim: outside any dispatch, the original cond lengths
    alone can still show that stock never made one batched call."""
    reducers, applied = _naming_reducers()
    tree = _Tree(proj=_Linear())
    qas.install(tree, {"cfg": 2, "ulysses": 2}, family="chroma", reducers=reducers)
    x = torch.randn(8, 4, dtype=torch.bfloat16)

    cfg_dispatch.record_cfg_pair_lengths([100, 28])
    tree.proj(x)
    assert applied == ["sp"]  # cfg dropped by the fold fact, not a dispatch


def test_a_pair_that_would_not_fold_drops_the_cfg_reducer_at_pure_cfg2():
    """Pure cfg2 (no sp) outside a dispatch: only the fold fact can drop cfg."""
    reducers, applied = _naming_reducers()
    tree = _Tree(proj=_Linear())
    qas.install(tree, {"cfg": 2}, family="chroma", reducers=reducers)
    x = torch.randn(8, 4, dtype=torch.bfloat16)

    cfg_dispatch.record_cfg_pair_lengths([100, 28])
    tree.proj(x)
    assert applied == []
    # Recorded, not scoped: cfg stays dropped until a folding pair is recorded,
    # unlike a dispatch, which drops it only inside its scope.
    tree.proj(x)
    assert applied == []
    cfg_dispatch.record_cfg_pair_lengths([44, 44])
    tree.proj(x)
    assert applied == ["cfg"]


def test_an_lcm_repeat_at_the_cap_keeps_the_cfg_reducer():
    """44/44 (equal) and a pair at an LCM repeat of 4 both fold and keep cfg;
    only a repeat past 4 drops it."""
    reducers, applied = _naming_reducers()
    tree = _Tree(proj=_Linear())
    qas.install(tree, {"cfg": 2}, family="chroma", reducers=reducers)
    x = torch.randn(8, 4, dtype=torch.bfloat16)

    cfg_dispatch.record_cfg_pair_lengths([2, 8])  # lcm repeat 4, at the cap
    tree.proj(x)
    assert applied == ["cfg"]


def test_the_dispatched_scale_is_the_scale_one_gpu_takes_for_that_cond():
    """The premise: a merged amax is a different number from the cond's own."""
    torch.manual_seed(5)
    cond = torch.randn(16, 16, dtype=torch.bfloat16)
    uncond = torch.randn(16, 16, dtype=torch.bfloat16)
    uncond[3, 3] = 40.0  # the outlier lives in the other rank's conditioning
    merged = torch.maximum(cond.abs().amax(), uncond.abs().amax())
    merged = merged.to(torch.float32).reshape(1)

    plan = qas.plan_for_topology({"cfg": 2}, reducers={"cfg": _replay_reducer(merged)})
    with _dispatch_scope():
        dispatched = qas.shared_activation_scale(
            cond, qas.plan_for_call(plan), qas.NVFP4_SCALE_DIVISOR)
    reference = QuantizedTensor.from_float(cond, NVFP4).params.scale
    assert torch.equal(dispatched.to(torch.float32), reference)
    # What the cfg reducer would have written instead, and one GPU never does.
    assert not torch.equal(
        qas.shared_activation_scale(cond, plan, qas.NVFP4_SCALE_DIVISOR).to(
            torch.float32), reference)


def test_the_trace_names_the_reducers_the_call_ran(monkeypatch, caplog):
    """The first-step line is the hardware leg's evidence, so it must not name
    a reducer this call skipped."""
    monkeypatch.setenv(qas.LOG_ENV, "1")
    reducers, _applied = _naming_reducers()
    tree = _Tree(proj=_Linear())
    qas.install(tree, {"cfg": 2, "ulysses": 2}, family="chroma", reducers=reducers)
    x = torch.randn(8, 4, dtype=torch.bfloat16)
    with caplog.at_level("INFO", logger="dgx_monarch.adapters.quant_activation_scale"):
        with _dispatch_scope():
            tree.proj(x)
    line = next(record.getMessage() for record in caplog.records
                if "local_amax" in record.getMessage())
    assert "reducers=sp" in line and "cfg" not in line


def test_the_trace_names_no_reducers_when_the_original_pair_does_not_fold(
        monkeypatch, caplog):
    """The 100/28 pair: no dispatch, but the original lengths alone drop cfg,
    and the first-step line is the hardware leg's evidence for it."""
    monkeypatch.setenv(qas.LOG_ENV, "1")
    cfg_dispatch.record_cfg_pair_lengths([100, 28])
    reducers, _applied = _naming_reducers()
    tree = _Tree(proj=_Linear())
    qas.install(tree, {"cfg": 2}, family="chroma", reducers=reducers)
    x = torch.randn(8, 4, dtype=torch.bfloat16)
    with caplog.at_level("INFO", logger="dgx_monarch.adapters.quant_activation_scale"):
        tree.proj(x)
    line = next(record.getMessage() for record in caplog.records
                if "local_amax" in record.getMessage())
    assert "reducers=none" in line


def test_the_per_call_decision_never_moves_what_the_bar_reads():
    """install plans from the topology; the dispatch decides one call only."""
    from dgx_monarch.adapters.base import UnsupportedModelError

    reducers, _applied = _naming_reducers()
    tree = _Tree(nvfp4=_Linear(NVFP4))
    with _dispatch_scope():
        coverage = qas.install(tree, {"cfg": 2}, family="chroma", reducers=reducers)
    assert coverage.reducers == ("cfg",)
    assert getattr(tree.nvfp4, qas.WRAPPED_ATTR, None) == ("cfg",)
    _bind({"cfg": 2})
    with _dispatch_scope():
        with pytest.raises(UnsupportedModelError, match=r"0\.129"):
            qas.assert_shard_quant_scale_covered(tree)


def test_the_fold_fact_never_moves_what_the_bar_reads():
    """Same proof for the fold fact: it changes the reducers a call runs,
    never the plan `install` recorded or the coverage the class K bar reads."""
    from dgx_monarch.adapters.base import UnsupportedModelError

    reducers, _applied = _naming_reducers()
    tree = _Tree(nvfp4=_Linear(NVFP4))
    cfg_dispatch.record_cfg_pair_lengths([100, 28])
    coverage = qas.install(tree, {"cfg": 2}, family="chroma", reducers=reducers)
    assert coverage.reducers == ("cfg",)
    assert getattr(tree.nvfp4, qas.WRAPPED_ATTR, None) == ("cfg",)
    _bind({"cfg": 2})
    with pytest.raises(UnsupportedModelError, match=r"0\.129"):
        qas.assert_shard_quant_scale_covered(tree)


def test_every_layout_is_classified_and_nvfp4_is_the_only_shard_dependent_one():
    assert qas.SHARD_DEPENDENT_LAYOUTS == frozenset({NVFP4})
    assert set(qas.ACTIVATION_SCALE_RULES.values()) == {
        "constant", "per_block", "not_quantized", "whole_tensor"}
    for name in ("TensorCoreFP8E4M3Layout", "TensorCoreFP8E5M2Layout"):
        assert qas.ACTIVATION_SCALE_RULES[name] == "constant"
    assert qas.ACTIVATION_SCALE_RULES["TensorCoreMXFP8Layout"] == "per_block"
    assert qas.ACTIVATION_SCALE_RULES["TensorWiseINT8Layout"] == "not_quantized"


def test_the_nvfp4_fallback_really_is_a_whole_tensor_statistic():
    """The premise of the shared scale, read from comfy_kitchen."""
    torch.manual_seed(2)
    x = torch.randn(32, 32, dtype=torch.bfloat16)
    y = x.clone()
    y[31, 31] = 40.0  # a change far outside row 0's first 16-element block
    first = QuantizedTensor.from_float(x, NVFP4).params.scale
    second = QuantizedTensor.from_float(y, NVFP4).params.scale
    assert not torch.equal(first, second)
    assert qas.NVFP4_SCALE_DIVISOR == 448.0 * 6.0


def test_the_sp_quant_check_now_sees_cfg():
    from dgx_monarch.adapters import cfg_parallel
    from dgx_monarch.adapters.base import UnsupportedModelError

    class _Adapter:
        family = "probe"
        sp_validated_quants = frozenset({"bf16"})
        sp_quant_refusal_reason = "probe measured wrong"

    adapter = _Adapter()
    with pytest.raises(UnsupportedModelError, match="probe measured wrong"):
        cfg_parallel.assert_sp_quant_supported(adapter, 1, "nvfp4", cfg=2)
    with pytest.raises(UnsupportedModelError, match="probe measured wrong"):
        cfg_parallel.assert_sp_quant_supported(adapter, 2, "nvfp4")
    # dm-cfg2 is not sharded, and a call that passes no cfg checks sp alone.
    cfg_parallel.assert_sp_quant_supported(adapter, 1, "nvfp4", cfg=2, dual_model_cfg=True)
    cfg_parallel.assert_sp_quant_supported(adapter, 1, "nvfp4")
    cfg_parallel.assert_sp_quant_supported(adapter, 2, "bf16")


KIND = "waive-known-wrong:shard-quant"


@pytest.fixture(autouse=True)
def _clear_waivers():
    from dgx_monarch import accuracy_waiver

    accuracy_waiver.clear()
    yield
    accuracy_waiver.clear()


def _uncovered(family="chroma", monkeypatch=None):
    """A model whose nvfp4 layers the shared scale did not reach."""
    tree = _Tree(nvfp4=_Linear(NVFP4, pre_quant_scale=torch.ones(4)))
    qas.install(tree, {"ulysses": 2}, family=family, reducers={"sp": lambda t: t})
    return tree


def _bind(topology=None, world=2):
    from dgx_monarch import accuracy_waiver

    accuracy_waiver.bind_model("model.safetensors", {}, [],
                               topology or {"ulysses": 2}, world, dispatch=True)


def test_an_uncovered_sharded_nvfp4_render_refuses_with_its_measurement():
    from dgx_monarch import consent_descriptor
    from dgx_monarch.adapters.base import UnsupportedModelError

    tree = _uncovered()
    assert qas.coverage_of(tree).covered is False
    _bind()
    with pytest.raises(UnsupportedModelError) as raised:
        qas.assert_shard_quant_scale_covered(tree)
    message = str(raised.value)
    assert "[dgxm:K guard=shard_quant_scale:chroma waivable=1]" in message
    assert "nvfp4 layer(s)" in message and "its own shard" in message
    assert "0.120" in message and "0.131" in message and "0.032" in message
    assert "DGXM_WAIVE_KNOWN_WRONG" in message
    assert "TROUBLESHOOTING.md #95" in message
    assert "fp8, mxfp8 or int8" in message
    # The hook was on and the cause is a layer it cannot reach, so the message
    # must not name an off switch nobody set.
    assert qas.ENABLE_ENV not in message and qas.ENABLE_ARG not in message
    assert message.count("Use a single or dp topology") == 1
    descriptor = consent_descriptor.parse(message)
    assert descriptor is not None and descriptor.kind == KIND
    assert descriptor.measured["probe"] == "shard_quant_scale:chroma"


def test_a_granted_waiver_proceeds_and_stamps_the_render():
    from dgx_monarch import accuracy_waiver

    tree = _uncovered()
    _bind()
    accuracy_waiver.activate({"_dgxm_render_id": "r-1", "_dgxm_accuracy_waivers": {
        KIND: {"kind": KIND, "id": "c" * 32, "consent_source": "panel"}}})
    qas.assert_shard_quant_scale_covered(tree)
    stamps = accuracy_waiver.stamps()
    assert [entry["guard"] for entry in stamps] == ["shard_quant_scale:chroma"]
    assert stamps[0]["kind"] == KIND
    assert "0.120" in stamps[0]["stamp"] and "0.131" in stamps[0]["stamp"]


@pytest.mark.parametrize("topology", [{"ulysses": 2}, {"cfg": 2}])
def test_a_covered_chroma_render_still_refuses_because_the_scale_is_not_enough(topology):
    """The bar is chroma's measurement with the hook on, not the coverage."""
    from dgx_monarch.adapters.base import UnsupportedModelError

    covered = _Tree(nvfp4=_Linear(NVFP4))
    qas.install(covered, topology, family="chroma",
                reducers={"sp": lambda t: t, "cfg": lambda t: t})
    assert qas.coverage_of(covered).covered is True
    _bind(topology)
    with pytest.raises(UnsupportedModelError) as raised:
        qas.assert_shard_quant_scale_covered(covered)
    message = str(raised.value)
    assert "[dgxm:K guard=shard_quant_scale:chroma waivable=1]" in message
    assert "0.120" in message and "0.129" in message and "0.032" in message
    assert "TROUBLESHOOTING.md #95" in message
    # The hook reached every layer, so the message blames neither a layer it
    # missed nor an off switch nobody set.
    assert "nvfp4 layer(s)" not in message and "its own shard" not in message
    assert qas.ENABLE_ENV not in message and qas.ENABLE_ARG not in message
    assert message.count("Use a single or dp topology") == 1


def test_a_topology_the_shared_scale_never_reduces_across_stays_unrefused():
    """dp, single and dm-cfg2 are each their own one-GPU reference."""
    unsharded = _Tree(nvfp4=_Linear(NVFP4, pre_quant_scale=torch.ones(4)))
    qas.install(unsharded, {"dp": 2}, family="chroma")
    qas.assert_shard_quant_scale_covered(unsharded)

    single = _Tree(nvfp4=_Linear(NVFP4, pre_quant_scale=torch.ones(4)))
    qas.install(single, None, family="chroma")
    qas.assert_shard_quant_scale_covered(single)

    dual = _Tree(nvfp4=_Linear(NVFP4))
    qas.install(dual, {"cfg": 2}, family="chroma", dual_model_cfg=True)
    qas.assert_shard_quant_scale_covered(dual)
    # A model this build never installed on carries no record and no verdict.
    qas.assert_shard_quant_scale_covered(_Tree(nvfp4=_Linear(NVFP4)))


def test_the_off_switch_is_named_only_when_the_off_switch_is_the_cause(monkeypatch):
    from dgx_monarch.adapters.base import UnsupportedModelError

    monkeypatch.setenv(qas.ENABLE_ENV, "0")
    tree = _Tree(nvfp4=_Linear(NVFP4))
    qas.install(tree, {"ulysses": 2}, family="chroma", reducers={"sp": lambda t: t})
    _bind()
    with pytest.raises(UnsupportedModelError) as raised:
        qas.assert_shard_quant_scale_covered(tree)
    message = str(raised.value)
    assert f"{qas.ENABLE_ARG} = true in worker_args" in message
    assert f"{qas.ENABLE_ENV} unset on every worker" in message
    assert "0.131" in message and "fp8, mxfp8 or int8" in message


def test_krea2_is_not_refused_because_its_measurement_is_under_the_floor():
    assert set(qas.SHARD_QUANT_GUARDS) == {"chroma"}
    _bind()
    qas.assert_shard_quant_scale_covered(_uncovered(family="krea2"))
    covered = _Tree(nvfp4=_Linear(NVFP4))
    qas.install(covered, {"ulysses": 2}, family="krea2", reducers={"sp": lambda t: t})
    qas.assert_shard_quant_scale_covered(covered)


def test_the_guard_is_read_on_the_sample_path_not_the_load_path():
    """A class-K grant lives for one dispatch, so the load carries none."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch" / "actor"
    sample = (src / "sample_protocol.py").read_text()
    worker = (src / "worker.py").read_text()
    assert "assert_shard_quant_scale_covered" in sample
    assert "assert_shard_quant_scale_covered" not in worker
    assert "quant_activation_scale.install" in worker
    activate = sample.index("activate_accuracy_waivers(request)")
    raised = sample.index("assert_shard_quant_scale_covered")
    assert raised > activate
    # And before the readiness exchange, inside the load guard, so a rank that
    # refuses here flags its peer instead of leaving it to wait at its first
    # all-reduce until the group timeout.
    assert raised < sample.index("readiness.ready_or_raise()")


def test_the_kind_is_class_k_and_can_never_be_automatic():
    from dgx_monarch.consent_kinds import KIND_SPECS
    from dgx_monarch.gate_audit_vocab import KIND_CLASS, KIND_GUARD_PREFIXES, check_guard
    from dgx_monarch.refusal import GUARDS, RefusalClass

    assert KIND_CLASS[KIND] == "K"
    spec = KIND_SPECS[KIND]
    assert spec.auto_eligible is False and spec.wired is True
    assert "0.120" in spec.measured and "0.131" in spec.measured
    assert "2026-09-05" in spec.measured
    assert spec.default_guard == KIND_GUARD_PREFIXES[KIND] == qas.SHARD_QUANT_GUARD
    guard = f"{qas.SHARD_QUANT_GUARD}:chroma"
    assert check_guard(KIND, guard) == guard
    assert GUARDS[guard].refusal_class is RefusalClass.KNOWN_WRONG
    assert GUARDS[guard].consent_kind == KIND
