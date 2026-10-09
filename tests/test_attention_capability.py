"""The kernel capability rule: substitute where a kernel is too narrow, refuse
where nothing carries the head, and never crash inside the ring.

Every test here runs on the CPU with a fake kernel, because the thing under
test is the decision and not the kernel: the real probe is the function
yunchang's ring path selects, and it only exists where CUDA does.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from dgx_monarch.adapters import ADAPTERS
from dgx_monarch.adapters import attention_capability as capability
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.refusal import parse_refusal_tag

REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "src" / "dgx_monarch" / "actor" / "worker.py"

# comfy's Ideogram4Transformer2DModel takes attention_head_dim=256.
IDEOGRAM4_HEAD_DIM = 256
# A head no kernel this build tables carries, so nothing can be substituted.
UNCARRIED_HEAD_DIM = 512


def _declining_probe(*args, **kwargs):
    raise RuntimeError("No available kernel. Aborting execution.")


class _RecordingProbe:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((tuple(tensor.shape for tensor in args), kwargs))
        return args[0], torch.zeros(1)


def _tag(exc: BaseException):
    tag = parse_refusal_tag(str(exc))
    assert tag is not None, f"untyped refusal: {exc}"
    return tag


@pytest.mark.parametrize("kernel", ["TORCH_CUDNN", "SAGE_AUTO", "SAGE_FP8", "SAGE_FP16"])
def test_a_kernel_too_narrow_for_the_head_gives_way_to_flash(kernel: str) -> None:
    assert capability.effective_kernel(kernel, "ideogram4", IDEOGRAM4_HEAD_DIM) \
        == "TORCH_FLASH"


@pytest.mark.parametrize(("kernel", "head_dim"), [
    ("TORCH_FLASH", IDEOGRAM4_HEAD_DIM),
    ("TORCH_CUDNN", 128),
    ("TORCH_CUDNN", None),
    ("TORCH_EFFICIENT", IDEOGRAM4_HEAD_DIM),
    ("SOL_ATTN_TAU0.7", IDEOGRAM4_HEAD_DIM),
])
def test_every_other_selection_is_left_alone(kernel: str, head_dim: int | None) -> None:
    assert capability.effective_kernel(kernel, "family", head_dim) == kernel


def test_a_head_nothing_carries_is_not_substituted() -> None:
    """Nothing to swap to, so the selection stands and the refusal answers."""
    assert capability.effective_kernel("TORCH_CUDNN", "wide", UNCARRIED_HEAD_DIM) \
        == "TORCH_CUDNN"


def test_the_substitution_note_names_both_kernels_and_the_head() -> None:
    note = capability.substitution_note("TORCH_CUDNN", "ideogram4", IDEOGRAM4_HEAD_DIM)
    assert "TORCH_FLASH" in note and "TORCH_CUDNN" in note and "256" in note
    assert capability.substitution_note("TORCH_FLASH", "ideogram4", IDEOGRAM4_HEAD_DIM) == ""


def test_the_substitute_is_flash_and_never_math() -> None:
    """yunchang's math branch returns a log-sum-exp of zeros, wrong on a ring."""
    assert capability.SUBSTITUTE_ORDER == ("TORCH_FLASH",)
    assert "TORCH_MATH" not in capability.KERNEL_HEAD_DIM_LIMIT


def test_a_head_no_kernel_carries_refuses_class_p() -> None:
    with pytest.raises(UnsupportedModelError) as caught:
        capability.assert_static_capability("TORCH_CUDNN", "wide", UNCARRIED_HEAD_DIM)
    tag = _tag(caught.value)
    assert tag.refusal_class.value == "P"
    assert tag.guard is None and not tag.waivable
    text = str(caught.value)
    assert "512" in text and "cfg or single topology" in text
    # The card names the entry that tells an operator what to do about it.
    assert "docs/TROUBLESHOOTING.md #98" in text


def test_the_kernel_that_will_run_is_not_refused() -> None:
    capability.assert_static_capability("TORCH_FLASH", "ideogram4", IDEOGRAM4_HEAD_DIM)
    capability.assert_static_capability("TORCH_CUDNN", "family", None)


def test_a_declining_kernel_refuses_typed_and_names_the_shape() -> None:
    with pytest.raises(UnsupportedModelError) as caught:
        capability.assert_kernel_capability(
            "TORCH_EFFICIENT", "ideogram4", IDEOGRAM4_HEAD_DIM,
            probe=_declining_probe)
    text = str(caught.value)
    assert _tag(caught.value).refusal_class.value == "P"
    assert "TORCH_EFFICIENT" in text and "256" in text
    assert str(capability.PROBE_SEQUENCE) in text and "RuntimeError" in text
    assert "no mask" in text, "the ring path passes no mask and the text says so"
    # The same entry documents both class P arms, so both cards name it.
    assert "docs/TROUBLESHOOTING.md #98" in text
    assert capability.capability_records()[("ideogram4", "TORCH_EFFICIENT")] \
        .startswith("declined")


def test_an_admitting_kernel_records_and_returns() -> None:
    probe = _RecordingProbe()
    capability.assert_kernel_capability(
        "TORCH_EFFICIENT", "wide_family", IDEOGRAM4_HEAD_DIM, probe=probe)
    assert capability.capability_records()[("wide_family", "TORCH_EFFICIENT")] == "admitted"


def test_the_probe_is_the_ring_call_verbatim() -> None:
    probe = _RecordingProbe()
    capability.assert_kernel_capability(
        "TORCH_EFFICIENT", "wide_family", 64, probe=probe)
    (shapes, kwargs), = probe.calls
    expected = (capability.PROBE_BATCH, capability.PROBE_SEQUENCE,
                capability.PROBE_HEADS, 64)
    assert shapes == (torch.Size(expected),) * 3
    # xfuser/core/long_ctx_attention/ring/ring_flash_attn.py hands the selected
    # forward exactly these keywords, and no attention mask exists to hand it.
    assert kwargs == {"dropout_p": 0.0, "softmax_scale": None, "causal": False,
                      "window_size": (-1, -1), "softcap": 0.0,
                      "alibi_slopes": None, "return_softmax": False}
    assert "attn_mask" not in kwargs and "mask" not in kwargs


def test_the_probe_runs_in_the_models_compute_dtype_not_its_stored_weights() -> None:
    class _Model:
        manual_cast_dtype = torch.bfloat16
        diffusion_model = type("_D", (), {"dtype": torch.float8_e4m3fn})()

    seen: list[torch.dtype] = []

    def probe(q, k, v, **kwargs):
        seen.append(q.dtype)
        return q, torch.zeros(1)

    capability.assert_kernel_capability(
        "TORCH_EFFICIENT", "fp8_family", 64, _Model(), probe=probe)
    assert seen == [torch.bfloat16]


def test_a_sol_kernel_keeps_its_own_guards() -> None:
    def probe(*args, **kwargs):
        raise AssertionError("a sol kernel must not reach this probe")

    capability.assert_kernel_capability(
        "SOL_ATTN_TAU0.7", "ideogram4", IDEOGRAM4_HEAD_DIM, probe=probe)


def test_a_capacity_shortfall_is_not_a_statement_about_the_kernel() -> None:
    def probe(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("out of memory")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        capability.assert_kernel_capability(
            "TORCH_EFFICIENT", "family", 64, probe=probe)


def test_every_rank_answers_alike_because_every_input_is_shared() -> None:
    """Rank symmetry: the inputs carry no rank, so the verdict cannot either."""
    verdicts = []
    for _rank in range(4):
        try:
            capability.assert_kernel_capability(
                "TORCH_CUDNN", "ideogram4", IDEOGRAM4_HEAD_DIM,
                probe=_declining_probe)
            verdicts.append("render")
        except UnsupportedModelError as exc:
            verdicts.append(_tag(exc).refusal_class.value)
    assert verdicts == ["P"] * 4


def _dispatch(monkeypatch):
    """A real dispatcher whose construction is faked; the logic is the shipped one."""
    from dgx_monarch import adapters
    from dgx_monarch.actor import attention_dispatch as dispatch_mod

    built: list[str] = []

    def fake_usp(kernel_name, sync=True):
        built.append(kernel_name)

        def implementation(*_a, **_k):
            raise AssertionError("the stand-in is never called")

        return implementation

    monkeypatch.setattr(adapters, "make_usp_attention", fake_usp)
    return dispatch_mod._AttentionDispatch(), built


def test_setup_builds_the_selected_kernel_before_any_family_is_known(monkeypatch):
    dispatch, built = _dispatch(monkeypatch)
    dispatch.configure("TORCH_CUDNN", True, topology={"ulysses": 2, "ring": 1},
                       world=2, setup_generation=1)
    assert built == ["TORCH_CUDNN"]
    assert (dispatch.kernel, dispatch.effective_kernel) == ("TORCH_CUDNN", "TORCH_CUDNN")


def test_binding_the_family_rebuilds_under_the_kernel_that_carries_it(monkeypatch):
    dispatch, built = _dispatch(monkeypatch)
    dispatch.configure("TORCH_CUDNN", True, topology={"ulysses": 2, "ring": 1},
                       world=2, setup_generation=1)
    dispatch.bind_capability("ideogram4", IDEOGRAM4_HEAD_DIM)
    assert built == ["TORCH_CUDNN", "TORCH_FLASH"]
    # The selection the driver stamped survives; only the build changed.
    assert dispatch.kernel == "TORCH_CUDNN"
    assert dispatch.effective_kernel == "TORCH_FLASH"


def test_a_family_that_needs_no_substitution_rebuilds_nothing(monkeypatch):
    dispatch, built = _dispatch(monkeypatch)
    dispatch.configure("TORCH_CUDNN", True, topology={"ulysses": 2, "ring": 1},
                       world=2, setup_generation=1)
    dispatch.bind_capability("flux", None)
    assert built == ["TORCH_CUDNN"]
    assert dispatch.effective_kernel == "TORCH_CUDNN"


def test_the_per_render_kernel_flip_answers_the_same_way(monkeypatch):
    """actor/sample_protocol.py reconfigures per render, after injection."""
    dispatch, built = _dispatch(monkeypatch)
    dispatch.configure("TORCH_FLASH", True, topology={"ulysses": 2, "ring": 1},
                       world=2, setup_generation=1)
    dispatch.bind_capability("ideogram4", IDEOGRAM4_HEAD_DIM)
    dispatch.configure("SAGE_AUTO", True, topology={"ulysses": 2, "ring": 1},
                       world=2, setup_generation=1)
    assert built == ["TORCH_FLASH", "TORCH_FLASH"]
    assert (dispatch.kernel, dispatch.effective_kernel) == ("SAGE_AUTO", "TORCH_FLASH")


def test_a_retired_setup_generation_forgets_the_family(monkeypatch):
    dispatch, _built = _dispatch(monkeypatch)
    dispatch.configure("TORCH_CUDNN", True, topology={"ulysses": 2, "ring": 1},
                       world=2, setup_generation=1)
    dispatch.bind_capability("ideogram4", IDEOGRAM4_HEAD_DIM)
    dispatch.invalidate()
    assert dispatch.effective_kernel == ""
    dispatch.configure("TORCH_CUDNN", True, topology={"ulysses": 2, "ring": 1},
                       world=2, setup_generation=2)
    assert dispatch.effective_kernel == "TORCH_CUDNN"


def test_the_substitution_reaches_the_log_the_sweep_journal_reads(monkeypatch, caplog):
    dispatch, _built = _dispatch(monkeypatch)
    with caplog.at_level("INFO", logger="dgx_monarch.actor.attention_dispatch"):
        dispatch.configure("TORCH_CUDNN", True, topology={"ulysses": 2, "ring": 1},
                           world=2, setup_generation=1)
        dispatch.bind_capability("ideogram4", IDEOGRAM4_HEAD_DIM)
    lines = [record.getMessage() for record in caplog.records]
    assert any(line.startswith("USP attention kernel: TORCH_FLASH")
               and "substituted for TORCH_CUDNN" in line for line in lines), lines


def test_ideogram4_declares_the_head_dimension_the_model_file_fixes() -> None:
    adapter = next(item for item in ADAPTERS if item.family == "ideogram4")
    assert adapter.attention_head_dim == IDEOGRAM4_HEAD_DIM


def test_the_worker_binds_the_family_then_checks_then_injects() -> None:
    """Order matters: a bound forward is a render that has already started."""
    tree = ast.parse(WORKER.read_text())
    inject = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_inject_for_topology")
    wanted = ("bind_capability", "assert_kernel_capability", "inject_usp")
    names = [node.attr for node in ast.walk(inject)
             if isinstance(node, ast.Attribute) and node.attr in wanted]
    assert names == list(wanted), names
