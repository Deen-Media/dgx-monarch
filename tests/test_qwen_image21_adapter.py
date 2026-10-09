"""CPU contract coverage for the separate Qwen-Image-2.1 adapter."""
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch

from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError, usp_options
from dgx_monarch.adapters.qwen_image21 import (
    QwenImage21Adapter,
    _take_native_attention_tensor,
    agree_prefix_cache,
    block_causal_target_attention,
    distributed_prefix_cache_key,
    prefix_cache_is_exact,
    validate_reference_count,
)


def test_native_attention_container_is_taken_but_tensor_path_is_unchanged(monkeypatch):
    """Comfy callbacks may pass a single-owner AttentionTensorContainer instead of a tensor."""
    class Container:
        def __init__(self, tensor):
            self.tensor = tensor

        def take(self):
            if self.tensor is None:
                raise RuntimeError("already consumed")
            tensor, self.tensor = self.tensor, None
            return tensor

    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    modules = types.ModuleType("comfy.ldm.modules")
    attention = types.ModuleType("comfy.ldm.modules.attention")
    attention.AttentionTensorContainer = Container
    comfy.ldm, ldm.modules, modules.attention = ldm, modules, attention
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.ldm", ldm)
    monkeypatch.setitem(sys.modules, "comfy.ldm.modules", modules)
    monkeypatch.setitem(sys.modules, "comfy.ldm.modules.attention", attention)

    tensor = torch.randn(1, 2, 3, 4)
    assert _take_native_attention_tensor(tensor) is tensor
    container = Container(tensor)
    assert _take_native_attention_tensor(container) is tensor
    assert container.tensor is None
    with pytest.raises(TypeError, match="unsupported attention input"):
        _take_native_attention_tensor(object())


def _comfy_dir() -> Path | None:
    """Return the Comfy checkout the environment names; the ordinary CPU CI job has none."""
    raw = os.environ.get("COMFY_DIR") or os.environ.get("COMFYUI_DIR")
    if not raw:
        return None
    path = Path(raw)
    return path if (path / "comfy" / "options.py").is_file() else None


def _require_pinned_comfy() -> None:
    if _comfy_dir() is None:
        pytest.skip("real Comfy canary requires COMFY_DIR or COMFYUI_DIR")


def _configure_pinned_comfy_for_cpu() -> None:
    """Configure Comfy in a spawned canary child, never the parent process."""
    path = _comfy_dir()
    if path is None:
        raise RuntimeError("real Comfy canary requires COMFY_DIR or COMFYUI_DIR")
    sys.path.insert(0, str(path))
    import comfy.options

    sys.argv = ["pytest-qwen-image21", "--cpu"]
    comfy.options.args_parsing = True


def test_declared_contract_is_distinct_from_legacy_qwen():
    assert QwenImage21Adapter.family == "qwen_image21"
    assert QwenImage21Adapter.model_base_classes == ("QwenImage21",)
    assert QwenImage21Adapter.cfg_cond_padding == "none"


def test_reference_count_accepts_public_one_to_ten_surface():
    assert validate_reference_count(None) == ()
    assert len(validate_reference_count([object()] * 10)) == 10
    with pytest.raises(UnsupportedModelError, match="at most 10"):
        validate_reference_count([object()] * 11)


def test_exact_prefix_cache_requires_distributed_owner_and_no_hooks():
    assert prefix_cache_is_exact(enabled=True, hooks_present=False, refs=[object()],
                                 image_slots=[3], cache_owner="distributed")
    assert not prefix_cache_is_exact(enabled=True, hooks_present=False, refs=[],
                                     image_slots=[], cache_owner=None)
    assert not prefix_cache_is_exact(enabled=True, hooks_present=True, refs=[],
                                     image_slots=[], cache_owner="distributed")
    with pytest.raises(UnsupportedModelError, match="more entries"):
        prefix_cache_is_exact(enabled=True, hooks_present=False, refs=[object()],
                              image_slots=[1, 2], cache_owner="distributed")


def test_distributed_cache_key_preserves_native_key_and_cohort_identity():
    native = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rank0 = distributed_prefix_cache_key(native, topology_sp=2, world=2, rank=0)
    rank1 = distributed_prefix_cache_key(native, topology_sp=2, world=2, rank=1)
    assert torch.equal(rank0[:, :-3], native)
    assert torch.equal(rank0[:, -3:], torch.tensor([[2.0, 2.0, 0.0]]).expand(2, -1))
    assert not torch.equal(rank0, rank1)
    with pytest.raises(UnsupportedModelError, match="cohort"):
        distributed_prefix_cache_key(native, topology_sp=2, world=1, rank=0)


@pytest.mark.parametrize(
    ("cohort", "expected_cache", "expected_cached"),
    [
        (((True, True), (False, False)), False, False),
        (((True, True), (True, False)), True, False),
        (((True, True), (True, True)), True, True),
        (((True, False), (True, False)), True, False),
    ],
    ids=("mixed-present", "mixed-hit", "unanimous-hit", "unanimous-miss"),
)
def test_prefix_cache_agreement_selects_one_cohort_branch(monkeypatch, cohort,
                                                           expected_cache, expected_cached):
    import dgx_monarch.adapters.qwen_image21 as q21

    calls = []
    monkeypatch.setattr(q21.base, "sp_world", lambda: 2)

    def gather(local, original, dim=1):
        calls.append((local.clone(), original, dim))
        return local.new_tensor(cohort).reshape(-1)

    monkeypatch.setattr(q21, "sp_gather", gather)
    cache = object()
    actual_cache, actual_cached = agree_prefix_cache(cache, True, torch.device("cpu"))
    assert (actual_cache is cache) is expected_cache
    assert actual_cached is expected_cached
    assert len(calls) == 1
    assert calls[0][1:] == (4, 0)


def test_block_causal_target_attention_preserves_native_masks_and_options(monkeypatch):
    """The adapter must pass native segment masks unchanged to Comfy attention."""
    import dgx_monarch.adapters.qwen_image21 as q21

    calls = []
    marker = object()
    preferred = object()
    options = {"native": "options"}

    def optimized(q, k, v, heads, *, mask, transformer_options, preferred_attention):
        calls.append((q.shape, k.shape, v.shape, heads, mask, transformer_options, preferred_attention))
        return q

    monkeypatch.setattr(q21, "_optimized_attention", optimized)
    q = torch.randn(1, 9, 3, 4)
    prefix_out, target_out = block_causal_target_attention(
        q[:, :4], q[:, :4], q[:, :4], q[:, 4:], q[:, 4:], q[:, 4:],
        [(0, 2, marker), (2, 4, None)], 3,
        transformer_options=options, preferred_attention=preferred,
    )
    assert torch.equal(prefix_out, q[:, :4])
    assert torch.equal(target_out, q[:, 4:])
    assert [call[4] for call in calls] == [marker, None, None]
    assert all(call[5] is options and call[6] is preferred for call in calls)


def test_block_causal_target_attention_accepts_a_target_only_sequence(monkeypatch):
    """An empty prefix runs one target-only attention call and returns an empty prefix."""
    import dgx_monarch.adapters.qwen_image21 as q21

    calls = []

    def optimized(q, k, v, heads, **kwargs):
        calls.append((q, k, v, heads, kwargs))
        return q

    monkeypatch.setattr(q21, "_optimized_attention", optimized)
    prefix = torch.empty(1, 0, 2, 4)
    target = torch.randn(1, 3, 2, 4)
    prefix_out, target_out = block_causal_target_attention(
        prefix, prefix, prefix, target, target, target, [], 2,
        transformer_options={}, preferred_attention=None,
    )
    assert torch.equal(prefix_out, prefix)
    assert torch.equal(target_out, target)
    assert len(calls) == 1
    assert calls[0][1].shape[1] == target.shape[1]


def test_qwen21_large_ref_attention_structure_has_no_adapter_quadratic_buffers(monkeypatch):
    """Ten 1024-square references reach native attention as segments, never as dense scores or masks."""
    import dgx_monarch.adapters.qwen_image21 as q21

    calls = []
    marker = object()
    options = {"test": "large-meta"}

    def optimized(q, k, v, heads, *, mask, transformer_options, preferred_attention):
        calls.append((q.shape, k.shape, v.shape, heads, mask, transformer_options, preferred_attention))
        return q

    def forbidden(*_args, **_kwargs):
        raise AssertionError("adapter must not construct dense QK scores or masks")

    monkeypatch.setattr(q21, "_optimized_attention", optimized)
    monkeypatch.setattr(q21.torch, "einsum", forbidden)
    monkeypatch.setattr(q21.torch, "ones", forbidden)
    # Qwen's documented /16 latent turns a 1024-square target/reference into
    # 64x64=4096 rows. The local uly2 target Q is 2048; K/V are still gathered.
    heads, dim, text_rows, image_rows = 32, 128, 512, 64 * 64
    prefix_rows, full_target_rows = text_rows + 10 * image_rows, image_rows
    q_prefix = torch.empty((1, prefix_rows, heads, dim), device="meta")
    q_target = torch.empty((1, full_target_rows // 2, heads, dim), device="meta")
    gathered_target = torch.empty((1, full_target_rows, heads, dim), device="meta")
    segments = [(0, text_rows, marker)] + [
        (text_rows + index * image_rows, text_rows + (index + 1) * image_rows, None)
        for index in range(10)
    ]
    prefix_out, target_out = block_causal_target_attention(
        q_prefix, q_prefix, q_prefix, q_target, gathered_target, gathered_target, segments, heads,
        transformer_options=options, preferred_attention=None,
    )
    assert prefix_out.shape == q_prefix.shape and target_out.shape == q_target.shape
    assert len(calls) == 12
    assert calls[0][4] is marker
    assert max(call[0][1] for call in calls) == image_rows
    assert calls[-1][0][1] == full_target_rows // 2
    assert calls[-1][1][1] == prefix_rows + full_target_rows
    assert all(call[5] is options for call in calls)

    calls.clear()
    cached = q21.cached_target_attention(
        q_prefix[:, :image_rows], q_prefix[:, :image_rows], q_target,
        gathered_target, gathered_target, heads,
        transformer_options=options, preferred_attention=None,
    )
    assert cached.shape == q_target.shape
    assert len(calls) == 1
    assert calls[0][0][1] == full_target_rows // 2
    assert calls[0][1][1] == image_rows + full_target_rows


def _stock_qwen21_cache_worker(_rank, result_file):
    """Run comfy's real Qwen Image 2.1 module on CPU without model weights.

    The two-rank Gloo tests below cover the collective. This one catches
    forward or signature changes in comfy and proves cache-off stock execution
    accepts zero, one and several reference prefixes at batch two.
    """
    _configure_pinned_comfy_for_cpu()
    import comfy.ops
    from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel

    torch.manual_seed(4)
    model = QwenImage21Transformer2DModel(
        in_channels=64, out_channels=64, num_layers=1,
        attention_head_dim=6, num_attention_heads=1, context_in_dim=4,
        mlp_ratio=1, axes_dims_rope=(2, 2, 2), operations=comfy.ops.disable_weight_init,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    model.reset_prefix_cache(False)
    x = torch.randn(2, 64, 2, 3)
    t = torch.tensor([0.25, 0.75])
    context = torch.randn(2, 3, 4)
    answers = []
    for refs, slots in (([], []), ([torch.randn(2, 64, 1, 2)], [1]),
                        ([torch.randn(2, 64, 1, 1), torch.randn(2, 64, 1, 2)], [1, 2])):
        out = model._forward(x, t, context, ref_latents=refs, image_slots=slots)
        answers.append(out.shape == x.shape and bool(torch.isfinite(out).all()))
    Path(result_file).write_text(repr(answers))


def test_tiny_pinned_comfy_qwen21_stock_cache_off_cpu():
    """Run the real-Comfy cache-off canary in a spawned child process."""
    _require_pinned_comfy()
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as directory:
        result_file = str(Path(directory) / "result")
        mp.spawn(_stock_qwen21_cache_worker, args=(result_file,), nprocs=1, join=True)
        assert Path(result_file).read_text() == "[True, True, True]"


def _gloo_qwen21_worker(rank, init_file, result_file):
    """Compare the real injected forward with stock on two Gloo ranks, cache off."""
    _configure_pinned_comfy_for_cpu()
    import comfy.ops
    import torch.distributed as dist
    from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel

    import dgx_monarch.adapters.base as base
    import dgx_monarch.adapters.qwen_image21 as q21

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    base.sp_world = lambda: 2
    base.sp_rank = lambda: rank

    def gather(t, orig, dim=1):
        pieces = [torch.empty_like(t) for _ in range(2)]
        dist.all_gather(pieces, t.contiguous())
        return torch.cat(pieces, dim=dim).narrow(dim, 0, orig)

    q21.sp_gather = gather
    q21.agree_prefix_cache = lambda *args: (_ for _ in ()).throw(
        AssertionError("cache-off forward must not enter cache agreement"))
    torch.manual_seed(31)
    model = QwenImage21Transformer2DModel(
        in_channels=64, out_channels=64, num_layers=1, attention_head_dim=6,
        num_attention_heads=1, context_in_dim=4, mlp_ratio=1,
        axes_dims_rope=(2, 2, 2), operations=comfy.ops.disable_weight_init,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    model.reset_prefix_cache(False)
    # Five target rows pad to six over two ranks, so the injected SP path carries
    # one synthetic row until the final gather trims to the stock 1x5 latent.
    x, t, context = torch.randn(2, 64, 1, 5), torch.tensor([.25, .75]), torch.randn(2, 3, 4)
    cases = (
        [],
        [torch.randn(2, 64, 1, 2)],
        [torch.randn(2, 64, 1, 1), torch.randn(2, 64, 1, 2)],
        [torch.randn(2, 64, 1, 1) for _ in range(10)],
    )
    answers = []
    for refs in cases:
        # The tiny canary has only three text rows. Reusing slot one keeps all
        # ten references in the native ordered-prefix contract without making
        # an invalid decreasing text interval.
        slots = [1] * len(refs) if len(refs) == 10 else list(range(1, len(refs) + 1))
        stock = model._forward(x, t, context, ref_latents=refs, image_slots=slots)
        QwenImage21Adapter().inject_usp(
            model, InjectionContext(topology_sp=2, usp_attention=types.SimpleNamespace(
                effective_kernel="TORCH_FLASH")))
        candidate = model._forward(x, t, context, ref_latents=refs, image_slots=slots)
        answers.append(torch.allclose(stock, candidate, atol=2e-6, rtol=2e-6))
        model._forward = type(model)._forward.__get__(model, type(model))
    if rank == 0:
        Path(result_file).write_text(repr(answers))
    dist.destroy_process_group()


def test_tiny_pinned_comfy_qwen21_injected_two_rank_gloo_cpu():
    _require_pinned_comfy()
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as directory:
        init_file = str(Path(directory) / "gloo-init")
        result_file = str(Path(directory) / "result")
        mp.spawn(_gloo_qwen21_worker, args=(init_file, result_file), nprocs=2, join=True)
        assert Path(result_file).read_text() == "[True, True, True, True]"


def _gloo_qwen21_cache_worker(rank, init_file, result_file, cache_dtype, agreement_case=None):
    """Exercise cache fill, reuse and fresh slots through comfy's real module."""
    _configure_pinned_comfy_for_cpu()
    import comfy.ops
    import torch.distributed as dist
    from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel

    import dgx_monarch.adapters.base as base
    import dgx_monarch.adapters.qwen_image21 as q21

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    base.sp_world = lambda: 2
    base.sp_rank = lambda: rank

    def gather(t, orig, dim=1):
        pieces = [torch.empty_like(t) for _ in range(2)]
        dist.all_gather(pieces, t.contiguous())
        return torch.cat(pieces, dim=dim).narrow(dim, 0, orig)

    q21.sp_gather = gather
    torch.manual_seed(45)
    model = QwenImage21Transformer2DModel(
        # The native W4A4 cache requires a 64-wide quantization group, so use
        # a 64-wide head to exercise both real low-bit layouts.
        in_channels=64, out_channels=64, num_layers=1, attention_head_dim=64,
        num_attention_heads=1, context_in_dim=4, mlp_ratio=1,
        axes_dims_rope=(16, 16, 32), operations=comfy.ops.disable_weight_init,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    x = torch.randn(2, 64, 1, 5)
    context = torch.randn(2, 3, 4)
    refs = [torch.randn(2, 64, 1, 2)]
    slots = [1]
    # Compare with a separate stock model in the same native cache mode: int8
    # and int4 are lossy, so cache-off stock is the wrong reference for them.
    # The same prefix reuses the first slot at a later timestep; changed text,
    # ref bytes and slot layout allocate fresh entries, matching stock.
    stock = QwenImage21Transformer2DModel(
        in_channels=64, out_channels=64, num_layers=1, attention_head_dim=64,
        num_attention_heads=1, context_in_dim=4, mlp_ratio=1,
        axes_dims_rope=(16, 16, 32), operations=comfy.ops.disable_weight_init,
    )
    stock.load_state_dict(model.state_dict())
    stock.current_patcher = types.SimpleNamespace(get_free_memory=lambda _device: 1 << 60)
    stock.reset_prefix_cache(True)
    options = {"qwen_image21_cache": {"device": "gpu", "dtype": cache_dtype}}
    expected = [
        stock._forward(x, torch.tensor([.25, .75]), context, ref_latents=refs, image_slots=slots,
                       transformer_options=options),
        stock._forward(x, torch.tensor([.5, .125]), context, ref_latents=refs, image_slots=slots,
                       transformer_options=options),
        stock._forward(x, torch.tensor([.25, .75]), context + 0.1, ref_latents=refs, image_slots=slots,
                       transformer_options=options),
        stock._forward(x, torch.tensor([.25, .75]), context, ref_latents=[refs[0] + 0.1], image_slots=[2],
                       transformer_options=options),
    ]
    QwenImage21Adapter().inject_usp(
        model, InjectionContext(topology_sp=2, usp_attention=types.SimpleNamespace(
            effective_kernel="TORCH_FLASH")))
    model.current_patcher = types.SimpleNamespace(get_free_memory=lambda _device: 1 << 60)
    model.reset_prefix_cache(agreement_case != "mixed-enabled" or rank == 0)
    agreement = q21.agree_prefix_cache
    traces = []

    def traced_agreement(cache, cached, device):
        selected, selected_cached = agreement(cache, cached, device)
        traces.append((cache is not None, cached, selected is not None, selected_cached))
        return selected, selected_cached

    q21.agree_prefix_cache = traced_agreement
    actual = [model._forward(x, torch.tensor([.25, .75]), context, ref_latents=refs, image_slots=slots,
                             transformer_options=options)]
    select_prefix_cache = model.select_prefix_cache
    if agreement_case == "mixed-present" and rank == 1:
        model.select_prefix_cache = lambda *args: (None, False)
    elif agreement_case == "mixed-hit" and rank == 1:
        def select_prefix_cache(*args):
            selected, _cached = model.__class__.select_prefix_cache(model, *args)
            return selected, False

        model.select_prefix_cache = select_prefix_cache
    actual.append(model._forward(x, torch.tensor([.5, .125]), context, ref_latents=refs, image_slots=slots,
                                 transformer_options=options))
    model.select_prefix_cache = select_prefix_cache
    actual.extend([
        model._forward(x, torch.tensor([.25, .75]), context + 0.1, ref_latents=refs, image_slots=slots,
                       transformer_options=options),
        model._forward(x, torch.tensor([.25, .75]), context, ref_latents=[refs[0] + 0.1], image_slots=[2],
                       transformer_options=options),
    ])
    answers = [torch.allclose(left, right, atol=3e-6, rtol=3e-6)
               for left, right in zip(expected, actual, strict=True)]
    slots_used = len(model.prefix_cache.slots) if model.prefix_cache is not None else 0
    trace = torch.tensor(traces[1], dtype=torch.int8)
    cohort_trace = [torch.empty_like(trace) for _ in range(2)]
    dist.all_gather(cohort_trace, trace)
    if rank == 0:
        result = (answers, slots_used) if agreement_case is None else (answers, tuple(
            tuple(row.tolist()) for row in cohort_trace))
        Path(result_file).write_text(repr(result))
    dist.destroy_process_group()


@pytest.mark.parametrize("cache_dtype", ["default", "int8", "int4"])
def test_tiny_pinned_comfy_qwen21_cache_two_rank_gloo_cpu(cache_dtype):
    _require_pinned_comfy()
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as directory:
        init_file = str(Path(directory) / "gloo-init")
        result_file = str(Path(directory) / "result")
        mp.spawn(_gloo_qwen21_cache_worker,
                 args=(init_file, result_file, cache_dtype), nprocs=2, join=True)
        # The first two forwards share one prefix slot; the changed text and the
        # changed reference each allocate another, so three slots.
        assert Path(result_file).read_text() == "([True, True, True, True], 3)"


@pytest.mark.parametrize(
    ("agreement_case", "expected"),
    [
        ("mixed-enabled", "([True, True, True, True], ((1, 0, 0, 0), (0, 0, 0, 0)))"),
        ("mixed-present", "([True, True, True, True], ((1, 1, 0, 0), (0, 0, 0, 0)))"),
        ("mixed-hit", "([True, True, True, True], ((1, 1, 1, 0), (1, 0, 1, 0)))"),
    ],
)
def test_tiny_pinned_comfy_qwen21_cache_agreement_two_rank_gloo_cpu(agreement_case, expected):
    """Ranks whose native cache selection differs must agree on one branch before the blocks run."""
    _require_pinned_comfy()
    import torch.multiprocessing as mp

    with tempfile.TemporaryDirectory() as directory:
        init_file = str(Path(directory) / "gloo-init")
        result_file = str(Path(directory) / "result")
        mp.spawn(_gloo_qwen21_cache_worker,
                 args=(init_file, result_file, "default", agreement_case), nprocs=2, join=True)
        assert Path(result_file).read_text() == expected


def test_usp_disables_cache_and_binds_exact_forward():
    calls = []
    fake = types.SimpleNamespace(transformer_blocks=[], reset_prefix_cache=lambda enabled: calls.append(enabled))
    QwenImage21Adapter().inject_usp(
        fake, InjectionContext(topology_sp=2, usp_attention=types.SimpleNamespace(
            effective_kernel="TORCH_FLASH")))
    assert calls == [False]
    assert callable(fake._forward)


def test_usp_rechecks_the_mutable_worker_selector_before_sequence_building():
    dispatch = types.SimpleNamespace(effective_kernel="TORCH_FLASH")
    calls = []
    fake = types.SimpleNamespace(
        transformer_blocks=[],
        reset_prefix_cache=lambda enabled: None,
        build_sequence=lambda *_args: calls.append("build") or (_ for _ in ()).throw(AssertionError()),
    )
    QwenImage21Adapter().inject_usp(fake, InjectionContext(topology_sp=2, usp_attention=dispatch))
    dispatch.effective_kernel = "SAGE_AUTO"
    with pytest.raises(UnsupportedModelError, match="requires TORCH_FLASH"):
        fake._forward(None, None, None)
    assert calls == []


def test_usp_rejects_foreign_attention_override_before_sequence_building():
    calls = []
    fake = types.SimpleNamespace(
        transformer_blocks=[],
        reset_prefix_cache=lambda enabled: None,
        build_sequence=lambda *_args: calls.append("build") or (_ for _ in ()).throw(AssertionError()),
    )
    QwenImage21Adapter().inject_usp(
        fake, InjectionContext(topology_sp=2, usp_attention=types.SimpleNamespace(
            effective_kernel="TORCH_FLASH")))

    with pytest.raises(UnsupportedModelError, match="optimized_attention_override"):
        fake._forward(
            torch.empty(1, 64, 1, 1), torch.zeros(1), torch.empty(1, 1, 1),
            transformer_options={"optimized_attention_override": lambda *_args, **_kwargs: None},
        )
    assert calls == []

    with pytest.raises(AssertionError):
        fake._forward(
            torch.empty(1, 64, 1, 1), torch.zeros(1), torch.empty(1, 1, 1),
            transformer_options=usp_options({}, object()),
        )
    assert calls == ["build"]


def test_qwen21_signature_does_not_fall_into_legacy_qwen():
    from dgx_monarch.adapters.detect import detect_family_from_keys

    keys = [
        "model.diffusion_model.modulation.1.weight",
        "model.diffusion_model.transformer_blocks.0.attn.to_q.weight",
        "model.diffusion_model.img_in.weight",
    ]
    assert detect_family_from_keys(keys) == "qwen_image21"
