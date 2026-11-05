"""Compare dgx-monarch's per-key bake with ComfyUI's bake on CPU.

A key stock patches at cast time bakes to stock's cast-time weight
(actor/lora_cast_time.py, #145); any other key follows ComfyUI's classic
patch_weight_to_device. Changes to dtype handling, rounding, seeds or adapter
math can make them diverge; runtime byte verification checks pristine weights,
so it cannot detect a bake mismatch. Ambient verification checks baked results
only for the keys sampled at each lazy swap (swap_verify, default 2), after the
user runs the changed ComfyUI.

The cast-time cases build ComfyUI's own MixedPrecisionOps Linear layers, one
holding a synthetic INT8 ConvRot weight and one a plain BF16 weight, and
compute stock's weight with comfy.ops' own post_cast: the transfer buffer
filled by cast_to_gathered, as aimdo fills it, then
resolve_cast_module_with_vbar with the arguments the layer's forward passes.
The bake, the classic load through the installed patch_weight_to_device, and
an unpatch back to the pristine weight must all match it bit for bit.

For each dtype and adapter format, twin patchers bake the same key through both
paths and must produce identical bits. The slice cases route LoRA units into
thirds of one fused qkv weight through ``(key, (dim, offset, size))`` entries,
the key map ComfyUI builds for fused checkpoints: ComfyUI stores the offset
with each patch and ``calculate_weight`` narrows to it. The comfy-canary
workflow runs against ComfyUI master daily. Pytest uses the importable
ComfyUI, with COMFY_DIR as an override.
"""
import copy
import os
import sys


def _ensure_comfy():
    comfy_dir = os.path.abspath(os.environ.get("COMFY_DIR", "../ComfyUI"))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    sys.argv = ["bake-equivalence-canary", "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()


def main() -> None:
    _ensure_comfy()
    import comfy.lora
    import comfy.model_patcher
    import torch

    from dgx_monarch.actor.model_store import ModelStore
    from dgx_monarch.actor.unbake import live_tensor

    torch.manual_seed(0)
    dim, rank = 32, 4
    checked = 0

    for dtype in (torch.float32, torch.bfloat16):
        for fmt in ("lora", "diff"):
            class _Tiny(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.attn = torch.nn.Linear(dim, dim, bias=False)

            pristine = _Tiny().to(dtype)
            key = "attn.weight"
            if fmt == "lora":
                sd = {
                    "unit.lora_up.weight": torch.randn(dim, rank),
                    "unit.lora_down.weight": torch.randn(rank, dim),
                    "unit.alpha": torch.tensor(float(rank)),
                }
            else:
                sd = {"unit.diff": (torch.randn(dim, dim, dtype=dtype) * 0.01)}
            patches = comfy.lora.load_lora(sd, {"unit": key})
            assert patches, f"comfy.lora.load_lora produced no patches for {fmt}"

            # Reference path: comfy computes the baked weight without writing.
            ref_patcher = comfy.model_patcher.ModelPatcher(
                copy.deepcopy(pristine), torch.device("cpu"), torch.device("cpu"))
            ref_patcher.add_patches(patches, 0.7)
            ref = ref_patcher.patch_weight_to_device(
                key, device_to=torch.device("cpu"), return_weight=True)

            # The lazy swap's per-key bake writes in place.
            our_patcher = comfy.model_patcher.ModelPatcher(
                copy.deepcopy(pristine), torch.device("cpu"), torch.device("cpu"))
            our_patcher.add_patches(patches, 0.7)
            ModelStore._bake_key(our_patcher, key)
            ours = live_tensor(our_patcher.model, key)

            assert ours.dtype == ref.dtype, (dtype, fmt, ours.dtype, ref.dtype)
            assert torch.equal(ours, ref), (
                f"BAKE DIVERGENCE: dtype={dtype} format={fmt}; dgx-monarch's "
                "per-key bake no longer bit-matches comfy's patch_weight_to_device")
            checked += 1
            print(f"equivalent: dtype={dtype} format={fmt}")

    checked += _check_slice_tuples(dim, rank)
    checked += _check_cast_time()
    if checked == 0:
        raise SystemExit("no combinations checked; canary is vacuous")
    print(f"bake equivalence green ({checked} combinations)")


def _check_slice_tuples(dim: int, rank: int) -> int:
    """A fused [3 * dim, dim] weight patched one third at a time, both paths."""
    import comfy.lora
    import comfy.model_patcher
    import torch

    from dgx_monarch.actor.model_store import ModelStore
    from dgx_monarch.actor.unbake import live_tensor

    checked = 0
    key = "qkv.weight"
    for dtype in (torch.float32, torch.bfloat16):
        for case, thirds in (("q", (0,)), ("qkv", (0, 1, 2))):
            class _Fused(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.qkv = torch.nn.Linear(dim, 3 * dim, bias=False)

            pristine = _Fused().to(dtype)
            sd, to_load = {}, {}
            for third in thirds:
                unit = f"unit{third}"
                sd[f"{unit}.lora_up.weight"] = torch.randn(dim, rank)
                sd[f"{unit}.lora_down.weight"] = torch.randn(rank, dim)
                sd[f"{unit}.alpha"] = torch.tensor(float(rank))
                to_load[unit] = (key, (0, third * dim, dim))
            patches = comfy.lora.load_lora(sd, to_load)
            assert set(patches) == set(to_load.values()), (
                f"comfy.lora.load_lora dropped a slice-tuple entry for {case}")

            ref_patcher = comfy.model_patcher.ModelPatcher(
                copy.deepcopy(pristine), torch.device("cpu"), torch.device("cpu"))
            ref_patcher.add_patches(patches, 0.7)
            assert len(ref_patcher.patches[key]) == len(thirds)
            ref = ref_patcher.patch_weight_to_device(
                key, device_to=torch.device("cpu"), return_weight=True)

            our_patcher = comfy.model_patcher.ModelPatcher(
                copy.deepcopy(pristine), torch.device("cpu"), torch.device("cpu"))
            our_patcher.add_patches(patches, 0.7)
            ModelStore._bake_key(our_patcher, key)
            ours = live_tensor(our_patcher.model, key)

            assert ours.dtype == ref.dtype, (dtype, case, ours.dtype, ref.dtype)
            assert torch.equal(ours, ref), (
                f"BAKE DIVERGENCE: dtype={dtype} slice={case}; dgx-monarch's per-key "
                "bake no longer bit-matches comfy's patch_weight_to_device")
            # The slice narrowed: rows outside the patched thirds keep their bits.
            untouched = [i for i in range(3) if i not in thirds]
            base = live_tensor(pristine, key)
            for third in untouched:
                rows = slice(third * dim, (third + 1) * dim)
                assert torch.equal(ours[rows], base[rows]), (dtype, case, third)
            checked += 1
            print(f"equivalent: dtype={dtype} slice={case}")
    return checked


def tiny_cast_time_model(out_features: int = 64, in_features: int = 512, rank: int = 4):
    """A root with one INT8 ConvRot layer ``q`` and one BF16 layer ``p``, and LoRA patches for both.

    Both are ComfyUI's MixedPrecisionOps Linear at a BF16 compute dtype, loaded
    from a state dict as a checkpoint loads them. Returns (root, patches).
    """
    import json

    import comfy.lora
    import comfy.ops
    import torch
    from comfy.quant_ops import QuantizedTensor

    ops = comfy.ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16)

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = ops.Linear(in_features, out_features, bias=True)
            self.p = ops.Linear(in_features, out_features, bias=False)

    class _Root(torch.nn.Module):
        manual_cast_dtype = torch.bfloat16

        def __init__(self):
            super().__init__()
            self.diffusion_model = _Block()

        def get_dtype_inference(self):
            return torch.bfloat16

    gen = torch.Generator().manual_seed(7)
    w = (torch.randn(out_features, in_features, generator=gen) * 0.05).to(torch.bfloat16)
    qt = QuantizedTensor.from_float(w, "TensorWiseINT8Layout", per_channel=True, convrot=True,
                                    convrot_groupsize=256, is_weight=True)
    layer = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
    root = _Root()
    root.load_state_dict({
        "diffusion_model.q.weight": qt._qdata,
        "diffusion_model.q.weight_scale": qt._params.scale,
        "diffusion_model.q.comfy_quant": torch.tensor(list(layer.encode()), dtype=torch.uint8),
        "diffusion_model.q.bias": torch.randn(out_features, generator=gen).to(torch.bfloat16),
        "diffusion_model.p.weight": (torch.randn(out_features, in_features, generator=gen) * 0.05).to(torch.bfloat16),
    }, strict=True)
    lora, key_map = {}, {}
    for name in ("q", "p"):
        unit = f"diffusion_model.{name}"
        lora[f"{unit}.lora_up.weight"] = torch.randn(out_features, rank, generator=gen) * 0.01
        lora[f"{unit}.lora_down.weight"] = torch.randn(rank, in_features, generator=gen) * 0.01
        lora[f"{unit}.alpha"] = torch.tensor(float(rank))
        key_map[unit] = f"{unit}.weight"
    return root, comfy.lora.load_lora(lora, key_map)


def stock_cast_time_weight(root, patcher, key: str):
    """The weight stock's DynamicVRAM cast gives ``key``'s layer, from comfy.ops' own post_cast."""
    import comfy.memory_management as cmm
    import comfy.model_management as mm
    import comfy.model_patcher
    import comfy.ops
    import comfy.utils
    import torch
    from comfy.quant_ops import QuantizedTensor

    module_key = key.rpartition(".")[0]
    s = comfy.utils.get_attr(root, module_key)
    cpu = torch.device("cpu")
    s.seed_key = module_key
    s.weight_lowvram_function = comfy.model_patcher.LowVramPatch(key, patcher.patches)
    s.bias_lowvram_function = None
    s.weight_function, s.bias_function = [], []
    geometries = cmm.tensors_to_geometries([s.weight, s.bias])
    buffer = torch.empty((cmm.vram_aligned_size([s.weight, s.bias]),), dtype=torch.uint8)
    mm.cast_to_gathered([s.weight, s.bias], buffer)
    s._prefetch = {"signature": None, "resident": False, "xfer_dest": buffer, "cast_dest": None,
                   "cast_geometry": geometries, "needs_cast": False}
    try:
        if isinstance(s.weight, QuantizedTensor):
            # The weight-only INT8 branch of MixedPrecisionOps.Linear.forward_comfy_cast_weights.
            weight, _bias = comfy.ops.resolve_cast_module_with_vbar(
                s, s.weight.dtype, cpu, torch.bfloat16, torch.bfloat16, True)
        else:
            # A plain layer: the input dtype, no quantized input.
            weight, _bias = comfy.ops.resolve_cast_module_with_vbar(
                s, torch.bfloat16, cpu, torch.bfloat16, torch.bfloat16, False)
    finally:
        for attr in ("seed_key", "weight_lowvram_function", "bias_lowvram_function", "_prefetch"):
            if attr in vars(s):
                delattr(s, attr)
        s.weight_function, s.bias_function = [], []
    return weight


def same_weight(a, b) -> bool:
    """Bit equality of two weights, a quantized one by its data and scale."""
    import torch
    from comfy.quant_ops import QuantizedTensor

    if isinstance(a, QuantizedTensor) or isinstance(b, QuantizedTensor):
        return (isinstance(a, QuantizedTensor) and isinstance(b, QuantizedTensor)
                and torch.equal(a._qdata, b._qdata) and torch.equal(a._params.scale, b._params.scale))
    return a.dtype == b.dtype and torch.equal(a, b)


def _check_cast_time() -> int:
    """INT8 ConvRot and BF16 keys bake to stock's cast-time weight on every path."""
    import comfy.model_patcher
    import comfy.utils
    import torch

    from dgx_monarch.actor import lora_cast_time
    from dgx_monarch.actor.model_store import ModelStore

    cpu = torch.device("cpu")
    checked = 0
    for key in ("diffusion_model.q.weight", "diffusion_model.p.weight"):
        root, patches = tiny_cast_time_model()
        patcher = comfy.model_patcher.ModelPatcher(root, cpu, cpu)
        patcher.add_patches(patches, 0.8)
        pristine = comfy.utils.get_attr(root, key)
        stock = stock_cast_time_weight(root, patcher, key)
        assert comfy.utils.get_attr(root, key) is pristine, f"stock's cast touched {key}"
        ours = lora_cast_time.cast_time_weight(patcher, key, cpu)
        assert ours is not None and same_weight(ours, stock), (
            f"BAKE DIVERGENCE: {key}; the cast-time bake no longer bit-matches stock's "
            "DynamicVRAM cast (comfy.ops post_cast)")
        installed = lora_cast_time.install()
        try:
            via = patcher.patch_weight_to_device(key, device_to=cpu, return_weight=True)
            assert same_weight(via, stock), f"BAKE DIVERGENCE: {key} through patch_weight_to_device"
            patcher.load(cpu, full_load=True)
            assert same_weight(comfy.utils.get_attr(root, key), stock), (
                f"BAKE DIVERGENCE: {key} through ComfyUI's classic load")
            patcher.unpatch_model(cpu)
            assert same_weight(comfy.utils.get_attr(root, key), pristine), f"unpatch left {key} baked"
        finally:
            if installed:
                lora_cast_time.uninstall()
        ModelStore._bake_key(patcher, key)
        assert same_weight(comfy.utils.get_attr(root, key), stock), (
            f"BAKE DIVERGENCE: {key} through the lazy swap's per-key bake")
        checked += 1
        print(f"equivalent: cast time {key}")
    return checked


if __name__ == "__main__":
    main()
