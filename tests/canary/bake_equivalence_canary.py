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

Which keys get a cast-time patch is stock's own choice, made in
ModelPatcherDynamic.load. The choice cases run that load on CPU (aimdo's host
buffer stood in, since no pinned memory is needed to choose) over BF16 and
INT8 ConvRot layers on both sides of its 16 KiB cut-off, a bias patch and a
LoRA that resizes its weight, and require the bake to choose the same keys.

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
    checked += check_stock_choice()
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


def _choice_model():
    """Layers on both sides of stock's 16 KiB cut-off, and patches for every one.

    BF16 rows ``p62`` to ``p66`` over 128 inputs (15872 to 16896 bytes), a BF16
    layer ``pb`` whose bias carries it past the cut-off and gets its own bias
    patch, INT8 ConvRot rows ``q58`` to ``q66`` over 256 inputs (the scale and the
    layer record add to the INT8 bytes), and a 32 KiB layer ``r`` whose LoRA
    resizes it. Returns (root, patches).
    """
    import json

    import comfy.lora
    import comfy.ops
    import torch
    from comfy.quant_ops import QuantizedTensor

    ops = comfy.ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16)
    gen = torch.Generator().manual_seed(11)

    class _Root(torch.nn.Module):
        def get_dtype_inference(self):
            return torch.bfloat16

    root = _Root()
    layout = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})
    shapes = {f"p{rows}": (rows, 128, False) for rows in range(62, 67)}
    shapes.update({f"q{rows}": (rows, 256, False) for rows in range(58, 67)}, pb=(64, 128, True), r=(128, 128, False))
    for name, (rows, inputs, bias) in shapes.items():
        layer = ops.Linear(inputs, rows, bias=bias)
        root.add_module(name, layer)
        w = (torch.randn(rows, inputs, generator=gen) * 0.05).to(torch.bfloat16)
        sd = {"weight": w}
        if name.startswith("q"):
            qt = QuantizedTensor.from_float(w, "TensorWiseINT8Layout", per_channel=True, convrot=True,
                                            convrot_groupsize=256, is_weight=True)
            sd = {"weight": qt._qdata, "weight_scale": qt._params.scale,
                  "comfy_quant": torch.tensor(list(layout.encode()), dtype=torch.uint8)}
        if bias:
            sd["bias"] = torch.randn(rows, generator=gen).to(torch.bfloat16)
        layer.load_state_dict(sd, strict=True)
    lora, key_map = {}, {}
    for name, (rows, inputs, _bias) in shapes.items():
        lora[f"{name}.lora_up.weight"] = torch.randn(rows, 2, generator=gen) * 0.01
        lora[f"{name}.lora_down.weight"] = torch.randn(2, inputs, generator=gen) * 0.01
        key_map[name] = f"{name}.weight"
    lora["pb.diff_b"] = torch.randn(64, generator=gen).to(torch.bfloat16) * 0.01
    patches = comfy.lora.load_lora(lora, key_map)
    patches["r.weight"] = ("diff", ((torch.randn(160, 128, generator=gen) * 0.01).to(torch.bfloat16),
                                    {"pad_weight": True}))
    return root, patches


def stock_load_choice(root, patches) -> dict:
    """Run stock's ModelPatcherDynamic.load on CPU; return each patched key's path.

    "cast" is a key the load gave a cast-time LowVramPatch, "classic" one it
    force-loaded through patch_weight_to_device. aimdo's host buffer is stood
    in for the call: on CPU the load makes no vbar and pins nothing.
    """
    import comfy.model_patcher
    import comfy.utils
    import comfy_aimdo.host_buffer
    import torch

    class _HostBuffer:
        def __init__(self, *_args, **_kwargs):
            pass

    cpu = torch.device("cpu")
    real_buffer = comfy_aimdo.host_buffer.HostBuffer
    comfy_aimdo.host_buffer.HostBuffer = _HostBuffer
    dynamic = object.__new__(comfy.model_patcher.ModelPatcherDynamic)  # its __new__ reroutes a CPU device
    forced = []
    try:
        dynamic.__init__(root, cpu, cpu)
        dynamic.add_patches(patches, 1.0)
        assert set(dynamic.patches) == set(patches), "add_patches dropped a key"
        stock_bake = dynamic.patch_weight_to_device

        def recording(key, *args, **kwargs):
            forced.append(key)
            return stock_bake(key, *args, **kwargs)

        dynamic.patch_weight_to_device = recording
        dynamic.load(cpu)
    finally:
        comfy_aimdo.host_buffer.HostBuffer = real_buffer
        getattr(root, "dynamic_patchers", set()).discard(id(dynamic))  # no unpin at collection
    choice = {}
    for key in patches:
        module_key, _, param = key.rpartition(".")
        patch = getattr(comfy.utils.get_attr(root, module_key), f"{param}_lowvram_function", None)
        if isinstance(patch, comfy.model_patcher.LowVramPatch):
            choice[key] = "cast"
        elif key in forced:
            choice[key] = "classic"
    return choice


def check_stock_choice() -> int:
    """The bake gives a cast-time weight to exactly the keys stock's own load does."""
    import comfy.model_management as mm
    import comfy.model_patcher
    import torch

    from dgx_monarch.actor import lora_cast_time

    stock_root, stock_patches = _choice_model()
    stock = stock_load_choice(stock_root, stock_patches)
    root, patches = _choice_model()
    cpu = torch.device("cpu")
    patcher = comfy.model_patcher.ModelPatcher(root, cpu, cpu)
    patcher.add_patches(patches, 1.0)
    assert set(stock) == set(patches), f"stock's load chose no path for {set(patches) - set(stock)}"
    for key in patches:
        ours = "cast" if lora_cast_time.stock_patches_at_cast_time(patcher, key) else "classic"
        assert ours == stock[key], (
            f"BAKE CHOICE DIVERGENCE: {key}; stock's load takes the {stock[key]} path and the "
            f"bake takes {ours}")
        baked = lora_cast_time.cast_time_weight(patcher, key, cpu)
        assert (baked is not None) == (ours == "cast"), key
    # The matrix straddles the cut-off in both kinds, so a moved cut-off shows.
    for prefix in ("p", "q"):
        sizes: dict[str, list[int]] = {}
        for name, layer in root.named_children():
            if name[0] == prefix and name[1:].isdigit():
                sizes.setdefault(stock[f"{name}.weight"], []).append(mm.module_size(layer))
        assert set(sizes) == {"cast", "classic"}, (prefix, sizes)
        assert max(sizes["classic"]) <= lora_cast_time.SMALL_MODULE_BYTES < min(sizes["cast"]), (prefix, sizes)
        print(f"stock's cut-off ({prefix}): classic up to {max(sizes['classic'])} bytes, "
              f"cast time from {min(sizes['cast'])}")
    assert stock["pb.bias"] == "cast" and stock["r.weight"] == "classic", stock
    print(f"equivalent: stock's cast-time choice on {len(patches)} keys across the 16 KiB cut-off")
    return len(patches)


if __name__ == "__main__":
    main()
