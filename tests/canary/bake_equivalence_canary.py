"""Compare dgx-monarch's per-key bake with ComfyUI's bake on CPU.

ModelStore._bake_key follows patch_weight_to_device. Changes to dtype handling,
rounding or adapter math can make them diverge; runtime byte verification checks
pristine weights, so it cannot detect a bake mismatch. Ambient verification
checks baked results only for the keys sampled at each lazy swap (swap_verify,
default 2), after the user runs the changed ComfyUI.

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


if __name__ == "__main__":
    main()
