"""The vendored Triton INT8 GEMM against this box's comfy-kitchen CUTLASS kernel, on a GPU.

Each case builds comfy-kitchen's own call: x int8 (M, K), w int8 (N, K),
x_scale float32 (M,), w_scale float32 (N,) or one value, a bias in the output
dtype or none, out (M, N) in bf16, fp16 or fp32. CUTLASS's
``_C.cutlass_int8_dequant`` writes the reference. The Triton kernel must match
it with torch.equal twice: called directly, and through ``CProxy``, the path a
render takes. The shapes are the 2026-10-28 harness's 13 (MiniMax H3 at 29184
rows per rank, Krea2 and Qwen stills, odd rows) and 7 more that reach the
kernel's masked tiers, M at exactly 128, H3's odd 29221 rows and the
Ideogram4 and Chroma linears of docs/BENCHMARKS.md.

The check runs only on a CUDA device with comfy-kitchen installed, and only
when ``DGXM_INT8_GEMM_EQUALITY_JSON`` names the report to write, so the CPU
battery never loads a GPU beside a render. Run it on an idle box before adding
a comfy-kitchen release to ``VALIDATED_KITCHEN`` (docs/MODELS.md):

    DGXM_INT8_GEMM_EQUALITY_JSON=int8-gemm-equality.json \\
        pytest -q tests/test_int8_gemm_triton_equality.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

from dgx_monarch.kernels import int8_gemm_triton as vendored

REPORT_ENV = "DGXM_INT8_GEMM_EQUALITY_JSON"
REPORT = os.environ.get(REPORT_ENV, "")
UPSTREAM = ("kanthai/gb10-comfyui-tuning@d68916bf06c7ab01e0c557c67c64a1a06eb0ceef, "
            "patches/ck_int8_patch.py")

pytestmark = [
    pytest.mark.skipif(not REPORT, reason=f"set {REPORT_ENV} to the report path to run the GPU check"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernels need a CUDA device"),
]

# (M, N, K). The first 13 are the 2026-10-28 harness's shapes.
SHAPES = [
    (29184, 21504, 5376), (29184, 5376, 5376), (29184, 28672, 5376), (29184, 5376, 14336),
    (4032, 12288, 3072), (4032, 3072, 12288), (4032, 9216, 3072), (4096, 3072, 3072),
    (1095, 2048, 1024), (2049, 4096, 4096), (129, 256, 256), (127, 256, 256), (8336, 21504, 5376),
    # M at exactly MIN_M; H3's odd per-rank rows; Ideogram4 adaln and qkv;
    # Chroma linear1 at 6144 rows; K a multiple of 16 but not of 64 (the
    # masked K loop); N off the tile and K under one tile.
    (128, 256, 256), (29221, 5376, 5376), (8192, 18432, 512), (1024, 9216, 3072),
    (6144, 15360, 3072), (4100, 3072, 3088), (300, 200, 48),
]
OUT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def _kitchen():
    pytest.importorskip("comfy_kitchen", reason="the reference is comfy-kitchen's CUTLASS kernel")
    from comfy_kitchen.backends import cuda as ckc
    from comfy_kitchen.backends.eager.quantization import DTYPE_CODE_TO_DTYPE

    current = ckc._C
    real = current._real if isinstance(current, vendored.CProxy) else current
    if real is None or not hasattr(real, "cutlass_int8_dequant"):
        pytest.skip("comfy-kitchen has no CUDA extension in this process")
    return ckc, real, {dt: code for code, dt in DTYPE_CODE_TO_DTYPE.items()}


def _operands(M, N, K, out_dtype, bias_on, per_tensor, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    x8 = torch.randint(-127, 128, (M, K), device=device, dtype=torch.int8, generator=g)
    w8 = torch.randint(-127, 128, (N, K), device=device, dtype=torch.int8, generator=g)
    xs = torch.rand(M, device=device, generator=g) * 2e-2 + 1e-3
    ws = torch.rand(1 if per_tensor else N, device=device, generator=g) * 2e-3 + 1e-4
    bias = ((torch.randn(N, device=device, generator=g) * 0.5).to(out_dtype) if bias_on
            else torch.empty(0, device=device, dtype=out_dtype))
    return x8, w8, xs, ws, bias


def _call(fn, ckc, code_of, x8, w8, xs, ws, bias, out):
    """One cutlass_int8_dequant call exactly as comfy-kitchen's int8 path makes it."""
    used = fn(ckc._wrap_for_dlpack(x8), ckc._wrap_for_dlpack(w8),
              ckc._wrap_for_dlpack(xs.reshape(-1, 1)),
              ckc._wrap_for_dlpack(ws if ws.numel() == w8.shape[0] else ws.expand(w8.shape[0]).contiguous()),
              ckc._wrap_for_dlpack(bias), ckc._wrap_for_dlpack(out), code_of[out.dtype],
              torch.cuda.current_stream(out.device).cuda_stream)
    torch.cuda.synchronize()
    return bool(used)


def test_the_triton_kernel_equals_cutlass_on_every_case():
    ckc, real, code_of = _kitchen()
    device = torch.device("cuda")
    gemm = vendored.build_triton_gemm()
    calls = []

    def counted(*args):
        calls.append(1)
        return gemm(*args)

    failures = []
    proxy = vendored.CProxy(real, counted, failures.append)
    results = []
    for M, N, K in SHAPES:
        for out_dtype in OUT_DTYPES:
            for bias_on in (False, True):
                for per_tensor in (False, True):
                    x8, w8, xs, ws, bias = _operands(M, N, K, out_dtype, bias_on, per_tensor,
                                                     len(results), device)
                    ref = torch.zeros(M, N, device=device, dtype=out_dtype)
                    used = _call(real.cutlass_int8_dequant, ckc, code_of, x8, w8, xs, ws, bias, ref)
                    direct = torch.zeros_like(ref)
                    gemm(x8, w8, xs, ws, bias, direct)
                    torch.cuda.synchronize()
                    before = len(calls)
                    through = torch.zeros_like(ref)
                    proxy_used = _call(proxy.cutlass_int8_dequant, ckc, code_of, x8, w8, xs, ws, bias, through)
                    results.append({
                        "M": M, "N": N, "K": K, "out": str(out_dtype).split(".")[-1], "bias": bias_on,
                        "per_tensor_w_scale": per_tensor, "cutlass_used": used, "proxy_used": proxy_used,
                        "triton_ran_in_proxy": len(calls) > before,
                        "equal_direct": bool(torch.equal(ref, direct)),
                        "equal_proxy": bool(torch.equal(ref, through)),
                    })
                    del x8, w8, xs, ws, bias, ref, direct, through
    torch.cuda.empty_cache()

    import comfy_kitchen
    import triton

    unequal = [r for r in results if not (r["equal_direct"] and r["equal_proxy"])]
    columns = list(results[0])
    report = {
        "check": "vendored Triton INT8 GEMM against comfy-kitchen's CUTLASS cutlass_int8_dequant, torch.equal per case",
        "date": time.strftime("%Y-%m-%d"),
        "kernel_source": UPSTREAM,
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda, "triton": triton.__version__,
                     "comfy_kitchen": vendored.kitchen_version() or getattr(comfy_kitchen, "__version__", None)},
        "validated_set": list(vendored.VALIDATED_KITCHEN),
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "shapes": len(SHAPES),
        "cases": len(results),
        "cutlass_declined": sum(not r["cutlass_used"] for r in results),
        "equal_direct": sum(r["equal_direct"] for r in results),
        "equal_through_proxy": sum(r["equal_proxy"] for r in results),
        "triton_ran_in_proxy": sum(r["triton_ran_in_proxy"] for r in results),
        "proxy_kernel_failures": len(failures),
        "unequal": len(unequal),
        "columns": columns,
    }
    # The summary indented, then one compact line per case.
    rows = ",\n  ".join(json.dumps([r[c] for c in columns]) for r in results)
    text = json.dumps(report, indent=1)[:-2] + ',\n "rows": [\n  ' + rows + "\n ]\n}\n"
    assert json.loads(text)["rows"][0] == [results[0][c] for c in columns]
    Path(REPORT).write_text(text, encoding="utf-8")
    assert not failures, failures
    assert all(r["cutlass_used"] for r in results), "CUTLASS declined a case this table expects it to take"
    assert not unequal, unequal[:5]
    # The proxy keeps rows under MIN_M on CUTLASS and runs Triton for the rest.
    for r in results:
        assert r["triton_ran_in_proxy"] is (r["M"] >= vendored.MIN_M), r


def test_the_proxy_hands_cutlass_every_call_it_declines():
    ckc, real, code_of = _kitchen()
    device = torch.device("cuda")
    calls = []
    proxy = vendored.CProxy(real, lambda *args: calls.append(args), lambda exc: None)

    def both(x8, w8, xs, ws, bias, shape, dtype=torch.bfloat16):
        ref = torch.zeros(shape, device=device, dtype=dtype)
        through = torch.zeros_like(ref)
        used = _call(real.cutlass_int8_dequant, ckc, code_of, x8, w8, xs, ws, bias, ref)
        proxy_used = _call(proxy.cutlass_int8_dequant, ckc, code_of, x8, w8, xs, ws, bias, through)
        return used, proxy_used, torch.equal(ref, through)

    empty = torch.empty(0, device=device, dtype=torch.bfloat16)
    # K not a multiple of 16 and a strided activation: the proxy hands each call
    # to CUTLASS and returns what CUTLASS returns (comfy-kitchen 0.2.40's CUTLASS
    # takes K = 1000 itself; a declined call goes on to comfy-kitchen's cuBLAS).
    x8, w8, xs, ws, _ = _operands(256, 256, 1000, torch.bfloat16, False, False, 1, device)
    used, proxy_used, equal = both(x8, w8, xs, ws, empty, (256, 256))
    assert (proxy_used, equal) == (used, True)
    wide, w8, xs, ws, _ = _operands(256, 256, 512, torch.bfloat16, False, False, 2, device)
    used, proxy_used, equal = both(wide[:, :256], w8[:, :256].contiguous(), xs, ws, empty, (256, 256))
    assert (proxy_used, equal) == (used, True)
    # Rows under MIN_M stay on CUTLASS.
    x8, w8, xs, ws, _ = _operands(vendored.MIN_M - 1, 256, 256, torch.bfloat16, False, False, 3, device)
    assert both(x8, w8, xs, ws, empty, (vendored.MIN_M - 1, 256)) == (True, True, True)
    assert calls == [], "the Triton kernel ran on a call the proxy must decline"
    # The fused-residual kernel is another extension function, untouched.
    if hasattr(real, "cutlass_int8_dequant_residual"):
        assert proxy.cutlass_int8_dequant_residual is real.cutlass_int8_dequant_residual
