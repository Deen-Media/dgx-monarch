# MIT License
#
# Copyright (c) 2026 kanthai
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
"""Triton INT8 GEMM for comfy-kitchen's INT8 ConvRot linears on GB10 (sm_121).

Vendored from kan11's gb10-comfyui-tuning
(https://github.com/kanthai/gb10-comfyui-tuning), file patches/ck_int8_patch.py
at commit d68916bf06c7ab01e0c557c67c64a1a06eb0ceef, under the MIT license
above. The changes listed below are distributed under the same terms. It is the
repository's one vendored source file (docs/DESIGN.md section 3.1,
LICENSE-NOTES.md, THIRD-PARTY-NOTICES.txt).

comfy-kitchen routes INT8 linears on sm_80 and newer to
``_C.cutlass_int8_dequant``: an INT8 GEMM with the per-row activation scale,
the per-channel weight scale and the bias in its epilogue. ``CProxy`` stands in
for the extension module and runs that one call as the Triton kernel below.
Fused-residual calls use another extension function and stay on CUTLASS, and
the proxy hands CUTLASS itself the rows under ``MIN_M`` and every call with K
not a multiple of 16 or a strided tensor, so those take the stock route,
cuBLAS included when CUTLASS declines one. Every other attribute falls through
to the real extension.

The kernel body and the proxy's decline rules are upstream's. Changes for
dgx-monarch:

- No mode file, no ``.pth`` file and no install at import. ``int8_gemm.py``
  beside this file owns the knob, the install points, the version guard's use
  and every log line, and calls into this module.
- The cublas mode and the fixed Python.h fallback directory are gone; a missing
  Python.h is one of ``int8_gemm.py``'s fallback reasons.
- ``VALIDATED_KITCHEN`` admits 0.2.40 beside 0.2.37 (docs/VALIDATION.md, "INT8
  GEMM kernel comparison").
- A failed Triton build or launch reports through a callback instead of
  printing a traceback; the import hook calls back instead of applying a mode.
- Names lost their leading underscore, and the code gained type hints.

The top level imports the standard library only: actors register the import
hook before CUDA is initialized, and torch and Triton load inside functions.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any, cast

TARGET = "comfy_kitchen.backends.cuda"
TOP = "comfy_kitchen"
# The comfy-kitchen releases this kernel was proven exact against. A release
# outside the set keeps stock CUTLASS; tests/test_int8_gemm_kitchen_version.py
# fails when ComfyUI's requirements or the installed package move past it.
VALIDATED_KITCHEN = ("0.2.37", "0.2.40")
MIN_M = 128


def kitchen_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("comfy-kitchen")
    except Exception:
        return None


def build_triton_gemm() -> Callable[..., None]:
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(a, b, xs, ws, bias, c, M, N, K,
                sam, sak, sbn, sbk, scm, scn, ws_n,
                HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, G: tl.constexpr, EVEN: tl.constexpr, EVEN_K: tl.constexpr):
        pid = tl.program_id(0)
        npm = tl.cdiv(M, BM)
        npn = tl.cdiv(N, BN)
        gid = pid // (G * npn)
        fm = gid * G
        gs = min(npm - fm, G)
        pm = fm + (pid % gs)
        pn = (pid % (G * npn)) // gs
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        # int64 offsets: M*N or M*K past 2^31 elements (CFG batches, long clips) would wrap in int32
        rm64 = rm.to(tl.int64)
        rn64 = rn.to(tl.int64)
        A = a + rm64[:, None] * sam + rk[None, :] * sak
        B = b + rn64[None, :] * sbn + rk[:, None] * sbk
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        for k0 in range(0, K, BK):
            if EVEN:   # M, N, K are tile multiples: unmasked loads, ~6% faster sustained (2026-10-26)
                acc += tl.dot(tl.load(A), tl.load(B))
            elif EVEN_K:   # only K is a tile multiple: row/column masks only, ~4% faster
                acc += tl.dot(tl.load(A, mask=rm[:, None] < M, other=0),
                              tl.load(B, mask=rn[None, :] < N, other=0))
            else:
                km = (k0 + rk) < K
                acc += tl.dot(tl.load(A, mask=(rm[:, None] < M) & km[None, :], other=0),
                              tl.load(B, mask=(rn[None, :] < N) & km[:, None], other=0))
            A += BK * sak
            B += BK * sbk
        out = acc.to(tl.float32) * tl.load(xs + rm, mask=rm < M, other=0.0)[:, None]
        out = out * tl.load(ws + rn * ws_n, mask=rn < N, other=0.0)[None, :]
        if HAS_BIAS:
            out += tl.load(bias + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
        C = c + rm64[:, None] * scm + rn64[None, :] * scn
        tl.store(C, out.to(c.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))

    def gemm(x8, w8, x_scale, w_scale, bias, out):
        M, K = x8.shape
        N = w8.shape[0]
        xs = x_scale.reshape(-1)
        ws = w_scale.reshape(-1)
        ws_n = 0 if ws.numel() == 1 else 1
        has_bias = bias.numel() > 0
        # 128x128x64 / 8 warps / 4 stages / group 8: best or within 1% on every
        # H3 shape at 29k and 36k tokens (sweeps 2026-10-25/26; 48 tile/stage/warp variants).
        # Unmasked K loop when every dim is a tile multiple: 161-165 -> 174-176 TOPS, bit-identical.
        even_k = K % 64 == 0
        even = even_k and M % 128 == 0 and N % 128 == 0
        grid = (triton.cdiv(M, 128) * triton.cdiv(N, 128),)
        _kernel[grid](x8, w8, xs, ws, bias if has_bias else out, out, M, N, K,
                      x8.stride(0), x8.stride(1), w8.stride(0), w8.stride(1),
                      out.stride(0), out.stride(1), ws_n,
                      HAS_BIAS=has_bias, BM=128, BN=128, BK=64, G=8, EVEN=even, EVEN_K=even_k,
                      num_warps=8, num_stages=4)

    return gemm


class CProxy:
    """Stands in for comfy_kitchen.backends.cuda._C and takes over cutlass_int8_dequant."""

    def __init__(self, real: Any, gemm: Callable[..., None],
                 on_failure: Callable[[BaseException], None]) -> None:
        self._real = real
        self._gemm = gemm
        self._on_failure = on_failure
        self._broken = False

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._real, name)

    def cutlass_int8_dequant(self, x, w, x_scale, w_scale, bias, out, dtype_code, stream):
        import torch
        tx, tw, txs, tws, tb, tout = (torch.from_dlpack(t) for t in (x, w, x_scale, w_scale, bias, out))
        M, K = tx.shape if tx.dim() == 2 else (0, 0)
        N = tw.shape[0] if tw.dim() == 2 else 0
        # Mirror CUTLASS eligibility (K % 16, contiguous) so anything CUTLASS would decline (and ck then
        # routes to cuBLAS) keeps the stock route; the Triton kernel itself would handle those too.
        if (self._broken or M < MIN_M or K % 16 or not (tx.is_contiguous() and tw.is_contiguous() and tout.is_contiguous())
                or tx.dtype != torch.int8 or tw.dtype != torch.int8
                or tw.dim() != 2 or tw.shape[1] != K or tuple(tout.shape) != (M, N)
                or tout.dtype not in (torch.bfloat16, torch.float16, torch.float32)
                or txs.dtype != torch.float32 or txs.numel() != M
                or tws.dtype != torch.float32 or tws.numel() not in (1, N)
                or tb.numel() not in (0, N) or (tb.numel() and tb.dtype != tout.dtype)):
            return self._real.cutlass_int8_dequant(*(t.__dlpack__(stream=-1) for t in (tx, tw, txs, tws, tb, tout)),
                                                   dtype_code, stream)
        try:
            self._gemm(tx, tw, txs, tws, tb, tout)
        except Exception as exc:  # e.g. Triton cannot build its launcher: stay on stock CUTLASS
            self._broken = True
            try:
                self._on_failure(exc)
            except Exception:
                pass
            return self._real.cutlass_int8_dequant(*(t.__dlpack__(stream=-1) for t in (tx, tw, txs, tws, tb, tout)),
                                                   dtype_code, stream)
        return True


def compile_safe_int8_linear(top: ModuleType) -> bool:
    """Route comfy_kitchen.int8_linear through its custom op while torch.compile traces.

    ComfyUI's linear_input_act path once called this plain Python function, so
    dynamo traced into the CUDA backend and failed on ``__dlpack__`` of a
    FakeTensor. While dynamo traces, the call goes to the registered custom op
    ``comfy_kitchen::int8_linear``, which has a fake kernel; eager calls are
    untouched. Returns whether the wrapper is in place.
    """
    import torch
    from comfy_kitchen.backends.eager.quantization import DTYPE_CODE_TO_DTYPE
    if kitchen_version() not in VALIDATED_KITCHEN:
        return False
    if getattr(top.int8_linear, "_ck_compile_safe", False):
        return True
    code_of = {dt: code for code, dt in DTYPE_CODE_TO_DTYPE.items()}
    eager = top.int8_linear

    def int8_linear(x, weight, weight_scale, bias=None, out_dtype=None, convrot=False,
                    convrot_groupsize=256, input_act=None, input_act_weight=None,
                    input_act_eps=0.0, residual=None, residual_scale=None):
        if torch.compiler.is_compiling():
            return torch.ops.comfy_kitchen.int8_linear(
                x, weight, weight_scale, bias, code_of[out_dtype or torch.bfloat16],
                convrot, convrot_groupsize, input_act, input_act_weight, input_act_eps,
                residual, residual_scale)
        return eager(x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize,
                     input_act, input_act_weight, input_act_eps, residual, residual_scale)

    cast(Any, int8_linear)._ck_compile_safe = True
    cast(Any, int8_linear).__wrapped__ = eager
    cast(Any, top).int8_linear = int8_linear
    return True


class Finder(importlib.abc.MetaPathFinder):
    """Calls ``on_exec(name, module)`` right after comfy_kitchen or its CUDA backend executes."""

    def __init__(self, on_exec: Callable[[str, ModuleType], None]) -> None:
        self._on_exec = on_exec

    def find_spec(self, fullname: str, path: Sequence[str] | None,
                  target: ModuleType | None = None) -> importlib.machinery.ModuleSpec | None:
        if fullname not in (TARGET, TOP):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        loader, on_exec = spec.loader, self._on_exec
        orig_exec = getattr(loader, "exec_module", None)
        if orig_exec is None:
            return None

        def exec_module(module: ModuleType) -> None:
            orig_exec(module)
            try:
                on_exec(fullname, module)
            except Exception:
                pass

        cast(Any, loader).exec_module = exec_module
        return spec
