"""MOSS-Image kernel parity: the helpers the fused-norm route reuses, and the GPU seam.

The CPU half pins the FLUX.3 helper generalisations MOSS-Image reuses
(adapters/flux3_adaln_parity.py): ``loads_global`` reads the name it is given,
``route_global`` rebuilds stock's code with one global wrapped and leaves the
module alone, and both fused-norm messages carry the family-neutral prefix.

The GPU half is the seam behind every uly2 bit-identity claim, run on the
installed comfy-kitchen at MOSS-Image's own shapes (width 3072, 24 heads of
128): each rank window of a ``rms_adaln`` call, filled to stock's rows with
``at_stock_rows`` (prefix placement), returns stock's bits; the fused q/k norm
and RoPE returns stock's bits on strided thirds of a fused qkv with the
window's rotary rows; and the INT8 ConvRot linears, the fused SiTU input of
ffn_down included, are row-invariant. Any gating failure here is a design
stop for the uly2 rows (docs/ADAPTERS.md). The unfilled windows are printed
only, to show whether this kitchen still branches below 1024 kernel rows. It
needs a CUDA device and an idle pair; equality does not depend on the clock.
"""
from __future__ import annotations

import math
import types
from importlib import metadata

import pytest
import torch

from dgx_monarch.adapters import flux3_adaln_parity
from dgx_monarch.adapters.flux3_adaln_parity import (
    at_stock_rows,
    loads_global,
    loads_norm_global,
    pad_rows,
    route_global,
)

CALLS: list[str] = []


def modulated_rms_norm(x, norm, scale, shift):
    CALLS.append("stock rms")
    return x + shift


def modulated_norm(x, norm, scale, shift, modulation_dims=None):
    CALLS.append("stock layer")
    return x + shift


def _reads_rms(x, shift, scale=2.0, *, keyword=3.0):
    """Stand-in for MOSS-Image's Block.forward: one global, defaults on both kinds."""
    return modulated_rms_norm(x, None, scale, shift) * keyword


def _reads_layer(x, shift):
    return modulated_norm(x, None, None, shift)


def test_loads_global_reads_the_name_it_is_given():
    """loads_global finds a module global by name in a function's code, and only there."""
    assert loads_global(_reads_rms, "modulated_rms_norm")
    assert not loads_global(_reads_rms, "modulated_norm")
    assert not loads_norm_global(_reads_rms)
    assert loads_norm_global(_reads_layer) and loads_global(_reads_layer, "modulated_norm")
    assert not loads_global(len, "modulated_rms_norm")       # not a Python function
    assert loads_global(types.MethodType(_reads_rms, object()), "modulated_rms_norm")


def test_route_global_wraps_one_global_and_keeps_the_module_and_stocks_code():
    """route_global wraps one global for one call and leaves stock's code and the module as they were."""
    module_global = globals()["modulated_rms_norm"]
    seen: list = []

    def wrap(stock):
        seen.append(stock)

        def routed(x, norm, scale, shift):
            CALLS.append("routed")
            return stock(x, norm, scale, shift)

        return routed

    CALLS.clear()
    routed = route_global(_reads_rms, "modulated_rms_norm", wrap)
    assert seen == [module_global]
    assert routed.__code__ is _reads_rms.__code__
    assert routed.__defaults__ == _reads_rms.__defaults__ == (2.0,)
    assert routed.__kwdefaults__ == _reads_rms.__kwdefaults__ == {"keyword": 3.0}
    assert routed.__qualname__ == _reads_rms.__qualname__
    x = torch.ones(1, 2, 3)
    assert torch.equal(routed(x, x), _reads_rms(x, x))
    assert CALLS == ["routed", "stock rms", "stock rms"]
    assert globals()["modulated_rms_norm"] is module_global   # the module never changed


def test_route_global_refuses_a_global_that_is_not_callable(monkeypatch):
    """A missing or uncallable global raises a TypeError with the family-neutral prefix."""
    monkeypatch.setitem(globals(), "modulated_rms_norm", None)
    with pytest.raises(TypeError, match=r"^fused-norm parity: .*modulated_rms_norm is NoneType"):
        route_global(_reads_rms, "modulated_rms_norm", lambda stock: stock)
    monkeypatch.delitem(globals(), "modulated_rms_norm")
    with pytest.raises(TypeError, match="fused-norm parity"):
        route_global(_reads_rms, "modulated_rms_norm", lambda stock: stock)


def test_pad_rows_names_the_family_neutral_route():
    """pad_rows refuses a piece longer than stock's call with the family-neutral prefix."""
    with pytest.raises(ValueError, match=r"^fused-norm parity: a 11-row piece cannot sit in a 10-row"):
        pad_rows(11, 10)
    assert "MOSS-Image's binder" in (flux3_adaln_parity.__doc__ or "")


# --- the GPU seam ---------------------------------------------------------------------

WIDTH, HEADS, HEAD_DIM, EPS = 3072, 24, 128, 1e-6
# (batch, N, ranks): 1 MP-class streams on both sides of the 1024 kernel-row
# branch at batch 1 and 2, the vendor tag prompt at 1024 square (N 4119) on two
# and four ranks, and a 4 MP stream.
GEOMETRIES = [(1, 1047, 2), (2, 1047, 2), (1, 1031, 2), (1, 4119, 2), (1, 4119, 4), (1, 16420, 2)]
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel branches need a CUDA device")


def _kitchen():
    return pytest.importorskip("comfy_kitchen", reason="the seam is comfy-kitchen's kernel")


def _windows(rows: int, ranks: int) -> list[tuple[int, int]]:
    size = math.ceil(rows / ranks)
    return [(start, min(start + size, rows)) for start in range(0, rows, size)]


def _differ(candidate: torch.Tensor, reference: torch.Tensor) -> int:
    return int((candidate != reference).sum())


def _header() -> str:
    return (f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
            f"comfy-kitchen {metadata.version('comfy-kitchen')}")


@GPU
@pytest.mark.parametrize("batch,rows,ranks", GEOMETRIES)
def test_every_rank_window_of_rms_adaln_filled_to_stocks_rows_returns_stocks_bits(batch, rows, ranks):
    ck = _kitchen()
    generator = torch.Generator(device="cuda").manual_seed(rows * 10 + batch)

    def draw(*shape, scale=1.0, offset=0.0):
        values = torch.randn(*shape, generator=generator, device="cuda", dtype=torch.float32)
        return (values * scale + offset).to(torch.bfloat16)

    x = draw(batch, rows, WIDTH, scale=3.0)
    weight = draw(WIDTH, scale=0.1, offset=1.0)
    scale, shift = draw(batch, 1, WIDTH, scale=0.2), draw(batch, 1, WIDTH, scale=0.2)
    folded = weight * (1 + scale) - 1       # modulated_rms_norm's folded scale, [B, 1, D]
    stock = ck.rms_adaln(x, folded, shift, EPS)
    print(_header())
    for start, end in _windows(rows, ranks):
        piece = x[:, start:end].contiguous()
        filled = at_stock_rows(lambda call: ck.rms_adaln(call, folded, shift, EPS), piece, rows)
        unfilled = ck.rms_adaln(piece, folded, shift, EPS)
        print(f"rms_adaln B{batch} N{rows} window [{start}, {end}): filled "
              f"{_differ(filled, stock[:, start:end])} differ, unfilled "
              f"{_differ(unfilled, stock[:, start:end])} differ (informational)")
        assert torch.equal(filled, stock[:, start:end]), (batch, rows, start, end)


@GPU
@pytest.mark.parametrize("batch,rows,ranks", GEOMETRIES)
def test_the_fused_qk_norm_and_rope_on_a_window_returns_stocks_bits(batch, rows, ranks):
    """The call layout the sharded forward hands it: strided thirds of a fused qkv
    viewed [B, n, 24, 128], and the window's rows of the rotary table."""
    ck = _kitchen()
    generator = torch.Generator(device="cuda").manual_seed(rows + 7 * batch)
    qkv = torch.randn(batch, rows, 3 * HEADS * HEAD_DIM, generator=generator, device="cuda",
                      dtype=torch.float32).to(torch.bfloat16)
    q, k, _v = qkv.chunk(3, dim=-1)
    q, k = q.view(batch, rows, HEADS, HEAD_DIM), k.view(batch, rows, HEADS, HEAD_DIM)
    assert not q.is_contiguous()
    angle = torch.rand(1, rows, 1, 64, generator=generator, device="cuda") * 6.283
    pe = torch.stack([angle.cos(), -angle.sin(), angle.sin(), angle.cos()], -1).view(
        1, rows, 1, 64, 2, 2).contiguous()
    q_scale = (torch.randn(HEAD_DIM, generator=generator, device="cuda") * 0.1 + 1).to(torch.bfloat16)
    k_scale = (torch.randn(HEAD_DIM, generator=generator, device="cuda") * 0.1 + 1).to(torch.bfloat16)
    stock_q, stock_k = ck.rms_rope_split_half(q, k, pe, q_scale, k_scale, EPS)
    for start, end in _windows(rows, ranks):
        # A rank's own fused qkv output: contiguous rows, strided thirds, as on the mesh.
        piece_q, piece_k, _piece_v = qkv[:, start:end].contiguous().chunk(3, dim=-1)
        shape = (batch, end - start, HEADS, HEAD_DIM)
        out_q, out_k = ck.rms_rope_split_half(piece_q.view(shape), piece_k.view(shape),
                                              pe[:, start:end].contiguous(), q_scale, k_scale, EPS)
        assert torch.equal(out_q, stock_q[:, start:end]), ("q", batch, rows, start, end)
        assert torch.equal(out_k, stock_k[:, start:end]), ("k", batch, rows, start, end)


# The four block linears: (output rows, input width, input activation).
INT8_SHAPES = {"qkv": (9216, 3072, None), "o": (3072, 3072, None),
               "ffn_up_gate": (16384, 3072, None), "ffn_down": (3072, 16384, "situ")}


@GPU
@pytest.mark.parametrize("name", sorted(INT8_SHAPES))
@pytest.mark.parametrize("batch,rows", [(1, 1047), (2, 1047), (1, 4119)])
def test_the_int8_convrot_linears_are_row_invariant(name, batch, rows):
    """ffn_down takes the [hidden | gate] input whole and applies SiTU inside its
    quantizer, which kitchen fuses when ComfyUI calls ``linear_input_act``."""
    ck = _kitchen()
    out_rows, in_width, act = INT8_SHAPES[name]
    weight_width = in_width // 2 if act == "situ" else in_width
    generator = torch.Generator(device="cuda").manual_seed(out_rows + rows + batch)
    weight = torch.randint(-127, 128, (out_rows, weight_width), generator=generator,
                           device="cuda", dtype=torch.int8)
    weight_scale = torch.rand(out_rows, 1, generator=generator, device="cuda") * 1e-3
    x = torch.randn(batch, rows, in_width, generator=generator, device="cuda",
                    dtype=torch.float32).to(torch.bfloat16)

    def linear(rows_in):
        return ck.int8_linear(rows_in, weight, weight_scale, None, torch.bfloat16, convrot=True,
                              convrot_groupsize=256, input_act=act)

    stock = linear(x.reshape(-1, in_width)).view(batch, rows, out_rows)
    for start, end in _windows(rows, 2):
        piece = linear(x[:, start:end].reshape(-1, in_width)).view(batch, end - start, out_rows)
        assert torch.equal(piece, stock[:, start:end]), (name, batch, rows, start, end)
