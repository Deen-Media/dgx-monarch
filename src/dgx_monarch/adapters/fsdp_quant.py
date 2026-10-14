"""Quantized checkpoints under FSDP: shard the bytes, rebuild the wrapper.

In comfy_kitchen 0.2.36, ``QuantizedTensor`` is a
wrapper subclass whose dispatch table knows the compute ops (linear, mm,
addmm, t) and a few generic ones (detach, clone, to, copy_), but not the
structural ops FSDP2's init and all-gather run on a parameter (chunk,
narrow, new_zeros, as_strided, record_stream, and view outside the fp8
layouts). Those gaps, not the scale tensors, are what keep a plain
QuantizedTensor out of FSDP2.

``ShardedQuantWeight`` is a ``QuantizedTensor`` subclass that adds those
structural ops, applied to the quantized bytes only, and the two FSDP2
all-gather extension hooks. Every layout parameter stays whole on every rank.
A per-tensor scale is one value. A per-row fp32 scale is 4 bytes per row
(144 KiB for a 36864-row weight), so each rank keeps all of it rather than
gathering it. The wrapper records which params run along the rows once, at
full size, and never infers that again from a shard: a shard's row count says
nothing about its params, and inferring it there made ranks disagree on what
to gather.

* ``fsdp_pre_all_gather`` hands FSDP exactly one input on every rank: this
  rank's quantized bytes, viewed as uint8 and padded to FSDP2's chunk size on
  the ranks that need it. The whole layout parameters travel as metadata;
* ``fsdp_post_all_gather`` rebuilds the full-size wrapper around the gathered
  bytes and those whole parameters, so the module sees the same
  ``QuantizedTensor`` type it sees resident and comfy's registered kernels run
  unchanged.

A shard's params therefore describe the full weight, not its own rows, so
only the rebuilt wrapper may be dequantized or used in compute.

Everything the class does not handle falls through to comfy_kitchen's own
dispatch, which is what keeps the compute path the resident one. Admitted
layouts are the per-tensor fp8 layouts and tensorwise int8 with or without the
convrot rotation, whose groups run along the reduction dim, so dim-0 row
chunks never split a group. Block-scaled layouts (mxfp8, nvfp4) and
transposed storage typed-refuse.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from .base import UnsupportedModelError

log = get_logger(__name__)

# Layout names (comfy_kitchen registry) with a registered shard layout.
ADMITTED_LAYOUTS = frozenset({
    "TensorCoreFP8Layout",
    "TensorCoreFP8E4M3Layout",
    "TensorCoreFP8E5M2Layout",
    "TensorWiseINT8Layout",
})


def _quantized_tensor_class():
    """Return the QuantizedTensor class ComfyUI re-exports from comfy_kitchen.

    Use ``comfy.quant_ops.QuantizedTensor`` when available. Unit environments
    without ComfyUI import the same class directly from comfy_kitchen.
    """
    try:
        from comfy.quant_ops import QuantizedTensor
    except ImportError:
        from comfy_kitchen.tensor import QuantizedTensor
    return QuantizedTensor


def shard_layout_refusal(layout_cls: str, params: Any) -> str | None:
    """Why this layout cannot be sharded, or None when it is admitted."""
    if layout_cls not in ADMITTED_LAYOUTS:
        return (f"layout {layout_cls} has no registered shard layout (admitted: "
                f"{', '.join(sorted(ADMITTED_LAYOUTS))}); block-scaled and packed "
                "layouts stay resident-only")
    if getattr(params, "transposed", False):
        return f"layout {layout_cls} stores the weight transposed; dim-0 chunks would split columns"
    return None


def _row_aligned(params: Any, rows: int) -> list[str]:
    """Names of tensor params whose first dim runs along the weight's rows.

    Call it only on a full-size weight (``from_quantized``). On a shard the
    row count no longer matches the params, which stay whole.
    """
    names = []
    for field in dataclasses.fields(params):
        value = getattr(params, field.name)
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == rows and rows > 1:
            names.append(field.name)
    return names


def _make(cls, qdata: torch.Tensor, layout_cls: str, params: Any,
          row_fields: tuple[str, ...] = ()):
    """A wrapper whose logical shape follows the (possibly chunked) qdata rows."""
    shape = tuple(qdata.shape)
    params = dataclasses.replace(params, orig_shape=shape)
    obj = cls.__new__(cls, qdata, layout_cls, params)
    cls.__init__(obj, qdata, layout_cls, params)
    obj._row_fields = tuple(row_fields)
    return obj


class ShardedQuantWeight:  # sharded_quant_class() replaces this on first use
    pass


def _build_class():
    QuantizedTensor = _quantized_tensor_class()
    aten = torch.ops.aten

    class _ShardedQuantWeight(QuantizedTensor):
        """A QuantizedTensor that FSDP2 can chunk, pad, gather, and rebuild."""

        # Names of the params that run along the rows, recorded once at full
        # size. They stay whole on every shard, like every other layout param.
        _row_fields = ()

        @classmethod
        def from_quantized(cls, qt):
            reason = shard_layout_refusal(qt._layout_cls, qt._params)
            if reason is not None:
                raise UnsupportedModelError(refusal(
                    RefusalClass.PHYSICS,
                    f"FSDP quantized shard: {reason}. Use a resident topology for "
                    "this checkpoint, or a per-tensor fp8 or tensorwise int8 file."))
            rows = int(qt._qdata.shape[0]) if qt._qdata.ndim else 1
            return _make(cls, qt._qdata, qt._layout_cls, qt._params,
                         tuple(_row_aligned(qt._params, rows)))

        def _rows(self) -> int:
            return int(self._qdata.shape[0]) if self._qdata.ndim else 1

        def _with(self, qdata: torch.Tensor, params: Any = None):
            return _make(type(self), qdata, self._layout_cls,
                         self._params if params is None else params, self._row_fields)

        def _slice_rows(self, start: int, stop: int):
            """Rows [start, stop) of the bytes; every layout param stays whole."""
            return self._with(self._qdata[start:stop])

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            qt = args[0] if args and isinstance(args[0], cls) else None
            if qt is not None:
                if func in (aten.split.Tensor, aten.chunk.default):
                    return qt._split(func, args, kwargs)
                if func in (aten.narrow.default, aten.slice.Tensor):
                    return qt._narrow(func, args, kwargs)
                if func in (aten.new_zeros.default, aten.new_empty.default):
                    return qt._new_rows(args[1], zeros=func is aten.new_zeros.default)
                if func in (aten.view.default, aten.reshape.default, aten._unsafe_view.default):
                    return qt._view(args[1])
                if func is aten.as_strided.default:
                    return qt._as_strided(args[1], args[2])
                if func is aten.record_stream.default:
                    qt._qdata.record_stream(args[1])
                    for field in dataclasses.fields(qt._params):
                        value = getattr(qt._params, field.name)
                        if isinstance(value, torch.Tensor):
                            value.record_stream(args[1])
                    return None
                if func in (aten.detach.default, aten.alias.default):
                    return qt._with(qt._qdata.detach())
                if func in (aten._to_copy.default, aten.to.dtype_layout, aten.to.dtype,
                            aten.to.device, aten.to.prim_Device):
                    return qt._to(args, kwargs, copy=func is aten._to_copy.default)
                if func is aten.clone.default:
                    return qt._with(qt._qdata.clone(), qt._params.clone())
                if func is aten.copy_.default:
                    return qt._copy_from(args[1], kwargs)
                if func is aten.is_contiguous.default:
                    return qt._qdata.is_contiguous()
                if func is aten.contiguous.default:
                    return qt if qt._qdata.is_contiguous() else qt._with(qt._qdata.contiguous())
                if func is aten._has_compatible_shallow_copy_type.default:
                    return True
            return QuantizedTensor.__torch_dispatch__(func, types, args, kwargs)

        def _split(self, func, args, kwargs):
            dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
            if dim not in (0, -self._qdata.ndim):
                raise RuntimeError("FSDP quantized shard: only dim-0 chunking is supported")
            rows = self._rows()
            if func is aten.chunk.default:
                chunk = math.ceil(rows / int(args[1]))
            else:
                size = args[1]
                if isinstance(size, (list, tuple)):
                    raise RuntimeError("FSDP quantized shard: uneven split lists are not supported")
                chunk = int(size)
            out = []
            start = 0
            while start < rows:
                stop = min(start + chunk, rows)
                out.append(self._slice_rows(start, stop))
                start = stop
            return out

        def _narrow(self, func, args, kwargs):
            if func is aten.narrow.default:
                dim, start, length = int(args[1]), int(args[2]), int(args[3])
                stop = start + length
            else:
                dim = int(args[1]) if len(args) > 1 else 0
                start = int(args[2]) if len(args) > 2 and args[2] is not None else 0
                stop = int(args[3]) if len(args) > 3 and args[3] is not None else self._rows()
                stop = min(stop, self._rows())
            if dim not in (0, -self._qdata.ndim):
                raise RuntimeError("FSDP quantized shard: only dim-0 views are supported")
            return self._slice_rows(start, stop)

        def _new_rows(self, size, *, zeros: bool):
            size = tuple(int(s) for s in size)
            if size == (0,):
                # FSDP2's _chunk_with_empty fills a rank whose dim-0 chunk is
                # empty with tensor.new_empty(0), a flat size for an N-D
                # parameter (for example, a [1, C] tensor at world 2). The
                # filler means "zero rows of this parameter", so it keeps the
                # column shape by construction.
                size = (0, *self._qdata.shape[1:])
            if len(size) != self._qdata.ndim or size[1:] != tuple(self._qdata.shape[1:]):
                raise UnsupportedModelError(
                    "FSDP quantized shard: a new buffer must keep the column shape")
            maker = self._qdata.new_zeros if zeros else self._qdata.new_empty
            # The params are not zeroed: FSDP2 skips the copy into an empty
            # rank's buffer, which must still carry the weight's whole scale.
            return self._with(maker(size))

        def _view(self, shape):
            shape = tuple(int(s) for s in shape)
            if shape == tuple(self._qdata.shape):
                return self._with(self._qdata)  # a view op must return a new object
            if shape == (-1,) or len(shape) == 1:
                # FSDP2 flattens the padded buffer for bookkeeping only; the
                # flattened view is never handed to a kernel.
                return self._with(self._qdata.reshape(-1))
            raise RuntimeError(
                f"FSDP quantized shard: view to {shape} is not a row view")

        def _as_strided(self, size, stride):
            size = tuple(int(s) for s in size)
            if size != tuple(self._qdata.shape):
                raise UnsupportedModelError(
                    "FSDP quantized shard: as_strided must keep the gathered shape")
            return self._with(self._qdata)  # same storage, new wrapper object

        def _to(self, args, kwargs, *, copy: bool):
            device = kwargs.get("device")
            for arg in args[1:]:
                if isinstance(arg, torch.device):
                    device = arg
                elif isinstance(arg, str) and (
                        arg in ("cpu", "cuda", "meta") or arg.startswith("cuda:")):
                    device = torch.device(arg)
            if device is None or torch.device(device) == self._qdata.device:
                return self._with(self._qdata.clone(), self._params.clone()) if copy else self
            qdata = self._qdata.to(device=device, non_blocking=bool(kwargs.get("non_blocking", False)))
            return self._with(qdata, self._params.to_device(torch.device(device)))

        def _copy_from(self, src, kwargs):
            if not isinstance(src, QuantizedTensor):
                raise TypeError(f"cannot copy {type(src).__name__} into a quantized shard")
            if src._layout_cls != self._layout_cls:
                raise TypeError(f"layout mismatch: {self._layout_cls} vs {src._layout_cls}")
            non_blocking = bool(kwargs.get("non_blocking", False))
            self._qdata.copy_(src._qdata, non_blocking=non_blocking)
            # Row-aligned params are whole and shared by every wrapper cut
            # from one weight, so a row copy never writes them.
            for field in dataclasses.fields(self._params):
                value = getattr(src._params, field.name)
                if isinstance(value, torch.Tensor) and field.name not in self._row_fields:
                    getattr(self._params, field.name).copy_(value, non_blocking=non_blocking)
            return self

        def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
            world = mesh.size()
            rows_total = int(outer_size[0])
            padded_rows = math.ceil(rows_total / world)
            for name in self._row_fields:
                # Every rank holds the same whole param, so this refuses on
                # every rank at once rather than leaving one in the gather.
                if getattr(self._params, name).shape[0] != rows_total:
                    raise RuntimeError(
                        f"FSDP quantized shard: row-aligned param {name} does not "
                        f"hold the weight's {rows_total} rows")
            qdata = self._qdata
            if qdata.shape[0] != padded_rows:
                pad = padded_rows - qdata.shape[0]
                qdata = torch.cat([qdata, qdata.new_zeros((pad, *qdata.shape[1:]))], dim=0)
            # One input on every rank. Both admitted layouts store one byte per
            # element, so the uint8 view has the padded sharded size torch
            # requires of every input.
            metadata = {
                "layout_cls": self._layout_cls,
                "qdtype": self._qdata.dtype,
                "rows": rows_total,
                "params": self._params,
            }
            return (qdata.view(torch.uint8),), metadata

        def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
            (data,) = all_gather_outputs
            rows = int(metadata["rows"])
            qdata = data.view(metadata["qdtype"])[:rows]
            params = dataclasses.replace(metadata["params"], orig_shape=tuple(qdata.shape))
            if out is not None:
                out._qdata = qdata
                out._params = params
                return None
            rebuilt = _make(type(self), qdata, metadata["layout_cls"], params, self._row_fields)
            return rebuilt, tuple(all_gather_outputs)

    _ShardedQuantWeight.__name__ = "ShardedQuantWeight"
    _ShardedQuantWeight.__qualname__ = "ShardedQuantWeight"
    return _ShardedQuantWeight


_CLASS: Any = None


def sharded_quant_class():
    """The wrapper class, built on first use so comfy_kitchen imports lazily."""
    global _CLASS, ShardedQuantWeight
    if _CLASS is None:
        _CLASS = _build_class()
        ShardedQuantWeight = _CLASS
    return _CLASS


def wrap_quantized_parameters(diffusion_model: torch.nn.Module) -> int:
    """Replace every QuantizedTensor parameter with a shardable wrapper.

    Returns the number of parameters wrapped. Refuses typed before touching
    anything when a layout has no registered shard layout.
    """
    QuantizedTensor = _quantized_tensor_class()
    cls = sharded_quant_class()
    targets: list[tuple[torch.nn.Module, str, Any]] = []
    for module in diffusion_model.modules():
        for name, param in list(module._parameters.items()):
            if param is None or not isinstance(param, QuantizedTensor):
                continue
            quant: Any = param
            reason = shard_layout_refusal(quant._layout_cls, quant._params)
            if reason is not None:
                raise UnsupportedModelError(refusal(
                    RefusalClass.PHYSICS,
                    f"FSDP quantized shard: {reason}. Use a resident topology for "
                    "this checkpoint, or a per-tensor fp8 or tensorwise int8 file."))
            targets.append((module, name, param))
    for module, name, param in targets:
        wrapped = cls.from_quantized(param)
        module._parameters[name] = torch.nn.Parameter(wrapped, requires_grad=False)
    return len(targets)


def local_nbytes(param: torch.Tensor) -> int:
    """Bytes a parameter really holds on this rank (quantized storage aware)."""
    local = getattr(param, "_local_tensor", param)
    nbytes = getattr(local, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    return local.numel() * local.element_size()
