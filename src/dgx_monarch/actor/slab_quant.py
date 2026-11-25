"""Quantized weights on a slab: moving a wrapper's payload and scales back home.

``WeightSlab`` takes these methods from ``QuantReabsorb``. ComfyUI's bake and
its loader leave a quantized weight's payload or scales in cudaMalloc beside
their slab regions; these methods copy them back into the regions the file
gave them, count what cannot move, and say before a bake whether its payload
will fit. They reach the live model and the slab's memory only through
``WeightSlab._live`` and ``WeightSlab._wrap``, so ``actor/slab.py`` stays the
one module that owns both.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import torch

from ..log import get_logger
from ..transfer_utils import failure_summary, safe_call

log = get_logger(__name__)


class QuantReabsorb:
    """The quantized half of ``WeightSlab``'s reabsorb, per weight."""

    key_map: dict[str, Any]
    regions: dict[str, Any]
    arena: Any
    _strays: dict[str, int]

    def contains(self, ptr: int) -> bool:
        raise NotImplementedError

    def _region_for_stray(self, name: str, tensor: torch.Tensor) -> Any:
        raise NotImplementedError

    @staticmethod
    def _live(module: Any, name: str) -> Any:
        raise NotImplementedError

    @staticmethod
    def _wrap(ptr: int, nbytes: int, cuda: bool) -> torch.Tensor:
        raise NotImplementedError

    def _region_for(self, key: str, nbytes: int) -> Any:
        region = self.key_map.get(key) or self.regions.get(key)
        return region if region is not None and region.nbytes == nbytes else None

    def _outside(self, tensors: list[torch.Tensor]) -> int:
        return sum(t.numel() * t.element_size() for t in tensors
                   if t.numel() and not self.contains(t.data_ptr()))

    def _into_region(self, region: Any, tensor: torch.Tensor) -> torch.Tensor:
        """Copy ``tensor`` into ``region`` and return the slab view that holds it."""
        nbytes = tensor.numel() * tensor.element_size()
        dest = self._wrap(self.arena.base + region.offset, nbytes, tensor.device.type == "cuda")
        held = dest.view(tensor.dtype).reshape(tensor.shape)
        held.copy_(tensor.detach())
        return held

    @staticmethod
    def _wrapper(live: Any) -> Any:
        # the wrapper attrs may sit on the Parameter or on its .data,
        # depending on how the subclass propagates through Parameter()
        for cand in (live, getattr(live, "data", None)):
            if cand is not None and getattr(cand, "_qdata", None) is not None:
                return cand
        return None

    @staticmethod
    def _scale_fields(wrapper: Any, layout_name: str, params: Any) -> dict[str, tuple[str | None, torch.Tensor]]:
        """{params field: (file component suffix or None, tensor)} for each tensor the params hold.

        The layout's own serialization map (``state_dict_tensors``) names the
        file component of each scale, so ``block_scale`` maps to
        ``weight_scale`` on NVFP4 and ``scale`` to ``weight_scale_2``. Params
        that are not a dataclass (a stand-in) carry no tensors this reads.
        """
        if not dataclasses.is_dataclass(params) or isinstance(params, type):
            return {}
        tensors = {field.name: getattr(params, field.name) for field in dataclasses.fields(params)
                   if isinstance(getattr(params, field.name), torch.Tensor)}
        suffixes: dict[str, str] = {}
        try:
            import comfy.quant_ops as quant_ops

            named = quant_ops.get_layout_class(layout_name).state_dict_tensors(wrapper._qdata, params)
        except Exception:
            named = {}
        for suffix, tensor in dict(named).items():
            for field, value in tensors.items():
                if suffix and value is tensor:
                    suffixes[field] = suffix
        return {field: (suffixes.get(field), value) for field, value in tensors.items()}

    def _reabsorb_quant(self, module: Any, name: str, prefix: str) -> tuple[int, int]:
        """Migrate one quantized weight's payload and scales into their slab regions.

        comfy's bake replaces the whole wrapper: `set_weight` requantizes and
        assigns a new Parameter in cudaMalloc, with a new payload and new
        scales. The wrapper is cloned, never re-derived: the payload bytes
        (`_qdata`) are copied into the layer's original slab region and a new
        wrapper is built from the same layout name and the same params. Never
        rebuild a baked wrapper from file metadata: baked bytes need not match
        the file's serialization transforms (int8-convrot diverged; see
        docs/VALIDATION.md).

        The scales go next, each into its own file component's region when the
        sizes agree (an NVFP4 block scale is an eighth of the payload). They
        are set on the live wrapper's params in place, because comfy_kitchen
        clones a wrapper's params whenever it is wrapped in a Parameter: a
        freshly loaded layer's scales sit in cudaMalloc too, beside their idle
        regions. The payload move stands on its own: a scale that cannot move
        stays where it is and is counted.

        A payload already in the slab is not copied onto itself and keeps its
        Parameter. Returns (migrated bytes, bytes still outside the slab).
        Each step changes the module only as its last action, so an exception
        leaves the model intact.
        """
        module_key, _, param_name = name.rpartition(".")
        live = self._live(module, name)
        wrapper = self._wrapper(live)
        layout_name = getattr(wrapper, "_layout_cls", None)
        params = getattr(wrapper, "_params", None)
        if (param_name != "weight" or not module_key or wrapper is None
                or not isinstance(layout_name, str) or params is None):
            return 0, self._outside([live])
        qdata = wrapper._qdata
        fkey = f"{prefix}{module_key}.{param_name}"
        moved = 0
        if not self.contains(qdata.data_ptr()):
            region = self._region_for(fkey, qdata.numel() * qdata.element_size())
            if region is not None:  # else padded storage or layout drift: leave it be
                slab_qdata = self._into_region(region, qdata)
                constructed = type(wrapper)(slab_qdata, layout_name, params)
                adopted = torch.nn.Parameter(constructed, requires_grad=False)
                op: Any = self._live(module, module_key)
                op.weight = adopted
                moved += region.nbytes
                # The live wrapper is the Parameter itself when the subclass keeps
                # its attributes through Parameter(), as comfy_kitchen's does.
                rebuilt = self._wrapper(adopted)
                wrapper = constructed if rebuilt is None else rebuilt
        try:
            moved += self._reabsorb_scales(wrapper, layout_name, fkey)
        except Exception as exc:
            safe_call(log.warning, "scale reabsorb failed for %s (%s); its scales stay in cudaMalloc",
                      name, failure_summary(exc))
        fields = self._scale_fields(wrapper, layout_name, wrapper._params)
        return moved, self._outside([wrapper._qdata] + [value for _suffix, value in fields.values()])

    def _reabsorb_scales(self, wrapper: Any, layout_name: str, fkey: str) -> int:
        """Move a live wrapper's scales into their regions, in place on its params; returns bytes moved."""
        params = wrapper._params
        replaced: dict[str, torch.Tensor] = {}
        moved = 0
        for field, (suffix, value) in self._scale_fields(wrapper, layout_name, params).items():
            if suffix is None or not value.numel() or self.contains(value.data_ptr()) or not value.is_contiguous():
                continue
            region = self._region_for(fkey + suffix, value.numel() * value.element_size())
            if region is not None:
                replaced[field] = self._into_region(region, value)
                moved += region.nbytes
        if replaced:
            wrapper._params = dataclasses.replace(params, **replaced)
        return moved

    def _quant_outside(self, module: Any, name: str) -> int:
        """A failed migration's figure: what the live weight holds outside the slab."""
        live = self._live(module, name)
        wrapper = self._wrapper(live)
        if wrapper is None:
            return self._outside([live])
        params = getattr(wrapper, "_params", None)
        fields = self._scale_fields(wrapper, getattr(wrapper, "_layout_cls", ""), params)
        return self._outside([wrapper._qdata] + [value for _suffix, value in fields.values()])

    def _settle_quant(self, module: Any, name: str, prefix: str) -> int:
        """Reabsorb one quantized weight and record what stays outside; returns moved bytes."""
        try:
            moved, stray = self._reabsorb_quant(module, name, prefix)
        except Exception as exc:
            try:
                log.warning(
                    "quant reabsorb failed for %s (%s); the tensor stays in cudaMalloc",
                    name,
                    failure_summary(exc),
                )
            except BaseException:
                pass
            moved = 0
            try:
                stray = self._quant_outside(module, name)
            except Exception:
                stray = 0
        if stray:
            self._strays[prefix + name] = stray
        else:
            self._strays.pop(prefix + name, None)
        return moved

    def baked_payload_fits(self, module: torch.nn.Module, name: str, prefix: str = "") -> bool:
        """Whether a bake of this quantized weight leaves a payload its own region can take back.

        A requantize stores the payload at the layout's storage shape
        (``get_storage_shape``, the padded shape on NVFP4 and MXFP8; the live
        shape for a layout that does not pad). A file that stored it unpadded
        has a smaller region, and the baked payload would stay in cudaMalloc.
        Scales are not asked: ComfyUI's loader gives the kernel the file's own
        scale bytes, so a scale the file carries has the requantize's layout.
        A plain tensor always fits; a quantized one with no wrapper never does.
        """
        live = self._live(module, name)
        if type(getattr(live, "data", live)) is torch.Tensor:
            return True
        wrapper = self._wrapper(live)
        module_key, _, param_name = name.rpartition(".")
        if wrapper is None or param_name != "weight" or not module_key:
            return False
        qdata = wrapper._qdata
        shape = tuple(qdata.shape)
        try:
            import comfy.quant_ops as quant_ops

            storage = getattr(quant_ops.get_layout_class(wrapper._layout_cls), "get_storage_shape", None)
            if callable(storage):
                shape = tuple(storage(tuple(wrapper._params.orig_shape)))
        except Exception:
            pass
        nbytes = qdata.element_size()
        for dim in shape:
            nbytes *= int(dim)
        return self._region_for(f"{prefix}{name}", nbytes) is not None

    def strays(self, name: str, prefix: str = "") -> bool:
        """Whether this weight already holds bytes outside the slab (counted in ``stray_bytes``)."""
        return bool(self._strays.get(prefix + name))

    def reabsorb_key(self, module: torch.nn.Module, name: str, prefix: str = "") -> int:
        """Return one parameter a lazy swap just restored or re-baked to its slab region.

        The store calls this as the swap passes each key, so a patched weight
        never sits twice in unified memory: at most the key in hand has a
        second copy. A quantized weight goes back as the fresh load's pass
        moves it; a plain weight the swap wrote out of place goes back to its
        own region when the size and dtype agree. Returns the bytes moved, and
        keeps ``stray_bytes`` true for this key.
        """
        live = self._live(module, name)
        t = getattr(live, "data", live)
        if type(t) is not torch.Tensor:
            return self._settle_quant(module, name, prefix)
        if self.contains(t.data_ptr()):
            return 0
        region = self._region_for_stray(prefix + name, t)
        if region is None:
            return 0  # no region of its own: it stays where the swap put it
        live.data = self._into_region(region, t)
        return region.nbytes
