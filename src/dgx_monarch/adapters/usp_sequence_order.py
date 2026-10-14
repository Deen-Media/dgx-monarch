"""Canonicalize one rank-major two-segment Ulysses full sequence.

The permutation, its inverse on the device and a host copy of the inverse
depend only on the descriptor, the gathered row count and the device, so they
are built once per layout and cached. The pad rows are then remapped on the
host from that copy, with no read from the device, and the output restore
reuses the cached inverse.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import NamedTuple

import torch

from .index_cache import BoundedIndexCache, device_key


class _JointIndex(NamedTuple):
    permutation: torch.Tensor   # on the key's device
    inverse: torch.Tensor       # on the key's device
    host_inverse: array[int]    # the same values, read by the pad-row remap


# One render uses one or two layouts (cond and uncond text lengths); the bound
# leaves room for several renders. Dataclass equality compares the class too,
# so a subclass descriptor never shares an entry with its parent.
_JOINT_INDEX: BoundedIndexCache[_JointIndex] = BoundedIndexCache(16)


@dataclass(frozen=True)
class RankMajorJointOrder:
    """Two equally sharded segments gathered as rank-major local pairs.

    A joint attention that concatenates each rank's own text and image shards
    gives each rank ``[text_rank, image_rank]``. Ulysses' head all-to-all
    gathers those chunks as ``[text0,image0,text1,image1]``; stock attends
    ``[text0,text1,image0,image1]``, and flash rounding depends on key order.
    This descriptor maps only that self-attention layout and carries the
    inverse for output return. Lens subclasses it for its image-first order;
    the cache builds every index from ``permutation``, so a subclass that
    overrides it stays in step.
    """

    text_rows: int
    image_rows: int

    @staticmethod
    def _refuse(detail: str):
        from ..refusal import RefusalClass, refusal
        from .base import UnsupportedModelError

        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"joint Ulysses order cannot preserve this attention contract: {detail}. "
            "Use topology 'single' (mode=local with gpus_per_host=1) for this render.",
        ))

    def _world(self, rows: int) -> int:
        local = self.text_rows + self.image_rows
        if self.text_rows < 1 or self.image_rows < 1 or local < 1 or rows % local:
            self._refuse("the gathered sequence does not tile its local text/image pair")
        world = rows // local
        if world < 2:
            self._refuse("the Ulysses degree is less than two")
        return world

    def permutation(self, rows: int, device: torch.device) -> torch.Tensor:
        """Indices mapping canonical order to the current rank-major order."""
        # Build on the device, not from Python lists of every row: those took 0.7 ms
        # per double-block call on the CPU, against 0.02 ms here (docs/VALIDATION.md,
        # Flux-family key order under pure Ulysses).
        starts = torch.arange(self._world(rows), device=device).unsqueeze(1) * (
            self.text_rows + self.image_rows)
        text = starts + torch.arange(self.text_rows, device=device)
        image = starts + self.text_rows + torch.arange(self.image_rows, device=device)
        return torch.cat((text.reshape(-1), image.reshape(-1)))

    def inverse(self, permutation: torch.Tensor) -> torch.Tensor:
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
        return inverse

    def joint_index(self, rows: int, device) -> _JointIndex:
        """This layout's cached indices, built on first use.

        The device tensors are built on the device; the host copy comes from
        the same arithmetic run on the CPU, never from reading the device.
        """
        device = device_key(device)
        key = (self, rows, device)
        found = _JOINT_INDEX.get(key)
        if found is not None:
            return found
        with torch.inference_mode(False):
            permutation = self.permutation(rows, device)
            inverse = self.inverse(permutation)
            host = (inverse if device.type == "cpu"
                    else self.inverse(self.permutation(rows, torch.device("cpu"))))
            # array("q") keeps 8 bytes a row against a list's 36.
            found = _JointIndex(permutation, inverse, array("q", host.tolist()))
        return _JOINT_INDEX.put(key, found)

    def canonicalize(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        drop_rows, kv_drop_rows, groups,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int] | None,
               list[int] | None, torch.Tensor]:
        """Reorder q, k and v to stock order, and remap both pad-row sets from rank-major rows."""
        if groups:
            self._refuse("query-bias group coordinates are present")
        if any(value.ndim != 4 for value in (q, k, v)) or q.shape != k.shape or q.shape != v.shape:
            self._refuse("q/k/v are not equal B,L,H,D self-attention tensors")
        gathered = q.shape[1]
        index = self.joint_index(gathered, q.device)
        permutation, host_inverse = index.permutation, index.host_inverse

        def remap(rows):
            if rows is None:
                return None
            values = list(rows)
            if any(type(row) is not int or row < 0 or row >= gathered for row in values):
                self._refuse("a rank-major pad row is outside the gathered sequence")
            return sorted(host_inverse[row] for row in values)

        query_rows = remap(drop_rows)
        # A self-attention passes one row set for both sides; keep it one
        # object so attend_without_pads can reuse the query index for the keys.
        key_rows = query_rows if kv_drop_rows is drop_rows else remap(kv_drop_rows)
        return (q[:, permutation], k[:, permutation], v[:, permutation],
                query_rows, key_rows, permutation)

    def restore(self, output: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
        """Return ``output`` to rank-major order.

        A permutation from ``canonicalize`` is the cached one, so its cached
        inverse serves; any other permutation gets its inverse built here.
        """
        if output.ndim != 4 or output.shape[1] != permutation.numel():
            self._refuse("the attention output no longer matches the gathered axis")
        found = _JOINT_INDEX.peek((self, permutation.numel(), device_key(permutation.device)))
        if found is not None and found.permutation is permutation:
            return output[:, found.inverse]
        return output[:, self.inverse(permutation)]


def joint_sequence_order(context, text_rows: int, image_rows: int):
    """Descriptor only for the pure-Ulysses flux-family double-block route."""
    return RankMajorJointOrder(text_rows, image_rows) if context.pure_ulysses else None


def joint_double_options(options, attention, drop_rows, context, text_rows, image_rows):
    """Build the Flux-family double-block override with stock joint key order."""
    from .base import usp_options

    return usp_options(options, attention, drop_rows,
                       joint_sequence_order(context, text_rows, image_rows))
