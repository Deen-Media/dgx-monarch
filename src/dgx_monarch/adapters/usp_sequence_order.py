"""Canonicalize one rank-major two-segment Ulysses full sequence."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RankMajorJointOrder:
    """Two equally sharded segments gathered as rank-major local pairs.

    A joint attention that concatenates each rank's own text and image shards
    gives each rank ``[text_rank, image_rank]``. Ulysses' head all-to-all
    gathers those chunks as ``[text0,image0,text1,image1]``; stock attends
    ``[text0,text1,image0,image1]``, and flash rounding depends on key order.
    This descriptor maps only that self-attention layout and carries the
    inverse for output return. Lens subclasses it for its image-first order.
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
        permutation = self.permutation(q.shape[1], q.device)
        inverse = self.inverse(permutation)

        def remap(rows):
            if rows is None:
                return None
            values = list(rows)
            if any(type(index) is not int or index < 0 or index >= permutation.numel()
                   for index in values):
                self._refuse("a rank-major pad row is outside the gathered sequence")
            return sorted(int(inverse[index]) for index in values)

        return (q[:, permutation], k[:, permutation], v[:, permutation],
                remap(drop_rows), remap(kv_drop_rows), permutation)

    @staticmethod
    def restore(output: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
        if output.ndim != 4 or output.shape[1] != permutation.numel():
            RankMajorJointOrder._refuse("the attention output no longer matches the gathered axis")
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(permutation.numel(), device=permutation.device)
        return output[:, inverse]


def joint_sequence_order(context, text_rows: int, image_rows: int):
    """Descriptor only for the pure-Ulysses flux-family double-block route."""
    return RankMajorJointOrder(text_rows, image_rows) if context.pure_ulysses else None


def joint_double_options(options, attention, drop_rows, context, text_rows, image_rows):
    """Build the Flux-family double-block override with stock joint key order."""
    from .base import usp_options

    return usp_options(options, attention, drop_rows,
                       joint_sequence_order(context, text_rows, image_rows))
