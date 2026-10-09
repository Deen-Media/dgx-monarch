"""Worker-side durable ownership for RDMA latent result descriptors."""
from __future__ import annotations

import socket
from typing import Any

from monarch.actor import endpoint

from ..rdma_ownership import HandoffRegistry
from ..transfer import LatentReturn
from ..transfer_utils import raise_with_distinct_cause, reconcile_error


class RDMAHandoffMixin:
    """Keep descriptor resources actor-owned until a generation-bound ACK."""

    _latent_handoffs: HandoffRegistry
    _latent_return: LatentReturn
    _setup_generation: int | None
    rank: int | None

    def _init_rdma_handoffs(self) -> None:
        self._latent_handoffs = HandoffRegistry()

    def _handoff_registry(self) -> HandoffRegistry:
        registry = getattr(self, "_latent_handoffs", None)
        if registry is None:
            registry = self._latent_handoffs = HandoffRegistry()
        return registry

    @staticmethod
    def _message_descriptor(key: str, tensor: Any, seq: int, depth: int) -> dict:
        return LatentReturn("message", min_bytes=1).pack(
            key, tensor, seq=seq, depth=depth)

    def _pack_latent_result(self, result: dict, out: dict, request: dict) -> dict:
        leader_samples = out.pop("samples")
        seq = int(request.get("render_seq", 0))
        depth = max(1, int(request.get("pipeline_depth", 1) or 1))
        latent_return = self._latent_return
        if latent_return.mode != "rdma":
            result["latent"] = latent_return.pack(
                "samples", leader_samples, seq=seq, depth=depth)
            result["latent_extra"] = out
            return result

        generation = self._setup_generation
        if generation is None:
            raise RuntimeError("RDMA latent result has no active setup generation")
        registry = self._handoff_registry()
        handoff: dict[str, Any] = {}
        try:
            registry.publish(handoff, generation, depth)
            if not registry.owns(handoff):
                result["latent"] = self._message_descriptor(
                    "samples", leader_samples, seq, depth)
            else:
                descriptor = latent_return.pack(
                    "samples", leader_samples, seq=seq, depth=depth,
                    handoff=handoff)
                if descriptor.get("kind") == "rdma":
                    descriptor["owner_token"] = handoff["token"]
                    descriptor["setup_generation"] = generation
                    registry.mark_ready(handoff)
                elif not registry.reconcile(handoff):
                    raise RuntimeError(
                        "RDMA registration ownership is ambiguous; Attached mesh reset required")
                result["latent"] = descriptor
            result["latent_extra"] = out
            return result
        except BaseException as primary:
            # Empty attempts are safe to forget.  Any ambiguity after native
            # construction remains registry-owned until actor recycle.
            try:
                registry.reconcile(handoff)
            except BaseException as reconciliation_error:
                winner, cause = reconcile_error(
                    primary, reconciliation_error,
                    "RDMA handoff reconciliation interrupted")
                if winner is not primary:
                    raise_with_distinct_cause(winner, cause)
            raise primary

    def _ack_latent_handoff_impl(self, setup_generation: int, token: str) -> dict:
        registry = getattr(self, "_latent_handoffs", None)
        status = (
            registry.acknowledge(setup_generation, token)
            if registry is not None else "unknown"
        )
        return {
            "host": socket.gethostname(),
            "rank": self.rank,
            "setup_generation": setup_generation,
            "token": token,
            "status": status,
        }

    @endpoint
    async def ack_latent_handoff(self, setup_generation: int, token: str) -> dict:
        """Retire ownership after confirmed local drops and read-authority exit.

        The driver's native read may have completed or may have been proven not
        to have started; the ACK certifies the local settlement, not a read.
        """
        return self._ack_latent_handoff_impl(setup_generation, token)
