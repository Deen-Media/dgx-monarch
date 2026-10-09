"""The rendezvous store worker setup hands to torch.distributed."""
from __future__ import annotations

from datetime import timedelta


def generation_store(env: dict, rank: int, world: int, generation: int):
    """A TCP rendezvous store whose keys all live under this setup generation.

    torch's ``tcp://`` rendezvous runs rank 0's store with ``multi_tenant=True``,
    which shares one server per port inside a process, and that server outlives
    ``destroy_process_group``; process-group names restart after a destroy. The
    driver's master port repeats every 16 setup generations per driver process,
    so a later generation on a reused port could read an earlier generation's
    Gloo addresses for a group of the same name, connect to a closed port and be
    refused at once while rank 0 waited out its timeout. Prefixing every key with
    the generation keeps those stale keys out of sight whatever the port.
    """
    import torch.distributed as dist

    store = dist.TCPStore(
        str(env["master_addr"]), int(env["master_port"]), world, rank == 0,
        timeout=timedelta(seconds=int(env.get("nccl_timeout_s", 600))),
        multi_tenant=True,
    )
    return dist.PrefixStore(f"dgxm/setup-generation/{int(generation)}", store)
