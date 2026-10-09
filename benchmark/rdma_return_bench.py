#!/usr/bin/env python3
"""Measure tokenless RDMABuffer bandwidth and integrity.

Compare single-buffer and split (multi-QP) LatentReturn/read transfers. This
microbenchmark does not exercise production descriptor ownership, token ACKs,
sample leases, or actor recycling.

Native RDMA return is disabled by default (DESIGN.md section 5.6). This tool
forces it inside its own actor regardless of cluster.rdma_latent_return. Run
only as an explicit RDMA test; cross-rail pairs can fail until
rdma_ibverbs_target pins the NIC. The output identifies the ibverbs devices.

From the head Spark, with the monarch environment, Worker services available
(`dgxm up`), and a cluster.toml:

    ~/monarch-env/bin/python benchmark/rdma_return_bench.py \
        [--config /path/cluster.toml] [--sizes-mib 32,128,256] [--splits 1,2,4]

Flat GiB/s across splits indicates no multi-QP benefit. If split=2 approaches
the cable's nominal 200 Gb/s (about 23 GiB/s), per-QP host staging or CQ polling
is a likely bottleneck. Production still uses DGXM_RDMA_QP_SPLIT=1: splits
above one can select multiple equally ranked rails and expose the cross-rail
hazard (DESIGN.md section 5.6).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from monarch.actor import Actor, attach_to_workers, enable_transport, endpoint  # noqa: E402

from dgx_monarch.config import find_config_path, load_cluster_config  # noqa: E402
from dgx_monarch.error_utils import raise_with_distinct_cause, reconcile_error  # noqa: E402
from dgx_monarch.transfer import read_latent_result  # noqa: E402


class ReturnProbe(Actor):
    def __init__(self) -> None:
        self._lr = None  # tokenless benchmark keepalive until the direct read

    @endpoint
    def info(self) -> dict:
        import monarch.rdma as rdma

        try:
            devs = sorted(os.listdir("/sys/class/infiniband"))
        except OSError:
            devs = []
        return {
            "host": socket.gethostname(),
            "ibverbs": rdma.is_ibverbs_available(),
            "backend": str(rdma.get_rdma_backend()),
            "ib_devices": devs,
        }

    @endpoint
    def pack(self, nbytes: int, split: int, seed: int):
        """Register an nbytes latent for one-sided return at the given QP split.

        Returns (descriptor, sha256, kind, nparts). The no-handoff LatentReturn
        keepalive, not production ACK ownership, keeps the backing and MRs
        alive until the direct read."""
        import torch

        from dgx_monarch.transfer import LatentReturn

        os.environ["DGXM_RDMA_QP_SPLIT"] = str(split)
        gen = torch.Generator().manual_seed(seed)
        tensor = torch.randint(0, 256, (nbytes,), generator=gen, dtype=torch.uint8)
        self._lr = LatentReturn("rdma", min_bytes=1)  # min_bytes=1 => no size gate
        desc = self._lr.pack("bench", tensor)
        sha = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        return desc, sha, desc["kind"], len(desc.get("parts", []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="cluster.toml (default: search order)")
    ap.add_argument("--sizes-mib", default="32,128,256")
    ap.add_argument("--splits", default="1,2,4")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    try:
        sizes = [int(x) for x in args.sizes_mib.split(",") if x]
        splits = [int(x) for x in args.splits.split(",") if x]
    except ValueError as exc:
        print(f"invalid numeric benchmark list: {exc}", file=sys.stderr)
        return 2
    if not sizes or any(value <= 0 for value in sizes):
        print("--sizes-mib must contain positive integers", file=sys.stderr)
        return 2
    if not splits or any(value <= 0 for value in splits):
        print("--splits must contain positive integers", file=sys.stderr)
        return 2
    if args.repeats <= 0:
        print("--repeats must be greater than zero", file=sys.stderr)
        return 2

    cfg_path = find_config_path(args.config or None)
    if cfg_path is None:
        print("no cluster.toml found; pass --config <path>", file=sys.stderr)
        return 2
    cfg = load_cluster_config(cfg_path)
    if not cfg.client_bind:
        print("cluster.toml needs cluster.client_bind (fabric IP)", file=sys.stderr)
        return 2

    enable_transport(cfg.client_bind)
    addrs = [h.address for h in cfg.hosts]
    hosts = attach_to_workers(ca="trust_all_connections", workers=addrs, name="rdmabench")
    hosts.initialized.get(timeout=30)
    # The attached HostMesh belongs to the persistent worker services. This
    # benchmark owns only the ProcMesh it spawns below, and must never stop the
    # workers because the client-side measurement ends or fails.
    procs = hosts.spawn_procs(per_host={"gpus": 1})
    failed = False
    primary: BaseException | None = None
    try:
        actors = procs.spawn("rdmabench", ReturnProbe)
        for point, info in actors.info.call().get():
            print(
                f"  actor {dict(point)} host={info['host']} ibverbs={info['ibverbs']} "
                f"backend={info['backend']} ib_devices={info['ib_devices']}"
            )
        multi_host = "hosts" in actors.extent.labels
        # The leader is the far Spark, as in a real leader-to-driver return over the cable.
        leader = actors.slice(hosts=len(addrs) - 1) if multi_host else actors
        print(
            "\n  NOTE: torchmonarch hashes each registration over every rail tied for the best CPU\n"
            "  path, separately on each end, and a cross-rail pairing has no path (measured\n"
            "  2026-07-29; DESIGN.md section 5.6).\n"
        )

        print(f"  {'size':>7} {'split':>5} {'best_s':>9} {'GiB/s':>8}  integrity")
        for mib in sizes:
            nbytes = mib * 2**20
            for split in splits:
                desc, want_sha, kind, _ = leader.pack.call_one(nbytes, split, 1234).get()
                if kind != "rdma":
                    print(f"  {mib:>5}Mi {split:>5}  fell back to {kind!r}; check ibverbs/fabric")
                    failed = True
                    continue
                # warmup + integrity check
                got = read_latent_result(desc)
                got_sha = hashlib.sha256(got.numpy().tobytes()).hexdigest()
                integrity = "OK" if got_sha == want_sha else "MISMATCH"
                failed |= integrity != "OK"
                # timed: re-register each iteration (fresh MR) to mirror per-render cost
                best = float("inf")
                for _ in range(args.repeats):
                    desc, _, _, _ = leader.pack.call_one(nbytes, split, 1234).get()
                    t0 = time.perf_counter()
                    read_latent_result(desc)
                    best = min(best, time.perf_counter() - t0)
                gibps = (nbytes / 2**30) / best
                print(f"  {mib:>5}Mi {split:>5} {best:>9.4f} {gibps:>8.2f}  {integrity}")
    except BaseException as exc:
        primary = exc

    cleanup: BaseException | None = None
    try:
        procs.stop("dgx-monarch rdma return benchmark").get(timeout=60)
    except BaseException as exc:
        cleanup = exc
    if cleanup is not None:
        strongest, cause = reconcile_error(
            primary, cleanup, "the RDMA benchmark body and its ProcMesh stop both failed")
        raise_with_distinct_cause(strongest, cause)
    if primary is not None:
        raise primary
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
