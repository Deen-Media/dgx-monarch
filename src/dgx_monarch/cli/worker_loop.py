"""Per-host Monarch worker loop: `python -m dgx_monarch.cli.worker_loop`.

One of these runs on every worker host (systemd unit or nohup via `dgxm up`),
bound to the fabric IP. Per torchmonarch's `run_worker_loop_forever`, the loop
kills all current work when the client disconnects and waits for a new client,
so the next driver attaches without a loop restart.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    from ..runtime_provenance import enforce_source_only_imports

    enforce_source_only_imports()
    parser = argparse.ArgumentParser(description="dgx-monarch per-host worker service")
    parser.add_argument("--address", required=True, help="bind address, such as tcp://192.0.2.12:26600")
    parser.add_argument("--runtime-policy", default="", help="validated Monarch process policy")
    parser.add_argument("--native-rdma", action="store_true", help="require selected native RDMA policy")
    args = parser.parse_args()

    from ..config import format_tcp_address, tcp_endpoint, unmapped_bind_ip

    try:
        bind_host, bind_port = tcp_endpoint(args.address)
        bind_ip, mapped = unmapped_bind_ip(bind_host)
        if bind_port == 0:
            raise ValueError("worker-loop port must be nonzero")
        if mapped:
            raise ValueError(
                "worker-loop address is the IPv4-mapped spelling of "
                f"{bind_ip.compressed}; write the IPv4 address itself")
        if bind_ip.is_unspecified or bind_ip.is_multicast:
            raise ValueError("worker-loop address must be a unicast interface address")
        args.address = format_tcp_address(bind_ip.compressed, bind_port)
    except ValueError as exc:
        parser.error(str(exc))

    # Keep this before the first Monarch import. Native launcher diagnostics
    # may render LaunchOptions, including every inherited env value, so the
    # worker loop keeps only the names worker_process_env allows.
    from .worker_process_env import INDUCTOR_CACHE_ENV, prepare_worker_process_environment

    prepare_worker_process_environment(Path(__file__).resolve().parents[2])

    import os
    import sys

    if not os.environ.get(INDUCTOR_CACHE_ENV):
        print("[dgx-monarch] torch inductor cache: no private directory under "
              "~/.cache/dgx-monarch/inductor, so torch keeps it under TMPDIR, which a "
              "reboot may clear (docs/TROUBLESHOOTING.md #14)", file=sys.stderr, flush=True)

    from ..monarch_diagnostics import configure_policy_logged
    from ..monarch_launch import NATIVE_ENV
    from ..monarch_policy import MonarchPolicy, apply_policy, policy_from_token

    policy = policy_from_token(args.runtime_policy) if args.runtime_policy else MonarchPolicy()
    apply_policy(policy, role="worker", native_enabled=args.native_rdma)
    os.environ[NATIVE_ENV] = "1" if args.native_rdma else "0"
    configure_policy_logged(policy.diagnostics, "worker")  # an ERROR; the loop still serves

    # Serve a new Monarch ProcId on every start (worker_proc_id.py), and
    # publish it in the loop record before the listener exists, so the probe
    # the driver runs before each attach reads the id this process serves. A
    # daemon thread then keeps the record and the actor identities current
    # for later loops and lifecycle sweeps: run_worker_loop_forever blocks
    # in native code.
    from .. import worker_proc_id
    from .actor_ledger import publish_loop, start_tracker

    proc_id = worker_proc_id.current()
    if not publish_loop(args.address, proc_id=proc_id):
        print("[dgx-monarch] worker service could not publish its id in the actor "
              "ledger; driver attaches refuse until it can "
              f"({worker_proc_id.ENTRY})", file=sys.stderr, flush=True)
    start_tracker(args.address, proc_id=proc_id)

    from monarch.actor import enable_transport, run_worker_loop_forever

    from ..monarch_capabilities import assert_supported_runtime
    from ..monarch_policy import activate_native_policy

    assert_supported_runtime()
    activate_native_policy()
    served = worker_proc_id.proc_addr(args.address, proc_id)

    # Secondary channels must advertise the same fabric IP as the loop bind, and
    # their ports are dynamic, so SECURITY.md asks for a rule on the whole
    # interface. The bare "tcp" shortcut resolves to TcpWithHostname, the
    # hostname-derived address DESIGN section 5.1 bans (it can resolve to loopback
    # through the 127.0.1.1 /etc/hosts line). Port 0 keeps the loop's own port free.
    host_part = args.address.rsplit(":", 1)[0]  # tcp://<fabric-ip>
    enable_transport(f"{host_part}:0")
    # Monarch spells the id in base58; doctor and the attach trace print hex.
    print(f"[dgx-monarch] worker service listening on {served} (proc_id={proc_id})", flush=True)
    run_worker_loop_forever(address=served, ca="trust_all_connections")


if __name__ == "__main__":
    main()
