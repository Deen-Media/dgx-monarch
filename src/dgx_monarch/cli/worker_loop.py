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
    from .worker_process_env import prepare_worker_process_environment

    prepare_worker_process_environment(Path(__file__).resolve().parents[2])

    # Persist actor identities for later loops and lifecycle sweeps. Use a
    # daemon thread because run_worker_loop_forever blocks in native code.
    from .actor_ledger import start_tracker

    start_tracker(args.address)

    from monarch.actor import enable_transport, run_worker_loop_forever

    # Secondary channels must advertise the same fabric IP as the loop bind, and
    # their ports are dynamic, so SECURITY.md asks for a rule on the whole
    # interface. The bare "tcp" shortcut resolves to TcpWithHostname, the
    # hostname-derived address DESIGN section 5.1 bans (it can resolve to loopback
    # through the 127.0.1.1 /etc/hosts line). Port 0 keeps the loop's own port free.
    host_part = args.address.rsplit(":", 1)[0]  # tcp://<fabric-ip>
    enable_transport(f"{host_part}:0")
    print(f"[dgx-monarch] worker service listening on {args.address}", flush=True)
    run_worker_loop_forever(address=args.address, ca="trust_all_connections")


if __name__ == "__main__":
    main()
