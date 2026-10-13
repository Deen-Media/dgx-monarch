#!/usr/bin/env python3
"""Qualify production CPU-staged latent return on two persistent Workers.

No GPU calls, models, services, SSH, Job ownership, or hardware discovery.
The operator must separately prove the pair idle and preserve Worker services.
Run --preflight-only on each participant with its source/interpreter selected;
this prints passive source facts and never imports Torch/Monarch. Use the live
tools/process_inspection.py --check for process-readability proof; neither
command establishes idleness.
Native mode requires cluster.rdma_latent_return=true and explicit per-host
NIC selection. Message mode requires false. The driver's host is the [[hosts]]
entry whose address shares client_bind's IP, the rule the production driver
uses; --driver-host is an optional check that must name that same host. Each
QP/application-chunk variant uses a fresh process and config; this tool never
mutates a latched policy.
The old tokenless --sizes-mib/--splits bandwidth mode is intentionally retired:
it did not exercise production ownership and could bypass policy admission.
Run hardware only after reviewing this file. Never put the hardware invocation
under timeout(1): HOLD deliberately retains native owners until operator recovery.
This proves only staged CPU returns, not GPU/render/fidelity qualification.

Run with --expect-sha FULL_SHA --expect-manifest MANIFEST --config cluster.toml
[--driver-host CONFIG_HOSTNAME] --return-mode message|native --output NEW.json.
Output is a new mode-0600 file in an existing directory. Results never change
cluster defaults or authorize a model or full sampling gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

THRESHOLD = 8 * 1024 * 1024
# Process-lifetime roots also survive exceptions and failed evidence writes.
OWNED: list[object] = []


class QualificationRefused(RuntimeError):
    """A fixed, disclosure-safe qualifier invariant failed."""


def require(condition, reason):
    if not condition:
        raise QualificationRefused(reason)


def source_facts():
    import dgx_monarch
    from dgx_monarch.runtime_provenance import (
        _tracked_source_paths,
        source_manifest_sha256,
    )

    package = Path(dgx_monarch.__file__).resolve().parent
    root = package.parents[1]
    git, tracked = {"mode": "managed_copy"}, None
    if (root / ".git").exists():
        def read_git(*arguments, binary=False):
            result = subprocess.run(
                ["/usr/bin/git", "--no-optional-locks", "--no-replace-objects",
                 "-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false",
                 "-C", str(root), *arguments], check=True, capture_output=True,
                text=not binary, timeout=15,
                env={"PATH": os.defpath, "LANG": "C", "LC_ALL": "C",
                     "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"})
            return result.stdout
        commit = read_git("rev-parse", "HEAD").strip()
        require(re.fullmatch(r"[0-9a-f]{40}", commit), "invalid Git commit")
        require(Path(read_git("rev-parse", "--show-toplevel").strip()) == root,
                "source Git root mismatch")
        require(not read_git("status", "--porcelain=v1", "--untracked-files=all",
                             "--ignore-submodules=none").strip(), "dirty source checkout")
        require(not any(line[:1].islower() for line in read_git("ls-files", "-v").splitlines())
                and not any(line.startswith("S ") for line in read_git("ls-files", "-t").splitlines()),
                "hidden source checkout flags")
        tracked = _tracked_source_paths(read_git("ls-tree", "-r", "-z", "--name-only",
            commit, "--", "src/dgx_monarch", binary=True), tracked_prefix="src/dgx_monarch",
            scope="dgx_monarch", custom_nodes_disabled=False)
        git = {"mode": "git", "commit": commit}
    manifest = source_manifest_sha256(
        package, scope="dgx_monarch", expected_paths=tracked)
    return {"source_mode": git["mode"], "source_sha": git.get("commit"),
            "source_manifest_sha256": manifest,
            "torchmonarch": importlib.metadata.version("torchmonarch"),
            "torch": importlib.metadata.version("torch"),
            "dgx_monarch": dgx_monarch.__version__}


def check_source(facts, sha, manifest):
    require(facts["source_manifest_sha256"] == manifest, "source manifest mismatch")
    require(facts["source_sha"] in (None, sha), "source commit mismatch")
    require(facts["torchmonarch"] == "0.7.0", "torchmonarch must equal 0.7.0")
    # Managed copies are bound to the clean driver's exact SHA by payload digest;
    # do not report a nonexistent remote Git checkout as verified.
    return {**facts, "bound_source_sha": sha,
            "commit_binding": "git" if facts["source_sha"] else "manifest_to_driver"}


class Evidence:
    def __init__(self, output, data):
        self.path = Path(output).absolute()
        require(self.path.parent.is_dir(), "evidence parent must already exist")
        require(not self.path.exists() and not self.path.is_symlink(),
                "evidence file must be new")
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                          | os.O_NOFOLLOW, 0o600)
        self.data = data
        self.save()

    def save(self):
        payload = json.dumps(self.data, sort_keys=True, indent=2).encode() + b"\n"
        os.lseek(self.fd, 0, os.SEEK_SET)
        view = memoryview(payload)
        while view:
            view = view[os.write(self.fd, view):]
        os.ftruncate(self.fd, len(payload))
        os.fsync(self.fd)


def park(evidence, reason, report_interval=30):
    evidence.data.update(result="HOLD", clean_exit=False,
                         retained_owner=True, retained_driver_pid=os.getpid(),
                         hold_reason=reason)
    try:
        evidence.save()
    except BaseException:
        pass
    print("HOLD: ownership retained; do not kill or restart participants. "
          "Operator recovery required.", flush=True)
    # Retain uncertain owners: no destructive retries, ACK retries, or service actions.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    forever = threading.Event()
    while True:
        forever.wait(report_interval)
        print("HOLD persists: inspect the private evidence and arrange reviewed "
              "owner recovery. This process has not retried reads, ACKs or stop.", flush=True)


def actor_class():
    # Monarch cloudpickles this local class; its production mixin stays source-bound.
    from monarch.actor import Actor, current_rank, endpoint

    from dgx_monarch.actor.rdma_handoff import RDMAHandoffMixin

    class ReturnProbe(RDMAHandoffMixin, Actor):
        def __init__(self, sha, manifest, identities, return_mode):
            from dgx_monarch.monarch_policy import verify_native_policy
            from dgx_monarch.transfer import LatentReturn
            self.rank = int(current_rank()["hosts"])
            self.identity = check_source(source_facts(), sha, manifest)
            self.expected_sha, self.expected_manifest = sha, manifest
            self.policy = verify_native_policy()
            require(self.policy["transport_sha256"] == identities[self.rank],
                    "actor transport identity mismatch")
            if return_mode == "native":
                require(self.policy["rdma"]["target"].startswith("nic:"),
                        "actor lacks selected NIC")
            self._setup_generation = 1
            self._init_rdma_handoffs()
            self._latent_return = LatentReturn(
                "rdma" if return_mode == "native" else "message", min_bytes=THRESHOLD)

        @endpoint
        async def facts(self, fresh_source=False):
            from dgx_monarch import transfer
            from dgx_monarch.rdma_read_job import pending_job_count
            identity = check_source(source_facts(), self.expected_sha, self.expected_manifest
                                    ) if fresh_source else self.identity
            return {"rank": self.rank, "identity": identity,
                    "policy": self.policy,
                    "registry_live": self._latent_handoffs.live_count,
                    "registry_tombstones": self._latent_handoffs.tombstone_count,
                    "native_poison": transfer._rdma_poisoned(),
                    "pending_jobs": pending_job_count()}

        @endpoint
        async def pack(self, size, seed, sequence, depth):
            import asyncio

            # Preserve Monarch's actor context while keeping native registration
            # off the asyncio loop, as GPUWorker's production return path does.
            return await asyncio.to_thread(self._pack_impl, size, seed, sequence, depth)

        def _pack_impl(self, size, seed, sequence, depth):
            import torch
            tensor = torch.randint(0, 256, (size,), dtype=torch.uint8,
                                   device="cpu", generator=torch.Generator(
                                       device="cpu").manual_seed(seed))
            return self._pack_latent_result({}, {"samples": tensor},
                {"render_seq": sequence, "pipeline_depth": depth})

    return ReturnProbe


def admitted_driver_host(cfg, requested=None):
    """The driver's host by production's client_bind rule; --driver-host must agree."""
    from dgx_monarch.monarch_policy_io import driver_host

    require(bool(cfg.client_bind), "explicit driver bind required")
    resolved = driver_host(cfg)
    require(resolved in {h.name for h in cfg.hosts}, "client_bind matches no configured host")
    require(requested in (None, resolved), "--driver-host disagrees with the host client_bind selects")
    return resolved


def admit_config(cfg, return_mode, driver_host=None):
    """Configuration admission is not proof of a working native path."""
    from dataclasses import replace

    from dgx_monarch.monarch_policy import transport_identity

    require(cfg.rdma_latent_return is (return_mode == "native"),
            "configured native enablement must match --return-mode")
    require(cfg.rdma_min_bytes == THRESHOLD, "qualification requires the 8 MiB threshold")
    require(len(cfg.hosts) == 2, "qualifier scope is exactly two persistent Workers")
    admitted_driver_host(cfg, driver_host)
    require(not cfg.monarch.rdma.allow_tcp_fallback, "TCP fallback must be disabled")
    require(not cfg.monarch.diagnostics.otlp_endpoint,
            "private qualifier refuses network diagnostic export")
    identities = []
    for host in cfg.hosts:
        resolved = cfg.monarch.rdma.resolved(host.name)
        if return_mode == "native":
            require(resolved.target.startswith("nic:"), "every host needs selected NIC")
        identities.append(transport_identity(replace(cfg.monarch, rdma=resolved)))
    return identities


def stop_owned(procs, handoff, handle, pending_jobs, poisoned):
    """Stop only this exact child after the caller's ownership checks passed."""
    from dgx_monarch.mesh_factory import owned_spawned_proc

    require(owned_spawned_proc(handoff, procs) is procs, "ProcMesh ownership changed")
    require(not handle.sample_leases and not handle.abandoned_sample_leases
            and pending_jobs == 0 and not poisoned, "receiver unsettled at owned stop")
    procs.stop("CPU return qualification complete").get(timeout=90)


def run(args, evidence, cfg, driver_facts):
    from dgx_monarch.monarch_diagnostics import configure_policy
    from dgx_monarch.monarch_policy import (
        activate_native_policy,
        apply_policy,
    )

    driver_host = admitted_driver_host(cfg, args.driver_host)
    identities = admit_config(cfg, args.return_mode, driver_host)
    apply_policy(cfg.monarch, driver_host, role="driver",
                 native_enabled=args.return_mode == "native")
    configure_policy(cfg.monarch.diagnostics)
    from monarch.actor import attach_to_workers, enable_transport

    from dgx_monarch.monarch_capabilities import assert_supported_runtime
    assert_supported_runtime()
    driver_policy = activate_native_policy()
    evidence.data.update(driver=driver_facts, driver_policy=driver_policy,
                         expected_participant_policy_sha256=identities)
    evidence.save()

    import torch
    from monarch._src.actor.proc_mesh import (
        register_proc_mesh_spawn_callback,
        unregister_proc_mesh_spawn_callback,
    )

    from dgx_monarch import transfer
    from dgx_monarch.mesh_factory import (
        FleetHandoff,
        _isolated_spawn_host,
        owned_spawned_proc,
    )
    from dgx_monarch.mesh_lease import SetupBoundFuture
    from dgx_monarch.mesh_runtime import bootstrap_proc
    from dgx_monarch.monarch_launch import actor_bootstrap_command
    from dgx_monarch.rdma_read_job import pending_job_count

    # Transport startup can have effects even when raising: park on uncertainty.
    evidence.data["ownership_started"] = True
    evidence.save()
    enable_transport(cfg.client_bind)
    Probe = actor_class()
    sequence = 0
    for cycle in range(2):
        evidence.data["stage"] = "attach"
        row = {"cycle": cycle, "cases": [], "owned_stop_confirmed": False}
        evidence.data["cycles"].append(row)
        hosts = attach_to_workers(ca="trust_all_connections",
            workers=[host.address for host in cfg.hosts], name=f"native-return-{cycle}")
        OWNED.append(hosts)
        hosts.initialized.get(timeout=90)
        evidence.data["stage"] = "spawn_owned_procs"
        handoff = FleetHandoff()
        OWNED.append(handoff)
        handoff.spawn_host, handoff.spawn_host_isolated = _isolated_spawn_host(hosts)
        capture = handoff.proc_meshes.append
        register_proc_mesh_spawn_callback(capture)
        try:
            handoff.proc_spawn_called = True
            procs = handoff.spawn_host.spawn_procs(per_host={"procs": 1},
                name=f"native-return-{cycle}", bootstrap=bootstrap_proc,
                bootstrap_command=actor_bootstrap_command(cfg, True))
            OWNED.append(procs)
        finally:
            unregister_proc_mesh_spawn_callback(capture)
        require(owned_spawned_proc(handoff, procs) is procs, "owned ProcMesh unresolved")
        procs.initialized.get(timeout=90)
        actors = procs.spawn("return-probe", Probe, args.expect_sha,
                             args.expect_manifest, identities, args.return_mode)
        OWNED.append(actors)
        actors.initialized.get(timeout=90)

        def all_rows(method, *values, timeout_s=90, _actors=actors):
            result = getattr(_actors, method).call(*values).get(timeout=timeout_s)
            return [value for _point, value in result.items()]

        handle = SimpleNamespace(lock=threading.RLock(), sample_leases={},
            abandoned_sample_leases={}, world=2, call_all=all_rows)
        OWNED.append(handle)

        def facts(expected_live=None, fresh_source=False, _all_rows=all_rows):
            rows = sorted(_all_rows("facts", fresh_source), key=lambda item: item["rank"])
            require([r["rank"] for r in rows] == [0, 1], "partial participant set")
            require(all(not r["native_poison"] and r["pending_jobs"] == 0
                        for r in rows), "participant ownership not settled")
            if expected_live is not None:
                require(sum(r["registry_live"] for r in rows) == expected_live,
                        "registry count mismatch")
            return rows

        row["participants"] = facts(0)
        evidence.save()
        # Exercise each selected host as owner, including the cross-host return.
        for rank in (0, 1):
            owner = actors.slice(hosts=rank)

            def dispatch(size, seed, depth, _handle=handle, _owner=owner):
                nonlocal sequence
                evidence.data["stage"] = "dispatch_pack"
                started = time.monotonic()
                sequence += 1
                guard = SetupBoundFuture.prepared(_handle, 1)
                OWNED.append(guard)
                with _handle.lock:
                    guard._register_locked()
                    guard._mark_enqueue_started_locked()
                future = _owner.pack.call_one(size, seed, sequence, depth)
                OWNED.append(future)
                with _handle.lock:
                    guard._bind_future_locked(future)
                result = guard.get(timeout=90)
                OWNED.append(result)
                return guard, result["latent"], size, seed, time.monotonic() - started

            def consume(item, expected_kind, label, _rank=rank, _row=row, _all_rows=all_rows):
                evidence.data["stage"] = "read_drop_ack"
                evidence.data["active_case"] = {"label": label, "owner_rank": _rank}
                evidence.save()
                guard, descriptor, size, seed, pack_s = item
                require(descriptor.get("kind") == expected_kind,
                        "descriptor kind mismatch: native fallback is not PASS")
                started = time.monotonic()
                value = transfer.read_latent_result(descriptor, read_guard=guard)
                read_s = time.monotonic() - started
                expected = torch.randint(0, 256, (size,), dtype=torch.uint8,
                    device="cpu", generator=torch.Generator(device="cpu").manual_seed(seed))
                require(value.device.type == "cpu" and value.dtype == torch.uint8
                        and tuple(value.shape) == (size,) and torch.equal(value, expected),
                        "CPU tensor exact integrity mismatch")
                require(guard.readers == 0 and pending_job_count() == 0
                        and not transfer._rdma_poisoned(), "receiver not settled")
                guard.release()
                if expected_kind == "rdma":
                    duplicate = _all_rows("ack_latent_handoff", 1, descriptor["owner_token"])
                    require(sorted(r["status"] for r in duplicate) ==
                            ["already_released", "unknown"], "duplicate ACK not idempotent")
                _row["cases"].append({"label": label, "owner_rank": _rank,
                    "bytes": size, "kind": expected_kind, "exact_equal": True,
                    "pack_rpc_s": pack_s, "read_drop_ack_s": read_s,
                    "tensor_sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
                    "receiver_jobs": 0, "reader_tokens": 0})
                evidence.save()

            for delta in (-16384, 0, 16384):
                consume(dispatch(THRESHOLD + delta, 110 + delta, 2),
                        "message" if delta < 0 or args.return_mode == "message" else "rdma",
                        f"threshold{delta:+}")
                facts(0)
            if args.return_mode == "message":
                require(not handle.sample_leases and not handle.abandoned_sample_leases,
                        "message sample leases did not settle")
                continue
            first = dispatch(THRESHOLD, 201, 2)
            second = dispatch(THRESHOLD + 16384, 202, 2)
            require(first[1]["kind"] == second[1]["kind"] == "rdma",
                    "pipeline native registration unavailable")
            facts(2)
            wrong = all_rows("ack_latent_handoff", 2, first[1]["owner_token"])
            require(all(r["status"] == "unknown" for r in wrong),
                    "wrong generation unexpectedly ACKed")
            facts(2)
            overflow = dispatch(THRESHOLD, 203, 2)
            consume(overflow, "message", "pipeline_capacity_message_fallback")
            facts(2)
            consume(first, "rdma", "pipeline_first")
            facts(1)
            replacement = dispatch(THRESHOLD, 204, 2)
            facts(2)
            consume(second, "rdma", "pipeline_second")
            consume(replacement, "rdma", "pipeline_capacity_recovered")
            facts(0)
            require(not handle.sample_leases and not handle.abandoned_sample_leases,
                    "sample leases did not settle")
        row["settled_participants"] = facts(0, fresh_source=True)
        require(pending_job_count() == 0 and not transfer._rdma_poisoned(),
                "driver has native ownership at stop boundary")
        # Only this exact invocation-owned ProcMesh; never HostMesh.stop/Job.kill.
        evidence.data["stage"] = "owned_stop"
        evidence.save()
        stop_owned(procs, handoff, handle, pending_job_count(), transfer._rdma_poisoned())
        row["owned_stop_confirmed"] = True
        evidence.save()
    evidence.data["ownership_started"] = False
    evidence.data["stage"] = "final_source_check"
    check_source(source_facts(), args.expect_sha, args.expect_manifest)
    evidence.data.update(result="PASS", clean_exit=True, retained_owner=False,
                         reattach_cycles_completed=2,
                         native_acceptance="PASS" if args.return_mode == "native" else "NOT_RUN")
    evidence.save()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.environ.get(
        "DGX_MONARCH_DIR", str(Path(__file__).resolve().parents[1])),
                        help="exact source checkout; source/src precedes installed package")
    parser.add_argument("--expect-sha", required=True)
    parser.add_argument("--expect-manifest", required=True)
    parser.add_argument("--preflight-only", action="store_true",
                        help="passive source facts only; no Torch/Monarch import or mutation")
    parser.add_argument("--config", help="explicit private selected-NIC cluster TOML")
    parser.add_argument("--driver-host",
                        help="optional check: the configured host name client_bind must select")
    parser.add_argument("--output", help="new private JSON evidence path in existing directory")
    parser.add_argument("--return-mode", choices=("message", "native"), default="message",
                        help="default-off control or explicit native ownership qualification")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    require(re.fullmatch(r"[0-9a-f]{40}", args.expect_sha), "expected SHA must be full lowercase hex")
    require(re.fullmatch(r"[0-9a-f]{64}", args.expect_manifest), "invalid expected manifest")
    require(args.source, "explicit source required")
    root = Path(args.source).expanduser().resolve(strict=True)
    sys.dont_write_bytecode, sys.pycache_prefix = True, "/proc/self/fd/2147483647"
    sys.path.insert(0, str(root / "src"))
    from dgx_monarch.runtime_provenance import enforce_source_only_imports
    enforce_source_only_imports()
    driver = check_source(source_facts(), args.expect_sha, args.expect_manifest)
    import dgx_monarch
    require(Path(dgx_monarch.__file__).resolve() == root / "src/dgx_monarch/__init__.py",
            "loaded package is not the requested source")
    if args.preflight_only:
        print(json.dumps({"source": driver, "idle_authority": False,
            "process_preflight": "tools/process_inspection.py --check plus operator no-work proof"},
            sort_keys=True))
        return 0
    require(args.config and args.output, "config and output required")
    require(driver["source_mode"] == "git", "driver needs clean exact Git checkout")
    from dgx_monarch.config import load_cluster_config
    cfg = load_cluster_config(args.config)
    evidence = Evidence(args.output, {"schema": 1, "result": "RUNNING",
        "scope": "cpu_staged_production_return_only", "return_mode": args.return_mode,
        "native_acceptance": "NOT_RUN", "clean_exit": False,
        "render_fidelity": "NOT_RUN", "gpu": "NOT_QUERIED_OR_INITIALIZED",
        "full_sampling_gate": "NOT_RUN", "persistent_worker_policy": "OPERATOR_PREFLIGHT_REQUIRED",
        "threshold_bytes": THRESHOLD, "pipeline_depth": 2, "cycles": [],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_unix_s": time.time(), "ownership_started": False})
    OWNED.append(evidence)
    try:
        run(args, evidence, cfg, driver)
    except BaseException as exc:
        # No exception strings/tracebacks: they can contain peer URLs or tokens.
        evidence.data["error_type"] = type(exc).__name__
        if isinstance(exc, QualificationRefused):
            evidence.data["refusal"] = str(exc)
        if evidence.data["ownership_started"]:
            park(evidence, "operation or settlement incomplete; no destructive cleanup attempted")
        evidence.data.update(result="REFUSED", clean_exit=True, retained_owner=False)
        evidence.save()
        print("REFUSED before transport startup; review private evidence.", flush=True)
        return 2
    print(f"PASS: exact CPU staged {args.return_mode} returns and owned cleanup; no render claim.")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception as error:
        print(f"REFUSED before qualification: {type(error).__name__}", file=sys.stderr)
        code = 2
    raise SystemExit(code)
