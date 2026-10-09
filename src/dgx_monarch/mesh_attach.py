"""Cluster attach: transport enable, out-of-band loop healing and the in-process retry."""
from __future__ import annotations

import os
import time
from typing import Any

from . import attach_trace, mesh_teardown
from .config import ClusterConfig
from .error_utils import failure_summary
from .mesh_runtime import ATTACH_INIT_WAIT_S

# How long one loop's listener-generation reading may take before the attach
# gives up on it. The reading is evidence, never a gate, so a host that will
# not answer costs the attach this and nothing more.
GENERATION_PROBE_S = 3.0


def attach_cluster(runtime: dict[str, Any], config: ClusterConfig, handoff: Any):
    """Enable one process transport, heal loops, and attach the configured hosts."""
    factory = runtime["mesh_factory"]
    error_type = runtime["MeshAttachError"]
    # Read here, not where it is used. Whether a liveness proof authorized this
    # replacement is a fact about how the bring-up started. The note's window is
    # the creation timeout, and a heal that restarts and settles spends a good
    # share of it, so a read after the first failure can miss the note and lose
    # the fail-fast in the slowest incident.
    released = mesh_teardown.released_predecessor_recently(
        addresses=[host.address for host in config.hosts])
    try:
        from monarch.actor import enable_transport
    except ImportError as exc:
        factory.safe_transport_failure(handoff)
        evidence = factory.safe_failure_message(exc)
        raise error_type(
            f"torchmonarch is not importable ({evidence}). Install dgx-monarch and its pinned dependencies "
            "into ComfyUI's Python with the docs/INSTALL.md install block; a bare `pip install -r "
            "requirements.txt` can pick another Python and cannot find the patched xfuser wheel the guide builds."
        ) from exc
    if config.transport_security != "trusted_fabric":
        factory.safe_transport_failure(handoff)
        raise error_type(
            "cluster attach refused: cluster.transport_security must be "
            "'trusted_fabric'. Peer authentication is unavailable at the attach API "
            "dgx-monarch calls; isolate and source-filter the complete fabric interface "
            "(SECURITY.md).")
    if not config.client_bind:
        factory.safe_transport_failure(handoff)
        raise error_type(
            "cluster.toml is missing cluster.client_bind. The default transport "
            "advertises the hostname, which often resolves to loopback, and the "
            "attach then fails with MESH_ATTACH_CONFIG_TIMEOUT (docs/TROUBLESHOOTING.md "
            '#1). Set it to a fabric address of this host, for example client_bind = "tcp://<fabric-ip>:0".')
    addresses = [host.address for host in config.hosts]
    trace = attach_trace.AttachTrace(runtime["log"], addresses)
    trace.opened(
        client_bind=config.client_bind,
        attach_config_timeout=os.environ.get(
            "HYPERACTOR_MESH_ATTACH_CONFIG_TIMEOUT", "unset"),
        transport_bound=runtime["_TRANSPORT_BIND"] is not None,
        released=released, auto_heal=config.auto_heal,
        init_wait_s=ATTACH_INIT_WAIT_S)
    # Always close the trace, including unexpected failures. Known failure
    # paths record their own stage; the guard supplies stage=unexpected otherwise.
    with attach_trace.terminal(trace):
        with runtime["_TRANSPORT_ENABLE_LOCK"]:
            if runtime["_TRANSPORT_POISON"] is not None:
                trace.closed("refused", stage="transport_poisoned",
                             reason="earlier_attempt")
                raise error_type(
                    "cluster transport initialization is poisoned by an earlier attempt; "
                    "restart ComfyUI before retrying")
            bound = runtime["_TRANSPORT_BIND"]
            if bound is not None and bound != config.client_bind:
                factory.safe_transport_failure(handoff)
                trace.closed("refused", stage="transport_bind",
                             reason="client_bind_mismatch")
                raise error_type(
                    "this ComfyUI session already enabled the cluster transport on "
                    f"{bound}; switching it to {config.client_bind} requires a restart")
            if bound is None:
                try:
                    with trace.phase("transport_enable"):
                        enable_transport(config.client_bind)
                    runtime["_TRANSPORT_BIND"] = config.client_bind
                except BaseException as exc:
                    evidence = factory.safe_failure_evidence(exc)
                    runtime["_TRANSPORT_POISON"] = (
                        f"cluster transport enable failed: {evidence}")
                    # This poisons the session before any attach runs, and the
                    # runbook (docs/TROUBLESHOOTING.md #101) says the close line
                    # names the stage that poisoned it. Without this close the
                    # header is the attempt's last line.
                    trace.closed("poisoned", stage="transport_enable",
                                 evidence=evidence)
                    if not isinstance(exc, Exception):
                        raise
                    raise error_type(
                        f"cluster transport initialization failed ({evidence}); Monarch "
                        "may have partially initialized process-global state, so restart "
                        "ComfyUI before retrying") from exc
        with attach_trace.active(trace):
            return _attach_and_retry(
                runtime, config, handoff, addresses, trace, released=released)


def snapshot_generations(config: ClusterConfig, trace: attach_trace.AttachTrace,
                         *, phase: str) -> None:
    """Record each loop's listener generation without raising.

    Read immediately before attach to diagnose listener changes. Restarting one
    loop under a live session can require restarting both loops together
    (docs/TROUBLESHOOTING.md #2).
    """
    if not config.hosts:
        return
    try:
        from .cli import lifecycle, listener_generation

        readings = listener_generation.fleet(
            config, runner=lifecycle.run_on_host, timeout=GENERATION_PROBE_S)
    except Exception as exc:
        trace.emit("loop_generation", phase=phase, error=type(exc).__name__)
        return
    for address, fields in readings.items():
        trace.generation(address, fields, phase=phase)


def _traced_attach(runtime: dict[str, Any], config: ClusterConfig,
                   trace: attach_trace.AttachTrace, addresses: list[str],
                   *, attempt: int):
    """One attach attempt, with each loop's generation read just before it."""
    snapshot_generations(config, trace, phase=f"before_attach_{attempt}")
    with trace.phase("attach", attempt=attempt):
        return runtime["_attach_once"](addresses)


def _attach_and_retry(runtime: dict[str, Any], config: ClusterConfig, handoff: Any,
                      addresses: list[str], trace: attach_trace.AttachTrace,
                      *, released: bool):
    """Heal and attach, with no second attempt after a liveness-based replacement.

    Hardware checks found that a driver which released a dead fleet's
    replacement latch could not attach its replacement in the same
    process. Give it one attempt so an upstream fix remains observable. Ordinary
    attaches retain two attempts. Recovery requires a fresh driver process; see
    docs/TROUBLESHOOTING.md #2.
    """
    factory = runtime["mesh_factory"]
    error_type = runtime["MeshAttachError"]
    if config.auto_heal:
        try:
            with trace.phase("auto_heal"):
                runtime["_heal_dead_loops"](config, addresses)
        except Exception as exc:
            factory.safe_transport_failure(handoff)
            evidence = factory.safe_failure_evidence(exc)
            # No attach ran, so the attempt ends here. The close says so
            # rather than leaving the heal lines as the tail of the sequence.
            trace.closed("failed", stage="auto_heal", evidence=evidence)
            raise error_type(
                f"pre-attach Worker-service auto-heal failed ({evidence}); no worker "
                "attach was attempted, so fix the lifecycle/SSH error and retry") from exc
    started = time.perf_counter()
    try:
        hosts = _traced_attach(runtime, config, trace, addresses, attempt=1)
    except BaseException as first_exc:
        first_evidence = factory.safe_failure_evidence(first_exc)
        if not isinstance(first_exc, Exception):
            runtime["_TRANSPORT_POISON"] = (
                f"cluster attach failed: {first_evidence}")
            trace.closed("poisoned", stage="attach_1", evidence=first_evidence)
            raise
        # A released fleet stops here rather than spending a second attempt
        # it cannot win (see the docstring above). One attempt still runs, not
        # zero: a torchmonarch release that fixes in-process re-attach then
        # shows in the trace as attempt_1 succeeding, instead of hiding behind
        # a fail-fast that never tries.
        if released:
            runtime["_TRANSPORT_POISON"] = (
                f"cluster attach failed: {first_evidence}")
            trace.closed("poisoned", stage="attach_1_released",
                         evidence=first_evidence)
            restarted = (restart_after_failed_attach(config, first_exc, runtime["log"])
                         if config.auto_heal else None)
            runtime["_raise_released_attach"](
                addresses, first_exc, auto_heal=config.auto_heal, restarted=restarted)
        runtime["log"].warning(
            "first cluster attach failed (%s); retrying once in-process",
            first_evidence)
        try:
            hosts = _traced_attach(runtime, config, trace, addresses, attempt=2)
        except BaseException as exc:
            evidence = factory.safe_failure_evidence(exc)
            if not isinstance(exc, Exception):
                runtime["_TRANSPORT_POISON"] = f"cluster attach failed: {evidence}"
                trace.closed("poisoned", stage="attach_2", evidence=evidence)
                raise
            runtime["_TRANSPORT_POISON"] = f"cluster attach failed: {evidence}"
            trace.closed("poisoned", stage="attach_2_failed", evidence=evidence)
            if config.auto_heal:
                restart_after_failed_attach(config, exc, runtime["log"])
            runtime["_raise_attach"](addresses, exc)
    runtime["log"].info(
        "attached %d host(s) in %.2fs", len(addresses),
        time.perf_counter() - started)
    trace.closed("attached", hosts=len(addresses))
    return hosts


def heal_dead_loops(config: ClusterConfig, addresses: list[str], logger: Any) -> None:
    """Restart Worker services only after observing at least one loop down.

    An unreachable host blocks healing and raises the pre-attach error. Missing
    SSH or host evidence cannot justify restarting a possibly healthy fleet.
    """
    from .cli import lifecycle
    from .cli.worker_health import passive_unhealthy_workers

    del addresses  # retained by the mesh.py seam for caller/test compatibility
    deadline = time.monotonic() + 30.0
    health = passive_unhealthy_workers(
        config, runner=lifecycle.run_on_host, deadline=deadline)
    attach_trace.note(logger, "heal_probe", dead=health.dead,
                      unobserved=health.unobserved)
    if health.unobserved:
        raise RuntimeError(
            "worker health was not observed on "
            f"{list(health.unobserved)}, so auto-heal restarted nothing: a probe "
            "that reached no verdict is not evidence a loop is down. Fix the host "
            "or SSH fault and retry, or set cluster.toml auto_heal=false")
    if not health.dead:
        return
    dead = list(health.dead)
    logger.warning(
        "Worker service unhealthy (%s); auto-heal is restarting it before "
        "attach (disable via cluster.toml auto_heal=false)", dead)
    # The restart is fleet-wide although the probe named only some loops: it
    # stops and starts every configured loop. A hand restart of one loop can
    # leave the pair unattachable (docs/TROUBLESHOOTING.md #2), so the line
    # names both the loops seen down and the loops the restart touched.
    started = time.monotonic()
    attach_trace.note(logger, "heal_restart", dead=dead,
                      restarted=[host.address for host in config.hosts])
    try:
        from .telemetry import emit

        emit("auto_heal", reason=f"dead loops: {dead}", phase="pre-attach")
        if not lifecycle.restart(config, sync=False):
            logger.warning(
                "auto-heal restart reported failure; attempting attach anyway")
            attach_trace.note(logger, "heal_settled", settled=False,
                              reason="restart_reported_failure",
                              wall_s=time.monotonic() - started)
            return
    except Exception as exc:
        raise RuntimeError(
            f"auto-heal restart raised: {failure_summary(exc)}") from exc
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        settled = passive_unhealthy_workers(
            config, runner=lifecycle.run_on_host, deadline=deadline)
        if not settled.dead and not settled.unobserved:
            time.sleep(min(3.0, max(0.0, deadline - time.monotonic())))
            attach_trace.note(logger, "heal_settled", settled=True,
                              wall_s=time.monotonic() - started)
            return
        time.sleep(1.0)
    logger.warning("auto-heal: Worker service health did not settle within 30s; "
                   "attach will likely fail")
    attach_trace.note(logger, "heal_settled", settled=False,
                      reason="health_did_not_settle",
                      wall_s=time.monotonic() - started)


def restart_after_failed_attach(config: ClusterConfig, exc: BaseException, logger: Any) -> bool | None:
    """Best-effort Worker-service restart after attach poisoned this client.

    Returns the lifecycle's own answer (True done, False failed, None unknown),
    or False when it raised, so a message never claims a restart that did not
    happen."""
    logger.warning(
        "attach failed (%s); restarting the Worker service so "
        "the next ComfyUI session starts clean", failure_summary(exc))
    attach_trace.note(logger, "post_fail_restart",
                      restarted=[host.address for host in config.hosts],
                      failure=type(exc).__name__)
    try:
        from .cli import lifecycle
        from .telemetry import emit

        emit("auto_heal", reason=failure_summary(exc)[:120], phase="post-fail")
        return lifecycle.restart(config, sync=False)
    except Exception as heal_exc:
        logger.warning("post-failure Worker-service restart failed too: %r", heal_exc)
        return False


def raise_attach(error_type: type[RuntimeError], addresses: list[str], exc: BaseException) -> None:
    """Raise the stable operator-facing error after both ordinary attempts fail.

    Only the two-attempt path calls this; a released fleet fails fast through
    ``raise_released_attach``, so the message needs no attempt count.
    """
    raise error_type(
        f"attach to workers {addresses} failed ({failure_summary(exc)}); the "
        "in-process retry already ran. Rule out version skew first "
        "(`dgxm doctor`), then a wedged Worker service: run `dgxm restart` and "
        "restart ComfyUI before retrying (docs/TROUBLESHOOTING.md #1, #2)."
    ) from exc


def raise_released_attach(error_type: type[RuntimeError], addresses: list[str],
                          exc: BaseException, *, auto_heal: bool,
                          restarted: bool | None = None) -> None:
    """Explain a failed attach after liveness-based fleet replacement.

    No second attempt ran. Name the supported recovery: restart both worker loops
    together, then start a fresh driver. Include the manual route when auto-heal
    is disabled.
    """
    if not auto_heal:
        restart_sentence = "Run `dgxm restart` first."
    elif restarted is True:
        restart_sentence = "The worker loops were restarted together."
    else:
        restart_sentence = ("The worker-loop restart did not confirm (see the log line "
                            "above), so run `dgxm restart` first.")
    raise error_type(
        f"attach to workers {addresses} failed ({failure_summary(exc)}) on a "
        "fleet this driver released after proving the old workers gone. A "
        "driver cannot attach again in the same process once its fleet "
        "changed under it (reproduced on hardware, 2026-10-01), so no in-process "
        f"retry ran. {restart_sentence} Restart ComfyUI and the next render "
        "attaches (docs/TROUBLESHOOTING.md #101)."
    ) from exc
