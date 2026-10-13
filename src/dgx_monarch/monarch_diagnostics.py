"""Opt-in public Monarch metrics and spans; no Jobs, servers, or lifecycle IO.

Configure once outside runtime ownership locks. Metrics use a bounded daemon
queue so an exporter cannot block a render. OTLP destination and collection
interval belong to Monarch's startup environment, not this adapter. Disabling
stops new records; it does not claim to flush an external telemetry backend.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
from typing import Any

from . import runtime_diagnostics

_QUEUE_CAPACITY = 256
_LOCK = threading.Lock()
_EXPORTER: _MetricExporter | None = None
_STATE: dict[str, Any] = {"metrics": False, "traces": False, "available": True}
_POLICY_STATE: dict[str, Any] = {}


def configure_policy(policy: Any) -> dict[str, Any]:
    """Apply validated explicit policy before native provider initialization.

    In 0.7 the endpoint enables upstream logs as well as metrics. Their content
    is not the sanitized DGX diagnostic schema, so separate consent is required.
    An already initialized provider cannot be redirected or flushed here.
    """
    global _POLICY_STATE
    from dataclasses import asdict

    from .monarch_policy import parse_policy

    validated = parse_policy({"diagnostics": asdict(policy)}).diagnostics
    endpoint = validated.otlp_endpoint
    reason = start_refusal(validated)
    if reason:
        return _policy_refused(reason)
    if endpoint:
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    configure(metrics=validated.metrics)
    with _LOCK:
        _POLICY_STATE = {
            "export_scope": "upstream_metrics_and_logs" if endpoint else "no_managed_otlp",
            "upstream_log_export_acknowledged": validated.otlp_logs,
        }
    return status()


def start_refusal(policy: Any) -> str | None:
    """Why configure_policy would refuse this policy's endpoint, or None; reads only.

    The driver asks before it admits anything, so a refused edit never stops
    the running exporter: a refusal through configure_policy would.
    """
    endpoint = policy.otlp_endpoint
    existing = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    if not endpoint or existing == endpoint:
        return None
    if existing:
        return "ambient_otlp_endpoint_conflict"
    if any(name.startswith("monarch._rust_bindings") for name in sys.modules):
        return "otlp_restart_required"
    return None


def _policy_refused(reason: str) -> dict[str, Any]:
    """Persist startup failure so ignored return values cannot imply success."""
    global _POLICY_STATE

    configure()
    with _LOCK:
        _POLICY_STATE = {"available": False, "reason": reason, "export_scope": "unavailable"}
    return status()


def requested_refusal(state: Any, policy: Any) -> str | None:
    """Why requested metrics or OTLP export are unavailable; None when not requested or running."""
    if not (policy.metrics or policy.otlp_endpoint) or not isinstance(state, dict):
        return None
    return None if state.get("available") is not False else str(state.get("reason") or "unavailable")


def configure_policy_logged(policy: Any, role: str) -> dict[str, Any]:
    """Actor, Worker and driver-restore form: log a refusal at ERROR and continue.

    A raise here would surface as a spawn failure, which poisons the driver
    transport; only the driver refuses (monarch_launch.prepare_driver). Its
    restore_driver also uses this form, since a failed restore must not mask
    the error that caused it.
    """
    state = configure_policy(policy)
    reason = requested_refusal(state, policy)
    if reason:
        from .log import get_logger

        get_logger(__name__).error(
            "Monarch diagnostics requested for this %s are unavailable (%s); it continues "
            "without them", role, reason)
    return state


class _MetricExporter:
    def __init__(self, meter: Any) -> None:
        # The 0.7 Python wrapper does not implement gauges/observables or
        # custom histogram boundaries. Never request unsupported instruments.
        self._count = meter.create_counter("dgxm.phase.completed")
        self._duration = meter.create_histogram("dgxm.phase.duration.seconds")
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(_QUEUE_CAPACITY)
        self._stopped = threading.Event()
        self.dropped = 0
        self.errors = 0
        self._thread = threading.Thread(target=self._run, name="dgxm-metrics", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, event: dict[str, Any]) -> None:
        if self._stopped.is_set():
            return
        try:
            self._queue.put_nowait(dict(event))
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        while not self._stopped.is_set():
            try:
                event = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                labels = {key: event[key] for key in ("phase", "outcome", "return_mode")}
                self._count.add(1, labels)
                self._duration.record(event["duration_s"], labels)
            except BaseException:
                self.errors += 1
            finally:
                self._queue.task_done()

    def stop(self) -> None:
        self._stopped.set()


def configure(*, metrics: bool = False, traces: bool = False) -> dict[str, Any]:
    """Enable supported public instruments explicitly; imports remain lazy.

An unavailable runtime reports failure without enabling either consumer.
Calling again with unchanged settings is idempotent. Disable before changing
settings. A blocked old exporter prevents a replacement, avoiding accumulation
of daemon threads when an external backend is unhealthy.
"""
    global _EXPORTER, _STATE
    with _LOCK:
        if (metrics == _STATE["metrics"] and traces == _STATE["traces"]
                and _STATE["available"]):
            return status()
        runtime_diagnostics.configure_export()
        if _EXPORTER is not None:
            _EXPORTER.stop()
            if _EXPORTER._thread.ident is not None:
                _EXPORTER._thread.join(timeout=0.2)
            if _EXPORTER._thread.is_alive():
                _STATE = {"metrics": False, "traces": False, "available": False,
                          "reason": "exporter_still_running"}
                return status()
            _EXPORTER = None
        _STATE = {"metrics": False, "traces": False, "available": True}
        if not metrics and not traces:
            return status()
        try:
            from monarch.actor import get_meter, span

            exporter = _MetricExporter(get_meter()) if metrics else None
            _EXPORTER = exporter
            if exporter is not None:
                exporter.start()
            runtime_diagnostics.configure_export(
                exporter.submit if exporter else None,
                (lambda name: span(f"dgxm.{name}")) if traces else None,
            )
            _STATE = {"metrics": metrics, "traces": traces, "available": True}
        except BaseException:
            if _EXPORTER is not None:
                _EXPORTER.stop()
            _STATE = {"metrics": False, "traces": False, "available": False,
                      "reason": "instrument_setup_failed"}
        return status()


def status() -> dict[str, Any]:
    """Consumer status only; it makes no export-delivery or coverage claim."""
    result = {**_STATE, **_POLICY_STATE}
    exporter = _EXPORTER
    if exporter is not None:
        result.update(queue_capacity=_QUEUE_CAPACITY, queued=exporter._queue.qsize(),
                      dropped=exporter.dropped, errors=exporter.errors)
    return result
