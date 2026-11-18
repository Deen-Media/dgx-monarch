"""Occupancy levels through the driver's own metrics path to an OTLP endpoint.

test_occupancy_otlp.py runs it as a subprocess with OTEL_EXPORTER_OTLP_ENDPOINT
pointing at a loopback sink, set before this interpreter starts, as the
driver's policy installs it before Monarch loads. The levels come from a
stand-in reader; the policy admission, the exporter thread, the instruments
and torchmonarch's meter and OTLP exporter are the real ones. Metrics turn off
and on again in the middle, with the level changed while they were off.

Prints exactly one JSON object as the last stdout line.
"""
from __future__ import annotations

import json
import os
import time


def main() -> None:
    from dgx_monarch import monarch_diagnostics, occupancy_metrics
    from dgx_monarch.monarch_policy import DiagnosticsPolicy

    level = {"dgxm.mesh.active_leases": 3}
    occupancy_metrics.read_levels = lambda: dict(level)
    endpoint = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
    # Each hold spans several one-second samples and one-second exports, so a
    # loaded machine still delivers every level before the next step.
    first = monarch_diagnostics.configure_policy(
        DiagnosticsPolicy(metrics=True, otlp_endpoint=endpoint, otlp_logs=True))
    time.sleep(6.0)
    monarch_diagnostics.configure()
    level["dgxm.mesh.active_leases"] = 5
    time.sleep(3.0)
    again = monarch_diagnostics.configure(metrics=True)
    time.sleep(6.0)
    level["dgxm.mesh.active_leases"] = 0
    time.sleep(6.0)
    print(json.dumps({"first": first.get("metrics"), "again": again.get("metrics")}), flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
