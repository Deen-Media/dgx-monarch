"""Driver admission with a per-signal OTLP endpoint in the environment.

test_otlp_signal_guard.py runs it as a subprocess with one of
OTEL_EXPORTER_OTLP_{LOGS,METRICS,TRACES}_ENDPOINT pointing at a loopback sink
and export off in the policy. It admits the driver the way a mesh build does
(monarch_launch.prepare_driver). A refused admission prints the refusal and
exits before Monarch loads. An admitted one spawns a local CPU actor, calls
it, and waits so torchmonarch 0.8.0's exporters would post what they hold.

Prints exactly one JSON object as the last stdout line.
"""
from __future__ import annotations

import json
import sys
import time


def main() -> None:
    from dgx_monarch.config import ClusterConfig
    from dgx_monarch.config_schema import ClusterConfigError
    from dgx_monarch.monarch_launch import prepare_driver

    try:
        prepare_driver(ClusterConfig())
    except ClusterConfigError as exc:
        print(json.dumps({"admitted": False, "refusal": str(exc),
                          "monarch_loaded": "monarch" in sys.modules}), flush=True)
        return
    import monarch.actor as monarch_actor

    class Echo(monarch_actor.Actor):
        @monarch_actor.endpoint
        async def echo(self, value: int) -> int:
            return value

    procs = monarch_actor.this_host().spawn_procs(per_host={"procs": 1})
    echo = procs.spawn("echo", Echo)
    for value in range(5):
        echo.echo.call(value).get(timeout=60)
    time.sleep(8.0)
    procs.stop("otlp probe done").get(timeout=30)
    time.sleep(2.0)
    print(json.dumps({"admitted": True}), flush=True)


if __name__ == "__main__":
    main()
