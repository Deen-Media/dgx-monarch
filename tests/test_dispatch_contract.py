"""Pin the dispatch property every out-of-band worker call depends on.

cancel_sample (actor/cancellation.py), status and the pipelined
artifact_identity preflight (actor/worker.py) assume the actor services a
second endpoint while sample() is in flight. Nothing else in the suite
exercises that property through a real actor mesh, so a torchmonarch
dispatch-model change (such as the 0.6.0 queue-dispatch default, upstream PR
meta-pytorch/monarch#4211) would pass every other test with cancellation
broken. Rerun this test, and reread the actor endpoints, on every
torchmonarch pin bump.

The probe runs in a subprocess because monarch allows one transport per
process and the suite must stay reusable after this test.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROBE = Path(__file__).with_name("dispatch_contract_probe.py")


def test_out_of_band_calls_interleave_with_inflight_sample():
    proc = subprocess.run(
        [sys.executable, str(PROBE)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, (
        f"probe crashed rc={proc.returncode}\n"
        f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
    )
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    # Interleaved dispatch answers in milliseconds; a serialized dispatch
    # loop holds both calls for the rest of the 3 s probe body (SLOW_S).
    assert result["status_latency_s"] < 1.0, result
    assert result["cancel_latency_s"] < 1.0, result
    # The cancel reached the render before it finished: a serialized loop
    # would run cancel only after sample had returned "completed".
    assert result["sample_outcome"] == "cancelled", result
    # Error contract of the bare @concurrent_endpoint form: a raising body
    # reaches the caller as ActorError and the actor survives, so a
    # recoverable render error never evicts the mesh. A SupervisionError means
    # the endpoint fails the actor as an explicit_response_port endpoint does.
    assert result["error_contract"] == "ActorError", result
    assert result["actor_survives"] is True, result
