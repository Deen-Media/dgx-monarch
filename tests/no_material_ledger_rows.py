"""Gate ledger rows as the verdict half writes them, for the consent readers' tests.

``write_ceremony_rows`` calls the real writer (``gate_verdict.record_verdict_rows``)
on a real ``GateLedger``: the exercised row, and for a ceremony that rendered,
its Fleet row and two explicit-on slab sibling rows. The contexts are a gate's
own, so none of them equals a load's consent scope.
"""
from __future__ import annotations

from typing import Any

GATE_CONTEXT = {"worker_args": {"slab_weights": "auto"}, "world": 2,
                "topology": {"ulysses": 2, "ring": 1, "cfg": 1, "dp": 1, "fsdp": False}}
FLEET_CONTEXT = {"worker_args": {"slab_weights": "auto"}, "world": 2,
                 "capability": "fleet_per_rank_residency"}
SIBLING_CONTEXTS = (
    {"worker_args": {"slab_weights": True}, "world": 2,
     "topology": {"ulysses": 2, "ring": 1, "cfg": 1, "dp": 1, "fsdp": False}},
    {"worker_args": {"slab_weights": True}, "world": 2,
     "capability": "fleet_per_rank_residency"},
)


def write_ceremony_rows(ledger: Any, key: str, artifacts: Any, *, kind: str = "no_material",
                        rendered: bool = True, commit: str = "deadbeef") -> None:
    """Write one INCONCLUSIVE verdict of ``kind`` ("no_material" or "unproven").

    ``rendered`` False writes what the no-render path writes: the exercised row alone.
    """
    from dgx_monarch.nodes import gate_inconclusive, gate_verdict

    result = {"verdict": "INCONCLUSIVE", gate_inconclusive.RESULT_KEY: kind}
    detail = {"model": "m.safetensors", "loras": 0, "origin": "auto_first_use", "run_id": "",
              "max_abs_latent_diff": 0.0 if rendered else None, "quarantine_levers": []}
    gate_verdict.record_verdict_rows(
        {"_gate_inconclusive": gate_inconclusive}, ledger, key, artifacts, commit,
        "INCONCLUSIVE", detail, result, GATE_CONTEXT,
        FLEET_CONTEXT if rendered else None, list(SIBLING_CONTEXTS) if rendered else [])
