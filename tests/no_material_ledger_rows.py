"""Gate ledger rows as a ceremony writes them, for the consent readers' tests.

``write_ceremony_rows`` calls the real writers on a real ``GateLedger``: for a
ceremony that rendered, the RETESTING row it opens before its first side
effect (``GateLedger.begin_retest_required``, its contexts in the ceremony's
order: the exercised one, the Fleet one, the explicit-on slab siblings), then
the verdict rows (``gate_verdict.record_verdict_rows``): the exercised row,
its Fleet row and two explicit-on slab sibling rows. The no-render path writes
the exercised row alone. ``write_explicit_on_rows`` writes what a ceremony run
with ``slab_weights`` on writes: those siblings are its own contexts, and it
has none. The contexts are a gate's own, so none of them equals a load's
consent scope.
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
# What the explicit-on ceremony's own two contexts are: the auto ceremony's siblings.
EXPLICIT_ON_CONTEXT, EXPLICIT_ON_FLEET_CONTEXT = SIBLING_CONTEXTS


def _open_retest(ledger: Any, key: str, artifacts: Any, commit: str, contexts: list,
                 origin: str) -> None:
    ledger.begin_retest_required(key, artifacts, commit, contexts, {
        "model": "m.safetensors", "loras": 0, "origin": origin, "run_id": "",
        "phase": "ceremony-preflight"})


def open_auto_ceremony(ledger: Any, key: str, artifacts: Any, commit: str = "deadbeef") -> None:
    """The RETESTING row alone of a ceremony on the auto graph, as one that was cut off leaves it."""
    _open_retest(ledger, key, artifacts, commit, [GATE_CONTEXT, FLEET_CONTEXT, *SIBLING_CONTEXTS],
                 "auto_first_use")


def write_ceremony_rows(ledger: Any, key: str, artifacts: Any, *, kind: str = "no_material",
                        rendered: bool = True, commit: str = "deadbeef") -> None:
    """Write one verdict of ``kind``: INCONCLUSIVE "no_material" or "unproven", a "stock
    reference" INCONCLUSIVE (agreeing slab legs, a stock leg a typed class C refusal kept from
    loading), or a terminal "PASS" or "FAIL" whose cross-residency leg passed or diverged.

    ``rendered`` False writes what the no-render path writes: the exercised row alone. A
    PASS or FAIL is written with its explicit-on siblings, stamped by slab-mode equivalence,
    as ``gate_identity.equivalent_slab_mode_contexts`` returns them for such a cross leg.
    """
    from dgx_monarch.nodes import gate_inconclusive, gate_verdict

    if rendered:
        open_auto_ceremony(ledger, key, artifacts, commit)
    terminal = kind in ("PASS", "FAIL")
    verdict = kind if terminal else "INCONCLUSIVE"
    classified = gate_inconclusive.KIND_UNPROVEN if kind == "stock reference" else kind
    result = {"verdict": verdict} if terminal else {"verdict": verdict, gate_inconclusive.RESULT_KEY: classified}
    detail = {"model": "m.safetensors", "loras": 0, "origin": "auto_first_use", "run_id": "",
              "max_abs_latent_diff": 0.0 if rendered else None,
              "quarantine_levers": ["slab_weights"] if kind == "FAIL" else []}
    if terminal:
        detail["cross_mode"] = kind
    elif kind == "stock reference":
        detail.update(cross_mode="CAPACITY",
                      capacity_detail="StockLoadCapacityError/class-C guard=stock_load_preflight")
    gate_verdict.record_verdict_rows(
        {"_gate_inconclusive": gate_inconclusive}, ledger, key, artifacts, commit,
        verdict, detail, result, GATE_CONTEXT,
        FLEET_CONTEXT if rendered else None, list(SIBLING_CONTEXTS) if rendered else [])


def write_explicit_on_rows(ledger: Any, key: str, artifacts: Any, outcome: str,
                           commit: str = "deadbeef") -> None:
    """What a ceremony run with ``slab_weights`` on writes, in its own two contexts.

    ``outcome``: "repeat diverged" (INCONCLUSIVE: the same-residency repeat
    diverged though the cross-residency leg passed), "cross leg error"
    (INCONCLUSIVE on an ERROR of that leg), "cut off" (the RETESTING row alone,
    as a ceremony that was interrupted leaves it), "cross leg failed" (a FAIL
    whose cross-residency leg diverged) or "PASS".
    """
    from dgx_monarch.nodes import gate_inconclusive, gate_verdict

    own = [EXPLICIT_ON_CONTEXT, EXPLICIT_ON_FLEET_CONTEXT]
    _open_retest(ledger, key, artifacts, commit, own, "explicit")
    if outcome == "cut off":
        return
    verdict = {"PASS": "PASS", "cross leg failed": "FAIL"}.get(outcome, "INCONCLUSIVE")
    cross = {"repeat diverged": "PASS", "cross leg error": "ERROR", "cross leg failed": "FAIL",
             "PASS": "PASS"}[outcome]
    detail = {"model": "m.safetensors", "loras": 0, "origin": "explicit", "run_id": "",
              "max_abs_latent_diff": 0.0625 if outcome in ("repeat diverged", "cross leg failed") else 0.0,
              "quarantine_levers": ["slab_weights"] if verdict == "FAIL" else [], "cross_mode": cross}
    result = {"verdict": verdict}
    if verdict == "INCONCLUSIVE":
        result[gate_inconclusive.RESULT_KEY] = gate_inconclusive.KIND_UNPROVEN
    # An explicit-on policy has no sibling (gate_identity.equivalent_slab_mode_contexts).
    gate_verdict.record_verdict_rows(
        {"_gate_inconclusive": gate_inconclusive}, ledger, key, artifacts, commit,
        verdict, detail, result, EXPLICIT_ON_CONTEXT, EXPLICIT_ON_FLEET_CONTEXT, [])
