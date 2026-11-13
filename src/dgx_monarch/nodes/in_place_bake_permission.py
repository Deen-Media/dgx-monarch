"""Tell the Workers when the identity gate has passed this render's in-place LoRA bake.

With ``lora_low_rss`` off a Worker bakes a LoRA stack ComfyUI's classic way.
Where that bake's price does not fit and baking in place on the loaded model
would, the store may bake in place (``actor/in_place_fallback.py``). That is
another bake path, and a memory rule must not move a combination onto a path
the identity gate has not passed. A graph that turns the switch off runs no
ceremony at all (``nodes/auto_gate.py``), so "the pack did not force the
switch off" proves nothing.

``stamp`` therefore sends a permission and never a prohibition. It sets the
request's ``capacity_classic_bake.IN_PLACE_BAKE_PROVEN`` to ``True`` only
when the gate ledger holds an exact PASS under the key the gate's own grant
reads (the pair's key for a dual-model render, the model's own otherwise), and
no FAIL in the ledger or in this session under that key or, for a dual-model
render, the conditional model's own. The capability context it reads is this
render's own with ``lora_low_rss`` on (or auto, or unset: the three ways a
graph asks for the in-place bake), which is where a default render of the
same combination records its PASS and is the context the in-place bake runs
under. A FAIL under any of the three withholds it. Anything else leaves the
field out: a combination never rendered with
the switch on, a FAIL, a ledger that cannot be read, a render with no stack
or with the switch already on, a gate ceremony in progress. An absent field is
no permission on every Worker, so an older driver, a requeue built without it
and a recycle all keep the classic bake.

The field rides the per-dispatch sample request and never the worker
arguments: the gate's capability context holds those whole
(``nodes/gate_identity.py``), and a new one would change every combination's
key. ``render_submit`` strips the field from every incoming request before it
stamps, so nothing but this lookup can set it.
"""
from __future__ import annotations

from typing import Any

from ..capacity_classic_bake import IN_PLACE_BAKE_PROVEN
from ..log import get_logger

log = get_logger(__name__)


def proven(model: Any, handle: Any, *, resolved_topology: Any = None, resolved_attention: str | None = None,
           uncond: dict | None = None) -> bool:
    """Whether the gate ledger holds an exact PASS for this render's in-place bake, and no FAIL."""
    from .common import _AUTO_GATE_ACTIVE

    if getattr(_AUTO_GATE_ACTIVE, "on", False) or handle is None:
        return False
    if not (getattr(model, "loras", None) or (uncond or {}).get("loras")):
        return False
    try:
        from ..gate_ledger import GateLedger, comfy_commit, gate_verdict_token
        from .common import _process_gate_verdict
        from .gate import _combo_of, _dual_combo_of, _effective_worker_args, _ledger_dir, gate_capability_context

        effective = dict(_effective_worker_args(model, handle))
        if effective.get("lora_low_rss") is not False:
            return False  # the default already bakes in place; there is nothing to permit
        keys = ([_dual_combo_of(model, uncond)] if uncond is not None else []) + [_combo_of(model)]
        ledger, commit = GateLedger(_ledger_dir()), comfy_commit()
        rest = {name: value for name, value in effective.items() if name != "lora_low_rss"}
        passed = False
        # The three ways a graph says "bake in place": the switch on, auto, or not set at all.
        for worker_args in ({**rest, "lora_low_rss": True}, {**rest, "lora_low_rss": "auto"}, rest):
            context = gate_capability_context(
                model, handle, effective_worker_args=worker_args,
                resolved_topology=resolved_topology, resolved_attention=resolved_attention)
            states = []
            for key, artifacts in keys:
                digest = artifacts if isinstance(artifacts, str) else artifacts.current
                if _process_gate_verdict(gate_verdict_token(key, digest, commit, context)) == "FAIL":
                    return False
                states.append(ledger.lookup(key, artifacts, commit, context))
            if "fail" in states:
                return False
            # The PASS is the one the gate's own grant reads: the pair's key for a dual-model
            # render (nodes/gate_dual.py), the model's own otherwise. It is first in ``keys``.
            passed = passed or states[0] == "pass"
        return passed
    except Exception as exc:
        # No permission is always safe: the classic bake keeps its own price and its own refusal.
        log.info("in-place bake permission: the gate ledger could not be read (%r), so none is sent", exc)
        return False


def stamp(request: dict, model: Any, handle: Any, *, resolved_topology: Any = None,
          resolved_attention: str | None = None) -> None:
    """Set the request's permission field when ``proven``; leave it absent otherwise."""
    request.pop(IN_PLACE_BAKE_PROVEN, None)
    if proven(model, handle, resolved_topology=resolved_topology, resolved_attention=resolved_attention,
              uncond=request.get("uncond_model")):
        request[IN_PLACE_BAKE_PROVEN] = True
        log.info("in-place bake permission: the identity gate has passed this combination's in-place "
                 "bake, so a Worker may bake its stack in place where the classic bake does not fit")


__all__ = ["proven", "stamp"]
