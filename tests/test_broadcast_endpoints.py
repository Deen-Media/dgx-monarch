"""Every broadcast in the package names a plain endpoint whose body cannot raise.

``broadcast()`` sends no reply port. A plain ``@endpoint`` that raises there
fails its actor in 0.7.0 and 0.8.0 alike; since torchmonarch 0.8.0 a raising
``@concurrent_endpoint`` under broadcast fails the actor too
(meta-pytorch/monarch#5283), where 0.7.0 dropped the error. A new broadcast
site fails this test until it is reviewed and added to ``REVIEWED``.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"

# Endpoint name: why its body cannot raise.
REVIEWED = {
    # actor_lifetime.ActorLifetime.renew turns any value into a float or a
    # disabled lease under its own lock; nothing in it raises.
    "renew_client_lease": "actor_lifetime.renew",
    # Takes str() of a str the driver sends, edits a bounded dict under a
    # lock and sets an Event; no I/O beyond gethostname.
    "cancel_sample": "actor/cancellation.py",
}


def _broadcast_endpoints() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "broadcast"):
                # An endpoint reached any other way than ``<mesh>.<endpoint>`` cannot be
                # named here, so it fails the test until someone reviews it.
                name = (node.func.value.attr if isinstance(node.func.value, ast.Attribute)
                        else f"<unnamed: {ast.unparse(node.func)}>")
                found.setdefault(name, []).append(f"{path.relative_to(SRC)}:{node.lineno}")
    return found


def _endpoint_decorators() -> dict[str, set[str]]:
    decorators: dict[str, set[str]] = {}
    for path in sorted((SRC / "actor").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = {getattr(item, "id", getattr(item, "attr", None)) for item in node.decorator_list}
                if names & {"endpoint", "concurrent_endpoint"}:
                    decorators.setdefault(node.name, set()).update(names)
    return decorators


def test_every_broadcast_site_names_a_reviewed_plain_endpoint():
    sites = _broadcast_endpoints()
    assert sites, "the scan found no broadcast site; the guard would pass vacuously"
    assert set(sites) <= set(REVIEWED), {name: where for name, where in sites.items()
                                         if name not in REVIEWED}
    decorators = _endpoint_decorators()
    for name in sites:
        assert decorators.get(name) == {"endpoint"}, (name, decorators.get(name))


@pytest.mark.parametrize("value", [None, "x", float("nan"), math.inf, -1, 0, 1e300, object()])
def test_the_lease_renewal_body_takes_any_value(value):
    from dgx_monarch.actor_lifetime import ActorLifetime

    ActorLifetime().renew(value)  # type: ignore[arg-type]
