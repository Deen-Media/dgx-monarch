"""The one line-limit ledger for source modules above the default cap."""
from __future__ import annotations

# Any src module over this needs a reviewed ledger row below.
DEFAULT_CAP = 500

CEILINGS = {
    # Lifecycle and operator coordinators retain their transaction spines here.
    "nodes/auto_gate.py": 506, "mesh.py": 898, "cli/doctor.py": 774,
    "cli/actor_reaper.py": 603, "adoption_evidence.py": 732,
    # Loader, worker, and Comfy integration seams stay with the state they bind.
    "actor/unbake.py": 692, "config.py": 658, "actor/sampling.py": 620,
    "actor/comfy_bridge.py": 543, "actor/worker.py": 509, "comfy_surface.py": 526,
    # FSDP and capacity pricing are deliberately centralized for exact accounting.
    "adapters/fsdp.py": 589, "actor/capacity_quote.py": 539,
    "capacity_agreement.py": 567, "gate_ledger.py": 527,
    "nodes/gate_identity.py": 583, "nodes/loader_preflight.py": 593,
    "runtime_provenance.py": 596, "adapters/flux_family.py": 560,
}


def ceiling_for(relative: str) -> int:
    """This file's ceiling: its ledger row, or the default cap when it has none.

    A file that leaves the ledger keeps a guard rather than losing one, and a
    file whose row is raised raises everywhere in the same edit.
    """
    return CEILINGS.get(relative, DEFAULT_CAP)
