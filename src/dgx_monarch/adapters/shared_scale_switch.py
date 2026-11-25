"""The shared NVFP4 activation scale's two switches, as a Worker reads them.

Off switch and first-step trace, both read once per install and both failing
open, like the env-gated compile in actor/worker_compile.py. The off switch
decides whether a rank issues a collective, so every rank must read the same
value: a Worker writes its variable from the worker args the driver sent it,
an absent key reading on (``env_from_worker_args``), never from its own
environment, so a variable exported by hand on one box cannot make the ranks
differ. The payload stays the published policy, so the policy a Worker
records still equals the one a residency grant names (mesh_residency.py). As
a second line the post-load readiness exchange carries each rank's count of
wrapped Linears and refuses on disagreement (actor/rank_readiness.py). The
trace switch only logs, and its bare variable stays a single-box aid. The
scale itself lives in adapters/quant_activation_scale.py, which re-exports
these names.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

ENABLE_ENV = "DGXM_SHARED_ACT_SCALE"
LOG_ENV = "DGXM_LOG_ACT_SCALE"
ENABLE_ARG = "shared_act_scale"
LOG_ARG = "log_act_scale"


def enabled_now() -> bool:
    """Whether the shared scale is on for the install about to run."""
    return os.environ.get(ENABLE_ENV, "1") != "0"


def env_from_worker_args(worker_args: Mapping[str, Any],
                         previous: Mapping[str, Any] | None = None) -> bool:
    """Write both variables from the worker args; report an enable change.

    The off switch is written on every call, from the payload alone (an absent
    key reads on), so this rank's own environment never decides whether it
    issues a collective. The trace switch keeps its single-box aid: a key the
    last call carried and this one drops pops its variable, and a key neither
    call carried is left alone. The hook wraps forwards at load, so a caller
    holding residents evicts on a True return.
    """
    was = enabled_now()
    os.environ[ENABLE_ENV] = "1" if worker_args.get(ENABLE_ARG, True) else "0"
    if LOG_ARG in worker_args:
        os.environ[LOG_ENV] = "1" if worker_args.get(LOG_ARG) else "0"
    elif previous is not None and LOG_ARG in previous:
        os.environ.pop(LOG_ENV, None)
    return was is not enabled_now()
