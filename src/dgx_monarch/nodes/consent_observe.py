"""Node-side spelling of the refusal observer in ``dgx_monarch.consent_observe``.

`pending` reads this module at call time and the accuracy-waiver tests reach
`_already_granted` through it, so both spellings stay live.
"""
from __future__ import annotations

from ..consent_observe import (  # noqa: F401  # Public consent_observe compatibility API.
    _already_granted,
    observe_refusal,
    refusal_text,
)
