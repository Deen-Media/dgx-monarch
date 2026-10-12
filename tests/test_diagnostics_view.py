"""Browser and terminal show the same bounded measurements, never raw labels."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dgx_monarch.tui.diagnostics_view import diagnostics_view


def test_browser_and_terminal_projection_agree():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node unavailable")
    payload = {"diagnostics": {"schema": 1, "available": True, "recent": [
        {"phase": "collect", "outcome": "ok", "return_mode": "message", "duration_s": 0.02},
        {"phase": "private payload", "outcome": "ok", "return_mode": "rdma", "duration_s": 1},
        {"phase": "rdma_read", "outcome": "timeout", "return_mode": "rdma", "duration_s": 2.5},
    ]}}
    # Worker cache freshness cannot make these current driver timings stale,
    # and cached worker phases must never masquerade as driver observations.
    stale_workers = {**payload, "status_poll": {"fresh": False}, "workers": [{
        "diagnostics": {"schema": 1, "available": True, "recent": [{
            "phase": "compute", "outcome": "ok", "return_mode": "none", "duration_s": 99,
        }]},
    }]}
    fixtures = [payload, stale_workers, {"workers": stale_workers["workers"]},
                {**payload, "telemetry_error": "stale"}, {}, None]
    assert diagnostics_view(stale_workers) == diagnostics_view(payload)
    assert "99" not in diagnostics_view(stale_workers)["summary"]
    model = Path(__file__).parents[1] / "web/js/dgx_monarch_diagnostics_model.js"
    program = (f"import {{ diagnosticsPanelModel }} from {json.dumps(model.as_uri())};"
               f"process.stdout.write(JSON.stringify({json.dumps(fixtures)}.map(diagnosticsPanelModel)));")
    result = subprocess.run([node, "--input-type=module", "-e", program],
                            capture_output=True, text=True, timeout=10, check=True)
    assert json.loads(result.stdout) == [diagnostics_view(item) for item in fixtures]
    assert "private payload" not in result.stdout
