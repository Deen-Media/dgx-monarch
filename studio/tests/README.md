# CPU HTTP integration fixture

`mock_driver.py` is a deliberately synthetic, loopback-only Comfy-compatible
HTTP fixture. It has one named `StudioCPUFixture` node, performs no inference,
and returns a small deterministic RGB PNG. It cannot contact hardware or execute
management operations. Its telemetry explicitly identifies the CPU fixture.

Start the fixture and export a workflow:

```sh
python3 studio/tests/mock_driver.py --port 18189 --export-workflow /tmp/studio-cpu-workflow.json
```

In a Python environment with the optional Studio dependency installed, start a
separate companion with a new temporary state directory:

```sh
python -m dgx_monarch.studio --port 18221 --driver http://127.0.0.1:18189 --state-dir /tmp/studio-cpu-smoke
```

Do not pass a cluster configuration, a driver service, or `--allow-management`.
Copy the ephemeral owner token into a mode-0600 temporary file, then run:

```sh
python3 studio/tests/http_smoke.py --companion http://127.0.0.1:18221 --token-file /tmp/studio-token
```

The smoke verifies workflow inspection, submission preview, idempotent duplicate
submission, request reconciliation, terminal job state, immutable workflow/seed,
and authenticated artifact retrieval through the real companion HTTP routes.
It refuses drivers without the explicit fixture node and refuses enabled
management operations other than the read-only version operation.

For manual browser verification, open that separate companion, enter its token,
and import `/tmp/studio-cpu-workflow.json`. Edit the visible prompt/seed controls,
review and submit, then select the resulting PNG from recent outputs. This is
synthetic integration evidence only; it is not GPU generation or fidelity proof.
Terminate both fixture processes and delete the temporary token when finished.

The HTTP smoke passed during development. The initial interface had native-browser
verification for offline-driver operation, durable records, navigation and
presentation mode. After component extraction, the available browser inventory
was empty, so final component structure has type/unit/build and HTTP integration
coverage, with browser regression verification still required.
