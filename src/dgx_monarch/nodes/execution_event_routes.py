"""GET /dgxm/execution-events: replay of prompt-bound runtime observations.

This route is not read-only. Every GET or HEAD renews the 30 s observer lease
in `execution_events.LEASE_SECONDS`. While the lease holds, each render
captures its ComfyUI execution context, carries the `_dgxm_observe_runtime`
request key and asks its Workers for `runtime_identity`; once the lease lapses
those hooks stop. The lease grants no authority over sampling, teardown, ACK
or retry, and the journal it fills keeps at most `MAX_EVENTS` events and
`MAX_BYTES` bytes.

Every driver registers the route, whether or not Studio is installed. It has
no action-header check: ComfyUI's own middleware refuses cross-site browser
requests here as on every route, and `--enable-cors-header` opens it as it
opens POST /prompt. docs/THREAT_MODEL.md lists it under "Browser to ComfyUI
driver".
"""
from __future__ import annotations

from ..execution_events import context_available, journal


def register(app, web) -> None:
    @app.routes.get("/dgxm/execution-events")
    async def execution_events(request):
        try:
            after = int(request.query.get("after", "0"))
            producer = request.query.get("producer")
            if producer is not None and len(producer) > 128:
                raise ValueError
            page = journal.read(after, producer)
            page["context_available"] = context_available()
            return web.json_response(page)
        except ValueError:
            return web.json_response({"error": "Invalid execution-event cursor"}, status=400)
