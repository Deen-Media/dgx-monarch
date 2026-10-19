"""Read-only opt-in replay of explicitly prompt-bound runtime observations."""
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
