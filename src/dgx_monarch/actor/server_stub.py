"""Provide the PromptServer attributes custom nodes need during worker imports.

Workers run no ComfyUI server, but custom packs may access
``server.PromptServer.instance`` to register routes or inspect server state.
Failed preload prevents sampler registration and by-reference deserialization
of SAMPLER/GUIDER objects. Stock ComfyUI may then silently substitute Euler for
an unknown sampler name; see docs/TROUBLESHOOTING.md #11 for the res_2s failure.

The stub supports the import-time surface used by packs: routes, app,
send/send_sync/send_progress_text, client_id, and supports. Registered routes
remain inert because workers serve no requests.

Keep server and aiohttp imports local: ``ensure_comfy`` must first add the
ComfyUI checkout to sys.path and guard argv.
"""
from __future__ import annotations

from ..log import get_logger
from ..transfer_utils import failure_summary, safe_call

log = get_logger(__name__)


def ensure_prompt_server_stub() -> bool:
    """Install an inert PromptServer.instance if none exists.

    Returns True when an instance (real or stub) is available. Fail-open on
    every path: when comfy's server module cannot import, or the stub cannot
    be built or installed, it logs a warning and returns False; packs that
    register server routes then fail their preload, which logs each one.
    """
    try:
        import server as comfy_server
    except Exception as exc:
        safe_call(
            log.warning,
            "comfy server module unavailable in this worker (%s); custom node packs "
            "that register server routes will not preload",
            failure_summary(exc),
        )
        return False

    try:
        if getattr(comfy_server.PromptServer, "instance", None) is not None:
            return True

        try:
            from aiohttp import web
        except Exception as exc:  # comfy requires aiohttp; only a broken env lands here
            safe_call(
                log.warning,
                "aiohttp unavailable (%s); PromptServer stub not installed",
                failure_summary(exc),
            )
            return False

        class _HeadlessPromptServer:
            """The import-time surface of server.PromptServer, inert."""

            def __init__(self) -> None:
                self.routes = web.RouteTableDef()   # real: pack decorators must work
                self.app = web.Application()        # real: packs probe app.router.frozen
                self.supports: list[str] = []
                self.client_id = None
                self.last_node_id = None
                self.last_prompt_id = None
                self.number = 0
                self.sockets: dict = {}
                self.on_prompt_handlers: list = []
                self.loop = None
                self.prompt_queue = None

            def send_sync(self, *args, **kwargs) -> None:
                pass

            async def send(self, *args, **kwargs) -> None:
                pass

            def send_progress_text(self, *args, **kwargs) -> None:
                pass

            def add_on_prompt_handler(self, handler) -> None:
                self.on_prompt_handlers.append(handler)

            def queue_updated(self, *args, **kwargs) -> None:
                pass

        comfy_server.PromptServer.instance = _HeadlessPromptServer()
    except Exception as exc:  # fail-open: an unexpected server/aiohttp surface must not abort preload
        safe_call(
            log.warning,
            "PromptServer stub could not be installed (%s); custom node packs that "
            "register server routes will not preload",
            failure_summary(exc),
        )
        return False

    log.info("headless PromptServer stub installed; custom node packs with server routes can preload")
    return True
