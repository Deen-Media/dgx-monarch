"""Logging for driver and actors.

Actor-side logs stream to the driver through Monarch's native log forwarding
and land in the normal comfy log; the host tag makes cross-host interleaving
readable (DESIGN.md §5.8).
"""
import logging
import socket

_FORMAT = "[dgx-monarch %(hosttag)s] %(levelname)s %(message)s"


class _HostTag(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.tag = socket.gethostname()

    def filter(self, record: logging.LogRecord) -> bool:
        record.hosttag = self.tag
        return True


def get_logger(name: str = "dgx_monarch") -> logging.Logger:
    logger = logging.getLogger(name)
    if not getattr(logger, "_dgxm_configured", False):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler.addFilter(_HostTag())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger._dgxm_configured = True  # type: ignore[attr-defined]
    return logger
