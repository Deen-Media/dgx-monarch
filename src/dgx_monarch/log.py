"""Logging for driver and actors.

Actor-side logs stream to the driver through Monarch's native log forwarding
and land in the normal comfy log; the host tag makes cross-host interleaving
readable (DESIGN.md §5.8).
"""
import logging
import socket
import sys

_FORMAT = "[dgx-monarch %(hosttag)s] %(levelname)s %(message)s"


class SilentInLogs:
    """Mixin for an exception that is no fault once the driver is exiting: a record
    that carries one as an object is not logged then.

    ``driver_exit.DriverExitingError`` carries it, and only ``driver_exit``
    raises that, only once the exit has begun. A new call it did not send to the
    Workers is no failure, so a handler that catches the error and logs it as an
    argument or as the record's exception (itself or as an explicit cause; a
    warning that an unload or a status poll failed) prints nothing, and neither
    does one that logs its text: a message that names it (``driver_exit.TEXT_MARKS``:
    the class name, or the sentence every exit refusal opens with) is dropped
    too. Only the message is read, never a traceback, so a different error
    logged with its traceback still prints. The filter also reads the exit flag,
    so before the exit it drops no record at all.
    """


def _reports_silent(record: logging.LogRecord) -> bool:
    """Whether the record's arguments or exception carry a ``SilentInLogs``, as itself or
    as an explicit cause (``raise ... from``), or its message names the exit's refusal.
    An error raised while one was being handled, linked only by ``__context__``, is
    another failure and is kept, and so is its traceback text."""
    args = record.args
    values = list(args.values()) if isinstance(args, dict) else list(args or ())
    if record.exc_info:
        values.append(record.exc_info[1])
    for value in values:
        for _ in range(8):  # a cause chain is short; a cycle must not spin
            if not isinstance(value, BaseException):
                break
            if isinstance(value, SilentInLogs):
                return True
            value = value.__cause__
    return _names_the_refusal(record)


def _names_the_refusal(record: logging.LogRecord) -> bool:
    """Whether the record's message, as formatted (not its traceback), names the exit's refusal."""
    marks = getattr(sys.modules.get("dgx_monarch.driver_exit"), "TEXT_MARKS", ())
    try:
        message = record.getMessage()
    except Exception:  # a record whose arguments do not format is kept, as logging keeps it
        return False
    return any(mark in message for mark in marks)


def _driver_exiting() -> bool:
    """Whether the driver's exit has begun; read through sys.modules, since driver_exit imports this module."""
    module = sys.modules.get("dgx_monarch.driver_exit")
    return bool(module is not None and module.exiting())


def _reports_no_silent(record: logging.LogRecord) -> bool:
    """The logger filter: keep every record, but one that reports a ``SilentInLogs``
    once the driver is exiting. Before the exit, nothing is dropped."""
    return not (_driver_exiting() and _reports_silent(record))


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
        logger.addFilter(_reports_no_silent)  # on the logger: before every handler, a test's too
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger._dgxm_configured = True  # type: ignore[attr-defined]
    return logger
