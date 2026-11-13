"""Name the box a Worker's own error came from, where the driver receives it.

A Worker words its walls with no host ("stock residency cannot load X here"):
it does not know which box the driver runs on (DESIGN.md section 5.9). The
driver does, and it knows the actor it dispatched to, except at a collective
call of more than one rank, where Monarch's wrapper carries no rank; there the
Worker marks its rank on the error (``actor/failure.mark_rank``). Each point
that receives a Worker's error (``mesh_rpc.call_all_bound``,
``mesh_lease.collect_sample``, the Fleet drain, the Fleet parity proof, and
``mesh_setup_transition.ensure_setup`` for the rank whose setup it awaits)
calls ``name``. It puts the box ahead of the Worker's sentence in ``box_names``'
words, after the tag when there is one, and removes the mark, so every later
reader (the #145 lift, the driver unload, the gate report, the consent card,
Studio) reads "On the sibling, stock residency cannot load X here: ...".

An untagged error stays Monarch's wrapper in the prompt error (#145 lifts only
a tagged one), so the wrapper's own first line names the box too: "On the
sibling, actor call dgxm_worker.load_model failed." ``message`` and
``exception_formatted`` are plain attributes ``ActorError.__init__`` sets; this
module reads them with ``getattr`` and writes only strings.
"""
from __future__ import annotations

from typing import Any

from . import box_names
from .log import get_logger
from .refusal import parse_leading_refusal_tag, read_rank

log = get_logger(__name__)


def name(exc: BaseException, handle: Any, index: int | None = None) -> None:
    """Put the box ahead of a Worker error's own words, in place; never raises.

    ``index`` is the actor the driver dispatched to (a Fleet job). Without it
    the rank is the one the Worker marked, or 0 on a fleet of one rank; with
    none, the error only loses its mark. A driver-side error, with no Worker
    inside and no mark, is left as it is.
    """
    try:
        _name(exc, handle, index)
    except Exception as err:  # Wording must never replace the error it words.
        log.debug("could not name the box of a Worker error (%r)", err)


def lead(text: str, words: str) -> str:
    """``words`` ahead of a message's first word, after its tag when it has one.

    A first word that is a capitalized ordinary word ("Actor call ...") is
    lowercased behind the words; an acronym ("CUDA error") is not. Text that
    already carries the words is returned as it is.
    """
    if not words:
        return text
    tagged = parse_leading_refusal_tag(text) is not None
    head, body = text.split("] ", 1) if tagged else ("", text)
    if body.startswith(words):
        return text
    if body[:1].isupper() and body[1:2].islower():
        body = body[:1].lower() + body[1:]
    return (f"{head}] " if tagged else "") + words + body


def _name(exc: BaseException, handle: Any, index: int | None) -> None:
    inner = getattr(exc, "exception", None)
    wrapped = isinstance(inner, BaseException)
    target = inner if isinstance(inner, BaseException) else exc
    text = _message(target)
    if text is None:
        return
    text, rank = read_rank(text)
    if not wrapped and rank is None:
        return
    world = _world(handle)
    if index is not None:
        rank = int(index)
    elif rank is None and world == 1:
        rank = 0
    words = ""
    if rank is not None:
        labels = box_names.fleet_labels(handle, world)
        if 0 <= rank < len(labels):
            words = box_names.on_box(labels[rank])
    _set(target, lead(text, words))
    if not wrapped:
        return
    message = getattr(exc, "message", None)
    if isinstance(message, str):
        exc.message = lead(message, words)  # type: ignore[attr-defined]
    formatted = getattr(exc, "exception_formatted", None)
    if isinstance(formatted, str):
        exc.exception_formatted = read_rank(formatted)[0]  # type: ignore[attr-defined]


def _message(target: BaseException) -> str | None:
    """The one message an exception prints, or None for any other shape."""
    args: tuple[Any, ...] = tuple(getattr(target, "args", ()) or ())
    message = args[0] if len(args) == 1 else None
    if not isinstance(message, str) or str(target) != message:
        return None
    return message


def _set(target: BaseException, text: str) -> None:
    if target.args != (text,):
        target.args = (text,)


def _world(handle: Any) -> int:
    try:
        return max(int(getattr(handle, "world", 1) or 1), 1)
    except (TypeError, ValueError):
        return 1
