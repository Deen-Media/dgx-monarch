"""error_utils.exception_chain: every chained exception once, bounded, never raising."""
from __future__ import annotations

from itertools import pairwise

from dgx_monarch.error_utils import exception_chain


def _linked(count):
    errors = [RuntimeError(f"link {index}") for index in range(count)]
    for nearer, further in pairwise(errors):
        nearer.__cause__ = further
    return errors


def test_the_chain_lists_nearest_first_with_the_cause_before_the_context():
    top, cause, context, deeper = (RuntimeError(name) for name in ("top", "cause", "context", "deeper"))
    top.__cause__, top.__context__ = cause, context
    context.__context__ = deeper
    assert exception_chain(top) == [top, cause, context, deeper]


def test_an_exception_reached_twice_appears_once():
    top, failure = RuntimeError("wrapper"), RuntimeError("failure")
    top.__cause__ = top.__context__ = failure  # raise ... from failure inside its except
    assert exception_chain(top) == [top, failure]


def test_a_cyclic_chain_ends():
    first, second = RuntimeError("first"), RuntimeError("second")
    first.__context__, second.__context__ = second, first
    assert exception_chain(first) == [first, second]


def test_the_walk_stops_depth_links_behind_the_top():
    errors = _linked(20)
    assert exception_chain(errors[0]) == errors[:9]
    assert exception_chain(errors[0], depth=2) == errors[:3]
    assert exception_chain(errors[0], depth=0) == errors[:1]


def test_an_opaque_exception_is_listed_but_never_looked_behind():
    class Opaque(RuntimeError):
        pass

    top, opaque, hidden = RuntimeError("top"), Opaque("endpoint"), RuntimeError("behind")
    top.__cause__, opaque.__context__ = opaque, hidden
    assert exception_chain(top, opaque=Opaque) == [top, opaque]
    assert exception_chain(top) == [top, opaque, hidden]


def test_a_link_that_raises_on_access_ends_that_branch_without_raising():
    class Hostile(RuntimeError):
        @property
        def __cause__(self):
            raise RuntimeError("no cause for you")

    hostile = Hostile("top")
    assert exception_chain(hostile) == [hostile]
