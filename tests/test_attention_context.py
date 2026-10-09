from __future__ import annotations

import pytest

from dgx_monarch.actor.attention_context import injection_context


@pytest.mark.parametrize(
    ("topology", "sp", "expected"),
    [
        ({"ulysses": 2, "ring": 1, "cfg": 1}, 2, True),
        ({"ulysses": 2, "ring": 1, "cfg": 2}, 2, True),
        ({"ulysses": 1, "ring": 2, "cfg": 1}, 2, False),
        ({"ulysses": 2, "ring": 2, "cfg": 1}, 4, False),
    ],
)
def test_worker_attention_context_marks_only_pure_ulysses(
    topology, sp, expected
):
    context = injection_context(topology, sp, object())
    assert context.pure_ulysses is expected
