"""Require a major release when removing registered node keys.

Saved workflows refer to NODE_CLASS_MAPPINGS keys; removing one causes a
missing_node_type error at queue time. Startup, doctor and template lint do
not check compatibility with older workflows. test_node_contract.py pins the
key set but does not tie removals to major releases.

tests/fixtures/node_mapping_keys.json records the current major's keys. New
keys are allowed. A removal requires a major release, moving the key into the
fixture's removed map, and a CHANGELOG entry naming the removed class.

The package builds its mappings without importing ComfyUI.
"""
from __future__ import annotations

import json
from pathlib import Path

import dgx_monarch
from dgx_monarch.nodes import NODE_CLASS_MAPPINGS

BASELINE_PATH = Path(__file__).parent / "fixtures" / "node_mapping_keys.json"
BASELINE = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

_REGENERATE = (
    'rewrite tests/fixtures/node_mapping_keys.json as {"major": <package major>, '
    '"keys": sorted(NODE_CLASS_MAPPINGS)}'
)


def _package_major() -> int:
    return int(dgx_monarch.__version__.split(".")[0])


def test_baseline_tracks_the_package_major() -> None:
    assert BASELINE["major"] == _package_major(), (
        f"the node-key baseline is pinned to major {BASELINE['major']} and the package is at "
        f"major {_package_major()}: {_REGENERATE}. Refresh it in the commit that bumps the "
        "major, so the guard keeps covering the keys the new major ships."
    )


def test_registered_node_keys_survive_inside_the_major() -> None:
    missing = sorted(set(BASELINE["keys"]) - set(NODE_CLASS_MAPPINGS))
    assert not missing, (
        f"node mapping keys left major {BASELINE['major']}: {', '.join(missing)}. "
        "Every saved workflow that places one of these now fails at queue time with "
        "missing_node_type, which reads as a corrupt file. Removing a key takes a major "
        f"version bump, a deliberate baseline update ({_REGENERATE}, moving the removed "
        'key into the fixture\'s "removed" map), and a CHANGELOG entry naming the removed '
        "class. Adding keys is always fine."
    )


def test_baseline_keys_are_sorted_and_unique() -> None:
    keys = BASELINE["keys"]
    assert keys == sorted(set(keys)), (
        "the node-key baseline must stay sorted with no repeats, so a diff on it reads as "
        f"the keys that moved: {_REGENERATE}."
    )


def test_removed_keys_stay_gone_and_named_in_the_changelog() -> None:
    removed = BASELINE["removed"]
    changelog = (Path(__file__).parents[1] / "CHANGELOG.md").read_text(encoding="utf-8")
    for major, keys in removed.items():
        assert keys == sorted(set(keys)), (
            f"removed keys for major {major} must stay sorted with no repeats"
        )
        for key in keys:
            assert key not in NODE_CLASS_MAPPINGS, (
                f"{key} is listed as removed at major {major} but is still registered; "
                "a re-added key moves back into the baseline keys"
            )
            assert key in changelog, (
                f"{key} was removed at major {major} but no CHANGELOG entry names it; "
                "the entry is what an upgrading user finds when a workflow refuses"
            )
