"""Init node widget ordering is append-only.

ComfyUI serializes widget values by position, so inserting a widget mid-list
on the Init node silently shifts every saved workflow's values after the
insertion point. Two invariants prevent that:

1. The current INPUT_TYPES widget order must extend the frozen order by
   appending only (never insert, remove, or reorder).
2. web/js/dgx_monarch_widgets.js heals older saves through an epoch table
   (INIT_ORDERS) keyed by array length; its last entry must equal the live
   Python order, and every epoch length must be unique or the length-keyed
   lookup would be ambiguous.
"""
import re
from itertools import pairwise
from pathlib import Path

from dgx_monarch.nodes.init import DGXMonarchInit

_JS = Path(__file__).resolve().parents[1] / "web" / "js" / "dgx_monarch_widgets.js"

# The widget order at the 2026-07-09 positional-drift fix. Append new widgets
# after these, never between them, and add a row to INIT_ORDERS in the JS file
# so older saves keep healing.
FROZEN_ORDER = [
    "topology", "mode", "attention", "config_path", "gpus_per_host",
    "reserve_vram_gb", "sync_ulysses", "compile_dit", "pipeline_depth",
    "mmap_fallback", "lora_low_rss", "slab_weights", "auto_gate",
    "load_profile", "swap_verify", "uma_reserve_gb",
]


def _current_order():
    inputs = DGXMonarchInit.INPUT_TYPES()
    return [*inputs.get("required", {}), *inputs.get("optional", {})]


def _js_epochs():
    # inner rows end with "],", so the first "];" is the closing bracket
    # regardless of how the file is whitespace-formatted
    block = re.search(r"const INIT_ORDERS = \[(.*?)\]\s*;", _JS.read_text(), re.S)
    assert block, "INIT_ORDERS literal not found in dgx_monarch_widgets.js"
    epochs = [re.findall(r'"([a-z_]+)"', row)
              for row in re.findall(r"\[([^\[\]]+)\]", block.group(1))]
    assert epochs, "INIT_ORDERS parsed empty"
    return epochs


def test_widget_additions_are_append_only():
    current = _current_order()
    assert current[: len(FROZEN_ORDER)] == FROZEN_ORDER, (
        "Init widgets were inserted mid-list or reordered. ComfyUI stores "
        "widget values by POSITION: this breaks every saved workflow "
        "(issue #75). Append new widgets at the END of 'optional'."
    )


def test_js_epoch_table_last_entry_matches_python():
    assert _js_epochs()[-1] == _current_order(), (
        "web/js/dgx_monarch_widgets.js INIT_ORDERS is stale: append a new "
        "row equal to the current Python INPUT_TYPES order so workflows "
        "saved before this widget addition keep healing."
    )


def test_js_epoch_lengths_are_unique_and_increasing():
    lengths = [len(e) for e in _js_epochs()]
    assert lengths == sorted(set(lengths)), (
        f"INIT_ORDERS lengths {lengths} must be strictly increasing: the "
        "legacy heal identifies the saving layout by array length alone."
    )


def test_js_epochs_only_insert_never_reorder():
    # Each epoch's names appear in the next epoch in the same relative order:
    # a subsequence check, so a swap fails too. The one rename, disable_mmap
    # to mmap_fallback, kept its slot and type and is recorded under the new
    # name.
    epochs = _js_epochs()
    for older, newer in pairwise(epochs):
        it = iter(newer)
        assert all(name in it for name in older), (
            f"epoch {older} is not an ordered subsequence of {newer}: "
            "history rows may only INSERT names, never remove or reorder"
        )
