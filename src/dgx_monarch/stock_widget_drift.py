"""Stock widgets that a ComfyUI release this project supports still lacks.

The templates follow ComfyUI master's widget order (DESIGN.md section 6.3).
When ComfyUI inserts a widget in the middle of a stock node, an older ComfyUI
reads the template's ``widgets_values`` one slot off from there on. Each row
names such a widget, the widget it follows, and the value every shipped
template holds for it, which is its default. A row stays until every ComfyUI
release this project supports declares the widget.

A reader that maps ``widgets_values`` by a running ComfyUI's own widget order
(``/object_info``) uses ``with_newer`` only when the serialized values outrun
that order, so a graph saved on the older ComfyUI keeps reading as it did. It
then drops a missing widget that holds its default, as ComfyUI's executor drops
an input a node does not declare, and keeps any other value, so the reader's
own check reports it. tests/canary/template_widget_canary.py requires each
shipped value to be the default.

Stdlib only: Studio, the sweep converter, the template generator and the
canaries read this table.
"""
from __future__ import annotations

from typing import Any

# node class -> ((widget, the widget it follows, the value a template holds), ...)
NEWER: dict[str, tuple[tuple[str, str, Any], ...]] = {
    # ComfyUI #17463, first released in 0.44.0. On 0.43 and older the
    # frontend's positional restore puts "auto" in codec: Create Video encodes
    # the clip at once, and Save Video copies the video and encodes the audio
    # again (docs/MODELS.md). web/js/dgx_monarch_widgets.js restores Create
    # Video by name, so codec stays "none".
    "CreateVideo": (("chroma_subsampling", "color_space", "auto"),),
}


def missing(node_class: str, names: list[str]) -> dict[str, Any]:
    """Each newer widget ``names`` lacks whose anchor it has, with its template value."""
    return {widget: value for widget, after, value in NEWER.get(node_class, ())
            if widget not in names and after in names}


def with_newer(node_class: str, names: list[str]) -> list[str]:
    """``names`` with each missing newer widget put back right after its anchor."""
    order = list(names)
    for widget, after, _value in NEWER.get(node_class, ()):
        if widget not in order and after in order:
            order.insert(order.index(after) + 1, widget)
    return order


def without_defaults(inputs: dict[str, Any], absent: dict[str, Any]) -> dict[str, Any]:
    """``inputs`` less each absent newer widget that holds its template value."""
    return {name: value for name, value in inputs.items()
            if not (name in absent and value == absent[name])}
