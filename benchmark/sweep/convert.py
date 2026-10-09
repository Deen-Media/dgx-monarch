"""Convert UI templates to API graphs and rewrite supported probe schedules.

Widget order comes from the live driver's ``/object_info`` or, offline, from
``tools/gen_templates.py``. CI compares the generator map with the frontend
schema. Multi-type inputs require ``widgetType``; see ``_draws_a_widget``.

``STEP_WIDGETS`` defines which schedules a probe can shorten and which widget
each rewrite changes.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONTROL = frozenset({"fixed", "increment", "decrement", "randomize"})
SKIP_TYPES = frozenset({"Note", "MarkdownNote", "Reroute"})
SEED_NAMES = ("seed", "noise_seed")
WIDGET_KINDS = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN"})
# Comfy io types that draw a widget under a name of their own. A dynamic combo
# (COMFY_DYNAMICCOMBO_V3, comfy_api/latest/_io.py) draws a combo whose chosen
# key can reveal further inputs; every committed template picks a key that
# reveals none, so it holds one slot like a plain combo.
WIDGET_IO_TYPES = frozenset({"COMBO", "COMFY_DYNAMICCOMBO_V3"})

# The widget each node type keeps its step count in. A KSampler-style node
# names it `steps`. A scheduler of a family's own carries the count instead of
# the sampler, under its own first widget (Ideogram4Scheduler is
# [steps, width, height, mu, std]; LTXVScheduler is
# [steps, max_shift, base_shift, stretch, terminal]). The LTX pack writes the
# schedule out in full, so ManualSigmas holds a sigma list rather than a
# number and its step count is the number of intervals in it.
#
# The table decides two things at once: which graphs get a one-step probe leg
# (matrix.graph_facts) and what the probe leg rewrites (run.patch_graph). One
# table, so the two can never disagree and hand a probe leg a full schedule.
#
# The table is the only guard. A node holding its count under a name outside
# STEP_WIDGET_NAMES cannot be spotted at all, so a future scheduler calling it
# `num_steps` reads probe True and renders its full schedule under the probe
# leg. Add each such type here when it lands.
STEP_WIDGETS = {
    "DGXMonarchKSampler": "steps",
    "DGXMonarchKSamplerAdvanced": "steps",
    "DGXMonarchFleetKSampler": "steps",
    "DGXMonarchIdentityGate": "steps",
    "Ideogram4Scheduler": "steps",
    "LTXVScheduler": "steps",
    "ManualSigmas": "sigmas",
}
STEP_WIDGET_NAMES = frozenset(STEP_WIDGETS.values())
# A sampler that renders one slice of a shared schedule. A graph carrying
# these runs two stages off one step count, so the probe cuts the windows too.
STEP_WINDOW = ("start_at_step", "end_at_step")

WidgetNames = Callable[[str], list[str]]


def _draws_a_widget(spec) -> bool:
    """Does the frontend draw a widget for this ``/object_info`` input spec?

    The declared type answers for a plain input. A multi-type input cannot:
    comfy joins the permitted types into one comma-separated string, so
    ``LTXVEmptyLatentAudio.frame_rate`` reads ``FLOAT,INT`` since comfy
    a3572c48 and matches no single kind. Comfy sets ``widgetType`` on exactly
    the multi-type inputs that wrap a widget input (``MultiType.Input.__init__``
    in ``comfy_api/latest/_io.py``), and that is the kind the frontend draws,
    so it decides whenever the type string cannot.

    Dropping a widget here does more than lose it: the remaining names take the
    dropped one's value, so every later widget of that node is set from its
    neighbour. tests/canary/template_widget_canary.py guards both halves.
    """
    kind = spec[0] if isinstance(spec, (list, tuple)) else spec
    options = spec[1] if isinstance(spec, (list, tuple)) and len(spec) > 1 else None
    widget_type = options.get("widgetType") if isinstance(options, dict) else None
    return _is_widget_kind(kind) or _is_widget_kind(widget_type)


def _is_widget_kind(kind) -> bool:
    return isinstance(kind, list) or kind in WIDGET_IO_TYPES or kind in WIDGET_KINDS


def object_info_widget_names(inputs: dict) -> list[str]:
    """Non-connection inputs in declaration order, required then optional.

    ``inputs`` is one node's ``/object_info`` ``input`` block, which is that
    node's ``INPUT_TYPES()``.
    """
    return [name for section in ("required", "optional")
            for name, spec in (inputs.get(section) or {}).items()
            if _draws_a_widget(spec)]


def driver_widget_names(driver: str) -> WidgetNames:
    """Widget order read from a live driver's ``/object_info``."""
    cache: dict[str, list[str]] = {}

    def names(node_class: str) -> list[str]:
        if node_class not in cache:
            with urllib.request.urlopen(f"{driver}/object_info/{node_class}", timeout=30) as resp:
                info = json.load(resp)[node_class]
            cache[node_class] = object_info_widget_names(info["input"])
        return cache[node_class]

    return names


def repo_widget_names(repo: Path = REPO) -> WidgetNames:
    """Widget order read from the template generator, no driver needed.

    The tools directory is on sys.path only for the import. The generator
    installs its comfy stubs around each widget read and puts them back, so a
    caller's sys.modules and sys.path read afterwards as they did before.
    """
    tools = str(repo / "tools")
    added = tools not in sys.path
    if added:
        sys.path.insert(0, tools)
    try:
        import gen_templates
    finally:
        if added:
            sys.path.remove(tools)

    def names(node_class: str) -> list[str]:
        declared = gen_templates._widget_names(node_class)
        return [name for name in declared if name != gen_templates.CONTROL_WIDGET]

    return names


def widget_source(driver: str | None, repo: Path = REPO) -> tuple[WidgetNames, str]:
    """The widget map and where it came from. A live driver wins."""
    if driver:
        try:
            names = driver_widget_names(driver)
            names("DGXMonarchInit")
            return names, "driver"
        except OSError:
            pass
    return repo_widget_names(repo), "repo"


def convert(ui: dict, names: WidgetNames) -> dict:
    """UI workflow dict to the API graph the /prompt endpoint takes."""
    links = {link[0]: [str(link[1]), link[2]] for link in ui.get("links", [])}
    api: dict[str, dict] = {}
    for node in ui["nodes"]:
        node_class = node["type"]
        if node_class in SKIP_TYPES or node.get("mode") in (2, 4):
            continue
        inputs: dict = {}
        for slot in node.get("inputs") or []:
            if slot.get("link") is not None:
                inputs[slot["name"]] = list(links[slot["link"]])
        values = list(node.get("widgets_values") or [])
        index = 0
        for name in names(node_class):
            if name in inputs:
                # A linked V3 widget still owns its positional serialization
                # slot.  Keep the link as the API value, but consume that slot
                # before reading the next widget; otherwise CreateVideo with a
                # linked FPS turns [fps, bit_depth, color_space, codec] into
                # [fps, "", 8, "sRGB"].
                if index < len(values):
                    index += 1
                    follower = values[index] if index < len(values) else None
                    if name in SEED_NAMES and isinstance(follower, str) and follower in CONTROL:
                        index += 1
                continue
            if index >= len(values):
                break
            inputs[name] = values[index]
            index += 1
            follower = values[index] if index < len(values) else None
            if name in SEED_NAMES and isinstance(follower, str) and follower in CONTROL:
                index += 1
        api[str(node["id"])] = {"class_type": node_class, "inputs": inputs}
    return api


def _step_widget(node: dict) -> str:
    """The widget this node keeps a step count in, or "" for none.

    A widget the node reads off a link is not a widget the probe can set, so a
    connected input answers "" the same way an absent one does.
    """
    widget = STEP_WIDGETS.get(node["class_type"], "")
    value = node["inputs"].get(widget) if widget else None
    if widget == "sigmas":
        return widget if isinstance(value, str) else ""
    return widget if isinstance(value, int) and not isinstance(value, bool) else ""


def step_nodes(graph: dict) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    """Return writable step widgets, unknown types, and blocked known types.

    The first list contains ``(node id, widget)`` pairs. The second contains
    unlisted node types with a recognized step-widget name. The third contains
    known types whose step widget is missing or linked. Either unresolved list
    disables the graph's probe to avoid leaving part of its schedule at full length.
    """
    known, unknown, unsettable = [], set(), set()
    for node_id, node in sorted(graph.items(), key=lambda item: int(item[0])):
        widget = _step_widget(node)
        if widget:
            known.append((node_id, widget))
            continue
        inputs = node["inputs"]
        if node["class_type"] in STEP_WIDGETS:
            unsettable.add(node["class_type"])
        elif any(isinstance(inputs.get(name), (int, str))
                 and not isinstance(inputs.get(name), bool)
                 for name in STEP_WIDGET_NAMES):
            unknown.add(node["class_type"])
    return known, sorted(unknown), sorted(unsettable)


def probe_support(graph: dict) -> tuple[bool, str]:
    """Return whether every stage supports a probe, with a reason when it does not.

    One unsupported node disables the whole probe. The reason identifies each
    node type and whether its step count is unknown, missing, or linked.
    """
    known, unknown, unsettable = step_nodes(graph)
    reasons = []
    if unknown:
        reasons.append("the step count lives in " + ", ".join(unknown)
                       + ", which the probe rewrite does not know")
    if unsettable:
        reasons.append(", ".join(unsettable) + " reads its step count off a link, "
                       "which the probe rewrite cannot set")
    if reasons:
        return False, "; ".join(reasons)
    if not known:
        return False, "no node in this graph carries a step count the probe rewrite can set"
    return True, ""


def probe_sigmas(text: str, steps: int) -> str:
    """The same schedule cut to ``steps`` intervals, in the template's own values.

    The values travel as the strings the template wrote, never as re-formatted
    floats: the candidate and its reference have to be handed the identical
    schedule, and a round trip through float would be one more difference to
    account for. A schedule already at or under the probe length is left alone.
    """
    values = [part.strip() for part in text.split(",") if part.strip()]
    if steps < 1 or len(values) <= steps + 1:
        return text
    picks = [round(index * (len(values) - 1) / steps) for index in range(steps + 1)]
    return ", ".join(values[index] for index in picks)


def set_probe_steps(graph: dict, steps: int) -> None:
    """Rewrite every sampler stage to take ``steps`` steps.

    Staged graphs retain all stages. Set the shared schedule to ``steps`` per
    stage and recut windows in execution order, even if the template lists the
    refiner first. Allocating ``steps`` to the whole graph could empty later windows.

    ``probe_support`` rejects graphs with step counts this rewrite cannot reach.
    A staged probe takes more than ``steps`` total sampler steps; the one-step
    NRMS floor needs separate calibration for that case (see sweep.example.toml).
    """
    known = step_nodes(graph)[0]
    windowed = [node_id for node_id, widget in known
                if widget == "steps"
                and all(isinstance(graph[node_id]["inputs"].get(name), int)
                        for name in STEP_WINDOW)]
    windowed.sort(key=lambda node_id: (graph[node_id]["inputs"]["start_at_step"], int(node_id)))
    for node_id, widget in known:
        inputs = graph[node_id]["inputs"]
        if widget == "sigmas":
            inputs["sigmas"] = probe_sigmas(inputs["sigmas"], steps)
        else:
            inputs["steps"] = steps * len(windowed) if node_id in windowed else steps
    for index, node_id in enumerate(windowed):
        graph[node_id]["inputs"].update({"start_at_step": index * steps,
                                         "end_at_step": (index + 1) * steps})


def convert_dir(template_dirs: list[Path], out_dir: Path,
                names: WidgetNames) -> tuple[dict[str, Path], list[str]]:
    """Convert every template. Returns the written graphs and the ones that failed."""
    graphs: dict[str, Path] = {}
    failed: list[str] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for directory in template_dirs:
        if not directory.is_dir():
            print(f"template dir absent, skipped: {directory}")
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                api = convert(json.loads(path.read_text()), names)
            except (KeyError, OSError, ValueError) as exc:
                failed.append(f"{path.stem}: {type(exc).__name__}: {exc}")
                continue
            target = out_dir / f"{path.stem}.json"
            target.write_text(json.dumps(api, indent=1))
            graphs[path.stem] = target
    return graphs, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.sweep.convert",
        description="Convert UI-format templates to API graphs under <out_dir>/graphs.")
    parser.add_argument("--config", required=True, help="sweep TOML (see sweep.example.toml)")
    parser.add_argument("--source", choices=("auto", "driver", "repo"), default="auto",
                        help="where widget order comes from: driver (its /object_info), repo "
                             "(tools/gen_templates.py), or auto (the driver if it answers, else repo)")
    args = parser.parse_args(argv)
    from .matrix import load_config

    config = load_config(Path(args.config))
    names, source = widget_source(None if args.source == "repo" else config["driver"])
    if args.source == "driver" and source != "driver":
        parser.error(f"driver {config['driver']} did not answer /object_info")
    graphs, failed = convert_dir(config["template_dirs"], config["out_dir"] / "graphs", names)
    print(f"widget order from the {source}; wrote {len(graphs)} graphs to "
          f"{config['out_dir'] / 'graphs'}")
    for line in failed:
        print(f"not converted: {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
