#!/usr/bin/env python3
"""Check workflow model references against example_workflows/artifacts.toml.

The manifest has one row per model file (folder, URL, size and SHA-256) and a
file list per workflow. Input media are excluded.

The default check uses only repository files and never opens a socket. It
validates manifest rows, compares each workflow's file set, and checks canonical
rendering. Reordered or repeated list entries are not detected. --models-dir
also checks locally staged files. Missing URL, size or digest values are
reported as pending; --strict makes them failures.

    python tools/check_artifacts.py
    python tools/check_artifacts.py --models-dir ~/ComfyUI/models --hash
    python tools/check_artifacts.py --sync
    python tools/check_artifacts.py --capture --models-dir ~/ComfyUI/models
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = REPO / "example_workflows"
TESTING_DIR = REPO / "tests" / "fixtures" / "workflows" / "generated"
MANIFEST_PATH = TEMPLATE_DIR / "artifacts.toml"

# Which loader widget names an artifact, and the models subfolder that loader
# resolves the name against. Keyed on the node class a template places. The
# ComfyUI side of each folder name is pinned by the "loader model folders"
# seam in tests/canary/comfy_seam_contracts.py.
LOADER_WIDGETS: dict[str, dict[str, str]] = {
    "DGXMonarchUNETLoader": {"unet_name": "diffusion_models"},
    "DGXMonarchUncondUNETLoader": {"unet_name": "diffusion_models"},
    "DGXMonarchLoraLoader": {"lora_name": "loras"},
    "CLIPLoader": {"clip_name": "text_encoders"},
    "DualCLIPLoader": {"clip_name1": "text_encoders", "clip_name2": "text_encoders"},
    "CLIPVisionLoader": {"clip_name": "clip_vision"},
    "VAELoader": {"vae_name": "vae"},
    "LatentUpscaleModelLoader": {"model_name": "latent_upscale_models"},
}

# A DGX Monarch node carries a dgxm_widgets name to value map; a stock comfy
# node stores its widgets positionally, so the order names them.
# tests/test_artifact_manifest.py pins this to STOCK in tools/gen_templates.py,
# the home of stock widget order.
STOCK_WIDGET_ORDER: dict[str, list[str]] = {
    "CLIPLoader": ["clip_name", "type", "device"],
    "DualCLIPLoader": ["clip_name1", "clip_name2", "type", "device"],
    "CLIPVisionLoader": ["clip_name"],
    "VAELoader": ["vae_name"],
    "LatentUpscaleModelLoader": ["model_name"],
}

# Dropdown entries that are not files. comfy's VAELoader appends pixel_space
# to its own list and builds a one-tensor passthrough for it, so a template
# that picks it needs nothing staged.
BUILTIN_NAMES: dict[str, frozenset[str]] = {"vae": frozenset({"pixel_space"})}

DESTS = frozenset(
    dest for widgets in LOADER_WIDGETS.values() for dest in widgets.values())
ROW_KEYS = ("file", "dest", "url", "size", "sha256", "placeholder", "note")
_HEX = frozenset("0123456789abcdef")
_HASH_BLOCK = 1 << 22

HEADER = """\
# Model files required by the public workflows and generated test fixtures.
#
# Model files only. A template also names a demo image, video or audio clip
# for its input nodes; those are not model files and take no row here.
#
# One [[artifact]] row per distinct file. `dest` is the models subfolder the
# loader resolves the name against, `url` the vendor or repack address the
# bytes come from, `size` the file size in bytes and `sha256` the file hash.
# A row with a blank url, a zero size or a blank sha256 is pending. Write its
# url here by hand; `--capture` with `--models-dir` sets size and sha256 from
# the staged file and replaces any value already there. `placeholder = true`
# marks an example file name: the reader supplies their own file, so the row
# stays blank and is never pending.
#
# [templates] lists, per template, the rows that template names, as
# "<dest>/<file>". `python tools/check_artifacts.py --sync` rebuilds the rows
# and these lists from the graphs and keeps every value already filled in.
# To change which files a template needs, edit the template and sync; never
# edit a row's file or dest, or a list, by hand. Run the tool with no
# arguments to check this file; it never downloads anything.
"""


def template_paths() -> list[Path]:
    """Every shipped template, then every testing-only one, in name order."""
    shipped = sorted(TEMPLATE_DIR.glob("*.json"))
    testing = sorted(TESTING_DIR.glob("*.json"))
    return shipped + testing


def template_key(path: Path) -> str:
    root = TEMPLATE_DIR if path.parent == TEMPLATE_DIR else REPO
    return str(path.relative_to(root).with_suffix(""))


def named_widgets(node: dict) -> dict[str, object]:
    """The widget name to value map for one node, however it stores them."""
    named = node.get("dgxm_widgets")
    if isinstance(named, dict):
        return named
    order = STOCK_WIDGET_ORDER.get(str(node.get("type")))
    if order is None:
        return {}
    return dict(zip(order, node.get("widgets_values") or [], strict=False))


def template_refs(path: Path) -> tuple[list[str], list[str]]:
    """The artifact references one template names, and any problems found."""
    problems: list[str] = []
    refs: list[str] = []
    document = json.loads(path.read_text(encoding="utf-8"))
    for node in document.get("nodes") or []:
        widgets = LOADER_WIDGETS.get(str(node.get("type")))
        if widgets is None:
            continue
        named = named_widgets(node)
        if not named:
            problems.append(
                f"{path.name}: node {node.get('id')!r} of class "
                f"{node.get('type')!r} carries no readable widget names")
            continue
        for widget, dest in sorted(widgets.items()):
            value = named.get(widget)
            if not isinstance(value, str) or not value:
                problems.append(
                    f"{path.name}: node {node.get('id')!r} has no {widget} value")
                continue
            if value in BUILTIN_NAMES.get(dest, frozenset()):
                continue
            ref = f"{dest}/{value}"
            if ref not in refs:
                refs.append(ref)
    return sorted(refs), problems


def scan_templates() -> tuple[dict[str, list[str]], list[str]]:
    wanted: dict[str, list[str]] = {}
    problems: list[str] = []
    for path in template_paths():
        refs, found = template_refs(path)
        problems.extend(found)
        wanted[template_key(path)] = refs
    return wanted, problems


def is_pending(row: dict) -> bool:
    return not row["url"] or not row["sha256"] or int(row["size"]) <= 0


def writable(text: str) -> bool:
    """Whether the string holds no quote and no backslash.

    render() writes no escapes and the check compares its output with the
    committed file, so no row may hold either character. validate_row
    and _quote both call this, so they cannot disagree.
    """
    return '"' not in text and "\\" not in text


def validate_row(row: dict, where: str) -> list[str]:
    """Everything a row must satisfy before anything reads a byte."""
    problems = []
    unknown = sorted(set(row) - set(ROW_KEYS))
    if unknown:
        problems.append(f"{where}: unknown keys {unknown}")
    for key in ("file", "dest", "url", "size", "sha256"):
        if key not in row:
            problems.append(f"{where}: no {key}")
    if problems:
        return problems
    name = row["file"]
    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
        problems.append(f"{where}: file must be a bare filename, got {name!r}")
    elif name in {".", ".."} or name.startswith("."):
        problems.append(f"{where}: file {name!r} starts with a dot; rename it in the template and sync")
    if row["dest"] not in DESTS:
        problems.append(f"{where}: dest {row['dest']!r} is not one of {sorted(DESTS)}")
    url = row["url"]
    if not isinstance(url, str) or (url and not url.startswith("https://")):
        problems.append(f"{where}: url must be https, got {url!r}")
    size = row["size"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        problems.append(f"{where}: size must be a byte count, got {size!r}")
    digest = row["sha256"]
    if not isinstance(digest, str) or (
            digest and (len(digest) != 64 or not set(digest) <= _HEX)):
        problems.append(f"{where}: sha256 must be 64 lowercase hex, got {digest!r}")
    if row.get("placeholder") is not None:
        if row["placeholder"] is not True:
            problems.append(f"{where}: placeholder must be true or absent")
        elif url or digest or size:
            problems.append(
                f"{where}: a placeholder row names a file the reader supplies, "
                "so it carries no url, size or sha256")
    if row.get("note") is not None and not isinstance(row["note"], str):
        problems.append(f"{where}: note must be text")
    for key in ("file", "url", "note"):
        value = row.get(key)
        if isinstance(value, str) and not writable(value):
            problems.append(
                f"{where}: {key} holds a quote or a backslash; the manifest "
                "is written without escapes, so remove it")
    return problems


def load_manifest(path: Path) -> tuple[list[dict], dict[str, list[str]], list[str]]:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [], {}, [f"{path}: {exc}"]
    problems = []
    rows = document.get("artifact") or []
    if not isinstance(rows, list):
        return [], {}, [f"{path}: [[artifact]] is not a list of rows"]
    templates = document.get("templates") or {}
    if not isinstance(templates, dict):
        return [], {}, [f"{path}: [templates] is not a table"]
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            problems.append(f"artifact row {index} is not a table")
            continue
        problems.extend(validate_row(row, f"artifact row {index}"))
    return rows, templates, problems


def validate_manifest(rows: list[dict], templates: dict[str, list[str]],
                      wanted: dict[str, list[str]]) -> list[str]:
    """Internal consistency: unique sorted rows, and each side naming the other."""
    problems = []
    seen: set[str] = set()
    refs: set[str] = set()
    for row in rows:
        ref = f"{row['dest']}/{row['file']}"
        if ref in seen:
            problems.append(f"two artifact rows for {ref}")
        seen.add(ref)
        refs.add(ref)
    order = [f"{row['dest']}/{row['file']}" for row in rows]
    if order != sorted(order):
        problems.append("artifact rows are not sorted by dest then file")
    for name in sorted(set(templates) - set(wanted)):
        problems.append(f"[templates] names {name!r}, which is not a template")
    for name in sorted(set(wanted) - set(templates)):
        problems.append(f"template {name!r} has no [templates] list")
    for name in sorted(set(templates) & set(wanted)):
        listed = templates[name]
        if not isinstance(listed, list) or any(
                not isinstance(ref, str) for ref in listed):
            problems.append(f"[templates] {name!r} is not a list of references")
            continue
        for ref in sorted(set(listed) - refs):
            problems.append(f"template {name!r} names {ref!r}, which has no row")
        for ref in sorted(set(wanted[name]) - set(listed)):
            problems.append(f"template {name!r} names {ref!r} in a loader but not in its [templates] list")
        for ref in sorted(set(listed) - set(wanted[name])):
            problems.append(f"template {name!r} lists {ref!r}, which none of its loaders names")
    used = {ref for listed in templates.values() if isinstance(listed, list)
            for ref in listed}
    for ref in sorted(refs - used):
        problems.append(f"artifact row {ref} is named by no template")
    return problems


def _quote(text: str) -> str:
    if not writable(text):
        raise ValueError(f"cannot write {text!r} to TOML")
    return f'"{text}"'


def render(rows: list[dict], templates: dict[str, list[str]]) -> str:
    lines = [HEADER]
    for row in rows:
        lines.append("[[artifact]]")
        lines.append(f"file = {_quote(row['file'])}")
        lines.append(f"dest = {_quote(row['dest'])}")
        if row.get("placeholder"):
            lines.append("placeholder = true")
        if row.get("note"):
            lines.append(f"note = {_quote(row['note'])}")
        lines.append(f"url = {_quote(row['url'])}")
        lines.append(f"size = {int(row['size'])}")
        lines.append(f"sha256 = {_quote(row['sha256'])}")
        lines.append("")
    lines.append("[templates]")
    for name in sorted(templates):
        lines.append(f"{_quote(name)} = [")
        for ref in templates[name]:
            lines.append(f"    {_quote(ref)},")
        lines.append("]")
    lines.append("")
    return "\n".join(lines)


def rebuild(rows: list[dict], wanted: dict[str, list[str]]) -> list[dict]:
    """The row set the templates ask for, keeping every field already filled."""
    held = {f"{row['dest']}/{row['file']}": row for row in rows
            if isinstance(row.get("dest"), str) and isinstance(row.get("file"), str)}
    refs = sorted({ref for listed in wanted.values() for ref in listed})
    rebuilt = []
    for ref in refs:
        dest, _, name = ref.partition("/")
        row = dict(held.get(ref) or {})
        row["dest"], row["file"] = dest, name
        row.setdefault("url", "")
        row.setdefault("size", 0)
        row.setdefault("sha256", "")
        rebuilt.append({key: row[key] for key in ROW_KEYS if key in row})
    return rebuilt


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_HASH_BLOCK):
            digest.update(block)
    return digest.hexdigest()


def compare_local(rows: list[dict], models_dir: Path,
                  want_hash: bool) -> tuple[list[str], list[str]]:
    """Each row's staged file against its manifest size, and sha256 if asked."""
    report, problems = [], []
    for row in rows:
        ref = f"{row['dest']}/{row['file']}"
        path = models_dir / row["dest"] / row["file"]
        if row.get("placeholder"):
            report.append(f"placeholder {ref}")
            continue
        if not path.is_file():
            report.append(f"absent      {ref}")
            continue
        size = path.stat().st_size
        if row["size"] and size != int(row["size"]):
            problems.append(f"{ref}: on disk {size} bytes, manifest {row['size']}")
            continue
        if want_hash and row["sha256"]:
            digest = file_digest(path)
            if digest != row["sha256"]:
                problems.append(f"{ref}: on disk sha256 {digest}, manifest "
                                f"{row['sha256']}")
                continue
        state = "present" if row["size"] or row["sha256"] else "unpinned"
        report.append(f"{state:11s} {ref} ({size} bytes)")
    return report, problems


def capture(rows: list[dict], models_dir: Path) -> list[str]:
    """Fill size and sha256 from the files staged on this host."""
    filled = []
    for row in rows:
        if row.get("placeholder"):
            continue
        path = models_dir / row["dest"] / row["file"]
        if not path.is_file():
            continue
        row["size"] = path.stat().st_size
        row["sha256"] = file_digest(path)
        filled.append(f"{row['dest']}/{row['file']}")
    return filled


def report_pending(rows: list[dict]) -> list[str]:
    pending = []
    for row in rows:
        if row.get("placeholder") or not is_pending(row):
            continue
        blank = [key for key in ("url", "size", "sha256")
                 if not row[key] or row[key] == 0]
        pending.append(f"{row['dest']}/{row['file']} (no {', '.join(blank)})")
    return pending


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                        help="manifest to check or write (default: the committed one)")
    parser.add_argument("--models-dir", type=Path,
                        help="compare the files staged under this models root")
    parser.add_argument("--hash", action="store_true",
                        help="with --models-dir, also hash each staged file whose row has a sha256")
    parser.add_argument("--sync", action="store_true",
                        help="rebuild rows and lists from the templates, keeping filled values")
    parser.add_argument("--capture", action="store_true",
                        help="with --models-dir, sync, then overwrite size and sha256 from staged files")
    parser.add_argument("--pending", action="store_true",
                        help="list every row that still needs a url, a size or a sha256")
    parser.add_argument("--strict", action="store_true",
                        help="fail when any row is still pending")
    args = parser.parse_args(argv)

    if args.capture and args.models_dir is None:
        print("--capture needs --models-dir", file=sys.stderr)
        return 2
    if args.hash and args.models_dir is None:
        print("--hash needs --models-dir", file=sys.stderr)
        return 2

    wanted, problems = scan_templates()
    writing = args.sync or args.capture
    if writing and not args.manifest.exists():
        rows, templates = [], {}  # first write of a new manifest
    else:
        rows, templates, found = load_manifest(args.manifest)
        problems.extend(found)
    if problems:
        print("the templates or the artifact manifest are not valid:\n  " + "\n  ".join(problems),
              file=sys.stderr)
        return 1

    if writing:
        rows = rebuild(rows, wanted)
        if args.capture:
            filled = capture(rows, args.models_dir)
            print(f"captured size and sha256 for {len(filled)} artifacts")
        args.manifest.write_text(render(rows, wanted), encoding="utf-8")
        print(f"wrote {args.manifest} ({len(rows)} artifacts, "
              f"{len(wanted)} templates)")
        return 0

    problems.extend(validate_manifest(rows, templates, wanted))
    if not problems and render(rows, templates) != args.manifest.read_text(
            encoding="utf-8"):
        problems.append(f"{args.manifest} is not in generated form; run "
                        "tools/check_artifacts.py --sync")
    if problems:
        print("artifact manifest drift:\n  " + "\n  ".join(problems),
              file=sys.stderr)
        return 1

    if args.models_dir is not None:
        report, disagreements = compare_local(rows, args.models_dir, args.hash)
        print("\n".join(report))
        if disagreements:
            print("staged files disagree with the manifest:\n  "
                  + "\n  ".join(disagreements), file=sys.stderr)
            return 1

    pending = report_pending(rows)
    placeholders = sum(1 for row in rows if row.get("placeholder"))
    print(f"artifact manifest in sync ({len(rows)} artifacts across "
          f"{len(wanted)} templates, {placeholders} supplied by the reader)")
    if pending:
        print(f"pending: {len(pending)} rows still need a url, a size or a "
              "sha256" + ("" if args.pending else " (--pending names them)"))
        if args.pending:
            print("  " + "\n  ".join(pending))
        if args.strict:
            print("--strict: a pending row is a failure", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
