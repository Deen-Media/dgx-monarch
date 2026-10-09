#!/usr/bin/env python3
"""Export sweep records as sanitized CSVs and generated campaign reports.

Flatten every per-cell JSON record into a row, redact private identifiers,
and write:

* one gzip CSV per campaign in ``benchmark/reports/``, with every recorded
  field, including leg timings, memory summaries, fidelity, verdicts and pins;
* ``docs/CAMPAIGN_COLUMNS.md``, defining the shared columns;
* ``docs/CAMPAIGN_RESULTS.md`` and per-campaign pages in
  ``docs/campaign-results/``, computed from the exports.

Rows and columns are sorted; gzip uses mtime 0 for reproducibility. Supply all
campaigns together so they share a header and consistent private LoRA labels:

    python tools/campaign_stats.py \
        --campaign stable_source_2026-09=PATH/TO/out/cells \
        --campaign sweep349_2026-09=PATH/TO/out/cells \
        --campaign followon_2026-09=PATH/TO/out/cells

``--check`` writes nothing and exits 1 on drift. With ``--campaign``, it checks
exports and documents against source records. Without it, it checks documents
against committed exports; no private records are needed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import io
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
REPORTS = Path("benchmark/reports")
COLUMNS_DOC = Path("docs/CAMPAIGN_COLUMNS.md")
EXPORT_SUFFIX = "_cells.csv.gz"
IDENTITY = ("campaign", "record", "cell_id", "attempt")
PRIMARY_RE = re.compile(r"^(?P<id>[0-9a-f]{12})(?:\.attempt(?P<n>\d+))?\.json$")
JOURNAL_LINE = re.compile(r"^\[dgx-monarch (?P<host>[^\]\s]+)\] (?P<rest>.*)$", re.DOTALL)
ACTOR_PREFIX = re.compile(r"\[actor=.*?\}>\] ")
SAFETENSORS_NAME = re.compile(r"[\w.+-]+\.safetensors")
# A runner message can start mid-path, so the leading slash is optional.
HOME = re.compile(r"(?<![A-Za-z0-9._-])/?(?:home|Users)/(?P<user>[A-Za-z0-9._-]+)")
REFUSAL_TAG = re.compile(r"\[dgxm:(?P<cls>[A-Z])(?: guard=(?P<guard>[^\s\]]+))?(?: waivable=(?P<w>\d))?\]")
KERNEL_LINE = re.compile(r"USP attention kernel: (\S+)")
SLAB_LINE = re.compile(r"slab residency: ([\d.]+) GiB slab, reabsorbed (\d+) strays \(([\d.]+) GiB\)")
FSDP_LINE = re.compile(r"FSDP capacity mode: sharded (\S+) across \d+ ranks, local weights ([\d.]+) GiB")


def _load_leak_check():
    spec = importlib.util.spec_from_file_location("dgxm_tools_leak_check", REPO / "tools" / "leak_check.py")
    if spec is None or spec.loader is None:
        raise SystemExit("tools/leak_check.py is missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


leak_check = _load_leak_check()


class CampaignStatsError(RuntimeError):
    """The records could not be exported safely."""


def load_records(cells: Path) -> list[tuple[str, dict[str, Any]]]:
    """Every ``<id>.json`` and ``<id>.attemptN.json`` record, sorted by name."""
    out = []
    for path in sorted(cells.iterdir()):
        if PRIMARY_RE.match(path.name):
            with path.open(encoding="utf-8") as fh:
                out.append((path.name, json.load(fh)))
    if not out:
        raise CampaignStatsError(f"no cell records in {cells.name}")
    return out


def public_lora_names(repo: Path = REPO) -> set[str]:
    """Collect shipped safetensors names to preserve them in public LoRA records."""
    names: set[str] = set()
    with (repo / "example_workflows" / "artifacts.toml").open("rb") as fh:
        manifest = tomllib.load(fh)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            names.update(SAFETENSORS_NAME.findall(node))

    walk(manifest)
    for path in sorted((repo / "example_workflows").rglob("*.json")):
        walk(json.loads(path.read_text(encoding="utf-8")))
    return names


def learn_hosts(campaigns: dict[str, list[tuple[str, dict[str, Any]]]]) -> dict[str, str]:
    """Map machine names to ``head`` and ``sibling`` from the records themselves.

    The head's name is the one its own worker journal lines carry; the sibling
    is the other key of the memory and journal maps. No name is spelled here.
    """
    head: set[str] = set()
    others: set[str] = set()
    for records in campaigns.values():
        for _name, record in records:
            for host, lines in (record.get("journal") or {}).items():
                for line in lines:
                    match = JOURNAL_LINE.match(line)
                    if match and host == "head":
                        head.add(match.group("host"))
                if host != "head":
                    others.add(host)
            for leg in (record.get("legs") or {}).values():
                others.update(h for h in (leg.get("memory") or {}) if h != "head")
            others.update(h for h in (record.get("memory_floor") or {}).get("readings", {}) if h != "head")
    if len(head) > 1 or len(others) > 1:
        raise CampaignStatsError("the records name more than one head or more than one sibling; extend learn_hosts")
    mapping = dict.fromkeys(head, "head")
    mapping.update(dict.fromkeys(others, "sibling"))
    return mapping


def lora_labels(campaigns: dict[str, list[tuple[str, dict[str, Any]]]], public: set[str]) -> dict[str, str]:
    """``lora_N`` for every LoRA the repository does not ship, by first use."""
    first: dict[str, tuple[str, str]] = {}
    for records in campaigns.values():
        for name, record in records:
            for lora in record.get("cell", {}).get("loras") or []:
                lora_name = str(lora.get("name", ""))
                key = (str(record.get("started", "")), name)
                if lora_name and lora_name not in public and (lora_name not in first or key < first[lora_name]):
                    first[lora_name] = key
    ordered = sorted(first, key=lambda n: (first[n], n))
    return {name: f"lora_{index}" for index, name in enumerate(ordered, 1)}


class Scrubber:
    """Replace private names in every string the export or the report writes."""

    def __init__(self, hosts: dict[str, str], loras: dict[str, str]):
        self.hosts = hosts
        self.loras = loras
        literal: dict[str, str] = dict(hosts)
        for name, label in loras.items():
            literal[name] = label + ".safetensors" if name.endswith(".safetensors") else label
            stem = name.removesuffix(".safetensors")
            if len(stem) >= 6:
                literal.setdefault(stem, label)
        self.literal = literal
        alternation = "|".join(re.escape(k) for k in sorted(literal, key=lambda k: (-len(k), k)))
        self.pattern = re.compile(alternation) if alternation else None
        self.users: set[str] = set()

    def _home(self, match: re.Match[str]) -> str:
        self.users.add(match.group("user"))
        return "~"

    def text(self, value: str) -> str:
        if self.pattern is not None:
            value = self.pattern.sub(lambda m: self.literal[m.group(0)], value)
        value = HOME.sub(self._home, value)
        data = value.encode("utf-8")
        for rule in leak_check.RULES:
            replacement = b"<ip>" if rule.name == "network-address" else b"<redacted>"
            data = rule.pattern.sub(replacement, data)
        return data.decode("utf-8")

    def host(self, key: str) -> str:
        return self.hosts.get(key, key)


def assert_clean(label: str, data: bytes) -> None:
    """Fail closed when a scrubbed output still matches a leak-check rule."""
    findings = leak_check.scan_bytes(label, data)
    if findings:
        rules = sorted({finding.rule for finding in findings})
        raise CampaignStatsError(f"{label}: {len(findings)} leak-check finding(s) after scrubbing: {rules}")


def scalar(value: Any, scrub: Scrubber) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return scrub.text(value)
    return json.dumps(scrub_json(value, scrub), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def scrub_json(value: Any, scrub: Scrubber) -> Any:
    if isinstance(value, str):
        return scrub.text(value)
    if isinstance(value, list):
        return [scrub_json(v, scrub) for v in value]
    if isinstance(value, dict):
        return {scrub.host(str(k)): scrub_json(v, scrub) for k, v in value.items()}
    return value


def flatten(node: Any, path: str, out: dict[str, str], scrub: Scrubber) -> None:
    """Dotted paths for dicts; a list becomes one JSON-array column."""
    if isinstance(node, dict):
        for key, value in node.items():
            flatten(value, f"{path}.{scrub.host(str(key))}" if path else scrub.host(str(key)), out, scrub)
    else:
        out[path] = scalar(node, scrub)


def _row_paths(node: Any, path: str, out: set[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            _row_paths(value, f"{path}.{key}" if path else str(key), out)
    else:
        out.add(path)


def _dig(node: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def resolved_topology(rows: list[dict[str, Any]]) -> str:
    """The topology the gate key recorded, e.g. ``uly2`` or ``cfg2+fsdp``."""
    for row in rows:
        contexts = list(row.get("blocked_contexts") or [])
        contexts += [row.get(k) for k in ("capability_context", "target_context") if row.get(k)]
        for context in contexts:
            try:
                parsed = json.loads(context)
            except (TypeError, ValueError):
                continue
            topo = parsed.get("resolved_topology") or parsed.get("worker_topology")
            if isinstance(topo, dict):
                parts = [f"{key}{topo[key]}" for key in ("cfg", "dp") if int(topo.get(key) or 1) > 1]
                parts += [f"{name}{topo[key]}" for key, name in (("ulysses", "uly"), ("ring", "ring"))
                          if int(topo.get(key) or 1) > 1]
                if topo.get("fsdp"):
                    parts.append("fsdp")
                return "+".join(parts) or "single"
    return ""


def flatten_record(campaign: str, name: str, record: dict[str, Any], scrub: Scrubber) -> dict[str, str]:
    match = PRIMARY_RE.match(name)
    if match is None:
        raise CampaignStatsError(f"unexpected record name {name}")
    row: dict[str, str] = {"campaign": campaign, "record": name, "cell_id": match.group("id"),
                           "attempt": match.group("n") or "0"}
    special = {"journal", "ledger_rows"}
    flatten({k: v for k, v in record.items() if k not in special}, "", row, scrub)
    cell = record.get("cell") or {}
    loras = cell.get("loras") or []
    row.pop("cell.loras", None)
    row["cell.loras.count"] = str(len(loras))
    for field in sorted({k for lora in loras for k in lora}):
        row[f"cell.loras.{field}"] = scalar([lora.get(field) for lora in loras], scrub)
    ledger = record.get("ledger_rows") or []
    row["ledger.count"] = str(len(ledger))
    paths: set[str] = set()
    for entry in ledger:
        _row_paths(entry, "", paths)
    for path in sorted(paths):
        row[f"ledger.{path}"] = scalar([_dig(entry, path) for entry in ledger], scrub)
    for host, lines in (record.get("journal") or {}).items():
        mapped = scrub.host(host)
        row[f"journal.{mapped}.lines"] = str(len(lines))
        kept = [ACTOR_PREFIX.sub("", JOURNAL_LINE.sub(r"\g<rest>", line), count=1) for line in lines]
        row[f"journal.{mapped}.text"] = scalar(kept, scrub)
    _derive(row, record, ledger)
    return row


def _derive(row: dict[str, str], record: dict[str, Any], ledger: list[dict[str, Any]]) -> None:
    cell = record.get("cell") or {}
    resolved = resolved_topology(ledger)
    preset = str(cell.get("preset") or "")
    row["derived.resolved_topology"] = resolved
    row["derived.topology"] = f"auto:{resolved or '?'}" if preset == "auto" else preset
    hosts: set[str] = set()
    for leg_name, leg in (record.get("legs") or {}).items():
        for host in leg.get("memory") or {}:
            hosts.add(host)
        tag = REFUSAL_TAG.search(str(leg.get("message") or ""))
        if tag:
            row[f"derived.{leg_name}.refusal_class"] = tag.group("cls")
            row[f"derived.{leg_name}.refusal_guard"] = tag.group("guard") or ""
            row[f"derived.{leg_name}.refusal_waivable"] = tag.group("w") or ""
    row["derived.sampled_hosts"] = "+".join(sorted("head" if h == "head" else "sibling" for h in hosts))
    kernels: set[str] = set()
    slab: list[tuple[float, int, float]] = []
    fsdp: list[tuple[float, str]] = []
    for lines in (record.get("journal") or {}).values():
        for line in lines:
            if m := KERNEL_LINE.search(line):
                kernels.add(m.group(1))
            if m := SLAB_LINE.search(line):
                slab.append((float(m.group(1)), int(m.group(2)), float(m.group(3))))
            if m := FSDP_LINE.search(line):
                fsdp.append((float(m.group(2)), m.group(1)))
    row["derived.journal_kernels"] = "+".join(sorted(kernels))
    if slab:
        top = max(slab)
        row["derived.slab_gib"], row["derived.slab_strays"], row["derived.slab_stray_gib"] = (
            repr(top[0]), str(top[1]), repr(top[2]))
    if fsdp:
        top_f = max(fsdp)
        row["derived.fsdp_local_gib"], row["derived.fsdp_module"] = repr(top_f[0]), top_f[1]


def header_for(rows: list[dict[str, str]]) -> list[str]:
    rest = sorted({key for row in rows for key in row} - set(IDENTITY))
    return [*IDENTITY, *rest]


def csv_bytes(rows: list[dict[str, str]], header: list[str]) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    for row in sorted(rows, key=lambda r: (r["cell_id"], int(r["attempt"]), r["record"])):
        writer.writerow([row.get(column, "") for column in header])
    return buf.getvalue().encode("utf-8")


def gzip_bytes(data: bytes) -> bytes:
    """gzip at mtime 0, at the highest level whose bytes pass the leak checker.

    tools/leak_check.py reads every tracked file as bytes, compressed ones
    included, and compressed bytes can spell an address-shaped run by chance.
    The level search is deterministic, so the same input gives the same file.
    """
    for level in range(9, 0, -1):
        out = gzip.compress(data, compresslevel=level, mtime=0)
        if not leak_check.scan_bytes("export.csv.gz", out):
            return out
    raise CampaignStatsError("every gzip level writes bytes that match a leak-check rule; change the export")


def read_export(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def build_exports(sources: dict[str, Path], repo: Path = REPO) -> tuple[dict[str, bytes], dict[str, list[dict[str, str]]]]:
    """Scrub every record into a row; return one CSV per campaign over a shared header, and the rows."""
    campaigns = {name: load_records(path) for name, path in sources.items()}
    scrub = Scrubber(learn_hosts(campaigns), lora_labels(campaigns, public_lora_names(repo)))
    rows = {name: [flatten_record(name, file, record, scrub) for file, record in records]
            for name, records in campaigns.items()}
    header = header_for([row for campaign_rows in rows.values() for row in campaign_rows])
    exports = {}
    for name, campaign_rows in rows.items():
        data = csv_bytes(campaign_rows, header)
        assert_clean(f"{name}{EXPORT_SUFFIX}", data)
        # Every account name a home path carried must be gone. A message cut
        # mid-path leaves a short account stub, which the scrub already replaced
        # and which a substring test would find in unrelated text.
        accounts = {user for user in scrub.users if len(user) >= 4}
        survivors = [kind for kind, names in (("account", accounts), ("machine", scrub.hosts), ("LoRA", scrub.loras))
                     if any(n.encode("utf-8") in data for n in names)]
        if survivors:
            raise CampaignStatsError(f"{name}{EXPORT_SUFFIX}: a scrubbed {' and '.join(survivors)} name survived")
        exports[name] = data
    return exports, rows


def _parse_sources(values: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", name):
            raise SystemExit(f"--campaign wants NAME=CELLS_DIR, got {value!r}")
        sources[name] = Path(path).expanduser()
    return sources


def committed_rows(root: Path) -> dict[str, list[dict[str, str]]]:
    rows = {}
    for path in sorted((root / REPORTS).glob(f"*{EXPORT_SUFFIX}")):
        rows[path.name.removesuffix(EXPORT_SUFFIX)] = read_export(path)
    if not rows:
        raise CampaignStatsError(f"no {EXPORT_SUFFIX} exports under {REPORTS}")
    return rows


def render_documents(rows: dict[str, list[dict[str, str]]]) -> dict[Path, bytes]:
    report = _load_sibling("campaign_stats_report")
    index = _load_sibling("campaign_stats_index")
    rows = {name: rows[name] for name in report.ordered(rows)}
    docs = {Path(path): text.encode("utf-8") for path, text in index.render(rows).items()}
    docs[COLUMNS_DOC] = index.render_columns(rows).encode("utf-8")
    for path, data in docs.items():
        assert_clean(path.as_posix(), data)
    return docs


def _load_sibling(name: str):
    if f"dgxm_tools_{name}" in sys.modules:
        return sys.modules[f"dgxm_tools_{name}"]
    spec = importlib.util.spec_from_file_location(f"dgxm_tools_{name}", Path(__file__).with_name(f"{name}.py"))
    if spec is None or spec.loader is None:
        raise SystemExit(f"tools/{name}.py is missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(sources: dict[str, Path], root: Path, check: bool, summary: Path | None) -> list[str]:
    """Write (or compare) every output; return the drifted paths."""
    outputs: dict[Path, bytes] = {}
    if sources:
        exports, _rows = build_exports(sources)
        for name, data in exports.items():
            outputs[REPORTS / f"{name}{EXPORT_SUFFIX}"] = gzip_bytes(data)
        rows = {name: list(csv.DictReader(io.StringIO(data.decode("utf-8")))) for name, data in exports.items()}
    else:
        rows = committed_rows(root)
    outputs.update(render_documents(rows))
    drift = []
    for relative, data in sorted(outputs.items()):
        target = root / relative
        same = target.exists() and target.read_bytes() == data
        if check:
            if not same:
                drift.append(relative.as_posix())
        elif not same:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    if summary is not None and not check:
        rows = {name: rows[name] for name in _load_sibling("campaign_stats_report").ordered(rows)}
        data = _load_sibling("campaign_stats_index").summary_json(rows).encode("utf-8")
        assert_clean(summary.name, data)
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_bytes(data)
    return drift


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--campaign", action="append", default=[], metavar="NAME=CELLS_DIR",
                        help="a campaign name and its cells directory; repeat it for every campaign in one run")
    parser.add_argument("--root", type=Path, default=REPO, help="root to write or check (default: this repository)")
    parser.add_argument("--check", action="store_true", help="write nothing; exit 1 when an output would change")
    parser.add_argument("--summary-json", type=Path, help="also write a per-group JSON summary (not with --check)")
    args = parser.parse_args(argv)
    try:
        drift = run(_parse_sources(args.campaign), args.root, args.check, args.summary_json)
    except CampaignStatsError as exc:
        print(f"campaign_stats: {exc}", file=sys.stderr)
        return 2
    if drift:
        print("campaign_stats: out of date: " + ", ".join(drift), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
