"""Build campaign pages, the column dictionary, and dashboard summaries.

``render`` writes the index and per-campaign pages, ``render_columns`` defines
the export columns, and ``summary_json`` supplies --summary-json. All three
read export rows; tables and statistics come from campaign_stats_report.py.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

Row = dict[str, str]
MEMORY_COLUMN = re.compile(r"^legs\.(?P<leg>[^.]+)\.memory\.(?P<host>[^.]+)\.(?P<counter>[^.]+)\.(?P<stat>min|peak|last)$")
LEG_TEXT = {
    "cold": "the cell's first render", "warm": "the second render, at a moved seed",
    "probe": "the one-step fidelity leg", "resident": "the resident twin an FSDP cell is held against",
    "waived": "the cold leg run again after the runner tries to accept its waiver card",
}
COUNTER_TEXT = {
    "MemTotal": "total memory", "MemFree": "unused memory", "MemAvailable": "memory the kernel can hand out "
    "without swapping (the number that decides what fits)", "Cached": "page cache", "Buffers": "block buffers",
    "Shmem": "shared memory, including slab-resident weights", "AnonPages": "anonymous pages (process heaps "
    "and the CUDA arena on unified memory)", "Dirty": "pages waiting to be written", "Writeback": "pages being "
    "written", "Active(anon)": "recently used anonymous pages", "Inactive(anon)": "idle anonymous pages",
    "Active(file)": "recently used file pages", "Inactive(file)": "idle file pages", "Unevictable": "pages the "
    "kernel cannot evict", "Mlocked": "locked pages", "SReclaimable": "reclaimable kernel slab",
    "SwapTotal": "swap size", "SwapFree": "unused swap",
}
# (pattern, meaning); the first match wins.
MEANINGS = (
    (r"^campaign$", "Export name of the campaign."),
    (r"^record$", "Record file: `<cell>.json` is the primary record, `<cell>.attemptN.json` an earlier "
                  "attempt the runner kept for audit."),
    (r"^cell_id$", "Matrix cell id."), (r"^attempt$", "0 for the primary record, N for `attemptN`."),
    (r"^cell\.loras\.count$", "LoRAs in the cell's stack."),
    (r"^cell\.loras\.name$", "LoRA file per stack slot. Files the repository does not ship read `lora_N`."),
    (r"^cell\.loras\.", "LoRA stack field, one array entry per LoRA."),
    (r"^cell\.levers\.", "Worker setting changed by the cell."),
    (r"^cell\.unets\.", "Diffusion model file on that loader node."),
    (r"^cell\.text_tokens\.", "Prompt length in tokens (the text itself is not recorded)."),
    (r"^cell\.reference\.", "Reference specification: kind, cell, comparison criterion (`bar`), preset and probe steps."),
    (r"^cell\.label$", "The outcome the matrix predicted: render, or refuse:<class>."),
    (r"^cell\.klass$", "image or video."), (r"^cell\.session$", "cluster (both hosts) or local (one GPU)."),
    (r"^cell\.preset$", "The topology preset the cell asked for."),
    (r"^cell\.artifact_quant$", "The checkpoint file's precision."),
    (r"^cell\.(launch_)?quant$", "The checkpoint's precision as the runtime reads it from the file: `quant` for "
                                  "the auto topology table, `launch_quant` for the FSDP launch gate."),
    (r"^cell\.attention$", "The attention backend the cell asked for."),
    (r"^cell\.resolved_attention$", "The attention backend the matrix predicts the render runs."),
    (r"^cell\.waiver", "Whether the cell renders under a waiver, and the guards it waives."),
    (r"^cell\.capacity_risk", "Whether the matrix flagged the cell as a capacity risk, and why."),
    (r"^cell\.checkpoint_gib$", "Checkpoint size in GiB."),
    (r"^cell\.", "Cell configuration from the derived matrix."),
    (r"^pin\.nccl$", "The NCCL version the harness process reports; `pin.nccl_source` says where it was read. "
                     "Records before 2026-10-06 hold the version torch was built against."),
    (r"^pin\.nccl_source$", "Where the NCCL value came from: loaded-library (the harness process's libnccl "
                            "answered), torch-runtime (torch's build-time value: every record before 2026-10-06, "
                            "and the fallback after it) or harness-env (torch missing or built without NCCL, so "
                            "the nvidia-nccl-cu13 package version)."),
    (r"^pin\.", "The run's environment: commits, source digest, torch and where it was read, gate protocol, "
                "driver URL and world size."),
    (r"^started$", "Cell start time, the driver host's local clock."),
    (r"^graph_digest$", "Digest of the converted template graph, before the runner patches it for each leg."),
    (r"^legs\.[^.]+\.wall_s$", "Elapsed time in seconds, at the runner's 2 s polling resolution."),
    (r"^legs\.[^.]+\.outcome$", "What the leg did: render, refuse:<class> (refuse:untyped for an untagged "
                                 "refusal), crash, timeout, or cell-error (a harness error)."),
    (r"^legs\.[^.]+\.exception_type$", "Exception class the leg ended on, empty for a render."),
    (r"^legs\.[^.]+\.message$", "The leg's error or refusal text, scrubbed: its last 8000 characters, or its "
                                "first 2000 in records made before the runner change of 2026-09-03."),
    (r"^legs\.[^.]+\.frames$", "PNG frames the leg saved."),
    (r"^legs\.[^.]+\.memory\.[^.]+\.read_error$", "Why that host's meminfo could not be read on the leg."),
    (r"^legs\.", "Leg field as the runner wrote it."),
    (r"^reset\.", "The fleet reset before the cell: whether the fleet was kept or recycled, how long the "
                  "clear took and why."),
    (r"^memory_floor\.", "MemAvailable per host read before the cell, the runner's floor, whether the reading "
                         "cleared it, and read errors."),
    (r"^cache_bust\.", "How the execution-cache reset before that leg ended: idle, still busy or an error. The "
                       "reset also unloads every model."),
    (r"^cleared_stale_files$", "How many output files of an earlier run of the cell the runner deleted first."),
    (r"^consent\.", "The waiver card the runner tried to accept after a waivable refusal, and the result."),
    (r"^ledger\.count$", "Gate ledger rows the cell wrote."),
    (r"^ledger\.", "Gate ledger field: a JSON array with one entry per row the cell wrote, null where a row "
                   "lacks it."),
    (r"^journal\.[^.]+\.lines$", "Worker journal lines the runner kept from that host."),
    (r"^journal\.[^.]+\.text$", "Those lines, a JSON array, with the host and actor prefix removed."),
    (r"^findings$", "The runner's findings for the cell."), (r"^notes$", "The runner's notes for the cell."),
    (r"^observed$", "The cold leg's outcome, which the verdict compares with `cell.label`. It reads "
                    "graph-rejected when ComfyUI rejected the graph, blocked when the runner did not run the cell "
                    "because its reference is a crash record, and not-run when the cold leg's queue would not "
                    "clear."),
    (r"^verdict$", "PASS, PASS-capacity, CHECK or FINDING."),
    (r"^fidelity\.", "Fidelity comparison: criterion (`bar`), NRMS, max_abs (8-bit pixel levels), identical, frames, files, "
                     "and any error or skip reason."),
    (r"^rejection$", "ComfyUI's rejection of the queued graph."),
    (r"^waived_render$", "The `waived-nrms` note: measured NRMS and why no error threshold applies."),
    (r"^derived\.topology$", "Derived: the preset, with `auto:<topology>` from the gate key."),
    (r"^derived\.resolved_topology$", "Derived: the topology the gate key recorded."),
    (r"^derived\.sampled_hosts$", "Derived: hosts whose meminfo any leg sampled."),
    (r"^derived\.journal_kernels$", "Derived: attention kernels named by `USP attention kernel` lines."),
    (r"^derived\.slab", "Derived: the largest `slab residency` line (GiB, strays, stray GiB)."),
    (r"^derived\.fsdp", "Derived: the largest `FSDP capacity mode` line (module, local weights GiB)."),
    (r"^derived\.[^.]+\.refusal", "Derived: class, guard and waivable flag of the leg's `[dgxm:...]` tag."),
)


def _report():
    name = "dgxm_tools_campaign_stats_report"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("campaign_stats_report.py"))
        if spec is None or spec.loader is None:
            raise SystemExit("tools/campaign_stats_report.py is missing")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def meaning(column: str) -> str:
    for pattern, text in MEANINGS:
        if re.search(pattern, column):
            return text
    return "Record field as the runner wrote it."


def kind(values: list[str]) -> str:
    present = [v for v in values if v != ""]
    if not present:
        return "empty"
    if all(v in ("true", "false") for v in present):
        return "bool"
    if all(re.fullmatch(r"-?\d+", v) for v in present):
        return "int"
    if all(re.fullmatch(r"-?(\d+(\.\d*)?|\.\d+)(e-?\d+)?|nan|inf", v) for v in present):
        return "float"
    if all(v.startswith("[") for v in present):
        return "JSON array"
    if all(v.startswith("{") for v in present):
        return "JSON object"
    return "text"


def render_columns(campaigns: dict[str, list[Row]]) -> str:
    names = list(campaigns)
    kept = sorted({str(n) for rows in campaigns.values() for r in rows for n in _report().json_list(
        r.get("cell.loras.name", "")) if not re.fullmatch(r"lora_\d+(\.safetensors)?", str(n))})
    header = list(next(iter(campaigns.values()))[0].keys()) if campaigns else []
    present = {c: {n: sum(1 for r in rows if r.get(c)) for n, rows in campaigns.items()} for c in header}
    out = ["# Campaign export columns", "",
           "Generated by `tools/campaign_stats.py`; do not edit by hand. It defines every column of the "
           "exports in `benchmark/reports/*_cells.csv.gz`, which [CAMPAIGN_RESULTS.md](CAMPAIGN_RESULTS.md) "
           "summarizes. Every export has the same header.", "", "## Encoding", "",
           "- Each export is a UTF-8 CSV with one header row, sorted by cell id and attempt. gzip runs with "
           "a fixed timestamp for reproducible output.",
           "- An empty cell means the record has no such field, or holds null. Booleans read `true` or `false`; "
           "numbers are written exactly as the record holds them.",
           "- A list is one JSON array. Gate ledger columns hold one array entry per gate row, in order, with "
           "null where a row lacks the field.",
           "- Walls are in seconds; memory, floors and checkpoint sizes in GiB; max_abs in 8-bit pixel levels.",
           "", "## What was replaced", "",
           "- **LoRA files.** A LoRA file named in the repository's templates or `example_workflows/artifacts.toml` "
           "keeps its name. Every other LoRA file reads `lora_1`, `lora_2` and so on, numbered by first use "
           "across the campaigns of one run, in every column, the journal and messages included. Its file stem, "
           "if six characters or longer, is replaced the same way.",
           "- **Machines.** Machine names read `head` and `sibling`, learned from the records: the head is the "
           "name its own journal lines carry, the sibling the other host key.",
           "- **Paths and addresses.** Home directories read `~`, private and CGNAT addresses read `<ip>`, and "
           "anything else that matches a `tools/leak_check.py` rule reads `<redacted>`. The tool scans every "
           "output with those rules and refuses to write one that still matches.",
           "- **Prompts.** The records hold token counts and graph digests, never prompt text.", ""]
    if kept:
        out += ["LoRA files kept by name in these exports: " + ", ".join(f"`{k}`" for k in kept) + ".", ""]
    rest = [c for c in header if not MEMORY_COLUMN.match(c)]
    out += ["## Columns", "", f"Records with a value, per export: {', '.join(names)}.", "",
            "| column | type | records with a value | meaning |", "|---|---|--:|---|"]
    for column in rest:
        values = [r.get(column, "") for rows in campaigns.values() for r in rows]
        out.append(f"| `{column}` | {kind(values)} | {' / '.join(str(present[column][n]) for n in names)} "
                   f"| {meaning(column)} |")
    out += ["", "## Memory columns", "",
            "Named `legs.<leg>.memory.<host>.<counter>.<stat>`: the min, peak or last value of one "
            "/proc/meminfo counter on one host across one leg, in GiB, sampled about once a second. "
            "`records sampled` counts the records that hold all eighteen counters; records made before the "
            "runner change of 2026-09-03 hold only MemAvailable and AnonPages.", "",
            "| leg | meaning | host | records sampled |", "|---|---|---|--:|"]
    sampled: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0] * len(names))
    for index, rows in enumerate(campaigns.values()):
        for leg in LEG_TEXT:
            for host in ("head", "sibling"):
                sampled[(leg, host)][index] = sum(1 for r in rows if r.get(f"legs.{leg}.memory.{host}.MemTotal.min"))
    for (leg, host), tally in sorted(sampled.items(), key=lambda kv: (list(LEG_TEXT).index(kv[0][0]), kv[0][1])):
        out.append(f"| `{leg}` | {LEG_TEXT[leg]} | `{host}` | {' / '.join(map(str, tally))} |")
    counters = sorted({m.group("counter") for c in header if (m := MEMORY_COLUMN.match(c))},
                      key=lambda c: (list(COUNTER_TEXT).index(c) if c in COUNTER_TEXT else 99, c))
    out += ["", "| counter | meaning |", "|---|---|"]
    out += [f"| `{c}` | {COUNTER_TEXT.get(c, 'meminfo counter')} |" for c in counters]
    return "\n".join(out) + "\n"


def _r(value: float) -> float:
    return round(value, 4)


def _dist(values) -> dict | None:
    d = _report().dist(values)
    return None if d is None else {k: _r(v) if k != "n" else v for k, v in d.items()}


def summary_json(campaigns: dict[str, list[Row]]) -> str:
    """Medians and ranges of every stat group per (family, template, topology, quant, kernel)."""
    rep = _report()
    out: dict = {}
    for name, rows in campaigns.items():
        refs = rep.references(rows)
        groups = []
        key = lambda r: (r.get("cell.family", ""), rep.template(r), rep.topology(r), rep.quant(r), rep.kernel(r))  # noqa: E731
        for (family, tmpl, topo, q, k), members in rep.group(rep.primary(rows), key):
            entry: dict = {"family": family, "template": tmpl, "topology": topo, "quant": q, "kernel": k,
                           "records": len(members), "verdicts": dict(Counter(r.get("verdict", "") for r in members)),
                           "observed": dict(Counter(r.get("observed", "") for r in members))}
            entry["walls_s"] = {leg: d for leg in rep.LEGS if (d := _dist(rep.wall(r, leg) for r in members))}
            entry["ref_warm_s"] = _dist(rep.wall(x, "warm") for r in members if (x := rep.ref_of(r, refs)))
            entry["speedup"] = _dist(rep.speedup(r, refs) for r in members)
            entry["fidelity"] = {
                "bars": dict(Counter(r.get("fidelity.bar", "") for r in members if r.get("fidelity.bar"))),
                "nrms": _dist(rep.num(r.get("fidelity.nrms")) for r in members),
                "max_abs": _dist(rep.num(r.get("fidelity.max_abs")) for r in members),
                "identical": sum(1 for r in members if r.get("fidelity.identical") == "true"),
                "scored": sum(1 for r in members if r.get("fidelity.nrms"))}
            memory: dict = {}
            for leg in rep.LEGS:
                for host in rep.HOSTS:
                    stats = {f"{c}.{s}": d for c, s, _ in rep.KEY_MEMORY
                             if (d := _dist(rep.mem(r, leg, host, c, s) for r in members))}
                    if stats:
                        memory.setdefault(leg, {})[host] = stats
            entry["memory_gib"] = memory
            entry["memory_floor_gib"] = {h: d for h in rep.HOSTS
                                         if (d := _dist(rep.num(r.get(f"memory_floor.readings.{h}")) for r in members))}
            entry["reset"] = {"clear_s": _dist(rep.num(r.get("reset.clear_s")) for r in members),
                              "recycled": sum(1 for r in members if r.get("reset.recycle")),
                              "fleet_kept": sum(1 for r in members if r.get("reset.fleet_kept") == "true")}
            finals = [v[-1] for r in members if (v := rep.json_list(r.get("ledger.verdict", "")))]
            entry["ledger"] = {"final_verdicts": dict(Counter(str(v) for v in finals)), "max_latent_diff": _dist(
                float(v) for r in members for v in rep.json_list(r.get("ledger.max_abs_latent_diff", ""))
                if isinstance(v, (int, float)))}
            refusals = Counter(f"{leg}:{r.get(f'legs.{leg}.outcome')}:{r.get(f'derived.{leg}.refusal_guard') or '-'}"
                               for r in members for leg in rep.LEGS
                               if r.get(f"legs.{leg}.outcome") not in (None, "", "render"))
            entry["refusals"] = dict(refusals)
            groups.append(entry)
        out[name] = {"records": len(rows), "primary": len(rep.primary(rows)), "groups": groups}
    return json.dumps(out, sort_keys=True, separators=(",", ":")) + "\n"


READING = (
    "- **Records.** Tables count primary records once. CSV exports retain earlier attempts separately. "
    "All recorded outcomes remain in the data; these pages do not revise historical verdicts.",
    "- **Verdicts.** PASS means the recorded expectation was met, which can include an expected refusal. "
    "PASS-capacity is an expected capacity refusal, not a completed render. CHECK lacks a passing "
    "comparison; FINDING differs from the expected outcome and includes crashes.",
    "- **Comparisons.** Scored does not mean passed. The CSV records each comparison method, result and "
    "reference. One-step, full-render and resident-versus-FSDP comparisons cannot be substituted for "
    "one another. See [VALIDATION.md](VALIDATION.md) for the supported claims and remaining limitations.",
    "- **Timings and memory.** The CSV retains per-leg times and sampled memory counters, including failed "
    "legs. Timing resolution is about 2 seconds; memory samples are about one second apart and do not "
    "establish absolute peaks. Detailed grouped statistics can be generated as described below.",
    "- **Hardware.** The first Spark was capped at 2100 MHz; the second was unrestricted. Campaign notes "
    "identify references run on the second Spark. Source groups remain separate in the export metadata.",
)



def page_path(name: str) -> str:
    return f"docs/campaign-results/{name.replace('_', '-')}.md"


def render(campaigns: dict[str, list[Row]]) -> dict[str, str]:
    """The index page and one page per campaign, keyed by repository path."""
    rep = _report()
    pages: dict[str, str] = {}
    totals = []
    links = []
    for name in rep.ordered(campaigns):
        title = rep.CAMPAIGNS.get(name, (name, ""))[0]
        rows = campaigns[name]
        prim = rep.primary(rows)
        header = rep.campaign_header(name, rows)
        heading = header[0].removeprefix("## ")
        families = sorted(Counter(r.get("cell.family", "") for r in prim).items())
        page = [f"# {heading}", "", "Generated by `tools/campaign_stats.py`; do not edit by hand. "
                "[How to read the tables](../CAMPAIGN_RESULTS.md#how-to-read-the-tables) and "
                "[the column dictionary](../CAMPAIGN_COLUMNS.md) apply to every table here.", "",
                "Families: " + ", ".join(f"[{fam}](#{_anchor(rep.family_title(fam, title))}) ({n})"
                                         for fam, n in families) + ".", "", *header[2:]]
        page += rep.family_overview(prim, title)
        page += ["## Data", "",
                 f"[Download all records](../../benchmark/reports/{name}_cells.csv.gz). "
                 "The export retains recipe settings, comparison references, timing, memory, refusal "
                 "messages, gate records and earlier attempts. "
                 "[Column definitions](../CAMPAIGN_COLUMNS.md) explain the fields.", ""]
        relative = page_path(name)
        pages[relative] = "\n".join(page).rstrip("\n") + "\n"
        v = Counter(r.get("verdict") for r in prim)
        crashes = sum(1 for r in prim if r.get("observed") == "crash")
        link = f"[{heading}](campaign-results/{relative.rsplit('/', 1)[-1]})"
        totals.append([link, str(len(rows)), str(len(prim)), *[str(v[k]) for k in rep.VERDICTS], str(crashes),
                       str(len(families))])
        links.append(f"- {link}: " + ", ".join(
            f"[{fam}](campaign-results/{relative.rsplit('/', 1)[-1]}#{_anchor(rep.family_title(fam, title))}) ({n})"
            for fam, n in families))
    index = ["# Campaign results", "",
             "Generated by `tools/campaign_stats.py` from the exports in `benchmark/reports/`; do not edit by hand. "
             "The exports contain one row per retained record and one column per recorded field; "
             "[CAMPAIGN_COLUMNS.md](CAMPAIGN_COLUMNS.md) defines each column. "
             "Each campaign page summarizes coverage and outcomes by family. "
             "[VALIDATION.md](VALIDATION.md) explains the comparisons and limitations; "
             "[BENCHMARKS.md](BENCHMARKS.md) lists selected timings. See [MODELS.md](MODELS.md) for validated hardware scopes.", "",
             "Regenerate these pages from the committed exports with `python tools/campaign_stats.py`, "
             "or verify them with `python tools/campaign_stats.py --check`. No original run directories "
             "are needed.", "", "## Campaigns", "",
             "Verdicts count primary records. `crashed` counts the primary records whose observed outcome was a "
             "crash; each of those also reads FINDING.", ""]
    index += rep.table(["campaign", "records", "primary", *rep.VERDICTS, "crashed", "families"], totals)
    index += ["## Families", "", *links, "", "## How to read the tables", "", *READING,
              "", "## Detailed analysis", "",
              "All three sanitized exports are retained, including the earlier sweep and unsuccessful "
              "records. Keeping them avoids selecting only favorable runs and preserves the references "
              "needed to assess later comparisons. Historical build identifiers are data fields, not "
              "requirements for using DGX Monarch.", "",
              "Generate detailed statistics from those exports without model files or GPUs:", "",
              "```bash", "python tools/campaign_stats.py --summary-json /tmp/campaign-summary.json", "```", "",
              "The JSON groups records by family, template, topology, precision and kernel. It includes "
              "timing ranges, paired speedups, comparison counts, memory summaries, refusals and gate "
              "outcomes. Groups may still contain different builds or settings; inspect the CSV before "
              "using an aggregate as a performance claim."]
    pages["docs/CAMPAIGN_RESULTS.md"] = "\n".join(index).rstrip("\n") + "\n"
    return pages


def _anchor(heading: str) -> str:
    """GitHub's heading anchor: lower case, punctuation dropped, spaces to hyphens."""
    kept = "".join(c for c in heading.lower() if c.isalnum() or c in " -_")
    return kept.replace(" ", "-")
