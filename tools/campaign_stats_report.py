"""Build campaign tables and sections from exported CSV rows.

All values come from tools/campaign_stats.py exports, so the committed pages
can be checked without private source records. Tables count primary records;
exports also retain earlier attempts.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable

Row = dict[str, str]
LEGS = ("cold", "warm", "probe", "resident", "waived")
HOSTS = ("head", "sibling")
# The memory table's counters: (counter, stat each leg reports, worse direction).
KEY_MEMORY = (("MemAvailable", "min", min), ("MemFree", "min", min), ("AnonPages", "peak", max),
              ("Cached", "peak", max), ("Shmem", "peak", max), ("Active(anon)", "peak", max),
              ("Inactive(anon)", "peak", max), ("Active(file)", "peak", max), ("Inactive(file)", "peak", max))
OTHER_MEMORY = ("MemTotal", "Buffers", "Dirty", "Writeback", "Unevictable", "Mlocked", "SReclaimable",
                "SwapTotal", "SwapFree")
VERDICTS = ("PASS", "PASS-capacity", "CHECK", "FINDING")
NCCL_LOADED_FROM = "2026-10-06"
NCCL_STAMP_NOTE = (
    f"Before {NCCL_LOADED_FROM} the sweep runner stamped the NCCL version torch was built against (2.29.7), "
    "but the hosts loaded 2.30.7. The records keep the stamped value.")
CAMPAIGNS = {
    "stable_source_2026-09": (
        "Stable-source campaign",
        "The main September campaign used a stable build, with nine earlier accepted records retained "
        "as a separate build group. The 35 selected cases that never ran have no export row. "
        "See [validation](../VALIDATION.md#campaign-summary) for coverage and remaining limitations. "
        "Three HunyuanImage BF16 local references ran on the uncapped second Spark after the first "
        "Spark refused the stock load for capacity. Their timings compare a capped pair with an uncapped GPU."),
    "sweep349_2026-09": (
        "Earlier 2026-09 sweep",
        "This sweep ran across changing builds and matrix labels before the stable-source campaign. "
        "Its records are not one directly comparable set. Later waves replaced some primary records; "
        "the export does not include every earlier snapshot. Early local runs sampled only the first "
        "Spark, so a one-host reference here reflects a monitoring difference. "
        "See [validation](../VALIDATION.md#campaign-summary) for coverage."),
    "followon_2026-09": (
        "98-record follow-on",
        "This campaign revisited 56 crashes and 35 dependent cases that had not run. It also repeated "
        "one Flux2 LoRA capacity refusal and refreshed six Flux2 local references. Six records were "
        "repeated on a later build; the groups below keep that distinction."),
}


def num(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def dist(values: Iterable[float | None]) -> dict[str, float] | None:
    data = sorted(v for v in values if v is not None)
    if not data:
        return None
    return {"n": len(data), "median": statistics.median(data), "min": data[0], "max": data[-1]}


def fmt(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    if digits == 0:
        return str(round(value))
    return f"{value:.{digits}f}"


def spread(values: Iterable[float | None], digits: int = 1) -> str:
    """Median, with the range when the values differ."""
    d = dist(values)
    if d is None:
        return "-"
    if d["min"] == d["max"]:
        return fmt(d["median"], digits)
    return f"{fmt(d['median'], digits)} ({fmt(d['min'], digits)}-{fmt(d['max'], digits)})"


def counts(values: Iterable[str]) -> str:
    tally = Counter(v for v in values if v)
    return ", ".join(f"{k} {n}" for k, n in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))) or "-"


def json_list(value: str) -> list:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except ValueError:
        return []
    return parsed if isinstance(parsed, list) else []


def primary(rows: list[Row]) -> list[Row]:
    return [r for r in rows if r.get("attempt") == "0"]


def template(row: Row) -> str:
    return row.get("cell.template", "").removeprefix("dgx-monarch-")


def kernel(row: Row) -> str:
    return row.get("cell.resolved_attention") or row.get("cell.attention") or "-"


def quant(row: Row) -> str:
    return row.get("cell.artifact_quant") or row.get("cell.quant") or "-"


def topology(row: Row) -> str:
    return row.get("derived.topology") or row.get("cell.preset") or "-"


def wall(row: Row, leg: str) -> float | None:
    if row.get(f"legs.{leg}.outcome") != "render":
        return None
    return num(row.get(f"legs.{leg}.wall_s"))


def mem(row: Row, leg: str, host: str, counter: str, stat: str) -> float | None:
    return num(row.get(f"legs.{leg}.memory.{host}.{counter}.{stat}"))


def references(rows: list[Row]) -> dict[str, Row]:
    return {r["cell_id"]: r for r in primary(rows)}


def ref_of(row: Row, refs: dict[str, Row]) -> Row | None:
    """Return the pair cell's one-GPU reference, excluding its auto-setting comparison."""
    if row.get("cell.session") != "cluster" or row.get("cell.reference.kind") != "cell":
        return None
    ref = refs.get(row.get("cell.reference.cell", ""))
    return ref if ref is not None and ref.get("cell.session") == "local" else None


def speedup(row: Row, refs: dict[str, Row]) -> float | None:
    ref = ref_of(row, refs)
    mine, theirs = wall(row, "warm"), wall(ref, "warm") if ref else None
    if mine and theirs:
        return theirs / mine
    return None


def group(rows: Iterable[Row], key: Callable[[Row], tuple]) -> list[tuple[tuple, list[Row]]]:
    out: dict[tuple, list[Row]] = defaultdict(list)
    for row in rows:
        out[key(row)].append(row)
    return sorted(out.items())


def table(header: list[str], body: list[list[str]], right: int = 1) -> list[str]:
    """A markdown table; columns from ``right`` on are right-aligned."""
    if not body:
        return ["No records.", ""]
    align = ["---" if i < right else "--:" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(cells) + " |" for cells in body]
    return [*lines, ""]


def verdict_table(rows: list[Row], attempts: Counter) -> list[str]:
    observed = sorted({r.get("observed", "") for r in rows} - {""})
    body = []
    for (name,), members in group(rows, lambda r: (template(r),)):
        v = Counter(r.get("verdict") for r in members)
        o = Counter(r.get("observed") for r in members)
        body.append([name, str(len(members)), str(attempts[name]), *[str(v[k]) for k in VERDICTS],
                     *[str(o[k]) for k in observed],
                     str(sum(1 for r in members if json_list(r.get("findings", "")))),
                     str(sum(1 for r in members if json_list(r.get("notes", ""))))])
    header = ["template", "records", "attempts", *VERDICTS, *observed, "with findings", "with notes"]
    return table(header, body)


def wall_table(rows: list[Row], refs: dict[str, Row], key: Callable[[Row], tuple], names: list[str],
               only: tuple[str, ...] = LEGS) -> list[str]:
    legs = [leg for leg in only if any(wall(r, leg) is not None for r in rows)]
    body = []
    for keys, members in group(rows, key):
        if not any(wall(r, leg) is not None for r in members for leg in legs):
            continue
        ref_rows = [ref_of(r, refs) for r in members]
        ref_walls = [wall(x, "warm") for x in ref_rows if x]
        one_host = any(x and x.get("derived.sampled_hosts") == "head" for x in ref_rows)
        rendered = sum(1 for r in members if any(wall(r, leg) is not None for leg in legs))
        body.append([*keys, f"{rendered}/{len(members)}", *[spread(wall(r, leg) for r in members)
                     for leg in legs], spread(ref_walls) + (" *" if one_host and ref_walls else ""),
                     _speedup_text([speedup(r, refs) for r in members])])
    return table([*names, "rendered", *[f"{leg} s" for leg in legs], "one-GPU ref warm s", "speedup"], body,
                 right=len(names))


def _speedup_text(values: list[float | None]) -> str:
    d = dist(values)
    if d is None:
        return "-"
    if d["min"] == d["max"]:
        return f"{d['median']:.2f}x"
    return f"{d['median']:.2f}x ({d['min']:.2f}-{d['max']:.2f})"


def fidelity_table(rows: list[Row], key: Callable[[Row], tuple], names: list[str]) -> list[str]:
    scored = [r for r in rows if r.get("fidelity.bar")]
    body = []
    for keys, members in group(scored, lambda r: (r.get("fidelity.bar", ""), *key(r))):
        nrms = [num(r.get("fidelity.nrms")) for r in members]
        max_abs = [num(r.get("fidelity.max_abs")) for r in members]
        with_score = [r for r in members if r.get("fidelity.nrms")]
        identical = sum(1 for r in with_score if r.get("fidelity.identical") == "true")
        unscored = len(members) - len(with_score)
        d_nrms, d_abs = dist(nrms), dist(max_abs)
        body.append([*keys, str(len(with_score)), fmt(d_nrms and d_nrms["median"], 4),
                     fmt(d_nrms and d_nrms["max"], 4), fmt(d_abs and d_abs["median"], 0),
                     fmt(d_abs and d_abs["max"], 0), f"{identical}/{len(with_score)}", str(unscored)])
    return table(["bar", *names, "scored", "NRMS median", "NRMS max", "max_abs median", "max_abs max", "identical",
                  "no score"], body, right=1 + len(names))


def memory_table(rows: list[Row]) -> list[str]:
    body = []
    for (name, session), members in group(rows, lambda r: (template(r), r.get("cell.session", ""))):
        for leg in LEGS:
            for host in HOSTS if session != "local" else ("head",):
                sampled = [r for r in members if mem(r, leg, host, "MemAvailable", "min") is not None]
                if not sampled:
                    continue
                cells = []
                for counter, stat, worse in KEY_MEMORY:
                    values = [v for r in sampled if (v := mem(r, leg, host, counter, stat)) is not None]
                    cells.append(f"{fmt(statistics.median(values))} ({fmt(worse(values))})" if values else "-")
                body.append([name, session, leg, host, str(len(sampled)), *cells])
    header = ["template", "session", "leg", "host", "n", *[f"{c} {s}" for c, s, _ in KEY_MEMORY]]
    return table(header, body, right=4)


def refusal_table(rows: list[Row]) -> list[str]:
    tally: Counter = Counter()
    for r in rows:
        for leg in LEGS:
            outcome = r.get(f"legs.{leg}.outcome", "")
            if outcome and outcome != "render":
                exc = r.get(f"legs.{leg}.exception_type", "").rsplit(".", 1)[-1] or "-"
                tally[(template(r), leg, outcome, r.get(f"derived.{leg}.refusal_class") or "-",
                       r.get(f"derived.{leg}.refusal_guard") or "-", exc)] += 1
    body = [[*key, str(n)] for key, n in sorted(tally.items())]
    if not body:
        return ["Every leg rendered.", ""]
    return table(["template", "leg", "outcome", "class", "guard", "exception", "legs"], body, right=6)


def ledger_table(rows: list[Row]) -> list[str]:
    body = []
    for (name,), members in group(rows, lambda r: (template(r),)):
        with_rows = [r for r in members if num(r.get("ledger.count"))]
        finals = [str(json_list(r.get("ledger.verdict", ""))[-1]) for r in with_rows if json_list(r.get("ledger.verdict", ""))]
        every = [str(v) for r in with_rows for v in json_list(r.get("ledger.verdict", "")) if v]
        diffs = [float(v) for r in with_rows for v in json_list(r.get("ledger.max_abs_latent_diff", ""))
                 if isinstance(v, (int, float))]
        body.append([name, f"{len(with_rows)}/{len(members)}", str(len(every)), counts(finals), counts(every),
                     spread(diffs, 3),
                     counts(str(v) for r in with_rows for v in json_list(r.get("ledger.cross_mode", "")) if v),
                     counts(str(v) for r in with_rows for v in json_list(r.get("ledger.inconclusive_kind", "")) if v)])
    return table(["template", "records with rows", "rows", "final verdict", "every row", "max latent diff",
                  "cross-mode", "inconclusive"], body, right=1)


def reset_table(rows: list[Row]) -> list[str]:
    body = []
    for (name,), members in group(rows, lambda r: (template(r),)):
        floor = {h: [num(r.get(f"memory_floor.readings.{h}")) for r in members] for h in HOSTS}
        body.append([name, str(len(members)),
                     *[_floor_text(floor[h]) for h in HOSTS],
                     spread([num(r.get("memory_floor.floor_gib")) for r in members]),
                     str(sum(1 for r in members if any(k.startswith("memory_floor.errors.") and v
                                                       for k, v in r.items()))),
                     str(sum(1 for r in members if r.get("reset.fleet_kept") == "true")),
                     str(sum(1 for r in members if r.get("reset.recycle"))),
                     spread([num(r.get("reset.clear_s")) for r in members]),
                     counts(r.get("reset.clear_state", "") for r in members),
                     str(sum(1 for r in members if r.get("reset.drain"))),
                     str(int(sum(num(r.get("cleared_stale_files")) or 0 for r in members)))])
    return table(["template", "records", "floor head GiB", "floor sibling GiB", "floor needed GiB",
                  "floor read errors", "fleet kept", "recycled", "clear s", "clear state", "drained",
                  "stale files cleared"], body)


def _floor_text(values: list[float | None]) -> str:
    d = dist(values)
    return "-" if d is None else f"{d['median']:.1f} (min {d['min']:.1f})"


def video_table(rows: list[Row]) -> list[str]:
    body = []
    for (name,), members in group(rows, lambda r: (template(r),)):
        frames = {leg: [num(r.get(f"legs.{leg}.frames")) for r in members if wall(r, leg)] for leg in ("cold", "warm")}
        body.append([name, str(len(members)), str(sum(1 for r in members if r.get("cell.audio") == "true")),
                     spread(frames["cold"], 0), spread(frames["warm"], 0),
                     spread([num(r.get("fidelity.frames")) for r in members], 0),
                     str(sum(1 for r in members if r.get("fidelity.audio.error"))),
                     str(sum(1 for r in members if r.get("fidelity.error"))),
                     str(sum(1 for r in members if r.get("legs.waived.outcome") == "render"))])
    return table(["template", "records", "audio cells", "cold frames", "warm frames", "scored frames",
                  "audio comparator errors", "fidelity errors", "waived renders"], body)


def family_title(family: str, campaign_title: str) -> str:
    return f"{family} ({campaign_title.lower()})"


def campaign_header(name: str, rows: list[Row]) -> list[str]:
    title, note = CAMPAIGNS.get(name, (name, ""))
    prim = primary(rows)
    dates = sorted(r.get("started", "")[:10] for r in prim if r.get("started"))
    span = f"{dates[0]} to {dates[-1]}" if dates else "undated"
    out = [f"## {title} ({span})", ""]
    if note:
        out += [note, ""]
    out += [f"{len(rows)} records: {len(prim)} primary and {len(rows) - len(prim)} attempt records. The tables "
            f"count the primary records; the export `benchmark/reports/{name}_cells.csv.gz` holds all {len(rows)}. "
            "The head ran under its 2100 MHz GPU clock cap and the sibling ran uncapped.", ""]
    pins = Counter((r.get("pin.monarch_commit", "")[:7], r.get("pin.source_digest", "") or "-") for r in prim)
    out += [f"**Build coverage.** {len(pins)} recorded build group(s), distinguished by repository "
            "revision and source manifest. The CSV retains both identifiers; groups below are ordered "
            "by record count. Results across groups are not a single-build comparison.", ""]
    out += table(["build group", "primary records"],
                 [[str(i), str(n)] for i, (_, n) in enumerate(
                     sorted(pins.items(), key=lambda kv: (-kv[1], kv[0])), 1)])
    other = []
    for label, column in (("ComfyUI", "pin.comfy_commit"), ("torch", "pin.torch"), ("NCCL", "pin.nccl"),
                          ("gate protocol", "pin.gate_protocol_version"), ("world", "pin.world")):
        values = Counter((r.get(column, "") or "-")[:12] if column == "pin.comfy_commit" else r.get(column, "") or "-"
                         for r in prim)
        other.append(f"{label} " + ", ".join(f"`{v}` ({n})" for v, n in sorted(values.items())))
    out += ["**Software and execution settings.** Values with primary-record counts: " + "; ".join(other) + ".", ""]
    # A row falls back to "9", after any date, only when the export has no `started`
    # column; an empty start time reads "" and so counts as before the date.
    if any(r.get("pin.nccl_source") == "torch-runtime" and r.get("started", "9") < NCCL_LOADED_FROM for r in prim):
        out += [NCCL_STAMP_NOTE, ""]
    observed = sorted({r.get("observed", "") for r in prim} - {""})
    body = []
    for verdict in sorted({r.get("verdict", "") for r in prim}):
        members = [r for r in prim if r.get("verdict") == verdict]
        o = Counter(r.get("observed") for r in members)
        body.append([verdict, str(len(members)), *[str(o[k]) for k in observed]])
    out += ["**Verdicts by observed outcome.**", "", *table(["verdict", "records", *observed], body)]
    return out


def other_memory(rows: list[Row]) -> list[str]:
    body = []
    for counter in OTHER_MEMORY:
        for host in HOSTS:
            lows = [mem(r, leg, host, counter, "min") for r in rows for leg in LEGS]
            highs = [mem(r, leg, host, counter, "peak") for r in rows for leg in LEGS]
            lasts = [mem(r, leg, host, counter, "last") for r in rows for leg in LEGS]
            if dist(lows) is None:
                continue
            body.append([counter, host, str(dist(lows)["n"]), fmt(dist(lows)["min"], 2),
                         fmt(dist(lasts)["median"], 2), fmt(dist(highs)["max"], 2)])
    return ["**The other meminfo counters (GiB, every leg).** The per-family tables carry the nine counters " "that move with a render; these nine stay near constant, and the export holds every value.", "", *table(["counter", "host", "legs", "lowest min", "median last", "highest peak"], body, right=2)]


def kernel_proof(rows: list[Row]) -> list[str]:
    tally = Counter((kernel(r), r.get("derived.journal_kernels") or "no kernel line") for r in rows
                    if r.get("cell.session") == "cluster")
    body = [[k, j, str(n)] for (k, j), n in sorted(tally.items())]
    return ["**Attention kernel: resolved versus the worker journal (cluster records).** A worker logs a kernel " "line only when it builds a kernel. A record with no kernel line stopped before the kernel was built, " "kept no journal, or ran on a kept fleet that already held the kernel.", "", *table(["resolved attention", "journal kernel lines", "records"], body, right=2)]


def family_section(family: str, rows: list[Row], everything: list[Row], title: str) -> list[str]:
    refs = references(everything)
    attempts = Counter(template(r) for r in everything if r.get("cell.family") == family and r.get("attempt") != "0")
    v = Counter(r.get("verdict") for r in rows)
    out = [f"### {family_title(family, title)}", "",
           f"{len(rows)} records over {len({template(r) for r in rows})} template(s): "
           + ", ".join(f"{k} {v[k]}" for k in VERDICTS if v[k]) + ".", ""]
    out += ["**Verdicts and outcomes.**", "", *verdict_table(rows, attempts)]
    out += ["**Walls (s), every kernel.**", ""]
    out += wall_table(rows, refs, lambda r: (template(r), topology(r), quant(r)), ["template", "topology", "quant"])
    out += ["**Warm elapsed time (s) by kernel, every template and quant.**", ""]
    out += wall_table(rows, refs, lambda r: (topology(r), kernel(r)), ["topology", "kernel"], ("warm",))
    out += ["**Fidelity by topology and kernel.**", ""]
    out += fidelity_table(rows, lambda r: (topology(r), kernel(r)), ["topology", "kernel"])
    out += ["**Fidelity by quant.**", "", *fidelity_table(rows, lambda r: (quant(r),), ["quant"])]
    out += ["**Memory per host and leg (GiB, median with the worst record in parentheses).**", ""]
    out += memory_table(rows)
    out += ["**Refusals and crashes (legs that did not render).**", "", *refusal_table(rows)]
    out += ["**Gate ledger.**", "", *ledger_table(rows)]
    out += ["**Resets and memory floor.**", "", *reset_table(rows)]
    if any(r.get("cell.klass") == "video" or r.get("cell.audio") == "true" for r in rows):
        out += ["**Video and audio.**", "", *video_table(rows)]
    return out


def ordered(campaigns: dict[str, list[Row]]) -> list[str]:
    """Known campaigns in their fixed order, then any other name sorted."""
    return [n for n in CAMPAIGNS if n in campaigns] + sorted(n for n in campaigns if n not in CAMPAIGNS)


def family_overview(rows: list[Row], campaign_title: str) -> list[str]:
    """All primary outcomes, without pooling incompatible timing or fidelity recipes."""
    body = []
    for (family,), members in group(rows, lambda r: (r.get("cell.family", ""),)):
        verdicts = Counter(r.get("verdict", "") for r in members)
        observed = Counter(r.get("observed", "") for r in members)
        scored = sum(num(r.get("fidelity.nrms")) is not None for r in members)
        missing = sum(bool(r.get("fidelity.bar")) and num(r.get("fidelity.nrms")) is None for r in members)
        anchor = "".join(c for c in family_title(family, campaign_title).lower() if c.isalnum() or c in " -_")
        label = f'<a id="{anchor.replace(" ", "-")}"></a>{family}'
        body.append([label, str(len(members)), *[str(verdicts[k]) for k in VERDICTS],
                     str(observed["crash"]), str(scored), str(missing)])
    return ["## Families", "",
            "Every primary record is counted below, including refusals and failures. Crashes are also "
            "FINDINGs. Scored records have a numeric comparison; that count includes failed comparisons. "
            "Missing scores count named comparisons without a number. Different recipes and comparison "
            "methods are not pooled into a speed or accuracy average.", "",
            *table(["family", "primary", *VERDICTS, "crashed", "scored", "missing score"], body)]
