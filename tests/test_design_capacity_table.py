"""The capacity table in DESIGN.md 5.9, checked against the source.

The section says one row per rung the ladder can return, and every row names
the file and the function that charges the price. These tests fail when the
table and ``capacity_quote.RUNGS`` name different rungs, or when a row names a
module or function the source no longer defines.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from dgx_monarch.actor import capacity_quote as cq

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "docs" / "DESIGN.md"
SRC = ROOT / "src" / "dgx_monarch"

HEADER = "| rung | what it charges | file and function |"

# A backticked token in the third column: an optional package directory, a
# module, and an optional dotted name inside it. A token with no dot is a whole
# module or a bare name from a module named earlier in the same cell.
_TOKEN = re.compile(r"`([A-Za-z0-9_./]+)`")


def _rows() -> list[list[str]]:
    """The table's data rows, each split into its three cells."""
    lines = DESIGN.read_text().splitlines()
    assert HEADER in lines, (
        f"the capacity table header moved; DESIGN.md 5.9 no longer holds {HEADER!r}"
    )
    start = lines.index(HEADER) + 2      # skip the header and its separator
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    assert rows, "the capacity table has a header and no rows"
    return rows


def _top_level_names(path: Path) -> set[str]:
    """Every name the module binds at module scope, without importing it."""
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def test_the_table_carries_one_row_for_every_rung_the_ladder_returns():
    """The section's own claim: a rung cannot be added without a row here."""
    named: set[str] = set()
    for cells in _rows():
        named.update(_TOKEN.findall(cells[0]))
    assert named == cq.RUNGS, (
        "the DESIGN.md 5.9 capacity table and capacity_quote.RUNGS disagree; "
        f"only in the table {sorted(named - cq.RUNGS)}, "
        f"only in the ladder {sorted(cq.RUNGS - named)}"
    )


def test_every_function_the_table_names_still_exists():
    """A row that names a moved function is a wrong row, not a stale one."""
    failures: list[str] = []
    for cells in _rows():
        modules: list[Path] = []
        for token in _TOKEN.findall(cells[2]):
            if "." not in token:
                whole = SRC / (token + ".py")
                if whole.exists():
                    # A module named on its own, with no symbol after it.
                    modules.append(whole)
                elif not any(token in _top_level_names(seen) for seen in modules):
                    # A bare name: it belongs to a module this cell already named.
                    failures.append(
                        f"{token!r} is defined by no module this row names")
                continue
            module, _, symbol = token.rpartition(".")
            path = SRC / (module + ".py")
            if not path.exists():
                failures.append(f"{token!r} names no module under src/dgx_monarch")
                continue
            modules.append(path)
            if symbol not in _top_level_names(path):
                failures.append(f"{module} defines no {symbol!r}")
    assert not failures, (
        "the DESIGN.md 5.9 capacity table names something the source no longer "
        "has:\n" + "\n".join(failures)
    )
