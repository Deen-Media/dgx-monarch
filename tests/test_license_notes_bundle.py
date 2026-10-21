"""The compiled Studio bundle is named wherever the tree says what it ships.

LICENSE-NOTES.md is the canonical distribution-boundary inventory
(DESIGN.md section 3). The wheel ships a compiled Svelte and Vite bundle, so
the inventory, the notices file, the FAQ, DESIGN.md's provenance rule and the
pull-request checkbox must all name it, and the notice versions must match the
lockfile the bundle is built from.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LICENSE_NOTES = REPO / "LICENSE-NOTES.md"
NOTICES = REPO / "THIRD-PARTY-NOTICES.txt"
LOCKFILE = REPO / "studio" / "package-lock.json"
BUNDLE = REPO / "src" / "dgx_monarch" / "studio" / "static" / "assets"

# The packages whose code the compiled bundle carries. Everything else in the
# lockfile is a build or check tool that puts no code in the bundle.
BUNDLED = {"svelte": "Svelte", "vite": "Vite", "clsx": "clsx", "esm-env": "esm-env"}


def _flat(path: Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def _bundle_notices() -> dict[str, str]:
    """{display name: version} from the bundled-runtime notice headings."""
    text = NOTICES.read_text(encoding="utf-8")
    assert text.count("Bundled Studio browser runtime notices") == 1
    section = text.split("Bundled Studio browser runtime notices", 1)[1]
    headings = re.findall(r"^(\S+) (\d+\.\d+\.\d+)\b.*\n-{3,}$", section, re.M)
    return dict(headings)


def test_the_compiled_bundle_exists():
    assert any(BUNDLE.glob("*.js")), "the shipped Studio bundle is missing"


def test_bundle_notice_versions_match_the_lockfile():
    lock = json.loads(LOCKFILE.read_text(encoding="utf-8"))["packages"]
    notices = _bundle_notices()
    assert set(notices) == set(BUNDLED.values()), (
        f"THIRD-PARTY-NOTICES.txt names {sorted(notices)}; the bundle carries {sorted(BUNDLED.values())}"
    )
    for package, name in BUNDLED.items():
        entry = lock[f"node_modules/{package}"]
        assert notices[name] == entry["version"], (
            f"{name}: THIRD-PARTY-NOTICES.txt says {notices[name]}, "
            f"studio/package-lock.json locks {entry['version']}"
        )
        assert entry.get("license") == "MIT", f"{package} is no longer MIT; review its notice"


def test_license_notes_name_the_bundle_and_link_its_notices():
    text = _flat(LICENSE_NOTES)
    assert "The compiled Studio bundle" in text
    assert "is the one shipped third-party runtime" in text
    assert "(THIRD-PARTY-NOTICES.txt)" in text
    assert "The repository includes no vendored dependency implementations" not in text
    rows = {line.split("|")[1].strip() for line in LICENSE_NOTES.read_text().splitlines()
            if line.startswith("| ") and not line.startswith("|---")}
    assert set(BUNDLED.values()) <= rows


def test_license_notes_list_every_studio_extra_dependency():
    project = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extra = project["optional-dependencies"]["studio"]
    names = {re.split(r"[<>=!~ \[;]", requirement, maxsplit=1)[0].lower() for requirement in extra}
    rows = {line.split("|")[1].strip().lower() for line in LICENSE_NOTES.read_text().splitlines()
            if line.startswith("| ")}
    assert names, "the studio extra is empty"
    assert not sorted(names - rows), "LICENSE-NOTES.md lacks a row for a studio extra dependency"
    assert "| aiohttp | Apache-2.0 |" in LICENSE_NOTES.read_text()
    assert "| Pillow | MIT-CMU |" in LICENSE_NOTES.read_text()


def test_faq_design_and_pr_template_no_longer_deny_the_bundle():
    faq = _flat(REPO / "docs" / "FAQ.md")
    assert "This tree vendors no third-party code." not in faq
    assert "compiled Studio bundle" in faq and "THIRD-PARTY-NOTICES.txt" in faq

    design = (REPO / "docs" / "DESIGN.md").read_text(encoding="utf-8")
    rules = " ".join(design.split("### 3.1 The rules", 1)[1].split("### 3.2", 1)[0].split())
    rule_one = rules.split(" 2. ", 1)[0]
    assert "compiled Studio bundle" in rule_one and "THIRD-PARTY-NOTICES.txt" in rule_one

    template = (REPO / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")
    checkbox = next(line for line in template.splitlines() if "vendored" in line)
    assert "compiled Studio bundle" in checkbox and "THIRD-PARTY-NOTICES.txt" in checkbox
