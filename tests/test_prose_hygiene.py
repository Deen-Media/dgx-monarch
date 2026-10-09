from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _authored_text_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / item.decode() for item in completed.stdout.split(b"\0") if item]


def test_repository_authored_text_has_no_em_or_en_dash() -> None:
    forbidden = {chr(0x2013): "en dash", chr(0x2014): "em dash"}
    matches: list[str] = []
    for path in _authored_text_files():
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for char, name in forbidden.items():
                if char in line:
                    matches.append(f"{path.relative_to(ROOT)}:{number} ({name})")
    assert not matches, (
        "em or en dash found in repository-authored text: " + ", ".join(matches)
    )
