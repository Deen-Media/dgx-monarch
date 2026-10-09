"""Keep artifact manifest rows, template loader values, and checker tables aligned.

Renaming a template artifact without updating example_workflows/artifacts.toml
leaves an unstaged filename; a stale manifest row describes an unused file.
These CPU-only, offline checks compare both directions without importing
ComfyUI. The "loader model folders" seam in tests/canary/comfy_seam_contracts.py
checks which model subfolder each real stock loader resolves.
"""
from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "example_workflows" / "artifacts.toml"

_SPEC = importlib.util.spec_from_file_location(
    "dgxm_tools_check_artifacts", REPO / "tools" / "check_artifacts.py"
)
assert _SPEC and _SPEC.loader
check_artifacts = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = check_artifacts
_SPEC.loader.exec_module(check_artifacts)

# Read the template node allowlist from tests/test_example_workflows.py, not a
# copy, so a loader added there and not classified here fails.
_WORKFLOW_LINT = importlib.util.spec_from_file_location(
    "dgxm_tests_example_workflows", REPO / "tests" / "test_example_workflows.py"
)
assert _WORKFLOW_LINT and _WORKFLOW_LINT.loader
example_workflows = importlib.util.module_from_spec(_WORKFLOW_LINT)
sys.modules[_WORKFLOW_LINT.name] = example_workflows
_WORKFLOW_LINT.loader.exec_module(example_workflows)

# Classes that load a file which is not a model artifact: a template names
# demo media for these and benchmark/sweep/media.py writes the sweep's own.
_INPUT_LOADERS = frozenset({"LoadAudio", "LoadImage", "LoadImageMask", "LoadVideo"})


def _manifest() -> dict:
    return tomllib.loads(MANIFEST.read_text(encoding="utf-8"))


def _rows() -> dict[str, dict]:
    return {f"{row['dest']}/{row['file']}": row for row in _manifest()["artifact"]}


def _run(tmp_path: Path, text: str, *extra: str) -> int:
    copy = tmp_path / "artifacts.toml"
    copy.write_text(text, encoding="utf-8")
    return check_artifacts.main(["--manifest", str(copy), *extra])


def test_the_committed_manifest_passes_the_offline_check(capsys):
    assert check_artifacts.main([]) == 0
    assert "in sync" in capsys.readouterr().out


def test_every_artifact_a_template_names_has_a_row():
    wanted, problems = check_artifacts.scan_templates()
    assert not problems, problems
    assert wanted, "no templates were read"
    rows = _rows()
    named = {ref for refs in wanted.values() for ref in refs}
    assert not named - set(rows), (
        "templates name artifacts the manifest has no row for: "
        f"{sorted(named - set(rows))}"
    )
    assert not set(rows) - named, (
        f"manifest rows no template names: {sorted(set(rows) - named)}"
    )


def test_every_template_carries_its_own_list():
    wanted, _problems = check_artifacts.scan_templates()
    listed = _manifest()["templates"]
    assert set(listed) == set(wanted)
    for name, refs in wanted.items():
        assert listed[name] == refs, f"{name}: list drifted from the graph"


def test_stock_widget_order_matches_the_generator():
    # tools/gen_templates.py owns stock widget order and writes the positional
    # values; tools/check_artifacts.py reads them back through its own copy of
    # that order, so the two must change together when upstream adds a widget.
    original_path = list(sys.path)
    sys.path.insert(0, str(REPO / "tools"))
    try:
        import gen_templates
    finally:
        sys.path[:] = original_path
    for node_class, order in check_artifacts.STOCK_WIDGET_ORDER.items():
        assert order == gen_templates.STOCK[node_class], node_class


def test_every_loader_a_template_may_place_is_classified():
    # A node that loads a file by name ends in Loader (artifact loaders) or
    # starts with Load (input nodes). Reading one spelling only would let the
    # other kind land with no dest, unnoticed.
    loaders = {
        name for name in example_workflows._KNOWN_NODE_TYPES
        if name.endswith("Loader") or name.startswith("Load")
    }
    assert loaders & _INPUT_LOADERS == _INPUT_LOADERS, (
        "the input loaders no longer reach this check, so its second half is "
        f"dead: {sorted(_INPUT_LOADERS - loaders)}"
    )
    unclassified = loaders - set(check_artifacts.LOADER_WIDGETS) - _INPUT_LOADERS
    assert not unclassified, (
        "loader classes a template may place with no manifest dest: "
        f"{sorted(unclassified)}; add the widget and its models subfolder to "
        "tools/check_artifacts.py in the commit that ships the template"
    )


def test_a_stock_loader_with_no_named_widget_map_is_still_read():
    # The stock nodes store widgets positionally and carry no dgxm_widgets.
    node = {"type": "DualCLIPLoader",
            "widgets_values": ["clip_l.safetensors", "t5xxl_fp16.safetensors",
                               "flux", "default"]}
    assert check_artifacts.named_widgets(node)["clip_name2"] == "t5xxl_fp16.safetensors"


def test_the_synthetic_vae_entry_takes_no_row():
    # comfy's VAELoader appends pixel_space to its own list and builds a
    # passthrough for it, so the templates that pick it stage nothing.
    assert check_artifacts.BUILTIN_NAMES["vae"] == frozenset({"pixel_space"})
    assert "vae/pixel_space" not in _rows()


def test_placeholder_rows_are_the_names_a_reader_supplies():
    placeholders = {ref for ref, row in _rows().items() if row.get("placeholder")}
    assert placeholders == {
        "loras/test_lora_placeholder.safetensors",
        "loras/your_first_lora.safetensors",
        "loras/your_second_lora.safetensors",
    }
    for ref in placeholders:
        row = _rows()[ref]
        assert row["url"] == "" and row["size"] == 0 and row["sha256"] == ""
        assert row["note"], f"{ref}: a placeholder row says who supplies the file"


def test_a_pending_row_reports_but_does_not_fail(capsys):
    assert check_artifacts.main([]) == 0
    output = capsys.readouterr().out
    assert "pending" in output
    pending = check_artifacts.report_pending(_manifest()["artifact"])
    assert pending, "the set is fully filled; drop this check and turn on --strict"


def test_strict_fails_while_any_row_is_pending():
    assert check_artifacts.main(["--strict"]) == 1


def test_sync_writes_the_committed_manifest_back(tmp_path):
    copy = tmp_path / "artifacts.toml"
    copy.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    assert check_artifacts.main(["--manifest", str(copy), "--sync"]) == 0
    assert copy.read_text(encoding="utf-8") == MANIFEST.read_text(encoding="utf-8")


@pytest.mark.parametrize(("old", "new", "says"), [
    # A rename on one side: the file a template names has no row.
    ('file = "ae.safetensors"', 'file = "ae_v2.safetensors"', "has no row"),
    # Bare filenames only: a row can never write outside its dest folder.
    ('file = "ae.safetensors"', 'file = "../ae.safetensors"', "bare filename"),
    ('file = "ae.safetensors"', 'file = "sub/ae.safetensors"', "bare filename"),
    # https only, and only a real hash.
    ('url = ""\nsize = 0\nsha256 = ""\n\n[templates]',
     'url = "http://example.test/a"\nsize = 0\nsha256 = ""\n\n[templates]',
     "url must be https"),
    ('url = ""\nsize = 0\nsha256 = ""\n\n[templates]',
     'url = ""\nsize = 0\nsha256 = "abc"\n\n[templates]',
     "64 lowercase hex"),
    ('url = ""\nsize = 0\nsha256 = ""\n\n[templates]',
     'url = ""\nsize = -1\nsha256 = ""\n\n[templates]',
     "size must be a byte count"),
    # A typo in a field name would read as pending for ever.
    ('dest = "clip_vision"', 'dest = "clip_vision"\nsha265 = ""', "unknown keys"),
    ('dest = "clip_vision"', 'dest = "clipvision"', "is not one of"),
    # render() is the only writer and the check compares its output against
    # the committed file, so a value it cannot write back is a bad row, not a
    # traceback. An operator fills urls by hand, which is where one arrives.
    ('url = ""\nsize = 0\nsha256 = ""\n\n[templates]',
     'url = "https://example.test/a\\"b"\nsize = 0\nsha256 = ""\n\n[templates]',
     "quote or a backslash"),
    # A placeholder row filled in every field, with a size that is not a
    # number, reads its size like any other row: it reports rather than
    # stopping the run.
    ('note = "the second slot of the same example stack"\n'
     'url = ""\nsize = 0\nsha256 = ""',
     'note = "the second slot of the same example stack"\n'
     f'url = "https://example.test/a"\nsize = "big"\nsha256 = "{"a1" * 32}"',
     "size must be a byte count"),
])
def test_a_malformed_row_fails_the_check(tmp_path, old, new, says, capsys):
    text = MANIFEST.read_text(encoding="utf-8")
    assert text.count(old) >= 1
    assert _run(tmp_path, text.replace(old, new, 1)) == 1
    assert says in capsys.readouterr().err


def test_a_row_that_is_not_a_table_is_reported(tmp_path, capsys):
    # Hand editing is how these rows get filled, so a mistyped file reaches the
    # checker. Every shape it can arrive in has to read back as a problem.
    assert _run(tmp_path, "artifact = [1, 2]\n[templates]\n") == 1
    assert "artifact row 0 is not a table" in capsys.readouterr().err


def _with_extra_row(file: str) -> str:
    """The committed manifest with one more row in front of [templates]."""
    row = f'[[artifact]]\nfile = "{file}"\ndest = "vae"\n' \
          'url = ""\nsize = 0\nsha256 = ""\n\n'
    text = MANIFEST.read_text(encoding="utf-8")
    anchor = "\n[templates]\n"
    assert text.count(anchor) == 1
    return text.replace(anchor, "\n" + row + "[templates]\n", 1)


def test_two_rows_for_one_file_fail(tmp_path, capsys):
    assert _run(tmp_path, _with_extra_row("ae.safetensors")) == 1
    assert "two artifact rows for vae/ae.safetensors" in capsys.readouterr().err


def test_a_row_no_template_names_fails(tmp_path, capsys):
    assert _run(tmp_path, _with_extra_row("zz_unused.safetensors")) == 1
    assert "named by no template" in capsys.readouterr().err


def test_a_placeholder_row_may_not_carry_a_source(tmp_path, capsys):
    text = MANIFEST.read_text(encoding="utf-8")
    old = 'placeholder = true\nnote = "the second slot of the same example stack"\nurl = ""'
    new = old.replace('url = ""', 'url = "https://example.test/a.safetensors"')
    assert old in text
    assert _run(tmp_path, text.replace(old, new, 1)) == 1
    assert "no url, size or sha256" in capsys.readouterr().err


def test_rows_out_of_order_fail(tmp_path, capsys):
    # Sorted rows keep the file reviewable and the diff of a new artifact one
    # block long, so an out-of-place row is drift like any other.
    text = MANIFEST.read_text(encoding="utf-8")
    moved = text.replace('file = "ae.safetensors"', 'file = "zz_ae.safetensors"', 1)
    moved = moved.replace('"vae/ae.safetensors"', '"vae/zz_ae.safetensors"')
    assert _run(tmp_path, moved) == 1
    assert "sorted by dest then file" in capsys.readouterr().err


def _chroma_manifest(*, url: str, size: int, sha256: str) -> str:
    manifest = _manifest()
    rows = manifest["artifact"]
    matches = [row for row in rows
               if row["dest"] == "diffusion_models"
               and row["file"] == "Chroma1-HD-fp8mixed.safetensors"]
    assert len(matches) == 1
    matches[0].update(url=url, size=size, sha256=sha256)
    return check_artifacts.render(rows, manifest["templates"])


def test_a_filled_row_passes_and_stops_being_pending(tmp_path):
    pending = _chroma_manifest(url="", size=0, sha256="")
    assert _run(tmp_path, pending) == 0
    pending_rows = check_artifacts.report_pending(tomllib.loads(pending)["artifact"])
    assert any("Chroma1-HD-fp8mixed.safetensors" in row for row in pending_rows)

    filled = _chroma_manifest(
        url="https://example.test/Chroma1-HD-fp8mixed.safetensors",
        size=1264219396,
        sha256="a1" * 32,
    )
    assert _run(tmp_path, filled) == 0
    still = check_artifacts.report_pending(tomllib.loads(filled)["artifact"])
    assert len(still) == len(pending_rows) - 1
    assert not any("Chroma1-HD-fp8mixed.safetensors" in row for row in still)


def test_a_staged_file_that_disagrees_is_reported(tmp_path, capsys):
    # Compare local fixture bytes without downloading model files.
    models = tmp_path / "models" / "diffusion_models"
    models.mkdir(parents=True)
    (models / "Chroma1-HD-fp8mixed.safetensors").write_bytes(b"not the real weights")
    text = _chroma_manifest(
        url="https://example.test/Chroma1-HD-fp8mixed.safetensors",
        size=999,
        sha256="a1" * 32,
    )
    code = _run(tmp_path, text, "--models-dir", str(tmp_path / "models"))
    assert code == 1
    assert "on disk 20 bytes, manifest 999" in capsys.readouterr().err


def test_a_staged_file_that_agrees_passes(tmp_path):
    models = tmp_path / "models" / "diffusion_models"
    models.mkdir(parents=True)
    body = b"not the real weights"
    (models / "Chroma1-HD-fp8mixed.safetensors").write_bytes(body)
    import hashlib
    digest = hashlib.sha256(body).hexdigest()
    text = _chroma_manifest(
        url="https://example.test/Chroma1-HD-fp8mixed.safetensors",
        size=len(body),
        sha256=digest,
    )
    assert _run(tmp_path, text, "--models-dir", str(tmp_path / "models"), "--hash") == 0
