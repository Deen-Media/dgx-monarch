"""Hardware status and runtime warning policy must not drift.

This module pins each support-matrix row's status, the matrix marks, the
review footers, the RDMA hold on operator surfaces and corrected doc wording.
The runtime warning exemption is stricter than a row's HW status, because
header sniffing cannot bind the graph, artifact, LoRA, quantization, and
topology scope of a hardware comparison. No current family qualifies for a family-wide exemption.
"""

import ast
import json
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "src", "dgx_monarch", "nodes", "common.py")
MODELS = os.path.join(HERE, "..", "docs", "MODELS.md")
VALIDATION = os.path.join(HERE, "..", "docs", "VALIDATION.md")
DESIGN = os.path.join(HERE, "..", "docs", "DESIGN.md")
TROUBLESHOOTING = os.path.join(HERE, "..", "docs", "TROUBLESHOOTING.md")
TRUST = os.path.join(HERE, "..", "docs", "TRUST.md")
TUI = os.path.join(HERE, "..", "docs", "TUI.md")
INSTALL = os.path.join(HERE, "..", "docs", "INSTALL.md")
THREAT_MODEL = os.path.join(HERE, "..", "docs", "THREAT_MODEL.md")
SECURITY = os.path.join(HERE, "..", "SECURITY.md")
README = os.path.join(HERE, "..", "README.md")
CLUSTER = os.path.join(HERE, "..", "docs", "CLUSTER.md")
CONCEPTS = os.path.join(HERE, "..", "docs", "CONCEPTS.md")
ADAPTERS = os.path.join(HERE, "..", "docs", "ADAPTERS.md")
CHANGELOG = os.path.join(HERE, "..", "CHANGELOG.md")
BUG_REPORT = os.path.join(
    HERE, "..", ".github", "ISSUE_TEMPLATE", "bug-report.yml"
)

_RowKey = tuple[str, str]
_SECTION_HEADINGS = {
    "### Image models": "image",
    "### Video models": "video",
}
_SUPPORT_HEADER = [
    "Model",
    "single",
    "cfg2",
    "Ulysses",
    "ring",
    "FSDP",
    "Status",
    "Notes",
]

# Complete row -> (runtime family, evidence status) contract. Section and full
# model-cell text are identity: a rename, move, addition, deletion, promotion, or
# demotion requires an evidence review.
_SUPPORT_ROW_CONTRACT: dict[_RowKey, tuple[str, str]] = {
    # image models
    ("image", "Krea2 (RAW+Turbo)"): ("krea2", "HW"),
    ("image", "Chroma"): ("chroma", "HW"),
    ("image", "Radiance"): ("chroma", "HW"),
    ("image", "Ideogram4"): ("ideogram4", "HW"),
    ("image", "Flux 1 Dev"): ("flux", "HW"),
    ("image", "Flux 1 Schnell"): ("flux", "HW"),
    ("image", "Flux2"): ("flux2", "HW"),
    ("image", "LongCat-Image"): ("longcat", "HW"),
    ("image", "HunyuanImage 2.1 (+refiner)"): ("hunyuan", "HW"),
    ("image", "Qwen-Image"): ("qwen_image", "HW"),
    ("image", "Mage-Flow (T2I/Edit, quality + Turbo)"): ("mage_flow", "HW"),
    ("image", "Qwen Image 2.1"): ("qwen_image21", "HW"),
    ("image", "Ernie-Image"): ("ernie", "HW"),
    ("image", "Z-Image latent"): ("zimage", "HW"),
    ("image", "Z-Image DCT PixelSpace"): ("zimage", "HW"),
    ("image", "Lens"): ("lens", "HW"),
    ("image", "Omnigen2"): ("omnigen2", "HW"),
    ("image", "Anima"): ("anima", "HW"),
    ("image", "Boogu"): ("boogu", "HW"),
    ("image", "PixelDiT / PiD"): ("pixeldit_comfy", "HW"),
    ("image", "Kandinsky5-Image"): ("kandinsky5", "HW"),
    # video models
    ("video", "LTX 2.3 (t2v)"): ("ltx", "HW"),
    ("video", "LTX 2.3 (i2v/AV packed)"): ("ltx", "HW"),
    ("video", "LTX 2.5 (t2v/AV packed)"): ("ltx", "HW"),
    ("video", "Wan 2.1 (t2v)"): ("wan", "HW"),
    ("video", "Wan 2.2 (t2v, high/low)"): ("wan", "HW"),
    ("video", "Wan 2.2 (i2v)"): ("wan", "HW"),
    ("video", "Wan 2.1 (i2v)"): ("wan", "HW"),
    ("video", "Wan FlowRVS"): ("wan", "HW"),
    ("video", "Wan Bernini-R"): ("wan", "HW"),
    ("video", "Wan SCAIL Preview"): ("wan_scail", "HW"),
    ("video", "Wan SCAIL2"): ("wan_scail", "HW"),
    ("video", "WanDancer"): ("wan_dancer", "HW"),
    ("video", "Wan-Animate 2"): ("wan_animate2", "HW"),
    ("video", "HunyuanVideo 1.5 (+SR)"): ("hunyuan", "HW"),
    ("video", "CogVideoX 1.5 (t2v)"): ("cogvideo", "HW"),
    ("video", "CogVideoX 1.5 (i2v)"): ("cogvideo", "HW"),
    ("video", "CogVideoX 1.5 (inpaint)"): ("cogvideo", "impl"),
    ("video", "Kandinsky5-Video Lite"): ("kandinsky5", "HW"),
    ("video", "Kandinsky5-Video Pro"): ("kandinsky5", "HW"),
    ("video", "MiniMax H3 (t2va/fl2va/ref2va AV)"): ("minimax_h3", "HW"),
}

# Detailed declarations must agree with the matrix and link to their own
# recorded comparison results. Measurement dates are not status declarations.
_REQUIRED_DETAIL_EVIDENCE: dict[
    str, tuple[str | None, str | None, tuple[str, ...]]
] = {
    "LTX 2.5 (t2v/AV packed)": (
        "VALIDATION.md",
        '<a id="evidence-ltx-2-5-t2v-av-packed"></a>',
        ("stock one GPU", "0.063/0.018", "0.044/0.024"),
    ),
    "Wan 2.1 (t2v)": (None, None, ()),
    "Wan 2.2 (t2v, high/low)": (None, None, ()),
    "Wan 2.2 (i2v)": (
        "VALIDATION.md",
        '<a id="evidence-wan-2-2-i2v"></a>',
        ("four-step", "640 square", "0.000"),
    ),
    "Wan 2.1 (i2v)": (
        "VALIDATION.md",
        '<a id="evidence-wan-2-1-i2v"></a>',
        ("81-frame", "matched", "local decoded one-step"),
    ),
    "Wan FlowRVS": (
        "VALIDATION.md",
        '<a id="evidence-wan-flowrvs"></a>',
        ("BF16", "17 decoded one-step frames", "matched"),
    ),
    "Wan Bernini-R": (
        "VALIDATION.md",
        '<a id="evidence-wan-bernini-r"></a>',
        ("FP8-scaled", "81-frame", "matched"),
    ),
    "Wan SCAIL Preview": (
        "VALIDATION.md",
        '<a id="evidence-wan-scail-preview"></a>',
        ("FP16", "256x256", "Ulysses"),
    ),
    "Wan SCAIL2": (
        "VALIDATION.md",
        '<a id="evidence-wan-scail2"></a>',
        ("81 decoded one-step frames", "matched", "HELD"),
    ),
    "CogVideoX 1.5 (i2v)": (
        "VALIDATION.md",
        '<a id="evidence-cogvideox-1-5-i2v"></a>',
        ("BF16", "45-frame", "0.015932"),
    ),
    "MiniMax H3 (t2va/fl2va/ref2va AV)": (
        "VALIDATION.md",
        '<a id="evidence-minimax-h3-t2va-fl2va-ref2va-av"></a>',
        ("512x320x5", "0.000", "0.023"),
    ),
    "WanDancer": (
        "VALIDATION.md",
        '<a id="evidence-wandancer"></a>',
        ("298", "three sampler", "cross-rank", "INCONCLUSIVE/no-material"),
    ),
    "Mage-Flow (T2I/Edit, quality + Turbo)": (
        "VALIDATION.md",
        '<a id="evidence-mage-flow-t2i-edit-quality-turbo"></a>',
        ("0.081165/0.028193", "0.0", "CFG2"),
    ),
    "Qwen Image 2.1": (
        "VALIDATION.md",
        '<a id="evidence-qwen-image-2-1"></a>',
        ("native references", "25-step", "matched"),
    ),
    "Wan-Animate 2": (
        "VALIDATION.md",
        '<a id="evidence-wan-animate-2"></a>',
        ("INT8", "BF16", "identical rank latents"),
    ),
}
_DETAIL_STATUS_RE = re.compile(
    r"Detail status \(`(?P<name>[^`]+)`\): "
    r"(?P<status>\*\*HW\*\*|impl)\."
)
_WARNING_EXEMPT_FAMILY_CONTRACT: frozenset[str] = frozenset()


def _markdown_cells(line: str) -> list[str]:
    trimmed = line.strip()
    assert (
        trimmed.startswith("|")
        and trimmed.endswith("|")
        and not trimmed.startswith("||")
        and not trimmed.endswith("||")
    ), f"expected exactly one outer pipe on a Markdown table row, got: {trimmed}"
    return [cell.strip() for cell in trimmed[1:-1].split("|")]


def _parse_code_families(src: str) -> frozenset[str]:
    tree = ast.parse(src)
    stores = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "_HW_VALIDATED_FAMILIES" and isinstance(node.ctx, ast.Store)
    ]
    assignments = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_HW_VALIDATED_FAMILIES"
    ]
    assert len(stores) == 1 and len(assignments) == 1, (
        "_HW_VALIDATED_FAMILIES must have exactly one plain top-level assignment"
    )

    value = assignments[0].value
    assert (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "frozenset"
        and len(value.args) <= 1
        and not value.keywords
        and (not value.args or isinstance(value.args[0], ast.Set))
    ), "_HW_VALIDATED_FAMILIES must be exactly frozenset() or frozenset({...})"
    elements = value.args[0].elts if value.args else []
    assert all(isinstance(element, ast.Constant) and isinstance(element.value, str) for element in elements), (
        "_HW_VALIDATED_FAMILIES must contain only string literals"
    )
    families = [element.value for element in elements if isinstance(element, ast.Constant)]
    assert len(families) == len(set(families)), "_HW_VALIDATED_FAMILIES contains duplicate entries"
    return frozenset(families)


def _code_families() -> frozenset[str]:
    with open(COMMON, encoding="utf-8") as fh:
        return _parse_code_families(fh.read())


def _parse_support_rows(lines: list[str]) -> dict[_RowKey, str]:
    rows: dict[_RowKey, str] = {}
    for heading, section in _SECTION_HEADINGS.items():
        positions = [index for index, line in enumerate(lines) if line.strip() == heading]
        assert len(positions) == 1, f"MODELS.md must contain exactly one {heading!r}; found {len(positions)}"
        position = positions[0] + 1
        while position < len(lines) and not lines[position].strip():
            position += 1

        assert position < len(lines), f"MODELS.md has no table after {heading}"
        header = _markdown_cells(lines[position])
        assert header == _SUPPORT_HEADER, f"MODELS.md {section} support header drift: {header}"
        position += 1

        assert position < len(lines), f"MODELS.md has no table delimiter after {heading}"
        delimiter = _markdown_cells(lines[position])
        assert len(delimiter) == len(_SUPPORT_HEADER) and all(
            re.fullmatch(r":?-{2,}:?", cell) for cell in delimiter
        ), f"MODELS.md {section} support delimiter drift: {delimiter}"
        position += 1

        section_rows = 0
        while position < len(lines) and lines[position].startswith("|"):
            cells = _markdown_cells(lines[position])
            assert len(cells) == len(_SUPPORT_HEADER), (
                f"MODELS.md {section} support row must have "
                f"{len(_SUPPORT_HEADER)} columns, got {len(cells)}: "
                f"{lines[position].rstrip()}"
            )
            name = cells[0]
            raw_status = cells[6]
            if raw_status == "**HW**":
                status = "HW"
            elif raw_status == "impl":
                status = "impl"
            else:
                raise AssertionError(
                    f"MODELS.md support row {(section, name)!r} has unknown status token {raw_status!r}"
                )
            key = (section, name)
            assert key not in rows, f"duplicate MODELS.md support row: {key!r}"
            rows[key] = status
            section_rows += 1
            position += 1
        assert section_rows, f"MODELS.md {section} support table is empty"
    return rows


def _doc_support_rows() -> dict[_RowKey, str]:
    with open(MODELS, encoding="utf-8") as fh:
        return _parse_support_rows(fh.readlines())


def _models_lines() -> list[str]:
    with open(MODELS, encoding="utf-8") as fh:
        return fh.readlines()


def _validation_text() -> str:
    with open(VALIDATION, encoding="utf-8") as fh:
        return fh.read()


def _assert_no_global_hardware_review_claim(lines: list[str], *, name: str) -> None:
    text = " ".join(line.strip() for line in lines)
    assert not re.search(r"(?:Last|Latest|Current) hardware (?:review|validation|acceptance)(?: date)?:", text), (
        f"{name} must not imply a document-wide current hardware qualification"
    )
    for paragraph in "".join(lines).split("\n\n"):
        if "Last documentation review:" in paragraph:
            assert re.search(r"(?:not|is not) (?:a )?hardware(?:-validation evidence| test)", paragraph), (
                f"{name} must not present a documentation review as hardware evidence"
            )


def _parse_detail_statuses(lines: list[str]) -> dict[str, str]:
    details: dict[str, str] = {}
    for line in lines:
        if not line.startswith("Detail status "):
            continue
        match = _DETAIL_STATUS_RE.fullmatch(line.strip())
        assert match is not None, f"malformed MODELS.md detail-status declaration: {line.rstrip()}"
        name = match.group("name")
        assert name not in details, f"duplicate MODELS.md detail-status declaration: {name!r}"
        details[name] = "HW" if match.group("status") == "**HW**" else "impl"
    return details


def _assert_evidence_source_record(*, name: str, source: str, marker: str, results: tuple[str, ...]) -> None:
    assert source == "VALIDATION.md", f"unknown model evidence source: {source!r}"
    source_text = _validation_text()
    assert marker in source_text, (
        f"MODELS.md cites {source!r} for {name}, but that source "
        "does not contain the named acceptance record"
    )
    section = source_text.split(marker, 1)[1].split('<a id=', 1)[0]
    body = " ".join(section.split())
    for result in results:
        assert result in body, f"{name} acceptance record lacks result: {result}"
    if name == "WanDancer":
        assert re.search(r"(?:identical|equal|match(?:ed|ing)?).{0,100}cross-rank|cross-rank.{0,100}(?:identical|equal|match(?:ed|ing)?)", body), (
            "WanDancer acceptance record must retain exact cross-rank comparison"
        )


def _assert_detail_status_contract(lines: list[str]) -> None:
    support_rows = _parse_support_rows(lines)
    details = _parse_detail_statuses(lines)
    missing = sorted(_REQUIRED_DETAIL_EVIDENCE.keys() - details.keys())
    assert not missing, f"MODELS.md is missing required detail-status declarations: {missing}"
    support_keys_by_name: dict[str, list[_RowKey]] = {}
    for key in support_rows:
        support_keys_by_name.setdefault(key[1], []).append(key)
    for name, detail_status in details.items():
        matches = support_keys_by_name.get(name, [])
        assert len(matches) == 1, (
            f"MODELS.md detail status {name!r} must name exactly one support row; found {matches}"
        )
        support_status = support_rows[matches[0]]
        assert detail_status == support_status, (
            f"MODELS.md detail/status drift for {name!r}: detail says "
            f"{detail_status}, support table says {support_status}"
        )
        if name not in _REQUIRED_DETAIL_EVIDENCE:
            continue
        source, marker, results = _REQUIRED_DETAIL_EVIDENCE[name]
        if source is None:
            continue
        assert marker is not None and results
        anchor = re.fullmatch(r'<a id="([^\"]+)"></a>', marker)
        assert anchor is not None
        section = "".join(lines).split(f"Detail status (`{name}`):", 1)[1].split("\n### ", 1)[0]
        assert f"({source}#{anchor.group(1)})" in section, (
            f"MODELS.md promotion source drift for {name!r}: "
            "its detail section must link to its own acceptance record"
        )
        _assert_evidence_source_record(name=name, source=source, marker=marker, results=results)


def _mutate_detail_status(lines: list[str], name: str, *, new_status: str) -> list[str]:
    mutated = list(lines)
    positions = [index for index, line in enumerate(mutated) if line.startswith(f"Detail status (`{name}`):")]
    assert len(positions) == 1, f"expected exactly one detail-status declaration for {name!r}"
    assert _DETAIL_STATUS_RE.fullmatch(mutated[positions[0]].strip()) is not None
    mutated[positions[0]] = f"Detail status (`{name}`): {new_status}.\n"
    return mutated


def _assert_models_footer_provenance(lines: list[str]) -> None:
    _assert_no_global_hardware_review_claim(lines, name="MODELS.md")
    section = _validation_text().split('<a id="evidence-wandancer"></a>', 1)[1].split('<a id=', 1)[0]
    body = " ".join(section.split())
    assert "broader acceptance remains HELD" in body
    assert "298 healthy frames" in body and "identical cross-rank latents" in body


def _row_index(lines: list[str], name: str) -> int:
    positions = [index for index, line in enumerate(lines) if line.startswith(f"| {name} |")]
    assert len(positions) == 1, f"expected exactly one MODELS.md support row named {name!r}; found {len(positions)}"
    return positions[0]


def _mutate_row(
    lines: list[str],
    name: str,
    *,
    new_name: str | None = None,
    new_status: str | None = None,
) -> list[str]:
    mutated = list(lines)
    position = _row_index(mutated, name)
    cells = _markdown_cells(mutated[position])
    if new_name is not None:
        cells[0] = new_name
    if new_status is not None:
        cells[6] = new_status
    mutated[position] = f"| {' | '.join(cells)} |\n"
    return mutated


def _assert_support_contract(actual: dict[_RowKey, str]) -> None:
    expected = {key: status for key, (_family, status) in _SUPPORT_ROW_CONTRACT.items()}
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    changed = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in sorted(expected.keys() & actual.keys())
        if expected[key] != actual[key]
    }
    assert not missing and not unexpected and not changed, (
        "MODELS.md support-row status drift; every row needs its own evidence "
        f"review; missing={missing}, unexpected={unexpected}, changed={changed}"
    )


def _doc_hw_families() -> tuple[frozenset[str], list[_RowKey]]:
    rows = _doc_support_rows()
    unmapped = sorted(rows.keys() - _SUPPORT_ROW_CONTRACT.keys())
    statuses: dict[str, set[str]] = {}
    for key, status in rows.items():
        if key in _SUPPORT_ROW_CONTRACT:
            statuses.setdefault(_SUPPORT_ROW_CONTRACT[key][0], set()).add(status)
    families = {family for family, values in statuses.items() if values == {"HW"}}
    return frozenset(families), unmapped


def _doc_family_statuses() -> dict[str, set[str]]:
    rows = _doc_support_rows()
    statuses: dict[str, set[str]] = {}
    for key, status in rows.items():
        if key in _SUPPORT_ROW_CONTRACT:
            statuses.setdefault(_SUPPORT_ROW_CONTRACT[key][0], set()).add(status)
    return statuses


@pytest.mark.parametrize(
    "line",
    (
        "|| Model | single | cfg2 | Ulysses | ring | FSDP | Status | Notes |",
        "| Model | single | cfg2 | Ulysses | ring | FSDP | Status | Notes ||",
    ),
)
def test_markdown_cells_rejects_double_outer_pipes(line: str) -> None:
    with pytest.raises(AssertionError, match="exactly one outer pipe"):
        _markdown_cells(line)


def test_support_matrix_row_statuses_match_evidence_contract() -> None:
    _assert_support_contract(_doc_support_rows())


def test_detailed_statuses_match_support_matrix_and_promotion_provenance() -> None:
    _assert_detail_status_contract(_models_lines())


@pytest.mark.parametrize("name", ("Wan 2.2 (i2v)", "WanDancer"))
def test_detail_status_contract_rejects_narrative_demotion(name: str) -> None:
    lines = _mutate_detail_status(
        _models_lines(),
        name,
        new_status="impl",
    )
    with pytest.raises(AssertionError, match="detail/status drift"):
        _assert_detail_status_contract(lines)


def test_detail_status_contract_requires_each_promoted_narrative() -> None:
    lines = _models_lines()
    position = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("Detail status (`WanDancer`):")
    )
    lines.pop(position)
    with pytest.raises(AssertionError, match="missing required detail-status"):
        _assert_detail_status_contract(lines)


def test_detail_status_contract_rejects_wrong_model_evidence_link() -> None:
    lines = [line.replace("VALIDATION.md#evidence-wandancer", "VALIDATION.md#evidence-wan-animate-2") for line in _models_lines()]
    with pytest.raises(AssertionError, match="promotion source drift"):
        _assert_detail_status_contract(lines)


def test_detail_status_contract_rejects_source_without_acceptance(monkeypatch) -> None:
    original = _validation_text
    marker = '<a id="evidence-wandancer"></a>'
    monkeypatch.setitem(
        globals(), "_validation_text", lambda: original().replace(marker, ""),
    )
    with pytest.raises(AssertionError, match="does not contain the named acceptance"):
        _assert_detail_status_contract(_models_lines())


@pytest.mark.parametrize(
    "removed",
    (
        "298",
        "three sampler",
        "INCONCLUSIVE/no-material",
    ),
)
def test_wandancer_promotion_rejects_missing_acceptance_result(monkeypatch, removed) -> None:
    original = _validation_text()
    marker = '<a id="evidence-wandancer"></a>'
    before, heading, after = original.partition(marker)
    section, separator, rest = after.partition('<a id=')
    assert removed in section
    changed = before + heading + section.replace(removed, "") + separator + rest
    monkeypatch.setitem(globals(), "_validation_text", lambda: changed)
    with pytest.raises(AssertionError, match="WanDancer acceptance record lacks result"):
        _assert_detail_status_contract(_models_lines())


def test_models_keeps_documentation_review_separate_from_hardware_evidence() -> None:
    _assert_models_footer_provenance(_models_lines())


@pytest.mark.parametrize("claim", ("Last hardware validation: current.", "Last documentation review: today. All hardware is qualified."))
def test_models_rejects_unscoped_hardware_review_claims(claim) -> None:
    with pytest.raises(AssertionError, match="hardware"):
        _assert_no_global_hardware_review_claim([claim], name="MODELS.md")


def test_wandancer_acceptance_wording_retains_instantiated_graph_scope() -> None:
    validation = " ".join(_validation_text().split())
    section = validation.split('<a id="evidence-wandancer"></a>', 1)[1].split('<a id=', 1)[0]
    for term in ("global-to-local", "uly2", "FP8", "segment", "image", "audio", "decode", "reference_latent", "FSDP"):
        assert term.casefold() in section.casefold(), term
    assert "committed uly2 graph verbatim" not in section


@pytest.mark.parametrize(
    "name",
    (
        "CogVideoX 1.5 (inpaint)",
    ),
)
def test_support_contract_rejects_unsupported_hardware_promotion(name: str) -> None:
    actual = _parse_support_rows(_mutate_row(_models_lines(), name, new_status="**HW**"))
    with pytest.raises(AssertionError, match="changed"):
        _assert_support_contract(actual)


def test_support_contract_rejects_hardware_demotion() -> None:
    actual = _parse_support_rows(
        _mutate_row(
            _models_lines(),
            "LTX 2.3 (i2v/AV packed)",
            new_status="impl",
        )
    )
    with pytest.raises(AssertionError, match="changed"):
        _assert_support_contract(actual)


def test_support_contract_rejects_row_rename() -> None:
    actual = _parse_support_rows(_mutate_row(_models_lines(), "Lens", new_name="Lens renamed"))
    with pytest.raises(AssertionError, match=r"missing=.*unexpected"):
        _assert_support_contract(actual)


def test_support_contract_rejects_row_deletion() -> None:
    lines = _models_lines()
    lines.pop(_row_index(lines, "Wan SCAIL2"))
    actual = _parse_support_rows(lines)
    with pytest.raises(AssertionError, match="missing"):
        _assert_support_contract(actual)


def test_support_contract_rejects_row_addition() -> None:
    lines = _models_lines()
    original_position = _row_index(lines, "WanDancer")
    future_row = _mutate_row(lines, "WanDancer", new_name="Future Video")[original_position]
    lines.insert(original_position + 1, future_row)
    actual = _parse_support_rows(lines)
    with pytest.raises(AssertionError, match="unexpected"):
        _assert_support_contract(actual)


def test_support_parser_rejects_unknown_status_token() -> None:
    lines = _mutate_row(_models_lines(), "LTX 2.3 (i2v/AV packed)", new_status="hardware")
    with pytest.raises(AssertionError, match="unknown status token"):
        _parse_support_rows(lines)


def test_code_family_parser_rejects_composed_assignment() -> None:
    src = '_HW_VALIDATED_FAMILIES = frozenset({"ltx", "wan"}) | {"flux2"}\n'
    with pytest.raises(AssertionError, match=r"exactly frozenset"):
        _parse_code_families(src)


def test_hw_warning_exempt_families_are_explicitly_reviewed() -> None:
    code = _code_families()
    _doc, unmapped = _doc_hw_families()
    assert not unmapped, (
        f"MODELS.md support rows without a family mapping in this test: "
        f"{unmapped}; add them to _SUPPORT_ROW_CONTRACT"
    )
    assert code == _WARNING_EXEMPT_FAMILY_CONTRACT, (
        "runtime auto-warning exemptions require family-wide named evidence; "
        f"review code and this contract together, got {sorted(code)}"
    )
    statuses = _doc_family_statuses()
    assert statuses["wan"] == {"HW"} and "wan" not in code
    assert statuses["wan_scail"] == {"HW"} and "wan_scail" not in code
    assert statuses["wan_dancer"] == {"HW"} and "wan_dancer" not in code
    assert statuses["ideogram4"] == {"HW"} and "ideogram4" not in code
    assert statuses["zimage"] == {"HW"} and "zimage" not in code
    assert statuses["chroma"] == {"HW"} and "chroma" not in code
    assert statuses["flux"] == {"HW"} and "flux" not in code
    assert statuses["cogvideo"] == {"HW", "impl"} and "cogvideo" not in code
    # The 2026-08-04 campaign exercised the former time_shift_slope
    # contract; the carried-audio ODE change demoted the row on 2026-08-06,
    # and the 2026-08-07 re-gate ladder repromoted it (docs/VALIDATION.md).
    assert statuses["minimax_h3"] == {"HW"} and "minimax_h3" not in code


def test_auto_markers_and_refusals_match_runtime_contract() -> None:
    """The Notes of the rows below name the constraint that keeps operators off an unsafe path.

    The detailed cards own the evidence narratives, so the matrix needs only
    the constraint and its Details link; the AUTO_TABLE and grant tests below
    own the mode and evidence contracts.
    """
    lines = _models_lines()

    chroma = _markdown_cells(lines[_row_index(lines, "Chroma")])
    assert chroma[6] == "**HW**"
    assert "NVFP4 is waiver-only" in chroma[7]
    assert "Ulysses rejects effective masks" in chroma[7]

    pixeldit = _markdown_cells(lines[_row_index(lines, "PixelDiT / PiD")])
    assert "quantized sequence sharding and cfg2 refuse" in pixeldit[7]

    zimage = _markdown_cells(
        lines[_row_index(lines, "Z-Image latent")]
    )
    assert "cfg2 requires equal-token prompts" in zimage[7]

    ideogram = _markdown_cells(lines[_row_index(lines, "Ideogram4")])
    assert "Explicit cfg2 splits the two checkpoints" in ideogram[7]
    assert "Torch Flash" in ideogram[7]

    longcat = _markdown_cells(lines[_row_index(lines, "LongCat-Image")])
    assert "ring/hybrid padding is guarded" in longcat[7]

    lens = _markdown_cells(lines[_row_index(lines, "Lens")])
    assert lens[6] == "**HW**"
    assert "Asymmetric cfg2 is measured" in lens[7]
    assert "padded ring remains guarded" in lens[7]

    anima = _markdown_cells(lines[_row_index(lines, "Anima")])
    assert anima[6] == "**HW**"
    assert "supplied graph for cross-attention and masks" in anima[7]


# Matrix glyphs derive from the runtime table and the recorded grants.

_MODE_COLUMN = {"cfg2": 2, "uly": 3, "ring": 4, "fsdp": 5}
_AUTO_MARK = "★"
_PENDING_MARK = "⏳"
_CONSTRAINED_MARK = "🔒"

# Megapixel probes that straddle every crossover boundary in AUTO_TABLE, so a
# row that only selects cfg2 below 1.2 MP is seen as well as one that only
# selects Ulysses above it.
_MP_PROBES = (0.26, 0.79, 1.05, 1.19, 1.5, 2.36, 8.0)

# Rows whose canonical checkpoint is guidance distilled and renders at CFG 1.0.
# choose_auto_topology drops every cfg-parallel row at that CFG, so auto cannot
# reach cfg2 for them. This is a fact about the checkpoint, not a glyph: the
# stars below derive from it.
_DISTILLED_CFG1_ROWS = frozenset({"Flux 1 Dev", "Flux 1 Schnell", "Flux2"})
_REAL_CFG = 3.5

# What the dated hardware evidence behind an **HW** row covers, per starred
# mode. One reviewed entry per starred cell, and four states:
#
#   granted   a comparison covering this mode passed at this row's scope
#   scoped    the comparison covers a narrower topology than the star
#   withdrawn the mode was granted once and the grant no longer holds
#   unrun     auto selects the mode and no comparison for it has ever run
#
# A withdrawn or unrun cell carries the requalification mark (_PENDING_MARK);
# a granted or scoped one does not. For scoped, withdrawn and unrun, the second
# field is a substring the row's Notes must hold. A star on an HW row with no
# entry here fails CI: the state is a review decision, and a glyph cannot
# imply it.
_STARRED_MODE_GRANT: dict[tuple[str, str], tuple[str, str]] = {
    ("Krea2 (RAW+Turbo)", "uly"): (
        "granted", "2026-10-06 bit-identical to dp2 and one GPU; detail table fp8 uly2; VALIDATION 2026-07-10 turbo fp8 uly2 vs dp2"),
    ("Chroma", "cfg2"): ("granted", "2026-08-11 FP8 cfg2 1024x1024, exact cross-rank"),
    ("Chroma", "uly"): ("granted", "2026-08-21 requalification: fp8 NRMS 0.023 / bf16 0.010 at 1536"),
    ("Ideogram4", "uly"): (
        "granted", "2026-10-06 bit-identical to dp2 and one GPU; 2026-08-11 fp8 dual-model pair, DP2 NRMS 0.000"),
    ("Flux 1 Dev", "uly"): ("granted", "2026-10-06 bit-identical to dp2 and one GPU; 2026-08-11 DP2 NRMS 0.008"),
    ("Flux2", "uly"): ("scoped", "BF16 `uly2+fsdp`"),
    ("LongCat-Image", "cfg2"): ("granted", "2026-08-11 DP2 NRMS 0.009"),
    ("LongCat-Image", "uly"): ("granted", "2026-10-06 bit-identical to dp2 and one GPU; 2026-08-11 DP2 NRMS 0.097"),
    ("HunyuanImage 2.1 (+refiner)", "uly"): (
        "granted", "2026-10-06 bit-identical to dp2 and one GPU; AUTO_TABLE row 46: uly2 vouched for Image 2.1 and refiner 2026-07-15"),
    ("Qwen-Image", "uly"): (
        "granted", "2026-10-06 bit-identical to dp2 and one GPU; 2026-08-11 base 2512 FP8, DP2 NRMS 0.049; 2026-09-14 uly2 TORCH_FLASH, "
        "one-GPU NRMS 0.082 (c2b9258ffce7); auto keeps the declared kernel, SAGE read 0.162 (#377)"),
    ("Mage-Flow (T2I/Edit, quality + Turbo)", "uly"): (
        "granted", "2026-10-06 bit-identical to dp2 and one GPU; twelve retained T2I precision rows"),
    ("Qwen Image 2.1", "uly"): ("granted", "twelve retained DiT artifacts"),
    ("Ernie-Image", "uly"): ("granted", "2026-10-06 bit-identical to dp2 and one GPU; DP2 NRMS 0.000; fresh 1024x1024 2026-08-11"),
    ("Radiance", "cfg2"): ("scoped", "equal-length prompts"),
    ("Radiance", "uly"): ("granted", "2026-08-21 DP2 NRMS 0.036 at 1024, fp32 artifact"),
    ("Flux 1 Schnell", "uly"): ("granted", "2026-08-21 DP2 NRMS 0.048 at 1024"),
    ("Z-Image DCT PixelSpace", "uly"): ("granted", "2026-08-21 DP2 NRMS 0.000 at 1536"),
    ("Z-Image latent", "uly"): ("granted", "2026-08-11 DP2 NRMS 0.000, fresh 1536x1536"),
    ("Lens", "uly"): ("granted", "2026-10-06 bit-identical to one GPU; 2026-07-28 DP2 NRMS 0.060-0.061 after the pad-row fix"),
    ("Omnigen2", "ring"): ("granted", "certified DP2 NRMS 0.002; fresh run 2026-08-11"),
    ("Anima", "uly"): ("granted", "byte-identical to true single GPU 2026-08-11"),
    ("Boogu", "uly"): ("granted", "2026-10-06 bit-identical to dp2; 2026-08-11 DP2 NRMS 0.022"),
    ("PixelDiT / PiD", "uly"): ("granted", "world-2 bf16: uly2 matched dp2 exactly 2026-08-11"),
    ("Kandinsky5-Image", "uly"): (
        "granted", "AUTO_TABLE row 44: uly2 vouched for Image 2026-07-13"),
    ("LTX 2.3 (t2v)", "uly"): ("granted", "t2v smoke, 22B distilled + distill LoRA under uly2"),
    ("LTX 2.3 (i2v/AV packed)", "uly"): (
        "granted", "2026-07-26 packed i2v/AV campaign, three one-step comparisons"),
    ("LTX 2.5 (t2v/AV packed)", "uly"): ("granted", "2026-08-12 world-2 t2v/AV packed scope"),
    ("Wan 2.1 (t2v)", "uly"): ("granted", "2026-08-11 FP8 832x480, exact cross-rank"),
    ("Wan 2.2 (t2v, high/low)", "uly"): (
        "granted", "2026-08-11 BF16 high-to-low handoff, exact cross-rank"),
    ("Wan 2.2 (i2v)", "uly"): ("granted", "2026-07-28 four-step lightx2v, DP2 NRMS 0.000"),
    ("Wan 2.1 (i2v)", "uly"): ("granted", "2026-09-27 archived video comparison"),
    ("Wan FlowRVS", "uly"): ("granted", "2026-09-27 archived video comparison"),
    ("Wan Bernini-R", "uly"): ("granted", "2026-09-27 archived video comparison"),
    ("Wan SCAIL Preview", "uly"): ("scoped", "World-2 Ulysses, FP16 one- and four-step synthetic 256x256x5 only"),
    ("Wan SCAIL2", "uly"): ("granted", "2026-09-27 archived video comparison"),
    ("WanDancer", "uly"): ("granted", "2026-08-06 two-pass FP8 acceptance, cross-rank identity"),
    ("Wan-Animate 2", "uly"): ("granted", "bounded true-native Base twelve-format"),
    ("HunyuanVideo 1.5 (+SR)", "uly"): (
        "granted", "2026-10-06 bit-identical to dp2 and one GPU; AUTO_TABLE row 46: uly2 vouched for Video 1.5 2026-07-15"),
    ("CogVideoX 1.5 (t2v)", "uly"): ("granted", "2026-08-11 certified DP2 NRMS 0.082"),
    ("CogVideoX 1.5 (i2v)", "uly"): ("granted", "2026-09-27 archived video comparison"),
    ("Kandinsky5-Video Lite", "uly"): (
        "granted", "AUTO_TABLE row 44: uly2 vouched for Video Lite 2026-07-13"),
    ("Kandinsky5-Video Pro", "uly"): ("scoped", "FSDP validated"),
    ("MiniMax H3 (t2va/fl2va/ref2va AV)", "uly"): (
        "granted", "2026-08-07 carried-audio scope, NRMS 0.000 per stream"),
}

# FSDP cells whose artifacts FSDP admits but no FSDP comparison qualifies; each
# must read constrained (_CONSTRAINED_MARK). The name dates from before
# 2026-08-26, when the bf16-only FSDP precision grant refused these artifacts.
# The rule is one-directional: the constrained mark also covers cells with no
# entry here, such as Ideogram4 ring measuring over the fidelity limit or Flux 1
# cfg2 being unreachable on a distilled checkpoint.
_TYPED_REFUSAL: dict[tuple[str, str], str] = {
    ("LTX 2.5 (t2v/AV packed)", "fsdp"): (
        "the bf16 artifacts are admitted through the fp32-islands profile and the "
        "int8-convrot releases as stored bytes, but no FSDP render has run"),
    ("MiniMax H3 (t2va/fl2va/ref2va AV)", "fsdp"): (
        "the bf16 FSDP execution completed after auxiliary-module replication, "
        "but its reference comparison remains unqualified"),
    ("HunyuanVideo 1.5 (+SR)", "fsdp"): (
        "the FP8 SR and fp16 Video artifacts are admitted but no FSDP render of "
        "them has run"),
}


def _support_matrix_rows() -> dict[str, list[str]]:
    """{model cell: the row's eight cells} for every support-matrix row."""
    rows: dict[str, list[str]] = {}
    for line in _models_lines():
        if not line.startswith("|"):
            continue
        cells = _markdown_cells(line)
        if len(cells) != 8 or cells[0] in ("Model", ""):
            continue
        if set(cells[1]) <= {"-", ":"}:
            continue
        rows[cells[0]] = cells
    assert rows, "no support-matrix rows parsed out of docs/MODELS.md"
    return rows


def _auto_selected_modes(family: str, cfg_value: float) -> set[str]:
    """Every parallel mode `auto` can land on for this family at this CFG."""
    from dgx_monarch.topology import choose_auto_topology

    modes: set[str] = set()
    for quant in ("bf16", "fp8"):
        for megapixels in _MP_PROBES:
            topology = choose_auto_topology(
                family, quant, megapixels, 2, cfg_value=cfg_value
            ).topology
            if topology.cfg > 1:
                modes.add("cfg2")
            if topology.ulysses > 1:
                modes.add("uly")
            if topology.ring > 1:
                modes.add("ring")
            if topology.fsdp:
                modes.add("fsdp")
    return modes


def test_auto_stars_derive_from_the_runtime_auto_table() -> None:
    """The star (_AUTO_MARK) marks exactly the modes AUTO_TABLE selects.

    The expected set comes from the runtime, so the matrix cannot star a mode
    the table stopped choosing, or omit one it still chooses.
    """
    rows = _support_matrix_rows()
    failures = []
    for (_section, model), (family, _status) in _SUPPORT_ROW_CONTRACT.items():
        cells = rows[model]
        cfg_value = 1.0 if model in _DISTILLED_CFG1_ROWS else _REAL_CFG
        selected = _auto_selected_modes(family, cfg_value)
        starred = {mode for mode, column in _MODE_COLUMN.items()
                   if _AUTO_MARK in cells[column]}
        if starred != selected:
            failures.append(
                f"{model}: matrix stars {sorted(starred)}, but AUTO_TABLE selects "
                f"{sorted(selected)} for family {family!r} at cfg {cfg_value}"
            )
    assert not failures, (
        "support-matrix stars disagree with src/dgx_monarch/topology.py:\n"
        + "\n".join(failures)
    )


def test_every_starred_hardware_cell_shows_the_state_of_its_grant() -> None:
    """A star says auto picks the mode; the second mark says whether evidence backs it.

    An **HW** row whose starred mode has a withdrawn or unrun grant must carry
    the requalification mark and say so in its Notes, so an operator reading
    one line cannot mistake `auto`'s route for measured evidence.
    """
    rows = _support_matrix_rows()
    failures = []
    covered = set()
    for (_section, model), (_family, _status) in _SUPPORT_ROW_CONTRACT.items():
        cells = rows[model]
        if cells[6] != "**HW**":
            continue
        for mode, column in _MODE_COLUMN.items():
            if _AUTO_MARK not in cells[column]:
                continue
            record = _STARRED_MODE_GRANT.get((model, mode))
            if record is None:
                failures.append(
                    f"{model} {mode}: starred on an HW row with no entry in "
                    "_STARRED_MODE_GRANT; record what the hardware evidence covers"
                )
                continue
            covered.add((model, mode))
            state, detail = record
            pending = _PENDING_MARK in cells[column]
            if state in ("withdrawn", "unrun"):
                if not pending:
                    failures.append(
                        f"{model} {mode}: grant is {state}, so the cell must carry "
                        f"{_PENDING_MARK}"
                    )
                if detail not in cells[7]:
                    failures.append(
                        f"{model} {mode}: grant is {state}; the Notes must say so "
                        f"({detail!r} is missing)"
                    )
            else:
                if pending:
                    failures.append(
                        f"{model} {mode}: grant is {state}, so the cell must not "
                        f"carry {_PENDING_MARK}"
                    )
                if state == "scoped" and detail not in cells[7]:
                    failures.append(
                        f"{model} {mode}: the grant covers a narrower topology than "
                        f"the star; the Notes must name it ({detail!r} is missing)"
                    )
    stale = sorted(set(_STARRED_MODE_GRANT) - covered)
    assert not stale, (
        f"_STARRED_MODE_GRANT names cells that are no longer starred HW rows: {stale}"
    )
    assert not failures, "\n".join(failures)


def test_documented_typed_refusals_read_as_constrained_cells() -> None:
    """Each _TYPED_REFUSAL cell reads constrained, not supported.

    FSDP admits the row's artifacts, but no FSDP comparison qualifies them
    (the Known limitations in docs/MODELS.md).
    """
    rows = _support_matrix_rows()
    failures = [
        f"{model} {mode}: {reason}, so the cell must read {_CONSTRAINED_MARK}, "
        f"found {rows[model][_MODE_COLUMN[mode]]!r}"
        for (model, mode), reason in _TYPED_REFUSAL.items()
        if _CONSTRAINED_MARK not in rows[model][_MODE_COLUMN[mode]]
    ]
    assert not failures, "\n".join(failures)


# Cells a review marked constrained because the runtime admits the mode but no
# comparison qualifies it. Unlike _TYPED_REFUSAL, each entry also holds the
# exact Notes phrase that names the mode as unqualified. Like it, the rule
# covers only the listed cells: other rows whose Notes call a mode unqualified
# keep the mark their own review gave them.
_ADMITTED_UNQUALIFIED: dict[tuple[str, str], tuple[str, str]] = {
    ("Wan-Animate 2", "ring"): (
        "the adapter's spatial gather runs for any sequence-parallel degree, but "
        "no Animate2 ring comparison has run (VALIDATION.md evidence history)",
        "FSDP, ring and cache have not been tested on hardware"),
    ("Wan-Animate 2", "fsdp"): (
        "FSDP admits the exact Animate2 model pair, but no Animate2 FSDP "
        "comparison has run (VALIDATION.md evidence history)",
        "FSDP, ring and cache have not been tested on hardware"),
}
_MODE_NAME = {"cfg2": "cfg2", "uly": "Ulysses", "ring": "ring", "fsdp": "FSDP"}


def test_admitted_but_unqualified_modes_read_as_constrained_cells() -> None:
    rows = _support_matrix_rows()
    failures = []
    for (model, mode), (reason, phrase) in _ADMITTED_UNQUALIFIED.items():
        assert _MODE_NAME[mode] in phrase and phrase.endswith("have not been tested on hardware"), (
            f"{model} {mode}: pinned phrase {phrase!r} must name "
            f"{_MODE_NAME[mode]} without a passing comparison")
        cells = rows[model]
        if _CONSTRAINED_MARK not in cells[_MODE_COLUMN[mode]]:
            failures.append(
                f"{model} {mode}: {reason}, so the cell must read "
                f"{_CONSTRAINED_MARK}, found {cells[_MODE_COLUMN[mode]]!r}")
        if phrase not in cells[7]:
            failures.append(
                f"{model} {mode}: the Notes must name {_MODE_NAME[mode]} as "
                f"without a passing comparison ({phrase!r} is missing), found {cells[7]!r}")
    assert not failures, "\n".join(failures)


def test_issue_223_documentation_corrections_remain_in_sync() -> None:
    def read(path: str) -> str:
        with open(path, encoding="utf-8") as fh:
            return fh.read()

    security = read(SECURITY)
    install = read(INSTALL)
    tui = read(TUI)
    threat = read(THREAT_MODEL)
    design = read(DESIGN)
    troubleshooting = read(TROUBLESHOOTING)
    readme = read(README)
    cluster = read(CLUSTER)
    concepts = read(CONCEPTS)
    adapters = read(ADAPTERS)
    bug_report = read(BUG_REPORT)
    trust = read(TRUST)

    assert "current default branch" in security
    assert "current default branch directly from the remote" in " ".join(install.split())
    assert "`origin/HEAD`" in install
    _assert_no_global_hardware_review_claim(threat.splitlines(), name="THREAT_MODEL.md")
    assert "docs/TROUBLESHOOTING.md #58" in tui
    assert "full-fidelity diagnostic surface" in " ".join(tui.split())
    assert "one owner-held, non-symlink regular file with mode `0600`" in " ".join(
        tui.split()
    )
    assert "crossover table row 10" in design
    assert "crossover table row 14" not in design
    assert "runs on a stock Spark (entry #59)" in troubleshooting
    assert "## 77." in troubleshooting
    assert "## 78." in troubleshooting
    assert "definite non-timeout ordinary failure" in troubleshooting
    assert "chooses from measured rows" not in readme
    assert "may stop that confirmed Worker-service generation once" in " ".join(security.split())
    assert "It never issues a second stop" in security
    assert "missing, changed, or replaced-and-restored marker (an ABA mismatch)" in " ".join(
        security.split()
    )
    assert "canonical generation-invalidating `ExecStartPre`" in " ".join(
        security.split()
    )
    assert "two fence-held readbacks both prove" in " ".join(install.split())
    assert "zero Worker/actor/listener runtime" in " ".join(install.split())
    assert "Before cleanup dispatches a stop, it durably records that intent" in " ".join(security.split())
    assert "single-node` is rejected" in cluster
    assert "`NCCL_PROTO` is not an operator fabric-profile knob" in cluster
    assert "selects one from measured benchmarks" not in concepts
    assert "Receipt schema v2 records an indeterminate step as `unknown`" in concepts
    assert "cfg_parallel_supported = True" in adapters
    assert "immediately after adapter detection" in adapters
    assert (
        "compatible FAIL remains the effective quarantine across later "
        "`INCONCLUSIVE` or `RETESTING` rows"
    ) in " ".join(trust.split())
    assert "../blob/HEAD/SECURITY.md" in bug_report


LTX_TEMPLATE = os.path.join(
    HERE, "..", "example_workflows", "dgx-monarch-ltx-t2v.json"
)

# A claim wrapped across lines reads the same as an inline one. A table row and
# a list item are each their own claim: merging them would let one row's
# qualifier excuse the row beside it.
_LIST_ITEM = re.compile(r"^\s*([*+-]|\d+\.)\s")


def _prose_units(text: str) -> list[str]:
    units: list[str] = []
    block: list[str] = []

    def flush() -> None:
        nonlocal block
        if block:
            units.append(" ".join(block))
            block = []

    for line in text.splitlines():
        if line.startswith("|") or not line.strip():
            flush()
            if line.startswith("|"):
                units.append(line)
            continue
        if _LIST_ITEM.match(line):
            flush()
        block.append(line.strip())
    flush()
    return units


def _rdma_claim_surfaces() -> list[tuple[str, str]]:
    """The operator-facing surfaces that offer RDMA latent return as a feature.

    The model page, the front page, and the one shipped template whose note
    mentions it. Records that discuss the path rather than offer it (VALIDATION,
    docs/TROUBLESHOOTING.md #75, CLUSTER, CONTRIBUTING) are out of scope; the
    test below pins the VALIDATION and TROUBLESHOOTING records.
    """
    surfaces = []
    for name, path in (("docs/MODELS.md", MODELS), ("README.md", README)):
        with open(path, encoding="utf-8") as fh:
            surfaces.append((name, fh.read()))
    with open(LTX_TEMPLATE, encoding="utf-8") as fh:
        template = json.load(fh)
    notes = " ".join(
        value
        for node in template["nodes"]
        for value in (node.get("widgets_values") or [])
        if isinstance(value, str)
    )
    surfaces.append(("example_workflows/dgx-monarch-ltx-t2v.json", notes))
    return surfaces


def _units_claiming_rdma_return(text: str) -> list[str]:
    """Units that offer the return path, however the sentence spells it."""
    return [
        unit
        for unit in _prose_units(text)
        if "RDMA" in unit and ("return" in unit or "come back" in unit)
    ]


def test_historical_failed_native_attempt_remains_scoped_and_honest():
    with open(TROUBLESHOOTING, encoding="utf-8") as fh:
        troubleshooting = fh.read()
    validation = _validation_text()

    assert (
        "## 28. Operator-armed setup-kill handoff does not reach PASS"
        in troubleshooting
    )
    assert (
        "## 75. Native RDMA fails at QP RTR because the two ends selected different rails"
    ) in troubleshooting
    assert "leave `rdma_latent_return=false`" in troubleshooting
    assert "do not retry that native" in troubleshooting
    assert "Device hiding is not the supported selector" in troubleshooting

    partial = validation.split("## Native RDMA", 1)[1].split("\n## ", 1)[0]
    for term in ("exactly 8 MiB", "disconnected rails", "FAILED", "NOT RUN / HOLD", "native", "ACK"):
        assert term in partial, term

    # A bare "latents can return over RDMA" reads as a present capability, so
    # every unit on an operator surface that claims the return path must carry
    # the default-off posture and the HOLD. A check on TROUBLESHOOTING alone stays
    # green while a surface drops the HOLD.
    for surface, text in _rdma_claim_surfaces():
        for unit in _units_claiming_rdma_return(text):
            assert "off by default" in unit, (
                f"{surface}: RDMA latent-return claim omits the default-off "
                f"posture: {unit!r}"
            )
            assert "NOT RUN / HOLD" in unit, (
                f"{surface}: RDMA latent-return claim omits the HOLD "
                f"(DESIGN.md 5.6): {unit!r}"
            )


# Docs corrections of 2026-08-13. Each test below pins the corrected wording of
# a statement that contradicted the code or another document.

CONTRIBUTING = os.path.join(HERE, "..", "CONTRIBUTING.md")
PR_TEMPLATE = os.path.join(HERE, "..", ".github", "PULL_REQUEST_TEMPLATE.md")
CI_WORKFLOW = os.path.join(HERE, "..", ".github", "workflows", "ci.yml")
OPERATOR_SKILL = os.path.join(
    HERE, "..", "skills", "dgx-monarch", "SKILL.md"
)
DEV_SKILL = os.path.join(
    HERE, "..", "skills", "dgx-monarch-dev", "SKILL.md"
)
SCRIPTS_README = os.path.join(HERE, "..", "scripts", "README.md")
TEST_CPU = os.path.join(HERE, "..", "scripts", "test_cpu.sh")
LTX_ADAPTER = os.path.join(HERE, "..", "src", "dgx_monarch", "adapters", "ltx.py")
GEN_TEMPLATES = os.path.join(HERE, "..", "tools", "gen_templates.py")

_CI_RUFF = "ruff check __init__.py src tests benchmark tools"
_CI_PYTEST = "bash scripts/test_cpu.sh"
_LTX_COMFY_FLOOR = "bd34f338"


def _file_text(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_auto_mode_prose_matches_the_decision_table() -> None:
    """`AUTO_TABLE` holds rows seeded from another family and rows that record
    part of their scope as unmeasured or unclaimed, `_FALLBACK_RULE` covers
    unknown families, and no row is data-parallel. DESIGN must not call the
    default the measured best or claim data-parallel rows."""
    from dgx_monarch.topology import AUTO_TABLE

    dp_rows = [rule.row for rule in AUTO_TABLE if "dp" in rule.topology]
    assert not dp_rows, f"AUTO_TABLE gained data-parallel rows: {dp_rows}"

    design = " ".join(_file_text(DESIGN).split())
    assert "the measured best" not in design
    assert "data-parallel rows" not in design
    assert "Data parallelism is never a table row" in design
    assert "**The default topology is benchmark-seeded.**" in design


def test_readme_defers_the_support_matrix_to_models() -> None:
    """A second support matrix goes stale. MODELS.md holds the only one, and
    DESIGN's documentation contract points there."""
    readme = " ".join(_file_text(README).split())
    assert "docs/MODELS.md" in readme
    assert "| single | cfg2 | Ulysses | ring | FSDP |" not in readme
    assert "The distributed implementations cover the families below" not in readme
    assert "pointer to the docs/MODELS.md support matrix" in " ".join(
        _file_text(DESIGN).split()
    )

    # Template counts in prose go stale with each new template; README states none.
    assert "Seventeen **workflow templates**" not in readme
    assert "The other twelve are per-family" not in readme
    assert "and use Use `dgxm down`" not in readme


def test_install_examples_survive_an_existing_node_pack_path() -> None:
    """The supported path is often a symlink to the one working checkout, and
    a bare clone onto it aborts."""
    install = " ".join(_file_text(INSTALL).split())
    skill = " ".join(_file_text(OPERATOR_SKILL).split())
    assert "If the intended node-pack path is absent, clone" in install
    assert "An existing checkout or symlink must pass the inspection" in install
    assert "Never overwrite an existing path or create a second copy" in install
    assert "symlink target before using an existing path" in skill
    assert "Preserve local changes and unrelated custom nodes" in skill


def test_launcher_example_passes_the_launchers_own_gate() -> None:
    """comfy-driver.sh refuses unless its checkout is discoverable under
    `$COMFY_DIR/custom_nodes`; a `$PWD/../ComfyUI` example implies a checkout
    beside ComfyUI, which fails that check unless custom_nodes holds a symlink
    to it."""
    prose = _file_text(SCRIPTS_README)
    assert '--comfy-dir "$COMFY_DIR"' in prose
    assert "$PWD/../ComfyUI" not in prose


def test_guided_setup_examples_name_the_comfyui_interpreter() -> None:
    """`--python-bin` defaults to `python3`, which rarely carries torch."""
    for path in (INSTALL,):
        assert '--python-bin "$COMFY_PYTHON"' in _file_text(path), path


def test_documented_dev_loop_matches_ci() -> None:
    """One ruff invocation and one pytest invocation across every surface that
    tells a contributor what to run."""
    ci = _file_text(CI_WORKFLOW)
    assert f"run: {_CI_RUFF}" in ci
    assert f"run: {_CI_PYTEST}" in ci
    for path in (CONTRIBUTING, DEV_SKILL, PR_TEMPLATE):
        text = _file_text(path)
        assert _CI_RUFF in text, path
        assert _CI_PYTEST in text, path
    wrapper = _file_text(TEST_CPU)
    assert 'HYPOTHESIS_PROFILE=ci "$test_python" -m pytest tests/ -q --durations=25' in wrapper
    contributing = _file_text(CONTRIBUTING)
    assert '-e "$REPO_DIR[dev]"' in contributing
    assert "[docs/INSTALL.md](docs/INSTALL.md)" in contributing
    assert "concurrent_endpoint" in contributing


def test_ltx_guides_preserve_the_supported_floor_and_adapter_pointer() -> None:
    """Keep the compatibility minimum, not an old source-inspection timestamp."""
    for path in (MODELS, VALIDATION, GEN_TEMPLATES):
        assert _LTX_COMFY_FLOOR in _file_text(path), path
    header = " ".join(_file_text(LTX_ADAPTER).split('"""')[1].split())
    assert "docs/MODELS.md for the required ComfyUI revision" in header


def test_initial_release_notes_link_scoped_results_and_preserve_rdma_hold() -> None:
    changelog = " ".join(_file_text(CHANGELOG).split())
    for target in ("docs/MODELS.md", "docs/BENCHMARKS.md", "docs/VALIDATION.md"):
        assert f"]({target})" in changelog, target
    assert "Native latent RDMA remains **NOT RUN / HOLD**" in changelog
    assert "off by default" in changelog
    assert "Latents return through actor messages" in changelog
    assert "distributed rendering uses NCCL" in changelog
    assert "A workflow template does not establish accuracy" in changelog
    assert "FSDP is a capacity option and can be slower" in changelog


def test_scail_preview_hardware_scope_keeps_precision_and_workflow_limits() -> None:
    validation = " ".join(_validation_text().split())
    section = validation.split('<a id="evidence-wan-scail-preview"></a>', 1)[1].split('<a id=', 1)[0]
    for term in (
        "seed 20261008", "checkpoint executed in FP16", "native ComfyUI",
        "synthetic", "256x256", "five frames", "one UniPC/simple step",
        "CFG 5", "shift 3", "1x16x2x32x32", "NRMS 0.0", "limit of 0.10",
        "Cross-rank latents were equal", "does not establish accuracy for BF16 execution",
        "ring, FSDP", "40-step, 81-frame", "historical comparison remains CHECK",
        "separate four-step control", "not an extension of the one-step NRMS",
    ):
        assert term in section, term
    assert "wan_scail" not in _code_families()
