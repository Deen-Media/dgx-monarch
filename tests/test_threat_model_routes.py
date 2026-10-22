"""Driver routes and the Studio boundary stay documented where operators look.

Every `/dgxm` route the driver registers must appear in the threat model's
browser-to-driver list, the route that renews the observer lease must say so,
and the Studio companion's two boundaries, its startup and request refusals,
its specification status and its design-map rows must match the code they
describe.
"""

from __future__ import annotations

import ast
import re
import threading
from pathlib import Path

from dgx_monarch import execution_events
from dgx_monarch.execution_events import LEASE_SECONDS, MAX_BYTES, MAX_EVENTS, Journal, Scope
from dgx_monarch.studio.management_fields import FIELDS

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "dgx_monarch"
THREAT_MODEL = REPO / "docs" / "THREAT_MODEL.md"
SECURITY = REPO / "SECURITY.md"
TROUBLESHOOTING = REPO / "docs" / "TROUBLESHOOTING.md"
STUDIO = REPO / "docs" / "STUDIO.md"
STUDIO_SPEC = REPO / "docs" / "STUDIO_SPEC.md"
STUDIO_ACCEPTANCE = REPO / "docs" / "STUDIO_ACCEPTANCE.md"
DESIGN = REPO / "docs" / "DESIGN.md"

_METHODS = {"get", "post", "put", "patch", "delete", "head"}


def _flat(text: str) -> str:
    return " ".join(text.split())


def _section(path: Path, heading: str) -> str:
    """The body under one heading, up to the next heading of the same or a
    higher level."""
    text = path.read_text()
    level = heading.split(" ", 1)[0]
    assert text.count(f"\n{heading}\n") == 1, f"{path.name}: expected one {heading!r}"
    body = text.split(f"\n{heading}\n", 1)[1]
    stop = re.search(rf"^#{{1,{len(level)}}} ", body, re.M)
    return body[: stop.start()] if stop else body


def _slug(heading: str) -> str:
    """GitHub's anchor for a Markdown heading."""
    text = heading.lstrip("#").strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(path: Path) -> set[str]:
    return {_slug(line) for line in path.read_text().splitlines() if re.match(r"^#{1,6} ", line)}


def _driver_routes() -> set[tuple[str, str]]:
    """(METHOD, path) for every `@<app>.routes.<method>("/dgxm/...")` handler
    outside the Studio companion, read from source so no ComfyUI is needed."""
    routes: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        if "studio" in path.relative_to(SRC).parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not (isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)):
                    continue
                func = decorator.func
                if func.attr not in _METHODS or not (
                    isinstance(func.value, ast.Attribute) and func.value.attr == "routes"
                ):
                    continue
                first = decorator.args[0] if decorator.args else None
                if isinstance(first, ast.Constant) and str(first.value).startswith("/dgxm"):
                    routes.add((func.attr.upper(), first.value))
    return routes


# The driver route list


def test_every_driver_route_appears_in_the_browser_to_driver_boundary():
    routes = _driver_routes()
    # Non-vacuous: the scan must still see the routes it exists to cover.
    assert {("GET", "/dgxm/telemetry"), ("POST", "/dgxm/recycle"),
            ("GET", "/dgxm/execution-events")} <= routes
    section = _flat(_section(THREAT_MODEL, "### Browser to ComfyUI driver"))
    missing = sorted(f"{method} {path}" for method, path in routes if f"`{method} {path}`" not in section)
    assert not missing, (
        "driver routes missing from docs/THREAT_MODEL.md 'Browser to ComfyUI driver'; "
        f"name each as `METHOD /path`: {missing}"
    )


def test_the_threat_model_states_the_observer_lease_from_its_constants():
    section = _flat(_section(THREAT_MODEL, "### Browser to ComfyUI driver"))
    assert "`GET /dgxm/execution-events`" in section
    assert "it is not read-only" in section
    assert f"{LEASE_SECONDS:g} s observer lease" in section
    assert f"{MAX_EVENTS:,} events and {MAX_BYTES // (1024 * 1024)} MiB" in section


def test_the_lease_route_docstrings_do_not_call_it_read_only():
    route_source = (SRC / "nodes" / "execution_event_routes.py").read_text()
    route_doc = _flat(ast.get_docstring(ast.parse(route_source)) or "")
    assert not route_doc.lower().startswith("read-only")
    assert "This route is not read-only." in route_doc
    assert f"{LEASE_SECONDS:g} s observer lease" in route_doc
    assert "THREAT_MODEL.md" in route_doc
    routes_source = (SRC / "nodes" / "routes.py").read_text()
    routes_doc = _flat(ast.get_docstring(ast.parse(routes_source)) or "")
    assert "GET /dgxm/execution-events is the one GET that changes driver state" in routes_doc
    assert "The other GET routes are read-only." in routes_doc
    assert "The GET routes are read-only." not in routes_doc


def test_journal_read_copies_its_page_outside_the_lock(monkeypatch):
    journal = Journal()
    journal.read()  # arm the observer lease
    journal.publish(Scope("prompt", "node", None, "render"), "worker_results", {"value": [1, 2]})
    held: list[bool] = []
    real_deepcopy = execution_events.deepcopy

    def observing_deepcopy(value):
        held.append(journal._lock.locked())
        return real_deepcopy(value)

    monkeypatch.setattr(execution_events, "deepcopy", observing_deepcopy)
    page = journal.read()
    assert held == [False], "Journal.read must not hold its lock while it copies a page"
    assert [event["facts"]["value"] for event in page["events"]] == [[1, 2]]
    page["events"][0]["facts"]["value"].append(3)
    assert journal.read()["events"][0]["facts"]["value"] == [1, 2], "the page must be a copy"


def test_publish_is_not_blocked_by_a_reader_copying_a_page(monkeypatch):
    journal = Journal()
    journal.read()
    scope = Scope("prompt", "node", None, "render")
    journal.publish(scope, "worker_results", {"value": 1})
    copying = threading.Event()
    release = threading.Event()
    real_deepcopy = execution_events.deepcopy

    def slow_deepcopy(value):
        copying.set()
        assert release.wait(5)
        return real_deepcopy(value)

    monkeypatch.setattr(execution_events, "deepcopy", slow_deepcopy)
    reader = threading.Thread(target=journal.read)
    reader.start()
    try:
        assert copying.wait(5)
        published = threading.Thread(target=journal.publish, args=(scope, "result_validated", {"value": 2}))
        published.start()
        published.join(2)
        assert not published.is_alive(), "a page copy must not block a render's publish"
    finally:
        release.set()
        reader.join(5)


# The Studio companion's two boundaries


def test_the_threat_model_records_both_studio_companion_boundaries():
    browser = _flat(_section(THREAT_MODEL, "### Browser to Studio companion"))
    driver = _flat(_section(THREAT_MODEL, "### Companion to driver, systemd and CLI"))
    for section in (browser, driver):
        assert "(STUDIO.md#" in section, "each boundary links to STUDIO.md rather than restating it"
    assert "Without `--allow-management`" in browser
    assert "With `--allow-management`" in browser
    assert "code execution as the worker account on every host" in browser
    for place in ("terminal", "`0600`", "`--token-file`", "journal", "`sessionStorage`"):
        assert place in browser, place
    assert "second authorized caller" in driver
    assert "`POST /dgxm/recycle`" in driver and "`POST /dgxm/consent`" in driver
    assert "carries the owner token's authority" in driver


def test_the_threat_model_records_the_sign_in_link_and_the_service():
    browser = _flat(_section(THREAT_MODEL, "### Browser to Studio companion"))
    for fact in ("`#token=`", "`history.replaceState`", "`sign-in.html`", "process arguments",
                 "history database", "never signs in by link", "Could not request local forwarding",
                 "`hidepid`", "clipboard", "`GatewayPorts yes`", "Address already in use"):
        assert fact in browser, fact
    driver = _flat(_section(THREAT_MODEL, "### Companion to driver, systemd and CLI"))
    for fact in ("`dgxm-studio.service`", "`Wants=`", "control group", "token --rotate", "`xdg-open`",
                 "working directory"):
        assert fact in driver, fact


def test_the_token_scope_claims_match_the_management_inputs():
    # The threat model says setup and init run a browser-supplied interpreter
    # with a browser-supplied key on browser-supplied hosts, and that only the
    # flag unlocks them. Read both facts from the code that makes them true.
    for action in ("setup", "init"):
        names = {entry["name"] for entry in FIELDS[action]}
        assert {"hosts", "python_bin", "ssh_key"} <= names, action
    operations = ast.parse((SRC / "studio" / "operations.py").read_text())
    read_only: set[str] = set()
    for node in ast.walk(operations):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "READ_ONLY" for target in node.targets
        ):
            read_only = {element.value for element in ast.walk(node.value)
                         if isinstance(element, ast.Constant) and isinstance(element.value, str)}
    assert read_only, "READ_ONLY not found in studio/operations.py"
    assert not {"setup", "init", "update", "launcher", "actors.recycle", "consent.apply"} & read_only


def test_the_waiver_ledger_note_names_every_source_the_consent_route_takes():
    tree = ast.parse((SRC / "nodes" / "consent_routes.py").read_text())
    sources = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "BODY_SOURCES" for target in node.targets))
    driver = _flat(_section(THREAT_MODEL, "### Companion to driver, systemd and CLI"))
    assert "reads as a panel click" not in driver
    assert "A body without the field records `panel`" in driver
    for source in sources:
        assert f"`{source}`" in driver, source


def test_attacker_stories_and_severity_cover_the_companion():
    stories = _flat(_section(THREAT_MODEL, "## Attack Surface, Mitigations, and Attacker Stories"))
    for story in ("**Studio owner token.**", "**Hostile library file.**", "**Hostile driver.**"):
        assert story in stories, story
    severity = _section(THREAT_MODEL, "## Severity Calibration (Critical, High, Medium, Low)")
    high = _flat(next(item for item in severity.split("\n- ") if item.startswith("**High:**")))
    assert "disclosure of a Studio owner token" in high


def test_threat_model_links_to_studio_resolve():
    anchors = _anchors(STUDIO)
    targets = re.findall(r"\]\(STUDIO\.md#([^)]+)\)", THREAT_MODEL.read_text())
    assert targets
    assert not sorted(set(targets) - anchors), "THREAT_MODEL.md links to a missing STUDIO.md heading"


def test_security_md_points_at_the_new_boundaries():
    text = SECURITY.read_text()
    anchors = _anchors(THREAT_MODEL)
    for anchor in ("browser-to-studio-companion", "companion-to-driver-systemd-and-cli"):
        assert anchor in anchors, anchor
        assert f"(docs/THREAT_MODEL.md#{anchor})" in text, anchor


# The companion's refusals have a troubleshooting entry


_COMPANION_REFUSALS = (
    "Unrecognized companion host.",
    "Cross-origin access is forbidden.",
    "Enter this companion's owner token.",
    "A companion already owns this state directory.",
    "State directories cannot be symlinks.",
    "State directory must belong to this user with mode 0700.",
    "State database must be an owner-private regular file.",
    "Unsupported storage schema; export with the matching companion.",
)


def _studio_strings() -> set[str]:
    found: set[str] = set()
    for path in sorted((SRC / "studio").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.add(node.value)
    return found


def _companion_entry() -> str:
    text = TROUBLESHOOTING.read_text()
    headings = [line for line in text.splitlines()
                if re.match(r"^## \d+\. Studio refuses to start", line)]
    assert len(headings) == 1, "expected one TROUBLESHOOTING entry for the companion's refusals"
    return _section(TROUBLESHOOTING, headings[0])


def test_troubleshooting_quotes_every_companion_boundary_refusal():
    entry = _companion_entry()
    quoted = set(re.findall(r"^- `([^`]+)`", entry, re.M))
    assert set(_COMPANION_REFUSALS) <= quoted
    # Doc to code: a quoted refusal the companion no longer raises is stale.
    stale = sorted(quoted - _studio_strings())
    assert not stale, f"TROUBLESHOOTING quotes refusals the companion no longer raises: {stale}"


def test_troubleshooting_explains_the_exact_host_rule_and_the_lock():
    entry = _flat(_companion_entry())
    assert "`localhost:8220` reaches the same socket but is refused" in entry
    assert "new token at every launch" in entry
    assert "Do not delete `companion.lock`" in entry
    assert "(mode `0700`)" in entry
    assert "(STUDIO.md#install-and-start)" in entry


# The specification's status and service wording


def test_studio_spec_is_marked_as_the_original_specification():
    text = STUDIO_SPEC.read_text()
    status = text.split("\n\n", 2)[1]
    assert status.startswith("Status: original specification")
    assert "(STUDIO.md)" in status and "(STUDIO_ACCEPTANCE.md)" in status
    assert "**draft**" in status
    for stale in ("proposed work, not existing features", "proposed engineering work",
                  "The known backend currently mounts", "unless the user explicitly enables its service"):
        assert stale not in text, stale
    # The original sentence stays, and the dated amendment follows it in the
    # same paragraph, covering the unit, the CLI command and the config table.
    paragraph = next(block for block in text.split("\n\n") if "Studio installs no service" in block)
    original = paragraph.index("Studio installs no service")
    amendment = paragraph.index("Amended 2026-10-22 with the owner's approval:")
    assert original < amendment
    amended = _flat(paragraph[amendment:])
    for fact in ("`dgxm studio up` installs one systemd user unit", "`dgxm-studio.service`",
                 "`dgxm studio down` removes it", "`dgxm studio` joins the CLI and `[studio]` joins cluster.toml",
                 "(STUDIO.md#keep-studio-running)"):
        assert fact in amended, fact


def test_the_three_studio_docs_share_one_status_word():
    spec_status = STUDIO_SPEC.read_text().split("\n\n", 2)[1]
    studio = STUDIO.read_text()
    acceptance_status = STUDIO_ACCEPTANCE.read_text().split("\n\n", 2)[1]
    assert "draft" in spec_status
    assert "draft" in studio.splitlines()[0].lower()
    assert "draft" in acceptance_status.lower()
    for status in (spec_status, acceptance_status):
        assert "proposed" not in status.lower()


# The design map names the new boundary owners


def _design_map_paths() -> list[str]:
    block = DESIGN.read_text().split("## 7. Repository layout & maintainability", 1)[1]
    block = block.split("```", 2)[1]
    stack: list[str] = []
    paths: list[str] = []
    for line in block.splitlines():
        match = re.search(r"[├└]── (\S+)", line)
        if not match:
            continue
        depth = match.start() // 4
        name = match.group(1)
        stack[depth:] = [name]
        paths.append("".join(stack))
    return paths


def test_the_design_map_lists_the_new_lifecycle_and_release_owners():
    paths = set(_design_map_paths())
    required = {
        "src/dgx_monarch/studio_resource_validation.py",
        "src/dgx_monarch/execution_events.py",
        "src/dgx_monarch/runtime_observer.py",
        "src/dgx_monarch/nodes/execution_event_routes.py",
        "src/dgx_monarch/cli/update_repository.py",
        "tools/gen_studio_inventory.py",
        "src/dgx_monarch/studio/",
        "src/dgx_monarch/cli/gate_last.py",
        "src/dgx_monarch/cli/sol_staging.py",
        "src/dgx_monarch/finite_json.py",
    }
    assert not sorted(required - paths), "DESIGN.md section 7 lacks rows"
    studio_row = DESIGN.read_text().split("│   ├── studio/", 1)[1].split("│   ├──", 1)[0]
    assert "docs/STUDIO.md" in studio_row


def test_every_design_map_row_names_an_existing_path():
    missing = [
        path for path in _design_map_paths()
        if not (REPO / path.removeprefix("dgx-monarch/")).exists()
    ]
    assert not missing, f"DESIGN.md section 7 names paths that do not exist: {missing}"
