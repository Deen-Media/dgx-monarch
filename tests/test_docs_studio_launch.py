"""The Studio launch docs quote the code they describe.

TROUBLESHOOTING's three Studio launch entries quote refusal and status lines
that live as constants under ``src/dgx_monarch/studio/``, every Studio link
into TROUBLESHOOTING names its heading's anchor, the driver unit STUDIO.md
shows passes the driver adapter's own command and endpoint checks, the tunnel
lines STUDIO.md shows are the ones the code prints, and INSTALL states the
deploy order for the ``[studio]`` table. Constants are read from the source
with ``ast``, and the two Studio modules a test calls are imported inside
it, so this file imports no aiohttp.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STUDIO_SRC = REPO / "src" / "dgx_monarch" / "studio"
TROUBLESHOOTING = REPO / "docs" / "TROUBLESHOOTING.md"
STUDIO = REPO / "docs" / "STUDIO.md"
INSTALL = REPO / "docs" / "INSTALL.md"
LINKING_DOCS = (STUDIO, REPO / "CHANGELOG.md", REPO / "docs" / "THREAT_MODEL.md")

# Entry titles without their numbers, which are taken at rebase time.
DRIVER_NOT_RUNNING = "Studio shows Driver not running"
ALWAYS_ON = "dgxm studio up refuses, or the always-on Studio does not start"
SIGN_IN = "The sign-in link does not open Studio"
ENTRIES = (DRIVER_NOT_RUNNING, ALWAYS_ON, SIGN_IN)

# The constants each entry quotes, by module and name.
QUOTED = {
    DRIVER_NOT_RUNNING: {"comfy.py": ("DRIVER_NOT_RUNNING",)},
    ALWAYS_ON: {
        "unit.py": ("RUN_AS_ROOT", "NO_USER_MANAGER", "FOREIGN_UNIT", "UNIT_SYMLINK", "FOREIGN_DROPIN",
                    "SHADOWED", "DRIVER_UNIT_UNKNOWN", "DRIVER_UNIT_ORIGIN", "DRIVER_UNIT_NOTE", "LOCK_HELD",
                    "START_FAILED", "NOT_READY", "STILL_PRESENT", "NOT_STOPPED", "NO_ROTATE", "MANAGEMENT_REMOVED",
                    "LINGER_NOTE", "DRIFT_NOTE", "STALE_CODE", "UNINSTALL_STUDIO_FIRST"),
        "unit_state.py": ("UNIT_BUSY", "BUSY", "BUSY_UNKNOWN"),
        "unit_paths.py": ("PYTHONPATH_SET", "UNIT_PATH_OWNER", "UNIT_DIR_UNSAFE", "UNIT_PATH_CHARS"),
        "launch.py": ("UNIT_RUNS", "NO_DRIVER_ORIGIN", "CONFIG_NOT_PRIVATE", "NO_HOSTS_NOTE", "WORKDIR_CONFIG"),
    },
    SIGN_IN: {
        "sign_in.py": ("NOT_RUNNING", "NO_SIGN_IN_PAGE", "NO_TOKEN_FILE", "NO_PIPE", "NOT_A_BROWSER",
                       "NO_OPENER", "NO_DISPLAY", "MANAGEMENT_NO_LINK"),
        "launch.py": ("TERMINAL_STUDIO",),
        "unit.py": ("UNIT_NOT_READY",),
        "tunnel.py": ("BAD_HOST", "NAME_THE_HOST", "NO_SSH", "SAME_ADDRESS", "FORWARD_FAILED"),
    },
}


def _flat(text: str) -> str:
    return " ".join(text.split())


def _headings() -> dict[int, str]:
    return {int(number): title for number, title in
            re.findall(r"^## (\d+)\. (.+)$", TROUBLESHOOTING.read_text(), re.MULTILINE)}


def _slug(number: int, title: str) -> str:
    """GitHub's anchor for a ``## N. title`` heading."""
    text = re.sub(r"[^\w\- ]", "", f"{number}. {title}".lower())
    return text.replace(" ", "-")


def _number(title: str) -> int:
    numbers = [number for number, heading in _headings().items() if heading == title]
    assert len(numbers) == 1, f"expected one TROUBLESHOOTING entry titled {title!r}"
    return numbers[0]


def _entry(title: str) -> str:
    text = TROUBLESHOOTING.read_text()
    heading = f"\n## {_number(title)}. {title}\n"
    body = text.split(heading, 1)[1]
    stop = re.search(r"^## ", body, re.M)
    return body[: stop.start()] if stop else body


def _quoted(body: str) -> list[str]:
    return re.findall(r"^- `([^`]+)`", body, re.M)


def _studio_strings() -> set[str]:
    found: set[str] = set()
    for path in sorted(STUDIO_SRC.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found.add(node.value)
    return found


def _constants(module: str) -> dict[str, str]:
    """Module-level ``NAME = "text"`` assignments of one Studio module."""
    path = STUDIO_SRC / module
    assert path.is_file(), f"src/dgx_monarch/studio/{module} does not exist yet"
    found: dict[str, str] = {}
    for node in ast.parse(path.read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        value = getattr(node, "value", None)
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            found.update({target.id: value.value for target in targets if isinstance(target, ast.Name)})
    return found


def _block(text: str, first_line: str, fence: str) -> str:
    """The fenced block of kind ``fence`` whose first line is ``first_line``."""
    blocks = [block for block in re.findall(rf"^```{fence}\n(.*?)^```", text, re.M | re.S)
              if block.startswith(first_line + "\n")]
    assert len(blocks) == 1, f"expected one ```{fence} block starting {first_line!r}"
    return blocks[0]


# TROUBLESHOOTING


def test_the_three_entries_are_numbered_one_after_another():
    numbers = [_number(title) for title in ENTRIES]
    assert numbers == list(range(numbers[0], numbers[0] + 3))
    refusals = [title for title in _headings().values() if title.startswith("Studio refuses to start")]
    assert len(refusals) == 1, "the companion's startup refusals keep one entry"


@pytest.mark.parametrize("title", ENTRIES)
def test_every_quoted_line_in_the_entry_is_a_studio_constant(title):
    quoted = _quoted(_entry(title))
    assert quoted, title
    stale = sorted(set(quoted) - _studio_strings())
    assert not stale, f"TROUBLESHOOTING quotes lines no Studio module holds: {stale}"


@pytest.mark.parametrize("title", ENTRIES)
def test_the_entry_quotes_each_constant_it_names(title):
    quoted = set(_quoted(_entry(title)))
    missing = []
    for module, names in QUOTED[title].items():
        constants = _constants(module)
        missing += [f"{module}:{name}" for name in names
                    if name not in constants or constants[name] not in quoted]
    assert not missing, f"quote these constants byte for byte as `- ` bullets: {missing}"


def test_studio_links_into_troubleshooting_name_their_headings():
    headings = _headings()
    linked = set()
    for path in LINKING_DOCS:
        for number, rest in re.findall(r"TROUBLESHOOTING\.md#(\d+)-([\w\-]+)\)", path.read_text()):
            title = headings.get(int(number))
            if title is None or "studio" not in title.lower():
                continue
            assert f"{number}-{rest}" == _slug(int(number), title), (
                f"{path.name} links #{number}-{rest}, not the heading's anchor")
            if path == STUDIO:
                linked.add(title)
    assert set(ENTRIES) <= linked, "STUDIO.md links each Studio launch entry"


# STUDIO.md


def test_studio_md_keeps_the_headings_the_unit_and_the_page_link_to():
    lines = STUDIO.read_text().splitlines()
    for heading in ("## Install and start", "## Configure Studio in cluster.toml", "## Keep Studio running",
                    "## Reach Studio from another computer", "## Start Studio with the driver",
                    "## When the driver is not running", "## Development"):
        assert lines.count(heading) == 1, heading
    documentation = [line for line in lines if line.startswith("Documentation=")]
    assert documentation == [
        "Documentation=https://github.com/Deen-Media/dgx-monarch/blob/main/docs/STUDIO.md#keep-studio-running"]


def test_the_studio_unit_in_studio_md_is_the_one_up_writes():
    from dgx_monarch.studio import launch, unit, unit_paths

    home = Path("/home/user")
    settings = launch.StudioSettings(
        port=8220, driver="http://127.0.0.1:8191", driver_source="[studio] driver",
        state_dir=home / ".local/state/dgx-monarch/studio", config=home / ".config/dgx-monarch/cluster.toml",
        config_source="user config", driver_service="comfyui-driver.service", allow_management=False)
    runtime = unit_paths.UnitRuntime(interpreter=home / "monarch-env/bin/python",
                                     source=home / "projects/dgx-monarch/src",
                                     dependencies=home / "monarch-env/lib/python3.12/site-packages")
    digest = "0" * 64
    rendered = unit.render_unit(settings, runtime, source_digest=digest).replace(digest, "<64 hex characters>")
    assert _block(STUDIO.read_text(), unit.UNIT_MARKER, "ini") == rendered


def test_the_driver_unit_in_studio_md_passes_the_adapter_checks():
    from dgx_monarch.studio import driver_service

    text = STUDIO.read_text()
    unit = _block(text, "# ~/.config/systemd/user/comfyui-driver.service", "ini")
    name = unit.splitlines()[0].rsplit("/", 1)[1]
    driver_service.validate_unit_name(name)
    settings = dict(line.split("=", 1) for line in unit.splitlines() if "=" in line and not line.startswith("#"))
    assert settings["KillMode"] in {"control-group", "mixed"}
    for helper in ("ExecStartPre", "ExecStartPost", "ExecStop", "ExecStopPost", "ExecReload", "ExecCondition",
                   "EnvironmentFile", "Requires", "Wants", "BindsTo", "PartOf", "Requisite", "Conflicts"):
        assert helper not in settings, helper
    argv = settings["ExecStart"].split()
    serialized = (f"{{ path={argv[0]} ; argv[]={' '.join(argv)} ; ignore_errors=no ; start_time=[n/a] ; "
                  "stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }")
    _executable, main, checked = driver_service._direct_command(serialized)
    assert main.name == "main.py" and checked == argv
    table = text.split(unit, 1)[1]
    origin = re.search(r'^driver = "([^"]+)"$', table, re.M)
    service = re.search(r'^driver_service = "([^"]+)"$', table, re.M)
    assert origin and service and service.group(1) == name, "show the [studio] table right after the unit"
    assert driver_service.origin_mismatch(driver_service.listen_endpoint(argv), origin.group(1)) is None


def test_studio_md_shows_the_tunnel_lines_the_code_prints():
    from dgx_monarch.studio import sign_in

    text = STUDIO.read_text()
    output = _block(text, "Port: 8220 (from the running Studio)", "text").splitlines()
    forward = output[output.index("On the computer with the browser, run:") + 1]
    assert forward == "  " + sign_in.forward_line("user@192.0.2.10", 8220)
    assert sign_in.another_way_line(8220) in output
    tunnel = _constants("tunnel.py")
    assert output[-2:] == [tunnel["SAME_ADDRESS"], tunnel["FORWARD_FAILED"]]
    journey = _flat(text.split("## Reach Studio from another computer", 1)[1])
    assert f"`{sign_in.forward_line('user@studio-host', 8220)}`" in journey


def test_links_to_the_cluster_studio_section_name_its_heading():
    cluster = REPO / "docs" / "CLUSTER.md"
    anchors = {re.sub(r"[^\w\- ]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
               for line in cluster.read_text().splitlines() if re.match(r"^#{1,6} ", line)}
    for path in (*LINKING_DOCS, INSTALL):
        for anchor in re.findall(r"CLUSTER\.md#([\w\-]+)\)", path.read_text()):
            if "studio" in anchor:
                assert anchor in anchors, f"{path.name} links CLUSTER.md#{anchor}, which has no heading"


# INSTALL.md


def test_install_update_commands_state_the_deploy_order_for_the_studio_table():
    section = INSTALL.read_text().split("\n### Update commands\n", 1)[1].split("\n### ", 1)[0]
    assert ("Add the `[studio]` table to cluster.toml only after the driver has restarted on this release, "
            "and when no render runs.") in _flat(section)
